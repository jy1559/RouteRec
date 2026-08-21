from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE = Path(__file__).with_name("session_data.py")
if MODULE.is_file():
    SPEC = importlib.util.spec_from_file_location("session_data_runtime_under_test", MODULE)
    assert SPEC is not None and SPEC.loader is not None
    MOD = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MOD
    SPEC.loader.exec_module(MOD)
else:
    import routerec.session_data as MOD


class DatasetLike:
    dataset_name = "toy"

    def __init__(self, root: Path) -> None:
        self.config = {"data_path": str(root), "dataset": "toy", "model": "TiSASRec"}


def _fixture(root: Path, unit: str, timestamps: list[str]) -> DatasetLike:
    directory = root / "toy"
    directory.mkdir()
    body = "session_id:token\titem_id:token\ttimestamp:float\tuser_id:token\tf:float\n"
    body += "\n".join(
        f"s1\ti{index}\t{timestamp}\tu1\t0.{index}" for index, timestamp in enumerate(timestamps, 1)
    ) + "\n"
    for split in ("train", "valid", "test"):
        (directory / f"toy.{split}.inter").write_text(body, encoding="utf-8")
    (directory / "feature_metadata.json").write_text(
        '{"timestamp_unit":"' + unit + '"}', encoding="utf-8"
    )
    return DatasetLike(root)


class TiSASRuntimeSecondsTest(unittest.TestCase):
    def test_exact_loader_converts_milliseconds_before_float32(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _fixture(Path(raw), "ms", ["1593051631915", "1593051862703"])
            values = MOD._load_tisas_elapsed_seconds(
                dataset, ["train", "valid", "test"], "train", 2
            )
            self.assertEqual(str(values.dtype), "torch.float64")
            self.assertEqual(values.tolist(), [0.0, 230.788])

    def test_exact_loader_preserves_decimal_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _fixture(Path(raw), "s", ["1593051631.915", "1593051862.703"])
            values = MOD._load_tisas_elapsed_seconds(
                dataset, ["train", "valid", "test"], "train", 2
            )
            self.assertEqual(values.tolist(), [0.0, 230.788])

    def test_exact_loader_rejects_unknown_unit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = _fixture(Path(raw), "minutes", ["1", "2"])
            with self.assertRaisesRegex(ValueError, "timestamp_unit"):
                MOD._load_tisas_elapsed_seconds(
                    dataset, ["train", "valid", "test"], "train", 2
                )


if __name__ == "__main__":
    unittest.main()
