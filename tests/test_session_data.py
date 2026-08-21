from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.session_data import (
    _benchmark_data_root,
    _cache_key,
    _config_get,
    _history_fields,
    _history_profile,
    _load_tisas_elapsed_seconds,
)


class RecboleConfigDefaultTest(unittest.TestCase):
    def test_history_projection_is_model_specific(self) -> None:
        class DatasetLike:
            uid_field = "session_id"
            iid_field = "item_id"
            time_field = "timestamp"

            def __init__(self, model: str) -> None:
                self.config = {"model": model}

        fields = ["session_id", "item_id", "timestamp", "mid_focus", "mic_tempo"]
        route = DatasetLike("RouteRec")
        self.assertEqual(
            _history_fields(route, fields), ["item_id", "mid_focus", "mic_tempo"]
        )
        tisas = DatasetLike("TiSASRec")
        self.assertEqual(_history_fields(tisas, fields), ["item_id", "timestamp"])
        sas = DatasetLike("SASRec")
        self.assertEqual(_history_fields(sas, fields), ["item_id"])
        fdsa = DatasetLike("FDSA")
        self.assertEqual(_history_fields(fdsa, fields), ["item_id"])
        # Unknown/future models retain all fields rather than
        # silently losing an interaction sequence they might consume.
        unknown = DatasetLike("FutureContextModel")
        self.assertEqual(
            _history_fields(unknown, fields),
            ["item_id", "timestamp", "mid_focus", "mic_tempo"],
        )
        self.assertEqual(_history_profile(route.config), "routerec_route_features")
        self.assertEqual(
            _history_profile(tisas.config),
            "tisasrec_item_timestamp_elapsed_seconds_v1",
        )
        self.assertEqual(_history_profile(sas.config), "baseline_item_only")
        self.assertEqual(
            _history_profile(unknown.config),
            "all_fields",
        )

    def test_cache_key_separates_history_profiles(self) -> None:
        class DatasetLike:
            dataset_name = "toy"
            uid_field = "session_id"
            iid_field = "item_id"
            time_field = "timestamp"

            def __init__(self, root: Path, model: str) -> None:
                self.config = {
                    "data_path": str(root),
                    "dataset": "toy",
                    "model": model,
                    "MAX_ITEM_LIST_LENGTH": 20,
                    "load_col": {"inter": ["session_id", "item_id", "timestamp"]},
                    "routerec_feature_fp16": True,
                }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset_dir = root / "toy"
            dataset_dir.mkdir()
            for split in ("train", "valid", "test"):
                (dataset_dir / f"toy.{split}.inter").write_text(
                    "session_id:token\titem_id:token\ttimestamp:float\n"
                    f"s-{split}\ti-{split}\t1\n",
                    encoding="utf-8",
                )

            split_names = ["train", "valid", "test"]
            sas_key = _cache_key(DatasetLike(root, "SASRec"), split_names)
            gru_key = _cache_key(DatasetLike(root, "GRU4Rec"), split_names)
            tisas_key = _cache_key(DatasetLike(root, "TiSASRec"), split_names)
            route_key = _cache_key(DatasetLike(root, "RouteRec"), split_names)
            unknown_key = _cache_key(
                DatasetLike(root, "FutureContextModel"), split_names
            )

            # Models with the exact same materialized history may share a cache.
            self.assertEqual(sas_key, gru_key)
            # Incompatible Interaction schemas must never share a pickle.
            self.assertEqual(len({sas_key, tisas_key, route_key, unknown_key}), 4)

            with (dataset_dir / "toy.train.inter").open("a", encoding="utf-8") as stream:
                stream.write("s-new\ti-new\t2\n")
            self.assertNotEqual(
                sas_key,
                _cache_key(DatasetLike(root, "SASRec"), split_names),
            )

    def test_tisas_exact_loader_converts_ms_to_session_elapsed_seconds(self) -> None:
        class DatasetLike:
            dataset_name = "toy"

            def __init__(self, root: Path) -> None:
                self.config = {"data_path": str(root), "dataset": "toy", "model": "TiSASRec"}

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset_dir = root / "toy"
            dataset_dir.mkdir()
            rows = (
                "session_id:token\titem_id:token\ttimestamp:float\tuser_id:token\tf:float\n"
                "s1\ti1\t1593051631915\tu1\t0.1\n"
                "s1\ti2\t1593051862703\tu1\t0.2\n"
                "s2\ti3\t1700000000000\tu2\t0.3\n"
            )
            for split in ("train", "valid", "test"):
                (dataset_dir / f"toy.{split}.inter").write_text(rows, encoding="utf-8")
            (dataset_dir / "feature_metadata.json").write_text(
                '{"timestamp_unit":"ms"}', encoding="utf-8"
            )
            values = _load_tisas_elapsed_seconds(
                DatasetLike(root), ["train", "valid", "test"], "train", 3
            )
            self.assertEqual(values.dtype, __import__("torch").float64)
            self.assertEqual(values.tolist(), [0.0, 230.788, 0.0])

    def test_tisas_exact_loader_preserves_decimal_seconds(self) -> None:
        class DatasetLike:
            dataset_name = "toy"

            def __init__(self, root: Path) -> None:
                self.config = {"data_path": str(root), "dataset": "toy", "model": "TiSASRec"}

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset_dir = root / "toy"
            dataset_dir.mkdir()
            content = (
                "session_id:token\titem_id:token\ttimestamp:float\tuser_id:token\n"
                "s1\ti1\t1593051631.915\tu1\n"
                "s1\ti2\t1593051862.703\tu1\n"
            )
            for split in ("train", "valid", "test"):
                (dataset_dir / f"toy.{split}.inter").write_text(content, encoding="utf-8")
            (dataset_dir / "feature_metadata.json").write_text(
                '{"timestamp_unit":"s"}', encoding="utf-8"
            )
            values = _load_tisas_elapsed_seconds(
                DatasetLike(root), ["train", "valid", "test"], "train", 2
            )
            self.assertEqual(values.tolist(), [0.0, 230.788])

    def test_none_from_recbole_style_config_uses_default(self) -> None:
        class ConfigLike:
            def __getitem__(self, key: str):
                return None

        self.assertEqual(_config_get(ConfigLike(), "missing", 16384), 16384)
        self.assertTrue(_config_get(ConfigLike(), "flag", True))

    def test_explicit_non_none_value_is_preserved(self) -> None:
        self.assertEqual(_config_get({"chunk": 4096}, "chunk", 16384), 4096)

    def test_benchmark_root_accepts_parent_or_dataset_directory(self) -> None:
        class DatasetLike:
            dataset_name = "beauty"

            def __init__(self, data_path: Path) -> None:
                self.config = {"data_path": str(data_path), "dataset": "beauty"}

        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            dataset_dir = parent / "beauty"
            dataset_dir.mkdir()
            for split in ("train", "valid", "test"):
                (dataset_dir / f"beauty.{split}.inter").write_text("header\n", encoding="utf-8")

            self.assertEqual(
                _benchmark_data_root(DatasetLike(parent), ["train", "valid", "test"]),
                dataset_dir.resolve(),
            )
            self.assertEqual(
                _benchmark_data_root(DatasetLike(dataset_dir), ["train", "valid", "test"]),
                dataset_dir.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
