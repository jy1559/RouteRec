from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.datasets import default_dataset_roots, infer_recbole_dataset_config, normalize_dataset_name, resolve_dataset_runtime


class DatasetResolutionTest(unittest.TestCase):
    def test_normalize_dataset_aliases(self) -> None:
        self.assertEqual(normalize_dataset_name("ml-1m"), "movielens1m")
        self.assertEqual(normalize_dataset_name("kuairec"), "KuaiRecLargeStrictPosV2_0.2")
        self.assertEqual(normalize_dataset_name("retailrocket"), "retail_rocket")

    def test_auto_resolution_prefers_feature_added_v4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            dataset_dir = repo_root / "Datasets/processed/feature_added_v4/movielens1m"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "movielens1m.train.inter").write_text("header\n", encoding="utf-8")

            resolved = resolve_dataset_runtime(
                dataset="ml-1m",
                data_path=None,
                repo_root=repo_root,
                require_existing=True,
            )

            self.assertEqual(resolved.dataset_name, "movielens1m")
            self.assertEqual(resolved.dataset_dir, dataset_dir)
            self.assertEqual(resolved.data_path, str(dataset_dir.parent))
            self.assertTrue(resolved.auto_discovered)

    def test_default_roots_prefer_release_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            roots = default_dataset_roots(repo_root)
            self.assertEqual(roots[0], (repo_root / "Datasets/release").resolve())

            release_dir = repo_root / "Datasets/release/movielens1m"
            fallback_dir = repo_root / "Datasets/processed/feature_added_v4/movielens1m"
            release_dir.mkdir(parents=True)
            fallback_dir.mkdir(parents=True)
            (release_dir / "movielens1m.train.inter").write_text("header\n", encoding="utf-8")
            (fallback_dir / "movielens1m.train.inter").write_text("header\n", encoding="utf-8")

            resolved = resolve_dataset_runtime(
                dataset="ml-1m",
                data_path=None,
                repo_root=repo_root,
                require_existing=True,
            )

            self.assertEqual(resolved.dataset_dir, release_dir)
            self.assertEqual(resolved.data_path, str(release_dir.parent))

    def test_direct_dataset_path_normalizes_to_parent_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "movielens1m"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "movielens1m.item").write_text("item_id\n", encoding="utf-8")

            resolved = resolve_dataset_runtime(
                dataset="movielens1m",
                data_path=str(dataset_dir),
                require_existing=True,
            )

            self.assertEqual(resolved.dataset_dir, dataset_dir)
            self.assertEqual(resolved.data_path, str(dataset_dir.parent))
            self.assertFalse(resolved.auto_discovered)

    def test_missing_dataset_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_dataset_runtime(
                    dataset="lastfm",
                    data_path=None,
                    repo_root=repo_root,
                    require_existing=True,
                )

            self.assertIn("normalized to 'lastfm0.03'", str(ctx.exception))

    def test_infer_recbole_dataset_config_from_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "movielens1m"
            dataset_dir.mkdir(parents=True)
            header = "session_id:token\titem_id:token\ttimestamp:float\tuser_id:token\tfeat_a:float\n"
            row = "s1\ti1\t1.0\tu1\t0.1\n"
            for split_name in ("train", "valid", "test"):
                (dataset_dir / f"movielens1m.{split_name}.inter").write_text(header + row, encoding="utf-8")
            (dataset_dir / "movielens1m.session_split_summary.json").write_text(
                '{"ratios": {"train": 0.7, "valid": 0.15, "test": 0.15}}',
                encoding="utf-8",
            )

            inferred = infer_recbole_dataset_config(
                dataset="ml-1m",
                data_path=str(dataset_dir.parent),
            )

            self.assertEqual(inferred["USER_ID_FIELD"], "user_id")
            self.assertEqual(inferred["ITEM_ID_FIELD"], "item_id")
            self.assertEqual(inferred["TIME_FIELD"], "timestamp")
            self.assertNotIn("benchmark_filename", inferred)
            self.assertEqual(inferred["eval_args"]["split"]["RS"], [0.7, 0.15, 0.15])
            self.assertEqual(inferred["eval_args"]["order"], "TO")
            self.assertEqual(
                inferred["load_col"]["inter"][:5],
                ["session_id", "item_id", "timestamp", "user_id", "feat_a"],
            )


if __name__ == "__main__":
    unittest.main()