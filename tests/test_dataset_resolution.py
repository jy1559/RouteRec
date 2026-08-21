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
from routerec.model_registry import recommended_routerec_config


class DatasetResolutionTest(unittest.TestCase):
    def test_normalize_dataset_aliases(self) -> None:
        self.assertEqual(normalize_dataset_name("ml-1m"), "movielens1m_core5_v1")
        self.assertEqual(normalize_dataset_name("kuairec"), "kuairec_adaptive_core5_v1")
        self.assertEqual(normalize_dataset_name("retailrocket"), "retail_rocket_core5_v1")
        self.assertEqual(normalize_dataset_name("amazon_beauty"), "beauty_core5_v1")
        self.assertEqual(normalize_dataset_name("beauty_core5_v1"), "beauty_core5_v1")
        self.assertEqual(
            normalize_dataset_name("kuairec_adaptive_core5_v1"),
            "kuairec_adaptive_core5_v1",
        )
        # Bare aliases must never drift to a different prepared identity.
        self.assertEqual(normalize_dataset_name("kuairec"), "kuairec_adaptive_core5_v1")
        self.assertEqual(normalize_dataset_name("lastfm"), "lastfm_recovered_core5_v1")

    def test_core5_identities_receive_explicit_initialization_presets(self) -> None:
        kuai = recommended_routerec_config("kuairec_adaptive_core5_v1")
        lastfm = recommended_routerec_config("lastfm_recovered_core5_v1")
        self.assertEqual(kuai["MAX_ITEM_LIST_LENGTH"], 20)
        self.assertEqual(lastfm["MAX_ITEM_LIST_LENGTH"], 30)
        self.assertEqual(kuai["learning_rate_range"], [3.0e-4, 5.0e-3])
        self.assertEqual(lastfm["learning_rate_range"], [8.0e-5, 1.2e-3])

    def test_public_core5_root_and_fixed_history_lengths(self) -> None:
        expected_lengths = {
            "beauty_core5_v1": 20,
            "foursquare_core5_v1": 30,
            "movielens1m_core5_v1": 50,
            "retail_rocket_core5_v1": 20,
            "kuairec_adaptive_core5_v1": 20,
            "lastfm_recovered_core5_v1": 30,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            for dataset, history in expected_lengths.items():
                dataset_dir = repo_root / "Datasets/core5" / dataset
                dataset_dir.mkdir(parents=True)
                (dataset_dir / f"{dataset}.train.inter").write_text("header\n", encoding="utf-8")
                resolved = resolve_dataset_runtime(
                    dataset=dataset, data_path=None, repo_root=repo_root, require_existing=True
                )
                self.assertEqual(resolved.dataset_name, dataset)
                self.assertEqual(resolved.dataset_dir, dataset_dir)
                self.assertEqual(
                    recommended_routerec_config(dataset)["MAX_ITEM_LIST_LENGTH"], history
                )

    def test_auto_resolution_prefers_core5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            dataset_dir = repo_root / "Datasets/core5/movielens1m_core5_v1"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "movielens1m_core5_v1.train.inter").write_text(
                "header\n", encoding="utf-8"
            )

            resolved = resolve_dataset_runtime(
                dataset="ml-1m",
                data_path=None,
                repo_root=repo_root,
                require_existing=True,
            )

            self.assertEqual(resolved.dataset_name, "movielens1m_core5_v1")
            self.assertEqual(resolved.dataset_dir, dataset_dir)
            self.assertEqual(resolved.data_path, str(dataset_dir.parent))
            self.assertTrue(resolved.auto_discovered)

    def test_default_root_is_experiment_facing_core5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            roots = default_dataset_roots(repo_root)
            self.assertEqual(roots, [(repo_root / "Datasets/core5").resolve()])

            core5_dir = repo_root / "Datasets/core5/movielens1m_core5_v1"
            core5_dir.mkdir(parents=True)
            (core5_dir / "movielens1m_core5_v1.train.inter").write_text(
                "header\n", encoding="utf-8"
            )

            resolved = resolve_dataset_runtime(
                dataset="ml-1m",
                data_path=None,
                repo_root=repo_root,
                require_existing=True,
            )

            self.assertEqual(resolved.dataset_dir, core5_dir)
            self.assertEqual(resolved.data_path, str(core5_dir.parent))

    def test_direct_dataset_path_normalizes_to_parent_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "movielens1m_core5_v1"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "movielens1m_core5_v1.item").write_text(
                "item_id\n", encoding="utf-8"
            )

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

            self.assertIn("normalized to 'lastfm_recovered_core5_v1'", str(ctx.exception))

    def test_infer_recbole_dataset_config_from_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "movielens1m_core5_v1"
            dataset_dir.mkdir(parents=True)
            header = "session_id:token\titem_id:token\ttimestamp:float\tuser_id:token\tfeat_a:float\n"
            row = "s1\ti1\t1.0\tu1\t0.1\n"
            for split_name in ("train", "valid", "test"):
                (dataset_dir / f"movielens1m_core5_v1.{split_name}.inter").write_text(
                    header + row, encoding="utf-8"
                )
            (dataset_dir / "movielens1m_core5_v1.session_split_summary.json").write_text(
                '{"ratios": {"train": 0.7, "valid": 0.15, "test": 0.15}}',
                encoding="utf-8",
            )

            inferred = infer_recbole_dataset_config(
                dataset="ml-1m",
                data_path=str(dataset_dir.parent),
            )

            self.assertEqual(inferred["USER_ID_FIELD"], "session_id")
            self.assertEqual(inferred["SESSION_ID_FIELD"], "session_id")
            self.assertEqual(inferred["ITEM_ID_FIELD"], "item_id")
            self.assertEqual(inferred["TIME_FIELD"], "timestamp")
            self.assertEqual(inferred["benchmark_filename"], ["train", "valid", "test"])
            self.assertTrue(inferred["routerec_frozen_split"])
            self.assertEqual(inferred["eval_args"]["split"]["RS"], [0.7, 0.15, 0.15])
            self.assertEqual(inferred["eval_args"]["order"], "TO")
            self.assertEqual(
                inferred["load_col"]["inter"][:5],
                ["session_id", "item_id", "timestamp", "user_id", "feat_a"],
            )


if __name__ == "__main__":
    unittest.main()
