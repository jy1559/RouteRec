from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.layout import REQUIRED_DIRECTORIES, REQUIRED_FILES
from routerec.model_registry import DATASET_LR_INTERVALS, PAPER_BASELINES, ROUTEREC_DEFAULT, recommended_routerec_config


class RepoLayoutTest(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        for rel_path in REQUIRED_DIRECTORIES:
            self.assertTrue((ROOT / rel_path).is_dir(), rel_path)

    def test_required_files_exist(self) -> None:
        for rel_path in REQUIRED_FILES:
            self.assertTrue((ROOT / rel_path).is_file(), rel_path)

    def test_routerec_default_is_routerec(self) -> None:
        self.assertEqual(ROUTEREC_DEFAULT["model"], "RouteRec")
        self.assertIn("SASRec", PAPER_BASELINES)
        self.assertIn("lastfm_recovered_core5_v1", DATASET_LR_INTERVALS)

    def test_public_cli_scripts_exist(self) -> None:
        self.assertTrue((ROOT / "scripts/train.py").is_file())
        self.assertTrue((ROOT / "scripts/evaluate.py").is_file())
        self.assertTrue((ROOT / "scripts/build_core5_splits.py").is_file())
        self.assertTrue((ROOT / "scripts/build_core5_features.py").is_file())
        self.assertTrue((ROOT / "scripts/validate_core5_splits.py").is_file())
        self.assertTrue((ROOT / "scripts/validate_feature_dataset.py").is_file())

    def test_only_stable_routerec_package_is_public(self) -> None:
        self.assertTrue((ROOT / "src/routerec/models/routerec/model.py").is_file())
        for old_name in ("RouteRec", "RouteRecBase", "RouteRecN", "RouteRecV2"):
            self.assertFalse((ROOT / "src/routerec/models" / old_name).exists())

    def test_dataset_preset_overrides_generic_default(self) -> None:
        generic = recommended_routerec_config()
        preset = recommended_routerec_config("ml-1m")
        self.assertEqual(generic["embedding_size"], 192)
        self.assertEqual(preset["embedding_size"], 128)
        self.assertEqual(
            preset["learning_rate_range"],
            list(DATASET_LR_INTERVALS["movielens1m_core5_v1"]),
        )


if __name__ == "__main__":
    unittest.main()
