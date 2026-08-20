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
        self.assertIn("lastfm0.03", DATASET_LR_INTERVALS)

    def test_new_cli_scripts_exist(self) -> None:
        self.assertTrue((ROOT / "scripts/train.py").is_file())
        self.assertTrue((ROOT / "scripts/test.py").is_file())
        self.assertTrue((ROOT / "scripts/rebuild_camera_ready_core5_basic.py").is_file())
        self.assertTrue((ROOT / "scripts/build_camera_ready_core5_features.py").is_file())
        self.assertTrue((ROOT / "scripts/validate_camera_ready_core5_basic.py").is_file())

    def test_dataset_preset_overrides_generic_default(self) -> None:
        generic = recommended_routerec_config()
        preset = recommended_routerec_config("ml-1m")
        self.assertEqual(generic["embedding_size"], 192)
        self.assertEqual(preset["embedding_size"], 128)
        self.assertEqual(preset["learning_rate_range"], list(DATASET_LR_INTERVALS["movielens1m"]))


if __name__ == "__main__":
    unittest.main()
