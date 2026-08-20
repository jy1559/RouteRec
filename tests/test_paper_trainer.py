from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # Lightweight local documentation environments.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if torch is not None:
    from routerec.runner import _bind_isolated_cuda_config, _build_paper_trainer
else:
    _bind_isolated_cuda_config = None
    _build_paper_trainer = None


class _Rows:
    def __init__(self, rows: list[int]):
        self.rows = list(rows)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        if index == []:
            return _Rows([])
        return _Rows([self.rows[int(i)] for i in index])


class _BaseTrainer:
    def _train_epoch(self, _train_data, _epoch_idx, loss_func=None, show_progress=False):
        return 1.0

    def _full_sort_batch_eval(self, _batched_data):
        scores = torch.tensor(
            [[0.0, 0.9, 0.8, 0.7], [0.0, 0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        return _Rows([10, 11]), scores, torch.tensor([0, 1]), torch.tensor([1, 3])

    def evaluate(self, _data, load_best_model=False, show_progress=False):
        return {"hit@10": 0.9, "ndcg@10": 0.6, "mrr@10": 0.3}


@unittest.skipIf(torch is None, "PyTorch is required for trainer tests")
class PaperTrainerTest(unittest.TestCase):
    def test_recbole_gpu_id_preserves_scheduler_physical_binding(self) -> None:
        config = {"gpu_id": 0, "use_gpu": False}
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            _bind_isolated_cuda_config(config)
        self.assertEqual(config["gpu_id"], "3")
        self.assertTrue(config["use_gpu"])

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,2"}, clear=False):
            with self.assertRaises(RuntimeError):
                _bind_isolated_cuda_config(config)

    def test_route_schedule_uses_final_budget_across_rungs(self) -> None:
        class _Config(dict):
            pass

        class _Model:
            def set_schedule_epoch(self, epoch, total):
                self.last_schedule = (epoch, total)

        trainer_cls = _build_paper_trainer(_BaseTrainer, ("Hit@10", "NDCG@10", "MRR@10"))
        trainer = object.__new__(trainer_cls)
        trainer.model = _Model()
        trainer.config = _Config(epochs=20, routerec_schedule_total_epochs=80)
        trainer._train_epoch(object(), 7)
        self.assertEqual(trainer.model.last_schedule, (7, 80))

    def test_composite_validation_objective_and_history(self) -> None:
        trainer_cls = _build_paper_trainer(_BaseTrainer, ("Hit@10", "NDCG@10", "MRR@10"))
        trainer = object.__new__(trainer_cls)
        trainer._routerec_epoch = 4
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer._routerec_history_path = str(Path(tmp_dir) / "history.jsonl")
            score, result = trainer._valid_epoch(object())
            self.assertAlmostEqual(score, 0.6)
            self.assertEqual(result["mrr@10"], 0.3)
            self.assertEqual(trainer._routerec_best_epoch, 4)
            self.assertTrue(Path(trainer._routerec_history_path).is_file())

    def test_train_seen_mask_drops_unseen_positive_rows(self) -> None:
        trainer_cls = _build_paper_trainer(_BaseTrainer, ("Hit@10", "NDCG@10", "MRR@10"))
        trainer = object.__new__(trainer_cls)
        trainer._routerec_train_item_mask = torch.tensor([False, True, True, False])
        trainer._routerec_current_filter_stats = {
            "total_targets": 0,
            "seen_targets": 0,
            "unseen_targets": 0,
            "dropped_rows": 0,
        }
        interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(None)
        self.assertEqual(interaction.rows, [10])
        self.assertEqual(tuple(scores.shape), (1, 4))
        self.assertEqual(positive_u.tolist(), [0])
        self.assertEqual(positive_i.tolist(), [1])
        self.assertTrue(torch.isneginf(scores[0, 0]))
        self.assertTrue(torch.isneginf(scores[0, 3]))
        self.assertEqual(trainer._routerec_current_filter_stats["unseen_targets"], 1)


if __name__ == "__main__":
    unittest.main()
