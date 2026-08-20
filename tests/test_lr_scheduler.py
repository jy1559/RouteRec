from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if torch is not None:
    from routerec.lr_scheduler import (
        build_lr_scheduler,
        resolve_schedule_total_epochs,
        scheduler_spec,
        warmup_cosine_multiplier,
    )


@unittest.skipIf(torch is None, "PyTorch is required")
class LearningRateSchedulerTest(unittest.TestCase):
    def test_continuation_horizon_overrides_current_rung_budget(self) -> None:
        config = {"epochs": 20, "routerec_schedule_total_epochs": 100}
        self.assertEqual(resolve_schedule_total_epochs(config), 100)
        self.assertEqual(resolve_schedule_total_epochs({"epochs": 20}), 20)

        class RecboleLikeConfig:
            def __init__(self) -> None:
                self.values = {"epochs": 20, "routerec_schedule_total_epochs": 100}

            def __contains__(self, key: str) -> bool:
                return key in self.values

            def __getitem__(self, key: str) -> object:
                return self.values[key]

        self.assertEqual(resolve_schedule_total_epochs(RecboleLikeConfig()), 100)

    def test_legacy_100_epoch_multipliers(self) -> None:
        expected = {
            1: 1.0,
            20: 0.920650,
            40: 0.697180,
            62: 0.389397,
            80: 0.187628,
            100: 0.1,
        }
        for epoch, value in expected.items():
            actual = warmup_cosine_multiplier(
                epoch - 1,
                total_epochs=100,
                warmup_ratio=0.0,
                min_lr_ratio=0.1,
            )
            self.assertAlmostEqual(actual, value, places=5)

    def test_constant_scheduler_is_state_free(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler, spec = build_lr_scheduler(
            optimizer,
            {"lr_scheduler_type": "constant", "epochs": 100},
        )
        self.assertIsNone(scheduler)
        self.assertEqual(spec, scheduler_spec({"lr_scheduler_type": "constant", "epochs": 100}))
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_warmup_cosine_state_resumes_exact_lr_sequence(self) -> None:
        config = {
            "lr_scheduler_type": "warmup_cosine",
            "lr_scheduler_warmup_ratio": 0.05,
            "lr_scheduler_min_lr_ratio": 0.02,
            "routerec_schedule_total_epochs": 100,
            "epochs": 20,
        }
        first_parameter = torch.nn.Parameter(torch.tensor(1.0))
        first_optimizer = torch.optim.Adam([first_parameter], lr=2e-3)
        first_scheduler, _ = build_lr_scheduler(first_optimizer, config)
        first_lrs = []
        for _ in range(40):
            first_lrs.append(first_optimizer.param_groups[0]["lr"])
            first_optimizer.step()
            first_scheduler.step()

        second_parameter = torch.nn.Parameter(torch.tensor(1.0))
        second_optimizer = torch.optim.Adam([second_parameter], lr=2e-3)
        second_scheduler, _ = build_lr_scheduler(second_optimizer, config)
        second_optimizer.load_state_dict(first_optimizer.state_dict())
        second_scheduler.load_state_dict(first_scheduler.state_dict())
        resumed_lrs = []
        for _ in range(40, 100):
            resumed_lrs.append(second_optimizer.param_groups[0]["lr"])
            second_optimizer.step()
            second_scheduler.step()

        reference_parameter = torch.nn.Parameter(torch.tensor(1.0))
        reference_optimizer = torch.optim.Adam([reference_parameter], lr=2e-3)
        reference_scheduler, _ = build_lr_scheduler(reference_optimizer, config)
        reference_lrs = []
        for _ in range(100):
            reference_lrs.append(reference_optimizer.param_groups[0]["lr"])
            reference_optimizer.step()
            reference_scheduler.step()
        self.assertEqual(first_lrs, reference_lrs[:40])
        self.assertEqual(resumed_lrs, reference_lrs[40:])


if __name__ == "__main__":
    unittest.main()
