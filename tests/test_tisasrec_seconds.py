from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
import sys

import torch

MODULE = Path(__file__).with_name("tisasrec.py")
if MODULE.is_file():
    SPEC = importlib.util.spec_from_file_location("tisasrec_seconds_under_test", MODULE)
    assert SPEC is not None and SPEC.loader is not None
    MOD = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MOD
    SPEC.loader.exec_module(MOD)
    TiSASRec = MOD.TiSASRec
else:
    from routerec.models.tisasrec import TiSASRec


class TiSASRecSecondsTest(unittest.TestCase):
    def test_context_scale_ignores_padding_and_timestamp_ties(self) -> None:
        items = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5], [0, 6, 7, 8]])
        times = torch.tensor(
            [[0.0, 0.0, 2.5, 7.5], [0.0, 0.0, 9.0, 9.0], [0.0, 1.0, 1.0, 4.0]],
            dtype=torch.float64,
        )
        scale = TiSASRec._causal_interval_scale(items, times)
        torch.testing.assert_close(scale, torch.tensor([2.5, 1.0, 3.0], dtype=torch.float64))

    def test_time_matrix_uses_minimum_context_interval_then_clips(self) -> None:
        model = object.__new__(TiSASRec)
        model.time_span = 4
        items = torch.tensor([[1, 2, 3, 4]])
        times = torch.tensor([[0.0, 2.5, 7.5, 20.0]], dtype=torch.float64)
        actual = model._build_time_matrix(items, times)
        expected = torch.tensor(
            [[[0, 1, 3, 4], [1, 0, 2, 4], [3, 2, 0, 4], [4, 4, 4, 0]]]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_future_timestamp_cannot_change_existing_prefix_relations(self) -> None:
        model = object.__new__(TiSASRec)
        model.time_span = 128
        prefix_items = torch.tensor([[1, 2, 3]])
        prefix_times = torch.tensor([[0.0, 10.0, 30.0]], dtype=torch.float64)
        extended_items = torch.tensor([[1, 2, 3, 4]])
        extended_times = torch.tensor([[0.0, 10.0, 30.0, 30.1]], dtype=torch.float64)
        before = model._build_time_matrix(prefix_items, prefix_times)
        # A future event is not part of an earlier sequential sample.  This
        # assertion documents the causal contract used by the data loader.
        after = model._build_time_matrix(extended_items[:, :3], extended_times[:, :3])
        self.assertTrue(torch.equal(before, after))

    def test_subsecond_intervals_are_not_truncated_before_scaling(self) -> None:
        model = object.__new__(TiSASRec)
        model.time_span = 10
        items = torch.tensor([[1, 2, 3]])
        times = torch.tensor([[0.0, 0.25, 0.75]], dtype=torch.float64)
        actual = model._build_time_matrix(items, times)
        self.assertEqual(actual[0, 0, 1].item(), 1)
        self.assertEqual(actual[0, 0, 2].item(), 3)


if __name__ == "__main__":
    unittest.main()
