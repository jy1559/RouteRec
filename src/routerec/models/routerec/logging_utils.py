"""
Logging utilities for RouteRec routing analysis.

Tracks per-epoch, per-stage, per-expert gating weight statistics to enable
downstream analysis of:
  - Which experts are selected per stage (average weight distribution).
  - Expert utilisation balance / collapse detection.
  - Feature-driven routing patterns.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _base_stage_name(stage_name: str) -> str:
    """Normalize repeated-stage key (e.g., macro@2 -> macro)."""
    if not isinstance(stage_name, str):
        return str(stage_name)
    return stage_name.split("@", 1)[0]


class MoELogger:
    """Accumulates gating statistics across batches within an epoch.

    Usage in the model:
        # In calculate_loss (every batch):
        self.moe_logger.accumulate(gate_weights, item_seq_len)

        # At epoch end:
        summary = self.moe_logger.get_and_reset()
    """

    def __init__(self, expert_names: Dict[str, List[str]]):
        """
        Args:
            expert_names: {stage_name: [expert_name, ...]}
        """
        self.expert_names = expert_names
        self._reset()

    def _reset(self):
        """Reset accumulators."""
        # Per-stage: running sums and counts for mean computation
        self._weight_sums: Dict[str, torch.Tensor] = {}  # {stage: [K]}
        self._weight_sq_sums: Dict[str, torch.Tensor] = {}  # for std
        self._counts: Dict[str, int] = defaultdict(int)
        self._n_batches = 0

        # Per-batch max/min tracking
        self._max_weights: Dict[str, float] = defaultdict(lambda: 0.0)
        self._min_weights: Dict[str, float] = defaultdict(lambda: 1.0)

    @torch.no_grad()
    def accumulate(
        self,
        gate_weights: Dict[str, torch.Tensor],
        item_seq_len: Optional[torch.Tensor] = None,
    ):
        """Accumulate gating weights for one batch.

        Args:
            gate_weights: {stage_name: [B, T, K]} — gating weights.
            item_seq_len: [B] — if provided, only count non-padded positions.
        """
        self._n_batches += 1

        for stage_name, w in gate_weights.items():
            # w: [B, T, K]
            B, T, K = w.shape

            if item_seq_len is not None:
                # Create mask for valid (non-padding) positions
                # [B, T]
                arange = torch.arange(T, device=w.device).unsqueeze(0)
                mask = arange < item_seq_len.unsqueeze(1)  # [B, T] True = valid
                # Mask out padding: [B, T, 1]
                w_masked = w * mask.unsqueeze(-1).float()
                n_valid = mask.sum().item()
            else:
                w_masked = w
                n_valid = B * T

            # Sum of gating weights across all valid tokens: [K]
            batch_sum = w_masked.reshape(-1, K).sum(dim=0).cpu()
            batch_sq_sum = (w_masked ** 2).reshape(-1, K).sum(dim=0).cpu()

            if stage_name not in self._weight_sums:
                self._weight_sums[stage_name] = torch.zeros(K)
                self._weight_sq_sums[stage_name] = torch.zeros(K)

            self._weight_sums[stage_name] += batch_sum
            self._weight_sq_sums[stage_name] += batch_sq_sum
            self._counts[stage_name] += n_valid

            # Track per-token extremes (for the batch mean)
            batch_mean = batch_sum / max(n_valid, 1)
            self._max_weights[stage_name] = max(
                self._max_weights[stage_name],
                batch_mean.max().item(),
            )
            self._min_weights[stage_name] = min(
                self._min_weights[stage_name],
                batch_mean.min().item(),
            )

    def get_and_reset(self) -> Dict:
        """Return epoch summary and reset accumulators.

        Returns:
            dict with per-stage statistics:
            {
                "n_batches": int,
                "stages": {
                    stage_name: {
                        "expert_names": [str, ...],
                        "mean_weights": [float, ...],   # avg gate weight per expert
                        "std_weights":  [float, ...],   # std of gate weight per expert
                        "max_mean_weight": float,
                        "min_mean_weight": float,
                        "balance_score": float,  # 1.0 = perfect balance
                    }
                }
            }
        """
        summary = {"n_batches": self._n_batches, "stages": {}}

        for stage_name in self._weight_sums:
            n = max(self._counts[stage_name], 1)
            mean = (self._weight_sums[stage_name] / n).tolist()
            # Var = E[X^2] - E[X]^2
            mean_sq = (self._weight_sq_sums[stage_name] / n)
            var = (mean_sq - torch.tensor(mean) ** 2).clamp(min=0)
            std = var.sqrt().tolist()

            base_stage = _base_stage_name(stage_name)
            names = list(self.expert_names.get(stage_name, self.expert_names.get(base_stage, [])))
            if len(names) != len(mean):
                logger.warning(
                    "MoELogger: expert name count mismatch at stage '%s' (base='%s', names=%d, experts=%d).",
                    stage_name, base_stage, len(names), len(mean),
                )
                names = [f"E{i}" for i in range(len(mean))]

            K = len(mean)
            # Balance score: 1 - K * Σ(mean_k - 1/K)^2  (1.0 = perfect)
            target = 1.0 / K
            deviation = sum((m - target) ** 2 for m in mean)
            balance = max(0.0, 1.0 - K * deviation)

            summary["stages"][stage_name] = {
                "expert_names": names,
                "mean_weights": mean,
                "std_weights": std,
                "max_mean_weight": self._max_weights[stage_name],
                "min_mean_weight": self._min_weights[stage_name],
                "balance_score": balance,
            }

        self._reset()
        return summary

    @staticmethod
    def format_summary(summary: Dict) -> str:
        """Pretty-print an epoch summary dict."""
        lines = [f"MoE Expert Summary ({summary['n_batches']} batches):"]
        for stage_name, s in summary.get("stages", {}).items():
            names = s["expert_names"]
            means = s["mean_weights"]
            stds = s["std_weights"]
            lines.append(f"  [{stage_name}] balance={s['balance_score']:.4f}")
            for name, m, sd in zip(names, means, stds):
                lines.append(f"    {name:20s}: mean={m:.4f}  std={sd:.4f}")

        group_stages = summary.get("group_stages", {})
        if group_stages:
            lines.append("  [group-routing]")
            for stage_name, s in group_stages.items():
                names = s.get("expert_names", [])
                means = s.get("mean_weights", [])
                stds = s.get("std_weights", [])
                lines.append(f"    {stage_name}: balance={s.get('balance_score', 0.0):.4f}")
                for name, m, sd in zip(names, means, stds):
                    lines.append(f"      {name:18s}: mean={m:.4f}  std={sd:.4f}")

        router_stats = summary.get("router", {})
        if router_stats:
            lines.append("  [router-shape]")
            for stage_name, s in router_stats.items():
                active = s.get("active_clones_per_group", [])
                active_txt = ",".join(f"{float(v):.2f}" for v in active) if active else "-"
                lines.append(
                    f"    {stage_name}: group_entropy={float(s.get('group_entropy', 0.0)):.4f} "
                    f"active_clones/group={active_txt}"
                )
                if "teacher_guided_clone_entropy" in s:
                    lines.append(
                        f"      teacher_guided_clone_entropy={float(s.get('teacher_guided_clone_entropy', 0.0)):.4f}"
                    )
                clone_load = s.get("clone_load", [])
                clone_std = s.get("clone_load_std", [])
                group_names = s.get("group_names", [])
                if clone_load:
                    for idx, load_vec in enumerate(clone_load):
                        gname = group_names[idx] if idx < len(group_names) else f"group_{idx}"
                        std_vec = clone_std[idx] if idx < len(clone_std) else []
                        load_txt = ",".join(f"{float(v):.3f}" for v in load_vec)
                        std_txt = ",".join(f"{float(v):.3f}" for v in std_vec) if std_vec else "-"
                        lines.append(
                            f"      {gname:18s}: clone_load=[{load_txt}] clone_std=[{std_txt}]"
                        )
        return "\n".join(lines)
