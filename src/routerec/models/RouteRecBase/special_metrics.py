"""Aggregated slice metrics for RouteRecBase evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .feature_config import feature_list_field


_POPULARITY_BINS = (
    ("cold_0", 0, 0),
    ("rare_1_5", 1, 5),
    ("6_20", 6, 20),
    ("21_100", 21, 100),
    ("101+", 101, None),
)
_POPULARITY_BINS_LEGACY = (
    ("<=5", 0, 5),
    ("6-20", 6, 20),
    ("21-100", 21, 100),
    (">100", 101, None),
)
_SESSION_LEN_BINS = (
    ("<=7", 1, 7),
    ("8-12", 8, 12),
    ("13+", 13, None),
)
_SESSION_LEN_BINS_LEGACY = (
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11+", 11, None),
)


@dataclass
class _MetricAccumulator:
    count: int = 0
    hit5_sum: float = 0.0
    hit10_sum: float = 0.0
    hit20_sum: float = 0.0
    ndcg5_sum: float = 0.0
    ndcg10_sum: float = 0.0
    ndcg20_sum: float = 0.0
    mrr5_sum: float = 0.0
    mrr10_sum: float = 0.0
    mrr20_sum: float = 0.0

    def update(self, *, metrics: Dict[str, float]) -> None:
        self.count += 1
        self.hit5_sum += float(metrics.get("hit@5", 0.0))
        self.hit10_sum += float(metrics.get("hit@10", 0.0))
        self.hit20_sum += float(metrics.get("hit@20", 0.0))
        self.ndcg5_sum += float(metrics.get("ndcg@5", 0.0))
        self.ndcg10_sum += float(metrics.get("ndcg@10", 0.0))
        self.ndcg20_sum += float(metrics.get("ndcg@20", 0.0))
        self.mrr5_sum += float(metrics.get("mrr@5", 0.0))
        self.mrr10_sum += float(metrics.get("mrr@10", 0.0))
        self.mrr20_sum += float(metrics.get("mrr@20", 0.0))

    def finalize(self) -> dict:
        if self.count <= 0:
            return {
                "count": 0,
                "hit@5": 0.0, "hit@10": 0.0, "hit@20": 0.0,
                "ndcg@5": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0,
                "mrr@5": 0.0, "mrr@10": 0.0, "mrr@20": 0.0,
            }
        denom = float(self.count)
        return {
            "count": int(self.count),
            "hit@5": self.hit5_sum / denom,
            "hit@10": self.hit10_sum / denom,
            "hit@20": self.hit20_sum / denom,
            "ndcg@5": self.ndcg5_sum / denom,
            "ndcg@10": self.ndcg10_sum / denom,
            "ndcg@20": self.ndcg20_sum / denom,
            "mrr@5": self.mrr5_sum / denom,
            "mrr@10": self.mrr10_sum / denom,
            "mrr@20": self.mrr20_sum / denom,
        }


def _bucket_name(value: int, bins) -> str:
    for name, lo, hi in bins:
        if hi is None and value >= lo:
            return name
        if lo <= value <= hi:
            return name
    return "unknown"


def _safe_tensor(values, *, device) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device)
    return torch.as_tensor(values, device=device)


def _extract_sequence_feature(
    interaction,
    field_name: str,
    row_idx: torch.Tensor,
    item_seq_len: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if field_name not in interaction:
        return None
    values = interaction[field_name]
    if not torch.is_tensor(values):
        values = torch.as_tensor(values, device=row_idx.device)
    else:
        values = values.to(device=row_idx.device)
    if values.ndim <= 0 or values.size(0) <= 0:
        return None
    if row_idx.numel() <= 0:
        return None
    max_rows = int(values.size(0))
    if max_rows <= 0:
        return None
    valid_row_mask = (row_idx >= 0) & (row_idx < max_rows)
    if not bool(valid_row_mask.all()):
        if not bool(valid_row_mask.any()):
            return None
        row_idx = row_idx[valid_row_mask]
    values = values.index_select(0, row_idx)
    if values.ndim <= 1:
        return values.float()
    if item_seq_len is None:
        last_idx = torch.full(
            (values.size(0),),
            fill_value=max(values.size(1) - 1, 0),
            dtype=torch.long,
            device=values.device,
        )
    else:
        if not torch.is_tensor(item_seq_len):
            item_seq_len = torch.as_tensor(item_seq_len, device=values.device)
        else:
            item_seq_len = item_seq_len.to(device=values.device)
        if item_seq_len.ndim <= 0 or item_seq_len.size(0) <= 0:
            return None
        max_len_rows = int(item_seq_len.size(0))
        valid_len_mask = (row_idx >= 0) & (row_idx < max_len_rows)
        if not bool(valid_len_mask.all()):
            if not bool(valid_len_mask.any()):
                return None
            row_idx = row_idx[valid_len_mask]
            values = values[valid_len_mask]
        last_idx = item_seq_len.index_select(0, row_idx).long().clamp(min=1, max=values.size(1)) - 1
    return values[torch.arange(values.size(0), device=values.device), last_idx].float()


class SpecialMetricCollector:
    """Collect run-level slice metrics for valid/test evaluation."""

    def __init__(
        self,
        *,
        split_name: str,
        item_counts: torch.Tensor,
        item_seq_len_field: str,
        new_user_field: Optional[str],
        config_snapshot: Dict[str, object],
    ):
        self.split_name = str(split_name)
        self.item_counts = item_counts.long().cpu()
        self.item_seq_len_field = str(item_seq_len_field)
        self.new_user_field = str(new_user_field) if new_user_field else None
        self.config_snapshot = dict(config_snapshot)

        self.overall = _MetricAccumulator()
        self.overall_seen_target = _MetricAccumulator()
        self.overall_unseen_target = _MetricAccumulator()
        self.slices = {
            "target_popularity_abs": {name: _MetricAccumulator() for name, _, _ in _POPULARITY_BINS},
            "target_popularity_abs_legacy": {name: _MetricAccumulator() for name, _, _ in _POPULARITY_BINS_LEGACY},
            "session_len": {name: _MetricAccumulator() for name, _, _ in _SESSION_LEN_BINS},
            "session_len_legacy": {name: _MetricAccumulator() for name, _, _ in _SESSION_LEN_BINS_LEGACY},
        }
        if self.new_user_field:
            self.slices["new_user"] = {
                "existing": _MetricAccumulator(),
                "new": _MetricAccumulator(),
            }

    @staticmethod
    def _rank_metrics(ranks: torch.Tensor) -> Dict[str, torch.Tensor]:
        ranks = ranks.float()
        out: Dict[str, torch.Tensor] = {}
        for k in (5, 10, 20):
            hit = (ranks <= k).float()
            ndcg = torch.where(
                ranks <= k,
                1.0 / torch.log2(ranks + 1.0),
                torch.zeros_like(ranks),
            )
            mrr = torch.where(ranks <= k, 1.0 / ranks, torch.zeros_like(ranks))
            out[f"hit@{k}"] = hit
            out[f"ndcg@{k}"] = ndcg
            out[f"mrr@{k}"] = mrr
        return out

    def update(self, *, interaction, scores: torch.Tensor, positive_u, positive_i) -> None:
        if scores is None:
            return

        device = scores.device
        row_idx = _safe_tensor(positive_u, device=device).long()
        pos_idx = _safe_tensor(positive_i, device=device).long()
        if row_idx.numel() == 0:
            return

        batch_scores = scores.index_select(0, row_idx)
        pos_scores = batch_scores.gather(1, pos_idx.view(-1, 1)).squeeze(1)
        ranks = (batch_scores > pos_scores.unsqueeze(1)).sum(dim=1) + 1
        metrics_tensor = self._rank_metrics(ranks)

        item_seq_len = None
        if self.item_seq_len_field in interaction:
            item_seq_len = interaction[self.item_seq_len_field]
            if not torch.is_tensor(item_seq_len):
                item_seq_len = torch.as_tensor(item_seq_len, device=device)
            else:
                item_seq_len = item_seq_len.to(device=device)
        elif "item_length" in interaction:
            item_seq_len = interaction["item_length"]
            if not torch.is_tensor(item_seq_len):
                item_seq_len = torch.as_tensor(item_seq_len, device=device)
            else:
                item_seq_len = item_seq_len.to(device=device)

        if item_seq_len is None:
            session_len = torch.ones_like(row_idx, dtype=torch.long)
        else:
            session_len = item_seq_len.index_select(0, row_idx).long()

        pop_values = self.item_counts.index_select(0, pos_idx.detach().cpu()).tolist()

        if self.new_user_field:
            new_user = _extract_sequence_feature(
                interaction,
                self.new_user_field,
                row_idx,
                item_seq_len,
            )
        else:
            new_user = None

        metrics_values = {
            key: tensor.detach().cpu().tolist()
            for key, tensor in metrics_tensor.items()
        }
        session_len_values = session_len.detach().cpu().tolist()
        new_user_values = new_user.detach().cpu().tolist() if new_user is not None else None

        for idx, pop_count in enumerate(pop_values):
            point = {k: float(v[idx]) for k, v in metrics_values.items()}

            self.overall.update(metrics=point)
            if int(pop_count) <= 0:
                self.overall_unseen_target.update(metrics=point)
            else:
                self.overall_seen_target.update(metrics=point)
            pop_bucket = _bucket_name(int(pop_count), _POPULARITY_BINS)
            self.slices["target_popularity_abs"][pop_bucket].update(metrics=point)
            pop_bucket_legacy = _bucket_name(int(pop_count), _POPULARITY_BINS_LEGACY)
            self.slices["target_popularity_abs_legacy"][pop_bucket_legacy].update(metrics=point)

            session_bucket = _bucket_name(int(session_len_values[idx]), _SESSION_LEN_BINS)
            self.slices["session_len"][session_bucket].update(metrics=point)
            session_bucket_legacy = _bucket_name(int(session_len_values[idx]), _SESSION_LEN_BINS_LEGACY)
            self.slices["session_len_legacy"][session_bucket_legacy].update(metrics=point)

            if new_user_values is not None and "new_user" in self.slices:
                label = "new" if float(new_user_values[idx]) >= 0.5 else "existing"
                self.slices["new_user"][label].update(metrics=point)

    def finalize(self) -> dict:
        slice_payload = {
            slice_name: {
                bucket_name: accumulator.finalize()
                for bucket_name, accumulator in bucket_map.items()
                if accumulator.count > 0
            }
            for slice_name, bucket_map in self.slices.items()
        }
        count_payload = {
            slice_name: {
                bucket_name: int(accumulator.count)
                for bucket_name, accumulator in bucket_map.items()
                if accumulator.count > 0
            }
            for slice_name, bucket_map in self.slices.items()
        }
        return {
            "split": self.split_name,
            "overall": self.overall.finalize(),
            "overall_seen_target": self.overall_seen_target.finalize(),
            "overall_unseen_target": self.overall_unseen_target.finalize(),
            "slices": slice_payload,
            "counts": count_payload,
            "config_snapshot": dict(self.config_snapshot),
        }


def build_special_metric_config_snapshot(
    *,
    feature_available: bool,
    new_user_available: bool,
) -> dict:
    return {
        "target_popularity_abs_bins": [name for name, _, _ in _POPULARITY_BINS],
        "target_popularity_abs_bins_legacy": [name for name, _, _ in _POPULARITY_BINS_LEGACY],
        "session_len_bins": [name for name, _, _ in _SESSION_LEN_BINS],
        "session_len_bins_legacy": [name for name, _, _ in _SESSION_LEN_BINS_LEGACY],
        "item_count_source": "unknown",
        "item_count_definition": "train_interaction_count_per_item",
        "new_user_enabled": bool(new_user_available),
        "feature_logging_available": bool(feature_available),
    }


def default_new_user_field() -> str:
    return feature_list_field("mac_is_new")
