"""
Expert 선택 패턴 고급 분석 모듈.

두 가지 분석을 제공:

1. **Expert-Performance 상관 분석**
   정답/오답 예측 시 expert별 평균 gating weight 비교.
   어떤 expert가 정확한 예측과 상관관계가 있는지 파악.

2. **Feature-Conditioned Expert 선택 편향 분석**
   주요 feature 값 구간(ratio bucket)별 expert 선택 분포 추적.
   예: entropy가 높을 때 Spectrum expert가 더 많이 선택되는지 등.

사용법::

    logger = ExpertAnalysisLogger(expert_names, col2idx, ...)
    # 학습 루프 내 매 배치:
    logger.accumulate(gate_weights, feat, logits, pos_items, item_seq_len)
    # 에포크 종료 시:
    summary = logger.get_and_reset()
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# 각 stage별 기본 분석 대상 feature (대표적 2~3개씩)
# -----------------------------------------------------------------------
DEFAULT_KEY_FEATURES: Dict[str, List[str]] = {
    "macro": ["mac_user_level", "mac_hist_ent", "mac_hist_pop_avg", "mac_hist_top1"],
    "mid":   ["mid_win_ent", "mid_pop_avg", "mid_int_avg", "mid_uniq_cat_r"],
    "micro": ["mic_last_int", "mic_switch", "mic_pop_avg", "mic_uniq_c", "mic_cat_ent", "mic_cat_top1"],
}


def _base_stage_name(stage_name: str) -> str:
    """Normalize repeated-stage key (e.g., macro@2 -> macro)."""
    if not isinstance(stage_name, str):
        return str(stage_name)
    return stage_name.split("@", 1)[0]

class ExpertAnalysisLogger:
    """Expert 선택 패턴 고급 분석기.

    Parameters
    ----------
    expert_names : dict
        {stage_name: [expert_name, ...]}
    col2idx : dict
        feature_name → feat tensor 내 인덱스.
    key_features : dict or None
        {stage_name: [feature_name, ...]} — 분석할 feature 목록.
        None이면 DEFAULT_KEY_FEATURES 사용.
    n_bins : int
        Feature 값 ratio 구간 수 (기본 5).
    sample_rate : float
        배치 샘플링 비율 (0.0~1.0). 분석 오버헤드 제어용.
    """

    def __init__(
        self,
        expert_names: Dict[str, List[str]],
        col2idx: Dict[str, int],
        key_features: Optional[Dict[str, List[str]]] = None,
        n_bins: int = 5,
        sample_rate: float = 0.2,
    ):
        self.expert_names = expert_names
        self.col2idx = col2idx
        self.n_bins = n_bins
        self.sample_rate = sample_rate

        # 분석 대상 feature 확정 (col2idx에 존재하는 것만)
        raw_key = key_features or DEFAULT_KEY_FEATURES
        self.key_features: Dict[str, List[str]] = {}
        self.key_feature_indices: Dict[str, List[int]] = {}
        for stage, feats in raw_key.items():
            if stage not in expert_names:
                continue
            valid = [f for f in feats if f in col2idx]
            if valid:
                self.key_features[stage] = valid
                self.key_feature_indices[stage] = [col2idx[f] for f in valid]

        self._idx2name = {v: k for k, v in col2idx.items()}
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 관리
    # ------------------------------------------------------------------

    def _reset(self):
        """누적 통계 초기화."""
        # Performance correlation
        self._correct_sums: Dict[str, torch.Tensor] = {}
        self._incorrect_sums: Dict[str, torch.Tensor] = {}
        self._n_correct = 0
        self._n_incorrect = 0

        # Feature-conditioned selection: {stage: {feat_idx: [n_bins, K]}}
        self._bin_weight_sums: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._bin_counts: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)

        self._batch_count = 0
        self._sampled_count = 0

    # ------------------------------------------------------------------
    # 배치 누적
    # ------------------------------------------------------------------

    @torch.no_grad()
    def accumulate(
        self,
        gate_weights: Dict[str, torch.Tensor],
        feat: torch.Tensor,
        logits: torch.Tensor,
        pos_items: torch.Tensor,
        item_seq_len: torch.Tensor,
    ):
        """배치 하나의 분석 데이터를 누적한다.

        Args:
            gate_weights : {stage: [B, T, K]} — 각 stage의 gating weight.
            feat         : [B, T, D_total]    — 전체 feature 텐서.
            logits       : [B, n_items]        — last-position 예측 logit.
            pos_items    : [B]                 — 정답 item ID.
            item_seq_len : [B]                 — 시퀀스 유효 길이.
        """
        self._batch_count += 1

        # 확률적 샘플링으로 오버헤드 제어
        if torch.rand(1).item() > self.sample_rate:
            return
        self._sampled_count += 1

        B = logits.shape[0]
        device = logits.device

        # 정답/오답 판별
        pred = logits.argmax(dim=-1)              # [B]
        correct_mask = (pred == pos_items)         # [B]
        n_corr = correct_mask.sum().item()
        n_incorr = B - n_corr

        for stage_name, w in gate_weights.items():
            full_stage_name = stage_name
            base_stage_name = _base_stage_name(full_stage_name)
            B_w, T_w, K = w.shape

            # Padding mask
            arange = torch.arange(T_w, device=device).unsqueeze(0)     # [1, T]
            valid_mask = arange < item_seq_len.unsqueeze(1)             # [B, T]

            # Sequence-mean gate weight: [B, K]
            w_masked = w * valid_mask.unsqueeze(-1).float()
            seq_len_safe = item_seq_len.float().clamp(min=1).unsqueeze(-1)
            w_seq_mean = w_masked.sum(dim=1) / seq_len_safe             # [B, K]

            # --- Performance correlation ---
            if full_stage_name not in self._correct_sums:
                self._correct_sums[full_stage_name] = torch.zeros(K)
                self._incorrect_sums[full_stage_name] = torch.zeros(K)

            if n_corr > 0:
                self._correct_sums[full_stage_name] += w_seq_mean[correct_mask].sum(0).cpu()
            if n_incorr > 0:
                self._incorrect_sums[full_stage_name] += w_seq_mean[~correct_mask].sum(0).cpu()

            # --- Feature-conditioned expert selection ---
            if base_stage_name not in self.key_feature_indices:
                continue

            # Flatten valid tokens
            flat_mask = valid_mask.reshape(-1)                          # [B*T]
            w_flat = w.reshape(-1, K)[flat_mask].cpu()                  # [N, K]
            feat_flat = feat.reshape(-1, feat.shape[-1])[flat_mask]     # [N, D] (GPU)

            for feat_col_idx in self.key_feature_indices[base_stage_name]:
                vals = feat_flat[:, feat_col_idx]                        # [N]
                finite_mask = torch.isfinite(vals)
                if not finite_mask.any():
                    continue

                vals = vals[finite_mask]
                w_feat = w_flat[finite_mask.cpu()]                       # [Nf, K]

                bins = self._ratio_bin(vals, self.n_bins).cpu()          # [Nf] int

                if feat_col_idx not in self._bin_weight_sums[full_stage_name]:
                    self._bin_weight_sums[full_stage_name][feat_col_idx] = torch.zeros(self.n_bins, K)
                    self._bin_counts[full_stage_name][feat_col_idx] = torch.zeros(self.n_bins)

                # Vectorized scatter accumulation
                bins_expanded = bins.unsqueeze(-1).expand_as(w_feat).long()
                self._bin_weight_sums[full_stage_name][feat_col_idx].scatter_add_(
                    0, bins_expanded, w_feat,
                )
                ones = torch.ones(bins.shape[0])
                self._bin_counts[full_stage_name][feat_col_idx].scatter_add_(
                    0, bins.long(), ones,
                )

        self._n_correct += n_corr
        self._n_incorrect += n_incorr

    # ------------------------------------------------------------------
    # Binning 유틸리티
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ratio(values: torch.Tensor) -> torch.Tensor:
        """Convert feature values to [0, 1] ratio scale.

        - If values are already in [0, 1], use them directly.
        - Otherwise, map z-like continuous values with Normal CDF.
        """
        if values.numel() == 0:
            return values

        x = values.float().contiguous()
        finite = torch.isfinite(x)
        if not finite.any():
            return torch.zeros_like(x)

        xf = x[finite]
        lo = xf.min()
        hi = xf.max()

        # Already ratio-like feature.
        if lo >= -1e-6 and hi <= 1.0 + 1e-6:
            ratio = x.clamp(0.0, 1.0)
            ratio[~finite] = 0.5
            return ratio

        # z-like feature: Normal CDF
        ratio = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        ratio = ratio.clamp(0.0, 1.0)
        ratio[~finite] = 0.5
        return ratio

    @classmethod
    def _ratio_bin(cls, values: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Assign ratio buckets: [0,1] -> bins [0..n_bins-1]."""
        if values.numel() == 0:
            return torch.zeros_like(values, dtype=torch.long)

        ratio = cls._to_ratio(values)
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=ratio.device)[1:-1].contiguous()
        return torch.bucketize(ratio.contiguous(), edges).clamp(0, n_bins - 1)

    @staticmethod
    def _ratio_label(bin_idx: int, n_bins: int) -> str:
        start = int(round(100 * bin_idx / n_bins))
        end = int(round(100 * (bin_idx + 1) / n_bins))
        return f"R{start}-{end}"

    # ------------------------------------------------------------------
    # 에포크 결과 반환
    # ------------------------------------------------------------------

    def get_and_reset(self) -> Dict:
        """에포크 분석 결과를 반환하고 내부 상태를 초기화한다.

        Returns
        -------
        dict
            {
                "n_batches": int,
                "n_sampled": int,
                "performance": {stage: {...}},
                "feature_bias": {stage: {feat_name: {"bins": [...]}}}
            }
        """
        result: Dict = {
            "n_batches": self._batch_count,
            "n_sampled": self._sampled_count,
            "performance": {},
            "feature_bias": {},
        }

        if self._sampled_count == 0:
            self._reset()
            return result

        # --- Performance ---
        n_corr = max(self._n_correct, 1)
        n_incorr = max(self._n_incorrect, 1)

        for stage_name in self._correct_sums:
            corr_mean = (self._correct_sums[stage_name] / n_corr).tolist()
            incorr_mean = (self._incorrect_sums[stage_name] / n_incorr).tolist()
            diff = [c - i for c, i in zip(corr_mean, incorr_mean)]
            base_stage_name = _base_stage_name(stage_name)
            expert_names = list(self.expert_names.get(base_stage_name, []))
            if len(expert_names) != len(corr_mean):
                expert_names = [f"E{i}" for i in range(len(corr_mean))]

            result["performance"][stage_name] = {
                "expert_names": expert_names,
                "correct_mean_weights": corr_mean,
                "incorrect_mean_weights": incorr_mean,
                "weight_diff": diff,
                "n_correct": self._n_correct,
                "n_incorrect": self._n_incorrect,
            }

        # --- Feature bias ---
        for stage_name, feat_map in self._bin_weight_sums.items():
            stage_data = {}
            base_stage_name = _base_stage_name(stage_name)
            allowed_feat_idx = set(self.key_feature_indices.get(base_stage_name, []))

            for feat_col_idx, weight_sums in feat_map.items():
                if allowed_feat_idx and feat_col_idx not in allowed_feat_idx:
                    continue

                feat_name = self._idx2name.get(feat_col_idx, f"feat_{feat_col_idx}")
                counts = self._bin_counts[stage_name][feat_col_idx]

                bins_data = []
                for b in range(self.n_bins):
                    count = int(counts[b].item())
                    if count <= 0:
                        continue
                    c = count
                    mean_w = (weight_sums[b] / c).tolist()
                    bins_data.append({
                        "bin": b,
                        "label": self._ratio_label(b, self.n_bins),
                        "mean_weights": mean_w,
                        "count": count,
                    })
                if bins_data:
                    stage_data[feat_name] = {"bins": bins_data}

            if stage_data:
                result["feature_bias"][stage_name] = stage_data

        self._reset()
        return result

    # ------------------------------------------------------------------
    # 콘솔 출력 포맷
    # ------------------------------------------------------------------

    @staticmethod
    def format_summary(analysis: Dict) -> str:
        """분석 결과를 읽기 쉬운 텍스트로 포맷한다."""
        if analysis.get("n_sampled", 0) == 0:
            return ""

        lines = [
            f"Expert Analysis ({analysis['n_sampled']}/{analysis['n_batches']} "
            f"batches sampled):"
        ]

        # Performance Correlation
        perf = analysis.get("performance", {})
        if perf:
            lines.append("  [Performance ↔ Expert]")
            for stage, data in perf.items():
                names = data["expert_names"]
                corr = data["correct_mean_weights"]
                incorr = data["incorrect_mean_weights"]
                diff = data["weight_diff"]
                nc, ni = data["n_correct"], data["n_incorrect"]
                lines.append(f"    {stage} (✓{nc} ✗{ni}):")
                for n, c, i, d in zip(names, corr, incorr, diff):
                    marker = "↑" if d > 0.005 else ("↓" if d < -0.005 else "·")
                    lines.append(
                        f"      {n:20s}: ✓={c:.4f}  ✗={i:.4f}  "
                        f"Δ={d:+.4f} {marker}"
                    )

        # Feature-Conditioned Selection
        fbias = analysis.get("feature_bias", {})
        if fbias:
            lines.append("  [Feature → Expert Bias]")
            for stage, features in fbias.items():
                lines.append(f"    {stage}:")
                for feat_name, data in features.items():
                    parts = []
                    for bd in data["bins"]:
                        w = bd["mean_weights"]
                        if not w:
                            continue
                        max_idx = w.index(max(w))
                        lbl = bd.get("label", f"Q{bd['bin']+1}")
                        parts.append(f"{lbl}→E{max_idx}({max(w):.3f})")
                    lines.append(f"      {feat_name:20s}: {' | '.join(parts)}")

        return "\n".join(lines)
