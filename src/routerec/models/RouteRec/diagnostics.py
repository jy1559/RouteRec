"""Evaluation diagnostics for RouteRec."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _base_stage_name(stage_key: str) -> str:
    text = str(stage_key or "")
    if "@" in text:
        text = text.split("@", 1)[0]
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _as_tensor(value, *, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _valid_mask(item_seq_len: Optional[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
    if item_seq_len is None:
        return torch.ones(1, seq_len, dtype=torch.bool, device=device)
    lens = item_seq_len.to(device=device).long().clamp(min=1, max=seq_len)
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    return arange < lens.unsqueeze(1)


def _float_list(tensor: torch.Tensor) -> List[float]:
    return [float(v) for v in tensor.detach().cpu().tolist()]


def _metric_name_token(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "group"
    out = []
    prev_us = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    token = "".join(out).strip("_")
    return token or "group"


def _build_group_routing_payload(raw: dict) -> dict:
    """Summarize group-level routing stats from accumulated stage data."""
    n_groups = int(raw.get("n_groups", 0))
    if n_groups <= 0:
        return {}
    family_names = list(raw.get("family_names", []))

    group_usage_sum = raw["group_usage_sum"]
    group_top1_count = raw["group_top1_count"]
    g_total = float(group_usage_sum.sum().item())
    if g_total > 0:
        group_share = (group_usage_sum / g_total).tolist()
        g_cv = float(group_usage_sum.std(unbiased=False).item() / max(group_usage_sum.mean().item(), 1e-8))
        g_n_eff = float(1.0 / (group_usage_sum / g_total).pow(2).sum().item()) if n_groups > 1 else 1.0
        g_top1_total = float(group_top1_count.sum().item())
        g_top1_max = float(group_top1_count.max().item() / max(g_top1_total, 1.0))
    else:
        group_share = [0.0] * n_groups
        g_cv, g_n_eff, g_top1_max = 0.0, 0.0, 0.0

    g_n_tok = max(int(raw.get("group_n_tokens", 0)), 1)
    group_entropy_mean = float(raw.get("group_entropy_sum", 0.0)) / g_n_tok

    prior_n = max(int(raw.get("group_prior_n", 0)), 1)
    group_prior_sum = raw["group_prior_sum"]
    group_prior_mean = (group_prior_sum / prior_n).tolist() if float(group_prior_sum.sum()) > 0 else [0.0] * n_groups

    fg_n_tok = max(int(raw.get("factored_group_n_tokens", 0)), 1)
    factored_group_entropy_mean = float(raw.get("factored_group_entropy_sum", 0.0)) / fg_n_tok

    return {
        "group_names": family_names[:n_groups],
        "group_share": group_share,
        "group_n_eff": g_n_eff,
        "group_cv_usage": g_cv,
        "group_top1_max_frac": g_top1_max,
        "group_entropy_mean": group_entropy_mean,
        "group_prior_mean": group_prior_mean,
        "factored_group_entropy_mean": factored_group_entropy_mean,
    }


def _entropy_norm(entropy_mean: float, support_size: int) -> float:
    support = max(int(support_size), 1)
    if support <= 1:
        return 0.0
    return float(entropy_mean / max(math.log(float(support)), 1e-8))


def _top1_monopoly_norm(top1_max_frac: float, support_size: int) -> float:
    support = max(int(support_size), 1)
    if support <= 1:
        return float(max(top1_max_frac, 0.0))
    uniform = 1.0 / float(support)
    denom = max(1.0 - uniform, 1e-8)
    return float((float(top1_max_frac) - uniform) / denom)


def _js_to_score(js_value: float) -> float:
    return float(math.exp(-float(js_value)))


class N3DiagnosticCollector:
    """Collect compact routing/conditioning diagnostics over one evaluation split."""

    def __init__(
        self,
        *,
        split_name: str,
        stage_family_features: Dict[str, Dict[str, List[str]]],
        stage_expert_names: Dict[str, List[str]],
        stage_router_granularity: Optional[Dict[str, str]] = None,
        all_feature_columns: List[str],
        max_positions: int,
        feature_mode: str = "none",
        consistency_pairs: int = 4,
        pair_max_points: int = 4096,
        pair_bin_count: int = 20,
    ):
        self.split_name = str(split_name)
        self.stage_family_features = {
            str(stage): {str(group): list(cols) for group, cols in groups.items()}
            for stage, groups in dict(stage_family_features or {}).items()
        }
        self.stage_expert_names = {
            str(stage): list(names) for stage, names in dict(stage_expert_names or {}).items()
        }
        self.stage_router_granularity = {
            str(stage): str(gran).lower().strip()
            for stage, gran in dict(stage_router_granularity or {}).items()
        }
        self.col2idx = {name: idx for idx, name in enumerate(list(all_feature_columns or []))}
        self.max_positions = max(int(max_positions), 1)
        self.feature_mode = str(feature_mode or "none")
        self.consistency_pairs = max(int(consistency_pairs), 1)
        self.pair_max_points = max(int(pair_max_points), 0)
        self.pair_bin_count = max(int(pair_bin_count), 4)
        self.pair_points_per_update = min(1024, max(self.pair_max_points, 1))
        self._stage_data: Dict[str, dict] = {}

    def _stage_aggregation_level(self, stage_key: str) -> str:
        base = _base_stage_name(stage_key)
        gran = str(self.stage_router_granularity.get(base, "token") or "token").lower().strip()
        return "session" if gran == "session" else "token"

    @staticmethod
    def _dominant_name(counter: Dict[str, int], default: str = "") -> str:
        if not counter:
            return default
        return max(counter.items(), key=lambda kv: int(kv[1]))[0]

    def _ensure_node_acc(
        self,
        stage_store: dict,
        *,
        node_name: str,
        node_kind: str,
        route_space: str,
        support_size: int,
        aggregation_level: str,
        wrapper_name: str,
        source_type: str = "",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        node_map = stage_store.setdefault("node_acc", {})
        if node_name not in node_map:
            node_map[node_name] = {
                "node_name": str(node_name),
                "node_kind": str(node_kind),
                "route_space": str(route_space),
                "support_size": int(max(support_size, 1)),
                "aggregation_level": str(aggregation_level),
                "wrapper_name": str(wrapper_name or ""),
                "source_type": str(source_type or ""),
                "temperature": None if temperature is None else float(temperature),
                "top_k": None if top_k is None else int(top_k),
                "usage_sum": torch.zeros(int(max(support_size, 1))),
                "top1_count": torch.zeros(int(max(support_size, 1))),
                "entropy_sum": 0.0,
                "n_tokens": 0,
                "knn_js_sum": 0.0,
                "knn_js_count": 0,
            }
        node = node_map[node_name]
        if node.get("wrapper_name", "") == "" and wrapper_name:
            node["wrapper_name"] = str(wrapper_name)
        return node

    def _accumulate_node_probs(
        self,
        stage_store: dict,
        *,
        node_name: str,
        node_kind: str,
        route_space: str,
        probs: torch.Tensor,
        mask: torch.Tensor,
        aggregation_level: str,
        wrapper_name: str,
        neigh_idx: Optional[torch.Tensor] = None,
        neigh_k: int = 0,
        source_type: str = "",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if not torch.is_tensor(probs) or probs.ndim != 3 or probs.size(-1) <= 0:
            return
        if not torch.is_tensor(mask) or mask.ndim != 2:
            return
        if probs.size(0) != mask.size(0) or probs.size(1) != mask.size(1):
            return

        node = self._ensure_node_acc(
            stage_store,
            node_name=node_name,
            node_kind=node_kind,
            route_space=route_space,
            support_size=int(probs.size(-1)),
            aggregation_level=aggregation_level,
            wrapper_name=wrapper_name,
            source_type=source_type,
            temperature=temperature,
            top_k=top_k,
        )
        p = probs.detach()
        valid = mask.unsqueeze(-1).float()
        n_valid = int(mask.sum().item())
        if n_valid <= 0:
            return
        node["usage_sum"] += (p * valid).sum(dim=(0, 1)).cpu()
        entropy = -(p.clamp(min=1e-8) * p.clamp(min=1e-8).log()).sum(dim=-1)
        node["entropy_sum"] += float((entropy * mask.float()).sum().item())
        node["n_tokens"] += n_valid
        top1 = p.argmax(dim=-1)
        node["top1_count"] += torch.bincount(top1[mask].cpu(), minlength=int(p.size(-1))).float()

        if neigh_idx is not None and neigh_k > 0 and int(p.size(0)) > 1:
            sess_prob = (p * mask.unsqueeze(-1).float()).sum(dim=1)
            sess_prob = sess_prob / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            sess_prob = sess_prob.clamp(min=1e-8)
            pi = sess_prob.unsqueeze(1).expand(-1, neigh_k, -1)
            pj = sess_prob.index_select(0, neigh_idx.reshape(-1)).reshape(int(p.size(0)), neigh_k, -1).clamp(min=1e-8)
            mix = 0.5 * (pi + pj)
            js = 0.5 * (
                (pi * (pi.log() - mix.log())).sum(dim=-1)
                + (pj * (pj.log() - mix.log())).sum(dim=-1)
            )
            node["knn_js_sum"] += float(js.sum().item())
            node["knn_js_count"] += int(js.numel())

    def _ensure_stage(self, stage_key: str, *, n_slots: int, mode: str) -> dict:
        if stage_key not in self._stage_data:
            base_stage = _base_stage_name(stage_key)
            family_names = list(self.stage_family_features.get(base_stage, {}).keys())
            n_groups = max(len(family_names), 1)
            self._stage_data[stage_key] = {
                "mode": mode,
                "n_slots": int(n_slots),
                "aggregation_level": self._stage_aggregation_level(stage_key),
                "usage_sum": torch.zeros(n_slots),
                "top1_count": torch.zeros(n_slots),
                "entropy_sum": 0.0,
                "n_tokens": 0,
                "n_sessions": 0,
                "dead_tokens": 0,
                "jitter_adj_sum": 0.0,
                "jitter_adj_count": 0,
                "jitter_session_sum": 0.0,
                "family_expert": torch.zeros(len(family_names), n_slots),
                "family_names": family_names,
                "position_usage": torch.zeros(self.max_positions, n_slots),
                "transition": torch.zeros(n_slots, n_slots),
                "cond_norm_sum": 0.0,
                "cond_norm_count": 0,
                "delta_norm_sum": 0.0,
                "delta_norm_count": 0,
                "shared_delta_norm_sum": 0.0,
                "shared_delta_norm_count": 0,
                "moe_delta_norm_sum": 0.0,
                "moe_delta_norm_count": 0,
                "residual_delta_norm_sum": 0.0,
                "residual_delta_norm_count": 0,
                "alpha_value_sum": 0.0,
                "alpha_value_count": 0,
                "alpha_effective_sum": 0.0,
                "alpha_effective_count": 0,
                "expert_out_sum": torch.zeros(n_slots, 1),
                # Group-level routing diagnostics
                "group_usage_sum": torch.zeros(n_groups),
                "group_top1_count": torch.zeros(n_groups),
                "group_entropy_sum": 0.0,
                "group_n_tokens": 0,
                "group_prior_sum": torch.zeros(n_groups),
                "group_prior_n": 0,
                "factored_group_entropy_sum": 0.0,
                "factored_group_n_tokens": 0,
                "n_groups": n_groups,
                # KNN feature-route consistency diagnostics
                "consistency_js_sum": 0.0,
                "consistency_js_count": 0,
                "group_consistency_js_sum": 0.0,
                "group_consistency_js_count": 0,
                "intra_group_consistency_js_sum": torch.zeros(n_groups),
                "intra_group_consistency_js_count": torch.zeros(n_groups),
                "feature_group_consistency_js_sum": torch.zeros(n_groups),
                "feature_group_consistency_js_count": torch.zeros(n_groups),
                # Pair-level feature-vs-routing similarity logging
                "pair_feat_sum": torch.zeros(self.pair_bin_count),
                "pair_route_sum": torch.zeros(self.pair_bin_count),
                "pair_js_sum": torch.zeros(self.pair_bin_count),
                "pair_count": torch.zeros(self.pair_bin_count),
                "pair_sample_points": [],
                "wrapper_name_hist": defaultdict(int),
                "node_acc": {},
            }
        return self._stage_data[stage_key]

    @staticmethod
    def _pairwise_js(pi: torch.Tensor, pj: torch.Tensor) -> torch.Tensor:
        mix = 0.5 * (pi + pj)
        return 0.5 * (
            (pi * (pi.log() - mix.log())).sum(dim=-1)
            + (pj * (pj.log() - mix.log())).sum(dim=-1)
        )

    def _update_pair_similarity_logging(
        self,
        *,
        stage_store: dict,
        feat_norm: torch.Tensor,
        route_session_prob: torch.Tensor,
    ) -> None:
        if self.pair_max_points <= 0:
            return
        if not torch.is_tensor(feat_norm) or not torch.is_tensor(route_session_prob):
            return
        if feat_norm.ndim != 2 or route_session_prob.ndim != 2:
            return
        if feat_norm.size(0) != route_session_prob.size(0):
            return
        n_sessions = int(feat_norm.size(0))
        if n_sessions <= 1:
            return

        n_pairs = min(self.pair_points_per_update, n_sessions * (n_sessions - 1))
        if n_pairs <= 0:
            return
        device = feat_norm.device
        i = torch.randint(0, n_sessions, (n_pairs,), device=device)
        j = torch.randint(0, n_sessions - 1, (n_pairs,), device=device)
        j = torch.where(j >= i, j + 1, j)

        feature_cos = (feat_norm.index_select(0, i) * feat_norm.index_select(0, j)).sum(dim=-1).clamp(min=-1.0, max=1.0)
        pi = route_session_prob.index_select(0, i).clamp(min=1e-8)
        pj = route_session_prob.index_select(0, j).clamp(min=1e-8)
        route_js = self._pairwise_js(pi, pj)
        route_sim = torch.exp(-route_js)

        bin_idx = (((feature_cos + 1.0) * 0.5) * float(self.pair_bin_count)).long().clamp(min=0, max=self.pair_bin_count - 1)
        bin_cpu = bin_idx.detach().cpu()
        feature_cpu = feature_cos.detach().cpu()
        route_cpu = route_sim.detach().cpu()
        js_cpu = route_js.detach().cpu()

        stage_store["pair_count"] += torch.bincount(bin_cpu, minlength=self.pair_bin_count).float()
        stage_store["pair_feat_sum"] += torch.bincount(bin_cpu, weights=feature_cpu, minlength=self.pair_bin_count).float()
        stage_store["pair_route_sum"] += torch.bincount(bin_cpu, weights=route_cpu, minlength=self.pair_bin_count).float()
        stage_store["pair_js_sum"] += torch.bincount(bin_cpu, weights=js_cpu, minlength=self.pair_bin_count).float()

        remain = max(self.pair_max_points - len(stage_store["pair_sample_points"]), 0)
        if remain <= 0:
            return
        keep = min(remain, int(feature_cpu.numel()))
        for idx in range(keep):
            stage_store["pair_sample_points"].append(
                {
                    "feature_cosine": float(feature_cpu[idx].item()),
                    "routing_js": float(js_cpu[idx].item()),
                    "routing_similarity": float(route_cpu[idx].item()),
                }
            )

    @torch.no_grad()
    def update(
        self,
        *,
        interaction,
        feat: Optional[torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
        aux_data: Dict[str, object],
    ) -> None:
        gate_weights = dict(aux_data.get("gate_weights", {}) or {})
        router_aux = dict(aux_data.get("router_aux", {}) or {})
        dense_aux = dict(aux_data.get("dense_aux", {}) or {})
        if not gate_weights and not dense_aux:
            return

        device = item_seq_len.device if torch.is_tensor(item_seq_len) else None
        if device is None and gate_weights:
            device = next(iter(gate_weights.values())).device
        if device is None and dense_aux:
            device = next(iter(dense_aux.values())).get("delta_norm", torch.zeros(1)).device
        if device is None:
            return

        neigh_idx = None
        neigh_k = 0
        feat_session_norm = None
        if torch.is_tensor(feat) and feat.ndim == 3 and feat.size(0) > 1 and feat.size(-1) > 0:
            feat_repr = feat.to(device=device).mean(dim=1)
            feat_session_norm = F.normalize(feat_repr, p=2, dim=-1)
            sim = feat_session_norm @ feat_session_norm.transpose(0, 1)
            sim.fill_diagonal_(-1e9)
            neigh_k = min(int(self.consistency_pairs), int(feat.size(0)) - 1)
            if neigh_k > 0:
                neigh_idx = sim.topk(k=neigh_k, dim=1).indices

        for stage_key, weights in gate_weights.items():
            if not torch.is_tensor(weights):
                continue
            seq_len = weights.size(1)
            mask = _valid_mask(item_seq_len, seq_len, weights.device)
            if mask.size(0) == 1 and weights.size(0) > 1:
                mask = mask.expand(weights.size(0), -1)

            stage_store = self._ensure_stage(stage_key, n_slots=weights.size(-1), mode="moe")
            w = weights.detach()
            valid = mask.unsqueeze(-1).float()
            n_valid = int(mask.sum().item())
            if n_valid <= 0:
                continue
            wrapper_alias = str(dict(router_aux.get("wrapper_alias", {}) or {}).get(stage_key, "") or "").strip()
            wrapper_name = str(dict(router_aux.get("wrapper_name", {}) or {}).get(stage_key, "") or wrapper_alias).strip()
            if wrapper_name:
                stage_store["wrapper_name_hist"][wrapper_name] += 1
            aggregation_level = str(stage_store.get("aggregation_level", "token") or "token")
            stage_store["usage_sum"] += (w * valid).sum(dim=(0, 1)).cpu()
            entropy = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
            stage_store["entropy_sum"] += float((entropy * mask.float()).sum().item())
            stage_store["n_tokens"] += n_valid
            stage_store["n_sessions"] += int(weights.size(0))

            self._accumulate_node_probs(
                stage_store,
                node_name="final.expert",
                node_kind="final",
                route_space="expert",
                probs=w,
                mask=mask,
                aggregation_level=aggregation_level,
                wrapper_name=wrapper_name,
                neigh_idx=neigh_idx,
                neigh_k=neigh_k,
            )

            top1 = w.argmax(dim=-1)
            flat_top1 = top1[mask]
            stage_store["top1_count"] += torch.bincount(flat_top1.cpu(), minlength=weights.size(-1)).float()

            session_prob = None

            if neigh_idx is not None and neigh_k > 0 and int(weights.size(0)) > 1:
                session_prob = (w * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                session_prob = session_prob.clamp(min=1e-8)
                pi = session_prob.unsqueeze(1).expand(-1, neigh_k, -1)
                pj = session_prob.index_select(0, neigh_idx.reshape(-1)).reshape(int(weights.size(0)), neigh_k, -1).clamp(min=1e-8)
                mix = 0.5 * (pi + pj)
                js = 0.5 * (
                    (pi * (pi.log() - mix.log())).sum(dim=-1)
                    + (pj * (pj.log() - mix.log())).sum(dim=-1)
                )
                stage_store["consistency_js_sum"] += float(js.sum().item())
                stage_store["consistency_js_count"] += int(js.numel())

            if session_prob is None:
                session_prob = (w * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                session_prob = session_prob.clamp(min=1e-8)
            if feat_session_norm is not None:
                self._update_pair_similarity_logging(
                    stage_store=stage_store,
                    feat_norm=feat_session_norm,
                    route_session_prob=session_prob,
                )

            for b_idx in range(weights.size(0)):
                valid_pos = mask[b_idx].nonzero(as_tuple=False).view(-1)
                if valid_pos.numel() <= 0:
                    continue
                top1_seq = top1[b_idx, valid_pos]
                if top1_seq.numel() > 1:
                    changes = (top1_seq[1:] != top1_seq[:-1]).float()
                    stage_store["jitter_adj_sum"] += float(changes.sum().item())
                    stage_store["jitter_adj_count"] += int(changes.numel())
                    majority = torch.bincount(top1_seq.cpu(), minlength=weights.size(-1)).float().max().item()
                    stage_store["jitter_session_sum"] += 1.0 - (majority / float(top1_seq.numel()))
                    src = top1_seq[:-1]
                    dst = top1_seq[1:]
                    trans = torch.zeros_like(stage_store["transition"])
                    trans.index_put_(
                        (src.cpu(), dst.cpu()),
                        torch.ones(src.numel()),
                        accumulate=True,
                    )
                    stage_store["transition"] += trans

                capped = valid_pos[: self.max_positions]
                top1_cap = top1[b_idx, capped]
                pos_idx = torch.arange(top1_cap.numel(), dtype=torch.long)
                pos_updates = torch.zeros_like(stage_store["position_usage"])
                pos_updates.index_put_((pos_idx, top1_cap.cpu()), torch.ones(top1_cap.numel()), accumulate=True)
                stage_store["position_usage"] += pos_updates

            family_map = self.stage_family_features.get(_base_stage_name(stage_key), {})
            if feat is not None and family_map:
                raw_feat = feat.to(device=weights.device)
                for fam_idx, family_name in enumerate(stage_store["family_names"]):
                    cols = [self.col2idx[col] for col in family_map.get(family_name, []) if col in self.col2idx]
                    if not cols:
                        continue
                    fam_score = raw_feat.index_select(-1, torch.tensor(cols, device=raw_feat.device)).mean(dim=-1)
                    family_usage = (fam_score.unsqueeze(-1) * w * mask.unsqueeze(-1).float()).sum(dim=(0, 1)).cpu()
                    stage_store["family_expert"][fam_idx] += family_usage

            cond = dict(dense_aux.get(stage_key, {}) or {})
            if cond:
                delta_norm = cond.get("delta_norm")
                cond_norm = cond.get("condition_norm")
                if torch.is_tensor(delta_norm):
                    stage_store["delta_norm_sum"] += float(delta_norm.detach().sum().item())
                    stage_store["delta_norm_count"] += int(delta_norm.numel())
                if torch.is_tensor(cond_norm):
                    stage_store["cond_norm_sum"] += float(cond_norm.detach().sum().item())
                    stage_store["cond_norm_count"] += int(cond_norm.numel())
                shared_delta_norm = cond.get("shared_delta_norm")
                if torch.is_tensor(shared_delta_norm):
                    stage_store["shared_delta_norm_sum"] += float(shared_delta_norm.detach().sum().item())
                    stage_store["shared_delta_norm_count"] += int(shared_delta_norm.numel())
                moe_delta_norm = cond.get("moe_delta_norm")
                if torch.is_tensor(moe_delta_norm):
                    stage_store["moe_delta_norm_sum"] += float(moe_delta_norm.detach().sum().item())
                    stage_store["moe_delta_norm_count"] += int(moe_delta_norm.numel())
                residual_delta_norm = cond.get("residual_delta_norm")
                if torch.is_tensor(residual_delta_norm):
                    stage_store["residual_delta_norm_sum"] += float(residual_delta_norm.detach().sum().item())
                    stage_store["residual_delta_norm_count"] += int(residual_delta_norm.numel())
                alpha_value = cond.get("alpha_value")
                if torch.is_tensor(alpha_value):
                    stage_store["alpha_value_sum"] += float(alpha_value.detach().sum().item())
                    stage_store["alpha_value_count"] += int(alpha_value.numel())
                alpha_effective = cond.get("alpha_effective")
                if torch.is_tensor(alpha_effective):
                    stage_store["alpha_effective_sum"] += float(alpha_effective.detach().sum().item())
                    stage_store["alpha_effective_count"] += int(alpha_effective.numel())

            expert_out_mean = dict(router_aux.get("expert_output_mean", {}) or {}).get(stage_key)
            if torch.is_tensor(expert_out_mean) and expert_out_mean.ndim == 2:
                stage_store["expert_out_sum"] = stage_store["expert_out_sum"].new_zeros(expert_out_mean.size(0), expert_out_mean.size(1))
                stage_store["expert_out_sum"] += expert_out_mean.detach().cpu()
                stage_store["expert_out_dim"] = int(expert_out_mean.size(1))

            # --- Group-level routing diagnostics ---
            # group_weights: actual per-group routing distribution (aggregated from expert weights)
            group_weights = dict(router_aux.get("group_weights", {}) or {}).get(stage_key)
            if torch.is_tensor(group_weights) and group_weights.size(-1) >= 1:
                gw = group_weights.detach()
                n_grps = gw.size(-1)
                if stage_store["n_groups"] != n_grps:
                    # Resize group accumulators if needed (e.g. first time n_groups was unknown)
                    stage_store["group_usage_sum"] = torch.zeros(n_grps)
                    stage_store["group_top1_count"] = torch.zeros(n_grps)
                    stage_store["group_prior_sum"] = torch.zeros(n_grps)
                    stage_store["intra_group_consistency_js_sum"] = torch.zeros(n_grps)
                    stage_store["intra_group_consistency_js_count"] = torch.zeros(n_grps)
                    stage_store["feature_group_consistency_js_sum"] = torch.zeros(n_grps)
                    stage_store["feature_group_consistency_js_count"] = torch.zeros(n_grps)
                    stage_store["n_groups"] = n_grps
                self._accumulate_node_probs(
                    stage_store,
                    node_name="final.group",
                    node_kind="final",
                    route_space="group",
                    probs=gw,
                    mask=mask,
                    aggregation_level=aggregation_level,
                    wrapper_name=wrapper_name,
                    neigh_idx=neigh_idx,
                    neigh_k=neigh_k,
                )
                gvalid = mask.unsqueeze(-1).float()
                stage_store["group_usage_sum"] += (gw * gvalid).sum(dim=(0, 1)).cpu()
                g_entropy = -(gw.clamp(min=1e-8) * gw.clamp(min=1e-8).log()).sum(dim=-1)
                stage_store["group_entropy_sum"] += float((g_entropy * mask.float()).sum().item())
                g_top1 = gw.argmax(dim=-1)
                flat_gtop1 = g_top1[mask]
                stage_store["group_top1_count"] += torch.bincount(flat_gtop1.cpu(), minlength=n_grps).float()
                stage_store["group_n_tokens"] += n_valid

                if neigh_idx is not None and neigh_k > 0 and int(weights.size(0)) > 1:
                    group_session_prob = (gw * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                    group_session_prob = group_session_prob.clamp(min=1e-8)
                    g_pi = group_session_prob.unsqueeze(1).expand(-1, neigh_k, -1)
                    g_pj = group_session_prob.index_select(0, neigh_idx.reshape(-1)).reshape(int(weights.size(0)), neigh_k, -1).clamp(min=1e-8)
                    g_mix = 0.5 * (g_pi + g_pj)
                    g_js = 0.5 * (
                        (g_pi * (g_pi.log() - g_mix.log())).sum(dim=-1)
                        + (g_pj * (g_pj.log() - g_mix.log())).sum(dim=-1)
                    )
                    stage_store["group_consistency_js_sum"] += float(g_js.sum().item())
                    stage_store["group_consistency_js_count"] += int(g_js.numel())

                    expert_group_idx = dict(router_aux.get("expert_group_idx", {}) or {}).get(stage_key)
                    if torch.is_tensor(expert_group_idx) and expert_group_idx.numel() >= int(weights.size(-1)):
                        eg = expert_group_idx[: int(weights.size(-1))].to(device=weights.device, dtype=torch.long)
                        for grp_idx in range(int(n_grps)):
                            grp_members = (eg == grp_idx).nonzero(as_tuple=False).view(-1)
                            if grp_members.numel() <= 1:
                                continue
                            grp_weight = w.index_select(-1, grp_members)
                            grp_session_prob = (grp_weight * mask.unsqueeze(-1).float()).sum(dim=1)
                            grp_session_prob = grp_session_prob / grp_session_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                            grp_session_prob = grp_session_prob.clamp(min=1e-8)
                            gi = grp_session_prob.unsqueeze(1).expand(-1, neigh_k, -1)
                            gj = grp_session_prob.index_select(0, neigh_idx.reshape(-1)).reshape(int(weights.size(0)), neigh_k, -1).clamp(min=1e-8)
                            gmix = 0.5 * (gi + gj)
                            gjs = 0.5 * (
                                (gi * (gi.log() - gmix.log())).sum(dim=-1)
                                + (gj * (gj.log() - gmix.log())).sum(dim=-1)
                            )
                            stage_store["intra_group_consistency_js_sum"][grp_idx] += float(gjs.sum().item())
                            stage_store["intra_group_consistency_js_count"][grp_idx] += float(gjs.numel())

                expert_group_idx = dict(router_aux.get("expert_group_idx", {}) or {}).get(stage_key)
                if torch.is_tensor(expert_group_idx) and expert_group_idx.numel() >= int(weights.size(-1)):
                    eg = expert_group_idx[: int(weights.size(-1))].to(device=weights.device, dtype=torch.long)
                    group_names = list(stage_store.get("family_names", []) or [])
                    for grp_idx in range(int(n_grps)):
                        grp_members = (eg == grp_idx).nonzero(as_tuple=False).view(-1)
                        if grp_members.numel() <= 0:
                            continue
                        grp_weight = w.index_select(-1, grp_members)
                        grp_mass = grp_weight.sum(dim=-1, keepdim=True)
                        grp_mask = mask & (grp_mass.squeeze(-1) > 1e-12)
                        grp_intra = grp_weight / grp_mass.clamp(min=1e-8)
                        group_name = (
                            str(group_names[grp_idx])
                            if grp_idx < len(group_names)
                            else f"group_{grp_idx}"
                        )
                        self._accumulate_node_probs(
                            stage_store,
                            node_name=f"final.intra.{group_name}",
                            node_kind="final",
                            route_space="intra",
                            probs=grp_intra,
                            mask=grp_mask,
                            aggregation_level=aggregation_level,
                            wrapper_name=wrapper_name,
                            neigh_idx=neigh_idx,
                            neigh_k=neigh_k,
                        )

            # Feature-group-specific KNN consistency:
            # neighbors are built per family feature subset, but JS is measured on full expert routing.
            family_map = self.stage_family_features.get(_base_stage_name(stage_key), {})
            if feat is not None and family_map:
                raw_feat = feat.to(device=weights.device)
                if neigh_k > 0 and int(weights.size(0)) > 1:
                    if session_prob is None:
                        session_prob = (w * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                        session_prob = session_prob.clamp(min=1e-8)
                    for fam_idx, family_name in enumerate(stage_store["family_names"]):
                        cols = [self.col2idx[col] for col in family_map.get(family_name, []) if col in self.col2idx]
                        if not cols:
                            continue
                        fam_idx_tensor = torch.tensor(cols, device=raw_feat.device, dtype=torch.long)
                        fam_feat = raw_feat.index_select(-1, fam_idx_tensor).mean(dim=1)
                        if fam_feat.ndim != 2 or int(fam_feat.size(0)) <= 1:
                            continue
                        fam_norm = F.normalize(fam_feat, p=2, dim=-1)
                        fam_sim = fam_norm @ fam_norm.transpose(0, 1)
                        fam_sim.fill_diagonal_(-1e9)
                        fam_neigh_idx = fam_sim.topk(k=neigh_k, dim=1).indices
                        fi = session_prob.unsqueeze(1).expand(-1, neigh_k, -1)
                        fj = session_prob.index_select(0, fam_neigh_idx.reshape(-1)).reshape(int(weights.size(0)), neigh_k, -1).clamp(min=1e-8)
                        fmix = 0.5 * (fi + fj)
                        fjs = 0.5 * (
                            (fi * (fi.log() - fmix.log())).sum(dim=-1)
                            + (fj * (fj.log() - fmix.log())).sum(dim=-1)
                        )
                        if fam_idx < int(stage_store["feature_group_consistency_js_sum"].numel()):
                            stage_store["feature_group_consistency_js_sum"][fam_idx] += float(fjs.sum().item())
                            stage_store["feature_group_consistency_js_count"][fam_idx] += float(fjs.numel())

            # group_prior: feature-derived expected group distribution
            group_prior = dict(router_aux.get("group_prior", {}) or {}).get(stage_key)
            if torch.is_tensor(group_prior) and group_prior.size(-1) >= 1:
                gp = group_prior.detach()
                gvalid = mask.unsqueeze(-1).float()
                stage_store["group_prior_sum"] += (gp * gvalid).sum(dim=(0, 1)).cpu()
                stage_store["group_prior_n"] += n_valid

            # factored_group_logits: raw logits from group_router_net in factored router
            fgl = dict(router_aux.get("factored_group_logits", {}) or {}).get(stage_key)
            if torch.is_tensor(fgl) and fgl.size(-1) >= 1:
                fg_probs = F.softmax(fgl.detach(), dim=-1)
                # Entropy of factored group logits distribution
                fg_entropy = -(fg_probs.clamp(min=1e-8) * fg_probs.clamp(min=1e-8).log()).sum(dim=-1)
                # fgl is (B, S, n_groups) for token-granularity or (B, n_groups) for session
                if fgl.ndim == 3:
                    fgl_mask = mask.float()
                    stage_store["factored_group_entropy_sum"] += float((fg_entropy * fgl_mask).sum().item())
                    stage_store["factored_group_n_tokens"] += n_valid
                elif fgl.ndim == 2:
                    stage_store["factored_group_entropy_sum"] += float(fg_entropy.sum().item())
                    stage_store["factored_group_n_tokens"] += int(fg_entropy.numel())

            wrapper_group_probs = dict(router_aux.get("group_probs", {}) or {}).get(stage_key)
            if torch.is_tensor(wrapper_group_probs) and wrapper_group_probs.ndim == 3:
                self._accumulate_node_probs(
                    stage_store,
                    node_name="wrapper.group",
                    node_kind="wrapper",
                    route_space="group",
                    probs=wrapper_group_probs.detach(),
                    mask=mask,
                    aggregation_level=aggregation_level,
                    wrapper_name=wrapper_name,
                    neigh_idx=neigh_idx,
                    neigh_k=neigh_k,
                )

            wrapper_intra_probs = dict(router_aux.get("intra_probs", {}) or {}).get(stage_key)
            if torch.is_tensor(wrapper_intra_probs) and wrapper_intra_probs.ndim == 4:
                group_names = list(stage_store.get("family_names", []) or [])
                for grp_idx in range(int(wrapper_intra_probs.size(-2))):
                    group_name = (
                        str(group_names[grp_idx])
                        if grp_idx < len(group_names)
                        else f"group_{grp_idx}"
                    )
                    self._accumulate_node_probs(
                        stage_store,
                        node_name=f"wrapper.intra.{group_name}",
                        node_kind="wrapper",
                        route_space="intra",
                        probs=wrapper_intra_probs[..., grp_idx, :].detach(),
                        mask=mask,
                        aggregation_level=aggregation_level,
                        wrapper_name=wrapper_name,
                        neigh_idx=neigh_idx,
                        neigh_k=neigh_k,
                    )

            primitive_outputs = dict(router_aux.get("primitive_outputs", {}) or {}).get(stage_key)
            if isinstance(primitive_outputs, dict):
                group_names = list(stage_store.get("family_names", []) or [])
                for primitive_name, payload in primitive_outputs.items():
                    if not isinstance(payload, dict):
                        continue
                    probs = payload.get("probs")
                    source_type = str(payload.get("source", "") or "")
                    temperature = payload.get("temperature")
                    top_k = payload.get("top_k")
                    if torch.is_tensor(probs) and probs.ndim == 3:
                        if primitive_name in {"a_joint"}:
                            route_space = "expert"
                        elif primitive_name in {"b_group", "e_scalar"}:
                            route_space = "group"
                        else:
                            route_space = "intra"
                        self._accumulate_node_probs(
                            stage_store,
                            node_name=f"primitive.{primitive_name}",
                            node_kind="primitive",
                            route_space=route_space,
                            probs=probs.detach(),
                            mask=mask,
                            aggregation_level=aggregation_level,
                            wrapper_name=wrapper_name,
                            neigh_idx=neigh_idx,
                            neigh_k=neigh_k,
                            source_type=source_type,
                            temperature=temperature if temperature is None else float(temperature),
                            top_k=None if top_k is None else int(top_k),
                        )
                    elif torch.is_tensor(probs) and probs.ndim == 4:
                        for grp_idx in range(int(probs.size(-2))):
                            group_name = (
                                str(group_names[grp_idx])
                                if grp_idx < len(group_names)
                                else f"group_{grp_idx}"
                            )
                            self._accumulate_node_probs(
                                stage_store,
                                node_name=f"primitive.{primitive_name}.{group_name}",
                                node_kind="primitive",
                                route_space="intra",
                                probs=probs[..., grp_idx, :].detach(),
                                mask=mask,
                                aggregation_level=aggregation_level,
                                wrapper_name=wrapper_name,
                                neigh_idx=neigh_idx,
                                neigh_k=neigh_k,
                                source_type=source_type,
                                temperature=temperature if temperature is None else float(temperature),
                                top_k=None if top_k is None else int(top_k),
                            )

        for stage_key, cond in dense_aux.items():
            stage_store = self._ensure_stage(stage_key, n_slots=1, mode="dense")
            delta_norm = cond.get("delta_norm")
            cond_norm = cond.get("condition_norm")
            if torch.is_tensor(delta_norm):
                stage_store["delta_norm_sum"] += float(delta_norm.detach().sum().item())
                stage_store["delta_norm_count"] += int(delta_norm.numel())
            if torch.is_tensor(cond_norm):
                stage_store["cond_norm_sum"] += float(cond_norm.detach().sum().item())
                stage_store["cond_norm_count"] += int(cond_norm.numel())
            shared_delta_norm = cond.get("shared_delta_norm")
            if torch.is_tensor(shared_delta_norm):
                stage_store["shared_delta_norm_sum"] += float(shared_delta_norm.detach().sum().item())
                stage_store["shared_delta_norm_count"] += int(shared_delta_norm.numel())
            moe_delta_norm = cond.get("moe_delta_norm")
            if torch.is_tensor(moe_delta_norm):
                stage_store["moe_delta_norm_sum"] += float(moe_delta_norm.detach().sum().item())
                stage_store["moe_delta_norm_count"] += int(moe_delta_norm.numel())
            residual_delta_norm = cond.get("residual_delta_norm")
            if torch.is_tensor(residual_delta_norm):
                stage_store["residual_delta_norm_sum"] += float(residual_delta_norm.detach().sum().item())
                stage_store["residual_delta_norm_count"] += int(residual_delta_norm.numel())
            alpha_value = cond.get("alpha_value")
            if torch.is_tensor(alpha_value):
                stage_store["alpha_value_sum"] += float(alpha_value.detach().sum().item())
                stage_store["alpha_value_count"] += int(alpha_value.numel())
            alpha_effective = cond.get("alpha_effective")
            if torch.is_tensor(alpha_effective):
                stage_store["alpha_effective_sum"] += float(alpha_effective.detach().sum().item())
                stage_store["alpha_effective_count"] += int(alpha_effective.numel())

    @staticmethod
    def _usage_scalars(usage_sum: torch.Tensor, top1_count: torch.Tensor) -> dict:
        if usage_sum.numel() <= 0 or float(usage_sum.sum().item()) <= 0:
            return {
                "usage_share": [],
                "n_eff": 0.0,
                "cv_usage": 0.0,
                "dead_expert_frac": 0.0,
                "top1_max_frac": 0.0,
            }
        usage_share = usage_sum / usage_sum.sum().clamp(min=1e-8)
        usage_mean = float(usage_share.mean().item())
        usage_std = float(usage_share.std(unbiased=False).item())
        top1_total = float(top1_count.sum().item())
        top1_max_frac = float((top1_count.max().item() / top1_total) if top1_total > 0 else 0.0)
        return {
            "usage_share": _float_list(usage_share),
            "n_eff": float((1.0 / usage_share.pow(2).sum().item()) if usage_share.numel() > 0 else 0.0),
            "cv_usage": float(usage_std / max(usage_mean, 1e-8)),
            "dead_expert_frac": float((usage_sum <= 1e-8).float().mean().item()),
            "top1_max_frac": top1_max_frac,
        }

    @staticmethod
    def _expert_similarity(expert_out_sum: torch.Tensor) -> dict:
        if expert_out_sum.ndim != 2 or expert_out_sum.size(0) <= 1 or expert_out_sum.size(1) <= 1:
            return {"expert_similarity_mean": 0.0, "expert_similarity_max": 0.0}
        normed = F.normalize(expert_out_sum, dim=-1)
        sim = normed @ normed.transpose(0, 1)
        off = sim[~torch.eye(sim.size(0), dtype=torch.bool)]
        if off.numel() <= 0:
            return {"expert_similarity_mean": 0.0, "expert_similarity_max": 0.0}
        return {
            "expert_similarity_mean": float(off.mean().item()),
            "expert_similarity_max": float(off.max().item()),
        }

    def finalize(self) -> dict:
        stage_payload = {}
        flat_scalars = {}
        diagnostic_nodes: List[dict] = []
        tier_a_final_rows: List[dict] = []
        tier_b_internal_rows: List[dict] = []
        for stage_key, raw in self._stage_data.items():
            usage_scalars = self._usage_scalars(raw["usage_sum"], raw["top1_count"])
            n_tokens = max(int(raw["n_tokens"]), 1)
            entropy_mean = float(raw["entropy_sum"] / n_tokens) if raw["mode"] == "moe" else 0.0
            jitter_adj = float(raw["jitter_adj_sum"] / max(raw["jitter_adj_count"], 1))
            jitter_session = float(raw["jitter_session_sum"] / max(raw["n_sessions"], 1))
            condition_norm = float(raw["cond_norm_sum"] / max(raw["cond_norm_count"], 1))
            delta_norm = float(raw["delta_norm_sum"] / max(raw["delta_norm_count"], 1))
            shared_delta_norm = float(raw["shared_delta_norm_sum"] / max(raw["shared_delta_norm_count"], 1))
            moe_delta_norm = float(raw["moe_delta_norm_sum"] / max(raw["moe_delta_norm_count"], 1))
            residual_delta_norm = float(raw["residual_delta_norm_sum"] / max(raw["residual_delta_norm_count"], 1))
            alpha_value = float(raw["alpha_value_sum"] / max(raw["alpha_value_count"], 1))
            alpha_effective = float(raw["alpha_effective_sum"] / max(raw["alpha_effective_count"], 1))
            sim_payload = self._expert_similarity(raw["expert_out_sum"])
            consistency_js = float(raw["consistency_js_sum"] / max(raw["consistency_js_count"], 1))
            consistency_score = float(math.exp(-consistency_js))
            group_consistency_js = float(raw["group_consistency_js_sum"] / max(raw["group_consistency_js_count"], 1))
            group_consistency_score = float(math.exp(-group_consistency_js))
            group_names = list(raw.get("family_names", []) or [])
            n_groups = int(raw.get("n_groups", len(group_names) or 0))
            if len(group_names) < n_groups:
                group_names = group_names + [f"group_{i}" for i in range(len(group_names), n_groups)]
            intra_js_sum = raw.get("intra_group_consistency_js_sum")
            intra_js_count = raw.get("intra_group_consistency_js_count")
            intra_js_by_group: List[float] = []
            intra_score_by_group: List[float] = []
            if torch.is_tensor(intra_js_sum) and torch.is_tensor(intra_js_count):
                limit = min(int(intra_js_sum.numel()), int(intra_js_count.numel()), max(n_groups, 0))
                for idx in range(limit):
                    denom = float(max(intra_js_count[idx].item(), 1.0))
                    js_val = float(intra_js_sum[idx].item() / denom)
                    intra_js_by_group.append(js_val)
                    intra_score_by_group.append(float(math.exp(-js_val)))
            if not intra_js_by_group and n_groups > 0:
                intra_js_by_group = [0.0] * n_groups
                intra_score_by_group = [1.0] * n_groups
            intra_mean_js = float(sum(intra_js_by_group) / max(len(intra_js_by_group), 1))
            intra_mean_score = float(sum(intra_score_by_group) / max(len(intra_score_by_group), 1))
            feature_js_sum = raw.get("feature_group_consistency_js_sum")
            feature_js_count = raw.get("feature_group_consistency_js_count")
            feature_js_by_group: List[float] = []
            feature_score_by_group: List[float] = []
            if torch.is_tensor(feature_js_sum) and torch.is_tensor(feature_js_count):
                limit = min(int(feature_js_sum.numel()), int(feature_js_count.numel()), max(n_groups, 0))
                for idx in range(limit):
                    denom = float(max(feature_js_count[idx].item(), 1.0))
                    js_val = float(feature_js_sum[idx].item() / denom)
                    feature_js_by_group.append(js_val)
                    feature_score_by_group.append(float(math.exp(-js_val)))
            if not feature_js_by_group and n_groups > 0:
                feature_js_by_group = [0.0] * n_groups
                feature_score_by_group = [1.0] * n_groups
            feature_mean_js = float(sum(feature_js_by_group) / max(len(feature_js_by_group), 1))
            feature_mean_score = float(sum(feature_score_by_group) / max(len(feature_score_by_group), 1))
            pair_count = raw.get("pair_count")
            pair_feat_sum = raw.get("pair_feat_sum")
            pair_route_sum = raw.get("pair_route_sum")
            pair_js_sum = raw.get("pair_js_sum")
            pair_bins: List[dict] = []
            if (
                torch.is_tensor(pair_count)
                and torch.is_tensor(pair_feat_sum)
                and torch.is_tensor(pair_route_sum)
                and torch.is_tensor(pair_js_sum)
            ):
                n_bins = min(
                    int(pair_count.numel()),
                    int(pair_feat_sum.numel()),
                    int(pair_route_sum.numel()),
                    int(pair_js_sum.numel()),
                    int(self.pair_bin_count),
                )
                for idx in range(n_bins):
                    cnt = float(pair_count[idx].item())
                    if cnt <= 0:
                        continue
                    left = -1.0 + (2.0 * idx / float(max(n_bins, 1)))
                    right = -1.0 + (2.0 * (idx + 1) / float(max(n_bins, 1)))
                    pair_bins.append(
                        {
                            "bin_index": int(idx),
                            "left": float(left),
                            "right": float(right),
                            "count": float(cnt),
                            "feature_cosine_mean": float(pair_feat_sum[idx].item() / cnt),
                            "routing_similarity_mean": float(pair_route_sum[idx].item() / cnt),
                            "routing_js_mean": float(pair_js_sum[idx].item() / cnt),
                        }
                    )
            pair_samples = list(raw.get("pair_sample_points", []) or [])
            wrapper_name = self._dominant_name(dict(raw.get("wrapper_name_hist", {}) or {}), default="")
            family_payload = {
                "mean_top_expert_share": 0.0,
                "family_top_expert": [],
            }
            fam = raw.get("family_expert")
            if torch.is_tensor(fam) and fam.ndim == 2 and fam.numel() > 0:
                row_sum = fam.sum(dim=1, keepdim=True).clamp(min=1e-8)
                fam_share = fam / row_sum
                top_share, top_idx = fam_share.max(dim=1)
                fam_names = list(raw.get("family_names", []) or [])
                exp_names = list(self.stage_expert_names.get(_base_stage_name(stage_key), []))
                entries = []
                for i in range(fam_share.size(0)):
                    fam_name = fam_names[i] if i < len(fam_names) else f"family_{i}"
                    exp_i = int(top_idx[i].item())
                    exp_name = exp_names[exp_i] if exp_i < len(exp_names) else f"expert_{exp_i}"
                    entries.append(
                        {
                            "family": str(fam_name),
                            "expert": str(exp_name),
                            "top_share": float(top_share[i].item()),
                        }
                    )
                family_payload = {
                    "mean_top_expert_share": float(top_share.mean().item()),
                    "family_top_expert": entries,
                }
            stage_payload[stage_key] = {
                "mode": raw["mode"],
                "n_slots": raw["n_slots"],
                "expert_names": list(self.stage_expert_names.get(_base_stage_name(stage_key), [])),
                "family_names": list(raw["family_names"]),
                "usage_sum": _float_list(raw["usage_sum"]),
                "top1_count": _float_list(raw["top1_count"]),
                "entropy_mean": entropy_mean,
                "route_jitter_adjacent": jitter_adj,
                "route_jitter_session": jitter_session,
                "route_consistency_knn_js": consistency_js,
                "route_consistency_knn_score": consistency_score,
                "route_consistency_group_knn_js": group_consistency_js,
                "route_consistency_group_knn_score": group_consistency_score,
                "route_consistency_intra_group_knn": {
                    "group_names": group_names[: len(intra_js_by_group)],
                    "js_by_group": intra_js_by_group,
                    "score_by_group": intra_score_by_group,
                    "mean_js": intra_mean_js,
                    "mean_score": intra_mean_score,
                },
                "route_consistency_feature_group_knn": {
                    "group_names": group_names[: len(feature_js_by_group)],
                    "js_by_group": feature_js_by_group,
                    "score_by_group": feature_score_by_group,
                    "mean_js": feature_mean_js,
                    "mean_score": feature_mean_score,
                },
                "feature_route_pair_similarity": {
                    "bin_stats": pair_bins,
                    "sample_points": pair_samples,
                    "sample_count": int(len(pair_samples)),
                    "bin_count": int(self.pair_bin_count),
                },
                "condition_norm": condition_norm,
                "stage_delta_norm": delta_norm,
                "shared_delta_norm": shared_delta_norm,
                "moe_delta_norm": moe_delta_norm,
                "residual_delta_norm": residual_delta_norm,
                "alpha_value": alpha_value,
                "alpha_effective": alpha_effective,
                "usage_share": usage_scalars["usage_share"],
                "n_eff": usage_scalars["n_eff"],
                "cv_usage": usage_scalars["cv_usage"],
                "dead_expert_frac": usage_scalars["dead_expert_frac"],
                "top1_max_frac": usage_scalars["top1_max_frac"],
                "wrapper_name": wrapper_name,
                "feature_family_expert_heatmap": {
                    "family_names": list(raw["family_names"]),
                    "values": raw["family_expert"].tolist(),
                },
                "position_expert_usage": {
                    "values": raw["position_usage"].tolist(),
                },
                "route_transition_matrix": {
                    "values": raw["transition"].tolist(),
                },
                # Group-level routing metrics
                "group_routing": _build_group_routing_payload(raw),
                "specialization_summary": family_payload,
                **sim_payload,
            }
            prefix = stage_key.replace("@", "_")
            flat_scalars[f"{prefix}.entropy_mean"] = entropy_mean
            flat_scalars[f"{prefix}.n_eff"] = usage_scalars["n_eff"]
            flat_scalars[f"{prefix}.cv_usage"] = usage_scalars["cv_usage"]
            flat_scalars[f"{prefix}.dead_expert_frac"] = usage_scalars["dead_expert_frac"]
            flat_scalars[f"{prefix}.top1_max_frac"] = usage_scalars["top1_max_frac"]
            flat_scalars[f"{prefix}.route_jitter_adjacent"] = jitter_adj
            flat_scalars[f"{prefix}.route_jitter_session"] = jitter_session
            flat_scalars[f"{prefix}.route_consistency_knn_js"] = consistency_js
            flat_scalars[f"{prefix}.route_consistency_knn_score"] = consistency_score
            flat_scalars[f"{prefix}.route_consistency_group_knn_js"] = group_consistency_js
            flat_scalars[f"{prefix}.route_consistency_group_knn_score"] = group_consistency_score
            flat_scalars[f"{prefix}.route_consistency_intra_group_knn_mean_js"] = intra_mean_js
            flat_scalars[f"{prefix}.route_consistency_intra_group_knn_mean_score"] = intra_mean_score
            flat_scalars[f"{prefix}.route_consistency_feature_group_knn_mean_js"] = feature_mean_js
            flat_scalars[f"{prefix}.route_consistency_feature_group_knn_mean_score"] = feature_mean_score
            for idx, name in enumerate(group_names[: len(feature_js_by_group)]):
                token = _metric_name_token(name)
                flat_scalars[f"{prefix}.route_consistency_feature_group_knn_{token}_js"] = float(feature_js_by_group[idx])
                flat_scalars[f"{prefix}.route_consistency_feature_group_knn_{token}_score"] = float(feature_score_by_group[idx])
            flat_scalars[f"{prefix}.condition_norm"] = condition_norm
            flat_scalars[f"{prefix}.stage_delta_norm"] = delta_norm
            flat_scalars[f"{prefix}.shared_delta_norm"] = shared_delta_norm
            flat_scalars[f"{prefix}.moe_delta_norm"] = moe_delta_norm
            flat_scalars[f"{prefix}.residual_delta_norm"] = residual_delta_norm
            flat_scalars[f"{prefix}.alpha_value"] = alpha_value
            flat_scalars[f"{prefix}.alpha_effective"] = alpha_effective
            flat_scalars[f"{prefix}.expert_similarity_mean"] = sim_payload["expert_similarity_mean"]
            flat_scalars[f"{prefix}.expert_similarity_max"] = sim_payload["expert_similarity_max"]
            grp_payload = stage_payload[stage_key].get("group_routing", {})
            if grp_payload:
                flat_scalars[f"{prefix}.group_n_eff"] = float(grp_payload.get("group_n_eff", 0.0))
                flat_scalars[f"{prefix}.group_cv_usage"] = float(grp_payload.get("group_cv_usage", 0.0))
                flat_scalars[f"{prefix}.group_top1_max_frac"] = float(grp_payload.get("group_top1_max_frac", 0.0))
                flat_scalars[f"{prefix}.group_entropy_mean"] = float(grp_payload.get("group_entropy_mean", 0.0))
                flat_scalars[f"{prefix}.factored_group_entropy_mean"] = float(grp_payload.get("factored_group_entropy_mean", 0.0))

            node_acc = dict(raw.get("node_acc", {}) or {})
            for node_name in sorted(node_acc.keys()):
                node = dict(node_acc.get(node_name, {}) or {})
                support_size = int(node.get("support_size", 0) or 0)
                usage_sum = node.get("usage_sum")
                top1_count = node.get("top1_count")
                if not torch.is_tensor(usage_sum) or not torch.is_tensor(top1_count):
                    continue
                node_usage = self._usage_scalars(usage_sum, top1_count)
                node_tokens = max(int(node.get("n_tokens", 0) or 0), 1)
                entropy_mean_node = float(node.get("entropy_sum", 0.0) or 0.0) / node_tokens
                knn_js = float(node.get("knn_js_sum", 0.0) or 0.0) / max(int(node.get("knn_js_count", 0) or 0), 1)
                knn_score = _js_to_score(knn_js)
                node_wrapper_name = str(node.get("wrapper_name", "") or wrapper_name)
                row = {
                    "stage_name": _base_stage_name(stage_key),
                    "stage_key": stage_key,
                    "split": self.split_name,
                    "aggregation_level": str(node.get("aggregation_level", raw.get("aggregation_level", "token"))),
                    "node_kind": str(node.get("node_kind", "")),
                    "node_name": str(node.get("node_name", node_name)),
                    "route_space": str(node.get("route_space", "")),
                    "support_size": support_size,
                    "wrapper_name": node_wrapper_name,
                    "source_type": str(node.get("source_type", "")),
                    "temperature": node.get("temperature"),
                    "top_k": node.get("top_k"),
                    "entropy_norm": _entropy_norm(entropy_mean_node, support_size),
                    "n_eff_norm": float(node_usage.get("n_eff", 0.0)) / max(float(support_size), 1.0),
                    "top1_monopoly_norm": _top1_monopoly_norm(float(node_usage.get("top1_max_frac", 0.0)), support_size),
                    "jitter_adj_norm": 0.0,
                    "knn_consistency_score": knn_score,
                    "knn_consistency_js": knn_js,
                    "n_eff": float(node_usage.get("n_eff", 0.0)),
                    "cv_usage": float(node_usage.get("cv_usage", 0.0)),
                    "top1_max_frac": float(node_usage.get("top1_max_frac", 0.0)),
                    "entropy_mean": entropy_mean_node,
                }
                if row["node_name"] == "final.expert":
                    row["jitter_adj_norm"] = jitter_adj
                diagnostic_nodes.append(row)
                if row["node_kind"] == "final":
                    tier_a_final_rows.append(row)
                else:
                    tier_b_internal_rows.append(row)

        readable_stages = []
        for stage_key in sorted(stage_payload.keys()):
            st = stage_payload[stage_key]
            readable_stages.append(
                {
                    "stage": stage_key,
                    "n_eff": float(st.get("n_eff", 0.0)),
                    "cv_usage": float(st.get("cv_usage", 0.0)),
                    "top1_max_frac": float(st.get("top1_max_frac", 0.0)),
                    "entropy_mean": float(st.get("entropy_mean", 0.0)),
                    "jitter_adj": float(st.get("route_jitter_adjacent", 0.0)),
                    "consistency_knn_js": float(st.get("route_consistency_knn_js", 0.0)),
                    "consistency_knn_score": float(st.get("route_consistency_knn_score", 0.0)),
                    "consistency_group_knn_js": float(st.get("route_consistency_group_knn_js", 0.0)),
                    "consistency_group_knn_score": float(st.get("route_consistency_group_knn_score", 0.0)),
                    "consistency_intra_group_knn_mean_score": float(
                        ((st.get("route_consistency_intra_group_knn") or {}).get("mean_score", 0.0))
                    ),
                    "consistency_feature_group_knn_mean_score": float(
                        ((st.get("route_consistency_feature_group_knn") or {}).get("mean_score", 0.0))
                    ),
                    "family_top_share_mean": float(
                        ((st.get("specialization_summary") or {}).get("mean_top_expert_share", 0.0))
                    ),
                }
            )

        return {
            "split": self.split_name,
            "feature_mode": self.feature_mode,
            "stage_metrics": stage_payload,
            "scalar_metrics": flat_scalars,
            "diagnostic_nodes": diagnostic_nodes,
            "diag_tiers": {
                "tier_a_final": tier_a_final_rows,
                "tier_b_internal": tier_b_internal_rows,
                "tier_c_viz": {
                    "viz_feature_pca": [],
                    "viz_router_input_pca": [],
                    "viz_group_feature_pca": [],
                },
            },
            "readable_summary": {
                "stages": readable_stages,
            },
        }
