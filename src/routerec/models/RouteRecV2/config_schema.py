"""Config resolver and validation for RouteRecV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


# Removed/deprecated in v2 (hard error when explicitly set at model scope).
_REMOVED_KEYS = {
    "stage_moe_repeat_after_pre_layer",
    "n_pre_layer",
    "n_pre_macro",
    "n_pre_mid",
    "n_pre_micro",
    "n_post_layer",
    "alpha_warmup_steps",
    "temperature_warmup_steps",
    "moe_top_k_warmup_steps",
    "routerec_schedule_log_every",
}

_GROUP_KEYS = (
    "model_core",
    "layout_execution",
    "routing_common",
    "parallel_merge",
    "schedule",
    "loss_regularization",
    "runtime",
)


@dataclass
class ConfigResolver:
    raw: Any

    _MISSING = object()

    def _top_value(self, key: str, default: Any = _MISSING) -> Any:
        if isinstance(self.raw, dict):
            return self.raw.get(key, default)

        try:
            if key in self.raw:
                return self.raw[key]
        except Exception:
            pass

        final_cfg = getattr(self.raw, "final_config_dict", None)
        if isinstance(final_cfg, dict):
            return final_cfg.get(key, default)
        return default

    def _group_value(self, key: str) -> Optional[Any]:
        for group_name in _GROUP_KEYS:
            group = self._top_value(group_name, self._MISSING)
            if isinstance(group, dict) and key in group:
                return group[key]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        top = self._top_value(key, self._MISSING)
        if top is not self._MISSING:
            return top
        grouped = self._group_value(key)
        if grouped is not None:
            return grouped
        return default

    def has(self, key: str) -> bool:
        if self._top_value(key, self._MISSING) is not self._MISSING:
            return True
        return self._group_value(key) is not None

    def require(self, key: str) -> Any:
        if not self.has(key):
            raise ValueError(f"RouteRecV2 requires config key '{key}'")
        return self.get(key)

    def assert_removed_keys(self) -> None:
        found = [k for k in sorted(_REMOVED_KEYS) if self.has(k)]
        if found:
            joined = ", ".join(found)
            raise ValueError(
                "RouteRecV2 does not support removed v1 keys: "
                f"{joined}. Please use routerec_v2_* / layout object schema keys."
            )

    def assert_embedding_only_dimension(self) -> None:
        # hidden_size exists in base config globally; only block explicit mismatch.
        if self.has("hidden_size"):
            h = int(self.get("hidden_size"))
            e = int(self.get("embedding_size", h))
            if h != e:
                raise ValueError(
                    "RouteRecV2 uses embedding_size as the single dimension key. "
                    f"hidden_size({h}) must match embedding_size({e}) or be removed from overrides."
                )


def parse_layout_catalog_from_config(resolver: ConfigResolver) -> Iterable[Dict[str, Any]]:
    catalog = resolver.get("routerec_v2_layout_catalog", None)
    if catalog is None:
        # Default fallback catalog (single serial layout).
        return [
            {
                "id": "L0",
                "execution": resolver.get("routerec_stage_execution_mode", "serial"),
                "global_pre_layers": 1,
                "global_post_layers": 0,
                "stages": {
                    "macro": {"pass_layers": 1, "moe_blocks": 1},
                    "mid": {"pass_layers": 1, "moe_blocks": 1},
                    "micro": {"pass_layers": 1, "moe_blocks": 1},
                },
            }
        ]
    return catalog
