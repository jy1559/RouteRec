"""Layout schema for RouteRecV2.

Layout objects explicitly encode non-MoE pass depth and repeated MoE blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


_STAGE_NAMES: Tuple[str, ...] = ("macro", "mid", "micro")
_ALLOWED_EXECUTION = {"serial", "parallel"}
_MAX_TOTAL_ATTN_LAYERS = 6


@dataclass(frozen=True)
class StageLayoutSpec:
    pass_layers: int
    moe_blocks: int


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    execution: str
    global_pre_layers: int
    global_post_layers: int
    stages: Dict[str, StageLayoutSpec]


def _to_non_negative_int(value: Any, field_name: str) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be int >= 0, got {value}") from None
    if iv < 0:
        raise ValueError(f"{field_name} must be int >= 0, got {value}")
    return iv


def _parse_stage(stage_name: str, payload: Any) -> StageLayoutSpec:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"routerec_v2_layout_catalog[*].stages.{stage_name} must be object, got {type(payload).__name__}"
        )
    pass_layers = _to_non_negative_int(payload.get("pass_layers", 0), f"stages.{stage_name}.pass_layers")
    moe_blocks = _to_non_negative_int(payload.get("moe_blocks", 0), f"stages.{stage_name}.moe_blocks")
    return StageLayoutSpec(pass_layers=pass_layers, moe_blocks=moe_blocks)


def parse_layout_catalog(raw_catalog: Any) -> List[LayoutSpec]:
    """Parse and validate v2 layout catalog.

    Expected schema:
      - id: str
      - execution: serial|parallel
      - global_pre_layers: int >= 0
      - global_post_layers: int >= 0
      - stages:
          macro/mid/micro: {pass_layers: int>=0, moe_blocks: int>=0}
    """
    if not isinstance(raw_catalog, Iterable) or isinstance(raw_catalog, (str, bytes, dict)):
        raise ValueError("routerec_v2_layout_catalog must be a non-empty list of layout objects")

    parsed: List[LayoutSpec] = []
    for idx, item in enumerate(raw_catalog):
        if not isinstance(item, dict):
            raise ValueError(
                f"routerec_v2_layout_catalog[{idx}] must be object, got {type(item).__name__}"
            )

        layout_id = str(item.get("id", f"L{idx}")).strip() or f"L{idx}"
        execution = str(item.get("execution", "serial")).strip().lower()
        if execution not in _ALLOWED_EXECUTION:
            raise ValueError(
                f"routerec_v2_layout_catalog[{idx}].execution must be one of {sorted(_ALLOWED_EXECUTION)}, "
                f"got {execution}"
            )

        global_pre_layers = _to_non_negative_int(
            item.get("global_pre_layers", 0), f"routerec_v2_layout_catalog[{idx}].global_pre_layers"
        )
        global_post_layers = _to_non_negative_int(
            item.get("global_post_layers", 0), f"routerec_v2_layout_catalog[{idx}].global_post_layers"
        )

        raw_stages = item.get("stages", {})
        if not isinstance(raw_stages, dict):
            raise ValueError(
                f"routerec_v2_layout_catalog[{idx}].stages must be object, got {type(raw_stages).__name__}"
            )

        stages = {
            stage_name: _parse_stage(stage_name, raw_stages.get(stage_name, {}))
            for stage_name in _STAGE_NAMES
        }

        total_layers = global_pre_layers + global_post_layers
        for stage_name in _STAGE_NAMES:
            stage_spec = stages[stage_name]
            total_layers += stage_spec.pass_layers + stage_spec.moe_blocks
        if total_layers > _MAX_TOTAL_ATTN_LAYERS:
            raise ValueError(
                f"routerec_v2_layout_catalog[{idx}] total_layers={total_layers} exceeds "
                f"limit({_MAX_TOTAL_ATTN_LAYERS})"
            )

        parsed.append(
            LayoutSpec(
                layout_id=layout_id,
                execution=execution,
                global_pre_layers=global_pre_layers,
                global_post_layers=global_post_layers,
                stages=stages,
            )
        )

    if not parsed:
        raise ValueError("routerec_v2_layout_catalog must be non-empty")
    return parsed


def active_stage_names(layout: LayoutSpec) -> List[str]:
    """Return stages that have any stage-local compute in v2."""
    out: List[str] = []
    for stage_name in _STAGE_NAMES:
        spec = layout.stages[stage_name]
        if spec.pass_layers > 0 or spec.moe_blocks > 0:
            out.append(stage_name)
    return out


def total_stage_moe_blocks(layout: LayoutSpec) -> int:
    return sum(layout.stages[s].moe_blocks for s in _STAGE_NAMES)


def stage_boundary_summary(layout: LayoutSpec) -> Dict[str, Dict[str, int]]:
    """Readable boundary map for logs/docs: pass_layers then moe_blocks."""
    return {
        stage: {
            "pass_layers": layout.stages[stage].pass_layers,
            "moe_blocks": layout.stages[stage].moe_blocks,
        }
        for stage in _STAGE_NAMES
    }
