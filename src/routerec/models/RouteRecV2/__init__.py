"""RouteRecV2 package.

Key upgrades over v1:
- Object-based layout schema with explicit pass/MoE boundaries.
- Unified execution modes: serial, parallel, parallel+repeat.
- Optional stage-merge auxiliary loss in parallel mode.
"""

from .routerec_v2 import RouteRecBase_V2

__all__ = ["RouteRecBase_V2"]
