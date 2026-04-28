"""Feature config wrapper for RouteRecV2.

v2 reuses the proven feature stage/bundle definitions from v1.
"""

from ..RouteRecBase.feature_config import (  # noqa: F401
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    feature_list_field,
    build_column_to_index,
    build_expert_indices,
    build_stage_indices,
)
