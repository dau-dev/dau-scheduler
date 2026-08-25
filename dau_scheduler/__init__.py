__version__ = "0.1.0"

from .errors import PostureError, SplitError
from .merge import (
    ADDITIVE,
    AGGREGATE_REFUSALS,
    BOUNDARY_STATS,
    DEFERRED_MEAN,
    MergeRule,
    check_split,
    merge_keyed_groups,
    merge_ordered_groups,
    merge_sorted_runs,
    merge_top_k,
    merge_totals,
)
from .posture import POSTURES, HostCost, Posture, WorkSplit, batch_count, plan_split
from .profile import NodeCost, ProfileCache, QueryProfile, host_identity, profile_from_spans
from .rows import RowBatch, RowRange, RowSource, as_row_source
from .split import SplitArm, SplitExecution, execute_split

__all__ = (
    "ADDITIVE",
    "AGGREGATE_REFUSALS",
    "BOUNDARY_STATS",
    "DEFERRED_MEAN",
    "POSTURES",
    "HostCost",
    "MergeRule",
    "NodeCost",
    "Posture",
    "PostureError",
    "ProfileCache",
    "QueryProfile",
    "RowBatch",
    "RowRange",
    "RowSource",
    "SplitArm",
    "SplitError",
    "SplitExecution",
    "WorkSplit",
    "__version__",
    "as_row_source",
    "batch_count",
    "check_split",
    "execute_split",
    "host_identity",
    "merge_keyed_groups",
    "merge_ordered_groups",
    "merge_sorted_runs",
    "merge_top_k",
    "merge_totals",
    "plan_split",
    "profile_from_spans",
)
