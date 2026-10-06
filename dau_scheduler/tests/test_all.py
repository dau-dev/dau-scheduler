"""The package exports what its README and docs say it does."""

import dau_scheduler


def test_the_documented_surface_is_exported() -> None:
    for name in (
        "ADDITIVE",
        "MergeRule",
        "ProfileCache",
        "QueryProfile",
        "LinkRate",
        "RowBatch",
        "RowRange",
        "SplitError",
        "PostureError",
        "WorkSplit",
        "execute_split",
        "merge_totals",
        "plan_split",
        "profile_from_spans",
        "host_identity",
    ):
        assert name in dau_scheduler.__all__, name
        assert hasattr(dau_scheduler, name), name
