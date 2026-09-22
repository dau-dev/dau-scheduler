"""The executor: two engines over disjoint row ranges, one answer.

Both arms here are plain Python functions over the same rows, which is the
point of the seam -- the executor never learns what is on either end of it.
Every merged answer is checked against the same query run over the WHOLE
table at several shares, including the degenerate 0.0 and 1.0.
"""

from __future__ import annotations

import threading
import time

import pytest

from dau_scheduler import (
    ADDITIVE,
    MergeRule,
    RowBatch,
    RowRange,
    SplitError,
    WorkSplit,
    execute_split,
    merge_totals,
    plan_split,
)

SHARES = (0.0, 0.13, 0.5, 0.6, 0.87, 1.0)

ROWS = [{"key": (seed * 7) % 5, "value": (seed * 31) % 100, "day": seed % 60} for seed in range(500)]

QUERY = {"filter": lambda row: row["day"] < 30, "outputs": ("total", "len")}


def _split(share: float) -> WorkSplit:
    return WorkSplit(posture="collaborative", device_share=share, rationale="a share under test")


def _rule() -> MergeRule:
    return MergeRule(
        terminal="reductions",
        combine=lambda partials: merge_totals(partials, QUERY["outputs"]),
        mergeable=ADDITIVE,
    )


def _run(query, rows, build):
    """One arm: the whole query over the rows it was given."""
    kept = [row for row in rows.values() if query["filter"](row)]
    return {"total": sum(row["value"] for row in kept), "len": len(kept)}


class _GriddedArm:
    """An arm that can only take row counts on a grid, the way a device whose
    job length must sit on a DMA grid can."""

    def __init__(self, row_granularity: int) -> None:
        self.row_granularity = row_granularity

    def __call__(self, query, rows, build):
        return _run(query, rows, build)


def _execute(rows, split, **kwargs):
    kwargs.setdefault("device", _run)
    kwargs.setdefault("host", _run)
    return execute_split(QUERY, rows, split, _rule(), **kwargs)


def test_the_merged_answer_is_the_unsplit_answer_at_every_share() -> None:
    whole = _run(QUERY, RowBatch(ROWS), None)
    for share in SHARES:
        execution = _execute(ROWS, _split(share))
        assert execution.result == whole, share
        assert execution.device_rows + execution.host_rows == len(ROWS), share


def test_the_arms_split_the_rows_at_the_planned_share() -> None:
    execution = _execute(ROWS, plan_split("collaborative", cpu_seconds=0.02763, device_seconds=0.01860))
    assert execution.device_rows == 299  # the balance point, in rows
    assert execution.host_rows == len(ROWS) - 299
    assert execution.posture == "collaborative"
    assert execution.terminal == "reductions"


def test_the_leading_rows_go_to_the_first_arm() -> None:
    """Part of the contract, not an implementation detail: a grouped first and
    last and a top-k's arrival tie-break are defined by it."""
    seen: dict[str, tuple[int, int]] = {}

    def arm(name):
        def run(query, rows, build):
            seen[name] = (rows.start, len(rows))
            return _run(query, rows, build)

        return run

    _execute(ROWS, _split(0.6), device=arm("device"), host=arm("host"))
    assert seen == {"device": (0, 300), "host": (300, 200)}


def test_a_degenerate_share_launches_one_engine() -> None:
    """At 0.0 and 1.0 there is no second arm to overlap with, and starting one
    over zero rows would be pure per-call cost."""
    launched: list[int] = []

    def counting(query, rows, build):
        launched.append(len(rows))
        return _run(query, rows, build)

    for share in (0.0, 1.0):
        launched.clear()
        execution = _execute(ROWS, _split(share), device=counting, host=counting)
        assert launched == [len(ROWS)], share
        assert execution.overlap_seconds == 0.0, share
    assert _execute(ROWS, _split(0.0)).device_rows == 0
    assert _execute(ROWS, _split(1.0)).host_rows == 0


def test_an_empty_table_still_answers() -> None:
    execution = _execute([], _split(0.6))
    assert execution.device_rows == execution.host_rows == 0
    assert execution.result == {"total": 0, "len": 0}


def test_the_two_arms_are_in_flight_together() -> None:
    """The test a serialized executor cannot pass. Each arm waits for the
    other to arrive, so a split that ran its halves one after the other would
    break the barrier and fail here rather than quietly report a speedup it
    never delivered."""
    both_arrived = threading.Barrier(2, timeout=10)

    def arm(query, rows, build):
        both_arrived.wait()
        time.sleep(0.005)
        return _run(query, rows, build)

    execution = _execute(ROWS, _split(0.5), device=arm, host=arm)
    assert not both_arrived.broken
    assert execution.overlap_seconds > 0
    assert execution.wall_seconds < execution.device_seconds + execution.host_seconds


def test_a_blocking_arm_overlaps_the_other() -> None:
    """The arms overlap when they release the interpreter, which is what an
    arm waiting on a link and an arm inside a native engine both do."""

    def arm(query, rows, build):
        time.sleep(0.05)
        return _run(query, rows, build)

    execution = _execute(ROWS, _split(0.5), device=arm, host=arm)
    assert execution.overlap_seconds > 0.02
    assert execution.wall_seconds < execution.device_seconds + execution.host_seconds


def test_a_row_range_splits_a_table_neither_arm_reads() -> None:
    """A real split's arms each need one number: an engine holding its own
    copy needs a row COUNT, and one holding a resident frame needs an OFFSET.
    Materializing rows in front of them is pure loss, so a range -- which
    cannot be materialized at all -- has to reach them intact."""
    seen: dict[str, tuple[int, int]] = {}

    def resident(name, total):
        def run(query, rows, build):
            seen[name] = (rows.start, len(rows))
            return {"total": total, "len": len(rows)}

        return run

    execution = execute_split(
        QUERY,
        RowRange(0, 6_000_000),
        _split(0.6),
        _rule(),
        device=resident("device", 11),
        host=resident("host", 31),
    )
    assert seen == {"device": (0, 3_600_000), "host": (3_600_000, 2_400_000)}
    assert execution.result == {"total": 42, "len": 6_000_000}
    assert execution.materialize_seconds < 0.01


def test_a_row_range_refuses_the_arm_that_reads_values() -> None:
    """The executor does not paper over the mismatch by materializing the
    table behind the caller's back to keep an arm fed."""
    with pytest.raises(SplitError, match="names its rows by OFFSET and carries no values"):
        _execute(RowRange(0, 500), _split(0.5))


def test_the_callers_own_rows_reach_the_arm_uncopied() -> None:
    """A materialized batch is WRAPPED, not re-tupled -- identity is what pins
    it, because a copy compares equal and fails only ``is``."""
    seen: dict[str, object] = {}

    def arm(query, rows, build):
        seen["values"] = rows.values()
        return _run(query, rows, build)

    _execute(ROWS, _split(0.0), host=arm)
    assert seen["values"] is ROWS


def test_the_accounting_covers_the_marshalling_it_makes_the_caller_pay() -> None:
    """A timing field that excludes the dominant term is worse than no field,
    so the window spans the whole call and the marshalling has a name."""

    def staged():
        # an ITERATOR has to be drained -- both arms read from one source --
        # so this is marshalling the executor really does impose
        time.sleep(0.05)
        yield from ROWS

    execution = _execute(staged(), _split(0.5))
    assert execution.materialize_seconds >= 0.05
    assert execution.wall_seconds >= execution.materialize_seconds + execution.merge_seconds
    assert execution.result == _run(QUERY, RowBatch(ROWS), None)
    # a sequence is wrapped in place, so handing the same rows over is free
    assert _execute(ROWS, _split(0.5)).materialize_seconds < 0.01


def test_a_refusal_arrives_before_either_arm_starts() -> None:
    """A refusal that arrives after the work is a refusal that already cost
    the work."""
    started: list[str] = []

    def arm(query, rows, build):
        started.append("ran")
        return _run(query, rows, build)

    with pytest.raises(SplitError, match="average: a mean is not row-partitionable"):
        _execute(ROWS, _split(0.5), device=arm, host=arm, aggregates=[("average", "mean")])
    assert started == []


def test_a_declared_aggregate_passes_the_gate() -> None:
    execution = _execute(ROWS, _split(0.5), aggregates=[("total", "sum"), ("len", "count")])
    assert execution.result == _run(QUERY, RowBatch(ROWS), None)


def test_a_broadcast_build_side_reaches_both_arms_whole() -> None:
    """The build side of a join is a TABLE, not part of the row range being
    split: halving it too would drop every match whose build row landed in the
    other arm, and the sums would still look like sums."""
    build_rows = [{"key": key} for key in (0, 3)]
    seen: list[int] = []

    def arm(query, rows, build):
        seen.append(len(build.values()))
        kept = [row for row in rows.values() if query["filter"](row) and any(row["key"] == other["key"] for other in build.values())]
        return {"total": sum(row["value"] for row in kept), "len": len(kept)}

    execution = _execute(ROWS, _split(0.5), device=arm, host=arm, build_rows=build_rows)
    assert seen == [len(build_rows), len(build_rows)]
    expected = [row for row in ROWS if row["day"] < 30 and row["key"] in (0, 3)]
    assert execution.result == {"total": sum(row["value"] for row in expected), "len": len(expected)}


def test_a_failed_arm_reports_its_failure_rather_than_a_short_answer() -> None:
    """An arm that could not complete carries a partial that says so, and
    merging it into the other arm's would return a plausible answer missing
    half the rows."""
    rule = MergeRule(
        terminal="reductions",
        combine=lambda partials: merge_totals(partials, ("total",)),
        failed=lambda partial: partial["error"] is not None,
    )

    def failing(query, rows, build):
        return {"total": 0, "error": "capacity exceeded"}

    def working(query, rows, build):
        return {"total": 10, "error": None}

    execution = execute_split(QUERY, ROWS, _split(0.5), rule, device=failing, host=working)
    assert execution.result == {"total": 0, "error": "capacity exceeded"}


def test_the_boundary_snaps_to_the_device_granularity() -> None:
    """A device whose job length must sit on a DMA grid can only take a row
    count that divides it. 0.5975 of 500 rows is 298.75 -> 299, which a
    granularity of 4 snaps to 300; the merged answer is unchanged and the
    count reported is the one the arm was given."""
    whole = _run(QUERY, RowBatch(ROWS), None)
    execution = _execute(ROWS, _split(0.5975), device=_GriddedArm(4))
    assert execution.device_rows == 300
    assert execution.host_rows == 200
    assert execution.device_granularity == 4
    assert execution.result == whole


def test_the_degenerate_shares_survive_a_granularity() -> None:
    """0.0 still launches no device arm, and 1.0 still hands the device
    everything -- when the table is on the grid. 500 rows on a granularity
    of 8 is NOT: the device can take at most 496, the host gets the 4-row
    tail, and the merged answer is still the unsplit one. That is the
    device's limit being respected rather than a share being mis-served."""
    whole = _run(QUERY, RowBatch(ROWS), None)
    on_grid = ROWS[:496]
    assert _execute(on_grid, _split(1.0), device=_GriddedArm(8)).device_rows == 496
    assert _execute(ROWS, _split(0.0), device=_GriddedArm(8)).device_rows == 0

    off_grid = _execute(ROWS, _split(1.0), device=_GriddedArm(8))
    assert off_grid.device_rows == 496 and off_grid.host_rows == 4
    assert off_grid.result == whole

    # the clamp is to the grid, not the table: 501 rows at share 1.0 must not
    # snap UP to 504 and then clamp back to an off-grid 501
    plus_one = ROWS + [{"key": 1, "value": 7, "day": 3}]
    beyond = _execute(plus_one, _split(1.0), device=_GriddedArm(8))
    assert beyond.device_rows == 496 and beyond.host_rows == 5
    assert beyond.result == _run(QUERY, RowBatch(plus_one), None)

    with pytest.raises(ValueError, match="row_granularity must be at least 1"):
        _execute(ROWS, _split(0.5), device=_GriddedArm(0))


def test_an_invalid_granularity_is_refused_before_the_rows_are_consumed() -> None:
    """A one-shot iterable must survive a refused call, or the retry with a
    corrected argument runs over nothing."""
    rows = iter(ROWS)
    with pytest.raises(ValueError, match="row_granularity must be at least 1"):
        _execute(rows, _split(0.5), device=_GriddedArm(0))
    assert next(rows) is ROWS[0], "the refusal consumed nothing"
