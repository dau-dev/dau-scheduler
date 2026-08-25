"""The merge algebra: what combines from two partial answers, and what does not.

Each merge is checked against the same query run over the WHOLE table, which
is the only assertion that means anything here -- a merge that agrees with
itself proves nothing. The refusals are tested as carefully as the successes,
because an aggregate that combines wrongly answers confidently rather than
failing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby

import pytest

from dau_scheduler import (
    ADDITIVE,
    BOUNDARY_STATS,
    DEFERRED_MEAN,
    MergeRule,
    SplitError,
    check_split,
    merge_keyed_groups,
    merge_ordered_groups,
    merge_sorted_runs,
    merge_top_k,
    merge_totals,
)

ROWS = [{"key": (seed * 7) % 5, "value": (seed * 31) % 100} for seed in range(120)]


@dataclass(frozen=True)
class Group:
    key: int
    total: int
    count: int


@dataclass(frozen=True)
class Stats:
    key: int
    first: int
    last: int
    minimum: int
    maximum: int
    count: int


def _totals(rows):
    return {"total": sum(row["value"] for row in rows), "len": len(rows)}


def _groups(rows):
    """Per-key records from an ordered engine: a group CLOSES when the key
    changes, which is what makes the boundary pair the only shared group."""
    return [Group(key=key, total=sum(row["value"] for row in members), count=len(members)) for key, members in _runs(rows)]


def _stats(rows):
    return [
        Stats(
            key=key,
            first=members[0]["value"],
            last=members[-1]["value"],
            minimum=min(row["value"] for row in members),
            maximum=max(row["value"] for row in members),
            count=len(members),
        )
        for key, members in _runs(rows)
    ]


def _runs(rows):
    return [(key, list(members)) for key, members in groupby(rows, key=lambda row: row["key"])]


def _fold_groups(earlier: Group, later: Group) -> Group:
    return replace(earlier, total=earlier.total + later.total, count=earlier.count + later.count)


def _fold_stats(earlier: Stats, later: Stats) -> Stats:
    return Stats(
        key=earlier.key,
        first=earlier.first,
        last=later.last,
        minimum=min(earlier.minimum, later.minimum),
        maximum=max(earlier.maximum, later.maximum),
        count=earlier.count + later.count,
    )


def _halves(rows):
    boundary = len(rows) // 2
    return rows[:boundary], rows[boundary:]


def test_sums_and_counts_add() -> None:
    device, host = _halves(ROWS)
    merged = merge_totals([_totals(device), _totals(host)], ("total", "len"))
    assert merged == _totals(ROWS)


def test_a_derived_output_is_recomputed_from_the_merged_totals() -> None:
    """An output that is an EXPRESSION over the aggregates is not an aggregate:
    combining the arms' ratios would produce a number that is not a ratio of
    anything, and it would look exactly like a total."""

    def with_average(totals):
        return {**totals, "average": totals["total"] / totals["len"]}

    # an UNEVEN share over a trending column, so the arms' own averages differ
    # from each other and from the whole table's -- an even split of a
    # symmetric fixture would let a wrong merge pass
    rows = [{"key": seed % 5, "value": seed} for seed in range(120)]
    device, host = rows[:24], rows[24:]
    partials = [with_average(_totals(device)), with_average(_totals(host))]
    merged = merge_totals(partials, ("total", "len"), finalize=with_average)
    assert merged == with_average(_totals(rows))
    # the wrong answer this guards: the arms' averages added, or halved
    assert merged["average"] != pytest.approx(sum(partial["average"] for partial in partials))
    assert merged["average"] != pytest.approx(sum(partial["average"] for partial in partials) / 2)


def test_a_mean_survives_only_as_a_sum_and_a_count() -> None:
    """The remedy the refusal names, tested: a mean asked for as an expression
    over two additive outputs splits exactly, while the mean of the arms'
    means does not."""
    rows = [{"key": seed % 5, "value": seed} for seed in range(120)]
    device, host = rows[:24], rows[24:]
    merged = merge_totals([_totals(device), _totals(host)], ("total", "len"))
    assert merged["total"] / merged["len"] == pytest.approx(sum(row["value"] for row in rows) / len(rows))
    arm_means = [_totals(arm)["total"] / _totals(arm)["len"] for arm in (device, host)]
    assert sum(arm_means) / 2 != pytest.approx(merged["total"] / merged["len"])


def test_ordered_groups_fold_the_group_that_straddles_the_boundary() -> None:
    """The hazard an ordered engine makes concrete: the group the boundary
    lands inside closes in BOTH arms, and the merge has to fold those two
    partial records rather than emit the key twice."""
    rows = sorted(ROWS, key=lambda row: row["key"])
    device, host = _halves(rows)
    assert device[-1]["key"] == host[0]["key"]  # or this test proves nothing
    merged = merge_ordered_groups([_groups(device), _groups(host)], key=lambda group: group.key, fold=_fold_groups)
    assert merged == _groups(rows)
    assert len({group.key for group in merged}) == len(merged)


def test_boundary_stats_come_from_the_ranges_that_own_them() -> None:
    """``first`` and ``last`` are defined by position, which is why the arms'
    order is part of the contract: the straddling group opens in the earlier
    range and closes in the later one."""
    rows = sorted(ROWS, key=lambda row: row["key"])
    device, host = _halves(rows)
    merged = merge_ordered_groups([_stats(device), _stats(host)], key=lambda group: group.key, fold=_fold_stats)
    assert merged == _stats(rows)
    straddler = next(group for group in merged if group.key == host[0]["key"])
    assert straddler.first == _stats(device)[-1].first
    assert straddler.last == _stats(host)[0].last


def test_unsorted_rows_refuse_the_ordered_merge() -> None:
    """The conditional merge, checked rather than assumed. An ordered engine
    closes a group when the key changes, so on unsorted rows it emits several
    records per key -- merging those by key would collapse them into an answer
    no run produced."""
    device, host = _halves(ROWS)  # NOT sorted on the key
    with pytest.raises(SplitError, match="not sorted on the group key"):
        merge_ordered_groups([_groups(device), _groups(host)], key=lambda group: group.key, fold=_fold_groups)


def test_arms_that_arrive_out_of_order_refuse_the_ordered_merge() -> None:
    """Each arm ascends, but the later arm reopens a key the earlier one
    closed -- so the two do not meet on one boundary key and the fold is not
    the merge this is."""
    ascending = sorted(ROWS, key=lambda row: row["key"])
    with pytest.raises(SplitError, match="opens after key"):
        merge_ordered_groups(
            [_groups(ascending[60:]), _groups(ascending[:60])],
            key=lambda group: group.key,
            fold=_fold_groups,
        )


def test_keyed_groups_merge_over_the_whole_key_set() -> None:
    """The contrast with the ordered merge: an engine holding one accumulator
    per distinct key can meet a key anywhere, so a key appears in both arms
    and there is no boundary pair."""

    def keyed(rows):
        accumulated: dict[int, Group] = {}
        for row in rows:
            existing = accumulated.get(row["key"])
            group = Group(key=row["key"], total=row["value"], count=1)
            accumulated[row["key"]] = group if existing is None else _fold_groups(existing, group)
        return [accumulated[key] for key in sorted(accumulated)]

    device, host = _halves(ROWS)
    # the fixture really does put every key in both arms
    assert {group.key for group in keyed(device)} == {group.key for group in keyed(host)}
    merged = merge_keyed_groups([keyed(device), keyed(host)], key=lambda group: group.key, fold=_fold_groups)
    assert merged == keyed(ROWS)


def test_a_top_k_merges_two_k_lists() -> None:
    """Every row of the whole query's top k is in the top k of the range it
    arrived in, so the two k-lists are the complete candidate set."""

    def top_k(rows, k):
        return sorted(rows, key=lambda row: row["value"], reverse=True)[:k]

    device, host = _halves(ROWS)
    merged = merge_top_k([top_k(device, 8), top_k(host, 8)], limit=8, key=lambda row: row["value"])
    assert merged == top_k(ROWS, 8)


def test_a_merged_top_k_keeps_the_earliest_arrival_tie_break() -> None:
    """The fixture's keys repeat, so the answer is decided by ties rather than
    by keys alone: a stable re-rank with the leading range first keeps
    earliest-arrival ties true of the whole table."""
    rows = [{"key": 0, "value": value, "arrival": index} for index, value in enumerate([5, 9, 5, 9, 5, 9, 5, 9])]

    def top_k(part, k):
        return sorted(part, key=lambda row: row["value"], reverse=True)[:k]

    device, host = rows[:4], rows[4:]
    merged = merge_top_k([top_k(device, 3), top_k(host, 3)], limit=3, key=lambda row: row["value"])
    assert [row["arrival"] for row in merged] == [1, 3, 5]


def test_a_negative_limit_is_refused() -> None:
    with pytest.raises(SplitError, match="cannot be negative"):
        merge_top_k([[], []], limit=-1, key=lambda row: row)


def test_sorted_runs_merge_stably() -> None:
    """Equal keys take the earlier range first, which is what keeps a stable
    sort's arrival-order ties true of the whole table rather than of each
    half."""
    rows = [{"key": (seed * 37) % 6, "arrival": seed} for seed in range(40)]

    def run(part, *, descending=False):
        return sorted(part, key=lambda row: row["key"], reverse=descending)

    device, host = _halves(rows)
    merged = merge_sorted_runs([run(device), run(host)], key=lambda row: row["key"])
    assert merged == run(rows)
    assert [row["arrival"] for row in merged] == [row["arrival"] for row in run(rows)]

    descending = merge_sorted_runs([run(device, descending=True), run(host, descending=True)], key=lambda row: row["key"], descending=True)
    assert descending == run(rows, descending=True)


def test_an_empty_merge_answers() -> None:
    assert merge_sorted_runs([], key=lambda row: row) == []
    assert merge_keyed_groups([], key=lambda group: group.key, fold=_fold_groups) == []
    assert merge_ordered_groups([], key=lambda group: group.key, fold=_fold_groups) == []


def _reductions(mergeable=ADDITIVE) -> MergeRule:
    return MergeRule(terminal="reductions", combine=lambda partials: merge_totals(partials, ("total", "len")), mergeable=mergeable)


def test_check_split_refuses_a_mean_by_name() -> None:
    """The aggregate that looks additive and is not."""
    with pytest.raises(SplitError, match="average: a mean is not row-partitionable"):
        check_split(_reductions(), [("average", "mean")])


def test_check_split_refuses_a_distinct_count_by_name() -> None:
    with pytest.raises(SplitError, match="kinds: a distinct count is not row-partitionable"):
        check_split(_reductions(), [("kinds", "n_unique")])


@pytest.mark.parametrize("kind", ("median", "std", "var"))
def test_the_order_statistics_and_moments_are_refused_by_name(kind: str) -> None:
    with pytest.raises(SplitError, match="output: a .*not row-partitionable"):
        check_split(_reductions(), [("output", kind)])


def test_an_unknown_aggregate_is_refused_rather_than_summed() -> None:
    """Default-deny: an aggregate nobody has heard of is not added on the
    assumption that every aggregate is additive."""
    with pytest.raises(SplitError, match="shape: 'kurtosis' is not an aggregate the reductions terminal can merge"):
        check_split(_reductions(), [("shape", "kurtosis")])


def test_a_declared_mean_passes_when_the_partials_carry_it() -> None:
    """Whether a mean survives is a property of what the ENGINES emit, not
    something the merge step can recover -- so a caller whose partials carry
    sum and count separately says so, and the check believes it."""
    check_split(_reductions(ADDITIVE | DEFERRED_MEAN), [("average", "mean"), ("total", "sum")])


def test_boundary_stats_are_declarable_too() -> None:
    check_split(_reductions(ADDITIVE | DEFERRED_MEAN | BOUNDARY_STATS), [("high", "max"), ("opening", "first")])


def test_a_shape_that_cannot_split_refuses_before_any_aggregate() -> None:
    """A windowed result whose window spans the boundary reads rows from both
    ranges, so no aggregate list makes it mergeable."""
    windowed = MergeRule(terminal="rolling", refusal="rolling: a window spanning the split boundary reads rows from both ranges")
    with pytest.raises(SplitError, match="window spanning the split boundary"):
        check_split(windowed, [("total", "sum")])


def test_a_rule_with_no_combine_says_so() -> None:
    """The default is not a silent identity. A rule declared without a combine
    step has nothing to reassemble partials with, and saying that beats
    returning one arm's answer."""
    with pytest.raises(SplitError, match="carries no combine step"):
        MergeRule(terminal="mystery").merge([1, 2])


def test_one_partial_is_the_answer() -> None:
    """A degenerate share ran the whole query on one engine, so no merge is
    involved and none is attempted."""
    assert MergeRule(terminal="mystery").merge([{"total": 7}]) == {"total": 7}


def test_a_failed_arm_is_not_merged_into_a_short_answer() -> None:
    """An engine that invalidated its own work returns a partial that says so.
    Merging that into the other arm's would produce a plausible answer missing
    half the rows, so the failure is what reaches the caller."""
    rule = MergeRule(
        terminal="reductions",
        combine=lambda partials: merge_totals(partials, ("total",)),
        failed=lambda partial: partial.get("error") is not None,
    )
    good, bad = {"total": 10, "error": None}, {"total": 0, "error": "capacity exceeded"}
    assert rule.merge([good, bad]) == bad
    assert rule.merge([good, dict(good)]) == {"total": 20}
