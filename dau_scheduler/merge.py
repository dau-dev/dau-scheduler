"""Which results survive being split across two engines, and how they combine.

Splitting a query by rows means each engine computes a PARTIAL answer over
the rows it owns, and the answer the caller asked for has to be reassembled
from two of them. **Not every result can be.** This module is the taxonomy of
which can, what each one's merge is, and what the merge quietly assumes.

The aggregates:

* **Sums and counts add.** They are the easy case and the reason the rest
  looks easier than it is.
* **A mean does not merge.** The mean of two ranges is not the mean of their
  means, and weighting by row count is just deriving it from the sum and the
  count after the fact. So a mean survives a split only when the partials
  carry sum and count SEPARATELY and the division happens after the merge --
  which is a property of what the engines emit, not something a merge step
  can recover.
* **A distinct count cannot be combined at all** without the elements it
  counted. Two ranges that each saw 100 distinct keys may between them have
  seen 100 or 200, and nothing in the partials says which.
* **Variance and standard deviation** merge only from count, sum and
  sum-of-squares carried separately, for the same reason as the mean.
* **A median, or any order statistic of the whole**, needs the values rather
  than two partial answers.
* **Minimum and maximum merge trivially**; **first and last merge by
  POSITION**, which is why the arms' row ranges have a defined order -- the
  merged group opens in the earlier range and closes in the later one.

The result shapes:

* **A top-k merges by re-ranking the two k-lists.** A row in the whole
  query's top k ranks at least as high within the range it arrived in, so it
  is in that range's k-list: the two k-lists are the complete candidate set,
  and nothing outside them can qualify.
* **Sorted rows merge as two sorted runs**, with equal keys taking the
  earlier range first so a stable sort stays stable across the boundary.
* **A grouped aggregation merges by key.** Two cases, and confusing them is a
  wrong answer rather than an error. When the partials are ORDERED by key,
  only the group that straddles the boundary can appear in both, and merging
  is a concatenation with that one pair folded -- but that is true only if the
  input really is sorted on the group key, so :func:`merge_ordered_groups`
  VERIFIES it rather than trusting it. When the partials are keyed rather
  than ordered, any key may appear in both arms and the merge is over the
  whole key set (:func:`merge_keyed_groups`).
* **A windowed result does not merge at all** when a window can span the
  boundary: the window reads rows from both ranges, so neither arm's record
  is the query's answer for those rows.

**Everything here is default-deny.** :func:`check_split` refuses an aggregate
that is not named as mergeable rather than adding it on the assumption that
every aggregate is additive. Refusing costs a caller one message; the
alternative is a confident wrong answer, which is the failure this whole
layer exists to make impossible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import SplitError

__all__ = (
    "ADDITIVE",
    "AGGREGATE_REFUSALS",
    "BOUNDARY_STATS",
    "DEFERRED_MEAN",
    "MergeRule",
    "check_split",
    "merge_keyed_groups",
    "merge_ordered_groups",
    "merge_sorted_runs",
    "merge_top_k",
    "merge_totals",
)

# The aggregates whose partials combine by addition, and nothing else does
# without help.
ADDITIVE: frozenset[str] = frozenset({"sum", "count"})

# A mean is mergeable ONLY where the partial carries its sum and its count
# separately and the caller divides after the merge. Naming the set is how a
# caller says that is what its partials do.
DEFERRED_MEAN: frozenset[str] = frozenset({"mean"})

# Aggregates decided at the edges of a range rather than accumulated across
# it: min and max compare, first and last are chosen by which range is
# earlier. Both need the arms' order to be defined, which it is.
BOUNDARY_STATS: frozenset[str] = frozenset({"min", "max", "first", "last"})

# Why a named aggregate cannot be merged, in the words the caller needs to fix
# it. Anything not here gets the generic refusal, which still names the
# aggregate and the result shape it was asked of.
AGGREGATE_REFUSALS: dict[str, str] = {
    "mean": "a mean is not row-partitionable; carry the sum and the count as separate outputs and divide after the merge",
    "n_unique": "a distinct count is not row-partitionable: counts over disjoint ranges cannot be combined without the elements themselves",
    "median": "a median is not row-partitionable: an order statistic of the whole needs the values, not two partial answers",
    "std": "a standard deviation is not row-partitionable; carry count, sum and sum-of-squares and derive it after the merge",
    "var": "a variance is not row-partitionable; carry count, sum and sum-of-squares and derive it after the merge",
}


def _no_combine(partials: Sequence[Any]) -> Any:
    raise SplitError("this merge rule carries no combine step, so its partials cannot be reassembled")


@dataclass(frozen=True, slots=True)
class MergeRule:
    """What one query's partials are, and whether they can be put together.

    ``terminal`` names the result shape, for messages. ``combine`` takes the
    arms' partials IN ARM ORDER -- the leading rows first -- and returns the
    answer the unsplit query would have produced. ``mergeable`` is the set of
    aggregate kinds whose partials this shape can combine, consulted by
    :func:`check_split` and default-deny.

    ``refusal``, when set, says this shape cannot be split at all, and no
    ``combine`` is reached.

    ``failed`` is the guard for an arm that could not complete. An engine that
    invalidated its own work returns a partial that says so, and merging that
    into the other arm's would produce a plausible SHORT answer -- so if any
    partial reports failure, the merged result is that partial, failure and
    all, rather than a total nobody can tell is missing rows.
    """

    terminal: str
    combine: Callable[[Sequence[Any]], Any] = field(default=_no_combine)
    mergeable: frozenset[str] = frozenset()
    failed: Callable[[Any], bool] | None = None
    refusal: str | None = None

    def merge(self, partials: Sequence[Any]) -> Any:
        """The merged answer for these partials."""
        if self.failed is not None:
            for partial in partials:
                if self.failed(partial):
                    return partial
        if len(partials) == 1:
            # a degenerate share ran the whole query on one engine, so its
            # partial IS the answer and no merge is involved
            return partials[0]
        return self.combine(partials)


def check_split(rule: MergeRule, aggregates: Iterable[tuple[str, str]] = ()) -> None:
    """Refuse a query whose result cannot be reassembled, by name.

    ``aggregates`` are ``(output name, aggregate kind)`` pairs, so the refusal
    names the output the caller has to change rather than only the kind.

    Kept separate from execution so a planner can ask whether a query is
    splittable at all before committing anything to either engine.
    """
    if rule.refusal is not None:
        raise SplitError(rule.refusal)
    for name, kind in aggregates:
        if kind not in rule.mergeable:
            reason = AGGREGATE_REFUSALS.get(kind, f"{kind!r} is not an aggregate the {rule.terminal} terminal can merge {sorted(rule.mergeable)}")
            raise SplitError(f"{name}: {reason}")


def merge_totals(partials: Sequence[dict[str, Any]], names: Sequence[str], *, finalize: Callable[[dict[str, Any]], Any] | None = None) -> Any:
    """Add the named aggregates, then RE-DERIVE anything computed from them.

    ``finalize`` is where an output that is an EXPRESSION over the aggregates
    -- a ratio, a share, a rate -- is recomputed from the merged totals.
    Adding the two arms' ratios would produce a number that is not a ratio of
    anything, and it would look exactly like a total.
    """
    merged = {name: sum(partial[name] for partial in partials) for name in names}
    return merged if finalize is None else finalize(merged)


def merge_ordered_groups(partials: Sequence[Sequence[Any]], *, key: Callable[[Any], Any], fold: Callable[[Any, Any], Any]) -> list[Any]:
    """Concatenate per-group records, folding the group that straddles the
    boundary.

    Valid only when the rows are SORTED on the group key, because that is what
    makes the boundary pair the only group two arms can share. The
    precondition is verified rather than trusted: on unsorted rows an
    order-sensitive engine emits several records per key, and merging those by
    key would collapse them into an answer no run produced.

    ``fold`` combines two partial records of the same group; ``key`` reads a
    record's group key.
    """
    merged: list[Any] = []
    for records in partials:
        _require_ascending(records, key)
        records = list(records)
        if merged and records:
            if key(merged[-1]) > key(records[0]):
                raise SplitError(
                    f"group key {key(records[0])} opens after key {key(merged[-1])} closed in an earlier row range: "
                    "folding partial groups at the boundary needs the rows sorted on the group key"
                )
            if key(merged[-1]) == key(records[0]):
                merged[-1] = fold(merged[-1], records[0])
                records = records[1:]
        merged.extend(records)
    return merged


def _require_ascending(records: Sequence[Any], key: Callable[[Any], Any]) -> None:
    for index in range(1, len(records)):
        earlier, later = key(records[index - 1]), key(records[index])
        if earlier >= later:
            raise SplitError(
                f"one row range closed group key {earlier} before {later}: the rows are not sorted on the group key, "
                "so their partial groups are not the boundary pair this merge can fold"
            )


def merge_keyed_groups(partials: Sequence[Sequence[Any]], *, key: Callable[[Any], Any], fold: Callable[[Any, Any], Any]) -> list[Any]:
    """Merge per-key records that carry no ordering, over the whole key set.

    The contrast with :func:`merge_ordered_groups` is the point. An engine
    holding one accumulator per distinct key can meet a key anywhere in its
    rows, so a key may appear in both arms and there is no boundary pair to
    fold -- every key has to be looked up. The result is key-ascending, which
    is the order such an engine drains in.
    """
    accumulated: dict[Any, Any] = {}
    for records in partials:
        for record in records:
            existing = accumulated.get(key(record))
            accumulated[key(record)] = record if existing is None else fold(existing, record)
    return [accumulated[group] for group in sorted(accumulated)]


def merge_top_k(partials: Sequence[Sequence[Any]], *, limit: int, key: Callable[[Any], Any], descending: bool = True) -> list[Any]:
    """Re-rank the arms' k-lists into one.

    A row in the whole query's top k ranks at least as high within the range
    it arrived in, so it is in that range's k-list: the arms' outputs are the
    complete candidate set. Each arm emits equal keys in arrival order and the
    arms are given in row order, so a STABLE re-rank keeps an earliest-arrival
    tie-break true of the whole table rather than of each half.
    """
    if limit < 0:
        raise SplitError(f"a top-k limit cannot be negative, got {limit}")
    candidates = [row for partial in partials for row in partial]
    return sorted(candidates, key=key, reverse=descending)[:limit]


def merge_sorted_runs(partials: Sequence[Sequence[Any]], *, key: Callable[[Any], Any], descending: bool = False) -> list[Any]:
    """Merge the arms' sorted runs into one sorted run.

    Equal keys take the earlier range first, which is what keeps a stable
    sort's arrival-order ties true of the whole table rather than of each
    half.
    """
    if not partials:
        return []
    merged = list(partials[0])
    for partial in partials[1:]:
        merged = _merge_two_runs(merged, list(partial), key=key, descending=descending)
    return merged


def _merge_two_runs(earlier: list[Any], later: list[Any], *, key: Callable[[Any], Any], descending: bool) -> list[Any]:
    merged: list[Any] = []
    left = right = 0
    while left < len(earlier) and right < len(later):
        takes_earlier = key(earlier[left]) >= key(later[right]) if descending else key(earlier[left]) <= key(later[right])
        if takes_earlier:
            merged.append(earlier[left])
            left += 1
        else:
            merged.append(later[right])
            right += 1
    merged.extend(earlier[left:])
    merged.extend(later[right:])
    return merged
