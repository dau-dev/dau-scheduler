# Choose a merge rule

Splitting rows means each engine computes a partial answer, and not every result
can be reassembled from partials. A `MergeRule` says what a query's partials are
and how they combine. This guide picks the rule for each result shape.

## Totals

Sums and counts add. Name the outputs and the rule is `merge_totals`:

```python
from dau_scheduler import ADDITIVE, MergeRule, merge_totals

rule = MergeRule(
    terminal="reductions",
    combine=lambda partials: merge_totals(partials, ("revenue", "len")),
    mergeable=ADDITIVE,
)
```

`merge_totals` takes an optional `finalize` callable for the step after the
addition, which is where a deferred mean is divided.

## A mean

A mean is not row-partitionable. Carry the sum and the count as separate
outputs, add them, and divide in `finalize`. Declare `DEFERRED_MEAN` as
mergeable only when your partials really do carry both; that declaration is how
you tell the checker the shape supports it.

## Grouped results

Two helpers, for two kinds of engine:

- `merge_ordered_groups(partials, key=..., fold=...)` is for rows sorted on
  the group key. Only the group straddling the boundary can appear in both
  arms, so it is folded and the rest concatenate. The precondition is checked:
  unsorted rows are refused rather than merged into an answer no run produced.
- `merge_keyed_groups(partials, key=..., fold=...)` is for an engine that
  keeps one accumulator per distinct key. A key can appear in both arms, so
  every key is looked up, and the result comes back key-ascending.

`fold` combines two partial records of the same group; `key` reads a record's
group key.

## Top-k and sorted output

- `merge_top_k(partials, limit=k, key=..., descending=True)` re-ranks the
  arms' k-lists. A row in the whole query's top k ranks at least as high within
  its own range, so the arms' lists are the complete candidate set. The re-rank
  is stable and the arms are given in row order, which keeps an
  earliest-arrival tie-break true of the whole table.
- `merge_sorted_runs(partials, key=..., descending=False)` merges two sorted
  runs, taking the earlier range first on equal keys for the same reason.

## Boundary statistics

`min`, `max`, `first` and `last` are decided at the edges of a range rather
than accumulated across it. They need the arms' order to be defined, which it
is: the device arm takes the leading rows. Declare `BOUNDARY_STATS` when your
combine step handles them.

## What is refused

`check_split(rule, aggregates)` runs before anything executes. An aggregate
kind outside the rule's `mergeable` set is refused by output name, with a
reason from `AGGREGATE_REFUSALS` where one exists: a mean, an exact distinct
count, a median, a standard deviation or a variance each say what to carry
instead. A rule with `refusal` set cannot be split at all.

## An arm that failed

An engine that invalidated its own work returns a partial that says so. Give
the rule a `failed` predicate and a merge that sees such a partial returns it,
failure and all, rather than a plausible short total nobody can tell is
missing rows.
