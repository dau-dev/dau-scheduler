# Merge rules

## `MergeRule`

| Field       | Meaning                                                                                   |
| ----------- | ----------------------------------------------------------------------------------------- |
| `terminal`  | the result shape, used in messages                                                        |
| `combine`   | takes the arms' partials in arm order (leading rows first) and returns the unsplit answer |
| `mergeable` | the aggregate kinds this shape can combine; consulted by `check_split`, default-deny      |
| `failed`    | optional predicate; a partial it accepts is returned as the merged result                 |
| `refusal`   | when set, the shape cannot be split at all                                                |

`MergeRule.merge(partials)` returns the single partial when only one arm ran,
so a degenerate share runs exactly the unsplit path.

## Aggregate kinds

| Set              | Kinds                         | Condition                                                  |
| ---------------- | ----------------------------- | ---------------------------------------------------------- |
| `ADDITIVE`       | `sum`, `count`                | partials add                                               |
| `DEFERRED_MEAN`  | `mean`                        | the partial carries sum and count separately; divide after |
| `BOUNDARY_STATS` | `min`, `max`, `first`, `last` | decided at range edges; the arms' order defines first/last |

`AGGREGATE_REFUSALS` holds the reason given for `mean`, `n_unique`, `median`,
`std` and `var`. Any other kind outside `mergeable` gets a generic refusal that
still names the aggregate and the terminal.

## Helpers

| Function                                               | Combines                                                                                      |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `merge_totals(partials, names, finalize=None)`         | named totals by addition, then `finalize`                                                     |
| `merge_ordered_groups(partials, key=, fold=)`          | per-group records of rows sorted on the key; folds the boundary group; refuses unsorted input |
| `merge_keyed_groups(partials, key=, fold=)`            | per-key records with no ordering; result key-ascending                                        |
| `merge_top_k(partials, limit=, key=, descending=True)` | the arms' k-lists, stable re-rank                                                             |
| `merge_sorted_runs(partials, key=, descending=False)`  | two sorted runs, earlier range first on ties                                                  |

## `check_split(rule, aggregates)`

`aggregates` are `(output name, aggregate kind)` pairs. The refusal names the
output, so the caller knows what to change. It is separate from execution so a
planner can ask whether a query is splittable before committing anything to
either engine.
