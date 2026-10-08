# Split a query across two engines

This tutorial runs one query on two engines at once and reads the result back as
if nothing had been split. Both engines are plain Python functions here, so it
runs on any machine with `dau-scheduler` installed; the library never learns what
is on either end.

## 1. A query and its rows

The query is whatever your engines understand. The executor passes it through
untouched.

```python
query = {"filter": lambda row: row["day"] < 30}
rows = [{"revenue": (seed * 31) % 1000, "day": seed % 60} for seed in range(100_000)]
```

## 2. An arm

An arm is a callable that takes the query, the rows it owns and the broadcast
side of a join (none here), and returns the partial answer an unsplit run would
have produced for those rows.

```python
def arm(query, rows, build):
    kept = [row for row in rows.values() if query["filter"](row)]
    return {"revenue": sum(row["revenue"] for row in kept), "len": len(kept)}
```

`rows` arrives as a row source. `rows.values()` is how an arm that needs the
values asks for them; an arm that holds its own copy of the table reads only
`rows.start` and `len(rows)`.

## 3. Decide the split

`plan_split` turns a posture into a share. The `collaborative` posture needs
each engine's whole-query time; the figures below are illustrative.

```python
from dau_scheduler import plan_split

split = plan_split("collaborative", cpu_seconds=0.0276, device_seconds=0.0186)
print(round(split.device_share, 3), round(split.predicted_speedup, 2))
print(split.rationale)
```

The share is the point where both arms finish together. A split that could not
beat the host alone comes back as a `cpu` split with a rationale saying why,
rather than a split predicted to win by a margin the fixed cost would eat.

## 4. Say how partials combine

A `MergeRule` names the result shape, the function that combines the arms'
partials, and the aggregate kinds this shape can merge. Sums and counts add.

```python
from dau_scheduler import ADDITIVE, MergeRule, merge_totals

merge = MergeRule(
    terminal="reductions",
    combine=lambda partials: merge_totals(partials, ("revenue", "len")),
    mergeable=ADDITIVE,
)
```

## 5. Run both arms

```python
from dau_scheduler import execute_split

execution = execute_split(
    query,
    rows,
    split,
    merge,
    device=arm,
    host=arm,
    aggregates=[("revenue", "sum"), ("len", "count")],
)
print(execution.result)
print(execution.device_rows, execution.host_rows)
print(round(execution.overlap_seconds, 6))
```

`result` is the same object an unsplit run returns, so nothing downstream
changes. `overlap_seconds` is how long both arms were in flight together: the
property a split needs and the one a serialized executor loses.

## 6. Watch a refusal

`aggregates` is what makes refusal possible. Ask for a mean and the split is
refused before either engine starts, naming the output and what to do instead.

```python
from dau_scheduler import SplitError

try:
    execute_split(query, rows, split, merge, device=arm, host=arm, aggregates=[("average", "mean")])
except SplitError as refusal:
    print(refusal)
```

The fix the message asks for is to carry the sum and the count as separate
outputs and divide after the merge.

## Where next

- [Measure host cost](../how-to/measure-host-cost.md) replaces the illustrative
  `cpu_seconds` with a profile of the query's first run.
- [Choose a merge rule](../how-to/choose-a-merge-rule.md) covers grouped
  results, top-k and sorted output.
- [Postures](../reference/postures.md) lists what each posture means.
