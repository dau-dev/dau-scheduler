# Row sources and execution

## Row sources

| Type       | Carries                                    | Cost                                 |
| ---------- | ------------------------------------------ | ------------------------------------ |
| `RowRange` | offsets only (`start`, `stop`)             | two integers, whatever the row count |
| `RowBatch` | offsets and the values, wrapped not copied | nothing per row                      |

Both expose `start`, `stop`, `len()` and `slice(start, stop)` with relative
indices and absolute offsets, so an arm learns where its rows sit in the table.
`RowBatch.values()` returns the rows; a `RowRange` asked for values raises a
typed refusal instead of materializing a table behind the caller's back.

`as_row_source(rows)` passes a source through, wraps a sequence in place, and
drains only an iterator, because both arms read from one source and an iterator
cannot serve two.

## `execute_split`

```python
execute_split(query, rows, split, merge, *, device, host, build_rows=None, aggregates=())
```

Runs the device arm on rows `[0, share * n)` and the host arm on the rest, at
the same time, and merges the partials. That order is part of the contract:
grouped `first`/`last` and a top-k's earliest-arrival tie-break are defined by
it. An arm with no rows is not launched.

`build_rows` is the broadcast side of a join, handed to both arms whole.
`aggregates` is checked against `merge` before anything runs.

A device arm with a `row_granularity` attribute takes a row count that divides
it; the boundary is snapped to the nearest multiple and the count given is
reported as `device_rows`. It rides the arm, not the split, because it is a
fact about the engine rather than part of the share the planner chose.

## `SplitExecution`

| Field                            | Meaning                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `result`                         | what an unsplit run of the query returns                                       |
| `device_rows`, `host_rows`       | the row counts each arm was given                                              |
| `device_seconds`, `host_seconds` | each arm's wall time                                                           |
| `wall_seconds`                   | the whole call                                                                 |
| `overlap_seconds`                | the intersection of the two arms' intervals: that they were in flight together |
| `materialize_seconds`            | everything done to the rows before an arm started                              |
| `merge_seconds`                  | the combine step                                                               |
| `device_granularity`             | the row multiple the device count was snapped to; 1 means none                 |

`overlap_seconds` measures concurrency of the arms. It does not claim parallel
progress inside one interpreter, which nothing here could promise on the
caller's behalf.

## Errors

`SplitError` is raised for a refused aggregate, a rule with no combine step, a
rule marked unsplittable, and a grouped merge whose rows are not sorted.
