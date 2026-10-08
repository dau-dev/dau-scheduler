# Why rows, measured costs and refusals

`dau-scheduler` is small on purpose. It makes one decision, keeps one promise,
and refuses what it cannot keep. This page explains the three choices that
shape it.

## Rows, not stages

A query can be split two ways: by stage, where one engine runs the scan and
filter and the other the aggregation, or by rows, where both run the whole
query over their own slice. Splitting by stage ships every intermediate across
the link between the engines, and for the workloads this library serves the
link is the bottleneck. So the split is always over rows: each engine runs the
same query, and only partial results cross.

That choice is also what makes the executor engine-agnostic. An arm is a
callable handed the query and the rows it owns; nothing here knows what is on
the other end, which is how the same executor serves two processors, a
processor and an accelerator, or a local worker and a remote one.

## Cost is measured

The collaborative share has a closed form, but its inputs are each engine's
time for the whole query, and a fabricated number under a real decision is
worse than no decision: the share changes which engine runs which rows, so an
error in it reinforces itself. `plan_split` therefore refuses to guess. The
host figure is measured from the query's first run, which has nothing to split
onto anyway, keyed by plan and host, and re-measured when the row count or the
engine has moved too far from what was measured. A speedup without the machine
it was measured on is not a figure, it is a story.

The device side of the balance includes a fixed cost per job. One measurement
put it at a few milliseconds beside a 12.3 ms host time, which is enough to
rule a split out entirely for small queries. When that happens the answer is
host-only and says why.

## Merging is default-deny

Not every result can be reassembled from two partial answers. Sums and counts
add; a mean needs its sum and count carried separately; a top-k merges two
k-lists; a grouped aggregation needs the boundary group folded; an exact
distinct count cannot be combined without the elements themselves. The library
writes that taxonomy down, and a `MergeRule` declares which kinds its shape can
combine. Anything not declared is refused before either engine starts, by
output name and with the fix in the message. A silently mis-merged answer is
the failure this library exists to prevent, so the default is to refuse.

## What overlap measures

`SplitExecution.overlap_seconds` is the intersection of the two arms'
wall-clock intervals. It establishes that the arms were in flight together,
which is the property a split needs and the one a serialized executor loses.
It does not claim that two Python callables made parallel progress inside one
interpreter; a real second engine runs outside it, and the figure is honest
about what it can see.
