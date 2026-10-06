"""Run a planned split: two engines, disjoint row ranges, one answer.

:mod:`dau_scheduler.posture` decides how much of a query the second engine
should carry; this runs that decision. The second engine takes the LEADING
row range and the first the trailing one, both run the WHOLE query over
their own rows, and the partial results merge into the answer an unsplit run
would have produced -- the same object, so a split is a drop-in for whatever
the caller ran before rather than a second result shape to handle.

Both engines are seams. An arm is a callable handed the query, its own
:class:`~dau_scheduler.rows.RowSource`, and the broadcast side of a join if
there is one; what it does with them is entirely its own business, and
nothing here assumes either arm is in this process.

**The refusal comes before either arm starts.** A result that cannot be
reassembled is refused while it still costs nothing; a refusal that arrives
after the work is a refusal that already paid for it.

**The accounting covers the whole call.** ``wall_seconds`` spans from entry
to answer, and the two terms that are not an arm are named:
``materialize_seconds`` for everything done to the rows before an arm starts,
and ``merge_seconds`` for putting the partials together. A timing field that
excludes the dominant term is worse than no field -- and on a large split
where the caller passes an iterator, marshalling IS the dominant term.

**The collector stays the caller's.** Staging tens of millions of row tuples
leaves a heap whose collection pauses cost more than the query being split,
and a library that quietly repartitions its caller's heap is an imposition.
Handing an arm a range rather than a fresh list removes this module's own
contribution to that; the rest belongs to whoever allocated the rows.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from pydantic import BaseModel, model_validator

from .merge import MergeRule, check_split
from .posture import POSTURES, WorkSplit
from .rows import RowSource, as_row_source

__all__ = ("SplitArm", "SplitExecution", "execute_split")

# One engine's half of a split: run the query over ITS rows and return the
# partial result, exactly as an unsplit run would for those rows.
# ``build_rows`` is the broadcast side of a join, which reaches both arms
# whole rather than being divided -- it is a table, not part of the row range
# being split, and halving it would drop every match whose build row landed
# in the other arm while the totals still looked like totals.
SplitArm = Callable[[Any, RowSource, "RowSource | None"], Any]
#: An arm may carry ``row_granularity``: the row multiple it can accept, for
#: an engine whose job length must sit on a fixed row grid (a transfer unit,
#: a lane width). The executor reads it off
#: the arm rather than taking it as a parameter, so the fact travels with the
#: engine that has it and cannot be omitted by a caller. A plain function has
#: none and is taken at 1.


class SplitExecution(BaseModel):
    """One executed split: what each engine ran, how long, and the answer.

    ``result`` is whatever an UNSPLIT run of this query returns, so the
    caller's decode path is unchanged by having split the work.

    ``overlap_seconds`` is the intersection of the two arms' wall-clock
    intervals. It measures that the arms were IN FLIGHT together, which is
    the property a split needs and the one a serialized executor loses. It
    does not claim parallel progress inside a single interpreter, which
    nothing here could promise on the caller's behalf.
    """

    model_config: ClassVar = {"frozen": True}

    posture: str
    device_share: float
    terminal: str
    device_rows: int
    host_rows: int
    device_seconds: float
    host_seconds: float
    wall_seconds: float
    overlap_seconds: float
    materialize_seconds: float
    merge_seconds: float
    # the row multiple the leading arm's count was snapped to (1 = none)
    device_granularity: int = 1
    result: Any = None

    @model_validator(mode="after")
    def _figures_are_sane(self) -> SplitExecution:
        if self.posture not in POSTURES:
            raise ValueError(f"unknown posture {self.posture!r}; known: {POSTURES}")
        if not math.isfinite(self.device_share) or not 0.0 <= self.device_share <= 1.0:
            raise ValueError(f"device_share {self.device_share} is not a fraction")
        for name in ("device_rows", "host_rows", "device_granularity"):
            value = getattr(self, name)
            if value < 0 or (name == "device_granularity" and value < 1):
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name in ("device_seconds", "host_seconds", "wall_seconds", "overlap_seconds", "materialize_seconds", "merge_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.overlap_seconds > min(self.device_seconds, self.host_seconds) + 1e-9 and self.device_rows and self.host_rows:
            raise ValueError("overlap_seconds cannot exceed the shorter arm")
        return self


def execute_split(
    query: Any,
    rows: RowSource | Iterable[Any],
    split: WorkSplit,
    merge: MergeRule,
    *,
    device: SplitArm,
    host: SplitArm,
    build_rows: RowSource | Iterable[Any] | None = None,
    aggregates: Iterable[tuple[str, str]] = (),
) -> SplitExecution:
    """Run ``split``'s share of the rows on one engine and the rest on the
    other, at the same time, and merge the partials into the query's answer.

    ``query`` is passed to both arms untouched; this module never inspects it.

    The ``device`` arm takes rows ``[0, share*n)`` and the ``host`` arm the
    rest. That order is part of the contract, not an implementation detail:
    a grouped ``first``/``last`` and a top-k's earliest-arrival tie-break are
    defined by it. An arm with no rows is not launched at all -- a share of
    0.0 or 1.0 runs exactly the unsplit path (given a table on the device's
    granularity), which is what makes the degenerate ends of a share sweep
    meaningful.

    ``rows`` is the whole table as a :class:`~dau_scheduler.rows.RowSource` --
    a :class:`~dau_scheduler.rows.RowRange` when both arms run from data they
    already hold, a :class:`~dau_scheduler.rows.RowBatch` when one of them
    reads values -- or any iterable of rows, which is wrapped and charged to
    ``materialize_seconds``.

    ``aggregates`` are ``(output name, aggregate kind)`` pairs checked against
    ``merge`` before anything runs (see
    :func:`dau_scheduler.merge.check_split`).

    A ``device`` arm carrying ``row_granularity`` (see :data:`SplitArm`) can
    only take a row count that divides it -- an engine whose job length must
    sit on a fixed row grid. The boundary is snapped to the nearest such multiple
    and the count actually given is reported as ``device_rows``. It is an
    execution fact about the engine, not part of the share the planner
    chose, which is why it rides the arm rather than the
    :class:`~dau_scheduler.posture.WorkSplit`.
    """
    started = time.perf_counter()
    # refuse BEFORE running anything: a refusal that arrives after the work
    # is a refusal that already cost the work -- and before touching the
    # rows, which may be a one-shot iterable a retry could not replay
    split.check()
    device_granularity = int(getattr(device, "row_granularity", 1))
    if device_granularity < 1:
        raise ValueError(f"the device arm's row_granularity must be at least 1, got {device_granularity}")
    check_split(merge, aggregates)
    # everything done TO the rows before an arm starts is one term, because
    # cutting a materialized batch at the boundary costs the same kind of
    # per-row copy that wrapping it avoids -- reporting only the wrap would
    # rebuild the hole this field exists to close
    marshalled = time.perf_counter()
    source = as_row_source(rows)
    build = None if build_rows is None else as_row_source(build_rows)
    boundary = _device_row_count(split.device_share, len(source), device_granularity)

    arms: list[tuple[str, SplitArm, RowSource]] = []
    if boundary:
        arms.append(("device", device, source.slice(0, boundary)))
    if boundary < len(source) or not arms:
        arms.append(("host", host, source.slice(boundary, len(source))))
    materialize = time.perf_counter() - marshalled

    with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="dau-split") as pool:
        # every arm is submitted before ANY result is collected. Collecting
        # one before submitting the next would serialize the split: still
        # correct, still reporting a speedup, and worthless.
        futures = [(name, len(arm_rows), pool.submit(_timed, arm, query, arm_rows, build)) for name, arm, arm_rows in arms]
        finished = [(name, count, future.result()) for name, count, future in futures]

    spans = {name: (start, end) for name, _count, (_partial, start, end) in finished}
    counts = {name: count for name, count, _timing in finished}
    seconds = {name: end - start for name, (start, end) in spans.items()}
    # leading range first -- the merges that care about row order read it
    # from this sequence
    partials = [partial for _name, _count, (partial, _start, _end) in finished]
    merging = time.perf_counter()
    result = merge.merge(partials)
    merged = time.perf_counter() - merging
    return SplitExecution(
        posture=split.posture,
        device_share=split.device_share,
        terminal=merge.terminal,
        device_rows=counts.get("device", 0),
        host_rows=counts.get("host", 0),
        device_seconds=seconds.get("device", 0.0),
        host_seconds=seconds.get("host", 0.0),
        wall_seconds=time.perf_counter() - started,
        overlap_seconds=_overlap(list(spans.values())),
        materialize_seconds=materialize,
        merge_seconds=merged,
        device_granularity=device_granularity,
        result=result,
    )


def _device_row_count(share: float, rows: int, granularity: int = 1) -> int:
    """The leading arm's row count for a share. Rounded to the nearest
    multiple of ``granularity``, then clamped to the table. A share of 0.0
    launches no device arm; a share of 1.0 leaves the other arm nothing
    when the table is on the grid, and otherwise leaves it the tail the
    device cannot take -- the device's limit, not a short-changed share."""
    if granularity < 1:
        raise ValueError(f"device_granularity must be at least 1, got {granularity}")
    snapped = round(share * rows / granularity) * granularity
    # clamp to the largest count on the grid, not to the table: a table that
    # is itself off the grid must leave its ragged tail to the other arm
    return min(rows - rows % granularity, max(0, snapped))


def _timed(arm: SplitArm, query: Any, rows: RowSource, build: RowSource | None) -> tuple[Any, float, float]:
    start = time.perf_counter()
    partial = arm(query, rows, build)
    return partial, start, time.perf_counter()


def _overlap(spans: list[tuple[float, float]]) -> float:
    if len(spans) < 2:
        return 0.0
    return max(0.0, min(end for _start, end in spans) - max(start for start, _end in spans))
