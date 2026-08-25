"""Measured host cost: profile the first run instead of modelling it.

The closed form in :mod:`dau_scheduler.posture` needs one number for each
engine. For an engine whose work is analytic -- a fixed rate over a known
number of rows -- that number can be derived. For a general-purpose query
engine on a general-purpose processor it cannot, and an analytic model of one
would be guesswork.

It does not need to be modelled. **Callers run the same query many times over
changing data, and the first execution has nothing to split onto anyway** --
so profiling it is free and turns the estimate into a measurement.

**Fixed cost is separated from per-row cost, and this is not a nicety.** A
query engine does plan-time work that does not scale with data -- one
measurement had it at 11.2 ms of a 14.6 ms total on a 1,000-row frame.
Folding that into a per-row rate would over-estimate a six-million-row query
by orders of magnitude. So a profile carries ``fixed_seconds`` and
``per_row_seconds`` separately and estimates ``fixed + per_row * rows``.

A single profile cannot fully separate the two -- it attributes the nodes
known to be plan-time and treats the rest as scaling. Profiling at two row
counts would fit the line properly, and the shape here supports that later
without changing callers.

A profile is keyed by ``(plan, host)`` because it is not portable between
machines: two hosts running the same query differed by ~1.6x in one
measurement, so a profile keyed by plan alone silently steers a split on the
wrong machine.
"""

from __future__ import annotations

import json
import math
import platform as _platform
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, model_validator

__all__ = ("NodeCost", "ProfileCache", "QueryProfile", "host_identity", "profile_from_spans")


def host_identity() -> str:
    """A name for the machine a profile was measured on."""
    return f"{_platform.node()}/{_platform.machine()}"


class NodeCost(BaseModel):
    """One plan node's measured cost, and whether it scales with rows."""

    model_config: ClassVar = {"frozen": True}

    node: str
    seconds: float
    scales_with_rows: bool

    @model_validator(mode="after")
    def _seconds_is_finite(self) -> NodeCost:
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError(f"node {self.node!r}: seconds must be finite and non-negative, got {self.seconds}")
        return self


class QueryProfile(BaseModel):
    """A measured host cost for one plan on one host.

    ``selectivity`` is the fraction of rows surviving the query's filters when
    the caller measured it. It is left ``None`` rather than inferred: nothing
    in a timing profile reveals row counts, and a guessed selectivity poisons
    everything downstream that prices bytes.

    ``engine_version`` records which build of the engine was measured. It
    defaults to ``""`` -- unknown -- and NOT to whatever is installed, which
    would let a hand-written profile claim to have been measured under an
    engine it never ran on.
    """

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}

    plan_key: str
    host: str
    rows: int
    fixed_seconds: float
    per_row_seconds: float
    nodes: tuple[NodeCost, ...] = ()
    selectivity: float | None = None
    engine_version: str = ""

    @model_validator(mode="after")
    def _costs_are_sane(self) -> QueryProfile:
        if self.rows <= 0:
            raise ValueError(f"profile over {self.rows} rows carries no information")
        for name, value in (("fixed_seconds", self.fixed_seconds), ("per_row_seconds", self.per_row_seconds)):
            # NaN fails every comparison, so a bare `< 0` check admits it and
            # it then propagates silently through every downstream estimate
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.selectivity is not None and not 0.0 <= self.selectivity <= 1.0:
            raise ValueError(f"selectivity {self.selectivity} is not a fraction")
        return self

    def estimate(self, rows: int) -> float:
        """Host seconds for ``rows``, as ``fixed + per_row * rows``."""
        if rows < 0:
            raise ValueError(f"negative rows {rows}")
        return self.fixed_seconds + self.per_row_seconds * rows

    def is_stale_for(self, rows: int, *, engine_version: str, factor: float = 10.0) -> bool:
        """Whether this measurement can still price ``rows`` on the engine here
        now.

        Two things invalidate a measurement, and **both are answered by this
        one predicate on purpose**: a caller that has to remember to ask a
        second question is how a version stamp comes to be recorded on every
        profile and consulted by nothing.

        *Row count.* Extrapolating one measurement across orders of magnitude
        is where a profile-guided split goes confidently wrong, and
        confidently wrong is worse than absent: the split decides which engine
        runs which rows, so the error is self-reinforcing.

        *Engine version.* The argument for keying on the host -- two machines
        measured ~1.6x apart, so a profile keyed by plan alone steers the
        split on the wrong one -- applies verbatim to the engine being
        measured, and this module exists BECAUSE an analytic model of that
        engine would be guesswork. A measurement taken under one build says
        nothing about another.

        A profile with no recorded version is **unknown, not current**, and is
        stale for the same reason a guessed selectivity is refused: it may
        have been measured under anything.
        """
        if not self.engine_version or self.engine_version != engine_version:
            return True
        if rows <= 0:
            return True
        ratio = rows / self.rows
        return ratio > factor or ratio < 1.0 / factor


def profile_from_spans(
    spans: Iterable[tuple[int, int, str]],
    *,
    rows: int,
    plan_key: str,
    host: str | None = None,
    is_fixed: Callable[[str], bool] = lambda node: False,
    selectivity: float | None = None,
    engine_version: str = "",
) -> QueryProfile:
    """Build a profile from an engine's per-node timings.

    ``spans`` are ``(start, end, node name)`` in MICROSECONDS, as a profiling
    query engine reports them. ``is_fixed`` names the plan-time nodes -- the
    ones that must not be divided by the row count.

    **Nodes overlap (an engine runs some in parallel) and leave gaps, so
    summing ``end - start`` is NOT elapsed time.** It double-counts the
    overlap and drops the gaps entirely: with optimization 0-1000, scan
    1000-11000 and filter 5000-15000, the true elapsed cost is 15 ms and the
    sum reports 21 ms.

    So the span is taken as elapsed wall time, the fixed nodes are measured as
    a UNION of their intervals, and everything else -- including the gaps,
    which are real elapsed cost -- is attributed to scaling work.
    """
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    records = [(int(start), int(end), str(node)) for start, end, node in spans]
    if not records:
        raise ValueError("profile produced no timing rows")
    elapsed = (max(end for _start, end, _node in records) - min(start for start, _end, _node in records)) / 1e6

    fixed = _union_seconds(sorted((start, end) for start, end, node in records if is_fixed(node)))
    scaling = max(0.0, elapsed - fixed)
    costs = tuple(NodeCost(node=node, seconds=max(0, end - start) / 1e6, scales_with_rows=not is_fixed(node)) for start, end, node in records)
    return QueryProfile(
        plan_key=plan_key,
        host=host or host_identity(),
        rows=rows,
        fixed_seconds=fixed,
        per_row_seconds=scaling / rows,
        nodes=costs,
        selectivity=selectivity,
        engine_version=engine_version,
    )


def _union_seconds(spans: Sequence[tuple[int, int]]) -> float:
    """Total length of a set of microsecond intervals, counting overlap once."""
    total = 0
    current: tuple[int, int] | None = None
    for start, end in spans:
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    if current is not None:
        total += current[1] - current[0]
    return total / 1e6


class ProfileCache:
    """Profiles on disk, keyed by plan and host.

    Deliberately a plain directory of JSON rather than anything cleverer: a
    profile is small, human-readable, and worth being able to delete by hand
    when it goes stale.

    The engine version is deliberately **not** part of the key. A host is a
    key because you cannot re-measure on a machine you are not on; an engine
    version is not, because you can always re-measure under the one you have.
    So it is staleness (:meth:`QueryProfile.is_stale_for`), handled exactly as
    a row count that has moved too far, rather than a second slot on disk.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, plan_key: str, host: str) -> Path:
        safe = host.replace("/", "_")
        return self.root / f"{plan_key}.{safe}.profile.json"

    def get(self, plan_key: str, host: str | None = None) -> QueryProfile | None:
        path = self._path(plan_key, host or host_identity())
        if not path.is_file():
            return None
        return QueryProfile(**json.loads(path.read_text()))

    def put(self, profile: QueryProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(profile.plan_key, profile.host)
        path.write_text(json.dumps(profile.model_dump(), indent=2, sort_keys=True))
        return path

    def entries(self) -> Iterable[Path]:
        return sorted(self.root.glob("*.profile.json")) if self.root.is_dir() else ()
