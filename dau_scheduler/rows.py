"""One contiguous stretch of rows, handed to one engine.

Both engines run the whole query over disjoint stretches of the same table,
so each arm needs to be told which stretch is its own. Every source names
that the same way -- ``start``, ``stop``, and a length -- because that is
all an arm needs: an engine holding its own copy of the table runs
``len(source)`` rows, and an engine holding a resident frame slices it at
``source.start``. Neither dereferences a row.

Whether the VALUES come along is what separates the two concrete sources,
and it is a type distinction rather than a flag so that an arm which cannot
work from offsets alone says so at the seam. :class:`RowRange` is two
integers and costs nothing at any row count; :class:`RowBatch` also carries
the rows, which is what an in-process arm reads.

The distinction is not academic. A seam typed as an iterable of rows makes
the caller materialize the table whether or not either arm reads it -- at
six million rows that was measured at 183 ms of marshalling in front of a
7 ms query, 26x the work being divided, to build 1.2 GB of tuples nothing
dereferenced. :meth:`RowSource.values` is therefore default-deny: a source
that has no values refuses by name rather than fabricating them, and the
executor never materializes a table on an arm's behalf.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import SplitError

__all__ = ("RowBatch", "RowRange", "RowSource", "as_row_source")


class RowSource:
    """The rows one arm owns, named by offset and optionally by value."""

    __slots__ = ()

    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def slice(self, start: int, stop: int) -> RowSource:
        """The sub-source covering ``[start, stop)`` of THIS source's rows.

        Relative indices, absolute offsets: the result carries where it sits
        in the whole table, which is the number an arm slices its own
        resident copy at.
        """
        raise NotImplementedError

    def values(self) -> Sequence[Any]:
        raise SplitError(
            f"{self!r} names its rows by OFFSET and carries no values, and this arm asked for the rows themselves. "
            "An arm that reads rows needs a RowBatch -- pass the materialized rows to execute_split, or wrap them "
            "with as_row_source. An arm that runs from data already held by its own engine should take its row "
            "count from len(source) and its offset from source.start rather than the rows"
        )

    def __iter__(self) -> Iterator[Any]:
        # an arm written against a row iterator reaches the same refusal as
        # one that asks for `values`, rather than silently seeing no rows
        return iter(self.values())


@dataclass(frozen=True, slots=True)
class RowRange(RowSource):
    """Rows named by their offsets in the table, and nothing else.

    The source a split between two engines that each hold the data wants: it
    costs two integers no matter how many rows it names, so a sweep over
    millions of rows allocates nothing per call.
    """

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise SplitError(f"a row range needs 0 <= start <= stop, got [{self.start}, {self.stop})")

    def slice(self, start: int, stop: int) -> RowRange:
        return RowRange(self.start + start, self.start + stop)


@dataclass(frozen=True, slots=True)
class RowBatch(RowSource):
    """Rows the caller has already materialized: the offsets AND the values.

    The rows are WRAPPED rather than copied. Whatever normalization an arm
    performs on its own input is not repeated here, because repeating it
    over the whole table was the cost this seam exists to remove.
    """

    rows: Sequence[Any]
    start: int = 0

    @property
    def stop(self) -> int:
        return self.start + len(self.rows)

    def __repr__(self) -> str:
        # the generated repr would print the entire table, and this one lands
        # in the refusal message above
        return f"RowBatch(len={len(self.rows)}, start={self.start})"

    def slice(self, start: int, stop: int) -> RowBatch:
        if start == 0 and stop == len(self.rows):
            # a degenerate share hands one arm the whole table; copying the
            # caller's sequence to say so is the same cost in miniature
            return self
        return RowBatch(self.rows[start:stop], self.start + start)

    def values(self) -> Sequence[Any]:
        return self.rows


def as_row_source(rows: RowSource | Iterable[Any]) -> RowSource:
    """The seam's argument, from whatever the caller is holding.

    A source passes through untouched -- that is the point, and the reason
    this is not ``list(rows)`` under a better name. A sequence is wrapped in
    place, so a caller who already has the rows pays nothing to hand them
    over. Only an ITERATOR is drained, and it has to be: both arms read from
    one source, and an iterator cannot serve two.
    """
    if isinstance(rows, RowSource):
        return rows
    if isinstance(rows, Sequence):
        return RowBatch(rows)
    return RowBatch(list(rows))
