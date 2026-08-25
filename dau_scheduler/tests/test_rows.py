"""Naming an arm's rows: offsets always, values only when someone reads them."""

from __future__ import annotations

import pytest

from dau_scheduler import RowBatch, RowRange, SplitError, as_row_source


def test_a_range_is_two_integers_at_any_row_count() -> None:
    """The source a split between two engines that each hold the data wants:
    six million rows and not one of them exists."""
    source = RowRange(0, 6_000_000)
    assert len(source) == 6_000_000
    device, host = source.slice(0, 3_600_000), source.slice(3_600_000, 6_000_000)
    assert (device.start, len(device)) == (0, 3_600_000)
    assert (host.start, len(host)) == (3_600_000, 2_400_000)


def test_a_slice_carries_absolute_offsets() -> None:
    """Relative indices in, absolute offsets out: the offset is the number an
    arm slices its own resident copy at, so it has to name the whole table."""
    inner = RowRange(1000, 2000).slice(100, 200)
    assert (inner.start, inner.stop) == (1100, 1200)
    batch = RowBatch(list(range(1000)), start=500).slice(100, 200)
    assert (batch.start, batch.stop) == (600, 700)
    assert batch.values() == list(range(100, 200))


def test_a_range_refuses_the_arm_that_reads_values() -> None:
    """Refuse rather than approximate, and refuse rather than materialize the
    table behind the caller's back to keep an arm fed."""
    with pytest.raises(SplitError, match="names its rows by OFFSET and carries no values") as refusal:
        RowRange(0, 500).values()
    # the refusal names both remedies
    assert "RowBatch" in str(refusal.value)
    assert "len(source)" in str(refusal.value)


def test_iterating_a_range_reaches_the_same_refusal() -> None:
    """An arm written against a row iterator would otherwise see no rows and
    return a confident empty answer."""
    with pytest.raises(SplitError, match="carries no values"):
        list(RowRange(0, 500))


def test_a_range_cannot_be_inverted() -> None:
    with pytest.raises(SplitError, match=r"0 <= start <= stop"):
        RowRange(40, 12)


def test_a_batch_repr_does_not_print_the_table() -> None:
    """It lands in the refusal message above, and a six-million-row repr is
    not a message."""
    assert repr(RowBatch(list(range(1_000)))) == "RowBatch(len=1000, start=0)"


def test_a_sequence_is_wrapped_in_place() -> None:
    """A caller who already has the rows pays nothing to hand them over --
    identity is what pins that, because a copy compares equal and fails only
    ``is``."""
    rows = [(index, index * 2) for index in range(50)]
    assert as_row_source(rows).values() is rows
    assert as_row_source(RowRange(0, 10)) == RowRange(0, 10)


def test_a_whole_table_slice_does_not_copy() -> None:
    """A degenerate share hands one arm everything; copying the caller's
    sequence to say so is the same cost in miniature."""
    batch = RowBatch([(index,) for index in range(50)])
    assert batch.slice(0, 50) is batch


def test_an_iterator_is_drained_because_it_cannot_serve_two_arms() -> None:
    rows = iter([(index,) for index in range(10)])
    source = as_row_source(rows)
    assert len(source) == 10
    assert len(source.slice(0, 4)) == 4
