# dau scheduler

Split row-parallel dataframe work across two compute engines and merge the partial results

[![Build Status](https://github.com/dau-dev/dau-scheduler/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/dau-dev/dau-scheduler/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/dau-dev/dau-scheduler/branch/main/graph/badge.svg)](https://codecov.io/gh/dau-dev/dau-scheduler)
[![License](https://img.shields.io/github/license/dau-dev/dau-scheduler)](https://github.com/dau-dev/dau-scheduler)
[![PyPI](https://img.shields.io/pypi/v/dau-scheduler.svg)](https://pypi.python.org/pypi/dau-scheduler)

## Overview

When a machine has two engines that can both run the same query — two
processors, a processor and an accelerator, a local worker and a remote one —
the usual choice is all-or-nothing: run it here, or send it there. Running it in
both places at once is often faster than either, but only for some queries, and
only at the right split.

`dau-scheduler` makes that decision and makes the result correct. It answers
three questions:

- **Should this be split, and by how much?** A closed form over each engine's
  measured throughput gives the share of rows that finishes both arms at the
  same time. When the engines are comparable it approaches a 2x speedup; when
  one is far faster than the other it correctly declines to bother.
- **Can this query survive being split?** Splitting rows means each engine
  computes a partial answer, and not every result can be reassembled from
  partials. Sums and counts add. A mean needs its sum and count carried
  separately rather than averaged. A top-k merges two k-lists. A grouped
  aggregation needs a key merge with the group straddling the boundary folded
  back together. A distinct count cannot be combined at all without the
  elements themselves. This library writes that taxonomy down and enforces it.
- **What if it can't?** Refusal is by name and default-deny. A result the merge
  step does not know how to combine is refused, explicitly and with a reason,
  rather than silently mis-merged into a wrong answer.

Two commitments shape the rest of the design. **Rows, not stages** — splitting a
query by pipeline stage ships every intermediate across the link between the
engines, so the split is always over rows. And **cost is measured, not
assumed** — host cost is profiled per `(plan, host)` pair and re-measured when
the profile goes stale, because a speedup ratio is meaningless without naming
the machine it was measured on.

The library is engine-agnostic. Both arms are injected as callables, and nothing
here knows what is on the other end.

## Status

Early. This repository is a scaffold; the placement logic has not landed yet.
