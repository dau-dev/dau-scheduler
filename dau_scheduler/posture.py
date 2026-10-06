"""How much of a query each engine should carry.

A caller with two engines has one number to choose: the fraction of rows the
second engine runs. Everything else follows from it.

**The postures optimize different things and are not quality tiers.** On a
measured pairing (the collaborative sweep of 2026-08-12 in the benchmarks
ledger) the host alone took 27.6 ms and the device alone 18.6 ms; the split
finished in 11.9 ms. Running everything on the device therefore takes 1.57x
the wall clock of splitting, and in exchange hands the host back entirely.
A planner that only minimized wall time would pick wrongly for a caller whose
processors have other work to do.

The closed form under ``collaborative`` is the balance point where both
engines finish together::

    device_share = T_cpu / (T_cpu + T_device)
    speedup      = 1 + T_cpu / T_device

which needs no search -- one cost estimate per side and the share follows.

The speedup expression is also the reason to check whether a split is worth
building at all: **collaborative placement pays most when the engines are
comparable.** From the form above (these are analytic values, not
measurements): at parity it returns 2x; against an engine ten times faster
than the host it buys 1.1x over simply handing the whole query over, which
is unlikely to repay the machinery.

**The share is a fraction of ROWS, not of pipeline stages.** Both engines run
the whole query over disjoint row ranges and their partial results combine
(see :mod:`dau_scheduler.merge`). Splitting by stage instead would send every
intermediate result across the link between the engines, which is the round
trip a split exists to avoid -- and it makes the second engine wait for the
first rather than running beside it.
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, model_validator

from .errors import PostureError

__all__ = ("POSTURES", "HostCost", "Posture", "PostureError", "WorkSplit", "batch_count", "plan_split")

Posture = Literal["cpu", "offload", "collaborative", "adaptive"]

POSTURES: tuple[str, ...] = ("cpu", "offload", "collaborative", "adaptive")


@runtime_checkable
class HostCost(Protocol):
    """Anything that can price a row count on the engine the caller is on.

    :class:`dau_scheduler.profile.QueryProfile` is one, measured rather than
    modelled; a caller with an analytic model of its own host can pass that
    instead.
    """

    def estimate(self, rows: int) -> float: ...


class WorkSplit(BaseModel):
    """How one query's rows divide between the two engines."""

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}

    posture: str
    device_share: float
    rationale: str
    predicted_speedup: float | None = None

    def check(self) -> WorkSplit:
        """Re-run the invariants and return ``self``.

        Construction validates, but ``model_copy(update=...)`` does not, so
        every consumer calls this before acting on a split rather than
        trusting that the instance it was handed went through ``__init__``.
        """
        if not math.isfinite(self.device_share) or not 0.0 <= self.device_share <= 1.0:
            raise ValueError(f"device_share {self.device_share} is not a fraction")
        if self.posture not in POSTURES:
            raise ValueError(f"unknown posture {self.posture!r}; known: {POSTURES}")
        if self.predicted_speedup is not None and (not math.isfinite(self.predicted_speedup) or self.predicted_speedup <= 0):
            raise ValueError(f"predicted_speedup must be finite and positive, got {self.predicted_speedup}")
        return self

    @model_validator(mode="after")
    def _share_is_a_fraction(self) -> WorkSplit:
        return self.check()


def plan_split(
    posture: Posture = "collaborative",
    *,
    cpu_seconds: float | None = None,
    device_seconds: float | None = None,
    profile: HostCost | None = None,
    rows: int | None = None,
    device_fixed_seconds: float = 0.0,
) -> WorkSplit:
    """Resolve a posture into a row split.

    ``cpu_seconds`` and ``device_seconds`` are whole-query estimates for each
    engine alone. Only ``collaborative`` needs them; the other postures are
    definitional and take none, which is the point of offering them -- a
    caller who does not want to pay for cost estimation can still express an
    intent.

    The host figure may instead come from a measurement: pass a
    :class:`HostCost` and the ``rows`` this run will process, and the estimate
    is taken from it. An explicit ``cpu_seconds`` still wins -- a caller who
    measured the thing itself should not be overridden by a model of it.

    What is refused is guessing. With neither a figure nor a profile,
    ``collaborative`` declines to invent one, because a fabricated number
    underneath a real decision is worse than no decision: the split changes
    which engine runs which rows, so an error in it is self-reinforcing.

    ``device_fixed_seconds`` is what one device job costs before it has
    processed a row -- launching it, polling it, reading its records -- a
    per-invocation term the proportional form omitted and one measurement
    put at a few milliseconds beside a 12.3 ms host time (2026-09-27). With it
    the balance is ``(1 - s) * cpu = fixed + s * device``, and a share that
    cannot beat the host alone (the fixed cost is already most of the host
    time) is a ``cpu`` split that says why, not a split predicted to win by
    a margin the fixed cost eats.
    """
    if cpu_seconds is None and profile is not None:
        if rows is None:
            raise ValueError("a profile prices a row count; pass rows= alongside profile=")
        cpu_seconds = profile.estimate(rows)
    if posture == "cpu":
        return WorkSplit(
            posture=posture,
            device_share=0.0,
            rationale="host-only by request; the second engine is left idle",
            predicted_speedup=1.0,
        )

    if posture == "offload":
        # NOT the wall-clock optimum, and deliberately so: this posture
        # minimizes host occupancy, which is a different objective. It costs
        # real speedup against `collaborative` and buys back the whole host.
        speedup = None if not (cpu_seconds and device_seconds) else cpu_seconds / device_seconds
        return WorkSplit(
            posture=posture,
            device_share=1.0,
            rationale="device-only by request; frees the host at a cost in wall time",
            predicted_speedup=speedup,
        )

    if posture == "collaborative":
        if cpu_seconds is None or device_seconds is None:
            raise PostureError(
                "collaborative needs whole-query estimates for both engines (cpu_seconds, device_seconds). "
                "Supply them, or pass profile=/rows= -- a profile measures the host side from a plan's first "
                "execution, which is host-only anyway"
            )
        if cpu_seconds <= 0 or device_seconds <= 0:
            raise PostureError(f"estimates must be positive, got cpu={cpu_seconds}, device={device_seconds}")
        if device_fixed_seconds < 0:
            raise PostureError(f"device_fixed_seconds must be non-negative, got {device_fixed_seconds}")
        share = (cpu_seconds - device_fixed_seconds) / (cpu_seconds + device_seconds)
        if share <= 0:
            return WorkSplit(
                posture="cpu",
                device_share=0.0,
                rationale=(
                    f"no share beats the host alone: one device job costs {device_fixed_seconds * 1e3:.2f} ms before its first row "
                    f"and the host finishes in {cpu_seconds * 1e3:.2f} ms"
                ),
                predicted_speedup=1.0,
            )
        wall = device_fixed_seconds + share * device_seconds
        fixed_note = f" after {device_fixed_seconds * 1e3:.2f} ms of device fixed cost" if device_fixed_seconds else ""
        return WorkSplit(
            posture=posture,
            device_share=share,
            rationale=(f"balance point: host {cpu_seconds * 1e3:.2f} ms, device {device_seconds * 1e3:.2f} ms{fixed_note}, both finish together"),
            predicted_speedup=cpu_seconds / wall,
        )

    if posture == "adaptive":
        # A refusal, not an unfinished branch: adaptive chooses a BOUNDARY
        # in the plan rather than a row share, so it cannot be expressed as
        # a WorkSplit and needs inputs this signature does not carry
        # (candidate cuts and a per-node cost). The consumer that holds the
        # plan answers it; answering it here with a row split would silently
        # substitute a different decision.
        raise PostureError(
            "adaptive is not a row split: it chooses where to cut the plan, which needs candidate cuts and a "
            "profiled per-node cost rather than two whole-query figures; plan it where the plan is"
        )

    raise PostureError(f"unknown posture {posture!r}; known: {POSTURES}")


def batch_count(payload_bytes: int, *, resident: bool, knee_bytes: int) -> int:
    """How many batches one engine's share should be delivered in.

    **Batching exists to hide the transfer behind the compute.** Each batch's
    transfer overlaps the previous batch's work, so the pipelined cost falls
    toward the transfer time alone. With the data already RESIDENT on the
    engine there is no transfer to hide and every batch is pure per-call
    overhead, so the answer is one.

    Measured on one link with a 45.8 MiB payload:

    ==========  ==========  ==========
    batches     streamed    resident
    ==========  ==========  ==========
    1           22.08 ms    **5.27 ms**
    3           **18.78**   5.91 ms
    12          22.59 ms    --
    ==========  ==========  ==========

    The optima are opposite, so a single tuned batch count is wrong for one of
    them -- which is why this takes ``resident`` rather than a constant.

    Streamed, the count targets the link's transfer knee rather than a fitted
    model. ``knee_bytes`` is the payload below which a transfer loses rate to
    per-transfer setup; it is a property of the link and has to be measured on
    it, so there is no default here. Fitting ``T(B) = transfer + work/B + c*B``
    to the measured points above gives ``c`` anywhere from 83 to 568
    microseconds depending on which pair is fitted, so the model does not
    generalize and the measured knee is the honest thing to encode. At a
    15 MiB knee, 45.8 MiB rounds to 3, which is what measured fastest.
    """
    if payload_bytes < 0:
        raise ValueError(f"negative payload {payload_bytes}")
    if knee_bytes <= 0:
        raise ValueError(f"knee_bytes must be positive, got {knee_bytes}")
    if resident or payload_bytes == 0:
        return 1
    # FLOOR, not round: rounding up splits an above-knee payload into
    # below-knee batches, contradicting the measurement this targets. At 1.5x
    # the knee, rounding gave two batches at 0.75x the knee each.
    return max(1, payload_bytes // knee_bytes)
