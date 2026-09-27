"""Resolving a posture into a share, and delivering that share."""

from __future__ import annotations

import pytest

from dau_scheduler import POSTURES, PostureError, WorkSplit, batch_count, plan_split

# One measured pairing, used throughout: the same query took 27.63 ms on the
# host alone and 18.60 ms on the second engine alone, and a share sweep found
# the wall-clock optimum at a 60% device share, 2.33x faster than the host.
HOST_SECONDS, DEVICE_SECONDS = 0.02763, 0.01860
MEASURED_SHARE, MEASURED_SPEEDUP, CLAIMED_ERROR = 0.60, 2.33, 0.07

# The knee of one measured link: below this a transfer loses rate to
# per-transfer setup. It is a property of the link, so it is the caller's
# number and not a default.
KNEE_BYTES = 15 << 20


def test_cpu_and_offload_need_no_estimates() -> None:
    """The definitional postures are the point of offering them: a caller who
    will not pay for cost estimation can still state an intent."""
    assert plan_split("cpu").device_share == 0.0
    assert plan_split("offload").device_share == 1.0


def test_collaborative_balances_the_engines() -> None:
    """The share is where both finish together, so neither waits."""
    split = plan_split("collaborative", cpu_seconds=HOST_SECONDS, device_seconds=DEVICE_SECONDS)
    assert split.device_share == pytest.approx(HOST_SECONDS / (HOST_SECONDS + DEVICE_SECONDS))
    assert split.predicted_speedup == pytest.approx(1 + HOST_SECONDS / DEVICE_SECONDS)
    # host time at the balance point equals device time at it
    host_left = HOST_SECONDS * (1 - split.device_share)
    device_used = DEVICE_SECONDS * split.device_share
    assert host_left == pytest.approx(device_used)


def test_the_closed_form_lands_on_a_measured_optimum() -> None:
    """Constrained against the MEASUREMENT, not against the formula restated.

    Asserting the formula's own output would pass against any formula, so the
    assertions are that the predicted SHARE lands on the measured optimum and
    that the predicted speedup is within the error this model claims.
    """
    split = plan_split("collaborative", cpu_seconds=HOST_SECONDS, device_seconds=DEVICE_SECONDS)
    assert split.device_share == pytest.approx(MEASURED_SHARE, abs=0.01)
    relative_error = abs(split.predicted_speedup - MEASURED_SPEEDUP) / MEASURED_SPEEDUP
    assert relative_error <= CLAIMED_ERROR, f"model error {relative_error:.1%} exceeds the claimed {CLAIMED_ERROR:.0%}"


def test_a_split_pays_least_when_one_engine_dominates() -> None:
    """The reason to check before building any of this: at parity a split
    returns 2x, and against a ten-times-faster engine it buys 1.1x over
    simply handing the whole query over."""
    parity = plan_split("collaborative", cpu_seconds=0.01, device_seconds=0.01)
    lopsided = plan_split("collaborative", cpu_seconds=0.01, device_seconds=0.001)
    assert parity.predicted_speedup == pytest.approx(2.0)
    assert lopsided.predicted_speedup / (0.01 / 0.001) == pytest.approx(1.1)


def test_collaborative_refuses_rather_than_inventing_a_cost_model() -> None:
    """A host cost may be measured, but it is never guessed: with neither a
    figure nor a profile, a fabricated number would sit underneath a real
    decision."""
    with pytest.raises(PostureError, match="needs whole-query estimates"):
        plan_split("collaborative")
    with pytest.raises(PostureError, match="needs whole-query estimates"):
        plan_split("collaborative", cpu_seconds=0.01)


def test_collaborative_rejects_nonpositive_estimates() -> None:
    with pytest.raises(PostureError, match="must be positive"):
        plan_split("collaborative", cpu_seconds=0.0, device_seconds=0.01)


def test_a_measured_profile_supplies_the_host_figure() -> None:
    """The measurement closes the loop the closed form leaves open: a caller
    that profiled its host does not also have to price it by hand."""

    class Measured:
        def estimate(self, rows: int) -> float:
            return 1e-6 * rows

    split = plan_split("collaborative", device_seconds=0.01, profile=Measured(), rows=10_000)
    assert split.device_share == pytest.approx(0.01 / (0.01 + 0.01))
    # an explicit figure still wins: whoever measured the thing itself is the
    # authority on it
    explicit = plan_split("collaborative", cpu_seconds=0.03, device_seconds=0.01, profile=Measured(), rows=10_000)
    assert explicit.device_share == pytest.approx(0.75)


def test_a_profile_prices_a_row_count() -> None:
    class Measured:
        def estimate(self, rows: int) -> float:
            return 1e-6 * rows

    with pytest.raises(ValueError, match="pass rows="):
        plan_split("collaborative", device_seconds=0.01, profile=Measured())


def test_adaptive_refuses_instead_of_falling_back() -> None:
    """A silent fall back to collaborative would answer a different question
    than the caller asked -- adaptive chooses a boundary in the plan, not a
    row share."""
    with pytest.raises(NotImplementedError, match="not a row split"):
        plan_split("adaptive")


def test_offload_is_not_the_wall_clock_optimum() -> None:
    """Recorded because it is counterintuitive and a planner that only
    minimized wall time would pick wrongly: on the measured pairing offload
    reaches 1.49x where collaborative reaches 2.49x, and buys back the host."""
    off = plan_split("offload", cpu_seconds=HOST_SECONDS, device_seconds=DEVICE_SECONDS)
    collab = plan_split("collaborative", cpu_seconds=HOST_SECONDS, device_seconds=DEVICE_SECONDS)
    assert off.predicted_speedup < collab.predicted_speedup
    assert off.device_share > collab.device_share  # more device, less speed


def test_unknown_posture_is_refused() -> None:
    with pytest.raises(PostureError, match="unknown posture"):
        plan_split("turbo")  # type: ignore[arg-type]


def test_split_validates_its_own_fields() -> None:
    with pytest.raises(ValueError, match="not a fraction"):
        WorkSplit(posture="cpu", device_share=1.5, rationale="x")
    with pytest.raises(ValueError, match="unknown posture"):
        WorkSplit(posture="nope", device_share=0.5, rationale="x")


def test_every_posture_is_reachable_or_explicitly_unimplemented() -> None:
    """No posture may be silently absent -- a name in POSTURES that nothing
    handles would fall through to the unknown-posture branch and read as a
    typo rather than a gap."""
    for posture in POSTURES:
        try:
            plan_split(posture, cpu_seconds=0.02, device_seconds=0.01)  # type: ignore[arg-type]
        except NotImplementedError:
            assert posture == "adaptive"
        except PostureError as error:  # pragma: no cover - would be a gap
            pytest.fail(f"posture {posture!r} is declared but unhandled: {error}")


def test_resident_wants_one_batch_at_any_size() -> None:
    """With no transfer to hide, every batch is pure per-call overhead.
    Measured: 5.27 ms at one batch against 5.91 at three."""
    for size in (1 << 20, 48 << 20, 4 << 30):
        assert batch_count(size, resident=True, knee_bytes=KNEE_BYTES) == 1


def test_streamed_reproduces_the_measured_optimum() -> None:
    """45.8 MiB measured fastest at 3 batches (18.78 ms), against 22.08 at one
    and 22.59 at twelve. Targeting the link's knee lands there."""
    assert batch_count(48_009_728, resident=False, knee_bytes=KNEE_BYTES) == 3


def test_the_two_optima_are_opposite() -> None:
    """Which is the entire reason the batch count depends on residency: a
    single tuned constant would be wrong for one of them."""
    payload = 48_009_728
    assert batch_count(payload, resident=False, knee_bytes=KNEE_BYTES) != batch_count(payload, resident=True, knee_bytes=KNEE_BYTES)


def test_small_payloads_are_not_split_below_the_knee() -> None:
    """Splitting below the knee costs transfer rate for a tail that is already
    short."""
    assert batch_count(8 << 20, resident=False, knee_bytes=KNEE_BYTES) == 1
    assert batch_count(0, resident=False, knee_bytes=KNEE_BYTES) == 1


def test_batches_are_never_split_below_the_knee() -> None:
    """Rounding up contradicted the measurement this targets: at 1.5x the knee
    it produced two batches of 0.75x each, in the degraded region."""
    for multiple in (1.0, 1.5, 1.9, 2.5, 3.05, 7.4):
        payload = int(multiple * KNEE_BYTES)
        count = batch_count(payload, resident=False, knee_bytes=KNEE_BYTES)
        assert payload / count >= KNEE_BYTES or count == 1, f"{multiple}x knee -> {count} sub-knee batches"


def test_batch_count_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="negative payload"):
        batch_count(-1, resident=False, knee_bytes=KNEE_BYTES)
    with pytest.raises(ValueError, match="knee_bytes must be positive"):
        batch_count(1 << 20, resident=False, knee_bytes=0)


def test_the_knee_has_no_default() -> None:
    """It is a measurement of one link, and a library-wide constant would be
    wrong for every link that is not the one it was measured on."""
    with pytest.raises(TypeError):
        batch_count(1 << 20, resident=False)  # type: ignore[call-arg]


def test_the_device_fixed_cost_moves_the_balance_and_can_rule_the_split_out() -> None:
    """A device job costs something before its first row. The balance point
    hands the device less by exactly that much, the predicted speedup is
    against the wall that includes it, and when the fixed cost is most of
    the host's whole time no share can win and the answer is host-only."""
    plain = plan_split("collaborative", cpu_seconds=0.013, device_seconds=0.100)
    fixed = plan_split("collaborative", cpu_seconds=0.013, device_seconds=0.100, device_fixed_seconds=0.002)
    assert fixed.device_share == pytest.approx((0.013 - 0.002) / 0.113)
    assert fixed.device_share < plain.device_share
    wall = 0.002 + fixed.device_share * 0.100
    assert wall == pytest.approx(0.013 * (1 - fixed.device_share)), "both engines still finish together"
    assert fixed.predicted_speedup == pytest.approx(0.013 / wall)
    assert fixed.predicted_speedup < plain.predicted_speedup
    assert "fixed cost" in fixed.rationale

    ruled_out = plan_split("collaborative", cpu_seconds=0.013, device_seconds=0.100, device_fixed_seconds=0.013)
    assert ruled_out.posture == "cpu" and ruled_out.device_share == 0.0 and ruled_out.predicted_speedup == 1.0
    assert "no share beats the host alone" in ruled_out.rationale
    with pytest.raises(PostureError, match="non-negative"):
        plan_split("collaborative", cpu_seconds=0.013, device_seconds=0.100, device_fixed_seconds=-0.001)
