"""Measured host cost: the model, its staleness rules, and the cache."""

from __future__ import annotations

import json

import pytest

from dau_scheduler import LinkRate, NodeCost, ProfileCache, QueryProfile, host_identity, plan_split, profile_from_spans


def _profile(**overrides: object) -> QueryProfile:
    fields: dict[str, object] = {
        "plan_key": "q",
        "host": "bench/arm64",
        "rows": 1_000,
        "fixed_seconds": 0.0112,
        "per_row_seconds": 3.4e-6,
        "engine_version": "1.42.1",
    }
    return QueryProfile.model_validate({**fields, **overrides})


def test_fixed_cost_is_not_scaled_by_the_row_count() -> None:
    """Plan-time work does not grow with data, and folding it into a per-row
    rate over-estimates a large query by orders of magnitude."""
    profile = _profile()
    assert profile.estimate(1_000) == pytest.approx(0.0112 + 3.4e-3)
    assert profile.estimate(0) == pytest.approx(0.0112)
    # doubling the rows must not double the fixed term
    assert profile.estimate(2_000) - profile.estimate(1_000) == pytest.approx(3.4e-3)


def test_a_profile_refuses_numbers_it_cannot_price_with() -> None:
    with pytest.raises(ValueError, match="carries no information"):
        _profile(rows=0)
    with pytest.raises(ValueError, match="fixed_seconds must be finite"):
        _profile(fixed_seconds=float("nan"))
    with pytest.raises(ValueError, match="per_row_seconds must be finite"):
        _profile(per_row_seconds=-1.0)
    with pytest.raises(ValueError, match="not a fraction"):
        _profile(selectivity=1.4)
    with pytest.raises(ValueError, match="negative rows"):
        _profile().estimate(-1)


def test_a_node_cost_refuses_a_nan() -> None:
    """NaN fails every comparison, so a bare ``< 0`` check admits it and it
    then propagates through every estimate built on it."""
    with pytest.raises(ValueError, match="must be finite"):
        NodeCost(node="scan", seconds=float("nan"), scales_with_rows=True)


def test_a_profile_goes_stale_when_the_row_count_moves_too_far() -> None:
    """Extrapolating one measurement across orders of magnitude is where a
    profile-guided split goes confidently wrong."""
    profile = _profile()
    assert not profile.is_stale_for(3_000, engine_version="1.42.1")
    assert profile.is_stale_for(6_000_000, engine_version="1.42.1")
    assert profile.is_stale_for(10, engine_version="1.42.1")
    assert profile.is_stale_for(0, engine_version="1.42.1")


def test_a_profile_goes_stale_under_a_different_engine() -> None:
    """A measurement taken under one build says nothing about another, and
    this module exists BECAUSE an analytic model of that engine would be
    guesswork."""
    assert _profile().is_stale_for(1_000, engine_version="1.43.2")
    assert not _profile().is_stale_for(1_000, engine_version="1.42.1")
    # exact, not major.minor: a patch release can change a kernel, and the
    # cost of being wrong is a split, while the cost of re-measuring is one
    # host-only run that was going to happen anyway
    assert _profile().is_stale_for(1_000, engine_version="1.42.2")


def test_an_unrecorded_engine_version_is_unknown_and_therefore_stale() -> None:
    """It may have been measured under anything -- defaulting to the installed
    build would let a hand-written profile claim a measurement it never
    made."""
    assert _profile(engine_version="").is_stale_for(1_000, engine_version="1.42.1")


def test_the_profile_prices_a_split() -> None:
    """What the measurement is for: the closed form takes the host figure from
    it instead of the caller inventing one."""
    profile = _profile(fixed_seconds=0.0, per_row_seconds=1e-6)
    split = plan_split("collaborative", device_seconds=0.01, profile=profile, rows=10_000)
    assert split.device_share == pytest.approx(0.5)


def test_overlapping_nodes_are_not_summed_into_elapsed_time() -> None:
    """Summing ``end - start`` double-counts the overlap and drops the gaps:
    with these spans the true elapsed cost is 15 ms and the sum reports 21."""
    spans = [(0, 1_000, "optimization"), (1_000, 11_000, "scan"), (5_000, 15_000, "filter")]
    profile = profile_from_spans(spans, rows=1_000, plan_key="q", host="h", is_fixed=lambda node: node == "optimization")
    assert profile.fixed_seconds == pytest.approx(0.001)
    assert profile.per_row_seconds * profile.rows == pytest.approx(0.014)
    assert profile.estimate(1_000) == pytest.approx(0.015)


def test_gaps_between_nodes_are_real_elapsed_cost() -> None:
    """Anything not attributed to a plan-time node scales, including the time
    the engine spent between the nodes it reported."""
    spans = [(0, 1_000, "scan"), (5_000, 6_000, "filter")]
    profile = profile_from_spans(spans, rows=100, plan_key="q", host="h")
    assert profile.fixed_seconds == 0.0
    assert profile.per_row_seconds * 100 == pytest.approx(0.006)


def test_fixed_nodes_count_their_overlap_once() -> None:
    spans = [(0, 4_000, "optimization"), (2_000, 6_000, "optimization"), (6_000, 10_000, "scan")]
    profile = profile_from_spans(spans, rows=10, plan_key="q", host="h", is_fixed=lambda node: node == "optimization")
    assert profile.fixed_seconds == pytest.approx(0.006)


def test_every_node_is_kept_with_its_own_cost() -> None:
    """The per-node breakdown is what a boundary-choosing planner needs; the
    whole-query figure is what the closed form needs. One measurement serves
    both."""
    spans = [(0, 1_000, "optimization"), (1_000, 3_000, "scan")]
    profile = profile_from_spans(spans, rows=10, plan_key="q", host="h", is_fixed=lambda node: node == "optimization")
    assert [(node.node, node.scales_with_rows) for node in profile.nodes] == [("optimization", False), ("scan", True)]
    assert profile.nodes[1].seconds == pytest.approx(0.002)


def test_a_profile_needs_timings_and_a_row_count() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        profile_from_spans([(0, 1, "scan")], rows=0, plan_key="q", host="h")
    with pytest.raises(ValueError, match="no timing rows"):
        profile_from_spans([], rows=10, plan_key="q", host="h")


def test_the_host_is_part_of_the_identity() -> None:
    """Two machines running the same query differed ~1.6x in one measurement,
    so a profile keyed by plan alone steers a split on the wrong one."""
    assert "/" in host_identity()
    assert profile_from_spans([(0, 1_000, "scan")], rows=10, plan_key="q").host == host_identity()


def test_the_cache_round_trips_by_plan_and_host(tmp_path) -> None:
    cache = ProfileCache(tmp_path)
    assert cache.get("q", "bench/arm64") is None
    path = cache.put(_profile())
    assert cache.get("q", "bench/arm64") == _profile()
    assert cache.get("q", "other/arm64") is None
    assert list(cache.entries()) == [path]
    # human-readable on purpose: it is worth being able to delete one by hand
    assert json.loads(path.read_text())["engine_version"] == "1.42.1"


def test_an_empty_cache_directory_lists_nothing(tmp_path) -> None:
    assert list(ProfileCache(tmp_path / "missing").entries()) == []


def test_the_engine_version_is_not_part_of_the_key(tmp_path) -> None:
    """A host is a key because you cannot re-measure on a machine you are not
    on; an engine version is not, because you can always re-measure under the
    one you have. So it is staleness, not a second slot on disk."""
    cache = ProfileCache(tmp_path)
    cache.put(_profile(engine_version="1.42.1"))
    cache.put(_profile(engine_version="1.43.2"))
    assert len(list(cache.entries())) == 1
    stored = cache.get("q", "bench/arm64")
    assert stored is not None and stored.engine_version == "1.43.2"


def test_a_link_rate_is_keyed_by_host_device_and_transfer_size_and_round_trips(tmp_path) -> None:
    """The rate the cut model prices every byte crossing with is a measured
    fact about one host's link to one device at one transfer size, kept
    beside the host profiles and deletable by hand like them."""
    cache = ProfileCache(tmp_path)
    rate = LinkRate(
        host="bench/x86_64",
        device="engine-a/link-0",
        bytes_per_second=3.2e9,
        transfer_bytes=8 << 20,
        samples=(3.1e9, 3.2e9, 3.3e9),
        measured_at=1.0e9,
    )
    assert cache.get_link_rate("engine-a/link-0", 8 << 20, "bench/x86_64") is None
    path = cache.put_link_rate(rate)
    assert path.name.endswith(".rate.json") and path.parent == tmp_path
    assert cache.get_link_rate("engine-a/link-0", 8 << 20, "bench/x86_64") == rate
    assert cache.get_link_rate("engine-a/link-0", 8 << 20, "other/arm64") is None, "another host's link is another measurement"
    assert cache.get_link_rate("engine-b/link-0", 8 << 20, "bench/x86_64") is None, "and so is another device's"
    assert cache.get_link_rate("engine-a/link-0", 64 << 10, "bench/x86_64") is None, "and so is another transfer size's"
    assert list(cache.entries()) == [], "a link rate is not a plan profile"


def test_a_link_rate_goes_stale_by_age_and_an_undated_one_is_stale() -> None:
    fields: dict[str, object] = {"host": "h", "device": "d", "bytes_per_second": 1e9, "transfer_bytes": 64}
    dated = LinkRate.model_validate({**fields, "measured_at": 1_000.0})
    assert not dated.is_stale(now=1_500.0, max_age_seconds=600.0)
    assert dated.is_stale(now=1_601.0, max_age_seconds=600.0)
    assert LinkRate.model_validate(fields).is_stale(now=1.0, max_age_seconds=1e12), "never dated: re-measure"


@pytest.mark.parametrize(
    "bad",
    [
        {"bytes_per_second": 0.0},
        {"bytes_per_second": float("nan")},
        {"bytes_per_second": float("inf")},
        {"transfer_bytes": 0},
        {"samples": (1.0, 0.0)},
        {"measured_at": -1.0},
        {"measured_at": float("nan")},
    ],
)
def test_a_link_rate_refuses_numbers_it_cannot_price_with(bad) -> None:
    fields: dict[str, object] = {"host": "h", "device": "d", "bytes_per_second": 1e9, "transfer_bytes": 64, "samples": (1e9,)}
    with pytest.raises(ValueError):
        LinkRate.model_validate({**fields, **bad})


def test_a_copied_profile_is_checked_where_it_is_used(tmp_path) -> None:
    """``model_copy(update=...)`` skips validation, so an invalid copy must
    be caught at the use site rather than priced: estimating with it,
    asking whether it is stale, and caching it all refuse."""
    broken = _profile().model_copy(update={"per_row_seconds": float("nan")})
    with pytest.raises(ValueError, match="per_row_seconds"):
        broken.estimate(10)
    with pytest.raises(ValueError, match="per_row_seconds"):
        broken.is_stale_for(10, engine_version="1.42.1")
    with pytest.raises(ValueError, match="per_row_seconds"):
        ProfileCache(tmp_path).put(broken)
    empty = _profile().model_copy(update={"rows": 0})
    with pytest.raises(ValueError, match="carries no information"):
        empty.estimate(10)


def test_a_copied_link_rate_is_checked_where_it_is_used(tmp_path) -> None:
    rate = LinkRate(host="h", device="d", bytes_per_second=1e9, transfer_bytes=64, measured_at=1.0)
    zero = rate.model_copy(update={"bytes_per_second": 0.0})
    with pytest.raises(ValueError, match="bytes_per_second"):
        zero.is_stale(now=2.0, max_age_seconds=10.0)
    with pytest.raises(ValueError, match="bytes_per_second"):
        ProfileCache(tmp_path).put_link_rate(zero)


def test_models_refuse_fields_they_do_not_declare() -> None:
    for build in (
        lambda: QueryProfile.model_validate({"plan_key": "q", "host": "h", "rows": 1, "fixed_seconds": 0.0, "per_row_seconds": 0.0, "extra": 1}),
        lambda: NodeCost.model_validate({"node": "n", "seconds": 0.0, "scales_with_rows": True, "extra": 1}),
        lambda: LinkRate.model_validate({"host": "h", "device": "d", "bytes_per_second": 1.0, "transfer_bytes": 1, "extra": 1}),
    ):
        with pytest.raises(ValueError, match="extra"):
            build()


@pytest.mark.parametrize("spans", [[(10, 5, "reversed")], [(-1, 5, "negative")]])
def test_a_reversed_or_negative_span_is_a_reporting_fault_not_a_cheap_node(spans) -> None:
    with pytest.raises(ValueError, match="not a valid microsecond interval"):
        profile_from_spans(spans, rows=10, plan_key="k")


def test_cache_entries_stay_directly_under_the_root(tmp_path) -> None:
    """A plan key or host is caller text. One containing a slash or ``..``
    must not name a file outside the cache, and two keys that differ only
    in such characters must not collide."""
    cache = ProfileCache(tmp_path)
    for plan_key, host in (("../escape", "h"), ("a/b", "x/y"), ("a_b", "x_y"), ("..", "..")):
        path = cache.put(_profile(plan_key=plan_key, host=host))
        assert path.parent == tmp_path and path.is_file()
        assert cache.get(plan_key, host) == _profile(plan_key=plan_key, host=host)
    assert cache.get("a_b", "x_y") != cache.get("a/b", "x/y") or _profile(plan_key="a_b", host="x_y") == _profile(plan_key="a/b", host="x/y")
    assert len(list(cache.entries())) == 4, "distinct keys are distinct files"
    assert tmp_path.parent.joinpath("escape.h.profile.json").exists() is False


def test_the_cache_root_expands_a_home_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ProfileCache("~/profiles").root == tmp_path / "profiles"
