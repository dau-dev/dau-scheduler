# Measure host cost

`plan_split` refuses to guess the host side of a `collaborative` split. This
guide measures it from a query's first run and caches the measurement so later
runs are priced from it.

## Build a profile from the engine's timings

A profiling query engine reports per-node spans. Pass them as
`(start_us, end_us, node)` tuples with the row count they were measured over,
and name the nodes whose cost does not grow with rows.

```python
from dau_scheduler import profile_from_spans

engine_timings = [(0, 11_200, "optimization"), (11_200, 12_900, "scan"), (12_900, 14_600, "filter")]
profile = profile_from_spans(
    engine_timings,
    rows=1_000,
    plan_key="revenue-by-day",
    is_fixed=lambda node: node.startswith("optimization"),
    engine_version="1.42.1",
)
```

`is_fixed` is what keeps plan-time cost separate from per-row cost. An
optimizer does not get slower with more rows, and folding it into a per-row rate
over-estimates a large query by orders of magnitude.

The profile records the host it was measured on (`host_identity()` by default).
A speedup ratio means nothing without the machine it was measured on, so the
host is part of the key.

## Price a run from it

```python
from dau_scheduler import plan_split

split = plan_split("collaborative", device_seconds=0.0186, profile=profile, rows=6_000_000)
```

An explicit `cpu_seconds` still wins over a profile: a caller who measured the
thing itself is not overridden by a model of it.

## Cache it

A `ProfileCache` is a directory of JSON files keyed by plan and host. Small,
readable, and deletable by hand when a measurement goes stale.

```python
from dau_scheduler import ProfileCache

cache = ProfileCache("profiles")
cache.put(profile)
later = cache.get("revenue-by-day")
```

The engine version is deliberately not part of the key. You cannot re-measure
on a machine you are not on, so the host is a key; you can always re-measure
under the engine you have, so the version is a staleness question instead.

## Know when to re-measure

One predicate answers both ways a measurement goes stale: the row count moved
too far from the measured one (a factor of ten by default), or the engine is a
different build.

```python
if profile.is_stale_for(6_000_000, engine_version="1.42.1"):
    pass  # re-measure rather than extrapolate across orders of magnitude
```

## Link rates

A streamed split prices every byte that crosses the link. `LinkRate` is the
measured transfer rate of one link, keyed by host, device and transfer size,
because what a link sustains depends on the host, the slot, the driver and the
size of the transfer. The cache stores them beside the profiles
(`put_link_rate`, `get_link_rate`), and `LinkRate.is_stale` judges age, since a
driver change or a reboot cannot invalidate an entry by itself.
