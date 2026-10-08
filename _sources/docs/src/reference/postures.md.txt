# Postures and work splits

A posture is an intent; `plan_split` resolves it into a `WorkSplit`.

## Postures

| Posture         | Device share                 | Needs                                                       |
| --------------- | ---------------------------- | ----------------------------------------------------------- |
| `cpu`           | 0.0                          | nothing                                                     |
| `offload`       | 1.0                          | nothing; a speedup is reported if both costs are given      |
| `collaborative` | the balance point, see below | `cpu_seconds` and `device_seconds`, or `profile` and `rows` |
| `adaptive`      | refused with `PostureError`  | a plan boundary, not a row share; planned where the plan is |

`offload` minimizes host occupancy, not wall time: it costs speedup against
`collaborative` and buys back the whole host.

## The balance point

With a whole-query time for each engine and a per-job fixed cost for the
device, the collaborative share solves

```
(1 - s) * cpu = fixed + s * device
```

so `s = (cpu - fixed) / (cpu + device)` and both arms finish together. When no
share beats the host alone, because the fixed cost is already most of the host
time, the result is a `cpu` split whose rationale says so.

## `WorkSplit`

| Field               | Meaning                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| `posture`           | the posture that produced it, which may be `cpu` after a collaborative refusal   |
| `device_share`      | the fraction of rows the device arm takes, from the start of the table           |
| `rationale`         | one sentence saying where the share came from                                    |
| `predicted_speedup` | wall time of the host alone over the predicted wall time, or `None` when unknown |

`WorkSplit.check()` re-runs the invariants; every consumer calls it, because
`model_copy` does not validate.

## Batches

`batch_count(payload_bytes, resident=..., knee_bytes=...)` says how many batches
one engine's share is delivered in. Batching hides the transfer behind the
compute, so resident data gets one batch and a streamed payload gets the count
that keeps each batch above the link's measured transfer knee, rounded down.
The knee is a property of the link and is measured, so there is no default.

## Errors

`PostureError` is raised for an unknown posture, a collaborative request with
no costs, a non-positive estimate, or the adaptive posture.
