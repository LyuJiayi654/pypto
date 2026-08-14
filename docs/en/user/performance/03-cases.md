# Worked Cases

Six times through the loop. Each one names the symptom, the tool that located it, the
change, and how the change was confirmed.

> **Prerequisites:** [The measurement loop](00-methodology.md).

## How to read these

Each case gives four things and deliberately withholds a fifth:

| Given | Why |
| ----- | --- |
| **Symptom** | What you would actually observe |
| **Located by** | Which tool, and what it showed |
| **Change** | The edit |
| **Confirmed by** | Which tool, and what should move |

**No speedup figures.** They depend on your shapes, platform and version — a number here
would be stale and misleading, and you would have no way to tell. The *confirmation step*
is the transferable part: run it on your own kernel and you get your own number.

## Single-node

### Case 1: the compiler already knew

**Symptom.** A vector kernel is slower than its arithmetic suggests, across every shape.

**Located by.** Step 1 — `<output_dir>/report/perf_hints.log` carries a `PH001`
`TileInnermostDimGranularity` hint. Nothing further down the loop was needed, and the
stderr summary had been printed on every compile.

**Change.** Reshape the tile so its innermost dimension lands on the granularity the
hardware moves.

**Confirmed by.** The hint disappears from `perf_hints.log`. Then step 2 to see the segment
actually shrink — a hint is advisory, so its absence proves the pattern is gone, not that
it mattered.

### Case 2: the time was never on the device

**Symptom.** A kernel that "should" be fast dominates a benchmark loop.

**Located by.** Step 2 — `stats.print_mean_tree(spread="stdev")` shows the weight in
**host** spans, not device.

**Change.** Stop paying setup per call: register the programs against one worker, and keep
unchanging weights resident with `pypto.runtime.DeviceTensor`.

**Confirmed by.** The host spans shrink and the device spans do not move. If device time
changes too, something else was edited.

### Case 3: fused in name only

**Symptom.** A matmul-then-bias kernel is roughly the sum of its parts.

**Located by.** Step 5 — the L2 swimlane shows the cube and vector spans **abutting** rather
than overlapping. Two scopes, two dispatches.

**Change.** One `pl.at` scope with `optimizations=[pl.split(pl.SplitMode.UP_DOWN)]` —
[Mixed kernels](../tutorials/03-mixed-kernel.md).

**Confirmed by.** The swimlane again: the spans should now overlap. Expect to also need
`pl.cross_core_slot(slot_num=N)` — the default 8-slot ring does not fit a `[128, 128]` FP32
crossing tile, and compilation says so.

### Case 4: parallel graph, serial execution

**Symptom.** Sibling tasks that should overlap run one after another.

**Located by.** Step 5 — `enable_dep_gen=True`, and the rendered graph is a chain where a
fan-out was expected. The OverlapMap saw two tasks writing disjoint regions of one tensor
and could not prove disjointness.

**Change.** Opt the argument out at the narrowest scope that expresses the claim —
`pl.at(..., no_dep_args=[t])` before `manual_dep=True` before `manual_scope`.

**Confirmed by.** The graph becomes a fan-out, *and* the swimlane shows real overlap. Both,
because a fan-out in the graph still executes serially if a ring is saturated.

> **Fatal pitfall:** every opt-out is an assertion the compiler cannot check. If the regions
> are not actually disjoint, this case's "fix" is a race that reproduces intermittently.

## Distributed

### Case 5: one rank sets the pace

**Symptom.** Scaling out stops paying off; the mean per-round time barely improves.

**Located by.** `per_rank("device")` — one rank is consistently slower, so every collective
waits on it. `per_round` alone would have shown only a flat mean.

**Change.** Depends on the cause, and the point is that you now know to look at that rank:
uneven sharding, a rank doing extra host work, or arriving late.

**Confirmed by.** `per_rank` spread narrows. Only then is `per_round` worth optimising.

### Case 6: collective memory, not collective time

**Symptom.** Adding ranks makes allocation fail rather than making the run slower.

**Located by.** The default `mode="mesh"` needs **O(P)** windows — direct exchange with a
window per peer.

**Change.** `mode="ring"`: chunked reduce-scatter + allgather, **O(1)** windows. Not a
one-argument edit — see [Distributed](02-distributed.md) for what else the call needs.

**Confirmed by.** It allocates. Then `per_round` — `ring` pays `2(P−1)` sequential steps, so
for few ranks or small payloads it can be slower than the `mesh` it replaced. Check rather
than assume.

## The pattern across all six

| Case | Step that found it |
| ---- | ------------------ |
| 1 | 1 — compile-time hints |
| 2 | 2 — host vs device split |
| 3, 4 | 5 — graph and swimlane |
| 5, 6 | Distributed metrics, before any tuning |

Four of six were located in steps 1, 2 or 5 — the cheap ones. Only after those does the
per-instruction view in step 3 earn its cost.

## See Also

- [The measurement loop](00-methodology.md) — the five steps.
- [Single-node techniques](01-single-node.md) / [Distributed](02-distributed.md) — the
  catalogs these cases draw from.
- [Precision cases](../precision/01-cases.md) — the same treatment for wrong answers.
