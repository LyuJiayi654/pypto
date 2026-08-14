# Single-Node Techniques

A catalog, not a checklist. Each entry says when it applies, what it costs, and — the part
that matters — how to confirm it did anything.

> **Prerequisites:** [The measurement loop](00-methodology.md). Come here with a step from
> that loop already pointing somewhere; applying techniques from a catalog without a
> measurement is how kernels get slower.

## How to read an entry

Every technique below carries the four fields from the previous page. **How to confirm** is
not optional — a change you cannot see in one of the five tools is a change you cannot
defend, and it will be the first thing someone deletes.

## Splitting and parallelism

| Technique | When it applies |
| --------- | --------------- |
| `pl.at(level=...)` | Deciding which hardware level runs a region at all |
| `pl.spmd(N)` | The same kernel over N independent blocks |
| `pl.split(SplitMode)` | One scope mixing cube and vector work |
| `pl.split_aiv(n, mode=)` | `pl.split` cannot express the shape — per-lane addressing |
| `pl.cluster()` | Work that should co-schedule across a cluster |

**`pl.spmd(N)`** — *Cost:* it is an **assertion** that the N blocks are independent. If they
are not, you get a race, not a diagnostic. *Confirm:* the dependency graph
(`enable_dep_gen`) should show N siblings, not a chain.

**`pl.split(SplitMode)`** — the cube/vector overlap covered in
[Mixed kernels](../tutorials/03-mixed-kernel.md). *Cost:* the cross-core ring is sized in
whole tiles at the boundary, so it can push a kernel past the vector budget — pair it with
`pl.cross_core_slot`. *Confirm:* the L2 swimlane should show cube and vector spans
overlapping rather than abutting.

> **Fatal pitfall:** `pl.split` halves only the **vector** sub-region; the cube sub-region
> stays full-sized. If you picked a mode expecting each unit to take half a tile, the
> measurement will not match the mental model.

## Pipelining and unrolling

| Technique | When it applies |
| --------- | --------------- |
| `pl.pipeline(..., stage=N)` | A loop whose iterations can overlap in flight |
| `pl.unroll` | A short loop where per-iteration overhead dominates |
| `pl.cross_core_slot(slot_num=N)` | Sizing the cube↔vector ring |

**`pl.pipeline`** — *Cost:* buffers, multiplied. Overlapping N stages means N live copies of
the staged tiles. *Confirm:* the [memory map](00-methodology.md) shows the extra copies;
if the allocation does not grow, the pipeline did not take.

**`pl.cross_core_slot(slot_num=N)`** — *Cost / benefit both:* a deeper ring lets the
producer run further ahead before blocking, and costs `N × tile` of vector buffer. The
default is 8 slots, which for a `[128, 128]` FP32 tile is 512 KB against a ~184 KB budget —
so on real shapes this is usually a *reduction*, not an increase. **Pick the largest depth
that fits.** *Confirm:* it compiles at all; then the swimlane, for whether the producer
still stalls.

## The matmul path

| Technique | When it applies |
| --------- | --------------- |
| `AutoTileMatmulL0` | Automatic — a matmul that does not fit the cube's L0 buffers |
| `enable_pypto_l0c_double_buffer` | L0c has headroom and the matmul is accumulation-bound |
| `a_trans` / `b_trans` | An operand whose layout does not match what the cube wants |
| Split-K + atomic add | K is large and M/N cannot fill the cores |

**`AutoTileMatmulL0`** runs whether you ask or not, which is worth knowing precisely because
it makes `pl.matmul` on tensor operands **not one instruction** — it is a loop nest the pass
wrote. Hand-blocking K (see [Tiled matmul](../tutorials/02-matmul.md)) overrides its choice
on that axis only.

**Split-K** — *Cost:* **accumulation order across cores is not fixed**, so repeated runs may
differ in the last bits, and the output must be zeroed first. *Confirm:* the graph should
show the K-slices as siblings; and check the tolerance your test uses still holds.

## Memory

| Technique | When it applies |
| --------- | --------------- |
| `target_memory=` on `pl.create_tile` | A tile the compiler places somewhere unhelpful |
| `memory_planner=` | Choosing between PyPTO's planner and PTOAS's |
| Data residency | The same weights are used across many launches |

**`memory_planner=PTOAS`** hands allocation to PTOAS, which skips PyPTO's `MemoryReuse` and
`AllocateMemoryAddr`. *Cost:* different failure modes and a different set of planner bugs;
the semantics-required aliases still run either way. *Confirm:* the memory map, before and
after — this changes the allocation, which is exactly what that tool draws.

**Data residency** — `pypto.runtime.DeviceTensor` keeps a tensor on the device across
launches, removing a host-to-device copy per call. *When:* weights, KV caches, anything
whose contents outlive one launch. *Confirm:* the **host** spans in the benchmark tree
shrink; the device spans should not move.

## Scheduling

These are the subject of [Shaping the task graph](../tutorials/04-task-graph.md); listed
here for completeness with their costs.

| Technique | Changes correctness | Cost |
| --------- | ------------------- | ---- |
| `no_dep` / `manual_dep=True` / `manual_scope` | **Yes** | An assertion the compiler cannot check |
| `predicate=` | No | Only `tensor[indices] OP int-literal` is expressible |
| `allow_early_resolve=` | No | Consumers pre-stage only when *all* producers are flagged |
| `pl.system.task_dummy` | No | One extra task, to collapse a fan-in |
| `ring_task_window` / `ring_heap` / `ring_dep_pool` | No | Runtime memory |

*Confirm, for all of them:* `enable_dep_gen` for the shape of the graph, the swimlane for
whether the shape turned into overlap. Ring sizing is the one case where `scope_stats`
answers directly — if a peak was not at capacity, raising that ring cannot help.

## Paying setup once

Worker setup is per-worker, not per-program. Registering several programs against one
worker removes a full setup from every run after the first — see
`examples/runtime/multi_program_kv_cache.py`, where a prefill and a decode program share one
KV cache and one worker.

*Confirm:* the host spans, again. This never shows up in device time.

## Order of attack

When the measurement points at the device and you have several of these available:

1. **Fix what `perf_hints` flagged.** A `PH001` tile granularity issue caps everything else.
2. **Fix the graph** before the kernels — a serialized graph wastes whatever the kernels save.
3. **Overlap the units** (`pl.split`) before micro-tuning either one.
4. **Then** pipeline depth, ring depth, allocation.

Reversing 2 and 3 is the common mistake: a beautifully mixed kernel still runs alone if
nothing else in the graph is allowed to run beside it.

## See Also

- [The measurement loop](00-methodology.md) — decide *what* to fix first.
- [Distributed performance](02-distributed.md) — what changes with more than one rank.
- [Mixed kernels](../tutorials/03-mixed-kernel.md) — `pl.split` hands-on.
- [Shaping the task graph](../tutorials/04-task-graph.md) — the scheduling controls.
