# The Measurement Loop

Five questions, in order. Each one has a tool, and each answer decides whether the next
question is worth asking.

> **Prerequisites:** [Tuning the schedule](../tutorials/05-scheduling-tuning.md) covers the
> scheduling half of this hands-on; this page is the whole loop.

## Why the order matters

Every one of these tools will happily produce a number. The order exists because a number
from step 3 means nothing if step 1 already told you the kernel is written badly, and a
scheduling fix in step 5 is wasted if step 2 shows the time is going to the host.

```text
Performance is below expectation
├─ 1. Did the compiler already say something?  → report/perf_hints.log
├─ 2. Which segment is it?                     → benchmark span tree (host vs device)
├─ 3. Inside one kernel?                       → incore msprof, per-instruction
├─ 4. Is a resource full or wasted?            → memory map, scope stats
└─ 5. Is the schedule serialized?              → dependency graph, L2 swimlane
```

Work down the list, and stop at the first step that explains the gap.

## The tools

| Step | Observes | Entry point | Output |
| ---- | -------- | ----------- | ------ |
| 1 | Code patterns | Automatic during `compile()` / `@pl.jit` | `<output_dir>/report/perf_hints.log` |
| 2 | End-to-end segments | `pypto.runtime.benchmark` | `stats.print_mean_tree(...)` |
| 3 | One kernel, cycle level | Ascend op-simulator (`msprof`) | Insight trace |
| 4 | On-chip buffers | `python -m pypto.tools.memory_map DUMP.py` | HTML |
| 4 | Runtime ring levels | `RunConfig(enable_scope_stats=True)` | `dfx_outputs/scope_stats/scope_stats.jsonl` |
| 3 | Per-pipe utilisation | `RunConfig(enable_pmu=2)` | `dfx_outputs/pmu.csv` |
| 5 | Task graph, concurrency | `RunConfig(enable_dep_gen / enable_l2_swimlane)` | `dfx_outputs/*` |

## Step 1: read what the compiler already told you

Performance hints are emitted during compilation — you do not turn them on. With a report
instrument in the context (always the case through `compile()` and `@pl.jit`), every hint is
appended to `perf_hints.log`, and stderr gets a one-line summary pointing at the file:

```text
[perf_hint] N hints across M sites; see <path>/perf_hints.log
```

The release default for `PYPTO_LOG_LEVEL` is `INFO`, so that summary reaches the console
without any setup. `PYPTO_LOG_LEVEL=warn` mutes it — the file still gets the detail.

The first check to know by name is **`PH001` `TileInnermostDimGranularity`**: a tile whose
innermost dimension does not land on the granularity the hardware moves. It is a code
pattern, not a scheduling problem, so no amount of tuning further down this list will
recover it. See [Diagnostics](../../dev/passes/92-diagnostics.md) for the full registry and
for `disabled_diagnostics`.

## Step 2: find the segment before optimising anything

`benchmark` registers a compiled program once and dispatches N timed launches, so the
numbers exclude setup:

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, args, rounds=100, warmup=3)
stats.print_mean_tree(spread="stdev")
```

`warmup=3` matters: the first launches pay one-time costs that would otherwise be averaged
into every round.

The tree separates **host** from **device** spans. That split is the point of this step:

- Time in device spans → continue to step 3 or 5.
- Time in host spans → no kernel change will help. Look at dispatch overhead, argument
  marshalling, or setup you are paying per call instead of once (see
  [Single-node techniques](01-single-node.md)).

`spread=` selects what is printed next to each mean; `stdev` is a good default because a
large spread usually means you are measuring something other than the kernel.

## Step 3: inside one kernel

When a device span dominates and it is one kernel, the op-simulator gives per-instruction
and per-pipe timing, viewable as an Insight trace. This is the only tool here that shows
*which pipe* is the limit rather than *that* the kernel is slow.

`RunConfig(enable_pmu=2)` (`PIPE_UTILIZATION`) is the cheaper approximation, writing
`dfx_outputs/pmu.csv` — use it to decide whether a full simulator run is worth it.

## Step 4: is a resource full, or wasted?

Two different resources, two tools:

**On-chip buffers** — `pypto.tools.memory_map` renders the allocation as HTML: address
across, lifetime down, with the IR alongside. Its input is a **pass dump**, not a run, so
ask for one at compile time first:

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

prog = kernel.lower(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

Read it for tiles alive longer than they need to be, and for the headroom that decides
whether a deeper pipeline or a deeper cross-core ring will fit.

**Runtime rings** — `RunConfig(enable_scope_stats=True)` records per-scope peaks for
`task_window`, `heap`, and `tensormap`. A peak sitting at capacity is a ceiling; peaks well
below it mean the rings are not your problem and you should not be resizing them.

## Step 5: is the schedule the limit?

Covered hands-on in [Tuning the schedule](../tutorials/05-scheduling-tuning.md). In short:
`enable_dep_gen=True` to see the graph the runtime built, `enable_l2_swimlane=True` to see
whether tasks actually overlapped.

> **Simulator caveat:** on `*sim` platforms the swimlane is single-pass and emits only
> `l2_swimlane_records.json`. On an onboard platform the same flag runs the workload
> **twice**, because collection perturbs timing — so never read wall-clock from a
> swimlane-enabled onboard run.

## Evaluating a change

Each technique in the next two pages is described with the same four fields, because a
speedup with an unstated cost is not a result:

| Field | What it answers |
| ----- | --------------- |
| **When it applies** | The shape or pattern that has to hold |
| **Cost** | What you give up — determinism, memory, readability, portability |
| **How to enable** | The exact flag, keyword, or rewrite |
| **How to confirm** | Which of the tools above shows it worked |

The fourth is the one people skip. A change that cannot be confirmed by one of the tools
above is a guess, and guesses accumulate.

## See Also

- [Single-node techniques](01-single-node.md) — the catalog for one device.
- [Distributed performance](02-distributed.md) — what changes with more than one rank.
- [Worked cases](03-cases.md) — the loop applied end to end.
- [Diagnostics](../../dev/passes/92-diagnostics.md) — the compile-time hint registry.
