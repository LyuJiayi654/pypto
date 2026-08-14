# Performance

Measure, locate, change, confirm — in that order, on one device or many.

> **Prerequisites:** [Tutorials](../tutorials/index.md), at least through
> [Tuning the schedule](../tutorials/05-scheduling-tuning.md).

## What this chapter is

Not a list of tricks. A **loop**: five questions that narrow a gap to one cause, then a
catalog of techniques each stated with its cost and — the part that keeps a codebase
honest — how to confirm it worked.

The loop is the same whether you have one rank or sixty-four; only the dominant cost
changes.

## Contents

| Page | Covers |
| ---- | ------ |
| [The measurement loop](00-methodology.md) | Five questions, their tools, and why the order matters |
| [Single-node techniques](01-single-node.md) | Splitting, pipelining, the matmul path, memory, scheduling |
| [Distributed performance](02-distributed.md) | Rank skew, collective algorithms, overlap, residency |
| [Worked cases](03-cases.md) | The loop applied end to end |

## Where to start

```text
                    00-methodology
                   (always start here)
                          │
            ┌─────────────┴─────────────┐
     01-single-node                02-distributed
   (per-device work)          (adds: skew, collectives)
            └─────────────┬─────────────┘
                     03-cases
```

**Read `00-methodology` first even if you think you know the cause.** Its whole purpose is
to stop you optimising the wrong layer — a scheduling fix cannot help when the time is
going to the host, and no amount of tuning recovers a tile whose innermost dimension the
compiler already flagged.

## What this chapter does not give you

**Numbers.** No page here says "this made it 30% faster", because that number depends on
your shapes, your platform, and the version you are on — it would be wrong by the time you
read it, and wrong in a way that is hard to notice.

What every technique does carry is **how to confirm it**: which tool shows the change took
effect. That is the durable half. A speedup you cannot see in a tool is a
speedup you cannot defend when someone deletes it six months from now.

## See Also

- [Precision](../precision/index.md) — the sibling loop, for when the answer is *wrong*
  rather than slow.
- [Tuning the schedule](../tutorials/05-scheduling-tuning.md) — the hands-on version of
  step 5.
- [Diagnostics](../../dev/passes/92-diagnostics.md) — the compile-time hint registry behind
  step 1.
