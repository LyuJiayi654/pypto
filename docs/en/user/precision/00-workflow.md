# Narrowing Down a Precision Gap

The result does not match the golden. This page is the order in which to suspect things —
not an API reference.

> **Prerequisites:** [Your first operator](../tutorials/00-elementwise.md) for the
> `allclose` comparison this page assumes you already have.

## The order

```text
Result does not match the golden
├─ 1. Is the golden itself right?         → write_golden; are rtol/atol sane?
├─ 2. Should there be a difference?       → the acceptable-difference table below
├─ 3. Did the compiler already warn?      → verification level, diagnostics
├─ 4. Which pass introduced it?           → torch codegen + validate_ir, bisect
└─ 5. Which tensor is wrong?              → dump_tag / dumps= + enable_dump_args
```

Steps 1 and 2 cost minutes and remove most reports. Steps 4 and 5 cost hours. Do not start
at 4.

## The tools

| Step | Layer | Entry point |
| ---- | ----- | ----------- |
| 1 | End to end | `pypto.runtime.write_golden` + `RunConfig(rtol=, atol=, golden_data_dir=)` |
| 3 | IR legality | `ir.compile(verification_level=...)` / `PYPTO_VERIFY_LEVEL` |
| 3 | Compile-time warnings | `diagnostic_phase` / `disabled_diagnostics` |
| 4 | IR semantics | `pypto.debug.torch_codegen` |
| 4 | Per-pass check | `CompiledProgram.validate_ir` |
| 4 | IR structure | `ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` |
| 5 | Runtime data | `pl.dump_tag(t)` / `dumps=[t]` + `RunConfig(enable_dump_args=1\|2)` |

## Step 1: is the golden right?

The default tolerance is `rtol=1e-5`, which is **wrong for FP16 inputs** — those carry about
three decimal digits, so a correct FP16 matmul fails against an FP32 reference at `1e-5`.
Before investigating the kernel, check that the tolerance matches the *input* precision.

`write_golden` records a reference so later runs compare against a fixed artefact rather
than a recomputed one. That matters when the reference itself is nondeterministic.

## Step 2: should there be a difference?

Some differences are the correct behaviour of a correct compiler. Rule these out before
bisecting anything.

| Source | Difference | Notes |
| ------ | ---------- | ----- |
| Split-K / atomic add | Last bits, run to run | Accumulation order across cores is not fixed |
| FP16 / BF16 accumulation | Grows with reduction length | Accumulate in FP32 where you can |
| Reduction shape | Binary-tree vs sequential | Whether `col_sum` gets a `tmp_tile` changes the order |
| Backend differences | Instruction-level | The same op need not be bit-identical across backends |
| Multi-hop cast | **Usually none** — see below | `LegalizeTileCast` expands what the ISA cannot do in one step |

**The multi-hop cast is worth stating precisely, because it is easy to blame.** On A5,
`INT32→FP16` is expanded to `INT32→FP32→FP16`. That chain is **bit-identical** to a direct
conversion: FP16 saturates above 65504, every integer below that is exact in FP32, so the
FP32 hop never rounds and only the final hop does. A chain only introduces a difference when
an intermediate cannot represent the in-range source values exactly. Check
[LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) for the class your chain falls
into rather than assuming the hop is the culprit.

## Step 3: did the compiler already say something?

Raise the verification level and re-compile before doing any bisection — a malformed-IR
report names the pass for you:

```python
prog = ir.compile(program, verification_level=...)   # or PYPTO_VERIFY_LEVEL
```

## Step 4: which pass introduced it?

This is the expensive step, and the one worth doing properly.

`pypto.debug.torch_codegen` turns an IR `Program` or `Function` into executable torch, so
the IR's *semantics* can be run on the host and compared against your reference — no device
involved:

```python
from pypto.debug import torch_codegen

src = torch_codegen(prog)          # check_shapes=True to assert shapes as it goes
```

`CompiledProgram.validate_ir` runs that comparison per pass. The bisection is then
mechanical: the first pass whose IR stops matching is the one that introduced the
difference. Dump the IR either side of it with
`ir.compile(dump_passes=PassDumpLevel.EXPLICIT)` and read the two.

This locates a *semantic* change. It cannot see a difference that only appears on device —
for that, step 5.

## Step 5: which tensor is wrong?

When the IR is right at every pass but the device result is not, compare actual data:

```python
t = pl.dump_tag(t)       # mark the tensors you care about
cfg = RunConfig(platform="a2a3sim", enable_dump_args=1)
```

Level `1` dumps only tagged tensors; level `2` dumps every task's inputs and outputs. Read
them with `python -m simpler_setup.tools.dump_viewer`.

> **Fatal pitfall:** a full dump (`enable_dump_args=2`) on a large workload can saturate the
> host-side collector (~42 MB/s drain) and get the AICPU killed by a STARS op-execute
> timeout. Prefer level `1` plus `pl.dump_tag` on the specific tensors you are chasing.

## Edge Cases

| Symptom | Likely cause | Step |
| ------- | ------------ | ---- |
| **Correct kernel fails `allclose`** | `rtol=1e-5` against FP16 inputs | 1 |
| **Differs run to run, same input** | Split-K atomic accumulation order | 2 |
| **Differs only for long reductions** | FP16/BF16 accumulator | 2 |
| **Blamed on a multi-hop cast** | Often bit-identical — check the class | 2 |
| **IR matches every pass, device does not** | Not a semantic bug | 5 |
| **Row maxima all `0.0`** | Padding participated in a reduction | See [Reduction and softmax](../tutorials/01-reduction-softmax.md) |

## See Also

- [Worked cases](01-cases.md) — this order applied end to end.
- [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) — when a cast chain is exact.
- [Reduction and softmax](../tutorials/01-reduction-softmax.md) — padding and reductions.
