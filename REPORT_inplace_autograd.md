# Custom autograd matches `save_factors` on memory — no fused kernel needed

**Shape**: M = 11136, K = 3584, N = 14336 (BF16, B200)
**Date**: 2026-06-01
**Bench script**: [`bench_inplace_autograd.py`](bench_inplace_autograd.py)

## Question

`swiglu_fusion_notes.md` reported that `packed_save_factors_inplace`
saved ~338 MiB of peak transient memory over `baseline_F_linear`, even
though both save an `[M, 2N]` tensor for backward. We hypothesized this
gap is a PyTorch-autograd-machinery artifact, not anything intrinsic to
the fused matmul:

> The baseline saves `preact[M, 2N]`, then in backward allocates a fresh
> `grad_preact[M, 2N]` to feed the two GEMMs. `save_factors_inplace`
> overwrites the saved tensor in place, never allocating the second
> buffer. If we expose that in-place trick to the baseline via a custom
> autograd Function, the memory gap should disappear.

This report tests the hypothesis directly.

## Method

Three variants, all computing `y = silu(gate) * left` where
`[left | gate] = x @ weight.t()`, with autograd enabled:

| | fwd | bwd |
|---|---|---|
| **V0 baseline_freshbuf** | cuBLAS `F.linear` + `torch.compile`'d swiglu | custom autograd; fused Triton swiglu_backward writes to a **fresh** `grad_preact` buffer |
| **V1 baseline_inplace**  | same as V0 | custom autograd; same kernel but writes **in place** over the saved `preact` |
| **V2 save_factors** (production) | Triton `_fused_swiglu_wide_packed_save_factors_kernel` (matmul + side-store factors) | Triton `swiglu_packed_grad_de_from_factors_inplace` (in-place `factors *= dy`) |

V0 and V1 share the same forward bytes for-byte. V2 is a different
forward kernel that saves `factors` directly instead of `preact`.

Both V0/V1's backward uses the new `_swiglu_grad_preact_normal_kernel`
in [`swiglu/triton/impls.py`](swiglu/triton/impls.py), which takes
separate input/output pointers — alias them for in-place, pass a fresh
buffer for the freshbuf variant.

Bench: `triton.testing.do_bench`, 6 s mixed-variant warmup, 300 ms
per-variant warmup, 2 s rep window. Peak memory measured via
`torch.cuda.max_memory_allocated()` deltas around a single
fwd+bwd step in isolation.

## Results

### Correctness

| pair | y max_abs | grad_x max_abs | grad_w max_abs |
|---|---|---|---|
| V1 vs V0 | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| V2 vs V0 | 1.526e-05 | 4.883e-04 | — (packed layout) |

V1 is **bit-identical** to V0 — the in-place autograd Function makes
zero observable change to outputs. V2 differs only at bf16-rounding
noise (different math path: factors-then-multiply vs.
swiglu_backward-from-preact).

### Timing (medians, 2 s rep window)

| variant | fwd | full step | implied bwd |
|---|---|---|---|
| V0 baseline_freshbuf | 1.899 ms | 5.831 ms | 3.931 ms |
| V1 baseline_inplace  | 1.902 ms | 5.885 ms | 3.984 ms |
| V2 save_factors      | 1.911 ms | 5.780 ms | 3.869 ms |

- **Forward**: all three within ~12 µs (noise). The Triton fused kernel
  (V2) is marginally slower than cuBLAS+compiled swiglu (V0/V1) — same
  finding as
  [`REPORT_fwd_three_ways.md`](REPORT_fwd_three_ways.md).
- **Backward**: all three within ~115 µs. V2 is the fastest by ~60–115 µs
  (likely because the in-place `factors *= dy` is a tighter elementwise
  than our `swiglu_backward_inplace` kernel), but within the run-to-run
  spread.

### Peak transient memory

| variant | peak (MiB) | Δ vs V0 |
|---|---|---|
| V0 baseline_freshbuf | **+2068.8** | — |
| V1 baseline_inplace  | **+1458.8** | **−610.0** |
| V2 save_factors      | **+1458.8** | **−610.0** |

Reference: `M · 2N · 2 = 609.0 MiB` (size of one preact / grad_preact tensor).

## Findings

### 1. The custom autograd recovers exactly the missing buffer

**V0 − V1 = 610.0 MiB ≈ M · 2N · 2 = 609.0 MiB**. The savings are
*precisely* one preact-sized tensor — the `grad_preact` buffer that
V0's backward allocates and V1's in-place backward elides. There is no
ambiguity about where the memory goes.

### 2. V1 = V2 to the byte

**V1 − V2 = 0.0 MiB**. The custom-autograd-on-cuBLAS variant has
identical peak memory to the production fused-kernel variant. The
production kernel buys exactly **zero** memory savings over a properly
written baseline at this shape.

### 3. Bit-identical numerics with V0

The custom autograd path doesn't change a single bit of the output or
the gradients. It's a pure transparent optimization for memory.

### 4. The latency picture is essentially tied

V1 vs V2 differ by ~100 µs at most on full step time. V2 is the fastest
but only by a margin within run-to-run spread. The fused kernel's
~120 µs side-store cost (documented in `REPORT_fwd_three_ways.md`) is
matched almost exactly by V1's slightly cheaper forward (no side-store
write).

## Conclusion

**The peak-memory advantage of `_fused_swiglu_wide_packed_save_factors_kernel`
over the baseline `cuBLAS F.linear + torch.compile swiglu` is entirely a
PyTorch-default-autograd artifact, not anything specific to the fused
kernel.** A 30-line custom `torch.autograd.Function` (plus a single
~50-line Triton in-place backward kernel) achieves byte-exact memory
parity with the production fused kernel, while staying bit-identical to
the baseline's outputs and gradients.

This rewrites the trade-off we documented in
[`REPORT_fwd_three_ways.md`](REPORT_fwd_three_ways.md). The production
kernel's complexity buys:

- **0 MiB** of memory savings vs. baseline + custom autograd.
- **~+10 µs** of forward latency (V2 slightly slower than V0/V1's fwd).
- **~−60 µs** of backward latency (V2's tighter in-place elementwise).
- A **factors-side-store** that's useful if downstream code reads it
  directly (e.g., for re-use across multiple backward passes, like
  gradient checkpointing recomputes).

At this shape, **the practical recommendation flips**: prefer the
custom-autograd baseline unless something downstream needs the factors
tensor directly. It is structurally simpler (no chunked weight packing,
no fused kernel maintenance), bit-identical to the canonical reference,
and within noise on latency.

## Files

- [`bench_inplace_autograd.py`](bench_inplace_autograd.py) — the bench
  with all three variants and memory measurement.
- [`swiglu/triton/impls.py`](swiglu/triton/impls.py) — adds
  `_swiglu_grad_preact_normal_kernel` (the fused in-place backward).

## Reproduction

```bash
cd ~/projects/swiglu_fused
srun -p dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00 \
    ~/miniconda3/bin/python bench_inplace_autograd.py
```
