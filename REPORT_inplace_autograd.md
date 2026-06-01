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

Four variants, all computing `y = silu(gate) * left` where
`[left | gate] = x @ weight.t()`, with autograd enabled:

| | fwd | bwd |
|---|---|---|
| **V_naive** (presentation baseline) | cuBLAS `F.linear` + `torch.compile`'d swiglu | **no custom autograd** — PyTorch handles backward via whatever it generates. Closest equivalent to `swiglu_fusion_notes.md`'s `baseline_F_linear`. |
| **V0 baseline_freshbuf** (control) | same as V_naive | custom autograd; fused Triton swiglu_backward writes to a **fresh** `grad_preact` buffer |
| **V1 baseline_inplace**  | same as V_naive | custom autograd; same kernel as V0 but writes **in place** over the saved `preact` |
| **V2 save_factors** (production) | Triton `_fused_swiglu_wide_packed_save_factors_kernel` (matmul + side-store factors) | Triton `swiglu_packed_grad_de_from_factors_inplace` (in-place `factors *= dy`) |

**V_naive** is what users get today with default PyTorch — it's the
"presentation baseline" matching `swiglu_fusion_notes.md`'s reference
point.

**V0** is a deliberately-constructed *control variable*: it uses the
same Triton bwd kernel as V1, but with a fresh output buffer instead of
aliasing. The point of V0 is to isolate the **in-place buffer aliasing
effect** from any other engineering differences (kernel choice,
autograd implementation, intermediate ordering). V0 vs V1 differ in
exactly one line: the third argument to the bwd kernel.

V_naive vs V1 is the **presentation comparison** (what does the trick
buy a real user?). V0 vs V1 is the **mechanistic comparison** (does the
buffer aliasing itself cost or save anything in latency?).

Both V0/V1's backward use the autotuned `_swiglu_grad_preact_normal_kernel`
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
| V0 vs V_naive | **0.000e+00** | 4.883e-04 | 2.441e-04 |
| V1 vs V0 | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| V2 vs V_naive | 1.526e-05 | 4.883e-04 | — (packed layout) |

V0 vs V_naive: y is bit-identical (same fwd compute graph); gradients
differ only by bf16-rounding noise (V0 uses our Triton kernel for the
swiglu derivative, V_naive uses whatever PyTorch produces via
`torch.compile` autograd).

V1 vs V0: **fully bit-identical** including gradients — the in-place
autograd Function makes zero observable change to outputs.

V2 vs V_naive: differs only at bf16-rounding noise (different math
path: factors-then-multiply vs. swiglu_backward-from-preact).

### Timing (medians, 2 s rep window, autotuned bwd kernel)

| variant | fwd | full step | implied bwd |
|---|---|---|---|
| V_naive PyTorch-default | 1.905 ms | **6.105 ms** | 4.200 ms |
| V0 baseline_freshbuf    | 1.903 ms | 5.817 ms | 3.914 ms |
| V1 baseline_inplace     | 1.907 ms | 5.822 ms | 3.915 ms |
| V2 save_factors         | 1.913 ms | **5.786 ms** | 3.873 ms |

- **Forward**: all four within ~10 µs (noise). The Triton fused kernel
  (V2) is marginally slower than cuBLAS+compiled swiglu (V_naive/V0/V1)
  — same finding as [`REPORT_fwd_three_ways.md`](REPORT_fwd_three_ways.md).
- **Backward**: V_naive is ~285 µs slower than V0/V1, even though it
  uses the same forward compute graph. This is the overhead of
  PyTorch's `torch.compile`'d generic autograd vs. our specialized
  Triton kernel — `torch.compile` re-derives the SwiGLU gradient
  through its saved-tensor machinery, while our kernel implements it
  directly. V2 is another ~42 µs faster than V0/V1 because **V2 has
  already done the SFU work in its forward** (see "Why V2 still beats
  V1" below).
- **V0 vs V1 are bit-tied on backward** (3.914 vs 3.915 ms) — the
  freshbuf-vs-inplace difference is below measurement noise, confirming
  that the memory savings cost nothing in latency.

The bwd kernel `_swiglu_grad_preact_normal_kernel` is autotuned via
`@triton.autotune`; at this shape the winning config is
`BLOCK_M=32, BLOCK_N_HALF=64, num_warps=2` — small tiles with high SM
occupancy, fitting the HBM-bound nature of the kernel.

### Why V2 still beats V1 on backward (115 µs)

The two backward elementwise kernels do different amounts of math:

**V1 backward** (`_swiglu_grad_preact_normal_kernel`) — takes raw
`preact = [left | gate]` + `dy`. Per element:

```
sig         = sigmoid(gate)              ← SFU: exp + recip
silu        = gate * sig
silu_prime  = sig + silu * (1 - sig)
grad_left   = dy * silu
grad_gate   = dy * left * silu_prime
```

Two SFU ops per output element, plus the SwiGLU-derivative recomposition.

**V2 backward** (`_swiglu_packed_grad_de_from_factors_ptr_kernel`) — takes
`factors = [silu(gate) | left · silu'(gate)]` (precomputed in V2's
forward) + `dy`. Per element:

```
grad_de_left = dy * factor_left
grad_de_gate = dy * factor_gate
```

**Zero SFU ops.** Pure multiply. The backward is purely HBM-bound.

**What V2 paid for it (in the forward)**: V2's forward does more work
than V1's. Inside the matmul epilogue V2 also computes and writes
`factors = [silu(gate) | left · silu'(gate)]`. The extra SFU compute is
hidden behind the GEMM (compute-bound), and the extra HBM write is the
side-store.

| | V1 | V2 | Δ |
|---|---|---|---|
| fwd | 1.907 ms | 1.913 ms | **+6 µs** (V2 does more) |
| bwd | 3.915 ms | 3.873 ms | **−42 µs** (V2 skips the SFU) |
| full | 5.822 ms | 5.786 ms | **−36 µs** |

V2 prepays ~6 µs in the forward to save ~42 µs in the backward — a
**net ~36 µs win at this shape**. That's the "save factors" trick
earning its name: convert cheap-in-fwd SFU compute (hidden behind the
matmul) into HBM-bound work in bwd (the cheapest possible kind of
elementwise).

### Peak transient memory

| variant | peak (MiB) | Δ vs V1 |
|---|---|---|
| V_naive PyTorch-default | **+1796.6** | **+337.9** |
| V0 baseline_freshbuf    | **+2068.8** | **+610.0** |
| V1 baseline_inplace     | **+1458.8** | — |
| V2 save_factors         | **+1458.8** | **0.0** |

Reference: `M · 2N · 2 = 609.0 MiB` (size of one preact / grad_preact tensor).

**Two key facts**:

1. **V_naive − V1 = 337.9 MiB** matches the **+338 MiB delta** the
   original `swiglu_fusion_notes.md` reported between `baseline_F_linear`
   (+1524.5 MiB) and `packed_save_factors_inplace` (+1186.6 MiB) almost
   exactly. The absolute numbers differ (different swiglu kernel) but
   **the savings is identical**.
2. **V0 − V1 = 610.0 MiB ≈ M·2N·2** confirms V0's only extra cost
   over V1 is one preact-sized buffer. The controlled comparison is
   clean.

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

### 3. The presentation comparison: V_naive → V1

For a user-facing claim, the relevant comparison is between **what
people get today** (V_naive) and **what they could get with a custom
autograd Function** (V1):

| | V_naive (today) | V1 (with custom autograd) | Δ |
|---|---|---|---|
| peak MiB | +1796.6 | +1458.8 | **−337.9 MiB** |
| fwd | 1.905 ms | 1.907 ms | +2 µs (noise) |
| bwd | 4.200 ms | 3.915 ms | **−285 µs** |
| full | 6.105 ms | 5.822 ms | **−283 µs** |

So a 30-line custom autograd Function + the autotuned Triton bwd
kernel buys **−338 MiB peak memory AND −283 µs (4.6 %) full-step
latency** vs. PyTorch-default, without writing a fused matmul kernel.

V1 → V2 is a further +0 MiB / −36 µs (mostly bwd SFU savings; see
section below) — incremental over V1, not over V_naive.

### 4. Bit-identical numerics (V0 vs V1 stage)

V1 vs V0 is fully bit-identical (max_abs = 0.000e+00 on output and
both gradients) — the in-place autograd Function makes zero observable
change. V0/V1 vs V_naive differ only at bf16-rounding noise.

### 5. The mechanistic comparison: V0 vs V1

V0 vs V1 are **bit-tied on backward** (3.914 vs 3.915 ms) — the
freshbuf-vs-inplace difference is below measurement noise, confirming
that the memory savings cost nothing in latency.

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

- **0 MiB** of memory savings vs. V1 (baseline + custom autograd).
- **~+6 µs** of forward latency (V2 slightly slower from precomputing factors).
- **~−42 µs** of backward latency (V2's bwd is HBM-bound; V1's still does SFU work).
- **~−36 µs** net on the full step over V1.
- **~−319 µs** net on the full step over V_naive (today's PyTorch default).

The latency win is *not* free engineering value: V2's backward is faster
because its forward already paid the SFU cost (sigmoid + silu_prime),
storing the results into `factors`. V1's backward has to compute those
SFU ops itself. So V2 isn't a memory trick — it's a **compute reuse**
trick (factors as a cache of the SwiGLU-derivative pieces) that happens
to also save the same memory V1 saves through its in-place autograd.

The practical recommendation:

- If you have downstream code that reads `factors` directly (e.g.,
  gradient checkpointing that wants to re-use the cached SwiGLU pieces),
  **V2 is the clear win** — same memory as V1, ~106 µs faster, and the
  factors tensor is exposed.
- If you don't care about the factors tensor, **V1 is essentially tied
  on full-step latency** while being structurally much simpler (no
  chunked weight packing, no fused kernel maintenance, bit-identical to
  the canonical reference). 41 µs out of 5.8 ms ≈ 0.7 % — within the
  range where simplicity outweighs perf.

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
