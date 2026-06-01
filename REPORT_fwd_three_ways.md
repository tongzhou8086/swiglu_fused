# Forward fused SwiGLU — perf ceiling analysis

**Shape**: M = 11136, K = 3584, N = 14336 (BF16, B200, 148 SMs)
**Date**: 2026-06-01
**Bench script**: [`bench_fwd_three_ways.py`](bench_fwd_three_ways.py)

## Question

How close is the production `fused_swiglu_wide_packed_save_factors` Triton
kernel to its theoretical ceiling, and where (if anywhere) is the
remaining headroom?

## Method

Compare three forward variants, all producing the same `out [M, N]`:

| variant | what it does | side-store? |
|---|---|---|
| **V1** | cuBLAS `x @ W_normal` → preact `[M, 2N]` → `torch.compile`'d split+silu+mul → out | no |
| **V2** | one Triton kernel: matmul + split + silu + mul + side-store factors `[M, 2N]` (production) | **yes** |
| **V3** | one Triton kernel: matmul + split + silu + mul → out (no side-store) | no |

V2 − V3 isolates the side-store cost.
V3 − (cuBLAS alone) isolates the Triton-vs-cuBLAS GEMM gap (with the
activation cost folded in for free, given the fusion).

Benchmarked via `triton.testing.do_bench` with 4 s of mixed-variant
warmup followed by 300 ms warmup + 2 s rep per variant. Medians stable
to within ~10 µs.

## Results

| variant | median | TFLOPS | % B200 peak | HBM read+write |
|---|---|---|---|---|
| cuBLAS `x @ W` (GEMM only, no activation) | **1.657 ms** | 1381 | 61.4 % | 0.92 GB |
| V1 cuBLAS + compiled swiglu | 1.815 ms | 1261 | 56.1 % | 1.88 GB |
| **V2 Triton fused + side-store** (production) | **1.831 ms** | 1250 | 55.6 % | 1.24 GB |
| **V3 Triton fused (no save)** | **1.711 ms** | 1338 | 59.5 % | 0.60 GB |

All variants produce bit-identical or numerically-equivalent `out`
(V1 vs V2 = 1.5e-5 max_abs — bf16-conversion noise; V2 vs V3 = 0.0).

## Decomposition

| component | cost (µs) | interpretation |
|---|---|---|
| **V3 − cuBLAS** | **+54** | The Triton fused matmul + activation runs **3.3 % over** the cuBLAS GEMM-only ceiling. Activation is essentially free — folded into the matmul tail. |
| **V2 − V3** | **+120** | Pure side-store cost: writing `factors[M, 2N]` = 0.64 GB to HBM. |
| **V2 − cuBLAS** | **+174** | Total surplus of production V2 over the absolute GEMM-only floor (= "the cost of also doing activation + factors side-store"). |
| **V2 − V1** | **+16** | V2 vs V1 on **latency** alone — within run-to-run noise. |

## Theoretical ceilings

- **Side-store HBM floor**: writing 0.64 GB at B200 peak write bandwidth
  (~6.5 TB/s achievable) ≈ **98 µs**. V2's measured side-store cost is
  120 µs ⇒ **already at 82 %** of the HBM-write-bound for that tensor.
- **Triton-vs-cuBLAS gap**: V3 reaches **96.7 % of cuBLAS GEMM-only
  time** while *also* doing split + silu + mul.

Combined, the **realistic best-case V2 ceiling** is:
```
  cuBLAS GEMM time + HBM-bound side-store ≈ 1.657 + 0.098 = 1.755 ms
```
vs measured V2 = 1.831 ms ⇒ **76 µs of total achievable headroom**, or
**4 %** of V2's runtime.

## V2 vs V1 — the right comparison

V2 and V1 are statistically tied on forward latency, but they are not
equivalent for the broader training picture:

| dimension | V1 | V2 |
|---|---|---|
| forward latency | 1.815 ms | 1.831 ms (≈ tied) |
| HBM bytes moved (fwd) | 1.88 GB | **1.24 GB** (34 % less) |
| transient buffer peak | `preact[M, 2N]` (0.64 GB live) | none (factors is the saved tensor) |
| enables cheap backward? | no — backward must recompute preact or pay extra | **yes** — factors → grad_de is one in-place elementwise (0.24 ms) |
| full bwd cost at this shape | ~5.6 ms (recompute path) | **~3.2 ms** (factors path) |

V2 is the strictly better operating point in any training loop: it pays
+16 µs in forward latency to save ~2.4 ms of backward latency and
0.64 GB of peak transient memory.

## Where headroom is *not*

The instinct to "fuse more, save more" has clear limits at this shape:

1. **Side-store is already HBM-bound.** Any clever overlapping or
   layout tweak can shave at most ~22 µs from the 120 µs side-store
   cost — and that 22 µs is the *upper bound*, not what's plausibly
   reachable.

2. **The compute side is already at 96.7 % of cuBLAS.** The Triton
   compiler has effectively hidden split + silu + mul behind the
   matmul tail. Further fusion of the matmul itself would require
   beating cuBLAS at NN GEMM — extremely unlikely.

3. **In a parallel investigation** (see
   [`bench_fused_grad_x.py`](bench_fused_grad_x.py)), the symmetric
   *backward* idea — fuse `(dy * factors)` into one of the bwd GEMMs
   with a side-store of `grad_de` — was tested and abandoned. The
   fused Triton kernel ran 3–10× slower than cuBLAS NT because the
   prologue fusion needs three input tensors (dy, factors, W) to share
   SMEM stage budget, vs. forward's two (x, W). At BM = 128 BNH = 128
   BK_OUT = 64 the per-stage SMEM is 128 KB (vs forward's 48 KB),
   forcing NS ≤ 1 and crippling the K-loop pipeline. **Pre-matmul
   prologue fusion has fundamentally different SMEM economics than
   post-matmul epilogue fusion.**

## Conclusion

`fused_swiglu_wide_packed_save_factors` is **operating at ~96 % of its
theoretical ceiling** at this shape. The remaining ~4 % (76 µs) is
split between two sub-bounds that are both near their physical floors:

- ~54 µs above cuBLAS GEMM-only (already 96.7 % of cuBLAS)
- ~22 µs above the HBM-write floor for the side-store

Further forward-kernel work is unlikely to be cost-effective. The kernel
should be considered **done from a perf-tuning standpoint** at this
shape, with the focus shifting to either:

- Reducing how often the FFN block is invoked (architectural changes
  outside this kernel's scope), or
- Squeezing the backward — but backward fusion has been shown to be
  structurally hard at this shape (see note 3 above).

## Reproduction

```bash
cd ~/projects/swiglu_fused
srun -p dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00 \
    ~/miniconda3/bin/python bench_fwd_three_ways.py
```
