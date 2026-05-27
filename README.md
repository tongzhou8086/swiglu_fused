# swiglu_fused

Fused SwiGLU FFN stage for Blackwell (B200): a single kernel computing

```
h = silu(x @ W_gate) * up        # up = x @ W_up, precomputed by the caller
```

The GEMM mainloop is the `b42_gsm` kernel (tuned `GROUP_SIZE_M`); the fusion
lives in the epilogue, where `silu(gate) * up` is applied to the fp32
accumulator before the bf16 down-cast — so `gate` is never materialised to
HBM, and the standalone SiLU·mul kernel is eliminated.

## Layout
- `swiglu/_matmul_silu_mul.cu` — the kernel (tcgen05 MMA + fused epilogue).
- `swiglu/matmul_silu_mul.py` — autotuned launcher (`BN, BK, NS, GSM` sweep).
- `swiglu/_pycuda_loader.py`, `swiglu/_tma_utils.py` — infra copied from mymatmul.
- `bench.py` — correctness + fused-vs-unfused benchmark.

## Run (B200)
```
srun --partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:30:00 \
    python bench.py --shapes 32768x3072x12288
```

## Status
On `M=32768, K=3072, N=12288` (BF16): **1.12×** faster than the unfused
baseline (cuBLAS gate GEMM + torch `silu*mul`), output bit-identical.

Epilogue optimization notes:
- Fast reciprocal divide (`__fdividef`) was the dominant win — the activation
  is SFU-bound (SASS: exactly 1 `MUFU.EX2` + 1 `MUFU.RCP` per element).
- Wide TMEM loads (`tcgen05.ld ... x32`) cut `wait::ld` stalls; `x64` plateaus
  on register pressure.
- `up`-load hoisting was negligible (compute-bound, not load-latency-bound).

Next lever: overlap epilogue Phase 1 (TMEM→SMEM) with Phase 2 (SMEM→GMEM) to
hide the `h` writes behind the SFU math (~0.31 ms epilogue vs ~0.2 ms BW floor).
