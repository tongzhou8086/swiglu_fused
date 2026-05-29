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

Three baselines on `M=32768, K=3072, N=12288` (BF16, B200), from `recap.py`:

| version | what it is | launches | total |
|---|---|---|---|
| V1 unfused eager      | cuBLAS + `silu` + `mul`              | 3 | 2.536 ms |
| V2 matmul + compiled  | cuBLAS + `torch.compile`d `silu*up`  | 2 | 2.277 ms |
| V3 fully fused (ours)  | single kernel                        | 1 | 2.281 ms |

- **V3 vs V1: 1.11×** (but V1 is a strawman — eager runs the activation as
  two separate kernels).
- **V3 vs V2: 0.998× — a tie.** `torch.compile` fuses `silu*up` into one
  elementwise kernel (0.388 ms), nearly matching our epilogue (0.342 ms).
  V2 is the fair baseline.

Output is bit-identical to the exact-divide reference.

### Why we only tie V2 (and how to win)
Memory accounting per call:
- V2 activation re-reads `gate` from HBM: 2.4 GB (gate + up + h) @ ~6.2 TB/s.
- Our epilogue never spills `gate`: 1.6 GB (up + h) — *should* be ~0.2 ms at
  full BW, but measures 0.342 ms (~4.7 TB/s).

So we move less data yet run no faster: the epilogue's SFU math runs serially
after the TMEM read instead of being hidden behind the memory traffic. That
~0.14 ms of stall cancels the ~0.2 ms saved by skipping the gate round-trip.
**The fusion's real edge (~0.2 ms of saved gate traffic) is currently masked.**

### Epilogue optimization notes (so far)
- Fast reciprocal divide (`__fdividef`) was the dominant win — the activation
  is SFU-bound (SASS: exactly 1 `MUFU.EX2` + 1 `MUFU.RCP` per element).
- Wide TMEM loads (`tcgen05.ld ... x32`) cut `wait::ld` stalls; `x64` plateaus
  on register pressure.
- `up`-load hoisting was negligible (compute-bound, not load-latency-bound).

### Next lever
Overlap epilogue Phase 1 (TMEM→SMEM) with Phase 2 (SMEM→GMEM) — or go
TMEM→regs→GMEM directly — so the `h` writes / `up` reads hide the SFU math.
Target: ~0.34 ms → ~0.2 ms epilogue, turning the V2 tie into a ~1.06× win.


---

## V4 — single-kernel, both GEMMs fused (`matmul_fused_swiglu`)

V3 still leaves the `up = x @ W_up` GEMM to cuBLAS.  V4 absorbs both
GEMMs into a single kernel, so neither `gate` nor `up` ever round-trips
through HBM.  Input is a packed `[K, 2N]` weight `W = [W_up | W_gate]`.

### Design

- **Dual TMEM.**  `tcgen05.alloc(2 * BN cols)` → `taddr_g` cols `[0, BN)`
  for gate accumulator, `taddr_x` cols `[BN, 2BN)` for up accumulator.
- **Unified K-loop of length `2 * (K/BK)`.**  First half streams
  `W[:, N:2N]` into `taddr_g`; second half streams `W[:, :N]` into
  `taddr_x`.  Same b42 main loop, just run twice with different B
  halves and different TMEM destinations.
- **`gate_done` mbarrier** fires the instant the MMA warp commits the
  last gate MMA, *before* it queues the first up MMA.  `tcgen05.mma`
  is fire-and-forget, so gate's last MMA → gate_done arrive → up's
  first MMA all go back-to-back with no wait between passes.
- **Phase A (overlapped with up K-loop):** warps 4..7 wait
  `gate_done`, read `taddr_g` via `tcgen05.ld.x32`, compute silu in
  fp32, pack to bf16, and *hold the result in registers*.  No SMEM
  stash, no TMEM contention.  Per-lane stash = BN bf16 = 128 b32 regs.
- **Phase B-1:** silu warps wait `all_done`, read `taddr_x`, multiply
  by held silu-gate regs, stage bf16 to SMEM (the now-dead K-loop ring).
- **Phase B-2:** all 8 warps drive coalesced SMEM → GMEM stores.

Files: `swiglu/_matmul_fused_swiglu.cu`, `swiglu/matmul_fused_swiglu.py`.

### Result

On `M=32768, K=3072, N=12288` (BF16, B200), via `bench_fused.py`:

| version | what it is | total | TFLOPS |
|---|---|---|---|
| B1 cuBLAS + eager `silu*l`            | 2 launches  | 5.207 ms | 950  |
| B2 cuBLAS + `torch.compile` `silu*l`  | 2 launches  | 3.922 ms | 1262 |
| **V4 dual-TMEM fused**                | 1 launch    | **3.940 ms** | **1257** |

V4 ties B2 — both deliver the fused output at essentially the
GEMM-only time (cuBLAS `[M, 2N]` alone runs 3.507 ms).  The entire
activation cost is hidden under the up K-loop in V4; B2 hides it
behind the `[M, 2N]` HBM round-trip the activation kernel needs anyway.

Memory-axis win: V4's peak transient allocation is **0** vs B2's
**1.5 GB** intermediate `[M, 2N]` buffer.  That's the real reason to
prefer V4 even at parity latency — it removes a per-call activation
buffer from the critical path of HBM pressure.

### Why V4 doesn't beat B2

Honest cost accounting: V4 streams A twice through the K-loop (once
per pass; second time L2-resident, not HBM).  The silu/up-loop
overlap saves ~0.4 ms; the 2× A traffic costs ~0.4 ms in L2 BW.
They cancel.  The pipeline design works *exactly* as planned — it
just hands back what it saves elsewhere.


---

## T1 — colleague's Triton implementation (`fused_swiglu_wide_packed`)

`swiglu/triton/impls.py` contains a Triton implementation of the same
fwd op (plus five backward-saving variants).  Bench harness loads
the inference-only one (`fused_swiglu_wide_packed`) as **T1**.

### Design (the structurally important parts)

- **Wide accumulator, single K-loop.**  Instead of two TMEM regions
  with two MMA streams, the accumulator is just twice as wide:
  `acc[BM, 2*BN_HALF]`.  One fat `tl.dot(a, b, acc)` per K-iter
  produces both halves at once, with B in chunk-interleaved packed
  layout `[left_chunk_0 | gate_chunk_0 | left_chunk_1 | ...]`.
  **A is streamed once.**
- **Persistent CTAs.**  Grid = `NUM_SMS = 148`, not `num_tiles`.
  Each CTA walks many output tiles.
- **`FLATTEN=True`.**  Triton fuses the outer tile loop and the inner
  K-loop into one pipeline.  Activation+store of tile T overlaps with
  the K-loop of tile T+1 — the **cross-tile** analog of V4's
  intra-tile dual-TMEM pipeline.
- Same macro tile as V4 (BM=128, BN=256 effective, BK=64, 8 warps).
  NS=4 (vs V4's NS=6) because persistence amortizes the per-tile
  pipeline cost.  GSM=32 in CTA-tiles (vs V4's GSM=8 in cluster-tiles).
  **No `cta_group::2` cluster** — single-CTA tiles, with
  per-SM parallelism instead.

### Result

| version | time | TFLOPS | % cuBLAS GEMM | activation surcharge |
|---|---|---|---|---|
| cuBLAS `[M, 2N]` only                 | 3.507 ms | 1411 | 100 %   | — |
| **T1 Triton wide_packed**             | **3.630 ms** | **1363** | **96.6 %** | **+3.4 %** |
| B2 cuBLAS + compiled act              | 3.922 ms | 1262 | 89.4 %  | +11.8 % |
| V4 our CUDA fused                     | 3.940 ms | 1257 | 89.1 %  | +12.3 % |

**T1 beats B2 (and V4) by 8 %**, and beats cuBLAS-only on a
TFLOPS basis once you remember T1 also computes the activation.
Activation surcharge over pure GEMM is ~3× lower than B2 / V4.
Numerics are also tighter (max_abs 0.047 vs V4's 0.107) because a
single wide fp32 reduction has one fewer rounding boundary than two
separately accumulated halves.

### The architectural lesson

> Cross-tile overlap via persistent CTAs is strictly cheaper than
> intra-tile overlap via dual TMEM, because the intra-tile approach
> forces 2× A streaming as the price of admission.

V4's pipeline design works, but it's solving the wrong problem.  The
silu-hidden-under-the-up-K-loop optimization is real; it just buys
exactly what it costs elsewhere.  Once you commit to "single K-loop
produces both halves in one shot" (wide accumulator + packed weight),
A streams once, B streams once, and persistent scheduling hides the
activation behind the *next* tile's K-loop without needing any
intra-tile pipeline machinery.

For an FFN that's hot enough to repay a weight-repack pass,
`fused_swiglu_wide_packed` is the design to copy.  V4 is preserved
in-tree as a CUDA reference for the dual-TMEM technique and for the
register-held silu(gate) stash (which is itself reusable in other
fused epilogues where cross-tile overlap isn't an option).
