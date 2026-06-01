"""Forward perf analysis: 3 kernel variants at M=11136, K=3584, N=14336.

  V1  cuBLAS big GEMM (out [M, 2N]) + torch.compile'd swiglu activation
      (split + silu + mul).  No fusion into the matmul.

  V2  Triton fused, NO save factors.  One kernel does GEMM + split + silu
      + mul + writes out only.  Same work as V3 minus the 1.6 GB factors
      write — establishes V3's "ceiling".

  V3  Triton fused + side-store of factors  (production save_factors path).
      One kernel does GEMM + split + silu + mul + writes [out | factors].

V3 - V2 = the side-store cost.  V2 - V1's GEMM-only cuBLAS = the
"Triton-vs-cuBLAS GEMM gap" for the fused kernel.
"""
import math, os, sys
import torch
import triton.testing as tt

sys.path.insert(0, os.path.expanduser("~/projects/swiglu_fused/swiglu/swiglu_layer"))
import fused_swiglu_wide_packed as swp


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
device = "cuda"
torch.manual_seed(0)

FLOPS = 2 * M * K * (2 * N)
B200_PEAK = 2250e12

# HBM byte volumes for analysis
B_x        = M * K * 2 / 1e9         # 0.08 GB
B_W        = K * 2*N * 2 / 1e9       # 0.20 GB
B_out      = M * N * 2 / 1e9         # 0.32 GB
B_factors  = M * 2*N * 2 / 1e9       # 0.64 GB
B_preact   = M * 2*N * 2 / 1e9       # 0.64 GB (intermediate in V1)

x       = torch.randn(M, K,     device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = swp.pack_swiglu_weight_chunked_torch(W_normal)

print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"shape  : M={M}  K={K}  N={N}  2N={2*N}")
print(f"FLOPs  : {FLOPS/1e12:.3f} T")
print(f"B200 BF16 peak : 2250 TFLOPS")
print()

# ── V1: cuBLAS GEMM + compiled swiglu activation ──
# preact is laid out as the NORMAL [left|gate] (not chunk-packed) — that's
# what cuBLAS computes from x @ W_normal.  Then a compiled elementwise
# kernel splits, applies silu, multiplies.
@torch.compile(fullgraph=True, dynamic=False)
def _compiled_swiglu(preact):
    left, gate = preact.chunk(2, dim=-1)
    return left * torch.nn.functional.silu(gate)

# Warm up compile
out_warm = _compiled_swiglu(torch.empty(M, 2*N, device=device, dtype=DTYPE))
del out_warm
torch.cuda.synchronize()

def v1_cublas_plus_compiled():
    preact = x @ W_normal      # [M, 2N], cuBLAS NN
    return _compiled_swiglu(preact)

# ── V2: Triton fused, no save factors ──
def v2_triton_no_save():
    return swp.fused_swiglu_wide_packed(x, W_packed)

# ── V3: Triton fused + save factors (production) ──
def v3_triton_save_factors():
    return swp.fused_swiglu_wide_packed_save_factors(x, W_packed)

# ── correctness ─────────────────────────────────────────────────────
out_v1 = v1_cublas_plus_compiled()
out_v2 = v2_triton_no_save()
out_v3, fac_v3 = v3_triton_save_factors()

e12 = (out_v1.float() - out_v2.float()).abs().max().item()
e13 = (out_v1.float() - out_v3.float()).abs().max().item()
e23 = (out_v2.float() - out_v3.float()).abs().max().item()
print(f"  V1 vs V2 out max_abs = {e12:.3e}")
print(f"  V1 vs V3 out max_abs = {e13:.3e}")
print(f"  V2 vs V3 out max_abs = {e23:.3e}")
print()


def bench(fn, label, total_bytes=None):
    fn(); torch.cuda.synchronize()
    ms, mn, mx = tt.do_bench(fn, warmup=300, rep=2000, quantiles=(0.5, 0.0, 1.0))
    tflops = FLOPS / (ms/1e3) / 1e12
    pct = tflops / 2250 * 100
    extras = [f"{tflops:>6.0f} TF", f"{pct:>5.1f}% peak"]
    if total_bytes is not None:
        extras.append(f"HBM={total_bytes:>4.2f} GB → {total_bytes/(ms/1e3):>5.0f} GB/s")
    print(f"  {label:<48s}  med={ms:6.3f}  min={mn:6.3f}  max={mx:6.3f}  " + "  ".join(extras))
    return ms


# ── timings ─────────────────────────────────────────────────────────
# Warmup all variants together first to settle GPU clock state.
print("global warmup: 4s of mixed calls ...", flush=True)
import time
t0 = time.time()
i = 0
while time.time() - t0 < 4.0:
    v1_cublas_plus_compiled()
    v2_triton_no_save()
    v3_triton_save_factors()
    i += 1
torch.cuda.synchronize()
print(f"  warm calls: {i*3}", flush=True)
print()

print("=== timings ===")
# HBM accounting per variant (rough):
#   V1 : read x + read W + write preact + read preact + write out
#        = 0.08 + 0.20 + 0.64 + 0.64 + 0.32 = 1.88 GB
#   V2 : read x + read W + write out
#        = 0.08 + 0.20 + 0.32 = 0.60 GB
#   V3 : read x + read W + write out + write factors
#        = 0.08 + 0.20 + 0.32 + 0.64 = 1.24 GB
bytes_v1 = B_x + B_W + B_preact + B_preact + B_out      # 1.88 GB
bytes_v2 = B_x + B_W + B_out                            # 0.60 GB
bytes_v3 = B_x + B_W + B_out + B_factors                # 1.24 GB

# cuBLAS-only ceiling
def cublas_only():
    return x @ W_normal

t_cublas      = bench(cublas_only,                "cuBLAS x @ W (no activation, no fusion)",
                      total_bytes=B_x + B_W + B_preact)
t_v1          = bench(v1_cublas_plus_compiled,    "V1 cuBLAS + compiled swiglu",
                      total_bytes=bytes_v1)
t_v2          = bench(v2_triton_no_save,          "V2 Triton fused (no save)",
                      total_bytes=bytes_v2)
t_v3          = bench(v3_triton_save_factors,     "V3 Triton fused + side-store factors",
                      total_bytes=bytes_v3)
print()

print("=== analysis ===")
print(f"  V3 − V2 (side-store cost)        : {(t_v3 - t_v2)*1000:+.0f} µs")
print(f"  V2 − cuBLAS (Triton-vs-cuBLAS gap for GEMM-only): {(t_v2 - t_cublas)*1000:+.0f} µs")
print(f"  V3 − V1 (production savings)     : {(t_v3 - t_v1)*1000:+.0f} µs")
print(f"  V2 / cuBLAS                      : {t_v2 / t_cublas * 100:5.1f}% of GEMM-only time")
print(f"  V3 / V2                          : {t_v3 / t_v2 * 100:5.1f}% (side-store ratio)")
print()

print("=== HBM efficiency ===")
print(f"  V3 HBM/s : {bytes_v3 / (t_v3/1e3):.0f} GB/s   (B200 peak ~7700 GB/s)")
print(f"  V2 HBM/s : {bytes_v2 / (t_v2/1e3):.0f} GB/s")
print()

print("=== upper-bound floors ===")
# All times in ms.  do_bench returns ms; HBM bytes are in GB; 1 GB / (GB/s) = s, *1000 = ms.
hbm_floor_v3 = bytes_v3 / 6500 * 1000   # ms
print(f"  HBM-bound floor at 6500 GB/s for V3 : {hbm_floor_v3:.3f} ms")
print(f"  compute-bound floor (= cuBLAS GEMM) : {t_cublas:.3f} ms")
print(f"  V3 measured                         : {t_v3:.3f} ms")
