"""Breakdown of packed_save_factors_inplace BACKWARD components.

Backward is:
  1. grad_de = dy * factors   (Triton elementwise, in-place over factors)
  2. grad_x  = grad_de @ W.t()    (cuBLAS, on stream_x in parallel mode)
     grad_W  = x.t() @ grad_de    (cuBLAS, on stream_w in parallel mode)

We time:
  - each component in isolation (single-call)
  - the actual back-to-back call sequence (the realistic bwd cost)
  - both ENABLE_PARALLEL_BWD_GEMMS=True (default) AND =False
"""
import math, os, sys, time
import torch
import triton.testing as tt

sys.path.insert(0, os.path.expanduser("~/projects/swiglu_fused/swiglu/swiglu_layer"))
import fused_swiglu_wide_packed as swp
from swiglu.triton import impls as triton_impls

M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
device = "cuda"
torch.manual_seed(0)

FLOPS_PER_GEMM = 2 * M * K * (2 * N)

x       = torch.randn(M, K,     device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
W       = torch.randn(K, 2 * N, device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
W_t     = W.t().contiguous()   # transpose used by bwd_x  (cuBLAS will op flag .t() view anyway)
dy      = torch.randn(M, N,     device=device, dtype=DTYPE)

# bf16 byte volumes (for HBM accounting)
B_factors = M * 2*N * 2 / 1e9   # 1.60 GB
B_dy      = M *   N * 2 / 1e9   # 0.40 GB
B_grad_de = M * 2*N * 2 / 1e9   # 1.60 GB
B_W       = K * 2*N * 2 / 1e9   # 0.20 GB
B_x       = M *   K * 2 / 1e9   # 0.08 GB
B_grad_x  = M *   K * 2 / 1e9   # 0.08 GB
B_grad_W  = K * 2*N * 2 / 1e9   # 0.20 GB

print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"shape  : M={M}  K={K}  N={N}  2N={2*N}")
print()

# Warm up the JIT for the Triton elementwise kernel.
factors_warmup = torch.randn(M, 2*N, device=device, dtype=DTYPE)
_ = swp.swiglu_packed_grad_de_from_factors_inplace(factors_warmup, dy)
torch.cuda.synchronize()


def bench(fn, label, total_bytes=None, total_flops=None):
    fn(); torch.cuda.synchronize()
    ms, _, _ = tt.do_bench(fn, warmup=200, rep=1000, quantiles=(0.5, 0.0, 1.0))
    extras = []
    if total_flops:
        extras.append(f"{total_flops / (ms/1e3) / 1e12:>6.0f} TF")
    if total_bytes:
        extras.append(f"{total_bytes / (ms/1e3):>5.1f} GB/s")
    print(f"  {label:<46s}  {ms:6.3f} ms   " + "   ".join(extras))
    return ms


print("=== single-call timings ===")

def elementwise():
    factors = torch.randn(M, 2*N, device=device, dtype=DTYPE)
    return swp.swiglu_packed_grad_de_from_factors_inplace(factors, dy)
def bwd_x_only():
    grad_de = torch.randn(M, 2*N, device=device, dtype=DTYPE)
    return grad_de @ W.t()
def bwd_W_only():
    grad_de = torch.randn(M, 2*N, device=device, dtype=DTYPE)
    return x.t() @ grad_de

# Pre-allocate persistent grad_de for benches that need a stable buffer.
grad_de_persist = torch.randn(M, 2*N, device=device, dtype=DTYPE)

def elementwise_inplace_persist():
    # In-place rewrite over a persistent buffer — same as what backward does
    # (but the buffer is reused, so cache state may differ from real backward).
    return swp.swiglu_packed_grad_de_from_factors_inplace(grad_de_persist, dy)
def bwd_x_persist():
    return grad_de_persist @ W.t()
def bwd_W_persist():
    return x.t() @ grad_de_persist

t_em = bench(elementwise_inplace_persist, "1) elementwise  dy * factors  (in-place)",
             total_bytes=(B_factors + B_dy + B_grad_de) * 1e9)
t_bx = bench(bwd_x_persist, "2) bwd_x   grad_de @ W.t()  (cuBLAS NT)",
             total_flops=FLOPS_PER_GEMM)
t_bw = bench(bwd_W_persist, "3) bwd_W   x.t() @ grad_de  (cuBLAS TN)",
             total_flops=FLOPS_PER_GEMM)
print()

# ── actual sequential backward (no stream parallelism) ──
print("=== sequential backward (ENABLE_PARALLEL_BWD_GEMMS=False) ===")
triton_impls.ENABLE_PARALLEL_BWD_GEMMS = False

def bwd_seq():
    factors = torch.randn(M, 2*N, device=device, dtype=DTYPE)   # fresh each call
    grad_de = swp.swiglu_packed_grad_de_from_factors_inplace(factors, dy)
    grad_x, grad_weight = swp._packed_grad_input_weight(x, W, grad_de)
    return grad_x, grad_weight

t_seq = bench(bwd_seq, "full bwd sequential (alloc factors + 3 kernels)")
print()

# ── actual parallel backward ──
print("=== parallel backward (ENABLE_PARALLEL_BWD_GEMMS=True) ===")
triton_impls.ENABLE_PARALLEL_BWD_GEMMS = True

def bwd_par():
    factors = torch.randn(M, 2*N, device=device, dtype=DTYPE)
    grad_de = swp.swiglu_packed_grad_de_from_factors_inplace(factors, dy)
    grad_x, grad_weight = swp._packed_grad_input_weight(x, W, grad_de)
    return grad_x, grad_weight

t_par = bench(bwd_par, "full bwd parallel (alloc factors + 3 kernels)")
print()

# ── version that avoids per-call allocation noise (use persistent factors) ──
print("=== persistent-factors variants (no torch.randn allocation cost) ===")
def bwd_seq_persist():
    triton_impls.ENABLE_PARALLEL_BWD_GEMMS = False
    grad_de = swp.swiglu_packed_grad_de_from_factors_inplace(grad_de_persist, dy)
    return swp._packed_grad_input_weight(x, W, grad_de)
def bwd_par_persist():
    triton_impls.ENABLE_PARALLEL_BWD_GEMMS = True
    grad_de = swp.swiglu_packed_grad_de_from_factors_inplace(grad_de_persist, dy)
    return swp._packed_grad_input_weight(x, W, grad_de)

t_seq_p = bench(bwd_seq_persist, "full bwd sequential (persistent factors)")
t_par_p = bench(bwd_par_persist, "full bwd parallel   (persistent factors)")
print()

print("=== arithmetic summary ===")
print(f"  Σ components (single-call) : {t_em + t_bx + t_bw:6.3f} ms")
print(f"  max(bwd_x, bwd_W)          : {max(t_bx, t_bw):6.3f} ms   (ideal parallel-2-streams floor)")
print(f"  elementwise + max(GEMMs)   : {t_em + max(t_bx, t_bw):6.3f} ms")
print()
print(f"  measured sequential bwd    : {t_seq_p:6.3f} ms")
print(f"  measured parallel   bwd    : {t_par_p:6.3f} ms")
print()
print(f"  elementwise / total bwd    : {t_em / t_par_p * 100:5.1f}%   (= upper bound on the fusion win)")
