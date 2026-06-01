"""Minimal experiment: fused (dy * factors) @ W.t() vs status quo.

T_fused          : one Triton kernel doing both the elementwise and the GEMM.
status quo (sum) : Triton elementwise + cuBLAS NT matmul.

If T_fused < (status-quo sum), the fusion direction is viable and we'd then
add the side-store of grad_de + a custom second GEMM.
"""
import math, os, sys
import torch
import triton.testing as tt

sys.path.insert(0, os.path.expanduser("~/projects/swiglu_fused"))
from swiglu.triton import impls as triton_impls

M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
device = "cuda"
torch.manual_seed(0)

FLOPS = 2 * M * K * (2 * N)

x       = torch.randn(M,     K,     device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
W       = torch.randn(K,     2 * N, device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
dy      = torch.randn(M,     N,     device=device, dtype=DTYPE)
factors = torch.randn(M, 2 * N,     device=device, dtype=DTYPE)

print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"shape  : M={M}  K={K}  N={N}  2N={2*N}")
print()

# ── correctness ─────────────────────────────────────────────────────
# Reference: do exactly what status-quo does.
factors_ref = factors.clone()
import sys as _s
_s.path.insert(0, os.path.expanduser("~/projects/swiglu_fused/swiglu/swiglu_layer"))
import fused_swiglu_wide_packed as swp
grad_de_ref = swp.swiglu_packed_grad_de_from_factors_inplace(factors_ref, dy).clone()
# .clone() because next line would mutate; keep a copy.
grad_x_ref  = grad_de_ref @ W.t()

# Fused candidate (uses the ORIGINAL factors, not mutated).
grad_x_fused = triton_impls.fused_grad_x(dy, factors, W)

err = (grad_x_ref.float() - grad_x_fused.float()).abs().max().item()
rel = err / grad_x_ref.float().abs().max().item()
atol = max(1.0, math.sqrt(2*N) / 16)
ok = err <= atol
print(f"correctness: max_abs={err:.3e}  rel={rel:.3e}  atol={atol:.2f}  → {'OK' if ok else 'FAIL'}")
print()

if not ok:
    print("(NOTE: bench will still run, but numbers are suspect)")

# ── timing ──────────────────────────────────────────────────────────
def bench(fn, label, total_flops=None, total_bytes=None):
    fn(); torch.cuda.synchronize()
    ms, mn, mx = tt.do_bench(fn, warmup=200, rep=1500, quantiles=(0.5, 0.0, 1.0))
    extras = []
    if total_flops:
        extras.append(f"{total_flops / (ms/1e3) / 1e12:>6.0f} TFLOPS")
    if total_bytes:
        extras.append(f"{total_bytes / (ms/1e3):>5.1f} GB/s")
    print(f"  {label:<48s}  median={ms:7.3f}  min={mn:7.3f}  max={mx:7.3f}  " + "  ".join(extras))
    return ms


# Out-of-place (non-mutating) elementwise that mirrors what backward effectively
# does: produce grad_de from (factors, dy).  We allocate a fresh buffer each call
# so factors stays clean for the fused variant to consume from a known state.
grad_de_buf = torch.empty(M, 2*N, device=device, dtype=DTYPE)

def fn_elementwise_only():
    # Use the existing kernel but write to a separate buffer (not in-place).
    # Triton kernel writes to grad_de_ptr; if we pass a different buffer than
    # factors, it's effectively out-of-place.
    from swiglu.triton.impls import _swiglu_packed_grad_de_from_factors_ptr_kernel
    import triton as _tr
    m, n2 = factors.shape
    grid = (
        _tr.cdiv(m, triton_impls.BWD_FACTORS_BLOCK_SIZE_M)
        * _tr.cdiv(n2 // 2, triton_impls.BWD_FACTORS_BLOCK_SIZE_N_HALF),
    )
    _swiglu_packed_grad_de_from_factors_ptr_kernel[grid](
        factors, dy, grad_de_buf,
        m, n2 // 2,
        BLOCK_SIZE_M_=triton_impls.BWD_FACTORS_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=triton_impls.BWD_FACTORS_BLOCK_SIZE_N_HALF,
        num_warps=triton_impls.BWD_FACTORS_NUM_WARPS,
    )
    return grad_de_buf

def fn_cublas_nt_only():
    return grad_de_buf @ W.t()

def fn_status_quo():
    grad_de = fn_elementwise_only()
    return grad_de @ W.t()

def fn_fused():
    return triton_impls.fused_grad_x(dy, factors, W)


print("=== timings ===")
t_em = bench(fn_elementwise_only, "Triton elementwise (in-place)",         total_bytes=(M*2*N*2 + M*N*2 + M*2*N*2))
t_nt = bench(fn_cublas_nt_only,   "cuBLAS NT  grad_de @ W.t()",            total_flops=FLOPS)
t_sq = bench(fn_status_quo,       "STATUS QUO  elementwise + cuBLAS",      total_flops=FLOPS)
t_fu = bench(fn_fused,            "FUSED       (dy * factors) @ W.t()",    total_flops=FLOPS)
print()

print("=== verdict ===")
sum_quo = t_em + t_nt
print(f"  Σ (elementwise + cuBLAS) single-call : {sum_quo:6.3f} ms")
print(f"  measured back-to-back status quo     : {t_sq:6.3f} ms")
print(f"  measured fused                       : {t_fu:6.3f} ms")
print()
delta_vs_quo = t_fu - t_sq
delta_vs_sum = t_fu - sum_quo
print(f"  fused − status quo (measured)        : {delta_vs_quo:+.3f} ms  ({delta_vs_quo/t_sq*100:+.1f}%)")
print(f"  fused − Σ single-call                : {delta_vs_sum:+.3f} ms  ({delta_vs_sum/sum_quo*100:+.1f}%)")
print(f"  fused − cuBLAS NT only               : {t_fu - t_nt:+.3f} ms  (this is the cost of being non-cuBLAS)")
