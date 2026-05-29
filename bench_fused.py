"""Benchmark the single-kernel fused SwiGLU.

Setup uses ONE fused weight matrix W = [W_up | W_gate] of shape [K, 2N].

Baselines:
  B1  cuBLAS:        gate||up = x @ W   (one big GEMM into [M, 2N])
                     then  out = silu(right) * left  (eager, 2 kernels)
  B2  cuBLAS + compile:  same, but activation fused via torch.compile (1 kernel)
  V4  our kernel:    one launch, no [M, 2N] intermediate

Run on B200:
  srun --partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:30:00 \\
      ~/miniconda3/bin/python bench_fused.py
"""
import torch
import torch.nn.functional as F
import triton.testing as tt

from swiglu.matmul_fused_swiglu import matmul_fused_swiglu

# Optional: colleague's Triton implementation (swiglu/triton/impls.py).
try:
    from swiglu.triton.impls import (
        fused_swiglu_wide_packed,
        pack_swiglu_weight_chunked_torch,
    )
    HAS_TRITON_VARIANT = True
except Exception as _e:
    print(f"[note] Triton variant unavailable: {_e}")
    HAS_TRITON_VARIANT = False

M, K, N = 32768, 3072, 12288
torch.manual_seed(0)

x = torch.randn(M, K,     dtype=torch.bfloat16, device="cuda")
W = torch.randn(K, 2 * N, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)

# ── Reference (fp32) ──
gate_up_f32 = x.float() @ W.float()                  # [M, 2N]
up_ref      = gate_up_f32[:, :N]
gate_ref    = gate_up_f32[:, N:]
C_ref       = F.silu(gate_ref) * up_ref

# ── Correctness ──
C = matmul_fused_swiglu(x, W).float()
diff = (C - C_ref).abs()
atol = max(1.0, K ** 0.5 / 16)
print(f"\nvalidate V4: max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}  "
      f"atol={atol:.2f}  →  {'OK' if diff.max() <= atol else 'FAIL'}")

# ── Baselines (compiled activation: silu(right) * left) ──
@torch.compile
def act_compiled(gu, N):
    return F.silu(gu[:, N:]) * gu[:, :N]

def b1_eager():
    gu = x @ W
    return F.silu(gu[:, N:]) * gu[:, :N]

def b2_compiled():
    gu = x @ W
    return act_compiled(gu, N)

def v4_fused():
    return matmul_fused_swiglu(x, W)

def bench(fn, name):
    fn(); torch.cuda.synchronize()
    ms = tt.do_bench(fn, warmup=200, rep=2000, quantiles=(0.5, 0.0, 1.0))[0]
    print(f"  {name:<48s} {ms:7.3f} ms")
    return ms

print()
print(f"=== timings (M={M}  K={K}  N={N}) ===")
t_b1 = bench(b1_eager,    "B1  cuBLAS [M,2N] + eager silu(r)*l")
t_b2 = bench(b2_compiled, "B2  cuBLAS [M,2N] + torch.compile silu(r)*l")
t_v4 = bench(v4_fused,    "V4  CUDA single-kernel (dual-TMEM, ours)")

if HAS_TRITON_VARIANT:
    # Triton "wide_packed" expects [up|gate] chunk-interleaved layout.
    # Our W convention is [up|gate] concatenated (left half = up, right = gate),
    # which matches the input the pack helper expects: pack_swiglu_weight_chunked_torch
    # ingests [K, 2N_HALF] = [K, N_HALF (left) | N_HALF (gate)] and emits the
    # chunk-interleaved packed buffer.
    W_packed = pack_swiglu_weight_chunked_torch(W)

    # Validate against the same fp32 reference.
    C_t = fused_swiglu_wide_packed(x, W_packed).float()
    diff_t = (C_t - C_ref).abs()
    print(f"validate Triton: max_abs={diff_t.max():.3e}  mean_abs={diff_t.mean():.3e}  "
          f"atol={atol:.2f}  →  {'OK' if diff_t.max() <= atol else 'FAIL'}")

    def triton_fused():
        return fused_swiglu_wide_packed(x, W_packed)

    t_tr = bench(triton_fused, "T1  Triton wide_packed (colleague)")
else:
    t_tr = None

# Component breakdown
t_gemm = bench(lambda: x @ W, "    cuBLAS [M,2N] GEMM only")
gu = x @ W
t_act_eager = bench(lambda: F.silu(gu[:, N:]) * gu[:, :N], "    activation eager only")
t_act_comp  = bench(lambda: act_compiled(gu, N),           "    activation compiled only")

print()
print(f"V4 vs B1: {t_b1 / t_v4:.3f}x")
print(f"V4 vs B2: {t_b2 / t_v4:.3f}x   ← fair baseline")
if t_tr is not None:
    print(f"V4 vs T1: {t_tr / t_v4:.3f}x   (>1 = V4 faster, <1 = Triton faster)")
    print(f"T1 vs B2: {t_b2 / t_tr:.3f}x")
