"""Hypothesis: the baseline (cuBLAS F.linear + torch.compile swiglu) has higher
peak memory than `_fused_swiglu_wide_packed_save_factors_kernel` ONLY because
PyTorch's default autograd allocates a fresh grad_preact buffer in backward
instead of overwriting the saved preact.

Variants tested:
  V0  cuBLAS + compiled fwd, default-shape backward (FRESH grad_preact buffer)
  V1  cuBLAS + compiled fwd, IN-PLACE backward over preact via custom autograd
  V2  Triton fused save_factors fwd + in-place backward elementwise (production)

V0 = "what PyTorch would do today, eager-friendly".
V1 = "fix the PyTorch issue without writing a fused matmul kernel".
V2 = the production path we want V1 to be directly comparable to.

All three share the same forward COMPUTE GRAPH semantics
  y = silu(gate) * left  where [left|gate] = x @ weight.t()
"""
import gc, math, os, sys
import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

sys.path.insert(0, os.path.expanduser("~/projects/swiglu_fused"))
sys.path.insert(0, os.path.expanduser("~/projects/swiglu_fused/swiglu/swiglu_layer"))
import fused_swiglu_wide_packed as swp
from swiglu.triton.impls import (
    _swiglu_grad_preact_normal_kernel,
    INPLACE_BWD_BLOCK_M as BM_BWD,
    INPLACE_BWD_BLOCK_N_HALF as BN_BWD,
    INPLACE_BWD_NUM_WARPS as NW_BWD,
)


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
device = "cuda"
torch.manual_seed(0)


# ─────────────────────────────────────────────────────────────────────
# Helpers — fused swiglu backward kernel callable.
# Passing `out is preact` → in-place; passing fresh `out` → extra buffer.
# ─────────────────────────────────────────────────────────────────────
def swiglu_grad_preact_normal(preact, dy, out):
    M_, twoN_ = preact.shape
    N_ = twoN_ // 2
    grid = (triton.cdiv(M_, BM_BWD) * triton.cdiv(N_, BN_BWD),)
    _swiglu_grad_preact_normal_kernel[grid](
        preact, dy, out, M_, N_,
        BLOCK_M=BM_BWD, BLOCK_N_HALF=BN_BWD, num_warps=NW_BWD,
    )
    return out


# ─────────────────────────────────────────────────────────────────────
# Compiled fwd activation — the SAME function used by V0 and V1.
# ─────────────────────────────────────────────────────────────────────
@torch.compile(fullgraph=True, dynamic=False)
def _compiled_swiglu(preact):
    n_half = preact.shape[-1] // 2
    left = preact[..., :n_half]
    gate = preact[..., n_half:]
    return left * F.silu(gate)


# ─────────────────────────────────────────────────────────────────────
# V0  baseline_freshbuf : default-style backward (fresh grad_preact buffer).
# ─────────────────────────────────────────────────────────────────────
class BaselineFreshBufSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        preact = F.linear(x, weight)
        ctx.save_for_backward(x, weight, preact)
        return _compiled_swiglu(preact)

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, preact = ctx.saved_tensors
        grad_preact = torch.empty_like(preact)               # ← extra buffer
        swiglu_grad_preact_normal(preact, grad_y, grad_preact)
        grad_x = grad_preact @ weight
        grad_weight = grad_preact.t() @ x
        return grad_x, grad_weight


def baseline_freshbuf(x, weight):
    return BaselineFreshBufSwiGLU.apply(x, weight)


# ─────────────────────────────────────────────────────────────────────
# V1  baseline_inplace : same fwd as V0; backward writes IN-PLACE over preact.
# ─────────────────────────────────────────────────────────────────────
class BaselineInplaceSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        preact = F.linear(x, weight)
        ctx.save_for_backward(x, weight, preact)
        return _compiled_swiglu(preact)

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, preact = ctx.saved_tensors
        swiglu_grad_preact_normal(preact, grad_y, preact)    # ← in-place
        grad_preact = preact                                  # alias for clarity
        grad_x = grad_preact @ weight
        grad_weight = grad_preact.t() @ x
        ctx.maybe_clear_saved_tensors()
        return grad_x, grad_weight


def baseline_inplace(x, weight):
    return BaselineInplaceSwiGLU.apply(x, weight)


# ─────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────
def make_inputs(seed=0):
    torch.manual_seed(seed)
    x      = torch.randn(M, K,   device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
    weight = torch.randn(2*N, K, device=device, dtype=DTYPE) * (1.0 / math.sqrt(K))
    W_kxn  = weight.t().contiguous()
    W_pack = swp.pack_swiglu_weight_chunked_torch(W_kxn)
    grad_y = torch.randn(M, N,   device=device, dtype=DTYPE)
    return x, weight, W_pack, grad_y


def fresh_leaves(x_buf, w_leaf):
    x = x_buf.detach().clone().requires_grad_(True)
    w = w_leaf.detach().clone().requires_grad_(True)
    return x, w


# ─────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────
def correctness_check(x0, weight, W_packed, grad_y):
    print("correctness:")
    # V0 reference.
    xa, wa = fresh_leaves(x0, weight)
    ya = baseline_freshbuf(xa, wa); ya.backward(grad_y)
    # V1.
    xb, wb = fresh_leaves(x0, weight)
    yb = baseline_inplace(xb, wb); yb.backward(grad_y)
    # V2.
    xc, wc = fresh_leaves(x0, W_packed)
    yc = swp.fused_swiglu_wide_packed_save_factors_autograd(xc, wc); yc.backward(grad_y)

    e_y   = (ya - yb).float().abs().max().item()
    e_gx  = (xa.grad - xb.grad).float().abs().max().item()
    e_gw  = (wa.grad - wb.grad).float().abs().max().item()
    print(f"  V1 vs V0  y={e_y:.3e}  grad_x={e_gx:.3e}  grad_w={e_gw:.3e}")
    e_y2  = (ya - yc).float().abs().max().item()
    e_gx2 = (xa.grad - xc.grad).float().abs().max().item()
    print(f"  V2 vs V0  y={e_y2:.3e}  grad_x={e_gx2:.3e}  (weight grad in packed layout, skipped)")
    print()


# ─────────────────────────────────────────────────────────────────────
# Step builders
# ─────────────────────────────────────────────────────────────────────
def step_v0(x_buf, w_buf, grad_y):
    x, w = fresh_leaves(x_buf, w_buf)
    y = baseline_freshbuf(x, w); y.backward(grad_y)

def step_v1(x_buf, w_buf, grad_y):
    x, w = fresh_leaves(x_buf, w_buf)
    y = baseline_inplace(x, w); y.backward(grad_y)

def step_v2(x_buf, w_packed, grad_y):
    x, w = fresh_leaves(x_buf, w_packed)
    y = swp.fused_swiglu_wide_packed_save_factors_autograd(x, w); y.backward(grad_y)


def fwd_v0(x_buf, w_buf):
    x, w = fresh_leaves(x_buf, w_buf);  return baseline_freshbuf(x, w)
def fwd_v1(x_buf, w_buf):
    x, w = fresh_leaves(x_buf, w_buf);  return baseline_inplace(x, w)
def fwd_v2(x_buf, w_packed):
    x, w = fresh_leaves(x_buf, w_packed); return swp.fused_swiglu_wide_packed_save_factors_autograd(x, w)


# ─────────────────────────────────────────────────────────────────────
# Memory measurement
# ─────────────────────────────────────────────────────────────────────
def measure_peak_alloc(make_step):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    make_step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────
def bench(fn, label):
    fn(); torch.cuda.synchronize()
    ms, mn, mx = tt.do_bench(fn, warmup=300, rep=2000, quantiles=(0.5, 0.0, 1.0))
    print(f"  {label:<36s}  med={ms:6.3f}  min={mn:6.3f}  max={mx:6.3f}")
    return ms


def main():
    print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"shape  : M={M}  K={K}  N={N}  2N={2*N}")
    print()
    x_buf, w_buf, W_packed, grad_y = make_inputs()

    correctness_check(x_buf, w_buf, W_packed, grad_y)

    # Warmup all variants (includes torch.compile JIT for V0/V1).
    print("global warmup: 6 s mixed calls ...", flush=True)
    import time
    t0 = time.time()
    while time.time() - t0 < 6.0:
        step_v0(x_buf, w_buf, grad_y)
        step_v1(x_buf, w_buf, grad_y)
        step_v2(x_buf, W_packed, grad_y)
    torch.cuda.synchronize()
    print()

    print("=== fwd timings (1 forward pass per call) ===")
    t_fwd0 = bench(lambda: fwd_v0(x_buf, w_buf),    "V0 baseline_freshbuf fwd")
    t_fwd1 = bench(lambda: fwd_v1(x_buf, w_buf),    "V1 baseline_inplace  fwd")
    t_fwd2 = bench(lambda: fwd_v2(x_buf, W_packed), "V2 save_factors      fwd")
    print()

    print("=== full step timings (fwd + bwd per call) ===")
    t_full0 = bench(lambda: step_v0(x_buf, w_buf,    grad_y), "V0 baseline_freshbuf full")
    t_full1 = bench(lambda: step_v1(x_buf, w_buf,    grad_y), "V1 baseline_inplace  full")
    t_full2 = bench(lambda: step_v2(x_buf, W_packed, grad_y), "V2 save_factors      full")
    print()

    print("=== implied bwd-only (full − fwd) ===")
    print(f"  V0  {(t_full0 - t_fwd0):6.3f} ms")
    print(f"  V1  {(t_full1 - t_fwd1):6.3f} ms")
    print(f"  V2  {(t_full2 - t_fwd2):6.3f} ms")
    print()

    print("=== peak transient allocation per full step (MiB) ===")
    peak0 = measure_peak_alloc(lambda: step_v0(x_buf, w_buf,    grad_y))
    peak1 = measure_peak_alloc(lambda: step_v1(x_buf, w_buf,    grad_y))
    peak2 = measure_peak_alloc(lambda: step_v2(x_buf, W_packed, grad_y))
    print(f"  V0 baseline_freshbuf       peak  +{peak0:7.1f} MiB")
    print(f"  V1 baseline_inplace        peak  +{peak1:7.1f} MiB")
    print(f"  V2 save_factors            peak  +{peak2:7.1f} MiB")
    print()
    print(f"  V0 − V1 (custom autograd savings) : {peak0 - peak1:+7.1f} MiB")
    print(f"  V1 − V2 (residual after in-place) : {peak1 - peak2:+7.1f} MiB")
    print()
    print(f"  reference: M·2N·2 (preact / grad_preact size) = "
          f"{M * 2 * N * 2 / (1024*1024):.1f} MiB")


if __name__ == "__main__":
    main()
