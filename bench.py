"""Validate + benchmark the fused SwiGLU stage  h = silu(x @ W_gate) * up.

Compares the fused kernel against the unfused baseline (cuBLAS gate GEMM +
torch elementwise silu*mul). `up` (= x @ W_up) is precomputed identically for
both and excluded from timing — we measure only the gate-GEMM + activation
portion that the fusion targets.

Run on a B200:
  srun ... ~/miniconda3/bin/python bench.py --shapes 32768x3072x12288
"""
import argparse
import torch
import torch.nn.functional as F
import triton.testing

from swiglu.matmul_silu_mul import matmul_silu_mul

WARMUP_MS = 200
REP_MS = 2000


def tflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e12


def parse_shape(s):
    p = s.split("x")
    return (int(p[0]), int(p[0]), int(p[0])) if len(p) == 1 else tuple(int(x) for x in p)


def run_shape(M, K, N):
    torch.manual_seed(0)
    x  = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    Wg = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    Wu = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    up = (x @ Wu)  # precomputed, shared by both paths

    # ── Correctness vs fp32 reference ──
    gate_f32 = x.float() @ Wg.float()
    ref = (F.silu(gate_f32) * up.float())
    h = matmul_silu_mul(x, Wg, up).float()
    diff = (h - ref).abs()
    atol = max(1.0, K ** 0.5 / 32)
    ok = torch.allclose(h, ref, rtol=1e-2, atol=atol)
    print(f"  validate: max_abs={diff.max():.3e} mean_abs={diff.mean():.3e}  "
          f"{'OK' if ok else 'FAIL (atol=%.2f)' % atol}")

    # ── Fused timing ──
    ms_fused, _, _ = triton.testing.do_bench(
        lambda: matmul_silu_mul(x, Wg, up), warmup=WARMUP_MS, rep=REP_MS,
        quantiles=(0.5, 0.0, 1.0))

    # ── Unfused baseline: cuBLAS gate GEMM + torch silu*mul ──
    def unfused():
        gate = x @ Wg
        return F.silu(gate) * up
    ms_unfused, _, _ = triton.testing.do_bench(
        unfused, warmup=WARMUP_MS, rep=REP_MS, quantiles=(0.5, 0.0, 1.0))

    print(f"  fused   : {ms_fused:7.3f} ms  ({tflops(M,N,K,ms_fused):7.1f} TFLOPS, gate-GEMM only)")
    print(f"  unfused : {ms_unfused:7.3f} ms  (cuBLAS gate GEMM + torch silu*mul)")
    print(f"  speedup : {ms_unfused/ms_fused:.3f}x")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["32768x3072x12288"],
                   help="MxKxN  (M tokens, K=d_model, N=d_ff)")
    args = p.parse_args()
    print(f"[swiglu] device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for s in args.shapes:
        M, K, N = parse_shape(s)
        print(f"\n=== M={M} K={K} N={N} ===")
        run_shape(M, K, N)


if __name__ == "__main__":
    main()
