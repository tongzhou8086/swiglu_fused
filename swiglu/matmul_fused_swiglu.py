"""Single-kernel fused SwiGLU: C = silu(gate) * up  where [up|gate] = A @ W.

The weight `W` is a single [K, 2N] bf16 tensor with `up` in the left
half and `gate` in the right — matching what a one-shot cuBLAS GEMM
baseline produces.  The fused kernel never materialises `gate` or
`up` to HBM.

Implementation: dual-TMEM kernel.  Allocates 2*BN cols of TMEM, runs a
unified K-loop of length 2*(K/BK) — first pass accumulates `gate` into
the lower half (reading W[:, N:2N]), second pass accumulates `up` into
the upper half (reading W[:, :N]).  An mbarrier `gate_done` fires
between passes so epilogue warps 4..7 can read the gate accumulator
and apply silu *while the up K-loop is still running*.  Final epilogue
multiplies up * silu(gate) and stores.
"""
import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from ._pycuda_loader import get_module_jit, SM_ARCH
from . import _tma_utils as tma

DTYPE = torch.bfloat16
_HERE = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_HERE, "_matmul_fused_swiglu.cu")
_CUBIN   = os.path.join(_HERE, f"_matmul_fused_swiglu_{SM_ARCH}.cubin")

BM = 128
NW = 8
CTA_GROUP = 2

_CONFIGS = [
    (256, 64, 7, 8),
    (256, 64, 6, 8),
    (256, 64, 5, 8),
    (256, 64, 4, 8),
    (256, 64, 7, 4),
    (256, 64, 7, 16),
]

_MAX_SMEM = 228 * 1024


def _smem(bn, bk, ns):
    # SMEM is reused across phases:
    #   K-loop ring : NS * (BM + BN/CTA_GROUP) * BK * 2
    #   C staging  : BM * (BN + 8) * 2  (aliases the ring; lives in phase B)
    # Allocate max of the two.
    ring = ns * (BM + bn // CTA_GROUP) * bk * 2
    cstg = BM * (bn + 8) * 2
    return max(ring, cstg)


def _legal(bn, bk, ns):
    return _smem(bn, bk, ns) <= _MAX_SMEM


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bn, bk, ns, gsm):
    return f"matmul_fused_swiglu_bm{BM}_bn{bn}_bk{bk}_ns{ns}_gsm{gsm}"


_tmap_cache: dict = {}


def _setup(A, W, N_out, bn, bk):
    key = (A.data_ptr(), W.data_ptr(), bn, bk)
    hit = _tmap_cache.get(key)
    if hit is not None:
        return hit
    M, K = A.shape
    K2, N2 = W.shape
    assert K == K2 and N2 == 2 * N_out
    A_tmap = tma.build_tma_2d(A.data_ptr(), M, K,  BM, 64, tma.SWIZZLE_128B)
    W_tmap = tma.build_tma_2d(W.data_ptr(), K, N2, bk, 64, tma.SWIZZLE_128B)
    _tmap_cache[key] = (A_tmap, W_tmap)
    return A_tmap, W_tmap


def _launch(mod, kname, A, W, bn, bk, smem_bytes):
    M, K = A.shape
    _, N2 = W.shape
    N = N2 // 2
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)
    A_tmap, W_tmap = _setup(A, W, N, bn, bk)

    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    block = (NW * 32, 1, 1)
    grid_x = (M // BM) * (N // bn)
    assert grid_x % CTA_GROUP == 0
    grid = (grid_x, 1, 1)
    fn(A_tmap, W_tmap,
       np.intp(C.data_ptr()),
       np.int32(M), np.int32(N), np.int32(K),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _legal_for_problem(cfg, M, N, K):
    bn, bk, ns, gsm = cfg
    if M % BM != 0 or N % bn != 0 or K % bk != 0:
        return False
    if K // bk < ns:
        return False
    if (M // BM) * (N // bn) % CTA_GROUP != 0:
        return False
    if (M // BM) % 2 != 0:
        return False
    return _legal(bn, bk, ns)


def _tune(A, W):
    M, K = A.shape
    _, N2 = W.shape
    N = N2 // 2
    mod = _get_mod()
    cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)
    for idx, cfg in enumerate(cfgs):
        bn, bk, ns, gsm = cfg
        kn = _kname(bn, bk, ns, gsm)
        sb = _smem(bn, bk, ns)
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda kn=kn, bn=bn, bk=bk, sb=sb:
                    _launch(mod, kn, A, W, bn, bk, sb),
                warmup=20, rep=200, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BN={bn} BK={bk} NS={ns} GSM={gsm}  FAILED: {e}")
            continue
        tflops = 2 * M * N * K * 2 / (ms_med / 1e3) / 1e12   # 2 GEMMs in one kernel
        print(f"  [{idx+1:2d}/{n}] BN={bn:3d} BK={bk:3d} NS={ns} GSM={gsm:2d}  "
              f"{ms_med*1e3:7.1f} µs  {tflops:6.1f} TFLOPS (both GEMMs)")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_fused_swiglu(A, W):
    """Single-kernel SwiGLU.  C = silu(gate) * up  where [up|gate] = A @ W.

    A : [M, K]  bf16
    W : [K, 2N] bf16 — left half = W_up, right half = W_gate
    returns: [M, N] bf16
    """
    M, K = A.shape
    K2, N2 = W.shape
    assert K == K2, f"K mismatch: A.K={K} vs W.K={K2}"
    assert N2 % 2 == 0, f"W's 2nd dim ({N2}) must be even"
    N = N2 // 2
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
        print(f"[fused_swiglu] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(A, W)
        bn, bk, ns, gsm = _best[key]
        print(f"[fused_swiglu] best: BN={bn} BK={bk} NS={ns} GSM={gsm}")
    bn, bk, ns, gsm = _best[key]
    return _launch(_get_mod(), _kname(bn, bk, ns, gsm), A, W, bn, bk, _smem(bn, bk, ns))
