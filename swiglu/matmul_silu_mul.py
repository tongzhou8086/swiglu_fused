"""Fused SwiGLU stage: h = silu(x @ W_gate) * up.

Built on the b42_gsm GEMM (tuned GROUP_SIZE_M). The mainloop accumulates
`gate = x @ W_gate` in TMEM; the epilogue fuses `silu(gate) * up` in fp32
before the bf16 down-cast, so `gate` is never written to HBM.

`up` ( = x @ W_up ) is computed by the caller and passed in.

Autotunes over (BN, BK, NS, GSM), same config space as b42_gsm.
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
_CU_PATH = os.path.join(_HERE, "_matmul_silu_mul.cu")
_CUBIN   = os.path.join(_HERE, f"_matmul_silu_mul_{SM_ARCH}.cubin")

BM = 128
NW = 8
CTA_GROUP = 2

_GSM_VALUES = [1, 4, 8, 16]
_BASE_CONFIGS = [
    (256,  64, 7),
    (256,  64, 6),
    (256,  64, 5),
    (256,  64, 4),
    (256, 128, 3),
]
_CONFIGS = [(bn, bk, ns, gsm)
            for (bn, bk, ns) in _BASE_CONFIGS
            for gsm in _GSM_VALUES]
_BEST_FIELDS = ("BN", "BK", "NS", "GSM")

_MAX_SMEM = 228 * 1024


def _smem(bn, bk, ns):
    return ns * (BM + bn // CTA_GROUP) * bk * 2


def _legal(bn, bk, ns):
    return _smem(bn, bk, ns) <= _MAX_SMEM


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bn, bk, ns, gsm):
    return f"matmul_silu_mul_bm{BM}_bn{bn}_bk{bk}_ns{ns}_gsm{gsm}"


_tmap_cache: dict = {}    # (A_ptr, B_ptr, bn, bk) → (A_tmap, B_tmap)


def _setup(A, B, bn, bk):
    tmap_key = (A.data_ptr(), B.data_ptr(), bn, bk)
    hit = _tmap_cache.get(tmap_key)
    if hit is not None:
        return hit
    M, K = A.shape
    _, N = B.shape
    A_tmap = tma.build_tma_2d(A.data_ptr(), M, K, BM, 64, tma.SWIZZLE_128B)
    B_tmap = tma.build_tma_2d(B.data_ptr(), K, N, bk, 64, tma.SWIZZLE_128B)
    _tmap_cache[tmap_key] = (A_tmap, B_tmap)
    return A_tmap, B_tmap


def _launch(mod, kname, A, B, up, bn, bk, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)
    A_tmap, B_tmap = _setup(A, B, bn, bk)

    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    block = (NW * 32, 1, 1)
    grid_x = (M // BM) * (N // bn)
    assert grid_x % CTA_GROUP == 0
    grid = (grid_x, 1, 1)
    fn(A_tmap, B_tmap,
       np.intp(C.data_ptr()), np.intp(up.data_ptr()),
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


def _tune(A, B, up):
    M, K = A.shape
    _, N = B.shape
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
                    _launch(mod, kn, A, B, up, bn, bk, sb),
                warmup=20, rep=200, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BN={bn} BK={bk} NS={ns} GSM={gsm}  FAILED: {e}")
            continue
        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:2d}/{n}] BN={bn:3d} BK={bk:3d} NS={ns} GSM={gsm:2d}  {tflops:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_silu_mul(A, B_gate, up):
    """h = silu(A @ B_gate) * up.

    A      : [M, K] bf16
    B_gate : [K, N] bf16
    up     : [M, N] bf16  (precomputed x @ W_up)
    returns: [M, N] bf16
    """
    M, K = A.shape
    _, N = B_gate.shape
    assert up.shape == (M, N), f"up shape {tuple(up.shape)} != {(M, N)}"
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
        print(f"[silu_mul] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(A, B_gate, up)
        bn, bk, ns, gsm = _best[key]
        print(f"[silu_mul] best: BN={bn} BK={bk} NS={ns} GSM={gsm}")
    bn, bk, ns, gsm = _best[key]
    return _launch(_get_mod(), _kname(bn, bk, ns, gsm), A, B_gate, up, bn, bk, _smem(bn, bk, ns))
