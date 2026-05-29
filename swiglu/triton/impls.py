"""B200 experiment: fused Linear + SwiGLU with one wide accumulator.

The normal weight layout is [all left columns | all gate columns].  This kernel
expects the weight to be packed by output chunks:

    [left chunk 0 | gate chunk 0 | left chunk 1 | gate chunk 1 | ...]

That layout lets each CTA compute both halves with one wide dot, split the
accumulator in the epilogue, apply SwiGLU, and store [M, N_HALF].

The current best B200 config from the production shapes is fixed deliberately:
BM=128, BNH=128, BK=64, GROUP_SIZE_M=32, num_warps=8, num_stages=4,
warp_specialize=True.
"""

from __future__ import annotations

import argparse
import functools
import math

import torch
import triton
import triton.language as tl

# NOTE: original code depends on internal MeshyLearning modules.  We don't use
# those symbols anywhere in this file's body, so stub them out for standalone
# benchmarking.
try:
    from MeshyLearning.nn.functional import swiglu                  # type: ignore  # noqa: F401
    from MeshyLearning.nn.triton.runtime_setup import setup_triton  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    swiglu = None
    setup_triton = None


BLOCK_SIZE_M = 128
BLOCK_SIZE_N_HALF = 128
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 32
NUM_WARPS = 8
NUM_STAGES = 4
SAVE_NUM_STAGES = 4
BWD_NUM_STAGES = 2
BWD_FACTORS_BLOCK_SIZE_M = 64
BWD_FACTORS_BLOCK_SIZE_N_HALF = 128
BWD_FACTORS_NUM_WARPS = 4
USE_PTR_FACTORS_GRAD_DE = True
SAVE_FACTORS_NORMAL_BLOCK_SIZE_K = 64
SAVE_FACTORS_NORMAL_GROUP_SIZE_M = 16
SAVE_FACTORS_NORMAL_NUM_STAGES = 4
GATE_SAVE_NUM_STAGES = 4
GATE_RECOMPUTE_NUM_STAGES = 4
WARP_SPECIALIZE = True
USE_TILE_ID_C = False
PACK_BLOCK_K = 16
PACK_NUM_WARPS = 8
BWD_GX_BLOCK_SIZE_M = 64
BWD_GX_BLOCK_SIZE_K = 64
BWD_GX_NUM_WARPS = 4
BWD_GX_NUM_STAGES = 3
BWD_GX_BW_BLOCK_SIZE_M = 64
BWD_GX_BW_BLOCK_SIZE_K = 128
BWD_GX_BW_GROUP_SIZE_M = 16
BWD_GX_BW_NUM_WARPS = 8
BWD_GX_BW_NUM_STAGES = 1
BWD_GX_FACTORS_BW_BLOCK_SIZE_M = 128
BWD_GX_FACTORS_BW_BLOCK_SIZE_K = 128
BWD_GX_FACTORS_BW_GROUP_SIZE_M = 16
BWD_GX_FACTORS_BW_NUM_WARPS = 8
BWD_GX_FACTORS_BW_NUM_STAGES = 1
BWD_GW_BLOCK_SIZE_K = 64
BWD_GW_BLOCK_SIZE_M = 64
BWD_GW_GROUP_SIZE_K = 16
BWD_GW_NUM_WARPS = 8
BWD_GW_NUM_STAGES = 3
ENABLE_PARALLEL_BWD_GEMMS = False
PARALLEL_BWD_GEMMS_MIN_M = 16384


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_allocator() -> None:
    triton.set_allocator(_tma_alloc)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _bwd_gemm_streams(device_index: int):
    with torch.cuda.device(device_index):
        return torch.cuda.Stream(), torch.cuda.Stream()


def _packed_grad_input_weight(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_de: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.view(-1, x.shape[-1])
    if not ENABLE_PARALLEL_BWD_GEMMS or x2.shape[0] < PARALLEL_BWD_GEMMS_MIN_M:
        grad_x = grad_de @ packed_weight.t()
        grad_weight = x2.t().to(packed_weight.dtype) @ grad_de
        return grad_x, grad_weight

    device_index = grad_de.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    current = torch.cuda.current_stream(device_index)
    stream_x, stream_w = _bwd_gemm_streams(device_index)
    stream_x.wait_stream(current)
    stream_w.wait_stream(current)
    with torch.cuda.stream(stream_x):
        grad_x = grad_de @ packed_weight.t()
    with torch.cuda.stream(stream_w):
        grad_weight = x2.t().to(packed_weight.dtype) @ grad_de
    current.wait_stream(stream_x)
    current.wait_stream(stream_w)
    return grad_x, grad_weight


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group: tl.constexpr,
    num_pid_m: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M_
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M_)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _fused_swiglu_wide_packed_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    USE_TILE_ID_C_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        x, gate = tl.split(acc3)
        out = x * (gate * tl.sigmoid(gate))

        if USE_TILE_ID_C_:
            tile_id_c += NUM_SMS
            pid_m, pid_n = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
            )
            offs_m = pid_m * BLOCK_SIZE_M_

        c_desc.store(
            [offs_m, pid_n * BLOCK_SIZE_N_HALF_], out.to(c_ptr.dtype.element_ty)
        )


@triton.jit
def _fused_swiglu_wide_packed_save_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    preact_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2
        preact_rows = offs_m + tl.arange(0, BLOCK_SIZE_M_)
        preact_cols = offs_n2 + tl.arange(0, BLOCK_SIZE_N2)

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        preact_ptrs = (
            preact_ptr
            + preact_rows[:, None] * (N_HALF * 2)
            + preact_cols[None, :]
        )
        tl.store(preact_ptrs, acc.to(preact_ptr.dtype.element_ty))
        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        x, gate = tl.split(acc3)
        out = x * (gate * tl.sigmoid(gate))
        c_desc.store([offs_m, offs_n], out.to(c_ptr.dtype.element_ty))


@triton.jit
def _fused_swiglu_wide_packed_save_gate_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    gate_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    gate_desc = tl.make_tensor_descriptor(
        gate_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        out = left * (gate * tl.sigmoid(gate))
        gate_desc.store([offs_m, offs_n], gate.to(gate_ptr.dtype.element_ty))
        c_desc.store([offs_m, offs_n], out.to(c_ptr.dtype.element_ty))


@triton.jit
def _fused_swiglu_wide_packed_save_factors_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    factors_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        factor_gate = left * silu_prime
        factors_desc.store([offs_m, offs_n2], silu.to(factors_ptr.dtype.element_ty))
        factors_desc.store(
            [offs_m, offs_n2 + BLOCK_SIZE_N_HALF_],
            factor_gate.to(factors_ptr.dtype.element_ty),
        )
        c_desc.store([offs_m, offs_n], (left * silu).to(c_ptr.dtype.element_ty))


@triton.jit
def _fused_swiglu_wide_packed_save_factors_normal_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    factors_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        factors_desc.store([offs_m, offs_n], silu.to(factors_ptr.dtype.element_ty))
        factors_desc.store(
            [offs_m, N_HALF + offs_n],
            (left * silu_prime).to(factors_ptr.dtype.element_ty),
        )
        c_desc.store([offs_m, offs_n], (left * silu).to(c_ptr.dtype.element_ty))


@triton.jit
def _fused_swiglu_wide_normal_save_factors_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    factors_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N_HALF_],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_ * 2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b_left = b_desc.load([offs_k, offs_n])
            b_gate = b_desc.load([offs_k, N_HALF + offs_n])
            b = tl.cat(b_left, b_gate, dim=1)
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        factors_desc.store([offs_m, offs_n], silu.to(factors_ptr.dtype.element_ty))
        factors_desc.store(
            [offs_m, N_HALF + offs_n],
            (left * silu_prime).to(factors_ptr.dtype.element_ty),
        )
        c_desc.store([offs_m, offs_n], (left * silu).to(c_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_kernel(
    a_ptr,
    bp_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        x, gate = tl.split(acc3)

        dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        grad_x = dy * silu
        grad_gate = dy * x * silu_prime
        grad_de = tl.cat(grad_x, grad_gate, dim=1)
        grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_preact_to_factors_inplace_kernel(
    preact_ptr,
    out_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
    preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    preact3 = tl.permute(preact3, (0, 2, 1))
    left, gate = tl.split(preact3)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_prime = sig + silu * (1.0 - sig)
    factor_gate = left * silu_prime
    factors = tl.cat(silu, factor_gate, dim=1)
    preact_desc.store([offs_m, offs_n2], factors.to(preact_ptr.dtype.element_ty))
    out_desc.store([offs_m, offs_n], (left * silu).to(out_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_preact_kernel(
    preact_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
    preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    preact3 = tl.permute(preact3, (0, 2, 1))
    x, gate = tl.split(preact3)

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_prime = sig + silu * (1.0 - sig)
    grad_x = dy * silu
    grad_gate = dy * x * silu_prime
    grad_de = tl.cat(grad_x, grad_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_factors_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    factors = factors_desc.load([offs_m, offs_n2]).to(tl.float32)
    factors3 = tl.reshape(factors, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    factors3 = tl.permute(factors3, (0, 2, 1))
    factor_left, factor_gate = tl.split(factors3)
    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    grad_de = tl.cat(dy * factor_left, dy * factor_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_factors_ptr_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2
    rows = offs_m + tl.arange(0, BLOCK_SIZE_M_)
    cols = tl.arange(0, BLOCK_SIZE_N_HALF_)

    dy = tl.load(
        dy_ptr + rows[:, None] * N_HALF + (offs_n + cols)[None, :]
    ).to(tl.float32)
    factor_left = tl.load(
        factors_ptr + rows[:, None] * (N_HALF * 2) + (offs_n2 + cols)[None, :]
    ).to(tl.float32)
    factor_gate = tl.load(
        factors_ptr
        + rows[:, None] * (N_HALF * 2)
        + (offs_n2 + BLOCK_SIZE_N_HALF_ + cols)[None, :]
    ).to(tl.float32)
    tl.store(
        grad_de_ptr + rows[:, None] * (N_HALF * 2) + (offs_n2 + cols)[None, :],
        (dy * factor_left).to(grad_de_ptr.dtype.element_ty),
    )
    tl.store(
        grad_de_ptr
        + rows[:, None] * (N_HALF * 2)
        + (offs_n2 + BLOCK_SIZE_N_HALF_ + cols)[None, :],
        (dy * factor_gate).to(grad_de_ptr.dtype.element_ty),
    )


@triton.jit
def _swiglu_normal_grad_de_from_packed_factors_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    factors = factors_desc.load([offs_m, offs_n2]).to(tl.float32)
    factors3 = tl.reshape(factors, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    factors3 = tl.permute(factors3, (0, 2, 1))
    factor_left, factor_gate = tl.split(factors3)
    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    grad_desc.store(
        [offs_m, offs_n],
        (dy * factor_left).to(grad_de_ptr.dtype.element_ty),
    )
    grad_desc.store(
        [offs_m, N_HALF + offs_n],
        (dy * factor_gate).to(grad_de_ptr.dtype.element_ty),
    )


@triton.jit
def _swiglu_packed_grad_de_from_normal_factors_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    factor_left = factors_desc.load([offs_m, offs_n]).to(tl.float32)
    factor_gate = factors_desc.load([offs_m, N_HALF + offs_n]).to(tl.float32)
    grad_de = tl.cat(dy * factor_left, dy * factor_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_normal_factors_ptr_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2
    rows = offs_m + tl.arange(0, BLOCK_SIZE_M_)
    cols = tl.arange(0, BLOCK_SIZE_N_HALF_)

    dy = tl.load(
        dy_ptr + rows[:, None] * N_HALF + (offs_n + cols)[None, :]
    ).to(tl.float32)
    factor_left = tl.load(
        factors_ptr + rows[:, None] * (N_HALF * 2) + (offs_n + cols)[None, :]
    ).to(tl.float32)
    factor_gate = tl.load(
        factors_ptr
        + rows[:, None] * (N_HALF * 2)
        + (N_HALF + offs_n + cols)[None, :]
    ).to(tl.float32)
    tl.store(
        grad_de_ptr + rows[:, None] * (N_HALF * 2) + (offs_n2 + cols)[None, :],
        (dy * factor_left).to(grad_de_ptr.dtype.element_ty),
    )
    tl.store(
        grad_de_ptr
        + rows[:, None] * (N_HALF * 2)
        + (offs_n2 + BLOCK_SIZE_N_HALF_ + cols)[None, :],
        (dy * factor_gate).to(grad_de_ptr.dtype.element_ty),
    )


@triton.jit
def _swiglu_normal_grad_de_from_normal_factors_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_

    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    factor_left = factors_desc.load([offs_m, offs_n]).to(tl.float32)
    factor_gate = factors_desc.load([offs_m, N_HALF + offs_n]).to(tl.float32)
    grad_desc.store(
        [offs_m, offs_n],
        (dy * factor_left).to(grad_de_ptr.dtype.element_ty),
    )
    grad_desc.store(
        [offs_m, N_HALF + offs_n],
        (dy * factor_gate).to(grad_de_ptr.dtype.element_ty),
    )


@triton.jit
def _swiglu_normal_grad_de_from_packed_preact_kernel(
    preact_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
    preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    preact3 = tl.permute(preact3, (0, 2, 1))
    left, gate = tl.split(preact3)

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_prime = sig + silu * (1.0 - sig)
    grad_left = dy * silu
    grad_gate = dy * left * silu_prime
    grad_desc.store([offs_m, offs_n], grad_left.to(grad_de_ptr.dtype.element_ty))
    grad_desc.store(
        [offs_m, N_HALF + offs_n], grad_gate.to(grad_de_ptr.dtype.element_ty)
    )


@triton.jit
def _swiglu_packed_grad_de_from_gate_kernel(
    out_ptr,
    gate_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    gate_desc = tl.make_tensor_descriptor(
        gate_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    out = out_desc.load([offs_m, offs_n]).to(tl.float32)
    gate = gate_desc.load([offs_m, offs_n]).to(tl.float32)
    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    left = out / silu
    silu_prime = sig + silu * (1.0 - sig)
    grad_left = dy * silu
    grad_gate = dy * left * silu_prime
    grad_de = tl.cat(grad_left, grad_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_gate_recompute_left_kernel(
    a_ptr,
    bp_ptr,
    gate_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N_HALF_],
    )
    gate_desc = tl.make_tensor_descriptor(
        gate_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        left = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            w_left = bp_desc.load([offs_k, offs_n2])
            left = tl.dot(a, w_left, left)

        gate = gate_desc.load([offs_m, offs_n]).to(tl.float32)
        dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        grad_left = dy * silu
        grad_gate = dy * left * silu_prime
        grad_de = tl.cat(grad_left, grad_gate, dim=1)
        grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_x_from_preact_kernel(
    preact_ptr,
    dy_ptr,
    weight_ptr,
    grad_x_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    n_tiles: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_k = pid_k * BLOCK_SIZE_K_

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    grad_x_desc = tl.make_tensor_descriptor(
        grad_x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )

    acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_K_), dtype=tl.float32)
    for ni in range(n_tiles):
        offs_n = ni * BLOCK_SIZE_N_HALF_
        offs_n2 = ni * BLOCK_SIZE_N2
        preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
        preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        preact3 = tl.permute(preact3, (0, 2, 1))
        left, gate = tl.split(preact3)

        dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        grad_left = (dy * silu).to(weight_ptr.dtype.element_ty)
        grad_gate = (dy * left * silu_prime).to(weight_ptr.dtype.element_ty)

        weight = weight_desc.load([offs_k, offs_n2])
        weight3 = tl.reshape(weight, (BLOCK_SIZE_K_, 2, BLOCK_SIZE_N_HALF_))
        weight3 = tl.permute(weight3, (0, 2, 1))
        w_left, w_gate = tl.split(weight3)

        acc = tl.dot(grad_left, tl.trans(w_left), acc)
        acc = tl.dot(grad_gate, tl.trans(w_gate), acc)

    grad_x_desc.store([offs_m, offs_k], acc.to(grad_x_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_x_from_preact_bw_kernel(
    preact_ptr,
    dy_ptr,
    weight_ptr,
    grad_x_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_K_OUT_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_OUT_)
    n_tiles: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_k
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_k

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_OUT_, BLOCK_SIZE_N2],
    )
    grad_x_desc = tl.make_tensor_descriptor(
        grad_x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_OUT_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_k = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_k = pid_k * BLOCK_SIZE_K_OUT_

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_K_OUT_), dtype=tl.float32)
        for ni in range(n_tiles):
            offs_n = ni * BLOCK_SIZE_N_HALF_
            offs_n2 = ni * BLOCK_SIZE_N2

            preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
            preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
            preact3 = tl.permute(preact3, (0, 2, 1))
            left, gate = tl.split(preact3)

            dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
            sig = tl.sigmoid(gate)
            silu = gate * sig
            silu_prime = sig + silu * (1.0 - sig)
            grad_left = (dy * silu).to(weight_ptr.dtype.element_ty)
            grad_gate = (dy * left * silu_prime).to(weight_ptr.dtype.element_ty)
            grad_de = tl.cat(grad_left, grad_gate, dim=1)

            weight = weight_desc.load([offs_k, offs_n2])
            acc = tl.dot(grad_de, tl.trans(weight), acc)

        grad_x_desc.store([offs_m, offs_k], acc.to(grad_x_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_x_from_factors_bw_kernel(
    factors_ptr,
    dy_ptr,
    weight_ptr,
    grad_x_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_K_OUT_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_OUT_)
    n_tiles: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_k
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_k

    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_OUT_, BLOCK_SIZE_N2],
    )
    grad_x_desc = tl.make_tensor_descriptor(
        grad_x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_OUT_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_k = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_k = pid_k * BLOCK_SIZE_K_OUT_

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_K_OUT_), dtype=tl.float32)
        for ni in range(n_tiles):
            offs_n = ni * BLOCK_SIZE_N_HALF_
            offs_n2 = ni * BLOCK_SIZE_N2

            factors = factors_desc.load([offs_m, offs_n2]).to(tl.float32)
            factors3 = tl.reshape(factors, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
            factors3 = tl.permute(factors3, (0, 2, 1))
            factor_left, factor_gate = tl.split(factors3)

            dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
            grad_left = (dy * factor_left).to(weight_ptr.dtype.element_ty)
            grad_gate = (dy * factor_gate).to(weight_ptr.dtype.element_ty)
            grad_de = tl.cat(grad_left, grad_gate, dim=1)

            weight = weight_desc.load([offs_k, offs_n2])
            acc = tl.dot(grad_de, tl.trans(weight), acc)

        grad_x_desc.store([offs_m, offs_k], acc.to(grad_x_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_weight_from_preact_kernel(
    x_ptr,
    preact_ptr,
    dy_ptr,
    grad_weight_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    GROUP_SIZE_K_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    m_tiles: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_tiles: tl.constexpr = num_pid_k * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_K_ * num_pid_n

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_weight_desc = tl.make_tensor_descriptor(
        grad_weight_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[
            BLOCK_SIZE_K_,
            BLOCK_SIZE_N2 // 2 if EPILOGUE_SUBTILE else BLOCK_SIZE_N2,
        ],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_k, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_k, GROUP_SIZE_K_
        )
        offs_k = pid_k * BLOCK_SIZE_K_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_K_, BLOCK_SIZE_N2), dtype=tl.float32)
        for mi in range(m_tiles):
            offs_m = mi * BLOCK_SIZE_M_
            x = x_desc.load([offs_m, offs_k])
            preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
            preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
            preact3 = tl.permute(preact3, (0, 2, 1))
            left, gate = tl.split(preact3)

            dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
            sig = tl.sigmoid(gate)
            silu = gate * sig
            silu_prime = sig + silu * (1.0 - sig)
            grad_left = (dy * silu).to(x_ptr.dtype.element_ty)
            grad_gate = (dy * left * silu_prime).to(x_ptr.dtype.element_ty)
            grad_de = tl.cat(grad_left, grad_gate, dim=1)

            acc = tl.dot(tl.trans(x), grad_de, acc)

        if EPILOGUE_SUBTILE:
            acc3 = tl.reshape(acc, (BLOCK_SIZE_K_, 2, BLOCK_SIZE_N_HALF_))
            acc3 = tl.permute(acc3, (0, 2, 1))
            acc0, acc1 = tl.split(acc3)
            grad_weight_desc.store(
                [offs_k, offs_n2], acc0.to(grad_weight_ptr.dtype.element_ty)
            )
            grad_weight_desc.store(
                [offs_k, offs_n2 + BLOCK_SIZE_N_HALF_],
                acc1.to(grad_weight_ptr.dtype.element_ty),
            )
        else:
            grad_weight_desc.store(
                [offs_k, offs_n2], acc.to(grad_weight_ptr.dtype.element_ty)
            )


@triton.jit
def _swiglu_packed_grad_weight_from_factors_kernel(
    x_ptr,
    factors_ptr,
    dy_ptr,
    grad_weight_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    GROUP_SIZE_K_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_k: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    m_tiles: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_tiles: tl.constexpr = num_pid_k * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_K_ * num_pid_n

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_weight_desc = tl.make_tensor_descriptor(
        grad_weight_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[
            BLOCK_SIZE_K_,
            BLOCK_SIZE_N2 // 2 if EPILOGUE_SUBTILE else BLOCK_SIZE_N2,
        ],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_k, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_k, GROUP_SIZE_K_
        )
        offs_k = pid_k * BLOCK_SIZE_K_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_K_, BLOCK_SIZE_N2), dtype=tl.float32)
        for mi in range(m_tiles):
            offs_m = mi * BLOCK_SIZE_M_
            x = x_desc.load([offs_m, offs_k])
            factors = factors_desc.load([offs_m, offs_n2]).to(tl.float32)
            factors3 = tl.reshape(factors, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
            factors3 = tl.permute(factors3, (0, 2, 1))
            factor_left, factor_gate = tl.split(factors3)

            dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
            grad_left = (dy * factor_left).to(x_ptr.dtype.element_ty)
            grad_gate = (dy * factor_gate).to(x_ptr.dtype.element_ty)
            grad_de = tl.cat(grad_left, grad_gate, dim=1)

            acc = tl.dot(tl.trans(x), grad_de, acc)

        if EPILOGUE_SUBTILE:
            acc3 = tl.reshape(acc, (BLOCK_SIZE_K_, 2, BLOCK_SIZE_N_HALF_))
            acc3 = tl.permute(acc3, (0, 2, 1))
            acc0, acc1 = tl.split(acc3)
            grad_weight_desc.store(
                [offs_k, offs_n2], acc0.to(grad_weight_ptr.dtype.element_ty)
            )
            grad_weight_desc.store(
                [offs_k, offs_n2 + BLOCK_SIZE_N_HALF_],
                acc1.to(grad_weight_ptr.dtype.element_ty),
            )
        else:
            grad_weight_desc.store(
                [offs_k, offs_n2], acc.to(grad_weight_ptr.dtype.element_ty)
            )


@triton.jit
def _pack_swiglu_weight_chunked_kernel(
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_HALF_: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    N2: tl.constexpr = N_HALF * 2

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N_HALF_ + tl.arange(0, BLOCK_N_HALF_)
    dst_base = pid_n * BLOCK_N_HALF_ * 2

    src_left = offs_k[:, None] * N2 + offs_n[None, :]
    src_gate = offs_k[:, None] * N2 + (N_HALF + offs_n)[None, :]
    dst_left = offs_k[:, None] * N2 + (dst_base + tl.arange(0, BLOCK_N_HALF_))[None, :]
    dst_gate = (
        offs_k[:, None] * N2
        + (dst_base + BLOCK_N_HALF_ + tl.arange(0, BLOCK_N_HALF_))[None, :]
    )
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_HALF)

    left = tl.load(weight_ptr + src_left, mask=mask)
    gate = tl.load(weight_ptr + src_gate, mask=mask)
    tl.store(out_ptr + dst_left, left, mask=mask)
    tl.store(out_ptr + dst_gate, gate, mask=mask)


def pack_swiglu_weight_chunked_into(
    weight: torch.Tensor,
    out: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N_HALF] SwiGLU weight into a preallocated output buffer."""
    assert weight.is_cuda and weight.is_contiguous()
    assert out.is_cuda and out.is_contiguous()
    assert out.shape == weight.shape and out.dtype == weight.dtype
    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    chunks = n_half // block_n_half
    left = weight[:, :n_half].view(k, chunks, block_n_half)
    gate = weight[:, n_half:].view(k, chunks, block_n_half)
    packed = out.view(k, chunks, 2, block_n_half)
    packed[:, :, 0, :].copy_(left)
    packed[:, :, 1, :].copy_(gate)
    return out


def pack_swiglu_weight_chunked_into_triton(
    weight: torch.Tensor,
    out: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N_HALF] with a custom Triton copy kernel."""
    assert weight.is_cuda and weight.is_contiguous()
    assert out.is_cuda and out.is_contiguous()
    assert out.shape == weight.shape and out.dtype == weight.dtype
    assert block_n_half == BLOCK_SIZE_N_HALF

    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    grid = (triton.cdiv(k, PACK_BLOCK_K), triton.cdiv(n_half, block_n_half))
    _pack_swiglu_weight_chunked_kernel[grid](
        weight,
        out,
        k,
        n_half,
        BLOCK_K=PACK_BLOCK_K,
        BLOCK_N_HALF_=block_n_half,
        num_warps=PACK_NUM_WARPS,
    )
    return out


def pack_swiglu_weight_chunked(
    weight: torch.Tensor, block_n_half: int = BLOCK_SIZE_N_HALF
) -> torch.Tensor:
    """Pack [K, 2*N_HALF] SwiGLU weight into chunk-interleaved layout."""
    return pack_swiglu_weight_chunked_into_triton(
        weight, torch.empty_like(weight), block_n_half
    )


def pack_swiglu_weight_chunked_torch(
    weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N_HALF] on CPU or GPU using regular torch copies."""
    assert weight.is_contiguous()
    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    out = torch.empty_like(weight, requires_grad=False)
    chunks = n_half // block_n_half
    left = weight[:, :n_half].view(k, chunks, block_n_half)
    gate = weight[:, n_half:].view(k, chunks, block_n_half)
    packed = out.view(k, chunks, 2, block_n_half)
    packed[:, :, 0, :].copy_(left)
    packed[:, :, 1, :].copy_(gate)
    return out


def unpack_swiglu_weight_chunked_torch(
    packed_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Undo chunk-interleaved packing back to [K, 2*N_HALF]."""
    assert packed_weight.is_contiguous()
    k, n2 = packed_weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    out = torch.empty_like(packed_weight, requires_grad=False)
    chunks = n_half // block_n_half
    packed = packed_weight.view(k, chunks, 2, block_n_half)
    left = out[:, :n_half].view(k, chunks, block_n_half)
    gate = out[:, n_half:].view(k, chunks, block_n_half)
    left.copy_(packed[:, :, 0, :])
    gate.copy_(packed[:, :, 1, :])
    return out


def pack_swiglu_linear_weight(
    linear_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack a normal Linear weight [2*N_HALF, K] into internal [K, 2*N_HALF]."""
    assert linear_weight.ndim == 2
    return pack_swiglu_weight_chunked_torch(
        linear_weight.t().contiguous(), block_n_half
    )


def unpack_swiglu_linear_weight(
    packed_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Return a normal Linear weight [2*N_HALF, K] from internal packed storage."""
    assert packed_weight.ndim == 2
    return unpack_swiglu_weight_chunked_torch(
        packed_weight.contiguous(), block_n_half
    ).t().contiguous()


def fused_swiglu_wide_packed(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    use_tile_id_c: bool = USE_TILE_ID_C,
) -> torch.Tensor:
    """Compute swiglu(x @ weight) with a prepacked [K, 2*N_HALF] weight."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_kernel[grid](
        x,
        packed_weight,
        out,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        USE_TILE_ID_C_=use_tile_id_c,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def fused_swiglu_wide_packed_save_preact(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused SwiGLU projection and save packed preactivation."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    preact = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_save_kernel[grid](
        x,
        packed_weight,
        out,
        preact,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=SAVE_NUM_STAGES,
    )
    return out, preact


def fused_swiglu_wide_packed_save_gate(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused SwiGLU projection and save only gate preactivation."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    gate = torch.empty_like(out)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_save_gate_kernel[grid](
        x,
        packed_weight,
        out,
        gate,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=GATE_SAVE_NUM_STAGES,
    )
    return out, gate


def fused_swiglu_wide_packed_save_factors(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    factors_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused SwiGLU projection and save packed backward factors."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    if factors_dtype is None:
        factors_dtype = x.dtype
    factors = torch.empty((m, n2), device=x.device, dtype=factors_dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_save_factors_kernel[grid](
        x,
        packed_weight,
        out,
        factors,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=SAVE_NUM_STAGES,
    )
    return out, factors


def fused_swiglu_wide_packed_save_factors_normal(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    factors_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused projection and save backward factors in normal layout."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    if factors_dtype is None:
        factors_dtype = x.dtype
    factors = torch.empty((m, n2), device=x.device, dtype=factors_dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_save_factors_normal_kernel[grid](
        x,
        packed_weight,
        out,
        factors,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=SAVE_FACTORS_NORMAL_BLOCK_SIZE_K,
        GROUP_SIZE_M_=SAVE_FACTORS_NORMAL_GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=SAVE_FACTORS_NORMAL_NUM_STAGES,
    )
    return out, factors


def fused_swiglu_wide_normal_save_factors(
    x: torch.Tensor,
    weight: torch.Tensor,
    factors_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused projection from normal-layout weight and save normal factors."""
    _ensure_allocator()
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()

    m, k = x.shape
    k2, n2 = weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    if factors_dtype is None:
        factors_dtype = x.dtype
    factors = torch.empty((m, n2), device=x.device, dtype=factors_dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_normal_save_factors_kernel[grid](
        x,
        weight,
        out,
        factors,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=SAVE_NUM_STAGES,
    )
    return out, factors


def swiglu_packed_grad_de(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute d(swiglu)/d(x@W) in the packed column layout."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda and grad_out.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _swiglu_packed_grad_de_kernel[grid](
        x,
        packed_weight,
        grad_out,
        grad_de,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=BWD_NUM_STAGES,
    )
    return grad_de


def swiglu_packed_grad_de_cublas_recompute(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_de by cuBLAS recompute + Triton derivative epilogue."""
    assert x.is_cuda and packed_weight.is_cuda and grad_out.is_cuda
    if not x.is_contiguous():
        x = x.contiguous()
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
    preact = x @ packed_weight
    return swiglu_packed_grad_de_from_preact(preact, grad_out)


def swiglu_packed_preact_to_factors_inplace(
    preact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn packed preactivation into packed backward factors in place."""
    _ensure_allocator()
    assert preact.is_cuda and preact.is_contiguous()

    m, n2 = preact.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=preact.device, dtype=preact.dtype)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_packed_preact_to_factors_inplace_kernel[grid](
        preact,
        out,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return out, preact


def fused_swiglu_wide_packed_cublas_save_factors(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cuBLAS GEMM followed by in-place packed factors epilogue."""
    assert x.is_cuda and packed_weight.is_cuda
    if not x.is_contiguous():
        x = x.contiguous()
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
    preact = x @ packed_weight
    return swiglu_packed_preact_to_factors_inplace(preact)


def swiglu_packed_grad_de_from_preact(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_de from saved packed preactivation."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda
    assert preact.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty_like(preact)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_packed_grad_de_from_preact_kernel[grid](
        preact,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return grad_de


def swiglu_packed_grad_de_from_factors(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_de from saved packed backward factors."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=factors.device, dtype=grad_out.dtype)
    grid = (
        triton.cdiv(m, BWD_FACTORS_BLOCK_SIZE_M)
        * triton.cdiv(n_half, BWD_FACTORS_BLOCK_SIZE_N_HALF),
    )
    if USE_PTR_FACTORS_GRAD_DE:
        _swiglu_packed_grad_de_from_factors_ptr_kernel[grid](
            factors,
            grad_out,
            grad_de,
            m,
            n_half,
            BLOCK_SIZE_M_=BWD_FACTORS_BLOCK_SIZE_M,
            BLOCK_SIZE_N_HALF_=BWD_FACTORS_BLOCK_SIZE_N_HALF,
            num_warps=BWD_FACTORS_NUM_WARPS,
        )
    else:
        _swiglu_packed_grad_de_from_factors_kernel[grid](
            factors,
            grad_out,
            grad_de,
            m,
            n_half,
            BLOCK_SIZE_M_=BWD_FACTORS_BLOCK_SIZE_M,
            BLOCK_SIZE_N_HALF_=BWD_FACTORS_BLOCK_SIZE_N_HALF,
            num_warps=BWD_FACTORS_NUM_WARPS,
        )
    return grad_de


def swiglu_packed_grad_de_from_factors_inplace(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Overwrite packed factors with packed grad_de and return the same tensor."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert USE_PTR_FACTORS_GRAD_DE

    grid = (
        triton.cdiv(m, BWD_FACTORS_BLOCK_SIZE_M)
        * triton.cdiv(n_half, BWD_FACTORS_BLOCK_SIZE_N_HALF),
    )
    _swiglu_packed_grad_de_from_factors_ptr_kernel[grid](
        factors,
        grad_out,
        factors,
        m,
        n_half,
        BLOCK_SIZE_M_=BWD_FACTORS_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BWD_FACTORS_BLOCK_SIZE_N_HALF,
        num_warps=BWD_FACTORS_NUM_WARPS,
    )
    return factors


def swiglu_normal_grad_de_from_packed_factors(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute normal [all-left | all-gate] grad_de from packed factors."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=factors.device, dtype=grad_out.dtype)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_normal_grad_de_from_packed_factors_kernel[grid](
        factors,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return grad_de


def swiglu_packed_grad_de_from_normal_factors(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_de from normal-layout saved factors."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=factors.device, dtype=grad_out.dtype)
    grid = (
        triton.cdiv(m, BWD_FACTORS_BLOCK_SIZE_M)
        * triton.cdiv(n_half, BWD_FACTORS_BLOCK_SIZE_N_HALF),
    )
    _swiglu_packed_grad_de_from_normal_factors_ptr_kernel[grid](
        factors,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BWD_FACTORS_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BWD_FACTORS_BLOCK_SIZE_N_HALF,
        num_warps=BWD_FACTORS_NUM_WARPS,
    )
    return grad_de


def swiglu_normal_grad_de_from_normal_factors(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute normal-layout grad_de from normal-layout saved factors."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=factors.device, dtype=grad_out.dtype)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_normal_grad_de_from_normal_factors_kernel[grid](
        factors,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return grad_de


def swiglu_normal_grad_de_from_packed_preact(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute normal [all-left | all-gate] grad_de from packed preactivation."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda
    assert preact.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty_like(preact)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_normal_grad_de_from_packed_preact_kernel[grid](
        preact,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return grad_de


def swiglu_packed_grad_de_from_gate(
    out: torch.Tensor,
    gate: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Approximate packed grad_de from saved output and gate preactivation."""
    _ensure_allocator()
    assert out.is_cuda and gate.is_cuda and grad_out.is_cuda
    assert out.is_contiguous() and gate.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n_half = out.shape
    assert gate.shape == out.shape
    assert grad_out.shape == out.shape
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n_half * 2), device=out.device, dtype=out.dtype)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_packed_grad_de_from_gate_kernel[grid](
        out,
        gate,
        grad_out,
        grad_de,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return grad_de


def swiglu_packed_grad_de_from_gate_recompute_left(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    gate: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_de from saved gate and recomputed left preactivation."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda and gate.is_cuda and grad_out.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous() and gate.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert gate.shape == (m, n_half)
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grad_de = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _swiglu_packed_grad_de_from_gate_recompute_left_kernel[grid](
        x,
        packed_weight,
        gate,
        grad_out,
        grad_de,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=GATE_RECOMPUTE_NUM_STAGES,
    )
    return grad_de


def swiglu_packed_grad_x_from_preact(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute grad_x without materializing packed grad_de."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda and packed_weight.is_cuda
    assert preact.is_contiguous() and packed_weight.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    k, n2_w = packed_weight.shape
    assert n2 == n2_w and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert k % BWD_GX_BLOCK_SIZE_K == 0

    grad_x = torch.empty((m, k), device=preact.device, dtype=preact.dtype)
    grid = (
        triton.cdiv(m, BWD_GX_BLOCK_SIZE_M),
        triton.cdiv(k, BWD_GX_BLOCK_SIZE_K),
    )
    _swiglu_packed_grad_x_from_preact_kernel[grid](
        preact,
        grad_out,
        packed_weight,
        grad_x,
        m,
        n_half,
        k,
        BLOCK_SIZE_M_=BWD_GX_BLOCK_SIZE_M,
        BLOCK_SIZE_K_=BWD_GX_BLOCK_SIZE_K,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=BWD_GX_NUM_WARPS,
        num_stages=BWD_GX_NUM_STAGES,
    )
    return grad_x


def swiglu_packed_grad_x_from_preact_bw(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Experimental Blackwell/TMA persistent grad_x without materialized grad_de."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda and packed_weight.is_cuda
    assert preact.is_contiguous() and packed_weight.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    k, n2_w = packed_weight.shape
    assert n2 == n2_w and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert k % BWD_GX_BW_BLOCK_SIZE_K == 0

    grad_x = torch.empty((m, k), device=preact.device, dtype=preact.dtype)
    device_index = preact.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BWD_GX_BW_BLOCK_SIZE_M)
            * triton.cdiv(k, BWD_GX_BW_BLOCK_SIZE_K),
        ),
    )
    _swiglu_packed_grad_x_from_preact_bw_kernel[grid](
        preact,
        grad_out,
        packed_weight,
        grad_x,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BWD_GX_BW_BLOCK_SIZE_M,
        BLOCK_SIZE_K_OUT_=BWD_GX_BW_BLOCK_SIZE_K,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        GROUP_SIZE_M_=BWD_GX_BW_GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=BWD_GX_BW_NUM_WARPS,
        num_stages=BWD_GX_BW_NUM_STAGES,
    )
    return grad_x


def swiglu_packed_grad_x_from_factors_bw(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Experimental Blackwell/TMA grad_x from saved factors without grad_de."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda and packed_weight.is_cuda
    assert factors.is_contiguous() and packed_weight.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    k, n2_w = packed_weight.shape
    assert n2 == n2_w and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert k % BWD_GX_FACTORS_BW_BLOCK_SIZE_K == 0

    grad_x = torch.empty((m, k), device=factors.device, dtype=grad_out.dtype)
    device_index = factors.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BWD_GX_FACTORS_BW_BLOCK_SIZE_M)
            * triton.cdiv(k, BWD_GX_FACTORS_BW_BLOCK_SIZE_K),
        ),
    )
    _swiglu_packed_grad_x_from_factors_bw_kernel[grid](
        factors,
        grad_out,
        packed_weight,
        grad_x,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BWD_GX_FACTORS_BW_BLOCK_SIZE_M,
        BLOCK_SIZE_K_OUT_=BWD_GX_FACTORS_BW_BLOCK_SIZE_K,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        GROUP_SIZE_M_=BWD_GX_FACTORS_BW_GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=BWD_GX_FACTORS_BW_NUM_WARPS,
        num_stages=BWD_GX_FACTORS_BW_NUM_STAGES,
    )
    return grad_x


def swiglu_packed_grad_weight_from_preact(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_weight while generating packed grad_de on the fly."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda and x.is_cuda
    assert preact.is_contiguous() and x.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    m_x, k = x.shape
    assert m == m_x and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert k % BWD_GW_BLOCK_SIZE_K == 0
    assert m % BWD_GW_BLOCK_SIZE_M == 0

    grad_weight = torch.empty((k, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(k, BWD_GW_BLOCK_SIZE_K)
            * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _swiglu_packed_grad_weight_from_preact_kernel[grid](
        x,
        preact,
        grad_out,
        grad_weight,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_K_=BWD_GW_BLOCK_SIZE_K,
        BLOCK_SIZE_M_=BWD_GW_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        GROUP_SIZE_K_=BWD_GW_GROUP_SIZE_K,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        EPILOGUE_SUBTILE=True,
        FLATTEN=True,
        num_warps=BWD_GW_NUM_WARPS,
        num_stages=BWD_GW_NUM_STAGES,
    )
    return grad_weight


def swiglu_packed_grad_weight_from_factors(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grad_weight while generating packed grad_de from factors."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda and x.is_cuda
    assert factors.is_contiguous() and x.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    m_x, k = x.shape
    assert m == m_x and n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0
    assert k % BWD_GW_BLOCK_SIZE_K == 0
    assert m % BWD_GW_BLOCK_SIZE_M == 0

    grad_weight = torch.empty((k, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(k, BWD_GW_BLOCK_SIZE_K)
            * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _swiglu_packed_grad_weight_from_factors_kernel[grid](
        x,
        factors,
        grad_out,
        grad_weight,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_K_=BWD_GW_BLOCK_SIZE_K,
        BLOCK_SIZE_M_=BWD_GW_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        GROUP_SIZE_K_=BWD_GW_GROUP_SIZE_K,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        EPILOGUE_SUBTILE=True,
        FLATTEN=True,
        num_warps=BWD_GW_NUM_WARPS,
        num_stages=BWD_GW_NUM_STAGES,
    )
    return grad_weight


class _FusedSwiGLUPackedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        ctx.save_for_backward(x, packed_weight)
        ctx.x_shape = x.shape
        return fused_swiglu_wide_packed(x, packed_weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de(x, packed_weight, grad_out)
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Autograd wrapper for packed-native SwiGLU projection."""
    return _FusedSwiGLUPackedFn.apply(x, packed_weight)


class _FusedSwiGLUPackedCublasRecomputeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        ctx.save_for_backward(x, packed_weight)
        ctx.x_shape = x.shape
        return fused_swiglu_wide_packed(x, packed_weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_cublas_recompute(x, packed_weight, grad_out)
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_cublas_recompute_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """No-save forward; backward recomputes preactivation with cuBLAS."""
    return _FusedSwiGLUPackedCublasRecomputeFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, preact = fused_swiglu_wide_packed_save_preact(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, preact)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, preact = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_preact(preact, grad_out)
        ctx.maybe_clear_saved_tensors()
        del preact
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Autograd wrapper that saves packed preactivation for faster backward."""
    return _FusedSwiGLUPackedSaveFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFactorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_factors_inplace(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_factors_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Autograd wrapper that saves packed backward factors."""
    return _FusedSwiGLUPackedSaveFactorsFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFactorsOnflyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_x = swiglu_packed_grad_x_from_factors_bw(factors, grad_out, packed_weight)
        grad_weight = swiglu_packed_grad_weight_from_factors(factors, grad_out, x)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_factors_onfly_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Save factors forward, then compute both backward GEMMs on the fly."""
    return _FusedSwiGLUPackedSaveFactorsOnflyFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFactorsCuteBwdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        try:
            from cute_swiglu_factors_bwd import swiglu_factors_backward_cute

            grad_x, grad_weight = swiglu_factors_backward_cute(
                x, factors, grad_out, packed_weight
            )
        except Exception:
            grad_de = swiglu_packed_grad_de_from_factors_inplace(factors, grad_out)
            grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_factors_cute_bwd_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Experimental CuTe backward: generate dy*factors inside GEMM mainloops."""
    return _FusedSwiGLUPackedSaveFactorsCuteBwdFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFactorsNormalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors_normal(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_normal_factors(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        del factors
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_factors_normal_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Packed-weight autograd wrapper with normal-layout saved factors."""
    return _FusedSwiGLUPackedSaveFactorsNormalFn.apply(x, packed_weight)


class _FusedSwiGLUPackedCuBLASSaveFactorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_cublas_save_factors(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_factors_inplace(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_cublas_save_factors_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Autograd wrapper for cuBLAS GEMM plus in-place factors epilogue."""
    return _FusedSwiGLUPackedCuBLASSaveFactorsFn.apply(x, packed_weight)


class _FusedSwiGLUNormalWeightSavePreactFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        packed_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        out, preact = fused_swiglu_wide_packed_save_preact(x, packed_weight)
        ctx.save_for_backward(x, weight, preact)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, preact = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_normal_grad_de_from_packed_preact(preact, grad_out)
        ctx.maybe_clear_saved_tensors()
        del preact
        grad_x = grad_de @ weight.t()
        grad_weight = x.view(-1, x.shape[-1]).t().to(weight.dtype) @ grad_de
        return grad_x.view(ctx.x_shape), grad_weight, None


def fused_swiglu_normal_weight_save_preact_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Autograd wrapper for normal-layout weight with packed forward cache."""
    return _FusedSwiGLUNormalWeightSavePreactFn.apply(x, weight, packed_weight)


class _FusedSwiGLUNormalWeightSaveFactorsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        packed_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors(x, packed_weight)
        ctx.save_for_backward(x, weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_normal_grad_de_from_packed_factors(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        del factors
        grad_x = grad_de @ weight.t()
        grad_weight = x.view(-1, x.shape[-1]).t().to(weight.dtype) @ grad_de
        return grad_x.view(ctx.x_shape), grad_weight, None


def fused_swiglu_normal_weight_save_factors_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Autograd wrapper for normal-layout weight with saved backward factors."""
    return _FusedSwiGLUNormalWeightSaveFactorsFn.apply(x, weight, packed_weight)


class _FusedSwiGLUNormalWeightSaveFactorsNormalFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        packed_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors_normal(x, packed_weight)
        ctx.save_for_backward(x, weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_normal_grad_de_from_normal_factors(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        del factors
        x2 = x.view(-1, x.shape[-1])
        grad_x = grad_de @ weight.t()
        grad_weight = x2.t().to(weight.dtype) @ grad_de
        return grad_x.view(ctx.x_shape), grad_weight, None


def fused_swiglu_normal_weight_save_factors_normal_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Normal-layout weight path with normal-layout saved backward factors."""
    return _FusedSwiGLUNormalWeightSaveFactorsNormalFn.apply(x, weight, packed_weight)


class _FusedSwiGLUNormalWeightOnLoadSaveFactorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        out, factors = fused_swiglu_wide_normal_save_factors(x, weight)
        ctx.save_for_backward(x, weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_normal_grad_de_from_normal_factors(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        del factors
        x2 = x.view(-1, x.shape[-1])
        grad_x = grad_de @ weight.t()
        grad_weight = x2.t().to(weight.dtype) @ grad_de
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_normal_weight_onload_save_factors_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Normal-layout weight path that packs left/gate tiles inside the kernel."""
    return _FusedSwiGLUNormalWeightOnLoadSaveFactorsFn.apply(x, weight)


class _FusedSwiGLUPackedSaveGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, gate = fused_swiglu_wide_packed_save_gate(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, gate)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, gate = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_gate_recompute_left(
            x, packed_weight, gate, grad_out
        )
        ctx.maybe_clear_saved_tensors()
        del gate
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_gate_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Autograd wrapper that saves gate and recomputes left exactly."""
    return _FusedSwiGLUPackedSaveGateFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveGateOutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, gate = fused_swiglu_wide_packed_save_gate(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, out, gate)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, out, gate = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_gate(out, gate, grad_out)
        ctx.maybe_clear_saved_tensors()
        del out, gate
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_gate_out_autograd(
    x: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    """Save gate plus output reference, then derive left as out / silu(gate)."""
    return _FusedSwiGLUPackedSaveGateOutFn.apply(x, packed_weight)


class PackedSwiGLULinear(torch.nn.Module):
    """SwiGLU input projection with selectable internal parameter layout.

    Internal parameter layout:
        packed: [K, 2*N_HALF], chunk-interleaved for native packed training.
        normal: [K, 2*N_HALF], [all left columns | all gate columns].

    The packed layout is the speed-first default for this experiment. The normal
    layout remains available when optimizer/checkpoint ergonomics matter more
    than the last few tenths of a millisecond.

    State-dict layout:
        weight: [2*N_HALF, K], matching torch.nn.Linear and existing checkpoints.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        block_n_half: int = BLOCK_SIZE_N_HALF,
        backward_mode: str = "save_factors_normal",
        parameter_layout: str = "packed",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert backward_mode in {
            "save_preact",
            "save_factors",
            "save_factors_normal",
            "save_factors_cute_bwd",
            "onload_save_factors",
            "cublas_save_factors",
        }
        assert parameter_layout in {"packed", "normal"}
        assert hidden_features % block_n_half == 0
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.block_n_half = block_n_half
        self.backward_mode = backward_mode
        self.parameter_layout = parameter_layout
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty(
                in_features,
                hidden_features * 2,
                **factory_kwargs,
            )
        )
        self.register_buffer(
            "_packed_weight_cache",
            torch.empty(0, **factory_kwargs),
            persistent=False,
        )
        self._packed_weight_cache_version = -1
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        *,
        block_n_half: int = BLOCK_SIZE_N_HALF,
        backward_mode: str = "save_factors_normal",
        parameter_layout: str = "packed",
    ) -> "PackedSwiGLULinear":
        assert linear.bias is None
        assert linear.out_features % 2 == 0
        module = cls(
            linear.in_features,
            linear.out_features // 2,
            block_n_half=block_n_half,
            backward_mode=backward_mode,
            parameter_layout=parameter_layout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.load_linear_weight_(linear.weight)
        return module

    @torch.no_grad()
    def reset_parameters(self) -> None:
        normal_weight = torch.empty(
            self.hidden_features * 2,
            self.in_features,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        torch.nn.init.normal_(
            normal_weight[: self.hidden_features],
            std=1.0 / math.sqrt(self.in_features),
        )
        torch.nn.init.kaiming_normal_(normal_weight[self.hidden_features :])
        self.load_linear_weight_(normal_weight)

    @torch.no_grad()
    def load_linear_weight_(self, linear_weight: torch.Tensor) -> None:
        expected = (self.hidden_features * 2, self.in_features)
        assert tuple(linear_weight.shape) == expected
        linear_weight = linear_weight.detach().to(
            device=self.weight.device, dtype=self.weight.dtype
        ).contiguous()
        if self.parameter_layout == "packed":
            self.weight.copy_(pack_swiglu_linear_weight(linear_weight, self.block_n_half))
        else:
            self.weight.copy_(linear_weight.t().contiguous())
            self._packed_weight_cache_version = -1

    def linear_weight(self) -> torch.Tensor:
        with torch.no_grad():
            if self.parameter_layout == "packed":
                return unpack_swiglu_linear_weight(self.weight, self.block_n_half)
            return self.weight.t().contiguous()

    def _packed_weight_for_forward(self) -> torch.Tensor:
        cache = self._packed_weight_cache
        version = self.weight._version
        if (
            tuple(cache.shape) != tuple(self.weight.shape)
            or cache.device != self.weight.device
            or cache.dtype != self.weight.dtype
        ):
            self._packed_weight_cache = torch.empty_like(
                self.weight, requires_grad=False
            )
            cache = self._packed_weight_cache
            self._packed_weight_cache_version = -1
        if self._packed_weight_cache_version != version:
            pack_swiglu_weight_chunked_into_triton(
                self.weight.detach(), cache, self.block_n_half
            )
            self._packed_weight_cache_version = version
        return cache

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight = self.linear_weight()
        if not keep_vars:
            weight = weight.detach()
        destination[prefix + "weight"] = weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "weight"
        original = state_dict.get(key)
        if original is not None:
            normal_shape = (self.hidden_features * 2, self.in_features)
            internal_shape = (self.in_features, self.hidden_features * 2)
            if tuple(original.shape) == normal_shape:
                if self.parameter_layout == "packed":
                    state_dict[key] = pack_swiglu_linear_weight(
                        original.detach().contiguous(), self.block_n_half
                    )
                else:
                    state_dict[key] = original.detach().t().contiguous()
            elif tuple(original.shape) != internal_shape:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(original.shape)} from checkpoint, expected "
                    f"{normal_shape} linear or {internal_shape} internal normal"
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if original is not None:
            state_dict[key] = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        if self.parameter_layout == "packed":
            if self.backward_mode == "save_preact":
                out = fused_swiglu_wide_packed_save_autograd(x, self.weight)
            elif self.backward_mode == "save_factors_normal":
                out = fused_swiglu_wide_packed_save_factors_normal_autograd(
                    x, self.weight
                )
            elif self.backward_mode == "cublas_save_factors":
                out = fused_swiglu_wide_packed_cublas_save_factors_autograd(
                    x, self.weight
                )
            elif self.backward_mode == "save_factors_cute_bwd":
                out = fused_swiglu_wide_packed_save_factors_cute_bwd_autograd(
                    x, self.weight
                )
            else:
                out = fused_swiglu_wide_packed_save_factors_autograd(x, self.weight)
        else:
            if self.backward_mode == "onload_save_factors":
                out = fused_swiglu_normal_weight_onload_save_factors_autograd(
                    x, self.weight
                )
            else:
                packed_weight = self._packed_weight_for_forward()
                if self.backward_mode == "save_preact":
                    out = fused_swiglu_normal_weight_save_preact_autograd(
                        x, self.weight, packed_weight
                    )
                elif self.backward_mode == "save_factors_normal":
                    out = fused_swiglu_normal_weight_save_factors_normal_autograd(
                        x, self.weight, packed_weight
                    )
                else:
                    out = fused_swiglu_normal_weight_save_factors_autograd(
                        x, self.weight, packed_weight
                    )
        return out.view(*x_shape[:-1], self.hidden_features)


def _bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _samples_text(samples: list[float]) -> str:
    return ",".join(f"{x:.3f}" for x in samples)


def _bench_pack_alloc(weight: torch.Tensor, iters: int, warmup: int) -> float:
    return _bench(lambda: pack_swiglu_weight_chunked(weight), iters, warmup)


def _bench_pack_into(weight: torch.Tensor, iters: int, warmup: int) -> float:
    out = torch.empty_like(weight)
    return _bench(lambda: pack_swiglu_weight_chunked_into(weight, out), iters, warmup)


def _run_shape(
    name: str, m: int, k: int, n_half: int, iters: int, warmup: int
) -> None:
    from fused_swiglu_v2 import fused_swiglu_v2

    dtype = torch.bfloat16
    print(f"\n## {name}: M={m} K={k} N_HALF={n_half} 2N={2 * n_half}", flush=True)
    x = torch.empty((m, k), device="cuda", dtype=dtype).normal_(0, 0.1)
    weight = torch.empty((k, 2 * n_half), device="cuda", dtype=dtype).normal_(
        0, 1.0 / (k**0.5)
    )
    packed = pack_swiglu_weight_chunked(weight)

    projection = x @ weight
    ref = swiglu(projection)
    actual = fused_swiglu_wide_packed(x, packed, use_tile_id_c=USE_TILE_ID_C)
    torch.cuda.synchronize()
    max_err = (ref.float() - actual.float()).abs().max().item()
    ref_mag = ref.float().abs().max().item()
    print(f"numerics wide-packed max_err={max_err:.3e} rel={max_err / ref_mag:.2%}")

    gemm_samples = [_bench(lambda: x @ weight, iters, warmup) for _ in range(3)]
    activation_samples = [_bench(lambda: swiglu(projection), iters, warmup) for _ in range(3)]
    base_samples = [
        _bench(lambda: swiglu(x @ weight), iters, warmup) for _ in range(3)
    ]
    v2_samples = [
        _bench(lambda: fused_swiglu_v2(x, weight), iters, warmup) for _ in range(2)
    ]
    wide_no_tile_id_c_samples = [
        _bench(
            lambda: fused_swiglu_wide_packed(x, packed, use_tile_id_c=False),
            iters,
            warmup,
        )
        for _ in range(3)
    ]
    wide_tile_id_c_samples = [
        _bench(
            lambda: fused_swiglu_wide_packed(x, packed, use_tile_id_c=True),
            iters,
            warmup,
        )
        for _ in range(3)
    ]
    pack_alloc = _bench_pack_alloc(weight, max(10, iters // 2), max(3, warmup // 4))
    pack_into = _bench_pack_into(weight, max(10, iters // 2), max(3, warmup // 4))

    gemm = min(gemm_samples)
    activation = min(activation_samples)
    base = min(base_samples)
    v2 = min(v2_samples)
    wide_no_tile_id_c = min(wide_no_tile_id_c_samples)
    wide_tile_id_c = min(wide_tile_id_c_samples)
    wide = wide_tile_id_c if USE_TILE_ID_C else wide_no_tile_id_c
    print(
        f"gemm-only cuBLAS      : {gemm:.3f} ms "
        f"samples={_samples_text(gemm_samples)}"
    )
    print(
        f"activation-only      : {activation:.3f} ms "
        f"samples={_samples_text(activation_samples)}"
    )
    print(
        f"baseline cuBLAS+swiglu : {base:.3f} ms "
        f"samples={_samples_text(base_samples)}"
    )
    print(
        f"removable window     : {base - gemm:.3f} ms "
        f"baseline/gemm={base / gemm:.3f}x"
    )
    print(f"v2 dual-acc singlepass: {v2:.3f} ms ratio={v2 / base:.3f}x")
    print(
        f"wide packed no tile_c : {wide_no_tile_id_c:.3f} ms "
        f"vs_base={wide_no_tile_id_c / base:.3f}x "
        f"vs_gemm={wide_no_tile_id_c / gemm:.3f}x "
        f"samples={_samples_text(wide_no_tile_id_c_samples)}"
    )
    print(
        f"wide packed tile_c    : {wide_tile_id_c:.3f} ms "
        f"vs_base={wide_tile_id_c / base:.3f}x "
        f"vs_gemm={wide_tile_id_c / gemm:.3f}x "
        f"samples={_samples_text(wide_tile_id_c_samples)}"
    )
    print(
        f"wide packed incl pack : {wide + pack_into:.3f} ms "
        f"vs_base={(wide + pack_into) / base:.3f}x "
        f"pack_into={pack_into:.3f} ms pack_alloc={pack_alloc:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    setup_triton()
    _ensure_allocator()
    print(f"# GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"# torch={torch.__version__} triton={triton.__version__}")
    print(
        "# config: "
        f"BM={BLOCK_SIZE_M} BNH={BLOCK_SIZE_N_HALF} BK={BLOCK_SIZE_K} "
        f"GM={GROUP_SIZE_M} nw={NUM_WARPS} ns={NUM_STAGES} "
        f"ws={WARP_SPECIALIZE}"
    )

    _run_shape(
        "AVOCADO", m=8192, k=3584, n_half=12288, iters=args.iters, warmup=args.warmup
    )
    _run_shape(
        "BLUEBERRY", m=32768, k=3072, n_half=12288, iters=args.iters, warmup=args.warmup
    )


if __name__ == "__main__":
    main()
