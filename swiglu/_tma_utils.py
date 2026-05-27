"""ctypes wrapper around `cuTensorMapEncodeTiled` for TMA setup.

pycuda 13.2 has no TMA API, so we call the CUDA driver function directly.
`build_tma_2d` returns a 128-byte CUtensorMap struct as a numpy uint8 array
that can be passed straight to a pycuda kernel as a `__grid_constant__` param.

CUDA driver type constants (from cuda.h):
"""

import ctypes
import numpy as np


# CUtensorMapDataType — from cuda.h
TENSOR_DTYPE_UINT8       = 0
TENSOR_DTYPE_UINT16      = 1
TENSOR_DTYPE_UINT32      = 2
TENSOR_DTYPE_INT32       = 3
TENSOR_DTYPE_UINT64      = 4
TENSOR_DTYPE_INT64       = 5
TENSOR_DTYPE_FLOAT16     = 6
TENSOR_DTYPE_FLOAT32     = 7
TENSOR_DTYPE_FLOAT64     = 8
TENSOR_DTYPE_BFLOAT16    = 9
TENSOR_DTYPE_FLOAT32_FTZ = 10
TENSOR_DTYPE_TFLOAT32    = 11
TENSOR_DTYPE_TFLOAT32_FTZ = 12

# CUtensorMapInterleave
INTERLEAVE_NONE = 0
INTERLEAVE_16B  = 1
INTERLEAVE_32B  = 2

# CUtensorMapSwizzle
SWIZZLE_NONE = 0
SWIZZLE_32B  = 1
SWIZZLE_64B  = 2
SWIZZLE_128B = 3

# CUtensorMapL2promotion
L2_PROMOTION_NONE = 0
L2_PROMOTION_L2_64B  = 1
L2_PROMOTION_L2_128B = 2
L2_PROMOTION_L2_256B = 3

# CUtensorMapFloatOOBfill
OOB_FILL_NONE = 0
OOB_FILL_NAN_REQUEST_ZERO_FMA = 1

_libcuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
_cuTensorMapEncodeTiled = _libcuda.cuTensorMapEncodeTiled
_cuTensorMapEncodeTiled.restype = ctypes.c_int
_cuTensorMapEncodeTiled.argtypes = [
    ctypes.c_void_p,                            # CUtensorMap *tensorMap
    ctypes.c_int,                               # CUtensorMapDataType
    ctypes.c_uint32,                            # tensorRank
    ctypes.c_void_p,                            # globalAddress
    ctypes.POINTER(ctypes.c_uint64),            # globalDim
    ctypes.POINTER(ctypes.c_uint64),            # globalStrides (rank-1)
    ctypes.POINTER(ctypes.c_uint32),            # boxDim
    ctypes.POINTER(ctypes.c_uint32),            # elementStrides
    ctypes.c_int,                               # interleave
    ctypes.c_int,                               # swizzle
    ctypes.c_int,                               # l2 promotion
    ctypes.c_int,                               # oob fill
]


def build_tma_2d(
    gptr: int,
    global_height: int,
    global_width: int,
    box_height: int,
    box_width: int,
    swizzle: int = SWIZZLE_128B,
    dtype: int = TENSOR_DTYPE_BFLOAT16,
    elem_bytes: int = 2,
) -> np.ndarray:
    """Build a 2D `CUtensorMap` matching gau-nernst's `init_tmap_2d_simple`.

    Returns a numpy uint8 array of length 128 holding the opaque tensor map,
    suitable to pass to a pycuda kernel as a parameter.

    Args mirror cuTensorMapEncodeTiled's conventions:
      - global tensor is (global_height, global_width) row-major
      - box is (box_height, box_width)
      - global_strides[0] = global_width * elem_bytes (in bytes)
    """
    tmap = np.zeros(128, dtype=np.uint8)
    global_dim = (ctypes.c_uint64 * 2)(global_width, global_height)
    global_strides = (ctypes.c_uint64 * 1)(global_width * elem_bytes)
    box_dim = (ctypes.c_uint32 * 2)(box_width, box_height)
    elem_strides = (ctypes.c_uint32 * 2)(1, 1)

    err = _cuTensorMapEncodeTiled(
        tmap.ctypes.data,
        dtype,
        2,            # rank
        gptr,
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        INTERLEAVE_NONE,
        swizzle,
        L2_PROMOTION_NONE,
        OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    return tmap


def build_tma_3d_128B(
    gptr: int,
    global_height: int,
    global_width: int,
    box_height: int,
    box_width: int,
    dtype: int = TENSOR_DTYPE_BFLOAT16,
    elem_bytes: int = 2,
) -> np.ndarray:
    """Build a 3D `CUtensorMap` matching gau-nernst's `init_tmap_3d_128B`.

    Reshapes a row-major (global_height, global_width) tensor as
    [global_width/64, global_height, 64] (permuted), so one TMA bulk loads
    `box_width / 64` slabs of (box_height, 64) into SMEM with a single
    instruction — replacing the `box_width / 64` separate 2D bulks the
    caller would otherwise issue.

    Inner-most dim is fixed to 64 BF16 elements = 128 bytes, the natural
    swizzle slab.  Swizzle is always 128B.
    """
    assert box_width % 64 == 0, "box_width must be a multiple of 64"
    assert global_width % 64 == 0, "global_width must be a multiple of 64"
    tmap = np.zeros(128, dtype=np.uint8)
    # [inner=64, height, width/64]
    global_dim     = (ctypes.c_uint64 * 3)(64, global_height, global_width // 64)
    # strides for the non-innermost dims, in bytes:
    #   - height stride = global_width * elem_bytes  (one row down)
    #   - slab stride   = 64 * elem_bytes = 128       (one 64-wide slab right)
    global_strides = (ctypes.c_uint64 * 2)(global_width * elem_bytes, 64 * elem_bytes)
    box_dim        = (ctypes.c_uint32 * 3)(64, box_height, box_width // 64)
    elem_strides   = (ctypes.c_uint32 * 3)(1, 1, 1)

    err = _cuTensorMapEncodeTiled(
        tmap.ctypes.data,
        dtype,
        3,            # rank
        gptr,
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        INTERLEAVE_NONE,
        SWIZZLE_128B,
        L2_PROMOTION_NONE,
        OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled (3D) failed: {err}")
    return tmap
