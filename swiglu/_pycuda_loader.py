"""Lazy PyCUDA module loader for CUDA kernels.

Compiles .cu files with nvcc on first use and caches the resulting .cubin
next to the source file.  Shares PyTorch's primary CUDA context so tensors
and kernels can freely exchange device pointers.
"""

import atexit
import os
import subprocess
import threading

import torch
import pycuda.driver as drv

NVCC  = "/usr/local/cuda/bin/nvcc"
PTXAS = "/usr/local/cuda-12.8/bin/ptxas"


def _detect_sm_arch() -> str:
    if not torch.cuda.is_available():
        return "sm_89"
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


SM_ARCH = _detect_sm_arch()

# Per-extension extra nvcc flags (e.g. register caps).
_EXTRA_FLAGS: dict[str, list[str]] = {}

_lock = threading.Lock()
_ctx: drv.Context | None = None
_modules: dict[str, drv.Module] = {}


def _pop_ctx() -> None:
    global _ctx
    if _ctx is not None:
        try:
            _ctx.pop()
        except Exception:
            pass
        _ctx = None


def _ensure_ctx() -> None:
    global _ctx
    if _ctx is not None:
        return
    torch.cuda.init()
    drv.init()
    _ctx = drv.Device(torch.cuda.current_device()).retain_primary_context()
    _ctx.push()
    atexit.register(_pop_ctx)


def _find_cu(ext_name: str) -> str:
    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    for sub in ("cuda_core", "tensor_core", "hopper"):
        path = os.path.join(gpu_dir, sub, f"{ext_name}.cu")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{ext_name}.cu not found under cuda_core/, tensor_core/, or hopper/")


def _cubin_path(cu_path: str) -> str:
    return cu_path[:-3] + f"_{SM_ARCH}.cubin"


def _compile(cu_path: str, cubin: str, extra_flags: list[str] | None = None) -> None:
    cmd = [NVCC, f"-arch={SM_ARCH}", "-O3", "--std=c++17", "--cubin"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd += [cu_path, "-o", cubin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stderr}")


def get_module(ext_name: str) -> drv.Module:
    with _lock:
        if ext_name in _modules:
            return _modules[ext_name]
        _ensure_ctx()
        cu_path = _find_cu(ext_name)
        cubin = _cubin_path(cu_path)
        if not os.path.exists(cubin) or os.path.getmtime(cu_path) > os.path.getmtime(cubin):
            print(f"[pycuda] compiling {os.path.basename(cu_path)} ...", end=" ", flush=True)
            _compile(cu_path, cubin, _EXTRA_FLAGS.get(ext_name))
            print("done")
        mod = drv.module_from_file(cubin)
        _modules[ext_name] = mod
        return mod


def get_kernel(ext_name: str, kernel_name: str) -> drv.Function:
    return get_module(ext_name).get_function(kernel_name)


def _compile_ptx(ptx_path: str, cubin: str) -> None:
    cmd = [PTXAS, f"-arch={SM_ARCH}", ptx_path, "-o", cubin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ptxas failed:\n{r.stderr}")


def get_module_ptx(ptx_path: str) -> drv.Module:
    """Compile a .ptx file to a cubin (via ptxas) and load it.

    The cubin is cached next to the .ptx file as <name>_sm89.cubin.
    """
    ptx_path = os.path.abspath(ptx_path)
    cubin = ptx_path[:-4] + f"_{SM_ARCH}.cubin"
    with _lock:
        if cubin in _modules:
            return _modules[cubin]
        _ensure_ctx()
        if not os.path.exists(cubin) or os.path.getmtime(ptx_path) > os.path.getmtime(cubin):
            print(f"[pycuda] compiling {os.path.basename(ptx_path)} ...", end=" ", flush=True)
            _compile_ptx(ptx_path, cubin)
            print("done")
        mod = drv.module_from_file(cubin)
        _modules[cubin] = mod
        return mod


def get_module_jit(cu_path: str, cubin_path: str, extra_flags: list[str]) -> drv.Module:
    """Compile and load a module with an explicit cubin path and extra flags.

    Used by stage 7 to JIT-compile per-(M,N,K) cubins from a shared template.
    The cubin_path acts as the cache key.
    """
    with _lock:
        if cubin_path in _modules:
            return _modules[cubin_path]
        _ensure_ctx()
        if not os.path.exists(cubin_path) or os.path.getmtime(cu_path) > os.path.getmtime(cubin_path):
            print(f"[pycuda] compiling {os.path.basename(cubin_path)} ...", end=" ", flush=True)
            _compile(cu_path, cubin_path, extra_flags)
            print("done")
        mod = drv.module_from_file(cubin_path)
        _modules[cubin_path] = mod
        return mod


def launch_matmul_raw(mod: drv.Module, kernel_name: str, A, B,
                      block: tuple, grid: tuple, smem_bytes: int = 0):
    """Launch a kernel whose signature is (const float* A, const float* B, float* C).

    Used by stage 7 kernels where M/N/K are baked in as compile-time constants.
    """
    import numpy as np
    M, _K = A.shape
    _K2, N = B.shape
    C = torch.zeros((M, N), device="cuda", dtype=A.dtype)
    fn = mod.get_function(kernel_name)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       block=block, grid=grid, shared=smem_bytes)
    return C


def launch_matmul(ext_name: str, kernel_name: str, A, B,
                  block: tuple, grid: tuple, out_dtype=None, smem_bytes: int = 0):
    """Launch a PyCUDA matmul kernel and return a new output tensor.

    The kernel signature must be:
        (const float* A, const float* B, float* C, int M, int K, int N)
    or the bf16 variant for stage 5.

    block and grid are (x, y, z) tuples as required by PyCUDA.
    out_dtype defaults to A.dtype.
    smem_bytes: if > 0, sets MAX_DYNAMIC_SHARED_SIZE_BYTES and passes shared=smem_bytes.
    """
    import numpy as np
    import torch
    M, K = A.shape
    _K2, N = B.shape
    dtype = out_dtype if out_dtype is not None else A.dtype
    C = torch.zeros((M, N), device="cuda", dtype=dtype)
    fn = get_kernel(ext_name, kernel_name)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C
