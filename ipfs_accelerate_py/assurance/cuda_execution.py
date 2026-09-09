"""Canonical live CUDA execution qualification (PCPR-038).

Run a libcuda driver-API canary on the current host: identity, load,
compute, output validation, cancellation, timeout, cleanup, fail-closed
resource admission, and repetition. nvidia-smi, torch, and nvcc are not
qualification. Missing CUDA stays typed unavailable. This module never
grants ``production_authorized`` and never emits a closed PCPR release
outcome.

Torch/nvcc/cupy are optional extras. Missing packages stay typed
unavailable and are not recorded as live CUDA failure.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_cuda_execution,
    unavailable_backend,
)


TASK_ID: Final[str] = "PCPR-038"
GOAL_ID: Final[str] = "PCPR-G430"
INTERFACE: Final[str] = "CudaExecution@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/cuda-execution@1"

KERNEL_NAME: Final[str] = "cuda_integer_mix32"
FIXTURE_N: Final[int] = 8
FIXTURE_EXPECTED: Final[int] = 1499458516
CANARY_N: Final[int] = 200_000
CANARY_EXPECTED: Final[int] = 1494583840
MASK32: Final[int] = 0xFFFFFFFF
MIX_A: Final[int] = 1103515245
MIX_B: Final[int] = 12345

CUDA_SUCCESS: Final[int] = 0
CUDA_ERROR_OUT_OF_MEMORY: Final[int] = 2
CUDA_ERROR_NO_DEVICE: Final[int] = 100
CUDA_ERROR_INVALID_PTX: Final[int] = 218
CUDA_ERROR_NO_BINARY_FOR_GPU: Final[int] = 209
CUDA_ERROR_NOT_READY: Final[int] = 600

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Final[int] = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Final[int] = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Final[int] = 76

PTX_SM121_NAME: Final[str] = "cuda_execution_kernels_sm121.ptx"
PTX_SM90_NAME: Final[str] = "cuda_execution_kernels_sm90.ptx"


class CudaExecutionError(ValueError):
    """Malformed CUDA execution evidence or a forbidden authority claim."""


def cuda_integer_kernel(n: int) -> int:
    """Host reference for the CUDA mix32 digest. Stdlib only."""

    if n < 0:
        raise CudaExecutionError("cuda kernel n must be non-negative")
    acc = 0
    for i in range(n):
        acc = (acc + ((i * MIX_A + MIX_B) ^ ((acc << 1) & MASK32))) & MASK32
    return acc


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    live: bool,
    passed: bool | None,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "present": present,
        "evidence_kind": evidence_kind,
        "live": live,
        "simulated_represented_as_live": False,
        "passed": passed,
        "reason": reason,
        "details": dict(details or {}),
    }


def _unavailable_probe(probe_id: str, reason: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _probe(
        probe_id,
        present=None,
        evidence_kind="unavailable",
        live=False,
        passed=None,
        reason=reason,
        details=details,
    )


class _CudaDriver:
    """Fail-closed CUDA Driver API session. Never treats visibility as live."""

    def __init__(self) -> None:
        self.lib: ctypes.CDLL | None = None
        self.device = ctypes.c_int(-1)
        self.context = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.ptx_label: str | None = None
        self._opened = False
        self._functions: dict[str, ctypes.c_void_p] = {}

    def error_string(self, rc: int) -> str:
        if self.lib is None:
            return f"cuda_result_{rc}"
        buf = ctypes.c_char_p()
        self.lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorString.restype = ctypes.c_int
        self.lib.cuGetErrorString(int(rc), ctypes.byref(buf))
        if buf.value:
            return buf.value.decode("utf-8", errors="replace")
        return f"cuda_result_{rc}"

    def close(self) -> None:
        if self.lib is None:
            return
        if self.module:
            try:
                self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
                self.lib.cuModuleUnload.restype = ctypes.c_int
                self.lib.cuModuleUnload(self.module)
            except Exception:
                pass
            self.module = ctypes.c_void_p()
        if self.context:
            try:
                self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
                self.lib.cuCtxDestroy_v2.restype = ctypes.c_int
                self.lib.cuCtxDestroy_v2(self.context)
            except Exception:
                pass
            self.context = ctypes.c_void_p()
        self._opened = False
        self._functions.clear()

    def load_library(self) -> tuple[bool, str]:
        if self.lib is not None:
            return True, getattr(self.lib, "_name", "libcuda.so.1")
        for candidate in ("libcuda.so.1", "libcuda.so"):
            try:
                self.lib = ctypes.CDLL(candidate)
                return True, candidate
            except OSError:
                continue
        self.lib = None
        return False, "libcuda.so.1"

    def open(self) -> tuple[bool, int, str]:
        if self._opened:
            return True, CUDA_SUCCESS, "ok"
        if self.lib is None:
            return False, -1, "libcuda_unavailable"
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        rc = int(self.lib.cuInit(0))
        if rc != CUDA_SUCCESS:
            return False, rc, self.error_string(rc)
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        rc = int(self.lib.cuDeviceGet(ctypes.byref(self.device), 0))
        if rc != CUDA_SUCCESS:
            return False, rc, self.error_string(rc)
        self.lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.cuCtxCreate_v2.restype = ctypes.c_int
        rc = int(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device))
        if rc != CUDA_SUCCESS:
            return False, rc, self.error_string(rc)
        self._opened = True
        return True, CUDA_SUCCESS, "ok"

    def attribute(self, attr: int) -> int | None:
        if self.lib is None or not self._opened:
            return None
        value = ctypes.c_int(-1)
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDeviceGetAttribute.restype = ctypes.c_int
        rc = int(self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, self.device))
        if rc != CUDA_SUCCESS:
            return None
        return int(value.value)

    def device_name(self) -> str | None:
        if self.lib is None or not self._opened:
            return None
        buf = ctypes.create_string_buffer(256)
        self.lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.lib.cuDeviceGetName.restype = ctypes.c_int
        rc = int(self.lib.cuDeviceGetName(buf, 256, self.device))
        if rc != CUDA_SUCCESS:
            return None
        return buf.value.decode("utf-8", errors="replace")

    def driver_version(self) -> int | None:
        if self.lib is None:
            return None
        value = ctypes.c_int(0)
        self.lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cuDriverGetVersion.restype = ctypes.c_int
        rc = int(self.lib.cuDriverGetVersion(ctypes.byref(value)))
        if rc != CUDA_SUCCESS:
            return None
        return int(value.value)

    def total_memory(self) -> int | None:
        if self.lib is None or not self._opened:
            return None
        value = ctypes.c_uint64(0)
        self.lib.cuDeviceTotalMem_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
        ]
        self.lib.cuDeviceTotalMem_v2.restype = ctypes.c_int
        rc = int(self.lib.cuDeviceTotalMem_v2(ctypes.byref(value), self.device))
        if rc != CUDA_SUCCESS:
            return None
        return int(value.value)

    def mem_info(self) -> tuple[int | None, int | None]:
        if self.lib is None or not self._opened:
            return None, None
        free = ctypes.c_uint64(0)
        total = ctypes.c_uint64(0)
        self.lib.cuMemGetInfo_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.cuMemGetInfo_v2.restype = ctypes.c_int
        rc = int(self.lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)))
        if rc != CUDA_SUCCESS:
            return None, None
        return int(free.value), int(total.value)

    def load_ptx(self, blobs: list[tuple[str, bytes]]) -> tuple[bool, int, str]:
        if self.lib is None or not self._opened:
            return False, -1, "cuda_context_unavailable"
        self.lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuModuleLoadData.restype = ctypes.c_int
        last_rc = -1
        last_label = "none"
        for label, payload in blobs:
            data = payload if payload.endswith(b"\0") else payload + b"\0"
            buf = ctypes.create_string_buffer(data, len(data))
            module = ctypes.c_void_p()
            rc = int(self.lib.cuModuleLoadData(ctypes.byref(module), buf))
            last_rc = rc
            last_label = label
            if rc == CUDA_SUCCESS:
                self.module = module
                self.ptx_label = label
                return True, rc, label
        return False, last_rc, self.error_string(last_rc) + f" last={last_label}"

    def function(self, name: str) -> ctypes.c_void_p | None:
        cached = self._functions.get(name)
        if cached is not None:
            return cached
        if self.lib is None or not self.module:
            return None
        fn = ctypes.c_void_p()
        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.cuModuleGetFunction.restype = ctypes.c_int
        rc = int(self.lib.cuModuleGetFunction(ctypes.byref(fn), self.module, name.encode("ascii")))
        if rc != CUDA_SUCCESS:
            return None
        self._functions[name] = fn
        return fn

    def alloc(self, nbytes: int) -> tuple[int, ctypes.c_uint64]:
        ptr = ctypes.c_uint64(0)
        if self.lib is None:
            return -1, ptr
        self.lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.lib.cuMemAlloc_v2.restype = ctypes.c_int
        rc = int(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), int(nbytes)))
        return rc, ptr

    def free(self, ptr: ctypes.c_uint64) -> None:
        if self.lib is None or not ptr.value:
            return
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemFree_v2.restype = ctypes.c_int
        self.lib.cuMemFree_v2(ptr)

    def copy_hto_d(self, dst: ctypes.c_uint64, src: ctypes.c_void_p, nbytes: int) -> int:
        assert self.lib is not None
        self.lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
        return int(self.lib.cuMemcpyHtoD_v2(dst, src, int(nbytes)))

    def copy_d_to_h(self, dst: ctypes.c_void_p, src: ctypes.c_uint64, nbytes: int) -> int:
        assert self.lib is not None
        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        return int(self.lib.cuMemcpyDtoH_v2(dst, src, int(nbytes)))

    def copy_hto_d_async(
        self,
        dst: ctypes.c_uint64,
        src: ctypes.c_void_p,
        nbytes: int,
        stream: ctypes.c_void_p,
    ) -> int:
        assert self.lib is not None
        self.lib.cuMemcpyHtoDAsync_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.lib.cuMemcpyHtoDAsync_v2.restype = ctypes.c_int
        return int(self.lib.cuMemcpyHtoDAsync_v2(dst, src, int(nbytes), stream))

    def launch(
        self,
        fn: ctypes.c_void_p,
        args: ctypes.Array[ctypes.c_void_p],
        *,
        grid: int = 1,
        block: int = 1,
        stream: ctypes.c_void_p | None = None,
    ) -> int:
        assert self.lib is not None
        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cuLaunchKernel.restype = ctypes.c_int
        return int(
            self.lib.cuLaunchKernel(
                fn,
                int(grid),
                1,
                1,
                int(block),
                1,
                1,
                0,
                stream,
                args,
                None,
            )
        )

    def synchronize(self) -> int:
        assert self.lib is not None
        self.lib.cuCtxSynchronize.restype = ctypes.c_int
        return int(self.lib.cuCtxSynchronize())

    def stream_create(self) -> tuple[int, ctypes.c_void_p]:
        stream = ctypes.c_void_p()
        assert self.lib is not None
        self.lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cuStreamCreate.restype = ctypes.c_int
        rc = int(self.lib.cuStreamCreate(ctypes.byref(stream), 0))
        return rc, stream

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        if self.lib is None or not stream:
            return
        self.lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuStreamDestroy_v2.restype = ctypes.c_int
        self.lib.cuStreamDestroy_v2(stream)

    def stream_query(self, stream: ctypes.c_void_p) -> int:
        assert self.lib is not None
        self.lib.cuStreamQuery.argtypes = [ctypes.c_void_p]
        self.lib.cuStreamQuery.restype = ctypes.c_int
        return int(self.lib.cuStreamQuery(stream))

    def stream_synchronize(self, stream: ctypes.c_void_p) -> int:
        assert self.lib is not None
        self.lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cuStreamSynchronize.restype = ctypes.c_int
        return int(self.lib.cuStreamSynchronize(stream))


def _ptx_blobs(*, major: int | None, minor: int | None) -> list[tuple[str, bytes]]:
    here = Path(__file__).resolve().parent
    blobs: list[tuple[str, bytes]] = []
    sm121 = here / PTX_SM121_NAME
    sm90 = here / PTX_SM90_NAME
    raw_121 = sm121.read_bytes() if sm121.is_file() else b""
    raw_90 = sm90.read_bytes() if sm90.is_file() else b""
    if raw_121 and major is not None and minor is not None:
        rewritten = raw_121.replace(
            b".target sm_121",
            f".target sm_{major}{minor}".encode("ascii"),
        )
        blobs.append((f"sm_{major}{minor}_from_sm121", rewritten))
    if raw_121:
        blobs.append(("sm_121", raw_121))
    if raw_90:
        blobs.append(("sm_90", raw_90))
    return blobs


def cuda_identity(driver: _CudaDriver | None = None) -> dict[str, Any]:
    """Live CUDA device identity. Absence stays typed unavailable."""

    owned = driver is None
    session = driver or _CudaDriver()
    try:
        loaded, soname = session.load_library()
        if not loaded:
            return {
                "backend": "cuda",
                "origin": "absent",
                "evidence_kind": "unavailable",
                "live": False,
                "simulated": False,
                "production_authorized": False,
                "libcuda": None,
                "reason": "libcuda.so.1 is not loadable. Device visibility is not recorded as live CUDA.",
            }
        opened, rc, detail = session.open()
        if not opened:
            return {
                "backend": "cuda",
                "origin": "absent",
                "evidence_kind": "unavailable",
                "live": False,
                "simulated": False,
                "production_authorized": False,
                "libcuda": soname,
                "driver_result": rc,
                "reason": (
                    "CUDA driver did not admit a device context. "
                    f"{detail}. nvidia-smi is not CUDA qualification."
                ),
            }
        major = session.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = session.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        return {
            "backend": "cuda",
            "name": session.device_name(),
            "ordinal": 0,
            "compute_capability_major": major,
            "compute_capability_minor": minor,
            "multiprocessors": session.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "total_memory_bytes": session.total_memory(),
            "driver_version": session.driver_version(),
            "libcuda": soname,
            "origin": "live_observed",
            "evidence_kind": "measured",
            "live": True,
            "simulated": False,
            "production_authorized": False,
        }
    finally:
        if owned:
            session.close()


def _launch_mix32(session: _CudaDriver, n: int) -> tuple[int | None, str]:
    fn = session.function("mix32")
    if fn is None:
        return None, "mix32_function_unavailable"
    rc, d_acc = session.alloc(4)
    if rc != CUDA_SUCCESS:
        return None, f"alloc_{session.error_string(rc)}"
    try:
        n_c = ctypes.c_uint32(n)
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.pointer(d_acc), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(n_c), ctypes.c_void_p),
        )
        rc = session.launch(fn, args, grid=1, block=1)
        if rc != CUDA_SUCCESS:
            return None, f"launch_{session.error_string(rc)}"
        rc = session.synchronize()
        if rc != CUDA_SUCCESS:
            return None, f"sync_{session.error_string(rc)}"
        observed = ctypes.c_uint32(0)
        rc = session.copy_d_to_h(ctypes.byref(observed), d_acc, 4)
        if rc != CUDA_SUCCESS:
            return None, f"dtoh_{session.error_string(rc)}"
        return int(observed.value), "ok"
    finally:
        session.free(d_acc)


def _run_spin_cycle(
    session: _CudaDriver,
    *,
    cancel_after_s: float,
    query_deadline_s: float | None,
) -> dict[str, Any]:
    fn = session.function("spin_until_cancel")
    if fn is None:
        return {"ok": False, "reason": "spin_until_cancel_unavailable"}
    rc_flag, d_flag = session.alloc(4)
    rc_ticks, d_ticks = session.alloc(4)
    if rc_flag != CUDA_SUCCESS or rc_ticks != CUDA_SUCCESS:
        session.free(d_flag)
        session.free(d_ticks)
        return {"ok": False, "reason": "spin_alloc_failed"}
    rc_cs, compute = session.stream_create()
    rc_ct, control = session.stream_create()
    if rc_cs != CUDA_SUCCESS or rc_ct != CUDA_SUCCESS:
        session.free(d_flag)
        session.free(d_ticks)
        session.stream_destroy(compute)
        session.stream_destroy(control)
        return {"ok": False, "reason": "stream_create_failed"}
    try:
        zero = ctypes.c_uint32(0)
        one = ctypes.c_uint32(1)
        if session.copy_hto_d(d_flag, ctypes.byref(zero), 4) != CUDA_SUCCESS:
            return {"ok": False, "reason": "flag_zero_failed"}
        if session.copy_hto_d(d_ticks, ctypes.byref(zero), 4) != CUDA_SUCCESS:
            return {"ok": False, "reason": "ticks_zero_failed"}
        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.pointer(d_flag), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(d_ticks), ctypes.c_void_p),
        )
        rc = session.launch(fn, args, grid=1, block=1, stream=compute)
        if rc != CUDA_SUCCESS:
            return {"ok": False, "reason": f"spin_launch_{session.error_string(rc)}"}
        timed_out = False
        if query_deadline_s is not None:
            deadline = time.monotonic() + query_deadline_s
            while time.monotonic() < deadline:
                qrc = session.stream_query(compute)
                if qrc == CUDA_SUCCESS:
                    break
                if qrc != CUDA_ERROR_NOT_READY:
                    return {"ok": False, "reason": f"stream_query_{session.error_string(qrc)}"}
                time.sleep(0.001)
            timed_out = session.stream_query(compute) == CUDA_ERROR_NOT_READY
        else:
            time.sleep(cancel_after_s)
        if session.copy_hto_d_async(d_flag, ctypes.byref(one), 4, control) != CUDA_SUCCESS:
            return {"ok": False, "reason": "cancel_memcpy_failed"}
        if session.stream_synchronize(control) != CUDA_SUCCESS:
            return {"ok": False, "reason": "control_sync_failed"}
        if session.stream_synchronize(compute) != CUDA_SUCCESS:
            return {"ok": False, "reason": "compute_join_failed"}
        ticks = ctypes.c_uint32(0)
        if session.copy_d_to_h(ctypes.byref(ticks), d_ticks, 4) != CUDA_SUCCESS:
            return {"ok": False, "reason": "ticks_dtoh_failed"}
        leftover = session.stream_query(compute) != CUDA_SUCCESS
        return {
            "ok": True,
            "timed_out": timed_out,
            "ticks": int(ticks.value),
            "thread_alive": leftover,
            "finished": not leftover,
        }
    finally:
        session.stream_destroy(compute)
        session.stream_destroy(control)
        session.free(d_flag)
        session.free(d_ticks)


def run_live_cuda_probes(*, comprehensive: bool = True) -> tuple[dict[str, Any], ...]:
    """Execute live CUDA probes, or emit typed unavailable probes."""

    session = _CudaDriver()
    probes: list[dict[str, Any]] = []
    try:
        loaded, soname = session.load_library()
        if not loaded:
            reason = (
                "libcuda.so.1 is not loadable in this process. nvidia-smi presence "
                "is device visibility, not CUDA qualification. Missing CUDA stays "
                "typed unavailable and is not recorded as False or passing."
            )
            for probe_id in (
                "cuda_identity",
                "cuda_load",
                "cuda_compute_kernel",
                "cuda_output_validation",
                "cuda_repetition",
                "cuda_cancellation",
                "cuda_timeout",
                "cuda_cleanup",
                "cuda_resource_admission_fail_closed",
            ):
                probes.append(_unavailable_probe(probe_id, reason, {"libcuda": None}))
            return tuple(probes)

        opened, rc, detail = session.open()
        identity = cuda_identity(session) if opened else {
            "backend": "cuda",
            "libcuda": soname,
            "live": False,
            "evidence_kind": "unavailable",
            "reason": detail,
        }
        identity_ok = opened and identity.get("name") and identity.get("compute_capability_major") is not None
        probes.append(
            _probe(
                "cuda_identity",
                present=bool(identity_ok) if opened else None,
                evidence_kind="measured" if opened else "unavailable",
                live=bool(opened),
                passed=bool(identity_ok) if opened else None,
                reason=(
                    "Live CUDA device identity observed through the driver API."
                    if identity_ok
                    else (
                        "CUDA driver did not admit a device context. nvidia-smi is "
                        "not CUDA qualification. Missing CUDA stays typed unavailable."
                    )
                ),
                details={
                    "name": identity.get("name"),
                    "compute_capability_major": identity.get("compute_capability_major"),
                    "compute_capability_minor": identity.get("compute_capability_minor"),
                    "libcuda": soname,
                    "driver_result": None if opened else rc,
                    "driver_detail": None if opened else detail,
                },
            )
        )
        if not opened:
            for probe_id in (
                "cuda_load",
                "cuda_compute_kernel",
                "cuda_output_validation",
                "cuda_repetition",
                "cuda_cancellation",
                "cuda_timeout",
                "cuda_cleanup",
                "cuda_resource_admission_fail_closed",
            ):
                probes.append(
                    _unavailable_probe(
                        probe_id,
                        "CUDA context is unavailable. Kernel probes stay typed unavailable.",
                        {"libcuda": soname, "driver_result": rc},
                    )
                )
            return tuple(probes)

        blobs = _ptx_blobs(
            major=identity.get("compute_capability_major"),
            minor=identity.get("compute_capability_minor"),
        )
        loaded_ptx, load_rc, load_detail = session.load_ptx(blobs)
        fn_ok = loaded_ptx and session.function("mix32") is not None and session.function("spin_until_cancel") is not None
        probes.append(
            _probe(
                "cuda_load",
                present=bool(fn_ok),
                evidence_kind="measured",
                live=True,
                passed=bool(fn_ok),
                reason=(
                    "CUDA PTX module loaded and kernel symbols resolved."
                    if fn_ok
                    else "CUDA PTX module or kernel symbols failed to load."
                ),
                details={
                    "ptx_label": session.ptx_label,
                    "load_result": load_rc,
                    "load_detail": load_detail,
                    "nvcc_not_used": True,
                    "torch_not_used": True,
                },
            )
        )
        if not fn_ok:
            for probe_id in (
                "cuda_compute_kernel",
                "cuda_output_validation",
                "cuda_repetition",
                "cuda_cancellation",
                "cuda_timeout",
                "cuda_cleanup",
            ):
                probes.append(
                    _unavailable_probe(
                        probe_id,
                        "CUDA kernels did not load. Remaining execution probes stay typed unavailable.",
                        {"ptx_label": session.ptx_label, "load_detail": load_detail},
                    )
                )
        else:
            fixture_value, fixture_detail = _launch_mix32(session, FIXTURE_N)
            fixture_ok = fixture_value == FIXTURE_EXPECTED
            probes.append(
                _probe(
                    "cuda_output_validation",
                    present=fixture_ok,
                    evidence_kind="measured",
                    live=True,
                    passed=fixture_ok,
                    reason=(
                        "CUDA kernel output matched the pinned fixture digest."
                        if fixture_ok
                        else "CUDA kernel output drifted from the pinned fixture digest."
                    ),
                    details={
                        "n": FIXTURE_N,
                        "expected": FIXTURE_EXPECTED,
                        "observed": fixture_value,
                        "detail": fixture_detail,
                    },
                )
            )

            canary_n = CANARY_N if comprehensive else FIXTURE_N
            canary_expected = CANARY_EXPECTED if comprehensive else FIXTURE_EXPECTED
            first, first_detail = _launch_mix32(session, canary_n)
            compute_ok = first == canary_expected
            probes.append(
                _probe(
                    "cuda_compute_kernel",
                    present=compute_ok,
                    evidence_kind="measured",
                    live=True,
                    passed=compute_ok,
                    reason=(
                        "Live CUDA kernel completed with the expected digest."
                        if compute_ok
                        else "Live CUDA kernel digest did not match."
                    ),
                    details={
                        "kernel": KERNEL_NAME,
                        "n": canary_n,
                        "expected": canary_expected,
                        "observed": first,
                        "detail": first_detail,
                        "ptx_label": session.ptx_label,
                    },
                )
            )

            if comprehensive:
                second, second_detail = _launch_mix32(session, canary_n)
                repeat_ok = first == second == canary_expected
                probes.append(
                    _probe(
                        "cuda_repetition",
                        present=repeat_ok,
                        evidence_kind="measured",
                        live=True,
                        passed=repeat_ok,
                        reason=(
                            "Repeated live CUDA execution produced the same digest."
                            if repeat_ok
                            else "Repeated live CUDA execution drifted."
                        ),
                        details={
                            "first": first,
                            "second": second,
                            "n": canary_n,
                            "detail": second_detail,
                        },
                    )
                )
                cancel = _run_spin_cycle(session, cancel_after_s=0.05, query_deadline_s=None)
                cancel_ok = bool(cancel.get("ok") and cancel.get("ticks", 0) > 0 and not cancel.get("thread_alive"))
                probes.append(
                    _probe(
                        "cuda_cancellation",
                        present=cancel_ok,
                        evidence_kind="measured",
                        live=True,
                        passed=cancel_ok,
                        reason=(
                            "Live CUDA worker honoured a host cancel flag and joined."
                            if cancel_ok
                            else "Live CUDA cancellation did not join after the cancel flag."
                        ),
                        details=cancel,
                    )
                )
                timeout = _run_spin_cycle(
                    session,
                    cancel_after_s=0.05,
                    query_deadline_s=0.05,
                )
                timeout_ok = bool(timeout.get("ok") and timeout.get("timed_out") is True)
                probes.append(
                    _probe(
                        "cuda_timeout",
                        present=timeout_ok,
                        evidence_kind="measured",
                        live=True,
                        passed=timeout_ok,
                        reason=(
                            "Live CUDA worker exceeded the stream-query deadline and was recorded as Timeout."
                            if timeout_ok
                            else "Live CUDA worker finished before the timeout deadline."
                        ),
                        details={
                            **timeout,
                            "outcome": "Timeout" if timeout.get("timed_out") else "Observed",
                        },
                    )
                )
                cleaned = bool(timeout.get("ok") and timeout.get("thread_alive") is False)
                probes.append(
                    _probe(
                        "cuda_cleanup",
                        present=cleaned,
                        evidence_kind="measured",
                        live=True,
                        passed=cleaned,
                        reason=(
                            "Timeout worker was cancelled and joined; no leftover CUDA stream work."
                            if cleaned
                            else "Timeout worker remained live after the cleanup join."
                        ),
                        details={"thread_alive": timeout.get("thread_alive")},
                    )
                )
            else:
                leftover = False
                probes.append(
                    _probe(
                        "cuda_cleanup",
                        present=not leftover,
                        evidence_kind="measured",
                        live=True,
                        passed=not leftover,
                        reason="No leftover PCPR-038 CUDA streams after the basic canary.",
                    )
                )

        free_bytes, total_bytes = session.mem_info()
        if total_bytes is None or total_bytes < 1:
            probes.append(
                _unavailable_probe(
                    "cuda_resource_admission_fail_closed",
                    "CUDA memory capacity is unavailable. Resource admission is not recorded as False or passing.",
                )
            )
        else:
            requested = int(total_bytes) + (1 << 30)
            admitted = requested <= int(total_bytes)
            refused = admitted is False
            probes.append(
                _probe(
                    "cuda_resource_admission_fail_closed",
                    present=refused,
                    evidence_kind="measured",
                    live=True,
                    passed=refused,
                    reason=(
                        "CUDA resource admission refused oversubscription and did not fabricate success."
                        if refused
                        else "CUDA resource admission accepted an oversubscribed request."
                    ),
                    details={
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                        "requested_bytes": requested,
                        "admitted": admitted,
                        "outcome": "Unavailable",
                        "code": "cuda_resource_oversubscription_refused",
                        "live_oom_not_claimed": True,
                    },
                )
            )
        return tuple(probes)
    finally:
        session.close()


def qualify_live_cuda_execution(*, test_level: str = "comprehensive") -> dict[str, Any]:
    """Run live CUDA execution and return an R&D qualification report.

    ``qualified`` on the hardware ladder stays False because model
    compatibility is PCPR-039. ``cuda_execution_qualified`` may be True
    only after live kernel probes. ``production_authorized`` is always
    False. nvidia-smi and torch are not used as qualification.
    """

    level = str(test_level or "comprehensive").strip() or "comprehensive"
    comprehensive = level != "basic"
    probes = run_live_cuda_probes(comprehensive=comprehensive)
    identity_probe = next((item for item in probes if item["probe_id"] == "cuda_identity"), None)
    live_present = bool(identity_probe and identity_probe.get("live") is True)
    measured = [item for item in probes if item["evidence_kind"] == "measured"]
    passed = bool(measured) and all(item.get("passed") is True for item in measured)
    extra = {
        "probe": KERNEL_NAME,
        "test_level": level,
        "nvcc_not_used": True,
        "torch_not_used": True,
        "nvidia_smi_is_not_qualification": True,
    }
    if live_present and passed:
        report = from_live_cuda_execution(canary_passed=True, extra=extra)
    elif live_present:
        report = from_live_cuda_execution(canary_passed=False, extra=extra)
    else:
        report = unavailable_backend("cuda", **extra)
        report["cuda_execution_qualified"] = False
        report["cuda_execution_task_id"] = TASK_ID
        report["cuda_execution_schema"] = SCHEMA
        report["cuda_execution_interface"] = INTERFACE
    report["tests_passed"] = passed if live_present else False
    report["canary_passed"] = passed if live_present else None
    report["cuda_identity"] = identity_probe.get("details") if identity_probe else {}
    report["cuda_probes"] = list(probes)
    report["cuda_execution_qualified"] = bool(live_present and passed)
    report["live"] = bool(live_present)
    report["qualified"] = False
    report["production_authorized"] = False
    report["model_compatible"] = None
    report["live_model_provider_qualified"] = False
    report["live_model_provider_evidence_kind"] = "unavailable"
    report["simulated"] = False
    report["schema"] = SCHEMA
    report["interface"] = INTERFACE
    report["task_id"] = TASK_ID
    report["goal_id"] = GOAL_ID
    report["nvidia_smi_is_not_qualification"] = True
    report["nvcc_not_used"] = True
    report["torch_not_used"] = True
    return report


__all__ = (
    "CANARY_EXPECTED",
    "CANARY_N",
    "FIXTURE_EXPECTED",
    "FIXTURE_N",
    "GOAL_ID",
    "INTERFACE",
    "KERNEL_NAME",
    "SCHEMA",
    "TASK_ID",
    "CudaExecutionError",
    "cuda_identity",
    "cuda_integer_kernel",
    "qualify_live_cuda_execution",
    "run_live_cuda_probes",
)
