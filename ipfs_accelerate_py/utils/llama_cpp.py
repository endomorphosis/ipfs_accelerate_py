"""llama.cpp dependency and server helpers.

This module intentionally owns local llama.cpp lifecycle details for
``ipfs_accelerate_py``.  Project-specific callers should use the router provider
instead of shelling out to llama.cpp directly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import shlex
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence


DEFAULT_LEANSTRAL_REPO_ID = "Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4"
DEFAULT_LEANSTRAL_FILENAME = "Leanstral-1.5-119B-A6B-NVFP4.gguf"
DEFAULT_LEANSTRAL_QUANTIZATION = "NVFP4"
DEFAULT_LEANSTRAL_MODEL_REF = f"{DEFAULT_LEANSTRAL_REPO_ID}:{DEFAULT_LEANSTRAL_QUANTIZATION}"
DEFAULT_LLAMA_CPP_HOST = "127.0.0.1"
DEFAULT_LLAMA_CPP_PORT = 8080
DEFAULT_LLAMA_CPP_CONTEXT_SIZE = 8096
DEFAULT_LLAMA_CPP_INSTALL_CMD = "curl -LsSf https://llama.app/install.sh | sh"
DEFAULT_LLAMA_CPP_SOURCE_REPO = "https://github.com/ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_SOURCE_REF = "master"
DEFAULT_LLAMA_CPP_INSTALL_METHOD = "source"


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + str(repo_id or "").strip().replace("/", "--")


def _truthy(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


def _coalesce_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _hf_cache_root() -> Path:
    configured = _coalesce_env(
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "IPFS_ACCELERATE_HF_HUB_CACHE",
    )
    if configured:
        return Path(configured).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home and str(hf_home).strip():
        return Path(str(hf_home).strip()).expanduser() / "hub"
    cache_root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_root).expanduser() / "huggingface" / "hub"


def _default_cache_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_root) / "ipfs_accelerate_py" / "llama_cpp"


def _managed_source_dir() -> Path:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_SOURCE_DIR",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_SOURCE_DIR",
    )
    if configured:
        return Path(configured).expanduser()
    return _default_cache_dir() / "source" / "llama.cpp"


def _managed_build_dir() -> Path:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_BUILD_DIR",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_BUILD_DIR",
    )
    if configured:
        return Path(configured).expanduser()
    return _default_cache_dir() / "build"


def _managed_bin_dir() -> Path:
    return _managed_build_dir() / "bin"


def _managed_server_executable() -> Path:
    return _managed_bin_dir() / "llama-server"


def _managed_cli_executable() -> Path:
    return _managed_bin_dir() / "llama-cli"


@dataclass(frozen=True)
class LlamaCppInstallResult:
    available: bool
    executable: str = ""
    command_kind: str = ""
    installed: bool = False
    updated: bool = False
    method: str = ""
    message: str = ""
    install_command: tuple[str, ...] = ()
    update_command: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppServerConfig:
    model_ref: str = DEFAULT_LEANSTRAL_MODEL_REF
    hf_file: str = ""
    model_path: str = ""
    model_cid: str = ""
    host: str = DEFAULT_LLAMA_CPP_HOST
    port: int = DEFAULT_LLAMA_CPP_PORT
    context_size: int = DEFAULT_LLAMA_CPP_CONTEXT_SIZE
    threads: int = 0
    gpu_layers: Optional[int] = None
    extra_args: tuple[str, ...] = ()
    log_dir: str = ""
    auto_sizing: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{int(self.port)}/v1"


@dataclass(frozen=True)
class LlamaCppModelCacheStatus:
    repo_id: str
    filename: str
    cache_root: str
    repo_cache_dir: str
    complete: bool = False
    downloading: bool = False
    local_path: str = ""
    local_size_bytes: int = 0
    partial_paths: tuple[str, ...] = ()
    partial_size_bytes: int = 0
    message: str = ""
    cache_backend: str = "hf"
    content_cid: str = ""
    content_sha256: str = ""
    content_multihash_sha256: str = ""
    content_cid_v1: str = ""
    content_cid_v1_path: str = ""
    content_addressed_path: str = ""
    content_hash_pending: bool = False
    content_hash_job_pid: int = 0
    content_hash_job_path: str = ""
    ipfs_kit_available: bool = False
    ipfs_remote_attempted: bool = False
    ipfs_remote_loaded: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppDeviceInfo:
    identifier: str
    name: str
    free_bytes: int = 0
    total_bytes: int = 0
    raw: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppGgufTensorInfo:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppGgufSizingInfo:
    path: str
    file_size_bytes: int
    alignment: int
    tensor_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)
    tensors: tuple[LlamaCppGgufTensorInfo, ...] = ()
    layer_count: int = 0
    repeating_layer_bytes: tuple[int, ...] = ()
    non_repeating_bytes: int = 0
    max_layer_bytes: int = 0
    avg_layer_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppAutoSizingPlan:
    enabled: bool
    reason: str = ""
    gpu_layers: Optional[int] = None
    batch_size: int = 0
    ubatch_size: int = 0
    context_size: int = 0
    device_free_bytes: int = 0
    device_total_bytes: int = 0
    usable_device_bytes: int = 0
    reserve_bytes: int = 0
    estimated_non_layer_bytes: int = 0
    estimated_layer_bytes: int = 0
    estimated_kv_bytes: int = 0
    estimated_workspace_bytes: int = 0
    model_path: str = ""
    device_identifier: str = ""
    device_name: str = ""
    layer_count: int = 0
    layer_multiplier: float = 0.0
    command_extra_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppServerResult:
    running: bool
    started: bool = False
    pid: Optional[int] = None
    base_url: str = ""
    command: tuple[str, ...] = ()
    log_path: str = ""
    install: LlamaCppInstallResult = field(default_factory=lambda: LlamaCppInstallResult(False))
    model_cache: LlamaCppModelCacheStatus = field(
        default_factory=lambda: LlamaCppModelCacheStatus(
            repo_id="",
            filename="",
            cache_root="",
            repo_cache_dir="",
        )
    )
    auto_sizing: LlamaCppAutoSizingPlan = field(
        default_factory=lambda: LlamaCppAutoSizingPlan(False, reason="not_requested")
    )
    message: str = ""
    model_manager_registered: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["install"] = self.install.to_dict()
        payload["model_cache"] = self.model_cache.to_dict()
        payload["auto_sizing"] = self.auto_sizing.to_dict()
        return payload


def _split_command(command: str) -> tuple[str, ...]:
    return tuple(shlex.split(command.strip())) if command and command.strip() else ()


def _run_command(command: Sequence[str], *, timeout_seconds: float = 600.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
        timeout=float(timeout_seconds),
        env=os.environ.copy(),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _llama_cpp_install_method() -> str:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_INSTALL_METHOD",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_INSTALL_METHOD",
        "ipfs_accelerate_py_LLAMA_CPP_INSTALL_METHOD",
        default=DEFAULT_LLAMA_CPP_INSTALL_METHOD,
    )
    return str(configured or DEFAULT_LLAMA_CPP_INSTALL_METHOD).strip().lower()


def _command_kind_for_executable(executable: object) -> str:
    name = Path(str(executable or "")).name
    if name == "llama-server":
        return "llama-server"
    return "llama"


def _candidate_executables(explicit_binary: str = "") -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if explicit_binary:
        kind = _command_kind_for_executable(explicit_binary)
        candidates.append((explicit_binary, kind))

    env_binary = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_BIN",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_BIN",
        "ipfs_accelerate_py_LLAMA_CPP_BIN",
    )
    if env_binary:
        kind = _command_kind_for_executable(env_binary)
        candidates.append((env_binary, kind))

    for path, kind in (
        (_managed_server_executable(), "llama-server"),
        (_managed_build_dir() / "llama-server", "llama-server"),
        (_managed_build_dir() / "llama", "llama"),
    ):
        candidates.append((str(path), kind))

    for name, kind in (("llama", "llama"), ("llama-server", "llama-server")):
        found = shutil.which(name)
        if found:
            candidates.append((found, kind))

    for path, kind in (
        (Path.home() / ".local" / "bin" / "llama", "llama"),
        (Path.home() / ".local" / "bin" / "llama-server", "llama-server"),
        (Path.home() / ".llama" / "bin" / "llama", "llama"),
        (Path.home() / ".llama" / "bin" / "llama-server", "llama-server"),
    ):
        candidates.append((str(path), kind))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for executable, kind in candidates:
        if not executable or executable in seen:
            continue
        seen.add(executable)
        unique.append((executable, kind))
    return unique


def find_llama_cpp_executable(explicit_binary: str = "") -> tuple[str, str]:
    """Return ``(executable, command_kind)`` for a usable llama.cpp command."""

    for executable, kind in _candidate_executables(explicit_binary):
        if os.path.isabs(executable):
            if os.access(executable, os.X_OK):
                return executable, kind
            continue
        found = shutil.which(executable)
        if found:
            return found, kind
    return "", ""


def _installer_command() -> tuple[str, ...]:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_INSTALL_CMD",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_INSTALL_CMD",
        "ipfs_accelerate_py_LLAMA_CPP_INSTALL_CMD",
    )
    command = configured or DEFAULT_LLAMA_CPP_INSTALL_CMD
    return ("sh", "-c", command)


def _source_install_command() -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "ipfs_accelerate_py.utils.llama_cpp",
        "--install-llama-cpp",
    )


def _update_command() -> tuple[str, ...]:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_UPDATE_CMD",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_UPDATE_CMD",
        "ipfs_accelerate_py_LLAMA_CPP_UPDATE_CMD",
    )
    if configured:
        return ("sh", "-c", configured)
    if _llama_cpp_install_method() == "source":
        return _source_install_command()
    return _installer_command()


def _nproc() -> str:
    try:
        value = os.cpu_count()
    except Exception:
        value = None
    return str(max(1, int(value or 1)))


def _tool_available(name: str) -> bool:
    return bool(shutil.which(name))


def _cuda_toolkit_available() -> bool:
    return bool(shutil.which("nvcc")) and Path("/usr/local/cuda").exists()


def _cuda_architectures() -> str:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_CUDA_ARCHITECTURES",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_CUDA_ARCHITECTURES",
    )
    if configured:
        return configured
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    try:
        code, stdout, _stderr = _run_command(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            timeout_seconds=10.0,
        )
    except Exception:
        return ""
    if code != 0:
        return ""
    for line in stdout.splitlines():
        value = line.strip()
        if not value or value == "[N/A]":
            continue
        match = re.search(r"(\d+)\.(\d+)", value)
        if match:
            return f"{match.group(1)}{match.group(2)}"
    return ""


def _append_output(log: list[str], label: str, code: int, stdout: str, stderr: str) -> None:
    detail = (stderr or stdout or "").strip()
    if len(detail) > 4000:
        detail = detail[-4000:]
    log.append(f"{label}: exit={code}" + (f"\n{detail}" if detail else ""))


def _install_llama_cpp_from_source(
    *,
    auto_update: bool = False,
    timeout_seconds: float = 1800.0,
) -> LlamaCppInstallResult:
    missing = [name for name in ("git", "cmake") if not _tool_available(name)]
    if missing:
        return LlamaCppInstallResult(
            available=False,
            method="source_build",
            message=f"missing build tools: {', '.join(missing)}",
            install_command=_source_install_command(),
        )

    source_dir = _managed_source_dir()
    build_dir = _managed_build_dir()
    repo = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_SOURCE_REPO",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_SOURCE_REPO",
        default=DEFAULT_LLAMA_CPP_SOURCE_REPO,
    )
    ref = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_SOURCE_REF",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_SOURCE_REF",
        default=DEFAULT_LLAMA_CPP_SOURCE_REF,
    )
    logs: list[str] = []

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if (source_dir / ".git").exists():
        if auto_update:
            fetch_cmd = ["git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", ref]
            code, stdout, stderr = _run_command(fetch_cmd, timeout_seconds=timeout_seconds)
            _append_output(logs, "git_fetch", code, stdout, stderr)
            if code != 0:
                return LlamaCppInstallResult(
                    available=False,
                    method="source_build",
                    message="\n".join(logs),
                    install_command=_source_install_command(),
                    update_command=tuple(fetch_cmd),
                )
            reset_cmd = ["git", "-C", str(source_dir), "reset", "--hard", "FETCH_HEAD"]
            code, stdout, stderr = _run_command(reset_cmd, timeout_seconds=timeout_seconds)
            _append_output(logs, "git_reset", code, stdout, stderr)
            if code != 0:
                return LlamaCppInstallResult(
                    available=False,
                    method="source_build",
                    message="\n".join(logs),
                    install_command=_source_install_command(),
                    update_command=tuple(reset_cmd),
                )
    else:
        clone_cmd = ["git", "clone", "--depth", "1", "--branch", ref, repo, str(source_dir)]
        code, stdout, stderr = _run_command(clone_cmd, timeout_seconds=timeout_seconds)
        _append_output(logs, "git_clone", code, stdout, stderr)
        if code != 0:
            clone_cmd = ["git", "clone", "--depth", "1", repo, str(source_dir)]
            code, stdout, stderr = _run_command(clone_cmd, timeout_seconds=timeout_seconds)
            _append_output(logs, "git_clone_default", code, stdout, stderr)
            if code != 0:
                return LlamaCppInstallResult(
                    available=False,
                    method="source_build",
                    message="\n".join(logs),
                    install_command=_source_install_command(),
                )

    cuda_enabled = _truthy(
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_BUILD_CUDA",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_BUILD_CUDA",
            default="1" if _cuda_toolkit_available() else "0",
        ),
        default=_cuda_toolkit_available(),
    )
    configure_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        f"-DGGML_CUDA={'ON' if cuda_enabled else 'OFF'}",
    ]
    build_ui = _truthy(
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_BUILD_UI",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_BUILD_UI",
            default="0",
        ),
        default=False,
    )
    configure_cmd.append(f"-DLLAMA_BUILD_UI={'ON' if build_ui else 'OFF'}")
    configure_cmd.append(f"-DLLAMA_USE_PREBUILT_UI={'ON' if build_ui else 'OFF'}")
    if cuda_enabled:
        arch = _cuda_architectures()
        if arch:
            configure_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    code, stdout, stderr = _run_command(configure_cmd, timeout_seconds=timeout_seconds)
    _append_output(logs, "cmake_configure", code, stdout, stderr)
    if code != 0:
        return LlamaCppInstallResult(
            available=False,
            method="source_build",
            message="\n".join(logs),
            install_command=tuple(configure_cmd),
        )

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-cli",
        "llama-server",
        "-j",
        _nproc(),
    ]
    code, stdout, stderr = _run_command(build_cmd, timeout_seconds=timeout_seconds)
    _append_output(logs, "cmake_build", code, stdout, stderr)
    executable = _managed_server_executable()
    if code != 0 or not os.access(executable, os.X_OK):
        return LlamaCppInstallResult(
            available=False,
            method="source_build",
            message="\n".join(logs),
            install_command=tuple(build_cmd),
        )
    return LlamaCppInstallResult(
        available=True,
        executable=str(executable),
        command_kind="llama-server",
        installed=True,
        updated=bool(auto_update),
        method="source_build",
        message="\n".join(logs[-2:]),
        install_command=_source_install_command(),
        update_command=_source_install_command() if auto_update else (),
    )


def ensure_llama_cpp(
    *,
    auto_install: bool = False,
    auto_update: bool = False,
    explicit_binary: str = "",
    timeout_seconds: float = 900.0,
) -> LlamaCppInstallResult:
    """Ensure a llama.cpp server-capable executable is available.

    ``auto_install`` and ``auto_update`` are deliberately explicit.  They may
    execute operator-owned install commands and therefore should only be enabled
    by deployment configuration or CLI flags.
    """

    executable, kind = find_llama_cpp_executable(explicit_binary)
    installed = False
    updated = False
    install_cmd: tuple[str, ...] = ()
    update_cmd: tuple[str, ...] = ()
    install_method = _llama_cpp_install_method()
    explicit_or_env_binary = bool(
        str(explicit_binary or "").strip()
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_BIN",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_BIN",
            "ipfs_accelerate_py_LLAMA_CPP_BIN",
        )
    )

    if executable and auto_update:
        if install_method == "source" and not explicit_or_env_binary:
            result = _install_llama_cpp_from_source(
                auto_update=True,
                timeout_seconds=timeout_seconds,
            )
            if result.available:
                return result
            return LlamaCppInstallResult(
                available=True,
                executable=executable,
                command_kind=kind,
                updated=False,
                method="existing_binary",
                message=result.message or "managed llama.cpp source update failed",
                update_command=result.update_command or result.install_command,
            )
        else:
            update_cmd = _update_command()
            code, stdout, stderr = _run_command(update_cmd, timeout_seconds=timeout_seconds)
            if code != 0:
                return LlamaCppInstallResult(
                    available=True,
                    executable=executable,
                    command_kind=kind,
                    updated=False,
                    method="existing_binary",
                    message=(stderr or stdout or "llama.cpp update command failed").strip(),
                    update_command=update_cmd,
                )
            updated = True
            executable, kind = find_llama_cpp_executable(explicit_binary)

    prefer_managed_cuda = _truthy(
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_PREFER_MANAGED_CUDA",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_PREFER_MANAGED_CUDA",
            default="1",
        ),
        default=True,
    )
    if (
        executable
        and auto_install
        and install_method == "source"
        and prefer_managed_cuda
        and not explicit_or_env_binary
        and _cuda_toolkit_available()
        and _llama_cpp_cublas_major(executable) < 13
    ):
        result = _install_llama_cpp_from_source(
            auto_update=auto_update,
            timeout_seconds=timeout_seconds,
        )
        if result.available:
            return result

    if not executable and auto_install:
        if install_method == "source":
            result = _install_llama_cpp_from_source(
                auto_update=auto_update,
                timeout_seconds=timeout_seconds,
            )
            if not result.available:
                return result
            return result
        else:
            install_cmd = _installer_command()
            code, stdout, stderr = _run_command(install_cmd, timeout_seconds=timeout_seconds)
            if code != 0:
                return LlamaCppInstallResult(
                    available=False,
                    installed=False,
                    method="install_command",
                    message=(stderr or stdout or "llama.cpp install failed").strip(),
                    install_command=install_cmd,
                )
            installed = True
            executable, kind = find_llama_cpp_executable(explicit_binary)

    if executable:
        return LlamaCppInstallResult(
            available=True,
            executable=executable,
            command_kind=kind,
            installed=installed,
            updated=updated,
            method="install_command" if installed else "existing_binary",
            install_command=install_cmd,
            update_command=update_cmd,
        )

    return LlamaCppInstallResult(
        available=False,
        method="not_found",
        message=(
            "llama.cpp CLI not found. Install it manually, set "
            "IPFS_ACCELERATE_LLAMA_CPP_BIN, or rerun with auto_install enabled."
        ),
        install_command=_installer_command(),
    )


def build_llama_cpp_server_command(
    config: LlamaCppServerConfig,
    *,
    executable: str,
    command_kind: str,
) -> tuple[str, ...]:
    """Build a llama.cpp OpenAI-compatible server command."""

    model_path = str(config.model_path or "").strip()
    if command_kind == "llama-server":
        cmd = [executable, "-m", model_path] if model_path else [executable, "-hf", config.model_ref]
    else:
        cmd = [executable, "serve", "-m", model_path] if model_path else [executable, "serve", "-hf", config.model_ref]

    if not model_path and str(config.hf_file or "").strip():
        cmd.extend(["--hf-file", str(config.hf_file).strip()])
    cmd.extend(["--host", str(config.host), "--port", str(int(config.port))])
    if int(config.context_size) > 0:
        cmd.extend(["-c", str(int(config.context_size))])
    if int(config.threads or 0) > 0:
        cmd.extend(["-t", str(int(config.threads))])
    if config.gpu_layers is not None:
        cmd.extend(["-ngl", str(int(config.gpu_layers))])
    cmd.extend([str(arg) for arg in config.extra_args if str(arg).strip()])
    return tuple(cmd)


def _repo_id_from_model_ref(model_ref: str) -> str:
    value = str(model_ref or "").strip()
    if ":" in value:
        value = value.split(":", 1)[0]
    return value


def _safe_path_component(value: object, *, default: str = "model") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or default


def _model_cache_filename(config: LlamaCppServerConfig) -> str:
    filename = str(config.hf_file or "").strip()
    if filename:
        return Path(filename).name
    model_path = str(config.model_path or "").strip()
    if model_path:
        return Path(model_path).name
    return DEFAULT_LEANSTRAL_FILENAME


def _model_artifact_cache_root() -> Path:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CACHE_DIR",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL_CACHE_DIR",
    )
    if configured:
        return Path(configured).expanduser()
    return _default_cache_dir() / "models"


def _model_artifact_cache_enabled(*, default: bool = False) -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CACHE",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL_CACHE",
    )
    if configured:
        return _truthy(configured, default=default)
    return bool(default)


def _model_artifact_backend() -> str:
    value = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_BACKEND",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CACHE_BACKEND",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL_CACHE_BACKEND",
        default="local",
    )
    return str(value or "local").strip().lower().replace("-", "_")


def _model_artifact_ipfs_enabled() -> bool:
    return _model_artifact_backend() in {"auto", "ipfs", "ipfs_kit", "remote"}


def _model_cid_from_config(config: LlamaCppServerConfig) -> str:
    return str(config.model_cid or _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CID",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CID",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL_CID",
    )).strip()


def _model_ref_manifest_path(config: LlamaCppServerConfig) -> Path:
    repo_id = _repo_id_from_model_ref(config.model_ref) or "local"
    filename = _model_cache_filename(config)
    return (
        _model_artifact_cache_root()
        / "refs"
        / _safe_path_component(repo_id, default="repo")
        / f"{_safe_path_component(filename, default='model.gguf')}.json"
    )


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _status_from_existing_model_path(
    config: LlamaCppServerConfig,
    path: Path,
    *,
    cache_backend: str,
    message: str = "complete",
    content_cid: str = "",
    content_sha256: str = "",
    content_multihash_sha256: str = "",
    content_cid_v1: str = "",
    content_cid_v1_path: str = "",
    content_addressed_path: str = "",
    content_hash_pending: bool = False,
    content_hash_job_pid: int = 0,
    content_hash_job_path: str = "",
    cache_root: str = "",
    repo_cache_dir: str = "",
    ipfs_kit_available: bool = False,
    ipfs_remote_attempted: bool = False,
    ipfs_remote_loaded: bool = False,
) -> LlamaCppModelCacheStatus:
    resolved = path.expanduser()
    try:
        size = int(resolved.stat().st_size)
    except Exception:
        size = 0
    repo_id = _repo_id_from_model_ref(config.model_ref)
    filename = _model_cache_filename(config)
    root_text = cache_root or str(resolved.parent)
    repo_cache_text = repo_cache_dir or str(resolved.parent)
    if content_sha256 and (not content_multihash_sha256 or not content_cid_v1):
        content_multihash_sha256, content_cid_v1 = _raw_sha256_cid_v1_from_digest(
            content_sha256
        )
    return LlamaCppModelCacheStatus(
        repo_id=repo_id,
        filename=filename,
        cache_root=root_text,
        repo_cache_dir=repo_cache_text,
        complete=bool(size > 0 and resolved.exists()),
        downloading=False,
        local_path=str(resolved),
        local_size_bytes=size,
        message=message,
        cache_backend=cache_backend,
        content_cid=content_cid,
        content_sha256=content_sha256,
        content_multihash_sha256=content_multihash_sha256,
        content_cid_v1=content_cid_v1,
        content_cid_v1_path=content_cid_v1_path,
        content_addressed_path=content_addressed_path or str(resolved),
        content_hash_pending=bool(content_hash_pending),
        content_hash_job_pid=int(content_hash_job_pid or 0),
        content_hash_job_path=content_hash_job_path,
        ipfs_kit_available=ipfs_kit_available,
        ipfs_remote_attempted=ipfs_remote_attempted,
        ipfs_remote_loaded=ipfs_remote_loaded,
    )


def _model_artifact_status_from_manifest(config: LlamaCppServerConfig) -> Optional[LlamaCppModelCacheStatus]:
    payload = _read_json_file(_model_ref_manifest_path(config))
    local_path = str(payload.get("content_addressed_path") or payload.get("local_path") or "").strip()
    if not local_path:
        return None
    candidate = Path(local_path).expanduser()
    if not candidate.exists():
        return None
    return _status_from_existing_model_path(
        config,
        candidate,
        cache_backend=str(payload.get("cache_backend") or "content_addressed_disk"),
        content_cid=str(payload.get("content_cid") or ""),
        content_sha256=str(payload.get("content_sha256") or payload.get("sha256") or ""),
        content_multihash_sha256=str(payload.get("content_multihash_sha256") or ""),
        content_cid_v1=str(payload.get("content_cid_v1") or ""),
        content_cid_v1_path=str(payload.get("content_cid_v1_path") or ""),
        content_addressed_path=str(candidate),
        content_hash_pending=bool(
            payload.get("content_hash_pending")
            or payload.get("hash_pending")
            or False
        ),
        content_hash_job_pid=int(payload.get("content_hash_job_pid") or 0),
        content_hash_job_path=str(payload.get("content_hash_job_path") or ""),
        cache_root=str(_model_artifact_cache_root()),
        repo_cache_dir=str(candidate.parent),
        ipfs_kit_available=bool(payload.get("ipfs_kit_available") or False),
        ipfs_remote_attempted=bool(payload.get("ipfs_remote_attempted") or False),
        ipfs_remote_loaded=bool(payload.get("ipfs_remote_loaded") or False),
    )


def _model_artifact_status_from_cid_alias(
    config: LlamaCppServerConfig,
    cid_v1: str,
) -> Optional[LlamaCppModelCacheStatus]:
    cid = str(cid_v1 or "").strip()
    if not cid:
        return None
    candidate = _content_cid_v1_model_path(cid, _model_cache_filename(config))
    if not candidate.exists():
        return None
    payload = _read_json_file(candidate.with_name(f"{candidate.name}.json"))
    return _status_from_existing_model_path(
        config,
        candidate,
        cache_backend=str(payload.get("cache_backend") or "content_addressed_disk"),
        content_cid=str(payload.get("content_cid") or ""),
        content_sha256=str(payload.get("content_sha256") or payload.get("sha256") or ""),
        content_multihash_sha256=str(payload.get("content_multihash_sha256") or ""),
        content_cid_v1=str(payload.get("content_cid_v1") or cid),
        content_cid_v1_path=str(candidate),
        content_addressed_path=str(payload.get("content_addressed_path") or candidate),
        content_hash_pending=bool(
            payload.get("content_hash_pending")
            or payload.get("hash_pending")
            or False
        ),
        content_hash_job_pid=int(payload.get("content_hash_job_pid") or 0),
        content_hash_job_path=str(payload.get("content_hash_job_path") or ""),
        cache_root=str(_model_artifact_cache_root()),
        repo_cache_dir=str(candidate.parent),
        ipfs_kit_available=bool(payload.get("ipfs_kit_available") or False),
        ipfs_remote_attempted=bool(payload.get("ipfs_remote_attempted") or False),
        ipfs_remote_loaded=bool(payload.get("ipfs_remote_loaded") or False),
    )


def _hash_file_sha256(path: Path, *, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unsigned_varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("varint_value_must_be_non_negative")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _base32_lower_no_padding(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").lower().rstrip("=")


def _raw_sha256_cid_v1_from_digest(sha256_hex: str) -> tuple[str, str]:
    try:
        digest = bytes.fromhex(str(sha256_hex or "").strip())
    except ValueError:
        return "", ""
    if len(digest) != 32:
        return "", ""
    multihash_bytes = _unsigned_varint(0x12) + _unsigned_varint(len(digest)) + digest
    cid_bytes = _unsigned_varint(1) + _unsigned_varint(0x55) + multihash_bytes
    return multihash_bytes.hex(), "b" + _base32_lower_no_padding(cid_bytes)


def _content_addressed_model_path(sha256_hex: str, filename: str) -> Path:
    digest = _safe_path_component(sha256_hex, default="sha256")
    return (
        _model_artifact_cache_root()
        / "sha256"
        / digest[:2]
        / digest
        / _safe_path_component(Path(filename).name, default="model.gguf")
    )


def _content_cid_v1_model_path(cid_v1: str, filename: str) -> Path:
    return (
        _model_artifact_cache_root()
        / "cid-v1"
        / _safe_path_component(cid_v1, default="cid")
        / _safe_path_component(Path(filename).name, default="model.gguf")
    )


def _model_artifact_hash_jobs_dir() -> Path:
    return _model_artifact_cache_root() / "hash-jobs"


def _async_model_hash_enabled(*, default: bool = True) -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_ASYNC_MODEL_HASH",
        "ipfs_accelerate_py_LLAMA_CPP_ASYNC_MODEL_HASH",
    )
    if configured:
        return _truthy(configured, default=default)
    return bool(default)


def _async_model_hash_min_bytes() -> int:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH_MIN_BYTES",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_ASYNC_MODEL_HASH_MIN_BYTES",
        "ipfs_accelerate_py_LLAMA_CPP_ASYNC_MODEL_HASH_MIN_BYTES",
        default=str(1024**3),
    )
    try:
        return max(0, int(configured))
    except (TypeError, ValueError):
        return 1024**3


def _should_hash_model_async(
    source_path: Path,
    *,
    async_hash: Optional[bool] = None,
) -> bool:
    enabled = (
        _async_model_hash_enabled(default=True)
        if async_hash is None
        else bool(async_hash)
    )
    if not enabled:
        return False
    try:
        size = int(Path(source_path).expanduser().stat().st_size)
    except Exception:
        return False
    return size >= _async_model_hash_min_bytes()


def _model_artifact_hash_job_path(
    config: LlamaCppServerConfig,
    source_path: Path,
    *,
    content_cid: str = "",
) -> Path:
    source = Path(source_path).expanduser()
    try:
        stat_result = source.stat()
        source_identity = {
            "path": str(source.resolve()),
            "size": int(stat_result.st_size),
            "mtime_ns": int(getattr(stat_result, "st_mtime_ns", 0) or 0),
        }
    except Exception:
        source_identity = {"path": str(source)}
    payload = {
        "model_ref": config.model_ref,
        "filename": _model_cache_filename(config),
        "content_cid": str(content_cid or "").strip(),
        "source": source_identity,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return _model_artifact_hash_jobs_dir() / f"{key}.json"


def _hash_pending_backend(source_backend: str) -> str:
    backend = str(source_backend or "local").strip() or "local"
    if backend.endswith("_hash_pending"):
        return backend
    return f"{backend}_hash_pending"


def _write_hash_pending_manifest(
    config: LlamaCppServerConfig,
    source_path: Path,
    *,
    source_backend: str,
    content_cid: str = "",
    pin: bool = False,
    ipfs_remote_attempted: bool = False,
    ipfs_remote_loaded: bool = False,
    job_path: Path = Path(),
    job_pid: int = 0,
) -> Mapping[str, object]:
    source = Path(source_path).expanduser()
    try:
        source = source.resolve()
    except Exception:
        pass
    filename = _model_cache_filename(config)
    try:
        size_bytes = int(source.stat().st_size)
    except Exception:
        size_bytes = 0
    manifest: dict[str, object] = {
        "repo_id": _repo_id_from_model_ref(config.model_ref),
        "filename": filename,
        "source_path": str(source),
        "local_path": str(source),
        "source_backend": source_backend,
        "cache_backend": _hash_pending_backend(source_backend),
        "content_cid": str(content_cid or "").strip(),
        "content_sha256": "",
        "content_multihash_sha256": "",
        "content_cid_v1": "",
        "content_cid_v1_path": "",
        "content_addressed_path": "",
        "size_bytes": size_bytes,
        "materialized_by": "pending_hash",
        "ipfs_kit_available": _model_artifact_ipfs_enabled(),
        "ipfs_remote_attempted": ipfs_remote_attempted,
        "ipfs_remote_loaded": ipfs_remote_loaded,
        "pin_requested": bool(pin),
        "hash_pending": True,
        "content_hash_pending": True,
        "content_hash_job_pid": int(job_pid or 0),
        "content_hash_job_path": str(job_path)
        if str(job_path or "").strip() not in {"", "."}
        else "",
        "updated_at": time.time(),
    }
    _write_json_atomic(_model_ref_manifest_path(config), manifest)
    return manifest


def _hash_pending_status_from_manifest(
    config: LlamaCppServerConfig,
    source_path: Path,
    manifest: Mapping[str, object],
) -> LlamaCppModelCacheStatus:
    return _status_from_existing_model_path(
        config,
        Path(source_path),
        cache_backend=str(manifest.get("cache_backend") or "local_hash_pending"),
        message="complete_hash_pending",
        content_cid=str(manifest.get("content_cid") or ""),
        content_hash_pending=True,
        content_hash_job_pid=int(manifest.get("content_hash_job_pid") or 0),
        content_hash_job_path=str(manifest.get("content_hash_job_path") or ""),
        cache_root=str(_model_artifact_cache_root()),
        repo_cache_dir=str(Path(source_path).expanduser().parent),
        ipfs_kit_available=bool(manifest.get("ipfs_kit_available") or False),
        ipfs_remote_attempted=bool(manifest.get("ipfs_remote_attempted") or False),
        ipfs_remote_loaded=bool(manifest.get("ipfs_remote_loaded") or False),
    )


def _model_artifact_link_modes() -> tuple[str, ...]:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_LINK_MODES",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CACHE_LINK_MODES",
        "ipfs_accelerate_py_LLAMA_CPP_MODEL_CACHE_LINK_MODES",
        default="hardlink,symlink,copy",
    )
    modes = tuple(
        mode.strip().lower().replace("-", "_")
        for mode in re.split(r"[,:\s]+", configured)
        if mode.strip()
    )
    return modes or ("hardlink", "symlink", "copy")


def _link_or_copy_model_file(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    last_error: Optional[BaseException] = None
    for mode in _model_artifact_link_modes():
        try:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if mode == "hardlink":
                os.link(source, tmp_path)
            elif mode == "symlink":
                os.symlink(source, tmp_path)
            elif mode == "copy":
                shutil.copy2(source, tmp_path)
            else:
                continue
            os.replace(tmp_path, destination)
            return mode
        except Exception as exc:
            last_error = exc
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
    if last_error is not None:
        raise last_error
    raise OSError("no_model_cache_link_modes_available")


def _extract_ipfs_cid(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("cid", "CID", "Cid", "Hash", "hash", "multihash"):
            if key in value:
                cid = _extract_ipfs_cid(value.get(key))
                if cid:
                    return cid
        for key in ("result", "data", "response"):
            if key in value:
                cid = _extract_ipfs_cid(value.get(key))
                if cid:
                    return cid
    if isinstance(value, (list, tuple)):
        for item in value:
            cid = _extract_ipfs_cid(item)
            if cid:
                return cid
    for attr in ("cid", "hash", "Hash"):
        try:
            cid = _extract_ipfs_cid(getattr(value, attr))
        except Exception:
            cid = ""
        if cid:
            return cid
    return ""


def _coerce_ipfs_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Mapping):
        for key in ("bytes", "data", "content", "payload"):
            if key in value:
                data = _coerce_ipfs_bytes(value.get(key))
                if data:
                    return data
    return b""


def _ipfs_kit_model_storage():
    try:
        from ipfs_accelerate_py.ipfs_kit_integration import IPFSKitStorage
    except Exception:
        try:
            from ipfs_kit_integration import IPFSKitStorage  # type: ignore
        except Exception:
            return None
    try:
        return IPFSKitStorage(
            enable_ipfs_kit=True,
            cache_dir=str(_model_artifact_cache_root() / "ipfs_kit"),
        )
    except Exception:
        return None


def _ipfs_storage_targets(storage: object) -> tuple[object, ...]:
    targets: list[object] = []
    for value in (
        getattr(storage, "ipfs_kit_client", None),
        getattr(storage, "storage", None),
        storage,
    ):
        if value is None:
            continue
        if value not in targets:
            targets.append(value)
        if isinstance(value, Mapping):
            for nested in value.values():
                if nested is not None and nested not in targets and not isinstance(nested, type):
                    targets.append(nested)
    return tuple(targets)


def _call_ipfs_method(target: object, name: str):
    if isinstance(target, Mapping):
        method = target.get(name)
    else:
        method = getattr(target, name, None)
    return method if callable(method) else None


def _store_model_path_with_ipfs_kit(path: Path, *, filename: str, pin: bool) -> tuple[str, bool]:
    storage = _ipfs_kit_model_storage()
    if storage is None:
        return "", False
    is_available = bool(getattr(storage, "is_available", lambda: False)())
    if not is_available:
        return "", False
    for target in _ipfs_storage_targets(storage):
        for name in ("add_file", "add", "ipfs_add", "store_file", "store"):
            method = _call_ipfs_method(target, name)
            if method is None:
                continue
            attempts = (
                ((str(path),), {"pin": bool(pin), "filename": filename}),
                ((str(path),), {"pin": bool(pin)}),
                ((str(path),), {}),
            )
            for args, kwargs in attempts:
                try:
                    cid = _extract_ipfs_cid(method(*args, **kwargs))
                except TypeError:
                    continue
                except Exception:
                    continue
                if cid:
                    return cid, is_available
    return "", is_available


def _retrieve_model_cid_to_source_path(
    storage: object,
    cid: str,
    output_path: Path,
    *,
    allow_remote: bool,
) -> tuple[Path, bool]:
    cache_dir = getattr(storage, "cache_dir", None)
    if cache_dir is not None:
        local_payload = Path(cache_dir).expanduser() / cid
        if local_payload.exists():
            return local_payload, False
    if not allow_remote:
        return Path(), False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for target in _ipfs_storage_targets(storage):
        for name in ("get_file", "download", "retrieve_file", "retrieve", "cat", "get", "ipfs_cat"):
            method = _call_ipfs_method(target, name)
            if method is None:
                continue
            attempts = (
                ((cid, str(output_path)), {}),
                ((cid,), {"output_path": str(output_path)}),
                ((cid,), {"path": str(output_path)}),
                ((cid,), {"lpath": str(output_path)}),
                ((cid,), {}),
                ((f"/ipfs/{cid}",), {}),
            )
            for args, kwargs in attempts:
                try:
                    result = method(*args, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    continue
                if output_path.exists() and output_path.stat().st_size > 0:
                    return output_path, True
                if isinstance(result, (str, os.PathLike)):
                    result_path = Path(result).expanduser()
                    if result_path.exists():
                        return result_path, True
                data = _coerce_ipfs_bytes(result)
                if data:
                    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
                    tmp_path.write_bytes(data)
                    os.replace(tmp_path, output_path)
                    return output_path, True
    return Path(), True


def _register_llama_cpp_model_artifact(
    config: LlamaCppServerConfig,
    source_path: Path,
    *,
    source_backend: str = "hf",
    content_cid: str = "",
    pin: bool = False,
    ipfs_remote_attempted: bool = False,
    ipfs_remote_loaded: bool = False,
) -> LlamaCppModelCacheStatus:
    source = Path(source_path).expanduser()
    if not source.exists():
        return LlamaCppModelCacheStatus(
            repo_id=_repo_id_from_model_ref(config.model_ref),
            filename=_model_cache_filename(config),
            cache_root=str(_model_artifact_cache_root()),
            repo_cache_dir=str(source.parent),
            message="artifact_source_missing",
            cache_backend=source_backend,
            content_cid=content_cid,
            ipfs_remote_attempted=ipfs_remote_attempted,
            ipfs_remote_loaded=ipfs_remote_loaded,
        )
    try:
        source = source.resolve()
    except Exception:
        pass
    sha256_hex = _hash_file_sha256(source)
    content_multihash_sha256, content_cid_v1 = _raw_sha256_cid_v1_from_digest(sha256_hex)
    filename = _model_cache_filename(config)
    destination = _content_addressed_model_path(sha256_hex, filename)
    materialized_by = _link_or_copy_model_file(source, destination)
    cid_v1_destination = Path()
    cid_v1_materialized_by = ""
    if content_cid_v1:
        cid_v1_destination = _content_cid_v1_model_path(content_cid_v1, filename)
        try:
            if cid_v1_destination.resolve() != destination.resolve():
                cid_v1_materialized_by = _link_or_copy_model_file(destination, cid_v1_destination)
        except Exception:
            cid_v1_materialized_by = _link_or_copy_model_file(destination, cid_v1_destination)
    stored_cid = str(content_cid or "").strip()
    ipfs_available = False
    if _model_artifact_ipfs_enabled():
        remote_cid, ipfs_available = _store_model_path_with_ipfs_kit(
            destination,
            filename=filename,
            pin=pin,
        )
        stored_cid = remote_cid or stored_cid
    cache_backend = "ipfs_kit" if stored_cid and (ipfs_available or source_backend == "ipfs_kit") else "content_addressed_disk"
    manifest = {
        "repo_id": _repo_id_from_model_ref(config.model_ref),
        "filename": filename,
        "source_path": str(source),
        "source_backend": source_backend,
        "cache_backend": cache_backend,
        "content_cid": stored_cid,
        "content_sha256": sha256_hex,
        "content_multihash_sha256": content_multihash_sha256,
        "content_cid_v1": content_cid_v1,
        "content_cid_v1_path": str(cid_v1_destination) if cid_v1_destination else "",
        "content_addressed_path": str(destination),
        "size_bytes": int(destination.stat().st_size),
        "materialized_by": materialized_by,
        "cid_v1_materialized_by": cid_v1_materialized_by,
        "ipfs_kit_available": ipfs_available,
        "ipfs_remote_attempted": ipfs_remote_attempted,
        "ipfs_remote_loaded": ipfs_remote_loaded,
        "updated_at": time.time(),
    }
    _write_json_atomic(_model_ref_manifest_path(config), manifest)
    if cid_v1_destination:
        _write_json_atomic(cid_v1_destination.with_name(f"{cid_v1_destination.name}.json"), manifest)
    return _status_from_existing_model_path(
        config,
        destination,
        cache_backend=cache_backend,
        content_cid=stored_cid,
        content_sha256=sha256_hex,
        content_multihash_sha256=content_multihash_sha256,
        content_cid_v1=content_cid_v1,
        content_cid_v1_path=str(cid_v1_destination) if cid_v1_destination else "",
        content_addressed_path=str(destination),
        cache_root=str(_model_artifact_cache_root()),
        repo_cache_dir=str(destination.parent),
        ipfs_kit_available=ipfs_available,
        ipfs_remote_attempted=ipfs_remote_attempted,
        ipfs_remote_loaded=ipfs_remote_loaded,
    )


def _finalize_llama_cpp_model_artifact_hash(
    config: LlamaCppServerConfig,
    source_path: Path,
    *,
    source_backend: str = "hf",
    content_cid: str = "",
    pin: bool = False,
    ipfs_remote_attempted: bool = False,
    ipfs_remote_loaded: bool = False,
    track_model: bool = False,
    job_path: Path = Path(),
) -> LlamaCppModelCacheStatus:
    """Finalize SHA/CID identity for a usable local model path."""

    job_path = (
        Path(job_path)
        if str(job_path or "").strip() not in {"", "."}
        else Path()
    )
    try:
        status = _register_llama_cpp_model_artifact(
            config,
            source_path,
            source_backend=source_backend,
            content_cid=content_cid,
            pin=pin,
            ipfs_remote_attempted=ipfs_remote_attempted,
            ipfs_remote_loaded=ipfs_remote_loaded,
        )
        model_manager_registered = _register_llama_cpp_model_with_manager(
            config,
            status,
            enabled=bool(track_model) and bool(status.complete),
        )
        if str(job_path or "").strip() not in {"", "."}:
            _write_json_atomic(
                Path(job_path),
                {
                    "status": "complete" if status.complete else "failed",
                    "model_ref": config.model_ref,
                    "filename": _model_cache_filename(config),
                    "source_path": str(Path(source_path).expanduser()),
                    "result": status.to_dict(),
                    "model_manager_registered": model_manager_registered,
                    "updated_at": time.time(),
                },
            )
        return status
    except Exception as exc:
        if str(job_path or "").strip() not in {"", "."}:
            try:
                _write_json_atomic(
                    Path(job_path),
                    {
                        "status": "failed",
                        "model_ref": config.model_ref,
                        "filename": _model_cache_filename(config),
                        "source_path": str(Path(source_path).expanduser()),
                        "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                        "updated_at": time.time(),
                    },
                )
            except Exception:
                pass
        raise


def _module_pythonpath_env() -> Mapping[str, str]:
    env = os.environ.copy()
    try:
        package_root = str(Path(__file__).resolve().parents[2])
    except Exception:
        package_root = ""
    existing = str(env.get("PYTHONPATH") or "")
    if package_root and package_root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            package_root
            if not existing
            else package_root + os.pathsep + existing
        )
    env["IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH"] = "0"
    return env


def _register_llama_cpp_model_artifact_async(
    config: LlamaCppServerConfig,
    source_path: Path,
    *,
    source_backend: str = "hf",
    content_cid: str = "",
    pin: bool = False,
    ipfs_remote_attempted: bool = False,
    ipfs_remote_loaded: bool = False,
    track_model: bool = False,
) -> LlamaCppModelCacheStatus:
    source = Path(source_path).expanduser()
    if not source.exists():
        return LlamaCppModelCacheStatus(
            repo_id=_repo_id_from_model_ref(config.model_ref),
            filename=_model_cache_filename(config),
            cache_root=str(_model_artifact_cache_root()),
            repo_cache_dir=str(source.parent),
            message="artifact_source_missing",
            cache_backend=source_backend,
            content_cid=content_cid,
            ipfs_remote_attempted=ipfs_remote_attempted,
            ipfs_remote_loaded=ipfs_remote_loaded,
        )
    try:
        source = source.resolve()
    except Exception:
        pass
    job_path = _model_artifact_hash_job_path(config, source, content_cid=content_cid)
    existing_job = _read_json_file(job_path)
    existing_pid = int(existing_job.get("pid") or 0)
    if existing_pid and _process_alive(existing_pid):
        manifest = _write_hash_pending_manifest(
            config,
            source,
            source_backend=source_backend,
            content_cid=content_cid,
            pin=pin,
            ipfs_remote_attempted=ipfs_remote_attempted,
            ipfs_remote_loaded=ipfs_remote_loaded,
            job_path=job_path,
            job_pid=existing_pid,
        )
        return _hash_pending_status_from_manifest(config, source, manifest)

    manifest = _write_hash_pending_manifest(
        config,
        source,
        source_backend=source_backend,
        content_cid=content_cid,
        pin=pin,
        ipfs_remote_attempted=ipfs_remote_attempted,
        ipfs_remote_loaded=ipfs_remote_loaded,
        job_path=job_path,
        job_pid=0,
    )
    log_path = job_path.with_suffix(".log")
    command: list[str] = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.utils.llama_cpp",
        "--finalize-model-artifact",
        "--finalize-source-path",
        str(source),
        "--finalize-source-backend",
        str(source_backend or "hf"),
        "--model-ref",
        config.model_ref,
        "--hf-file",
        _model_cache_filename(config),
        "--model-path",
        str(source),
        "--finalize-job-path",
        str(job_path),
    ]
    if content_cid:
        command.extend(["--finalize-content-cid", str(content_cid)])
    if pin:
        command.append("--pin-model")
    if ipfs_remote_attempted:
        command.append("--finalize-ipfs-remote-attempted")
    if ipfs_remote_loaded:
        command.append("--finalize-ipfs-remote-loaded")
    if track_model:
        command.append("--track-model")
    else:
        command.append("--no-track-model")

    try:
        job_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=dict(_module_pythonpath_env()),
            )
        finally:
            log_handle.close()
        job_payload = {
            "status": "running",
            "pid": proc.pid,
            "command": command,
            "log_path": str(log_path),
            "model_ref": config.model_ref,
            "filename": _model_cache_filename(config),
            "source_path": str(source),
            "source_backend": source_backend,
            "content_cid": str(content_cid or "").strip(),
            "manifest_path": str(_model_ref_manifest_path(config)),
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        _write_json_atomic(job_path, job_payload)
        manifest = dict(manifest)
        manifest["content_hash_job_pid"] = int(proc.pid)
        manifest["content_hash_job_path"] = str(job_path)
        manifest["updated_at"] = time.time()
        _write_json_atomic(_model_ref_manifest_path(config), manifest)
        return _hash_pending_status_from_manifest(config, source, manifest)
    except Exception as exc:
        job_payload = {
            "status": "spawn_failed",
            "model_ref": config.model_ref,
            "filename": _model_cache_filename(config),
            "source_path": str(source),
            "source_backend": source_backend,
            "content_cid": str(content_cid or "").strip(),
            "error": f"{type(exc).__name__}:{str(exc)[:240]}",
            "updated_at": time.time(),
        }
        try:
            _write_json_atomic(job_path, job_payload)
        except Exception:
            pass
        manifest = dict(manifest)
        manifest["content_hash_job_path"] = str(job_path)
        _write_json_atomic(_model_ref_manifest_path(config), manifest)
        status = _hash_pending_status_from_manifest(config, source, manifest)
        return replace(
            status,
            message=f"complete_hash_pending_spawn_failed:{type(exc).__name__}",
        )


def _retrieve_llama_cpp_model_artifact_from_cid(
    config: LlamaCppServerConfig,
    cid: str,
    *,
    allow_remote: bool,
    async_hash: Optional[bool] = None,
    track_model: bool = False,
) -> LlamaCppModelCacheStatus:
    cid_status = _model_artifact_status_from_cid_alias(config, cid)
    if cid_status is not None and cid_status.complete:
        return cid_status
    storage = _ipfs_kit_model_storage()
    if storage is None:
        return LlamaCppModelCacheStatus(
            repo_id=_repo_id_from_model_ref(config.model_ref),
            filename=_model_cache_filename(config),
            cache_root=str(_model_artifact_cache_root()),
            repo_cache_dir=str(_model_artifact_cache_root()),
            message="ipfs_kit_unavailable",
            cache_backend="ipfs_kit",
            content_cid=cid,
            ipfs_remote_attempted=bool(allow_remote),
        )
    downloads_dir = _model_artifact_cache_root() / "downloads" / _safe_path_component(cid, default="cid")
    candidate_path = downloads_dir / _model_cache_filename(config)
    source_path, remote_loaded = _retrieve_model_cid_to_source_path(
        storage,
        cid,
        candidate_path,
        allow_remote=allow_remote,
    )
    if not source_path:
        return LlamaCppModelCacheStatus(
            repo_id=_repo_id_from_model_ref(config.model_ref),
            filename=_model_cache_filename(config),
            cache_root=str(_model_artifact_cache_root()),
            repo_cache_dir=str(downloads_dir),
            message="ipfs_model_not_found",
            cache_backend="ipfs_kit",
            content_cid=cid,
            ipfs_kit_available=bool(getattr(storage, "is_available", lambda: False)()),
            ipfs_remote_attempted=bool(allow_remote),
        )
    if _should_hash_model_async(source_path, async_hash=async_hash):
        return _register_llama_cpp_model_artifact_async(
            config,
            source_path,
            source_backend="ipfs_kit",
            content_cid=cid,
            ipfs_remote_attempted=bool(allow_remote),
            ipfs_remote_loaded=bool(remote_loaded),
            track_model=track_model,
        )
    return _register_llama_cpp_model_artifact(
        config,
        source_path,
        source_backend="ipfs_kit",
        content_cid=cid,
        ipfs_remote_attempted=bool(allow_remote),
        ipfs_remote_loaded=bool(remote_loaded),
    )


def _serve_model_cache_via_local_path(model_cache: LlamaCppModelCacheStatus) -> bool:
    backend = str(model_cache.cache_backend or "")
    return bool(
        model_cache.complete
        and model_cache.local_path
        and (
            backend in {"content_addressed_disk", "ipfs_kit", "model_path"}
            or backend.endswith("_hash_pending")
        )
    )


def llama_cpp_model_cache_status(
    config: LlamaCppServerConfig,
    *,
    cache_root: Optional[object] = None,
) -> LlamaCppModelCacheStatus:
    """Inspect the configured local, artifact, and Hugging Face GGUF caches."""

    repo_id = _repo_id_from_model_ref(config.model_ref)
    filename = str(config.hf_file or "").strip()
    root = Path(cache_root).expanduser() if cache_root else _hf_cache_root()
    repo_dir = root / _repo_cache_name(repo_id)
    configured_model_path = str(config.model_path or "").strip()
    if configured_model_path:
        model_path = Path(configured_model_path).expanduser()
        if model_path.exists():
            return _status_from_existing_model_path(
                config,
                model_path,
                cache_backend="model_path",
                cache_root=str(model_path.parent),
                repo_cache_dir=str(model_path.parent),
            )
        return LlamaCppModelCacheStatus(
            repo_id=repo_id,
            filename=filename or Path(configured_model_path).name,
            cache_root=str(model_path.parent),
            repo_cache_dir=str(model_path.parent),
            message="model_path_missing",
            cache_backend="model_path",
        )
    if cache_root is None and _model_artifact_cache_enabled(default=False):
        artifact_status = _model_artifact_status_from_manifest(config)
        if artifact_status is not None and artifact_status.complete:
            return artifact_status
        cid_status = _model_artifact_status_from_cid_alias(config, _model_cid_from_config(config))
        if cid_status is not None and cid_status.complete:
            return cid_status
    if not repo_id:
        return LlamaCppModelCacheStatus(
            repo_id=repo_id,
            filename=filename,
            cache_root=str(root),
            repo_cache_dir=str(repo_dir),
            message="missing_repo_id",
        )
    if not filename:
        return LlamaCppModelCacheStatus(
            repo_id=repo_id,
            filename=filename,
            cache_root=str(root),
            repo_cache_dir=str(repo_dir),
            message="missing_hf_file",
        )

    complete_path = ""
    complete_size = 0
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        for candidate in snapshots_dir.glob(f"*/{filename}"):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists():
                complete_path = str(candidate)
                try:
                    complete_size = int(resolved.stat().st_size)
                except Exception:
                    complete_size = 0
                break

    partials: list[str] = []
    partial_size = 0
    if repo_dir.exists():
        for pattern in ("*.downloadInProgress", "*.incomplete"):
            for candidate in repo_dir.rglob(pattern):
                partials.append(str(candidate))
                try:
                    partial_size += int(candidate.stat().st_size)
                except Exception:
                    pass

    complete = bool(complete_path and complete_size > 0)
    downloading = bool(partials) and not complete
    if complete:
        message = "complete"
    elif downloading:
        message = "download_in_progress"
    elif repo_dir.exists():
        message = "repo_cache_present_without_file"
    else:
        message = "not_downloaded"
    return LlamaCppModelCacheStatus(
        repo_id=repo_id,
        filename=filename,
        cache_root=str(root),
        repo_cache_dir=str(repo_dir),
        complete=complete,
        downloading=downloading,
        local_path=complete_path,
        local_size_bytes=complete_size,
        partial_paths=tuple(sorted(partials)),
        partial_size_bytes=partial_size,
        message=message,
    )


_GGUF_VALUE_TYPE_UINT8 = 0
_GGUF_VALUE_TYPE_INT8 = 1
_GGUF_VALUE_TYPE_UINT16 = 2
_GGUF_VALUE_TYPE_INT16 = 3
_GGUF_VALUE_TYPE_UINT32 = 4
_GGUF_VALUE_TYPE_INT32 = 5
_GGUF_VALUE_TYPE_FLOAT32 = 6
_GGUF_VALUE_TYPE_BOOL = 7
_GGUF_VALUE_TYPE_STRING = 8
_GGUF_VALUE_TYPE_ARRAY = 9
_GGUF_VALUE_TYPE_UINT64 = 10
_GGUF_VALUE_TYPE_INT64 = 11
_GGUF_VALUE_TYPE_FLOAT64 = 12
_GGUF_SCALAR_FORMATS = {
    _GGUF_VALUE_TYPE_UINT8: "<B",
    _GGUF_VALUE_TYPE_INT8: "<b",
    _GGUF_VALUE_TYPE_UINT16: "<H",
    _GGUF_VALUE_TYPE_INT16: "<h",
    _GGUF_VALUE_TYPE_UINT32: "<I",
    _GGUF_VALUE_TYPE_INT32: "<i",
    _GGUF_VALUE_TYPE_FLOAT32: "<f",
    _GGUF_VALUE_TYPE_BOOL: "<?",
    _GGUF_VALUE_TYPE_UINT64: "<Q",
    _GGUF_VALUE_TYPE_INT64: "<q",
    _GGUF_VALUE_TYPE_FLOAT64: "<d",
}
_GGUF_SCALAR_SIZES = {
    value_type: struct.calcsize(fmt) for value_type, fmt in _GGUF_SCALAR_FORMATS.items()
}
_GGUF_MAX_STORED_ARRAY_ITEMS = 64
_LAYER_TENSOR_PATTERNS = (
    re.compile(r"^blk\.(\d+)\."),
    re.compile(r"^layers\.(\d+)\."),
    re.compile(r"^model\.layers\.(\d+)\."),
    re.compile(r"\.layers\.(\d+)\."),
)
_MEMORY_UNIT_BYTES = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}
_LLAMA_DEVICE_RE = re.compile(
    r"^\s*(?P<identifier>[A-Za-z]+[0-9]+)\s*:\s*"
    r"(?P<name>.*?)\s*"
    r"\((?P<total>[0-9.]+)\s*(?P<total_unit>[KMGT]i?B|[KMGT]B|B),\s*"
    r"(?P<free>[0-9.]+)\s*(?P<free_unit>[KMGT]i?B|[KMGT]B|B)\s+free\)",
    re.IGNORECASE,
)
_AUTO_GPU_LAYER_WORDS = {"auto", "autosize", "auto-size", "safe", "stable", "analytical"}


class _GgufReader:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def tell(self) -> int:
        return int(self.file_obj.tell())

    def seek_forward(self, size: int) -> None:
        if int(size) > 0:
            self.file_obj.seek(int(size), os.SEEK_CUR)

    def read_exact(self, size: int) -> bytes:
        data = self.file_obj.read(int(size))
        if len(data) != int(size):
            raise ValueError("truncated_gguf")
        return data

    def read_u32(self) -> int:
        return int(struct.unpack("<I", self.read_exact(4))[0])

    def read_u64(self) -> int:
        return int(struct.unpack("<Q", self.read_exact(8))[0])

    def read_string(self) -> str:
        length = self.read_u64()
        return self.read_exact(length).decode("utf-8", errors="replace")


def _read_gguf_value(reader: _GgufReader, value_type: int) -> object:
    if value_type == _GGUF_VALUE_TYPE_STRING:
        return reader.read_string()
    if value_type == _GGUF_VALUE_TYPE_ARRAY:
        element_type = reader.read_u32()
        length = reader.read_u64()
        if length <= _GGUF_MAX_STORED_ARRAY_ITEMS:
            return tuple(_read_gguf_value(reader, element_type) for _ in range(length))
        _skip_gguf_values(reader, element_type, length)
        return {"array_type": element_type, "length": length}
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if not fmt:
        raise ValueError(f"unsupported_gguf_metadata_type:{value_type}")
    return struct.unpack(fmt, reader.read_exact(struct.calcsize(fmt)))[0]


def _skip_gguf_values(reader: _GgufReader, value_type: int, count: int) -> None:
    if count <= 0:
        return
    if value_type == _GGUF_VALUE_TYPE_STRING:
        for _ in range(count):
            reader.seek_forward(reader.read_u64())
        return
    if value_type == _GGUF_VALUE_TYPE_ARRAY:
        for _ in range(count):
            element_type = reader.read_u32()
            length = reader.read_u64()
            _skip_gguf_values(reader, element_type, length)
        return
    scalar_size = _GGUF_SCALAR_SIZES.get(value_type)
    if not scalar_size:
        raise ValueError(f"unsupported_gguf_array_type:{value_type}")
    reader.seek_forward(int(scalar_size) * int(count))


def _align_offset(offset: int, alignment: int) -> int:
    alignment = max(1, int(alignment or 1))
    return ((int(offset) + alignment - 1) // alignment) * alignment


def _metadata_int(metadata: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    for key, value in metadata.items():
        if not any(str(key).endswith(suffix) for suffix in names):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _tensor_layer_index(name: str) -> Optional[int]:
    for pattern in _LAYER_TENSOR_PATTERNS:
        match = pattern.search(str(name or ""))
        if match:
            return int(match.group(1))
    return None


def read_llama_cpp_gguf_sizing_info(path: object) -> LlamaCppGgufSizingInfo:
    """Read enough GGUF metadata to estimate layer placement memory.

    The parser intentionally reads only the GGUF header, metadata table, and
    tensor table.  Tensor payload bytes are not loaded into memory.
    """

    model_path = Path(path).expanduser()
    file_size = int(model_path.stat().st_size)
    with model_path.open("rb") as file_obj:
        reader = _GgufReader(file_obj)
        if reader.read_exact(4) != b"GGUF":
            raise ValueError("not_gguf")
        _version = reader.read_u32()
        tensor_count = reader.read_u64()
        metadata_count = reader.read_u64()
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = reader.read_string()
            value_type = reader.read_u32()
            metadata[key] = _read_gguf_value(reader, value_type)
        alignment = int(metadata.get("general.alignment") or 32)
        tensors: list[LlamaCppGgufTensorInfo] = []
        for _ in range(tensor_count):
            name = reader.read_string()
            dims_count = reader.read_u32()
            dimensions = tuple(int(reader.read_u64()) for _ in range(dims_count))
            ggml_type = reader.read_u32()
            offset = reader.read_u64()
            tensors.append(
                LlamaCppGgufTensorInfo(
                    name=name,
                    dimensions=dimensions,
                    ggml_type=int(ggml_type),
                    offset=int(offset),
                )
            )
        data_start = _align_offset(reader.tell(), alignment)

    tensor_payload_size = max(0, file_size - data_start)
    sorted_indices = sorted(range(len(tensors)), key=lambda index: (tensors[index].offset, index))
    tensor_sizes = [0] * len(tensors)
    for position, tensor_index in enumerate(sorted_indices):
        offset = max(0, int(tensors[tensor_index].offset))
        if position + 1 < len(sorted_indices):
            next_offset = max(0, int(tensors[sorted_indices[position + 1]].offset))
        else:
            next_offset = tensor_payload_size
        tensor_sizes[tensor_index] = max(0, next_offset - offset)
    sized_tensors = tuple(
        replace(tensor, size_bytes=int(tensor_sizes[index])) for index, tensor in enumerate(tensors)
    )

    layer_bytes_by_index: dict[int, int] = {}
    for tensor in sized_tensors:
        layer_index = _tensor_layer_index(tensor.name)
        if layer_index is None:
            continue
        layer_bytes_by_index[layer_index] = layer_bytes_by_index.get(layer_index, 0) + int(
            tensor.size_bytes
        )
    layer_count = _metadata_int(
        metadata,
        "llama.block_count",
        "mistral.block_count",
        "qwen2.block_count",
        "deepseek2.block_count",
        "block_count",
    )
    if not layer_count and layer_bytes_by_index:
        layer_count = max(layer_bytes_by_index) + 1
    repeating_layer_bytes = tuple(
        int(layer_bytes_by_index.get(index, 0)) for index in range(max(0, int(layer_count)))
    )
    nonzero_layer_bytes = tuple(size for size in repeating_layer_bytes if size > 0)
    total_layer_bytes = sum(repeating_layer_bytes)
    non_repeating_bytes = max(0, file_size - total_layer_bytes) if total_layer_bytes else file_size
    return LlamaCppGgufSizingInfo(
        path=str(model_path),
        file_size_bytes=file_size,
        alignment=alignment,
        tensor_count=int(tensor_count),
        metadata=metadata,
        tensors=sized_tensors,
        layer_count=int(layer_count),
        repeating_layer_bytes=repeating_layer_bytes,
        non_repeating_bytes=int(non_repeating_bytes),
        max_layer_bytes=max(nonzero_layer_bytes) if nonzero_layer_bytes else 0,
        avg_layer_bytes=(sum(nonzero_layer_bytes) // len(nonzero_layer_bytes))
        if nonzero_layer_bytes
        else 0,
    )


def _memory_to_bytes(value: object, unit: str) -> int:
    multiplier = _MEMORY_UNIT_BYTES.get(str(unit or "B").upper(), 1)
    return int(float(value) * multiplier)


def _parse_llama_cpp_device_list(output: str) -> tuple[LlamaCppDeviceInfo, ...]:
    devices: list[LlamaCppDeviceInfo] = []
    for raw_line in str(output or "").splitlines():
        match = _LLAMA_DEVICE_RE.search(raw_line)
        if not match:
            continue
        devices.append(
            LlamaCppDeviceInfo(
                identifier=match.group("identifier"),
                name=match.group("name").strip(),
                total_bytes=_memory_to_bytes(match.group("total"), match.group("total_unit")),
                free_bytes=_memory_to_bytes(match.group("free"), match.group("free_unit")),
                raw=raw_line.strip(),
            )
        )
    return tuple(devices)


def _nvidia_smi_devices(*, timeout_seconds: float = 10.0) -> tuple[LlamaCppDeviceInfo, ...]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ()
    try:
        code, stdout, _stderr = _run_command(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        return ()
    if code != 0:
        return ()
    devices: list[LlamaCppDeviceInfo] = []
    for index, line in enumerate(stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total_bytes = int(float(parts[1]) * 1024 * 1024)
            free_bytes = int(float(parts[2]) * 1024 * 1024)
        except Exception:
            continue
        devices.append(
            LlamaCppDeviceInfo(
                identifier=f"CUDA{index}",
                name=parts[0],
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                raw=line.strip(),
            )
        )
    return tuple(devices)


def llama_cpp_list_devices(
    *,
    executable: str = "",
    command_kind: str = "",
    timeout_seconds: float = 10.0,
) -> tuple[LlamaCppDeviceInfo, ...]:
    """Return device memory from llama.cpp, falling back to ``nvidia-smi``."""

    candidates: list[tuple[str, str]] = []
    if executable and command_kind != "llama-server":
        candidates.append((executable, command_kind or "llama"))
    for candidate, kind in _candidate_executables():
        if kind != "llama":
            continue
        if (candidate, kind) not in candidates:
            candidates.append((candidate, kind))
    for candidate, _kind in candidates:
        if not candidate:
            continue
        try:
            code, stdout, stderr = _run_command(
                [candidate, "cli", "--list-devices"],
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            continue
        if code != 0:
            continue
        devices = _parse_llama_cpp_device_list((stdout or "") + "\n" + (stderr or ""))
        if devices:
            return devices
    return _nvidia_smi_devices(timeout_seconds=timeout_seconds)


def _env_int(names: Sequence[str], default: int) -> int:
    raw = _coalesce_env(*names)
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(names: Sequence[str], default: float) -> float:
    raw = _coalesce_env(*names)
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _extra_args_have_flag(extra_args: Sequence[object], *flags: str) -> bool:
    values = [str(arg) for arg in extra_args]
    for value in values:
        if value in flags:
            return True
        if any(value.startswith(f"{flag}=") for flag in flags):
            return True
    return False


def _extra_arg_int(extra_args: Sequence[object], flags: Sequence[str], default: int) -> int:
    values = [str(arg) for arg in extra_args]
    for index, value in enumerate(values[:-1]):
        if value in flags:
            try:
                return int(values[index + 1])
            except Exception:
                return int(default)
    for value in values:
        for flag in flags:
            prefix = f"{flag}="
            if value.startswith(prefix):
                try:
                    return int(value[len(prefix) :])
                except Exception:
                    return int(default)
    return int(default)


def _append_extra_arg_if_missing(
    extra_args: Sequence[object],
    flags: Sequence[str],
    value: int,
    preferred_flag: str,
) -> tuple[str, ...]:
    existing = tuple(str(arg) for arg in extra_args if str(arg).strip())
    if int(value) <= 0 or _extra_args_have_flag(existing, *flags):
        return existing
    return existing + (preferred_flag, str(int(value)))


def _append_extra_arg_text_if_missing(
    extra_args: Sequence[object],
    flags: Sequence[str],
    value: object,
    preferred_flag: str,
) -> tuple[str, ...]:
    existing = tuple(str(arg) for arg in extra_args if str(arg).strip())
    text = str(value or "").strip()
    if not text or _extra_args_have_flag(existing, *flags):
        return existing
    return existing + (preferred_flag, text)


def _append_extra_arg_flag_if_missing(
    extra_args: Sequence[object],
    flags: Sequence[str],
    preferred_flag: str,
) -> tuple[str, ...]:
    existing = tuple(str(arg) for arg in extra_args if str(arg).strip())
    if _extra_args_have_flag(existing, *flags):
        return existing
    return existing + (preferred_flag,)


def _floor_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _estimate_kv_cache_bytes(
    metadata: Mapping[str, object],
    *,
    layer_count: int,
    context_size: int,
    extra_args: Sequence[object],
) -> int:
    context = max(0, int(context_size or 0))
    layers = max(
        0,
        int(layer_count)
        or _metadata_int(metadata, "llama.block_count", "mistral.block_count", "block_count"),
    )
    parallel = max(
        1,
        _extra_arg_int(
            extra_args,
            ("--parallel", "-np"),
            _env_int(
                (
                    "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_PARALLEL",
                    "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_PARALLEL",
                ),
                1,
            ),
        ),
    )
    embedding = _metadata_int(metadata, "llama.embedding_length", "embedding_length")
    head_count = _metadata_int(metadata, "llama.attention.head_count", "attention.head_count")
    kv_head_count = _metadata_int(
        metadata,
        "llama.attention.head_count_kv",
        "attention.head_count_kv",
    )
    key_length = _metadata_int(metadata, "llama.attention.key_length", "attention.key_length")
    value_length = _metadata_int(metadata, "llama.attention.value_length", "attention.value_length")
    if not kv_head_count:
        kv_head_count = head_count
    if not key_length and embedding and head_count:
        key_length = max(1, embedding // max(1, head_count))
    if not value_length:
        value_length = key_length
    if layers and context and kv_head_count and key_length and value_length:
        kv_bytes = layers * context * parallel * kv_head_count * (key_length + value_length) * 2
    else:
        kv_per_1k_mib = _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_KV_MIB_PER_1K_CTX",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_KV_MIB_PER_1K_CTX",
            ),
            1024,
        )
        kv_bytes = int(kv_per_1k_mib * 1024 * 1024 * max(1, context) / 1024)
    multiplier = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_KV_MULTIPLIER",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_KV_MULTIPLIER",
        ),
        1.25,
    )
    return max(0, int(kv_bytes * max(0.0, multiplier)))


def _estimate_batch_token_bytes(
    metadata: Mapping[str, object],
    *,
    layer_count: int,
    cuda13_large_nvfp4: bool,
    legacy_large_nvfp4: bool,
) -> int:
    configured_mib = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_BATCH_TOKEN_MIB",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_BATCH_TOKEN_MIB",
        ),
        0.0,
    )
    if configured_mib > 0:
        return max(1, int(configured_mib * 1024 * 1024))

    embedding = _metadata_int(metadata, "llama.embedding_length", "embedding_length")
    layers = max(1, int(layer_count or 0))
    bytes_per_value = 2
    activation_bytes = max(1, embedding) * layers * bytes_per_value
    if legacy_large_nvfp4:
        floor_mib = 32
    elif cuda13_large_nvfp4:
        floor_mib = 8
    else:
        floor_mib = 4
    return max(floor_mib * 1024 * 1024, int(activation_bytes * 4))


def _select_llama_cpp_device(devices: Sequence[LlamaCppDeviceInfo]) -> Optional[LlamaCppDeviceInfo]:
    candidates = [device for device in devices if int(device.free_bytes or 0) > 0]
    if not candidates:
        return None
    for device in candidates:
        if str(device.identifier).upper().startswith("CUDA"):
            return device
    return candidates[0]


def _auto_sizing_model_traits(config: LlamaCppServerConfig, sizing: LlamaCppGgufSizingInfo) -> dict[str, bool]:
    text = " ".join(
        [
            str(config.model_ref or ""),
            str(config.hf_file or ""),
            str(sizing.metadata.get("general.file_type") or ""),
            str(sizing.metadata.get("general.name") or ""),
        ]
    ).upper()
    return {
        "large": int(sizing.file_size_bytes or 0) >= 50 * 1024**3,
        "nvfp4": "NVFP4" in text,
    }


def _llama_cpp_cublas_major(executable: str) -> int:
    if not executable:
        return 0
    try:
        code, stdout, stderr = _run_command(["ldd", str(executable)], timeout_seconds=5.0)
    except Exception:
        return 0
    if code != 0:
        return 0
    majors = [
        int(match)
        for match in re.findall(
            r"libcublas(?:Lt)?\.so\.(\d+)",
            f"{stdout or ''}\n{stderr or ''}",
        )
    ]
    return max(majors) if majors else 0


def calculate_llama_cpp_auto_sizing(
    config: LlamaCppServerConfig,
    *,
    executable: str = "",
    command_kind: str = "",
    cache_status: Optional[LlamaCppModelCacheStatus] = None,
) -> LlamaCppAutoSizingPlan:
    """Calculate conservative ``-ngl``, ``-b``, and ``-ub`` values.

    This is an analytical guardrail, not a replacement for benchmarking.  It
    deliberately keeps a large reserve because llama.cpp can allocate CUDA
    workspaces and dequantization buffers that are not visible in GGUF tensor
    byte counts.
    """

    model_cache = cache_status or llama_cpp_model_cache_status(config)
    context_size = max(0, int(config.context_size or 0))
    if not model_cache.complete or not model_cache.local_path:
        batch = _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_BATCH_MAX",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_BATCH_MAX",
            ),
            64,
        )
        ubatch = min(
            batch,
            _env_int(
                (
                    "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_UBATCH_MAX",
                    "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_UBATCH_MAX",
                ),
                64,
            ),
        )
        extra_args: tuple[str, ...] = ()
        if not _extra_args_have_flag(config.extra_args, "-b", "--batch-size"):
            extra_args += ("-b", str(max(1, batch)))
        if not _extra_args_have_flag(config.extra_args, "-ub", "--ubatch-size"):
            extra_args += ("-ub", str(max(1, ubatch)))
        return LlamaCppAutoSizingPlan(
            enabled=True,
            reason=f"model_cache_{model_cache.message or 'unavailable'}_cpu_safe",
            gpu_layers=0,
            batch_size=max(1, batch),
            ubatch_size=max(1, ubatch),
            context_size=context_size,
            command_extra_args=extra_args,
        )

    try:
        sizing = read_llama_cpp_gguf_sizing_info(model_cache.local_path)
    except Exception as exc:
        return LlamaCppAutoSizingPlan(
            enabled=True,
            reason=f"gguf_sizing_failed:{type(exc).__name__}",
            gpu_layers=0,
            batch_size=64,
            ubatch_size=64,
            context_size=context_size,
            model_path=model_cache.local_path,
            command_extra_args=("-b", "64", "-ub", "64"),
        )
    devices = llama_cpp_list_devices(
        executable=executable,
        command_kind=command_kind,
    )
    device = _select_llama_cpp_device(devices)
    if device is None:
        return LlamaCppAutoSizingPlan(
            enabled=True,
            reason="no_cuda_device_cpu_safe",
            gpu_layers=0,
            batch_size=64,
            ubatch_size=64,
            context_size=context_size,
            model_path=sizing.path,
            command_extra_args=("-b", "64", "-ub", "64"),
        )

    traits = _auto_sizing_model_traits(config, sizing)
    cublas_major = _llama_cpp_cublas_major(executable)
    cuda13_large_nvfp4 = traits["large"] and traits["nvfp4"] and cublas_major >= 13
    legacy_large_nvfp4 = traits["large"] and traits["nvfp4"] and not cuda13_large_nvfp4
    reserve_mib_default = 8192 if cuda13_large_nvfp4 else 16384 if legacy_large_nvfp4 else 8192
    workspace_mib_default = 8192 if cuda13_large_nvfp4 else 16384 if legacy_large_nvfp4 else 4096
    reserve_fraction_default = 0.12 if cuda13_large_nvfp4 else 0.30 if legacy_large_nvfp4 else 0.15
    target_utilization_default = 0.85 if cuda13_large_nvfp4 else 0.60 if legacy_large_nvfp4 else 0.82
    layer_multiplier_default = 1.25 if cuda13_large_nvfp4 else 4.0 if legacy_large_nvfp4 else 1.50
    reserve_bytes = max(
        _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_RESERVE_MIB",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_RESERVE_MIB",
            ),
            reserve_mib_default,
        )
        * 1024
        * 1024,
        int(int(device.free_bytes or 0) * reserve_fraction_default),
    )
    reserve_fraction = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_RESERVE_FRACTION",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_RESERVE_FRACTION",
        ),
        reserve_fraction_default,
    )
    reserve_bytes = max(reserve_bytes, int(int(device.free_bytes or 0) * max(0.0, reserve_fraction)))
    workspace_bytes = _env_int(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_WORKSPACE_MIB",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_WORKSPACE_MIB",
        ),
        workspace_mib_default,
    ) * 1024 * 1024
    kv_bytes = _estimate_kv_cache_bytes(
        sizing.metadata,
        layer_count=sizing.layer_count,
        context_size=context_size,
        extra_args=config.extra_args,
    )
    target_utilization = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_TARGET_UTILIZATION",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_TARGET_UTILIZATION",
        ),
        target_utilization_default,
    )
    free_bytes = int(device.free_bytes or 0)
    usable_bytes = max(
        0,
        min(
            int(free_bytes * max(0.05, min(0.98, target_utilization))),
            free_bytes - reserve_bytes - workspace_bytes - kv_bytes,
        ),
    )
    non_layer_fraction = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_NON_LAYER_GPU_FRACTION",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_NON_LAYER_GPU_FRACTION",
        ),
        0.20,
    )
    estimated_non_layer_bytes = int(max(0, sizing.non_repeating_bytes) * max(0.0, non_layer_fraction))
    layer_multiplier = _env_float(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_LAYER_MULTIPLIER",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_LAYER_MULTIPLIER",
        ),
        layer_multiplier_default,
    )
    per_layer_bytes = max(int(sizing.max_layer_bytes or 0), int(sizing.avg_layer_bytes or 0))
    effective_layer_bytes = max(1, int(per_layer_bytes * max(0.1, layer_multiplier)))
    layer_budget = max(0, usable_bytes - estimated_non_layer_bytes)
    calculated_layers = min(max(0, int(sizing.layer_count or 0)), layer_budget // effective_layer_bytes)
    # The llama.app build previously bundled on GB10 reported free CUDA memory
    # correctly but failed in cublasCreate_v2 for this large NVFP4 DeepSeek2 MoE
    # model at three or more offloaded layers.  CUDA 13-linked llama.cpp builds
    # have been verified to handle full offload, so only cap the legacy path.
    default_max_gpu_layers = 2 if legacy_large_nvfp4 else int(sizing.layer_count or 0)
    max_gpu_layers = _env_int(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_MAX_GPU_LAYERS",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_MAX_GPU_LAYERS",
        ),
        default_max_gpu_layers,
    )
    min_gpu_layers = _env_int(
        (
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_MIN_GPU_LAYERS",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_MIN_GPU_LAYERS",
        ),
        0,
    )
    gpu_layers = min(max_gpu_layers, calculated_layers)
    if gpu_layers > 0:
        gpu_layers = max(min_gpu_layers, gpu_layers)
    estimated_layer_bytes = int(gpu_layers) * effective_layer_bytes
    remaining_bytes = max(0, usable_bytes - estimated_non_layer_bytes - estimated_layer_bytes)

    default_batch_max = 512 if cuda13_large_nvfp4 else 64 if legacy_large_nvfp4 else 256
    default_ubatch_max = 256 if cuda13_large_nvfp4 else 64 if legacy_large_nvfp4 else 128
    batch_max = max(
        1,
        _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_BATCH_MAX",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_BATCH_MAX",
            ),
            default_batch_max,
        ),
    )
    ubatch_max = max(
        1,
        _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_UBATCH_MAX",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_UBATCH_MAX",
            ),
            default_ubatch_max,
        ),
    )
    batch_token_bytes = _estimate_batch_token_bytes(
        sizing.metadata,
        layer_count=sizing.layer_count,
        cuda13_large_nvfp4=cuda13_large_nvfp4,
        legacy_large_nvfp4=legacy_large_nvfp4,
    )
    context_batch_cap = max(1, int(context_size or batch_max))
    memory_batch_cap = max(1, remaining_bytes // max(1, batch_token_bytes))
    batch_size = min(batch_max, context_batch_cap, _floor_power_of_two(memory_batch_cap))
    ubatch_size = min(ubatch_max, batch_size)
    extra_args = ()
    if not _extra_args_have_flag(config.extra_args, "--device"):
        extra_args += ("--device", str(device.identifier))
    if traits["large"] and traits["nvfp4"]:
        fit_target_mib = _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_FIT_TARGET_MIB",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_FIT_TARGET_MIB",
            ),
            8192,
        )
        if not _extra_args_have_flag(config.extra_args, "--fit", "-fit"):
            extra_args += ("--fit", "on")
        if fit_target_mib > 0 and not _extra_args_have_flag(
            config.extra_args,
            "--fit-target",
            "-fitt",
        ):
            extra_args += ("--fit-target", str(fit_target_mib))
        if _truthy(
            _coalesce_env(
                "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_NO_WARMUP",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_NO_WARMUP",
                default="1",
            ),
            default=True,
        ) and not _extra_args_have_flag(config.extra_args, "--warmup", "--no-warmup"):
            extra_args += ("--no-warmup",)
    if not _extra_args_have_flag(config.extra_args, "-b", "--batch-size"):
        extra_args += ("-b", str(int(batch_size)))
    if not _extra_args_have_flag(config.extra_args, "-ub", "--ubatch-size"):
        extra_args += ("-ub", str(int(ubatch_size)))
    return LlamaCppAutoSizingPlan(
        enabled=True,
        reason="calculated",
        gpu_layers=int(gpu_layers),
        batch_size=int(batch_size),
        ubatch_size=int(ubatch_size),
        context_size=context_size,
        device_free_bytes=free_bytes,
        device_total_bytes=int(device.total_bytes or 0),
        usable_device_bytes=int(usable_bytes),
        reserve_bytes=int(reserve_bytes),
        estimated_non_layer_bytes=int(estimated_non_layer_bytes),
        estimated_layer_bytes=int(estimated_layer_bytes),
        estimated_kv_bytes=int(kv_bytes),
        estimated_workspace_bytes=int(workspace_bytes),
        model_path=sizing.path,
        device_identifier=device.identifier,
        device_name=device.name,
        layer_count=int(sizing.layer_count or 0),
        layer_multiplier=float(layer_multiplier),
        command_extra_args=extra_args,
    )


def auto_size_llama_cpp_server_config(
    config: LlamaCppServerConfig,
    *,
    executable: str = "",
    command_kind: str = "",
    cache_status: Optional[LlamaCppModelCacheStatus] = None,
) -> tuple[LlamaCppServerConfig, LlamaCppAutoSizingPlan]:
    """Return a config with analytical safe CUDA and batch limits applied."""

    model_cache = cache_status or llama_cpp_model_cache_status(config)
    plan = calculate_llama_cpp_auto_sizing(
        config,
        executable=executable,
        command_kind=command_kind,
        cache_status=model_cache,
    )
    if not plan.enabled:
        return config, plan
    override_gpu_layers = _truthy(
        _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING_OVERRIDE_GPU_LAYERS",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING_OVERRIDE_GPU_LAYERS",
        ),
        default=False,
    )
    gpu_layers = config.gpu_layers
    if gpu_layers is None or override_gpu_layers:
        gpu_layers = plan.gpu_layers
    extra_args = tuple(str(arg) for arg in config.extra_args if str(arg).strip())
    extra_args = _append_extra_arg_text_if_missing(
        extra_args,
        ("--device",),
        plan.device_identifier,
        "--device",
    )
    if "--fit" in plan.command_extra_args:
        extra_args = _append_extra_arg_text_if_missing(
            extra_args,
            ("--fit", "-fit"),
            "on",
            "--fit",
        )
    if "--fit-target" in plan.command_extra_args:
        fit_target_index = plan.command_extra_args.index("--fit-target")
        fit_target = plan.command_extra_args[fit_target_index + 1]
        extra_args = _append_extra_arg_text_if_missing(
            extra_args,
            ("--fit-target", "-fitt"),
            fit_target,
            "--fit-target",
        )
    if "--no-warmup" in plan.command_extra_args:
        extra_args = _append_extra_arg_flag_if_missing(
            extra_args,
            ("--warmup", "--no-warmup"),
            "--no-warmup",
        )
    extra_args = _append_extra_arg_if_missing(
        extra_args,
        ("-b", "--batch-size"),
        plan.batch_size,
        "-b",
    )
    extra_args = _append_extra_arg_if_missing(
        extra_args,
        ("-ub", "--ubatch-size"),
        plan.ubatch_size,
        "-ub",
    )
    model_path = config.model_path
    if _serve_model_cache_via_local_path(model_cache):
        model_path = model_cache.local_path
    return replace(config, model_path=model_path, gpu_layers=gpu_layers, extra_args=extra_args), plan


def prefetch_llama_cpp_model(
    config: LlamaCppServerConfig,
    *,
    force_download: bool = False,
    local_files_only: bool = False,
    model_cache: Optional[bool] = None,
    model_cid: str = "",
    pin_model: bool = False,
    async_hash: Optional[bool] = None,
    track_model: bool = False,
) -> LlamaCppModelCacheStatus:
    """Materialize the configured GGUF and report local cache status.

    When model artifact caching is enabled, this keeps the serving path
    local-first: explicit local path/CAS, optional IPFS kit retrieval, then HF.
    """

    repo_id = _repo_id_from_model_ref(config.model_ref)
    filename = str(config.hf_file or "").strip()
    cid = str(model_cid or _model_cid_from_config(config)).strip()
    cache_enabled = (
        bool(model_cache)
        if model_cache is not None
        else _model_artifact_cache_enabled(default=bool(cid))
    )
    status = llama_cpp_model_cache_status(config)
    if status.complete:
        needs_cache_registration = status.cache_backend not in {
            "content_addressed_disk",
            "ipfs_kit",
        } or not str(status.content_cid_v1_path or "").strip()
        if cache_enabled and needs_cache_registration:
            if _should_hash_model_async(Path(status.local_path), async_hash=async_hash):
                return _register_llama_cpp_model_artifact_async(
                    config,
                    Path(status.local_path),
                    source_backend=status.cache_backend or "hf",
                    content_cid=status.content_cid,
                    pin=pin_model,
                    track_model=track_model,
                )
            return _register_llama_cpp_model_artifact(
                config,
                Path(status.local_path),
                source_backend=status.cache_backend or "hf",
                content_cid=status.content_cid,
                pin=pin_model,
            )
        return status

    if cache_enabled and cid:
        ipfs_status = _retrieve_llama_cpp_model_artifact_from_cid(
            config,
            cid,
            allow_remote=not bool(local_files_only),
            async_hash=async_hash,
            track_model=track_model,
        )
        if ipfs_status.complete:
            return ipfs_status

    if not repo_id or not filename:
        return status
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        return LlamaCppModelCacheStatus(
            **{
                **status.to_dict(),
                "message": f"huggingface_hub_unavailable:{type(exc).__name__}",
            }
        )

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            force_download=bool(force_download),
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:
        status = llama_cpp_model_cache_status(config)
        return LlamaCppModelCacheStatus(
            **{
                **status.to_dict(),
                "message": f"download_failed:{type(exc).__name__}:{str(exc)[:240]}",
            }
        )
    status = llama_cpp_model_cache_status(config)
    if cache_enabled and status.complete and status.local_path:
        if _should_hash_model_async(Path(status.local_path), async_hash=async_hash):
            return _register_llama_cpp_model_artifact_async(
                config,
                Path(status.local_path),
                source_backend=status.cache_backend or "hf",
                content_cid=status.content_cid,
                pin=pin_model,
                track_model=track_model,
            )
        return _register_llama_cpp_model_artifact(
            config,
            Path(status.local_path),
            source_backend=status.cache_backend or "hf",
            content_cid=status.content_cid,
            pin=pin_model,
        )
    return status


def llama_cpp_server_ready(base_url: str, *, timeout_seconds: float = 2.0) -> bool:
    """Check whether an OpenAI-compatible llama.cpp server is reachable."""

    url = str(base_url or "").rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
            return int(getattr(resp, "status", 200) or 200) < 500
    except Exception:
        return False


def _pidfile_path(config: LlamaCppServerConfig) -> Path:
    safe_port = str(int(config.port))
    return _default_cache_dir() / f"server-{safe_port}.json"


def _process_alive(pid: object) -> bool:
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _active_pidfile_payload(config: LlamaCppServerConfig) -> dict[str, object]:
    path = _pidfile_path(config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if not _process_alive(payload.get("pid")):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return {}
    return payload


def _touch_active_pidfile(
    config: LlamaCppServerConfig,
    active: Mapping[str, object],
    **updates: object,
) -> None:
    path = _pidfile_path(config)
    payload = dict(active)
    payload.update(updates)
    payload["last_accessed_at"] = time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _managed_pidfile_payloads() -> list[tuple[Path, dict[str, object]]]:
    servers: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(_default_cache_dir().glob("server-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not _process_alive(payload.get("pid")):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        servers.append((path, payload))
    return servers


def llama_cpp_warm_servers() -> tuple[dict[str, object], ...]:
    """Return active llama.cpp servers managed by this helper."""

    out: list[dict[str, object]] = []
    for path, payload in _managed_pidfile_payloads():
        item = dict(payload)
        item["pidfile"] = str(path)
        item["alive"] = True
        out.append(item)
    return tuple(out)


def _terminate_pidfile_process(path: Path, payload: Mapping[str, object]) -> bool:
    try:
        pid = int(payload.get("pid") or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    if _process_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            return False
        deadline = time.time() + 10.0
        while time.time() < deadline and _process_alive(pid):
            time.sleep(0.25)
        if _process_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                return False
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    return not _process_alive(pid)


def _device_free_bytes(identifier: str = "") -> int:
    devices = llama_cpp_list_devices()
    if not devices:
        return 0
    selected = None
    wanted = str(identifier or "").strip()
    if wanted:
        for device in devices:
            if str(device.identifier) == wanted:
                selected = device
                break
    if selected is None:
        selected = _select_llama_cpp_device(devices)
    return int(getattr(selected, "free_bytes", 0) or 0)


def evict_llama_cpp_warm_servers(
    *,
    exclude_base_urls: Sequence[str] = (),
    required_free_bytes: int = 0,
    device_identifier: str = "",
    max_to_evict: int = 0,
) -> dict[str, object]:
    """Evict least-recently-used managed llama.cpp servers until capacity is free."""

    excluded = {str(url or "").rstrip("/") for url in exclude_base_urls if str(url or "").strip()}
    servers = [
        (path, payload)
        for path, payload in _managed_pidfile_payloads()
        if str(payload.get("base_url") or "").rstrip("/") not in excluded
    ]
    servers.sort(
        key=lambda item: float(
            item[1].get("last_accessed_at") or item[1].get("started_at") or 0.0
        )
    )
    before_free = _device_free_bytes(device_identifier)
    evicted: list[dict[str, object]] = []
    required = max(0, int(required_free_bytes or 0))
    limit = max(0, int(max_to_evict or 0))
    for path, payload in servers:
        if required and _device_free_bytes(device_identifier) >= required:
            break
        if limit and len(evicted) >= limit:
            break
        if _terminate_pidfile_process(path, payload):
            evicted.append(
                {
                    "pid": int(payload.get("pid") or 0),
                    "base_url": str(payload.get("base_url") or ""),
                    "pidfile": str(path),
                }
            )
    return {
        "evicted": tuple(evicted),
        "before_free_bytes": before_free,
        "after_free_bytes": _device_free_bytes(device_identifier),
        "required_free_bytes": required,
    }


def _command_arg_value(command: Sequence[object], *flags: str) -> str:
    values = [str(arg) for arg in command]
    for index, value in enumerate(values[:-1]):
        if value in flags:
            return values[index + 1]
    for value in values:
        for flag in flags:
            prefix = f"{flag}="
            if value.startswith(prefix):
                return value[len(prefix) :]
    return ""


def _command_contains_subsequence(command: Sequence[object], expected: Sequence[object]) -> bool:
    values = [str(arg) for arg in command]
    needle = [str(arg) for arg in expected if str(arg).strip()]
    if not needle:
        return True
    if len(needle) > len(values):
        return False
    width = len(needle)
    return any(values[index : index + width] == needle for index in range(len(values) - width + 1))


def _active_pidfile_matches_config(config: LlamaCppServerConfig, active: Mapping[str, object]) -> bool:
    command = tuple(str(arg) for arg in active.get("command") or ())
    if not command:
        return True
    if str(active.get("base_url") or "").rstrip("/") != config.base_url.rstrip("/"):
        return False
    model_path = str(config.model_path or "").strip()
    if model_path:
        active_model_path = _command_arg_value(command, "-m", "--model", "--model-path")
        if active_model_path != model_path:
            return False
    else:
        if _command_arg_value(command, "-hf", "--hf", "--hf-repo") != str(config.model_ref):
            return False
        if str(config.hf_file or "").strip() and (
            _command_arg_value(command, "--hf-file") != str(config.hf_file).strip()
        ):
            return False
    if _command_arg_value(command, "--host") != str(config.host):
        return False
    if _command_arg_value(command, "--port") != str(int(config.port)):
        return False
    if int(config.context_size or 0) > 0:
        active_context = _command_arg_value(command, "-c", "--ctx-size", "--context-size")
        if active_context != str(int(config.context_size)):
            return False
    if config.gpu_layers is not None:
        active_gpu_layers = _command_arg_value(
            command,
            "-ngl",
            "--gpu-layers",
            "--n-gpu-layers",
        )
        if active_gpu_layers != str(int(config.gpu_layers)):
            return False
    if config.extra_args and not _command_contains_subsequence(command, config.extra_args):
        return False
    return True


def _restart_on_config_mismatch_enabled() -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_RESTART_ON_CONFIG_MISMATCH",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_RESTART_ON_CONFIG_MISMATCH",
        "ipfs_accelerate_py_LLAMA_CPP_RESTART_ON_CONFIG_MISMATCH",
    )
    return _truthy(configured, default=True)


def _evict_to_fit_enabled() -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_EVICT_TO_FIT",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_EVICT_TO_FIT",
        "ipfs_accelerate_py_LLAMA_CPP_EVICT_TO_FIT",
    )
    return _truthy(configured, default=True)


def _max_warm_servers() -> int:
    return max(
        0,
        _env_int(
            (
                "IPFS_ACCELERATE_LLAMA_CPP_MAX_WARM_SERVERS",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_MAX_WARM_SERVERS",
                "ipfs_accelerate_py_LLAMA_CPP_MAX_WARM_SERVERS",
            ),
            1,
        ),
    )


def _plan_required_free_bytes(plan: LlamaCppAutoSizingPlan) -> int:
    if not plan.enabled or int(plan.gpu_layers or 0) <= 0:
        return 0
    return int(
        max(0, plan.estimated_layer_bytes)
        + max(0, plan.estimated_non_layer_bytes)
        + max(0, plan.estimated_kv_bytes)
        + max(0, plan.estimated_workspace_bytes)
        + max(0, plan.reserve_bytes)
    )


def _model_manager_tracking_enabled(*, default: bool = False) -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_TRACK_MODEL",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_TRACK_MODEL",
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_MANAGER",
        "ipfs_accelerate_py_LLAMA_CPP_TRACK_MODEL",
    )
    if configured:
        return _truthy(configured, default=default)
    return bool(default)


def _llama_cpp_model_manager_storage_path() -> Path:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER_PATH",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_MANAGER_PATH",
        "IPFS_ACCELERATE_MODEL_MANAGER_PATH",
        "MODEL_MANAGER_JSON_PATH",
        "MODEL_MANAGER_DB_PATH",
    )
    if configured:
        return Path(configured).expanduser()
    return _default_cache_dir() / "model_manager.json"


def _llama_cpp_model_manager_use_database(path: Path) -> bool:
    configured = _coalesce_env(
        "IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER_USE_DATABASE",
        "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_MANAGER_USE_DATABASE",
    )
    if configured:
        return _truthy(configured, default=False)
    return path.suffix.lower() in {".db", ".duckdb"}


def _llama_cpp_model_manager_model_id(
    config: LlamaCppServerConfig,
    model_cache: LlamaCppModelCacheStatus,
) -> str:
    repo_id = _repo_id_from_model_ref(config.model_ref) or "local"
    filename = str(model_cache.filename or _model_cache_filename(config) or "").strip()
    if filename:
        return f"{repo_id}:{filename}"
    return repo_id


def _llama_cpp_launch_flag_args(config: LlamaCppServerConfig) -> dict[str, object]:
    model_path = str(config.model_path or "").strip()
    launch_args: dict[str, object] = {"--host": config.host, "--port": int(config.port)}
    if model_path:
        launch_args["-m"] = model_path
    else:
        launch_args["-hf"] = config.model_ref
        if str(config.hf_file or "").strip():
            launch_args["--hf-file"] = str(config.hf_file).strip()
    if int(config.context_size or 0) > 0:
        launch_args["-c"] = int(config.context_size)
    if int(config.threads or 0) > 0:
        launch_args["-t"] = int(config.threads)
    if config.gpu_layers is not None:
        launch_args["-ngl"] = int(config.gpu_layers)
    return launch_args


def _llama_cpp_model_architecture(model_cache: LlamaCppModelCacheStatus) -> tuple[str, dict[str, object]]:
    path = str(model_cache.local_path or "").strip()
    if not path:
        return "gguf", {}
    try:
        sizing = read_llama_cpp_gguf_sizing_info(path)
    except Exception:
        return "gguf", {}
    metadata = dict(sizing.metadata)
    architecture = str(
        metadata.get("general.architecture")
        or metadata.get("general.name")
        or "gguf"
    )
    return f"gguf:{architecture}", {
        "layer_count": sizing.layer_count,
        "tensor_count": sizing.tensor_count,
        "file_size_bytes": sizing.file_size_bytes,
        "non_repeating_bytes": sizing.non_repeating_bytes,
        "max_layer_bytes": sizing.max_layer_bytes,
        "avg_layer_bytes": sizing.avg_layer_bytes,
        "metadata": metadata,
    }


def _json_ready(value: object) -> object:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _register_llama_cpp_model_with_manager(
    config: LlamaCppServerConfig,
    model_cache: LlamaCppModelCacheStatus,
    *,
    auto_sizing: Optional[LlamaCppAutoSizingPlan] = None,
    command: Sequence[object] = (),
    command_kind: str = "",
    install: Optional[LlamaCppInstallResult] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Best-effort ModelManager registration for llama.cpp cache/runtime state."""

    if enabled is None:
        enabled = _model_manager_tracking_enabled(default=False)
    if not enabled:
        return False

    try:
        from ipfs_accelerate_py.model_manager import (
            DataType,
            IOSpec,
            ModelManager,
            ModelMetadata,
            ModelType,
            ServingConfig,
        )
    except Exception:
        try:
            from model_manager import (  # type: ignore
                DataType,
                IOSpec,
                ModelManager,
                ModelMetadata,
                ModelType,
                ServingConfig,
            )
        except Exception:
            return False

    storage_path = _llama_cpp_model_manager_storage_path()
    plan = auto_sizing or LlamaCppAutoSizingPlan(False, reason="not_requested")
    command_tuple = tuple(str(arg) for arg in command if str(arg).strip())
    cache_payload = model_cache.to_dict()
    plan_payload = plan.to_dict()
    config_payload = asdict(config)
    architecture, gguf_summary = _llama_cpp_model_architecture(model_cache)
    repo_id = _repo_id_from_model_ref(config.model_ref) or "local"
    filename = str(model_cache.filename or _model_cache_filename(config) or "").strip()
    model_id = _llama_cpp_model_manager_model_id(config, model_cache)
    model_name = Path(filename).stem if filename else Path(str(config.model_path or repo_id)).name
    local_path = str(model_cache.local_path or config.model_path or "").strip()
    model_cid = str(model_cache.content_cid or model_cache.content_cid_v1 or "").strip()
    source_url = f"https://huggingface.co/{repo_id}" if repo_id and repo_id != "local" else None
    hardware_affinity = ["cpu"]
    if int(config.gpu_layers or 0) > 0 or int(plan.gpu_layers or 0) > 0:
        hardware_affinity.insert(0, "cuda")

    serving_config = ServingConfig(
        engine="llama.cpp",
        launch_args=_llama_cpp_launch_flag_args(config),
        default_generation_params={
            "temperature": 0.0,
            "max_tokens": 512,
        },
        endpoint_schema={
            "type": "openai_chat_completions",
            "base_url": config.base_url,
        },
        hardware_affinity=hardware_affinity,
    ).to_dict()
    serving_config.update(
        {
            "resolved_command": list(command_tuple),
            "execution_args": {
                "command_kind": command_kind,
                "extra_args": list(config.extra_args),
                "base_url": config.base_url,
                "model_ref": config.model_ref,
                "hf_file": config.hf_file,
                "model_path": config.model_path,
                "model_cid": config.model_cid,
            },
            "cache": cache_payload,
            "analytical_capabilities": plan_payload,
            "install": install.to_dict() if install is not None else {},
            "pipeline_types": [
                "text-generation",
                "chat-completion",
                "legal-ir-guidance",
                "lean-proof-guidance",
            ],
        }
    )
    file_record = {
        "size": int(model_cache.local_size_bytes or 0),
        "local_path": local_path,
        "cache_backend": model_cache.cache_backend,
        "sha256": model_cache.content_sha256,
        "multihash_sha256": model_cache.content_multihash_sha256,
        "cid_v1_raw_sha256": model_cache.content_cid_v1,
        "cid_v1_path": model_cache.content_cid_v1_path,
        "ipfs_cid": model_cache.content_cid,
    }
    repository_structure = {
        "model_ref": config.model_ref,
        "repo_id": repo_id,
        "filename": filename,
        "cache_root": model_cache.cache_root,
        "content_addressed_path": model_cache.content_addressed_path,
        "files": {filename or "model": file_record},
        "total_files": 1 if filename or local_path else 0,
        "total_size": int(model_cache.local_size_bytes or 0),
    }
    if gguf_summary:
        repository_structure["gguf"] = gguf_summary

    metadata_kwargs = {
        "model_id": model_id,
        "model_name": model_name or model_id,
        "model_type": ModelType.LANGUAGE_MODEL,
        "architecture": architecture,
        "inputs": [
            IOSpec(
                name="messages",
                data_type=DataType.TEXT,
                dtype="json",
                description="OpenAI-compatible chat messages",
            ),
            IOSpec(
                name="prompt",
                data_type=DataType.TEXT,
                dtype="str",
                description="Single text prompt",
                optional=True,
            ),
        ],
        "outputs": [
            IOSpec(
                name="text",
                data_type=DataType.TEXT,
                dtype="str",
                description="Generated text",
            ),
            IOSpec(
                name="logits",
                data_type=DataType.LOGITS,
                dtype="float32",
                description="Model logits when exposed by backend",
                optional=True,
            ),
        ],
        "huggingface_config": {
            "model_ref": config.model_ref,
            "repo_id": repo_id,
            "filename": filename,
            "quantization": DEFAULT_LEANSTRAL_QUANTIZATION
            if repo_id == DEFAULT_LEANSTRAL_REPO_ID
            else "",
        },
        "inference_code_location": "ipfs_accelerate_py.utils.llama_cpp",
        "supported_backends": ["llama.cpp", "llama_cpp", "llm_router"],
        "hardware_requirements": {
            "cuda": "cuda" in hardware_affinity,
            "gpu_layers": config.gpu_layers,
            "context_size": config.context_size,
            "batch_size": plan.batch_size,
            "ubatch_size": plan.ubatch_size,
            "device_identifier": plan.device_identifier,
            "device_name": plan.device_name,
            "device_total_bytes": plan.device_total_bytes,
            "device_free_bytes_observed": plan.device_free_bytes,
            "required_free_bytes_estimate": _plan_required_free_bytes(plan),
        },
        "performance_metrics": {
            "analytical_sizing": plan_payload,
            "cache": cache_payload,
        },
        "tags": [
            "gguf",
            "llama.cpp",
            "llm_router",
            "content-addressed-cache",
        ],
        "source_url": source_url,
        "description": f"llama.cpp serving profile for {config.model_ref}",
        "repository_structure": _json_ready(repository_structure),
        "model_cid": model_cid or None,
        "artifact_cid": str(model_cache.content_cid or "").strip() or None,
        "model_revision": (
            model_cache.content_sha256
            or model_cache.content_cid
            or model_cache.content_cid_v1
            or None
        ),
        "serving_config": _json_ready(serving_config),
    }

    manager = None
    try:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        manager = ModelManager(
            storage_path=str(storage_path),
            use_database=_llama_cpp_model_manager_use_database(storage_path),
            enable_ipfs=_model_artifact_ipfs_enabled(),
        )
        existing = getattr(manager, "models", {}).get(model_id)
        if existing is not None:
            metadata_kwargs["created_at"] = getattr(existing, "created_at", None)
            metadata_kwargs["last_used_at"] = getattr(existing, "last_used_at", None)
            metadata_kwargs["last_inference_cid"] = getattr(existing, "last_inference_cid", None)
            metadata_kwargs["last_run_id"] = getattr(existing, "last_run_id", None)
            metadata_kwargs["inference_count"] = int(getattr(existing, "inference_count", 0) or 0)
            existing_serving_config = getattr(existing, "serving_config", None)
            if isinstance(existing_serving_config, dict) and not command_tuple and not plan.enabled:
                merged_serving_config = dict(existing_serving_config)
                merged_serving_config["cache"] = cache_payload
                execution_args = merged_serving_config.get("execution_args")
                if isinstance(execution_args, dict):
                    execution_args = dict(execution_args)
                    execution_args.update(
                        {
                            "model_ref": config.model_ref,
                            "hf_file": config.hf_file,
                            "model_path": config.model_path,
                            "model_cid": config.model_cid,
                        }
                    )
                    merged_serving_config["execution_args"] = execution_args
                metadata_kwargs["serving_config"] = _json_ready(merged_serving_config)
                existing_hardware = getattr(existing, "hardware_requirements", None)
                if isinstance(existing_hardware, dict):
                    metadata_kwargs["hardware_requirements"] = _json_ready(existing_hardware)
                existing_metrics = getattr(existing, "performance_metrics", None)
                if isinstance(existing_metrics, dict):
                    merged_metrics = dict(existing_metrics)
                    merged_metrics["cache"] = cache_payload
                    metadata_kwargs["performance_metrics"] = _json_ready(merged_metrics)
        return bool(manager.add_model(ModelMetadata(**metadata_kwargs)))
    except Exception:
        return False
    finally:
        if manager is not None:
            try:
                manager.close()
            except Exception:
                pass


def _terminate_active_pidfile_process(config: LlamaCppServerConfig, active: Mapping[str, object]) -> bool:
    try:
        pid = int(active.get("pid") or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    if not _process_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        return False
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _process_alive(pid):
            break
        time.sleep(0.25)
    if _process_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            return False
    try:
        _pidfile_path(config).unlink(missing_ok=True)
    except Exception:
        pass
    return not _process_alive(pid)


def _log_path(config: LlamaCppServerConfig) -> Path:
    base = Path(config.log_dir).expanduser() if config.log_dir else _default_cache_dir() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"llama-cpp-{int(config.port)}.log"


def ensure_llama_cpp_server(
    config: LlamaCppServerConfig,
    *,
    autostart: bool = False,
    auto_install: bool = False,
    auto_update: bool = False,
    prefetch_model: bool = False,
    prefetch_force_download: bool = False,
    prefetch_local_files_only: bool = False,
    prefetch_model_cache: Optional[bool] = None,
    prefetch_model_cid: str = "",
    prefetch_pin_model: bool = False,
    prefetch_async_hash: Optional[bool] = None,
    startup_timeout_seconds: float = 60.0,
    install_timeout_seconds: float = 900.0,
    explicit_binary: str = "",
    track_model: Optional[bool] = None,
) -> LlamaCppServerResult:
    """Ensure a local OpenAI-compatible llama.cpp server is reachable."""

    auto_sizing = LlamaCppAutoSizingPlan(False, reason="not_requested")
    track_model_enabled = (
        _model_manager_tracking_enabled(default=False)
        if track_model is None
        else bool(track_model)
    )
    if bool(config.auto_sizing):
        sizing_executable, sizing_command_kind = find_llama_cpp_executable(explicit_binary)
        config, auto_sizing = auto_size_llama_cpp_server_config(
            config,
            executable=sizing_executable,
            command_kind=sizing_command_kind,
        )
    initial_model_cache = llama_cpp_model_cache_status(config)
    if _serve_model_cache_via_local_path(initial_model_cache):
        config = replace(config, model_path=initial_model_cache.local_path)
    base_url = config.base_url
    active = _active_pidfile_payload(config)
    if active and not _active_pidfile_matches_config(config, active):
        if _restart_on_config_mismatch_enabled():
            if not _terminate_active_pidfile_process(config, active):
                return LlamaCppServerResult(
                    running=False,
                    started=False,
                    pid=int(active.get("pid") or 0),
                    base_url=base_url,
                    command=tuple(str(arg) for arg in active.get("command") or ()),
                    model_cache=llama_cpp_model_cache_status(config),
                    auto_sizing=auto_sizing,
                    message="server_config_mismatch_restart_failed",
                )
            active = {}
        elif llama_cpp_server_ready(base_url):
            _touch_active_pidfile(config, active)
            active_model_cache = llama_cpp_model_cache_status(config)
            model_manager_registered = _register_llama_cpp_model_with_manager(
                config,
                active_model_cache,
                auto_sizing=auto_sizing,
                command=tuple(str(arg) for arg in active.get("command") or ()),
                enabled=track_model_enabled,
            )
            return LlamaCppServerResult(
                running=True,
                started=False,
                pid=int(active.get("pid") or 0),
                base_url=base_url,
                command=tuple(str(arg) for arg in active.get("command") or ()),
                model_cache=active_model_cache,
                auto_sizing=auto_sizing,
                model_manager_registered=model_manager_registered,
                message="already_running_config_mismatch",
            )
    if llama_cpp_server_ready(base_url):
        if active:
            _touch_active_pidfile(config, active)
        active_model_cache = llama_cpp_model_cache_status(config)
        model_manager_registered = _register_llama_cpp_model_with_manager(
            config,
            active_model_cache,
            auto_sizing=auto_sizing,
            command=tuple(str(arg) for arg in active.get("command") or ()),
            enabled=track_model_enabled,
        )
        return LlamaCppServerResult(
            running=True,
            base_url=base_url,
            model_cache=active_model_cache,
            auto_sizing=auto_sizing,
            model_manager_registered=model_manager_registered,
            message="already_running",
        )

    install = ensure_llama_cpp(
        auto_install=auto_install,
        auto_update=auto_update,
        explicit_binary=explicit_binary,
        timeout_seconds=install_timeout_seconds,
    )
    if not install.available:
        return LlamaCppServerResult(
            running=False,
            base_url=base_url,
            install=install,
            auto_sizing=auto_sizing,
            message=install.message or "llama.cpp executable unavailable",
        )

    model_cache = llama_cpp_model_cache_status(config)
    if _serve_model_cache_via_local_path(model_cache):
        config = replace(config, model_path=model_cache.local_path)
        base_url = config.base_url
    if bool(config.auto_sizing):
        config, auto_sizing = auto_size_llama_cpp_server_config(
            replace(config, gpu_layers=None),
            executable=install.executable,
            command_kind=install.command_kind,
            cache_status=model_cache,
        )
        base_url = config.base_url
        active = _active_pidfile_payload(config)
        if active and not _active_pidfile_matches_config(config, active):
            if _restart_on_config_mismatch_enabled():
                if not _terminate_active_pidfile_process(config, active):
                    return LlamaCppServerResult(
                        running=False,
                        started=False,
                        pid=int(active.get("pid") or 0),
                        base_url=base_url,
                        command=tuple(str(arg) for arg in active.get("command") or ()),
                        install=install,
                        model_cache=model_cache,
                        auto_sizing=auto_sizing,
                        message="server_post_install_config_mismatch_restart_failed",
                    )
                active = {}
            elif llama_cpp_server_ready(base_url):
                _touch_active_pidfile(config, active)
                return LlamaCppServerResult(
                    running=True,
                    started=False,
                    pid=int(active.get("pid") or 0),
                    base_url=base_url,
                    command=tuple(str(arg) for arg in active.get("command") or ()),
                    model_cache=model_cache,
                    auto_sizing=auto_sizing,
                    message="already_running_post_install_config_mismatch",
                )
    if prefetch_model:
        model_cache = prefetch_llama_cpp_model(
            config,
            force_download=prefetch_force_download,
            local_files_only=prefetch_local_files_only,
            model_cache=prefetch_model_cache,
            model_cid=prefetch_model_cid,
            pin_model=prefetch_pin_model,
            async_hash=prefetch_async_hash,
            track_model=track_model_enabled,
        )
        if not model_cache.complete:
            command = build_llama_cpp_server_command(
                config,
                executable=install.executable,
                command_kind=install.command_kind,
            )
            return LlamaCppServerResult(
                running=False,
                base_url=base_url,
                command=command,
                install=install,
                model_cache=model_cache,
                auto_sizing=auto_sizing,
                message=f"model_prefetch_{model_cache.message}",
            )
        if _serve_model_cache_via_local_path(model_cache):
            config = replace(config, model_path=model_cache.local_path)
            base_url = config.base_url
        if bool(config.auto_sizing) and auto_sizing.reason.endswith("_cpu_safe"):
            config, auto_sizing = auto_size_llama_cpp_server_config(
                replace(config, gpu_layers=None),
                executable=install.executable,
                command_kind=install.command_kind,
                cache_status=model_cache,
            )
            base_url = config.base_url
    if not autostart:
        command = build_llama_cpp_server_command(
            config,
            executable=install.executable,
            command_kind=install.command_kind,
        )
        model_manager_registered = _register_llama_cpp_model_with_manager(
            config,
            model_cache,
            auto_sizing=auto_sizing,
            command=command,
            command_kind=install.command_kind,
            install=install,
            enabled=track_model_enabled,
        )
        return LlamaCppServerResult(
            running=False,
            base_url=base_url,
            command=command,
            install=install,
            model_cache=model_cache,
            auto_sizing=auto_sizing,
            model_manager_registered=model_manager_registered,
            message="server_not_running_autostart_disabled",
        )

    active = active or _active_pidfile_payload(config)
    if active:
        pid = int(active.get("pid") or 0)
        command = tuple(str(arg) for arg in active.get("command") or ())
        log_path_text = str(active.get("log_path") or _log_path(config))
        deadline = time.time() + max(0.0, float(startup_timeout_seconds))
        while time.time() < deadline and _process_alive(pid):
            if llama_cpp_server_ready(base_url, timeout_seconds=2.0):
                _touch_active_pidfile(config, active)
                return LlamaCppServerResult(
                    running=True,
                    started=False,
                    pid=pid,
                    base_url=base_url,
                    command=command,
                    log_path=log_path_text,
                    install=install,
                    model_cache=model_cache,
                    auto_sizing=auto_sizing,
                    message="existing_startup_finished",
                )
            time.sleep(1.0)
        if _process_alive(pid):
            return LlamaCppServerResult(
                running=False,
                started=False,
                pid=pid,
                base_url=base_url,
                command=command,
                log_path=log_path_text,
                install=install,
                model_cache=model_cache,
                auto_sizing=auto_sizing,
                message="server_existing_startup_timeout",
            )

    if _evict_to_fit_enabled():
        warm_servers = [
            payload
            for _path, payload in _managed_pidfile_payloads()
            if str(payload.get("base_url") or "").rstrip("/") != base_url.rstrip("/")
        ]
        allowed_existing = max(0, _max_warm_servers() - 1)
        excess_to_evict = max(0, len(warm_servers) - allowed_existing)
        required_free = _plan_required_free_bytes(auto_sizing)
        if excess_to_evict or (required_free and _device_free_bytes(auto_sizing.device_identifier) < required_free):
            evict_llama_cpp_warm_servers(
                exclude_base_urls=(base_url,),
                required_free_bytes=required_free,
                device_identifier=auto_sizing.device_identifier,
                max_to_evict=0 if required_free else excess_to_evict,
            )
            if bool(config.auto_sizing):
                config, auto_sizing = auto_size_llama_cpp_server_config(
                    replace(config, gpu_layers=None),
                    executable=install.executable,
                    command_kind=install.command_kind,
                    cache_status=model_cache,
                )
                base_url = config.base_url

    log_path = _log_path(config)
    command = build_llama_cpp_server_command(
        config,
        executable=install.executable,
        command_kind=install.command_kind,
    )
    model_manager_registered = _register_llama_cpp_model_with_manager(
        config,
        model_cache,
        auto_sizing=auto_sizing,
        command=command,
        command_kind=install.command_kind,
        install=install,
        enabled=track_model_enabled,
    )
    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        list(command),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    log_handle.close()

    pidfile = _pidfile_path(config)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "base_url": base_url,
                "command": list(command),
                "log_path": str(log_path),
                "model_ref": config.model_ref,
                "hf_file": config.hf_file,
                "model_path": config.model_path,
                "port": int(config.port),
                "started_at": time.time(),
                "last_accessed_at": time.time(),
                "auto_sizing": auto_sizing.to_dict(),
                "model_cache": model_cache.to_dict(),
                "model_manager_registered": model_manager_registered,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    deadline = time.time() + max(0.0, float(startup_timeout_seconds))
    while time.time() < deadline:
        if proc.poll() is not None:
            return LlamaCppServerResult(
                running=False,
                started=True,
                pid=proc.pid,
                base_url=base_url,
                command=command,
                log_path=str(log_path),
                install=install,
                model_cache=model_cache,
                auto_sizing=auto_sizing,
                model_manager_registered=model_manager_registered,
                message=f"server_exited_{proc.returncode}",
            )
        if llama_cpp_server_ready(base_url, timeout_seconds=2.0):
            _touch_active_pidfile(
                config,
                {
                    "pid": proc.pid,
                    "base_url": base_url,
                    "command": list(command),
                    "log_path": str(log_path),
                    "model_ref": config.model_ref,
                    "hf_file": config.hf_file,
                    "model_path": config.model_path,
                    "port": int(config.port),
                    "started_at": time.time(),
                    "auto_sizing": auto_sizing.to_dict(),
                    "model_cache": model_cache.to_dict(),
                    "model_manager_registered": model_manager_registered,
                },
            )
            return LlamaCppServerResult(
                running=True,
                started=True,
                pid=proc.pid,
                base_url=base_url,
                command=command,
                log_path=str(log_path),
                install=install,
                model_cache=model_cache,
                auto_sizing=auto_sizing,
                model_manager_registered=model_manager_registered,
                message="started",
            )
        time.sleep(1.0)

    return LlamaCppServerResult(
        running=False,
        started=True,
        pid=proc.pid,
        base_url=base_url,
        command=command,
        log_path=str(log_path),
        install=install,
        model_cache=model_cache,
        auto_sizing=auto_sizing,
        model_manager_registered=model_manager_registered,
        message="server_startup_timeout",
    )


def config_from_env(**overrides: object) -> LlamaCppServerConfig:
    """Build server config from ``IPFS_ACCELERATE_LLAMA_CPP_*`` env vars."""

    model_ref = str(
        overrides.get("model_ref")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_REF",
            "ipfs_accelerate_py_LLAMA_CPP_MODEL_REF",
            default=DEFAULT_LEANSTRAL_MODEL_REF,
        )
    ).strip()
    hf_file_override = overrides.get("hf_file")
    hf_file = str(
        hf_file_override
        if hf_file_override is not None
        else _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_HF_FILE",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_HF_FILE",
            "ipfs_accelerate_py_LLAMA_CPP_HF_FILE",
        )
    ).strip()
    if not hf_file and model_ref.split(":", 1)[0] == DEFAULT_LEANSTRAL_REPO_ID:
        hf_file = DEFAULT_LEANSTRAL_FILENAME
    model_path = str(
        overrides.get("model_path")
        if overrides.get("model_path") is not None
        else _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_PATH",
            "ipfs_accelerate_py_LLAMA_CPP_MODEL_PATH",
        )
    ).strip()
    model_cid = str(
        overrides.get("model_cid")
        if overrides.get("model_cid") is not None
        else _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_MODEL_CID",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_MODEL_CID",
            "ipfs_accelerate_py_LLAMA_CPP_MODEL_CID",
        )
    ).strip()
    host = str(
        overrides.get("host")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_HOST",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_HOST",
            "ipfs_accelerate_py_LLAMA_CPP_HOST",
            default=DEFAULT_LLAMA_CPP_HOST,
        )
    ).strip()
    port = int(
        overrides.get("port")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_PORT",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_PORT",
            "ipfs_accelerate_py_LLAMA_CPP_PORT",
            default=str(DEFAULT_LLAMA_CPP_PORT),
        )
    )
    context_size = int(
        overrides.get("context_size")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_CONTEXT_SIZE",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_CONTEXT_SIZE",
            "ipfs_accelerate_py_LLAMA_CPP_CONTEXT_SIZE",
            default=str(DEFAULT_LLAMA_CPP_CONTEXT_SIZE),
        )
    )
    threads = int(
        overrides.get("threads")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_THREADS",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_THREADS",
            "ipfs_accelerate_py_LLAMA_CPP_THREADS",
            default="0",
        )
    )
    raw_gpu_layers = overrides.get("gpu_layers")
    if raw_gpu_layers is None:
        raw_gpu_layers = _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_GPU_LAYERS",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_GPU_LAYERS",
            "ipfs_accelerate_py_LLAMA_CPP_GPU_LAYERS",
        )
    raw_auto_sizing = overrides.get("auto_sizing")
    if raw_auto_sizing is None:
        raw_auto_sizing = _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_AUTO_SIZING",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_AUTO_SIZING",
            "ipfs_accelerate_py_LLAMA_CPP_AUTO_SIZING",
        )
    auto_sizing = _truthy(raw_auto_sizing, default=False)
    raw_gpu_layers_text = str(raw_gpu_layers or "").strip()
    if raw_gpu_layers_text.lower() in _AUTO_GPU_LAYER_WORDS:
        gpu_layers = None
        auto_sizing = True
    else:
        gpu_layers = int(raw_gpu_layers_text) if raw_gpu_layers_text else None
    raw_extra_value = overrides.get("extra_args")
    if isinstance(raw_extra_value, (list, tuple)):
        extra_args = tuple(str(arg) for arg in raw_extra_value if str(arg).strip())
    else:
        raw_extra = str(
            raw_extra_value
            or _coalesce_env(
                "IPFS_ACCELERATE_LLAMA_CPP_EXTRA_ARGS",
                "IPFS_ACCELERATE_PY_LLAMA_CPP_EXTRA_ARGS",
                "ipfs_accelerate_py_LLAMA_CPP_EXTRA_ARGS",
            )
            or ""
        )
        extra_args = tuple(shlex.split(raw_extra)) if raw_extra.strip() else ()
    log_dir = str(
        overrides.get("log_dir")
        or _coalesce_env(
            "IPFS_ACCELERATE_LLAMA_CPP_LOG_DIR",
            "IPFS_ACCELERATE_PY_LLAMA_CPP_LOG_DIR",
            "ipfs_accelerate_py_LLAMA_CPP_LOG_DIR",
        )
        or ""
    )
    return LlamaCppServerConfig(
        model_ref=model_ref,
        hf_file=hf_file,
        model_path=model_path,
        model_cid=model_cid,
        host=host,
        port=port,
        context_size=context_size,
        threads=threads,
        gpu_layers=gpu_layers,
        extra_args=extra_args,
        log_dir=log_dir,
        auto_sizing=auto_sizing,
    )


class llama_cpp_utils:
    """Backward-compatible wrapper used by older skillset code."""

    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}

    def init(self):
        return ensure_llama_cpp()

    def ensure(self, **kwargs):
        return ensure_llama_cpp(**kwargs)

    def serve(self, **kwargs):
        config = config_from_env(**{k: v for k, v in kwargs.items() if k in {"model_ref", "hf_file", "model_path", "model_cid", "host", "port", "context_size", "threads", "gpu_layers", "extra_args", "log_dir", "auto_sizing"}})
        return ensure_llama_cpp_server(
            config,
            autostart=bool(kwargs.get("autostart", True)),
            auto_install=bool(kwargs.get("auto_install", False)),
            auto_update=bool(kwargs.get("auto_update", False)),
            prefetch_model=bool(kwargs.get("prefetch_model", False)),
            prefetch_force_download=bool(kwargs.get("prefetch_force_download", False)),
            prefetch_local_files_only=bool(kwargs.get("prefetch_local_files_only", False)),
            prefetch_model_cache=kwargs.get("prefetch_model_cache"),
            prefetch_model_cid=str(kwargs.get("prefetch_model_cid") or ""),
            prefetch_pin_model=bool(kwargs.get("prefetch_pin_model", False)),
            prefetch_async_hash=kwargs.get("prefetch_async_hash"),
            track_model=kwargs.get("track_model"),
        )

    def __test__(self):
        return ensure_llama_cpp().to_dict()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check or start a local llama.cpp OpenAI-compatible server.")
    parser.add_argument("--serve", action="store_true", help="Start the server when it is not already reachable.")
    parser.add_argument("--probe", action="store_true", help="Only probe the configured server.")
    parser.add_argument("--model-status", action="store_true", help="Report local Hugging Face GGUF cache status.")
    parser.add_argument("--prefetch-model", action="store_true", help="Download the configured GGUF before serving.")
    parser.add_argument("--install-llama-cpp", action="store_true", help="Build/install managed upstream llama.cpp.")
    parser.add_argument("--warm-servers", action="store_true", help="List managed warm llama.cpp servers.")
    parser.add_argument("--evict-warm-servers", action="store_true", help="Evict managed warm llama.cpp servers.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download when prefetching.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not use the network when prefetching.")
    parser.add_argument("--auto-install", action="store_true", help="Run the configured installer if llama.cpp is missing.")
    parser.add_argument("--auto-update", action="store_true", help="Run the configured updater/installer before serving.")
    parser.add_argument("--model-ref", default="", help="HF model ref accepted by llama.cpp, e.g. repo:quant.")
    parser.add_argument("--hf-file", default="", help="Exact Hugging Face GGUF file passed as --hf-file.")
    parser.add_argument("--model-path", default="", help="Local GGUF path to serve with -m.")
    parser.add_argument("--model-cid", default="", help="Optional IPFS/IPFS Kit CID used to populate the local model cache.")
    parser.add_argument("--model-cache", action="store_true", help="Materialize GGUFs into local content-addressed cache.")
    parser.add_argument("--model-cache-backend", default="", help="Model cache backend: local, auto, or ipfs_kit.")
    parser.add_argument("--async-model-hash", dest="async_model_hash", action="store_true", default=None, help="Hash large downloaded GGUFs in a detached background finalizer.")
    parser.add_argument("--sync-model-hash", dest="async_model_hash", action="store_false", help="Hash downloaded GGUFs before returning from prefetch/startup.")
    parser.add_argument("--async-model-hash-min-bytes", type=int, default=-1, help="Minimum model size for async hashing; defaults to 1 GiB.")
    parser.add_argument("--track-model", dest="track_model", action="store_true", default=True, help="Record cache and launch metadata in ModelManager.")
    parser.add_argument("--no-track-model", dest="track_model", action="store_false", help="Do not update ModelManager metadata.")
    parser.add_argument("--model-manager-path", default="", help="ModelManager JSON/DuckDB path for llama.cpp model records.")
    parser.add_argument("--pin-model", action="store_true", help="Request pinning when the model cache uses IPFS Kit.")
    parser.add_argument("--finalize-model-artifact", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-source-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-source-backend", default="hf", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-content-cid", default="", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-job-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-ipfs-remote-attempted", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-ipfs-remote-loaded", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="", help="Server host.")
    parser.add_argument("--port", type=int, default=0, help="Server port.")
    parser.add_argument("--context-size", type=int, default=0, help="Context window passed as -c.")
    parser.add_argument("--threads", type=int, default=0, help="Thread count passed as -t.")
    parser.add_argument("--gpu-layers", default=None, help="GPU layer count passed as -ngl; use 'auto' for analytical safe sizing.")
    parser.add_argument("--extra-args", default="", help="Additional llama.cpp server args.")
    parser.add_argument("--log-dir", default="", help="Directory for server logs.")
    parser.add_argument("--auto-size", action="store_true", help="Apply analytical safe -ngl, -b, and -ub sizing.")
    parser.add_argument("--auto-size-plan", action="store_true", help="Print the analytical sizing plan and command, then exit.")
    parser.add_argument("--required-free-mib", type=int, default=0, help="Free-device-memory target for warm eviction.")
    parser.add_argument("--startup-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.install_llama_cpp:
        payload = _install_llama_cpp_from_source(
            auto_update=bool(args.auto_update),
            timeout_seconds=1800.0,
        ).to_dict()
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("available") else 1
    if args.warm_servers:
        print(json.dumps({"servers": llama_cpp_warm_servers()}, sort_keys=True))
        return 0
    if args.evict_warm_servers:
        payload = evict_llama_cpp_warm_servers(
            required_free_bytes=max(0, int(args.required_free_mib or 0)) * 1024 * 1024,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.model_cache:
        os.environ["IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE"] = "1"
    if args.model_cache_backend:
        os.environ["IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_BACKEND"] = str(args.model_cache_backend)
    if args.model_cid:
        os.environ["IPFS_ACCELERATE_LLAMA_CPP_MODEL_CID"] = str(args.model_cid)
    if args.model_manager_path:
        os.environ["IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER_PATH"] = str(args.model_manager_path)
    if int(args.async_model_hash_min_bytes or -1) >= 0:
        os.environ["IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH_MIN_BYTES"] = str(
            int(args.async_model_hash_min_bytes)
        )

    overrides = {
        "model_ref": args.model_ref or None,
        "hf_file": args.hf_file or None,
        "model_path": args.model_path or None,
        "model_cid": args.model_cid or None,
        "host": args.host or None,
        "port": args.port or None,
        "context_size": args.context_size or None,
        "threads": args.threads or None,
        "gpu_layers": args.gpu_layers,
        "extra_args": args.extra_args or None,
        "log_dir": args.log_dir or None,
        "auto_sizing": bool(args.auto_size) or None,
    }
    config = config_from_env(**{k: v for k, v in overrides.items() if v is not None})
    if args.finalize_model_artifact:
        source_path = Path(str(args.finalize_source_path or "")).expanduser()
        if not source_path.exists():
            print(
                json.dumps(
                    {
                        "complete": False,
                        "message": "finalize_source_missing",
                        "source_path": str(source_path),
                    },
                    sort_keys=True,
                )
            )
            return 1
        try:
            status = _finalize_llama_cpp_model_artifact_hash(
                config,
                source_path,
                source_backend=str(args.finalize_source_backend or "hf"),
                content_cid=str(args.finalize_content_cid or ""),
                pin=bool(args.pin_model),
                ipfs_remote_attempted=bool(args.finalize_ipfs_remote_attempted),
                ipfs_remote_loaded=bool(args.finalize_ipfs_remote_loaded),
                track_model=bool(args.track_model),
                job_path=Path(str(args.finalize_job_path or "")).expanduser()
                if str(args.finalize_job_path or "").strip()
                else Path(),
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "complete": False,
                        "message": f"finalize_failed:{type(exc).__name__}:{str(exc)[:240]}",
                        "source_path": str(source_path),
                    },
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(status.to_dict(), sort_keys=True))
        return 0 if status.complete else 1
    if args.auto_size_plan:
        executable, command_kind = find_llama_cpp_executable()
        model_cache = llama_cpp_model_cache_status(replace(config, auto_sizing=True))
        sized_config, plan = auto_size_llama_cpp_server_config(
            replace(config, auto_sizing=True),
            executable=executable,
            command_kind=command_kind,
            cache_status=model_cache,
        )
        command = (
            build_llama_cpp_server_command(
                sized_config,
                executable=executable,
                command_kind=command_kind,
            )
            if executable
            else ()
        )
        model_manager_registered = _register_llama_cpp_model_with_manager(
            sized_config,
            model_cache,
            auto_sizing=plan,
            command=command,
            command_kind=command_kind,
            enabled=bool(args.track_model),
        )
        print(
            json.dumps(
                {
                    "config": asdict(sized_config),
                    "plan": plan.to_dict(),
                    "command": command,
                    "model_manager_registered": model_manager_registered,
                },
                sort_keys=True,
            )
        )
        return 0 if plan.enabled else 1
    if args.model_status:
        status = llama_cpp_model_cache_status(config)
        payload = status.to_dict()
        payload["model_manager_registered"] = _register_llama_cpp_model_with_manager(
            config,
            status,
            enabled=bool(args.track_model) and bool(status.complete),
        )
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("complete") else 1
    if args.prefetch_model and not args.serve:
        status = prefetch_llama_cpp_model(
            config,
            force_download=bool(args.force_download),
            local_files_only=bool(args.local_files_only),
            model_cache=bool(args.model_cache) or None,
            model_cid=args.model_cid,
            pin_model=bool(args.pin_model),
            async_hash=args.async_model_hash,
            track_model=bool(args.track_model),
        )
        payload = status.to_dict()
        payload["model_manager_registered"] = _register_llama_cpp_model_with_manager(
            config,
            status,
            enabled=bool(args.track_model) and bool(status.complete),
        )
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("complete") else 1
    if args.probe:
        payload = {"running": llama_cpp_server_ready(config.base_url), "base_url": config.base_url}
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["running"] else 1

    result = ensure_llama_cpp_server(
        config,
        autostart=bool(args.serve),
        auto_install=bool(args.auto_install),
        auto_update=bool(args.auto_update),
        prefetch_model=bool(args.prefetch_model),
        prefetch_force_download=bool(args.force_download),
        prefetch_local_files_only=bool(args.local_files_only),
        prefetch_model_cache=bool(args.model_cache) or None,
        prefetch_model_cid=args.model_cid,
        prefetch_pin_model=bool(args.pin_model),
        prefetch_async_hash=args.async_model_hash,
        startup_timeout_seconds=args.startup_timeout_seconds,
        track_model=bool(args.track_model),
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.running or (not args.serve and result.install.available) else 1


if __name__ == "__main__":
    raise SystemExit(main())
