"""Explicit, lazy provisioning for proof-gated contract-repair toolchains.

The capability probe is intentionally detect-only.  This module is the
separate provisioning boundary used by an operator (or by packaging) when a
reviewed dependency is missing:

* Python tools are installed from a closed allowlist of version-constrained
  requirements.
* TypeScript is installed with npm into a versioned user-local directory.  It
  is never represented as a Python requirement.
* imports and package-manager processes occur only when a public probe/ensure
  function is called; importing this module is side-effect free.

An ``ensure`` call must pass ``auto_install=True`` to mutate the environment.
That keeps supervisor analysis and proof admission hermetic while still
providing a first-use lazy loader for explicitly requested provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

PINNED_TYPESCRIPT_VERSION: Final = "5.9.3"
TYPESCRIPT_REQUIREMENT: Final = f"typescript@{PINNED_TYPESCRIPT_VERSION}"
DEFAULT_INSTALL_TIMEOUT_SECONDS: Final = 600.0
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 120.0
DEPENDENCY_RECEIPT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-dependency@1"
)


@dataclass(frozen=True)
class PythonDependencySpec:
    """One allowlisted Python dependency and its runtime surfaces."""

    dependency_id: str
    import_name: str
    distribution: str
    requirement: str
    executable: str = ""
    executable_required: bool = False


PYTHON_DEPENDENCY_SPECS: Final[Mapping[str, PythonDependencySpec]] = MappingProxyType(
    {
        "z3": PythonDependencySpec(
            dependency_id="z3",
            import_name="z3",
            distribution="z3-solver",
            requirement="z3-solver>=4.12.0,<5.0.0",
            executable="z3",
            executable_required=True,
        ),
        "cvc5": PythonDependencySpec(
            dependency_id="cvc5",
            import_name="cvc5",
            distribution="cvc5",
            requirement="cvc5==1.3.3",
            executable="cvc5",
        ),
        "mypy": PythonDependencySpec(
            dependency_id="mypy",
            import_name="mypy",
            distribution="mypy",
            requirement="mypy>=1.8.0,<2.0.0",
            executable="mypy",
        ),
        "ruff": PythonDependencySpec(
            dependency_id="ruff",
            import_name="ruff",
            distribution="ruff",
            requirement="ruff>=0.12.0,<1.0.0",
            executable="ruff",
        ),
    }
)

CONTRACT_REPAIR_PYTHON_REQUIREMENTS: Final[tuple[str, ...]] = tuple(
    spec.requirement for spec in PYTHON_DEPENDENCY_SPECS.values()
)


@dataclass(frozen=True)
class ContractRepairDependencyReceipt:
    """Typed result from detection or an explicitly requested installation."""

    dependency_id: str
    status: str
    requirement: str
    version: str = ""
    module_path: str = ""
    executable_path: str = ""
    install_attempted: bool = False
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DEPENDENCY_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {
            "available",
            "missing",
            "incompatible",
            "install_disabled",
            "install_failed",
            "timed_out",
        }:
            raise ValueError(f"unsupported dependency status: {self.status}")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dependency_id": self.dependency_id,
            "status": self.status,
            "available": self.available,
            "requirement": self.requirement,
            "version": self.version,
            "module_path": self.module_path,
            "executable_path": self.executable_path,
            "install_attempted": self.install_attempted,
            "reason": self.reason,
            "details": dict(self.details),
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _scripts_directories() -> tuple[Path, ...]:
    candidates: list[Path] = [Path(sys.executable).resolve().parent]
    configured = sysconfig.get_path("scripts")
    if configured:
        candidates.append(Path(configured))
    user_base = getattr(sys, "base_exec_prefix", "")
    if user_base:
        candidates.append(Path(user_base) / ("Scripts" if os.name == "nt" else "bin"))
    try:
        candidates.append(Path.home() / ".local" / ("Scripts" if os.name == "nt" else "bin"))
    except (OSError, RuntimeError):
        pass
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def find_python_executable(
    command: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Find a Python-installed command even when its scripts dir is off PATH."""

    located = which(command)
    if located:
        return str(Path(located).resolve())
    names = (command, f"{command}.exe", f"{command}.cmd") if os.name == "nt" else (command,)
    for directory in _scripts_directories():
        for name in names:
            candidate = directory / name
            if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
                return str(candidate.resolve())
    return ""


def _module_path(module: Any) -> str:
    path = str(getattr(module, "__file__", "") or "")
    return str(Path(path).resolve()) if path else ""


def probe_python_dependency(
    dependency_id: str,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    which: Callable[[str], str | None] = shutil.which,
) -> ContractRepairDependencyReceipt:
    """Import and inspect one Python tool only when its feature is requested."""

    try:
        spec = PYTHON_DEPENDENCY_SPECS[dependency_id]
    except KeyError as exc:
        raise ValueError(f"unsupported contract-repair dependency: {dependency_id}") from exc
    try:
        module = importer(spec.import_name)
    except ModuleNotFoundError as exc:
        missing_root = str(getattr(exc, "name", "") or "").split(".", 1)[0]
        requested_root = spec.import_name.split(".", 1)[0]
        reason = (
            "requested_module_missing"
            if missing_root in {"", requested_root}
            else f"transitive_module_missing:{missing_root}"
        )
        return ContractRepairDependencyReceipt(
            dependency_id,
            "missing",
            spec.requirement,
            reason=reason,
        )
    except Exception as exc:
        return ContractRepairDependencyReceipt(
            dependency_id,
            "incompatible",
            spec.requirement,
            reason=f"import_failed:{type(exc).__name__}",
        )

    try:
        version = importlib.metadata.version(spec.distribution)
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "") or "")
    if dependency_id == "cvc5" and version != "1.3.3":
        return ContractRepairDependencyReceipt(
            dependency_id,
            "incompatible",
            spec.requirement,
            version=version,
            module_path=_module_path(module),
            reason="python_binding_version_mismatch",
            details={"expected_version": "1.3.3"},
        )
    executable_path = (
        find_python_executable(spec.executable, which=which) if spec.executable else ""
    )
    if spec.executable_required and not executable_path:
        return ContractRepairDependencyReceipt(
            dependency_id,
            "incompatible",
            spec.requirement,
            version=version,
            module_path=_module_path(module),
            reason="required_executable_missing",
        )
    return ContractRepairDependencyReceipt(
        dependency_id,
        "available",
        spec.requirement,
        version=version,
        module_path=_module_path(module),
        executable_path=executable_path,
        details={
            "executable_optional": bool(spec.executable and not spec.executable_required),
            "python_module_command": [sys.executable, "-m", spec.import_name],
        },
    )


def _data_home(environ: Mapping[str, str]) -> Path:
    configured = str(environ.get("XDG_DATA_HOME", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share"


def typescript_toolchain_root(
    *,
    environ: Mapping[str, str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the versioned managed TypeScript root without creating it."""

    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ if environ is None else environ
    configured = str(env.get("IPFS_ACCELERATE_CONTRACT_REPAIR_TYPESCRIPT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        _data_home(env)
        / "ipfs_accelerate_py"
        / "contract-repair"
        / f"typescript-{PINNED_TYPESCRIPT_VERSION}"
    ).resolve()


def _typescript_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    package_dir = root / "node_modules" / "typescript"
    package_json = package_dir / "package.json"
    compiler_api = package_dir / "lib" / "typescript.js"
    if os.name == "nt":
        tsc = root / "node_modules" / ".bin" / "tsc.cmd"
    else:
        tsc = root / "node_modules" / ".bin" / "tsc"
    return package_dir, package_json, compiler_api, tsc


def _run(
    runner: Runner,
    command: Sequence[str],
    *,
    timeout_seconds: float,
    environ: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(os.environ if environ is None else environ),
    )


def _in_virtual_environment() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _pip_install_flags() -> list[str]:
    """Use a user-site override only for an explicit PEP 668-managed install."""

    if _in_virtual_environment():
        return []
    stdlib = sysconfig.get_path("stdlib")
    externally_managed = bool(stdlib and (Path(stdlib) / "EXTERNALLY-MANAGED").is_file())
    explicitly_allowed = os.environ.get("PIP_BREAK_SYSTEM_PACKAGES") == "1"
    if externally_managed:
        # ``--user`` prevents an explicit lazy install from mutating the
        # distribution-owned prefix; PEP 668 still requires the paired opt-out.
        return ["--user", "--break-system-packages"]
    if explicitly_allowed:
        return ["--break-system-packages"]
    return []


def _bounded_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    return output[-2000:]


def _version_from_output(completed: subprocess.CompletedProcess[str]) -> str:
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    for token in output.replace("\n", " ").split():
        if token and token[0].isdigit() and token.count(".") >= 1:
            return token.strip()
    return ""


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_typescript_toolchain(
    *,
    environ: Mapping[str, str] | None = None,
    root: str | os.PathLike[str] | None = None,
    runner: Runner = subprocess.run,
    timeout_seconds: float = 15.0,
) -> ContractRepairDependencyReceipt:
    """Verify both the TypeScript CLI and compiler API at the reviewed version."""

    env = dict(os.environ if environ is None else environ)
    managed_root = typescript_toolchain_root(environ=env, root=root)
    package_dir, package_json, compiler_api, tsc = _typescript_paths(managed_root)
    if not package_json.is_file() or not compiler_api.is_file() or not tsc.is_file():
        return ContractRepairDependencyReceipt(
            "typescript",
            "missing",
            TYPESCRIPT_REQUIREMENT,
            reason="managed_cli_or_compiler_api_missing",
            details={
                "toolchain_root": str(managed_root),
                "typescript_path": str(package_dir),
            },
        )
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ContractRepairDependencyReceipt(
            "typescript",
            "incompatible",
            TYPESCRIPT_REQUIREMENT,
            reason=f"package_metadata_invalid:{type(exc).__name__}",
        )
    metadata_version = str(package.get("version", "") or "")
    if metadata_version != PINNED_TYPESCRIPT_VERSION:
        return ContractRepairDependencyReceipt(
            "typescript",
            "incompatible",
            TYPESCRIPT_REQUIREMENT,
            version=metadata_version,
            module_path=str(package_dir),
            executable_path=str(tsc),
            reason="package_version_mismatch",
            details={"expected_version": PINNED_TYPESCRIPT_VERSION},
        )
    node = shutil.which("node", path=env.get("PATH"))
    if not node:
        return ContractRepairDependencyReceipt(
            "typescript",
            "missing",
            TYPESCRIPT_REQUIREMENT,
            version=metadata_version,
            module_path=str(package_dir),
            executable_path=str(tsc),
            reason="node_executable_missing",
        )
    try:
        completed = _run(
            runner,
            [str(node), str(package_dir / "bin" / "tsc"), "--version"],
            timeout_seconds=timeout_seconds,
            environ=env,
        )
    except subprocess.TimeoutExpired:
        return ContractRepairDependencyReceipt(
            "typescript",
            "timed_out",
            TYPESCRIPT_REQUIREMENT,
            version=metadata_version,
            module_path=str(package_dir),
            executable_path=str(tsc),
            reason="version_probe_timed_out",
        )
    except OSError as exc:
        return ContractRepairDependencyReceipt(
            "typescript",
            "incompatible",
            TYPESCRIPT_REQUIREMENT,
            version=metadata_version,
            module_path=str(package_dir),
            executable_path=str(tsc),
            reason=f"version_probe_failed:{type(exc).__name__}",
        )
    observed_version = _version_from_output(completed)
    if completed.returncode != 0 or observed_version != PINNED_TYPESCRIPT_VERSION:
        return ContractRepairDependencyReceipt(
            "typescript",
            "incompatible",
            TYPESCRIPT_REQUIREMENT,
            version=observed_version or metadata_version,
            module_path=str(package_dir),
            executable_path=str(tsc),
            reason="compiler_version_mismatch",
            details={"expected_version": PINNED_TYPESCRIPT_VERSION},
        )
    return ContractRepairDependencyReceipt(
        "typescript",
        "available",
        TYPESCRIPT_REQUIREMENT,
        version=observed_version,
        module_path=str(package_dir.resolve()),
        executable_path=str(tsc.resolve()),
        details={
            "toolchain_root": str(managed_root),
            "typescript_path": str(package_dir.resolve()),
            "compiler_api_path": str(compiler_api.resolve()),
            "compiler_api_sha256": _sha256(compiler_api),
            "package_lock_sha256": _sha256(managed_root / "package-lock.json"),
            "node_executable": str(Path(node).resolve()),
        },
    )


class _InstallLock:
    """Small cross-process lock used only around package-manager mutation."""

    def __init__(self, path: Path, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._acquired = False

    def __enter__(self) -> _InstallLock:
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n")
                self._acquired = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > max(self.timeout_seconds * 5, 1800):
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for dependency lock {self.path}"
                    ) from None
                time.sleep(0.1)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def ensure_python_dependency(
    dependency_id: str,
    *,
    auto_install: bool = False,
    runner: Runner = subprocess.run,
    timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> ContractRepairDependencyReceipt:
    """Resolve one Python dependency, installing only after explicit opt-in."""

    current = probe_python_dependency(dependency_id)
    if current.available:
        return current
    spec = PYTHON_DEPENDENCY_SPECS.get(dependency_id)
    if spec is None:
        raise ValueError(f"unsupported contract-repair dependency: {dependency_id}")
    if not auto_install:
        return ContractRepairDependencyReceipt(
            dependency_id,
            "install_disabled",
            spec.requirement,
            reason=current.reason or "explicit_install_required",
        )
    lock_path = (
        Path(tempfile.gettempdir()) / "ipfs_accelerate_py-contract-repair-python-dependencies.lock"
    )
    try:
        with _InstallLock(lock_path, lock_timeout_seconds):
            current = probe_python_dependency(dependency_id)
            if current.available:
                return current
            completed = _run(
                runner,
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    *_pip_install_flags(),
                    spec.requirement,
                ],
                timeout_seconds=timeout_seconds,
            )
    except (subprocess.TimeoutExpired, TimeoutError):
        return ContractRepairDependencyReceipt(
            dependency_id,
            "timed_out",
            spec.requirement,
            install_attempted=True,
            reason="pip_install_timed_out",
        )
    except OSError as exc:
        return ContractRepairDependencyReceipt(
            dependency_id,
            "install_failed",
            spec.requirement,
            install_attempted=True,
            reason=f"pip_install_failed:{type(exc).__name__}",
        )
    if completed.returncode != 0:
        return ContractRepairDependencyReceipt(
            dependency_id,
            "install_failed",
            spec.requirement,
            install_attempted=True,
            reason="pip_nonzero_exit",
            details={
                "returncode": completed.returncode,
                "output": _bounded_process_output(completed),
            },
        )
    importlib.invalidate_caches()
    installed = probe_python_dependency(dependency_id)
    return ContractRepairDependencyReceipt(
        installed.dependency_id,
        installed.status,
        installed.requirement,
        version=installed.version,
        module_path=installed.module_path,
        executable_path=installed.executable_path,
        install_attempted=True,
        reason=installed.reason,
        details=installed.details,
    )


def ensure_typescript_toolchain(
    *,
    auto_install: bool = False,
    environ: Mapping[str, str] | None = None,
    root: str | os.PathLike[str] | None = None,
    runner: Runner = subprocess.run,
    timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> ContractRepairDependencyReceipt:
    """Resolve TypeScript, invoking npm only after explicit opt-in."""

    env = dict(os.environ if environ is None else environ)
    managed_root = typescript_toolchain_root(environ=env, root=root)
    current = probe_typescript_toolchain(
        environ=env,
        root=managed_root,
        runner=runner,
        timeout_seconds=min(timeout_seconds, 30.0),
    )
    if current.available:
        return current
    if not auto_install:
        return ContractRepairDependencyReceipt(
            "typescript",
            "install_disabled",
            TYPESCRIPT_REQUIREMENT,
            reason=current.reason or "explicit_install_required",
            details={"toolchain_root": str(managed_root)},
        )
    npm = shutil.which("npm", path=env.get("PATH"))
    if not npm:
        return ContractRepairDependencyReceipt(
            "typescript",
            "install_failed",
            TYPESCRIPT_REQUIREMENT,
            install_attempted=False,
            reason="npm_executable_missing",
        )
    lock_path = managed_root.parent / f".typescript-{PINNED_TYPESCRIPT_VERSION}.lock"
    try:
        with _InstallLock(lock_path, lock_timeout_seconds):
            current = probe_typescript_toolchain(
                environ=env,
                root=managed_root,
                runner=runner,
                timeout_seconds=min(timeout_seconds, 30.0),
            )
            if current.available:
                return current
            managed_root.mkdir(parents=True, exist_ok=True)
            completed = _run(
                runner,
                [
                    str(npm),
                    "install",
                    "--prefix",
                    str(managed_root),
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                    "--save-exact",
                    TYPESCRIPT_REQUIREMENT,
                ],
                timeout_seconds=timeout_seconds,
                environ=env,
            )
    except (subprocess.TimeoutExpired, TimeoutError):
        return ContractRepairDependencyReceipt(
            "typescript",
            "timed_out",
            TYPESCRIPT_REQUIREMENT,
            install_attempted=True,
            reason="npm_install_timed_out",
        )
    except OSError as exc:
        return ContractRepairDependencyReceipt(
            "typescript",
            "install_failed",
            TYPESCRIPT_REQUIREMENT,
            install_attempted=True,
            reason=f"npm_install_failed:{type(exc).__name__}",
        )
    if completed.returncode != 0:
        return ContractRepairDependencyReceipt(
            "typescript",
            "install_failed",
            TYPESCRIPT_REQUIREMENT,
            install_attempted=True,
            reason="npm_nonzero_exit",
            details={
                "returncode": completed.returncode,
                "output": _bounded_process_output(completed),
            },
        )
    installed = probe_typescript_toolchain(
        environ=env,
        root=managed_root,
        runner=runner,
        timeout_seconds=min(timeout_seconds, 30.0),
    )
    return ContractRepairDependencyReceipt(
        installed.dependency_id,
        installed.status,
        installed.requirement,
        version=installed.version,
        module_path=installed.module_path,
        executable_path=installed.executable_path,
        install_attempted=True,
        reason=installed.reason,
        details=installed.details,
    )


def find_contract_repair_executable(command: str) -> str | None:
    """Detect installed or managed tools without invoking an installer."""

    if command == "tsc":
        receipt = probe_typescript_toolchain()
        if receipt.available:
            return receipt.executable_path
    located = find_python_executable(command)
    return located or None


def contract_repair_toolchain_environment(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return child-process bindings for an already provisioned toolchain."""

    env = dict(os.environ if environ is None else environ)
    receipt = probe_typescript_toolchain(environ=env)
    if not receipt.available:
        return {}
    bin_dir = str(Path(receipt.executable_path).parent)
    existing_path = str(env.get("PATH", "") or "")
    return {
        "TYPESCRIPT_PATH": receipt.module_path,
        "PATH": os.pathsep.join(part for part in (bin_dir, existing_path) if part),
    }


def ensure_contract_repair_dependencies(
    dependency_ids: Sequence[str] | None = None,
    *,
    auto_install: bool = False,
) -> tuple[ContractRepairDependencyReceipt, ...]:
    """Resolve the requested closed dependency set."""

    selected = tuple(dependency_ids or (*PYTHON_DEPENDENCY_SPECS.keys(), "typescript"))
    receipts: list[ContractRepairDependencyReceipt] = []
    for dependency_id in selected:
        if dependency_id == "typescript":
            receipts.append(ensure_typescript_toolchain(auto_install=auto_install))
        else:
            receipts.append(
                ensure_python_dependency(
                    dependency_id,
                    auto_install=auto_install,
                )
            )
    return tuple(receipts)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe or explicitly install proof-gated contract-repair dependencies."
    )
    parser.add_argument(
        "dependencies",
        nargs="*",
        choices=tuple(PYTHON_DEPENDENCY_SPECS) + ("typescript",),
        help="closed dependency ids; omitted means all",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="explicitly permit pip/npm installation for missing dependencies",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print detected TYPESCRIPT_PATH and PATH bindings",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print_env:
        for key, value in contract_repair_toolchain_environment().items():
            print(f"{key}={value}")
        return 0
    receipts = ensure_contract_repair_dependencies(
        args.dependencies or None,
        auto_install=bool(args.install),
    )
    print(json.dumps([receipt.to_dict() for receipt in receipts], indent=2, sort_keys=True))
    return 0 if all(receipt.available for receipt in receipts) else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "CONTRACT_REPAIR_PYTHON_REQUIREMENTS",
    "DEPENDENCY_RECEIPT_SCHEMA_VERSION",
    "PINNED_TYPESCRIPT_VERSION",
    "PYTHON_DEPENDENCY_SPECS",
    "TYPESCRIPT_REQUIREMENT",
    "ContractRepairDependencyReceipt",
    "PythonDependencySpec",
    "contract_repair_toolchain_environment",
    "ensure_contract_repair_dependencies",
    "ensure_python_dependency",
    "ensure_typescript_toolchain",
    "find_contract_repair_executable",
    "find_python_executable",
    "probe_python_dependency",
    "probe_typescript_toolchain",
    "typescript_toolchain_root",
]
