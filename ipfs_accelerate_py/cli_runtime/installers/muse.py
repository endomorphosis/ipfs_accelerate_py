"""Detect-only Muse Code CLI discovery with an explicit official installer.

Discovery order (never installs during import or implicit discovery)::

    1. explicit path argument / ``IPFS_ACCELERATE_MUSE_PATH``
    2. ``PATH`` lookup for ``muse``
    3. default launcher location ``~/.local/bin/muse``
       (or ``$MUSE_INSTALL_DIR/muse``)

Installation is opt-in via :func:`ensure_muse` only. Generic provider
discovery must call :func:`discover_muse` (detect-only). The official Meta
installer is ``curl -fsSL https://dev.meta.ai/install.sh | sh`` (the script
is bash); this module downloads that script to a temp file and runs it with
bash only when auto-install is explicitly enabled. The official script
writes a launcher to ``~/.local/bin/muse`` and then runs
``MUSE_LAUNCHER_INSTALL=1 muse``.

Authentication readiness is separate from binary availability. Headless /
CI runs use ``META_API_KEY`` (with the same Meta Model API key aliases).
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

MUSE_EXECUTABLE = "muse"
OFFICIAL_INSTALL_URL = "https://dev.meta.ai/install.sh"
OFFICIAL_INSTALL_HOST = "dev.meta.ai"
DEFAULT_LAUNCHER_DIR_NAME = ".local/bin"

_AUTO_INSTALL_ENV_NAMES = (
    "IPFS_ACCELERATE_MUSE_AUTO_INSTALL",
    "IPFS_ACCELERATE_PY_MUSE_AUTO_INSTALL",
    "ipfs_accelerate_py_MUSE_AUTO_INSTALL",
)
_PATH_ENV_NAMES = (
    "IPFS_ACCELERATE_MUSE_PATH",
    "IPFS_ACCELERATE_PY_MUSE_PATH",
    "ipfs_accelerate_py_MUSE_BIN",
    "IPFS_ACCELERATE_PY_MUSE_BIN",
    "IPFS_ACCELERATE_AGENT_MUSE_BIN",
    "MUSE_BIN",
    "MUSE_CLI_PATH",
)
_INSTALL_DIR_ENV_NAMES = (
    "MUSE_INSTALL_DIR",
    "IPFS_ACCELERATE_MUSE_INSTALL_DIR",
    "IPFS_ACCELERATE_PY_MUSE_INSTALL_DIR",
)
_AUTH_ENV_NAMES = (
    "META_API_KEY",
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "IPFS_ACCELERATE_PY_META_AI_API_KEY",
)

_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_INSTALL_LOCK = threading.Lock()
_MAX_INSTALL_SCRIPT_BYTES = 1_048_576

DownloadFn = Callable[[str, Path], None]
RunFn = Callable[..., subprocess.CompletedProcess]
WhichFn = Callable[[str, Optional[Mapping[str, str]]], Optional[str]]


@dataclass(frozen=True)
class MuseInstallResult:
    """Outcome of one idempotent Muse Code discovery or installation attempt."""

    available: bool
    installed: bool = False
    executable: str = ""
    version: str = ""
    method: str = ""
    reason: str = ""
    details: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class MuseReadiness:
    """Typed readiness independent of installation and authentication.

    ``installed`` means a Muse Code binary was discovered. ``authenticated``
    is a coarse marker that a Meta API key env var is present; this module
    never runs ``muse login`` or inspects secret values beyond emptiness.
    ``ready`` is true only when both are true.
    """

    installed: bool
    authenticated: bool
    ready: bool
    executable: str = ""
    version: str = ""
    reason: str = ""


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _falsey(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _FALSE_VALUES


def muse_auto_install_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether explicit ``ensure_muse`` may invoke the official installer."""
    env = os.environ if environ is None else environ
    for name in _AUTO_INSTALL_ENV_NAMES:
        raw = env.get(name)
        if raw is None:
            continue
        if _falsey(raw):
            return False
        if _truthy(raw) or str(raw).strip():
            return True
    # Default: allow explicit ensure_muse unless the operator disabled it.
    return True


def default_install_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Return the directory the official installer uses for ``muse``."""
    env = os.environ if environ is None else environ
    for name in _INSTALL_DIR_ENV_NAMES:
        raw = str(env.get(name) or "").strip()
        if raw:
            return Path(os.path.expanduser(raw))
    home = Path(os.path.expanduser(env.get("HOME") or Path.home()))
    return home / ".local" / "bin"


def default_launcher_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    return default_install_dir(environ) / MUSE_EXECUTABLE


def muse_auth_available(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when a Meta API key env var is non-empty (value is not returned)."""
    env = os.environ if environ is None else environ
    for name in _AUTH_ENV_NAMES:
        if str(env.get(name) or "").strip():
            return True
    try:
        from ...common.meta_model_api import resolve_meta_model_api_key

        return bool(resolve_meta_model_api_key())
    except Exception:
        return False


def _which(
    name: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    which_fn: Optional[WhichFn] = None,
) -> Optional[str]:
    if which_fn is not None:
        return which_fn(name, environ)
    env = os.environ if environ is None else environ
    path = env.get("PATH")
    return shutil.which(name, path=path)


def _is_executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _probe_version(
    executable: str,
    *,
    run_fn: Optional[RunFn] = None,
    timeout: float = 8.0,
) -> str:
    runner = run_fn or subprocess.run
    try:
        proc = runner(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    text = (getattr(proc, "stdout", "") or "") + "\n" + (getattr(proc, "stderr", "") or "")
    return str(text).strip().splitlines()[0].strip() if str(text).strip() else ""


def _candidate_from_env(
    environ: Mapping[str, str],
    *,
    which_fn: Optional[WhichFn] = None,
) -> Optional[str]:
    for name in _PATH_ENV_NAMES:
        raw = str(environ.get(name) or "").strip()
        if not raw:
            continue
        path = Path(os.path.expanduser(raw))
        if _is_executable(path):
            return str(path)
        found = _which(raw, environ=environ, which_fn=which_fn)
        if found:
            return found
    return None


def discover_muse(
    *,
    explicit_path: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    probe_version: bool = False,
    run_fn: Optional[RunFn] = None,
    which_fn: Optional[WhichFn] = None,
) -> MuseInstallResult:
    """Detect-only Muse Code discovery. Never downloads or installs."""
    env = os.environ if environ is None else environ
    if explicit_path and str(explicit_path).strip():
        raw = str(explicit_path).strip()
        path = Path(os.path.expanduser(raw))
        if _is_executable(path):
            version = _probe_version(str(path), run_fn=run_fn) if probe_version else ""
            return MuseInstallResult(
                available=True,
                executable=str(path),
                version=version,
                method="explicit_path",
            )
        found = _which(raw, environ=env, which_fn=which_fn)
        if found:
            version = _probe_version(found, run_fn=run_fn) if probe_version else ""
            return MuseInstallResult(
                available=True,
                executable=found,
                version=version,
                method="explicit_which",
            )
        return MuseInstallResult(
            available=False,
            method="explicit_path",
            reason="not_installed",
            details={"path": raw},
        )

    configured = _candidate_from_env(env, which_fn=which_fn)
    if configured:
        version = _probe_version(configured, run_fn=run_fn) if probe_version else ""
        return MuseInstallResult(
            available=True,
            executable=configured,
            version=version,
            method="env_path",
        )

    found = _which(MUSE_EXECUTABLE, environ=env, which_fn=which_fn)
    if found:
        version = _probe_version(found, run_fn=run_fn) if probe_version else ""
        return MuseInstallResult(
            available=True,
            executable=found,
            version=version,
            method="path",
        )

    launcher = default_launcher_path(env)
    if _is_executable(launcher):
        version = _probe_version(str(launcher), run_fn=run_fn) if probe_version else ""
        return MuseInstallResult(
            available=True,
            executable=str(launcher),
            version=version,
            method="default_launcher",
        )

    return MuseInstallResult(
        available=False,
        method="discover",
        reason="not_installed",
        details={"expected": str(launcher)},
    )


def _default_download(url: str, dest: Path) -> None:
    curl = shutil.which("curl")
    if curl:
        proc = subprocess.run(
            [
                curl,
                "-fsSL",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--tlsv1.2",
                "--max-redirs",
                "3",
                "-A",
                "ipfs-accelerate-py-muse",
                "-o",
                str(dest),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if (
            proc.returncode == 0
            and dest.is_file()
            and 16 <= dest.stat().st_size <= _MAX_INSTALL_SCRIPT_BYTES
        ):
            return
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
    request = Request(url, method="GET", headers={"User-Agent": "ipfs-accelerate-py-muse"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - pinned https host
        data = response.read(_MAX_INSTALL_SCRIPT_BYTES + 1)
    if len(data) > _MAX_INSTALL_SCRIPT_BYTES:
        raise RuntimeError("Muse installer script exceeded size bound")
    dest.write_bytes(data)


def _run_official_installer(
    *,
    install_dir: Path,
    environ: Mapping[str, str],
    download_fn: DownloadFn,
    run_fn: RunFn,
    installer_url: str = OFFICIAL_INSTALL_URL,
) -> MuseInstallResult:
    install_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="muse-install-") as tmp:
        script = Path(tmp) / "install.sh"
        try:
            download_fn(installer_url, script)
        except (OSError, URLError, RuntimeError) as exc:
            return MuseInstallResult(
                available=False,
                method="official_installer",
                reason="download_failed",
                details={"error": type(exc).__name__},
            )
        if not script.is_file() or script.stat().st_size < 16:
            return MuseInstallResult(
                available=False,
                method="official_installer",
                reason="empty_installer",
            )
        try:
            header = script.read_text(encoding="utf-8", errors="replace")[:80]
        except OSError:
            header = ""
        if "bash" not in header and "#!/bin" not in header:
            return MuseInstallResult(
                available=False,
                method="official_installer",
                reason="invalid_installer",
            )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        child_env = dict(environ)
        child_env["MUSE_INSTALL_DIR"] = str(install_dir)
        child_env["MUSE_NO_MODIFY_PATH"] = "1"
        try:
            proc = run_fn(
                ["bash", str(script)],
                cwd=str(install_dir),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return MuseInstallResult(
                available=False,
                method="official_installer",
                reason="install_failed",
                details={"error": type(exc).__name__},
            )
        launcher = install_dir / MUSE_EXECUTABLE
        script_failed = getattr(proc, "returncode", 1) not in {0, None}
        # Official install.sh may exit 1 after writing the launcher when
        # MUSE_NO_MODIFY_PATH is set (PATH messaging). Prefer the binary.
        if script_failed and not _is_executable(launcher):
            return MuseInstallResult(
                available=False,
                method="official_installer",
                reason="install_nonzero_exit",
                details={"returncode": str(getattr(proc, "returncode", ""))},
            )
        if _is_executable(launcher):
            version = _probe_version(str(launcher), run_fn=run_fn, timeout=30.0)
            if not version:
                launcher_env = dict(child_env)
                launcher_env["MUSE_LAUNCHER_INSTALL"] = "1"
                try:
                    run_fn(
                        [str(launcher)],
                        cwd=str(install_dir),
                        env=launcher_env,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
                version = _probe_version(str(launcher), run_fn=run_fn, timeout=30.0)
            return MuseInstallResult(
                available=True,
                installed=True,
                executable=str(launcher),
                version=version,
                method="official_installer",
            )
    launcher = install_dir / MUSE_EXECUTABLE
    if not _is_executable(launcher):
        return MuseInstallResult(
            available=False,
            method="official_installer",
            reason="installer_did_not_produce_binary",
            details={"expected": str(launcher)},
        )
    version = _probe_version(str(launcher), run_fn=run_fn, timeout=30.0)
    return MuseInstallResult(
        available=True,
        installed=True,
        executable=str(launcher),
        version=version,
        method="official_installer",
    )


def ensure_muse(
    *,
    explicit_path: Optional[str] = None,
    auto_install: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    probe_version: bool = True,
    download_fn: Optional[DownloadFn] = None,
    run_fn: Optional[RunFn] = None,
    which_fn: Optional[WhichFn] = None,
    installer_url: str = OFFICIAL_INSTALL_URL,
) -> MuseInstallResult:
    """Discover Muse Code, optionally installing via the official script.

    ``auto_install=False`` (default) is detect-only. ``auto_install=True``
    may download ``https://dev.meta.ai/install.sh`` only when
    :func:`muse_auto_install_enabled` is true.
    """
    env = os.environ if environ is None else environ
    found = discover_muse(
        explicit_path=explicit_path,
        environ=env,
        probe_version=probe_version,
        run_fn=run_fn,
        which_fn=which_fn,
    )
    if found.available:
        return found
    if not auto_install:
        return found
    if not muse_auto_install_enabled(env):
        return MuseInstallResult(
            available=False,
            method="official_installer",
            reason="auto_install_disabled",
        )
    with _INSTALL_LOCK:
        found_locked = discover_muse(
            explicit_path=explicit_path,
            environ=env,
            probe_version=probe_version,
            run_fn=run_fn,
            which_fn=which_fn,
        )
        if found_locked.available:
            return found_locked
        return _run_official_installer(
            install_dir=default_install_dir(env),
            environ=env,
            download_fn=download_fn or _default_download,
            run_fn=run_fn or subprocess.run,
            installer_url=installer_url,
        )


def assess_muse_readiness(
    *,
    install_result: Optional[MuseInstallResult] = None,
    environ: Optional[Mapping[str, str]] = None,
    auto_install: bool = False,
    **discover_kwargs: Any,
) -> MuseReadiness:
    """Combine binary discovery with authentication markers."""
    env = os.environ if environ is None else environ
    result = install_result
    if result is None:
        if auto_install:
            result = ensure_muse(auto_install=True, environ=env, **discover_kwargs)
        else:
            result = discover_muse(environ=env, **discover_kwargs)
    authenticated = muse_auth_available(env)
    installed = bool(result.available and result.executable)
    if not installed:
        return MuseReadiness(
            installed=False,
            authenticated=authenticated,
            ready=False,
            executable=result.executable,
            version=result.version,
            reason=result.reason or "not_installed",
        )
    if not authenticated:
        return MuseReadiness(
            installed=True,
            authenticated=False,
            ready=False,
            executable=result.executable,
            version=result.version,
            reason="missing_auth",
        )
    return MuseReadiness(
        installed=True,
        authenticated=True,
        ready=True,
        executable=result.executable,
        version=result.version,
        reason="ready",
    )


__all__ = [
    "MUSE_EXECUTABLE",
    "OFFICIAL_INSTALL_URL",
    "OFFICIAL_INSTALL_HOST",
    "MuseInstallResult",
    "MuseReadiness",
    "assess_muse_readiness",
    "default_install_dir",
    "default_launcher_path",
    "discover_muse",
    "ensure_muse",
    "muse_auth_available",
    "muse_auto_install_enabled",
]
