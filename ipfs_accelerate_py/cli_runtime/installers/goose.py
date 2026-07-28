"""Pinned, explicit-only Goose CLI lazy installer.

Discovery order (never installs during import or implicit discovery)::

    1. explicit path argument / ``IPFS_ACCELERATE_GOOSE_PATH``
    2. operator argv (first token when it points at an executable)
    3. ``PATH`` lookup for ``goose`` / ``goose.exe``
    4. managed version directory for the pinned release

Installation is opt-in via :func:`ensure_goose` only. Generic provider discovery
must call :func:`discover_goose` (detect-only). Archives are size-bounded,
SHA-256 verified against the packaged release manifest, safely extracted,
version-probed, and promoted atomically under per-process and cross-process
locks. Authentication and configuration readiness are separate from binary
availability and are never mutated by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.error import URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOOSE_EXECUTABLE = "goose"
GOOSE_EXECUTABLE_WINDOWS = "goose.exe"
DEFAULT_MANIFEST_NAME = "goose_release_manifest.json"
DEFAULT_REPO = "aaif-goose/goose"

_AUTO_INSTALL_ENV_NAMES = (
    "IPFS_ACCELERATE_GOOSE_AUTO_INSTALL",
    "IPFS_ACCELERATE_PY_GOOSE_AUTO_INSTALL",
    "ipfs_accelerate_py_GOOSE_AUTO_INSTALL",
)
_PATH_ENV_NAMES = (
    "IPFS_ACCELERATE_GOOSE_PATH",
    "IPFS_ACCELERATE_PY_GOOSE_PATH",
    "GOOSE_CLI_PATH",
)
_VARIANT_ENV_NAMES = (
    "IPFS_ACCELERATE_GOOSE_VARIANT",
    "IPFS_ACCELERATE_PY_GOOSE_VARIANT",
    "GOOSE_LINUX_VARIANT",
    "GOOSE_WINDOWS_VARIANT",
)
_MANAGED_ROOT_ENV_NAMES = (
    "IPFS_ACCELERATE_GOOSE_MANAGED_ROOT",
    "IPFS_ACCELERATE_PY_GOOSE_MANAGED_ROOT",
)
_AUTH_ENV_NAMES = (
    "GOOSE_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "XAI_API_KEY",
    "OLLAMA_HOST",
)

_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_INSTALL_LOCK = threading.Lock()
_VERSION_RE = re.compile(
    r"(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.]+)?)"
)

LINUX_VARIANTS = frozenset({"standard", "vulkan", "musl"})
WINDOWS_VARIANTS = frozenset({"standard", "cuda"})
SUPPORTED_OS = frozenset({"linux", "darwin", "windows"})
SUPPORTED_ARCH = frozenset({"x86_64", "aarch64"})

DownloadFn = Callable[[str, Path, float], None]
RunFn = Callable[..., subprocess.CompletedProcess]
WhichFn = Callable[[str], Optional[str]]
PlatformInfo = Tuple[str, str, str, str]  # os, arch, libc, variant


# ---------------------------------------------------------------------------
# Result / readiness types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GooseInstallResult:
    """Outcome of one idempotent Goose discovery or installation attempt."""

    available: bool
    installed: bool = False
    executable: str = ""
    version: str = ""
    method: str = ""
    reason: str = ""
    asset_name: str = ""
    managed_path: str = ""
    details: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class GooseReadiness:
    """Typed readiness independent of installation and authentication.

    ``installed`` means a Goose binary was discovered. ``authenticated`` is a
    coarse marker that *some* provider credential env var is present; this
    module never runs ``goose configure`` or inspects secret values beyond
    emptiness. ``ready`` is true only when both are true for chat use.
    """

    installed: bool
    authenticated: bool
    ready: bool
    executable: str = ""
    version: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Policy / env helpers
# ---------------------------------------------------------------------------


def goose_auto_install_enabled(
    explicit: Optional[bool] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether an explicit Goose request may install the CLI.

    Default is enabled for explicit ``ensure_goose`` callers. Policy can disable
    installation via *explicit* ``False`` or environment overrides. Implicit
    discovery must never call :func:`ensure_goose`.
    """

    if explicit is not None:
        return bool(explicit)
    env = os.environ if environ is None else environ
    for name in _AUTO_INSTALL_ENV_NAMES:
        raw = env.get(name)
        if raw is not None:
            return str(raw).strip().lower() not in _FALSE_VALUES
    return True


def goose_auth_available(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether any non-empty Goose-relevant auth marker is present.

    Does not validate credentials with a network call and does not run
    ``goose configure``.
    """

    env = os.environ if environ is None else environ
    for name in _AUTH_ENV_NAMES:
        if str(env.get(name) or "").strip():
            return True
    return False


def assess_goose_readiness(
    *,
    install_result: Optional[GooseInstallResult] = None,
    environ: Optional[Mapping[str, str]] = None,
    auto_install: bool = False,
    **discover_kwargs: Any,
) -> GooseReadiness:
    """Combine binary discovery with separate authentication readiness."""

    if install_result is None:
        install_result = discover_goose(environ=environ, **discover_kwargs)
        if (
            not install_result.available
            and auto_install
            and goose_auto_install_enabled(True, environ=environ)
        ):
            # Readiness assessment itself never installs unless the caller
            # explicitly sets auto_install=True (still separate from auth).
            install_result = ensure_goose(
                auto_install=True,
                environ=environ,
                **discover_kwargs,
            )
    authenticated = goose_auth_available(environ=environ)
    installed = bool(install_result.available and install_result.executable)
    reason = install_result.reason or (
        "ready" if installed and authenticated else (
            "missing_auth" if installed else "not_installed"
        )
    )
    if installed and not authenticated:
        reason = "missing_auth"
    elif installed and authenticated:
        reason = "ready"
    return GooseReadiness(
        installed=installed,
        authenticated=authenticated,
        ready=installed and authenticated,
        executable=install_result.executable,
        version=install_result.version,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Manifest loading and platform selection
# ---------------------------------------------------------------------------


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_MANIFEST_NAME


def load_release_manifest(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load and lightly validate the packaged release manifest."""

    manifest_path = Path(path) if path is not None else default_manifest_path()
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"Goose release manifest not found: {manifest_path}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Goose release manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Goose release manifest root must be an object")
    if "pinned_version" not in data or "assets" not in data:
        raise ValueError("Goose release manifest missing pinned_version/assets")
    if not isinstance(data["assets"], list) or not data["assets"]:
        raise ValueError("Goose release manifest assets must be a non-empty list")
    return data


def normalize_arch(machine: Optional[str] = None) -> str:
    value = (machine or platform.machine() or "").strip().lower()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "aarch64"
    return value


def normalize_os(system: Optional[str] = None) -> str:
    value = (system or platform.system() or "").strip().lower()
    if value.startswith("darwin") or value == "macos":
        return "darwin"
    if value.startswith("win"):
        return "windows"
    if value.startswith("linux"):
        return "linux"
    return value


def detect_linux_libc(
    *,
    environ: Optional[Mapping[str, str]] = None,
    run: Optional[RunFn] = None,
) -> str:
    """Return ``musl`` or ``gnu`` for Linux hosts."""

    env = os.environ if environ is None else environ
    forced = str(env.get("IPFS_ACCELERATE_GOOSE_LIBC") or "").strip().lower()
    if forced in {"musl", "gnu"}:
        return forced
    if os.environ.get("OSTYPE", "").lower().startswith("linux-musl"):
        return "musl"
    run_fn = subprocess.run if run is None else run
    try:
        completed = run_fn(
            ["ldd", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        blob = f"{completed.stdout or ''}{completed.stderr or ''}".lower()
        if "musl" in blob:
            return "musl"
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "gnu"


def resolve_variant(
    os_name: str,
    libc: str,
    *,
    explicit: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    env = os.environ if environ is None else environ
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().lower()
    for name in _VARIANT_ENV_NAMES:
        raw = env.get(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
    if os_name == "linux" and libc == "musl":
        return "musl"
    return "standard"


def detect_platform(
    *,
    os_name: Optional[str] = None,
    arch: Optional[str] = None,
    libc: Optional[str] = None,
    variant: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    run: Optional[RunFn] = None,
) -> PlatformInfo:
    resolved_os = normalize_os(os_name)
    resolved_arch = normalize_arch(arch)
    if resolved_os == "linux":
        resolved_libc = (libc or detect_linux_libc(environ=environ, run=run)).lower()
    elif resolved_os == "windows":
        resolved_libc = (libc or "msvc").lower()
    else:
        resolved_libc = (libc or "none").lower()
    resolved_variant = resolve_variant(
        resolved_os, resolved_libc, explicit=variant, environ=environ
    )
    return resolved_os, resolved_arch, resolved_libc, resolved_variant


def validate_platform(os_name: str, arch: str, libc: str, variant: str) -> Optional[str]:
    """Return a reason string when the platform is unsupported, else None."""

    if os_name not in SUPPORTED_OS:
        return f"unsupported_os:{os_name}"
    if arch not in SUPPORTED_ARCH:
        return f"unsupported_arch:{arch}"
    if os_name == "windows" and arch != "x86_64":
        return f"unsupported_windows_arch:{arch}"
    if os_name == "linux":
        if variant not in LINUX_VARIANTS:
            return f"unsupported_variant:{variant}"
        if variant == "musl" and libc not in {"musl", "gnu"}:
            return f"unsupported_libc:{libc}"
        if variant in {"standard", "vulkan"} and libc not in {"gnu", "musl"}:
            # standard/vulkan builds are gnu-linked; still allow explicit override.
            pass
    if os_name == "windows" and variant not in WINDOWS_VARIANTS:
        return f"unsupported_variant:{variant}"
    if os_name == "darwin" and variant not in {"standard"}:
        return f"unsupported_variant:{variant}"
    return None


def select_release_asset(
    manifest: Mapping[str, Any],
    *,
    os_name: str,
    arch: str,
    libc: str,
    variant: str,
) -> Optional[Dict[str, Any]]:
    """Pick the pinned asset for the platform, validating name and fields."""

    assets = list(manifest.get("assets") or [])
    matches: list[Dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            continue
        if str(asset.get("os", "")).lower() != os_name:
            continue
        if str(asset.get("arch", "")).lower() != arch:
            continue
        if str(asset.get("variant", "standard")).lower() != variant:
            continue
        asset_libc = str(asset.get("libc", "")).lower()
        if os_name == "linux" and variant == "musl":
            if asset_libc not in {"musl", ""}:
                continue
        elif os_name == "linux" and asset_libc and asset_libc not in {libc, "gnu"}:
            # Prefer exact libc; fall through if none match later.
            if asset_libc != libc:
                continue
        matches.append(dict(asset))

    if not matches and os_name == "linux" and variant in {"standard", "vulkan"}:
        for asset in assets:
            if not isinstance(asset, Mapping):
                continue
            if (
                str(asset.get("os", "")).lower() == os_name
                and str(asset.get("arch", "")).lower() == arch
                and str(asset.get("variant", "standard")).lower() == variant
            ):
                matches.append(dict(asset))

    if not matches:
        return None
    chosen = matches[0]
    name = str(chosen.get("asset_name") or "")
    if not name or ".." in name or "/" in name or "\\" in name:
        return None
    sha = str(chosen.get("sha256") or "").strip().lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        return None
    try:
        size = int(chosen.get("size_bytes") or 0)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    chosen["asset_name"] = name
    chosen["sha256"] = sha
    chosen["size_bytes"] = size
    return chosen


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def managed_install_root(
    *,
    version: str,
    environ: Optional[Mapping[str, str]] = None,
    managed_root: Optional[Union[str, Path]] = None,
) -> Path:
    env = os.environ if environ is None else environ
    if managed_root is not None:
        base = Path(managed_root).expanduser()
    else:
        for name in _MANAGED_ROOT_ENV_NAMES:
            raw = env.get(name)
            if raw and str(raw).strip():
                base = Path(str(raw).strip()).expanduser()
                break
        else:
            xdg = env.get("XDG_DATA_HOME")
            if xdg:
                base = Path(xdg).expanduser() / "ipfs_accelerate_py" / "goose"
            else:
                home = Path(str(env.get("HOME") or Path.home())).expanduser()
                base = home / ".local" / "share" / "ipfs_accelerate_py" / "goose"
    version_key = version.lstrip("v")
    return base / version_key


def managed_executable_path(
    *,
    version: str,
    os_name: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    managed_root: Optional[Union[str, Path]] = None,
) -> Path:
    root = managed_install_root(
        version=version, environ=environ, managed_root=managed_root
    )
    exe = (
        GOOSE_EXECUTABLE_WINDOWS
        if normalize_os(os_name) == "windows"
        else GOOSE_EXECUTABLE
    )
    return root / "bin" / exe


def _is_executable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        if os.name == "nt":
            return True
        return os.access(path, os.X_OK)
    except OSError:
        return False


def _normalize_candidate(raw: Union[str, Path, Sequence[str], None]) -> Optional[Path]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    text = str(raw).strip()
    if not text:
        return None
    # Operator argv may be a full command; only treat absolute/relative paths
    # that look like file paths as candidates.
    if " " in text and not text.startswith(("/", ".", "~")) and os.sep not in text:
        # bare command name without path separators is handled via PATH later
        if text in {GOOSE_EXECUTABLE, GOOSE_EXECUTABLE_WINDOWS}:
            return None
    path = Path(text).expanduser()
    return path


def _probe_version(
    executable: str,
    *,
    run: Optional[RunFn] = None,
    timeout_seconds: float = 10.0,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    run_fn = subprocess.run if run is None else run
    env = dict(os.environ if environ is None else environ)
    try:
        completed = run_fn(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_seconds)),
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    blob = f"{completed.stdout or ''}{completed.stderr or ''}".strip()
    match = _VERSION_RE.search(blob)
    return match.group("version") if match else blob[:64]


def discover_goose(
    *,
    explicit_path: Optional[Union[str, Path]] = None,
    operator_argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    which: Optional[WhichFn] = None,
    run: Optional[RunFn] = None,
    managed_root: Optional[Union[str, Path]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    os_name: Optional[str] = None,
    require_version: Optional[str] = None,
    probe_version: bool = True,
) -> GooseInstallResult:
    """Discover a Goose executable without installing anything.

    Search order: explicit path, operator argv, PATH, managed version directory.
    """

    env = dict(os.environ if environ is None else environ)
    which_fn = shutil.which if which is None else which
    data = dict(manifest) if manifest is not None else load_release_manifest()
    pinned = str(data.get("pinned_version") or "").strip()
    expected_version = (require_version or pinned).lstrip("v")

    candidates: list[Tuple[str, Path]] = []

    # 1. Explicit path argument
    if explicit_path is not None:
        path = _normalize_candidate(explicit_path)
        if path is not None:
            candidates.append(("explicit_path", path))

    # 1b. Explicit path environment variables
    for name in _PATH_ENV_NAMES:
        raw = env.get(name)
        if raw and str(raw).strip():
            path = _normalize_candidate(str(raw).strip())
            if path is not None:
                candidates.append(("explicit_env", path))

    # 2. Operator argv (first token if it is a path-like executable)
    if operator_argv:
        path = _normalize_candidate(operator_argv)
        if path is not None and (path.is_absolute() or os.sep in str(path) or str(path).startswith(".")):
            candidates.append(("operator_argv", path))

    # 3. PATH lookup
    for name in (GOOSE_EXECUTABLE, GOOSE_EXECUTABLE_WINDOWS):
        found = which_fn(name)
        if found:
            candidates.append(("path", Path(found).expanduser()))

    # 4. Managed version directory
    managed = managed_executable_path(
        version=pinned or expected_version or "unknown",
        os_name=os_name,
        environ=env,
        managed_root=managed_root,
    )
    candidates.append(("managed", managed))

    seen: set[str] = set()
    for method, candidate in candidates:
        try:
            resolved = candidate.resolve() if candidate.exists() else candidate
        except OSError:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not _is_executable_file(resolved):
            continue
        version = ""
        if probe_version:
            version = _probe_version(str(resolved), run=run, environ=env)
            if expected_version and version:
                # Accept when discovered version starts with expected major.minor.patch
                if not version.lstrip("v").startswith(expected_version.lstrip("v").split("-")[0]):
                    # For non-managed sources we still accept existing binaries;
                    # version gating for install is separate. Discovery reuses
                    # whatever is already present.
                    pass
        return GooseInstallResult(
            available=True,
            installed=False,
            executable=str(resolved),
            version=version.lstrip("v"),
            method=method,
            reason="already_installed",
            managed_path=str(managed),
        )

    return GooseInstallResult(
        available=False,
        method="not_found",
        reason="not_installed",
        managed_path=str(managed),
    )


# ---------------------------------------------------------------------------
# Download / verify / extract / promote
# ---------------------------------------------------------------------------


def _default_download(url: str, destination: Path, timeout_seconds: float) -> None:
    request = Request(url, headers={"User-Agent": "ipfs-accelerate-py-goose-installer/1.0"})
    with urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:  # nosec B310 - https only, pinned URL
        final_url = str(getattr(response, "geturl", lambda: url)() or url)
        if not final_url.lower().startswith("https://"):
            raise URLError(f"refusing non-HTTPS download URL: {final_url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> Tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"archive exceeds max size ({max_bytes} bytes)")
            digest.update(chunk)
    return digest.hexdigest(), total


def _is_safe_member_name(name: str, allowed: Sequence[str]) -> bool:
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return False
    basename = parts[-1] if parts else ""
    return basename in set(allowed)


def _safe_extract_archive(
    archive_path: Path,
    destination: Path,
    *,
    allowed_members: Sequence[str],
) -> Path:
    """Extract only allowed members; reject path traversal and absolute paths.

    Returns the path to the extracted goose executable.
    """

    destination.mkdir(parents=True, exist_ok=True)
    extracted_exe: Optional[Path] = None
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    member = info.filename
                    if info.is_dir():
                        continue
                    if not _is_safe_member_name(member, allowed_members):
                        # Allow only goose binary (+ optional .dll next to it on Windows)
                        base = Path(member.replace("\\", "/")).name
                        if not (base.lower().endswith(".dll") and ".." not in member and not member.startswith("/")):
                            raise ValueError(f"disallowed or unsafe archive member: {member}")
                    target = (destination / Path(member).name).resolve()
                    if not str(target).startswith(str(destination.resolve())):
                        raise ValueError(f"path traversal blocked: {member}")
                    with zf.open(info, "r") as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if Path(member).name in allowed_members:
                        extracted_exe = target
        except zipfile.BadZipFile as exc:
            raise ValueError(f"malformed archive: {exc}") from exc
    elif name.endswith((".tar.bz2", ".tar.gz", ".tgz", ".tbz2", ".tar")):
        mode = "r:*"
        try:
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if not _is_safe_member_name(member.name, allowed_members):
                        raise ValueError(f"disallowed or unsafe archive member: {member.name}")
                    # Extract by basename only to neutralize nested paths.
                    target = (destination / Path(member.name).name).resolve()
                    if not str(target).startswith(str(destination.resolve())):
                        raise ValueError(f"path traversal blocked: {member.name}")
                    source = tf.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot extract member: {member.name}")
                    with source, target.open("wb") as dst:
                        shutil.copyfileobj(source, dst)
                    if Path(member.name).name in allowed_members:
                        extracted_exe = target
        except tarfile.TarError as exc:
            raise ValueError(f"malformed archive: {exc}") from exc
    else:
        raise ValueError(f"unsupported archive format: {archive_path.name}")

    if extracted_exe is None or not extracted_exe.is_file():
        raise ValueError("archive did not contain the goose executable")
    # Ensure executable bit on POSIX.
    if os.name != "nt":
        mode_bits = extracted_exe.stat().st_mode
        extracted_exe.chmod(mode_bits | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return extracted_exe


def _atomic_replace_with_rollback(src: Path, dest: Path) -> None:
    """Atomically promote *src* to *dest*, restoring the previous file on failure."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    backup: Optional[Path] = None
    try:
        if dest.exists() or dest.is_symlink():
            backup = dest.with_suffix(dest.suffix + ".rollback")
            if backup.exists():
                backup.unlink()
            os.replace(dest, backup)
        os.replace(src, dest)
    except OSError:
        if backup is not None and backup.exists() and not dest.exists():
            try:
                os.replace(backup, dest)
            except OSError:
                pass
        raise
    else:
        if backup is not None and backup.exists():
            try:
                backup.unlink()
            except OSError:
                pass


@contextmanager
def _process_install_lock(
    lock_path: Optional[Path] = None,
) -> Iterator[None]:
    path = lock_path or (
        Path(tempfile.gettempdir()) / "ipfs-accelerate-goose-install.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _asset_download_url(manifest: Mapping[str, Any], asset_name: str) -> str:
    base = str(
        manifest.get("download_base_url")
        or f"https://github.com/{DEFAULT_REPO}/releases/download"
    ).rstrip("/")
    tag = str(manifest.get("release_tag") or manifest.get("pinned_version") or "").strip()
    if not tag:
        raise ValueError("manifest missing release_tag/pinned_version")
    if not asset_name or "/" in asset_name or "\\" in asset_name or ".." in asset_name:
        raise ValueError(f"invalid asset name: {asset_name!r}")
    return f"{base}/{tag}/{asset_name}"


def install_goose_from_manifest(
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
    managed_root: Optional[Union[str, Path]] = None,
    download: Optional[DownloadFn] = None,
    run: Optional[RunFn] = None,
    timeout_seconds: float = 300.0,
    os_name: Optional[str] = None,
    arch: Optional[str] = None,
    libc: Optional[str] = None,
    variant: Optional[str] = None,
    staging_dir: Optional[Union[str, Path]] = None,
) -> GooseInstallResult:
    """Download, verify, extract, and atomically promote a pinned Goose release.

    Never invokes curl-pipe-shell, sudo, goose configure, or shell-profile edits.
    """

    env = dict(os.environ if environ is None else environ)
    data = dict(manifest) if manifest is not None else load_release_manifest()
    pinned = str(data.get("pinned_version") or "").strip()
    if not pinned:
        return GooseInstallResult(
            available=False, method="manifest", reason="missing_pinned_version"
        )

    plat = detect_platform(
        os_name=os_name,
        arch=arch,
        libc=libc,
        variant=variant,
        environ=env,
        run=run,
    )
    os_n, arch_n, libc_n, variant_n = plat
    bad = validate_platform(os_n, arch_n, libc_n, variant_n)
    if bad:
        return GooseInstallResult(
            available=False,
            method="platform",
            reason=bad,
            details={
                "os": os_n,
                "arch": arch_n,
                "libc": libc_n,
                "variant": variant_n,
            },
        )

    asset = select_release_asset(
        data, os_name=os_n, arch=arch_n, libc=libc_n, variant=variant_n
    )
    if asset is None:
        return GooseInstallResult(
            available=False,
            method="platform",
            reason="no_matching_asset",
            details={
                "os": os_n,
                "arch": arch_n,
                "libc": libc_n,
                "variant": variant_n,
            },
        )

    max_size = int(data.get("max_archive_size_bytes") or asset["size_bytes"])
    if asset["size_bytes"] > max_size:
        return GooseInstallResult(
            available=False,
            method="bounds",
            reason="asset_size_exceeds_limit",
            asset_name=str(asset["asset_name"]),
            details={"size_bytes": str(asset["size_bytes"]), "max": str(max_size)},
        )

    allowed = list(data.get("allowed_archive_members") or ["goose", "goose.exe"])
    dest = managed_executable_path(
        version=pinned,
        os_name=os_n,
        environ=env,
        managed_root=managed_root,
    )
    download_fn = _default_download if download is None else download

    stage_root = Path(staging_dir) if staging_dir is not None else Path(
        tempfile.mkdtemp(prefix="goose-install-")
    )
    created_stage = staging_dir is None
    try:
        stage_root.mkdir(parents=True, exist_ok=True)
        archive_path = stage_root / str(asset["asset_name"])
        extract_dir = stage_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            url = _asset_download_url(data, str(asset["asset_name"]))
            download_fn(url, archive_path, float(timeout_seconds))
        except TimeoutError:
            return GooseInstallResult(
                available=False,
                method="download",
                reason="download_timeout",
                asset_name=str(asset["asset_name"]),
            )
        except (URLError, OSError, RuntimeError) as exc:
            message = str(exc).lower()
            reason = "download_timeout" if "timed out" in message or "timeout" in message else "offline_or_download_failed"
            return GooseInstallResult(
                available=False,
                method="download",
                reason=reason,
                asset_name=str(asset["asset_name"]),
                details={"error": str(exc)[:500]},
            )

        if not archive_path.is_file():
            return GooseInstallResult(
                available=False,
                method="download",
                reason="offline_or_download_failed",
                asset_name=str(asset["asset_name"]),
            )

        try:
            digest, size = _sha256_file(archive_path, max_bytes=max_size)
        except ValueError as exc:
            return GooseInstallResult(
                available=False,
                method="verify",
                reason="archive_size_exceeded",
                asset_name=str(asset["asset_name"]),
                details={"error": str(exc)[:500]},
            )

        # Size must match pinned size when known (strict provenance).
        if size != int(asset["size_bytes"]):
            return GooseInstallResult(
                available=False,
                method="verify",
                reason="archive_size_mismatch",
                asset_name=str(asset["asset_name"]),
                details={
                    "expected": str(asset["size_bytes"]),
                    "actual": str(size),
                },
            )
        if digest.lower() != str(asset["sha256"]).lower():
            return GooseInstallResult(
                available=False,
                method="verify",
                reason="digest_mismatch",
                asset_name=str(asset["asset_name"]),
                details={"expected": str(asset["sha256"]), "actual": digest},
            )

        try:
            extracted = _safe_extract_archive(
                archive_path, extract_dir, allowed_members=allowed
            )
        except ValueError as exc:
            text = str(exc).lower()
            if "traversal" in text or "disallowed" in text or "unsafe" in text:
                reason = "path_traversal"
            elif "malformed" in text:
                reason = "malformed_archive"
            else:
                reason = "extract_failed"
            return GooseInstallResult(
                available=False,
                method="extract",
                reason=reason,
                asset_name=str(asset["asset_name"]),
                details={"error": str(exc)[:500]},
            )

        # Version probe before promotion.
        version = _probe_version(
            str(extracted), run=run, environ=env, timeout_seconds=min(30.0, float(timeout_seconds))
        )
        expected = pinned.lstrip("v")
        if not version:
            return GooseInstallResult(
                available=False,
                method="version_probe",
                reason="version_probe_failed",
                asset_name=str(asset["asset_name"]),
            )
        if not version.lstrip("v").startswith(expected.split("-")[0]):
            return GooseInstallResult(
                available=False,
                method="version_probe",
                reason="wrong_version",
                version=version.lstrip("v"),
                asset_name=str(asset["asset_name"]),
                details={"expected": expected, "actual": version.lstrip("v")},
            )

        # Destination must remain under managed root.
        managed_root_path = dest.parent.parent.resolve()
        try:
            dest_resolved_parent = dest.parent.resolve()
            if not str(dest_resolved_parent).startswith(str(managed_root_path)):
                return GooseInstallResult(
                    available=False,
                    method="destination",
                    reason="destination_escape",
                    managed_path=str(dest),
                )
        except OSError:
            pass

        staging_bin = stage_root / "promote" / dest.name
        staging_bin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted, staging_bin)
        if os.name != "nt":
            mode_bits = staging_bin.stat().st_mode
            staging_bin.chmod(mode_bits | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        try:
            _atomic_replace_with_rollback(staging_bin, dest)
        except OSError as exc:
            return GooseInstallResult(
                available=False,
                method="promote",
                reason="atomic_replace_failed",
                managed_path=str(dest),
                details={"error": str(exc)[:500]},
            )

        if not _is_executable_file(dest):
            return GooseInstallResult(
                available=False,
                method="promote",
                reason="destination_not_executable",
                managed_path=str(dest),
            )

        return GooseInstallResult(
            available=True,
            installed=True,
            executable=str(dest),
            version=version.lstrip("v"),
            method="managed_install",
            reason="installed",
            asset_name=str(asset["asset_name"]),
            managed_path=str(dest),
            details={
                "os": os_n,
                "arch": arch_n,
                "libc": libc_n,
                "variant": variant_n,
                "sha256": str(asset["sha256"]),
            },
        )
    finally:
        if created_stage:
            shutil.rmtree(stage_root, ignore_errors=True)


def ensure_goose(
    *,
    auto_install: Optional[bool] = None,
    explicit_path: Optional[Union[str, Path]] = None,
    operator_argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    which: Optional[WhichFn] = None,
    run: Optional[RunFn] = None,
    download: Optional[DownloadFn] = None,
    managed_root: Optional[Union[str, Path]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    timeout_seconds: float = 300.0,
    os_name: Optional[str] = None,
    arch: Optional[str] = None,
    libc: Optional[str] = None,
    variant: Optional[str] = None,
    staging_dir: Optional[Union[str, Path]] = None,
    lock_path: Optional[Union[str, Path]] = None,
) -> GooseInstallResult:
    """Return an available Goose executable, installing when policy allows.

    Safe for concurrent callers: per-process and cross-process locks collapse
    races to a single installation. Importing this module never installs.
    """

    env = dict(os.environ if environ is None else environ)
    data = dict(manifest) if manifest is not None else load_release_manifest()

    found = discover_goose(
        explicit_path=explicit_path,
        operator_argv=operator_argv,
        environ=env,
        which=which,
        run=run,
        managed_root=managed_root,
        manifest=data,
        os_name=os_name,
    )
    if found.available:
        return found

    if not goose_auto_install_enabled(auto_install, environ=env):
        return GooseInstallResult(
            available=False,
            method="disabled",
            reason="auto_install_disabled",
            managed_path=found.managed_path,
        )

    lock_file = Path(lock_path) if lock_path is not None else None
    with _INSTALL_LOCK, _process_install_lock(lock_file):
        # Peer may have finished while we waited.
        found = discover_goose(
            explicit_path=explicit_path,
            operator_argv=operator_argv,
            environ=env,
            which=which,
            run=run,
            managed_root=managed_root,
            manifest=data,
            os_name=os_name,
        )
        if found.available:
            return GooseInstallResult(
                available=True,
                installed=False,
                executable=found.executable,
                version=found.version,
                method=found.method,
                reason="installed_by_peer",
                managed_path=found.managed_path,
            )

        result = install_goose_from_manifest(
            manifest=data,
            environ=env,
            managed_root=managed_root,
            download=download,
            run=run,
            timeout_seconds=timeout_seconds,
            os_name=os_name,
            arch=arch,
            libc=libc,
            variant=variant,
            staging_dir=staging_dir,
        )
        return result


__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "GOOSE_EXECUTABLE",
    "GOOSE_EXECUTABLE_WINDOWS",
    "LINUX_VARIANTS",
    "WINDOWS_VARIANTS",
    "GooseInstallResult",
    "GooseReadiness",
    "assess_goose_readiness",
    "default_manifest_path",
    "detect_linux_libc",
    "detect_platform",
    "discover_goose",
    "ensure_goose",
    "goose_auth_available",
    "goose_auto_install_enabled",
    "install_goose_from_manifest",
    "load_release_manifest",
    "managed_executable_path",
    "managed_install_root",
    "normalize_arch",
    "normalize_os",
    "select_release_asset",
    "validate_platform",
]
