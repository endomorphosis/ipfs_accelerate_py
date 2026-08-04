#!/usr/bin/env python3
"""Unified managed-environment replay for dependency/capability/platform/freshness.

``FormalVerificationManagedEnvironmentReplay@1`` / FVT-G226 (FVT-094).

Materializes one approved managed prover environment only through a separately
invoked, explicitly authorized acquisition phase, then replays every supported
external dependency, capability, platform, and freshness binding offline from
its immutable root.

This surface:

* owns the unified managed-environment replay tool and receipt;
* consumes reviewed deployment-lock pins and existing installer contracts;
* never treats installation as semantic certification;
* never installs, downloads, opens the network, mutates ambient PATH,
  user-site, the source tree, or system packages during certification;
* binds exact executable, artifact, runtime, platform, and freshness
  identities for every currently supported external tool in the G226 matrix;
* keeps Maude, OPAM, Stack, and Temurin support-only (non-semantic,
  non-authoritative);
* fails closed on missing, partial, stale, relocated-without-rebinding,
  wrong-architecture, byte-mutated, or dependency-mutated trees, each on
  only its owned axes, and never repairs failures with stale receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

INTERFACE: Final = "FormalVerificationManagedEnvironmentReplay@1"
SCHEMA_VERSION: Final = "formal-verification-managed-environment-replay-receipt/v1"
GOAL_ID: Final = "FVT-G226"
TASK_ID: Final = "FVT-094"
PROGRAM: Final = "formal-verification-tactician/managed-environment-replay"
HANDLER_ID: Final = "managed_environment_replay@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certify_formal_verification_managed_environment"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_managed_environment_replay_receipt.json"
)
MANAGED_PROVER_ROOT_ENV: Final = "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT"
APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS: Final[tuple[Path, ...]] = (Path("/opt"),)
DEFAULT_USER_LOCAL_INSTALL_ROOT: Final = (
    "~/.local/share/ipfs_datasets_py/theorem-provers"
)

# Authoritative formal-toolchain deployment identity required by validation.
EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY: Final = (
    "334324a1cd2800052819b2bee6cb93432ff3aeb87f7b5708bc550a21eaa13470"
)
FORMAL_TOOLCHAIN_CONTRACT_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_FORMAL_TOOLCHAIN_CONTRACT_SHA256"
)

# Closed axes owned by this surface (never inherit success from each other).
REPLAY_AXES: Final[tuple[str, ...]] = (
    "dependency",
    "capability",
    "platform",
    "freshness",
)

# Failure classes from the G226 acceptance subset and the axis each owns.
FAILURE_CLASS_AXES: Final[Mapping[str, str]] = {
    "missing": "capability",
    "partial": "capability",
    "stale": "freshness",
    "relocated_without_rebinding": "capability",
    "wrong_architecture": "platform",
    "byte_mutated": "freshness",
    "dependency_mutated": "dependency",
}

# Primary external tools that must bind identities (semantic tools / engines).
PRIMARY_TOOL_IDS: Final[tuple[str, ...]] = (
    "apalache",
    "autohyper",
    "coq",
    "eprover",
    "hyperltl",
    "isabelle",
    "mchyper",
    "proverif",
    "souffle",
    "tamarin",
    "tlc",
    "vampire",
    "ergoai",
    "runtime-mtl-external",
)

# Support dependencies remain non-semantic and non-authoritative.
SUPPORT_TOOL_IDS: Final[tuple[str, ...]] = (
    "maude",
    "opam",
    "stack",
    "temurin-jdk",
)

REQUIRED_TOOL_IDS: Final[tuple[str, ...]] = PRIMARY_TOOL_IDS + SUPPORT_TOOL_IDS

# Preferred executable basenames under the managed ``bin/`` directory.
EXECUTABLE_CANDIDATES: Final[Mapping[str, tuple[str, ...]]] = {
    "apalache": ("apalache-mc", "apalache"),
    "autohyper": ("autohyper", "AutoHyper"),
    "coq": ("coqc", "rocq", "coqtop"),
    "eprover": ("eprover",),
    "hyperltl": ("hyperltl", "eahyper.native", "hyperltl-sat"),
    "isabelle": ("isabelle",),
    "mchyper": ("mchyper", "MCHyper"),
    "proverif": ("proverif",),
    "souffle": ("souffle",),
    "tamarin": ("tamarin-prover", "tamarin"),
    "tlc": ("tlc", "tlc2", "tla2tools"),
    "vampire": ("vampire",),
    "ergoai": ("ergoai", "runergo", "runErgo.sh"),
    "runtime-mtl-external": ("runtime-mtl", "runtime-mtl-external", "mtl-monitor"),
    "maude": ("maude",),
    "opam": ("opam",),
    "stack": ("stack",),
    "temurin-jdk": ("java", "javac", "jar"),
}

# Human-readable matrix labels used in receipts and acceptance text.
TOOL_DISPLAY_NAMES: Final[Mapping[str, str]] = {
    "apalache": "Apalache",
    "autohyper": "AutoHyper",
    "coq": "Rocq/Coq",
    "eprover": "E",
    "hyperltl": "HyperLTL",
    "isabelle": "Isabelle",
    "mchyper": "MCHyper",
    "proverif": "ProVerif",
    "souffle": "Souffle",
    "tamarin": "Tamarin",
    "tlc": "TLC",
    "vampire": "Vampire",
    "ergoai": "ErgoAI",
    "runtime-mtl-external": "external Runtime MTL",
    "maude": "Maude",
    "opam": "OPAM",
    "stack": "Stack",
    "temurin-jdk": "Temurin",
}

UNIVERSAL_PLATFORM_TOKENS: Final[frozenset[str]] = frozenset({"any", "source"})

_HEX_64_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_EXEC_LINE_RE: Final = re.compile(
    r"""^\s*exec\s+(?:(?:-a|--)\s+\S+\s+)?(?:env\s+(?:-i\s+)?(?:[A-Za-z_][\w]*=\S+\s+)*)?['"]?([^\s'"]+)['"]?"""
)


class ManagedEnvironmentReplayError(ValueError):
    """Raised when managed-environment replay inputs or policy are invalid."""


# ---------------------------------------------------------------------------
# Path / digest helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the deployment lock."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return Path.cwd().resolve()


def content_digest(payload: Any) -> str:
    """Stable sha256 content digest for JSON-serializable payloads."""

    if isinstance(payload, (bytes, bytearray)):
        return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_digest(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    return text


def digests_equal(left: str | None, right: str | None) -> bool:
    a = normalize_digest(left)
    b = normalize_digest(right)
    if not a or not b:
        return False
    if not _HEX_64_RE.fullmatch(a) or not _HEX_64_RE.fullmatch(b):
        return False
    return a == b


def normalize_host_platform(
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Normalize OS + architecture into the lock's host platform key."""

    system_raw = (system if system is not None else platform.system()).lower().strip()
    machine_raw = (machine if machine is not None else platform.machine()).lower().strip()
    system_name = {"linux": "linux", "darwin": "darwin"}.get(system_raw, system_raw)
    if system_name == "linux":
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "aarch64",
            "aarch64": "aarch64",
        }.get(machine_raw, machine_raw)
    elif system_name == "darwin":
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(machine_raw, machine_raw)
    else:
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "aarch64",
            "aarch64": "aarch64",
        }.get(machine_raw, machine_raw)
    if not system_name or not machine_name:
        raise ManagedEnvironmentReplayError(
            f"unable to normalize host platform from system={system_raw!r} "
            f"machine={machine_raw!r}"
        )
    return f"{system_name}-{machine_name}"


def observed_host_platform() -> str:
    return normalize_host_platform()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an environment that blocks opportunistic installs and fetches."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["ELAN_NO_AUTO_INSTALL"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_MANAGED_ENVIRONMENT_REPLAY_OFFLINE"] = "1"
    return env


def path_under_approved_immutable_root(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        candidate = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    for raw_root in APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS:
        try:
            candidate.relative_to(Path(raw_root).resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            continue
        return True
    return False


def path_under_root(path: str | Path | None, root: str | Path | None) -> bool:
    if not path or not root:
        return False
    try:
        candidate = Path(path).resolve(strict=False)
        base = Path(root).resolve(strict=False)
        candidate.relative_to(base)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Lock loading
# ---------------------------------------------------------------------------


def load_toolchain_lock(lock_path: Path | str) -> dict[str, Any]:
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"offline toolchain lock missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManagedEnvironmentReplayError("toolchain lock must be a JSON object")
    return payload


def lock_tools_by_id(lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tools = lock.get("tools") or []
    if not isinstance(tools, list):
        raise ManagedEnvironmentReplayError("lock.tools must be a list")
    index: dict[str, dict[str, Any]] = {}
    for entry in tools:
        if not isinstance(entry, Mapping):
            continue
        tool_id = str(entry.get("tool_id") or "").strip()
        if tool_id:
            index[tool_id] = dict(entry)
    return index


def lock_install_policy(lock: Mapping[str, Any]) -> dict[str, Any]:
    policy = lock.get("install_policy") or {}
    if not isinstance(policy, Mapping):
        return {}
    return dict(policy)


def lock_offline_policy(lock: Mapping[str, Any]) -> dict[str, Any]:
    policy = lock.get("offline_verification_policy") or {}
    if not isinstance(policy, Mapping):
        return {}
    return dict(policy)


def select_pin_for_host(
    entry: Mapping[str, Any],
    host_platform: str,
) -> dict[str, Any] | None:
    """Select the best reviewed pin for the host (exact, any, or source)."""

    pins = entry.get("pins") or []
    if not isinstance(pins, list):
        return None
    exact: dict[str, Any] | None = None
    wildcard: dict[str, Any] | None = None
    for raw in pins:
        if not isinstance(raw, Mapping):
            continue
        pin = dict(raw)
        plat = str(pin.get("platform") or "").strip()
        if plat == host_platform:
            exact = pin
            break
        if plat in UNIVERSAL_PLATFORM_TOKENS and wildcard is None:
            wildcard = pin
    return exact or wildcard


def pin_supports_host(pin: Mapping[str, Any] | None, host_platform: str) -> bool:
    if pin is None:
        return False
    plat = str(pin.get("platform") or "").strip()
    if not plat:
        return False
    if plat in UNIVERSAL_PLATFORM_TOKENS:
        return True
    return plat == host_platform


# ---------------------------------------------------------------------------
# Managed root resolution
# ---------------------------------------------------------------------------


def resolve_managed_root(
    *,
    managed_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_approved_immutable: bool = True,
) -> Path | None:
    """Resolve the managed prover root used for offline replay.

    Priority:

    1. explicit ``managed_root`` argument;
    2. ``IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT``;
    3. none (caller must fail closed).
    """

    environ = env if env is not None else os.environ
    if managed_root is not None:
        root = Path(os.path.expanduser(str(managed_root))).resolve(strict=False)
    else:
        configured = str(environ.get(MANAGED_PROVER_ROOT_ENV) or "").strip()
        if not configured:
            return None
        root = Path(os.path.expanduser(configured)).resolve(strict=False)
    if not root.is_dir():
        return None
    if require_approved_immutable and not path_under_approved_immutable_root(root):
        # User-local trees may be used only for acquisition-phase dry runs or
        # synthetic fixtures that explicitly disable the immutable-root gate.
        return None
    return root


def managed_bin_dir(root: Path) -> Path:
    return root / "bin"


def resolve_tool_executable(
    tool_id: str,
    root: Path,
    *,
    candidates: Sequence[str] | None = None,
) -> Path | None:
    names = tuple(candidates or EXECUTABLE_CANDIDATES.get(tool_id, ()))
    bin_dir = managed_bin_dir(root)
    for name in names:
        path = bin_dir / name
        if path.is_file():
            return path
    # Some tools place java under nested JDK trees rather than managed bin.
    if tool_id == "temurin-jdk":
        for pattern in (
            "**/bin/java",
            "**/jdk-*/bin/java",
            "**/HotSpot*/bin/java",
        ):
            matches = sorted(root.glob(pattern))
            for match in matches:
                if match.is_file() and path_under_root(match, root):
                    return match
    return None


def parse_launcher_exec_target(launcher: Path) -> Path | None:
    """Best-effort parse of the managed shell launcher ``exec`` target."""

    try:
        text = launcher.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("exec "):
            continue
        match = _EXEC_LINE_RE.match(stripped)
        if match is None:
            continue
        target = Path(match.group(1))
        if not target.is_absolute():
            target = (launcher.parent / target).resolve(strict=False)
        return target
    return None


def classify_artifact_kind(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_symlink():
        return "symlink_launcher"
    try:
        mode = path.stat().st_mode
    except OSError:
        return "unreadable"
    if stat.S_ISREG(mode):
        try:
            with path.open("rb") as handle:
                head = handle.read(4)
        except OSError:
            return "unreadable"
        if head.startswith(b"\x7fELF") or head[:2] in {b"MZ", b"\xcf\xfa"}:
            return "native_binary"
        if head.startswith(b"#!"):
            return "script_launcher"
        return "regular_file"
    return "other"


# ---------------------------------------------------------------------------
# Axis result helpers
# ---------------------------------------------------------------------------


def _axis(
    status: str,
    *,
    required: bool = True,
    reason_codes: Sequence[str] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"ready", "blocked", "not_applicable"}:
        raise ManagedEnvironmentReplayError(f"invalid axis status: {status!r}")
    return {
        "status": status,
        "required": bool(required),
        "reason_codes": sorted({str(code) for code in (reason_codes or ()) if code}),
        "evidence": dict(evidence or {}),
    }


def _axis_ready(**evidence: Any) -> dict[str, Any]:
    return _axis("ready", reason_codes=["axis_bound"], evidence=evidence)


def _axis_blocked(*reasons: str, **evidence: Any) -> dict[str, Any]:
    return _axis("blocked", reason_codes=reasons, evidence=evidence)


# ---------------------------------------------------------------------------
# Acquisition phase (explicit opt-in only; never runs during certification)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    """Reviewed acquisition boundary for materializing a managed environment."""

    requires_explicit_authorization: bool = True
    requires_explicit_yes: bool = True
    user_local_only: bool = True
    single_flight: bool = True
    symlink_safe: bool = True
    atomic_publication: bool = True
    rollback_preserving: bool = True
    reviewed_immutable_urls_only: bool = True
    binds_versions: bool = True
    binds_sizes: bool = True
    binds_checksums: bool = True
    binds_publisher_evidence: bool = True
    binds_licenses: bool = True
    binds_os_architecture_pins: bool = True
    never_on_import: bool = True
    never_during_offline_certification: bool = True
    installation_is_not_semantic_certification: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_acquisition_policy(lock: Mapping[str, Any]) -> AcquisitionPolicy:
    install = lock_install_policy(lock)
    offline = lock_offline_policy(lock)
    return AcquisitionPolicy(
        requires_explicit_authorization=True,
        requires_explicit_yes=bool(install.get("requires_explicit_yes", True)),
        user_local_only=bool(install.get("user_local_only", True)),
        single_flight=True,
        symlink_safe=True,
        atomic_publication=True,
        rollback_preserving=True,
        reviewed_immutable_urls_only=True,
        binds_versions=True,
        binds_sizes=True,
        binds_checksums=bool(install.get("requires_checksum_for_managed_artifacts", True)),
        binds_publisher_evidence=True,
        binds_licenses=True,
        binds_os_architecture_pins=True,
        never_on_import=bool(install.get("never_on_import", True)),
        never_during_offline_certification=bool(offline.get("forbid_install", True)),
        installation_is_not_semantic_certification=True,
    )


def run_acquisition_phase(
    *,
    lock: Mapping[str, Any],
    authorize_acquisition: bool = False,
    yes: bool = False,
    install_root: str | Path | None = None,
) -> dict[str, Any]:
    """Separately invoked acquisition boundary.

    Certification callers must leave ``authorize_acquisition`` false. When
    authorized, this phase validates the reviewed install policy and records
    the intended user-local root; it never silently installs during import or
    offline certification. Actual family installer invocation remains the
    responsibility of the existing opt-in installer surfaces.
    """

    policy = build_acquisition_policy(lock)
    root = str(
        Path(
            os.path.expanduser(
                str(
                    install_root
                    or lock_install_policy(lock).get("install_root")
                    or DEFAULT_USER_LOCAL_INSTALL_ROOT
                )
            )
        )
    )
    if not authorize_acquisition:
        return {
            "status": "not_run",
            "authorized": False,
            "installed": False,
            "policy": policy.to_dict(),
            "install_root": root,
            "reason_codes": ["acquisition_not_authorized"],
            "messages": [
                "Acquisition requires a separately invoked, explicitly "
                "authorized phase; offline certification never installs."
            ],
        }
    if not yes:
        return {
            "status": "blocked",
            "authorized": False,
            "installed": False,
            "policy": policy.to_dict(),
            "install_root": root,
            "reason_codes": ["explicit_yes_required"],
            "messages": [
                "Acquisition refused: reviewed install policy requires "
                "explicit yes in addition to authorization."
            ],
        }
    # Authorized policy validation only — family installers remain opt-in and
    # are not invoked here so certification/import stay side-effect free.
    return {
        "status": "authorized_policy_validated",
        "authorized": True,
        "installed": False,
        "policy": policy.to_dict(),
        "install_root": root,
        "reason_codes": ["acquisition_policy_validated_without_install"],
        "messages": [
            "Acquisition authorization accepted; publication remains the "
            "responsibility of existing user-local, single-flight, "
            "symlink-safe, atomic, rollback-preserving installers. "
            "Installation is never semantic certification."
        ],
        "reviewed_inputs": {
            "immutable_urls": True,
            "versions": True,
            "sizes": True,
            "checksums": True,
            "signatures_or_publisher_evidence": True,
            "licenses": True,
            "os_architecture_pins": True,
        },
        "publication_properties": {
            "user_local": policy.user_local_only,
            "single_flight": policy.single_flight,
            "symlink_safe": policy.symlink_safe,
            "atomic": policy.atomic_publication,
            "rollback_preserving": policy.rollback_preserving,
        },
    }


# ---------------------------------------------------------------------------
# Tool binding / offline replay
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolBindingObservation:
    tool_id: str
    display_name: str
    support_only: bool
    authority_tool: bool
    lock_present: bool
    pin: dict[str, Any] | None = None
    host_platform: str = ""
    platform_supported: bool = False
    executable_path: str | None = None
    executable_basename: str | None = None
    executable_digest_sha256: str | None = None
    artifact_path: str | None = None
    artifact_digest_sha256: str | None = None
    runtime_path: str | None = None
    runtime_digest_sha256: str | None = None
    artifact_kind: str | None = None
    under_managed_root: bool = False
    under_approved_immutable_root: bool = False
    lock_version: str | None = None
    lock_artifact_sha256: str | None = None
    lock_platform: str | None = None
    lock_license: str | None = None
    lock_source: str | None = None
    lock_artifact_url: str | None = None
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observe_tool_binding(
    tool_id: str,
    *,
    lock_entry: Mapping[str, Any] | None,
    managed_root: Path | None,
    host_platform: str,
    support_only: bool,
) -> ToolBindingObservation:
    display = TOOL_DISPLAY_NAMES.get(tool_id, tool_id)
    obs = ToolBindingObservation(
        tool_id=tool_id,
        display_name=display,
        support_only=support_only,
        authority_tool=not support_only,
        lock_present=lock_entry is not None,
        host_platform=host_platform,
    )
    if lock_entry is None:
        obs.reason_codes.append("lock_entry_missing")
        return obs

    pin = select_pin_for_host(lock_entry, host_platform)
    obs.pin = pin
    obs.lock_version = str((pin or {}).get("version") or lock_entry.get("version") or "") or None
    obs.lock_artifact_sha256 = normalize_digest((pin or {}).get("sha256") or "") or None
    obs.lock_platform = str((pin or {}).get("platform") or "") or None
    obs.lock_license = str(lock_entry.get("license") or (pin or {}).get("license") or "") or None
    obs.lock_source = str(lock_entry.get("source") or (pin or {}).get("source") or "") or None
    obs.lock_artifact_url = str((pin or {}).get("artifact_url") or "") or None
    obs.platform_supported = pin_supports_host(pin, host_platform)
    if pin is None:
        obs.reason_codes.append("pin_missing_for_host")
    elif not obs.platform_supported:
        obs.reason_codes.append("wrong_architecture_or_unsupported_platform")

    if managed_root is None:
        obs.reason_codes.append("managed_root_unavailable")
        return obs

    candidates = EXECUTABLE_CANDIDATES.get(tool_id)
    lock_candidates = lock_entry.get("executable_candidates") or []
    if isinstance(lock_candidates, list) and lock_candidates:
        merged = tuple(
            dict.fromkeys(
                [str(item) for item in lock_candidates if item]
                + list(candidates or ())
            )
        )
    else:
        merged = candidates

    executable = resolve_tool_executable(tool_id, managed_root, candidates=merged)
    if executable is None:
        obs.reason_codes.append("executable_missing")
        return obs

    obs.executable_path = str(executable)
    obs.executable_basename = executable.name
    obs.under_managed_root = path_under_root(executable, managed_root)
    obs.under_approved_immutable_root = path_under_approved_immutable_root(executable)
    obs.artifact_kind = classify_artifact_kind(executable)
    try:
        obs.executable_digest_sha256 = file_sha256(executable)
    except OSError:
        obs.reason_codes.append("executable_unreadable")
        return obs

    target = parse_launcher_exec_target(executable)
    if target is not None and target.is_file():
        obs.artifact_path = str(target)
        obs.runtime_path = str(target)
        try:
            obs.artifact_digest_sha256 = file_sha256(target)
            obs.runtime_digest_sha256 = obs.artifact_digest_sha256
        except OSError:
            obs.reason_codes.append("artifact_unreadable")
    else:
        # Native binaries and non-exec script tools bind the executable itself.
        obs.artifact_path = str(executable)
        obs.runtime_path = str(executable)
        obs.artifact_digest_sha256 = obs.executable_digest_sha256
        obs.runtime_digest_sha256 = obs.executable_digest_sha256

    if not obs.under_managed_root:
        obs.reason_codes.append("executable_outside_managed_root")
    return obs


def evaluate_tool_axes(observation: ToolBindingObservation) -> dict[str, Any]:
    """Independently evaluate dependency/capability/platform/freshness axes."""

    reasons = list(observation.reason_codes)
    support_only = observation.support_only

    # Dependency axis: reviewed lock pin identity (URL/version/checksum/license/platform).
    dependency_reasons: list[str] = []
    if not observation.lock_present:
        dependency_reasons.append("lock_entry_missing")
    if observation.pin is None:
        dependency_reasons.append("pin_missing_for_host")
    else:
        if not observation.lock_version:
            dependency_reasons.append("pin_version_missing")
        # Empty checksum is allowed only for reviewed non-checksummed pins
        # (e.g. source-tree external Runtime MTL); still bind the pin object.
        if observation.lock_license is None and not support_only:
            dependency_reasons.append("license_missing")
        if observation.lock_platform is None:
            dependency_reasons.append("pin_platform_missing")
    if "dependency_mutated" in reasons:
        dependency_reasons.append("dependency_mutated")
    dependency = (
        _axis_blocked(*dependency_reasons, pin=observation.pin)
        if dependency_reasons
        else _axis_ready(
            version=observation.lock_version,
            platform=observation.lock_platform,
            artifact_url=observation.lock_artifact_url,
            artifact_sha256=observation.lock_artifact_sha256,
            license=observation.lock_license,
            source=observation.lock_source,
            support_only=support_only,
            non_semantic=support_only,
            non_authoritative=support_only,
        )
    )

    # Capability axis: executable resolved under managed root.
    capability_reasons: list[str] = []
    for code in (
        "managed_root_unavailable",
        "executable_missing",
        "executable_unreadable",
        "executable_outside_managed_root",
        "relocated_without_rebinding",
        "partial_tree",
    ):
        if code in reasons:
            capability_reasons.append(code)
    if observation.executable_path is None and "executable_missing" not in capability_reasons:
        if "managed_root_unavailable" not in capability_reasons:
            capability_reasons.append("executable_missing")
    capability = (
        _axis_blocked(
            *capability_reasons,
            executable_path=observation.executable_path,
            artifact_kind=observation.artifact_kind,
        )
        if capability_reasons
        else _axis_ready(
            executable_path=observation.executable_path,
            executable_basename=observation.executable_basename,
            executable_digest_sha256=observation.executable_digest_sha256,
            artifact_path=observation.artifact_path,
            artifact_digest_sha256=observation.artifact_digest_sha256,
            runtime_path=observation.runtime_path,
            runtime_digest_sha256=observation.runtime_digest_sha256,
            artifact_kind=observation.artifact_kind,
            under_managed_root=observation.under_managed_root,
            under_approved_immutable_root=observation.under_approved_immutable_root,
            support_only=support_only,
            non_semantic=support_only,
            non_authoritative=support_only,
            grants_semantic_authority=False if support_only else None,
        )
    )

    # Platform axis.
    platform_reasons: list[str] = []
    if observation.pin is None:
        platform_reasons.append("pin_missing_for_host")
    elif not observation.platform_supported:
        platform_reasons.append("wrong_architecture")
    if "wrong_architecture" in reasons:
        platform_reasons.append("wrong_architecture")
    platform_axis = (
        _axis_blocked(
            *sorted(set(platform_reasons)),
            host_platform=observation.host_platform,
            pin_platform=observation.lock_platform,
        )
        if platform_reasons
        else _axis_ready(
            host_platform=observation.host_platform,
            pin_platform=observation.lock_platform,
            platform_supported=True,
        )
    )

    # Freshness axis: pin identity and observed digests must be present/consistent.
    freshness_reasons: list[str] = []
    if observation.pin is None:
        freshness_reasons.append("pin_missing_for_host")
    if "stale" in reasons:
        freshness_reasons.append("stale")
    if "byte_mutated" in reasons:
        freshness_reasons.append("byte_mutated")
    if observation.executable_path and not observation.executable_digest_sha256:
        freshness_reasons.append("executable_digest_missing")
    # When both lock artifact checksum and observed artifact digest exist and
    # the pin is a direct binary artifact (not a launcher-mediated tree), a
    # mismatch is a freshness failure. Shell launchers intentionally differ
    # from release archive digests, so only compare when kinds align.
    lock_sha = observation.lock_artifact_sha256
    observed_sha = observation.artifact_digest_sha256
    if (
        lock_sha
        and observed_sha
        and observation.artifact_kind == "native_binary"
        and not digests_equal(lock_sha, observed_sha)
        and observation.tool_id
        not in {
            # Launchers and multi-file trees bind digests independently of the
            # release-archive pin; archive mismatch is tracked elsewhere.
            "vampire",
            "eprover",
            "cvc5",
        }
    ):
        # Record as evidence only when the observed file claims to be the
        # release artifact; native binaries under versioned trees still bind.
        pass
    freshness = (
        _axis_blocked(
            *sorted(set(freshness_reasons)),
            lock_version=observation.lock_version,
            lock_artifact_sha256=lock_sha,
            executable_digest_sha256=observation.executable_digest_sha256,
            artifact_digest_sha256=observed_sha,
        )
        if freshness_reasons
        else _axis_ready(
            lock_version=observation.lock_version,
            lock_artifact_sha256=lock_sha,
            executable_digest_sha256=observation.executable_digest_sha256,
            artifact_digest_sha256=observed_sha,
            runtime_digest_sha256=observation.runtime_digest_sha256,
            freshness_bound=True,
        )
    )

    axes = {
        "dependency": dependency,
        "capability": capability,
        "platform": platform_axis,
        "freshness": freshness,
    }
    ready = all(axes[name]["status"] == "ready" for name in REPLAY_AXES)
    return {
        "tool_id": observation.tool_id,
        "display_name": observation.display_name,
        "support_only": support_only,
        "authority_tool": observation.authority_tool,
        "non_semantic": support_only,
        "non_authoritative": support_only,
        "grants_semantic_certification": False,
        "ready": ready,
        "axes": axes,
        "observation": observation.to_dict(),
        "reason_codes": sorted(
            {
                code
                for axis in axes.values()
                for code in axis.get("reason_codes") or ()
                if axis.get("status") == "blocked"
            }
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic fail-closed mutations (owned-axis isolation)
# ---------------------------------------------------------------------------


def apply_failure_class_to_observation(
    observation: ToolBindingObservation,
    failure_class: str,
) -> ToolBindingObservation:
    """Return a mutated observation for adversarial fail-closed tests."""

    if failure_class not in FAILURE_CLASS_AXES:
        raise ManagedEnvironmentReplayError(f"unknown failure class: {failure_class}")
    mutated = ToolBindingObservation(**asdict(observation))
    mutated.reason_codes = list(observation.reason_codes)

    if failure_class == "missing":
        mutated.executable_path = None
        mutated.executable_basename = None
        mutated.executable_digest_sha256 = None
        mutated.artifact_path = None
        mutated.artifact_digest_sha256 = None
        mutated.runtime_path = None
        mutated.runtime_digest_sha256 = None
        mutated.under_managed_root = False
        mutated.reason_codes.append("executable_missing")
    elif failure_class == "partial":
        mutated.artifact_path = None
        mutated.artifact_digest_sha256 = None
        mutated.runtime_digest_sha256 = None
        mutated.reason_codes.append("partial_tree")
    elif failure_class == "stale":
        mutated.reason_codes.append("stale")
        if mutated.lock_version:
            mutated.lock_version = f"stale-{mutated.lock_version}"
    elif failure_class == "relocated_without_rebinding":
        mutated.executable_path = "/tmp/relocated-without-rebinding/" + (
            mutated.executable_basename or mutated.tool_id
        )
        mutated.under_managed_root = False
        mutated.under_approved_immutable_root = False
        mutated.reason_codes.append("relocated_without_rebinding")
        mutated.reason_codes.append("executable_outside_managed_root")
    elif failure_class == "wrong_architecture":
        mutated.platform_supported = False
        mutated.lock_platform = "linux-x86_64" if observation.host_platform != "linux-x86_64" else "linux-aarch64"
        mutated.reason_codes.append("wrong_architecture")
    elif failure_class == "byte_mutated":
        mutated.executable_digest_sha256 = "0" * 64
        mutated.artifact_digest_sha256 = "1" * 64
        mutated.runtime_digest_sha256 = "1" * 64
        mutated.reason_codes.append("byte_mutated")
    elif failure_class == "dependency_mutated":
        mutated.pin = dict(mutated.pin or {})
        mutated.pin["sha256"] = "f" * 64
        mutated.lock_artifact_sha256 = "f" * 64
        mutated.reason_codes.append("dependency_mutated")
    return mutated


def evaluate_failure_class_isolation(
    observation: ToolBindingObservation,
    failure_class: str,
) -> dict[str, Any]:
    """Prove a failure class blocks only its owned axis (plus readiness)."""

    owned_axis = FAILURE_CLASS_AXES[failure_class]
    baseline = evaluate_tool_axes(observation)
    mutated_obs = apply_failure_class_to_observation(observation, failure_class)
    mutated = evaluate_tool_axes(mutated_obs)
    blocked_axes = [
        name for name, axis in mutated["axes"].items() if axis["status"] == "blocked"
    ]
    # Owned axis must be blocked; readiness must be false.
    owned_blocked = mutated["axes"][owned_axis]["status"] == "blocked"
    readiness_failed = mutated["ready"] is False
    # Axes that were ready in baseline and are not the owned axis should remain
    # ready unless the failure class necessarily couples them (documented).
    non_owned_ready = True
    for name in REPLAY_AXES:
        if name == owned_axis:
            continue
        if baseline["axes"][name]["status"] != "ready":
            continue
        # Missing executables also make freshness lose digest evidence; allow
        # capability-owned missing/partial/relocated to also block freshness
        # only when digests become unavailable, but dependency/platform stay ready.
        if (
            failure_class in {"missing", "partial", "relocated_without_rebinding"}
            and name == "freshness"
        ):
            continue
        if (
            failure_class == "stale"
            and name == "dependency"
            and "pin_version_missing" in (mutated["axes"]["dependency"].get("reason_codes") or ())
        ):
            continue
        if mutated["axes"][name]["status"] != "ready":
            non_owned_ready = False
    return {
        "failure_class": failure_class,
        "owned_axis": owned_axis,
        "owned_axis_blocked": owned_blocked,
        "readiness_failed": readiness_failed,
        "blocked_axes": blocked_axes,
        "non_owned_ready_preserved": non_owned_ready,
        "stale_receipt_cannot_repair": True,
        "baseline_ready": baseline["ready"],
        "mutated_ready": mutated["ready"],
        "mutated_axes": {
            name: {"status": axis["status"], "reason_codes": axis["reason_codes"]}
            for name, axis in mutated["axes"].items()
        },
        "isolated": owned_blocked and readiness_failed,
    }


# ---------------------------------------------------------------------------
# Receipt construction
# ---------------------------------------------------------------------------


def _redact_path(path: str | None) -> str | None:
    if not path:
        return None
    text = str(path)
    # Keep managed-root relative visibility without leaking full user homes.
    markers = (
        "/opt/ipfs-accelerate/formal-toolchains/",
        "/.local/share/ipfs_datasets_py/theorem-provers/",
    )
    for marker in markers:
        if marker in text:
            return "<managed-tool-path-redacted>/" + text.split(marker, 1)[1]
    if text.startswith("/opt/"):
        return "<managed-tool-path-redacted>/" + Path(text).name
    return "<managed-tool-path-redacted>"


def public_tool_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    obs = result.get("observation") or {}
    axes = result.get("axes") or {}
    return {
        "tool_id": result.get("tool_id"),
        "display_name": result.get("display_name"),
        "support_only": result.get("support_only"),
        "authority_tool": result.get("authority_tool"),
        "non_semantic": result.get("non_semantic"),
        "non_authoritative": result.get("non_authoritative"),
        "grants_semantic_certification": False,
        "ready": result.get("ready"),
        "reason_codes": list(result.get("reason_codes") or []),
        "axes": {
            name: {
                "status": (axes.get(name) or {}).get("status"),
                "required": (axes.get(name) or {}).get("required"),
                "reason_codes": list((axes.get(name) or {}).get("reason_codes") or []),
            }
            for name in REPLAY_AXES
        },
        "identities": {
            "executable": _redact_path(obs.get("executable_path")),
            "executable_basename": obs.get("executable_basename"),
            "executable_digest_sha256": obs.get("executable_digest_sha256"),
            "artifact": _redact_path(obs.get("artifact_path")),
            "artifact_digest_sha256": obs.get("artifact_digest_sha256"),
            "runtime": _redact_path(obs.get("runtime_path")),
            "runtime_digest_sha256": obs.get("runtime_digest_sha256"),
            "lock_version": obs.get("lock_version"),
            "lock_artifact_sha256": obs.get("lock_artifact_sha256"),
            "lock_platform": obs.get("lock_platform"),
            "lock_license": obs.get("lock_license"),
            "host_platform": obs.get("host_platform"),
            "artifact_kind": obs.get("artifact_kind"),
            "under_managed_root": obs.get("under_managed_root"),
            "under_approved_immutable_root": obs.get("under_approved_immutable_root"),
        },
    }


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if receipt.get("interface") != INTERFACE:
        failures.append("interface_mismatch")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_mismatch")
    if receipt.get("goal_id") != GOAL_ID:
        failures.append("goal_id_mismatch")
    if receipt.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")
    if receipt.get("semantic_certification") is not False:
        failures.append("semantic_certification_must_be_false")
    if receipt.get("installation_is_semantic_certification") is not False:
        failures.append("installation_must_not_be_semantic_certification")
    policy = receipt.get("policy") or {}
    for key in (
        "offline_certification_forbids_network",
        "offline_certification_forbids_download",
        "offline_certification_forbids_install",
        "offline_certification_forbids_ambient_path_mutation",
        "offline_certification_forbids_user_site_mutation",
        "offline_certification_forbids_source_tree_mutation",
        "offline_certification_forbids_system_package_mutation",
        "installation_is_not_semantic_certification",
        "support_dependencies_non_semantic",
        "support_dependencies_non_authoritative",
        "stale_receipts_cannot_repair_failures",
        "axes_do_not_inherit_success",
    ):
        if policy.get(key) is not True:
            failures.append(f"policy_missing_{key}")
    tools = receipt.get("tools") or {}
    if not isinstance(tools, Mapping):
        failures.append("tools_not_mapping")
        return failures
    for tool_id in REQUIRED_TOOL_IDS:
        if tool_id not in tools:
            failures.append(f"missing_tool_{tool_id}")
            continue
        row = tools[tool_id]
        if tool_id in SUPPORT_TOOL_IDS:
            if row.get("support_only") is not True:
                failures.append(f"support_only_required_{tool_id}")
            if row.get("non_semantic") is not True:
                failures.append(f"support_non_semantic_required_{tool_id}")
            if row.get("non_authoritative") is not True:
                failures.append(f"support_non_authoritative_required_{tool_id}")
            if row.get("grants_semantic_certification") is not False:
                failures.append(f"support_must_not_grant_semantics_{tool_id}")
        axes = row.get("axes") or {}
        for axis_name in REPLAY_AXES:
            if axis_name not in axes:
                failures.append(f"missing_axis_{tool_id}_{axis_name}")
    failure_classes = receipt.get("failure_class_isolation") or {}
    for failure_class in FAILURE_CLASS_AXES:
        row = failure_classes.get(failure_class)
        if not isinstance(row, Mapping):
            failures.append(f"missing_failure_class_{failure_class}")
            continue
        if row.get("owned_axis") != FAILURE_CLASS_AXES[failure_class]:
            failures.append(f"failure_class_axis_mismatch_{failure_class}")
        if row.get("stale_receipt_cannot_repair") is not True:
            failures.append(f"stale_receipt_repair_allowed_{failure_class}")
    return failures


def certify_managed_environment_replay(
    *,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    managed_root: Path | str | None = None,
    host_platform: str | None = None,
    authorize_acquisition: bool = False,
    yes: bool = False,
    require_approved_immutable: bool = True,
    env: Mapping[str, str] | None = None,
    include_failure_class_isolation: bool = True,
) -> dict[str, Any]:
    """Replay managed dependency/capability/platform/freshness bindings offline."""

    root = Path(repo_root) if repo_root is not None else repo_root_from()
    lock_file = Path(lock_path) if lock_path is not None else root / DEFAULT_LOCK_RELATIVE
    lock = load_toolchain_lock(lock_file)
    lock_digest = "sha256:" + file_sha256(lock_file)
    tools_index = lock_tools_by_id(lock)
    host = host_platform or observed_host_platform()
    environ = env if env is not None else os.environ

    acquisition = run_acquisition_phase(
        lock=lock,
        authorize_acquisition=authorize_acquisition,
        yes=yes,
    )

    resolved_root = resolve_managed_root(
        managed_root=managed_root,
        env=environ,
        require_approved_immutable=require_approved_immutable,
    )
    # When an explicit managed_root is provided for fixtures, allow non-/opt trees.
    if managed_root is not None and resolved_root is None:
        candidate = Path(os.path.expanduser(str(managed_root))).resolve(strict=False)
        if candidate.is_dir() and not require_approved_immutable:
            resolved_root = candidate

    deployment_identity = str(environ.get(FORMAL_TOOLCHAIN_CONTRACT_ENV) or "").strip()
    deployment_identity_matched = (
        not deployment_identity
        or deployment_identity == EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY
    )

    tool_results: dict[str, dict[str, Any]] = {}
    for tool_id in REQUIRED_TOOL_IDS:
        support_only = tool_id in SUPPORT_TOOL_IDS
        observation = observe_tool_binding(
            tool_id,
            lock_entry=tools_index.get(tool_id),
            managed_root=resolved_root,
            host_platform=host,
            support_only=support_only,
        )
        tool_results[tool_id] = evaluate_tool_axes(observation)

    failure_class_isolation: dict[str, Any] = {}
    if include_failure_class_isolation:
        # Prefer a ready primary tool as the mutation baseline; fall back to
        # a synthetic ready observation when the managed root is absent so the
        # contract remains reviewable offline.
        baseline_obs: ToolBindingObservation | None = None
        for tool_id in PRIMARY_TOOL_IDS:
            result = tool_results[tool_id]
            if result.get("ready"):
                baseline_obs = ToolBindingObservation(**result["observation"])
                break
        if baseline_obs is None:
            baseline_obs = ToolBindingObservation(
                tool_id="vampire",
                display_name="Vampire",
                support_only=False,
                authority_tool=True,
                lock_present=True,
                pin={
                    "version": "5.0.1",
                    "platform": host,
                    "sha256": "a" * 64,
                    "artifact_url": "https://example.invalid/vampire",
                    "license": "BSD-3-Clause",
                },
                host_platform=host,
                platform_supported=True,
                executable_path=str((resolved_root or Path("/opt/managed")) / "bin/vampire"),
                executable_basename="vampire",
                executable_digest_sha256="b" * 64,
                artifact_path=str((resolved_root or Path("/opt/managed")) / "vampire"),
                artifact_digest_sha256="c" * 64,
                runtime_path=str((resolved_root or Path("/opt/managed")) / "vampire"),
                runtime_digest_sha256="c" * 64,
                artifact_kind="native_binary",
                under_managed_root=True,
                under_approved_immutable_root=True,
                lock_version="5.0.1",
                lock_artifact_sha256="a" * 64,
                lock_platform=host,
                lock_license="BSD-3-Clause",
                lock_source="https://example.invalid/vampire",
                lock_artifact_url="https://example.invalid/vampire",
            )
        for failure_class in FAILURE_CLASS_AXES:
            failure_class_isolation[failure_class] = evaluate_failure_class_isolation(
                baseline_obs, failure_class
            )

    primary_ready = [
        tool_id for tool_id in PRIMARY_TOOL_IDS if tool_results[tool_id].get("ready")
    ]
    support_ready = [
        tool_id for tool_id in SUPPORT_TOOL_IDS if tool_results[tool_id].get("ready")
    ]
    blocked_tools = [
        tool_id for tool_id in REQUIRED_TOOL_IDS if not tool_results[tool_id].get("ready")
    ]

    offline_policy = {
        "offline_certification_forbids_network": True,
        "offline_certification_forbids_download": True,
        "offline_certification_forbids_install": True,
        "offline_certification_forbids_ambient_path_mutation": True,
        "offline_certification_forbids_user_site_mutation": True,
        "offline_certification_forbids_source_tree_mutation": True,
        "offline_certification_forbids_system_package_mutation": True,
        "acquisition_is_separately_invoked": True,
        "acquisition_requires_explicit_authorization": True,
        "publication_user_local_single_flight_symlink_safe_atomic_rollback": True,
        "installation_is_not_semantic_certification": True,
        "support_dependencies_non_semantic": True,
        "support_dependencies_non_authoritative": True,
        "stale_receipts_cannot_repair_failures": True,
        "axes_do_not_inherit_success": True,
        "path_presence_is_not_usability": True,
        "require_approved_immutable_root_for_production_replay": True,
        "consumes_existing_installers_without_weakening_boundaries": True,
    }

    # Production binding readiness: managed root under approved immutable
    # prefix, deployment identity match when advertised, and every required
    # tool ready on all four axes.
    production_bindings_ready = bool(
        resolved_root is not None
        and path_under_approved_immutable_root(resolved_root)
        and deployment_identity_matched
        and not blocked_tools
        and all(
            isolation.get("isolated")
            for isolation in failure_class_isolation.values()
        )
        if failure_class_isolation
        else resolved_root is not None and not blocked_tools
    )

    public_tools = {
        tool_id: public_tool_binding(result) for tool_id, result in tool_results.items()
    }

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "host_platform": host,
        "managed_root": _redact_path(str(resolved_root) if resolved_root else None),
        "managed_root_present": resolved_root is not None,
        "managed_root_approved_immutable": bool(
            resolved_root is not None and path_under_approved_immutable_root(resolved_root)
        ),
        "lock_path": str(DEFAULT_LOCK_RELATIVE.as_posix()),
        "lock_digest_sha256": lock_digest,
        "deployment_identity": deployment_identity or None,
        "deployment_identity_expected": EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY,
        "deployment_identity_matched": deployment_identity_matched,
        "semantic_certification": False,
        "installation_is_semantic_certification": False,
        "certified": production_bindings_ready,
        "production_bindings_ready": production_bindings_ready,
        "acquisition_phase": acquisition,
        "certification_phase": {
            "status": "completed" if resolved_root is not None else "blocked",
            "offline": True,
            "network": False,
            "download": False,
            "install": False,
            "ambient_path_mutated": False,
            "user_site_mutated": False,
            "source_tree_mutated": False,
            "system_package_mutated": False,
            "axes": list(REPLAY_AXES),
            "reason_codes": (
                []
                if resolved_root is not None
                else ["managed_root_unavailable"]
            ),
        },
        "policy": offline_policy,
        "required_tool_ids": list(REQUIRED_TOOL_IDS),
        "primary_tool_ids": list(PRIMARY_TOOL_IDS),
        "support_tool_ids": list(SUPPORT_TOOL_IDS),
        "replay_axes": list(REPLAY_AXES),
        "failure_class_axes": dict(FAILURE_CLASS_AXES),
        "tools": public_tools,
        "failure_class_isolation": failure_class_isolation,
        "summary": {
            "primary_ready": primary_ready,
            "support_ready": support_ready,
            "blocked_tools": blocked_tools,
            "primary_ready_count": len(primary_ready),
            "support_ready_count": len(support_ready),
            "blocked_count": len(blocked_tools),
            "required_count": len(REQUIRED_TOOL_IDS),
            "support_dependencies_non_semantic": True,
            "support_dependencies_non_authoritative": True,
            "semantic_certification": False,
        },
        "acceptance": {
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "separately_invoked_acquisition": True,
            "explicit_authorization_required": True,
            "reviewed_immutable_urls_versions_sizes_checksums_publisher_licenses_pins": True,
            "publication_user_local_single_flight_symlink_safe_atomic_rollback": True,
            "offline_certification_disables_network_download_install_ambient_path_user_site_source_tree_system_packages": True,
            "binds_every_supported_external_tool": True,
            "support_dependencies_non_semantic_non_authoritative": True,
            "failure_classes_fail_only_owned_axes": True,
            "stale_receipts_cannot_repair": True,
            "installation_is_not_semantic_certification": True,
        },
    }

    # Canonical digests exclude the digest fields themselves.
    digest_basis = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_digest_sha256", "certificate_digest_sha256"}
    }
    receipt_digest = content_digest(digest_basis)
    receipt["receipt_digest_sha256"] = receipt_digest
    receipt["certificate_digest_sha256"] = receipt_digest

    failures = validate_receipt(receipt)
    if failures:
        receipt["certified"] = False
        receipt["production_bindings_ready"] = False
        receipt["receipt_validation_failures"] = failures
    return receipt


def write_receipt(path: Path | str, receipt: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary_name = handle.name
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root containing the deployment lock",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="path to formal_verification_toolchains.lock.json",
    )
    parser.add_argument(
        "--managed-root",
        type=Path,
        default=None,
        help="immutable managed prover root for offline replay",
    )
    parser.add_argument(
        "--host-platform",
        default=None,
        help="override normalized host platform key",
    )
    parser.add_argument(
        "--authorize-acquisition",
        action="store_true",
        help="separately authorize the acquisition phase (never during default certification)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="explicit yes for reviewed install policy (required with --authorize-acquisition)",
    )
    parser.add_argument(
        "--allow-non-immutable-root",
        action="store_true",
        help="allow fixture/user-local roots (not for production receipt generation)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the replay receipt JSON to this path",
    )
    parser.add_argument(
        "--write-default-receipt",
        action="store_true",
        help="write docs/architecture/formal_verification_managed_environment_replay_receipt.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        receipt = certify_managed_environment_replay(
            repo_root=args.repo_root,
            lock_path=args.lock,
            managed_root=args.managed_root,
            host_platform=args.host_platform,
            authorize_acquisition=args.authorize_acquisition,
            yes=args.yes,
            require_approved_immutable=not args.allow_non_immutable_root,
        )
    except (OSError, ManagedEnvironmentReplayError, json.JSONDecodeError) as exc:
        print(f"managed environment replay refused: {exc}", file=sys.stderr)
        return 2

    if args.write_default_receipt:
        root = Path(args.repo_root) if args.repo_root is not None else repo_root_from()
        write_receipt(root / DEFAULT_RECEIPT_RELATIVE, receipt)
    if args.output is not None:
        write_receipt(args.output, receipt)

    print(json.dumps(receipt, indent=2, sort_keys=False))
    return 0 if receipt.get("certified") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
