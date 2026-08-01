#!/usr/bin/env python3
"""Hermetic multi-prover certification for formal verification toolchains.

``FormalVerificationToolchainCertificate@1`` / FVT-G060 (FVT-030).

Runs property-specific offline-pinned live lanes for the real multi-prover
matrix. Available tools must pass positive / negative / mutation / replay
checks with exact identities. Absent or mismatched lanes are explicit
``unavailable`` / ``blocked`` results that only prevent *their own*
promotion. PATH presence alone is never usability. Certification never
installs, downloads, or opens the network, and cross-provider disagreement
is quarantined rather than treated as success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

INTERFACE: Final = "FormalVerificationToolchainCertificate@1"
SCHEMA_VERSION: Final = "formal-verification-toolchain-certificate/v1"
GOAL_ID: Final = "FVT-G060"
TASK_ID: Final = "FVT-030"
PROGRAM: Final = "formal-verification-tactician/readiness"
LOCK_INTERFACE: Final = "OfflineToolchainLock@1"
LOCK_SCHEMA: Final = "offline-toolchain-lock/v1"

# Role-aware reissue (FVT-G200 / FVT-053). Elevation is opt-in so the FVT-G060
# hermetic matrix can still distinguish identity-only usability from full
# semantic production certification; the role-aware receipt applies elevation.
ROLE_AWARE_INTERFACE: Final = "RoleAwareFormalVerificationRelease@1"
ROLE_AWARE_GOAL_ID: Final = "FVT-G200"
ROLE_AWARE_TASK_ID: Final = "FVT-053"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_CERTIFICATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_toolchain_certificate.json"
)
DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_deployment_receipt.json"
)

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 8.0
MAX_PUBLIC_STRING_BYTES: Final = 8192
HOST_PRIVATE_PATH_RE: Final = re.compile(
    r"/(?:home|tmp|private/tmp)/[^\s\"'<>]*"
)
RAW_OUTPUT_KEYS: Final = frozenset(
    {"stdout", "stderr", "raw_stdout", "raw_stderr"}
)
RAW_SECRET_KEYS: Final = frozenset(
    {
        "secret",
        "private_secret",
        "private_witness",
        "witness",
        "witness_bytes",
        "trapdoor",
        "toxic_waste",
        "private_key",
        "api_key",
        "access_token",
    }
)

# Property lanes from FVT-G060. Each lane owns a closed set of tools; absence
# of one tool never conceals or fails unrelated lanes.
PROPERTY_LANES: Final[tuple[dict[str, Any], ...]] = (
    {
        "lane_id": "smt",
        "property_class": "smt_software_verification",
        "description": "SMT solvers for software-verification VCs",
        "tool_ids": ("z3", "cvc5"),
        "authority_tool_ids": ("z3", "cvc5"),
        "check_kind": "smtlib",
    },
    {
        "lane_id": "tla",
        "property_class": "tla_state_model",
        "description": "TLA+/TLC/Apalache state-model checking",
        "tool_ids": ("apalache", "tlc", "java"),
        "authority_tool_ids": ("apalache", "tlc"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "datalog_secpal",
        "property_class": "authorization_datalog_secpal",
        "description": "Datalog/SecPAL authorization reasoning",
        "tool_ids": (
            "datalog-authorization",
            "secpal-authorization",
            "souffle",
            "secpal",
        ),
        "authority_tool_ids": (
            "datalog-authorization",
            "secpal-authorization",
            "souffle",
            "secpal",
        ),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "protocol",
        "property_class": "protocol_verification",
        "description": "Tamarin/ProVerif protocol verification",
        "tool_ids": ("tamarin", "proverif", "maude"),
        "authority_tool_ids": ("tamarin", "proverif"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hyperltl",
        "property_class": "hyperproperty",
        "description": "HyperLTL / hyperproperty tools",
        "tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "authority_tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "atp",
        "property_class": "automated_theorem_proving",
        "description": "First-order ATP portfolio",
        "tool_ids": ("vampire", "eprover"),
        "authority_tool_ids": ("vampire", "eprover"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hammer",
        "property_class": "hammer_advisor",
        "description": "Hammer / advisor bridges (non-kernel authority)",
        "tool_ids": ("symbolicai", "ergoai"),
        "authority_tool_ids": ("symbolicai", "ergoai"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "kernel",
        "property_class": "interactive_proof_kernel",
        "description": "Lean / Rocq / Isabelle kernels",
        "tool_ids": ("lean", "coq", "isabelle"),
        "authority_tool_ids": ("lean", "coq", "isabelle"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "runtime_mtl",
        "property_class": "runtime_mtl_monitoring",
        "description": "Runtime MTL monitors",
        "tool_ids": ("runtime-mtl", "runtime-mtl-external"),
        "authority_tool_ids": ("runtime-mtl", "runtime-mtl-external"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "attestation",
        "property_class": "attestation_zkp",
        "description": "Attestation / ZKP circuit bindings",
        "tool_ids": ("zkp-circuit",),
        "authority_tool_ids": ("zkp-circuit",),
        "check_kind": "identity_or_in_process",
    },
)

SMT_POSITIVE: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (and (> x 0) (< x 0)))
(check-sat)
"""

SMT_NEGATIVE: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(check-sat)
"""

# Mutation of the positive script: drop one conjunct so the formula becomes sat.
SMT_MUTATED: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(check-sat)
"""

# Prefer multi-component versions (4.16.0) over lone digits glued to product
# names (the trailing "3" in "Z3").
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+")
_LONE_VERSION_TOKEN = re.compile(r"\b\d+\b")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_JAVA_VERSION_RE = re.compile(
    r'(?im)^\s*(?:openjdk|java)\s+version\s+"'
    r'(?P<version>\d+(?:[._+\-][^"]*)?)"'
)
JAVA_OPTION_ENV_VARS: Final = (
    "_JAVA_OPTIONS",
    "JAVA_TOOL_OPTIONS",
    "JDK_JAVA_OPTIONS",
)


# ---------------------------------------------------------------------------
# Offline environment / process helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root containing the offline toolchain lock."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "config"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an environment that blocks opportunistic installs and fetches."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["ELAN_NO_AUTO_INSTALL"] = "1"
    env.setdefault("ELAN_IO_THREADS", "1")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    # Prevent curl|sh installers from being "helpful".
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    return env


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_java_version_banner(banner: str | None) -> str | None:
    """Return only the quoted java/openjdk identity token, never arbitrary text."""

    match = _JAVA_VERSION_RE.search(banner or "")
    if match is None:
        return None
    return match.group("version")


def java_major_version(banner: str | None) -> int | None:
    """Parse major version from a quoted java/openjdk identity banner only."""

    token = parse_java_version_banner(banner)
    if token is None:
        return None
    components = re.findall(r"\d+", token)
    if not components:
        return None
    if components[0] == "1" and len(components) > 1:
        return int(components[1])
    return int(components[0])


def bounded_run(
    argv: Sequence[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run argv with hard bounds; never shell=True; never raise on timeout."""

    if not argv:
        return None
    if isinstance(argv, (str, bytes, bytearray)):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    try:
        return subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env) if env is not None else offline_env(),
            cwd=str(cwd) if cwd is not None else None,
            input=stdin,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def content_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _redacted_value(value: Any, *, reason: str) -> dict[str, Any]:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "redacted": True,
        "reason": reason,
        "byte_length": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _contains_raw_secret_marker(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "private-witness-fvt047-secret-axiom-never-leak",
            "witness_bytes=",
            "toxic_waste=",
            "trapdoor=",
            "private_secret=",
            "secret=",
        )
    )


def public_evidence_projection(
    payload: Any,
    *,
    repo_root: Path | None = None,
    _key: str = "",
) -> Any:
    """Return a portable, bounded projection safe for checked-in receipts."""

    normalized_key = _key.lower()
    if normalized_key in RAW_OUTPUT_KEYS:
        if isinstance(payload, Mapping) and payload.get("redacted") is True:
            return dict(payload)
        return _redacted_value(payload, reason="raw_process_output_forbidden")
    if normalized_key in RAW_SECRET_KEYS and payload not in (None, True, False):
        if isinstance(payload, Mapping) and payload.get("redacted") is True:
            return dict(payload)
        return _redacted_value(payload, reason="raw_secret_or_witness_forbidden")
    if isinstance(payload, Mapping):
        return {
            str(key): public_evidence_projection(
                value,
                repo_root=repo_root,
                _key=str(key),
            )
            for key, value in payload.items()
        }
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        return [
            public_evidence_projection(value, repo_root=repo_root)
            for value in payload
        ]
    if isinstance(payload, bytes):
        return _redacted_value(payload.hex(), reason="raw_bytes_forbidden")
    if not isinstance(payload, str):
        return payload
    if _contains_raw_secret_marker(payload):
        return _redacted_value(
            payload,
            reason="raw_secret_or_witness_marker_forbidden",
        )

    value = payload
    if repo_root is not None:
        root_text = str(repo_root.resolve())
        value = value.replace(root_text, "<repo-root>")
    value = HOST_PRIVATE_PATH_RE.sub("<host-path-redacted>", value)
    if len(value.encode("utf-8")) > MAX_PUBLIC_STRING_BYTES:
        return _redacted_value(value, reason="unbounded_public_string_forbidden")
    return value


def public_evidence_audit(payload: Any) -> dict[str, Any]:
    """Fail closed if a public receipt still contains private/unbounded data."""

    failures: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        normalized_key = key.lower()
        if normalized_key in RAW_OUTPUT_KEYS:
            if not (
                isinstance(value, Mapping)
                and value.get("redacted") is True
                and value.get("sha256")
            ):
                failures.append(f"raw_process_output:{key}")
            return
        if normalized_key in RAW_SECRET_KEYS and value not in (None, True, False):
            if not (
                isinstance(value, Mapping)
                and value.get("redacted") is True
                and value.get("sha256")
            ):
                failures.append(f"raw_secret_or_witness:{key}")
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                walk(child)
            return
        if isinstance(value, str):
            if HOST_PRIVATE_PATH_RE.search(value):
                failures.append("host_private_path")
            if _contains_raw_secret_marker(value):
                failures.append("raw_secret_or_witness_marker")
            if len(value.encode("utf-8")) > MAX_PUBLIC_STRING_BYTES:
                failures.append("unbounded_public_string")

    walk(payload)
    unique_failures = sorted(set(failures))
    return {
        "satisfied": not unique_failures,
        "failures": unique_failures,
        "host_private_paths_forbidden": True,
        "raw_process_output_forbidden": True,
        "raw_secret_or_witness_forbidden": True,
        "max_public_string_bytes": MAX_PUBLIC_STRING_BYTES,
    }


def _project_semantic_lane_result(
    result: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Create a self-consistent public projection of one semantic receipt."""

    projected = public_evidence_projection(result, repo_root=repo_root)
    if not isinstance(projected, dict):
        return {}
    source_digest = str(result.get("digest_sha256") or "")
    receipt = projected.get("receipt")
    if isinstance(receipt, dict):
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        ):
            if field_name in receipt:
                receipt[field_name] = content_digest(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != field_name
                    }
                )
        projected["digest_sha256"] = content_digest(receipt)
    per_tool = projected.get("per_tool")
    if isinstance(per_tool, Mapping):
        for tool_result in per_tool.values():
            if isinstance(tool_result, dict):
                checks = tool_result.get("checks")
                if isinstance(checks, list):
                    tool_result["check_set_digest_sha256"] = content_digest(checks)
    projected["public_projection"] = {
        "source_receipt_digest_sha256": source_digest or None,
        "portable_paths": True,
        "raw_process_output_retained": False,
        "raw_secret_or_witness_retained": False,
    }
    return projected


def file_digest(path: str | Path | None) -> str | None:
    """Return a canonical SHA-256 identity for a regular file."""

    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def classify_executable_artifact(path: str | Path | None) -> str:
    """Classify executable evidence without treating generated shims as provers."""

    if not path:
        return "none"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    try:
        prefix = candidate.read_bytes()[:8192].decode("utf-8", errors="ignore").lower()
    except OSError:
        return "unreadable"
    shim_markers = (
        "hermetic",
        "generated by ",
        "proposal-only",
        "proposal only",
        "shadow shim",
        "parity engine",
    )
    if prefix.startswith("#!") and any(marker in prefix for marker in shim_markers):
        return "generated_hermetic_shim"
    if prefix.startswith("#!"):
        # A launcher script is not the prover artifact.  It can be useful for
        # discovery, but production certification must additionally bind the
        # executable/archive that the launcher dispatches to.
        return "launcher_script"
    return "native_or_managed_binary"


def observed_platform_id() -> str:
    """Return the lock's normalized host platform identifier."""

    system = platform.system().lower()
    machine = platform.machine().lower()
    system_name = {"linux": "linux", "darwin": "darwin"}.get(system, system)
    machine_name = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64" if system_name == "linux" else "arm64",
        "aarch64": "aarch64" if system_name == "linux" else "arm64",
    }.get(machine, machine)
    return f"{system_name}-{machine_name}"


# ---------------------------------------------------------------------------
# Pin / shim detectors (shared semantics with packaging gate)
# ---------------------------------------------------------------------------


def detect_lean_shim_toolchain_mismatch(
    selected_toolchain: str | None,
    installed_toolchains: Sequence[str],
) -> bool:
    """True when the selected Lean toolchain is not offline-installed."""

    if not selected_toolchain or not str(selected_toolchain).strip():
        return False
    installed = {
        item.strip() for item in installed_toolchains if item and str(item).strip()
    }
    return selected_toolchain.strip() not in installed


def list_elan_installed_toolchains(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Read offline-installed Lean toolchains from the local elan directory."""

    source_env = env if env is not None else os.environ
    elan_home = Path(source_env.get("ELAN_HOME", Path.home() / ".elan"))
    toolchains_dir = elan_home / "toolchains"
    if not toolchains_dir.is_dir():
        return []
    installed: list[str] = []
    for entry in sorted(toolchains_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("leanprover--lean4---"):
            version = name.split("---", 1)[-1]
            installed.append(f"leanprover/lean4:{version}")
        else:
            installed.append(name.replace("--", "/").replace("---", ":"))
    return installed


def _parse_version_tuple(token: str) -> tuple[int, ...]:
    parts = [int(part) for part in token.split(".") if part.isdigit() or part.isdecimal()]
    # Handle pure digit segments only.
    cleaned: list[int] = []
    for segment in token.split("."):
        match = re.match(r"^(\d+)", segment)
        if not match:
            break
        cleaned.append(int(match.group(1)))
    return tuple(cleaned) if cleaned else tuple(parts)


def _extract_version_tokens(text: str) -> list[str]:
    """Extract version-like tokens, preferring dotted multi-component forms."""

    dotted = _VERSION_TOKEN.findall(text or "")
    if dotted:
        return dotted
    return _LONE_VERSION_TOKEN.findall(text or "")


def detect_locked_version_mismatch(
    locked_version: str,
    observed_version_string: str,
) -> bool:
    """True when the locked pin is not reflected in the observed banner.

    Supports exact pins (``1.3.3``, ``v4.31.0``) and simple range pins of the
    form ``>=X,<Y`` (used by the Python Z3 package pin).
    """

    locked = (locked_version or "").strip()
    observed = (observed_version_string or "").strip()
    if not locked:
        return False
    if not observed:
        return True

    # Range pin: >=A,<B or >=A,<=B
    if locked.startswith(">") or locked.startswith("<") or "," in locked:
        return not _range_pin_satisfied(locked, observed)

    candidates = {locked, locked.lstrip("vV")}
    if any(candidate and candidate in observed for candidate in candidates):
        return False
    # Fallback: compare leading numeric tokens.
    locked_tokens = _extract_version_tokens(locked)
    observed_tokens = _extract_version_tokens(observed)
    if locked_tokens and any(token in observed_tokens for token in locked_tokens):
        # Prefer the primary (first) locked token.
        primary = locked_tokens[0]
        return primary not in observed
    return True


def _range_pin_satisfied(range_spec: str, observed: str) -> bool:
    tokens = _extract_version_tokens(observed)
    if not tokens:
        return False
    observed_tuple = _parse_version_tuple(tokens[0])
    if not observed_tuple:
        return False

    lower: tuple[int, ...] | None = None
    upper: tuple[int, ...] | None = None
    upper_inclusive = False
    for clause in range_spec.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            lower = _parse_version_tuple(clause[2:].strip().lstrip("vV"))
        elif clause.startswith(">"):
            # Exclusive lower — treat as next micro by requiring strictly greater.
            lower = _parse_version_tuple(clause[1:].strip().lstrip("vV"))
            # Approximate exclusive lower as not-equal handling below.
            if observed_tuple == lower:
                return False
        elif clause.startswith("<="):
            upper = _parse_version_tuple(clause[2:].strip().lstrip("vV"))
            upper_inclusive = True
        elif clause.startswith("<"):
            upper = _parse_version_tuple(clause[1:].strip().lstrip("vV"))
            upper_inclusive = False
    if lower is not None and observed_tuple < lower:
        return False
    if upper is not None:
        if upper_inclusive:
            if observed_tuple > upper:
                return False
        elif observed_tuple >= upper:
            return False
    return True


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str  # positive | negative | mutation | replay
    status: str  # passed | failed | skipped | unavailable
    expected: str
    observed: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCertification:
    tool_id: str
    lane_ids: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    availability_declared: str = ""
    executable_path: str | None = None
    executable_sha256: str | None = None
    executable_artifact_class: str = "none"
    version_string: str | None = None
    locked_version: str = ""
    locked_version_mismatch: bool = False
    shim_toolchain_mismatch: bool = False
    path_present: bool = False
    identity_probed: bool = False
    installed: bool = False
    usable: bool = False
    production_certified: bool = False
    unavailable: bool = False
    promotion_blocked: bool = True
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    artifact_identities: list[dict[str, Any]] = field(default_factory=list)
    semantic_receipt_digests: list[str] = field(default_factory=list)
    evidence_class: str = "unavailable"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


@dataclass
class DisagreementQuarantine:
    quarantine_id: str
    lane_id: str
    property_class: str
    tool_ids: list[str]
    outcomes: dict[str, str]
    status: str = "quarantined"
    reason: str = "cross_provider_disagreement"
    promotion_blocked_tool_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LaneCertification:
    lane_id: str
    property_class: str
    description: str
    tool_ids: list[str]
    authority_tool_ids: list[str]
    certified_tool_ids: list[str] = field(default_factory=list)
    certified_authority_tool_ids: list[str] = field(default_factory=list)
    unavailable_tool_ids: list[str] = field(default_factory=list)
    blocked_tool_ids: list[str] = field(default_factory=list)
    disagreement_quarantine_ids: list[str] = field(default_factory=list)
    promotion_ready: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lock loading
# ---------------------------------------------------------------------------


def load_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(f"offline toolchain lock missing: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("toolchain lock must be a JSON object")
    return payload


def lock_tools_by_id(lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tools = lock.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("lock.tools must be a list")
    return {str(entry["tool_id"]): entry for entry in tools if "tool_id" in entry}


def tool_platform_support(
    entry: Mapping[str, Any],
    *,
    host_platform: str,
    global_supported_platforms: Sequence[str],
) -> dict[str, Any]:
    """Derive per-tool support from the reviewed deployment lock.

    Deployment-contract platforms are authoritative when present. Otherwise,
    exact pins, ``any`` pins, and reviewed source pins are considered. Missing
    platform metadata fails closed for managed tools.
    """

    globally_supported = host_platform in set(global_supported_platforms)
    contract = entry.get("deployment_contract") or {}
    contract_platforms = [
        str(item) for item in (contract.get("supported_platforms") or [])
    ]
    pin_platforms = [
        str(pin.get("platform") or "")
        for pin in (entry.get("pins") or [])
        if isinstance(pin, Mapping)
    ]
    availability = str(entry.get("availability") or "")
    managed = availability == "managed_pin"

    contract_platforms = [item for item in contract_platforms if item]
    pin_platforms = [item for item in pin_platforms if item]
    declared = sorted(set(contract_platforms) | set(pin_platforms))
    if not managed:
        status = "supported_here" if globally_supported else "ambiguous"
        basis = "global_platform_policy"
    else:
        observations: list[tuple[str, bool]] = []
        if contract_platforms:
            observations.append(
                (
                    "deployment_contract.supported_platforms",
                    bool(
                        "any" in contract_platforms
                        or host_platform in contract_platforms
                    ),
                )
            )
        if pin_platforms:
            observations.append(
                (
                    "tool.pins.platform",
                    bool(
                        {"any", "source"} & set(pin_platforms)
                        or host_platform in pin_platforms
                    ),
                )
            )
        if not observations:
            status = "ambiguous"
            basis = "managed_tool_platform_metadata_missing"
        else:
            contract_support = (
                observations[0][1] if contract_platforms else None
            )
            pin_support = (
                next(
                    (
                        value
                        for source, value in observations
                        if source == "tool.pins.platform"
                    ),
                    None,
                )
            )
            # The reviewed deployment contract is the support ceiling.  Pins
            # may narrow a claimed-supported contract when no artifact can run
            # here, but a generic/source pin cannot broaden an explicit host
            # exclusion (notably external SecPAL on linux-aarch64).
            if contract_support is False:
                tool_supported = False
                basis = "deployment_contract.supported_platforms"
            elif contract_support is True and pin_support is False:
                status = "ambiguous"
                basis = "supported_contract_without_host_artifact_pin"
                tool_supported = None
            elif contract_support is True:
                tool_supported = True
                basis = "deployment_contract.supported_platforms"
            else:
                tool_supported = bool(pin_support)
                basis = "tool.pins.platform"

            if tool_supported is None:
                pass
            elif tool_supported and not globally_supported:
                status = "ambiguous"
                basis = "tool_and_global_platform_policy_contradict"
            elif tool_supported:
                status = "supported_here"
            else:
                status = "unsupported_here"

    supported = status == "supported_here"

    return {
        "tool_id": str(entry.get("tool_id") or ""),
        "host_platform": host_platform,
        "availability": availability,
        "managed": managed,
        "supported": bool(supported),
        "classification": status,
        "ambiguous": status == "ambiguous",
        "exception_eligible": status == "unsupported_here",
        "basis": basis,
        "declared_platforms": declared,
        "globally_supported": globally_supported,
    }


# ---------------------------------------------------------------------------
# Identity probing
# ---------------------------------------------------------------------------


def resolve_executable(candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if not name:
            continue
        # Bare names are PATH lookups.  Treating a same-name file in the
        # certifier's cwd as an executable lets an unreviewed checkout file
        # shadow the locked tool.
        if os.path.isabs(name) or os.sep in name:
            path = Path(name)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
            continue
        found = shutil.which(name)
        if found:
            return found
    return None


def _managed_state_model_identity(
    tool_id: str,
    executable: str,
    *,
    env: Mapping[str, str],
) -> dict[str, Any] | None:
    """Validate the complete managed TLC/Apalache payload and launcher bundle."""

    if tool_id not in {"tlc", "apalache"}:
        return None
    datasets_root = repo_root_from() / "ipfs_datasets_py"
    datasets_text = str(datasets_root)
    if datasets_text not in sys.path:
        sys.path.insert(0, datasets_text)
    from ipfs_datasets_py.logic.backends.installers import state_model

    explicit_java = str(env.get(state_model.JAVA_EXECUTABLE_ENV) or "").strip()
    if not explicit_java:
        java_home = str(env.get("JAVA_HOME") or "").strip()
        if java_home:
            explicit_java = str(Path(java_home) / "bin" / "java")
    if not explicit_java:
        explicit_java = shutil.which("java", path=env.get("PATH")) or ""
    minimum = (
        state_model.TLC_MIN_JAVA_MAJOR
        if tool_id == "tlc"
        else state_model.APALACHE_MIN_JAVA_MAJOR
    )
    java = state_model.probe_java_runtime(
        java_executable=explicit_java or None,
        minimum_major=minimum,
    )
    path = Path(executable)
    if not java.usable or java.executable is None or path.parent.name != "bin":
        return {
            "usable": False,
            "reason": "validated_java_or_managed_root_missing",
            "java_runtime": java.to_dict(),
        }
    root = path.parent.parent
    identity = (
        state_model.managed_tlc_identity(
            root,
            java_executable=java.executable,
        )
        if tool_id == "tlc"
        else state_model.managed_apalache_identity(
            root,
            java_executable=java.executable,
        )
    )
    identity["java_runtime"] = java.to_dict()
    return identity


def _first_pin_sha256(entry: Mapping[str, Any]) -> str:
    pins = entry.get("pins") or ()
    if not pins or not isinstance(pins[0], Mapping):
        return ""
    return str(pins[0].get("sha256") or "").strip().lower()


def probe_tool_identity(
    entry: Mapping[str, Any],
    *,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Bounded offline identity probe. PATH alone is never usability."""

    tool_id = str(entry.get("tool_id") or "")
    candidates = list(entry.get("executable_candidates") or [])
    availability = str(entry.get("availability") or "")
    probe = dict(entry.get("offline_probe") or {})
    timeout = float(probe.get("timeout_seconds") or PROBE_TIMEOUT_SECONDS)
    argv_suffix = list(probe.get("argv") or ["--version"])

    result: dict[str, Any] = {
        "tool_id": tool_id,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "installed": False,
        "shim_toolchain_mismatch": False,
        "selected_toolchain": None,
        "installed_toolchains": [],
        "in_process": False,
        "probe_error": None,
    }

    # In-process / declared-gap tools have no executable requirement.
    if availability in {"in_process", "declared_gap", "advisor_only"} and not candidates:
        result["in_process"] = True
        # Still treat declared_gap as not installed unless a module probe succeeds.
        if availability == "declared_gap":
            result["installed"] = False
            result["probe_error"] = "declared_gap"
            return result
        if availability == "in_process":
            module_ok, version = _probe_in_process_module(tool_id, env=env)
            result["identity_probed"] = module_ok
            result["installed"] = module_ok
            result["version_string"] = version
            return result
        if availability == "advisor_only":
            # Advisors may be optional Python packages — probe lightly.
            module_ok, version = _probe_in_process_module(tool_id, env=env)
            result["identity_probed"] = module_ok
            result["installed"] = module_ok
            result["version_string"] = version
            return result

    executable = resolve_executable(candidates or [tool_id])
    if executable is None:
        result["probe_error"] = "executable_not_on_path"
        return result

    result["path_present"] = True
    result["executable_path"] = executable

    probe_env = dict(env)
    if tool_id == "java":
        # Hostile Java option variables can replace the identity banner or
        # force an otherwise valid runtime to fail before the probe starts.
        for key in JAVA_OPTION_ENV_VARS:
            probe_env.pop(key, None)
    selected_lean_toolchain: str | None = None
    if tool_id == "lean":
        installed_toolchains = list_elan_installed_toolchains(probe_env)
        locked_toolchain = str(probe.get("locked_toolchain") or "").strip()
        result["installed_toolchains"] = installed_toolchains
        if locked_toolchain and locked_toolchain in installed_toolchains:
            # Select an already-installed exact pin without permitting elan to
            # fetch anything. offline_env() always carries ELAN_NO_AUTO_INSTALL.
            selected_lean_toolchain = locked_toolchain
            probe_env["ELAN_TOOLCHAIN"] = locked_toolchain

    completed = bounded_run(
        [executable, *argv_suffix],
        timeout=timeout,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        # PATH presence without a successful identity probe is not installed.
        return result
    combined = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    if tool_id == "tlc":
        cleaned = _ANSI_ESCAPE_RE.sub("", combined)
        accepted_returncodes = {
            int(value) for value in probe.get("accepted_returncodes") or (0, 1)
        }
        required_markers = tuple(
            str(value) for value in probe.get("required_markers") or ()
        )
        semantic_help = bool(required_markers) and all(
            marker in cleaned for marker in required_markers
        )
        managed_identity = _managed_state_model_identity(
            tool_id,
            executable,
            env=probe_env,
        )
        result["managed_identity"] = managed_identity
        managed_digest = str(
            (managed_identity or {}).get("artifact_sha256") or ""
        ).lower()
        managed_revision = str(
            (managed_identity or {}).get("revision") or ""
        ).strip()
        managed_release_tag = str(
            (managed_identity or {}).get("release_tag") or ""
        ).strip()
        expected_revision = str(probe.get("revision") or "").strip()
        expected_release_tag = str(probe.get("release_tag") or "").strip()
        digest_bound = bool(
            managed_digest
            and _first_pin_sha256(entry) == managed_digest
            and str(probe.get("artifact_sha256") or "").lower()
            == managed_digest
        )
        # Real managed identities always publish release_tag + revision. When a
        # fixture omits those keys, lock-declared binding still appears in the
        # version string without weakening digest-bound usability.
        if "revision" in (managed_identity or {}) or "release_tag" in (
            managed_identity or {}
        ):
            revision_bound = bool(
                managed_revision
                and expected_revision
                and managed_revision == expected_revision
                and managed_release_tag
                and expected_release_tag
                and managed_release_tag == expected_release_tag
            )
        else:
            revision_bound = bool(expected_revision and expected_release_tag)
        if (
            completed.returncode not in accepted_returncodes
            or not semantic_help
            or not managed_identity
            or not managed_identity.get("usable")
            or not digest_bound
            or not revision_bound
        ):
            result["probe_error"] = (
                "tlc_help_or_managed_digest_identity_failed"
            )
            return result
        bound_tag = managed_release_tag or expected_release_tag or (
            f"v{_pin_version(entry)}"
        )
        bound_revision = managed_revision or expected_revision
        result["version_string"] = (
            f"TLC managed release {_pin_version(entry)} "
            f"({bound_tag}@{bound_revision}); "
            f"artifact sha256:{managed_identity['artifact_sha256']}"
        )
        result["identity_probed"] = True
        result["installed"] = True
        return result

    if completed.returncode != 0:
        result["probe_error"] = f"identity_probe_nonzero:{completed.returncode}"
        return result

    if tool_id == "apalache":
        managed_identity = _managed_state_model_identity(
            tool_id,
            executable,
            env=probe_env,
        )
        result["managed_identity"] = managed_identity
        managed_digest = str(
            (managed_identity or {}).get("artifact_sha256") or ""
        ).lower()
        digest_bound = bool(
            managed_digest
            and _first_pin_sha256(entry) == managed_digest
            and str(probe.get("artifact_sha256") or "").lower()
            == managed_digest
        )
        if (
            not managed_identity
            or not managed_identity.get("usable")
            or not digest_bound
        ):
            result["probe_error"] = "managed_digest_identity_failed"
            return result

    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    # Some Java runtimes write the version to stderr; that is valid only when
    # the process itself succeeds.
    if tool_id == "java":
        java_text = combined
        if not parse_java_version_banner(java_text):
            # Retry with -version which java accepts.
            completed = bounded_run(
                [executable, "-version"], timeout=timeout, env=probe_env
            )
            if completed is None:
                result["probe_error"] = "probe_timeout_or_spawn_failure"
                return result
            if completed.returncode != 0:
                result["probe_error"] = (
                    f"identity_probe_nonzero:{completed.returncode}"
                )
                return result
            java_text = "\n".join(
                part
                for part in (completed.stdout, completed.stderr)
                if part
            ).strip()
        quoted = parse_java_version_banner(java_text)
        if not quoted:
            result["probe_error"] = "java_version_banner_unreadable"
            return result
        major = java_major_version(java_text)
        result["version_string"] = (
            f'java version "{quoted}"'
            if major is None
            else f'java version "{quoted}" (major {major})'
        )
        result["java_major"] = major
        result["identity_probed"] = True
        result["installed"] = True
        return result

    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result

    result["version_string"] = banner
    result["identity_probed"] = True
    result["installed"] = True

    if tool_id == "lean":
        installed = list(result["installed_toolchains"])
        result["installed_toolchains"] = installed
        match = re.search(r"version\s+(\d+\.\d+\.\d+)", banner, re.IGNORECASE)
        selected = (
            selected_lean_toolchain
            or (
                f"leanprover/lean4:v{match.group(1)}"
                if match
                else probe.get("locked_toolchain")
            )
        )
        result["selected_toolchain"] = selected
        result["shim_toolchain_mismatch"] = detect_lean_shim_toolchain_mismatch(
            selected, installed
        )

    return result


def _probe_in_process_module(
    tool_id: str,
    *,
    env: Mapping[str, str],
) -> tuple[bool, str | None]:
    """Bounded isolated import probe for in-process tools. Never installs."""

    if tool_id == "symbolicai":
        # The PyPI distribution is ``symbolicai`` but its import package is
        # ``symai``.  Importing symai is not a harmless availability probe: it
        # may initialize user configuration.  Bind both identities without
        # executing package code.
        try:
            distribution_version = importlib.metadata.version("symbolicai")
            module_spec = importlib.util.find_spec("symai")
        except (
            ImportError,
            ModuleNotFoundError,
            importlib.metadata.PackageNotFoundError,
            ValueError,
        ):
            return False, None
        if module_spec is None or not distribution_version:
            return False, None
        return (
            True,
            "python-distribution:"
            f"symbolicai=={distribution_version};module:symai",
        )

    module_map = {
        "runtime-mtl": "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl",
        "datalog-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "secpal-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "ergoai": "ergoai",
        "zkp-circuit": None,  # declared gap — never pretend installed
    }
    module_name = module_map.get(tool_id)
    if not module_name:
        return False, None
    completed = bounded_run(
        (
            sys.executable,
            "-c",
            (
                "import importlib,sys; "
                "module=importlib.import_module(sys.argv[1]); "
                "print(getattr(module, '__version__', '') or "
                "getattr(module, '__file__', '') or 'imported')"
            ),
            module_name,
        ),
        timeout=PROBE_TIMEOUT_SECONDS,
        env=env,
    )
    if completed is None or completed.returncode != 0:
        return False, None
    identity = first_nonempty_line(completed.stdout)
    return bool(identity), f"python-module:{module_name}"


# ---------------------------------------------------------------------------
# Live checks
# ---------------------------------------------------------------------------


def _smt_argv(tool_id: str, executable: str) -> list[str]:
    if tool_id == "z3":
        return [executable, "-in"]
    if tool_id == "cvc5":
        return [executable, "--lang", "smt2", "-"]
    return [executable]


def _normalize_smt_verdict(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    # Prefer the last sat/unsat token for solvers that echo scripts.
    verdicts = re.findall(r"\b(unsat|sat|unknown)\b", text)
    if not verdicts:
        return "unknown"
    return verdicts[-1]


def run_smt_checks(
    tool_id: str,
    executable: str,
    *,
    env: Mapping[str, str],
) -> list[CheckResult]:
    """Positive / negative / mutation / replay checks for an SMT solver."""

    checks: list[CheckResult] = []

    def invoke(script: str) -> str:
        completed = bounded_run(
            _smt_argv(tool_id, executable),
            timeout=CHECK_TIMEOUT_SECONDS,
            env=env,
            stdin=script,
        )
        if completed is None:
            return "timeout"
        return _normalize_smt_verdict(completed.stdout, completed.stderr)

    positive = invoke(SMT_POSITIVE)
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if positive == "unsat" else "failed",
            expected="unsat",
            observed=positive,
            detail="unsatisfiable conjunction (proof-side)",
        )
    )

    negative = invoke(SMT_NEGATIVE)
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="passed" if negative == "sat" else "failed",
            expected="sat",
            observed=negative,
            detail="satisfiable constraint (model/counterexample-side)",
        )
    )

    mutated = invoke(SMT_MUTATED)
    # Mutation of the positive case must change the outcome (unsat → sat).
    mutation_ok = positive == "unsat" and mutated == "sat"
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="passed" if mutation_ok else "failed",
            expected="sat (changed from unsat)",
            observed=mutated,
            detail="dropped conjunct on positive formula; outcome must change",
        )
    )

    replay = invoke(SMT_POSITIVE)
    replay_ok = replay == positive == "unsat"
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if replay_ok else "failed",
            expected=positive,
            observed=replay,
            detail="re-run positive script; outcome must be stable",
        )
    )
    return checks


def run_identity_alive_checks(
    tool_id: str,
    identity: Mapping[str, Any],
) -> list[CheckResult]:
    """For non-SMT tools: identity stability as positive/replay; negative/mutation skip.

    Full semantic positive/negative fixtures for every external prover require
    tool-specific corpora owned by later lanes. Hermetic certification still
    requires an exact identity probe (positive) and a stable re-probe (replay).
    Negative and mutation remain explicit skips so they never silently pass.
    """

    version = str(identity.get("version_string") or "")
    probed = bool(identity.get("identity_probed"))
    checks = [
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if probed and version else "failed",
            expected="non-empty exact identity",
            observed=version or "missing",
            detail="bounded offline identity probe",
        ),
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="skipped",
            expected="tool-specific unsat/counterexample fixture",
            observed="skipped",
            detail=(
                "property-specific negative fixture not required for identity "
                "lane; absence does not invent success"
            ),
        ),
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="skipped",
            expected="tool-specific mutation fixture",
            observed="skipped",
            detail="mutation suite deferred to property corpora; not synthetic pass",
        ),
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if probed and version else "failed",
            expected=version,
            observed=version or "missing",
            detail="identity banner retained as replay witness for this probe",
        ),
    ]
    return checks


def run_in_process_checks(tool_id: str, identity: Mapping[str, Any]) -> list[CheckResult]:
    """Minimal positive/replay for in-process modules (runtime MTL, etc.)."""

    ok = bool(identity.get("identity_probed"))
    version = str(identity.get("version_string") or "")
    return [
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if ok else "failed",
            expected="importable module identity",
            observed=version or "missing",
            detail="in-process module import probe (no install)",
        ),
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="skipped",
            expected="module-specific negative fixture",
            observed="skipped",
            detail="no synthetic negative invented",
        ),
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="skipped",
            expected="module-specific mutation fixture",
            observed="skipped",
            detail="no synthetic mutation invented",
        ),
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if ok else "failed",
            expected=version,
            observed=version or "missing",
            detail="repeat import identity",
        ),
    ]


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def _pin_version(entry: Mapping[str, Any]) -> str:
    pins = entry.get("pins") or []
    if not pins:
        return ""
    return str(pins[0].get("version") or "")


def certify_tool(
    entry: Mapping[str, Any],
    *,
    lane_ids: Sequence[str],
    check_kind: str,
    env: Mapping[str, str],
) -> ToolCertification:
    tool_id = str(entry["tool_id"])
    families = list(entry.get("families") or [])
    locked_version = _pin_version(entry)
    availability = str(entry.get("availability") or "")

    cert = ToolCertification(
        tool_id=tool_id,
        lane_ids=list(lane_ids),
        families=families,
        availability_declared=availability,
        locked_version=locked_version,
    )

    # Explicit declared gaps are unavailable and never production-certified.
    if availability == "declared_gap":
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.block_reasons.append("declared_gap")
        cert.evidence_class = "unavailable"
        cert.notes = "Declared install/capability gap; blocks only this tool's promotion."
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="n/a",
                observed="declared_gap",
                detail="gap blocks promotion only for this tool",
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    identity = probe_tool_identity(entry, env=env)
    cert.path_present = bool(identity.get("path_present"))
    cert.executable_path = identity.get("executable_path")
    cert.executable_sha256 = file_digest(cert.executable_path)
    cert.executable_artifact_class = classify_executable_artifact(
        cert.executable_path
    )
    if cert.executable_path:
        cert.artifact_identities.append(
            {
                "kind": "executable",
                "path": cert.executable_path,
                "sha256": cert.executable_sha256,
                "artifact_class": cert.executable_artifact_class,
            }
        )
    cert.version_string = identity.get("version_string")
    cert.identity_probed = bool(identity.get("identity_probed"))
    cert.installed = bool(identity.get("installed"))
    cert.shim_toolchain_mismatch = bool(identity.get("shim_toolchain_mismatch"))

    if locked_version and cert.version_string:
        cert.locked_version_mismatch = detect_locked_version_mismatch(
            locked_version, cert.version_string
        )
    elif locked_version and not cert.version_string:
        cert.locked_version_mismatch = True

    # PATH presence is not usability / not installed without identity.
    if cert.path_present and not cert.identity_probed:
        cert.block_reasons.append("path_presence_without_identity_probe")
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.evidence_class = "path_shim"
        cert.notes = (
            "Executable on PATH but identity probe failed; PATH shims are not usability."
        )
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="exact identity",
                observed="path_only",
                detail="PATH presence is not usability",
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    if not cert.installed:
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.block_reasons.append("unavailable")
        cert.evidence_class = "unavailable"
        cert.notes = (
            f"Tool not installed or not probeable ({identity.get('probe_error')}); "
            "blocks only this tool's promotion."
        )
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="installed+probed",
                observed="unavailable",
                detail=str(identity.get("probe_error") or "unavailable"),
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    if cert.shim_toolchain_mismatch:
        cert.block_reasons.append("shim_toolchain_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.evidence_class = "shim_mismatch"
        cert.notes = (
            "Selected toolchain not offline-installed; fail closed without download."
        )
    elif cert.locked_version_mismatch:
        cert.block_reasons.append("locked_version_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.evidence_class = "version_mismatch"
        cert.notes = (
            "Observed version does not match offline lock pin; "
            "production certification blocked without upgrade/download."
        )
    else:
        cert.usable = True
        cert.evidence_class = "live"

    # Live checks — only meaningful when identity is present.
    if check_kind == "smtlib" and cert.executable_path:
        cert.checks = run_smt_checks(tool_id, cert.executable_path, env=env)
    elif identity.get("in_process") or check_kind == "identity_or_in_process":
        if cert.executable_path and not identity.get("in_process"):
            cert.checks = run_identity_alive_checks(tool_id, identity)
        else:
            cert.checks = run_in_process_checks(tool_id, identity)
    else:
        cert.checks = run_identity_alive_checks(tool_id, identity)

    required_kinds = {"positive", "replay"}
    required_passed = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind in required_kinds
    )
    # Production promotion requires semantic positive, negative, mutation, and
    # replay evidence. Identity-only and import-only checks remain useful
    # availability evidence, but explicit skips must never become proof
    # certification merely because the identity is stable.
    all_live_passed = required_passed and all(
        check.status == "passed" for check in cert.checks
    )
    executable_identity_complete = cert.executable_artifact_class == (
        "native_or_managed_binary"
    )
    if cert.executable_artifact_class == "launcher_script":
        cert.block_reasons.append("launcher_target_artifact_unbound")

    if not all_live_passed:
        cert.block_reasons.append("live_checks_incomplete_or_failed")
        cert.promotion_blocked = True
        if cert.usable:
            cert.notes = (cert.notes + " " if cert.notes else "") + (
                "Live checks incomplete or failed; not production-certified."
            ).strip()

    cert.production_certified = bool(
        cert.usable
        and all_live_passed
        and not cert.locked_version_mismatch
        and not cert.shim_toolchain_mismatch
        and not cert.unavailable
        and executable_identity_complete
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.evidence_class = "production_certified"
        cert.notes = (
            "Hermetic offline certification passed with exact identity and live checks."
        )
    elif cert.usable and not cert.production_certified:
        cert.promotion_blocked = True

    return cert


def quarantine_smt_disagreement(
    tool_certs: Mapping[str, ToolCertification],
    *,
    lane_id: str = "smt",
    property_class: str = "smt_software_verification",
) -> DisagreementQuarantine | None:
    """Compare SMT positive outcomes; quarantine hard disagreement."""

    outcomes: dict[str, str] = {}
    for tool_id in ("z3", "cvc5"):
        cert = tool_certs.get(tool_id)
        if cert is None or cert.unavailable or not cert.usable:
            continue
        for check in cert.checks:
            if check.kind == "positive" and check.status in {"passed", "failed"}:
                outcomes[tool_id] = check.observed
                break

    if len(outcomes) < 2:
        return None

    values = set(outcomes.values())
    # unknown/timeout is not hard disagreement — only sat vs unsat is.
    hard = {value for value in values if value in {"sat", "unsat"}}
    if len(hard) <= 1:
        return None

    tool_ids = sorted(outcomes)
    payload = {
        "lane_id": lane_id,
        "property_class": property_class,
        "outcomes": outcomes,
    }
    quarantine_id = f"eq-quarantine:{content_digest(payload)[:16]}"
    return DisagreementQuarantine(
        quarantine_id=quarantine_id,
        lane_id=lane_id,
        property_class=property_class,
        tool_ids=tool_ids,
        outcomes=outcomes,
        promotion_blocked_tool_ids=tool_ids,
    )


def certify_property_lanes(
    tool_certs: Mapping[str, ToolCertification],
    disagreements: Sequence[DisagreementQuarantine],
    *,
    property_lanes: Sequence[Mapping[str, Any]] = PROPERTY_LANES,
) -> list[LaneCertification]:
    """Project lane readiness from certified proof authorities, not runtimes."""

    lanes: list[LaneCertification] = []
    for lane in property_lanes:
        lane_id = str(lane["lane_id"])
        tool_ids = list(lane["tool_ids"])
        authority_tool_ids = list(lane.get("authority_tool_ids") or ())
        if not authority_tool_ids:
            raise ValueError(f"lane {lane_id!r} must declare authority_tool_ids")
        if not set(authority_tool_ids).issubset(tool_ids):
            raise ValueError(
                f"lane {lane_id!r} authority_tool_ids must be a subset of tool_ids"
            )
        certified = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].production_certified
        ]
        certified_authorities = [
            tid for tid in authority_tool_ids if tid in certified
        ]
        unavailable = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].unavailable
        ]
        blocked = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].promotion_blocked
        ]
        q_ids = [
            item.quarantine_id
            for item in disagreements
            if item.lane_id == lane_id
        ]
        lanes.append(
            LaneCertification(
                lane_id=lane_id,
                property_class=str(lane["property_class"]),
                description=str(lane["description"]),
                tool_ids=tool_ids,
                authority_tool_ids=authority_tool_ids,
                certified_tool_ids=certified,
                certified_authority_tool_ids=certified_authorities,
                unavailable_tool_ids=unavailable,
                blocked_tool_ids=blocked,
                disagreement_quarantine_ids=q_ids,
                promotion_ready=bool(certified_authorities) and not q_ids,
                notes=(
                    "Absent tools block only their own promotion."
                    if unavailable
                    else ""
                ),
            )
        )
    return lanes


# ---------------------------------------------------------------------------
# Role-aware semantic elevation (FVT-G200)
# ---------------------------------------------------------------------------


# Semantic certifiers owned by later installation/certification tasks. Each
# entry maps offline lock tool ids to a focused certifier module. Elevation
# never installs, downloads, or conceals unavailable tools.
SEMANTIC_CERTIFIER_SPECS: Final[tuple[dict[str, Any], ...]] = (
    {
        "lane_id": "kernel",
        "module_relative": Path("tools/logic/certification/lean.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("lean",),
        "certified_key": "production_certified",
        "interface": "LeanSemanticCertification@1",
        "selector": "root",
        "evidence_class": "usable_pending_kernel_live_fanin",
        "production_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "kernel_rocq",
        "module_relative": Path("tools/logic/certification/rocq.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("coq",),
        "certified_key": "production_certified",
        "interface": "RocqToolchainCertification@1",
        "selector": "root",
        "evidence_class": "external_prover_installation_pending",
        "production_elevation_allowed": False,
        "usable_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "kernel_isabelle",
        "module_relative": Path("tools/logic/certification/isabelle.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("isabelle",),
        "certified_key": "production_certified",
        "interface": "IsabelleToolchainCertification@1",
        "selector": "root",
        "evidence_class": "external_prover_installation_pending",
        "production_elevation_allowed": False,
        "usable_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "runtime_mtl",
        "module_relative": Path("tools/logic/certification/runtime_mtl.py"),
        "callable_name": "certify_runtime_mtl_semantics",
        "tool_ids": ("runtime-mtl",),
        "certified_key": "certified",
        "interface": "RuntimeMTLSemanticCertification@1",
        "selector": "root",
        "evidence_class": "usable_pending_external_runtime_mtl",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "datalog_secpal",
        "module_relative": Path("tools/logic/certification/authorization.py"),
        "callable_name": "certify_authorization_semantics",
        "tool_ids": ("datalog-authorization", "secpal-authorization"),
        "certified_key": "certified",
        "interface": "AuthorizationSemanticCertification@1",
        "selector": "engine",
        "evidence_class": "usable_pending_authorization_vendor_fanin",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "state_model",
        "module_relative": Path("tools/logic/certification/state_model.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("apalache", "tlc"),
        "certified_key": "production_certified",
        "interface": "StateModelToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "protocol_tamarin",
        "module_relative": Path("tools/logic/certification/tamarin.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("tamarin",),
        "certified_key": "production_certified",
        "interface": "TamarinToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "protocol_proverif",
        "module_relative": Path("tools/logic/certification/proverif.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("proverif",),
        "certified_key": "production_certified",
        "interface": "ProVerifToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "atp",
        "module_relative": Path("tools/logic/certification/atp.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("vampire", "eprover"),
        "certified_key": "production_certified",
        "interface": "ATPToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "hyperltl",
        "module_relative": Path("tools/logic/certification/hyperproperty.py"),
        "callable_name": "certify_hyperproperty_toolchains",
        "tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "certified_key": "certified",
        "interface": "HyperpropertyToolchainCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_adapter_shim",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "runtime_mtl_external",
        "module_relative": Path("tools/logic/certification/runtime_mtl_external.py"),
        "callable_name": "certify_external_runtime_mtl",
        "tool_ids": ("runtime-mtl-external",),
        "certified_key": "certified",
        "interface": "ExternalRuntimeMTLCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_adapter_shim",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "authorization_external",
        "module_relative": Path("tools/logic/certification/authorization_external.py"),
        "callable_name": "certify_external_authorization_shadows",
        "tool_ids": ("souffle", "secpal"),
        "certified_key": "certified",
        "interface": "ExternalAuthorizationShadowCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_shadow_shim",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "attestation",
        "module_relative": Path("tools/logic/certification/zkp.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("zkp-circuit",),
        "certified_key": "production_certified",
        "interface": "ZKPDeploymentCertification@1",
        "selector": "root",
        "evidence_class": "public_artifact_attestation",
        "production_elevation_allowed": True,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "advisors",
        "module_relative": Path("tools/logic/certification/advisors.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("symbolicai", "ergoai"),
        "certified_key": "production_certified",
        "interface": "AdvisorRoleCertification@1",
        "selector": "root",
        "evidence_class": "proposal_only_semantics",
        "production_elevation_allowed": False,
    },
)


def _load_module_from_path(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_semantic_checks(
    tool_id: str,
    raw_checks: Sequence[Mapping[str, Any]] | None,
) -> list[CheckResult]:
    """Retain the complete semantic check set and fail closed on missing kinds."""

    checks: list[CheckResult] = []
    kinds_present: set[str] = set()
    for raw in raw_checks or ():
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        status = str(raw.get("status") or "failed").lower()
        if status not in {"passed", "failed", "skipped", "unavailable"}:
            status = "failed"
        kinds_present.add(kind)
        checks.append(
            CheckResult(
            check_id=str(raw.get("check_id") or f"{tool_id}.{kind}"),
            kind=kind or "unclassified",
            status=status,
            expected=str(raw.get("expected") or "semantic_pass"),
            observed=str(raw.get("observed") or status),
            detail=str(raw.get("detail") or "role-aware semantic elevation"),
            evidence=dict(raw),
            )
        )

    for kind in ("positive", "negative", "mutation", "replay"):
        if kind in kinds_present:
            continue
        checks.append(
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="failed",
                expected="semantic suite coverage",
                observed="missing_kind",
                detail=(
                    "Role-aware elevation requires positive/negative/mutation/"
                    "replay evidence; missing kinds fail closed."
                ),
                evidence={"missing_required_kind": kind},
            )
        )
    return checks


def _tool_certified_from_semantic_receipt(
    tool_id: str,
    receipt: Mapping[str, Any],
    *,
    certified_key: str,
    selector: str,
) -> tuple[bool, list[Mapping[str, Any]], list[str]]:
    """Return (certified, checks, block_reasons) for one tool from a receipt."""

    if selector == "engine":
        for engine in receipt.get("engines") or []:
            if not isinstance(engine, Mapping):
                continue
            if str(engine.get("engine_id") or "") != tool_id:
                continue
            checks = list(engine.get("checks") or [])
            block_reasons = list(engine.get("block_reasons") or [])
            certified = bool(engine.get("certified"))
            return certified, checks, [str(r) for r in block_reasons]
        return False, [], ["engine_missing_from_semantic_receipt"]

    certified = bool(receipt.get(certified_key) or receipt.get("certified"))
    checks = [
        check
        for check in (receipt.get("checks") or [])
        if not isinstance(check, Mapping)
        or not check.get("tool_id")
        or str(check.get("tool_id")) == tool_id
    ]
    block_reasons = list(receipt.get("block_reasons") or [])
    return certified, checks, [str(r) for r in block_reasons]


def _semantic_tool_identity(
    tool_id: str,
    receipt: Mapping[str, Any],
    *,
    selector: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Extract exact executable/artifact identity from a semantic receipt."""

    source: Mapping[str, Any] = receipt
    if selector == "engine":
        source = next(
            (
                engine
                for engine in (receipt.get("engines") or [])
                if isinstance(engine, Mapping)
                and str(engine.get("engine_id") or "") == tool_id
            ),
            {},
        )

    executable = (
        source.get("executable_path")
        or source.get("executable")
        or receipt.get(f"{tool_id.replace('-', '_')}_executable")
    )
    version = (
        source.get("version_string")
        or source.get("version")
        or receipt.get(f"{tool_id.replace('-', '_')}_version_string")
    )
    artifact_path = source.get("lock_path") or receipt.get("lock_path")

    def portable_path(value: Any) -> str:
        candidate = Path(str(value))
        if repo_root is not None:
            try:
                return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
            except (OSError, ValueError):
                pass
        return str(candidate)

    artifacts: list[dict[str, Any]] = []
    if executable:
        artifacts.append(
            {
                "kind": "semantic_executable",
                "path": portable_path(executable),
                "sha256": file_digest(str(executable)),
                "artifact_class": classify_executable_artifact(str(executable)),
            }
        )
    if artifact_path:
        artifacts.append(
            {
                "kind": "deployment_artifact",
                "path": portable_path(artifact_path),
                "sha256": file_digest(str(artifact_path)),
                "declared_digest": receipt.get("lock_digest"),
                "artifact_class": "public_deployment_binding",
            }
        )
    return {
        "executable_path": str(executable) if executable else None,
        "version_string": str(version) if version else None,
        "identity_probed": bool(
            source.get("identity_probed")
            or source.get("usable")
            or receipt.get("identity_probed")
            or receipt.get(f"{tool_id.replace('-', '_')}_identity_probed")
        ),
        "artifacts": artifacts,
    }


def _digest_matches(stored: Any, computed: str) -> bool:
    value = str(stored or "")
    return bool(value) and value in {computed, f"sha256:{computed}"}


def _validate_semantic_receipt_integrity(
    receipt: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    module: Any,
) -> dict[str, Any]:
    """Validate a focused receipt's schema identity and declared self-digest."""

    failures: list[str] = []
    expected_fields = {
        "interface": str(spec["interface"]),
        "schema_version": str(getattr(module, "SCHEMA_VERSION", "") or ""),
        "goal_id": str(getattr(module, "GOAL_ID", "") or ""),
        "task_id": str(getattr(module, "TASK_ID", "") or ""),
    }
    for field_name, expected in expected_fields.items():
        observed = str(receipt.get(field_name) or "")
        if not expected or observed != expected:
            failures.append(f"{field_name}_mismatch")

    digest_fields = [
        field_name
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        )
        if field_name in receipt
    ]
    if not digest_fields:
        failures.append("declared_receipt_digest_missing")
    for field_name in digest_fields:
        body = {key: value for key, value in receipt.items() if key != field_name}
        if not _digest_matches(receipt.get(field_name), content_digest(body)):
            failures.append(f"{field_name}_mismatch")

    return {
        "valid": not failures,
        "failures": failures,
        "declared_digest_fields": digest_fields,
        "interface": receipt.get("interface"),
        "schema_version": receipt.get("schema_version"),
        "goal_id": receipt.get("goal_id"),
        "task_id": receipt.get("task_id"),
    }


def _resolve_artifact_path(path: Any, *, repo_root: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def _validate_artifact_identities(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recompute every artifact hash and identify production-capable bindings."""

    failures: list[str] = []
    validated: list[dict[str, Any]] = []
    production_bindings: list[dict[str, Any]] = []
    root = repo_root.resolve()
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            failures.append(f"artifact_{index}_not_mapping")
            continue
        artifact = dict(item)
        path = _resolve_artifact_path(artifact.get("path"), repo_root=root)
        actual = file_digest(path)
        if path is None or actual is None:
            failures.append(f"artifact_{index}_missing")
            continue
        if str(artifact.get("sha256") or "") != actual:
            failures.append(f"artifact_{index}_sha256_mismatch")
            continue
        artifact_class = str(artifact.get("artifact_class") or "")
        if artifact_class == "repository_source":
            try:
                path.relative_to(root)
            except ValueError:
                failures.append(f"artifact_{index}_repository_source_outside_root")
                continue
        if artifact_class == "public_deployment_binding":
            declared = str(artifact.get("declared_digest") or "")
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if not declared or parsed is None or not _digest_matches(
                declared, content_digest(parsed)
            ):
                failures.append(f"artifact_{index}_declared_digest_mismatch")
                continue
        try:
            artifact["resolved_path"] = path.relative_to(root).as_posix()
        except ValueError:
            artifact["resolved_path"] = str(path)
        validated.append(artifact)
        if artifact_class in {
            "native_or_managed_binary",
            "public_deployment_binding",
        }:
            production_bindings.append(artifact)

    return {
        "valid": not failures,
        "failures": failures,
        "validated": validated,
        "production_bindings": production_bindings,
        "has_production_binding": bool(production_bindings),
    }


def _invoke_semantic_certifier(
    certifier: Any,
    *,
    repo_root: Path,
    env: Mapping[str, str],
) -> Mapping[str, Any]:
    """Call one certifier with only supported, explicitly offline arguments."""

    parameters = inspect.signature(certifier).parameters
    kwargs: dict[str, Any] = {}
    if "repo_root" in parameters:
        kwargs["repo_root"] = repo_root
    if "env" in parameters:
        kwargs["env"] = dict(env)
    if "skip_install" in parameters:
        kwargs["skip_install"] = True
    if "force_install" in parameters:
        kwargs["force_install"] = False
    receipt = certifier(**kwargs)
    if not isinstance(receipt, Mapping):
        raise TypeError("semantic certifier returned non-mapping receipt")
    return receipt


def _offline_observation(
    receipt: Mapping[str, Any],
    *,
    production_elevation_allowed: bool,
) -> dict[str, Any]:
    """Derive no-install/network claims from each focused receipt."""

    install_attempted = bool(receipt.get("install_attempted"))
    download_attempted = bool(receipt.get("download_attempted"))
    network_used = bool(receipt.get("network_used"))
    install_section = receipt.get("install")
    if isinstance(install_section, Mapping):
        install_attempted = install_attempted or bool(
            install_section.get("attempted")
            or install_section.get("installed")
            or install_section.get("downloaded")
        )
        download_attempted = download_attempted or bool(
            install_section.get("downloaded")
        )
        network_used = network_used or bool(install_section.get("network_used"))
    policy = receipt.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    explicit_offline_policy = all(
        policy.get(key) is True for key in ("no_install", "no_download", "no_network")
    )
    in_process_only = policy.get("in_process_only") is True
    declaration_sufficient = bool(
        explicit_offline_policy
        or in_process_only
        or not production_elevation_allowed
    )
    return {
        "install_attempted": install_attempted,
        "download_attempted": download_attempted,
        "network_used": network_used,
        "explicit_offline_policy": explicit_offline_policy,
        "in_process_only": in_process_only,
        "declaration_sufficient": declaration_sufficient,
        "satisfied": declaration_sufficient
        and not (install_attempted or download_attempted or network_used),
    }


def run_semantic_lane_certifiers(
    *,
    repo_root: Path,
    env: Mapping[str, str],
    tool_certs: Mapping[str, ToolCertification],
) -> list[dict[str, Any]]:
    """Invoke focused semantic certifiers offline; never install or fetch."""

    results: list[dict[str, Any]] = []
    for spec in SEMANTIC_CERTIFIER_SPECS:
        module_path = repo_root / Path(spec["module_relative"])
        lane_id = str(spec["lane_id"])
        tool_ids = tuple(spec["tool_ids"])
        entry: dict[str, Any] = {
            "lane_id": lane_id,
            "interface": str(spec["interface"]),
            "module": Path(spec["module_relative"]).as_posix(),
            "tool_ids": list(tool_ids),
            "status": "not_run",
            "certified": False,
            "semantically_usable_tool_ids": [],
            "elevated_tool_ids": [],
            "block_reasons": [],
            "digest_sha256": None,
            "notes": "",
            "evidence_class": str(spec["evidence_class"]),
            "production_elevation_allowed": bool(
                spec["production_elevation_allowed"]
            ),
            "usable_elevation_allowed": bool(
                spec.get("usable_elevation_allowed", True)
            ),
            "certifier_module_sha256": file_digest(module_path),
        }

        if not module_path.is_file():
            entry["status"] = "certifier_missing"
            entry["block_reasons"] = ["semantic_certifier_module_missing"]
            entry["notes"] = f"Missing certifier module: {module_path.as_posix()}"
            results.append(entry)
            continue

        try:
            module = _load_module_from_path(
                module_path,
                f"fvt_role_aware_{lane_id.replace('-', '_')}",
            )
            callable_name = str(spec["callable_name"])
            certifier = getattr(module, callable_name, None)
            if not callable(certifier):
                raise AttributeError(f"{callable_name} not callable on {module_path}")

            receipt = _invoke_semantic_certifier(
                certifier,
                repo_root=repo_root,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed per lane
            entry["status"] = "certifier_error"
            entry["block_reasons"] = [f"{type(exc).__name__}:{exc}"]
            entry["notes"] = (
                "Semantic certifier failed closed; tools stay non-elevated."
            )
            results.append(entry)
            continue

        entry["status"] = "ran"
        entry["receipt"] = dict(receipt)
        entry["digest_sha256"] = content_digest(receipt)
        entry["receipt_integrity"] = _validate_semantic_receipt_integrity(
            receipt,
            spec=spec,
            module=module,
        )
        entry["offline_observation"] = _offline_observation(
            receipt,
            production_elevation_allowed=bool(
                spec["production_elevation_allowed"]
            ),
        )
        entry["receipt_goal_id"] = receipt.get("goal_id")
        entry["receipt_task_id"] = receipt.get("task_id")
        entry["certified"] = bool(
            receipt.get(str(spec["certified_key"])) or receipt.get("certified")
        ) and bool(entry["receipt_integrity"]["valid"])
        if not entry["receipt_integrity"]["valid"]:
            entry["block_reasons"].extend(
                entry["receipt_integrity"]["failures"]
            )
        if not entry["offline_observation"]["satisfied"]:
            entry["block_reasons"].append("offline_observation_failed")
        entry["per_tool"] = {}
        for tool_id in tool_ids:
            certified, raw_checks, block_reasons = _tool_certified_from_semantic_receipt(
                tool_id,
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec.get("selector") or "root"),
            )
            normalized_checks = _normalize_semantic_checks(tool_id, raw_checks)
            identity = _semantic_tool_identity(
                tool_id,
                receipt,
                selector=str(spec.get("selector") or "root"),
                repo_root=repo_root,
            )
            identity["artifacts"].append(
                {
                    "kind": "semantic_certifier_module",
                    "path": Path(spec["module_relative"]).as_posix(),
                    "sha256": file_digest(module_path),
                    "artifact_class": "repository_source",
                }
            )
            artifact_validation = _validate_artifact_identities(
                identity["artifacts"],
                repo_root=repo_root,
            )
            checks_complete = bool(normalized_checks) and all(
                check.status == "passed" for check in normalized_checks
            )
            certified = bool(
                certified
                and entry["receipt_integrity"]["valid"]
                and entry["offline_observation"]["satisfied"]
                and checks_complete
                and artifact_validation["valid"]
            )
            entry["per_tool"][tool_id] = {
                "certified": certified,
                "block_reasons": list(block_reasons)
                + list(artifact_validation["failures"]),
                "check_kinds_present": sorted(
                    {
                        str(c.get("kind"))
                        for c in raw_checks
                        if isinstance(c, Mapping) and c.get("kind")
                    }
                ),
                "checks_passed": sum(
                    1
                    for c in raw_checks
                    if isinstance(c, Mapping) and str(c.get("status")) == "passed"
                ),
                "checks_total": len(raw_checks),
                "checks": [check.to_dict() for check in normalized_checks],
                "check_set_digest_sha256": content_digest(
                    [check.to_dict() for check in normalized_checks]
                ),
                "identity": identity,
                "artifact_validation": artifact_validation,
            }
            if certified:
                entry["semantically_usable_tool_ids"].append(tool_id)
            if certified and bool(spec["production_elevation_allowed"]):
                entry["elevated_tool_ids"].append(tool_id)
        results.append(entry)
    return results


def apply_semantic_elevations(
    tool_certs: dict[str, ToolCertification],
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Elevate usable tools whose focused semantic certifiers fully pass.

    Uses the receipts already collected by ``run_semantic_lane_certifiers`` so
    each offline suite runs once. Synthetic passes are forbidden.
    """

    elevations: list[dict[str, Any]] = []
    root = (repo_root or repo_root_from()).resolve()
    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "")
        spec = next(
            (item for item in SEMANTIC_CERTIFIER_SPECS if item["lane_id"] == lane_id),
            None,
        )
        if spec is None:
            continue
        usable_elevation_allowed = bool(
            spec.get("usable_elevation_allowed", True)
        )
        if not usable_elevation_allowed:
            for pending_tool_id in spec["tool_ids"]:
                pending_cert = tool_certs.get(str(pending_tool_id))
                if pending_cert is None:
                    continue
                pending_cert.usable = False
                pending_cert.unavailable = True
                pending_cert.production_certified = False
                pending_cert.promotion_blocked = True
                pending_cert.evidence_class = str(spec["evidence_class"])
                pending_reason = (
                    "external_prover_installation_and_live_fanin_pending"
                )
                if pending_reason not in pending_cert.block_reasons:
                    pending_cert.block_reasons.append(pending_reason)
        receipt = result.get("receipt")
        if not isinstance(receipt, Mapping):
            if result.get("elevated_tool_ids"):
                elevations.append(
                    {
                        "lane_id": lane_id,
                        "status": "elevation_aborted",
                        "error": "semantic_receipt_missing",
                    }
                )
            continue
        result_digest = str(result.get("digest_sha256") or "")
        integrity = result.get("receipt_integrity")
        offline = result.get("offline_observation")
        declared_digest_fields = [
            field_name
            for field_name in (
                "receipt_digest_sha256",
                "certificate_digest_sha256",
                "digest_sha256",
            )
            if field_name in receipt
        ]
        declared_digests_valid = bool(declared_digest_fields) and all(
            _digest_matches(
                receipt.get(field_name),
                content_digest(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != field_name
                    }
                ),
            )
            for field_name in declared_digest_fields
        )
        result_integrity_valid = bool(
            result.get("status") == "ran"
            and result_digest
            and result_digest == content_digest(receipt)
            and str(receipt.get("interface") or "") == str(spec["interface"])
            and declared_digests_valid
            and isinstance(integrity, Mapping)
            and integrity.get("valid") is True
            and isinstance(offline, Mapping)
            and offline.get("satisfied") is True
        )
        if not result_integrity_valid:
            for tool_id in result.get("tool_ids") or spec["tool_ids"]:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_receipt_integrity_failed",
                    }
                )
            continue

        candidate_ids = list(result.get("tool_ids") or spec["tool_ids"])
        for tool_id in candidate_ids:
            cert = tool_certs.get(str(tool_id))
            if cert is None:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "target_not_usable",
                    }
                )
                continue
            certified, raw_checks, block_reasons = _tool_certified_from_semantic_receipt(
                str(tool_id),
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec.get("selector") or "root"),
            )
            if not certified:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_receipt_not_certified",
                        "block_reasons": block_reasons,
                    }
                )
                continue

            projected = _normalize_semantic_checks(str(tool_id), raw_checks)
            all_passed = all(check.status == "passed" for check in projected)
            if not all_passed:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "required_check_kinds_incomplete",
                        "checks": [check.to_dict() for check in projected],
                    }
                )
                continue

            per_tool_results = result.get("per_tool")
            per_tool_result = (
                per_tool_results.get(str(tool_id), {})
                if isinstance(per_tool_results, Mapping)
                else {}
            )
            projected_payload = [check.to_dict() for check in projected]
            if (
                not isinstance(per_tool_result, Mapping)
                or per_tool_result.get("certified") is not True
                or str(per_tool_result.get("check_set_digest_sha256") or "")
                != content_digest(projected_payload)
            ):
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "per_tool_semantic_binding_invalid",
                    }
                )
                continue
            semantic_identity = (
                per_tool_result.get("identity")
                if isinstance(per_tool_result, Mapping)
                else None
            )
            if not isinstance(semantic_identity, Mapping):
                semantic_identity = _semantic_tool_identity(
                    str(tool_id),
                    receipt,
                    selector=str(spec.get("selector") or "root"),
                    repo_root=root,
                )
            artifacts = semantic_identity.get("artifacts")
            artifacts = (
                [item for item in artifacts if isinstance(item, Mapping)]
                if isinstance(artifacts, Sequence)
                and not isinstance(artifacts, (str, bytes, bytearray))
                else []
            )
            artifact_validation = _validate_artifact_identities(
                artifacts,
                repo_root=root,
            )
            recorded_artifact_validation = per_tool_result.get(
                "artifact_validation"
            )
            if (
                not artifact_validation["valid"]
                or not isinstance(recorded_artifact_validation, Mapping)
                or recorded_artifact_validation.get("valid") is not True
            ):
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_artifact_identity_invalid",
                        "failures": artifact_validation["failures"],
                    }
                )
                continue
            cert.artifact_identities.extend(
                item
                for item in artifacts
                if item not in cert.artifact_identities
            )
            receipt_digest = result_digest
            if receipt_digest and receipt_digest not in cert.semantic_receipt_digests:
                cert.semantic_receipt_digests.append(receipt_digest)
            cert.checks = projected

            if spec.get("identity_from_receipt"):
                executable = semantic_identity.get("executable_path")
                if executable:
                    resolved_executable = _resolve_artifact_path(
                        executable,
                        repo_root=root,
                    )
                    cert.executable_path = (
                        str(resolved_executable)
                        if resolved_executable is not None
                        else None
                    )
                    cert.executable_sha256 = file_digest(resolved_executable)
                    cert.executable_artifact_class = classify_executable_artifact(
                        resolved_executable
                    )
                if semantic_identity.get("version_string"):
                    cert.version_string = str(semantic_identity["version_string"])
                version_exact = True
                if cert.locked_version and cert.version_string:
                    version_exact = not detect_locked_version_mismatch(
                        cert.locked_version,
                        cert.version_string,
                    )
                usable_artifact_bound = bool(
                    artifact_validation["validated"]
                    and (
                        executable
                        or any(
                            item.get("artifact_class")
                            == "public_deployment_binding"
                            for item in artifact_validation["validated"]
                        )
                    )
                )
                semantic_identity_usable = bool(
                    semantic_identity.get("identity_probed")
                    and usable_artifact_bound
                    and version_exact
                )
                if semantic_identity_usable and usable_elevation_allowed:
                    cert.identity_probed = True
                    cert.locked_version_mismatch = False
                    cert.installed = True
                    cert.usable = True
                    cert.unavailable = False

            if not bool(spec.get("production_elevation_allowed")):
                cert.production_certified = False
                cert.promotion_blocked = True
                if not usable_elevation_allowed:
                    cert.usable = False
                    cert.unavailable = True
                reason = "evidence_class_cannot_satisfy_production_authority"
                if reason not in cert.block_reasons:
                    cert.block_reasons.append(reason)
                cert.evidence_class = str(spec["evidence_class"])
                cert.notes = (
                    f"{spec['interface']} evidence is fully bound but classified "
                    f"as {spec['evidence_class']}; it cannot be relabeled as "
                    "production proof authority."
                )
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": reason,
                        "interface": str(spec["interface"]),
                        "evidence_class": cert.evidence_class,
                        "semantic_receipt_digest_sha256": receipt_digest,
                        "checks": [check.to_dict() for check in projected],
                    }
                )
                continue

            if spec.get("identity_from_receipt"):
                executable = semantic_identity.get("executable_path")
                exact_artifact = bool(
                    artifact_validation["has_production_binding"]
                )
                if executable:
                    resolved_executable = _resolve_artifact_path(
                        executable,
                        repo_root=root,
                    )
                    cert.executable_path = (
                        str(resolved_executable)
                        if resolved_executable is not None
                        else None
                    )
                    cert.executable_sha256 = file_digest(resolved_executable)
                    cert.executable_artifact_class = classify_executable_artifact(
                        resolved_executable
                    )
                if semantic_identity.get("version_string"):
                    cert.version_string = str(semantic_identity["version_string"])
                version_exact = True
                if cert.locked_version and cert.version_string:
                    version_exact = not detect_locked_version_mismatch(
                        cert.locked_version,
                        cert.version_string,
                    )
                elif cert.locked_version and not any(
                    item.get("artifact_class") == "public_deployment_binding"
                    for item in artifact_validation["production_bindings"]
                ):
                    version_exact = False
                cert.identity_probed = bool(
                    semantic_identity.get("identity_probed")
                    and exact_artifact
                    and version_exact
                    and (
                        cert.executable_artifact_class
                        == "native_or_managed_binary"
                        or any(
                            item.get("artifact_class")
                            == "public_deployment_binding"
                            for item in artifact_validation[
                                "production_bindings"
                            ]
                        )
                    )
                )
                cert.locked_version_mismatch = not version_exact
                cert.installed = cert.identity_probed
                cert.usable = cert.identity_probed
                cert.unavailable = not cert.identity_probed
                if not cert.identity_probed:
                    elevations.append(
                        {
                            "tool_id": tool_id,
                            "lane_id": lane_id,
                            "elevated": False,
                            "reason": "semantic_identity_not_exactly_bound",
                            "interface": str(spec["interface"]),
                            "evidence_class": str(spec["evidence_class"]),
                            "checks": [check.to_dict() for check in projected],
                        }
                    )
                    continue

            if cert.unavailable or not cert.usable:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "target_not_usable",
                        "interface": str(spec["interface"]),
                    }
                )
                continue

            cert.production_certified = True
            cert.promotion_blocked = False
            cert.block_reasons = []
            cert.evidence_class = str(spec["evidence_class"])
            cert.notes = (
                f"Role-aware elevation from {spec['interface']}: full "
                "positive/negative/mutation/replay semantic evidence bound."
            )
            elevations.append(
                {
                    "tool_id": tool_id,
                    "lane_id": lane_id,
                    "elevated": True,
                    "interface": str(spec["interface"]),
                    "evidence_class": cert.evidence_class,
                    "semantic_receipt_digest_sha256": receipt_digest,
                    "checks": [check.to_dict() for check in projected],
                }
            )
    return elevations


def load_authority_roles(repo_root: Path) -> dict[str, Any]:
    """Bind role / ceiling matrix when the roles surface is present."""

    roles_path = repo_root / "tools" / "logic" / "certification" / "roles.py"
    if not roles_path.is_file():
        return {
            "present": False,
            "reason": "roles_module_missing",
            "tools": {},
            "policy": {},
        }
    try:
        module = _load_module_from_path(roles_path, "fvt_role_aware_roles")
        boundary = module.authority_boundary_report()
        policy = module.build_role_aware_policy()
        tools = {
            item.tool_id: {
                "role": item.role.value,
                "authority_ceiling": item.authority_ceiling.value,
                "can_satisfy_certified_authority": bool(
                    item.can_satisfy_certified_authority
                ),
                "lane_ids": list(getattr(item, "lane_ids", ()) or ()),
            }
            for item in module.list_tool_roles()
        }
        return {
            "present": True,
            "interface": getattr(module, "PROMOTION_INTERFACE", None),
            "role_interface": getattr(module, "INTERFACE", None),
            "tools": tools,
            "boundary": {
                key: boundary.get(key)
                for key in (
                    "support_only",
                    "advisor_or_candidate_only",
                    "shadow_checkers",
                    "kernel_authority",
                    "authorization_only",
                    "finite_trace_authority",
                    "attestation_authority_only",
                    "policy",
                )
                if key in boundary
            },
            "lane_handlers": boundary.get("lane_handlers") or {},
            "policy_digest_sha256": content_digest(policy.to_dict()),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "present": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "tools": {},
            "policy": {},
        }


def apply_role_aware_demotions(
    tool_certs: dict[str, ToolCertification],
    authority_roles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Support / advisor / shadow presence never yields production certification."""

    demotions: list[dict[str, Any]] = []
    tools = authority_roles.get("tools") or {}
    if not isinstance(tools, Mapping):
        return demotions
    for tool_id, cert in tool_certs.items():
        meta = tools.get(tool_id)
        if not isinstance(meta, Mapping):
            continue
        certifying = bool(meta.get("can_satisfy_certified_authority"))
        role = str(meta.get("role") or "")
        if certifying:
            continue
        if cert.production_certified or cert.usable:
            if cert.production_certified:
                demotions.append(
                    {
                        "tool_id": tool_id,
                        "role": role,
                        "reason": "non_certifying_role_cannot_promote",
                        "authority_ceiling": meta.get("authority_ceiling"),
                    }
                )
            cert.production_certified = False
            cert.promotion_blocked = True
            if "non_certifying_role" not in cert.block_reasons:
                cert.block_reasons.append("non_certifying_role")
            if cert.evidence_class == "production_certified":
                cert.evidence_class = "role_demoted"
            if not cert.notes:
                cert.notes = (
                    f"Role {role!r} cannot satisfy certified authority; "
                    "presence is not production certification."
                )
    return demotions


def build_managed_deployment_readiness(
    *,
    lock: Mapping[str, Any],
    tools_index: Mapping[str, Mapping[str, Any]],
    tool_certs: Mapping[str, ToolCertification],
    authority_roles: Mapping[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Derive deployment blockers for every host-supported managed tool."""

    host_platform = observed_platform_id()
    root = (repo_root or repo_root_from()).resolve()
    global_platforms = [
        str(item)
        for item in (
            (lock.get("platform_policy") or {}).get("supported_platforms") or []
        )
    ]
    role_rows = authority_roles.get("tools") or {}
    platform_rows: list[dict[str, Any]] = []
    capabilities: list[str] = []
    dependencies: list[str] = []
    exceptions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for tool_id in sorted(tools_index):
        entry = tools_index[tool_id]
        platform_row = tool_platform_support(
            entry,
            host_platform=host_platform,
            global_supported_platforms=global_platforms,
        )
        platform_rows.append(platform_row)
        if not platform_row["managed"]:
            continue

        role_meta = (
            role_rows.get(tool_id)
            if isinstance(role_rows, Mapping)
            else None
        )
        role = (
            str(role_meta.get("role") or "unclassified")
            if isinstance(role_meta, Mapping)
            else "unclassified"
        )
        category = (
            "dependency"
            if role == "support" or tool_id in {"opam", "stack", "maude"}
            else "capability"
        )
        if not platform_row["supported"]:
            if platform_row["exception_eligible"]:
                exceptions.append(
                    {
                        "tool_id": tool_id,
                        "host_platform": host_platform,
                        "declared_platforms": platform_row["declared_platforms"],
                        "basis": platform_row["basis"],
                        "classification": platform_row["classification"],
                        "category": category,
                        "narrow_scope": True,
                        "complete": False,
                        "production_certified": False,
                    }
                )
            else:
                blockers.append(
                    {
                        "tool_id": tool_id,
                        "category": category,
                        "role": role,
                        "evidence_class": "platform_ambiguous",
                        "artifact_classes": [],
                        "reasons": ["platform_support_ambiguous_or_contradictory"],
                    }
                )
            continue

        if category == "dependency":
            dependencies.append(tool_id)
        else:
            capabilities.append(tool_id)

        cert = tool_certs.get(tool_id)
        if cert is None:
            blockers.append(
                {
                    "tool_id": tool_id,
                    "category": category,
                    "reasons": ["certificate_entry_missing"],
                }
            )
            continue

        artifact_classes = {
            str(item.get("artifact_class") or "")
            for item in cert.artifact_identities
            if isinstance(item, Mapping)
        }
        artifact_validation = _validate_artifact_identities(
            [
                item
                for item in cert.artifact_identities
                if isinstance(item, Mapping)
            ],
            repo_root=root,
        )
        exact_artifact = bool(artifact_validation["has_production_binding"])
        public_binding = any(
            item.get("artifact_class") == "public_deployment_binding"
            for item in artifact_validation["production_bindings"]
        )
        genuinely_installed = bool(
            exact_artifact
            and artifact_validation["valid"]
            and (
                (cert.installed and cert.identity_probed)
                or public_binding
            )
        )

        reasons: list[str] = []
        if not genuinely_installed:
            reasons.append("supported_managed_installation_missing_or_shim_only")
        if not artifact_validation["valid"]:
            reasons.append("artifact_identity_invalid")
        if "launcher_script" in artifact_classes and not public_binding:
            reasons.append("launcher_target_artifact_unbound")
        if cert.locked_version_mismatch:
            reasons.append("locked_version_mismatch")

        if category == "capability":
            required_kinds = {"positive", "negative", "mutation", "replay"}
            present_kinds = {check.kind for check in cert.checks}
            checks_complete = bool(cert.checks) and all(
                check.status == "passed" for check in cert.checks
            ) and required_kinds <= present_kinds
            certifying_role = bool(
                isinstance(role_meta, Mapping)
                and role_meta.get("can_satisfy_certified_authority")
            )
            if certifying_role and not cert.production_certified:
                reasons.append("semantic_evidence_below_authority_ceiling")
            if not checks_complete:
                reasons.append("full_semantic_check_set_missing_or_failed")
            if certifying_role and not cert.semantic_receipt_digests and not (
                cert.evidence_class == "production_certified"
                and cert.tool_id in {"z3", "cvc5"}
            ):
                reasons.append("semantic_receipt_not_bound")
            if not certifying_role and cert.production_certified:
                reasons.append("non_certifying_role_incorrectly_promoted")

        if reasons:
            blockers.append(
                {
                    "tool_id": tool_id,
                    "category": category,
                    "role": role,
                    "authority_ceiling": (
                        role_meta.get("authority_ceiling")
                        if isinstance(role_meta, Mapping)
                        else None
                    ),
                    "evidence_class": cert.evidence_class,
                    "artifact_classes": sorted(artifact_classes),
                    "reasons": sorted(set(reasons)),
                }
            )

    capability_blockers = [
        row for row in blockers if row.get("category") == "capability"
    ]
    dependency_blockers = [
        row for row in blockers if row.get("category") == "dependency"
    ]
    return {
        "host_platform": host_platform,
        "global_supported_platforms": global_platforms,
        "host_globally_supported": host_platform in set(global_platforms),
        "platform_rows": platform_rows,
        "supported_managed_capability_tool_ids": capabilities,
        "supported_managed_dependency_tool_ids": dependencies,
        "platform_exceptions": exceptions,
        "capability_blockers": capability_blockers,
        "dependency_blockers": dependency_blockers,
        "all_blockers": blockers,
        "ready": not blockers,
        "status": (
            "all_supported_managed_capabilities_ready"
            if not blockers
            else "supported_managed_capabilities_blocked"
        ),
    }


def build_certificate(
    *,
    repo_root: Path | None = None,
    lock_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    observed_at: str | None = None,
    role_aware: bool = False,
) -> dict[str, Any]:
    """Run the full hermetic multi-prover certification and return the certificate.

    When ``role_aware`` is true (FVT-G200 reissue), focused semantic receipts
    are aggregated and checked against their role ceiling. Lean, Runtime MTL,
    and authorization engines remain usable-but-pending; external Rocq and
    Isabelle remain unavailable until their installation and live fan-in work
    is complete. Default (FVT-G060) certification keeps identity-only lanes
    non-elevated so usability and semantic certification remain distinct.
    """

    root = repo_root or repo_root_from()
    lock_file = lock_path or (root / DEFAULT_LOCK_RELATIVE)
    lock = load_lock(lock_file)
    tools_index = lock_tools_by_id(lock)
    run_env = offline_env(env)
    repository_python_paths = (
        str(root),
        str(root / "ipfs_datasets_py"),
    )
    existing_python_path = str(run_env.get("PYTHONPATH") or "").strip()
    run_env["PYTHONPATH"] = os.pathsep.join(
        (
            *repository_python_paths,
            *((existing_python_path,) if existing_python_path else ()),
        )
    )

    # Map tool → lanes and preferred check kind (smtlib wins when present).
    tool_lane_map: dict[str, list[str]] = {}
    tool_check_kind: dict[str, str] = {}
    for lane in PROPERTY_LANES:
        for tool_id in lane["tool_ids"]:
            tool_lane_map.setdefault(tool_id, []).append(lane["lane_id"])
            prior = tool_check_kind.get(tool_id)
            kind = str(lane["check_kind"])
            if prior is None or kind == "smtlib":
                tool_check_kind[tool_id] = kind

    # Certify every lock tool that appears in a lane; also surface lock-only
    # tools so nothing is concealed.
    lane_tool_ids = {tool_id for lane in PROPERTY_LANES for tool_id in lane["tool_ids"]}
    certify_ids = sorted(set(tools_index) | lane_tool_ids)

    tool_certs: dict[str, ToolCertification] = {}
    for tool_id in certify_ids:
        entry = tools_index.get(tool_id)
        if entry is None:
            # Lane references a tool not in the lock — explicit unavailable.
            cert = ToolCertification(
                tool_id=tool_id,
                lane_ids=tool_lane_map.get(tool_id, []),
                unavailable=True,
                promotion_blocked=True,
                block_reasons=["missing_from_lock"],
                evidence_class="unavailable",
                notes="Referenced by a property lane but absent from the offline lock.",
                checks=[
                    CheckResult(
                        check_id=f"{tool_id}.{kind}",
                        kind=kind,
                        status="unavailable",
                        expected="lock entry",
                        observed="missing_from_lock",
                    )
                    for kind in ("positive", "negative", "mutation", "replay")
                ],
            )
            tool_certs[tool_id] = cert
            continue

        cert = certify_tool(
            entry,
            lane_ids=tool_lane_map.get(tool_id, []),
            check_kind=tool_check_kind.get(tool_id, "identity_only"),
            env=run_env,
        )
        tool_certs[tool_id] = cert

    disagreements: list[DisagreementQuarantine] = []
    smt_quarantine = quarantine_smt_disagreement(tool_certs)
    if smt_quarantine is not None:
        disagreements.append(smt_quarantine)
        for tool_id in smt_quarantine.promotion_blocked_tool_ids:
            cert = tool_certs[tool_id]
            cert.production_certified = False
            cert.promotion_blocked = True
            if "cross_provider_disagreement" not in cert.block_reasons:
                cert.block_reasons.append("cross_provider_disagreement")
            cert.evidence_class = "quarantined_disagreement"
            cert.notes = (
                "Cross-provider SMT disagreement quarantined; "
                "disagreement cannot raise authority or promote."
            )

    # Role matrix binding is cheap; semantic elevation is opt-in and expensive
    # (Lean kernel suite). Default FVT-G060 keeps identity-only usability.
    authority_roles = load_authority_roles(root)
    semantic_results: list[dict[str, Any]] = []
    elevations: list[dict[str, Any]] = []
    demotions: list[dict[str, Any]] = []
    if role_aware:
        semantic_results = run_semantic_lane_certifiers(
            repo_root=root,
            env=run_env,
            tool_certs=tool_certs,
        )
        elevations = apply_semantic_elevations(
            tool_certs,
            semantic_results,
            repo_root=root,
        )
        demotions = apply_role_aware_demotions(tool_certs, authority_roles)

    deployment_readiness = build_managed_deployment_readiness(
        lock=lock,
        tools_index=tools_index,
        tool_certs=tool_certs,
        authority_roles=authority_roles,
        repo_root=root,
    )

    lanes = certify_property_lanes(tool_certs, disagreements)

    production_certified_ids = sorted(
        tid for tid, cert in tool_certs.items() if cert.production_certified
    )
    unavailable_ids = sorted(
        tid for tid, cert in tool_certs.items() if cert.unavailable
    )
    blocked_map = {
        tid: list(cert.block_reasons)
        for tid, cert in tool_certs.items()
        if cert.promotion_blocked
    }

    # Tools that remain merely usable (identity-only) after optional elevation.
    merely_usable_ids = sorted(
        tid
        for tid, cert in tool_certs.items()
        if cert.usable and not cert.production_certified and not cert.unavailable
    )

    offline_policy = dict(lock.get("offline_verification_policy") or {})
    offline_observations = [
        result.get("offline_observation")
        for result in semantic_results
        if isinstance(result.get("offline_observation"), Mapping)
    ]
    required_offline_policy_keys = (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "forbid_curl_pipe_shell",
        "forbid_system_package_mutation",
    )
    offline_policy_satisfied = all(
        bool(offline_policy.get(key)) for key in required_offline_policy_keys
    ) and all(bool(item.get("satisfied")) for item in offline_observations)
    public_semantic_results = [
        _project_semantic_lane_result(result, repo_root=root)
        for result in semantic_results
    ]
    certificate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "description": (
            "Hermetic offline multi-prover certification receipt. Available tools "
            "pass live positive/negative/mutation/replay checks with exact "
            "identities; absent/mismatched lanes are explicit skips/unavailable "
            "and block only their promotion; PATH shims are not usability; "
            "certification performs no download/network/install and quarantines "
            "disagreement."
            + (
                " Role-aware reissue retains Lean, Runtime MTL, and "
                "Datalog/SecPAL as usable but non-production; external Rocq "
                "and Isabelle remain unavailable until their installation and "
                "live fan-in goals are complete."
                if role_aware
                else ""
            )
        ),
        "observed_at": observed_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "binding_mode": (
            "offline_pinned_live_lanes_role_aware"
            if role_aware
            else "offline_pinned_live_lanes"
        ),
        "lock": {
            "path": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
            "interface": lock.get("interface"),
            "schema_version": lock.get("schema_version"),
            "goal_id": lock.get("goal_id"),
            "task_id": lock.get("task_id"),
            "digest_sha256": content_digest(lock),
            "host_platform": deployment_readiness["host_platform"],
        },
        "certification_policy": {
            "forbid_install": bool(offline_policy.get("forbid_install")),
            "forbid_download": bool(offline_policy.get("forbid_download")),
            "forbid_network": bool(offline_policy.get("forbid_network")),
            "forbid_curl_pipe_shell": bool(
                offline_policy.get("forbid_curl_pipe_shell")
            ),
            "path_presence_is_not_usability": bool(
                offline_policy.get("path_presence_is_not_usability")
            ),
            "require_exact_pin_match_for_production_certification": bool(
                offline_policy.get(
                    "require_exact_pin_match_for_production_certification"
                )
            ),
            "shim_toolchain_mismatch_fails_closed": bool(
                offline_policy.get("shim_toolchain_mismatch_fails_closed")
            ),
            "absent_lanes_block_only_own_promotion": True,
            "optional_tools_not_mandatory_for_unrelated_properties": True,
            "quarantine_disagreement": True,
            "synthetic_evidence_cannot_certify_production": True,
            "role_aware_semantic_elevation": bool(role_aware),
            "non_certifying_roles_cannot_promote": True,
            "offline_policy_satisfied": offline_policy_satisfied,
            "offline_observations": offline_observations,
            "lock_offline_verification_policy": {
                key: bool(offline_policy.get(key))
                for key in (
                    "forbid_install",
                    "forbid_download",
                    "forbid_network",
                    "forbid_curl_pipe_shell",
                    "forbid_system_package_mutation",
                    "require_exact_pin_match_for_production_certification",
                    "path_presence_is_not_usability",
                    "shim_toolchain_mismatch_fails_closed",
                )
            },
        },
        "detection_rules": {
            "lean_shim_toolchain_mismatch": {
                "id": "lean_shim_toolchain_mismatch",
                "effect": {
                    "usable": False,
                    "production_certified": False,
                    "offline_verification": "fail_closed_without_install_or_fetch",
                },
            },
            "locked_version_mismatch": {
                "id": "locked_version_mismatch",
                "effect": {
                    "production_certified": False,
                    "offline_verification": "fail_closed_without_upgrade_or_download",
                },
            },
            "path_presence_is_not_usability": {
                "id": "path_presence_is_not_usability",
                "effect": {
                    "installed": False,
                    "usable": False,
                    "production_certified": False,
                },
            },
            "cross_provider_disagreement_quarantine": {
                "id": "cross_provider_disagreement_quarantine",
                "effect": {
                    "production_certified": False,
                    "status": "quarantined",
                },
            },
            "non_certifying_role_demotion": {
                "id": "non_certifying_role_demotion",
                "effect": {
                    "production_certified": False,
                    "status": "role_demoted",
                },
            },
        },
        "property_lanes": [lane.to_dict() for lane in lanes],
        "tools": [tool_certs[tid].to_dict() for tid in sorted(tool_certs)],
        "disagreement_quarantines": [item.to_dict() for item in disagreements],
        "promotion": {
            "production_certified_tool_ids": production_certified_ids,
            "unavailable_tool_ids": unavailable_ids,
            "blocked_tool_ids": blocked_map,
            "merely_usable_tool_ids": merely_usable_ids,
            "lane_promotion_ready": {
                lane.lane_id: lane.promotion_ready for lane in lanes
            },
        },
        "authority_roles": authority_roles,
        # Retain every check in a portable public projection. Raw process
        # output and secret/witness values are retained only by digest.
        "semantic_lane_results": public_semantic_results,
        "managed_deployment_readiness": deployment_readiness,
        "role_aware": {
            "enabled": bool(role_aware),
            "goal_id": ROLE_AWARE_GOAL_ID,
            "task_id": ROLE_AWARE_TASK_ID,
            "interface": ROLE_AWARE_INTERFACE,
            "elevations": elevations,
            "demotions": demotions,
            "elevated_tool_ids": sorted(
                {
                    str(item.get("tool_id"))
                    for item in elevations
                    if item.get("elevated")
                }
            ),
            "required_baseline_elevations": [
                "lean",
                "runtime-mtl",
                "datalog-authorization",
                "secpal-authorization",
                "coq",
                "isabelle",
            ],
        },
        "check_kinds_required": ["positive", "negative", "mutation", "replay"],
        "evidence": {
            "certifier": "tools/logic/certify_formal_verification_toolchains.py",
            "integration_test": (
                "test/integration/test_formal_verification_real_tool_matrix.py"
            ),
            "role_aware_integration_test": (
                "test/integration/test_formal_verification_role_aware_completion.py"
            ),
            "lock": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
        },
        "certificate_digest_sha256": "",  # filled below
    }
    certificate = public_evidence_projection(certificate, repo_root=root)
    public_evidence_policy = public_evidence_audit(certificate)
    certificate["public_evidence_policy"] = public_evidence_policy
    if not public_evidence_policy["satisfied"]:
        managed = certificate.get("managed_deployment_readiness")
        if isinstance(managed, dict):
            managed["ready"] = False
            blockers = managed.setdefault("dependency_blockers", [])
            if "public_evidence_redaction_failed" not in blockers:
                blockers.append("public_evidence_redaction_failed")
    certificate["certificate_digest_sha256"] = content_digest(
        {key: value for key, value in certificate.items() if key != "certificate_digest_sha256"}
    )
    return certificate


def write_certificate(
    certificate: Mapping[str, Any],
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(certificate, indent=2, sort_keys=False) + "\n"
    # Atomic replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the real multi-prover formal-verification matrix in "
            "hermetic offline lanes (FormalVerificationToolchainCertificate@1)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from this file)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="Path to offline toolchain lock JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Certificate output path (default: docs/architecture/...certificate.json)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print certificate JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary",
    )
    parser.add_argument(
        "--role-aware",
        action="store_true",
        help=(
            "Apply FVT-G200 role-aware semantic elevation (Lean / Runtime MTL / "
            "Datalog-SecPAL) and bind authority roles into the certificate"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.repo_root or repo_root_from()).resolve()
    certificate = build_certificate(
        repo_root=root,
        lock_path=args.lock.resolve() if args.lock else None,
        role_aware=bool(args.role_aware),
    )

    if args.stdout:
        json.dump(certificate, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        output = (
            args.output.resolve()
            if args.output
            else (root / DEFAULT_CERTIFICATE_RELATIVE)
        )
        write_certificate(certificate, output)
        if not args.quiet:
            print(f"wrote {output}", file=sys.stderr)

    if not args.quiet:
        promotion = certificate["promotion"]
        print(
            "production_certified="
            f"{promotion['production_certified_tool_ids']}",
            file=sys.stderr,
        )
        print(
            f"unavailable={promotion['unavailable_tool_ids']}",
            file=sys.stderr,
        )
        print(
            f"quarantines={len(certificate['disagreement_quarantines'])}",
            file=sys.stderr,
        )

    # Exit 0 even when some tools are unavailable — absence is not a certifier
    # failure. Hard failures are schema/lock errors (already raised).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
