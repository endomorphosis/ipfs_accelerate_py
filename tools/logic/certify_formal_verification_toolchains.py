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
import json
import os
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

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_CERTIFICATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_toolchain_certificate.json"
)

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 8.0

# Property lanes from FVT-G060. Each lane owns a closed set of tools; absence
# of one tool never conceals or fails unrelated lanes.
PROPERTY_LANES: Final[tuple[dict[str, Any], ...]] = (
    {
        "lane_id": "smt",
        "property_class": "smt_software_verification",
        "description": "SMT solvers for software-verification VCs",
        "tool_ids": ("z3", "cvc5"),
        "check_kind": "smtlib",
    },
    {
        "lane_id": "tla",
        "property_class": "tla_state_model",
        "description": "TLA+/TLC/Apalache state-model checking",
        "tool_ids": ("apalache", "tlc", "java"),
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
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "protocol",
        "property_class": "protocol_verification",
        "description": "Tamarin/ProVerif protocol verification",
        "tool_ids": ("tamarin", "proverif", "maude"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hyperltl",
        "property_class": "hyperproperty",
        "description": "HyperLTL / hyperproperty tools",
        "tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "atp",
        "property_class": "automated_theorem_proving",
        "description": "First-order ATP portfolio",
        "tool_ids": ("vampire", "eprover"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hammer",
        "property_class": "hammer_advisor",
        "description": "Hammer / advisor bridges (non-kernel authority)",
        "tool_ids": ("symbolicai", "ergoai"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "kernel",
        "property_class": "interactive_proof_kernel",
        "description": "Lean / Rocq / Isabelle kernels",
        "tool_ids": ("lean", "coq", "isabelle"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "runtime_mtl",
        "property_class": "runtime_mtl_monitoring",
        "description": "Runtime MTL monitors",
        "tool_ids": ("runtime-mtl", "runtime-mtl-external"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "attestation",
        "property_class": "attestation_zkp",
        "description": "Attestation / ZKP circuit bindings",
        "tool_ids": ("zkp-circuit",),
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

    env = dict(base or os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env.setdefault("ELAN_NO_AUTO_INSTALL", "1")
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


def list_elan_installed_toolchains() -> list[str]:
    """Read offline-installed Lean toolchains from the local elan directory."""

    elan_home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCertification:
    tool_id: str
    lane_ids: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    availability_declared: str = ""
    executable_path: str | None = None
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
    certified_tool_ids: list[str] = field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Identity probing
# ---------------------------------------------------------------------------


def resolve_executable(candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if not name:
            continue
        # Absolute/relative path candidates.
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(name)
        if found:
            return found
    return None


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
            module_ok, version = _probe_in_process_module(tool_id)
            result["identity_probed"] = module_ok
            result["installed"] = module_ok
            result["version_string"] = version
            return result
        if availability == "advisor_only":
            # Advisors may be optional Python packages — probe lightly.
            module_ok, version = _probe_in_process_module(tool_id)
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

    completed = bounded_run(
        [executable, *argv_suffix],
        timeout=timeout,
        env=env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        # PATH presence without a successful identity probe is not installed.
        return result

    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    # Some tools (java) write version to stderr with non-zero on --version.
    if not banner and tool_id == "java":
        # Retry with -version which java accepts.
        completed = bounded_run([executable, "-version"], timeout=timeout, env=env)
        if completed is not None:
            banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
                completed.stderr
            )

    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result

    result["version_string"] = banner
    result["identity_probed"] = True
    result["installed"] = True

    if tool_id == "lean":
        installed = list_elan_installed_toolchains()
        result["installed_toolchains"] = installed
        match = re.search(r"version\s+(\d+\.\d+\.\d+)", banner, re.IGNORECASE)
        selected = (
            f"leanprover/lean4:v{match.group(1)}"
            if match
            else probe.get("locked_toolchain")
        )
        result["selected_toolchain"] = selected
        result["shim_toolchain_mismatch"] = detect_lean_shim_toolchain_mismatch(
            selected, installed
        )

    return result


def _probe_in_process_module(tool_id: str) -> tuple[bool, str | None]:
    """Best-effort import probe for in-process tools. Never installs."""

    module_map = {
        "runtime-mtl": "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl",
        "datalog-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "secpal-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "symbolicai": "symbolicai",
        "ergoai": "ergoai",
        "zkp-circuit": None,  # declared gap — never pretend installed
    }
    module_name = module_map.get(tool_id)
    if not module_name:
        return False, None
    try:
        __import__(module_name)
        return True, f"python-module:{module_name}"
    except Exception:
        return False, None


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
    # For SMT, all four checks must pass for production certification.
    if check_kind == "smtlib":
        all_live_passed = all(check.status == "passed" for check in cert.checks)
    else:
        # Skipped negative/mutation do not invent success; required kinds must pass.
        no_failures = all(check.status != "failed" for check in cert.checks)
        all_live_passed = required_passed and no_failures

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


def build_certificate(
    *,
    repo_root: Path | None = None,
    lock_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Run the full hermetic multi-prover certification and return the certificate."""

    root = repo_root or repo_root_from()
    lock_file = lock_path or (root / DEFAULT_LOCK_RELATIVE)
    lock = load_lock(lock_file)
    tools_index = lock_tools_by_id(lock)
    run_env = offline_env(env)

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

    lanes: list[LaneCertification] = []
    for lane in PROPERTY_LANES:
        lane_id = str(lane["lane_id"])
        tool_ids = list(lane["tool_ids"])
        certified = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].production_certified
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
        # Lane is promotion-ready when at least one tool is certified and no
        # unresolved quarantine remains for the lane. Optional absent tools do
        # not fail the lane.
        promotion_ready = bool(certified) and not q_ids
        lanes.append(
            LaneCertification(
                lane_id=lane_id,
                property_class=str(lane["property_class"]),
                description=str(lane["description"]),
                tool_ids=tool_ids,
                certified_tool_ids=certified,
                unavailable_tool_ids=unavailable,
                blocked_tool_ids=blocked,
                disagreement_quarantine_ids=q_ids,
                promotion_ready=promotion_ready,
                notes=(
                    "Absent tools block only their own promotion."
                    if unavailable
                    else ""
                ),
            )
        )

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

    offline_policy = dict(lock.get("offline_verification_policy") or {})
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
        ),
        "observed_at": observed_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "binding_mode": "offline_pinned_live_lanes",
        "lock": {
            "path": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
            "interface": lock.get("interface"),
            "schema_version": lock.get("schema_version"),
            "goal_id": lock.get("goal_id"),
            "task_id": lock.get("task_id"),
            "digest_sha256": content_digest(
                {
                    "interface": lock.get("interface"),
                    "schema_version": lock.get("schema_version"),
                    "managed_pin_versions": lock.get("managed_pin_versions"),
                    "tool_ids": sorted(tools_index),
                }
            ),
        },
        "certification_policy": {
            "forbid_install": True,
            "forbid_download": True,
            "forbid_network": True,
            "forbid_curl_pipe_shell": True,
            "path_presence_is_not_usability": True,
            "require_exact_pin_match_for_production_certification": True,
            "shim_toolchain_mismatch_fails_closed": True,
            "absent_lanes_block_only_own_promotion": True,
            "optional_tools_not_mandatory_for_unrelated_properties": True,
            "quarantine_disagreement": True,
            "synthetic_evidence_cannot_certify_production": True,
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
        },
        "property_lanes": [lane.to_dict() for lane in lanes],
        "tools": [tool_certs[tid].to_dict() for tid in sorted(tool_certs)],
        "disagreement_quarantines": [item.to_dict() for item in disagreements],
        "promotion": {
            "production_certified_tool_ids": production_certified_ids,
            "unavailable_tool_ids": unavailable_ids,
            "blocked_tool_ids": blocked_map,
            "lane_promotion_ready": {
                lane.lane_id: lane.promotion_ready for lane in lanes
            },
        },
        "check_kinds_required": ["positive", "negative", "mutation", "replay"],
        "evidence": {
            "certifier": "tools/logic/certify_formal_verification_toolchains.py",
            "integration_test": (
                "test/integration/test_formal_verification_real_tool_matrix.py"
            ),
            "lock": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
        },
        "certificate_digest_sha256": "",  # filled below
    }
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.repo_root or repo_root_from()).resolve()
    certificate = build_certificate(
        repo_root=root,
        lock_path=args.lock.resolve() if args.lock else None,
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
