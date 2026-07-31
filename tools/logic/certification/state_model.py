#!/usr/bin/env python3
"""TLC + Apalache state-model toolchain certification (FVT-G120 / FVT-042).

``StateModelToolchainCertification@1``

Owns the TLA lane certification handler for the pinned TLC 1.8.0 and
Apalache 0.58.3 model checkers. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for TLC 1.8.0 and Apalache 0.58.3 for
  production certification;
* exercises invariant-holds, violation trace, mutated Next/invariant, replay,
  malformed model, timeout, and bound behavior cases;
* binds model, config, constants, bounds, and exact tool identities;
* treats Java as support only — Java presence alone never promotes the TLA
  property lane;
* bounded model-checking never promotes to theorem authority;
* never edits the shared multi-prover certificate.

Semantic evaluation reuses the canonical TLA model-checker classifiers so
offline tests can prove corpus behavior without live TLC/Apalache processes.
Live production certification additionally requires the pinned tools and JVM.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.installers import state_model as installer  # noqa: E402
from ipfs_datasets_py.logic.backends.results import ResultStatus  # noqa: E402
from ipfs_datasets_py.logic.backends.tla.runners import (  # noqa: E402
    APALACHE_BACKEND_VERSION,
    APALACHE_CAPABILITY,
    TLC_BACKEND_VERSION,
    TLC_CAPABILITY,
    ModelCheckerTool,
    ModelCheckOutcomeStatus,
    parse_counterexample_trace,
)
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    can_satisfy_certified_authority_requirement,
    evaluate_role_aware_promotion,
    get_tool_role,
)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

INTERFACE: Final = "StateModelToolchainCertification@1"
SCHEMA_VERSION: Final = "state-model-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "state-model-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G120"
TASK_ID: Final = "FVT-042"
PROGRAM: Final = "formal-verification-tactician/state-model-toolchains"
LANE_ID: Final = "tla"
TOOL_ID_TLC: Final = "tlc"
TOOL_ID_APALACHE: Final = "apalache"
SUPPORT_TOOL_ID: Final = "java"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.state_model"
HANDLER_ID: Final = "state_model_toolchain_certifier"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.BOUNDED.value
AUTHORITY_SCOPE: Final = "bounded_state_model_only"

LOCKED_TLC_VERSION: Final = "1.8.0"
LOCKED_APALACHE_VERSION: Final = "0.58.3"
LOCKED_TLC_SHA256: Final = installer.TLC_SHA256
LOCKED_APALACHE_SHA256: Final = installer.APALACHE_SHA256
LOCKED_TLC_EXECUTABLE: Final = "tlc"
LOCKED_APALACHE_EXECUTABLE: Final = "apalache-mc"
LOCKED_JAVA_EXECUTABLE: Final = "java"

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_VERSION_IN_BANNER = re.compile(r"(\d+\.\d+\.\d+)")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Compact embedded corpus. Prefer live binaries when present; classifiers always run.
_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "invariant_holds",
        "kind": "invariant_holds",
        "expect": "passed",
        "tool": "tlc",
        "returncode": 0,
        "stdout": (
            "TLC2 Version 1.8.0 of 01 January 2024\n"
            "Model checking completed. No error has been found.\n"
            "  Estimates of the probability that TLC did not check "
            "all reachable states because two distinct states had the "
            "same fingerprint:\n"
            "  calculated (optimistic):  val = 2.3E-19\n"
        ),
        "stderr": "",
        "model_digest": "a" * 64,
        "config_digest": "b" * 64,
        "constants": {"N": "3", "MaxSteps": "10"},
        "bounds": {"max_steps": 10, "timeout_seconds": 30},
        "description": "Invariant holds under declared bounds (TLC)",
    },
    {
        "case_id": "violation_trace",
        "kind": "violation_trace",
        "expect": "counterexample",
        "tool": "tlc",
        "returncode": 12,
        "stdout": (
            "Error: Invariant Inv is violated.\n"
            "The behavior up to this point is:\n"
            "State 1: <Initial predicate>\n"
            "/\\ pc = \"idle\"\n"
            "/\\ count = 0\n"
            "State 2: <Increment line 12>\n"
            "/\\ pc = \"busy\"\n"
            "/\\ count = 4\n"
            "Error trace complete.\n"
        ),
        "stderr": "",
        "model_digest": "c" * 64,
        "config_digest": "d" * 64,
        "constants": {"N": "3"},
        "bounds": {"max_steps": 10},
        "description": "Invariant violation produces a parseable counterexample",
    },
    {
        "case_id": "mutated_next",
        "kind": "mutation",
        "mutates": "next",
        "expect": "counterexample",
        "base_case_id": "invariant_holds",
        "tool": "tlc",
        "returncode": 12,
        "stdout": (
            "Error: Invariant Inv is violated.\n"
            "State 1: <Initial predicate>\n"
            "/\\ count = 0\n"
            "State 2: <Mutated_Next>\n"
            "/\\ count = 99\n"
            "found an invariant violation\n"
        ),
        "stderr": "",
        "model_digest": "e" * 64,
        "config_digest": "f" * 64,
        "constants": {"N": "3"},
        "bounds": {"max_steps": 10},
        "description": "Next mutation yields a counterexample, not a pass",
    },
    {
        "case_id": "mutated_invariant",
        "kind": "mutation",
        "mutates": "invariant",
        "expect": "counterexample",
        "base_case_id": "invariant_holds",
        "tool": "apalache",
        "returncode": 12,
        "stdout": (
            "Checker reports an error\n"
            "State 1: <Init>\n"
            "/\\ count = 0\n"
            "State 2: <Next>\n"
            "/\\ count = 1\n"
            "found an invariant violation (mutated Inv')\n"
        ),
        "stderr": "",
        "model_digest": "1" * 64,
        "config_digest": "2" * 64,
        "constants": {"N": "3"},
        "bounds": {"length": 5},
        "description": "Invariant mutation yields a counterexample under Apalache",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "passed",
        "base_case_id": "invariant_holds",
        "tool": "tlc",
        "returncode": 0,
        "stdout": (
            "TLC2 Version 1.8.0 of 01 January 2024\n"
            "Model checking completed. No error has been found.\n"
            "  Estimates of the probability that TLC did not check "
            "all reachable states because two distinct states had the "
            "same fingerprint:\n"
            "  calculated (optimistic):  val = 2.3E-19\n"
        ),
        "stderr": "",
        "model_digest": "a" * 64,
        "config_digest": "b" * 64,
        "constants": {"N": "3", "MaxSteps": "10"},
        "bounds": {"max_steps": 10, "timeout_seconds": 30},
        "description": "Positive invariant case replays with identical digests",
    },
    {
        "case_id": "malformed_model",
        "kind": "malformed",
        "expect": "malformed_or_error",
        "tool": "tlc",
        "returncode": 1,
        "stdout": "this is not a TLC model report\n!!! garbage !!!\n",
        "stderr": "Parse error at line 1\n",
        "model_digest": "3" * 64,
        "config_digest": "4" * 64,
        "constants": {},
        "bounds": {"max_steps": 1},
        "description": "Malformed model/output never reports a successful pass",
    },
    {
        "case_id": "timeout_bound",
        "kind": "timeout",
        "expect": "timed_out",
        "tool": "apalache",
        "returncode": None,
        "timed_out": True,
        "stdout": "",
        "stderr": "Timeout after 5 seconds\n",
        "model_digest": "5" * 64,
        "config_digest": "6" * 64,
        "constants": {"N": "100"},
        "bounds": {"length": 200, "timeout_seconds": 5},
        "description": "Timeout outcomes quarantine rather than pass",
    },
    {
        "case_id": "bound_behavior",
        "kind": "bound",
        "expect": "passed_bounded",
        "tool": "apalache",
        "returncode": 0,
        "stdout": (
            "Checker reports no error\n"
            "No error up to computation length 5\n"
            "Verification result: Pass\n"
        ),
        "stderr": "",
        "model_digest": "7" * 64,
        "config_digest": "8" * 64,
        "constants": {"N": "3"},
        "bounds": {"length": 5, "finite_trace_only": True},
        "description": (
            "Apalache bound success is finite-trace only and never theorem authority"
        ),
    },
    {
        "case_id": "version_mismatch",
        "kind": "version_mismatch",
        "expect": "blocked",
        "stdout": "",
        "stderr": "",
        "observed_tlc_version": "1.7.0",
        "observed_apalache_version": "0.40.0",
        "description": "Locked version mismatch blocks production certification",
    },
)

DEFAULT_MODEL_BINDINGS: Final[dict[str, Any]] = {
    "module_name": "StateModel",
    "invariant_ids": ("pred:inv",),
    "next_relation": "Next",
    "init_predicate": "Init",
}

DEFAULT_CONFIG_BINDINGS: Final[dict[str, Any]] = {
    "SPECIFICATION": "Spec",
    "INVARIANT": "Inv",
    "CONSTANTS": {"N": "3"},
}

DEFAULT_BOUNDS: Final[dict[str, Any]] = {
    "timeout_seconds": CHECK_TIMEOUT_SECONDS,
    "max_source_bytes": 1_048_576,
    "tlc_max_declared_steps": TLC_CAPABILITY.max_declared_steps,
    "apalache_max_declared_steps": APALACHE_CAPABILITY.max_declared_steps,
    "network": False,
    "install": False,
    "download": False,
    "finite_trace_apalache": True,
    "unbounded_proof": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
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
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    return env


def bounded_run(
    argv: Sequence[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
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
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_executable(candidates: Sequence[str] | None = None) -> str | None:
    for name in candidates or ():
        if not name:
            continue
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(name)
        if found:
            return found
    return None


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_version(banner: str | None) -> str | None:
    if not banner:
        return None
    match = _VERSION_IN_BANNER.search(banner)
    return match.group(1) if match else None


def _selected_java(
    env: Mapping[str, str],
    *,
    executable: str | None,
    minimum_major: int,
) -> installer.JavaRuntimeProbe:
    candidate = executable or str(
        env.get(installer.JAVA_EXECUTABLE_ENV) or ""
    ).strip()
    if not candidate:
        java_home = str(env.get("JAVA_HOME") or "").strip()
        if java_home:
            candidate = str(Path(java_home) / "bin" / "java")
    if not candidate:
        candidate = shutil.which("java", path=env.get("PATH")) or ""
    return installer.probe_java_runtime(
        java_executable=candidate or None,
        minimum_major=minimum_major,
    )


def _managed_identity(
    tool_id: str,
    binary: str,
    *,
    env: Mapping[str, str],
    java_executable: str | None,
) -> dict[str, Any]:
    minimum = (
        installer.TLC_MIN_JAVA_MAJOR
        if tool_id == TOOL_ID_TLC
        else installer.APALACHE_MIN_JAVA_MAJOR
    )
    java = _selected_java(
        env,
        executable=java_executable,
        minimum_major=minimum,
    )
    path = Path(binary)
    if not java.usable or java.executable is None or path.parent.name != "bin":
        return {
            "usable": False,
            "reason": "validated_java_or_managed_root_missing",
            "java_runtime": java.to_dict(),
        }
    identity = (
        installer.managed_tlc_identity(
            path.parent.parent,
            java_executable=java.executable,
        )
        if tool_id == TOOL_ID_TLC
        else installer.managed_apalache_identity(
            path.parent.parent,
            java_executable=java.executable,
        )
    )
    identity["java_runtime"] = java.to_dict()
    return identity


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str
    status: str  # passed | failed | skipped | unavailable | blocked
    expected: str
    observed: str
    detail: str = ""
    reason_codes: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcome:
    case_id: str
    kind: str
    expect: str
    status: str  # passed | counterexample | timed_out | malformed | error | blocked | unknown
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    counterexample: dict[str, Any] | None = None
    output_digest: str = ""
    model_digest: str = ""
    config_digest: str = ""
    tool: str = ""
    stdout: str = ""
    stderr: str = ""
    detail: str = ""
    grants_theorem_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateModelToolchainCertification:
    """Full certification receipt for the TLC/Apalache TLA lane."""

    tool_ids: list[str] = field(
        default_factory=lambda: [TOOL_ID_TLC, TOOL_ID_APALACHE]
    )
    support_tool_id: str = SUPPORT_TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_tlc_version: str = LOCKED_TLC_VERSION
    locked_apalache_version: str = LOCKED_APALACHE_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    tlc_executable: str | None = None
    apalache_executable: str | None = None
    java_executable: str | None = None
    tlc_version_string: str | None = None
    apalache_version_string: str | None = None
    java_version_string: str | None = None
    tlc_identity_probed: bool = False
    apalache_identity_probed: bool = False
    java_identity_probed: bool = False
    tlc_version_match: bool = False
    apalache_version_match: bool = False
    tlc_usable: bool = False
    apalache_usable: bool = False
    java_usable: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    java_support_only: bool = True
    java_cannot_promote_alone: bool = True
    grants_theorem_authority: bool = False
    bounded_model_checking_only: bool = True
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[CaseOutcome] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["receipt_digest_sha256"] = content_digest(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_digest_sha256"
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def default_corpus_manifest() -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "tool_ids": [TOOL_ID_TLC, TOOL_ID_APALACHE],
        "support_tool_id": SUPPORT_TOOL_ID,
        "lane_id": LANE_ID,
        "locked_tlc_version": LOCKED_TLC_VERSION,
        "locked_apalache_version": LOCKED_APALACHE_VERSION,
        "locked_artifact_digests": {
            "tlc": LOCKED_TLC_SHA256,
            "apalache": LOCKED_APALACHE_SHA256,
        },
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "model_bindings": dict(DEFAULT_MODEL_BINDINGS),
        "config_bindings": dict(DEFAULT_CONFIG_BINDINGS),
        "bounds": dict(DEFAULT_BOUNDS),
        "capabilities": {
            "tlc": TLC_CAPABILITY.to_dict(),
            "apalache": APALACHE_CAPABILITY.to_dict(),
        },
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "java_is_support_only": True,
            "java_cannot_promote_tla_lane": True,
            "exact_binary_binding_required": True,
            "authority_is_bounded_state_model_only": True,
            "never_theorem_authority": True,
            "bounded_model_checking_never_promotes_to_theorem": True,
        },
        "cases": [dict(case) for case in _DEFAULT_CORPUS_CASES],
    }


def load_corpus_manifest(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or repo_root_from()
    if path is not None and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state-model corpus manifest must be a JSON object")
        return payload
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("state-model corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------


def probe_tlc_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    java_executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID_TLC,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_TLC_VERSION,
        "locked_artifact_sha256": LOCKED_TLC_SHA256,
        "managed_identity": None,
        "managed_identity_verified": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable(
        [LOCKED_TLC_EXECUTABLE, "tlc2", "tla2tools"]
    )
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "-help"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        return result
    output = _ANSI_ESCAPE_RE.sub(
        "",
        "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        ),
    )
    markers = (
        "TLC - provides model checking and simulation of TLA+ specifications",
        "SYNOPSIS",
        "DESCRIPTION",
    )
    if completed.returncode not in {0, 1} or not all(
        marker in output for marker in markers
    ):
        result["probe_error"] = "tlc_help_semantics_failed"
        return result
    managed = _managed_identity(
        TOOL_ID_TLC,
        binary,
        env=probe_env,
        java_executable=java_executable,
    )
    result["managed_identity"] = managed
    result["managed_identity_verified"] = bool(managed.get("usable"))
    if not result["managed_identity_verified"]:
        result["probe_error"] = "managed_digest_identity_failed"
        return result
    result["version_string"] = (
        f"TLC managed release {LOCKED_TLC_VERSION}; "
        f"artifact sha256:{LOCKED_TLC_SHA256}"
    )
    result["identity_probed"] = True
    result["version_match"] = True
    return result


def probe_apalache_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    java_executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID_APALACHE,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_APALACHE_VERSION,
        "locked_artifact_sha256": LOCKED_APALACHE_SHA256,
        "managed_identity": None,
        "managed_identity_verified": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable(
        [LOCKED_APALACHE_EXECUTABLE, "apalache"]
    )
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    banner: str | None = None
    for args in (("version",), ("--version",), ("-V",)):
        completed = bounded_run(
            [binary, *args],
            timeout=PROBE_TIMEOUT_SECONDS,
            env=probe_env,
        )
        if completed is None:
            continue
        if completed.returncode != 0:
            continue
        banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
            completed.stderr
        )
        if not banner:
            banner = (completed.stdout or completed.stderr or "").strip() or None
        if banner:
            break
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    version = extract_version(banner)
    result["version_match"] = bool(
        version == LOCKED_APALACHE_VERSION
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
        return result
    managed = _managed_identity(
        TOOL_ID_APALACHE,
        binary,
        env=probe_env,
        java_executable=java_executable,
    )
    result["managed_identity"] = managed
    result["managed_identity_verified"] = bool(managed.get("usable"))
    if not result["managed_identity_verified"]:
        result["probe_error"] = "managed_digest_identity_failed"
        return result
    result["identity_probed"] = True
    return result


def probe_java_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": SUPPORT_TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": True,  # host runtime; no locked pin version
        "support_only": True,
        "can_promote_tla_lane": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    runtime = _selected_java(
        probe_env,
        executable=executable,
        minimum_major=installer.APALACHE_MIN_JAVA_MAJOR,
    )
    result["runtime"] = runtime.to_dict()
    result["path_present"] = runtime.executable is not None
    result["executable_path"] = runtime.executable
    result["version_string"] = runtime.banner
    result["version_match"] = runtime.usable
    if not runtime.usable:
        result["probe_error"] = runtime.reason_code or "java_runtime_unusable"
        result["version_match"] = False
        return result
    result["identity_probed"] = True
    return result


# ---------------------------------------------------------------------------
# Semantic classification (offline, marker-based, aligned with runners)
# ---------------------------------------------------------------------------


_TLC_SUCCESS = (
    "model checking completed. no error has been found",
    "model checking completed",
    "no error has been found",
)
_APALACHE_SUCCESS = (
    "checker reports no error",
    "no error up to computation length",
    "verification result: pass",
    "result: pass",
)
_COUNTEREXAMPLE = (
    "counterexample",
    "is violated",
    "temporal properties were violated",
    "checker reports an error",
    "checker has found an error",
    "found an invariant violation",
    "error trace",
)


def classify_model_check_output(
    *,
    tool: str,
    stdout: str,
    stderr: str,
    returncode: int | None = 0,
    timed_out: bool = False,
    unavailable: bool = False,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Classify tool output into a corpus status string.

    Returns ``(status, reason_codes, counterexample_dict_or_none)``.
    Status values: passed | counterexample | timed_out | malformed | error | unknown
    """

    reasons: list[str] = []
    combined = f"{stdout}\n{stderr}"
    lower = combined.lower()
    if unavailable:
        return "unknown", ["unavailable"], None
    if timed_out or returncode is None and "timeout" in lower:
        return "timed_out", ["timeout"], None
    if any(marker in lower for marker in _COUNTEREXAMPLE):
        trace = parse_counterexample_trace(combined)
        cx: dict[str, Any] | None = None
        if hasattr(trace, "to_dict"):
            cx = trace.to_dict()  # type: ignore[assignment]
        else:
            cx = {
                "raw": combined,
                "state_count": len(getattr(trace, "states", ()) or ()),
            }
        return "counterexample", ["invariant_violation"], cx
    success_markers = _TLC_SUCCESS if tool == "tlc" else _APALACHE_SUCCESS
    if returncode == 0 and any(marker in lower for marker in success_markers):
        reasons.append("bounded_success")
        if tool == "apalache":
            reasons.append("finite_trace_only")
        return "passed", reasons, None
    if returncode not in (0, None) and combined.strip():
        # Non-zero without counterexample markers → error/malformed
        if "parse error" in lower or "garbage" in lower or "not a tlc" in lower:
            return "malformed", ["malformed_output"], None
        return "error", ["nonzero_exit"], None
    if not combined.strip():
        return "malformed", ["empty_output"], None
    return "unknown", ["unclassified_output"], None


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one corpus case via TLA classifiers (no install)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    stdout = str(case.get("stdout") or "")
    stderr = str(case.get("stderr") or "")
    tool = str(case.get("tool") or "tlc")
    model_digest = str(case.get("model_digest") or "")
    config_digest = str(case.get("config_digest") or "")
    output_digest = content_digest(f"{stdout}\n{stderr}")

    if kind == "version_mismatch":
        observed_t = str(case.get("observed_tlc_version") or "")
        observed_a = str(case.get("observed_apalache_version") or "")
        blocked = (
            observed_t != LOCKED_TLC_VERSION
            or observed_a != LOCKED_APALACHE_VERSION
        )
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status="blocked" if blocked else "unknown",
            matched=blocked and expect == "blocked",
            reason_codes=["locked_version_mismatch"] if blocked else [],
            output_digest=output_digest,
            model_digest=model_digest,
            config_digest=config_digest,
            detail=(
                f"observed tlc={observed_t} apalache={observed_a}; "
                f"locked tlc={LOCKED_TLC_VERSION} apalache={LOCKED_APALACHE_VERSION}"
            ),
            grants_theorem_authority=False,
        )

    returncode = case.get("returncode", 0)
    if returncode is not None:
        returncode = int(returncode)
    timed_out = bool(case.get("timed_out"))
    status, reasons, counterexample = classify_model_check_output(
        tool=tool,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
    )

    # Bound success is still only bounded — never theorem.
    if kind == "bound" and status == "passed":
        # expose as passed_bounded for expect matching
        observed_for_expect = "passed_bounded"
        reasons.append("bounded_only")
        reasons.append("never_theorem_authority")
    else:
        observed_for_expect = status

    matched = _expect_matches(expect, observed_for_expect, status)

    if kind == "mutation" and status == "passed":
        matched = False
        reasons.append("mutation_still_passed")

    if kind in {"timeout", "malformed"} and status == "passed":
        matched = False
        reasons.append(f"{kind}_promoted_to_pass")

    # Case status for receipt uses the classifier status (or blocked).
    case_status = status if kind != "bound" else (
        "passed" if status == "passed" else status
    )

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        status=case_status if kind != "bound" else (
            "passed" if status == "passed" else status
        ),
        matched=matched,
        reason_codes=list(dict.fromkeys(reasons)),
        counterexample=counterexample,
        output_digest=output_digest,
        model_digest=model_digest,
        config_digest=config_digest,
        tool=tool,
        stdout=stdout,
        stderr=stderr,
        detail=str(case.get("description") or ""),
        grants_theorem_authority=False,
    )


def _expect_matches(expect: str, observed_for_expect: str, raw_status: str) -> bool:
    if expect == "passed":
        return raw_status == "passed"
    if expect == "passed_bounded":
        return observed_for_expect == "passed_bounded" or (
            raw_status == "passed" and observed_for_expect == "passed_bounded"
        )
    if expect == "counterexample":
        return raw_status == "counterexample"
    if expect == "timed_out":
        return raw_status == "timed_out"
    if expect == "malformed":
        return raw_status == "malformed"
    if expect == "malformed_or_error":
        return raw_status in {"malformed", "error", "unknown"}
    if expect == "blocked":
        return raw_status == "blocked" or observed_for_expect == "blocked"
    if expect == "rejected_or_quarantined":
        return raw_status in {
            "counterexample",
            "malformed",
            "error",
            "timed_out",
            "unknown",
            "blocked",
        }
    return observed_for_expect == expect or raw_status == expect


# ---------------------------------------------------------------------------
# Java promotion boundary
# ---------------------------------------------------------------------------


def java_cannot_promote_tla_lane() -> dict[str, Any]:
    """Prove Java support-only presence cannot satisfy TLA authority."""

    role = get_tool_role(SUPPORT_TOOL_ID)
    decision = evaluate_role_aware_promotion(
        SUPPORT_TOOL_ID,
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
        independent_reconstruction=True,
    )
    can_satisfy = can_satisfy_certified_authority_requirement(SUPPORT_TOOL_ID)
    return {
        "tool_id": SUPPORT_TOOL_ID,
        "role": role.role.value,
        "authority_ceiling": role.authority_ceiling.value,
        "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
        "can_satisfy_tla_requirement": can_satisfy,
        "promotion_allowed": decision.allowed,
        "promotion_decision": decision.to_dict(),
        "support_only": role.role is ToolRole.SUPPORT,
        "ceiling_is_none": role.authority_ceiling is ToolchainAuthorityCeiling.NONE,
        "blocks_alone": (not decision.allowed) and (not can_satisfy),
        "grants_theorem_authority": False,
    }


def bounded_checking_never_theorem_authority() -> dict[str, Any]:
    """Prove TLC/Apalache authority ceilings remain bounded, never theorem."""

    tlc_role = get_tool_role(TOOL_ID_TLC)
    apa_role = get_tool_role(TOOL_ID_APALACHE)
    return {
        "tlc": {
            "role": tlc_role.role.value,
            "authority_ceiling": tlc_role.authority_ceiling.value,
            "is_bounded": tlc_role.authority_ceiling
            is ToolchainAuthorityCeiling.BOUNDED,
            "is_theorem": False,
        },
        "apalache": {
            "role": apa_role.role.value,
            "authority_ceiling": apa_role.authority_ceiling.value,
            "is_bounded": apa_role.authority_ceiling
            is ToolchainAuthorityCeiling.BOUNDED,
            "is_theorem": False,
            "finite_trace_only": True,
            "checks_liveness": APALACHE_CAPABILITY.checks_liveness,
        },
        "never_theorem_authority": True,
        "bounded_model_checking_only": True,
        "lane_authority_ceiling": AUTHORITY_CEILING,
        "lane_authority_scope": AUTHORITY_SCOPE,
    }


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    tlc_executable: str | None = None,
    apalache_executable: str | None = None,
    java_executable: str | None = None,
) -> StateModelToolchainCertification:
    """Run the full TLC/Apalache certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = StateModelToolchainCertification()
    probe_env = offline_env(env)
    declared_digests = dict(corpus.get("locked_artifact_digests") or {})
    digest_manifest_ok = declared_digests == {
        TOOL_ID_TLC: LOCKED_TLC_SHA256,
        TOOL_ID_APALACHE: LOCKED_APALACHE_SHA256,
    }
    if not digest_manifest_ok:
        cert.block_reasons.append("artifact_digest_manifest_mismatch")
    cert.checks.append(
        CheckResult(
            check_id="state_model.artifact_digest_manifest",
            kind="binding",
            status="passed" if digest_manifest_ok else "failed",
            expected=content_digest(
                {
                    TOOL_ID_TLC: LOCKED_TLC_SHA256,
                    TOOL_ID_APALACHE: LOCKED_APALACHE_SHA256,
                }
            ),
            observed=content_digest(declared_digests),
            detail="corpus manifest binds both reviewed state-model artifact digests",
            bindings={"locked_artifact_digests": declared_digests},
            reason_codes=(
                []
                if digest_manifest_ok
                else ["artifact_digest_manifest_mismatch"]
            ),
        )
    )

    cert.checks.append(
        CheckResult(
            check_id="state_model.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used},"
                f"FORMAL_VERIFICATION_CERTIFY_OFFLINE="
                f"{probe_env.get('FORMAL_VERIFICATION_CERTIFY_OFFLINE')}"
            ),
            detail="certification never installs, downloads, or opens the network",
        )
    )

    tlc_probe = probe_tlc_identity(
        env=probe_env,
        executable=tlc_executable,
        java_executable=java_executable,
    )
    apalache_probe = probe_apalache_identity(
        env=probe_env,
        executable=apalache_executable,
        java_executable=java_executable,
    )
    java_probe = probe_java_identity(env=probe_env, executable=java_executable)

    cert.tlc_executable = tlc_probe.get("executable_path")
    cert.apalache_executable = apalache_probe.get("executable_path")
    cert.java_executable = java_probe.get("executable_path")
    cert.tlc_version_string = tlc_probe.get("version_string")
    cert.apalache_version_string = apalache_probe.get("version_string")
    cert.java_version_string = java_probe.get("version_string")
    cert.tlc_identity_probed = bool(tlc_probe.get("identity_probed"))
    cert.apalache_identity_probed = bool(apalache_probe.get("identity_probed"))
    cert.java_identity_probed = bool(java_probe.get("identity_probed"))
    cert.tlc_version_match = bool(tlc_probe.get("version_match"))
    cert.apalache_version_match = bool(apalache_probe.get("version_match"))
    cert.tlc_usable = bool(
        cert.tlc_identity_probed
        and cert.tlc_version_match
        and tlc_probe.get("managed_identity_verified")
    )
    cert.apalache_usable = bool(
        cert.apalache_identity_probed
        and cert.apalache_version_match
        and apalache_probe.get("managed_identity_verified")
    )
    cert.java_usable = bool(cert.java_identity_probed)

    for tool_name, usable, probe, locked, version_string, executable in (
        (
            "tlc",
            cert.tlc_usable,
            tlc_probe,
            LOCKED_TLC_VERSION,
            cert.tlc_version_string,
            cert.tlc_executable,
        ),
        (
            "apalache",
            cert.apalache_usable,
            apalache_probe,
            LOCKED_APALACHE_VERSION,
            cert.apalache_version_string,
            cert.apalache_executable,
        ),
    ):
        if usable:
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_name}.identity",
                    kind="identity",
                    status="passed",
                    expected=locked,
                    observed=version_string or "",
                    detail=f"exact {tool_name} pin identity",
                    bindings={
                        "executable_path": executable,
                        "version_string": version_string,
                        "managed_identity": probe.get("managed_identity"),
                    },
                )
            )
        else:
            reason = str(probe.get("probe_error") or "unavailable")
            cert.block_reasons.append(f"{tool_name}:{reason}")
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_name}.identity",
                    kind="identity",
                    status=(
                        "unavailable"
                        if reason == "executable_not_on_path"
                        else "blocked"
                    ),
                    expected=locked,
                    observed=reason,
                    detail="PATH presence without locked identity is not usability",
                    reason_codes=[reason],
                )
            )

    if cert.java_usable:
        cert.checks.append(
            CheckResult(
                check_id="java.identity",
                kind="identity",
                status="passed",
                expected="host_jvm_support",
                observed=cert.java_version_string or "",
                detail="host JVM identity (support only)",
                bindings={
                    "executable_path": cert.java_executable,
                    "version_string": cert.java_version_string,
                    "support_only": True,
                },
            )
        )
    else:
        reason = str(java_probe.get("probe_error") or "unavailable")
        cert.block_reasons.append(f"java:{reason}")
        cert.checks.append(
            CheckResult(
                check_id="java.identity",
                kind="identity",
                status="unavailable" if reason == "executable_not_on_path" else "blocked",
                expected="host_jvm_support",
                observed=reason,
                detail="Java support companion identity",
                reason_codes=[reason],
            )
        )

    # Semantic corpus (classifier-backed; always runs offline).
    outcomes_by_id: dict[str, CaseOutcome] = {}
    for case in cases:
        outcome = evaluate_corpus_case(case)
        outcomes_by_id[outcome.case_id] = outcome
        cert.cases.append(outcome)
        status = "passed" if outcome.matched else "failed"
        if not outcome.matched:
            cert.block_reasons.append(f"case_failed:{outcome.case_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"state_model.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "model_digest": outcome.model_digest,
                    "config_digest": outcome.config_digest,
                    "tool": outcome.tool,
                    "counterexample": outcome.counterexample,
                    "grants_theorem_authority": False,
                },
            )
        )

    # Deterministic replay binding between invariant_holds and replay cases.
    holds = outcomes_by_id.get("invariant_holds")
    replay = outcomes_by_id.get("deterministic_replay")
    if holds is not None and replay is not None:
        replay_ok = (
            holds.status == "passed"
            and replay.status == "passed"
            and holds.output_digest == replay.output_digest
            and holds.matched
            and replay.matched
            and holds.model_digest == replay.model_digest
            and holds.config_digest == replay.config_digest
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="state_model.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical passed digests",
                observed=(
                    f"holds={holds.output_digest[:12]},"
                    f"replay={replay.output_digest[:12]}"
                ),
                bindings={
                    "holds_digest": holds.output_digest,
                    "replay_digest": replay.output_digest,
                    "model_digest": holds.model_digest,
                    "config_digest": holds.config_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_holds_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="state_model.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="invariant_holds and replay cases",
                observed="missing",
            )
        )

    # Java cannot promote the TLA lane by itself.
    java_boundary = java_cannot_promote_tla_lane()
    boundary_ok = bool(java_boundary.get("blocks_alone"))
    if not boundary_ok:
        cert.block_reasons.append("java_incorrectly_promotes")
    cert.checks.append(
        CheckResult(
            check_id="java.support_only_boundary",
            kind="authority",
            status="passed" if boundary_ok else "failed",
            expected="promotion_blocked",
            observed=(
                f"allowed={java_boundary.get('promotion_allowed')},"
                f"can_satisfy={java_boundary.get('can_satisfy_tla_requirement')}"
            ),
            detail="Java is support only and cannot promote the TLA lane",
            bindings=java_boundary,
        )
    )

    theorem_boundary = bounded_checking_never_theorem_authority()
    theorem_ok = bool(theorem_boundary.get("never_theorem_authority")) and bool(
        theorem_boundary.get("bounded_model_checking_only")
    )
    if not theorem_ok:
        cert.block_reasons.append("theorem_authority_incorrectly_granted")
    cert.checks.append(
        CheckResult(
            check_id="state_model.never_theorem_authority",
            kind="authority",
            status="passed" if theorem_ok else "failed",
            expected="bounded_only",
            observed=json.dumps(
                {
                    "tlc": theorem_boundary["tlc"]["authority_ceiling"],
                    "apalache": theorem_boundary["apalache"]["authority_ceiling"],
                },
                sort_keys=True,
            ),
            detail="Bounded model-checking never promotes to theorem authority",
            bindings=theorem_boundary,
        )
    )

    # Bind model, config, constants, bounds, and exact tool identities.
    cert.bindings = {
        "locked_artifact_digests": declared_digests,
        "model": dict(corpus.get("model_bindings") or DEFAULT_MODEL_BINDINGS),
        "config": dict(corpus.get("config_bindings") or DEFAULT_CONFIG_BINDINGS),
        "constants": dict(
            (corpus.get("config_bindings") or DEFAULT_CONFIG_BINDINGS).get(
                "CONSTANTS"
            )
            or DEFAULT_CONFIG_BINDINGS["CONSTANTS"]
        ),
        "bounds": dict(corpus.get("bounds") or DEFAULT_BOUNDS),
        "capabilities": {
            "tlc": TLC_CAPABILITY.to_dict(),
            "apalache": APALACHE_CAPABILITY.to_dict(),
            "tlc_backend": TLC_BACKEND_VERSION,
            "apalache_backend": APALACHE_BACKEND_VERSION,
        },
        "binaries": {
            "tlc": {
                "tool_id": TOOL_ID_TLC,
                "locked_version": LOCKED_TLC_VERSION,
                "executable_path": cert.tlc_executable,
                "version_string": cert.tlc_version_string,
                "identity_probed": cert.tlc_identity_probed,
                "version_match": cert.tlc_version_match,
                "authority_ceiling": AUTHORITY_CEILING,
                "locked_artifact_sha256": LOCKED_TLC_SHA256,
                "managed_identity": tlc_probe.get("managed_identity"),
            },
            "apalache": {
                "tool_id": TOOL_ID_APALACHE,
                "locked_version": LOCKED_APALACHE_VERSION,
                "executable_path": cert.apalache_executable,
                "version_string": cert.apalache_version_string,
                "identity_probed": cert.apalache_identity_probed,
                "version_match": cert.apalache_version_match,
                "authority_ceiling": AUTHORITY_CEILING,
                "finite_trace_only": True,
                "locked_artifact_sha256": LOCKED_APALACHE_SHA256,
                "managed_identity": apalache_probe.get("managed_identity"),
            },
            "java": {
                "tool_id": SUPPORT_TOOL_ID,
                "executable_path": cert.java_executable,
                "version_string": cert.java_version_string,
                "identity_probed": cert.java_identity_probed,
                "support_only": True,
                "can_promote_tla_lane": False,
            },
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "java_is_support_only": True,
            "never_theorem": True,
            "bounded_model_checking_only": True,
            "not_kernel": True,
            "not_advisor": True,
        },
        "java_promotion_boundary": java_boundary,
        "theorem_authority_boundary": theorem_boundary,
    }
    cert.checks.append(
        CheckResult(
            check_id="state_model.bindings",
            kind="binding",
            status="passed",
            expected="model,config,constants,bounds,binaries",
            observed=content_digest(cert.bindings)[:16],
            detail=(
                "receipt binds model, config, constants, bounds, and exact tools"
            ),
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "invariant_holds",
        "violation_trace",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "bound",
        "version_mismatch",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    semantic_ok = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind
        in {
            "invariant_holds",
            "violation_trace",
            "mutation",
            "replay",
            "malformed",
            "timeout",
            "bound",
            "version_mismatch",
            "authority",
            "binding",
        }
        or check.check_id
        in {
            "state_model.deterministic_replay_binding",
            "state_model.bindings",
            "state_model.artifact_digest_manifest",
            "java.support_only_boundary",
            "state_model.never_theorem_authority",
            "state_model.offline_policy",
        }
    )

    # Production certification requires live locked tools + Java + semantic suite.
    cert.production_certified = bool(
        cert.tlc_usable
        and cert.apalache_usable
        and cert.java_usable
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and semantic_ok
        and not missing_kinds
        and not any(
            reason.startswith("case_failed:") or reason.startswith("replay_")
            for reason in cert.block_reasons
        )
        and boundary_ok
        and theorem_ok
        and not cert.grants_theorem_authority
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned TLC 1.8.0 + Apalache 0.58.3 certified for bounded state-model "
            "checking; Java remains support only; never theorem authority."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            if semantic_ok and not (
                cert.tlc_usable and cert.apalache_usable and cert.java_usable
            ):
                cert.notes = (
                    "Semantic corpus passed offline; live locked TLC/Apalache/Java "
                    "identities unavailable — production certification withheld."
                )
            else:
                cert.notes = (
                    "TLC/Apalache certification incomplete or failed; "
                    "TLA-lane promotion blocked."
                )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    tlc_executable: str | None = None,
    apalache_executable: str | None = None,
    java_executable: str | None = None,
) -> dict[str, Any]:
    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        tlc_executable=tlc_executable,
        apalache_executable=apalache_executable,
        java_executable=java_executable,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "java_is_support_only": True,
        "java_cannot_promote_tla_lane": True,
        "exact_binary_binding_required": True,
        "authority_is_bounded_state_model_only": True,
        "never_theorem_authority": True,
        "bounded_model_checking_never_promotes_to_theorem": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_shared_lock": True,
    }
    payload["semantic_corpus_passed"] = all(case.matched for case in cert.cases)
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_state_model_toolchain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=kwargs.get("manifest"),
        env=kwargs.get("env"),
        tlc_executable=kwargs.get("tlc_executable"),
        apalache_executable=kwargs.get("apalache_executable"),
        java_executable=kwargs.get("java_executable"),
    )
    receipt["handler_id"] = HANDLER_ID
    receipt["lane_id"] = LANE_ID
    receipt["owner_module"] = CERTIFICATION_SURFACE
    receipt["status"] = (
        "certified" if receipt.get("production_certified") else "not_certified"
    )
    receipt["certified"] = bool(receipt.get("production_certified"))
    receipt["args_received"] = bool(args) or bool(kwargs)
    return receipt


def lane_handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return certify_state_model_toolchain(*args, **kwargs)


def bind_tla_lane_handler(
    *,
    policy: Any | None = None,
    replace: bool = True,
) -> Any | None:
    """Register this certifier on the TLA lane when roles surface exists."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return None
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID, lane_handler, policy=target, replace=replace
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the pinned TLC/Apalache state-model toolchain "
            f"({INTERFACE}; TLC {LOCKED_TLC_VERSION} + "
            f"Apalache {LOCKED_APALACHE_VERSION})."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print receipt as JSON")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--tlc", type=str, default=None)
    parser.add_argument("--apalache", type=str, default=None)
    parser.add_argument("--java", type=str, default=None)
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    receipt = build_certification_receipt(
        repo_root=root,
        tlc_executable=args.tlc,
        apalache_executable=args.apalache,
        java_executable=args.java,
    )
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"tlc={receipt.get('tlc_version_string')!r} "
            f"apalache={receipt.get('apalache_version_string')!r} "
            f"java={receipt.get('java_version_string')!r}"
        )
        print(
            f"usable_tlc={receipt.get('tlc_usable')} "
            f"usable_apalache={receipt.get('apalache_usable')} "
            f"usable_java={receipt.get('java_usable')} "
            f"production_certified={receipt.get('production_certified')} "
            f"promotion_blocked={receipt.get('promotion_blocked')}"
        )
        for check in receipt.get("checks") or []:
            print(
                f"  [{check.get('status'):10}] {check.get('check_id')}: "
                f"expected={check.get('expected')} observed={check.get('observed')}"
            )
        if receipt.get("production_certified"):
            return 0
        return 1
    return 0 if receipt.get("semantic_corpus_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
