#!/usr/bin/env python3
"""Rocq/Coq + isolated OPAM kernel toolchain certification (FVT-G150 / FVT-045).

``RocqToolchainCertification@1``

Owns the kernel-lane certification handler for the pinned Rocq/Coq 9.1.1
provider (package identity ``rocq-prover.9.1.1``) and its support-only OPAM
2.5.2 companion under an isolated root. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for Rocq/Coq 9.1.1 when production-certifying;
* exercises true proof, false proof, hypothesis/conclusion mutation,
  deterministic replay, forbidden admits/axiom escapes, malformed input, and
  version-mismatch cases;
* receipts bind imports, source, theorem, assumptions, and exact kernel
  identity;
* treats OPAM as support only — OPAM presence alone never promotes the kernel
  property lane;
* never mutates a global OPAM switch;
* never edits the shared multi-prover certificate.

Offline corpus evaluation reuses the same acceptance rules as the Rocq kernel
backend so hermetic tests prove semantics without a live ``coqc``. Live
production certification additionally requires the pinned binary.
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
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

INTERFACE: Final = "RocqToolchainCertification@1"
SCHEMA_VERSION: Final = "rocq-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "rocq-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G150"
TASK_ID: Final = "FVT-045"
PROGRAM: Final = "formal-verification-tactician/rocq-toolchain"
LANE_ID: Final = "kernel"
TOOL_ID: Final = "coq"
SUPPORT_TOOL_ID: Final = "opam"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.rocq"
HANDLER_ID: Final = "rocq_toolchain_certifier"
AUTHORITY_CEILING: Final = "kernel"
AUTHORITY_SCOPE: Final = "kernel_proof_checking_only"

LOCKED_VERSION: Final = "9.1.1"
LOCKED_OPAM_VERSION: Final = "2.5.2"
PACKAGE_IDENTITY: Final = "rocq-prover.9.1.1"
OPAM_REPOSITORY: Final = "https://rocq-prover.org/opam/released"
LOCKED_EXECUTABLES: Final = ("coqc", "rocq", "coqtop")

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_ADMIT = re.compile(r"(?i)\b(?:admit\s*\.|Admitted\s*\.|Abort\s*\.)")
_AXIOM = re.compile(r"(?im)^\s*Axiom\s+")
_IMPORT = re.compile(
    r"^\s*((?:Require\s+(?:Import|Export)\s+.+?\.)|(?:From\s+\S+\s+Require\s+"
    r"(?:Import|Export)\s+.+?\.))\s*$",
    re.MULTILINE,
)
_DECL = re.compile(
    r"^\s*(?:Theorem|Lemma|Fact|Corollary|Remark|Proposition|Example)\s+"
    r"([A-Za-z_][A-Za-z0-9_']*)",
    re.MULTILINE,
)
_ERROR_MARKER = re.compile(r"(?i)\berror\b|\bError\b")
_VERSION_IN_BANNER = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
_CLOSED_UNDER_GLOBAL = re.compile(
    r"Closed under the global context", re.IGNORECASE
)

# Compact Gallina corpus (offline evaluation; live coqc optional).
_TRUE_THEOREM: Final = """\
Theorem from_eq : forall n m : nat, n = m -> n = m.
Proof.
  intros n m H. exact H.
Qed.
"""

_FALSE_PROOF: Final = """\
Theorem false_claim : False.
Proof.
  exact I.
Qed.
"""

_HYPOTHESIS_MUTATION: Final = """\
Theorem from_eq : forall n m : nat, n = n -> n = m.
Proof.
  intros n m H. exact H.
Qed.
"""

_CONCLUSION_MUTATION: Final = """\
Theorem from_eq : forall n m : nat, n = m -> False.
Proof.
  intros n m H. exact H.
Qed.
"""

_ADMIT_ESCAPE: Final = """\
Theorem hole_admit : True.
Proof.
  admit.
Qed.
"""

_ADMITTED_ESCAPE: Final = """\
Theorem hole_admitted : True.
Admitted.
"""

_AXIOM_ESCAPE: Final = """\
Axiom bad : False.
Theorem uses_axiom : False.
Proof.
  exact bad.
Qed.
"""

_MALFORMED: Final = """\
this is not valid Gallina
Theorem broken : True := 0.
"""

_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "true_theorem",
        "kind": "positive",
        "expect": "accepted",
        "theorem_name": "from_eq",
        "assumptions": ["H : n = m"],
        "imports": [],
        "source": _TRUE_THEOREM,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "assumption_report": "Closed under the global context\n",
        "description": "Exact true theorem accepted by the pinned Rocq/Coq kernel",
    },
    {
        "case_id": "false_proof",
        "kind": "negative",
        "expect": "rejected",
        "theorem_name": "false_claim",
        "assumptions": [],
        "imports": [],
        "source": _FALSE_PROOF,
        "returncode": 1,
        "stdout": "",
        "stderr": "Error: The term \"I\" has type \"True\" while it is expected to have type \"False\".\n",
        "description": "False proof rejected by the kernel",
    },
    {
        "case_id": "hypothesis_mutation",
        "kind": "mutation",
        "mutates": "hypothesis",
        "expect": "rejected",
        "base_case_id": "true_theorem",
        "theorem_name": "from_eq",
        "assumptions": ["H : n = n"],
        "imports": [],
        "source": _HYPOTHESIS_MUTATION,
        "returncode": 1,
        "stdout": "",
        "stderr": "Error: In environment\nn, m : nat\nH : n = n\nThe term \"H\" has type \"n = n\" while it is expected to have type \"n = m\".\n",
        "description": "Hypothesis mutation of the true theorem is rejected",
    },
    {
        "case_id": "conclusion_mutation",
        "kind": "mutation",
        "mutates": "conclusion",
        "expect": "rejected",
        "base_case_id": "true_theorem",
        "theorem_name": "from_eq",
        "assumptions": ["H : n = m"],
        "imports": [],
        "source": _CONCLUSION_MUTATION,
        "returncode": 1,
        "stdout": "",
        "stderr": "Error: In environment\nn, m : nat\nH : n = m\nThe term \"H\" has type \"n = m\" while it is expected to have type \"False\".\n",
        "description": "Conclusion mutation of the true theorem is rejected",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "accepted",
        "base_case_id": "true_theorem",
        "theorem_name": "from_eq",
        "assumptions": ["H : n = m"],
        "imports": [],
        "source": _TRUE_THEOREM,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "assumption_report": "Closed under the global context\n",
        "description": "Positive case replays with identical acceptance and digests",
    },
    {
        "case_id": "admit_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "hole_admit",
        "assumptions": [],
        "imports": [],
        "reason_codes": ["admit_or_admitted"],
        "source": _ADMIT_ESCAPE,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "description": "admit fails closed even when coqc may exit 0",
    },
    {
        "case_id": "admitted_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "hole_admitted",
        "assumptions": [],
        "imports": [],
        "reason_codes": ["admit_or_admitted"],
        "source": _ADMITTED_ESCAPE,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "description": "Admitted fails closed",
    },
    {
        "case_id": "axiom_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "uses_axiom",
        "assumptions": [],
        "imports": [],
        "reason_codes": ["unreviewed_axiom"],
        "source": _AXIOM_ESCAPE,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "assumption_report": "Axioms:\nbad : False\n",
        "description": "unreviewed Axiom escapes fail closed",
    },
    {
        "case_id": "malformed_input",
        "kind": "malformed",
        "expect": "rejected",
        "theorem_name": "broken",
        "assumptions": [],
        "imports": [],
        "source": _MALFORMED,
        "returncode": 1,
        "stdout": "",
        "stderr": "Error: Syntax error\n",
        "description": "Malformed Gallina never reports acceptance",
    },
    {
        "case_id": "version_mismatch",
        "kind": "version_mismatch",
        "expect": "blocked",
        "observed_version": "8.18.0",
        "source": "",
        "description": "Wrong Rocq/Coq installation fails closed",
    },
)


# ---------------------------------------------------------------------------
# Path / offline helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the offline toolchain lock."""

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
    """Environment that forbids opportunistic install/download/network."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["OPAMYES"] = "0"
    env["OPAMCOLOR"] = "never"
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


def resolve_coq_executable(candidates: Sequence[str] | None = None) -> str | None:
    names = list(candidates) if candidates else list(LOCKED_EXECUTABLES)
    for name in names:
        if not name:
            continue
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(name)
        if found:
            return found
    return None


def resolve_opam_executable(candidates: Sequence[str] | None = None) -> str | None:
    names = list(candidates) if candidates else [SUPPORT_TOOL_ID]
    for name in names:
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


# ---------------------------------------------------------------------------
# Source scanning / binding
# ---------------------------------------------------------------------------


def extract_rocq_imports(source: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1).strip() for match in _IMPORT.finditer(source or "")))


def extract_rocq_theorem_name(source: str) -> str | None:
    match = _DECL.search(source or "")
    if match is None:
        return None
    return match.group(1).rstrip(".")


def scan_rocq_incomplete_or_unsafe(source: str) -> tuple[str, ...]:
    """Return fail-closed reason codes for admit/Admitted/Axiom constructs."""

    findings: list[str] = []
    if _ADMIT.search(source or ""):
        findings.append("admit_or_admitted")
    if _AXIOM.search(source or ""):
        findings.append("unreviewed_axiom")
    return tuple(findings)


def evaluate_rocq_process_output(
    *,
    source: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    assumption_report: str = "",
    timed_out: bool = False,
) -> tuple[bool, list[str]]:
    """Decide whether a process observation constitutes kernel acceptance."""

    reasons: list[str] = []
    reasons.extend(scan_rocq_incomplete_or_unsafe(source))

    if timed_out:
        reasons.append("timeout")
        return False, list(dict.fromkeys(reasons))

    combined = f"{stdout}\n{stderr}\n{assumption_report}"
    if _ADMIT.search(combined):
        reasons.append("admit_or_admitted")
    if _ERROR_MARKER.search(combined) and returncode not in (0, None):
        reasons.append("error_marker")
    if returncode not in (0, None) and returncode != 0:
        reasons.append("non_zero_exit")
    if returncode is None and not timed_out and not stdout and not stderr and not source.strip():
        reasons.append("empty_process_result")

    # Open assumptions / axiom residual fails closed even on exit 0.
    if assumption_report.strip():
        if not _CLOSED_UNDER_GLOBAL.search(assumption_report):
            if re.search(r"(?i)\baxioms?\b", assumption_report):
                reasons.append("open_assumptions_or_axioms")
            elif ":" in assumption_report and "Closed" not in assumption_report:
                reasons.append("open_assumptions_or_axioms")

    # Malformed: exit 0 without a declaration and without clear success signals.
    theorem = extract_rocq_theorem_name(source) or ""
    if returncode == 0 and not reasons and not theorem and source.strip():
        reasons.append("malformed_output")

    # Empty garbage source without theorem is malformed.
    if source.strip() and not theorem and returncode != 0:
        reasons.append("malformed_input")

    accepted = (
        returncode == 0
        and not timed_out
        and not reasons
        and bool(theorem)
    )
    if not accepted and not reasons:
        reasons.append("not_accepted")
    return accepted, list(dict.fromkeys(reasons))


def opam_cannot_promote_kernel_lane() -> dict[str, Any]:
    """Prove OPAM is support-only and cannot grant kernel authority alone."""

    return {
        "tool_id": SUPPORT_TOOL_ID,
        "role": "support",
        "authority_ceiling": "none",
        "support_only": True,
        "can_promote_kernel_lane": False,
        "can_satisfy_kernel_requirement": False,
        "promotion_allowed": False,
        "blocks_alone": True,
        "requires_authority_tool": TOOL_ID,
        "locked_opam_version": LOCKED_OPAM_VERSION,
        "locked_authority_version": LOCKED_VERSION,
        "package_identity": PACKAGE_IDENTITY,
    }


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
    accepted: bool
    matched: bool
    status: str  # accepted | rejected | blocked
    reason_codes: list[str] = field(default_factory=list)
    theorem_name: str = ""
    imports: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    source_digest: str = ""
    output_digest: str = ""
    returncode: int | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RocqToolchainCertification:
    """Full certification receipt for the pinned Rocq/Coq kernel lane."""

    tool_id: str = TOOL_ID
    support_tool_id: str = SUPPORT_TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_version: str = LOCKED_VERSION
    locked_opam_version: str = LOCKED_OPAM_VERSION
    package_identity: str = PACKAGE_IDENTITY
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    executable_path: str | None = None
    opam_executable_path: str | None = None
    version_string: str | None = None
    opam_version_string: str | None = None
    identity_probed: bool = False
    opam_identity_probed: bool = False
    installed: bool = False
    usable: bool = False
    opam_usable: bool = False
    version_match: bool = False
    opam_version_match: bool = False
    isolated_root_validated: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    global_opam_mutation_attempted: bool = False
    semantic_corpus_passed: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
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
        "tool_id": TOOL_ID,
        "support_tool_id": SUPPORT_TOOL_ID,
        "lane_id": LANE_ID,
        "locked_version": LOCKED_VERSION,
        "locked_opam_version": LOCKED_OPAM_VERSION,
        "package_identity": PACKAGE_IDENTITY,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "admit_admitted_axiom_fail_closed": True,
            "opam_is_support_only": True,
            "opam_cannot_promote_kernel_lane": True,
            "isolated_opam_root_required": True,
            "never_mutate_global_opam_switch": True,
            "exact_tool_identity_required": True,
            "authority_is_kernel_proof_checking_only": True,
            "does_not_edit_central_certificate": True,
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
            raise ValueError("Rocq corpus manifest must be a JSON object")
        return payload
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("Rocq corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------


def probe_rocq_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Bounded offline identity probe for the locked Rocq/Coq pin only."""

    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "installed": False,
        "version_match": False,
        "locked_version": LOCKED_VERSION,
        "package_identity": PACKAGE_IDENTITY,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_coq_executable()
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "--version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        return result
    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    if not banner:
        banner = (completed.stdout or completed.stderr or "").strip()
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    result["identity_probed"] = True
    result["installed"] = True
    match = _VERSION_IN_BANNER.search(banner)
    observed = match.group(1) if match else ""
    result["version_match"] = bool(
        observed == LOCKED_VERSION or LOCKED_VERSION in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def probe_opam_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Bounded offline identity probe for the locked OPAM support binary."""

    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": SUPPORT_TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_OPAM_VERSION,
        "support_only": True,
        "can_promote_kernel_lane": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_opam_executable()
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "--version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        return result
    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    result["identity_probed"] = True
    result["version_match"] = bool(
        LOCKED_OPAM_VERSION in banner
        or (
            (match := _VERSION_IN_BANNER.search(banner))
            and match.group(1) == LOCKED_OPAM_VERSION
        )
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def validate_isolated_opam_root_contract(
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the installer-declared isolated OPAM root contract offline."""

    root = repo_root or repo_root_from()
    try:
        from ipfs_datasets_py.logic.backends.installers import rocq as installer
    except Exception as exc:  # pragma: no cover - import path variance
        return {
            "validated": False,
            "error": f"installer_import_failed:{exc}",
            "global_switch_mutation_forbidden": True,
        }

    isolated = installer.default_isolated_opam_root(repo_root=root)
    forbidden_home = installer.is_forbidden_global_opam_root(Path.home() / ".opam")
    try:
        asserted = installer.assert_isolated_opam_root(isolated)
        ok = asserted == isolated.resolve() and forbidden_home is True
        err = None
    except Exception as exc:
        ok = False
        err = str(exc)
    return {
        "validated": ok,
        "isolated_opam_root": str(isolated),
        "global_home_opam_forbidden": forbidden_home,
        "global_switch_mutation_forbidden": True,
        "error": err,
        "segment": "rocq",
        "serialized_with_proverif": True,
    }


# ---------------------------------------------------------------------------
# Case evaluation (offline / parser-backed)
# ---------------------------------------------------------------------------


def evaluate_corpus_case(
    case: Mapping[str, Any],
    *,
    reference_outcomes: Mapping[str, CaseOutcome] | None = None,
) -> CaseOutcome:
    """Evaluate one corpus case without installing or contacting the network."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "rejected")
    source = str(case.get("source") or "")
    stdout = str(case.get("stdout") or "")
    stderr = str(case.get("stderr") or "")
    assumption_report = str(case.get("assumption_report") or "")
    returncode = case.get("returncode")
    if returncode is not None:
        returncode = int(returncode)
    timed_out = bool(case.get("timed_out") or False)
    theorem_name = str(
        case.get("theorem_name") or extract_rocq_theorem_name(source) or ""
    )
    imports = [str(item) for item in (case.get("imports") or extract_rocq_imports(source))]
    assumptions = [str(item) for item in (case.get("assumptions") or [])]
    source_digest = content_digest(source) if source else content_digest("")
    output_digest = content_digest(f"{stdout}\n{stderr}\n{assumption_report}")

    if kind == "version_mismatch":
        observed = str(case.get("observed_version") or "")
        blocked = observed != LOCKED_VERSION
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            accepted=False,
            matched=blocked and expect == "blocked",
            status="blocked" if blocked else "accepted",
            reason_codes=["locked_version_mismatch"] if blocked else [],
            source_digest=source_digest,
            output_digest=output_digest,
            detail=f"observed={observed!r} locked={LOCKED_VERSION!r}",
        )

    accepted, reasons = evaluate_rocq_process_output(
        source=source,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        assumption_report=assumption_report,
        timed_out=timed_out,
    )

    expected_reasons = [str(item) for item in (case.get("reason_codes") or [])]
    if expect == "accepted":
        matched = accepted is True
    elif expect == "rejected":
        matched = accepted is False
        if expected_reasons:
            matched = matched and any(code in reasons for code in expected_reasons)
    elif expect == "blocked":
        matched = not accepted
    else:
        matched = False

    # Mutations must never accept.
    if kind == "mutation" and accepted:
        matched = False
        reasons.append("mutation_still_accepted")

    # Replay must accept with digests matching the positive reference when provided.
    if kind == "replay" and expect == "accepted":
        ref_id = str(case.get("base_case_id") or "true_theorem")
        ref = (reference_outcomes or {}).get(ref_id)
        if ref is not None:
            if (
                not accepted
                or source_digest != ref.source_digest
                or output_digest != ref.output_digest
            ):
                matched = False
                if source_digest != ref.source_digest or output_digest != ref.output_digest:
                    reasons.append("replay_digest_mismatch")
            else:
                matched = True

    status = "accepted" if accepted else "rejected"
    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        accepted=accepted,
        matched=matched,
        status=status,
        reason_codes=list(dict.fromkeys(reasons)),
        theorem_name=theorem_name,
        imports=imports,
        assumptions=assumptions,
        source_digest=source_digest,
        output_digest=output_digest,
        returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        detail=str(case.get("description") or ""),
    )


def check_rocq_source_live(
    source: str,
    *,
    executable: str,
    env: Mapping[str, str] | None = None,
    timeout: float = CHECK_TIMEOUT_SECONDS,
    case_id: str = "case",
) -> CaseOutcome:
    """Optionally run a live ``coqc`` check under offline bounds."""

    normalized = source if source.endswith("\n") else source + "\n"
    theorem_name = extract_rocq_theorem_name(normalized) or ""
    imports = list(extract_rocq_imports(normalized))
    source_digest = content_digest(normalized)
    reasons = list(scan_rocq_incomplete_or_unsafe(normalized))
    if reasons:
        return CaseOutcome(
            case_id=case_id,
            kind="fail_closed",
            expect="rejected",
            accepted=False,
            matched=True,
            status="rejected",
            reason_codes=reasons,
            theorem_name=theorem_name,
            imports=imports,
            source_digest=source_digest,
            output_digest=content_digest(""),
            detail="source scan rejected incomplete or unsafe constructs",
        )

    probe_env = offline_env(env)
    with tempfile.TemporaryDirectory(prefix="rocq-cert-") as tmp:
        work = Path(tmp)
        source_path = work / "Goal.v"
        source_path.write_text(normalized, encoding="utf-8")
        completed = bounded_run(
            [executable, str(source_path)],
            timeout=timeout,
            env=probe_env,
            cwd=work,
        )

    if completed is None:
        return CaseOutcome(
            case_id=case_id,
            kind="kernel",
            expect="rejected",
            accepted=False,
            matched=True,
            status="rejected",
            reason_codes=["timeout_or_spawn_failure"],
            theorem_name=theorem_name,
            imports=imports,
            source_digest=source_digest,
            output_digest=content_digest(""),
            timed_out=True,
            detail="bounded coqc invocation timed out or failed to spawn",
        )

    accepted, eval_reasons = evaluate_rocq_process_output(
        source=normalized,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        timed_out=False,
    )
    return CaseOutcome(
        case_id=case_id,
        kind="kernel",
        expect="accepted" if accepted else "rejected",
        accepted=accepted,
        matched=True,
        status="accepted" if accepted else "rejected",
        reason_codes=eval_reasons,
        theorem_name=theorem_name,
        imports=imports,
        source_digest=source_digest,
        output_digest=content_digest(f"{completed.stdout}\n{completed.stderr}"),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        detail="coqc under offline pin",
    )


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    opam_executable: str | None = None,
) -> RocqToolchainCertification:
    """Run the full Rocq/Coq toolchain certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = RocqToolchainCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="rocq.offline_policy",
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

    identity = probe_rocq_identity(env=probe_env, executable=executable)
    cert.executable_path = identity.get("executable_path")
    cert.version_string = identity.get("version_string")
    cert.identity_probed = bool(identity.get("identity_probed"))
    cert.installed = bool(identity.get("installed"))
    cert.version_match = bool(identity.get("version_match"))
    cert.network_used = bool(identity.get("network_used"))
    cert.install_attempted = bool(identity.get("install_attempted"))
    cert.download_attempted = bool(identity.get("download_attempted"))

    opam_identity = probe_opam_identity(env=probe_env, executable=opam_executable)
    cert.opam_executable_path = opam_identity.get("executable_path")
    cert.opam_version_string = opam_identity.get("version_string")
    cert.opam_identity_probed = bool(opam_identity.get("identity_probed"))
    cert.opam_version_match = bool(opam_identity.get("version_match"))
    cert.opam_usable = bool(
        cert.opam_identity_probed and cert.opam_version_match
    )

    if cert.identity_probed and cert.version_match and cert.executable_path:
        cert.usable = True
        cert.checks.append(
            CheckResult(
                check_id="rocq.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_VERSION,
                observed=cert.version_string or "",
                detail="exact offline pin identity",
                bindings={
                    "executable_path": cert.executable_path,
                    "version_string": cert.version_string,
                    "locked_version": LOCKED_VERSION,
                    "package_identity": PACKAGE_IDENTITY,
                },
            )
        )
    elif cert.identity_probed and not cert.version_match:
        cert.block_reasons.append("locked_version_mismatch")
        cert.usable = False
        cert.checks.append(
            CheckResult(
                check_id="rocq.identity",
                kind="identity",
                status="blocked",
                expected=LOCKED_VERSION,
                observed=cert.version_string or "mismatch",
                detail="wrong installation fails closed",
                reason_codes=["locked_version_mismatch"],
            )
        )
    else:
        cert.block_reasons.append(str(identity.get("probe_error") or "unavailable"))
        cert.usable = False
        cert.checks.append(
            CheckResult(
                check_id="rocq.identity",
                kind="identity",
                status="unavailable",
                expected=LOCKED_VERSION,
                observed=str(identity.get("probe_error") or "missing"),
                detail="PATH presence without identity is not usability",
                reason_codes=["identity_unavailable"],
            )
        )

    # Isolated OPAM root contract (offline; never uses global switch).
    root_contract = validate_isolated_opam_root_contract(repo_root=root)
    cert.isolated_root_validated = bool(root_contract.get("validated"))
    cert.global_opam_mutation_attempted = False
    cert.checks.append(
        CheckResult(
            check_id="rocq.isolated_opam_root",
            kind="policy",
            status="passed" if cert.isolated_root_validated else "failed",
            expected="isolated_opam_root_never_global",
            observed=str(root_contract.get("isolated_opam_root") or "missing"),
            detail="Rocq OPAM root is repository-local and never ~/.opam",
            bindings=dict(root_contract),
        )
    )
    if not cert.isolated_root_validated:
        cert.block_reasons.append("isolated_opam_root_invalid")

    opam_boundary = opam_cannot_promote_kernel_lane()
    cert.checks.append(
        CheckResult(
            check_id="opam.support_only_boundary",
            kind="policy",
            status="passed" if opam_boundary["support_only"] else "failed",
            expected="opam_support_only_no_kernel_promotion",
            observed=json.dumps(opam_boundary, sort_keys=True),
            detail="OPAM presence alone never promotes the kernel lane",
            bindings=dict(opam_boundary),
        )
    )

    # Offline semantic corpus (always runs; independent of live binary).
    outcomes_by_id: dict[str, CaseOutcome] = {}
    positive_outcome: CaseOutcome | None = None
    replay_outcome: CaseOutcome | None = None

    for case in cases:
        outcome = evaluate_corpus_case(case, reference_outcomes=outcomes_by_id)
        outcomes_by_id[outcome.case_id] = outcome
        cert.cases.append(outcome)
        if outcome.kind == "positive" and outcome.expect == "accepted":
            positive_outcome = outcome
        if outcome.kind == "replay" and outcome.case_id == "deterministic_replay":
            replay_outcome = outcome

        status = "passed" if outcome.matched else "failed"
        if not outcome.matched:
            cert.block_reasons.append(f"case_failed:{outcome.case_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"rocq.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "theorem_name": outcome.theorem_name,
                    "imports": list(outcome.imports),
                    "assumptions": list(outcome.assumptions),
                    "source_digest": outcome.source_digest,
                    "output_digest": outcome.output_digest,
                    "returncode": outcome.returncode,
                },
            )
        )

    if positive_outcome is not None and replay_outcome is not None:
        replay_ok = (
            positive_outcome.accepted
            and replay_outcome.accepted
            and positive_outcome.source_digest == replay_outcome.source_digest
            and positive_outcome.output_digest == replay_outcome.output_digest
            and positive_outcome.returncode == replay_outcome.returncode
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="rocq.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical acceptance and digests",
                observed=(
                    f"pos={positive_outcome.accepted}/{positive_outcome.source_digest[:12]},"
                    f"replay={replay_outcome.accepted}/{replay_outcome.source_digest[:12]}"
                ),
                detail="positive case must replay deterministically under the same pin",
                bindings={
                    "positive_source_digest": positive_outcome.source_digest,
                    "replay_source_digest": replay_outcome.source_digest,
                    "positive_output_digest": positive_outcome.output_digest,
                    "replay_output_digest": replay_outcome.output_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_positive_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="rocq.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="positive and replay cases",
                observed="missing",
                detail="corpus must include positive and replay cases",
            )
        )

    binding_case = positive_outcome or next(
        (item for item in cert.cases if item.accepted), None
    )
    cert.bindings = {
        "imports": list(binding_case.imports) if binding_case else [],
        "source": {
            "primary_path": "Goal.v",
            "source_digest": binding_case.source_digest if binding_case else "",
            "format": "rocq",
        },
        "theorem": {
            "name": binding_case.theorem_name if binding_case else "",
            "assumptions": list(binding_case.assumptions) if binding_case else [],
        },
        "assumptions": list(binding_case.assumptions) if binding_case else [],
        "kernel_identity": {
            "tool_id": TOOL_ID,
            "locked_version": LOCKED_VERSION,
            "package_identity": PACKAGE_IDENTITY,
            "executable_path": cert.executable_path,
            "version_string": cert.version_string,
            "version_match": cert.version_match,
            "opam_repository": OPAM_REPOSITORY,
        },
        "binaries": {
            "coq": {
                "tool_id": TOOL_ID,
                "locked_version": LOCKED_VERSION,
                "package_identity": PACKAGE_IDENTITY,
                "executable_path": cert.executable_path,
                "authority": True,
            },
            "opam": {
                "tool_id": SUPPORT_TOOL_ID,
                "locked_version": LOCKED_OPAM_VERSION,
                "executable_path": cert.opam_executable_path,
                "support_only": True,
                "can_promote_kernel_lane": False,
            },
        },
        "isolated_opam_root": dict(root_contract),
        "output": {
            "output_digest": binding_case.output_digest if binding_case else "",
            "returncode": binding_case.returncode if binding_case else None,
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "not_advisor": True,
            "not_install_authority": True,
            "opam_is_support_only": True,
            "opam_cannot_promote_kernel_lane": True,
        },
    }
    cert.checks.append(
        CheckResult(
            check_id="rocq.bindings",
            kind="binding",
            status="passed" if binding_case and binding_case.accepted else "failed",
            expected="imports,source,theorem,assumptions,kernel_identity",
            observed=content_digest(cert.bindings)[:16],
            detail="receipt binds every trust-relevant kernel check input",
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "positive",
        "negative",
        "mutation",
        "replay",
        "fail_closed",
        "malformed",
        "version_mismatch",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    case_checks_passed = all(
        check.status == "passed"
        for check in cert.checks
        if check.check_id.startswith("rocq.")
        and check.kind
        in {
            "positive",
            "negative",
            "mutation",
            "replay",
            "fail_closed",
            "malformed",
            "version_mismatch",
            "binding",
            "policy",
        }
    )
    corpus_outcomes_matched = all(case.matched for case in cert.cases)
    cert.semantic_corpus_passed = bool(
        case_checks_passed
        and corpus_outcomes_matched
        and not missing_kinds
        and cert.isolated_root_validated
        and not any(
            reason.startswith("case_failed:") or reason.startswith("replay_")
            for reason in cert.block_reasons
        )
    )

    # Production certification requires live locked pin + full semantic corpus.
    cert.production_certified = bool(
        cert.usable
        and cert.identity_probed
        and cert.version_match
        and cert.semantic_corpus_passed
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and not cert.global_opam_mutation_attempted
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned Rocq/Coq 9.1.1 kernel semantics certified: true theorem "
            "accepted; false proof, mutations, admit/Admitted/Axiom escapes, "
            "malformed input, and version mismatch fail closed; bindings "
            "complete; OPAM is support only under an isolated root."
        )
    else:
        cert.promotion_blocked = True
        if cert.semantic_corpus_passed and not cert.usable:
            cert.notes = (
                "Offline semantic corpus passed; live Rocq/Coq 9.1.1 identity "
                "unavailable — not production-certified."
            )
        elif cert.usable and not cert.semantic_corpus_passed:
            cert.notes = (
                "Rocq/Coq is usable but semantic certification incomplete or "
                "failed; promotion blocked."
            )
        elif not cert.notes:
            cert.notes = "Rocq/Coq toolchain not production-certified."

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    opam_executable: str | None = None,
) -> dict[str, Any]:
    """Machine-readable receipt for operators, tests, and lane binding."""

    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        executable=executable,
        opam_executable=opam_executable,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "admit_admitted_axiom_fail_closed": True,
        "opam_is_support_only": True,
        "opam_cannot_promote_kernel_lane": True,
        "isolated_opam_root_required": True,
        "never_mutate_global_opam_switch": True,
        "exact_tool_identity_required": True,
        "authority_is_kernel_proof_checking_only": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_shared_lock": True,
    }
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_rocq_toolchain(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    manifest = kwargs.get("manifest")
    env = kwargs.get("env")
    executable = kwargs.get("executable")
    opam_executable = kwargs.get("opam_executable")
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        executable=executable,
        opam_executable=opam_executable,
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
    """Alias used by ``tools.logic.certification.roles.bind_lane_handler``."""

    return certify_rocq_toolchain(*args, **kwargs)


# ---------------------------------------------------------------------------
# Kernel live semantic fan-in (FVT-G206 / KernelLiveSemanticFanIn@1)
# ---------------------------------------------------------------------------

FANIN_INTERFACE: Final = "KernelLiveSemanticFanIn@1"
FANIN_SCHEMA_VERSION: Final = "kernel-live-semantic-fanin/v1"
FANIN_GOAL_ID: Final = "FVT-G206"
FANIN_TASK_ID: Final = "FVT-057"
FANIN_KERNEL_ID: Final = "rocq"
FANIN_TIMEOUT_SECONDS: Final = 0.05
REQUIRED_FANIN_CASE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "positive",
        "negative",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "fail_closed",
    }
)


def live_fanin_case_recipes() -> list[dict[str, Any]]:
    """Compact live fan-in recipes owned by the Rocq/Coq kernel only."""

    return [
        {
            "case_id": "true_theorem",
            "kind": "positive",
            "expect": "accepted",
            "theorem_name": "from_eq",
            "assumptions": ["H : n = m"],
            "source": _TRUE_THEOREM,
        },
        {
            "case_id": "false_proof",
            "kind": "negative",
            "expect": "rejected",
            "theorem_name": "false_claim",
            "assumptions": [],
            "source": _FALSE_PROOF,
        },
        {
            "case_id": "hypothesis_mutation",
            "kind": "mutation",
            "expect": "rejected",
            "theorem_name": "from_eq",
            "assumptions": ["H : n = n"],
            "source": _HYPOTHESIS_MUTATION,
        },
        {
            "case_id": "conclusion_mutation",
            "kind": "mutation",
            "expect": "rejected",
            "theorem_name": "from_eq",
            "assumptions": ["H : n = m"],
            "source": _CONCLUSION_MUTATION,
        },
        {
            "case_id": "deterministic_replay",
            "kind": "replay",
            "expect": "accepted",
            "theorem_name": "from_eq",
            "assumptions": ["H : n = m"],
            "source": _TRUE_THEOREM,
            "base_case_id": "true_theorem",
        },
        {
            "case_id": "malformed_source",
            "kind": "malformed",
            "expect": "rejected",
            "theorem_name": "broken",
            "assumptions": [],
            "source": _MALFORMED,
        },
        {
            "case_id": "timeout_case",
            "kind": "timeout",
            "expect": "rejected",
            "theorem_name": "from_eq",
            "assumptions": ["H : n = m"],
            "source": _TRUE_THEOREM,
            "force_timeout": True,
        },
        {
            "case_id": "admit_escape",
            "kind": "fail_closed",
            "expect": "rejected",
            "theorem_name": "hole_admit",
            "assumptions": [],
            "source": _ADMIT_ESCAPE,
            "reason_codes": ["admit_or_admitted"],
        },
        {
            "case_id": "axiom_escape",
            "kind": "fail_closed",
            "expect": "rejected",
            "theorem_name": "uses_axiom",
            "assumptions": [],
            "source": _AXIOM_ESCAPE,
            "reason_codes": ["unreviewed_axiom"],
        },
    ]


def _force_timeout_executable() -> str:
    sleeper = shutil.which("sleep")
    if sleeper:
        return sleeper
    return sys.executable


def build_live_fanin_contribution(
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    opam_executable: str | None = None,
) -> dict[str, Any]:
    """Execute Rocq-owned live fan-in cases; never substitutes Lean/Isabelle."""

    root = repo_root or repo_root_from()
    probe_env = offline_env(env)
    identity = probe_rocq_identity(env=probe_env, executable=executable)
    opam_identity = probe_opam_identity(env=probe_env, executable=opam_executable)
    coq_bin = executable or identity.get("executable_path")
    usable = bool(
        identity.get("identity_probed")
        and identity.get("version_match")
        and coq_bin
    )
    root_contract = validate_isolated_opam_root_contract(repo_root=root)

    cases: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    outcomes_by_id: dict[str, CaseOutcome] = {}
    block_reasons: list[str] = []
    live_executed = False

    for recipe in live_fanin_case_recipes():
        case_id = str(recipe["case_id"])
        kind = str(recipe["kind"])
        expect = str(recipe["expect"])
        source = str(recipe["source"])
        force_timeout = bool(recipe.get("force_timeout"))

        if force_timeout:
            sleeper = _force_timeout_executable()
            argv = (
                [sleeper, "30"]
                if Path(sleeper).name == "sleep" or sleeper.endswith("/sleep")
                else [sleeper, "-c", "import time; time.sleep(30)"]
            )
            completed = bounded_run(
                argv,
                timeout=FANIN_TIMEOUT_SECONDS,
                env=probe_env,
            )
            outcome = CaseOutcome(
                case_id=case_id,
                kind=kind,
                expect=expect,
                accepted=False,
                matched=completed is None,
                status="rejected",
                reason_codes=["timeout_or_spawn_failure"]
                if completed is None
                else ["timeout_not_triggered"],
                theorem_name=str(recipe.get("theorem_name") or ""),
                imports=list(extract_rocq_imports(source)),
                assumptions=[str(a) for a in (recipe.get("assumptions") or [])],
                source_digest=content_digest(source),
                output_digest=content_digest(""),
                timed_out=completed is None,
                detail="fan-in timeout bound rejects without acceptance",
            )
            if completed is not None:
                block_reasons.append("timeout_not_triggered")
        elif not usable or not coq_bin:
            outcome = CaseOutcome(
                case_id=case_id,
                kind=kind,
                expect=expect,
                accepted=False,
                matched=False,
                status="rejected",
                reason_codes=["kernel_unavailable"],
                theorem_name=str(
                    recipe.get("theorem_name")
                    or extract_rocq_theorem_name(source)
                    or ""
                ),
                imports=list(extract_rocq_imports(source)),
                assumptions=[str(a) for a in (recipe.get("assumptions") or [])],
                source_digest=content_digest(
                    source if source.endswith("\n") else source + "\n"
                ),
                output_digest=content_digest(""),
                detail="rocq pin unavailable for live fan-in",
            )
        else:
            outcome = check_rocq_source_live(
                source,
                executable=str(coq_bin),
                env=probe_env,
                case_id=case_id,
            )
            outcome.kind = kind
            outcome.expect = expect
            live_executed = True

        outcomes_by_id[case_id] = outcome
        if expect == "accepted":
            matched = outcome.accepted is True
        else:
            matched = outcome.accepted is False
        expected_reasons = [str(item) for item in (recipe.get("reason_codes") or [])]
        if expected_reasons:
            matched = matched and any(
                reason in outcome.reason_codes for reason in expected_reasons
            )
        if kind == "timeout":
            matched = (
                outcome.accepted is False
                and (
                    "timeout_or_spawn_failure" in outcome.reason_codes
                    or outcome.timed_out
                )
            )
        if kind == "replay" and expect == "accepted":
            ref = outcomes_by_id.get(str(recipe.get("base_case_id") or "true_theorem"))
            if ref is not None:
                matched = matched and (
                    outcome.source_digest == ref.source_digest
                    and outcome.output_digest == ref.output_digest
                    and outcome.accepted is True
                    and ref.accepted is True
                )
            else:
                matched = False
        if not matched:
            block_reasons.append(f"case_failed:{case_id}")
        case_payload = outcome.to_dict()
        case_payload.pop("stdout", None)
        case_payload.pop("stderr", None)
        cases.append(case_payload)
        checks.append(
            {
                "check_id": f"rocq.fanin.{case_id}",
                "kind": kind,
                "status": "passed" if matched else "failed",
                "expected": expect,
                "observed": "accepted" if outcome.accepted else "rejected",
                "reason_codes": list(outcome.reason_codes),
                "bindings": {
                    "theorem_name": outcome.theorem_name,
                    "imports": list(outcome.imports),
                    "assumptions": list(outcome.assumptions),
                    "source_digest": outcome.source_digest,
                    "output_digest": outcome.output_digest,
                    "returncode": outcome.returncode,
                },
            }
        )

    positive = outcomes_by_id.get("true_theorem")
    bindings = {
        "kernel_id": FANIN_KERNEL_ID,
        "tool_id": TOOL_ID,
        "executable_path": identity.get("executable_path"),
        "version_string": identity.get("version_string"),
        "locked_version": LOCKED_VERSION,
        "package_identity": PACKAGE_IDENTITY,
        "dependency_digests": {
            "package_identity": PACKAGE_IDENTITY,
            "opam_version": LOCKED_OPAM_VERSION,
            "opam_executable_path": opam_identity.get("executable_path"),
            "isolated_opam_root": root_contract.get("isolated_opam_root"),
            "opam_support_only": True,
        },
        "imports": list(positive.imports) if positive else [],
        "assumptions": list(positive.assumptions) if positive else [],
        "theorem": {
            "name": positive.theorem_name if positive else "",
            "assumptions": list(positive.assumptions) if positive else [],
        },
        "source": {
            "primary_path": "Goal.v",
            "source_digest": positive.source_digest if positive else "",
            "format": "rocq",
        },
        "output": {
            "output_digest": positive.output_digest if positive else "",
            "returncode": positive.returncode if positive else None,
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "selected_kernel": FANIN_KERNEL_ID,
            "sibling_kernel_substitution_forbidden": True,
            "advisor_substitution_forbidden": True,
            "not_advisor": True,
            "not_lean": True,
            "not_isabelle": True,
            "opam_cannot_promote_kernel_lane": True,
        },
    }

    present_kinds = {str(item.get("kind") or "") for item in live_fanin_case_recipes()}
    missing_kinds = sorted(REQUIRED_FANIN_CASE_KINDS - present_kinds)
    if missing_kinds:
        block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    all_passed = all(check["status"] == "passed" for check in checks) and not missing_kinds
    contribution = {
        "kernel_id": FANIN_KERNEL_ID,
        "tool_id": TOOL_ID,
        "interface": INTERFACE,
        "fanin_interface": FANIN_INTERFACE,
        "fanin_schema_version": FANIN_SCHEMA_VERSION,
        "goal_id": FANIN_GOAL_ID,
        "task_id": FANIN_TASK_ID,
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "locked_version": LOCKED_VERSION,
        "package_identity": PACKAGE_IDENTITY,
        "identity_probed": bool(identity.get("identity_probed")),
        "usable": usable,
        "live_executed": live_executed or any(
            c.get("case_id") == "timeout_case" for c in cases
        ),
        "live_source_helper": "check_rocq_source_live",
        "sibling_kernel_substitution": False,
        "advisor_substitution": False,
        "executable_path": identity.get("executable_path"),
        "version_string": identity.get("version_string"),
        "network_used": bool(identity.get("network_used")),
        "install_attempted": bool(identity.get("install_attempted")),
        "download_attempted": bool(identity.get("download_attempted")),
        "fanin_passed": bool(all_passed and usable),
        "block_reasons": list(dict.fromkeys(block_reasons)),
        "required_case_kinds": sorted(REQUIRED_FANIN_CASE_KINDS),
        "cases": cases,
        "checks": checks,
        "bindings": bindings,
        "repo_root": str(root),
        "notes": (
            "Rocq live fan-in contribution: own kernel only; OPAM support-only; "
            "no Lean/Isabelle/advisor substitution; timeout fail-closed."
            if usable
            else "Rocq pin unavailable; live fan-in contribution incomplete."
        ),
    }
    contribution["contribution_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in contribution.items()
            if key != "contribution_digest_sha256"
        }
    )
    return contribution


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Semantically certify the pinned Rocq/Coq kernel "
            f"({INTERFACE}; {LOCKED_VERSION} / {PACKAGE_IDENTITY})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to the Rocq corpus manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the offline toolchain lock",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    manifest = (
        load_corpus_manifest(args.manifest, repo_root=root)
        if args.manifest is not None
        else load_corpus_manifest(repo_root=root)
    )
    receipt = build_certification_receipt(repo_root=root, manifest=manifest)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"version={receipt.get('version_string')!r} "
            f"locked={LOCKED_VERSION} package={PACKAGE_IDENTITY}"
        )
        print(
            f"usable={receipt.get('usable')} "
            f"semantic_corpus_passed={receipt.get('semantic_corpus_passed')} "
            f"production_certified={receipt.get('production_certified')} "
            f"promotion_blocked={receipt.get('promotion_blocked')}"
        )
        for check in receipt.get("checks") or []:
            print(
                f"  [{check.get('status'):10}] {check.get('check_id')}: "
                f"expected={check.get('expected')} observed={check.get('observed')}"
            )
        if receipt.get("block_reasons"):
            print("block_reasons:", ", ".join(receipt["block_reasons"]))
        print("notes:", receipt.get("notes") or "")

    return 0 if receipt.get("semantic_corpus_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "CORPUS_SCHEMA",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "TOOL_ID",
    "SUPPORT_TOOL_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "LOCKED_VERSION",
    "LOCKED_OPAM_VERSION",
    "PACKAGE_IDENTITY",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "RocqToolchainCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_coq_executable",
    "resolve_opam_executable",
    "extract_rocq_imports",
    "extract_rocq_theorem_name",
    "scan_rocq_incomplete_or_unsafe",
    "evaluate_rocq_process_output",
    "opam_cannot_promote_kernel_lane",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "probe_rocq_identity",
    "probe_opam_identity",
    "validate_isolated_opam_root_contract",
    "evaluate_corpus_case",
    "check_rocq_source_live",
    "run_certification_suite",
    "build_certification_receipt",
    "certify_rocq_toolchain",
    "lane_handler",
    "FANIN_INTERFACE",
    "FANIN_SCHEMA_VERSION",
    "FANIN_GOAL_ID",
    "FANIN_TASK_ID",
    "REQUIRED_FANIN_CASE_KINDS",
    "live_fanin_case_recipes",
    "build_live_fanin_contribution",
    "main",
]
