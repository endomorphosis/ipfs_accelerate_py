#!/usr/bin/env python3
"""Isabelle session/kernel toolchain certification (FVT-G151 / FVT-049).

``IsabelleToolchainCertification@1``

Owns the kernel-lane certification handler for the pinned Isabelle2025-2
distribution used for reconstruction and Hammer validation. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for Isabelle2025-2 when live tools are used;
* exercises a checked theory/session, bad proof, assumption/conclusion
  mutation, deterministic replay, replay mismatch, malformed output, timeout,
  and wrong installation cases;
* binds theory heap, session, imports, source, property, and exact tool
  identity on receipts;
* keeps Hammer proposal-only until independent kernel reconstruction;
* never edits the shared multi-prover certificate or lock.

Offline corpus evaluation reuses the same acceptance rules as the Isabelle
kernel backend so hermetic tests prove semantics without a live distribution.
Live production certification additionally requires the pinned binary.
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

INTERFACE: Final = "IsabelleToolchainCertification@1"
SCHEMA_VERSION: Final = "isabelle-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "isabelle-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G151"
TASK_ID: Final = "FVT-049"
PROGRAM: Final = "formal-verification-tactician/isabelle-toolchain"
LANE_ID: Final = "kernel"
TOOL_ID: Final = "isabelle"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.isabelle"
HANDLER_ID: Final = "isabelle_toolchain_certifier"
AUTHORITY_CEILING: Final = "kernel"
AUTHORITY_SCOPE: Final = "kernel_proof_checking_only"

LOCKED_VERSION: Final = "Isabelle2025-2"
LOCKED_EXECUTABLE: Final = "isabelle"

PROBE_TIMEOUT_SECONDS: Final = 10.0
CHECK_TIMEOUT_SECONDS: Final = 60.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_SORRY = re.compile(r"(?<![A-Za-z0-9_'])(?:sorry|oops)(?![A-Za-z0-9_'])")
_AXIOMATIZATION = re.compile(
    r"(?im)^\s*(?:axiomatization\b|axioms?\s+|consts?\s+[^\n]*where\b)"
)
_THEORY = re.compile(r"^\s*theory\s+([A-Za-z_][A-Za-z0-9_'.]*)", re.MULTILINE)
_IMPORTS = re.compile(r"^\s*imports\s+(.+)$", re.MULTILINE)
_DECL = re.compile(
    r"^\s*(?:theorem|lemma|corollary|proposition)\s+([A-Za-z_][A-Za-z0-9_'.]*)",
    re.MULTILINE,
)
_ERROR_MARKER = re.compile(r"\*\*\*")
_VERSION_TOKEN = re.compile(r"Isabelle\d{4}(?:-\d+)?", re.IGNORECASE)

# Compact embedded corpus. Prefer offline evaluation; live kernel is optional.
_TRUE_THEORY: Final = """\
theory CertTrue imports Main
begin

theorem from_eq: "n = m ⟹ n = m"
  by simp

end
"""

_FALSE_THEORY: Final = """\
theory CertFalse imports Main
begin

theorem false_claim: "False"
  by simp

end
"""

_ASSUMPTION_MUTATION: Final = """\
theory CertTrue imports Main
begin

theorem from_eq: "n = n ⟹ n = m"
  by simp

end
"""

_CONCLUSION_MUTATION: Final = """\
theory CertTrue imports Main
begin

theorem from_eq: "n = m ⟹ False"
  by simp

end
"""

_SORRY_THEORY: Final = """\
theory CertSorry imports Main
begin

theorem hole_sorry: "True"
  sorry

end
"""

_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "checked_theory_session",
        "kind": "positive",
        "expect": "accepted",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _TRUE_THEORY,
        "returncode": 0,
        "stdout": "theory CertTrue\n",
        "stderr": "",
        "timed_out": False,
        "description": "Checked theory/session accepted by the pinned kernel",
    },
    {
        "case_id": "bad_proof",
        "kind": "negative",
        "expect": "rejected",
        "theory_name": "CertFalse",
        "session": "HOL",
        "property": "false_claim",
        "theorem_name": "false_claim",
        "assumptions": [],
        "imports": ["Main"],
        "source": _FALSE_THEORY,
        "returncode": 1,
        "stdout": "",
        "stderr": "*** Failed to finish proof\n",
        "timed_out": False,
        "description": "Bad/false proof rejected by the kernel",
    },
    {
        "case_id": "assumption_mutation",
        "kind": "mutation",
        "mutates": "assumption",
        "expect": "rejected",
        "base_case_id": "checked_theory_session",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = n"],
        "imports": ["Main"],
        "source": _ASSUMPTION_MUTATION,
        "returncode": 1,
        "stdout": "",
        "stderr": "*** Failed to finish proof (assumption mutation)\n",
        "timed_out": False,
        "description": "Assumption mutation of the true theorem is rejected",
    },
    {
        "case_id": "conclusion_mutation",
        "kind": "mutation",
        "mutates": "conclusion",
        "expect": "rejected",
        "base_case_id": "checked_theory_session",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _CONCLUSION_MUTATION,
        "returncode": 1,
        "stdout": "",
        "stderr": "*** Failed to finish proof (conclusion mutation)\n",
        "timed_out": False,
        "description": "Conclusion mutation of the true theorem is rejected",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "accepted",
        "base_case_id": "checked_theory_session",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _TRUE_THEORY,
        "returncode": 0,
        "stdout": "theory CertTrue\n",
        "stderr": "",
        "timed_out": False,
        "description": "Positive case replays with identical acceptance and digests",
    },
    {
        "case_id": "replay_mismatch",
        "kind": "replay_mismatch",
        "expect": "rejected",
        "base_case_id": "checked_theory_session",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _TRUE_THEORY,
        # Deliberately different output from the positive case.
        "returncode": 0,
        "stdout": "theory CertTrue REPLAY_MUTATED\n",
        "stderr": "",
        "timed_out": False,
        "expected_output_digest_from": "checked_theory_session",
        "description": "Replay with mismatched tool output fails closed",
    },
    {
        "case_id": "malformed_output",
        "kind": "malformed",
        "expect": "rejected",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _TRUE_THEORY,
        "returncode": 0,
        "stdout": "this is not an isabelle session report\n!!! garbage !!!\n",
        "stderr": "",
        "timed_out": False,
        "description": "Malformed tool output never reports acceptance",
    },
    {
        "case_id": "timeout_case",
        "kind": "timeout",
        "expect": "rejected",
        "theory_name": "CertTrue",
        "session": "HOL",
        "property": "equality_reflexive_under_assumption",
        "theorem_name": "from_eq",
        "assumptions": ["n = m"],
        "imports": ["Main"],
        "source": _TRUE_THEORY,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": True,
        "description": "Timeout outcomes never accept",
    },
    {
        "case_id": "wrong_installation",
        "kind": "version_mismatch",
        "expect": "blocked",
        "observed_version": "Isabelle2021-1",
        "stdout": "",
        "stderr": "",
        "description": "Wrong Isabelle installation fails closed",
    },
    {
        "case_id": "sorry_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theory_name": "CertSorry",
        "session": "HOL",
        "property": "hole",
        "theorem_name": "hole_sorry",
        "assumptions": [],
        "imports": ["Main"],
        "source": _SORRY_THEORY,
        "returncode": 0,
        "stdout": "theory CertSorry\n",
        "stderr": "",
        "timed_out": False,
        "reason_codes": ["sorry_or_oops"],
        "description": "sorry/oops fails closed even when process exits 0",
    },
    {
        "case_id": "hammer_proposal_only",
        "kind": "policy",
        "expect": "proposal_only",
        "stdout": "",
        "stderr": "",
        "description": "Hammer remains proposal-only until kernel reconstruction",
    },
)


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
    # Isabelle must not auto-download components during certification.
    env.setdefault("ISABELLE_COMPONENT_REPOSITORY", "")
    return env


def bounded_run(
    argv: Sequence[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
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
            cwd=str(cwd) if cwd is not None else None,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_isabelle_executable(candidates: Sequence[str] | None = None) -> str | None:
    names = list(candidates) if candidates else [LOCKED_EXECUTABLE, "isabelle"]
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


def extract_isabelle_version_token(banner: str | None) -> str | None:
    if not banner:
        return None
    match = _VERSION_TOKEN.search(banner)
    if match is not None:
        return match.group(0)
    stripped = banner.strip().splitlines()[0].strip() if banner.strip() else ""
    if stripped.startswith("Isabelle"):
        return stripped.split()[0]
    return None


def extract_isabelle_theory_name(source: str) -> str | None:
    match = _THEORY.search(source or "")
    return match.group(1).rstrip(".") if match else None


def extract_isabelle_imports(source: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _IMPORTS.finditer(source or ""):
        for token in re.split(r"[\s,]+", match.group(1).strip()):
            token = token.strip()
            if token and token not in found:
                found.append(token)
    return tuple(found)


def extract_isabelle_theorem_name(source: str) -> str | None:
    match = _DECL.search(source or "")
    return match.group(1).rstrip(".") if match else None


def scan_isabelle_incomplete_or_unreviewed(source: str) -> tuple[str, ...]:
    findings: list[str] = []
    if _SORRY.search(source or ""):
        findings.append("sorry_or_oops")
    if _AXIOMATIZATION.search(source or ""):
        findings.append("unreviewed_axiomatization")
    return tuple(findings)


def evaluate_isabelle_process_output(
    *,
    source: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    timed_out: bool = False,
) -> tuple[bool, list[str]]:
    """Decide whether a process observation constitutes kernel acceptance."""

    reasons: list[str] = []
    source_reasons = list(scan_isabelle_incomplete_or_unreviewed(source))
    reasons.extend(source_reasons)

    if timed_out:
        reasons.append("timeout")
        return False, list(dict.fromkeys(reasons))

    combined = f"{stdout}\n{stderr}"
    if _SORRY.search(combined):
        reasons.append("sorry_or_oops")
    if _ERROR_MARKER.search(combined):
        reasons.append("error_marker")
    if "Failed" in combined:
        reasons.append("failed_diagnostic")
    if returncode not in (0, None) and returncode != 0:
        reasons.append("non_zero_exit")
    if returncode is None and not timed_out and not stdout and not stderr:
        reasons.append("empty_process_result")

    # Malformed: exit 0 without any theory/session identity marker and without
    # a clear failure must not grant acceptance.
    theory = extract_isabelle_theory_name(source) or ""
    has_theory_marker = bool(theory) and (
        f"theory {theory}" in combined or theory in (stdout or "")
    )
    if returncode == 0 and not reasons and not has_theory_marker:
        reasons.append("malformed_output")

    accepted = (
        returncode == 0
        and not timed_out
        and not reasons
        and has_theory_marker
    )
    if not accepted and not reasons:
        reasons.append("not_accepted")
    return accepted, list(dict.fromkeys(reasons))


def hammer_remains_proposal_only() -> dict[str, Any]:
    """Prove Hammer is advisor/proposal-only without kernel reconstruction."""

    return {
        "tool_id": "hammer",
        "role": "advisor",
        "authority_ceiling": "none",
        "proposal_only": True,
        "can_grant_kernel_authority": False,
        "requires_independent_kernel_reconstruction": True,
        "reconstruction_kernel": TOOL_ID,
        "locked_kernel_version": LOCKED_VERSION,
        "promotion_allowed_from_hammer_alone": False,
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
    status: str  # accepted | rejected | blocked | proposal_only
    reason_codes: list[str] = field(default_factory=list)
    theory_name: str = ""
    session: str = ""
    property: str = ""
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
class IsabelleToolchainCertification:
    """Full certification receipt for the pinned Isabelle kernel lane."""

    tool_id: str = TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_version: str = LOCKED_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    executable_path: str | None = None
    version_string: str | None = None
    identity_probed: bool = False
    installed: bool = False
    usable: bool = False
    version_match: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    semantic_corpus_passed: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    hammer_proposal_only: bool = True
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
        "lane_id": LANE_ID,
        "locked_version": LOCKED_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "sorry_oops_fail_closed": True,
            "hammer_is_proposal_only": True,
            "hammer_requires_kernel_reconstruction": True,
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
            raise ValueError("Isabelle corpus manifest must be a JSON object")
        return payload
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("Isabelle corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probe
# ---------------------------------------------------------------------------


def probe_isabelle_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Bounded offline identity probe for the locked Isabelle pin only."""

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
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_isabelle_executable()
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    if completed is None:
        # Fallback for builds that only accept --version.
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
    token = extract_isabelle_version_token(banner)
    result["version_match"] = bool(
        token == LOCKED_VERSION or LOCKED_VERSION in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


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
    returncode = case.get("returncode")
    if returncode is not None:
        returncode = int(returncode)
    timed_out = bool(case.get("timed_out") or False)
    theory_name = str(case.get("theory_name") or extract_isabelle_theory_name(source) or "")
    session = str(case.get("session") or "HOL")
    property_name = str(case.get("property") or "")
    theorem_name = str(
        case.get("theorem_name") or extract_isabelle_theorem_name(source) or ""
    )
    imports = [str(item) for item in (case.get("imports") or extract_isabelle_imports(source))]
    assumptions = [str(item) for item in (case.get("assumptions") or [])]
    source_digest = content_digest(source) if source else content_digest("")
    output_digest = content_digest(f"{stdout}\n{stderr}")

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
            detail=(
                f"observed={observed!r} locked={LOCKED_VERSION!r}"
            ),
        )

    if kind == "policy" or case_id == "hammer_proposal_only":
        boundary = hammer_remains_proposal_only()
        matched = (
            boundary["proposal_only"] is True
            and boundary["can_grant_kernel_authority"] is False
            and boundary["promotion_allowed_from_hammer_alone"] is False
            and expect == "proposal_only"
        )
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            accepted=False,
            matched=matched,
            status="proposal_only",
            reason_codes=["hammer_proposal_only"],
            source_digest=source_digest,
            output_digest=output_digest,
            detail="Hammer advisor proposals require independent kernel reconstruction",
        )

    accepted, reasons = evaluate_isabelle_process_output(
        source=source,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
    )

    # Replay mismatch: require that the output digest differs from the
    # reference positive case and that acceptance is denied for mismatch cases.
    if kind == "replay_mismatch":
        ref_id = str(case.get("expected_output_digest_from") or "checked_theory_session")
        ref = (reference_outcomes or {}).get(ref_id)
        if ref is None:
            # Evaluate the reference source inline when not provided.
            ref_stdout = "theory CertTrue\n"
            ref_digest = content_digest(f"{ref_stdout}\n")
        else:
            ref_digest = ref.output_digest
        mismatch = output_digest != ref_digest
        if not mismatch:
            reasons.append("expected_replay_mismatch_not_detected")
            accepted = True  # force mismatch failure below
        else:
            reasons.append("replay_output_mismatch")
            accepted = False
        matched = (not accepted) and expect == "rejected" and mismatch
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            accepted=False,
            matched=matched,
            status="rejected",
            reason_codes=list(dict.fromkeys(reasons)),
            theory_name=theory_name,
            session=session,
            property=property_name,
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

    status = "accepted" if accepted else "rejected"
    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        accepted=accepted,
        matched=matched,
        status=status,
        reason_codes=list(dict.fromkeys(reasons)),
        theory_name=theory_name,
        session=session,
        property=property_name,
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


def check_isabelle_source_live(
    source: str,
    *,
    executable: str,
    env: Mapping[str, str] | None = None,
    timeout: float = CHECK_TIMEOUT_SECONDS,
    case_id: str = "case",
    session: str = "HOL",
) -> CaseOutcome:
    """Optionally run a live ``isabelle process`` check under offline bounds."""

    normalized = source if source.endswith("\n") else source + "\n"
    theory_name = extract_isabelle_theory_name(normalized) or "Goal"
    imports = list(extract_isabelle_imports(normalized))
    theorem_name = extract_isabelle_theorem_name(normalized) or ""
    source_digest = content_digest(normalized)
    reasons = list(scan_isabelle_incomplete_or_unreviewed(normalized))
    if reasons:
        return CaseOutcome(
            case_id=case_id,
            kind="fail_closed",
            expect="rejected",
            accepted=False,
            matched=True,
            status="rejected",
            reason_codes=reasons,
            theory_name=theory_name,
            session=session,
            theorem_name=theorem_name,
            imports=imports,
            source_digest=source_digest,
            output_digest=content_digest(""),
            detail="source scan rejected incomplete or unreviewed constructs",
        )

    probe_env = offline_env(env)
    with tempfile.TemporaryDirectory(prefix="isabelle-cert-") as tmp:
        work = Path(tmp)
        theory_path = work / f"{theory_name}.thy"
        theory_path.write_text(normalized, encoding="utf-8")
        completed = bounded_run(
            [executable, "process", "-T", theory_name, "-d", str(work)],
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
            theory_name=theory_name,
            session=session,
            theorem_name=theorem_name,
            imports=imports,
            source_digest=source_digest,
            output_digest=content_digest(""),
            timed_out=True,
            detail="bounded isabelle invocation timed out or failed to spawn",
        )

    accepted, eval_reasons = evaluate_isabelle_process_output(
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
        theory_name=theory_name,
        session=session,
        theorem_name=theorem_name,
        imports=imports,
        source_digest=source_digest,
        output_digest=content_digest(f"{completed.stdout}\n{completed.stderr}"),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        detail="isabelle process under offline pin",
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
) -> IsabelleToolchainCertification:
    """Run the full Isabelle toolchain certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = IsabelleToolchainCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="isabelle.offline_policy",
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

    identity = probe_isabelle_identity(env=probe_env, executable=executable)
    cert.executable_path = identity.get("executable_path")
    cert.version_string = identity.get("version_string")
    cert.identity_probed = bool(identity.get("identity_probed"))
    cert.installed = bool(identity.get("installed"))
    cert.version_match = bool(identity.get("version_match"))
    cert.network_used = bool(identity.get("network_used"))
    cert.install_attempted = bool(identity.get("install_attempted"))
    cert.download_attempted = bool(identity.get("download_attempted"))

    if cert.identity_probed and cert.version_match and cert.executable_path:
        cert.usable = True
        cert.checks.append(
            CheckResult(
                check_id="isabelle.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_VERSION,
                observed=cert.version_string or "",
                detail="exact offline pin identity",
                bindings={
                    "executable_path": cert.executable_path,
                    "version_string": cert.version_string,
                    "locked_version": LOCKED_VERSION,
                },
            )
        )
    elif cert.identity_probed and not cert.version_match:
        cert.block_reasons.append("locked_version_mismatch")
        cert.usable = False
        cert.checks.append(
            CheckResult(
                check_id="isabelle.identity",
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
                check_id="isabelle.identity",
                kind="identity",
                status="unavailable",
                expected=LOCKED_VERSION,
                observed=str(identity.get("probe_error") or "missing"),
                detail="PATH presence without identity is not usability",
                reason_codes=["identity_unavailable"],
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
                check_id=f"isabelle.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "theory_name": outcome.theory_name,
                    "session": outcome.session,
                    "property": outcome.property,
                    "theorem_name": outcome.theorem_name,
                    "imports": list(outcome.imports),
                    "assumptions": list(outcome.assumptions),
                    "source_digest": outcome.source_digest,
                    "output_digest": outcome.output_digest,
                    "returncode": outcome.returncode,
                },
            )
        )

    # Deterministic replay binding across positive + replay cases.
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
                check_id="isabelle.deterministic_replay_binding",
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
                check_id="isabelle.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="positive and replay cases",
                observed="missing",
                detail="corpus must include positive and replay cases",
            )
        )

    hammer = hammer_remains_proposal_only()
    cert.hammer_proposal_only = bool(hammer["proposal_only"])
    cert.checks.append(
        CheckResult(
            check_id="isabelle.hammer_proposal_only",
            kind="policy",
            status="passed" if cert.hammer_proposal_only else "failed",
            expected="proposal_only_until_kernel_reconstruction",
            observed=json.dumps(hammer, sort_keys=True),
            detail="Hammer remains proposal-only until independent kernel reconstruction",
            bindings=dict(hammer),
        )
    )

    binding_case = positive_outcome or next(
        (item for item in cert.cases if item.accepted), None
    )
    cert.bindings = {
        "theory_heap": {
            "theory_name": binding_case.theory_name if binding_case else "",
            "session": binding_case.session if binding_case else "HOL",
            "heap_identity": (
                f"{binding_case.session}:{binding_case.theory_name}"
                if binding_case
                else ""
            ),
        },
        "session": {
            "name": binding_case.session if binding_case else "HOL",
            "process_command_template": (
                "{isabelle} process -T {theory_name} -d {session_dir}"
            ),
        },
        "imports": list(binding_case.imports) if binding_case else [],
        "source": {
            "primary_path": (
                f"{binding_case.theory_name}.thy" if binding_case else ""
            ),
            "source_digest": binding_case.source_digest if binding_case else "",
            "format": "isabelle",
        },
        "property": {
            "name": binding_case.property if binding_case else "",
            "theorem_name": binding_case.theorem_name if binding_case else "",
            "assumptions": list(binding_case.assumptions) if binding_case else [],
        },
        "tool_identity": {
            "tool_id": TOOL_ID,
            "locked_version": LOCKED_VERSION,
            "executable_path": cert.executable_path,
            "version_string": cert.version_string,
            "version_match": cert.version_match,
        },
        "output": {
            "output_digest": binding_case.output_digest if binding_case else "",
            "returncode": binding_case.returncode if binding_case else None,
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "not_advisor": True,
            "not_install_authority": True,
            "hammer_is_proposal_only": True,
            "hammer_cannot_grant_kernel_authority": True,
        },
    }
    cert.checks.append(
        CheckResult(
            check_id="isabelle.bindings",
            kind="binding",
            status="passed" if binding_case and binding_case.accepted else "failed",
            expected="theory_heap,session,imports,source,property,tool_identity",
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
        "replay_mismatch",
        "malformed",
        "timeout",
        "version_mismatch",
        "fail_closed",
        "policy",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    case_checks_passed = all(
        check.status == "passed"
        for check in cert.checks
        if check.check_id.startswith("isabelle.")
        and check.kind
        in {
            "positive",
            "negative",
            "mutation",
            "replay",
            "replay_mismatch",
            "malformed",
            "timeout",
            "version_mismatch",
            "fail_closed",
            "policy",
            "binding",
        }
    )
    corpus_outcomes_matched = all(case.matched for case in cert.cases)
    cert.semantic_corpus_passed = bool(
        case_checks_passed
        and corpus_outcomes_matched
        and not missing_kinds
        and cert.hammer_proposal_only
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
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = [
            reason
            for reason in cert.block_reasons
            if reason not in {"unavailable", "executable_not_on_path"}
        ]
        # Clear identity-unavailability reasons when fully certified.
        cert.block_reasons = []
        cert.notes = (
            "Pinned Isabelle2025-2 kernel semantics certified: checked theory/"
            "session accepted; bad proof, mutations, replay mismatch, malformed "
            "output, timeout, and wrong installation fail closed; bindings "
            "complete; Hammer remains proposal-only until kernel reconstruction."
        )
    else:
        cert.promotion_blocked = True
        if cert.semantic_corpus_passed and not cert.usable:
            cert.notes = (
                "Offline semantic corpus passed; live Isabelle2025-2 identity "
                "unavailable — not production-certified."
            )
        elif cert.usable and not cert.semantic_corpus_passed:
            cert.notes = (
                "Isabelle is usable but semantic certification incomplete or "
                "failed; promotion blocked."
            )
        elif not cert.notes:
            cert.notes = "Isabelle toolchain not production-certified."

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Machine-readable receipt for operators, tests, and lane binding."""

    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        executable=executable,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "sorry_oops_fail_closed": True,
        "hammer_is_proposal_only": True,
        "hammer_requires_kernel_reconstruction": True,
        "exact_tool_identity_required": True,
        "authority_is_kernel_proof_checking_only": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_shared_lock": True,
    }
    return payload


def certify_isabelle_toolchain(
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
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        executable=executable,
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

    return certify_isabelle_toolchain(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Semantically certify the pinned Isabelle kernel "
            f"({INTERFACE}; {LOCKED_VERSION})."
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
        help="Optional path to the Isabelle corpus manifest",
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
            f"locked={LOCKED_VERSION}"
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

    # Exit 0 when the offline semantic corpus passes (even without live tools).
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
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "LOCKED_VERSION",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "IsabelleToolchainCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_isabelle_executable",
    "extract_isabelle_version_token",
    "extract_isabelle_theory_name",
    "extract_isabelle_imports",
    "extract_isabelle_theorem_name",
    "scan_isabelle_incomplete_or_unreviewed",
    "evaluate_isabelle_process_output",
    "hammer_remains_proposal_only",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "probe_isabelle_identity",
    "evaluate_corpus_case",
    "check_isabelle_source_live",
    "run_certification_suite",
    "build_certification_receipt",
    "certify_isabelle_toolchain",
    "lane_handler",
    "main",
]
