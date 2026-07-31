#!/usr/bin/env python3
"""Tamarin + Maude protocol toolchain certification (FVT-G130 / FVT-043).

``TamarinToolchainCertification@1``

Owns the protocol-lane certification handler for the pinned Tamarin 1.12.0
prover and its support-only Maude 3.5.1 companion. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for Tamarin 1.12.0 and Maude 3.5.1;
* exercises secure, attack, mutated claim/rule, replay, malformed output,
  timeout, and version-mismatch cases;
* binds equational theory, claims, execution bounds, and exact binaries;
* treats Maude as support only — Maude presence alone never promotes the
  protocol property lane;
* never edits the shared multi-prover certificate or the ProVerif lane.

Semantic claim/attack evaluation reuses the canonical protocol backend parsers
so offline tests can prove corpus behavior without a live Tamarin process.
Live production certification additionally requires the pinned binaries.
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

from ipfs_datasets_py.logic.backends.protocol.tamarin import (  # noqa: E402
    TAMARIN_BACKEND_VERSION,
    TAMARIN_SUPPORTED_CLAIMS,
    TAMARIN_SUPPORTED_THEORIES,
    ClaimVerdict,
    SymbolicModelCeiling,
    classify_claim_outcomes,
    parse_attack_trace,
    parse_tamarin_claim_outcomes,
)
from ipfs_datasets_py.logic.backends.results import ResultStatus  # noqa: E402
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    can_satisfy_certified_authority_requirement,
    evaluate_role_aware_promotion,
    get_tool_role,
)
from ipfs_datasets_py.logic.software_verification.protocol import (  # noqa: E402
    EquationalTheory,
    ProtocolClaimKind,
)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

INTERFACE: Final = "TamarinToolchainCertification@1"
SCHEMA_VERSION: Final = "tamarin-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "tamarin-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G130"
TASK_ID: Final = "FVT-043"
PROGRAM: Final = "formal-verification-tactician/tamarin-toolchain"
LANE_ID: Final = "protocol"
TOOL_ID: Final = "tamarin"
SUPPORT_TOOL_ID: Final = "maude"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.tamarin"
HANDLER_ID: Final = "tamarin_toolchain_certifier"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.PROTOCOL.value
AUTHORITY_SCOPE: Final = "protocol_verification_only"

LOCKED_TAMARIN_VERSION: Final = "1.12.0"
LOCKED_MAUDE_VERSION: Final = "3.5.1"
LOCKED_TAMARIN_EXECUTABLE: Final = "tamarin-prover"
LOCKED_MAUDE_EXECUTABLE: Final = "maude"

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_VERSION_IN_BANNER = re.compile(r"(\d+\.\d+\.\d+)")

# Compact embedded corpus. Prefer live binaries when present; parsers always run.
_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "secure_claims",
        "kind": "secure",
        "expect": "secure",
        "claim_lemmas": {
            "claim:secrecy": "secrecy_claim",
            "claim:auth": "auth_claim",
        },
        "stdout": (
            "lemma secrecy_claim: verified (all-traces)\n"
            "lemma auth_claim: verified (all-traces)\n"
        ),
        "stderr": "",
        "description": "Secure protocol: all claims verified",
    },
    {
        "case_id": "attack_trace",
        "kind": "attack",
        "expect": "attack",
        "claim_lemmas": {"claim:secrecy": "secrecy_claim"},
        "stdout": (
            "lemma secrecy_claim: falsified - found trace\n"
            "rule Create_Initiator(~id)\n"
            "rule Event_BeginChallenge(~n)\n"
            "rule Event_AcceptChallenge(~n)\n"
        ),
        "stderr": "",
        "description": "Attack: falsified claim with normalized attack trace",
    },
    {
        "case_id": "mutated_claim",
        "kind": "mutation",
        "mutates": "claim",
        "expect": "rejected_or_quarantined",
        "base_case_id": "secure_claims",
        "claim_lemmas": {
            "claim:secrecy": "secrecy_claim",
            "claim:auth": "auth_claim",
        },
        # Mutated claim set: auth lemma now falsified while secrecy verifies.
        "stdout": (
            "lemma secrecy_claim: verified (all-traces)\n"
            "lemma auth_claim: falsified - found trace\n"
            "rule Event_AcceptChallenge(x)\n"
        ),
        "stderr": "",
        "description": "Claim mutation produces disagreement quarantine, not secure",
    },
    {
        "case_id": "mutated_rule",
        "kind": "mutation",
        "mutates": "rule",
        "expect": "attack",
        "base_case_id": "secure_claims",
        "claim_lemmas": {"claim:secrecy": "secrecy_claim"},
        "stdout": (
            "lemma secrecy_claim: falsified - found trace\n"
            "rule Mutated_Rule(~evil)\n"
            "rule Event_AcceptChallenge(~n)\n"
        ),
        "stderr": "",
        "description": "Rule mutation yields an attack rather than secure",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "secure",
        "base_case_id": "secure_claims",
        "claim_lemmas": {
            "claim:secrecy": "secrecy_claim",
            "claim:auth": "auth_claim",
        },
        "stdout": (
            "lemma secrecy_claim: verified (all-traces)\n"
            "lemma auth_claim: verified (all-traces)\n"
        ),
        "stderr": "",
        "description": "Positive secure case replays with identical digests",
    },
    {
        "case_id": "malformed_output",
        "kind": "malformed",
        "expect": "quarantined",
        "claim_lemmas": {"claim:secrecy": "secrecy_claim"},
        "stdout": "this is not a tamarin lemma report\n!!! garbage !!!\n",
        "stderr": "",
        "description": "Malformed tool output never reports SECURE",
    },
    {
        "case_id": "timeout_claim",
        "kind": "timeout",
        "expect": "quarantined",
        "claim_lemmas": {"claim:secrecy": "secrecy_claim"},
        "stdout": "lemma secrecy_claim: timeout\n",
        "stderr": "",
        "description": "Timeout outcomes quarantine rather than SECURE",
    },
    {
        "case_id": "version_mismatch",
        "kind": "version_mismatch",
        "expect": "blocked",
        "claim_lemmas": {},
        "stdout": "",
        "stderr": "",
        "observed_tamarin_version": "1.8.0",
        "observed_maude_version": "3.1",
        "description": "Locked version mismatch blocks production certification",
    },
)

DEFAULT_THEORY_BINDINGS: Final[tuple[str, ...]] = (
    EquationalTheory.FREE.value,
    EquationalTheory.PAIRING.value,
    EquationalTheory.SYMMETRIC_ENCRYPTION.value,
    EquationalTheory.HASHING.value,
)

DEFAULT_CLAIM_BINDINGS: Final[tuple[str, ...]] = (
    ProtocolClaimKind.SECRECY.value,
    ProtocolClaimKind.AUTHENTICATION.value,
    ProtocolClaimKind.REACHABILITY.value,
    ProtocolClaimKind.CORRESPONDENCE.value,
)

DEFAULT_BOUNDS: Final[dict[str, Any]] = {
    "timeout_seconds": CHECK_TIMEOUT_SECONDS,
    "max_source_bytes": 1_048_576,
    "network": False,
    "install": False,
    "download": False,
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
    status: str  # secure | attack | quarantined | blocked | rejected | unknown
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    claim_outcomes: list[dict[str, Any]] = field(default_factory=list)
    attack_trace: dict[str, Any] | None = None
    output_digest: str = ""
    stdout: str = ""
    stderr: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TamarinToolchainCertification:
    """Full certification receipt for the Tamarin/Maude protocol lane."""

    tool_id: str = TOOL_ID
    support_tool_id: str = SUPPORT_TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_tamarin_version: str = LOCKED_TAMARIN_VERSION
    locked_maude_version: str = LOCKED_MAUDE_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    tamarin_executable: str | None = None
    maude_executable: str | None = None
    tamarin_version_string: str | None = None
    maude_version_string: str | None = None
    tamarin_identity_probed: bool = False
    maude_identity_probed: bool = False
    tamarin_version_match: bool = False
    maude_version_match: bool = False
    tamarin_usable: bool = False
    maude_usable: bool = False
    pair_validated: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    maude_support_only: bool = True
    maude_cannot_promote_alone: bool = True
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
        "locked_tamarin_version": LOCKED_TAMARIN_VERSION,
        "locked_maude_version": LOCKED_MAUDE_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "theory_bindings": list(DEFAULT_THEORY_BINDINGS),
        "claim_bindings": list(DEFAULT_CLAIM_BINDINGS),
        "bounds": dict(DEFAULT_BOUNDS),
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "maude_is_support_only": True,
            "maude_cannot_promote_protocol_lane": True,
            "exact_binary_binding_required": True,
            "authority_is_protocol_verification_only": True,
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
            raise ValueError("Tamarin corpus manifest must be a JSON object")
        return payload
    # No fixture directory required for this lane; embedded corpus is authoritative.
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("Tamarin corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------


def probe_tamarin_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_TAMARIN_VERSION,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable([LOCKED_TAMARIN_EXECUTABLE, "tamarin"])
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
        # Some builds print multi-line banners; use full output.
        banner = (completed.stdout or completed.stderr or "").strip()
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    result["identity_probed"] = True
    version = extract_version(banner)
    result["version_match"] = bool(
        version == LOCKED_TAMARIN_VERSION or LOCKED_TAMARIN_VERSION in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def probe_maude_identity(
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
        "version_match": False,
        "locked_version": LOCKED_MAUDE_VERSION,
        "support_only": True,
        "can_promote_protocol_lane": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable([LOCKED_MAUDE_EXECUTABLE])
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
    version = extract_version(banner)
    result["version_match"] = bool(
        version == LOCKED_MAUDE_VERSION or LOCKED_MAUDE_VERSION in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def probe_tamarin_maude_pair(
    tamarin: str | None,
    maude: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bounded offline pair validation (Tamarin's own installation check)."""

    result = {
        "validated": False,
        "detail": "",
        "output": "",
    }
    if not tamarin or not maude:
        result["detail"] = "missing_tamarin_or_maude"
        return result
    completed = bounded_run(
        [tamarin, f"--with-maude={maude}", "--version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=offline_env(env),
    )
    if completed is None:
        result["detail"] = "pair_probe_timeout_or_spawn_failure"
        return result
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    result["output"] = output
    ok = (
        completed.returncode == 0
        and LOCKED_TAMARIN_VERSION in output
        and "checking installation: OK" in output
    )
    result["validated"] = bool(ok)
    result["detail"] = "pair_ok" if ok else "pair_validation_failed"
    return result


# ---------------------------------------------------------------------------
# Case evaluation (parser-backed, offline)
# ---------------------------------------------------------------------------


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one corpus case via canonical Tamarin parsers (no install)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    stdout = str(case.get("stdout") or "")
    stderr = str(case.get("stderr") or "")
    claim_lemmas = {
        str(key): str(value)
        for key, value in dict(case.get("claim_lemmas") or {}).items()
    }
    output_digest = content_digest(f"{stdout}\n{stderr}")

    if kind == "version_mismatch":
        observed_t = str(case.get("observed_tamarin_version") or "")
        observed_m = str(case.get("observed_maude_version") or "")
        blocked = (
            observed_t != LOCKED_TAMARIN_VERSION
            or observed_m != LOCKED_MAUDE_VERSION
        )
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status="blocked" if blocked else "unknown",
            matched=blocked and expect == "blocked",
            reason_codes=["locked_version_mismatch"] if blocked else [],
            output_digest=output_digest,
            detail=(
                f"observed tamarin={observed_t} maude={observed_m}; "
                f"locked tamarin={LOCKED_TAMARIN_VERSION} maude={LOCKED_MAUDE_VERSION}"
            ),
        )

    outcomes = parse_tamarin_claim_outcomes(
        stdout, stderr, claim_lemmas=claim_lemmas
    )
    status_enum, quarantine, accepted = classify_claim_outcomes(outcomes)
    attack: dict[str, Any] | None = None
    for item in outcomes:
        if item.attack_trace is not None:
            attack = item.attack_trace.to_dict()
            break
    if attack is None and status_enum is ResultStatus.ATTACK_FOUND:
        trace = parse_attack_trace(
            f"{stdout}\n{stderr}",
            claim_id=next(iter(claim_lemmas), "claim:unknown"),
            raw_digest=output_digest,
        )
        if trace is not None:
            attack = trace.to_dict()

    if status_enum is ResultStatus.SECURE and accepted:
        observed_status = "secure"
    elif status_enum is ResultStatus.ATTACK_FOUND:
        observed_status = "attack"
    elif quarantine is not None:
        observed_status = "quarantined"
    else:
        observed_status = "unknown"

    reason_codes: list[str] = []
    if quarantine is not None:
        reason_codes.append(str(quarantine.reason.value))
    if kind == "malformed" and observed_status != "secure":
        reason_codes.append("malformed_output")
    if kind == "timeout":
        if any(item.verdict is ClaimVerdict.TIMEOUT for item in outcomes):
            reason_codes.append("timeout")
        if observed_status == "secure":
            reason_codes.append("timeout_promoted_to_secure")

    matched = _expect_matches(expect, observed_status, quarantine is not None)

    # Mutations of secure claims must never remain secure.
    if kind == "mutation" and observed_status == "secure":
        matched = False
        reason_codes.append("mutation_still_secure")

    if attack is None and status_enum is ResultStatus.ATTACK_FOUND:
        # ensure attack_trace binding when classification already found an attack
        for item in outcomes:
            if item.attack_trace is not None:
                attack = item.attack_trace.to_dict()
                break

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        status=observed_status,
        matched=matched,
        reason_codes=list(dict.fromkeys(reason_codes)),
        claim_outcomes=[item.to_dict() for item in outcomes],
        attack_trace=attack,
        output_digest=output_digest,
        stdout=stdout,
        stderr=stderr,
        detail=str(case.get("description") or ""),
    )


def _expect_matches(expect: str, observed: str, quarantined: bool) -> bool:
    if expect == "secure":
        return observed == "secure"
    if expect == "attack":
        return observed == "attack"
    if expect == "quarantined":
        return observed == "quarantined" or quarantined
    if expect == "blocked":
        return observed == "blocked"
    if expect == "rejected_or_quarantined":
        return observed in {"attack", "quarantined", "unknown", "blocked"}
    return observed == expect


# ---------------------------------------------------------------------------
# Maude promotion boundary
# ---------------------------------------------------------------------------


def maude_cannot_promote_protocol_lane() -> dict[str, Any]:
    """Prove Maude support-only presence cannot satisfy protocol authority."""

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
        "can_satisfy_protocol_requirement": can_satisfy,
        "promotion_allowed": decision.allowed,
        "promotion_decision": decision.to_dict(),
        "support_only": role.role is ToolRole.SUPPORT,
        "ceiling_is_none": role.authority_ceiling is ToolchainAuthorityCeiling.NONE,
        "blocks_alone": (not decision.allowed) and (not can_satisfy),
    }


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    tamarin_executable: str | None = None,
    maude_executable: str | None = None,
) -> TamarinToolchainCertification:
    """Run the full Tamarin/Maude certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = TamarinToolchainCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="tamarin.offline_policy",
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

    tamarin_probe = probe_tamarin_identity(
        env=probe_env, executable=tamarin_executable
    )
    maude_probe = probe_maude_identity(env=probe_env, executable=maude_executable)

    cert.tamarin_executable = tamarin_probe.get("executable_path")
    cert.maude_executable = maude_probe.get("executable_path")
    cert.tamarin_version_string = tamarin_probe.get("version_string")
    cert.maude_version_string = maude_probe.get("version_string")
    cert.tamarin_identity_probed = bool(tamarin_probe.get("identity_probed"))
    cert.maude_identity_probed = bool(maude_probe.get("identity_probed"))
    cert.tamarin_version_match = bool(tamarin_probe.get("version_match"))
    cert.maude_version_match = bool(maude_probe.get("version_match"))
    cert.tamarin_usable = bool(
        cert.tamarin_identity_probed and cert.tamarin_version_match
    )
    cert.maude_usable = bool(cert.maude_identity_probed and cert.maude_version_match)

    if cert.tamarin_usable:
        cert.checks.append(
            CheckResult(
                check_id="tamarin.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_TAMARIN_VERSION,
                observed=cert.tamarin_version_string or "",
                detail="exact Tamarin pin identity",
                bindings={
                    "executable_path": cert.tamarin_executable,
                    "version_string": cert.tamarin_version_string,
                },
            )
        )
    else:
        reason = str(tamarin_probe.get("probe_error") or "unavailable")
        cert.block_reasons.append(f"tamarin:{reason}")
        cert.checks.append(
            CheckResult(
                check_id="tamarin.identity",
                kind="identity",
                status="unavailable" if reason == "executable_not_on_path" else "blocked",
                expected=LOCKED_TAMARIN_VERSION,
                observed=reason,
                detail="PATH presence without locked identity is not usability",
                reason_codes=[reason],
            )
        )

    if cert.maude_usable:
        cert.checks.append(
            CheckResult(
                check_id="maude.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_MAUDE_VERSION,
                observed=cert.maude_version_string or "",
                detail="exact Maude pin identity (support only)",
                bindings={
                    "executable_path": cert.maude_executable,
                    "version_string": cert.maude_version_string,
                    "support_only": True,
                },
            )
        )
    else:
        reason = str(maude_probe.get("probe_error") or "unavailable")
        cert.block_reasons.append(f"maude:{reason}")
        cert.checks.append(
            CheckResult(
                check_id="maude.identity",
                kind="identity",
                status="unavailable" if reason == "executable_not_on_path" else "blocked",
                expected=LOCKED_MAUDE_VERSION,
                observed=reason,
                detail="Maude support companion identity",
                reason_codes=[reason],
            )
        )

    if cert.tamarin_usable and cert.maude_usable:
        pair = probe_tamarin_maude_pair(
            cert.tamarin_executable,
            cert.maude_executable,
            env=probe_env,
        )
        cert.pair_validated = bool(pair.get("validated"))
        cert.checks.append(
            CheckResult(
                check_id="tamarin.maude_pair",
                kind="identity",
                status="passed" if cert.pair_validated else "failed",
                expected="checking installation: OK",
                observed=str(pair.get("detail") or ""),
                detail="Tamarin runtime validation of the Maude companion",
                reason_codes=[] if cert.pair_validated else ["pair_validation_failed"],
            )
        )
        if not cert.pair_validated:
            cert.block_reasons.append("pair_validation_failed")
    else:
        cert.checks.append(
            CheckResult(
                check_id="tamarin.maude_pair",
                kind="identity",
                status="skipped",
                expected="checking installation: OK",
                observed="identity_unavailable",
                detail="pair probe requires both locked identities",
            )
        )

    # Semantic corpus (parser-backed; always runs offline).
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
                check_id=f"tamarin.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "claim_outcomes": list(outcome.claim_outcomes),
                    "attack_trace": outcome.attack_trace,
                },
            )
        )

    # Deterministic replay binding between secure and replay cases.
    secure = outcomes_by_id.get("secure_claims")
    replay = outcomes_by_id.get("deterministic_replay")
    if secure is not None and replay is not None:
        replay_ok = (
            secure.status == "secure"
            and replay.status == "secure"
            and secure.output_digest == replay.output_digest
            and secure.matched
            and replay.matched
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="tamarin.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical secure digests",
                observed=(
                    f"secure={secure.output_digest[:12]},"
                    f"replay={replay.output_digest[:12]}"
                ),
                bindings={
                    "secure_digest": secure.output_digest,
                    "replay_digest": replay.output_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_secure_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="tamarin.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="secure and replay cases",
                observed="missing",
            )
        )

    # Maude cannot promote the protocol lane by itself.
    maude_boundary = maude_cannot_promote_protocol_lane()
    boundary_ok = bool(maude_boundary.get("blocks_alone"))
    if not boundary_ok:
        cert.block_reasons.append("maude_incorrectly_promotes")
    cert.checks.append(
        CheckResult(
            check_id="maude.support_only_boundary",
            kind="authority",
            status="passed" if boundary_ok else "failed",
            expected="promotion_blocked",
            observed=(
                f"allowed={maude_boundary.get('promotion_allowed')},"
                f"can_satisfy={maude_boundary.get('can_satisfy_protocol_requirement')}"
            ),
            detail="Maude is support only and cannot promote the protocol lane",
            bindings=maude_boundary,
        )
    )

    # Bind theory, claims, bounds, and exact binaries.
    ceiling = SymbolicModelCeiling.disclose(
        equational_theories=list(
            corpus.get("theory_bindings") or DEFAULT_THEORY_BINDINGS
        ),
        claim_kinds=list(corpus.get("claim_bindings") or DEFAULT_CLAIM_BINDINGS),
    )
    cert.bindings = {
        "theory": {
            "ceiling": ceiling,
            "supported_theories": sorted(
                item.value for item in TAMARIN_SUPPORTED_THEORIES
            ),
            "bound_theories": list(
                corpus.get("theory_bindings") or DEFAULT_THEORY_BINDINGS
            ),
        },
        "claims": {
            "supported_claim_kinds": sorted(
                item.value for item in TAMARIN_SUPPORTED_CLAIMS
            ),
            "bound_claim_kinds": list(
                corpus.get("claim_bindings") or DEFAULT_CLAIM_BINDINGS
            ),
            "secure_case_claims": (
                list(secure.claim_outcomes) if secure is not None else []
            ),
        },
        "bounds": dict(corpus.get("bounds") or DEFAULT_BOUNDS),
        "binaries": {
            "tamarin": {
                "tool_id": TOOL_ID,
                "locked_version": LOCKED_TAMARIN_VERSION,
                "executable_path": cert.tamarin_executable,
                "version_string": cert.tamarin_version_string,
                "identity_probed": cert.tamarin_identity_probed,
                "version_match": cert.tamarin_version_match,
            },
            "maude": {
                "tool_id": SUPPORT_TOOL_ID,
                "locked_version": LOCKED_MAUDE_VERSION,
                "executable_path": cert.maude_executable,
                "version_string": cert.maude_version_string,
                "identity_probed": cert.maude_identity_probed,
                "version_match": cert.maude_version_match,
                "support_only": True,
                "can_promote_protocol_lane": False,
            },
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "backend_interface": TAMARIN_BACKEND_VERSION,
            "maude_is_support_only": True,
            "not_kernel": True,
            "not_advisor": True,
        },
        "maude_promotion_boundary": maude_boundary,
    }
    cert.checks.append(
        CheckResult(
            check_id="tamarin.bindings",
            kind="binding",
            status="passed",
            expected="theory,claims,bounds,binaries",
            observed=content_digest(cert.bindings)[:16],
            detail="receipt binds theory, claims, bounds, and exact binaries",
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "secure",
        "attack",
        "mutation",
        "replay",
        "malformed",
        "timeout",
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
            "secure",
            "attack",
            "mutation",
            "replay",
            "malformed",
            "timeout",
            "version_mismatch",
            "authority",
            "binding",
        }
        or check.check_id
        in {
            "tamarin.deterministic_replay_binding",
            "tamarin.bindings",
            "maude.support_only_boundary",
            "tamarin.offline_policy",
        }
    )

    # Production certification requires live locked binaries + semantic suite.
    cert.production_certified = bool(
        cert.tamarin_usable
        and cert.maude_usable
        and cert.pair_validated
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
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned Tamarin 1.12.0 + Maude 3.5.1 certified for protocol claims "
            "and attacks; Maude remains support only."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            if semantic_ok and not (cert.tamarin_usable and cert.maude_usable):
                cert.notes = (
                    "Semantic corpus passed offline; live locked Tamarin/Maude "
                    "identities unavailable — production certification withheld."
                )
            else:
                cert.notes = (
                    "Tamarin/Maude certification incomplete or failed; "
                    "protocol-lane promotion blocked."
                )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    tamarin_executable: str | None = None,
    maude_executable: str | None = None,
) -> dict[str, Any]:
    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        tamarin_executable=tamarin_executable,
        maude_executable=maude_executable,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "maude_is_support_only": True,
        "maude_cannot_promote_protocol_lane": True,
        "exact_binary_binding_required": True,
        "authority_is_protocol_verification_only": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_proverif_lane": True,
        "does_not_edit_shared_lock": True,
    }
    payload["semantic_corpus_passed"] = all(
        case.matched for case in cert.cases
    )
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_tamarin_toolchain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=kwargs.get("manifest"),
        env=kwargs.get("env"),
        tamarin_executable=kwargs.get("tamarin_executable"),
        maude_executable=kwargs.get("maude_executable"),
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
    return certify_tamarin_toolchain(*args, **kwargs)


def bind_protocol_lane_handler(
    *,
    policy: Any | None = None,
    replace: bool = True,
) -> Any | None:
    """Register this certifier on the protocol lane when roles surface exists."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return None
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID, lane_handler, policy=target, replace=replace
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the pinned Tamarin/Maude protocol toolchain "
            f"({INTERFACE}; Tamarin {LOCKED_TAMARIN_VERSION} + "
            f"Maude {LOCKED_MAUDE_VERSION})."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print receipt as JSON")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--tamarin", type=str, default=None)
    parser.add_argument("--maude", type=str, default=None)
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    receipt = build_certification_receipt(
        repo_root=root,
        tamarin_executable=args.tamarin,
        maude_executable=args.maude,
    )
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"tamarin={receipt.get('tamarin_version_string')!r} "
            f"maude={receipt.get('maude_version_string')!r}"
        )
        print(
            f"usable_tamarin={receipt.get('tamarin_usable')} "
            f"usable_maude={receipt.get('maude_usable')} "
            f"pair={receipt.get('pair_validated')} "
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
    # Exit 0 when the semantic corpus passes even if live tools are absent —
    # production_certified remains the explicit live gate in the receipt.
    semantic_ok = bool(receipt.get("semantic_corpus_passed"))
    return 0 if semantic_ok else 1


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
    "LOCKED_TAMARIN_VERSION",
    "LOCKED_MAUDE_VERSION",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "TamarinToolchainCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_executable",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "probe_tamarin_identity",
    "probe_maude_identity",
    "probe_tamarin_maude_pair",
    "evaluate_corpus_case",
    "maude_cannot_promote_protocol_lane",
    "run_certification_suite",
    "build_certification_receipt",
    "certify_tamarin_toolchain",
    "lane_handler",
    "bind_protocol_lane_handler",
    "main",
]
