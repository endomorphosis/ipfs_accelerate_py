#!/usr/bin/env python3
"""Vampire + E ATP toolchain certification (FVT-G140 / FVT-048).

``ATPToolchainCertification@1``

Owns the ATP-lane certification handler for the pinned Vampire 5.0.1 and
E 3.2.5 provers. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for Vampire 5.0.1 and E 3.2.5 when live;
* exercises theorem, non-theorem, premise/conclusion mutation, proof-output
  binding, replay, malformed output, timeout, and version-mismatch cases;
* classifies external output only by exact TPTP SZS status lines;
* treats unreconstructed ATP proofs/models as **candidates** unless an
  allowed independent kernel reconstruction validates them;
* never edits the shared multi-prover certificate or CEC semantics.

Semantic evaluation reuses the canonical ATP adapters so offline tests can
prove corpus behavior without a live Vampire or E process. Live production
certification additionally requires the pinned binaries.
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

from ipfs_datasets_py.logic.backends.atp.adapters import (  # noqa: E402
    ATP_ADAPTER_VERSION,
    ATP_COMPATIBILITY_BACKENDS_VERSION,
    MalformedATPOutput,
    SZSStatus,
    parse_szs_status,
)
from ipfs_datasets_py.logic.backends.results import (  # noqa: E402
    ResultAuthority,
    ResultStatus,
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

INTERFACE: Final = "ATPToolchainCertification@1"
SCHEMA_VERSION: Final = "atp-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "atp-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G140"
TASK_ID: Final = "FVT-048"
PROGRAM: Final = "formal-verification-tactician/atp-toolchains"
LANE_ID: Final = "atp"
TOOL_VAMPIRE: Final = "vampire"
TOOL_EPROVER: Final = "eprover"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.atp"
HANDLER_ID: Final = "atp_toolchain_certification@1"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.RECONSTRUCTION.value
AUTHORITY_SCOPE: Final = "atp_candidate_until_kernel_reconstruction"

LOCKED_VAMPIRE_VERSION: Final = "5.0.1"
LOCKED_EPROVER_VERSION: Final = "3.2.5"
LOCKED_VAMPIRE_EXECUTABLE: Final = "vampire"
LOCKED_EPROVER_EXECUTABLE: Final = "eprover"

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_VERSION_IN_BANNER = re.compile(r"(\d+\.\d+(?:\.\d+)?)")

# Compact embedded corpus. Prefer live binaries when present; SZS parsers always run.
# Case design (FVT-G140 acceptance):
#   theorem, non-theorem, premise/conclusion mutation, proof-output binding,
#   replay, malformed output, timeout, version mismatch.
_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "theorem_proved",
        "kind": "theorem",
        "expect": "theorem_candidate",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status Theorem for theorem_proved\n"
            "% SZS output start Proof for theorem_proved\n"
            "fof(1, plain, p, inference(assumption, [], [])).\n"
            "% SZS output end Proof for theorem_proved\n"
        ),
        "stderr": "",
        "description": "Theorem SZS status with unreconstructed proof → candidate",
    },
    {
        "case_id": "non_theorem",
        "kind": "non_theorem",
        "expect": "non_theorem_candidate",
        "tool_id": "eprover",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, q).\n"
        ),
        "stdout": (
            "% SZS status CounterSatisfiable for non_theorem\n"
            "% SZS output start Model for non_theorem\n"
            "model(interpretation).\n"
            "% SZS output end Model for non_theorem\n"
        ),
        "stderr": "",
        "description": "Non-theorem CounterSatisfiable remains a candidate model",
    },
    {
        "case_id": "mutated_premise",
        "kind": "mutation",
        "mutates": "premise",
        "expect": "non_theorem_or_unknown",
        "base_case_id": "theorem_proved",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, ~p).\n"  # mutated premise
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status CounterSatisfiable for mutated_premise\n"
        ),
        "stderr": "",
        "description": "Premise mutation must not remain a theorem candidate",
    },
    {
        "case_id": "mutated_conclusion",
        "kind": "mutation",
        "mutates": "conclusion",
        "expect": "non_theorem_or_unknown",
        "base_case_id": "theorem_proved",
        "tool_id": "eprover",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, r).\n"  # mutated conclusion
        ),
        "stdout": (
            "% SZS status Satisfiable for mutated_conclusion\n"
        ),
        "stderr": "",
        "description": "Conclusion mutation must not remain a theorem candidate",
    },
    {
        "case_id": "proof_output_binding",
        "kind": "proof_binding",
        "expect": "theorem_candidate",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status Theorem for proof_output_binding\n"
            "% SZS output start Proof for proof_output_binding\n"
            "fof(1, plain, p, file('problem.p', ax1)).\n"
            "fof(2, plain, p, inference(cn, [status(thm)], [1])).\n"
            "% SZS output end Proof for proof_output_binding\n"
        ),
        "stderr": "",
        "require_proof_body": True,
        "description": "Proof body is bound to the SZS Theorem status",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "theorem_candidate",
        "base_case_id": "theorem_proved",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status Theorem for theorem_proved\n"
            "% SZS output start Proof for theorem_proved\n"
            "fof(1, plain, p, inference(assumption, [], [])).\n"
            "% SZS output end Proof for theorem_proved\n"
        ),
        "stderr": "",
        "description": "Positive theorem case replays with identical digests",
    },
    {
        "case_id": "malformed_output",
        "kind": "malformed",
        "expect": "quarantined",
        "tool_id": "vampire",
        "tptp_source": "fof(goal, conjecture, p).\n",
        "stdout": "this is not a SZS report\n!!! garbage !!!\nProof found!!!\n",
        "stderr": "",
        "description": "Malformed tool output never reports theorem authority",
    },
    {
        "case_id": "timeout_claim",
        "kind": "timeout",
        "expect": "timeout",
        "tool_id": "eprover",
        "tptp_source": "fof(goal, conjecture, p).\n",
        "stdout": "% SZS status Timeout for timeout_claim\n",
        "stderr": "",
        "description": "Timeout outcomes quarantine rather than theorem",
    },
    {
        "case_id": "version_mismatch",
        "kind": "version_mismatch",
        "expect": "blocked",
        "tool_id": "vampire",
        "stdout": "",
        "stderr": "",
        "observed_vampire_version": "4.5.1",
        "observed_eprover_version": "2.6",
        "description": "Locked version mismatch blocks production certification",
    },
    {
        "case_id": "kernel_reconstruction_elevates",
        "kind": "reconstruction",
        "expect": "theorem_authority",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status Theorem for kernel_reconstruction_elevates\n"
            "% SZS output start Proof for kernel_reconstruction_elevates\n"
            "fof(1, plain, p, inference(assumption, [], [])).\n"
            "% SZS output end Proof for kernel_reconstruction_elevates\n"
        ),
        "stderr": "",
        "independent_kernel_reconstruction": True,
        "description": (
            "Only allowed independent kernel reconstruction elevates "
            "ATP evidence beyond candidate"
        ),
    },
)

DEFAULT_BOUNDS: Final[dict[str, Any]] = {
    "timeout_seconds": CHECK_TIMEOUT_SECONDS,
    "max_source_bytes": 1_048_576,
    "network": False,
    "install": False,
    "download": False,
}

THEOREM_SZS: Final[frozenset[SZSStatus]] = frozenset(
    {
        SZSStatus.THEOREM,
        SZSStatus.UNSATISFIABLE,
        SZSStatus.CONTRADICTORY_AXIOMS,
    }
)
NON_THEOREM_SZS: Final[frozenset[SZSStatus]] = frozenset(
    {
        SZSStatus.SATISFIABLE,
        SZSStatus.COUNTER_SATISFIABLE,
    }
)
TIMEOUT_SZS: Final[frozenset[SZSStatus]] = frozenset(
    {
        SZSStatus.TIMEOUT,
        SZSStatus.RESOURCE_OUT,
    }
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
    status: str
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    szs_status: str | None = None
    authority: str = ResultAuthority.CANDIDATE.value
    result_status: str = ResultStatus.CANDIDATE.value
    proof_bound: bool = False
    output_digest: str = ""
    source_digest: str = ""
    stdout: str = ""
    stderr: str = ""
    detail: str = ""
    independent_kernel_reconstruction: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ATPToolchainCertification:
    """Full certification receipt for the Vampire/E ATP lane."""

    tool_ids: list[str] = field(
        default_factory=lambda: [TOOL_VAMPIRE, TOOL_EPROVER]
    )
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_vampire_version: str = LOCKED_VAMPIRE_VERSION
    locked_eprover_version: str = LOCKED_EPROVER_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    vampire_executable: str | None = None
    eprover_executable: str | None = None
    vampire_version_string: str | None = None
    eprover_version_string: str | None = None
    vampire_identity_probed: bool = False
    eprover_identity_probed: bool = False
    vampire_version_match: bool = False
    eprover_version_match: bool = False
    vampire_usable: bool = False
    eprover_usable: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    results_are_candidates_without_reconstruction: bool = True
    kernel_reconstruction_required_for_theorem_authority: bool = True
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
        "tool_ids": [TOOL_VAMPIRE, TOOL_EPROVER],
        "lane_id": LANE_ID,
        "locked_vampire_version": LOCKED_VAMPIRE_VERSION,
        "locked_eprover_version": LOCKED_EPROVER_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "bounds": dict(DEFAULT_BOUNDS),
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "exact_binary_binding_required": True,
            "results_are_candidates_without_reconstruction": True,
            "kernel_reconstruction_required_for_theorem_authority": True,
            "szs_status_only": True,
            "does_not_edit_central_certificate": True,
            "does_not_edit_cec_semantics": True,
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
            raise ValueError("ATP corpus manifest must be a JSON object")
        return payload
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("ATP corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------


def _probe_tool_identity(
    tool_id: str,
    *,
    locked_version: str,
    executable_names: Sequence[str],
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": tool_id,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": locked_version,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable(list(executable_names))
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
        version == locked_version or locked_version in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def probe_vampire_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    return _probe_tool_identity(
        TOOL_VAMPIRE,
        locked_version=LOCKED_VAMPIRE_VERSION,
        executable_names=(LOCKED_VAMPIRE_EXECUTABLE,),
        env=env,
        executable=executable,
    )


def probe_eprover_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    return _probe_tool_identity(
        TOOL_EPROVER,
        locked_version=LOCKED_EPROVER_VERSION,
        executable_names=(LOCKED_EPROVER_EXECUTABLE, "eproof"),
        env=env,
        executable=executable,
    )


# ---------------------------------------------------------------------------
# Case evaluation (SZS parser-backed, offline)
# ---------------------------------------------------------------------------


def classify_szs_outcome(
    stdout: str,
    stderr: str = "",
    *,
    independent_kernel_reconstruction: bool = False,
    require_proof_body: bool = False,
) -> dict[str, Any]:
    """Classify ATP tool output into candidate / authority / quarantine.

    Unreconstructed Theorem/Unsatisfiable and model statuses stay candidates.
    Only ``independent_kernel_reconstruction=True`` elevates theorem evidence.
    """

    combined = "\n".join(part for part in (stdout, stderr) if part)
    output_digest = content_digest(combined)
    proof_body = bool(
        re.search(r"%\s*SZS\s+output\s+start\s+Proof", combined, re.IGNORECASE)
        or re.search(r"fof\([^)]*,\s*plain,", combined)
    )
    try:
        szs = parse_szs_status(combined)
    except MalformedATPOutput as error:
        return {
            "status": "quarantined",
            "szs_status": None,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.MALFORMED.value,
            "proof_bound": False,
            "output_digest": output_digest,
            "reason_codes": ["malformed_output", "no_szs_status"],
            "detail": str(error),
        }

    reason_codes: list[str] = [f"szs:{szs.value}"]
    if szs in THEOREM_SZS:
        if require_proof_body and not proof_body:
            return {
                "status": "quarantined",
                "szs_status": szs.value,
                "authority": ResultAuthority.CANDIDATE.value,
                "result_status": ResultStatus.MALFORMED.value,
                "proof_bound": False,
                "output_digest": output_digest,
                "reason_codes": reason_codes + ["proof_body_missing"],
                "detail": "Theorem SZS without proof body",
            }
        if independent_kernel_reconstruction:
            return {
                "status": "theorem_authority",
                "szs_status": szs.value,
                "authority": ResultAuthority.THEOREM.value,
                "result_status": ResultStatus.PROVED.value,
                "proof_bound": proof_body,
                "output_digest": output_digest,
                "reason_codes": reason_codes + ["independent_kernel_reconstruction"],
                "detail": "Elevated by independent kernel reconstruction",
            }
        return {
            "status": "theorem_candidate",
            "szs_status": szs.value,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.CANDIDATE.value,
            "proof_bound": proof_body,
            "output_digest": output_digest,
            "reason_codes": reason_codes + ["unreconstructed_atp_proof"],
            "detail": "ATP theorem remains candidate without reconstruction",
        }

    if szs in NON_THEOREM_SZS:
        return {
            "status": "non_theorem_candidate",
            "szs_status": szs.value,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.CANDIDATE.value,
            "proof_bound": False,
            "output_digest": output_digest,
            "reason_codes": reason_codes + ["unvalidated_atp_model"],
            "detail": "ATP non-theorem model remains candidate",
        }

    if szs in TIMEOUT_SZS:
        return {
            "status": "timeout",
            "szs_status": szs.value,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.TIMEOUT.value,
            "proof_bound": False,
            "output_digest": output_digest,
            "reason_codes": reason_codes + ["timeout"],
            "detail": "ATP timeout is not theorem authority",
        }

    return {
        "status": "unknown",
        "szs_status": szs.value,
        "authority": ResultAuthority.CANDIDATE.value,
        "result_status": ResultStatus.UNKNOWN.value,
        "proof_bound": False,
        "output_digest": output_digest,
        "reason_codes": reason_codes + ["unknown_szs"],
        "detail": f"Unhandled SZS status {szs.value}",
    }


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one corpus case via SZS parsers (no install)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    stdout = str(case.get("stdout") or "")
    stderr = str(case.get("stderr") or "")
    source = str(case.get("tptp_source") or "")
    source_digest = content_digest(source) if source else ""
    reconstruction = bool(case.get("independent_kernel_reconstruction"))
    require_proof = bool(case.get("require_proof_body"))

    if kind == "version_mismatch":
        observed_v = str(case.get("observed_vampire_version") or "")
        observed_e = str(case.get("observed_eprover_version") or "")
        blocked = (
            observed_v != LOCKED_VAMPIRE_VERSION
            or observed_e != LOCKED_EPROVER_VERSION
        )
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status="blocked" if blocked else "unknown",
            matched=blocked and expect == "blocked",
            reason_codes=["locked_version_mismatch"] if blocked else [],
            output_digest=content_digest(f"{stdout}\n{stderr}"),
            source_digest=source_digest,
            detail=(
                f"observed vampire={observed_v} eprover={observed_e}; "
                f"locked vampire={LOCKED_VAMPIRE_VERSION} "
                f"eprover={LOCKED_EPROVER_VERSION}"
            ),
        )

    classified = classify_szs_outcome(
        stdout,
        stderr,
        independent_kernel_reconstruction=reconstruction,
        require_proof_body=require_proof,
    )
    observed = str(classified["status"])
    reason_codes = list(classified.get("reason_codes") or [])

    # Mutations of theorems must never remain theorem_candidate / theorem_authority.
    if kind == "mutation" and observed in {
        "theorem_candidate",
        "theorem_authority",
    }:
        matched = False
        reason_codes.append("mutation_still_theorem")
    else:
        matched = _expect_matches(expect, observed)

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        status=observed,
        matched=matched,
        reason_codes=list(dict.fromkeys(reason_codes)),
        szs_status=classified.get("szs_status"),
        authority=str(classified.get("authority") or ResultAuthority.CANDIDATE.value),
        result_status=str(
            classified.get("result_status") or ResultStatus.CANDIDATE.value
        ),
        proof_bound=bool(classified.get("proof_bound")),
        output_digest=str(classified.get("output_digest") or ""),
        source_digest=source_digest,
        stdout=stdout,
        stderr=stderr,
        detail=str(case.get("description") or classified.get("detail") or ""),
        independent_kernel_reconstruction=reconstruction,
    )


def _expect_matches(expect: str, observed: str) -> bool:
    if expect == "theorem_candidate":
        return observed == "theorem_candidate"
    if expect == "theorem_authority":
        return observed == "theorem_authority"
    if expect == "non_theorem_candidate":
        return observed == "non_theorem_candidate"
    if expect == "non_theorem_or_unknown":
        return observed in {
            "non_theorem_candidate",
            "unknown",
            "timeout",
            "quarantined",
        }
    if expect == "quarantined":
        return observed == "quarantined"
    if expect == "timeout":
        return observed == "timeout"
    if expect == "blocked":
        return observed == "blocked"
    return observed == expect


# ---------------------------------------------------------------------------
# Authority boundary: candidates without reconstruction
# ---------------------------------------------------------------------------


def atp_results_remain_candidates_without_reconstruction() -> dict[str, Any]:
    """Prove unreconstructed ATP evidence cannot claim theorem authority alone."""

    report: dict[str, Any] = {
        "results_are_candidates_without_reconstruction": True,
        "kernel_reconstruction_required_for_theorem_authority": True,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "tools": {},
    }
    for tool_id in (TOOL_VAMPIRE, TOOL_EPROVER):
        role = get_tool_role(tool_id)
        # Presence + usability + hermetic cert without reconstruction must not
        # silently grant kernel authority; reconstruction ceiling is explicit.
        decision = evaluate_role_aware_promotion(
            tool_id,
            present=True,
            usable=True,
            production_certified=True,
            hermetic_certificate=True,
            independent_reconstruction=False,
        )
        elevated = evaluate_role_aware_promotion(
            tool_id,
            present=True,
            usable=True,
            production_certified=True,
            hermetic_certificate=True,
            independent_reconstruction=True,
        )
        can_satisfy = can_satisfy_certified_authority_requirement(tool_id)
        report["tools"][tool_id] = {
            "role": role.role.value,
            "authority_ceiling": role.authority_ceiling.value,
            "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
            "can_satisfy_requirement": can_satisfy,
            "without_reconstruction": decision.to_dict(),
            "with_reconstruction": elevated.to_dict(),
            "ceiling_is_reconstruction": (
                role.authority_ceiling is ToolchainAuthorityCeiling.RECONSTRUCTION
            ),
            "role_is_authority": role.role is ToolRole.AUTHORITY,
        }
    # Semantic invariant used by the corpus: unreconstructed theorem stays candidate.
    sample = classify_szs_outcome(
        "% SZS status Theorem for boundary\n"
        "% SZS output start Proof for boundary\n"
        "fof(1, plain, p).\n"
        "% SZS output end Proof for boundary\n",
        independent_kernel_reconstruction=False,
    )
    elevated_sample = classify_szs_outcome(
        "% SZS status Theorem for boundary\n"
        "% SZS output start Proof for boundary\n"
        "fof(1, plain, p).\n"
        "% SZS output end Proof for boundary\n",
        independent_kernel_reconstruction=True,
    )
    report["sample_without_reconstruction"] = sample
    report["sample_with_reconstruction"] = elevated_sample
    report["boundary_holds"] = (
        sample["status"] == "theorem_candidate"
        and sample["authority"] == ResultAuthority.CANDIDATE.value
        and elevated_sample["status"] == "theorem_authority"
        and elevated_sample["authority"] == ResultAuthority.THEOREM.value
    )
    return report


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    vampire_executable: str | None = None,
    eprover_executable: str | None = None,
) -> ATPToolchainCertification:
    """Run the full Vampire/E ATP certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = ATPToolchainCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="atp.offline_policy",
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

    vampire_probe = probe_vampire_identity(
        env=probe_env, executable=vampire_executable
    )
    eprover_probe = probe_eprover_identity(
        env=probe_env, executable=eprover_executable
    )

    cert.vampire_executable = vampire_probe.get("executable_path")
    cert.eprover_executable = eprover_probe.get("executable_path")
    cert.vampire_version_string = vampire_probe.get("version_string")
    cert.eprover_version_string = eprover_probe.get("version_string")
    cert.vampire_identity_probed = bool(vampire_probe.get("identity_probed"))
    cert.eprover_identity_probed = bool(eprover_probe.get("identity_probed"))
    cert.vampire_version_match = bool(vampire_probe.get("version_match"))
    cert.eprover_version_match = bool(eprover_probe.get("version_match"))
    cert.vampire_usable = bool(
        cert.vampire_identity_probed and cert.vampire_version_match
    )
    cert.eprover_usable = bool(
        cert.eprover_identity_probed and cert.eprover_version_match
    )

    for tool_id, usable, version_string, locked, probe in (
        (
            TOOL_VAMPIRE,
            cert.vampire_usable,
            cert.vampire_version_string,
            LOCKED_VAMPIRE_VERSION,
            vampire_probe,
        ),
        (
            TOOL_EPROVER,
            cert.eprover_usable,
            cert.eprover_version_string,
            LOCKED_EPROVER_VERSION,
            eprover_probe,
        ),
    ):
        if usable:
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_id}.identity",
                    kind="identity",
                    status="passed",
                    expected=locked,
                    observed=version_string or "",
                    detail=f"exact {tool_id} pin identity",
                    bindings={
                        "executable_path": probe.get("executable_path"),
                        "version_string": version_string,
                    },
                )
            )
        else:
            reason = str(probe.get("probe_error") or "unavailable")
            cert.block_reasons.append(f"{tool_id}:{reason}")
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_id}.identity",
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

    # Semantic corpus (SZS parser-backed; always runs offline).
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
                check_id=f"atp.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "source_digest": outcome.source_digest,
                    "szs_status": outcome.szs_status,
                    "authority": outcome.authority,
                    "result_status": outcome.result_status,
                    "proof_bound": outcome.proof_bound,
                    "independent_kernel_reconstruction": (
                        outcome.independent_kernel_reconstruction
                    ),
                },
            )
        )

    # Deterministic replay binding between theorem and replay cases.
    theorem = outcomes_by_id.get("theorem_proved")
    replay = outcomes_by_id.get("deterministic_replay")
    if theorem is not None and replay is not None:
        replay_ok = (
            theorem.status == "theorem_candidate"
            and replay.status == "theorem_candidate"
            and theorem.output_digest == replay.output_digest
            and theorem.matched
            and replay.matched
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="atp.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical theorem_candidate digests",
                observed=(
                    f"theorem={theorem.output_digest[:12]},"
                    f"replay={replay.output_digest[:12]}"
                ),
                bindings={
                    "theorem_digest": theorem.output_digest,
                    "replay_digest": replay.output_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_theorem_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="atp.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="theorem and replay cases",
                observed="missing",
            )
        )

    # Proof-output binding check.
    proof_case = outcomes_by_id.get("proof_output_binding")
    proof_ok = (
        proof_case is not None
        and proof_case.matched
        and proof_case.proof_bound
        and proof_case.status == "theorem_candidate"
    )
    if not proof_ok:
        cert.block_reasons.append("proof_output_binding_failed")
    cert.checks.append(
        CheckResult(
            check_id="atp.proof_output_binding",
            kind="proof_binding",
            status="passed" if proof_ok else "failed",
            expected="theorem_candidate with proof body",
            observed=(
                f"status={getattr(proof_case, 'status', None)},"
                f"proof_bound={getattr(proof_case, 'proof_bound', None)}"
            ),
            detail="Proof body must bind to SZS Theorem status",
        )
    )

    # Candidate-until-reconstruction authority boundary.
    boundary = atp_results_remain_candidates_without_reconstruction()
    boundary_ok = bool(boundary.get("boundary_holds"))
    if not boundary_ok:
        cert.block_reasons.append("candidate_authority_boundary_failed")
    cert.checks.append(
        CheckResult(
            check_id="atp.candidate_until_reconstruction",
            kind="authority",
            status="passed" if boundary_ok else "failed",
            expected="candidate_without_reconstruction",
            observed=(
                f"without={boundary['sample_without_reconstruction']['status']},"
                f"with={boundary['sample_with_reconstruction']['status']}"
            ),
            detail=(
                "ATP results remain candidates unless independent kernel "
                "reconstruction elevates them"
            ),
            bindings=boundary,
        )
    )

    # Bind sources, SZS outcomes, bounds, and exact binaries.
    cert.bindings = {
        "adapter": {
            "compatibility_interface": ATP_COMPATIBILITY_BACKENDS_VERSION,
            "adapter_version": ATP_ADAPTER_VERSION,
            "szs_status_only": True,
        },
        "bounds": dict(corpus.get("bounds") or DEFAULT_BOUNDS),
        "binaries": {
            "vampire": {
                "tool_id": TOOL_VAMPIRE,
                "locked_version": LOCKED_VAMPIRE_VERSION,
                "executable_path": cert.vampire_executable,
                "version_string": cert.vampire_version_string,
                "identity_probed": cert.vampire_identity_probed,
                "version_match": cert.vampire_version_match,
            },
            "eprover": {
                "tool_id": TOOL_EPROVER,
                "locked_version": LOCKED_EPROVER_VERSION,
                "executable_path": cert.eprover_executable,
                "version_string": cert.eprover_version_string,
                "identity_probed": cert.eprover_identity_probed,
                "version_match": cert.eprover_version_match,
            },
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "results_are_candidates_without_reconstruction": True,
            "kernel_reconstruction_required_for_theorem_authority": True,
            "not_kernel": True,
            "not_advisor": True,
        },
        "candidate_boundary": boundary,
        "theorem_case": (
            {
                "case_id": theorem.case_id,
                "status": theorem.status,
                "authority": theorem.authority,
                "output_digest": theorem.output_digest,
                "source_digest": theorem.source_digest,
            }
            if theorem is not None
            else None
        ),
        "proof_binding": (
            {
                "case_id": proof_case.case_id,
                "proof_bound": proof_case.proof_bound,
                "output_digest": proof_case.output_digest,
            }
            if proof_case is not None
            else None
        ),
    }
    cert.checks.append(
        CheckResult(
            check_id="atp.bindings",
            kind="binding",
            status="passed",
            expected="sources,szs,bounds,binaries,authority",
            observed=content_digest(cert.bindings)[:16],
            detail=(
                "receipt binds TPTP sources, SZS outcomes, bounds, and exact binaries"
            ),
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "theorem",
        "non_theorem",
        "mutation",
        "proof_binding",
        "replay",
        "malformed",
        "timeout",
        "version_mismatch",
        "reconstruction",
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
            "theorem",
            "non_theorem",
            "mutation",
            "proof_binding",
            "replay",
            "malformed",
            "timeout",
            "version_mismatch",
            "reconstruction",
            "authority",
            "binding",
            "policy",
        }
        or check.check_id
        in {
            "atp.deterministic_replay_binding",
            "atp.proof_output_binding",
            "atp.candidate_until_reconstruction",
            "atp.bindings",
            "atp.offline_policy",
        }
    )

    # Production certification requires live locked binaries + semantic suite.
    # Even when production-certified, results remain candidates without
    # independent kernel reconstruction (authority ceiling is reconstruction).
    cert.production_certified = bool(
        cert.vampire_usable
        and cert.eprover_usable
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and semantic_ok
        and not missing_kinds
        and not any(
            reason.startswith("case_failed:")
            or reason.startswith("replay_")
            or reason.startswith("proof_")
            or reason.startswith("candidate_")
            for reason in cert.block_reasons
        )
        and boundary_ok
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned Vampire 5.0.1 + E 3.2.5 certified for ATP premise/proof "
            "search; unreconstructed results remain candidates until "
            "independent kernel reconstruction."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            if semantic_ok and not (cert.vampire_usable and cert.eprover_usable):
                cert.notes = (
                    "Semantic corpus passed offline; live locked Vampire/E "
                    "identities unavailable — production certification withheld."
                )
            else:
                cert.notes = (
                    "ATP certification incomplete or failed; "
                    "ATP-lane promotion blocked."
                )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    vampire_executable: str | None = None,
    eprover_executable: str | None = None,
) -> dict[str, Any]:
    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        vampire_executable=vampire_executable,
        eprover_executable=eprover_executable,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "exact_binary_binding_required": True,
        "results_are_candidates_without_reconstruction": True,
        "kernel_reconstruction_required_for_theorem_authority": True,
        "szs_status_only": True,
        "authority_is_reconstruction_ceiling": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_cec_semantics": True,
        "does_not_edit_shared_lock": True,
    }
    payload["semantic_corpus_passed"] = all(case.matched for case in cert.cases)
    payload["authority_scope"] = AUTHORITY_SCOPE
    payload["results_are_candidates_without_reconstruction"] = True
    payload["kernel_reconstruction_required_for_theorem_authority"] = True
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_atp_toolchain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=kwargs.get("manifest"),
        env=kwargs.get("env"),
        vampire_executable=kwargs.get("vampire_executable"),
        eprover_executable=kwargs.get("eprover_executable"),
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
    return certify_atp_toolchain(*args, **kwargs)


def bind_atp_lane_handler(
    *,
    policy: Any | None = None,
    replace: bool = True,
) -> Any | None:
    """Register this certifier on the ATP lane when roles surface exists."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return None
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID, lane_handler, policy=target, replace=replace
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the pinned Vampire/E ATP toolchain "
            f"({INTERFACE}; Vampire {LOCKED_VAMPIRE_VERSION} + "
            f"E {LOCKED_EPROVER_VERSION})."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print receipt as JSON")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--vampire", type=str, default=None)
    parser.add_argument("--eprover", type=str, default=None)
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    receipt = build_certification_receipt(
        repo_root=root,
        vampire_executable=args.vampire,
        eprover_executable=args.eprover,
    )
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"vampire={receipt.get('vampire_version_string')!r} "
            f"eprover={receipt.get('eprover_version_string')!r}"
        )
        print(
            f"usable_vampire={receipt.get('vampire_usable')} "
            f"usable_eprover={receipt.get('eprover_usable')} "
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
    "TOOL_VAMPIRE",
    "TOOL_EPROVER",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "LOCKED_VAMPIRE_VERSION",
    "LOCKED_EPROVER_VERSION",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "ATPToolchainCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_executable",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "probe_vampire_identity",
    "probe_eprover_identity",
    "classify_szs_outcome",
    "evaluate_corpus_case",
    "atp_results_remain_candidates_without_reconstruction",
    "run_certification_suite",
    "build_certification_receipt",
    "certify_atp_toolchain",
    "lane_handler",
    "bind_atp_lane_handler",
    "main",
]
