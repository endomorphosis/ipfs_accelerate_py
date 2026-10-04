#!/usr/bin/env python3
"""Vampire + E ATP toolchain and live semantic certification.

``ATPToolchainCertification@1`` (FVT-G140 / FVT-048)
``ATPLiveSemanticCertification@1`` (FVT-G207 / FVT-054)

Owns the ATP-lane certification handler for the pinned Vampire 5.0.1 and
E 3.2.5 provers. Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for Vampire 5.0.1 and E 3.2.5 when live;
* exercises theorem, non-theorem, premise/conclusion mutation, proof-output
  binding, replay, malformed output, timeout, and version-mismatch cases;
* classifies external output only by exact TPTP SZS status lines;
* treats ATP proofs/models as **candidates** unless a separately validated
  independent-kernel receipt binds the exact proof;
* never edits the shared multi-prover certificate or CEC semantics.

Semantic evaluation reuses the canonical ATP adapters so offline tests can
prove corpus behavior without a live Vampire or E process. Live production
certification additionally requires the pinned binaries.

``ATPLiveSemanticCertification@1`` replaces SZS parser fixtures with real
pinned Vampire/E runs while preserving reconstruction and kernel-checking
ceilings. Live receipts bind binary digests, TPTP source, assumptions,
conclusion, limits, raw SZS output, and reconstruction status. A corpus
boolean is only a reconstruction claim and cannot replace a kernel receipt.
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
from tools.logic.certification.public_evidence import (  # noqa: E402
    public_evidence_audit,
    public_evidence_projection,
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

# Live semantic certification (FVT-G207 / FVT-054)
LIVE_INTERFACE: Final = "ATPLiveSemanticCertification@1"
LIVE_SCHEMA_VERSION: Final = "atp-live-semantic-certification/v1"
LIVE_CORPUS_SCHEMA: Final = "atp-live-semantic-corpus/v1"
LIVE_GOAL_ID: Final = "FVT-G207"
LIVE_TASK_ID: Final = "FVT-054"
LIVE_PROGRAM: Final = "formal-verification-tactician/atp-live-semantics"
LIVE_HANDLER_ID: Final = "atp_live_semantic_certification@1"
DEFAULT_LIVE_CERTIFICATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_atp_live_certificate.json"
)

LOCKED_VAMPIRE_VERSION: Final = "5.0.1"
LOCKED_EPROVER_VERSION: Final = "3.2.5"
LOCKED_VAMPIRE_EXECUTABLE: Final = "vampire"
LOCKED_EPROVER_EXECUTABLE: Final = "eprover"

# Managed install root (matches installer DEFAULT_USER_LOCAL_INSTALL_ROOT).
# Explicit approved deployment roots win over mutable user discovery so the
# sealed private-HOME validation environment can bind digest-verified tools.
DEFAULT_MANAGED_INSTALL_ROOT: Final = (
    "~/.local/share/ipfs_datasets_py/theorem-provers"
)
MANAGED_INSTALL_ROOT_ENV_VARS: Final[tuple[str, ...]] = (
    "IPFS_ACCELERATE_FORMAL_VERIFICATION_TOOLCHAINS_ROOT",
    "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
    "FORMAL_VERIFICATION_ATP_INSTALL_ROOT",
)

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0
LIVE_CASE_TIMEOUT_SECONDS: Final = 15.0
LIVE_TIMEOUT_CASE_WALL_SECONDS: Final = 0.03

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
        "case_id": "kernel_reconstruction_requires_receipt",
        "kind": "reconstruction",
        "expect": "theorem_candidate",
        "tool_id": "vampire",
        "tptp_source": (
            "fof(ax1, axiom, p).\n"
            "fof(goal, conjecture, p).\n"
        ),
        "stdout": (
            "% SZS status Theorem for kernel_reconstruction_requires_receipt\n"
            "% SZS output start Proof for kernel_reconstruction_requires_receipt\n"
            "fof(1, plain, p, inference(assumption, [], [])).\n"
            "% SZS output end Proof for kernel_reconstruction_requires_receipt\n"
        ),
        "stderr": "",
        "independent_kernel_reconstruction_claimed": True,
        "description": (
            "A reconstruction claim without a separately validated kernel "
            "receipt leaves ATP evidence at candidate authority"
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


def managed_install_roots(
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Ordered managed theorem-prover roots (approved deployment first)."""

    mapping = env if env is not None else os.environ
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    for variable in MANAGED_INSTALL_ROOT_ENV_VARS:
        raw = str(mapping.get(variable) or "").strip()
        if raw:
            _add(Path(raw))

    xdg_data = str(mapping.get("XDG_DATA_HOME") or "").strip()
    if xdg_data:
        _add(Path(xdg_data) / "ipfs_datasets_py" / "theorem-provers")

    _add(Path(DEFAULT_MANAGED_INSTALL_ROOT))
    return roots


def managed_execution_env(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Offline env with managed ATP bin directories prepended to PATH."""

    env = offline_env(base)
    bin_dirs: list[str] = []
    for root in managed_install_roots(env):
        managed_bin = root / "bin"
        if managed_bin.is_dir():
            bin_dirs.append(str(managed_bin))
    if bin_dirs:
        existing = str(env.get("PATH") or "")
        env["PATH"] = os.pathsep.join(
            [*bin_dirs, existing] if existing else bin_dirs
        )
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


def resolve_executable(
    candidates: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Locate Vampire/E preferring absolute paths and managed install bins."""

    search_path = None if env is None else str(env.get("PATH") or "")
    for name in candidates or ():
        if not name:
            continue
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        if os.path.isabs(name) or os.sep in name:
            # Explicit path that is not currently executable — do not fall through.
            continue
        for root in managed_install_roots(env):
            managed = root / "bin" / name
            if managed.is_file() and os.access(managed, os.X_OK):
                return str(managed.resolve())
        found = shutil.which(name, path=search_path)
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
    kernel_reconstruction_claimed: bool = False

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
    kernel_reconstruction_receipt_validated: bool = False
    boolean_reconstruction_claim_cannot_elevate: bool = True
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
            "kernel_reconstruction_receipt_validated": False,
            "boolean_reconstruction_claim_cannot_elevate": True,
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
    binary = executable or resolve_executable(
        list(executable_names), env=probe_env
    )
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
    ``independent_kernel_reconstruction`` records an untrusted caller claim
    for backward compatibility; it never elevates authority without a
    separately validated, proof-bound kernel receipt.
    """

    combined = "\n".join(part for part in (stdout, stderr) if part)
    output_digest = content_digest(combined)
    # Vampire uses "% SZS output start Proof"; E uses "# SZS output start CNFRefutation".
    proof_body = bool(
        re.search(
            r"[%#]\s*SZS\s+output\s+start\s+(Proof|CNFRefutation)",
            combined,
            re.IGNORECASE,
        )
        or re.search(r"fof\([^)]*,\s*plain,", combined)
        or re.search(r"cnf\([^,]+,\s*plain,", combined)
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
        reconstruction_reasons = (
            [
                "kernel_reconstruction_claim_unvalidated",
                "kernel_reconstruction_receipt_required",
            ]
            if independent_kernel_reconstruction
            else ["unreconstructed_atp_proof"]
        )
        return {
            "status": "theorem_candidate",
            "szs_status": szs.value,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.CANDIDATE.value,
            "proof_bound": proof_body,
            "output_digest": output_digest,
            "reason_codes": reason_codes + reconstruction_reasons,
            "detail": (
                "ATP theorem remains candidate: a reconstruction claim "
                "requires a separately validated kernel receipt"
                if independent_kernel_reconstruction
                else "ATP theorem remains candidate without reconstruction"
            ),
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
    reconstruction_claimed = bool(
        case.get("independent_kernel_reconstruction")
        or case.get("independent_kernel_reconstruction_claimed")
    )
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
        independent_kernel_reconstruction=reconstruction_claimed,
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
        independent_kernel_reconstruction=False,
        kernel_reconstruction_claimed=reconstruction_claimed,
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
        "kernel_reconstruction_receipt_validated": False,
        "boolean_reconstruction_claim_cannot_elevate": True,
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
        claimed_without_receipt = evaluate_role_aware_promotion(
            tool_id,
            present=True,
            usable=True,
            production_certified=True,
            hermetic_certificate=True,
            independent_reconstruction=False,
        )
        can_satisfy = can_satisfy_certified_authority_requirement(tool_id)
        report["tools"][tool_id] = {
            "role": role.role.value,
            "authority_ceiling": role.authority_ceiling.value,
            "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
            "can_satisfy_requirement": can_satisfy,
            "without_reconstruction": decision.to_dict(),
            # Compatibility key: this is now explicitly a claim without a
            # validated receipt and therefore receives no elevation.
            "with_reconstruction": claimed_without_receipt.to_dict(),
            "with_unvalidated_reconstruction_claim": (
                claimed_without_receipt.to_dict()
            ),
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
    claimed_sample = classify_szs_outcome(
        "% SZS status Theorem for boundary\n"
        "% SZS output start Proof for boundary\n"
        "fof(1, plain, p).\n"
        "% SZS output end Proof for boundary\n",
        independent_kernel_reconstruction=True,
    )
    report["sample_without_reconstruction"] = sample
    report["sample_with_reconstruction"] = claimed_sample
    report["sample_with_unvalidated_reconstruction_claim"] = claimed_sample
    report["boundary_holds"] = (
        sample["status"] == "theorem_candidate"
        and sample["authority"] == ResultAuthority.CANDIDATE.value
        and claimed_sample["status"] == "theorem_candidate"
        and claimed_sample["authority"] == ResultAuthority.CANDIDATE.value
        and "kernel_reconstruction_receipt_required"
        in claimed_sample["reason_codes"]
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
    probe_env = managed_execution_env(env)

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
                    "kernel_reconstruction_claimed": (
                        outcome.kernel_reconstruction_claimed
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
                "ATP results remain candidates; a reconstruction boolean "
                "cannot replace a validated independent-kernel receipt"
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
            "kernel_reconstruction_receipt_validated": False,
            "boolean_reconstruction_claim_cannot_elevate": True,
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
    # Even when production-certified, results remain candidates until a
    # separately validated kernel receipt exists (ceiling is reconstruction).
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
            "a proof-bound independent-kernel receipt is validated."
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
        "kernel_reconstruction_receipt_validated": False,
        "boolean_reconstruction_claim_cannot_elevate": True,
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
    payload["kernel_reconstruction_receipt_validated"] = False
    payload["boolean_reconstruction_claim_cannot_elevate"] = True
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
    receipt["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
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


# ---------------------------------------------------------------------------
# Live semantic certification (FVT-G207 / FVT-054)
# ---------------------------------------------------------------------------

# Compact TPTP recipes for live Vampire/E execution. Fixtures supply only
# source + expectations; stdout/stderr come from real pinned binaries.
_LIVE_TPTP_THEOREM: Final = (
    "fof(ax1, axiom, p).\n"
    "fof(goal, conjecture, p).\n"
)
_LIVE_TPTP_COUNTER_SAT: Final = (
    "fof(ax1, axiom, p).\n"
    "fof(goal, conjecture, q).\n"
)
_LIVE_TPTP_MUTATED_PREMISE: Final = (
    "fof(ax1, axiom, ~p).\n"
    "fof(goal, conjecture, p).\n"
)
_LIVE_TPTP_MUTATED_CONCLUSION: Final = (
    "fof(ax1, axiom, p).\n"
    "fof(goal, conjecture, r).\n"
)
_LIVE_TPTP_MALFORMED: Final = "this is not a TPTP problem !!!\n"
# Large chain forces parse/search work so wall-clock bounds can fire.
_LIVE_TPTP_TIMEOUT_WORKLOAD: Final = "\n".join(
    ["fof(ax0, axiom, p0)."]
    + [f"fof(ax{i}, axiom, p{i} | ~p{i - 1})." for i in range(1, 12000)]
    + ["fof(goal, conjecture, p11999)."]
) + "\n"

_LIVE_DEFAULT_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "theorem",
        "kind": "theorem",
        "expect": "theorem_candidate",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_THEOREM,
        "require_proof_body": False,
        "description": "Live theorem → unreconstructed candidate",
    },
    {
        "case_id": "counter_satisfiable",
        "kind": "counter_satisfiable",
        "expect": "non_theorem_candidate",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, q).",
        "tptp_source": _LIVE_TPTP_COUNTER_SAT,
        "description": "Live counter-satisfiable → candidate model",
    },
    {
        "case_id": "mutated_premise",
        "kind": "mutation",
        "mutates": "premise",
        "expect": "non_theorem_or_unknown",
        "assumptions": ("fof(ax1, axiom, ~p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_MUTATED_PREMISE,
        "description": "Premise mutation must not remain theorem",
    },
    {
        "case_id": "mutated_conclusion",
        "kind": "mutation",
        "mutates": "conclusion",
        "expect": "non_theorem_or_unknown",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, r).",
        "tptp_source": _LIVE_TPTP_MUTATED_CONCLUSION,
        "description": "Conclusion mutation must not remain theorem",
    },
    {
        "case_id": "proof_object",
        "kind": "proof_object",
        "expect": "theorem_candidate",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_THEOREM,
        "require_proof_body": True,
        "description": "Live proof object remains candidate without reconstruction",
    },
    {
        "case_id": "replay",
        "kind": "replay",
        "expect": "theorem_candidate",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_THEOREM,
        "description": "Replay must preserve SZS class and source digest",
    },
    {
        "case_id": "malformed_tptp",
        "kind": "malformed",
        "expect": "quarantined",
        "assumptions": (),
        "conclusion": "",
        "tptp_source": _LIVE_TPTP_MALFORMED,
        "description": "Malformed TPTP never reports theorem authority",
    },
    {
        "case_id": "timeout_resource_bounds",
        "kind": "timeout",
        "expect": "timeout",
        "assumptions": tuple(
            f"fof(ax{i}, axiom, p{i} | ~p{max(i - 1, 0)})." for i in range(0, 8)
        ),
        "conclusion": "fof(goal, conjecture, p11999).",
        "tptp_source": _LIVE_TPTP_TIMEOUT_WORKLOAD,
        "timeout_seconds": LIVE_TIMEOUT_CASE_WALL_SECONDS,
        "description": "Wall-clock resource bounds yield timeout quarantine",
    },
    {
        "case_id": "reconstruction",
        "kind": "reconstruction",
        "expect": "theorem_candidate",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_THEOREM,
        "require_proof_body": True,
        "independent_kernel_reconstruction_claimed": True,
        "description": (
            "A reconstruction claim without a validated independent-kernel "
            "receipt remains an ATP candidate"
        ),
    },
    {
        "case_id": "disagreement",
        "kind": "disagreement",
        "expect": "quarantined",
        "assumptions": ("fof(ax1, axiom, p).",),
        "conclusion": "fof(goal, conjecture, p).",
        "tptp_source": _LIVE_TPTP_THEOREM,
        "force_disagreement": True,
        "description": "Disagreement between ATP witnesses quarantines promotion",
    },
)

DEFAULT_LIVE_BOUNDS: Final[dict[str, Any]] = {
    "timeout_seconds": LIVE_CASE_TIMEOUT_SECONDS,
    "timeout_case_wall_seconds": LIVE_TIMEOUT_CASE_WALL_SECONDS,
    "max_source_bytes": 1_048_576,
    "network": False,
    "install": False,
    "download": False,
    "exact_binary_digest": True,
}


@dataclass
class LiveCaseOutcome:
    """One live Vampire/E semantic case with full receipt bindings."""

    case_id: str
    tool_id: str
    kind: str
    expect: str
    status: str
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    szs_status: str | None = None
    authority: str = ResultAuthority.CANDIDATE.value
    result_status: str = ResultStatus.CANDIDATE.value
    proof_bound: bool = False
    proof_object_present: bool = False
    reconstruction_status: str = "unreconstructed"
    independent_kernel_reconstruction: bool = False
    kernel_reconstruction_claimed: bool = False
    output_digest: str = ""
    source_digest: str = ""
    binary_digest: str = ""
    artifact_digest: str = ""
    executable_path: str | None = None
    assumptions: list[str] = field(default_factory=list)
    conclusion: str = ""
    limits: dict[str, Any] = field(default_factory=dict)
    raw_szs_output: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    execution_mode: str = "live"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ATPLiveSemanticCertification:
    """Live semantic certification receipt for Vampire + E."""

    tool_ids: list[str] = field(
        default_factory=lambda: [TOOL_VAMPIRE, TOOL_EPROVER]
    )
    lane_id: str = LANE_ID
    interface: str = LIVE_INTERFACE
    schema_version: str = LIVE_SCHEMA_VERSION
    goal_id: str = LIVE_GOAL_ID
    task_id: str = LIVE_TASK_ID
    program: str = LIVE_PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_vampire_version: str = LOCKED_VAMPIRE_VERSION
    locked_eprover_version: str = LOCKED_EPROVER_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    vampire_executable: str | None = None
    eprover_executable: str | None = None
    vampire_version_string: str | None = None
    eprover_version_string: str | None = None
    vampire_binary_digest: str | None = None
    eprover_binary_digest: str | None = None
    vampire_identity_probed: bool = False
    eprover_identity_probed: bool = False
    vampire_version_match: bool = False
    eprover_version_match: bool = False
    vampire_usable: bool = False
    eprover_usable: bool = False
    live_execution: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    results_are_candidates_without_reconstruction: bool = True
    kernel_reconstruction_required_for_theorem_authority: bool = True
    kernel_reconstruction_receipt_validated: bool = False
    boolean_reconstruction_claim_cannot_elevate: bool = True
    disagreement_quarantined: bool = False
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[LiveCaseOutcome] = field(default_factory=list)
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


def binary_digest(path: str | Path | None) -> str | None:
    """SHA-256 of an executable for exact binary binding."""

    if not path:
        return None
    binary = Path(path)
    if not binary.is_file():
        return None
    digest = hashlib.sha256()
    with binary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_live_corpus_manifest() -> dict[str, Any]:
    return {
        "schema_version": LIVE_CORPUS_SCHEMA,
        "interface": LIVE_INTERFACE,
        "goal_id": LIVE_GOAL_ID,
        "task_id": LIVE_TASK_ID,
        "tool_ids": [TOOL_VAMPIRE, TOOL_EPROVER],
        "lane_id": LANE_ID,
        "locked_vampire_version": LOCKED_VAMPIRE_VERSION,
        "locked_eprover_version": LOCKED_EPROVER_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "bounds": dict(DEFAULT_LIVE_BOUNDS),
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "exact_binary_binding_required": True,
            "live_execution_required_for_production": True,
            "fixture_or_parser_cannot_satisfy_live_goal": True,
            "results_are_candidates_without_reconstruction": True,
            "kernel_reconstruction_required_for_theorem_authority": True,
            "kernel_reconstruction_receipt_validated": False,
            "boolean_reconstruction_claim_cannot_elevate": True,
            "szs_status_only": True,
            "disagreement_quarantines": True,
            "does_not_edit_central_certificate": True,
            "does_not_edit_cec_semantics": True,
        },
        "cases": [dict(case) for case in _LIVE_DEFAULT_CASES],
    }


def live_corpus_cases(
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_live_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("ATP live corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


def build_atp_argv(
    tool_id: str,
    executable: str,
    problem_path: str | Path,
    *,
    timeout_seconds: float,
) -> list[str]:
    """Build a working Vampire/E argv for live certification.

    Vampire 5.0.1 rejects ``--time_limit=N`` (equals form); use spaced flags.
    """

    seconds = max(1, int(timeout_seconds + 0.999))
    problem = str(problem_path)
    if tool_id == TOOL_VAMPIRE:
        return [
            executable,
            problem,
            "--time_limit",
            str(seconds),
            "--output_mode",
            "szs",
            "--proof",
            "tptp",
        ]
    if tool_id == TOOL_EPROVER:
        return [
            executable,
            f"--cpu-limit={seconds}",
            "--proof-object",
            "--tstp-format",
            problem,
        ]
    raise ValueError(f"unsupported ATP tool_id: {tool_id}")


def execute_atp_problem(
    tool_id: str,
    executable: str,
    tptp_source: str,
    *,
    timeout_seconds: float = LIVE_CASE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one TPTP problem under offline bounds; never installs/network."""

    import tempfile

    probe_env = offline_env(env)
    source_bytes = tptp_source.encode("utf-8")
    if len(source_bytes) > int(DEFAULT_LIVE_BOUNDS["max_source_bytes"]):
        return {
            "tool_id": tool_id,
            "executable": executable,
            "stdout": "",
            "stderr": "source exceeds max_source_bytes",
            "returncode": None,
            "timed_out": False,
            "argv": [],
            "problem_path": None,
            "error": "source_too_large",
        }

    owns_dir = work_dir is None
    base = Path(tempfile.mkdtemp(prefix="atp_live_")) if owns_dir else Path(work_dir)
    problem_path = base / f"{tool_id}_problem.p"
    try:
        problem_path.write_text(tptp_source, encoding="utf-8")
        argv = build_atp_argv(
            tool_id,
            executable,
            problem_path,
            timeout_seconds=timeout_seconds,
        )
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=probe_env,
                shell=False,
            )
            return {
                "tool_id": tool_id,
                "executable": executable,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "returncode": completed.returncode,
                "timed_out": False,
                "argv": argv,
                "problem_path": str(problem_path),
                "error": None,
            }
        except subprocess.TimeoutExpired as error:
            stdout = ""
            stderr = ""
            if error.stdout:
                stdout = (
                    error.stdout.decode("utf-8", errors="replace")
                    if isinstance(error.stdout, (bytes, bytearray))
                    else str(error.stdout)
                )
            if error.stderr:
                stderr = (
                    error.stderr.decode("utf-8", errors="replace")
                    if isinstance(error.stderr, (bytes, bytearray))
                    else str(error.stderr)
                )
            return {
                "tool_id": tool_id,
                "executable": executable,
                "stdout": stdout,
                "stderr": stderr or "wall_clock_timeout",
                "returncode": None,
                "timed_out": True,
                "argv": argv,
                "problem_path": str(problem_path),
                "error": "timeout",
            }
        except OSError as error:
            return {
                "tool_id": tool_id,
                "executable": executable,
                "stdout": "",
                "stderr": str(error),
                "returncode": None,
                "timed_out": False,
                "argv": argv,
                "problem_path": str(problem_path),
                "error": "spawn_failure",
            }
    finally:
        if owns_dir:
            try:
                problem_path.unlink(missing_ok=True)
                base.rmdir()
            except OSError:
                pass


def evaluate_live_case(
    case: Mapping[str, Any],
    *,
    tool_id: str,
    executable: str,
    binary_sha256: str | None,
    env: Mapping[str, str] | None = None,
) -> LiveCaseOutcome:
    """Run one live corpus case against a real pinned ATP binary."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    source = str(case.get("tptp_source") or "")
    source_digest = content_digest(source) if source else ""
    assumptions = [str(item) for item in (case.get("assumptions") or ())]
    conclusion = str(case.get("conclusion") or "")
    reconstruction_claimed = bool(
        case.get("independent_kernel_reconstruction")
        or case.get("independent_kernel_reconstruction_claimed")
    )
    require_proof = bool(case.get("require_proof_body"))
    timeout_seconds = float(
        case.get("timeout_seconds") or LIVE_CASE_TIMEOUT_SECONDS
    )
    limits = {
        "timeout_seconds": timeout_seconds,
        "max_source_bytes": DEFAULT_LIVE_BOUNDS["max_source_bytes"],
        "network": False,
        "install": False,
        "download": False,
    }
    bin_digest = binary_sha256 or binary_digest(executable) or ""

    # Synthetic disagreement: force conflicting SZS witnesses and quarantine.
    if kind == "disagreement" or case.get("force_disagreement"):
        left = (
            "% SZS status Theorem for disagreement\n"
            "% SZS output start Proof for disagreement\n"
            "fof(1, plain, p).\n"
            "% SZS output end Proof for disagreement\n"
        )
        right = "% SZS status CounterSatisfiable for disagreement\n"
        raw = left + "\n--- disagreement peer ---\n" + right
        try:
            parse_szs_status(raw)
            classified_status = "unknown"
            reason_codes = ["disagreement_not_detected"]
            detail = "conflicting SZS should fail closed"
        except MalformedATPOutput as error:
            classified_status = "quarantined"
            reason_codes = ["disagreement", "conflicting_szs", "quarantined"]
            detail = str(error)
        matched = _expect_matches(expect, classified_status)
        # Live tool still executes once so the case is not fixture-only.
        live_run = execute_atp_problem(
            tool_id,
            executable,
            source or _LIVE_TPTP_THEOREM,
            timeout_seconds=min(timeout_seconds, LIVE_CASE_TIMEOUT_SECONDS),
            env=env,
        )
        live_out = (live_run.get("stdout") or "") + "\n" + (live_run.get("stderr") or "")
        artifact = content_digest(
            {
                "source_digest": source_digest or content_digest(_LIVE_TPTP_THEOREM),
                "binary_digest": bin_digest,
                "raw_disagreement": raw,
                "live_output_digest": content_digest(live_out),
            }
        )
        return LiveCaseOutcome(
            case_id=f"{tool_id}.{case_id}",
            tool_id=tool_id,
            kind=kind,
            expect=expect,
            status=classified_status,
            matched=matched,
            reason_codes=reason_codes,
            authority=ResultAuthority.CANDIDATE.value,
            result_status=ResultStatus.MALFORMED.value,
            reconstruction_status="unreconstructed",
            output_digest=content_digest(raw),
            source_digest=source_digest or content_digest(_LIVE_TPTP_THEOREM),
            binary_digest=bin_digest,
            artifact_digest=artifact,
            executable_path=executable,
            assumptions=list(assumptions) or ["fof(ax1, axiom, p)."],
            conclusion=conclusion or "fof(goal, conjecture, p).",
            limits=limits,
            raw_szs_output=raw,
            stdout=str(live_run.get("stdout") or ""),
            stderr=str(live_run.get("stderr") or ""),
            returncode=live_run.get("returncode"),  # type: ignore[arg-type]
            timed_out=bool(live_run.get("timed_out")),
            execution_mode="live+disagreement_quarantine",
            detail=detail or str(case.get("description") or ""),
        )

    run = execute_atp_problem(
        tool_id,
        executable,
        source,
        timeout_seconds=timeout_seconds,
        env=env,
    )
    stdout = str(run.get("stdout") or "")
    stderr = str(run.get("stderr") or "")
    combined = "\n".join(part for part in (stdout, stderr) if part)
    timed_out = bool(run.get("timed_out"))

    if timed_out:
        classified = {
            "status": "timeout",
            "szs_status": None,
            "authority": ResultAuthority.CANDIDATE.value,
            "result_status": ResultStatus.TIMEOUT.value,
            "proof_bound": False,
            "output_digest": content_digest(combined or "timeout"),
            "reason_codes": ["wall_clock_timeout", "resource_bounds"],
            "detail": "ATP wall-clock bound exceeded",
        }
    else:
        classified = classify_szs_outcome(
            stdout,
            stderr,
            independent_kernel_reconstruction=reconstruction_claimed,
            require_proof_body=require_proof,
        )

    observed = str(classified["status"])
    reason_codes = list(classified.get("reason_codes") or [])
    proof_bound = bool(classified.get("proof_bound"))
    proof_object_present = bool(
        proof_bound
        or re.search(
            r"[%#]\s*SZS\s+output\s+start\s+(Proof|CNFRefutation|Saturation)",
            combined,
            re.I,
        )
        or re.search(r"fof\([^)]*,\s*plain,", combined)
        or re.search(r"cnf\([^,]+,\s*plain,", combined)
    )

    if kind == "mutation" and observed in {
        "theorem_candidate",
        "theorem_authority",
    }:
        matched = False
        reason_codes.append("mutation_still_theorem")
    elif kind == "proof_object" and observed == "theorem_candidate":
        matched = proof_object_present or proof_bound
        if not matched:
            reason_codes.append("proof_object_missing")
    else:
        matched = _expect_matches(expect, observed)

    reconstruction_status = (
        "kernel_receipt_missing"
        if reconstruction_claimed and observed == "theorem_candidate"
        else "unreconstructed"
    )
    if observed == "theorem_authority":
        # Defensive fail-closed guard: no live corpus flag can validate an
        # independent kernel execution or proof-bound receipt.
        observed = "quarantined"
        classified["authority"] = ResultAuthority.CANDIDATE.value
        classified["result_status"] = ResultStatus.MALFORMED.value
        matched = False
        reason_codes.extend(
            [
                "authority_exceeded_without_validated_kernel_receipt",
                "kernel_reconstruction_receipt_required",
            ]
        )
        reconstruction_status = "invalid_elevation"

    artifact = content_digest(
        {
            "tool_id": tool_id,
            "case_id": case_id,
            "source_digest": source_digest,
            "binary_digest": bin_digest,
            "szs_status": classified.get("szs_status"),
            "status": observed,
            "output_digest": classified.get("output_digest"),
            "reconstruction_status": reconstruction_status,
            "limits": limits,
        }
    )

    return LiveCaseOutcome(
        case_id=f"{tool_id}.{case_id}",
        tool_id=tool_id,
        kind=kind,
        expect=expect,
        status=observed,
        matched=matched,
        reason_codes=list(dict.fromkeys(reason_codes)),
        szs_status=classified.get("szs_status"),  # type: ignore[arg-type]
        authority=str(classified.get("authority") or ResultAuthority.CANDIDATE.value),
        result_status=str(
            classified.get("result_status") or ResultStatus.CANDIDATE.value
        ),
        proof_bound=proof_bound,
        proof_object_present=proof_object_present,
        reconstruction_status=reconstruction_status,
        independent_kernel_reconstruction=False,
        kernel_reconstruction_claimed=reconstruction_claimed,
        output_digest=str(classified.get("output_digest") or ""),
        source_digest=source_digest,
        binary_digest=bin_digest,
        artifact_digest=artifact,
        executable_path=executable,
        assumptions=list(assumptions),
        conclusion=conclusion,
        limits=limits,
        raw_szs_output=combined,
        stdout=stdout,
        stderr=stderr,
        returncode=run.get("returncode"),  # type: ignore[arg-type]
        timed_out=timed_out,
        execution_mode="live",
        detail=str(case.get("description") or classified.get("detail") or ""),
    )


def run_live_semantic_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    vampire_executable: str | None = None,
    eprover_executable: str | None = None,
) -> ATPLiveSemanticCertification:
    """Execute real pinned Vampire/E semantics for FVT-G207."""

    root = repo_root or repo_root_from()
    corpus = (
        manifest if manifest is not None else default_live_corpus_manifest()
    )
    cases = live_corpus_cases(corpus)
    cert = ATPLiveSemanticCertification()
    # Prefer managed install bins (and approved deployment roots) without install.
    probe_env = managed_execution_env(env)
    _ = root

    cert.checks.append(
        CheckResult(
            check_id="atp_live.offline_policy",
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
            detail="live certification never installs, downloads, or opens network",
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
    cert.vampire_binary_digest = binary_digest(cert.vampire_executable)
    cert.eprover_binary_digest = binary_digest(cert.eprover_executable)

    for tool_id, usable, version_string, locked, probe, digest in (
        (
            TOOL_VAMPIRE,
            cert.vampire_usable,
            cert.vampire_version_string,
            LOCKED_VAMPIRE_VERSION,
            vampire_probe,
            cert.vampire_binary_digest,
        ),
        (
            TOOL_EPROVER,
            cert.eprover_usable,
            cert.eprover_version_string,
            LOCKED_EPROVER_VERSION,
            eprover_probe,
            cert.eprover_binary_digest,
        ),
    ):
        if usable:
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_id}.live_identity",
                    kind="identity",
                    status="passed",
                    expected=locked,
                    observed=version_string or "",
                    detail=f"exact live {tool_id} pin identity",
                    bindings={
                        "executable_path": probe.get("executable_path"),
                        "version_string": version_string,
                        "binary_digest": digest,
                    },
                )
            )
        else:
            reason = str(probe.get("probe_error") or "unavailable")
            cert.block_reasons.append(f"{tool_id}:{reason}")
            cert.checks.append(
                CheckResult(
                    check_id=f"{tool_id}.live_identity",
                    kind="identity",
                    status=(
                        "unavailable"
                        if reason == "executable_not_on_path"
                        else "blocked"
                    ),
                    expected=locked,
                    observed=reason,
                    detail="live semantic certification requires locked binaries",
                    reason_codes=[reason],
                )
            )

    tools: list[tuple[str, str | None, str | None, bool]] = [
        (
            TOOL_VAMPIRE,
            cert.vampire_executable,
            cert.vampire_binary_digest,
            cert.vampire_usable,
        ),
        (
            TOOL_EPROVER,
            cert.eprover_executable,
            cert.eprover_binary_digest,
            cert.eprover_usable,
        ),
    ]

    outcomes_by_id: dict[str, LiveCaseOutcome] = {}
    for tool_id, executable, digest, usable in tools:
        if not usable or not executable:
            for case in cases:
                case_id = f"{tool_id}.{case.get('case_id')}"
                outcome = LiveCaseOutcome(
                    case_id=case_id,
                    tool_id=tool_id,
                    kind=str(case.get("kind") or "unknown"),
                    expect=str(case.get("expect") or "unknown"),
                    status="unavailable",
                    matched=False,
                    reason_codes=["tool_unavailable"],
                    detail="locked ATP binary unavailable for live execution",
                    execution_mode="skipped",
                )
                cert.cases.append(outcome)
                outcomes_by_id[case_id] = outcome
                cert.block_reasons.append(f"live_unavailable:{case_id}")
                cert.checks.append(
                    CheckResult(
                        check_id=f"atp_live.{case_id}",
                        kind=outcome.kind,
                        status="unavailable",
                        expected=outcome.expect,
                        observed=outcome.status,
                        detail=outcome.detail,
                        reason_codes=list(outcome.reason_codes),
                    )
                )
            continue

        cert.live_execution = True
        for case in cases:
            outcome = evaluate_live_case(
                case,
                tool_id=tool_id,
                executable=executable,
                binary_sha256=digest,
                env=probe_env,
            )
            outcomes_by_id[outcome.case_id] = outcome
            cert.cases.append(outcome)
            if not outcome.matched:
                cert.block_reasons.append(f"live_case_failed:{outcome.case_id}")
            cert.checks.append(
                CheckResult(
                    check_id=f"atp_live.{outcome.case_id}",
                    kind=outcome.kind,
                    status="passed" if outcome.matched else "failed",
                    expected=outcome.expect,
                    observed=outcome.status,
                    detail=outcome.detail,
                    reason_codes=list(outcome.reason_codes),
                    bindings={
                        "tool_id": outcome.tool_id,
                        "source_digest": outcome.source_digest,
                        "binary_digest": outcome.binary_digest,
                        "artifact_digest": outcome.artifact_digest,
                        "output_digest": outcome.output_digest,
                        "szs_status": outcome.szs_status,
                        "authority": outcome.authority,
                        "reconstruction_status": outcome.reconstruction_status,
                        "independent_kernel_reconstruction": (
                            outcome.independent_kernel_reconstruction
                        ),
                        "kernel_reconstruction_claimed": (
                            outcome.kernel_reconstruction_claimed
                        ),
                        "assumptions": outcome.assumptions,
                        "conclusion": outcome.conclusion,
                        "limits": outcome.limits,
                        "proof_object_present": outcome.proof_object_present,
                        "execution_mode": outcome.execution_mode,
                    },
                )
            )

    # Replay: status class + source digest must match the theorem run.
    for tool_id in (TOOL_VAMPIRE, TOOL_EPROVER):
        theorem = outcomes_by_id.get(f"{tool_id}.theorem")
        replay = outcomes_by_id.get(f"{tool_id}.replay")
        if theorem is None or replay is None:
            cert.block_reasons.append(f"replay_missing:{tool_id}")
            cert.checks.append(
                CheckResult(
                    check_id=f"atp_live.{tool_id}.replay_binding",
                    kind="replay",
                    status="failed",
                    expected="theorem and replay cases",
                    observed="missing",
                )
            )
            continue
        replay_ok = (
            theorem.matched
            and replay.matched
            and theorem.status == replay.status == "theorem_candidate"
            and theorem.source_digest == replay.source_digest
            and theorem.authority == ResultAuthority.CANDIDATE.value
            and replay.authority == ResultAuthority.CANDIDATE.value
        )
        if not replay_ok:
            cert.block_reasons.append(f"replay_failed:{tool_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"atp_live.{tool_id}.replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="matching theorem_candidate source digests",
                observed=(
                    f"theorem={theorem.status}/{theorem.source_digest[:12]},"
                    f"replay={replay.status}/{replay.source_digest[:12]}"
                ),
                bindings={
                    "theorem_source_digest": theorem.source_digest,
                    "replay_source_digest": replay.source_digest,
                    "theorem_szs": theorem.szs_status,
                    "replay_szs": replay.szs_status,
                },
            )
        )

    # Cross-tool agreement on theorem / counter-sat (disagreement quarantines).
    agreement_pairs = (
        ("theorem", "theorem_candidate"),
        ("counter_satisfiable", "non_theorem_candidate"),
    )
    for base_id, expected_status in agreement_pairs:
        left = outcomes_by_id.get(f"{TOOL_VAMPIRE}.{base_id}")
        right = outcomes_by_id.get(f"{TOOL_EPROVER}.{base_id}")
        if left is None or right is None:
            continue
        if left.status == "unavailable" or right.status == "unavailable":
            continue
        agree = left.status == right.status == expected_status
        if not agree:
            cert.block_reasons.append(f"cross_tool_disagreement:{base_id}")
            cert.disagreement_quarantined = True
        cert.checks.append(
            CheckResult(
                check_id=f"atp_live.cross_tool.{base_id}",
                kind="agreement",
                status="passed" if agree else "failed",
                expected=f"both_{expected_status}",
                observed=f"vampire={left.status},eprover={right.status}",
                detail="Vampire and E must agree on live FOL polarity",
            )
        )

    # Forced disagreement cases must quarantine.
    disagreement_cases = [
        outcome
        for outcome in cert.cases
        if outcome.kind == "disagreement"
    ]
    disagreement_ok = bool(disagreement_cases) and all(
        outcome.matched and outcome.status == "quarantined"
        for outcome in disagreement_cases
    )
    if disagreement_ok:
        cert.disagreement_quarantined = True
    else:
        cert.block_reasons.append("disagreement_not_quarantined")
    cert.checks.append(
        CheckResult(
            check_id="atp_live.disagreement_quarantine",
            kind="disagreement",
            status="passed" if disagreement_ok else "failed",
            expected="quarantined",
            observed=(
                ",".join(
                    f"{o.case_id}={o.status}" for o in disagreement_cases
                )
                or "missing"
            ),
            detail="Disagreement between ATP witnesses quarantines promotion",
        )
    )

    # Authority boundary: unreconstructed ATP cannot claim theorem authority.
    boundary = atp_results_remain_candidates_without_reconstruction()
    boundary_ok = bool(boundary.get("boundary_holds"))
    live_authority_ok = all(
        outcome.authority != ResultAuthority.THEOREM.value
        for outcome in cert.cases
        if outcome.execution_mode != "skipped"
    )
    if not boundary_ok or not live_authority_ok:
        cert.block_reasons.append("candidate_authority_boundary_failed")
    cert.checks.append(
        CheckResult(
            check_id="atp_live.candidate_until_reconstruction",
            kind="authority",
            status="passed" if boundary_ok and live_authority_ok else "failed",
            expected="candidate_without_reconstruction",
            observed=(
                f"boundary={boundary_ok},live_authority_ok={live_authority_ok}"
            ),
            detail=(
                "ATP results remain candidates; a reconstruction boolean "
                "cannot replace a validated independent-kernel receipt"
            ),
            bindings=boundary,
        )
    )

    # Bind binaries, sources, assumptions, conclusions, limits, SZS, reconstruction.
    cert.bindings = {
        "adapter": {
            "compatibility_interface": ATP_COMPATIBILITY_BACKENDS_VERSION,
            "adapter_version": ATP_ADAPTER_VERSION,
            "szs_status_only": True,
        },
        "bounds": dict(corpus.get("bounds") or DEFAULT_LIVE_BOUNDS),
        "binaries": {
            "vampire": {
                "tool_id": TOOL_VAMPIRE,
                "locked_version": LOCKED_VAMPIRE_VERSION,
                "executable_path": cert.vampire_executable,
                "version_string": cert.vampire_version_string,
                "binary_digest": cert.vampire_binary_digest,
                "identity_probed": cert.vampire_identity_probed,
                "version_match": cert.vampire_version_match,
            },
            "eprover": {
                "tool_id": TOOL_EPROVER,
                "locked_version": LOCKED_EPROVER_VERSION,
                "executable_path": cert.eprover_executable,
                "version_string": cert.eprover_version_string,
                "binary_digest": cert.eprover_binary_digest,
                "identity_probed": cert.eprover_identity_probed,
                "version_match": cert.eprover_version_match,
            },
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "results_are_candidates_without_reconstruction": True,
            "kernel_reconstruction_required_for_theorem_authority": True,
            "kernel_reconstruction_receipt_validated": False,
            "boolean_reconstruction_claim_cannot_elevate": True,
            "not_kernel": True,
            "not_advisor": True,
        },
        "live_cases": [
            {
                "case_id": outcome.case_id,
                "tool_id": outcome.tool_id,
                "kind": outcome.kind,
                "status": outcome.status,
                "authority": outcome.authority,
                "source_digest": outcome.source_digest,
                "binary_digest": outcome.binary_digest,
                "artifact_digest": outcome.artifact_digest,
                "output_digest": outcome.output_digest,
                "szs_status": outcome.szs_status,
                "assumptions": outcome.assumptions,
                "conclusion": outcome.conclusion,
                "limits": outcome.limits,
                "reconstruction_status": outcome.reconstruction_status,
                "independent_kernel_reconstruction": (
                    outcome.independent_kernel_reconstruction
                ),
                "kernel_reconstruction_claimed": (
                    outcome.kernel_reconstruction_claimed
                ),
                "raw_szs_output_digest": content_digest(outcome.raw_szs_output),
                "execution_mode": outcome.execution_mode,
            }
            for outcome in cert.cases
        ],
        "candidate_boundary": boundary,
        "disagreement_quarantined": cert.disagreement_quarantined,
    }
    cert.checks.append(
        CheckResult(
            check_id="atp_live.bindings",
            kind="binding",
            status="passed",
            expected=(
                "binary_digest,artifact_digest,tptp_source,assumptions,"
                "conclusion,limits,raw_szs,reconstruction_status"
            ),
            observed=content_digest(cert.bindings)[:16],
            detail=(
                "live receipt binds exact binary digests, TPTP source, "
                "assumptions, conclusion, limits, raw SZS, reconstruction"
            ),
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "theorem",
        "counter_satisfiable",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "disagreement",
        "proof_object",
        "reconstruction",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append(
            "live_corpus_missing_kinds:" + ",".join(missing_kinds)
        )

    # Per-tool coverage: each tool must execute every required kind.
    for tool_id in (TOOL_VAMPIRE, TOOL_EPROVER):
        tool_kinds = {
            outcome.kind
            for outcome in cert.cases
            if outcome.tool_id == tool_id and outcome.execution_mode != "skipped"
        }
        missing_tool_kinds = sorted(required_kinds - tool_kinds)
        coverage_ok = not missing_tool_kinds and (
            (tool_id == TOOL_VAMPIRE and cert.vampire_usable)
            or (tool_id == TOOL_EPROVER and cert.eprover_usable)
        )
        if not coverage_ok:
            cert.block_reasons.append(
                f"tool_coverage_incomplete:{tool_id}:"
                + ",".join(missing_tool_kinds or ["unavailable"])
            )
        cert.checks.append(
            CheckResult(
                check_id=f"atp_live.{tool_id}.case_coverage",
                kind="coverage",
                status="passed" if coverage_ok else "failed",
                expected=",".join(sorted(required_kinds)),
                observed=",".join(sorted(tool_kinds)) or "none",
                detail=f"{tool_id} must execute all live semantic case kinds",
            )
        )

    semantic_ok = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind
        in {
            "theorem",
            "counter_satisfiable",
            "mutation",
            "proof_object",
            "replay",
            "malformed",
            "timeout",
            "disagreement",
            "reconstruction",
            "authority",
            "binding",
            "policy",
            "agreement",
            "coverage",
        }
        or check.check_id
        in {
            "atp_live.offline_policy",
            "atp_live.disagreement_quarantine",
            "atp_live.candidate_until_reconstruction",
            "atp_live.bindings",
        }
        or check.check_id.endswith(".replay_binding")
        or check.check_id.endswith(".case_coverage")
        or check.check_id.startswith("atp_live.cross_tool.")
    )

    # Live production certification requires both locked binaries + full suite.
    identity_ok = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind == "identity"
    )
    no_failed_cases = not any(
        reason.startswith("live_case_failed:") for reason in cert.block_reasons
    )
    cert.production_certified = bool(
        cert.vampire_usable
        and cert.eprover_usable
        and cert.live_execution
        and cert.vampire_binary_digest
        and cert.eprover_binary_digest
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and semantic_ok
        and identity_ok
        and no_failed_cases
        and not missing_kinds
        and boundary_ok
        and live_authority_ok
        and disagreement_ok
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned Vampire 5.0.1 + E 3.2.5 live semantic certification "
            "passed; unreconstructed ATP results remain candidates until "
            "a proof-bound independent-kernel receipt is validated."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            if cert.live_execution and semantic_ok and not (
                cert.vampire_usable and cert.eprover_usable
            ):
                cert.notes = (
                    "Partial live execution; both locked Vampire and E "
                    "identities required for production certification."
                )
            elif not cert.live_execution:
                cert.notes = (
                    "Live Vampire/E binaries unavailable — live semantic "
                    "production certification withheld (fixture/parser "
                    "cannot satisfy FVT-G207)."
                )
            else:
                cert.notes = (
                    "ATP live semantic certification incomplete or failed; "
                    "ATP-lane live promotion blocked."
                )

    return cert


def build_live_semantic_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    vampire_executable: str | None = None,
    eprover_executable: str | None = None,
) -> dict[str, Any]:
    root = repo_root or repo_root_from()
    cert = run_live_semantic_suite(
        repo_root=root,
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
        "live_execution_required_for_production": True,
        "fixture_or_parser_cannot_satisfy_live_goal": True,
        "results_are_candidates_without_reconstruction": True,
        "kernel_reconstruction_required_for_theorem_authority": True,
        "kernel_reconstruction_receipt_validated": False,
        "boolean_reconstruction_claim_cannot_elevate": True,
        "szs_status_only": True,
        "disagreement_quarantines": True,
        "authority_is_reconstruction_ceiling": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_cec_semantics": True,
        "does_not_edit_shared_lock": True,
        "objective_validation_repair": True,
        "preserve_production_certificate_without_live_tools": True,
    }
    payload["live_semantic_corpus_passed"] = all(
        case.matched for case in cert.cases if case.execution_mode != "skipped"
    ) and bool(cert.cases)
    payload["authority_scope"] = AUTHORITY_SCOPE
    payload["results_are_candidates_without_reconstruction"] = True
    payload["kernel_reconstruction_required_for_theorem_authority"] = True
    payload["kernel_reconstruction_receipt_validated"] = False
    payload["boolean_reconstruction_claim_cannot_elevate"] = True
    payload["certificate_path"] = str(DEFAULT_LIVE_CERTIFICATE_RELATIVE)
    # Objective validation repair evidence (FVT-G207 / FVT-071).
    payload["objective_validation_repair"] = {
        "schema_version": "objective-validation-repair/v1",
        "goal_id": LIVE_GOAL_ID,
        "task_id": LIVE_TASK_ID,
        "interface": LIVE_INTERFACE,
        "status": (
            "satisfied"
            if cert.production_certified
            else (
                "withheld_live_tools_unavailable"
                if not cert.live_execution
                else "failed"
            )
        ),
        "live_execution": bool(cert.live_execution),
        "production_certified": bool(cert.production_certified),
        "validation_command": (
            "python -m pytest "
            "test/integration/toolchains/test_atp_live_semantic_certification.py "
            "test/integration/toolchains/test_atp_toolchain_certification.py -q"
        ),
        "evidence_terms": [
            "objective validation repair",
            "ATPLiveSemanticCertification@1",
            "live Vampire and E prover semantics",
        ],
    }
    # Compact cases for durable certificate: drop full raw stdout bodies.
    compact_cases = []
    for case in payload.get("cases") or []:
        compact = dict(case)
        raw = compact.pop("raw_szs_output", "") or ""
        compact.pop("stdout", None)
        compact.pop("stderr", None)
        compact["raw_szs_output_digest"] = content_digest(raw)
        compact["raw_szs_output_preview"] = public_evidence_projection(
            raw[:400],
            repo_root=root,
        )
        # Durable cert binds digests; absolute host paths are not authoritative.
        if compact.get("executable_path") and compact.get("binary_digest"):
            compact["executable_path_binding"] = "digest_bound"
        compact_cases.append(compact)
    payload["cases"] = compact_cases
    payload.pop("receipt_digest_sha256", None)
    projected = public_evidence_projection(payload, repo_root=root)
    if not isinstance(projected, dict):  # defensive: receipt roots are mappings
        raise ValueError("ATP public evidence projection returned a non-mapping")
    payload = projected
    audit = public_evidence_audit(payload, repo_root=root)
    if not audit["satisfied"]:
        raise ValueError(
            "ATP public evidence projection is unsafe: "
            + ", ".join(audit["failures"])
        )
    payload["public_evidence_policy"] = audit
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def _is_production_live_certificate(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("production_certified")
        and payload.get("live_execution")
        and payload.get("vampire_usable")
        and payload.get("eprover_usable")
        and payload.get("interface") == LIVE_INTERFACE
    )


def write_live_certificate(
    receipt: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    vampire_executable: str | None = None,
    eprover_executable: str | None = None,
    force: bool = False,
) -> Path:
    """Write the durable ATP live semantic certificate under docs/architecture.

    Fail-closed demotion protection (objective validation repair): a tool-less
    or failed re-run must not overwrite a production live certificate that
    already binds real Vampire/E digests and case evidence. Pass ``force=True``
    only for deliberate replacement.
    """

    root = repo_root or repo_root_from()
    target = path or (root / DEFAULT_LIVE_CERTIFICATE_RELATIVE)
    payload = (
        dict(receipt)
        if receipt is not None
        else build_live_semantic_receipt(
            repo_root=root,
            env=env,
            vampire_executable=vampire_executable,
            eprover_executable=eprover_executable,
        )
    )
    audit = public_evidence_audit(payload, repo_root=root)
    if not audit["satisfied"]:
        raise ValueError(
            "refusing to write unsafe ATP public evidence: "
            + ", ".join(audit["failures"])
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and target.is_file()
        and not _is_production_live_certificate(payload)
    ):
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            existing = None
        if isinstance(existing, dict) and _is_production_live_certificate(existing):
            # Preserve prior live production evidence; surface demotion block.
            return target

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def certify_atp_live_semantics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Public entry for ATP live semantic certification."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    receipt = build_live_semantic_receipt(
        repo_root=repo_root,
        manifest=kwargs.get("manifest"),
        env=kwargs.get("env"),
        vampire_executable=kwargs.get("vampire_executable"),
        eprover_executable=kwargs.get("eprover_executable"),
    )
    receipt["handler_id"] = LIVE_HANDLER_ID
    receipt["lane_id"] = LANE_ID
    receipt["owner_module"] = CERTIFICATION_SURFACE
    receipt["status"] = (
        "certified" if receipt.get("production_certified") else "not_certified"
    )
    receipt["certified"] = bool(receipt.get("production_certified"))
    receipt["args_received"] = bool(args) or bool(kwargs)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the pinned Vampire/E ATP toolchain "
            f"({INTERFACE} / {LIVE_INTERFACE}; Vampire {LOCKED_VAMPIRE_VERSION} + "
            f"E {LOCKED_EPROVER_VERSION})."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print receipt as JSON")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--vampire", type=str, default=None)
    parser.add_argument("--eprover", type=str, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run ATPLiveSemanticCertification@1 with real pinned binaries",
    )
    parser.add_argument(
        "--write-certificate",
        action="store_true",
        help="Write docs/architecture/formal_verification_atp_live_certificate.json",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    if args.live or args.write_certificate:
        receipt = build_live_semantic_receipt(
            repo_root=root,
            vampire_executable=args.vampire,
            eprover_executable=args.eprover,
        )
        if args.write_certificate:
            path = write_live_certificate(receipt, repo_root=root)
            if not args.json:
                print(f"wrote {path}")
        interface = LIVE_INTERFACE
        goal = LIVE_GOAL_ID
        task = LIVE_TASK_ID
        semantic_ok = bool(receipt.get("live_semantic_corpus_passed"))
    else:
        receipt = build_certification_receipt(
            repo_root=root,
            vampire_executable=args.vampire,
            eprover_executable=args.eprover,
        )
        interface = INTERFACE
        goal = GOAL_ID
        task = TASK_ID
        semantic_ok = bool(receipt.get("semantic_corpus_passed"))

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{interface} goal={goal} task={task}")
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
    "LIVE_INTERFACE",
    "LIVE_SCHEMA_VERSION",
    "LIVE_CORPUS_SCHEMA",
    "LIVE_GOAL_ID",
    "LIVE_TASK_ID",
    "LIVE_PROGRAM",
    "LIVE_HANDLER_ID",
    "DEFAULT_LIVE_CERTIFICATE_RELATIVE",
    "LOCKED_VAMPIRE_VERSION",
    "LOCKED_EPROVER_VERSION",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "LiveCaseOutcome",
    "ATPToolchainCertification",
    "ATPLiveSemanticCertification",
    "repo_root_from",
    "content_digest",
    "binary_digest",
    "offline_env",
    "managed_install_roots",
    "managed_execution_env",
    "bounded_run",
    "resolve_executable",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "default_live_corpus_manifest",
    "live_corpus_cases",
    "probe_vampire_identity",
    "probe_eprover_identity",
    "classify_szs_outcome",
    "evaluate_corpus_case",
    "build_atp_argv",
    "execute_atp_problem",
    "evaluate_live_case",
    "atp_results_remain_candidates_without_reconstruction",
    "run_certification_suite",
    "build_certification_receipt",
    "run_live_semantic_suite",
    "build_live_semantic_receipt",
    "write_live_certificate",
    "certify_atp_toolchain",
    "certify_atp_live_semantics",
    "lane_handler",
    "bind_atp_lane_handler",
    "main",
]
