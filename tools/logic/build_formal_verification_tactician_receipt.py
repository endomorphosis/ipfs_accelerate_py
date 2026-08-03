#!/usr/bin/env python3
"""Build the formal verification tactician completion receipt.

``FormalVerificationTacticianCompletionReceipt@1`` / FVT-G090 (FVT-036).

Recomputes current-tree implementation completion and machine-specific
deployment certification from immutable child evidence. Never edits source
evidence to make the gate pass, never invents tool success, and never uses
hardcoded success counters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

INTERFACE: Final = "FormalVerificationTacticianCompletionReceipt@1"
PROGRAM_INTERFACE: Final = "FormalVerificationTacticianRelease@1"
SCHEMA_VERSION: Final = "formal-verification-tactician-completion-receipt/v1"
PROGRAM_GOAL_ID: Final = "FVT-G000"
COMPLETION_GOAL_ID: Final = "FVT-G090"
TASK_ID: Final = "FVT-036"
PROGRAM: Final = "formal-verification-tactician/readiness"

# Role-aware deployment reissue (FVT-G200). FVT-083 is the completed
# validation-gate successor and the only supervisor identity accepted for
# release evidence. FVT-053 is retained solely as legacy display context.
ROLE_AWARE_INTERFACE: Final = "RoleAwareFormalVerificationRelease@1"
ROLE_AWARE_SCHEMA_VERSION: Final = "formal-verification-role-aware-deployment-receipt/v1"
ROLE_AWARE_GOAL_ID: Final = "FVT-G200"
ROLE_AWARE_TASK_ID: Final = "FVT-083"
ROLE_AWARE_REPAIR_TASK_ID: Final = ROLE_AWARE_TASK_ID
ROLE_AWARE_LEGACY_DISPLAY_TASK_ID: Final = "FVT-053"
ROLE_AWARE_CANONICAL_TASK_CID: Final = (
    "baguqeerajpm5osvlu5g4ljby6tnibgz3oxsnjpnapmtmyxzpcallkkw4viga"
)
ROLE_AWARE_CANONICAL_TASK_KEY: Final = (
    "task/v1/4bd9d74aaba74dc5a438f4da809b3b75e4d4bda07b26cc5f2f1016b52adcaa0c"
)
ROLE_AWARE_INTEGRATION_BRANCH: Final = "agent/software-verification-prover-matrix"
SUPERVISOR_VALIDATION_DAG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/validation-dag-receipt@3"
)
ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE: Final = "objective validation repair"
ROLE_AWARE_OBJECTIVE_VALIDATION_COMMAND: Final = (
    "python -m pytest "
    "test/integration/test_formal_verification_real_tool_matrix.py "
    "test/integration/test_formal_verification_role_aware_completion.py "
    "test/api/test_formal_verification_tactician_readiness_completion.py -q"
)

# Role-aware release candidate (FVT-G213 / FVT-066). Pre-merge fan-in only;
# never claims its own future merge or deployment attestation.
RELEASE_CANDIDATE_INTERFACE: Final = "RoleAwareFormalVerificationReleaseCandidate@1"
RELEASE_CANDIDATE_SCHEMA_VERSION: Final = (
    "formal-verification-role-aware-release-candidate/v1"
)
RELEASE_CANDIDATE_GOAL_ID: Final = "FVT-G213"
RELEASE_CANDIDATE_TASK_ID: Final = "FVT-066"
RELEASE_CANDIDATE_MAX_STAGE: Final = "release_candidate"
RELEASE_CANDIDATE_PROGRAM: Final = (
    "formal-verification-tactician/toolchain-release-candidate"
)

# Production-semantic elevation fan-in (FVT-G213 / FVT-081). This receipt
# independently reconstructs the required PNMR evidence before a production
# elevation may appear on any release-candidate surface.
PRODUCTION_ELEVATION_FANIN_INTERFACE: Final = (
    "ProductionSemanticElevationFanIn@1"
)
PRODUCTION_ELEVATION_FANIN_SCHEMA_VERSION: Final = (
    "formal-verification-production-semantic-elevation-fanin/v1"
)
PRODUCTION_ELEVATION_FANIN_GOAL_ID: Final = "FVT-G213"
PRODUCTION_ELEVATION_FANIN_TASK_ID: Final = "FVT-081"
PRODUCTION_ELEVATION_FANIN_PROGRAM: Final = RELEASE_CANDIDATE_PROGRAM
PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS: Final[tuple[str, ...]] = (
    "positive",
    "negative",
    "mutation",
    "replay",
)
PRODUCTION_ELEVATION_FANIN_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_formal_verification_production_elevation_fanin.py "
    "test/integration/test_formal_verification_role_aware_release_candidate.py "
    "test/integration/test_formal_verification_real_tool_matrix.py -q"
)

DEFAULT_OBJECTIVES_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_readiness.objectives.md"
)
DEFAULT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json"
)
DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_deployment_receipt.json"
)
DEFAULT_RELEASE_CANDIDATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_release_candidate.json"
)
DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_production_elevation_fanin_receipt.json"
)
DEFAULT_BUILDER_RELATIVE: Final = Path(
    "tools/logic/build_formal_verification_tactician_receipt.py"
)
DEFAULT_COMPLETION_TEST_RELATIVE: Final = Path(
    "test/api/test_formal_verification_tactician_readiness_completion.py"
)
DEFAULT_ROLE_AWARE_TEST_RELATIVE: Final = Path(
    "test/integration/test_formal_verification_role_aware_completion.py"
)
DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE: Final = Path(
    "test/integration/test_formal_verification_role_aware_release_candidate.py"
)
DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE: Final = Path(
    "test/integration/toolchains/"
    "test_formal_verification_production_elevation_fanin.py"
)
DEFAULT_CERTIFIER_RELATIVE: Final = Path(
    "tools/logic/certify_formal_verification_toolchains.py"
)
DEFAULT_TOOLCHAIN_LOCK_RELATIVE: Final = Path(
    "config/formal_verification_toolchains.lock.json"
)
SUPERVISOR_COMPLETION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.member_completion_receipt@1"
)
SUPERVISOR_RELEASE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.release_evidence@1"
)
SUPERVISOR_RELEASE_EVIDENCE_INTERFACE: Final = "AgentSupervisorReleaseEvidence@1"
SUPERVISOR_RELEASE_EVIDENCE_GOAL_ID: Final = "FVT-G212"
SUPERVISOR_RELEASE_EVIDENCE_EXPORTER_RELATIVE: Final = Path(
    "ipfs_accelerate_py/agent_supervisor/release_evidence.py"
)
BENCHMARK_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "goal_tactician_authoritative_benchmark_evidence@1"
)
BENCHMARK_AUTHORITY_INTERFACE: Final = (
    "GoalTacticianAuthoritativeBenchmarkEvidence@1"
)
BENCHMARK_AUTHORITY_GOAL_ID: Final = "FVT-G063"
BENCHMARK_AUTHORITY_VERIFIER_RELATIVE: Final = Path(
    "ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_metrics.py"
)
BENCHMARK_AUTHORITY_VERIFIER_FUNCTION: Final = (
    "verify_authoritative_benchmark_evidence"
)
ROLE_AWARE_ATTESTATION_PATHS: Final[frozenset[str]] = frozenset(
    {
        DEFAULT_RECEIPT_RELATIVE.as_posix(),
        DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE.as_posix(),
        DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
        "docs/architecture/formal_verification_toolchain_certificate.json",
    }
)
RELEASE_CANDIDATE_ATTESTATION_PATHS: Final[frozenset[str]] = frozenset(
    {
        DEFAULT_RECEIPT_RELATIVE.as_posix(),
        DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
        "docs/architecture/formal_verification_toolchain_certificate.json",
        DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix(),
    }
)

# Evidence classes that cannot raise readiness past their declared ceiling.
# When present as the sole evidence for a supported managed capability they
# block candidate readiness rather than silently promote.
NON_AUTHORITATIVE_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "identity_plus_fixture_parser",
        "hermetic_adapter_shim",
        "hermetic_shadow_shim",
        "proposal_only_semantics",
        "fixture",
        "canned",
        "parser_only",
        "identity_only",
        "advisor",
        "shadow",
        "ambiguous",
        "stale",
        "incomplete",
        "external_prover_installation_pending",
        "quarantined_disagreement",
        "unavailable",
    }
)

BASELINE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_readiness_baseline.json"
)
TOOLCHAIN_CERT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_toolchain_certificate.json"
)
BENCHMARK_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_benchmark.json"
)
LIVE_REPORT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_live_example_report.json"
)
ROLLOUT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_rollout.md"
)
LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
CORPUS_MANIFEST_RELATIVE: Final = Path(
    "ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json"
)
PRODUCT_DOC_RELATIVE: Final = Path("docs/formal_verification_tactician.md")
RUNBOOK_RELATIVE: Final = Path(
    "docs/operations/formal_verification_tactician_runbook.md"
)
PUBLIC_API_TEST_RELATIVE: Final = Path(
    "ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py"
)
CLI_MCP_TEST_RELATIVE: Final = Path(
    "test/api/test_goal_tactician_cli_mcp_parity.py"
)
ADVERSARIAL_ROOT_TEST_RELATIVE: Final = Path(
    "test/security/test_formal_verification_tactician_adversarial.py"
)
ADVERSARIAL_DATASETS_TEST_RELATIVE: Final = Path(
    "ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py"
)
METRICS_MODULE_RELATIVE: Final = Path(
    "ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_metrics.py"
)

GIT_TIMEOUT_SECONDS: Final = 10.0
COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
GOAL_HEADING_RE: Final = re.compile(
    r"^## (FVT-G\d+) (.+)$",
    flags=re.MULTILINE,
)

HARD_ZERO_GATE_KEYS: Final[tuple[str, ...]] = (
    "false_proof_count",
    "false_closure_count",
    "secret_or_witness_leakage_count",
    "authority_boundary_violations",
    "unresolved_cross_provider_disagreement_count",
)

# Artifacts bound into the receipt with content identities. Keys are stable
# receipt field names; values are repo-relative paths.
BOUND_ARTIFACTS: Final[dict[str, Path]] = {
    "objectives_heap": DEFAULT_OBJECTIVES_RELATIVE,
    "readiness_baseline": BASELINE_RELATIVE,
    "toolchain_certificate": TOOLCHAIN_CERT_RELATIVE,
    "toolchain_lock": LOCK_RELATIVE,
    "benchmark_report": BENCHMARK_RELATIVE,
    "live_example_report": LIVE_REPORT_RELATIVE,
    "rollout_policy": ROLLOUT_RELATIVE,
    "corpus_manifest": CORPUS_MANIFEST_RELATIVE,
    "product_documentation": PRODUCT_DOC_RELATIVE,
    "operator_runbook": RUNBOOK_RELATIVE,
    "public_api_test": PUBLIC_API_TEST_RELATIVE,
    "cli_mcp_parity_test": CLI_MCP_TEST_RELATIVE,
    "adversarial_root_test": ADVERSARIAL_ROOT_TEST_RELATIVE,
    "adversarial_datasets_test": ADVERSARIAL_DATASETS_TEST_RELATIVE,
    "metrics_module": METRICS_MODULE_RELATIVE,
    "receipt_builder": DEFAULT_BUILDER_RELATIVE,
    "completion_test": DEFAULT_COMPLETION_TEST_RELATIVE,
    "role_aware_certifier": DEFAULT_CERTIFIER_RELATIVE,
    "role_aware_completion_test": DEFAULT_ROLE_AWARE_TEST_RELATIVE,
}

# Usable capabilities that remain below production authority until the open
# kernel/vendor/live-fan-in goals publish their specialized certificates.
REQUIRED_SEMANTIC_ELEVATIONS: Final[tuple[str, ...]] = (
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
    "coq",
    "isabelle",
)

# A benchmark can measure a hard-zero gate only when its cohort is explicitly
# live.  Fixture, simulated, synthetic, offline, parser, and canned cohorts are
# useful regression evidence, but they cannot establish a deployment claim.
AUTHORITATIVE_BENCHMARK_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "live",
        "calibrated",
    }
)
NON_AUTHORITATIVE_BENCHMARK_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "fixture",
        "simulated",
        "synthetic",
        "offline",
        "parser",
        "canned",
        "shadow",
    }
)

# Known open P0 findings apply pressure to the hard-zero dimensions they
# invalidate. Unknown P0 findings conservatively affect every local hard-zero
# dimension until they are classified and resolved.
P0_FINDING_GATE_MAP: Final[dict[str, tuple[str, ...]]] = {
    "receipt_verification_fail_open": (
        "false_proof_count",
        "false_closure_count",
        "authority_boundary_violations",
    ),
    "public_counterexample_raw_leak": (
        "secret_or_witness_leakage_count",
    ),
    "structural_repair_as_closure": (
        "false_proof_count",
        "false_closure_count",
        "authority_boundary_violations",
    ),
}


# ---------------------------------------------------------------------------
# Repo / git helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root for the formal-verification readiness tree."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_OBJECTIVES_RELATIVE).is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
        if (candidate / "config").is_dir() and (candidate / "tools" / "logic").is_dir():
            return candidate
    return Path.cwd().resolve()


def _git(
    repository: Path,
    *arguments: str,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_stdout(
    repository: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> str | None:
    completed = _git(repository, *arguments)
    if completed is None or completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if allow_empty:
        return value
    return value or None


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def content_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return sha256_bytes(encoded.encode("utf-8"))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def split_csv_field(value: str) -> list[str]:
    text = (value or "").strip()
    if not text or text.lower() in {"none", "—", "-", "n/a"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Objective heap parsing
# ---------------------------------------------------------------------------


def parse_objective_goals(objectives_text: str) -> list[dict[str, Any]]:
    """Parse FVT-G* goals from the objective heap markdown."""

    matches = list(GOAL_HEADING_RE.finditer(objectives_text))
    if not matches:
        raise ValueError("objective heap contains no FVT-G* headings")

    goals: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        goal_id = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(objectives_text)
        body = objectives_text[start:end]

        def field(name: str) -> str:
            found = re.search(rf"^- {re.escape(name)}: (.+)$", body, flags=re.MULTILINE)
            return found.group(1).strip() if found else ""

        goals.append(
            {
                "goal_id": goal_id,
                "title": title,
                "status": field("Status"),
                "parent": field("Parent") or None,
                "depends_on": split_csv_field(field("Depends on")),
                "evidence": split_csv_field(field("Evidence")),
                "outputs": split_csv_field(field("Outputs")),
                "validation": field("Validation"),
                "interfaces": split_csv_field(field("Interfaces")),
                "track": field("Track"),
                "acceptance": field("Acceptance"),
            }
        )
    return goals


def bind_child_goal(
    goal: Mapping[str, Any],
    *,
    repo_root: Path,
    self_generated: frozenset[str],
) -> dict[str, Any]:
    """Bind one executable child goal to present/missing evidence paths."""

    evidence = list(goal.get("evidence") or [])
    outputs = list(goal.get("outputs") or [])
    present: list[str] = []
    missing: list[str] = []
    for relative in evidence:
        path = repo_root / relative
        if path.is_file() or relative in self_generated:
            present.append(relative)
        else:
            missing.append(relative)

    output_present: list[str] = []
    output_missing: list[str] = []
    for relative in outputs:
        path = repo_root / relative
        if path.is_file() or relative in self_generated:
            output_present.append(relative)
        else:
            output_missing.append(relative)

    # A goal is bound when every evidence term is present (or is this receipt's
    # own generated surface while the builder runs).
    bound = not missing
    return {
        "goal_id": goal["goal_id"],
        "title": goal["title"],
        "status": goal.get("status") or "",
        "parent": goal.get("parent"),
        "depends_on": list(goal.get("depends_on") or []),
        "interfaces": list(goal.get("interfaces") or []),
        "evidence": evidence,
        "outputs": outputs,
        "validation": goal.get("validation") or "",
        "track": goal.get("track") or "",
        "evidence_present": present,
        "evidence_missing": missing,
        "outputs_present": output_present,
        "outputs_missing": output_missing,
        "bound": bound,
    }


# ---------------------------------------------------------------------------
# Tree / publication alignment
# ---------------------------------------------------------------------------


def observe_tree_alignment(repo_root: Path) -> dict[str, Any]:
    """Record parent, gitlink, embedded HEAD, and origin refs independently."""

    parent_commit = _git_stdout(repo_root, "rev-parse", "HEAD")
    parent_tree = _git_stdout(repo_root, "rev-parse", "HEAD^{tree}")
    parent_origin = _git_stdout(repo_root, "rev-parse", "origin/main")
    parent_porcelain = _git_stdout(repo_root, "status", "--porcelain", allow_empty=True) or ""

    gitlink_line = _git_stdout(repo_root, "ls-tree", "HEAD", "ipfs_datasets_py") or ""
    gitlink_commit = None
    for token in gitlink_line.split():
        if COMMIT_RE.fullmatch(token):
            gitlink_commit = token
            break

    datasets_root = repo_root / "ipfs_datasets_py"
    datasets_head = (
        _git_stdout(datasets_root, "rev-parse", "HEAD") if datasets_root.is_dir() else None
    )
    datasets_origin = (
        _git_stdout(datasets_root, "rev-parse", "origin/main")
        if datasets_root.is_dir()
        else None
    )
    datasets_porcelain = (
        _git_stdout(datasets_root, "status", "--porcelain", allow_empty=True) or ""
        if datasets_root.is_dir()
        else "missing_checkout"
    )

    parent_clean = parent_porcelain == ""
    datasets_clean = datasets_porcelain == ""
    gitlink_matches_embedded = bool(
        gitlink_commit and datasets_head and gitlink_commit == datasets_head
    )
    publication_lag = bool(
        datasets_origin
        and gitlink_commit
        and datasets_origin != gitlink_commit
    )
    parent_publication_lag = bool(
        parent_origin and parent_commit and parent_origin != parent_commit
    )

    diagnostics: list[str] = []
    if not parent_clean:
        diagnostics.append("parent_working_tree_dirty")
    if not datasets_clean:
        diagnostics.append("datasets_working_tree_dirty")
    if not gitlink_matches_embedded:
        diagnostics.append("gitlink_embedded_head_mismatch")
    if publication_lag:
        diagnostics.append("datasets_publication_lag_vs_origin_main")
    if parent_publication_lag:
        diagnostics.append("parent_publication_lag_vs_origin_main")

    return {
        "description": (
            "Parent commit, datasets gitlink, embedded HEAD, and origin/main "
            "are independent dimensions. Alignment of one never implies the others."
        ),
        "dimensions": [
            "parent_commit",
            "datasets_gitlink",
            "datasets_embedded_head",
            "datasets_origin_main",
            "datasets_cleanliness",
            "parent_origin_main",
        ],
        "parent": {
            "path": ".",
            "commit": parent_commit,
            "tree": parent_tree,
            "origin_main": parent_origin,
            "clean": parent_clean,
            "matches_origin_main": bool(
                parent_commit and parent_origin and parent_commit == parent_origin
            ),
        },
        "gitlink": {
            "path": "ipfs_datasets_py",
            "mode": "160000",
            "commit": gitlink_commit,
        },
        "embedded_checkout": {
            "path": "ipfs_datasets_py",
            "head": datasets_head,
            "clean": datasets_clean,
            "matches_gitlink": gitlink_matches_embedded,
            "origin_main": datasets_origin,
            "matches_origin_main": bool(
                datasets_head and datasets_origin and datasets_head == datasets_origin
            ),
        },
        "publication_gates": {
            "parent_publication_lag": parent_publication_lag,
            "datasets_publication_lag": publication_lag,
            "fetches_forbidden": True,
            "note": (
                "origin/main is a local remote-tracking ref only; this receipt "
                "never fetches. Publication lag is a separate gate from semantic "
                "implementation completion."
            ),
        },
        "aligned_for_implementation": bool(
            parent_commit
            and parent_tree
            and gitlink_commit
            and datasets_head
            and gitlink_matches_embedded
            and parent_clean
            and datasets_clean
        ),
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Child artifact binding + section assembly
# ---------------------------------------------------------------------------


def bind_artifacts(repo_root: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for key, relative in BOUND_ARTIFACTS.items():
        path = repo_root / relative
        identity = sha256_file(path)
        artifacts[key] = {
            "path": relative.as_posix(),
            "present": path.is_file(),
            "content_identity": identity,
        }
    return artifacts


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any, *, default: int = -1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _declared_mapping_identity_valid(
    payload: Mapping[str, Any],
    *,
    field: str,
    prefix: str = "",
    ensure_ascii: bool = False,
) -> bool:
    """Recompute a canonical identity instead of trusting a claimed digest."""

    stored = str(payload.get(field) or "")
    body = {key: value for key, value in payload.items() if key != field}
    if ensure_ascii:
        # FormalVerificationToolchainCertificate@1 is emitted by the certifier,
        # whose canonical JSON digest uses json.dumps' escaped-Unicode default.
        # Preserve that schema's canonicalization instead of silently applying
        # this builder's unescaped-Unicode receipt encoding.
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        computed = sha256_bytes(encoded.encode("utf-8"))
    else:
        computed = content_digest(body)
    if prefix:
        computed = prefix + computed.removeprefix("sha256:")
    candidates = {computed, computed.removeprefix("sha256:")}
    return bool(stored) and stored in candidates


def _benchmark_authority_anchor(
    benchmark: Mapping[str, Any],
    *,
    report_id: str,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Require the repository-owned verifier for a live benchmark claim."""

    authority = _safe_dict(benchmark.get("authoritative_measurement"))
    failures: list[str] = []
    result: dict[str, Any] = {
        "present": bool(authority),
        "bound": False,
        "schema": authority.get("schema"),
        "interface": authority.get("interface"),
        "goal_id": authority.get("goal_id"),
        "content_id": authority.get("content_id"),
        "report_id": authority.get("report_id"),
        "verifier": {},
        "failures": failures,
    }
    if not authority:
        failures.append("benchmark_authoritative_measurement_anchor_missing")
        return result
    if authority.get("schema") != BENCHMARK_AUTHORITY_SCHEMA:
        failures.append("benchmark_authority_schema_mismatch")
    if authority.get("interface") != BENCHMARK_AUTHORITY_INTERFACE:
        failures.append("benchmark_authority_interface_mismatch")
    if authority.get("goal_id") != BENCHMARK_AUTHORITY_GOAL_ID:
        failures.append("benchmark_authority_goal_mismatch")
    if not report_id or authority.get("report_id") != report_id:
        failures.append("benchmark_authority_report_id_mismatch")
    if not _declared_mapping_identity_valid(authority, field="content_id"):
        failures.append("benchmark_authority_content_id_mismatch")
    if repo_root is None:
        failures.append("benchmark_authority_repository_missing")
        return result

    root = repo_root.resolve()
    verifier_path = root / BENCHMARK_AUTHORITY_VERIFIER_RELATIVE
    verifier_claim = _safe_dict(authority.get("verifier"))
    expected_verifier_sha256 = sha256_file(verifier_path)
    verifier_bound = bool(
        verifier_claim.get("path")
        == BENCHMARK_AUTHORITY_VERIFIER_RELATIVE.as_posix()
        and expected_verifier_sha256
        and verifier_claim.get("sha256") == expected_verifier_sha256
    )
    result["verifier"] = {
        "path": BENCHMARK_AUTHORITY_VERIFIER_RELATIVE.as_posix(),
        "function": BENCHMARK_AUTHORITY_VERIFIER_FUNCTION,
        "present": verifier_path.is_file(),
        "sha256": expected_verifier_sha256,
        "bound": verifier_bound,
    }
    if not verifier_path.is_file():
        failures.append("benchmark_authority_verifier_missing")
        return result
    if not verifier_bound:
        failures.append("benchmark_authority_verifier_identity_mismatch")
        return result
    if failures:
        return result

    try:
        import importlib.util

        module_spec = importlib.util.spec_from_file_location(
            "formal_verification_goal_tactician_benchmark_authority",
            verifier_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("cannot load benchmark authority verifier")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        verifier = getattr(
            module,
            BENCHMARK_AUTHORITY_VERIFIER_FUNCTION,
            None,
        )
        if not callable(verifier):
            failures.append("benchmark_authority_verifier_not_callable")
            return result
        verified = verifier(dict(benchmark), repo_root=root)
    except Exception as exc:  # noqa: BLE001 - benchmark evidence fails closed
        failures.append(f"benchmark_authority_verifier_error:{type(exc).__name__}")
        return result

    verified_mapping = _safe_dict(verified)
    if (
        verified_mapping.get("valid") is not True
        or verified_mapping.get("report_id") != report_id
        or verified_mapping.get("authority_content_id")
        != authority.get("content_id")
    ):
        failures.append("benchmark_authority_verifier_rejected")
        return result

    result["bound"] = True
    return result


def _benchmark_hard_gate_evidence(
    benchmark: Mapping[str, Any] | None,
    *,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Classify whether benchmark hard gates are live and content-bound."""

    failures: list[str] = []
    if benchmark is None:
        return {
            "authoritative": False,
            "failures": ["benchmark_missing"],
            "hard_gates": {},
            "evidence_classes": [],
            "report_id": None,
            "report_id_valid": False,
            "authority_anchor": {
                "present": False,
                "bound": False,
                "failures": ["benchmark_missing"],
            },
        }

    report = _safe_dict(benchmark.get("report"))
    metrics = _safe_dict(report.get("metrics"))
    hard_gates = _safe_dict(_safe_dict(report.get("gates")).get("hard"))
    metric_hard_gates = _safe_dict(metrics.get("hard_gates"))
    evidence_classes = sorted(
        {
            str(item).strip().lower()
            for item in _safe_list(metrics.get("evidence_classes"))
            if str(item).strip()
        }
    )
    evidence_class_set = set(evidence_classes)
    report_id = str(report.get("report_id") or "")
    report_id_valid = _declared_mapping_identity_valid(
        report,
        field="report_id",
        prefix="goal-tactician-bench-",
    )

    if benchmark.get("schema") != (
        "ipfs_accelerate_py/agent-supervisor/goal-tactician-benchmark@1"
    ):
        failures.append("benchmark_schema_mismatch")
    if benchmark.get("interface") != "GoalTacticianBenchmark@1":
        failures.append("benchmark_interface_mismatch")
    if report.get("schema") != (
        "ipfs_accelerate_py/agent-supervisor/goal-tactician-benchmark-report@1"
    ):
        failures.append("benchmark_report_schema_mismatch")
    if report.get("interface") != "GoalTacticianBenchmark@1":
        failures.append("benchmark_report_interface_mismatch")
    if report.get("source") != "cohort_receipts":
        failures.append("benchmark_report_source_not_cohort_receipts")
    if metrics.get("source") != "cohort_receipts":
        failures.append("benchmark_metrics_source_not_cohort_receipts")
    if not report_id_valid:
        failures.append("benchmark_report_id_mismatch")
    authority_anchor = _benchmark_authority_anchor(
        benchmark,
        report_id=report_id,
        repo_root=repo_root,
    )
    failures.extend(authority_anchor.get("failures") or [])

    synthetic_markers = (
        benchmark.get("synthetic_distributions"),
        report.get("synthetic_distributions"),
        metrics.get("synthetic_distributions"),
    )
    if any(marker is not False for marker in synthetic_markers):
        failures.append("benchmark_synthetic_status_missing_or_true")
    if not evidence_classes:
        failures.append("benchmark_evidence_classes_missing")
    if evidence_class_set & NON_AUTHORITATIVE_BENCHMARK_EVIDENCE_CLASSES:
        failures.append("benchmark_fixture_or_synthetic_evidence")
    if not evidence_class_set <= AUTHORITATIVE_BENCHMARK_EVIDENCE_CLASSES:
        failures.append("benchmark_evidence_class_not_authoritative")

    receipt_ids = [
        str(item) for item in _safe_list(report.get("receipt_ids")) if str(item)
    ]
    if (
        not receipt_ids
        or len(receipt_ids) != len(set(receipt_ids))
        or _nonnegative_int(report.get("receipt_count")) != len(receipt_ids)
    ):
        failures.append("benchmark_receipt_population_invalid")
    if any(
        any(marker in receipt_id.lower() for marker in ("fixture", "synthetic", "simulated"))
        for receipt_id in receipt_ids
    ):
        failures.append("benchmark_receipt_population_non_authoritative")

    observed_passes: list[bool] = []
    for gate_name in ("correctness", "privacy", "authority"):
        gate = _safe_dict(hard_gates.get(gate_name))
        actual = gate.get("actual_bps")
        required = gate.get("required_bps")
        status = str(gate.get("status") or "").strip().lower()
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or not isinstance(required, int)
            or isinstance(required, bool)
            or required != 10000
            or actual < 0
            or actual > required
            or status not in {"pass", "fail"}
            or (status == "pass") != (actual >= required)
        ):
            failures.append(f"benchmark_hard_gate_{gate_name}_invalid")
            continue
        if metric_hard_gates.get(f"{gate_name}_bps") != actual:
            failures.append(f"benchmark_metric_{gate_name}_mismatch")
        observed_passes.append(actual >= required)
    if (
        len(observed_passes) != 3
        or metric_hard_gates.get("passed") is not all(observed_passes)
    ):
        failures.append("benchmark_aggregate_hard_gate_mismatch")

    return {
        "authoritative": not failures,
        "failures": sorted(set(failures)),
        "hard_gates": hard_gates,
        "evidence_classes": evidence_classes,
        "report_id": report_id or None,
        "report_id_valid": report_id_valid,
        "authority_anchor": authority_anchor,
    }


def _baseline_p0_gate_pressure(
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Map unresolved P0 findings to the hard-zero dimensions they invalidate."""

    gate_pressure = {
        gate: 0
        for gate in HARD_ZERO_GATE_KEYS
        if gate != "unresolved_cross_provider_disagreement_count"
    }
    open_findings: list[dict[str, Any]] = []
    open_p0_findings: list[dict[str, Any]] = []
    failures: list[str] = []

    if baseline is None:
        failures.append("baseline_missing")
        for gate in gate_pressure:
            gate_pressure[gate] = 1
        return {
            "gate_pressure": gate_pressure,
            "open_findings": open_findings,
            "open_p0_findings": open_p0_findings,
            "failures": failures,
        }

    findings = baseline.get("known_findings")
    if not isinstance(findings, list):
        failures.append("baseline_known_findings_missing_or_invalid")
        for gate in gate_pressure:
            gate_pressure[gate] = 1
        return {
            "gate_pressure": gate_pressure,
            "open_findings": open_findings,
            "open_p0_findings": open_p0_findings,
            "failures": failures,
        }

    all_local_gates = tuple(gate_pressure)
    for finding in findings:
        if not isinstance(finding, Mapping):
            failures.append("baseline_finding_not_mapping")
            for gate in gate_pressure:
                gate_pressure[gate] = max(gate_pressure[gate], 1)
            continue
        if str(finding.get("status") or "").strip().lower() != "open":
            continue
        finding_id = str(finding.get("id") or "").strip()
        severity = str(finding.get("severity") or "").strip().lower()
        projected = {
            "id": finding_id or None,
            "severity": severity or None,
            "summary": finding.get("summary"),
        }
        open_findings.append(projected)
        if severity != "p0":
            continue
        gates = P0_FINDING_GATE_MAP.get(finding_id, all_local_gates)
        mapped = {**projected, "mapped_hard_zero_gates": list(gates)}
        open_p0_findings.append(mapped)
        for gate in gates:
            gate_pressure[gate] = gate_pressure.get(gate, 0) + 1

    return {
        "gate_pressure": gate_pressure,
        "open_findings": open_findings,
        "open_p0_findings": open_p0_findings,
        "failures": sorted(set(failures)),
    }


def derive_hard_zero_gates(
    *,
    certificate: Mapping[str, Any] | None,
    benchmark: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Derive hard-zero counters from bound, authoritative child evidence.

    Content-addressed certificate disagreement evidence and a live benchmark
    cohort are required. Fixture/synthetic cohorts never clear deployment
    gates. Open P0 baseline findings apply nonzero pressure until explicitly
    resolved; an unknown P0 conservatively affects every local hard-zero gate.
    """

    missing_measurements: list[str] = []
    disagreement = 1
    certificate_identity_valid = bool(
        certificate is not None
        and certificate.get("interface") == "FormalVerificationToolchainCertificate@1"
        and _declared_mapping_identity_valid(
            certificate,
            field="certificate_digest_sha256",
            ensure_ascii=True,
        )
    )
    if certificate_identity_valid and certificate is not None:
        quarantines = certificate.get("disagreement_quarantines")
        if isinstance(quarantines, list):
            disagreement = len(quarantines)
        else:
            missing_measurements.append("certificate.disagreement_quarantines")
    else:
        if certificate is None:
            missing_measurements.append("certificate")
        missing_measurements.append("certificate.content_identity")

    def _violations_from_bps(status: Mapping[str, Any] | None) -> int:
        if not status:
            return 1
        actual = status.get("actual_bps")
        required = status.get("required_bps")
        if (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and isinstance(required, int)
            and not isinstance(required, bool)
        ):
            return max(0, required - actual)
        return 1

    false_proof = 1
    false_closure = 1
    leakage = 1
    authority = 1
    gate_source = (
        "content_bound_certificate_live_benchmark_and_open_p0_baseline_findings"
    )

    benchmark_evidence = _benchmark_hard_gate_evidence(
        benchmark,
        repo_root=repo_root,
    )
    if benchmark_evidence["authoritative"]:
        hard = _safe_dict(benchmark_evidence.get("hard_gates"))
        false_proof = _violations_from_bps(
            _safe_dict(hard.get("correctness")) or None
        )
        authority = _violations_from_bps(_safe_dict(hard.get("authority")) or None)
        privacy = _violations_from_bps(_safe_dict(hard.get("privacy")) or None)
        leakage = privacy
        if false_proof == 0 and authority == 0 and disagreement == 0:
            false_closure = 0
        else:
            false_closure = max(false_proof, authority, 1 if disagreement else 0)
    else:
        if benchmark is None:
            missing_measurements.append("benchmark")
        missing_measurements.extend(
            f"benchmark.{failure}"
            for failure in benchmark_evidence.get("failures") or []
        )

    baseline_pressure = _baseline_p0_gate_pressure(baseline)
    missing_measurements.extend(
        f"baseline.{failure}"
        for failure in baseline_pressure.get("failures") or []
    )
    pressure = _safe_dict(baseline_pressure.get("gate_pressure"))
    false_proof = max(false_proof, int(pressure.get("false_proof_count") or 0))
    false_closure = max(
        false_closure,
        int(pressure.get("false_closure_count") or 0),
    )
    leakage = max(
        leakage,
        int(pressure.get("secret_or_witness_leakage_count") or 0),
    )
    authority = max(
        authority,
        int(pressure.get("authority_boundary_violations") or 0),
    )
    open_p0_findings = _safe_list(baseline_pressure.get("open_p0_findings"))
    if open_p0_findings:
        missing_measurements.append("baseline.unresolved_open_p0_findings")

    return {
        "false_proof_count": int(false_proof),
        "false_closure_count": int(false_closure),
        "secret_or_witness_leakage_count": int(leakage),
        "authority_boundary_violations": int(authority),
        "unresolved_cross_provider_disagreement_count": int(disagreement),
        "derivation": {
            "source": gate_source,
            "hardcoded_success_counters_forbidden": True,
            "complete": not missing_measurements,
            "missing_measurements": sorted(set(missing_measurements)),
            "certificate_identity_valid": certificate_identity_valid,
            "benchmark_evidence": benchmark_evidence,
            "benchmark_hard_gates_required_bps": 10000,
            "fixture_or_synthetic_benchmark_cannot_clear": True,
            "open_p0_findings_block_clearance": True,
            "open_baseline_findings_disclosed": len(
                baseline_pressure.get("open_findings") or []
            ),
            "open_baseline_findings": baseline_pressure.get("open_findings") or [],
            "open_p0_findings": open_p0_findings,
            "p0_gate_pressure": pressure,
        },
    }


def build_implementation_section(
    *,
    child_goals: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    corpus: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bound_children = [g for g in child_goals if g.get("bound")]
    unbound = [g["goal_id"] for g in child_goals if not g.get("bound")]

    corpus_case_count = 0
    if corpus is not None:
        cases = corpus.get("cases") or corpus.get("case_ids") or []
        if isinstance(cases, list):
            corpus_case_count = len(cases)

    schema_surfaces = [
        key
        for key in (
            "readiness_baseline",
            "toolchain_certificate",
            "toolchain_lock",
            "corpus_manifest",
            "benchmark_report",
            "rollout_policy",
        )
        if artifacts.get(key, {}).get("present")
    ]

    public_ops = all(
        artifacts.get(key, {}).get("present")
        for key in ("public_api_test", "cli_mcp_parity_test")
    )
    metrics_bound = bool(artifacts.get("metrics_module", {}).get("present")) and bool(
        artifacts.get("benchmark_report", {}).get("present")
    )
    rollout_bound = bool(artifacts.get("rollout_policy", {}).get("present"))
    schemas_bound = len(schema_surfaces) >= 4
    corpus_bound = bool(artifacts.get("corpus_manifest", {}).get("present"))

    complete = (
        not unbound
        and schemas_bound
        and corpus_bound
        and public_ops
        and metrics_bound
        and rollout_bound
        and bool(artifacts.get("receipt_builder", {}).get("present"))
        and bool(artifacts.get("completion_test", {}).get("present"))
    )

    return {
        "status": "complete" if complete else "incomplete",
        "description": (
            "Current-tree implementation completion: child goal evidence, "
            "schemas/contracts, golden corpus, public operations, metrics, and "
            "rollout policy are bound by content identity. Implementation is "
            "not deployment certification."
        ),
        "child_goal_count": len(child_goals),
        "child_goals_bound": len(bound_children),
        "child_goals_unbound": unbound,
        "schemas_bound": schemas_bound,
        "schema_surfaces": schema_surfaces,
        "corpus_bound": corpus_bound,
        "corpus_case_count": corpus_case_count,
        "corpus_interface": (corpus or {}).get("interface"),
        "public_operations_bound": public_ops,
        "public_operation_evidence": [
            artifacts["public_api_test"]["path"],
            artifacts["cli_mcp_parity_test"]["path"],
        ]
        if public_ops
        else [],
        "metrics_bound": metrics_bound,
        "rollout_policy_bound": rollout_bound,
        "completion_surfaces_bound": bool(
            artifacts.get("receipt_builder", {}).get("present")
        )
        and bool(artifacts.get("completion_test", {}).get("present")),
        "hardcoded_success_counters": False,
    }


def build_deployment_section(
    *,
    alignment: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    live_report: Mapping[str, Any] | None,
    hard_zero: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_summary = _safe_dict((baseline or {}).get("summary"))
    tools = _safe_list((baseline or {}).get("tools"))
    cert_promotion = _safe_dict((certificate or {}).get("promotion"))
    certified_tools = {
        str(tool.get("tool_id") or ""): tool
        for tool in _safe_list((certificate or {}).get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    }
    live_claims = _safe_dict((live_report or {}).get("production_readiness_claims"))
    evidence_policy = _safe_dict((live_report or {}).get("evidence_class_policy"))

    tool_rows: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        tool_id = str(tool.get("tool_id") or "")
        certification_tool_id = "coq" if tool_id == "coqc" else tool_id
        certification = _safe_dict(certified_tools.get(certification_tool_id))
        statuses = _safe_dict(tool.get("statuses"))
        identity = _safe_dict(tool.get("identity"))
        if certification:
            for key in (
                "installed",
                "usable",
                "production_certified",
                "unavailable",
            ):
                statuses[key] = bool(certification.get(key))
        certified_version = certification.get("version_string")
        baseline_version = identity.get("version_string")
        identity_matches_certificate = bool(
            certified_version and certified_version == baseline_version
        )
        package_or_toolchain = identity.get("package_or_toolchain")
        if certification and certified_version and not identity_matches_certificate:
            locked_version = str(certification.get("locked_version") or "").strip()
            package_or_toolchain = (
                f"{certification_tool_id}@{locked_version}"
                if locked_version
                else None
            )
        tool_rows.append(
            {
                "tool_id": tool_id,
                "certification_tool_id": certification_tool_id,
                "statuses": statuses,
                "executable_path": (
                    certification.get("executable_path")
                    if certification
                    else identity.get("executable_path")
                ),
                "version_string": (
                    certified_version if certification else baseline_version
                ),
                "package_or_toolchain": package_or_toolchain,
                "certification": {
                    key: certification.get(key)
                    for key in (
                        "locked_version",
                        "locked_version_mismatch",
                        "shim_toolchain_mismatch",
                        "identity_probed",
                        "evidence_class",
                        "production_certified",
                        "promotion_blocked",
                        "block_reasons",
                    )
                }
                if certification
                else None,
            }
        )

    # The historical readiness baseline covers only the original tool surface.
    # Role-aware certification adds in-process and newly managed tools that do
    # not have a baseline row.  Retain those exact certified identities here so
    # the deployment inventory and its production-certified summary cannot
    # disagree or silently omit an elevated authority lane.
    represented_certification_ids = {
        str(row.get("certification_tool_id") or "") for row in tool_rows
    }
    for certification_tool_id, certification in sorted(certified_tools.items()):
        if certification_tool_id in represented_certification_ids:
            continue
        locked_version = str(certification.get("locked_version") or "").strip()
        tool_rows.append(
            {
                "tool_id": certification_tool_id,
                "certification_tool_id": certification_tool_id,
                "statuses": {
                    key: bool(certification.get(key))
                    for key in (
                        "installed",
                        "usable",
                        "production_certified",
                        "unavailable",
                    )
                },
                "executable_path": certification.get("executable_path"),
                "version_string": certification.get("version_string"),
                "package_or_toolchain": (
                    f"{certification_tool_id}@{locked_version}"
                    if locked_version
                    else certification.get("version_string")
                ),
                "certification": {
                    key: certification.get(key)
                    for key in (
                        "locked_version",
                        "locked_version_mismatch",
                        "shim_toolchain_mismatch",
                        "identity_probed",
                        "evidence_class",
                        "production_certified",
                        "promotion_blocked",
                        "block_reasons",
                    )
                },
            }
        )

    usable = list(baseline_summary.get("usable_tools") or [])
    unavailable = list(baseline_summary.get("unavailable_tools") or [])
    production_certified = list(
        cert_promotion.get("production_certified_tool_ids") or []
    )
    cert_unavailable = list(cert_promotion.get("unavailable_tool_ids") or [])
    blocked = _safe_dict(cert_promotion.get("blocked_tool_ids"))
    lane_ready = _safe_dict(cert_promotion.get("lane_promotion_ready"))

    # Classify live / simulated / skipped from live report lanes when present.
    live_cases = 0
    simulated_cases = 0
    skipped_cases = 0
    unsupported: list[str] = []
    if live_report is not None:
        for case in _safe_list(live_report.get("cases")):
            if not isinstance(case, Mapping):
                continue
            evidence_class = str(
                case.get("evidence_class") or case.get("class") or ""
            ).lower()
            if evidence_class in {"live"}:
                live_cases += 1
            elif evidence_class in {"simulated", "fixture", "offline"}:
                simulated_cases += 1
            elif evidence_class in {"skipped", "unavailable"}:
                skipped_cases += 1
            if evidence_class == "unsupported" or case.get("unsupported"):
                unsupported.append(str(case.get("case_id") or case.get("id") or evidence_class))
        for lane in _safe_list(live_report.get("lanes")):
            if not isinstance(lane, Mapping):
                continue
            if lane.get("unsupported"):
                unsupported.append(str(lane.get("lane_id") or "unsupported_lane"))

    # Assurance ceiling: deployment is certified only for tools with hermetic
    # production certificates; PATH presence never elevates the ceiling.
    hard_zero_clear = all(int(hard_zero.get(key) or 0) == 0 for key in HARD_ZERO_GATE_KEYS)
    publication_clear = not alignment.get("publication_gates", {}).get(
        "datasets_publication_lag"
    ) and not alignment.get("publication_gates", {}).get("parent_publication_lag")
    machine_certified = bool(production_certified) and hard_zero_clear

    return {
        "status": (
            "machine_specific_partial"
            if machine_certified and (unavailable or cert_unavailable or not publication_clear)
            else ("machine_specific_certified" if machine_certified and publication_clear else "not_deployment_certified")
        ),
        "description": (
            "Machine-specific deployment certification binds exact tool "
            "identities, live/simulated/skipped evidence classes, publication "
            "alignment, and assurance ceilings. Implementation completeness "
            "never implies production certification."
        ),
        "tree_alignment": alignment,
        "exact_tools": tool_rows,
        "usable_tools": usable,
        "unavailable_tools": sorted(set(unavailable) | set(cert_unavailable)),
        "production_certified_tool_ids": production_certified,
        "blocked_tool_ids": blocked,
        "lane_promotion_ready": lane_ready,
        "live_simulated_skipped": {
            "live_case_count": live_cases,
            "simulated_or_fixture_case_count": simulated_cases,
            "skipped_or_unavailable_case_count": skipped_cases,
            "evidence_class_policy": evidence_policy,
            "production_readiness_claims": live_claims,
        },
        "unsupported_semantics": sorted(set(unsupported)),
        "publication_gates": alignment.get("publication_gates"),
        "assurance_ceilings": {
            "path_presence_is_not_usability": True,
            "source_presence_is_not_usability": True,
            "fixture_is_not_production_certified": True,
            "live_without_hermetic_certificate_is_not_production_certified": True,
            "lfv_completion_is_not_deployment_certificate": True,
            "synthetic_evidence_cannot_certify_production": True,
            "production_certified_count": len(production_certified),
            "baseline_production_certified_count": int(
                baseline_summary.get("production_certified_count") or 0
            ),
            "machine_certified": machine_certified,
            "publication_aligned": publication_clear,
            "hard_zero_clear": hard_zero_clear,
        },
        "hardcoded_success_counters": False,
    }


def build_acceptance(
    *,
    implementation: Mapping[str, Any],
    deployment: Mapping[str, Any],
    hard_zero: Mapping[str, Any],
    child_goals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hard_zero_clear = all(int(hard_zero.get(key) or 0) == 0 for key in HARD_ZERO_GATE_KEYS)
    return {
        "implementation_complete": implementation.get("status") == "complete",
        "deployment_section_present": True,
        "implementation_section_present": True,
        "child_goal_count": len(child_goals),
        "child_goals_bound": implementation.get("child_goals_bound"),
        "schemas_corpus_public_ops_metrics_rollout_bound": all(
            [
                implementation.get("schemas_bound"),
                implementation.get("corpus_bound"),
                implementation.get("public_operations_bound"),
                implementation.get("metrics_bound"),
                implementation.get("rollout_policy_bound"),
            ]
        ),
        "parent_datasets_publication_bound": True,
        "exact_tools_bound": bool(deployment.get("exact_tools") is not None),
        "live_simulated_skipped_bound": bool(deployment.get("live_simulated_skipped")),
        "hard_zero_gates_clear": hard_zero_clear,
        "hardcoded_success_counters": False,
        "false_proof_count": hard_zero.get("false_proof_count"),
        "false_closure_count": hard_zero.get("false_closure_count"),
        "secret_or_witness_leakage_count": hard_zero.get(
            "secret_or_witness_leakage_count"
        ),
        "authority_boundary_violations": hard_zero.get("authority_boundary_violations"),
        "unresolved_cross_provider_disagreement_count": hard_zero.get(
            "unresolved_cross_provider_disagreement_count"
        ),
    }


def _load_certifier_module(repo_root: Path) -> Any:
    """Load the multi-prover certifier without requiring package install."""

    import importlib.util

    path = repo_root / DEFAULT_CERTIFIER_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"missing certifier: {path}")
    spec = importlib.util.spec_from_file_location(
        "certify_formal_verification_toolchains",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load certifier from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity_matches(
    payload: Mapping[str, Any],
    field: str,
    *,
    prefixed: bool,
) -> bool:
    stored = str(payload.get(field) or "")
    body = {key: value for key, value in payload.items() if key != field}
    computed = content_digest(body)
    if not prefixed:
        computed = computed.removeprefix("sha256:")
    return bool(stored) and stored == computed


def _dirty_paths(repo_root: Path) -> list[str]:
    porcelain = _git_stdout(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        allow_empty=True,
    ) or ""
    paths: list[str] = []
    for line in porcelain.splitlines():
        # `_git_stdout` trims the complete command output.  When the first
        # porcelain row has an unstaged-only status (` M path`), that removes
        # its leading status-space.  Accept both the canonical `XY path` form
        # and that first-row `Y path` form without dropping the first filename
        # character (notably the dot in `.gitignore`).
        if len(line) >= 3 and line[2] == " ":
            raw = line[3:]
        elif len(line) >= 2 and line[1] == " ":
            raw = line[2:]
        else:
            raw = ""
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        if raw:
            paths.append(raw.strip('"'))
    return sorted(set(paths))


def build_source_attestation(repo_root: Path) -> dict[str, Any]:
    """Bind a source commit, never the not-yet-published receipt itself."""

    alignment = observe_tree_alignment(repo_root)
    dirty = _dirty_paths(repo_root)
    non_attestation_dirty = sorted(
        set(dirty) - set(ROLE_AWARE_ATTESTATION_PATHS)
    )
    parent = _safe_dict(alignment.get("parent"))
    gitlink = _safe_dict(alignment.get("gitlink"))
    embedded = _safe_dict(alignment.get("embedded_checkout"))
    source_commit_bound = bool(
        parent.get("commit")
        and parent.get("tree")
        and gitlink.get("commit")
        and gitlink.get("commit") == embedded.get("head")
        and embedded.get("clean")
    )
    source_candidate_clean = not dirty
    return {
        "model": "two_phase_source_then_attestation_publication/v1",
        "certified_source_commit": parent.get("commit"),
        "certified_source_tree": parent.get("tree"),
        "datasets_gitlink": gitlink.get("commit"),
        "datasets_embedded_head": embedded.get("head"),
        "source_commit_bound": source_commit_bound,
        "source_candidate_clean": source_candidate_clean,
        "dirty_paths_at_certification": dirty,
        "non_attestation_dirty_paths": non_attestation_dirty,
        "attestation_paths": sorted(ROLE_AWARE_ATTESTATION_PATHS),
        "attestation_excluded_from_source_tree": True,
        "publication_verification_required": True,
        "publication_rule": (
            "Publish the generated attestation in a descendant commit whose "
            "certified-source ancestor is this commit and whose attestation "
            "diff is restricted to declared generated receipt/certificate paths."
        ),
        "valid_for_attestation": bool(
            source_commit_bound and (source_candidate_clean or not non_attestation_dirty)
        ),
        "tree_alignment": alignment,
    }


def load_supervisor_evidence_snapshot(
    *,
    task_state_path: Path,
    event_log_path: Path,
    task_id: str = ROLE_AWARE_TASK_ID,
) -> dict[str, Any]:
    """Read durable supervisor evidence without mutating JSON, logs, or DuckDB."""

    state = load_json(task_state_path) or {}
    relevant_events: list[dict[str, Any]] = []
    chain_errors: list[str] = []
    previous_event_id = ""
    previous_sequence = 0
    stream_id = ""
    snapshot_id = ""
    canonical_event_count = 0
    if event_log_path.is_file():
        for line_number, raw_line in enumerate(
            event_log_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                chain_errors.append(f"line_{line_number}:malformed_json")
                continue
            if not isinstance(event, Mapping):
                chain_errors.append(f"line_{line_number}:non_object_event")
                continue
            event = dict(event)
            sequence = event.get("sequence")
            observed_event_id = str(event.get("event_id") or "")
            expected_event_id = content_digest(
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_id"
                }
            )
            observed_stream = str(event.get("stream_id") or "")
            observed_snapshot = str(event.get("snapshot_id") or "")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence != previous_sequence + 1
            ):
                chain_errors.append(f"line_{line_number}:sequence_not_contiguous")
            if observed_event_id != expected_event_id:
                chain_errors.append(f"line_{line_number}:event_id_not_canonical")
            if previous_sequence and str(event.get("previous_event_id") or "") != (
                previous_event_id
            ):
                chain_errors.append(f"line_{line_number}:previous_event_id_mismatch")
            if not previous_sequence and str(event.get("previous_event_id") or ""):
                chain_errors.append(
                    f"line_{line_number}:first_previous_event_id_not_empty"
                )
            if not observed_stream or not observed_snapshot:
                chain_errors.append(f"line_{line_number}:stream_identity_missing")
            elif canonical_event_count:
                if observed_stream != stream_id or observed_snapshot != snapshot_id:
                    chain_errors.append(
                        f"line_{line_number}:stream_identity_changed"
                    )
            else:
                stream_id = observed_stream
                snapshot_id = observed_snapshot
            previous_sequence = sequence if isinstance(sequence, int) else previous_sequence
            previous_event_id = observed_event_id
            canonical_event_count += 1

            if str(event.get("task_id") or "") != task_id:
                continue
            completion_receipts: list[dict[str, Any]] = []
            for source in (
                event.get("completion_receipts"),
                _safe_dict(event.get("todo_update_result")).get(
                    "completion_receipts"
                ),
            ):
                for receipt in _safe_list(source):
                    if isinstance(receipt, Mapping):
                        completion_receipts.append(dict(receipt))
            merge = _safe_dict(event.get("merge_result"))
            validation = _safe_dict(event.get("validation_result"))
            relevant_events.append(
                {
                    "sequence": event.get("sequence"),
                    "event_id": event.get("event_id"),
                    "previous_event_id": event.get("previous_event_id"),
                    "snapshot_id": event.get("snapshot_id"),
                    "stream_id": event.get("stream_id"),
                    "type": event.get("type"),
                    "timestamp": event.get("timestamp"),
                    "task_id": event.get("task_id"),
                    "canonical_task_cid": (
                        event.get("canonical_task_cid") or event.get("task_cid")
                    ),
                    "canonical_task_key": event.get("canonical_task_key"),
                    "implementation_commit": event.get("implementation_commit"),
                    "validation": {
                        "attempted": validation.get("attempted"),
                        "passed": validation.get("passed"),
                        "returncode": validation.get("returncode"),
                        "target_commit": validation.get("target_commit"),
                        "receipt_id": validation.get("receipt_id"),
                    }
                    if validation
                    else {},
                    "merge": {
                        "merged": merge.get("merged"),
                        "implementation_commit": merge.get(
                            "implementation_commit"
                        ),
                        "merge_commit": merge.get("merge_commit"),
                        "target_branch": merge.get("target_branch"),
                        "integration_commit_proof": merge.get(
                            "integration_commit_proof"
                        ),
                    }
                    if merge
                    else {},
                    "completion_receipts": completion_receipts,
                }
            )

    identity = _safe_dict(_safe_dict(state.get("task_identities")).get(task_id))
    return {
        "schema_version": "formal-verification-supervisor-evidence-snapshot/v1",
        "task_id": task_id,
        "lane_id": task_state_path.parent.parent.name,
        "task_state_source": {
            "path": str(task_state_path),
            "sha256": sha256_file(task_state_path),
        },
        "event_log_source": {
            "path": str(event_log_path),
            "sha256": sha256_file(event_log_path),
        },
        "event_chain": {
            "valid": bool(canonical_event_count) and not chain_errors,
            "event_count": canonical_event_count,
            "last_sequence": previous_sequence,
            "last_event_id": previous_event_id or None,
            "stream_id": stream_id or None,
            "snapshot_id": snapshot_id or None,
            "errors": chain_errors,
        },
        "task_state": {
            "active_task_id": state.get("active_task_id"),
            "active_task_cid": state.get("active_task_cid"),
            "active_task_key": state.get("active_task_key"),
            "implementation_in_progress": state.get(
                "implementation_in_progress"
            ),
            "last_implementation_task_id": state.get(
                "last_implementation_task_id"
            ),
            "last_implementation_task_cid": state.get(
                "last_implementation_task_cid"
            ),
            "last_implementation_commit": state.get(
                "last_implementation_commit"
            ),
            "last_merge_commit": state.get("last_merge_commit"),
            "task_status": _safe_dict(state.get("task_statuses")).get(task_id),
            "canonical_identity": identity,
        },
        "events": relevant_events,
    }


def load_supervisor_release_evidence(path: Path) -> dict[str, Any]:
    """Load one explicit content-addressed G212 export.

    Raw task-state and event files intentionally use the separate diagnostic
    loader above and never pass through this API as release authority.
    """

    payload = load_json(path.resolve())
    if not isinstance(payload, Mapping):
        raise ValueError("supervisor release evidence must be one JSON object")
    return dict(payload)


def finalize_supervisor_release_evidence(
    *,
    evidence_path: Path,
    repo_root: Path,
    integration_branch: str = ROLE_AWARE_INTEGRATION_BRANCH,
) -> dict[str, Any]:
    """Recheck a provisional export after its merge is published.

    Callers must fetch ``origin`` before invoking this function. The returned
    binding becomes final only when the recorded merge is an ancestor of the
    local ``refs/remotes/origin/main``; no field in the export can self-assert
    publication.
    """

    evidence = load_supervisor_release_evidence(evidence_path)
    return derive_supervisor_binding(
        evidence,
        repo_root=repo_root.resolve(),
        integration_branch=integration_branch,
    )


def _derive_git_commit_binding(
    *,
    repo_root: Path | None,
    implementation_commit: str,
    merge_commit: str,
    target_branch: str,
    integration_proof: Mapping[str, Any],
    integration_branch: str = ROLE_AWARE_INTEGRATION_BRANCH,
) -> dict[str, Any]:
    """Independently bind supervisor claims to the local and published Git DAG.

    ``provisional_valid`` proves the completed implementation was directly
    merged, that the merge tree is exactly the implementation tree, and that
    the recorded target is the configured integration branch (or an
    ``origin/main`` alias). ``valid`` additionally requires the merge to be an
    ancestor of ``refs/remotes/origin/main``. This is the deliberate two-phase
    publication boundary.
    """

    result: dict[str, Any] = {
        "valid": False,
        "provisional_valid": False,
        "repository_bound": False,
        "implementation_commit_exists": False,
        "merge_commit_exists": False,
        "implementation_is_ancestor": False,
        "implementation_is_direct_parent": False,
        "source_trees_bound": False,
        "merge_tree_matches_implementation": False,
        "integration_proof_bound": False,
        "target_branch_bound": False,
        "target_ref_contains_merge": False,
        "published_to_origin_main": False,
        "implementation_commit": implementation_commit or None,
        "merge_commit": merge_commit or None,
        "implementation_tree": None,
        "merge_tree": None,
        "publication_ref": "refs/remotes/origin/main",
        "configured_integration_branch": integration_branch,
        "failures": [],
    }
    failures: list[str] = result["failures"]
    if repo_root is None:
        failures.append("supervisor_repository_root_missing")
        return result
    root = repo_root.resolve()
    result["repository_bound"] = (
        _git_stdout(root, "rev-parse", "--show-toplevel") == str(root)
    )
    if not result["repository_bound"]:
        failures.append("supervisor_repository_not_bound")
        return result

    resolved_implementation = _git_stdout(
        root,
        "rev-parse",
        "--verify",
        f"{implementation_commit}^{{commit}}",
    )
    resolved_merge = _git_stdout(
        root,
        "rev-parse",
        "--verify",
        f"{merge_commit}^{{commit}}",
    )
    result["implementation_commit_exists"] = (
        bool(COMMIT_RE.fullmatch(implementation_commit))
        and resolved_implementation == implementation_commit
    )
    result["merge_commit_exists"] = (
        bool(COMMIT_RE.fullmatch(merge_commit))
        and resolved_merge == merge_commit
    )
    if not result["implementation_commit_exists"]:
        failures.append("implementation_commit_unreachable")
    if not result["merge_commit_exists"]:
        failures.append("merge_commit_unreachable")
    if not (
        result["implementation_commit_exists"]
        and result["merge_commit_exists"]
    ):
        return result

    implementation_tree = _git_stdout(
        root, "rev-parse", f"{implementation_commit}^{{tree}}"
    )
    merge_tree = _git_stdout(root, "rev-parse", f"{merge_commit}^{{tree}}")
    result["implementation_tree"] = implementation_tree
    result["merge_tree"] = merge_tree
    result["implementation_is_ancestor"] = (
        _git(root, "merge-base", "--is-ancestor", implementation_commit, merge_commit)
        or subprocess.CompletedProcess([], 1)
    ).returncode == 0
    if not result["implementation_is_ancestor"]:
        failures.append("implementation_not_ancestor_of_merge")

    parent_line = _git_stdout(root, "rev-list", "--parents", "-n", "1", merge_commit)
    merge_parents = (
        parent_line.split()[1:]
        if isinstance(parent_line, str) and parent_line
        else []
    )
    result["merge_parents"] = merge_parents
    result["implementation_is_direct_parent"] = (
        implementation_commit in merge_parents
    )
    if not result["implementation_is_direct_parent"]:
        failures.append("implementation_not_direct_merge_parent")

    claimed_implementation_tree = str(
        integration_proof.get("implementation_tree") or ""
    )
    claimed_merge_tree = str(integration_proof.get("merge_tree") or "")
    optional_tree_claims_match = bool(
        (not claimed_implementation_tree or claimed_implementation_tree == implementation_tree)
        and (not claimed_merge_tree or claimed_merge_tree == merge_tree)
    )
    result["merge_tree_matches_implementation"] = bool(
        implementation_tree and merge_tree == implementation_tree
    )
    result["source_trees_bound"] = bool(
        COMMIT_RE.fullmatch(str(implementation_tree or ""))
        and COMMIT_RE.fullmatch(str(merge_tree or ""))
        and optional_tree_claims_match
        and result["merge_tree_matches_implementation"]
    )
    if not result["source_trees_bound"]:
        failures.append("commit_source_trees_not_bound")

    configured_ref = f"refs/heads/{integration_branch}"
    accepted_targets = {
        integration_branch,
        configured_ref,
        "origin/main",
        "refs/remotes/origin/main",
    }
    result["accepted_target_branches"] = sorted(accepted_targets)
    result["target_branch_bound"] = target_branch in accepted_targets
    if not result["target_branch_bound"]:
        failures.append("merge_target_not_configured_integration_branch")

    proof_target = str(integration_proof.get("target_branch") or "")
    proof_ref = str(integration_proof.get("integration_ref") or "")
    proof_ref_commit = (
        _git_stdout(root, "rev-parse", "--verify", f"{proof_ref}^{{commit}}")
        if proof_ref
        else None
    )
    result["integration_proof_bound"] = bool(
        integration_proof.get("passed") is True
        and not _safe_list(integration_proof.get("reasons"))
        and str(integration_proof.get("implementation_commit") or "")
        == implementation_commit
        and str(
            integration_proof.get("integration_commit")
            or integration_proof.get("merge_commit")
            or ""
        )
        == merge_commit
        and proof_target == target_branch
        and proof_target in accepted_targets
        and (proof_ref == merge_commit or proof_ref_commit == merge_commit)
    )
    if not result["integration_proof_bound"]:
        failures.append("integration_commit_proof_not_bound")

    target_ref = (
        configured_ref
        if target_branch in {integration_branch, configured_ref}
        else "refs/remotes/origin/main"
    )
    target_ref_commit = _git_stdout(
        root,
        "rev-parse",
        "--verify",
        f"{target_ref}^{{commit}}",
    )
    target_contains = _git(
        root,
        "merge-base",
        "--is-ancestor",
        merge_commit,
        target_ref,
    )
    result["target_ref"] = target_ref
    result["target_ref_commit"] = target_ref_commit
    result["target_ref_contains_merge"] = bool(
        target_ref_commit and target_contains and target_contains.returncode == 0
    )
    if not result["target_ref_contains_merge"]:
        failures.append("merge_commit_not_on_configured_target_ref")

    published_ref = _git_stdout(
        root,
        "rev-parse",
        "--verify",
        "refs/remotes/origin/main^{commit}",
    )
    result["published_to_origin_main"] = bool(
        published_ref
        and (
            _git(
                root,
                "merge-base",
                "--is-ancestor",
                merge_commit,
                "refs/remotes/origin/main",
            )
            or subprocess.CompletedProcess([], 1)
        ).returncode
        == 0
    )
    result["published_ref_commit"] = published_ref
    if not result["published_to_origin_main"]:
        failures.append("merge_commit_not_published_to_origin_main")

    publication_failure = "merge_commit_not_published_to_origin_main"
    result["provisional_valid"] = not [
        failure for failure in failures if failure != publication_failure
    ]
    result["valid"] = bool(
        result["provisional_valid"] and result["published_to_origin_main"]
    )
    return result


def _trusted_release_evidence_snapshot(
    evidence: Mapping[str, Any] | None,
    *,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Verify a canonical G212 export before exposing its projected snapshot.

    Raw task-state/event JSON is operational state, not release authority. Only
    the repository-bound G212 exporter may turn those mutable files into a
    content-addressed release-evidence object, and its verifier must approve the
    complete object. Until that exporter exists, this gate remains closed.
    """

    payload = dict(evidence) if isinstance(evidence, Mapping) else {}
    failures: list[str] = []
    result: dict[str, Any] = {
        "present": bool(payload),
        "bound": False,
        "schema": payload.get("schema"),
        "interface": payload.get("interface"),
        "goal_id": payload.get("goal_id"),
        "content_id": payload.get("content_id"),
        "exporter": {},
        "snapshot": {},
        "failures": failures,
    }
    if not payload:
        failures.append("trusted_release_evidence_missing")
        return result
    if payload.get("schema") != SUPERVISOR_RELEASE_EVIDENCE_SCHEMA:
        failures.append("trusted_release_evidence_schema_mismatch")
    if payload.get("interface") != SUPERVISOR_RELEASE_EVIDENCE_INTERFACE:
        failures.append("trusted_release_evidence_interface_mismatch")
    if payload.get("goal_id") != SUPERVISOR_RELEASE_EVIDENCE_GOAL_ID:
        failures.append("trusted_release_evidence_goal_mismatch")
    if any(
        key in payload
        for key in ("task_state_source", "event_log_source", "task_state", "events")
    ):
        failures.append("raw_supervisor_state_is_not_release_evidence")
    if not _declared_mapping_identity_valid(payload, field="content_id"):
        failures.append("trusted_release_evidence_content_id_mismatch")
    if repo_root is None:
        failures.append("trusted_release_evidence_repository_missing")
        return result

    root = repo_root.resolve()
    exporter_path = root / SUPERVISOR_RELEASE_EVIDENCE_EXPORTER_RELATIVE
    exporter = _safe_dict(payload.get("exporter"))
    expected_exporter_sha256 = sha256_file(exporter_path)
    exporter_bound = bool(
        exporter.get("path")
        == SUPERVISOR_RELEASE_EVIDENCE_EXPORTER_RELATIVE.as_posix()
        and expected_exporter_sha256
        and exporter.get("sha256") == expected_exporter_sha256
    )
    result["exporter"] = {
        "path": SUPERVISOR_RELEASE_EVIDENCE_EXPORTER_RELATIVE.as_posix(),
        "present": exporter_path.is_file(),
        "sha256": expected_exporter_sha256,
        "bound": exporter_bound,
    }
    if not exporter_path.is_file():
        failures.append("trusted_release_evidence_exporter_missing")
        return result
    if not exporter_bound:
        failures.append("trusted_release_evidence_exporter_identity_mismatch")
        return result

    try:
        import importlib.util

        module_spec = importlib.util.spec_from_file_location(
            "formal_verification_g212_release_evidence",
            exporter_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("cannot load G212 release-evidence exporter")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        verifier = getattr(module, "verify_release_evidence", None)
        if not callable(verifier):
            raise AttributeError("verify_release_evidence is not callable")
        verified = verifier(dict(payload), repo_root=root)
    except Exception as exc:  # noqa: BLE001 - release evidence fails closed
        failures.append(
            f"trusted_release_evidence_verifier_error:{type(exc).__name__}"
        )
        return result

    if isinstance(verified, Mapping):
        verifier_valid = verified.get("valid") is True
        verified_snapshot = verified.get("snapshot")
    else:
        verifier_valid = verified is True
        verified_snapshot = payload.get("snapshot")
    snapshot = (
        dict(verified_snapshot)
        if isinstance(verified_snapshot, Mapping)
        else {}
    )
    if not verifier_valid:
        failures.append("trusted_release_evidence_verifier_rejected")
    if not snapshot:
        failures.append("trusted_release_evidence_snapshot_missing")
    if failures:
        return result

    result["bound"] = True
    result["snapshot"] = snapshot
    return result


def _derive_validation_dag_binding(
    validation: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Validate the completed supervisor validation-DAG schema.

    The validation runner executes against a sealed candidate overlay whose
    recorded ``target_commit`` is the baseline commit. Candidate identity is
    carried separately by ``candidate_binding``. Requiring
    ``target_commit == implementation_commit`` would reject the actual durable
    schema, so this verifier instead binds the baseline Git object, proposal,
    candidate fingerprint, complete selected DAG, and successful mandatory
    nodes.
    """

    data = _safe_dict(validation)
    dag = _safe_dict(data.get("validation_dag_receipt"))
    candidate = _safe_dict(data.get("candidate_binding"))
    proposal = _safe_dict(data.get("proposal_gate"))
    nodes = [
        dict(item)
        for item in _safe_list(dag.get("nodes"))
        if isinstance(item, Mapping)
    ]
    selected_ids = {
        str(item) for item in _safe_list(dag.get("selected_node_ids")) if item
    }
    selected_nodes = [
        node for node in nodes if str(node.get("node_id") or "") in selected_ids
    ]
    required_validation_ids = {
        str(item)
        for item in _safe_list(dag.get("required_validation_ids"))
        if item
    }
    selected_validation_ids = {
        str(node.get("validation_id") or "")
        for node in selected_nodes
        if node.get("validation_id")
    }
    current_fingerprint = str(candidate.get("current_fingerprint") or "")
    expected_fingerprint = str(candidate.get("expected_fingerprint") or "")
    target_commit = str(data.get("target_commit") or "")
    target_exists = bool(
        repo_root is not None
        and COMMIT_RE.fullmatch(target_commit)
        and _git_stdout(
            repo_root.resolve(),
            "rev-parse",
            "--verify",
            f"{target_commit}^{{commit}}",
        )
        == target_commit
    )
    proposal_receipt_id = str(proposal.get("receipt_id") or "")
    dag_proposal_receipt_id = str(dag.get("proposal_receipt_id") or "")
    authority_gates = [
        item
        for item in _safe_list(data.get("authority_gates"))
        if isinstance(item, Mapping)
    ]
    dag_authority_gates = [
        item
        for item in _safe_list(dag.get("authority_gates"))
        if isinstance(item, Mapping)
    ]

    checks = {
        "validation_passed": bool(
            data.get("attempted") is True
            and data.get("passed") is True
            and data.get("returncode") == 0
        ),
        "candidate_binding_bound": bool(
            candidate.get("verified") is True
            and current_fingerprint == expected_fingerprint
            and SHA256_RE.fullmatch(
                current_fingerprint.removeprefix("sha256:")
            )
        ),
        "proposal_gate_bound": bool(
            proposal.get("attempted") is True
            and proposal.get("accepted") is True
            and proposal_receipt_id
            and proposal_receipt_id == dag_proposal_receipt_id
            and not _safe_list(proposal.get("reason_codes"))
        ),
        "dag_identity_bound": bool(
            dag.get("schema") == SUPERVISOR_VALIDATION_DAG_SCHEMA
            and dag.get("objective_id") == ROLE_AWARE_GOAL_ID
            and SHA256_RE.fullmatch(str(dag.get("receipt_id") or ""))
            and SHA256_RE.fullmatch(str(dag.get("graph_id") or ""))
        ),
        "dag_coverage_bound": bool(
            dag.get("passed") is True
            and dag.get("coverage_complete") is True
            and dag.get("uncovered_impact") is False
            and not _safe_list(data.get("coverage_errors"))
        ),
        "dag_nodes_bound": bool(
            nodes
            and selected_ids
            and len(selected_nodes) == len(selected_ids)
            and all(
                node.get("selected") is True
                and node.get("mandatory") is True
                and node.get("disposition") == "succeeded"
                and node.get("returncode") == 0
                and SHA256_RE.fullmatch(str(node.get("node_id") or ""))
                and SHA256_RE.fullmatch(str(node.get("result_digest") or ""))
                for node in selected_nodes
            )
            and required_validation_ids == selected_validation_ids
        ),
        "baseline_commit_bound": bool(
            target_exists
            and dag.get("repository_tree_id") == target_commit
            and _safe_dict(dag.get("impact_graph")).get("repository_tree_id")
            == target_commit
            and proposal.get("repository_tree_id") == target_commit
        ),
        "non_authoritative_ceiling_preserved": bool(
            data.get("authoritative") is False
            and data.get("completion_authoritative") is False
            and data.get("code_proof_authoritative") is False
            and data.get("proof_authoritative") is False
            and data.get("freshness_authoritative") is False
            and dag.get("completion_authoritative") is False
            and dag.get("code_proof_authoritative") is False
            and dag.get("proof_authoritative") is False
            and authority_gates
            and dag_authority_gates
            and all(
                gate.get("disposition") == "pending"
                and gate.get("reason")
                == "validation_passed_requires_independent_authority"
                for gate in [*authority_gates, *dag_authority_gates]
            )
        ),
    }
    failures = [
        name for name, satisfied in checks.items() if not satisfied
    ]
    return {
        "valid": not failures,
        "checks": checks,
        "failures": failures,
        "target_commit": target_commit or None,
        "candidate_fingerprint": current_fingerprint or None,
        "dag_receipt_id": dag.get("receipt_id"),
        "dag_graph_id": dag.get("graph_id"),
        "selected_node_ids": sorted(selected_ids),
        "required_validation_ids": sorted(required_validation_ids),
        "supervisor_execution_authoritative": False,
        "authority_gates": authority_gates,
    }


def derive_supervisor_binding(
    evidence: Mapping[str, Any] | None,
    *,
    repo_root: Path | None = None,
    integration_branch: str = ROLE_AWARE_INTEGRATION_BRANCH,
) -> dict[str, Any]:
    """Validate the exact FVT-083 terminal, merge, and publication evidence."""

    trusted_release = _trusted_release_evidence_snapshot(
        evidence,
        repo_root=repo_root,
    )
    snapshot = _safe_dict(trusted_release.get("snapshot"))
    task_state = _safe_dict(snapshot.get("task_state"))
    identity = _safe_dict(task_state.get("canonical_identity"))
    expected_cid = str(identity.get("canonical_task_cid") or "")
    expected_key = str(identity.get("canonical_task_key") or "")
    expected_identity_bound = bool(
        identity.get("task_id") == ROLE_AWARE_TASK_ID
        and expected_cid == ROLE_AWARE_CANONICAL_TASK_CID
        and expected_key == ROLE_AWARE_CANONICAL_TASK_KEY
        and snapshot.get("task_id") == ROLE_AWARE_TASK_ID
    )
    source_files_bound = trusted_release.get("bound") is True
    durable_snapshot_matches = trusted_release.get("bound") is True

    chain = _safe_dict(snapshot.get("event_chain"))
    event_chain_bound = bool(
        source_files_bound
        and durable_snapshot_matches
        and chain.get("valid") is True
        and int(chain.get("event_count") or 0) > 0
        and SHA256_RE.fullmatch(
            str(chain.get("last_event_id") or "").removeprefix("sha256:")
        )
        and not _safe_list(chain.get("errors"))
    )
    events = [
        dict(event)
        for event in _safe_list(snapshot.get("events"))
        if isinstance(event, Mapping)
    ]
    events_have_canonical_identity = bool(events) and all(
        (
            not str(event.get("task_id") or "").strip()
            or (
                str(event.get("task_id") or "") == ROLE_AWARE_TASK_ID
                and str(event.get("canonical_task_cid") or "")
                == ROLE_AWARE_CANONICAL_TASK_CID
                and str(event.get("canonical_task_key") or "")
                == ROLE_AWARE_CANONICAL_TASK_KEY
            )
        )
        and isinstance(event.get("sequence"), int)
        and not isinstance(event.get("sequence"), bool)
        and SHA256_RE.fullmatch(
            str(event.get("event_id") or "").removeprefix("sha256:")
        )
        for event in events
    )
    successful_receipts = [
        dict(receipt)
        for receipt in _safe_list(snapshot.get("member_completion_receipts"))
        if isinstance(receipt, Mapping)
        and receipt.get("schema") == SUPERVISOR_COMPLETION_SCHEMA
        and receipt.get("status") == "succeeded"
        and receipt.get("task_id") == ROLE_AWARE_TASK_ID
        and receipt.get("canonical_task_cid")
        == ROLE_AWARE_CANONICAL_TASK_CID
        and receipt.get("canonical_task_key")
        == ROLE_AWARE_CANONICAL_TASK_KEY
    ]
    receipt_events = [
        event
        for event in events
        if event.get("type") == "todo_status_updated"
        and event.get("task_id") == ROLE_AWARE_TASK_ID
        and any(
            isinstance(receipt, Mapping)
            and receipt.get("schema") == SUPERVISOR_COMPLETION_SCHEMA
            and receipt.get("status") == "succeeded"
            and receipt.get("task_id") == ROLE_AWARE_TASK_ID
            and receipt.get("canonical_task_cid")
            == ROLE_AWARE_CANONICAL_TASK_CID
            and receipt.get("canonical_task_key")
            == ROLE_AWARE_CANONICAL_TASK_KEY
            for receipt in _safe_list(event.get("completion_receipts"))
        )
    ]
    finished_events = [
        event
        for event in events
        if event.get("type") == "implementation_finished"
        and event.get("task_id") == ROLE_AWARE_TASK_ID
    ]
    terminal_events: list[Mapping[str, Any]] = []
    terminal_commit_bindings: list[Mapping[str, Any]] = []
    terminal_validation_bindings: list[Mapping[str, Any]] = []
    for event in finished_events:
        validation = _safe_dict(event.get("validation"))
        merge = _safe_dict(event.get("merge"))
        implementation_commit = str(
            event.get("implementation_commit")
            or merge.get("implementation_commit")
            or ""
        )
        merge_commit = str(merge.get("merge_commit") or "")
        commit_binding = _derive_git_commit_binding(
            repo_root=repo_root,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            target_branch=str(merge.get("target_branch") or ""),
            integration_proof=_safe_dict(merge.get("integration_commit_proof")),
            integration_branch=integration_branch,
        )
        validation_binding = _derive_validation_dag_binding(
            validation,
            repo_root=repo_root,
        )
        output_invariant = _safe_dict(
            merge.get("post_merge_declared_output_invariant")
        )
        invariant_checks = [
            item
            for item in _safe_list(output_invariant.get("checks"))
            if isinstance(item, Mapping)
        ]
        output_invariant_bound = bool(
            output_invariant.get("passed") is True
            and output_invariant.get("mode") == "repository_tree"
            and output_invariant.get("repository_ref") == merge_commit
            and _safe_list(output_invariant.get("task_ids"))
            == [ROLE_AWARE_TASK_ID]
            and not _safe_list(output_invariant.get("missing_outputs"))
            and not _safe_list(output_invariant.get("unsafe_outputs"))
            and not _safe_list(output_invariant.get("untracked_outputs"))
            and invariant_checks
            and all(
                check.get("task_id") == ROLE_AWARE_TASK_ID
                and check.get("repository_ref") == merge_commit
                and check.get("exists") is True
                and check.get("tracked") is True
                for check in invariant_checks
            )
        )
        receipt_terminal_compatible = bool(
            successful_receipts
            and all(
                (
                    not receipt.get("implementation_commit")
                    or receipt.get("implementation_commit")
                    == implementation_commit
                )
                and (
                    not receipt.get("merge_commit")
                    or receipt.get("merge_commit") == merge_commit
                )
                for receipt in successful_receipts
            )
        )
        coherent = bool(
            receipt_terminal_compatible
            and validation_binding["valid"] is True
            and merge.get("attempted") is True
            and merge.get("merged") is True
            and merge.get("returncode") == 0
            and COMMIT_RE.fullmatch(implementation_commit)
            and COMMIT_RE.fullmatch(merge_commit)
            and str(merge.get("implementation_commit") or "")
            == implementation_commit
            and output_invariant_bound
            and commit_binding["provisional_valid"] is True
        )
        if coherent:
            terminal_events.append(event)
            terminal_commit_bindings.append(commit_binding)
            terminal_validation_bindings.append(validation_binding)

    terminal_event = terminal_events[-1] if terminal_events else {}
    terminal_merge = _safe_dict(terminal_event.get("merge"))
    terminal_implementation_commit = str(
        terminal_event.get("implementation_commit")
        or terminal_merge.get("implementation_commit")
        or ""
    )
    terminal_merge_commit = str(terminal_merge.get("merge_commit") or "")
    terminal_sequence = int(terminal_event.get("sequence") or 0)
    completion_events = [
        event
        for event in events
        if event.get("type") == "task_completed"
        and event.get("task_id") == ROLE_AWARE_TASK_ID
        and event.get("canonical_task_cid")
        == ROLE_AWARE_CANONICAL_TASK_CID
        and event.get("canonical_task_key")
        == ROLE_AWARE_CANONICAL_TASK_KEY
        and isinstance(event.get("sequence"), int)
        and event.get("sequence") > terminal_sequence
    ]
    state_terminal_bound = bool(
        task_state.get("implementation_in_progress") is False
        and str(task_state.get("last_implementation_task_id") or "")
        == ROLE_AWARE_TASK_ID
        and str(task_state.get("last_implementation_task_cid") or "")
        == expected_cid
        and str(task_state.get("last_implementation_commit") or "")
        == terminal_implementation_commit
        and str(task_state.get("last_merge_commit") or "")
        == terminal_merge_commit
        and str(task_state.get("task_status") or "")
        in {"completed", "succeeded", "merged"}
        and completion_events
    )
    task_cid_bound = bool(
        expected_identity_bound
        and events_have_canonical_identity
    )
    provisional_bound = bool(
        task_cid_bound
        and event_chain_bound
        and state_terminal_bound
        and successful_receipts
        and receipt_events
        and terminal_events
    )
    publication_bound = bool(
        provisional_bound
        and terminal_commit_bindings
        and terminal_commit_bindings[-1].get("published_to_origin_main") is True
    )
    bound = bool(provisional_bound and publication_bound)
    publication_phase = (
        "published_final"
        if bound
        else "provisional_merge"
        if provisional_bound
        else "unbound"
    )
    block_reasons = [
        reason
        for reason, condition in (
            ("supervisor_snapshot_missing", bool(snapshot)),
            (
                "trusted_g212_release_evidence_not_bound",
                trusted_release.get("bound") is True,
            ),
            ("durable_supervisor_sources_not_bound", source_files_bound),
            ("supervisor_snapshot_not_durable", durable_snapshot_matches),
            ("canonical_event_chain_not_bound", event_chain_bound),
            ("canonical_successor_identity_not_bound", task_cid_bound),
            ("terminal_task_state_not_bound", state_terminal_bound),
            (
                "member_completion_receipt_missing",
                bool(successful_receipts and receipt_events),
            ),
            ("validation_dag_evidence_missing", bool(terminal_events)),
            (
                "merge_commit_tree_evidence_missing",
                bool(terminal_commit_bindings),
            ),
            ("merge_commit_not_published_to_origin_main", publication_bound),
        )
        if not condition
    ]
    return {
        "present": bool(evidence),
        "bound": bound,
        "provisional_bound": provisional_bound,
        "publication_bound": publication_bound,
        "publication_phase": publication_phase,
        "post_push_finalization_required": bool(
            provisional_bound and not publication_bound
        ),
        "configured_integration_branch": integration_branch,
        "publication_ref": "refs/remotes/origin/main",
        "legacy_display_task_id": ROLE_AWARE_LEGACY_DISPLAY_TASK_ID,
        "trusted_successor_task_id": ROLE_AWARE_TASK_ID,
        "trusted_release_evidence_bound": trusted_release.get("bound") is True,
        "trusted_release_evidence": {
            key: value
            for key, value in trusted_release.items()
            if key != "snapshot"
        },
        "source_files_bound": source_files_bound,
        "durable_snapshot_matches": durable_snapshot_matches,
        "event_chain_bound": event_chain_bound,
        "state_terminal_bound": state_terminal_bound,
        "task_cid_bound": task_cid_bound,
        "member_completion_receipt_bound": bool(
            successful_receipts and receipt_events
        ),
        "validation_bound": bool(terminal_events),
        "merge_commit_tree_bound": bool(terminal_events),
        "canonical_task_cid": expected_cid or None,
        "canonical_task_key": expected_key or None,
        "successful_completion_receipts": successful_receipts,
        "validation_events": terminal_events,
        "merge_events": terminal_events,
        "task_completed_events": completion_events,
        "validation_dag_bindings": terminal_validation_bindings,
        "commit_bindings": terminal_commit_bindings,
        "snapshot_digest_sha256": (
            content_digest(snapshot) if snapshot else None
        ),
        "snapshot": snapshot,
        "block_reasons": block_reasons,
    }


def build_role_aware_deployment_receipt(
    *,
    repo_root: Path,
    observed_at: str | None = None,
    completion_receipt: Mapping[str, Any] | None = None,
    role_aware_certificate: Mapping[str, Any] | None = None,
    supervisor_evidence: Mapping[str, Any] | None = None,
    supervisor_integration_branch: str = ROLE_AWARE_INTEGRATION_BRANCH,
) -> dict[str, Any]:
    """Build a fail-closed, two-phase role-aware deployment attestation."""

    repo_root = repo_root.resolve()
    timestamp = observed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    certifier = _load_certifier_module(repo_root)
    certificate = (
        dict(role_aware_certificate)
        if role_aware_certificate is not None
        else certifier.build_certificate(repo_root=repo_root, role_aware=True)
    )

    completion = (
        dict(completion_receipt)
        if completion_receipt is not None
        else build_receipt(repo_root=repo_root, observed_at=timestamp)
    )

    promotion = _safe_dict(certificate.get("promotion"))
    role_aware = _safe_dict(certificate.get("role_aware"))
    authority_roles = _safe_dict(certificate.get("authority_roles"))
    managed = _safe_dict(certificate.get("managed_deployment_readiness"))
    tools = {
        str(tool.get("tool_id") or ""): tool
        for tool in _safe_list(certificate.get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    }

    elevated = sorted(set(role_aware.get("elevated_tool_ids") or []))
    missing_required = [
        tid
        for tid in REQUIRED_SEMANTIC_ELEVATIONS
        if not tools.get(tid, {}).get("production_certified")
    ]

    certificate_digest_valid = bool(
        certificate.get("certificate_digest_sha256")
        and certificate.get("certificate_digest_sha256")
        == certifier.content_digest(
            {
                key: value
                for key, value in certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    completion_identity_valid = _identity_matches(
        completion,
        "receipt_identity",
        prefixed=True,
    )
    completion_implementation = _safe_dict(completion.get("implementation"))
    completion_acceptance = _safe_dict(completion.get("acceptance"))
    completion_children = [
        child
        for child in _safe_list(completion.get("child_goals"))
        if isinstance(child, Mapping)
    ]
    try:
        objective_child_ids = [
            str(goal.get("goal_id") or "")
            for goal in parse_objective_goals(
                (repo_root / DEFAULT_OBJECTIVES_RELATIVE).read_text(
                    encoding="utf-8"
                )
            )
            if str(goal.get("goal_id") or "") != PROGRAM_GOAL_ID
        ]
    except (OSError, ValueError):
        objective_child_ids = []
    objective_child_count = len(objective_child_ids)
    completion_child_ids = [
        str(child.get("goal_id") or "")
        for child in completion_children
    ]
    exact_objective_children_bound = bool(
        objective_child_ids
        and len(completion_child_ids) == len(set(completion_child_ids))
        and set(completion_child_ids) == set(objective_child_ids)
    )
    declared_child_count = _nonnegative_int(
        completion_implementation.get("child_goal_count")
    )
    declared_bound_count = _nonnegative_int(
        completion_implementation.get("child_goals_bound")
    )
    implementation_complete_and_all_child_goals_bound = bool(
        completion_acceptance.get("implementation_complete") is True
        and completion_implementation.get("status") == "complete"
        and objective_child_count > 0
        and declared_child_count == objective_child_count
        and declared_bound_count == objective_child_count
        and len(completion_children) == objective_child_count
        and exact_objective_children_bound
        and all(child.get("bound") is True for child in completion_children)
        and not _safe_list(completion_implementation.get("child_goals_unbound"))
    )

    semantic_results = [
        result
        for result in _safe_list(certificate.get("semantic_lane_results"))
        if isinstance(result, Mapping)
    ]
    semantic_receipts_full_and_bound = bool(semantic_results)
    semantic_binding_failures: list[str] = []
    semantic_audit = _audit_semantic_lane_results(
        certifier=certifier,
        repo_root=repo_root,
        semantic_results=semantic_results,
    )
    if semantic_audit.get("valid") is not True:
        semantic_receipts_full_and_bound = False
        semantic_binding_failures.extend(
            _safe_list(semantic_audit.get("failures"))
        )
    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "unknown")
        lane_status = str(result.get("status") or "")
        # A supported semantic lane that did not run has no canonical receipt
        # to bind. It therefore blocks this acceptance gate even when it
        # correctly discloses why it did not run. Platform exceptions are
        # reported separately and likewise never count as completed evidence.
        if lane_status != "ran":
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(
                f"{lane_id}:semantic_lane_not_run"
            )
            if not _safe_list(result.get("block_reasons")):
                semantic_binding_failures.append(
                    f"{lane_id}:block_reasons_missing_for_non_ran_lane"
                )
            if _safe_list(result.get("elevated_tool_ids")) or _safe_list(
                result.get("semantically_usable_tool_ids")
            ):
                semantic_binding_failures.append(
                    f"{lane_id}:non_ran_lane_cannot_claim_elevation"
                )
            continue
        raw_receipt = result.get("receipt")
        if not isinstance(raw_receipt, Mapping):
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(f"{lane_id}:raw_receipt_missing")
            continue
        if str(result.get("digest_sha256") or "") != certifier.content_digest(
            raw_receipt
        ):
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(f"{lane_id}:receipt_digest_mismatch")
        receipt_integrity = _safe_dict(result.get("receipt_integrity"))
        if receipt_integrity.get("valid") is not True:
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(
                f"{lane_id}:declared_receipt_integrity_invalid"
            )
        digest_fields = [
            field_name
            for field_name in (
                "receipt_digest_sha256",
                "certificate_digest_sha256",
                "digest_sha256",
            )
            if field_name in raw_receipt
        ]
        if not digest_fields:
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(
                f"{lane_id}:declared_receipt_digest_missing"
            )
        for field_name in digest_fields:
            body = {
                key: value
                for key, value in raw_receipt.items()
                if key != field_name
            }
            computed = certifier.content_digest(body)
            if str(raw_receipt.get(field_name) or "") not in {
                computed,
                f"sha256:{computed}",
            }:
                semantic_receipts_full_and_bound = False
                semantic_binding_failures.append(
                    f"{lane_id}:{field_name}_mismatch"
                )
        semantic_spec = next(
            (
                _safe_dict(spec)
                for spec in certifier.SEMANTIC_CERTIFIER_SPECS
                if str(_safe_dict(spec).get("lane_id") or "") == lane_id
            ),
            {},
        )
        expected_tool_ids = {
            str(value)
            for value in _safe_list(semantic_spec.get("tool_ids"))
        }
        per_tool_rows = _safe_dict(result.get("per_tool"))
        if set(str(key) for key in per_tool_rows) != expected_tool_ids:
            semantic_receipts_full_and_bound = False
            semantic_binding_failures.append(
                f"{lane_id}:per_tool_population_mismatch"
            )
        for tool_id, per_tool in per_tool_rows.items():
            per_tool = _safe_dict(per_tool)
            projected_checks = per_tool.get("checks")
            recorded_digest = str(
                per_tool.get("check_set_digest_sha256") or ""
            )
            recomputed = certifier.recompute_semantic_tool_check_binding(
                result,
                str(tool_id),
            )
            check_binding_valid = bool(
                recomputed.get("valid") is True
                and recorded_digest
                == recomputed.get("check_set_digest_sha256")
                and int(per_tool.get("checks_total") or 0)
                == int(recomputed.get("checks_total") or 0)
                and int(per_tool.get("checks_passed") or 0)
                == int(recomputed.get("checks_passed") or 0)
                and sorted(_safe_list(per_tool.get("check_kinds_present")))
                == sorted(
                    _safe_list(recomputed.get("check_kinds_present"))
                )
                and _safe_dict(per_tool.get("check_status_counts"))
                == _safe_dict(recomputed.get("check_status_counts"))
                and (
                    isinstance(projected_checks, list)
                    and certifier.content_digest(projected_checks)
                    == recorded_digest
                    or (
                        not isinstance(projected_checks, list)
                        and _safe_dict(result.get("projection_policy")).get(
                            "per_tool_checks_bound_by_digest"
                        )
                        is True
                    )
                )
            )
            if not check_binding_valid:
                semantic_receipts_full_and_bound = False
                semantic_binding_failures.append(
                    f"{lane_id}:{tool_id}:check_set_digest_mismatch"
                )
            if (
                _safe_dict(_safe_dict(per_tool).get("artifact_validation")).get(
                    "valid"
                )
                is not True
            ):
                semantic_receipts_full_and_bound = False
                semantic_binding_failures.append(
                    f"{lane_id}:{tool_id}:artifact_identity_invalid"
                )

    platform_audit = _audit_platform_support(
        certifier=certifier,
        repo_root=repo_root,
        managed=managed,
        certificate_lock=_safe_dict(certificate.get("lock")),
        tools=tools,
        authority_roles=authority_roles,
        semantic_results=semantic_results,
    )
    platform_exceptions = [
        dict(item)
        for item in _safe_list(managed.get("platform_exceptions"))
        if isinstance(item, Mapping)
    ]
    elevation_audit = _audit_required_elevations(
        certifier=certifier,
        repo_root=repo_root,
        certificate=certificate,
        semantic_audit=semantic_audit,
    )
    missing_required = list(
        _safe_list(elevation_audit.get("missing"))
    )
    authority_roles_valid = bool(
        authority_roles.get("present") is True
        and authority_roles
        == certifier.load_authority_roles(repo_root)
    )

    quarantines = _safe_list(certificate.get("disagreement_quarantines"))
    hard_zero = _safe_dict(completion.get("hard_zero_gates"))
    hard_zero_derivation = _safe_dict(hard_zero.get("derivation"))
    hard_zero_clear = all(
        int(hard_zero.get(key) or 0) == 0 for key in HARD_ZERO_GATE_KEYS
    )
    hard_zero_derived = bool(
        hard_zero_derivation.get("source")
        and hard_zero_derivation.get("hardcoded_success_counters_forbidden")
        is True
        and hard_zero_derivation.get("complete") is True
        and not _safe_list(hard_zero_derivation.get("missing_measurements"))
    )

    source_attestation = build_source_attestation(repo_root)
    supervisor = derive_supervisor_binding(
        supervisor_evidence,
        repo_root=repo_root,
        integration_branch=supervisor_integration_branch,
    )

    checked_certificate = load_json(repo_root / TOOLCHAIN_CERT_RELATIVE)
    checked_completion = load_json(repo_root / DEFAULT_RECEIPT_RELATIVE)
    checked_certificate_valid = bool(
        checked_certificate
        and checked_certificate.get("certificate_digest_sha256")
        == certifier.content_digest(
            {
                key: value
                for key, value in checked_certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    checked_completion_valid = bool(
        checked_completion
        and _identity_matches(
            checked_completion,
            "receipt_identity",
            prefixed=True,
        )
    )
    checked_certificate_matches = bool(
        checked_certificate_valid
        and checked_certificate.get("certificate_digest_sha256")
        == certificate.get("certificate_digest_sha256")
    )
    checked_completion_matches = bool(
        checked_completion_valid
        and checked_completion.get("receipt_identity")
        == completion.get("receipt_identity")
    )

    artifacts = {
        "toolchain_certificate": {
            "path": TOOLCHAIN_CERT_RELATIVE.as_posix(),
            "present": (repo_root / TOOLCHAIN_CERT_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / TOOLCHAIN_CERT_RELATIVE),
            "role_aware_digest": certificate.get("certificate_digest_sha256"),
            "identity_valid": checked_certificate_valid,
            "matches_dynamic_certificate": checked_certificate_matches,
        },
        "completion_receipt": {
            "path": DEFAULT_RECEIPT_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_RECEIPT_RELATIVE).is_file(),
            "content_identity": completion.get("receipt_identity"),
            "file_sha256": sha256_file(repo_root / DEFAULT_RECEIPT_RELATIVE),
            "identity_valid": checked_completion_valid,
            "matches_dynamic_completion": checked_completion_matches,
        },
        "role_aware_deployment_receipt": {
            "path": DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE.as_posix(),
            "present_before_generation": (
                repo_root / DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE
            ).is_file(),
            "content_identity_before_generation": sha256_file(
                repo_root / DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE
            ),
            "publication_identity": "self:receipt_identity",
        },
        "role_aware_completion_test": {
            "path": DEFAULT_ROLE_AWARE_TEST_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_ROLE_AWARE_TEST_RELATIVE).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_ROLE_AWARE_TEST_RELATIVE
            ),
        },
        "certifier": {
            "path": DEFAULT_CERTIFIER_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_CERTIFIER_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / DEFAULT_CERTIFIER_RELATIVE),
        },
        "receipt_builder": {
            "path": DEFAULT_BUILDER_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_BUILDER_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / DEFAULT_BUILDER_RELATIVE),
        },
    }

    artifacts_present = all(
        bool(item.get("present"))
        for key, item in artifacts.items()
        if key != "role_aware_deployment_receipt"
    )
    platform_exceptions_valid = bool(platform_audit.get("valid"))

    non_authoritative_classes = {
        "identity_plus_fixture_parser",
        "hermetic_adapter_shim",
        "hermetic_shadow_shim",
        "proposal_only_semantics",
    }
    synthetic_evidence_cannot_certify = all(
        not (
            tool.get("production_certified")
            and (
                tool.get("evidence_class") in non_authoritative_classes
                or tool.get("executable_artifact_class")
                == "generated_hermetic_shim"
            )
        )
        for tool in tools.values()
    )
    role_tools = _safe_dict(authority_roles.get("tools"))
    authority_ceiling_respected = all(
        not (
            tools.get(tool_id, {}).get("production_certified")
            and not _safe_dict(meta).get("can_satisfy_certified_authority")
        )
        for tool_id, meta in role_tools.items()
    )
    offline_policy_satisfied = bool(
        _safe_dict(certificate.get("certification_policy")).get(
            "offline_policy_satisfied"
        )
    )
    public_evidence_safe = bool(
        _safe_dict(certificate.get("public_evidence_policy")).get("satisfied")
    )

    acceptance = {
        "role_aware_certificate_bound": certificate_digest_valid,
        "completion_receipt_bound": completion_identity_valid,
        "implementation_complete_and_all_child_goals_bound": (
            implementation_complete_and_all_child_goals_bound
        ),
        "checked_in_certificate_matches": checked_certificate_matches,
        "checked_in_completion_matches": checked_completion_matches,
        "certified_source_bound": bool(source_attestation["source_commit_bound"]),
        "source_candidate_valid_for_attestation": bool(
            source_attestation["valid_for_attestation"]
        ),
        "datasets_gitlink_bound": bool(source_attestation["datasets_gitlink"]),
        "authority_roles_bound": authority_roles_valid,
        "authority_ceiling_respected": authority_ceiling_respected,
        "disagreement_quarantines_bound": certificate_digest_valid
        and isinstance(certificate.get("disagreement_quarantines"), list),
        "public_surfaces_bound": bool(
            (completion.get("implementation") or {}).get("public_operations_bound")
        ),
        "supervisor_evidence_bound": supervisor["bound"],
        "semantic_receipts_full_and_bound": semantic_receipts_full_and_bound,
        "lean_runtime_mtl_authorization_elevated": bool(
            elevation_audit.get("valid") and not missing_required
        ),
        "required_semantic_elevations": list(REQUIRED_SEMANTIC_ELEVATIONS),
        "required_elevations_present": elevated,
        "required_elevations_missing": missing_required,
        "supported_managed_capabilities_ready": bool(managed.get("ready")),
        "supported_managed_capability_blockers": _safe_list(
            managed.get("capability_blockers")
        ),
        "supported_managed_dependency_blockers": _safe_list(
            managed.get("dependency_blockers")
        ),
        "platform_exceptions_derived_and_narrow": platform_exceptions_valid,
        "platform_exceptions_cannot_count_as_complete": platform_exceptions_valid,
        "hard_zero_gates_clear": hard_zero_clear,
        "hard_zero_gates_derived": hard_zero_derived,
        "hardcoded_success_counters_detected": not hard_zero_derived,
        "synthetic_evidence_cannot_certify_production": (
            synthetic_evidence_cannot_certify
        ),
        "no_install_during_offline_certification": offline_policy_satisfied,
        "public_evidence_safe": public_evidence_safe,
        "artifacts_present": artifacts_present,
    }

    # FVT-083 objective validation repair: re-prove FVT-G200 acceptance and
    # bind the synthetic discovery term. True only when the role-aware matrix
    # ran with a digest-bound certificate for this goal/interface.
    role_aware_matrix_bound = bool(
        role_aware.get("enabled")
        and role_aware.get("interface") == ROLE_AWARE_INTERFACE
        and role_aware.get("goal_id") == ROLE_AWARE_GOAL_ID
        and certificate.get("certificate_digest_sha256")
        and semantic_results
    )
    objective_validation_repair = bool(role_aware_matrix_bound)
    acceptance["objective_validation_repair"] = objective_validation_repair
    acceptance["objective_validation_evidence"] = (
        ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE
    )
    acceptance["repair_task_id"] = ROLE_AWARE_REPAIR_TASK_ID
    acceptance["role_aware_matrix_executed"] = role_aware_matrix_bound

    readiness_requirements = {
        key: bool(acceptance[key])
        for key in (
            "role_aware_certificate_bound",
            "completion_receipt_bound",
            "implementation_complete_and_all_child_goals_bound",
            "checked_in_certificate_matches",
            "checked_in_completion_matches",
            "source_candidate_valid_for_attestation",
            "datasets_gitlink_bound",
            "authority_roles_bound",
            "authority_ceiling_respected",
            "disagreement_quarantines_bound",
            "public_surfaces_bound",
            "supervisor_evidence_bound",
            "semantic_receipts_full_and_bound",
            "lean_runtime_mtl_authorization_elevated",
            "supported_managed_capabilities_ready",
            "platform_exceptions_derived_and_narrow",
            "hard_zero_gates_clear",
            "hard_zero_gates_derived",
            "synthetic_evidence_cannot_certify_production",
            "no_install_during_offline_certification",
            "public_evidence_safe",
            "artifacts_present",
        )
    }
    status = (
        "role_aware_deployment_ready_for_attestation_publication"
        if all(readiness_requirements.values())
        else "role_aware_deployment_blocked"
    )
    deployment_blockers = sorted(
        [
            key
            for key, satisfied in readiness_requirements.items()
            if not satisfied
        ]
        + semantic_binding_failures
        + _safe_list(platform_audit.get("failures"))
        + _safe_list(elevation_audit.get("failures"))
        + [
            f"managed:{item.get('tool_id')}:{reason}"
            for item in _safe_list(managed.get("all_blockers"))
            if isinstance(item, Mapping)
            for reason in _safe_list(item.get("reasons"))
        ]
        + [f"supervisor:{reason}" for reason in supervisor["block_reasons"]]
    )

    receipt: dict[str, Any] = {
        "schema_version": ROLE_AWARE_SCHEMA_VERSION,
        "interface": ROLE_AWARE_INTERFACE,
        "program_interface": PROGRAM_INTERFACE,
        "program_goal_id": PROGRAM_GOAL_ID,
        "goal_id": ROLE_AWARE_GOAL_ID,
        "task_id": ROLE_AWARE_TASK_ID,
        "repair_task_id": ROLE_AWARE_REPAIR_TASK_ID,
        "legacy_display_task_id": ROLE_AWARE_LEGACY_DISPLAY_TASK_ID,
        "program": "formal-verification-tactician/toolchain-release",
        "observed_at": timestamp,
        "binding_mode": "two_phase_source_then_attestation_publication",
        "status": status,
        "description": (
            "Fail-closed role-aware deployment attestation. Full semantic "
            "receipts and check sets live in the bound toolchain certificate; "
            "this receipt digest-binds that matrix, distinguishes unsupported "
            "platforms from missing supported capabilities, and cannot claim "
            "readiness without authoritative supervisor validation/merge evidence."
        ),
        # FVT-083 objective validation repair discovery binding.
        "objective_validation_evidence": ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": objective_validation_repair,
        "objective_validation_command": ROLE_AWARE_OBJECTIVE_VALIDATION_COMMAND,
        "source": source_attestation,
        "acceptance": acceptance,
        "semantic_audit": semantic_audit,
        "platform_support_audit": platform_audit,
        "required_elevation_audit": elevation_audit,
        "readiness_requirements": readiness_requirements,
        "deployment_blockers": deployment_blockers,
        "hard_zero_gates": {
            key: hard_zero.get(key, 0) for key in HARD_ZERO_GATE_KEYS
        }
        | {"derivation": hard_zero.get("derivation")},
        "role_aware_certificate": {
            "interface": certificate.get("interface"),
            "schema_version": certificate.get("schema_version"),
            "goal_id": certificate.get("goal_id"),
            "task_id": certificate.get("task_id"),
            "binding_mode": certificate.get("binding_mode"),
            "certificate_digest_sha256": certificate.get("certificate_digest_sha256"),
            "role_aware": {
                "enabled": bool(role_aware.get("enabled")),
                "goal_id": role_aware.get("goal_id"),
                "task_id": role_aware.get("task_id"),
                "repair_task_id": role_aware.get("repair_task_id")
                or ROLE_AWARE_REPAIR_TASK_ID,
                "interface": role_aware.get("interface"),
                "objective_validation_evidence": role_aware.get(
                    "objective_validation_evidence"
                )
                or ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE,
                "objective_validation_repair": bool(
                    role_aware.get("objective_validation_repair")
                    if "objective_validation_repair" in role_aware
                    else objective_validation_repair
                ),
                "objective_validation_command": role_aware.get(
                    "objective_validation_command"
                )
                or ROLE_AWARE_OBJECTIVE_VALIDATION_COMMAND,
                "elevated_tool_ids": elevated,
                "required_baseline_elevations": list(
                    role_aware.get("required_baseline_elevations")
                    or list(REQUIRED_SEMANTIC_ELEVATIONS)
                ),
                "elevation_count": len(_safe_list(role_aware.get("elevations"))),
                "demotion_count": len(_safe_list(role_aware.get("demotions"))),
            },
            "promotion": {
                "production_certified_tool_ids": list(
                    promotion.get("production_certified_tool_ids") or []
                ),
                "merely_usable_tool_ids": list(
                    promotion.get("merely_usable_tool_ids") or []
                ),
                "unavailable_tool_ids": list(
                    promotion.get("unavailable_tool_ids") or []
                ),
            },
            "property_lanes": certificate.get("property_lanes"),
            "disagreement_quarantines": quarantines,
            "authority_roles": {
                key: authority_roles.get(key)
                for key in (
                    "present",
                    "interface",
                    "role_interface",
                    "boundary",
                    "policy_digest_sha256",
                )
            },
            # Bulk lane receipts and tool check dumps stay in the certificate;
            # this surface digest-binds them for the deployment attestation.
            "semantic_lane_results": [
                _compact_semantic_lane(result) for result in semantic_results
            ],
            "specialized_receipt_aggregation": _compact_specialized_deployment_binding(
                _safe_dict(certificate.get("specialized_receipt_aggregation"))
            ),
            "managed_deployment_readiness": _compact_managed_readiness(managed),
            "tools": [
                _compact_tool_binding(
                    tools[tool_id],
                    checks_digest=content_digest(
                        _safe_list(tools[tool_id].get("checks"))
                    ),
                    artifact_digests=[
                        str(item.get("sha256") or "")
                        for item in _safe_list(
                            tools[tool_id].get("artifact_identities")
                        )
                        if isinstance(item, Mapping) and item.get("sha256")
                    ],
                )
                for tool_id in sorted(tools)
            ],
        },
        "supervisor_evidence": supervisor,
        "completion": {
            "interface": completion.get("interface"),
            "completion_goal_id": completion.get("completion_goal_id"),
            "task_id": completion.get("task_id"),
            "receipt_identity": completion.get("receipt_identity"),
            "implementation_status": (completion.get("implementation") or {}).get(
                "status"
            ),
            "deployment_status": (completion.get("deployment") or {}).get("status"),
            "child_goals_bound": (completion.get("implementation") or {}).get(
                "child_goals_bound"
            ),
            "child_goal_count": (completion.get("implementation") or {}).get(
                "child_goal_count"
            ),
            "child_goals_unbound": (
                completion.get("implementation") or {}
            ).get("child_goals_unbound"),
            "objective_child_count": objective_child_count,
            "exact_objective_child_population_bound": (
                exact_objective_children_bound
            ),
        },
        "elevations": {
            "required": list(REQUIRED_SEMANTIC_ELEVATIONS),
            "elevated_tool_ids": elevated,
            "missing_required": missing_required,
            "merely_usable_tool_ids": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
            "production_certified_tool_ids": list(
                promotion.get("production_certified_tool_ids") or []
            ),
            # Compact elevation details: identity + outcome + check digest only.
            "details": [
                {
                    "tool_id": item.get("tool_id"),
                    "lane_id": item.get("lane_id"),
                    "elevated": bool(item.get("elevated")),
                    "reason": item.get("reason"),
                    "evidence_class": item.get("evidence_class"),
                    "semantic_receipt_digest_sha256": item.get(
                        "semantic_receipt_digest_sha256"
                    ),
                    "checks_digest_sha256": content_digest(
                        _safe_list(item.get("checks"))
                    )
                    if item.get("checks") is not None
                    else None,
                }
                for item in _safe_list(role_aware.get("elevations"))
                if isinstance(item, Mapping)
            ],
        },
        "platform_exceptions": platform_exceptions,
        "artifacts": artifacts,
        "disclosures": {
            "unavailable_tools": list(
                promotion.get("unavailable_tool_ids") or []
            ),
            "merely_usable_tools": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
            "missing_required_elevations": missing_required,
            "supported_managed_capability_blockers": _safe_list(
                managed.get("capability_blockers")
            ),
            "supported_managed_dependency_blockers": _safe_list(
                managed.get("dependency_blockers")
            ),
            "publication_gates": _safe_dict(
                _safe_dict(source_attestation.get("tree_alignment")).get(
                    "publication_gates"
                )
            ),
            "assurance_ceilings": {
                "path_presence_is_not_usability": True,
                "source_presence_is_not_usability": True,
                "fixture_is_not_production_certified": True,
                "synthetic_evidence_cannot_certify_production": True,
                "advisor_support_shadow_cannot_certify": True,
                "unavailable_cannot_count_as_complete": True,
                "lfv_completion_is_not_deployment_certificate": True,
            },
            "remaining_bounds": [
                "Machine-specific production certification is limited to tools "
                "with hermetic offline or semantic certificates.",
                "Unavailable managed external capabilities remain open "
                "capability-expansion work and never invent success.",
                "Generated parser/hermetic/shadow evidence remains below its "
                "declared authority ceiling.",
                "Attestation publication is a second phase; this receipt never "
                "claims to hash the tree containing itself.",
            ],
        },
        "notes": [
            "RoleAwareFormalVerificationRelease@1 reissues deployment certification "
            "after FVT-G101–G190 installation and semantic certification.",
            "All semantic check records are retained; no first-only projection "
            "can hide an omitted or failing case.",
            "Platform exceptions come only from lock-declared host support. "
            "Missing supported installations remain blockers.",
            "Supervisor binding requires durable member_completion_receipt@1, "
            "canonical task CID/key, passing validation, and merge commit evidence.",
        ],
    }
    receipt = certifier.public_evidence_projection(receipt, repo_root=repo_root)
    public_evidence_policy = certifier.public_evidence_audit(receipt)
    receipt["public_evidence_policy"] = public_evidence_policy
    if not public_evidence_policy["satisfied"]:
        receipt["acceptance"]["public_evidence_safe"] = False
        receipt["acceptance"]["hard_zero_gates_clear"] = False
        receipt["readiness_requirements"]["public_evidence_safe"] = False
        receipt["readiness_requirements"]["hard_zero_gates_clear"] = False
        receipt["status"] = "role_aware_deployment_blocked"
        blockers = receipt["deployment_blockers"]
        if "public_evidence_safe" not in blockers:
            blockers.append("public_evidence_safe")
        leakage_count = len(public_evidence_policy["failures"])
        receipt["hard_zero_gates"]["secret_or_witness_leakage_count"] = max(
            int(
                receipt["hard_zero_gates"].get(
                    "secret_or_witness_leakage_count"
                )
                or 0
            ),
            leakage_count,
        )
    receipt["receipt_identity"] = content_digest(receipt)
    return receipt


def build_receipt(
    *,
    repo_root: Path,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the completion receipt from the current tree."""

    repo_root = repo_root.resolve()
    objectives_path = repo_root / DEFAULT_OBJECTIVES_RELATIVE
    if not objectives_path.is_file():
        raise FileNotFoundError(f"missing objective heap: {objectives_path}")

    objectives_text = objectives_path.read_text(encoding="utf-8")
    goals = parse_objective_goals(objectives_text)
    if not goals or goals[0]["goal_id"] != PROGRAM_GOAL_ID:
        raise ValueError("objective heap must begin with FVT-G000")

    self_generated = frozenset(
        {
            DEFAULT_RECEIPT_RELATIVE.as_posix(),
            DEFAULT_BUILDER_RELATIVE.as_posix(),
            DEFAULT_COMPLETION_TEST_RELATIVE.as_posix(),
            DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE.as_posix(),
            DEFAULT_ROLE_AWARE_TEST_RELATIVE.as_posix(),
        }
    )

    child_goals = [
        bind_child_goal(goal, repo_root=repo_root, self_generated=self_generated)
        for goal in goals
        if goal["goal_id"] != PROGRAM_GOAL_ID
    ]

    artifacts = bind_artifacts(repo_root)
    # The completion receipt is this document. Its content address is
    # ``receipt_identity`` (computed after assembly), not a nested self-hash.
    artifacts["completion_receipt"] = {
        "path": DEFAULT_RECEIPT_RELATIVE.as_posix(),
        "present": True,
        "content_identity": "self:receipt_identity",
    }

    baseline = load_json(repo_root / BASELINE_RELATIVE)
    certificate = load_json(repo_root / TOOLCHAIN_CERT_RELATIVE)
    benchmark = load_json(repo_root / BENCHMARK_RELATIVE)
    live_report = load_json(repo_root / LIVE_REPORT_RELATIVE)
    corpus = load_json(repo_root / CORPUS_MANIFEST_RELATIVE)

    alignment = observe_tree_alignment(repo_root)
    hard_zero = derive_hard_zero_gates(
        certificate=certificate,
        benchmark=benchmark,
        baseline=baseline,
        repo_root=repo_root,
    )
    implementation = build_implementation_section(
        child_goals=child_goals,
        artifacts=artifacts,
        corpus=corpus,
    )
    deployment = build_deployment_section(
        alignment=alignment,
        baseline=baseline,
        certificate=certificate,
        live_report=live_report,
        hard_zero=hard_zero,
    )
    acceptance = build_acceptance(
        implementation=implementation,
        deployment=deployment,
        hard_zero=hard_zero,
        child_goals=child_goals,
    )

    timestamp = observed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parent = alignment.get("parent") or {}
    gitlink = alignment.get("gitlink") or {}
    embedded = alignment.get("embedded_checkout") or {}

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "program_interface": PROGRAM_INTERFACE,
        "program_goal_id": PROGRAM_GOAL_ID,
        "completion_goal_id": COMPLETION_GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "observed_at": timestamp,
        "binding_mode": "current_tree_content_identity",
        "source": {
            "parent_commit": parent.get("commit"),
            "parent_tree": parent.get("tree"),
            "datasets_gitlink": gitlink.get("commit"),
            "datasets_embedded_head": embedded.get("head"),
            "datasets_origin_main": embedded.get("origin_main"),
            "parent_origin_main": parent.get("origin_main"),
            "binding_mode": "current_tree_content_identity",
        },
        "acceptance": acceptance,
        "hard_zero_gates": {
            key: hard_zero[key] for key in HARD_ZERO_GATE_KEYS
        }
        | {"derivation": hard_zero.get("derivation")},
        "implementation": implementation,
        "deployment": deployment,
        "artifacts": artifacts,
        "child_goals": child_goals,
        "disclosures": {
            "remaining_bounds": [
                "Machine-specific production certification is limited to tools "
                "with hermetic offline certificates.",
                "Publication lag against origin/main is disclosed and never "
                "fetched away.",
                "Unavailable tools and unsupported semantics remain explicit "
                "non-success outcomes.",
            ],
            "unsupported_semantics": deployment.get("unsupported_semantics") or [],
            "unavailable_tools": deployment.get("unavailable_tools") or [],
            "publication_gates": deployment.get("publication_gates") or {},
            "assurance_ceilings": deployment.get("assurance_ceilings") or {},
            "open_baseline_findings": (
                hard_zero.get("derivation") or {}
            ).get("open_baseline_findings")
            or [],
        },
        "notes": [
            "Receipt binds every executable child goal of FVT-G000 under "
            "FormalVerificationTacticianCompletionReceipt@1.",
            "Implementation and deployment sections are separate; code "
            "presence never invents production certification.",
            "Hard-zero gates are derived from child certificates and benchmark "
            "hard gates; hardcoded success counters are forbidden.",
            "Generate only from a clean current tree and immutable evidence.",
        ],
    }

    # The standalone completion receipt is a checked-in public artifact just
    # like the toolchain certificate and role-aware deployment receipt.  Apply
    # the same portable projection before content-addressing it so an exact
    # deployment path, raw process output, or private witness cannot escape
    # through the historical (non-role-aware) surface.  Redactions retain
    # bounded byte-length/content-digest metadata, while repository-relative
    # paths and pre-existing artifact identities remain useful.
    certifier = _load_certifier_module(repo_root)
    projected = certifier.public_evidence_projection(receipt, repo_root=repo_root)
    if not isinstance(projected, dict):
        raise TypeError("completion receipt public projection must be a mapping")
    receipt = projected

    public_evidence_policy = certifier.public_evidence_audit(receipt)
    receipt["public_evidence_policy"] = public_evidence_policy
    receipt["acceptance"]["public_evidence_safe"] = bool(
        public_evidence_policy["satisfied"]
    )
    if not public_evidence_policy["satisfied"]:
        leakage_count = max(
            int(
                receipt["hard_zero_gates"].get(
                    "secret_or_witness_leakage_count"
                )
                or 0
            ),
            len(public_evidence_policy["failures"]),
        )
        receipt["hard_zero_gates"]["secret_or_witness_leakage_count"] = (
            leakage_count
        )
        receipt["acceptance"]["secret_or_witness_leakage_count"] = leakage_count
        receipt["acceptance"]["hard_zero_gates_clear"] = False

    # Content-address the projected receipt excluding the identity field
    # itself.  The identity therefore covers the portable public form, not
    # host-private source values.
    receipt["receipt_identity"] = content_digest(receipt)
    return receipt


def build_release_candidate_source_attestation(repo_root: Path) -> dict[str, Any]:
    """Bind an explicit certified source commit/tree for the release candidate.

    The generated candidate identity is never used as its own source identity.
    A prior checked-in version of the candidate may already be present in the
    certified Git tree; that fact is measured rather than falsely described as
    path exclusion. Merge and deployment remain unclaimed until FVT-G214
    post-merge attestation.
    """

    base = build_source_attestation(repo_root)
    dirty = list(base.get("dirty_paths_at_certification") or [])
    non_candidate_dirty = sorted(
        set(dirty) - set(RELEASE_CANDIDATE_ATTESTATION_PATHS)
    )
    source_commit = base.get("certified_source_commit")
    source_tree = base.get("certified_source_tree")
    source_commit_bound = bool(
        source_commit
        and source_tree
        and COMMIT_RE.fullmatch(str(source_commit))
        and COMMIT_RE.fullmatch(str(source_tree))
        and base.get("source_commit_bound")
    )
    candidate_path_present_in_source_tree = False
    if source_commit and COMMIT_RE.fullmatch(str(source_commit)):
        candidate_entry = _git(
            repo_root,
            "cat-file",
            "-e",
            (
                f"{source_commit}:"
                f"{DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix()}"
            ),
        )
        candidate_path_present_in_source_tree = bool(
            candidate_entry is not None and candidate_entry.returncode == 0
        )
    return {
        "model": "pre_merge_release_candidate_source/v1",
        "certified_source_commit": source_commit,
        "certified_source_tree": source_tree,
        "datasets_gitlink": base.get("datasets_gitlink"),
        "datasets_embedded_head": base.get("datasets_embedded_head"),
        "source_commit_bound": source_commit_bound,
        "source_candidate_clean": not dirty,
        "dirty_paths_at_certification": dirty,
        "non_candidate_dirty_paths": non_candidate_dirty,
        "attestation_paths": sorted(RELEASE_CANDIDATE_ATTESTATION_PATHS),
        "candidate_path_present_in_source_tree": (
            candidate_path_present_in_source_tree
        ),
        "candidate_excluded_from_source_tree": (
            not candidate_path_present_in_source_tree
        ),
        "generated_candidate_identity_excluded_from_source_identity": True,
        "self_referential_current_tree_claim_forbidden": True,
        "source_binding_uses_committed_tree_not_candidate_identity": True,
        "merge_event_required_to_exceed_release_candidate": True,
        "merge_event_present": False,
        "deployment_attestation_present": False,
        "claims_own_future_merge": False,
        "claims_own_future_deployment": False,
        "valid_for_release_candidate": bool(
            source_commit_bound and (not dirty or not non_candidate_dirty)
        ),
        "tree_alignment": base.get("tree_alignment"),
    }


def _compact_tool_binding(
    tool: Mapping[str, Any],
    *,
    checks_digest: str,
    artifact_digests: Sequence[str],
) -> dict[str, Any]:
    """Project a tool row into digest-bound release-candidate form."""

    return {
        "tool_id": tool.get("tool_id"),
        "evidence_class": tool.get("evidence_class"),
        "production_certified": bool(tool.get("production_certified")),
        "executable_artifact_class": tool.get("executable_artifact_class"),
        "executable_sha256": tool.get("executable_sha256"),
        "checks_digest_sha256": checks_digest,
        "artifact_digests": list(artifact_digests),
        "role": tool.get("role"),
        "assurance_ceiling": tool.get("assurance_ceiling"),
    }


HOST_PATH_REDACTION = "<host-path-redacted>"
MANAGED_PROVER_ROOT_ENV = "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"
ELAN_HOME_ENV = "ELAN_HOME"
JAVA_HOME_ENV = "JAVA_HOME"


def _redacted_host_basename(raw_path: str) -> str | None:
    """Return the single safe basename carried by a public host marker."""

    prefix = HOST_PATH_REDACTION + "/"
    if not raw_path.startswith(prefix):
        return None
    suffix = raw_path.removeprefix(prefix)
    if (
        not suffix
        or suffix in {".", ".."}
        or "/" in suffix
        or "\\" in suffix
        or "\0" in suffix
        or Path(suffix).name != suffix
    ):
        return None
    return suffix


def _approved_managed_prover_roots(
    lock_entry: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Return installer-policy roots allowed to resolve public path markers."""

    roots: list[Path] = []
    seen: set[str] = set()

    def add_root(raw_root: object) -> None:
        if raw_root in (None, ""):
            return
        try:
            resolved = Path(
                os.path.expanduser(str(raw_root))
            ).resolve()
        except OSError:
            return
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    # This explicit deployment root is part of the sealed validation
    # environment and is never inferred from a redacted receipt value.
    add_root(os.environ.get(MANAGED_PROVER_ROOT_ENV))
    try:
        installer_registry = importlib.import_module(
            "ipfs_datasets_py.logic.backends.installers.registry"
        )
        add_root(
            getattr(
                installer_registry,
                "DEFAULT_USER_LOCAL_INSTALL_ROOT",
                None,
            )
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        pass

    plugin_name = str(lock_entry.get("installer_plugin") or "").strip()
    if plugin_name == "solver":
        # Python-distributed SMT launchers are installed under the interpreter
        # user base rather than the theorem-prover plugin root.
        try:
            site_module = importlib.import_module("site")
            add_root(site_module.getuserbase())
        except (ImportError, OSError, RuntimeError, ValueError):
            pass
    if str(lock_entry.get("tool_id") or "") == "lean":
        # Lean's reviewed managed launcher is owned by elan rather than an
        # ipfs_datasets_py installer plugin.  Restrict reconstruction to the
        # configured/default elan root; digest and artifact-class equality are
        # still mandatory before this path can satisfy an evidence binding.
        add_root(
            os.environ.get(ELAN_HOME_ENV)
            or (Path.home() / ".elan")
        )
    if str(lock_entry.get("tool_id") or "") == "java":
        # The state-model lane binds the reviewed JVM through the explicitly
        # sealed JAVA_HOME rather than the theorem-prover root's top-level bin.
        # Keep the same exact basename, digest, and artifact-class checks used
        # for every other public host-path reconstruction.
        add_root(os.environ.get(JAVA_HOME_ENV))

    if re.fullmatch(r"[a-z][a-z0-9_]*", plugin_name):
        try:
            installer = importlib.import_module(
                "ipfs_datasets_py.logic.backends.installers."
                + plugin_name
            )
            expand_root = getattr(
                installer,
                "expand_user_local_root",
                None,
            )
            if callable(expand_root):
                add_root(expand_root())
        except (ImportError, OSError, RuntimeError, ValueError):
            # Missing installer policy is a closed resolution failure; callers
            # retain no ambient-PATH fallback.
            pass
    return tuple(roots)


def _approved_redacted_executable_candidates(
    *,
    certifier,
    lock_entry: Mapping[str, Any],
    raw_path: str,
) -> tuple[Path, ...]:
    """Resolve a public host marker only inside approved managed bin roots."""

    marker_basename = _redacted_host_basename(raw_path)
    if raw_path != HOST_PATH_REDACTION and marker_basename is None:
        return ()

    declared_names = [
        str(value).strip()
        for value in _safe_list(lock_entry.get("executable_candidates"))
        if str(value).strip()
    ]
    declared_basenames = {
        Path(name).name
        for name in declared_names
        if Path(name).name not in {"", ".", ".."}
    }
    if marker_basename is not None:
        if marker_basename not in declared_basenames:
            return ()
        search_basenames = [marker_basename]
    else:
        search_basenames = sorted(declared_basenames)

    roots = _approved_managed_prover_roots(lock_entry)
    managed_bins = [
        (root / "bin").resolve()
        for root in roots
    ]
    managed_path = os.pathsep.join(str(path) for path in managed_bins)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(raw_candidate: str | Path) -> None:
        try:
            candidate = Path(raw_candidate).resolve()
        except OSError:
            return
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return
        if not any(
            candidate.parent == managed_bin
            for managed_bin in managed_bins
        ):
            return
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    for managed_bin in managed_bins:
        for basename in search_basenames:
            add_candidate(managed_bin / basename)

    # Resolve the lock's reviewed bare candidates against only the approved
    # managed bins. Never consult the process's ambient PATH here.
    if managed_path:
        for candidate_name in declared_names:
            if marker_basename is not None and (
                Path(candidate_name).name != marker_basename
            ):
                continue
            resolved = certifier.resolve_executable(
                [candidate_name],
                env={"PATH": managed_path},
            )
            if resolved:
                add_candidate(resolved)
    return tuple(candidates)


def _matching_approved_redacted_executables(
    *,
    certifier,
    lock_entry: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Resolve and verify one redacted executable identity fail closed."""

    raw_path = str(artifact.get("path") or "")
    return tuple(
        candidate
        for candidate in _approved_redacted_executable_candidates(
            certifier=certifier,
            lock_entry=lock_entry,
            raw_path=raw_path,
        )
        if (
            certifier.file_digest(candidate) == artifact.get("sha256")
            and certifier.classify_executable_artifact(candidate)
            == artifact.get("artifact_class")
        )
    )


def _matching_approved_redacted_artifacts(
    *,
    certifier,
    lock_entry: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Resolve a redacted managed artifact only inside reviewed root shapes."""

    raw_path = str(artifact.get("path") or "")
    basename = _redacted_host_basename(raw_path)
    if basename is None:
        return ()
    kind = str(artifact.get("kind") or "")
    artifact_class = str(artifact.get("artifact_class") or "")
    candidates: list[Path] = []
    if kind in {"semantic_executable", "executable"}:
        candidates.extend(
            _matching_approved_redacted_executables(
                certifier=certifier,
                lock_entry=lock_entry,
                artifact=artifact,
            )
        )

    allowed_parents = {
        "managed_release_archive": {
            "downloads",
            *(
                ("tlc",)
                if str(lock_entry.get("tool_id") or "") == "tlc"
                else ()
            ),
        },
        "managed_runtime_manifest": {"manifests"},
    }.get(kind, set())
    recursive_kinds = {
        "launcher_target",
        "launcher_runtime",
        "managed_release_archive",
        "managed_runtime_manifest",
    }
    if kind in recursive_kinds:
        for root in _approved_managed_prover_roots(lock_entry):
            try:
                root = root.resolve()
            except OSError:
                continue
            direct_candidates: list[Path] = []
            for allowed_parent in sorted(allowed_parents):
                direct_candidates.append(root / allowed_parent / basename)
            else:
                direct_candidates.append(root / "bin" / basename)
            try:
                direct_candidates.extend(root.rglob(basename))
            except OSError:
                pass
            for candidate in direct_candidates:
                try:
                    resolved = candidate.resolve()
                    relative = resolved.relative_to(root)
                except (OSError, RuntimeError, ValueError):
                    continue
                if allowed_parents and (
                    not relative.parts
                    or relative.parts[0] not in allowed_parents
                ):
                    continue
                if kind == "launcher_runtime" and resolved.name != "java":
                    continue
                if kind == "launcher_target" and relative.parts[:1] in {
                    ("downloads",),
                    ("manifests",),
                }:
                    continue
                if not resolved.is_file():
                    continue
                if certifier.file_digest(resolved) != artifact.get("sha256"):
                    continue
                if artifact_class in {
                    "native_or_managed_binary",
                    "launcher_script",
                    "generated_hermetic_shim",
                } and (
                    certifier.classify_executable_artifact(resolved)
                    != artifact_class
                ):
                    continue
                candidates.append(resolved)

    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique.setdefault(str(candidate), candidate)
    return tuple(unique.values())


def _audited_checked_vendor_fanin_policy(
    *,
    certifier,
    repo_root: Path,
    spec: Mapping[str, Any],
    semantic_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Freshly rerun and independently join a checked vendor differential lane."""

    lane_id = str(spec.get("lane_id") or "")
    vendor_spec = _safe_dict(
        getattr(certifier, "CHECKED_VENDOR_FANIN_SPECS", {}).get(lane_id)
    )
    expected_tool_ids = [
        str(value) for value in _safe_list(spec.get("tool_ids"))
    ]
    static_allowed = bool(spec.get("production_elevation_allowed"))
    default_evidence_class = str(spec.get("evidence_class") or "")
    receipt = _safe_dict(semantic_result.get("receipt"))
    recorded = _safe_dict(semantic_result.get("checked_vendor_fanin"))
    receipt_recorded = _safe_dict(receipt.get("checked_vendor_fanin"))
    if semantic_result.get("status") != "ran":
        failures = []
        if recorded or receipt_recorded:
            failures.append("non_ran_lane_claimed_checked_vendor_fanin")
        return {
            "valid": not failures,
            "failures": failures,
            "live_claimed": False,
            "vendor_claimed": bool(recorded or receipt_recorded),
            "fanin_satisfied": False,
            "eligible_tool_ids": [],
            "production_allowed_tool_ids": (
                expected_tool_ids if static_allowed else []
            ),
            "lane_production_elevation_allowed": static_allowed,
            "evidence_class": default_evidence_class,
        }

    failures: list[str] = []
    configured_targets = {
        str(tool_id)
        for tool_id in _safe_dict(
            vendor_spec.get("expected_reference_checks")
        )
    }
    if not vendor_spec:
        failures.append("checked_vendor_fanin_policy_not_configured")
    if configured_targets != set(expected_tool_ids):
        failures.append("checked_vendor_fanin_target_population_mismatch")

    module = None
    try:
        module_path = repo_root / Path(spec["module_relative"])
        module = certifier._load_module_from_path(
            module_path,
            f"fvt_builder_checked_vendor_reference_{lane_id}",
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(
            f"checked_vendor_reference_module_unavailable:{type(exc).__name__}"
        )

    audit_env = certifier.offline_env(os.environ)
    prebuilt = certifier._runtime_mtl_managed_prebuilt_binding(
        repo_root,
        env=audit_env,
    )
    invocation = _safe_dict(prebuilt.get("invocation"))
    sealed_root = (
        Path(str(invocation["sealed_root"]))
        if invocation.get("sealed_root")
        else None
    )
    fresh: dict[str, Any] = {}
    if module is not None and vendor_spec:
        fresh = certifier._build_checked_vendor_fanin(
            repo_root=repo_root,
            sealed_root=sealed_root,
            semantic_spec=spec,
            semantic_module=module,
            reference_receipt=receipt,
        )

    for label, value in (
        ("recorded", recorded),
        ("receipt", receipt_recorded),
        ("fresh", fresh),
    ):
        declared = str(value.get("digest_sha256") or "")
        computed = certifier.content_digest(
            {
                key: item
                for key, item in value.items()
                if key != "digest_sha256"
            }
        )
        if not declared or declared != computed:
            failures.append(
                f"checked_vendor_fanin_{label}_self_digest_invalid"
            )
    if recorded != receipt_recorded:
        failures.append("checked_vendor_fanin_recording_disagrees_with_receipt")
    if recorded != fresh:
        failures.append("checked_vendor_fanin_fresh_replay_mismatch")

    expected_vendor_checks = int(
        vendor_spec.get("expected_vendor_checks") or 0
    )
    fresh_checked = _safe_dict(fresh.get("checked_install_receipt"))
    fresh_live = _safe_dict(fresh.get("live_certificate"))
    vendor_ready = bool(
        fresh
        and not _safe_list(fresh.get("failures"))
        and fresh_checked.get("exact_live_nested_match") is True
        and fresh_live.get("certified") is True
        and int(fresh_live.get("checks_passed") or 0)
        == expected_vendor_checks
        and int(fresh_live.get("checks_total") or 0)
        == expected_vendor_checks
        and len(_safe_list(fresh_live.get("check_ids")))
        == expected_vendor_checks
        and len(set(_safe_list(fresh_live.get("check_ids"))))
        == expected_vendor_checks
        and all(
            str(check_id)
            for check_id in _safe_list(fresh_live.get("check_ids"))
        )
        and str(
            fresh_live.get("nested_install_receipt_digest_sha256")
            or ""
        )
        == str(fresh_checked.get("self_digest_sha256") or "")
    )

    compact_tools = _safe_dict(semantic_result.get("per_tool"))
    fresh_references = _safe_dict(fresh.get("reference_bindings"))
    eligible: list[str] = []
    reference_audits: dict[str, Any] = {}
    for tool_id in expected_tool_ids:
        binding = _safe_dict(fresh_references.get(tool_id))
        compact_tool = _safe_dict(compact_tools.get(tool_id))
        pnmr = _independent_pnmr_reconstruction(
            certifier=certifier,
            semantic_result=semantic_result,
            tool_id=tool_id,
            compact_tool=compact_tool,
        )
        raw_certified, raw_checks, raw_reasons = (
            certifier._tool_certified_from_semantic_receipt(
                tool_id,
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec.get("selector") or "root"),
            )
        )
        expected_marker = _safe_dict(
            vendor_spec.get("expected_reference_checks")
        ).get(tool_id)
        try:
            expected_count = int(binding.get("expected_checks_total"))
        except (TypeError, ValueError):
            expected_count = -1
        expected_count_valid = expected_count > 0
        if expected_marker != "closed_manifest":
            try:
                expected_count_valid = (
                    expected_count == int(expected_marker)
                )
            except (TypeError, ValueError):
                expected_count_valid = False
        normalized = certifier.recompute_semantic_tool_check_binding(
            semantic_result,
            tool_id,
        )
        reference_ready = bool(
            raw_certified
            and not raw_reasons
            and len(raw_checks) == expected_count
            and expected_count_valid
            and pnmr.get("valid") is True
            and normalized.get("valid") is True
            and normalized.get("checks_passed") == expected_count
            and normalized.get("checks_total") == expected_count
            and str(normalized.get("check_set_digest_sha256") or "")
            == str(binding.get("check_set_digest_sha256") or "")
        )
        reference_audits[tool_id] = {
            "valid": reference_ready,
            "expected_checks_total": expected_count,
            "raw_checks_total": len(raw_checks),
            "pnmr": pnmr,
            "check_set_digest_sha256": normalized.get(
                "check_set_digest_sha256"
            ),
            "block_reasons": list(raw_reasons),
        }
        if vendor_ready and reference_ready:
            eligible.append(tool_id)

    expected_eligible = sorted(eligible)
    if sorted(str(item) for item in _safe_list(fresh.get("eligible_tool_ids"))) != (
        expected_eligible
    ):
        failures.append("checked_vendor_fanin_fresh_eligibility_mismatch")
    if sorted(str(item) for item in _safe_list(recorded.get("eligible_tool_ids"))) != (
        expected_eligible
    ):
        failures.append("checked_vendor_fanin_recorded_eligibility_mismatch")

    production_allowed_ids = sorted(
        set(expected_tool_ids if static_allowed else ()) | set(eligible)
    )
    lane_allowed = bool(production_allowed_ids)
    expected_evidence_class = (
        str(vendor_spec.get("evidence_class") or default_evidence_class)
        if eligible
        else default_evidence_class
    )
    if semantic_result.get("production_elevation_allowed") is not lane_allowed:
        failures.append("checked_vendor_fanin_lane_policy_flag_mismatch")
    if semantic_result.get("evidence_class") != expected_evidence_class:
        failures.append("checked_vendor_fanin_lane_evidence_class_mismatch")
    if failures:
        eligible = []
        production_allowed_ids = (
            expected_tool_ids if static_allowed else []
        )
        lane_allowed = static_allowed
        expected_evidence_class = default_evidence_class
    return {
        "valid": not failures,
        "failures": sorted(set(failures)),
        "live_claimed": False,
        "vendor_claimed": True,
        "fanin_satisfied": bool(vendor_ready and not failures),
        "eligible_tool_ids": eligible,
        "production_allowed_tool_ids": production_allowed_ids,
        "lane_production_elevation_allowed": lane_allowed,
        "evidence_class": expected_evidence_class,
        "reference_audits": reference_audits,
        "sealed_root_authenticated": bool(
            _safe_dict(prebuilt.get("public")).get("authenticated") is True
            and sealed_root is not None
        ),
    }


def _audited_semantic_elevation_policy(
    *,
    certifier,
    repo_root: Path,
    spec: Mapping[str, Any],
    semantic_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently derive static or live-specialized elevation authority."""

    failures: list[str] = []
    expected_tool_ids = [
        str(value) for value in _safe_list(spec.get("tool_ids"))
    ]
    static_allowed = bool(spec.get("production_elevation_allowed"))
    lane_id = str(spec.get("lane_id") or "")
    if lane_id in getattr(certifier, "CHECKED_VENDOR_FANIN_SPECS", {}):
        return _audited_checked_vendor_fanin_policy(
            certifier=certifier,
            repo_root=repo_root,
            spec=spec,
            semantic_result=semantic_result,
        )
    live_summary = _safe_dict(
        semantic_result.get("live_specialized_receipt")
    )
    receipt = _safe_dict(semantic_result.get("receipt"))
    adapter_meta = _safe_dict(receipt.get("live_specialized_receipt"))
    configured = _safe_dict(
        getattr(certifier, "LIVE_SPECIALIZED_RECEIPT_SPECS", {}).get(
            str(spec.get("lane_id") or "")
        )
    )
    summary_eligible = [
        str(value)
        for value in _safe_list(live_summary.get("eligible_tool_ids"))
    ]
    adapter_eligible = [
        str(value)
        for value in _safe_list(adapter_meta.get("eligible_tool_ids"))
    ]
    live_claimed = bool(
        live_summary.get("valid") is True
        or adapter_meta
        or summary_eligible
        or semantic_result.get("evidence_class")
        == "live_specialized_semantic_receipt"
        or (
            semantic_result.get("production_elevation_allowed") is True
            and not static_allowed
        )
    )
    if not live_claimed:
        return {
            "valid": True,
            "failures": [],
            "live_claimed": False,
            "eligible_tool_ids": [],
            "production_allowed_tool_ids": (
                expected_tool_ids if static_allowed else []
            ),
            "lane_production_elevation_allowed": static_allowed,
            "evidence_class": str(spec.get("evidence_class") or ""),
        }

    if not configured:
        failures.append("live_specialized_policy_not_configured")
        return {
            "valid": False,
            "failures": failures,
            "live_claimed": True,
            "eligible_tool_ids": [],
            "production_allowed_tool_ids": (
                expected_tool_ids if static_allowed else []
            ),
            "lane_production_elevation_allowed": static_allowed,
            "evidence_class": str(spec.get("evidence_class") or ""),
        }

    live_relative = Path(str(configured.get("path") or ""))
    live_path = repo_root / live_relative
    try:
        live_receipt = json.loads(live_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        live_receipt = None
    if not isinstance(live_receipt, Mapping):
        failures.append("live_specialized_source_receipt_unreadable")
        live_receipt = {}

    for field_name in ("schema_version", "interface", "goal_id", "task_id"):
        if str(live_receipt.get(field_name) or "") != str(
            configured.get(field_name) or ""
        ):
            failures.append(
                f"live_specialized_source_{field_name}_mismatch"
            )
    digest_valid, self_digest, digest_failures = (
        certifier._live_receipt_digest_validation(live_receipt)
    )
    failures.extend(str(value) for value in digest_failures)
    if digest_valid is not True or not self_digest:
        failures.append("live_specialized_source_self_digest_invalid")
    if certifier.public_evidence_audit(
        live_receipt,
        repo_root=repo_root,
    ).get("satisfied") is not True:
        failures.append("live_specialized_source_public_evidence_invalid")

    declared_surfaces = certifier._live_receipt_surface_names(
        live_receipt
    )
    source_artifacts: list[dict[str, Any]] = []
    for raw_relative in _safe_list(configured.get("source_modules")):
        relative = Path(str(raw_relative))
        source_path = repo_root / relative
        digest = certifier.file_digest(source_path)
        dotted = relative.with_suffix("").as_posix().replace("/", ".")
        if not digest:
            failures.append(
                f"live_specialized_source_missing:{relative.as_posix()}"
            )
            continue
        if dotted not in declared_surfaces:
            failures.append(
                f"live_specialized_source_surface_unbound:{relative.as_posix()}"
            )
        source_artifacts.append(
            {
                "kind": "live_semantic_certifier_module",
                "path": relative.as_posix(),
                "sha256": digest,
                "artifact_class": "repository_source",
            }
        )
    source_artifacts.append(
        {
            "kind": "live_specialized_receipt",
            "path": live_relative.as_posix(),
            "sha256": certifier.file_digest(live_path),
            "artifact_class": "repository_source",
        }
    )
    source_validation = certifier._validate_artifact_identities(
        source_artifacts,
        repo_root=repo_root,
    )
    if source_validation.get("valid") is not True:
        failures.extend(
            f"live_specialized_source:{value}"
            for value in _safe_list(source_validation.get("failures"))
        )
    source_set_digest = certifier.content_digest(source_artifacts)
    expected_metadata = {
        "path": live_relative.as_posix(),
        "file_sha256": certifier.file_digest(live_path),
        "self_digest_sha256": self_digest,
        "interface": live_receipt.get("interface"),
        "schema_version": live_receipt.get("schema_version"),
        "goal_id": live_receipt.get("goal_id"),
        "task_id": live_receipt.get("task_id"),
        "source_set_digest_sha256": source_set_digest,
    }
    for field_name, expected in expected_metadata.items():
        if adapter_meta.get(field_name) != expected:
            failures.append(
                f"live_specialized_adapter_{field_name}_mismatch"
            )
        if field_name in {
            "path",
            "file_sha256",
            "self_digest_sha256",
            "source_set_digest_sha256",
        } and live_summary.get(field_name) != expected:
            failures.append(
                f"live_specialized_summary_{field_name}_mismatch"
            )
    if (
        live_summary.get("available") is not True
        or live_summary.get("valid") is not True
        or list(_safe_list(live_summary.get("failures")))
    ):
        failures.append("live_specialized_summary_not_valid")
    if list(_safe_list(live_summary.get("source_artifacts"))) != source_artifacts:
        failures.append("live_specialized_source_artifact_set_mismatch")
    if (
        summary_eligible != adapter_eligible
        or len(summary_eligible) != len(set(summary_eligible))
        or not set(summary_eligible) <= set(expected_tool_ids)
    ):
        failures.append("live_specialized_eligible_population_mismatch")

    family = str(configured.get("family") or "")
    adapter_checks = [
        _safe_dict(check)
        for check in _safe_list(receipt.get("checks"))
        if isinstance(check, Mapping)
    ]
    compact_tools = _safe_dict(semantic_result.get("per_tool"))
    summary_tool_failures = _safe_dict(
        live_summary.get("per_tool_failures")
    )
    for tool_id in summary_eligible:
        original_tool = certifier._live_tool_payload(
            live_receipt,
            family=family,
            tool_id=tool_id,
        )
        nested_digest_field = (
            "contribution_digest_sha256"
            if family == "kernel"
            else "receipt_digest_sha256"
            if family == "protocol"
            else None
        )
        if nested_digest_field:
            nested_computed = certifier.content_digest(
                {
                    key: value
                    for key, value in original_tool.items()
                    if key != nested_digest_field
                }
            )
            if not certifier._digest_matches(
                original_tool.get(nested_digest_field),
                nested_computed,
            ):
                failures.append(
                    f"{tool_id}:live_specialized_nested_digest_mismatch"
                )
        original_checks = certifier._live_tool_checks(
            live_receipt,
            original_tool,
            family=family,
            tool_id=tool_id,
        )
        if not certifier._live_tool_claims_production(
            live_receipt,
            original_tool,
            family=family,
            tool_id=tool_id,
        ):
            failures.append(
                f"{tool_id}:live_specialized_production_claim_invalid"
            )
        if not original_checks or any(
            str(check.get("status") or "") != "passed"
            for check in original_checks
        ):
            failures.append(
                f"{tool_id}:live_specialized_checks_incomplete"
            )
        for canonical_kind, aliases in (
            certifier._LIVE_SEMANTIC_KIND_ALIASES.items()
        ):
            if not any(
                str(check.get("status") or "") == "passed"
                and str(check.get("kind") or "").lower() in aliases
                for check in original_checks
            ):
                failures.append(
                    f"{tool_id}:live_specialized_kind_missing:"
                    f"{canonical_kind}"
                )

        tool_checks = [
            check
            for check in adapter_checks
            if str(check.get("tool_id") or "") == tool_id
        ]
        canonical_passes = {
            str(check.get("kind") or "")
            for check in tool_checks
            if check.get("status") == "passed"
            and str(check.get("kind") or "")
            in set(PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS)
        }
        binding_passed = any(
            check.get("status") == "passed"
            and str(check.get("check_id") or "").endswith(
                ".live_specialized.current_binding"
            )
            for check in tool_checks
        )
        if (
            set(PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS)
            - canonical_passes
            or not binding_passed
            or any(check.get("status") != "passed" for check in tool_checks)
        ):
            failures.append(
                f"{tool_id}:live_specialized_adapter_checks_invalid"
            )
        source_check_digests = {
            certifier.content_digest(check) for check in original_checks
        }
        if not source_check_digests <= {
            str(check.get("source_check_digest_sha256") or "")
            for check in tool_checks
        }:
            failures.append(
                f"{tool_id}:live_specialized_source_check_binding_missing"
            )
        if list(_safe_list(summary_tool_failures.get(tool_id))):
            failures.append(
                f"{tool_id}:live_specialized_summary_tool_failures"
            )

        compact_identity = _safe_dict(
            _safe_dict(compact_tools.get(tool_id)).get("identity")
        )
        compact_artifacts = [
            dict(item)
            for item in _safe_list(compact_identity.get("artifacts"))
            if isinstance(item, Mapping)
        ]
        if not all(item in compact_artifacts for item in source_artifacts):
            failures.append(
                f"{tool_id}:live_specialized_source_binding_missing"
            )
        if not any(
            artifact.get("artifact_class")
            in {
                "native_or_managed_binary",
                "managed_release_archive",
                "public_deployment_binding",
            }
            for artifact in compact_artifacts
        ):
            failures.append(
                f"{tool_id}:live_specialized_production_binding_missing"
            )
        declared_binary = certifier._live_receipt_binary_digest(
            live_receipt,
            original_tool,
            tool_id=tool_id,
        )
        if declared_binary and declared_binary not in {
            str(artifact.get("sha256") or "").removeprefix("sha256:")
            for artifact in compact_artifacts
            if artifact.get("kind")
            in {"semantic_executable", "executable"}
        }:
            failures.append(
                f"{tool_id}:live_specialized_binary_binding_mismatch"
            )

    eligible = summary_eligible if not failures else []
    production_allowed_ids = sorted(
        set(expected_tool_ids if static_allowed else ()) | set(eligible)
    )
    lane_allowed = bool(production_allowed_ids)
    expected_evidence_class = (
        "live_specialized_semantic_receipt"
        if eligible
        else str(spec.get("evidence_class") or "")
    )
    if (
        semantic_result.get("production_elevation_allowed") is not lane_allowed
    ):
        failures.append("live_specialized_lane_policy_flag_mismatch")
    if semantic_result.get("evidence_class") != expected_evidence_class:
        failures.append("live_specialized_lane_evidence_class_mismatch")
    if failures:
        eligible = []
        production_allowed_ids = (
            expected_tool_ids if static_allowed else []
        )
        lane_allowed = static_allowed
        expected_evidence_class = str(spec.get("evidence_class") or "")
    return {
        "valid": not failures,
        "failures": sorted(set(failures)),
        "live_claimed": True,
        "eligible_tool_ids": eligible,
        "production_allowed_tool_ids": production_allowed_ids,
        "lane_production_elevation_allowed": lane_allowed,
        "evidence_class": expected_evidence_class,
        "source_set_digest_sha256": source_set_digest,
    }


def _recompute_semantic_tool_payload(
    *,
    certifier,
    repo_root: Path,
    semantic_result: Mapping[str, Any],
    spec: Mapping[str, Any],
    tool_id: str,
    compact_tool: Mapping[str, Any],
    receipt_integrity: Mapping[str, Any],
    offline_observation: Mapping[str, Any],
    production_elevation_allowed: bool,
) -> dict[str, Any]:
    """Re-run semantic derivation and live artifact validation for one tool."""

    failures: list[str] = []
    receipt = _safe_dict(semantic_result.get("receipt"))
    certified_from_receipt, raw_checks, raw_reasons = (
        certifier._tool_certified_from_semantic_receipt(
            tool_id,
            receipt,
            certified_key=str(spec["certified_key"]),
            selector=str(spec.get("selector") or "root"),
        )
    )
    normalized = certifier._normalize_semantic_checks(
        tool_id,
        raw_checks,
    )
    checks = [check.to_dict() for check in normalized]

    derived_identity = certifier._semantic_tool_identity(
        tool_id,
        receipt,
        selector=str(spec.get("selector") or "root"),
        repo_root=repo_root,
    )
    # The semantic result is already a public projection, so repository paths
    # in its receipt are represented as ``<repo-root>/...``.  Resolve those
    # placeholders before deriving the canonical portable identity; otherwise
    # a valid deployment artifact loses both its digest and relative path when
    # it is independently re-read.
    for artifact in _safe_list(derived_identity.get("artifacts")):
        if not isinstance(artifact, dict):
            continue
        raw_derived_path = str(artifact.get("path") or "")
        if not raw_derived_path.startswith("<repo-root>/"):
            continue
        relative_path = raw_derived_path.removeprefix("<repo-root>/")
        resolved_path = repo_root / relative_path
        artifact["path"] = Path(relative_path).as_posix()
        artifact["sha256"] = certifier.file_digest(resolved_path)
    # A public ``<repo-root>/...`` scalar binding and the explicit portable
    # artifact row can normalize to the same identity.  Canonicalize only
    # exact-equal rows after recomputing the repository path and digest; a
    # differing kind, hash, class, or declared digest remains a population
    # mismatch.
    normalized_derived_artifacts: list[dict[str, Any]] = []
    for raw_artifact in _safe_list(derived_identity.get("artifacts")):
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact = dict(raw_artifact)
        if artifact not in normalized_derived_artifacts:
            normalized_derived_artifacts.append(artifact)
    derived_identity["artifacts"] = normalized_derived_artifacts
    compact_identity = _safe_dict(compact_tool.get("identity"))
    compact_artifacts = [
        dict(item)
        for item in _safe_list(compact_identity.get("artifacts"))
        if isinstance(item, Mapping)
    ]
    module_relative = Path(spec["module_relative"]).as_posix()
    module_artifact = {
        "kind": "semantic_certifier_module",
        "path": module_relative,
        "sha256": certifier.file_digest(
            repo_root / Path(spec["module_relative"])
        ),
        "artifact_class": "repository_source",
    }
    derived_artifacts = [
        dict(item)
        for item in _safe_list(derived_identity.get("artifacts"))
        if isinstance(item, Mapping)
    ] + [module_artifact]
    derivable_artifact_population_valid = bool(
        len(derived_artifacts) == len(compact_artifacts)
        and all(
            derived.get("kind") == compact.get("kind")
            and derived.get("path") == compact.get("path")
            and (
                derived.get("kind") == "semantic_executable"
                or all(
                    derived.get(field) == compact.get(field)
                    for field in (
                        "artifact_class",
                        "declared_digest",
                    )
                    if field in derived or field in compact
                )
            )
            for derived, compact in zip(
                derived_artifacts,
                compact_artifacts,
                strict=True,
            )
        )
    )
    if not derivable_artifact_population_valid:
        failures.append("semantic_artifact_population_mismatch")
    derived_scalar_identity = {
        key: derived_identity.get(key)
        for key in (
            "executable_path",
            "version_string",
            "identity_probed",
        )
    }
    compact_scalar_identity = {
        key: compact_identity.get(key)
        for key in (
            "executable_path",
            "version_string",
            "identity_probed",
        )
    }
    if derived_scalar_identity != compact_scalar_identity:
        failures.append("semantic_identity_scalar_mismatch")

    lock = certifier.load_lock(
        repo_root / certifier.DEFAULT_LOCK_RELATIVE
    )
    lock_entry = _safe_dict(
        certifier.lock_tools_by_id(lock).get(tool_id)
    )
    actual_artifacts: list[dict[str, Any]] = []
    synthetic_artifacts: list[dict[str, Any]] = []
    for artifact in compact_artifacts:
        kind = str(artifact.get("kind") or "")
        if kind == "semantic_certifier_module":
            if artifact != module_artifact:
                failures.append("semantic_module_artifact_mismatch")
            actual_artifacts.append(
                {
                    **module_artifact,
                    "path": module_relative,
                }
            )
            continue

        raw_path = str(artifact.get("path") or "")
        candidate_paths: list[Path] = []
        redacted_host_path = bool(
            raw_path == HOST_PATH_REDACTION
            or _redacted_host_basename(raw_path) is not None
        )
        if (
            raw_path
            and not redacted_host_path
            and raw_path != "<repo-root>"
        ):
            if raw_path.startswith("<repo-root>/"):
                candidate_paths.append(
                    repo_root / raw_path.removeprefix("<repo-root>/")
                )
            else:
                path = Path(raw_path)
                candidate_paths.append(
                    path if path.is_absolute() else repo_root / path
                )
        if redacted_host_path:
            candidate_paths.extend(
                _matching_approved_redacted_artifacts(
                    certifier=certifier,
                    lock_entry=lock_entry,
                    artifact=artifact,
                )
            )
        unique_candidates: list[Path] = []
        seen_candidates: set[str] = set()
        for candidate_path in candidate_paths:
            try:
                normalized_path = candidate_path.resolve()
            except OSError:
                normalized_path = candidate_path.absolute()
            key = str(normalized_path)
            if key not in seen_candidates:
                seen_candidates.add(key)
                unique_candidates.append(normalized_path)
        matching_paths = [
            candidate_path
            for candidate_path in unique_candidates
            if certifier.file_digest(candidate_path)
            == artifact.get("sha256")
            and (
                artifact.get("artifact_class")
                not in {
                    "native_or_managed_binary",
                    "launcher_script",
                    "generated_hermetic_shim",
                }
                or certifier.classify_executable_artifact(
                    candidate_path
                )
                == artifact.get("artifact_class")
            )
        ]
        if (
            not matching_paths
            and artifact.get("artifact_class")
            == "generated_hermetic_shim"
            and not production_elevation_allowed
        ):
            # Focused shadow certifiers use deleted temporary executables.
            # Their digest-bound row may remain regression evidence, but this
            # class is never production-capable or elevation-authoritative.
            synthetic_artifacts.append(dict(artifact))
            continue
        if not matching_paths:
            failures.append(
                f"{kind or 'artifact'}_live_identity_unavailable"
            )
            continue
        actual_artifact = dict(artifact)
        if raw_path.startswith("<") or Path(raw_path).is_absolute():
            actual_artifact["path"] = str(matching_paths[0])
        actual_artifacts.append(actual_artifact)

    live_artifact_validation = certifier._validate_artifact_identities(
        actual_artifacts,
        repo_root=repo_root,
    )
    public_artifact_validation = certifier.public_evidence_projection(
        live_artifact_validation,
        repo_root=repo_root,
    )
    live_validated_by_identity = {
        (
            str(item.get("kind") or ""),
            str(item.get("sha256") or ""),
        ): dict(item)
        for item in _safe_list(
            _safe_dict(public_artifact_validation).get("validated")
        )
        if isinstance(item, Mapping)
    }
    synthetic_by_identity = {
        (
            str(item.get("kind") or ""),
            str(item.get("sha256") or ""),
        ): {**item, "resolved_path": item.get("path")}
        for item in synthetic_artifacts
    }
    ordered_validated = [
        (
            live_validated_by_identity.get(
                (
                    str(item.get("kind") or ""),
                    str(item.get("sha256") or ""),
                )
            )
            or synthetic_by_identity.get(
                (
                    str(item.get("kind") or ""),
                    str(item.get("sha256") or ""),
                )
            )
        )
        for item in compact_artifacts
    ]
    if any(item is None for item in ordered_validated):
        failures.append("validated_artifact_population_mismatch")
    ordered_validated = [
        dict(item) for item in ordered_validated if item is not None
    ]
    production_bindings = [
        item
        for item in ordered_validated
        if item.get("artifact_class")
        in {
            "native_or_managed_binary",
            "managed_release_archive",
            "public_deployment_binding",
        }
    ]
    public_artifact_validation = {
        "valid": bool(
            _safe_dict(public_artifact_validation).get("valid")
            and not failures
        ),
        "failures": list(
            _safe_list(
                _safe_dict(public_artifact_validation).get("failures")
            )
        ),
        "validated": ordered_validated,
        "production_bindings": production_bindings,
        "has_production_binding": bool(production_bindings),
    }
    if (
        production_elevation_allowed
        and not production_bindings
    ):
        failures.append(
            "production_elevation_artifact_binding_missing"
        )
        public_artifact_validation["valid"] = False
    public_artifacts = [
        {
            key: value
            for key, value in item.items()
            if key != "resolved_path"
        }
        for item in _safe_list(
            _safe_dict(public_artifact_validation).get("validated")
        )
        if isinstance(item, Mapping)
    ]
    if public_artifacts != compact_artifacts:
        failures.append("semantic_artifact_identity_mismatch")

    identity = {
        **derived_scalar_identity,
        "artifacts": public_artifacts,
    }
    second_failed, second_reasons = (
        certifier.second_failed_check_blocks_promotion(normalized)
    )
    checks_complete = bool(normalized) and all(
        check.status == "passed" for check in normalized
    )
    artifact_validation_valid = bool(
        _safe_dict(public_artifact_validation).get("valid") is True
        and not failures
    )
    certified = bool(
        certified_from_receipt
        and receipt_integrity.get("valid") is True
        and offline_observation.get("satisfied") is True
        and checks_complete
        and artifact_validation_valid
        and not second_failed
    )
    block_reasons = list(raw_reasons) + list(
        _safe_list(
            _safe_dict(public_artifact_validation).get("failures")
        )
    )
    if second_failed:
        block_reasons.extend(second_reasons)
    full_tool = {
        "certified": certified,
        "block_reasons": block_reasons,
        "check_kinds_present": sorted(
            {
                str(check.get("kind"))
                for check in raw_checks
                if isinstance(check, Mapping) and check.get("kind")
            }
        ),
        "checks_retained_without_kind_collapse": True,
        "checks_passed": sum(
            1
            for check in raw_checks
            if isinstance(check, Mapping)
            and str(check.get("status")) == "passed"
        ),
        "checks_total": len(raw_checks),
        "checks": checks,
        "check_set_digest_sha256": certifier.content_digest(checks),
        "identity": identity,
        "artifact_validation": dict(public_artifact_validation),
        "handler_key": (
            f"{spec.get('property_lane_id') or semantic_result.get('lane_id')}"
            f"::{tool_id}"
        ),
    }
    expected_compact = certifier._compact_semantic_tool_projection(
        full_tool
    )
    compact_matches = bool(
        not failures and _safe_dict(compact_tool) == expected_compact
    )
    if not compact_matches:
        failures.append("semantic_tool_projection_mismatch")
    return {
        "valid": not failures and compact_matches,
        "full_tool": full_tool,
        "expected_compact": expected_compact,
        "public_artifact_validation": public_artifact_validation,
        "synthetic_artifacts_digest_only": bool(
            synthetic_artifacts
        ),
        "failures": failures,
    }


def _audit_semantic_lane_results(
    *,
    certifier,
    repo_root: Path,
    semantic_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate canonical lane receipts and every compact per-tool projection."""

    specs = {
        str(_safe_dict(spec).get("lane_id") or ""): _safe_dict(spec)
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
    }
    lane_ids = [str(row.get("lane_id") or "") for row in semantic_results]
    lanes = {
        str(row.get("lane_id") or ""): row
        for row in semantic_results
        if str(row.get("lane_id") or "")
    }
    failures: list[str] = []
    population_valid = bool(
        len(lane_ids) == len(specs)
        and len(lane_ids) == len(set(lane_ids))
        and set(lane_ids) == set(specs)
    )
    if not population_valid:
        failures.append("semantic_lane_population_mismatch")

    lane_verification: dict[str, dict[str, Any]] = {}
    for lane_id, spec in sorted(specs.items()):
        result = _safe_dict(lanes.get(lane_id))
        status_ran = result.get("status") == "ran"
        lane_failures: list[str] = []
        elevation_policy = _audited_semantic_elevation_policy(
            certifier=certifier,
            repo_root=repo_root,
            spec=spec,
            semantic_result=result,
        )
        if elevation_policy.get("valid") is not True:
            policy_prefix = (
                "checked_vendor"
                if elevation_policy.get("vendor_claimed")
                else "live_specialized"
            )
            lane_failures.extend(
                f"{policy_prefix}:{failure}"
                for failure in _safe_list(
                    elevation_policy.get("failures")
                )
            )
        production_allowed_tool_ids = {
            str(value)
            for value in _safe_list(
                elevation_policy.get("production_allowed_tool_ids")
            )
        }
        expected_tool_id_list = [
            str(value) for value in _safe_list(spec.get("tool_ids"))
        ]
        expected_tool_ids = set(expected_tool_id_list)
        declared_tool_id_list = [
            str(value) for value in _safe_list(result.get("tool_ids"))
        ]
        declared_tool_ids = set(declared_tool_id_list)
        per_tool_rows = _safe_dict(result.get("per_tool"))
        if (
            declared_tool_id_list != expected_tool_id_list
            or len(declared_tool_id_list) != len(declared_tool_ids)
        ):
            lane_failures.append("declared_tool_population_mismatch")
        if status_ran and (
            set(str(key) for key in per_tool_rows) != expected_tool_ids
            or len(per_tool_rows) != len(expected_tool_id_list)
        ):
            lane_failures.append("per_tool_population_mismatch")
        if not status_ran:
            lane_failures.append("semantic_lane_not_run")
            if (
                result.get("certified") is not False
                or result.get("digest_sha256") not in (None, "")
                or result.get("receipt") not in (None, {})
                or result.get("receipt_integrity") not in (None, {})
                or result.get("offline_observation") not in (None, {})
                or per_tool_rows
                or _safe_list(result.get("elevated_tool_ids"))
                or _safe_list(
                    result.get("semantically_usable_tool_ids")
                )
                or not _safe_list(result.get("block_reasons"))
            ):
                lane_failures.append(
                    "non_ran_lane_gap_structure_invalid"
                )

        receipt = result.get("receipt")
        receipt_mapping = (
            receipt if isinstance(receipt, Mapping) else {}
        )
        if status_ran and not receipt_mapping:
            lane_failures.append("canonical_receipt_missing")

        module_relative = Path(spec["module_relative"]).as_posix()
        expected_metadata: dict[str, Any] = {
            "property_lane_id": spec.get("property_lane_id") or lane_id,
            "certifier_family": spec.get("certifier_family") or lane_id,
            "interface": spec.get("interface"),
            "module": module_relative,
            "evidence_class": elevation_policy.get("evidence_class"),
            "production_elevation_allowed": bool(
                elevation_policy.get(
                    "lane_production_elevation_allowed"
                )
            ),
            "usable_elevation_allowed": bool(
                spec.get("usable_elevation_allowed", True)
            ),
        }
        for field_name, expected_value in expected_metadata.items():
            if result.get(field_name) != expected_value:
                lane_failures.append(f"lane_{field_name}_mismatch")

        expected_identity: dict[str, Any] = {
            "interface": spec.get("interface"),
        }
        module = None
        try:
            module_path = repo_root / Path(spec["module_relative"])
            module = certifier._load_module_from_path(
                module_path,
                f"fvt_semantic_contract_{lane_id}",
            )
            expected_identity.update(
                {
                    "schema_version": getattr(
                        module, "SCHEMA_VERSION", None
                    ),
                    "goal_id": getattr(module, "GOAL_ID", None),
                    "task_id": getattr(module, "TASK_ID", None),
                }
            )
            if (
                result.get("certifier_module_sha256")
                != certifier.file_digest(module_path)
            ):
                lane_failures.append("certifier_module_digest_mismatch")
        except Exception as exc:  # noqa: BLE001
            lane_failures.append(
                f"semantic_identity_contract_unavailable:{type(exc).__name__}"
            )
        if status_ran:
            for field_name, expected_value in expected_identity.items():
                if (
                    expected_value is not None
                    and receipt_mapping.get(field_name)
                    != expected_value
                ):
                    lane_failures.append(
                        f"receipt_{field_name}_mismatch"
                    )
            if (
                result.get("interface") != spec.get("interface")
                or result.get("receipt_goal_id")
                != receipt_mapping.get("goal_id")
                or result.get("receipt_task_id")
                != receipt_mapping.get("task_id")
            ):
                lane_failures.append("lane_receipt_identity_mismatch")

        lane_digest = str(result.get("digest_sha256") or "")
        if (
            status_ran
            and (
                not SHA256_RE.fullmatch(lane_digest)
                or lane_digest != certifier.content_digest(receipt_mapping)
            )
        ):
            lane_failures.append("outer_receipt_digest_mismatch")
        digest_fields = [
            field_name
            for field_name in (
                "receipt_digest_sha256",
                "certificate_digest_sha256",
                "digest_sha256",
            )
            if field_name in receipt_mapping
        ]
        if status_ran and not digest_fields:
            lane_failures.append("declared_receipt_digest_missing")
        for field_name in digest_fields:
            computed = certifier.content_digest(
                {
                    key: value
                    for key, value in receipt_mapping.items()
                    if key != field_name
                }
            )
            if str(receipt_mapping.get(field_name) or "") not in {
                computed,
                f"sha256:{computed}",
            }:
                lane_failures.append(
                    f"receipt_{field_name}_mismatch"
                )

        tool_verification: dict[str, dict[str, Any]] = {}
        if status_ran:
            independent_integrity: dict[str, Any] = {}
            independent_offline: dict[str, Any] = {}
            if module is None:
                lane_failures.append("independent_receipt_audit_unavailable")
            else:
                independent_integrity = (
                    certifier._validate_semantic_receipt_integrity(
                        receipt_mapping,
                        spec=spec,
                        module=module,
                    )
                )
                if (
                    _safe_dict(result.get("receipt_integrity"))
                    != independent_integrity
                    or independent_integrity.get("valid") is not True
                ):
                    lane_failures.append(
                        "declared_receipt_integrity_mismatch"
                    )
                independent_offline = certifier._offline_observation(
                    receipt_mapping,
                    production_elevation_allowed=bool(
                        elevation_policy.get(
                            "lane_production_elevation_allowed"
                        )
                    ),
                )
                if (
                    _safe_dict(result.get("offline_observation"))
                    != independent_offline
                    or independent_offline.get("satisfied") is not True
                ):
                    lane_failures.append(
                        "declared_offline_observation_mismatch"
                    )
            reconstructed_tools: dict[str, dict[str, Any]] = {}
            for tool_id in expected_tool_id_list:
                per_tool = _safe_dict(per_tool_rows.get(tool_id))
                recomputed_tool = _recompute_semantic_tool_payload(
                    certifier=certifier,
                    repo_root=repo_root,
                    semantic_result=result,
                    spec=spec,
                    tool_id=tool_id,
                    compact_tool=per_tool,
                    receipt_integrity=independent_integrity,
                    offline_observation=independent_offline,
                    production_elevation_allowed=(
                        tool_id in production_allowed_tool_ids
                    ),
                )
                reconstructed_tool = _safe_dict(
                    recomputed_tool.get("full_tool")
                )
                reconstructed_tools[tool_id] = reconstructed_tool
                if recomputed_tool.get("valid") is not True:
                    lane_failures.extend(
                        f"{tool_id}:{failure}"
                        for failure in _safe_list(
                            recomputed_tool.get("failures")
                        )
                    )
                tool_verification[tool_id] = {
                    "checks_match_canonical_receipt": bool(
                        recomputed_tool.get("valid")
                    ),
                    "check_set_digest_sha256": reconstructed_tool.get(
                        "check_set_digest_sha256"
                    ),
                    "certified": bool(
                        reconstructed_tool.get("certified")
                    ),
                    "block_reasons": list(
                        _safe_list(
                            reconstructed_tool.get("block_reasons")
                        )
                    ),
                    "artifact_validation_valid": bool(
                        _safe_dict(
                            recomputed_tool.get(
                                "public_artifact_validation"
                            )
                        ).get("valid")
                        and recomputed_tool.get("valid")
                    ),
                    "expected_validated_artifacts": list(
                        _safe_list(
                            _safe_dict(
                                recomputed_tool.get(
                                    "public_artifact_validation"
                                )
                            ).get("validated")
                        )
                    ),
                    "expected_production_bindings": (
                        list(
                            _safe_list(
                                _safe_dict(
                                    recomputed_tool.get(
                                        "public_artifact_validation"
                                    )
                                ).get("production_bindings")
                            )
                        )
                    ),
                }

            expected_usable = [
                tool_id
                for tool_id in expected_tool_id_list
                if _safe_dict(
                    reconstructed_tools.get(tool_id)
                ).get("certified")
                is True
            ]
            expected_elevated = [
                tool_id
                for tool_id in expected_usable
                if tool_id in production_allowed_tool_ids
            ]
            expected_lane_certified = bool(
                receipt_mapping.get(str(spec["certified_key"]))
                or receipt_mapping.get("certified")
            ) and independent_integrity.get("valid") is True
            expected_lane_block_reasons = list(
                _safe_list(independent_integrity.get("failures"))
            )
            if independent_offline.get("satisfied") is not True:
                expected_lane_block_reasons.append(
                    "offline_observation_failed"
                )
            if (
                result.get("certified") is not expected_lane_certified
                or list(
                    _safe_list(
                        result.get("semantically_usable_tool_ids")
                    )
                )
                != expected_usable
                or list(
                    _safe_list(result.get("elevated_tool_ids"))
                )
                != expected_elevated
                or list(_safe_list(result.get("block_reasons")))
                != expected_lane_block_reasons
            ):
                lane_failures.append(
                    "lane_semantic_outcomes_not_independently_derived"
                )

        if lane_failures:
            failures.extend(
                f"{lane_id}:{failure}" for failure in lane_failures
            )
        lane_verification[lane_id] = {
            "valid": not lane_failures,
            "structurally_valid": not [
                failure
                for failure in lane_failures
                if failure != "semantic_lane_not_run"
            ],
            "status_ran": status_ran,
            "expected_tool_ids": sorted(expected_tool_ids),
            "elevation_policy": elevation_policy,
            "failures": lane_failures,
            "tools": tool_verification,
        }

    structural_failures = [
        failure
        for failure in failures
        if not str(failure).endswith(":semantic_lane_not_run")
    ]
    return {
        "valid": population_valid and not failures,
        "complete": population_valid and not failures,
        "structurally_valid": (
            population_valid and not structural_failures
        ),
        "population_valid": population_valid,
        "expected_lane_count": len(specs),
        "observed_lane_count": len(lane_ids),
        "lanes": lane_verification,
        "failures": sorted(set(failures)),
        "structural_failures": sorted(set(structural_failures)),
    }


def _compact_semantic_lane(
    result: Mapping[str, Any],
    *,
    per_tool_evidence_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Keep lane/per-tool identities and digests without bulk raw receipts."""

    integrity = _safe_dict(result.get("receipt_integrity"))
    supplied_digests = _safe_dict(per_tool_evidence_digests)
    return {
        "lane_id": result.get("lane_id"),
        "status": result.get("status"),
        "digest_sha256": result.get("digest_sha256"),
        "block_reasons": list(_safe_list(result.get("block_reasons"))),
        "receipt_integrity_valid": integrity.get("valid"),
        "per_tool_bindings": {
            str(tool_id): {
                "check_set_digest_sha256": _safe_dict(per_tool).get(
                    "check_set_digest_sha256"
                ),
                "tool_evidence_digest_sha256": (
                    supplied_digests.get(str(tool_id))
                    or content_digest(_safe_dict(per_tool))
                ),
                "artifact_validation_valid": _safe_dict(
                    _safe_dict(per_tool).get("artifact_validation")
                ).get("valid"),
            }
            for tool_id, per_tool in _safe_dict(result.get("per_tool")).items()
        },
    }


def _audit_platform_support(
    *,
    certifier,
    repo_root: Path,
    managed: Mapping[str, Any],
    certificate_lock: Mapping[str, Any],
    tools: Mapping[str, Mapping[str, Any]],
    authority_roles: Mapping[str, Any],
    semantic_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute managed platform rows, exceptions, and semantic support."""

    failures: list[str] = []
    lock = certifier.load_lock(
        repo_root / certifier.DEFAULT_LOCK_RELATIVE
    )
    tools_index = certifier.lock_tools_by_id(lock)
    host_platform = certifier.observed_platform_id()
    global_platforms = [
        str(item)
        for item in _safe_list(
            _safe_dict(lock.get("platform_policy")).get(
                "supported_platforms"
            )
        )
    ]
    expected_certificate_lock = {
        "path": certifier.DEFAULT_LOCK_RELATIVE.as_posix(),
        "interface": lock.get("interface"),
        "schema_version": lock.get("schema_version"),
        "goal_id": lock.get("goal_id"),
        "task_id": lock.get("task_id"),
        "digest_sha256": certifier.content_digest(lock),
        "host_platform": host_platform,
    }
    lock_binding_valid = bool(
        _safe_dict(certificate_lock) == expected_certificate_lock
    )
    if not lock_binding_valid:
        failures.append("certificate_lock_binding_mismatch")
    expected_rows = [
        certifier.tool_platform_support(
            tools_index[tool_id],
            host_platform=host_platform,
            global_supported_platforms=global_platforms,
        )
        for tool_id in sorted(tools_index)
    ]
    observed_rows = [
        dict(row)
        for row in _safe_list(managed.get("platform_rows"))
        if isinstance(row, Mapping)
    ]
    row_ids = [str(row.get("tool_id") or "") for row in observed_rows]
    if (
        observed_rows != expected_rows
        or len(row_ids) != len(set(row_ids))
        or any(not tool_id for tool_id in row_ids)
        or managed.get("host_platform") != host_platform
        or list(
            _safe_list(managed.get("global_supported_platforms"))
        )
        != global_platforms
        or bool(managed.get("host_globally_supported"))
        != (host_platform in set(global_platforms))
    ):
        failures.append("platform_rows_not_independently_derived")

    role_tools = _safe_dict(authority_roles.get("tools"))

    def category_for(tool_id: str) -> str:
        role = str(
            _safe_dict(role_tools.get(tool_id)).get("role")
            or "unclassified"
        )
        return (
            "dependency"
            if role == "support"
            or tool_id in {"opam", "stack", "maude"}
            else "capability"
        )

    expected_exceptions: list[dict[str, Any]] = []
    supported_capabilities: list[str] = []
    supported_dependencies: list[str] = []
    expected_row_by_tool = {
        str(row.get("tool_id") or ""): row for row in expected_rows
    }
    for row in expected_rows:
        tool_id = str(row.get("tool_id") or "")
        if row.get("managed") is not True:
            continue
        category = category_for(tool_id)
        if row.get("supported") is True:
            (
                supported_dependencies
                if category == "dependency"
                else supported_capabilities
            ).append(tool_id)
            continue
        if (
            row.get("classification") == "unsupported_here"
            and row.get("exception_eligible") is True
        ):
            expected_exceptions.append(
                {
                    "tool_id": tool_id,
                    "host_platform": host_platform,
                    "declared_platforms": list(
                        _safe_list(row.get("declared_platforms"))
                    ),
                    "basis": row.get("basis"),
                    "classification": "unsupported_here",
                    "category": category,
                    "narrow_scope": True,
                    "complete": False,
                    "production_certified": False,
                }
            )
        else:
            failures.append(
                f"platform:{tool_id}:unsupported_not_exception_eligible"
            )

    observed_exceptions = [
        dict(item)
        for item in _safe_list(managed.get("platform_exceptions"))
        if isinstance(item, Mapping)
    ]
    exception_ids = [
        str(item.get("tool_id") or "") for item in observed_exceptions
    ]
    exception_tools_not_promoted = all(
        bool(tool_id)
        and _safe_dict(tools.get(tool_id)).get(
            "production_certified"
        )
        is not True
        and _safe_dict(expected_row_by_tool.get(tool_id)).get(
            "supported"
        )
        is False
        for tool_id in exception_ids
    )
    exceptions_valid = bool(
        observed_exceptions == expected_exceptions
        and len(exception_ids) == len(set(exception_ids))
        and all(exception_ids)
        and exception_tools_not_promoted
    )
    if not exceptions_valid:
        failures.append("platform_exceptions_not_exactly_derived")

    supported_lists_valid = bool(
        list(
            _safe_list(
                managed.get(
                    "supported_managed_capability_tool_ids"
                )
            )
        )
        == supported_capabilities
        and list(
            _safe_list(
                managed.get(
                    "supported_managed_dependency_tool_ids"
                )
            )
        )
        == supported_dependencies
    )
    if not supported_lists_valid:
        failures.append("supported_managed_tool_lists_not_derived")

    tool_population_valid = set(tools) == set(tools_index)
    if not tool_population_valid:
        failures.append("certificate_tool_population_mismatch")
    semantic_lane_ids = [
        str(item.get("lane_id") or "")
        for item in semantic_results
        if isinstance(item, Mapping)
    ]
    semantic_results_by_lane = {
        str(item.get("lane_id") or ""): _safe_dict(item)
        for item in semantic_results
        if isinstance(item, Mapping)
        and str(item.get("lane_id") or "")
    }
    if (
        any(not lane_id for lane_id in semantic_lane_ids)
        or len(semantic_lane_ids) != len(set(semantic_lane_ids))
    ):
        failures.append("semantic_lane_population_not_unique")
    non_production_semantic_artifact_bindings: dict[
        str, list[dict[str, Any]]
    ] = {}
    for raw_spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        spec = _safe_dict(raw_spec)
        if spec.get("production_elevation_allowed") is not False:
            continue
        lane_id = str(spec.get("lane_id") or "")
        result = _safe_dict(semantic_results_by_lane.get(lane_id))
        receipt = result.get("receipt")
        expected_tool_ids = {
            str(item) for item in _safe_list(spec.get("tool_ids"))
        }
        raw_result_tool_ids = [
            str(item) for item in _safe_list(result.get("tool_ids"))
        ]
        result_tool_ids = set(raw_result_tool_ids)
        if (
            result.get("status") != "ran"
            or result.get("certified") is not True
            or result.get("production_elevation_allowed") is not False
            or result.get("evidence_class") != spec.get("evidence_class")
            or result.get("interface") != spec.get("interface")
            or result.get("certifier_family")
            != spec.get("certifier_family")
            or result.get("property_lane_id")
            != spec.get("property_lane_id")
            or len(raw_result_tool_ids) != len(result_tool_ids)
            or result_tool_ids != expected_tool_ids
            or not isinstance(receipt, Mapping)
            or result.get("digest_sha256")
            != certifier.content_digest(receipt)
            or _safe_dict(result.get("receipt_integrity")).get("valid")
            is not True
            or _safe_dict(result.get("offline_observation")).get(
                "satisfied"
            )
            is not True
        ):
            continue
        per_tool = _safe_dict(result.get("per_tool"))
        if set(str(tool_id) for tool_id in per_tool) != expected_tool_ids:
            continue
        for raw_tool_id in _safe_list(spec.get("tool_ids")):
            tool_id = str(raw_tool_id)
            tool_result = _safe_dict(per_tool.get(tool_id))
            artifact_validation = _safe_dict(
                tool_result.get("artifact_validation")
            )
            if (
                tool_id not in result_tool_ids
                or tool_result.get("certified") is not True
                or artifact_validation.get("valid") is not True
                or artifact_validation.get("has_production_binding")
                is not False
            ):
                continue
            identity = _safe_dict(tool_result.get("identity"))
            for raw_artifact in _safe_list(identity.get("artifacts")):
                if not isinstance(raw_artifact, Mapping):
                    continue
                artifact = dict(raw_artifact)
                if (
                    artifact.get("kind") == "semantic_executable"
                    and artifact.get("artifact_class")
                    == "generated_hermetic_shim"
                ):
                    non_production_semantic_artifact_bindings.setdefault(
                        tool_id, []
                    ).append(
                        {
                            "lane_id": lane_id,
                            "artifact": artifact,
                        }
                    )

    reconstructed_tool_certs: dict[str, Any] = {}
    semantic_artifact_population_failures: list[str] = []
    for tool_id, bindings in sorted(
        non_production_semantic_artifact_bindings.items()
    ):
        global_artifacts = [
            dict(artifact)
            for artifact in _safe_list(
                _safe_dict(tools.get(tool_id)).get(
                    "artifact_identities"
                )
            )
            if isinstance(artifact, Mapping)
        ]
        seen_bindings: set[str] = set()
        for binding in bindings:
            artifact = binding["artifact"]
            binding_key = json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            population = sum(
                candidate == artifact for candidate in global_artifacts
            )
            if binding_key in seen_bindings or population != 1:
                semantic_artifact_population_failures.append(
                    f"{tool_id}:{binding['lane_id']}:"
                    "semantic_tool_artifact_population_mismatch"
                )
            seen_bindings.add(binding_key)
    if semantic_artifact_population_failures:
        failures.append("semantic_tool_artifact_population_mismatch")

    primary_executable_binding_failures: list[str] = []
    for tool_id, raw_tool in sorted(tools.items()):
        tool = _safe_dict(raw_tool)
        primary_artifacts = [
            dict(artifact)
            for artifact in _safe_list(tool.get("artifact_identities"))
            if isinstance(artifact, Mapping)
            and artifact.get("kind") == "executable"
        ]
        scalar_path = tool.get("executable_path")
        scalar_digest = tool.get("executable_sha256")
        scalar_class = tool.get("executable_artifact_class")
        scalar_path_present = scalar_path not in (None, "")
        scalar_digest_present = scalar_digest not in (None, "")
        scalar_class_present = scalar_class not in (
            None,
            "",
            "none",
        )
        scalars_present = bool(
            scalar_path_present
            or scalar_digest_present
            or scalar_class_present
        )
        scalar_binding_valid = (
            len(primary_artifacts) == 1
            and scalar_path_present
            and scalar_digest_present
            and scalar_class_present
            and primary_artifacts[0].get("path") == scalar_path
            and primary_artifacts[0].get("sha256") == scalar_digest
            and primary_artifacts[0].get("artifact_class")
            == scalar_class
        )
        if (scalars_present and not scalar_binding_valid) or (
            not scalars_present and primary_artifacts
        ):
            primary_executable_binding_failures.append(
                f"{tool_id}:primary_executable_artifact_binding_mismatch"
            )
    if primary_executable_binding_failures:
        failures.append("primary_executable_artifact_binding_mismatch")

    live_artifact_failures: list[str] = [
        *semantic_artifact_population_failures,
        *primary_executable_binding_failures,
    ]
    non_production_artifact_omissions: list[dict[str, Any]] = []
    tool_field_names = set(
        certifier.ToolCertification.__dataclass_fields__
    )
    for tool_id in sorted(tools_index):
        tool = _safe_dict(tools.get(tool_id))
        entry = _safe_dict(tools_index.get(tool_id))
        check_models = [
            certifier.CheckResult(
                check_id=str(check.get("check_id") or ""),
                kind=str(check.get("kind") or ""),
                status=str(check.get("status") or "failed"),
                expected=str(check.get("expected") or ""),
                observed=str(check.get("observed") or ""),
                detail=str(check.get("detail") or ""),
                evidence=_safe_dict(check.get("evidence")),
            )
            for check in (
                _safe_dict(item)
                for item in _safe_list(tool.get("checks"))
                if isinstance(item, Mapping)
            )
        ]
        actual_artifacts: list[dict[str, Any]] = []
        for raw_artifact in _safe_list(
            tool.get("artifact_identities")
        ):
            if not isinstance(raw_artifact, Mapping):
                live_artifact_failures.append(
                    f"{tool_id}:artifact_not_mapping"
                )
                continue
            artifact = dict(raw_artifact)
            raw_path = str(artifact.get("path") or "")
            redacted_host_path = bool(
                raw_path == HOST_PATH_REDACTION
                or _redacted_host_basename(raw_path) is not None
            )
            if redacted_host_path:
                matches = list(
                    _matching_approved_redacted_artifacts(
                        certifier=certifier,
                        lock_entry=entry,
                        artifact=artifact,
                    )
                )
                if not matches:
                    semantic_bindings = (
                        non_production_semantic_artifact_bindings.get(
                            tool_id, []
                        )
                    )
                    semantic_binding_index = next(
                        (
                            index
                            for index, binding in enumerate(
                                semantic_bindings
                            )
                            if artifact == binding["artifact"]
                        ),
                        None,
                    )
                    semantic_binding = (
                        semantic_bindings[semantic_binding_index]
                        if semantic_binding_index is not None
                        else None
                    )
                    if (
                        artifact.get("kind") == "semantic_executable"
                        and artifact.get("artifact_class")
                        == "generated_hermetic_shim"
                        and tool.get("production_certified") is False
                        and semantic_binding is not None
                    ):
                        non_production_artifact_omissions.append(
                            {
                                "tool_id": tool_id,
                                "lane_id": semantic_binding["lane_id"],
                                "kind": artifact["kind"],
                                "sha256": artifact.get("sha256"),
                                "artifact_class": artifact[
                                    "artifact_class"
                                ],
                                "basis": (
                                    "exact_non_production_semantic_lane_"
                                    "binding_without_live_managed_authority"
                                ),
                            }
                        )
                        semantic_bindings.pop(semantic_binding_index)
                        continue
                    live_artifact_failures.append(
                        f"{tool_id}:artifact_live_identity_unavailable"
                    )
                    continue
                artifact["path"] = str(matches[0])
            elif raw_path.startswith("<repo-root>/"):
                artifact["path"] = str(
                    repo_root
                    / raw_path.removeprefix("<repo-root>/")
                )
            actual_artifacts.append(artifact)

        kwargs = {
            key: value
            for key, value in tool.items()
            if key in tool_field_names
            and key not in {"checks", "artifact_identities"}
        }
        kwargs["tool_id"] = tool_id
        kwargs["checks"] = check_models
        kwargs["artifact_identities"] = actual_artifacts
        try:
            reconstructed_tool_certs[tool_id] = (
                certifier.ToolCertification(**kwargs)
            )
        except TypeError as exc:
            live_artifact_failures.append(
                f"{tool_id}:tool_model_invalid:{type(exc).__name__}"
            )

    reconstructed_managed = certifier.build_managed_deployment_readiness(
        lock=lock,
        tools_index=tools_index,
        tool_certs=reconstructed_tool_certs,
        authority_roles=authority_roles,
        repo_root=repo_root,
    )
    # Deleted hermetic executables are intentionally omitted only after their
    # exact non-production semantic binding is independently verified above.
    # Preserve their disclosed artifact class in the reconstructed blocker
    # projection so comparison with the live certificate is lossless, without
    # treating the missing shim as an installed or authority-bearing artifact.
    omitted_classes_by_tool: dict[str, set[str]] = {}
    for omission in non_production_artifact_omissions:
        omitted_classes_by_tool.setdefault(
            str(omission.get("tool_id") or ""),
            set(),
        ).add(str(omission.get("artifact_class") or ""))
    for blocker_field in (
        "capability_blockers",
        "dependency_blockers",
        "all_blockers",
    ):
        for blocker in _safe_list(
            reconstructed_managed.get(blocker_field)
        ):
            if not isinstance(blocker, dict):
                continue
            omitted_classes = omitted_classes_by_tool.get(
                str(blocker.get("tool_id") or "")
            )
            if omitted_classes:
                blocker["artifact_classes"] = sorted(
                    {
                        str(item)
                        for item in _safe_list(
                            blocker.get("artifact_classes")
                        )
                        if str(item)
                    }
                    | {item for item in omitted_classes if item}
                )
    reconstructed_managed = certifier.public_evidence_projection(
        reconstructed_managed,
        repo_root=repo_root,
    )
    # A deleted hermetic shim cannot be passed to the managed-readiness
    # validator as a live artifact. Preserve only its independently bound,
    # non-authoritative class in the reconstructed blocker metadata so the
    # comparison remains exact without granting artifact validity.
    for omission in non_production_artifact_omissions:
        tool_id = omission["tool_id"]
        artifact_class = omission["artifact_class"]
        for blocker_key in (
            "capability_blockers",
            "dependency_blockers",
            "all_blockers",
        ):
            for blocker in _safe_list(
                reconstructed_managed.get(blocker_key)
            ):
                if (
                    isinstance(blocker, dict)
                    and blocker.get("tool_id") == tool_id
                ):
                    blocker["artifact_classes"] = sorted(
                        {
                            *(
                                str(item)
                                for item in _safe_list(
                                    blocker.get("artifact_classes")
                                )
                            ),
                            str(artifact_class),
                        }
                    )
    blockers_and_ready_valid = bool(
        not live_artifact_failures
        and reconstructed_managed == _safe_dict(managed)
    )
    if not blockers_and_ready_valid:
        failures.append("managed_blockers_or_ready_not_derived")

    exception_id_set = set(exception_ids)
    non_ran_lane_support: dict[str, dict[str, Any]] = {}
    for result in semantic_results:
        if result.get("status") == "ran":
            continue
        lane_id = str(result.get("lane_id") or "unknown")
        tool_ids = [
            str(item) for item in _safe_list(result.get("tool_ids"))
        ]
        supported_siblings = sorted(
            tool_id
            for tool_id in tool_ids
            if _safe_dict(expected_row_by_tool.get(tool_id)).get(
                "supported"
            )
            is True
        )
        exception_siblings = sorted(
            tool_id
            for tool_id in tool_ids
            if tool_id in exception_id_set
        )
        unknown_siblings = sorted(
            set(tool_ids)
            - set(supported_siblings)
            - set(exception_siblings)
        )
        if supported_siblings:
            classification = "supported_semantic_blocker"
        elif (
            tool_ids
            and len(exception_siblings) == len(tool_ids)
            and not unknown_siblings
        ):
            classification = "platform_exception_incomplete"
        else:
            classification = "mixed_or_unknown_support_blocker"
            failures.append(
                f"platform:{lane_id}:non_ran_support_ambiguous"
            )
        non_ran_lane_support[lane_id] = {
            "classification": classification,
            "tool_ids": tool_ids,
            "supported_tool_ids": supported_siblings,
            "exception_tool_ids": exception_siblings,
            "unknown_tool_ids": unknown_siblings,
            "can_elevate": False,
            "complete": False,
        }

    return {
        "valid": not failures,
        "certificate_lock_binding_valid": lock_binding_valid,
        "canonical_lock_digest_sha256": expected_certificate_lock[
            "digest_sha256"
        ],
        "platform_rows_valid": observed_rows == expected_rows,
        "platform_exceptions_valid": exceptions_valid,
        "supported_lists_valid": supported_lists_valid,
        "managed_blockers_and_ready_valid": blockers_and_ready_valid,
        "live_artifact_failures": live_artifact_failures,
        "semantic_artifact_population_failures": sorted(
            set(semantic_artifact_population_failures)
        ),
        "primary_executable_binding_failures": sorted(
            set(primary_executable_binding_failures)
        ),
        "non_production_artifact_omissions": sorted(
            non_production_artifact_omissions,
            key=lambda item: (
                str(item.get("tool_id") or ""),
                str(item.get("lane_id") or ""),
                str(item.get("kind") or ""),
                str(item.get("sha256") or ""),
            ),
        ),
        "exception_tools_not_promoted": exception_tools_not_promoted,
        "supported_managed_capability_tool_ids": (
            supported_capabilities
        ),
        "supported_managed_dependency_tool_ids": (
            supported_dependencies
        ),
        "platform_exception_tool_ids": sorted(exception_id_set),
        "non_ran_lane_support": non_ran_lane_support,
        "failures": sorted(set(failures)),
    }


def _audit_required_elevations(
    *,
    certifier,
    repo_root: Path,
    certificate: Mapping[str, Any],
    semantic_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind global production authority and required semantic evidence."""

    failures: list[str] = []
    role_aware = _safe_dict(certificate.get("role_aware"))
    promotion = _safe_dict(certificate.get("promotion"))
    tools = {
        str(item.get("tool_id") or ""): _safe_dict(item)
        for item in _safe_list(certificate.get("tools"))
        if isinstance(item, Mapping)
        and str(item.get("tool_id") or "")
    }
    semantic_results = {
        str(item.get("lane_id") or ""): _safe_dict(item)
        for item in _safe_list(
            certificate.get("semantic_lane_results")
        )
        if isinstance(item, Mapping)
        and str(item.get("lane_id") or "")
    }
    specs_by_tool: dict[str, tuple[dict[str, Any], str]] = {}
    for raw_spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        spec = _safe_dict(raw_spec)
        for raw_tool_id in _safe_list(spec.get("tool_ids")):
            specs_by_tool[str(raw_tool_id)] = (
                spec,
                str(spec.get("lane_id") or ""),
            )

    # Re-run every direct lane that can produce full semantic
    # positive/negative/mutation/replay evidence.  Today that is the SMT lane;
    # deriving the population from PROPERTY_LANES avoids trusting a claimed
    # production flag or a hard-coded tool list in the certificate.
    lock = certifier.load_lock(
        repo_root / certifier.DEFAULT_LOCK_RELATIVE
    )
    lock_tools = certifier.lock_tools_by_id(lock)
    tool_lane_map: dict[str, list[str]] = {}
    tool_check_kind: dict[str, str] = {}
    for lane in certifier.PROPERTY_LANES:
        lane_id = str(lane.get("lane_id") or "")
        check_kind = str(lane.get("check_kind") or "identity_only")
        for raw_tool_id in _safe_list(lane.get("tool_ids")):
            tool_id = str(raw_tool_id)
            tool_lane_map.setdefault(tool_id, []).append(lane_id)
            prior = tool_check_kind.get(tool_id)
            if prior is None or check_kind == "smtlib":
                tool_check_kind[tool_id] = check_kind
    direct_candidate_ids = sorted(
        tool_id
        for tool_id, check_kind in tool_check_kind.items()
        if check_kind == "smtlib" and tool_id in lock_tools
    )
    direct_env = certifier.offline_env(os.environ)
    direct_env["PYTHONPATH"] = os.pathsep.join(
        (
            str(repo_root),
            str(repo_root / "ipfs_datasets_py"),
            *(
                (str(direct_env.get("PYTHONPATH")),)
                if direct_env.get("PYTHONPATH")
                else ()
            ),
        )
    )
    direct_certs = {
        tool_id: certifier.certify_tool(
            lock_tools[tool_id],
            lane_ids=tool_lane_map.get(tool_id, []),
            check_kind=tool_check_kind[tool_id],
            env=direct_env,
        )
        for tool_id in direct_candidate_ids
    }
    direct_quarantine = certifier.quarantine_smt_disagreement(
        direct_certs
    )
    if direct_quarantine is not None:
        for tool_id in direct_quarantine.promotion_blocked_tool_ids:
            direct_cert = direct_certs.get(tool_id)
            if direct_cert is not None:
                direct_cert.production_certified = False
                direct_cert.promotion_blocked = True
    canonical_roles = certifier.load_authority_roles(repo_root)
    certifier.apply_role_aware_demotions(
        direct_certs,
        canonical_roles,
    )
    expected_direct_production_ids = sorted(
        tool_id
        for tool_id, direct_cert in direct_certs.items()
        if direct_cert.production_certified
    )

    declared_required = [
        str(item)
        for item in _safe_list(
            role_aware.get("required_baseline_elevations")
        )
    ]
    expected_required = list(REQUIRED_SEMANTIC_ELEVATIONS)
    if (
        declared_required != expected_required
        or len(declared_required) != len(set(declared_required))
    ):
        failures.append("required_elevation_population_mismatch")

    role_elevated = [
        str(item)
        for item in _safe_list(role_aware.get("elevated_tool_ids"))
    ]
    promotion_ids = [
        str(item)
        for item in _safe_list(
            promotion.get("production_certified_tool_ids")
        )
    ]
    derived_production_ids = sorted(
        tool_id
        for tool_id, tool in tools.items()
        if tool.get("production_certified") is True
    )
    if (
        promotion_ids != derived_production_ids
        or len(promotion_ids) != len(set(promotion_ids))
    ):
        failures.append("promotion_population_not_derived_from_tools")

    derived_lane_elevated = sorted(
        {
            str(tool_id)
            for result in semantic_results.values()
            for tool_id in _safe_list(result.get("elevated_tool_ids"))
        }
    )

    elevation_rows = [
        _safe_dict(item)
        for item in _safe_list(role_aware.get("elevations"))
        if isinstance(item, Mapping)
    ]
    elevation_row_ids = [
        str(item.get("tool_id") or "") for item in elevation_rows
    ]
    if (
        len(elevation_row_ids) != len(set(elevation_row_ids))
        or any(not item for item in elevation_row_ids)
    ):
        failures.append("elevation_decision_population_invalid")
    elevation_by_tool = {
        str(item.get("tool_id") or ""): item
        for item in elevation_rows
        if str(item.get("tool_id") or "")
    }

    lane_audits = _safe_dict(semantic_audit.get("lanes"))
    canonical_role_tools = _safe_dict(canonical_roles.get("tools"))
    expected_semantic_production_ids: list[str] = []
    for tool_id, (spec, lane_id) in specs_by_tool.items():
        lane_audit = _safe_dict(lane_audits.get(lane_id))
        elevation_policy = _safe_dict(
            lane_audit.get("elevation_policy")
        )
        if (
            elevation_policy.get("valid") is not True
            or tool_id
            not in {
                str(value)
                for value in _safe_list(
                    elevation_policy.get(
                        "production_allowed_tool_ids"
                    )
                )
            }
        ):
            continue
        lane = _safe_dict(semantic_results.get(lane_id))
        compact_tool = _safe_dict(
            _safe_dict(lane.get("per_tool")).get(tool_id)
        )
        tool_audit = _safe_dict(
            _safe_dict(
                _safe_dict(lane_audits.get(lane_id)).get("tools")
            ).get(tool_id)
        )
        role = _safe_dict(canonical_role_tools.get(tool_id))
        independently_elevatable = bool(
            lane_audit.get("valid")
            and lane.get("status") == "ran"
            and compact_tool.get("certified") is True
            and tool_audit.get("checks_match_canonical_receipt") is True
            and tool_audit.get("artifact_validation_valid") is True
            and _safe_list(
                tool_audit.get("expected_production_bindings")
            )
            and role.get("can_satisfy_certified_authority") is True
        )
        if independently_elevatable:
            expected_semantic_production_ids.append(tool_id)
    expected_semantic_production_ids = sorted(
        expected_semantic_production_ids
    )
    expected_global_production_ids = sorted(
        set(expected_direct_production_ids)
        | set(expected_semantic_production_ids)
    )

    if (
        promotion_ids != expected_global_production_ids
        or derived_production_ids != expected_global_production_ids
    ):
        failures.append(
            "global_production_authority_not_independently_derived"
        )
    if (
        role_elevated != expected_semantic_production_ids
        or derived_lane_elevated != expected_semantic_production_ids
        or len(role_elevated) != len(set(role_elevated))
    ):
        failures.append(
            "role_elevation_population_not_independently_derived"
        )

    expected_decision_ids = [
        str(tool_id)
        for raw_spec in certifier.SEMANTIC_CERTIFIER_SPECS
        for tool_id in _safe_list(raw_spec.get("tool_ids"))
        if _safe_dict(
            semantic_results.get(str(raw_spec.get("lane_id") or ""))
        ).get("status")
        == "ran"
    ]
    if elevation_row_ids != expected_decision_ids:
        failures.append("elevation_decision_population_mismatch")
    expected_semantic_set = set(expected_semantic_production_ids)
    for tool_id, decision in elevation_by_tool.items():
        spec, lane_id = specs_by_tool.get(tool_id, ({}, ""))
        if (
            decision.get("lane_id") != lane_id
            or (decision.get("elevated") is True)
            != (tool_id in expected_semantic_set)
        ):
            failures.append(
                f"elevation_decision:{tool_id}:outcome_not_derived"
            )

    present: list[str] = []
    per_tool: dict[str, dict[str, Any]] = {}
    for tool_id in expected_required:
        spec, lane_id = specs_by_tool.get(tool_id, ({}, ""))
        lane = _safe_dict(semantic_results.get(lane_id))
        compact_tool = _safe_dict(
            _safe_dict(lane.get("per_tool")).get(tool_id)
        )
        tool_audit = _safe_dict(
            _safe_dict(
                _safe_dict(lane_audits.get(lane_id)).get("tools")
            ).get(tool_id)
        )
        decision = _safe_dict(elevation_by_tool.get(tool_id))
        surface_values = {
            "role_aware": tool_id in set(role_elevated),
            "promotion": tool_id in set(promotion_ids),
            "tool": _safe_dict(tools.get(tool_id)).get(
                "production_certified"
            )
            is True,
            "lane": tool_id
            in {
                str(item)
                for item in _safe_list(
                    lane.get("elevated_tool_ids")
                )
            },
            "decision": decision.get("elevated") is True,
        }
        surfaces_consistent = len(set(surface_values.values())) == 1
        evidence_valid = bool(
            lane.get("status") == "ran"
            and compact_tool.get("certified") is True
            and tool_audit.get(
                "checks_match_canonical_receipt"
            )
            is True
            and tool_audit.get("artifact_validation_valid") is True
            and decision.get("lane_id") == lane_id
            and decision.get("interface") == spec.get("interface")
            and decision.get(
                "semantic_receipt_digest_sha256"
            )
            == lane.get("digest_sha256")
            and decision.get("checks_digest_sha256")
            == compact_tool.get("check_set_digest_sha256")
            and int(decision.get("checks_count") or 0)
            == int(compact_tool.get("checks_total") or 0)
        )
        elevation_policy = _safe_dict(
            _safe_dict(lane_audits.get(lane_id)).get(
                "elevation_policy"
            )
        )
        can_elevate = bool(
            elevation_policy.get("valid") is True
            and tool_id
            in {
                str(value)
                for value in _safe_list(
                    elevation_policy.get(
                        "production_allowed_tool_ids"
                    )
                )
            }
        )
        actually_present = bool(
            surfaces_consistent
            and all(surface_values.values())
            and evidence_valid
            and can_elevate
        )
        if actually_present:
            present.append(tool_id)
        if not surfaces_consistent:
            failures.append(
                f"required_elevation:{tool_id}:surface_mismatch"
            )
        if any(surface_values.values()) and not (
            evidence_valid and can_elevate
        ):
            failures.append(
                f"required_elevation:{tool_id}:unsupported_promotion"
            )
        per_tool[tool_id] = {
            "lane_id": lane_id,
            "surfaces": surface_values,
            "surfaces_consistent": surfaces_consistent,
            "semantic_evidence_valid": evidence_valid,
            "production_elevation_allowed": can_elevate,
            "present": actually_present,
        }

    missing = [
        tool_id
        for tool_id in expected_required
        if tool_id not in set(present)
    ]
    return {
        "valid": not failures,
        "direct_production_candidate_tool_ids": direct_candidate_ids,
        "independently_certified_direct_tool_ids": (
            expected_direct_production_ids
        ),
        "independently_elevated_semantic_tool_ids": (
            expected_semantic_production_ids
        ),
        "expected_global_production_certified_tool_ids": (
            expected_global_production_ids
        ),
        "required": expected_required,
        "present": present,
        "missing": missing,
        "tools": per_tool,
        "failures": sorted(set(failures)),
    }


def _compact_specialized_deployment_binding(
    specialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind FVT-053 to both compact and source digests without claiming audit."""

    handlers = _safe_dict(specialized.get("specialized_by_handler"))
    return {
        "schema_version": specialized.get("schema_version"),
        "interface": specialized.get("interface"),
        "goal_id": specialized.get("goal_id"),
        "task_id": specialized.get("task_id"),
        "enabled": specialized.get("enabled"),
        "lossless": specialized.get("lossless"),
        "projection_aggregation_digest_sha256": specialized.get(
            "aggregation_digest_sha256"
        ),
        "source_aggregation_digest_sha256": specialized.get(
            "source_aggregation_digest_sha256"
        ),
        "projection_handler_digests": {
            str(handler_key): _safe_dict(handler).get(
                "tool_evidence_digest_sha256"
            )
            for handler_key, handler in sorted(handlers.items())
        },
        "source_handler_digests": {
            str(handler_key): _safe_dict(handler).get(
                "source_tool_evidence_digest_sha256"
            )
            for handler_key, handler in sorted(handlers.items())
        },
        "handler_count": len(handlers),
        "source_evidence_independently_verified": False,
        "verification_scope": (
            "certificate_identity_binding_only; FVT-066 independently audits "
            "the full in-memory source aggregation when supplied"
        ),
    }


def _specialized_source_aggregation_digest(
    certifier,
    source: Mapping[str, Any],
) -> str:
    """Recompute the lossless pre-compaction aggregation digest."""

    return certifier.content_digest(
        {
            "composite_lanes": {
                str(key): value
                for key, value in sorted(
                    _safe_dict(source.get("composite_lanes")).items()
                )
            },
            "specialized_by_handler": {
                str(key): value
                for key, value in sorted(
                    _safe_dict(source.get("specialized_by_handler")).items()
                )
            },
            "certifier_families_represented": list(
                _safe_list(source.get("certifier_families_represented"))
            ),
            "missing_certifier_families": list(
                _safe_list(source.get("missing_certifier_families"))
            ),
        }
    )


def _expected_specialized_handler_bindings(certifier) -> dict[str, dict[str, str]]:
    """Return the fixed lane/tool/handler population owned by the certifier."""

    expected: dict[str, dict[str, str]] = {}
    for raw_spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        spec = _safe_dict(raw_spec)
        semantic_lane_id = str(spec.get("lane_id") or "")
        property_lane_id = str(
            spec.get("property_lane_id") or semantic_lane_id
        )
        for raw_tool_id in _safe_list(spec.get("tool_ids")):
            tool_id = str(raw_tool_id)
            handler_key = f"{property_lane_id}::{tool_id}"
            expected[handler_key] = {
                "handler_key": handler_key,
                "tool_id": tool_id,
                "semantic_lane_id": semantic_lane_id,
                "property_lane_id": property_lane_id,
                "certifier_family": str(
                    spec.get("certifier_family") or property_lane_id
                ),
            }
    return expected


def _reconstruct_specialized_source(
    *,
    certifier,
    repo_root: Path,
    semantic_results: Sequence[Mapping[str, Any]],
    authority_roles: Mapping[str, Any],
    semantic_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild full handler evidence from canonical receipts and commitments."""

    failures: list[str] = [
        f"specialized:semantic_audit:{failure}"
        for failure in _safe_list(semantic_audit.get("failures"))
        if not str(failure).endswith(":semantic_lane_not_run")
    ]
    specs = {
        str(_safe_dict(spec).get("lane_id") or ""): _safe_dict(spec)
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
    }
    reconstructed_results: list[dict[str, Any]] = []
    for raw_result in semantic_results:
        result = dict(raw_result)
        lane_id = str(result.get("lane_id") or "")
        spec = _safe_dict(specs.get(lane_id))
        if not spec:
            failures.append(
                f"specialized:{lane_id or 'unknown'}:semantic_spec_missing"
            )
            continue
        compact_per_tool = _safe_dict(result.get("per_tool"))
        full_per_tool: dict[str, dict[str, Any]] = {}
        if result.get("status") == "ran":
            elevation_policy = _safe_dict(
                _safe_dict(
                    _safe_dict(semantic_audit.get("lanes")).get(lane_id)
                ).get("elevation_policy")
            )
            if not elevation_policy:
                elevation_policy = _audited_semantic_elevation_policy(
                    certifier=certifier,
                    repo_root=repo_root,
                    spec=spec,
                    semantic_result=result,
                )
            production_allowed_tool_ids = {
                str(value)
                for value in _safe_list(
                    elevation_policy.get(
                        "production_allowed_tool_ids"
                    )
                )
            }
            receipt_integrity = _safe_dict(
                result.get("receipt_integrity")
            )
            offline_observation = _safe_dict(
                result.get("offline_observation")
            )
            for raw_tool_id in _safe_list(spec.get("tool_ids")):
                tool_id = str(raw_tool_id)
                compact_tool = _safe_dict(
                    compact_per_tool.get(tool_id)
                )
                recomputed_tool = _recompute_semantic_tool_payload(
                    certifier=certifier,
                    repo_root=repo_root,
                    semantic_result=result,
                    spec=spec,
                    tool_id=tool_id,
                    compact_tool=compact_tool,
                    receipt_integrity=receipt_integrity,
                    offline_observation=offline_observation,
                    production_elevation_allowed=(
                        tool_id in production_allowed_tool_ids
                    ),
                )
                if recomputed_tool.get("valid") is not True:
                    failures.append(
                        f"specialized:{lane_id}:{tool_id}:"
                        "semantic_tool_projection_not_reconstructable"
                    )
                    failures.extend(
                        f"specialized:{lane_id}:{tool_id}:{failure}"
                        for failure in _safe_list(
                            recomputed_tool.get("failures")
                        )
                    )
                full_per_tool[tool_id] = _safe_dict(
                    recomputed_tool.get("full_tool")
                )
        result["per_tool"] = full_per_tool
        result.pop("projection_policy", None)
        reconstructed_results.append(result)

    reconstructed_source = certifier.aggregate_specialized_receipts(
        reconstructed_results,
        authority_roles=authority_roles,
        property_lanes=certifier.PROPERTY_LANES,
    )
    return {
        "valid": not failures,
        "source": reconstructed_source,
        "semantic_lane_results": reconstructed_results,
        "failures": sorted(set(failures)),
    }


def _audit_specialized_aggregation(
    *,
    certifier,
    repo_root: Path,
    specialized: Mapping[str, Any],
    semantic_results: Sequence[Mapping[str, Any]],
    source_specialized: Mapping[str, Any] | None,
    authority_roles: Mapping[str, Any],
    semantic_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently audit compact and optional full specialized evidence."""

    expected = _expected_specialized_handler_bindings(certifier)
    expected_keys = set(expected)
    compact_handlers = _safe_dict(specialized.get("specialized_by_handler"))
    compact_keys = set(str(key) for key in compact_handlers)
    lane_by_id = {
        str(row.get("lane_id") or ""): row
        for row in semantic_results
        if str(row.get("lane_id") or "")
    }
    failures: list[str] = []

    projection_digest = str(
        specialized.get("aggregation_digest_sha256") or ""
    )
    projection_computed = certifier.content_digest(
        {
            key: value
            for key, value in specialized.items()
            if key != "aggregation_digest_sha256"
        }
    )
    projection_digest_valid = bool(
        SHA256_RE.fullmatch(projection_digest)
        and projection_digest == projection_computed
    )
    if not projection_digest_valid:
        failures.append("specialized:projection_aggregation_digest_mismatch")

    handler_population_valid = bool(
        compact_keys == expected_keys
        and len(compact_handlers) == len(expected)
    )
    if not handler_population_valid:
        failures.append("specialized:handler_population_mismatch")

    expected_composites: dict[str, dict[str, set[str]]] = {}
    for handler_key, binding in expected.items():
        composite = expected_composites.setdefault(
            binding["property_lane_id"],
            {"handler_keys": set(), "tool_ids": set()},
        )
        composite["handler_keys"].add(handler_key)
        composite["tool_ids"].add(binding["tool_id"])
    compact_composites = _safe_dict(specialized.get("composite_lanes"))
    compact_composite_keys = set(str(key) for key in compact_composites)
    observed_composite_handlers: list[str] = []
    composite_rows_valid = (
        compact_composite_keys == set(expected_composites)
        and len(compact_composites) == len(expected_composites)
    )
    for property_lane_id, expected_composite in sorted(
        expected_composites.items()
    ):
        compact_composite = _safe_dict(
            compact_composites.get(property_lane_id)
        )
        handler_keys = [
            str(value)
            for value in _safe_list(compact_composite.get("handler_keys"))
        ]
        tool_ids = {
            str(value)
            for value in _safe_list(compact_composite.get("tool_ids"))
        }
        observed_composite_handlers.extend(handler_keys)
        if not (
            str(compact_composite.get("property_lane_id") or "")
            == property_lane_id
            and set(handler_keys) == expected_composite["handler_keys"]
            and len(handler_keys) == len(set(handler_keys))
            and tool_ids == expected_composite["tool_ids"]
            and SHA256_RE.fullmatch(
                str(compact_composite.get("digest_sha256") or "")
            )
        ):
            composite_rows_valid = False
    composite_coverage_valid = bool(
        composite_rows_valid
        and len(compact_composites) == 9
        and len(observed_composite_handlers) == len(expected)
        and len(observed_composite_handlers)
        == len(set(observed_composite_handlers))
        and set(observed_composite_handlers) == expected_keys
    )
    if not composite_coverage_valid:
        failures.append("specialized:composite_handler_coverage_mismatch")

    projection_flags_valid = bool(
        specialized.get("enabled") is True
        and specialized.get("lossless") is True
    )
    if not projection_flags_valid:
        failures.append("specialized:projection_policy_flags_invalid")

    handler_verification: dict[str, dict[str, Any]] = {}
    compact_handlers_valid = True
    for handler_key, expected_binding in sorted(expected.items()):
        handler = _safe_dict(compact_handlers.get(handler_key))
        declared_projection = str(
            handler.get("tool_evidence_digest_sha256") or ""
        )
        computed_projection = certifier.content_digest(
            {
                key: value
                for key, value in handler.items()
                if key != "tool_evidence_digest_sha256"
            }
        )
        projection_valid = bool(
            handler
            and SHA256_RE.fullmatch(declared_projection)
            and declared_projection == computed_projection
        )
        mapping_valid = bool(
            handler
            and all(
                str(handler.get(field) or "") == expected_value
                for field, expected_value in expected_binding.items()
            )
        )
        lane = _safe_dict(
            lane_by_id.get(expected_binding["semantic_lane_id"])
        )
        lane_ran = lane.get("status") == "ran"
        receipt_digest = str(handler.get("receipt_digest_sha256") or "")
        receipt_binding_valid = bool(
            (
                lane_ran
                and SHA256_RE.fullmatch(receipt_digest)
                and receipt_digest == str(lane.get("digest_sha256") or "")
            )
            or (not lane_ran and not receipt_digest)
        )
        source_digest = str(
            handler.get("source_tool_evidence_digest_sha256") or ""
        )
        source_digest_well_formed = bool(
            SHA256_RE.fullmatch(source_digest)
        )
        handler_verification[handler_key] = {
            "projection_tool_evidence_digest_sha256": declared_projection
            or None,
            "projection_tool_evidence_digest_computed_sha256": (
                computed_projection
            ),
            "projection_tool_evidence_digest_valid": projection_valid,
            "source_tool_evidence_digest_sha256": source_digest or None,
            "source_tool_evidence_digest_well_formed": (
                source_digest_well_formed
            ),
            "source_tool_evidence_digest_verified": False,
            "mapping_valid": mapping_valid,
            "receipt_binding_valid": receipt_binding_valid,
        }
        if not projection_valid:
            compact_handlers_valid = False
            failures.append(
                f"specialized:{handler_key}:projection_handler_digest_mismatch"
            )
        if not source_digest_well_formed:
            compact_handlers_valid = False
            failures.append(
                f"specialized:{handler_key}:source_handler_digest_missing"
            )
        if not mapping_valid:
            compact_handlers_valid = False
            failures.append(
                f"specialized:{handler_key}:handler_mapping_mismatch"
            )
        if not receipt_binding_valid:
            compact_handlers_valid = False
            failures.append(
                f"specialized:{handler_key}:receipt_binding_mismatch"
            )

    source_supplied = isinstance(source_specialized, Mapping)
    source_digest = str(
        specialized.get("source_aggregation_digest_sha256") or ""
    )
    source_digest_well_formed = bool(SHA256_RE.fullmatch(source_digest))
    source_digest_computed: str | None = None
    source_digest_valid = False
    source_handler_population_valid = False
    source_handlers_valid = False
    compact_projection_matches_source = False
    source_composites_valid = False
    source_matches_reconstruction = False
    reconstruction = _reconstruct_specialized_source(
        certifier=certifier,
        repo_root=repo_root,
        semantic_results=semantic_results,
        authority_roles=authority_roles,
        semantic_audit=semantic_audit,
    )
    failures.extend(_safe_list(reconstruction.get("failures")))
    reconstructed_source = _safe_dict(reconstruction.get("source"))
    reconstructed_source_digest = str(
        reconstructed_source.get("aggregation_digest_sha256") or ""
    )
    if not source_digest_well_formed:
        failures.append("specialized:source_aggregation_digest_missing")
    if source_supplied:
        source = _safe_dict(source_specialized)
        source_declared = str(source.get("aggregation_digest_sha256") or "")
        source_digest_computed = _specialized_source_aggregation_digest(
            certifier,
            source,
        )
        source_digest_valid = bool(
            source_digest_well_formed
            and SHA256_RE.fullmatch(source_declared)
            and source_declared == source_digest_computed == source_digest
        )
        if not source_digest_valid:
            failures.append("specialized:source_aggregation_digest_mismatch")

        source_matches_reconstruction = bool(
            reconstruction.get("valid") is True
            and source == reconstructed_source
            and source_declared == reconstructed_source_digest
        )
        if not source_matches_reconstruction:
            failures.append(
                "specialized:source_not_independently_reconstructed"
            )

        recomputed_compact_projection = (
            certifier._compact_specialized_receipt_aggregation(
                reconstructed_source
            )
        )
        compact_projection_matches_source = bool(
            source_digest_valid
            and source_matches_reconstruction
            and specialized == recomputed_compact_projection
        )
        if not compact_projection_matches_source:
            failures.append("specialized:compact_projection_source_mismatch")

        source_composites = _safe_dict(source.get("composite_lanes"))
        source_composites_valid = bool(
            set(str(key) for key in source_composites)
            == set(expected_composites)
            and all(
                _safe_dict(compact_composites.get(property_lane_id))
                == {
                    "property_lane_id": (
                        _safe_dict(source_composite).get(
                            "property_lane_id"
                        )
                        or property_lane_id
                    ),
                    "tool_ids": list(
                        _safe_list(
                            _safe_dict(source_composite).get("tool_ids")
                        )
                    ),
                    "handler_keys": list(
                        _safe_list(
                            _safe_dict(source_composite).get(
                                "specialized_handler_keys"
                            )
                        )
                    ),
                    "digest_sha256": certifier.content_digest(
                        _safe_dict(source_composite)
                    ),
                }
                for property_lane_id, source_composite in (
                    source_composites.items()
                )
            )
        )
        if not source_composites_valid:
            failures.append("specialized:source_composite_binding_mismatch")

        source_handlers = _safe_dict(source.get("specialized_by_handler"))
        source_handler_population_valid = bool(
            set(str(key) for key in source_handlers) == expected_keys
            and len(source_handlers) == len(expected)
        )
        if not source_handler_population_valid:
            failures.append("specialized:source_handler_population_mismatch")

        source_handlers_valid = source_handler_population_valid
        for handler_key, expected_binding in sorted(expected.items()):
            source_handler = _safe_dict(source_handlers.get(handler_key))
            declared_source_handler = str(
                source_handler.get("tool_evidence_digest_sha256") or ""
            )
            computed_source_handler = certifier.content_digest(
                {
                    key: value
                    for key, value in source_handler.items()
                    if key != "tool_evidence_digest_sha256"
                }
            )
            source_handler_valid = bool(
                source_handler
                and SHA256_RE.fullmatch(declared_source_handler)
                and declared_source_handler == computed_source_handler
                and declared_source_handler
                == str(
                    _safe_dict(compact_handlers.get(handler_key)).get(
                        "source_tool_evidence_digest_sha256"
                    )
                    or ""
                )
                and all(
                    str(source_handler.get(field) or "") == expected_value
                    for field, expected_value in expected_binding.items()
                )
            )
            compact_handler_matches_source = bool(
                _safe_dict(compact_handlers.get(handler_key))
                == _safe_dict(
                    recomputed_compact_projection.get(
                        "specialized_by_handler"
                    )
                ).get(handler_key)
            )
            source_handler_valid = bool(
                source_handler_valid and compact_handler_matches_source
            )
            if handler_key in handler_verification:
                handler_verification[handler_key][
                    "source_tool_evidence_digest_computed_sha256"
                ] = computed_source_handler
                handler_verification[handler_key][
                    "source_tool_evidence_digest_verified"
                ] = source_handler_valid
                handler_verification[handler_key][
                    "compact_handler_matches_source"
                ] = compact_handler_matches_source
            if not source_handler_valid:
                source_handlers_valid = False
                failures.append(
                    f"specialized:{handler_key}:source_handler_digest_mismatch"
                )
    else:
        failures.append("specialized:source_evidence_not_supplied")

    projection_valid = bool(
        projection_digest_valid
        and handler_population_valid
        and compact_handlers_valid
        and composite_coverage_valid
        and projection_flags_valid
        and specialized.get("interface")
        == "FormalVerificationSpecializedReceiptAggregation@1"
    )
    source_valid = bool(
        source_supplied
        and source_digest_valid
        and source_matches_reconstruction
        and reconstruction.get("valid") is True
        and source_handler_population_valid
        and source_handlers_valid
        and compact_projection_matches_source
        and source_composites_valid
    )
    return {
        "projection_valid": projection_valid,
        "source_valid": source_valid,
        "independent_full_evidence_valid": source_valid,
        "source_evidence_supplied": source_supplied,
        "expected_handler_count": len(expected),
        "handler_count": len(compact_handlers),
        "handler_population_valid": handler_population_valid,
        "composite_count": len(compact_composites),
        "composite_handler_coverage_valid": composite_coverage_valid,
        "projection_policy_flags_valid": projection_flags_valid,
        "source_handler_population_valid": source_handler_population_valid,
        "source_composites_valid": source_composites_valid,
        "source_matches_independent_reconstruction": (
            source_matches_reconstruction
        ),
        "compact_projection_matches_source": (
            compact_projection_matches_source
        ),
        "projection_aggregation_digest_sha256": projection_digest or None,
        "projection_aggregation_digest_computed_sha256": projection_computed,
        "projection_aggregation_digest_valid": projection_digest_valid,
        "source_aggregation_digest_sha256": source_digest or None,
        "source_aggregation_digest_computed_sha256": source_digest_computed,
        "source_aggregation_digest_reconstructed_sha256": (
            reconstructed_source_digest or None
        ),
        "source_aggregation_digest_well_formed": source_digest_well_formed,
        "source_aggregation_digest_verified": source_digest_valid,
        "handlers": handler_verification,
        "failures": sorted(set(failures)),
    }


def _compact_specialized_aggregation(
    specialized: Mapping[str, Any],
    *,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry the exact compact projection plus its independent audit."""

    return {
        "projection": {
            str(key): value for key, value in specialized.items()
        },
        "verification": {
            str(key): value for key, value in verification.items()
        },
    }


def _compact_managed_readiness(managed: Mapping[str, Any]) -> dict[str, Any]:
    """Retain platform exceptions and blockers without bulk platform rows."""

    return {
        "host_platform": managed.get("host_platform"),
        "global_supported_platforms": managed.get("global_supported_platforms"),
        "host_globally_supported": managed.get("host_globally_supported"),
        "supported_managed_capability_tool_ids": list(
            _safe_list(managed.get("supported_managed_capability_tool_ids"))
        ),
        "supported_managed_dependency_tool_ids": list(
            _safe_list(managed.get("supported_managed_dependency_tool_ids"))
        ),
        "platform_exceptions": [
            dict(item)
            for item in _safe_list(managed.get("platform_exceptions"))
            if isinstance(item, Mapping)
        ],
        "capability_blockers": list(_safe_list(managed.get("capability_blockers"))),
        "dependency_blockers": list(_safe_list(managed.get("dependency_blockers"))),
        "all_blockers": [
            {
                "tool_id": item.get("tool_id"),
                "reasons": list(_safe_list(item.get("reasons"))),
            }
            for item in _safe_list(managed.get("all_blockers"))
            if isinstance(item, Mapping)
        ],
        "ready": bool(managed.get("ready")),
        "status": managed.get("status"),
        # Compact platform_rows to identity + support flags only.
        "platform_rows": [
            {
                "tool_id": row.get("tool_id"),
                "managed": bool(row.get("managed")),
                "supported": bool(row.get("supported")),
                "installed": bool(row.get("installed")),
                "ready": bool(row.get("ready")),
            }
            for row in _safe_list(managed.get("platform_rows"))
            if isinstance(row, Mapping)
        ],
    }


def build_role_aware_release_candidate(
    *,
    repo_root: Path,
    observed_at: str | None = None,
    role_aware_certificate: Mapping[str, Any] | None = None,
    source_specialized_receipt_aggregation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build fail-closed RoleAwareFormalVerificationReleaseCandidate@1.

    Fans in the complete supported matrix into a pre-merge release candidate.
    Success booleans, blockers, platform exceptions, offline policy, quarantine
    state, and public-surface bindings are derived from bound evidence. Bulk
    certificate bodies are bound by digest only so the checked-in artifact
    stays under proposal size budgets. The candidate never claims merge or
    deployment and cannot exceed ``release_candidate`` before a merge event.
    """

    repo_root = repo_root.resolve()
    timestamp = observed_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    certifier = _load_certifier_module(repo_root)
    full_evidence: dict[str, Any] = {}
    certificate = (
        dict(role_aware_certificate)
        if role_aware_certificate is not None
        else certifier.build_certificate(
            repo_root=repo_root,
            role_aware=True,
            full_evidence_out=full_evidence,
        )
    )
    source_specialized = (
        dict(source_specialized_receipt_aggregation)
        if isinstance(source_specialized_receipt_aggregation, Mapping)
        else _safe_dict(
            full_evidence.get("specialized_receipt_aggregation")
        )
        or None
    )

    promotion = _safe_dict(certificate.get("promotion"))
    role_aware = _safe_dict(certificate.get("role_aware"))
    authority_roles = _safe_dict(certificate.get("authority_roles"))
    managed = _safe_dict(certificate.get("managed_deployment_readiness"))
    certification_policy = _safe_dict(certificate.get("certification_policy"))
    public_certificate_policy = _safe_dict(
        certificate.get("public_evidence_policy")
    )
    # The certificate policy is evidence, not authority. Re-audit the complete
    # supplied certificate so a digest-valid document cannot conceal private
    # evidence behind a forged ``satisfied: true`` declaration.
    recomputed_public_certificate_policy = certifier.public_evidence_audit(
        certificate,
        repo_root=repo_root,
    )
    tools = {
        str(tool.get("tool_id") or ""): tool
        for tool in _safe_list(certificate.get("tools"))
        if isinstance(tool, Mapping) and str(tool.get("tool_id") or "")
    }

    certificate_digest_valid = bool(
        certificate.get("certificate_digest_sha256")
        and certificate.get("certificate_digest_sha256")
        == certifier.content_digest(
            {
                key: value
                for key, value in certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )

    tool_check_digests: dict[str, str] = {}
    tool_artifact_digests: dict[str, list[str]] = {}
    for tool_id, tool in tools.items():
        checks = _safe_list(tool.get("checks"))
        tool_check_digests[tool_id] = certifier.content_digest(checks)
        artifact_ids: list[str] = []
        for artifact in _safe_list(tool.get("artifact_identities")):
            if isinstance(artifact, Mapping):
                digest = str(
                    artifact.get("sha256")
                    or artifact.get("digest_sha256")
                    or artifact.get("content_digest")
                    or ""
                )
                if digest:
                    artifact_ids.append(digest)
            elif isinstance(artifact, str) and artifact:
                artifact_ids.append(artifact)
        executable = str(tool.get("executable_sha256") or "")
        if executable:
            artifact_ids.append(executable)
        tool_artifact_digests[tool_id] = sorted(set(artifact_ids))

    semantic_results = [
        result
        for result in _safe_list(certificate.get("semantic_lane_results"))
        if isinstance(result, Mapping)
    ]
    semantic_audit = (
        _audit_semantic_lane_results(
            certifier=certifier,
            repo_root=repo_root,
            semantic_results=semantic_results,
        )
        if role_aware.get("enabled")
        else {"valid": True, "failures": []}
    )
    semantic_receipts_full_and_bound = bool(semantic_audit.get("valid"))
    semantic_binding_failures: list[str] = list(
        _safe_list(semantic_audit.get("failures"))
    )
    semantic_receipt_digests: dict[str, str] = {}
    semantic_per_tool_evidence_digests: dict[str, dict[str, str]] = {}
    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "unknown")
        per_tool_rows = _safe_dict(result.get("per_tool"))
        semantic_per_tool_evidence_digests[lane_id] = {
            str(tool_id): certifier.content_digest(_safe_dict(per_tool))
            for tool_id, per_tool in per_tool_rows.items()
        }
        if result.get("status") == "ran":
            semantic_receipt_digests[lane_id] = str(
                result.get("digest_sha256") or ""
            )

    specialized = _safe_dict(certificate.get("specialized_receipt_aggregation"))
    specialized_verification = _audit_specialized_aggregation(
        certifier=certifier,
        repo_root=repo_root,
        specialized=specialized,
        semantic_results=semantic_results,
        source_specialized=source_specialized,
        authority_roles=authority_roles,
        semantic_audit=semantic_audit,
    )
    specialized_binding_failures = list(
        _safe_list(specialized_verification.get("failures"))
    )
    specialized_handlers = _safe_dict(
        specialized.get("specialized_by_handler")
    )
    projection_handler_digests = {
        str(handler_key): _safe_dict(handler).get(
            "tool_evidence_digest_sha256"
        )
        for handler_key, handler in sorted(specialized_handlers.items())
    }
    source_handler_digests = {
        str(handler_key): _safe_dict(handler).get(
            "source_tool_evidence_digest_sha256"
        )
        for handler_key, handler in sorted(specialized_handlers.items())
    }

    digest_material = {
        "certificate_digest_sha256": certificate.get("certificate_digest_sha256"),
        "tool_check_digests": tool_check_digests,
        "tool_artifact_digests": tool_artifact_digests,
        "semantic_receipt_digests": semantic_receipt_digests,
        "specialized_projection_aggregation_digest": (
            specialized.get("aggregation_digest_sha256")
        ),
        "specialized_source_aggregation_digest": (
            specialized.get("source_aggregation_digest_sha256")
        ),
        "specialized_projection_handler_digests": (
            projection_handler_digests
        ),
        "specialized_source_handler_digests": source_handler_digests,
        "authority_roles_policy_digest": authority_roles.get(
            "policy_digest_sha256"
        ),
        "lock_digest": _safe_dict(certificate.get("lock")).get("digest_sha256"),
        "quarantine_digest": certifier.content_digest(
            _safe_list(certificate.get("disagreement_quarantines"))
        ),
    }

    elevated = sorted(set(role_aware.get("elevated_tool_ids") or []))
    elevation_audit = _audit_required_elevations(
        certifier=certifier,
        repo_root=repo_root,
        certificate=certificate,
        semantic_audit=semantic_audit,
    )
    missing_required = list(
        _safe_list(elevation_audit.get("missing"))
    )
    production_elevation_fanin = (
        build_production_semantic_elevation_fanin(
            repo_root=repo_root,
            observed_at=timestamp,
            role_aware_certificate=certificate,
            certifier_module=certifier,
        )
    )
    production_elevation_fanin_binding = (
        compact_production_elevation_fanin_binding(
            production_elevation_fanin
        )
    )
    checked_production_elevation_fanin = (
        verify_checked_production_elevation_fanin(
            repo_root=repo_root,
            live_fanin=production_elevation_fanin,
        )
    )
    fanin_summary = _safe_dict(
        production_elevation_fanin.get("summary")
    )
    fanin_acceptance = _safe_dict(
        production_elevation_fanin.get("acceptance")
    )
    production_elevation_fanin_bound = bool(
        fanin_summary.get("structurally_valid") is True
        and fanin_summary.get("all_required_reconstructions_valid") is True
        and fanin_acceptance.get(
            "role_aware_certificate_identity_bound"
        )
        is True
        and fanin_acceptance.get("checks_never_collapsed") is True
        and fanin_acceptance.get("offline_only") is True
        and checked_production_elevation_fanin.get("matches_live") is True
    )
    production_elevation_fanin_closed = bool(
        production_elevation_fanin_bound
        and fanin_summary.get("fanin_closed") is True
    )
    digest_material.update(
        {
            "production_elevation_fanin_receipt_digest": (
                production_elevation_fanin.get(
                    "receipt_digest_sha256"
                )
            ),
            "checked_production_elevation_fanin_content_identity": (
                checked_production_elevation_fanin.get(
                    "content_identity"
                )
            ),
        }
    )

    source = build_release_candidate_source_attestation(repo_root)

    platform_exceptions = [
        dict(item)
        for item in _safe_list(managed.get("platform_exceptions"))
        if isinstance(item, Mapping)
    ]
    platform_audit = _audit_platform_support(
        certifier=certifier,
        repo_root=repo_root,
        managed=managed,
        certificate_lock=_safe_dict(certificate.get("lock")),
        tools=tools,
        authority_roles=authority_roles,
        semantic_results=semantic_results,
    )
    platform_exceptions_valid = bool(platform_audit.get("valid"))
    exceptions_disjoint_from_supported = bool(
        platform_audit.get("platform_exceptions_valid")
        and platform_audit.get("supported_lists_valid")
    )

    non_authoritative_promotions: list[str] = []
    for tool_id, tool in tools.items():
        evidence_class = str(tool.get("evidence_class") or "")
        artifact_class = str(tool.get("executable_artifact_class") or "")
        if tool.get("production_certified") and (
            evidence_class in NON_AUTHORITATIVE_EVIDENCE_CLASSES
            or artifact_class == "generated_hermetic_shim"
            or evidence_class
            in {
                "identity_plus_fixture_parser",
                "hermetic_adapter_shim",
                "hermetic_shadow_shim",
                "proposal_only_semantics",
            }
        ):
            non_authoritative_promotions.append(tool_id)
    synthetic_evidence_cannot_certify = not non_authoritative_promotions

    role_tools = _safe_dict(authority_roles.get("tools"))
    authority_ceiling_respected = all(
        not (
            tools.get(tool_id, {}).get("production_certified")
            and not _safe_dict(meta).get("can_satisfy_certified_authority")
        )
        for tool_id, meta in role_tools.items()
    )

    offline_policy_satisfied = bool(
        certification_policy.get("offline_policy_satisfied")
    )
    public_evidence_safe = bool(
        public_certificate_policy.get("satisfied")
        and recomputed_public_certificate_policy.get("satisfied")
    )
    quarantines = _safe_list(certificate.get("disagreement_quarantines"))
    quarantines_bound = certificate_digest_valid and isinstance(
        certificate.get("disagreement_quarantines"), list
    )

    host_support = {
        "host_platform": managed.get("host_platform"),
        "global_supported_platforms": managed.get("global_supported_platforms"),
        "host_globally_supported": managed.get("host_globally_supported"),
        "derived_from": "managed_deployment_readiness",
    }

    roles_summary = {
        "present": bool(authority_roles.get("present")),
        "interface": authority_roles.get("interface"),
        "role_interface": authority_roles.get("role_interface"),
        "policy_digest_sha256": authority_roles.get("policy_digest_sha256"),
        "boundary": authority_roles.get("boundary"),
        "tool_count": len(role_tools),
    }
    authority_roles_valid = bool(
        authority_roles.get("present") is True
        and authority_roles
        == certifier.load_authority_roles(repo_root)
    )

    ceilings = {
        "max_stage": RELEASE_CANDIDATE_MAX_STAGE,
        "merge_event_present": False,
        "deployment_claimed": False,
        "post_merge_attestation_claimed": False,
        "cannot_exceed_release_candidate_without_merge_event": True,
        "authority_ceiling_respected": authority_ceiling_respected,
        "non_authoritative_promotions": sorted(non_authoritative_promotions),
    }

    evidence_classes = sorted(
        {str(tool.get("evidence_class") or "unknown") for tool in tools.values()}
    )

    public_surfaces = {
        "certificate_public_evidence_policy": {
            "declared": {
                "satisfied": public_certificate_policy.get("satisfied"),
                "host_private_paths_forbidden": public_certificate_policy.get(
                    "host_private_paths_forbidden"
                ),
                "raw_process_output_forbidden": public_certificate_policy.get(
                    "raw_process_output_forbidden"
                ),
                "raw_secret_or_witness_forbidden": (
                    public_certificate_policy.get(
                        "raw_secret_or_witness_forbidden"
                    )
                ),
            },
            "recomputed": recomputed_public_certificate_policy,
        },
        "host_private_paths_forbidden": bool(
            public_certificate_policy.get("host_private_paths_forbidden")
        ),
        "raw_process_output_forbidden": bool(
            public_certificate_policy.get("raw_process_output_forbidden")
        ),
        "raw_secret_or_witness_forbidden": bool(
            public_certificate_policy.get("raw_secret_or_witness_forbidden")
        ),
        "bound": public_evidence_safe,
    }

    artifacts = {
        "toolchain_certificate": {
            "path": TOOLCHAIN_CERT_RELATIVE.as_posix(),
            "present": (repo_root / TOOLCHAIN_CERT_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / TOOLCHAIN_CERT_RELATIVE),
            "role_aware_digest": certificate.get("certificate_digest_sha256"),
        },
        "release_candidate": {
            "path": DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix(),
            "previous_candidate_content_not_read": True,
            "generated_after_certified_source": True,
            "publication_identity": "self:candidate_identity",
        },
        "release_candidate_test": {
            "path": DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE
            ),
        },
        "production_elevation_fanin_receipt": {
            "path": (
                DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix()
            ),
            "present": checked_production_elevation_fanin.get("present"),
            "content_identity": checked_production_elevation_fanin.get(
                "content_identity"
            ),
            "stored_digest_valid": (
                checked_production_elevation_fanin.get(
                    "stored_digest_valid"
                )
            ),
            "matches_live": checked_production_elevation_fanin.get(
                "matches_live"
            ),
            "live_digest_sha256": production_elevation_fanin.get(
                "receipt_digest_sha256"
            ),
        },
        "production_elevation_fanin_test": {
            "path": (
                DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE.as_posix()
            ),
            "present": (
                repo_root / DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE
            ).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE
            ),
        },
        "certifier": {
            "path": DEFAULT_CERTIFIER_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_CERTIFIER_RELATIVE).is_file(),
            "content_identity": sha256_file(
                repo_root / DEFAULT_CERTIFIER_RELATIVE
            ),
        },
        "receipt_builder": {
            "path": DEFAULT_BUILDER_RELATIVE.as_posix(),
            "present": (repo_root / DEFAULT_BUILDER_RELATIVE).is_file(),
            "content_identity": sha256_file(repo_root / DEFAULT_BUILDER_RELATIVE),
        },
    }
    artifacts_present = all(
        bool(item.get("present"))
        for key, item in artifacts.items()
        if key != "release_candidate"
    )

    specialized_bound = bool(
        (
            specialized.get("interface")
            == "FormalVerificationSpecializedReceiptAggregation@1"
            and specialized_verification.get("projection_valid") is True
            and specialized_verification.get("source_valid") is True
            and not specialized_binding_failures
        )
        or not bool(role_aware.get("enabled"))
    )

    acceptance = {
        "role_aware_certificate_bound": certificate_digest_valid,
        "certificate_digest_participates_in_identity": certificate_digest_valid,
        "raw_receipt_check_case_binding_digests_bound": bool(tool_check_digests)
        and certificate_digest_valid,
        "executable_artifact_dependency_digests_bound": certificate_digest_valid,
        "specialized_receipt_aggregation_bound": specialized_bound
        or not bool(role_aware.get("enabled")),
        "specialized_compact_projection_bound": bool(
            specialized_verification.get("projection_valid")
        )
        or not bool(role_aware.get("enabled")),
        "specialized_source_evidence_independently_verified": bool(
            specialized_verification.get("source_valid")
        )
        or not bool(role_aware.get("enabled")),
        "certified_source_bound": bool(source.get("source_commit_bound")),
        "source_valid_for_release_candidate": bool(
            source.get("valid_for_release_candidate")
        ),
        "self_referential_current_tree_claim_absent": bool(
            source.get("self_referential_current_tree_claim_forbidden")
        )
        and bool(
            source.get(
                "generated_candidate_identity_excluded_from_source_identity"
            )
        )
        and bool(
            source.get(
                "source_binding_uses_committed_tree_not_candidate_identity"
            )
        ),
        "host_support_derived": bool(host_support.get("host_platform")),
        "roles_bound": bool(
            authority_roles_valid
        ),
        "ceilings_derived": True,
        "evidence_classes_derived": bool(evidence_classes),
        "platform_exceptions_derived_and_narrow": platform_exceptions_valid,
        "unsupported_only_as_narrow_exceptions": (
            platform_exceptions_valid and exceptions_disjoint_from_supported
        ),
        "blockers_derived_not_concealed": True,
        "offline_policy_satisfied": offline_policy_satisfied,
        "quarantine_state_bound": quarantines_bound,
        "public_surfaces_bound": public_evidence_safe,
        "authority_ceiling_respected": authority_ceiling_respected,
        "synthetic_evidence_cannot_certify_production": (
            synthetic_evidence_cannot_certify
        ),
        "no_install_during_offline_certification": offline_policy_satisfied,
        "semantic_receipts_full_and_bound": semantic_receipts_full_and_bound,
        "required_semantic_elevations_present": bool(
            elevation_audit.get("valid") and not missing_required
        ),
        "supported_managed_capabilities_ready": bool(managed.get("ready")),
        "supported_managed_capability_blockers": _safe_list(
            managed.get("capability_blockers")
        ),
        "supported_managed_dependency_blockers": _safe_list(
            managed.get("dependency_blockers")
        ),
        "merge_not_claimed": True,
        "deployment_not_claimed": True,
        "stage_at_most_release_candidate": True,
        "artifacts_present": artifacts_present,
        "production_semantic_elevation_fanin_bound": (
            production_elevation_fanin_bound
        ),
        "production_semantic_elevation_fanin_closed": (
            production_elevation_fanin_closed
        ),
        "production_elevation_requires_independent_pnmr": bool(
            fanin_acceptance.get(
                "no_elevation_without_reconstruction"
            )
            is True
        ),
        "checked_production_elevation_fanin_matches_live": bool(
            checked_production_elevation_fanin.get("matches_live")
            is True
        ),
    }

    readiness_requirements = {
        key: bool(acceptance[key])
        for key in (
            "role_aware_certificate_bound",
            "raw_receipt_check_case_binding_digests_bound",
            "executable_artifact_dependency_digests_bound",
            "specialized_receipt_aggregation_bound",
            "specialized_compact_projection_bound",
            "specialized_source_evidence_independently_verified",
            "certified_source_bound",
            "source_valid_for_release_candidate",
            "self_referential_current_tree_claim_absent",
            "host_support_derived",
            "roles_bound",
            "platform_exceptions_derived_and_narrow",
            "unsupported_only_as_narrow_exceptions",
            "offline_policy_satisfied",
            "quarantine_state_bound",
            "public_surfaces_bound",
            "authority_ceiling_respected",
            "synthetic_evidence_cannot_certify_production",
            "no_install_during_offline_certification",
            "semantic_receipts_full_and_bound",
            "required_semantic_elevations_present",
            "supported_managed_capabilities_ready",
            "production_semantic_elevation_fanin_bound",
            "production_semantic_elevation_fanin_closed",
            "production_elevation_requires_independent_pnmr",
            "checked_production_elevation_fanin_matches_live",
            "merge_not_claimed",
            "deployment_not_claimed",
            "stage_at_most_release_candidate",
            "artifacts_present",
        )
    }

    candidate_ready = all(readiness_requirements.values())
    readiness_stage = (
        RELEASE_CANDIDATE_MAX_STAGE if candidate_ready else "blocked"
    )
    if readiness_stage not in {"blocked", RELEASE_CANDIDATE_MAX_STAGE}:
        readiness_stage = "blocked"
    status = (
        "role_aware_release_candidate_ready"
        if candidate_ready
        else "role_aware_release_candidate_blocked"
    )

    blockers = sorted(
        [
            key
            for key, satisfied in readiness_requirements.items()
            if not satisfied
        ]
        + semantic_binding_failures
        + specialized_binding_failures
        + _safe_list(platform_audit.get("failures"))
        + _safe_list(elevation_audit.get("failures"))
        + [
            f"production_elevation_fanin:{failure}"
            for failure in _safe_list(fanin_summary.get("failures"))
        ]
        + [
            f"managed:{item.get('tool_id')}:{reason}"
            for item in _safe_list(managed.get("all_blockers"))
            if isinstance(item, Mapping)
            for reason in _safe_list(item.get("reasons"))
        ]
        + [
            f"non_authoritative_promotion:{tool_id}"
            for tool_id in non_authoritative_promotions
        ]
        + [
            f"required_elevation_missing:{tool_id}"
            for tool_id in missing_required
        ]
    )

    # Compact blockers in acceptance: keep counts, not full nested dumps.
    acceptance["supported_managed_capability_blockers"] = list(
        _safe_list(managed.get("capability_blockers"))
    )
    acceptance["supported_managed_dependency_blockers"] = list(
        _safe_list(managed.get("dependency_blockers"))
    )

    compact_role_aware = {
        "enabled": bool(role_aware.get("enabled")),
        "goal_id": role_aware.get("goal_id"),
        "task_id": role_aware.get("task_id"),
        "interface": role_aware.get("interface"),
        "elevated_tool_ids": elevated,
        "required_baseline_elevations": list(
            role_aware.get("required_baseline_elevations")
            or list(REQUIRED_SEMANTIC_ELEVATIONS)
        ),
        "elevation_count": len(_safe_list(role_aware.get("elevations"))),
        "demotion_count": len(_safe_list(role_aware.get("demotions"))),
        "release_candidate": _safe_dict(role_aware.get("release_candidate")),
        "production_semantic_elevation_fanin": _safe_dict(
            role_aware.get("production_semantic_elevation_fanin")
        ),
    }

    candidate: dict[str, Any] = {
        "schema_version": RELEASE_CANDIDATE_SCHEMA_VERSION,
        "interface": RELEASE_CANDIDATE_INTERFACE,
        "program_interface": PROGRAM_INTERFACE,
        "program_goal_id": PROGRAM_GOAL_ID,
        "goal_id": RELEASE_CANDIDATE_GOAL_ID,
        "task_id": RELEASE_CANDIDATE_TASK_ID,
        "program": RELEASE_CANDIDATE_PROGRAM,
        "observed_at": timestamp,
        "binding_mode": "pre_merge_role_aware_release_candidate",
        "status": status,
        "readiness_stage": readiness_stage,
        "description": (
            "Fail-closed role-aware release candidate. Fans in host support, "
            "roles, ceilings, evidence classes, platform exceptions, blockers, "
            "offline policy, quarantine state, and public surfaces from bound "
            "matrix evidence without claiming merge or deployment. Maximum "
            "stage before a merge event exists is release_candidate. Bulk "
            "certificate bodies are bound by digest only."
        ),
        "source": source,
        "host_support": host_support,
        "roles": roles_summary,
        "ceilings": ceilings,
        "evidence_classes": evidence_classes,
        "offline_policy": {
            "satisfied": offline_policy_satisfied,
            "forbid_install": bool(certification_policy.get("forbid_install")),
            "forbid_download": bool(certification_policy.get("forbid_download")),
            "forbid_network": bool(certification_policy.get("forbid_network")),
            "lock_offline_verification_policy": certification_policy.get(
                "lock_offline_verification_policy"
            ),
            "offline_observation_count": len(
                _safe_list(certification_policy.get("offline_observations"))
            ),
        },
        "quarantine_state": {
            "bound": quarantines_bound,
            "disagreement_quarantines": quarantines,
            "count": len(quarantines),
        },
        "public_surfaces": public_surfaces,
        "semantic_audit": semantic_audit,
        "platform_support_audit": platform_audit,
        "required_elevation_audit": elevation_audit,
        "production_semantic_elevation_fanin": (
            production_elevation_fanin_binding
        ),
        "checked_production_semantic_elevation_fanin": (
            checked_production_elevation_fanin
        ),
        "acceptance": acceptance,
        "readiness_requirements": readiness_requirements,
        "blockers": blockers,
        "digest_material": digest_material,
        "role_aware_certificate": {
            "projection_model": "digest_bound_compact_projection/v1",
            "raw_certificate_embedded": False,
            "interface": certificate.get("interface"),
            "schema_version": certificate.get("schema_version"),
            "goal_id": certificate.get("goal_id"),
            "task_id": certificate.get("task_id"),
            "binding_mode": certificate.get("binding_mode"),
            "certificate_digest_sha256": certificate.get(
                "certificate_digest_sha256"
            ),
            "role_aware": compact_role_aware,
            "promotion": {
                "production_certified_tool_ids": list(
                    promotion.get("production_certified_tool_ids") or []
                ),
                "merely_usable_tool_ids": list(
                    promotion.get("merely_usable_tool_ids") or []
                ),
                "unavailable_tool_ids": list(
                    promotion.get("unavailable_tool_ids") or []
                ),
            },
            "property_lanes": certificate.get("property_lanes"),
            "disagreement_quarantines": quarantines,
            "authority_roles": {
                key: authority_roles.get(key)
                for key in (
                    "present",
                    "interface",
                    "role_interface",
                    "boundary",
                    "policy_digest_sha256",
                )
            },
            "semantic_lane_results": [
                _compact_semantic_lane(
                    result,
                    per_tool_evidence_digests=(
                        semantic_per_tool_evidence_digests.get(
                            str(result.get("lane_id") or "unknown"),
                            {},
                        )
                    ),
                )
                for result in semantic_results
            ],
            "specialized_receipt_aggregation": _compact_specialized_aggregation(
                specialized,
                verification=specialized_verification,
            ),
            "managed_deployment_readiness": _compact_managed_readiness(managed),
            "tools": [
                _compact_tool_binding(
                    tools[tool_id],
                    checks_digest=tool_check_digests[tool_id],
                    artifact_digests=tool_artifact_digests[tool_id],
                )
                for tool_id in sorted(tools)
            ],
            "certification_policy": {
                "offline_policy_satisfied": certification_policy.get(
                    "offline_policy_satisfied"
                ),
                "forbid_install": certification_policy.get("forbid_install"),
                "forbid_download": certification_policy.get("forbid_download"),
                "forbid_network": certification_policy.get("forbid_network"),
            },
            "public_evidence_policy": {
                "declared_satisfied": public_certificate_policy.get(
                    "satisfied"
                ),
                "recomputed": recomputed_public_certificate_policy,
            },
        },
        "elevations": {
            "required": list(REQUIRED_SEMANTIC_ELEVATIONS),
            "elevated_tool_ids": elevated,
            "missing_required": missing_required,
            "merely_usable_tool_ids": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
            "production_certified_tool_ids": list(
                promotion.get("production_certified_tool_ids") or []
            ),
            # Compact elevation details: tool_id + elevated flag only.
            "details": [
                {
                    "tool_id": item.get("tool_id"),
                    "elevated": bool(item.get("elevated")),
                    "evidence_class": item.get("evidence_class"),
                }
                for item in _safe_list(role_aware.get("elevations"))
                if isinstance(item, Mapping)
            ],
        },
        "platform_exceptions": platform_exceptions,
        "artifacts": artifacts,
        "claims": {
            "merge": False,
            "deployment": False,
            "post_merge_attestation": False,
            "self_referential_current_tree": False,
            "max_stage": RELEASE_CANDIDATE_MAX_STAGE,
            "merge_event_present": False,
        },
        "disclosures": {
            "unavailable_tools": list(promotion.get("unavailable_tool_ids") or []),
            "merely_usable_tools": list(
                promotion.get("merely_usable_tool_ids") or []
            ),
            "missing_required_elevations": missing_required,
            "supported_managed_capability_blockers": list(
                _safe_list(managed.get("capability_blockers"))
            ),
            "supported_managed_dependency_blockers": list(
                _safe_list(managed.get("dependency_blockers"))
            ),
            "assurance_ceilings": {
                "path_presence_is_not_usability": True,
                "source_presence_is_not_usability": True,
                "fixture_is_not_production_certified": True,
                "synthetic_evidence_cannot_certify_production": True,
                "advisor_support_shadow_cannot_certify": True,
                "unavailable_cannot_count_as_complete": True,
                "release_candidate_is_not_deployment_certificate": True,
                "release_candidate_cannot_claim_own_merge": True,
            },
            "remaining_bounds": [
                "Maximum stage before a merge event exists is release_candidate.",
                "Post-merge deployment attestation is FVT-G214 / "
                "RoleAwareFormalVerificationRelease@1 and is never claimed here.",
                "Unsupported host platforms yield narrow exceptions; missing "
                "supported installations remain blockers.",
                "Non-authoritative evidence classes cannot raise production "
                "or deployment ceilings.",
            ],
        },
        "notes": [
            "RoleAwareFormalVerificationReleaseCandidate@1 owns central "
            "candidate fan-in for FVT-G213 without installing during offline "
            "certification or concealing blockers.",
            "ProductionSemanticElevationFanIn@1 independently reconstructs "
            "positive, negative, mutation, and replay evidence and exact "
            "compact bindings for every required semantic elevation.",
            "Every raw receipt, check, case, binding, executable, artifact, "
            "and dependency digest participates in the certificate digest "
            "and therefore in this candidate identity.",
            "The checked-in candidate binds an explicit certified source "
            "commit/tree. A prior candidate path may be present in that tree; "
            "the newly generated candidate identity is never used as its own "
            "source identity.",
            "Bulk formal artifacts are bound by digest; rebuild at load time "
            "from the live certificate rather than embedding full dumps.",
        ],
    }
    candidate = certifier.public_evidence_projection(
        candidate, repo_root=repo_root
    )
    public_evidence_policy = certifier.public_evidence_audit(
        candidate,
        repo_root=repo_root,
    )
    candidate["public_evidence_policy"] = public_evidence_policy
    if not public_evidence_policy["satisfied"]:
        candidate["acceptance"]["public_surfaces_bound"] = False
        candidate["readiness_requirements"]["public_surfaces_bound"] = False
        candidate["status"] = "role_aware_release_candidate_blocked"
        candidate["readiness_stage"] = "blocked"
        blockers_list = candidate["blockers"]
        if "public_surfaces_bound" not in blockers_list:
            blockers_list.append("public_surfaces_bound")
        candidate["public_surfaces"]["bound"] = False
    candidate["candidate_identity"] = content_digest(candidate)
    return candidate


def _tool_spec_for_elevation(certifier, tool_id: str) -> dict[str, Any] | None:
    """Locate the semantic-certifier policy row that owns ``tool_id``."""

    for raw_spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        spec = _safe_dict(raw_spec)
        if tool_id in {
            str(item) for item in _safe_list(spec.get("tool_ids"))
        }:
            return spec
    return None


def _certificate_identity_for_production_fanin(
    *,
    certifier,
    certificate: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Independently verify the outer certificate used by the fan-in."""

    stored_digest = str(certificate.get("certificate_digest_sha256") or "")
    computed_digest = certifier.content_digest(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_digest_sha256"
        }
    )
    digest_valid = bool(stored_digest and stored_digest == computed_digest)
    interface_valid = bool(
        certificate.get("interface") == certifier.INTERFACE
        and certificate.get("schema_version") == certifier.SCHEMA_VERSION
        and certificate.get("goal_id") == certifier.GOAL_ID
        and certificate.get("task_id") == certifier.TASK_ID
    )
    role_aware = _safe_dict(certificate.get("role_aware"))
    role_aware_valid = bool(
        role_aware.get("enabled") is True
        and role_aware.get("interface") == certifier.ROLE_AWARE_INTERFACE
        and role_aware.get("goal_id") == certifier.ROLE_AWARE_GOAL_ID
        and role_aware.get("task_id") == certifier.ROLE_AWARE_TASK_ID
    )
    declared_public = _safe_dict(certificate.get("public_evidence_policy"))
    recomputed_public = certifier.public_evidence_audit(
        certificate,
        repo_root=repo_root,
    )
    public_evidence_valid = bool(
        declared_public.get("satisfied") is True
        and recomputed_public.get("satisfied") is True
    )
    return {
        "valid": bool(
            digest_valid
            and interface_valid
            and role_aware_valid
            and public_evidence_valid
        ),
        "digest_valid": digest_valid,
        "stored_digest_sha256": stored_digest or None,
        "computed_digest_sha256": computed_digest,
        "interface_valid": interface_valid,
        "role_aware_identity_valid": role_aware_valid,
        "public_evidence_valid": public_evidence_valid,
        "recomputed_public_evidence_policy": recomputed_public,
    }


def _independent_pnmr_reconstruction(
    *,
    certifier,
    semantic_result: Mapping[str, Any],
    tool_id: str,
    compact_tool: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct and bind positive/negative/mutation/replay evidence."""

    required = set(PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS)
    recomputed = certifier.recompute_semantic_tool_check_binding(
        semantic_result,
        tool_id,
    )
    checks = [
        _safe_dict(item)
        for item in _safe_list(recomputed.get("checks"))
        if isinstance(item, Mapping)
    ]
    kinds_present = {
        str(kind)
        for kind in _safe_list(recomputed.get("check_kinds_present"))
        if str(kind or "")
    }
    required_checks = [
        check
        for check in checks
        if str(check.get("kind") or "") in required
    ]
    passed_required_kinds = {
        str(check.get("kind") or "")
        for check in required_checks
        if str(check.get("status") or "") == "passed"
    }
    failed_required_kinds = sorted(
        {
            str(check.get("kind") or "")
            for check in required_checks
            if str(check.get("status") or "") != "passed"
        }
    )
    required_kinds_present = required <= kinds_present
    required_kinds_all_passed = bool(
        required <= passed_required_kinds and not failed_required_kinds
    )

    compact_status_counts = _safe_dict(
        compact_tool.get("check_status_counts")
    )
    recomputed_status_counts = _safe_dict(
        recomputed.get("check_status_counts")
    )
    compact_binding_valid = bool(
        str(compact_tool.get("check_set_digest_sha256") or "")
        == str(recomputed.get("check_set_digest_sha256") or "")
        and int(compact_tool.get("checks_total") or 0)
        == int(recomputed.get("checks_total") or 0)
        and int(compact_tool.get("checks_passed") or 0)
        == int(recomputed.get("checks_passed") or 0)
        and sorted(_safe_list(compact_tool.get("check_kinds_present")))
        == sorted(_safe_list(recomputed.get("check_kinds_present")))
        and compact_status_counts == recomputed_status_counts
    )
    valid = bool(
        recomputed.get("valid") is True
        and required_kinds_present
        and required_kinds_all_passed
        and compact_binding_valid
        and str(recomputed.get("check_set_digest_sha256") or "")
    )
    # Compact by construction: raw check bodies stay in the canonical lane
    # receipt and this independent surface keeps only exact commitments.
    return {
        "valid": valid,
        "recompute_valid": bool(recomputed.get("valid")),
        "recompute_failure": recomputed.get("failure"),
        "check_kinds_present": sorted(kinds_present),
        "required_check_kinds": list(
            PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS
        ),
        "required_kinds_present": required_kinds_present,
        "required_kinds_all_passed": required_kinds_all_passed,
        "required_kinds_failed": failed_required_kinds,
        "required_kinds_missing": sorted(required - kinds_present),
        "check_set_digest_sha256": recomputed.get(
            "check_set_digest_sha256"
        ),
        "compact_check_set_digest_sha256": compact_tool.get(
            "check_set_digest_sha256"
        ),
        "compact_binding_valid": compact_binding_valid,
        "checks_total": int(recomputed.get("checks_total") or 0),
        "checks_passed": int(recomputed.get("checks_passed") or 0),
        "check_status_counts": recomputed_status_counts,
        "raw_checks_embedded": False,
    }


def _production_fanin_offline_policy(
    *,
    certificate: Mapping[str, Any],
    semantic_results: Mapping[str, Mapping[str, Any]],
    required_lane_ids: Sequence[str],
) -> dict[str, Any]:
    """Derive the no-install/download/network gate from bound evidence."""

    policy = _safe_dict(certificate.get("certification_policy"))
    policy_satisfied = bool(
        policy.get("offline_policy_satisfied") is True
        and policy.get("forbid_install") is True
        and policy.get("forbid_download") is True
        and policy.get("forbid_network") is True
    )
    lanes: dict[str, dict[str, Any]] = {}
    for lane_id in sorted(set(required_lane_ids)):
        lane = _safe_dict(semantic_results.get(lane_id))
        observation = _safe_dict(lane.get("offline_observation"))
        lane_satisfied = bool(
            lane.get("status") == "ran"
            and observation.get("satisfied") is True
            and observation.get("install_attempted") is not True
            and observation.get("download_attempted") is not True
            and observation.get("network_used") is not True
        )
        lanes[lane_id] = {
            "satisfied": lane_satisfied,
            "install_attempted": bool(
                observation.get("install_attempted")
            ),
            "download_attempted": bool(
                observation.get("download_attempted")
            ),
            "network_used": bool(observation.get("network_used")),
        }
    return {
        "satisfied": bool(
            policy_satisfied
            and lanes
            and all(row["satisfied"] for row in lanes.values())
        ),
        "certificate_policy_satisfied": policy_satisfied,
        "required_lanes": lanes,
    }


def build_production_semantic_elevation_fanin(
    *,
    repo_root: Path | None = None,
    observed_at: str | None = None,
    role_aware_certificate: Mapping[str, Any] | None = None,
    certifier_module: Any | None = None,
) -> dict[str, Any]:
    """Build the fail-closed production-semantic elevation fan-in.

    Each required baseline tool is independently reconstructed from its
    canonical semantic receipt. No missing, mismatched, failed, non-authority,
    or non-production artifact binding can become elevation eligibility.
    """

    root = (repo_root or repo_root_from()).resolve()
    certifier = certifier_module or _load_certifier_module(root)
    certificate = (
        dict(role_aware_certificate)
        if isinstance(role_aware_certificate, Mapping)
        else certifier.build_certificate(repo_root=root, role_aware=True)
    )
    certificate_identity = _certificate_identity_for_production_fanin(
        certifier=certifier,
        certificate=certificate,
        repo_root=root,
    )
    semantic_results = {
        str(item.get("lane_id") or ""): _safe_dict(item)
        for item in _safe_list(certificate.get("semantic_lane_results"))
        if isinstance(item, Mapping) and str(item.get("lane_id") or "")
    }
    tools_by_id = {
        str(item.get("tool_id") or ""): _safe_dict(item)
        for item in _safe_list(certificate.get("tools"))
        if isinstance(item, Mapping) and str(item.get("tool_id") or "")
    }
    role_aware = _safe_dict(certificate.get("role_aware"))
    promotion = _safe_dict(certificate.get("promotion"))
    elevated_ids = {
        str(item)
        for item in _safe_list(role_aware.get("elevated_tool_ids"))
    }
    promotion_ids = {
        str(item)
        for item in _safe_list(
            promotion.get("production_certified_tool_ids")
        )
    }
    elevation_decisions = {
        str(item.get("tool_id") or ""): _safe_dict(item)
        for item in _safe_list(role_aware.get("elevations"))
        if isinstance(item, Mapping) and str(item.get("tool_id") or "")
    }

    semantic_audit = _audit_semantic_lane_results(
        certifier=certifier,
        repo_root=root,
        semantic_results=[
            semantic_results[lane_id]
            for lane_id in sorted(semantic_results)
        ],
    )
    elevation_audit = _audit_required_elevations(
        certifier=certifier,
        repo_root=root,
        certificate=certificate,
        semantic_audit=semantic_audit,
    )
    canonical_roles = _safe_dict(
        certifier.load_authority_roles(root).get("tools")
    )

    required_specs = {
        tool_id: _tool_spec_for_elevation(certifier, tool_id) or {}
        for tool_id in REQUIRED_SEMANTIC_ELEVATIONS
    }
    offline = _production_fanin_offline_policy(
        certificate=certificate,
        semantic_results=semantic_results,
        required_lane_ids=[
            str(spec.get("lane_id") or "")
            for spec in required_specs.values()
        ],
    )

    per_tool: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    reconstruction_complete: list[str] = []
    reconstruction_incomplete: list[str] = []
    elevation_present: list[str] = []
    elevation_missing: list[str] = []
    elevation_without_reconstruction: list[str] = []
    elevation_with_disallowed_class: list[str] = []

    lane_audits = _safe_dict(semantic_audit.get("lanes"))
    required_elevation_tools = _safe_dict(elevation_audit.get("tools"))
    for tool_id in REQUIRED_SEMANTIC_ELEVATIONS:
        spec = required_specs[tool_id]
        lane_id = str(spec.get("lane_id") or "")
        lane = _safe_dict(semantic_results.get(lane_id))
        compact_tool = _safe_dict(
            _safe_dict(lane.get("per_tool")).get(tool_id)
        )
        tool_row = _safe_dict(tools_by_id.get(tool_id))
        reconstruction = (
            _independent_pnmr_reconstruction(
                certifier=certifier,
                semantic_result=lane,
                tool_id=tool_id,
                compact_tool=compact_tool,
            )
            if lane.get("status") == "ran"
            else {
                "valid": False,
                "recompute_valid": False,
                "recompute_failure": "semantic_lane_not_run",
                "check_kinds_present": [],
                "required_check_kinds": list(
                    PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS
                ),
                "required_kinds_present": False,
                "required_kinds_all_passed": False,
                "required_kinds_failed": [],
                "required_kinds_missing": list(
                    PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS
                ),
                "check_set_digest_sha256": None,
                "compact_check_set_digest_sha256": compact_tool.get(
                    "check_set_digest_sha256"
                ),
                "compact_binding_valid": False,
                "checks_total": 0,
                "checks_passed": 0,
                "check_status_counts": {},
                "raw_checks_embedded": False,
            }
        )
        lane_audit = _safe_dict(lane_audits.get(lane_id))
        lane_tool_audit = _safe_dict(
            _safe_dict(lane_audit.get("tools")).get(tool_id)
        )
        elevation_tool_audit = _safe_dict(
            required_elevation_tools.get(tool_id)
        )
        production_allowed = bool(
            _safe_dict(lane_audit.get("elevation_policy")).get(
                "valid"
            )
            is True
            and tool_id
            in {
                str(value)
                for value in _safe_list(
                    _safe_dict(
                        lane_audit.get("elevation_policy")
                    ).get("production_allowed_tool_ids")
                )
            }
        )
        authority_role = _safe_dict(canonical_roles.get(tool_id))
        decision = _safe_dict(elevation_decisions.get(tool_id))
        surfaces = {
            "role_aware": tool_id in elevated_ids,
            "promotion": tool_id in promotion_ids,
            "tool": tool_row.get("production_certified") is True,
            "lane": tool_id
            in {
                str(item)
                for item in _safe_list(lane.get("elevated_tool_ids"))
            },
            "decision": decision.get("elevated") is True,
        }
        surfaces_consistent = len(set(surfaces.values())) == 1
        production_present = bool(
            elevation_tool_audit.get("present") is True
            and surfaces_consistent
            and all(surfaces.values())
        )
        semantic_authority_valid = bool(
            lane_audit.get("valid") is True
            and lane_tool_audit.get(
                "checks_match_canonical_receipt"
            )
            is True
            and lane_tool_audit.get("artifact_validation_valid") is True
            and _safe_list(
                lane_tool_audit.get("expected_production_bindings")
            )
            and elevation_tool_audit.get("semantic_evidence_valid") is True
            and authority_role.get(
                "can_satisfy_certified_authority"
            )
            is True
        )
        eligible = bool(
            reconstruction.get("valid") is True
            and production_allowed
            and compact_tool.get("certified") is True
            and semantic_authority_valid
        )

        block_reasons: list[str] = []
        if lane.get("status") != "ran":
            block_reasons.append("semantic_lane_not_run")
        if reconstruction.get("valid") is not True:
            block_reasons.append(
                "independent_pnmr_reconstruction_incomplete"
            )
        if reconstruction.get("compact_binding_valid") is not True:
            block_reasons.append("compact_pnmr_binding_mismatch")
        if not production_allowed:
            block_reasons.append(
                "production_elevation_not_allowed_by_evidence_class"
            )
        if compact_tool.get("certified") is not True:
            block_reasons.append("semantic_tool_not_certified")
        if production_allowed and not semantic_authority_valid:
            block_reasons.append(
                "production_authority_or_artifact_binding_invalid"
            )
        if not surfaces_consistent and any(surfaces.values()):
            block_reasons.append("elevation_surface_mismatch")
            failures.append(f"{tool_id}:elevation_surface_mismatch")
        if production_present and reconstruction.get("valid") is not True:
            block_reasons.append(
                "elevation_without_independent_reconstruction"
            )
            elevation_without_reconstruction.append(tool_id)
            failures.append(
                f"{tool_id}:elevation_without_independent_reconstruction"
            )
        if production_present and not production_allowed:
            block_reasons.append(
                "elevation_with_disallowed_evidence_class"
            )
            elevation_with_disallowed_class.append(tool_id)
            failures.append(
                f"{tool_id}:elevation_with_disallowed_evidence_class"
            )
        if production_present and not eligible:
            block_reasons.append(
                "elevation_without_production_authority"
            )
            failures.append(
                f"{tool_id}:elevation_without_production_authority"
            )

        if reconstruction.get("valid") is True:
            reconstruction_complete.append(tool_id)
        else:
            reconstruction_incomplete.append(tool_id)
        if production_present:
            elevation_present.append(tool_id)
        else:
            elevation_missing.append(tool_id)

        per_tool[tool_id] = {
            "tool_id": tool_id,
            "lane_id": lane_id,
            "interface": spec.get("interface"),
            "evidence_class": (
                tool_row.get("evidence_class")
                or spec.get("evidence_class")
            ),
            "production_elevation_allowed": production_allowed,
            "lane_status": lane.get("status") or "missing",
            "lane_digest_sha256": lane.get("digest_sha256"),
            "compact_tool_certified": (
                compact_tool.get("certified") is True
            ),
            "independent_reconstruction": reconstruction,
            "semantic_authority_valid": semantic_authority_valid,
            "surfaces": surfaces,
            "surfaces_consistent": surfaces_consistent,
            "production_elevation_present": production_present,
            "eligible_for_production_elevation": eligible,
            "required_elevation_audit_present": bool(
                elevation_tool_audit.get("present")
            ),
            "block_reasons": sorted(set(block_reasons)),
        }

    declared_required = [
        str(item)
        for item in _safe_list(
            role_aware.get("required_baseline_elevations")
        )
    ]
    population_exact = bool(
        declared_required == list(REQUIRED_SEMANTIC_ELEVATIONS)
        and len(declared_required) == len(set(declared_required))
        and set(per_tool) == set(REQUIRED_SEMANTIC_ELEVATIONS)
    )
    reconstruction_surfaces_exact = bool(
        set(per_tool) == set(REQUIRED_SEMANTIC_ELEVATIONS)
        and all(
            isinstance(row.get("independent_reconstruction"), Mapping)
            for row in per_tool.values()
        )
    )
    checks_never_collapsed = bool(
        reconstruction_surfaces_exact
        and all(
            _safe_dict(row.get("independent_reconstruction")).get(
                "compact_binding_valid"
            )
            is True
            for row in per_tool.values()
        )
    )
    raw_checks_not_reembedded = bool(
        all(
            "checks"
            not in _safe_dict(row.get("independent_reconstruction"))
            and _safe_dict(row.get("independent_reconstruction")).get(
                "raw_checks_embedded"
            )
            is False
            for row in per_tool.values()
        )
    )
    all_reconstructions_valid = bool(
        not reconstruction_incomplete
        and set(reconstruction_complete)
        == set(REQUIRED_SEMANTIC_ELEVATIONS)
    )
    if not population_exact:
        failures.append("required_elevation_population_mismatch")
    if not certificate_identity["valid"]:
        failures.append("role_aware_certificate_identity_invalid")
    if elevation_audit.get("valid") is not True:
        failures.extend(
            f"required_elevation_audit:{item}"
            for item in _safe_list(elevation_audit.get("failures"))
        )
    if not checks_never_collapsed:
        failures.append("required_pnmr_compact_binding_invalid")
    if not raw_checks_not_reembedded:
        failures.append("raw_checks_reembedded")
    if not offline["satisfied"]:
        failures.append("offline_policy_not_satisfied")

    no_elevation_without_reconstruction = not (
        elevation_without_reconstruction
    )
    production_allowed_respected = not (
        elevation_with_disallowed_class
    )
    structurally_valid = bool(
        population_exact
        and reconstruction_surfaces_exact
        and certificate_identity["valid"]
        and checks_never_collapsed
        and raw_checks_not_reembedded
        and offline["satisfied"]
        and no_elevation_without_reconstruction
        and production_allowed_respected
        and elevation_audit.get("valid") is True
        and not failures
    )
    fanin_closed = bool(
        structurally_valid
        and all_reconstructions_valid
        and not elevation_missing
        and set(elevation_present) == set(REQUIRED_SEMANTIC_ELEVATIONS)
        and all(
            row.get("eligible_for_production_elevation") is True
            for row in per_tool.values()
        )
    )

    observed = observed_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    receipt: dict[str, Any] = {
        "schema_version": PRODUCTION_ELEVATION_FANIN_SCHEMA_VERSION,
        "interface": PRODUCTION_ELEVATION_FANIN_INTERFACE,
        "program_interface": PROGRAM_INTERFACE,
        "program_goal_id": PROGRAM_GOAL_ID,
        "goal_id": PRODUCTION_ELEVATION_FANIN_GOAL_ID,
        "task_id": PRODUCTION_ELEVATION_FANIN_TASK_ID,
        "program": PRODUCTION_ELEVATION_FANIN_PROGRAM,
        "observed_at": observed,
        "description": (
            "Fail-closed production-semantic elevation fan-in. Each required "
            "tool is independently reconstructed from its canonical PNMR "
            "receipt and exact compact binding before production elevation."
        ),
        "policy": {
            "required_check_kinds": list(
                PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS
            ),
            "independent_reconstruction_required_before_production_elevation": True,
            "production_elevation_allowed_gate_required": True,
            "hardcoded_success_forbidden": True,
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "self_referential_current_tree_claim_forbidden": True,
            "merge_claim_forbidden": True,
            "deployment_claim_forbidden": True,
            "raw_checks_bound_by_digest_only": True,
        },
        "required_tools": list(REQUIRED_SEMANTIC_ELEVATIONS),
        "tools": per_tool,
        "certificate_identity": certificate_identity,
        "offline_derivation": offline,
        "summary": {
            "required_count": len(REQUIRED_SEMANTIC_ELEVATIONS),
            "independent_reconstruction_complete": reconstruction_complete,
            "independent_reconstruction_incomplete": reconstruction_incomplete,
            "all_required_reconstructions_valid": (
                all_reconstructions_valid
            ),
            "production_elevation_present": elevation_present,
            "production_elevation_missing": elevation_missing,
            "elevation_without_reconstruction": (
                elevation_without_reconstruction
            ),
            "elevation_with_disallowed_evidence_class": (
                elevation_with_disallowed_class
            ),
            "structurally_valid": structurally_valid,
            "fanin_closed": fanin_closed,
            "failures": sorted(set(failures)),
        },
        "required_elevation_audit": {
            "valid": elevation_audit.get("valid"),
            "required": list(_safe_list(elevation_audit.get("required"))),
            "present": list(_safe_list(elevation_audit.get("present"))),
            "missing": list(_safe_list(elevation_audit.get("missing"))),
            "expected_global_production_certified_tool_ids": list(
                _safe_list(
                    elevation_audit.get(
                        "expected_global_production_certified_tool_ids"
                    )
                )
            ),
            "failures": list(
                _safe_list(elevation_audit.get("failures"))
            ),
        },
        "role_aware_certificate": {
            "digest_sha256": certificate.get(
                "certificate_digest_sha256"
            ),
            "interface": certificate.get("interface"),
            "projection_model": "digest_bound_compact_projection/v1",
            "raw_certificate_embedded": False,
        },
        "acceptance": {
            "role_aware_certificate_identity_bound": (
                certificate_identity["valid"]
            ),
            "required_tools_population_exact": population_exact,
            "each_required_tool_has_independent_reconstruction_surface": (
                reconstruction_surfaces_exact
            ),
            "all_required_reconstructions_valid": (
                all_reconstructions_valid
            ),
            "production_elevation_requires_independent_pnmr": (
                no_elevation_without_reconstruction
            ),
            "no_elevation_without_reconstruction": (
                no_elevation_without_reconstruction
            ),
            "production_elevation_allowed_respected": (
                production_allowed_respected
            ),
            "checks_never_collapsed": checks_never_collapsed,
            "raw_checks_not_reembedded": raw_checks_not_reembedded,
            "offline_only": offline["satisfied"],
            "structurally_valid": structurally_valid,
            "fanin_closed": fanin_closed,
            "merge_not_claimed": True,
            "deployment_not_claimed": True,
            "required_elevation_audit_bound": (
                elevation_audit.get("valid") is True
            ),
        },
        "evidence": {
            "integration_test": (
                DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE.as_posix()
            ),
            "receipt": (
                DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix()
            ),
            "release_candidate": (
                DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix()
            ),
            "release_candidate_integration_test": (
                DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE.as_posix()
            ),
            "certifier": DEFAULT_CERTIFIER_RELATIVE.as_posix(),
            "receipt_builder": DEFAULT_BUILDER_RELATIVE.as_posix(),
            "validation_command": (
                PRODUCTION_ELEVATION_FANIN_VALIDATION_COMMAND
            ),
        },
        "claims": {
            "merge": False,
            "deployment": False,
            "post_merge_attestation": False,
            "self_referential_current_tree": False,
            "max_stage": RELEASE_CANDIDATE_MAX_STAGE,
        },
        "status": (
            "production_semantic_elevation_fanin_closed"
            if fanin_closed
            else (
                "production_semantic_elevation_fanin_structurally_valid"
                if structurally_valid
                else "production_semantic_elevation_fanin_blocked"
            )
        ),
    }
    receipt = certifier.public_evidence_projection(
        receipt,
        repo_root=root,
    )
    public_policy = certifier.public_evidence_audit(
        receipt,
        repo_root=root,
    )
    receipt["public_evidence_policy"] = public_policy
    if not public_policy.get("satisfied"):
        receipt["acceptance"]["offline_only"] = False
        receipt["acceptance"]["structurally_valid"] = False
        receipt["acceptance"]["fanin_closed"] = False
        receipt["summary"]["structurally_valid"] = False
        receipt["summary"]["fanin_closed"] = False
        receipt["status"] = "production_semantic_elevation_fanin_blocked"
        public_failures = receipt["summary"]["failures"]
        if "public_evidence_redaction_failed" not in public_failures:
            public_failures.append("public_evidence_redaction_failed")
    receipt["receipt_digest_sha256"] = content_digest(receipt)
    return receipt


def compact_production_elevation_fanin_binding(
    fanin: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the digest-bound fan-in projection embedded in a candidate."""

    tools = _safe_dict(fanin.get("tools"))
    summary = _safe_dict(fanin.get("summary"))
    acceptance = _safe_dict(fanin.get("acceptance"))
    return {
        "interface": fanin.get("interface"),
        "schema_version": fanin.get("schema_version"),
        "goal_id": fanin.get("goal_id"),
        "task_id": fanin.get("task_id"),
        "status": fanin.get("status"),
        "receipt_digest_sha256": fanin.get("receipt_digest_sha256"),
        "certificate_identity_valid": _safe_dict(
            fanin.get("certificate_identity")
        ).get("valid"),
        "structurally_valid": summary.get("structurally_valid"),
        "all_required_reconstructions_valid": summary.get(
            "all_required_reconstructions_valid"
        ),
        "fanin_closed": summary.get("fanin_closed"),
        "required_tools": list(_safe_list(fanin.get("required_tools"))),
        "independent_reconstruction_complete": list(
            _safe_list(
                summary.get("independent_reconstruction_complete")
            )
        ),
        "production_elevation_present": list(
            _safe_list(summary.get("production_elevation_present"))
        ),
        "production_elevation_missing": list(
            _safe_list(summary.get("production_elevation_missing"))
        ),
        "tool_reconstruction_digests": {
            tool_id: _safe_dict(
                _safe_dict(tools.get(tool_id)).get(
                    "independent_reconstruction"
                )
            ).get("check_set_digest_sha256")
            for tool_id in _safe_list(fanin.get("required_tools"))
        },
        "checks_never_collapsed": acceptance.get(
            "checks_never_collapsed"
        ),
        "offline_only": acceptance.get("offline_only"),
        "path": (
            DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix()
        ),
        "integration_test": (
            DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE.as_posix()
        ),
        "raw_receipt_embedded": False,
    }


def verify_checked_production_elevation_fanin(
    *,
    repo_root: Path,
    live_fanin: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that the durable fan-in artifact exactly matches live evidence."""

    path = repo_root / DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE
    checked = load_json(path)
    if not isinstance(checked, Mapping):
        return {
            "present": False,
            "content_identity": None,
            "stored_digest_valid": False,
            "identity_valid": False,
            "matches_live": False,
            "stored_digest_sha256": None,
            "live_digest_sha256": live_fanin.get(
                "receipt_digest_sha256"
            ),
        }
    stored_digest = str(checked.get("receipt_digest_sha256") or "")
    computed_digest = content_digest(
        {
            key: value
            for key, value in checked.items()
            if key != "receipt_digest_sha256"
        }
    )
    stored_digest_valid = bool(
        stored_digest and stored_digest == computed_digest
    )
    identity_valid = bool(
        checked.get("interface") == PRODUCTION_ELEVATION_FANIN_INTERFACE
        and checked.get("schema_version")
        == PRODUCTION_ELEVATION_FANIN_SCHEMA_VERSION
        and checked.get("goal_id") == PRODUCTION_ELEVATION_FANIN_GOAL_ID
        and checked.get("task_id") == PRODUCTION_ELEVATION_FANIN_TASK_ID
    )
    live_digest = str(live_fanin.get("receipt_digest_sha256") or "")
    return {
        "present": True,
        "content_identity": sha256_file(path),
        "stored_digest_valid": stored_digest_valid,
        "identity_valid": identity_valid,
        "matches_live": bool(
            stored_digest_valid
            and identity_valid
            and stored_digest == live_digest
        ),
        "stored_digest_sha256": stored_digest or None,
        "live_digest_sha256": live_digest or None,
    }


def write_receipt(receipt: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    # Atomic replace so partial writes never leave a corrupt receipt.
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the formal verification tactician completion receipt "
            f"({INTERFACE}), optional role-aware deployment receipt "
            f"({ROLE_AWARE_INTERFACE}), and optional role-aware release "
            f"candidate ({RELEASE_CANDIDATE_INTERFACE})."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from this file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Receipt output path (default: docs/architecture/...receipt.json)",
    )
    parser.add_argument(
        "--role-aware-output",
        type=Path,
        default=None,
        help=(
            "Also write the role-aware deployment receipt "
            f"(default when --role-aware: {DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE})"
        ),
    )
    parser.add_argument(
        "--role-aware",
        action="store_true",
        help=(
            "Build and write RoleAwareFormalVerificationRelease@1 after the "
            "completion receipt (FVT-G200 reissue)"
        ),
    )
    parser.add_argument(
        "--release-candidate-output",
        type=Path,
        default=None,
        help=(
            "Also write the role-aware release candidate "
            f"(default when --release-candidate: "
            f"{DEFAULT_RELEASE_CANDIDATE_RELATIVE})"
        ),
    )
    parser.add_argument(
        "--release-candidate",
        action="store_true",
        help=(
            "Build and write RoleAwareFormalVerificationReleaseCandidate@1 "
            "from the complete role-aware matrix (FVT-G213)"
        ),
    )
    parser.add_argument(
        "--production-elevation-fanin-output",
        type=Path,
        default=None,
        help=(
            "Write the production-semantic elevation fan-in "
            f"(default: "
            f"{DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE})"
        ),
    )
    parser.add_argument(
        "--production-elevation-fanin",
        action="store_true",
        help=(
            "Build and write ProductionSemanticElevationFanIn@1 "
            "(FVT-G213 / FVT-081)"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print receipt JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary",
    )
    parser.add_argument(
        "--observed-at",
        type=str,
        default=None,
        help="Override observed_at timestamp (ISO-8601 UTC)",
    )
    parser.add_argument(
        "--supervisor-release-evidence",
        type=Path,
        default=None,
        help=(
            "Explicit AgentSupervisorReleaseEvidence@1 JSON export for FVT-083. "
            "After publication, run `git fetch origin` and invoke this command "
            "again to finalize the origin/main ancestry phase."
        ),
    )
    parser.add_argument(
        "--supervisor-integration-branch",
        type=str,
        default=ROLE_AWARE_INTEGRATION_BRANCH,
        help=(
            "Exact local integration branch accepted in supervisor merge "
            f"evidence (default: {ROLE_AWARE_INTEGRATION_BRANCH})"
        ),
    )
    parser.add_argument(
        "--supervisor-task-state",
        type=Path,
        default=None,
        help=(
            "Diagnostic-only raw supervisor task-state JSON; cannot grant "
            "release authority"
        ),
    )
    parser.add_argument(
        "--supervisor-event-log",
        type=Path,
        default=None,
        help=(
            "Diagnostic-only raw supervisor event JSONL; cannot grant release "
            "authority"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if bool(args.supervisor_task_state) != bool(args.supervisor_event_log):
        parser.error(
            "--supervisor-task-state and --supervisor-event-log must be supplied together"
        )
    if args.supervisor_task_state or args.supervisor_event_log:
        parser.error(
            "raw --supervisor-task-state/--supervisor-event-log inputs are "
            "diagnostic only and cannot be used by the release builder; export "
            "AgentSupervisorReleaseEvidence@1 and pass "
            "--supervisor-release-evidence"
        )
    integration_branch = str(args.supervisor_integration_branch or "").strip()
    if not integration_branch:
        parser.error("--supervisor-integration-branch must not be empty")

    root = (args.repo_root or repo_root_from()).resolve()
    want_release_candidate = bool(
        args.release_candidate or args.release_candidate_output is not None
    )
    want_production_elevation_fanin = bool(
        args.production_elevation_fanin
        or args.production_elevation_fanin_output is not None
        or want_release_candidate
    )
    want_role_aware = bool(args.role_aware or args.role_aware_output is not None)
    if args.supervisor_release_evidence and not want_role_aware:
        parser.error(
            "--supervisor-release-evidence requires --role-aware or "
            "--role-aware-output"
        )
    if want_role_aware and not args.supervisor_release_evidence:
        parser.error(
            "role-aware generation requires --supervisor-release-evidence; "
            "raw supervisor state/events are not release authority"
        )
    receipt = build_receipt(repo_root=root, observed_at=args.observed_at)

    if (
        args.stdout
        and not want_role_aware
        and not want_release_candidate
        and not want_production_elevation_fanin
    ):
        json.dump(receipt, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        output = (
            args.output.resolve()
            if args.output
            else (root / DEFAULT_RECEIPT_RELATIVE)
        )
        write_receipt(receipt, output)
        if not args.quiet:
            print(f"wrote {output}", file=sys.stderr)

    role_aware_receipt: dict[str, Any] | None = None
    release_candidate: dict[str, Any] | None = None
    production_elevation_fanin: dict[str, Any] | None = None
    role_certificate: dict[str, Any] | None = None
    role_full_evidence: dict[str, Any] = {}
    if (
        want_role_aware
        or want_release_candidate
        or want_production_elevation_fanin
    ):
        certifier = _load_certifier_module(root)
        role_certificate = certifier.build_certificate(
            repo_root=root,
            observed_at=args.observed_at or receipt.get("observed_at"),
            role_aware=True,
            full_evidence_out=role_full_evidence,
        )
        # Only rewrite the durable certificate when reissuing role-aware
        # deployment attestation. Release-candidate mode reads live evidence
        # without mutating the checked-in certificate artifact.
        if want_role_aware:
            certifier.write_certificate(
                role_certificate,
                root / TOOLCHAIN_CERT_RELATIVE,
            )
            # Rebuild completion after the role-aware certificate is durable so
            # its artifact binding cannot silently refer to the predecessor cert.
            receipt = build_receipt(
                repo_root=root,
                observed_at=args.observed_at or receipt.get("observed_at"),
            )
            completion_output = (
                args.output.resolve()
                if args.output
                else (root / DEFAULT_RECEIPT_RELATIVE)
            )
            write_receipt(receipt, completion_output)

    if want_role_aware:
        assert role_certificate is not None
        supervisor_snapshot = None
        if args.supervisor_release_evidence:
            try:
                supervisor_snapshot = load_supervisor_release_evidence(
                    args.supervisor_release_evidence
                )
            except ValueError as exc:
                parser.error(str(exc))
        role_aware_receipt = build_role_aware_deployment_receipt(
            repo_root=root,
            observed_at=args.observed_at or receipt.get("observed_at"),
            completion_receipt=receipt,
            role_aware_certificate=role_certificate,
            supervisor_evidence=supervisor_snapshot,
            supervisor_integration_branch=integration_branch,
        )
        if args.stdout and args.role_aware and args.output is None and not (
            args.release_candidate and args.release_candidate_output is None
        ):
            json.dump(role_aware_receipt, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
        else:
            role_output = (
                args.role_aware_output.resolve()
                if args.role_aware_output
                else (root / DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE)
            )
            write_receipt(role_aware_receipt, role_output)
            if not args.quiet:
                print(f"wrote {role_output}", file=sys.stderr)

    if want_production_elevation_fanin:
        assert role_certificate is not None
        production_elevation_fanin = (
            build_production_semantic_elevation_fanin(
                repo_root=root,
                observed_at=args.observed_at or receipt.get("observed_at"),
                role_aware_certificate=role_certificate,
            )
        )
        fanin_output = (
            args.production_elevation_fanin_output.resolve()
            if args.production_elevation_fanin_output
            else (
                root
                / DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE
            )
        )
        write_receipt(production_elevation_fanin, fanin_output)
        if not args.quiet:
            print(f"wrote {fanin_output}", file=sys.stderr)

    if want_release_candidate:
        assert role_certificate is not None
        release_candidate = build_role_aware_release_candidate(
            repo_root=root,
            observed_at=args.observed_at or receipt.get("observed_at"),
            role_aware_certificate=role_certificate,
            source_specialized_receipt_aggregation=_safe_dict(
                role_full_evidence.get("specialized_receipt_aggregation")
            ),
        )
        if (
            args.stdout
            and args.release_candidate
            and args.output is None
            and args.release_candidate_output is None
        ):
            json.dump(release_candidate, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
        else:
            candidate_output = (
                args.release_candidate_output.resolve()
                if args.release_candidate_output
                else (root / DEFAULT_RELEASE_CANDIDATE_RELATIVE)
            )
            write_receipt(release_candidate, candidate_output)
            if not args.quiet:
                print(f"wrote {candidate_output}", file=sys.stderr)

    if not args.quiet:
        implementation = receipt["implementation"]
        deployment = receipt["deployment"]
        hard_zero = receipt["hard_zero_gates"]
        print(
            f"implementation={implementation['status']} "
            f"bound={implementation['child_goals_bound']}/"
            f"{implementation['child_goal_count']}",
            file=sys.stderr,
        )
        print(
            f"deployment={deployment['status']} "
            f"production_certified={deployment['production_certified_tool_ids']}",
            file=sys.stderr,
        )
        print(
            "hard_zero="
            + json.dumps({k: hard_zero[k] for k in HARD_ZERO_GATE_KEYS}),
            file=sys.stderr,
        )
        print(f"receipt_identity={receipt['receipt_identity']}", file=sys.stderr)
        if role_aware_receipt is not None:
            print(
                f"role_aware_status={role_aware_receipt['status']} "
                f"elevated={role_aware_receipt['elevations']['elevated_tool_ids']}",
                file=sys.stderr,
            )
            print(
                f"role_aware_identity={role_aware_receipt['receipt_identity']}",
                file=sys.stderr,
            )
        if production_elevation_fanin is not None:
            fanin_summary = _safe_dict(
                production_elevation_fanin.get("summary")
            )
            print(
                "production_elevation_fanin_status="
                f"{production_elevation_fanin.get('status')} "
                f"structurally_valid="
                f"{fanin_summary.get('structurally_valid')} "
                f"closed={fanin_summary.get('fanin_closed')}",
                file=sys.stderr,
            )
        if release_candidate is not None:
            print(
                f"release_candidate_status={release_candidate['status']} "
                f"stage={release_candidate['readiness_stage']} "
                f"blockers={len(release_candidate['blockers'])}",
                file=sys.stderr,
            )
            print(
                f"release_candidate_identity="
                f"{release_candidate['candidate_identity']}",
                file=sys.stderr,
            )

    # Exit 0 when the receipt was produced. Incomplete implementation or
    # partial deployment is recorded in the receipt, not as a builder crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
