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
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

INTERFACE: Final = "FormalVerificationTacticianCompletionReceipt@1"
PROGRAM_INTERFACE: Final = "FormalVerificationTacticianRelease@1"
SCHEMA_VERSION: Final = "formal-verification-tactician-completion-receipt/v1"
PROGRAM_GOAL_ID: Final = "FVT-G000"
COMPLETION_GOAL_ID: Final = "FVT-G090"
TASK_ID: Final = "FVT-036"
PROGRAM: Final = "formal-verification-tactician/readiness"

# Role-aware deployment reissue (FVT-G200 / FVT-053).
ROLE_AWARE_INTERFACE: Final = "RoleAwareFormalVerificationRelease@1"
ROLE_AWARE_SCHEMA_VERSION: Final = "formal-verification-role-aware-deployment-receipt/v1"
ROLE_AWARE_GOAL_ID: Final = "FVT-G200"
ROLE_AWARE_TASK_ID: Final = "FVT-053"

DEFAULT_OBJECTIVES_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_readiness.objectives.md"
)
DEFAULT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json"
)
DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_deployment_receipt.json"
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
        "docs/architecture/formal_verification_toolchain_certificate.json",
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


def _derive_git_commit_binding(
    *,
    repo_root: Path | None,
    implementation_commit: str,
    merge_commit: str,
    target_branch: str,
    integration_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently bind supervisor commit claims to the published Git DAG."""

    result: dict[str, Any] = {
        "valid": False,
        "repository_bound": False,
        "implementation_commit_exists": False,
        "merge_commit_exists": False,
        "implementation_is_ancestor": False,
        "source_trees_bound": False,
        "target_branch_bound": False,
        "published_to_origin_main": False,
        "implementation_commit": implementation_commit or None,
        "merge_commit": merge_commit or None,
        "implementation_tree": None,
        "merge_tree": None,
        "publication_ref": "refs/remotes/origin/main",
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

    result["source_trees_bound"] = bool(
        COMMIT_RE.fullmatch(str(implementation_tree or ""))
        and COMMIT_RE.fullmatch(str(merge_tree or ""))
        and integration_proof.get("implementation_tree") == implementation_tree
        and integration_proof.get("merge_tree") == merge_tree
    )
    if not result["source_trees_bound"]:
        failures.append("commit_source_trees_not_bound")

    result["target_branch_bound"] = target_branch in {
        "origin/main",
        "refs/remotes/origin/main",
    }
    if not result["target_branch_bound"]:
        failures.append("merge_target_not_origin_main")

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

    result["valid"] = not failures
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


def derive_supervisor_binding(
    evidence: Mapping[str, Any] | None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a G212 export's task/CID, validation, and merge evidence."""

    trusted_release = _trusted_release_evidence_snapshot(
        evidence,
        repo_root=repo_root,
    )
    snapshot = _safe_dict(trusted_release.get("snapshot"))
    task_state = _safe_dict(snapshot.get("task_state"))
    identity = _safe_dict(task_state.get("canonical_identity"))
    expected_cid = str(identity.get("canonical_task_cid") or "")
    expected_key = str(identity.get("canonical_task_key") or "")
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
        event
        for event in _safe_list(snapshot.get("events"))
        if isinstance(event, Mapping)
    ]
    events_have_canonical_identity = bool(events) and all(
        str(event.get("task_id") or "") == ROLE_AWARE_TASK_ID
        and str(event.get("canonical_task_cid") or "") == expected_cid
        and str(event.get("canonical_task_key") or "") == expected_key
        and isinstance(event.get("sequence"), int)
        and SHA256_RE.fullmatch(
            str(event.get("event_id") or "").removeprefix("sha256:")
        )
        for event in events
    )
    finished_events = [
        event
        for event in events
        if event.get("type") == "implementation_finished"
    ]
    terminal_events: list[Mapping[str, Any]] = []
    successful_receipts: list[Mapping[str, Any]] = []
    terminal_commit_bindings: list[Mapping[str, Any]] = []
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
        )
        receipts = [
            receipt
            for receipt in _safe_list(event.get("completion_receipts"))
            if isinstance(receipt, Mapping)
            and receipt.get("schema") == SUPERVISOR_COMPLETION_SCHEMA
            and receipt.get("status") == "succeeded"
            and str(receipt.get("task_id") or "") == ROLE_AWARE_TASK_ID
            and str(receipt.get("canonical_task_cid") or "") == expected_cid
            and str(receipt.get("canonical_task_key") or "") == expected_key
            and str(receipt.get("implementation_commit") or "")
            == implementation_commit
            and str(receipt.get("merge_commit") or "") == merge_commit
        ]
        coherent = bool(
            receipts
            and validation.get("attempted") is True
            and validation.get("passed") is True
            and validation.get("returncode") == 0
            and str(validation.get("target_commit") or "")
            == implementation_commit
            and merge.get("merged") is True
            and COMMIT_RE.fullmatch(implementation_commit)
            and COMMIT_RE.fullmatch(merge_commit)
            and str(merge.get("implementation_commit") or "")
            == implementation_commit
            and commit_binding["valid"] is True
        )
        if coherent:
            terminal_events.append(event)
            successful_receipts.extend(receipts)
            terminal_commit_bindings.append(commit_binding)

    terminal_event = terminal_events[-1] if terminal_events else {}
    terminal_merge = _safe_dict(terminal_event.get("merge"))
    terminal_implementation_commit = str(
        terminal_event.get("implementation_commit")
        or terminal_merge.get("implementation_commit")
        or ""
    )
    terminal_merge_commit = str(terminal_merge.get("merge_commit") or "")
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
    )
    task_cid_bound = bool(
        expected_cid
        and expected_key
        and events_have_canonical_identity
        and str(snapshot.get("task_id") or "") == ROLE_AWARE_TASK_ID
    )
    bound = bool(
        task_cid_bound
        and event_chain_bound
        and state_terminal_bound
        and successful_receipts
        and terminal_events
    )
    return {
        "present": bool(evidence),
        "bound": bound,
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
        "member_completion_receipt_bound": bool(successful_receipts),
        "validation_bound": bool(terminal_events),
        "merge_commit_tree_bound": bool(terminal_events),
        "canonical_task_cid": expected_cid or None,
        "canonical_task_key": expected_key or None,
        "successful_completion_receipts": successful_receipts,
        "validation_events": terminal_events,
        "merge_events": terminal_events,
        "commit_bindings": terminal_commit_bindings,
        "snapshot_digest_sha256": (
            content_digest(snapshot) if snapshot else None
        ),
        "snapshot": snapshot,
        "block_reasons": [
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
                ("canonical_task_cid_not_bound", task_cid_bound),
                ("terminal_task_state_not_bound", state_terminal_bound),
                (
                    "member_completion_receipt_missing",
                    bool(successful_receipts),
                ),
                ("validation_evidence_missing", bool(terminal_events)),
                (
                    "merge_commit_tree_evidence_missing",
                    bool(terminal_commit_bindings),
                ),
            )
            if not condition
        ],
    }


def build_role_aware_deployment_receipt(
    *,
    repo_root: Path,
    observed_at: str | None = None,
    completion_receipt: Mapping[str, Any] | None = None,
    role_aware_certificate: Mapping[str, Any] | None = None,
    supervisor_evidence: Mapping[str, Any] | None = None,
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
    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "unknown")
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
        for tool_id, per_tool in _safe_dict(result.get("per_tool")).items():
            projected_checks = _safe_list(_safe_dict(per_tool).get("checks"))
            projected_digest = certifier.content_digest(projected_checks)
            if projected_digest != _safe_dict(per_tool).get(
                "check_set_digest_sha256"
            ):
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
    platform_exceptions = [
        dict(item)
        for item in _safe_list(managed.get("platform_exceptions"))
        if isinstance(item, Mapping)
    ]
    platform_exceptions_valid = all(
        item.get("narrow_scope") is True
        and item.get("complete") is False
        and item.get("production_certified") is False
        and item.get("classification") == "unsupported_here"
        and bool(
            set(str(item.get("basis") or "").split("+"))
            <= {
                "deployment_contract.supported_platforms",
                "tool.pins.platform",
            }
        )
        for item in platform_exceptions
    )

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
        "authority_roles_bound": bool(
            authority_roles.get("present")
            and authority_roles.get("policy_digest_sha256")
        ),
        "authority_ceiling_respected": authority_ceiling_respected,
        "disagreement_quarantines_bound": certificate_digest_valid
        and isinstance(certificate.get("disagreement_quarantines"), list),
        "public_surfaces_bound": bool(
            (completion.get("implementation") or {}).get("public_operations_bound")
        ),
        "supervisor_evidence_bound": supervisor["bound"],
        "semantic_receipts_full_and_bound": semantic_receipts_full_and_bound,
        "lean_runtime_mtl_authorization_elevated": not missing_required,
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
        "program": "formal-verification-tactician/toolchain-release",
        "observed_at": timestamp,
        "binding_mode": "two_phase_source_then_attestation_publication",
        "status": status,
        "description": (
            "Fail-closed role-aware deployment attestation. It retains complete "
            "semantic receipts and exact identities, distinguishes unsupported "
            "platforms from missing supported capabilities, and cannot claim "
            "readiness without authoritative supervisor validation/merge evidence."
        ),
        "source": source_attestation,
        "acceptance": acceptance,
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
            "role_aware": role_aware,
            "promotion": promotion,
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
            "semantic_lane_results": certificate.get("semantic_lane_results") or [],
            "managed_deployment_readiness": managed,
            "tools": [tools[tool_id] for tool_id in sorted(tools)],
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
            "details": role_aware.get("elevations") or [],
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
            f"({INTERFACE}) and optional role-aware deployment receipt "
            f"({ROLE_AWARE_INTERFACE})."
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
        "--supervisor-task-state",
        type=Path,
        default=None,
        help="Read-only FVT-053 supervisor task-state JSON snapshot",
    )
    parser.add_argument(
        "--supervisor-event-log",
        type=Path,
        default=None,
        help="Read-only FVT-053 supervisor durable event JSONL",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if bool(args.supervisor_task_state) != bool(args.supervisor_event_log):
        parser.error(
            "--supervisor-task-state and --supervisor-event-log must be supplied together"
        )

    root = (args.repo_root or repo_root_from()).resolve()
    receipt = build_receipt(repo_root=root, observed_at=args.observed_at)

    if args.stdout and not args.role_aware:
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
    if args.role_aware or args.role_aware_output is not None:
        certifier = _load_certifier_module(root)
        role_certificate = certifier.build_certificate(
            repo_root=root,
            observed_at=args.observed_at or receipt.get("observed_at"),
            role_aware=True,
        )
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
        supervisor_snapshot = None
        if args.supervisor_task_state and args.supervisor_event_log:
            supervisor_snapshot = load_supervisor_evidence_snapshot(
                task_state_path=args.supervisor_task_state.resolve(),
                event_log_path=args.supervisor_event_log.resolve(),
            )
        role_aware_receipt = build_role_aware_deployment_receipt(
            repo_root=root,
            observed_at=args.observed_at or receipt.get("observed_at"),
            completion_receipt=receipt,
            role_aware_certificate=role_certificate,
            supervisor_evidence=supervisor_snapshot,
        )
        if args.stdout and args.role_aware and args.output is None:
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

    # Exit 0 when the receipt was produced. Incomplete implementation or
    # partial deployment is recorded in the receipt, not as a builder crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
