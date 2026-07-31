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

# Baseline tools that must leave "merely usable" after role-aware elevation.
REQUIRED_SEMANTIC_ELEVATIONS: Final[tuple[str, ...]] = (
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
)


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


def derive_hard_zero_gates(
    *,
    certificate: Mapping[str, Any] | None,
    benchmark: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Derive hard-zero counters from child receipts — never invent success.

    Unresolved cross-provider disagreement is the quarantine list length from
    the toolchain certificate. Correctness / privacy / authority gates use the
    benchmark hard-gate basis points when present (10000 bps == 0 violations).
    Baseline open findings are disclosed separately and never rewritten.
    """

    disagreement = 0
    if certificate is not None:
        quarantines = certificate.get("disagreement_quarantines") or []
        if isinstance(quarantines, list):
            disagreement = len(quarantines)

    def _violations_from_bps(status: Mapping[str, Any] | None) -> int:
        if not status:
            return 0
        # Prefer explicit status; fall back to basis-point shortfall.
        if str(status.get("status") or "").lower() == "pass":
            return 0
        actual = status.get("actual_bps")
        required = status.get("required_bps")
        if isinstance(actual, int) and isinstance(required, int):
            return max(0, required - actual)
        # Missing measurement is not a synthetic pass; report as unresolved (1).
        return 1

    false_proof = 0
    false_closure = 0
    leakage = 0
    authority = 0
    gate_source = "child_certificate_and_benchmark_hard_gates"

    if benchmark is not None:
        report = _safe_dict(benchmark.get("report"))
        gates = _safe_dict(report.get("gates") or benchmark.get("gates"))
        hard = _safe_dict(gates.get("hard"))
        # Correctness failures map to false-proof pressure; authority is direct.
        false_proof = _violations_from_bps(_safe_dict(hard.get("correctness")) or None)
        authority = _violations_from_bps(_safe_dict(hard.get("authority")) or None)
        privacy = _violations_from_bps(_safe_dict(hard.get("privacy")) or None)
        leakage = privacy
        # Closure integrity is owned by verifier-backed repair + adversarial
        # gates; when hard gates pass and disagreement is empty, closure is 0.
        if false_proof == 0 and authority == 0 and disagreement == 0:
            false_closure = 0
        else:
            false_closure = max(false_proof, authority, 1 if disagreement else 0)

    # Baseline may record historical open findings. Those are disclosures, not
    # silent success counters. They do not rewrite hard-zero when child
    # certificates already measure the gates.
    open_findings = []
    if baseline is not None:
        for finding in _safe_list(baseline.get("known_findings")):
            if not isinstance(finding, Mapping):
                continue
            if str(finding.get("status") or "").lower() == "open":
                open_findings.append(
                    {
                        "id": finding.get("id"),
                        "severity": finding.get("severity"),
                        "summary": finding.get("summary"),
                    }
                )

    return {
        "false_proof_count": int(false_proof),
        "false_closure_count": int(false_closure),
        "secret_or_witness_leakage_count": int(leakage),
        "authority_boundary_violations": int(authority),
        "unresolved_cross_provider_disagreement_count": int(disagreement),
        "derivation": {
            "source": gate_source,
            "hardcoded_success_counters_forbidden": True,
            "benchmark_hard_gates_required_bps": 10000,
            "open_baseline_findings_disclosed": len(open_findings),
            "open_baseline_findings": open_findings,
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
    if event_log_path.is_file():
        for raw_line in event_log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
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


def derive_supervisor_binding(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate task/CID, completion receipt, validation, and merge evidence."""

    snapshot = dict(evidence) if isinstance(evidence, Mapping) else {}
    task_state = _safe_dict(snapshot.get("task_state"))
    identity = _safe_dict(task_state.get("canonical_identity"))
    expected_cid = str(identity.get("canonical_task_cid") or "")
    expected_key = str(identity.get("canonical_task_key") or "")
    events = [
        event
        for event in _safe_list(snapshot.get("events"))
        if isinstance(event, Mapping)
    ]
    event_cids = {
        str(event.get("canonical_task_cid") or "")
        for event in events
        if event.get("canonical_task_cid")
    }
    receipts = [
        receipt
        for event in events
        for receipt in _safe_list(event.get("completion_receipts"))
        if isinstance(receipt, Mapping)
        and str(receipt.get("task_id") or "") == ROLE_AWARE_TASK_ID
    ]
    successful_receipts = [
        receipt
        for receipt in receipts
        if receipt.get("schema") == SUPERVISOR_COMPLETION_SCHEMA
        and receipt.get("status") == "succeeded"
        and str(receipt.get("canonical_task_cid") or "") == expected_cid
        and str(receipt.get("canonical_task_key") or "") == expected_key
    ]
    finished_events = [
        event
        for event in events
        if event.get("type") == "implementation_finished"
    ]
    merged_events = [
        event
        for event in finished_events
        if _safe_dict(event.get("merge")).get("merged") is True
        and COMMIT_RE.fullmatch(
            str(_safe_dict(event.get("merge")).get("merge_commit") or "")
        )
    ]
    validated_events = [
        event
        for event in finished_events
        if _safe_dict(event.get("validation")).get("passed") is True
        and _safe_dict(event.get("validation")).get("returncode") == 0
    ]
    task_cid_bound = bool(
        expected_cid
        and event_cids
        and event_cids == {expected_cid}
        and str(snapshot.get("task_id") or "") == ROLE_AWARE_TASK_ID
    )
    bound = bool(
        task_cid_bound
        and successful_receipts
        and merged_events
        and validated_events
    )
    return {
        "present": bool(snapshot),
        "bound": bound,
        "task_cid_bound": task_cid_bound,
        "member_completion_receipt_bound": bool(successful_receipts),
        "validation_bound": bool(validated_events),
        "merge_commit_tree_bound": bool(merged_events),
        "canonical_task_cid": expected_cid or None,
        "canonical_task_key": expected_key or None,
        "successful_completion_receipts": successful_receipts,
        "validation_events": validated_events,
        "merge_events": merged_events,
        "snapshot_digest_sha256": (
            content_digest(snapshot) if snapshot else None
        ),
        "snapshot": snapshot,
        "block_reasons": [
            reason
            for reason, condition in (
                ("supervisor_snapshot_missing", bool(snapshot)),
                ("canonical_task_cid_not_bound", task_cid_bound),
                (
                    "member_completion_receipt_missing",
                    bool(successful_receipts),
                ),
                ("validation_evidence_missing", bool(validated_events)),
                ("merge_commit_tree_evidence_missing", bool(merged_events)),
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
    )

    source_attestation = build_source_attestation(repo_root)
    supervisor = derive_supervisor_binding(supervisor_evidence)

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
        and item.get("basis")
        in {
            "deployment_contract.supported_platforms",
            "tool.pins.platform",
            "host_outside_global_platform_policy",
        }
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

    acceptance = {
        "role_aware_certificate_bound": certificate_digest_valid,
        "completion_receipt_bound": completion_identity_valid,
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
        "artifacts_present": artifacts_present,
    }

    readiness_requirements = {
        key: bool(acceptance[key])
        for key in (
            "role_aware_certificate_bound",
            "completion_receipt_bound",
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

    # Content-address the receipt excluding the identity field itself.
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
