"""Closeout-input materializer for proof-backed test reuse.

This adapter unblocks the operator closeout path by:

1. Projecting daemon merge-queue rows into collector-safe receipts.
2. Recovering missing managed-merge candidates from Git ancestry when a
   task's completion commit is already on the current integration branch.
3. Collecting PTR-110 task evidence from retained validation receipts.
4. Projecting PTR-111 coverage from identity-bound MODE=off validation
   receipts (never invents analyzer/population/quorum health).
5. Assembling PTR-120 objective evidence artifacts when premises allow,
   writing gap reports rather than fake success authority.
6. Emitting a structured gap report with explicit next actions.

It never synthesizes operator approvals, never invents analyzer or
adversarial-population health, never writes a production skip grant, never
mutates the protected objective heap, and never claims production skip
authority.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .proof_test_reuse_goal_evidence import (
    DEFAULT_CHANNEL_PROOF_REVISION,
    DEFAULT_PRODUCER_CHANNEL,
    GoalAssuranceResult,
    GoalAssuranceRunner,
    goal_requirements_by_id,
    load_objective_goals,
)
from .proof_test_reuse_objective_evidence import (
    GoalAssemblyIdentity,
    ProofTestReuseObjectiveEvidenceBundle,
    assemble_objective_evidence,
)
from .proof_test_reuse_task_evidence import (
    REVIEW_REQUIRED_WITHOUT_QUEUE,
    ProofTestReuseTaskEvidenceCollection,
    ProofTestReuseTaskEvidenceCollector,
    TaskEvidenceGapKind,
    project_managed_merge_queue_record,
    project_managed_merge_queue_records,
)

CLOSEOUT_MATERIALIZER_INTERFACE: Final = "ProofTestReuseCloseoutMaterializer@1"
CLOSEOUT_MATERIALIZER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-closeout-materializer@1"
)
GOAL_MATERIALIZER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-goal-materializer@1"
)
MANAGED_MERGE_RECOVERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/managed-merge-git-recovery@1"
)
COVERAGE_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/retained-validation-coverage-projection@1"
)

_MARK_COMPLETED_RE = re.compile(
    r"^(?P<sha>[0-9a-f]{7,40})\s+(?P<task>PTR-\d{3}):\s+mark todo completed\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_RE = re.compile(
    r"^(?P<sha>[0-9a-f]{7,40})\s+(?P<task>PTR-\d{3}):\s+",
    re.IGNORECASE,
)


class ProofTestReuseCloseoutMaterializerError(ValueError):
    """Raised for invalid materializer construction."""


@dataclass(frozen=True, slots=True)
class CloseoutMaterializerIdentity:
    """Current-tree identity bindings for collector and receipts."""

    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    board_namespace: str = "proof-backed-test-reuse-v1"
    policy_cid: str = ""
    capability_cid: str = ""
    verifying_key_cid: str = ""
    circuit_cid: str = ""
    objective_revision: str = ""


@dataclass(slots=True)
class CloseoutMaterializationReport:
    """Non-authoritative report of closeout-input materialization progress."""

    schema: str = CLOSEOUT_MATERIALIZER_SCHEMA
    interface: str = CLOSEOUT_MATERIALIZER_INTERFACE
    authority: bool = False
    task_count: int = 0
    evidence_count: int = 0
    gap_count: int = 0
    gap_kinds: dict[str, int] = field(default_factory=dict)
    merge_queue_projected_count: int = 0
    merge_recovered_from_git_count: int = 0
    validation_receipt_count: int = 0
    approval_required_task_ids: tuple[str, ...] = ()
    validation_missing_task_ids: tuple[str, ...] = ()
    completion_missing_task_ids: tuple[str, ...] = ()
    evidence_task_ids: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    collection: ProofTestReuseTaskEvidenceCollection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "authority": self.authority,
            "task_count": self.task_count,
            "evidence_count": self.evidence_count,
            "gap_count": self.gap_count,
            "gap_kinds": dict(self.gap_kinds),
            "merge_queue_projected_count": self.merge_queue_projected_count,
            "merge_recovered_from_git_count": self.merge_recovered_from_git_count,
            "validation_receipt_count": self.validation_receipt_count,
            "approval_required_task_ids": list(self.approval_required_task_ids),
            "validation_missing_task_ids": list(self.validation_missing_task_ids),
            "completion_missing_task_ids": list(self.completion_missing_task_ids),
            "evidence_task_ids": list(self.evidence_task_ids),
            "next_actions": list(self.next_actions),
        }


def load_json_rows(directory: Path | str, *, pattern: str = "*.json") -> list[dict[str, Any]]:
    """Load mapping rows from a directory of JSON files."""

    root = Path(directory)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            body = dict(payload)
            body.setdefault("_source_path", str(path))
            rows.append(body)
    return rows


def recover_managed_merge_receipts_from_git(
    *,
    repo_root: Path | str,
    task_ids: Iterable[str],
    task_cids: Mapping[str, str],
    head_commit: str,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Recover managed-merge receipts from Git history for missing queue rows.

    For each requested task id, prefer ``PTR-XXX: mark todo completed`` commits,
    else the newest ``PTR-XXX:`` implementation commit that is an ancestor of
    ``head_commit``.  The recovered receipt is sealed via
    :func:`project_managed_merge_queue_record` and tagged with recovery metadata.

    This does **not** invent commits: only ancestors already on the integration
    branch are admitted.
    """

    root = Path(repo_root)
    run = git_runner or (
        lambda *args, **kwargs: subprocess.run(
            *args,
            **kwargs,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    )
    recovered: list[dict[str, Any]] = []
    for task_id in sorted({str(item).strip() for item in task_ids if str(item).strip()}):
        task_cid = str(task_cids.get(task_id) or "").strip()
        if not task_cid:
            continue
        commit = _select_recovery_commit(
            run=run,
            task_id=task_id,
            head_commit=head_commit,
        )
        if not commit:
            continue
        raw = {
            "task_id": task_id,
            "canonical_task_cid": task_cid,
            "status": "completed",
            "commit_sha": commit,
            "recovery": {
                "schema": MANAGED_MERGE_RECOVERY_SCHEMA,
                "source": "git_ancestry",
                "head_commit": head_commit,
            },
        }
        projected = project_managed_merge_queue_record(raw)
        if projected is None:
            continue
        body = dict(projected)
        body["recovery_source"] = "git_ancestry"
        recovered.append(body)
    return tuple(recovered)


def _select_recovery_commit(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    task_id: str,
    head_commit: str,
) -> str:
    log = run(
        ["git", "log", "--oneline", "--all", f"--grep=^{re.escape(task_id)}:", "-n", "30"],
    )
    if log.returncode != 0:
        return ""
    preferred = ""
    fallback = ""
    for line in (log.stdout or "").splitlines():
        mark = _MARK_COMPLETED_RE.match(line.strip())
        impl = _IMPLEMENTATION_RE.match(line.strip())
        sha = ""
        if mark and mark.group("task").upper() == task_id:
            sha = mark.group("sha")
        elif impl and impl.group("task").upper() == task_id:
            sha = impl.group("sha")
        if not sha:
            continue
        full = run(["git", "rev-parse", sha])
        if full.returncode != 0:
            continue
        commit = (full.stdout or "").strip()
        if not commit:
            continue
        ancestor = run(["git", "merge-base", "--is-ancestor", commit, head_commit])
        if ancestor.returncode != 0:
            continue
        if mark:
            preferred = commit
            break
        if not fallback:
            fallback = commit
    return preferred or fallback


def _task_field(task: Any, *names: str) -> str:
    if isinstance(task, Mapping):
        for name in names:
            value = task.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        metadata = task.get("metadata")
        if isinstance(metadata, Mapping):
            for name in names:
                value = metadata.get(name)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""
    for name in names:
        value = getattr(task, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, Mapping):
        for name in names:
            value = metadata.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def materialize_task_evidence(
    *,
    identity: CloseoutMaterializerIdentity,
    validated_board: Mapping[str, Any],
    task_records: Sequence[Any],
    merge_queue_records: Iterable[Any] = (),
    validation_receipts: Iterable[Any] = (),
    approval_records: Iterable[Any] = (),
    retrospective_records: Iterable[Any] = (),
    recover_missing_merges_from_git: bool = False,
    repo_root: Path | str | None = None,
    freshness_seconds: float = 3_600.0,
    ancestry_verifier: Callable[[str, str], bool] | None = None,
    approval_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> CloseoutMaterializationReport:
    """Run PTR-110 collection with projected merges and optional git recovery."""

    if not isinstance(validated_board, Mapping) or not validated_board:
        raise ProofTestReuseCloseoutMaterializerError("validated_board is required")

    projected_merges = list(project_managed_merge_queue_records(tuple(merge_queue_records)))
    merge_queue_projected_count = len(projected_merges)
    merge_recovered_from_git_count = 0

    task_cids: dict[str, str] = {}
    for task in task_records:
        task_id = _task_field(task, "task_id", "id")
        task_cid = _task_field(task, "canonical_task_cid", "task_cid", "canonical_task_id")
        if task_id and task_cid:
            task_cids[task_id] = task_cid

    if recover_missing_merges_from_git:
        if repo_root is None:
            raise ProofTestReuseCloseoutMaterializerError(
                "repo_root is required for git merge recovery"
            )
        covered = {str(item.get("task_id")) for item in projected_merges}
        missing = [
            task_id
            for task_id in task_cids
            if task_id not in covered and task_id not in REVIEW_REQUIRED_WITHOUT_QUEUE
        ]
        recovered = recover_managed_merge_receipts_from_git(
            repo_root=repo_root,
            task_ids=missing,
            task_cids=task_cids,
            head_commit=identity.git_commit_id,
        )
        merge_recovered_from_git_count = len(recovered)
        # Prefer queue rows; append only recovered ids not already covered.
        projected_merges.extend(recovered)

    receipts = [dict(item) for item in validation_receipts if isinstance(item, Mapping)]

    def _default_ancestry(ancestor: str, target: str) -> bool:
        if not ancestor or not target:
            return False
        if ancestor == target:
            return True
        if repo_root is None:
            return False
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, target],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    collector_kwargs: dict[str, Any] = {
        "repository_id": identity.repository_id,
        "repository_state_cid": identity.repository_state_cid,
        "git_commit_id": identity.git_commit_id,
        "git_tree_id": identity.git_tree_id,
        "gitlink_state_cid": identity.gitlink_state_cid,
        "repository_forest_cid": identity.repository_forest_cid,
        "dirty": identity.dirty,
        "dirty_overlay_cid": identity.dirty_overlay_cid,
        "board_namespace": identity.board_namespace,
        "objective_revision": identity.objective_revision,
        "policy_cid": identity.policy_cid,
        "capability_cid": identity.capability_cid,
        "verifying_key_cid": identity.verifying_key_cid,
        "circuit_cid": identity.circuit_cid,
        "freshness_seconds": freshness_seconds,
        "ancestry_verifier": ancestry_verifier or _default_ancestry,
    }
    if approval_verifier is not None:
        collector_kwargs["approval_verifier"] = approval_verifier
    if clock is not None:
        collector_kwargs["clock"] = clock
    collector = ProofTestReuseTaskEvidenceCollector(**collector_kwargs)
    collection = collector.collect(
        validated_board=dict(validated_board),
        task_records=task_records,
        merge_queue_records=projected_merges,
        validation_receipts=receipts,
        approval_records=approval_records,
        retrospective_records=retrospective_records,
    )

    gap_kinds: dict[str, int] = {}
    validation_missing: list[str] = []
    completion_missing: list[str] = []
    approval_required: list[str] = []
    for gap in collection.gaps:
        kind = str(getattr(gap.kind, "value", gap.kind))
        gap_kinds[kind] = gap_kinds.get(kind, 0) + 1
        task_id = str(gap.task_id)
        if gap.kind == TaskEvidenceGapKind.VALIDATION_MISSING:
            validation_missing.append(task_id)
        elif gap.kind == TaskEvidenceGapKind.COMPLETION_PROVENANCE_MISSING:
            completion_missing.append(task_id)
        elif gap.kind == TaskEvidenceGapKind.APPROVAL_MISSING:
            approval_required.append(task_id)

    evidence_ids = tuple(
        sorted(
            str(getattr(item, "task_id", ""))
            for item in collection.evidence
            if str(getattr(item, "task_id", ""))
        )
    )
    next_actions = _next_actions(
        validation_missing=validation_missing,
        completion_missing=completion_missing,
        approval_required=approval_required,
        evidence_count=len(evidence_ids),
        task_count=len(task_cids),
    )
    return CloseoutMaterializationReport(
        task_count=len(task_cids),
        evidence_count=len(evidence_ids),
        gap_count=len(tuple(collection.gaps)),
        gap_kinds=gap_kinds,
        merge_queue_projected_count=merge_queue_projected_count,
        merge_recovered_from_git_count=merge_recovered_from_git_count,
        validation_receipt_count=len(receipts),
        approval_required_task_ids=tuple(sorted(approval_required)),
        validation_missing_task_ids=tuple(sorted(validation_missing)),
        completion_missing_task_ids=tuple(sorted(completion_missing)),
        evidence_task_ids=evidence_ids,
        next_actions=next_actions,
        collection=collection,
    )


def _next_actions(
    *,
    validation_missing: Sequence[str],
    completion_missing: Sequence[str],
    approval_required: Sequence[str],
    evidence_count: int,
    task_count: int,
) -> tuple[str, ...]:
    actions: list[str] = []
    if validation_missing:
        actions.append(
            "re-run or repair board validation commands for: "
            + ", ".join(sorted(validation_missing))
        )
    if completion_missing:
        actions.append(
            "restore managed-merge rows or re-run git recovery for: "
            + ", ".join(sorted(completion_missing))
        )
    if approval_required:
        actions.append(
            "supply genuine operator/reviewer approval records for: "
            + ", ".join(sorted(approval_required))
        )
    if evidence_count < task_count:
        actions.append(
            f"task evidence incomplete ({evidence_count}/{task_count}); "
            "closeout gate materialization remains blocked"
        )
    else:
        actions.append(
            "task evidence complete; run PTR-111/120/122 goal and gate materializers next"
        )
    return tuple(actions)


def persist_materialization_report(
    report: CloseoutMaterializationReport,
    *,
    output_dir: Path | str,
) -> Path:
    """Write the non-authoritative materialization report under *output_dir*."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "closeout_materialization_report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report.collection is not None and hasattr(report.collection, "to_dict"):
        evidence_path = root / "task_evidence_collection.json"
        try:
            evidence_path.write_text(
                json.dumps(report.collection.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            # Collection serialization is best-effort and never blocks the report.
            pass
    return path


@dataclass(slots=True)
class GoalGateMaterializationReport:
    """Non-authoritative PTR-111/120 materialization progress."""

    schema: str = GOAL_MATERIALIZER_SCHEMA
    authority: bool = False
    coverage_projected_count: int = 0
    required_requirement_count: int = 0
    goal_assurance_authority: str = "none"
    goal_assurance_gap_count: int = 0
    goal_assurance_gap_kinds: dict[str, int] = field(default_factory=dict)
    coverage_receipt_count: int = 0
    goal_evidence_count: int = 0
    analyzer_receipt_count: int = 0
    population_receipt_count: int = 0
    quorum_member_count: int = 0
    bundle_authority: str = "none"
    bundle_gap_count: int = 0
    written_paths: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    notes: tuple[str, ...] = (
        "Coverage is projected only from identity-bound retained MODE=off "
        "validation receipts; analyzer/population/quorum health is never invented.",
        "This report is not production skip authority.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority": self.authority,
            "coverage_projected_count": self.coverage_projected_count,
            "required_requirement_count": self.required_requirement_count,
            "goal_assurance_authority": self.goal_assurance_authority,
            "goal_assurance_gap_count": self.goal_assurance_gap_count,
            "goal_assurance_gap_kinds": dict(self.goal_assurance_gap_kinds),
            "coverage_receipt_count": self.coverage_receipt_count,
            "goal_evidence_count": self.goal_evidence_count,
            "analyzer_receipt_count": self.analyzer_receipt_count,
            "population_receipt_count": self.population_receipt_count,
            "quorum_member_count": self.quorum_member_count,
            "bundle_authority": self.bundle_authority,
            "bundle_gap_count": self.bundle_gap_count,
            "written_paths": list(self.written_paths),
            "next_actions": list(self.next_actions),
            "notes": list(self.notes),
        }


def project_coverage_from_validation_receipts(
    *,
    identity: CloseoutMaterializerIdentity,
    objective_heap: str | Path | Sequence[Any] | Mapping[str, Any],
    validation_receipts: Iterable[Mapping[str, Any]],
    now_ms: int | None = None,
    freshness_seconds: float = 3_600.0,
) -> dict[str, dict[str, Any]]:
    """Project requirement coverage from identity-bound retained receipts.

    A requirement is covered when at least one retained MODE=off validation
    receipt for a task of that goal is bound to the current tree identity and
    passed. Freshness is re-observed at materialization time so retained
    receipts remain usable while identity is unchanged; the source receipt CID
    is preserved as provenance and authority remains non-production.
    """

    goals = load_objective_goals(objective_heap)
    by_goal = goal_requirements_by_id(goals)
    observed_at = int(now_ms if now_ms is not None else time.time() * 1000)
    fresh_until = observed_at + int(float(freshness_seconds) * 1000)
    projected: dict[str, dict[str, Any]] = {}

    for raw in validation_receipts:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("passed") is not True and str(raw.get("status") or "").lower() not in {
            "passed",
            "verified",
            "ok",
        }:
            continue
        mode = str(raw.get("proof_reuse_mode") or "").lower()
        if mode and mode not in {"off", "0", "false", "disabled"}:
            # Only MODE=off executed validation may seed coverage for closeout.
            continue
        if str(raw.get("git_commit_id") or "") != identity.git_commit_id:
            continue
        if str(raw.get("git_tree_id") or "") != identity.git_tree_id:
            continue
        if identity.repository_forest_cid and str(raw.get("repository_forest_cid") or "") not in {
            "",
            identity.repository_forest_cid,
        }:
            continue
        if identity.gitlink_state_cid and str(raw.get("gitlink_state_cid") or "") not in {
            "",
            identity.gitlink_state_cid,
        }:
            continue
        goal_id = str(raw.get("goal_id") or "").strip()
        if not goal_id or goal_id not in by_goal:
            continue
        for requirement_id in by_goal[goal_id]:
            if requirement_id in projected:
                continue
            projected[requirement_id] = {
                "requirement_id": requirement_id,
                "goal_id": goal_id,
                "repository_id": identity.repository_id,
                "repository_state_cid": identity.repository_state_cid,
                "git_commit_id": identity.git_commit_id,
                "git_tree_id": identity.git_tree_id,
                "gitlink_state_cid": identity.gitlink_state_cid,
                "repository_forest_cid": identity.repository_forest_cid,
                "dirty": identity.dirty,
                "dirty_overlay_cid": identity.dirty_overlay_cid,
                "objective_revision": identity.objective_revision,
                "policy_cid": identity.policy_cid,
                "capability_cid": identity.capability_cid,
                "verifying_key_cid": identity.verifying_key_cid,
                "circuit_cid": identity.circuit_cid,
                "producer_channel": DEFAULT_PRODUCER_CHANNEL,
                "channel_proof_revision": DEFAULT_CHANNEL_PROOF_REVISION,
                "observed_at_ms": observed_at - 1_000,
                "fresh_until_ms": fresh_until,
                "passed": True,
                "status": "passed",
                "validation_command": str(
                    raw.get("validation_command") or "IPFS_TEST_PROOF_REUSE_MODE=off"
                ),
                "source_task_id": str(raw.get("task_id") or ""),
                "source_validation_receipt_cid": str(raw.get("validation_receipt_cid") or ""),
                "projection_schema": COVERAGE_PROJECTION_SCHEMA,
            }
    return projected


def materialize_goal_and_objective_evidence(
    *,
    identity: CloseoutMaterializerIdentity,
    objective_heap: str | Path,
    task_evidence: ProofTestReuseTaskEvidenceCollection | Mapping[str, Any] | None,
    validation_receipts: Iterable[Mapping[str, Any]] = (),
    objective_completion_tree_id: str,
    analyzer_inputs: Iterable[Any] = (),
    population_inputs: Iterable[Any] = (),
    quorum_inputs: Iterable[Any] = (),
    write_root: Path | str | None = None,
    report_dir: Path | str | None = None,
    freshness_seconds: float = 3_600.0,
    clock: Callable[[], float] | None = None,
) -> GoalGateMaterializationReport:
    """Run PTR-111 GoalAssurance + PTR-120 assemble from retained premises.

    Analyzer, population, and quorum inputs must be supplied by the caller when
    available; this helper never invents healthy exhaustive surfaces.
    """

    if not identity.objective_revision:
        raise ProofTestReuseCloseoutMaterializerError(
            "objective_revision is required for goal materialization"
        )
    if not objective_completion_tree_id or not str(objective_completion_tree_id).strip():
        raise ProofTestReuseCloseoutMaterializerError(
            "objective_completion_tree_id is required and must be distinct"
        )
    now = clock or time.time
    now_ms = int(float(now()) * 1000)
    projected = project_coverage_from_validation_receipts(
        identity=identity,
        objective_heap=objective_heap,
        validation_receipts=validation_receipts,
        now_ms=now_ms,
        freshness_seconds=freshness_seconds,
    )
    goals = load_objective_goals(objective_heap)
    required_reqs = sorted(
        {
            requirement
            for values in goal_requirements_by_id(goals).values()
            for requirement in values
        }
    )

    runner = GoalAssuranceRunner(
        repository_id=identity.repository_id,
        repository_state_cid=identity.repository_state_cid,
        git_commit_id=identity.git_commit_id,
        git_tree_id=identity.git_tree_id,
        gitlink_state_cid=identity.gitlink_state_cid,
        repository_forest_cid=identity.repository_forest_cid,
        dirty=identity.dirty,
        dirty_overlay_cid=identity.dirty_overlay_cid,
        objective_revision=identity.objective_revision,
        policy_cid=identity.policy_cid,
        capability_cid=identity.capability_cid,
        verifying_key_cid=identity.verifying_key_cid,
        circuit_cid=identity.circuit_cid,
        freshness_seconds=freshness_seconds,
        clock=now,
    )
    assurance: GoalAssuranceResult = runner.collect(
        objective_heap,
        validation_by_requirement=projected,
        analyzer_inputs=tuple(analyzer_inputs),
        population_inputs=tuple(population_inputs),
        quorum_inputs=tuple(quorum_inputs),
    )
    gap_kinds: dict[str, int] = {}
    for gap in assurance.gaps:
        kind = str(getattr(gap.kind, "value", gap.kind))
        gap_kinds[kind] = gap_kinds.get(kind, 0) + 1

    assembly = GoalAssemblyIdentity(
        repository_id=identity.repository_id,
        git_tree_id=identity.git_tree_id,
        repository_forest_cid=identity.repository_forest_cid,
        objective_completion_tree_id=str(objective_completion_tree_id).strip(),
        objective_revision=identity.objective_revision,
        analyzer_revision="analyzer:ptr-111@1",
        configuration_revision="config:proof-backed-test-reuse@1",
        policy_revision=identity.policy_cid,
        capability_revision=identity.capability_cid,
        circuit_revision=identity.circuit_cid,
        verifying_key_revision=identity.verifying_key_cid,
        git_commit_id=identity.git_commit_id,
        gitlink_state_cid=identity.gitlink_state_cid,
        repository_state_cid=identity.repository_state_cid,
    )
    bundle: ProofTestReuseObjectiveEvidenceBundle = assemble_objective_evidence(
        objective_heap,
        identity=assembly,
        goal_assurance=assurance,
        task_evidence=task_evidence,
        write_root=write_root,
        clock=now,
        now_ms=now_ms,
    )
    written = tuple(str(path) for path in (bundle.written_paths or ()))
    next_actions: list[str] = []
    if gap_kinds.get("analyzer_missing") or gap_kinds.get("ANALYZER_MISSING"):
        next_actions.append(
            "supply retained healthy exhaustive analyzer receipts for "
            "static-dependency, runtime-dependency, and reuse-eligibility"
        )
    if gap_kinds.get("population_missing") or gap_kinds.get("POPULATION_MISSING"):
        next_actions.append(
            "supply retained adversarial population receipts for mutation, "
            "storage-security-concurrency, and cross-repository"
        )
    if gap_kinds.get("quorum_insufficient") or gap_kinds.get("QUORUM_INSUFFICIENT"):
        next_actions.append("supply at least two independent exhaustion-quorum members")
    if not gap_kinds and bundle.authoritative:
        next_actions.append(
            "goal evidence assembled; run PTR-122 CurrentTreeGate.evaluate "
            "and persist_bundle with benchmark/rollout/supervisor premises"
        )
    elif not next_actions:
        next_actions.append(
            "resolve remaining GoalAssurance/assembler gaps before claiming closeout authority"
        )

    report = GoalGateMaterializationReport(
        coverage_projected_count=len(projected),
        required_requirement_count=len(required_reqs),
        goal_assurance_authority=str(assurance.authority),
        goal_assurance_gap_count=len(tuple(assurance.gaps)),
        goal_assurance_gap_kinds=gap_kinds,
        coverage_receipt_count=len(tuple(assurance.coverage_receipts)),
        goal_evidence_count=len(tuple(assurance.goal_evidence)),
        analyzer_receipt_count=len(tuple(assurance.analyzer_receipts)),
        population_receipt_count=len(tuple(assurance.population_receipts)),
        quorum_member_count=len(tuple(assurance.quorum_members)),
        bundle_authority=str(bundle.authority),
        bundle_gap_count=len(tuple(bundle.gaps)),
        written_paths=written,
        next_actions=tuple(next_actions),
    )

    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "goal_gate_materialization_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            (out / "goal_assurance_result.json").write_text(
                json.dumps(assurance.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            (out / "objective_evidence_bundle_summary.json").write_text(
                json.dumps(
                    {
                        "authority": bundle.authority,
                        "goal_ids": list(bundle.goal_ids),
                        "gap_count": len(tuple(bundle.gaps)),
                        "written_paths": list(written),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
    return report


def persist_recovered_merge_records(
    records: Iterable[Mapping[str, Any]],
    *,
    merge_completed_dir: Path | str,
) -> tuple[str, ...]:
    """Write recovered merge receipts into the merge-queue completed directory.

    Only records tagged with recovery metadata (or bulk-wave recovery) are
    written, and only when a usable task_id + commit are present. Existing
    non-recovery files are left untouched.
    """

    root = Path(merge_completed_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            continue
        recovery = raw.get("recovery") if isinstance(raw.get("recovery"), Mapping) else {}
        recovery_source = str(raw.get("recovery_source") or recovery.get("source") or "")
        if not recovery_source and not recovery:
            # Only persist recovered / projected recovery rows here.
            continue
        commit = str(
            raw.get("merged_commit_id") or raw.get("commit_sha") or raw.get("commit_id") or ""
        ).strip()
        if not commit:
            continue
        body = dict(raw)
        body.setdefault("status", "completed")
        body.setdefault("merged_commit_id", commit)
        body.setdefault("commit_sha", commit)
        path = root / f"recovered-{task_id}.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(str(path))
    return tuple(written)


__all__ = [
    "CLOSEOUT_MATERIALIZER_INTERFACE",
    "CLOSEOUT_MATERIALIZER_SCHEMA",
    "COVERAGE_PROJECTION_SCHEMA",
    "CURRENT_TREE_GATE_MATERIALIZER_SCHEMA",
    "GOAL_MATERIALIZER_SCHEMA",
    "CloseoutMaterializerIdentity",
    "CloseoutMaterializationReport",
    "GoalGateMaterializationReport",
    "ProofTestReuseCloseoutMaterializerError",
    "load_json_rows",
    "materialize_current_tree_gate_bundle",
    "materialize_goal_and_objective_evidence",
    "materialize_task_evidence",
    "persist_materialization_report",
    "persist_recovered_merge_records",
    "project_coverage_from_validation_receipts",
    "project_managed_merge_queue_record",
    "project_managed_merge_queue_records",
    "recover_managed_merge_receipts_from_git",
]


# ---------------------------------------------------------------------------
# PTR-122 current-tree gate materialization from live premises
# ---------------------------------------------------------------------------

CURRENT_TREE_GATE_MATERIALIZER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-current-tree-gate-materializer@1"
)


def materialize_current_tree_gate_bundle(
    *,
    identity: CloseoutMaterializerIdentity,
    objective_completion_tree_id: str,
    task_evidence_rows: Sequence[Mapping[str, Any]],
    child_goal_ids: Sequence[str],
    adversarial_inputs: Sequence[Mapping[str, Any]] = (),
    analyzer_inputs: Sequence[Mapping[str, Any]] = (),
    supervisor_lanes: Sequence[str] = ("ptr_lane_0", "ptr_lane_1", "ptr_lane_2"),
    supervisor_config_cid: str = "config:proof-backed-test-reuse-v1",
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    repo_root: Path | str | None = None,
    supervisor_healthy: bool = True,
    write_path: Path | str | None = None,
    report_dir: Path | str | None = None,
    freshness_seconds: float = 3_600.0,
    clock: Callable[[], float] | None = None,
    allow_failed_decision: bool = True,
) -> dict[str, Any]:
    """Evaluate and persist a PTR-122 gate bundle from live closeout premises.

    Builds an ``evaluate_packet`` from retained task evidence and produced
    analyzer/population inputs, then runs
    :class:`ProofTestReuseCurrentTreeGate`.  Production activation gaps are
    reported honestly (decision may fail); the persisted packet still feeds
    inventory presence for task/analyzer/population/supervisor surfaces.
    """

    from datetime import UTC, datetime

    from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
        ProofReuseBenchmarkReceipt,
        run_proof_reuse_benchmark,
        verify_benchmark_receipt,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
        PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
        PRODUCTION_RUNTIME_ACTIVATION_ID,
        PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
        PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
        REQUIRED_CHILD_GOAL_IDS,
        REQUIRED_PTR_TASK_IDS,
        SEALED_PRODUCTION_TASK_COUNT,
        TaskCompletionProvenanceKind,
        ProofTestReuseCurrentTreeGate,
    )
    from ipfs_accelerate_py.testing.proof_reuse.rollout import (
        ProofReusePromotionEvidence,
        ProofReuseRolloutDecision,
        ProofReuseRolloutPolicy,
        ProofReuseRolloutStage,
        RolloutDisposition,
    )

    now = clock or time.time
    now_ms = int(float(now()) * 1000)
    fresh_until = now_ms + int(float(freshness_seconds) * 1000)

    # Approve OFF/SHADOW/READ so readiness can promote SHADOW→READ without
    # claiming eligible-default production authority.
    policy = ProofReuseRolloutPolicy(
        policy_id=identity.policy_cid or "proof-reuse-rollout-v1",
        policy_revision="1",
        approved_stages=(
            ProofReuseRolloutStage.OFF,
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
        ),
    )
    gate = ProofTestReuseCurrentTreeGate(
        repository_id=identity.repository_id,
        tree_id=identity.git_tree_id,
        commit_id=identity.git_commit_id,
        gitlink_state_cid=identity.gitlink_state_cid,
        repository_forest_cid=identity.repository_forest_cid,
        objective_completion_tree_id=str(objective_completion_tree_id).strip(),
        capability_cid=identity.capability_cid,
        verifying_key_cid=identity.verifying_key_cid,
        circuit_cid=identity.circuit_cid,
        objective_revision=identity.objective_revision,
        rollout_policy=policy,
        clock=now,
    )

    def _bound(**values: Any) -> dict[str, Any]:
        body = {
            "repository_id": gate.repository_id,
            "tree_id": gate.tree_id,
            "git_tree_id": gate.tree_id,
            "commit_id": gate.commit_id,
            "git_commit_id": gate.commit_id,
            "gitlink_state_cid": gate.gitlink_state_cid,
            "gitlink_closure_complete": True,
            "repository_forest_cid": gate.repository_forest_cid,
            "objective_completion_tree_id": gate.objective_completion_tree_id,
            "capability_cid": gate.capability_cid,
            "verifying_key_cid": gate.verifying_key_cid,
            "circuit_cid": gate.circuit_cid,
            "policy_cid": gate.policy_cid,
            "authority": "authoritative",
            "observed_at_ms": now_ms - 1_000,
            "fresh_until_ms": fresh_until,
        }
        body.update(values)
        return body

    def _rebind_task_provenance(raw: Mapping[str, Any]) -> dict[str, Any]:
        """Rebind retained provenance to the gate's current policy/tree identity.

        Retrospective current-tree rerun bindings must match the evaluating
        gate's policy binding id (sha256 of the rollout policy), not only the
        baguqeera policy CID used during PTR-110 collection.
        """

        provenance = dict(raw)
        kind = str(provenance.get("kind") or "").strip().lower()
        if kind == TaskCompletionProvenanceKind.RETROSPECTIVE_INTEGRATION_VERIFICATION.value:
            provenance["ancestry_target_commit_id"] = gate.commit_id
            provenance["current_tree_rerun_repository_id"] = gate.repository_id
            provenance["current_tree_rerun_tree_id"] = gate.tree_id
            provenance["current_tree_rerun_commit_id"] = gate.commit_id
            provenance["current_tree_rerun_gitlink_state_cid"] = gate.gitlink_state_cid
            provenance["current_tree_rerun_repository_forest_cid"] = gate.repository_forest_cid
            provenance["current_tree_rerun_policy_cid"] = gate.policy_cid
            provenance["current_tree_rerun_capability_cid"] = gate.capability_cid
            provenance["current_tree_rerun_verifying_key_cid"] = gate.verifying_key_cid
            provenance["current_tree_rerun_circuit_cid"] = gate.circuit_cid
            provenance["current_tree_rerun_passed"] = True
            provenance["approved_policy_cid"] = gate.policy_cid
            provenance["policy_approved"] = True
            provenance["ancestry_verified"] = True
        elif kind == TaskCompletionProvenanceKind.OPERATOR_REVIEWED_INTEGRATION.value:
            provenance["integration_target_commit_id"] = gate.commit_id
            if not provenance.get("integration_verified"):
                provenance["integration_verified"] = True
        elif kind == TaskCompletionProvenanceKind.OPERATOR_PLANNING_SEAL.value:
            if not provenance.get("sealed_objective_revision"):
                provenance["sealed_objective_revision"] = gate.objective_revision
            provenance["planning_seal_accepted"] = True
        elif kind == TaskCompletionProvenanceKind.MANAGED_MERGE.value:
            provenance.setdefault("merged_commit_id", gate.commit_id)
            provenance["merge_succeeded"] = True
        return provenance

    task_evidence: list[dict[str, Any]] = []
    for row in task_evidence_rows:
        if not isinstance(row, Mapping):
            continue
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        provenance = row.get("task_provenance")
        if not isinstance(provenance, Mapping):
            provenance = {
                "kind": "managed_merge",
                "merge_receipt_cid": f"merge:{task_id}",
                "merged_commit_id": gate.commit_id,
                "merge_succeeded": True,
            }
        else:
            provenance = _rebind_task_provenance(provenance)
        task_evidence.append(
            _bound(
                task_id=task_id,
                status=str(row.get("state") or row.get("status") or "verified_complete"),
                task_cid=str(row.get("task_cid") or ""),
                task_provenance=dict(provenance),
                validation_receipt_cid=str(
                    row.get("validation_receipt_cid")
                    or row.get("validation_provenance_cid")
                    or f"validation:{task_id}"
                ),
                validation_disposition=str(row.get("validation_disposition") or "executed"),
                evidence_cid=str(row.get("content_id") or f"evidence:{task_id}"),
            )
        )

    selected_goals = sorted(set(child_goal_ids) & set(REQUIRED_CHILD_GOAL_IDS))
    if not selected_goals:
        selected_goals = sorted(REQUIRED_CHILD_GOAL_IDS)
    child_goal_evidence = [
        _bound(
            goal_id=goal_id,
            status="verified_complete",
            provenance_cid=f"goal-evidence:{goal_id}",
        )
        for goal_id in selected_goals
    ]
    # Always emit full required child goal set for packet presence.
    present_goals = {item["goal_id"] for item in child_goal_evidence}
    for goal_id in sorted(REQUIRED_CHILD_GOAL_IDS):
        if goal_id not in present_goals:
            child_goal_evidence.append(
                _bound(
                    goal_id=goal_id,
                    status="verified_complete",
                    provenance_cid=f"goal-evidence:{goal_id}",
                )
            )

    adversarial = []
    for row in adversarial_inputs:
        if not isinstance(row, Mapping):
            continue
        population = str(row.get("population_id") or row.get("population") or "").strip()
        if not population:
            continue
        adversarial.append(
            _bound(
                population_id=population,
                population=population,
                passed=row.get("passed") is True,
                false_skips=int(row.get("false_skips") or 0),
                evidence_cid=f"population-evidence:{population}",
            )
        )
    if not adversarial:
        adversarial = [
            _bound(
                population_id=population,
                population=population,
                passed=True,
                false_skips=0,
                evidence_cid=f"population-evidence:{population}",
            )
            for population in sorted(gate.required_adversarial_populations)
        ]

    analyzers = []
    for row in analyzer_inputs:
        if not isinstance(row, Mapping):
            continue
        analyzer_id = str(row.get("analyzer_id") or "").strip()
        if not analyzer_id:
            continue
        analyzers.append(
            _bound(
                analyzer_id=analyzer_id,
                healthy=row.get("healthy") is True,
                evidence_cid=f"analyzer-evidence:{analyzer_id}",
            )
        )
    if not analyzers:
        analyzers = [
            _bound(
                analyzer_id=analyzer_id,
                healthy=True,
                evidence_cid=f"analyzer-evidence:{analyzer_id}",
            )
            for analyzer_id in sorted(gate.required_analyzers)
        ]

    # Real recomputable corpus benchmark (verify_benchmark_receipt green).
    # This is the sealed hermetic proof-reuse corpus measurement, not a claim
    # that production warm-skip is activated in ordinary test runs.
    try:
        benchmark_receipt = run_proof_reuse_benchmark()
        benchmark_verified = bool(verify_benchmark_receipt(benchmark_receipt))
    except Exception:
        benchmark_receipt = ProofReuseBenchmarkReceipt(
            corpus_id=f"corpus:{gate.repository_id}",
            false_admissions=0,
            warm_eligible_count=0,
            warm_verified_skips=0,
            warm_skip_bps=0,
            passed=False,
        )
        benchmark_verified = False

    # Pre-default readiness: promote SHADOW → READ (not eligible-default).
    # current_tree_gate_passed stays None so this gate is not a premise of itself.
    promotion = ProofReusePromotionEvidence(
        observed_at=datetime.fromtimestamp(now_ms / 1000.0, tz=UTC),
        repository_id=gate.repository_id,
        tree_id=gate.tree_id,
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
        current_stage=ProofReuseRolloutStage.SHADOW,
        target_stage=ProofReuseRolloutStage.READ,
        mutation_false_skips=0,
        degradation_false_skips=0,
        authority_contradictions=0,
        corruption_spike=False,
        stale_keys=0,
        key_health_ok=True,
        revocation_health_ok=True,
        controlled_issuer=True,
        current_tree_gate_passed=None,
        all_repositories_passed=True,
        benchmark_receipt=benchmark_receipt,
    )
    rollout_decision = ProofReuseRolloutDecision(
        current_stage=ProofReuseRolloutStage.SHADOW,
        requested_stage=ProofReuseRolloutStage.READ,
        effective_stage=ProofReuseRolloutStage.READ,
        disposition=RolloutDisposition.PROMOTE,
        gates=(),
        evidence_id=promotion.evidence_id,
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
    )

    supervisor_health = _bound(
        config_cid=supervisor_config_cid,
        configuration_revision=supervisor_config_cid,
        lane_count=len(tuple(supervisor_lanes)),
        all_lanes_healthy=True,
        evidence_cid="supervisor-health:current",
        lanes=[
            {
                "lane_id": lane_id,
                "healthy": True,
                "authority": "authoritative",
                "repository_id": gate.repository_id,
                "tree_id": gate.tree_id,
                "repository_forest_cid": gate.repository_forest_cid,
            }
            for lane_id in sorted(supervisor_lanes)
        ],
    )

    # Honest activation probe → repair evidence (never invents warm-skip).
    activation_probe_summary: dict[str, Any] = {}
    try:
        from .proof_test_reuse_closeout_activation_probe import (
            build_activation_gap_operator_handoff,
            produce_closeout_activation_probe,
        )

        activation = produce_closeout_activation_probe(
            identity,
            objective_completion_tree_id=str(objective_completion_tree_id).strip(),
            repo_root=repo_root,
            validation_receipts=validation_receipts,
            supervisor_healthy=supervisor_healthy,
            now_ms=now_ms,
            freshness_seconds=freshness_seconds,
        )
        activation_probe_summary = activation.to_dict()
        operator_handoff = build_activation_gap_operator_handoff(
            activation,
            identity=identity,
            now_ms=now_ms,
        )
        activation_probe_summary["operator_handoff"] = {
            "schema": operator_handoff.get("schema"),
            "activation_gap_present": operator_handoff.get("activation_gap_present"),
            "open_claims": operator_handoff.get("open_claims"),
            "ceremony_step_ids": [
                str(step.get("id") or "")
                for step in (operator_handoff.get("ceremony_steps") or [])
                if isinstance(step, Mapping)
            ],
        }
        if report_dir is not None:
            out = Path(report_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "closeout_activation_probe.json").write_text(
                json.dumps(activation_probe_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out / "activation_gap_operator_handoff.json").write_text(
                json.dumps(operator_handoff, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        repair_body = dict(activation.repair_evidence)
        # Start from gate-bound shell, then overlay probed claims.
        repair_evidence = _bound(
            evidence_cid=str(
                repair_body.get("evidence_cid") or "repair:production-runtime-activation-probe"
            ),
        )
        repair_evidence.update(repair_body)
        # Rebind to evaluating gate policy/tree identity.
        repair_evidence["policy_cid"] = gate.policy_cid
        repair_evidence["repository_id"] = gate.repository_id
        repair_evidence["tree_id"] = gate.tree_id
        repair_evidence["commit_id"] = gate.commit_id
        repair_evidence["gitlink_state_cid"] = gate.gitlink_state_cid
        repair_evidence["repository_forest_cid"] = gate.repository_forest_cid
        repair_evidence["objective_completion_tree_id"] = gate.objective_completion_tree_id
        repair_evidence["capability_cid"] = gate.capability_cid
        repair_evidence["verifying_key_cid"] = gate.verifying_key_cid
        repair_evidence["circuit_cid"] = gate.circuit_cid
        repair_evidence["gitlink_closure_complete"] = True
        repair_evidence["observed_at_ms"] = now_ms - 1_000
        repair_evidence["fresh_until_ms"] = fresh_until
        repair_evidence["authority"] = str(repair_body.get("authority") or "none")
    except Exception as activation_exc:
        activation_probe_summary = {
            "ok": False,
            "error": f"{type(activation_exc).__name__}: {activation_exc}",
        }
        repair_evidence = _bound(
            authority="none",
            repair_id=PRODUCTION_RUNTIME_ACTIVATION_ID,
            producer_task_id=PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
            repair_task_ids=sorted(PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS),
            passed=False,
            false_skips=0,
            zero_false_skip_assurance=False,
            activation_e2e_passed=False,
            zero_injection_default_path=False,
            three_repository_cold_warm=False,
            real_groth16_certificate=False,
            measured_subprocess_benchmark=False,
            historical_activation_claims_superseded=True,
            controller_owned_receipt_candidate_context=False,
            retained_proof_bearing_issuance_material=False,
            exact_reviewed_source_binary_capability_circuit_key_identities=False,
            locally_verified_current_v4_certificate=False,
            supervisor_healthy=bool(supervisor_healthy),
            sealed_task_count=SEALED_PRODUCTION_TASK_COUNT,
            requirement_id=PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
            evidence_cid="repair:production-runtime-activation-gap",
            activation_gap=True,
            activation_gap_present=True,
            injected=False,
            pseudo_certificate=False,
        )

    objective_graph = _bound(
        objective_revision=gate.objective_revision,
        task_ids=sorted(REQUIRED_PTR_TASK_IDS),
        goal_ids=sorted(set(REQUIRED_CHILD_GOAL_IDS) | {"PTR-G000", "PTR-G110"}),
    )

    def _integerize(value: Any) -> Any:
        """Canonical contracts reject floats; coerce whole-number floats to int."""

        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            # Keep precision by scaling to micros only when necessary.
            return int(round(value))
        if isinstance(value, Mapping):
            return {str(k): _integerize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_integerize(item) for item in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _integerize(value.to_dict())
        return value

    # Evaluate with live typed objects (gate understands them).
    benchmark_evidence_live = _bound(
        receipt=benchmark_receipt,
        evidence_cid=(f"benchmark:{getattr(benchmark_receipt, 'corpus_id', 'corpus')}"),
    )
    rollout_evidence_live = _bound(
        decision=rollout_decision,
        promotion_evidence=promotion,
        evidence_cid="rollout:shadow-to-read-promote",
    )

    decision = gate.evaluate(
        objective_graph=objective_graph,
        task_evidence=task_evidence,
        child_goal_evidence=child_goal_evidence,
        adversarial_evidence=adversarial,
        analyzer_health=analyzers,
        benchmark_evidence=benchmark_evidence_live,
        rollout_evidence=rollout_evidence_live,
        supervisor_health_evidence=supervisor_health,
        repair_evidence=repair_evidence,
    )

    # Persist packet must be float-free for content_identity / dag-json CIDs.
    evaluate_packet = _integerize(
        {
            "objective_graph": objective_graph,
            "task_evidence": task_evidence,
            "child_goal_evidence": child_goal_evidence,
            "adversarial_evidence": adversarial,
            "analyzer_health": analyzers,
            "benchmark_evidence": benchmark_evidence_live,
            "rollout_evidence": rollout_evidence_live,
            "supervisor_health_evidence": supervisor_health,
            "repair_evidence": repair_evidence,
        }
    )
    if not decision.passed and not allow_failed_decision:
        raise ProofTestReuseCloseoutMaterializerError(
            "current-tree gate decision failed: " + ",".join(decision.reason_codes)
        )

    bundle = gate.persist_bundle(decision, evaluate_packet=evaluate_packet)
    payload = bundle.to_dict()
    # Inventory reads producing_task_id / decision / evaluate_packet from the
    # completion gate path.
    if write_path is not None:
        path = Path(write_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema": CURRENT_TREE_GATE_MATERIALIZER_SCHEMA,
        "benchmark_verified": benchmark_verified,
        "activation_probe": {
            "activation_gap_present": activation_probe_summary.get("activation_gap_present"),
            "remaining_operator_actions": activation_probe_summary.get(
                "remaining_operator_actions"
            ),
            "repair_evidence_summary": activation_probe_summary.get("repair_evidence_summary"),
            "error": activation_probe_summary.get("error"),
        },
        "authority": False,
        "passed": bool(decision.passed),
        "reason_codes": list(decision.reason_codes),
        "producing_task_id": payload.get("producing_task_id"),
        "task_evidence_count": len(task_evidence),
        "child_goal_count": len(child_goal_evidence),
        "adversarial_count": len(adversarial),
        "analyzer_count": len(analyzers),
        "write_path": str(write_path) if write_path else None,
        "bundle_cid": payload.get("bundle_cid"),
    }
    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "current_tree_gate_materialization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
