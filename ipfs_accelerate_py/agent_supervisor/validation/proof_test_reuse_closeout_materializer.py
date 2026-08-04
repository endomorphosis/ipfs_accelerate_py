"""Closeout-input materializer for proof-backed test reuse.

This adapter unblocks the operator closeout path by:

1. Projecting daemon merge-queue rows into collector-safe receipts.
2. Recovering missing managed-merge candidates from Git ancestry when a
   task's completion commit is already on the current integration branch.
3. Collecting PTR-110 task evidence from retained validation receipts.
4. Emitting a structured gap report with explicit next actions.

It never synthesizes operator approvals, never writes the protected
objective heap, and never claims production skip authority.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

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
MANAGED_MERGE_RECOVERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/managed-merge-git-recovery@1"
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
    recover_missing_merges_from_git: bool = False,
    repo_root: Path | str | None = None,
    freshness_seconds: float = 3_600.0,
    ancestry_verifier: Callable[[str, str], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> CloseoutMaterializationReport:
    """Run PTR-110 collection with projected merges and optional git recovery."""

    if not isinstance(validated_board, Mapping) or not validated_board:
        raise ProofTestReuseCloseoutMaterializerError("validated_board is required")

    projected_merges = list(
        project_managed_merge_queue_records(tuple(merge_queue_records))
    )
    merge_queue_projected_count = len(projected_merges)
    merge_recovered_from_git_count = 0

    task_cids: dict[str, str] = {}
    for task in task_records:
        task_id = _task_field(task, "task_id", "id")
        task_cid = _task_field(
            task, "canonical_task_cid", "task_cid", "canonical_task_id"
        )
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
    if clock is not None:
        collector_kwargs["clock"] = clock
    collector = ProofTestReuseTaskEvidenceCollector(**collector_kwargs)
    collection = collector.collect(
        validated_board=dict(validated_board),
        task_records=task_records,
        merge_queue_records=projected_merges,
        validation_receipts=receipts,
        approval_records=approval_records,
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
                json.dumps(report.collection.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            # Collection serialization is best-effort and never blocks the report.
            pass
    return path


__all__ = [
    "CLOSEOUT_MATERIALIZER_INTERFACE",
    "CLOSEOUT_MATERIALIZER_SCHEMA",
    "CloseoutMaterializerIdentity",
    "CloseoutMaterializationReport",
    "ProofTestReuseCloseoutMaterializerError",
    "load_json_rows",
    "materialize_task_evidence",
    "persist_materialization_report",
    "project_managed_merge_queue_record",
    "project_managed_merge_queue_records",
    "recover_managed_merge_receipts_from_git",
]
