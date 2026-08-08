"""Fail-closed validation for the prompt-only v3 convergence bootstrap.

ASE3-000 deliberately treats historical ASE/ASE2 state as evidence, never as
completion authority.  This module validates the bounded evidence packet that
binds v3 work to a current-main seed, accounts for every rescue-branch commit
and changed path, records historical state contradictions, and proves that the
dirty source checkout was not used as the integration worktree.

The configured-board preflight invokes this module with ``--check-all``.  The
command always prints one JSON object containing at least ``valid`` and
``errors`` and exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

CURRENT_MAIN_BASELINE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-current-main-baseline@1"
)
HISTORICAL_CONTRADICTION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-historical-contradictions@1"
)
RESCUE_DISPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-rescue-dispositions@1"
)
CLEAN_WORKTREE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-clean-worktree-receipt@1"
)
CONVERGENCE_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-manifest@1"
)
CONVERGENCE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-convergence-report@1"
)
POST_WAVE3_RESIDUAL_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.post-wave3-residual-report@1"
)

BOARD_NAMESPACE: Final = "agent-supervisor-prompt-only-self-improvement-v3"
POST_WAVE3_RESIDUAL_FILENAME: Final = "post_wave3_residuals_20260808.json"
ARTIFACT_FILENAMES: Final = (
    "current_main_baseline.json",
    "historical_state_contradictions.json",
    "rescue_artifact_dispositions.json",
    "clean_integration_worktree_receipt.json",
    POST_WAVE3_RESIDUAL_FILENAME,
)
MANIFEST_FILENAME: Final = "convergence_manifest.json"
DEFAULT_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
PROMPT_V3_TASKBOARD_RELATIVE_PATH: Final = Path(
    "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
)
PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: Final = (
    "provider_attempt_daemon_reload_receipt.json"
)
PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH: Final = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    + PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
)
DEFAULT_ARTIFACT_ROOT: Final = (
    DEFAULT_REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "prompt_only_self_improvement_v3"
    / "convergence"
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TASK_IDS: Final = frozenset(f"ASE3-{index:03d}" for index in range(15))
_DISPOSITIONS: Final = frozenset({"port", "rewrite", "superseded", "discard"})
_REQUIRED_CONTRADICTIONS: Final = frozenset(
    {
        "source-board-vs-eligible-index",
        "bundle-index-vs-eligible-index",
        "stale-process-projections",
        "drained-without-refill",
        "branch-local-completion",
    }
)
_POST_WAVE3_CREATED_AT: Final = "2026-08-08T09:53:00Z"
_POST_WAVE3_REPOSITORY: Final = {
    "head": "4370931d7dc556d56962a88ed1db511487be39d2",
    "tree": "1d472b508368a0574e1dbfa87467158377797e23",
    "branch": "agent/prompt-self-improvement-v3",
}
_POST_WAVE3_COMPLETED_TASKS: Final = {
    "ASE3-005": {
        "implementation_commit": "8b82c968d829a1191fcacff3e20804be0c232b0a",
        "merge_commit": "8945d1b08e564fb1baf26a38d7ea6909012a104b",
        "status_commit": "4370931d7dc556d56962a88ed1db511487be39d2",
        "declared_current_tree_tests_passed": 13,
        "declared_current_tree_tests_failed": 0,
    },
    "ASE3-007": {
        "implementation_commit": "5c4098a8adf7c29e24602a18b699f9042b3ca9f6",
        "merge_commit": "023bb9972ca8d9eb6009f565c3293c2ce8a16aea",
        "status_commit": "05773ac5abcf361a870404428f4e82dcd15168ce",
        "declared_current_tree_tests_passed": 87,
        "declared_current_tree_tests_failed": 0,
    },
}
_POST_WAVE3_RESIDUALS: Final = {
    "trusted-context-canonical-composition": (
        "ASE3-018",
        frozenset({"ASE3-001", "ASE3-002", "ASE3-005"}),
    ),
    "signed-authority-and-durable-provider-attempt": (
        "ASE3-019",
        frozenset({"ASE3-002", "ASE3-006"}),
    ),
    "production-durable-refill-wiring": (
        "ASE3-021",
        frozenset({"ASE3-007"}),
    ),
    "transactional-run-truth-and-effect-recovery": (
        "ASE3-020",
        frozenset({"ASE3-003", "ASE3-005", "ASE3-007"}),
    ),
}
_POST_WAVE3_PROVIDER_INCIDENT: Final = {
    "task_id": "ASE3-006",
    "event_id": "sha256:e2dee32eb866a9a4216c809318f4066bc49bf33e1e0ef3290365cf4ccaf58f97",
    "log_sha256": "sha256:2724af1a5b52fadae7130b4a80081cf9849dabc0f0104f839033474fff332596",
    "failure": "grok_authentication_unavailable",
    "attempt": 1,
    "attempt_consumed": False,
    "fallback_dispatched": False,
    "workspace_changed": False,
    "operator_fenced_before_retry": True,
}
_POST_WAVE3_DISPOSITION: Final = {
    "historical_task_status_authoritative": False,
    "declared_test_success_authorizes_goal_completion": False,
    "operator_reviewed_refill_required": True,
    "target_tasks": ["ASE3-018", "ASE3-019", "ASE3-021", "ASE3-020"],
    "gate_task": "ASE3-008",
    "completion_authority": False,
    "provider_policy_broadening_authorized": False,
    "attempt_counter_mutation_authorized": False,
}
_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID: Final = "ASE3-022"
_PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES: Final = (
    "ASE3-006",
    "ASE3-018",
    "ASE3-019",
)
_PROVIDER_ATTEMPT_RELOAD_GATE_BLOCKED_REASON: Final = (
    "provider-attempt daemon reload boundary not yet accepted"
)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name}: root must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _require_hex40(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        errors.append(f"{label}: expected a lowercase 40-hex Git identity")


def _require_sha256(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        errors.append(f"{label}: expected sha256:<64 lowercase hex>")


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class CurrentMainBaseline:
    """Immutable identities for current main, the seed, and rescue history."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CurrentMainBaseline:
        return cls(dict(payload))

    @property
    def integration_seed_commit(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("commit", ""))

    @property
    def rescue_head(self) -> str:
        return str(self.payload.get("rescue", {}).get("head", ""))

    @property
    def merge_base(self) -> str:
        return str(self.payload.get("rescue", {}).get("merge_base", ""))

    @property
    def upstream_main_commit(self) -> str:
        return str(self.payload.get("upstream_main", {}).get("commit", ""))

    @property
    def integration_seed_tree(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("tree", ""))

    @property
    def integration_branch(self) -> str:
        return str(self.payload.get("integration_seed", {}).get("branch", ""))

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CURRENT_MAIN_BASELINE_SCHEMA:
            errors.append("current_main_baseline.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("current_main_baseline.board_namespace: mismatch")
        for section, field in (
            ("upstream_main", "commit"),
            ("upstream_main", "tree"),
            ("integration_seed", "commit"),
            ("integration_seed", "tree"),
            ("rescue", "head"),
            ("rescue", "tree"),
            ("rescue", "merge_base"),
            ("rescue", "merge_base_tree"),
        ):
            block = self.payload.get(section, {})
            value = block.get(field) if isinstance(block, Mapping) else None
            _require_hex40(errors, f"current_main_baseline.{section}.{field}", value)
        rescue = self.payload.get("rescue", {})
        if not isinstance(rescue, Mapping):
            errors.append("current_main_baseline.rescue: expected object")
        else:
            for field in ("current_main_ahead", "rescue_ahead"):
                value = rescue.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(
                        f"current_main_baseline.rescue.{field}: expected nonnegative integer"
                    )
        integration = self.payload.get("integration_seed", {})
        if not isinstance(integration, Mapping):
            errors.append("current_main_baseline.integration_seed: expected object")
        else:
            _require_hex40(
                errors,
                "current_main_baseline.integration_seed.parent",
                integration.get("parent"),
            )
            if not str(integration.get("branch", "")).strip():
                errors.append("current_main_baseline.integration_seed.branch: required")
        upstream = self.payload.get("upstream_main", {})
        if not isinstance(upstream, Mapping):
            errors.append("current_main_baseline.upstream_main: expected object")
        elif not str(upstream.get("branch", "")).strip():
            errors.append("current_main_baseline.upstream_main.branch: required")
        checkout = self.payload.get("original_checkout", {})
        if not isinstance(checkout, Mapping):
            errors.append("current_main_baseline.original_checkout: expected object")
        else:
            if checkout.get("clean") is not False:
                errors.append("current_main_baseline.original_checkout.clean: must be false")
            if not isinstance(checkout.get("dirty_entry_count"), int) or int(
                checkout.get("dirty_entry_count", 0)
            ) <= 0:
                errors.append(
                    "current_main_baseline.original_checkout.dirty_entry_count: must be positive"
                )
            _require_sha256(
                errors,
                "current_main_baseline.original_checkout.status_sha256",
                checkout.get("status_sha256"),
            )
            if checkout.get("preservation_policy") != "read-only-protected":
                errors.append(
                    "current_main_baseline.original_checkout.preservation_policy: must be read-only-protected"
                )
            if not isinstance(checkout.get("path"), str) or not Path(
                str(checkout.get("path", ""))
            ).is_absolute():
                errors.append(
                    "current_main_baseline.original_checkout.path: expected absolute path"
                )
            _require_hex40(
                errors,
                "current_main_baseline.original_checkout.head",
                checkout.get("head"),
            )
            if checkout.get("head") != self.upstream_main_commit:
                errors.append(
                    "current_main_baseline.original_checkout.head: must equal upstream main commit"
                )
            if isinstance(upstream, Mapping) and checkout.get("branch") != upstream.get(
                "branch"
            ):
                errors.append(
                    "current_main_baseline.original_checkout.branch: must equal upstream main branch"
                )
            if checkout.get("status_snapshot_is_historical") is not True:
                errors.append(
                    "current_main_baseline.original_checkout.status_snapshot_is_historical: must be true"
                )
        submodules = self.payload.get("submodules")
        if not isinstance(submodules, Sequence) or isinstance(submodules, (str, bytes)):
            errors.append("current_main_baseline.submodules: expected list")
        else:
            paths: set[str] = set()
            for index, item in enumerate(submodules):
                if not isinstance(item, Mapping):
                    errors.append(f"current_main_baseline.submodules[{index}]: expected object")
                    continue
                path = item.get("path")
                if not isinstance(path, str) or not _is_safe_relative_path(path):
                    errors.append(
                        f"current_main_baseline.submodules[{index}].path: unsafe path"
                    )
                elif path in paths:
                    errors.append(
                        f"current_main_baseline.submodules[{index}].path: duplicate {path}"
                    )
                else:
                    paths.add(path)
                _require_hex40(
                    errors,
                    f"current_main_baseline.submodules[{index}].gitlink_commit",
                    item.get("gitlink_commit"),
                )
        return tuple(errors)


@dataclass(frozen=True)
class HistoricalStateContradictionReport:
    """Contradictory historical projections that are explicitly non-authoritative."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> HistoricalStateContradictionReport:
        return cls(dict(payload))

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != HISTORICAL_CONTRADICTION_SCHEMA:
            errors.append("historical_state_contradictions.schema: unsupported schema")
        if self.payload.get("authority") != "evidence-only":
            errors.append("historical_state_contradictions.authority: must be evidence-only")
        if self.payload.get("v3_completion_credit") is not False:
            errors.append(
                "historical_state_contradictions.v3_completion_credit: must be false"
            )
        sources = self.payload.get("sources")
        if not isinstance(sources, Mapping) or not sources:
            errors.append("historical_state_contradictions.sources: expected non-empty object")
        else:
            for source_id, source in sources.items():
                if not isinstance(source, Mapping):
                    errors.append(
                        f"historical_state_contradictions.sources.{source_id}: expected object"
                    )
                    continue
                _require_sha256(
                    errors,
                    f"historical_state_contradictions.sources.{source_id}.sha256",
                    source.get("sha256"),
                )
        records = self.payload.get("contradictions")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            errors.append(
                "historical_state_contradictions.contradictions: expected list"
            )
            return tuple(errors)
        codes: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}]: expected object"
                )
                continue
            code = record.get("code")
            if not isinstance(code, str) or not code:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].code: required"
                )
            elif code in codes:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].code: duplicate {code}"
                )
            else:
                codes.add(code)
            if record.get("authoritative") is not False:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].authoritative: must be false"
                )
            source_ids = record.get("source_ids")
            if not isinstance(source_ids, list) or not source_ids:
                errors.append(
                    f"historical_state_contradictions.contradictions[{index}].source_ids: required"
                )
            elif isinstance(sources, Mapping):
                unknown = sorted(set(source_ids) - set(sources))
                if unknown:
                    errors.append(
                        f"historical_state_contradictions.contradictions[{index}].source_ids: unknown {unknown}"
                    )
        missing = sorted(_REQUIRED_CONTRADICTIONS - codes)
        if missing:
            errors.append(
                "historical_state_contradictions.contradictions: missing required "
                + ",".join(missing)
            )
        return tuple(errors)


@dataclass(frozen=True)
class RescueArtifactDisposition:
    """Disposition for one rescue commit or one changed path."""

    kind: str
    identity: str
    disposition: str
    target_tasks: tuple[str, ...]
    rationale: str
    current_state: str = ""
    target_tasks_is_list: bool = True

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, kind: str
    ) -> RescueArtifactDisposition:
        identity_key = "commit" if kind == "commit" else "path"
        tasks = payload.get("target_tasks", [])
        return cls(
            kind=kind,
            identity=str(payload.get(identity_key, "")),
            disposition=str(payload.get("disposition", "")),
            target_tasks=tuple(str(item) for item in tasks) if isinstance(tasks, list) else (),
            rationale=str(payload.get("rationale", "")),
            current_state=str(payload.get("current_state", "")),
            target_tasks_is_list=isinstance(tasks, list),
        )

    def validate(self, *, index: int) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = f"rescue_artifact_dispositions.{self.kind}s[{index}]"
        if self.kind == "commit":
            _require_hex40(errors, f"{prefix}.commit", self.identity)
        elif not _is_safe_relative_path(self.identity):
            errors.append(f"{prefix}.path: unsafe path")
        if self.disposition not in _DISPOSITIONS:
            errors.append(f"{prefix}.disposition: unsupported {self.disposition!r}")
        if not self.target_tasks_is_list:
            errors.append(f"{prefix}.target_tasks: expected list")
        if self.disposition in {"port", "rewrite"} and not self.target_tasks:
            errors.append(f"{prefix}.target_tasks: required for {self.disposition}")
        for task in self.target_tasks:
            if task not in _TASK_IDS:
                errors.append(f"{prefix}.target_tasks: unknown task {task!r}")
        if len(self.target_tasks) != len(set(self.target_tasks)):
            errors.append(f"{prefix}.target_tasks: duplicate task")
        if not self.rationale.strip():
            errors.append(f"{prefix}.rationale: required")
        if self.kind == "file" and self.current_state not in {"missing", "diverged"}:
            errors.append(f"{prefix}.current_state: expected missing or diverged")
        return tuple(errors)


@dataclass(frozen=True)
class RescueDispositionReport:
    """Complete rescue population and its explicit convergence decisions."""

    payload: Mapping[str, Any]
    commits: tuple[RescueArtifactDisposition, ...]
    files: tuple[RescueArtifactDisposition, ...]
    shape_errors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RescueDispositionReport:
        commits_payload = payload.get("commits", [])
        files_payload = payload.get("files", [])
        shape_errors: list[str] = []

        def parse_population(
            value: Any,
            *,
            field: str,
            kind: str,
        ) -> tuple[RescueArtifactDisposition, ...]:
            if not isinstance(value, list):
                shape_errors.append(
                    f"rescue_artifact_dispositions.{field}: expected list"
                )
                return ()
            parsed: list[RescueArtifactDisposition] = []
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    shape_errors.append(
                        f"rescue_artifact_dispositions.{field}[{index}]: expected object"
                    )
                    item = {}
                parsed.append(RescueArtifactDisposition.from_dict(item, kind=kind))
            return tuple(parsed)

        commits = parse_population(
            commits_payload,
            field="commits",
            kind="commit",
        )
        files = parse_population(files_payload, field="files", kind="file")
        return cls(dict(payload), commits, files, tuple(shape_errors))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = list(self.shape_errors)
        if self.payload.get("schema") != RESCUE_DISPOSITION_SCHEMA:
            errors.append("rescue_artifact_dispositions.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("rescue_artifact_dispositions.board_namespace: mismatch")
        observed_at = self.payload.get("observed_at")
        if not isinstance(observed_at, str) or _UTC_TIMESTAMP.fullmatch(observed_at) is None:
            errors.append(
                "rescue_artifact_dispositions.observed_at: expected UTC timestamp"
            )
        if self.payload.get("historical_authority") != "evidence-only":
            errors.append(
                "rescue_artifact_dispositions.historical_authority: must be evidence-only"
            )
        if self.payload.get("bulk_merge_allowed") is not False:
            errors.append(
                "rescue_artifact_dispositions.bulk_merge_allowed: must be false"
            )
        for field, expected in (
            ("merge_base", baseline.merge_base),
            ("rescue_head", baseline.rescue_head),
            ("current_seed", baseline.integration_seed_commit),
        ):
            value = self.payload.get(field)
            _require_hex40(
                errors,
                f"rescue_artifact_dispositions.{field}",
                value,
            )
            if value != expected:
                errors.append(
                    f"rescue_artifact_dispositions.{field}: baseline mismatch"
                )
        if not str(self.payload.get("decision_rule", "")).strip():
            errors.append("rescue_artifact_dispositions.decision_rule: required")
        if len(self.commits) != 36:
            errors.append(
                f"rescue_artifact_dispositions.commits: expected 36, got {len(self.commits)}"
            )
        if len(self.files) != 35:
            errors.append(
                f"rescue_artifact_dispositions.files: expected 35, got {len(self.files)}"
            )
        for index, item in enumerate(self.commits):
            errors.extend(item.validate(index=index))
        for index, item in enumerate(self.files):
            errors.extend(item.validate(index=index))
        commit_ids = [item.identity for item in self.commits]
        file_paths = [item.identity for item in self.files]
        if len(commit_ids) != len(set(commit_ids)):
            errors.append("rescue_artifact_dispositions.commits: duplicate identity")
        if len(file_paths) != len(set(file_paths)):
            errors.append("rescue_artifact_dispositions.files: duplicate path")
        return tuple(errors)


@dataclass(frozen=True)
class CleanIntegrationWorktreeReceipt:
    """Receipt that separates the v3 integration lane from user changes."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> CleanIntegrationWorktreeReceipt:
        return cls(dict(payload))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CLEAN_WORKTREE_RECEIPT_SCHEMA:
            errors.append("clean_worktree_receipt.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("clean_worktree_receipt.board_namespace: mismatch")
        worktree = self.payload.get("worktree", {})
        if not isinstance(worktree, Mapping):
            errors.append("clean_worktree_receipt.worktree: expected object")
        else:
            if worktree.get("clean_at_creation") is not True:
                errors.append("clean_worktree_receipt.worktree.clean_at_creation: must be true")
            if worktree.get("creation_head") != baseline.integration_seed_commit:
                errors.append(
                    "clean_worktree_receipt.worktree.creation_head: must equal integration seed"
                )
            _require_hex40(
                errors,
                "clean_worktree_receipt.worktree.creation_tree",
                worktree.get("creation_tree"),
            )
            if worktree.get("creation_tree") != baseline.integration_seed_tree:
                errors.append(
                    "clean_worktree_receipt.worktree.creation_tree: must equal integration seed tree"
                )
            if worktree.get("branch") != baseline.integration_branch:
                errors.append(
                    "clean_worktree_receipt.worktree.branch: must equal integration branch"
                )
            if worktree.get("isolated_from_source_checkout") is not True:
                errors.append(
                    "clean_worktree_receipt.worktree.isolated_from_source_checkout: must be true"
                )
            if worktree.get("working_tree_is_expected_to_change_after_receipt") is not True:
                errors.append(
                    "clean_worktree_receipt.worktree.working_tree_is_expected_to_change_after_receipt: must be true"
                )
            path = worktree.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                errors.append(
                    "clean_worktree_receipt.worktree.path: expected absolute path"
                )
        source = self.payload.get("protected_source_checkout", {})
        baseline_source = baseline.payload.get("original_checkout", {})
        if not isinstance(source, Mapping) or not isinstance(baseline_source, Mapping):
            errors.append("clean_worktree_receipt.protected_source_checkout: expected object")
        else:
            for field in (
                "path",
                "head",
                "status_sha256",
                "dirty_entry_count",
                "preservation_policy",
            ):
                if source.get(field) != baseline_source.get(field):
                    errors.append(
                        "clean_worktree_receipt.protected_source_checkout."
                        f"{field}: baseline mismatch"
                    )
            if source.get("modified_by_bootstrap") is not False:
                errors.append(
                    "clean_worktree_receipt.protected_source_checkout.modified_by_bootstrap: must be false"
                )
        state = self.payload.get("state_namespace", {})
        if not isinstance(state, Mapping):
            errors.append("clean_worktree_receipt.state_namespace: expected object")
        else:
            value = str(state.get("path", ""))
            normalized_value = value.replace("_", "-")
            if "prompt-only-self-improvement-v3" not in normalized_value:
                errors.append(
                    "clean_worktree_receipt.state_namespace.path: must be a fresh v3 namespace"
                )
            if "prompt-only-entrypoints-v2" in normalized_value:
                errors.append(
                    "clean_worktree_receipt.state_namespace.path: historical namespace forbidden"
                )
            if state.get("fresh_for_board") is not True:
                errors.append(
                    "clean_worktree_receipt.state_namespace.fresh_for_board: must be true"
                )
            if state.get("historical_import_allowed") is not False:
                errors.append(
                    "clean_worktree_receipt.state_namespace.historical_import_allowed: must be false"
                )
            if state.get("generated_runtime_artifacts_are_completion_authority") is not False:
                errors.append(
                    "clean_worktree_receipt.state_namespace.generated_runtime_artifacts_are_completion_authority: must be false"
                )
        downstream = self.payload.get("downstream_binding", {})
        if not isinstance(downstream, Mapping):
            errors.append("clean_worktree_receipt.downstream_binding: expected object")
        else:
            if downstream.get("required_ancestor") != baseline.integration_seed_commit:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.required_ancestor: baseline mismatch"
                )
            if downstream.get("required_branch") != baseline.integration_branch:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.required_branch: baseline mismatch"
                )
            if downstream.get("changed_revision_requires_fresh_validation") is not True:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.changed_revision_requires_fresh_validation: must be true"
                )
            if downstream.get("historical_ase_or_ase2_receipt_satisfies_v3") is not False:
                errors.append(
                    "clean_worktree_receipt.downstream_binding.historical_ase_or_ase2_receipt_satisfies_v3: must be false"
                )
        return tuple(errors)


@dataclass(frozen=True)
class PostWave3ResidualReport:
    """Fail-closed residual audit that authorizes the post-wave-3 refill only."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PostWave3ResidualReport:
        return cls(dict(payload))

    @property
    def repository_head(self) -> str:
        repository = self.payload.get("repository", {})
        return str(repository.get("head", "")) if isinstance(repository, Mapping) else ""

    @property
    def repository_tree(self) -> str:
        repository = self.payload.get("repository", {})
        return str(repository.get("tree", "")) if isinstance(repository, Mapping) else ""

    @property
    def completed_task_evidence(self) -> Mapping[str, Any]:
        evidence = self.payload.get("completed_task_evidence", {})
        return evidence if isinstance(evidence, Mapping) else {}

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        prefix = "post_wave3_residuals"
        expected_fields = {
            "schema",
            "created_at",
            "board_namespace",
            "repository",
            "completed_task_evidence",
            "residuals",
            "provider_incident",
            "disposition",
        }
        if set(self.payload) != expected_fields:
            errors.append(f"{prefix}: field population mismatch")
        if self.payload.get("schema") != POST_WAVE3_RESIDUAL_SCHEMA:
            errors.append(f"{prefix}.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append(f"{prefix}.board_namespace: mismatch")
        created_at = self.payload.get("created_at")
        if (
            not isinstance(created_at, str)
            or _UTC_TIMESTAMP.fullmatch(created_at) is None
            or created_at != _POST_WAVE3_CREATED_AT
        ):
            errors.append(
                f"{prefix}.created_at: expected immutable UTC timestamp "
                f"{_POST_WAVE3_CREATED_AT}"
            )

        repository = self.payload.get("repository")
        if not isinstance(repository, Mapping):
            errors.append(f"{prefix}.repository: expected object")
        else:
            if set(repository) != set(_POST_WAVE3_REPOSITORY):
                errors.append(f"{prefix}.repository: field population mismatch")
            for field in ("head", "tree"):
                value = repository.get(field)
                _require_hex40(errors, f"{prefix}.repository.{field}", value)
                if value != _POST_WAVE3_REPOSITORY[field]:
                    errors.append(
                        f"{prefix}.repository.{field}: immutable identity mismatch"
                    )
            if repository.get("branch") != _POST_WAVE3_REPOSITORY["branch"]:
                errors.append(f"{prefix}.repository.branch: mismatch")

        completed = self.payload.get("completed_task_evidence")
        if not isinstance(completed, Mapping):
            errors.append(f"{prefix}.completed_task_evidence: expected object")
        else:
            if set(completed) != set(_POST_WAVE3_COMPLETED_TASKS):
                errors.append(
                    f"{prefix}.completed_task_evidence: expected exactly "
                    "ASE3-005 and ASE3-007"
                )
            for task_id, expected in _POST_WAVE3_COMPLETED_TASKS.items():
                item = completed.get(task_id)
                item_prefix = f"{prefix}.completed_task_evidence.{task_id}"
                if not isinstance(item, Mapping):
                    errors.append(f"{item_prefix}: expected object")
                    continue
                if set(item) != set(expected):
                    errors.append(f"{item_prefix}: field population mismatch")
                for field in (
                    "implementation_commit",
                    "merge_commit",
                    "status_commit",
                ):
                    value = item.get(field)
                    _require_hex40(errors, f"{item_prefix}.{field}", value)
                    if value != expected[field]:
                        errors.append(f"{item_prefix}.{field}: immutable identity mismatch")
                for field in (
                    "declared_current_tree_tests_passed",
                    "declared_current_tree_tests_failed",
                ):
                    value = item.get(field)
                    if type(value) is not int or value != expected[field]:
                        errors.append(
                            f"{item_prefix}.{field}: expected {expected[field]}"
                        )

        residuals = self.payload.get("residuals")
        observed_residuals: dict[str, Mapping[str, Any]] = {}
        if not isinstance(residuals, list):
            errors.append(f"{prefix}.residuals: expected list")
        else:
            if len(residuals) != len(_POST_WAVE3_RESIDUALS):
                errors.append(
                    f"{prefix}.residuals: expected exactly "
                    f"{len(_POST_WAVE3_RESIDUALS)} records"
                )
            residual_fields = {
                "gap_id",
                "severity",
                "source_tasks",
                "target_task",
                "evidence",
            }
            for index, record in enumerate(residuals):
                record_prefix = f"{prefix}.residuals[{index}]"
                if not isinstance(record, Mapping):
                    errors.append(f"{record_prefix}: expected object")
                    continue
                if set(record) != residual_fields:
                    errors.append(f"{record_prefix}: field population mismatch")
                gap_id = record.get("gap_id")
                if not isinstance(gap_id, str) or not gap_id:
                    errors.append(f"{record_prefix}.gap_id: required")
                    continue
                if gap_id in observed_residuals:
                    errors.append(f"{record_prefix}.gap_id: duplicate {gap_id}")
                    continue
                observed_residuals[gap_id] = record
                if record.get("severity") != "P0":
                    errors.append(f"{record_prefix}.severity: expected P0")
                evidence = record.get("evidence")
                if (
                    not isinstance(evidence, list)
                    or not evidence
                    or any(not isinstance(item, str) or not item.strip() for item in evidence)
                    or len(evidence) != len(set(evidence))
                ):
                    errors.append(
                        f"{record_prefix}.evidence: expected unique non-empty strings"
                    )
            if set(observed_residuals) != set(_POST_WAVE3_RESIDUALS):
                errors.append(f"{prefix}.residuals: gap population mismatch")
            for gap_id, (target_task, source_tasks) in _POST_WAVE3_RESIDUALS.items():
                record = observed_residuals.get(gap_id)
                if record is None:
                    continue
                record_prefix = f"{prefix}.residuals.{gap_id}"
                if record.get("target_task") != target_task:
                    errors.append(
                        f"{record_prefix}.target_task: expected {target_task}"
                    )
                observed_sources = record.get("source_tasks")
                if (
                    not isinstance(observed_sources, list)
                    or any(not isinstance(item, str) for item in observed_sources)
                    or len(observed_sources) != len(set(observed_sources))
                    or frozenset(observed_sources) != source_tasks
                ):
                    errors.append(f"{record_prefix}.source_tasks: population mismatch")

        provider = self.payload.get("provider_incident")
        if not isinstance(provider, Mapping):
            errors.append(f"{prefix}.provider_incident: expected object")
        else:
            if set(provider) != set(_POST_WAVE3_PROVIDER_INCIDENT):
                errors.append(f"{prefix}.provider_incident: field population mismatch")
            _require_sha256(
                errors,
                f"{prefix}.provider_incident.event_id",
                provider.get("event_id"),
            )
            _require_sha256(
                errors,
                f"{prefix}.provider_incident.log_sha256",
                provider.get("log_sha256"),
            )
            for field, expected in _POST_WAVE3_PROVIDER_INCIDENT.items():
                actual = provider.get(field)
                matches = (
                    actual is expected
                    if isinstance(expected, bool)
                    else type(actual) is int and actual == expected
                    if isinstance(expected, int)
                    else actual == expected
                )
                if not matches:
                    errors.append(
                        f"{prefix}.provider_incident.{field}: expected {expected!r}"
                    )

        disposition = self.payload.get("disposition")
        if not isinstance(disposition, Mapping):
            errors.append(f"{prefix}.disposition: expected object")
        else:
            if set(disposition) != set(_POST_WAVE3_DISPOSITION):
                errors.append(f"{prefix}.disposition: field population mismatch")
            for field, expected in _POST_WAVE3_DISPOSITION.items():
                actual = disposition.get(field)
                if isinstance(expected, bool):
                    matches = actual is expected
                elif isinstance(expected, list):
                    matches = isinstance(actual, list) and actual == expected
                else:
                    matches = actual == expected
                if not matches:
                    errors.append(
                        f"{prefix}.disposition.{field}: expected {expected!r}"
                    )
        return tuple(errors)


@dataclass(frozen=True)
class ConvergenceManifest:
    """Root binding for the bounded ASE3-000 evidence packet."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConvergenceManifest:
        return cls(dict(payload))

    def validate(self, baseline: CurrentMainBaseline) -> tuple[str, ...]:
        errors: list[str] = []
        if self.payload.get("schema") != CONVERGENCE_MANIFEST_SCHEMA:
            errors.append("convergence_manifest.schema: unsupported schema")
        if self.payload.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("convergence_manifest.board_namespace: mismatch")
        if self.payload.get("task_id") != "ASE3-000":
            errors.append("convergence_manifest.task_id: expected ASE3-000")
        if self.payload.get("goal_id") != "ASE3-G010":
            errors.append("convergence_manifest.goal_id: expected ASE3-G010")
        created_at = self.payload.get("created_at")
        if not isinstance(created_at, str) or _UTC_TIMESTAMP.fullmatch(created_at) is None:
            errors.append("convergence_manifest.created_at: expected UTC timestamp")
        if self.payload.get("integration_seed_commit") != baseline.integration_seed_commit:
            errors.append(
                "convergence_manifest.integration_seed_commit: baseline mismatch"
            )
        if self.payload.get("integration_seed_tree") != baseline.integration_seed_tree:
            errors.append("convergence_manifest.integration_seed_tree: baseline mismatch")
        if self.payload.get("historical_completion_authority") is not False:
            errors.append(
                "convergence_manifest.historical_completion_authority: must be false"
            )
        if self.payload.get("rescue_bulk_merge_allowed") is not False:
            errors.append("convergence_manifest.rescue_bulk_merge_allowed: must be false")
        components = self.payload.get("components")
        if not isinstance(components, Mapping):
            errors.append("convergence_manifest.components: expected object")
        else:
            if set(components) != set(ARTIFACT_FILENAMES):
                errors.append("convergence_manifest.components: population mismatch")
            for filename, digest in components.items():
                if not _is_safe_relative_path(str(filename)):
                    errors.append(f"convergence_manifest.components.{filename}: unsafe path")
                _require_sha256(errors, f"convergence_manifest.components.{filename}", digest)
        population = self.payload.get("population", {})
        if not isinstance(population, Mapping):
            errors.append("convergence_manifest.population: expected object")
        else:
            expected_population = {
                "rescue_commits": 36,
                "rescue_changed_paths": 35,
                "v2_tasks": 8,
                "historical_contradictions": 5,
                "v3_seed_tasks": 15,
                "v3_seed_goals": 9,
            }
            if set(population) != set(expected_population):
                errors.append("convergence_manifest.population: population mismatch")
            for key, value in expected_population.items():
                if population.get(key) != value:
                    errors.append(
                        f"convergence_manifest.population.{key}: expected {value}"
                    )
        completion_rules = self.payload.get("completion_rules", {})
        expected_completion_rules = {
            "historical_status_or_receipt_satisfies_v3": False,
            "branch_local_commit_satisfies_v3": False,
            "queue_drain_satisfies_goal_completion": False,
            "current_tree_acceptance_required": True,
            "forced_residual_scan_required": True,
        }
        if not isinstance(completion_rules, Mapping):
            errors.append("convergence_manifest.completion_rules: expected object")
        else:
            if set(completion_rules) != set(expected_completion_rules):
                errors.append(
                    "convergence_manifest.completion_rules: population mismatch"
                )
            for field, expected in expected_completion_rules.items():
                if completion_rules.get(field) is not expected:
                    errors.append(
                        f"convergence_manifest.completion_rules.{field}: expected {expected!r}"
                    )
        downstream_rules = self.payload.get("downstream_rules", {})
        expected_downstream_rules = {
            "required_ancestor": baseline.integration_seed_commit,
            "merge_target_branch": baseline.integration_branch,
            "rescue_disposition_required_before_use": True,
            "fresh_validation_receipt_required_per_task": True,
            "protected_source_checkout_may_be_modified": False,
        }
        if not isinstance(downstream_rules, Mapping):
            errors.append("convergence_manifest.downstream_rules: expected object")
        else:
            if set(downstream_rules) != set(expected_downstream_rules):
                errors.append("convergence_manifest.downstream_rules: population mismatch")
            for field, expected in expected_downstream_rules.items():
                actual = downstream_rules.get(field)
                matches = (
                    actual is expected
                    if isinstance(expected, bool)
                    else actual == expected
                )
                if not matches:
                    errors.append(
                        f"convergence_manifest.downstream_rules.{field}: expected {expected!r}"
                    )
        return tuple(errors)


@dataclass(frozen=True)
class ConvergenceValidationReport:
    """Machine-readable preflight result."""

    valid: bool
    errors: tuple[str, ...]
    checked_artifacts: tuple[str, ...]
    integration_seed_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONVERGENCE_REPORT_SCHEMA,
            "valid": self.valid,
            "errors": list(self.errors),
            "checked_artifacts": list(self.checked_artifacts),
            "integration_seed_commit": self.integration_seed_commit,
        }


def _load_taskboard_metadata(taskboard_path: Path) -> dict[str, dict[str, str]]:
    """Read only the bounded Markdown metadata needed by the bootstrap gate.

    The convergence validator is also executed directly by file path, where
    package-relative imports are unavailable.  Keep this parser deliberately
    small and reject duplicate task IDs or metadata keys instead of inheriting
    the runtime parser's last-value-wins behavior.
    """

    text = taskboard_path.read_text(encoding="utf-8")
    tasks: dict[str, dict[str, str]] = {}
    current_id = ""
    current_metadata: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_id, current_metadata
        if not current_id:
            return
        if current_id in tasks:
            raise ValueError(f"duplicate task id: {current_id}")
        tasks[current_id] = dict(current_metadata)
        current_id = ""
        current_metadata = {}

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            header = line[3:].strip()
            task_id = header.split(" ", 1)[0]
            if task_id.startswith("ASE3-"):
                current_id = task_id
            continue
        if not current_id:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in current_metadata:
            raise ValueError(
                f"duplicate metadata key for {current_id}: {normalized_key}"
            )
        current_metadata[normalized_key] = value.strip()
    flush()
    return tasks


def _taskboard_csv(metadata: Mapping[str, str], field: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(metadata.get(field, "")).split(",")
        if item.strip()
    )


def _validate_provider_attempt_reload_gate(
    *,
    taskboard_path: Path,
    artifact_root: Path,
) -> list[str]:
    """Validate the initial noncanonical reload gate.

    ASE3-022 may transition to ``completed`` only after this module gains a
    strict validator for ``provider_attempt_daemon_reload_receipt.json`` and
    the convergence manifest binds that receipt's digest.  Until both changes
    land atomically with the protected taskboard transition, the only accepted
    state is the exact blocked, review-only gate declared below.
    """

    errors: list[str] = []
    prefix = "provider_attempt_reload_gate"
    receipt_path = artifact_root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    try:
        receipt_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(f"{prefix}.receipt: unable to inspect reserved path: {exc}")
    else:
        errors.append(
            f"{prefix}.receipt: present without a strict validator and "
            "convergence-manifest binding"
        )

    try:
        tasks = _load_taskboard_metadata(taskboard_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{prefix}.taskboard: {exc}")
        return errors

    gate = tasks.get(_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID)
    if gate is None:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}: expected exactly one task"
        )
        return errors
    gate_status = gate.get("status", "todo").strip().lower()
    if gate_status == "completed":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.status: "
            "completion requires a strict reload "
            "receipt validator and convergence-manifest binding"
        )
    if gate_status != "blocked":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.status: expected blocked"
        )
    if gate.get("completion", "manual").strip().lower() != "manual":
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.completion: "
            "expected manual"
        )

    expected_metadata = {
        "is schedulable": "false",
        "review only": "true",
        "canonical board task": "false",
        "blocked reason": _PROVIDER_ATTEMPT_RELOAD_GATE_BLOCKED_REASON,
    }
    for field, expected in expected_metadata.items():
        actual = gate.get(field)
        if actual != expected:
            errors.append(
                f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}."
                f"{field.replace(' ', '_')}: "
                f"expected {expected!r}"
            )
    if _taskboard_csv(gate, "depends on") != _PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.depends_on: "
            "expected exactly "
            + ",".join(_PROVIDER_ATTEMPT_RELOAD_GATE_DEPENDENCIES)
        )
    if "goal id" in gate:
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.goal_id: must be absent"
        )
    if _taskboard_csv(gate, "outputs") != (
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    ):
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.outputs: expected only "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )
    if gate.get("predicted files") != (
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    ):
        errors.append(
            f"{prefix}.{_PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID}.predicted_files: "
            "expected only "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )

    refill_task = tasks.get("ASE3-021")
    if refill_task is None:
        errors.append(f"{prefix}.ASE3-021: expected exactly one task")
    elif _PROVIDER_ATTEMPT_RELOAD_GATE_TASK_ID not in _taskboard_csv(
        refill_task,
        "depends on",
    ):
        errors.append(f"{prefix}.ASE3-021.depends_on: missing ASE3-022")
    return errors


def _validate_repository_binding(
    *,
    repo_root: Path,
    baseline: CurrentMainBaseline,
    rescue: RescueDispositionReport,
    post_wave3: PostWave3ResidualReport,
) -> list[str]:
    errors: list[str] = []
    repo_root = repo_root.resolve()
    if not (repo_root / ".git").exists():
        # Linked worktrees use a .git file; ordinary clones use a directory.
        errors.append(f"repository_binding.repo_root: not a Git worktree: {repo_root}")
        return errors

    identity_sections = (
        ("upstream_main", baseline.payload.get("upstream_main", {}), "commit", "tree"),
        (
            "integration_seed",
            baseline.payload.get("integration_seed", {}),
            "commit",
            "tree",
        ),
        ("rescue_head", baseline.payload.get("rescue", {}), "head", "tree"),
        (
            "merge_base",
            baseline.payload.get("rescue", {}),
            "merge_base",
            "merge_base_tree",
        ),
    )
    for label, section, commit_field, tree_field in identity_sections:
        if not isinstance(section, Mapping):
            errors.append(f"repository_binding.{label}: baseline section unavailable")
            continue
        identity = str(section.get(commit_field, ""))
        expected_tree = str(section.get(tree_field, ""))
        result = _git(repo_root, "rev-parse", "--verify", f"{identity}^{{tree}}")
        if result.returncode != 0:
            errors.append(f"repository_binding.{label}: Git object unavailable")
        elif result.stdout.strip() != expected_tree:
            errors.append(f"repository_binding.{label}.tree: Git identity mismatch")

    integration = baseline.payload.get("integration_seed", {})
    expected_parent = (
        str(integration.get("parent", "")) if isinstance(integration, Mapping) else ""
    )
    parents = _git(
        repo_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        baseline.integration_seed_commit,
    )
    parent_fields = parents.stdout.strip().split()
    if parents.returncode != 0 or parent_fields[1:] != [expected_parent]:
        errors.append("repository_binding.integration_seed.parent: Git identity mismatch")

    actual_merge_base = _git(
        repo_root,
        "merge-base",
        baseline.upstream_main_commit,
        baseline.rescue_head,
    )
    if (
        actual_merge_base.returncode != 0
        or actual_merge_base.stdout.strip() != baseline.merge_base
    ):
        errors.append("repository_binding.merge_base: computed identity mismatch")

    divergence = _git(
        repo_root,
        "rev-list",
        "--left-right",
        "--count",
        f"{baseline.upstream_main_commit}...{baseline.rescue_head}",
    )
    rescue_payload = baseline.payload.get("rescue", {})
    try:
        current_main_ahead, rescue_ahead = (
            int(item) for item in divergence.stdout.strip().split()
        )
    except (TypeError, ValueError):
        current_main_ahead = rescue_ahead = -1
    if divergence.returncode != 0 or not isinstance(rescue_payload, Mapping):
        errors.append("repository_binding.rescue.divergence: unable to compute")
    elif (
        current_main_ahead != rescue_payload.get("current_main_ahead")
        or rescue_ahead != rescue_payload.get("rescue_ahead")
    ):
        errors.append("repository_binding.rescue.divergence: baseline mismatch")

    submodules = baseline.payload.get("submodules", ())
    if isinstance(submodules, Sequence) and not isinstance(submodules, (str, bytes)):
        for index, item in enumerate(submodules):
            if not isinstance(item, Mapping):
                continue
            relative = str(item.get("path", ""))
            expected_gitlink = str(item.get("gitlink_commit", ""))
            result = _git(
                repo_root,
                "ls-tree",
                baseline.integration_seed_commit,
                "--",
                relative,
            )
            match = re.fullmatch(
                rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}\n?",
                result.stdout,
            )
            if (
                result.returncode != 0
                or match is None
                or match.group(1) != expected_gitlink
            ):
                errors.append(
                    f"repository_binding.submodules[{index}].gitlink_commit: Git identity mismatch"
                )

    ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        baseline.integration_seed_commit,
        "HEAD",
    )
    if ancestor.returncode != 0:
        errors.append("repository_binding.integration_seed: not an ancestor of HEAD")

    commit_result = _git(
        repo_root,
        "rev-list",
        "--reverse",
        f"{baseline.merge_base}..{baseline.rescue_head}",
    )
    if commit_result.returncode != 0:
        errors.append("repository_binding.rescue_commits: unable to enumerate")
    else:
        actual_commits = tuple(line for line in commit_result.stdout.splitlines() if line)
        expected_commits = tuple(item.identity for item in rescue.commits)
        if actual_commits != expected_commits:
            errors.append("repository_binding.rescue_commits: manifest population mismatch")

    paths_result = _git(
        repo_root,
        "diff",
        "--name-only",
        baseline.merge_base,
        baseline.rescue_head,
    )
    if paths_result.returncode != 0:
        errors.append("repository_binding.rescue_paths: unable to enumerate")
    else:
        actual_paths = tuple(line for line in paths_result.stdout.splitlines() if line)
        expected_paths = tuple(item.identity for item in rescue.files)
        if actual_paths != expected_paths:
            errors.append("repository_binding.rescue_paths: manifest population mismatch")

    residual_tree = _git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{post_wave3.repository_head}^{{tree}}",
    )
    if residual_tree.returncode != 0:
        errors.append("repository_binding.post_wave3.head: Git object unavailable")
    elif residual_tree.stdout.strip() != post_wave3.repository_tree:
        errors.append("repository_binding.post_wave3.tree: Git identity mismatch")

    residual_ancestor = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        post_wave3.repository_head,
        "HEAD",
    )
    if residual_ancestor.returncode != 0:
        errors.append("repository_binding.post_wave3.head: not an ancestor of HEAD")

    for task_id in sorted(_POST_WAVE3_COMPLETED_TASKS):
        item = post_wave3.completed_task_evidence.get(task_id, {})
        if not isinstance(item, Mapping):
            continue
        identities = {
            field: str(item.get(field, ""))
            for field in (
                "implementation_commit",
                "merge_commit",
                "status_commit",
            )
        }
        for field, identity in identities.items():
            available = _git(
                repo_root,
                "rev-parse",
                "--verify",
                f"{identity}^{{commit}}",
            )
            if available.returncode != 0:
                errors.append(
                    f"repository_binding.post_wave3.{task_id}.{field}: "
                    "Git object unavailable"
                )
        ancestry_chain = (
            ("implementation_commit", "merge_commit"),
            ("merge_commit", "status_commit"),
        )
        for ancestor_field, descendant_field in ancestry_chain:
            ancestry = _git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                identities[ancestor_field],
                identities[descendant_field],
            )
            if ancestry.returncode != 0:
                errors.append(
                    f"repository_binding.post_wave3.{task_id}.{ancestor_field}: "
                    f"not an ancestor of {descendant_field}"
                )
        status_ancestry = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            identities["status_commit"],
            post_wave3.repository_head,
        )
        if status_ancestry.returncode != 0:
            errors.append(
                f"repository_binding.post_wave3.{task_id}.status_commit: "
                "not an ancestor of report head"
            )
    return errors


def validate_convergence_artifacts(
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    repo_root: Path | str | None = DEFAULT_REPOSITORY_ROOT,
    check_repository: bool = True,
    taskboard_path: Path | str | None = None,
) -> ConvergenceValidationReport:
    """Validate the entire ASE3-000 packet without trusting historical state."""

    root = Path(artifact_root)
    errors: list[str] = []
    checked: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for filename in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME):
        path = root / filename
        checked.append(filename)
        try:
            payloads[filename] = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        return ConvergenceValidationReport(False, tuple(errors), tuple(checked))

    baseline = CurrentMainBaseline.from_dict(payloads["current_main_baseline.json"])
    contradictions = HistoricalStateContradictionReport.from_dict(
        payloads["historical_state_contradictions.json"]
    )
    rescue = RescueDispositionReport.from_dict(
        payloads["rescue_artifact_dispositions.json"]
    )
    worktree = CleanIntegrationWorktreeReceipt.from_dict(
        payloads["clean_integration_worktree_receipt.json"]
    )
    post_wave3 = PostWave3ResidualReport.from_dict(
        payloads[POST_WAVE3_RESIDUAL_FILENAME]
    )
    manifest = ConvergenceManifest.from_dict(payloads[MANIFEST_FILENAME])

    errors.extend(baseline.validate())
    errors.extend(contradictions.validate())
    errors.extend(rescue.validate(baseline))
    errors.extend(worktree.validate(baseline))
    errors.extend(post_wave3.validate())
    errors.extend(manifest.validate(baseline))
    board_path = (
        Path(taskboard_path)
        if taskboard_path is not None
        else Path(repo_root or DEFAULT_REPOSITORY_ROOT)
        / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    )
    errors.extend(
        _validate_provider_attempt_reload_gate(
            taskboard_path=board_path,
            artifact_root=root,
        )
    )

    components = manifest.payload.get("components", {})
    if isinstance(components, Mapping):
        for filename in ARTIFACT_FILENAMES:
            expected = components.get(filename)
            actual = _sha256_file(root / filename)
            if expected != actual:
                errors.append(
                    f"convergence_manifest.components.{filename}: digest mismatch"
                )

    if check_repository and repo_root is not None and not errors:
        errors.extend(
            _validate_repository_binding(
                repo_root=Path(repo_root),
                baseline=baseline,
                rescue=rescue,
                post_wave3=post_wave3,
            )
        )
    return ConvergenceValidationReport(
        valid=not errors,
        errors=tuple(errors),
        checked_artifacts=tuple(checked),
        integration_seed_commit=baseline.integration_seed_commit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="validate all checked-in ASE3-000 convergence artifacts",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="bounded convergence artifact directory",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help="Git worktree used for live object/population checks",
    )
    parser.add_argument(
        "--taskboard-path",
        type=Path,
        default=None,
        help="protected v3 taskboard; defaults below --repo-root",
    )
    parser.add_argument(
        "--no-repository-check",
        action="store_true",
        help="validate packet structure and digests without live Git checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check_all:
        report = ConvergenceValidationReport(
            valid=False,
            errors=("--check-all is required",),
            checked_artifacts=(),
        )
    else:
        report = validate_convergence_artifacts(
            args.artifacts_root,
            repo_root=args.repo_root,
            check_repository=not args.no_repository_check,
            taskboard_path=args.taskboard_path,
        )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess test
    raise SystemExit(main())
