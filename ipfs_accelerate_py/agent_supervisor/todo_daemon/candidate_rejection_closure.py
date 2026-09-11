"""Exact observations for a protected candidate rejection; never acceptance.

Provider cleanup is admitted independently through the signed route/native CAS.
The two lifecycle events are emitted around the actual terminal CAS and successful
compare-delete. Neither terminal state nor an absent lifecycle row can replace
this pair. Initial scope deliberately excludes pooled checkout release.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from ..merge.worktree_lifecycle import WorkspaceLifecycleRecord

PROVIDER_SCHEMA = "candidate-provider-cleanup@1"
TERMINAL_SCHEMA = "candidate-rejection-lifecycle-terminal@1"
RELEASED_SCHEMA = "candidate-rejection-lifecycle-released@1"
CLOSURE_SCHEMA = "database-portal-candidate-rejection-closure@1"
CALLBACK_SCHEMA = "database-provider-callback-candidate-rejected@1"
TERMINAL_EVENT = "candidate_rejection_lifecycle_terminal"
RELEASED_EVENT = "candidate_rejection_lifecycle_released"


class CandidateClosureObservationUnknown(RuntimeError):
    """A protected handoff failed; no generic cleanup/release may follow."""


def sealed(body: Mapping[str, Any], field: str = "receipt_id") -> dict[str, Any]:
    value = dict(body)
    if field in value:
        raise ValueError("identity field already present")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**value, field: "sha256:" + hashlib.sha256(raw.encode()).hexdigest()}


def exact_seal(value: object, field: str = "receipt_id") -> bool:
    if not isinstance(value, Mapping) or len(value) > 64:
        return False
    try:
        return dict(value) == sealed(
            {k: v for k, v in value.items() if k != field}, field
        )
    except (TypeError, ValueError):
        return False


def validate_lifecycle_pair(prior: object, terminal: object) -> bool:
    """Require native record shapes and one exact terminal transition."""
    if not isinstance(prior, Mapping) or not isinstance(terminal, Mapping):
        return False
    try:
        before = WorkspaceLifecycleRecord.from_dict(prior)
        after = WorkspaceLifecycleRecord.from_dict(terminal)
    except (TypeError, ValueError, RuntimeError):
        return False
    if before.to_dict() != dict(prior) or after.to_dict() != dict(terminal):
        return False
    if (
        before.record_id != before.compute_record_id()
        or after.record_id != after.compute_record_id()
    ):
        return False
    changes = {"state", "fence", "updated_at", "expires_at", "terminal_reason"}
    return (
        before.is_nonterminal
        and after.is_terminal
        and after.fence == before.fence + 1
        and after.terminal_reason == "worktree_cleaned"
        and before.owner.pid > 0
        and before.owner.start_time_ticks > 0
        and all(prior[name] == terminal[name] for name in prior if name not in changes)
    )


class CandidateLifecycleHandoff:
    """Attempt-local observer passed only through nonpooled preservation."""

    def __init__(
        self,
        *,
        cleanup: Mapping[str, Any],
        record_event: Callable[..., Any],
        task_id: str,
        attempt: int,
        workspace_path: str,
        branch: str,
        preserved_commit: str,
        rescue_branch: str,
    ) -> None:
        if (
            not exact_seal(cleanup, "proof_id")
            or cleanup.get("schema") != PROVIDER_SCHEMA
            or cleanup.get("task_id") != task_id
            or cleanup.get("attempt") != attempt
            or cleanup.get("workspace_path") != workspace_path
            or not preserved_commit
            or not rescue_branch
        ):
            raise ValueError("candidate cleanup handoff binding mismatch")
        self._record_event = record_event
        self._binding = {
            "task_id": task_id,
            "attempt": attempt,
            "worktree_path": workspace_path,
            "branch": branch,
            "preserved_commit": preserved_commit,
            "rescue_branch": rescue_branch,
            "provider_cleanup": dict(cleanup),
            "completion_authority": False,
        }
        self._terminal: dict[str, Any] | None = None

    def terminal(
        self,
        prior: WorkspaceLifecycleRecord,
        terminal: WorkspaceLifecycleRecord,
    ) -> dict[str, Any]:
        before, after = prior.to_dict(), terminal.to_dict()
        cleanup = self._binding["provider_cleanup"]
        if (
            self._terminal is not None
            or not validate_lifecycle_pair(before, after)
            or before["task_id"] != self._binding["task_id"]
            or before["attempt"] != self._binding["attempt"]
            or before["branch"] != self._binding["branch"]
            or before["workspace_path"] != self._binding["worktree_path"]
            or before["canonical_task_cid"] != cleanup["task_revision_cid"]
        ):
            raise ValueError("candidate lifecycle terminal binding mismatch")
        value = sealed(
            {
                "schema": TERMINAL_SCHEMA,
                **self._binding,
                "prior": before,
                "terminal": after,
            }
        )
        # Set only after durable event publication succeeds.
        self._record_event(
            TERMINAL_EVENT,
            {
                "task_id": self._binding["task_id"],
                "attempt": self._binding["attempt"],
                "closure_terminal": value,
            },
        )
        self._terminal = value
        return value

    def released(
        self,
        prior: WorkspaceLifecycleRecord,
        terminal: WorkspaceLifecycleRecord,
    ) -> dict[str, Any]:
        value = self._terminal
        if (
            value is None
            or value["prior"] != prior.to_dict()
            or value["terminal"] != terminal.to_dict()
        ):
            raise ValueError("candidate lifecycle release has no exact predecessor")
        released = sealed(
            {
                "schema": RELEASED_SCHEMA,
                **self._binding,
                "terminal_receipt_id": value["receipt_id"],
                "compare_delete_succeeded": True,
            }
        )
        self._record_event(
            RELEASED_EVENT,
            {
                "task_id": self._binding["task_id"],
                "attempt": self._binding["attempt"],
                "closure_released": released,
            },
        )
        return released


def observe_provider_cleanup(*, repo_root, command_items, receipt_text):
    """Join exact owned command/log to a native terminal-only observation."""
    from ipfs_accelerate_py import agent_implementation_route as routes
    from ..runtime.provider_failure_policy import extract_grok_failure_receipts

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate route key")
            result[key] = value
        return result

    try:
        flag = "--agent-implementation-route-json"
        if (
            command_items.count(flag) != 1
            or "--agent-implementation-recovery-json" in command_items
        ):
            return None
        raw = command_items[command_items.index(flag) + 1]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 512 * 1024:
            return None
        binding = json.loads(raw, object_pairs_hook=unique)
        invocation = binding.get("invocation_binding")
        if not isinstance(invocation, Mapping):
            return None
        observed = routes.observe_agent_implementation_terminal_cleanup(
            store_path=invocation.get("provider_attempt_store"),
            expected_store_identity=invocation.get("provider_attempt_store_identity"),
            logical_attempt_id=invocation.get("logical_attempt_id"),
            repo_root=repo_root,
            max_age_ms=5 * 60 * 1000,
        )
        if type(observed) is not routes.AgentImplementationTerminalCleanupEvidence:
            return None
        evidence = json.loads(observed.evidence_json)
        if (
            evidence["route_binding"] != binding
            or extract_grok_failure_receipts(receipt_text)
            != (evidence["failure_receipt"],)
            or routes.extract_agent_implementation_route_outcomes(receipt_text)
            != (evidence["terminal_outcome"],)
            or routes.extract_agent_implementation_codex_capacity_receipts(receipt_text)
        ):
            return None
        return evidence["provider_cleanup"]
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RecursionError,
    ):
        return None
