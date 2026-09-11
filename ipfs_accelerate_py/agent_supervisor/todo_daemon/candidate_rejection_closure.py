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
JOURNAL_TERMINAL_SCHEMA = "candidate-rejection-lifecycle-terminal@2"
JOURNAL_RELEASED_SCHEMA = "candidate-rejection-lifecycle-released@2"
RECOVERED_CLOSURE_SCHEMA = "database-portal-candidate-rejection-journal-closure@1"
DISPOSITION_SCHEMA = "candidate-rejection-cleanup-disposition@1"
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
        journal_validation: Mapping[str, Any] | None = None,
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
        self._journal_validation = (
            rejection_validation(journal_validation)
            if journal_validation is not None
            else None
        )
        self._disposition: dict[str, Any] | None = None

    @property
    def journal_enabled(self) -> bool:
        return self._journal_validation is not None

    def capture_cleanup_disposition(self, cleanup: Mapping[str, Any]) -> None:
        """Capture only the completed nonpooled producer branch, before deletion."""
        if (
            not self.journal_enabled
            or self._disposition is not None
            or self._terminal is not None
            or cleanup.get("cleaned") is not True
            or cleanup.get("removed_worktree") is not True
            or cleanup.get("deleted_branch") is not True
            or cleanup.get("pooled") is True
            or cleanup.get("pool_release") is not None
            or cleanup.get("branch") != self._binding["branch"]
            or cleanup.get("worktree_path") != self._binding["worktree_path"]
            or cleanup.get("error")
            or not isinstance(cleanup.get("submodule_cleanup"), list)
        ):
            raise CandidateClosureObservationUnknown(
                "candidate cleanup disposition unavailable"
            )
        self._disposition = sealed(
            {
                "schema": DISPOSITION_SCHEMA,
                "validation": self._journal_validation,
                "cleaned": True,
                "removed_worktree": True,
                "deleted_branch": True,
                "submodule_cleanup_count": len(cleanup["submodule_cleanup"]),
                "pooled": False,
                "queue_publication": False,
                "provider_dispatched": True,
                "attempt_consumed": True,
                "completion_authority": False,
            }
        )

    def terminal(
        self,
        prior: WorkspaceLifecycleRecord,
        terminal: WorkspaceLifecycleRecord,
    ) -> dict[str, Any]:
        before, after = prior.to_dict(), terminal.to_dict()
        cleanup = self._binding["provider_cleanup"]
        if (
            self._terminal is not None
            or (self.journal_enabled and self._disposition is None)
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
                "schema": JOURNAL_TERMINAL_SCHEMA
                if self.journal_enabled
                else TERMINAL_SCHEMA,
                **self._binding,
                **({"disposition": self._disposition} if self.journal_enabled else {}),
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
        *,
        deletion=None,
    ) -> dict[str, Any]:
        value = self._terminal
        if (
            value is None
            or value["prior"] != prior.to_dict()
            or value["terminal"] != terminal.to_dict()
        ):
            raise ValueError("candidate lifecycle release has no exact predecessor")
        observed = None
        if self.journal_enabled:
            from ..merge.worktree_lifecycle_delete_journal import (
                CandidateObservedDeletion,
            )

            if type(deletion) is not CandidateObservedDeletion:
                raise CandidateClosureObservationUnknown(
                    "native deletion observation required"
                )
            observed = deletion.to_dict()
            if (
                observed["prepared"]["binding"]["terminal"] != terminal.to_dict()
                or observed["committed"]["handoff_receipt_id"] != value["receipt_id"]
            ):
                raise CandidateClosureObservationUnknown(
                    "native deletion handoff mismatch"
                )
        released = sealed(
            {
                "schema": JOURNAL_RELEASED_SCHEMA
                if self.journal_enabled
                else RELEASED_SCHEMA,
                **self._binding,
                **({"deletion": observed} if self.journal_enabled else {}),
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


def rejection_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Closed, bounded rejected-proposal summary; it never supplies authority."""
    if not isinstance(value, Mapping):
        raise TypeError("candidate validation must be a mapping")
    proposal = value.get("proposal_gate")
    if (
        value.get("passed") is not False
        or value.get("returncode") != 78
        or value.get("reason") != "proposal_gate_failed"
        or not isinstance(proposal, Mapping)
        or proposal.get("attempted") is not True
        or proposal.get("accepted") is not False
        or any(
            not isinstance(proposal.get(k), str) or not 1 <= len(proposal[k]) <= 1024
            for k in ("receipt_id", "proposal_id", "policy_id")
        )
        or not isinstance(proposal.get("reason_codes"), list)
        or len(proposal["reason_codes"]) > 32
        or any(not isinstance(k, str) or len(k) > 128 for k in proposal["reason_codes"])
    ):
        raise ValueError("candidate journal requires exact proposal rejection")
    return {
        "passed": False,
        "returncode": 78,
        "reason": "proposal_gate_failed",
        "proposal_gate": {
            k: proposal[k]
            for k in (
                "attempted",
                "accepted",
                "receipt_id",
                "proposal_id",
                "policy_id",
                "reason_codes",
            )
        },
    }


def journal_handoff(terminal_callback, released_callback):
    """Only the exact future producer object can select native journal deletion."""
    value = getattr(terminal_callback, "__self__", None)
    if type(value) is not CandidateLifecycleHandoff or not value.journal_enabled:
        return None
    if (
        getattr(terminal_callback, "__func__", None)
        is not CandidateLifecycleHandoff.terminal
        or getattr(released_callback, "__self__", None) is not value
        or getattr(released_callback, "__func__", None)
        is not CandidateLifecycleHandoff.released
    ):
        raise CandidateClosureObservationUnknown("candidate journal callback mismatch")
    return value


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
