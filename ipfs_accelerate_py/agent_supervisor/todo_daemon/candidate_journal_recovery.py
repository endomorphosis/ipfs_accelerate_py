"""Exact future candidate journal recovery; absent lifecycle rows prove nothing.

Observation never writes. Mutating resume is called only from the Bridge's
current, exact admitted attempt path; it cannot create a prepared operation.
No historical finish, preservation or release event is manufactured.
"""

from collections.abc import Mapping
from pathlib import Path

from ..merge.worktree_lifecycle import WorkspaceLifecycleRecord, WorktreeLifecycleStore
from .candidate_rejection_closure import (
    DISPOSITION_SCHEMA,
    JOURNAL_RELEASED_SCHEMA,
    JOURNAL_TERMINAL_SCHEMA,
    RECOVERED_CLOSURE_SCHEMA,
    RELEASED_EVENT,
    TERMINAL_EVENT,
    exact_seal,
    observe_provider_cleanup,
    rejection_validation,
    sealed,
    validate_lifecycle_pair,
)


def has_handoff(bridge, attempt) -> bool | None:
    """Negative-only classification: a future handoff must retain uncertainty.

    None means unavailable classification and must retain custody.
    This never grants deletion or callback authority. Paths are derived from
    this Bridge's exact attempt root, and even a malformed disposition retains
    custody once its bounded event chain identifies the future handoff schema.
    """
    from .database_portal_bridge import DatabasePortalBridgeError

    try:
        events = bridge._verified_event_chain(bridge._paths(attempt))
    except (
        DatabasePortalBridgeError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        RecursionError,
    ):
        return None
    return any(
        event.get("type") == TERMINAL_EVENT
        and event.get("task_id") == attempt.task_alias
        and event.get("attempt") == 1
        and isinstance(event.get("closure_terminal"), Mapping)
        and event["closure_terminal"].get("schema") == JOURNAL_TERMINAL_SCHEMA
        for event in events
    )


def _context(bridge, attempt):
    """Read the original, verified producer prefix and reobserve native cleanup."""
    from .implementation_daemon import (
        PROVIDER_CAPACITY_LOG_TAIL_BYTES,
        _stable_owned_log_tail,
    )

    if bridge.repository_root is None:
        return None
    paths, binding = bridge._recovery_attempt_binding(
        attempt, recovery_name="candidate deletion journal"
    )
    events = bridge._verified_event_chain(paths)
    selected = [
        [(i, e) for i, e in enumerate(events) if e.get("type") == kind]
        for kind in (
            "implementation_started",
            "implementation_proposal_validated",
            TERMINAL_EVENT,
        )
    ]
    if any(len(items) != 1 for items in selected):
        return None
    (si, started), (pi, proposed), (ti, terminal_event) = [
        items[0] for items in selected
    ]
    native = terminal_event.get("closure_terminal")
    if (
        not (si < pi < ti)
        or not exact_seal(native)
        or set(native)
        != {
            "schema",
            "task_id",
            "attempt",
            "worktree_path",
            "branch",
            "preserved_commit",
            "rescue_branch",
            "provider_cleanup",
            "completion_authority",
            "disposition",
            "prior",
            "terminal",
            "receipt_id",
        }
        or native.get("schema") != JOURNAL_TERMINAL_SCHEMA
        or not validate_lifecycle_pair(native.get("prior"), native.get("terminal"))
        or native.get("completion_authority") is not False
    ):
        return None
    alias = binding["task_alias"]
    number = getattr(attempt, "attempt_number", 0)
    projection = bridge._verify_projection(paths, binding)
    key, cid = bridge._portal_completion_event_identity(
        paths=paths, projection_text=projection, binding=binding
    )
    if (
        type(number) is not int
        or not 1 <= number < bridge.max_task_attempts
        or native.get("task_id") != alias
        or native.get("attempt") != 1
        or any(
            e.get("task_id") != alias
            or e.get("attempt") != 1
            or e.get("canonical_task_cid") != cid
            or e.get("canonical_task_key") != key
            for e in (started, proposed, terminal_event)
        )
    ):
        return None
    prior = native["prior"]
    if (
        prior.get("repo_root") != str(bridge.repository_root.resolve(strict=True))
        or prior.get("state_dir") != str(paths.state.parent.resolve(strict=True))
        or prior.get("task_id") != alias
        or prior.get("attempt") != 1
        or prior.get("canonical_task_cid") != cid
        or any(prior.get(k) != native.get(k) for k in ("branch",))
        or prior.get("workspace_path") != native.get("worktree_path")
        or native.get("worktree_path") != started.get("worktree_path")
        or native.get("branch") != started.get("branch")
        or not bridge._preserved_commit_exists(
            commit=native.get("preserved_commit"),
            rescue_branch=native.get("rescue_branch"),
        )
    ):
        return None
    disposition = native.get("disposition")
    if (
        not exact_seal(disposition)
        or set(disposition)
        != {
            "schema",
            "validation",
            "cleaned",
            "removed_worktree",
            "deleted_branch",
            "submodule_cleanup_count",
            "pooled",
            "queue_publication",
            "provider_dispatched",
            "attempt_consumed",
            "completion_authority",
            "receipt_id",
        }
        or disposition.get("schema") != DISPOSITION_SCHEMA
        or any(
            disposition.get(k) is not True
            for k in (
                "cleaned",
                "removed_worktree",
                "deleted_branch",
                "provider_dispatched",
                "attempt_consumed",
            )
        )
        or any(
            disposition.get(k) is not False
            for k in ("pooled", "queue_publication", "completion_authority")
        )
        or type(disposition.get("submodule_cleanup_count")) is not int
        or not 0 <= disposition["submodule_cleanup_count"] <= 10000
        or rejection_validation(disposition["validation"]) != disposition["validation"]
    ):
        return None
    proposal = disposition["validation"]["proposal_gate"]
    if any(proposed.get(k) != v for k, v in proposal.items()):
        return None
    # Later real producer events may exist. They cannot change this prefix-based
    # receipt, and contradictory effects or a second provider are never ignored.
    for event in events:
        if "merge_queued" in str(event.get("type")) or event.get("type") in {
            "implementation_merged",
            "task_completed",
            "implementation_completed",
        }:
            return None
    if any(
        e.get("type")
        in {
            RELEASED_EVENT,
            "failed_validation_worktree_preserved",
            "implementation_finished",
        }
        for e in events[:ti]
    ):
        return None
    tail = events[ti + 1 :]
    if any(
        e.get("type")
        not in {
            RELEASED_EVENT,
            "cleanup_finished",
            "failed_validation_worktree_preserved",
            "implementation_finished",
        }
        for e in tail
    ):
        return None
    for kind in (
        RELEASED_EVENT,
        "failed_validation_worktree_preserved",
        "implementation_finished",
    ):
        if sum(e.get("type") == kind for e in tail) > 1:
            return None
    for event in tail:
        if event.get("task_id") != alias or event.get("attempt") != 1:
            return None
        kind = event["type"]
        if kind == RELEASED_EVENT:
            released = event.get("closure_released")
            if (
                not exact_seal(released)
                or set(released)
                != {
                    "schema",
                    "task_id",
                    "attempt",
                    "worktree_path",
                    "branch",
                    "preserved_commit",
                    "rescue_branch",
                    "provider_cleanup",
                    "completion_authority",
                    "deletion",
                    "terminal_receipt_id",
                    "compare_delete_succeeded",
                    "receipt_id",
                }
                or released.get("schema") != JOURNAL_RELEASED_SCHEMA
                or released.get("terminal_receipt_id") != native["receipt_id"]
                or released.get("compare_delete_succeeded") is not True
                or any(
                    released.get(k) != native.get(k)
                    for k in (
                        "task_id",
                        "attempt",
                        "worktree_path",
                        "branch",
                        "preserved_commit",
                        "rescue_branch",
                        "provider_cleanup",
                        "completion_authority",
                    )
                )
            ):
                return None
            continue
        cleanup = event if kind == "cleanup_finished" else event.get("cleanup_result")
        if not _cleanup_matches(cleanup, native):
            return None
        if kind == "failed_validation_worktree_preserved" and (
            event.get("preserved") is not True
            or any(
                event.get(k) != native[k] for k in ("preserved_commit", "rescue_branch")
            )
            or rejection_validation(event.get("validation_result", {}))
            != disposition["validation"]
        ):
            return None
        if kind == "implementation_finished":
            preservation = event.get("failed_preservation_result")
            merge, board = event.get("merge_result"), event.get("board_completion")
            if (
                event.get("canonical_task_key") != key
                or event.get("canonical_task_cid") != cid
                or event.get("returncode") != 78
                or event.get("provider_dispatched") is not True
                or event.get("attempt_consumed") is not True
                or event.get("protected_path_violation") not in (None, {})
                or rejection_validation(event.get("validation_result", {}))
                != disposition["validation"]
                or not isinstance(merge, Mapping)
                or merge.get("merged") is not False
                or merge.get("queued") is True
                or (isinstance(board, Mapping) and board.get("complete") is True)
                or not isinstance(preservation, Mapping)
                or preservation.get("preserved") is not True
                or preservation.get("cleanup_result") != cleanup
                or any(
                    preservation.get(k) != native[k]
                    for k in ("preserved_commit", "rescue_branch")
                )
            ):
                return None
    command = started.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(v, str) for v in command)
    ):
        return None
    log = Path(str(started.get("log_path") or ""))
    log.resolve(strict=True).relative_to(paths.implementation_logs.resolve(strict=True))
    _, text = _stable_owned_log_tail(
        log, PROVIDER_CAPACITY_LOG_TAIL_BYTES, reject_group_writable=True
    )
    proof = observe_provider_cleanup(
        repo_root=bridge.repository_root, command_items=command, receipt_text=text
    )
    if (
        not isinstance(proof, Mapping)
        or proof != native.get("provider_cleanup")
        or proof.get("task_revision_cid") != cid
        or proof.get("task_id") != alias
        or proof.get("attempt") != 1
        or proof.get("workspace_path") != native.get("worktree_path")
    ):
        return None
    return {
        "binding": binding,
        "native": native,
        "proof": dict(proof),
        "event_ids": [e["event_id"] for e in (started, proposed, terminal_event)],
        "event_chain_tip": events[-1]["event_id"],
        "release_observations": [
            e["closure_released"]["deletion"]
            for e in tail
            if e["type"] == RELEASED_EVENT
        ],
    }


def _cleanup_matches(cleanup, native):
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("cleaned") is not True
        or cleanup.get("removed_worktree") is not True
        or cleanup.get("deleted_branch") is not True
        or cleanup.get("pooled") is True
        or cleanup.get("pool_release") is not None
        or cleanup.get("branch") != native["branch"]
        or cleanup.get("worktree_path") != native["worktree_path"]
    ):
        return False
    finalize = cleanup.get("lifecycle_finalize")
    return (
        isinstance(finalize, Mapping)
        and finalize.get("finalized") is True
        and finalize.get("terminal_callback") == native
    )


def _observe(bridge, context):
    # Canonical native store comes from the bound repository, never a receipt path.
    store = WorktreeLifecycleStore(repo_root=bridge.repository_root)
    native = context["native"]
    terminal = WorkspaceLifecycleRecord.from_dict(native["terminal"])
    observed = store.observe_candidate_deletion(
        terminal, handoff_receipt_id=native["receipt_id"]
    )
    if observed is None:
        return None
    evidence = observed.to_dict()
    if any(value != evidence for value in context["release_observations"]):
        return None
    return evidence


def _receipt(attempt, context, evidence):
    native, binding = context["native"], context["binding"]
    return sealed(
        {
            "schema": RECOVERED_CLOSURE_SCHEMA,
            "reason": "proposal_gate_failed",
            **{
                k: getattr(attempt, k)
                for k in (
                    "attempt_id",
                    "claim_id",
                    "lease_id",
                    "owner_session_id",
                    "task_cid",
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                )
            },
            "task_alias": binding["task_alias"],
            "binding_id": binding["binding_id"],
            "task_contract_digest": binding["task_contract_digest"],
            "repository_tree_id": binding["repository_tree_id"],
            "provider_cleanup": context["proof"],
            "terminal_receipt_id": native["receipt_id"],
            "deletion": evidence,
            "disposition_receipt_id": native["disposition"]["receipt_id"],
            "event_ids": context["event_ids"],
            "preserved_commit": native["preserved_commit"],
            "rescue_branch": native["rescue_branch"],
            "provider_dispatched": True,
            "attempt_consumed": True,
            "completion_authority": False,
            "queue_publication": False,
        }
    )


def observe(bridge, attempt):
    context = _context(bridge, attempt)
    if context is None:
        return None
    evidence = _observe(bridge, context)
    if (
        evidence is None
        or _context(bridge, attempt) != context
        or _observe(bridge, context) != evidence
    ):
        return None
    return _receipt(attempt, context, evidence)


def resume(bridge, attempt, *, protect_attempt_write):
    """Explicit mutation; caller admission is repeated immediately before resume."""
    context = _context(bridge, attempt)
    if context is None:
        return None
    native = context["native"]
    protect_attempt_write()
    store = WorktreeLifecycleStore(repo_root=bridge.repository_root)
    observed = store.resume_candidate_observed_delete(
        WorkspaceLifecycleRecord.from_dict(native["terminal"]),
        handoff_receipt_id=native["receipt_id"],
    )
    protect_attempt_write()
    if (
        _context(bridge, attempt) != context
        or _observe(bridge, context) != observed.to_dict()
    ):
        return None
    return _receipt(attempt, context, observed.to_dict())
