"""Reverify one historical Native checkpoint whose identity cache was empty.

This bridge appends current, explicitly labelled context verification for code
Native already committed and preserved. It does not rewrite the original event,
request, receipt, runtime configuration, claim, lifecycle, board or accounting.
The caller retains the same launcher/checkout/queue guards as the original API.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import re
import sys

from . import pending_task_claim_recovery as legacy

SCHEMA = "pending-candidate-context-reverification@1"
LEGACY_COMMIT = "70f436ba1cbf8d6b62efe8de2d538489918c6549"
LEGACY_SHA256 = "c3da1853b069e404424b2c6c55800d48b8bc02322965e62745b5db64afa3cd0f"
_IDENTITY = ("canonical_task_cid", "canonical_task_key", "board_namespace")


def _legacy_source():
    path = Path(legacy.__file__)
    relative = "ipfs_accelerate_py/agent_supervisor/todo_daemon/pending_task_claim_recovery.py"
    root = path.parents[3]
    legacy._need(path.is_absolute() and path.resolve() == path
        and legacy.__name__ == __package__ + ".pending_task_claim_recovery"
        and sys.modules.get(legacy.__name__) is legacy
        and legacy._sha(legacy._read(path)) == LEGACY_SHA256
        and legacy._git(root, "rev-parse", LEGACY_COMMIT + ":" + relative)
            == "f21c72c81b15efbabb4565519b7afa69d2c37657"
        and legacy._git(root, "rev-parse", "HEAD:" + relative)
            == "f21c72c81b15efbabb4565519b7afa69d2c37657", "original_sdk70_source_required")
    return {"path": str(path), "sha256": LEGACY_SHA256, "commit": LEGACY_COMMIT}


def _context(daemon, task_id):
    native = legacy._ordinary(daemon)
    source = _legacy_source()
    path = legacy._candidate_receipt_path(daemon, task_id)
    prior = legacy._candidate_prior(path)
    legacy._need(prior is not None and prior["phase"] == "prepared"
                 and not prior.get("preserved_candidate"), "original_prepared_checkpoint_required")
    request = prior["request"]; base = request["base"]
    legacy._need(request.get("schema") == legacy.CANDIDATE_SCHEMA and base["task_id"] == task_id,
                 "context_request_scope_changed")
    # The original strict checker is used unchanged. In particular, no missing
    # lifecycle fallback or runtime/source/claim guard is overridden here.
    life = daemon.worktree_lifecycle.load_workspace(base["workspace"])
    legacy._need(life is not None and life.to_dict() == prior["terminal_lifecycle"],
                 "original_terminal_custody_required")
    legacy._candidate_stable(daemon, request, prior)
    for name in ("lifecycle", "lifecycle_index", "claim"):
        pin = base["pins"][name]
        legacy._need(legacy._pin(Path(pin["path"])) == pin, "original_" + name + "_changed")
    legacy._need(not Path(base["workspace"]).exists()
                 and not daemon._worktree_path_registered_in_repo(daemon.repo_root, Path(base["workspace"]))
                 and not daemon._git_ref_exists(life.branch), "original_native_cleanup_required")
    rows = legacy._event_rows(legacy._read(daemon.events_path))
    expected_evidence = {"operation_id": prior["operation_id"],
        "candidate_manifest_sha256": legacy._sha(legacy._bytes(request["candidate"])), **legacy._FALSE}
    original = [row for row in rows if row.get("type") == "implementation_pending_candidate_preserved"
        and row.get("task_id") == task_id and type(row.get("attempt")) is int and row["attempt"] == base["attempt"]
        and row.get("pending_candidate_recovery") == expected_evidence
        and not any(key in row for key in _IDENTITY)]
    legacy._need(len(original) == 1, "exact_unenriched_native_checkpoint_required")
    original = original[0]
    accepted = next(row for row in rows if row["event_id"] == request["candidate"]["accepted_event_id"])
    start = next(row for row in rows if row["event_id"] == base["start_event_id"])
    proposal_types = {"implementation_proposal_validated", "implementation_proposal_receipt_reused",
                      "implementation_proposal_rejected"}
    decisions = [row for row in rows if row.get("type") in proposal_types
        and row.get("task_id") == task_id and row.get("canonical_task_cid") == base["task_cid"]
        and row.get("sequence", 0) > start["sequence"]]
    legacy._need(accepted.get("type") in proposal_types - {"implementation_proposal_rejected"}
        and accepted.get("accepted") is True and accepted.get("attempted") is True
        and not accepted.get("reason_codes") and accepted.get("task_id") == task_id
        and accepted.get("canonical_task_cid") == base["task_cid"]
        and accepted.get("canonical_task_key") == base["task_key"]
        and accepted.get("repository_tree_id") == base["baseline_ref"]
        and accepted.get("proposal_id") == request["candidate"]["proposal_id"]
        and ("attempt" not in accepted or type(accepted["attempt"]) is int and accepted["attempt"] == base["attempt"])
        and decisions and decisions[-1]["event_id"] == accepted["event_id"],
        "latest_accepted_checkpoint_proposal_required")
    ref = daemon._interrupted_worktree_rescue_branch_name(life.branch,
        "pending-candidate-" + prior["operation_id"].removeprefix("sha256:")[:20])
    legacy._need(original.get("preserved") is True and original.get("branch") == life.branch
        and original.get("worktree_path") == base["workspace"] and original.get("rescue_branch") == ref
        and re.fullmatch(r"[a-f0-9]{40}", str(original.get("preserved_commit"))) is not None
        and original["sequence"] > accepted["sequence"], "native_checkpoint_scope_changed")
    preserved = {"commit": original["preserved_commit"], "rescue_ref": ref}
    legacy._candidate_commit(daemon, request, preserved)
    cleanup = [row for row in rows if row.get("type") == "cleanup_finished"
        and row.get("sequence", 0) > original["sequence"] and row.get("branch") == life.branch
        and row.get("worktree_path") == base["workspace"] and row.get("cleaned") is True
        and row.get("removed_worktree") is True and row.get("deleted_branch") is True]
    legacy._need(bool(cleanup), "native_checkpoint_cleanup_history_required")
    bridge_source = Path(__file__).resolve()
    return {"schema": SCHEMA, "task_id": task_id, "operation_id": prior["operation_id"],
        "sdk_request_sha256": legacy._sha(legacy._bytes(request)), "sdk_receipt_path": str(path),
        "sdk_receipt_sha256": legacy._sha(legacy._read(path)), "legacy_source": source,
        "bridge_source": {"path": str(bridge_source), "sha256": legacy._sha(legacy._read(bridge_source))},
        "original_event_id": original["event_id"], "cleanup_event_id": cleanup[-1]["event_id"],
        "preserved": preserved, "canonical_task_cid": base["task_cid"],
        "canonical_task_key": base["task_key"], "board_namespace": base["board_namespace"],
        "state_sha256": legacy._state_hash(daemon), **legacy._FALSE}, original


def review(daemon, *, task_id: str) -> dict:
    """Dry review of exact SDK70/Native custody; no files or events published."""
    return _context(daemon, task_id)[0]


def _projections(rows, request, original):
    context = {"schema": SCHEMA, "review": request, "original_event_id": original["event_id"],
        "new_preservation_performed": False, "canonical_context_reverified": True, **legacy._FALSE}
    values = [row for row in rows if type(row.get("canonical_context_reverification")) is dict
              and row["canonical_context_reverification"].get("review") == request]
    for row in values:
        legacy._need(row.get("type") == "implementation_pending_candidate_preserved"
            and row.get("canonical_context_reverification") == context
            and all(row.get(key) == original[key] for key in ("task_id", "attempt", "branch", "worktree_path",
                "preserved", "rescue_branch", "preserved_commit", "pending_candidate_recovery"))
            and all(row.get(key) == request[key] for key in _IDENTITY), "canonical_context_projection_unproved")
    legacy._need(len(values) <= 1, "canonical_context_projection_ambiguous")
    return values


def reconcile(daemon, *, request: Mapping) -> dict:
    """Append current canonical context proof; leave SDK70 to resume its API."""
    result = {"reverified": False, "blocked": True, "provider_outcome": "unknown", **legacy._FALSE}
    try:
        legacy._need(isinstance(request, Mapping), "context_request_required")
        request = deepcopy(dict(request))
        native = legacy._ordinary(daemon)
        expected, _ = _context(daemon, request["task_id"])
        legacy._need(expected == request, "context_review_changed")
        path = Path(request["sdk_receipt_path"])
        prior = legacy._candidate_prior(path)
        base = prior["request"]["base"]
        with ExitStack() as stack:
            for lock in (daemon._implementation_lock_path(), path.parent / "recovery.lock",
                         Path(base["pins"]["claim"]["path"]), daemon.strategy_path):
                stack.enter_context(native.serialized_lock_update(lock))
            fresh, original = _context(daemon, request["task_id"])
            legacy._need(fresh == request, "context_review_changed")
            rows = legacy._event_rows(legacy._read(daemon.events_path))
            projections = _projections(rows, request, original)
            if not projections:
                payload = {key: original[key] for key in ("task_id", "attempt", "branch", "worktree_path",
                    "preserved", "rescue_branch", "preserved_commit", "pending_candidate_recovery")}
                payload.update({key: request[key] for key in _IDENTITY})
                payload["canonical_context_reverification"] = {"schema": SCHEMA, "review": request,
                    "original_event_id": original["event_id"], "new_preservation_performed": False,
                    "canonical_context_reverified": True, **legacy._FALSE}
                # The already proven Native checkpoint remains preserved.
                # This is a current verification projection, never a copy of
                # its old timestamp, commit operation or validation outcome.
                daemon._record_event("implementation_pending_candidate_preserved", payload)
                projections = _projections(legacy._event_rows(legacy._read(daemon.events_path)), request, original)
            legacy._need(len(projections) == 1 and all(projections[0].get(k) == request[k] for k in _IDENTITY),
                         "canonical_context_projection_unproved")
            legacy._need(_context(daemon, request["task_id"])[0] == request, "context_changed_after_projection")
            return {**result, "reverified": True, "blocked": False,
                "reason": "legacy_pending_candidate_context_reverified", "event_id": projections[0]["event_id"],
                "original_event_id": original["event_id"], "preserved": request["preserved"]}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
        return {**result, "reason": str(exc) if type(exc) is ValueError else type(exc).__name__}
