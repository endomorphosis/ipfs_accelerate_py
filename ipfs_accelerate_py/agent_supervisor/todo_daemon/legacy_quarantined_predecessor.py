"""Reconcile one quarantined predecessor before a legacy verification retry.

This separate profile verifies a completed queued implementation, its failed
dirty-checkout merges and recorded quarantine reconciliation, followed by an
uncommitted verification timeout. The original single-lifecycle recovery still
rejects all merge history. These observations grant neither retry nor task
completion; the operator must retain native maintenance custody and authorize
the bounded task CAS independently, preserving the old candidates and workspace.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import re
import subprocess
from typing import Mapping

from .database_portal_bridge import DatabasePortalBridgeError, _bounded_file, _strict_json_bytes
from .legacy_verification_retry import REASON, _digest, _inspect_legacy_verification_retry

SCHEMA = "ipfs_accelerate_py/agent-supervisor/legacy-quarantined-predecessor-retry-evidence@1"
_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _require(value, reason):
    if not value:
        raise DatabasePortalBridgeError("legacy predecessor " + reason)


def _failed_merge_facts(merge):
    _require(isinstance(merge, Mapping), "merge outcome is missing")
    _require(merge.get("attempted") is True and merge.get("merged") is False
             and merge.get("merge_commit") == "" and type(merge.get("returncode")) is int
             and merge["returncode"] == 2, "merge has an uncertain or applied outcome")
    for field in ("submodule_merge_results", "generated_submodule_reconciliation",
                  "identical_untracked_paths", "resolved_generated_conflicts",
                  "restored_generated_dirty_overlap", "restored_incidental_gitlinks"):
        _require(merge.get(field) == [], "merge contains unaccounted checkout changes")
    resolver = merge.get("llm_merge_resolver")
    _require(isinstance(resolver, Mapping) and resolver.get("applied") is False
             and resolver.get("command") == [] and resolver.get("llm_timeout") is False
             and type(resolver.get("llm_returncode")) is int and resolver["llm_returncode"] == 2
             and resolver.get("reason") == "main_checkout_dirty_conflict",
             "resolver outcome is unknown or applied")
    keys = ("branch", "target_branch", "main_worktree_path", "started_at", "finished_at",
            "dirty_paths", "llm_merge_resolver")
    _require(all(key in merge for key in keys) and bool(merge["dirty_paths"]),
             "merge identity is incomplete")
    return {key: merge[key] for key in keys}


def inspect_predecessor_events(events, *, identity, alias):
    """Consume the full, already verified chain from the shared inspector."""
    ordered = ("implementation_started", "merge_candidate_enqueued", "implementation_pending_merge",
               "implementation_finished", "merge_reconciled", "implementation_started", REASON,
               "protected_path_verification_deferred_worktree_retained", "implementation_finished")
    selected = [e for e in events if e.get("type") in ordered]
    _require([e["type"] for e in selected] == list(ordered), "provider/candidate population differs")
    started, queued, pending, finished, reconciled, tail, *_ = selected
    prefix = selected[:5]
    for event in prefix:
        _require(event.get("task_id") == alias
                 and event.get("canonical_task_cid") == identity["portal_canonical_task_cid"]
                 and event.get("canonical_task_key") == identity["portal_canonical_task_key"]
                 and type(event.get("attempt")) is int and event["attempt"] == 1,
                 "event task/attempt binding differs")
    request, commit, branch = queued.get("request_id"), queued.get("implementation_commit"), queued.get("branch")
    _require(isinstance(request, str) and 0 < len(request) <= 256
             and isinstance(commit, str) and _SHA.fullmatch(commit)
             and isinstance(branch, str) and branch.startswith("implementation/") and len(branch) <= 1024,
             "candidate identity is invalid")
    _require(all(e.get("implementation_commit") == commit and e.get("branch") == branch
                 for e in (pending, finished, reconciled)) and started.get("branch") == branch,
             "candidate identity changed")
    _require(queued.get("queued") is True and queued.get("merged") is False
             and queued.get("attempted") is False and queued.get("reason") == "merge_queued",
             "enqueue outcome differs")
    target = {key: queued.get(key) for key in ("target_repository_id", "target_branch", "queue_dir")}
    _require(all(isinstance(v, str) and v and len(v) <= 4096 for v in target.values()),
             "enqueue target is incomplete")
    for event in (pending, finished):
        merge, board = event.get("merge_result"), event.get("board_completion")
        _require(isinstance(merge, Mapping) and merge.get("request_id") == request
                 and merge.get("queued") is True and merge.get("merged") is False
                 and merge.get("reason") == "merge_queued"
                 and merge.get("target_branch") == target["target_branch"]
                 and isinstance(board, Mapping) and board.get("complete") is False
                 and board.get("pending_merge") is True, "queued lifecycle outcome differs")
    _require(finished.get("provider_dispatched") is True and finished.get("attempt_consumed") is True
             and type(finished.get("returncode")) is int and finished["returncode"] == 0
             and all(finished.get(k) is None for k in ("termination_result", "timeout_result", "exception_result")),
             "provider finish is unknown")
    _require(started.get("baseline_ref") == queued.get("baseline_ref") == finished.get("baseline_ref")
             and started.get("worktree_path") == finished.get("worktree_path"),
             "provider workspace identity differs")
    merge = reconciled.get("merge_result")
    _require(reconciled.get("resolved") is True and reconciled.get("request_status") == "quarantined"
             and reconciled.get("reason") == "stale_quarantined_merge"
             and reconciled.get("failure_reason") == "changed_submodule_merge_unverified"
             and reconciled.get("request_id") == request and isinstance(merge, Mapping)
             and merge.get("request_id") == request and merge.get("attempted") is False
             and merge.get("merged") is False and merge.get("reason") == "stale_quarantined_merge",
             "quarantine reconciliation is absent or uncertain")
    merge_events = [e for e in events if e.get("type") == "merge_finished"]
    _require(1 <= len(merge_events) <= 8, "merge failure population exceeds its bound")
    facts = []
    for event in merge_events:
        _require(finished["sequence"] < event["sequence"] < reconciled["sequence"]
                 and event.get("reason") == "main_checkout_dirty_conflict"
                 and event.get("branch") == branch and event.get("target_branch") == target["target_branch"],
                 "merge event is outside its predecessor lifecycle")
        facts.append(_digest(_failed_merge_facts(event)))
    allowed = {"merge_candidate_enqueued", "merge_finished", "merge_reconciliation_deferred", "merge_reconciled"}
    for event in events:
        kind = str(event.get("type", ""))
        if kind.startswith(("merge", "post_merge", "pending_merge_reconciliation")):
            _require(kind in allowed and event["sequence"] < tail["sequence"],
                     "unexpected merge or callback history")
            if kind == "merge_reconciliation_deferred":
                _require(finished["sequence"] < event["sequence"] < reconciled["sequence"]
                         and event.get("reason") == "main_checkout_dirty", "unrecognized reconciliation deferral")
    return {"request_id": request, "commit_sha": commit, "branch_name": branch,
            "canonical_task_cid": identity["portal_canonical_task_cid"],
            "canonical_task_key": identity["portal_canonical_task_key"], **target,
            "event_ids": [e["event_id"] for e in prefix],
            "merge_event_ids": [e["event_id"] for e in merge_events], "failed_merge_facts": facts,
            "queue_settlement_verified": False, "completion_authority": False}


def inspect_reconciled_legacy_verification_retry(task_row, **arguments):
    return _inspect_legacy_verification_retry(task_row, **arguments, reconciled_predecessor=True)


def inspect_legacy_retry_profile(task_row, **arguments):
    """Choose a fully verified profile; no generic retry on inspection failure."""
    from .legacy_verification_retry import inspect_legacy_verification_retry
    try:
        return inspect_legacy_verification_retry(task_row, **arguments)
    except DatabasePortalBridgeError:
        return inspect_reconciled_legacy_verification_retry(task_row, **arguments)


@contextmanager
def hold_legacy_verification_retry_observation(
    task_row, *, task_projection, allowed_attempt_root, retained_worktree_root,
    expected_task_revision, queue_dir, target_repository_id, target_branch, repository_root,
):
    """Compose the selected event profile with its native queue guard for a CAS."""
    from .legacy_verification_retry import hold_legacy_retry_queue_absence
    evidence = inspect_legacy_retry_profile(task_row, task_projection=task_projection,
        allowed_attempt_root=allowed_attempt_root, retained_worktree_root=retained_worktree_root,
        expected_task_revision=expected_task_revision)
    arguments = dict(queue_dir=queue_dir,target_repository_id=target_repository_id,
                     target_branch=target_branch,evidence=evidence)
    if evidence["schema"] == SCHEMA:
        guard = hold_reconciled_legacy_retry_queue(repository_root=repository_root, **arguments)
    else:
        guard = hold_legacy_retry_queue_absence(**arguments)
    with guard as queue_evidence:
        yield evidence, queue_evidence


@contextmanager
def hold_reconciled_legacy_retry_queue(*, queue_dir: Path, target_repository_id: str,
                                      target_branch: str, repository_root: Path,
                                      evidence: Mapping):
    """Read the retained quarantine while holding the native queue writer guard.

    The caller must also retain native process/source maintenance custody and
    independently authorize the bounded task CAS. No grants or live stores are
    created here. An unavailable observation never means an empty queue.
    """
    import duckdb
    from ..merge.merge_queue import (hold_merge_queue_settlement, connect_duckdb_with_policy,
                                    _settlement_regular_file_identity)
    from ..merge.checkout_lock import checkout_repository_id
    from .database_portal_bridge import verify_database_portal_attempt_projection_identity

    material = dict(evidence)
    _require(material.pop("evidence_id", None) == _digest(material) and material.get("schema") == SCHEMA,
             "queue guard requires verified evidence")
    prior = material["predecessor"]
    _require(prior["queue_dir"] == str(queue_dir) and prior["target_repository_id"] == target_repository_id
             and prior["target_branch"] == target_branch
             and checkout_repository_id(repository_root) == target_repository_id, "queue target differs")
    identity = material["projection_identity"]

    def revalidate():
        _require(verify_database_portal_attempt_projection_identity(Path(identity["projection_path"])) == identity,
                 "immutable projection changed")
        _require("sha256:" + hashlib.sha256(_bounded_file(Path(material["events_path"]),
                    limit=8 * 1024 * 1024)).hexdigest() == material["events_digest"], "event history changed")

    revalidate()
    with hold_merge_queue_settlement(queue_dir, target_repository_id=target_repository_id,
                                    target_branch=target_branch) as settlement:
        _require(settlement["row_count"] <= 8192, "queue population exceeds its bound")
        database = Path(settlement["database"]["path"])
        before = _settlement_regular_file_identity(database, label="database")
        _require(before == {k: settlement["database"][k] for k in before}, "queue identity changed")
        connection = connect_duckdb_with_policy(duckdb, database, read_only=True)
        try:
            rows = connection.execute("""SELECT request_id,
                CASE WHEN octet_length(encode(task_id)) <= 1024 THEN task_id END,
                CASE WHEN octet_length(encode(canonical_task_id)) <= 1024 THEN canonical_task_id END,
                CASE WHEN octet_length(encode(bindings)) <= 65536 THEN bindings END
                FROM (SELECT request_id, task_id, canonical_task_id,
                    CAST(json_extract(metadata_json, '$.completion_task_cids') AS VARCHAR) AS bindings
                    FROM merge_requests) ORDER BY request_id LIMIT 8193""").fetchall()
            _require(len(rows) == settlement["row_count"], "queue population changed")
            matches = []
            for request, alias, cid, bindings in rows:
                _require(isinstance(alias, str) and alias and isinstance(cid, str) and cid
                         and isinstance(bindings, str), "queue task binding is missing")
                binding = _strict_json_bytes(bindings.encode(), noun="legacy predecessor queue bindings")
                _require(isinstance(binding, dict) and binding.get(alias) == cid
                         and all(isinstance(k,str) and k and isinstance(v,str) and v for k,v in binding.items()),
                         "queue task binding is invalid")
                cids = {material["task_cid"], prior["canonical_task_cid"]}
                if (alias == material["task_alias"] or cid in cids or material["task_alias"] in binding
                        or cids.intersection(binding.values())):
                    matches.append(request)
            _require(matches == [prior["request_id"]], "queue has additional or missing task candidates")
            row = connection.execute("""SELECT branch_name, commit_sha, canonical_task_id, canonical_task_key,
                status, claim_token, consumer_id, failure_count, claim_generation,
                CASE WHEN octet_length(encode(metadata_json)) <= 524288 THEN metadata_json END,
                finished_at FROM merge_requests WHERE request_id=?""", [prior["request_id"]]).fetchone()
            if row is not None:
                row = tuple(row[i] for i in range(len(row)))
        finally:
            connection.close()
        _require(row is not None and row[:4] == (prior["branch_name"], prior["commit_sha"],
                    prior["canonical_task_cid"], prior["canonical_task_key"])
                 and row[4:7] == ("quarantined", "", "") and type(row[7]) is int
                 and row[7] == len(prior["failed_merge_facts"]) and type(row[8]) is int and row[8] > 0
                 and isinstance(row[9],str) and type(row[10]) in (int,float)
                 and math.isfinite(row[10]) and row[10] > 0, "queue claim or quarantine identity differs")
        metadata = _strict_json_bytes(row[9].encode(), noun="legacy predecessor metadata")
        _require(metadata.get("target_repository_id") == target_repository_id
                 and metadata.get("target_branch") == target_branch
                 and metadata.get("events_path") == material["events_path"]
                 and metadata.get("completion_task_cids") == {material["task_alias"]:prior["canonical_task_cid"]}
                 and all(k not in metadata for k in ("completion", "completion_receipt", "post_merge_completion")),
                 "queue metadata target or completion differs")
        failures, terminal = metadata.get("failure_metadata"), metadata.get("quarantine")
        _require(isinstance(failures,list) and len(failures) + 1 == row[7]
                 and isinstance(terminal,dict), "queue failure history is incomplete")
        for index, (failure, digest) in enumerate(zip([*failures, terminal],prior["failed_merge_facts"]),1):
            _require(isinstance(failure,dict) and failure.get("request_id") == prior["request_id"]
                     and failure.get("commit_sha") == prior["commit_sha"]
                     and failure.get("canonical_task_id") == prior["canonical_task_cid"]
                     and type(failure.get("failure_count")) is int and failure["failure_count"] == index
                     and all(failure.get(k) is False for k in ("accepted", "integrated", "acceptance_pending", "merged"))
                     and failure.get("reason") == "changed_submodule_merge_unverified"
                     and failure.get("status") == ("quarantined" if index == row[7] else "retrying"),
                     "queue failure outcome is uncertain or differs")
            _require(_digest(_failed_merge_facts(failure.get("merge_result"))) == digest,
                     "queue and event merge outcomes differ")
        result = subprocess.run(['git','rev-parse','--verify','refs/heads/'+prior['branch_name']],
            cwd=repository_root,capture_output=True,text=True,timeout=10,check=False)
        _require(result.returncode == 0 and result.stdout.strip() == prior["commit_sha"],
                 "retained candidate branch is absent or moved")
        result = subprocess.run(['git','rev-parse','--verify',prior['commit_sha']+'^{tree}'],
            cwd=repository_root,capture_output=True,text=True,timeout=10,check=False)
        _require(result.returncode == 0 and result.stdout.strip() == metadata.get("candidate_tree"),
                 "retained candidate tree differs")
        revalidate()
        _require(_settlement_regular_file_identity(database,label="database") == before,
                 "queue changed during reconciliation")
        receipt = {"schema": SCHEMA + "/queue-reconciliation", "evidence_id": evidence["evidence_id"],
            "queue_settlement":settlement, "matching_queue_rows":1, "predecessor_request_id":prior["request_id"],
            "predecessor_metadata_digest":_digest(metadata), "retained_candidate_tree":metadata["candidate_tree"],
            "guard_retained":True, "retry_authorized":False, "completion_authority":False,
            "quarantined_predecessor_unchanged":True}
        yield {**receipt,"receipt_id":_digest(receipt)}
