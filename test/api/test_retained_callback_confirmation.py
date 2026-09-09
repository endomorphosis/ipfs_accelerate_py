"""Exact retained producer compatibility does not admit foreign auxiliary evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_completion_events import (
    retained_callback_confirmation_is_exact,
)


@pytest.mark.parametrize(
    "tamper",
    [
        "",
        "foreign_launch",
        "foreign_worktree",
        "foreign_task",
        "foreign_candidate",
        "foreign_request",
        "nonempty_key",
        "foreign_target",
        "failed_validation",
        "unknown_provider",
        "consumed_attempt",
        "changed_receipt",
        "changed_provenance",
        "missing_reconciliation",
        "duplicate_reconciliation",
        "wrong_completion",
        "missing_exact_completion",
        "foreign_metadata_worktree",
        "failed_exact_reconciliation",
    ],
)
def test_retained_callback_requires_exact_source_and_native_reconciliation(
    tmp_path, monkeypatch, tamper
):
    events = json.loads(
        (Path(__file__).parent / "fixtures/retained_callback_events.json").read_text()
    )
    start, enqueue, queued, reconciliation, source, completion = events
    receipt = deepcopy(source["merge_result"]["train_result"])
    worktree = source["worktree_path"]
    exact = {
        "implementation_commit": source["implementation_commit"],
        "baseline_commit": source["baseline_ref"],
        "completion_source_event_id": source["event_id"],
        "completion_source_event_type": "implementation_finished",
        "completion_source_portal_attempt": source["attempt"],
        "completion_event_id": completion["event_id"],
    }
    if tamper == "foreign_launch":
        start["stream_id"] = "foreign-launch"
    if tamper == "foreign_worktree":
        enqueue["worktree_path"] += "-foreign"
    if tamper == "foreign_task":
        enqueue["canonical_task_cid"] = "foreign-task"
    if tamper == "foreign_candidate":
        queued["implementation_commit"] = "e" * 40
    if tamper == "foreign_request":
        queued["merge_result"]["request_id"] = "foreign-request"
    if tamper == "nonempty_key":
        reconciliation["canonical_task_key"] = "foreign-key"
    if tamper == "foreign_target":
        source["merge_result"]["target_commit"] = "e" * 40
    if tamper == "failed_validation":
        source["validation_result"]["passed"] = False
    if tamper == "unknown_provider":
        source["provider_dispatched"] = "unknown"
    if tamper == "consumed_attempt":
        source["attempt_consumed"] = True
    if tamper == "changed_receipt":
        source["merge_result"]["train_result"]["accepted"] = False
    if tamper == "changed_provenance":
        queued["merge_queue_synchronous_source"]["source_projection_id"] = "foreign"
    if tamper == "missing_reconciliation":
        events.remove(reconciliation)
    if tamper == "duplicate_reconciliation":
        events.insert(3, deepcopy(reconciliation))
    if tamper == "wrong_completion":
        completion["completion_receipt_repair"] = False
    if tamper == "missing_exact_completion":
        exact = None
    if tamper == "foreign_metadata_worktree":
        worktree += "-foreign"
    for check in reconciliation["post_merge_declared_output_invariant"]["checks"]:
        (tmp_path / check["repository"]).mkdir(parents=True, exist_ok=True)
    refs = {
        check["repository"]: check["repository_ref"]
        for check in reconciliation["post_merge_declared_output_invariant"]["checks"]
    }
    monkeypatch.setattr(
        bridge_module,
        "_gitlink_oid_at_commit",
        lambda root, *, commit, repository: refs[repository],
    )
    monkeypatch.setattr(
        bridge_module, "_regular_blob_oid_at_commit", lambda *args, **kwargs: "a" * 40
    )
    calls = []

    def verify(rec, queue):
        calls.append((rec, queue))
        return (
            tamper != "failed_exact_reconciliation"
            and DatabasePortalExecutionBridge._exact_callback_reconciliation_for_completion_source(
                rec,
                queue,
                alias=source["task_id"],
                task_cid=source["canonical_task_cid"],
                task_key=source["canonical_task_key"],
                repository_root=tmp_path,
            )
        )

    result = retained_callback_confirmation_is_exact(
        events,
        source=source,
        completion=completion,
        exact_completion=exact,
        receipt=receipt,
        worktree_path=worktree,
        alias=source["task_id"],
        task_cid=source["canonical_task_cid"],
        task_key=source["canonical_task_key"],
        verify_reconciliation=verify,
    )
    assert result is (not tamper)
    if not tamper:
        assert len(calls) == 1
        assert "canonical_task_key" not in queued
        assert "canonical_task_key" not in reconciliation


@pytest.mark.parametrize(
    "tamper",
    [
        "",
        "foreign_candidate_ancestry",
        "foreign_current_ancestry",
        "changed_candidate_blob",
        "changed_current_blob",
        "foreign_integration_gitlink",
    ],
)
def test_retained_callback_requires_each_nested_version_and_ancestry(
    tmp_path, monkeypatch, tamper
):
    import subprocess
    from types import SimpleNamespace

    fixtures = Path(__file__).parent / "fixtures"
    events = json.loads((fixtures / "retained_callback_events.json").read_text())
    raw = json.loads((fixtures / "retained_callback_request.json").read_text())
    source, completion = events[-2:]
    receipt = deepcopy(source["merge_result"]["train_result"])
    request = SimpleNamespace(**raw, canonical_identity=raw["canonical_task_key"])
    request.metadata["events_path"] = str(tmp_path / "events.jsonl")
    projection = SimpleNamespace(
        paths=SimpleNamespace(events=tmp_path / "events.jsonl"),
        binding={"task_cid": raw["canonical_task_id"]},
    )
    receipt_path = tmp_path / "train.json"
    receipt_path.write_text(json.dumps(receipt))
    train = SimpleNamespace(
        _dedupe_key=lambda *args: raw["dedupe_key"],
        _read_receipt=lambda key: receipt,
        _receipt_path=lambda key: receipt_path,
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.repository_root = tmp_path
    bridge.merge_target_branch = receipt["target_branch"]
    bridge.merge_queue = SimpleNamespace(
        target_repository_id=raw["metadata"]["target_repository_id"]
    )
    bridge.worktree_submodule_paths = ("external/ipfs_accelerate",)
    bridge._verified_event_chain = lambda paths: events
    bridge._completion_event_evidence = lambda *args, **kwargs: {
        "implementation_commit": source["implementation_commit"],
        "baseline_commit": source["baseline_ref"],
        "completion_source_event_id": source["event_id"],
        "completion_source_event_type": "implementation_finished",
        "completion_source_portal_attempt": source["attempt"],
        "completion_event_id": completion["event_id"],
    }
    candidate = source["implementation_commit"]
    integration = receipt["merge_commit"]
    current = "a" * 40
    child_candidate = "b" * 40
    child_integration = receipt["merge_result"]["post_merge_declared_output_invariant"][
        "checks"
    ][0]["repository_ref"]
    child_current = "c" * 40
    child_root = tmp_path / "external/ipfs_accelerate"
    child_root.mkdir(parents=True)
    monkeypatch.setattr(
        bridge_module,
        "_gitlink_oid_at_commit",
        lambda *args, **kwargs: child_integration,
    )
    monkeypatch.setattr(
        bridge_module, "_regular_blob_oid_at_commit", lambda *args, **kwargs: "a" * 40
    )
    ancestors = []

    def git(argv, *, cwd, **kwargs):
        args = argv[1:]
        code = 0
        out = b""
        if args[0] == "rev-parse":
            ref = args[-1]
            value = (
                current
                if ref.startswith("refs/heads/")
                else raw["metadata"]["candidate_tree"]
                if ref == candidate + "^{tree}"
                else "d" * 40
            )
            out = value.encode() + b"\n"
        elif args[0] == "rev-list":
            out = f"{candidate} {source['baseline_ref']}\n".encode()
        elif args[0] == "merge-base":
            if Path(cwd) == child_root:
                ancestors.append(args[-2:])
                if tamper == "foreign_candidate_ancestry" and args[-2:] == [
                    child_candidate,
                    child_integration,
                ]:
                    code = 1
                if tamper == "foreign_current_ancestry" and args[-2:] == [
                    child_integration,
                    child_current,
                ]:
                    code = 1
        elif args[0] == "ls-tree":
            commit, path = args[2], args[-1]
            if Path(cwd) == tmp_path:
                oid = {
                    candidate: child_candidate,
                    integration: child_integration,
                    current: child_current,
                }[commit]
                if tamper == "foreign_integration_gitlink" and commit == integration:
                    oid = "e" * 40
                out = f"160000 commit {oid}\t{path}\0".encode()
            else:
                oid = (
                    "e" * 40
                    if (
                        tamper == "changed_candidate_blob" and commit == child_candidate
                    )
                    or (tamper == "changed_current_blob" and commit == child_current)
                    else "f" * 40
                )
                out = f"100644 blob {oid}\t{path}\0".encode()
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(argv, code, out, b"")

    monkeypatch.setattr(subprocess, "run", git)
    evidence = bridge._callback_integration_source_evidence(
        request, projection, train=train
    )
    assert (evidence is not None) is (not tamper)
    if not tamper:
        assert [child_candidate, child_integration] in ancestors
        assert [child_integration, child_current] in ancestors
        assert len(evidence["entries"]) == len(raw["metadata"]["task"]["outputs"])
