"""Native Terminal preparation reaches a historical world-input audit.

Git, signed local admission, IntentRepository, the semantic/vector/world
producers, daemon compilation, source-session packet admission, runner capture,
and the bounded audit child are real. The planner proposal and provider response
are authored fixtures. This does not qualify a production worker CLI handoff,
live provider delivery, task completion, or complete-run token savings.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_world_nomination_audit as nominations
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import (
    _proposal_graph, original,
)
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_parent import (
    compile_implementation_world_packet,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask, TodoImplementationDaemon,
)
from test.api.test_repository_world_parent import worker_compiler
from test.api.test_repository_source_index import scheduler
from test.api.test_semantic_router_integration import provider


KINDS = ("semantic-context", "code-retrieval-context", "intent-world-context")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
                          check=True, capture_output=True).stdout


def _context_text(references, kind):
    rows = sorted((row for row in references if row.kind == kind),
                  key=lambda row: row.reference_id)
    assert rows and all(row.required for row in rows)
    return "".join(row.summary for row in rows)


@pytest.fixture
def native_prepared_world(original, scheduler, provider, monkeypatch, tmp_path, request):
    root, instruction, state = original
    public_source = (root / "bottle.py").read_bytes()
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
                            disable_intent_autoencoder=True)
    graph = _proposal_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    assert verified["receipt"]["planning_permitted"] is True
    assert verified["receipt"]["completion_authority"] is False
    with IntentRepository(state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    expected = prep.context(state=state)
    assert expected["provider_calls"] == 0
    assert expected["indexed_symbols"] == expected["native_fact_rows_replayed"] == 1
    assert expected["learned_embeddings"] is False
    assert (root / "bottle.py").read_bytes() == public_source
    bundle_path = root / expected["context_bundle"]["artifact"]
    bundle_raw = bundle_path.read_bytes()
    bundle = json.loads(bundle_raw)
    assert bundle["schema"] == bundles.SCHEMA
    assert set(bundle["tasks"][0]) == {"task_cid", "task_id", "metadata"}
    assert _sha(bundle_raw) == expected["context_bundle"]["sha256"]

    # Build the mandatory evidence through the actual daemon consumer. Its
    # sealed world context remains historical; no live owner is lent to it.
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task = intent.get_task(expected["task_cid"])
        watermark = intent.event_watermark()
        spec = prepared["spec"]
        portal = PortalTask(task_id=task["task_alias"], title=task["body"]["title"],
            canonical_task_cid=task["task_cid"], status="ready", completion="manual",
            priority="P2", track="implementation",
            outputs=[row["path"] for row in spec["outputs"]],
            validation=[shlex.join(row["argv"]) for row in spec["validations"]],
            acceptance=" ; ".join(row["criterion"] for row in spec["acceptance"]),
            metadata={"database task cid": task["task_cid"],
                "local planning contract cid": local.content_identity(task["body"][local.CONTRACT_KEY])})
        daemon = TodoImplementationDaemon(todo_path=state / "pending.todo.md",
            state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
            events_path=state / "worker/events.jsonl", repo_root=root, task_header_prefix="## TB-",
            implementation_protected_paths=("bottle.py",))
        daemon._task_context_nomination_bundle = expected["context_bundle"]
        try:
            daemon_prompt = daemon._build_implementation_prompt(portal, attempt=1)
            parent = daemon._implementation_parent(portal)
            assert parent is not None
            original_references = tuple(ContextReference.from_dict(row.to_record())
                for row in parent[0].evidence)
            assert daemon_prompt == render_context_capsule(parent[0])
        finally:
            daemon.close_event_runtime()
        assert intent.get_task(task["task_cid"]) == task
        assert intent.event_watermark() == watermark
    bodies = {kind: _context_text(original_references, kind) for kind in KINDS}
    semantic, retrieval, world = (json.loads(bodies[kind]) for kind in KINDS)
    assert semantic["semantic_root_cid"] == world["semantic_root_cid"] == expected["semantic_root_cid"]
    assert retrieval["index_id"] == expected["index_id"]
    assert world["world_snapshot_cid"] == expected["world_snapshot_cid"]
    assert world["intent_freshness_checked"] is False
    wire, _ = json.JSONDecoder().raw_decode(daemon_prompt)
    for kind, text in bodies.items():
        rows = sorted((row for row in wire["evidence"] if row["kind"] == kind),
                      key=lambda row: row["reference_id"])
        assert "".join(row["summary"] for row in rows) == text

    reference_source = original_references
    dispatched_bodies = dict(bodies)
    corruption = getattr(request, "param", None)
    if corruption is not None:
        # Negative cases still go through the real compiler, packet builder,
        # and capture path. Independent nomination replay must detect this
        # changed evidence even when every packet/receipt digest agrees.
        kind, field = {"retrieval_body": ("code-retrieval-context", "query_id"),
                       "world_body": ("intent-world-context", "plan_projection_cid")}[corruption]
        changed = json.loads(bodies[kind])
        changed[field] = "sha256:" + "0" * 64
        dispatched_bodies[kind] = _raw(changed).decode()
        first = next(row for row in original_references if row.kind == kind)
        changed_references = build_text_context_references(dispatched_bodies[kind],
            reference_prefix=first.reference_id.rsplit(":", 1)[0], kind=kind,
            path=first.path, repository_id=first.repository_id, tree_id=first.tree_id,
            required=True, chunk_bytes=6144, coverage_ids=first.coverage_ids)
        reference_source = tuple(row for row in original_references if row.kind != kind) + changed_references

    workspace_root = tmp_path / "workers"
    workspace_root.mkdir()
    workspace = workspace_root / "allocated"
    _git(root, "worktree", "add", "--detach", "-q", str(workspace))
    assert (workspace / "bottle.py").read_bytes() == public_source
    for name, binding in prepared["manifest"]["payload"]["sources"].items():
        assert _sha((workspace / name).read_bytes()) == binding["sha256"]

    seen = provider[1]
    with monkeypatch.context() as worker:
        worker.chdir(workspace)
        worker.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
        with RepositorySourceSession(workspace, tmp_path / "source-session",
                index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
            dispatched_parent = (parent[0] if corruption is None else
                replace(parent[0], evidence=reference_source))
            dispatched_prompt = (daemon_prompt if corruption is None else
                render_context_capsule(dispatched_parent))
            adapter = session.context_adapter(worker_compiler())
            task_text = prepared["query"]
            packet = compile_implementation_world_packet(adapter=adapter,
                parent=dispatched_parent, rendered_prompt=dispatched_prompt,
                expected_parent_cid=dispatched_parent.capsule_id,
                task_text=task_text, expected_task_text_sha256=_sha(task_text.encode()),
                query="application", unresolved_question="Which implementation satisfies the signed public task?",
                max_bytes=262144)
            compiled_parent = packet._capsule
            for name in ("goal", "scope", "acceptance", "objective_id", "objective_revision",
                         "policy_id", "policy_revision"):
                assert getattr(compiled_parent, name) == getattr(dispatched_parent, name)
            for name, value in dispatched_parent.authority.items():
                assert compiled_parent.authority["task"][name] == value
            assert compiled_parent.scope["protected_edit_paths"] == ("bottle.py",)
            assert compiled_parent.authority["task"]["implementation_context_budget"]
            assert compiled_parent.authority["task"]["implementation_timeout_policy"]
            assert compiled_parent.authority["task"]["generic_prompt_policy"]
            for old in reference_source:
                new = next(row for row in compiled_parent.evidence if row.reference_id == old.reference_id)
                assert old.summary == new.summary and old.referenced_content_id == new.referenced_content_id
                assert old.metadata == new.metadata and old.path == new.path
            for kind, text in dispatched_bodies.items():
                assert _context_text(compiled_parent.evidence, kind) == text
            output, receipt = runner.run(prompt=task_text, provider="codex_cli", model="pinned",
                timeout=30, max_output_tokens=128, world_task_packet=packet,
                world_packet_audit_directory=state / "world-invocations")
            assert output == "literal summary" and len(seen) == 1
            assert receipt["world_task_packet"]["dispatch_rechecked"] is True
    assert scheduler.snapshot()["active_lease_count"] == 0
    receipt["phase"] = "coding"
    _git(root, "worktree", "remove", str(workspace))
    assert not workspace.exists() and bundle_path.is_file()
    assert (root / "bottle.py").read_bytes() == public_source
    return dict(root=root, state=state, prepared=prepared, expected=expected,
        bundle_path=bundle_path, bundle_raw=bundle_raw, workspace_root=workspace_root,
        workspace=workspace, receipts=[receipt], seen=seen, bodies=bodies,
        daemon_capsule_id=parent[0].capsule_id, corruption=corruption,
        native_parent=parent[0], dispatched_parent=dispatched_parent, worker_parent=compiled_parent)


def _observe(case):
    return audit.collect_terminal_context_audit(state=case["state"], receipts=case["receipts"],
                                               workspace_root=case["workspace_root"])


def test_native_preparation_bundle_and_required_bodies_replay_after_worker_removal(
        native_prepared_world, record_property):
    case = native_prepared_world
    # Both original and worker source files are unnecessary for replay. Source
    # blocks retained alongside the semantic nomination supply historical bytes.
    (case["root"] / "bottle.py").unlink()
    result = _observe(case)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["coding_receipts_seen"] == result["usable_coding_receipts"] == 1
    assert all(result["capsules"][0]["checks"].values())
    for kind, text in case["bodies"].items():
        assert result["capsules"][0]["context_chunks"][kind]["sha256"] == _sha(text.encode())
    match = result["matches"][0]
    projection = match["world_context_nominations"]
    assert projection["schema"] == nominations.SCHEMA
    assert projection["context_bundle_sha256"] == case["expected"]["context_bundle"]["sha256"]
    assert projection["ordinary_bundle_task_selection_verified"] is True
    assert projection["semantic_nomination_exact_text_verified"] is True
    assert projection["retrieval_nomination_historical_projection_verified"] is True
    assert projection["world_nomination_historical_projection_verified"] is True
    assert 0 < projection["retained_artifact_bytes"] <= nominations.MAX_TOTAL_BYTES
    for key in ("semantic_producer_replayed", "intent_freshness_reverified",
                "current_sources_checked", "owner_signature_replayed", "token_counts_replayed",
                "proof_authority", "execution_authority", "completion_authority"):
        assert projection[key] is False
    assert match["model_prompt_sha256"] == _sha(case["seen"][0][0].encode())
    assert match["native_task_cid"] == case["expected"]["task_cid"]
    assert match["world_packet_task_cid"] != match["native_task_cid"]
    replay = result["world_packet_replays"][0]
    assert replay["historical_input_reconstructed"] is True
    assert replay["provider_delivery_verified"] is replay["token_counts_replayed"] is False
    assert result["provider_calls"] == 0
    assert result["task_correctness_established"] is result["completion_authority"] is False
    assert case["bundle_path"].read_bytes() == case["bundle_raw"]
    exported = json.dumps(result)
    assert case["prepared"]["query"] not in exported
    assert str(case["root"]) not in exported and str(case["workspace"]) not in exported
    with IntentRepository(case["state"] / "intent.duckdb", install_schema=False) as intent:
        assert intent.get_task(case["expected"]["task_cid"])["status"] == "ready"
    record_property("terminal_world_native_preparation_cycle", json.dumps(dict(
        real_git_baseline=True, real_signed_local_admission=True, native_intent_repository=True,
        native_metadata_only_bundle=True, native_semantic_vector_world_producers=True,
        native_daemon_compilation=True, required_context_families_preserved=3,
        native_daemon_core_preserved=True, native_daemon_selected_evidence_preserved=True,
        protected_paths_preserved=True, context_and_timeout_policies_preserved=True,
        live_source_session_packet=True, authored_planner_proposals=1,
        authored_provider_calls=1, live_provider_calls=0, private_invocation_capture=True,
        source_session_closed=True, original_worker_workspace_unavailable=True,
        canonical_preparation_repository_retained=True, bounded_audit_child=True,
        canonical_program_source_unavailable=True, retained_source_blocks_replayed=True,
        nomination_projections_verified=True,
        exact_model_input_reconstructed=True, production_worker_handoff_qualified=False,
        provider_delivery_verified=False, task_correctness_established=False,
        full_session_token_savings=None), sort_keys=True))


@pytest.mark.parametrize("mutation", ["pin", "task_cid", "task_alias", "hidden_selected_family",
    "semantic_pin", "retrieval_pin", "world_pin", "missing", "missing_source_block"])
def test_native_preparation_bundle_failure_cannot_be_bypassed_by_valid_world_bytes(
        native_prepared_world, mutation):
    case = native_prepared_world
    expected = deepcopy(case["expected"])
    path = case["bundle_path"]
    if mutation == "pin":
        expected["context_bundle"]["sha256"] = "0" * 64
    elif mutation == "missing":
        path.unlink()
    elif mutation == "missing_source_block":
        nomination = json.loads(case["bundle_raw"])["tasks"][0]["metadata"]
        semantic = json.loads(case["bodies"]["semantic-context"])
        artifact = case["root"] / nomination["semantic context artifact"]
        block = artifact.parent / "blocks" / semantic["manifest"]["bottle.py"]["source_cid"]
        assert block.is_file()
        block.unlink()
    else:
        payload = json.loads(case["bundle_raw"])
        task = payload["tasks"][0]
        if mutation in ("task_cid", "task_alias"):
            task["task_cid" if mutation == "task_cid" else "task_id"] = "foreign:task"
        elif mutation in ("semantic_pin", "retrieval_pin", "world_pin"):
            key = {"semantic_pin": "semantic context sha256", "retrieval_pin": "code retrieval sha256",
                   "world_pin": "world context sha256"}[mutation]
            assert key in task["metadata"]
            task["metadata"][key] = "0" * 64
        else:
            # A correctly pinned historical envelope now selects a family that
            # the retained preparation and packet do not declare. Its minimal
            # inline receipt is transport data, not a qualified model result.
            payload["schema"] = bundles.SOURCE384_SCHEMA
            task["source384_context"] = {
                "schema": "terminal-source384-repository-context@1",
                "summary": {"completion_authority": False},
            }
        raw = _raw(payload)
        path.write_bytes(raw)
        expected["context_bundle"]["sha256"] = _sha(raw)
    prep._write(case["state"] / "context-result.json", expected)
    result = _observe(case)
    assert result["all_observed_coding_inputs_verified"] is False, result
    assert result["any_model_input_verified"] is False, result
    assert result["status"] == "unverified", result
    assert result["read_errors"]
    assert result["provider_calls"] == 0 and len(case["seen"]) == 1
    assert result["completion_authority"] is False


def _nomination_replay(case):
    receipt = case["receipts"][0]
    capture = case["state"] / "world-invocations" / (receipt["invocation_id"] + ".json")
    packet = json.loads(capture.read_bytes())["packet_artifact"]
    return nominations.audit_world_nominations(packet=packet, expected=case["expected"],
                                               prepared=case["prepared"])


def test_nomination_files_are_bounded_regular_files_with_no_symlink_parents(
        native_prepared_world, tmp_path):
    case = native_prepared_world
    metadata = json.loads(case["bundle_raw"])["tasks"][0]["metadata"]
    path = case["root"] / metadata["world context artifact"]
    original = path.read_bytes()
    outside = tmp_path / "same-world-bytes.json"
    outside.write_bytes(original)
    for mutation in ("symlink", "fifo", "oversized"):
        path.unlink()
        try:
            if mutation == "symlink":
                path.symlink_to(outside)
            elif mutation == "fifo":
                os.mkfifo(path)
            else:
                with path.open("wb") as stream:
                    stream.truncate(nominations.MAX_FILE_BYTES + 1)
            result = _observe(case)
            assert result["status"] == "unverified", (mutation, result)
            assert result["any_model_input_verified"] is False
            assert result["read_errors"]
        finally:
            path.unlink()
            path.write_bytes(original)
    parent = path.parent
    moved = parent.with_name(parent.name + "-moved")
    parent.rename(moved)
    try:
        parent.symlink_to(moved, target_is_directory=True)
        result = _observe(case)
        assert result["status"] == "unverified", result
        assert result["any_model_input_verified"] is False and result["read_errors"]
    finally:
        parent.unlink()
        moved.rename(parent)


def test_nomination_population_budget_and_private_staging_cleanup(
        native_prepared_world, monkeypatch):
    case = native_prepared_world
    created = []
    temporary_directory = nominations.TemporaryDirectory

    def tracked_directory(**options):
        directory = temporary_directory(**options)
        created.append(Path(directory.name))
        return directory

    monkeypatch.setattr(nominations, "TemporaryDirectory", tracked_directory)
    for key, limit in (("MAX_TOTAL_BYTES", len(case["bundle_raw"])), ("MAX_FILE_BYTES", 1)):
        with monkeypatch.context() as limited:
            limited.setattr(nominations, key, limit)
            with pytest.raises(ValueError):
                _nomination_replay(case)
        assert all(not path.exists() for path in created)
    result = _nomination_replay(case)
    assert result["world_nomination_historical_projection_verified"] is True
    assert created and all(not path.exists() for path in created)


def test_native_world_loader_uses_captured_bytes_after_original_file_changes(
        native_prepared_world, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.semantic_state import intent_world_snapshot as world
    case = native_prepared_world
    metadata = json.loads(case["bundle_raw"])["tasks"][0]["metadata"]
    original = case["root"] / metadata["world context artifact"]
    original_bytes = original.read_bytes()
    load = world.load_intent_world_context
    used = []

    def replace_original_before_native_read(**options):
        private = options["artifact"]
        assert private != original and private.read_bytes() == original_bytes
        original.write_bytes(b"changed after bounded capture")
        used.append(private)
        return load(**options)

    monkeypatch.setattr(world, "load_intent_world_context", replace_original_before_native_read)
    result = _nomination_replay(case)
    assert result["world_nomination_historical_projection_verified"] is True
    assert used and all(not path.exists() for path in used)


def test_world_capsule_check_does_not_reopen_bundle_after_nomination_capture(
        native_prepared_world, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_world_packet_audit as world
    case = native_prepared_world
    replay = nominations.audit_world_nominations
    called = []

    def remove_after_capture(**options):
        result = replay(**options)
        case["bundle_path"].unlink()
        called.append(result)
        return result

    monkeypatch.setattr(nominations, "audit_world_nominations", remove_after_capture)
    result = world.audit_world_terminal_context(state=case["state"], receipts=case["receipts"],
                                               workspace_root=case["workspace_root"])
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert len(called) == 1
    assert case["expected"]["context_bundle"]["sha256"] == _sha(case["bundle_raw"])


def test_referenced_retrieval_snapshot_replays_and_enforces_declared_size(
        native_prepared_world):
    from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
    case = native_prepared_world
    bundle = json.loads(case["bundle_raw"])
    metadata = bundle["tasks"][0]["metadata"]
    path = case["root"] / metadata["code retrieval artifact"]
    artifact = json.loads(path.read_bytes())
    assert artifact["schema"] == retrieval.SCHEMA
    snapshot = _raw(artifact.pop("snapshot"))
    snapshot_path = path.with_name(path.name + ".index.json")
    snapshot_path.write_bytes(snapshot)
    artifact["schema"] = retrieval.REFERENCED_SCHEMA
    artifact["snapshot_ref"] = dict(path=snapshot_path.relative_to(case["root"]).as_posix(),
        sha256=_sha(snapshot), bytes=len(snapshot), index_id=case["expected"]["index_id"])

    def repin():
        raw = _raw(artifact)
        path.write_bytes(raw)
        metadata["code retrieval sha256"] = _sha(raw)
        raw_bundle = _raw(bundle)
        case["bundle_path"].write_bytes(raw_bundle)
        case["expected"]["context_bundle"]["sha256"] = _sha(raw_bundle)

    repin()
    result = _nomination_replay(case)
    assert result["retrieval_nomination_historical_projection_verified"] is True
    for declared_bytes in (nominations.MAX_FILE_BYTES + 1, len(snapshot) + 1):
        artifact["snapshot_ref"]["bytes"] = declared_bytes
        repin()
        with pytest.raises(ValueError):
            _nomination_replay(case)


@pytest.mark.parametrize("native_prepared_world", ["retrieval_body", "world_body"], indirect=True)
def test_repacked_world_input_cannot_change_native_nominated_context_bodies(native_prepared_world):
    case = native_prepared_world
    assert case["corruption"] in {"retrieval_body", "world_body"}
    assert case["bundle_path"].read_bytes() == case["bundle_raw"]
    result = _observe(case)
    assert result["all_observed_coding_inputs_verified"] is False, result
    assert result["any_model_input_verified"] is False, result
    assert result["status"] == "unverified", result
    assert result["read_errors"]
    assert result["provider_calls"] == 0 and len(case["seen"]) == 1
    assert result["completion_authority"] is False
