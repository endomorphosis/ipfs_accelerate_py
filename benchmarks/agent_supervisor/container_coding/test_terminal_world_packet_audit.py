"""Real packet capture and bounded Terminal audit after its source owner exits.

Native source indexing, compilation, packet construction, the runner, MCP, and
historical replay are real. Provider responses and byte-counter measurements
are authored; none of these observations establish live task success or savings.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as terminal_policy
from benchmarks.agent_supervisor.container_coding import terminal_world_packet_audit as world_audit
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    prepare_repository_world_packet, repository_world_task_binding,
)
from test.api.test_repository_context_compiler import compiler
from test.api.test_repository_source_index import scheduler
from test.api.test_repository_source_mcp import real_client
from test.api.test_repository_world_dispatch import counter


TASK = "Change bottle.answer to return two and retain its public name."
TASK_ID = "TB-CODE-TASK"
SOURCE = "# Canonical public comment remains in the source.\ndef answer():\n    return 1\n"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


@pytest.fixture
def terminal_world(tmp_path, scheduler, monkeypatch):
    """Keep Terminal's original context checks alongside a real source packet."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    state = tmp_path / "state"
    state.mkdir()
    workspace_root = tmp_path / "workers"
    source = workspace_root / "workspace-fixture"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "bottle.py").write_text(SOURCE)
    digest = _sha(SOURCE.encode())
    expected = {"task_cid": "cid:task", "semantic_root_cid": "cid:semantic",
                "world_snapshot_cid": "cid:world", "index_id": "cid:vectors"}
    prepared = {"query": TASK, "spec": {"task_key": TASK_ID},
                "worker_inputs": ["bottle.py"], "manifest": {"payload": {
                    "sources": {"bottle.py": {"sha256": digest}}}}}
    payloads = {
        "semantic-context": {"task_id": TASK_ID, "semantic_root_cid": "cid:semantic",
            "manifest": {"bottle.py": {"sha256": digest}}, "completion_authority": False},
        "code-retrieval-context": {"task_id": TASK_ID, "index_id": "cid:vectors",
            "result_id": "cid:result", "query_id": "cid:query", "query_text": TASK,
            "source_sha256": {"bottle.py": digest}, "status": "current", "nomination_only": True,
            "semantic_authority": False, "execution_authority": False, "completion_authority": False},
        "intent-world-context": {"semantic_root_cid": "cid:semantic", "world_snapshot_cid": "cid:world",
            "plan_projection_cid": "cid:projection", "intent_freshness_checked": False,
            "tasks": [{"task_cid": "cid:task", "body": {"local_planning_contract_cid": "cid:contract"}}],
            "execution_authority": False, "completion_authority": False},
    }
    (state / "context-result.json").write_bytes(_json(expected))
    (state / "prepared.json").write_bytes(_json(prepared))
    monkeypatch.chdir(source)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-native-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    fixture = dict(state=state, workspace_root=workspace_root, source=source,
                   prepared=prepared, expected=expected, seen=[], callback=None)

    def generate(prompt, **kwargs):
        fixture["seen"].append((prompt, kwargs))
        observation = {"exit_code": 0, "prompt_tokens": 15, "completion_tokens": 4}
        binding = kwargs.get("codex_repository_source_binding")
        if binding is not None:
            repository_source_policy.validate_binding(binding, workspace=source, materialized=True)
            observation.update(codex_repository_source_configured=True,
                codex_repository_source_config_sha256=binding["config_sha256"])
        schema = kwargs.get("codex_output_schema")
        if schema is not None:
            raw = _json(schema)
            observation.update(codex_output_schema_sha256=_sha(raw), codex_output_schema_bytes=len(raw))
        set_last_cli_observation("codex_cli", observation)
        if fixture["callback"] is not None:
            fixture["callback"](binding)
        return ('{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
                if schema is not None else "Authored candidate; owner validation remains required.")

    monkeypatch.setattr(llm_router, "generate_text", generate)
    with RepositorySourceSession(source, tmp_path / "source-session",
            index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        adapter = session.context_adapter(compiler())
        fixture["session"] = session

        def capture(*, missing_kind=None, task_text=TASK, context_schemas=None, **options):
            refs = []
            for index, (kind, payload) in enumerate(payloads.items()):
                if kind != missing_kind:
                    if context_schemas is not None and kind in context_schemas:
                        payload = {**payload, "schema": context_schemas[kind]}
                    refs.extend(build_text_context_references(json.dumps(payload, ensure_ascii=False),
                        reference_prefix=f"context-{index}", kind=kind, chunk_bytes=257, required=True))
            effects, paths = ("read_source", "write_source", "run_validation"), ("bottle.py",)
            task_binding = repository_world_task_binding(task_id=TASK_ID, task_text=task_text,
                allowed_effects=effects, write_paths=paths)
            compiled = adapter.compile(generation=1, query="answer",
                unresolved_question="Which definition implements the requested answer?",
                required_evidence=tuple(refs),
                required_reference_ids=tuple(ref.reference_id for ref in refs),
                required_coverage_ids=tuple(sorted({key for ref in refs for key in ref.coverage_ids})),
                objective_id=TASK_ID, objective_revision="sha256:authored-task",
                policy_id="policy:authored-review", policy_revision="sha256:authored-policy",
                goal={"intent": TASK}, authority={"mode": "proposal", "world_task_packet": task_binding},
                scope={"paths": paths}, acceptance={"criteria": ["Retain the public name."]})
            packet = prepare_repository_world_packet(adapter=adapter, compilation=compiled,
                generation=1, task_id=TASK_ID, task_text=task_text,
                unresolved_obligations=("Return the requested value without changing the public name.",),
                validation_commands=("python -c 'from bottle import answer; assert answer() == 2'",),
                allowed_effects=effects, write_paths=paths)
            _, receipt = runner.run(prompt=task_text, provider="codex_cli", model="pinned",
                timeout=30, max_output_tokens=128, world_task_packet=packet,
                world_packet_audit_directory=state / "world-invocations", **options)
            receipt["phase"] = "coding"
            fixture.update(packet=packet, compilation=compiled, receipts=[receipt])
            return receipt

        fixture["capture"] = capture
        yield fixture
    assert scheduler.snapshot()["active_lease_count"] == 0


def _observe(fixture, *, child=False):
    function = audit.collect_terminal_context_audit if child else audit.audit_terminal_context
    return function(state=fixture["state"], receipts=fixture["receipts"],
                    workspace_root=fixture["workspace_root"])


def _artifact_path(fixture):
    receipt = fixture["receipts"][0]
    return fixture["state"] / "world-invocations" / (receipt["invocation_id"] + ".json")


def _unverified(result):
    assert result["all_observed_coding_inputs_verified"] is not True, result
    assert result["status"] != "verified_all_observed_model_inputs", result
    assert result["provider_calls"] == 0
    assert result["completion_authority"] is False


@pytest.mark.parametrize("ordinary", [False, True])
def test_actual_runner_packet_is_audited_in_bounded_child(terminal_world, ordinary):
    options = {"coding_reply_mode": "ordinary-completion@1", "prompt_token_counter": counter()} if ordinary else {}
    receipt = terminal_world["capture"](**options)
    terminal_world["session"].close()
    result = _observe(terminal_world, child=True)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["coding_receipts_seen"] == result["usable_coding_receipts"] == 1
    assert all(result["capsules"][0]["checks"].values())
    assert result["capsules"][0]["context_chunks"]["code-retrieval-context"]["count"] > 1
    assert result["matches"][0]["model_prompt_sha256"] == _sha(terminal_world["seen"][0][0].encode())
    assert result["matches"][0]["native_prompt_sha256"] == _sha(TASK.encode())
    assert result["matches"][0]["compiled_capsule_prompt_sha256"] != _sha(TASK.encode())
    assert result["matches"][0]["native_task_cid"] != result["matches"][0]["world_packet_task_cid"]
    assert result["provider_calls"] == 0
    replayed = result["world_packet_replays"][0]
    assert replayed["historical_input_reconstructed"] is True
    assert replayed["provider_delivery_verified"] is replayed["token_counts_replayed"] is False
    assert replayed["recorded_token_measurement_present"] is ordinary
    if ordinary:
        assert receipt["coding_reply_contract"]["response_validated"] is True
        assert replayed["native_output_schema_sha256"] == receipt["coding_reply_contract"]["native_output_schema_sha256"]
    exported = json.dumps(result)
    assert TASK not in exported and SOURCE not in exported and str(terminal_world["source"]) not in exported
    assert result["task_correctness_established"] is result["completion_authority"] is False


def test_real_compact_mcp_edit_audits_after_workspace_removal(terminal_world, monkeypatch, tmp_path, record_property):
    def edit(binding):
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            found, failed = client.call("repository_context_query", generation=1, query="answer", limit=8)
            assert not failed, found
            candidate = next(row for row in found["candidates"]
                if row["path"] == "bottle.py" and row["name"] == "bottle.answer")
            expanded, failed = client.call("repository_capsule_expand", generation=1,
                semantic_root_cid=found["semantic_root_cid"], stable_symbol_ids=[candidate["symbol_id"]])
            assert not failed and len(expanded["capsules"]) == 1, expanded
            assert expanded["capsules"][0]["capsule_cid"] == candidate["capsule_cid"]
            assert expanded["semantic_root_cid"] == found["semantic_root_cid"]
            whole_file, failed = client.call("repository_read", generation=1, path="bottle.py")
            assert not failed and whole_file["representation"] == "compact", whole_file
            assert len(whole_file["content"].encode()) < len(SOURCE.encode())
            assert "Canonical public comment" not in whole_file["content"]
            # The tiny definition can correctly select original bytes when
            # compaction provides no reduction. Edit using its returned view.
            read, failed = client.call("repository_read", generation=1, path="bottle.py",
                symbol_id=candidate["symbol_id"])
            assert not failed, read
            changed, failed = client.call("repository_replace_symbol", generation=1, path="bottle.py",
                symbol=read["symbol"], expected_sha256=read["source_sha256"],
                representation=read["representation"], replacement=read["content"].replace("return 1", "return 2"))
            assert not failed and changed["canonical_changed"] is True, changed

    terminal_world["callback"] = edit
    receipt = terminal_world["capture"](repository_source_policy="repository-source-compact@1")
    source = terminal_world["source"]
    assert (source / "bottle.py").read_text() == SOURCE.replace("return 1", "return 2")
    binding = receipt["repository_source_policy"]
    assert binding["usage"]["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    terminal_world["session"].close()
    monkeypatch.chdir(tmp_path)
    source.rename(tmp_path / "unavailable-workspace")
    assert not source.exists()
    result = _observe(terminal_world, child=True)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["matches"][0]["model_prompt_sha256"] == _sha(terminal_world["seen"][0][0].encode())
    observed = terminal_policy.repository_source_dispatch_observation(dict(
        repository_source_policy="repository-source-compact@1",
        provider_invocations=terminal_world["receipts"], context_input_audit=result),
        "repository-source-compact@1")
    assert observed["effective_repository_tools_observed"] is True, observed
    assert observed["invocations"][0]["confirmed_canonical_edit_observed"] is True
    assert observed["invocations"][0]["capsule_retrieval_observed"] is True
    assert observed["token_savings_established"] is observed["proof_authority"] is observed["completion_authority"] is False
    assert result["provider_calls"] == 0 and len(terminal_world["seen"]) == 1
    record_property("terminal_world_packet_audit_cycle", json.dumps(dict(
        native_source_index=True, required_context_families=3, real_child_mcp=True,
        authored_provider_calls=1, live_provider_calls=0, canonical_edits=1,
        native_capsules_expanded=1, compact_whole_file_reads=1, symbol_reads=1,
        source_session_closed=True, original_workspace_unavailable=True,
        bounded_audit_child=True, exact_model_input_reconstructed=True,
        repository_tool_observation_joined=True, provider_delivery_verified=False,
        task_correctness_established=False, full_session_token_savings=None), sort_keys=True))


@pytest.mark.parametrize("mutation", ["missing", "missing_reference", "digest", "bytes", "changed_bytes", "packet_digest", "invocation_link"])
def test_missing_or_changed_world_artifact_never_qualifies(terminal_world, mutation):
    receipt = terminal_world["capture"]()
    path = _artifact_path(terminal_world)
    reference = receipt["world_packet_replay_artifact"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "missing_reference":
        del receipt["world_packet_replay_artifact"]
    elif mutation == "changed_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "bytes":
        reference["artifact_bytes"] += 1
    elif mutation == "invocation_link":
        reference["invocation_id"] = "a" * 32
    else:
        reference["artifact_sha256" if mutation == "digest" else "packet_artifact_sha256"] = "0" * 64
    _unverified(_observe(terminal_world))


@pytest.mark.parametrize("kind", ["file_symlink", "parent_symlink", "fifo"])
def test_world_artifact_loader_refuses_nonregular_or_redirected_files(terminal_world, tmp_path, kind):
    terminal_world["capture"]()
    path = _artifact_path(terminal_world)
    original = path.read_bytes()
    if kind == "file_symlink":
        target = tmp_path / "redirected-invocation.json"
        path.rename(target)
        path.symlink_to(target)
    elif kind == "parent_symlink":
        parent = path.parent
        target = tmp_path / "redirected-world-invocations"
        parent.rename(target)
        parent.symlink_to(target, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path, 0o600)
    # The FIFO case uses the real bounded collector, so even a broken read
    # implementation cannot leave this test waiting on an unbounded pipe open.
    result = _observe(terminal_world, child=kind == "fifo")
    _unverified(result)
    assert result["any_model_input_verified"] is not True
    if kind == "file_symlink":
        assert target.read_bytes() == original
    elif kind == "parent_symlink":
        assert (target / path.name).read_bytes() == original


@pytest.mark.parametrize("kind", ["declared", "actual"])
def test_oversized_world_artifacts_are_refused_before_replay(terminal_world, monkeypatch, kind):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay

    receipt = terminal_world["capture"]()
    if kind == "declared":
        receipt["world_packet_replay_artifact"]["artifact_bytes"] = world_audit.MAX_ARTIFACT_BYTES + 1
    else:
        # A sparse oversized regular file tests stat-before-read without
        # allocating or populating a large malformed JSON object.
        with _artifact_path(terminal_world).open("r+b") as artifact:
            artifact.truncate(world_audit.MAX_ARTIFACT_BYTES + 1)
    replay_calls = []

    def forbidden(*args, **kwargs):
        replay_calls.append(True)
        raise AssertionError("oversized artifact reached replay")

    monkeypatch.setattr(repository_world_invocation_replay, "replay_repository_world_invocation", forbidden)
    _unverified(_observe(terminal_world))
    assert replay_calls == []


def test_aggregate_artifact_budget_limits_file_reads_and_never_claims_completeness(terminal_world, monkeypatch):
    receipt = terminal_world["capture"]()
    count = world_audit.MAX_AGGREGATE_ARTIFACT_BYTES // world_audit.MAX_ARTIFACT_BYTES
    assert 0 < count < audit.MAX_RECEIPTS
    population = []
    for index in range(count + 1):
        row = deepcopy(receipt)
        row["invocation_id"] = f"{index:032x}"
        row["world_packet_replay_artifact"].update(invocation_id=row["invocation_id"],
            artifact_bytes=world_audit.MAX_ARTIFACT_BYTES)
        population.append(row)
    terminal_world["receipts"] = population
    attempted = []
    read = audit._read_at

    def observe_read(root_fd, relative, *args, **kwargs):
        if relative.startswith("world-invocations/"):
            attempted.append(relative)
        return read(root_fd, relative, *args, **kwargs)

    monkeypatch.setattr(audit, "_read_at", observe_read)
    result = _observe(terminal_world)
    _unverified(result)
    assert result["coding_receipts_seen"] == count + 1
    assert result["usable_coding_receipts"] == len(attempted) == count
    assert {row["invocation_id"] + ".json" for row in population[:count]} == {
        Path(relative).name for relative in attempted}
    assert any(row["kind"] == "world_receipt" and row["error_type"] == "InvalidReceipt"
               for row in result["read_errors"])


@pytest.mark.parametrize("field", ["model_prompt_sha256", "native_prompt_sha256", "workspace", "world_task_packet"])
def test_changed_receipt_cannot_reuse_valid_artifact(terminal_world, field):
    receipt = terminal_world["capture"]()
    if field == "workspace":
        receipt[field] += "-foreign"
    elif field == "world_task_packet":
        receipt[field]["context_cid"] = "sha256:" + "0" * 64
    else:
        receipt[field] = "0" * 64
    _unverified(_observe(terminal_world))


@pytest.mark.parametrize("kind", audit.KINDS)
def test_exact_replay_does_not_bypass_required_terminal_contexts(terminal_world, kind):
    terminal_world["capture"](missing_kind=kind)
    _unverified(_observe(terminal_world))


@pytest.mark.parametrize("kind,schema", [
    ("semantic-context", "supervisor-semantic-worker-context@999"),
    ("semantic-context", None),
    ("semantic-context", 1),
    ("code-retrieval-context", "supervisor-code-retrieval-context@999"),
    ("intent-world-context", "supervisor-intent-world-planning-context@999"),
])
def test_explicit_unsupported_context_schema_cannot_use_legacy_shape_checks(terminal_world, kind, schema):
    terminal_world["capture"](context_schemas={kind: schema})
    _unverified(_observe(terminal_world))


def test_supported_explicit_context_schemas_retain_historical_checks(terminal_world):
    terminal_world["capture"](context_schemas={
        "semantic-context": "supervisor-semantic-worker-context@1",
        "code-retrieval-context": "supervisor-code-retrieval-context@1",
        "intent-world-context": "supervisor-intent-world-planning-context@1",
    })
    result = _observe(terminal_world)
    assert result["all_observed_coding_inputs_verified"] is True, result
    assert all(result["capsules"][0]["checks"].values())


def test_packet_task_text_must_match_prepared_terminal_instruction(terminal_world):
    terminal_world["capture"](task_text="A different bounded task for bottle.answer.")
    _unverified(_observe(terminal_world))


def test_duplicate_invocations_do_not_become_a_complete_audited_population(terminal_world):
    receipt = terminal_world["capture"]()
    terminal_world["receipts"].append(deepcopy(receipt))
    _unverified(_observe(terminal_world))


def test_mixed_world_and_unreplayed_native_receipt_population_stays_incomplete(terminal_world):
    receipt = terminal_world["capture"]()
    legacy = deepcopy(receipt)
    legacy["invocation_id"] = "a" * 32
    del legacy["world_task_packet"]
    del legacy["world_packet_replay_artifact"]
    terminal_world["receipts"].append(legacy)
    _unverified(_observe(terminal_world))


def test_mixed_world_and_legacy_routes_each_require_their_own_reconstruction(terminal_world):
    terminal_world["capture"]()
    capsule = terminal_world["compilation"].capsule
    folder = terminal_world["state"] / "launch/implementation-logs"
    folder.mkdir(parents=True)
    (folder / "tb-code-task-base-context-capsule.json").write_text(capsule.to_json())
    native = render_context_capsule(capsule)
    _, legacy = runner.run(prompt=native, provider="codex_cli", model="pinned",
        timeout=30, max_output_tokens=128)
    legacy["phase"] = "coding"
    terminal_world["receipts"].append(legacy)
    terminal_world["session"].close()
    result = _observe(terminal_world, child=True)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["coding_receipts_seen"] == result["usable_coding_receipts"] == 2
    assert len(result["matches"]) == 2 and len(result["world_packet_replays"]) == 1
    assert {row["native_prompt_sha256"] for row in result["matches"]} == {
        _sha(TASK.encode()), _sha(native.encode())}
    assert len(terminal_world["seen"]) == 2 and result["provider_calls"] == 0


def test_advanced_retained_selections_require_their_own_world_replay_qualification(terminal_world):
    terminal_world["capture"]()
    path = terminal_world["state"] / "context-result.json"
    for key in ("source384_context", "numerical_formalization",
                "numerical_selection_384", "terminal_semantic_scope"):
        path.write_bytes(_json({**terminal_world["expected"], key: {}}))
        result = _observe(terminal_world)
        _unverified(result)
        assert result["invocations"][0]["reason"] == "world_packet_unsupported_retained_formalization_or_signed_scope"


@pytest.mark.parametrize("binding", [None, {}, {"artifact": "bundle.json"},
    {"artifact": "bundle.json", "sha256": "0" * 64, "extra": True}])
def test_present_bundle_requires_complete_closed_binding(terminal_world, binding):
    terminal_world["capture"]()
    path = terminal_world["state"] / "context-result.json"
    path.write_bytes(_json({**terminal_world["expected"], "context_bundle": binding}))
    _unverified(_observe(terminal_world))


@pytest.mark.parametrize("field", ["semantic_translation", "source_input_view", "query_reference_view",
    "public_instruction", "source_lookup_retrieval", "prompt_selection"])
def test_collector_refuses_incompatible_fields_before_projection_or_spawn(terminal_world, monkeypatch, field):
    receipt = terminal_world["capture"]()
    receipt[field] = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible receipt launched an audit child")

    monkeypatch.setattr(audit.subprocess, "run", forbidden)
    _unverified(_observe(terminal_world, child=True))
