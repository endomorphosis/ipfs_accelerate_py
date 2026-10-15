"""Native producer and snapshot custody for exact required query references."""
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import build_code_symbol_vector_index
from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget, ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    load_code_retrieval_context, prepare_code_retrieval_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import SemanticProjectionBindings
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import (
    SemanticDecisionPolicy, build_semantic_decision_packet,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_retrieval_query_reference as reference


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def build_retrieval_native_fixture(tmp_path):
    """Authored native producer/vector fixture; never benchmark task admission."""
    repository = tmp_path / "retrieval-source"
    repository.mkdir(mode=0o755)
    source = "def calculate(value):\n    return value + 1\n\ndef another(value):\n    return calculate(value)\n"
    query = ("Inspect calculate and preserve π and café literally. Read mod.py before editing.\n" * 12)
    instruction = ".supervisor-instruction.md"
    (repository / "mod.py").write_text(source)
    (repository / instruction).write_text(query)
    for arguments in (("init", "-q"), ("add", "mod.py", instruction),
                      ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "native fixture")):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)
    workspace = tmp_path / "allocated-retrieval-worktree"
    subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", "-q", str(workspace)],
                   check=True, capture_output=True)
    runtime = repository / ".runtime"
    runtime.mkdir(mode=0o755)
    semantic_artifact = runtime / "semantic/worker-context.json"
    prepared = prepare_semantic_context(repository=repository, paths=["mod.py", instruction],
        required_raw_paths=["mod.py", instruction], objective=" ".join(query.split()), task_id="RETQUERY-1",
        output=semantic_artifact.parent, worker_query=query, worker_max_bytes=65536, context_input_tokens=32768)
    semantic_text = semantic_artifact.read_text()
    semantic_parent = json.loads(semantic_text)
    ast = build_program_evidence_index({"mod.py": source}).ast_index
    snapshot = build_code_symbol_vector_index(ast, forest_id="forest:authored-native", tree_id="tree:authored-native",
        coverage_id=ast.index_id, dimensions=2, model_id="authored-2d", model_revision="1",
        configuration_id="authored:2d", vectors=lambda row: (1.0, 0.0) if "calculate" in row.symbol else (0.0, 1.0))
    result = snapshot.search((1.0, 0.0), max_results=2)
    retrieval_artifact = runtime / "retrieval.json"
    persisted = prepare_code_retrieval_context(repository=repository, task_id="RETQUERY-1", query_text=query,
        snapshot=snapshot, result=result, output=retrieval_artifact)
    binding = reference.SemanticRetrievalQueryBinding(artifact=retrieval_artifact.relative_to(repository).as_posix(),
        expected_sha256=persisted["sha256"], task_id="RETQUERY-1")
    native_text = load_code_retrieval_context(repository=repository, **binding.to_dict())
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
    repository_id = cid_for_payload({"repository": str(repository)})
    tree_id = semantic_parent["scope_cid"]
    semantic_bindings = SemanticProjectionBindings(artifact=semantic_artifact.relative_to(repository).as_posix(),
        expected_sha256=_sha(semantic_text.encode()), task_id=binding.task_id,
        semantic_root_cid=semantic_parent["semantic_root_cid"], scope_cid=semantic_parent["scope_cid"],
        repository_id=repository_id, tree_id=tree_id)
    ids = tuple(row["capsule_cid"] for row in semantic_parent["capsules"])
    assert ids, "authored native producer must retain a capsule"
    policy = SemanticDecisionPolicy(required_capsule_ids=ids, optional_capsule_ids=(), task_text_source_path=instruction)
    semantic_view = build_semantic_decision_packet(original_text=semantic_text, bindings=semantic_bindings, policy=policy)
    retrieval_refs = build_text_context_references(native_text, reference_prefix="code-retrieval", kind="code-retrieval-context",
        path=binding.artifact, repository_id=repository_id, tree_id=tree_id, required=True, chunk_bytes=480,
        coverage_ids=("retrieval:original", "source:mod"))
    semantic_refs = build_text_context_references(semantic_text, reference_prefix="semantic-context", kind="semantic-context",
        path=semantic_bindings.artifact, repository_id=repository_id, tree_id=tree_id, required=True, chunk_bytes=1201,
        coverage_ids=("semantic:original", "source:instruction"))
    unrelated = build_text_context_references("Keep the original acceptance literal.", reference_prefix="failure",
        kind="failure-context", repository_id=repository_id, tree_id=tree_id, required=True)[0]
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id=repository_id, tree_id=tree_id, objective_id=binding.task_id,
        objective_revision="sha256:authored-task", policy_id="policy:authored", policy_revision="sha256:authored-policy",
        caller="supervisor:authored", stage="implementation", goal={"id": binding.task_id},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["mod.py", instruction]}, acceptance={"criteria": ["authored native source validation"]},
        evidence=(*retrieval_refs, unrelated, *semantic_refs))
    suffix = "\nLiteral public and Doctor suffix remains unchanged.\n"
    prompt = render_context_capsule(compiled.capsule) + suffix
    wire, _, _, _, _, _ = codec._prompt_parts(prompt)
    return SimpleNamespace(repository=repository, workspace=workspace, native_prompt=prompt, prompt=prompt,
        native_text=native_text, binding=binding, policy=policy, semantic_original_text=semantic_text,
        semantic_bindings=semantic_bindings, semantic_view=semantic_view, semantic_parent=semantic_parent,
        semantic_artifact=semantic_artifact, retrieval_artifact=retrieval_artifact, snapshot=snapshot, result=result,
        wire=wire, source=source, instruction_source_path=instruction, query=query, suffix=suffix, prepared=prepared,
        metadata={"Code retrieval artifact": binding.artifact, "Code retrieval sha256": binding.expected_sha256,
                  "Semantic context artifact": semantic_bindings.artifact,
                  "Semantic context sha256": semantic_bindings.expected_sha256})


@pytest.fixture
def retrieval_native(tmp_path):
    return build_retrieval_native_fixture(tmp_path)


def _view_arguments(native):
    return dict(native_text=native.native_text, binding=native.binding, semantic_view=native.semantic_view,
                semantic_original_text=native.semantic_original_text, semantic_bindings=native.semantic_bindings,
                policy=native.policy)


def _load_arguments(native):
    return dict(repository=native.repository, binding=native.binding,
                semantic_original_text=native.semantic_original_text, semantic_bindings=native.semantic_bindings)


def test_independently_pinned_native_parents_project_and_restore_required_retrieval(retrieval_native):
    native = retrieval_native
    assert reference.load_live_retrieval_query_context(**_load_arguments(native)) == native.native_text
    assert reference.load_historical_retrieval_query_context(**_load_arguments(native)) == native.native_text
    positions, refs = reference.validate_native_retrieval_references(
        wire=native.wire, binding=native.binding, native_text=native.native_text)
    assert len(positions) == len(refs) > 1
    original = json.loads(native.native_text)
    view = reference.build_retrieval_query_reference(**_view_arguments(native))
    assert view["schema"] == reference.SCHEMA
    assert view["origin"] == native.binding.to_dict()
    assert view["required_context"] == {key: value for key, value in original.items() if key != "query_text"}
    assert view["required_query_text_reference"]["required"] is True
    assert view["required_query_text_reference"]["target_packet_sha256"] == _sha(_encoded(native.semantic_view))
    assert reference.restore_retrieval_query_reference(view=view, **_view_arguments(native)) == native.native_text
    assert "π" in native.query
    new = reference.retrieval_query_reference_references(view=view, **_view_arguments(native),
        coverage_ids=("retrieval:original", "source:mod"), chunk_bytes=256)
    assert new and all(row.required and row.kind == "code-retrieval-context" and row.path == native.binding.artifact for row in new)
    assert all(set(row.coverage_ids) >= {"retrieval:original", "source:mod"} for row in new)
    assert _encoded(view).decode() == "".join(row.summary for row in new)
    assert native.retrieval_artifact.read_bytes() and _sha(native.retrieval_artifact.read_bytes()) == native.binding.expected_sha256


def test_historical_replay_authenticates_original_cas_sources_after_host_source_changes(retrieval_native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
    native = retrieval_native
    (native.repository / "mod.py").write_text("def calculate(value):\n    return value - 9\n")
    assert json.loads(load_code_retrieval_context(repository=native.repository, **native.binding.to_dict()))["status"] == "stale"
    with pytest.raises(ValueError, match="stale"):
        reference.load_live_retrieval_query_context(**_load_arguments(native))
    monkeypatch.setattr(retrieval, "_sources", lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError("historical replay must not inspect current source")))
    assert reference.load_historical_retrieval_query_context(**_load_arguments(native)) == native.native_text


def test_historical_replay_checks_referenced_snapshot_custody(retrieval_native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
    native = retrieval_native
    payload = json.loads(native.retrieval_artifact.read_bytes())
    limit = len(retrieval._json(payload).encode()) - len(retrieval._json(payload["snapshot"]).encode()) // 2
    monkeypatch.setattr(retrieval, "MAX_ARTIFACT_BYTES", limit)
    path = native.repository / ".runtime/referenced-retrieval.json"
    result = prepare_code_retrieval_context(repository=native.repository, task_id=native.binding.task_id,
        query_text=native.query, snapshot=native.snapshot, result=native.result, output=path)
    binding = replace(native.binding, artifact=path.relative_to(native.repository).as_posix(), expected_sha256=result["sha256"])
    arguments = {**_load_arguments(native), "binding": binding}
    assert reference.load_live_retrieval_query_context(**arguments) == native.native_text
    assert reference.load_historical_retrieval_query_context(**arguments) == native.native_text
    payload = json.loads(path.read_bytes())
    assert payload["schema"] == retrieval.REFERENCED_SCHEMA
    snapshot_path = native.repository / payload["snapshot_ref"]["path"]
    original = snapshot_path.read_bytes()
    snapshot_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="byte count|digest|bounded"):
        reference.load_historical_retrieval_query_context(**arguments)


def test_rehashed_forged_result_cannot_pass_historical_native_replay(retrieval_native):
    native = retrieval_native
    forged = replace(native.result, hits=(replace(native.result.hits[0], score=0.5), *native.result.hits[1:]))
    payload = json.loads(native.retrieval_artifact.read_bytes())
    payload["result"] = forged.to_dict()
    raw = _encoded(payload)
    native.retrieval_artifact.write_bytes(raw)
    binding = replace(native.binding, expected_sha256=_sha(raw))
    with pytest.raises(ValueError, match="does not replay"):
        reference.load_historical_retrieval_query_context(**{**_load_arguments(native), "binding": binding})


@pytest.mark.parametrize("case", ["raw_pin", "foreign_task", "retained_source", "duplicate_raw_key", "symlink_raw"],
                         ids=["raw_pin", "foreign_task", "retained_source", "duplicate_raw_key", "symlink_raw"])
def test_parent_custody_errors_refuse_historical_replay(retrieval_native, case):
    native = retrieval_native
    arguments = _load_arguments(native)
    if case == "raw_pin":
        arguments["binding"] = replace(native.binding, expected_sha256="0" * 64)
    elif case == "foreign_task":
        arguments["binding"] = replace(native.binding, task_id="FOREIGN")
    elif case == "retained_source":
        source = native.semantic_parent["manifest"]["mod.py"]["source_cid"]
        (native.semantic_artifact.parent / "blocks" / source).write_bytes(b"changed retained bytes")
    elif case == "duplicate_raw_key":
        raw = native.retrieval_artifact.read_bytes().replace(b"{", b'{"task_id":"duplicate",', 1)
        native.retrieval_artifact.write_bytes(raw)
        arguments["binding"] = replace(native.binding, expected_sha256=_sha(raw))
    else:
        raw = native.retrieval_artifact.read_bytes()
        target = native.repository / "unrelated-copy.json"
        target.write_bytes(raw)
        native.retrieval_artifact.unlink()
        native.retrieval_artifact.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        reference.load_historical_retrieval_query_context(**arguments)


@pytest.mark.parametrize("changes", [
    {"artifact": "../foreign.json"}, {"artifact": "/absolute.json"}, {"artifact": "a//b.json"},
    {"artifact": "a/./b.json"}, {"artifact": "a\n.json"}, {"expected_sha256": "A" * 64},
    {"expected_sha256": "1"}, {"task_id": ""}, {"task_id": " task "}, {"task_id": "x" * 513},
], ids=["parent", "absolute", "empty_component", "dot", "newline", "uppercase_sha", "short_sha", "empty_task", "task_space", "task_bound"])
def test_binding_requires_closed_literal_independent_fields(changes):
    valid = dict(artifact=".runtime/retrieval.json", expected_sha256="1" * 64, task_id="TASK-1")
    with pytest.raises(ValueError):
        reference.SemanticRetrievalQueryBinding(**{**valid, **changes})


def test_binding_round_trip_does_not_accept_extra_or_missing_fields():
    valid = reference.SemanticRetrievalQueryBinding(".runtime/retrieval.json", "1" * 64, "TASK-1")
    assert reference.SemanticRetrievalQueryBinding.from_dict(valid.to_dict()) == valid
    with pytest.raises(ValueError, match="exact"):
        reference.SemanticRetrievalQueryBinding.from_dict({**valid.to_dict(), "source_freshness_verified": True})
    with pytest.raises(ValueError, match="exact"):
        reference.SemanticRetrievalQueryBinding.from_dict({"artifact": valid.artifact, "expected_sha256": valid.expected_sha256})


@pytest.mark.parametrize("case", ["query", "task", "stale", "authority", "source", "noncanonical", "optional_target", "missing_target", "target_tamper", "policy_v1"],
                         ids=["query", "task", "stale", "authority", "source", "noncanonical", "optional_target", "missing_target", "target_tamper", "policy_v1"])
def test_query_reference_refuses_nonexact_or_unavailable_required_targets(retrieval_native, case):
    native = retrieval_native
    arguments = _view_arguments(native)
    body = json.loads(native.native_text)
    if case == "query":
        body["query_text"] += " "
    elif case == "task":
        body["task_id"] = "FOREIGN"
    elif case == "stale":
        body.update(status="stale", stale_paths=["mod.py"], hits=[])
    elif case == "authority":
        body["semantic_authority"] = True
    elif case == "source":
        body["source_sha256"]["mod.py"] = "0" * 64
    elif case == "noncanonical":
        arguments["native_text"] = json.dumps(body, indent=2, ensure_ascii=False)
    elif case in {"optional_target", "missing_target", "target_tamper"}:
        arguments["semantic_view"] = copy.deepcopy(native.semantic_view)
        context = arguments["semantic_view"]["required_context"]
        if case == "optional_target":
            context["required_raw_paths"].remove(native.instruction_source_path)
        elif case == "missing_target":
            del context["raw_sources"][native.instruction_source_path]
        else:
            context["raw_sources"][native.instruction_source_path] += " "
    elif case == "policy_v1":
        arguments["policy"] = replace(native.policy, task_text_source_path=None)
    if case in {"query", "task", "stale", "authority", "source"}:
        arguments["native_text"] = _encoded(body).decode()
    with pytest.raises(ValueError):
        reference.build_retrieval_query_reference(**arguments)


@pytest.mark.parametrize("case", ["status", "hits", "origin", "target", "coverage", "reference_required"],
                         ids=["status", "hits", "origin", "target", "coverage", "reference_required"])
def test_rehashed_reference_view_cannot_change_unique_required_constraints(retrieval_native, case):
    native = retrieval_native
    view = reference.build_retrieval_query_reference(**_view_arguments(native))
    if case == "status":
        view["required_context"]["status"] = "stale"
    elif case == "hits":
        view["required_context"]["hits"] = []
    elif case == "origin":
        view["origin"]["expected_sha256"] = "0" * 64
    elif case == "target":
        view["required_query_text_reference"]["target_selector"][-1] = "mod.py"
    elif case == "coverage":
        view["required_coverage_sha256"] = "0" * 64
    else:
        view["required_query_text_reference"]["required"] = False
    # Even a repaired local commitment cannot replace independent regeneration.
    material = {key: view[key] for key in ("origin", "semantic_origin", "native_context_sha256", "required_context", "required_query_text_reference")}
    view["required_coverage_sha256"] = _sha(_encoded(material))
    if case == "coverage":
        view["native_context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs"):
        reference.restore_retrieval_query_reference(view=view, **_view_arguments(native))


@pytest.mark.parametrize("case", ["reorder", "missing", "duplicate", "requiredness", "repository", "tree", "path", "chunk_index", "chunk_count", "whole_hash", "body", "byte_count"],
                         ids=["reorder", "missing", "duplicate", "requiredness", "repository", "tree", "path", "chunk_index", "chunk_count", "whole_hash", "body", "byte_count"])
def test_original_native_chunks_cannot_be_rebound_truncated_or_promoted(retrieval_native, case):
    native = retrieval_native
    wire = copy.deepcopy(native.wire)
    positions = [index for index, row in enumerate(wire["evidence"]) if row["kind"] == "code-retrieval-context"]
    row = wire["evidence"][positions[0]]
    if case == "reorder":
        wire["evidence"][positions[0]], wire["evidence"][positions[-1]] = wire["evidence"][positions[-1]], row
    elif case == "missing":
        del wire["evidence"][positions[-1]]
    elif case == "duplicate":
        wire["evidence"].append(copy.deepcopy(row))
    else:
        if case == "requiredness":
            row["tier"] = "evidence"
            row["metadata"]["required"] = False
        elif case in {"repository", "tree"}:
            row[case + "_id"] = "foreign:scope"
        elif case == "path":
            row["path"] = ".runtime/foreign.json"
        elif case == "chunk_index":
            row["metadata"]["chunk_index"] = 1
        elif case == "chunk_count":
            row["metadata"]["chunk_count"] += 1
        elif case == "whole_hash":
            row["metadata"]["artifact_content_id"] = "sha256:" + "0" * 64
        elif case == "body":
            row["summary"] += " "
            row["byte_count"] = len(row["summary"].encode())
            row["referenced_content_id"] = "sha256:" + _sha(row["summary"].encode())
        else:
            row["byte_count"] += 1
        row.pop("content_id", None)
        row.pop("reference_content_id", None)
        wire["evidence"][positions[0]] = ContextReference.from_dict(row).to_record()
    with pytest.raises(ValueError):
        reference.validate_native_retrieval_references(wire=wire, binding=native.binding, native_text=native.native_text)


def test_projection_opt_in_preserves_core_other_evidence_literal_suffix_and_both_coverage_groups(retrieval_native):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import project_semantic_router_prompt
    native = retrieval_native
    original_raw, semantic_raw = native.retrieval_artifact.read_bytes(), native.semantic_artifact.read_bytes()
    default = project_semantic_router_prompt(prompt=native.prompt, repository=native.repository, decision_policy=native.policy)
    assert "retrieval_query_reference" not in default.receipt
    old_wire, _ = json.JSONDecoder().raw_decode(default.projected_prompt)
    assert [row for row in old_wire["evidence"] if row["kind"] == "code-retrieval-context"] == [
        row for row in native.wire["evidence"] if row["kind"] == "code-retrieval-context"]
    projected = project_semantic_router_prompt(prompt=native.prompt, repository=native.repository,
        decision_policy=native.policy, retrieval_query_binding=native.binding)
    wire, end = json.JSONDecoder().raw_decode(projected.projected_prompt)
    assert {key: value for key, value in wire.items() if key != "evidence"} == {
        key: value for key, value in native.wire.items() if key != "evidence"}
    assert projected.projected_prompt[end:] == native.suffix
    assert [row for row in wire["evidence"] if row["kind"] not in {"semantic-context", "code-retrieval-context"}] == [
        row for row in native.wire["evidence"] if row["kind"] not in {"semantic-context", "code-retrieval-context"}]
    view = json.loads("".join(row["summary"] for row in wire["evidence"] if row["kind"] == "code-retrieval-context"))
    assert reference.restore_retrieval_query_reference(view=view, **_view_arguments(native)) == native.native_text
    for row in wire["evidence"]:
        if row["kind"] == "code-retrieval-context":
            assert row["metadata"]["required"] is True
            assert set(row["metadata"]["coverage_ids"]) >= {"retrieval:original", "source:mod"}
        elif row["kind"] == "semantic-context":
            assert set(row["metadata"]["coverage_ids"]) >= {"semantic:original", "source:instruction"}
    receipt = projected.receipt["retrieval_query_reference"]
    assert receipt["binding"] == native.binding.to_dict()
    assert receipt["native_context_sha256"] == _sha(native.native_text.encode())
    assert receipt["source_freshness_verified"] is True and receipt["historical_replay"] is False
    assert receipt["exact_native_context_and_reference_inverse"] is True
    assert native.retrieval_artifact.read_bytes() == original_raw and native.semantic_artifact.read_bytes() == semantic_raw


def test_projection_rechecks_both_parents_after_view_construction(retrieval_native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_projection_dispatch as dispatch
    native = retrieval_native
    original = reference.retrieval_query_reference_references
    def change_after_build(**kwargs):
        refs = original(**kwargs)
        native.retrieval_artifact.write_bytes(b"{}")
        return refs
    monkeypatch.setattr(reference, "retrieval_query_reference_references", change_after_build)
    with pytest.raises(ValueError, match="digest"):
        dispatch.project_semantic_router_prompt(prompt=native.prompt, repository=native.repository,
            decision_policy=native.policy, retrieval_query_binding=native.binding)


def test_native_snapshot_parser_accepts_genuine_531_by_384_vectors_above_projection_node_limit():
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot
    source = "\n".join(f"def symbol_{index:04d}(value):\n    return value + {index}\n" for index in range(531))
    ast = build_program_evidence_index({"large.py": source}).ast_index
    snapshot = build_code_symbol_vector_index(ast, forest_id="forest:authored-large", tree_id="tree:authored-large",
        coverage_id=ast.index_id, dimensions=384, model_id="authored-large-vectors", model_revision="1",
        configuration_id="authored:384", vectors=lambda row: (0.0,) * 383 + (1.0,))
    assert len(snapshot.rows) == 531
    assert sum(len(row.embedding) for row in snapshot.rows) == 203904
    raw = _encoded(snapshot.to_dict())
    parsed = reference._parse(raw, reference.MAX_SNAPSHOT_BYTES)
    restored = CodeVectorIndexSnapshot.from_dict(parsed)
    assert restored == snapshot and restored.index_id == snapshot.index_id
    assert _encoded(restored.to_dict()) == raw


@pytest.mark.parametrize("raw", [b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}', b'{"value":1,"value":2}'],
                         ids=["nan", "infinity", "numeric_overflow", "duplicate_key"])
def test_bounded_native_snapshot_parser_rejects_nonfinite_or_duplicate_json(raw):
    with pytest.raises(ValueError):
        reference._parse(raw, reference.MAX_SNAPSHOT_BYTES)


def test_opted_in_projection_refuses_new_reference_id_collision_with_native_unrelated_evidence(retrieval_native):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import project_semantic_router_prompt
    native = retrieval_native
    records = copy.deepcopy(native.wire["evidence"])
    failure = next(row for row in records if row["kind"] == "failure-context")
    failure["reference_id"] = "code-retrieval-query-reference:0001"
    failure.pop("content_id", None)
    failure.pop("reference_content_id", None)
    references = [ContextReference.from_dict(row) for row in records]
    # The native compiler admits unique original IDs; only the opted-in
    # derived replacement would introduce the conflicting identity.
    arguments = {name: native.wire[name] for name in (
        "repository_id", "tree_id", "objective_id", "objective_revision", "policy_id", "policy_revision",
        "caller", "stage", "goal", "authority", "scope", "acceptance",
    )}
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(**arguments, evidence=references)
    prompt = render_context_capsule(compiled.capsule) + native.suffix
    original_ids = [row.reference_id for row in compiled.capsule.evidence]
    assert len(original_ids) == len(set(original_ids))
    with pytest.raises(ValueError, match="reference identities are ambiguous"):
        project_semantic_router_prompt(prompt=prompt, repository=native.repository,
            decision_policy=native.policy, retrieval_query_binding=native.binding)


def test_legacy_projection_bootstraps_with_optional_reference_leaf_absent(retrieval_native, tmp_path):
    native = retrieval_native
    accelerate = Path(codec.__file__).resolve().parents[3]
    import ipfs_datasets_py
    roots = [str(accelerate), str(Path(ipfs_datasets_py.__file__).resolve().parent.parent),
             str(accelerate.parent / "ipfs_kit")]
    prompt_path = tmp_path / "legacy-native-input.txt"
    prompt_path.write_text(native.prompt)
    scratch = tmp_path / "legacy-bootstrap-state"
    scratch.mkdir(mode=0o700)
    program = r'''
import importlib.abc,json,pathlib,sys
sys.dont_write_bytecode=True
sys.path[:0]=json.loads(sys.argv[1])
target='ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference'
class MissingOptional(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference':
            raise ModuleNotFoundError('optional retrieval reference is intentionally absent',name=fullname)
sys.meta_path.insert(0,MissingOptional())
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import project_semantic_router_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker import build_decision_launch_policy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
repository=pathlib.Path(sys.argv[2])
parent=json.loads((repository/'.runtime/semantic/worker-context.json').read_text())
policy=SemanticDecisionPolicy(required_capsule_ids=tuple(row['capsule_cid'] for row in parent['capsules']),
    optional_capsule_ids=(),task_text_source_path='.supervisor-instruction.md')
result=project_semantic_router_prompt(prompt=pathlib.Path(sys.argv[3]).read_text(),repository=repository,
    decision_policy=policy)
assert target not in sys.modules
assert 'retrieval_query_reference' not in result.receipt
assert result.receipt['source_freshness_checked'] is True
descriptor=build_decision_launch_policy(native_prompt=pathlib.Path(sys.argv[3]).read_text(),
    policy=policy,repository=repository)
assert descriptor['schema']=='supervisor-semantic-decision-launch-policy@2'
assert 'retrieval_query_binding' not in descriptor
assert target not in sys.modules
print('legacy projection verified with optional leaf absent')
'''
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0",
        "IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "IPFS_ACCELERATE_META_INDEX_DUCKDB": str(scratch / "meta.duckdb"),
        "IPFS_ACCELERATE_META_INDEX_DUCKLAKE": str(scratch / "ducklake"),
        "IPFS_TEST_PROOF_REUSE_CACHE_DIR": str(scratch / "proof-reuse"),
        "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR": str(scratch / "orchestration"),
    })
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, json.dumps(roots),
                             str(native.repository), str(prompt_path)], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    assert "legacy projection verified with optional leaf absent" in result.stdout
