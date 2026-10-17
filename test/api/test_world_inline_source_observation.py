"""Real pure compiler/world transport with authored historical semantic payloads.

No native semantic producer, source index, scheduler, provider or database runs.
The assertions establish selected carrier integrity and observation only.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler, build_text_context_references
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget, ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline
from ipfs_accelerate_py.agent_supervisor.runtime import world_inline_source_observation as world
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
    GUIDANCE, _render_repository_world_packet, repository_world_task_binding,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_read_observation import make_read_event
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def semantic_payload(*, version=1, sources=None):
    sources = {"calc.py": 'value = "λ\\Ω"\r\n', "empty.py": ""} if sources is None else sources
    value = {"schema": f"supervisor-semantic-worker-context@{version}", "task_id": "task:authored",
        "objective": "Inspect selected sources", "scope_cid": "cid:authored-scope",
        "semantic_root_cid": "cid:authored-root", "manifest": {
            path: {"sha256": sha(content), "source_cid": "cid:authored-source"}
            for path, content in sources.items()}, "required_raw_paths": sorted(sources),
        "preparation_bounds": {"max_symbols": 16, "context_input_tokens": 20000},
        "doctor_snapshot_id": "doctor:authored", "doctor_manifest_cid": "cid:authored-doctor",
        "reconstruction": {}, "refresh_lineage": None, "completion_authority": False,
        "capsules": [], "admissions": [], "raw_sources": sources, "pack": {}}
    if version == 2:
        value["program_paths"] = ["calc.py"] if "calc.py" in sources else []
    return value


def compiled_world(*, payload=None, text=None, kind="semantic-context", chunk_bytes=257):
    payload = semantic_payload() if payload is None else payload
    text = wire(payload) if text is None else text
    repository, tree, task = "repo:authored", "sha256:" + "1" * 64, "task:authored"
    effects, paths = ("read", "modify"), ("calc.py",)
    with public_replay_without_metadata():
        refs = build_text_context_references(text, reference_prefix="semantic-context", kind=kind,
            repository_id=repository, tree_id=tree, path="context/semantic.json",
            required=True, chunk_bytes=chunk_bytes)
        authority = {"repository_context": {"generation": 1, "source_tree_sha256": "1" * 64,
            "repository_id": repository, "semantic_state_cid": "cid:authored-state",
            "semantic_root_cid": "cid:authored-root", "capsule_index_cid": "cid:authored-index",
            "required_reference_ids": sorted(row.reference_id for row in refs), "required_coverage_ids": []},
            "task": {"world_task_packet": repository_world_task_binding(task, "Inspect the sources", effects, paths)}}
        result = ContextCompiler(ContextBudget(max_input_tokens=80000, max_items=128,
            max_item_bytes=32768, max_serialized_bytes=262144)).compile(
            repository_id=repository, tree_id=tree, objective_id=task, objective_revision="revision:authored",
            policy_id="policy:authored", policy_revision="revision:policy", caller="caller:authored",
            stage="implementation", goal={"task_id": task}, authority=authority,
            scope={"paths": list(paths)}, acceptance={"criteria": ["pending validation"]}, evidence=refs)
        prompt, _, _ = _render_repository_world_packet(compilation=result, generation=1,
            task_id=task, task_text="Inspect the sources", unresolved_obligations=("Inspect source text",),
            validation_commands=("python -m pytest -q",), allowed_effects=effects, write_paths=paths,
            max_bytes=262144)
    return json.loads(prompt[len(GUIDANCE) + 1:])


def semantic_members(envelope):
    return [row for row in envelope["packet"]["members"]
            if row["payload"].get("reference", {}).get("kind") == "semantic-context"]


def refresh_member(member):
    raw = member["payload"]["reference"]
    raw.pop("content_id", None)
    member["payload"]["reference"] = ContextReference.from_dict(raw).to_record()
    member["digest"] = "sha256:" + sha(wire({key: member[key] for key in ("kind", "summary", "path", "payload")}))


@pytest.fixture
def selected_world():
    return compiled_world()


def world_case():
    """Pure fixture hook for the caller's census/collector integration tests."""
    envelope = compiled_world()
    advisory, invocation = "Workspace λ:\n", "world-invocation"
    return dict(envelope=envelope, sources=semantic_payload()["raw_sources"],
        advisory=advisory, invocation_id=invocation,
        model=advisory + GUIDANCE + "\n" + wire(envelope))


@pytest.mark.parametrize("version", [1, 2])
def test_complete_real_selected_chunks_expose_only_literal_source_slots(version):
    payload = semantic_payload(version=version)
    envelope = compiled_world(payload=payload)
    assert len(semantic_members(envelope)) > 1
    sources, manifest, scope = world.world_sources(envelope)
    assert sources == payload["raw_sources"] and manifest == payload["manifest"]
    assert scope == world.WORLD_SCOPE
    sources["calc.py"] = "caller-local mutation"
    assert world.world_sources(envelope)[0] == payload["raw_sources"]


def test_world_observer_pins_final_input_and_uses_original_file_read_identity(selected_world):
    advisory, invocation = "Workspace λ:\n", "world-invocation"
    model = advisory + GUIDANCE + "\n" + wire(selected_world) + "\nOpaque suffix"
    result = inline.observe_world_inline_sources(model_prompt=model,
        workspace_advisory=advisory, invocation_id=invocation)
    assert result["status"] == "observed", result
    assert result["scope"] == world.WORLD_SCOPE
    assert result["model_prompt_sha256"] == sha(model)
    assert result["model_prompt_bytes"] == len(model.encode())
    assert result["invocation_id_sha256"] == sha(invocation)
    expected = []
    for path, content in semantic_payload()["raw_sources"].items():
        event = make_read_event({"path": path, "generation": 1}, {
            "path": path, "generation": 1, "content": content, "source_sha256": sha(content),
            "representation": "original", "encoding": "utf-8", "proof_authority": False}, True)
        expected.append({key: value for key, value in event.items() if key not in {"schema", "generation"}})
    assert result["entries"] == sorted(expected, key=lambda row: row["source_identity_sha256"])
    assert result["source_count"] == 2
    assert result["logical_source_bytes"] == sum(len(text.encode()) for text in semantic_payload()["raw_sources"].values())
    assert result["provider_tokens"] is result["wire_overlap_bytes"] is result["provider_ingestion_verified"] is None
    assert result["full_prompt_source_coverage"] is None
    assert inline.validate_inline_sources(result, model_prompt_sha256=sha(model),
        model_prompt_bytes=len(model.encode()), invocation_id_sha256=sha(invocation)) == result
    exported = wire(result)
    assert "calc.py" not in exported and "value =" not in exported and "context/semantic.json" not in exported


def test_manifest_only_and_capsule_text_remain_outside_literal_scope():
    payload = semantic_payload()
    payload["manifest"]["omitted.py"] = {"sha256": sha("OMITTED"), "source_cid": "cid:omitted"}
    payload["capsules"] = [{"summary": wire({"raw_sources": {"invented.py": "INVENTED"}})}]
    payload["worker_projection"] = {"raw_source_fetch_required": {"omitted.py": payload["manifest"]["omitted.py"]}}
    sources, manifest, _ = world.world_sources(compiled_world(payload=payload))
    assert set(sources) == {"calc.py", "empty.py"} and "omitted.py" in manifest


def test_supported_empty_literal_population_is_zero_only_inside_complete_carrier():
    sources, manifest, scope = world.world_sources(compiled_world(payload=semantic_payload(sources={})))
    assert sources == manifest == {} and scope == world.WORLD_SCOPE


def test_selected_chunk_order_is_derived_only_from_unique_bound_indexes(selected_world):
    selected_world["packet"]["members"].reverse()
    assert world.world_sources(selected_world)[0] == semantic_payload()["raw_sources"]


def test_absolute_artifact_locator_is_rejected_by_existing_reference_contract(selected_world):
    # The daemon forwards str(semantic_artifact), but the shared reference
    # constructor itself requires repository-relative locators. The observer
    # does not loosen this producer contract or open that location.
    member = semantic_members(selected_world)[0]
    member["payload"]["reference"]["path"] = "/private/context/semantic.json"
    with pytest.raises(ValueError, match="^invalid_world_source_carrier$"):
        world.world_sources(selected_world)


@pytest.mark.parametrize("change", ["missing_chunk", "duplicate_chunk", "summary", "chunk_digest",
    "artifact_digest", "chunk_index", "boolean_index", "chunk_count", "byte_count", "reference_id",
    "missing_reference_cid", "reference_cid", "member_digest", "body_id", "path", "tree", "repository",
    "required", "tier", "metadata_field", "multiple_artifacts"])
def test_incomplete_or_rebound_selected_population_refuses_as_a_whole(selected_world, change):
    members = semantic_members(selected_world)
    member, reference = members[0], members[0]["payload"]["reference"]
    refresh = True
    if change == "missing_chunk": selected_world["packet"]["members"].remove(member)
    elif change == "duplicate_chunk":
        copied = deepcopy(member)
        copied["item_id"] += "-duplicate"
        selected_world["packet"]["members"].append(copied)
    elif change == "summary": reference["summary"] += " "
    elif change == "chunk_digest": reference["referenced_content_id"] = "sha256:" + "0" * 64
    elif change == "artifact_digest": reference["metadata"]["artifact_content_id"] = "sha256:" + "0" * 64
    elif change == "chunk_index": reference["metadata"]["chunk_index"] = 1
    elif change == "boolean_index": reference["metadata"]["chunk_index"] = False
    elif change == "chunk_count": reference["metadata"]["chunk_count"] += 1
    elif change == "byte_count": reference["byte_count"] += 1
    elif change == "reference_id": reference["reference_id"] += "-other"
    elif change == "missing_reference_cid":
        reference.pop("content_id")
        refresh = False
    elif change == "reference_cid":
        reference["content_id"] = "sha256:" + "0" * 64
        refresh = False
    elif change == "member_digest":
        member["digest"] = "sha256:" + "0" * 64
        refresh = False
    elif change == "body_id": member["payload"]["evidence_id"] += "-other"
    elif change == "path": reference["path"] = "context/other.json"
    elif change == "tree": reference["tree_id"] += "-other"
    elif change == "repository": reference["repository_id"] += "-other"
    elif change == "required": reference["metadata"]["required"] = False
    elif change == "tier": reference["tier"] = "evidence"
    elif change == "metadata_field": reference["metadata"]["extra"] = "ignored?"
    else:
        other = semantic_members(compiled_world(payload=semantic_payload(sources={"other.py": "different"})))
        for item in other:
            item["item_id"] += "-other-artifact"
        selected_world["packet"]["members"].extend(other)
    if refresh:
        refresh_member(member)
    with pytest.raises(ValueError, match="^invalid_world_source_carrier$"):
        world.world_sources(selected_world)


@pytest.mark.parametrize("change", ["reference", "hash", "missing_manifest", "manifest_list", "path",
    "completion", "extra_field", "required_missing", "duplicate_required", "missing_raw", "raw_list"])
def test_complete_chunks_do_not_authorize_malformed_semantic_literal_slots(change):
    payload = semantic_payload()
    if change == "reference": payload["raw_sources"]["calc.py"] = {"$source": "elsewhere"}
    elif change == "hash": payload["manifest"]["calc.py"]["sha256"] = "0" * 64
    elif change == "missing_manifest": del payload["manifest"]["calc.py"]
    elif change == "manifest_list": payload["manifest"] = []
    elif change == "path":
        payload["raw_sources"]["../bad.py"] = payload["raw_sources"].pop("calc.py")
        payload["manifest"]["../bad.py"] = payload["manifest"].pop("calc.py")
        payload["required_raw_paths"] = ["../bad.py", "empty.py"]
    elif change == "completion": payload["completion_authority"] = 0
    elif change == "extra_field": payload["new_future_slots"] = {}
    elif change == "required_missing": payload["required_raw_paths"].append("absent.py")
    elif change == "duplicate_required": payload["required_raw_paths"].append("calc.py")
    elif change == "missing_raw": del payload["raw_sources"]
    else: payload["raw_sources"] = []
    with pytest.raises(ValueError, match="^invalid_world_source_carrier$"):
        world.world_sources(compiled_world(payload=payload))


@pytest.mark.parametrize("text", [
    '{"schema":"x","schema":"y"}', '{"value":NaN}', '{"value":Infinity}',
    '{"value":' + '[' * 40 + '1' + ']' * 40 + '}', '{"value":',
    wire(semantic_payload()) + " trailing payload",
    json.dumps(semantic_payload(), indent=2),
])
def test_bound_artifact_parser_rejects_duplicates_nonfinite_depth_and_suffix(text):
    with pytest.raises(ValueError, match="^invalid_world_source_carrier$"):
        world.world_sources(compiled_world(text=text, chunk_bytes=6144))


def test_chunk_and_source_limits_refuse_without_partial_inventory(selected_world, monkeypatch):
    count = len(semantic_members(selected_world))
    monkeypatch.setattr(world, "MAX_REFERENCES", count)
    assert world.world_sources(selected_world)[0]
    monkeypatch.setattr(world, "MAX_REFERENCES", count - 1)
    with pytest.raises(ValueError): world.world_sources(selected_world)
    monkeypatch.setattr(world, "MAX_REFERENCES", 64)
    payload = semantic_payload(sources={f"file_{index}.py": "x" for index in range(65)})
    with pytest.raises(ValueError): world.world_sources(compiled_world(payload=payload, chunk_bytes=6144))


def test_only_declared_semantic_reference_carrier_is_interpreted():
    envelope = compiled_world(kind="other-context")
    with pytest.raises(ValueError, match="^unsupported_world_reference_source_carrier$"):
        world.world_sources(envelope)


@pytest.mark.parametrize("schema", [None, "supervisor-semantic-worker-context@3", "supervisor-semantic-role-view@1"])
def test_schema_less_and_unknown_bound_artifacts_are_unsupported(schema):
    payload = semantic_payload()
    if schema is None: payload.pop("schema")
    else: payload["schema"] = schema
    with pytest.raises(ValueError, match="^unsupported_world_reference_source_carrier$"):
        world.world_sources(compiled_world(payload=payload))


def test_nested_summary_looking_like_source_without_reference_is_unsupported():
    envelope = compiled_world(kind="other-context")
    member = envelope["packet"]["members"][0]
    member["summary"] = wire(semantic_payload())
    with pytest.raises(ValueError, match="^unsupported_world_reference_source_carrier$"):
        world.world_sources(envelope)
