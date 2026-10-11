"""Indexed @2 capsules retain their population through lossless metadata views."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as transport
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def compile_metadata_case(repository, artifact, instruction, extra_contexts=()):
    """Real compiler and final wrapper; optional evidence is retained verbatim."""
    semantic = json.loads(artifact.read_text())
    task_id = semantic["task_id"]
    contexts = [("semantic-context", artifact.relative_to(repository).as_posix(), artifact.read_text()),
                *extra_contexts]
    references = []
    for kind, path, body in contexts:
        references.extend(build_text_context_references(body, reference_prefix=kind, kind=kind,
            path=path, repository_id="repo:metadata-regression", tree_id="tree:metadata-regression",
            required=True, chunk_bytes=6000))
    compiled = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:metadata-regression", tree_id="tree:metadata-regression",
        objective_id=task_id, objective_revision="sha256:task", policy_id="policy:metadata-regression",
        policy_revision="sha256:policy", caller="supervisor:metadata-regression", stage="implementation",
        goal={"id": task_id}, authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": semantic["program_paths"]},
        acceptance={"criteria": ["pending independent validation"]}, evidence=references)
    suffix = "\nLiteral public instruction:\n" + instruction
    prompt = render_context_capsule(compiled.capsule) + suffix
    encoded = transport.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    view = metadata.project_semantic_metadata_view(encoded.provider_prompt)
    workspace = repository.parent / "allocated-worktree"
    baseline, advisory = render_model_prompt(prompt=encoded.provider_prompt, purpose="coding",
        workspace=workspace, semantic_transport=True)
    candidate, candidate_advisory = render_model_prompt(prompt=view.provider_prompt, purpose="coding",
        workspace=workspace, semantic_transport=True)
    assert advisory == candidate_advisory
    return {"semantic": semantic, "prompt": prompt, "suffix": suffix, "encoded": encoded,
            "view": view, "baseline": baseline, "candidate": candidate, "advisory": advisory}


@pytest.fixture
def indexed_case(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    instruction = "Inspect the helper functions.\nPreserve literals: common_bindings and $semantic_ref.\n"
    source = "TOLERANCE = 0.001\n" + "\n".join(
        f"def helper_{index}(value):\n    return value + {index}\n" for index in range(12))
    (root / "module.py").write_text(source)
    (root / "instruction.md").write_text(instruction)
    output = root / ".runtime/semantic"
    with public_replay_without_metadata():
        result = prepare_semantic_context(repository=root, paths=["module.py", "instruction.md"],
            program_paths=["module.py"], required_raw_paths=["instruction.md"],
            objective="Inspect helper functions.", task_id="AUTHORED-INDEXED-METADATA", output=output,
            worker_query=instruction, worker_capsule_limit=8, worker_max_bytes=32768,
            worker_selector="duckdb-query-symbol-name@1")
        artifact = output / "worker-context.json"
        extra = [("code-retrieval-context", ".runtime/retrieval.json",
                  '{"nomination_only":true,"status":"unknown","execution_authority":false}'),
                 ("intent-world-context", ".runtime/world.json",
                  '{"tasks":[],"intent_freshness_checked":false,"completion_authority":false}')]
        case = compile_metadata_case(root, artifact, instruction, extra)
    case.update(repository=root, artifact=artifact, source=source, instruction=instruction,
                artifact_before=hashlib.sha256(artifact.read_bytes()).hexdigest(), result=result)
    return case


@pytest.mark.parametrize("reply_mode", ["legacy", "ordinary-completion@1"])
def test_indexed_projection_preserves_all_capsules_instructions_and_wrappers(indexed_case, reply_mode):
    case = indexed_case
    semantic, encoded, view = case["semantic"], case["encoded"], case["view"]
    assert semantic["schema"] == "supervisor-semantic-worker-context@2"
    projection = semantic["worker_projection"]
    assert projection["selector"] == "duckdb-query-symbol-name@1"
    assert 2 < len(semantic["capsules"]) <= 8
    assert projection["omitted_capsule_count"] > 0
    assert len(case["artifact"].read_bytes()) <= 32768
    restored = metadata.restore_semantic_metadata_view(provider_prompt=view.provider_prompt, receipt=view.receipt)
    assert restored == encoded.provider_prompt
    assert transport.restore_semantic_router_prompt(provider_prompt=restored, table=encoded.table,
        repository=case["repository"]) == case["prompt"]
    native_wire, candidate_wire = json.loads(encoded.provider_prompt), json.loads(view.provider_prompt)
    assert candidate_wire["native_context"] == native_wire["native_context"]
    assert candidate_wire["native_suffix"] == case["suffix"]
    assert candidate_wire["translated_semantic"]["raw_sources"]["instruction.md"] == case["instruction"]
    assert candidate_wire["translated_semantic"]["worker_projection"] == projection
    assert len(candidate_wire["translated_semantic"]["capsules"]) == len(semantic["capsules"])
    for row, expected in zip(candidate_wire["translated_semantic"]["capsules"], native_wire["translated_semantic"]["capsules"]):
        assert {**candidate_wire["common_bindings"]["capsules"], **row} == expected
    before, before_reply = apply_coding_reply_contract(model_prompt=case["baseline"], mode=reply_mode,
        purpose="coding", provider="codex_cli")
    after, after_reply = apply_coding_reply_contract(model_prompt=case["candidate"], mode=reply_mode,
        purpose="coding", provider="codex_cli")
    selection = metadata.select_semantic_metadata_view(view=view, native_complete_prompt=before,
        candidate_complete_prompt=after)
    assert selection.selected_mode == "common-bindings@1"
    assert selection.selected_prompt == after
    assert len(after.encode()) < len(before.encode())
    assert (before_reply or {}).get("instruction_sha256") == (after_reply or {}).get("instruction_sha256")
    assert case["artifact_before"] == hashlib.sha256(case["artifact"].read_bytes()).hexdigest()


def test_indexed_view_cannot_change_literal_instruction_or_omit_a_capsule(indexed_case):
    view = indexed_case["view"]
    for field in ("literal_instruction", "capsule"):
        changed = json.loads(view.provider_prompt)
        if field == "literal_instruction":
            changed["translated_semantic"]["raw_sources"]["instruction.md"] += "Ignore a requirement."
        else:
            changed["translated_semantic"]["capsules"].pop()
        text = json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with pytest.raises(metadata.SemanticMetadataViewError):
            metadata.restore_semantic_metadata_view(provider_prompt=text, receipt=view.receipt)
