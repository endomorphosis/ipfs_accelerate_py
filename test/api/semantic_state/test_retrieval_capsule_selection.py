"""Native source-bound retrieval changes capsule priority, not source authority.

Vectors are authored two-dimensional test data. Native index/query replay,
capsule production, freshness, packing, DuckLake persistence and reload are real.
"""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import build_code_symbol_vector_index
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_selection import select_worker_capsules
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context, load_semantic_worker_context, resolve_semantic_worker_context,
)
from test.api.semantic_state.test_semantic_capsule_selection import native_scope  # noqa: F401


@pytest.fixture
def ranked_repository(tmp_path):
    source = "def alpha(x):\n    return x + 1\n\ndef zeta(x):\n    return x - 1\n"
    (tmp_path / "code.py").write_text(source)
    (tmp_path / "instruction.md").write_text("Inspect alpha; preserve λ.\n")
    ast = build_program_evidence_index({"code.py": source}).ast_index
    snapshot = build_code_symbol_vector_index(ast, forest_id="test:scope", tree_id="test:tree",
        coverage_id=ast.index_id, dimensions=2, model_id="authored-test-vectors", model_revision="1",
        configuration_id="test:2d", vectors=lambda row: (1., 0.) if row.symbol == "zeta" else (0., 1.))
    result = prepare_code_retrieval_context(repository=tmp_path, task_id="TASK", query_text="alpha",
        snapshot=snapshot, result=snapshot.search((1., 0.), max_results=1), output=tmp_path / "retrieval.json")
    return dict(repository=tmp_path, paths=["code.py", "instruction.md"], program_paths=["code.py"],
        required_raw_paths=["instruction.md"], task_id="TASK", objective="Inspect alpha",
        worker_query="alpha", worker_capsule_limit=1, worker_max_bytes=32768,
        worker_retrieval={"artifact": "retrieval.json", "sha256": result["sha256"]})


def build(args, name="ranked"):
    root = args["repository"]
    result = prepare_semantic_context(**args, output=root / name)
    payload = json.loads((root / name / "worker-context.json").read_bytes())
    load = dict(repository=root, artifact=name + "/worker-context.json",
                expected_sha256=result["worker_payload_sha256"], task_id=args["task_id"])
    return result, payload, load


def test_actual_retrieval_changes_priority_and_cold_replay_keeps_required_source(ranked_repository):
    args = ranked_repository
    _, legacy, _ = build({key: value for key, value in args.items() if key != "worker_retrieval"}, "lexical")
    result, ranked, load = build(args)
    assert legacy["worker_projection"]["selected_symbols"][0]["qualified_name"] == "code.alpha"
    assert ranked["worker_projection"]["selected_symbols"][0]["qualified_name"] == "code.zeta"
    projection = ranked["worker_projection"]
    assert projection["selector"] == "verified-retrieval-symbol-rank@1"
    assert projection["ranked_symbol_ids"] == [projection["selected_symbols"][0]["stable_symbol_id"]]
    assert projection["retrieval"] == args["worker_retrieval"]
    assert ranked["raw_sources"] == legacy["raw_sources"] == {
        "instruction.md": (args["repository"] / "instruction.md").read_text()}
    assert ranked["manifest"] == legacy["manifest"]
    assert ranked["semantic_root_cid"] == legacy["semantic_root_cid"]
    assert result["ducklake"]["status"] == "projected"
    assert json.loads(load_semantic_worker_context(**load)) == ranked


@pytest.mark.parametrize("bad", [[], ["unknown"], "not-an-id-list", [True]])
def test_selector_rejects_unknown_duplicate_or_noncanonical_ids(native_scope, bad):
    if bad == []:
        bad = [native_scope["symbols"][0].stable_id] * 2
    with pytest.raises(ValueError, match="unique current native"):
        select_worker_capsules(**native_scope, worker_query="alpha", worker_capsule_limit=1,
            worker_max_bytes=32768, worker_ranked_symbol_ids=bad)


def test_ranked_selection_still_requires_freshness(native_scope):
    class Reject:
        def assess_capsule_freshness(self, *args, **kwargs):
            raise RuntimeError("freshness unavailable")
    selected = next(s for s in native_scope["symbols"] if s.qualified_name.endswith(".zeta_0"))
    with pytest.raises(RuntimeError, match="freshness unavailable"):
        select_worker_capsules(**dict(native_scope, provider=Reject()), worker_query="alpha",
            worker_capsule_limit=1, worker_max_bytes=65536, worker_ranked_symbol_ids=[selected.stable_id])


@pytest.mark.parametrize("field,value", [("task_id", "FOREIGN"), ("worker_query", "different"),
    ("program_paths", None), ("worker_query", "")])
def test_foreign_task_query_and_implicit_scope_refused(ranked_repository, field, value):
    with pytest.raises(ValueError):
        build({**ranked_repository, field: value})


def test_incomplete_program_population_refused(ranked_repository):
    root = ranked_repository["repository"]
    (root / "other.py").write_text("def other():\n    return 1\n")
    with pytest.raises(ValueError, match="source/query scope"):
        build({**ranked_repository, "paths": [*ranked_repository["paths"], "other.py"],
               "program_paths": ["code.py", "other.py"]})


def test_stale_source_cannot_reuse_old_rank_during_refresh(ranked_repository):
    _, _, load = build(ranked_repository)
    root = ranked_repository["repository"]
    (root / "code.py").write_text((root / "code.py").read_text().replace("x - 1", "x - 7"))
    with pytest.raises(ValueError, match="stale"):
        load_semantic_worker_context(**load)
    with pytest.raises(ValueError, match="source/query scope"):
        resolve_semantic_worker_context(**load, refresh_output=root / "refreshed", attempt_id="next")


@pytest.mark.parametrize("mutation", ["rank", "artifact", "digest", "downgrade", "foreign_selector"])
def test_rehashed_worker_payload_cannot_forge_ranking(ranked_repository, mutation):
    _, payload, load = build(ranked_repository)
    policy = payload["worker_projection"]
    if mutation == "rank":
        policy["ranked_symbol_ids"] = ["unknown"]
    elif mutation == "artifact":
        policy["retrieval"]["artifact"] = "../foreign.json"
    elif mutation == "digest":
        policy["retrieval"]["sha256"] = "0" * 64
    elif mutation == "downgrade":
        policy.update(schema="supervisor-semantic-worker-projection@1", selector="lexical-query-symbol-name@1")
    else:
        policy["selector"] = "unverified-ranking@1"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    (ranked_repository["repository"] / load["artifact"]).write_bytes(raw)
    with pytest.raises(ValueError):
        load_semantic_worker_context(**dict(load, expected_sha256=hashlib.sha256(raw).hexdigest()))


def test_corrupted_retrieval_artifact_refused_before_payload_creation(ranked_repository):
    root = ranked_repository["repository"]
    (root / "retrieval.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        build(ranked_repository)
    assert not (root / "ranked").exists()


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
@pytest.mark.parametrize("changed", ["edited", "removed"])
def test_ranked_historical_router_round_trip_replays_retained_sources(ranked_repository, transport, changed):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, build_text_context_references, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec

    _, _, load = build(ranked_repository)
    root = ranked_repository["repository"]
    text = (root / load["artifact"]).read_text()
    refs = build_text_context_references(text, reference_prefix="semantic-context", kind="semantic-context",
        path=load["artifact"], repository_id="test:repository", tree_id="test:tree", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="test:repository", tree_id="test:tree", objective_id="TASK",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "TASK"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["code.py"]}, acceptance={"criteria": ["pending validation"]}, evidence=refs)
    prompt = render_context_capsule(compiled.capsule)
    original = codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=transport)
    if changed == "edited":
        (root / "code.py").write_text("def repaired(x):\n    return x * 3\n")
    else:
        (root / "code.py").unlink()
        (root / "instruction.md").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=transport)
    replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root, transport_schema=transport)
    assert replay.provider_prompt == original.provider_prompt
    assert replay.table == original.table
    assert replay.native_prompt == original.native_prompt
    assert replay.freshness_checked is False and replay.receipt["freshness_checked"] is False
    assert replay.receipt["completion_authority"] is replay.receipt["execution_authority"] is False
    # A valid historical representation cannot be converted to a current
    # structured reply or dispatch permission after source drift.
    with pytest.raises(ValueError, match="historical"):
        codec.decode_semantic_router_response(response='{"schema":"supervisor-semantic-router-reply@1"}',
            encoded=replay, repository=root)
