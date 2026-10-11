"""Real producer nomination, omitted-capsule expansion and refusal boundaries."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_expansion import (
    expand_semantic_capsules,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    IpfsDatasetsSemanticStateProvider,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SortedPairIndex,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    repository = tmp_path_factory.mktemp("capsule-expansion")
    (repository / "required.py").write_text("def target(value):\n    return value\n")
    (repository / "dependencies.py").write_text(
        "def alpha(value: int) -> int:\n    return value + 1\n\n"
        "def beta(value: int) -> int:\n    return value - 1\n")
    output = repository / ".semantic/context"
    result = prepare_semantic_context(repository=repository,
        paths=["required.py", "dependencies.py"], required_raw_paths=["required.py"],
        objective="Inspect beta while preserving target", task_id="EXPAND-001",
        worker_query="alpha", worker_capsule_limit=1, output=output)
    payload = json.loads((output / "worker-context.json").read_bytes())
    provider = IpfsDatasetsSemanticStateProvider()
    view = provider.open_verified_view(result["semantic_root_cid"],
        lambda cid: (output / "blocks" / cid).read_bytes())
    index = SortedPairIndex.from_dict({
        **json.loads((output / "blocks" / view.root.capsule_index_cid).read_bytes()),
        "index_cid": view.root.capsule_index_cid})
    capsules = {stable: view.capsule(stable).to_dict() for stable, _ in index.pairs}
    selected = {item["stable_symbol_id"] for item in payload["capsules"]}
    omitted = next(stable for stable in capsules if stable not in selected
        and capsules[stable]["source_slice_path"] == "dependencies.py")
    return SimpleNamespace(repository=repository, output=output, payload=payload,
        result=result, capsules=capsules, omitted=omitted, view=view)


@pytest.fixture
def context(produced):
    # Reuse a real producer run while restoring every file tests may change.
    paths = [produced.repository / "required.py", produced.repository / "dependencies.py",
             produced.output / "worker-context.json", *list((produced.output / "blocks").iterdir())]
    originals = {path: path.read_bytes() for path in paths}
    yield produced
    for path in (produced.output / "blocks").iterdir():
        if path not in originals:
            path.unlink()
    for path, raw in originals.items():
        if path.is_symlink():
            path.unlink()
        path.write_bytes(raw)


def args(context):
    return dict(repository=context.repository, artifact=".semantic/context/worker-context.json",
        expected_sha256=context.result["worker_payload_sha256"], task_id="EXPAND-001",
        stable_symbol_ids=[context.omitted])


def nominate(context, payload):
    raw = encoded(payload)
    (context.output / "worker-context.json").write_bytes(raw)
    return dict(args(context), expected_sha256=hashlib.sha256(raw).hexdigest())


def test_expands_omitted_exact_native_capsule_under_whole_response_budget(context):
    response = expand_semantic_capsules(**args(context))
    assert response["capsules"] == [context.capsules[context.omitted]]
    assert response["semantic_root_cid"] == context.result["semantic_root_cid"]
    assert response["capsule_index_cid"] == context.view.root.capsule_index_cid
    assert response["admissions"][0]["freshness"] == "fresh"
    assert response["raw_source_fetch_required"] == {
        "dependencies.py": context.payload["manifest"]["dependencies.py"]}
    assert all(response[field] is False for field in (
        "semantic_equivalence_claimed", "proof_authority", "execution_authority", "completion_authority"))
    size = len(encoded(response))
    assert expand_semantic_capsules(**args(context), max_bytes=size) == response
    with pytest.raises(ValueError, match="response exceeds byte budget"):
        expand_semantic_capsules(**args(context), max_bytes=size - 1)


def test_expansion_preserves_request_order_and_required_raw_admission(context):
    required = next(stable for stable, capsule in context.capsules.items()
        if capsule["source_slice_path"] == "required.py")
    requested = [required, context.omitted]
    response = expand_semantic_capsules(**dict(args(context), stable_symbol_ids=requested))
    assert response["capsules"] == [context.capsules[stable] for stable in requested]
    assert response["admissions"][0]["admission"] == "raw_source_required"


def test_dirty_dependency_invalidates_expansion_without_git_head_change(context):
    (context.repository / "required.py").write_text("def target(value): return 0\n")
    with pytest.raises(ValueError, match="stale"):
        expand_semantic_capsules(**args(context))


@pytest.mark.parametrize("kind", ["root", "index", "capsule"])
@pytest.mark.parametrize("mutation", ["missing", "forged", "symlink"])
def test_missing_tampered_or_symlinked_blocks_never_expand(context, kind, mutation, tmp_path):
    cid = {"root": context.view.root.root_cid, "index": context.view.root.capsule_index_cid,
           "capsule": context.capsules[context.omitted]["capsule_cid"]}[kind]
    path = context.output / "blocks" / cid
    raw = path.read_bytes()
    path.unlink()
    if mutation == "forged":
        path.write_bytes(b"{}")
    elif mutation == "symlink":
        external = tmp_path / "valid-but-outside"
        external.write_bytes(raw)
        path.symlink_to(external)
    with pytest.raises((ValueError, FileNotFoundError)):
        expand_semantic_capsules(**args(context))


def test_nominated_projection_cannot_redirect_capsule_index(context):
    payload = json.loads(encoded(context.payload))
    payload["worker_projection"]["capsule_index_cid"] = context.view.root.symbol_node_index_cid
    with pytest.raises(ValueError, match="capsule index differs"):
        expand_semantic_capsules(**nominate(context, payload))


def test_symbol_outside_native_root_is_rejected(context):
    unknown = cid_for_structured({"outside": "nominated symbol index"})
    with pytest.raises(ValueError, match="unknown stable_symbol_id"):
        expand_semantic_capsules(**dict(args(context), stable_symbol_ids=[unknown]))


def test_valid_capsule_root_cannot_expand_beyond_nominated_source_scope(context):
    payload = json.loads(encoded(context.payload))
    payload["manifest"].pop("dependencies.py")
    payload["scope_cid"] = cid_for_payload({"schema": "supervisor-source-scope@1",
        "sources": payload["manifest"]})
    with pytest.raises(ValueError, match="outside nominated scope"):
        expand_semantic_capsules(**nominate(context, payload))


def test_forged_native_source_binding_is_rejected_even_with_matching_artifact_hash(context):
    payload = json.loads(encoded(context.payload))
    payload["manifest"]["dependencies.py"]["source_cid"] = payload["manifest"]["required.py"]["source_cid"]
    payload["scope_cid"] = cid_for_payload({"schema": "supervisor-source-scope@1",
        "sources": payload["manifest"]})
    with pytest.raises(ValueError, match="outside nominated scope"):
        expand_semantic_capsules(**nominate(context, payload))


@pytest.mark.parametrize("value", [[], "not-a-list", ["../worker-context.json"], ["bogus"],
    ["x"] * 33, [None], [True]])
def test_requested_ids_have_bounded_shape_before_loading_artifact(context, value):
    with pytest.raises(ValueError):
        expand_semantic_capsules(**dict(args(context), stable_symbol_ids=value))


def test_duplicate_ids_are_rejected(context):
    with pytest.raises(ValueError, match="unique"):
        expand_semantic_capsules(**dict(args(context), stable_symbol_ids=[context.omitted] * 2))


@pytest.mark.parametrize("max_bytes", [0, -1, 65537, True, 1.5])
def test_response_budget_is_hard_bounded(context, max_bytes):
    with pytest.raises(ValueError, match="max_bytes"):
        expand_semantic_capsules(**args(context), max_bytes=max_bytes)


def test_expansion_reads_only_requested_blocks_and_catches_mid_read_dirty_edit(context, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_expansion as expansion
    original = expansion._block_reader
    reads = []

    def reader(directory):
        get_block = original(directory)
        def get(cid):
            reads.append(cid)
            raw = get_block(cid)
            if cid == context.capsules[context.omitted]["capsule_cid"]:
                (context.repository / "required.py").write_text("def target(value): return 42\n")
            return raw
        return get

    monkeypatch.setattr(expansion, "_block_reader", reader)
    with pytest.raises(ValueError, match="changed during expansion"):
        expand_semantic_capsules(**args(context))
    unused = {item["capsule_cid"] for stable, item in context.capsules.items() if stable != context.omitted}
    assert not unused.intersection(reads)


def test_cli_returns_exact_native_json_and_failure_has_no_partial_stdout(context, capsys):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_expansion import main
    argv = ["--repository", str(context.repository), "--artifact", args(context)["artifact"],
        "--expected-sha256", args(context)["expected_sha256"], "--task-id", "EXPAND-001",
        "--symbol", context.omitted]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["capsules"] == [context.capsules[context.omitted]]
    assert main([*argv, "--max-bytes", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "response exceeds byte budget" in captured.err


def test_explicit_program_scope_replays_and_expands_real_producer(context):
    output = context.repository / ".semantic/explicit"
    result = prepare_semantic_context(repository=context.repository,
        paths=["dependencies.py", "required.py"], program_paths=["dependencies.py"],
        required_raw_paths=["required.py"], objective="Inspect dependencies",
        task_id="EXPAND-EXPLICIT", output=output, worker_query="alpha", worker_capsule_limit=0)
    payload = json.loads((output / "worker-context.json").read_bytes())
    assert payload["schema"] == "supervisor-semantic-worker-context@2"
    assert payload["capsules"] == []
    response = expand_semantic_capsules(repository=context.repository,
        artifact=".semantic/explicit/worker-context.json", task_id="EXPAND-EXPLICIT",
        expected_sha256=result["worker_payload_sha256"], stable_symbol_ids=[context.omitted])
    assert response["capsules"][0]["stable_symbol_id"] == context.omitted
    assert response["capsules"][0]["source_slice_path"] == "dependencies.py"


def test_source_growth_after_loader_is_bounded(context, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_expansion as expansion
    original = expansion._block_reader
    def reader(directory):
        get_block = original(directory)
        def get(cid):
            raw = get_block(cid)
            if cid == context.capsules[context.omitted]["capsule_cid"]:
                (context.repository / "dependencies.py").write_bytes(b"x" * 1_000_001)
            return raw
        return get
    monkeypatch.setattr(expansion, "_block_reader", reader)
    with pytest.raises(ValueError, match="source path or size changed"):
        expand_semantic_capsules(**args(context))


def test_optional_projection_corruption_cannot_disappear_as_missing(context):
    cid = context.capsules[context.omitted]["relevant_binding_projection_cid"]
    assert cid is not None
    (context.output / "blocks" / cid).write_bytes(b"{}")
    with pytest.raises(ValueError):
        expand_semantic_capsules(**args(context))
