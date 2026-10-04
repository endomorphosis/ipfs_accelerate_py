"""Complete producer state with a bounded, explicitly partial worker view."""

import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context, load_semantic_worker_context, resolve_semantic_worker_context,
    SemanticContextStale,
)


def test_full_native_index_survives_bounded_projection_and_source_refresh(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = "".join(f"def operation_{i}(value):\n    return value + {i}\n" for i in range(270))
    (root / "code.py").write_text(source)
    (root / "instruction.md").write_text("Inspect operation_42 and preserve public behavior.\n")
    result = prepare_semantic_context(repository=root, paths=["code.py", "instruction.md"],
        required_raw_paths=["instruction.md"], objective="Inspect operation_42", task_id="TASK",
        output=root / ".runtime/context", max_symbols=1024, worker_query="operation_42",
        worker_capsule_limit=3, worker_max_bytes=32768)
    args = dict(repository=root, artifact=".runtime/context/worker-context.json",
        expected_sha256=result["worker_payload_sha256"], task_id="TASK")
    payload = json.loads(load_semantic_worker_context(**args))
    assert result["capsules"] == 271
    assert 0 < result["worker_capsules"] <= 3
    assert len((root / args["artifact"]).read_bytes()) <= 32768
    projection = payload["worker_projection"]
    assert any(item["qualified_name"].endswith(".operation_42") for item in projection["selected_symbols"])
    assert projection["omitted_capsule_count"] == 271 - result["worker_capsules"]
    assert set(payload["raw_sources"]) == {"instruction.md"}
    assert projection["raw_source_fetch_required"]["code.py"] == payload["manifest"]["code.py"]
    assert projection["semantic_equivalence_claimed"] is False
    assert projection["completion_authority"] is False
    # Full producer index/root, not the selected view, is durable and verifiable.
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    blocks = root / ".runtime/context/blocks"
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(result["semantic_root_cid"], lambda cid: (blocks / cid).read_bytes())
    assert view.root.capsule_index_cid == projection["capsule_index_cid"]
    assert result["ducklake"]["status"] == "projected"
    (root / "code.py").write_text(source.replace("return value + 42", "return value + 43"))
    with pytest.raises(SemanticContextStale):
        load_semantic_worker_context(**args)
    refreshed = resolve_semantic_worker_context(**args, refresh_output=root / ".runtime/refresh", attempt_id="TASK:2")
    updated = json.loads(refreshed["text"])
    assert updated["semantic_root_cid"] != payload["semantic_root_cid"]
    assert updated["preparation_bounds"] == payload["preparation_bounds"]
    assert updated["worker_projection"]["query"] == "operation_42"
    assert updated["worker_projection"]["full_capsule_count"] == 271
    assert updated["worker_projection"]["semantic_equivalence_claimed"] is False


@pytest.mark.parametrize("field,value", [("max_symbols", True), ("max_symbols", 1025),
    ("context_input_tokens", 128001), ("worker_max_bytes", 65537)])
def test_projection_bounds_cannot_expand_without_limit(tmp_path, field, value):
    (tmp_path / "code.py").write_text("value=1\n")
    with pytest.raises(ValueError):
        prepare_semantic_context(repository=tmp_path, paths=["code.py"], required_raw_paths=["code.py"],
            objective="Inspect", task_id="T", output=tmp_path / "out", **{field: value})


def test_exact_repack_keeps_required_text_and_full_root_when_estimate_is_small(tmp_path):
    (tmp_path / "code.py").write_text("def inspect(value):\n    return value\n")
    instruction = "Keep this complete public instruction verbatim.\n"
    (tmp_path / "instruction.md").write_text(instruction)
    result = prepare_semantic_context(repository=tmp_path, paths=["code.py", "instruction.md"],
        required_raw_paths=["instruction.md"], objective="Inspect", task_id="T",
        output=tmp_path / "out", worker_query="inspect " * 1000, worker_max_bytes=16384)
    payload = json.loads((tmp_path / "out/worker-context.json").read_text())
    projection = payload["worker_projection"]
    assert projection["exact_byte_budget_repacked"] is True
    assert result["compact_bytes"] <= 16384
    assert payload["raw_sources"]["instruction.md"] == instruction
    assert projection["full_capsule_count"] == result["capsules"] == 2
    assert projection["omitted_capsule_count"] > 0
    assert result["worker_capsules"] == projection["selected_capsule_count"]
    assert payload["pack"]["pack_cid"] == result["pack_cid"]
