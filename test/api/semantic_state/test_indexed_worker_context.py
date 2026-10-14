"""Source replay, budget and recovery boundaries for indexed worker context."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    INDEXED_SELECTOR, load_semantic_worker_context, prepare_semantic_context,
)


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / "instruction.txt").write_text("Preserve addition and café.\n")
    (tmp_path / "code.py").write_text("\n".join(
        f"def add_{i}(value):\n    return value + {i}\n" for i in range(12)))
    output = tmp_path / "context"
    result = prepare_semantic_context(repository=tmp_path,
        paths=["code.py", "instruction.txt"], program_paths=["code.py"],
        required_raw_paths=["instruction.txt"], objective="Preserve addition",
        task_id="indexed-context", output=output, worker_query="add_7",
        worker_selector=INDEXED_SELECTOR, worker_capsule_limit=2)
    args = dict(repository=tmp_path, artifact="context/worker-context.json",
        expected_sha256=result["worker_payload_sha256"], task_id="indexed-context")
    return tmp_path, result, args


def test_current_source_replays_without_mutable_metadata(prepared):
    root, result, args = prepared
    original = load_semantic_worker_context(**args)
    payload = json.loads(original)
    assert payload["worker_projection"]["selected_symbols"][0]["qualified_name"] == "code.add_7"
    assert result["worker_capsules"] == 2 < result["capsules"]
    assert result["compact_bytes"] <= 32768
    assert result["provider_token_usage"] is None
    assert payload["raw_sources"]["instruction.txt"] == (root / "instruction.txt").read_text()
    assert "code.py" in payload["worker_projection"]["raw_source_fetch_required"]
    snapshot = payload["worker_projection"]["candidate_snapshot_cid"]
    assert (root / "context/blocks" / snapshot).is_file()
    # Live source and immutable producer bytes are replay authority, not SQL history.
    (root / "context/metadata.duckdb").unlink()
    assert load_semantic_worker_context(**args) == original


@pytest.mark.parametrize("defect", ["snapshot", "selection", "scope", "selector"])
def test_forged_indexed_nomination_fails_cold_replay(prepared, defect):
    root, _, args = prepared
    artifact = root / args["artifact"]
    payload = json.loads(artifact.read_bytes())
    projection = payload["worker_projection"]
    if defect == "snapshot":
        projection["candidate_snapshot_cid"] = payload["scope_cid"]
    elif defect == "selection":
        projection["selected_symbols"].reverse()
    elif defect == "scope":
        payload["schema"] = "supervisor-semantic-worker-context@1"
    else:
        projection["selector"] = "invented-selector"
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    artifact.write_bytes(raw)
    with pytest.raises(ValueError):
        load_semantic_worker_context(**dict(args, expected_sha256=hashlib.sha256(raw).hexdigest()))


def test_required_source_over_budget_is_never_silently_dropped(tmp_path):
    (tmp_path / "instruction.txt").write_text("must preserve this requirement\n" * 400)
    with pytest.raises(ValueError, match="exceeds configured budget"):
        prepare_semantic_context(repository=tmp_path, paths=["instruction.txt"],
            program_paths=[], required_raw_paths=["instruction.txt"], objective="Preserve",
            task_id="budget", output=tmp_path / "context", worker_query="Preserve",
            worker_selector=INDEXED_SELECTOR, worker_max_bytes=8192)
    assert not (tmp_path / "context").exists()
