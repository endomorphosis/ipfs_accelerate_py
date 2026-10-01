"""Exercise the actual cached learned model and native index boundaries."""

from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.learned_vector_preflight import (
    _model_manifest,
    qualify,
)


REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    / REVISION
)


def test_model_manifest_rejects_executable_snapshot(tmp_path):
    (tmp_path / "custom_model.py").write_text("raise RuntimeError('must not load')\n")
    with pytest.raises(ValueError, match="safetensors"):
        _model_manifest(tmp_path)


@pytest.mark.parametrize(
    ("filename", "source"),
    [("bad.py", "def (broken:\n"), ("bad.unknownextension", "unsupported language")],
)
def test_incomplete_inputs_rejected_before_model_load(tmp_path, filename, source):
    (tmp_path / "worker.py").write_text("def work():\n    return 1\n")
    (tmp_path / filename).write_text(source)
    model = tmp_path / REVISION
    model.mkdir()  # No weights: rejection must precede even model validation.
    with pytest.raises(ValueError, match="AST coverage is incomplete"):
        qualify(tmp_path, tmp_path / "output", ["worker.py", filename], "work", model, REVISION)
    assert not (tmp_path / "output").exists()


def test_existing_learned_model_native_index_roundtrip(tmp_path, monkeypatch):
    if not (MODEL / "model.safetensors").is_file():
        pytest.skip("qualification needs the existing local MiniLM snapshot; never downloads")
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "worker.py").write_text(
        "def verified_seed_predecessor(seed):\n    return seed.parent\n\n"
        "def unrelated_token_count(text):\n    return len(text.split())\n"
    )
    (repository / "package").mkdir()
    (repository / "package/__init__.py").write_text("def initialize_package():\n    pass\n")
    (repository / "__init__.py").write_text("def initialize_root():\n    pass\n")
    result = qualify(
        repository=repository,
        output=tmp_path / "index",
        paths=["worker.py", "package/__init__.py", "__init__.py"],
        query="verified_seed_predecessor",
        model_snapshot=MODEL,
        model_revision=REVISION,
    )
    assert result["status"] == "qualified"
    assert result["dimensions"] == 384
    assert result["symbols"] == result["native_fact_rows_replayed"] == 4
    assert result["canary"]["disposition"] == "passed"
    assert result["policy"]["model_artifact_id"] == result["model_artifact_id"]
    assert result["policy"]["allow_remote"] is False
    assert result["local_model_calls"] == 3  # Native canary, corpus, query.
    assert result["remote_provider_calls"] == 0
    assert result["learned_embeddings"] is True
    assert result["nomination_only"] is True
    assert result["semantic_authority"] is False
    assert result["full_system_qualified"] is False
    assert result["token_savings"] is result["quality_advantage"] is None
    assert result["ducklake"]["status"] == "projected"
    assert result["metadata_retrieval"]["worker.py"]["n"] > 0
    assert result["hits"]["hits"][0]["row"]["symbol"] == "verified_seed_predecessor"
    qualified_symbols = {hit["row"]["qualified_symbol"] for hit in result["hits"]["hits"]}
    assert {"package.initialize_package", "initialize_root"} <= qualified_symbols
    assert result["input_statuses"] == {
        "worker.py": "success",
        "package/__init__.py": "success",
        "__init__.py": "success",
    }
    assert all(trace["fallback_used"] is False for trace in result["router_traces"])
    assert all(trace["provider_used"] == "huggingface" for trace in result["router_traces"])
    assert result["configuration"]["local_files_only"] is True
    assert result["configuration"]["trust_remote_code"] is False
    assert result["configuration"]["use_safetensors"] is True
    assert (tmp_path / "index/model-manifest.json").is_file()
    assert (tmp_path / "index/evidence.json").is_file()
    assert (tmp_path / "index/result.json").is_file()
