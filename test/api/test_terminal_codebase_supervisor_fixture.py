"""Real native preparation/training and complete, source-bound fixture export."""
import hashlib
import json
import copy
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_supervisor_fixture as fixture_api
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep


@pytest.fixture
def public_inputs(tmp_path):
    source = tmp_path / "bottle.py"
    # Passive extraction must not import or execute the repository module.
    source.write_text("raise RuntimeError('source must never execute')\n"
                      "def alpha(n):\n    return n + 1\n"
                      "def beta(n):\n    return n * 2\n")
    source.chmod(0o755)
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Inspect bottle.py alpha and beta and report vulnerabilities in report.jsonl.")
    return source, instruction


@pytest.fixture
def prepared_fixture(public_inputs, tmp_path, monkeypatch):
    source, instruction = public_inputs
    def forbidden(*args, **kwargs):
        pytest.fail("codebase fixture may not plan, admit or invoke a worker")
    monkeypatch.setattr(prep, "plan", forbidden)
    monkeypatch.setattr(prep.local, "admit_local_benchmark_plan", forbidden)
    result = fixture_api.prepare_terminal_codebase_fixture(
        source=source, instruction=instruction, output=tmp_path / "experiment")
    return result, source, instruction


def test_real_fixture_preserves_source_and_empty_native_preplanning_boundary(prepared_fixture):
    result, source, instruction = prepared_fixture
    root = Path(result["repository"])
    assert root.joinpath("bottle.py").read_bytes() == source.read_bytes()
    assert root.joinpath(prep.INSTRUCTION).read_bytes() == instruction.read_bytes()
    assert root.joinpath("bottle.py").stat().st_mode & 0o777 == 0o755
    assert result["source_provenance"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    captured = subprocess.run(["git", "-C", str(root), "ls-tree", "--name-only", result["captured_commit"]],
                              check=True, capture_output=True, text=True).stdout.splitlines()
    assert captured == ["bottle.py"]
    assert not root.joinpath("report.jsonl").exists()
    assert not Path(result["state"]).joinpath("admission.json").exists()
    assert result["initial_context"]["world_task_count"] == 0
    assert result["initial_context"]["canonical_tasks_created"] is False
    assert result["provider_calls"] == 0
    assert result["planning_provider_invoked"] is result["admission_performed"] is result["workers_started"] is False
    metrics = result["learner"]["metrics"]
    assert metrics["epochs"] == 24 and metrics["seed"] == 1729
    assert metrics["initial_weights_sha256"] != metrics["final_weights_sha256"]
    assert metrics["native_kernel_calls"] > 0
    assert metrics["after_reconstruction_loss"] < metrics["before_reconstruction_loss"]
    assert metrics["holdout_evaluated"] is False
    assert prep.INSTRUCTION not in result["learner"]["source_hashes"]


def test_complete_metadata_matches_native_source_and_trained_vectors(prepared_fixture):
    result, source, instruction = prepared_fixture
    records = fixture_api.extract_terminal_codebase_metadata_records(result)
    assert {row["path"] for row in records["sources"]} == set(result["prepared"]["worker_inputs"])
    original = next(row for row in records["sources"] if row["path"] == "bottle.py")
    assert original["source_text"].encode() == source.read_bytes()
    assert {row["qualified_name"] for row in records["symbols"] if row["module_path"] == "bottle.py"} == {"bottle", "bottle.alpha", "bottle.beta"}
    assert records["symbols"] == records["semantic_indexes"][0]["symbols"]
    assert records["artifacts"] == records["semantic_indexes"][0]["artifacts"]
    assert records["kg"] == records["semantic_indexes"][0]["edges"]
    assert len(records["retrieval_vectors"]) == result["initial_context"]["indexed_symbols"]
    assert {row["row_id"] for row in records["features"]} == {row["row_id"] for row in records["vectors"]}
    assert len(records["vectors"]) == result["learner"]["sample_count"]
    assert all(len(row["latent"]) == 8 for row in records["vectors"])
    assert all(row["proof_authority"] is False for row in records["vectors"])
    assert records["planning"][0]["empty_world_capture"]["plan_projection"]["tasks"] == []
    assert records["contracts"][1]["analysis"]["status"] == "unsupported"
    assert records["contracts"][1]["candidate_applied"] is False
    assert Path(result["output"]).joinpath("metadata-records.json").is_file()
    before = Path(result["output"]).joinpath("metadata-records.json").stat()
    assert fixture_api.extract_terminal_codebase_metadata_records(result) == records
    after = Path(result["output"]).joinpath("metadata-records.json").stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


@pytest.mark.parametrize("mutation", ["source", "checkpoint", "fixture"])
def test_export_refuses_current_source_checkpoint_or_declaration_drift(prepared_fixture, mutation):
    result, source, instruction = prepared_fixture
    if mutation == "source":
        path = Path(result["repository"]) / "bottle.py"
        path.write_text(path.read_text().replace("n + 1", "n + 9"))
    elif mutation == "checkpoint":
        path = Path(result["learner"]["output"]) / "checkpoint.json"
        value = json.loads(path.read_bytes())
        value["weights"][0][0][0] += 1
        path.chmod(0o600)
        path.write_text(json.dumps(value))
    else:
        result["source_bytes_unchanged"] = False
    with pytest.raises(ValueError):
        fixture_api.extract_terminal_codebase_metadata_records(result)
    assert not Path(result["output"]).joinpath("metadata-records.json").exists()


@pytest.mark.parametrize("mutation", ["evidence", "ast", "foreign_snapshot", "existing_export"])
def test_export_rejects_unbound_native_inventory_before_hydration(prepared_fixture, mutation):
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot

    result, source, instruction = prepared_fixture
    vectors = Path(result["repository"]) / ".runtime/terminal-vectors"
    target = Path(result["output"]) / "metadata-records.json"
    if mutation == "evidence":
        path = vectors / "evidence.json"
        value = json.loads(path.read_bytes())
        value["results"][0]["facts"].append({"invented_fact": "not in captured source"})
        path.write_text(json.dumps(value))
    elif mutation == "ast":
        path = vectors / "ast.json"
        value = json.loads(path.read_bytes())
        value["invented_index_field"] = "not in captured source"
        path.write_text(json.dumps(value))
    elif mutation == "foreign_snapshot":
        expected_id = result["initial_context"]["index_id"]
        with duckdb.connect(str(vectors / "vectors.duckdb"), config={"threads": 1}) as connection:
            row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [expected_id]).fetchone()
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
            foreign = replace(snapshot, config=replace(snapshot.config, model_revision="foreign-generation"))
            assert len(foreign.rows) == len(snapshot.rows)
            assert foreign.index_id != expected_id
            assert CodeVectorIndexSnapshot.from_dict(foreign.to_dict()) == foreign
            connection.execute("UPDATE snapshots SET payload=? WHERE id=?", [json.dumps(foreign.to_dict()), expected_id])
    else:
        target.write_text('{"foreign":[]}')
    with pytest.raises(ValueError, match="AST|snapshot|metadata export"):
        fixture_api.extract_terminal_codebase_metadata_records(result)
    if mutation == "existing_export":
        assert target.read_text() == '{"foreign":[]}'
    else:
        assert not target.exists()


def test_preparation_rejects_existing_output_and_symlink_before_native_work(public_inputs, tmp_path, monkeypatch):
    source, instruction = public_inputs
    def forbidden(**kwargs):
        pytest.fail("invalid acquisition may not enter Supervisor preparation")
    monkeypatch.setattr(prep, "prepare", forbidden)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        fixture_api.prepare_terminal_codebase_fixture(source=source, instruction=instruction, output=existing)
    link = tmp_path / "links"
    link.mkdir()
    symlink = link / "bottle.py"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="canonical"):
        fixture_api.prepare_terminal_codebase_fixture(source=symlink, instruction=instruction, output=tmp_path / "fresh")
    assert not tmp_path.joinpath("fresh").exists()


def _large_metadata():
    nested = {"native_source": "unmodified"}
    for _ in range(35):
        nested = {"child": nested}
    return {"ast": [{"native_id": "small", "all_fields": [1, False, None]}],
            "ast_indexes": [{"native_id": "unicode-body", "source": "αβ\n" * 70_000,
                "other_field": {"proof_authority": False, "all_source_fields": [1, 2, 3]}},
                {"native_id": "deep-container", "nested": nested}], "empty": []}


def test_bounded_metadata_reconstructs_every_original_field_and_is_stable():
    original = _large_metadata()
    bounded = fixture_api.bound_terminal_codebase_metadata_records(original)
    assert bounded["ast"] == original["ast"]
    assert all(row["schema"] == fixture_api.ARTIFACT_SCHEMA for row in bounded["ast_indexes"])
    assert fixture_api.reconstruct_terminal_codebase_metadata_records(bounded) == original
    assert fixture_api.bound_terminal_codebase_metadata_records(bounded) == bounded
    from benchmarks.agent_supervisor.container_coding.codebase_ir_metadata import LIMITS, _json
    assert all(len(_json(row)) <= LIMITS["row_bytes"] for rows in bounded.values() for row in rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "bytes", "orphan", "descriptor"])
def test_bounded_metadata_rejects_incomplete_or_modified_artifacts(mutation):
    bounded = copy.deepcopy(fixture_api.bound_terminal_codebase_metadata_records(_large_metadata()))
    chunks = bounded[fixture_api.CHUNK_FAMILY]
    if mutation == "missing":
        chunks.pop(0)
    elif mutation == "duplicate":
        chunks.append(chunks[0])
    elif mutation == "order":
        chunks[0], chunks[1] = chunks[1], chunks[0]
    elif mutation == "bytes":
        chunks[0]["base64"] = "AAAA"
    elif mutation == "orphan":
        bounded["ast_indexes"].pop(1)
    else:
        bounded["ast_indexes"][0]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        fixture_api.reconstruct_terminal_codebase_metadata_records(bounded)


def test_bounded_metadata_survives_real_native_lake_fresh_process_replay(tmp_path):
    from benchmarks.agent_supervisor.container_coding.codebase_ir_metadata import (
        hydrate_codebase_ir_metadata, validate_codebase_ir_metadata,
    )
    original = _large_metadata()
    records = fixture_api.bound_terminal_codebase_metadata_records(original)
    output = tmp_path / "metadata"
    hydrated = hydrate_codebase_ir_metadata(records=records, output=output,
        source_snapshot={"source_sha256": "test:complete-unicode-source", "checkpoint": "test:unchanged"})
    replay = validate_codebase_ir_metadata(output=output, expected=hydrated, fresh_process=True)
    exports = {family: [json.loads(line)["payload"] for line in (output / reference["relative_path"]).read_text().splitlines()]
               for family, reference in replay["exports"].items()}
    assert fixture_api.reconstruct_terminal_codebase_metadata_records(exports) == {
        **original, "kg": [], "vectors": [], "contracts": []}
