"""Native scoped absence must survive replay without invented embeddings."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import empty_code_retrieval as empty
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    load_code_retrieval_context, prepare_empty_code_retrieval_context,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _scope(tmp_path, program=True):
    root = tmp_path / "repo"
    root.mkdir()
    support = {}
    for name, role, text in (("instruction.md", "instruction", "Create answer.py.\n"),
            ("task.json", "task_profile", "{}\n"),
            ("smoke.py", "structural_smoke", "def support_check():\n    return True\n")):
        (root / name).write_text(text)
        support[name] = {"role": role, "sha256": _sha(text.encode())}
    paths = ["empty.py"] if program else []
    if program:
        (root / "empty.py").write_text("# No addressable declarations.\n")
    return root, paths, support


def _prepared(tmp_path, program=True):
    root, paths, support = _scope(tmp_path, program)
    result = prepare_empty_code_retrieval_context(repository=root, task_id="EMPTY-001",
        query_text="Create answer.py.\n", program_paths=paths, support_hashes=support,
        output=root / ".runtime/retrieval.json")
    args = dict(repository=root, artifact=".runtime/retrieval.json",
        expected_sha256=result["sha256"], task_id="EMPTY-001")
    return root, paths, support, result, args


@pytest.mark.parametrize("program", [False, True])
def test_actual_scoped_empty_population_replays_without_harness_symbols(tmp_path, program):
    root, paths, support, result, args = _prepared(tmp_path, program)
    before = (root / args["artifact"]).read_bytes()
    payload, context = empty.read_empty_code_retrieval_artifact(**args)
    assert json.loads(load_code_retrieval_context(**args)) == context
    assert empty.validate_empty_worker_context(context) is None
    assert payload["schema"] == empty.SCHEMA and context["retrieval_schema"] == empty.SCHEMA
    assert context["status"] == "current" and context["stale_paths"] == []
    assert context["disposition"] == ("zero_qualified_symbols" if program else "no_program_inputs")
    assert context["program_paths"] == paths and context["support_hashes"] == support
    assert context["native_scan"]["path_count"] == len(paths)
    assert context["native_scan"]["qualified_symbol_count"] == 0
    assert context["native_scan"]["coverage_complete"] is True
    assert set(context["source_sha256"]) == set(paths)
    assert result["index_id"] is context["index_id"] is None
    assert result["source_population_cid"] == context["source_population_cid"]
    assert context["embedding_calls"] == 0 and context["hits"] == []
    assert all(context[key] is False for key in empty.FALSE)
    assert "snapshot" not in payload and "dimensions" not in payload and "config" not in payload
    assert (root / args["artifact"]).read_bytes() == before
    # The successful Python support file contains a genuine function, but is
    # bound only as support and never becomes the task's program population.
    assert "support_check" not in json.dumps(context)


@pytest.mark.parametrize("program", [False, True])
@pytest.mark.parametrize("change", ["instruction.md", "task.json", "smoke.py"])
@pytest.mark.parametrize("deleted", [False, True])
def test_support_drift_returns_stale_without_reclassifying_harness(tmp_path, program, change, deleted):
    root, _, support, result, args = _prepared(tmp_path, program)
    path = root / change
    if deleted:
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    context = json.loads(load_code_retrieval_context(**args))
    assert context["status"] == "stale" and context["stale_paths"] == [change]
    assert context["hits"] == [] and context["index_id"] is None
    assert context["source_population_cid"] == result["source_population_cid"]
    assert context["support_hashes"][change]["role"] == support[change]["role"]
    assert context["support_hashes"][change]["sha256"] == (None if deleted else _sha(path.read_bytes()))
    assert empty.validate_empty_worker_context(context) is None
    assert context["original_support_hashes"] == support


@pytest.mark.parametrize("new_source", [None, b"def new_function():\n    return 1\n", b"\xff\x00"])
def test_program_drift_is_stale_without_reusing_historical_absence(tmp_path, new_source):
    root, _, _, result, args = _prepared(tmp_path)
    path = root / "empty.py"
    if new_source is None:
        path.unlink()
    else:
        path.write_bytes(new_source)
    context = json.loads(load_code_retrieval_context(**args))
    assert context["status"] == "stale" and context["stale_paths"] == ["empty.py"]
    assert context["hits"] == [] and context["source_population_cid"] == result["source_population_cid"]


def test_symbol_bearing_native_scan_is_not_an_empty_observation(tmp_path):
    root, paths, support = _scope(tmp_path)
    (root / "empty.py").write_text("def genuine():\n    return 1\n")
    assert empty.observe_empty_program_population(repository=root,
        program_paths=paths, support_hashes=support) is None
    with pytest.raises(ValueError, match="qualified symbols"):
        prepare_empty_code_retrieval_context(repository=root, task_id="EMPTY-001", query_text="Create",
            program_paths=paths, support_hashes=support, output=root / "retrieval.json")
    assert not (root / "retrieval.json").exists()


@pytest.mark.parametrize("name,text", [("bad.py", "def broken(:\n"), ("unknown.zzz", "opaque data\n")])
def test_parse_failure_or_unsupported_language_cannot_be_relabeled_empty(tmp_path, name, text):
    root, _, support = _scope(tmp_path, False)
    (root / name).write_text(text)
    with pytest.raises(ValueError, match="complete successful native"):
        empty.observe_empty_program_population(repository=root, program_paths=[name], support_hashes=support)


@pytest.mark.parametrize("change", ["duplicate", "escape", "dot", "absolute", "overlap", "support_digest", "support_extra", "role"])
def test_noncanonical_or_unbound_scope_is_rejected_before_scan(tmp_path, monkeypatch, change):
    root, paths, support = _scope(tmp_path)
    if change == "duplicate": paths *= 2
    elif change == "escape": paths = ["../empty.py"]
    elif change == "dot": paths = ["./empty.py"]
    elif change == "absolute": paths = [str(root / "empty.py")]
    elif change == "overlap": paths = ["smoke.py"]
    elif change == "support_digest": support["smoke.py"]["sha256"] = "main"
    elif change == "support_extra": support["smoke.py"]["include_as_program"] = True
    else: support["smoke.py"]["role"] = "arbitrary role with whitespace"
    monkeypatch.setattr(empty, "build_program_evidence_index", lambda *a, **k: pytest.fail("invalid scope scanned"))
    with pytest.raises(ValueError):
        empty.observe_empty_program_population(repository=root, program_paths=paths, support_hashes=support)


def _write_rehashed(root, args, payload):
    payload["result_id"] = empty._result_id(payload)
    raw = empty._wire(payload)
    (root / args["artifact"]).write_bytes(raw)
    return {**args, "expected_sha256": _sha(raw)}


@pytest.mark.parametrize("change", ["authority", "hits", "index", "dimensions", "scan_count", "population", "query", "task"])
def test_rehashed_empty_artifact_cannot_forge_identity_or_authority(tmp_path, change):
    root, _, _, _, args = _prepared(tmp_path)
    payload = json.loads((root / args["artifact"]).read_bytes())
    if change == "authority": payload["proof_authority"] = True
    elif change == "hits": payload["hits"] = [{"symbol": "invented"}]
    elif change == "index": payload["index_id"] = "invented-vector"
    elif change == "dimensions": payload["dimensions"] = 1
    elif change == "scan_count": payload["native_scan"]["qualified_symbol_count"] = 1
    elif change == "population": payload["source_population_cid"] = "invented-population"
    elif change == "query": payload["query_text"] = "Different query"
    else: payload["task_id"] = "OTHER"
    with pytest.raises(ValueError):
        load_code_retrieval_context(**_write_rehashed(root, args, payload))


def test_rehashed_marker_over_current_real_symbols_fails_native_replay(tmp_path):
    root, paths, support, _, args = _prepared(tmp_path)
    payload = json.loads((root / args["artifact"]).read_bytes())
    source = "def genuine():\n    return 1\n"
    (root / "empty.py").write_text(source)
    payload["source_sha256"]["empty.py"] = _sha(source.encode())
    payload["native_scan"]["ast_index_id"] = empty.build_program_evidence_index({"empty.py": source}).ast_index.index_id
    payload["source_population_cid"] = empty._population(paths, payload["source_sha256"],
        support, payload["native_scan"], payload["disposition"])
    payload["query_id"] = empty._query_id(payload)
    with pytest.raises(ValueError, match="native absence does not replay"):
        load_code_retrieval_context(**_write_rehashed(root, args, payload))


@pytest.mark.parametrize("changed", ["empty.py", "smoke.py"])
def test_native_scan_cannot_change_captured_source_or_support(tmp_path, monkeypatch, changed):
    root, paths, support = _scope(tmp_path)
    scan = empty.build_program_evidence_index
    def mutate(sources):
        result = scan(sources)
        (root / changed).write_bytes((root / changed).read_bytes() + b"\n")
        return result
    monkeypatch.setattr(empty, "build_program_evidence_index", mutate)
    with pytest.raises(ValueError, match="changed during native observation"):
        empty.observe_empty_program_population(repository=root, program_paths=paths, support_hashes=support)


@pytest.mark.parametrize("changed", ["artifact", "caller"])
def test_replay_closing_fence_rejects_artifact_or_caller_mutation(tmp_path, monkeypatch, changed):
    root, _, _, _, args = _prepared(tmp_path)
    payload = json.loads((root / args["artifact"]).read_bytes())
    before = deepcopy(payload)
    scan = empty.build_program_evidence_index
    def mutate(sources):
        result = scan(sources)
        if changed == "artifact":
            path = root / args["artifact"]
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            payload["proof_authority"] = True
        return result
    monkeypatch.setattr(empty, "build_program_evidence_index", mutate)
    with pytest.raises(ValueError, match="changed during"):
        if changed == "artifact":
            empty.read_empty_code_retrieval_artifact(**args)
        else:
            empty.validate_empty_code_retrieval_artifact(payload, repository=root, task_id=args["task_id"])
    assert before["proof_authority"] is False


@pytest.mark.parametrize("change", ["original_hash", "role", "status", "stale_paths", "extra", "authority"])
def test_historical_worker_validator_rejects_forged_projection_without_live_files(tmp_path, monkeypatch, change):
    root, _, _, _, args = _prepared(tmp_path)
    context = json.loads(load_code_retrieval_context(**args))
    if change == "original_hash": context["original_source_sha256"]["empty.py"] = "a" * 64
    elif change == "role": context["support_hashes"]["smoke.py"]["role"] = "program"
    elif change == "status": context["status"] = "stale"
    elif change == "stale_paths": context["stale_paths"] = ["smoke.py"]
    elif change == "extra": context["trust_saved"] = True
    else: context["proof_authority"] = True
    monkeypatch.setattr(empty, "_read", lambda *a, **k: pytest.fail("historical audit read live bytes"))
    monkeypatch.setattr(empty, "build_program_evidence_index", lambda *a, **k: pytest.fail("historical audit claimed native replay"))
    with pytest.raises(ValueError):
        empty.validate_empty_worker_context(context)
