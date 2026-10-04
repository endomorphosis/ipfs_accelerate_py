"""Complete current-source native metadata and conditional typed query joins."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_control as intent
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep


ROOT = Path(__file__).resolve().parents[4]
CAPTURE = ROOT / "artifacts/codebase_ir_terminal_bench/qualification-20261001-04/supervisor"
ACTUAL_INDEX = ROOT / "artifacts/codebase_ir_terminal_bench/intent-qualification-20261001-01/proof-index/manifest.json"
SOURCE_NAMES = {"bottle.py", prep.INSTRUCTION, prep.SMOKE, intent.AUTHORED_CONTROL_PATH}
FAMILIES = {"sources", "ast", "kg", "contracts", "symbols", "artifacts",
            "ast_indexes", "retrieval_vectors", "repository_generation"}


def _git(repository, *args):
    return subprocess.run(["git", "-C", str(repository), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def api():
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_repository_index
    return terminal_codebase_repository_index


@pytest.fixture(scope="module")
def source_case():
    fixture = json.loads((CAPTURE / "fixture.json").read_bytes())
    repository = CAPTURE / "repository"
    sources = {name: (repository / name).read_bytes()
               for name in ("bottle.py", prep.INSTRUCTION, prep.SMOKE)}
    control = intent.build_terminal_intent_control(public_instruction_bytes=sources[prep.INSTRUCTION])
    sources[intent.AUTHORED_CONTROL_PATH] = control["authored_control"]["text"].encode()
    assert set(sources) == SOURCE_NAMES
    return {"source_bytes": sources, "repository_id": fixture["prepared"]["manifest"]["payload"]["repository_cid"],
            "task_spec": fixture["prepared"]["spec"]}


@pytest.fixture(scope="module")
def current(api, source_case):
    return api.build_current_repository_metadata(**source_case)


@pytest.fixture(scope="module")
def proof_manifest():
    return json.loads(ACTUAL_INDEX.read_bytes())


@pytest.fixture(scope="module")
def stored(api, current, proof_manifest, tmp_path_factory):
    output = tmp_path_factory.mktemp("current-native-repository-index") / "index"
    manifest = api.persist_repository_index(current=current, proof_index_manifest=proof_manifest, output=output)
    return {"output": output, "expected": manifest}


def _query():
    return {"source_path": "bottle.py", "symbols": ["_hkey", "_hval"],
            "property": "header_delimiter_rejection", "limit": 8}


def _header_function(raw, name):
    text = raw.decode()
    node = next(item for item in ast.parse(text).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    lines = raw.splitlines(keepends=True)
    offset = lambda line, column: sum(len(part) for part in lines[:line - 1]) + column
    start, end = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
    return node, {"start_byte": start, "end_byte": end,
                  "sha256": hashlib.sha256(raw[start:end]).hexdigest()}


def test_complete_four_source_records_equal_actual_native_semantic_inventory(current, source_case):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _scan_scoped_sources
    native = _scan_scoped_sources(source_case["source_bytes"], repository_id=source_case["repository_id"], max_symbols=1024)
    records = current["records"]
    assert set(records) == FAMILIES
    assert {row["path"] for row in records["sources"]} == SOURCE_NAMES
    assert records["symbols"] == [row.to_dict() for row in native.symbols]
    assert records["artifacts"] == [row.to_dict() for row in native.artifacts]
    assert records["kg"] == [row.to_dict() for row in native.edges]
    assert len(records["symbols"]) == 531
    assert len(records["kg"]) == 4652
    assert len(records["artifacts"]) == 3
    assert {row["path"] for row in records["artifacts"]} >= {prep.INSTRUCTION, intent.AUTHORED_CONTROL_PATH}


@pytest.mark.parametrize("family", ["ast", "kg", "symbols", "retrieval_vectors"])
def test_poisoned_native_metadata_is_refused_before_persist_even_with_rebuilt_manifest(api, current, proof_manifest, family, tmp_path):
    poisoned = deepcopy(current)
    poisoned["records"][family][0]["caller_authored_unverified_field"] = "not in the native producer"
    with pytest.raises(ValueError):
        api.persist_repository_index(current=poisoned, proof_index_manifest=proof_manifest, output=tmp_path / "index")


@pytest.mark.parametrize("family", ["ast", "kg", "symbols", "retrieval_vectors", "sources"])
def test_missing_native_metadata_rows_are_refused_before_persist(api, current, proof_manifest, family, tmp_path):
    poisoned = deepcopy(current)
    assert poisoned["records"][family]
    poisoned["records"][family].pop()
    with pytest.raises(ValueError):
        api.persist_repository_index(current=poisoned, proof_index_manifest=proof_manifest, output=tmp_path / "index")


@pytest.mark.parametrize("family", ["ast", "kg", "symbols", "retrieval_vectors"])
def test_extra_native_metadata_rows_are_refused_before_persist(api, current, proof_manifest, family, tmp_path):
    poisoned = deepcopy(current)
    extra = deepcopy(poisoned["records"][family][0])
    extra["caller_authored_extra_row"] = True
    poisoned["records"][family].append(extra)
    with pytest.raises(ValueError):
        api.persist_repository_index(current=poisoned, proof_index_manifest=proof_manifest, output=tmp_path / "index")


@pytest.mark.parametrize("path", sorted(SOURCE_NAMES))
def test_every_current_source_edit_invalidates_whole_generation(api, source_case, proof_manifest, stored, path):
    changed = deepcopy(source_case)
    changed["source_bytes"][path] += b"\n"
    with pytest.raises(ValueError):
        api.validate_repository_index(**changed, proof_index_manifest=proof_manifest, **stored)
    with pytest.raises(ValueError):
        api.query_repository_index(**changed, proof_index_manifest=proof_manifest, query=_query(), **stored)


@pytest.mark.parametrize("change", ["missing_source", "extra_source", "repository_id", "task_dependency", "task_validation"])
def test_scope_owner_or_task_contract_drift_cannot_reuse_current_index(api, source_case, proof_manifest, stored, change):
    changed = deepcopy(source_case)
    if change == "missing_source":
        changed["source_bytes"].pop(intent.AUTHORED_CONTROL_PATH)
    elif change == "extra_source":
        changed["source_bytes"]["undeclared.py"] = b"def unrelated():\n    return 0\n"
    elif change == "repository_id":
        changed["repository_id"] += ":foreign-repository"
    elif change == "task_dependency":
        changed["task_spec"]["dependencies"] = ["caller-authored-dependency"]
    else:
        changed["task_spec"]["validations"][0]["argv"].append("--caller-authored")
    with pytest.raises(ValueError):
        api.validate_repository_index(**changed, proof_index_manifest=proof_manifest, **stored)


def test_same_head_dirty_edit_is_detected_and_fresh_generation_is_explicit(api, source_case, current, proof_manifest, stored, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    for path, raw in source_case["source_bytes"].items():
        (repository / path).write_bytes(raw)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Isolated current-source test")
    _git(repository, "config", "user.email", "current-source@example.invalid")
    _git(repository, "add", "--", *sorted(SOURCE_NAMES))
    _git(repository, "commit", "-qm", "Capture permitted current sources")
    head = _git(repository, "rev-parse", "HEAD")
    source = repository / "bottle.py"
    old = source.read_bytes()
    source.write_bytes(old.replace(b"def _hval(value):\n    value = touni(value)\n    return value",
                                  b"def _hval(value):\n    value = touni(value)\n    return value + 'dirty-edit'"))
    assert source.read_bytes() != old
    assert _git(repository, "rev-parse", "HEAD") == head
    assert _git(repository, "status", "--porcelain", "--", "bottle.py")
    changed = {**source_case, "source_bytes": {path: (repository / path).read_bytes() for path in SOURCE_NAMES}}
    with pytest.raises(ValueError):
        api.query_repository_index(**changed, proof_index_manifest=proof_manifest, query=_query(), **stored)
    rebuilt = api.build_current_repository_metadata(**changed)
    assert rebuilt["source_snapshot"] != current["source_snapshot"]
    old_symbols = {row["qualified_name"]: row for row in current["records"]["symbols"]}
    new_symbols = {row["qualified_name"]: row for row in rebuilt["records"]["symbols"]}
    assert new_symbols["bottle._hval"]["version_cid"] != old_symbols["bottle._hval"]["version_cid"]
    output = tmp_path / "fresh-current-index"
    expected = api.persist_repository_index(current=rebuilt, output=output)
    assert api.validate_repository_index(**changed, expected=expected, output=output)
    assert _git(repository, "rev-parse", "HEAD") == head
    assert source.read_bytes() == changed["source_bytes"]["bottle.py"]


@pytest.mark.parametrize("path", ["../bottle.py", "/bottle.py", "./bottle.py", "unsafe\\bottle.py", ""])
def test_uncanonical_source_paths_are_refused_before_native_scanner(api, source_case, path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid source path reached filesystem-backed native scan")
    monkeypatch.setattr(semantic_context_runtime, "_scan_scoped_sources", forbidden)
    changed = deepcopy(source_case)
    changed["source_bytes"][path] = changed["source_bytes"].pop("bottle.py")
    with pytest.raises(ValueError):
        api.build_current_repository_metadata(**changed)


def test_passive_native_extraction_never_executes_source(api, source_case):
    changed = deepcopy(source_case)
    changed["source_bytes"]["bottle.py"] = (b"raise AssertionError('passive source must not execute')\n"
                                             + changed["source_bytes"]["bottle.py"])
    reconstructed = api.build_current_repository_metadata(**changed)
    assert {row["path"] for row in reconstructed["records"]["sources"]} == SOURCE_NAMES
    assert any(row["qualified_name"] == "bottle._hkey" for row in reconstructed["records"]["symbols"])


def test_cold_native_sql_replay_preserves_all_rows_and_authority(api, source_case, proof_manifest, stored, current):
    before = {name: (stored["output"] / name).read_bytes()
              for name in ("manifest.json", "records.json", "repository-index.duckdb")}
    report = api.validate_repository_index(**source_case, proof_index_manifest=proof_manifest,
                                          fresh_process=True, **stored)
    assert report["status"] == "verified"
    assert report["fresh_process_validated"] is True
    assert report["complete_native_producers_reconstructed"] is True
    assert report["complete_source_count"] == 4
    assert report["native_table_counts"] == stored["expected"]["native_table_counts"]
    assert report["source_snapshot_id"] == current["source_snapshot"]["snapshot_id"]
    assert report["training_steps"] == report["checker_invocations"] == 0
    assert report["source_body_execution"] is False
    assert all(report[field] is False for field in api.AUTHORITY)
    assert {name: (stored["output"] / name).read_bytes() for name in before} == before


def test_native_typed_query_joins_exact_source_units_proofs_contracts_and_lexical_rows(api, source_case, proof_manifest, stored, current):
    result = api.query_repository_index(**source_case, proof_index_manifest=proof_manifest, query=_query(), **stored)
    assert result["status"] == "nominated_conditional_model_only"
    assert len(result["joins"]) == 8  # Six SMT rows plus both symbols of one Lean model row.
    assert set(result["nominated_entry_ids"]) == {row["entry_id"] for row in proof_manifest["entries"]}
    symbols = {row["qualified_name"]: row for row in current["records"]["symbols"]}
    raw = source_case["source_bytes"]["bottle.py"]
    for joined in result["joins"]:
        symbol = symbols[joined["qualified_symbol"]]
        node, span = _header_function(raw, joined["qualified_symbol"].split(".")[-1])
        assert joined["stable_symbol_id"] == symbol["stable_id"]
        assert joined["symbol_version_cid"] == symbol["version_cid"]
        assert joined["source_path"] == "bottle.py"
        assert joined["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert (joined["start_byte"], joined["end_byte"], joined["span_sha256"]) == (
            span["start_byte"], span["end_byte"], span["sha256"])
        assert joined["source_ast_sha256"] == hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        assert joined["lexical_row_id"] is not None
        assert joined["generation_id"] == current["source_snapshot"]["generation_id"]
        assert all(joined[field] is False for field in api.AUTHORITY)
        assert joined["runtime_refutation"] is False
    assert result["complete_lexical_inventory_count"] == len(current["records"]["retrieval_vectors"])
    assert result["complete_lexical_inventory_count"] > _query()["limit"]
    assert result["ranked_lexical_context"] is not None
    assert result["current_behavioral_facts"] == result["behavioral_satisfied_requirements"] == []
    assert {"query_semantic_alignment_unproved", "request_domain_coverage_unproved",
            "conditional_model_source_semantics_unqualified"} <= set(result["reasons"])
    assert all(result[field] is False for field in api.AUTHORITY)


@pytest.mark.parametrize("symbol", ["_hkey", "_hval"])
def test_dependency_footprint_does_not_nominate_other_headers_specific_obligations(api, source_case, proof_manifest, stored, symbol):
    query = {**_query(), "symbols": [symbol]}
    result = api.query_repository_index(**source_case, proof_index_manifest=proof_manifest, query=query, **stored)
    admitted = {row["entry_id"] for row in proof_manifest["entries"]
                if row["evidence"]["symbol"] == symbol
                or row["evidence"]["symbol"] == "_hkey+_hval"}
    other = {row["entry_id"] for row in proof_manifest["entries"]
             if row["evidence"]["symbol"] not in {symbol, "_hkey+_hval"}}
    assert set(result["nominated_entry_ids"]) == admitted
    assert set(result["nominated_entry_ids"]).isdisjoint(other)
    assert len(result["joins"]) == 4
    assert {row["qualified_symbol"] for row in result["joins"]} == {"bottle." + symbol}
    assert result["current_behavioral_facts"] == result["behavioral_satisfied_requirements"] == []


@pytest.mark.parametrize("change", ["unsupported_property", "duplicate_symbol"])
def test_unsupported_or_ambiguous_focus_stays_unknown_without_behavioral_upgrade(api, source_case, proof_manifest, stored, current, change):
    query = _query()
    if change == "unsupported_property":
        query["property"] = "caller-authored-general-vulnerability-freedom"
    else:
        query["symbols"] = ["_hkey", "_hkey"]
    result = api.query_repository_index(**source_case, proof_index_manifest=proof_manifest, query=query, **stored)
    assert result["status"] == "unknown"
    assert result["joins"] == result["nominated_entry_ids"] == []
    assert result["ranked_lexical_context"] is None
    assert result["complete_lexical_inventory_count"] == len(current["records"]["retrieval_vectors"])
    assert result["current_behavioral_facts"] == result["behavioral_satisfied_requirements"] == []
    assert all(result[field] is False for field in api.AUTHORITY)


@pytest.mark.parametrize("change", ["behavioral_facts", "eligible_fact", "excessive_limit", "boolean_limit"])
def test_closed_bounded_query_refuses_caller_semantics_and_unbounded_controls(api, source_case, proof_manifest, stored, change):
    query = _query()
    if change == "behavioral_facts":
        query["current_behavioral_facts"] = ["caller-authored-success"]
    elif change == "eligible_fact":
        query["eligible"] = True
    elif change == "excessive_limit":
        query["limit"] = 17
    else:
        query["limit"] = True
    with pytest.raises(ValueError):
        api.query_repository_index(**source_case, proof_index_manifest=proof_manifest, query=query, **stored)


def test_absent_conditional_index_keeps_live_lexical_context_and_explicit_unknown(api, source_case, current, tmp_path):
    output = tmp_path / "current-without-conditional-proofs"
    expected = api.persist_repository_index(current=current, output=output)
    result = api.query_repository_index(**source_case, expected=expected, output=output, query=_query())
    assert result["status"] == "unknown"
    assert result["joins"] == result["nominated_entry_ids"] == []
    assert result["ranked_lexical_context"] is not None
    assert "no_exact_conditional_evidence_join_means_unknown" in result["reasons"]
    assert result["current_behavioral_facts"] == result["behavioral_satisfied_requirements"] == []
    assert all(result[field] is False for field in api.AUTHORITY)


def _clone_index(stored, tmp_path):
    output = tmp_path / "copied-isolated-index"
    shutil.copytree(stored["output"], output)
    expected = deepcopy(stored["expected"])
    expected["output"] = str(output)
    return output, expected


def _reseal_files(api, output, expected):
    for name, reference in expected["files"].items():
        path = output / reference["relative_path"]
        raw = path.read_bytes()
        expected["files"][name] = {**reference, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    expected.pop("manifest_id", None)
    expected["manifest_id"] = api._identity(expected)
    (output / "manifest.json").write_bytes(api._wire(expected) + b"\n")


@pytest.mark.parametrize("table,statement", [
    ("symbols", "UPDATE symbols SET start_byte=start_byte+1 WHERE qualified_symbol='bottle._hkey'"),
    ("kg_edges", "UPDATE kg_edges SET confidence='exact' WHERE edge_id=(SELECT edge_id FROM kg_edges WHERE confidence='conservative' ORDER BY edge_id LIMIT 1)"),
    ("ast_facts", "DELETE FROM ast_facts WHERE fact_id=(SELECT fact_id FROM ast_facts ORDER BY fact_id LIMIT 1)"),
    ("lexical_vectors", "UPDATE lexical_vectors SET embedding=[0.0] WHERE row_id=(SELECT row_id FROM lexical_vectors ORDER BY row_id LIMIT 1)"),
    ("conditional_proofs", "UPDATE conditional_proofs SET source_ast_sha256='caller-authored-ast'"),
])
def test_changed_typed_database_rows_fail_native_reconstruction_with_updated_file_pins(api, source_case, proof_manifest, stored, table, statement, tmp_path):
    output, expected = _clone_index(stored, tmp_path)
    with api._connect(output / "repository-index.duckdb") as connection:
        connection.execute(statement)
        connection.execute("CHECKPOINT")
    _reseal_files(api, output, expected)
    with pytest.raises(ValueError, match="native complete row reconstruction differs: " + table):
        api.validate_repository_index(**source_case, proof_index_manifest=proof_manifest, expected=expected, output=output)


def test_extra_typed_row_fails_even_with_updated_native_identity_and_all_file_pins(api, source_case, proof_manifest, stored, tmp_path):
    output, expected = _clone_index(stored, tmp_path)
    with api._connect(output / "repository-index.duckdb") as connection:
        row = list(connection.execute("SELECT * FROM kg_edges ORDER BY edge_id LIMIT 1").fetchone())
        row[0] += ":caller-authored-extra"
        connection.execute("INSERT INTO kg_edges VALUES (" + ",".join("?" for _ in row) + ")", row)
        actual_rows = {name: connection.execute("SELECT * FROM " + name).fetchall() for name in api._TABLES}
        expected["identity"]["rows_sha256"] = api._rows_digest(actual_rows)
        expected["identity"]["table_counts"] = {name: len(rows) for name, rows in actual_rows.items()}
        expected["native_table_counts"] = expected["identity"]["table_counts"]
        connection.execute("UPDATE repository_identity SET identity_json=?", [api._wire(expected["identity"]).decode()])
        connection.execute("CHECKPOINT")
    _reseal_files(api, output, expected)
    with pytest.raises(ValueError, match="current complete native row/proof/schema identity differs"):
        api.validate_repository_index(**source_case, proof_index_manifest=proof_manifest, expected=expected, output=output)


def test_changed_complete_records_refuse_even_with_updated_records_file_pin(api, source_case, proof_manifest, stored, tmp_path):
    output, expected = _clone_index(stored, tmp_path)
    records = json.loads((output / "records.json").read_bytes())
    records["symbols"][0]["caller-authored-field"] = "not in native source"
    (output / "records.json").write_bytes(api._wire(records) + b"\n")
    _reseal_files(api, output, expected)
    with pytest.raises(ValueError, match="persisted complete records differ from current native producers"):
        api.validate_repository_index(**source_case, proof_index_manifest=proof_manifest, expected=expected, output=output)


def test_column_rename_refuses_verified_schema_despite_unchanged_row_tuples_and_updated_file_pin(api, source_case, proof_manifest, stored, tmp_path):
    output, expected = _clone_index(stored, tmp_path)
    with api._connect(output / "repository-index.duckdb") as connection:
        before = connection.execute("SELECT * FROM sources").fetchall()
        connection.execute("ALTER TABLE sources RENAME path TO caller_path")
        assert connection.execute("SELECT * FROM sources").fetchall() == before
        connection.execute("CHECKPOINT")
    _reseal_files(api, output, expected)
    with pytest.raises(ValueError, match="schema|catalog|column"):
        api.validate_repository_index(**source_case, proof_index_manifest=proof_manifest, expected=expected, output=output)


def test_resealed_conditional_manifest_cannot_claim_prompt_semantic_alignment(api, current, proof_manifest, tmp_path):
    forged = deepcopy(proof_manifest)
    forged["semantic_alignment_verified"] = True
    forged.pop("manifest_id", None)
    forged["manifest_id"] = api._identity(forged)
    with pytest.raises(ValueError, match="alignment|authority"):
        api.persist_repository_index(current=current, proof_index_manifest=forged, output=tmp_path / "index")
