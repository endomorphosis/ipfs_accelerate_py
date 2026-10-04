"""Source correspondence controls and actual native header-model checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_decoder_logic as api
from test.api.test_terminal_codebase_logic_qualification import PROGRAM


def candidates(source=PROGRAM.encode()):
    units = api._source_units(source)
    rows = []
    for name in ("clean_label", "clean_payload"):
        unit = units[name]
        rows.append({"unit_id": "oracle:" + name, "qualified_name": name, "symbol": name,
            "source_path": "headers.py", "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_binding": unit["binding"], "source_ast_sha256": unit["source_ast_sha256"],
            "normalized_body_sha256": hashlib.sha256(unit["body"]).hexdigest(),
            "candidate_source": unit["body"].decode(), "split": "authored_oracle_control"})
    return rows


@pytest.fixture(scope="module")
def qualified_oracle(tmp_path_factory):
    return api.qualify_terminal_codebase_decoder_oracle(source_bytes=PROGRAM.encode(),
        source_path="headers.py", candidates=candidates(), output=tmp_path_factory.mktemp("decoder-oracle") / "logic")


def test_exact_oracle_control_runs_actual_checkers_without_learned_attribution(qualified_oracle):
    report = qualified_oracle
    assert report["status"] == "qualified_oracle_control"
    assert report["candidate_generation"] == "authored_oracle_control"
    assert report["decoder_provenance"]["checkpoint"] is None
    assert report["decoder_inference_independently_replayed"] is False
    assert report["model_generated_function_candidate_count"] == 0
    assert report["learned_source_matched_header_candidate_count"] == 0
    assert report["learned_formula_count"] == 0
    assert report["header_candidate_coverage_complete"]
    assert report["solver_calls"] == 6
    assert report["native_smt_check"]["status"] == "checked_local_model"
    assert [row["status"] for row in report["lean_checks"]] == ["passed", "rejected"]
    assert all(row["matches_expectation"] and row["backend_executed"] for row in report["lean_checks"])
    artifact = report["lean_checks"][0]["compiled_artifacts"][0]
    raw = Path(artifact["path"]).read_bytes()
    assert len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
    assert all(report[key] is False for key in api.AUTHORITY)
    assert report["canonical_tasks_created"] is report["source_executed"] is False
    assert len(report["metadata_records"]) == 8
    assert json.loads((Path(report["output"]) / "decoder-logic-result.json").read_bytes()) == report


def test_exact_source_asts_retain_original_byte_maps_and_typed_frontiers(qualified_oracle):
    for row in qualified_oracle["candidate_source_correspondence"]:
        assert row["candidate_ast_matches_source"]
        assert row["candidate_ast_sha256"] == row["source_ast_sha256"]
        binding = row["source_binding"]
        raw = PROGRAM.encode()[binding["start_byte"]:binding["end_byte"]]
        assert hashlib.sha256(raw).hexdigest() == binding["span_sha256"]
        assert row["pure_v2_status"] == "unsupported"
        assert row["pure_v2_frontier"] == "unsupported_expression:Call"
    inventory = {row["family_id"]: row for row in qualified_oracle["family_inventory"]}
    assert len(inventory) == 40
    assert inventory["first_order"]["status"] == "checked_local_header_string_model"
    assert inventory["propositional"]["lean_compiled"] is True
    assert all(row["learned_family_decoder_trained"] is False for row in inventory.values())
    assert inventory["program"]["status"] == "unsupported"


@pytest.mark.parametrize("mutation", ["wrong_ast", "empty", "missing", "extra_statement"])
def test_incorrect_or_incomplete_candidates_never_reach_checkers(tmp_path, monkeypatch, mutation):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    rows = candidates()
    if mutation == "wrong_ast":
        rows[0]["candidate_source"] = rows[0]["candidate_source"].replace("title()", "lower()")
    elif mutation == "empty":
        rows[0]["candidate_source"] = None
    elif mutation == "missing":
        rows.pop()
    else:
        rows[0]["candidate_source"] += "\nraise RuntimeError('never execute')\n"
    def forbidden(*args, **kwargs):
        pytest.fail("unmatched learned source candidates may not invoke any checker")
    monkeypatch.setattr(header, "check_header_semantics", forbidden)
    monkeypatch.setattr(api.logic, "_compile_lean", forbidden)
    report = api.qualify_terminal_codebase_decoder_oracle(source_bytes=PROGRAM.encode(),
        source_path="headers.py", candidates=rows, output=tmp_path / "not-qualified")
    assert report["status"] == "not_qualified"
    assert report["solver_calls"] == 0 and report["lean_checks"] == []
    assert report["header_candidate_coverage_complete"] is False
    assert report["native_header_derivation"] is None
    assert report["metadata_records"]["decoder_native_header"] == []
    assert all(row["status"] == "unsupported" for row in report["family_inventory"])
    assert report["missing_header_symbols"]


@pytest.mark.parametrize("mutation", ["source_hash", "ast_hash", "span", "path", "body_hash"])
def test_forged_source_identity_rejects_before_output(tmp_path, mutation):
    rows = deepcopy(candidates())
    if mutation == "source_hash":
        rows[0]["source_sha256"] = "0" * 64
    elif mutation == "ast_hash":
        rows[0]["source_ast_sha256"] = "0" * 64
    elif mutation == "span":
        rows[0]["source_binding"]["end_byte"] += 1
    elif mutation == "path":
        rows[0]["source_path"] = "foreign.py"
    else:
        rows[0]["normalized_body_sha256"] = "0" * 64
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="span/hash/AST"):
        api.qualify_terminal_codebase_decoder_oracle(source_bytes=PROGRAM.encode(),
            source_path="headers.py", candidates=rows, output=target)
    assert not target.exists()


def test_unsupported_module_retains_source_candidate_frontier_without_repair(tmp_path):
    raw = b"def identity(value):\n    return value\n"
    unit = api._source_units(raw)["identity"]
    row = {"unit_id": "oracle:identity", "qualified_name": "identity", "symbol": "identity",
        "source_path": "simple.py", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_binding": unit["binding"], "source_ast_sha256": unit["source_ast_sha256"],
        "normalized_body_sha256": hashlib.sha256(unit["body"]).hexdigest(), "candidate_source": raw.decode()}
    report = api.qualify_terminal_codebase_decoder_oracle(source_bytes=raw, source_path="simple.py",
        candidates=[row], output=tmp_path / "unsupported")
    assert report["status"] == "not_qualified" and report["solver_calls"] == 0
    assert report["candidate_source_correspondence"][0]["candidate_ast_matches_source"]
    assert report["candidate_source_correspondence"][0]["pure_v2_status"] == "typed_models_match"
    assert "outside supported header" in report["candidate_source_correspondence"][0]["frontiers"][0]
    assert not report["header_candidate_coverage_complete"]


def test_existing_output_and_duplicate_candidate_bindings_refuse_overwrite(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match="fresh canonical"):
        api.qualify_terminal_codebase_decoder_oracle(source_bytes=PROGRAM.encode(), source_path="headers.py",
            candidates=candidates(), output=target)
    assert sentinel.read_text() == "original"
    rows = candidates()
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        api.qualify_terminal_codebase_decoder_oracle(source_bytes=PROGRAM.encode(), source_path="headers.py",
            candidates=rows, output=tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_unrelated_ambiguous_nested_function_does_not_erase_unique_header_units():
    source = PROGRAM + "\ndef unrelated(flag):\n    if flag:\n        def nested():\n            return 1\n    else:\n        def nested():\n            return 2\n"
    units = api._source_units(source.encode())
    assert units["unrelated.nested"] is None
    assert units["clean_label"]["node"].name == "clean_label"
    assert units["clean_payload"]["node"].name == "clean_payload"


def _public_bottle():
    root = Path(__file__).resolve().parents[4]
    return (root / "artifacts/terminal_bench_supervisor/full-integration-20260929/terminal-source-inputs-01/permitted-inputs/bottle.py").read_bytes()


@pytest.mark.parametrize("claim", ["oracle", "validated_flag", "checkpoint_claim"])
def test_actual_lane_refuses_caller_provenance_claim_before_checkers(tmp_path, monkeypatch, claim):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    def forbidden(*args, **kwargs):
        pytest.fail("unverified decoder provenance must not reach native checking")
    monkeypatch.setattr(header, "check_header_semantics", forbidden)
    monkeypatch.setattr(api.logic, "_compile_lean", forbidden)
    descriptor = {"schema": api.ORACLE_SCHEMA, "candidates": candidates()}
    if claim == "validated_flag":
        descriptor.update(training_artifacts_validated=True, decoder_inference_replayed=True)
    elif claim == "checkpoint_claim":
        descriptor["checkpoint"] = {"weights_sha256": "0" * 64, "validated": True}
    with pytest.raises(ValueError, match="exact decoder experiment report"):
        api.qualify_terminal_codebase_decoder_logic(source_bytes=_public_bottle(), source_path="bottle.py",
            decoder=descriptor, output=tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_actual_lane_invokes_validator_on_full_supplied_source_and_descriptor(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_decoder_training as training
    raw, observed = _public_bottle(), []
    descriptor = {"schema": "caller-claimed-replayed", "validated": True}
    def reject(*, expected, source_bytes, source_path):
        observed.append((expected, source_bytes, source_path))
        raise ValueError("actual native checkpoint refused")
    monkeypatch.setattr(training, "validate_terminal_codebase_decoder", reject)
    with pytest.raises(ValueError, match="actual native checkpoint refused"):
        api.qualify_terminal_codebase_decoder_logic(source_bytes=raw, source_path="bottle.py",
            decoder=descriptor, output=tmp_path / "must-not-exist")
    assert observed == [(descriptor, raw, "bottle.py")]
    assert not (tmp_path / "must-not-exist").exists()


def test_actual_lane_refuses_generic_authored_source_before_replay(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_decoder_training as training
    def forbidden(**kwargs):
        pytest.fail("wrong public source may not reach checkpoint replay")
    monkeypatch.setattr(training, "validate_terminal_codebase_decoder", forbidden)
    with pytest.raises(ValueError, match="exact declared public Bottle"):
        api.qualify_terminal_codebase_decoder_logic(source_bytes=PROGRAM.encode(), source_path="headers.py",
            decoder={"validated": True}, output=tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_nested_same_ast_cannot_replace_top_level_header_candidate(tmp_path):
    source = (PROGRAM + "\ndef outer():\n    def clean_label(raw):\n"
              "        label = convert(raw)\n        return label.title().replace('_', '-')\n").encode()
    units = api._source_units(source)
    rows = candidates(source)
    nested = units["outer.clean_label"]
    rows[0].update(qualified_name="outer.clean_label", source_binding=nested["binding"],
        normalized_body_sha256=hashlib.sha256(nested["body"]).hexdigest(),
        source_ast_sha256=nested["source_ast_sha256"], candidate_source=nested["body"].decode())
    assert nested["source_ast_sha256"] == units["clean_label"]["source_ast_sha256"]
    report = api.qualify_terminal_codebase_decoder_oracle(source_bytes=source, source_path="headers.py",
        candidates=rows, output=tmp_path / "nested")
    assert report["status"] == "not_qualified"
    assert report["solver_calls"] == 0 and report["lean_checks"] == []
    assert report["missing_header_symbols"] == ["clean_label"]
    assert report["candidate_source_correspondence"][0]["frontiers"] == [
        "nested_source_unit_outside_declared_top_level_header_binding"]
