"""Native exact-key experiments against the retained public learned-decoder run.

The module fixture performs actual no-fit inference and native SMT/Lean replay.
Absence of the externally produced experiment is a failure, never a skipped or
fabricated successful native proof. All corruption uses disposable copies.
"""
import copy
import json
import os
import shutil
from pathlib import Path

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_proof_index as index


@pytest.fixture(scope="module")
def native_index(tmp_path_factory):
    root = Path(os.environ.get("TERMINAL_CODEBASE_DECODER_EXPERIMENT",
        "/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/decoder-qualification-20261001-01"))
    assert (root / "result.json").is_file(), "actual frozen public decoder experiment required"
    experiment = json.loads((root / "result.json").read_bytes())
    qualification = json.loads((root / "logic/decoder-logic-result.json").read_bytes())
    source = Path(experiment["parent_capture"]["source"]["path"]).read_bytes()
    output = tmp_path_factory.mktemp("native-model-proof-index") / "index"
    manifest = index.persist_terminal_codebase_proof_index(source_bytes=source,
        source_path="bottle.py", decoder_experiment=experiment, qualification=qualification, output=output)
    return {"output": output, "expected": manifest, "source_bytes": source,
        "source_path": "bottle.py", "decoder_experiment": experiment, "qualification": qualification}


def _validate(fixture, **changes):
    return index.validate_terminal_codebase_proof_index(**{**fixture, **changes})


def _copy(fixture, tmp_path):
    output = tmp_path / "index"
    shutil.copytree(fixture["output"], output)
    expected = copy.deepcopy(fixture["expected"])
    expected["output"] = str(output)
    expected["database"] = index._file(output / "proof-index.duckdb", index.MAX_DATABASE_BYTES)
    _seal(output, expected)
    return {**fixture, "output": output, "expected": expected}


def _seal(output, expected):
    expected.pop("manifest_id", None)
    expected["manifest_id"] = index._identity(expected)
    (output / "manifest.json").write_bytes(index._wire(expected) + b"\n")


def test_actual_native_production_and_fresh_process_reconstruction(native_index):
    result = _validate(native_index, fresh_process=True)
    expected = native_index["expected"]
    assert result["entry_count"] == 7 and result["fresh_process_validated"] is True
    assert expected["actual_solver_calls"] == 6 and expected["actual_lean_invocations"] == 2
    assert expected["optimizer_steps"] == result["optimizer_steps"] == 0
    assert result["checker_invocations_here"] == 0
    assert result["source_snapshot"] == expected["source_snapshot"]
    rows = [row["evidence"] for row in result["entries"]]
    assert sum(row["classification"] == "conditional_model_sat_witness" for row in rows) == 4
    assert sum(row["classification"] == "conditional_model_unsat" for row in rows) == 2
    assert sum(row["classification"] == "conditional_model_kernel_checked" for row in rows) == 1
    assert all(row["behavioral_satisfaction"] is False and row["source_semantics_verified"] is False for row in rows)
    assert all(row["premises"] and row["open_frontiers"] for row in rows)
    lean = next(row for row in rows if row["classification"] == "conditional_model_kernel_checked")
    assert lean["checker_receipt"]["compiled_artifacts"]
    assert lean["checker_receipt"]["returncode"] == 0
    assert expected["database"]["bytes"] < index.MAX_DATABASE_BYTES


def test_native_exact_lookup_and_complete_consumer_envelope(native_index):
    expected = native_index["expected"]
    entry = expected["entries"][0]
    envelope = index.lookup_terminal_codebase_model_evidence(output=native_index["output"],
        expected=expected, expected_key=entry["key_relationship"], expected_environment=expected["environment"])
    assert envelope["status"] == "hit"
    assert envelope["evidence"] == entry["evidence"]
    assert index.validate_terminal_codebase_model_evidence_lookup(lookup=envelope,
        expected_entry=entry, expected_environment=expected["environment"]) == envelope


def test_shared_environment_reference_body_is_bound_separately_from_full_inventory(native_index):
    expected = native_index["expected"]
    snapshot = expected["source_snapshot"]
    environment = expected["entries"][0]["key_relationship"]["dimensions"]["environment"]
    assert snapshot["environment_sha256"] == environment["environment_sha256"]
    assert snapshot["environment_ref_sha256"] == index._sha(index._wire(environment))
    altered = copy.deepcopy(environment)
    altered["lean"]["version"] += " forged"
    assert altered["environment_sha256"] == snapshot["environment_sha256"]
    assert index._sha(index._wire(altered)) != snapshot["environment_ref_sha256"]


@pytest.mark.parametrize("field", ["source", "expression", "formalization", "slice", "obligation",
    "assumptions", "bounds", "translation", "provider", "environment", "policy", "schema", "checker",
    "network_policy", "kernel", "theorem_registry"])
def test_every_scoped_identity_change_changes_relationship(native_index, field):
    relationship = native_index["expected"]["entries"][0]["key_relationship"]
    dimensions = copy.deepcopy(relationship["dimensions"])
    if field in {"provider", "checker"}:
        dimensions[field] += "-different"
    elif field == "assumptions":
        dimensions[field].append("additional_unproved_premise")
    else:
        dimensions[field] = {"changed_exact_input": dimensions[field]}
    altered = index.build_terminal_codebase_proof_key_relationship(dimensions=dimensions)
    assert altered["relationship_id"] != relationship["relationship_id"]
    assert altered["accelerate_key_id"] != relationship["accelerate_key_id"]
    # kernel/registry exist on the supervisor side; they are not invented
    # aliases for the datasets schema's independent dimensions.
    if field not in {"kernel", "theorem_registry"}:
        assert altered["datasets_key_id"] != relationship["datasets_key_id"]


def test_current_scoped_key_miss_is_explicit_and_never_behavioral(native_index):
    expected = native_index["expected"]
    dims = copy.deepcopy(expected["entries"][0]["key_relationship"]["dimensions"])
    dims["obligation"] = {"unsupported_new_obligation": True}
    request = index.build_terminal_codebase_proof_key_relationship(dimensions=dims)
    result = index.lookup_terminal_codebase_model_evidence(output=native_index["output"],
        expected=expected, expected_key=request, expected_environment=expected["environment"])
    assert result["status"] == "miss" and result["evidence"] is None
    assert result["behavioral_satisfaction"] is False


def test_stale_current_source_is_rejected(native_index):
    with pytest.raises(ValueError, match="source/dependency capture"):
        _validate(native_index, source_bytes=native_index["source_bytes"] + b"\n")


def test_altered_frozen_decoder_descriptor_is_rejected(native_index):
    experiment = copy.deepcopy(native_index["decoder_experiment"])
    experiment["decoder"]["checkpoint"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="complete artifact"):
        _validate(native_index, decoder_experiment=experiment)


def test_expected_checker_environment_change_is_rejected(native_index):
    expected = native_index["expected"]
    environment = copy.deepcopy(expected["environment"])
    environment["lean"]["version"] += " stale"
    with pytest.raises(ValueError, match="cross-environment"):
        index.lookup_terminal_codebase_model_evidence(output=native_index["output"], expected=expected,
            expected_key=expected["entries"][0]["key_relationship"], expected_environment=environment)


@pytest.mark.parametrize("corruption", ["missing", "altered_receipt", "key_body", "extra"])
def test_native_database_rows_are_not_trusted_even_when_database_pin_is_updated(native_index, tmp_path, corruption):
    fixture = _copy(native_index, tmp_path)
    output, expected = fixture["output"], fixture["expected"]
    with duckdb.connect(str(output / "proof-index.duckdb")) as connection:
        identifier = expected["entries"][0]["entry_id"]
        if corruption == "missing":
            connection.execute("DELETE FROM model_evidence WHERE entry_id = ?", [identifier])
        elif corruption == "altered_receipt":
            row = copy.deepcopy(expected["entries"][0])
            row["evidence"]["checker_receipt"]["solver_answer"] = "unsat"
            connection.execute("UPDATE model_evidence SET row_json = ?, row_sha256 = ? WHERE entry_id = ?",
                [index._wire(row).decode(), index._sha(index._wire(row)), identifier])
        elif corruption == "key_body":
            connection.execute("UPDATE model_evidence SET key_json = '{}' WHERE entry_id = ?", [identifier])
        else:
            connection.execute("INSERT INTO model_evidence VALUES ('forged','forged','{}','{}','forged')")
        connection.execute("CHECKPOINT")
    expected["database"] = index._file(output / "proof-index.duckdb", index.MAX_DATABASE_BYTES)
    _seal(output, expected)
    with pytest.raises(ValueError, match="evidence|integrity"):
        index._database_entries(output, expected)


def test_missing_positive_lean_object_is_rejected_before_native_lookup(native_index):
    expected = native_index["expected"]
    pin = expected["replayed_qualification"]["lean_checks"][0]["compiled_artifacts"][0]
    path = Path(pin["path"])
    backup = path.with_suffix(".temporarily-held")
    path.rename(backup)
    try:
        with pytest.raises(ValueError, match="missing"):
            _validate(native_index)
    finally:
        backup.rename(path)


@pytest.mark.parametrize("field", ["behavioral_satisfaction", "proof_authority", "source_semantics_verified"])
def test_authentic_envelope_cannot_be_promoted_by_caller_flags(native_index, field):
    expected = native_index["expected"]
    entry = expected["entries"][0]
    envelope = index._lookup(entry, expected["environment"])
    envelope["evidence"] = copy.deepcopy(envelope["evidence"])
    envelope["evidence"][field] = True
    with pytest.raises(ValueError, match="complete expected entry"):
        index.validate_terminal_codebase_model_evidence_lookup(lookup=envelope,
            expected_entry=entry, expected_environment=expected["environment"])


def test_old_positive_marker_cannot_replace_actual_checker_replay(native_index, tmp_path):
    altered = copy.deepcopy(native_index["qualification"])
    altered["native_smt_check"]["results"][0]["model_text"] = "caller fabricated witness"
    with pytest.raises(ValueError, match="actual independent checker/inference replay"):
        index.persist_terminal_codebase_proof_index(source_bytes=native_index["source_bytes"],
            source_path="bottle.py", decoder_experiment=native_index["decoder_experiment"],
            qualification=altered, output=tmp_path / "forged-qualification")


def test_truncated_key_relation_is_rejected(native_index):
    key = copy.deepcopy(native_index["expected"]["entries"][0]["key_relationship"])
    del key["dimensions"]["network_policy"]
    with pytest.raises(ValueError, match="closed complete"):
        index.build_terminal_codebase_proof_key_relationship(dimensions=key["dimensions"])
