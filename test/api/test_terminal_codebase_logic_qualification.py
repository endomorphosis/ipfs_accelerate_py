"""Actual local checker controls for the bounded source-model qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_logic_qualification as api
from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as contracts


PROGRAM = '''def convert(raw, encoding='utf8', errors='strict'):
    if isinstance(raw, (bytes, bytearray)):
        return str(raw, encoding, errors)
    return '' if raw is None else str(raw)

def clean_label(raw):
    label = convert(raw)
    return label.title().replace('_', '-')

def clean_payload(raw):
    payload = convert(raw)
    return payload

class WireResponse:
    def put(self, label, payload):
        self._values[clean_label(label)] = [clean_payload(payload)]

    @property
    def fields_for_wire(self):
        pairs = list(self._values.items())
        return [(label, value) for label, values in pairs for value in values]

def application(environ, start_response):
    response = WireResponse()
    start_response('200 OK', response.fields_for_wire)
'''
METRICS = {"before_reconstruction_loss": 0.037, "after_reconstruction_loss": 0.023,
           "training_scope": "authored_test_measurements_not_an_actual_training_run"}
LEAN, _ = api._native_lean(None)
TOOLS = LEAN is not None and shutil.which("z3") is not None


@pytest.fixture(scope="module")
def qualified(tmp_path_factory):
    if not TOOLS:
        pytest.skip("installed native Lean and Z3 are required for actual checker qualification")
    return api.qualify_terminal_codebase_logic(source_bytes=PROGRAM.encode(), source_path="headers.py",
        output=tmp_path_factory.mktemp("header-qualification") / "logic", training_metrics=METRICS)


def test_original_header_has_countermodels_without_promoting_expected_behavior(qualified):
    assert qualified["status"] == "qualified_local_model_and_recorded_losses"
    assert qualified["source_guard_present"] == [False, False]
    check = qualified["native_smt_check"]
    assert check["status"] == "checked_local_model" and check["solver_calls"] == 6
    assert [row["solver_answer"] for row in check["results"]].count("sat") == 4
    assert [row["solver_answer"] for row in check["results"]].count("unsat") == 2
    assert all(row["matches_model_expectation"] for row in check["results"])
    assert all(not qualified[key] for key in api.AUTHORITY)
    assert qualified["learned_formula_count"] == 0
    assert not qualified["feature_autoencoder_generated_these_formulas"]


def test_native_lean_emits_objects_and_refuses_false_guard_and_reverse_loss(qualified):
    receipts = qualified["lean_checks"]
    assert len(receipts) == 4 and all(row["matches_expectation"] for row in receipts)
    assert [row["status"] for row in receipts] == ["passed", "rejected", "passed", "rejected"]
    for receipt in receipts:
        assert receipt["backend_executed"] and receipt["returncode"] is not None
        if receipt["expected_success"]:
            assert len(receipt["compiled_artifacts"]) == 1
            artifact = receipt["compiled_artifacts"][0]
            raw = Path(artifact["path"]).read_bytes()
            assert len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
        else:
            assert "Tactic `decide` proved that the proposition" in receipt["stdout"]
            assert "is false" in receipt["stdout"]
    assert qualified["finite_loss_evidence"]["after"] == {"decimal": "0.023", "numerator": 23, "denominator": 1000}
    assert not qualified["finite_loss_evidence"]["training_run_independently_replayed"]


def test_exact_original_byte_spans_native_artifacts_and_all_family_frontiers(qualified):
    native = qualified["native_header_derivation"]
    assert native["source_sha256"] == hashlib.sha256(PROGRAM.encode()).hexdigest()
    assert native["formula_count"] == 12
    for row in native["modeled_symbols"]:
        span = row["source_span"]
        assert hashlib.sha256(PROGRAM.encode()[span["start_byte"]:span["end_byte"]]).hexdigest() == span["sha256"]
    inventory = {row["family_id"]: row for row in qualified["family_inventory"]}
    assert len(inventory) == 40
    for family in ("higher_order", "modal", "temporal", "deontic", "dcec", "tdfol"):
        assert inventory[family]["status"] == "unsupported"
        assert not inventory[family]["learned_family_decoder_trained"]
    assert inventory["first_order"]["status"] == "checked_local_header_string_model"
    assert inventory["transition_system"]["status"] == "native_security_ir_declarations_only"
    records = qualified["metadata_records"]
    assert len(records["native_security_declarations"]) == 1
    assert len(records["native_formalization_claims"]) == 12
    assert len(records["source_bound_smt_obligations"]) == 6
    persisted = json.loads((Path(qualified["output"]) / "qualification.json").read_text())
    assert persisted == qualified


@pytest.mark.skipif(not TOOLS, reason="installed native Lean and Z3 are required")
def test_actual_guarded_source_changes_model_not_claimed_source_authority(tmp_path):
    protocol = contracts.WsgiHeaderProtocolContract(api.REVIEW_PREMISE)
    candidate = contracts.analyze_http_header_contracts(PROGRAM, protocol=protocol).candidate
    assert contracts.verify_header_candidate(PROGRAM, candidate)
    result = api.qualify_terminal_codebase_logic(source_bytes=candidate.source.encode(), source_path="headers.py",
        output=tmp_path / "guarded", training_metrics=METRICS)
    assert result["status"] == "qualified_local_model_and_recorded_losses"
    assert result["source_guard_present"] == [True, True]
    assert all(row["solver_answer"] == "unsat" for row in result["native_smt_check"]["results"])
    assert not result["source_semantics_verified"]


@pytest.mark.parametrize("losses", [
    (0.01, 0.02), (0.01, 0.01), (float("nan"), 0.01), (0.1, float("inf")), (True, 0.01),
])
def test_non_decreasing_or_invalid_measurements_refused_before_output(tmp_path, losses):
    metrics = deepcopy(METRICS)
    metrics["before_reconstruction_loss"], metrics["after_reconstruction_loss"] = losses
    with pytest.raises(ValueError, match="losses|loss decrease"):
        api.qualify_terminal_codebase_logic(source_bytes=PROGRAM.encode(), source_path="headers.py",
            output=tmp_path / "must-not-exist", training_metrics=metrics)
    assert not (tmp_path / "must-not-exist").exists()


def test_existing_namespace_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="fresh canonical"):
        api.qualify_terminal_codebase_logic(source_bytes=PROGRAM.encode(), source_path="headers.py",
            output=output, training_metrics=METRICS)
    assert sentinel.read_text() == "keep" and list(output.iterdir()) == [sentinel]


def test_unsupported_source_and_missing_backend_remain_unavailable(tmp_path):
    result = api.qualify_terminal_codebase_logic(source_bytes=b"def identity(value):\n    return value\n",
        source_path="unsupported.py", output=tmp_path / "unsupported", training_metrics=METRICS,
        lean_executable="missing-terminal-codebase-lean", z3_executable="missing-terminal-codebase-z3")
    assert result["status"] == "unavailable_or_failed"
    assert result["native_header_derivation"]["status"] == "unsupported"
    assert result["native_smt_check"]["solver_calls"] == 0
    assert result["lean_checks"] == []
    assert all(row["status"] == "unsupported" for row in result["family_inventory"])
    assert result["metadata_records"]["native_security_declarations"] == []
