"""Source-state advice retains predictions, explicit bounds and fail-open scope."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import security_source_program_advisor_384 as parent
from ipfs_accelerate_py.agent_supervisor.runtime import security_source_state_advisor_384 as subject
from test.api.semantic_state.test_security_source_program_advisor_384 import (
    PIN, TEXT, candidate, config, install_normalized_runtime as _install_normalized_runtime,
    install_runtime as _install_runtime, rows,
)

COMPATIBILITY = {"schema": "security-source-384-checkpoint-compatibility/v1",
    "artifact_sha256": PIN, "runtime_view_sha256": "b" * 64, "applied": False}


def _install_compatible_transport(monkeypatch):
    """Mock only checkpoint transport; source qualification remains native."""
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatible
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384 as strict
    from ipfs_datasets_py.logic.formalization.autoencoder import normalized_source_program_runtime_384 as normalized

    def load(path, *, expected_sha256, decoder, input_view):
        assert input_view in {"raw", "guarded_ast_normalized"}
        selected = (normalized.load_normalized_source_program_decoder_384 if input_view == "guarded_ast_normalized"
            else strict.load_source_program_decoder_384)
        runtime = selected(path, expected_sha256=expected_sha256, decoder=decoder)
        return SimpleNamespace(describe=lambda: {**runtime.describe(), "checkpoint_compatibility": deepcopy(COMPATIBILITY)},
            infer_texts=lambda *args, **kwargs: {**runtime.infer_texts(*args, **kwargs), "checkpoint_compatibility": deepcopy(COMPATIBILITY)})

    def verify(receipt, path, expected_sha256):
        assert path == "/models/security.json" and expected_sha256 == PIN
        if receipt != COMPATIBILITY:
            raise ValueError("compatibility receipt differs from checkpoint replay")
        return deepcopy(receipt)

    monkeypatch.setattr(compatible, "load_source_program_decoder_384_v2", load)
    monkeypatch.setattr(compatible, "verify_checkpoint_compatibility", verify)


def install_runtime(monkeypatch, **options):
    result = _install_runtime(monkeypatch, **options)
    _install_compatible_transport(monkeypatch)
    return result


def install_normalized_runtime(monkeypatch, **options):
    result = _install_normalized_runtime(monkeypatch, **options)
    _install_compatible_transport(monkeypatch)
    return result


def domains(**change):
    result = {"example.py": {"capacity": {"lower": -1, "upper": 1}, "threshold": {"lower": 0, "upper": 1}}}
    result.update(change)
    return result


def selected(**change):
    return config(schema=parent.STATE_CONFIG_SCHEMA, finite_state_domains=domains(), **change)


def inputs(prediction=None):
    sources = rows()
    bindings = [dict(source_id="example.py", inference_id="input-0", source_sha256=sources[0]["source_sha256"])]
    inference = dict(domain_id="security_ir", checkpoint_sha256=PIN, rows=[dict(id="input-0",
        source_sha256=sources[0]["source_sha256"], candidate_ir=candidate() if prediction is None else prediction)])
    return dict(inference=inference, source_rows=sources, input_bindings=bindings, finite_state_domains=domains())


class FakeOwner:
    """Boundary transport fake; real derivation is exercised separately below."""
    def __init__(self, change=None):
        self.calls = []
        self.change = change
        self.issued = {}

    def prepare_source_state_lean(self, rows):
        self.calls.append(("prepare", deepcopy(rows)))
        result = dict(schema="source-state-384-lake/v1", status="prepared", source_executed=False,
            automatic_operational_model=True, source_replay_passed=True, input_sha256=subject._sha(subject._wire(rows)),
            candidate_repaired=False, model_inference_performed=False, input_domains_inferred=False,
            proof_authority=False, execution_authority=False, completion_authority=False,
            source_semantics_verified=False, claim_proved=False,
            rows=[dict(id=row["id"], source_sha256=hashlib.sha256(row["source_text"].encode()).hexdigest(),
                candidate_sha256=subject._sha(subject._wire(row["candidate_ir"])),
                input_domains_sha256=subject._sha(subject._wire(row["input_domains"])),
                status="prepared" if row["candidate_ir"] is not None else "unsupported",
                semantic_lowering_supported=row["candidate_ir"] is not None,
                lake_status="not_run", sany_status="not_run", reason=None) for row in rows])
        if self.change:
            self.change(result)
        return result

    def build_source_state_lake(self, rows, **options):
        self.calls.append(("build", deepcopy(rows), options))
        result = self.prepare_source_state_lean(rows)
        handle = SimpleNamespace(to_dict=lambda: deepcopy(result))
        self.issued[id(handle)] = (deepcopy(rows), result)
        return handle

    def verify_source_state_lake(self, handle, rows):
        self.calls.append(("verify", deepcopy(rows)))
        expected, result = self.issued[id(handle)]
        assert expected == rows
        return deepcopy(result)


def owner(monkeypatch, change=None):
    value = FakeOwner(change)
    monkeypatch.setattr(subject, "_gate", lambda: value)
    return value


def test_v1_does_not_import_optional_state_owner(monkeypatch):
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatible
    install_runtime(monkeypatch)
    monkeypatch.setattr(subject, "_gate", lambda: pytest.fail("v1 requested optional state model"))
    monkeypatch.setattr(compatible, "load_source_program_decoder_384_v2",
        lambda *args, **kwargs: pytest.fail("v1 bypassed its strict checkpoint loader"))
    result = parent.prepare_security_source_program_advice(config=config(), source_rows=rows())
    assert result["schema"] == parent.SCHEMA and "source_state" not in result
    assert result["inference"]["rows"][0]["candidate_ir"] == candidate()


@pytest.mark.parametrize("selection", [config(finite_state_domains=domains()),
    config(schema=parent.STATE_CONFIG_SCHEMA), config(schema="unknown/v3")])
def test_state_configuration_requires_the_explicit_version(monkeypatch, selection):
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384
    monkeypatch.setattr(source_program_runtime_384, "load_source_program_decoder_384",
        lambda *args, **kwargs: pytest.fail("ambiguous configuration reached checkpoint weights"))
    result = parent.prepare_security_source_program_advice(config=selection, source_rows=rows())
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "configuration"
    assert result["continue_planning"] and result["inference"] is None


def test_v2_joins_exact_source_candidate_and_explicit_parameter_domains(monkeypatch):
    install_runtime(monkeypatch)
    gate = owner(monkeypatch)
    source = rows()
    before = deepcopy(source)
    result = parent.prepare_security_source_program_advice(config=selected(), source_rows=source)
    assert result["schema"] == parent.STATE_SCHEMA and result["status"] == "source_candidate_advice"
    state = result["source_state"]
    assert state["status"] == "state_candidate_advice" and state["rows"][0]["source_id"] == "example.py"
    assert gate.calls == [("prepare", [dict(id="input-0", source_text=TEXT, candidate_ir=candidate(), input_domains=domains()["example.py"])])]
    assert source == before and result["inference"]["rows"][0]["candidate_ir"] == candidate()
    assert result["checkpoint_compatibility"] == result["runtime"]["checkpoint_compatibility"] == result["inference"]["checkpoint_compatibility"] == COMPATIBILITY
    assert all(state[key] is False for key in subject.FALSE)
    assert state["continue_planning"] and not state["live_build_verified"] and not state["source_executed"]


def test_optional_build_is_live_verified_before_serialization(monkeypatch):
    gate = owner(monkeypatch)
    args = inputs()
    before = deepcopy(args)
    result = subject.consume_source_state_advice(**args, lake={"executable": "/tools/lake", "timeout_seconds": 7})
    assert result["live_build_verified"] and result["status"] == "state_candidate_advice"
    assert [call[0] for call in gate.calls] == ["build", "prepare", "verify"]
    assert gate.calls[0][2] == dict(lake_executable="/tools/lake", timeout_seconds=7)
    assert args == before


@pytest.mark.parametrize("change", ["missing", "foreign", "boolean", "too_many", "reversed", "extra", "wrong_name"])
def test_invalid_explicit_domains_fail_open_without_discarding_inference(monkeypatch, change):
    install_runtime(monkeypatch)
    gate = owner(monkeypatch)
    selection = selected()
    bounds = selection["finite_state_domains"]
    if change == "missing": bounds.clear()
    elif change == "foreign": bounds["another.py"] = bounds.pop("example.py")
    elif change == "boolean": bounds["example.py"]["capacity"]["lower"] = False
    elif change == "too_many": bounds["example.py"]["capacity"]["upper"] = 100
    elif change == "reversed": bounds["example.py"]["capacity"] = dict(lower=2, upper=1)
    elif change == "extra": bounds["example.py"]["capacity"]["assumed_true"] = True
    else: bounds["example.py"] = {"capacity": dict(lower=0, upper=1)}
    result = parent.prepare_security_source_program_advice(config=selection, source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["inference"]["rows"][0]["candidate_ir"] == candidate()
    assert result["source_state"]["status"] == "fail_open_unavailable"
    assert result["source_state"]["failure_stage"] == "source_state_inputs" and gate.calls == []
    assert result["continue_planning"]


@pytest.mark.parametrize("change", ["candidate_hash", "domain_hash", "source_hash", "population", "authority", "inferred_bounds", "repaired", "replay"])
def test_native_identity_or_scope_drift_is_not_forwarded(monkeypatch, change):
    def mutate(native):
        if change.endswith("_hash"):
            key = {"candidate_hash": "candidate_sha256", "domain_hash": "input_domains_sha256", "source_hash": "source_sha256"}[change]
            native["rows"][0][key] = "0" * 64
        elif change == "population": native["rows"] *= 2
        elif change == "authority": native["rows"][0]["proof_authority"] = True
        elif change == "inferred_bounds": native["input_domains_inferred"] = True
        elif change == "repaired": native["candidate_repaired"] = True
        else: native["input_sha256"] = "0" * 64
    owner(monkeypatch, mutate)
    args = inputs()
    before = deepcopy(args)
    result = subject.consume_source_state_advice(**args)
    assert result["status"] == "fail_open_unavailable" and result["native"] is None and result["rows"] == []
    assert args == before and result["continue_planning"]


@pytest.mark.parametrize("change", ["prompt_source", "target", "duplicate_prediction", "forged_binding"])
def test_prompt_or_foreign_source_never_substitutes_code_evidence(monkeypatch, change):
    gate = owner(monkeypatch)
    args = inputs()
    if change == "prompt_source": args["source_rows"][0]["source_text"] = "Please classify the manifest."
    elif change == "target": args["source_rows"][0]["target"] = candidate()
    elif change == "duplicate_prediction": args["inference"]["rows"] *= 2
    else: args["input_bindings"][0]["source_id"] = "foreign.py"
    result = subject.consume_source_state_advice(**args)
    assert result["status"] == "fail_open_unavailable" and gate.calls == []


def test_abstention_retained_without_replacing_candidate(monkeypatch):
    gate = owner(monkeypatch)
    args = inputs()
    args["inference"]["rows"][0]["candidate_ir"] = None
    result = subject.consume_source_state_advice(**args)
    assert result["status"] == "fail_open_no_supported_state_candidates"
    assert result["rows"][0]["status"] == "unsupported" and gate.calls[0][1][0]["candidate_ir"] is None
    assert args["inference"]["rows"][0]["candidate_ir"] is None


def test_unavailable_owner_and_oversized_state_evidence_preserve_parent_inference(monkeypatch):
    install_runtime(monkeypatch)
    def unavailable(): raise ImportError("sensitive environment detail")
    monkeypatch.setattr(subject, "_gate", unavailable)
    result = parent.prepare_security_source_program_advice(config=selected(), source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["inference"] is not None
    assert result["source_state"]["error_type"] == "ImportError" and "sensitive environment detail" not in json.dumps(result)
    owner(monkeypatch, lambda native: native.update(lean_source="x" * 1_048_576))
    result = parent.prepare_security_source_program_advice(config=selected(), source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["inference"] is not None
    assert result["source_state"]["failure_stage"] == "source_state_serialization"


def test_failed_live_verification_drops_only_optional_state_evidence(monkeypatch):
    gate = owner(monkeypatch)
    def reject(*args): raise ValueError("saved dictionary is not a live issued handle")
    gate.verify_source_state_lake = reject
    args = inputs()
    result = subject.consume_source_state_advice(**args, lake={"executable": "/tools/lake", "timeout_seconds": 5})
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "datasets_state_live_verification"
    assert result["native"] is None and not result["live_build_verified"]
    assert args["inference"]["rows"][0]["candidate_ir"] == candidate()


def test_combined_optional_receipts_cannot_evict_original_inference(monkeypatch):
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384
    install_runtime(monkeypatch)
    owner(monkeypatch, lambda native: native.update(lean_source="x" * 900_000))
    monkeypatch.setattr(source_program_runtime_384, "build_decoded_source_program_lake",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: dict(status="passed", lean_source="y" * 1_300_000)))
    result = parent.prepare_security_source_program_advice(
        config=selected(lake=dict(executable="/tools/lake", timeout_seconds=1)), source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["inference"]["rows"][0]["candidate_ir"] == candidate()
    assert result["lake"]["status"] == "passed" and result["source_state"]["status"] == "fail_open_advice_over_budget"
    assert len(parent._wire(result)) <= parent.MAX_ADVICE_BYTES


def test_normalized_embedding_view_still_sends_original_code_to_state_owner(monkeypatch):
    original = TEXT.replace("    return", "    # original source provenance\n    return")
    install_normalized_runtime(monkeypatch, text=original)
    gate = owner(monkeypatch)
    result = parent.prepare_security_source_program_advice(config=selected(input_view="guarded_ast_normalized"), source_rows=rows(original))
    assert result["source_state"]["status"] == "state_candidate_advice"
    assert gate.calls[0][1][0]["source_text"] == original
    assert result["inference"]["rows"][0]["source_normalization"]["normalized_source_text"] == TEXT
    assert result["checkpoint_compatibility"] == result["inference"]["checkpoint_compatibility"] == COMPATIBILITY


@pytest.mark.parametrize("boundary", ["describe", "inference"])
@pytest.mark.parametrize("change", ["missing", "artifact", "extra_authority", "other_pin"])
def test_v2_compatibility_requires_independent_artifact_replay(monkeypatch, boundary, change):
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatible
    install_runtime(monkeypatch)
    owner(monkeypatch)
    load = compatible.load_source_program_decoder_384_v2

    def mutated_load(*args, **kwargs):
        runtime = load(*args, **kwargs)
        def mutate(value):
            if change == "missing": value.pop("checkpoint_compatibility")
            elif change == "artifact": value["checkpoint_compatibility"]["artifact_sha256"] = "0" * 64
            elif change == "extra_authority": value["checkpoint_compatibility"]["proof_authority"] = True
            else: value["checkpoint_compatibility"]["changed_pins"] = [{"path": "security.model", "old": "a", "new": "b"}]
            return value
        return SimpleNamespace(describe=lambda: mutate(runtime.describe()) if boundary == "describe" else runtime.describe(),
            infer_texts=lambda *a, **kw: mutate(runtime.infer_texts(*a, **kw)) if boundary == "inference" else runtime.infer_texts(*a, **kw))

    monkeypatch.setattr(compatible, "load_source_program_decoder_384_v2", mutated_load)
    result = parent.prepare_security_source_program_advice(config=selected(), source_rows=rows())
    assert result["status"] == "fail_open_unavailable" and result["inference"] is None
    assert result["failure_stage"] == ("checkpoint_loading" if boundary == "describe" else "embedding_and_inference")
    assert result["continue_planning"] and result["source_state"] is None


def test_v2_sequence_decoder_retains_strict_existing_loader(monkeypatch):
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384 as strict
    from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatible
    install_runtime(monkeypatch)
    original_loader = strict.load_source_program_decoder_384
    calls = []
    def strict_sequence(path, **options):
        calls.append(options)
        assert options["decoder"] == "sequence_v2"
        return original_loader(path, **{**options, "decoder": "structured"})
    monkeypatch.setattr(strict, "load_source_program_decoder_384", strict_sequence)
    monkeypatch.setattr(compatible, "load_source_program_decoder_384_v2",
        lambda *args, **kwargs: pytest.fail("sequence decoder received a compatibility exception"))
    owner(monkeypatch)
    result = parent.prepare_security_source_program_advice(config=selected(decoder="sequence_v2"), source_rows=rows())
    assert calls == [dict(expected_sha256=PIN, decoder="sequence_v2")]
    assert result["status"] == "source_candidate_advice" and "checkpoint_compatibility" not in result


def test_real_published_checkpoint_uses_narrow_v2_compatibility_with_actual_embeddings():
    checkpoint, checksum, snapshot = (os.environ.get(name) for name in (
        "IR384_TEST_PUBLISHED_SECURITY_CHECKPOINT", "IR384_TEST_PUBLISHED_SECURITY_SHA256", "IR384_TEST_GTE_SMALL_SNAPSHOT"))
    if not all((checkpoint, checksum, snapshot)):
        pytest.skip("explicit immutable published checkpoint and cached GTE assets required")
    original = "def derive(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"
    before = Path(checkpoint).read_bytes()
    assert hashlib.sha256(before).hexdigest() == checksum
    options = selected(checkpoint_path=checkpoint, checkpoint_sha256=checksum, embedding_snapshot_path=snapshot)
    result = parent.prepare_security_source_program_advice(config=options, source_rows=rows(original))
    assert result["status"] == "source_candidate_advice", result
    assert result["checkpoint_compatibility"] == result["runtime"]["checkpoint_compatibility"] == result["inference"]["checkpoint_compatibility"]
    assert result["checkpoint_compatibility"]["artifact_sha256"] == checksum
    assert result["checkpoint_compatibility"]["applied"] is True
    assert result["source_state"]["status"] == "state_candidate_advice", result["source_state"]
    assert result["source_state"]["native"]["rows"][0]["case_count"] == 6
    assert result["inference"]["rows"][0]["source_contract"]["status"] == "qualified"
    assert Path(checkpoint).read_bytes() == before and all(result[key] is False for key in parent.FALSE)


def test_native_derivation_preserves_wrong_prediction_as_unsupported(monkeypatch):
    pytest.importorskip("ipfs_datasets_py.logic.formalization.autoencoder.source_state_lake")
    install_runtime(monkeypatch, operator="<=")
    result = parent.prepare_security_source_program_advice(config=selected(), source_rows=rows())
    assert result["status"] == "fail_open_no_qualified_candidates"
    state = result["source_state"]
    assert state["status"] == "fail_open_no_supported_state_candidates", state
    assert state["rows"][0]["status"] == "unsupported" and state["native"]["rows"][0]["model"] is None
    assert result["inference"]["rows"][0]["candidate_ir"] == candidate("<=")


def test_real_lake_checks_source_state_cases_without_source_or_intent_authority(monkeypatch):
    lake = os.environ.get("IR384_TEST_LAKE_EXECUTABLE")
    if not lake:
        pytest.skip("real Lake executable required")
    pytest.importorskip("ipfs_datasets_py.logic.formalization.autoencoder.source_state_lake")
    result = subject.consume_source_state_advice(**inputs(), lake={"executable": lake, "timeout_seconds": 60})
    assert result["status"] == "state_candidate_advice", result
    assert result["live_build_verified"] and result["native"]["all_candidates_checked"]
    row = result["native"]["rows"][0]
    assert row["case_count"] == 6 and row["finite_correspondence_kernel_checked"]
    assert row["lake_status"] == "passed" and row["sany_status"] == "not_run"
    assert all(result[key] is False for key in subject.FALSE)
    assert result["source_executed"] is False and result["continue_planning"]
