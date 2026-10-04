"""Supervisor consumes shared Security candidates without granting authority."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import security_source_program_advisor_384 as subject
from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384 as shared
from ipfs_datasets_py.logic.software_verification.program import ProgramExpression

TEXT = "def assess(capacity: int, threshold: int) -> bool:\n    return capacity < threshold\n"
PIN = "a" * 64


def config(**updates):
    return {"schema": subject.CONFIG_SCHEMA, "checkpoint_path": "/models/security.json", "checkpoint_sha256": PIN,
        "decoder": "structured", "embedding_snapshot_path": "/models/gte-small", "lake": None, **updates}


def rows(text=TEXT):
    return [dict(id="example.py", source_text=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())]


def candidate(operator="<"):
    refs = ("expr:capacity", "expr:threshold")
    return dict(kind="program_expression", document=ProgramExpression("expr:result", "binary", "boolean",
        operand_ids=refs, evaluation_order=refs, operator=operator, source_ref_ids=("source",)).to_dict())


def install_runtime(monkeypatch, *, operator="<", change_hash=False, on_inference=None):
    calls = []

    def infer(source_rows, **options):
        return dict(domain_id="security_ir", rows=[dict(id=row["id"], source_sha256=
            hashlib.sha256(row["source_text"].encode()).hexdigest(), candidate_ir=candidate(operator),
            status="unqualified_candidate") for row in source_rows])

    runtime = shared.SourceProgramDecoder384(SimpleNamespace(infer=infer,
        describe=lambda: dict(domain_id="security_ir")), checkpoint_sha256=PIN)

    def texts(values, **options):
        calls.append((values, options))
        result = runtime.infer([dict(id="input-" + str(index), source_text=text, embedding=[.125] * 384)
                                for index, text in enumerate(values)])
        if change_hash:
            result["rows"][0]["source_sha256"] = "0" * 64
        if on_inference:
            on_inference()
        return result

    def load(path, **options):
        assert path == "/models/security.json"
        assert options == dict(expected_sha256=PIN, decoder="structured")
        return SimpleNamespace(describe=runtime.describe, infer_texts=texts)

    monkeypatch.setattr(shared, "load_source_program_decoder_384", load)
    return calls


def test_disabled_is_inert_and_no_default_model_is_selected(monkeypatch):
    monkeypatch.setattr(shared, "load_source_program_decoder_384", lambda *a, **k: pytest.fail("implicit model loading"))
    result = subject.prepare_security_source_program_advice()
    assert result["status"] == "disabled" and result["checkpoint_selection"] is None
    assert result["continue_planning"] and all(result[key] is False for key in subject.FALSE)


def test_explicit_config_uses_shared_runtime_and_actual_native_source_qualification(monkeypatch):
    calls = install_runtime(monkeypatch)
    selected, source = config(), rows()
    before = deepcopy(source)
    result = subject.prepare_security_source_program_advice(config=selected, source_rows=source)
    assert calls == [([TEXT], {"snapshot_path": "/models/gte-small"})]
    assert result["status"] == "source_candidate_advice" and result["qualified_candidate_count"] == 1
    assert result["checkpoint_selection"] == selected and result["source_hashes"] == {"example.py": source[0]["source_sha256"]}
    prediction = result["inference"]["rows"][0]
    assert prediction["candidate_ir"] == candidate()
    assert prediction["source_contract"]["schema"] == "security-source-program-binding-384/v2"
    assert prediction["source_contract"]["status"] == "qualified"
    assert result["lake"] is None and source == before
    assert result["training_steps"] == result["provider_calls"] == result["download_calls"] == 0
    assert all(result[key] is False for key in subject.FALSE)


@pytest.mark.parametrize("text,operator,status", [(TEXT, "<=", "mismatch"), (TEXT.replace(": int", ""), "<", "unsupported")])
def test_wrong_or_unsupported_candidates_remain_unchanged_and_fail_open(monkeypatch, text, operator, status):
    install_runtime(monkeypatch, operator=operator)
    result = subject.prepare_security_source_program_advice(config=config(), source_rows=rows(text))
    assert result["status"] == "fail_open_no_qualified_candidates"
    predicted = result["inference"]["rows"][0]
    assert predicted["candidate_ir"] == candidate(operator)
    assert predicted["source_contract"]["status"] == status
    assert predicted["source_contract"]["projections"] == [] and result["continue_planning"]


@pytest.mark.parametrize("change", ["extra_target", "sha", "duplicate", "too_many"])
def test_target_leak_or_input_identity_mismatch_never_reaches_decoder(monkeypatch, change):
    monkeypatch.setattr(shared, "load_source_program_decoder_384", lambda *a, **k: pytest.fail("invalid inputs reached weights"))
    inputs = rows()
    if change == "extra_target":
        inputs[0]["target"] = candidate()
    elif change == "sha":
        inputs[0]["source_sha256"] = "0" * 64
    elif change == "duplicate":
        inputs *= 2
    else:
        inputs *= 17
    result = subject.prepare_security_source_program_advice(config=config(), source_rows=inputs)
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "source_inputs"
    assert result["inference"] is None and result["continue_planning"]


@pytest.mark.parametrize("selected", [config(checkpoint_sha256="main"), config(extra_target={}),
    config(lake={"executable": "/tools/lake", "timeout_seconds": 61}), config(checkpoint_path="relative.json"),
    config(input_view="auto"), config(input_view=None), config(input_view=True),
    config(input_view={"mode": "guarded_ast_normalized"})])
def test_bad_optional_configuration_fails_open(selected):
    result = subject.prepare_security_source_program_advice(config=selected, source_rows=rows())
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "configuration"


def test_unavailable_checkpoint_is_reported_without_blocking_planning(monkeypatch):
    def load(*args, **kwargs):
        raise FileNotFoundError("sensitive local path must not reach diagnostics")
    monkeypatch.setattr(shared, "load_source_program_decoder_384", load)
    result = subject.prepare_security_source_program_advice(config=config(), source_rows=rows())
    assert result["failure_stage"] == "checkpoint_loading" and result["error_type"] == "FileNotFoundError"
    assert result["continue_planning"] and result["inference"] is None
    assert "sensitive local path" not in json.dumps(result)


def test_changed_inference_source_hash_is_discarded(monkeypatch):
    install_runtime(monkeypatch, change_hash=True)
    result = subject.prepare_security_source_program_advice(config=config(), source_rows=rows())
    assert result["status"] == "fail_open_unavailable" and result["inference"] is None


def test_explicit_lake_selection_is_bounded_and_receives_unchanged_source(monkeypatch):
    install_runtime(monkeypatch)
    calls = []
    def build(report, source, **options):
        calls.append((report, source, options))
        return SimpleNamespace(to_dict=lambda: dict(status="passed", backend_executed=True, proof_authority=False))
    monkeypatch.setattr(shared, "build_decoded_source_program_lake", build)
    result = subject.prepare_security_source_program_advice(config=config(lake=dict(executable="/tools/lake", timeout_seconds=17)),
        source_rows=rows())
    assert calls[0][1] == [dict(id="input-0", source_text=TEXT)]
    assert calls[0][2] == dict(lake_executable="/tools/lake", timeout_seconds=17)
    assert result["lake"]["status"] == "passed" and result["proof_authority"] is False


@pytest.mark.parametrize("phase", ["loading", "inference"])
@pytest.mark.parametrize("mutation", ["row_fields", "append", "clear"])
def test_owner_callbacks_cannot_replace_captured_source_inputs(monkeypatch, phase, mutation):
    source = rows()
    original = deepcopy(source)
    mutated = []

    def mutate_caller():
        if mutation == "row_fields":
            changed = TEXT.replace("capacity < threshold", "capacity > threshold")
            source[0].update(id="replacement.py", source_text=changed,
                source_sha256=hashlib.sha256(changed.encode()).hexdigest(), target=candidate(">"))
        elif mutation == "append":
            source.append({**rows()[0], "id": "added.py"})
        else:
            source.clear()
        mutated.append(deepcopy(source))

    inference_calls = install_runtime(monkeypatch,
        on_inference=mutate_caller if phase == "inference" else None)
    if phase == "loading":
        original_loader = shared.load_source_program_decoder_384

        def load(*args, **options):
            mutate_caller()
            return original_loader(*args, **options)

        monkeypatch.setattr(shared, "load_source_program_decoder_384", load)
    lake_calls = []

    def build(report, gate_rows, **options):
        lake_calls.append(deepcopy(gate_rows))
        return SimpleNamespace(to_dict=lambda: dict(status="passed", backend_executed=True,
            proof_authority=False))

    monkeypatch.setattr(shared, "build_decoded_source_program_lake", build)
    result = subject.prepare_security_source_program_advice(
        config=config(lake=dict(executable="/tools/lake", timeout_seconds=17)), source_rows=source)

    assert inference_calls == [([TEXT], {"snapshot_path": "/models/gte-small"})]
    assert lake_calls == [[dict(id="input-0", source_text=TEXT)]]
    assert result["status"] == "source_candidate_advice" and result["source_count"] == 1
    assert result["source_hashes"] == {original[0]["id"]: original[0]["source_sha256"]}
    assert result["input_bindings"] == [dict(source_id="example.py", inference_id="input-0",
        source_sha256=original[0]["source_sha256"])]
    assert result["inference"]["rows"][0]["source_sha256"] == original[0]["source_sha256"]
    assert result["lake"]["status"] == "passed" and all(result[key] is False for key in subject.FALSE)
    assert len(mutated) == 1 and source == mutated[0] and source != original


def test_optional_lake_failure_keeps_the_original_candidate(monkeypatch):
    install_runtime(monkeypatch)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("lake unavailable")
    monkeypatch.setattr(shared, "build_decoded_source_program_lake", unavailable)
    result = subject.prepare_security_source_program_advice(config=config(lake=dict(executable="/tools/lake", timeout_seconds=5)),
        source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["lake"]["status"] == "unavailable"
    assert result["inference"]["rows"][0]["candidate_ir"] == candidate()


def test_repository_scope_is_persisted_and_rechecked(monkeypatch, tmp_path):
    install_runtime(monkeypatch)
    (tmp_path / "example.py").write_text(TEXT)
    (tmp_path / "notes.txt").write_text("ordinary task notes")
    output = tmp_path / "advice.json"
    result = subject.prepare_repository_source_program_advice(repository=tmp_path,
        paths=["example.py", "notes.txt"], config=config(), output=output)
    assert result["status"] == "source_candidate_advice"
    assert result["unsupported_paths"] == [dict(id="notes.txt", reason="not_python_source")]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result["artifact_sha256"]
    saved = json.loads(output.read_bytes())
    assert saved["source_hashes"] == {"example.py": hashlib.sha256(TEXT.encode()).hexdigest()}
    assert "raw_source" not in saved


def test_repository_source_drift_discards_stale_advice(monkeypatch, tmp_path):
    path = tmp_path / "example.py"
    path.write_text(TEXT)
    install_runtime(monkeypatch, on_inference=lambda: path.write_text(TEXT + "# changed\n"))
    result = subject.prepare_repository_source_program_advice(repository=tmp_path, paths=["example.py"], config=config())
    assert result["status"] == "fail_open_source_capture" and result["inference"] is None


@pytest.mark.parametrize("path", ["../foreign.py", "/foreign.py", "alias.py"])
def test_repository_escape_or_symlink_is_not_read(monkeypatch, tmp_path, path):
    (tmp_path / "real.py").write_text(TEXT)
    (tmp_path / "alias.py").symlink_to(tmp_path / "real.py")
    monkeypatch.setattr(shared, "load_source_program_decoder_384", lambda *a, **k: pytest.fail("unsafe source loaded"))
    result = subject.prepare_repository_source_program_advice(repository=tmp_path, paths=[path], config=config())
    assert result["status"] == "fail_open_source_capture" and result["inference"] is None


def test_config_file_selection_is_explicit_and_local(monkeypatch, tmp_path):
    install_runtime(monkeypatch)
    selected = tmp_path / "config.json"
    selected.write_text(json.dumps(config()))
    result = subject.prepare_security_source_program_advice(config=selected, source_rows=rows())
    assert result["status"] == "source_candidate_advice" and result["checkpoint_selection"] == config()


@pytest.mark.parametrize("explicit", [False, True])
def test_raw_input_never_dispatches_normalized_runtime(monkeypatch, explicit):
    from ipfs_datasets_py.logic.formalization.autoencoder import normalized_source_program_runtime_384 as normalized
    calls = install_runtime(monkeypatch)
    monkeypatch.setattr(normalized, "load_normalized_source_program_decoder_384",
        lambda *args, **kwargs: pytest.fail("raw selection silently normalized source"))
    selected = config(**({"input_view": "raw"} if explicit else {}))
    result = subject.prepare_security_source_program_advice(config=selected, source_rows=rows())
    assert calls == [([TEXT], {"snapshot_path": "/models/gte-small"})]
    assert result["checkpoint_selection"] == selected
    assert result["input_view"] == "raw" and result["normalization_profile"] is None
    assert result["inference"]["rows"][0]["candidate_ir"] == candidate()


def install_normalized_runtime(monkeypatch, *, text=TEXT, alter_profile=None):
    """Stub numerical inference, retain real shared source qualification checks."""
    from ipfs_datasets_py.logic.formalization.autoencoder import normalized_source_program_runtime_384 as normalized
    calls = install_runtime(monkeypatch)
    base = shared.load_source_program_decoder_384("/models/security.json", expected_sha256=PIN, decoder="structured")
    profile = dict(input_view="guarded_ast_normalized", hybrid_profile=subject.NORMALIZED_INPUT_PROFILE,
        target_dependent_normalization=False, prediction_repair_performed=False)
    receipt = dict(original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        normalized_source_sha256=hashlib.sha256(TEXT.encode()).hexdigest(),
        normalized_source_text=TEXT, status="normalized" if ": int" in text else "unsupported")
    if receipt["status"] == "unsupported":
        receipt.update(normalized_source_text=text, normalized_source_sha256=receipt["original_source_sha256"])
    loader_calls = []

    def describe():
        return {**base.describe(), **profile}

    def infer_texts(texts, **options):
        report = {**base.infer_texts(texts, **options), **profile}
        report["rows"][0]["source_normalization"] = deepcopy(receipt)
        if alter_profile:
            report.update(alter_profile)
        return report

    def load(path, **options):
        loader_calls.append((path, options))
        return SimpleNamespace(describe=describe, infer_texts=infer_texts)

    monkeypatch.setattr(normalized, "load_normalized_source_program_decoder_384", load)
    monkeypatch.setattr(shared, "load_source_program_decoder_384",
        lambda *args, **kwargs: pytest.fail("explicit normalized selection used raw loader"))
    return calls, loader_calls, receipt


def test_explicit_normalized_dispatch_retains_provenance_and_original_lake_source(monkeypatch):
    original = TEXT.replace("    return", "    # Formatting belongs to the original source.\n    return")
    calls, loader_calls, receipt = install_normalized_runtime(monkeypatch, text=original)
    lake_calls = []

    def build(report, source, **options):
        lake_calls.append((report, source, options))
        return SimpleNamespace(to_dict=lambda: dict(status="passed", backend_executed=True, proof_authority=False))

    monkeypatch.setattr(shared, "build_decoded_source_program_lake", build)
    selected = config(input_view="guarded_ast_normalized", lake=dict(executable="/tools/lake", timeout_seconds=11))
    result = subject.prepare_security_source_program_advice(config=selected, source_rows=rows(original))
    assert loader_calls == [("/models/security.json", dict(expected_sha256=PIN, decoder="structured"))]
    assert calls == [([original], {"snapshot_path": "/models/gte-small"})]
    assert result["status"] == "source_candidate_advice"
    assert result["input_view"] == "guarded_ast_normalized"
    assert result["normalization_profile"] == subject.NORMALIZED_INPUT_PROFILE
    assert result["runtime"]["prediction_repair_performed"] is False
    assert result["inference"]["target_dependent_normalization"] is False
    prediction = result["inference"]["rows"][0]
    assert prediction["source_normalization"] == receipt
    assert prediction["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert prediction["candidate_ir"] == candidate()
    assert lake_calls[0][1] == [dict(id="input-0", source_text=original)]
    assert all(result[key] is False for key in subject.FALSE)


@pytest.mark.parametrize("changed", [dict(input_view="raw"), dict(hybrid_profile="unknown"),
    dict(target_dependent_normalization=True), dict(prediction_repair_performed=True)])
def test_normalized_inference_must_report_selected_profile(monkeypatch, changed):
    install_normalized_runtime(monkeypatch, alter_profile=changed)
    result = subject.prepare_security_source_program_advice(config=config(input_view="guarded_ast_normalized"), source_rows=rows())
    assert result["status"] == "fail_open_unavailable"
    assert result["failure_stage"] == "embedding_and_inference"
    assert result["inference"] is None and result["continue_planning"]


def test_normalized_unsupported_source_keeps_original_candidate_and_abstention(monkeypatch):
    original = TEXT.replace(": int", "")
    calls, _, receipt = install_normalized_runtime(monkeypatch, text=original)
    result = subject.prepare_security_source_program_advice(config=config(input_view="guarded_ast_normalized"),
        source_rows=rows(original))
    assert calls[0][0] == [original]
    assert result["status"] == "fail_open_no_qualified_candidates"
    prediction = result["inference"]["rows"][0]
    assert prediction["candidate_ir"] == candidate()
    assert prediction["source_contract"]["status"] == "unsupported"
    assert prediction["source_normalization"] == receipt
    assert receipt["normalized_source_text"] == original
