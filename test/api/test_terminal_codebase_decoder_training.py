"""Bounded experiment admission and immutable pre-fit policy controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_decoder_training as api


ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts/codebase_ir_terminal_bench/qualification-20261001-04/supervisor/repository/bottle.py"
BINDING = ROOT / "artifacts/terminal_bench_supervisor/security-code-projections-20260929/published-binding-descriptor.json"
PARENT = ROOT / "artifacts/terminal_bench_supervisor/security-formalization-gaps-20260929/decoder-descriptor-01.json"
ACTUAL = ROOT / "artifacts/codebase_ir_terminal_bench/decoder-qualification-20261001-01/decoder/decoder-report.json"


def _source():
    return PUBLIC.read_bytes()


def _descriptor(path):
    return json.loads(path.read_bytes())


def test_real_parent_history_and_public_corpus_are_admitted_without_fitting():
    parent = _descriptor(PARENT)
    corpus = api.prepare_terminal_codebase_decoder_corpus(source_bytes=_source(), source_path="bottle.py",
                                                          parent_checkpoint=parent)
    assert corpus["training_steps"] == 0
    assert corpus["training_input_contract_satisfied"]
    assert corpus["lineage"]["checked_before_fit"]
    assert corpus["lineage"]["historical_identity_count"] > 0
    assert corpus["lineage"]["same_role_matches"] == []
    assert corpus["lineage"]["parent_head_weights_copied"] is False
    assert corpus["counts"] == {"functions_observed": 358, "v1_supported": 3, "v1_unsupported": 355,
        "v2_supported": 1, "typed_v2_logic_supported": 0, "selected_samples": 3}
    assert all(corpus[key] is False for key in api._AUTHORITY)
    assert hashlib.sha256(_source()).hexdigest() == api.PUBLIC_BOTTLE_SHA256


def test_missing_genuine_initializer_refuses_before_creating_output(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="exact existing published initializer"):
        api.train_terminal_codebase_decoder(source_bytes=_source(), source_path="bottle.py", output=output)
    assert not output.exists()


def test_policy_and_complete_corpus_are_frozen_before_any_native_fit(tmp_path, monkeypatch):
    decoder, _, _ = api._apis()
    output = tmp_path / "frozen-experiment"
    called = []

    def sentinel(**kwargs):
        policy = json.loads((output / "policy.json").read_bytes())
        corpus = json.loads((output / "corpus.json").read_bytes())
        assert policy["frozen_before_fit"]
        assert policy["epochs"] == kwargs["epochs"] == 160
        assert policy["seed"] == kwargs["seed"] == 1729
        assert policy["success_selected_retries"] == 0
        assert policy["checkpoint_selection"] == "final checkpoint after fixed budget; no selection by heldout score"
        assert corpus["counts"]["functions_observed"] == 358
        assert kwargs["samples"] == corpus["samples"]
        assert kwargs["training_provenance_sha256"] == policy["corpus_sha256"] == corpus["corpus_sha256"]
        assert [sample["split"] for sample in kwargs["samples"]] == ["train", "validation", "test"]
        assert kwargs["output"] == output / "checkpoint"
        called.append(True)
        raise RuntimeError("test sentinel: fitting deliberately not performed")

    monkeypatch.setattr(decoder, "train_security_formula_decoder", sentinel)
    with pytest.raises(RuntimeError, match="test sentinel"):
        api.train_terminal_codebase_decoder(source_bytes=_source(), source_path="bottle.py", output=output,
            parent_checkpoint=_descriptor(PARENT), published_binding=_descriptor(BINDING))
    assert called == [True]
    assert {path.name for path in output.iterdir()} == {"policy.json", "corpus.json"}


def test_changed_public_source_refuses_before_creating_policy(tmp_path, monkeypatch):
    decoder, _, _ = api._apis()
    called = []
    monkeypatch.setattr(decoder, "train_security_formula_decoder", lambda **kwargs: called.append(kwargs))
    output = tmp_path / "wrong-public-source"
    with pytest.raises(ValueError, match="complete public source identity differs"):
        api.train_terminal_codebase_decoder(source_bytes=_source() + b"\n", source_path="bottle.py", output=output,
            parent_checkpoint=_descriptor(PARENT), published_binding=_descriptor(BINDING))
    assert called == [] and not output.exists()


def test_initializer_descriptor_mismatch_refuses_before_creating_policy(tmp_path):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.published_legal_initializer import validate_published_legal_initializer
    binding = _descriptor(BINDING)
    transfer = deepcopy(validate_published_legal_initializer(expected_receipt=binding)["initializer"])
    transfer["initializer_sha256"] = "0" * 64
    output = tmp_path / "wrong-initializer"
    with pytest.raises(ValueError, match="selected genuine initializer differs"):
        api.train_terminal_codebase_decoder(source_bytes=_source(), source_path="bottle.py", output=output,
            weight_transfer=transfer, published_binding=binding)
    assert not output.exists()


@pytest.mark.parametrize("path", ["/bottle.py", "../bottle.py", "secret.env", "dir\\bottle.py"])
def test_unsafe_or_non_python_path_refuses(path):
    with pytest.raises(ValueError, match="relative path required"):
        api.prepare_terminal_codebase_decoder_corpus(source_bytes=_source(), source_path=path)


def test_frozen_policy_does_not_alias_mutable_caller_descriptors():
    parent = _descriptor(PARENT)
    corpus = api.prepare_terminal_codebase_decoder_corpus(source_bytes=_source(), source_path="bottle.py",
                                                          parent_checkpoint=parent)
    corpus["role_policy"]["_hkey"] = "test"
    corpus["lineage"]["parent_checkpoint"]["weights_sha256"] = "0" * 64
    assert api.ROLE_POLICY["_hkey"] == "train"
    assert parent["weights_sha256"] == _descriptor(PARENT)["weights_sha256"]


def test_v2_history_without_original_v1_roles_is_explicitly_refused():
    with pytest.raises(ValueError, match="v2 historical ancestor roles unavailable"):
        api.validate_terminal_decoder_split_history(samples=[],
            parent_checkpoint={"schema": "security-formula-production-decoder@2"})


@pytest.fixture
def replay_copy(tmp_path, monkeypatch):
    """Copy authentic frozen bytes; fitting is forbidden throughout these tests."""
    decoder, _, _ = api._apis()

    def forbidden_fit(**kwargs):
        raise AssertionError("a frozen-package replay attempted new optimization")

    monkeypatch.setattr(decoder, "train_security_formula_decoder", forbidden_fit)
    original = _descriptor(ACTUAL)
    output = tmp_path / "isolated-native-replay"
    output.mkdir()
    shutil.copytree(Path(original["checkpoint"]["output"]), output / "checkpoint")
    checkpoint = dict(original["checkpoint"], output=str(output / "checkpoint"))
    report = api._report(output=output, policy=original["policy"], corpus=original["corpus"],
        checkpoint=checkpoint, published_binding=original["published_binding"],
        weight_transfer=original["weight_transfer"])
    api._write(output / "policy.json", report["policy"])
    api._write(output / "corpus.json", report["corpus"])
    api._write(output / "decoder-report.json", report)
    return report


def _persist_report(report):
    value = deepcopy(report)
    value.pop("report_sha256")
    report["report_sha256"] = api._sha(api._wire(value))
    (Path(report["output"]) / "decoder-report.json").write_bytes(api._wire(report))


def test_actual_frozen_package_replays_every_raw_candidate_without_fitting(replay_copy):
    actual = api.validate_terminal_codebase_decoder(expected=replay_copy, source_bytes=_source(), source_path="bottle.py")
    assert actual == replay_copy
    assert actual["training_steps"] == 160
    assert actual["native_kernel_calls"] == 480
    assert actual["learned_candidate_count"] == 3
    assert actual["native_complete_program_ir_count"] == 0
    receipt = actual["loaded_training_receipt"]
    assert receipt["heldout_used_for_fit"] is False
    assert receipt["initial_head_sha256"] != receipt["final_head_sha256"]
    assert len(receipt["training_losses"]) == len(receipt["gradient_norms"]) == 160
    assert all(row["decode"]["validation"]["source_AST_equivalent"] for row in actual["candidates"])
    assert {row["decode"]["status"] for row in actual["candidates"]} == {"candidate"}
    assert actual["source_AST_checks_are_runtime_semantics_proofs"] is False
    assert actual["asymptotic_optimizer_convergence_proved"] is False


def test_real_model_off_and_zero_head_controls_emit_no_source_template(replay_copy):
    controls = replay_copy["controls"]
    model_off = [row for row in controls if row["name"] == "model_off"]
    zero = [row for row in controls if row["name"] == "zero_production_heads"]
    assert len(model_off) == len(zero) == 3
    assert all(row["candidate_emitted"] is False for row in model_off + zero)
    assert all(row["decode"]["predicted_productions"] == [] for row in model_off)
    assert all(row["decode"]["status"] == "rejected" for row in zero)
    assert all(row["decode"]["predicted_productions"] for row in zero)
    assert all(value == 0 for row in zero for node in row["decode"]["predicted_productions"] for value in node["logits"])
    assert {(row["name"], row["status"]) for row in controls if "status" in row} == {
        ("wrong_source", "refused"), ("wrong_checkpoint", "refused")}


@pytest.mark.parametrize("kind", ["raw_logits", "candidate_source", "model_off_candidate", "training_trace"])
def test_resealed_report_cannot_replace_actual_raw_inference_or_receipt(replay_copy, kind):
    report = deepcopy(replay_copy)
    if kind == "raw_logits":
        candidate = report["candidates"][0]
        candidate["decode"]["predicted_productions"][0]["logits"][0] += 1
        candidate["inference_sha256"] = api._sha(api._wire(candidate["decode"]))
    elif kind == "candidate_source":
        report["candidates"][0]["candidate_source"] = "def _hkey(key):\n    return key\n"
    elif kind == "model_off_candidate":
        report["controls"][0]["candidate_emitted"] = True
        report["controls"][0]["decode"]["candidate_source"] = report["candidates"][0]["candidate_source"]
        report["controls"][0]["inference_sha256"] = api._sha(api._wire(report["controls"][0]["decode"]))
    else:
        report["loaded_training_receipt"]["training_losses"][-1] = 0.0
    _persist_report(report)
    with pytest.raises(ValueError, match="raw candidate/control replay differs"):
        api.validate_terminal_codebase_decoder(expected=report, source_bytes=_source(), source_path="bottle.py")


@pytest.mark.parametrize("field,value", [("epochs", 161), ("source_sha256", "0" * 64),
    ("teacher_scope", "heldout targets used for fitting"), ("proof_authority", True)])
def test_resealed_policy_cannot_change_fixed_source_budget_or_teacher_scope(replay_copy, field, value):
    report = deepcopy(replay_copy)
    report["policy"][field] = value
    (Path(report["output"]) / "policy.json").write_bytes(api._wire(report["policy"]))
    _persist_report(report)
    with pytest.raises(ValueError, match="policy"):
        api.validate_terminal_codebase_decoder(expected=report, source_bytes=_source(), source_path="bottle.py")


def test_changed_source_cannot_replay_original_checkpoint_claims(replay_copy):
    with pytest.raises(ValueError, match="complete public source identity differs"):
        api.validate_terminal_codebase_decoder(expected=replay_copy, source_bytes=_source() + b"\n", source_path="bottle.py")


def test_changed_checkpoint_descriptor_is_refused_despite_resealed_report(replay_copy):
    report = deepcopy(replay_copy)
    report["checkpoint"]["weights_sha256"] = "0" * 64
    _persist_report(report)
    with pytest.raises(ValueError, match="package identity"):
        api.validate_terminal_codebase_decoder(expected=report, source_bytes=_source(), source_path="bottle.py")


def test_actual_weight_file_drift_is_refused_without_touching_parent_or_original(replay_copy):
    path = Path(replay_copy["checkpoint"]["output"]) / "weights.json"
    weights = json.loads(path.read_bytes())
    weights["parameters"][0][0][0] += 1
    path.chmod(0o600)
    path.write_bytes(api._wire(weights))
    with pytest.raises(ValueError, match="artifact drift"):
        api.validate_terminal_codebase_decoder(expected=replay_copy, source_bytes=_source(), source_path="bottle.py")
