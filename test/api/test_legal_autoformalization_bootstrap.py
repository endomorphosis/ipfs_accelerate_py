"""Behavioral checks for evidence surviving linked-checkout lifecycle changes."""
from __future__ import annotations

import copy
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/legal_autoformalization_bootstrap.py"


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _captured_inventory(tmp_path):
    external = tmp_path / "external"
    checkout_a, checkout_b = tmp_path / "checkout-a", tmp_path / "checkout-b"
    for checkout in (checkout_a, checkout_b):
        (checkout / "scripts").mkdir(parents=True)
        shutil.copyfile(SOURCE, checkout / "scripts" / SOURCE.name)
    first = _module(checkout_a / "scripts" / SOURCE.name, "lac_fixture_first")
    second = _module(checkout_b / "scripts" / SOURCE.name, "lac_fixture_second")
    plan = external / "implementation_plan/legal_autoformalization_convergence_20261013/tasks.json"
    _json(plan, {"tasks": []})
    sources = {"campaign_id": "fixture", "workspace_root": str(external),
               "datasets_root": str(external / "datasets"),
               "accelerator_root": str(checkout_a),
               "original_plan_sha256": first.file_identity(plan)["sha256"]}
    relative_config = "config/legal_autoformalization_bootstrap_sources.json"
    config_a, config_b = checkout_a / relative_config, checkout_b / relative_config
    for config in (config_a, config_b):
        _json(config, sources)
    report = external / "report.json"
    _json(report, {"actual_new_updates": 32, "latent_enabled": False})
    audits = external / "artifacts/legal-autoencoder-training-history-20261013"
    _json(audits / "retained-run-index.json", {"runs": []})
    _json(audits / "historical-and-feature-runs.json", {
        "june_review": {"runs": []}, "september_daemons": [], "feature_workers": []})
    _json(audits / "recent-training-receipts.json", {"runs": [{
        "cell": "temporal-margin", "report": str(report),
        "sha256": first.file_identity(report)["sha256"]}]})
    result = first.common_result("LAC-002", config_a, sources)
    first.collect_runs(result, sources)
    assert first.validate(result, "LAC-002", config_a, sources)["valid"] is True
    return first, second, result, sources, config_b, report


def test_bootstrap_inventory_replays_after_captured_checkout_is_removed(tmp_path):
    first, second, result, sources, config_b, _ = _captured_inventory(tmp_path)
    captured_paths = [row["path"] for row in result["input_artifacts"]
                      if row["role"] == "repository_immutable"]
    assert captured_paths and all(Path(path).is_file() for path in captured_paths)
    shutil.rmtree(first.ROOT)

    checked = second.validate(result, "LAC-002", config_b, sources)

    assert checked["valid"] is True
    assert checked["original_task_acceptance"] == "partial"
    assert all(not Path(path).exists() for path in captured_paths)
    assert captured_paths == [row["path"] for row in result["input_artifacts"]
                              if row["role"] == "repository_immutable"]


def test_bootstrap_repo_owned_inputs_reject_escape_missing_and_tampered_bytes(tmp_path):
    _, second, result, sources, config_b, _ = _captured_inventory(tmp_path)
    altered = copy.deepcopy(result)
    next(row for row in altered["input_artifacts"]
         if row["role"] == "repository_immutable")["repo_relative_path"] = "../../outside"
    with pytest.raises(ValueError):
        second.validate(altered, "LAC-002", config_b, sources)

    original_config = config_b.read_bytes()
    config_b.unlink()
    with pytest.raises(OSError):
        second.validate(result, "LAC-002", config_b, sources)
    config_b.write_bytes(original_config)
    outside = tmp_path / "outside-config.json"
    outside.write_bytes(original_config)
    config_b.unlink()
    config_b.symlink_to(outside)
    with pytest.raises(ValueError):
        second.validate(result, "LAC-002", config_b, sources)
    config_b.unlink()
    config_b.write_bytes(original_config)

    script = second.ROOT / "scripts" / SOURCE.name
    script.write_bytes(script.read_bytes() + b"\n# changed after capture\n")
    with pytest.raises(ValueError, match="Collector code moved"):
        second.validate(result, "LAC-002", config_b, sources)


def test_bootstrap_external_inputs_remain_strict_and_output_cannot_escape(tmp_path):
    _, second, result, sources, config_b, report = _captured_inventory(tmp_path)
    report.write_text('{"actual_new_updates": 999}', encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable input digest mismatch"):
        second.validate(result, "LAC-002", config_b, sources)
    with pytest.raises(ValueError, match="Output must remain inside"):
        second.output_path("LAC-002", str(tmp_path / "escaped-result.json"))


def test_bootstrap_materialize_creates_once_then_validates_without_rewrite(tmp_path, monkeypatch):
    first, _, _, sources, _, _ = _captured_inventory(tmp_path)
    config = first.ROOT / "config/legal_autoformalization_bootstrap_sources.json"
    output = first.output_path("LAC-002", None)
    created = first.materialize_result("LAC-002", config, sources, output)
    assert created["valid"] is True and created["materialized"] is True
    before_bytes, before_stat = output.read_bytes(), output.stat()

    def unexpected_collect(*args):
        raise AssertionError("Existing result must never be recollected")

    monkeypatch.setattr(first, "collect_runs", unexpected_collect)
    checked = first.materialize_result("LAC-002", config, sources, output)
    assert checked["valid"] is True and checked["materialized"] is False
    assert output.read_bytes() == before_bytes
    assert output.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert output.stat().st_ino == before_stat.st_ino


def test_bootstrap_materialize_rejects_bad_existing_output_without_repair(tmp_path):
    first, _, _, sources, _, _ = _captured_inventory(tmp_path)
    config = first.ROOT / "config/legal_autoformalization_bootstrap_sources.json"
    output = first.output_path("LAC-002", None)
    first.materialize_result("LAC-002", config, sources, output)
    bad = json.loads(output.read_text())
    bad["side_effects"]["training_updates"] = 1
    output.write_text(json.dumps(bad), encoding="utf-8")
    before = output.read_bytes()

    with pytest.raises(ValueError, match="Unexpected declared side effects"):
        first.materialize_result("LAC-002", config, sources, output)

    assert output.read_bytes() == before


def test_bootstrap_materialize_rejects_stale_input_without_rewriting_result(tmp_path):
    first, _, _, sources, _, report = _captured_inventory(tmp_path)
    config = first.ROOT / "config/legal_autoformalization_bootstrap_sources.json"
    output = first.output_path("LAC-002", None)
    first.materialize_result("LAC-002", config, sources, output)
    before = output.read_bytes()
    report.write_text('{"actual_new_updates": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="Immutable input digest mismatch"):
        first.materialize_result("LAC-002", config, sources, output)

    assert output.read_bytes() == before


def test_bootstrap_prior_attempt_receipts_are_strict_immutable_inputs(tmp_path):
    first, _, _, sources, _, _ = _captured_inventory(tmp_path)
    config = first.ROOT / "config/legal_autoformalization_bootstrap_sources.json"
    prior = Path(sources["workspace_root"]) / "prior-wrapper-receipt.json"
    _json(prior, {"status": "ended", "native_stop_verified": True})
    sources["prior_supervisor_attempt_receipts"] = [
        {"path": str(prior), "sha256": first.file_identity(prior)["sha256"]}]
    _json(config, sources)
    result = first.common_result("LAC-002", config, sources)
    first.collect_runs(result, sources)
    assert first.validate(result, "LAC-002", config, sources)["valid"] is True
    missing_history = copy.deepcopy(result)
    missing_history["input_artifacts"] = [row for row in result["input_artifacts"]
                                          if row["path"] != str(prior)]
    with pytest.raises(ValueError, match="Prior attempt history pin missing or changed"):
        first.validate(missing_history, "LAC-002", config, sources)
    prior.write_text('{"status": "changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Prior supervisor attempt receipt pin differs"):
        first.common_result("LAC-002", config, sources)


def _training_envelope_fixture(tmp_path):
    first, _, full, sources, _, report = _captured_inventory(tmp_path)
    helper_source = SOURCE.with_name("legal_autoformalization_training_inventory.py")
    helper_copy = first.ROOT / "scripts" / helper_source.name
    shutil.copyfile(helper_source, helper_copy)
    helper = _module(helper_copy, "lac_training_envelope_fixture")
    config = first.ROOT / "config/legal_autoformalization_bootstrap_sources.json"
    return helper, full, sources, config, report


def test_training_inventory_envelope_is_lossless_and_preserves_full_population(tmp_path):
    helper, full, sources, config, _ = _training_envelope_fixture(tmp_path)
    full["additional_retained_evidence"] = {"unicode": "§ 123 — duty", "nested": [1, None, False]}
    envelope = helper.pack_result(full)

    assert helper.unpack_result(envelope) == full
    assert helper.validate_envelope(envelope, config, sources)["valid"] is True
    assert envelope["summary"]["retained_run_count"] == len(full["observations"]["runs"])
    assert envelope["summary"]["input_artifact_count"] == len(full["input_artifacts"])
    assert envelope["summary"]["runs_by_lineage"] == {"recent_source_span": 1}
    assert len(helper.core.canonical(envelope)) < helper.MAX_FILE_BYTES < 1_000_000


def test_training_inventory_envelope_checker_rejects_omitted_run_membership(tmp_path):
    helper, full, sources, config, _ = _training_envelope_fixture(tmp_path)
    full["observations"]["runs"] = []
    full["observations"]["deduplicated_run_count"] = 0
    envelope = helper.pack_result(full)
    # Even a self-consistent new checksum and summary cannot omit source runs.
    with pytest.raises(ValueError, match="Audit run coverage differs"):
        helper.validate_envelope(envelope, config, sources)


def test_training_inventory_envelope_rejects_summary_payload_and_writer_tampering(tmp_path):
    helper, full, _, _, _ = _training_envelope_fixture(tmp_path)
    envelope = helper.pack_result(full)
    tampered = copy.deepcopy(envelope)
    tampered["summary"]["retained_run_count"] += 1
    with pytest.raises(ValueError, match="summary differs"):
        helper.unpack_result(tampered)
    tampered = copy.deepcopy(envelope)
    data = tampered["payload"]["data"]
    tampered["payload"]["data"] = ("A" if data[0] != "A" else "B") + data[1:]
    with pytest.raises(ValueError, match="Compressed training payload identity differs"):
        helper.unpack_result(tampered)
    tampered = copy.deepcopy(envelope)
    tampered["writer"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="writer code moved"):
        helper.unpack_result(tampered)


def test_training_inventory_envelope_rejects_expansion_bombs_and_oversize(tmp_path):
    helper, full, _, _, _ = _training_envelope_fixture(tmp_path)
    envelope = helper.pack_result(full)
    bomb = copy.deepcopy(envelope)
    compressed = gzip.compress(b" " * 100_000, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    bomb["payload"].update(raw_bytes=1, raw_sha256=helper.sha256(b" "),
                           compressed_bytes=len(compressed), compressed_sha256=helper.sha256(compressed),
                           encoded_bytes=len(encoded), data=encoded)
    with pytest.raises(ValueError, match="expansion bound"):
        helper.unpack_result(bomb)
    oversize = copy.deepcopy(envelope)
    oversize["payload"]["raw_bytes"] = helper.MAX_RAW_BYTES + 1
    with pytest.raises(ValueError, match="raw_bytes exceeds cap"):
        helper.unpack_result(oversize)
    oversize = copy.deepcopy(envelope)
    oversize["padding"] = "x" * helper.MAX_FILE_BYTES
    with pytest.raises(ValueError, match="envelope exceeds file cap"):
        helper.unpack_result(oversize)
    large_file = tmp_path / "oversize.json"
    large_file.write_bytes(b" " * (helper.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="file exceeds cap"):
        helper.load_envelope(large_file)


def test_training_inventory_materialize_is_idempotent_and_preserves_bad_output(tmp_path, monkeypatch):
    helper, _, sources, config, _ = _training_envelope_fixture(tmp_path)
    output = helper.core.output_path("LAC-002", None)
    created = helper.materialize(config, sources, output)
    assert created["valid"] is True and created["materialized"] is True
    before, before_stat = output.read_bytes(), output.stat()

    def unexpected_collect(*args):
        raise AssertionError("Existing envelope must not be recollected")

    monkeypatch.setattr(helper.core, "collect_runs", unexpected_collect)
    assert helper.materialize(config, sources, output)["materialized"] is False
    assert output.read_bytes() == before
    assert output.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert output.stat().st_ino == before_stat.st_ino
    bad = json.loads(output.read_text())
    bad["summary"]["retained_run_count"] = 999
    output.write_text(json.dumps(bad), encoding="utf-8")
    bad_bytes = output.read_bytes()
    with pytest.raises(ValueError, match="summary differs"):
        helper.materialize(config, sources, output)
    assert output.read_bytes() == bad_bytes
