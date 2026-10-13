"""Behavioral checks for evidence surviving linked-checkout lifecycle changes."""
from __future__ import annotations

import copy
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
