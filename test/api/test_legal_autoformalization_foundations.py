"""Qualification receipts must preserve scope, input pins and execution identity."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/legal_autoformalization_foundations.py"
spec = importlib.util.spec_from_file_location("foundation_driver_test", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    controller = tmp_path / "controller"
    datasets = tmp_path / "datasets"
    controller.mkdir()
    datasets.mkdir()
    (datasets / "owner.py").write_text("PROFILE = 'fictional-source-contract'\n")
    (datasets / "test_owner.py").write_text("def test_fictional_fixture():\n    assert 'é'.encode() == b'\\xc3\\xa9'\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "Freeze fixture"]):
        subprocess.run(["git", *args], cwd=datasets, check=True, capture_output=True)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"status": "inventory_bootstrap_complete"}))
    cfg = {"schema": "legal-autoformalization-foundations-sources/v1", "campaign_id": "fixture",
        "datasets": {"root": str(datasets), "commit": driver.git(datasets, "rev-parse", "HEAD")},
        "prior_bootstrap_completion": driver.identity(prior),
        "jobs": {"LAC-S2-005": {"test_paths": ["test_owner.py"], "code_paths": ["owner.py"],
                                 "minimum_tests": 1, "timeout_seconds": 30}},
        "immutable_inputs": [{"path": name, "sha256": driver.identity(datasets / name)["sha256"]}
                             for name in ("owner.py", "test_owner.py")]}
    sources = controller / "sources.json"
    sources.write_text(json.dumps(cfg))
    monkeypatch.setattr(driver, "ROOT", controller)
    return datasets, sources, cfg


def test_frozen_qualification_executes_real_test_and_never_rewrites(frozen, monkeypatch):
    _, sources, _ = frozen
    first = driver.materialize("LAC-S2-005", sources)
    assert first["valid"] is True and first["test_count"] == 1
    target = driver.output("LAC-S2-005")
    before = target.read_bytes(), target.stat().st_mtime_ns
    monkeypatch.setattr(driver, "execute", lambda *args: pytest.fail("must not rerun during materialize"))
    assert driver.materialize("LAC-S2-005", sources)["materialized"] is False
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_tracked_source_mutation_refuses_qualification(frozen):
    datasets, sources, _ = frozen
    driver.materialize("LAC-S2-005", sources)
    (datasets / "owner.py").write_text("PROFILE = 'changed'\n")
    with pytest.raises(ValueError, match="tracked changes"):
        driver.materialize("LAC-S2-005", sources)


def test_tampered_receipt_and_scope_are_rejected_without_repair(frozen):
    _, sources, _ = frozen
    driver.materialize("LAC-S2-005", sources)
    target = driver.output("LAC-S2-005")
    original = json.loads(target.read_text())
    for field, value in (("legal_semantic_acceptance", True), ("corpus_coverage_complete", True)):
        altered = copy.deepcopy(original)
        altered[field] = value
        target.write_text(json.dumps(altered))
        before = target.read_bytes()
        with pytest.raises(ValueError, match="acceptance scope"):
            driver.materialize("LAC-S2-005", sources)
        assert target.read_bytes() == before
    altered = copy.deepcopy(original)
    altered["test_execution"]["passed_count"] = 17
    with pytest.raises(ValueError, match="denominator"):
        driver.validate(altered, "LAC-S2-005", sources)


def test_source_path_escape_and_missing_population_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S2-005"]["code_paths"] = ["../outside.py"]
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="relative source path"):
        driver.materialize("LAC-S2-005", sources)
    cfg["jobs"]["LAC-S2-005"]["code_paths"] = ["owner.py"]
    cfg["jobs"]["LAC-S2-005"]["minimum_tests"] = 2
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="population shrank"):
        driver.materialize("LAC-S2-005", sources)
    assert not driver.output("LAC-S2-005").exists()


def test_wrong_interpreter_and_output_alias_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S2-005"]["python_executable_sha256"] = "0" * 64
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="interpreter moved"):
        driver.materialize("LAC-S2-005", sources)
    outside = driver.ROOT.parent / "outside-results"
    outside.mkdir()
    (driver.ROOT / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="output path alias"):
        driver.output("LAC-S2-005")


def test_failing_real_test_cannot_produce_a_passing_receipt(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text("def test_failure():\n    assert False\n")
    with pytest.raises(ValueError, match="tests failed"):
        driver.execute(datasets, cfg["jobs"]["LAC-S2-005"])
