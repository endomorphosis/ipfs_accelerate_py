"""Qualification receipts must preserve scope, input pins and execution identity."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/legal_autoformalization_intake.py"
spec = importlib.util.spec_from_file_location("intake_driver_test", SCRIPT)
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
    prior.write_text(json.dumps({"status": "bounded_proposed_graph_contracts_complete"}))
    cfg = {"schema": "legal-autoformalization-intake-sources/v1", "campaign_id": "fixture",
        "datasets": {"root": str(datasets), "commit": driver.git(datasets, "rev-parse", "HEAD")},
        "prior_graphs_completion": driver.identity(prior),
        "jobs": {"LAC-S4-044": {"test_paths": ["test_owner.py"], "code_paths": ["owner.py"],
                                 "minimum_tests": 1, "timeout_seconds": 30}},
        "immutable_inputs": [{"path": name, "sha256": driver.identity(datasets / name)["sha256"]}
                             for name in ("owner.py", "test_owner.py")]}
    sources = controller / "sources.json"
    sources.write_text(json.dumps(cfg))
    monkeypatch.setattr(driver, "ROOT", controller)
    return datasets, sources, cfg


def test_frozen_qualification_executes_real_test_and_never_rewrites(frozen, monkeypatch):
    _, sources, _ = frozen
    first = driver.materialize("LAC-S4-044", sources)
    assert first["valid"] is True and first["test_count"] == 1
    target = driver.output("LAC-S4-044")
    before = target.read_bytes(), target.stat().st_mtime_ns
    monkeypatch.setattr(driver, "execute", lambda *args: pytest.fail("must not rerun during materialize"))
    assert driver.materialize("LAC-S4-044", sources)["materialized"] is False
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_tracked_source_mutation_refuses_qualification(frozen):
    datasets, sources, _ = frozen
    driver.materialize("LAC-S4-044", sources)
    (datasets / "owner.py").write_text("PROFILE = 'changed'\n")
    with pytest.raises(ValueError, match="tracked changes"):
        driver.materialize("LAC-S4-044", sources)


def test_tampered_receipt_and_scope_are_rejected_without_repair(frozen):
    _, sources, _ = frozen
    driver.materialize("LAC-S4-044", sources)
    target = driver.output("LAC-S4-044")
    original = json.loads(target.read_text())
    for field, value in (("legal_semantic_acceptance", True), ("corpus_coverage_complete", True)):
        altered = copy.deepcopy(original)
        altered[field] = value
        target.write_text(json.dumps(altered))
        before = target.read_bytes()
        with pytest.raises(ValueError, match="acceptance scope"):
            driver.materialize("LAC-S4-044", sources)
        assert target.read_bytes() == before
    altered = copy.deepcopy(original)
    altered["test_execution"]["passed_count"] = 17
    with pytest.raises(ValueError, match="denominator"):
        driver.validate(altered, "LAC-S4-044", sources)


def test_source_path_escape_and_missing_population_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S4-044"]["code_paths"] = ["../outside.py"]
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="relative source path"):
        driver.materialize("LAC-S4-044", sources)
    cfg["jobs"]["LAC-S4-044"]["code_paths"] = ["owner.py"]
    cfg["jobs"]["LAC-S4-044"]["minimum_tests"] = 2
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="population shrank"):
        driver.materialize("LAC-S4-044", sources)
    assert not driver.output("LAC-S4-044").exists()


def test_wrong_interpreter_and_output_alias_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S4-044"]["python_executable_sha256"] = "0" * 64
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="interpreter moved"):
        driver.materialize("LAC-S4-044", sources)
    outside = driver.ROOT.parent / "outside-results"
    outside.mkdir()
    (driver.ROOT / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="output path alias"):
        driver.output("LAC-S4-044")


def test_failing_real_test_cannot_produce_a_passing_receipt(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text("def test_failure():\n    assert False, 'named qualification cause'\n")
    with pytest.raises(driver.QualificationFailure, match="test_failure") as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S4-044"])
    diagnostics = caught.value.diagnostics
    assert diagnostics["failed_cases"][0]["name"] == "test_failure"
    assert "named qualification cause" in diagnostics["failed_cases"][0]["message"]
    assert diagnostics["failed_cases"][0]["status"] == "failure"
    assert diagnostics["returncode"] != 0
    payload = driver.refusal_payload(caught.value)
    assert payload["valid"] is False
    assert "test_failure" in payload["error"]
    assert json.loads(json.dumps(payload))["diagnostics"] == diagnostics


def test_qualification_binds_imports_and_clears_ambient_pytest_flags(frozen, monkeypatch):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import owner\ndef test_exact_pinned_import():\n"
        "    assert owner.PROFILE == 'fictional-source-contract'\n")
    monkeypatch.setenv("PYTHONPATH", str(datasets.parent / "unrelated-import-root"))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unrecognized-ambient-pytest-flag")
    observed = driver.execute(datasets, cfg["jobs"]["LAC-S4-044"])
    assert observed["returncode"] == 0
    assert observed["environment_policy"]["PYTHONPATH"] == str(datasets.resolve())
    assert observed["environment_policy"]["PYTEST_ADDOPTS"] == ""
    assert observed["environment_policy"]["IPFS_AUTO_INSTALL"] == "false"
    assert observed["environment_policy"]["IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS"] == "0"


def test_resealed_receipt_cannot_hide_different_execution_environment(frozen):
    _, sources, _ = frozen
    driver.materialize("LAC-S4-044", sources)
    result = driver.read_json(driver.output("LAC-S4-044"))
    result["test_execution"]["environment_policy"]["PYTHONPATH"] = "/unrelated/source"
    result["test_execution_sha256"] = driver.sha(driver.wire(result["test_execution"]))
    with pytest.raises(ValueError, match="environment policy differs"):
        driver.validate(result, "LAC-S4-044", sources)


def test_actual_failure_case_population_and_logs_are_bounded(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import pytest\n@pytest.mark.parametrize('index', range(7))\n"
        "def test_bounded_named_cause(index):\n"
        "    print('diagnostic-stream-' * 400)\n"
        "    assert False, 'retained cause ' + 'é' * 800\n", encoding="utf-8")
    with pytest.raises(driver.QualificationFailure) as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S4-044"])
    diagnostic = caught.value.diagnostics
    assert diagnostic["failed_case_count"] == diagnostic["test_count"] == 7
    assert len(diagnostic["failed_cases"]) == driver.MAX_FAILURE_CASES == 5
    for case in diagnostic["failed_cases"]:
        for key in ("class", "name", "message"):
            assert len(case[key].encode("utf-8")) <= driver.MAX_FAILURE_FIELD_BYTES
    for key in ("stdout_tail", "stderr_tail"):
        assert len(diagnostic[key].encode("utf-8")) <= driver.MAX_FAILURE_TAIL_BYTES


def test_skipped_test_is_refused_with_its_named_cause(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import pytest\ndef test_missing_required_fixture():\n"
        "    pytest.skip('required fixture unavailable')\n")
    with pytest.raises(driver.QualificationFailure) as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S4-044"])
    case = caught.value.diagnostics["failed_cases"][0]
    assert case["name"] == "test_missing_required_fixture"
    assert case["status"] == "skipped"
    assert "required fixture unavailable" in case["message"]


def test_current_branch_and_prerequisite_cannot_be_relabelled_complete(frozen):
    datasets, sources, cfg = frozen
    cfg["datasets"]["expected_branch"] = "wrong-source-branch"
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="branch differs"):
        driver.bindings("LAC-S4-044", sources)
    cfg["datasets"]["expected_branch"] = driver.git(datasets, "branch", "--show-current")
    prior = Path(cfg["prior_graphs_completion"]["path"])
    prior.write_text(json.dumps({"status": "incomplete"}))
    cfg["prior_graphs_completion"] = driver.identity(prior)
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="prerequisite is not complete"):
        driver.bindings("LAC-S4-044", sources)


def test_changed_training_audit_or_original_plan_blocks_qualification(frozen, tmp_path):
    _, sources, cfg = frozen
    audit = tmp_path / "audit.json"
    audit.write_text('{"scope":"authored diagnostics"}')
    plan = tmp_path / "plan.json"
    plan.write_text('{"original_acceptance":"partial"}')
    cfg["external_evidence"] = [driver.identity(audit)]
    cfg["original_plan"] = driver.identity(plan)
    sources.write_text(json.dumps(cfg))
    driver.materialize("LAC-S4-044", sources)
    audit.write_text('{"scope":"changed"}')
    with pytest.raises(ValueError, match="External evidence moved"):
        driver.validate(driver.read_json(driver.output("LAC-S4-044")), "LAC-S4-044", sources)
    cfg["external_evidence"] = [driver.identity(audit)]
    sources.write_text(json.dumps(cfg))
    plan.write_text('{"original_acceptance":"changed"}')
    with pytest.raises(ValueError, match="Original plan moved"):
        driver.bindings("LAC-S4-044", sources)


def test_resealed_real_input_report_cannot_replace_independent_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    expected = {"rows": 3, "coverage_scope": "fictional-test-population"}
    baseline.write_text(json.dumps(expected))
    job = {"real_input": {"spec": {"kind": "entity_index_census"},
        "baseline": {**driver.identity(baseline), "canonical_json_sha256": driver.sha(driver.wire(expected))}}}
    observed = {"kind": "entity_index_census", "report": expected,
                "report_sha256": driver.sha(driver.wire(expected))}
    driver.validate_real_input(observed, job)
    altered = copy.deepcopy(observed)
    altered["report"]["rows"] = 2
    altered["report_sha256"] = driver.sha(driver.wire(altered["report"]))
    with pytest.raises(ValueError, match="independently pinned baseline"):
        driver.validate_real_input(altered, job)
    with pytest.raises(ValueError, match="Unexpected real-input"):
        driver.validate_real_input(observed, {})
    baseline.write_text(json.dumps(altered["report"]))
    with pytest.raises(ValueError, match="baseline moved"):
        driver.validate_real_input(altered, job)


def test_real_input_executes_in_pinned_root_and_refuses_changed_outcome(frozen):
    datasets, _, cfg = frozen
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    owner = package / "legal_entity_index_intake.py"
    owner.write_text("def census_parquet(path, **kwargs):\n    return {'fixture_rows': 3}\n")
    raw = datasets / "fictional-index.bin"
    raw.write_bytes(b"fictional fixture, not parquet")
    job = dict(cfg["jobs"]["LAC-S4-044"])
    job["real_input"] = {"spec": {"kind": "entity_index_census", "path": str(raw),
        "file_sha256": driver.identity(raw)["sha256"], "options": {}}}
    expected = driver.run_real_input(datasets, job)
    assert expected == {"fixture_rows": 3}
    baseline = datasets / "baseline.json"
    baseline.write_text(json.dumps(expected))
    job["real_input"]["baseline"] = {**driver.identity(baseline),
        "canonical_json_sha256": driver.sha(driver.wire(expected))}
    assert driver.execute(datasets, job)["real_input_execution"]["report"] == expected
    owner.write_text("def census_parquet(path, **kwargs):\n    return {'fixture_rows': 2}\n")
    with pytest.raises(ValueError, match="independently pinned baseline"):
        driver.execute(datasets, job)
