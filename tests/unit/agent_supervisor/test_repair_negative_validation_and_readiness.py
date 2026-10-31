"""Prospective pure protocol/raw-compiler/readiness regressions.

The source compiler and private files below are real. Modeled report/manager
values test only parser/readiness predicates and grant no original owner,
Git/native/cgroup/systemd integration or production acceptance authority.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else owners.canonical(value) + b"\n"
    path.write_bytes(raw)
    return owners.descriptor(path)

def actual_syntax_import(source, monkeypatch):
    """Actual normal import + original raw compile, no AST extraction or exec."""
    pin = worker.descriptor(source)
    guard = worker.PinnedSources({str(source): pin["sha256"]}, [source.parent, source.parent / "D", source.parent / "trusted"])
    observations = {}; active = [True]
    def audit(event, values):
        if active[0] and event == "compile" and values[1] == str(source):
            raw = values[0].encode() if type(values[0]) is str else values[0]
            assert type(raw) is bytes and hashlib.sha256(raw).hexdigest() == pin["sha256"]
            observations[str(source)] = pin["sha256"]
    sys.addaudithook(audit)
    name = source.stem
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    assert name not in sys.modules and worker._ORIGINAL_COMPILE is compile
    try:
        with pytest.raises(SyntaxError):
            importlib.import_module(name)
    finally:
        active[0] = False
        sys.modules.pop(name, None)
    assert guard.executed == {} and guard.denials == []
    assert observations == {str(source): pin["sha256"]}
    return guard, observations

@pytest.mark.parametrize("raw", ["body_would_run = True\ndef broken(:\n", "if True:\npass\n"])
def test_actual_changed_source_compile_records_syntax_without_body_execution(tmp_path, monkeypatch, raw):
    source = tmp_path / "candidate" / "repair_known_negative_candidate_fixture.py"
    write(source, raw.encode())
    guard, observations = actual_syntax_import(source, monkeypatch)
    assert len(guard.syntax_failures) == 1
    row = guard.syntax_failures[0]
    assert row["path"] == row["filename"] == str(source)
    assert row["sha256"] == observations[str(source)] and row["bytes"] == len(raw.encode())
    assert row["exception_category"] == "SyntaxError" and row["body_executed"] is False
    assert row["exception_type"] in {"SyntaxError", "IndentationError"}
    assert row["line"] > 0 and row["message"]

def semantic_scope(tmp_path, monkeypatch, *, syntax=False, nested_mode=None):
    source = tmp_path / "candidate" / "repair_known_negative_candidate_fixture.py"
    write(source, b"def broken(:\n" if syntax else b"value = False\n")
    trusted = tmp_path / ("trusted_" + ("call" if nested_mode == "call_usage" else "collection" if nested_mode == "mixed_usage" else nested_mode) + ".py" if nested_mode else "trusted_test.py")
    test_raw = b"# Private parser fixture; never claimed as an original test run.\n"
    if nested_mode in {"call", "call_usage"}:
        test_raw = b"def test_known_original_case():\n    import importlib\n    importlib.import_module('repair_known_negative_candidate_fixture')\n"
    elif nested_mode in {"collection", "mixed_usage"}:
        test_raw = b"import importlib\nimportlib.import_module('repair_known_negative_candidate_fixture')\ndef test_known_original_case():\n    pass\n"
    elif nested_mode == "caught_unrelated_collection":
        test_raw = b"import importlib\ntry:\n    importlib.import_module('repair_known_negative_candidate_fixture')\nexcept SyntaxError:\n    importlib.import_module('unrelated_missing_dependency_private_fixture')\ndef test_known_original_case():\n    pass\n"
    write(trusted, test_raw)
    extra_trusted = None
    if nested_mode == "mixed_usage":
        extra_trusted = tmp_path / "trusted_call.py"
        write(extra_trusted, b"def test_known_original_case():\n    import importlib\n    importlib.import_module('repair_known_negative_candidate_fixture')\n")
    import _pytest.python as pytest_python
    support = owners.descriptor(Path(pytest_python.__file__).absolute())
    import _pytest.main as pytest_main
    main_support = owners.descriptor(Path(pytest_main.__file__).absolute())
    pins = {str(path): owners.descriptor(path)["sha256"] for path in (source, trusted)}
    if extra_trusted is not None:
        pins[str(extra_trusted)] = owners.descriptor(extra_trusted)["sha256"]
    pins[support["path"]] = support["sha256"]
    pins[main_support["path"]] = main_support["sha256"]
    invocation = tmp_path / "original-validate-invocation"; invocation.mkdir()
    attempt = tmp_path / "attempt"; (attempt / "exports").mkdir(parents=True)
    freeze = write(tmp_path / "freeze.json", {"private_protocol_fixture": True})
    node = str(trusted) + ("::missing_original_case" if nested_mode == "call_usage" else "::test_known_original_case")
    plan = {"freeze": freeze, "source_commitment": "c" * 64, "attempt_directory": str(attempt),
        "source_pins": pins, "selected_checks": [node], "expected_cases": 1, "required_candidate_bodies": [str(source)]}
    if extra_trusted is not None:
        plan["selected_checks"].append(str(extra_trusted) + "::missing_original_case")
        plan["expected_cases"] = len(plan["selected_checks"])
    plan_pin = write(invocation / "validation-plan.json", plan)
    policy = {"python": sys.executable, "validation": {"pytest_module": owners.descriptor(Path(pytest.__file__).absolute())}}
    phases = [{"nodeid": node, "when": phase, "outcome": "failed" if phase == "call" else "passed"}
        for phase in ("setup", "call", "teardown")]
    report = {"schema": owners.WORK_SCHEMA, "passed": False, "semantic_negative": True,
        "negative_profile": "trusted_pytest_call_failure", "source_READ_closed": True, "source_snapshot_stable": True,
        "cleanup_errors": [], "guard_denials": [], "error": None, "junit_error": None, "plan_sha256": plan_pin["sha256"],
        "freeze": freeze, "source_commitment": plan["source_commitment"], "full_source_manifest_sha256": owners.digest(pins),
        "full_source_file_count": len(pins), "pytest_module_origin": policy["validation"]["pytest_module"],
        "startup": {"python": sys.executable, "isolated": 1, "no_bytecode": True},
        "original_compile_primitive": {"module": "builtins", "name": "compile", "builtin": True},
        "pytest_returncode": 1, "collected_cases": [node], "natural_phase_counter_complete": True,
        "phases": phases, "executed_source_buffers": {str(source): pins[str(source)]}, "syntax_failures": [],
        "raw_compiled_sources": {str(trusted): pins[str(trusted)]}, "JUnit": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0},
        "failed_report_attributions": [{"nodeid": node, "report_path": str(trusted), "when": "call", "kind": "trusted_test_assertion",
            "exception_type": "AssertionError", "syntax_witness_indices": [], "candidate_traceback": [],
            "trusted_test_frame": {"path": str(trusted), "sha256": pins[str(trusted)]}, "trusted_wrapper_source": None}]}
    junit = b'<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0"/></testsuites>'
    if syntax:
        guard, compiled = actual_syntax_import(source, monkeypatch) if nested_mode is None else (None, {})
        report.update(negative_profile="changed_candidate_source_syntax_error", pytest_returncode=2,
            collected_cases=[], natural_phase_counter_complete=False, phases=[], executed_source_buffers={support["path"]: support["sha256"]},
            syntax_failures=guard.syntax_failures if guard else [], raw_compiled_sources=compiled,
            JUnit={"tests": 1, "failures": 0, "errors": 1, "skipped": 0},
            failed_report_attributions=[{"nodeid": str(trusted), "report_path": str(trusted), "when": "collect", "kind": "trusted_pytest_collect_wrapper",
                "exception_type": "CollectError", "syntax_witness_indices": [0], "candidate_traceback": [],
                "trusted_test_frame": None, "trusted_wrapper_source": support}])
        junit = b'<testsuites><testsuite tests="1" failures="0" errors="1" skipped="0"/></testsuites>'
    write(attempt / "exports/test-results.xml", junit)
    return policy, plan, report, invocation, source


def test_trusted_failed_call_profile_requires_all_original_semantic_facts(tmp_path, monkeypatch):
    policy, plan, report, invocation, _ = semantic_scope(tmp_path, monkeypatch)
    assert owners.verify_negative_semantic_report(policy, plan, report, invocation) == "trusted_pytest_call_failure"

@pytest.mark.parametrize("failure", ["exit_only", "denial", "read_open", "source_drift", "skipped", "setup_error", "foreign_profile", "unattributed_collection"])
def test_failed_validation_does_not_infer_negative_from_exit_or_incomplete_evidence(tmp_path, monkeypatch, failure):
    policy, plan, report, invocation, source = semantic_scope(tmp_path, monkeypatch)
    if failure == "exit_only":
        report["phases"][1]["outcome"] = "passed"
    elif failure == "denial":
        report["guard_denials"] = ["caught undeclared source"]
    elif failure == "read_open":
        report["source_READ_closed"] = False
    elif failure == "source_drift":
        source.write_text("value = True\n")
    elif failure == "skipped":
        report["phases"][1]["outcome"] = "skipped"
    elif failure == "setup_error":
        report["phases"][0]["outcome"] = "failed"
    elif failure == "foreign_profile":
        report["negative_profile"] = "changed_candidate_source_syntax_error"
    else:
        report["pytest_returncode"] = 2
    with pytest.raises(ValueError):
        owners.verify_negative_semantic_report(policy, plan, report, invocation)

@pytest.mark.parametrize("code", [2, 4])
def test_attributed_syntax_profile_has_no_full_phase_or_executed_body_claim(tmp_path, monkeypatch, code):
    policy, plan, report, invocation, source = semantic_scope(tmp_path, monkeypatch, syntax=True)
    if code == 4:
        model_original_no_collectors(plan, report)
    assert report["phases"] == [] and str(source) not in report["executed_source_buffers"] and report["natural_phase_counter_complete"] is False
    assert owners.verify_negative_semantic_report(policy, plan, report, invocation) == "changed_candidate_source_syntax_error"

def model_original_no_collectors(plan, report):
    """Modeled parser input only; no new original framework execution credit."""
    import _pytest.main as pytest_main
    pin = owners.descriptor(Path(pytest_main.__file__).absolute())
    lines = Path(pin["path"]).read_text().splitlines()
    line = next(index for index, text in enumerate(lines, 1) if text.strip() == "raise UsageError(*errors)")
    report.update(pytest_returncode=4, collection_usage_attribution={
        "schema": "agent-supervisor/original-pytest-no-collectors@1", "exception_type": "UsageError",
        "collector_source": pin, "original_raise_line": line, "no_collector_selectors": plan["selected_checks"][:]})
    report["executed_source_buffers"][pin["path"]] = pin["sha256"]

@pytest.mark.parametrize("failure", ["missing_witness", "wrong_raise", "wrong_selector", "missing_original_main", "unrelated_report", "call_usage", "mixed_usage"])
def test_code4_usage_is_unknown_without_exact_original_syntax_collection_evidence(tmp_path, monkeypatch, failure):
    policy, plan, report, invocation, source = semantic_scope(tmp_path, monkeypatch, syntax=True)
    model_original_no_collectors(plan, report)
    if failure == "missing_witness":
        report["collection_usage_attribution"] = None
    elif failure == "wrong_raise":
        report["collection_usage_attribution"]["original_raise_line"] = 1
    elif failure == "wrong_selector":
        report["collection_usage_attribution"]["no_collector_selectors"] = [str(source) + "::missing"]
    elif failure == "missing_original_main":
        pin = report["collection_usage_attribution"]["collector_source"]
        report["executed_source_buffers"].pop(pin["path"])
    elif failure == "unrelated_report":
        report["failed_report_attributions"][0]["kind"] = "unattributed"
    elif failure == "mixed_usage":
        report["collection_usage_attribution"]["no_collector_selectors"].append(str(source) + "::missing")
    else:
        report["phases"] = [{"nodeid": plan["selected_checks"][0], "when": "call", "outcome": "failed"}]
    with pytest.raises(ValueError):
        owners.verify_negative_semantic_report(policy, plan, report, invocation)

@pytest.mark.parametrize("failure", ["missing_compile", "executed_body", "foreign_filename", "foreign_source_sha"])
def test_syntax_negative_requires_original_changed_source_compile_attribution(tmp_path, monkeypatch, failure):
    policy, plan, report, invocation, source = semantic_scope(tmp_path, monkeypatch, syntax=True)
    if failure == "missing_compile":
        report["raw_compiled_sources"] = {}
    elif failure == "executed_body":
        report["executed_source_buffers"] = {str(source): plan["source_pins"][str(source)]}
    elif failure == "foreign_filename":
        report["syntax_failures"][0]["filename"] = str(tmp_path / "foreign.py")
    else:
        report["syntax_failures"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        owners.verify_negative_semantic_report(policy, plan, report, invocation)

@pytest.fixture
def readiness_scope(tmp_path, monkeypatch):
    # These are private manager responses to a pure readiness parser. The PID
    # birth is this real test process; no external unit is claimed to exist.
    config = write(tmp_path / "fleet.json", {"private_readiness_fixture": True})
    release = tmp_path / "release"; release.mkdir()
    modules = {"fleet_" + suffix: write(release / ("fleet_" + suffix + ".py"), b"# genuine private fixture source\n") for suffix in ("watchdog", "repair")}
    state = tmp_path / "state"; (state / "runtime-loaded").mkdir(parents=True)
    plan = {"state_dir": str(state), "source_commitment": "d" * 64, "release": str(release),
        "release_sources": modules, "contract": {"config": config}}
    monkeypatch.setattr(installer, "verify_repair_upgrade", lambda value, installed=False: None)
    identity = owners.proc_identity(os.getpid())
    invocations = {"watchdog": "a" * 32, "repair": "b" * 32}
    def artifact(suffix):
        return {"owner": identity, "invocation_id": invocations[suffix], "source_commitment": plan["source_commitment"],
            "release": str(release), "loaded_module": modules["fleet_" + suffix], "config": config}
    def path(suffix):
        return state / "runtime-loaded" / (invocations[suffix] + "-" + suffix + ".json")
    def props(suffix, invocation=None):
        return {"returncode": 0, "stdout": "MainPID=" + str(os.getpid()) + "\nInvocationID=" + (invocation or invocations[suffix]) + "\nActiveState=active\nResult=success\n"}
    return plan, artifact, path, props

@pytest.mark.parametrize("publication", ["missing", "partial_json"])
def test_type_simple_source_attestation_wait_is_read_only_and_birth_bound(readiness_scope, publication):
    plan, artifact, path, props = readiness_scope
    calls = []; counts = {"watchdog": 0, "repair": 0}; pending = []
    def runner(spec, *, cwd, timeout):
        assert spec["argv"][:3] == ["systemctl", "--user", "show"] and 0 < timeout <= 10
        suffix = "watchdog" if "watchdog.service" in spec["argv"][3] else "repair"
        calls.append(spec["argv"]); counts[suffix] += 1
        if suffix == "watchdog" and counts[suffix] == 1 and publication == "partial_json":
            stream = path(suffix).open("xb"); os.fchmod(stream.fileno(), 0o400); stream.write(b"{"); stream.flush(); pending.append(stream)
        elif not path(suffix).exists() or pending:
            raw = owners.canonical(artifact(suffix)) + b"\n"
            if pending:
                stream = pending.pop(); stream.write(raw[1:]); stream.flush(); os.fsync(stream.fileno()); stream.close()
            elif suffix != "watchdog" or counts[suffix] > 1:
                write(path(suffix), raw); path(suffix).chmod(0o400)
        return props(suffix)
    try:
        result = installer.observe_repair_upgrade(plan, runner, deadline=time.monotonic() + 2)
    finally:
        for stream in pending:
            stream.close()
    assert result["read_only_readiness_wait_closed"] is True and set(result["loaded_runtime"]) == {"watchdog", "repair"}
    assert counts["watchdog"] >= 2 and all(argv[:3] == ["systemctl", "--user", "show"] for argv in calls)

def test_missing_readiness_expires_without_any_restart_or_deadline_extension(readiness_scope):
    plan, _, _, props = readiness_scope; calls = []
    def runner(spec, *, cwd, timeout):
        calls.append(spec["argv"])
        return props("watchdog")
    started = time.monotonic()
    with pytest.raises(ValueError, match="deadline expired"):
        installer.observe_repair_upgrade(plan, runner, deadline=started + 0.02)
    assert time.monotonic() - started < 1 and calls and all(argv[:3] == ["systemctl", "--user", "show"] for argv in calls)

def test_readiness_refuses_a_new_invocation_without_replaying_restart(readiness_scope):
    plan, _, _, props = readiness_scope; calls = []
    def runner(spec, *, cwd, timeout):
        calls.append(spec["argv"])
        return props("watchdog", "c" * 32 if len(calls) > 1 else None)
    with pytest.raises(ValueError, match="owner/invocation changed"):
        installer.observe_repair_upgrade(plan, runner, deadline=time.monotonic() + 2)
    assert len(calls) == 2 and all(argv[:3] == ["systemctl", "--user", "show"] for argv in calls)

@pytest.mark.parametrize("failure", ["birth", "source", "commitment", "config"])
def test_readiness_refuses_foreign_loaded_source_or_identity(readiness_scope, failure):
    plan, artifact, path, props = readiness_scope; value = artifact("watchdog")
    if failure == "birth":
        value["owner"] = {**value["owner"], "birth": "0"}
    elif failure == "source":
        value["loaded_module"] = {**value["loaded_module"], "sha256": "f" * 64}
    elif failure == "commitment":
        value["source_commitment"] = "f" * 64
    else:
        value["config"] = {**value["config"], "sha256": "f" * 64}
    write(path("watchdog"), value); path("watchdog").chmod(0o400)
    def runner(spec, *, cwd, timeout):
        assert spec["argv"][:3] == ["systemctl", "--user", "show"]
        return props("watchdog")
    with pytest.raises(ValueError, match="source/config/PID"):
        installer.observe_repair_upgrade(plan, runner, deadline=time.monotonic() + 2)

def test_cutover_uses_actual_execution_start_for_unchanged_120_second_budget():
    identity = owners.proc_identity(os.getpid()); started = time.monotonic() - 1; calls = []
    def runner(spec, *, cwd, timeout):
        calls.append(spec["argv"])
        return {"returncode": 0, "stdout": "MainPID=" + str(os.getpid()) + "\nInvocationID=" + "a" * 32 + "\nExecMainStartTimestampMonotonic=" + str(int(started * 1000000)) + "\n"}
    clock = owners.original_cutover_clock("private-fixture-not-an-actual-unit.service", identity, runner)
    assert clock["existing_kernel_runtime_seconds"] == 120
    assert clock["deadline_monotonic"] - clock["started_monotonic"] == 120
    assert 118 < owners.cutover_remaining(clock["deadline_monotonic"]) < 120 and len(calls) == 1

def test_expired_cutover_budget_cannot_be_reset_by_observation():
    with pytest.raises(ValueError, match="deadline expired"):
        owners.cutover_remaining(time.monotonic() - 1)


@pytest.mark.parametrize("mode", ["call", "collection", "caught_unrelated_collection", "call_usage", "mixed_usage"])
def test_actual_pytest_report_attribution_handles_lazy_syntax_and_refuses_unrelated_collection(tmp_path, monkeypatch, mode):
    # A real nested Pytest run exercises report/exception/JUnit attribution.
    # No original public-D/native/cgroup/storage custody is claimed here.
    policy, plan, report, invocation, source = semantic_scope(tmp_path, monkeypatch, syntax=True, nested_mode=mode)
    import _pytest.python as pytest_python
    from _pytest.reports import TestReport, CollectReport
    trusted = Path(plan["selected_checks"][0].partition("::")[0])
    pins = dict(plan["source_pins"])
    guard = worker.PinnedSources(pins, [source.parent, trusted.parent, tmp_path / "trusted"])
    compiled = {}; active = [True]; nodes = []; phases = []; failed = []; loaded = set(); usage = [None]
    import _pytest.main as pytest_main
    from _pytest.config.exceptions import UsageError
    perform_collect = pytest_main.Session.perform_collect
    main_pin = owners.descriptor(Path(pytest_main.__file__).absolute())
    importer_code = pytest_python.importtestmodule.__code__; importer_globals = pytest_python.importtestmodule.__globals__
    support = owners.descriptor(Path(pytest_python.__file__).absolute())
    def audit(event, values):
        if active[0] and event == "compile" and values[1] in pins:
            raw = values[0].encode() if type(values[0]) is str else values[0]
            assert type(raw) is bytes and hashlib.sha256(raw).hexdigest() == pins[values[1]]
            compiled[values[1]] = pins[values[1]]
    sys.addaudithook(audit)
    class Evidence:
        @pytest.hookimpl(wrapper=True)
        def pytest_collection(self, session):
            try:
                return (yield)
            except BaseException as error:
                usage[0] = worker.original_pytest_no_collectors(error, session=session,
                    selected_checks=plan["selected_checks"], failed_reports=failed,
                    perform_collect_code=perform_collect.__code__, perform_collect_globals=perform_collect.__globals__,
                    usage_error_type=UsageError, source_pin=main_pin)
                raise
        def pytest_collection_finish(self, session):
            for item in session.items:
                nodes.append(str(Path(item.path).resolve()) + "::" + item.nodeid.partition("::")[2])
                loaded.add(item.module.__name__)
        def pytest_runtest_logreport(self, report):
            phases.append({"nodeid": str(trusted) + "::" + report.nodeid.partition("::")[2], "when": report.when, "outcome": report.outcome})
        def pytest_exception_interact(self, node, call, report):
            assert type(report) in {TestReport, CollectReport}
            when = "collect" if type(report) is CollectReport else report.when
            path = str(Path(node.path).resolve())
            nodeid = path + "::" + report.nodeid.partition("::")[2] if when != "collect" else report.nodeid
            failed.append(worker.original_pytest_failure_attribution(guard, error=call.excinfo.value,
                report_path=path, nodeid=nodeid, when=when, selected_files={str(trusted)},
                required_bodies={str(source)}, source_pins=pins, collector_error_type=pytest.Collector.CollectError,
                importer_code=importer_code, importer_globals=importer_globals, wrapper_pin=support))
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    try:
        code = pytest.main(["-c", "/dev/null", "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path), "--noconftest", "--capture=sys", "--assert=plain", "--import-mode=importlib",
            "-p", "no:cacheprovider", "--basetemp", str(tmp_path / "nested-pytest"),
            "--log-file=" + str(tmp_path / "nested-pytest.log"),
            "--junitxml", str(Path(plan["attempt_directory"]) / "exports/test-results.xml"), *plan["selected_checks"]], plugins=[Evidence()])
    finally:
        active[0] = False
        sys.modules.pop(source.stem, None)
        selected_files = {node.partition("::")[0] for node in plan["selected_checks"]}
        for name, module in tuple(sys.modules.items()):
            if name in loaded or getattr(module, "__file__", None) in selected_files:
                sys.modules.pop(name, None)
    if mode == "call_usage":
        assert int(code) == 4 and guard.syntax_failures == [] and str(source) not in compiled
        assert failed == [] and usage[0] is None and guard.denials == []
        return
    assert guard.syntax_failures and compiled.get(str(source)) == pins[str(source)]
    assert failed and guard.denials == []
    import xml.etree.ElementTree as ET
    suites = ET.fromstring((Path(plan["attempt_directory"]) / "exports/test-results.xml").read_bytes()).findall(".//testsuite")
    junit = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    report.update(pytest_returncode=int(code), phases=phases, collected_cases=nodes,
        natural_phase_counter_complete=len(phases) == 3, raw_compiled_sources=compiled,
        executed_source_buffers=dict(guard.executed), syntax_failures=guard.syntax_failures,
        failed_report_attributions=failed, collection_usage_attribution=usage[0], JUnit=junit)
    if mode == "call":
        assert int(code) == 1 and junit == {"tests": 1, "failures": 1, "errors": 0, "skipped": 0}
        assert failed[0]["kind"] == "original_source_syntax_error" and failed[0]["syntax_witness_indices"] == [0]
        assert str(source) not in guard.executed
        assert owners.verify_negative_semantic_report(policy, plan, report, invocation) == "changed_candidate_source_syntax_error"
    elif mode == "collection":
        assert int(code) == 4 and junit["errors"] == 1 and phases == []
        assert usage[0] is not None and usage[0]["no_collector_selectors"] == plan["selected_checks"]
        assert failed[0]["kind"] == "trusted_pytest_collect_wrapper" and failed[0]["syntax_witness_indices"] == [0]
        assert failed[0]["trusted_wrapper_source"] == support
        # This nested parser test uses an already-running original Pytest.
        # It proves the live CollectError/direct-cause/frame branch, while
        # truthfully lacking a new raw importer-body execution under guard.
        # The full owner must still refuse until cold standalone integration.
        assert guard.executed.get(support["path"]) != support["sha256"]
        with pytest.raises(ValueError):
            owners.verify_negative_semantic_report(policy, plan, report, invocation)
    elif mode == "mixed_usage":
        assert int(code) == 4 and junit["errors"] == 1 and phases == []
        assert failed[0]["kind"] == "trusted_pytest_collect_wrapper"
        assert usage[0] is None  # The other real selected module has a bad node.
        with pytest.raises(ValueError):
            owners.verify_negative_semantic_report(policy, plan, report, invocation)
    else:
        assert int(code) == 4 and junit["errors"] == 1
        assert usage[0] is None
        assert failed[0]["kind"] == "unattributed" and failed[0]["syntax_witness_indices"] == []
        with pytest.raises(ValueError):
            owners.verify_negative_semantic_report(policy, plan, report, invocation)
