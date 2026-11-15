"""Local tokenizer selection travels only through the explicit coding route."""
import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY


ARTIFACT = Path("/owner-provisioned/metadata-tokenizer.json")
PIN = "a" * 64
MODES = ("common-bindings@1", "common-bindings-instruction@1", "capsule-columns@1")


def _base():
    return dict(router=Path("/router"), model="pinned-model", reasoning="high", timeout=30,
                semantic_repository=Path("/app"), doctor=None)


def _selection():
    return dict(semantic_metadata_tokenizer_artifact=ARTIFACT, semantic_metadata_tokenizer_sha256=PIN)


@pytest.mark.parametrize("mode", MODES)
def test_explicit_local_pair_is_forwarded_once_without_asset_reads(mode):
    # This deliberately nonexistent path is provisioned by the real owner in
    # deployment; argv construction must neither open it nor download assets.
    argv = doctor.implementation_argv(**_base(), semantic_metadata_view=mode, **_selection())
    for flag, value in (("--semantic-metadata-tokenizer-artifact", str(ARTIFACT)),
                        ("--semantic-metadata-tokenizer-sha256", PIN)):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value


def test_omitted_selection_preserves_original_argv():
    old = doctor.implementation_argv(**_base())
    assert doctor.implementation_argv(**_base(), semantic_metadata_tokenizer_artifact=None,
                                      semantic_metadata_tokenizer_sha256=None) == old
    assert not any("metadata-tokenizer" in value for value in old)


@pytest.mark.parametrize("route", ["doctor_candidate", "doctor_contract_candidate"])
def test_provider_free_candidate_ignores_valid_model_fallback_selection(route):
    candidate = dict(route=route, status="candidate_ready", provider_calls=0,
                     artifact="/candidate.json", sha256="b" * 64, task_cid="authored-task")
    base = {**_base(), "doctor": candidate}
    original = doctor.implementation_argv(**base, semantic_metadata_view="common-bindings@1")
    selected = doctor.implementation_argv(**base, semantic_metadata_view="common-bindings@1", **_selection())
    assert selected == original and not any("metadata-tokenizer" in value for value in selected)


@pytest.mark.parametrize("change", ["missing_artifact", "missing_pin", "empty_pin", "upper_pin",
    "non_string_pin", "string_artifact", "relative_artifact", "parent_artifact", "legacy",
    "transport", "repository"])
def test_argv_rejects_partial_malformed_or_wrong_route_selection(change):
    options = {**_base(), "semantic_metadata_view": "common-bindings@1", **_selection()}
    if change == "missing_artifact": options["semantic_metadata_tokenizer_artifact"] = None
    elif change == "missing_pin": options["semantic_metadata_tokenizer_sha256"] = None
    elif change == "empty_pin": options["semantic_metadata_tokenizer_sha256"] = ""
    elif change == "upper_pin": options["semantic_metadata_tokenizer_sha256"] = "A" * 64
    elif change == "non_string_pin": options["semantic_metadata_tokenizer_sha256"] = True
    elif change == "string_artifact": options["semantic_metadata_tokenizer_artifact"] = str(ARTIFACT)
    elif change == "relative_artifact": options["semantic_metadata_tokenizer_artifact"] = Path("relative.json")
    elif change == "parent_artifact": options["semantic_metadata_tokenizer_artifact"] = Path("/owner/../foreign.json")
    elif change == "legacy": options["semantic_metadata_view"] = "legacy"
    elif change == "transport": options["semantic_transport_schema"] = "supervisor-semantic-router-input@2"
    else: options["semantic_repository"] = None
    with pytest.raises(ValueError, match="metadata tokenizer"):
        doctor.implementation_argv(**options)


@pytest.mark.parametrize("change", ["missing_artifact", "missing_pin", "legacy", "arm", "census"])
def test_driver_refuses_invalid_selection_before_preparation_or_work(tmp_path, monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid selection reached native work")
    monkeypatch.setattr(driver.preparation, "prepare", forbidden)
    monkeypatch.setattr(driver, "execution_budget", forbidden)
    options = dict(instruction=tmp_path / "instruction", state=tmp_path / "state", arm="full",
                   semantic_metadata_view="common-bindings@1", **_selection())
    if change == "missing_artifact": options["semantic_metadata_tokenizer_artifact"] = None
    elif change == "missing_pin": options["semantic_metadata_tokenizer_sha256"] = None
    elif change == "legacy": options["semantic_metadata_view"] = "legacy"
    elif change == "arm": options["arm"] = "no-index"
    else: options["metadata_census_only"] = True
    with pytest.raises(ValueError):
        driver.run(**options)


def test_driver_keeps_selection_out_of_planning_and_forwards_trusted_audit_pair(tmp_path, monkeypatch):
    state = tmp_path / "state/run"
    state.mkdir(parents=True)
    (state / "admission.json").write_text("{}")
    plans, routed, audits = [], [], []
    class ReachedWorkerRoute(RuntimeError):
        pass
    original_builder = doctor.implementation_argv
    def capture_builder(**kwargs):
        routed.append(original_builder(**kwargs))
        raise ReachedWorkerRoute("authored stop before worker execution")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *_: None)
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver.signal, "signal", lambda *_: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *_: None)
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *_args, **kwargs: audits.append(kwargs))
    monkeypatch.setattr(driver.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    monkeypatch.setattr(driver.preparation, "initial_context", lambda **kwargs: {})
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: plans.append(kwargs) or {"qualified": True})
    monkeypatch.setattr(driver.preparation, "context", lambda **kwargs: {"context_bundle": {}})
    monkeypatch.setattr(driver, "verify_local_benchmark_admission", lambda *_args, **_kwargs:
        {"graph": SimpleNamespace(tasks=[SimpleNamespace(task_cid="authored-task", task_key="authored")])})
    monkeypatch.setattr(doctor, "prepare_terminal_doctor_dispatch", lambda **kwargs:
        {"route": "model_router", "status": "residual"})
    monkeypatch.setattr(doctor, "implementation_argv", capture_builder)
    report = driver.run(instruction=tmp_path / "instruction", state=state, arm="full",
        semantic_metadata_view="common-bindings@1", source384_config=tmp_path / "authored-config", **_selection())
    assert report["error_phase"] == "implementation_setup"
    assert report["error"]["type"] == "ReachedWorkerRoute"
    assert report["provider_invocations"] == [] and len(plans) == 1
    assert not any("tokenizer" in key for key in plans[0])
    assert routed[0][routed[0].index("--semantic-metadata-tokenizer-artifact") + 1] == str(ARTIFACT)
    assert audits[0]["semantic_metadata_tokenizer_artifact"] == ARTIFACT
    assert audits[0]["semantic_metadata_tokenizer_sha256"] == PIN
    observation = report["semantic_metadata_tokenizer"]
    assert observation["artifact_sha256"] == PIN
    assert observation["artifact_path_sha256"] == hashlib.sha256(str(ARTIFACT).encode()).hexdigest()
    assert observation["descriptor_loaded"] is False and observation["provider_usage_verified"] is False
    assert str(ARTIFACT) not in json.dumps(observation)


def _worker_route(argv, provider="codex_cli"):
    """Execute actual generated argparse/forwarding with an authored router."""
    tree = ast.parse(WORKER_ENTRY)
    start = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    fake_sys = SimpleNamespace(argv=["worker-entry", *argv])
    events = []
    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(fake_sys.argv[1:] if args is None else args, namespace)
    def main():
        events.append(list(fake_sys.argv[1:]))
        return 0
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ipfs_accelerate_py.agent_supervisor.runtime":
            return SimpleNamespace(router_implementation_runner=SimpleNamespace(MAX_IMPLEMENTATION_TIMEOUT_SECONDS=600))
        if name == "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner":
            return SimpleNamespace(main=main)
        if name == "ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper":
            return SimpleNamespace(run_with_child_custody=lambda callback: callback())
        pytest.fail("unexpected provider, asset or candidate import")
    namespace = {"__builtins__": {**vars(builtins), "__import__": imports},
        "argparse": SimpleNamespace(ArgumentParser=Parser), "pathlib": __import__("pathlib"),
        "sys": fake_sys, "selected_provider": provider, "artifact": Path("/fixture/boundary"), "digest": "4" * 64}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), "<shipped-worker>", "exec"), namespace)
    return stopped.value.code, events


def _worker_selection():
    return ["--semantic-repository", "/app", "--semantic-metadata-view", "common-bindings@1",
            "--semantic-metadata-tokenizer-artifact", str(ARTIFACT), "--semantic-metadata-tokenizer-sha256", PIN]


@pytest.mark.parametrize("provider", ["codex_cli", "grok_cli"])
def test_generated_worker_forwards_exact_local_pair_to_actual_router_argv(provider):
    argv = _worker_selection() + ["--model", "grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol"]
    code, events = _worker_route(argv, provider=provider)
    assert code == 0 and len(events) == 1
    for flag in ("--semantic-metadata-tokenizer-artifact", "--semantic-metadata-tokenizer-sha256"):
        assert events[0].count(flag) == 1
        assert events[0][events[0].index(flag) + 1] == _worker_selection()[_worker_selection().index(flag) + 1]
    assert "--container-boundary" in events[0]


@pytest.mark.parametrize("change", ["missing_artifact", "missing_pin", "invalid_pin", "relative_artifact",
    "legacy", "repository", "transport", "planning", "preflight", "census", "candidate"])
def test_generated_worker_refuses_pair_gaps_or_non_coding_routes_before_import(change):
    argv = _worker_selection()
    remove = {"missing_artifact": "--semantic-metadata-tokenizer-artifact",
              "missing_pin": "--semantic-metadata-tokenizer-sha256", "repository": "--semantic-repository"}
    if change in remove:
        index = argv.index(remove[change]); del argv[index:index + 2]
    elif change == "invalid_pin": argv[-1] = "A" * 64
    elif change == "relative_artifact": argv[argv.index("--semantic-metadata-tokenizer-artifact") + 1] = "relative.json"
    elif change == "legacy": argv[argv.index("--semantic-metadata-view") + 1] = "legacy"
    elif change == "transport": argv += ["--semantic-transport-schema", "supervisor-semantic-router-input@2"]
    elif change == "planning": argv += ["--purpose", "planning"]
    elif change == "preflight": argv += ["--preflight"]
    elif change == "census": argv += ["--semantic-metadata-census-only"]
    else: argv += ["--doctor-candidate-artifact", "/candidate.json"]
    code, events = _worker_route(argv)
    assert code != 0 and events == []


def test_generated_worker_omits_unselected_tokenizer_controls():
    code, events = _worker_route([])
    assert code == 0 and not any("metadata-tokenizer" in value for value in events[0])


def test_supervisor_cli_retains_exact_explicit_selection(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(driver, "run", lambda **kwargs: captured.append(kwargs) or
        {"task_completed": True, "arm": "full", "seconds": 0, "provider_invocations": []})
    monkeypatch.setattr("sys.argv", ["supervisor", "--instruction", str(tmp_path / "instruction"),
        "--state", str(tmp_path / "state"), "--arm", "full", "--semantic-metadata-view", "common-bindings@1",
        "--semantic-metadata-tokenizer-artifact", str(ARTIFACT), "--semantic-metadata-tokenizer-sha256", PIN])
    assert driver.main() == 0
    assert captured[0]["semantic_metadata_tokenizer_artifact"] == ARTIFACT
    assert captured[0]["semantic_metadata_tokenizer_sha256"] == PIN
