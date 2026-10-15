"""Installed worker and actual driver command selection without provider calls.

The existing entrypoint fixture authors UID/boundary admission and captures the
router argv. The driver fixture authors preparation and runtime lifecycle while
executing the production driver and implementation-command builder.
"""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from benchmarks.agent_supervisor.container_coding.test_source_formalization_worker_deployment import _run as worker_run

MODE = "capsule-references@1"


def _source_args():
    return ["--semantic-repository", "/app", "--source-view-profile", "python-comments@1",
        "--source-view-descriptor", "/owner/source.json", "--source-view-descriptor-sha256", "a" * 64]


@pytest.mark.parametrize("selection", [None, "legacy", MODE])
def test_installed_worker_forwards_opt_in_once_and_preserves_exact_source_pins(tmp_path, selection):
    args = _source_args()
    if selection is not None:
        args += ["--semantic-working-set", selection]
    result = worker_run(tmp_path, args)
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)["argv"]
    for flag in ("--semantic-repository", "--source-view-profile", "--source-view-descriptor",
                 "--source-view-descriptor-sha256"):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == args[args.index(flag) + 1]
    if selection == MODE:
        assert argv.count("--semantic-working-set") == 1
        assert argv[argv.index("--semantic-working-set") + 1] == MODE
    else:
        assert "--semantic-working-set" not in argv
    assert argv.count("--container-boundary") == argv.count("--container-boundary-sha256") == 1


@pytest.mark.parametrize("change", ["missing_descriptor", "missing_digest", "missing_both_pins",
    "empty_digest", "missing_repository", "disabled_profile", "planning", "preflight", "metadata",
    "transport_v2", "census", "unknown_mode"])
def test_installed_worker_refuses_unbound_or_incompatible_working_set_before_router(tmp_path, change):
    args = _source_args() + ["--semantic-working-set", MODE]
    remove = {"missing_descriptor": "--source-view-descriptor",
              "missing_digest": "--source-view-descriptor-sha256",
              "missing_repository": "--semantic-repository"}
    if change in remove:
        index = args.index(remove[change]); del args[index:index + 2]
    elif change == "missing_both_pins":
        for flag in ("--source-view-descriptor", "--source-view-descriptor-sha256"):
            index = args.index(flag); del args[index:index + 2]
    elif change == "empty_digest":
        args[args.index("--source-view-descriptor-sha256") + 1] = ""
    elif change == "disabled_profile":
        args[args.index("--source-view-profile") + 1] = "disabled"
    elif change == "planning": args += ["--purpose", "planning"]
    elif change == "preflight": args += ["--preflight"]
    elif change == "metadata": args += ["--semantic-metadata-view", "common-bindings@1"]
    elif change == "transport_v2": args += ["--semantic-transport-schema", "supervisor-semantic-router-input@2"]
    elif change == "census": args += ["--semantic-metadata-census-only"]
    else: args[-1] = "capsule-references@2"
    result = worker_run(tmp_path, args)
    assert result.returncode != 0
    assert not result.stdout  # The controlled router prints argv if reached.


@pytest.mark.parametrize("route", ["model_router", "doctor_candidate", "doctor_contract_candidate"])
@pytest.mark.parametrize("selected", [False, True])
def test_production_driver_reaches_command_builder_and_captured_runtime_with_exact_selection(
        tmp_path, monkeypatch, route, selected):
    from test.api.test_terminal_doctor_dispatch import (
        test_driver_selects_before_owner_and_preserves_provider_accounting as exercise_driver,
    )
    real_builder = dispatch.implementation_argv
    calls = []
    def capture(**kwargs):
        argv = real_builder(**kwargs)
        calls.append((kwargs, argv))
        return argv
    monkeypatch.setattr(dispatch, "implementation_argv", capture)
    controls = {"source_view_profile": "python-comments@1"}
    if selected:
        controls["semantic_working_set"] = MODE
    # This existing fixture runs driver.run and inspects the command handed to
    # AdmittedBenchmarkRuntime.create, including provider-free nonapplication.
    exercise_driver(tmp_path, monkeypatch, "full", route, False,
        "supervisor-semantic-router-input@1", controls=controls)
    assert len(calls) == 1
    kwargs, argv = calls[0]
    assert ("semantic_working_set" in kwargs) is selected
    if selected:
        assert kwargs["semantic_working_set"] == MODE
    if selected and route == "model_router":
        assert argv.count("--semantic-working-set") == 1
        assert argv[argv.index("--semantic-working-set") + 1] == MODE
    else:
        assert "--semantic-working-set" not in argv
