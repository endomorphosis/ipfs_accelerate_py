"""Run shipped worker routing with authored boundary and router endpoints."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_live_census_driver import (
    _worker_parser_and_route,
)
from benchmarks.agent_supervisor.container_coding.test_source_formalization_worker_deployment import (
    _run as worker_run,
)


FLAG = "--semantic-working-set-census-only"


def _arguments():
    return ["--semantic-repository", "/app", "--semantic-working-set", "capsule-references@1",
        "--source-view-profile", "python-comments@1", "--source-view-descriptor", "/owner/source.json",
        "--source-view-descriptor-sha256", "a" * 64]


@pytest.mark.parametrize("census", [False, True])
def test_shipped_worker_forwards_explicit_census_once_through_normal_child_custody(census):
    args = _arguments() + ([FLAG] if census else [])
    code, events = _worker_parser_and_route(args)
    assert code == 0 and events[0] == ("custody",)
    assert len(events) == 2 and events[1][0] == "router"
    argv = events[1][1]
    assert argv.count(FLAG) == census
    assert "--semantic-metadata-census-only" not in argv
    for flag in ("--semantic-repository", "--semantic-working-set", "--source-view-profile",
            "--source-view-descriptor", "--source-view-descriptor-sha256"):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == args[args.index(flag) + 1]
    assert argv[argv.index("--container-boundary-sha256") + 1] == "4" * 64


@pytest.mark.parametrize("change", ["missing_mode", "legacy_mode", "missing_repository",
    "missing_descriptor", "missing_digest", "empty_digest", "disabled_profile", "planning",
    "preflight", "transport_v2", "metadata", "metadata_census", "source_edit", "source_lookup",
    "doctor_candidate", "doctor_contract", "finite_repository", "finite_query", "candidate_handoff"])
def test_shipped_worker_refuses_incompatible_census_before_any_candidate_or_router_import(change):
    args = _arguments() + [FLAG]
    removals = {"missing_mode": "--semantic-working-set", "missing_repository": "--semantic-repository",
        "missing_descriptor": "--source-view-descriptor", "missing_digest": "--source-view-descriptor-sha256"}
    replacements = {"legacy_mode": ("--semantic-working-set", "legacy"),
        "empty_digest": ("--source-view-descriptor-sha256", ""),
        "disabled_profile": ("--source-view-profile", "disabled")}
    additions = {
        "planning": ["--purpose", "planning"], "preflight": ["--preflight"],
        "transport_v2": ["--semantic-transport-schema", "supervisor-semantic-router-input@2"],
        "metadata": ["--semantic-metadata-view", "common-bindings@1"],
        "metadata_census": ["--semantic-metadata-census-only"],
        "source_edit": ["--source-edit-mode", "mapped-span@1"],
        "source_lookup": ["--source-lookup-descriptor", "/owner/lookup.json",
                          "--source-lookup-descriptor-sha256", "b" * 64],
        "doctor_candidate": ["--doctor-candidate-artifact", "/owner/doctor.json"],
        "doctor_contract": ["--doctor-contract-artifact", "/owner/contract.json"],
        "finite_repository": ["--finite-repository-artifact", "/owner/finite.json"],
        "finite_query": ["--finite-proof-query-context", "/owner/query.json"],
        "candidate_handoff": ["--candidate-dispatch-handoff", "/owner/handoff.json"],
    }
    if change in removals:
        index = args.index(removals[change])
        del args[index:index + 2]
    elif change in replacements:
        flag, value = replacements[change]
        args[args.index(flag) + 1] = value
    else:
        args += additions[change]
    code, events = _worker_parser_and_route(args)
    assert isinstance(code, str) and events == []


def test_installed_grok_worker_preserves_eight_turn_control_without_entering_provider(tmp_path):
    # The installed worker's actual parser and forwarding execute. The existing
    # fixture authors UID/boundary admission and replaces only router execution.
    args = _arguments() + [FLAG, "--provider", "grok_cli", "--model", "grok-4.7",
        "--grok-coding-max-turns", "8"]
    result = worker_run(tmp_path, args)
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)["argv"]
    assert argv.count(FLAG) == 1 and "--semantic-metadata-census-only" not in argv
    for flag in ("--provider", "--model", "--grok-coding-max-turns", "--semantic-working-set",
            "--source-view-descriptor", "--source-view-descriptor-sha256"):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == args[args.index(flag) + 1]


def test_census_only_flag_cannot_select_a_plain_worker_route():
    code, events = _worker_parser_and_route([FLAG])
    assert isinstance(code, str) and events == []
