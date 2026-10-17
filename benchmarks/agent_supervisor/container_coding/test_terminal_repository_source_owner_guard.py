"""Exercise shipped owner/worker refusal before source work; no live providers.

UID, boundary-file and clock observations are authored. The actual deployed
entrypoint statements and argparse guards run without sudo or native admission.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment


REFUSAL = "repository_source_owner_transport_unavailable"
MODES = ("repository-source-raw@1", "repository-source-compact@1")


class ReachedCandidateBoundary(RuntimeError):
    pass


def owner_prefix(arguments):
    """Run the installed prefix through the first candidate operation."""
    tree = ast.parse(deployment.OWNER_ENTRY)
    stop = next(index for index, node in enumerate(tree.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "cwd"
                        for target in node.targets))
    events = []

    class OwnerPath:
        def __init__(self, value):
            assert value == "/opt/ipfs-supervisor"

        def __truediv__(self, value):
            assert value == "container-boundary.json"
            return self

        def read_text(self):
            events.append("boundary_read")
            return '{"owner_uid":1000}'

        @staticmethod
        def cwd():
            events.append("candidate_access")
            raise ReachedCandidateBoundary

    def observe(name, value):
        events.append(name)
        return value

    owners = {
        "json": json,
        "os": SimpleNamespace(getuid=lambda: observe("uid", 1000),
                              geteuid=lambda: observe("euid", 1000)),
        "pathlib": SimpleNamespace(Path=OwnerPath),
        "sys": SimpleNamespace(argv=["router-worker", *arguments]),
        "time": SimpleNamespace(monotonic=lambda: observe("clock", 12.0)),
    }

    def imports(name, *_args, **_kwargs):
        assert name in owners, "refusal must precede provider/source imports"
        return owners[name]

    namespace = {"__builtins__": {**vars(builtins), "__import__": imports}}
    try:
        exec(compile(ast.Module(body=tree.body[:stop + 1], type_ignores=[]),
                     "<installed-owner-prefix>", "exec"), namespace)
    except SystemExit as error:
        return error.code, events
    except ReachedCandidateBoundary:
        return "legacy_candidate_boundary", events
    pytest.fail("owner prefix did not stop at its candidate boundary")


def worker_parser(arguments):
    """Run the installed parser/guards only, stopping before router import."""
    tree = ast.parse(deployment.WORKER_ENTRY)
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "parser"
                         for target in node.targets))
    stop = next(index for index, node in enumerate(tree.body)
                if isinstance(node, ast.ImportFrom)
                and node.module == "ipfs_accelerate_py.agent_supervisor.runtime"
                and any(alias.name == "router_implementation_runner" for alias in node.names))
    argv = ["worker-entry", *arguments]

    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(argv[1:] if args is None else args, namespace)

    def imports(*_args, **_kwargs):
        pytest.fail("worker refusal must precede any router/native/provider import")

    namespace = {"__builtins__": {**vars(builtins), "__import__": imports},
                 "argparse": SimpleNamespace(ArgumentParser=Parser),
                 "pathlib": __import__("pathlib"),
                 "sys": SimpleNamespace(argv=argv), "selected_provider": "codex_cli",
                 "artifact": Path("/authored/boundary.json"), "digest": "a" * 64}
    try:
        exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]),
                     "<installed-worker-guards>", "exec"), namespace)
    except SystemExit as error:
        return error.code, None
    return "legacy_router_boundary", namespace["args"]


def selected_arguments(mode, spelling):
    flag = {"exact": "--repository-source-policy", "equals": "--repository-source-policy",
            "short": "--repo", "short_equals": "--repo",
            "long_abbreviation": "--repository-s", "long_equals": "--repository-s"}[spelling]
    return [flag + "=" + mode] if "equals" in spelling else [flag, mode]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("spelling", ["exact", "equals", "short", "short_equals", "long_abbreviation", "long_equals"])
def test_owner_and_worker_reject_selected_source_route_before_candidate_or_router(mode, spelling):
    arguments = ["--semantic-repository", "/app", *selected_arguments(mode, spelling)]
    code, events = owner_prefix(arguments)
    assert code == REFUSAL
    assert events == ["clock", "boundary_read", "uid", "euid"]
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("legacy_last", [True, False])
def test_duplicate_legacy_cannot_hide_nonlegacy_selection(mode, legacy_last):
    values = [*selected_arguments(mode, "short_equals"), "--repository-source-policy", "legacy"]
    if not legacy_last:
        values = ["--repository-source-policy=legacy", *selected_arguments(mode, "short")]
    arguments = ["--semantic-repository", "/app", *values]
    assert owner_prefix(arguments)[0] == REFUSAL
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("arguments", [
    ["--repository-source-policy"], ["--repository-source-policy="],
    ["--repository-source-policy", "unknown"], ["--repo=unknown"],
    ["--repository-source-policy", "--model", "fixture"],
])
def test_malformed_selection_is_refused_without_native_imports(arguments):
    code, events = owner_prefix(arguments)
    assert code == REFUSAL and "candidate_access" not in events
    assert worker_parser(arguments)[0] == 2


@pytest.mark.parametrize("arguments", [
    [], ["--repository-source-policy", "legacy"], ["--repository-source-policy=legacy"],
    ["--repo", "legacy"], ["--repo=legacy", "--repository-source-policy", "legacy"],
    ["--reasoning-effort", "high"], ["--reasoning-effort=high"],
    ["--semantic-repository", "/app", "--source-input-policy", "source-input-raw@1"],
    ["--semantic-repository", "/app", "--source-view-advisory-policy", "source-view-raw@1"],
    ["--semantic-repository", "/app", "--tool-output-policy", "artifact-preview@1"],
    ["--semantic-repository", "/app", "--query-reference-policy", "required-instruction-ref@1"],
])
def test_legacy_and_other_long_options_retain_existing_boundary(arguments):
    code, events = owner_prefix(arguments)
    assert code == "legacy_candidate_boundary" and events[-1] == "candidate_access"
    code, parsed = worker_parser(arguments)
    assert code == "legacy_router_boundary" and parsed.repository_source_policy == "legacy"
