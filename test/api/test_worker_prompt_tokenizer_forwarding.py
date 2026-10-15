"""Execute generated worker argument dispatch without a container or provider.

The exact production parser/routing AST is executed with an authored boundary
path/digest and a stub router entry point. Unix identity, privilege isolation,
descriptor admission, deployment and model qualification are outside this test.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import pathlib
import sys
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import (
    WORKER_ENTRY,
)

DESCRIPTOR_ARGUMENTS = [
    "--prompt-tokenizer-artifact", "/owner/runtime/tokenizers/pinned descriptor.json",
    "--prompt-tokenizer-sha256", "a" * 64,
]
BUDGET_ARGUMENTS = ["--max-input-tokens", "4096"]
ROUTER_MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner"


@pytest.fixture
def dispatch(monkeypatch):
    parsed = ast.parse(WORKER_ENTRY, filename="<generated-worker-entry>")
    starts = [
        index for index, node in enumerate(parsed.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "parser"
    ]
    assert len(starts) == 1
    # Keep all production statements from parser creation through final route
    # execution. No alternate parser or transformed production AST is used.
    code = compile(
        ast.Module(body=parsed.body[starts[0]:], type_ignores=[]),
        "<generated-worker-entry-parser-and-routing>", "exec",
    )
    calls = []
    imports = []

    def main():
        calls.append(tuple(sys.argv))
        return 0

    def import_router(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append(name)
        assert name == ROUTER_MODULE and tuple(fromlist) == ("main",) and level == 0
        return SimpleNamespace(main=main)

    def run(arguments):
        calls.clear()
        imports.clear()
        monkeypatch.setattr(sys, "argv", ["worker-entry", *arguments])
        namespace = {
            "__builtins__": {**vars(builtins), "__import__": import_router},
            "argparse": argparse,
            "pathlib": pathlib,
            "sys": sys,
            "artifact": pathlib.Path("/fixture/container-boundary.json"),
            "digest": "b" * 64,
        }
        with pytest.raises(SystemExit) as exited:
            exec(code, namespace)
        return exited.value.code, tuple(calls), tuple(imports)

    return run


def test_default_router_argv_preserves_existing_flags_without_token_options(dispatch):
    status, calls, imports = dispatch([])
    assert status == 0 and imports == (ROUTER_MODULE,)
    assert calls == ((
        "worker-entry", "--model", "gpt-5.6-sol", "--reasoning-effort", "high",
        "--purpose", "coding", "--timeout", "90", "--max-output-tokens", "4096",
        "--container-boundary", "/fixture/container-boundary.json",
        "--container-boundary-sha256", "b" * 64,
    ),)


@pytest.mark.parametrize("purpose", ["coding", "planning"])
@pytest.mark.parametrize("budget", [[], BUDGET_ARGUMENTS])
def test_router_forwards_exact_descriptor_pair_and_optional_budget(dispatch, purpose, budget):
    status, calls, imports = dispatch(["--purpose", purpose, *DESCRIPTOR_ARGUMENTS, *budget])
    assert status == 0 and imports == (ROUTER_MODULE,) and len(calls) == 1
    forwarded = calls[0]
    start = forwarded.index("--prompt-tokenizer-artifact")
    assert forwarded[start:] == tuple([*DESCRIPTOR_ARGUMENTS, *budget])
    assert forwarded[forwarded.index("--purpose") + 1] == purpose


def test_budget_without_descriptor_is_forwarded_for_router_to_enforce(dispatch):
    status, calls, imports = dispatch(BUDGET_ARGUMENTS)
    assert status == 0 and imports == (ROUTER_MODULE,)
    assert calls[0][-2:] == tuple(BUDGET_ARGUMENTS)
    assert "--prompt-tokenizer-artifact" not in calls[0]
    assert "--prompt-tokenizer-sha256" not in calls[0]


@pytest.mark.parametrize("arguments", [
    DESCRIPTOR_ARGUMENTS[:2],
    DESCRIPTOR_ARGUMENTS[2:],
    [*DESCRIPTOR_ARGUMENTS[:2], "--prompt-tokenizer-sha256", ""],
    ["--prompt-tokenizer-sha256", ""],
])
def test_orphan_or_empty_descriptor_binding_refuses_before_router_import(dispatch, arguments):
    status, calls, imports = dispatch(arguments)
    assert status == "exact prompt tokenizer descriptor and digest required"
    assert calls == imports == ()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_budget_refuses_before_router_import(dispatch, value):
    status, calls, imports = dispatch(["--max-input-tokens", value])
    assert status != 0
    assert calls == imports == ()


FORBIDDEN_ROUTES = [
    ["--preflight"],
    ["--finite-repository-artifact", "/fixture/finite.json",
     "--finite-repository-sha256", "c" * 64, "--finite-repository-task-cid", "finite-task"],
    ["--finite-repository-artifact", "/fixture/finite.json",
     "--finite-repository-sha256", "c" * 64, "--finite-repository-task-cid", "finite-task",
     "--finite-proof-query-context", "/fixture/query.json",
     "--finite-proof-query-sha256", "d" * 64, "--finite-proof-query-context-cid", "query"],
    ["--doctor-candidate-artifact", "/fixture/doctor.json",
     "--doctor-candidate-sha256", "e" * 64, "--doctor-task-cid", "doctor-task"],
    ["--doctor-contract-artifact", "/fixture/contract.json",
     "--doctor-contract-sha256", "f" * 64, "--doctor-contract-task-cid", "contract-task"],
]


@pytest.mark.parametrize("route", FORBIDDEN_ROUTES)
@pytest.mark.parametrize("options", [DESCRIPTOR_ARGUMENTS, BUDGET_ARGUMENTS])
def test_tokenizer_or_budget_refuses_nonrouter_routes_before_any_import(dispatch, route, options):
    status, calls, imports = dispatch([*route, *options])
    assert status == "prompt tokenizer and token budget require router invocation"
    assert calls == imports == ()


def test_semantic_router_keeps_literal_residual_and_public_instruction_bindings(dispatch):
    route = [
        "--semantic-repository", "/owner/canonical repository",
        "--doctor-residual-artifact", "/fixture/residual.json",
        "--doctor-residual-sha256", "c" * 64,
        "--doctor-residual-task-cid", "residual-task",
        "--public-instruction-artifact", "/fixture/instruction.json",
        "--public-instruction-sha256", "d" * 64,
        "--public-instruction-task-cid", "instruction-task",
    ]
    status, calls, imports = dispatch([*route, *DESCRIPTOR_ARGUMENTS, *BUDGET_ARGUMENTS])
    assert status == 0 and imports == (ROUTER_MODULE,)
    forwarded = calls[0]
    start = forwarded.index("--semantic-repository")
    assert forwarded[start:] == tuple([
        *route[:2], *DESCRIPTOR_ARGUMENTS, *BUDGET_ARGUMENTS, *route[2:],
    ])


SEMANTIC_ARGUMENTS = ["--semantic-repository", "/owner/canonical repository",
                      "--semantic-slot-policy-revision", "closed-typed-identifiers@2"]


@pytest.mark.parametrize("projection", [[], ["--semantic-context-projection"]])
def test_generated_worker_forwards_explicit_compact_comparison_with_exact_token_binding(dispatch, projection):
    options = [*SEMANTIC_ARGUMENTS, *projection, *DESCRIPTOR_ARGUMENTS, *BUDGET_ARGUMENTS]
    status, calls, imports = dispatch(options)
    assert status == 0 and imports == (ROUTER_MODULE,) and len(calls) == 1
    forwarded = calls[0]
    start = forwarded.index("--semantic-repository")
    assert forwarded[start:] == tuple(options)


def test_explicit_legacy_policy_keeps_default_generated_router_argv(dispatch):
    _, baseline, _ = dispatch([])
    status, explicit, imports = dispatch(["--semantic-slot-policy-revision", "legacy@1"])
    assert status == 0 and imports == (ROUTER_MODULE,)
    assert explicit == baseline


@pytest.mark.parametrize("route", FORBIDDEN_ROUTES + [["--purpose", "planning"]])
def test_compact_comparison_refuses_nonmodel_or_planning_routes_before_import(dispatch, route):
    status, calls, imports = dispatch([*route, *SEMANTIC_ARGUMENTS, "--semantic-context-projection"])
    assert status == "semantic context options require bound model coding"
    assert calls == imports == ()


@pytest.mark.parametrize("arguments,reason", [
    (["--semantic-context-projection"], "semantic context projection requires the versioned identifier policy"),
    (SEMANTIC_ARGUMENTS[:2] + ["--semantic-context-projection"],
     "semantic context projection requires the versioned identifier policy"),
    (SEMANTIC_ARGUMENTS[2:], "semantic context options require bound model coding"),
    (SEMANTIC_ARGUMENTS[2:] + ["--semantic-context-projection"],
     "semantic context options require bound model coding"),
])
def test_compact_comparison_requires_versioned_policy_and_explicit_repository(dispatch, arguments, reason):
    status, calls, imports = dispatch(arguments)
    assert status == reason and calls == imports == ()


def test_unknown_semantic_policy_refuses_before_router_import(dispatch):
    status, calls, imports = dispatch(["--semantic-slot-policy-revision", "future@99"])
    assert status == 2 and calls == imports == ()
