"""Guarded worker parser/forwarding contracts with authored boundary facts.

Full parser tests retain the installed nonlegacy refusal. Separate dormant
fragment tests deliberately stop before that guard, then execute the real
final router branch with inert main/custody owners. They establish argument
plumbing only: no process, UID, filesystem authority, native lease, provider,
deployment or guard enablement is qualified here.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_owner_guard import (
    REFUSAL, owner_prefix, worker_parser,
)


POLICY = "--repository-source-policy"
POLICY_ALIASES = tuple(POLICY[:size] for size in range(len("--rep"), len(POLICY) + 1))
DESCRIPTOR = "--repository-source-owner-descriptor"
DIGEST = "--repository-source-owner-descriptor-sha256"
PUBLIC_FLAGS = (DESCRIPTOR, DIGEST)
DESCRIPTOR_PATH = "/opt/ipfs-supervisor/worker-observations/repository-mcp-" + "a" * 32 + ".json"
DESCRIPTOR_SHA256 = "d" * 64
BOUNDARY = Path("/opt/ipfs-supervisor/container-boundary.json")
BOUNDARY_SHA256 = "b" * 64
MODES = ("repository-source-raw@1", "repository-source-compact@1")
PUBLIC_ABBREVIATIONS = tuple(sorted({flag[:size] for flag in PUBLIC_FLAGS
    for size in range(3, len(flag))} - set(PUBLIC_FLAGS) - set(POLICY_ALIASES)))


def _option(flag, value, equals=False):
    return [flag + "=" + value] if equals else [flag, value]


def _arguments(*, mode=MODES[0], equals=False):
    return ["--semantic-repository", "/app", *_option(POLICY, mode, equals),
        *_option(DESCRIPTOR, DESCRIPTOR_PATH, equals),
        *_option(DIGEST, DESCRIPTOR_SHA256, equals)]


def _parser_namespace(arguments):
    """Provide only argparse/std-library values; all runtime imports refuse."""
    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(arguments if args is None else args, namespace)

    def imports(name, *_args, **_kwargs):
        pytest.fail("worker parser fragment imported runtime/native/provider: " + name)

    return {"__builtins__": {**vars(builtins), "__import__": imports},
        "argparse": SimpleNamespace(ArgumentParser=Parser), "pathlib": __import__("pathlib"),
        "sys": SimpleNamespace(argv=["worker-entry", *arguments]),
        "selected_provider": "grok_cli" if "grok_cli" in arguments else "codex_cli",
        "root": BOUNDARY.parent, "artifact": BOUNDARY, "digest": BOUNDARY_SHA256}


def _preguard(arguments):
    """Execute declarations and validation, explicitly without route admission."""
    tree = ast.parse(deployment.WORKER_ENTRY)
    begin = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets))
    guard = ast.dump(ast.parse(deployment._REPOSITORY_SOURCE_ISOLATION_GUARD).body[0],
                     include_attributes=False)
    end = next(index for index in range(begin, len(tree.body))
               if ast.dump(tree.body[index], include_attributes=False) == guard)
    namespace = _parser_namespace(arguments)
    exec(compile(ast.Module(body=tree.body[begin:end], type_ignores=[]),
                 "<dormant-worker-parser-before-retained-guard>", "exec"), namespace)
    return namespace


def _forward(namespace):
    """Execute only the real final router else, under authored call owners."""
    tree = ast.parse(deployment.WORKER_ENTRY)
    branch = next(node for node in tree.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "args" and node.test.attr == "preflight")
    while len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If):
        branch = branch.orelse[0]
    selected = branch.orelse
    assert any(isinstance(node, ast.ImportFrom)
        and node.module == "ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper"
        for node in selected)
    seen, custody = [], []

    def main():
        seen.append(list(namespace["sys"].argv))
        return 0

    def run_with_child_custody(callback):
        assert callback is main
        custody.append(callback)
        return callback()

    modules = {
        "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner": SimpleNamespace(main=main),
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper":
            SimpleNamespace(run_with_child_custody=run_with_child_custody),
    }

    def imports(name, _globals=None, _locals=None, fromlist=(), level=0):
        assert level == 0 and name in modules, "unexpected dormant router import: " + name
        return modules[name]

    namespace["__builtins__"] = {**vars(builtins), "__import__": imports}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<dormant-worker-router-branch>", "exec"), namespace)
    assert stopped.value.code == 0 and len(seen) == len(custody) == 1
    return seen[0]


@pytest.mark.parametrize("alias", POLICY_ALIASES)
@pytest.mark.parametrize("equals", [False, True])
def test_worker_owner_policy_prefixes_preserve_legacy_parser_behavior(alias, equals):
    code, args = worker_parser(_option(alias, "legacy", equals))
    assert code == "legacy_router_boundary" and args.repository_source_policy == "legacy"


@pytest.mark.parametrize("alias", POLICY_ALIASES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("equals", [False, True])
def test_worker_owner_policy_prefixes_keep_nonlegacy_guard(alias, mode, equals):
    arguments = ["--semantic-repository", "/app", *_option(alias, mode, equals)]
    assert owner_prefix(arguments)[0] == REFUSAL
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("alias", POLICY_ALIASES)
@pytest.mark.parametrize("mode", MODES)
def test_worker_owner_earlier_nonlegacy_prefix_cannot_be_hidden_by_legacy(alias, mode):
    arguments = ["--semantic-repository", "/app", *_option(alias, mode, True), POLICY, "legacy"]
    assert owner_prefix(arguments)[0] == REFUSAL
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("equals", [False, True])
def test_worker_owner_valid_public_pair_still_hits_full_route_guard(mode, equals):
    arguments = _arguments(mode=mode, equals=equals)
    assert owner_prefix(arguments)[0] == REFUSAL
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("change", ["missing_path", "missing_digest", "empty_path", "empty_digest",
    "duplicate_path", "duplicate_digest", "duplicate_path_equals", "duplicate_digest_equals"])
def test_worker_owner_partial_empty_and_duplicate_public_pair_refuses_before_guard(change):
    arguments = _arguments()
    flag = DESCRIPTOR if "path" in change else DIGEST
    index = arguments.index(flag)
    if change.startswith("missing"):
        del arguments[index:index + 2]
    elif change.startswith("empty"):
        arguments[index + 1] = ""
    else:
        arguments += _option(flag, arguments[index + 1], change.endswith("equals"))
    with pytest.raises(SystemExit) as stopped:
        _preguard(arguments)
    assert stopped.value.code not in (None, 0, REFUSAL)
    assert worker_parser(arguments)[0] not in ("legacy_router_boundary", REFUSAL)


@pytest.mark.parametrize("path", [
    "relative.json", ".", "/",
    DESCRIPTOR_PATH.replace("/worker-observations/", "/worker-observations//"),
    DESCRIPTOR_PATH.replace("/worker-observations/", "/worker-observations/../worker-observations/"),
    "/" + DESCRIPTOR_PATH, DESCRIPTOR_PATH + "/", DESCRIPTOR_PATH + "\n",
    DESCRIPTOR_PATH + "\0", DESCRIPTOR_PATH + "\x7f", DESCRIPTOR_PATH + "\\name",
    "/" + "a" * 4096, "/bad\ud800name",
])
def test_worker_owner_malformed_descriptor_spelling_refuses_before_guard(path):
    arguments = _arguments()
    arguments[arguments.index(DESCRIPTOR) + 1] = path
    with pytest.raises(SystemExit) as stopped:
        _preguard(arguments)
    assert stopped.value.code not in (None, 0, REFUSAL)


@pytest.mark.parametrize("path", [
    "/tmp/repository-mcp-" + "a" * 32 + ".json",
    DESCRIPTOR_PATH.replace("a" * 32, "A" * 32),
    DESCRIPTOR_PATH.replace("a" * 32, "a" * 31),
    DESCRIPTOR_PATH.replace("repository-mcp-", "repository-other-"),
    DESCRIPTOR_PATH.replace(".json", ".txt"), "/opt/ipfs-supervisor/worker-observations",
])
def test_worker_owner_lexical_path_check_does_not_claim_descriptor_admission(path):
    # Parent, basename, inode, permissions and live digest checks remain the
    # protocol owner's responsibility. Lexically valid strings stay untrusted.
    arguments = _arguments()
    arguments[arguments.index(DESCRIPTOR) + 1] = path
    assert _preguard(arguments)["args"].repository_source_owner_descriptor == path
    assert worker_parser(arguments)[0] == REFUSAL


@pytest.mark.parametrize("digest", ["", "d" * 63, "d" * 65, "D" * 64, "g" * 64,
    "d" * 64 + "\n", "é" * 64])
def test_worker_owner_malformed_digest_refuses_before_guard(digest):
    arguments = _arguments()
    arguments[arguments.index(DIGEST) + 1] = digest
    with pytest.raises(SystemExit) as stopped:
        _preguard(arguments)
    assert stopped.value.code not in (None, 0, REFUSAL)


@pytest.mark.parametrize("prefix", PUBLIC_ABBREVIATIONS)
@pytest.mark.parametrize("equals", [False, True])
def test_worker_owner_public_descriptor_options_require_exact_spelling(prefix, equals):
    arguments = _arguments() + _option(prefix, "authored", equals)
    with pytest.raises(SystemExit) as stopped:
        _preguard(arguments)
    assert stopped.value.code not in (None, 0, REFUSAL)


@pytest.mark.parametrize("change", ["legacy", "missing_repository", "grok", "planning", "preflight",
    "census", "source_advisory", "old_source_view", "doctor", "contract", "finite", "decision"])
def test_worker_owner_public_pair_refuses_incompatible_routes_before_guard(change):
    arguments = _arguments()
    if change == "legacy":
        arguments[arguments.index(POLICY) + 1] = "legacy"
    elif change == "missing_repository":
        del arguments[:2]
    elif change == "grok":
        arguments += ["--provider", "grok_cli"]
    elif change == "planning":
        arguments += ["--purpose", "planning"]
    elif change == "preflight":
        arguments += ["--preflight"]
    elif change == "census":
        arguments += ["--semantic-metadata-census-only", "--semantic-metadata-view", "common-bindings@1"]
    elif change == "source_advisory":
        arguments += ["--source-view-advisory-policy", "source-view-raw@1"]
    elif change == "old_source_view":
        arguments += ["--source-view-profile", "identity@1", "--source-view-descriptor", "/authored.json",
                      "--source-view-descriptor-sha256", "a" * 64]
    elif change == "decision":
        arguments += ["--semantic-decision-policy-artifact", "/authored.json",
                      "--semantic-decision-policy-sha256", "a" * 64]
    else:
        kind = {"doctor": "doctor-candidate", "contract": "doctor-contract", "finite": "finite-repository"}[change]
        cid = "--doctor-task-cid" if change == "doctor" else "--" + kind + "-task-cid"
        arguments += ["--" + kind + "-artifact", "/authored.json", "--" + kind + "-sha256", "a" * 64,
                      cid, "authored-cid"]
    with pytest.raises(SystemExit) as stopped:
        _preguard(arguments)
    assert stopped.value.code not in (None, 0, REFUSAL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("equals", [False, True])
def test_worker_owner_dormant_branch_forwards_exact_public_pair_and_bound_route(mode, equals):
    namespace = _preguard(_arguments(mode=mode, equals=equals))
    assert type(namespace["args"].repository_source_owner_descriptor) is str
    assert namespace["args"].repository_source_owner_descriptor == DESCRIPTOR_PATH
    forwarded = _forward(namespace)
    for flag, value in ((DESCRIPTOR, DESCRIPTOR_PATH), (DIGEST, DESCRIPTOR_SHA256), (POLICY, mode),
                        ("--container-boundary", str(BOUNDARY)), ("--container-boundary-sha256", BOUNDARY_SHA256)):
        assert forwarded.count(flag) == 1 and forwarded[forwarded.index(flag) + 1] == value
    assert not any(value in forwarded for value in ("--parent-lease", "--resource-grant", "--scheduler-state-path"))


def test_worker_owner_legacy_dormant_branch_omits_new_public_pair():
    namespace = _preguard(["--semantic-repository", "/app", POLICY, "legacy"])
    forwarded = _forward(namespace)
    assert DESCRIPTOR not in forwarded and DIGEST not in forwarded and POLICY not in forwarded
    assert forwarded[forwarded.index("--container-boundary") + 1] == str(BOUNDARY)
    assert forwarded[forwarded.index("--container-boundary-sha256") + 1] == BOUNDARY_SHA256


@pytest.mark.parametrize("mode,pair", [("legacy", False), (MODES[0], False),
    (MODES[1], False), (MODES[0], True), (MODES[1], True)])
def test_worker_owner_dormant_mandatory_pair_check_is_independent_of_retained_guard(mode, pair):
    arguments = _arguments(mode=mode) if pair else ["--semantic-repository", "/app", POLICY, mode]
    namespace = _preguard(arguments)
    reason = "repository source policy requires its owner descriptor pair"
    matches = [node for node in ast.parse(deployment.WORKER_ENTRY).body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Constant) and child.value == reason for child in ast.walk(node))]
    assert len(matches) == 1
    code = compile(ast.Module(body=matches, type_ignores=[]), "<dormant-worker-mandatory-pair>", "exec")
    if mode != "legacy" and not pair:
        with pytest.raises(SystemExit) as stopped:
            exec(code, namespace)
        assert stopped.value.code == reason
        assert worker_parser(arguments)[0] == REFUSAL
    else:
        exec(code, namespace)


@pytest.mark.parametrize("mode", MODES)
def test_worker_owner_dormant_argv_reaches_real_router_cli_with_inert_run(monkeypatch, capsys, mode):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    forwarded = _forward(_preguard(_arguments(mode=mode)))
    observations = []

    def run(**kwargs):
        observations.append(kwargs)
        return "authored router result", {"provider_called": False}

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner.sys, "argv", forwarded)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"authored native prompt")))
    assert runner.main() == 0
    assert len(observations) == 1
    value = observations[0]
    assert value["repository_source_owner_descriptor"] == Path(DESCRIPTOR_PATH)
    assert value["repository_source_owner_descriptor_sha256"] == DESCRIPTOR_SHA256
    assert value["repository_source_policy"] == mode
    assert value["container_boundary"] == BOUNDARY and value["container_boundary_sha256"] == BOUNDARY_SHA256
    assert value["prompt"] == "authored native prompt"
    assert capsys.readouterr().out == "authored router result\n"
