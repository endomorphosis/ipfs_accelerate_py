"""ASE-001 executable inventory of prompt-entrypoint friction.

These probes intentionally describe the pre-facade tree. A delivery task that
closes one of the gaps must update the corresponding probe and baseline
document rather than leaving a stale architectural claim behind.
"""

from __future__ import annotations

import argparse
import io
import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    _IDENTITY_ARGUMENTS,
    COMMAND_OPERATIONS,
    USAGE_CLI_COMMANDS,
    AgentCLIError,
    build_agent_request,
    register_agent_cli,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    READ_OPERATIONS,
    ControlContractError,
    Operation,
    OperationRequest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    default_state_root as bundle_state_root,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    default_state_root as objective_state_root,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    _merge_prompt_parameters,
    build_prompt_workflow_arg_parser,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_FLAGS = (
    "--repository-root",
    "--state-root",
    "--repository-id",
    "--tree-id",
    "--objective-id",
    "--objective-revision",
    "--policy-id",
    "--policy-revision",
    "--caller",
)


def _agent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_agent_cli(subparsers)
    return parser


def _bindings(repository_root: Path, state_root: Path) -> dict[str, str]:
    return {
        "repository_root": str(repository_root.resolve()),
        "state_root": str(state_root.resolve()),
        "repository_id": "repository:baseline",
        "tree_id": "tree:baseline",
        "objective_id": "objective:baseline",
        "objective_revision": "revision:baseline",
        "policy_id": "policy:baseline",
        "policy_revision": "revision:baseline",
        "caller": "caller:baseline",
    }


def _binding_argv(repository_root: Path, state_root: Path) -> list[str]:
    values = _bindings(repository_root, state_root)
    result: list[str] = []
    for name in _IDENTITY_ARGUMENTS:
        result.extend(("--" + name.replace("_", "-"), values[name]))
    return result


def _preview_bindings() -> dict[str, str]:
    return {
        "preview_ref": "preview:baseline",
        "preview_root": "root:baseline",
        "preview_repository_id": "repository:baseline",
        "preview_tree_id": "tree:baseline",
        "preview_objective_id": "objective:baseline",
        "preview_objective_revision": "revision:baseline",
        "preview_policy_id": "policy:baseline",
        "preview_policy_revision": "revision:baseline",
    }


def _capture_parser(function: object) -> argparse.ArgumentParser:
    with patch.object(
        argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: self,
    ):
        parser = function([])  # type: ignore[operator]
    assert isinstance(parser, argparse.ArgumentParser)
    return parser


def _long_options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


def _default(parser: argparse.ArgumentParser, destination: str) -> object:
    return next(
        action.default
        for action in parser._actions
        if action.dest == destination
    )


def test_direct_requests_require_all_nine_target_bindings() -> None:
    assert _IDENTITY_ARGUMENTS == tuple(
        flag.removeprefix("--").replace("-", "_") for flag in TARGET_FLAGS
    )
    args = _agent_parser().parse_args(
        [
            "agent",
            "workflow-preview",
            "--directory",
            ".",
            "--prompt",
            "inventory this repository",
        ]
    )

    with pytest.raises(AgentCLIError) as raised:
        build_agent_request(args)

    message = str(raised.value)
    assert message.startswith("explicit target bindings are required:")
    assert all(flag in message for flag in TARGET_FLAGS)


def test_real_mutations_require_all_guard_bindings(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    args = _agent_parser().parse_args(
        [
            "agent",
            "workflow-create",
            *_binding_argv(repository_root, state_root),
            "--parameters-json",
            json.dumps(_preview_bindings()),
        ]
    )

    with pytest.raises(AgentCLIError) as raised:
        build_agent_request(args)

    message = str(raised.value)
    for binding in (
        "authorization",
        "idempotency",
        "lease",
        "fencing",
        "expected effects",
    ):
        assert binding in message


def test_default_service_capability_report_has_no_prompt_handlers(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
    )
    request = OperationRequest(
        operation=Operation.CAPABILITIES,
        **_bindings(repository_root, state_root),
    )

    result = service.capabilities(request)
    operations = set(result.to_record()["data"]["operations"])

    assert result.succeeded
    assert operations == {operation.value for operation in READ_OPERATIONS}
    assert Operation.WORKFLOW_PREVIEW.value not in operations
    assert Operation.WORKFLOW_MATERIALIZE.value not in operations
    assert result.to_record()["data"]["optional_providers_loaded"] is False
    assert result.to_record()["data"]["processes_started"] is False


def test_prompt_body_has_no_bridge_after_request_construction(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    prompt = "baseline secret prompt body"
    args = _agent_parser().parse_args(
        [
            "agent",
            "workflow-preview",
            *_binding_argv(repository_root, state_root),
            "--directory",
            str(repository_root),
            "--prompt",
            prompt,
        ]
    )

    request = build_agent_request(args)
    source = dict(request.parameters["prompt_source"])
    serialized = json.dumps(request.to_record())

    assert set(source) == {"kind", "content_cid"}
    assert source["kind"] == "inline"
    assert source["content_cid"].startswith("b")
    assert prompt not in serialized
    assert "inline_text" not in source
    assert "_transient_body" not in source


def test_start_intent_is_dropped_by_unified_cli_and_rejected_by_catalog(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    parser = _agent_parser()
    args = parser.parse_args(
        [
            "agent",
            "workflow-create",
            *_binding_argv(repository_root, state_root),
            "--parameters-json",
            json.dumps(_preview_bindings()),
            "--dry-run",
            "--start",
        ]
    )
    request = build_agent_request(args)

    assert args.start_after is True
    assert "start_after_materialize" not in request.parameters

    module_args = build_prompt_workflow_arg_parser().parse_args(
        ["workflow-create", "--start"]
    )
    module_parameters: dict[str, object] = {}
    _merge_prompt_parameters(
        module_args,
        stdin_stream=io.StringIO(),
        parameters=module_parameters,
    )
    assert module_parameters == {"start_after_materialize": True}

    with pytest.raises(ControlContractError, match="unsupported fields"):
        OperationRequest(
            operation=Operation.WORKFLOW_MATERIALIZE,
            **_bindings(repository_root, state_root),
            parameters={
                **_preview_bindings(),
                **module_parameters,
            },
            dry_run=True,
        )


def test_low_level_entrypoint_flag_counts_are_measured_from_parsers() -> None:
    supervisor_options = _long_options(
        _capture_parser(implementation_supervisor.parse_args)
    )
    daemon_options = _long_options(
        _capture_parser(implementation_daemon.parse_args)
    )

    assert len(supervisor_options) == 140
    assert len(supervisor_options - {"--help"}) == 139
    assert len(daemon_options) == 57
    assert len(daemon_options - {"--help"}) == 56


def test_state_root_defaults_diverge_across_launch_surfaces(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path.resolve()
    supervisor_parser = _capture_parser(implementation_supervisor.parse_args)
    daemon_parser = _capture_parser(implementation_daemon.parse_args)
    prompt_parser = build_prompt_workflow_arg_parser()

    observed = {
        objective_state_root(repository_root),
        bundle_state_root(repository_root),
        repository_root
        / _default(supervisor_parser, "state_dir"),  # type: ignore[operator]
        repository_root
        / _default(daemon_parser, "state_dir"),  # type: ignore[operator]
    }

    assert observed == {
        repository_root / "data" / "agent_supervisor",
        repository_root / "data" / "agent_supervisor" / "bundle_lanes",
        repository_root / "data" / "portal_implementation" / "state",
    }
    assert _default(prompt_parser, "state_root") is None


def test_packaging_exposes_expert_binaries_but_no_prompt_console_script() -> None:
    scripts = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"]
    expected = {
        "ipfs-accelerate": "ipfs_accelerate_py.cli_entry:main",
        "ipfs-accelerate-agent-objective-daemon": (
            "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main"
        ),
        "ipfs-accelerate-agent-backlog-refinery": (
            "ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main"
        ),
        "ipfs-accelerate-agent-bundle-supervisor": (
            "ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor:main"
        ),
        "ipfs-accelerate-agent-artifact-query": (
            "ipfs_accelerate_py.agent_supervisor.runtime.artifact_store:main"
        ),
        "ipfs-accelerate-agent-implementation-daemon": (
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon:main"
        ),
        "ipfs-accelerate-agent-implementation-supervisor": (
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_supervisor:main"
        ),
        "ipfs-accelerate-agent-merge-resolver": (
            "ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main"
        ),
        "ipfs-accelerate-agent-llm-merge-resolver-fallback": (
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "llm_merge_resolver_fallback:main"
        ),
    }

    assert all(scripts.get(name) == target for name, target in expected.items())
    assert not any(
        "prompt" in name or "prompt_workflow" in target
        for name, target in scripts.items()
    )
    assert len(COMMAND_OPERATIONS) == 31
    assert len(USAGE_CLI_COMMANDS) == 15
