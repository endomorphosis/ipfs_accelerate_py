"""Tests for CLI integration compatibility facade and lazy registry (GOOSE-010).

Coverage:
- lazy discovery / list_cli_integrations does not instantiate wrappers
- unavailable tools degrade without process probes on import
- compatibility imports and getters remain valid
- Codex uses codex exec command parity
- Copilot uses production command contract (not github-copilot-cli suggest)
- no import-time process probes
- side-effecting ops disable generic cache and retries
- operator overrides are argv-only / shell-free
- llm_router provider discovery behavior is unchanged
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_subprocess_run_calls(mock_run: MagicMock) -> int:
    return int(mock_run.call_count)


# ---------------------------------------------------------------------------
# Compatibility imports
# ---------------------------------------------------------------------------


def test_compatibility_imports_remain_valid() -> None:
    from ipfs_accelerate_py.cli_integrations import (
        BaseCLIWrapper,
        CopilotCLIIntegration,
        GooseCLIIntegration,
        OpenAICodexCLIIntegration,
        get_all_cli_integrations,
        get_copilot_cli_integration,
        get_goose_cli_integration,
        get_openai_codex_cli_integration,
        list_cli_integrations,
    )

    assert BaseCLIWrapper is not None
    assert callable(get_goose_cli_integration)
    assert callable(get_openai_codex_cli_integration)
    assert callable(get_copilot_cli_integration)
    assert callable(get_all_cli_integrations)
    assert callable(list_cli_integrations)
    assert issubclass(GooseCLIIntegration, object)
    assert issubclass(OpenAICodexCLIIntegration, BaseCLIWrapper)
    assert issubclass(CopilotCLIIntegration, BaseCLIWrapper)


def test_direct_module_imports() -> None:
    goose = importlib.import_module(
        "ipfs_accelerate_py.cli_integrations.goose_cli_integration"
    )
    codex = importlib.import_module(
        "ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration"
    )
    copilot = importlib.import_module(
        "ipfs_accelerate_py.cli_integrations.copilot_cli_integration"
    )
    assert hasattr(goose, "GooseCLIIntegration")
    assert hasattr(goose, "get_goose_cli_integration")
    assert hasattr(codex, "OpenAICodexCLIIntegration")
    assert hasattr(codex, "build_codex_exec_argv")
    assert hasattr(copilot, "CopilotCLIIntegration")
    assert hasattr(copilot, "build_copilot_suggest_argv")


# ---------------------------------------------------------------------------
# Lazy discovery
# ---------------------------------------------------------------------------


def test_list_cli_integrations_is_metadata_only_no_instantiation() -> None:
    from ipfs_accelerate_py.cli_integrations import list_cli_integrations

    with patch(
        "ipfs_accelerate_py.cli_integrations.goose_cli_integration.GooseCLIIntegration",
        side_effect=AssertionError("must not construct Goose"),
    ), patch(
        "ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration.OpenAICodexCLIIntegration",
        side_effect=AssertionError("must not construct Codex"),
    ), patch(
        "ipfs_accelerate_py.cli_integrations.copilot_cli_integration.CopilotCLIIntegration",
        side_effect=AssertionError("must not construct Copilot"),
    ):
        listed = list_cli_integrations()

    names = {entry["name"] for entry in listed}
    assert "goose" in names
    assert "openai_codex" in names
    assert "copilot" in names
    assert "github" in names
    for entry in listed:
        assert "class_name" in entry
        assert "getter" in entry
        assert "module" in entry
        assert "command_contract" in entry


def test_get_all_cli_integrations_default_is_lazy_factories() -> None:
    from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations

    with patch(
        "subprocess.run",
        side_effect=AssertionError("subprocess.run must not run during listing"),
    ):
        mapping = get_all_cli_integrations()

    assert "goose" in mapping
    assert "openai_codex" in mapping
    assert "copilot" in mapping
    # Default is factories, not instances
    assert callable(mapping["goose"])
    assert callable(mapping["openai_codex"])


def test_get_all_cli_integrations_eager_emits_deprecation() -> None:
    from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        reset_goose_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        reset_openai_codex_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        reset_copilot_cli_integration,
    )

    reset_goose_cli_integration()
    reset_openai_codex_cli_integration()
    reset_copilot_cli_integration()

    # Constructing dual-mode wrappers may probe; only assert deprecation + keys.
    with pytest.warns(DeprecationWarning, match="eagerly constructs"):
        with patch("subprocess.run") as mock_run:
            # Prevent real probes if any dual-mode wrappers still probe.
            mock_run.return_value = types.SimpleNamespace(
                returncode=1, stdout="", stderr=""
            )
            mapping = get_all_cli_integrations(instantiate=True)

    assert "goose" in mapping
    assert "openai_codex" in mapping
    # Goose facade should not require subprocess on construct
    goose = mapping["goose"]
    assert goose.get_tool_name() == "Goose CLI"


# ---------------------------------------------------------------------------
# No import-time probes
# ---------------------------------------------------------------------------


def test_importing_cli_integrations_package_starts_no_subprocess() -> None:
    # Re-import in a controlled way while subprocess is blocked.
    with patch("subprocess.run", side_effect=AssertionError("import-time probe")):
        # Importing symbols already loaded is fine; re-execute key modules.
        importlib.reload(
            importlib.import_module(
                "ipfs_accelerate_py.cli_integrations.base_cli_wrapper"
            )
        )
        importlib.reload(
            importlib.import_module(
                "ipfs_accelerate_py.cli_integrations.goose_cli_integration"
            )
        )
        importlib.reload(
            importlib.import_module(
                "ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration"
            )
        )
        importlib.reload(
            importlib.import_module(
                "ipfs_accelerate_py.cli_integrations.copilot_cli_integration"
            )
        )


def test_constructing_goose_codex_copilot_facades_starts_no_subprocess() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        GooseCLIIntegration,
        reset_goose_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        OpenAICodexCLIIntegration,
        reset_openai_codex_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        CopilotCLIIntegration,
        reset_copilot_cli_integration,
    )

    reset_goose_cli_integration()
    reset_openai_codex_cli_integration()
    reset_copilot_cli_integration()

    with patch("subprocess.run", side_effect=AssertionError("init probe")):
        goose = GooseCLIIntegration()
        codex = OpenAICodexCLIIntegration()
        copilot = CopilotCLIIntegration()

    assert goose.get_tool_name() == "Goose CLI"
    assert codex.get_tool_name() == "OpenAI Codex CLI"
    assert copilot.get_tool_name() == "GitHub Copilot CLI"
    # Availability without probe must not call subprocess
    with patch("subprocess.run", side_effect=AssertionError("avail probe")):
        _ = goose.is_available(probe=False)
        _ = codex.is_available(probe=False)
        _ = copilot.is_available(probe=False)


# ---------------------------------------------------------------------------
# Unavailable tools
# ---------------------------------------------------------------------------


def test_unavailable_goose_is_detectable_without_crash() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        GooseCLIIntegration,
    )

    with patch(
        "ipfs_accelerate_py.cli_integrations.goose_cli_integration._discover_goose_executable",
        return_value=None,
    ), patch.dict(os.environ, {}, clear=False):
        # Clear goose-related env keys for the check path
        env_clear = {
            k: ""
            for k in list(os.environ)
            if "GOOSE" in k.upper()
        }
        with patch.dict(os.environ, env_clear, clear=False):
            integ = GooseCLIIntegration(goose_path="/nonexistent/goose-binary-xyz")
            assert integ.is_available(probe=False) is False


def test_unavailable_codex_path_is_reported() -> None:
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        OpenAICodexCLIIntegration,
    )

    integ = OpenAICodexCLIIntegration(
        codex_path="/nonexistent/codex-binary-xyz",
        command_override=["/nonexistent/codex-binary-xyz"],
    )
    assert integ.is_available(probe=False) is False


# ---------------------------------------------------------------------------
# Goose facade → canonical adapter
# ---------------------------------------------------------------------------


def test_goose_chat_delegates_to_canonical_adapter() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        GooseCLIIntegration,
    )

    mock_adapter = MagicMock()
    mock_adapter.generate.return_value = "hello from goose"

    integ = GooseCLIIntegration(adapter=mock_adapter)
    result = integ.chat("hi", model="test-model", timeout=12.0)

    mock_adapter.generate.assert_called_once()
    args, kwargs = mock_adapter.generate.call_args
    assert args[0] == "hi"
    assert kwargs.get("model_name") == "test-model"
    assert kwargs.get("agent") is False
    assert kwargs.get("timeout") == 12.0
    assert result["success"] is True
    assert result["text"] == "hello from goose"
    assert result["provider"] == "goose_cli"
    assert result["side_effecting"] is False


def test_goose_agent_delegates_with_side_effect_flags() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        GooseCLIIntegration,
    )

    mock_adapter = MagicMock()
    mock_adapter.generate.return_value = "agent done"

    integ = GooseCLIIntegration(adapter=mock_adapter)
    result = integ.agent(
        "do work",
        workspace="/tmp/ws",
        path_root="/tmp/ws",
        max_turns=5,
    )

    mock_adapter.generate.assert_called_once()
    _, kwargs = mock_adapter.generate.call_args
    assert kwargs.get("agent") is True
    assert kwargs.get("workspace") == "/tmp/ws"
    assert kwargs.get("path_root") == "/tmp/ws"
    assert kwargs.get("allow_side_effects") is True
    assert result["side_effecting"] is True
    assert result["mode"] == "agent"


def test_goose_get_adapter_uses_create_goose_provider() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        GooseCLIIntegration,
    )

    sentinel = MagicMock(name="GooseCLIProvider")
    with patch(
        "ipfs_accelerate_py.cli_runtime.providers.goose.create_goose_provider",
        return_value=sentinel,
    ) as create:
        integ = GooseCLIIntegration(goose_path="/usr/bin/goose")
        adapter = integ.get_adapter()
        assert adapter is sentinel
        create.assert_called_once()
        # Second call reuses
        assert integ.get_adapter() is sentinel
        assert create.call_count == 1


# ---------------------------------------------------------------------------
# Codex command parity
# ---------------------------------------------------------------------------


def test_codex_build_argv_uses_exec_not_completions() -> None:
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        build_codex_exec_argv,
    )

    argv = build_codex_exec_argv(
        base_argv=["codex"],
        model="chatgpt-5.6-terra",
        prompt="write a function",
        last_message_path="/tmp/last.txt",
        sandbox="auto",
        skip_git_repo_check=True,
    )
    assert argv[0] == "codex"
    assert "exec" in argv
    assert "api" not in argv
    assert "completions.create" not in argv
    assert "--skip-git-repo-check" in argv
    assert "-m" in argv
    assert "chatgpt-5.6-terra" in argv
    assert "--output-last-message" in argv
    assert argv[-1] == "-"
    # sandbox auto must not inject --sandbox
    assert "--sandbox" not in argv


def test_codex_generate_code_runs_exec_with_stdin_and_no_retry() -> None:
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        OpenAICodexCLIIntegration,
    )

    cache = MagicMock()
    cache.get.return_value = None

    integ = OpenAICodexCLIIntegration(
        codex_path="codex",
        enable_cache=True,
        cache=cache,
        max_retries=5,
    )

    completed = types.SimpleNamespace(
        returncode=0,
        stdout="unused",
        stderr="",
    )

    with patch("subprocess.run", return_value=completed) as mock_run, patch(
        "builtins.open",
        create=True,
    ) as mock_open:
        # Simulate last-message file content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "generated code"
        )
        result = integ.generate_code("make a fib", model="m1")

    assert result["success"] is True
    assert result["command_contract"] == "codex exec"
    assert result["attempts"] == 1
    assert result["side_effecting"] is True
    # Cache must not be consulted or written for side-effecting exec
    cache.get.assert_not_called()
    cache.put.assert_not_called()

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "codex"
    assert "exec" in cmd
    assert "completions.create" not in cmd
    assert call_args.kwargs.get("shell") is False or (
        len(call_args) > 1 and call_args[1].get("shell") is False
    )
    # stdin carries the prompt
    assert call_args.kwargs.get("input") == "make a fib" or (
        call_args[1].get("input") == "make a fib"
        if len(call_args) > 1
        else call_args.kwargs.get("input") == "make a fib"
    )


def test_codex_operator_command_override_is_argv_only() -> None:
    from ipfs_accelerate_py.cli_integrations.base_cli_wrapper import parse_argv_override
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        OpenAICodexCLIIntegration,
        build_codex_exec_argv,
    )

    override = parse_argv_override('/opt/custom/codex --flag "with space"')
    assert override == ["/opt/custom/codex", "--flag", "with space"]

    integ = OpenAICodexCLIIntegration(command_override=override)
    argv = build_codex_exec_argv(
        base_argv=integ._base_argv(),
        model="m",
        prompt="p",
        last_message_path="/tmp/x",
    )
    assert argv[0] == "/opt/custom/codex"
    assert argv[1] == "--flag"
    assert argv[2] == "with space"
    assert "exec" in argv


# ---------------------------------------------------------------------------
# Copilot production command contract
# ---------------------------------------------------------------------------


def test_copilot_build_argv_uses_production_flags_not_legacy_suggest() -> None:
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        build_copilot_explain_argv,
        build_copilot_suggest_argv,
    )

    argv = build_copilot_suggest_argv(
        base_argv=["npx", "--yes", "@github/copilot"],
        prompt="list files",
        model="gpt-5-mini",
    )
    assert argv[:3] == ["npx", "--yes", "@github/copilot"]
    assert "--silent" in argv
    assert "--stream" in argv
    assert "off" in argv
    assert "-i" in argv
    assert "list files" in argv
    # Legacy github-copilot-cli suggest syntax must not appear
    assert "--explain" not in argv
    assert argv[0] != "github-copilot-cli"

    explain = build_copilot_explain_argv(
        base_argv=["copilot"],
        command="ls -la",
        model="gpt-5-mini",
    )
    assert "--silent" in explain
    assert "-i" in explain
    assert any("ls -la" in part for part in explain)
    assert "--explain" not in explain


def test_copilot_suggest_command_runs_production_contract_no_cache() -> None:
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        CopilotCLIIntegration,
    )

    cache = MagicMock()
    cache.get.return_value = None

    integ = CopilotCLIIntegration(
        enable_cache=True,
        cache=cache,
        command_override=["npx", "--yes", "@github/copilot"],
        max_retries=4,
    )

    completed = types.SimpleNamespace(
        returncode=0,
        stdout="ls -la",
        stderr="",
    )
    with patch("subprocess.run", return_value=completed) as mock_run:
        result = integ.suggest_command("list all files", shell="bash")

    assert result["success"] is True
    assert result["command_contract"] == "copilot_production"
    assert result["attempts"] == 1
    assert result["side_effecting"] is True
    cache.get.assert_not_called()
    cache.put.assert_not_called()

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "npx"
    assert "@github/copilot" in cmd
    assert "--silent" in cmd
    assert "-i" in cmd
    assert "github-copilot-cli" not in cmd
    assert mock_run.call_args.kwargs.get("shell") is False


def test_copilot_env_override_is_argv_only() -> None:
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        CopilotCLIIntegration,
        reset_copilot_cli_integration,
    )

    reset_copilot_cli_integration()
    env = {
        "ipfs_accelerate_py_COPILOT_CLI_CMD": "copilot --config-dir /tmp/c",
    }
    with patch.dict(os.environ, env, clear=False):
        integ = CopilotCLIIntegration()
        base = integ._base_argv()
        assert base[0] == "copilot"
        assert "--config-dir" in base
        assert "/tmp/c" in base


# ---------------------------------------------------------------------------
# Side-effecting base policy
# ---------------------------------------------------------------------------


def test_base_wrapper_disables_cache_and_retry_when_side_effecting() -> None:
    from ipfs_accelerate_py.cli_integrations.base_cli_wrapper import BaseCLIWrapper

    class _Tool(BaseCLIWrapper):
        def get_tool_name(self) -> str:
            return "Tool"

    cache = MagicMock()
    cache.get.return_value = {"stdout": "cached"}
    tool = _Tool(
        cli_path="echo",
        cache=cache,
        enable_cache=True,
        max_retries=5,
        verify_on_init=False,
    )

    # Non-side-effecting: uses cache
    hit = tool._run_command_with_retry(["hi"], "op", side_effecting=False)
    assert hit.get("stdout") == "cached"
    cache.get.assert_called()

    cache.reset_mock()
    cache.get.return_value = None
    completed = types.SimpleNamespace(returncode=1, stdout="", stderr="fail")
    with patch("subprocess.run", return_value=completed) as mock_run:
        result = tool._run_command_with_retry(
            ["hi"], "op", side_effecting=True
        )

    assert result["attempts"] == 1
    assert result["side_effecting"] is True
    cache.get.assert_not_called()
    cache.put.assert_not_called()
    assert mock_run.call_count == 1
    assert mock_run.call_args.kwargs.get("shell") is False


def test_parse_argv_override_rejects_shell_string_interpolation_patterns() -> None:
    from ipfs_accelerate_py.cli_integrations.base_cli_wrapper import parse_argv_override

    # Metacharacters stay as data inside tokens; we never pass shell=True.
    parts = parse_argv_override('tool --msg "a; rm -rf /"')
    assert parts is not None
    assert parts[0] == "tool"
    assert parts[1] == "--msg"
    assert parts[2] == "a; rm -rf /"


# ---------------------------------------------------------------------------
# llm_router provider behavior unchanged
# ---------------------------------------------------------------------------


def test_llm_router_builtin_provider_discovery_unchanged() -> None:
    """Compatibility changes must not break router provider lookup APIs."""
    from ipfs_accelerate_py import llm_router_available
    from ipfs_accelerate_py.llm_router import _builtin_provider_by_name
    from ipfs_accelerate_py.router_deps import get_default_router_deps

    assert llm_router_available is True or llm_router_available is False
    # Ensure router deps singleton still constructs (unchanged public surface).
    assert get_default_router_deps() is not None
    # Function must remain callable and not raise for common CLI providers
    # even when the tool is missing (returns None or a provider).
    for name in ("codex_cli", "copilot_cli", "goose_cli", "claude_code"):
        provider = _builtin_provider_by_name(name)
        assert provider is None or hasattr(provider, "generate")


def test_llm_router_codex_still_uses_codex_exec_semantics() -> None:
    """Sanity: router codex provider source still references codex exec."""
    import inspect

    import ipfs_accelerate_py.llm_router as router

    source = inspect.getsource(router._get_codex_cli_provider)
    assert "codex" in source
    assert "exec" in source
    assert "completions.create" not in source


def test_llm_router_copilot_still_uses_production_default() -> None:
    import inspect

    import ipfs_accelerate_py.llm_router as router

    source = inspect.getsource(router._get_copilot_cli_provider)
    assert "@github/copilot" in source
    assert "github-copilot-cli" not in source


# ---------------------------------------------------------------------------
# Getter singletons
# ---------------------------------------------------------------------------


def test_getters_return_stable_singletons() -> None:
    from ipfs_accelerate_py.cli_integrations.goose_cli_integration import (
        get_goose_cli_integration,
        reset_goose_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration import (
        get_openai_codex_cli_integration,
        reset_openai_codex_cli_integration,
    )
    from ipfs_accelerate_py.cli_integrations.copilot_cli_integration import (
        get_copilot_cli_integration,
        reset_copilot_cli_integration,
    )

    reset_goose_cli_integration()
    reset_openai_codex_cli_integration()
    reset_copilot_cli_integration()

    with patch("subprocess.run", side_effect=AssertionError("probe")):
        g1 = get_goose_cli_integration()
        g2 = get_goose_cli_integration()
        c1 = get_openai_codex_cli_integration()
        c2 = get_openai_codex_cli_integration()
        p1 = get_copilot_cli_integration()
        p2 = get_copilot_cli_integration()

    assert g1 is g2
    assert c1 is c2
    assert p1 is p2
