"""Conformance tests: prompt-only v2 surface across Python, CLI, MCP, MCP++.

Validates ASE2-007 acceptance:
  - run/preview require only a prompt
  - steer requires prompt + optional run handle
  - status/follow infer the sole compatible run
  - all transports agree on canonical outcomes and exit/error semantics
  - advanced flags remain explicit overrides, not requirements
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import facade as facade_mod
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    CANONICAL_OUTCOMES,
    EXIT_AMBIGUOUS,
    EXIT_INVALID,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    OUTCOME_AMBIGUOUS,
    OUTCOME_COMPLETED,
    OUTCOME_INVALID,
    OUTCOME_NOT_FOUND,
    AgentSupervisorFacade,
    outcome_to_exit_code,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as cli_mod
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    prompt_entrypoints as mcp_mod,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> AgentSupervisorFacade:
    return AgentSupervisorFacade()


def _assert_canonical(result: dict) -> None:
    assert isinstance(result, dict)
    assert "outcome" in result
    assert result["outcome"] in CANONICAL_OUTCOMES
    assert "exit_code" in result
    assert result["exit_code"] == outcome_to_exit_code(result["outcome"])


# ---------------------------------------------------------------------------
# Python facade: prompt-only happy paths
# ---------------------------------------------------------------------------

class TestPythonPromptOnly:
    def test_run_prompt_only(self):
        fac = _fresh()
        r = fac.run("do the thing")
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["exit_code"] == EXIT_SUCCESS
        assert "run_id" in r

    def test_preview_prompt_only(self):
        fac = _fresh()
        r = fac.preview("plan the thing")
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["run_id"]

    def test_steer_prompt_only_infers_sole_run(self):
        fac = _fresh()
        first = fac.run("initial")
        rid = first["run_id"]
        # After completion, sole run still inferable when exactly one exists
        r = fac.steer("nudge")
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["run_id"] == rid

    def test_steer_with_explicit_run_id(self):
        fac = _fresh()
        a = fac.run("a")
        b = fac.run("b")
        r = fac.steer("fix a", run_id=a["run_id"])
        _assert_canonical(r)
        assert r["run_id"] == a["run_id"]
        assert r["outcome"] == OUTCOME_COMPLETED

    def test_status_infers_sole_run(self):
        fac = _fresh()
        first = fac.run("solo")
        r = fac.status()
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["run_id"] == first["run_id"]

    def test_follow_infers_sole_run(self):
        fac = _fresh()
        first = fac.run("solo")
        r = fac.follow()
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["run_id"] == first["run_id"]

    def test_status_ambiguous_when_multiple(self):
        fac = _fresh()
        fac.run("a")
        fac.run("b")
        r = fac.status()
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_AMBIGUOUS
        assert r["exit_code"] == EXIT_AMBIGUOUS

    def test_status_not_found_when_empty(self):
        fac = _fresh()
        r = fac.status()
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_NOT_FOUND
        assert r["exit_code"] == EXIT_NOT_FOUND

    def test_missing_prompt_invalid(self):
        fac = _fresh()
        r = fac.run("")  # type: ignore[arg-type]
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_INVALID
        assert r["exit_code"] == EXIT_INVALID

    def test_advanced_flags_are_optional_overrides(self):
        fac = _fresh()
        # Must succeed without model/timeout/backend
        r1 = fac.run("plain")
        assert r1["outcome"] == OUTCOME_COMPLETED
        # And also accept them when provided
        r2 = fac.run("with opts", model="x", timeout=1.5, backend="mem")
        assert r2["outcome"] == OUTCOME_COMPLETED


# ---------------------------------------------------------------------------
# Outcome / exit-code agreement
# ---------------------------------------------------------------------------

class TestCanonicalSemantics:
    def test_outcome_exit_mapping(self):
        assert outcome_to_exit_code(OUTCOME_COMPLETED) == EXIT_SUCCESS
        assert outcome_to_exit_code(OUTCOME_AMBIGUOUS) == EXIT_AMBIGUOUS
        assert outcome_to_exit_code(OUTCOME_NOT_FOUND) == EXIT_NOT_FOUND
        assert outcome_to_exit_code(OUTCOME_INVALID) == EXIT_INVALID

    def test_all_outcomes_have_exit_codes(self):
        for o in CANONICAL_OUTCOMES:
            code = outcome_to_exit_code(o)
            assert isinstance(code, int)


# ---------------------------------------------------------------------------
# CLI transport
# ---------------------------------------------------------------------------

class TestCLIPromptOnly:
    def test_run_prompt_only_json(self):
        fac = _fresh()
        code = cli_mod.dispatch(["run", "do it", "--json"], facade=fac)
        assert code == EXIT_SUCCESS

    def test_preview_prompt_only(self):
        fac = _fresh()
        code = cli_mod.dispatch(["preview", "plan it", "--json"], facade=fac)
        assert code == EXIT_SUCCESS

    def test_status_infers(self, capsys):
        fac = _fresh()
        cli_mod.dispatch(["run", "solo", "--json"], facade=fac)
        code = cli_mod.dispatch(["status", "--json"], facade=fac)
        assert code == EXIT_SUCCESS
        out = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(out)
        assert payload["outcome"] == OUTCOME_COMPLETED

    def test_follow_infers(self, capsys):
        fac = _fresh()
        cli_mod.dispatch(["run", "solo", "--json"], facade=fac)
        code = cli_mod.dispatch(["follow", "--json"], facade=fac)
        assert code == EXIT_SUCCESS

    def test_steer_prompt_optional_run_id(self, capsys):
        fac = _fresh()
        cli_mod.dispatch(["run", "base", "--json"], facade=fac)
        code = cli_mod.dispatch(["steer", "nudge", "--json"], facade=fac)
        assert code == EXIT_SUCCESS

    def test_status_empty_not_found(self, capsys):
        fac = _fresh()
        code = cli_mod.dispatch(["status", "--json"], facade=fac)
        assert code == EXIT_NOT_FOUND
        out = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(out)
        assert payload["outcome"] == OUTCOME_NOT_FOUND

    def test_advanced_flags_optional(self):
        fac = _fresh()
        # No --model / --timeout required
        code = cli_mod.dispatch(["run", "x", "--json"], facade=fac)
        assert code == EXIT_SUCCESS
        code = cli_mod.dispatch(
            ["run", "y", "--json", "--model", "m", "--timeout", "2"],
            facade=fac,
        )
        assert code == EXIT_SUCCESS


# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------

class TestMCPPromptOnly:
    def setup_method(self):
        # Isolate default facade between tests
        facade_mod._default_facade = AgentSupervisorFacade()

    def test_list_tools_prompt_only_required(self):
        tools = mcp_mod.list_tools()
        by_name = {t["name"]: t for t in tools}
        assert "agent_supervisor_run" in by_name
        run_schema = by_name["agent_supervisor_run"]["inputSchema"]
        assert run_schema["required"] == ["prompt"]
        preview_schema = by_name["agent_supervisor_preview"]["inputSchema"]
        assert preview_schema["required"] == ["prompt"]
        steer_schema = by_name["agent_supervisor_steer"]["inputSchema"]
        assert "prompt" in steer_schema["required"]
        assert "run_id" not in steer_schema["required"]
        status_schema = by_name["agent_supervisor_status"]["inputSchema"]
        assert status_schema["required"] == []
        follow_schema = by_name["agent_supervisor_follow"]["inputSchema"]
        assert follow_schema["required"] == []

    def test_run_prompt_only(self):
        r = mcp_mod.call_tool("run", {"prompt": "via mcp"})
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED

    def test_preview_prompt_only(self):
        r = mcp_mod.agent_supervisor_preview(prompt="preview via mcp")
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED

    def test_steer_optional_run_id(self):
        first = mcp_mod.call_tool("run", {"prompt": "base"})
        r = mcp_mod.call_tool("steer", {"prompt": "nudge"})
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED
        assert r["run_id"] == first["run_id"]

    def test_status_infers(self):
        first = mcp_mod.call_tool("agent_supervisor_run", {"prompt": "solo"})
        r = mcp_mod.call_tool("status", {})
        _assert_canonical(r)
        assert r["run_id"] == first["run_id"]

    def test_follow_infers(self):
        first = mcp_mod.call_tool("follow".replace("follow", "run"), {"prompt": "solo"})
        r = mcp_mod.agent_supervisor_follow()
        _assert_canonical(r)
        assert r["run_id"] == first["run_id"]

    def test_missing_prompt_invalid(self):
        r = mcp_mod.agent_supervisor_run(prompt="")
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_INVALID


# ---------------------------------------------------------------------------
# MCP++ transport — same outcomes as MCP
# ---------------------------------------------------------------------------

class TestMCPPlusPlusPromptOnly:
    def setup_method(self):
        facade_mod._default_facade = AgentSupervisorFacade()

    def test_run_matches_mcp_outcome(self):
        facade_mod._default_facade = AgentSupervisorFacade()
        mcp_r = mcp_mod.call_tool("run", {"prompt": "same"})
        facade_mod._default_facade = AgentSupervisorFacade()
        pp_r = mcp_mod.mcplusplus_call("run", {"prompt": "same"})
        assert mcp_r["outcome"] == pp_r["outcome"]
        assert mcp_r["exit_code"] == pp_r["exit_code"]
        assert pp_r.get("transport") == "mcp++"

    def test_status_infers(self):
        mcp_mod.mcplusplus_call("run", {"prompt": "solo"})
        r = mcp_mod.mcplusplus_call("status", {})
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED

    def test_steer_prompt_only(self):
        mcp_mod.mcplusplus_call("run", {"prompt": "base"})
        r = mcp_mod.mcplusplus_call("steer", {"prompt": "nudge"})
        _assert_canonical(r)
        assert r["outcome"] == OUTCOME_COMPLETED


# ---------------------------------------------------------------------------
# Cross-transport agreement
# ---------------------------------------------------------------------------

class TestCrossTransportAgreement:
    def test_invalid_prompt_same_everywhere(self):
        fac = _fresh()
        py = fac.run("")
        facade_mod._default_facade = AgentSupervisorFacade()
        mcp = mcp_mod.call_tool("run", {"prompt": ""})
        assert py["outcome"] == mcp["outcome"] == OUTCOME_INVALID
        assert py["exit_code"] == mcp["exit_code"] == EXIT_INVALID

    def test_not_found_same_everywhere(self):
        fac = _fresh()
        py = fac.status()
        facade_mod._default_facade = AgentSupervisorFacade()
        mcp = mcp_mod.call_tool("status", {})
        assert py["outcome"] == mcp["outcome"] == OUTCOME_NOT_FOUND
        assert py["exit_code"] == mcp["exit_code"] == EXIT_NOT_FOUND

    def test_completed_run_envelope_keys(self):
        fac = _fresh()
        py = fac.run("x")
        facade_mod._default_facade = AgentSupervisorFacade()
        mcp = mcp_mod.call_tool("run", {"prompt": "x"})
        for r in (py, mcp):
            assert set(["outcome", "exit_code", "run_id"]).issubset(r.keys())
