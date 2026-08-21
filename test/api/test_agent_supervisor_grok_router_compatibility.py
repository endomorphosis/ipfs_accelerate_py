"""Compatibility checks for the supervised Grok runner's router surface."""

from __future__ import annotations

import ast
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    GROK_QUOTA_PROBE_CONTRACT,
    build_grok_failure_receipt,
    valid_grok_failure_receipt,
)

from ipfs_accelerate_py import llm_router


def test_runtime_grok_runner_router_import_surface_is_complete() -> None:
    """Every statically declared router dependency remains importable."""

    source_path = Path(grok_cli_runner.__file__)
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    required = {
        alias.name
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module == "ipfs_accelerate_py.llm_router"
        for alias in node.names
    }

    assert required
    assert sorted(name for name in required if not hasattr(llm_router, name)) == []


def test_transient_preflight_retry_uses_current_validated_receipt() -> None:
    """The compatibility export admits only the exact Grok 4.6 artifact."""

    nonce = "a" * 64
    evidence = "Error: max turns reached\n"
    receipt = build_grok_failure_receipt(
        probe_stderr_text=evidence,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )

    assert llm_router.AGENT_IMPLEMENTATION_PRIMARY_MODEL_ID == "grok-4.6"
    assert GROK_QUOTA_PROBE_CONTRACT["model"] == "grok-4.6"
    assert (
        llm_router._LEGACY_AGENT_IMPLEMENTATION_ROUTE.primary_model_id
        == "grok-4.6"
    )
    assert "grok45" in llm_router._LEGACY_AGENT_IMPLEMENTATION_ROUTE.route_id
    assert (
        llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE.primary_model_id
        == "grok-4.6"
    )
    assert valid_grok_failure_receipt(
        receipt,
        nonce=nonce,
        model="grok-4.6",
        returncode=41,
    )
    assert llm_router.retryable_agent_implementation_preflight_failure(
        evidence,
        receipt,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
    )
    assert not llm_router.retryable_agent_implementation_preflight_failure(
        evidence.rstrip("\n"),
        receipt,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
    )
