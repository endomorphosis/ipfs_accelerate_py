"""PCCE-041 execution-command boundary tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken
from ipfs_accelerate_py.proof_context.adapters.external_patch import cid_for_bytes
from ipfs_accelerate_py.proof_context.adapters.models import (
    CONTEXT_PACK_SCHEMA, MODEL_ROUTE_DECISION_SCHEMA, TASK_SPECIFICATION_SCHEMA,
    ContextPack, ModelRouteDecision, TaskSpecification,
)
from ipfs_accelerate_py.proof_context.adapters.registry import AdapterConfiguration
from ipfs_accelerate_py.proof_context.cli.execution_commands import (
    ExecutionContext, RunRequest, cmd_resume, cmd_run, cmd_verify,
)
import ipfs_accelerate_py.proof_context.cli.execution_commands as execution_commands
from ipfs_accelerate_py.proof_context.cli.state_commands import _identities
from ipfs_accelerate_py.proof_context.bootstrap import create_ordinary_python_repository
from dataclasses import replace

from ipfs_accelerate_py.proof_context.facade import ENGINE_RECORD_SCHEMA, EngineRecord

PATCH = (b"diff --git a/src/demo/__init__.py b/src/demo/__init__.py\n--- a/src/demo/__init__.py\n+++ b/src/demo/__init__.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")


def _context(tmp_path: Path, command: str, *, patch_id: str | None = None) -> ExecutionContext:
    return ExecutionContext(command, tmp_path / "repo", "production", "PCCE-041", "corr-041", "json", "run-041", state_dir=tmp_path / "repo" / ".pcce-state", patch_id=patch_id)


def _request(tmp_path: Path, *, cancellation: CancellationToken | None = None) -> RunRequest:
    context = _context(tmp_path, "run")
    identities = _identities(context)
    route_cid = cid_for_bytes(b"route")
    task = TaskSpecification.from_mapping({"schema": TASK_SPECIFICATION_SCHEMA, "task_id": context.task_id, "objective_id": "PCCE-G400", "repository_state_cid": identities.repository_state_cid, "owned_paths": ["src/demo/__init__.py"], "declared_files": ["src/demo/__init__.py"], "route_cid": route_cid, "provenance": "live"})
    pack = ContextPack.from_mapping({"schema": CONTEXT_PACK_SCHEMA, "pack_cid": cid_for_bytes(b"pack"), "task_id": context.task_id, "repository_state_cid": identities.repository_state_cid, "sufficiency": "sufficient", "provenance": "live"})
    route = ModelRouteDecision.from_mapping({"schema": MODEL_ROUTE_DECISION_SCHEMA, "decision_cid": route_cid, "task_id": context.task_id, "repository_state_cid": identities.repository_state_cid, "provider": "external", "model": "external-patch", "revision": "r1", "tier": "medium", "provenance": "live"})
    return RunRequest(context, AdapterConfiguration("external-patch", {"patch": PATCH, "declared_files": ("src/demo/__init__.py",)}), task, pack, route, cancellation)


def test_run_uses_closed_registry_and_returns_patch_identity(tmp_path: Path) -> None:
    create_ordinary_python_repository(tmp_path / "repo")
    request = _request(tmp_path)
    result = cmd_run(request)
    assert result.status == "succeeded"
    assert result.payload["adapter"] == "external-patch"
    assert result.payload["adapter_registry_admitted"] is True
    assert result.payload["patch_id"] == cid_for_bytes(PATCH)
    assert result.identities["patch_id"] == cid_for_bytes(PATCH)


def test_run_rejects_unregistered_or_mismatched_task_before_runtime(tmp_path: Path) -> None:
    request = _request(tmp_path)
    bad = RunRequest(request.context, {"schema": "ipfs-accelerate.proof-context.v0.1/adapter-configuration@1", "name": "shell", "options": {}}, request.task, request.context_pack, request.route)
    assert cmd_run(bad).status == "invalid"
    wrong_task = TaskSpecification.from_mapping({**dict(request.task.to_mapping()), "task_id": "other"})
    assert cmd_run(RunRequest(request.context, request.adapter, wrong_task, request.context_pack, request.route)).status == "invalid"


def _run_record(context: ExecutionContext, patch_id: str) -> EngineRecord:
    return EngineRecord(ENGINE_RECORD_SCHEMA, "run", "succeeded", replace(_identities(context), patch_id=patch_id), cid_for_bytes(b"run"), "live", {"canonical_head": "head"})


def test_verify_rejects_wrong_run_or_patch_identity(tmp_path: Path) -> None:
    create_ordinary_python_repository(tmp_path / "repo")
    patch_id = cid_for_bytes(PATCH)
    context = _context(tmp_path, "verify", patch_id=patch_id)
    wrong = _run_record(_context(tmp_path, "run"), cid_for_bytes(b"other"))
    result = cmd_verify(context, run=wrong)
    assert result.status == "invalid"
    right = _run_record(_context(tmp_path, "run"), patch_id)
    assert cmd_verify(context, run=right).status in {"succeeded", "verification_failed"}
    assert cmd_verify(context, patch_id=patch_id).status in {"succeeded", "verification_failed"}
    assert cmd_verify(context, run=right, patch_id=patch_id).status == "invalid"


def test_cancelled_adapter_and_timeout_are_stable_terminal_results(tmp_path: Path) -> None:
    token = CancellationToken(); token.cancel()
    assert cmd_run(_request(tmp_path, cancellation=token)).status == "cancelled"

    def timed_out(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("adapter timed out")

    original = execution_commands.execute_propose
    execution_commands.execute_propose = timed_out
    try:
        assert cmd_run(_request(tmp_path)).status == "timeout"
    finally:
        execution_commands.execute_propose = original


def test_resume_is_identity_bound_and_idempotently_delegated(tmp_path: Path) -> None:
    create_ordinary_python_repository(tmp_path / "repo")
    context = _context(tmp_path, "resume")
    first = cmd_resume(context)
    second = cmd_resume(context)
    assert first.status == second.status
    assert first.payload["resumed_by_run_id"] == context.run_id
    assert second.payload["idempotent"] is True
    assert cmd_resume(context, checkpoint={"identities": {"task_id": context.task_id, "run_id": "other", "trace_id": context.correlation_id}}).status == "invalid"
