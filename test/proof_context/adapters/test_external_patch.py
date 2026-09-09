"""PCCE-034 external patch ingestion tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken, execute_propose
from ipfs_accelerate_py.proof_context.adapters.external_patch import (
    ADMISSION_SCHEMA, EXTERNAL_PROVENANCE, ExternalPatch, ExternalPatchAdapter,
    cid_for_bytes, parse_patch_paths,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CONTEXT_PACK_SCHEMA, MODEL_ROUTE_DECISION_SCHEMA, TASK_SPECIFICATION_SCHEMA,
    ContextPack, ModelRouteDecision, TaskSpecification,
)
from ipfs_accelerate_py.proof_context.errors import BoundaryViolationError, MalformedError, ProofCancelledError

CID = cid_for_bytes(b"repository")
PACK_CID = cid_for_bytes(b"pack")
ROUTE_CID = cid_for_bytes(b"route")
PATCH = b"diff --git a/src/demo.py b/src/demo.py\n--- a/src/demo.py\n+++ b/src/demo.py\n@@ -1 +1 @@\n-old\n+new\n"


def _records() -> tuple[TaskSpecification, ContextPack, ModelRouteDecision]:
    task = TaskSpecification.from_mapping({"schema": TASK_SPECIFICATION_SCHEMA, "task_id": "PCCE-034", "objective_id": "PCCE-G300", "repository_state_cid": CID, "owned_paths": ["src/demo.py"], "declared_files": ["src/demo.py"], "route_cid": ROUTE_CID, "provenance": "live"})
    pack = ContextPack.from_mapping({"schema": CONTEXT_PACK_SCHEMA, "pack_cid": PACK_CID, "repository_state_cid": CID, "task_id": task.task_id, "sufficiency": "sufficient", "provenance": "live"})
    route = ModelRouteDecision.from_mapping({"schema": MODEL_ROUTE_DECISION_SCHEMA, "decision_cid": ROUTE_CID, "task_id": task.task_id, "repository_state_cid": CID, "provider": "external", "model": "human-authored", "revision": "unspecified", "tier": "medium", "provenance": "live"})
    return task, pack, route


def test_external_patch_has_exact_byte_identity_and_declared_paths() -> None:
    patch = ExternalPatch(PATCH, ("src/demo.py",))
    assert patch.patch_cid == cid_for_bytes(PATCH)
    assert parse_patch_paths(PATCH) == ("src/demo.py",)


@pytest.mark.parametrize("patch", [b"", b"diff --git a/src/demo.py b/src/demo.py\n\x00", b"diff --git a/src/demo.py b/src/demo.py\n\xff", b"not a diff\n"])
def test_invalid_patch_encodings_or_formats_fail(patch: bytes) -> None:
    with pytest.raises(MalformedError):
        ExternalPatch(patch, ("src/demo.py",))


@pytest.mark.parametrize("path", ["../secret.py", "/etc/passwd", "src/../secret.py", "C:/secret.py"])
def test_traversal_and_absolute_paths_fail(path: str) -> None:
    patch = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n".encode()
    with pytest.raises(BoundaryViolationError):
        ExternalPatch(patch, (path,))


def test_parsed_paths_must_agree_exactly_with_declaration_and_scope() -> None:
    with pytest.raises(BoundaryViolationError):
        ExternalPatch(PATCH, ("src/other.py",))
    task, pack, route = _records()
    external = ExternalPatch(b"diff --git a/src/secret.py b/src/secret.py\n--- a/src/secret.py\n+++ b/src/secret.py\n@@ -0,0 +1 @@\n+x\n", ("src/secret.py",))
    with pytest.raises(BoundaryViolationError):
        execute_propose(ExternalPatchAdapter(external), task, pack, route)


def test_valid_external_patch_enters_normal_adapter_lifecycle_without_apply() -> None:
    task, pack, route = _records()
    result = execute_propose(ExternalPatchAdapter(PATCH, ("src/demo.py",)), task, pack, route)
    assert result.proposal.patch_cid == cid_for_bytes(PATCH)
    assert result.proposal.declared_files == ("src/demo.py",)
    assert result.proposal.provenance == result.invocation.provenance == "live"
    assert result.accepted is result.approved is False
    assert b'"provenance":"external"' in result.log_bytes
    assert ADMISSION_SCHEMA.encode() in result.log_bytes
    assert EXTERNAL_PROVENANCE == "external"


def test_adapter_cancellation_and_identity_binding_fail_closed() -> None:
    task, pack, route = _records()
    token = CancellationToken(); token.cancel()
    with pytest.raises(ProofCancelledError):
        execute_propose(ExternalPatchAdapter(PATCH, ("src/demo.py",)), task, pack, route, token)
    with pytest.raises(BoundaryViolationError):
        execute_propose(ExternalPatchAdapter(PATCH, ("src/demo.py",)), task, replace(pack, sufficiency="insufficient"), route)
