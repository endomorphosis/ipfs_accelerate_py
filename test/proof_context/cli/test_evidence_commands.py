"""PCCE-042 evidence-command admission tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.proof_context.cli.evidence_commands import (
    COMMANDS, EvidenceContext, EvidenceRequest, _admit_parent, cmd_expand_context,
)
from ipfs_accelerate_py.proof_context.errors import ProofContextError


def _context(repository: Path, command: str = "expand-context") -> EvidenceContext:
    return EvidenceContext(command=command, repository=repository, policy="production", task_id="PCCE-042", correlation_id="trace-42", output_mode="json", run_id="run-42", repository_id="repo-42", patch_id="patch-42")


def _parent(context: EvidenceContext, operation: str = "context-pack") -> dict[str, object]:
    from ipfs_accelerate_py.proof_context.cli.evidence_commands import _canonical_head
    return {"operation": operation, "status": "succeeded", "provenance": "live", "artifact_cid": "bafkreigh2akiscaildc3jnb2c2ntk2im4rwnvm3biagq4mxjydq7n7q6pm", "identities": {"repository_id": context.repository_id, "task_id": context.task_id, "run_id": context.run_id, "trace_id": context.correlation_id, "patch_id": context.patch_id}, "payload": {"canonical_head": _canonical_head(context.repository)}}


def test_command_set_is_closed() -> None:
    assert COMMANDS == ("expand-context", "explain-impact", "assurance", "seal", "report")


def test_parent_rejects_wrong_operation_and_staleness(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    (tmp_path / "x").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "x"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    context = _context(tmp_path)
    parent = _parent(context)
    assert _admit_parent(EvidenceRequest(context, parent))["artifact_cid"] == parent["artifact_cid"]
    parent["operation"] = "verify"
    with pytest.raises(ProofContextError, match="operation"):
        _admit_parent(EvidenceRequest(context, parent))
    parent["operation"] = "context-pack"
    parent["payload"] = {"canonical_head": "0" * 40}
    with pytest.raises(ProofContextError) as caught:
        _admit_parent(EvidenceRequest(context, parent))
    assert caught.value.status == "stale"


def test_command_rejects_missing_parent_without_opening_runtime(tmp_path: Path) -> None:
    result = cmd_expand_context(_context(tmp_path))
    assert result.status == "invalid"
    assert result.error == "malformed"


def test_unavailable_parent_remains_unavailable(tmp_path: Path) -> None:
    context = _context(tmp_path)
    parent = {"operation": "context-pack", "status": "unavailable"}
    with pytest.raises(ProofContextError) as caught:
        _admit_parent(EvidenceRequest(context, parent))
    assert caught.value.status == "unavailable"
