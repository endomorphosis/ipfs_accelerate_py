"""Identity-bound evidence and escalation CLI operations (PCCE-042).

The commands in this module deliberately consume an explicit immutable parent
record.  They do not reconstruct a run from a repository name, manufacture a
parent, or turn an unavailable/failed parent into a successful result.  This
makes the module suitable for the final CLI wiring while keeping canonical
semantic, assurance, and sealing producers authoritative.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Final

from ipfs_accelerate_py.proof_context.cli.app import CliResult
from ipfs_accelerate_py.proof_context.cli.state_commands import (
    _identities,
    _reject_escape,
    _state_dir,
    _failure_result,
    _result_from_record,
)
from ipfs_accelerate_py.proof_context.errors import ProofContextError, redact_text
from ipfs_accelerate_py.proof_context.facade import EngineRecord, OPERATION_CONTRACTS
from ipfs_accelerate_py.proof_context.policy import admit_mode

COMMANDS: Final[tuple[str, ...]] = (
    "expand-context",
    "explain-impact",
    "assurance",
    "seal",
    "report",
)

# Operation -> required canonical parent operation.  The parent is supplied by
# the caller as an immutable artifact envelope, rather than being guessed from
# transient in-memory runtime state.
PARENT_OPERATIONS: Final[Mapping[str, str]] = {
    "expand-context": "context-pack",
    "explain-impact": "run",
    "assurance": "verify",
    "seal": "assurance",
    "report": "seal",
}


@dataclass(frozen=True)
class EvidenceContext:
    """Explicit evidence-command context, kept separate from state commands."""

    command: str
    repository: Path
    policy: str
    task_id: str
    correlation_id: str
    output_mode: str
    run_id: str
    repository_id: str
    patch_id: str
    state_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.command not in COMMANDS:
            raise ProofContextError("unsupported evidence command", code="unknown_field")


@dataclass(frozen=True)
class EvidenceRequest:
    """One exact command invocation and its single immutable parent record."""

    context: EvidenceContext
    parent: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.context.command not in COMMANDS:
            raise ProofContextError("unsupported evidence command", code="unknown_field")
        if not isinstance(self.parent, Mapping):
            raise ProofContextError("immutable parent evidence is required", code="malformed")


def _canonical_head(repository: Path) -> str:
    """Read the exact Git head without accepting a best-effort substitute."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin", "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProofContextError(
            "repository head is unavailable", code="unavailable_capability"
        ) from exc
    head = completed.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ProofContextError("repository head is malformed", code="malformed")
    return head


def _parent_identities(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    identities = parent.get("identities")
    if not isinstance(identities, Mapping):
        raise ProofContextError("parent identities are required", code="malformed")
    return identities


def _admit_parent(request: EvidenceRequest) -> Mapping[str, Any]:
    """Verify operation, outcome, provenance, identity, and freshness."""

    parent = request.parent
    expected_operation = PARENT_OPERATIONS[request.context.command]
    if parent.get("operation") != expected_operation:
        raise ProofContextError("parent operation does not match command", code="identity_inconsistent")
    if parent.get("status") == "unavailable":
        raise ProofContextError(
            "unavailable parent evidence cannot be advanced", code="unavailable_capability"
        )
    if parent.get("status") != "succeeded":
        raise ProofContextError("failed parent evidence cannot be advanced", code="identity_inconsistent")
    if parent.get("provenance") != "live":
        raise ProofContextError("non-live evidence cannot be advanced", code="simulated_promoted")
    artifact_cid = parent.get("artifact_cid")
    if not isinstance(artifact_cid, str) or not artifact_cid.strip():
        raise ProofContextError("parent artifact identity is required", code="malformed")
    identities = _parent_identities(parent)
    expected = {
        "repository_id": request.context.repository_id,
        "task_id": request.context.task_id,
        "run_id": request.context.run_id,
        "trace_id": request.context.correlation_id,
        "patch_id": request.context.patch_id,
    }
    for field, context_value in expected.items():
        parent_value = identities.get(field)
        if not isinstance(parent_value, str) or not parent_value or parent_value != context_value:
            raise ProofContextError("parent identity does not match invocation", code="identity_inconsistent")
    payload = parent.get("payload")
    if not isinstance(payload, Mapping) or payload.get("canonical_head") != _canonical_head(request.context.repository):
        raise ProofContextError("parent evidence is stale for this repository head", code="stale_root")
    return parent


def _request(context: EvidenceContext, parent: Mapping[str, Any] | None) -> EvidenceRequest:
    # Evidence commands are intentionally stricter than read-only state
    # commands: they must name an existing run and repository identity.
    if not context.run_id or not context.repository_id or not context.patch_id:
        raise ProofContextError(
            "run-id, repository-id, and patch-id are required", code="malformed"
        )
    # Do not let an omitted parent fall through to operation comparison: absence
    # is malformed input, while a supplied but wrong parent is an identity error.
    # This distinction is intentional and, importantly, happens before opening
    # the runtime or inspecting the repository.
    if parent is None:
        raise ProofContextError("immutable parent evidence is required", code="malformed")
    return EvidenceRequest(context=context, parent=parent)


def _open_evidence_runtime(context: EvidenceContext):
    """Open the canonical runtime with the already-resolved patch identity."""

    from ipfs_accelerate_py.proof_context.bootstrap import RuntimeOptions, open_runtime

    state_dir = _reject_escape(context, _state_dir(context))
    identities = replace(_identities(context), patch_id=context.patch_id)
    return open_runtime(
        context.repository,
        identities=identities,
        mode=admit_mode(context.policy),
        options=RuntimeOptions(
            kit_root=state_dir,
            worktree_parent=state_dir / "worktrees",
            operator_id="proof-context-cli",
        ),
    )


def _with_lineage(context: EvidenceContext, record: EngineRecord, parent: Mapping[str, Any]) -> CliResult:
    result = _result_from_record(context, record)
    payload = dict(result.payload)
    payload.update(
        {
            "parent_artifact_cid": parent["artifact_cid"],
            "parent_operation": parent["operation"],
            "immutable_parent": True,
            "canonical_head": _canonical_head(context.repository),
        }
    )
    if context.command == "expand-context":
        payload["context_expansion"] = True
        payload["frontier_escalation_recorded"] = True
    return CliResult(
        command=context.command,
        status=result.status,
        policy=result.policy,
        correlation_id=result.correlation_id,
        output_mode=result.output_mode,
        payload=payload,
        identities=result.identities,
        artifact_cid=result.artifact_cid,
        provenance=result.provenance,
        contract=OPERATION_CONTRACTS.get(record.operation),
        error=result.error,
    )


def _run(
    context: EvidenceContext,
    parent: Mapping[str, Any] | None,
    operation: str,
    invoke: Callable[[Any], EngineRecord],
) -> CliResult:
    try:
        request = _request(context, parent)
        admitted_parent = _admit_parent(request)
        record = invoke(_open_evidence_runtime(context).engine)
        if record.status != "succeeded" or record.provenance != "live":
            raise ProofContextError("canonical operation did not emit live success", code="identity_inconsistent")
        return _with_lineage(context, record, admitted_parent)
    except ProofContextError as exc:
        return _failure_result(context, exc, status=str(exc.status), error=str(exc.code))
    except Exception as exc:  # noqa: BLE001 - CLI emits closed typed failures
        return _failure_result(context, exc, status="infrastructure_failure", error="infrastructure_failure")


def cmd_expand_context(context: EvidenceContext, *, parent: Mapping[str, Any] | None = None) -> CliResult:
    return _run(context, parent, "expand-context", lambda engine: engine.expand_context())


def cmd_assurance(context: EvidenceContext, *, parent: Mapping[str, Any] | None = None) -> CliResult:
    return _run(context, parent, "assurance", lambda engine: engine.assurance())


def cmd_seal(context: EvidenceContext, *, parent: Mapping[str, Any] | None = None) -> CliResult:
    return _run(context, parent, "seal", lambda engine: engine.seal())


def cmd_report(context: EvidenceContext, *, parent: Mapping[str, Any] | None = None) -> CliResult:
    return _run(context, parent, "report", lambda engine: engine.report())


def cmd_explain_impact(context: EvidenceContext, *, parent: Mapping[str, Any] | None = None) -> CliResult:
    """Emit an identity-bound impact artifact through the canonical semantic port."""

    try:
        request = _request(context, parent)
        admitted_parent = _admit_parent(request)
        bundle = _open_evidence_runtime(context)
        artifact = bundle.session.lifecycle_ports.semantic.impact(
            bundle.session.lifecycle_identities, context.repository
        )
        record = bundle.session.record_from_artifact("plan", artifact)
        if record.status != "succeeded" or record.provenance != "live":
            raise ProofContextError("canonical impact operation did not emit live success", code="identity_inconsistent")
        result = _with_lineage(context, record, admitted_parent)
        return CliResult(
            command="explain-impact", status=result.status, policy=result.policy,
            correlation_id=result.correlation_id, output_mode=result.output_mode,
            payload={**dict(result.payload), "impact_explanation": "canonical", "impact_artifact": True},
            identities=result.identities, artifact_cid=result.artifact_cid,
            provenance=result.provenance,
            contract="pcce/proof-context/v0.1/repository-state", error=result.error,
        )
    except ProofContextError as exc:
        return _failure_result(context, exc, status=str(exc.status), error=str(exc.code))
    except Exception as exc:  # noqa: BLE001
        return _failure_result(context, exc, status="infrastructure_failure", error=redact_text(str(exc)) or "infrastructure_failure")


__all__ = [
    "COMMANDS", "PARENT_OPERATIONS", "EvidenceContext", "EvidenceRequest", "cmd_assurance",
    "cmd_explain_impact", "cmd_expand_context", "cmd_report", "cmd_seal",
]
