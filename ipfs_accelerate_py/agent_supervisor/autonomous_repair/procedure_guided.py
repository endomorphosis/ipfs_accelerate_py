"""Fail-closed bridge from promoted procedures to isolated repair integration.

The generic repair components deliberately do not know whether a procedure is
current or whether its risk permits an autonomous merge.  This module makes
that decision explicit.  It never discovers a worktree, produces source
bytes, weakens validation, or grants completion authority.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..procedure_compiler.contracts import RiskClass
from ..procedure_compiler.registry import ProcedureRegistry, RegistryLifecycleState
from .contracts import (
    PostEditValidationReceipt,
    PublicationReceipt,
    RepairAdmissionReceipt,
    ReproofReceipt,
)

PROCEDURE_GUIDED_REPAIR_INTERFACE: Final[str] = "ProcedureGuidedRepairAdapter@1"
AUTONOMOUS_MERGE_CEILING_INTERFACE: Final[str] = "AutonomousMergeCeiling@1"

_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}


class ProcedureRepairDisposition(str, Enum):
    """Closed outcomes; only ``MERGED`` reports an observed merge."""

    MERGE_READY = "merge_ready"
    MERGED = "merged"
    PR_REQUIRED = "pr_required"
    REVIEW_REQUIRED = "review_required"
    ESCALATED = "escalated"


class ProcedureGuidedRepairError(ValueError):
    """A procedure-guided repair request is malformed or unsafe."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProcedureGuidedRepairError(f"{field} must be non-empty canonical text")
    if any(char.isspace() for char in value):
        raise ProcedureGuidedRepairError(f"{field} must not contain whitespace")
    return value


@dataclass(frozen=True)
class AutonomousMergeCeiling:
    """Every independent ceiling that must hold before invoking a merger."""

    max_risk: RiskClass = RiskClass.REVERSIBLE_LOCAL
    max_patch_bytes: int = 262_144
    allowed_paths: tuple[str, ...] = ()
    require_isolated_worktree: bool = True
    require_symlink_free: bool = True
    require_submodule_free: bool = True
    require_post_merge_tree: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_risk, RiskClass):
            try:
                object.__setattr__(self, "max_risk", RiskClass(self.max_risk))
            except (TypeError, ValueError) as exc:
                raise ProcedureGuidedRepairError("max_risk is unsupported") from exc
        if type(self.max_patch_bytes) is not int or not 0 < self.max_patch_bytes <= 8 * 1024 * 1024:
            raise ProcedureGuidedRepairError("max_patch_bytes must be a bounded positive integer")
        normalized = tuple(path.replace("\\", "/") for path in self.allowed_paths)
        if any(not path or path.startswith("/") or ".." in path.split("/") for path in normalized):
            raise ProcedureGuidedRepairError("allowed_paths contains an unsafe path")
        object.__setattr__(self, "allowed_paths", tuple(sorted(set(normalized))))
        for name in (
            "require_isolated_worktree", "require_symlink_free", "require_submodule_free",
            "require_post_merge_tree",
        ):
            if type(getattr(self, name)) is not bool:
                raise ProcedureGuidedRepairError(f"{name} must be boolean")

    def permits_risk(self, value: RiskClass) -> bool:
        return _RISK_RANK[value] <= _RISK_RANK[self.max_risk]


@dataclass(frozen=True)
class ProcedureGuidedRepairRequest:
    """Evidence supplied by the isolated repair pipeline, not an authority grant."""

    repair_id: str
    procedure_id: str
    expected_revision_id: str
    patch_digest: str
    patch_bytes: int
    changed_paths: tuple[str, ...]
    lease_id: str
    merge_authorization_cid: str
    admission: RepairAdmissionReceipt | None = None
    validation: PostEditValidationReceipt | None = None
    reproof: ReproofReceipt | None = None
    publication: PublicationReceipt | None = None
    isolated_worktree: bool = False
    symlink_free: bool = False
    submodule_free: bool = False
    patch_changes_bytes: bool = False
    tests_preserved: bool = False
    tests_passed: bool = False
    proofs_passed: bool = False
    post_merge_tree_id: str = ""

    def __post_init__(self) -> None:
        for value, field in (
            (self.repair_id, "repair_id"), (self.procedure_id, "procedure_id"),
            (self.expected_revision_id, "expected_revision_id"), (self.patch_digest, "patch_digest"),
            (self.lease_id, "lease_id"), (self.merge_authorization_cid, "merge_authorization_cid"),
        ):
            _text(value, field)
        if type(self.patch_bytes) is not int or self.patch_bytes <= 0:
            raise ProcedureGuidedRepairError("patch_bytes must be positive; no-op patches are refused")
        if not self.changed_paths or len(set(self.changed_paths)) != len(self.changed_paths):
            raise ProcedureGuidedRepairError("changed_paths must be non-empty and unique")
        normalized = tuple(path.replace("\\", "/") for path in self.changed_paths)
        if any(not path or path.startswith("/") or ".." in path.split("/") for path in normalized):
            raise ProcedureGuidedRepairError("changed_paths contains an unsafe path")
        object.__setattr__(self, "changed_paths", tuple(sorted(normalized)))
        for name in ("isolated_worktree", "symlink_free", "submodule_free"):
            if type(getattr(self, name)) is not bool:
                raise ProcedureGuidedRepairError(f"{name} must be boolean")
        for name in ("patch_changes_bytes", "tests_preserved", "tests_passed", "proofs_passed"):
            if type(getattr(self, name)) is not bool:
                raise ProcedureGuidedRepairError(f"{name} must be boolean")


@dataclass(frozen=True)
class ProcedureGuidedRepairResult:
    disposition: ProcedureRepairDisposition
    reason_codes: tuple[str, ...]
    procedure_revision_id: str = ""
    merge_invoked: bool = False
    merged: bool = False
    completion_authoritative: bool = False

    @property
    def merge_permitted(self) -> bool:
        return self.disposition is ProcedureRepairDisposition.MERGE_READY


class ProcedureGuidedRepairAdapter:
    """Evaluate procedure/currentness and every merge ceiling before integration."""

    revision: Final[str] = PROCEDURE_GUIDED_REPAIR_INTERFACE

    def __init__(self, registry: ProcedureRegistry, *, ceiling: AutonomousMergeCeiling | None = None) -> None:
        if not isinstance(registry, ProcedureRegistry):
            raise ProcedureGuidedRepairError("registry must be a ProcedureRegistry")
        self._registry = registry
        self._ceiling = ceiling or AutonomousMergeCeiling()

    def evaluate(self, request: ProcedureGuidedRepairRequest) -> ProcedureGuidedRepairResult:
        if not isinstance(request, ProcedureGuidedRepairRequest):
            raise ProcedureGuidedRepairError("request must be a ProcedureGuidedRepairRequest")
        try:
            revision = self._registry.get(request.procedure_id, demote_stale=False)
        except Exception:  # registry error is intentionally a non-completion escalation
            return self._result(ProcedureRepairDisposition.ESCALATED, "procedure_unavailable")
        if revision.revision_id != request.expected_revision_id:
            return self._result(ProcedureRepairDisposition.REVIEW_REQUIRED, "procedure_revision_stale", revision_id=revision.revision_id)
        if revision.state is not RegistryLifecycleState.PROMOTED:
            return self._result(ProcedureRepairDisposition.REVIEW_REQUIRED, "procedure_not_promoted", revision_id=revision.revision_id)
        if not self._ceiling.permits_risk(revision.risk_ceiling):
            return self._result(ProcedureRepairDisposition.ESCALATED, "risk_ceiling_exceeded", revision_id=revision.revision_id)
        reasons = self._evidence_failures(request)
        if reasons:
            return self._result(ProcedureRepairDisposition.PR_REQUIRED, *reasons, revision_id=revision.revision_id)
        boundary = self._boundary_failures(request)
        if boundary:
            return self._result(ProcedureRepairDisposition.REVIEW_REQUIRED, *boundary, revision_id=revision.revision_id)
        return self._result(ProcedureRepairDisposition.MERGE_READY, "all_merge_ceilings_satisfied", revision_id=revision.revision_id)

    def merge(
        self,
        request: ProcedureGuidedRepairRequest,
        *,
        merge_executor: Callable[[ProcedureGuidedRepairRequest], str],
    ) -> ProcedureGuidedRepairResult:
        """Invoke a supplied merger exactly once after successful evaluation.

        The merger must return the independently observed post-merge tree id.
        This adapter intentionally does not run Git itself or claim completion.
        """
        decision = self.evaluate(request)
        if not decision.merge_permitted:
            return decision
        if not callable(merge_executor):
            raise ProcedureGuidedRepairError("merge_executor must be callable")
        observed_tree = merge_executor(request)
        if not isinstance(observed_tree, str) or not observed_tree.strip():
            return self._result(ProcedureRepairDisposition.REVIEW_REQUIRED, "post_merge_tree_missing", revision_id=decision.procedure_revision_id, invoked=True)
        if self._ceiling.require_post_merge_tree and observed_tree != request.post_merge_tree_id:
            return self._result(ProcedureRepairDisposition.REVIEW_REQUIRED, "post_merge_tree_mismatch", revision_id=decision.procedure_revision_id, invoked=True)
        return ProcedureGuidedRepairResult(
            disposition=ProcedureRepairDisposition.MERGED,
            reason_codes=("merge_observed_under_all_ceilings",),
            procedure_revision_id=decision.procedure_revision_id,
            merge_invoked=True,
            merged=True,
            completion_authoritative=False,
        )

    def _evidence_failures(self, request: ProcedureGuidedRepairRequest) -> list[str]:
        admission, validation, reproof, publication = request.admission, request.validation, request.reproof, request.publication
        if not all((admission, validation, reproof, publication)):
            return ["admission_validation_reproof_publication_required"]
        assert admission is not None and validation is not None and reproof is not None and publication is not None
        if any(item.repair_id != request.repair_id for item in (admission, validation, reproof, publication)):
            return ["repair_evidence_id_mismatch"]
        if not (validation.passed and reproof.proved and publication.published):
            return ["validation_or_proof_not_passing"]
        if not request.patch_changes_bytes:
            return ["no_op_patch_refused"]
        if not request.tests_preserved:
            return ["test_deletion_or_weakening_not_admitted"]
        if not request.tests_passed:
            return ["test_evidence_missing_or_failing"]
        if not request.proofs_passed:
            return ["proof_evidence_missing_or_failing"]
        if not (validation.authority_roots == admission.authority_roots == reproof.authority_roots == publication.authority_roots):
            return ["repair_authority_roots_mismatch"]
        if validation.admission_receipt_cid != admission.content_id or reproof.admission_receipt_cid != admission.content_id:
            return ["admission_chain_mismatch"]
        if reproof.post_edit_validation_receipt_cid != validation.content_id or publication.post_edit_validation_receipt_cid != validation.content_id:
            return ["validation_chain_mismatch"]
        if publication.reproof_receipt_cid != reproof.content_id:
            return ["reproof_chain_mismatch"]
        if len({validation.mutation_receipt_cid, reproof.mutation_receipt_cid, publication.mutation_receipt_cid}) != 1:
            return ["mutation_chain_mismatch"]
        return []

    def _boundary_failures(self, request: ProcedureGuidedRepairRequest) -> list[str]:
        failures: list[str] = []
        if request.patch_bytes > self._ceiling.max_patch_bytes:
            failures.append("patch_byte_ceiling_exceeded")
        if self._ceiling.allowed_paths and not all(
            any(path == allowed or path.startswith(allowed + "/") for allowed in self._ceiling.allowed_paths)
            for path in request.changed_paths
        ):
            failures.append("patch_scope_exceeded")
        if self._ceiling.require_isolated_worktree and not request.isolated_worktree:
            failures.append("isolated_worktree_required")
        if self._ceiling.require_symlink_free and not request.symlink_free:
            failures.append("symlink_escape_risk")
        if self._ceiling.require_submodule_free and not request.submodule_free:
            failures.append("submodule_escape_risk")
        if self._ceiling.require_post_merge_tree and not request.post_merge_tree_id:
            failures.append("post_merge_tree_required")
        return failures

    @staticmethod
    def _result(
        disposition: ProcedureRepairDisposition,
        *reasons: str,
        revision_id: str = "",
        invoked: bool = False,
    ) -> ProcedureGuidedRepairResult:
        return ProcedureGuidedRepairResult(disposition, tuple(sorted(set(reasons))), revision_id, invoked, False, False)


__all__ = [
    "AUTONOMOUS_MERGE_CEILING_INTERFACE",
    "PROCEDURE_GUIDED_REPAIR_INTERFACE",
    "AutonomousMergeCeiling",
    "ProcedureGuidedRepairAdapter",
    "ProcedureGuidedRepairError",
    "ProcedureGuidedRepairRequest",
    "ProcedureGuidedRepairResult",
    "ProcedureRepairDisposition",
]
