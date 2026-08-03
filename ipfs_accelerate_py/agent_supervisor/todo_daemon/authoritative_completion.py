"""Fail-closed contracts for implementation acceptance.

An implementation commit, merge-queue status, or Git ancestry relationship is
evidence that code landed.  None of those facts authorizes a task-board
completion.  This module keeps the acceptance contract small enough to audit
independently from the implementation daemon and recomputes every gate from
commit-bound evidence.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..task_sources.taskboard_store import (
    locked_taskboard,
    replace_locked_taskboard,
)
from .post_merge_validation import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    verify_post_merge_validation_evidence,
)
from .status import (
    build_merged_pending_acceptance_status,
    build_reopened_acceptance_status,
    project_authoritative_acceptance_status,
)

__all__ = [
    name
    for name in (
        "ACCEPTANCE_REOPENED_STALE_EVENT ACCEPTANCE_STATE_AUTHORITATIVE "
        "ACCEPTANCE_STATE_MERGED_PENDING ACCEPTANCE_STATE_REOPENED "
        "AUTHORITATIVE_COMPLETION_ADMITTED_EVENT "
        "AUTHORITATIVE_COMPLETION_DENIED_EVENT "
        "AUTHORITATIVE_COMPLETION_GATE_KINDS "
        "AUTHORITATIVE_COMPLETION_GATE_SCHEMA "
        "DETERMINISTIC_ONLY_POLICY_SCHEMA "
        "DETERMINISTIC_ONLY_MODEL_REJECTED_EVENT "
        "IMPLEMENTATION_MERGED_PENDING_EVENT IMPLEMENTATION_RECEIPT_SCHEMA "
        "AUTHORITATIVE_GATE_EVIDENCE_SCHEMA "
        "POST_MERGE_VALIDATION_EVIDENCE_SCHEMA AuthoritativeCompletionGate "
        "AuthoritativeCompletionMixin DeterministicOnlyPolicy "
        "ImplementationReceipt authorize_completion_mutation "
        "bound_gate_evidence build_implementation_receipt "
        "evaluate_authoritative_completion_gate "
        "promote_authoritative_completion "
        "reopen_acceptance_for_stale_post_merge_validation"
    ).split()
]

IMPLEMENTATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-receipt@1"
)
AUTHORITATIVE_COMPLETION_GATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-completion-gate@1"
)
AUTHORITATIVE_GATE_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-gate-evidence@1"
)
DETERMINISTIC_ONLY_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-only-policy@1"
)

AUTHORITATIVE_COMPLETION_GATE_KINDS: tuple[str, ...] = (
    "merge",
    "freshness",
    "semantic",
    "proof",
    "provider_review",
    "deterministic_only",
)

ACCEPTANCE_STATE_MERGED_PENDING = "implemented_merged_but_pending"
ACCEPTANCE_STATE_AUTHORITATIVE = "authoritatively_completed"
ACCEPTANCE_STATE_REOPENED = "acceptance_reopened"
IMPLEMENTATION_MERGED_PENDING_EVENT = "implementation_merged_pending_acceptance"
AUTHORITATIVE_COMPLETION_ADMITTED_EVENT = "authoritative_task_completion_admitted"
AUTHORITATIVE_COMPLETION_DENIED_EVENT = "authoritative_task_completion_denied"
ACCEPTANCE_REOPENED_STALE_EVENT = "acceptance_reopened_stale_post_merge_validation"
DETERMINISTIC_ONLY_MODEL_REJECTED_EVENT = "deterministic_only_model_invocation_rejected"


def _strings(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not values:
        return ()
    return tuple(str(item) for item in values if str(item))


@dataclass(frozen=True)
class ImplementationReceipt:
    """Content-bound evidence for an implementation; not authority by itself."""

    task_id: str
    implementation_commit: str = ""
    merge_commit: str = ""
    repository_tree_id: str = ""
    merged: bool = False
    validation_passed: bool = False
    validation_stale: bool = False
    completion_authoritative: bool = False
    pending_gates: tuple[str, ...] = ()
    gate_evidence: Mapping[str, Any] = field(default_factory=dict)
    model_invocation_observed: bool = False
    deterministic_only: bool = False
    acceptance_state: str = ACCEPTANCE_STATE_MERGED_PENDING
    reason_codes: tuple[str, ...] = ()
    schema: str = IMPLEMENTATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "implementation_commit": self.implementation_commit,
            "merge_commit": self.merge_commit,
            "repository_tree_id": self.repository_tree_id,
            "merged": bool(self.merged),
            "validation_passed": bool(self.validation_passed),
            "validation_stale": bool(self.validation_stale),
            "completion_authoritative": bool(self.completion_authoritative),
            "pending_gates": list(self.pending_gates),
            "gate_evidence": dict(self.gate_evidence),
            "model_invocation_observed": bool(self.model_invocation_observed),
            "deterministic_only": bool(self.deterministic_only),
            "acceptance_state": self.acceptance_state,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("implementation receipt must be a mapping")
        if payload.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
            raise ValueError("implementation receipt schema is missing or unsupported")
        evidence = payload.get("gate_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("implementation receipt gate_evidence must be a mapping")
        return cls(
            task_id=str(payload.get("task_id") or ""),
            implementation_commit=str(payload.get("implementation_commit") or ""),
            merge_commit=str(payload.get("merge_commit") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            merged=bool(payload.get("merged", False)),
            validation_passed=bool(payload.get("validation_passed", False)),
            validation_stale=bool(payload.get("validation_stale", False)),
            completion_authoritative=bool(
                payload.get("completion_authoritative", False)
            ),
            pending_gates=_strings(payload.get("pending_gates")),
            gate_evidence=dict(evidence),
            model_invocation_observed=bool(
                payload.get("model_invocation_observed", False)
            ),
            deterministic_only=bool(payload.get("deterministic_only", False)),
            acceptance_state=str(
                payload.get("acceptance_state")
                or ACCEPTANCE_STATE_MERGED_PENDING
            ),
            reason_codes=_strings(payload.get("reason_codes")),
            schema=str(payload["schema"]),
        )


@dataclass(frozen=True)
class AuthoritativeCompletionGate:
    """Recomputed decision authorizing one exact task/commit/tree mutation."""

    admitted: bool
    task_id: str = ""
    completion_authoritative: bool = False
    pending_gates: tuple[str, ...] = ()
    satisfied_gates: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    acceptance_state: str = ACCEPTANCE_STATE_MERGED_PENDING
    implementation_commit: str = ""
    merge_commit: str = ""
    repository_tree_id: str = ""
    schema: str = AUTHORITATIVE_COMPLETION_GATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "admitted": bool(self.admitted),
            "task_id": self.task_id,
            "completion_authoritative": bool(self.completion_authoritative),
            "pending_gates": list(self.pending_gates),
            "satisfied_gates": list(self.satisfied_gates),
            "reason_codes": list(self.reason_codes),
            "acceptance_state": self.acceptance_state,
            "implementation_commit": self.implementation_commit,
            "merge_commit": self.merge_commit,
            "repository_tree_id": self.repository_tree_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AuthoritativeCompletionGate":
        if not isinstance(payload, Mapping):
            raise TypeError("authoritative completion gate must be a mapping")
        if payload.get("schema") != AUTHORITATIVE_COMPLETION_GATE_SCHEMA:
            raise ValueError(
                "authoritative completion gate schema is missing or unsupported"
            )
        return cls(
            admitted=bool(payload.get("admitted", False)),
            task_id=str(payload.get("task_id") or ""),
            completion_authoritative=bool(
                payload.get("completion_authoritative", False)
            ),
            pending_gates=_strings(payload.get("pending_gates")),
            satisfied_gates=_strings(payload.get("satisfied_gates")),
            reason_codes=_strings(payload.get("reason_codes")),
            acceptance_state=str(
                payload.get("acceptance_state")
                or ACCEPTANCE_STATE_MERGED_PENDING
            ),
            implementation_commit=str(payload.get("implementation_commit") or ""),
            merge_commit=str(payload.get("merge_commit") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            schema=str(payload["schema"]),
        )


@dataclass(frozen=True)
class DeterministicOnlyPolicy:
    """Reject provider/model dispatch for deterministic-only tasks."""

    task_id: str = ""
    deterministic_only: bool = False
    schema: str = DETERMINISTIC_ONLY_POLICY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "deterministic_only": bool(self.deterministic_only),
        }

    def allows_model_invocation(self) -> bool:
        return not self.deterministic_only

    def reject_model_invocation(
        self,
        *,
        provider: str = "",
        reason: str = "model_invocation",
    ) -> dict[str, Any]:
        if self.allows_model_invocation():
            return {
                "rejected": False,
                "task_id": self.task_id,
                "deterministic_only": False,
                "provider": provider,
                "reason": "not_deterministic_only",
            }
        return {
            "rejected": True,
            "task_id": self.task_id,
            "deterministic_only": True,
            "provider": str(provider or ""),
            "reason": "deterministic_only_forbids_model_invocation",
            "attempted_reason": str(reason or "model_invocation"),
            "completion_authoritative": False,
        }

    @classmethod
    def for_task(cls, task: Any) -> "DeterministicOnlyPolicy":
        if task is None:
            return cls()
        metadata = {
            str(key).strip().lower().replace("_", " "): str(value or "").strip()
            for key, value in (getattr(task, "metadata", {}) or {}).items()
        }
        role = (
            metadata.get("provider role")
            or metadata.get("implementation provider")
            or metadata.get("execution mode")
            or ""
        ).lower()
        return cls(
            task_id=str(getattr(task, "task_id", "") or ""),
            deterministic_only=role
            in {"deterministic-only", "deterministic only", "deterministic", "operator-only"},
        )


def bound_gate_evidence(
    gate_kind: str,
    *,
    task_id: str,
    implementation_commit: str,
    merge_commit: str,
    repository_tree_id: str,
    **details: Any,
) -> dict[str, Any]:
    """Build one explicit binding envelope; semantic claims remain caller-owned."""

    return {
        "schema": AUTHORITATIVE_GATE_EVIDENCE_SCHEMA,
        "gate_kind": gate_kind,
        "task_id": task_id,
        "implementation_commit": implementation_commit,
        "target_commit": merge_commit,
        "merge_commit": merge_commit,
        "repository_tree_id": repository_tree_id,
        **details,
    }


def _bound_entry_reason(
    kind: str,
    entry: Any,
    receipt: ImplementationReceipt,
) -> str:
    if not isinstance(entry, Mapping):
        return "evidence_missing"
    expected = {
        "schema": AUTHORITATIVE_GATE_EVIDENCE_SCHEMA,
        "gate_kind": kind,
        "task_id": receipt.task_id,
        "implementation_commit": receipt.implementation_commit,
        "target_commit": receipt.merge_commit,
        "repository_tree_id": receipt.repository_tree_id,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            return f"binding_mismatch:{key}"
    if kind == "merge":
        if entry.get("merge_commit") != receipt.merge_commit:
            return "binding_mismatch:merge_commit"
        if entry.get("satisfied") is not True or not receipt.merged:
            return "merge_not_verified"
        return ""
    if kind in {"freshness", "semantic"}:
        if entry.get("validation_scope") != "post_merge":
            return "post_merge_validation_required"
        if entry.get("passed") is not True or entry.get("satisfied") is not True:
            return f"{kind}_validation_failed"
        if entry.get("validation_receipt_id") in (None, ""):
            return "validation_receipt_id_missing"
        if kind == "freshness" and entry.get("stale") is not False:
            return "validation_stale"
        return ""
    if kind == "proof":
        if entry.get("not_applicable") is True:
            return (
                ""
                if entry.get("applicability_decision")
                == "no_declared_proof_obligation"
                else "proof_not_applicable_unbound"
            )
        if (
            entry.get("satisfied") is True
            and entry.get("proof_verified") is True
            and entry.get("proof_receipt_id") not in (None, "")
        ):
            return ""
        return "proof_not_verified"
    if kind == "provider_review":
        if entry.get("not_applicable") is True:
            if receipt.model_invocation_observed:
                return "model_assisted_provider_review_not_applicable_forbidden"
            if entry.get("route_kind") not in {
                "deterministic_only",
                "no_model_provider_declared",
            }:
                return "provider_review_not_applicable_unbound"
            if entry.get("model_invocation_observed") is not False:
                return "provider_model_observation_unbound"
            return "" if entry.get("satisfied") is True else "provider_review_unsatisfied"
        if (
            entry.get("satisfied") is True
            and entry.get("review_presence") == "independent"
            and entry.get("provider_result_admitted") is True
            and entry.get("review_receipt_id") not in (None, "")
        ):
            return ""
        return "independent_provider_review_missing"
    if kind == "deterministic_only":
        if entry.get("model_invocation_observed") is not receipt.model_invocation_observed:
            return "model_invocation_observation_mismatch"
        if receipt.deterministic_only:
            if entry.get("policy") != "deterministic_only":
                return "deterministic_policy_binding_missing"
            if receipt.model_invocation_observed or entry.get("satisfied") is not True:
                return "deterministic_only_model_invocation"
            return ""
        if (
            entry.get("not_applicable") is True
            and entry.get("policy") == "not_deterministic_only"
            and entry.get("satisfied") is True
        ):
            return ""
        return "deterministic_policy_binding_missing"
    return "unknown_gate_kind"


def _invalid_gate(
    reason: str,
    *,
    task_id: str = "",
    receipt: ImplementationReceipt | None = None,
) -> AuthoritativeCompletionGate:
    return AuthoritativeCompletionGate(
        admitted=False,
        task_id=task_id or (receipt.task_id if receipt else ""),
        completion_authoritative=False,
        pending_gates=AUTHORITATIVE_COMPLETION_GATE_KINDS,
        reason_codes=(reason,),
        implementation_commit=receipt.implementation_commit if receipt else "",
        merge_commit=receipt.merge_commit if receipt else "",
        repository_tree_id=receipt.repository_tree_id if receipt else "",
    )


def evaluate_authoritative_completion_gate(
    receipt: ImplementationReceipt | Mapping[str, Any] | None,
    *,
    require_completion_authoritative_flag: bool = True,
    expected_task_id: str = "",
) -> AuthoritativeCompletionGate:
    """Recompute all gates from bound evidence; never trust cached gate lists."""

    if receipt is None:
        return _invalid_gate("implementation_receipt_missing", task_id=expected_task_id)
    if isinstance(receipt, Mapping):
        try:
            receipt = ImplementationReceipt.from_dict(receipt)
        except (TypeError, ValueError):
            return _invalid_gate("implementation_receipt_invalid", task_id=expected_task_id)
    if receipt.schema != IMPLEMENTATION_RECEIPT_SCHEMA:
        return _invalid_gate("implementation_receipt_schema_invalid", receipt=receipt)
    if not receipt.task_id:
        return _invalid_gate("implementation_receipt_task_missing", receipt=receipt)
    if expected_task_id and receipt.task_id != expected_task_id:
        return _invalid_gate("implementation_receipt_task_mismatch", receipt=receipt)
    if not receipt.implementation_commit:
        return _invalid_gate("implementation_commit_missing", receipt=receipt)
    if not receipt.merge_commit:
        return _invalid_gate("merge_commit_missing", receipt=receipt)
    if not receipt.repository_tree_id:
        return _invalid_gate("repository_tree_id_missing", receipt=receipt)

    pending: list[str] = []
    reasons: list[str] = []
    for kind in AUTHORITATIVE_COMPLETION_GATE_KINDS:
        reason = _bound_entry_reason(kind, receipt.gate_evidence.get(kind), receipt)
        if reason:
            pending.append(kind)
            reasons.append(f"{kind}:{reason}")
    if receipt.validation_stale:
        if "freshness" not in pending:
            pending.append("freshness")
        reasons.append("validation_stale")
    if receipt.deterministic_only and receipt.model_invocation_observed:
        if "deterministic_only" not in pending:
            pending.append("deterministic_only")
        reasons.append("deterministic_only_model_invocation")
    if require_completion_authoritative_flag and not receipt.completion_authoritative:
        reasons.append("completion_authoritative_false")
    pending_set = set(pending)
    pending_tuple = tuple(
        kind for kind in AUTHORITATIVE_COMPLETION_GATE_KINDS if kind in pending_set
    )
    admitted = not pending_tuple and not (
        require_completion_authoritative_flag
        and not receipt.completion_authoritative
    )
    return AuthoritativeCompletionGate(
        admitted=admitted,
        task_id=receipt.task_id,
        completion_authoritative=admitted,
        pending_gates=pending_tuple,
        satisfied_gates=tuple(
            kind
            for kind in AUTHORITATIVE_COMPLETION_GATE_KINDS
            if kind not in pending_set
        ),
        reason_codes=() if admitted else tuple(dict.fromkeys(reasons)),
        acceptance_state=(
            ACCEPTANCE_STATE_AUTHORITATIVE
            if admitted
            else ACCEPTANCE_STATE_MERGED_PENDING
        ),
        implementation_commit=receipt.implementation_commit,
        merge_commit=receipt.merge_commit,
        repository_tree_id=receipt.repository_tree_id,
    )


def build_implementation_receipt(
    *,
    task_id: str,
    implementation_commit: str = "",
    merge_commit: str = "",
    repository_tree_id: str = "",
    merged: bool = False,
    validation_passed: bool = False,
    validation_stale: bool = False,
    gate_evidence: Mapping[str, Any] | None = None,
    model_invocation_observed: bool = False,
    deterministic_only: bool = False,
    completion_authoritative: bool = False,
) -> ImplementationReceipt:
    """Build a non-authoritative receipt and derive pending gates from evidence."""

    del completion_authoritative  # Callers cannot self-assert acceptance authority.
    provisional = ImplementationReceipt(
        task_id=str(task_id or ""),
        implementation_commit=str(implementation_commit or ""),
        merge_commit=str(merge_commit or ""),
        repository_tree_id=str(repository_tree_id or ""),
        merged=bool(merged),
        validation_passed=bool(validation_passed),
        validation_stale=bool(validation_stale),
        gate_evidence=dict(gate_evidence or {}),
        model_invocation_observed=bool(model_invocation_observed),
        deterministic_only=bool(deterministic_only),
        acceptance_state=(
            ACCEPTANCE_STATE_MERGED_PENDING
            if merged
            else "implementation_incomplete"
        ),
    )
    structural = evaluate_authoritative_completion_gate(
        provisional,
        require_completion_authoritative_flag=False,
    )
    return replace(
        provisional,
        pending_gates=structural.pending_gates,
        reason_codes=tuple(
            dict.fromkeys(
                [*structural.reason_codes, "completion_authoritative_false"]
            )
        ),
    )


def promote_authoritative_completion(
    receipt: ImplementationReceipt | Mapping[str, Any],
    *,
    expected_task_id: str = "",
) -> tuple[ImplementationReceipt, AuthoritativeCompletionGate]:
    """Promote only after a fresh recomputation of every evidence gate."""

    if isinstance(receipt, Mapping):
        try:
            base = ImplementationReceipt.from_dict(receipt)
        except (TypeError, ValueError):
            invalid = ImplementationReceipt(task_id=expected_task_id)
            return invalid, _invalid_gate(
                "implementation_receipt_invalid",
                task_id=expected_task_id,
            )
    else:
        base = receipt
    structural = evaluate_authoritative_completion_gate(
        base,
        require_completion_authoritative_flag=False,
        expected_task_id=expected_task_id,
    )
    if not structural.admitted:
        return replace(
            base,
            completion_authoritative=False,
            pending_gates=structural.pending_gates,
            reason_codes=structural.reason_codes,
            acceptance_state=ACCEPTANCE_STATE_MERGED_PENDING,
        ), structural
    promoted = replace(
        base,
        completion_authoritative=True,
        pending_gates=(),
        reason_codes=(),
        acceptance_state=ACCEPTANCE_STATE_AUTHORITATIVE,
    )
    return promoted, evaluate_authoritative_completion_gate(
        promoted,
        expected_task_id=expected_task_id,
    )


def reopen_acceptance_for_stale_post_merge_validation(
    receipt: ImplementationReceipt | Mapping[str, Any],
    *,
    stale_reason: str = "post_merge_validation_stale",
) -> ImplementationReceipt:
    """Invalidate acceptance while retaining exact implementation bindings."""

    base = (
        ImplementationReceipt.from_dict(receipt)
        if isinstance(receipt, Mapping)
        else receipt
    )
    evidence = dict(base.gate_evidence)
    freshness = dict(evidence.get("freshness") or {})
    freshness.update({"satisfied": False, "passed": False, "stale": True})
    evidence["freshness"] = freshness
    return replace(
        base,
        validation_passed=False,
        validation_stale=True,
        completion_authoritative=False,
        pending_gates=tuple(
            dict.fromkeys([*base.pending_gates, "freshness"])
        ),
        gate_evidence=evidence,
        acceptance_state=ACCEPTANCE_STATE_REOPENED,
        reason_codes=tuple(
            dict.fromkeys(
                [*base.reason_codes, stale_reason, "acceptance_reopened"]
            )
        ),
    )


def authorize_completion_mutation(
    task_id: str,
    receipt: ImplementationReceipt | Mapping[str, Any] | None,
    gate: AuthoritativeCompletionGate | Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Authorize a board mutation only for the recomputed, exact gate packet."""

    if receipt is None or gate is None:
        return False, "authoritative_completion_packet_missing"
    try:
        receipt_obj = (
            ImplementationReceipt.from_dict(receipt)
            if isinstance(receipt, Mapping)
            else receipt
        )
        gate_obj = (
            AuthoritativeCompletionGate.from_dict(gate)
            if isinstance(gate, Mapping)
            else gate
        )
    except (TypeError, ValueError):
        return False, "authoritative_completion_packet_invalid"
    recomputed = evaluate_authoritative_completion_gate(
        receipt_obj,
        expected_task_id=task_id,
    )
    if not recomputed.admitted or not receipt_obj.completion_authoritative:
        return False, "authoritative_completion_gate_denied"
    if gate_obj.to_dict() != recomputed.to_dict():
        return False, "authoritative_completion_gate_mismatch"
    return True, "authoritative_completion_gate_admitted"


class AuthoritativeCompletionMixin:
    """Thin daemon integration for the pure acceptance contracts above."""

    def _authorize_completion_board_mutation(
        self,
        task_id: str,
        receipt: ImplementationReceipt | Mapping[str, Any] | None,
        gate: AuthoritativeCompletionGate | Mapping[str, Any] | None,
    ) -> tuple[bool, str]:
        """Bind an admitted packet to real objects in this repository."""

        admitted, reason = authorize_completion_mutation(task_id, receipt, gate)
        if not admitted:
            return False, reason
        try:
            receipt_obj = (
                ImplementationReceipt.from_dict(receipt)
                if isinstance(receipt, Mapping)
                else receipt
            )
            if receipt_obj is None:
                raise ValueError("implementation receipt is missing")
            resolved_commit, resolved_tree, exact = (
                self._verified_acceptance_binding(
                    receipt_obj.implementation_commit,
                    receipt_obj.merge_commit,
                    receipt_obj.repository_tree_id,
                )
            )
        except (OSError, TypeError, ValueError):
            return False, "authoritative_completion_git_binding_invalid"
        if (
            not exact
            or resolved_commit != receipt_obj.merge_commit
            or resolved_tree != receipt_obj.repository_tree_id
        ):
            return False, "authoritative_completion_git_binding_invalid"
        return True, reason

    def _mark_task_completed_in_todo(
        self,
        task_id: str,
        *,
        authoritative_receipt: ImplementationReceipt | Mapping[str, Any] | None = None,
        authoritative_gate: AuthoritativeCompletionGate | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._mark_tasks_completed_in_todo(
            [task_id],
            primary_task_id=task_id,
            completion_reason="single_task",
            authoritative_receipt=authoritative_receipt,
            authoritative_gate=authoritative_gate,
        )

    def _mark_task_or_bundle_completed_in_todo(
        self,
        task: Any,
        *,
        authoritative_receipt: ImplementationReceipt | Mapping[str, Any] | None = None,
        authoritative_gate: AuthoritativeCompletionGate | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._mark_tasks_completed_in_todo(
            [task.task_id],
            primary_task_id=task.task_id,
            completion_reason="authoritative_acceptance",
            authoritative_receipt=authoritative_receipt,
            authoritative_gate=authoritative_gate,
        )

    def _mark_tasks_completed_in_todo(
        self,
        task_ids: Any,
        *,
        primary_task_id: str,
        completion_reason: str,
        bundle_work_order: dict[str, Any] | None = None,
        authoritative_receipt: ImplementationReceipt | Mapping[str, Any] | None = None,
        authoritative_gate: AuthoritativeCompletionGate | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_ids = tuple(
            dict.fromkeys(
                str(item).strip() for item in task_ids if str(item).strip()
            )
        )
        if target_ids != (primary_task_id,):
            admitted, reason = False, "bundle_member_authority_missing"
        else:
            admitted, reason = self._authorize_completion_board_mutation(
                primary_task_id,
                authoritative_receipt,
                authoritative_gate,
            )
        if not admitted:
            result = {
                "updated": False,
                "task_id": primary_task_id,
                "updated_task_ids": [],
                "reason": reason,
                "completion_authoritative": False,
            }
            self._record_event(AUTHORITATIVE_COMPLETION_DENIED_EVENT, result)
            return result
        return self._decision_runtime_mutation(
            "task_board_mutation",
            {
                "operation": "mark_tasks_completed",
                "todo_path": str(self.todo_path),
                "task_ids": target_ids,
                "primary_task_id": primary_task_id,
                "completion_reason": completion_reason,
            },
            lambda: self._mark_tasks_completed_in_todo_unchecked(
                target_ids,
                primary_task_id=primary_task_id,
                completion_reason=completion_reason,
                bundle_work_order=bundle_work_order,
                authoritative_receipt=authoritative_receipt,
                authoritative_gate=authoritative_gate,
            ),
        )

    def _reopen_task_acceptance_in_todo(
        self,
        task_id: str,
        *,
        stale_reason: str,
        authoritative_receipt: ImplementationReceipt,
    ) -> dict[str, Any]:
        """Move a previously completed board record back to an actionable state."""

        def mutate() -> dict[str, Any]:
            if self.task_source is not None:
                try:
                    current = self.task_source.get(task_id)
                    if current is None:
                        raise ValueError(f"task source does not contain {task_id!r}")
                    if str(current.status).strip().lower() not in {
                        "completed",
                        "complete",
                        "done",
                        "succeeded",
                    }:
                        return {
                            "updated": False,
                            "task_id": task_id,
                            "reason": "already_pending",
                        }
                    changed = self.task_source.compare_and_swap_status(
                        task_id,
                        expected_status=current.status,
                        new_status="ready",
                        expected_revision=current.revision,
                        receipt={
                            "operation": "reopen_stale_acceptance",
                            "reason": stale_reason,
                            "implementation_receipt": authoritative_receipt.to_dict(),
                        },
                    )
                    return {
                        "updated": True,
                        "task_id": task_id,
                        "reason": "acceptance_reopened",
                        "task_source_receipt_id": changed.receipt_id,
                    }
                except Exception as exc:
                    return {
                        "updated": False,
                        "task_id": task_id,
                        "reason": "task_source_reopen_failed",
                        "error": str(exc),
                    }
            try:
                with locked_taskboard(self.todo_path) as taskboard:
                    lines = taskboard.read().splitlines(keepends=True)
                    current_task_id = ""
                    updated = False
                    for index, line in enumerate(lines):
                        if line.startswith(self.task_header_prefix):
                            header = line[3:].strip()
                            current_task_id = (
                                header.split(" ", 1)[0] if header else ""
                            )
                            continue
                        if (
                            current_task_id == task_id
                            and line.startswith("- Status:")
                        ):
                            status = line.split(":", 1)[1].strip().lower()
                            if status in {
                                "completed",
                                "complete",
                                "done",
                                "succeeded",
                            }:
                                newline = "\n" if line.endswith("\n") else ""
                                lines[index] = f"- Status: todo{newline}"
                                replace_locked_taskboard(
                                    taskboard,
                                    "".join(lines),
                                )
                                updated = True
                            break
            except OSError as exc:
                return {
                    "updated": False,
                    "task_id": task_id,
                    "reason": "board_reopen_failed",
                    "error": str(exc),
                }
            result = {
                "updated": updated,
                "task_id": task_id,
                "reason": (
                    "acceptance_reopened" if updated else "already_pending"
                ),
            }
            if updated:
                commit_result = self._commit_generated_file_update(
                    self.todo_path,
                    task_id=task_id,
                    subject=f"{task_id}: reopen stale acceptance",
                )
                if commit_result:
                    result["commit_result"] = commit_result
            return result

        return self._decision_runtime_mutation(
            "task_board_mutation",
            {
                "operation": "reopen_stale_acceptance",
                "task_id": task_id,
                "stale_reason": stale_reason,
            },
            mutate,
        )

    def deterministic_only_policy_for_task(self, task: Any) -> DeterministicOnlyPolicy:
        if task is not None and self._task_uses_typed_local_execution(task):
            return DeterministicOnlyPolicy(
                task_id=str(task.task_id or ""),
                deterministic_only=True,
            )
        return DeterministicOnlyPolicy.for_task(task)

    def reject_deterministic_only_model_invocation(
        self,
        task: Any,
        *,
        provider: str = "",
        reason: str = "model_invocation",
    ) -> dict[str, Any]:
        decision = self.deterministic_only_policy_for_task(
            task
        ).reject_model_invocation(provider=provider, reason=reason)
        if decision.get("rejected"):
            self._record_event(DETERMINISTIC_ONLY_MODEL_REJECTED_EVENT, decision)
        return decision

    def _verified_acceptance_binding(
        self,
        implementation_commit: str,
        merge_commit: str,
        repository_tree_id: str,
    ) -> tuple[str, str, bool]:
        implementation_result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{implementation_commit}^{{commit}}",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        commit_result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{merge_commit}^{{commit}}",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        tree_result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{merge_commit}^{{tree}}",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        resolved_implementation = str(
            implementation_result.stdout or ""
        ).strip()
        resolved_commit = str(commit_result.stdout or "").strip()
        resolved_tree = str(tree_result.stdout or "").strip()
        tree_id = f"git-tree:{resolved_tree}" if resolved_tree else ""
        exact = bool(
            implementation_result.returncode == 0
            and commit_result.returncode == 0
            and tree_result.returncode == 0
            and resolved_implementation == implementation_commit
            and resolved_commit == merge_commit
            and tree_id == repository_tree_id
            and self._git_ref_is_ancestor(
                resolved_implementation,
                resolved_commit,
            )
        )
        return resolved_commit or merge_commit, tree_id or repository_tree_id, exact

    def _task_has_proof_obligation(self, task: Any) -> bool:
        return bool(
            self._task_metadata_value(task, "proof required")
            or self._task_metadata_value(task, "proof obligation")
        )

    def build_task_implementation_receipt(
        self,
        task: Any,
        *,
        implementation_commit: str = "",
        merge_commit: str = "",
        repository_tree_id: str = "",
        merged: bool = False,
        validation_result: Mapping[str, Any] | None = None,
        gate_evidence: Mapping[str, Any] | None = None,
        model_invocation_observed: bool = False,
    ) -> ImplementationReceipt:
        validation = dict(validation_result or {})
        resolved_commit, resolved_tree, merge_binding_verified = (
            self._verified_acceptance_binding(
                implementation_commit,
                merge_commit,
                repository_tree_id,
            )
        )
        merge_commit = resolved_commit
        repository_tree_id = resolved_tree
        binding = {
            "task_id": task.task_id,
            "implementation_commit": implementation_commit,
            "merge_commit": merge_commit,
            "repository_tree_id": repository_tree_id,
        }
        evidence = dict(gate_evidence or {})
        # Structural and validation gates are daemon-derived, never caller-overridden.
        for kind in ("merge", "freshness", "semantic", "deterministic_only"):
            evidence.pop(kind, None)
        if merged and merge_binding_verified:
            evidence["merge"] = bound_gate_evidence(
                "merge", **binding, satisfied=True
            )

        validation_stale = bool(
            validation.get("stale")
            or validation.get("validation_stale")
            or validation.get("freshness_authoritative") is False
        )
        validation_integrity_verified, _validation_integrity_reasons = (
            verify_post_merge_validation_evidence(
                validation,
                expected_task_id=task.task_id,
                expected_target_commit=merge_commit,
                expected_repository_tree_id=repository_tree_id,
            )
        )
        validation_bound = bool(
            validation_integrity_verified
            and validation.get("schema") == POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
            and validation.get("task_id") == task.task_id
            and validation.get("target_commit") == merge_commit
            and validation.get("repository_tree_id") == repository_tree_id
            and validation.get("validation_scope") == "post_merge"
            and validation.get("passed") is True
            and not validation_stale
            and validation.get("validation_receipt_id")
        )
        if validation_bound:
            common_validation = {
                **binding,
                "satisfied": True,
                "passed": True,
                "stale": False,
                "validation_scope": "post_merge",
                "validation_receipt_id": str(
                    validation["validation_receipt_id"]
                ),
            }
            evidence["freshness"] = bound_gate_evidence(
                "freshness", **common_validation
            )
            evidence["semantic"] = bound_gate_evidence(
                "semantic", **common_validation
            )

        if not self._task_has_proof_obligation(task):
            evidence["proof"] = bound_gate_evidence(
                "proof",
                **binding,
                satisfied=True,
                not_applicable=True,
                applicability_decision="no_declared_proof_obligation",
            )

        deterministic_only = self._task_uses_typed_local_execution(task)
        evidence["deterministic_only"] = bound_gate_evidence(
            "deterministic_only",
            **binding,
            satisfied=not (
                deterministic_only and model_invocation_observed
            ),
            not_applicable=not deterministic_only,
            policy=(
                "deterministic_only"
                if deterministic_only
                else "not_deterministic_only"
            ),
            model_invocation_observed=bool(model_invocation_observed),
        )
        if deterministic_only:
            evidence["provider_review"] = bound_gate_evidence(
                "provider_review",
                **binding,
                satisfied=True,
                not_applicable=True,
                route_kind="deterministic_only",
                model_invocation_observed=False,
            )
        elif (
            not model_invocation_observed
            and not self._task_declares_independent_codex_review(task)
            and not self._task_model_assisted_provider_roles(task)
        ):
            evidence["provider_review"] = bound_gate_evidence(
                "provider_review",
                **binding,
                satisfied=True,
                not_applicable=True,
                route_kind="no_model_provider_declared",
                model_invocation_observed=False,
            )

        return build_implementation_receipt(
            task_id=task.task_id,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            merged=merged and merge_binding_verified,
            validation_passed=validation_bound,
            validation_stale=validation_stale,
            gate_evidence=evidence,
            model_invocation_observed=model_invocation_observed,
            deterministic_only=deterministic_only,
        )

    def evaluate_task_authoritative_completion(
        self,
        receipt: ImplementationReceipt | Mapping[str, Any] | None,
    ) -> AuthoritativeCompletionGate:
        return evaluate_authoritative_completion_gate(receipt)

    def mark_authoritatively_completed_if_admitted(
        self,
        task: Any,
        receipt: ImplementationReceipt | Mapping[str, Any],
        *,
        promote: bool = True,
    ) -> dict[str, Any]:
        try:
            base = (
                ImplementationReceipt.from_dict(receipt)
                if isinstance(receipt, Mapping)
                else receipt
            )
        except (TypeError, ValueError):
            base = ImplementationReceipt(task_id=task.task_id)
        if promote:
            promoted, gate = promote_authoritative_completion(
                base,
                expected_task_id=task.task_id,
            )
        else:
            promoted = base
            gate = evaluate_authoritative_completion_gate(
                base,
                expected_task_id=task.task_id,
            )
        if not gate.admitted or not gate.completion_authoritative:
            payload = {
                "updated": False,
                "authoritatively_completed": False,
                "task_id": task.task_id,
                "reason": "authoritative_completion_not_admitted",
                "gate": gate.to_dict(),
                "receipt": promoted.to_dict(),
                "acceptance_state": gate.acceptance_state,
                "pending_gates": list(gate.pending_gates),
                "completion_authoritative": False,
                "implementation_commit": base.implementation_commit,
            }
            payload["status_projection"] = project_authoritative_acceptance_status(
                task_id=task.task_id,
                receipt=promoted.to_dict(),
                gate=gate.to_dict(),
            )
            self._record_event(AUTHORITATIVE_COMPLETION_DENIED_EVENT, payload)
            return payload
        todo_update = self._mark_task_or_bundle_completed_in_todo(
            task,
            authoritative_receipt=promoted,
            authoritative_gate=gate,
        )
        completed = bool(
            todo_update.get("updated")
            or task.task_id in set(
                todo_update.get("already_completed_task_ids") or ()
            )
        )
        payload = {
            "updated": bool(todo_update.get("updated")),
            "authoritatively_completed": completed,
            "task_id": task.task_id,
            "reason": (
                "authoritative_completion_admitted"
                if completed
                else "authoritative_completion_mutation_failed"
            ),
            "gate": gate.to_dict(),
            "receipt": promoted.to_dict(),
            "todo_update_result": todo_update,
            "acceptance_state": (
                ACCEPTANCE_STATE_AUTHORITATIVE
                if completed
                else ACCEPTANCE_STATE_MERGED_PENDING
            ),
            "pending_gates": [] if completed else list(gate.pending_gates),
            "completion_authoritative": completed,
            "implementation_commit": gate.implementation_commit,
        }
        payload["status_projection"] = project_authoritative_acceptance_status(
            task_id=task.task_id,
            receipt=promoted.to_dict(),
            gate=gate.to_dict() if completed else {"admitted": False},
        )
        self._record_event(
            AUTHORITATIVE_COMPLETION_ADMITTED_EVENT
            if completed
            else AUTHORITATIVE_COMPLETION_DENIED_EVENT,
            payload,
        )
        return payload

    def record_merged_pending_acceptance(
        self,
        task: Any,
        receipt: ImplementationReceipt | Mapping[str, Any],
        gate: AuthoritativeCompletionGate | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt_obj = (
            ImplementationReceipt.from_dict(receipt)
            if isinstance(receipt, Mapping)
            else receipt
        )
        gate_obj = evaluate_authoritative_completion_gate(
            receipt_obj,
            expected_task_id=task.task_id,
        )
        status = build_merged_pending_acceptance_status(
            task_id=task.task_id,
            implementation_commit=receipt_obj.implementation_commit,
            merge_commit=receipt_obj.merge_commit,
            pending_gates=gate_obj.pending_gates,
            reason_codes=gate_obj.reason_codes,
        )
        payload = {
            **status,
            "receipt": receipt_obj.to_dict(),
            "gate": gate_obj.to_dict(),
            "status_projection": status,
        }
        self._record_event(IMPLEMENTATION_MERGED_PENDING_EVENT, payload)
        return payload

    def reopen_stale_post_merge_acceptance(
        self,
        task: Any,
        receipt: ImplementationReceipt | Mapping[str, Any],
        *,
        stale_reason: str = "post_merge_validation_stale",
    ) -> dict[str, Any]:
        reopened = reopen_acceptance_for_stale_post_merge_validation(
            receipt,
            stale_reason=stale_reason,
        )
        gate = evaluate_authoritative_completion_gate(
            reopened,
            expected_task_id=task.task_id,
        )
        board_reopen = self._reopen_task_acceptance_in_todo(
            task.task_id,
            stale_reason=stale_reason,
            authoritative_receipt=reopened,
        )
        status = build_reopened_acceptance_status(
            task_id=task.task_id,
            implementation_commit=reopened.implementation_commit,
            merge_commit=reopened.merge_commit,
            pending_gates=gate.pending_gates,
            stale_reason=stale_reason,
        )
        payload = {
            **status,
            "authoritatively_completed": False,
            "implementation_commit_preserved": bool(
                reopened.implementation_commit
            ),
            "receipt": reopened.to_dict(),
            "gate": gate.to_dict(),
            "board_reopen_result": board_reopen,
            "status_projection": status,
        }
        self._record_event(ACCEPTANCE_REOPENED_STALE_EVENT, payload)
        return payload

    def apply_post_merge_authoritative_acceptance(
        self,
        task: Any,
        *,
        implementation_commit: str = "",
        merge_commit: str = "",
        repository_tree_id: str = "",
        validation_result: Mapping[str, Any] | None = None,
        gate_evidence: Mapping[str, Any] | None = None,
        model_invocation_observed: bool = False,
    ) -> dict[str, Any]:
        receipt = self.build_task_implementation_receipt(
            task,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            merged=True,
            validation_result=validation_result,
            gate_evidence=gate_evidence,
            model_invocation_observed=model_invocation_observed,
        )
        if receipt.validation_stale:
            return self.reopen_stale_post_merge_acceptance(task, receipt)
        promoted, gate = promote_authoritative_completion(
            receipt,
            expected_task_id=task.task_id,
        )
        if gate.admitted:
            return self.mark_authoritatively_completed_if_admitted(
                task,
                promoted,
                promote=False,
            )
        return self.record_merged_pending_acceptance(task, receipt, gate)
