"""Identity-bound deterministic evidence for the canonical routing ladder.

This module is deliberately an evaluator, not an authority for task mutation.
It turns small, closed observations into reproducible receipts which state why a
cache, impact analysis, static check, selected-test run, or proof did (or did
not) decide the question.  Negative answers are decisions too: a stale cache,
an expired fence, or a failed policy CAS is resolved as a rejection and must
not be silently sent to a model for reinterpretation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _closed,
    _text,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    DeterministicEvidenceState,
    LadderEvidence,
)


DETERMINISTIC_STAGES_INTERFACE = "DeterministicRoutingStages@1"
DETERMINISTIC_STAGES_SCHEMA = "semantic-state-deterministic-stages@1"
DETERMINISTIC_RECEIPT_SCHEMA = "semantic-state-deterministic-stage-receipt@1"

IDENTITY_FIELDS = (
    "repository_id",
    "tree_id",
    "objective_revision",
    "policy_id",
    "interface_id",
    "schema_id",
    "toolchain_id",
    "environment_id",
)
_DOC_SUFFIXES = (".md", ".mdx", ".rst", ".txt", ".adoc")


class DeterministicDisposition(str, Enum):
    RESOLVES = "resolves"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"
    INELIGIBLE = "ineligible"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _receipt_id(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _closed_identity(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessError(f"{name} must be an object")
    payload = _closed(value, frozenset(IDENTITY_FIELDS), name)
    return {
        field: _text(payload[field], f"{name}.{field}")
        for field in IDENTITY_FIELDS
    }


def _texts(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise HarnessError(f"{name} must be a list")
    return tuple(sorted({_text(item, name) for item in value}))


@dataclass(frozen=True)
class DeterministicStageReceipt:
    """Stable receipt for a single deterministic-stage observation."""

    stage: str
    disposition: str
    decisive: bool
    reason_codes: tuple[str, ...]
    subject: Mapping[str, Any]
    receipt_id: str = ""

    def __post_init__(self) -> None:
        stage = _text(self.stage, "stage")
        disposition = str(self.disposition)
        if disposition not in {item.value for item in DeterministicDisposition}:
            raise HarnessError("unsupported deterministic disposition")
        if not isinstance(self.subject, Mapping):
            raise HarnessError("subject must be an object")
        reasons = tuple(
            sorted({_text(code, "reason_code") for code in self.reason_codes})
        )
        decisive = _bool(self.decisive, "decisive")
        if decisive != (disposition == DeterministicDisposition.RESOLVES.value):
            raise HarnessError("decisive must exactly match resolves disposition")
        body = {
            "schema": DETERMINISTIC_RECEIPT_SCHEMA,
            "stage": stage,
            "disposition": disposition,
            "decisive": decisive,
            "reason_codes": list(reasons),
            "subject": dict(self.subject),
        }
        receipt_id = _receipt_id(body)
        if self.receipt_id and self.receipt_id != receipt_id:
            raise HarnessError("deterministic stage receipt_id does not rehash")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "receipt_id", receipt_id)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_RECEIPT_SCHEMA,
            "stage": self.stage,
            "disposition": self.disposition,
            "decisive": self.decisive,
            "reason_codes": list(self.reason_codes),
            "subject": dict(self.subject),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        if _receipt_id(body) != self.receipt_id:
            raise HarnessError("deterministic stage receipt was mutated")
        return {**body, "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeterministicStageReceipt":
        payload = _closed(
            data,
            frozenset(
                {
                    "schema",
                    "stage",
                    "disposition",
                    "decisive",
                    "reason_codes",
                    "subject",
                    "receipt_id",
                }
            ),
            "DeterministicStageReceipt",
        )
        if payload["schema"] != DETERMINISTIC_RECEIPT_SCHEMA:
            raise HarnessError("unsupported deterministic stage receipt schema")
        if not isinstance(payload["reason_codes"], list):
            raise HarnessError("reason_codes must be a list")
        if not isinstance(payload["subject"], Mapping):
            raise HarnessError("subject must be an object")
        return cls(
            stage=payload["stage"],
            disposition=payload["disposition"],
            decisive=payload["decisive"],
            reason_codes=tuple(payload["reason_codes"]),
            subject=payload["subject"],
            receipt_id=payload["receipt_id"],
        )


def _receipt(
    stage: str,
    disposition: DeterministicDisposition,
    reasons: Sequence[str],
    subject: Mapping[str, Any],
) -> DeterministicStageReceipt:
    return DeterministicStageReceipt(
        stage=stage,
        disposition=disposition.value,
        decisive=disposition is DeterministicDisposition.RESOLVES,
        reason_codes=tuple(reasons),
        subject=subject,
    )


def evaluate_exact_receipt_freshness(
    facts: Mapping[str, Any] | None,
) -> DeterministicStageReceipt:
    """Decide cache reuse with exact identity, lease/fence, and CAS bindings."""

    stage = "exact_current_authoritative_cached_receipt"
    if facts is None:
        return _receipt(
            stage, DeterministicDisposition.UNAVAILABLE, ("cache_absent",), {}
        )
    fields = frozenset({
        "receipt_available", "receipt_admitted", "receipt_identity", "current_identity",
        "receipt_lease_id", "current_lease_id", "receipt_fence", "current_fence",
        "policy_expected_id", "policy_current_id",
    })
    payload = _closed(facts, fields, "receipt freshness facts")
    available = _bool(payload["receipt_available"], "receipt_available")
    subject: dict[str, Any] = {"receipt_available": available}
    if not available:
        return _receipt(
            stage, DeterministicDisposition.UNAVAILABLE, ("cache_absent",), subject
        )
    admitted = _bool(payload["receipt_admitted"], "receipt_admitted")
    receipt_identity = _closed_identity(payload["receipt_identity"], "receipt_identity")
    current_identity = _closed_identity(payload["current_identity"], "current_identity")
    receipt_lease = _text(payload["receipt_lease_id"], "receipt_lease_id")
    current_lease = _text(payload["current_lease_id"], "current_lease_id")
    receipt_fence = payload["receipt_fence"]
    current_fence = payload["current_fence"]
    if (
        type(receipt_fence) is not int
        or isinstance(receipt_fence, bool)
        or receipt_fence < 0
    ):
        raise HarnessError("receipt_fence must be a nonnegative integer")
    if (
        type(current_fence) is not int
        or isinstance(current_fence, bool)
        or current_fence < 0
    ):
        raise HarnessError("current_fence must be a nonnegative integer")
    expected_policy = _text(payload["policy_expected_id"], "policy_expected_id")
    current_policy = _text(payload["policy_current_id"], "policy_current_id")
    mismatches = tuple(
        field
        for field in IDENTITY_FIELDS
        if receipt_identity[field] != current_identity[field]
    )
    subject.update(
        {
            "receipt_identity": receipt_identity,
            "current_identity": current_identity,
            "receipt_lease_id": receipt_lease,
            "current_lease_id": current_lease,
            "receipt_fence": receipt_fence,
            "current_fence": current_fence,
            "policy_expected_id": expected_policy,
            "policy_current_id": current_policy,
        }
    )
    if not admitted:
        return _receipt(
            stage, DeterministicDisposition.RESOLVES, ("receipt_not_admitted",), subject
        )
    if mismatches:
        return _receipt(
            stage,
            DeterministicDisposition.RESOLVES,
            tuple(f"identity_mismatch:{item}" for item in mismatches),
            subject,
        )
    if receipt_lease != current_lease or receipt_fence != current_fence:
        return _receipt(
            stage, DeterministicDisposition.RESOLVES, ("stale_lease_or_fence",), subject
        )
    if expected_policy != current_policy:
        return _receipt(
            stage,
            DeterministicDisposition.RESOLVES,
            ("policy_pointer_cas_mismatch",),
            subject,
        )
    return _receipt(
        stage,
        DeterministicDisposition.RESOLVES,
        ("exact_authoritative_receipt_fresh",),
        subject,
    )


def evaluate_impact_analysis(facts: Mapping[str, Any] | None) -> DeterministicStageReceipt:
    """Resolve documentation-only and complete AST/symbol/dependency impact cases."""

    stage = "ast_symbol_dependency_and_impact_analysis"
    if facts is None:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("impact_facts_absent",), {})
    payload = _closed(facts, frozenset({"changed_paths", "affected_symbols", "dependency_cone_complete", "impact_safe"}), "impact facts")
    paths = _texts(payload["changed_paths"], "changed_paths")
    symbols = _texts(payload["affected_symbols"], "affected_symbols")
    complete = _bool(payload["dependency_cone_complete"], "dependency_cone_complete")
    safe = _bool(payload["impact_safe"], "impact_safe")
    subject = {"changed_paths": list(paths), "affected_symbols": list(symbols), "dependency_cone_complete": complete, "impact_safe": safe}
    docs_only = bool(paths) and all(path.casefold().endswith(_DOC_SUFFIXES) for path in paths)
    if docs_only and not symbols:
        return _receipt(stage, DeterministicDisposition.RESOLVES, ("documentation_only_no_symbol_impact",), subject)
    if not complete:
        return _receipt(stage, DeterministicDisposition.UNRESOLVED, ("dependency_cone_incomplete",), subject)
    if safe:
        return _receipt(stage, DeterministicDisposition.RESOLVES, ("complete_dependency_impact_safe",), subject)
    return _receipt(stage, DeterministicDisposition.UNRESOLVED, ("impact_requires_validation",), subject)


def evaluate_static_contract_checks(facts: Mapping[str, Any] | None) -> DeterministicStageReceipt:
    """Resolve a complete schema/type/static/lint/contract result fail-closed."""

    stage = "schema_type_static_lint_and_contract_checks"
    if facts is None:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("static_facts_absent",), {})
    fields = frozenset({"schema_valid", "type_valid", "static_valid", "lint_valid", "contracts_valid", "policy_cas_current"})
    payload = _closed(facts, fields, "static facts")
    subject = {name: _bool(payload[name], name) for name in sorted(fields)}
    failed = tuple(name for name, passed in subject.items() if not passed)
    if failed:
        return _receipt(stage, DeterministicDisposition.RESOLVES, tuple(f"check_failed:{name}" for name in failed), subject)
    return _receipt(stage, DeterministicDisposition.RESOLVES, ("schema_type_static_lint_contracts_passed",), subject)


def evaluate_selected_tests(facts: Mapping[str, Any] | None) -> DeterministicStageReceipt:
    """Resolve only a known, exact selected-test set and its observed result."""

    stage = "selected_tests"
    if facts is None:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("test_selection_absent",), {})
    payload = _closed(facts, frozenset({"known_tests", "selected_tests", "selection_complete", "selected_tests_passed"}), "selected-test facts")
    known = _texts(payload["known_tests"], "known_tests")
    selected = _texts(payload["selected_tests"], "selected_tests")
    complete = _bool(payload["selection_complete"], "selection_complete")
    passed = _bool(payload["selected_tests_passed"], "selected_tests_passed")
    subject = {"known_tests": list(known), "selected_tests": list(selected), "selection_complete": complete, "selected_tests_passed": passed}
    unknown = tuple(item for item in selected if item not in known)
    if not known or not selected:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("known_or_selected_tests_empty",), subject)
    if unknown:
        return _receipt(stage, DeterministicDisposition.RESOLVES, tuple(f"unknown_selected_test:{item}" for item in unknown), subject)
    if not complete:
        return _receipt(stage, DeterministicDisposition.UNRESOLVED, ("test_selection_incomplete",), subject)
    if not passed:
        return _receipt(stage, DeterministicDisposition.RESOLVES, ("selected_tests_failed",), subject)
    return _receipt(stage, DeterministicDisposition.RESOLVES, ("known_selected_tests_passed",), subject)


def evaluate_incremental_proof(facts: Mapping[str, Any] | None) -> DeterministicStageReceipt:
    """Resolve only a proof whose full reuse identity exactly matches current work."""

    stage = "incremental_smt_or_theorem_prover"
    if facts is None:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("proof_absent",), {})
    payload = _closed(facts, frozenset({"proof_available", "proof_identity", "current_identity", "proof_passed"}), "proof facts")
    available = _bool(payload["proof_available"], "proof_available")
    subject: dict[str, Any] = {"proof_available": available}
    if not available:
        return _receipt(stage, DeterministicDisposition.UNAVAILABLE, ("proof_absent",), subject)
    proof_identity = _closed_identity(payload["proof_identity"], "proof_identity")
    current_identity = _closed_identity(payload["current_identity"], "current_identity")
    passed = _bool(payload["proof_passed"], "proof_passed")
    subject.update({"proof_identity": proof_identity, "current_identity": current_identity, "proof_passed": passed})
    mismatch = tuple(field for field in IDENTITY_FIELDS if proof_identity[field] != current_identity[field])
    if mismatch:
        return _receipt(stage, DeterministicDisposition.RESOLVES, tuple(f"proof_identity_mismatch:{item}" for item in mismatch), subject)
    if not passed:
        return _receipt(stage, DeterministicDisposition.RESOLVES, ("incremental_proof_failed",), subject)
    return _receipt(stage, DeterministicDisposition.RESOLVES, ("incremental_proof_identity_matched",), subject)


@dataclass(frozen=True)
class DeterministicStageDecision:
    """All five receipts and their direct projection onto the canonical ladder."""

    receipts: tuple[DeterministicStageReceipt, ...]
    evidence: LadderEvidence

    def __post_init__(self) -> None:
        expected = (
            "exact_current_authoritative_cached_receipt",
            "ast_symbol_dependency_and_impact_analysis",
            "schema_type_static_lint_and_contract_checks",
            "selected_tests",
            "incremental_smt_or_theorem_prover",
        )
        if tuple(item.stage for item in self.receipts) != expected:
            raise HarnessError("deterministic receipts must cover the five ladder stages in order")

    @property
    def receipt_ids(self) -> tuple[str, ...]:
        return tuple(item.receipt_id for item in self.receipts)


def evaluate_deterministic_stages(
    *,
    receipt_freshness: Mapping[str, Any] | None = None,
    impact: Mapping[str, Any] | None = None,
    static: Mapping[str, Any] | None = None,
    tests: Mapping[str, Any] | None = None,
    proof: Mapping[str, Any] | None = None,
    model_evidence: Mapping[str, Any] | LadderEvidence | None = None,
) -> DeterministicStageDecision:
    """Evaluate all deterministic stages before the existing model/human tail."""

    receipts = (
        evaluate_exact_receipt_freshness(receipt_freshness),
        evaluate_impact_analysis(impact),
        evaluate_static_contract_checks(static),
        evaluate_selected_tests(tests),
        evaluate_incremental_proof(proof),
    )
    if model_evidence is None:
        tail = LadderEvidence()
    elif isinstance(model_evidence, LadderEvidence):
        tail = model_evidence
    elif isinstance(model_evidence, Mapping):
        tail = LadderEvidence.from_dict(model_evidence)
    else:
        raise HarnessError("model_evidence must be LadderEvidence or mapping")
    values = [item.disposition for item in receipts]
    evidence = LadderEvidence(
        cached_receipt=values[0], ast_symbol_impact=values[1], schema_type_static=values[2],
        selected_tests=values[3], incremental_prover=values[4],
        small_model_required=tail.small_model_required, medium_model_required=tail.medium_model_required,
        frontier_model_required=tail.frontier_model_required, small_model_available=tail.small_model_available,
        medium_model_available=tail.medium_model_available, frontier_model_available=tail.frontier_model_available,
        human_review_required=tail.human_review_required,
    )
    return DeterministicStageDecision(receipts=receipts, evidence=evidence)


__all__ = [
    "DETERMINISTIC_RECEIPT_SCHEMA", "DETERMINISTIC_STAGES_INTERFACE", "DETERMINISTIC_STAGES_SCHEMA",
    "DeterministicDisposition", "DeterministicStageDecision", "DeterministicStageReceipt", "IDENTITY_FIELDS",
    "evaluate_deterministic_stages", "evaluate_exact_receipt_freshness", "evaluate_impact_analysis",
    "evaluate_incremental_proof", "evaluate_selected_tests", "evaluate_static_contract_checks",
]
