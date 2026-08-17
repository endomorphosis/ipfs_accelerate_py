"""Bridge IVP verification bundles and minimized counterexamples into audits (SCG-028).

``GovernorVerificationBridge`` and :func:`build_audit_verification_evidence`
open or run the exact selected/full/static/type/proof/review checks declared
for a task class, recompute production acceptance, minimize failure evidence,
and project closed verification evidence for shadow/differential audits.

Normative fail-closed invariants:

* Patch presence, model-route presence, a single passing test, receipt
  presence, or aggregate terminal status alone **cannot** accept.
* Absent or unknown task-class acceptance mapping fails closed.
* Any missing or non-success required matrix check fails closed.
* Stale, simulated, unavailable, timeout, cancelled, or invalid evidence
  remains nonaccepting.
* Receipt status is never upgraded during projection or translation.
* Reuses canonical ``VerificationBundle`` / ``TestReceipt`` / ``ProofReceipt``
  / ``CounterexampleReceipt`` and :func:`compute_production_acceptance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Final, Iterable, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    SemanticGovernorExecutionError,
    VerificationProjection,
)
from ipfs_accelerate_py.agent_supervisor.verification.bundle import (
    build_verification_bundle,
    build_verification_summary,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CounterexampleReceipt,
    ModelRouteDecision,
    TerminalStatus,
    VerificationBundle,
    VerificationContractError,
    VerificationPlan,
    VerificationReceipt,
    VerificationReceiptKind,
    VerificationSummary,
)
from ipfs_accelerate_py.agent_supervisor.verification.counterexamples import (
    minimize_counterexample,
)
from ipfs_accelerate_py.agent_supervisor.verification.executor import (
    VerificationExecutionResult,
    compute_production_acceptance,
    execute_verification_plan,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    production_eligible,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicy,
    TaskClassAcceptanceRequirements,
)

# ---------------------------------------------------------------------------
# Evidence / interface constants
# ---------------------------------------------------------------------------

SCG_VERIFICATION_BRIDGE_EVIDENCE: Final[str] = "scg/verification-bridge@1"
GOVERNOR_VERIFICATION_BRIDGE_INTERFACE: Final[str] = "GovernorVerificationBridge@1"
AUDIT_VERIFICATION_EVIDENCE_INTERFACE: Final[str] = "AuditVerificationEvidence@1"
AUDIT_VERIFICATION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "audit-verification-evidence@1"
)

# Closed matrix check kinds declared by task-class acceptance rows.
CHECK_SELECTED_TESTS: Final[str] = "selected_tests"
CHECK_FULL_SUITE: Final[str] = "full_suite"
CHECK_STATIC: Final[str] = "static_checks"
CHECK_TYPE: Final[str] = "type_checks"
CHECK_PROOFS: Final[str] = "proofs"
CHECK_HUMAN_REVIEW: Final[str] = "human_review"
CHECK_REVIEW: Final[str] = CHECK_HUMAN_REVIEW  # alias for review wording

KNOWN_CHECK_KINDS: Final[frozenset[str]] = frozenset(
    {
        CHECK_SELECTED_TESTS,
        CHECK_FULL_SUITE,
        CHECK_STATIC,
        CHECK_TYPE,
        CHECK_PROOFS,
        CHECK_HUMAN_REVIEW,
    }
)

# Terminal statuses that can never satisfy production for a required leaf.
_NON_PRODUCTION_TERMINALS: Final[frozenset[TerminalStatus]] = frozenset(
    {
        TerminalStatus.STALE,
        TerminalStatus.SIMULATED,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.TIMEOUT,
        TerminalStatus.CANCELLED,
        TerminalStatus.INVALID,
        TerminalStatus.UNKNOWN,
        TerminalStatus.NOT_MODELED,
        TerminalStatus.FAILED,
        TerminalStatus.DISPROVED,
    }
)

_PRODUCTION_SUCCESS: Final[frozenset[TerminalStatus]] = frozenset(
    {TerminalStatus.PASSED, TerminalStatus.PROVED}
)

_FAILURE_STATUSES: Final[frozenset[TerminalStatus]] = frozenset(
    {TerminalStatus.FAILED, TerminalStatus.DISPROVED}
)

MAX_REASON_CODES: Final[int] = 256
MAX_TEXT_CHARS: Final[int] = 16_384


class GovernorVerificationBridgeError(SemanticGovernorExecutionError):
    """Raised when audit verification bridging is malformed or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "verification_bridge_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class ConflictSignal(str, Enum):
    """Closed selected/full/proof conflict signals bound into audit evidence."""

    SELECTED_FULL_OUTCOME_DISCREPANCY = "selected_full_outcome_discrepancy"
    SELECTED_PASS_FULL_FAIL = "selected_pass_full_fail"
    SELECTED_FAIL_FULL_PASS = "selected_fail_full_pass"
    FULL_SUITE_PENDING = "full_suite_pending"
    FULL_SUITE_UNAVAILABLE = "full_suite_unavailable"
    PROOF_TEST_DISCREPANCY = "proof_test_discrepancy"
    PROOF_FAILED_TESTS_PASSED = "proof_failed_tests_passed"
    PROOF_UNRESOLVED = "proof_unresolved"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class PresenceClaim(str, Enum):
    """Evidence-presence claims that alone can never establish acceptance."""

    PATCH = "patch"
    MODEL = "model"
    ONE_TEST = "one_test"
    RECEIPT = "receipt"
    AGGREGATE = "aggregate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GovernorVerificationBridgeError(f"{name} must be a nonempty trimmed string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value or len(value) > 128:
        raise GovernorVerificationBridgeError(f"{name} must be NFC text within bounds")
    return value


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise GovernorVerificationBridgeError(f"{name} must be a nonempty CID string or null")
    return value


def _stable_unique(values: Iterable[str], *, limit: int = MAX_REASON_CODES) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in values:
        text = str(item)
        if text and text not in seen:
            seen[text] = None
        if len(seen) >= limit:
            break
    return tuple(seen)


def _require_bundle(value: Any) -> VerificationBundle:
    if isinstance(value, VerificationBundle):
        return VerificationBundle.from_dict(value.to_record())
    if isinstance(value, Mapping):
        try:
            return VerificationBundle.from_dict(value)
        except (VerificationContractError, TypeError, ValueError) as exc:
            raise GovernorVerificationBridgeError(
                f"invalid VerificationBundle: {exc}",
                reason_code="invalid_bundle",
            ) from exc
    raise GovernorVerificationBridgeError(
        "expected a VerificationBundle",
        reason_code="invalid_bundle",
    )


def _require_plan(value: Any) -> VerificationPlan:
    if isinstance(value, VerificationPlan):
        return VerificationPlan.from_dict(value.to_record())
    if isinstance(value, Mapping):
        try:
            return VerificationPlan.from_dict(value)
        except (VerificationContractError, TypeError, ValueError) as exc:
            raise GovernorVerificationBridgeError(
                f"invalid VerificationPlan: {exc}",
                reason_code="invalid_plan",
            ) from exc
    raise GovernorVerificationBridgeError(
        "expected a VerificationPlan",
        reason_code="invalid_plan",
    )


def _normalize_acceptance_row(
    value: TaskClassAcceptanceRequirements | Mapping[str, Any] | None,
) -> TaskClassAcceptanceRequirements | None:
    if value is None:
        return None
    if isinstance(value, TaskClassAcceptanceRequirements):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value:
                return TaskClassAcceptanceRequirements.from_dict(value)
            return TaskClassAcceptanceRequirements(
                task_class=value.get("task_class", ""),
                risk_class=value.get("risk_class", ""),
                require_selected_tests=bool(value.get("require_selected_tests", True)),
                require_full_suite_fallback=bool(
                    value.get("require_full_suite_fallback", True)
                ),
                require_static_checks=bool(value.get("require_static_checks", True)),
                require_type_checks=bool(value.get("require_type_checks", True)),
                require_proofs=bool(value.get("require_proofs", False)),
                require_human_review=bool(value.get("require_human_review", False)),
            )
        except Exception as exc:  # noqa: BLE001 — fail closed on malformed rows
            raise GovernorVerificationBridgeError(
                f"malformed task-class acceptance row: {exc}",
                reason_code="malformed_acceptance_row",
            ) from exc
    raise GovernorVerificationBridgeError(
        "acceptance requirements must be TaskClassAcceptanceRequirements or mapping",
        reason_code="malformed_acceptance_row",
    )


def resolve_task_class_acceptance(
    *,
    task_class: str,
    risk_class: str,
    compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
    acceptance_requirements: (
        TaskClassAcceptanceRequirements | Mapping[str, Any] | None
    ) = None,
) -> TaskClassAcceptanceRequirements | None:
    """Return the closed acceptance row, or None when mapping is absent/unknown."""

    task = _token(task_class, "task_class")
    risk = _token(risk_class, "risk_class")

    explicit = _normalize_acceptance_row(acceptance_requirements)
    if explicit is not None:
        if explicit.task_class == task and explicit.risk_class == risk:
            return explicit
        # Explicit row for a different class is not a match.
        return None

    if compression_policy is None:
        return None

    policy: CompressionPolicy
    if isinstance(compression_policy, CompressionPolicy):
        policy = compression_policy
    elif isinstance(compression_policy, Mapping):
        try:
            policy = CompressionPolicy.from_dict(compression_policy)
        except Exception as exc:  # noqa: BLE001
            raise GovernorVerificationBridgeError(
                f"malformed compression policy: {exc}",
                reason_code="malformed_compression_policy",
            ) from exc
    else:
        raise GovernorVerificationBridgeError(
            "compression_policy must be CompressionPolicy or mapping",
            reason_code="malformed_compression_policy",
        )
    return policy.acceptance_for(task, risk)


def required_check_kinds(
    acceptance: TaskClassAcceptanceRequirements,
) -> tuple[str, ...]:
    """Return ordered required matrix check kinds for *acceptance*."""

    required: list[str] = []
    if acceptance.require_selected_tests:
        required.append(CHECK_SELECTED_TESTS)
    if acceptance.require_full_suite_fallback:
        required.append(CHECK_FULL_SUITE)
    if acceptance.require_static_checks:
        required.append(CHECK_STATIC)
    if acceptance.require_type_checks:
        required.append(CHECK_TYPE)
    if acceptance.require_proofs:
        required.append(CHECK_PROOFS)
    if acceptance.require_human_review:
        required.append(CHECK_HUMAN_REVIEW)
    return tuple(required)


def reject_unknown_check_kinds(declared: Sequence[str] | None) -> tuple[str, ...]:
    """Reject any check kind outside the closed matrix vocabulary."""

    if declared is None:
        return ()
    if not isinstance(declared, (list, tuple)):
        raise GovernorVerificationBridgeError(
            "declared checks must be a sequence",
            reason_code="unknown_check_mapping",
        )
    unknown = [str(item) for item in declared if str(item) not in KNOWN_CHECK_KINDS]
    if unknown:
        raise GovernorVerificationBridgeError(
            f"unknown task-class check mapping: {', '.join(unknown)}",
            reason_code="unknown_check_mapping",
            details={"unknown": unknown},
        )
    return tuple(str(item) for item in declared if str(item) in KNOWN_CHECK_KINDS)


def _receipts_by_kind(
    bundle: VerificationBundle,
) -> dict[VerificationReceiptKind, tuple[VerificationReceipt, ...]]:
    grouped: dict[VerificationReceiptKind, list[VerificationReceipt]] = {
        kind: [] for kind in VerificationReceiptKind
    }
    for receipt in bundle.receipts:
        grouped[receipt.key.receipt_kind].append(receipt)
    return {kind: tuple(items) for kind, items in grouped.items()}


def _full_suite_key_ids(plan: VerificationPlan) -> frozenset[str]:
    return frozenset(str(item) for item in plan.full_suite_receipt_key_cids)


def _selected_test_receipts(bundle: VerificationBundle) -> tuple[VerificationReceipt, ...]:
    full_ids = _full_suite_key_ids(bundle.verification_plan)
    return tuple(
        receipt
        for receipt in bundle.receipts
        if receipt.key.receipt_kind is VerificationReceiptKind.TEST
        and receipt.key.key_id not in full_ids
    )


def _full_suite_receipts(bundle: VerificationBundle) -> tuple[VerificationReceipt, ...]:
    full_ids = _full_suite_key_ids(bundle.verification_plan)
    if not full_ids:
        return ()
    return tuple(
        receipt for receipt in bundle.receipts if receipt.key.key_id in full_ids
    )


def _leaf_production_success(receipt: VerificationReceipt) -> bool:
    """True only when a leaf is a current production success (never upgraded)."""

    if receipt.status not in _PRODUCTION_SUCCESS:
        return False
    if not receipt.terminal_success:
        return False
    if not production_eligible(receipt):
        return False
    if receipt.status in _NON_PRODUCTION_TERMINALS:
        return False
    return True


def _any_non_production(receipts: Sequence[VerificationReceipt]) -> bool:
    return any(
        receipt.status
        in {
            TerminalStatus.STALE,
            TerminalStatus.SIMULATED,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.INVALID,
        }
        for receipt in receipts
    )


def plan_covers_required_checks(
    plan: VerificationPlan,
    acceptance: TaskClassAcceptanceRequirements,
) -> tuple[str, ...]:
    """Return missing planned check kinds for the acceptance matrix row."""

    required = required_check_kinds(acceptance)
    missing: list[str] = []
    keys = plan.required_receipt_keys
    kinds = {key.receipt_kind for key in keys}
    full_ids = set(plan.full_suite_receipt_key_cids)
    selected_test_keys = [
        key
        for key in keys
        if key.receipt_kind is VerificationReceiptKind.TEST and key.key_id not in full_ids
    ]

    for check in required:
        if check == CHECK_SELECTED_TESTS:
            if not selected_test_keys and not plan.affected_tests:
                missing.append(check)
        elif check == CHECK_FULL_SUITE:
            if not plan.full_suite_required and not plan.full_suite_receipt_key_cids:
                missing.append(check)
        elif check == CHECK_STATIC:
            if (
                VerificationReceiptKind.STATIC_ANALYSIS not in kinds
                and not plan.required_static_checks
            ):
                missing.append(check)
        elif check == CHECK_TYPE:
            if (
                VerificationReceiptKind.TYPE_CHECK not in kinds
                and not plan.required_type_checks
            ):
                missing.append(check)
        elif check == CHECK_PROOFS:
            if (
                VerificationReceiptKind.PROOF not in kinds
                and not plan.affected_proof_obligation_cids
            ):
                missing.append(check)
        elif check == CHECK_HUMAN_REVIEW:
            if not plan.human_review_required:
                missing.append(check)
    return tuple(missing)


def evaluate_matrix_checks(
    bundle: VerificationBundle,
    acceptance: TaskClassAcceptanceRequirements,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (required, satisfied, missing, failed) matrix check kinds."""

    required = required_check_kinds(acceptance)
    satisfied: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    plan = bundle.verification_plan
    by_kind = _receipts_by_kind(bundle)
    selected = _selected_test_receipts(bundle)
    full = _full_suite_receipts(bundle)

    for check in required:
        if check == CHECK_SELECTED_TESTS:
            if not selected and not plan.affected_tests:
                missing.append(check)
            elif not selected:
                missing.append(check)
            elif any(not _leaf_production_success(item) for item in selected):
                failed.append(check)
            else:
                satisfied.append(check)
        elif check == CHECK_FULL_SUITE:
            if bundle.mandatory_fallback_pending:
                missing.append(check)
            elif not plan.full_suite_receipt_key_cids and not plan.full_suite_required:
                missing.append(check)
            elif not full and plan.full_suite_receipt_key_cids:
                missing.append(check)
            elif full and any(not _leaf_production_success(item) for item in full):
                failed.append(check)
            elif full:
                satisfied.append(check)
            else:
                missing.append(check)
        elif check == CHECK_STATIC:
            receipts = by_kind.get(VerificationReceiptKind.STATIC_ANALYSIS, ())
            if not receipts:
                missing.append(check)
            elif any(not _leaf_production_success(item) for item in receipts):
                failed.append(check)
            else:
                satisfied.append(check)
        elif check == CHECK_TYPE:
            receipts = by_kind.get(VerificationReceiptKind.TYPE_CHECK, ())
            if not receipts:
                missing.append(check)
            elif any(not _leaf_production_success(item) for item in receipts):
                failed.append(check)
            else:
                satisfied.append(check)
        elif check == CHECK_PROOFS:
            receipts = by_kind.get(VerificationReceiptKind.PROOF, ())
            if not receipts:
                missing.append(check)
            elif bundle.unresolved_proof_obligation_cids:
                failed.append(check)
            elif any(not _leaf_production_success(item) for item in receipts):
                failed.append(check)
            else:
                satisfied.append(check)
        elif check == CHECK_HUMAN_REVIEW:
            # Human review is a required gate: it is "satisfied" as a planned
            # obligation but always blocks production acceptance.
            if bundle.human_review_required or plan.human_review_required:
                satisfied.append(check)
            else:
                missing.append(check)

    return required, tuple(satisfied), tuple(missing), tuple(failed)


def detect_conflict_signals(
    bundle: VerificationBundle,
    *,
    acceptance: TaskClassAcceptanceRequirements | None = None,
) -> tuple[str, ...]:
    """Bind selected/full/proof conflict signals without upgrading statuses."""

    signals: list[str] = []
    plan = bundle.verification_plan
    selected = _selected_test_receipts(bundle)
    full = _full_suite_receipts(bundle)
    proofs = [
        receipt
        for receipt in bundle.receipts
        if receipt.key.receipt_kind is VerificationReceiptKind.PROOF
    ]
    tests = [
        receipt
        for receipt in bundle.receipts
        if receipt.key.receipt_kind is VerificationReceiptKind.TEST
    ]

    if bundle.human_review_required or plan.human_review_required:
        signals.append(ConflictSignal.HUMAN_REVIEW_REQUIRED.value)

    if bundle.mandatory_fallback_pending or (
        plan.full_suite_required and not full
    ):
        signals.append(ConflictSignal.FULL_SUITE_PENDING.value)

    if full and any(item.status is TerminalStatus.UNAVAILABLE for item in full):
        signals.append(ConflictSignal.FULL_SUITE_UNAVAILABLE.value)

    if selected and full:
        selected_ok = all(_leaf_production_success(item) for item in selected)
        full_ok = all(_leaf_production_success(item) for item in full)
        selected_fail = any(item.status in _FAILURE_STATUSES for item in selected)
        full_fail = any(item.status in _FAILURE_STATUSES for item in full)
        if selected_ok and full_fail:
            signals.append(ConflictSignal.SELECTED_PASS_FULL_FAIL.value)
            signals.append(ConflictSignal.SELECTED_FULL_OUTCOME_DISCREPANCY.value)
        elif selected_fail and full_ok:
            signals.append(ConflictSignal.SELECTED_FAIL_FULL_PASS.value)
            signals.append(ConflictSignal.SELECTED_FULL_OUTCOME_DISCREPANCY.value)
        elif selected_ok != full_ok:
            signals.append(ConflictSignal.SELECTED_FULL_OUTCOME_DISCREPANCY.value)

    if proofs and tests:
        tests_ok = all(_leaf_production_success(item) for item in tests)
        proof_failed = any(item.status in _FAILURE_STATUSES for item in proofs)
        proof_ok = all(_leaf_production_success(item) for item in proofs)
        if proof_failed and tests_ok:
            signals.append(ConflictSignal.PROOF_FAILED_TESTS_PASSED.value)
            signals.append(ConflictSignal.PROOF_TEST_DISCREPANCY.value)
        elif proof_ok != tests_ok:
            signals.append(ConflictSignal.PROOF_TEST_DISCREPANCY.value)

    if acceptance is not None and acceptance.require_proofs:
        if bundle.unresolved_proof_obligation_cids or (
            not proofs and acceptance.require_proofs
        ):
            signals.append(ConflictSignal.PROOF_UNRESOLVED.value)

    return _stable_unique(signals)


def recompute_audit_acceptance(
    bundle: VerificationBundle,
    *,
    acceptance: TaskClassAcceptanceRequirements | None,
    advisory_key_cids: Sequence[str] = (),
    presence_claims: Sequence[str] | Mapping[str, Any] | None = None,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Recompute production acceptance for audits.

    Returns ``(production_acceptance, acceptance_matrix_satisfied, reason_codes)``.
    Never upgrades receipt statuses. Presence-only claims cannot accept.
    """

    reasons: list[str] = []

    # Presence-only claims are recorded but never establish acceptance.
    claims = _normalize_presence_claims(presence_claims)
    if claims:
        reasons.append("presence_claim_insufficient")
        for claim in claims:
            reasons.append(f"presence_only:{claim}")

    if acceptance is None:
        reasons.append("absent_or_unknown_task_class_mapping")
        return False, False, _stable_unique(reasons)

    # Base IVP production acceptance (required leaves, no upgrade).
    base_accept = compute_production_acceptance(
        bundle, advisory_key_cids=advisory_key_cids
    )
    if not base_accept:
        reasons.append("ivp_production_acceptance_false")

    required, satisfied, missing, failed = evaluate_matrix_checks(bundle, acceptance)
    matrix_satisfied = bool(required) and not missing and not failed
    if not required:
        # Empty required set is still a mapping, but nothing admits acceptance
        # without explicit checks — fail closed.
        matrix_satisfied = False
        reasons.append("empty_required_check_set")
    if missing:
        reasons.append("missing_required_checks")
        reasons.extend(f"missing:{item}" for item in missing)
    if failed:
        reasons.append("failed_required_checks")
        reasons.extend(f"failed:{item}" for item in failed)

    if acceptance.require_human_review or bundle.human_review_required:
        reasons.append("human_review_required")
        matrix_satisfied = matrix_satisfied and CHECK_HUMAN_REVIEW in satisfied
        # Human review always blocks production eligibility.
        base_accept = False

    if bundle.mandatory_fallback_pending:
        reasons.append("mandatory_full_suite_fallback_pending")
        base_accept = False
        matrix_satisfied = False

    if _any_non_production(bundle.receipts):
        reasons.append("non_production_terminal_present")
        base_accept = False

    # Structurally incomplete bundles cannot accept even if aggregate looks green.
    if not bundle.structurally_complete and not bundle.human_review_required:
        # human_review forces structurally_complete false; already blocked above
        if not acceptance.require_human_review:
            reasons.append("bundle_not_structurally_complete")
            base_accept = False

    # Explicit: aggregate / one-test / receipt count cannot flip acceptance.
    production = bool(base_accept and matrix_satisfied)
    if production and claims:
        # Even a fully green matrix cannot be justified *only* by presence claims;
        # production may still be true from recomputation, but we never allow
        # presence claims to be the sole decision path. Recomputation already
        # ran; claims do not upgrade a false result.
        pass

    if not production and "ivp_production_acceptance_false" not in reasons:
        if not matrix_satisfied:
            reasons.append("acceptance_matrix_unsatisfied")

    return production, matrix_satisfied, _stable_unique(reasons)


def _normalize_presence_claims(
    presence_claims: Sequence[str] | Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if presence_claims is None:
        return ()
    if isinstance(presence_claims, Mapping):
        claimed = []
        for key, value in presence_claims.items():
            token = str(key)
            if value and token in {item.value for item in PresenceClaim}:
                claimed.append(token)
            elif value and token in {
                "patch_cid",
                "patch_present",
                "model_route",
                "model_present",
                "one_test_passed",
                "receipt_present",
                "receipt_count",
                "aggregate_terminal_status",
                "aggregate_passed",
            }:
                # Normalize common aliases to closed presence vocabulary.
                if "patch" in token:
                    claimed.append(PresenceClaim.PATCH.value)
                elif "model" in token:
                    claimed.append(PresenceClaim.MODEL.value)
                elif "one_test" in token:
                    claimed.append(PresenceClaim.ONE_TEST.value)
                elif "receipt" in token:
                    claimed.append(PresenceClaim.RECEIPT.value)
                elif "aggregate" in token:
                    claimed.append(PresenceClaim.AGGREGATE.value)
        return _stable_unique(claimed)
    if isinstance(presence_claims, (list, tuple)):
        claimed = []
        allowed = {item.value for item in PresenceClaim}
        for item in presence_claims:
            text = str(item)
            if text in allowed:
                claimed.append(text)
            else:
                raise GovernorVerificationBridgeError(
                    f"unknown presence claim: {text}",
                    reason_code="unknown_presence_claim",
                )
        return _stable_unique(claimed)
    raise GovernorVerificationBridgeError(
        "presence_claims must be a sequence or mapping",
        reason_code="invalid_presence_claims",
    )


def minimize_bundle_failures(
    bundle: VerificationBundle,
    *,
    existing: Sequence[CounterexampleReceipt] = (),
) -> tuple[CounterexampleReceipt, ...]:
    """Minimize failed receipts into counterexamples without status upgrades."""

    by_failed_key = {
        cx.failed_key_cid: cx for cx in existing if getattr(cx, "failed_key_cid", None)
    }
    # Prefer already-minimized bundle counterexamples.
    for cx in bundle.counterexamples:
        by_failed_key.setdefault(cx.failed_key_cid, cx)

    minimized: list[CounterexampleReceipt] = []
    for receipt in bundle.receipts:
        if receipt.status not in _FAILURE_STATUSES:
            continue
        existing_cx = by_failed_key.get(receipt.key.key_id)
        if existing_cx is not None:
            minimized.append(existing_cx)
            continue
        try:
            obligation = (
                receipt.key.proof_obligation_cid
                if receipt.key.receipt_kind is VerificationReceiptKind.PROOF
                else ""
            )
            result = minimize_counterexample(
                receipt,
                failed_obligation_cid=obligation or "",
            )
            cx = result.receipt
            if (
                receipt.key.receipt_kind is not VerificationReceiptKind.PROOF
                and cx.failed_obligation_cid
            ):
                payload = dict(cx.to_record())
                payload["failed_obligation_cid"] = ""
                payload.pop("counterexample_id", None)
                payload.pop("content_id", None)
                cx = CounterexampleReceipt.from_dict(payload)
            minimized.append(cx)
        except (
            AttributeError,
            KeyError,
            VerificationContractError,
            TypeError,
            ValueError,
        ):
            continue

    # Stable order by failed key.
    minimized.sort(key=lambda item: item.failed_key_cid)
    return tuple(minimized)


def project_verification(
    bundle: VerificationBundle,
    *,
    acceptance_matrix_satisfied: bool,
    production_eligible_flag: bool,
) -> VerificationProjection:
    """Project canonical bundle identity into a governor VerificationProjection."""

    selected = _selected_test_receipts(bundle)
    full = _full_suite_receipts(bundle)
    by_kind = _receipts_by_kind(bundle)
    static = by_kind.get(VerificationReceiptKind.STATIC_ANALYSIS, ())
    proofs = by_kind.get(VerificationReceiptKind.PROOF, ())

    def _tri(receipts: Sequence[VerificationReceipt]) -> bool | None:
        if not receipts:
            return None
        return all(_leaf_production_success(item) for item in receipts)

    return VerificationProjection(
        verification_bundle_cid=bundle.bundle_id,
        selected_tests_passed=_tri(selected),
        full_suite_passed=_tri(full),
        proofs_passed=_tri(proofs),
        static_checks_passed=_tri(static),
        counterexample_present=bool(bundle.counterexamples),
        acceptance_matrix_satisfied=bool(acceptance_matrix_satisfied),
        production_eligible=bool(
            production_eligible_flag and acceptance_matrix_satisfied
        ),
    )


# ---------------------------------------------------------------------------
# Audit evidence record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditVerificationEvidence:
    """Closed audit-facing verification evidence (bundles + counterexamples)."""

    task_class: str
    risk_class: str
    verification_bundle_cid: str
    production_acceptance: bool
    acceptance_matrix_satisfied: bool
    production_eligible: bool
    required_checks: tuple[str, ...]
    satisfied_checks: tuple[str, ...]
    missing_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    conflict_signals: tuple[str, ...]
    counterexample_cids: tuple[str, ...]
    counterexamples: tuple[CounterexampleReceipt, ...]
    verification: VerificationProjection
    reason_codes: tuple[str, ...]
    aggregate_terminal_status: str | None = None
    summary_cid: str | None = None
    policy_cid: str | None = None
    acceptance_requirements: TaskClassAcceptanceRequirements | None = None
    presence_claims_observed: tuple[str, ...] = ()
    notes: str | None = None

    INTERFACE: Final[str] = AUDIT_VERIFICATION_EVIDENCE_INTERFACE
    SCHEMA: Final[str] = AUDIT_VERIFICATION_EVIDENCE_SCHEMA
    EVIDENCE: Final[str] = SCG_VERIFICATION_BRIDGE_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_class", _token(self.task_class, "task_class"))
        object.__setattr__(self, "risk_class", _token(self.risk_class, "risk_class"))
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _token(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        for name in (
            "production_acceptance",
            "acceptance_matrix_satisfied",
            "production_eligible",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise GovernorVerificationBridgeError(f"{name} must be a boolean")
        if self.production_eligible and not self.acceptance_matrix_satisfied:
            raise GovernorVerificationBridgeError(
                "production_eligible requires acceptance_matrix_satisfied"
            )
        if self.production_eligible and not self.production_acceptance:
            raise GovernorVerificationBridgeError(
                "production_eligible requires production_acceptance"
            )
        object.__setattr__(
            self, "required_checks", _stable_unique(self.required_checks)
        )
        object.__setattr__(
            self, "satisfied_checks", _stable_unique(self.satisfied_checks)
        )
        object.__setattr__(self, "missing_checks", _stable_unique(self.missing_checks))
        object.__setattr__(self, "failed_checks", _stable_unique(self.failed_checks))
        object.__setattr__(
            self, "conflict_signals", _stable_unique(self.conflict_signals)
        )
        object.__setattr__(
            self, "counterexample_cids", _stable_unique(self.counterexample_cids)
        )
        object.__setattr__(self, "reason_codes", _stable_unique(self.reason_codes))
        object.__setattr__(
            self,
            "presence_claims_observed",
            _stable_unique(self.presence_claims_observed),
        )
        if not isinstance(self.verification, VerificationProjection):
            raise GovernorVerificationBridgeError(
                "verification must be a VerificationProjection"
            )
        if self.verification.verification_bundle_cid != self.verification_bundle_cid:
            raise GovernorVerificationBridgeError(
                "verification projection bundle CID mismatch"
            )
        object.__setattr__(
            self, "summary_cid", _optional_cid(self.summary_cid, "summary_cid")
        )
        object.__setattr__(
            self, "policy_cid", _optional_cid(self.policy_cid, "policy_cid")
        )
        if self.notes is not None:
            if type(self.notes) is not str or len(self.notes) > MAX_TEXT_CHARS:
                raise GovernorVerificationBridgeError("notes must be bounded text")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_VERIFICATION_EVIDENCE_SCHEMA,
            "interface_id": AUDIT_VERIFICATION_EVIDENCE_INTERFACE,
            "evidence": SCG_VERIFICATION_BRIDGE_EVIDENCE,
            "task_class": self.task_class,
            "risk_class": self.risk_class,
            "verification_bundle_cid": self.verification_bundle_cid,
            "production_acceptance": self.production_acceptance,
            "acceptance_matrix_satisfied": self.acceptance_matrix_satisfied,
            "production_eligible": self.production_eligible,
            "required_checks": list(self.required_checks),
            "satisfied_checks": list(self.satisfied_checks),
            "missing_checks": list(self.missing_checks),
            "failed_checks": list(self.failed_checks),
            "conflict_signals": list(self.conflict_signals),
            "counterexample_cids": list(self.counterexample_cids),
            "verification": self.verification.identity_payload(),
            "reason_codes": list(self.reason_codes),
            "aggregate_terminal_status": self.aggregate_terminal_status,
            "summary_cid": self.summary_cid,
            "policy_cid": self.policy_cid,
            "acceptance_requirements": (
                None
                if self.acceptance_requirements is None
                else self.acceptance_requirements.identity_payload()
            ),
            "presence_claims_observed": list(self.presence_claims_observed),
            "notes": self.notes,
        }

    @property
    def evidence_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["evidence_cid"] = self.evidence_cid
        payload["verification"] = self.verification.to_dict()
        payload["counterexample_count"] = len(self.counterexamples)
        return payload


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@dataclass
class GovernorVerificationBridge:
    """Open/run IVP checks for a task class and project audit verification evidence.

    The bridge never upgrades receipt terminal statuses, never treats patch /
    model / one-test / receipt / aggregate presence as acceptance, and fails
    closed when the task-class mapping is absent or any required check fails.
    """

    INTERFACE: Final[str] = GOVERNOR_VERIFICATION_BRIDGE_INTERFACE
    EVIDENCE: Final[str] = SCG_VERIFICATION_BRIDGE_EVIDENCE

    minimize_failures: bool = True
    require_resource_lease: bool = False

    def resolve_acceptance(
        self,
        *,
        task_class: str,
        risk_class: str,
        compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
        acceptance_requirements: (
            TaskClassAcceptanceRequirements | Mapping[str, Any] | None
        ) = None,
    ) -> TaskClassAcceptanceRequirements | None:
        return resolve_task_class_acceptance(
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
        )

    def validate_declared_checks(self, declared: Sequence[str] | None) -> tuple[str, ...]:
        return reject_unknown_check_kinds(declared)

    def validate_plan_for_acceptance(
        self,
        plan: VerificationPlan | Mapping[str, Any],
        acceptance: TaskClassAcceptanceRequirements,
    ) -> tuple[str, ...]:
        """Return missing planned checks; empty means the plan covers the matrix."""

        sealed = _require_plan(plan)
        return plan_covers_required_checks(sealed, acceptance)

    def open_bundle(
        self,
        bundle: VerificationBundle | Mapping[str, Any],
        *,
        task_class: str,
        risk_class: str,
        compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
        acceptance_requirements: (
            TaskClassAcceptanceRequirements | Mapping[str, Any] | None
        ) = None,
        declared_checks: Sequence[str] | None = None,
        presence_claims: Sequence[str] | Mapping[str, Any] | None = None,
        model_route_decision: ModelRouteDecision | None = None,
        summary: VerificationSummary | None = None,
    ) -> AuditVerificationEvidence:
        """Project an existing verification bundle into audit evidence."""

        if declared_checks is not None:
            self.validate_declared_checks(declared_checks)

        sealed = _require_bundle(bundle)
        acceptance = self.resolve_acceptance(
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
        )
        return self._evidence_from_bundle(
            sealed,
            task_class=task_class,
            risk_class=risk_class,
            acceptance=acceptance,
            presence_claims=presence_claims,
            model_route_decision=model_route_decision,
            summary=summary,
        )

    def run_plan(
        self,
        plan: VerificationPlan | Mapping[str, Any],
        *,
        task_class: str,
        risk_class: str,
        compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
        acceptance_requirements: (
            TaskClassAcceptanceRequirements | Mapping[str, Any] | None
        ) = None,
        declared_checks: Sequence[str] | None = None,
        presence_claims: Sequence[str] | Mapping[str, Any] | None = None,
        check_runner: Callable[..., Any] | None = None,
        model_route_decision: ModelRouteDecision | None = None,
        **executor_kwargs: Any,
    ) -> AuditVerificationEvidence:
        """Execute a sealed plan (or open-path via executor) and project evidence."""

        if declared_checks is not None:
            self.validate_declared_checks(declared_checks)

        sealed = _require_plan(plan)
        acceptance = self.resolve_acceptance(
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
        )
        if acceptance is None:
            # Still execute for diagnostic receipts, but acceptance fails closed.
            pass
        else:
            missing_planned = plan_covers_required_checks(sealed, acceptance)
            if missing_planned and declared_checks is None:
                # Plan does not declare matrix-required checks — fail closed at
                # evidence projection (execution may still run for diagnostics).
                pass

        result = execute_verification_plan(
            sealed,
            check_runner=check_runner,
            model_route_decision=model_route_decision,
            require_resource_lease=executor_kwargs.pop(
                "require_resource_lease", self.require_resource_lease
            ),
            minimize_failures=executor_kwargs.pop(
                "minimize_failures", self.minimize_failures
            ),
            **executor_kwargs,
        )
        return self.from_execution_result(
            result,
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
            presence_claims=presence_claims,
        )

    def from_execution_result(
        self,
        result: VerificationExecutionResult,
        *,
        task_class: str,
        risk_class: str,
        compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
        acceptance_requirements: (
            TaskClassAcceptanceRequirements | Mapping[str, Any] | None
        ) = None,
        presence_claims: Sequence[str] | Mapping[str, Any] | None = None,
    ) -> AuditVerificationEvidence:
        if not isinstance(result, VerificationExecutionResult):
            raise GovernorVerificationBridgeError(
                "expected a VerificationExecutionResult",
                reason_code="invalid_execution_result",
            )
        acceptance = self.resolve_acceptance(
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
        )
        return self._evidence_from_bundle(
            result.bundle,
            task_class=task_class,
            risk_class=risk_class,
            acceptance=acceptance,
            presence_claims=presence_claims,
            model_route_decision=result.model_route_decision,
            summary=result.summary,
            precomputed_counterexamples=result.counterexamples,
            executor_acceptance=result.production_acceptance,
        )

    def _evidence_from_bundle(
        self,
        bundle: VerificationBundle,
        *,
        task_class: str,
        risk_class: str,
        acceptance: TaskClassAcceptanceRequirements | None,
        presence_claims: Sequence[str] | Mapping[str, Any] | None,
        model_route_decision: ModelRouteDecision | None,
        summary: VerificationSummary | None,
        precomputed_counterexamples: Sequence[CounterexampleReceipt] = (),
        executor_acceptance: bool | None = None,
    ) -> AuditVerificationEvidence:
        reasons: list[str] = []
        claims = _normalize_presence_claims(presence_claims)

        if acceptance is None:
            production = False
            matrix_ok = False
            required: tuple[str, ...] = ()
            satisfied: tuple[str, ...] = ()
            missing: tuple[str, ...] = ()
            failed: tuple[str, ...] = ()
            reasons.append("absent_or_unknown_task_class_mapping")
        else:
            planned_missing = plan_covers_required_checks(
                bundle.verification_plan, acceptance
            )
            if planned_missing:
                reasons.append("plan_missing_required_check_kinds")
                reasons.extend(f"plan_missing:{item}" for item in planned_missing)
            required, satisfied, missing, failed = evaluate_matrix_checks(
                bundle, acceptance
            )
            production, matrix_ok, accept_reasons = recompute_audit_acceptance(
                bundle,
                acceptance=acceptance,
                presence_claims=claims,
            )
            reasons.extend(accept_reasons)
            # Executor acceptance cannot upgrade bridge recomputation.
            if executor_acceptance is False:
                production = False
                reasons.append("executor_production_acceptance_false")
            if planned_missing:
                production = False
                matrix_ok = False

        # Presence-only path: if only presence claims are supplied without a
        # green recomputation, stay nonaccepting (already enforced).
        if not production and claims:
            reasons.append("presence_claims_cannot_accept")

        counterexamples: tuple[CounterexampleReceipt, ...]
        if self.minimize_failures:
            counterexamples = minimize_bundle_failures(
                bundle, existing=precomputed_counterexamples
            )
        else:
            counterexamples = tuple(bundle.counterexamples) or tuple(
                precomputed_counterexamples
            )

        # If minimization produced new counterexamples, rebuild a projection-
        # only view; never mutate/upgrade original receipt statuses. Prefer the
        # original bundle identity when counterexamples already match.
        evidence_bundle = bundle
        if counterexamples and tuple(bundle.counterexamples) != counterexamples:
            try:
                by_receipt_id = {
                    item.receipt_id: item for item in bundle.receipts
                }
                reused = tuple(
                    by_receipt_id[cid]
                    for cid in bundle.reused_receipt_cids
                    if cid in by_receipt_id
                )
                executed = tuple(
                    by_receipt_id[cid]
                    for cid in bundle.executed_receipt_cids
                    if cid in by_receipt_id
                )
                if not reused and not executed:
                    executed = tuple(bundle.receipts)
                evidence_bundle = build_verification_bundle(
                    bundle.verification_plan,
                    reused_receipts=reused,
                    executed_receipts=executed,
                    counterexamples=counterexamples,
                    human_review_required=bundle.human_review_required,
                )
            except (VerificationContractError, TypeError, ValueError):
                # Keep original bundle; counterexamples still recorded separately.
                evidence_bundle = bundle
                reasons.append("counterexample_bundle_rebuild_skipped")

        conflicts = detect_conflict_signals(evidence_bundle, acceptance=acceptance)
        if conflicts:
            # Conflicts never upgrade acceptance; selected-pass/full-fail blocks.
            if (
                ConflictSignal.SELECTED_PASS_FULL_FAIL.value in conflicts
                or ConflictSignal.PROOF_FAILED_TESTS_PASSED.value in conflicts
                or ConflictSignal.FULL_SUITE_PENDING.value in conflicts
                or ConflictSignal.FULL_SUITE_UNAVAILABLE.value in conflicts
            ):
                production = False
                matrix_ok = False
                reasons.append("conflict_signal_blocks_acceptance")

        production_eligible_flag = bool(production and matrix_ok)
        projection = project_verification(
            evidence_bundle,
            acceptance_matrix_satisfied=matrix_ok,
            production_eligible_flag=production_eligible_flag,
        )

        aggregate: str | None = None
        summary_cid: str | None = None
        if summary is not None:
            aggregate = (
                summary.aggregate_terminal_status.value
                if hasattr(summary.aggregate_terminal_status, "value")
                else str(summary.aggregate_terminal_status)
            )
            summary_cid = getattr(summary, "summary_id", None) or getattr(
                summary, "content_id", None
            )
        elif model_route_decision is not None:
            try:
                built = build_verification_summary(
                    evidence_bundle,
                    model_route_decision,
                    verification_wall_time_ms=0,
                )
                aggregate = built.aggregate_terminal_status.value
                summary_cid = built.summary_id
            except (VerificationContractError, TypeError, ValueError, AttributeError):
                aggregate = None

        # Aggregate PASSED alone cannot accept — re-assert if someone tries.
        if aggregate in {"passed", "proved"} and not production_eligible_flag:
            reasons.append("aggregate_presence_cannot_accept")

        cx_cids = tuple(
            getattr(cx, "counterexample_id", None)
            or getattr(cx, "content_id", None)
            or cid_for_structured(cx.to_record())
            for cx in counterexamples
        )

        return AuditVerificationEvidence(
            task_class=_token(task_class, "task_class"),
            risk_class=_token(risk_class, "risk_class"),
            verification_bundle_cid=evidence_bundle.bundle_id,
            production_acceptance=bool(production_eligible_flag),
            acceptance_matrix_satisfied=bool(matrix_ok),
            production_eligible=bool(production_eligible_flag),
            required_checks=required,
            satisfied_checks=satisfied,
            missing_checks=missing,
            failed_checks=failed,
            conflict_signals=conflicts,
            counterexample_cids=tuple(str(item) for item in cx_cids if item),
            counterexamples=counterexamples,
            verification=projection,
            reason_codes=_stable_unique(reasons),
            aggregate_terminal_status=aggregate,
            summary_cid=summary_cid if isinstance(summary_cid, str) else None,
            policy_cid=evidence_bundle.verification_plan.policy_cid,
            acceptance_requirements=acceptance,
            presence_claims_observed=claims,
        )


def build_audit_verification_evidence(
    *,
    task_class: str,
    risk_class: str,
    verification_bundle: VerificationBundle | Mapping[str, Any] | None = None,
    verification_plan: VerificationPlan | Mapping[str, Any] | None = None,
    execution_result: VerificationExecutionResult | None = None,
    compression_policy: CompressionPolicy | Mapping[str, Any] | None = None,
    acceptance_requirements: (
        TaskClassAcceptanceRequirements | Mapping[str, Any] | None
    ) = None,
    declared_checks: Sequence[str] | None = None,
    presence_claims: Sequence[str] | Mapping[str, Any] | None = None,
    check_runner: Callable[..., Any] | None = None,
    model_route_decision: ModelRouteDecision | None = None,
    minimize_failures: bool = True,
    require_resource_lease: bool = False,
    **executor_kwargs: Any,
) -> AuditVerificationEvidence:
    """Build audit verification evidence from a bundle, plan run, or execution result.

    Exactly one primary evidence source should drive the result: prefer
    ``execution_result``, then ``verification_bundle``, then execute
    ``verification_plan``.
    """

    bridge = GovernorVerificationBridge(
        minimize_failures=minimize_failures,
        require_resource_lease=require_resource_lease,
    )
    if declared_checks is not None:
        bridge.validate_declared_checks(declared_checks)

    if execution_result is not None:
        return bridge.from_execution_result(
            execution_result,
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
            presence_claims=presence_claims,
        )
    if verification_bundle is not None:
        return bridge.open_bundle(
            verification_bundle,
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
            declared_checks=declared_checks,
            presence_claims=presence_claims,
            model_route_decision=model_route_decision,
        )
    if verification_plan is not None:
        return bridge.run_plan(
            verification_plan,
            task_class=task_class,
            risk_class=risk_class,
            compression_policy=compression_policy,
            acceptance_requirements=acceptance_requirements,
            declared_checks=declared_checks,
            presence_claims=presence_claims,
            check_runner=check_runner,
            model_route_decision=model_route_decision,
            **executor_kwargs,
        )
    raise GovernorVerificationBridgeError(
        "one of execution_result, verification_bundle, or verification_plan is required",
        reason_code="missing_verification_source",
    )


__all__ = [
    "AUDIT_VERIFICATION_EVIDENCE_INTERFACE",
    "AUDIT_VERIFICATION_EVIDENCE_SCHEMA",
    "CHECK_FULL_SUITE",
    "CHECK_HUMAN_REVIEW",
    "CHECK_PROOFS",
    "CHECK_REVIEW",
    "CHECK_SELECTED_TESTS",
    "CHECK_STATIC",
    "CHECK_TYPE",
    "ConflictSignal",
    "GOVERNOR_VERIFICATION_BRIDGE_INTERFACE",
    "KNOWN_CHECK_KINDS",
    "PresenceClaim",
    "SCG_VERIFICATION_BRIDGE_EVIDENCE",
    "AuditVerificationEvidence",
    "GovernorVerificationBridge",
    "GovernorVerificationBridgeError",
    "build_audit_verification_evidence",
    "detect_conflict_signals",
    "evaluate_matrix_checks",
    "minimize_bundle_failures",
    "plan_covers_required_checks",
    "project_verification",
    "recompute_audit_acceptance",
    "reject_unknown_check_kinds",
    "required_check_kinds",
    "resolve_task_class_acceptance",
]
