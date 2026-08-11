"""Verification bundles, compact summaries, and structural commitments.

This module assembles plan-bound verification results for ContextPack
consumption and builds the deterministic admitted-receipt Merkle commitment.

Authority and non-claims (normative):

* A verification commitment is **not** a zero-knowledge proof.  It is a
  domain-separated structural Merkle hash over admitted leaf records.
* Signed receipts do **not** prove test or proof execution unless the
  issuer is trusted under an external trust policy.  This module never
  invents issuer trust.
* Structural validation (schema round-trip, projection checks, CID
  re-derivation) is **not** cryptographic validation of underlying tool
  execution.  The admitted process runner and cache admission own that
  boundary.

Aggregation never upgrades a required leaf: the fail-closed lattice in
:func:`~.contracts.aggregate_terminal_status` and
:class:`~.contracts.VerificationCommitment` is authoritative.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    MAX_COLLECTION_ITEMS,
    MAX_DURATION_MS,
    MAX_SUMMARY_BYTES,
    CounterexampleReceipt,
    ModelRouteDecision,
    TerminalStatus,
    VerificationBundle,
    VerificationCommitment,
    VerificationContractError,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceipt,
    VerificationSummary,
    aggregate_terminal_status,
    build_verification_commitment as _contracts_build_verification_commitment,
)

# ---------------------------------------------------------------------------
# Evidence / schema constants
# ---------------------------------------------------------------------------

BUNDLE_BUILDER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-bundle-builder@1"
)
SUMMARY_BUILDER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-summary-builder@1"
)
VERIFICATION_SUMMARY_EVIDENCE: Final[str] = "ivp/verification-summary@1"
VERIFICATION_COMMITMENT_EVIDENCE: Final[str] = "ivp/verification-commitment@1"

# Re-export commitment domain constants so callers and tests bind the same
# codec without reaching into private Merkle helpers.
COMMITMENT_HASH_ALGORITHM: Final[str] = VerificationCommitment.HASH_ALGORITHM
COMMITMENT_LEAF_CODEC: Final[str] = VerificationCommitment.LEAF_CODEC
COMMITMENT_LEAF_DOMAIN: Final[str] = VerificationCommitment.LEAF_DOMAIN
COMMITMENT_NODE_DOMAIN: Final[str] = VerificationCommitment.NODE_DOMAIN
COMMITMENT_EMPTY_DOMAIN: Final[str] = VerificationCommitment.EMPTY_DOMAIN
COMMITMENT_IS_ZERO_KNOWLEDGE_PROOF: Final[bool] = (
    VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF
)

# Compact counterexample token estimate: ~4 UTF-8 bytes per token, clamped.
_BYTES_PER_TOKEN_ESTIMATE: Final[int] = 4
_MAX_COUNTEREXAMPLE_CONTEXT_TOKENS: Final[int] = 65_536


class VerificationBundleBuildError(VerificationContractError):
    """Fail-closed error while assembling a bundle or compact summary."""


# ---------------------------------------------------------------------------
# Classification view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleReceiptClassification:
    """Explicit partitions of a plan-bound verification bundle.

    Distinguishes:

    * **required** — every receipt key sealed into the plan
    * **admitted** — receipts currently carried by the bundle (reused ∪ executed)
    * **reused** — exact plan-approved cache hits
    * **executed** — freshly observed results for this plan
    * **unresolved** — required keys still missing from admitted receipts
    """

    required_key_cids: tuple[str, ...]
    admitted_receipt_cids: tuple[str, ...]
    reused_receipt_cids: tuple[str, ...]
    executed_receipt_cids: tuple[str, ...]
    unresolved_requirement_ids: tuple[str, ...]

    def to_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": BUNDLE_BUILDER_SCHEMA,
                "required_key_cids": self.required_key_cids,
                "admitted_receipt_cids": self.admitted_receipt_cids,
                "reused_receipt_cids": self.reused_receipt_cids,
                "executed_receipt_cids": self.executed_receipt_cids,
                "unresolved_requirement_ids": self.unresolved_requirement_ids,
            }
        )


def classify_bundle_receipts(
    verification_bundle: VerificationBundle,
) -> BundleReceiptClassification:
    """Return the required / admitted / reused / executed / unresolved partitions."""

    bundle = _require_bundle(verification_bundle)
    admitted = tuple(
        sorted(
            {item.receipt_id for item in bundle.receipts},
        )
    )
    return BundleReceiptClassification(
        required_key_cids=tuple(bundle.required_check_key_cids),
        admitted_receipt_cids=admitted,
        reused_receipt_cids=tuple(bundle.reused_receipt_cids),
        executed_receipt_cids=tuple(bundle.executed_receipt_cids),
        unresolved_requirement_ids=tuple(bundle.unresolved_requirement_ids),
    )


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def build_verification_bundle(
    verification_plan: VerificationPlan,
    *,
    reused_receipts: Sequence[VerificationReceipt] = (),
    executed_receipts: Sequence[VerificationReceipt] = (),
    counterexamples: Sequence[CounterexampleReceipt] = (),
    human_review_required: bool | None = None,
) -> VerificationBundle:
    """Assemble a plan-bound :class:`VerificationBundle`.

    Distinguishes reused vs executed admitted leaves and computes unresolved
    required keys as the exact missing set.  Rejects mixed repository-tree or
    environment leaves fail-closed.  Never upgrades a leaf terminal status.
    """

    plan = _require_plan(verification_plan)
    reused = _normalize_receipts(reused_receipts, field_name="reused_receipts")
    executed = _normalize_receipts(executed_receipts, field_name="executed_receipts")
    counterexample_items = _normalize_counterexamples(counterexamples)

    reused_ids = tuple(item.receipt_id for item in reused)
    executed_ids = tuple(item.receipt_id for item in executed)
    if set(reused_ids) & set(executed_ids):
        raise VerificationBundleBuildError(
            "reused and executed receipt sets must be disjoint"
        )

    # One admitted receipt per key (reused takes precedence only if the caller
    # supplies the same identity twice under both partitions, which is already
    # rejected above).
    receipts_by_key: dict[str, VerificationReceipt] = {}
    for receipt in (*reused, *executed):
        key_id = receipt.key.key_id
        if key_id in receipts_by_key:
            raise VerificationBundleBuildError(
                "bundle admits more than one result per required key"
            )
        receipts_by_key[key_id] = receipt

    admitted = tuple(
        sorted(
            receipts_by_key.values(),
            key=lambda item: (item.key.key_id, item.receipt_id),
        )
    )
    required_key_ids = {key.key_id for key in plan.required_receipt_keys}
    admitted_key_ids = {receipt.key.key_id for receipt in admitted}
    # Membership against the sealed required set is checked before tree /
    # environment homogeneity so out-of-scope leaves surface clearly even when
    # they also differ in environment (tool version is environment-bound).
    if not admitted_key_ids.issubset(required_key_ids):
        raise VerificationIdentityError(
            "bundle contains a receipt outside the required check set"
        )
    _reject_mixed_tree_or_environment(plan, admitted)
    unresolved = tuple(sorted(required_key_ids - admitted_key_ids))

    if human_review_required is None:
        review_flag = bool(plan.human_review_required)
    else:
        if not isinstance(human_review_required, bool):
            raise VerificationBundleBuildError(
                "human_review_required must be a boolean when provided"
            )
        if plan.human_review_required and not human_review_required:
            raise VerificationBundleBuildError(
                "bundle cannot downgrade plan-required human review"
            )
        review_flag = human_review_required

    # Construction re-validates plan-approved reuse, counterexample binding,
    # cardinality, and identity projections.
    try:
        return VerificationBundle(
            verification_plan=plan,
            receipts=admitted,
            reused_receipt_cids=reused_ids,
            executed_receipt_cids=executed_ids,
            counterexamples=counterexample_items,
            unresolved_requirement_ids=unresolved,
            human_review_required=review_flag,
        )
    except (VerificationContractError, VerificationIdentityError):
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise VerificationBundleBuildError(
            f"failed to assemble verification bundle: {exc}"
        ) from exc


def _reject_mixed_tree_or_environment(
    plan: VerificationPlan,
    receipts: Sequence[VerificationReceipt],
) -> None:
    for receipt in receipts:
        if (
            receipt.key.repository_tree_cid != plan.repository_tree_cid
            or receipt.key.environment_cid != plan.environment_cid
        ):
            raise VerificationIdentityError(
                "bundle contains mixed repository tree or environment receipts"
            )


# ---------------------------------------------------------------------------
# Compact summary
# ---------------------------------------------------------------------------


def build_verification_summary(
    verification_bundle: VerificationBundle,
    model_route_decision: ModelRouteDecision,
    *,
    changed_symbol_version_cids: Sequence[str] | None = None,
    dependency_cone_symbols: Sequence[str] | None = None,
    selected_tests: Sequence[str] | None = None,
    verification_wall_time_ms: int | None = None,
    reused_time_saved_ms: int | None = None,
    counterexample_context_tokens: int | None = None,
    policy_cid: str | None = None,
) -> VerificationSummary:
    """Project a bundle into a bounded ContextPack-ready summary.

    The summary always includes:

    * changed cone (symbol versions + dependency cone labels)
    * reused / executed check sets
    * failure receipt CIDs and compact counterexample CIDs
    * unresolved obligations and full-suite / human-review fallback state
    * wall-clock timing and reuse savings
    * provider-neutral model route

    Timing defaults sum executed / reused observation durations.  Token
    estimates for counterexamples are derived from compact canonical bytes,
    never from raw logs.
    """

    bundle = _require_bundle(verification_bundle)
    route = _require_route(model_route_decision)

    human_review_required = bool(
        bundle.human_review_required or route.requires_human_review
    )
    if route.requires_human_review != human_review_required:
        raise VerificationBundleBuildError(
            "model route cannot downgrade bundle-required human review"
        )

    receipts_by_id = {item.receipt_id: item for item in bundle.receipts}
    reused_keys = tuple(
        sorted(
            {
                receipts_by_id[receipt_cid].key.key_id
                for receipt_cid in bundle.reused_receipt_cids
            }
        )
    )
    executed_keys = tuple(
        sorted(
            {
                receipts_by_id[receipt_cid].key.key_id
                for receipt_cid in bundle.executed_receipt_cids
            }
        )
    )
    failure_receipt_cids = tuple(
        sorted(
            item.receipt_id
            for item in bundle.receipts
            if item.status in {TerminalStatus.FAILED, TerminalStatus.DISPROVED}
        )
    )
    counterexample_cids = tuple(
        sorted(item.counterexample_id for item in bundle.counterexamples)
    )
    unresolved_obligation_cids = tuple(bundle.unresolved_proof_obligation_cids)

    if changed_symbol_version_cids is None:
        symbol_versions: set[str] = set()
        for key in bundle.verification_plan.required_receipt_keys:
            symbol_versions.update(key.affected_symbol_version_cids)
        changed = tuple(sorted(symbol_versions))
    else:
        changed = _normalize_string_sequence(
            changed_symbol_version_cids,
            field_name="changed_symbol_version_cids",
        )

    if dependency_cone_symbols is None:
        cone = tuple(
            sorted(
                set(bundle.verification_plan.required_static_checks)
                | set(bundle.verification_plan.required_type_checks)
            )
        )
    else:
        cone = _normalize_string_sequence(
            dependency_cone_symbols,
            field_name="dependency_cone_symbols",
        )

    if selected_tests is None:
        tests = tuple(bundle.verification_plan.affected_tests)
    else:
        tests = _normalize_string_sequence(
            selected_tests,
            field_name="selected_tests",
        )

    wall_ms = (
        _nonnegative_int(
            verification_wall_time_ms,
            field_name="verification_wall_time_ms",
        )
        if verification_wall_time_ms is not None
        else _sum_receipt_durations(
            receipts_by_id[cid] for cid in bundle.executed_receipt_cids
        )
    )
    saved_ms = (
        _nonnegative_int(
            reused_time_saved_ms,
            field_name="reused_time_saved_ms",
        )
        if reused_time_saved_ms is not None
        else _sum_receipt_durations(
            receipts_by_id[cid] for cid in bundle.reused_receipt_cids
        )
    )
    if counterexample_context_tokens is None:
        tokens = _estimate_counterexample_tokens(bundle.counterexamples)
    else:
        tokens = _nonnegative_int(
            counterexample_context_tokens,
            field_name="counterexample_context_tokens",
        )
        if tokens > _MAX_COUNTEREXAMPLE_CONTEXT_TOKENS:
            raise VerificationBundleBuildError(
                "counterexample_context_tokens exceeds compact summary bound"
            )

    summary_policy = policy_cid if policy_cid is not None else bundle.policy_cid
    if not isinstance(summary_policy, str) or not summary_policy:
        raise VerificationBundleBuildError("policy_cid is required")

    aggregate = _bundle_aggregate_status(bundle)

    try:
        summary = VerificationSummary(
            repository_tree_cid=bundle.repository_tree_cid,
            environment_cid=bundle.environment_cid,
            changed_symbol_version_cids=changed,
            dependency_cone_symbols=cone,
            selected_tests=tests,
            reused_check_key_cids=reused_keys,
            executed_check_key_cids=executed_keys,
            failure_receipt_cids=failure_receipt_cids,
            counterexample_cids=counterexample_cids,
            unresolved_obligation_cids=unresolved_obligation_cids,
            full_suite_pending=bundle.mandatory_fallback_pending,
            human_review_required=human_review_required,
            verification_wall_time_ms=wall_ms,
            reused_time_saved_ms=saved_ms,
            counterexample_context_tokens=tokens,
            aggregate_terminal_status=aggregate,
            model_route_decision=route,
            policy_cid=summary_policy,
        )
    except (VerificationContractError, VerificationIdentityError):
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise VerificationBundleBuildError(
            f"failed to build verification summary: {exc}"
        ) from exc

    # Bound enforcement is also performed inside VerificationSummary; re-check
    # here so builder errors remain explicit for oversized projections.
    if len(summary.canonical_bytes()) > MAX_SUMMARY_BYTES:
        raise VerificationBundleBuildError(
            "verification summary exceeds ContextPack compact bound"
        )
    return summary


def _bundle_aggregate_status(bundle: VerificationBundle) -> TerminalStatus:
    """Fail-closed aggregate matching commitment lattice semantics."""

    incomplete = int(
        bool(
            bundle.mandatory_fallback_pending
            or bundle.human_review_required
            or bundle.unresolved_requirement_ids
        )
    )
    return aggregate_terminal_status(
        (item.status for item in bundle.receipts),
        unresolved_obligation_count=incomplete,
    )


def _sum_receipt_durations(receipts: Iterable[VerificationReceipt]) -> int:
    total = 0
    for receipt in receipts:
        duration = int(receipt.execution.duration_ms)
        if duration < 0:
            raise VerificationBundleBuildError(
                "receipt duration_ms must be non-negative"
            )
        total += duration
        if total > MAX_DURATION_MS:
            raise VerificationBundleBuildError(
                "aggregated duration exceeds verification duration bound"
            )
    return total


def _estimate_counterexample_tokens(
    counterexamples: Sequence[CounterexampleReceipt],
) -> int:
    total_bytes = 0
    for item in counterexamples:
        total_bytes += len(item.canonical_bytes())
    if total_bytes <= 0:
        return 0
    tokens = (total_bytes + _BYTES_PER_TOKEN_ESTIMATE - 1) // _BYTES_PER_TOKEN_ESTIMATE
    return min(tokens, _MAX_COUNTEREXAMPLE_CONTEXT_TOKENS)


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------


def build_verification_commitment(
    verification_bundle: VerificationBundle,
) -> VerificationCommitment:
    """Build a structural admitted-receipt Merkle commitment.

    Binds schema, hash algorithm, leaf codec, and the explicit domain tags
    ``IVP-LEAF@1`` / ``IVP-NODE@1`` / ``IVP-EMPTY@1`` with odd-node promotion.
    Emits Merkle root, public statement, repository tree CID, environment CID,
    required-check-set CID, unresolved-obligation count, and the fail-closed
    aggregate terminal status.

    **This commitment is not a ZK proof.**  Signed receipts require trusted
    issuers.  Structural validation is not cryptographic validation of
    execution.
    """

    bundle = _require_bundle(verification_bundle)
    # Mixed tree/environment leaves are already rejected by VerificationBundle;
    # re-assert for a clear builder-level failure path.
    _reject_mixed_tree_or_environment(bundle.verification_plan, bundle.receipts)
    commitment = _contracts_build_verification_commitment(bundle)
    _assert_commitment_contract(commitment)
    return commitment


def _assert_commitment_contract(commitment: VerificationCommitment) -> None:
    if commitment.IS_ZERO_KNOWLEDGE_PROOF:
        raise VerificationBundleBuildError(
            "verification commitment must not claim to be a ZK proof"
        )
    if commitment.HASH_ALGORITHM != COMMITMENT_HASH_ALGORITHM:
        raise VerificationBundleBuildError("unsupported commitment hash algorithm")
    if commitment.LEAF_CODEC != COMMITMENT_LEAF_CODEC:
        raise VerificationBundleBuildError("unsupported commitment leaf codec")
    if (
        commitment.LEAF_DOMAIN != COMMITMENT_LEAF_DOMAIN
        or commitment.NODE_DOMAIN != COMMITMENT_NODE_DOMAIN
        or commitment.EMPTY_DOMAIN != COMMITMENT_EMPTY_DOMAIN
    ):
        raise VerificationBundleBuildError(
            "commitment domain separation tags are not the sealed IVP@1 domains"
        )
    # Public statement must bind the identities the plan requires.
    statement = commitment.public_statement
    for field_name in (
        "repository_tree_cid",
        "environment_cid",
        "required_check_set_cid",
        "aggregate_terminal_status",
    ):
        if field_name not in statement:
            raise VerificationBundleBuildError(
                f"commitment public statement missing {field_name}"
            )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _require_plan(value: Any) -> VerificationPlan:
    if isinstance(value, VerificationPlan):
        # Detach caller-owned nested mappings.
        return VerificationPlan.from_dict(value.to_record())
    if isinstance(value, Mapping):
        return VerificationPlan.from_dict(value)
    raise VerificationBundleBuildError(
        "build_verification_bundle requires a VerificationPlan"
    )


def _require_bundle(value: Any) -> VerificationBundle:
    if isinstance(value, VerificationBundle):
        return VerificationBundle.from_dict(value.to_record())
    if isinstance(value, Mapping):
        try:
            return VerificationBundle.from_dict(value)
        except (VerificationContractError, VerificationIdentityError) as exc:
            raise VerificationBundleBuildError(
                "expected a VerificationBundle"
            ) from exc
    raise VerificationBundleBuildError(
        "expected a VerificationBundle"
    )


def _require_route(value: Any) -> ModelRouteDecision:
    if isinstance(value, ModelRouteDecision):
        return ModelRouteDecision.from_dict(value.to_record())
    if isinstance(value, Mapping):
        return ModelRouteDecision.from_dict(value)
    raise VerificationBundleBuildError(
        "build_verification_summary requires a ModelRouteDecision"
    )


def _normalize_receipts(
    values: Sequence[VerificationReceipt] | Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> tuple[VerificationReceipt, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationBundleBuildError(f"{field_name} must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBundleBuildError(f"{field_name} exceeds item bound")
    # Lazy import keeps receipt decoder private to contracts.
    from .contracts import (
        PROOF_RECEIPT_SCHEMA,
        STATIC_ANALYSIS_RECEIPT_SCHEMA,
        TEST_RECEIPT_SCHEMA,
        TYPE_CHECK_RECEIPT_SCHEMA,
        ProofReceipt,
        StaticAnalysisReceipt,
        TestReceipt,
        TypeCheckReceipt,
    )

    schema_map = {
        STATIC_ANALYSIS_RECEIPT_SCHEMA: StaticAnalysisReceipt,
        TYPE_CHECK_RECEIPT_SCHEMA: TypeCheckReceipt,
        TEST_RECEIPT_SCHEMA: TestReceipt,
        PROOF_RECEIPT_SCHEMA: ProofReceipt,
    }
    result: list[VerificationReceipt] = []
    seen_ids: set[str] = set()
    for item in values:
        if isinstance(
            item, (StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt)
        ):
            receipt: VerificationReceipt = item
        elif isinstance(item, Mapping):
            receipt_type = schema_map.get(item.get("schema"))  # type: ignore[arg-type]
            if receipt_type is None:
                raise VerificationBundleBuildError(
                    f"{field_name} contains an unsupported receipt schema"
                )
            receipt = receipt_type.from_dict(item)  # type: ignore[attr-defined]
        else:
            raise VerificationBundleBuildError(
                f"{field_name} contains an invalid receipt"
            )
        if receipt.receipt_id in seen_ids:
            raise VerificationBundleBuildError(
                f"{field_name} contains duplicate receipt identities"
            )
        seen_ids.add(receipt.receipt_id)
        result.append(receipt)
    return tuple(result)


def _normalize_counterexamples(
    values: Sequence[CounterexampleReceipt] | Sequence[Mapping[str, Any]],
) -> tuple[CounterexampleReceipt, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationBundleBuildError("counterexamples must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBundleBuildError("counterexamples exceeds item bound")
    result: list[CounterexampleReceipt] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, CounterexampleReceipt):
            counterexample = item
        elif isinstance(item, Mapping):
            counterexample = CounterexampleReceipt.from_dict(item)
        else:
            raise VerificationBundleBuildError(
                "counterexamples contains an invalid record"
            )
        if counterexample.counterexample_id in seen:
            raise VerificationBundleBuildError(
                "counterexamples contains duplicate identities"
            )
        seen.add(counterexample.counterexample_id)
        result.append(counterexample)
    return tuple(result)


def _normalize_string_sequence(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationBundleBuildError(f"{field_name} must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBundleBuildError(f"{field_name} exceeds item bound")
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise VerificationBundleBuildError(
                f"{field_name} items must be strings"
            )
        result.append(item)
    return tuple(result)


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationBundleBuildError(f"{field_name} must be an integer")
    if value < 0:
        raise VerificationBundleBuildError(f"{field_name} must be non-negative")
    if value > MAX_DURATION_MS and field_name.endswith("_ms"):
        raise VerificationBundleBuildError(f"{field_name} exceeds duration bound")
    return value


__all__ = [
    "BUNDLE_BUILDER_SCHEMA",
    "COMMITMENT_EMPTY_DOMAIN",
    "COMMITMENT_HASH_ALGORITHM",
    "COMMITMENT_IS_ZERO_KNOWLEDGE_PROOF",
    "COMMITMENT_LEAF_CODEC",
    "COMMITMENT_LEAF_DOMAIN",
    "COMMITMENT_NODE_DOMAIN",
    "SUMMARY_BUILDER_SCHEMA",
    "VERIFICATION_COMMITMENT_EVIDENCE",
    "VERIFICATION_SUMMARY_EVIDENCE",
    "BundleReceiptClassification",
    "VerificationBundleBuildError",
    "build_verification_bundle",
    "build_verification_commitment",
    "build_verification_summary",
    "classify_bundle_receipts",
]
