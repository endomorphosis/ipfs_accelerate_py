"""Exact Grok-first, Codex quota-only provider policy and attempt evidence.

ASE2-002 / ASE2-G020: separate immutable pre-launch provider policy from
per-attempt and fallback evidence.  The production and compatibility routes
bind exact model and reasoning identities.  Codex implementation fallback is
authorized only by fresh typed Grok quota exhaustion before any repository
effect, at most once, and never self-reviews.

Non-quota preferred-provider failures (unavailable, capacity, authentication,
network, timeout, bare status, nonzero exit, unclassified, post-effect,
repeated, model-drift, effort-drift, prompt-selected, scope-widening, and
self-review) fail closed without a Terra implementation attempt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
)

from .capability_resolver import (
    FALLBACK_PROVIDER,
    MAXIMUM_FALLBACK_DISPATCHES,
    PREFERRED_PROVIDER,
    ProviderFallbackReceipt,
)
from .contracts import (
    ProviderFallbackReason,
    ProviderRouteProvenance,
    ProviderSelection,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
PROVIDER_ROUTE_POLICY_SCHEMA: Final = f"{SCHEMA_PREFIX}/provider-route-policy@1"
PROVIDER_ATTEMPT_RECEIPT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/provider-attempt-receipt@1"
)
QUOTA_EXHAUSTION_EVIDENCE_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/quota-exhaustion-evidence@1"
)
INDEPENDENT_REVIEW_CONTINUATION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/independent-review-continuation@1"
)
ROUTE_EVALUATION_SCHEMA: Final = f"{SCHEMA_PREFIX}/provider-route-evaluation@1"

# Authoritative exact identities for the ASE2/ASE production route.
PRIMARY_MODEL_ID: Final = "grok-4.5"
FALLBACK_MODEL_ID: Final = "gpt-5.6-terra"
FALLBACK_REASONING_EFFORT: Final = "medium"

EXACT_QUOTA_FALLBACK_REQUIREMENT_ID: Final = (
    "provider_route.EXACT_QUOTA_FALLBACK_REQUIREMENT_ID"
)
TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID: Final = (
    "provider_route.TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID"
)

# Non-quota preferred-provider failure classes that must never become a
# Codex implementation attempt (ASE2-G020 acceptance matrix).
class PreferredFailureClass(str, Enum):
    UNAVAILABLE = "unavailable"
    CAPACITY = "capacity"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    TIMEOUT = "timeout"
    BARE_STATUS = "bare_status"
    NONZERO_EXIT = "nonzero_exit"
    UNCLASSIFIED = "unclassified"
    POST_EFFECT = "post_effect"
    REPEATED = "repeated"
    MODEL_DRIFT = "model_drift"
    EFFORT_DRIFT = "effort_drift"
    PROMPT_SELECTED = "prompt_selected"
    SCOPE_WIDENING = "scope_widening"
    SELF_REVIEW = "self_review"


FAIL_CLOSED_PREFERRED_FAILURES: Final[frozenset[PreferredFailureClass]] = (
    frozenset(PreferredFailureClass)
)

# Only this preferred failure may authorize Codex implementation fallback.
QUOTA_ONLY_FALLBACK_REASON: Final = (
    ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
)


class ProviderRouteError(ValueError):
    """Bounded public failure at the provider-route policy boundary."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _token(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text or text != str(value).strip() or any(ord(c) < 32 for c in text):
        raise ProviderRouteError(
            f"{field} must be a non-empty canonical token",
            reason_code="policy_invalid",
        )
    return text


def _require_cid(value: Any, field: str) -> str:
    text = _token(value, field)
    if not (
        text.startswith("baguqeer")
        or text.startswith("bafy")
        or text.startswith("sha256:")
        or text.startswith("cid:")
    ):
        # Accept content-addressed forms used by entrypoint contracts.
        if len(text) < 8:
            raise ProviderRouteError(
                f"{field} must be a content identity",
                reason_code="policy_invalid",
            )
    return text


def _bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ProviderRouteError(
        f"{field} must be a boolean",
        reason_code="policy_invalid",
    )


def _positive_int(value: Any, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderRouteError(
            f"{field} must be a positive integer",
            reason_code="policy_invalid",
        )
    if value > maximum:
        raise ProviderRouteError(
            f"{field} may not exceed {maximum}",
            reason_code="policy_invalid",
        )
    return value


@dataclass(frozen=True)
class ProviderRoutePolicy:
    """Immutable pre-launch provider policy (not per-attempt evidence)."""

    SCHEMA: ClassVar[str] = PROVIDER_ROUTE_POLICY_SCHEMA

    preferred_provider: str = PREFERRED_PROVIDER
    fallback_provider: str = FALLBACK_PROVIDER
    preferred_model_id: str = PRIMARY_MODEL_ID
    fallback_model_id: str = FALLBACK_MODEL_ID
    fallback_reasoning_effort: str = FALLBACK_REASONING_EFFORT
    maximum_fallback_dispatches: int = MAXIMUM_FALLBACK_DISPATCHES
    independent_review_required: bool = True
    quota_only_fallback: bool = True
    pre_effect_only: bool = True

    def __post_init__(self) -> None:
        preferred = _token(self.preferred_provider, "preferred_provider")
        fallback = _token(self.fallback_provider, "fallback_provider")
        if preferred != PREFERRED_PROVIDER or fallback != FALLBACK_PROVIDER:
            raise ProviderRouteError(
                "built-in policy must be Grok then Codex",
                reason_code="policy_identity_invalid",
            )
        object.__setattr__(self, "preferred_provider", preferred)
        object.__setattr__(self, "fallback_provider", fallback)
        preferred_model = _token(self.preferred_model_id, "preferred_model_id")
        fallback_model = _token(self.fallback_model_id, "fallback_model_id")
        effort = _token(
            self.fallback_reasoning_effort, "fallback_reasoning_effort"
        )
        if preferred_model != PRIMARY_MODEL_ID:
            raise ProviderRouteError(
                f"preferred model must be exactly {PRIMARY_MODEL_ID}",
                reason_code="model_identity_invalid",
            )
        if fallback_model != FALLBACK_MODEL_ID:
            raise ProviderRouteError(
                f"fallback model must be exactly {FALLBACK_MODEL_ID}",
                reason_code="model_identity_invalid",
            )
        if effort != FALLBACK_REASONING_EFFORT:
            raise ProviderRouteError(
                f"fallback reasoning effort must be exactly {FALLBACK_REASONING_EFFORT}",
                reason_code="effort_identity_invalid",
            )
        object.__setattr__(self, "preferred_model_id", preferred_model)
        object.__setattr__(self, "fallback_model_id", fallback_model)
        object.__setattr__(self, "fallback_reasoning_effort", effort)
        object.__setattr__(
            self,
            "maximum_fallback_dispatches",
            _positive_int(
                self.maximum_fallback_dispatches,
                "maximum_fallback_dispatches",
                maximum=MAXIMUM_FALLBACK_DISPATCHES,
            ),
        )
        for name in (
            "independent_review_required",
            "quota_only_fallback",
            "pre_effect_only",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if not self.independent_review_required:
            raise ProviderRouteError(
                "Codex fallback always requires independent review",
                reason_code="review_policy_invalid",
            )
        if not self.quota_only_fallback:
            raise ProviderRouteError(
                "only quota-exhaustion may authorize Codex fallback",
                reason_code="fallback_policy_invalid",
            )
        if not self.pre_effect_only:
            raise ProviderRouteError(
                "fallback is pre-effect only",
                reason_code="fallback_policy_invalid",
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "preferred_provider": self.preferred_provider,
            "fallback_provider": self.fallback_provider,
            "preferred_model_id": self.preferred_model_id,
            "fallback_model_id": self.fallback_model_id,
            "fallback_reasoning_effort": self.fallback_reasoning_effort,
            "maximum_fallback_dispatches": self.maximum_fallback_dispatches,
            "independent_review_required": self.independent_review_required,
            "quota_only_fallback": self.quota_only_fallback,
            "pre_effect_only": self.pre_effect_only,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload())


def default_provider_route_policy() -> ProviderRoutePolicy:
    """Return the single admitted exact-identity production policy."""

    return ProviderRoutePolicy()


@dataclass(frozen=True)
class QuotaExhaustionEvidence:
    """Fresh typed evidence that the preferred provider is quota-exhausted.

    Environment selection, bare HTTP status, CLI availability, and free-form
    provider text are non-authoritative without a committed usage evidence CID.
    """

    SCHEMA: ClassVar[str] = QUOTA_EXHAUSTION_EVIDENCE_SCHEMA

    preferred_provider: str
    preferred_model_id: str
    usage_evidence_cid: str
    observed_capability_cid: str
    observed_at_ms: int
    classifier_reason: str = "preferred_provider_quota_exhausted"
    fresh: bool = True
    post_effect: bool = False

    def __post_init__(self) -> None:
        preferred = _token(self.preferred_provider, "preferred_provider")
        model = _token(self.preferred_model_id, "preferred_model_id")
        if preferred != PREFERRED_PROVIDER:
            raise ProviderRouteError(
                "quota evidence must name the preferred provider",
                reason_code="quota_evidence_invalid",
            )
        if model != PRIMARY_MODEL_ID:
            raise ProviderRouteError(
                "quota evidence must bind the exact primary model",
                reason_code="quota_evidence_invalid",
            )
        object.__setattr__(self, "preferred_provider", preferred)
        object.__setattr__(self, "preferred_model_id", model)
        object.__setattr__(
            self,
            "usage_evidence_cid",
            _require_cid(self.usage_evidence_cid, "usage_evidence_cid"),
        )
        object.__setattr__(
            self,
            "observed_capability_cid",
            _require_cid(self.observed_capability_cid, "observed_capability_cid"),
        )
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 1
        ):
            raise ProviderRouteError(
                "observed_at_ms must be a positive integer",
                reason_code="quota_evidence_invalid",
            )
        classifier = _token(self.classifier_reason, "classifier_reason")
        if classifier != ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED.value:
            raise ProviderRouteError(
                "quota classifier must be preferred_provider_quota_exhausted",
                reason_code="quota_evidence_invalid",
            )
        object.__setattr__(self, "classifier_reason", classifier)
        object.__setattr__(self, "fresh", _bool(self.fresh, "fresh"))
        object.__setattr__(self, "post_effect", _bool(self.post_effect, "post_effect"))
        if not self.fresh:
            raise ProviderRouteError(
                "stale quota evidence cannot authorize fallback",
                reason_code="quota_evidence_stale",
            )
        if self.post_effect:
            raise ProviderRouteError(
                "post-effect quota claims cannot authorize fallback",
                reason_code="quota_evidence_post_effect",
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "preferred_provider": self.preferred_provider,
            "preferred_model_id": self.preferred_model_id,
            "usage_evidence_cid": self.usage_evidence_cid,
            "observed_capability_cid": self.observed_capability_cid,
            "observed_at_ms": self.observed_at_ms,
            "classifier_reason": self.classifier_reason,
            "fresh": self.fresh,
            "post_effect": self.post_effect,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload())


@dataclass(frozen=True)
class ProviderAttemptReceipt:
    """Per-attempt implementation identity distinct from immutable policy."""

    SCHEMA: ClassVar[str] = PROVIDER_ATTEMPT_RECEIPT_SCHEMA

    policy_cid: str
    provider: str
    model_id: str
    attempt_cid: str
    worktree_cid: str
    task_revision_cid: str
    process_identity: str
    reasoning_effort: str = ""
    dispatch_index: int = 1
    repository_effect_observed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )
        provider = _token(self.provider, "provider")
        model = _token(self.model_id, "model_id")
        if provider not in {PREFERRED_PROVIDER, FALLBACK_PROVIDER}:
            raise ProviderRouteError(
                "attempt provider must be grok or codex",
                reason_code="attempt_invalid",
            )
        if provider == PREFERRED_PROVIDER and model != PRIMARY_MODEL_ID:
            raise ProviderRouteError(
                f"Grok attempts must use {PRIMARY_MODEL_ID}",
                reason_code="model_drift",
            )
        if provider == FALLBACK_PROVIDER and model != FALLBACK_MODEL_ID:
            raise ProviderRouteError(
                f"Codex attempts must use {FALLBACK_MODEL_ID}",
                reason_code="model_drift",
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model_id", model)
        for name in (
            "attempt_cid",
            "worktree_cid",
            "task_revision_cid",
            "process_identity",
        ):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        effort = str(self.reasoning_effort or "").strip()
        if provider == FALLBACK_PROVIDER:
            if effort != FALLBACK_REASONING_EFFORT:
                raise ProviderRouteError(
                    f"Codex attempts require reasoning effort {FALLBACK_REASONING_EFFORT}",
                    reason_code="effort_drift",
                )
        elif effort:
            raise ProviderRouteError(
                "Grok attempts do not declare Codex reasoning effort",
                reason_code="effort_drift",
            )
        object.__setattr__(self, "reasoning_effort", effort)
        object.__setattr__(
            self,
            "dispatch_index",
            _positive_int(
                self.dispatch_index,
                "dispatch_index",
                maximum=MAXIMUM_FALLBACK_DISPATCHES + 1,
            ),
        )
        object.__setattr__(
            self,
            "repository_effect_observed",
            _bool(self.repository_effect_observed, "repository_effect_observed"),
        )
        if provider == FALLBACK_PROVIDER and self.dispatch_index != 1:
            raise ProviderRouteError(
                "Codex fallback may dispatch at most once",
                reason_code="repeated_fallback",
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "policy_cid": self.policy_cid,
            "provider": self.provider,
            "model_id": self.model_id,
            "attempt_cid": self.attempt_cid,
            "worktree_cid": self.worktree_cid,
            "task_revision_cid": self.task_revision_cid,
            "process_identity": self.process_identity,
            "reasoning_effort": self.reasoning_effort,
            "dispatch_index": self.dispatch_index,
            "repository_effect_observed": self.repository_effect_observed,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload())


@dataclass(frozen=True)
class IndependentReviewContinuation:
    """Distinct review authorization that cannot equal the implementer attempt."""

    SCHEMA: ClassVar[str] = INDEPENDENT_REVIEW_CONTINUATION_SCHEMA

    implementation_attempt_cid: str
    review_authorization: str
    reviewer_provider: str = FALLBACK_PROVIDER
    implementer_process_identity: str = ""
    self_review_forbidden: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation_attempt_cid",
            _require_cid(
                self.implementation_attempt_cid, "implementation_attempt_cid"
            ),
        )
        object.__setattr__(
            self,
            "review_authorization",
            _require_cid(self.review_authorization, "review_authorization"),
        )
        reviewer = _token(self.reviewer_provider, "reviewer_provider")
        object.__setattr__(self, "reviewer_provider", reviewer)
        implementer = str(self.implementer_process_identity or "").strip()
        if implementer:
            implementer = _require_cid(
                implementer, "implementer_process_identity"
            )
        object.__setattr__(self, "implementer_process_identity", implementer)
        object.__setattr__(
            self,
            "self_review_forbidden",
            _bool(self.self_review_forbidden, "self_review_forbidden"),
        )
        if not self.self_review_forbidden:
            raise ProviderRouteError(
                "self-review must remain forbidden",
                reason_code="self_review",
            )
        if self.review_authorization == self.implementation_attempt_cid:
            raise ProviderRouteError(
                "review authorization cannot equal the implementation attempt",
                reason_code="self_review",
            )
        if implementer and self.review_authorization == implementer:
            raise ProviderRouteError(
                "review authorization cannot equal implementer process identity",
                reason_code="self_review",
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "implementation_attempt_cid": self.implementation_attempt_cid,
            "review_authorization": self.review_authorization,
            "reviewer_provider": self.reviewer_provider,
            "implementer_process_identity": self.implementer_process_identity,
            "self_review_forbidden": self.self_review_forbidden,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload())


@dataclass(frozen=True)
class ProviderRouteEvaluation:
    """Deterministic route decision under one policy and one evidence slice."""

    SCHEMA: ClassVar[str] = ROUTE_EVALUATION_SCHEMA

    policy_cid: str
    selected_provider: ProviderSelection
    selected_model_id: str
    selected_reasoning_effort: str
    fallback_reason: ProviderFallbackReason
    admitted: bool
    reason_code: str
    fallback_receipt_cid: str = ""
    attempt_template: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "policy_cid": self.policy_cid,
            "selected_provider": self.selected_provider.value,
            "selected_model_id": self.selected_model_id,
            "selected_reasoning_effort": self.selected_reasoning_effort,
            "fallback_reason": self.fallback_reason.value,
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "fallback_receipt_cid": self.fallback_receipt_cid,
            "attempt_template": (
                dict(self.attempt_template) if self.attempt_template else None
            ),
        }


def classify_preferred_failure(raw_reason: str | PreferredFailureClass) -> PreferredFailureClass:
    """Map an operator/provider signal into a fail-closed failure class."""

    if isinstance(raw_reason, PreferredFailureClass):
        return raw_reason
    text = str(raw_reason or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "preferred_provider_unavailable": PreferredFailureClass.UNAVAILABLE,
        "unavailable": PreferredFailureClass.UNAVAILABLE,
        "preferred_provider_capacity_unavailable": PreferredFailureClass.CAPACITY,
        "capacity": PreferredFailureClass.CAPACITY,
        "auth": PreferredFailureClass.AUTHENTICATION,
        "authentication": PreferredFailureClass.AUTHENTICATION,
        "unauthorized": PreferredFailureClass.AUTHENTICATION,
        "network": PreferredFailureClass.NETWORK,
        "timeout": PreferredFailureClass.TIMEOUT,
        "bare_status": PreferredFailureClass.BARE_STATUS,
        "http_status": PreferredFailureClass.BARE_STATUS,
        "nonzero_exit": PreferredFailureClass.NONZERO_EXIT,
        "exit_code": PreferredFailureClass.NONZERO_EXIT,
        "unclassified": PreferredFailureClass.UNCLASSIFIED,
        "post_effect": PreferredFailureClass.POST_EFFECT,
        "repeated": PreferredFailureClass.REPEATED,
        "model_drift": PreferredFailureClass.MODEL_DRIFT,
        "effort_drift": PreferredFailureClass.EFFORT_DRIFT,
        "prompt_selected": PreferredFailureClass.PROMPT_SELECTED,
        "scope_widening": PreferredFailureClass.SCOPE_WIDENING,
        "self_review": PreferredFailureClass.SELF_REVIEW,
        "preferred_provider_quota_exhausted": PreferredFailureClass.UNCLASSIFIED,
    }
    # Quota is not a fail-closed PreferredFailureClass path for fallback;
    # callers must supply QuotaExhaustionEvidence instead.
    if text in aliases:
        return aliases[text]
    return PreferredFailureClass.UNCLASSIFIED


def assert_model_identities(
    *,
    preferred_model_id: str,
    fallback_model_id: str,
    fallback_reasoning_effort: str,
) -> None:
    """Fail closed on model or effort drift relative to the exact policy."""

    if preferred_model_id != PRIMARY_MODEL_ID:
        raise ProviderRouteError(
            f"requires Grok primary model {PRIMARY_MODEL_ID}",
            reason_code="model_drift",
        )
    if fallback_model_id != FALLBACK_MODEL_ID:
        raise ProviderRouteError(
            f"requires Codex fallback model {FALLBACK_MODEL_ID}",
            reason_code="model_drift",
        )
    if fallback_reasoning_effort != FALLBACK_REASONING_EFFORT:
        raise ProviderRouteError(
            f"requires Codex reasoning effort {FALLBACK_REASONING_EFFORT}",
            reason_code="effort_drift",
        )


def evaluate_preferred_route(
    policy: ProviderRoutePolicy | None = None,
    *,
    preferred_healthy: bool = True,
    preferred_failure: PreferredFailureClass | str | None = None,
) -> ProviderRouteEvaluation:
    """Admit Grok when healthy; fail closed on non-quota preferred failures."""

    policy = policy or default_provider_route_policy()
    if preferred_healthy and preferred_failure is None:
        return ProviderRouteEvaluation(
            policy_cid=policy.content_id,
            selected_provider=ProviderSelection.GROK,
            selected_model_id=policy.preferred_model_id,
            selected_reasoning_effort="",
            fallback_reason=ProviderFallbackReason.NONE,
            admitted=True,
            reason_code="admitted:grok-implement",
        )
    failure = classify_preferred_failure(
        preferred_failure or PreferredFailureClass.UNAVAILABLE
    )
    if failure not in FAIL_CLOSED_PREFERRED_FAILURES:
        failure = PreferredFailureClass.UNCLASSIFIED
    return ProviderRouteEvaluation(
        policy_cid=policy.content_id,
        selected_provider=ProviderSelection.UNAVAILABLE,
        selected_model_id="",
        selected_reasoning_effort="",
        fallback_reason=ProviderFallbackReason.PREFERRED_UNAVAILABLE
        if failure is PreferredFailureClass.UNAVAILABLE
        else ProviderFallbackReason.PREFERRED_CAPACITY_UNAVAILABLE
        if failure is PreferredFailureClass.CAPACITY
        else ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE,
        admitted=False,
        reason_code=f"fail_closed:{failure.value}",
    )


def evaluate_quota_fallback(
    policy: ProviderRoutePolicy | None = None,
    *,
    quota_evidence: QuotaExhaustionEvidence,
    repository_effect_observed: bool = False,
    prior_fallback_dispatches: int = 0,
    prompt_selected_fallback: bool = False,
    fallback_model_id: str = FALLBACK_MODEL_ID,
    fallback_reasoning_effort: str = FALLBACK_REASONING_EFFORT,
    scope_widened: bool = False,
) -> ProviderRouteEvaluation:
    """Admit at most one pre-effect Codex fallback under exact identities."""

    policy = policy or default_provider_route_policy()
    if not isinstance(quota_evidence, QuotaExhaustionEvidence):
        raise ProviderRouteError(
            "quota fallback requires typed QuotaExhaustionEvidence",
            reason_code="quota_evidence_invalid",
        )
    if prompt_selected_fallback:
        raise ProviderRouteError(
            "prompt text cannot select Codex fallback",
            reason_code="prompt_selected",
        )
    if scope_widened:
        raise ProviderRouteError(
            "scope-widening cannot authorize Codex fallback",
            reason_code="scope_widening",
        )
    if repository_effect_observed or quota_evidence.post_effect:
        raise ProviderRouteError(
            "post-effect failures cannot authorize Codex fallback",
            reason_code="post_effect",
        )
    if prior_fallback_dispatches >= policy.maximum_fallback_dispatches:
        raise ProviderRouteError(
            "Codex fallback may run at most once",
            reason_code="repeated_fallback",
        )
    assert_model_identities(
        preferred_model_id=quota_evidence.preferred_model_id,
        fallback_model_id=fallback_model_id,
        fallback_reasoning_effort=fallback_reasoning_effort,
    )
    return ProviderRouteEvaluation(
        policy_cid=policy.content_id,
        selected_provider=ProviderSelection.CODEX,
        selected_model_id=policy.fallback_model_id,
        selected_reasoning_effort=policy.fallback_reasoning_effort,
        fallback_reason=QUOTA_ONLY_FALLBACK_REASON,
        admitted=True,
        reason_code="admitted:codex-quota-fallback",
        fallback_receipt_cid="",  # caller commits ProviderFallbackReceipt
    )


def build_fallback_receipt(
    *,
    policy: ProviderRoutePolicy | None = None,
    quota_evidence: QuotaExhaustionEvidence,
    task_revision_cid: str,
    budget_cid: str,
    attempt_id: str,
    worktree_cid: str,
    implementer_process_identity: str,
    review_authorization: str,
) -> ProviderFallbackReceipt:
    """Commit a typed pre-effect Codex fallback receipt before dispatch."""

    policy = policy or default_provider_route_policy()
    evaluation = evaluate_quota_fallback(
        policy, quota_evidence=quota_evidence
    )
    if not evaluation.admitted:
        raise ProviderRouteError(
            "fallback receipt requires an admitted quota evaluation",
            reason_code=evaluation.reason_code,
        )
    return ProviderFallbackReceipt(
        preferred_provider=policy.preferred_provider,
        fallback_provider=policy.fallback_provider,
        reason_code=ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED,
        observed_capability_cid=quota_evidence.observed_capability_cid,
        task_revision_cid=task_revision_cid,
        budget_cid=budget_cid,
        attempt_id=attempt_id,
        usage_evidence_cid=quota_evidence.usage_evidence_cid,
        worktree_cid=worktree_cid,
        implementer_process_identity=implementer_process_identity,
        review_authorization=review_authorization,
        maximum_fallback_dispatches=policy.maximum_fallback_dispatches,
        independent_review_required=True,
        same_attempt_may_satisfy_review=False,
        committed_before_dispatch=True,
    )


def build_independent_review_continuation(
    *,
    implementation_attempt_cid: str,
    review_authorization: str,
    implementer_process_identity: str = "",
) -> IndependentReviewContinuation:
    """Bind a review continuation that cannot self-attest the implementer."""

    return IndependentReviewContinuation(
        implementation_attempt_cid=implementation_attempt_cid,
        review_authorization=review_authorization,
        implementer_process_identity=implementer_process_identity,
    )


def build_provider_route_provenance(
    *,
    policy: ProviderRoutePolicy | None = None,
    selected: ProviderSelection,
    fallback_reason: ProviderFallbackReason = ProviderFallbackReason.NONE,
    fallback_receipt_cid: str = "",
    observed_capability_cid: str,
    usage_evidence_cid: str,
    budget_cid: str,
    task_revision_cid: str,
    attempt_cid: str = "",
    worktree_cid: str = "",
    authenticated_profile_override_cid: str = "",
) -> ProviderRouteProvenance:
    """Assemble the frozen entrypoint contract for one route decision."""

    policy = policy or default_provider_route_policy()
    return ProviderRouteProvenance(
        preferred_provider=policy.preferred_provider,
        fallback_provider=policy.fallback_provider,
        selected_provider=selected,
        fallback_reason=fallback_reason,
        fallback_receipt_cid=fallback_receipt_cid,
        observed_capability_cid=observed_capability_cid,
        usage_evidence_cid=usage_evidence_cid,
        budget_cid=budget_cid,
        task_revision_cid=task_revision_cid,
        attempt_cid=attempt_cid,
        worktree_cid=worktree_cid,
        authenticated_profile_override_cid=authenticated_profile_override_cid,
        maximum_fallback_dispatches=policy.maximum_fallback_dispatches,
        independent_review_required=policy.independent_review_required,
    )


__all__ = [
    "EXACT_QUOTA_FALLBACK_REQUIREMENT_ID",
    "FAIL_CLOSED_PREFERRED_FAILURES",
    "FALLBACK_MODEL_ID",
    "FALLBACK_REASONING_EFFORT",
    "IndependentReviewContinuation",
    "PRIMARY_MODEL_ID",
    "PreferredFailureClass",
    "ProviderAttemptReceipt",
    "ProviderFallbackReceipt",
    "ProviderRouteError",
    "ProviderRouteEvaluation",
    "ProviderRoutePolicy",
    "QUOTA_ONLY_FALLBACK_REASON",
    "QuotaExhaustionEvidence",
    "TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID",
    "assert_model_identities",
    "build_fallback_receipt",
    "build_independent_review_continuation",
    "build_provider_route_provenance",
    "classify_preferred_failure",
    "default_provider_route_policy",
    "evaluate_preferred_route",
    "evaluate_quota_fallback",
]
