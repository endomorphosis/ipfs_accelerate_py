"""Immutable Grok-first/Codex-quota-only provider attempt admission.

Provider names in prompts are untrusted data.  A Codex fallback is admitted
only from a fresh, typed, pre-effect quota observation whose immutable bindings
are exactly equal to the attempted worktree, task, scope and budget.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .contracts import ProviderFallbackReason, ProviderSelection

PREFERRED_PROVIDER = "grok"; FALLBACK_PROVIDER = "codex"; MAXIMUM_FALLBACK_DISPATCHES = 1
# Compatibility evidence types live here, but the authoritative implementation
# route itself is owned by llm_router.  Do not introduce another tuple here.
PRIMARY_MODEL_ID = "grok-4.5"; FALLBACK_MODEL_ID = "gpt-5.6-terra"; FALLBACK_REASONING_EFFORT = "high"
PROVIDER_ROUTE_POLICY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/entrypoints/provider-route-policy@2"
PROVIDER_ATTEMPT_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/entrypoints/provider-attempt-receipt@2"
QUOTA_EXHAUSTION_EVIDENCE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/entrypoints/quota-exhaustion-evidence@2"
EXACT_QUOTA_FALLBACK_REQUIREMENT_ID = "provider_route.EXACT_QUOTA_FALLBACK_REQUIREMENT_ID"
TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID = "provider_route.TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID"

class ProviderRouteError(ValueError):
    def __init__(self, message: str, *, reason_code: str = "policy_invalid") -> None: super().__init__(message); self.reason_code = reason_code

class PreferredFailureClass(str, Enum):
    UNAVAILABLE="unavailable"; CAPACITY="capacity"; AUTHENTICATION="authentication"; NETWORK="network"; TIMEOUT="timeout"; BARE_STATUS="bare_status"; NONZERO_EXIT="nonzero_exit"; UNCLASSIFIED="unclassified"; POST_EFFECT="post_effect"; REPEATED="repeated"; MODEL_DRIFT="model_drift"; EFFORT_DRIFT="effort_drift"; PROMPT_SELECTED="prompt_selected"; SCOPE_WIDENING="scope_widening"; SELF_REVIEW="self_review"
FAIL_CLOSED_PREFERRED_FAILURES = frozenset(PreferredFailureClass)

def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value: raise ProviderRouteError(f"{field} must be a canonical non-empty identity")
    return value
def _cid(value: Any, field: str) -> str: return _id(value, field)
def _digest(value: Mapping[str, Any]) -> str: return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

@dataclass(frozen=True)
class ProviderRoutePolicy:
    preferred_provider: str=PREFERRED_PROVIDER; fallback_provider: str=FALLBACK_PROVIDER; preferred_model_id: str=PRIMARY_MODEL_ID; fallback_model_id: str=FALLBACK_MODEL_ID; fallback_reasoning_effort: str=FALLBACK_REASONING_EFFORT; maximum_fallback_dispatches: int=1; independent_review_required: bool=True; quota_only_fallback: bool=True; pre_effect_only: bool=True
    def __post_init__(self) -> None:
        if (self.preferred_provider, self.fallback_provider, self.preferred_model_id, self.fallback_model_id, self.fallback_reasoning_effort) != (PREFERRED_PROVIDER, FALLBACK_PROVIDER, PRIMARY_MODEL_ID, FALLBACK_MODEL_ID, FALLBACK_REASONING_EFFORT): raise ProviderRouteError("provider/model identity drift", reason_code="policy_identity_invalid")
        if self.maximum_fallback_dispatches != 1 or not self.independent_review_required or not self.quota_only_fallback or not self.pre_effect_only: raise ProviderRouteError("route policy must retain exact fail-closed limits", reason_code="fallback_policy_invalid")
    def to_dict(self) -> dict[str, Any]: return {"schema":PROVIDER_ROUTE_POLICY_SCHEMA, **self.__dict__}
    @property
    def content_id(self) -> str: return _digest(self.to_dict())

def default_provider_route_policy() -> ProviderRoutePolicy: return ProviderRoutePolicy()

@dataclass(frozen=True)
class QuotaExhaustionEvidence:
    preferred_provider: str; preferred_model_id: str; usage_evidence_cid: str; observed_capability_cid: str; observed_at_ms: int; classifier_reason: str="preferred_provider_quota_exhausted"; fresh: bool=True; post_effect: bool=False; task_revision_cid: str=""; worktree_cid: str=""; budget_cid: str=""; scope_cid: str=""
    def __post_init__(self) -> None:
        if self.preferred_provider != PREFERRED_PROVIDER or self.preferred_model_id != PRIMARY_MODEL_ID or self.classifier_reason != ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED.value: raise ProviderRouteError("not exact typed quota evidence", reason_code="quota_evidence_invalid")
        for name in ("usage_evidence_cid", "observed_capability_cid"): object.__setattr__(self, name, _cid(getattr(self,name),name))
        for name in ("task_revision_cid","worktree_cid","budget_cid","scope_cid"):
            value=getattr(self,name)
            if value: object.__setattr__(self,name,_cid(value,name))
        if isinstance(self.observed_at_ms,bool) or not isinstance(self.observed_at_ms,int) or self.observed_at_ms <= 0: raise ProviderRouteError("quota evidence timestamp invalid", reason_code="quota_evidence_invalid")
        if not self.fresh: raise ProviderRouteError("stale quota evidence", reason_code="quota_evidence_stale")
        if self.post_effect: raise ProviderRouteError("post-effect quota evidence", reason_code="quota_evidence_post_effect")
    def to_dict(self) -> dict[str, Any]: return {"schema":QUOTA_EXHAUSTION_EVIDENCE_SCHEMA, **self.__dict__}
    @property
    def content_id(self) -> str: return _digest(self.to_dict())

@dataclass(frozen=True)
class ProviderAttemptReceipt:
    policy_cid: str; provider: str; model_id: str; attempt_cid: str; worktree_cid: str; task_revision_cid: str; process_identity: str; reasoning_effort: str=""; dispatch_index: int=1; repository_effect_observed: bool=False; budget_cid: str=""; scope_cid: str=""
    def __post_init__(self) -> None:
        for name in ("policy_cid","attempt_cid","worktree_cid","task_revision_cid","process_identity") : object.__setattr__(self,name,_cid(getattr(self,name),name))
        for name in ("budget_cid","scope_cid"):
            if getattr(self,name): object.__setattr__(self,name,_cid(getattr(self,name),name))
        if self.provider == PREFERRED_PROVIDER and self.model_id == PRIMARY_MODEL_ID and not self.reasoning_effort: return
        if self.provider == FALLBACK_PROVIDER and self.model_id == FALLBACK_MODEL_ID and self.reasoning_effort == FALLBACK_REASONING_EFFORT and self.dispatch_index == 1 and not self.repository_effect_observed: return
        raise ProviderRouteError("attempt violates exact provider policy", reason_code="attempt_invalid")
    def to_dict(self) -> dict[str, Any]: return {"schema":PROVIDER_ATTEMPT_RECEIPT_SCHEMA, **self.__dict__}
    @property
    def content_id(self) -> str: return _digest(self.to_dict())

@dataclass(frozen=True)
class ProviderFallbackReceipt:
    preferred_provider: str; fallback_provider: str; reason_code: ProviderFallbackReason; observed_capability_cid: str; task_revision_cid: str; budget_cid: str; attempt_id: str; usage_evidence_cid: str; worktree_cid: str; implementer_process_identity: str; review_authorization: str; maximum_fallback_dispatches: int=1; independent_review_required: bool=True; same_attempt_may_satisfy_review: bool=False; committed_before_dispatch: bool=True; scope_cid: str=""
    def __post_init__(self) -> None:
        if not isinstance(self.reason_code, ProviderFallbackReason):
            try: object.__setattr__(self, "reason_code", ProviderFallbackReason(self.reason_code))
            except ValueError as exc: raise ProviderRouteError("unknown fallback reason", reason_code="fallback_receipt_invalid") from exc
        if self.preferred_provider != PREFERRED_PROVIDER or self.fallback_provider != FALLBACK_PROVIDER or self.reason_code is not ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED or self.maximum_fallback_dispatches != 1 or not self.independent_review_required or self.same_attempt_may_satisfy_review or not self.committed_before_dispatch: raise ProviderRouteError("fallback receipt violates policy", reason_code="fallback_receipt_invalid")
        for name in ("observed_capability_cid","task_revision_cid","budget_cid","attempt_id","usage_evidence_cid","worktree_cid","implementer_process_identity","review_authorization") : object.__setattr__(self,name,_cid(getattr(self,name),name))
        if self.review_authorization in {self.attempt_id,self.implementer_process_identity}: raise ProviderRouteError("fallback cannot self-review", reason_code="self_review")
    @property
    def content_id(self) -> str: return _digest(self.__dict__)

@dataclass(frozen=True)
class IndependentReviewContinuation:
    """A review authority separate from the fallback implementer identity."""
    implementation_attempt_cid: str; review_authorization: str; reviewer_provider: str=FALLBACK_PROVIDER; implementer_process_identity: str=""; self_review_forbidden: bool=True
    def __post_init__(self) -> None:
        object.__setattr__(self,"implementation_attempt_cid",_cid(self.implementation_attempt_cid,"implementation_attempt_cid")); object.__setattr__(self,"review_authorization",_cid(self.review_authorization,"review_authorization"))
        if not self.self_review_forbidden or self.review_authorization in {self.implementation_attempt_cid,self.implementer_process_identity}: raise ProviderRouteError("review must be independent",reason_code="self_review")

@dataclass(frozen=True)
class ProviderRouteEvaluation:
    policy_cid: str; selected_provider: ProviderSelection; selected_model_id: str; selected_reasoning_effort: str; fallback_reason: ProviderFallbackReason; admitted: bool; reason_code: str; fallback_receipt_cid: str=""; attempt_template: Mapping[str, Any] | None=None

def classify_preferred_failure(raw_reason: str | PreferredFailureClass) -> PreferredFailureClass:
    if isinstance(raw_reason, PreferredFailureClass): return raw_reason
    normalized=str(raw_reason or "").strip().casefold().replace("-","_").replace(" ","_")
    for item in PreferredFailureClass:
        if normalized in {item.value, "preferred_provider_"+item.value}: return item
    return PreferredFailureClass.UNCLASSIFIED

def assert_model_identities(*, preferred_model_id: str, fallback_model_id: str, fallback_reasoning_effort: str) -> None:
    if (preferred_model_id, fallback_model_id, fallback_reasoning_effort) != (PRIMARY_MODEL_ID, FALLBACK_MODEL_ID, FALLBACK_REASONING_EFFORT): raise ProviderRouteError("provider model identity drift",reason_code="model_drift")

def evaluate_preferred_route(policy: ProviderRoutePolicy | None=None, *, preferred_healthy: bool=True, preferred_failure: PreferredFailureClass | str | None=None) -> ProviderRouteEvaluation:
    policy=policy or default_provider_route_policy()
    if preferred_healthy and preferred_failure is None: return ProviderRouteEvaluation(policy.content_id,ProviderSelection.GROK,PRIMARY_MODEL_ID,"",ProviderFallbackReason.NONE,True,"admitted:grok-implement")
    failure=classify_preferred_failure(preferred_failure or PreferredFailureClass.UNAVAILABLE)
    return ProviderRouteEvaluation(policy.content_id,ProviderSelection.UNAVAILABLE,"","",ProviderFallbackReason.PREFERRED_UNAVAILABLE if failure is PreferredFailureClass.UNAVAILABLE else ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE,False,"fail_closed:"+failure.value)

def evaluate_quota_fallback(policy: ProviderRoutePolicy | None=None, *, quota_evidence: QuotaExhaustionEvidence, repository_effect_observed: bool=False, prior_fallback_dispatches: int=0, prompt_selected_fallback: bool=False, fallback_model_id: str=FALLBACK_MODEL_ID, fallback_reasoning_effort: str=FALLBACK_REASONING_EFFORT, scope_widened: bool=False, attempt: ProviderAttemptReceipt | None=None, now_ms: int | None=None, max_age_ms: int=300_000) -> ProviderRouteEvaluation:
    policy=policy or default_provider_route_policy()
    if not isinstance(quota_evidence,QuotaExhaustionEvidence): raise ProviderRouteError("typed quota evidence required",reason_code="quota_evidence_invalid")
    if prompt_selected_fallback: raise ProviderRouteError("prompt cannot select provider",reason_code="prompt_selected")
    if repository_effect_observed or scope_widened or quota_evidence.post_effect: raise ProviderRouteError("fallback must be pre-effect and same scope",reason_code="post_effect" if repository_effect_observed else "scope_widening")
    if prior_fallback_dispatches >= 1: raise ProviderRouteError("fallback already dispatched",reason_code="repeated_fallback")
    if fallback_model_id != FALLBACK_MODEL_ID or fallback_reasoning_effort != FALLBACK_REASONING_EFFORT: raise ProviderRouteError("fallback model/effort drift",reason_code="model_drift")
    if now_ms is not None and (now_ms < quota_evidence.observed_at_ms or now_ms-quota_evidence.observed_at_ms > max_age_ms): raise ProviderRouteError("quota evidence stale",reason_code="quota_evidence_stale")
    if attempt is not None:
        if attempt.policy_cid != policy.content_id: raise ProviderRouteError("attempt is bound to a different provider policy",reason_code="binding_mismatch")
        if attempt.provider != FALLBACK_PROVIDER: raise ProviderRouteError("fallback attempt provider mismatch",reason_code="attempt_invalid")
        for field in ("task_revision_cid","worktree_cid","budget_cid","scope_cid"):
            evidence_value=getattr(quota_evidence,field); attempt_value=getattr(attempt,field)
            if not evidence_value or not attempt_value or evidence_value != attempt_value: raise ProviderRouteError("fallback bindings must exactly match evidence",reason_code="binding_mismatch")
    return ProviderRouteEvaluation(policy.content_id,ProviderSelection.CODEX,FALLBACK_MODEL_ID,FALLBACK_REASONING_EFFORT,ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED,True,"admitted:codex-quota-fallback")

def build_fallback_receipt(*, policy: ProviderRoutePolicy | None=None, quota_evidence: QuotaExhaustionEvidence, task_revision_cid: str, budget_cid: str, attempt_id: str, worktree_cid: str, implementer_process_identity: str, review_authorization: str, scope_cid: str="") -> ProviderFallbackReceipt:
    evaluate_quota_fallback(policy, quota_evidence=quota_evidence)
    if any(getattr(quota_evidence,n) and getattr(quota_evidence,n) != value for n,value in (("task_revision_cid",task_revision_cid),("budget_cid",budget_cid),("worktree_cid",worktree_cid),("scope_cid",scope_cid))): raise ProviderRouteError("receipt binding differs from evidence",reason_code="binding_mismatch")
    return ProviderFallbackReceipt(PREFERRED_PROVIDER,FALLBACK_PROVIDER,ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED,quota_evidence.observed_capability_cid,task_revision_cid,budget_cid,attempt_id,quota_evidence.usage_evidence_cid,worktree_cid,implementer_process_identity,review_authorization,scope_cid=scope_cid)

def build_independent_review_continuation(*, implementation_attempt_cid: str, review_authorization: str, implementer_process_identity: str="") -> IndependentReviewContinuation:
    return IndependentReviewContinuation(implementation_attempt_cid, review_authorization, implementer_process_identity=implementer_process_identity)

def build_provider_route_provenance(*, policy: ProviderRoutePolicy | None=None, selected: ProviderSelection, fallback_reason: ProviderFallbackReason=ProviderFallbackReason.NONE, fallback_receipt_cid: str="", observed_capability_cid: str, usage_evidence_cid: str, budget_cid: str, task_revision_cid: str, attempt_cid: str="", worktree_cid: str="", authenticated_profile_override_cid: str=""):
    """Construct the existing closed entrypoint provenance only after policy checks."""
    from .contracts import ProviderRouteProvenance
    policy=policy or default_provider_route_policy()
    return ProviderRouteProvenance(preferred_provider=policy.preferred_provider, fallback_provider=policy.fallback_provider, selected_provider=selected, fallback_reason=fallback_reason, fallback_receipt_cid=fallback_receipt_cid, observed_capability_cid=observed_capability_cid, usage_evidence_cid=usage_evidence_cid, budget_cid=budget_cid, task_revision_cid=task_revision_cid, attempt_cid=attempt_cid, worktree_cid=worktree_cid, authenticated_profile_override_cid=authenticated_profile_override_cid, maximum_fallback_dispatches=1, independent_review_required=True)
