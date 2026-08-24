"""Frozen benchmark configurations C and D (PCCE-065).

This module is an admission and measurement boundary, not a provider or
governance implementation.  A caller supplies one explicitly permitted,
provider-neutral execution port.  The returned observation is checked against
the frozen PCCE-060 configuration, identity, isolation, routing, verification,
and governed-lifecycle contracts before a raw benchmark row is emitted.

Configuration C adds frozen routing plus incremental verification and proof
reuse to semantic ContextPacks.  Configuration D adds the complete governed
lifecycle: sufficiency, bounded expansion, assurance, incremental sealing, and
human escalation.  Simulated, stale, unavailable, malformed, or partially
bound evidence can never become an accepted result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.proof_context.benchmarks.configurations_ab import (
    METRIC_NAMES,
    BenchmarkRun,
    PairIdentity,
    SemanticContextPack,
    TaskAgentView,
)
from ipfs_accelerate_py.proof_context.lifecycle import STAGES as GOVERNED_LIFECYCLE_STAGES
from ipfs_accelerate_py.utils.cid_utils import cid_for_obj, validate_cid

CONFIGURATION_SCHEMA: Final[str] = "ipfs-datasets.proof-context.benchmark-configuration@1"
RAW_RESULT_SCHEMA: Final[str] = "ipfs-datasets.proof-context.benchmark-raw-result@1"
RUNNER_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.benchmark-configurations-cd@1"
PAIR_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.benchmark-pair-cd@1"
IDENTITY_PROFILE: Final[str] = "software-contract-cid-profile-v1"

PROVENANCE_CLASSES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
TERMINAL_STATUSES: Final[tuple[str, ...]] = (
    "succeeded",
    "rejected",
    "verification_failed",
    "proof_failed",
    "assurance_failed",
    "context_insufficient",
    "model_escalation_required",
    "human_review_required",
    "unavailable",
    "timeout",
    "cancelled",
    "invalid",
    "stale",
    "simulated",
    "infrastructure_failure",
    "partial_effect",
    "repair_required",
)
ROUTES: Final[tuple[str, ...]] = ("small", "local", "frontier", "human", "unavailable")
ROUTE_DECISIONS: Final[tuple[str, ...]] = (
    "selected",
    "escalated",
    "human",
    "unavailable",
)
REUSE_OUTCOMES: Final[tuple[str, ...]] = ("hit", "miss", "stale", "unavailable")
C_STAGES: Final[tuple[str, ...]] = (
    "context-pack",
    "route",
    "proposal",
    "incremental-verify",
    "hidden-full-scoring",
    "disposition",
)

_COMMON_CONFIGURATION: Final[dict[str, Any]] = {
    "schema": CONFIGURATION_SCHEMA,
    "hidden_full_scoring": True,
    "prompt_policy": "pcce-benchmark-prompt-policy-v1",
    "seed_policy": "sha256-corpus-task-first-unsigned-64-v1",
    "environment_policy": "exact-pcce-056-qualified-environment-or-unavailable",
    "context_estimator": "utf8-bytes-ceiling-divide-by-four-no-calibration@1",
}
_CONFIGURATION_C: Final[dict[str, Any]] = {
    **_COMMON_CONFIGURATION,
    "configuration_id": "C",
    "context_method": "semantic-context-pack-v0.1",
    "model_policy": "execution-permit-exact-frozen-route-policy@1",
    "verification_policy": "incremental-tests-proofs-with-hidden-full-scoring@1",
    "routing_enabled": True,
    "incremental_verification_enabled": True,
    "proof_reuse_enabled": True,
    "sufficiency_enabled": False,
    "context_expansion_enabled": False,
    "assurance_enabled": False,
    "incremental_seal_enabled": False,
    "human_escalation_enabled": False,
}
_CONFIGURATION_D: Final[dict[str, Any]] = {
    **_COMMON_CONFIGURATION,
    "configuration_id": "D",
    "context_method": "semantic-context-pack-v0.1",
    "model_policy": "execution-permit-exact-frozen-route-policy@1",
    "verification_policy": "incremental-tests-proofs-with-hidden-full-scoring@1",
    "routing_enabled": True,
    "incremental_verification_enabled": True,
    "proof_reuse_enabled": True,
    "sufficiency_enabled": True,
    "context_expansion_enabled": True,
    "assurance_enabled": True,
    "incremental_seal_enabled": True,
    "human_escalation_enabled": True,
}

CONFIGURATION_C_CID: Final[str] = cid_for_obj(_CONFIGURATION_C, codec="dag-json")
CONFIGURATION_D_CID: Final[str] = cid_for_obj(_CONFIGURATION_D, codec="dag-json")
CONFIGURATION_CIDS: Final[Mapping[str, str]] = MappingProxyType(
    {"C": CONFIGURATION_C_CID, "D": CONFIGURATION_D_CID}
)


class ConfigurationCDError(ValueError):
    """Raised when an input would weaken the frozen C/D contract."""


class HiddenDataDenied(ConfigurationCDError):
    """Raised before dispatch when a provider-visible projection is invalid."""


class ExecutionPortUnavailable(RuntimeError):
    """Typed port signal for an unavailable execution capability."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _cid(value: Any, field: str) -> str:
    try:
        return validate_cid(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationCDError(f"{field} must be a canonical CID") from exc


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationCDError(f"{field} must be a non-empty string")
    return value


def _reason_code(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) > 96 or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in text
    ):
        raise ConfigurationCDError(f"{field} must be a bounded lowercase typed reason code")
    return text


def _count(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationCDError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_count(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _count(value, field)


def _optional_cid(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _cid(value, field)


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ConfigurationCDError(f"{field} must be boolean")
    return value


def _optional_boolean(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field)


def _status(value: Any, field: str = "status") -> str:
    if value not in TERMINAL_STATUSES:
        raise ConfigurationCDError(f"{field} is outside the frozen terminal taxonomy")
    return value


def _provenance(value: Any, field: str = "provenance") -> str:
    if value not in PROVENANCE_CLASSES:
        raise ConfigurationCDError(f"{field} is outside the frozen provenance taxonomy")
    return value


def configuration_descriptor(configuration_id: str) -> dict[str, Any]:
    if configuration_id == "C":
        return _thaw(_CONFIGURATION_C)
    if configuration_id == "D":
        return _thaw(_CONFIGURATION_D)
    raise ConfigurationCDError("only frozen configurations C and D are admitted")


def configuration_cid(configuration_id: str) -> str:
    try:
        return CONFIGURATION_CIDS[configuration_id]
    except (KeyError, TypeError) as exc:
        raise ConfigurationCDError("only frozen configurations C and D are admitted") from exc


@dataclass(frozen=True)
class RouteExecutionPermit:
    """Exact PCCE-067 execution authority; absence preserves PCCE-056 NO-GO."""

    permit_cid: str
    route_policy_cid: str
    configuration_cid: str
    corpus_manifest_cid: str
    task_record_cid: str
    visible_projection_cid: str
    repository_state_cid: str
    environment_cid: str
    provider_id: str
    model_id: str
    model_revision: str
    task_id: str
    seed: int
    attempt: int
    provenance: str
    available: bool
    live_execution_eligible: bool
    reason: str

    def __post_init__(self) -> None:
        for field in (
            "permit_cid",
            "route_policy_cid",
            "configuration_cid",
            "corpus_manifest_cid",
            "task_record_cid",
            "visible_projection_cid",
            "repository_state_cid",
            "environment_cid",
        ):
            object.__setattr__(self, field, _cid(getattr(self, field), field))
        for field in ("provider_id", "model_id", "model_revision", "task_id"):
            _text(getattr(self, field), field)
        _count(self.seed, "seed")
        _count(self.attempt, "attempt", minimum=1)
        _reason_code(self.reason, "reason")
        _provenance(self.provenance)
        _boolean(self.available, "available")
        _boolean(self.live_execution_eligible, "live_execution_eligible")

    def admits(self, identity: PairIdentity, configuration_id: str) -> bool:
        return bool(
            self.configuration_cid == configuration_cid(configuration_id)
            and self.corpus_manifest_cid == identity.corpus_manifest_cid
            and self.task_record_cid == identity.task_record_cid
            and self.visible_projection_cid == identity.visible_projection_cid
            and self.repository_state_cid == identity.repository_state_cid
            and self.environment_cid == identity.environment_cid
            and self.provider_id == identity.provider_id
            and self.model_id == identity.model_id
            and self.model_revision == identity.model_revision
            and self.task_id == identity.task_id
            and self.seed == identity.seed
            and self.attempt == identity.attempt
            and self.available
            and (self.provenance != "live" or self.live_execution_eligible)
        )


@dataclass(frozen=True)
class StageEvidence:
    """One identity-bound stage result; bodies and hidden diagnostics are absent."""

    stage: str
    status: str
    provenance: str
    evidence_cid: str
    configuration_cid: str
    identity: PairIdentity

    def __post_init__(self) -> None:
        _text(self.stage, "stage")
        _status(self.status, "stage.status")
        _provenance(self.provenance, "stage.provenance")
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "stage.evidence_cid"))
        object.__setattr__(
            self,
            "configuration_cid",
            _cid(self.configuration_cid, "stage.configuration_cid"),
        )
        if type(self.identity) is not PairIdentity:
            raise ConfigurationCDError("stage identity must use the exact pair identity type")
        PairIdentity.__post_init__(self.identity)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "provenance": self.provenance,
            "evidence_cid": self.evidence_cid,
            "configuration_cid": self.configuration_cid,
            "identity": dict(self.identity.as_mapping()),
        }


@dataclass(frozen=True)
class ContextDecision:
    status: str
    provenance: str
    evidence_cid: str
    pack_cid: str
    visible_projection_cid: str
    initial_sufficient: bool | None
    sufficient_after_expansion: bool | None
    fallback_count: int
    expansion_count: int
    expansion_tokens: int

    def __post_init__(self) -> None:
        _status(self.status, "context.status")
        _provenance(self.provenance, "context.provenance")
        for field in ("evidence_cid", "pack_cid", "visible_projection_cid"):
            object.__setattr__(self, field, _cid(getattr(self, field), f"context.{field}"))
        _optional_boolean(self.initial_sufficient, "initial_sufficient")
        _optional_boolean(self.sufficient_after_expansion, "sufficient_after_expansion")
        for field in ("fallback_count", "expansion_count", "expansion_tokens"):
            _count(getattr(self, field), field)
        if self.expansion_count == 0 and self.expansion_tokens != 0:
            raise ConfigurationCDError("zero expansions cannot report expansion tokens")
        if self.expansion_count > 0 and self.expansion_tokens == 0:
            raise ConfigurationCDError("context expansions require observed expansion tokens")


@dataclass(frozen=True)
class RouteDecision:
    status: str
    provenance: str
    evidence_cid: str
    route_policy_cid: str
    route: str
    decision_kind: str
    provider_id: str
    model_id: str
    model_revision: str
    previous_route: str | None
    reason: str

    def __post_init__(self) -> None:
        _status(self.status, "route.status")
        _provenance(self.provenance, "route.provenance")
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "route.evidence_cid"))
        object.__setattr__(
            self,
            "route_policy_cid",
            _cid(self.route_policy_cid, "route.route_policy_cid"),
        )
        if self.route not in ROUTES:
            raise ConfigurationCDError("route is outside the frozen route taxonomy")
        if self.decision_kind not in ROUTE_DECISIONS:
            raise ConfigurationCDError("decision_kind is outside the frozen route taxonomy")
        for field in ("provider_id", "model_id", "model_revision"):
            _text(getattr(self, field), f"route.{field}")
        _reason_code(self.reason, "route.reason")
        if self.previous_route is not None and self.previous_route not in ROUTES:
            raise ConfigurationCDError("previous_route is outside the frozen route taxonomy")
        if self.decision_kind == "escalated":
            if self.previous_route is None or self.previous_route == self.route:
                raise ConfigurationCDError(
                    "explicit escalation requires a different previous route"
                )
        elif self.previous_route is not None:
            raise ConfigurationCDError("previous_route requires an explicit escalation decision")
        if self.route == "human" and self.decision_kind != "human":
            raise ConfigurationCDError("human routes must be visibly classified as human")
        if self.decision_kind == "human" and self.route != "human":
            raise ConfigurationCDError("human decisions must use the visible human route")
        if self.route == "unavailable" and self.decision_kind != "unavailable":
            raise ConfigurationCDError("unavailable routes must be visibly classified unavailable")
        if self.decision_kind == "unavailable" and self.route != "unavailable":
            raise ConfigurationCDError(
                "unavailable decisions must use the visible unavailable route"
            )


@dataclass(frozen=True)
class VerificationDecision:
    status: str
    provenance: str
    evidence_cid: str
    proposal_cid: str
    verification_plan_cid: str
    reuse_outcome: str
    reuse_receipt_cid: str | None
    selected_test_count: int
    selected_test_pass_count: int
    proof_selected_count: int
    proof_executed_count: int
    proof_pass_count: int
    proof_fail_count: int
    full_fallback_used: bool
    hidden_scoring_after_proposal: bool
    full_test_count: int
    full_test_pass_count: int
    hidden_test_total_count: int
    hidden_test_pass_count: int
    regression_count: int
    critical_regression_count: int
    out_of_scope_edit_count: int
    semantic_outcome_match: bool
    verification_cost_micros: int | None
    proof_cost_micros: int | None

    def __post_init__(self) -> None:
        _status(self.status, "verification.status")
        _provenance(self.provenance, "verification.provenance")
        for field in ("evidence_cid", "proposal_cid", "verification_plan_cid"):
            object.__setattr__(self, field, _cid(getattr(self, field), f"verification.{field}"))
        object.__setattr__(
            self,
            "reuse_receipt_cid",
            _optional_cid(self.reuse_receipt_cid, "verification.reuse_receipt_cid"),
        )
        if self.reuse_outcome not in REUSE_OUTCOMES:
            raise ConfigurationCDError("reuse outcome is outside the frozen taxonomy")
        if self.reuse_outcome == "hit" and self.reuse_receipt_cid is None:
            raise ConfigurationCDError("reuse hits require an exact receipt CID")
        if self.reuse_outcome != "hit" and self.reuse_receipt_cid is not None:
            raise ConfigurationCDError("only an admitted reuse hit may carry a receipt CID")
        for field in (
            "selected_test_count",
            "selected_test_pass_count",
            "proof_selected_count",
            "proof_executed_count",
            "proof_pass_count",
            "proof_fail_count",
            "full_test_count",
            "full_test_pass_count",
            "hidden_test_total_count",
            "hidden_test_pass_count",
            "regression_count",
            "critical_regression_count",
            "out_of_scope_edit_count",
        ):
            _count(getattr(self, field), field)
        if self.selected_test_pass_count > self.selected_test_count:
            raise ConfigurationCDError("selected test passes exceed selected tests")
        if self.proof_executed_count > self.proof_selected_count:
            raise ConfigurationCDError("executed proofs exceed selected proofs")
        if self.proof_pass_count + self.proof_fail_count != self.proof_executed_count:
            raise ConfigurationCDError("proof outcomes do not equal executed proofs")
        if self.reuse_outcome == "hit" and (
            self.proof_selected_count == 0 or self.proof_executed_count >= self.proof_selected_count
        ):
            raise ConfigurationCDError(
                "reuse hits require at least one selected proof not freshly executed"
            )
        if self.reuse_outcome == "miss" and self.proof_executed_count != self.proof_selected_count:
            raise ConfigurationCDError("reuse misses require every selected proof to execute")
        if self.full_test_count < 1 or self.full_test_pass_count > self.full_test_count:
            raise ConfigurationCDError("hidden full scoring requires a non-empty bounded suite")
        if (
            self.hidden_test_total_count < 1
            or self.hidden_test_pass_count > self.hidden_test_total_count
        ):
            raise ConfigurationCDError("hidden scoring requires a non-empty bounded suite")
        _boolean(self.full_fallback_used, "full_fallback_used")
        _boolean(self.hidden_scoring_after_proposal, "hidden_scoring_after_proposal")
        _boolean(self.semantic_outcome_match, "semantic_outcome_match")
        _optional_count(self.verification_cost_micros, "verification_cost_micros")
        _optional_count(self.proof_cost_micros, "proof_cost_micros")

    @property
    def accepted(self) -> bool:
        return bool(
            self.status == "succeeded"
            and self.reuse_outcome in {"hit", "miss"}
            and self.hidden_scoring_after_proposal
            and self.selected_test_pass_count == self.selected_test_count
            and self.proof_fail_count == 0
            and self.full_test_pass_count == self.full_test_count
            and self.hidden_test_pass_count == self.hidden_test_total_count
            and self.regression_count == 0
            and self.critical_regression_count == 0
            and self.out_of_scope_edit_count == 0
            and self.semantic_outcome_match
        )


@dataclass(frozen=True)
class AssuranceDecision:
    status: str
    provenance: str
    evidence_cid: str
    accepted: bool
    self_approved: bool
    hidden_benchmark_exposed: bool
    mutant_count: int
    mutant_detected_count: int
    omission_mutant_count: int
    omission_mutant_detected_count: int
    vacuity_mutant_count: int
    vacuity_mutant_detected_count: int
    context_expansion_mutant_count: int
    context_expansion_mutant_detected_count: int
    critical_mutant_accepted_count: int
    sample_count: int
    failure_count: int
    assurance_cost_micros: int | None

    def __post_init__(self) -> None:
        _status(self.status, "assurance.status")
        _provenance(self.provenance, "assurance.provenance")
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "assurance.evidence_cid"))
        for field in ("accepted", "self_approved", "hidden_benchmark_exposed"):
            _boolean(getattr(self, field), field)
        for field in (
            "mutant_count",
            "mutant_detected_count",
            "omission_mutant_count",
            "omission_mutant_detected_count",
            "vacuity_mutant_count",
            "vacuity_mutant_detected_count",
            "context_expansion_mutant_count",
            "context_expansion_mutant_detected_count",
            "critical_mutant_accepted_count",
            "sample_count",
            "failure_count",
        ):
            _count(getattr(self, field), field)
        for detected, total in (
            (self.mutant_detected_count, self.mutant_count),
            (self.omission_mutant_detected_count, self.omission_mutant_count),
            (self.vacuity_mutant_detected_count, self.vacuity_mutant_count),
            (
                self.context_expansion_mutant_detected_count,
                self.context_expansion_mutant_count,
            ),
        ):
            if detected > total:
                raise ConfigurationCDError("detected assurance mutants exceed sampled mutants")
        _optional_count(self.assurance_cost_micros, "assurance_cost_micros")
        if self.accepted and self.status != "succeeded":
            raise ConfigurationCDError("only successful assurance may be accepted")


@dataclass(frozen=True)
class SealDecision:
    status: str
    provenance: str
    evidence_cid: str
    seal_cid: str | None
    parent_proposal_cid: str
    incremental: bool

    def __post_init__(self) -> None:
        _status(self.status, "seal.status")
        _provenance(self.provenance, "seal.provenance")
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "seal.evidence_cid"))
        object.__setattr__(self, "seal_cid", _optional_cid(self.seal_cid, "seal.seal_cid"))
        object.__setattr__(
            self,
            "parent_proposal_cid",
            _cid(self.parent_proposal_cid, "seal.parent_proposal_cid"),
        )
        _boolean(self.incremental, "seal.incremental")
        if self.status == "succeeded" and self.seal_cid is None:
            raise ConfigurationCDError("successful incremental sealing requires a seal CID")


@dataclass(frozen=True)
class DispositionDecision:
    status: str
    provenance: str
    evidence_cid: str
    human_review_required: bool
    human_review_performed: bool
    human_review_correct: bool | None
    autonomous_accept: bool
    human_cost_micros: int | None

    def __post_init__(self) -> None:
        _status(self.status, "disposition.status")
        _provenance(self.provenance, "disposition.provenance")
        object.__setattr__(
            self,
            "evidence_cid",
            _cid(self.evidence_cid, "disposition.evidence_cid"),
        )
        for field in (
            "human_review_required",
            "human_review_performed",
            "autonomous_accept",
        ):
            _boolean(getattr(self, field), field)
        _optional_boolean(self.human_review_correct, "human_review_correct")
        _optional_count(self.human_cost_micros, "human_cost_micros")
        if self.human_review_performed and self.human_review_correct is None:
            raise ConfigurationCDError("performed human review requires a correctness observation")
        if not self.human_review_performed and self.human_review_correct is not None:
            raise ConfigurationCDError("unperformed human review cannot claim correctness")


@dataclass(frozen=True)
class CDExecutionRequest:
    identity: PairIdentity
    task: TaskAgentView
    configuration_id: str
    configuration_cid: str
    configuration: Mapping[str, Any]
    context: SemanticContextPack
    permit: RouteExecutionPermit


@dataclass(frozen=True)
class CDExecutionObservation:
    configuration_id: str
    status: str
    provenance: str
    context: ContextDecision
    route: RouteDecision | None
    verification: VerificationDecision | None
    assurance: AssuranceDecision | None
    seal: SealDecision | None
    disposition: DispositionDecision | None
    stage_trace: tuple[StageEvidence, ...]
    provider_status: str | None
    provider_evidence_cid: str | None
    proposal_cid: str | None
    provider_call_count: int
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    provider_cached_input_tokens: int | None
    inference_cost_micros: int | None
    failure_cost_micros: int | None
    reason: str

    def __post_init__(self) -> None:
        if self.configuration_id not in CONFIGURATION_CIDS:
            raise ConfigurationCDError("observation configuration is not C or D")
        _status(self.status)
        _provenance(self.provenance)
        if type(self.context) is not ContextDecision:
            raise ConfigurationCDError("observation requires an exact context decision")
        for field, expected in (
            ("route", RouteDecision),
            ("verification", VerificationDecision),
            ("assurance", AssuranceDecision),
            ("seal", SealDecision),
            ("disposition", DispositionDecision),
        ):
            value = getattr(self, field)
            if value is not None and type(value) is not expected:
                raise ConfigurationCDError(f"{field} uses a noncanonical observation type")
        if not isinstance(self.stage_trace, tuple) or any(
            type(stage) is not StageEvidence for stage in self.stage_trace
        ):
            raise ConfigurationCDError("stage_trace must be an exact tuple of stage evidence")
        if self.provider_status is not None:
            _status(self.provider_status, "provider_status")
        object.__setattr__(
            self,
            "provider_evidence_cid",
            _optional_cid(self.provider_evidence_cid, "provider_evidence_cid"),
        )
        object.__setattr__(self, "proposal_cid", _optional_cid(self.proposal_cid, "proposal_cid"))
        _count(self.provider_call_count, "provider_call_count")
        if self.provider_call_count not in {0, 1}:
            raise ConfigurationCDError("one arm admits at most one provider call")
        for field in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_cached_input_tokens",
            "inference_cost_micros",
            "failure_cost_micros",
        ):
            _optional_count(getattr(self, field), field)
        _reason_code(self.reason, "reason")
        if self.provider_call_count == 0:
            if any(
                value is not None
                for value in (
                    self.provider_status,
                    self.provider_evidence_cid,
                    self.proposal_cid,
                    self.provider_input_tokens,
                    self.provider_output_tokens,
                    self.provider_cached_input_tokens,
                    self.inference_cost_micros,
                    self.failure_cost_micros,
                )
            ):
                raise ConfigurationCDError("non-dispatched observations cannot claim provider data")
        else:
            if self.provider_status is None or self.provider_evidence_cid is None:
                raise ConfigurationCDError(
                    "dispatched observations require provider status and evidence"
                )
            if self.provider_status == "succeeded":
                if self.proposal_cid is None:
                    raise ConfigurationCDError(
                        "successful provider dispatch requires a proposal CID"
                    )
                for field in (
                    "provider_input_tokens",
                    "provider_output_tokens",
                    "inference_cost_micros",
                ):
                    if getattr(self, field) is None:
                        raise ConfigurationCDError(
                            f"successful provider dispatch requires observed {field}"
                        )


@runtime_checkable
class ConfigurationCDPort(Protocol):
    def execute(self, request: CDExecutionRequest) -> CDExecutionObservation: ...


@dataclass(frozen=True)
class PairedCDRun:
    arm_c: BenchmarkRun
    arm_d: BenchmarkRun
    pairing_record: Mapping[str, Any]
    pairing_cid: str


def _revalidate_observation(observation: CDExecutionObservation) -> None:
    CDExecutionObservation.__post_init__(observation)
    ContextDecision.__post_init__(observation.context)
    for component in (
        observation.route,
        observation.verification,
        observation.assurance,
        observation.seal,
        observation.disposition,
    ):
        if component is not None:
            component.__post_init__()
    for stage in observation.stage_trace:
        StageEvidence.__post_init__(stage)


def _empty_metrics() -> dict[str, int | None]:
    return dict.fromkeys(METRIC_NAMES)


def _context_metrics(
    metrics: dict[str, int | None],
    context: SemanticContextPack,
    decision: ContextDecision | None,
) -> None:
    metrics["context_pack_tokens"] = context.token_count
    metrics["exact_source_tokens"] = context.exact_source_tokens
    metrics["capsule_tokens"] = context.capsule_tokens
    metrics["context_fallback_count"] = (
        decision.fallback_count if decision is not None else context.fallback_count
    )
    metrics["context_expansion_count"] = decision.expansion_count if decision is not None else 0
    metrics["context_expansion_tokens"] = decision.expansion_tokens if decision is not None else 0


def _missingness(metrics: Mapping[str, int | None], reason: str) -> dict[str, str]:
    return {name: reason for name, value in metrics.items() if value is None}


def _raw_result(
    *,
    identity: PairIdentity,
    configuration_id: str,
    provenance: str,
    terminal_status: str,
    metrics: Mapping[str, int | None],
    reason: str,
    evidence_cids: Sequence[str],
) -> Mapping[str, Any]:
    _status(terminal_status, "terminal_status")
    _provenance(provenance)
    _reason_code(reason, "missingness.reason")
    if set(metrics) != set(METRIC_NAMES):
        raise ConfigurationCDError("raw result metric vocabulary is not the frozen 78 fields")
    body = {
        "schema": RAW_RESULT_SCHEMA,
        "run_key": (
            f"{identity.task_id}/{configuration_id}/{identity.seed}/{identity.attempt}/"
            f"{identity.environment_cid}"
        ),
        "corpus_manifest_cid": identity.corpus_manifest_cid,
        "task_record_cid": identity.task_record_cid,
        "visible_projection_cid": identity.visible_projection_cid,
        "configuration_id": configuration_id,
        "configuration_cid": configuration_cid(configuration_id),
        "repository_state_cid": identity.repository_state_cid,
        "environment_cid": identity.environment_cid,
        "provider_id": identity.provider_id,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "seed": identity.seed,
        "attempt": identity.attempt,
        "provenance": provenance,
        "terminal_status": terminal_status,
        "metrics": dict(metrics),
        "missingness": _missingness(metrics, reason),
        "evidence_cids": list(dict.fromkeys(evidence_cids)),
    }
    return _freeze(body)


def _base_metrics(context: SemanticContextPack, task: TaskAgentView) -> dict[str, int | None]:
    metrics = _empty_metrics()
    _context_metrics(metrics, context, None)
    metrics["eligible_task_count"] = 1
    metrics["routine_localized_task_count"] = int(task.routine_localized)
    return metrics


def _undispatched_run(
    *,
    configuration_id: str,
    identity: PairIdentity,
    task: TaskAgentView,
    context: SemanticContextPack,
    permit: RouteExecutionPermit,
    terminal_status: str,
    reason: str,
    provider_call_count: int | None,
) -> BenchmarkRun:
    metrics = _base_metrics(context, task)
    metrics["provider_call_count"] = provider_call_count
    if terminal_status == "unavailable":
        metrics["route_unavailable_count"] = 1
    if terminal_status == "simulated":
        metrics["simulated_success_accepted_count"] = 0
    raw = _raw_result(
        identity=identity,
        configuration_id=configuration_id,
        provenance=permit.provenance,
        terminal_status=terminal_status,
        metrics=metrics,
        reason=reason,
        evidence_cids=(permit.permit_cid, context.pack_cid),
    )
    audit = _freeze(
        {
            "schema": RUNNER_SCHEMA,
            "configuration_id": configuration_id,
            "port_dispatched": provider_call_count is None,
            "provider_call_count": provider_call_count,
            "accepted": False,
            "reason": reason,
        }
    )
    return BenchmarkRun(configuration_id, raw, audit)


def _trace_defect(
    observation: CDExecutionObservation,
    identity: PairIdentity,
    configuration_id: str,
) -> str | None:
    expected = C_STAGES if configuration_id == "C" else GOVERNED_LIFECYCLE_STAGES
    stages = observation.stage_trace
    names = tuple(stage.stage for stage in stages)
    if not stages or names != expected[: len(stages)]:
        return "lifecycle-stage-order-or-prefix-invalid"
    if observation.status == "succeeded" and names != expected:
        return "successful-observation-omits-required-lifecycle-stage"
    for index, stage in enumerate(stages):
        if stage.identity != identity:
            return "lifecycle-stage-identity-mismatch"
        if stage.configuration_cid != configuration_cid(configuration_id):
            return "lifecycle-stage-configuration-mismatch"
        if stage.provenance != observation.provenance:
            return "lifecycle-stage-provenance-mismatch"
        if index < len(stages) - 1 and stage.status != "succeeded":
            return "lifecycle-continued-after-non-success-stage"
    if observation.status == "succeeded" and any(stage.status != "succeeded" for stage in stages):
        return "successful-observation-contains-non-success-stage"
    if observation.status != "succeeded" and stages[-1].status != observation.status:
        return "terminal-status-is-not-bound-to-final-stage"
    return None


def _binding_defect(
    observation: CDExecutionObservation,
    *,
    configuration_id: str,
    identity: PairIdentity,
    context: SemanticContextPack,
    permit: RouteExecutionPermit,
) -> str | None:
    if observation.configuration_id not in CONFIGURATION_CIDS:
        return "observation-configuration-invalid"
    if observation.context.pack_cid != context.pack_cid:
        return "context-pack-identity-mismatch"
    if observation.context.visible_projection_cid != identity.visible_projection_cid:
        return "context-visible-projection-mismatch"
    if observation.context.provenance != observation.provenance:
        return "context-provenance-mismatch"
    if observation.context.fallback_count != context.fallback_count:
        return "context-fallback-count-mismatch"
    if configuration_id == "C" and (
        observation.context.initial_sufficient is not None
        or observation.context.sufficient_after_expansion is not None
        or observation.context.expansion_count != 0
        or observation.context.expansion_tokens != 0
    ):
        return "configuration-c-observed-d-only-context-governance"
    route = observation.route
    if route is not None:
        if route.provenance != observation.provenance:
            return "route-provenance-mismatch"
        if route.route_policy_cid != permit.route_policy_cid:
            return "route-policy-identity-mismatch"
        if (
            route.provider_id != identity.provider_id
            or route.model_id != identity.model_id
            or route.model_revision != identity.model_revision
        ):
            return "route-provider-model-revision-mismatch"
    verification = observation.verification
    if verification is not None:
        if verification.provenance != observation.provenance:
            return "verification-provenance-mismatch"
        if (
            observation.proposal_cid is None
            or verification.proposal_cid != observation.proposal_cid
        ):
            return "verification-proposal-identity-mismatch"
    assurance = observation.assurance
    if assurance is not None and assurance.provenance != observation.provenance:
        return "assurance-provenance-mismatch"
    if assurance is not None and (assurance.self_approved or assurance.hidden_benchmark_exposed):
        return "assurance-is-self-approved-or-hidden-exposed"
    seal = observation.seal
    if seal is not None:
        if seal.provenance != observation.provenance:
            return "seal-provenance-mismatch"
        if observation.proposal_cid is None or seal.parent_proposal_cid != observation.proposal_cid:
            return "seal-parent-proposal-mismatch"
    disposition = observation.disposition
    if disposition is not None and disposition.provenance != observation.provenance:
        return "disposition-provenance-mismatch"
    if (
        disposition is not None
        and disposition.autonomous_accept
        and disposition.human_review_required
    ):
        return "human-review-case-was-autonomously-accepted"
    if configuration_id == "C" and any(
        component is not None
        for component in (observation.assurance, observation.seal, observation.disposition)
    ):
        return "configuration-c-observed-d-only-governance"
    return None


def _component_terminal(observation: CDExecutionObservation) -> str:
    components: list[tuple[str, str]] = [
        (observation.context.provenance, observation.context.status)
    ]
    for component in (
        observation.route,
        observation.verification,
        observation.assurance,
        observation.seal,
        observation.disposition,
    ):
        if component is not None:
            components.append((component.provenance, component.status))
    components.extend((stage.provenance, stage.status) for stage in observation.stage_trace)
    if observation.provenance == "simulated" or any(
        provenance == "simulated" or status == "simulated" for provenance, status in components
    ):
        return "simulated"
    if any(status == "stale" for _, status in components):
        return "stale"
    if observation.verification is not None:
        if observation.verification.reuse_outcome == "stale":
            return "stale"
        if observation.verification.reuse_outcome == "unavailable":
            return "unavailable"
    if observation.route is not None and observation.route.route == "unavailable":
        return "unavailable"
    for _, status in components:
        if status != "succeeded":
            return status
    if observation.provider_status is not None and observation.provider_status != "succeeded":
        return observation.provider_status
    return observation.status


def _effective_provenance(
    observation: CDExecutionObservation,
    permit: RouteExecutionPermit,
) -> str:
    provenances = [permit.provenance, observation.provenance, observation.context.provenance]
    provenances.extend(
        component.provenance
        for component in (
            observation.route,
            observation.verification,
            observation.assurance,
            observation.seal,
            observation.disposition,
        )
        if component is not None
    )
    provenances.extend(stage.provenance for stage in observation.stage_trace)
    if "simulated" in provenances:
        return "simulated"
    if "replayed" in provenances:
        return "replayed"
    return "live"


def _success_defect(observation: CDExecutionObservation, configuration_id: str) -> str | None:
    if observation.status != "succeeded":
        return None
    if observation.context.status != "succeeded":
        return "successful-run-has-unsuccessful-context"
    if observation.route is None or observation.route.status != "succeeded":
        return "successful-run-omits-successful-route"
    if observation.route.route in {"human", "unavailable"}:
        return "successful-run-uses-non-provider-route"
    if (
        observation.provider_call_count != 1
        or observation.provider_status != "succeeded"
        or observation.proposal_cid is None
    ):
        return "successful-run-omits-provider-proposal"
    verification = observation.verification
    if verification is None or not verification.accepted:
        return "successful-run-omits-admitted-incremental-verification"
    if (
        verification.selected_test_count + verification.proof_selected_count == 0
        and not verification.full_fallback_used
    ):
        return "successful-run-silently-omits-incremental-verification"
    if verification.reuse_outcome not in {"hit", "miss"}:
        return "successful-run-uses-stale-or-unavailable-reuse"
    if not verification.hidden_scoring_after_proposal:
        return "hidden-full-scoring-ran-before-proposal"
    if configuration_id == "C":
        if observation.context.initial_sufficient is not None:
            return "configuration-c-cannot-run-sufficiency"
        if observation.context.expansion_count != 0 or observation.context.expansion_tokens != 0:
            return "configuration-c-cannot-expand-context"
        if any(
            component is not None
            for component in (observation.assurance, observation.seal, observation.disposition)
        ):
            return "configuration-c-cannot-run-d-only-governance"
        return None
    if observation.context.initial_sufficient is None:
        return "configuration-d-requires-a-sufficiency-decision"
    if observation.context.initial_sufficient is False:
        if (
            observation.context.expansion_count < 1
            or observation.context.expansion_tokens < 1
            or observation.context.sufficient_after_expansion is not True
        ):
            return "configuration-d-insufficient-context-requires-successful-expansion"
    elif observation.context.expansion_count != 0:
        return "configuration-d-expanded-an-already-sufficient-context"
    assurance = observation.assurance
    if assurance is None or assurance.status != "succeeded" or not assurance.accepted:
        return "configuration-d-requires-successful-assurance"
    if assurance.sample_count < 1:
        return "configuration-d-requires-observed-assurance-sampling"
    if assurance.self_approved or assurance.hidden_benchmark_exposed:
        return "configuration-d-assurance-is-self-approved-or-hidden-exposed"
    if assurance.critical_mutant_accepted_count != 0 or assurance.failure_count != 0:
        return "configuration-d-assurance-accepted-a-critical-failure"
    seal = observation.seal
    if seal is None or seal.status != "succeeded" or not seal.incremental:
        return "configuration-d-requires-a-successful-incremental-seal"
    disposition = observation.disposition
    if disposition is None or disposition.status != "succeeded":
        return "configuration-d-requires-a-terminal-disposition"
    if disposition.autonomous_accept and disposition.human_review_required:
        return "configuration-d-autonomously-accepted-a-human-review-case"
    if disposition.human_review_required and not disposition.human_review_performed:
        return "configuration-d-omits-required-human-review"
    if disposition.human_review_performed and disposition.human_review_correct is not True:
        return "configuration-d-cannot-accept-an-incorrect-human-review"
    return None


def _evidence_cids(
    observation: CDExecutionObservation,
    permit: RouteExecutionPermit,
    context: SemanticContextPack,
) -> tuple[str, ...]:
    values: list[str] = [permit.permit_cid, permit.route_policy_cid, context.pack_cid]
    values.append(observation.context.evidence_cid)
    for component in (
        observation.route,
        observation.verification,
        observation.assurance,
        observation.seal,
        observation.disposition,
    ):
        if component is not None:
            values.append(component.evidence_cid)
    if observation.verification is not None:
        values.append(observation.verification.verification_plan_cid)
        if observation.verification.reuse_receipt_cid is not None:
            values.append(observation.verification.reuse_receipt_cid)
    if observation.seal is not None and observation.seal.seal_cid is not None:
        values.append(observation.seal.seal_cid)
    if observation.provider_evidence_cid is not None:
        values.append(observation.provider_evidence_cid)
    if observation.proposal_cid is not None:
        values.append(observation.proposal_cid)
    values.extend(stage.evidence_cid for stage in observation.stage_trace)
    return tuple(dict.fromkeys(values))


def _observed_metrics(
    observation: CDExecutionObservation,
    *,
    terminal_status: str,
    configuration_id: str,
    identity: PairIdentity,
    task: TaskAgentView,
    context: SemanticContextPack,
) -> tuple[dict[str, int | None], bool]:
    metrics = _base_metrics(context, task)
    _context_metrics(metrics, context, observation.context)
    route = observation.route
    if route is not None:
        route_metric = {
            "small": "route_small_count",
            "local": "route_local_count",
            "frontier": "route_frontier_count",
            "human": "route_human_count",
            "unavailable": "route_unavailable_count",
        }[route.route]
        metrics[route_metric] = 1
        metrics["route_failure_count"] = int(route.status not in {"succeeded", "unavailable"})
        metrics["model_escalation_count"] = int(route.decision_kind == "escalated")
        metrics["frontier_escalation_count"] = int(route.route == "frontier")
        if task.routine_localized:
            metrics["routine_frontier_escalation_rate_bp"] = (
                10000 if route.route == "frontier" else 0
            )
    metrics["provider_call_count"] = observation.provider_call_count
    metrics["provider_input_tokens"] = observation.provider_input_tokens
    metrics["provider_output_tokens"] = observation.provider_output_tokens
    metrics["provider_cached_input_tokens"] = observation.provider_cached_input_tokens
    metrics["inference_cost_micros"] = observation.inference_cost_micros
    metrics["failure_cost_micros"] = observation.failure_cost_micros
    metrics["patch_proposal_count"] = int(observation.proposal_cid is not None)

    verification = observation.verification
    if verification is not None:
        metrics.update(
            {
                "selected_test_count": verification.selected_test_count,
                "selected_test_pass_count": verification.selected_test_pass_count,
                "selected_test_fail_count": (
                    verification.selected_test_count - verification.selected_test_pass_count
                ),
                "proof_selected_count": verification.proof_selected_count,
                "proof_executed_count": verification.proof_executed_count,
                "proof_pass_count": verification.proof_pass_count,
                "proof_fail_count": verification.proof_fail_count,
                "verification_reuse_hit_count": int(verification.reuse_outcome == "hit"),
                "verification_reuse_miss_count": int(verification.reuse_outcome == "miss"),
                "verification_full_fallback_count": int(verification.full_fallback_used),
                "stale_proof_rejected_count": int(verification.reuse_outcome == "stale"),
                "stale_proof_accepted_count": 0,
                "full_test_count": verification.full_test_count,
                "full_test_pass_count": verification.full_test_pass_count,
                "full_test_fail_count": (
                    verification.full_test_count - verification.full_test_pass_count
                ),
                "hidden_test_total_count": verification.hidden_test_total_count,
                "hidden_test_pass_count": verification.hidden_test_pass_count,
                "regression_count": verification.regression_count,
                "out_of_scope_edit_count": verification.out_of_scope_edit_count,
                "semantic_outcome_match_count": int(verification.semantic_outcome_match),
                "controlled_selected_test_false_negative_count": int(
                    verification.selected_test_count == verification.selected_test_pass_count
                    and verification.full_test_pass_count != verification.full_test_count
                ),
                "verification_cost_micros": verification.verification_cost_micros,
                "proof_cost_micros": verification.proof_cost_micros,
            }
        )
    if observation.context.status == "stale":
        metrics["stale_capsule_rejected_count"] = 1
        metrics["stale_capsule_accepted_count"] = 0
    elif observation.context.status == "succeeded":
        metrics["stale_capsule_rejected_count"] = 0
        metrics["stale_capsule_accepted_count"] = 0

    assurance = observation.assurance
    if assurance is not None:
        metrics.update(
            {
                "assurance_mutant_count": assurance.mutant_count,
                "assurance_mutant_detected_count": assurance.mutant_detected_count,
                "assurance_mutant_survivor_count": (
                    assurance.mutant_count - assurance.mutant_detected_count
                ),
                "omission_mutant_count": assurance.omission_mutant_count,
                "omission_mutant_detected_count": assurance.omission_mutant_detected_count,
                "vacuity_mutant_count": assurance.vacuity_mutant_count,
                "vacuity_mutant_detected_count": assurance.vacuity_mutant_detected_count,
                "context_expansion_mutant_count": assurance.context_expansion_mutant_count,
                "context_expansion_mutant_detected_count": (
                    assurance.context_expansion_mutant_detected_count
                ),
                "critical_mutant_accepted_count": assurance.critical_mutant_accepted_count,
                "assurance_sample_count": assurance.sample_count,
                "assurance_failure_count": assurance.failure_count,
                "assurance_cost_micros": assurance.assurance_cost_micros,
            }
        )
    elif configuration_id == "C" and observation.provider_call_count == 1:
        metrics["assurance_cost_micros"] = 0

    disposition = observation.disposition
    if disposition is not None:
        metrics["human_review_required_count"] = int(disposition.human_review_required)
        metrics["human_review_correct_count"] = int(
            disposition.human_review_performed and disposition.human_review_correct is True
        )
        metrics["negative_review_autonomous_accept_count"] = int(
            disposition.autonomous_accept and disposition.human_review_required
        )
        metrics["human_cost_micros"] = disposition.human_cost_micros
    elif configuration_id == "C" and observation.provider_call_count == 1:
        metrics["human_cost_micros"] = 0

    observed_stages = {stage.stage for stage in observation.stage_trace}
    if observation.provider_call_count == 1:
        if verification is None and "incremental-verify" not in observed_stages:
            metrics["verification_cost_micros"] = 0
            metrics["proof_cost_micros"] = 0
        if configuration_id == "D":
            if assurance is None and "assurance" not in observed_stages:
                metrics["assurance_cost_micros"] = 0
            if disposition is None and "disposition" not in observed_stages:
                metrics["human_cost_micros"] = 0

    accepted = bool(
        terminal_status == "succeeded"
        and verification is not None
        and verification.accepted
        and (
            configuration_id == "C"
            or (
                assurance is not None
                and assurance.accepted
                and observation.seal is not None
                and observation.seal.status == "succeeded"
                and disposition is not None
                and disposition.status == "succeeded"
            )
        )
    )
    if verification is not None:
        metrics["accepted_patch_count"] = int(accepted)
        metrics["correct_accepted_patch_count"] = int(accepted)
        metrics["critical_regression_accepted_count"] = (
            verification.critical_regression_count if accepted else 0
        )
        metrics["first_attempt_success_count"] = int(accepted and identity.attempt == 1)
        metrics["correct_accepted_patch_rate_bp"] = 10000 if accepted else 0
    if terminal_status == "simulated":
        metrics["simulated_success_accepted_count"] = 0

    if observation.provider_call_count == 1 and observation.provider_status != "succeeded":
        metrics["verification_cost_micros"] = 0
        metrics["proof_cost_micros"] = 0
        metrics["assurance_cost_micros"] = 0
        metrics["human_cost_micros"] = 0

    component_costs = (
        metrics["inference_cost_micros"],
        metrics["verification_cost_micros"],
        metrics["proof_cost_micros"],
        metrics["assurance_cost_micros"],
        metrics["failure_cost_micros"],
        metrics["human_cost_micros"],
    )
    if all(value is not None for value in component_costs):
        total = sum(value for value in component_costs if value is not None)
        metrics["total_cost_micros"] = total
        metrics["failed_attempt_cost_micros"] = 0 if accepted else total
        if accepted:
            metrics["cost_per_correct_accepted_patch_micros"] = total
    return metrics, accepted


def run_configuration(
    *,
    configuration_id: str,
    identity: PairIdentity,
    task: TaskAgentView,
    context: SemanticContextPack,
    permit: RouteExecutionPermit,
    port: ConfigurationCDPort,
) -> BenchmarkRun:
    """Run one frozen C/D arm through an explicitly injected execution port."""

    descriptor = configuration_descriptor(configuration_id)
    if type(identity) is not PairIdentity:
        raise ConfigurationCDError("identity must use the exact pair identity type")
    if type(task) is not TaskAgentView:
        raise HiddenDataDenied("task must use the exact provider-visible projection type")
    if type(context) is not SemanticContextPack:
        raise HiddenDataDenied("C/D require an exact datasets-owned semantic ContextPack")
    if type(permit) is not RouteExecutionPermit:
        raise ConfigurationCDError("permit must use the exact route execution permit type")
    PairIdentity.__post_init__(identity)
    TaskAgentView.__post_init__(task)
    SemanticContextPack.__post_init__(context)
    RouteExecutionPermit.__post_init__(permit)
    if context.visible_projection_cid != identity.visible_projection_cid:
        raise HiddenDataDenied("ContextPack is bound to a different visible projection")
    if not permit.admits(identity, configuration_id):
        return _undispatched_run(
            configuration_id=configuration_id,
            identity=identity,
            task=task,
            context=context,
            permit=permit,
            terminal_status="unavailable",
            reason="exact-pcce-067-route-execution-permit-unavailable",
            provider_call_count=0,
        )

    request = CDExecutionRequest(
        identity=replace(identity),
        task=replace(task),
        configuration_id=configuration_id,
        configuration_cid=configuration_cid(configuration_id),
        configuration=_freeze(descriptor),
        context=replace(context),
        permit=replace(permit),
    )
    try:
        observation = port.execute(request)
    except ExecutionPortUnavailable:
        return _undispatched_run(
            configuration_id=configuration_id,
            identity=identity,
            task=task,
            context=context,
            permit=permit,
            terminal_status="unavailable",
            reason="configuration-cd-execution-port-unavailable",
            provider_call_count=None,
        )
    except Exception:
        return _undispatched_run(
            configuration_id=configuration_id,
            identity=identity,
            task=task,
            context=context,
            permit=permit,
            terminal_status="infrastructure_failure",
            reason="configuration-cd-execution-port-exception-cost-unknown",
            provider_call_count=None,
        )
    if type(observation) is not CDExecutionObservation:
        return _undispatched_run(
            configuration_id=configuration_id,
            identity=identity,
            task=task,
            context=context,
            permit=permit,
            terminal_status="invalid",
            reason="configuration-cd-port-returned-noncanonical-observation",
            provider_call_count=None,
        )
    try:
        _revalidate_observation(observation)
    except (ConfigurationCDError, TypeError, ValueError):
        return _undispatched_run(
            configuration_id=configuration_id,
            identity=identity,
            task=task,
            context=context,
            permit=permit,
            terminal_status="invalid",
            reason="configuration-cd-port-returned-mutated-or-malformed-observation",
            provider_call_count=None,
        )

    defect = None
    if observation.configuration_id != configuration_id:
        defect = "observation-configuration-mismatch"
    elif observation.provenance != permit.provenance:
        defect = "observation-permit-provenance-mismatch"
    if defect is None:
        defect = _trace_defect(observation, identity, configuration_id)
    if defect is None:
        defect = _binding_defect(
            observation,
            configuration_id=configuration_id,
            identity=identity,
            context=context,
            permit=permit,
        )
    if defect is None:
        defect = _success_defect(observation, configuration_id)

    component_terminal = _component_terminal(observation)
    effective_provenance = _effective_provenance(observation, permit)
    if effective_provenance == "simulated":
        terminal_status = "simulated"
    elif component_terminal in {"stale", "simulated", "unavailable"}:
        terminal_status = component_terminal
    else:
        terminal_status = "invalid" if defect is not None else component_terminal
    metrics, accepted = _observed_metrics(
        observation,
        terminal_status=terminal_status,
        configuration_id=configuration_id,
        identity=identity,
        task=task,
        context=context,
    )
    reason = defect or observation.reason
    raw = _raw_result(
        identity=identity,
        configuration_id=configuration_id,
        provenance=effective_provenance,
        terminal_status=terminal_status,
        metrics=metrics,
        reason=reason,
        evidence_cids=_evidence_cids(observation, permit, context),
    )
    audit = _freeze(
        {
            "schema": RUNNER_SCHEMA,
            "configuration_id": configuration_id,
            "configuration_cid": configuration_cid(configuration_id),
            "identity_profile": IDENTITY_PROFILE,
            "permit_provenance": permit.provenance,
            "observation_provenance": observation.provenance,
            "effective_provenance": effective_provenance,
            "port_dispatched": True,
            "provider_call_count": observation.provider_call_count,
            "route": observation.route.route if observation.route is not None else None,
            "route_decision_kind": (
                observation.route.decision_kind if observation.route is not None else None
            ),
            "reuse_outcome": (
                observation.verification.reuse_outcome
                if observation.verification is not None
                else None
            ),
            "full_fallback_used": (
                observation.verification.full_fallback_used
                if observation.verification is not None
                else None
            ),
            "hidden_scoring_phase": (
                "after-patch-proposal"
                if observation.verification is not None
                and observation.verification.hidden_scoring_after_proposal
                else None
            ),
            "lifecycle_trace": [stage.as_dict() for stage in observation.stage_trace],
            "accepted": accepted,
            "defect": defect,
            "reason": reason,
        }
    )
    return BenchmarkRun(configuration_id, raw, audit)


def run_paired_cd(
    *,
    identity_c: PairIdentity,
    identity_d: PairIdentity,
    task_c: TaskAgentView,
    task_d: TaskAgentView,
    context_c: SemanticContextPack,
    context_d: SemanticContextPack,
    permit_c: RouteExecutionPermit,
    permit_d: RouteExecutionPermit,
    port_c: ConfigurationCDPort,
    port_d: ConfigurationCDPort,
) -> PairedCDRun:
    """Run a C/D pair with exact held-constant inputs and visible treatment."""

    if type(permit_c) is not RouteExecutionPermit or type(permit_d) is not RouteExecutionPermit:
        raise ConfigurationCDError("paired permits must use the exact route permit type")
    RouteExecutionPermit.__post_init__(permit_c)
    RouteExecutionPermit.__post_init__(permit_d)
    if (
        permit_c.configuration_cid != CONFIGURATION_C_CID
        or permit_d.configuration_cid != CONFIGURATION_D_CID
    ):
        raise ConfigurationCDError("paired permits do not bind their exact C/D configurations")
    if identity_c != identity_d:
        raise ConfigurationCDError("paired C/D identities differ")
    if task_c != task_d:
        raise ConfigurationCDError("paired C/D provider-visible task projections differ")
    if context_c != context_d:
        raise ConfigurationCDError("paired C/D initial semantic ContextPacks differ")
    held_permit_c = (
        permit_c.route_policy_cid,
        permit_c.corpus_manifest_cid,
        permit_c.task_record_cid,
        permit_c.visible_projection_cid,
        permit_c.repository_state_cid,
        permit_c.environment_cid,
        permit_c.provider_id,
        permit_c.model_id,
        permit_c.model_revision,
        permit_c.task_id,
        permit_c.seed,
        permit_c.attempt,
        permit_c.provenance,
        permit_c.available,
        permit_c.live_execution_eligible,
    )
    held_permit_d = (
        permit_d.route_policy_cid,
        permit_d.corpus_manifest_cid,
        permit_d.task_record_cid,
        permit_d.visible_projection_cid,
        permit_d.repository_state_cid,
        permit_d.environment_cid,
        permit_d.provider_id,
        permit_d.model_id,
        permit_d.model_revision,
        permit_d.task_id,
        permit_d.seed,
        permit_d.attempt,
        permit_d.provenance,
        permit_d.available,
        permit_d.live_execution_eligible,
    )
    if held_permit_c != held_permit_d:
        raise ConfigurationCDError("paired C/D permits change a held-constant field")
    arm_c = run_configuration(
        configuration_id="C",
        identity=identity_c,
        task=task_c,
        context=context_c,
        permit=permit_c,
        port=port_c,
    )
    arm_d = run_configuration(
        configuration_id="D",
        identity=identity_d,
        task=task_d,
        context=context_d,
        permit=permit_d,
        port=port_d,
    )
    pair_body = {
        "schema": PAIR_SCHEMA,
        "identity": dict(identity_c.as_mapping()),
        "configuration_c_cid": CONFIGURATION_C_CID,
        "configuration_d_cid": CONFIGURATION_D_CID,
        "arm_c_result_cid": arm_c.result_cid,
        "arm_d_result_cid": arm_d.result_cid,
        "only_treatment_differences": [
            "assurance_enabled",
            "context_expansion_enabled",
            "human_escalation_enabled",
            "incremental_seal_enabled",
            "sufficiency_enabled",
        ],
    }
    return PairedCDRun(
        arm_c=arm_c,
        arm_d=arm_d,
        pairing_record=_freeze(pair_body),
        pairing_cid=cid_for_obj(pair_body, codec="dag-json"),
    )


def runner_descriptor() -> dict[str, Any]:
    return {
        "schema": RUNNER_SCHEMA,
        "identity_profile": IDENTITY_PROFILE,
        "configuration_cids": dict(CONFIGURATION_CIDS),
        "configuration_c_stages": list(C_STAGES),
        "configuration_d_stages": list(GOVERNED_LIFECYCLE_STAGES),
        "provider_binding": "none-injected-configuration-cd-port-only",
        "live_dispatch": "exact-pcce-067-permit-only-pcce-056-no-go-preserved",
        "hidden_projection": "datasets-context-pack-plus-exact-task-view-no-hidden-bodies",
        "route_policy": "exact-visible-decision-no-silent-frontier-fallback",
        "verification_policy": "incremental-reuse-plus-post-proposal-hidden-full-scoring",
        "full_fallback_policy": "allowed-only-when-explicitly-counted",
        "configuration_d_policy": (
            "exact-governed-lifecycle-sufficiency-expansion-assurance-seal-human"
        ),
        "failure_policy": "stale-simulated-unavailable-invalid-never-accepted",
        "cost_policy": "observed-components-retained-unknowns-null-never-imputed-zero",
    }


RUNNER_DESCRIPTOR_CID: Final[str] = cid_for_obj(runner_descriptor(), codec="dag-json")

__all__ = [
    "AssuranceDecision",
    "CDExecutionObservation",
    "CDExecutionRequest",
    "CONFIGURATION_C_CID",
    "CONFIGURATION_CIDS",
    "CONFIGURATION_D_CID",
    "C_STAGES",
    "ConfigurationCDError",
    "ConfigurationCDPort",
    "ContextDecision",
    "DispositionDecision",
    "ExecutionPortUnavailable",
    "HiddenDataDenied",
    "PairedCDRun",
    "RUNNER_DESCRIPTOR_CID",
    "RouteDecision",
    "RouteExecutionPermit",
    "SealDecision",
    "StageEvidence",
    "VerificationDecision",
    "configuration_cid",
    "configuration_descriptor",
    "run_configuration",
    "run_paired_cd",
    "runner_descriptor",
]
