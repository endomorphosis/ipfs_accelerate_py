"""Frozen benchmark configurations A and B (PCCE-064).

The runner in this module is deliberately provider-neutral and side-effect free.
Provider proposal and full-verification operations are injected ports.  A live
port is never called without an explicit, eligible execution permit; replay and
simulation remain visibly labelled in every result.

Configuration A uses ordinary visible retrieval.  Configuration B changes only
the context method to the frozen semantic ContextPack contract.  Both use the
same exact frontier model and full verification, and neither enables routing,
incremental verification, proof reuse, or hidden-data access by the provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_obj, validate_cid

CONFIGURATION_SCHEMA: Final[str] = "ipfs-datasets.proof-context.benchmark-configuration@1"
RAW_RESULT_SCHEMA: Final[str] = "ipfs-datasets.proof-context.benchmark-raw-result@1"
RUNNER_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.benchmark-configurations-ab@1"
PAIR_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.benchmark-pair-ab@1"
IDENTITY_PROFILE: Final[str] = "cidv1-base32-raw-or-dag-json-sha2-256"

PROVENANCE_CLASSES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
PROVIDER_TERMINAL_STATUSES: Final[tuple[str, ...]] = (
    "succeeded",
    "rejected",
    "unavailable",
    "timeout",
    "cancelled",
    "invalid",
    "stale",
    "simulated",
    "infrastructure_failure",
)
RESULT_TERMINAL_STATUSES: Final[tuple[str, ...]] = (
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

_COMMON_CONFIGURATION: Final[dict[str, Any]] = {
    "schema": CONFIGURATION_SCHEMA,
    "hidden_full_scoring": True,
    "prompt_policy": "pcce-benchmark-prompt-policy-v1",
    "seed_policy": "sha256-corpus-task-first-unsigned-64-v1",
    "environment_policy": "exact-pcce-056-qualified-environment-or-unavailable",
    "context_estimator": "utf8-bytes-ceiling-divide-by-four-no-calibration@1",
}

_CONFIGURATION_A: Final[dict[str, Any]] = {
    **_COMMON_CONFIGURATION,
    "configuration_id": "A",
    "context_method": "ordinary-lexical-raw-retrieval@1",
    "model_policy": "execution-permit-exact-frontier-pair@1",
    "verification_policy": "full-runtime-verification@1",
    "routing_enabled": False,
    "incremental_verification_enabled": False,
    "proof_reuse_enabled": False,
    "sufficiency_enabled": False,
    "context_expansion_enabled": False,
    "assurance_enabled": False,
    "incremental_seal_enabled": False,
    "human_escalation_enabled": False,
}

_CONFIGURATION_B: Final[dict[str, Any]] = {
    **_COMMON_CONFIGURATION,
    "configuration_id": "B",
    "context_method": "semantic-context-pack-v0.1",
    "model_policy": "execution-permit-exact-frontier-pair@1",
    "verification_policy": "full-runtime-verification@1",
    "routing_enabled": False,
    "incremental_verification_enabled": False,
    "proof_reuse_enabled": False,
    "sufficiency_enabled": False,
    "context_expansion_enabled": False,
    "assurance_enabled": False,
    "incremental_seal_enabled": False,
    "human_escalation_enabled": False,
}

CONFIGURATION_A_CID: Final[str] = cid_for_obj(_CONFIGURATION_A, codec="dag-json")
CONFIGURATION_B_CID: Final[str] = cid_for_obj(_CONFIGURATION_B, codec="dag-json")
CONFIGURATION_CIDS: Final[Mapping[str, str]] = MappingProxyType(
    {"A": CONFIGURATION_A_CID, "B": CONFIGURATION_B_CID}
)

# The frozen PCCE-060 raw-result schema requires the complete metric vocabulary.
# An arm records only observations it owns; every other metric is explicit null
# with a typed missingness reason.  PCCE-066 performs cross-arm aggregation.
METRIC_NAMES: Final[tuple[str, ...]] = (
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_cached_input_tokens",
    "ordinary_retrieval_tokens",
    "context_pack_tokens",
    "exact_source_tokens",
    "capsule_tokens",
    "context_expansion_tokens",
    "context_fallback_count",
    "context_expansion_count",
    "context_reduction_bp",
    "eligible_task_count",
    "patch_proposal_count",
    "accepted_patch_count",
    "correct_accepted_patch_count",
    "hidden_test_pass_count",
    "hidden_test_total_count",
    "regression_count",
    "critical_regression_accepted_count",
    "out_of_scope_edit_count",
    "human_review_required_count",
    "human_review_correct_count",
    "first_attempt_success_count",
    "semantic_outcome_match_count",
    "correct_accepted_patch_rate_bp",
    "route_small_count",
    "route_local_count",
    "route_frontier_count",
    "route_human_count",
    "route_unavailable_count",
    "model_escalation_count",
    "frontier_escalation_count",
    "routine_localized_task_count",
    "route_failure_count",
    "routine_frontier_escalation_rate_bp",
    "selected_test_count",
    "selected_test_pass_count",
    "selected_test_fail_count",
    "full_test_count",
    "full_test_pass_count",
    "full_test_fail_count",
    "controlled_selected_test_false_negative_count",
    "proof_selected_count",
    "proof_executed_count",
    "proof_pass_count",
    "proof_fail_count",
    "verification_reuse_hit_count",
    "verification_reuse_miss_count",
    "verification_full_fallback_count",
    "stale_capsule_rejected_count",
    "stale_proof_rejected_count",
    "stale_capsule_accepted_count",
    "stale_proof_accepted_count",
    "simulated_success_accepted_count",
    "assurance_mutant_count",
    "assurance_mutant_detected_count",
    "assurance_mutant_survivor_count",
    "omission_mutant_count",
    "omission_mutant_detected_count",
    "vacuity_mutant_count",
    "vacuity_mutant_detected_count",
    "context_expansion_mutant_count",
    "context_expansion_mutant_detected_count",
    "critical_mutant_accepted_count",
    "negative_review_autonomous_accept_count",
    "assurance_sample_count",
    "assurance_failure_count",
    "provider_call_count",
    "inference_cost_micros",
    "verification_cost_micros",
    "proof_cost_micros",
    "assurance_cost_micros",
    "failure_cost_micros",
    "human_cost_micros",
    "total_cost_micros",
    "failed_attempt_cost_micros",
    "cost_per_correct_accepted_patch_micros",
    "total_cost_reduction_bp",
)

_PAIR_FIELDS: Final[tuple[str, ...]] = (
    "corpus_manifest_cid",
    "task_record_cid",
    "visible_projection_cid",
    "repository_state_cid",
    "environment_cid",
    "task_id",
    "provider_id",
    "model_id",
    "model_revision",
    "seed",
    "attempt",
)
_DENIED_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hidden",
        "_hidden",
        "answers",
        "answer",
        "expected",
        "evaluator",
        "sealed-evaluator",
        "sealed_evaluator",
    }
)


class ConfigurationABError(ValueError):
    """Raised when a frozen A/B invariant would be weakened."""


class HiddenDataDenied(ConfigurationABError):
    """Raised before provider dispatch when hidden data enters its projection."""


class ProviderUnavailable(RuntimeError):
    """Provider-port signal that remains an unavailable benchmark observation."""


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _cid(value: Any, field: str) -> str:
    try:
        return validate_cid(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationABError(f"{field} must be a canonical CID") from exc


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationABError(f"{field} must be a non-empty string")
    return value


def _count(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationABError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_count(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _count(value, field)


def _safe_path(value: Any, field: str) -> str:
    text = _nonempty(value, field)
    if text.startswith("/") or "\\" in text or "\x00" in text:
        raise HiddenDataDenied(f"{field} is not a safe visible path")
    parts = tuple(part for part in text.split("/") if part not in {"", "."})
    lowered = {part.lower() for part in parts}
    if ".." in parts or lowered & _DENIED_PATH_PARTS:
        raise HiddenDataDenied(f"{field} enters a hidden or evaluator namespace")
    return "/".join(parts)


def estimate_context_tokens(text: str) -> int:
    """Apply the exact frozen, deliberately uncalibrated context estimator."""

    if not isinstance(text, str):
        raise ConfigurationABError("context must be UTF-8 text")
    size = len(text.encode("utf-8"))
    return (size + 3) // 4


def configuration_descriptor(configuration_id: str) -> dict[str, Any]:
    """Return a mutable copy of one exact frozen descriptor."""

    if configuration_id == "A":
        return _thaw(_CONFIGURATION_A)
    if configuration_id == "B":
        return _thaw(_CONFIGURATION_B)
    raise ConfigurationABError("only frozen configurations A and B are admitted")


def configuration_cid(configuration_id: str) -> str:
    try:
        return CONFIGURATION_CIDS[configuration_id]
    except (KeyError, TypeError) as exc:
        raise ConfigurationABError("only frozen configurations A and B are admitted") from exc


@dataclass(frozen=True)
class PairIdentity:
    """Fields that must be byte-identical across paired A and B arms."""

    corpus_manifest_cid: str
    task_record_cid: str
    visible_projection_cid: str
    repository_state_cid: str
    environment_cid: str
    task_id: str
    provider_id: str
    model_id: str
    model_revision: str
    seed: int
    attempt: int = 1

    def __post_init__(self) -> None:
        for field in (
            "corpus_manifest_cid",
            "task_record_cid",
            "visible_projection_cid",
            "repository_state_cid",
            "environment_cid",
        ):
            object.__setattr__(self, field, _cid(getattr(self, field), field))
        for field in ("task_id", "provider_id", "model_id", "model_revision"):
            _nonempty(getattr(self, field), field)
        _count(self.seed, "seed")
        _count(self.attempt, "attempt", minimum=1)

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType({field: getattr(self, field) for field in _PAIR_FIELDS})


@dataclass(frozen=True)
class TaskAgentView:
    """Provider-visible task projection; evaluator/answer fields do not exist."""

    objective: str
    owned_paths: tuple[str, ...]
    routine_localized: bool
    risk_class: str

    def __post_init__(self) -> None:
        _nonempty(self.objective, "objective")
        if not isinstance(self.owned_paths, tuple) or not self.owned_paths:
            raise ConfigurationABError("owned_paths must be a non-empty tuple")
        safe = tuple(_safe_path(path, "owned_paths[]") for path in self.owned_paths)
        if len(safe) != len(set(safe)):
            raise ConfigurationABError("owned_paths must be unique")
        object.__setattr__(self, "owned_paths", safe)
        if type(self.routine_localized) is not bool:
            raise ConfigurationABError("routine_localized must be a boolean")
        _nonempty(self.risk_class, "risk_class")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskAgentView:
        fields = {"objective", "owned_paths", "routine_localized", "risk_class"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise HiddenDataDenied("task provider projection has missing or unknown fields")
        paths = value["owned_paths"]
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
            raise ConfigurationABError("owned_paths must be an array")
        return cls(
            objective=value["objective"],
            owned_paths=tuple(paths),
            routine_localized=value["routine_localized"],
            risk_class=value["risk_class"],
        )

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "objective": self.objective,
                "owned_paths": self.owned_paths,
                "routine_localized": self.routine_localized,
                "risk_class": self.risk_class,
            }
        )


@dataclass(frozen=True)
class ContextChunk:
    """One exact visible source chunk used by ordinary retrieval."""

    path: str
    text: str
    content_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_path(self.path, "chunk.path"))
        if not isinstance(self.text, str):
            raise ConfigurationABError("chunk.text must be UTF-8 text")
        expected = cid_for_bytes(self.text.encode("utf-8"), codec="raw")
        if _cid(self.content_cid, "chunk.content_cid") != expected:
            raise ConfigurationABError("chunk bytes do not match content_cid")


@dataclass(frozen=True)
class OrdinaryRetrieval:
    """Deterministic visible lexical/raw retrieval input for arm A."""

    chunks: tuple[ContextChunk, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.chunks, tuple) or not self.chunks:
            raise ConfigurationABError("ordinary retrieval requires visible chunks")
        order = tuple((item.path, item.content_cid) for item in self.chunks)
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ConfigurationABError("ordinary chunks must be uniquely canonical-ordered")

    @property
    def rendered_context(self) -> str:
        return "\n".join(
            f"<<<VISIBLE path={item.path} cid={item.content_cid}>>>\n{item.text}\n<<<END VISIBLE>>>"
            for item in self.chunks
        )

    @property
    def token_count(self) -> int:
        return estimate_context_tokens(self.rendered_context)

    @property
    def evidence_cids(self) -> tuple[str, ...]:
        return tuple(item.content_cid for item in self.chunks)


@dataclass(frozen=True)
class SemanticContextPack:
    """Already-built datasets-owned ContextPack projection for arm B."""

    pack_cid: str
    visible_projection_cid: str
    rendered_context: str
    declared_tokens: int
    exact_source_tokens: int
    capsule_tokens: int
    fallback_count: int = 0
    expansion_count: int = 0
    expansion_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pack_cid", _cid(self.pack_cid, "pack_cid"))
        object.__setattr__(
            self,
            "visible_projection_cid",
            _cid(self.visible_projection_cid, "visible_projection_cid"),
        )
        if not isinstance(self.rendered_context, str):
            raise ConfigurationABError("rendered_context must be UTF-8 text")
        for field in (
            "declared_tokens",
            "exact_source_tokens",
            "capsule_tokens",
            "fallback_count",
            "expansion_count",
            "expansion_tokens",
        ):
            _count(getattr(self, field), field)
        if self.declared_tokens != estimate_context_tokens(self.rendered_context):
            raise ConfigurationABError("ContextPack declared token count is not reproducible")
        if self.exact_source_tokens + self.capsule_tokens > self.declared_tokens:
            raise ConfigurationABError("ContextPack component tokens exceed rendered tokens")
        if self.expansion_count != 0 or self.expansion_tokens != 0:
            raise ConfigurationABError("configuration B does not enable context expansion")

    @property
    def token_count(self) -> int:
        return self.declared_tokens

    @property
    def evidence_cids(self) -> tuple[str, ...]:
        return (self.pack_cid,)


@dataclass(frozen=True)
class ExecutionPermit:
    """Exact provider/revision permit; unavailable remains non-dispatching."""

    permit_cid: str
    provider_id: str
    model_id: str
    model_revision: str
    environment_cid: str
    provenance: str
    available: bool
    live_execution_eligible: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "permit_cid", _cid(self.permit_cid, "permit_cid"))
        object.__setattr__(self, "environment_cid", _cid(self.environment_cid, "environment_cid"))
        for field in ("provider_id", "model_id", "model_revision", "reason"):
            _nonempty(getattr(self, field), field)
        if self.provenance not in PROVENANCE_CLASSES:
            raise ConfigurationABError("permit provenance is outside the frozen taxonomy")
        if type(self.available) is not bool or type(self.live_execution_eligible) is not bool:
            raise ConfigurationABError("permit availability fields must be booleans")

    def admits(self, identity: PairIdentity) -> bool:
        exact = (
            self.provider_id == identity.provider_id
            and self.model_id == identity.model_id
            and self.model_revision == identity.model_revision
            and self.environment_cid == identity.environment_cid
        )
        return bool(
            exact and self.available and (self.provenance != "live" or self.live_execution_eligible)
        )


@dataclass(frozen=True)
class BenchmarkInvocation:
    """The complete provider-visible request.  It has no hidden-answer field."""

    identity: PairIdentity
    task: TaskAgentView
    configuration_id: str
    configuration_cid: str
    configuration: Mapping[str, Any]
    context: str
    context_cids: tuple[str, ...]


@dataclass(frozen=True)
class ProviderObservation:
    status: str
    provenance: str
    evidence_cid: str
    proposal_cid: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    inference_cost_micros: int | None
    failure_cost_micros: int | None = None
    reason: str = "provider-observation"

    def __post_init__(self) -> None:
        if self.status not in PROVIDER_TERMINAL_STATUSES:
            raise ConfigurationABError("provider status is outside the closed taxonomy")
        if self.provenance not in PROVENANCE_CLASSES:
            raise ConfigurationABError("provider provenance is outside the closed taxonomy")
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "evidence_cid"))
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "inference_cost_micros",
            "failure_cost_micros",
        ):
            _optional_count(getattr(self, field), field)
        _nonempty(self.reason, "reason")
        if self.status == "succeeded":
            if self.proposal_cid is None:
                raise ConfigurationABError("successful provider observation requires proposal_cid")
            object.__setattr__(self, "proposal_cid", _cid(self.proposal_cid, "proposal_cid"))
            for field in ("input_tokens", "output_tokens", "inference_cost_micros"):
                if getattr(self, field) is None:
                    raise ConfigurationABError(f"successful provider observation requires {field}")
        elif self.proposal_cid is not None:
            object.__setattr__(self, "proposal_cid", _cid(self.proposal_cid, "proposal_cid"))


@dataclass(frozen=True)
class VerificationRequest:
    identity: PairIdentity
    configuration_id: str
    configuration_cid: str
    proposal_cid: str
    full_verification_required: bool = True
    hidden_mount_phase: str = "after-patch-proposal"


@dataclass(frozen=True)
class FullVerificationObservation:
    """Counts-only response; hidden test bodies never return to the provider."""

    proposal_cid: str
    evidence_cid: str
    full_verification: bool
    hidden_scoring_after_proposal: bool
    full_test_count: int
    full_test_pass_count: int
    hidden_test_total_count: int
    hidden_test_pass_count: int
    regression_count: int
    critical_regression_count: int
    out_of_scope_edit_count: int
    semantic_outcome_match: bool
    verification_cost_micros: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_cid", _cid(self.proposal_cid, "proposal_cid"))
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "evidence_cid"))
        if type(self.full_verification) is not bool:
            raise ConfigurationABError("full_verification must be boolean")
        if type(self.hidden_scoring_after_proposal) is not bool:
            raise ConfigurationABError("hidden_scoring_after_proposal must be boolean")
        for field in (
            "full_test_pass_count",
            "hidden_test_pass_count",
            "regression_count",
            "critical_regression_count",
            "out_of_scope_edit_count",
            "verification_cost_micros",
        ):
            _count(getattr(self, field), field)
        _count(self.full_test_count, "full_test_count", minimum=1)
        _count(self.hidden_test_total_count, "hidden_test_total_count", minimum=1)
        if self.full_test_pass_count > self.full_test_count:
            raise ConfigurationABError("full test passes exceed total")
        if self.hidden_test_pass_count > self.hidden_test_total_count:
            raise ConfigurationABError("hidden test passes exceed total")
        if type(self.semantic_outcome_match) is not bool:
            raise ConfigurationABError("semantic_outcome_match must be boolean")

    @property
    def accepted(self) -> bool:
        return bool(
            self.full_verification
            and self.hidden_scoring_after_proposal
            and self.full_test_pass_count == self.full_test_count
            and self.hidden_test_pass_count == self.hidden_test_total_count
            and self.regression_count == 0
            and self.critical_regression_count == 0
            and self.out_of_scope_edit_count == 0
            and self.semantic_outcome_match
        )


@runtime_checkable
class ProviderPort(Protocol):
    def propose(self, invocation: BenchmarkInvocation) -> ProviderObservation: ...


@runtime_checkable
class FullVerificationPort(Protocol):
    def verify(self, request: VerificationRequest) -> FullVerificationObservation: ...


@dataclass(frozen=True)
class BenchmarkRun:
    configuration_id: str
    raw_result: Mapping[str, Any]
    audit: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible frozen-schema representation."""

        return _thaw(self.raw_result)

    @property
    def result_cid(self) -> str:
        return cid_for_obj(self.as_dict(), codec="dag-json")


@dataclass(frozen=True)
class PairedBenchmarkRun:
    arm_a: BenchmarkRun
    arm_b: BenchmarkRun
    pairing_record: Mapping[str, Any]
    pairing_cid: str


def _empty_metrics() -> dict[str, int | None]:
    return dict.fromkeys(METRIC_NAMES)


def _set_context_metrics(
    metrics: dict[str, int | None],
    configuration_id: str,
    context: OrdinaryRetrieval | SemanticContextPack,
) -> None:
    if configuration_id == "A" and isinstance(context, OrdinaryRetrieval):
        metrics["ordinary_retrieval_tokens"] = context.token_count
        metrics["exact_source_tokens"] = context.token_count
        metrics["context_fallback_count"] = 0
        metrics["context_expansion_count"] = 0
        metrics["context_expansion_tokens"] = 0
    elif configuration_id == "B" and isinstance(context, SemanticContextPack):
        metrics["context_pack_tokens"] = context.token_count
        metrics["exact_source_tokens"] = context.exact_source_tokens
        metrics["capsule_tokens"] = context.capsule_tokens
        metrics["context_fallback_count"] = context.fallback_count
        metrics["context_expansion_count"] = context.expansion_count
        metrics["context_expansion_tokens"] = context.expansion_tokens
    else:
        raise ConfigurationABError("context kind does not match frozen configuration")


def _missingness(metrics: Mapping[str, int | None], *, reason: str) -> dict[str, str]:
    return {name: reason for name, value in metrics.items() if value is None}


def _raw_result(
    *,
    identity: PairIdentity,
    configuration_id: str,
    provenance: str,
    terminal_status: str,
    metrics: Mapping[str, int | None],
    missing_reason: str,
    evidence_cids: Sequence[str],
) -> Mapping[str, Any]:
    if terminal_status not in RESULT_TERMINAL_STATUSES:
        raise ConfigurationABError("result terminal status is outside frozen taxonomy")
    if set(metrics) != set(METRIC_NAMES):
        raise ConfigurationABError("raw metric vocabulary is incomplete")
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
        "missingness": _missingness(metrics, reason=missing_reason),
        "evidence_cids": list(dict.fromkeys(evidence_cids)),
    }
    return _freeze(body)


def _unavailable_run(
    *,
    configuration_id: str,
    identity: PairIdentity,
    context: OrdinaryRetrieval | SemanticContextPack,
    permit: ExecutionPermit,
    reason: str,
) -> BenchmarkRun:
    metrics = _empty_metrics()
    _set_context_metrics(metrics, configuration_id, context)
    metrics["eligible_task_count"] = 1
    metrics["provider_call_count"] = 0
    metrics["route_unavailable_count"] = 1
    raw = _raw_result(
        identity=identity,
        configuration_id=configuration_id,
        provenance=permit.provenance,
        terminal_status="unavailable",
        metrics=metrics,
        missing_reason=reason,
        evidence_cids=(permit.permit_cid, *context.evidence_cids),
    )
    audit = _freeze(
        {
            "schema": RUNNER_SCHEMA,
            "configuration_id": configuration_id,
            "provider_dispatched": False,
            "full_verification_executed": False,
            "hidden_data_shared_with_provider": False,
            "reason": reason,
        }
    )
    return BenchmarkRun(configuration_id, raw, audit)


def run_arm(
    *,
    configuration_id: str,
    identity: PairIdentity,
    task: TaskAgentView,
    context: OrdinaryRetrieval | SemanticContextPack,
    permit: ExecutionPermit,
    provider: ProviderPort,
    verifier: FullVerificationPort,
) -> BenchmarkRun:
    """Run one frozen arm through an injected provider and full verifier."""

    descriptor = configuration_descriptor(configuration_id)
    if configuration_id == "A" and not isinstance(context, OrdinaryRetrieval):
        raise ConfigurationABError("configuration A requires ordinary retrieval")
    if configuration_id == "B" and not isinstance(context, SemanticContextPack):
        raise ConfigurationABError("configuration B requires a semantic ContextPack")
    if (
        isinstance(context, SemanticContextPack)
        and context.visible_projection_cid != identity.visible_projection_cid
    ):
        raise HiddenDataDenied("ContextPack is bound to a different visible projection")
    if not permit.admits(identity):
        return _unavailable_run(
            configuration_id=configuration_id,
            identity=identity,
            context=context,
            permit=permit,
            reason="exact-provider-revision-environment-permit-unavailable",
        )

    invocation = BenchmarkInvocation(
        identity=identity,
        task=task,
        configuration_id=configuration_id,
        configuration_cid=configuration_cid(configuration_id),
        configuration=_freeze(descriptor),
        context=context.rendered_context,
        context_cids=context.evidence_cids,
    )
    try:
        provider_observation = provider.propose(invocation)
    except ProviderUnavailable:
        metrics = _empty_metrics()
        _set_context_metrics(metrics, configuration_id, context)
        metrics["eligible_task_count"] = 1
        metrics["provider_call_count"] = 1
        metrics["route_unavailable_count"] = 1
        raw = _raw_result(
            identity=identity,
            configuration_id=configuration_id,
            provenance=permit.provenance,
            terminal_status="unavailable",
            metrics=metrics,
            missing_reason="provider-call-unavailable-without-observation",
            evidence_cids=(permit.permit_cid, *context.evidence_cids),
        )
        return BenchmarkRun(
            configuration_id,
            raw,
            _freeze(
                {
                    "schema": RUNNER_SCHEMA,
                    "configuration_id": configuration_id,
                    "provider_dispatched": True,
                    "full_verification_executed": False,
                    "hidden_data_shared_with_provider": False,
                    "reason": "provider-call-unavailable-without-observation",
                }
            ),
        )

    if provider_observation.provenance != permit.provenance:
        raise ConfigurationABError("provider provenance differs from exact permit")
    if provider_observation.provenance == "live" and not permit.live_execution_eligible:
        raise ConfigurationABError("live output cannot repair an ineligible permit")

    metrics = _empty_metrics()
    _set_context_metrics(metrics, configuration_id, context)
    metrics["eligible_task_count"] = 1
    metrics["provider_call_count"] = 1
    metrics["provider_input_tokens"] = provider_observation.input_tokens
    metrics["provider_output_tokens"] = provider_observation.output_tokens
    metrics["provider_cached_input_tokens"] = provider_observation.cached_input_tokens
    metrics["inference_cost_micros"] = provider_observation.inference_cost_micros
    metrics["failure_cost_micros"] = provider_observation.failure_cost_micros
    metrics["route_frontier_count"] = 1
    metrics["patch_proposal_count"] = int(provider_observation.proposal_cid is not None)
    evidence = [permit.permit_cid, *context.evidence_cids, provider_observation.evidence_cid]

    if provider_observation.provenance == "simulated":
        metrics["simulated_success_accepted_count"] = 0
        raw = _raw_result(
            identity=identity,
            configuration_id=configuration_id,
            provenance="simulated",
            terminal_status="simulated",
            metrics=metrics,
            missing_reason="simulated-evidence-ineligible-for-observed-quality",
            evidence_cids=evidence,
        )
        return BenchmarkRun(
            configuration_id,
            raw,
            _freeze(
                {
                    "schema": RUNNER_SCHEMA,
                    "configuration_id": configuration_id,
                    "provider_dispatched": True,
                    "full_verification_executed": False,
                    "hidden_data_shared_with_provider": False,
                    "reason": "simulated-evidence-rejected",
                }
            ),
        )

    if provider_observation.status != "succeeded":
        terminal = provider_observation.status
        if terminal not in RESULT_TERMINAL_STATUSES:
            terminal = "infrastructure_failure"
        costs = (
            provider_observation.inference_cost_micros,
            provider_observation.failure_cost_micros,
        )
        if all(value is not None for value in costs):
            metrics["total_cost_micros"] = sum(value for value in costs if value is not None)
            metrics["failed_attempt_cost_micros"] = metrics["total_cost_micros"]
        raw = _raw_result(
            identity=identity,
            configuration_id=configuration_id,
            provenance=permit.provenance,
            terminal_status=terminal,
            metrics=metrics,
            missing_reason=provider_observation.reason,
            evidence_cids=evidence,
        )
        return BenchmarkRun(
            configuration_id,
            raw,
            _freeze(
                {
                    "schema": RUNNER_SCHEMA,
                    "configuration_id": configuration_id,
                    "provider_dispatched": True,
                    "full_verification_executed": False,
                    "hidden_data_shared_with_provider": False,
                    "reason": provider_observation.reason,
                }
            ),
        )

    assert provider_observation.proposal_cid is not None
    verification = verifier.verify(
        VerificationRequest(
            identity=identity,
            configuration_id=configuration_id,
            configuration_cid=configuration_cid(configuration_id),
            proposal_cid=provider_observation.proposal_cid,
        )
    )
    if verification.proposal_cid != provider_observation.proposal_cid:
        raise ConfigurationABError("verification trace is bound to a different proposal")
    if not verification.full_verification:
        raise ConfigurationABError("A/B cannot admit incremental-only verification")
    if not verification.hidden_scoring_after_proposal:
        raise HiddenDataDenied("hidden scoring occurred before the proposal was frozen")

    accepted = verification.accepted
    metrics.update(
        {
            "accepted_patch_count": int(accepted),
            "correct_accepted_patch_count": int(accepted),
            "hidden_test_pass_count": verification.hidden_test_pass_count,
            "hidden_test_total_count": verification.hidden_test_total_count,
            "regression_count": verification.regression_count,
            "critical_regression_accepted_count": (
                verification.critical_regression_count if accepted else 0
            ),
            "out_of_scope_edit_count": verification.out_of_scope_edit_count,
            "first_attempt_success_count": int(accepted and identity.attempt == 1),
            "semantic_outcome_match_count": int(verification.semantic_outcome_match),
            "correct_accepted_patch_rate_bp": 10000 if accepted else 0,
            "full_test_count": verification.full_test_count,
            "full_test_pass_count": verification.full_test_pass_count,
            "full_test_fail_count": (
                verification.full_test_count - verification.full_test_pass_count
            ),
            "verification_cost_micros": verification.verification_cost_micros,
            "proof_cost_micros": 0,
            "assurance_cost_micros": 0,
            "human_cost_micros": 0,
        }
    )
    if (
        provider_observation.inference_cost_micros is not None
        and provider_observation.failure_cost_micros is not None
    ):
        metrics["total_cost_micros"] = (
            provider_observation.inference_cost_micros
            + provider_observation.failure_cost_micros
            + verification.verification_cost_micros
        )
        metrics["failed_attempt_cost_micros"] = 0 if accepted else metrics["total_cost_micros"]
        if accepted:
            metrics["cost_per_correct_accepted_patch_micros"] = metrics["total_cost_micros"]

    evidence.append(verification.evidence_cid)
    raw = _raw_result(
        identity=identity,
        configuration_id=configuration_id,
        provenance=permit.provenance,
        terminal_status="succeeded" if accepted else "verification_failed",
        metrics=metrics,
        missing_reason="not-observed-or-not-applicable-in-configuration-ab-arm",
        evidence_cids=evidence,
    )
    audit = _freeze(
        {
            "schema": RUNNER_SCHEMA,
            "configuration_id": configuration_id,
            "provider_dispatched": True,
            "full_verification_executed": True,
            "incremental_verification_used": False,
            "hidden_scoring_phase": "after-patch-proposal",
            "hidden_data_shared_with_provider": False,
            "proposal_cid": provider_observation.proposal_cid,
            "verification_evidence_cid": verification.evidence_cid,
            "accepted": accepted,
        }
    )
    return BenchmarkRun(configuration_id, raw, audit)


def _with_metric(run: BenchmarkRun, name: str, value: int) -> BenchmarkRun:
    raw = _thaw(run.raw_result)
    raw["metrics"][name] = value
    raw["missingness"].pop(name, None)
    return BenchmarkRun(run.configuration_id, _freeze(raw), run.audit)


def run_paired_ab(
    *,
    identity_a: PairIdentity,
    identity_b: PairIdentity,
    task_a: TaskAgentView,
    task_b: TaskAgentView,
    ordinary_context: OrdinaryRetrieval,
    semantic_context: SemanticContextPack,
    permit_a: ExecutionPermit,
    permit_b: ExecutionPermit,
    provider_a: ProviderPort,
    provider_b: ProviderPort,
    verifier_a: FullVerificationPort,
    verifier_b: FullVerificationPort,
) -> PairedBenchmarkRun:
    """Execute a controlled pair and bind its shared identities and treatment."""

    if identity_a != identity_b:
        changed = [
            field
            for field in _PAIR_FIELDS
            if getattr(identity_a, field) != getattr(identity_b, field)
        ]
        raise ConfigurationABError(f"paired identity mismatch: {changed}")
    if task_a != task_b:
        raise ConfigurationABError("paired task projections differ")
    if (
        permit_a.provider_id,
        permit_a.model_id,
        permit_a.model_revision,
        permit_a.environment_cid,
        permit_a.provenance,
        permit_a.available,
        permit_a.live_execution_eligible,
    ) != (
        permit_b.provider_id,
        permit_b.model_id,
        permit_b.model_revision,
        permit_b.environment_cid,
        permit_b.provenance,
        permit_b.available,
        permit_b.live_execution_eligible,
    ):
        raise ConfigurationABError("paired permits change a held-constant field")

    arm_a = run_arm(
        configuration_id="A",
        identity=identity_a,
        task=task_a,
        context=ordinary_context,
        permit=permit_a,
        provider=provider_a,
        verifier=verifier_a,
    )
    arm_b = run_arm(
        configuration_id="B",
        identity=identity_b,
        task=task_b,
        context=semantic_context,
        permit=permit_b,
        provider=provider_b,
        verifier=verifier_b,
    )
    if ordinary_context.token_count > 0:
        reduction = (
            (ordinary_context.token_count - semantic_context.token_count)
            * 10000
            // ordinary_context.token_count
        )
        arm_b = _with_metric(arm_b, "context_reduction_bp", reduction)

    pair_body = {
        "schema": PAIR_SCHEMA,
        "identity": _thaw(identity_a.as_mapping()),
        "configuration_a_cid": CONFIGURATION_A_CID,
        "configuration_b_cid": CONFIGURATION_B_CID,
        "arm_a_result_cid": arm_a.result_cid,
        "arm_b_result_cid": arm_b.result_cid,
        "only_treatment_difference": "context_method",
        "verification_policy_both_arms": "full-runtime-verification@1",
        "full_verification_executed_both_arms": bool(
            arm_a.audit.get("full_verification_executed")
            and arm_b.audit.get("full_verification_executed")
        ),
        "hidden_provider_access": False,
    }
    return PairedBenchmarkRun(
        arm_a=arm_a,
        arm_b=arm_b,
        pairing_record=_freeze(pair_body),
        pairing_cid=cid_for_obj(pair_body, codec="dag-json"),
    )


def runner_descriptor() -> dict[str, Any]:
    """Return the immutable reviewed behavior surface for receipt binding."""

    return {
        "schema": RUNNER_SCHEMA,
        "identity_profile": IDENTITY_PROFILE,
        "configuration_cids": dict(CONFIGURATION_CIDS),
        "held_constant_fields": list(_PAIR_FIELDS),
        "only_a_to_b_difference": ["context_method"],
        "provider_visibility": "task-agent-view-plus-visible-context-only",
        "hidden_scoring_phase": "after-patch-proposal",
        "verification": "full-runtime-verification-both-arms",
        "live_dispatch": "exact-available-eligible-permit-only",
        "unavailable_policy": "preserve-unavailable-never-impute-live",
        "simulation_policy": "force-simulated-terminal-never-accepted",
        "expansion_policy": "disabled-and-zero-recorded",
    }


RUNNER_DESCRIPTOR_CID: Final[str] = cid_for_obj(runner_descriptor(), codec="dag-json")

__all__ = [
    "BenchmarkInvocation",
    "BenchmarkRun",
    "CONFIGURATION_A_CID",
    "CONFIGURATION_B_CID",
    "CONFIGURATION_CIDS",
    "ConfigurationABError",
    "ContextChunk",
    "ExecutionPermit",
    "FullVerificationObservation",
    "FullVerificationPort",
    "HiddenDataDenied",
    "METRIC_NAMES",
    "OrdinaryRetrieval",
    "PairIdentity",
    "PairedBenchmarkRun",
    "ProviderObservation",
    "ProviderPort",
    "ProviderUnavailable",
    "RUNNER_DESCRIPTOR_CID",
    "SemanticContextPack",
    "TaskAgentView",
    "VerificationRequest",
    "configuration_cid",
    "configuration_descriptor",
    "estimate_context_tokens",
    "run_arm",
    "run_paired_ab",
    "runner_descriptor",
]
