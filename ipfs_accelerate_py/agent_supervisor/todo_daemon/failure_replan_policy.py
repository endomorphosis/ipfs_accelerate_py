"""Formal replan + failure memory on retry (WPD-031 / FailureReplanPolicy@1).

On typed worker failure the retry path is:

```text
typed failure
  -> PlanFailureMemory (identical → backoff)
  -> FormalDeltaReplanner (invalidate only bound dependent suffix)
  -> sealed ResidualLlmPacket (optional LLM) OR abstain_review
  -> never free-form full-task re-prompt as sole context
```

Fail-closed rules:

* Repeated identical failure observations trigger exponential backoff and do
  not re-open the plan or authorize a provider call.
* Delta replan edits only the bound failure anchors and their transitive
  dependants; unaffected accepted steps are preserved.
* LLM / provider retry requires a sealed :class:`ResidualLlmPacket` under
  disposition ``residual_llm_authorized``. Free re-prompt (full task body as
  sole context) is rejected.
* Without a sealable residual packet the path yields ``abstain_review``.
* The policy never loads network clients or model-provider SDKs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..planning.formal_replanner import (
    DeltaPlan,
    DeltaReplanDecision,
    DeltaReplanLimits,
    DeltaReplanStopReason,
    FormalDeltaReplanner,
    FormalReplanner,
    ReplanLimits,
    ReplannerValidationError,
)
from ..planning.plan_failure_memory import (
    BranchFailureObservation,
    FailureBackoffPolicy,
    FailureMemoryDisposition,
    PlanFailureMemory,
    PlanFailureMemoryError,
)
from ..planning.residual_llm_packet import (
    ResidualLlmPacket,
    ResidualLlmPacketError,
    ResidualLlmPacketLimits,
    packet_satisfies_residual_llm_contract,
    residual_llm_packet_from_codex,
    seal_residual_llm_packet,
)
from ..proof.formal_verification_contracts import content_identity
from .implementation_disposition import ImplementationDisposition


# ---------------------------------------------------------------------------
# Interface identity / evidence
# ---------------------------------------------------------------------------

FAILURE_REPLAN_POLICY_INTERFACE: Final[str] = "FailureReplanPolicy@1"
FAILURE_REPLAN_POLICY_VERSION: Final[int] = 1
FAILURE_REPLAN_POLICY_EVIDENCE: Final[str] = "wpd/formal-replan-on-failure@1"
FAILURE_REPLAN_POLICY_PRODUCER: Final[str] = "failure-replan-policy@1"

FAILURE_REPLAN_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/failure-replan-request@1"
)
FAILURE_REPLAN_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/failure-replan-result@1"
)
FAILURE_REPLAN_POLICY_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/failure-replan-policy/discovery@1"
)
PROVIDER_RETRY_AUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-retry-authorization@1"
)

# Reason codes (stable, machine-readable)
REASON_REPLAN_RESIDUAL_AUTHORIZED: Final[str] = "replan_residual_llm_authorized"
REASON_REPLAN_ABSTAIN_NO_PACKET: Final[str] = "replan_abstain_missing_residual_packet"
REASON_UNCHANGED_FAILURE_BACKOFF: Final[str] = "unchanged_failure_backoff"
REASON_IDENTICAL_FAILURE_EXHAUSTED: Final[str] = "identical_failure_exhausted"
REASON_RETRY_BUDGET_EXHAUSTED: Final[str] = "retry_budget_exhausted"
REASON_MEMORY_BOUND_REACHED: Final[str] = "failure_memory_bound_reached"
REASON_UNBOUND_FAILURE: Final[str] = "unbound_failure"
REASON_REPAIR_BOUND_EXCEEDED: Final[str] = "repair_bound_exceeded"
REASON_DEADLINE_EXCEEDED: Final[str] = "deadline_exceeded"
REASON_CANCELLED: Final[str] = "cancelled"
REASON_RESIDUAL_PACKET_REQUIRED: Final[str] = "residual_packet_required_for_llm_retry"
REASON_FREE_REPROMPT_FORBIDDEN: Final[str] = "free_reprompt_forbidden"
REASON_PACKET_SEAL_FAILED: Final[str] = "residual_packet_seal_failed"
REASON_BOUND_RECORD_ESCAPE: Final[str] = "replan_bound_record_escape"
REASON_MALFORMED_REQUEST: Final[str] = "malformed_failure_replan_request"

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "code",
        "content",
        "contents",
        "file_text",
        "full_task_body",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_ast",
        "raw_log",
        "snippet",
        "source",
        "source_body",
        "source_text",
        "task_body",
        "task_prose",
        "transcript",
    }
)

_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)

_STOP_REASON_TO_POLICY_REASON: Final[
    Mapping[DeltaReplanStopReason, str]
] = {
    DeltaReplanStopReason.REPLAN_REQUIRED: REASON_REPLAN_ABSTAIN_NO_PACKET,
    DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF: (
        REASON_UNCHANGED_FAILURE_BACKOFF
    ),
    DeltaReplanStopReason.IDENTICAL_FAILURE_EXHAUSTED: (
        REASON_IDENTICAL_FAILURE_EXHAUSTED
    ),
    DeltaReplanStopReason.RETRY_BUDGET_EXHAUSTED: REASON_RETRY_BUDGET_EXHAUSTED,
    DeltaReplanStopReason.FAILURE_MEMORY_BOUND_REACHED: (
        REASON_MEMORY_BOUND_REACHED
    ),
    DeltaReplanStopReason.UNBOUND_FAILURE: REASON_UNBOUND_FAILURE,
    DeltaReplanStopReason.REPAIR_BOUND_EXCEEDED: REASON_REPAIR_BOUND_EXCEEDED,
    DeltaReplanStopReason.DEADLINE_EXCEEDED: REASON_DEADLINE_EXCEEDED,
    DeltaReplanStopReason.CANCELLED: REASON_CANCELLED,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FailureReplanPolicyError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete failure-replan step."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "failure_replan_policy_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "failure_replan_policy_error")


class FailureReplanPolicyInputError(FailureReplanPolicyError, ValueError):
    """Caller supplied a malformed replan request."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = REASON_MALFORMED_REQUEST,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class FreeRepromptForbiddenError(FailureReplanPolicyError):
    """Free-form full-task re-prompt is never admitted after typed failure."""

    def __init__(
        self,
        message: str = "free re-prompt after typed failure is forbidden",
        *,
        reason_code: str = REASON_FREE_REPROMPT_FORBIDDEN,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ResidualPacketRequiredError(FailureReplanPolicyError):
    """LLM retry was requested without a sealed residual packet."""

    def __init__(
        self,
        message: str = "residual packet is required for LLM retry",
        *,
        reason_code: str = REASON_RESIDUAL_PACKET_REQUIRED,
    ) -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------


class FailureReplanOutcome(str, Enum):
    """Closed outcomes of one failure-replan evaluation."""

    RESIDUAL_LLM_AUTHORIZED = "residual_llm_authorized"
    ABSTAIN_REVIEW = "abstain_review"
    BACKOFF = "backoff"
    EXHAUSTED = "exhausted"
    BOUND_EXCEEDED = "bound_exceeded"
    UNBOUND = "unbound"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Request / residual materials
# ---------------------------------------------------------------------------


def _assert_body_free(value: Any, field_name: str = "payload") -> None:
    if isinstance(value, float):
        raise FailureReplanPolicyInputError(
            f"{field_name} may not contain floating-point values",
            reason_code="body_or_secret_rejected",
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FailureReplanPolicyInputError(
                    f"{field_name} has a non-string key",
                    reason_code="body_or_secret_rejected",
                )
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or normalized in _SECRET_KEYS:
                raise FailureReplanPolicyInputError(
                    f"{field_name} may not contain secrets, source bodies, "
                    "or free-form task prose",
                    reason_code="body_or_secret_rejected",
                )
            if any(
                marker in normalized
                for marker in ("password", "private_key", "api_key", "task_body")
            ):
                raise FailureReplanPolicyInputError(
                    f"{field_name} may not contain secrets, source bodies, "
                    "or free-form task prose",
                    reason_code="body_or_secret_rejected",
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise FailureReplanPolicyInputError(
            f"{field_name} may not contain binary bodies",
            reason_code="body_or_secret_rejected",
        )


def _normalize_path(path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        raise FailureReplanPolicyInputError(
            "write path must be non-empty",
            reason_code="invalid_write_path",
        )
    if text.startswith("/") or ".." in text.split("/"):
        raise FailureReplanPolicyInputError(
            f"write path must be a relative repository path: {path!r}",
            reason_code="path_escape",
        )
    return text


def _normalize_ids(
    values: Sequence[str] | None,
    field_name: str,
    *,
    allow_spaces: bool = False,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise FailureReplanPolicyInputError(
            f"{field_name} must be a sequence of identifiers",
            reason_code="invalid_identifiers",
        )
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        if not allow_spaces and any(ch.isspace() for ch in text):
            raise FailureReplanPolicyInputError(
                f"{field_name} entries must be compact identifiers",
                reason_code="invalid_identifiers",
            )
        if text not in out:
            out.append(text)
    return tuple(out)


@dataclass(frozen=True)
class ResidualPacketMaterials:
    """Body-free materials required to seal a residual LLM packet.

    When present and valid, a replan that needs model residual work may
    authorize ``residual_llm_authorized``.  Missing materials force
    ``abstain_review`` rather than a free re-prompt.
    """

    task_id: str
    repository_id: str
    tree_id: str
    write_paths: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    counterexample_capsule: Mapping[str, Any]
    validation_commands: tuple[str, ...]
    policy_id: str = ""
    policy_revision: str = ""
    forest_id: str = ""
    acceptance_ids: tuple[str, ...] = ()
    authority_roots: Mapping[str, str] = field(default_factory=dict)
    codex_packet: Any = None
    limits: ResidualLlmPacketLimits | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(
            self, "repository_id", str(self.repository_id or "").strip()
        )
        object.__setattr__(self, "tree_id", str(self.tree_id or "").strip())
        if not self.task_id or not self.repository_id or not self.tree_id:
            raise FailureReplanPolicyInputError(
                "residual materials require task_id, repository_id, and tree_id",
                reason_code="missing_residual_identity",
            )
        object.__setattr__(
            self,
            "write_paths",
            tuple(_normalize_path(path) for path in (self.write_paths or ())),
        )
        if not self.write_paths:
            raise FailureReplanPolicyInputError(
                "residual materials require exact write_paths",
                reason_code="missing_write_paths",
            )
        object.__setattr__(
            self,
            "obligation_ids",
            _normalize_ids(self.obligation_ids, "obligation_ids"),
        )
        if not self.obligation_ids:
            raise FailureReplanPolicyInputError(
                "residual materials require obligation_ids",
                reason_code="missing_obligations",
            )
        object.__setattr__(
            self,
            "validation_commands",
            _normalize_ids(
                self.validation_commands,
                "validation_commands",
                allow_spaces=True,
            ),
        )
        if not self.validation_commands:
            raise FailureReplanPolicyInputError(
                "residual materials require validation_commands",
                reason_code="missing_validation_commands",
            )
        capsule = self.counterexample_capsule
        if capsule is None and self.codex_packet is None:
            raise FailureReplanPolicyInputError(
                "residual materials require counterexample_capsule or codex_packet",
                reason_code="missing_counterexample_capsule",
            )
        if capsule is not None:
            if not isinstance(capsule, Mapping):
                raise FailureReplanPolicyInputError(
                    "counterexample_capsule must be a mapping",
                    reason_code="invalid_capsule",
                )
            _assert_body_free(dict(capsule), "counterexample_capsule")
            object.__setattr__(self, "counterexample_capsule", dict(capsule))
        else:
            object.__setattr__(self, "counterexample_capsule", {})
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(
            self, "policy_revision", str(self.policy_revision or "").strip()
        )
        object.__setattr__(self, "forest_id", str(self.forest_id or "").strip())
        object.__setattr__(
            self,
            "acceptance_ids",
            _normalize_ids(self.acceptance_ids, "acceptance_ids"),
        )
        roots = dict(self.authority_roots or {})
        _assert_body_free(roots, "authority_roots")
        object.__setattr__(
            self,
            "authority_roots",
            {str(key): str(value) for key, value in roots.items()},
        )

    def seal(self) -> ResidualLlmPacket:
        """Seal materials into ResidualLlmPacket@1 (fail closed)."""

        if self.codex_packet is not None:
            return residual_llm_packet_from_codex(
                self.codex_packet,
                task_id=self.task_id,
                repository_id=self.repository_id,
                tree_id=self.tree_id,
                write_paths=self.write_paths,
                obligation_ids=self.obligation_ids,
                validation_commands=self.validation_commands,
                policy_id=self.policy_id,
                policy_revision=self.policy_revision,
                forest_id=self.forest_id,
                acceptance_ids=self.acceptance_ids,
                authority_roots=self.authority_roots,
                limits=self.limits,
            )
        return seal_residual_llm_packet(
            task_id=self.task_id,
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            write_paths=self.write_paths,
            obligation_ids=self.obligation_ids,
            counterexample_capsule=self.counterexample_capsule,
            validation_commands=self.validation_commands,
            policy_id=self.policy_id,
            policy_revision=self.policy_revision,
            forest_id=self.forest_id,
            acceptance_ids=self.acceptance_ids,
            authority_roots=self.authority_roots,
            limits=self.limits,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "counterexample_capsule": dict(self.counterexample_capsule),
            "validation_commands": list(self.validation_commands),
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "forest_id": self.forest_id,
            "acceptance_ids": list(self.acceptance_ids),
            "authority_roots": dict(self.authority_roots),
            "has_codex_packet": self.codex_packet is not None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualPacketMaterials":
        if not isinstance(payload, Mapping):
            raise FailureReplanPolicyInputError(
                "residual materials must be a mapping",
                reason_code="invalid_residual_materials",
            )
        _assert_body_free(payload, "residual materials")
        data = {
            key: payload[key]
            for key in (
                "task_id",
                "repository_id",
                "tree_id",
                "write_paths",
                "obligation_ids",
                "counterexample_capsule",
                "validation_commands",
                "policy_id",
                "policy_revision",
                "forest_id",
                "acceptance_ids",
                "authority_roots",
                "codex_packet",
                "limits",
            )
            if key in payload
        }
        for key in ("write_paths", "obligation_ids", "validation_commands", "acceptance_ids"):
            if key in data:
                data[key] = tuple(data[key] or ())
        return cls(**data)


@dataclass(frozen=True)
class FailureReplanRequest:
    """One typed failure presented for formal replan + residual admission.

    Free-form task prose is rejected.  Residual materials are optional: when
    absent, a replan that would otherwise need LLM residual yields
    ``abstain_review``.
    """

    plan: DeltaPlan | Mapping[str, Any]
    observation: BranchFailureObservation | Mapping[str, Any]
    residual_materials: ResidualPacketMaterials | Mapping[str, Any] | None = None
    residual_packet: ResidualLlmPacket | Mapping[str, Any] | None = None
    observed_at_milliseconds: int = 1
    now_milliseconds: int | None = None
    deadline_milliseconds: int | None = None
    free_reprompt_context: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.free_reprompt_context is not None:
            raise FreeRepromptForbiddenError(
                "free_reprompt_context is never admitted on the failure-replan path"
            )
        if int(self.observed_at_milliseconds) < 1:
            raise FailureReplanPolicyInputError(
                "observed_at_milliseconds must be >= 1",
                reason_code="invalid_timestamp",
            )
        object.__setattr__(
            self, "observed_at_milliseconds", int(self.observed_at_milliseconds)
        )
        metadata = dict(self.metadata or {})
        _assert_body_free(metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)
        if self.residual_materials is not None and not isinstance(
            self.residual_materials, ResidualPacketMaterials
        ):
            if not isinstance(self.residual_materials, Mapping):
                raise FailureReplanPolicyInputError(
                    "residual_materials must be ResidualPacketMaterials or mapping",
                    reason_code="invalid_residual_materials",
                )
            object.__setattr__(
                self,
                "residual_materials",
                ResidualPacketMaterials.from_dict(self.residual_materials),
            )
        if self.residual_packet is not None and not isinstance(
            self.residual_packet, ResidualLlmPacket
        ):
            if not isinstance(self.residual_packet, Mapping):
                raise FailureReplanPolicyInputError(
                    "residual_packet must be ResidualLlmPacket or mapping",
                    reason_code="invalid_residual_packet",
                )
            try:
                object.__setattr__(
                    self,
                    "residual_packet",
                    ResidualLlmPacket.from_dict(self.residual_packet),
                )
            except ResidualLlmPacketError as exc:
                raise FailureReplanPolicyInputError(
                    f"residual_packet seal invalid: {exc}",
                    reason_code=REASON_PACKET_SEAL_FAILED,
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        plan = self.plan
        plan_payload = (
            plan.to_dict() if isinstance(plan, DeltaPlan) else dict(plan)
        )
        observation = self.observation
        obs_payload = (
            observation.to_dict()
            if isinstance(observation, BranchFailureObservation)
            else dict(observation)
        )
        materials = self.residual_materials
        materials_payload = (
            materials.to_dict()
            if isinstance(materials, ResidualPacketMaterials)
            else (dict(materials) if materials is not None else None)
        )
        packet = self.residual_packet
        packet_payload = (
            packet.to_dict()
            if isinstance(packet, ResidualLlmPacket)
            else (dict(packet) if packet is not None else None)
        )
        return {
            "schema": FAILURE_REPLAN_REQUEST_SCHEMA,
            "plan": plan_payload,
            "observation": obs_payload,
            "residual_materials": materials_payload,
            "residual_packet": packet_payload,
            "observed_at_milliseconds": self.observed_at_milliseconds,
            "now_milliseconds": self.now_milliseconds,
            "deadline_milliseconds": self.deadline_milliseconds,
            "metadata": dict(self.metadata),
            "free_reprompt_context": None,
        }


# ---------------------------------------------------------------------------
# Result / authorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRetryAuthorization:
    """Whether a provider / LLM retry is admitted after typed failure."""

    authorized: bool
    reason_code: str
    disposition: ImplementationDisposition
    residual_packet_id: str = ""
    residual_packet: ResidualLlmPacket | None = None
    free_reprompt_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(
            self, "residual_packet_id", str(self.residual_packet_id or "").strip()
        )
        if self.free_reprompt_allowed:
            raise FreeRepromptForbiddenError(
                "provider retry authorization must never allow free re-prompt"
            )
        if self.authorized:
            if self.disposition is not ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED:
                raise FailureReplanPolicyError(
                    "authorized provider retry requires residual_llm_authorized",
                    reason_code="disposition_mismatch",
                )
            if not self.residual_packet_id or self.residual_packet is None:
                raise ResidualPacketRequiredError(
                    "authorized provider retry requires a sealed residual packet"
                )
            if not packet_satisfies_residual_llm_contract(self.residual_packet):
                raise FailureReplanPolicyError(
                    "authorized residual packet failed ResidualLlmPacket@1 contract",
                    reason_code=REASON_PACKET_SEAL_FAILED,
                )
        else:
            if self.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED:
                raise FailureReplanPolicyError(
                    "unauthorized retry cannot claim residual_llm_authorized",
                    reason_code="disposition_mismatch",
                )
            if self.residual_packet is not None:
                raise FailureReplanPolicyError(
                    "unauthorized retry must not carry a residual packet",
                    reason_code="packet_on_unauthorized",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_RETRY_AUTHORIZATION_SCHEMA,
            "authorized": self.authorized,
            "reason_code": self.reason_code,
            "disposition": self.disposition.value,
            "residual_packet_id": self.residual_packet_id,
            "residual_packet": (
                self.residual_packet.to_dict()
                if self.residual_packet is not None
                else None
            ),
            "free_reprompt_allowed": False,
            "authorizes_provider": self.authorized,
        }


@dataclass(frozen=True)
class FailureReplanResult:
    """Outcome of :meth:`FailureReplanPolicy.evaluate`.

    Carries the delta replan decision, failure-memory disposition, optional
    sealed residual packet, and the implementation disposition for the next
    worker step.  Provider hooks are never opened by this surface.
    """

    outcome: FailureReplanOutcome
    reason_code: str
    disposition: ImplementationDisposition
    delta_decision: DeltaReplanDecision
    memory_disposition: str
    backoff_milliseconds: int
    backoff_attempt: int
    bound_step_ids: tuple[str, ...]
    invalidated_step_ids: tuple[str, ...]
    preserved_step_ids: tuple[str, ...]
    residual_packet: ResidualLlmPacket | None = None
    residual_packet_required: bool = False
    residual_packet_sealed: bool = False
    provider_hook_count: int = 0
    free_reprompt_allowed: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", FailureReplanOutcome(self.outcome))
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(
            self, "memory_disposition", str(self.memory_disposition or "").strip()
        )
        object.__setattr__(
            self, "bound_step_ids", tuple(str(item) for item in self.bound_step_ids)
        )
        object.__setattr__(
            self,
            "invalidated_step_ids",
            tuple(str(item) for item in self.invalidated_step_ids),
        )
        object.__setattr__(
            self,
            "preserved_step_ids",
            tuple(str(item) for item in self.preserved_step_ids),
        )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "provider_hook_count", int(self.provider_hook_count))
        object.__setattr__(
            self, "backoff_milliseconds", int(self.backoff_milliseconds)
        )
        object.__setattr__(self, "backoff_attempt", int(self.backoff_attempt))
        if self.provider_hook_count != 0:
            raise FailureReplanPolicyError(
                "failure replan policy must never open a provider/LLM hook",
                reason_code="provider_hook_forbidden",
            )
        if self.free_reprompt_allowed:
            raise FreeRepromptForbiddenError()
        if (
            self.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
            and self.residual_packet is None
        ):
            raise ResidualPacketRequiredError(
                "residual_llm_authorized requires a sealed residual packet"
            )
        if self.residual_packet is not None and not self.residual_packet_sealed:
            raise FailureReplanPolicyError(
                "residual packet present but not marked sealed",
                reason_code=REASON_PACKET_SEAL_FAILED,
            )
        # Bound-record invariant: invalidated steps must be a subset of the
        # dependency-closed bound set, and preserved steps must be disjoint.
        invalidated = set(self.invalidated_step_ids)
        preserved = set(self.preserved_step_ids)
        if invalidated & preserved:
            raise FailureReplanPolicyError(
                "invalidated and preserved step sets must be disjoint",
                reason_code=REASON_BOUND_RECORD_ESCAPE,
            )
        if self.delta_decision.changed:
            bound = set(self.bound_step_ids)
            if not bound:
                raise FailureReplanPolicyError(
                    "active replan requires non-empty bound step ids",
                    reason_code=REASON_BOUND_RECORD_ESCAPE,
                )
            if not invalidated.issubset(bound):
                raise FailureReplanPolicyError(
                    "replan invalidated steps outside the bound dependent suffix",
                    reason_code=REASON_BOUND_RECORD_ESCAPE,
                )

    @property
    def authorizes_provider(self) -> bool:
        return (
            self.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
            and self.residual_packet is not None
        )

    @property
    def should_backoff(self) -> bool:
        return self.outcome is FailureReplanOutcome.BACKOFF

    @property
    def should_replan(self) -> bool:
        return self.delta_decision.changed

    @property
    def edits_only_bound_records(self) -> bool:
        """True when every invalidated step is inside the bound suffix."""

        if not self.delta_decision.changed:
            return True
        return set(self.invalidated_step_ids).issubset(set(self.bound_step_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_REPLAN_RESULT_SCHEMA,
            "interface": FAILURE_REPLAN_POLICY_INTERFACE,
            "evidence": FAILURE_REPLAN_POLICY_EVIDENCE,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "disposition": self.disposition.value,
            "delta_decision": self.delta_decision.to_dict(),
            "memory_disposition": self.memory_disposition,
            "backoff_milliseconds": self.backoff_milliseconds,
            "backoff_attempt": self.backoff_attempt,
            "bound_step_ids": list(self.bound_step_ids),
            "invalidated_step_ids": list(self.invalidated_step_ids),
            "preserved_step_ids": list(self.preserved_step_ids),
            "residual_packet": (
                self.residual_packet.to_dict()
                if self.residual_packet is not None
                else None
            ),
            "residual_packet_required": self.residual_packet_required,
            "residual_packet_sealed": self.residual_packet_sealed,
            "provider_hook_count": 0,
            "free_reprompt_allowed": False,
            "authorizes_provider": self.authorizes_provider,
            "edits_only_bound_records": self.edits_only_bound_records,
            "notes": list(self.notes),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class FailureReplanPolicy:
    """Compose FormalDeltaReplanner + PlanFailureMemory for worker retries.

    Interface: ``FailureReplanPolicy@1``
    Evidence: ``wpd/formal-replan-on-failure@1``
    """

    failure_memory: PlanFailureMemory | None = None
    delta_limits: DeltaReplanLimits | Mapping[str, Any] | None = None
    replan_limits: ReplanLimits | Mapping[str, Any] | None = None
    formal_replanner: FormalReplanner | None = None
    require_residual_packet_for_llm: bool = True

    def __post_init__(self) -> None:
        memory = self.failure_memory or PlanFailureMemory()
        if not isinstance(memory, PlanFailureMemory):
            raise FailureReplanPolicyInputError(
                "failure_memory must be PlanFailureMemory",
                reason_code="invalid_failure_memory",
            )
        self.failure_memory = memory
        if self.delta_limits is None:
            self.delta_limits = DeltaReplanLimits()
        elif isinstance(self.delta_limits, Mapping):
            self.delta_limits = DeltaReplanLimits.from_dict(self.delta_limits)
        if not isinstance(self.delta_limits, DeltaReplanLimits):
            raise FailureReplanPolicyInputError(
                "delta_limits must be DeltaReplanLimits or mapping",
                reason_code="invalid_delta_limits",
            )
        if self.replan_limits is None:
            self.replan_limits = ReplanLimits()
        elif isinstance(self.replan_limits, Mapping):
            self.replan_limits = ReplanLimits(**dict(self.replan_limits))
        if not isinstance(self.replan_limits, ReplanLimits):
            raise FailureReplanPolicyInputError(
                "replan_limits must be ReplanLimits or mapping",
                reason_code="invalid_replan_limits",
            )
        self._delta_replanner = FormalDeltaReplanner(
            failure_memory=self.failure_memory,
            limits=self.delta_limits,
        )

    @classmethod
    def discovery(cls) -> dict[str, Any]:
        return {
            "schema": FAILURE_REPLAN_POLICY_DISCOVERY_SCHEMA,
            "interface": FAILURE_REPLAN_POLICY_INTERFACE,
            "version": FAILURE_REPLAN_POLICY_VERSION,
            "evidence_key": FAILURE_REPLAN_POLICY_EVIDENCE,
            "producer": FAILURE_REPLAN_POLICY_PRODUCER,
            "uses_formal_replanner": True,
            "uses_plan_failure_memory": True,
            "edits_only_bound_records": True,
            "residual_packet_required_for_llm_retry": True,
            "free_reprompt_allowed": False,
            "llm_router_enabled": False,
            "automatic_fallback": False,
            "network_access": False,
            "provider_hooks": 0,
            "outcomes": sorted(item.value for item in FailureReplanOutcome),
            "backoff_on_identical_failure": True,
        }

    def evaluate(
        self,
        request: FailureReplanRequest | Mapping[str, Any],
        *,
        cancelled: Any = None,
    ) -> FailureReplanResult:
        """Run one formal replan + residual admission decision for a typed failure."""

        req = self._normalize_request(request)
        try:
            delta = self._delta_replanner.replan(
                req.plan,
                req.observation,
                observed_at_milliseconds=req.observed_at_milliseconds,
                now_milliseconds=req.now_milliseconds,
                deadline_milliseconds=req.deadline_milliseconds,
                cancelled=cancelled,
            )
        except ReplannerValidationError as exc:
            raise FailureReplanPolicyInputError(
                str(exc),
                reason_code=REASON_MALFORMED_REQUEST,
            ) from exc
        except PlanFailureMemoryError as exc:
            raise FailureReplanPolicyInputError(
                str(exc),
                reason_code=REASON_MALFORMED_REQUEST,
            ) from exc

        bound_ids = tuple(delta.invalidated_step_ids) if delta.changed else (
            tuple(delta.direct_failure_step_ids)
        )
        # Bound-record check: preserved steps must remain accepted and
        # outside the invalidated set (already enforced by DeltaReplanDecision).
        self._assert_bound_records_only(delta)

        memory_disposition = self._memory_disposition_for(delta)
        packet, packet_reason = self._resolve_residual_packet(req)

        if delta.stop_reason is DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF:
            return self._result(
                outcome=FailureReplanOutcome.BACKOFF,
                reason_code=REASON_UNCHANGED_FAILURE_BACKOFF,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=(
                    "identical_failure",
                    "backoff_applied",
                    "provider_forbidden",
                ),
            )

        if delta.stop_reason in {
            DeltaReplanStopReason.IDENTICAL_FAILURE_EXHAUSTED,
            DeltaReplanStopReason.RETRY_BUDGET_EXHAUSTED,
        }:
            return self._result(
                outcome=FailureReplanOutcome.EXHAUSTED,
                reason_code=_STOP_REASON_TO_POLICY_REASON[delta.stop_reason],
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("retry_exhausted", "provider_forbidden"),
            )

        if delta.stop_reason is DeltaReplanStopReason.FAILURE_MEMORY_BOUND_REACHED:
            return self._result(
                outcome=FailureReplanOutcome.BOUND_EXCEEDED,
                reason_code=REASON_MEMORY_BOUND_REACHED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("memory_bound", "provider_forbidden"),
            )

        if delta.stop_reason is DeltaReplanStopReason.UNBOUND_FAILURE:
            return self._result(
                outcome=FailureReplanOutcome.UNBOUND,
                reason_code=REASON_UNBOUND_FAILURE,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=(),
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("unbound_failure", "provider_forbidden"),
            )

        if delta.stop_reason is DeltaReplanStopReason.REPAIR_BOUND_EXCEEDED:
            return self._result(
                outcome=FailureReplanOutcome.BOUND_EXCEEDED,
                reason_code=REASON_REPAIR_BOUND_EXCEEDED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("repair_bound_exceeded", "provider_forbidden"),
            )

        if delta.stop_reason is DeltaReplanStopReason.DEADLINE_EXCEEDED:
            return self._result(
                outcome=FailureReplanOutcome.BOUND_EXCEEDED,
                reason_code=REASON_DEADLINE_EXCEEDED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("deadline_exceeded", "provider_forbidden"),
            )

        if delta.stop_reason is DeltaReplanStopReason.CANCELLED:
            return self._result(
                outcome=FailureReplanOutcome.CANCELLED,
                reason_code=REASON_CANCELLED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=False,
                residual_packet_sealed=False,
                notes=("cancelled", "provider_forbidden"),
            )

        # Active replan: residual packet required for LLM retry.
        if not delta.changed:
            # Defensive: unknown stop without change → abstain.
            return self._result(
                outcome=FailureReplanOutcome.ABSTAIN_REVIEW,
                reason_code=_STOP_REASON_TO_POLICY_REASON.get(
                    delta.stop_reason, REASON_REPLAN_ABSTAIN_NO_PACKET
                ),
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=bound_ids,
                residual_packet=None,
                residual_packet_required=self.require_residual_packet_for_llm,
                residual_packet_sealed=False,
                notes=("no_active_replan", "provider_forbidden"),
            )

        if packet is not None:
            return self._result(
                outcome=FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED,
                reason_code=REASON_REPLAN_RESIDUAL_AUTHORIZED,
                disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
                delta=delta,
                memory_disposition=memory_disposition,
                bound_step_ids=tuple(delta.invalidated_step_ids),
                residual_packet=packet,
                residual_packet_required=True,
                residual_packet_sealed=True,
                notes=(
                    "replan_required",
                    "bound_records_only",
                    "residual_packet_sealed",
                    "edits_only_bound_records",
                ),
            )

        # Replan ready but residual packet missing / unsealable → abstain.
        notes = [
            "replan_required",
            "bound_records_only",
            "residual_packet_required",
            "provider_forbidden",
        ]
        if packet_reason:
            notes.append(packet_reason)
        return self._result(
            outcome=FailureReplanOutcome.ABSTAIN_REVIEW,
            reason_code=REASON_REPLAN_ABSTAIN_NO_PACKET,
            disposition=ImplementationDisposition.ABSTAIN_REVIEW,
            delta=delta,
            memory_disposition=memory_disposition,
            bound_step_ids=tuple(delta.invalidated_step_ids),
            residual_packet=None,
            residual_packet_required=True,
            residual_packet_sealed=False,
            notes=tuple(notes),
        )

    def authorize_llm_retry(
        self,
        result: FailureReplanResult | Mapping[str, Any],
        *,
        residual_packet: ResidualLlmPacket | Mapping[str, Any] | None = None,
        free_reprompt_context: Any = None,
    ) -> ProviderRetryAuthorization:
        """Admit provider retry only with a sealed residual packet.

        Free re-prompt context is always rejected.  A prior
        :class:`FailureReplanResult` that already carries a sealed packet may
        be re-authorized; otherwise an explicit packet must be supplied.
        """

        if free_reprompt_context is not None:
            raise FreeRepromptForbiddenError(
                "LLM retry free_reprompt_context is forbidden after typed failure"
            )

        if isinstance(result, Mapping):
            # Mapping path only supports disposition + packet projection fields.
            disposition_token = str(result.get("disposition") or "").strip()
            try:
                disposition = ImplementationDisposition(disposition_token)
            except ValueError as exc:
                raise FailureReplanPolicyInputError(
                    f"unknown disposition: {disposition_token!r}",
                    reason_code=REASON_MALFORMED_REQUEST,
                ) from exc
            packet_payload = residual_packet
            if packet_payload is None:
                packet_payload = result.get("residual_packet")
            sealed = bool(result.get("residual_packet_sealed"))
            if packet_payload is None or not sealed:
                return ProviderRetryAuthorization(
                    authorized=False,
                    reason_code=REASON_RESIDUAL_PACKET_REQUIRED,
                    disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                    residual_packet_id="",
                    residual_packet=None,
                    free_reprompt_allowed=False,
                )
            packet = self._coerce_packet(packet_payload)
            if disposition is not ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED:
                return ProviderRetryAuthorization(
                    authorized=False,
                    reason_code=REASON_RESIDUAL_PACKET_REQUIRED,
                    disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                    residual_packet_id="",
                    residual_packet=None,
                    free_reprompt_allowed=False,
                )
            return ProviderRetryAuthorization(
                authorized=True,
                reason_code=REASON_REPLAN_RESIDUAL_AUTHORIZED,
                disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
                residual_packet_id=packet.packet_id or packet.content_id,
                residual_packet=packet,
                free_reprompt_allowed=False,
            )

        if not isinstance(result, FailureReplanResult):
            raise FailureReplanPolicyInputError(
                "result must be FailureReplanResult or mapping",
                reason_code=REASON_MALFORMED_REQUEST,
            )

        packet: ResidualLlmPacket | None = None
        if residual_packet is not None:
            packet = self._coerce_packet(residual_packet)
        elif result.residual_packet is not None and result.residual_packet_sealed:
            packet = result.residual_packet

        if packet is None:
            return ProviderRetryAuthorization(
                authorized=False,
                reason_code=REASON_RESIDUAL_PACKET_REQUIRED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                residual_packet_id="",
                residual_packet=None,
                free_reprompt_allowed=False,
            )

        if (
            result.disposition is not ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
            and residual_packet is None
        ):
            # Explicit new packet can elevate only when replan was active.
            if not result.should_replan:
                return ProviderRetryAuthorization(
                    authorized=False,
                    reason_code=REASON_RESIDUAL_PACKET_REQUIRED,
                    disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                    residual_packet_id="",
                    residual_packet=None,
                    free_reprompt_allowed=False,
                )

        if self.require_residual_packet_for_llm and not packet_satisfies_residual_llm_contract(
            packet
        ):
            return ProviderRetryAuthorization(
                authorized=False,
                reason_code=REASON_PACKET_SEAL_FAILED,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                residual_packet_id="",
                residual_packet=None,
                free_reprompt_allowed=False,
            )

        # Backoff / exhausted / unbound paths never authorize LLM.
        if result.outcome in {
            FailureReplanOutcome.BACKOFF,
            FailureReplanOutcome.EXHAUSTED,
            FailureReplanOutcome.UNBOUND,
            FailureReplanOutcome.BOUND_EXCEEDED,
            FailureReplanOutcome.CANCELLED,
        }:
            return ProviderRetryAuthorization(
                authorized=False,
                reason_code=result.reason_code,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                residual_packet_id="",
                residual_packet=None,
                free_reprompt_allowed=False,
            )

        return ProviderRetryAuthorization(
            authorized=True,
            reason_code=REASON_REPLAN_RESIDUAL_AUTHORIZED,
            disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            residual_packet_id=packet.packet_id or packet.content_id,
            residual_packet=packet,
            free_reprompt_allowed=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_request(
        self,
        request: FailureReplanRequest | Mapping[str, Any],
    ) -> FailureReplanRequest:
        if isinstance(request, FailureReplanRequest):
            return request
        if not isinstance(request, Mapping):
            raise FailureReplanPolicyInputError(
                "request must be FailureReplanRequest or mapping",
                reason_code=REASON_MALFORMED_REQUEST,
            )
        _assert_body_free(
            {
                key: value
                for key, value in request.items()
                if key
                not in {
                    "plan",
                    "observation",
                    "residual_materials",
                    "residual_packet",
                }
            },
            "request",
        )
        if request.get("free_reprompt_context") is not None:
            raise FreeRepromptForbiddenError()
        return FailureReplanRequest(
            plan=request.get("plan") or {},
            observation=request.get("observation") or {},
            residual_materials=request.get("residual_materials"),
            residual_packet=request.get("residual_packet"),
            observed_at_milliseconds=int(
                request.get("observed_at_milliseconds") or 1
            ),
            now_milliseconds=request.get("now_milliseconds"),
            deadline_milliseconds=request.get("deadline_milliseconds"),
            free_reprompt_context=request.get("free_reprompt_context"),
            metadata=request.get("metadata") or {},
        )

    def _resolve_residual_packet(
        self,
        request: FailureReplanRequest,
    ) -> tuple[ResidualLlmPacket | None, str]:
        if request.residual_packet is not None:
            packet = request.residual_packet
            if not isinstance(packet, ResidualLlmPacket):
                try:
                    packet = ResidualLlmPacket.from_dict(packet)  # type: ignore[arg-type]
                except ResidualLlmPacketError:
                    return None, REASON_PACKET_SEAL_FAILED
            if not packet_satisfies_residual_llm_contract(packet):
                return None, REASON_PACKET_SEAL_FAILED
            return packet, ""
        materials = request.residual_materials
        if materials is None:
            return None, REASON_REPLAN_ABSTAIN_NO_PACKET
        if not isinstance(materials, ResidualPacketMaterials):
            try:
                materials = ResidualPacketMaterials.from_dict(materials)  # type: ignore[arg-type]
            except FailureReplanPolicyError:
                return None, REASON_PACKET_SEAL_FAILED
        try:
            return materials.seal(), ""
        except ResidualLlmPacketError:
            return None, REASON_PACKET_SEAL_FAILED
        except FailureReplanPolicyError:
            return None, REASON_PACKET_SEAL_FAILED

    def _coerce_packet(
        self,
        packet: ResidualLlmPacket | Mapping[str, Any],
    ) -> ResidualLlmPacket:
        if isinstance(packet, ResidualLlmPacket):
            if not packet_satisfies_residual_llm_contract(packet):
                raise ResidualPacketRequiredError(
                    "residual packet failed ResidualLlmPacket@1 contract"
                )
            return packet
        if not isinstance(packet, Mapping):
            raise ResidualPacketRequiredError(
                "residual packet must be ResidualLlmPacket or mapping"
            )
        try:
            sealed = ResidualLlmPacket.from_dict(packet)
        except ResidualLlmPacketError as exc:
            raise ResidualPacketRequiredError(
                f"residual packet seal invalid: {exc}"
            ) from exc
        if not packet_satisfies_residual_llm_contract(sealed):
            raise ResidualPacketRequiredError(
                "residual packet failed ResidualLlmPacket@1 contract"
            )
        return sealed

    @staticmethod
    def _assert_bound_records_only(delta: DeltaReplanDecision) -> None:
        """Ensure replan edits only the dependency-closed bound suffix."""

        if not delta.changed:
            if delta.invalidated_step_ids:
                raise FailureReplanPolicyError(
                    "non-repair decision must not invalidate steps",
                    reason_code=REASON_BOUND_RECORD_ESCAPE,
                )
            return
        bound = set(delta.invalidated_step_ids)
        # Dependency-minimal suffix is already enforced by DeltaReplanDecision;
        # additionally require anchors ⊆ invalidated and preserved disjoint.
        anchors = set(delta.direct_failure_step_ids)
        if not anchors:
            raise FailureReplanPolicyError(
                "active replan requires bound failure anchors",
                reason_code=REASON_BOUND_RECORD_ESCAPE,
            )
        if not anchors.issubset(bound):
            raise FailureReplanPolicyError(
                "bound failure anchors must be included in invalidated steps",
                reason_code=REASON_BOUND_RECORD_ESCAPE,
            )
        if bound & set(delta.preserved_step_ids):
            raise FailureReplanPolicyError(
                "replan cannot preserve steps it invalidates",
                reason_code=REASON_BOUND_RECORD_ESCAPE,
            )
        # Unaffected accepted steps must stay accepted.
        by_id = {item.step_id: item for item in delta.resulting_plan.steps}
        for step_id in delta.preserved_step_ids:
            step = by_id.get(step_id)
            if step is None or not step.accepted:
                raise FailureReplanPolicyError(
                    f"preserved step {step_id!r} must remain accepted",
                    reason_code=REASON_BOUND_RECORD_ESCAPE,
                )
        for step_id in delta.invalidated_step_ids:
            step = by_id.get(step_id)
            if step is None or step.accepted or step.evidence_ids:
                raise FailureReplanPolicyError(
                    f"invalidated step {step_id!r} must be reopened without evidence",
                    reason_code=REASON_BOUND_RECORD_ESCAPE,
                )

    @staticmethod
    def _memory_disposition_for(delta: DeltaReplanDecision) -> str:
        mapping = {
            DeltaReplanStopReason.REPLAN_REQUIRED: (
                FailureMemoryDisposition.NEW_FAILURE.value
                if not delta.diagnostic_reused
                else FailureMemoryDisposition.CHANGED_EVIDENCE.value
            ),
            DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF: (
                FailureMemoryDisposition.UNCHANGED_BACKOFF.value
            ),
            DeltaReplanStopReason.IDENTICAL_FAILURE_EXHAUSTED: (
                FailureMemoryDisposition.IDENTICAL_FAILURE_EXHAUSTED.value
            ),
            DeltaReplanStopReason.RETRY_BUDGET_EXHAUSTED: (
                FailureMemoryDisposition.RETRY_BUDGET_EXHAUSTED.value
            ),
            DeltaReplanStopReason.FAILURE_MEMORY_BOUND_REACHED: (
                FailureMemoryDisposition.MEMORY_BOUND_REACHED.value
            ),
        }
        return mapping.get(delta.stop_reason, "")

    def _result(
        self,
        *,
        outcome: FailureReplanOutcome,
        reason_code: str,
        disposition: ImplementationDisposition,
        delta: DeltaReplanDecision,
        memory_disposition: str,
        bound_step_ids: tuple[str, ...],
        residual_packet: ResidualLlmPacket | None,
        residual_packet_required: bool,
        residual_packet_sealed: bool,
        notes: tuple[str, ...],
    ) -> FailureReplanResult:
        return FailureReplanResult(
            outcome=outcome,
            reason_code=reason_code,
            disposition=disposition,
            delta_decision=delta,
            memory_disposition=memory_disposition,
            backoff_milliseconds=delta.backoff_milliseconds,
            backoff_attempt=delta.backoff_attempt,
            bound_step_ids=bound_step_ids,
            invalidated_step_ids=tuple(delta.invalidated_step_ids),
            preserved_step_ids=tuple(delta.preserved_step_ids),
            residual_packet=residual_packet,
            residual_packet_required=residual_packet_required,
            residual_packet_sealed=residual_packet_sealed,
            provider_hook_count=0,
            free_reprompt_allowed=False,
            notes=notes,
        )


def build_failure_replan_policy(
    *,
    failure_memory: PlanFailureMemory | None = None,
    backoff_policy: FailureBackoffPolicy | None = None,
    delta_limits: DeltaReplanLimits | Mapping[str, Any] | None = None,
    replan_limits: ReplanLimits | Mapping[str, Any] | None = None,
    formal_replanner: FormalReplanner | None = None,
    require_residual_packet_for_llm: bool = True,
) -> FailureReplanPolicy:
    """Construct the production-default failure replan policy."""

    memory = failure_memory
    if memory is None:
        memory = PlanFailureMemory(policy=backoff_policy or FailureBackoffPolicy())
    elif backoff_policy is not None and memory.policy != backoff_policy:
        raise FailureReplanPolicyInputError(
            "failure_memory policy does not match requested backoff policy",
            reason_code="policy_mismatch",
        )
    return FailureReplanPolicy(
        failure_memory=memory,
        delta_limits=delta_limits,
        replan_limits=replan_limits,
        formal_replanner=formal_replanner,
        require_residual_packet_for_llm=require_residual_packet_for_llm,
    )


def evaluate_failure_replan(
    request: FailureReplanRequest | Mapping[str, Any],
    *,
    failure_memory: PlanFailureMemory | None = None,
    cancelled: Any = None,
) -> FailureReplanResult:
    """Module-level convenience wrapper around :meth:`FailureReplanPolicy.evaluate`."""

    return build_failure_replan_policy(
        failure_memory=failure_memory,
    ).evaluate(request, cancelled=cancelled)


def authorize_llm_retry_after_failure(
    result: FailureReplanResult | Mapping[str, Any],
    *,
    residual_packet: ResidualLlmPacket | Mapping[str, Any] | None = None,
    free_reprompt_context: Any = None,
    failure_memory: PlanFailureMemory | None = None,
) -> ProviderRetryAuthorization:
    """Module-level convenience wrapper for residual-gated LLM retry admission."""

    return build_failure_replan_policy(
        failure_memory=failure_memory,
    ).authorize_llm_retry(
        result,
        residual_packet=residual_packet,
        free_reprompt_context=free_reprompt_context,
    )


__all__ = [
    "FAILURE_REPLAN_POLICY_EVIDENCE",
    "FAILURE_REPLAN_POLICY_INTERFACE",
    "FAILURE_REPLAN_POLICY_PRODUCER",
    "FAILURE_REPLAN_POLICY_VERSION",
    "FAILURE_REPLAN_REQUEST_SCHEMA",
    "FAILURE_REPLAN_RESULT_SCHEMA",
    "PROVIDER_RETRY_AUTHORIZATION_SCHEMA",
    "REASON_FREE_REPROMPT_FORBIDDEN",
    "REASON_IDENTICAL_FAILURE_EXHAUSTED",
    "REASON_REPLAN_ABSTAIN_NO_PACKET",
    "REASON_REPLAN_RESIDUAL_AUTHORIZED",
    "REASON_RESIDUAL_PACKET_REQUIRED",
    "REASON_UNCHANGED_FAILURE_BACKOFF",
    "FailureReplanOutcome",
    "FailureReplanPolicy",
    "FailureReplanPolicyError",
    "FailureReplanPolicyInputError",
    "FailureReplanRequest",
    "FailureReplanResult",
    "FreeRepromptForbiddenError",
    "ProviderRetryAuthorization",
    "ResidualPacketMaterials",
    "ResidualPacketRequiredError",
    "authorize_llm_retry_after_failure",
    "build_failure_replan_policy",
    "evaluate_failure_replan",
]
