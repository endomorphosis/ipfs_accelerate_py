"""DCR-000 non-prompt adapter for an admitted deterministic repair machine.

This module deliberately contains no provider SDK, subprocess, network, or
prompt path.  It is a narrow adapter over a caller-injected state-machine
object with an ``advance`` method.  Inputs and outputs are typed, CIDs are
bound through the canonical DCR-002 envelopes, and every rejection is a typed
terminal result rather than a fallback request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from ..autonomous_repair.capabilities import CapabilityReceipt
from ..autonomous_repair.contracts import (
    DeterministicRepairDisposition,
    RepairEvidenceEnvelope,
)
from ..autonomous_repair.no_llm_policy import (
    DeterministicRepairAuthorityPolicy,
    NoLlmExecutionDenied,
    RepairExecutionRoute,
)

DETERMINISTIC_REPAIR_PROVIDER_INTERFACE: Final[str] = "DeterministicRepairProvider@1"
DETERMINISTIC_REPAIR_PROVIDER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-provider-result@1"
)


def _opaque_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        character.isspace() for character in value
    ):
        raise ValueError(f"{name} must be a non-empty opaque identifier")
    return value


def _zero_counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{name} must be exactly zero")
    return 0


def _forbidden_identifier(value: str) -> bool:
    parts = {
        part
        for part in "".join(char if char.isalnum() else "_" for char in value.lower()).split("_")
        if part
    }
    return bool(
        parts.intersection(
            {
                "model", "llm", "provider", "prompt", "fallback", "residual",
                "rescue", "retry", "shell", "command", "subprocess", "remote",
            }
        )
    )


@dataclass(frozen=True)
class DeterministicRepairRequest:
    """Typed request accepted by the adapter; strings/prompts are not requests."""

    request_id: str
    state: RepairEvidenceEnvelope
    model_call_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _opaque_id(self.request_id, "request_id"))
        if not isinstance(self.state, RepairEvidenceEnvelope):
            raise ValueError("state must be a RepairEvidenceEnvelope")
        object.__setattr__(self, "model_call_count", _zero_counter(self.model_call_count, "model_call_count"))

    @property
    def input_evidence_cid(self) -> str:
        return self.state.content_id


@dataclass(frozen=True)
class DeterministicRepairTransition:
    """A typed state-machine output, prior to adapter validation/publication."""

    disposition: DeterministicRepairDisposition
    state: RepairEvidenceEnvelope | None = None
    model_call_count: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, DeterministicRepairDisposition):
            raise ValueError("disposition must be DeterministicRepairDisposition")
        if self.state is not None and not isinstance(self.state, RepairEvidenceEnvelope):
            raise ValueError("transition state must be a RepairEvidenceEnvelope")
        object.__setattr__(self, "model_call_count", _zero_counter(self.model_call_count, "model_call_count"))
        if self.reason_code:
            object.__setattr__(self, "reason_code", _opaque_id(self.reason_code, "reason_code"))


@runtime_checkable
class DeterministicRepairStateMachine(Protocol):
    """The only executable dependency shape accepted by this adapter."""

    def advance(self, request: DeterministicRepairRequest) -> DeterministicRepairTransition:
        """Return one typed deterministic transition without side effects here."""


@dataclass(frozen=True)
class DeterministicRepairStateMachineBinding:
    """Explicit state-machine identity and local-logic admission pin."""

    machine_id: str
    pin: str
    machine: DeterministicRepairStateMachine
    capability: CapabilityReceipt
    route: RepairExecutionRoute = RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC
    declared_model_call_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "machine_id", _opaque_id(self.machine_id, "machine_id"))
        object.__setattr__(self, "pin", _opaque_id(self.pin, "pin"))
        if _forbidden_identifier(self.machine_id):
            raise ValueError("state machine identity names a forbidden route")
        if callable(self.machine) or not callable(getattr(self.machine, "advance", None)):
            raise ValueError("machine must be a non-callable object with advance(request)")
        if not isinstance(self.capability, CapabilityReceipt):
            raise ValueError("machine requires a CapabilityReceipt attestation")
        if (
            not self.capability.available
            or not self.capability.capability_id.startswith("ipfs_datasets_py.logic.")
            or not self.capability.origin
        ):
            raise ValueError("machine requires an available ipfs_datasets_py.logic capability")
        if self.pin != self.capability.receipt_id:
            raise ValueError("machine pin must equal the exact capability receipt_id")
        if self.route is not RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC:
            raise ValueError("provider admits deterministic local logic only")
        object.__setattr__(
            self,
            "declared_model_call_count",
            _zero_counter(self.declared_model_call_count, "declared_model_call_count"),
        )


@dataclass(frozen=True)
class DeterministicRepairProviderResult:
    """Published adapter outcome; this adapter never grants completion authority."""

    request_id: str
    disposition: DeterministicRepairDisposition
    input_evidence_cid: str = ""
    output_evidence_cid: str = ""
    reason_code: str = ""
    model_call_count: int = 0
    invoked: bool = False
    completion_authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _opaque_id(self.request_id, "request_id"))
        if not isinstance(self.disposition, DeterministicRepairDisposition):
            raise ValueError("disposition must be DeterministicRepairDisposition")
        for name in ("input_evidence_cid", "output_evidence_cid", "reason_code"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _opaque_id(value, name))
        object.__setattr__(self, "model_call_count", _zero_counter(self.model_call_count, "model_call_count"))
        if self.completion_authoritative:
            raise ValueError("deterministic repair provider cannot grant completion authority")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_REPAIR_PROVIDER_RESULT_SCHEMA,
            "interface": DETERMINISTIC_REPAIR_PROVIDER_INTERFACE,
            "request_id": self.request_id,
            "disposition": self.disposition.value,
            "input_evidence_cid": self.input_evidence_cid,
            "output_evidence_cid": self.output_evidence_cid,
            "reason_code": self.reason_code,
            "model_call_count": 0,
            "invoked": self.invoked,
            "completion_authoritative": False,
            "fallback_authorized": False,
        }


class DeterministicRepairProvider:
    """Adapt one explicitly admitted local state machine, with no fallback path."""

    INTERFACE: Final[str] = DETERMINISTIC_REPAIR_PROVIDER_INTERFACE

    def __init__(
        self,
        binding: DeterministicRepairStateMachineBinding,
        *,
        authority_policy: DeterministicRepairAuthorityPolicy,
    ) -> None:
        if not isinstance(binding, DeterministicRepairStateMachineBinding):
            raise ValueError("binding must be DeterministicRepairStateMachineBinding")
        self.binding = binding
        if not isinstance(authority_policy, DeterministicRepairAuthorityPolicy):
            raise ValueError("authority_policy must be DeterministicRepairAuthorityPolicy")
        self.authority_policy = authority_policy

    def execute(self, request: object) -> DeterministicRepairProviderResult:
        """Run one transition or return a typed rejection without a fallback."""

        if not isinstance(request, DeterministicRepairRequest):
            return self._rejected("invalid-request")
        if request.model_call_count != 0 or self.binding.declared_model_call_count != 0:
            return self._rejected("nonzero-model-counter", request)
        try:
            transition = self.authority_policy.invoke(
                self.binding.route,
                self.binding.machine.advance,
                request,
                pin=self.binding.pin,
            )
        except NoLlmExecutionDenied:
            return self._rejected("execution-route-denied", request)
        except Exception:
            # State-machine failure is a deterministic defer, never a rescue
            # provider, prompt, or retry route.
            return self._terminal(DeterministicRepairDisposition.DEFER_CAPABILITY, "state-machine-error", request)
        return self._publish(transition, request)

    run = execute

    def _publish(
        self,
        transition: object,
        request: DeterministicRepairRequest,
    ) -> DeterministicRepairProviderResult:
        if not isinstance(transition, DeterministicRepairTransition):
            return self._rejected("unknown-state-machine-output", request)
        if transition.model_call_count != 0:
            return self._rejected("nonzero-model-counter", request)
        if transition.state is None:
            if transition.disposition not in {
                DeterministicRepairDisposition.ABSTAIN_REVIEW,
                DeterministicRepairDisposition.DEFER_CAPABILITY,
                DeterministicRepairDisposition.REJECTED,
            }:
                return self._rejected("missing-output-state", request)
            return self._terminal(transition.disposition, transition.reason_code or "terminal-state-machine-result", request, invoked=True)
        try:
            transition.state.require_advances(request.state)
        except Exception:
            return self._rejected("invalid-state-transition", request)
        if transition.disposition is not transition.state.disposition:
            return self._rejected("output-disposition-mismatch", request)
        # A completed envelope is evidence supplied by the machine/contracts;
        # the adapter reports no completion authority of its own.
        return DeterministicRepairProviderResult(
            request_id=request.request_id,
            disposition=transition.disposition,
            input_evidence_cid=request.input_evidence_cid,
            output_evidence_cid=transition.state.content_id,
            reason_code=transition.reason_code or "typed-state-transition",
            invoked=True,
        )

    @staticmethod
    def _rejected(
        reason_code: str,
        request: DeterministicRepairRequest | None = None,
    ) -> DeterministicRepairProviderResult:
        return DeterministicRepairProviderResult(
            request_id=request.request_id if request else "rejected-request",
            disposition=DeterministicRepairDisposition.REJECTED,
            input_evidence_cid=request.input_evidence_cid if request else "",
            reason_code=reason_code,
        )

    @staticmethod
    def _terminal(
        disposition: DeterministicRepairDisposition,
        reason_code: str,
        request: DeterministicRepairRequest,
        *,
        invoked: bool = False,
    ) -> DeterministicRepairProviderResult:
        return DeterministicRepairProviderResult(
            request_id=request.request_id,
            disposition=disposition,
            input_evidence_cid=request.input_evidence_cid,
            reason_code=reason_code,
            invoked=invoked,
        )


def run_deterministic_repair(
    provider: DeterministicRepairProvider,
    request: object,
) -> DeterministicRepairProviderResult:
    """Functional entry point; no implicit provider or fallback is constructed."""

    if not isinstance(provider, DeterministicRepairProvider):
        return DeterministicRepairProvider._rejected("invalid-provider")
    return provider.execute(request)


__all__ = [
    "DETERMINISTIC_REPAIR_PROVIDER_INTERFACE",
    "DETERMINISTIC_REPAIR_PROVIDER_RESULT_SCHEMA",
    "DeterministicRepairProvider",
    "DeterministicRepairProviderResult",
    "DeterministicRepairRequest",
    "DeterministicRepairStateMachine",
    "DeterministicRepairStateMachineBinding",
    "DeterministicRepairTransition",
    "run_deterministic_repair",
]
