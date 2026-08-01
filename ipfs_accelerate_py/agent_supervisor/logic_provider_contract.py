"""Lazy supervisor facade for the canonical datasets logic-provider contract.

The canonical wire types live in ``ipfs_datasets_py.logic.backends.provider``.
This module is an additive adapter for the supervisor's established
``proof.formal_verification_provider`` API; it does not replace that API or
register routes.  Importantly, importing or constructing a facade does not
import the datasets package.  The canonical module and concrete provider are
loaded only for an explicit conversion or invocation.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping
from typing import Any, Final

from .proof.formal_verification_provider import (
    PROOF_PROVIDER_PROTOCOL_VERSION,
    CancellationToken,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
)

CANONICAL_LOGIC_PROVIDER_MODULE: Final = (
    "ipfs_datasets_py.logic.backends.provider"
)


class LogicProviderFacadeError(RuntimeError):
    """Raised when a lazy provider declaration cannot satisfy its contract."""


def _canonical_contract(module_name: str = CANONICAL_LOGIC_PROVIDER_MODULE) -> Any:
    """Import the canonical contract only after an explicit boundary call."""

    return importlib.import_module(module_name)


def _resource_budget_payload(request: ProviderRequest) -> dict[str, Any]:
    payload = request.resource_budget.to_dict()
    # The supervisor's durable contract calls this discriminator ``schema``;
    # the cross-package provider wire contract calls it ``schema_version``.
    payload.pop("schema", None)
    payload.pop("schema_version", None)
    return payload


def to_logic_provider_request(
    request: ProviderRequest,
    *,
    cancellation: CancellationToken | None = None,
    contract_module: Any | None = None,
) -> Any:
    """Convert a supervisor request to the canonical datasets request type.

    Cancellation is a wire snapshot.  In-process callers can invoke again with
    an updated snapshot; subprocess/process clients remain responsible for
    terminating a running child when their live token changes.
    """

    if not isinstance(request, ProviderRequest):
        raise TypeError("request must be a supervisor ProviderRequest")
    contract = contract_module or _canonical_contract()
    cancellation_payload = None
    if cancellation is not None:
        cancelled = bool(cancellation.is_cancelled())
        cancellation_payload = contract.ProviderCancellation(
            cancellation_id=f"request:{request.request_id}",
            cancelled=cancelled,
            reason="supervisor cancellation requested" if cancelled else "",
        )
    return contract.LogicProviderRequest(
        operation=request.operation.value,
        payload=request.payload,
        request_id=request.request_id,
        resource_budget=contract.ProviderResourceBudget(
            **_resource_budget_payload(request)
        ),
        cancellation=cancellation_payload,
        network_allowed=request.network_allowed,
        deadline_unix_ms=request.deadline_unix_ms,
        protocol_version=request.protocol_version,
    )


def to_supervisor_provider_response(
    request: ProviderRequest,
    response: Any,
    *,
    contract_module: Any | None = None,
) -> ProviderResponse:
    """Validate and convert a canonical response to the supervisor envelope."""

    if not isinstance(request, ProviderRequest):
        raise TypeError("request must be a supervisor ProviderRequest")
    contract = contract_module or _canonical_contract()
    if isinstance(response, Mapping):
        response = contract.LogicProviderResponse.from_dict(response)
    if not isinstance(response, contract.LogicProviderResponse):
        raise LogicProviderFacadeError(
            "canonical provider returned an unsupported response type"
        )
    if response.request_id != request.request_id:
        raise LogicProviderFacadeError("canonical provider response request_id mismatch")
    if response.operation.value != request.operation.value:
        raise LogicProviderFacadeError("canonical provider response operation mismatch")
    if response.protocol_version != request.protocol_version:
        raise LogicProviderFacadeError(
            "canonical provider response protocol version mismatch"
        )
    if response.ok:
        assert response.result is not None
        return ProviderResponse.success(
            request,
            response.result,
            provider_id=response.provider_id,
            provider_version=response.provider_version,
            duration_ms=response.duration_ms,
        )
    assert response.error is not None
    return ProviderResponse.failure(
        request,
        response.error.code.value,
        response.error.message,
        retryable=response.error.retryable,
        details=response.error.details,
        provider_id=response.provider_id,
        provider_version=response.provider_version,
        duration_ms=response.duration_ms,
    )


class LazyLogicProviderReference:
    """A ``module:attribute`` declaration that imports nothing until loaded."""

    def __init__(self, target: str) -> None:
        if (
            not isinstance(target, str)
            or target != target.strip()
            or target.count(":") != 1
        ):
            raise ValueError(
                "logic-provider target must be a trimmed module:attribute reference"
            )
        module_name, attribute = target.split(":", 1)
        if not module_name or not attribute or "." in attribute:
            raise ValueError(
                "logic-provider target must name one top-level module attribute"
            )
        self.target = target
        self._module_name = module_name
        self._attribute = attribute

    def load(self) -> Any:
        module = importlib.import_module(self._module_name)
        provider = getattr(module, self._attribute)
        # Provider factories are explicit at the reference boundary.  Classes
        # are instantiated; ordinary callable provider objects are preserved.
        return provider() if isinstance(provider, type) else provider


class SupervisorLogicProviderFacade:
    """Expose a canonical datasets ``LogicProvider`` through the supervisor API.

    The facade has stable declared identity before its provider is loaded, so
    metadata-only discovery and route planning remain side-effect-free.
    """

    protocol_version = PROOF_PROVIDER_PROTOCOL_VERSION

    def __init__(
        self,
        *,
        provider_id: str,
        provider_version: str,
        provider: Any | None = None,
        loader: Callable[[], Any] | None = None,
        contract_module_name: str = CANONICAL_LOGIC_PROVIDER_MODULE,
    ) -> None:
        if (provider is None) == (loader is None):
            raise ValueError("provide exactly one of provider or loader")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(provider_version, str) or not provider_version.strip():
            raise ValueError("provider_version must be a non-empty string")
        if not isinstance(contract_module_name, str) or not contract_module_name.strip():
            raise ValueError("contract_module_name must be a non-empty string")
        self.provider_id = provider_id.strip()
        self.provider_version = provider_version.strip()
        self._provider = provider
        self._loader = loader
        self._contract_module_name = contract_module_name.strip()
        self._contract_cache: Any | None = None
        self._load_lock = threading.Lock()
        if provider is not None:
            self._validate_provider_identity(provider)

    @classmethod
    def from_reference(
        cls,
        target: str,
        *,
        provider_id: str,
        provider_version: str,
        contract_module_name: str = CANONICAL_LOGIC_PROVIDER_MODULE,
    ) -> SupervisorLogicProviderFacade:
        reference = LazyLogicProviderReference(target)
        return cls(
            provider_id=provider_id,
            provider_version=provider_version,
            loader=reference.load,
            contract_module_name=contract_module_name,
        )

    @property
    def loaded(self) -> bool:
        """Whether the concrete provider has been resolved."""

        return self._provider is not None

    def _contract(self) -> Any:
        contract = self._contract_cache
        if contract is None:
            with self._load_lock:
                contract = self._contract_cache
                if contract is None:
                    contract = _canonical_contract(self._contract_module_name)
                    self._contract_cache = contract
        return contract

    def _validate_provider_identity(self, provider: Any) -> None:
        actual_id = str(getattr(provider, "provider_id", "")).strip()
        actual_version = str(getattr(provider, "provider_version", "")).strip()
        actual_protocol = getattr(provider, "protocol_version", None)
        if actual_id != self.provider_id or actual_version != self.provider_version:
            raise LogicProviderFacadeError(
                "loaded logic-provider identity differs from its declaration"
            )
        if actual_protocol != self.protocol_version:
            raise LogicProviderFacadeError(
                "loaded logic provider does not implement protocol version 1"
            )

    def _load_provider(self) -> Any:
        provider = self._provider
        if provider is not None:
            return provider
        with self._load_lock:
            provider = self._provider
            if provider is None:
                assert self._loader is not None
                provider = self._loader()
                self._validate_provider_identity(provider)
                self._provider = provider
        return provider

    def invoke(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ProviderResponse:
        """Invoke the canonical provider and return the legacy supervisor type."""

        if not isinstance(request, ProviderRequest):
            raise TypeError("request must be a supervisor ProviderRequest")
        if request.protocol_version != self.protocol_version:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.PROTOCOL_ERROR,
                "logic-provider facade does not support the requested protocol version",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        if cancellation is not None and cancellation.is_cancelled():
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.CANCELLED,
                "logic-provider request was cancelled",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        if request.expired:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.TIMED_OUT,
                "logic-provider request deadline has expired",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        try:
            contract = self._contract()
            canonical_request = to_logic_provider_request(
                request,
                cancellation=cancellation,
                contract_module=contract,
            )
        except Exception as error:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_REQUEST,
                f"request cannot cross the logic-provider boundary: {str(error)[:512]}",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        try:
            provider = self._load_provider()
        except Exception as error:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.UNAVAILABLE,
                f"logic provider could not be loaded: {type(error).__name__}",
                retryable=True,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )

        canonical_response = contract.dispatch_logic_provider_request(
            provider, canonical_request
        )
        try:
            response = to_supervisor_provider_response(
                request,
                canonical_response,
                contract_module=contract,
            )
        except Exception as error:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_RESPONSE,
                f"response cannot cross the logic-provider boundary: {str(error)[:512]}",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        if response.provider_id and response.provider_id != self.provider_id:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_RESPONSE,
                "logic provider response identity mismatch",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        if (
            response.provider_version
            and response.provider_version != self.provider_version
        ):
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.MALFORMED_RESPONSE,
                "logic provider response version mismatch",
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        return response

    def capability(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

    def translate(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

    def prove(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

    def reconstruct(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

    def verify(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)

    def attest(self, request: ProviderRequest) -> ProviderResponse:
        return self.invoke(request)


__all__ = [
    "CANONICAL_LOGIC_PROVIDER_MODULE",
    "LazyLogicProviderReference",
    "LogicProviderFacadeError",
    "SupervisorLogicProviderFacade",
    "to_logic_provider_request",
    "to_supervisor_provider_response",
]
