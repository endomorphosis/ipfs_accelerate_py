"""Capability-isolated Leanstral model-draft provider.

This adapter deliberately has a narrow authority boundary:

* model inference is always routed through :mod:`llm_router` with a concrete
  provider, model, timeout, and output-token limit;
* importing this module never imports Leanstral, spaCy, a codec, model
  bindings, or model weights;
* returned text is an immutable, unverified draft artifact; and
* kernel checking is not an operation exposed by this provider.

The generic proof-provider transport catches failures and enforces process
budgets.  This module adds the model-specific budget and trust constraints
needed before a Leanstral route can be scheduled.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from .formal_verification_contracts import AssuranceLevel, ProofStage
from .formal_verification_provider import (
    PROOF_PROVIDER_PROTOCOL_VERSION,
    ProofProviderError,
    ProviderFailureCode,
    ProviderRequest,
)


LEANSTRAL_PROOF_PROVIDER_ID: Final = "leanstral"
LEANSTRAL_PROOF_PROVIDER_VERSION: Final = "1.0.0"
LEANSTRAL_DRAFT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-proof-draft@1"
)
LEANSTRAL_MODEL_RESOURCE_CLASS: Final = "model"
LEAN_KERNEL_RESOURCE_CLASS: Final = "kernel"

DEFAULT_LEANSTRAL_LLM_PROVIDER: Final = "leanstral_local"
DEFAULT_LEANSTRAL_MODEL: Final = "Leanstral"
DEFAULT_LEANSTRAL_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_LEANSTRAL_MAX_NEW_TOKENS: Final = 1_400
DEFAULT_LEANSTRAL_MAX_PROMPT_BYTES: Final = 512 * 1024
DEFAULT_LEANSTRAL_MAX_OUTPUT_BYTES: Final = 2 * 1024 * 1024

_ROUTER_ALIASES = frozenset({"", "auto", "default", "llm_router", "router"})
_CANONICAL_MUTATION_KEYS = frozenset(
    {
        "canonical_source",
        "canonical_sources",
        "canonical_obligation",
        "canonical_obligations",
        "obligation",
        "obligations",
        "source",
        "sources",
    }
)

LLMGenerate = Callable[..., str]


def _nonempty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _json_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain strict JSON values") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must be an object")
    return decoded


def _identifiers(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        raise ValueError(f"{field_name} must be a string or array of strings")
    normalized: list[str] = []
    for raw in raw_values:
        item = _nonempty_text(raw, field_name=field_name)
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class LeanstralResourceIsolation:
    """Distinct scheduler classes for untrusted inference and trusted checking."""

    model_resource_class: str = LEANSTRAL_MODEL_RESOURCE_CLASS
    kernel_resource_class: str = LEAN_KERNEL_RESOURCE_CLASS

    def __post_init__(self) -> None:
        model = _nonempty_text(
            self.model_resource_class, field_name="model_resource_class"
        )
        kernel = _nonempty_text(
            self.kernel_resource_class, field_name="kernel_resource_class"
        )
        if model == kernel:
            raise ValueError(
                "Leanstral inference and kernel checking need different resource classes"
            )
        object.__setattr__(self, "model_resource_class", model)
        object.__setattr__(self, "kernel_resource_class", kernel)

    def to_dict(self) -> dict[str, str]:
        return {
            "model_inference": self.model_resource_class,
            "kernel_check": self.kernel_resource_class,
        }


@dataclass(frozen=True)
class LeanstralProofProviderConfig:
    """Pinned route and hard limits for Leanstral draft generation."""

    llm_provider: str = DEFAULT_LEANSTRAL_LLM_PROVIDER
    model: str = DEFAULT_LEANSTRAL_MODEL
    timeout_seconds: float = DEFAULT_LEANSTRAL_TIMEOUT_SECONDS
    max_new_tokens: int = DEFAULT_LEANSTRAL_MAX_NEW_TOKENS
    max_prompt_bytes: int = DEFAULT_LEANSTRAL_MAX_PROMPT_BYTES
    max_output_bytes: int = DEFAULT_LEANSTRAL_MAX_OUTPUT_BYTES
    temperature: float = 0.0
    vibe_agent: str = "lean"
    network_access_required: bool = False
    resource_isolation: LeanstralResourceIsolation = field(
        default_factory=LeanstralResourceIsolation
    )
    provider: str | None = None

    def __post_init__(self) -> None:
        configured_provider = self.llm_provider
        if self.provider is not None:
            if (
                self.llm_provider != DEFAULT_LEANSTRAL_LLM_PROVIDER
                and self.llm_provider != self.provider
            ):
                raise ValueError(
                    "provider and llm_provider cannot select different routes"
                )
            configured_provider = self.provider
        if not isinstance(configured_provider, str) or not configured_provider.strip():
            raise ValueError(
                "llm_provider must identify a concrete llm_router provider"
            )
        provider = configured_provider.strip()
        if provider.casefold() in _ROUTER_ALIASES:
            raise ValueError(
                "llm_provider must identify a concrete llm_router provider"
            )
        model = _nonempty_text(self.model, field_name="model")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name in (
            "max_new_tokens",
            "max_prompt_bytes",
            "max_output_bytes",
        ):
            _positive_integer(getattr(self, name), field_name=name)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative")
        if not isinstance(self.network_access_required, bool):
            raise ValueError("network_access_required must be a boolean")
        if not isinstance(self.resource_isolation, LeanstralResourceIsolation):
            raise ValueError(
                "resource_isolation must be LeanstralResourceIsolation"
            )
        agent = _nonempty_text(self.vibe_agent, field_name="vibe_agent")
        object.__setattr__(self, "llm_provider", provider)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "vibe_agent", agent)

    @property
    def model_resource_class(self) -> str:
        return self.resource_isolation.model_resource_class

    @property
    def kernel_resource_class(self) -> str:
        return self.resource_isolation.kernel_resource_class

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_provider": self.llm_provider,
            "provider": self.provider,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_output_bytes": self.max_output_bytes,
            "temperature": self.temperature,
            "vibe_agent": self.vibe_agent,
            "network_access_required": self.network_access_required,
            "resource_classes": self.resource_isolation.to_dict(),
        }


# A shorter public spelling for callers that already name the provider class.
LeanstralProviderConfig = LeanstralProofProviderConfig


@dataclass(frozen=True)
class LeanstralProofDraft:
    """Immutable model output with no proof or mutation authority."""

    artifact_id: str
    draft_text: str
    request_id: str
    llm_provider: str
    model: str
    obligation_ids: tuple[str, ...] = ()
    canonical_source_digest: str = ""
    prompt_sha256: str = ""
    output_sha256: str = ""
    timeout_ms: int = 0
    token_budget: int = 0
    resource_class: str = LEANSTRAL_MODEL_RESOURCE_CLASS
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LEANSTRAL_DRAFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "draft_text",
            "request_id",
            "llm_provider",
            "model",
            "prompt_sha256",
            "output_sha256",
            "resource_class",
        ):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), field_name=name)
            )
        if self.schema_version != LEANSTRAL_DRAFT_SCHEMA_VERSION:
            raise ValueError("unsupported Leanstral draft schema")
        if self.resource_class == LEAN_KERNEL_RESOURCE_CLASS:
            raise ValueError("model drafts cannot use the kernel resource class")
        object.__setattr__(
            self,
            "obligation_ids",
            _identifiers(self.obligation_ids, field_name="obligation_ids"),
        )
        object.__setattr__(
            self,
            "canonical_source_digest",
            str(self.canonical_source_digest or "").strip(),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _positive_integer(self.timeout_ms, field_name="timeout_ms"),
        )
        object.__setattr__(
            self,
            "token_budget",
            _positive_integer(self.token_budget, field_name="token_budget"),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    @property
    def verified(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize only draft data and immutable request bindings."""

        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_kind": "llm_output",
            "stage": ProofStage.MODEL_DRAFT.value,
            "draft_text": self.draft_text,
            "proof_text": self.draft_text,
            "request_id": self.request_id,
            "llm_provider": self.llm_provider,
            "model": self.model,
            "obligation_ids": list(self.obligation_ids),
            "canonical_source_digest": self.canonical_source_digest,
            "prompt_sha256": self.prompt_sha256,
            "output_sha256": self.output_sha256,
            "timeout_ms": self.timeout_ms,
            "token_budget": self.token_budget,
            "resource_class": self.resource_class,
            "assurance": AssuranceLevel.UNVERIFIED.value,
            "verified": False,
            "authoritative": False,
            "proof_attempted": False,
            "proof_success": False,
            "kernel_checked": False,
            "can_mutate_canonical_source": False,
            "can_mutate_obligations": False,
            "metadata": dict(self.metadata),
        }


def _default_llm_generate(prompt: str, **kwargs: Any) -> str:
    """Resolve the router only for an admitted model inference request."""

    try:
        router = importlib.import_module("ipfs_accelerate_py.llm_router")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProofProviderError(
            ProviderFailureCode.UNAVAILABLE,
            "llm_router is unavailable; Leanstral inference is degraded",
            details={"dependency": "llm_router"},
        ) from exc
    generate = getattr(router, "generate_text", None)
    if not callable(generate):
        raise ProofProviderError(
            ProviderFailureCode.UNAVAILABLE,
            "llm_router.generate_text is unavailable; Leanstral inference is degraded",
            details={"dependency": "llm_router.generate_text"},
        )
    return generate(prompt, **kwargs)


class LeanstralProofProvider:
    """Proof-provider implementation that can only produce model drafts."""

    provider_id = LEANSTRAL_PROOF_PROVIDER_ID
    provider_version = LEANSTRAL_PROOF_PROVIDER_VERSION
    protocol_version = PROOF_PROVIDER_PROTOCOL_VERSION

    def __init__(
        self,
        config: LeanstralProofProviderConfig | None = None,
        *,
        llm_generate: LLMGenerate | None = None,
    ) -> None:
        self.config = config or LeanstralProofProviderConfig()
        if llm_generate is not None and not callable(llm_generate):
            raise ValueError("llm_generate must be callable")
        self._llm_generate = llm_generate

    @property
    def model_resource_class(self) -> str:
        return self.config.model_resource_class

    @property
    def kernel_resource_class(self) -> str:
        return self.config.kernel_resource_class

    def capabilities(self) -> ProofProviderCapability:
        """Describe the narrow draft-only capability without loading a model."""

        return ProofProviderCapability(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            protocol_versions=(self.protocol_version,),
            operations=(
                ProofProviderOperation.CAPABILITY,
                ProofProviderOperation.PROVE,
            ),
            isolation=(
                ProofProviderIsolation.IN_PROCESS,
                ProofProviderIsolation.SUBPROCESS,
            ),
            network_access_required=self.config.network_access_required,
            resource_limits_supported=True,
            metadata={
                "assurance": AssuranceLevel.UNVERIFIED.value,
                "authoritative": False,
                "draft_only": True,
                "kernel_check_supported": False,
                "can_mutate_canonical_source": False,
                "can_mutate_obligations": False,
                "llm_provider": self.config.llm_provider,
                "model": self.config.model,
                "timeout_ms": int(self.config.timeout_seconds * 1000),
                "token_budget": self.config.max_new_tokens,
                "resource_classes": self.config.resource_isolation.to_dict(),
            },
        )

    def capability(self, request: ProviderRequest) -> Mapping[str, Any]:
        if request.operation is not ProofProviderOperation.CAPABILITY:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "Leanstral capability received the wrong operation",
            )
        return self.capabilities().to_dict()

    def _effective_timeout(self, request: ProviderRequest) -> float:
        timeout = self.config.timeout_seconds
        if request.resource_budget.wall_time_ms:
            timeout = min(timeout, request.resource_budget.wall_time_ms / 1000.0)
        if request.deadline_unix_ms is not None:
            timeout = min(
                timeout,
                max(0.0, request.deadline_unix_ms / 1000.0 - time.time()),
            )
        if timeout <= 0:
            raise ProofProviderError(
                ProviderFailureCode.TIMED_OUT,
                "Leanstral request has no positive model time budget before its deadline",
            )
        return timeout

    def _effective_token_budget(self, request: ProviderRequest) -> int:
        tokens = self.config.max_new_tokens
        if request.resource_budget.model_token_limit:
            tokens = min(tokens, request.resource_budget.model_token_limit)
        if tokens <= 0:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral request has no positive model token budget",
            )
        return tokens

    def _validate_payload(
        self, request: ProviderRequest
    ) -> tuple[str, tuple[str, ...], str, dict[str, Any]]:
        payload = request.payload
        try:
            prompt = _nonempty_text(payload.get("prompt"), field_name="prompt")
            obligation_ids = _identifiers(
                payload.get("obligation_ids", payload.get("obligation_id")),
                field_name="obligation_ids",
            )
        except ValueError as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST, str(exc)
            ) from exc
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.config.max_prompt_bytes:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral prompt exceeded the configured byte limit",
                details={"limit_bytes": self.config.max_prompt_bytes},
            )
        requested_class = str(
            payload.get("resource_class", self.model_resource_class)
        ).strip()
        if requested_class != self.model_resource_class:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "Leanstral inference must run in the model resource class",
                details={
                    "required_resource_class": self.model_resource_class,
                    "requested_resource_class": requested_class,
                },
            )
        if bool(payload.get("kernel_check", False)):
            raise ProofProviderError(
                ProviderFailureCode.UNSUPPORTED,
                "Leanstral model inference cannot perform a kernel check",
                details={"kernel_resource_class": self.kernel_resource_class},
            )
        source_digest = str(
            payload.get("canonical_source_digest", payload.get("source_digest", ""))
            or ""
        ).strip()
        raw_metadata = payload.get("metadata", {})
        try:
            metadata = _json_mapping(raw_metadata, field_name="metadata")
        except ValueError as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST, str(exc)
            ) from exc
        # Canonical objects are bindings owned by the supervisor.  They are not
        # forwarded as writable structures or copied into model-produced fields.
        forbidden = sorted(_CANONICAL_MUTATION_KEYS.intersection(metadata))
        if forbidden:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "Leanstral draft metadata cannot contain canonical source or obligations",
                details={"forbidden_keys": forbidden},
            )
        return prompt, obligation_ids, source_digest, metadata

    def prove(self, request: ProviderRequest) -> Mapping[str, Any]:
        """Generate one unverified draft through the explicitly pinned route."""

        if request.operation is not ProofProviderOperation.PROVE:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "Leanstral prove received the wrong operation",
            )
        if self.config.network_access_required and not request.network_allowed:
            raise ProofProviderError(
                ProviderFailureCode.NETWORK_DENIED,
                "the configured Leanstral model service requires network access",
            )
        prompt, obligation_ids, source_digest, metadata = self._validate_payload(
            request
        )
        timeout = self._effective_timeout(request)
        token_budget = self._effective_token_budget(request)
        generate = self._llm_generate or _default_llm_generate
        try:
            output = generate(
                prompt,
                provider=self.config.llm_provider,
                model_name=self.config.model,
                timeout=timeout,
                max_new_tokens=token_budget,
                allow_local_fallback=False,
                disable_model_retry=True,
                temperature=self.config.temperature,
                mistral_vibe_agent=self.config.vibe_agent,
            )
        except ProofProviderError:
            raise
        except TimeoutError as exc:
            raise ProofProviderError(
                ProviderFailureCode.TIMED_OUT,
                "Leanstral model inference exceeded its timeout",
                retryable=True,
            ) from exc
        except (ImportError, ModuleNotFoundError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.UNAVAILABLE,
                "a Leanstral model dependency is unavailable",
                details={"dependency": getattr(exc, "name", "") or "unknown"},
            ) from exc
        except Exception as exc:
            raise ProofProviderError(
                ProviderFailureCode.PROVIDER_ERROR,
                f"Leanstral model inference failed ({type(exc).__name__})",
                retryable=True,
            ) from exc
        if not isinstance(output, str):
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "llm_router returned a non-text Leanstral response",
            )
        draft_text = output.strip()
        if not draft_text:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "llm_router returned an empty Leanstral response",
            )
        output_bytes = draft_text.encode("utf-8")
        output_limit = self.config.max_output_bytes
        if request.resource_budget.max_output_bytes:
            output_limit = min(
                output_limit, request.resource_budget.max_output_bytes
            )
        if len(output_bytes) > output_limit:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral response exceeded the configured byte limit",
                details={"limit_bytes": output_limit},
            )

        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_sha256 = hashlib.sha256(output_bytes).hexdigest()
        identity = {
            "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
            "llm_provider": self.config.llm_provider,
            "model": self.config.model,
            "obligation_ids": list(obligation_ids),
            "canonical_source_digest": source_digest,
            "prompt_sha256": prompt_sha256,
            "output_sha256": output_sha256,
        }
        artifact_id = "leanstral-draft-" + hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        draft = LeanstralProofDraft(
            artifact_id=artifact_id,
            draft_text=draft_text,
            request_id=request.request_id,
            llm_provider=self.config.llm_provider,
            model=self.config.model,
            obligation_ids=obligation_ids,
            canonical_source_digest=source_digest,
            prompt_sha256=prompt_sha256,
            output_sha256=output_sha256,
            timeout_ms=max(1, int(timeout * 1000)),
            token_budget=token_budget,
            resource_class=self.model_resource_class,
            metadata=metadata,
        )
        return draft.to_dict()


def create_leanstral_proof_provider(
    config: LeanstralProofProviderConfig | None = None,
    *,
    llm_generate: LLMGenerate | None = None,
) -> LeanstralProofProvider:
    """Construct the provider without resolving any optional dependency."""

    return LeanstralProofProvider(config, llm_generate=llm_generate)


__all__ = [
    "DEFAULT_LEANSTRAL_LLM_PROVIDER",
    "DEFAULT_LEANSTRAL_MAX_NEW_TOKENS",
    "DEFAULT_LEANSTRAL_MAX_OUTPUT_BYTES",
    "DEFAULT_LEANSTRAL_MAX_PROMPT_BYTES",
    "DEFAULT_LEANSTRAL_MODEL",
    "DEFAULT_LEANSTRAL_TIMEOUT_SECONDS",
    "LEANSTRAL_DRAFT_SCHEMA_VERSION",
    "LEANSTRAL_MODEL_RESOURCE_CLASS",
    "LEANSTRAL_PROOF_PROVIDER_ID",
    "LEANSTRAL_PROOF_PROVIDER_VERSION",
    "LEAN_KERNEL_RESOURCE_CLASS",
    "LLMGenerate",
    "LeanstralProofDraft",
    "LeanstralProofProvider",
    "LeanstralProofProviderConfig",
    "LeanstralProviderConfig",
    "LeanstralResourceIsolation",
    "create_leanstral_proof_provider",
]
