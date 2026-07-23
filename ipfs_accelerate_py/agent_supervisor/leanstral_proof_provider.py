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
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .proof_context import (
    DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFTS,
    LEANSTRAL_PROOF_OUTPUT_SCHEMA,
    FixedTheoremIdentity,
    LeanstralPromptLimits,
    LeanstralProofContext,
    ProofContextBudgetError,
    ProofContextCapsule,
    ProofContextError,
    build_leanstral_proof_context,
    estimate_context_tokens,
)

LEANSTRAL_PROOF_PROVIDER_ID: Final = "leanstral"
LEANSTRAL_PROOF_PROVIDER_VERSION: Final = "1.1.0"
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
DEFAULT_LEANSTRAL_MAX_PROMPT_TOKENS: Final = 128 * 1024
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
    max_prompt_tokens: int = DEFAULT_LEANSTRAL_MAX_PROMPT_TOKENS
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
            "max_prompt_tokens",
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
            raise ValueError("resource_isolation must be LeanstralResourceIsolation")
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
            "max_prompt_tokens": self.max_prompt_tokens,
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
    theorem_id: str = ""
    theorem_equivalence_key: str = ""
    context_capsule_id: str = ""
    proposal_kind: str = "proof"
    decomposition: tuple[Mapping[str, Any], ...] = ()
    reused_artifact_ids: tuple[str, ...] = ()
    prompt_tokens: int = 0
    response_tokens: int = 0
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
        for name in (
            "theorem_id",
            "theorem_equivalence_key",
            "context_capsule_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        proposal_kind = str(self.proposal_kind or "proof").strip().casefold()
        if proposal_kind not in {"proof", "decomposition", "raw"}:
            raise ValueError("proposal_kind must be proof, decomposition, or raw")
        decomposition = tuple(
            _json_mapping(item, field_name="decomposition item")
            for item in self.decomposition
        )
        if proposal_kind == "decomposition" and not decomposition:
            raise ValueError("decomposition proposal must contain subgoals")
        object.__setattr__(self, "proposal_kind", proposal_kind)
        object.__setattr__(self, "decomposition", decomposition)
        object.__setattr__(
            self,
            "reused_artifact_ids",
            _identifiers(
                self.reused_artifact_ids,
                field_name="reused_artifact_ids",
            ),
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
        for name in ("prompt_tokens", "response_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
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
            "theorem_id": self.theorem_id,
            "theorem_equivalence_key": self.theorem_equivalence_key,
            "context_capsule_id": self.context_capsule_id,
            "proposal_kind": self.proposal_kind,
            "proposal_schema": (
                LEANSTRAL_PROOF_OUTPUT_SCHEMA if self.theorem_id else ""
            ),
            "decomposition": [dict(item) for item in self.decomposition],
            "reused_artifact_ids": list(self.reused_artifact_ids),
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
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


@dataclass(frozen=True)
class _LeanstralInvocation:
    prompt: str
    obligation_ids: tuple[str, ...]
    source_digest: str
    metadata: Mapping[str, Any]
    context: LeanstralProofContext | None = None


def _strict_response_object(text: str) -> dict[str, Any]:
    def no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate response key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral fixed-theorem response must be strict JSON",
        ) from exc
    if not isinstance(value, dict):
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral fixed-theorem response must be a JSON object",
        )
    return value


def _parse_fixed_theorem_response(
    text: str,
    context: LeanstralProofContext,
) -> tuple[str, str, tuple[Mapping[str, Any], ...]]:
    value = _strict_response_object(text)
    allowed = {
        "schema",
        "theorem_id",
        "proposal_kind",
        "proof_text",
        "decomposition",
    }
    extra = sorted(set(value) - allowed)
    if extra:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral response attempted to add or mutate fixed theorem fields",
            details={"forbidden_or_unknown_keys": extra},
        )
    if value.get("schema") != LEANSTRAL_PROOF_OUTPUT_SCHEMA:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral response used the wrong output schema",
        )
    if value.get("theorem_id") != context.theorem.theorem_id:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral response changed the fixed theorem identity",
        )
    kind = str(value.get("proposal_kind") or "").casefold()
    if kind == "proof":
        if set(value) != {
            "schema",
            "theorem_id",
            "proposal_kind",
            "proof_text",
        }:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "Leanstral proof response must contain only proof_text",
            )
        proof_text = value.get("proof_text")
        if not isinstance(proof_text, str) or not proof_text.strip():
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "Leanstral proof response has empty proof_text",
            )
        return kind, proof_text.strip(), ()
    if kind != "decomposition":
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral proposal_kind must be proof or decomposition",
        )
    if set(value) != {
        "schema",
        "theorem_id",
        "proposal_kind",
        "decomposition",
    }:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral decomposition response must contain only decomposition",
        )
    raw_items = value.get("decomposition")
    if not isinstance(raw_items, list) or not raw_items or len(raw_items) > 64:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_RESPONSE,
            "Leanstral decomposition must contain between 1 and 64 subgoals",
        )
    normalized: list[Mapping[str, Any]] = []
    known_ids: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping) or not set(raw).issubset(
            {"subgoal_id", "statement", "depends_on"}
        ):
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "Leanstral decomposition item does not match the output schema",
            )
        subgoal_id = raw.get("subgoal_id")
        statement = raw.get("statement")
        depends_on = raw.get("depends_on", [])
        if (
            not isinstance(subgoal_id, str)
            or not subgoal_id.strip()
            or subgoal_id in known_ids
            or not isinstance(statement, str)
            or not statement.strip()
            or not isinstance(depends_on, list)
            or any(not isinstance(item, str) or not item.strip() for item in depends_on)
        ):
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "Leanstral decomposition item has invalid fields",
            )
        known_ids.add(subgoal_id)
        normalized.append(
            {
                "subgoal_id": subgoal_id.strip(),
                "statement": statement.strip(),
                "depends_on": list(dict.fromkeys(item.strip() for item in depends_on)),
            }
        )
    for item in normalized:
        unknown = set(item["depends_on"]) - known_ids
        if unknown:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE,
                "Leanstral decomposition references an unknown subgoal",
                details={"unknown_subgoal_ids": sorted(unknown)},
            )
    normalized_text = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return kind, normalized_text, tuple(normalized)


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
        self._draft_lock = threading.Lock()
        self._drafts_by_equivalence: dict[str, list[dict[str, Any]]] = {}

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
                "prompt_byte_budget": self.config.max_prompt_bytes,
                "prompt_token_budget": self.config.max_prompt_tokens,
                "fixed_theorem_prompts": True,
                "response_schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
                "equivalent_draft_reuse": True,
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

    def build_prompt(
        self,
        capsule: ProofContextCapsule,
        theorem: FixedTheoremIdentity | Mapping[str, Any],
        *,
        allowed_premises: Sequence[Mapping[str, Any]] = (),
        trusted_prior_receipts: Sequence[Mapping[str, Any]] = (),
        compact_failures: Sequence[Mapping[str, Any]] = (),
        reusable_drafts: Sequence[Mapping[str, Any]] = (),
        limits: LeanstralPromptLimits | Mapping[str, Any] | None = None,
        max_premises: int = 0,
    ) -> LeanstralProofContext:
        """Build a prompt while automatically offering equivalent old drafts."""

        fixed = (
            theorem
            if isinstance(theorem, FixedTheoremIdentity)
            else FixedTheoremIdentity.from_dict(theorem)
        )
        requested_limits = (
            limits
            if isinstance(limits, LeanstralPromptLimits)
            else (
                LeanstralPromptLimits.from_dict(limits)
                if isinstance(limits, Mapping)
                else LeanstralPromptLimits(
                    max_bytes=capsule.limits.max_bytes,
                    max_tokens=capsule.limits.max_tokens,
                )
            )
        )
        premise_limit = requested_limits.max_premises
        if max_premises:
            premise_limit = min(premise_limit, max_premises)
        effective_limits = replace(
            requested_limits,
            max_bytes=min(
                requested_limits.max_bytes,
                capsule.limits.max_bytes,
                self.config.max_prompt_bytes,
            ),
            max_tokens=min(
                requested_limits.max_tokens,
                capsule.limits.max_tokens,
                self.config.max_prompt_tokens,
            ),
            max_premises=premise_limit,
        )
        with self._draft_lock:
            cached = tuple(self._drafts_by_equivalence.get(fixed.equivalence_key, ()))
        return build_leanstral_proof_context(
            capsule,
            fixed,
            allowed_premises=allowed_premises,
            trusted_prior_receipts=trusted_prior_receipts,
            compact_failures=compact_failures,
            reusable_drafts=(*reusable_drafts, *cached),
            limits=effective_limits,
        )

    def _validate_payload(self, request: ProviderRequest) -> _LeanstralInvocation:
        payload = request.payload
        try:
            raw_capsule = payload.get(
                "context_capsule",
                payload.get("proof_context_capsule", payload.get("proof_context")),
            )
            context: LeanstralProofContext | None = None
            if raw_capsule is not None:
                if not isinstance(raw_capsule, Mapping):
                    raise ProofContextError("context_capsule must be an object")
                capsule = ProofContextCapsule.from_dict(raw_capsule)
                raw_theorem = payload.get(
                    "fixed_theorem",
                    payload.get("theorem_identity", payload.get("theorem")),
                )
                if not isinstance(raw_theorem, Mapping):
                    raise ProofContextError(
                        "fixed_theorem is required with context_capsule"
                    )
                context = self.build_prompt(
                    capsule,
                    raw_theorem,
                    allowed_premises=tuple(payload.get("allowed_premises") or ()),
                    trusted_prior_receipts=tuple(
                        payload.get(
                            "trusted_prior_receipts",
                            payload.get("prior_receipts"),
                        )
                        or ()
                    ),
                    compact_failures=tuple(
                        payload.get("compact_failures", payload.get("failures")) or ()
                    ),
                    reusable_drafts=tuple(
                        payload.get(
                            "reusable_drafts",
                            payload.get("draft_artifacts"),
                        )
                        or ()
                    ),
                    limits=(
                        payload.get("prompt_limits")
                        if isinstance(payload.get("prompt_limits"), Mapping)
                        else None
                    ),
                    max_premises=request.resource_budget.max_premises,
                )
                prompt = context.to_prompt()
                obligation_ids = (context.theorem.obligation_id,)
                source_digest = context.theorem.canonical_source_digest
                supplied_ids = _identifiers(
                    payload.get("obligation_ids", payload.get("obligation_id")),
                    field_name="obligation_ids",
                )
                if supplied_ids and supplied_ids != obligation_ids:
                    raise ProofContextError(
                        "request obligation_ids do not match the fixed theorem"
                    )
                supplied_digest = str(
                    payload.get(
                        "canonical_source_digest",
                        payload.get("source_digest", ""),
                    )
                    or ""
                ).strip()
                if supplied_digest and supplied_digest != source_digest:
                    raise ProofContextError(
                        "request source digest does not match the fixed theorem"
                    )
            else:
                prompt = _nonempty_text(payload.get("prompt"), field_name="prompt")
                obligation_ids = _identifiers(
                    payload.get("obligation_ids", payload.get("obligation_id")),
                    field_name="obligation_ids",
                )
                source_digest = str(
                    payload.get(
                        "canonical_source_digest",
                        payload.get("source_digest", ""),
                    )
                    or ""
                ).strip()
        except ProofContextBudgetError as exc:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED, str(exc)
            ) from exc
        except (ValueError, ProofContextError) as exc:
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
        prompt_tokens = estimate_context_tokens(prompt)
        if prompt_tokens > self.config.max_prompt_tokens:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral prompt exceeded the configured token limit",
                details={"limit_tokens": self.config.max_prompt_tokens},
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
        return _LeanstralInvocation(
            prompt=prompt,
            obligation_ids=obligation_ids,
            source_digest=source_digest,
            metadata=metadata,
            context=context,
        )

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
        invocation = self._validate_payload(request)
        prompt = invocation.prompt
        obligation_ids = invocation.obligation_ids
        source_digest = invocation.source_digest
        metadata = dict(invocation.metadata)
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
            output_limit = min(output_limit, request.resource_budget.max_output_bytes)
        if len(output_bytes) > output_limit:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral response exceeded the configured byte limit",
                details={"limit_bytes": output_limit},
            )
        response_tokens = estimate_context_tokens(draft_text)
        if response_tokens > token_budget:
            raise ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED,
                "Leanstral response exceeded the model token budget",
                details={"limit_tokens": token_budget},
            )

        proposal_kind = "raw"
        decomposition: tuple[Mapping[str, Any], ...] = ()
        theorem_id = ""
        theorem_equivalence_key = ""
        context_capsule_id = ""
        reused_artifact_ids: tuple[str, ...] = ()
        if invocation.context is not None:
            proposal_kind, normalized_text, decomposition = (
                _parse_fixed_theorem_response(draft_text, invocation.context)
            )
            draft_text = normalized_text
            theorem_id = invocation.context.theorem.theorem_id
            theorem_equivalence_key = invocation.context.theorem.equivalence_key
            context_capsule_id = invocation.context.capsule_id
            reused_artifact_ids = tuple(
                item.artifact_id
                for item in invocation.context.reusable_untrusted_drafts
            )
            metadata = {
                **metadata,
                "structured_output": True,
                "fixed_theorem_identity_digest": (
                    invocation.context.theorem.identity_digest
                ),
            }

        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        artifact_output_bytes = draft_text.encode("utf-8")
        output_sha256 = hashlib.sha256(artifact_output_bytes).hexdigest()
        identity = {
            "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
            "llm_provider": self.config.llm_provider,
            "model": self.config.model,
            "obligation_ids": list(obligation_ids),
            "canonical_source_digest": source_digest,
            "theorem_id": theorem_id,
            "theorem_equivalence_key": theorem_equivalence_key,
            "context_capsule_id": context_capsule_id,
            "proposal_kind": proposal_kind,
            "prompt_sha256": prompt_sha256,
            "output_sha256": output_sha256,
        }
        artifact_id = (
            "leanstral-draft-"
            + hashlib.sha256(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
            ).hexdigest()
        )
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
            theorem_id=theorem_id,
            theorem_equivalence_key=theorem_equivalence_key,
            context_capsule_id=context_capsule_id,
            proposal_kind=proposal_kind,
            decomposition=decomposition,
            reused_artifact_ids=reused_artifact_ids,
            prompt_tokens=estimate_context_tokens(prompt),
            response_tokens=response_tokens,
            metadata=metadata,
        )
        result = draft.to_dict()
        if invocation.context is not None:
            reusable = {
                **result,
                # These redundant fields make the trust boundary explicit even
                # when a cache is persisted outside this provider instance.
                "assurance": AssuranceLevel.UNVERIFIED.value,
                "verified": False,
                "kernel_checked": False,
            }
            with self._draft_lock:
                bucket = self._drafts_by_equivalence.setdefault(
                    theorem_equivalence_key, []
                )
                bucket[:] = [
                    item
                    for item in bucket
                    if item.get("artifact_id") != result["artifact_id"]
                ]
                bucket.append(reusable)
                del bucket[:-DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFTS]
        return result


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
    "DEFAULT_LEANSTRAL_MAX_PROMPT_TOKENS",
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
    "LEANSTRAL_PROOF_OUTPUT_SCHEMA",
    "create_leanstral_proof_provider",
]
