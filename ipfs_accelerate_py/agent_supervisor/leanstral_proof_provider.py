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
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
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
from .kernel_verification import (
    DEFAULT_MAX_LEAN_PROOF_BYTES,
    IndependentKernelVerifier,
    KernelVerificationBindings,
    KernelVerificationResult,
    LeanProofAdmission,
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
LEANSTRAL_PROOF_PROVIDER_VERSION: Final = "1.2.0"
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
DEFAULT_LEANSTRAL_MAX_PATCH_BYTES: Final = 2 * 1024 * 1024
DEFAULT_LEANSTRAL_MAX_PATCH_FILES: Final = 128
DEFAULT_LEANSTRAL_PATCH_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_LEANSTRAL_VALIDATION_OUTPUT_BYTES: Final = 256 * 1024
LEANSTRAL_PROOF_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-proof-gate@1"
)
LEANSTRAL_PATCH_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-patch-gate@1"
)

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeanstralProofDraft":
        """Restore a persisted model artifact without accepting trust claims."""

        if not isinstance(payload, Mapping):
            raise ValueError("Leanstral draft payload must be an object")
        if payload.get("schema_version") != LEANSTRAL_DRAFT_SCHEMA_VERSION:
            raise ValueError("unsupported Leanstral draft schema")
        for name, expected in (
            ("artifact_kind", "llm_output"),
            ("stage", ProofStage.MODEL_DRAFT.value),
        ):
            if payload.get(name, expected) != expected:
                raise ValueError(f"Leanstral draft {name} is invalid")
        for name in (
            "verified",
            "authoritative",
            "kernel_checked",
            "proof_success",
        ):
            if payload.get(name, False) is not False:
                raise ValueError(f"model artifact cannot claim {name}")
        raw_decomposition = payload.get("decomposition") or ()
        if not isinstance(raw_decomposition, Sequence) or isinstance(
            raw_decomposition, (str, bytes, bytearray)
        ):
            raise ValueError("decomposition must be an array")
        return cls(
            schema_version=str(payload["schema_version"]),
            artifact_id=payload.get("artifact_id", ""),
            draft_text=payload.get("draft_text", payload.get("proof_text", "")),
            request_id=payload.get("request_id", ""),
            llm_provider=payload.get("llm_provider", ""),
            model=payload.get("model", ""),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            canonical_source_digest=payload.get("canonical_source_digest", ""),
            prompt_sha256=payload.get("prompt_sha256", ""),
            output_sha256=payload.get("output_sha256", ""),
            timeout_ms=payload.get("timeout_ms", 0),
            token_budget=payload.get("token_budget", 0),
            resource_class=payload.get(
                "resource_class", LEANSTRAL_MODEL_RESOURCE_CLASS
            ),
            theorem_id=payload.get("theorem_id", ""),
            theorem_equivalence_key=payload.get(
                "theorem_equivalence_key", ""
            ),
            context_capsule_id=payload.get("context_capsule_id", ""),
            proposal_kind=payload.get("proposal_kind", "proof"),
            decomposition=tuple(raw_decomposition),
            reused_artifact_ids=tuple(
                payload.get("reused_artifact_ids") or ()
            ),
            prompt_tokens=payload.get("prompt_tokens", 0),
            response_tokens=payload.get("response_tokens", 0),
            metadata=payload.get("metadata") or {},
        )


class LeanstralGateStatus(str, Enum):
    """Stable proof/patch proposal gate outcomes."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True)
class LeanstralProofGateResult:
    """Paired model and kernel artifacts with intentionally separate trust."""

    status: LeanstralGateStatus
    reason_codes: tuple[str, ...]
    model_artifact: LeanstralProofDraft
    admission: LeanProofAdmission
    kernel_verification: KernelVerificationResult
    schema: str = LEANSTRAL_PROOF_GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEANSTRAL_PROOF_GATE_SCHEMA:
            raise ValueError("unsupported Leanstral proof gate schema")
        status = (
            self.status
            if isinstance(self.status, LeanstralGateStatus)
            else LeanstralGateStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        if not isinstance(self.model_artifact, LeanstralProofDraft):
            raise ValueError("model_artifact must be LeanstralProofDraft")
        if not isinstance(self.admission, LeanProofAdmission):
            raise ValueError("admission must be LeanProofAdmission")
        if not isinstance(self.kernel_verification, KernelVerificationResult):
            raise ValueError(
                "kernel_verification must be KernelVerificationResult"
            )
        codes = _identifiers(self.reason_codes, field_name="reason_codes")
        object.__setattr__(self, "reason_codes", codes)
        if (status is LeanstralGateStatus.ACCEPTED) != (
            self.admission.accepted and self.kernel_verification.accepted
        ):
            raise ValueError("proof gate status disagrees with kernel evidence")

    @property
    def accepted(self) -> bool:
        return self.status is LeanstralGateStatus.ACCEPTED

    @property
    def assurance(self) -> AssuranceLevel:
        return self.kernel_verification.assurance

    @property
    def authoritative(self) -> bool:
        return self.kernel_verification.accepted

    @property
    def artifact_id(self) -> str:
        payload = self.to_dict()
        return "leanstral-proof-gate-" + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "model_artifact": self.model_artifact.to_dict(),
            "admission_artifact": self.admission.to_dict(),
            "kernel_artifact": self.kernel_verification.to_dict(),
            "provenance": {
                "model": {
                    "artifact_id": self.model_artifact.artifact_id,
                    "provider": self.model_artifact.llm_provider,
                    "model": self.model_artifact.model,
                    "assurance": AssuranceLevel.UNVERIFIED.value,
                    "authoritative": False,
                    "resource_class": self.model_artifact.resource_class,
                },
                "kernel": {
                    "artifact_id": self.kernel_verification.verification_id,
                    "kernel_id": self.kernel_verification.kernel_id,
                    "toolchain_id": self.kernel_verification.toolchain_id,
                    "assurance": self.kernel_verification.assurance.value,
                    "authoritative": self.kernel_verification.accepted,
                    "resource_class": LEAN_KERNEL_RESOURCE_CLASS,
                },
                "derived_from": {
                    "kernel_candidate": self.model_artifact.artifact_id
                },
            },
            "authoritative_assurance": self.assurance.value,
            "authoritative": self.authoritative,
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


def _coerce_leanstral_draft(
    value: LeanstralProofDraft | Mapping[str, Any],
) -> LeanstralProofDraft:
    if isinstance(value, LeanstralProofDraft):
        return value
    return LeanstralProofDraft.from_dict(value)


def _validate_draft_integrity(
    draft: LeanstralProofDraft,
    theorem: FixedTheoremIdentity,
) -> None:
    if draft.proposal_kind != "proof":
        raise ValueError("only Leanstral proof proposals can enter the kernel gate")
    if not draft.theorem_id or draft.theorem_id != theorem.theorem_id:
        raise ValueError("model artifact changed the fixed theorem identity")
    if draft.theorem_equivalence_key != theorem.equivalence_key:
        raise ValueError("model artifact does not match the fixed theorem")
    if draft.obligation_ids != (theorem.obligation_id,):
        raise ValueError("model artifact obligation does not match the fixed theorem")
    if (
        theorem.canonical_source_digest
        and draft.canonical_source_digest
        != theorem.canonical_source_digest
    ):
        raise ValueError("model artifact canonical source binding changed")
    actual_output = hashlib.sha256(draft.draft_text.encode("utf-8")).hexdigest()
    if draft.output_sha256 != actual_output:
        raise ValueError("model artifact output digest does not match proof text")
    identity = {
        "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
        "llm_provider": draft.llm_provider,
        "model": draft.model,
        "obligation_ids": list(draft.obligation_ids),
        "canonical_source_digest": draft.canonical_source_digest,
        "theorem_id": draft.theorem_id,
        "theorem_equivalence_key": draft.theorem_equivalence_key,
        "context_capsule_id": draft.context_capsule_id,
        "proposal_kind": draft.proposal_kind,
        "prompt_sha256": draft.prompt_sha256,
        "output_sha256": draft.output_sha256,
    }
    expected_artifact_id = "leanstral-draft-" + hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if draft.artifact_id != expected_artifact_id:
        raise ValueError("model artifact identity is corrupt")


def verify_leanstral_draft(
    draft: LeanstralProofDraft | Mapping[str, Any],
    theorem: FixedTheoremIdentity | Mapping[str, Any],
    *,
    native_source: str,
    bindings: KernelVerificationBindings,
    verifier: IndependentKernelVerifier | None = None,
    proof_placeholder: str = "sorry",
    canonical_source: str = "",
    max_proof_bytes: int = DEFAULT_MAX_LEAN_PROOF_BYTES,
    reconstruction_record: Any = None,
    reconstruction_evidence: Any = None,
    environment_lock: Any = None,
    timeout_seconds: float = 30.0,
    kernel_runner: Any = None,
) -> LeanstralProofGateResult:
    """Kernel-gate a persisted Leanstral artifact without trusting its flags.

    The resulting model artifact stays unverified.  Only the separately nested
    reconstruction artifact may carry ``KERNEL_VERIFIED`` assurance.
    """

    model_artifact = _coerce_leanstral_draft(draft)
    fixed = (
        theorem
        if isinstance(theorem, FixedTheoremIdentity)
        else FixedTheoremIdentity.from_dict(theorem)
    )
    _validate_draft_integrity(model_artifact, fixed)
    if bindings.obligation_id != fixed.obligation_id:
        raise ValueError("kernel binding obligation does not match fixed theorem")
    if bindings.request_id != model_artifact.request_id:
        raise ValueError("kernel binding request does not match model artifact")
    if fixed.declaration_name:
        declarations = re.findall(
            r"\b(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)",
            native_source,
        )
        if fixed.declaration_name not in declarations:
            raise ValueError(
                "canonical Lean source does not contain the fixed declaration"
            )
    digest_match = re.fullmatch(
        r"(?:sha256:)?([0-9a-fA-F]{64})",
        fixed.canonical_source_digest,
    )
    if digest_match:
        bound_source = canonical_source or native_source
        actual_digest = hashlib.sha256(bound_source.encode("utf-8")).hexdigest()
        if actual_digest.casefold() != digest_match.group(1).casefold():
            raise ValueError("canonical Lean source digest does not match theorem")
    checker = verifier or IndependentKernelVerifier()
    admission, verification = checker.verify_lean_proof_text(
        model_artifact.draft_text,
        native_source,
        bindings=bindings,
        theorem_id=fixed.theorem_id,
        declaration_name=fixed.declaration_name,
        model_artifact_id=model_artifact.artifact_id,
        proof_placeholder=proof_placeholder,
        canonical_source=canonical_source,
        max_proof_bytes=max_proof_bytes,
        reconstruction_record=reconstruction_record,
        reconstruction_evidence=reconstruction_evidence,
        environment_lock=environment_lock,
        timeout_seconds=timeout_seconds,
        kernel_runner=kernel_runner,
        provider_status="leanstral_model_draft",
    )
    accepted = admission.accepted and verification.accepted
    reason_codes = (
        ("independent_kernel_acceptance",)
        if accepted
        else tuple(
            dict.fromkeys(
                (
                    admission.failure_code.value,
                    verification.failure_code.value,
                )
            )
        )
    )
    reason_codes = tuple(item for item in reason_codes if item)
    return LeanstralProofGateResult(
        status=(
            LeanstralGateStatus.ACCEPTED
            if accepted
            else LeanstralGateStatus.REJECTED
        ),
        reason_codes=reason_codes,
        model_artifact=model_artifact,
        admission=admission,
        kernel_verification=verification,
    )


@dataclass(frozen=True)
class LeanstralPatchGatePolicy:
    """Task-owned patch scope and deterministic validation requirements."""

    allowed_paths: tuple[str, ...]
    validation_commands: tuple[str | tuple[str, ...], ...] = ()
    timeout_seconds: float = DEFAULT_LEANSTRAL_PATCH_TIMEOUT_SECONDS
    max_patch_bytes: int = DEFAULT_LEANSTRAL_MAX_PATCH_BYTES
    max_patch_files: int = DEFAULT_LEANSTRAL_MAX_PATCH_FILES
    max_validation_output_bytes: int = DEFAULT_LEANSTRAL_VALIDATION_OUTPUT_BYTES

    def __post_init__(self) -> None:
        paths: list[str] = []
        for raw in self.allowed_paths:
            path = _safe_patch_path(raw, allow_directory=True)
            if path not in paths:
                paths.append(path)
        if not paths:
            raise ValueError("allowed_paths must contain task-declared paths")
        object.__setattr__(self, "allowed_paths", tuple(paths))
        commands: list[str | tuple[str, ...]] = []
        for raw in self.validation_commands:
            if isinstance(raw, str):
                command = raw.strip()
                if not command:
                    raise ValueError("validation command must not be empty")
                commands.append(command)
            elif isinstance(raw, Sequence) and not isinstance(
                raw, (bytes, bytearray)
            ):
                command_tuple = tuple(str(item).strip() for item in raw)
                if not command_tuple or any(not item for item in command_tuple):
                    raise ValueError(
                        "validation argv must contain non-empty strings"
                    )
                commands.append(command_tuple)
            else:
                raise ValueError("validation command must be text or argv")
        object.__setattr__(self, "validation_commands", tuple(commands))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        for name in (
            "max_patch_bytes",
            "max_patch_files",
            "max_validation_output_bytes",
        ):
            _positive_integer(getattr(self, name), field_name=name)


@dataclass(frozen=True)
class LeanstralPatchGateResult:
    """Non-authoritative eligibility result for an untrusted model patch."""

    status: LeanstralGateStatus
    reason_codes: tuple[str, ...]
    model_artifact_id: str
    patch_sha256: str
    touched_paths: tuple[str, ...]
    apply_check: Mapping[str, Any]
    validation_results: tuple[Mapping[str, Any], ...] = ()
    schema: str = LEANSTRAL_PATCH_GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEANSTRAL_PATCH_GATE_SCHEMA:
            raise ValueError("unsupported Leanstral patch gate schema")
        status = (
            self.status
            if isinstance(self.status, LeanstralGateStatus)
            else LeanstralGateStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, field_name="reason_codes"),
        )
        object.__setattr__(
            self,
            "model_artifact_id",
            _nonempty_text(
                self.model_artifact_id, field_name="model_artifact_id"
            ),
        )
        object.__setattr__(
            self,
            "patch_sha256",
            _nonempty_text(self.patch_sha256, field_name="patch_sha256"),
        )
        object.__setattr__(
            self,
            "touched_paths",
            _identifiers(self.touched_paths, field_name="touched_paths"),
        )
        object.__setattr__(
            self,
            "apply_check",
            _json_mapping(self.apply_check, field_name="apply_check"),
        )
        object.__setattr__(
            self,
            "validation_results",
            tuple(
                _json_mapping(item, field_name="validation_result")
                for item in self.validation_results
            ),
        )

    @property
    def accepted(self) -> bool:
        return self.status is LeanstralGateStatus.ACCEPTED

    @property
    def assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    @property
    def authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "model_artifact": {
                "artifact_id": self.model_artifact_id,
                "artifact_kind": "llm_patch",
                "patch_sha256": self.patch_sha256,
                "assurance": AssuranceLevel.UNVERIFIED.value,
                "authoritative": False,
            },
            "touched_paths": list(self.touched_paths),
            "apply_check": dict(self.apply_check),
            "validation_artifacts": [
                dict(item) for item in self.validation_results
            ],
            "assurance": AssuranceLevel.UNVERIFIED.value,
            "authoritative": False,
            "can_apply": False,
        }


def _safe_patch_path(value: Any, *, allow_directory: bool = False) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("patch path must be text")
    raw = os.fspath(value).strip().replace("\\", "/")
    directory = allow_directory and raw.endswith("/")
    if raw.startswith(("a/", "b/")) and not allow_directory:
        raw = raw[2:]
    if (
        not raw
        or raw.startswith("/")
        or "\x00" in raw
        or any(ord(char) < 32 for char in raw)
    ):
        raise ValueError(f"unsafe patch path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe patch path: {raw!r}")
    normalized = path.as_posix()
    return normalized + "/" if directory else normalized


def _header_patch_path(value: str) -> str:
    value = value.strip()
    if value == "/dev/null":
        return ""
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ValueError("malformed quoted patch path") from exc
    if not parts:
        raise ValueError("empty patch path header")
    return _safe_patch_path(parts[0])


def _paths_from_patch(patch_text: str, *, max_files: int) -> tuple[str, ...]:
    touched: list[str] = []
    diff_headers = 0
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            diff_headers += 1
            try:
                parts = shlex.split(line)
            except ValueError as exc:
                raise ValueError("malformed diff --git header") from exc
            if len(parts) != 4 or parts[:2] != ["diff", "--git"]:
                raise ValueError("malformed diff --git header")
            for raw in parts[2:]:
                path = _safe_patch_path(raw)
                if path and path not in touched:
                    touched.append(path)
        elif line.startswith(("--- ", "+++ ")):
            path = _header_patch_path(line[4:])
            if path and path not in touched:
                touched.append(path)
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            path = _header_patch_path(line.split(" ", 2)[2])
            if path and path not in touched:
                touched.append(path)
        elif re.match(
            r"^(?:(?:new |deleted )?file mode|(?:new|old) mode) "
            r"(?:120000|160000)\b",
            line,
        ):
            raise ValueError("symlink and gitlink patches are forbidden")
        elif line.startswith(("GIT binary patch", "Binary files ")):
            raise ValueError("binary patches are forbidden")
    if not diff_headers or not touched:
        raise ValueError("proposal must be a Git-style unified diff")
    if len(touched) > max_files:
        raise ValueError("patch touches more files than policy allows")
    return tuple(touched)


def _path_in_task_scope(path: str, allowed_paths: Sequence[str]) -> bool:
    return any(
        path == allowed.rstrip("/")
        if not allowed.endswith("/")
        else path.startswith(allowed)
        for allowed in allowed_paths
    )


def _task_declared_scope(
    repo_root: Path,
    paths: Sequence[str | os.PathLike[str]],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in paths:
        raw = os.fspath(value)
        candidate = Path(raw)
        directory = raw.replace("\\", "/").endswith("/") or candidate.is_dir()
        if candidate.is_absolute():
            try:
                raw = candidate.resolve().relative_to(repo_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "task-declared path is outside repo_root"
                ) from exc
            if directory:
                raw += "/"
        item = _safe_patch_path(raw, allow_directory=True)
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("task_declared_paths must not be empty")
    return tuple(normalized)


def _command_payload(
    result: Any,
    *,
    command: Sequence[str],
    max_output_bytes: int,
) -> dict[str, Any]:
    if isinstance(result, Mapping):
        returncode = result.get("returncode")
        stdout = result.get("stdout", result.get("output", ""))
        stderr = result.get("stderr", "")
        timed_out = bool(result.get("timed_out", False))
    else:
        returncode = getattr(result, "returncode", None)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        timed_out = bool(getattr(result, "timed_out", False))
    try:
        code = int(returncode)
    except (TypeError, ValueError):
        code = 1
    combined = (str(stdout or "") + str(stderr or "")).encode(
        "utf-8", errors="replace"
    )
    truncated = len(combined) > max_output_bytes
    clipped = combined[:max_output_bytes].decode("utf-8", errors="replace")
    return {
        "command": [str(item) for item in command],
        "returncode": code,
        "passed": code == 0 and not timed_out,
        "timed_out": timed_out,
        "output": clipped,
        "output_truncated": truncated,
    }


def _execute_patch_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    input_text: str | None,
    runner: Callable[..., Any] | None,
    max_output_bytes: int,
) -> dict[str, Any]:
    if runner is not None:
        try:
            raw = runner(
                tuple(command),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                input_text=input_text,
            )
        except TypeError:
            raw = runner(
                tuple(command),
                cwd=cwd,
                timeout=timeout_seconds,
                stdin=input_text,
            )
        return _command_payload(
            raw, command=command, max_output_bytes=max_output_bytes
        )
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "NO_COLOR": "1"},
        )
        raw_result: Mapping[str, Any] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        raw_result = {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }
    return _command_payload(
        raw_result, command=command, max_output_bytes=max_output_bytes
    )


def check_leanstral_patch_proposal(
    patch_text: str,
    *,
    model_artifact_id: str,
    repo_root: str | os.PathLike[str],
    task_declared_paths: Sequence[str | os.PathLike[str]],
    validation_commands: Sequence[str | Sequence[str]] = (),
    policy: LeanstralPatchGatePolicy | None = None,
    command_runner: Callable[..., Any] | None = None,
) -> LeanstralPatchGateResult:
    """Fail-closed scope, apply, and validation gate for a model patch.

    With the default runner validation occurs in a temporary detached Git
    worktree after the patch is applied there.  The caller's worktree and
    index remain untouched.  A custom runner is treated as the supervisor's
    isolation boundary and receives the checked commands without applying to
    the caller's worktree.
    """

    artifact_id = _nonempty_text(
        model_artifact_id, field_name="model_artifact_id"
    )
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise ValueError("patch_text must be non-empty text")
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    declared_scope = _task_declared_scope(root, task_declared_paths)
    supplied_validations = tuple(
        item if isinstance(item, str) else tuple(item)
        for item in validation_commands
    )
    if policy is None:
        effective_policy = LeanstralPatchGatePolicy(
            allowed_paths=declared_scope,
            validation_commands=supplied_validations,
        )
    elif isinstance(policy, LeanstralPatchGatePolicy):
        combined_validations = tuple(
            dict.fromkeys((*policy.validation_commands, *supplied_validations))
        )
        effective_policy = replace(
            policy, validation_commands=combined_validations
        )
    else:
        raise ValueError("policy must be LeanstralPatchGatePolicy")
    if not isinstance(effective_policy, LeanstralPatchGatePolicy):
        raise ValueError("policy must be LeanstralPatchGatePolicy")
    patch_bytes = patch_text.encode("utf-8")
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()

    def rejected(
        reason: str,
        *,
        touched: tuple[str, ...] = (),
        apply_check: Mapping[str, Any] | None = None,
        validations: tuple[Mapping[str, Any], ...] = (),
    ) -> LeanstralPatchGateResult:
        return LeanstralPatchGateResult(
            status=LeanstralGateStatus.REJECTED,
            reason_codes=(reason,),
            model_artifact_id=artifact_id,
            patch_sha256=patch_sha256,
            touched_paths=touched,
            apply_check=apply_check or {
                "command": ["git", "apply", "--check"],
                "returncode": 1,
                "passed": False,
                "timed_out": False,
                "output": "",
                "output_truncated": False,
            },
            validation_results=validations,
        )

    if len(patch_bytes) > effective_policy.max_patch_bytes:
        return rejected("patch_too_large")
    try:
        touched = _paths_from_patch(
            patch_text, max_files=effective_policy.max_patch_files
        )
    except ValueError:
        return rejected("malformed_or_unsafe_patch")
    if any(
        not _path_in_task_scope(path, declared_scope)
        or not _path_in_task_scope(path, effective_policy.allowed_paths)
        for path in touched
    ):
        return rejected("path_outside_task_scope", touched=touched)

    check = _execute_patch_command(
        ("git", "apply", "--check", "--recount", "-"),
        cwd=root,
        timeout_seconds=effective_policy.timeout_seconds,
        input_text=patch_text,
        runner=command_runner,
        max_output_bytes=effective_policy.max_validation_output_bytes,
    )
    if not check["passed"]:
        return rejected("git_apply_check_failed", touched=touched, apply_check=check)

    validations: list[Mapping[str, Any]] = []
    validation_root = root
    temporary: tempfile.TemporaryDirectory[str] | None = None
    worktree_added = False
    try:
        if command_runner is None and effective_policy.validation_commands:
            temporary = tempfile.TemporaryDirectory(
                prefix="leanstral-patch-gate-", ignore_cleanup_errors=True
            )
            validation_root = Path(temporary.name) / "candidate"
            add = _execute_patch_command(
                ("git", "worktree", "add", "--detach", str(validation_root), "HEAD"),
                cwd=root,
                timeout_seconds=effective_policy.timeout_seconds,
                input_text=None,
                runner=None,
                max_output_bytes=effective_policy.max_validation_output_bytes,
            )
            if not add["passed"]:
                return rejected(
                    "validation_workspace_failed",
                    touched=touched,
                    apply_check=check,
                )
            worktree_added = True
            applied = _execute_patch_command(
                ("git", "apply", "--recount", "-"),
                cwd=validation_root,
                timeout_seconds=effective_policy.timeout_seconds,
                input_text=patch_text,
                runner=None,
                max_output_bytes=effective_policy.max_validation_output_bytes,
            )
            if not applied["passed"]:
                return rejected(
                    "isolated_patch_apply_failed",
                    touched=touched,
                    apply_check=check,
                )
        for configured in effective_policy.validation_commands:
            command = (
                ("/bin/bash", "-lc", configured)
                if isinstance(configured, str)
                else configured
            )
            result = _execute_patch_command(
                command,
                cwd=validation_root,
                timeout_seconds=effective_policy.timeout_seconds,
                input_text=None,
                runner=command_runner,
                max_output_bytes=effective_policy.max_validation_output_bytes,
            )
            validations.append(result)
            if not result["passed"]:
                return rejected(
                    "configured_validation_failed",
                    touched=touched,
                    apply_check=check,
                    validations=tuple(validations),
                )
    finally:
        if worktree_added:
            try:
                subprocess.run(
                    ("git", "worktree", "remove", "--force", str(validation_root)),
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=min(30.0, effective_policy.timeout_seconds),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                # TemporaryDirectory still removes only the isolated candidate.
                # A later ``git worktree prune`` may clear a stale registration.
                pass
        if temporary is not None:
            temporary.cleanup()
    return LeanstralPatchGateResult(
        status=LeanstralGateStatus.ACCEPTED,
        reason_codes=("scope_apply_and_validation_passed",),
        model_artifact_id=artifact_id,
        patch_sha256=patch_sha256,
        touched_paths=touched,
        apply_check=check,
        validation_results=tuple(validations),
    )


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

    def verify_draft(
        self,
        draft: LeanstralProofDraft | Mapping[str, Any],
        theorem: FixedTheoremIdentity | Mapping[str, Any],
        **kwargs: Any,
    ) -> LeanstralProofGateResult:
        """Delegate a model artifact to the supervisor-owned kernel gate."""

        return verify_leanstral_draft(draft, theorem, **kwargs)

    def check_patch_proposal(
        self, patch_text: str, **kwargs: Any
    ) -> LeanstralPatchGateResult:
        """Check an untrusted patch without giving the provider apply authority."""

        return check_leanstral_patch_proposal(patch_text, **kwargs)

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


kernel_check_leanstral_draft = verify_leanstral_draft
validate_leanstral_patch_proposal = check_leanstral_patch_proposal
gate_leanstral_patch_proposal = check_leanstral_patch_proposal


__all__ = [
    "DEFAULT_LEANSTRAL_LLM_PROVIDER",
    "DEFAULT_LEANSTRAL_MAX_PATCH_BYTES",
    "DEFAULT_LEANSTRAL_MAX_PATCH_FILES",
    "DEFAULT_LEANSTRAL_MAX_NEW_TOKENS",
    "DEFAULT_LEANSTRAL_MAX_OUTPUT_BYTES",
    "DEFAULT_LEANSTRAL_MAX_PROMPT_BYTES",
    "DEFAULT_LEANSTRAL_MAX_PROMPT_TOKENS",
    "DEFAULT_LEANSTRAL_MODEL",
    "DEFAULT_LEANSTRAL_TIMEOUT_SECONDS",
    "DEFAULT_LEANSTRAL_PATCH_TIMEOUT_SECONDS",
    "DEFAULT_LEANSTRAL_VALIDATION_OUTPUT_BYTES",
    "LEANSTRAL_DRAFT_SCHEMA_VERSION",
    "LEANSTRAL_MODEL_RESOURCE_CLASS",
    "LEANSTRAL_PROOF_PROVIDER_ID",
    "LEANSTRAL_PROOF_PROVIDER_VERSION",
    "LEANSTRAL_PROOF_GATE_SCHEMA",
    "LEANSTRAL_PATCH_GATE_SCHEMA",
    "LEAN_KERNEL_RESOURCE_CLASS",
    "LLMGenerate",
    "LeanstralProofDraft",
    "LeanstralGateStatus",
    "LeanstralPatchGatePolicy",
    "LeanstralPatchGateResult",
    "LeanstralProofGateResult",
    "LeanstralProofProvider",
    "LeanstralProofProviderConfig",
    "LeanstralProviderConfig",
    "LeanstralResourceIsolation",
    "LEANSTRAL_PROOF_OUTPUT_SCHEMA",
    "create_leanstral_proof_provider",
    "check_leanstral_patch_proposal",
    "gate_leanstral_patch_proposal",
    "kernel_check_leanstral_draft",
    "validate_leanstral_patch_proposal",
    "verify_leanstral_draft",
]
