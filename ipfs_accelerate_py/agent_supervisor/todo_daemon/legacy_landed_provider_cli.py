"""Exact no-fallback Grok/Codex adapters for legacy landed byte review.

The adapter sends ``LegacyLeafReviewRequest.canonical_prompt`` verbatim from a
fresh empty working directory.  It converts supervisor-observed child receipt
metadata into ``LegacyProviderObservation``; model output cannot author its
own effective-provider identity or execution receipt.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .legacy_landed_review import (
    MAX_LEAF_TOKENS,
    LegacyLandedReviewPolicy,
    LegacyLeafReviewRequest,
    LegacyProviderObservation,
    LegacyProviderPolicy,
)
from .llm import LLM_USAGE_MODE_ENFORCE, LlmChildResultEnvelope, LlmRouterInvocation

LEGACY_LANDED_CLI_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-cli-execution@1"
)
DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS: Final = 300
DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS: Final = 1_024

LegacyCLIInvoker = Callable[
    [str, LlmRouterInvocation],
    tuple[str, LlmChildResultEnvelope | None],
]


def _strict_json_object(value: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise RuntimeError("legacy provider response contains duplicate fields")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {item}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("legacy provider response is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("legacy provider response must contain an object")
    return parsed


@dataclass(frozen=True, slots=True)
class BoundLegacyLandedCLIProvider:
    """One operator-policy-bound effective CLI provider."""

    provider_policy: LegacyProviderPolicy
    timeout_seconds: int = DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS
    max_new_tokens: int = DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS
    invoker: LegacyCLIInvoker | None = None

    def __post_init__(self) -> None:
        if self.provider_policy.role not in {"grok_audit", "codex_audit"}:
            raise ValueError("legacy CLI adapter role is invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
        ):
            raise ValueError("legacy CLI timeout must be in [1, 300]")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or not 1 <= self.max_new_tokens <= 4_096
        ):
            raise ValueError("legacy CLI response token bound is invalid")

    def _invoke(
        self, prompt: str, invocation: LlmRouterInvocation
    ) -> tuple[str, LlmChildResultEnvelope | None]:
        if self.invoker is not None:
            return self.invoker(prompt, invocation)
        from .llm import call_llm_router_with_receipt

        return call_llm_router_with_receipt(prompt, invocation)

    def __call__(
        self, request: LegacyLeafReviewRequest
    ) -> LegacyProviderObservation:
        if not isinstance(request, LegacyLeafReviewRequest):
            raise TypeError("legacy CLI provider requires LegacyLeafReviewRequest")
        expected = self.provider_policy
        if (
            request.role != expected.role
            or request.provider != expected.provider
            or request.model != expected.model
        ):
            raise RuntimeError("legacy CLI request differs from operator policy")
        if request.token_upper_bound > MAX_LEAF_TOKENS:
            raise RuntimeError("legacy CLI full request exceeds 4096 tokens")
        try:
            prompt = request.canonical_prompt.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("legacy canonical request must be ASCII DAG-JSON") from exc
        if prompt.encode("ascii") != request.canonical_prompt:
            raise RuntimeError("legacy canonical request changed before invocation")

        with tempfile.TemporaryDirectory(
            prefix=f"ipfs-accelerate-{expected.role}-"
        ) as temporary_cwd:
            invocation = LlmRouterInvocation(
                repo_root=Path(temporary_cwd).resolve(),
                model_name=expected.model,
                provider=expected.provider,
                allow_local_fallback=False,
                timeout_seconds=self.timeout_seconds,
                max_new_tokens=self.max_new_tokens,
                max_prompt_chars=len(prompt),
                temperature=0.0,
                python_executable=sys.executable,
                timeout_grace_seconds=0,
                trace=True,
                reject_effective_provider_name=None,
                required_effective_providers=(expected.provider,),
                usage_mode=LLM_USAGE_MODE_ENFORCE,
                request_id=request.request_id,
                attempt=1,
                idempotency_key=request.request_id,
                side_effect_boundary="review_only",
                write_result_envelope=True,
            )
            output, child = self._invoke(prompt, invocation)

        if child is None:
            raise RuntimeError("legacy provider child receipt is missing")
        receipt = child.to_dict()
        exit_code = receipt.get("exit_code")
        if (
            receipt.get("status") != "ok"
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
            or receipt.get("request_id") != request.request_id
            or receipt.get("idempotency_key") != request.request_id
            or receipt.get("effective_provider") != expected.provider
        ):
            raise RuntimeError("legacy provider child receipt is not exactly bound")
        response = _strict_json_object(output)
        observation_body = {
            "schema": LEGACY_LANDED_CLI_EXECUTION_SCHEMA,
            "request_id": request.request_id,
            "role": expected.role,
            "configured_provider": expected.provider,
            "configured_model": expected.model,
            "effective_provider": receipt["effective_provider"],
            "child_result_schema": receipt["schema"],
            "child_result_status": receipt["status"],
            "child_exit_code": exit_code,
            "supervisor_receipt_id": str(receipt.get("supervisor_receipt_id") or ""),
            "endpoint_receipt_id": str(receipt.get("endpoint_receipt_id") or ""),
            "execution_result_id": str(receipt.get("execution_result_id") or ""),
            "response_id": content_identity(response),
            "full_request_token_upper_bound": request.token_upper_bound,
            "fallback_used": False,
            "repository_checkout_used_as_working_directory": False,
            "model_output_authored_execution_receipt": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        return LegacyProviderObservation(
            observation_id=content_identity(observation_body),
            requested_provider=expected.provider,
            requested_model=expected.model,
            effective_provider=expected.provider,
            effective_model=expected.model,
            provider_chain=(expected.provider,),
            fallback_used=False,
            supervisor_observed=True,
            response=response,
        )


def build_legacy_landed_cli_provider_pair(
    policy: LegacyLandedReviewPolicy,
    *,
    invoker: LegacyCLIInvoker | None = None,
) -> tuple[BoundLegacyLandedCLIProvider, BoundLegacyLandedCLIProvider]:
    """Build distinct exact Grok and Codex audit callables from operator policy."""

    if not isinstance(policy, LegacyLandedReviewPolicy):
        raise TypeError("parsed legacy landed review policy is required")
    return (
        BoundLegacyLandedCLIProvider(policy.grok, invoker=invoker),
        BoundLegacyLandedCLIProvider(policy.codex, invoker=invoker),
    )


__all__ = [
    "DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS",
    "DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS",
    "LEGACY_LANDED_CLI_EXECUTION_SCHEMA",
    "BoundLegacyLandedCLIProvider",
    "LegacyCLIInvoker",
    "build_legacy_landed_cli_provider_pair",
]
