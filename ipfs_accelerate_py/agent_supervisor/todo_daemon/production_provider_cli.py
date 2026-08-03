"""Fail-closed CLI adapters for the production Grok/Codex packet route.

The contract-packet router deliberately accepts callables instead of knowing
about any concrete model transport.  This module supplies the production
transport boundary for local supervisors:

* Grok Build CLI is the implementation provider;
* Codex CLI is a distinct, independent review provider;
* both receive only the router-authored bounded request and start from a fresh
  empty working directory instead of the repository checkout (this is working
  directory separation, not operating-system filesystem confinement); and
* the returned proposal includes supervisor-observed child/effective-provider
  provenance.  Model output never authors its own execution receipt.

The policy is an operator/runtime overlay.  It is intentionally separate from
task metadata so enabling it does not rewrite a reviewed board, change a
canonical task CID, or invalidate an immutable task-source binding.

This completion-capable route intentionally has no Codex *implementation*
fallback: Codex cannot implement and independently review the same proposal.
A legacy/best-effort Grok-to-Codex implementation fallback must remain
provider-review-pending unless a third independent reviewer is introduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .contract_packet_provider_router import (
    MAX_PROVIDER_PROMPT_TOKENS,
    MAX_PROVIDER_RESPONSE_BYTES,
    MAX_PROVIDER_TIMEOUT_SECONDS,
    ProviderRequest,
    ProviderRole,
)
from .legacy_landed_provider_cli import (
    _invoke_native_structured_cli,
)
from .llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
    LlmRouterInvocation,
)

PRODUCTION_CLI_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-cli-provider-policy@1"
)
PRODUCTION_CLI_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-cli-provider-execution@1"
)
PRODUCTION_NATIVE_STRUCTURED_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-native-structured-execution@1"
)
PRODUCTION_LANDED_TASK_GUARD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-landed-task-guard@1"
)
PRODUCTION_CLI_POLICY_NAME: Final = (
    "grok-implement-codex-independent-review"
)
DEFAULT_GROK_MODEL: Final = "grok-4.5"
DEFAULT_CODEX_MODEL: Final = "gpt-5.6-sol"
DEFAULT_CONTEXT_BUDGET_TOKENS: Final = MAX_PROVIDER_PROMPT_TOKENS
DEFAULT_PROVIDER_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_MAX_NEW_TOKENS: Final = 4_096


ProviderInvoker = Callable[
    [str, LlmRouterInvocation],
    tuple[str, LlmChildResultEnvelope | None],
]


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _policy_id(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _request_id(request: ProviderRequest, policy_id: str) -> str:
    material = {
        "policy_id": policy_id,
        "role": request.role.value,
        "packet_id": request.packet_id,
        "snapshot_id": request.snapshot_id,
        "task_id": request.task_id,
        "prompt_digest": hashlib.sha256(request.prompt).hexdigest(),
    }
    return "provider-request:" + hashlib.sha256(
        _canonical_json_bytes(material)
    ).hexdigest()


def _json_object_without_duplicates(value: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    parsed = json.loads(
        value,
        object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {item}")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError("provider response JSON must contain an object")
    return parsed


def _request_write_paths(request: ProviderRequest) -> tuple[str, ...]:
    """Return the exact operator-authored write scope for a Grok request."""

    packet = request.payload.get("contract_packet")
    scope = packet.get("scope") if isinstance(packet, Mapping) else None
    raw_paths = scope.get("write_paths") if isinstance(scope, Mapping) else None
    if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
        raise RuntimeError("production Grok request lacks an exact write scope")
    paths = tuple(str(item) for item in raw_paths)
    if (
        any(not item or item != item.strip() for item in paths)
        or len(paths) != len(set(paths))
    ):
        raise RuntimeError("production Grok write scope is not canonical")
    return paths


def _production_response_json_schema(
    request: ProviderRequest,
) -> dict[str, Any]:
    """Build one strict role- and request-bound native output schema."""

    binding_properties: dict[str, Any] = {
        "packet_id": {"type": "string", "enum": [request.packet_id]},
        "snapshot_id": {"type": "string", "enum": [request.snapshot_id]},
        "task_id": {"type": "string", "enum": [request.task_id]},
    }
    binding_required = ["packet_id", "snapshot_id", "task_id"]
    if request.role is ProviderRole.GROK_IMPLEMENT:
        paths = _request_write_paths(request)
        proposal_properties: dict[str, Any] = {
            "declared_paths": {
                "type": "array",
                "items": {"type": "string", "enum": list(paths)},
                "minItems": 1,
                "maxItems": len(paths),
                "uniqueItems": True,
            },
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string", "enum": list(paths)},
                        "content": {
                            "type": "string",
                            "maxLength": request.bounds.max_response_bytes,
                        },
                    },
                    "required": ["path", "content"],
                },
                "maxItems": len(paths),
            },
            "patch": {
                "type": "string",
                "maxLength": request.bounds.max_response_bytes,
            },
        }
        properties = {
            **binding_properties,
            "proposal": {
                "type": "object",
                "additionalProperties": False,
                "properties": proposal_properties,
                "required": ["declared_paths", "files", "patch"],
                "oneOf": [
                    {
                        "properties": {
                            "files": {"minItems": 1},
                            "patch": {"maxLength": 0},
                        }
                    },
                    {
                        "properties": {
                            "files": {"maxItems": 0},
                            "patch": {"minLength": 1},
                        }
                    },
                ],
            },
        }
        required = [*binding_required, "proposal"]
    elif request.role is ProviderRole.CODEX_REVIEW:
        properties = {
            **binding_properties,
            "decision": {"type": "string", "enum": ["approve", "reject"]},
            "findings": {
                "type": "array",
                "items": {"type": "string", "maxLength": 4_096},
                "maxItems": 0,
            },
        }
        required = [*binding_required, "decision", "findings"]
    else:
        raise RuntimeError("production native role is not policy-admissible")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _validate_production_native_response(
    response_text: str,
    response_schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Independently enforce the strict schema subset used by this route."""

    response = _json_object_without_duplicates(response_text)
    properties = response_schema.get("properties")
    required = response_schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise RuntimeError("production native response schema is invalid")
    if set(response) != set(required):
        raise RuntimeError("production native response violates its strict schema")
    for binding in ("packet_id", "snapshot_id", "task_id"):
        definition = properties.get(binding)
        expected = definition.get("enum") if isinstance(definition, Mapping) else None
        if expected != [response.get(binding)]:
            raise RuntimeError(
                "production native response violates its request binding"
            )

    if "proposal" in properties:
        proposal = response.get("proposal")
        proposal_schema = properties.get("proposal")
        proposal_properties = (
            proposal_schema.get("properties")
            if isinstance(proposal_schema, Mapping)
            else None
        )
        if not isinstance(proposal, Mapping) or not isinstance(
            proposal_properties, Mapping
        ):
            raise RuntimeError("production Grok response violates its strict schema")
        if set(proposal) != {"declared_paths", "files", "patch"}:
            raise RuntimeError("production Grok response violates its strict schema")
        declared = proposal.get("declared_paths")
        files = proposal.get("files")
        patch = proposal.get("patch")
        declared_schema = proposal_properties.get("declared_paths")
        item_schema = (
            declared_schema.get("items")
            if isinstance(declared_schema, Mapping)
            else None
        )
        allowed = item_schema.get("enum") if isinstance(item_schema, Mapping) else None
        if (
            not isinstance(declared, list)
            or not declared
            or len(declared) != len(set(declared))
            or not isinstance(allowed, list)
            or any(not isinstance(item, str) or item not in allowed for item in declared)
            or not isinstance(files, list)
            or not isinstance(patch, str)
            or bool(files) == bool(patch)
        ):
            raise RuntimeError("production Grok response violates its strict schema")
        file_paths: list[str] = []
        for item in files:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "content"}
                or not isinstance(item.get("path"), str)
                or item.get("path") not in allowed
                or not isinstance(item.get("content"), str)
            ):
                raise RuntimeError(
                    "production Grok response violates its strict schema"
                )
            file_paths.append(str(item["path"]))
        if files and (
            len(file_paths) != len(set(file_paths))
            or set(file_paths) != set(declared)
        ):
            raise RuntimeError("production Grok response violates its strict schema")
    else:
        decision = response.get("decision")
        findings = response.get("findings")
        if (
            decision not in {"approve", "reject"}
            or findings != []
        ):
            raise RuntimeError("production Codex response violates its strict schema")
    return response


@dataclass(frozen=True, slots=True)
class ProductionCLIProviderPolicy:
    """Operator-selected immutable Grok-implementation/Codex-review policy."""

    name: str = PRODUCTION_CLI_POLICY_NAME
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS
    provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    grok_provider: str = "grok_cli"
    grok_model: str = DEFAULT_GROK_MODEL
    codex_provider: str = "codex_cli"
    codex_model: str = DEFAULT_CODEX_MODEL

    def __post_init__(self) -> None:
        if self.name != PRODUCTION_CLI_POLICY_NAME:
            raise ValueError(
                "production CLI policy must be Grok implementation with "
                "independent Codex review"
            )
        if (
            isinstance(self.context_budget_tokens, bool)
            or not isinstance(self.context_budget_tokens, int)
            or not 1 <= self.context_budget_tokens <= MAX_PROVIDER_PROMPT_TOKENS
        ):
            raise ValueError(
                "context_budget_tokens must be in "
                f"[1, {MAX_PROVIDER_PROMPT_TOKENS}]"
            )
        timeout = self.provider_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1.0 <= float(timeout) <= MAX_PROVIDER_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "provider_timeout_seconds must be finite and in "
                f"[1, {MAX_PROVIDER_TIMEOUT_SECONDS:g}]"
            )
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or not 1 <= self.max_new_tokens <= MAX_PROVIDER_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_new_tokens must be a positive bounded integer"
            )
        for name, value in (
            ("grok_provider", self.grok_provider),
            ("grok_model", self.grok_model),
            ("codex_provider", self.codex_provider),
            ("codex_model", self.codex_model),
        ):
            if not str(value or "").strip():
                raise ValueError(f"{name} is required")
        if self.grok_provider != "grok_cli" or self.codex_provider != "codex_cli":
            raise ValueError(
                "production CLI policy requires exact Grok and Codex providers"
            )
        if self.grok_provider.strip().casefold() == self.codex_provider.strip().casefold():
            raise ValueError("implementation and review providers must be distinct")

    @property
    def declared_roles(self) -> tuple[str, str]:
        """Task-contract spelling consumed by implementation policy checks."""

        return ("grok-implement", "codex-review")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": PRODUCTION_CLI_POLICY_SCHEMA,
            "name": self.name,
            "declared_roles": list(self.declared_roles),
            "context_budget_tokens": int(self.context_budget_tokens),
            "provider_timeout_seconds": float(self.provider_timeout_seconds),
            "max_new_tokens": int(self.max_new_tokens),
            "implementation": {
                "role": ProviderRole.GROK_IMPLEMENT.value,
                "provider": self.grok_provider,
                "model": self.grok_model,
                "fallback_provider": "",
                "failure_disposition": "provider_review_pending",
            },
            "review": {
                "role": ProviderRole.CODEX_REVIEW.value,
                "provider": self.codex_provider,
                "model": self.codex_model,
                "independent": True,
            },
            "codex_implementation_fallback": {
                "enabled": False,
                "reason": "codex_cannot_implement_and_independently_self_review",
                "without_third_reviewer": "provider_review_pending",
            },
            "landed_task_recovery": {
                "blind_reimplementation_allowed": False,
                "review_only_requires_supervisor_observed_grok_provenance": True,
                "missing_typed_receipt": "provider_review_pending_no_reimplementation",
                "legacy_fallback_counts_as_review": False,
            },
            "task_metadata_mutated": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        return {**body, "policy_id": _policy_id(body)}

    @property
    def policy_id(self) -> str:
        return str(self.to_dict()["policy_id"])


@dataclass(frozen=True, slots=True)
class BoundProductionCLIProvider:
    """One role-bound provider callable for ``ImplementationProviderRouter``."""

    policy: ProductionCLIProviderPolicy
    role: ProviderRole
    provider_name: str
    model_name: str
    invoker: ProviderInvoker | None = None

    def __post_init__(self) -> None:
        expected = {
            ProviderRole.GROK_IMPLEMENT: (
                self.policy.grok_provider,
                self.policy.grok_model,
            ),
            ProviderRole.CODEX_REVIEW: (
                self.policy.codex_provider,
                self.policy.codex_model,
            ),
        }.get(self.role)
        if expected is None:
            raise ValueError("production CLI adapter supports only Grok/Codex roles")
        if (self.provider_name, self.model_name) != expected:
            raise ValueError(
                "provider name/model do not match their policy-bound role"
            )

    def _invoke(
        self,
        prompt: str,
        config: LlmRouterInvocation,
        response_schema: Mapping[str, Any],
        *,
        max_response_bytes: int,
    ) -> tuple[str, LlmChildResultEnvelope | None]:
        if self.invoker is not None:
            return self.invoker(prompt, config)
        return _invoke_native_structured_cli(
            prompt,
            config,
            response_schema,
            response_validator=_validate_production_native_response,
            execution_schema=PRODUCTION_NATIVE_STRUCTURED_EXECUTION_SCHEMA,
            max_response_bytes=max_response_bytes,
        )

    def __call__(self, request: ProviderRequest) -> Mapping[str, Any]:
        if not isinstance(request, ProviderRequest):
            raise TypeError("production CLI provider requires ProviderRequest")
        if request.role is not self.role:
            raise RuntimeError(
                f"provider role mismatch: expected {self.role.value}, "
                f"received {request.role.value}"
            )
        try:
            prompt = request.prompt.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("provider request prompt is not UTF-8") from exc

        request_id = _request_id(request, self.policy.policy_id)
        # Existing test/operator invokers keep their callable API and output
        # contract. Only the built-in production transport requires the native
        # CLI schema (and therefore the exact operator write scope).
        response_schema = (
            _production_response_json_schema(request)
            if self.invoker is None
            else {}
        )
        # The child provider is deliberately started from a fresh empty
        # directory, and the bounded prompt contains no checkout path.  This
        # avoids making the repository its working directory; it is not an OS
        # sandbox and does not block access to independently discovered
        # absolute filesystem paths.
        with tempfile.TemporaryDirectory(
            prefix=f"ipfs-accelerate-{self.role.value}-"
        ) as temporary_cwd:
            temporary_cwd_path = Path(temporary_cwd).resolve()
            invocation = LlmRouterInvocation(
                repo_root=temporary_cwd_path,
                model_name=self.model_name,
                provider=self.provider_name,
                allow_local_fallback=False,
                allow_cross_provider_fallback=False,
                timeout_seconds=max(
                    1,
                    min(
                        int(self.policy.provider_timeout_seconds),
                        max(1, int(request.bounds.timeout_seconds) - 1),
                    ),
                ),
                max_new_tokens=self.policy.max_new_tokens,
                max_prompt_chars=max(1, len(prompt)),
                temperature=0.0,
                python_executable=sys.executable,
                timeout_grace_seconds=0,
                trace=True,
                reject_effective_provider_name=None,
                required_effective_providers=(self.provider_name,),
                usage_mode=LLM_USAGE_MODE_ENFORCE,
                request_id=request_id,
                attempt=1,
                idempotency_key=request_id,
                side_effect_boundary="proposal_only",
                write_result_envelope=True,
            )
            output, child_receipt = self._invoke(
                prompt,
                invocation,
                response_schema,
                max_response_bytes=request.bounds.max_response_bytes,
            )

        if len(output.encode("utf-8")) > request.bounds.max_response_bytes:
            raise RuntimeError("provider response exceeds its exact byte bound")
        response = _json_object_without_duplicates(output)
        if child_receipt is None:
            raise RuntimeError("provider child did not return an execution receipt")
        receipt = child_receipt.to_dict()
        exit_code = receipt.get("exit_code")
        native_structured = self.invoker is None
        if (
            receipt.get("status") != "ok"
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
            or receipt.get("request_id") != request_id
            or receipt.get("idempotency_key") != request_id
            or receipt.get("effective_provider") != self.provider_name
            or receipt.get("attempt") != 1
            or receipt.get("usage_mode") != LLM_USAGE_MODE_ENFORCE
            or (
                native_structured
                and (
                    not str(receipt.get("supervisor_receipt_id") or "")
                    or not str(receipt.get("execution_result_id") or "")
                )
            )
        ):
            raise RuntimeError("provider child execution receipt is not bound")
        if self.role is ProviderRole.CODEX_REVIEW:
            decision = response.get("decision")
            findings = response.get("findings")
            if (
                decision not in {"approve", "reject"}
                or findings != []
                or response.get("proposal") not in (None, {})
            ):
                raise RuntimeError(
                    "production Codex review must be an approve/reject decision "
                    "with an empty findings list"
                )

        response["supervisor_provider_execution"] = {
            "schema": PRODUCTION_CLI_EXECUTION_SCHEMA,
            "policy_id": self.policy.policy_id,
            "request_id": request_id,
            "role": self.role.value,
            "configured_provider": self.provider_name,
            "configured_model": self.model_name,
            "effective_provider": receipt["effective_provider"],
            # Native execution passes this exact pinned model in argv. The
            # generic child envelope has no effective-model field, so this
            # supervisor-authored receipt records and enforces that boundary.
            "effective_model": self.model_name,
            "child_result_schema": receipt["schema"],
            "child_result_status": receipt["status"],
            "child_exit_code": exit_code,
            "supervisor_receipt_id": str(
                receipt.get("supervisor_receipt_id") or ""
            ),
            "endpoint_receipt_id": str(
                receipt.get("endpoint_receipt_id") or ""
            ),
            "execution_result_id": str(
                receipt.get("execution_result_id") or ""
            ),
            "native_output_schema_id": (
                _policy_id(response_schema) if native_structured else ""
            ),
            "native_structured_output_enforced": native_structured,
            "cross_provider_fallback_allowed": False,
            "model_output_authored_receipt": False,
            "repository_checkout_used_as_working_directory": False,
            "operating_system_filesystem_confinement": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        return response


def build_production_cli_provider_pair(
    policy: ProductionCLIProviderPolicy | None = None,
    *,
    invoker: ProviderInvoker | None = None,
) -> tuple[BoundProductionCLIProvider, BoundProductionCLIProvider]:
    """Return distinct Grok implementation and Codex review callables."""

    selected = policy or ProductionCLIProviderPolicy()
    return (
        BoundProductionCLIProvider(
            policy=selected,
            role=ProviderRole.GROK_IMPLEMENT,
            provider_name=selected.grok_provider,
            model_name=selected.grok_model,
            invoker=invoker,
        ),
        BoundProductionCLIProvider(
            policy=selected,
            role=ProviderRole.CODEX_REVIEW,
            provider_name=selected.codex_provider,
            model_name=selected.codex_model,
            invoker=invoker,
        ),
    )


def production_cli_policy_readiness(
    policy: ProductionCLIProviderPolicy | None = None,
) -> dict[str, Any]:
    """Bounded, read-only local binary/auth readiness for operator preflight."""

    selected = policy or ProductionCLIProviderPolicy()
    grok_binary = shutil.which("grok")
    codex_binary = shutil.which("codex")
    grok_authenticated = False
    try:
        from ...llm_router import _grok_cli_auth_available

        grok_authenticated = bool(_grok_cli_auth_available())
    except Exception:
        grok_authenticated = False
    codex_authenticated = False
    codex_auth_check = "binary_missing"
    if codex_binary:
        codex_auth_check = "codex_login_status_failed"
        try:
            status = subprocess.run(
                [codex_binary, "login", "status"],
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            codex_authenticated = status.returncode == 0
            codex_auth_check = (
                "codex_login_status_ok"
                if codex_authenticated
                else "codex_login_status_failed"
            )
    payload = {
        "policy_id": selected.policy_id,
        "ready": bool(
            grok_binary
            and grok_authenticated
            and codex_binary
            and codex_authenticated
        ),
        "implementation": {
            "provider": selected.grok_provider,
            "binary_available": bool(grok_binary),
            "authenticated": grok_authenticated,
        },
        "review": {
            "provider": selected.codex_provider,
            "binary_available": bool(codex_binary),
            "authenticated": codex_authenticated,
            "authentication_check": codex_auth_check,
            "independent": selected.grok_provider != selected.codex_provider,
        },
    }
    return payload


def production_landed_task_guard(
    *,
    recovered_binding: Mapping[str, Any] | None,
    workspace_clean: bool,
    typed_provider_receipt_available: bool = False,
) -> dict[str, Any]:
    """Prevent an already-landed task from being blindly reimplemented.

    An exact recovered implementation binding means code for the task already
    landed.  A missing historical typed receipt cannot be repaired by calling
    Grok again or by treating an ad-hoc Codex-only response as independent
    review of the original work.  Such work remains pending until a dedicated
    review-only protocol can bind the original Grok execution provenance.
    """

    recovered = bool(
        isinstance(recovered_binding, Mapping)
        and recovered_binding.get("recovered") is True
    )
    if not recovered:
        return {
            "schema": PRODUCTION_LANDED_TASK_GUARD_SCHEMA,
            "guarded": False,
            "action": "new_implementation_route_allowed",
            "invoke_grok_implementation": True,
            "invoke_codex_review": True,
            "provider_review_pending": False,
            "reason": "no_exact_landed_binding",
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
    has_typed_receipt = bool(typed_provider_receipt_available)
    if not workspace_clean:
        return {
            "schema": PRODUCTION_LANDED_TASK_GUARD_SCHEMA,
            "guarded": True,
            "action": "provider_review_pending_unlanded_workspace_changes",
            "invoke_grok_implementation": False,
            "invoke_codex_review": False,
            "provider_review_pending": True,
            "legacy_fallback_counts_as_review": False,
            "reason": "landed_binding_workspace_not_clean",
            "implementation_commit": str(
                recovered_binding.get("implementation_commit") or ""
            ),
            "merge_commit": str(
                recovered_binding.get("merge_commit")
                or recovered_binding.get("prior_merge_commit")
                or ""
            ),
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
    return {
        "schema": PRODUCTION_LANDED_TASK_GUARD_SCHEMA,
        "guarded": True,
        "action": (
            "verify_existing_typed_receipt"
            if has_typed_receipt
            else "provider_review_pending_no_reimplementation"
        ),
        "invoke_grok_implementation": False,
        # A dedicated future review-only route may set this true only after it
        # binds original Grok execution provenance, commit, tree, and packet.
        "invoke_codex_review": False,
        "provider_review_pending": not has_typed_receipt,
        "legacy_fallback_counts_as_review": False,
        "reason": (
            "existing_typed_receipt_must_be_verified"
            if has_typed_receipt
            else "landed_binding_has_no_typed_provider_receipt"
        ),
        "implementation_commit": str(
            recovered_binding.get("implementation_commit") or ""
        ),
        "merge_commit": str(
            recovered_binding.get("merge_commit")
            or recovered_binding.get("prior_merge_commit")
            or ""
        ),
        "completion_authoritative": False,
        "proof_authoritative": False,
    }


__all__ = [
    "BoundProductionCLIProvider",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
    "DEFAULT_GROK_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROVIDER_TIMEOUT_SECONDS",
    "PRODUCTION_CLI_EXECUTION_SCHEMA",
    "PRODUCTION_CLI_POLICY_NAME",
    "PRODUCTION_CLI_POLICY_SCHEMA",
    "PRODUCTION_LANDED_TASK_GUARD_SCHEMA",
    "PRODUCTION_NATIVE_STRUCTURED_EXECUTION_SCHEMA",
    "ProductionCLIProviderPolicy",
    "build_production_cli_provider_pair",
    "production_cli_policy_readiness",
    "production_landed_task_guard",
]
