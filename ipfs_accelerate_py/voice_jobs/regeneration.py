"""Endpoint-safe, resumable execution for deterministic voice TTS jobs.

The datasets package owns regeneration planning.  This module owns the small
runtime boundary that is allowed to call a synthesis provider.  A run is
bound to a deterministic dispatch manifest, has explicit item/request/cost
bounds, persists an atomic receipt after every attempt, and never treats a
dry-run manifest as authority to dispatch.

The runner deliberately accepts an injected provider.  Production callers can
use the normal voice-router provider while tests use an offline fake without
changing retry, resume, artifact, or receipt behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .contracts import VoiceJobResult, VoiceTTSJob
from .executor import (
    ArtifactPolicy,
    ArtifactResolver,
    VoiceJobExecutionError,
    execute_voice_tts_job,
)

REGENERATION_DISPATCH_SCHEMA_VERSION = "abby_voice_regeneration_dispatch_v1"
REGENERATION_RUN_RECEIPT_SCHEMA_VERSION = "abby_voice_regeneration_run_receipt_v1"

_TERMINAL_ITEM_STATUSES = frozenset(
    {"regenerated", "quarantined", "provider_exhausted"}
)
_ATTEMPT_OUTCOMES = frozenset(
    {
        "admitted",
        "provider_exhausted",
        "quarantined",
        "regenerated",
        "retry_scheduled",
    }
)
_SECRET_MARKERS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "token",
)


class VoiceRegenerationError(RuntimeError):
    """A regeneration manifest, authorization, or checkpoint is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VoiceRegenerationError("regeneration value is not canonical JSON") from exc


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VoiceRegenerationError(f"{name} must be a positive integer")
    return value


def _canonical_text(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise VoiceRegenerationError(f"{name} must be a non-empty canonical string")
    return value


def _safe_endpoint_url(value: Any) -> str:
    text = _canonical_text("endpoint_url", value)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise VoiceRegenerationError("endpoint_url must be a valid URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise VoiceRegenerationError("endpoint_url must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise VoiceRegenerationError("endpoint_url must not contain credentials")
    # Endpoint identity intentionally excludes query strings and fragments.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(marker in normalized for marker in _SECRET_MARKERS):
                return True
            if _contains_secret_key(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_secret_key(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class RegenerationRunnerPolicy:
    """Hard bounds that apply to one deterministic dispatch manifest."""

    max_items: int = 12
    max_attempts_per_item: int = 3
    max_provider_requests: int = 36
    cost_microusd_per_request: int = 1
    max_cost_microusd: int = 36
    initial_backoff_seconds: float = 0.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_items",
            "max_attempts_per_item",
            "max_provider_requests",
            "cost_microusd_per_request",
            "max_cost_microusd",
        ):
            _positive_int(name, getattr(self, name))
        if self.max_provider_requests > self.max_items * self.max_attempts_per_item:
            raise VoiceRegenerationError(
                "max_provider_requests must not exceed max_items * max_attempts_per_item"
            )
        required_cost = self.max_provider_requests * self.cost_microusd_per_request
        if required_cost > self.max_cost_microusd:
            raise VoiceRegenerationError(
                "max_cost_microusd cannot cover the declared provider request bound"
            )
        for name in ("initial_backoff_seconds", "max_backoff_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise VoiceRegenerationError(f"{name} must be finite and non-negative")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise VoiceRegenerationError(
                "initial_backoff_seconds must not exceed max_backoff_seconds"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_microusd_per_request": self.cost_microusd_per_request,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "max_attempts_per_item": self.max_attempts_per_item,
            "max_backoff_seconds": self.max_backoff_seconds,
            "max_cost_microusd": self.max_cost_microusd,
            "max_items": self.max_items,
            "max_provider_requests": self.max_provider_requests,
        }


@dataclass(frozen=True, slots=True)
class RegenerationEndpointContract:
    """Redacted identity of a read-only endpoint contract probe."""

    endpoint_url: str
    api_name: str
    function_index: int
    input_count: int
    recommended_mode: str
    config_sha256: str
    read_only: bool = True
    generation_request_count: int = 0
    upload_request_count: int = 0
    schema_version: str = "abby_voice_endpoint_contract_probe_v1"
    contract_id: str = ""

    def __post_init__(self) -> None:
        endpoint = _safe_endpoint_url(self.endpoint_url)
        api_name = _canonical_text("api_name", self.api_name)
        mode = _canonical_text("recommended_mode", self.recommended_mode)
        if (
            isinstance(self.function_index, bool)
            or not isinstance(self.function_index, int)
            or self.function_index < 0
        ):
            raise VoiceRegenerationError("function_index must be a non-negative integer")
        _positive_int("input_count", self.input_count)
        if (
            not isinstance(self.config_sha256, str)
            or len(self.config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.config_sha256)
        ):
            raise VoiceRegenerationError("config_sha256 must be lowercase SHA-256")
        if self.read_only is not True:
            raise VoiceRegenerationError("endpoint contract probe must be read-only")
        if self.generation_request_count != 0 or self.upload_request_count != 0:
            raise VoiceRegenerationError(
                "endpoint contract probe cannot contain generation or upload requests"
            )
        if self.schema_version != "abby_voice_endpoint_contract_probe_v1":
            raise VoiceRegenerationError("unsupported endpoint contract probe schema")
        object.__setattr__(self, "endpoint_url", endpoint)
        object.__setattr__(self, "api_name", api_name)
        object.__setattr__(self, "recommended_mode", mode)
        computed = _stable_id("abby-voice-endpoint-contract", self.identity_dict())
        if self.contract_id and self.contract_id != computed:
            raise VoiceRegenerationError(
                "contract_id does not match the endpoint contract identity"
            )
        object.__setattr__(self, "contract_id", computed)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegenerationEndpointContract":
        if not isinstance(value, Mapping) or _contains_secret_key(value):
            raise VoiceRegenerationError(
                "endpoint contract must be a credential-free mapping"
            )
        return cls(
            endpoint_url=str(value.get("endpoint_url") or value.get("endpointUrl") or ""),
            api_name=str(
                value.get("api_name")
                or value.get("apiName")
                or value.get("singleApiName")
                or ""
            ),
            function_index=value.get(  # type: ignore[arg-type]
                "function_index",
                value.get("functionIndex", value.get("singleFnIndex")),
            ),
            input_count=value.get(  # type: ignore[arg-type]
                "input_count",
                value.get("inputCount", value.get("singleInputCount")),
            ),
            recommended_mode=str(
                value.get("recommended_mode") or value.get("recommendedMode") or ""
            ),
            config_sha256=str(value.get("config_sha256") or value.get("configSha256") or ""),
            read_only=value.get("read_only", value.get("readOnly", False)) is True,
            generation_request_count=value.get(  # type: ignore[arg-type]
                "generation_request_count", value.get("generationRequestCount", 0)
            ),
            upload_request_count=value.get(  # type: ignore[arg-type]
                "upload_request_count", value.get("uploadRequestCount", 0)
            ),
            schema_version=str(
                value.get("schema_version")
                or value.get("schemaVersion")
                or "abby_voice_endpoint_contract_probe_v1"
            ),
            contract_id=str(value.get("contract_id") or value.get("contractId") or ""),
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "api_name": self.api_name,
            "config_sha256": self.config_sha256,
            "endpoint_url": self.endpoint_url,
            "function_index": self.function_index,
            "generation_request_count": self.generation_request_count,
            "input_count": self.input_count,
            "read_only": self.read_only,
            "recommended_mode": self.recommended_mode,
            "schema_version": self.schema_version,
            "upload_request_count": self.upload_request_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "contract_id": self.contract_id}


@dataclass(frozen=True, slots=True)
class RegenerationDispatchManifest:
    """A deterministic, bounded plan which by itself grants no dispatch."""

    endpoint_contract: RegenerationEndpointContract
    jobs: tuple[VoiceTTSJob, ...]
    policy: RegenerationRunnerPolicy
    dispatch_authorized: bool = False
    remote_mutation_authority: bool = False
    schema_version: str = REGENERATION_DISPATCH_SCHEMA_VERSION
    manifest_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_contract, RegenerationEndpointContract):
            raise VoiceRegenerationError(
                "endpoint_contract must be RegenerationEndpointContract"
            )
        if not isinstance(self.policy, RegenerationRunnerPolicy):
            raise VoiceRegenerationError("policy must be RegenerationRunnerPolicy")
        if self.schema_version != REGENERATION_DISPATCH_SCHEMA_VERSION:
            raise VoiceRegenerationError("unsupported regeneration dispatch schema")
        jobs = tuple(sorted(self.jobs, key=lambda job: job.task_id))
        if not jobs:
            raise VoiceRegenerationError("dispatch manifest requires at least one TTS job")
        if any(not isinstance(job, VoiceTTSJob) for job in jobs):
            raise VoiceRegenerationError("dispatch manifest accepts VoiceTTSJob values only")
        if len(jobs) > self.policy.max_items:
            raise VoiceRegenerationError("dispatch item count exceeds max_items")
        task_ids = [job.task_id for job in jobs]
        if len(task_ids) != len(set(task_ids)):
            raise VoiceRegenerationError("dispatch manifest has duplicate task IDs")
        if self.dispatch_authorized:
            raise VoiceRegenerationError(
                "a dry-run dispatch manifest cannot self-authorize live generation"
            )
        if self.remote_mutation_authority:
            raise VoiceRegenerationError(
                "regeneration dispatch never grants remote mutation authority"
            )
        object.__setattr__(self, "jobs", jobs)
        computed = _stable_id("abby-voice-regeneration-dispatch", self.identity_dict())
        if self.manifest_id and self.manifest_id != computed:
            raise VoiceRegenerationError(
                "manifest_id does not match deterministic dispatch content"
            )
        object.__setattr__(self, "manifest_id", computed)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "dispatch_authorized": False,
            "endpoint_contract": self.endpoint_contract.to_dict(),
            "items": [
                {
                    "locale": job.locale,
                    "model_name": job.model_name,
                    "provider": job.provider,
                    "spoken_text": job.spoken_text,
                    "spoken_text_sha256": job.spoken_text_sha256,
                    "task_id": job.task_id,
                    "voice": job.voice,
                    "work_item_id": job.lineage.work_item_id,
                }
                for job in self.jobs
            ],
            "limits": self.policy.to_dict(),
            "remote_mutation_authority": False,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "item_count": len(self.jobs),
            "manifest_id": self.manifest_id,
            "provider_request_count": 0,
            "state": "awaiting_operator_approval",
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


def build_regeneration_dispatch_manifest(
    jobs: Sequence[VoiceTTSJob],
    *,
    endpoint_contract: RegenerationEndpointContract | Mapping[str, Any],
    policy: RegenerationRunnerPolicy | None = None,
) -> RegenerationDispatchManifest:
    """Build a deterministic no-dispatch manifest for selected TTS jobs."""

    contract = (
        endpoint_contract
        if isinstance(endpoint_contract, RegenerationEndpointContract)
        else RegenerationEndpointContract.from_mapping(endpoint_contract)
    )
    return RegenerationDispatchManifest(
        endpoint_contract=contract,
        jobs=tuple(jobs),
        policy=policy or RegenerationRunnerPolicy(),
    )


class VoiceRegenerationRunner:
    """Run a bounded dispatch manifest and atomically checkpoint every attempt."""

    def __init__(
        self,
        *,
        provider: Callable[..., Any],
        contract_probe: Callable[
            [], RegenerationEndpointContract | Mapping[str, Any]
        ],
        checkpoint_path: str | Path,
        artifact_policy: ArtifactPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        if not callable(contract_probe):
            raise TypeError("contract_probe must be callable")
        self._provider = provider
        self._contract_probe = contract_probe
        self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._resolver = ArtifactResolver(artifact_policy)
        self._sleep = sleep

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @staticmethod
    def _new_item(job: VoiceTTSJob) -> dict[str, Any]:
        return {
            "attempt_count": 0,
            "attempts": [],
            "result": None,
            "status": "pending",
            "task_id": job.task_id,
            "work_item_id": job.lineage.work_item_id,
        }

    def _new_receipt(self, manifest: RegenerationDispatchManifest) -> dict[str, Any]:
        items = [self._new_item(job) for job in manifest.jobs]
        identity = {
            "contract_id": manifest.endpoint_contract.contract_id,
            "manifest_id": manifest.manifest_id,
            "schema_version": REGENERATION_RUN_RECEIPT_SCHEMA_VERSION,
        }
        return {
            **identity,
            "cost_microusd_spent": 0,
            "items": items,
            "provider_request_count": 0,
            "receipt_id": _stable_id("abby-voice-regeneration-run", identity),
            "summary": self._summary(items),
        }

    @staticmethod
    def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts = {
            "pending": 0,
            "provider_exhausted": 0,
            "quarantined": 0,
            "regenerated": 0,
        }
        for item in items:
            status = str(item.get("status") or "pending")
            if status not in counts:
                raise VoiceRegenerationError(
                    f"checkpoint contains unsupported item status {status!r}"
                )
            counts[status] += 1
        return counts

    def _write(self, receipt: dict[str, Any]) -> None:
        receipt["summary"] = self._summary(receipt["items"])
        data = json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._checkpoint_path.name}.",
            suffix=".tmp",
            dir=self._checkpoint_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load(self, manifest: RegenerationDispatchManifest) -> dict[str, Any]:
        if not self._checkpoint_path.exists():
            receipt = self._new_receipt(manifest)
            self._write(receipt)
            return receipt
        try:
            receipt = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VoiceRegenerationError("regeneration checkpoint is unreadable") from exc
        if not isinstance(receipt, dict):
            raise VoiceRegenerationError("regeneration checkpoint must be an object")
        allowed_receipt_fields = {
            "contract_id",
            "cost_microusd_spent",
            "items",
            "manifest_id",
            "provider_request_count",
            "receipt_id",
            "schema_version",
            "summary",
        }
        if set(receipt) != allowed_receipt_fields:
            raise VoiceRegenerationError(
                "regeneration checkpoint fields are invalid"
            )
        identity = {
            "contract_id": manifest.endpoint_contract.contract_id,
            "manifest_id": manifest.manifest_id,
            "schema_version": REGENERATION_RUN_RECEIPT_SCHEMA_VERSION,
        }
        if (
            receipt.get("schema_version") != REGENERATION_RUN_RECEIPT_SCHEMA_VERSION
            or receipt.get("manifest_id") != manifest.manifest_id
            or receipt.get("contract_id") != manifest.endpoint_contract.contract_id
            or receipt.get("receipt_id")
            != _stable_id("abby-voice-regeneration-run", identity)
        ):
            raise VoiceRegenerationError(
                "regeneration checkpoint does not match the dispatch manifest"
            )
        items = receipt.get("items")
        if not isinstance(items, list):
            raise VoiceRegenerationError("regeneration checkpoint items are invalid")
        expected = [(job.task_id, job.lineage.work_item_id) for job in manifest.jobs]
        observed = [
            (str(item.get("task_id") or ""), str(item.get("work_item_id") or ""))
            for item in items
            if isinstance(item, Mapping)
        ]
        if observed != expected:
            raise VoiceRegenerationError(
                "regeneration checkpoint item identities do not match the manifest"
            )
        self._validate_checkpoint_items(items, manifest)
        provider_requests = receipt.get("provider_request_count")
        cost_spent = receipt.get("cost_microusd_spent")
        if (
            isinstance(provider_requests, bool)
            or not isinstance(provider_requests, int)
            or provider_requests < 0
            or isinstance(cost_spent, bool)
            or not isinstance(cost_spent, int)
            or cost_spent < 0
        ):
            raise VoiceRegenerationError("regeneration checkpoint counters are invalid")
        attempt_total = sum(int(item["attempt_count"]) for item in items)
        expected_cost = provider_requests * manifest.policy.cost_microusd_per_request
        if (
            provider_requests != attempt_total
            or cost_spent != expected_cost
            or provider_requests > manifest.policy.max_provider_requests
            or cost_spent > manifest.policy.max_cost_microusd
        ):
            raise VoiceRegenerationError(
                "regeneration checkpoint counters do not match admitted attempts"
            )
        if receipt.get("summary") != self._summary(items):
            raise VoiceRegenerationError(
                "regeneration checkpoint summary does not match its items"
            )
        return receipt

    @staticmethod
    def _validate_checkpoint_items(
        items: Sequence[Mapping[str, Any]],
        manifest: RegenerationDispatchManifest,
    ) -> None:
        allowed_item_fields = {
            "attempt_count",
            "attempts",
            "result",
            "status",
            "task_id",
            "work_item_id",
        }
        allowed_attempt_fields = {
            "artifact_sha256",
            "attempt",
            "error_code",
            "outcome",
            "retryable",
        }
        for item, job in zip(items, manifest.jobs, strict=True):
            if set(item) != allowed_item_fields:
                raise VoiceRegenerationError(
                    "regeneration checkpoint item fields are invalid"
                )
            attempt_count = item.get("attempt_count")
            attempts = item.get("attempts")
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 0
                or attempt_count > manifest.policy.max_attempts_per_item
                or not isinstance(attempts, list)
            ):
                raise VoiceRegenerationError(
                    "regeneration checkpoint attempt ledger is invalid"
                )
            provider_attempts = []
            for attempt in attempts:
                if (
                    not isinstance(attempt, Mapping)
                    or not set(attempt).issubset(allowed_attempt_fields)
                    or set(attempt) < {"attempt", "outcome", "retryable"}
                    or not isinstance(attempt.get("retryable"), bool)
                ):
                    raise VoiceRegenerationError(
                        "regeneration checkpoint attempt entry is invalid"
                    )
                number = attempt.get("attempt")
                if (
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or number < 0
                    or attempt.get("outcome") not in _ATTEMPT_OUTCOMES
                ):
                    raise VoiceRegenerationError(
                        "regeneration checkpoint attempt identity is invalid"
                    )
                error_code = attempt.get("error_code")
                artifact_sha256 = attempt.get("artifact_sha256")
                if error_code is not None and (
                    not isinstance(error_code, str)
                    or re.fullmatch(r"[a-z0-9_]{1,64}", error_code) is None
                ):
                    raise VoiceRegenerationError(
                        "regeneration checkpoint error code is invalid"
                    )
                if artifact_sha256 is not None and (
                    not isinstance(artifact_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
                ):
                    raise VoiceRegenerationError(
                        "regeneration checkpoint artifact digest is invalid"
                    )
                outcome = attempt["outcome"]
                if (
                    (outcome == "regenerated")
                    != (artifact_sha256 is not None)
                    or (outcome in {"provider_exhausted", "quarantined", "retry_scheduled"})
                    != (error_code is not None)
                    or (outcome == "admitted" and len(attempt) != 3)
                ):
                    raise VoiceRegenerationError(
                        "regeneration checkpoint attempt fields are inconsistent"
                    )
                if attempt.get("error_code") != "dispatch_budget_exhausted":
                    provider_attempts.append(attempt)
            if [attempt["attempt"] for attempt in provider_attempts] != list(
                range(1, attempt_count + 1)
            ):
                raise VoiceRegenerationError(
                    "regeneration checkpoint attempt sequence is invalid"
                )
            status = str(item.get("status") or "")
            result_payload = item.get("result")
            if status == "regenerated":
                if not isinstance(result_payload, Mapping):
                    raise VoiceRegenerationError(
                        "regenerated checkpoint item requires a canonical result"
                    )
                try:
                    result = VoiceJobResult.from_payload(result_payload)
                except (TypeError, ValueError) as exc:
                    raise VoiceRegenerationError(
                        "regenerated checkpoint result is invalid"
                    ) from exc
                if (
                    result.task_id != job.task_id
                    or result.task_type != job.task_type
                    or result.status != "completed"
                    or result.lineage.to_dict() != job.lineage.to_dict()
                    or len(result.artifacts) != 1
                    or not result.artifacts[0].media_type.startswith("audio/")
                ):
                    raise VoiceRegenerationError(
                        "regenerated checkpoint result does not match its TTS job"
                    )
                final_attempt = provider_attempts[-1] if provider_attempts else {}
                if (
                    final_attempt.get("outcome") != "regenerated"
                    or final_attempt.get("artifact_sha256")
                    != result.artifacts[0].sha256
                ):
                    raise VoiceRegenerationError(
                        "regenerated checkpoint artifact receipt is inconsistent"
                    )
            elif status in {"quarantined", "provider_exhausted", "pending"}:
                if result_payload is not None:
                    raise VoiceRegenerationError(
                        "non-regenerated checkpoint item cannot retain a result"
                    )
                final_outcome = attempts[-1].get("outcome") if attempts else None
                expected_outcomes = {
                    "pending": {None, "admitted", "retry_scheduled"},
                    "quarantined": {"quarantined"},
                    "provider_exhausted": {"provider_exhausted"},
                }
                if final_outcome not in expected_outcomes[status]:
                    raise VoiceRegenerationError(
                        "regeneration checkpoint status does not match its attempt ledger"
                    )
            else:
                raise VoiceRegenerationError(
                    f"checkpoint contains unsupported item status {status!r}"
                )
        VoiceRegenerationRunner._summary(items)

    @staticmethod
    def _error(exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, VoiceJobExecutionError):
            return exc.code, exc.retryable
        # No provider message is serialized: it can contain a credential,
        # request body, local path, or other unsafe detail.
        return "runner_internal_error", False

    @staticmethod
    def _backoff(policy: RegenerationRunnerPolicy, attempt: int) -> float:
        if policy.initial_backoff_seconds == 0:
            return 0.0
        return min(
            policy.max_backoff_seconds,
            policy.initial_backoff_seconds * (2 ** max(0, attempt - 1)),
        )

    def run(
        self,
        manifest: RegenerationDispatchManifest,
        *,
        dispatch_authorized: bool = False,
    ) -> dict[str, Any]:
        """Execute or resume ``manifest``.

        ``dispatch_authorized`` is intentionally separate from the serialized
        manifest.  The manifest is always a dry-run artifact and cannot grant
        authority to itself.  The default fails before invoking the provider.
        """

        if not isinstance(manifest, RegenerationDispatchManifest):
            raise TypeError("manifest must be a RegenerationDispatchManifest")
        if dispatch_authorized is not True:
            raise VoiceRegenerationError(
                "live generation requires explicit dispatch_authorized=True"
            )
        try:
            probed_value = self._contract_probe()
            if (
                isinstance(probed_value, Mapping)
                and "compatible" in probed_value
                and probed_value.get("compatible") is not True
            ):
                raise VoiceRegenerationError(
                    "endpoint contract probe reported incompatible endpoint drift"
                )
            probed_contract = (
                probed_value
                if isinstance(probed_value, RegenerationEndpointContract)
                else RegenerationEndpointContract.from_mapping(probed_value)
            )
        except VoiceRegenerationError:
            raise
        except Exception as exc:
            # Do not serialize or echo endpoint/probe exception details because
            # they may contain credentials or response bodies.
            raise VoiceRegenerationError("endpoint contract probe failed") from exc
        if probed_contract.contract_id != manifest.endpoint_contract.contract_id:
            raise VoiceRegenerationError(
                "endpoint contract changed after dispatch manifest construction"
            )
        receipt = self._load(manifest)
        policy = manifest.policy

        for job, item in zip(manifest.jobs, receipt["items"], strict=True):
            if item["status"] in _TERMINAL_ITEM_STATUSES:
                continue
            if item["attempts"] and item["attempts"][-1]["outcome"] == "admitted":
                # The provider may have accepted the request before the process
                # stopped. Without a provider-side idempotency receipt it is
                # unsafe to issue a duplicate synthesis call.
                item["attempts"][-1] = {
                    "attempt": item["attempt_count"],
                    "error_code": "ambiguous_provider_attempt",
                    "outcome": "provider_exhausted",
                    "retryable": False,
                }
                item["status"] = "provider_exhausted"
                self._write(receipt)
                continue
            while item["attempt_count"] < policy.max_attempts_per_item:
                if (
                    receipt["provider_request_count"] >= policy.max_provider_requests
                    or receipt["cost_microusd_spent"]
                    + policy.cost_microusd_per_request
                    > policy.max_cost_microusd
                ):
                    item["status"] = "provider_exhausted"
                    item["attempts"].append(
                        {
                            "attempt": item["attempt_count"],
                            "error_code": "dispatch_budget_exhausted",
                            "outcome": "provider_exhausted",
                            "retryable": False,
                        }
                    )
                    self._write(receipt)
                    break

                attempt = item["attempt_count"] + 1
                # Persist admission before the provider call. A process death
                # cannot silently exceed the request/cost ceiling on resume.
                item["attempt_count"] = attempt
                receipt["provider_request_count"] += 1
                receipt["cost_microusd_spent"] += policy.cost_microusd_per_request
                item["attempts"].append(
                    {
                        "attempt": attempt,
                        "outcome": "admitted",
                        "retryable": False,
                    }
                )
                self._write(receipt)
                try:
                    result = execute_voice_tts_job(
                        job,
                        resolver=self._resolver,
                        text_to_speech_fn=self._provider,
                    )
                except Exception as exc:
                    code, retryable = self._error(exc)
                    item["attempts"][-1] = {
                        "attempt": attempt,
                        "error_code": code,
                        "outcome": "retry_scheduled" if retryable else "quarantined",
                        "retryable": retryable,
                    }
                    if not retryable:
                        item["status"] = "quarantined"
                    elif attempt >= policy.max_attempts_per_item:
                        item["status"] = "provider_exhausted"
                        item["attempts"][-1]["outcome"] = "provider_exhausted"
                    self._write(receipt)
                    if item["status"] in _TERMINAL_ITEM_STATUSES:
                        break
                    delay = self._backoff(policy, attempt)
                    if delay:
                        self._sleep(delay)
                    continue

                item["attempts"][-1] = {
                    "attempt": attempt,
                    "artifact_sha256": result["artifacts"][0]["sha256"],
                    "outcome": "regenerated",
                    "retryable": False,
                }
                item["result"] = result
                item["status"] = "regenerated"
                self._write(receipt)
                break

        # If the global ceiling was consumed, every remaining item receives a
        # terminal, content-bound provider-exhausted receipt without dispatch.
        for item in receipt["items"]:
            if item["status"] == "pending":
                item["status"] = "provider_exhausted"
                item["attempts"].append(
                    {
                        "attempt": item["attempt_count"],
                        "error_code": "dispatch_budget_exhausted",
                        "outcome": "provider_exhausted",
                        "retryable": False,
                    }
                )
        self._write(receipt)
        return receipt


__all__ = [
    "REGENERATION_DISPATCH_SCHEMA_VERSION",
    "REGENERATION_RUN_RECEIPT_SCHEMA_VERSION",
    "RegenerationDispatchManifest",
    "RegenerationEndpointContract",
    "RegenerationRunnerPolicy",
    "VoiceRegenerationError",
    "VoiceRegenerationRunner",
    "build_regeneration_dispatch_manifest",
]
