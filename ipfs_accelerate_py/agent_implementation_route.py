"""Agent-implementation route, fallback, and control-plane pin APIs.

Restored from the sealed provider-fallback authority surface that was dropped
from ``llm_router.py``. ``llm_router`` re-exports these names.
"""
from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import stat as stat_module
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_AGENT_IMPLEMENTATION_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
)
_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
)
_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
)
_AGENT_GROK_MODEL_ENV = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
_AGENT_CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
_AGENT_CODEX_REASONING_EFFORT_ENV = (
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
)
_AGENT_ROUTE_BOARD_NAMESPACE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE"
)
_AGENT_ROUTE_AUTHORIZATION_PATH_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH"
)
_AGENT_ROUTE_AUTHORIZATION_SHA256_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256"
)
_AGENT_ROUTE_AUTHORIZATION_ID_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID"
)
_AGENT_ROUTE_AUTHORIZATION_KIND_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND"
)
_AGENT_ROUTE_SOURCE_HEAD_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD"
)
_AGENT_ROUTE_SOURCE_TREE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE"
)
_AGENT_ROUTE_ID_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID"
_V3_AGENT_ROUTE_BOARD_NAMESPACE = (
    "agent-supervisor-prompt-only-self-improvement-v3"
)
_V3_AGENT_ROUTE_AUTHORIZATION_PATH = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "provider_fallback_policy_authorization_20260808.json"
)
_V3_AGENT_LIFECYCLE_ROOT_PIN_PATH = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "local_profile_lifecycle_root_pin_20260808.json"
)
_AGENT_ROUTE_AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "provider-fallback-policy-authorization@2"
)
_AGENT_ROUTE_REVIEW_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-policy-review@2"
)
_AGENT_LIFECYCLE_ROOT_PIN_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.local-profile-lifecycle-root-pin@1"
)
_AGENT_INVOCATION_BINDING_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-invocation@2"
)
_AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS = 5 * 60 * 1000
_AGENT_IMPLEMENTATION_MAX_CLOCK_SKEW_MS = 5 * 1000
AGENT_IMPLEMENTATION_CODEX_IMAGE_ID = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL = "2026-08-03-v2"
_AGENT_CONTROL_PLANE_PIN_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2"
)
_AGENT_CONTROL_PLANE_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.materialized-control-plane@1"
)
_AGENT_CONTROL_PLANE_MANIFEST_FILENAME = (
    ".agent-control-plane-manifest.json"
)
_AGENT_CONTROL_PLANE_MAX_FILE_BYTES = 4 * 1024 * 1024
_AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_AGENT_CONTROL_PLANE_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
# Non-supervisor roots plus the security-critical supervisor modules called out
# explicitly for auditability.  Capsule construction additionally walks and
# hashes the complete ``agent_supervisor`` Python source tree on every build and
# verification, so a newly added or indirect daemon/runner dependency cannot
# fall outside the pin.  Candidate worktrees are never roots.
_AGENT_CONTROL_PLANE_RELATIVE_FILES = (
    "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py/llm_router.py",
    "ipfs_accelerate_py/agent_implementation_route.py",
    "ipfs_accelerate_py/router_deps.py",
    "ipfs_accelerate_py/common/__init__.py",
    "ipfs_accelerate_py/common/meta_model_api.py",
    "ipfs_accelerate_py/model_catalog/__init__.py",
    "ipfs_accelerate_py/model_catalog/identity.py",
    "ipfs_accelerate_py/model_catalog/schema.py",
    "ipfs_accelerate_py/utils/__init__.py",
    "ipfs_accelerate_py/utils/mistral_vibe.py",
    "ipfs_accelerate_py/utils/cid_utils.py",
    "ipfs_accelerate_py/agent_supervisor/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_environment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_failure_policy.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/validation/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "scripts/ops/agent_supervisor/configured_board_scheduler.py",
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
)
_LEGACY_AGENT_IMPLEMENTATION_ROUTE_ID = (
    "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
)
_QUOTA_HIGH_AGENT_IMPLEMENTATION_ROUTE_ID = (
    "agent-supervisor-grok45-terra56-high-hard-quota-v1"
)
_V3_AGENT_IMPLEMENTATION_ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
# Runner and scheduler code import these projections instead of maintaining
# another provider/model/reasoning tuple.
AGENT_IMPLEMENTATION_CANONICAL_FALLBACK_MODEL_ID = "gpt-5.6-terra"
AGENT_IMPLEMENTATION_CANONICAL_FALLBACK_REASONING_EFFORT = "high"
_AGENT_IMPLEMENTATION_ROUTE_FIELDS = (
    "primary_provider_id",
    "primary_model_id",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_trigger",
    "fallback_reasoning_effort",
)
_AGENT_IMPLEMENTATION_GROK_ALIASES = frozenset(
    {
        "grok",
        "grok_cli",
        "grok-cli",
        "grok_build",
        "grok-build",
        "xai_cli",
        "xai-cli",
    }
)
_AGENT_IMPLEMENTATION_DIRECT_AUTH_EVIDENCE = frozenset(
    "sha256:" + hashlib.sha256(signal.encode("utf-8")).hexdigest()
    for signal in ("not signed in", "not authenticated")
)
_AGENT_IMPLEMENTATION_QUOTA_VERIFIER_RESULTS = frozenset(
    {"usage_pool_exhausted", "spending_limit_exhausted"}
)
AGENT_IMPLEMENTATION_GROK_NOT_SIGNED_IN_GUIDANCE = (
    "Error: Not signed in. To authenticate without a browser, run:\n"
    "  grok login --device-code\n\n"
    "Alternatively, set the XAI_API_KEY environment variable or run `grok "
    "login` on a machine with a browser."
)
_AGENT_IMPLEMENTATION_FAILURE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.grok-failure-receipt@4"
)
_AGENT_IMPLEMENTATION_MAX_FAILURE_EVIDENCE_BYTES = 128 * 1024
_AGENT_IMPLEMENTATION_FAILURE_SOURCE = (
    "isolated_no_tools_pre_dispatch_probe"
)
_AGENT_IMPLEMENTATION_PROBE_PROMPT = (
    "This is a provider-capacity preflight. Reply with exactly OK."
)
_AGENT_IMPLEMENTATION_PROBE_CONTRACT = {
    "schema": "ipfs_accelerate_py.agent_supervisor.grok-quota-probe@1",
    "model": "grok-4.5",
    "mode": "chat",
    "max_turns": 1,
    "permission_mode": "dontAsk",
    "tools": "",
    "no_plan": True,
    "no_subagents": True,
    "disable_web_search": True,
    "no_memory": True,
    "isolated_workspace": True,
    "task_context": False,
    "prompt": _AGENT_IMPLEMENTATION_PROBE_PROMPT,
    "timeout_seconds": 60,
}
_AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            _AGENT_IMPLEMENTATION_PROBE_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
)
_AGENT_HARD_QUOTA_PATTERN = re.compile(
    r"(?:"
    r"(?:grok(?:\s+build)?|xai)[^\r\n]{0,200}(?:"
    r"\b402\b|insufficient[_ ]quota|quota[_ ]exceeded|quota exhausted|"
    r"balance exhausted|usage balance exhausted)|"
    r"(?:\b402\b|insufficient[_ ]quota|quota[_ ]exceeded|quota exhausted|"
    r"balance exhausted|usage balance exhausted)[^\r\n]{0,200}"
    r"(?:grok(?:\s+build)?|xai)|"
    r"status\s+402|out of credits|usage balance exhausted|"
    r"over (?:your )?spending limit"
    r")",
    re.IGNORECASE,
)
_AGENT_RATE_LIMIT_PATTERN = re.compile(
    r"(?:\b429\b|rate[_ -]?limit(?:ed|s|_exceeded)?|too many requests|"
    r"resource[_ -]?exhausted|overloaded)",
    re.IGNORECASE,
)
_AGENT_AUTH_PATTERN = re.compile(
    r"(?:\b401\b|\b403\b|not signed in|not authenticated|authentication "
    r"failed|invalid api key|unauthorized|forbidden)",
    re.IGNORECASE,
)
_AGENT_AUTH_UNAVAILABLE_PATTERN = re.compile(
    r"\A\s*(?:error:\s*)?(?P<signal>not signed in|not authenticated)"
    r"[.!]?\s*\Z",
    re.IGNORECASE,
)
_AGENT_INVALID_REQUEST_PATTERN = re.compile(
    r"(?:\b400\b|invalid model|model not found|bad request|invalid argument)",
    re.IGNORECASE,
)
_AGENT_TRANSPORT_PATTERN = re.compile(
    r"(?:tls|certificate|connection (?:refused|reset)|dns|name resolution|"
    r"network unreachable|timed? out|timeout)",
    re.IGNORECASE,
)


def _content_addressed_mapping(
    value: Mapping[str, object],
    *,
    identity_field: str,
) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _agent_dag_json_content_identity(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    raw = b"\x01\xa9\x02\x12\x20" + hashlib.sha256(encoded).digest()
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def classify_agent_implementation_failure(
    stderr_text: str,
    *,
    max_evidence_bytes: int = 128 * 1024,
) -> dict[str, str]:
    """Sole classifier for bounded Grok implementation preflight evidence."""

    encoded = str(stderr_text or "").encode("utf-8", errors="replace")
    text = encoded[-int(max_evidence_bytes) :].decode(
        "utf-8",
        errors="replace",
    )
    stripped_text = text.strip()
    auth_unavailable = _AGENT_AUTH_UNAVAILABLE_PATTERN.fullmatch(text)
    if (
        auth_unavailable is not None
        or stripped_text == AGENT_IMPLEMENTATION_GROK_NOT_SIGNED_IN_GUIDANCE
    ):
        normalized = (
            " ".join(auth_unavailable.group("signal").lower().split())
            if auth_unavailable is not None
            else "not signed in"
        )
        return {
            "failure_class": "authentication_unavailable",
            "evidence_sha256": "sha256:"
            + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        }
    classes = (
        ("authentication", _AGENT_AUTH_PATTERN),
        ("invalid_request", _AGENT_INVALID_REQUEST_PATTERN),
        ("rate_limited", _AGENT_RATE_LIMIT_PATTERN),
        ("transport", _AGENT_TRANSPORT_PATTERN),
        ("hard_quota_exhausted", _AGENT_HARD_QUOTA_PATTERN),
    )
    for failure_class, pattern in classes:
        match = pattern.search(text)
        if match is None:
            continue
        normalized = " ".join(match.group(0).lower().split())
        return {
            "failure_class": failure_class,
            "evidence_sha256": "sha256:"
            + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        }
    return {
        "failure_class": "unknown",
        "evidence_sha256": "sha256:"
        + hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def build_agent_implementation_failure_receipt(
    *,
    probe_stderr_text: str,
    nonce: str,
    model: str,
    probe_returncode: int,
    primary_dispatched: bool = False,
    evidence_size: int | None = None,
    evidence_overflow: bool | None = None,
    observed_at_ms: int | None = None,
    freshness_ms: int = 60 * 1000,
) -> dict[str, object]:
    observed = (
        int(time.time() * 1000)
        if observed_at_ms is None
        else observed_at_ms
    )
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed <= 0
        or isinstance(freshness_ms, bool)
        or not isinstance(freshness_ms, int)
        or not 0 < freshness_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
    ):
        raise ValueError("failure receipt freshness is invalid")
    measured_size = (
        len(str(probe_stderr_text or "").encode("utf-8", errors="replace"))
        if evidence_size is None
        else int(evidence_size)
    )
    measured_overflow = (
        measured_size > _AGENT_IMPLEMENTATION_MAX_FAILURE_EVIDENCE_BYTES
        if evidence_overflow is None
        else bool(evidence_overflow)
    )
    classified = classify_agent_implementation_failure(probe_stderr_text)
    receipt: dict[str, object] = {
        "schema": _AGENT_IMPLEMENTATION_FAILURE_RECEIPT_SCHEMA,
        "source": _AGENT_IMPLEMENTATION_FAILURE_SOURCE,
        "probe_contract_id": _AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID,
        "nonce": str(nonce),
        "primary_provider": "grok",
        "primary_model": str(model),
        "primary_dispatched": bool(primary_dispatched),
        "probe_returncode": int(probe_returncode),
        "evidence_size": measured_size,
        "evidence_overflow": measured_overflow,
        "observed_at_ms": observed,
        "expires_at_ms": observed + freshness_ms,
        **classified,
    }
    receipt["receipt_id"] = _content_addressed_mapping(
        receipt,
        identity_field="receipt_id",
    )
    return receipt


def valid_agent_implementation_failure_receipt(
    receipt: Mapping[str, object],
    *,
    nonce: str,
    model: str,
    probe_returncode: int,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
) -> bool:
    expected_fields = {
        "schema",
        "source",
        "probe_contract_id",
        "nonce",
        "primary_provider",
        "primary_model",
        "primary_dispatched",
        "probe_returncode",
        "evidence_size",
        "evidence_overflow",
        "observed_at_ms",
        "expires_at_ms",
        "failure_class",
        "evidence_sha256",
        "receipt_id",
    }
    observed_returncode = receipt.get("probe_returncode")
    evidence_size = receipt.get("evidence_size")
    evidence_overflow = receipt.get("evidence_overflow")
    observed_at_ms = receipt.get("observed_at_ms")
    expires_at_ms = receipt.get("expires_at_ms")
    freshness_requested = now_ms is not None or max_age_ms is not None
    freshness_valid = bool(
        isinstance(observed_at_ms, int)
        and not isinstance(observed_at_ms, bool)
        and observed_at_ms > 0
        and isinstance(expires_at_ms, int)
        and not isinstance(expires_at_ms, bool)
        and observed_at_ms < expires_at_ms
        and expires_at_ms - observed_at_ms
        <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
    )
    if freshness_requested:
        freshness_valid = bool(
            freshness_valid
            and isinstance(now_ms, int)
            and not isinstance(now_ms, bool)
            and now_ms > 0
            and isinstance(max_age_ms, int)
            and not isinstance(max_age_ms, bool)
            and 0 < max_age_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
            and observed_at_ms <= now_ms + _AGENT_IMPLEMENTATION_MAX_CLOCK_SKEW_MS
            and now_ms <= expires_at_ms
            and now_ms - observed_at_ms <= max_age_ms
        )
    return bool(
        set(receipt) == expected_fields
        and receipt.get("schema")
        == _AGENT_IMPLEMENTATION_FAILURE_RECEIPT_SCHEMA
        and receipt.get("source") == _AGENT_IMPLEMENTATION_FAILURE_SOURCE
        and receipt.get("probe_contract_id")
        == _AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID
        and re.fullmatch(r"[0-9a-f]{64}", str(nonce or ""))
        and receipt.get("nonce") == nonce
        and receipt.get("primary_provider") == "grok"
        and receipt.get("primary_model") == model == "grok-4.5"
        and receipt.get("primary_dispatched") is False
        and isinstance(evidence_size, int)
        and not isinstance(evidence_size, bool)
        and evidence_size >= 0
        and isinstance(evidence_overflow, bool)
        and freshness_valid
        and evidence_overflow
        is (
            evidence_size
            > _AGENT_IMPLEMENTATION_MAX_FAILURE_EVIDENCE_BYTES
        )
        and receipt.get("failure_class")
        in {
            "hard_quota_exhausted",
            "authentication_unavailable",
            "authentication",
            "invalid_request",
            "rate_limited",
            "transport",
            "unknown",
        }
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("evidence_sha256") or ""),
        )
        and isinstance(observed_returncode, int)
        and not isinstance(observed_returncode, bool)
        and observed_returncode == probe_returncode != 0
        and receipt.get("receipt_id")
        == _content_addressed_mapping(receipt, identity_field="receipt_id")
    )


_AGENT_IMPLEMENTATION_PRIVATE_SEAL_KEY = secrets.token_bytes(32)


def _agent_implementation_private_seal(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(
        _AGENT_IMPLEMENTATION_PRIVATE_SEAL_KEY,
        encoded,
        hashlib.sha256,
    ).hexdigest()


def _agent_verify_did_signature(
    *, identity_did: str, payload: Mapping[str, object], signature: str
) -> None:
    """Verify canonical JSON with the real Ed25519 ``did:key`` verifier."""

    # Import lazily: llm_router is imported by the supervisor package itself.
    from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
        LocalProfileTampered,
        verify_did_key_signature,
    )

    try:
        verify_did_key_signature(
            identity_did=identity_did,
            payload=payload,
            signature=signature,
        )
    except LocalProfileTampered as exc:
        raise ValueError("agent route reviewer signature is invalid") from exc


def _agent_string(value: object, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(character in normalized for character in "\0\n\r"):
        raise ValueError(f"{name} must be a nonempty bound value")
    return normalized


@dataclass(frozen=True, slots=True)
class AgentImplementationAuthorityBounds:
    """Exact repository/effect/budget/resource authority signed by review."""

    repository_cid: str
    baseline_commit: str
    effects: tuple[str, ...]
    budget_cid: str
    resource_cid: str
    authority_cid: str

    def as_dict(self) -> dict[str, object]:
        return {
            "repository_cid": self.repository_cid,
            "baseline_commit": self.baseline_commit,
            "effects": list(self.effects),
            "budget_cid": self.budget_cid,
            "resource_cid": self.resource_cid,
            "authority_cid": self.authority_cid,
        }


@dataclass(frozen=True, slots=True)
class AgentImplementationControlPlanePin:
    """Immutable accepted-generation runner/capsule provenance."""

    schema: str
    runner_path: str
    runner_sha256: str
    capsule_root: str
    capsule_id: str
    source_head: str
    source_tree: str
    archive_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "runner_path": self.runner_path,
            "runner_sha256": self.runner_sha256,
            "capsule_root": self.capsule_root,
            "capsule_id": self.capsule_id,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "archive_sha256": self.archive_sha256,
        }


@dataclass(frozen=True, slots=True)
class AgentImplementationSealedControlPlane:
    """One owner-unmodifiable accepted-generation zipapp descriptor."""

    descriptor: int
    executable_path: str
    archive_sha256: str
    seals: int
    capsule_id: str


@dataclass(frozen=True, slots=True)
class AgentImplementationInvocationBinding:
    """Reviewer-signed equality contract for one logical provider attempt."""

    schema: str
    invocation_id: str
    logical_attempt_id: str
    task_id: str
    attempt: int
    task_revision_cid: str
    prompt_cid: str
    worktree_id: str
    workspace_path: str
    repository_cid: str
    baseline_commit: str
    effects: tuple[str, ...]
    scope_cid: str
    budget_cid: str
    resource_cid: str
    authority_cid: str
    route_id: str
    primary_provider_id: str
    primary_model_id: str
    fallback_provider_id: str
    fallback_model_id: str
    fallback_reasoning_effort: str
    fallback_implementer_identity: str
    reviewer_identity: str
    reviewer_provider: str
    profile_id: str
    profile_identity_did: str
    profile_lifecycle_anchor_id: str
    profile_lifecycle_generation: int
    profile_dir: str
    profile_lifecycle_dir: str
    issued_at_ms: int
    expires_at_ms: int
    provider_attempt_store: str
    provider_attempt_store_identity: str
    control_plane: AgentImplementationControlPlanePin
    reviewer_signature: str

    def signed_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "invocation_id": self.invocation_id,
            "logical_attempt_id": self.logical_attempt_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "task_revision_cid": self.task_revision_cid,
            "prompt_cid": self.prompt_cid,
            "worktree_id": self.worktree_id,
            "workspace_path": self.workspace_path,
            "repository_cid": self.repository_cid,
            "baseline_commit": self.baseline_commit,
            "effects": list(self.effects),
            "scope_cid": self.scope_cid,
            "budget_cid": self.budget_cid,
            "resource_cid": self.resource_cid,
            "authority_cid": self.authority_cid,
            "route_id": self.route_id,
            "primary_provider_id": self.primary_provider_id,
            "primary_model_id": self.primary_model_id,
            "fallback_provider_id": self.fallback_provider_id,
            "fallback_model_id": self.fallback_model_id,
            "fallback_reasoning_effort": self.fallback_reasoning_effort,
            "fallback_implementer_identity": self.fallback_implementer_identity,
            "reviewer_identity": self.reviewer_identity,
            "reviewer_provider": self.reviewer_provider,
            "profile_id": self.profile_id,
            "profile_identity_did": self.profile_identity_did,
            "profile_lifecycle_anchor_id": self.profile_lifecycle_anchor_id,
            "profile_lifecycle_generation": (
                self.profile_lifecycle_generation
            ),
            "profile_dir": self.profile_dir,
            "profile_lifecycle_dir": self.profile_lifecycle_dir,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "provider_attempt_store": self.provider_attempt_store,
            "provider_attempt_store_identity": (
                self.provider_attempt_store_identity
            ),
            "control_plane": self.control_plane.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "reviewer_signature": self.reviewer_signature,
        }

    @property
    def content_id(self) -> str:
        return _content_addressed_mapping(
            self.as_dict(), identity_field="content_id"
        )


@dataclass(frozen=True, slots=True)
class AgentImplementationRouteAuthorization:
    """Content-bound operator authority for the scoped auth/high route."""

    board_namespace: str
    artifact_path: str
    artifact_sha256: str
    authorization_kind: str
    source_head: str
    source_tree: str
    authorization_id: str
    reviewer_identity: str = ""
    reviewer_provider: str = ""
    reviewer_signature: str = ""
    reviewer_profile_id: str = ""
    reviewer_profile_content_id: str = ""
    reviewer_lifecycle_anchor_id: str = ""
    reviewer_lifecycle_generation: int = 0
    reviewer_witness_path: str = ""
    reviewer_witness_sha256: str = ""
    lifecycle_root_identity_did: str = ""
    lifecycle_witness_nonce: str = ""
    lifecycle_root_pin_path: str = ""
    lifecycle_root_pin_sha256: str = ""
    authorized_at_ms: int = 0
    fallback_implementer_identity: str = "codex"
    authority_bounds: AgentImplementationAuthorityBounds | None = None
    _validation_seal: str = field(default="", repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "board_namespace": self.board_namespace,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "authorization_kind": self.authorization_kind,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "authorization_id": self.authorization_id,
            "reviewer_identity": self.reviewer_identity,
            "reviewer_provider": self.reviewer_provider,
            "reviewer_signature": self.reviewer_signature,
            "reviewer_profile_id": self.reviewer_profile_id,
            "reviewer_profile_content_id": self.reviewer_profile_content_id,
            "reviewer_lifecycle_anchor_id": (
                self.reviewer_lifecycle_anchor_id
            ),
            "reviewer_lifecycle_generation": (
                self.reviewer_lifecycle_generation
            ),
            "reviewer_witness_path": self.reviewer_witness_path,
            "reviewer_witness_sha256": self.reviewer_witness_sha256,
            "lifecycle_root_identity_did": self.lifecycle_root_identity_did,
            "lifecycle_witness_nonce": self.lifecycle_witness_nonce,
            "lifecycle_root_pin_path": self.lifecycle_root_pin_path,
            "lifecycle_root_pin_sha256": self.lifecycle_root_pin_sha256,
            "authorized_at_ms": self.authorized_at_ms,
            "fallback_implementer_identity": self.fallback_implementer_identity,
            "authority_bounds": (
                self.authority_bounds.as_dict()
                if self.authority_bounds is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentImplementationRoutePlan:
    """Frozen exceptional route for typed side-effecting agent work.

    Generic side-effecting router requests remain cross-provider fail-closed.
    This plan is the narrow, explicit supervisor contract whose fallback can
    be authorized only by :func:`decide_agent_implementation_fallback`.
    """

    primary_provider_id: str
    primary_model_id: str
    fallback_provider_id: str
    fallback_model_id: str
    fallback_trigger: str
    fallback_reasoning_effort: str
    route_id: str
    authorization: AgentImplementationRouteAuthorization | None = None
    fallback_implementer_identity: str = "codex"
    invocation_binding: AgentImplementationInvocationBinding | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            field: str(getattr(self, field))
            for field in _AGENT_IMPLEMENTATION_ROUTE_FIELDS
        }

    def as_environment(self) -> dict[str, str]:
        environment = {
            _AGENT_IMPLEMENTATION_PROVIDER_ENV: self.primary_provider_id,
            _AGENT_IMPLEMENTATION_FALLBACK_PROVIDER_ENV: (
                self.fallback_provider_id
            ),
            _AGENT_IMPLEMENTATION_FALLBACK_TRIGGER_ENV: self.fallback_trigger,
            _AGENT_GROK_MODEL_ENV: self.primary_model_id,
            _AGENT_CODEX_MODEL_ENV: self.fallback_model_id,
            _AGENT_CODEX_REASONING_EFFORT_ENV: (
                self.fallback_reasoning_effort
            ),
        }
        if self.authorization is not None:
            environment.update(
                {
                    _AGENT_ROUTE_BOARD_NAMESPACE_ENV: (
                        self.authorization.board_namespace
                    ),
                    _AGENT_ROUTE_AUTHORIZATION_PATH_ENV: (
                        self.authorization.artifact_path
                    ),
                    _AGENT_ROUTE_AUTHORIZATION_SHA256_ENV: (
                        self.authorization.artifact_sha256
                    ),
                    _AGENT_ROUTE_AUTHORIZATION_ID_ENV: (
                        self.authorization.authorization_id
                    ),
                    _AGENT_ROUTE_AUTHORIZATION_KIND_ENV: (
                        self.authorization.authorization_kind
                    ),
                    _AGENT_ROUTE_SOURCE_HEAD_ENV: (
                        self.authorization.source_head
                    ),
                    _AGENT_ROUTE_SOURCE_TREE_ENV: (
                        self.authorization.source_tree
                    ),
                    _AGENT_ROUTE_ID_ENV: self.route_id,
                }
            )
        return environment

    def as_binding_dict(self) -> dict[str, object]:
        return {
            **self.as_dict(),
            "route_id": self.route_id,
            "authorization": (
                self.authorization.as_dict()
                if self.authorization is not None
                else None
            ),
            "fallback_implementer_identity": self.fallback_implementer_identity,
            "invocation_binding": (
                self.invocation_binding.as_dict()
                if self.invocation_binding is not None
                else None
            ),
        }

    def as_outcome_dict(self) -> dict[str, object]:
        """Return the bounded audit projection used in terminal log records.

        The command carries the complete signed binding.  Repeating that
        binding in a line-oriented terminal record can exceed the existing
        bounded record protocol once real task paths and content identities
        are present.  This projection retains the exact route plus
        authorization, invocation, reviewer, and accepted-capsule identities;
        the daemon reconstructs it from the full command binding before
        accepting the record.
        """

        if self.invocation_binding is None:
            return self.as_binding_dict()
        authorization = self.authorization
        invocation = self.invocation_binding
        return {
            **self.as_dict(),
            "route_id": self.route_id,
            "authorization_id": (
                authorization.authorization_id if authorization else ""
            ),
            "reviewer_identity": (
                authorization.reviewer_identity if authorization else ""
            ),
            "reviewer_provider": (
                authorization.reviewer_provider if authorization else ""
            ),
            "fallback_implementer_identity": (
                self.fallback_implementer_identity
            ),
            "invocation_binding_id": invocation.content_id,
            "accepted_control_plane": invocation.control_plane.as_dict(),
        }

    @property
    def permits_authentication_unavailable(self) -> bool:
        return self.fallback_trigger == "primary_quota_or_auth_unavailable"


def _agent_route_authorization_is_sealed(
    authorization: AgentImplementationRouteAuthorization | None,
) -> bool:
    if authorization is None:
        return False
    identity_body: dict[str, object] = {
        "schema": _AGENT_ROUTE_AUTHORIZATION_SCHEMA,
        "board_namespace": authorization.board_namespace,
        "artifact_path": authorization.artifact_path,
        "artifact_sha256": authorization.artifact_sha256,
        "authorization_kind": authorization.authorization_kind,
        "source_head": authorization.source_head,
        "source_tree": authorization.source_tree,
        "reviewer_identity": authorization.reviewer_identity,
        "reviewer_provider": authorization.reviewer_provider,
        "reviewer_signature": authorization.reviewer_signature,
        "reviewer_profile_id": authorization.reviewer_profile_id,
        "reviewer_profile_content_id": (
            authorization.reviewer_profile_content_id
        ),
        "reviewer_lifecycle_anchor_id": (
            authorization.reviewer_lifecycle_anchor_id
        ),
        "reviewer_lifecycle_generation": (
            authorization.reviewer_lifecycle_generation
        ),
        "reviewer_witness_path": authorization.reviewer_witness_path,
        "reviewer_witness_sha256": authorization.reviewer_witness_sha256,
        "lifecycle_root_identity_did": (
            authorization.lifecycle_root_identity_did
        ),
        "lifecycle_witness_nonce": authorization.lifecycle_witness_nonce,
        "lifecycle_root_pin_path": authorization.lifecycle_root_pin_path,
        "lifecycle_root_pin_sha256": (
            authorization.lifecycle_root_pin_sha256
        ),
        "authorized_at_ms": authorization.authorized_at_ms,
        "fallback_implementer_identity": (
            authorization.fallback_implementer_identity
        ),
        "authority_bounds": (
            authorization.authority_bounds.as_dict()
            if authorization.authority_bounds is not None
            else None
        ),
    }
    return bool(
        authorization._validation_seal
        == _agent_implementation_private_seal(identity_body)
        and authorization.reviewer_identity
        and authorization.reviewer_provider
        and authorization.reviewer_provider not in {"codex", "openai"}
        and authorization.reviewer_signature
        and authorization.authority_bounds is not None
    )


@dataclass(frozen=True, slots=True)
class AgentImplementationProviderCapacityObservation:
    """Strict immutable projection of ``ProviderCapacity.to_dict()``."""

    provider_id: str
    healthy: bool
    quota_remaining: int
    latency_ms: int
    context_window_tokens: int
    token_budget_remaining: int
    max_concurrency: int
    active_requests: int
    capabilities: tuple[str, ...]
    observed_at_ms: int
    retry_after_ms: int
    available_concurrency: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "AgentImplementationProviderCapacityObservation":
        if type(value) is not dict:
            raise ValueError("provider capacity observation must be a mapping")
        expected = {
            "provider_id",
            "healthy",
            "quota_remaining",
            "latency_ms",
            "context_window_tokens",
            "token_budget_remaining",
            "max_concurrency",
            "active_requests",
            "capabilities",
            "observed_at_ms",
            "retry_after_ms",
            "available_concurrency",
        }
        if set(value) != expected:
            raise ValueError("provider capacity observation has noncanonical fields")
        raw_provider = value.get("provider_id")
        if not isinstance(raw_provider, str) or not raw_provider.strip():
            raise ValueError("provider capacity provider_id is invalid")
        provider_alias = raw_provider.strip().casefold()
        aliases = {
            "grok": "grok_cli",
            "grok_cli": "grok_cli",
            "codex": "codex_cli",
            "codex_cli": "codex_cli",
        }
        provider_id = aliases.get(provider_alias)
        if provider_id is None:
            raise ValueError("provider capacity provider_id is not route-capable")
        healthy = value.get("healthy")
        if not isinstance(healthy, bool):
            raise ValueError("provider capacity healthy must be an exact boolean")
        integer_fields = (
            "quota_remaining",
            "latency_ms",
            "context_window_tokens",
            "token_budget_remaining",
            "max_concurrency",
            "active_requests",
            "observed_at_ms",
            "retry_after_ms",
            "available_concurrency",
        )
        integers: dict[str, int] = {}
        for name in integer_fields:
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(
                    f"provider capacity {name} must be an exact integer"
                )
            integers[name] = item
        for name in (
            "latency_ms",
            "max_concurrency",
            "active_requests",
            "observed_at_ms",
            "retry_after_ms",
            "available_concurrency",
        ):
            if integers[name] < 0:
                raise ValueError(f"provider capacity {name} is negative")
        for name in (
            "quota_remaining",
            "context_window_tokens",
            "token_budget_remaining",
        ):
            if integers[name] < -1:
                raise ValueError(f"provider capacity {name} is invalid")
        raw_capabilities = value.get("capabilities")
        if not isinstance(raw_capabilities, list):
            raise ValueError("provider capacity capabilities are invalid")
        capabilities: list[str] = []
        for item in raw_capabilities:
            if (
                not isinstance(item, str)
                or not item.strip()
                or item != item.strip()
                or any(character in item for character in "\0\n\r")
            ):
                raise ValueError("provider capacity capability is invalid")
            capabilities.append(item)
        canonical_capabilities = tuple(sorted(set(capabilities)))
        if tuple(capabilities) != canonical_capabilities:
            raise ValueError("provider capacity capabilities are noncanonical")
        expected_available = max(
            0,
            integers["max_concurrency"] - integers["active_requests"],
        )
        if integers["available_concurrency"] != expected_available:
            raise ValueError("provider capacity available_concurrency drifted")
        return cls(
            provider_id=provider_id,
            healthy=healthy,
            capabilities=canonical_capabilities,
            **integers,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "healthy": self.healthy,
            "quota_remaining": self.quota_remaining,
            "latency_ms": self.latency_ms,
            "context_window_tokens": self.context_window_tokens,
            "token_budget_remaining": self.token_budget_remaining,
            "max_concurrency": self.max_concurrency,
            "active_requests": self.active_requests,
            "capabilities": list(self.capabilities),
            "observed_at_ms": self.observed_at_ms,
            "retry_after_ms": self.retry_after_ms,
            "available_concurrency": self.available_concurrency,
        }

    @property
    def evidence_id(self) -> str:
        return _content_addressed_mapping(
            self.as_dict(), identity_field="evidence_id"
        )


@dataclass(frozen=True, slots=True)
class AgentImplementationRouteCapacityLane:
    """Audit-only lane within a logical route capacity projection."""

    role: str
    provider_id: str
    model_id: str
    reasoning_effort: str
    capacity_available: bool
    dispatch_authorized: bool
    observation_id: str
    available_concurrency: int
    observed_at_ms: int
    fresh_until_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "capacity_available": self.capacity_available,
            "dispatch_authorized": self.dispatch_authorized,
            "observation_id": self.observation_id,
            "available_concurrency": self.available_concurrency,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
        }


@dataclass(frozen=True, slots=True)
class AgentImplementationRouteCapacityProfile:
    """Canonical logical ``ProviderCapacity`` plus route lane evidence."""

    schema: str
    route_id: str
    provider_id: str
    healthy: bool
    quota_remaining: int
    latency_ms: int
    context_window_tokens: int
    token_budget_remaining: int
    max_concurrency: int
    active_requests: int
    capabilities: tuple[str, ...]
    observed_at_ms: int
    retry_after_ms: int
    available_concurrency: int
    max_age_ms: int
    fresh_until_ms: int
    lanes: tuple[AgentImplementationRouteCapacityLane, ...]
    profile_id: str

    @property
    def schedulable(self) -> bool:
        return self.healthy and self.available_concurrency > 0

    def as_provider_capacity(self) -> dict[str, object]:
        """Return the exact compiler ``ProviderCapacity.to_dict`` shape."""

        return {
            "provider_id": self.provider_id,
            "healthy": self.healthy,
            "quota_remaining": self.quota_remaining,
            "latency_ms": self.latency_ms,
            "context_window_tokens": self.context_window_tokens,
            "token_budget_remaining": self.token_budget_remaining,
            "max_concurrency": self.max_concurrency,
            "active_requests": self.active_requests,
            "capabilities": list(self.capabilities),
            "observed_at_ms": self.observed_at_ms,
            "retry_after_ms": self.retry_after_ms,
            "available_concurrency": self.available_concurrency,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_id": self.route_id,
            **self.as_provider_capacity(),
            "schedulable": self.schedulable,
            "max_age_ms": self.max_age_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "lanes": [lane.as_dict() for lane in self.lanes],
            "profile_id": self.profile_id,
        }

    def as_compiler_snapshot(self) -> dict[str, object]:
        """Return the evidence-bound snapshot scheduler/compiler must retain."""

        return self.as_dict()


def project_agent_implementation_route_capacity(
    route: AgentImplementationRoutePlan,
    *,
    observations: list[Mapping[str, object]],
    now_ms: int,
    max_age_ms: int,
) -> AgentImplementationRouteCapacityProfile:
    """Build capacity from monitor records without granting fallback effects.

    Callers pass raw, strict ``ProviderCapacity.to_dict()`` observations.  The
    router owns alias normalization, freshness, sealed-route interpretation,
    and the aggregate compiler record.  Neither caller booleans nor task
    failure evidence participate in this planning-only projection.
    """

    if (
        isinstance(now_ms, bool)
        or not isinstance(now_ms, int)
        or now_ms <= 0
        or isinstance(max_age_ms, bool)
        or not isinstance(max_age_ms, int)
        or not 0 < max_age_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
    ):
        raise ValueError("route capacity freshness bounds are invalid")
    if not isinstance(observations, list) or len(observations) != 2:
        raise ValueError("route capacity requires exactly two observations")
    if any(type(item) is not dict for item in observations):
        raise ValueError("route capacity observations must be mappings")
    parsed_observations = tuple(
        AgentImplementationProviderCapacityObservation.from_mapping(item)
        for item in observations
    )
    by_provider = {item.provider_id: item for item in parsed_observations}
    if set(by_provider) != {"grok_cli", "codex_cli"}:
        raise ValueError(
            "route capacity observations are missing, duplicate, or mixed"
        )
    primary = by_provider["grok_cli"]
    fallback = by_provider["codex_cli"]
    if primary.observed_at_ms <= 0 or fallback.observed_at_ms <= 0:
        raise ValueError("route capacity observations require positive timestamps")
    if route.primary_provider_id != "grok_cli":
        raise ValueError("route capacity primary identity drifted")

    def usable(
        item: AgentImplementationProviderCapacityObservation,
    ) -> bool:
        return bool(
            item.healthy
            and item.observed_at_ms <= now_ms
            and now_ms - item.observed_at_ms <= max_age_ms
            and item.available_concurrency > 0
            and item.quota_remaining != 0
            and item.context_window_tokens != 0
            and item.token_budget_remaining != 0
            and item.retry_after_ms == 0
        )

    primary_ready = usable(primary)
    sealed_fallback = bool(
        route.permits_authentication_unavailable
        and _agent_route_authorization_is_sealed(route.authorization)
        and route.fallback_provider_id == "codex"
        and route.fallback_model_id == "gpt-5.6-terra"
        and route.fallback_reasoning_effort == "high"
        and route.route_id == _V3_AGENT_IMPLEMENTATION_ROUTE_ID
    )
    fallback_ready = sealed_fallback and usable(fallback)
    lane_values = (
        (
            "primary",
            primary,
            route.primary_provider_id,
            route.primary_model_id,
            "",
            primary_ready,
        ),
        (
            "typed_fallback_capacity_only",
            fallback,
            route.fallback_provider_id,
            route.fallback_model_id,
            route.fallback_reasoning_effort,
            fallback_ready,
        ),
    )
    lanes = tuple(
        AgentImplementationRouteCapacityLane(
            role=role,
            provider_id=provider_id,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            capacity_available=ready,
            dispatch_authorized=False,
            observation_id=observation.evidence_id,
            available_concurrency=(
                observation.available_concurrency if ready else 0
            ),
            observed_at_ms=observation.observed_at_ms,
            fresh_until_ms=observation.observed_at_ms + max_age_ms,
        )
        for (
            role,
            observation,
            provider_id,
            model_id,
            reasoning_effort,
            ready,
        ) in lane_values
    )
    selected = primary if primary_ready else fallback if fallback_ready else None
    # Every dispatch probes the primary first.  Fallback headroom is therefore
    # an alternative only when the primary lane is unusable, never additive
    # speculative concurrency.
    available = (selected,) if selected is not None else ()

    def additive(name: str) -> int:
        values = tuple(getattr(item, name) for item in available)
        if any(value == -1 for value in values):
            return -1
        return sum(values)

    if available:
        capability_sets = tuple(set(item.capabilities) for item in available)
        capabilities = tuple(sorted(set.intersection(*capability_sets)))
        observed_at = min(item.observed_at_ms for item in available)
        latency_ms = min(item.latency_ms for item in available)
        context_window_tokens = max(
            item.context_window_tokens for item in available
        )
        max_concurrency = sum(item.max_concurrency for item in available)
        active_requests = sum(item.active_requests for item in available)
        available_concurrency = sum(
            item.available_concurrency for item in available
        )
    else:
        capabilities = ()
        observed_at = min(primary.observed_at_ms, fallback.observed_at_ms)
        latency_ms = 0
        context_window_tokens = 0
        max_concurrency = 0
        active_requests = 0
        available_concurrency = 0
    schema = (
        "ipfs_accelerate_py.agent_supervisor."
        "implementation-route-capacity@2"
    )
    body: dict[str, object] = {
        "schema": schema,
        "route_id": route.route_id,
        "provider_id": route.route_id,
        "healthy": bool(available_concurrency),
        "quota_remaining": additive("quota_remaining") if available else 0,
        "latency_ms": latency_ms,
        "context_window_tokens": context_window_tokens,
        "token_budget_remaining": (
            additive("token_budget_remaining") if available else 0
        ),
        "max_concurrency": max_concurrency,
        "active_requests": active_requests,
        "capabilities": list(capabilities),
        "observed_at_ms": observed_at,
        "retry_after_ms": 0,
        "available_concurrency": available_concurrency,
        "max_age_ms": max_age_ms,
        "fresh_until_ms": observed_at + max_age_ms,
        "lanes": [lane.as_dict() for lane in lanes],
    }
    return AgentImplementationRouteCapacityProfile(
        schema=schema,
        route_id=route.route_id,
        provider_id=route.route_id,
        healthy=bool(available_concurrency),
        quota_remaining=int(body["quota_remaining"]),
        latency_ms=latency_ms,
        context_window_tokens=context_window_tokens,
        token_budget_remaining=int(body["token_budget_remaining"]),
        max_concurrency=max_concurrency,
        active_requests=active_requests,
        capabilities=capabilities,
        observed_at_ms=observed_at,
        retry_after_ms=0,
        available_concurrency=available_concurrency,
        max_age_ms=max_age_ms,
        fresh_until_ms=observed_at + max_age_ms,
        lanes=lanes,
        profile_id=_content_addressed_mapping(body, identity_field="profile_id"),
    )


@dataclass(frozen=True, slots=True)
class AgentImplementationFallbackDecision:
    """Pure typed decision; process execution remains the caller's job."""

    authorized: bool
    requires_independent_quota_verification: bool
    reason_code: str
    verifier_status: str
    route_id: str = ""
    fallback_provider_id: str = ""
    fallback_model_id: str = ""
    fallback_reasoning_effort: str = ""
    reviewer_identity: str = ""
    reviewer_provider: str = ""
    invocation_binding_id: str = ""
    control_plane_id: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "authorized": self.authorized,
            "requires_independent_quota_verification": (
                self.requires_independent_quota_verification
            ),
            "reason_code": self.reason_code,
            "verifier_status": self.verifier_status,
            "route_id": self.route_id,
            "fallback_provider_id": self.fallback_provider_id,
            "fallback_model_id": self.fallback_model_id,
            "fallback_reasoning_effort": self.fallback_reasoning_effort,
            "reviewer_identity": self.reviewer_identity,
            "reviewer_provider": self.reviewer_provider,
            "invocation_binding_id": self.invocation_binding_id,
            "control_plane_id": self.control_plane_id,
        }

    @property
    def content_id(self) -> str:
        return _content_addressed_mapping(
            self.as_dict(),
            identity_field="content_id",
        )


_AGENT_EFFECT_AUTHORIZATION_CONTEXT_SCHEMA = (
    "ipfs_accelerate_py.agent-supervisor/"
    "provider-effect-authorization-context@1"
)


def build_agent_implementation_effect_authorization_context(
    *,
    route: AgentImplementationRoutePlan,
    repo_root: Path | str,
    failure_receipt: Mapping[str, object],
    decision: AgentImplementationFallbackDecision,
    expected_nonce: str,
    expected_model: str,
    expected_probe_returncode: int,
    quota_evidence: "AgentImplementationQuotaEvidence | None" = None,
) -> dict[str, object]:
    """Persist the exact pre-effect authority needed for crash adoption."""

    invocation = route.invocation_binding
    authorization = route.authorization
    if (
        invocation is None
        or authorization is None
        or decision.authorized is not True
    ):
        raise ValueError("effect authorization context requires a signed decision")
    if decision.content_id != _content_addressed_mapping(
        decision.as_dict(), identity_field="content_id"
    ):
        raise ValueError("effect authorization decision identity drifted")
    # Recovery cannot depend on a later checkout retaining the authorization,
    # root pin, or lifecycle witness.  Snapshot their exact already-validated
    # Git bytes into the winning CAS before releasing the inert container.
    root = resolve_agent_implementation_private_state_path(repo_root).resolve(
        strict=True
    )
    validated_authorization = load_agent_implementation_route_authorization(
        repo_root=root,
        artifact_path=authorization.artifact_path,
        board_namespace=authorization.board_namespace,
        expected_sha256=authorization.artifact_sha256,
        expected_authorization_id=authorization.authorization_id,
    )
    if validated_authorization.as_dict() != authorization.as_dict():
        raise ValueError("effect authorization changed before durable claim")

    def stable_authority_blob(relative: str, maximum_bytes: int) -> bytes:
        candidate = resolve_agent_implementation_private_state_path(
            root / relative
        )
        if not candidate.is_relative_to(root):
            raise ValueError("effect authorization snapshot escaped repository")
        return _agent_read_stable_file(candidate, maximum_bytes=maximum_bytes)

    artifact_raw = stable_authority_blob(
        authorization.artifact_path, 128 * 1024
    )
    root_pin_raw = stable_authority_blob(
        authorization.lifecycle_root_pin_path, 32 * 1024
    )
    witness_raw = stable_authority_blob(
        authorization.reviewer_witness_path, 128 * 1024
    )
    if (
        "sha256:" + hashlib.sha256(artifact_raw).hexdigest()
        != authorization.artifact_sha256
        or "sha256:" + hashlib.sha256(root_pin_raw).hexdigest()
        != authorization.lifecycle_root_pin_sha256
        or "sha256:" + hashlib.sha256(witness_raw).hexdigest()
        != authorization.reviewer_witness_sha256
    ):
        raise ValueError("effect authorization snapshot digest drifted")
    current_head = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    ).strip()
    current_tree = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{tree}"))
    ).strip()
    commit_time_raw = _agent_git_output(
        root,
        (
            "log",
            "-1",
            "--format=%ct",
            current_head,
            "--",
            authorization.artifact_path,
        ),
        maximum_bytes=64,
    ).strip()
    try:
        authorization_commit_time_ms = int(commit_time_raw) * 1000
    except ValueError as exc:
        raise ValueError("effect authorization commit time is invalid") from exc

    def git_blob_id(raw: bytes) -> str:
        prefix = f"blob {len(raw)}\0".encode("ascii")
        return hashlib.sha1(prefix + raw, usedforsecurity=False).hexdigest()

    repository_receipt: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent-supervisor/"
            "provider-effect-repository-authority@1"
        ),
        "accepted_head": current_head,
        "accepted_tree": current_tree,
        "authorization_path": authorization.artifact_path,
        "authorization_blob_id": git_blob_id(artifact_raw),
        "witness_path": authorization.reviewer_witness_path,
        "witness_blob_id": git_blob_id(witness_raw),
        "root_pin_path": authorization.lifecycle_root_pin_path,
        "root_pin_blob_id": git_blob_id(root_pin_raw),
        "authorization_commit_time_ms": authorization_commit_time_ms,
    }
    repository_receipt["receipt_id"] = _content_addressed_mapping(
        repository_receipt,
        identity_field="receipt_id",
    )
    evidence = (
        quota_evidence.audit_dict()
        if (
            isinstance(quota_evidence, AgentImplementationQuotaEvidence)
            and quota_evidence._signer_process_validated
        )
        else {}
    )
    body: dict[str, object] = {
        "schema": _AGENT_EFFECT_AUTHORIZATION_CONTEXT_SCHEMA,
        "route_binding": route.as_binding_dict(),
        "failure_receipt": dict(failure_receipt),
        "quota_evidence": evidence,
        "expected_nonce": str(expected_nonce),
        "expected_model": str(expected_model),
        "expected_probe_returncode": expected_probe_returncode,
        "decision": decision.as_dict(),
        "decision_id": decision.content_id,
        "invocation_binding_id": invocation.content_id,
        "logical_attempt_id": invocation.logical_attempt_id,
        "authorization_artifact_b64": base64.b64encode(artifact_raw).decode(
            "ascii"
        ),
        "lifecycle_root_pin_b64": base64.b64encode(root_pin_raw).decode(
            "ascii"
        ),
        "lifecycle_witness_b64": base64.b64encode(witness_raw).decode(
            "ascii"
        ),
        "repository_receipt": repository_receipt,
    }
    body["context_id"] = _content_addressed_mapping(
        body,
        identity_field="context_id",
    )
    if len(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ) > 512 * 1024:
        raise ValueError("effect authorization context is oversized")
    return body


AGENT_IMPLEMENTATION_ROUTE_OUTCOME_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.protected-route-outcome@1"
)
AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX = (
    "AGENT_IMPLEMENTATION_PROTECTED_ROUTE_OUTCOME_JSON:"
)
# The bounded eight-generation adoption lineage is intentionally complete,
# not merely a latest-receipt pointer.  Its canonical JSON is roughly 225KiB
# at the cap, so retain a fixed ceiling that covers the exact terminal chain
# while still rejecting unbounded/attacker-grown log records.
_AGENT_IMPLEMENTATION_ROUTE_OUTCOME_MAX_BYTES = 512 * 1024


def _agent_effect_detail_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _agent_effect_launch_details_valid(
    value: Mapping[str, object],
    *,
    workspace_path: str = "",
) -> bool:
    runtime = value.get("runtime_receipt")
    image = value.get("image_receipt")
    command = value.get("command_receipt")
    mounts = value.get("mount_receipt")
    environment = value.get("environment_receipt")
    cleanup = value.get("cleanup_receipt")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "path",
            "device",
            "inode",
            "mode",
            "uid",
            "size",
            "mtime_ns",
            "ctime_ns",
        }
        or runtime.get("path") not in {"/usr/bin/docker", "/usr/local/bin/docker"}
        or any(
            isinstance(runtime.get(name), bool)
            or not isinstance(runtime.get(name), int)
            or int(runtime.get(name) or 0) < 0
            for name in (
                "device",
                "inode",
                "mode",
                "uid",
                "size",
                "mtime_ns",
                "ctime_ns",
            )
        )
        or runtime.get("uid") != 0
        or not isinstance(image, Mapping)
        or set(image) != {"image_id", "image_label"}
        or image.get("image_id") != value.get("image_id")
        or image.get("image_label") != "2026-08-03-v2"
        or not isinstance(command, Mapping)
        or set(command) != {"create_argv", "start_argv", "provider_argv"}
        or not isinstance(mounts, list)
        or not mounts
        or any(not isinstance(item, str) or not item for item in mounts)
        or not isinstance(environment, Mapping)
        or set(environment) != {"docker_cli", "container"}
        or not isinstance(cleanup, Mapping)
        or set(cleanup)
        != {
            "schema",
            "lease_root",
            "docker_config",
            "cidfile",
            "provider_home",
            "prompt_path",
            "watchdog_pid",
            "watchdog_start_ticks",
            "receipt_id",
        }
    ):
        return False
    argv_values: dict[str, list[str]] = {}
    for name in ("create_argv", "start_argv", "provider_argv"):
        argv = command.get(name)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) for item in argv)
        ):
            return False
        argv_values[name] = argv
    docker_cli = environment.get("docker_cli")
    container = environment.get("container")
    if (
        not isinstance(docker_cli, Mapping)
        or not isinstance(container, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for mapping in (docker_cli, container)
            for key, item in mapping.items()
        )
    ):
        return False
    expected_container = {
        "BASH_ENV": "",
        "CODEX_HOME": "/opt/codex-home",
        "ENV": "",
        "HOME": "/opt/codex-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/opt/ipfs-task-tools/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "/opt/ipfs-validation-site-packages",
        "TERM": "dumb",
    }
    create_argv = argv_values["create_argv"]
    start_argv = argv_values["start_argv"]
    provider_argv = argv_values["provider_argv"]
    if len(provider_argv) != 14:
        return False
    provider_workspace = provider_argv[8]
    expected_provider_tail = [
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        provider_workspace,
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]
    if (
        not Path(provider_argv[0]).is_absolute()
        or provider_argv[1:] != expected_provider_tail
        or (workspace_path and provider_workspace != workspace_path)
    ):
        return False
    raw_container_id = str(value.get("container_id") or "").removeprefix(
        "sha256:"
    )
    container_name = str(value.get("container_name") or "")
    if len(create_argv) < 30 or len(start_argv) != 8:
        return False
    docker_path = str(runtime.get("path") or "")
    config_path = Path(create_argv[3])
    cidfile_path = Path(create_argv[20])
    user_identity = create_argv[27]
    if re.fullmatch(r"[0-9]+:[0-9]+", user_identity) is None:
        return False
    uid_text, gid_text = user_identity.split(":", 1)
    expected_prefix = [
        docker_path,
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(config_path),
        "create",
        "--pull=never",
        "--interactive",
        "--read-only",
        "--network=bridge",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--tmpfs",
        f"/var/tmp:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--tmpfs",
        f"/opt/codex-home:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--name",
        container_name,
        "--cidfile",
        str(cidfile_path),
        "--label",
        "ipfs_accelerate.codex_fallback_isolation=true",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
        "--user",
        user_identity,
        "--workdir",
        provider_workspace,
    ]
    overrides = [
        "BASH_ENV=",
        "CUDA_VISIBLE_DEVICES=-1",
        "ENV=",
        "LD_LIBRARY_PATH=",
        "LD_PRELOAD=",
        "LIBRARY_PATH=",
        "NVIDIA_DRIVER_CAPABILITIES=",
        "NVIDIA_REQUIRE_CUDA=",
        "NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS=",
        "NVIDIA_VISIBLE_DEVICES=void",
    ]
    cursor = len(expected_prefix)
    expected_environment_flags: list[str] = []
    for override in overrides:
        expected_environment_flags.extend(["--env", override])
    if (
        create_argv[:cursor] != expected_prefix
        or create_argv[cursor : cursor + len(expected_environment_flags)]
        != expected_environment_flags
        or not config_path.is_absolute()
        or config_path.name != "docker-config"
        or not config_path.parent.name.startswith("asref-codex-container-")
        or cidfile_path != config_path.parent / "container.cid"
    ):
        return False
    lease_root = config_path.parent
    provider_home = Path(str(cleanup.get("provider_home") or ""))
    prompt_path = Path(str(cleanup.get("prompt_path") or ""))
    watchdog_pid = cleanup.get("watchdog_pid")
    watchdog_start_ticks = cleanup.get("watchdog_start_ticks")
    cleanup_body = {
        key: item for key, item in cleanup.items() if key != "receipt_id"
    }
    if (
        cleanup.get("schema")
        != "ipfs_accelerate_py.agent_supervisor.provider-effect-cleanup@1"
        or cleanup.get("lease_root") != str(lease_root)
        or cleanup.get("docker_config") != str(config_path)
        or cleanup.get("cidfile") != str(cidfile_path)
        or lease_root.parent != Path(tempfile.gettempdir()).resolve()
        or not provider_home.is_absolute()
        or provider_home.parent != lease_root.parent
        or not provider_home.name.startswith("asref-codex-home-")
        or not prompt_path.is_absolute()
        or prompt_path.parent != lease_root.parent
        or not prompt_path.name.startswith("asref-grok-prompt-")
        or (workspace_path and provider_home.is_relative_to(Path(workspace_path)))
        or (workspace_path and prompt_path.is_relative_to(Path(workspace_path)))
        or isinstance(watchdog_pid, bool)
        or not isinstance(watchdog_pid, int)
        or watchdog_pid <= 0
        or isinstance(watchdog_start_ticks, bool)
        or not isinstance(watchdog_start_ticks, int)
        or watchdog_start_ticks < 0
        or cleanup.get("receipt_id") != _agent_effect_detail_id(cleanup_body)
        or value.get("cleanup_id") != cleanup.get("receipt_id")
    ):
        return False
    cursor += len(expected_environment_flags)
    parsed_mounts: list[str] = []
    while cursor < len(create_argv) and create_argv[cursor] == "--mount":
        if cursor + 1 >= len(create_argv):
            return False
        parsed_mounts.append(create_argv[cursor + 1])
        cursor += 2
    expected_assignments = [
        f"{name}={item}" for name, item in sorted(expected_container.items())
    ]
    inner = list(provider_argv)
    inner[6] = "danger-full-access"
    expected_suffix = [
        AGENT_IMPLEMENTATION_CODEX_IMAGE_ID,
        "-i",
        *expected_assignments,
        *inner,
    ]
    if (
        create_argv[cursor:] != expected_suffix
        or parsed_mounts != mounts
        or len(parsed_mounts) < 5
        or len(set(parsed_mounts)) != len(parsed_mounts)
    ):
        return False
    writable_mounts: list[tuple[str, str]] = []
    read_only_destinations: set[str] = set()
    for mount in parsed_mounts:
        fields = mount.split(",")
        if len(fields) not in {3, 4} or fields[0] != "type=bind":
            return False
        if not fields[1].startswith("src=") or not fields[2].startswith("dst="):
            return False
        source = fields[1].removeprefix("src=")
        destination = fields[2].removeprefix("dst=")
        if not Path(source).is_absolute() or not Path(destination).is_absolute():
            return False
        if len(fields) == 4:
            if fields[3] != "readonly":
                return False
            read_only_destinations.add(destination)
        else:
            writable_mounts.append((source, destination))
    if writable_mounts != [(provider_workspace, provider_workspace)]:
        return False
    required_read_only_sources = {
        "/usr": "/usr",
        "/etc/ssl/certs": "/etc/ssl/certs",
        "/opt/ipfs-task-tools/bin/python": "/usr/bin/python3.12",
    }
    if not set(required_read_only_sources).issubset(read_only_destinations):
        return False
    observed_sources: dict[str, str] = {}
    for mount in parsed_mounts:
        fields = mount.split(",")
        source = fields[1].removeprefix("src=")
        destination = fields[2].removeprefix("dst=")
        if destination in observed_sources:
            return False
        observed_sources[destination] = source
        if destination in required_read_only_sources:
            if source != required_read_only_sources[destination]:
                return False
            continue
        if destination == "/opt/codex-home/auth.json":
            # The credential source is account-local and varies by operator,
            # but it must remain the exact private auth.json selected by the
            # pre-effect loader.  It may never be substituted for a host
            # toolchain/root mount or be sourced from the candidate.
            source_path = Path(source)
            if (
                source_path.name != "auth.json"
                or source_path.is_relative_to(Path(provider_workspace))
            ):
                return False
            continue
        if destination == provider_workspace:
            continue
        if source != destination or not (
            destination.endswith("/.git")
            or "/.git/worktrees/" in destination
            or destination.endswith("/worktrees")
        ):
            return False
    if "/opt/codex-home/auth.json" not in observed_sources:
        return False
    expected_start = [
        docker_path,
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(config_path),
        "start",
        "--attach",
        "--interactive",
        raw_container_id,
    ]
    return bool(
        dict(container) == expected_container
        and dict(docker_cli) == expected_container
        and value.get("image_id") == AGENT_IMPLEMENTATION_CODEX_IMAGE_ID
        and image.get("image_label") == AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL
        and start_argv == expected_start
        and len(raw_container_id) == 64
        and value.get("runtime_id") == _agent_effect_detail_id(runtime)
        and value.get("command_id") == _agent_effect_detail_id(command)
        and value.get("mount_id") == _agent_effect_detail_id(mounts)
        and value.get("environment_id") == _agent_effect_detail_id(environment)
        and value.get("cleanup_id") == _agent_effect_detail_id(cleanup_body)
    )


def _agent_effect_receipt_valid(
    value: Mapping[str, object],
    *,
    logical_attempt_id: str,
    reservation_id: str,
    workspace_path: str = "",
) -> bool:
    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "cleanup_id",
        "container_name",
        "container_id",
        "claimed_at_ms",
        "runtime_receipt",
        "image_receipt",
        "command_receipt",
        "mount_receipt",
        "environment_receipt",
        "cleanup_receipt",
        "receipt_id",
    }
    integer_fields = (
        value.get("effect_owner_pid"),
        value.get("effect_owner_start_ticks"),
        value.get("claimed_at_ms"),
    )
    return bool(
        set(value) == expected
        and value.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/provider-effect-launch@2"
        and value.get("logical_attempt_id") == logical_attempt_id
        and value.get("reservation_id") == reservation_id
        and value.get("provider_id") == "codex"
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in integer_fields
        )
        and int(value.get("effect_owner_pid") or 0) > 0
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("effect_owner_id") or ""),
        )
        is not None
        and all(
            re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(value.get(name) or ""),
            )
            is not None
            for name in (
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
            )
        )
        and isinstance(value.get("container_name"), str)
        and bool(str(value.get("container_name") or "").strip())
        and not any(
            marker in str(value.get("container_name") or "")
            for marker in ("\0", "\n", "\r")
        )
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("container_id") or ""),
        )
        is not None
        and value.get("receipt_id")
        == "sha256:"
        + hashlib.sha256(
            json.dumps(
                {key: item for key, item in value.items() if key != "receipt_id"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        and _agent_effect_launch_details_valid(
            value,
            workspace_path=workspace_path,
        )
    )


def _agent_effect_adoption_receipt_valid(
    value: Mapping[str, object],
    *,
    launch: Mapping[str, object],
    logical_attempt_id: str,
    reservation_id: str,
) -> bool:
    """Validate the exact dead-owner/container inspection transition."""

    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "adoption_generation",
        "previous_receipt_id",
        "previous_owner_id",
        "previous_owner_pid",
        "previous_owner_start_ticks",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "transition_kind",
        "inspection_status",
        "inspection_runtime_id",
        "inspection_command_id",
        "inspection_observed_at_ms",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "container_name",
        "container_id",
        "container_returncode",
        "inspected_at_ms",
        "prior_adoption_receipts",
        "receipt_id",
    }
    if set(value) != expected:
        return False
    integer_names = (
        "adoption_generation",
        "previous_owner_pid",
        "previous_owner_start_ticks",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "inspection_observed_at_ms",
        "inspected_at_ms",
    )
    if any(
        isinstance(value.get(name), bool)
        or not isinstance(value.get(name), int)
        or int(value.get(name) or 0) < 0
        for name in integer_names
    ):
        return False
    status_value = value.get("inspection_status")
    returncode = value.get("container_returncode")
    container_id = value.get("container_id")
    generation = int(value.get("adoption_generation") or 0)
    prior = value.get("prior_adoption_receipts")
    if (
        not isinstance(prior, list)
        or not 1 <= generation <= 8
        or len(prior) != generation - 1
    ):
        return False
    if generation > 1:
        prior_owner = prior[-1]
        if (
            not isinstance(prior_owner, Mapping)
            or not _agent_effect_adoption_receipt_valid(
                prior_owner,
                launch=launch,
                logical_attempt_id=logical_attempt_id,
                reservation_id=reservation_id,
            )
            or prior[:-1] != prior_owner.get("prior_adoption_receipts")
        ):
            return False
    else:
        prior_owner = launch
    claimed_at = launch.get("claimed_at_ms")
    return bool(
        value.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/provider-effect-adoption@1"
        and value.get("logical_attempt_id") == logical_attempt_id
        and value.get("reservation_id") == reservation_id
        and value.get("adoption_generation") == generation
        and int(value.get("effect_owner_pid") or 0) > 0
        and value.get("transition_kind")
        in {"dead_owner_adoption", "winner_reconciliation"}
        and (
            (
                value.get("transition_kind") == "winner_reconciliation"
                and value.get("previous_owner_id")
                == value.get("effect_owner_id")
                and value.get("previous_owner_pid")
                == value.get("effect_owner_pid")
                and value.get("previous_owner_start_ticks")
                == value.get("effect_owner_start_ticks")
            )
            or (
                value.get("transition_kind") == "dead_owner_adoption"
                and value.get("previous_owner_id")
                != value.get("effect_owner_id")
            )
        )
        and status_value in {"created", "running", "exited", "absent"}
        and value.get("inspection_runtime_id") == launch.get("runtime_id")
        and value.get("inspection_observed_at_ms")
        == value.get("inspected_at_ms")
        and isinstance(claimed_at, int)
        and not isinstance(claimed_at, bool)
        and int(value.get("inspected_at_ms") or 0) >= claimed_at
        and (
            (
                status_value == "exited"
                and isinstance(returncode, int)
                and not isinstance(returncode, bool)
            )
            or (status_value != "exited" and returncode is None)
        )
        and (
            (
                status_value in {"created", "running", "exited"}
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}", str(container_id or "")
                )
                is not None
            )
            or (status_value == "absent" and container_id == "")
        )
        and (
            status_value == "absent"
            or container_id == launch.get("container_id")
        )
        and all(
            re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(value.get(name) or "")
            )
            is not None
            for name in (
                "previous_receipt_id",
                "previous_owner_id",
                "effect_owner_id",
                "inspection_command_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
            )
        )
        and value.get("provider_id") == launch.get("provider_id") == "codex"
        and all(
            value.get(name) == launch.get(name)
            for name in (
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
            )
        )
        and (
            value.get("previous_receipt_id") == prior_owner.get("receipt_id")
            and value.get("previous_owner_id")
            == prior_owner.get("effect_owner_id")
            and value.get("previous_owner_pid")
            == prior_owner.get("effect_owner_pid")
            and value.get("previous_owner_start_ticks")
            == prior_owner.get("effect_owner_start_ticks")
        )
        and value.get("receipt_id")
        == "sha256:"
        + hashlib.sha256(
            json.dumps(
                {key: item for key, item in value.items() if key != "receipt_id"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def _agent_effect_quarantine_receipt_valid(
    value: Mapping[str, object],
    *,
    launch: Mapping[str, object],
    adoption: Mapping[str, object],
    logical_attempt_id: str,
    reservation_id: str,
) -> bool:
    """Validate the bounded-adoption incident for one exact effect."""

    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "adoption_generation",
        "reason",
        "required_operator_action",
        "provider_id",
        "runtime_id",
        "container_name",
        "container_id",
        "inspection_status",
        "inspection_command_id",
        "container_returncode",
        "quarantined_at_ms",
        "incident_id",
    }
    status_value = value.get("inspection_status")
    returncode = value.get("container_returncode")
    quarantined_at_ms = value.get("quarantined_at_ms")
    return bool(
        set(value) == expected
        and value.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/provider-effect-quarantine@1"
        and value.get("logical_attempt_id") == logical_attempt_id
        and value.get("reservation_id") == reservation_id
        and value.get("adoption_generation")
        == adoption.get("adoption_generation")
        == 8
        and value.get("reason") == "adoption_transfer_limit_exhausted"
        and value.get("required_operator_action")
        == "inspect_exact_container_and_terminalize_without_relaunch"
        and value.get("provider_id") == launch.get("provider_id") == "codex"
        and value.get("runtime_id") == launch.get("runtime_id")
        and value.get("container_name") == launch.get("container_name")
        and status_value in {"created", "running", "exited", "absent"}
        and (
            (
                status_value == "absent"
                and value.get("container_id") == ""
                and returncode is None
            )
            or (
                status_value in {"created", "running"}
                and value.get("container_id") == launch.get("container_id")
                and returncode is None
            )
            or (
                status_value == "exited"
                and value.get("container_id") == launch.get("container_id")
                and isinstance(returncode, int)
                and not isinstance(returncode, bool)
            )
        )
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("inspection_command_id") or ""),
        )
        is not None
        and isinstance(quarantined_at_ms, int)
        and not isinstance(quarantined_at_ms, bool)
        and quarantined_at_ms >= int(adoption.get("inspected_at_ms") or 0)
        and value.get("incident_id")
        == _agent_effect_detail_id(
            {key: item for key, item in value.items() if key != "incident_id"}
        )
    )


def _agent_effect_quarantine_terminalization_receipt_valid(
    value: Mapping[str, object],
    *,
    launch: Mapping[str, object],
    quarantine: Mapping[str, object],
    logical_attempt_id: str,
    reservation_id: str,
) -> bool:
    """Validate the no-relaunch exact-container repair receipt and lineage."""

    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "incident_id",
        "repair_generation",
        "previous_repair_receipt_id",
        "prior_repair_receipts",
        "operator_action",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "container_name",
        "container_id",
        "inspection_status",
        "inspection_command_id",
        "inspected_at_ms",
        "container_returncode",
        "terminal_returncode",
        "outcome_decision",
        "fallback_dispatched",
        "receipt_id",
    }
    if set(value) != expected:
        return False
    generation = value.get("repair_generation")
    prior = value.get("prior_repair_receipts")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not isinstance(prior, list)
        or len(prior) != generation - 1
    ):
        return False
    if generation == 1:
        if value.get("previous_repair_receipt_id") != "" or prior:
            return False
    else:
        previous = prior[-1]
        if (
            not isinstance(previous, Mapping)
            or value.get("previous_repair_receipt_id")
            != previous.get("receipt_id")
            or prior[:-1] != previous.get("prior_repair_receipts")
            or not _agent_effect_quarantine_terminalization_receipt_valid(
                previous,
                launch=launch,
                quarantine=quarantine,
                logical_attempt_id=logical_attempt_id,
                reservation_id=reservation_id,
            )
        ):
            return False
    status_value = value.get("inspection_status")
    container_returncode = value.get("container_returncode")
    terminal_returncode = value.get("terminal_returncode")
    inspected_at_ms = value.get("inspected_at_ms")
    owner_pid = value.get("effect_owner_pid")
    owner_start = value.get("effect_owner_start_ticks")
    expected_decision = (
        "effect_not_created"
        if status_value == "absent"
        else (
            "fallback_succeeded"
            if container_returncode == 0
            else "fallback_failed"
        )
    )
    return bool(
        value.get("schema")
        == (
            "ipfs_accelerate_py/agent-supervisor/"
            "provider-effect-quarantine-terminalization@1"
        )
        and value.get("logical_attempt_id") == logical_attempt_id
        and value.get("reservation_id") == reservation_id
        and value.get("incident_id") == quarantine.get("incident_id")
        and value.get("operator_action")
        == "terminalize_exact_quarantined_effect_without_relaunch"
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("effect_owner_id") or ""),
        )
        is not None
        and isinstance(owner_pid, int)
        and not isinstance(owner_pid, bool)
        and owner_pid > 0
        and isinstance(owner_start, int)
        and not isinstance(owner_start, bool)
        and owner_start >= 0
        and all(
            value.get(name) == launch.get(name)
            for name in (
                "provider_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
            )
        )
        and status_value in {"absent", "exited"}
        and isinstance(inspected_at_ms, int)
        and not isinstance(inspected_at_ms, bool)
        and inspected_at_ms >= int(quarantine.get("quarantined_at_ms") or 0)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("inspection_command_id") or ""),
        )
        is not None
        and (
            (
                status_value == "absent"
                and value.get("container_id") == ""
                and container_returncode is None
                and terminal_returncode == 125
                and value.get("fallback_dispatched") is False
            )
            or (
                status_value == "exited"
                and value.get("container_id") == launch.get("container_id")
                and isinstance(container_returncode, int)
                and not isinstance(container_returncode, bool)
                and terminal_returncode == container_returncode
                and value.get("fallback_dispatched") is True
            )
        )
        and value.get("outcome_decision") == expected_decision
        and value.get("receipt_id")
        == _agent_effect_detail_id(
            {key: item for key, item in value.items() if key != "receipt_id"}
        )
    )


def build_agent_implementation_route_outcome(
    *,
    receipt: Mapping[str, object],
    route: AgentImplementationRoutePlan,
    decision: str,
    verifier_status: str,
    fallback_dispatched: bool,
    fallback_returncode: int | None,
    decision_id: str,
    quota_evidence: AgentImplementationQuotaEvidence | None = None,
    reservation_id: str = "",
    effect_launch_receipt: Mapping[str, object] | None = None,
    effect_adoption_receipt: Mapping[str, object] | None = None,
    effect_quarantine_receipt: Mapping[str, object] | None = None,
    effect_quarantine_terminalization_receipt: Mapping[
        str, object
    ] | None = None,
) -> dict[str, object]:
    """Build the sole protected v3 terminal record owned by the router."""

    invocation = route.invocation_binding
    if invocation is None:
        raise ValueError("protected route outcome requires an invocation")
    quota_audit = (
        quota_evidence.audit_dict()
        if (
            isinstance(quota_evidence, AgentImplementationQuotaEvidence)
            and quota_evidence._signer_process_validated
        )
        else {}
    )
    outcome: dict[str, object] = {
        "schema": AGENT_IMPLEMENTATION_ROUTE_OUTCOME_SCHEMA,
        "source": "grok_cli_runner",
        "preflight_receipt_id": str(receipt.get("receipt_id") or ""),
        "nonce": str(receipt.get("nonce") or ""),
        "route_plan": route.as_outcome_dict(),
        "invocation_binding_id": invocation.content_id,
        "control_plane_id": invocation.control_plane.capsule_id,
        "control_plane_source_head": invocation.control_plane.source_head,
        "control_plane_source_tree": invocation.control_plane.source_tree,
        "decision": str(decision),
        "decision_id": str(decision_id),
        "verifier_status": str(verifier_status),
        "quota_evidence_id": str(quota_audit.get("evidence_id") or ""),
        "quota_evidence": quota_audit,
        "fallback_dispatched": fallback_dispatched,
        "fallback_returncode": fallback_returncode,
        "reservation_id": str(reservation_id or ""),
        "effect_launch_receipt": dict(effect_launch_receipt or {}),
        "effect_adoption_receipt": dict(effect_adoption_receipt or {}),
        "effect_quarantine_receipt": dict(effect_quarantine_receipt or {}),
        "effect_quarantine_terminalization_receipt": dict(
            effect_quarantine_terminalization_receipt or {}
        ),
    }
    outcome["outcome_id"] = _content_addressed_mapping(
        outcome,
        identity_field="outcome_id",
    )
    if not valid_agent_implementation_route_outcome(
        outcome,
        receipt=receipt,
        route=route,
        runner_returncode=(
            int(fallback_returncode)
            if (
                (fallback_dispatched or decision == "effect_not_created")
                and isinstance(fallback_returncode, int)
                and not isinstance(fallback_returncode, bool)
            )
            else int(receipt.get("probe_returncode") or 1)
        ),
    ):
        raise ValueError("protected route outcome fields are invalid")
    return outcome


def valid_agent_implementation_route_outcome(
    outcome: Mapping[str, object],
    *,
    receipt: Mapping[str, object],
    route: AgentImplementationRoutePlan,
    runner_returncode: int,
) -> bool:
    """Validate exact protected route, capsule, CAS, and runtime equality."""

    expected = {
        "schema",
        "source",
        "preflight_receipt_id",
        "nonce",
        "route_plan",
        "invocation_binding_id",
        "control_plane_id",
        "control_plane_source_head",
        "control_plane_source_tree",
        "decision",
        "decision_id",
        "verifier_status",
        "quota_evidence_id",
        "quota_evidence",
        "fallback_dispatched",
        "fallback_returncode",
        "reservation_id",
        "effect_launch_receipt",
        "effect_adoption_receipt",
        "effect_quarantine_receipt",
        "effect_quarantine_terminalization_receipt",
        "outcome_id",
    }
    invocation = route.invocation_binding
    fallback_returncode = outcome.get("fallback_returncode")
    launch = outcome.get("effect_launch_receipt")
    adoption = outcome.get("effect_adoption_receipt")
    quarantine = outcome.get("effect_quarantine_receipt")
    quarantine_terminalization = outcome.get(
        "effect_quarantine_terminalization_receipt"
    )
    quota_evidence = outcome.get("quota_evidence")
    quota_required = bool(
        receipt.get("failure_class")
        in {"hard_quota_exhausted", "authentication"}
        and outcome.get("verifier_status") == "confirmed_quota"
    )
    common = bool(
        invocation is not None
        and set(outcome) == expected
        and outcome.get("schema") == AGENT_IMPLEMENTATION_ROUTE_OUTCOME_SCHEMA
        and outcome.get("source") == "grok_cli_runner"
        and outcome.get("preflight_receipt_id") == receipt.get("receipt_id")
        and outcome.get("nonce") == receipt.get("nonce")
        and outcome.get("route_plan") == route.as_outcome_dict()
        and outcome.get("invocation_binding_id") == invocation.content_id
        and outcome.get("control_plane_id")
        == invocation.control_plane.capsule_id
        and outcome.get("control_plane_source_head")
        == invocation.control_plane.source_head
        and outcome.get("control_plane_source_tree")
        == invocation.control_plane.source_tree
        and outcome.get("verifier_status")
        in {
            "not_run",
            "not_required_exact_auth",
            "native_signature_unavailable",
            "not_confirmed",
            "confirmed_quota",
        }
        and isinstance(quota_evidence, Mapping)
        and (
            (
                quota_required
                and bool(quota_evidence)
                and outcome.get("quota_evidence_id")
                == quota_evidence.get("evidence_id")
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(outcome.get("quota_evidence_id") or ""),
                )
                is not None
            )
            or (
                not quota_required
                and not quota_evidence
                and outcome.get("quota_evidence_id") == ""
            )
        )
        and isinstance(outcome.get("fallback_dispatched"), bool)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(outcome.get("decision_id") or ""),
        )
        is not None
        and isinstance(launch, Mapping)
        and isinstance(adoption, Mapping)
        and isinstance(quarantine, Mapping)
        and isinstance(quarantine_terminalization, Mapping)
        and outcome.get("outcome_id")
        == _content_addressed_mapping(outcome, identity_field="outcome_id")
    )
    if not common:
        return False
    if quota_required:
        claimed_at_ms = (
            launch.get("claimed_at_ms") if isinstance(launch, Mapping) else None
        )
        expected_parent_pid = (
            launch.get("effect_owner_pid")
            if isinstance(launch, Mapping)
            else quota_evidence.get("signer_parent_pid")
        )
        if (
            isinstance(expected_parent_pid, bool)
            or not isinstance(expected_parent_pid, int)
            or expected_parent_pid <= 0
            or isinstance(claimed_at_ms, bool)
            or not isinstance(claimed_at_ms, int)
            or claimed_at_ms <= 0
            or parse_agent_implementation_quota_evidence(
                quota_evidence,
                failure_receipt=receipt,
                invocation_binding=invocation,
                now_ms=claimed_at_ms,
                max_age_ms=_AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS,
                expected_signer_parent_pid=expected_parent_pid,
                require_current_lifecycle=False,
            )
            is None
        ):
            return False
    if outcome.get("decision") == "denied":
        return bool(
            outcome.get("fallback_dispatched") is False
            and fallback_returncode is None
            and not outcome.get("reservation_id")
            and not launch
            and not adoption
            and not quarantine
            and not quarantine_terminalization
            and runner_returncode == receipt.get("probe_returncode")
        )
    if outcome.get("decision") not in {
        "fallback_succeeded",
        "fallback_failed",
        "effect_not_created",
    }:
        return False
    reservation_id = str(outcome.get("reservation_id") or "")
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", reservation_id) is None
        or not _agent_effect_receipt_valid(
            launch,
            logical_attempt_id=invocation.logical_attempt_id,
            reservation_id=reservation_id,
            workspace_path=invocation.workspace_path,
        )
    ):
        return False
    adoption_valid = bool(
        adoption
        and _agent_effect_adoption_receipt_valid(
            adoption,
            launch=launch,
            logical_attempt_id=invocation.logical_attempt_id,
            reservation_id=reservation_id,
        )
    )
    quarantine_valid = bool(
        quarantine
        and adoption_valid
        and _agent_effect_quarantine_receipt_valid(
            quarantine,
            launch=launch,
            adoption=adoption,
            logical_attempt_id=invocation.logical_attempt_id,
            reservation_id=reservation_id,
        )
    )
    quarantine_terminalization_valid = bool(
        quarantine_terminalization
        and quarantine_valid
        and _agent_effect_quarantine_terminalization_receipt_valid(
            quarantine_terminalization,
            launch=launch,
            quarantine=quarantine,
            logical_attempt_id=invocation.logical_attempt_id,
            reservation_id=reservation_id,
        )
    )
    if bool(quarantine) != bool(quarantine_terminalization):
        return False
    if quarantine and not quarantine_terminalization_valid:
        return False
    if outcome.get("decision") == "effect_not_created":
        return bool(
            outcome.get("fallback_dispatched") is False
            and isinstance(fallback_returncode, int)
            and not isinstance(fallback_returncode, bool)
            and fallback_returncode == runner_returncode == 125
            and (
                (
                    not quarantine
                    and adoption_valid
                    and adoption.get("inspection_status") == "absent"
                    and adoption.get("container_returncode") is None
                )
                or (
                    quarantine_terminalization_valid
                    and quarantine_terminalization.get("inspection_status")
                    == "absent"
                    and quarantine_terminalization.get("terminal_returncode")
                    == 125
                    and quarantine_terminalization.get(
                        "fallback_dispatched"
                    )
                    is False
                )
            )
        )
    return bool(
        outcome.get("fallback_dispatched") is True
        and isinstance(fallback_returncode, int)
        and not isinstance(fallback_returncode, bool)
        and fallback_returncode == runner_returncode
        and (
            (outcome.get("decision") == "fallback_succeeded" and runner_returncode == 0)
            or (outcome.get("decision") == "fallback_failed" and runner_returncode != 0)
        )
        and (
            (
                quarantine_terminalization_valid
                and quarantine_terminalization.get("inspection_status")
                == "exited"
                and quarantine_terminalization.get("terminal_returncode")
                == fallback_returncode
                and quarantine_terminalization.get("outcome_decision")
                == outcome.get("decision")
                and quarantine_terminalization.get("fallback_dispatched")
                is True
            )
            or (
                not quarantine
                and (
                    not adoption
                    or (
                        adoption_valid
                        and adoption.get("inspection_status")
                        in {"created", "running", "exited"}
                        and (
                            adoption.get("inspection_status") != "exited"
                            or adoption.get("container_returncode")
                            == fallback_returncode
                        )
                    )
                )
            )
        )
    )


def render_agent_implementation_route_outcome(
    outcome: Mapping[str, object],
) -> str:
    return AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX + json.dumps(
        dict(outcome),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def extract_agent_implementation_route_outcomes(
    text: str,
) -> tuple[dict[str, object], ...]:
    outcomes: list[dict[str, object]] = []
    for line in str(text).split("\n"):
        if not line.startswith(AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX):
            continue
        raw = line[len(AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX) :]
        if "\r" in raw or len(raw.encode("utf-8")) > (
            _AGENT_IMPLEMENTATION_ROUTE_OUTCOME_MAX_BYTES
        ):
            continue

        def unique(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate protected outcome key")
                result[key] = value
            return result

        try:
            value = json.loads(raw, object_pairs_hook=unique)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            outcomes.append(value)
    return tuple(outcomes[-4:])


@dataclass(frozen=True, slots=True)
class AgentImplementationRouteInvocation:
    """One process-local route plan and its fresh failure-receipt nonce."""

    route_plan: AgentImplementationRoutePlan
    failure_receipt_nonce: str


_AGENT_IMPLEMENTATION_MAX_SESSION_BYTES = 16 * 1024 * 1024
_AGENT_IMPLEMENTATION_MAX_STREAM_EVENT_BYTES = 64 * 1024
_AGENT_IMPLEMENTATION_BALANCE_EXHAUSTED_MESSAGE = (
    "API error (status 402 Payment Required): Grok Build usage balance exhausted"
)
_AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)
_AGENT_IMPLEMENTATION_NATIVE_QUOTA_FAILURES = frozenset(
    {
        (
            "usage_pool_exhausted",
            _AGENT_IMPLEMENTATION_BALANCE_EXHAUSTED_MESSAGE,
        ),
        (
            "spending_limit_exhausted",
            _AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE,
        ),
    }
)
_AGENT_IMPLEMENTATION_QUOTA_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py.agent-supervisor/native-quota-evidence@2"
)
AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS = (
    "run_terminal_cmd,run_terminal_command,web_search,web_fetch,search_tool,"
    "use_tool,call_mcp_tool,list_mcp_resources,list_mcp_resource_templates,"
    "read_mcp_resource,fetch_mcp_resource,task,Agent,memory,lsp,spawn_subagent"
)


@dataclass(frozen=True, slots=True)
class AgentImplementationQuotaEvidence:
    """Router-validated native session, optionally signed for protected use."""

    schema: str
    preflight_receipt_id: str
    preflight_nonce: str
    primary_provider: str
    primary_model: str
    verifier_session_id: str
    verifier_provider: str
    verifier_model: str
    verifier_command: tuple[str, ...]
    verifier_command_id: str
    verifier_returncode: int
    verifier_result: str
    probe_contract_id: str
    transcript_sha256: str
    summary_sha256: str
    invocation_id: str
    logical_attempt_id: str
    route_id: str
    signer_identity_did: str
    signer_key_id: str
    signer_profile_id: str
    signer_profile_content_id: str
    signer_lifecycle_generation: int
    signer_lifecycle_anchor_id: str
    signer_provider: str
    signer_process_pid: int
    signer_parent_pid: int
    observed_at_ms: int
    issued_at_ms: int
    expires_at_ms: int
    signer_signature: str
    evidence_id: str
    _validation_seal: str = field(repr=False, compare=False)
    _signer_process_validated: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    def signing_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "preflight_receipt_id": self.preflight_receipt_id,
            "preflight_nonce": self.preflight_nonce,
            "primary_provider": self.primary_provider,
            "primary_model": self.primary_model,
            "verifier_session_id": self.verifier_session_id,
            "verifier_provider": self.verifier_provider,
            "verifier_model": self.verifier_model,
            "verifier_command": list(self.verifier_command),
            "verifier_command_id": self.verifier_command_id,
            "verifier_returncode": self.verifier_returncode,
            "verifier_result": self.verifier_result,
            "probe_contract_id": self.probe_contract_id,
            "transcript_sha256": self.transcript_sha256,
            "summary_sha256": self.summary_sha256,
            "invocation_id": self.invocation_id,
            "logical_attempt_id": self.logical_attempt_id,
            "route_id": self.route_id,
            "signer_identity_did": self.signer_identity_did,
            "signer_key_id": self.signer_key_id,
            "signer_profile_id": self.signer_profile_id,
            "signer_profile_content_id": self.signer_profile_content_id,
            "signer_lifecycle_generation": self.signer_lifecycle_generation,
            "signer_lifecycle_anchor_id": self.signer_lifecycle_anchor_id,
            "signer_provider": self.signer_provider,
            "signer_process_pid": self.signer_process_pid,
            "signer_parent_pid": self.signer_parent_pid,
            "observed_at_ms": self.observed_at_ms,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    def audit_dict(self) -> dict[str, object]:
        return {
            **self.signing_payload(),
            "signer_signature": self.signer_signature,
            "evidence_id": self.evidence_id,
        }


def _read_stable_agent_implementation_evidence_file(
    path: Path,
    *,
    reject_group_writable: bool = True,
) -> tuple[bytes, Path] | None:
    """Read one bounded regular file while pinning its path and inode state."""

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        path = resolve_agent_implementation_private_state_path(path)
        before_path = path.lstat()
        before_resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not stat_module.S_ISREG(before_path.st_mode)
            or before_path.st_uid != os.geteuid()
            or before_path.st_nlink != 1
            or stat_module.S_IMODE(before_path.st_mode)
            & (0o022 if reject_group_writable else 0o002)
            or not 0 < before_path.st_size
            <= _AGENT_IMPLEMENTATION_MAX_SESSION_BYTES
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            remaining = _AGENT_IMPLEMENTATION_MAX_SESSION_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        after_resolved = path.resolve(strict=True)
    except OSError:
        return None
    if (
        not stat_module.S_ISREG(opened.st_mode)
        or not stat_module.S_ISREG(after_open.st_mode)
        or path.is_symlink()
        or not stat_module.S_ISREG(after_path.st_mode)
        or any(
            item.st_uid != os.geteuid()
            or item.st_nlink != 1
            or stat_module.S_IMODE(item.st_mode)
            & (0o022 if reject_group_writable else 0o002)
            for item in (opened, after_open, after_path)
        )
        or not 0 < len(raw) <= _AGENT_IMPLEMENTATION_MAX_SESSION_BYTES
        or len(raw) != after_open.st_size
        or not (
            identity(before_path)
            == identity(opened)
            == identity(after_open)
            == identity(after_path)
        )
        or before_resolved != after_resolved
    ):
        return None
    return raw, after_resolved


def _agent_native_failure_type(line: str) -> str:
    if (
        not line
        or len(line.encode("utf-8", errors="replace"))
        > _AGENT_IMPLEMENTATION_MAX_STREAM_EVENT_BYTES
    ):
        return ""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict) or payload.get("method") not in {
        "_x.ai/session/update",
        "session/update",
    }:
        return ""
    params = payload.get("params")
    update = params.get("update") if isinstance(params, dict) else None
    if (
        not isinstance(update, dict)
        or update.get("sessionUpdate") != "retry_state"
        or update.get("type") != "failed"
    ):
        return ""
    error_type = str(update.get("error_type") or "").strip().casefold()
    message = str(update.get("message") or "").strip()
    if error_type in _AGENT_IMPLEMENTATION_QUOTA_VERIFIER_RESULTS:
        return error_type
    if (
        error_type == "api"
        and message == _AGENT_IMPLEMENTATION_BALANCE_EXHAUSTED_MESSAGE
    ):
        return "usage_pool_exhausted"
    if (
        error_type == "api"
        and message == _AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE
    ):
        return "spending_limit_exhausted"
    return error_type or "unknown"


def _canonical_agent_quota_verifier_command(
    command: object,
    *,
    expected_session_id: str,
    workspace: Path | str | None,
    prompt_path: Path | str | None,
    require_live_paths: bool = True,
) -> tuple[str, ...] | None:
    """Validate the exact isolated, tool-free native verifier invocation."""

    if (
        not isinstance(command, list)
        or workspace is None
        or prompt_path is None
        or not command
        or any(not isinstance(item, str) for item in command)
    ):
        return None
    raw_workspace = Path(workspace)
    raw_prompt = Path(prompt_path)
    raw_executable = Path(command[0])
    if (
        not raw_workspace.is_absolute()
        or not raw_prompt.is_absolute()
        or not raw_executable.is_absolute()
        or ".." in raw_workspace.parts
        or ".." in raw_prompt.parts
        or ".." in raw_executable.parts
    ):
        return None
    if require_live_paths:
        try:
            workspace_path = raw_workspace.resolve(strict=True)
            prompt = raw_prompt.resolve(strict=True)
            executable = raw_executable.resolve(strict=True)
            executable_stat = executable.stat()
        except OSError:
            return None
        if (
            not stat_module.S_ISREG(executable_stat.st_mode)
            or executable_stat.st_uid not in {0, os.geteuid()}
            or executable_stat.st_mode & 0o022
            or not os.access(executable, os.X_OK)
        ):
            return None
    else:
        workspace_path = raw_workspace
        prompt = raw_prompt
        executable = raw_executable
    if prompt.parent != workspace_path.parent:
        return None
    expected = [
        str(executable),
        "--model",
        "grok-4.5",
        "--max-turns",
        "1",
        "--cwd",
        str(workspace_path),
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "streaming-json",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--verbatim",
        "--tools",
        "",
        "--prompt-file",
        str(prompt),
        "--session-id",
        expected_session_id,
        "--disallowed-tools",
        AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
    ]
    return tuple(command) if command == expected else None


def validate_agent_implementation_quota_evidence(
    *,
    grok_home: Path | str,
    expected_session_id: str,
    verifier_returncode: int,
    failure_receipt: Mapping[str, object],
    invocation_binding: AgentImplementationInvocationBinding | None = None,
    verifier_command: list[str] | None = None,
    verifier_workspace: Path | str | None = None,
    verifier_prompt_path: Path | str | None = None,
    observed_at_ms: int | None = None,
    max_age_ms: int = 60 * 1000,
) -> AgentImplementationQuotaEvidence | None:
    """Validate one native, terminal-correlated isolated verifier session."""

    home = Path(grok_home)
    expected_model = failure_receipt.get("primary_model")
    preflight_receipt_id = failure_receipt.get("receipt_id")
    preflight_nonce = failure_receipt.get("nonce")
    probe_contract_id = failure_receipt.get("probe_contract_id")
    protected = invocation_binding is not None
    timestamp = (
        int(time.time() * 1000)
        if observed_at_ms is None
        else observed_at_ms
    )
    canonical_command = (
        _canonical_agent_quota_verifier_command(
            verifier_command,
            expected_session_id=expected_session_id,
            workspace=verifier_workspace,
            prompt_path=verifier_prompt_path,
        )
        if protected
        else ()
    )
    if (
        expected_model != "grok-4.5"
        or not isinstance(preflight_receipt_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", preflight_receipt_id) is None
        or not isinstance(preflight_nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", preflight_nonce) is None
        or probe_contract_id != _AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID
        or not isinstance(verifier_returncode, int)
        or isinstance(verifier_returncode, bool)
        or verifier_returncode == 0
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or timestamp <= 0
        or isinstance(max_age_ms, bool)
        or not isinstance(max_age_ms, int)
        or not 0 < max_age_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
        or (protected and canonical_command is None)
    ):
        return None
    try:
        uuid.UUID(expected_session_id)
    except ValueError:
        return None
    record = home / "sessions" / expected_session_id / "updates.jsonl"
    try:
        home_resolved = home.resolve(strict=True)
        transcript_read = _read_stable_agent_implementation_evidence_file(
            record,
            # The pre-existing legacy route consumed provider-native files
            # created under a private 0700 session home with the caller's
            # umask.  Preserve that contract; the signed protected route
            # requires non-group-writable evidence at both reads.
            reject_group_writable=protected,
        )
        if transcript_read is None:
            return None
        transcript_bytes, record_resolved = transcript_read
        if not record_resolved.is_relative_to(home_resolved):
            return None
        uuid.UUID(record.parent.name)
        transcript_text = transcript_bytes.decode("utf-8")
    except (OSError, UnicodeError, ValueError):
        return None
    recorded_session_id = record.parent.name
    if recorded_session_id != expected_session_id:
        return None

    observed_models: set[str] = set()
    latest_failure: tuple[str, str] | None = None
    latest_relevant = ""
    terminal_verdict = ""
    final_update_type = ""
    retry_failure_count = 0
    user_message_count = 0
    allowed_update_types = {
        "retry_state",
        "user_message_chunk",
        "turn_completed",
    }
    if not transcript_text.endswith("\n") or "\r" in transcript_text:
        return None

    def reject_duplicate_keys(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate native session JSON key")
            result[key] = value
        return result

    for raw_line in transcript_text.splitlines(keepends=True):
        if len(raw_line.encode("utf-8")) > (
            _AGENT_IMPLEMENTATION_MAX_SESSION_BYTES
        ):
            return None
        try:
            payload = json.loads(
                raw_line,
                object_pairs_hook=reject_duplicate_keys,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("method") not in {
            "_x.ai/session/update",
            "session/update",
        }:
            return None
        params = payload.get("params")
        if (
            not isinstance(params, dict)
            or params.get("sessionId") != recorded_session_id
        ):
            return None
        update = params.get("update")
        if not isinstance(update, dict):
            return None
        update_type = str(update.get("sessionUpdate") or "")
        if update_type not in allowed_update_types:
            return None
        final_update_type = update_type
        metadata = update.get("_meta")
        if isinstance(metadata, dict):
            model_id = str(metadata.get("modelId") or "").strip()
            if model_id:
                observed_models.add(model_id)
        if update_type == "retry_state":
            if update.get("type") != "failed":
                latest_failure = None
                latest_relevant = "retry_state"
                terminal_verdict = ""
                continue
            failure_type = _agent_native_failure_type(raw_line)
            failure_message = str(update.get("message") or "").strip()
            retry_failure_count += 1
            latest_failure = (failure_type, failure_message)
            latest_relevant = "retry_state"
            terminal_verdict = ""
        elif update_type == "turn_completed":
            terminal_verdict = ""
            if (
                str(update.get("stop_reason") or "").casefold() == "error"
                and latest_relevant == "retry_state"
                and latest_failure is not None
                and latest_failure[0]
                in _AGENT_IMPLEMENTATION_QUOTA_VERIFIER_RESULTS
                and latest_failure[1]
                and str(update.get("agent_result") or "").strip()
                == latest_failure[1]
            ):
                terminal_verdict = latest_failure[0]
            latest_relevant = "turn_completed"
        elif update_type == "user_message_chunk":
            user_message_count += 1

    summary_path = record.parent / "summary.json"
    try:
        summary_read = _read_stable_agent_implementation_evidence_file(
            summary_path,
            reject_group_writable=protected,
        )
        if summary_read is None:
            return None
        summary_bytes, summary_resolved = summary_read
        if (
            not summary_resolved.is_relative_to(home_resolved)
            or summary_resolved.parent != record_resolved.parent
        ):
            return None
        summary = json.loads(
            summary_bytes,
            object_pairs_hook=reject_duplicate_keys,
        )
        summary_info = summary.get("info") if isinstance(summary, dict) else None
        summary_home = Path(str(summary.get("grok_home") or "")).resolve(
            strict=True
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    if (
        final_update_type != "turn_completed"
        or not observed_models.issubset({expected_model})
        or retry_failure_count != 1
        or user_message_count > 1
        or not isinstance(summary_info, dict)
        or summary_info.get("id") != recorded_session_id
        or summary.get("current_model_id") != expected_model
        or summary_home != home_resolved
        or latest_failure not in _AGENT_IMPLEMENTATION_NATIVE_QUOTA_FAILURES
        or terminal_verdict not in _AGENT_IMPLEMENTATION_QUOTA_VERIFIER_RESULTS
    ):
        return None
    evidence_body: dict[str, object] = {
        "schema": _AGENT_IMPLEMENTATION_QUOTA_EVIDENCE_SCHEMA,
        "preflight_receipt_id": preflight_receipt_id,
        "preflight_nonce": preflight_nonce,
        "primary_provider": "grok",
        "primary_model": expected_model,
        "verifier_session_id": recorded_session_id,
        "verifier_provider": "grok_cli",
        "verifier_model": expected_model,
        "verifier_command": list(canonical_command or ()),
        "verifier_command_id": (
            _content_addressed_mapping(
                {"argv": list(canonical_command or ())},
                identity_field="command_id",
            )
            if protected
            else ""
        ),
        "verifier_returncode": verifier_returncode,
        "verifier_result": terminal_verdict,
        "probe_contract_id": probe_contract_id,
        "transcript_sha256": "sha256:"
        + hashlib.sha256(transcript_bytes).hexdigest(),
        "summary_sha256": "sha256:"
        + hashlib.sha256(summary_bytes).hexdigest(),
        "invocation_id": "",
        "logical_attempt_id": "",
        "route_id": "",
        "signer_identity_did": "",
        "signer_key_id": "",
        "signer_profile_id": "",
        "signer_profile_content_id": "",
        "signer_lifecycle_generation": 0,
        "signer_lifecycle_anchor_id": "",
        "signer_provider": "",
        "signer_process_pid": 0,
        "signer_parent_pid": 0,
        "observed_at_ms": 0,
        "issued_at_ms": 0,
        "expires_at_ms": 0,
    }
    signer_signature = ""
    if protected:
        assert invocation_binding is not None
        receipt_observed = failure_receipt.get("observed_at_ms")
        receipt_expires = failure_receipt.get("expires_at_ms")
        if (
            isinstance(receipt_observed, bool)
            or not isinstance(receipt_observed, int)
            or receipt_observed <= 0
            or isinstance(receipt_expires, bool)
            or not isinstance(receipt_expires, int)
            or not receipt_observed <= timestamp < receipt_expires
            or not invocation_binding.issued_at_ms <= timestamp
            < invocation_binding.expires_at_ms
        ):
            return None
        expires_at_ms = min(
            timestamp + max_age_ms,
            receipt_expires,
            invocation_binding.expires_at_ms,
        )
        if expires_at_ms <= timestamp:
            return None
        try:
            from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
                LocalProfileError,
                load_local_profile,
                sign_profile_binding,
            )

            profile = load_local_profile(
                repository_cid=invocation_binding.repository_cid,
                profile_dir=Path(invocation_binding.profile_dir),
                lifecycle_dir=Path(
                    invocation_binding.profile_lifecycle_dir
                ),
            )
            if (
                profile.profile_id != invocation_binding.profile_id
                or profile.identity_did
                != invocation_binding.profile_identity_did
                or profile.lifecycle_anchor_id
                != invocation_binding.profile_lifecycle_anchor_id
                or profile.lifecycle_generation
                != invocation_binding.profile_lifecycle_generation
                or profile.route_id != invocation_binding.route_id
                or profile.reviewer_identity
                != invocation_binding.reviewer_identity
                or profile.reviewer_provider
                != invocation_binding.reviewer_provider
            ):
                return None
            evidence_body.update(
                {
                    "invocation_id": invocation_binding.invocation_id,
                    "logical_attempt_id": (
                        invocation_binding.logical_attempt_id
                    ),
                    "route_id": invocation_binding.route_id,
                    "signer_identity_did": profile.identity_did,
                    "signer_key_id": profile.identity_did,
                    "signer_profile_id": profile.profile_id,
                    "signer_profile_content_id": profile.content_id,
                    "signer_lifecycle_generation": (
                        profile.lifecycle_generation
                    ),
                    "signer_lifecycle_anchor_id": (
                        profile.lifecycle_anchor_id
                    ),
                    "signer_provider": profile.reviewer_provider,
                    "signer_process_pid": os.getpid(),
                    "signer_parent_pid": os.getppid(),
                    "observed_at_ms": timestamp,
                    "issued_at_ms": timestamp,
                    "expires_at_ms": expires_at_ms,
                }
            )
            signed = sign_profile_binding(
                profile_dir=Path(invocation_binding.profile_dir),
                lifecycle_dir=Path(
                    invocation_binding.profile_lifecycle_dir
                ),
                payload=evidence_body,
            )
            if (
                signed.get("identity") != profile.identity_did
                or signed.get("profile_id") != profile.profile_id
            ):
                return None
            signer_signature = signed["signature"]
            # Rotation/revocation between the snapshot and signature must not
            # produce a portable receipt for either generation.
            after = load_local_profile(
                repository_cid=invocation_binding.repository_cid,
                profile_dir=Path(invocation_binding.profile_dir),
                lifecycle_dir=Path(
                    invocation_binding.profile_lifecycle_dir
                ),
            )
            if after.content_id != profile.content_id:
                return None
        except (LocalProfileError, KeyError, OSError, ValueError):
            return None
    audit_body = {**evidence_body, "signer_signature": signer_signature}
    evidence_id = _content_addressed_mapping(
        audit_body,
        identity_field="evidence_id",
    )
    audit_with_id = {**audit_body, "evidence_id": evidence_id}
    return AgentImplementationQuotaEvidence(
        **{
            **evidence_body,
            "verifier_command": tuple(
                evidence_body["verifier_command"]  # type: ignore[arg-type]
            ),
        },
        signer_signature=signer_signature,
        evidence_id=evidence_id,
        _validation_seal=_agent_implementation_private_seal(audit_with_id),
    )


def _valid_agent_implementation_quota_evidence(
    evidence: object,
    *,
    failure_receipt: Mapping[str, object],
    invocation: AgentImplementationInvocationBinding | None,
    now_ms: int | None,
    max_age_ms: int | None,
    require_current_lifecycle: bool = True,
) -> bool:
    if not isinstance(evidence, AgentImplementationQuotaEvidence):
        return False
    audit = evidence.audit_dict()
    if (
        evidence.schema != _AGENT_IMPLEMENTATION_QUOTA_EVIDENCE_SCHEMA
        or evidence._validation_seal
        != _agent_implementation_private_seal(audit)
        or evidence.evidence_id
        != _content_addressed_mapping(audit, identity_field="evidence_id")
        or evidence.preflight_receipt_id
        != failure_receipt.get("receipt_id")
        or evidence.preflight_nonce != failure_receipt.get("nonce")
        or evidence.primary_provider
        != failure_receipt.get("primary_provider")
        or evidence.primary_model
        != failure_receipt.get("primary_model")
        or evidence.primary_model != "grok-4.5"
        or evidence.verifier_provider != "grok_cli"
        or evidence.verifier_model != evidence.primary_model
        or isinstance(evidence.verifier_returncode, bool)
        or not isinstance(evidence.verifier_returncode, int)
        or evidence.verifier_returncode == 0
        or evidence.probe_contract_id
        != failure_receipt.get("probe_contract_id")
        or evidence.verifier_result
        not in _AGENT_IMPLEMENTATION_QUOTA_VERIFIER_RESULTS
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in (
                evidence.preflight_receipt_id,
                evidence.transcript_sha256,
                evidence.summary_sha256,
                evidence.evidence_id,
            )
        )
    ):
        return False
    if invocation is None:
        return bool(
            not evidence.verifier_command
            and not evidence.verifier_command_id
            and not evidence.invocation_id
            and not evidence.logical_attempt_id
            and not evidence.route_id
            and not evidence.signer_identity_did
            and not evidence.signer_key_id
            and not evidence.signer_profile_id
            and not evidence.signer_profile_content_id
            and evidence.signer_lifecycle_generation == 0
            and not evidence.signer_lifecycle_anchor_id
            and not evidence.signer_provider
            and evidence.signer_process_pid == 0
            and evidence.signer_parent_pid == 0
            and evidence.observed_at_ms == 0
            and evidence.issued_at_ms == 0
            and evidence.expires_at_ms == 0
            and not evidence.signer_signature
        )
    if (
        not isinstance(now_ms, int)
        or isinstance(now_ms, bool)
        or now_ms <= 0
        or not isinstance(max_age_ms, int)
        or isinstance(max_age_ms, bool)
        or not 0 < max_age_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
        or evidence.invocation_id != invocation.invocation_id
        or evidence.logical_attempt_id != invocation.logical_attempt_id
        or evidence.route_id != invocation.route_id
        or evidence.signer_identity_did != invocation.profile_identity_did
        or evidence.signer_key_id != evidence.signer_identity_did
        or evidence.signer_profile_id != invocation.profile_id
        or evidence.signer_profile_content_id != invocation.authority_cid
        or evidence.signer_lifecycle_generation
        != invocation.profile_lifecycle_generation
        or evidence.signer_lifecycle_anchor_id
        != invocation.profile_lifecycle_anchor_id
        or evidence.signer_provider != invocation.reviewer_provider
        or isinstance(evidence.signer_process_pid, bool)
        or not isinstance(evidence.signer_process_pid, int)
        or evidence.signer_process_pid <= 0
        or isinstance(evidence.signer_parent_pid, bool)
        or not isinstance(evidence.signer_parent_pid, int)
        or evidence.signer_parent_pid <= 0
        or evidence.signer_process_pid == evidence.signer_parent_pid
        or evidence._signer_process_validated is not True
        or not evidence.signer_signature
        or not (
            0
            < evidence.observed_at_ms
            <= evidence.issued_at_ms
            <= now_ms + _AGENT_IMPLEMENTATION_MAX_CLOCK_SKEW_MS
        )
        or now_ms > evidence.expires_at_ms
        or evidence.expires_at_ms - evidence.issued_at_ms > max_age_ms
        or now_ms - evidence.observed_at_ms > max_age_ms
        or evidence.expires_at_ms > invocation.expires_at_ms
        or evidence.expires_at_ms
        > int(failure_receipt.get("expires_at_ms") or 0)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", evidence.signer_profile_content_id
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", evidence.verifier_command_id
        )
        is None
    ):
        return False
    command = list(evidence.verifier_command)
    try:
        workspace_index = command.index("--cwd") + 1
        prompt_index = command.index("--prompt-file") + 1
        canonical_command = _canonical_agent_quota_verifier_command(
            command,
            expected_session_id=evidence.verifier_session_id,
            workspace=command[workspace_index],
            prompt_path=command[prompt_index],
            require_live_paths=False,
        )
    except (IndexError, ValueError):
        return False
    if (
        canonical_command != evidence.verifier_command
        or evidence.verifier_command_id
        != _content_addressed_mapping(
            {"argv": command}, identity_field="command_id"
        )
    ):
        return False
    try:
        from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
            LocalProfileError,
            load_local_profile,
        )

        _agent_verify_did_signature(
            identity_did=evidence.signer_identity_did,
            payload=evidence.signing_payload(),
            signature=evidence.signer_signature,
        )
        if not require_current_lifecycle:
            # Terminal audit is anchored to the immutable effect_started CAS
            # timestamp.  A later legitimate profile rotation/revocation may
            # not make an already-run provider effect impossible to record;
            # the signed generation/content binding remains verified here.
            return True
        profile = load_local_profile(
            repository_cid=invocation.repository_cid,
            profile_dir=Path(invocation.profile_dir),
            lifecycle_dir=Path(invocation.profile_lifecycle_dir),
        )
    except (LocalProfileError, OSError, ValueError):
        return False
    return bool(
        profile.profile_id == evidence.signer_profile_id
        and profile.identity_did == evidence.signer_identity_did
        and profile.content_id == evidence.signer_profile_content_id
        and profile.lifecycle_generation
        == evidence.signer_lifecycle_generation
        and profile.lifecycle_anchor_id
        == evidence.signer_lifecycle_anchor_id
        and profile.reviewer_provider == evidence.signer_provider
        and profile.route_id == evidence.route_id
    )


def parse_agent_implementation_quota_evidence(
    value: object,
    *,
    failure_receipt: Mapping[str, object],
    invocation_binding: AgentImplementationInvocationBinding,
    now_ms: int,
    max_age_ms: int,
    expected_signer_parent_pid: int | None = None,
    expected_signer_process_pid: int | None = None,
    require_current_lifecycle: bool = True,
) -> AgentImplementationQuotaEvidence | None:
    """Adopt a child-verifier's exact signed receipt into this process."""

    expected = {
        "schema",
        "preflight_receipt_id",
        "preflight_nonce",
        "primary_provider",
        "primary_model",
        "verifier_session_id",
        "verifier_provider",
        "verifier_model",
        "verifier_command",
        "verifier_command_id",
        "verifier_returncode",
        "verifier_result",
        "probe_contract_id",
        "transcript_sha256",
        "summary_sha256",
        "invocation_id",
        "logical_attempt_id",
        "route_id",
        "signer_identity_did",
        "signer_key_id",
        "signer_profile_id",
        "signer_profile_content_id",
        "signer_lifecycle_generation",
        "signer_lifecycle_anchor_id",
        "signer_provider",
        "signer_process_pid",
        "signer_parent_pid",
        "observed_at_ms",
        "issued_at_ms",
        "expires_at_ms",
        "signer_signature",
        "evidence_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return None
    command = value.get("verifier_command")
    integer_fields = {
        "verifier_returncode",
        "signer_lifecycle_generation",
        "signer_process_pid",
        "signer_parent_pid",
        "observed_at_ms",
        "issued_at_ms",
        "expires_at_ms",
    }
    text_fields = expected - integer_fields - {"verifier_command"}
    if (
        not isinstance(command, list)
        or any(not isinstance(item, str) for item in command)
        or any(not isinstance(value.get(name), str) for name in text_fields)
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int)
            for name in integer_fields
        )
    ):
        return None
    expected_parent = (
        os.getpid()
        if expected_signer_parent_pid is None
        else expected_signer_parent_pid
    )
    if (
        isinstance(expected_parent, bool)
        or not isinstance(expected_parent, int)
        or expected_parent <= 0
        or value.get("signer_parent_pid") != expected_parent
        or (
            expected_signer_process_pid is not None
            and (
                isinstance(expected_signer_process_pid, bool)
                or not isinstance(expected_signer_process_pid, int)
                or expected_signer_process_pid <= 0
                or value.get("signer_process_pid")
                != expected_signer_process_pid
            )
        )
    ):
        return None
    values = dict(value)
    values["verifier_command"] = tuple(command)
    evidence = AgentImplementationQuotaEvidence(
        **values,  # type: ignore[arg-type]
        _validation_seal=_agent_implementation_private_seal(dict(value)),
        _signer_process_validated=True,
    )
    return (
        evidence
        if _valid_agent_implementation_quota_evidence(
            evidence,
            failure_receipt=failure_receipt,
            invocation=invocation_binding,
            now_ms=now_ms,
            max_age_ms=max_age_ms,
            require_current_lifecycle=require_current_lifecycle,
        )
        else None
    )


@dataclass(frozen=True, slots=True)
class AgentImplementationEffectAuthorizationContext:
    """Historically verified authority captured by the winning CAS claim."""

    route: AgentImplementationRoutePlan
    failure_receipt: Mapping[str, object]
    quota_evidence: AgentImplementationQuotaEvidence | None
    decision: AgentImplementationFallbackDecision
    context_id: str


def _agent_decode_embedded_authority_json(
    value: object,
    *,
    maximum_bytes: int,
) -> tuple[bytes, dict[str, object]]:
    """Decode one exact JSON blob carried by a historical effect claim."""

    if not isinstance(value, str) or len(value) > 4 * maximum_bytes // 3 + 8:
        raise ValueError("embedded effect authority is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("embedded effect authority is invalid") from exc
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("embedded effect authority is invalid")

    def reject_duplicate_keys(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, item in pairs:
            if key in decoded:
                raise ValueError("embedded effect authority has duplicate keys")
            decoded[key] = item
        return decoded

    try:
        decoded = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("embedded effect authority is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("embedded effect authority must be an object")
    return raw, decoded


def _agent_authorization_from_historical_binding(
    value: object,
) -> AgentImplementationRouteAuthorization:
    """Strictly rebuild a sealed authorization from signed persisted fields."""

    expected = {
        "board_namespace",
        "artifact_path",
        "artifact_sha256",
        "authorization_kind",
        "source_head",
        "source_tree",
        "authorization_id",
        "reviewer_identity",
        "reviewer_provider",
        "reviewer_signature",
        "reviewer_profile_id",
        "reviewer_profile_content_id",
        "reviewer_lifecycle_anchor_id",
        "reviewer_lifecycle_generation",
        "reviewer_witness_path",
        "reviewer_witness_sha256",
        "lifecycle_root_identity_did",
        "lifecycle_witness_nonce",
        "lifecycle_root_pin_path",
        "lifecycle_root_pin_sha256",
        "authorized_at_ms",
        "fallback_implementer_identity",
        "authority_bounds",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("historical route authorization is invalid")
    bounds_raw = value.get("authority_bounds")
    if not isinstance(bounds_raw, Mapping) or set(bounds_raw) != {
        "repository_cid",
        "baseline_commit",
        "effects",
        "budget_cid",
        "resource_cid",
        "authority_cid",
    }:
        raise ValueError("historical route authority bounds are invalid")
    effects_raw = bounds_raw.get("effects")
    text_names = expected - {
        "reviewer_lifecycle_generation",
        "authorized_at_ms",
        "authority_bounds",
    }
    if (
        any(not isinstance(value.get(name), str) for name in text_names)
        or not isinstance(effects_raw, list)
        or not effects_raw
        or any(not isinstance(item, str) or not item for item in effects_raw)
        or any(
            not isinstance(bounds_raw.get(name), str)
            or not str(bounds_raw.get(name))
            for name in (
                "repository_cid",
                "baseline_commit",
                "budget_cid",
                "resource_cid",
                "authority_cid",
            )
        )
        or isinstance(value.get("reviewer_lifecycle_generation"), bool)
        or not isinstance(value.get("reviewer_lifecycle_generation"), int)
        or int(value.get("reviewer_lifecycle_generation") or 0) < 1
        or isinstance(value.get("authorized_at_ms"), bool)
        or not isinstance(value.get("authorized_at_ms"), int)
        or int(value.get("authorized_at_ms") or 0) <= 0
    ):
        raise ValueError("historical route authorization types are invalid")
    effects = tuple(effects_raw)
    if effects != tuple(sorted(effects)) or len(set(effects)) != len(effects):
        raise ValueError("historical route authority effects are invalid")
    bounds = AgentImplementationAuthorityBounds(
        repository_cid=str(bounds_raw["repository_cid"]),
        baseline_commit=str(bounds_raw["baseline_commit"]),
        effects=effects,
        budget_cid=str(bounds_raw["budget_cid"]),
        resource_cid=str(bounds_raw["resource_cid"]),
        authority_cid=str(bounds_raw["authority_cid"]),
    )
    identity_body: dict[str, object] = {
        "schema": _AGENT_ROUTE_AUTHORIZATION_SCHEMA,
        **{name: value[name] for name in text_names},
        "reviewer_lifecycle_generation": value[
            "reviewer_lifecycle_generation"
        ],
        "authorized_at_ms": value["authorized_at_ms"],
        "authority_bounds": bounds.as_dict(),
    }
    # authorization_id is derived from, rather than included in, the body.
    identity_body.pop("authorization_id", None)
    authorization_id = _agent_implementation_route_id(identity_body)
    if authorization_id != value.get("authorization_id"):
        raise ValueError("historical route authorization identity drifted")
    return AgentImplementationRouteAuthorization(
        board_namespace=str(value["board_namespace"]),
        artifact_path=str(value["artifact_path"]),
        artifact_sha256=str(value["artifact_sha256"]),
        authorization_kind=str(value["authorization_kind"]),
        source_head=str(value["source_head"]),
        source_tree=str(value["source_tree"]),
        authorization_id=authorization_id,
        reviewer_identity=str(value["reviewer_identity"]),
        reviewer_provider=str(value["reviewer_provider"]),
        reviewer_signature=str(value["reviewer_signature"]),
        reviewer_profile_id=str(value["reviewer_profile_id"]),
        reviewer_profile_content_id=str(value["reviewer_profile_content_id"]),
        reviewer_lifecycle_anchor_id=str(value["reviewer_lifecycle_anchor_id"]),
        reviewer_lifecycle_generation=int(value["reviewer_lifecycle_generation"]),
        reviewer_witness_path=str(value["reviewer_witness_path"]),
        reviewer_witness_sha256=str(value["reviewer_witness_sha256"]),
        lifecycle_root_identity_did=str(value["lifecycle_root_identity_did"]),
        lifecycle_witness_nonce=str(value["lifecycle_witness_nonce"]),
        lifecycle_root_pin_path=str(value["lifecycle_root_pin_path"]),
        lifecycle_root_pin_sha256=str(value["lifecycle_root_pin_sha256"]),
        authorized_at_ms=int(value["authorized_at_ms"]),
        fallback_implementer_identity=str(value["fallback_implementer_identity"]),
        authority_bounds=bounds,
        _validation_seal=_agent_implementation_private_seal(identity_body),
    )


def _agent_verify_historical_authority_snapshot(
    *,
    authorization: AgentImplementationRouteAuthorization,
    artifact_raw: bytes,
    artifact: Mapping[str, object],
    root_pin_raw: bytes,
    root_pin: Mapping[str, object],
    witness_raw: bytes,
    witness: Mapping[str, object],
    repository_receipt: object,
) -> None:
    """Verify the portable authority saved before the once-only effect."""

    bounds = authorization.authority_bounds
    if bounds is None:
        raise ValueError("historical authority bounds are unavailable")
    expected_top = {
        "schema",
        "board_namespace",
        "authorization_source",
        "route",
        "ownership_contract",
        "bootstrap_route_guarantees",
        "reviewer",
        "authority_bounds",
        "fallback_implementer_identity",
        "lifecycle_root_identity_did",
        "lifecycle_witness_nonce",
        "lifecycle_root_pin_path",
        "lifecycle_root_pin_sha256",
        "authorized_at_ms",
    }
    source = artifact.get("authorization_source")
    route = artifact.get("route")
    ownership = artifact.get("ownership_contract")
    bootstrap = artifact.get("bootstrap_route_guarantees")
    reviewer = artifact.get("reviewer")
    artifact_bounds = artifact.get("authority_bounds")
    if (
        set(artifact) != expected_top
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "kind",
            "source_head",
            "source_tree",
            "prospective_only",
            "requires_descendant_tree",
        }
        or not isinstance(route, Mapping)
        or set(route)
        != {
            "route_id",
            "primary_provider_id",
            "primary_model_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "allowed_trigger_classes",
        }
        or not isinstance(ownership, Mapping)
        or set(ownership)
        != {
            "canonical_route_plan_owner",
            "typed_fallback_decision_owner",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed",
        }
        or not isinstance(bootstrap, Mapping)
        or set(bootstrap) != {"explicit_codex_review_conflict_denied"}
        or not isinstance(reviewer, Mapping)
        or set(reviewer)
        != {
            "identity",
            "provider",
            "profile_id",
            "profile_content_id",
            "lifecycle_anchor_id",
            "generation",
            "witness_path",
            "witness_sha256",
            "signature",
        }
        or not isinstance(artifact_bounds, Mapping)
        or dict(artifact_bounds) != bounds.as_dict()
    ):
        raise ValueError("historical authorization artifact is noncanonical")
    expected_route = {
        "route_id": _V3_AGENT_IMPLEMENTATION_ROUTE_ID,
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    expected_reviewer = {
        "identity": authorization.reviewer_identity,
        "provider": authorization.reviewer_provider,
        "profile_id": authorization.reviewer_profile_id,
        "profile_content_id": authorization.reviewer_profile_content_id,
        "lifecycle_anchor_id": authorization.reviewer_lifecycle_anchor_id,
        "generation": authorization.reviewer_lifecycle_generation,
        "witness_path": authorization.reviewer_witness_path,
        "witness_sha256": authorization.reviewer_witness_sha256,
        "signature": authorization.reviewer_signature,
    }
    if (
        artifact.get("schema") != _AGENT_ROUTE_AUTHORIZATION_SCHEMA
        or artifact.get("board_namespace") != authorization.board_namespace
        or dict(source)
        != {
            "kind": authorization.authorization_kind,
            "source_head": authorization.source_head,
            "source_tree": authorization.source_tree,
            "prospective_only": True,
            "requires_descendant_tree": True,
        }
        or dict(route) != expected_route
        or dict(ownership)
        != {
            "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
            "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
        }
        or dict(bootstrap)
        != {"explicit_codex_review_conflict_denied": True}
        or dict(reviewer) != expected_reviewer
        or artifact.get("fallback_implementer_identity")
        != authorization.fallback_implementer_identity
        or artifact.get("lifecycle_root_identity_did")
        != authorization.lifecycle_root_identity_did
        or artifact.get("lifecycle_witness_nonce")
        != authorization.lifecycle_witness_nonce
        or artifact.get("lifecycle_root_pin_path")
        != authorization.lifecycle_root_pin_path
        or artifact.get("lifecycle_root_pin_sha256")
        != authorization.lifecycle_root_pin_sha256
        or artifact.get("authorized_at_ms") != authorization.authorized_at_ms
        or "sha256:" + hashlib.sha256(artifact_raw).hexdigest()
        != authorization.artifact_sha256
        or "sha256:" + hashlib.sha256(root_pin_raw).hexdigest()
        != authorization.lifecycle_root_pin_sha256
        or "sha256:" + hashlib.sha256(witness_raw).hexdigest()
        != authorization.reviewer_witness_sha256
    ):
        raise ValueError("historical authorization artifact drifted")
    root_pin_expected = {
        "schema",
        "board_namespace",
        "base_head",
        "base_tree",
        "root_identity_did",
        "pinned_at_ms",
        "pin_id",
    }
    if (
        set(root_pin) != root_pin_expected
        or root_pin.get("schema") != _AGENT_LIFECYCLE_ROOT_PIN_SCHEMA
        or root_pin.get("board_namespace") != authorization.board_namespace
        or root_pin.get("root_identity_did")
        != authorization.lifecycle_root_identity_did
        or any(
            not isinstance(root_pin.get(name), str)
            for name in root_pin_expected - {"pinned_at_ms"}
        )
        or isinstance(root_pin.get("pinned_at_ms"), bool)
        or not isinstance(root_pin.get("pinned_at_ms"), int)
        or int(root_pin.get("pinned_at_ms") or 0) <= 0
        or root_pin.get("pin_id")
        != _content_addressed_mapping(root_pin, identity_field="pin_id")
    ):
        raise ValueError("historical lifecycle root pin is invalid")
    from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
        LocalProfileTampered,
        verify_local_profile_lifecycle_witness,
    )

    try:
        profile = verify_local_profile_lifecycle_witness(
            witness,
            expected_board_namespace=authorization.board_namespace,
            expected_base_head=authorization.source_head,
            expected_base_tree=authorization.source_tree,
            expected_nonce=authorization.lifecycle_witness_nonce,
            expected_root_identity_did=authorization.lifecycle_root_identity_did,
            reference_time_ms=authorization.authorized_at_ms,
            max_age_ms=10 * 60 * 1000,
        )
    except (LocalProfileTampered, TypeError, ValueError) as exc:
        raise ValueError("historical lifecycle witness is invalid") from exc
    if (
        profile.identity_did != authorization.reviewer_identity
        or profile.reviewer_identity != authorization.reviewer_identity
        or profile.reviewer_provider != authorization.reviewer_provider
        or profile.profile_id != authorization.reviewer_profile_id
        or profile.content_id != authorization.reviewer_profile_content_id
        or profile.lifecycle_anchor_id
        != authorization.reviewer_lifecycle_anchor_id
        or profile.lifecycle_generation
        != authorization.reviewer_lifecycle_generation
        or profile.repository_cid != bounds.repository_cid
        or profile.baseline_commit != bounds.baseline_commit
        or tuple(profile.effect_bounds) != bounds.effects
        or profile.budget_cid != bounds.budget_cid
        or profile.resource_cid != bounds.resource_cid
        or profile.content_id != bounds.authority_cid
        or profile.route_id != _V3_AGENT_IMPLEMENTATION_ROUTE_ID
        or profile.fallback_provider_id != "codex"
        or profile.fallback_model_id != "gpt-5.6-terra"
        or profile.fallback_reasoning_effort != "high"
    ):
        raise ValueError("historical lifecycle witness authority drifted")
    review_payload = agent_implementation_route_review_payload(
        board_namespace=authorization.board_namespace,
        authorization_kind=authorization.authorization_kind,
        source_head=authorization.source_head,
        source_tree=authorization.source_tree,
        route=route,
        authority_bounds=bounds.as_dict(),
        reviewer_identity=authorization.reviewer_identity,
        reviewer_provider=authorization.reviewer_provider,
        reviewer_profile_id=authorization.reviewer_profile_id,
        reviewer_profile_content_id=authorization.reviewer_profile_content_id,
        reviewer_lifecycle_anchor_id=authorization.reviewer_lifecycle_anchor_id,
        reviewer_lifecycle_generation=authorization.reviewer_lifecycle_generation,
        reviewer_witness_path=authorization.reviewer_witness_path,
        reviewer_witness_sha256=authorization.reviewer_witness_sha256,
        lifecycle_root_identity_did=authorization.lifecycle_root_identity_did,
        lifecycle_witness_nonce=authorization.lifecycle_witness_nonce,
        lifecycle_root_pin_path=authorization.lifecycle_root_pin_path,
        lifecycle_root_pin_sha256=authorization.lifecycle_root_pin_sha256,
        authorized_at_ms=authorization.authorized_at_ms,
        fallback_implementer_identity=authorization.fallback_implementer_identity,
    )
    _agent_verify_did_signature(
        identity_did=authorization.reviewer_identity,
        payload=review_payload,
        signature=authorization.reviewer_signature,
    )
    receipt_expected = {
        "schema",
        "accepted_head",
        "accepted_tree",
        "authorization_path",
        "authorization_blob_id",
        "witness_path",
        "witness_blob_id",
        "root_pin_path",
        "root_pin_blob_id",
        "authorization_commit_time_ms",
        "receipt_id",
    }
    if not isinstance(repository_receipt, Mapping) or set(
        repository_receipt
    ) != receipt_expected:
        raise ValueError("historical repository authority receipt is invalid")

    def git_blob_id(raw: bytes) -> str:
        return hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw,
            usedforsecurity=False,
        ).hexdigest()

    commit_time_ms = repository_receipt.get("authorization_commit_time_ms")
    if (
        repository_receipt.get("schema")
        != "ipfs_accelerate_py.agent-supervisor/"
        "provider-effect-repository-authority@1"
        or any(
            re.fullmatch(r"[0-9a-f]{40}", str(repository_receipt.get(name)))
            is None
            for name in (
                "accepted_head",
                "accepted_tree",
                "authorization_blob_id",
                "witness_blob_id",
                "root_pin_blob_id",
            )
        )
        or repository_receipt.get("authorization_path")
        != authorization.artifact_path
        or repository_receipt.get("witness_path")
        != authorization.reviewer_witness_path
        or repository_receipt.get("root_pin_path")
        != authorization.lifecycle_root_pin_path
        or repository_receipt.get("authorization_blob_id")
        != git_blob_id(artifact_raw)
        or repository_receipt.get("witness_blob_id") != git_blob_id(witness_raw)
        or repository_receipt.get("root_pin_blob_id")
        != git_blob_id(root_pin_raw)
        or isinstance(commit_time_ms, bool)
        or not isinstance(commit_time_ms, int)
        or commit_time_ms <= 0
        or abs(commit_time_ms - authorization.authorized_at_ms) > 10 * 60 * 1000
        or repository_receipt.get("receipt_id")
        != _content_addressed_mapping(
            repository_receipt,
            identity_field="receipt_id",
        )
    ):
        raise ValueError("historical repository authority receipt drifted")


def parse_agent_implementation_effect_authorization_context(
    value: object,
    *,
    repo_root: Path | str,
    effect_started_at_ms: int,
    expected_signer_parent_pid: int,
    max_age_ms: int,
) -> AgentImplementationEffectAuthorizationContext | None:
    """Verify original signed route authority at its durable claim instant.

    This is terminal/recovery validation only.  It deliberately does not
    grant a new provider effect and must be paired with the exact durable CAS
    record whose ``effect_started_at_ms`` is supplied here.
    """

    expected = {
        "schema",
        "route_binding",
        "failure_receipt",
        "quota_evidence",
        "expected_nonce",
        "expected_model",
        "expected_probe_returncode",
        "decision",
        "decision_id",
        "invocation_binding_id",
        "logical_attempt_id",
        "authorization_artifact_b64",
        "lifecycle_root_pin_b64",
        "lifecycle_witness_b64",
        "repository_receipt",
        "context_id",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema") != _AGENT_EFFECT_AUTHORIZATION_CONTEXT_SCHEMA
        or isinstance(effect_started_at_ms, bool)
        or not isinstance(effect_started_at_ms, int)
        or effect_started_at_ms <= 0
        or isinstance(expected_signer_parent_pid, bool)
        or not isinstance(expected_signer_parent_pid, int)
        or expected_signer_parent_pid <= 0
        or value.get("context_id")
        != _content_addressed_mapping(value, identity_field="context_id")
    ):
        return None
    route_raw = value.get("route_binding")
    receipt = value.get("failure_receipt")
    quota_raw = value.get("quota_evidence")
    decision_raw = value.get("decision")
    expected_returncode = value.get("expected_probe_returncode")
    if (
        not isinstance(route_raw, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(quota_raw, Mapping)
        or not isinstance(decision_raw, Mapping)
        or isinstance(expected_returncode, bool)
        or not isinstance(expected_returncode, int)
        or not isinstance(value.get("expected_nonce"), str)
        or not isinstance(value.get("expected_model"), str)
    ):
        return None
    try:
        artifact_raw, artifact = _agent_decode_embedded_authority_json(
            value.get("authorization_artifact_b64"),
            maximum_bytes=128 * 1024,
        )
        root_pin_raw, root_pin = _agent_decode_embedded_authority_json(
            value.get("lifecycle_root_pin_b64"),
            maximum_bytes=32 * 1024,
        )
        witness_raw, witness = _agent_decode_embedded_authority_json(
            value.get("lifecycle_witness_b64"),
            maximum_bytes=128 * 1024,
        )
        authorization = _agent_authorization_from_historical_binding(
            route_raw.get("authorization")
        )
        _agent_verify_historical_authority_snapshot(
            authorization=authorization,
            artifact_raw=artifact_raw,
            artifact=artifact,
            root_pin_raw=root_pin_raw,
            root_pin=root_pin,
            witness_raw=witness_raw,
            witness=witness,
            repository_receipt=value.get("repository_receipt"),
        )
        expected_route_fields = {
            *_AGENT_IMPLEMENTATION_ROUTE_FIELDS,
            "route_id",
            "authorization",
            "fallback_implementer_identity",
            "invocation_binding",
        }
        if set(route_raw) != expected_route_fields:
            return None
        route = resolve_agent_implementation_route(
            **{
                field: str(route_raw.get(field) or "")
                for field in _AGENT_IMPLEMENTATION_ROUTE_FIELDS
            },
            authorization=authorization,
        )
        if (
            route.route_id != route_raw.get("route_id")
            or route.fallback_implementer_identity
            != route_raw.get("fallback_implementer_identity")
        ):
            return None
        route = replace(
            route,
            fallback_implementer_identity=str(
                route_raw.get("fallback_implementer_identity")
            ),
        )
        invocation_raw = route_raw.get("invocation_binding")
        if not isinstance(invocation_raw, Mapping):
            return None
        route = bind_agent_implementation_route_invocation(
            route,
            invocation_raw,
            repo_root=repo_root,
            workspace=str(invocation_raw.get("workspace_path") or ""),
            now_ms=effect_started_at_ms,
            max_age_ms=max_age_ms,
            historical_effect_started_at_ms=effect_started_at_ms,
        )
        invocation = route.invocation_binding
        if (
            invocation is None
            or invocation.content_id != value.get("invocation_binding_id")
            or invocation.logical_attempt_id != value.get("logical_attempt_id")
        ):
            return None
        quota_evidence = None
        if decision_raw.get("verifier_status") == "confirmed_quota":
            quota_evidence = parse_agent_implementation_quota_evidence(
                quota_raw,
                failure_receipt=receipt,
                invocation_binding=invocation,
                now_ms=effect_started_at_ms,
                max_age_ms=max_age_ms,
                expected_signer_parent_pid=expected_signer_parent_pid,
                require_current_lifecycle=False,
            )
            if quota_evidence is None:
                return None
        elif quota_raw:
            return None
        decision = decide_agent_implementation_fallback(
            route,
            repo_root=repo_root,
            failure_receipt=receipt,
            expected_nonce=str(value.get("expected_nonce")),
            expected_model=str(value.get("expected_model")),
            expected_probe_returncode=expected_returncode,
            independent_quota_evidence=quota_evidence,
            expected_invocation_binding=invocation.signed_payload(),
            now_ms=effect_started_at_ms,
            max_age_ms=max_age_ms,
            historical_effect_started_at_ms=effect_started_at_ms,
        )
    except (OSError, TypeError, ValueError):
        return None
    if (
        decision.authorized is not True
        or decision.as_dict() != dict(decision_raw)
        or decision.content_id != value.get("decision_id")
    ):
        return None
    return AgentImplementationEffectAuthorizationContext(
        route=route,
        failure_receipt=dict(receipt),
        quota_evidence=quota_evidence,
        decision=decision,
        context_id=str(value.get("context_id")),
    )


def _agent_implementation_route_id(
    values: Mapping[str, object],
    *,
    authorization_id: str = "",
) -> str:
    body: dict[str, object] = dict(values)
    body["authorization_id"] = str(authorization_id or "")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _agent_implementation_route_plan(
    *,
    fallback_trigger: str,
    fallback_reasoning_effort: str,
    route_id: str,
) -> AgentImplementationRoutePlan:
    values = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": fallback_trigger,
        "fallback_reasoning_effort": fallback_reasoning_effort,
    }
    return AgentImplementationRoutePlan(
        **values,
        route_id=route_id,
    )


def agent_implementation_route_review_payload(
    *,
    board_namespace: str,
    authorization_kind: str,
    source_head: str,
    source_tree: str,
    route: Mapping[str, object],
    authority_bounds: Mapping[str, object],
    reviewer_identity: str,
    reviewer_provider: str,
    reviewer_profile_id: str = "",
    reviewer_profile_content_id: str = "",
    reviewer_lifecycle_anchor_id: str = "",
    reviewer_lifecycle_generation: int = 0,
    reviewer_witness_path: str = "",
    reviewer_witness_sha256: str = "",
    lifecycle_root_identity_did: str = "",
    lifecycle_witness_nonce: str = "",
    lifecycle_root_pin_path: str = "",
    lifecycle_root_pin_sha256: str = "",
    authorized_at_ms: int = 0,
    fallback_implementer_identity: str = "codex",
) -> dict[str, object]:
    """Return the exact canonical payload a non-Codex reviewer must sign."""

    return {
        "schema": _AGENT_ROUTE_REVIEW_SCHEMA,
        "board_namespace": str(board_namespace),
        "authorization_source": {
            "kind": str(authorization_kind),
            "source_head": str(source_head),
            "source_tree": str(source_tree),
        },
        "route": dict(route),
        "authority_bounds": dict(authority_bounds),
        "reviewer": {
            "identity": str(reviewer_identity),
            "provider": str(reviewer_provider),
            "profile_id": str(reviewer_profile_id),
            "profile_content_id": str(reviewer_profile_content_id),
            "lifecycle_anchor_id": str(reviewer_lifecycle_anchor_id),
            "generation": reviewer_lifecycle_generation,
            "witness_path": str(reviewer_witness_path),
            "witness_sha256": str(reviewer_witness_sha256),
        },
        "lifecycle_root_identity_did": str(lifecycle_root_identity_did),
        "lifecycle_witness_nonce": str(lifecycle_witness_nonce),
        "lifecycle_root_pin_path": str(lifecycle_root_pin_path),
        "lifecycle_root_pin_sha256": str(lifecycle_root_pin_sha256),
        "authorized_at_ms": authorized_at_ms,
        "fallback_implementer_identity": str(fallback_implementer_identity),
    }


def load_agent_implementation_route_authorization(
    *,
    repo_root: Path | str,
    artifact_path: str,
    board_namespace: str,
    expected_sha256: str = "",
    expected_authorization_id: str = "",
) -> AgentImplementationRouteAuthorization:
    """Load and validate the one namespace-scoped bootstrap authority.

    Every argument is explicit.  The loader neither searches for a policy nor
    infers a board from ambient environment state.
    """

    unresolved_root = resolve_agent_implementation_private_state_path(repo_root)
    try:
        root = unresolved_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("agent route authorization repository is unavailable") from exc
    if root != unresolved_root:
        raise ValueError("agent route authorization repository contains a symlink")
    relative = str(artifact_path or "").strip()
    namespace = str(board_namespace or "").strip()
    if (
        relative != _V3_AGENT_ROUTE_AUTHORIZATION_PATH
        or namespace != _V3_AGENT_ROUTE_BOARD_NAMESPACE
    ):
        raise ValueError(
            "auth-or-quota/high route is not authorized for this board scope"
        )
    unresolved_candidate = resolve_agent_implementation_private_state_path(
        root / relative
    )
    try:
        candidate = unresolved_candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "agent route authorization artifact is unavailable"
        ) from exc
    if candidate != unresolved_candidate or not candidate.is_relative_to(root):
        raise ValueError("agent route authorization artifact is unavailable")
    raw = _agent_read_stable_file(candidate, maximum_bytes=128 * 1024)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    expected_digest = str(expected_sha256 or "").strip()
    if expected_digest and (
        re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest) is None
        or digest != expected_digest
    ):
        raise ValueError("agent route authorization artifact digest drifted")

    def reject_duplicate_keys(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "agent route authorization contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("agent route authorization is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004
            "agent route authorization must be an object"
        )
    expected_top = {
        "schema",
        "board_namespace",
        "authorization_source",
        "route",
        "ownership_contract",
        "bootstrap_route_guarantees",
        "reviewer",
        "authority_bounds",
        "fallback_implementer_identity",
        "lifecycle_root_identity_did",
        "lifecycle_witness_nonce",
        "lifecycle_root_pin_path",
        "lifecycle_root_pin_sha256",
        "authorized_at_ms",
    }
    if set(payload) != expected_top:
        raise ValueError(
            "agent route authorization has noncanonical top-level fields"
        )
    source = payload.get("authorization_source")
    route = payload.get("route")
    ownership = payload.get("ownership_contract")
    bootstrap_guarantees = payload.get("bootstrap_route_guarantees")
    reviewer = payload.get("reviewer")
    authority_bounds_raw = payload.get("authority_bounds")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "kind",
            "source_head",
            "source_tree",
            "prospective_only",
            "requires_descendant_tree",
        }
        or not isinstance(route, dict)
        or set(route)
        != {
            "route_id",
            "primary_provider_id",
            "primary_model_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "allowed_trigger_classes",
        }
    ):
        raise ValueError(  # noqa: TRY004
            "agent route authorization fields are incomplete"
        )
    if (
        not isinstance(ownership, dict)
        or set(ownership)
        != {
            "canonical_route_plan_owner",
            "typed_fallback_decision_owner",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed",
        }
        or not isinstance(bootstrap_guarantees, dict)
        or set(bootstrap_guarantees)
        != {"explicit_codex_review_conflict_denied"}
    ):
        raise ValueError(  # noqa: TRY004
            "agent route authorization ownership is incomplete"
        )
    if (
        not isinstance(reviewer, dict)
        or set(reviewer)
        != {
            "identity",
            "provider",
            "profile_id",
            "profile_content_id",
            "lifecycle_anchor_id",
            "generation",
            "witness_path",
            "witness_sha256",
            "signature",
        }
    ):
        raise ValueError("agent route authorization reviewer is incomplete")
    if not isinstance(authority_bounds_raw, dict) or set(authority_bounds_raw) != {
        "repository_cid",
        "baseline_commit",
        "effects",
        "budget_cid",
        "resource_cid",
        "authority_cid",
    }:
        raise ValueError("agent route authorization bounds are incomplete")
    exact_text_values = (
        source.get("source_head"),
        source.get("source_tree"),
        source.get("kind"),
        reviewer.get("identity"),
        reviewer.get("provider"),
        reviewer.get("profile_id"),
        reviewer.get("profile_content_id"),
        reviewer.get("lifecycle_anchor_id"),
        reviewer.get("witness_path"),
        reviewer.get("witness_sha256"),
        reviewer.get("signature"),
        payload.get("fallback_implementer_identity"),
        payload.get("lifecycle_root_identity_did"),
        payload.get("lifecycle_witness_nonce"),
        payload.get("lifecycle_root_pin_path"),
        payload.get("lifecycle_root_pin_sha256"),
        *(route.get(name) for name in route if name != "allowed_trigger_classes"),
        *(
            authority_bounds_raw.get(name)
            for name in authority_bounds_raw
            if name != "effects"
        ),
    )
    if (
        any(not isinstance(value, str) for value in exact_text_values)
        or not isinstance(authority_bounds_raw.get("effects"), list)
        or any(
            not isinstance(value, str)
            for value in authority_bounds_raw.get("effects", [])
        )
        or not isinstance(route.get("allowed_trigger_classes"), list)
        or any(
            not isinstance(value, str)
            for value in route.get("allowed_trigger_classes", [])
        )
        or isinstance(reviewer.get("generation"), bool)
        or not isinstance(reviewer.get("generation"), int)
        or isinstance(payload.get("authorized_at_ms"), bool)
        or not isinstance(payload.get("authorized_at_ms"), int)
    ):
        raise ValueError("agent route authorization types are invalid")
    source_head = source["source_head"]
    source_tree = source["source_tree"]
    authorization_kind = source["kind"]
    reviewer_identity = reviewer["identity"]
    reviewer_provider = reviewer["provider"]
    reviewer_signature = reviewer["signature"]
    reviewer_profile_id = reviewer["profile_id"]
    reviewer_profile_content_id = reviewer["profile_content_id"]
    reviewer_lifecycle_anchor_id = reviewer["lifecycle_anchor_id"]
    reviewer_lifecycle_generation = reviewer["generation"]
    reviewer_witness_path = reviewer["witness_path"]
    reviewer_witness_sha256 = reviewer["witness_sha256"]
    lifecycle_root_identity_did = payload["lifecycle_root_identity_did"]
    lifecycle_witness_nonce = payload["lifecycle_witness_nonce"]
    lifecycle_root_pin_path = payload["lifecycle_root_pin_path"]
    lifecycle_root_pin_sha256 = payload["lifecycle_root_pin_sha256"]
    authorized_at_ms = payload["authorized_at_ms"]
    fallback_implementer_identity = payload["fallback_implementer_identity"]
    effects_raw = authority_bounds_raw.get("effects")
    try:
        authority_bounds = AgentImplementationAuthorityBounds(
            repository_cid=_agent_string(
                authority_bounds_raw.get("repository_cid"), "repository_cid"
            ),
            baseline_commit=_agent_string(
                authority_bounds_raw.get("baseline_commit"), "baseline_commit"
            ),
            effects=tuple(
                _agent_string(value, "effect")
                for value in (effects_raw if isinstance(effects_raw, list) else ())
            ),
            budget_cid=_agent_string(
                authority_bounds_raw.get("budget_cid"), "budget_cid"
            ),
            resource_cid=_agent_string(
                authority_bounds_raw.get("resource_cid"), "resource_cid"
            ),
            authority_cid=_agent_string(
                authority_bounds_raw.get("authority_cid"), "authority_cid"
            ),
        )
    except ValueError as exc:
        raise ValueError("agent route authorization bounds are invalid") from exc
    expected_route = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
    }
    if (
        payload.get("schema") != _AGENT_ROUTE_AUTHORIZATION_SCHEMA
        or payload.get("board_namespace") != namespace
        or {key: route.get(key) for key in expected_route} != expected_route
        or route.get("route_id") != _V3_AGENT_IMPLEMENTATION_ROUTE_ID
        or route.get("allowed_trigger_classes")
        != [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ]
        or authorization_kind != "explicit_operator_override"
        or source.get("prospective_only") is not True
        or source.get("requires_descendant_tree") is not True
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or ownership.get("canonical_route_plan_owner")
        != "ipfs_accelerate_py.llm_router"
        or ownership.get("typed_fallback_decision_owner")
        != "ipfs_accelerate_py.llm_router"
        or ownership.get("duplicate_route_policy_or_failure_classification_outside_router_allowed")
        is not False
        or bootstrap_guarantees.get(
            "explicit_codex_review_conflict_denied"
        )
        is not True
        or not reviewer_identity
        or reviewer_provider != "local_operator"
        or not reviewer_signature
        or not reviewer_profile_id
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", reviewer_profile_content_id
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", reviewer_lifecycle_anchor_id
        )
        is None
        or reviewer_lifecycle_generation < 1
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", reviewer_witness_sha256
        )
        is None
        or not lifecycle_root_identity_did.startswith("did:key:z")
        or not lifecycle_witness_nonce
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", lifecycle_root_pin_sha256
        )
        is None
        or authorized_at_ms <= 0
        or fallback_implementer_identity != "codex"
        or not authority_bounds.effects
        or len(set(authority_bounds.effects)) != len(authority_bounds.effects)
        or tuple(sorted(authority_bounds.effects)) != authority_bounds.effects
        or authority_bounds.baseline_commit != source_head
    ):
        raise ValueError(
            "agent route authorization does not grant the exact scoped route"
        )
    if lifecycle_root_pin_path != _V3_AGENT_LIFECYCLE_ROOT_PIN_PATH:
        raise ValueError("agent route lifecycle root pin path is invalid")
    unresolved_root_pin = resolve_agent_implementation_private_state_path(
        root / lifecycle_root_pin_path
    )
    try:
        root_pin_candidate = unresolved_root_pin.resolve(strict=True)
    except OSError as exc:
        raise ValueError("agent route lifecycle root pin is unavailable") from exc
    if (
        root_pin_candidate != unresolved_root_pin
        or not root_pin_candidate.is_relative_to(root)
    ):
        raise ValueError("agent route lifecycle root pin is unavailable")
    root_pin_raw = _agent_read_stable_file(
        root_pin_candidate,
        maximum_bytes=32 * 1024,
    )
    if (
        "sha256:" + hashlib.sha256(root_pin_raw).hexdigest()
        != lifecycle_root_pin_sha256
    ):
        raise ValueError("agent route lifecycle root pin digest drifted")
    try:
        root_pin = json.loads(
            root_pin_raw,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("agent route lifecycle root pin is invalid") from exc
    root_pin_expected = {
        "schema",
        "board_namespace",
        "base_head",
        "base_tree",
        "root_identity_did",
        "pinned_at_ms",
        "pin_id",
    }
    if (
        not isinstance(root_pin, dict)
        or set(root_pin) != root_pin_expected
        or any(
            not isinstance(root_pin.get(name), str)
            for name in root_pin_expected - {"pinned_at_ms"}
        )
        or isinstance(root_pin.get("pinned_at_ms"), bool)
        or not isinstance(root_pin.get("pinned_at_ms"), int)
        or int(root_pin.get("pinned_at_ms") or 0) <= 0
        or root_pin.get("schema") != _AGENT_LIFECYCLE_ROOT_PIN_SCHEMA
        or root_pin.get("board_namespace") != namespace
        or root_pin.get("root_identity_did") != lifecycle_root_identity_did
        or re.fullmatch(
            r"[0-9a-f]{40}", str(root_pin.get("base_head") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}", str(root_pin.get("base_tree") or "")
        )
        is None
        or root_pin.get("pin_id")
        != _content_addressed_mapping(root_pin, identity_field="pin_id")
    ):
        raise ValueError("agent route lifecycle root pin is invalid")
    witness_relative_path = Path(reviewer_witness_path)
    if (
        witness_relative_path.is_absolute()
        or ".." in witness_relative_path.parts
        or witness_relative_path.as_posix() != reviewer_witness_path
        or witness_relative_path.suffix != ".json"
        or not reviewer_witness_path.startswith(
            "data/agent_supervisor/prompt_only_self_improvement_v3/"
            "convergence/"
        )
        or reviewer_witness_path == relative
    ):
        raise ValueError("agent route lifecycle witness path is invalid")
    unresolved_witness = resolve_agent_implementation_private_state_path(
        root / reviewer_witness_path
    )
    try:
        witness_candidate = unresolved_witness.resolve(strict=True)
    except OSError as exc:
        raise ValueError("agent route lifecycle witness is unavailable") from exc
    if (
        witness_candidate != unresolved_witness
        or not witness_candidate.is_relative_to(root)
    ):
        raise ValueError("agent route lifecycle witness is unavailable")
    witness_raw = _agent_read_stable_file(
        witness_candidate,
        maximum_bytes=128 * 1024,
    )
    if (
        "sha256:" + hashlib.sha256(witness_raw).hexdigest()
        != reviewer_witness_sha256
    ):
        raise ValueError("agent route lifecycle witness digest drifted")
    try:
        witness_payload = json.loads(
            witness_raw,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("agent route lifecycle witness is invalid") from exc
    if not isinstance(witness_payload, dict):
        raise ValueError("agent route lifecycle witness is invalid")
    from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
        LocalProfileTampered,
        verify_local_profile_lifecycle_witness,
    )

    try:
        witness_profile = verify_local_profile_lifecycle_witness(
            witness_payload,
            expected_board_namespace=namespace,
            expected_base_head=source_head,
            expected_base_tree=source_tree,
            expected_nonce=lifecycle_witness_nonce,
            expected_root_identity_did=lifecycle_root_identity_did,
            reference_time_ms=authorized_at_ms,
            max_age_ms=10 * 60 * 1000,
        )
    except (LocalProfileTampered, TypeError, ValueError) as exc:
        raise ValueError("agent route lifecycle witness is invalid") from exc
    if (
        witness_profile.identity_did != reviewer_identity
        or witness_profile.reviewer_identity != reviewer_identity
        or witness_profile.reviewer_provider != reviewer_provider
        or witness_profile.profile_id != reviewer_profile_id
        or witness_profile.content_id != reviewer_profile_content_id
        or witness_profile.lifecycle_anchor_id
        != reviewer_lifecycle_anchor_id
        or witness_profile.lifecycle_generation
        != reviewer_lifecycle_generation
        or witness_profile.repository_cid != authority_bounds.repository_cid
        or witness_profile.baseline_commit != authority_bounds.baseline_commit
        or tuple(witness_profile.effect_bounds) != authority_bounds.effects
        or witness_profile.budget_cid != authority_bounds.budget_cid
        or witness_profile.resource_cid != authority_bounds.resource_cid
        or witness_profile.content_id != authority_bounds.authority_cid
        or witness_profile.route_id != _V3_AGENT_IMPLEMENTATION_ROUTE_ID
        or witness_profile.fallback_provider_id != "codex"
        or witness_profile.fallback_model_id != "gpt-5.6-terra"
        or witness_profile.fallback_reasoning_effort != "high"
    ):
        raise ValueError(
            "agent route lifecycle witness does not match reviewer authority"
        )
    review_payload = agent_implementation_route_review_payload(
        board_namespace=namespace,
        authorization_kind=authorization_kind,
        source_head=source_head,
        source_tree=source_tree,
        route=route,
        authority_bounds=authority_bounds.as_dict(),
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        reviewer_profile_id=reviewer_profile_id,
        reviewer_profile_content_id=reviewer_profile_content_id,
        reviewer_lifecycle_anchor_id=reviewer_lifecycle_anchor_id,
        reviewer_lifecycle_generation=reviewer_lifecycle_generation,
        reviewer_witness_path=reviewer_witness_path,
        reviewer_witness_sha256=reviewer_witness_sha256,
        lifecycle_root_identity_did=lifecycle_root_identity_did,
        lifecycle_witness_nonce=lifecycle_witness_nonce,
        lifecycle_root_pin_path=lifecycle_root_pin_path,
        lifecycle_root_pin_sha256=lifecycle_root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity=fallback_implementer_identity,
    )
    _agent_verify_did_signature(
        identity_did=reviewer_identity,
        payload=review_payload,
        signature=reviewer_signature,
    )
    try:
        top_level = Path(
            os.fsdecode(
                _agent_git_output(root, ("rev-parse", "--show-toplevel"))
            ).strip()
        ).resolve(strict=True)
        current_head = os.fsdecode(
            _agent_git_output(
                root, ("rev-parse", "--verify", "HEAD^{commit}")
            )
        ).strip()
        head_tree_entry = _agent_git_output(
            root,
            ("ls-tree", current_head, "--", relative),
            maximum_bytes=1024,
        ).rstrip(b"\n")
        head_artifact = _agent_git_output(
            root,
            ("show", f"{current_head}:{relative}"),
            maximum_bytes=128 * 1024,
        )
        head_witness_entry = _agent_git_output(
            root,
            ("ls-tree", current_head, "--", reviewer_witness_path),
            maximum_bytes=1024,
        ).rstrip(b"\n")
        head_witness = _agent_git_output(
            root,
            ("show", f"{current_head}:{reviewer_witness_path}"),
            maximum_bytes=128 * 1024,
        )
        head_root_pin_entry = _agent_git_output(
            root,
            ("ls-tree", current_head, "--", lifecycle_root_pin_path),
            maximum_bytes=1024,
        ).rstrip(b"\n")
        head_root_pin = _agent_git_output(
            root,
            ("show", f"{current_head}:{lifecycle_root_pin_path}"),
            maximum_bytes=32 * 1024,
        )
        observed_root_pin_base_tree = os.fsdecode(
            _agent_git_output(
                root,
                (
                    "rev-parse",
                    "--verify",
                    f"{root_pin['base_head']}^{{tree}}",
                ),
            )
        ).strip()
        _agent_git_output(
            root,
            (
                "merge-base",
                "--is-ancestor",
                str(root_pin["base_head"]),
                current_head,
            ),
            maximum_bytes=64,
        )
        authorization_commit_seconds_raw = _agent_git_output(
            root,
            (
                "log",
                "-1",
                "--format=%ct",
                current_head,
                "--",
                relative,
            ),
            maximum_bytes=64,
        ).strip()
        authorization_reference_ms = (
            int(authorization_commit_seconds_raw) * 1000
        )
        observed_source_tree = os.fsdecode(
            _agent_git_output(
                root,
                ("rev-parse", "--verify", f"{source_head}^{{tree}}"),
            )
        ).strip()
        _agent_git_output(
            root,
            ("merge-base", "--is-ancestor", source_head, current_head),
            maximum_bytes=64,
        )
        final_head = os.fsdecode(
            _agent_git_output(
                root, ("rev-parse", "--verify", "HEAD^{commit}")
            )
        ).strip()
        final_raw = _agent_read_stable_file(
            unresolved_candidate,
            maximum_bytes=128 * 1024,
        )
        final_witness_raw = _agent_read_stable_file(
            unresolved_witness,
            maximum_bytes=128 * 1024,
        )
        final_root_pin_raw = _agent_read_stable_file(
            unresolved_root_pin,
            maximum_bytes=32 * 1024,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "agent route authorization repository binding is unavailable"
        ) from exc
    if (
        top_level != root
        or re.fullmatch(r"[0-9a-f]{40}", current_head) is None
        or re.fullmatch(
            rb"100(?:644|755) blob [0-9a-f]{40}\t" + re.escape(relative.encode()),
            head_tree_entry,
        )
        is None
        or head_artifact != raw
        or re.fullmatch(
            rb"100(?:644|755) blob [0-9a-f]{40}\t"
            + re.escape(reviewer_witness_path.encode()),
            head_witness_entry,
        )
        is None
        or head_witness != witness_raw
        or re.fullmatch(
            rb"100(?:644|755) blob [0-9a-f]{40}\t"
            + re.escape(lifecycle_root_pin_path.encode()),
            head_root_pin_entry,
        )
        is None
        or head_root_pin != root_pin_raw
        or observed_root_pin_base_tree != root_pin.get("base_tree")
        or observed_source_tree != source_tree
        or final_head != current_head
        or final_raw != raw
        or final_witness_raw != witness_raw
        or final_root_pin_raw != root_pin_raw
    ):
        raise ValueError(
            "agent route authorization is not bound to this descendant tree"
        )
    try:
        if abs(authorization_reference_ms - authorized_at_ms) > (
            10 * 60 * 1000
        ):
            raise ValueError("authorization time is outside its commit window")
        verify_local_profile_lifecycle_witness(
            witness_payload,
            expected_board_namespace=namespace,
            expected_base_head=source_head,
            expected_base_tree=source_tree,
            expected_nonce=lifecycle_witness_nonce,
            expected_root_identity_did=lifecycle_root_identity_did,
            reference_time_ms=authorized_at_ms,
            max_age_ms=10 * 60 * 1000,
        )
    except (LocalProfileTampered, TypeError, ValueError) as exc:
        raise ValueError(
            "agent route lifecycle witness is stale at authorization commit"
        ) from exc
    identity_body = {
        "schema": _AGENT_ROUTE_AUTHORIZATION_SCHEMA,
        "board_namespace": namespace,
        "artifact_path": relative,
        "artifact_sha256": digest,
        "authorization_kind": authorization_kind,
        "source_head": source_head,
        "source_tree": source_tree,
        "reviewer_identity": reviewer_identity,
        "reviewer_provider": reviewer_provider,
        "reviewer_signature": reviewer_signature,
        "reviewer_profile_id": reviewer_profile_id,
        "reviewer_profile_content_id": reviewer_profile_content_id,
        "reviewer_lifecycle_anchor_id": reviewer_lifecycle_anchor_id,
        "reviewer_lifecycle_generation": reviewer_lifecycle_generation,
        "reviewer_witness_path": reviewer_witness_path,
        "reviewer_witness_sha256": reviewer_witness_sha256,
        "lifecycle_root_identity_did": lifecycle_root_identity_did,
        "lifecycle_witness_nonce": lifecycle_witness_nonce,
        "lifecycle_root_pin_path": lifecycle_root_pin_path,
        "lifecycle_root_pin_sha256": lifecycle_root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
        "fallback_implementer_identity": fallback_implementer_identity,
        "authority_bounds": authority_bounds.as_dict(),
    }
    authorization_id = _agent_implementation_route_id(identity_body)
    expected_identity = str(expected_authorization_id or "").strip()
    if expected_identity and authorization_id != expected_identity:
        raise ValueError("agent route authorization identity drifted")
    return AgentImplementationRouteAuthorization(
        board_namespace=namespace,
        artifact_path=relative,
        artifact_sha256=digest,
        authorization_kind=authorization_kind,
        source_head=source_head,
        source_tree=source_tree,
        authorization_id=authorization_id,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        reviewer_signature=reviewer_signature,
        reviewer_profile_id=reviewer_profile_id,
        reviewer_profile_content_id=reviewer_profile_content_id,
        reviewer_lifecycle_anchor_id=reviewer_lifecycle_anchor_id,
        reviewer_lifecycle_generation=reviewer_lifecycle_generation,
        reviewer_witness_path=reviewer_witness_path,
        reviewer_witness_sha256=reviewer_witness_sha256,
        lifecycle_root_identity_did=lifecycle_root_identity_did,
        lifecycle_witness_nonce=lifecycle_witness_nonce,
        lifecycle_root_pin_path=lifecycle_root_pin_path,
        lifecycle_root_pin_sha256=lifecycle_root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity=fallback_implementer_identity,
        authority_bounds=authority_bounds,
        _validation_seal=_agent_implementation_private_seal(identity_body),
    )


_LEGACY_AGENT_IMPLEMENTATION_ROUTE = _agent_implementation_route_plan(
    fallback_trigger="primary_quota_exhausted",
    fallback_reasoning_effort="medium",
    route_id=_LEGACY_AGENT_IMPLEMENTATION_ROUTE_ID,
)
_QUOTA_HIGH_AGENT_IMPLEMENTATION_ROUTE = _agent_implementation_route_plan(
    fallback_trigger="primary_quota_exhausted",
    fallback_reasoning_effort="high",
    route_id=_QUOTA_HIGH_AGENT_IMPLEMENTATION_ROUTE_ID,
)
_AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE = (
    _agent_implementation_route_plan(
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        route_id=_V3_AGENT_IMPLEMENTATION_ROUTE_ID,
    )
)
_AGENT_IMPLEMENTATION_ROUTES = (
    _LEGACY_AGENT_IMPLEMENTATION_ROUTE,
    _QUOTA_HIGH_AGENT_IMPLEMENTATION_ROUTE,
    _AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE,
)


def create_legacy_agent_implementation_route_invocation(
) -> AgentImplementationRouteInvocation:
    """Create the canonical legacy plan with a fresh nonce for this process."""

    return AgentImplementationRouteInvocation(
        route_plan=_LEGACY_AGENT_IMPLEMENTATION_ROUTE,
        failure_receipt_nonce=secrets.token_hex(32),
    )


def resolve_agent_implementation_route(
    *,
    primary_provider_id: str = "",
    primary_model_id: str = "",
    fallback_provider_id: str = "",
    fallback_model_id: str = "",
    fallback_trigger: str = "",
    fallback_reasoning_effort: str = "",
    default_route: str | None = None,
    authorization: AgentImplementationRouteAuthorization | None = None,
) -> AgentImplementationRoutePlan:
    """Resolve exactly one reviewed agent route without reading ambient state.

    ``default_route="legacy"`` is an explicit caller choice used by legacy
    launchers to fill otherwise absent compatible fields.  With no default,
    all six fields are mandatory.  Hybrid tuples and unknown policy values
    always fail closed.
    """

    values = {
        "primary_provider_id": str(primary_provider_id or "").strip(),
        "primary_model_id": str(primary_model_id or "").strip(),
        "fallback_provider_id": str(fallback_provider_id or "").strip(),
        "fallback_model_id": str(fallback_model_id or "").strip(),
        "fallback_trigger": str(fallback_trigger or "").strip(),
        "fallback_reasoning_effort": str(
            fallback_reasoning_effort or ""
        ).strip(),
    }
    if values["primary_provider_id"].lower() in (
        _AGENT_IMPLEMENTATION_GROK_ALIASES
    ):
        values["primary_provider_id"] = "grok_cli"
    if default_route is not None:
        if default_route != "legacy":
            raise ValueError("unknown agent implementation route default")
        legacy = _LEGACY_AGENT_IMPLEMENTATION_ROUTE.as_dict()
        values = {
            field: value or legacy[field]
            for field, value in values.items()
        }
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise ValueError(
            "agent implementation route requires a complete six-field tuple; "
            "missing " + ", ".join(missing)
        )
    for route in _AGENT_IMPLEMENTATION_ROUTES:
        if values == route.as_dict():
            if route.permits_authentication_unavailable:
                if (
                    authorization is None
                    or authorization._validation_seal
                    != _agent_implementation_private_seal(
                        {
                            "schema": _AGENT_ROUTE_AUTHORIZATION_SCHEMA,
                            "board_namespace": authorization.board_namespace,
                            "artifact_path": authorization.artifact_path,
                            "artifact_sha256": authorization.artifact_sha256,
                            "authorization_kind": (
                                authorization.authorization_kind
                            ),
                            "source_head": authorization.source_head,
                            "source_tree": authorization.source_tree,
                            "reviewer_identity": authorization.reviewer_identity,
                            "reviewer_provider": authorization.reviewer_provider,
                            "reviewer_signature": authorization.reviewer_signature,
                            "reviewer_profile_id": (
                                authorization.reviewer_profile_id
                            ),
                            "reviewer_profile_content_id": (
                                authorization.reviewer_profile_content_id
                            ),
                            "reviewer_lifecycle_anchor_id": (
                                authorization.reviewer_lifecycle_anchor_id
                            ),
                            "reviewer_lifecycle_generation": (
                                authorization.reviewer_lifecycle_generation
                            ),
                            "reviewer_witness_path": (
                                authorization.reviewer_witness_path
                            ),
                            "reviewer_witness_sha256": (
                                authorization.reviewer_witness_sha256
                            ),
                            "lifecycle_root_identity_did": (
                                authorization.lifecycle_root_identity_did
                            ),
                            "lifecycle_witness_nonce": (
                                authorization.lifecycle_witness_nonce
                            ),
                            "lifecycle_root_pin_path": (
                                authorization.lifecycle_root_pin_path
                            ),
                            "lifecycle_root_pin_sha256": (
                                authorization.lifecycle_root_pin_sha256
                            ),
                            "authorized_at_ms": authorization.authorized_at_ms,
                            "fallback_implementer_identity": (
                                authorization.fallback_implementer_identity
                            ),
                            "authority_bounds": (
                                authorization.authority_bounds.as_dict()
                                if authorization.authority_bounds is not None
                                else None
                            ),
                        }
                    )
                    or not authorization.reviewer_identity
                    or not authorization.reviewer_provider
                    or authorization.reviewer_provider in {"codex", "openai"}
                    or not authorization.reviewer_signature
                    or authorization.authority_bounds is None
                ):
                    raise ValueError(
                        "auth-or-quota/high route requires scoped operator "
                        "authorization"
                    )
                return AgentImplementationRoutePlan(
                    **values,
                    route_id=_V3_AGENT_IMPLEMENTATION_ROUTE_ID,
                    authorization=authorization,
                    fallback_implementer_identity=(
                        authorization.fallback_implementer_identity
                    ),
                )
            if authorization is not None:
                raise ValueError(
                    "reviewed legacy quota route cannot carry auth authority"
                )
            return route
    details = ", ".join(
        f"{field}={values[field]!r}"
        for field in _AGENT_IMPLEMENTATION_ROUTE_FIELDS
    )
    raise ValueError(
        "agent implementation route must be exactly the reviewed legacy "
        "quota/medium tuple, quota/high tuple, or auth-or-quota/high "
        "tuple; " + details
    )


def _agent_read_stable_file(
    path: Path,
    *,
    maximum_bytes: int = _AGENT_CONTROL_PLANE_MAX_FILE_BYTES,
    exact_mode: int | None = None,
) -> bytes:
    """Read a bounded stable file through exactly one no-follow descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("accepted control-plane no-follow reads are unavailable")
    lexical = resolve_agent_implementation_private_state_path(path)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    parent_descriptor = os.open(lexical.anchor, directory_flags)
    try:
        for component in lexical.parts[1:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = child
        descriptor = os.open(
            lexical.name,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ValueError("accepted control-plane file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid not in {0, os.geteuid()}
            or (
                stat_module.S_IMODE(before.st_mode) != exact_mode
                if exact_mode is not None
                else bool(stat_module.S_IMODE(before.st_mode) & 0o022)
            )
            or before.st_size > maximum_bytes
        ):
            raise ValueError("accepted control-plane file is not immutable enough")

        remaining = maximum_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        final = os.stat(
            lexical.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError("accepted control-plane file is unavailable") from exc
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)

    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        not stat_module.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_uid not in {0, os.geteuid()}
        or (
            stat_module.S_IMODE(after.st_mode) != exact_mode
            if exact_mode is not None
            else bool(stat_module.S_IMODE(after.st_mode) & 0o022)
        )
        or len(raw) > maximum_bytes
        or len(raw) != after.st_size
        or not (identity(before) == identity(after) == identity(final))
    ):
        raise ValueError("accepted control-plane file is not immutable enough")
    return raw


def _agent_file_digest(path: Path, *, exact_mode: int | None = None) -> str:
    """Hash one bounded, stable, no-follow control-plane file."""

    raw = _agent_read_stable_file(path, exact_mode=exact_mode)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _agent_control_plane_source_files(
    root: Path,
    *,
    verify_loaded_origins: bool = True,
) -> tuple[Path, ...]:
    """Return every source that can enter the scoped supervisor process.

    The daemon has a deliberately broad import graph.  Maintaining a hand-made
    transitive list would silently lose coverage when that graph grows, so the
    accepted capsule binds the entire supervisor Python tree and then verifies
    the origins of every already-imported supervisor module against that tree.
    """

    supervisor_root = root / "ipfs_accelerate_py" / "agent_supervisor"
    try:
        entries = tuple(supervisor_root.rglob("*"))
        for entry in entries:
            if stat_module.S_ISLNK(os.lstat(entry).st_mode):
                raise ValueError(
                    "accepted control-plane package contains a symlink"
                )
    except OSError as exc:
        raise ValueError(
            "accepted control-plane package tree is unavailable"
        ) from exc
    tree_files = tuple(
        entry for entry in entries if entry.suffix == ".py"
    )
    required_files = tuple(
        root / relative for relative in _AGENT_CONTROL_PLANE_RELATIVE_FILES
    )
    files = tuple(
        sorted({*required_files, *tree_files}, key=lambda item: str(item))
    )

    if not verify_loaded_origins:
        return files
    file_set = set(files)
    observed_modules = tuple(sys.modules.items())
    exact_modules = {
        "ipfs_accelerate_py",
        "ipfs_accelerate_py.llm_router",
        "ipfs_accelerate_py.agent_implementation_route",
        "ipfs_accelerate_py.router_deps",
        "ipfs_accelerate_py.common",
        "ipfs_accelerate_py.common.meta_model_api",
        "ipfs_accelerate_py.model_catalog",
        "ipfs_accelerate_py.model_catalog.identity",
        "ipfs_accelerate_py.model_catalog.schema",
        "ipfs_accelerate_py.utils",
        "ipfs_accelerate_py.utils.mistral_vibe",
        "ipfs_accelerate_py.utils.cid_utils",
    }
    for module_name, module in observed_modules:
        if not (
            module_name in exact_modules
            or module_name.startswith("ipfs_accelerate_py.agent_supervisor.")
            or module_name == "ipfs_accelerate_py.agent_supervisor"
        ):
            continue
        origin = getattr(module, "__file__", None)
        try:
            origin_path = Path(str(origin)).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "accepted control-plane module origin is unavailable"
            ) from exc
        if origin_path not in file_set:
            raise ValueError(
                "accepted control-plane module crossed capsule roots"
            )
    return files


def _agent_private_directory(path: Path, *, mode: int) -> None:
    """Verify an exact owned, no-follow directory and its final path identity."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("accepted control-plane no-follow checks are unavailable")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
        )
    except OSError as exc:
        raise ValueError("accepted control-plane directory is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        final = os.lstat(path)
    except OSError as exc:
        raise ValueError("accepted control-plane directory changed") from exc
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        not stat_module.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat_module.S_IMODE(opened.st_mode) != mode
        or identity(opened) != identity(final)
    ):
        raise ValueError("accepted control-plane directory is not private")


def _agent_create_private_directory_chain(
    path: Path,
    *,
    final_mode: int,
) -> tuple[int, int, int, int]:
    """Create/open a private path component-by-component through dirfds."""

    candidate = resolve_agent_implementation_private_state_path(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError(
            "accepted control-plane no-follow creation is unavailable"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    descriptor = os.open(candidate.anchor, flags)
    try:
        for index, component in enumerate(candidate.parts[1:], start=1):
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                mode = final_mode if index == len(candidate.parts) - 1 else 0o700
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
    except OSError as exc:
        raise ValueError(
            "accepted control-plane directory cannot be created"
        ) from exc
    finally:
        os.close(descriptor)


def _agent_write_capsule_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ValueError("accepted control-plane snapshot write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _agent_manifest_json(raw: bytes) -> dict[str, object]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("control-plane manifest has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("accepted control-plane manifest is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("accepted control-plane manifest is invalid")
    return value


def _agent_git_output(
    root: Path,
    arguments: Sequence[str],
    *,
    maximum_bytes: int = _AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES,
) -> bytes:
    """Run one bounded, non-interactive Git identity/object query."""

    git_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    git_environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.quotepath=false",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                *arguments,
            ],
            cwd=root,
            env=git_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("accepted control-plane Git query failed") from exc
    if completed.returncode != 0 or len(completed.stdout) > maximum_bytes:
        raise ValueError("accepted control-plane Git query failed")
    return completed.stdout


def _agent_control_plane_git_state(
    root: Path,
    *,
    expected_head: str,
    expected_tree: str,
) -> tuple[str, str]:
    """Require one exact, clean repository generation at ``root``."""

    top_level = Path(
        os.fsdecode(
            _agent_git_output(root, ("rev-parse", "--show-toplevel"))
        ).strip()
    )
    try:
        exact_top_level = top_level.resolve(strict=True)
    except OSError as exc:
        raise ValueError("accepted control-plane Git root is unavailable") from exc
    head = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    ).strip()
    tree = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{tree}"))
    ).strip()
    status = _agent_git_output(
        root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
    )
    if (
        exact_top_level != root
        or head != expected_head
        or tree != expected_tree
        or status
    ):
        raise ValueError(
            "accepted control-plane source is not the exact clean Git generation"
        )
    return head, tree


def _agent_control_plane_head_payloads(
    root: Path,
    *,
    source_head: str,
) -> dict[str, bytes]:
    """Read the capsule exclusively from immutable blobs at one HEAD."""

    tree_output = _agent_git_output(
        root,
        (
            "ls-tree",
            "-r",
            "-z",
            source_head,
            "--",
            "ipfs_accelerate_py",
            "scripts/ops/agent_supervisor/configured_board_scheduler.py",
            "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        ),
        maximum_bytes=_AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES,
    )
    blobs: dict[str, str] = {}
    for raw_entry in tree_output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ValueError("accepted control-plane Git tree is invalid") from exc
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("accepted control-plane Git tree entry is invalid")
        selected = (
            relative in _AGENT_CONTROL_PLANE_RELATIVE_FILES
            or (
                relative.startswith("ipfs_accelerate_py/agent_supervisor/")
                and relative.endswith(".py")
            )
        )
        if object_type != "blob" or mode not in {"100644", "100755"}:
            if selected:
                raise ValueError(
                    "accepted control-plane Git dependency is invalid"
                )
            continue
        if re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
            raise ValueError("accepted control-plane Git tree entry is invalid")
        if selected:
            blobs[relative] = object_id
    required = set(_AGENT_CONTROL_PLANE_RELATIVE_FILES)
    if not required.issubset(blobs):
        raise ValueError("accepted control-plane HEAD is missing a dependency")
    payloads: dict[str, bytes] = {}
    for relative, object_id in sorted(blobs.items()):
        size_raw = _agent_git_output(
            root,
            ("cat-file", "-s", object_id),
            maximum_bytes=64,
        )
        try:
            size = int(size_raw.strip())
        except ValueError as exc:
            raise ValueError("accepted control-plane Git blob size is invalid") from exc
        if not 0 <= size <= _AGENT_CONTROL_PLANE_MAX_FILE_BYTES:
            raise ValueError("accepted control-plane Git blob is oversized")
        payload = _agent_git_output(
            root,
            ("cat-file", "blob", object_id),
            maximum_bytes=_AGENT_CONTROL_PLANE_MAX_FILE_BYTES,
        )
        if len(payload) != size:
            raise ValueError("accepted control-plane Git blob changed")
        payloads[relative] = payload
    return payloads


def materialize_agent_implementation_control_plane_capsule(
    *,
    source_root: Path | str,
    capsule_parent: Path | str,
    source_head: str,
    source_tree: str,
) -> AgentImplementationControlPlanePin:
    """Snapshot the daemon's loaded source generation into a private capsule."""

    unresolved_root = resolve_agent_implementation_private_state_path(source_root)
    root = unresolved_root.resolve(strict=True)
    if root != unresolved_root:
        raise ValueError("accepted control-plane source root contains a symlink")
    if re.fullmatch(r"[0-9a-f]{40}", source_head) is None or re.fullmatch(
        r"[0-9a-f]{40}", source_tree
    ) is None:
        raise ValueError("accepted control-plane Git identity is invalid")
    _agent_control_plane_git_state(
        root,
        expected_head=source_head,
        expected_tree=source_tree,
    )
    files = _agent_control_plane_source_files(root, verify_loaded_origins=True)
    payloads = _agent_control_plane_head_payloads(
        root,
        source_head=source_head,
    )
    disk_relatives = {str(path.relative_to(root)) for path in files}
    if disk_relatives != set(payloads):
        raise ValueError("accepted control-plane package differs from HEAD")
    for path in files:
        relative = str(path.relative_to(root))
        if _agent_read_stable_file(path) != payloads[relative]:
            raise ValueError("loaded control-plane module differs from HEAD")
    _agent_control_plane_git_state(
        root,
        expected_head=source_head,
        expected_tree=source_tree,
    )
    digests = {
        relative: "sha256:" + hashlib.sha256(raw).hexdigest()
        for relative, raw in payloads.items()
    }
    manifest: dict[str, object] = {
        "schema": _AGENT_CONTROL_PLANE_MANIFEST_SCHEMA,
        "source_head": source_head,
        "source_tree": source_tree,
        "files": dict(sorted(digests.items())),
    }
    manifest["capsule_id"] = _content_addressed_mapping(
        manifest,
        identity_field="capsule_id",
    )
    capsule_id = str(manifest["capsule_id"])
    parent = resolve_agent_implementation_private_state_path(capsule_parent)
    created_parent_identity: tuple[int, int, int, int] | None = None
    try:
        os.lstat(parent)
    except FileNotFoundError:
        created_parent_identity = _agent_create_private_directory_chain(
            parent,
            final_mode=0o700,
        )
    except OSError as exc:
        raise ValueError("accepted control-plane parent is unavailable") from exc
    if resolve_agent_implementation_private_state_path(parent) != parent:
        raise ValueError("accepted control-plane parent changed during creation")
    _agent_private_directory(parent, mode=0o700)
    if created_parent_identity is not None:
        observed_parent = os.lstat(parent)
        if (
            observed_parent.st_dev,
            observed_parent.st_ino,
            observed_parent.st_mode,
            observed_parent.st_uid,
        ) != created_parent_identity:
            raise ValueError(
                "accepted control-plane parent changed during creation"
            )
    destination = parent / capsule_id.removeprefix("sha256:")
    try:
        destination_metadata = os.lstat(destination)
    except FileNotFoundError:
        destination_metadata = None
    except OSError as exc:
        raise ValueError("accepted control-plane capsule is unavailable") from exc
    if destination_metadata is not None:
        if stat_module.S_ISLNK(destination_metadata.st_mode):
            raise ValueError("accepted control-plane capsule cannot be a symlink")
        return build_agent_implementation_control_plane_pin(
            runner_path=(
                destination
                / "ipfs_accelerate_py"
                / "agent_supervisor"
                / "runtime"
                / "grok_cli_runner.py"
            ),
            capsule_root=destination,
        )
    staging = Path(tempfile.mkdtemp(prefix=".capsule-", dir=parent))
    try:
        for relative, raw in payloads.items():
            _agent_write_capsule_file(staging / relative, raw)
        encoded_manifest = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        _agent_write_capsule_file(
            staging / _AGENT_CONTROL_PLANE_MANIFEST_FILENAME,
            encoded_manifest,
        )
        directories = sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            os.chmod(directory, 0o500)
        os.chmod(staging, 0o500)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            os.chmod(staging, 0o700)
            for directory in directories:
                try:
                    os.chmod(directory, 0o700)
                except FileNotFoundError:
                    pass
            shutil.rmtree(staging)
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if staging.exists():
            try:
                os.chmod(staging, 0o700)
            except OSError:
                pass
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return build_agent_implementation_control_plane_pin(
        runner_path=(
            destination
            / "ipfs_accelerate_py"
            / "agent_supervisor"
            / "runtime"
            / "grok_cli_runner.py"
        ),
        capsule_root=destination,
    )


def agent_implementation_control_plane_source_generation(
    source_root: Path | str,
) -> tuple[str, str]:
    """Return one sanitized clean HEAD/tree pair for capsule construction."""

    unresolved = resolve_agent_implementation_private_state_path(source_root)
    root = unresolved.resolve(strict=True)
    if root != unresolved:
        raise ValueError("accepted control-plane source root contains a symlink")
    head = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    ).strip()
    tree = os.fsdecode(
        _agent_git_output(root, ("rev-parse", "--verify", "HEAD^{tree}"))
    ).strip()
    _agent_control_plane_git_state(
        root,
        expected_head=head,
        expected_tree=tree,
    )
    return head, tree


def _agent_control_plane_capsule_id(
    *, capsule_root: Path, runner_path: Path
) -> str:
    return build_agent_implementation_control_plane_pin(
        runner_path=runner_path,
        capsule_root=capsule_root,
    ).capsule_id


def _agent_control_plane_archive_bytes(
    root: Path,
    *,
    manifest_raw: bytes,
    expected_digests: Mapping[Path, str],
) -> bytes:
    """Build a deterministic importable archive from stable capsule bytes."""

    main = (
        b"import runpy\n"
        b"runpy.run_module("
        b"'ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner',"
        b"run_name='__main__')\n"
    )
    entries: dict[str, bytes] = {
        "__main__.py": main,
        _AGENT_CONTROL_PLANE_MANIFEST_FILENAME: manifest_raw,
    }
    total = len(main) + len(manifest_raw)
    for relative, digest in sorted(
        expected_digests.items(), key=lambda item: item[0].as_posix()
    ):
        raw = _agent_read_stable_file(
            root / relative,
            maximum_bytes=_AGENT_CONTROL_PLANE_MAX_FILE_BYTES,
            exact_mode=0o400,
        )
        if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("accepted control-plane archive input drifted")
        total += len(raw)
        if total > _AGENT_CONTROL_PLANE_MAX_ARCHIVE_BYTES:
            raise ValueError("accepted control-plane archive is oversized")
        entries[relative.as_posix()] = raw
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for name, raw in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100400 << 16
            archive.writestr(info, raw)
    payload = stream.getvalue()
    if not payload or len(payload) > _AGENT_CONTROL_PLANE_MAX_ARCHIVE_BYTES:
        raise ValueError("accepted control-plane archive is oversized")
    return payload


def build_agent_implementation_control_plane_pin(
    *, runner_path: Path | str, capsule_root: Path | str
) -> AgentImplementationControlPlanePin:
    """Validate and pin one already-materialized private source capsule."""

    unresolved_root = resolve_agent_implementation_private_state_path(
        capsule_root
    )
    unresolved_runner = resolve_agent_implementation_private_state_path(
        runner_path
    )
    root = unresolved_root.resolve(strict=True)
    runner = unresolved_runner.resolve(strict=True)
    if root != unresolved_root or runner != unresolved_runner:
        raise ValueError("accepted control-plane capsule cannot contain symlinks")
    if not runner.is_absolute() or not root.is_absolute() or not runner.is_relative_to(root):
        raise ValueError("accepted control-plane paths must share one absolute capsule")
    _agent_private_directory(root, mode=0o500)
    manifest_path = root / _AGENT_CONTROL_PLANE_MANIFEST_FILENAME
    manifest_raw = _agent_read_stable_file(
        manifest_path,
        maximum_bytes=_AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES,
        exact_mode=0o400,
    )
    manifest = _agent_manifest_json(manifest_raw)
    if set(manifest) != {
        "schema",
        "source_head",
        "source_tree",
        "files",
        "capsule_id",
    }:
        raise ValueError("accepted control-plane manifest fields are invalid")
    files = manifest.get("files")
    if (
        manifest.get("schema") != _AGENT_CONTROL_PLANE_MANIFEST_SCHEMA
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_head") or ""))
        is None
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_tree") or ""))
        is None
        or not isinstance(files, dict)
        or not files
        or len(files) > 2048
        or manifest.get("capsule_id")
        != _content_addressed_mapping(manifest, identity_field="capsule_id")
    ):
        raise ValueError("accepted control-plane manifest identity is invalid")
    if not set(_AGENT_CONTROL_PLANE_RELATIVE_FILES).issubset(files):
        raise ValueError("accepted control-plane HEAD is missing a dependency")
    expected_entries = {Path(_AGENT_CONTROL_PLANE_MANIFEST_FILENAME)}
    expected_digests: dict[Path, str] = {}
    for relative_text, digest in files.items():
        if (
            not isinstance(relative_text, str)
            or not relative_text
            or Path(relative_text).is_absolute()
            or ".." in Path(relative_text).parts
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest or "")) is None
        ):
            raise ValueError("accepted control-plane manifest path is invalid")
        relative = Path(relative_text)
        expected_entries.add(relative)
        expected_digests[relative] = str(digest)
    actual_files: set[Path] = set()
    for entry in root.rglob("*"):
        metadata = os.lstat(entry)
        if stat_module.S_ISLNK(metadata.st_mode):
            raise ValueError("accepted control-plane capsule contains a symlink")
        relative = entry.relative_to(root)
        if stat_module.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or stat_module.S_IMODE(
                metadata.st_mode
            ) != 0o500:
                raise ValueError("accepted control-plane capsule directory is dirty")
            continue
        actual_files.add(relative)
    if actual_files != expected_entries:
        raise ValueError("accepted control-plane capsule contents are dirty")
    for relative, digest in expected_digests.items():
        if _agent_file_digest(root / relative, exact_mode=0o400) != digest:
            raise ValueError("accepted control-plane capsule content drifted")
    expected_runner = (
        root
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "runtime"
        / "grok_cli_runner.py"
    )
    if runner != expected_runner or str(runner.relative_to(root)) not in files:
        raise ValueError("accepted control-plane runner is not in its manifest")
    archive = _agent_control_plane_archive_bytes(
        root,
        manifest_raw=manifest_raw,
        expected_digests=expected_digests,
    )
    return AgentImplementationControlPlanePin(
        schema=_AGENT_CONTROL_PLANE_PIN_SCHEMA,
        runner_path=str(runner),
        runner_sha256=_agent_file_digest(runner, exact_mode=0o400),
        capsule_root=str(root),
        capsule_id=str(manifest["capsule_id"]),
        source_head=str(manifest["source_head"]),
        source_tree=str(manifest["source_tree"]),
        archive_sha256="sha256:" + hashlib.sha256(archive).hexdigest(),
    )


def verify_agent_implementation_sealed_control_plane(
    pin: AgentImplementationControlPlanePin,
    descriptor: int,
) -> str:
    """Verify a sealed memfd archive and return its isolated executable path."""

    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 3:
        raise ValueError("accepted control-plane descriptor is invalid")
    required_names = (
        "F_GET_SEALS",
        "F_SEAL_WRITE",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
    )
    if any(not hasattr(fcntl, name) for name in required_names):
        raise ValueError("accepted control-plane sealing is unavailable")
    required = (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SEAL
    )
    try:
        before = os.fstat(descriptor)
        seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
        if (
            not stat_module.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _AGENT_CONTROL_PLANE_MAX_ARCHIVE_BYTES
            or seals & required != required
        ):
            raise ValueError("accepted control-plane descriptor is not sealed")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(64 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("accepted control-plane descriptor is unavailable") from exc
    archive = b"".join(chunks)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        len(archive) != before.st_size
        or identity(before) != identity(after)
        or "sha256:" + hashlib.sha256(archive).hexdigest()
        != pin.archive_sha256
    ):
        raise ValueError("accepted control-plane sealed archive drifted")
    executable = f"/proc/self/fd/{descriptor}"
    try:
        path_stat = os.stat(executable)
    except OSError as exc:
        raise ValueError("accepted control-plane descriptor path is unavailable") from exc
    if (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino):
        raise ValueError("accepted control-plane descriptor path drifted")
    return executable


def seal_agent_implementation_control_plane_capsule(
    pin: AgentImplementationControlPlanePin,
) -> AgentImplementationSealedControlPlane:
    """Copy a validated capsule into a write-sealed deterministic zipapp."""

    verified = build_agent_implementation_control_plane_pin(
        runner_path=pin.runner_path,
        capsule_root=pin.capsule_root,
    )
    if verified != pin:
        raise ValueError("accepted control-plane pin drifted before sealing")
    root = Path(pin.capsule_root)
    manifest_raw = _agent_read_stable_file(
        root / _AGENT_CONTROL_PLANE_MANIFEST_FILENAME,
        maximum_bytes=_AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES,
        exact_mode=0o400,
    )
    manifest = _agent_manifest_json(manifest_raw)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("accepted control-plane manifest is invalid")
    expected = {Path(name): str(digest) for name, digest in files.items()}
    archive = _agent_control_plane_archive_bytes(
        root,
        manifest_raw=manifest_raw,
        expected_digests=expected,
    )
    if "sha256:" + hashlib.sha256(archive).hexdigest() != pin.archive_sha256:
        raise ValueError("accepted control-plane archive identity drifted")
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        raise ValueError("accepted control-plane memfd sealing is unavailable")
    descriptor = os.memfd_create(
        "ipfs-accelerate-accepted-control-plane",
        flags=getattr(os, "MFD_CLOEXEC", 0) | os.MFD_ALLOW_SEALING,
    )
    try:
        view = memoryview(archive)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ValueError("accepted control-plane archive write failed")
            view = view[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
        required = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required)
        executable = verify_agent_implementation_sealed_control_plane(
            pin,
            descriptor,
        )
        seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
    except Exception:
        os.close(descriptor)
        raise
    return AgentImplementationSealedControlPlane(
        descriptor=descriptor,
        executable_path=executable,
        archive_sha256=pin.archive_sha256,
        seals=seals,
        capsule_id=pin.capsule_id,
    )


def _parse_agent_control_plane_pin(
    value: object,
) -> AgentImplementationControlPlanePin:
    expected = {
        "schema",
        "runner_path",
        "runner_sha256",
        "capsule_root",
        "capsule_id",
        "source_head",
        "source_tree",
        "archive_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("accepted control-plane pin fields are invalid")
    pin = AgentImplementationControlPlanePin(
        **{name: _agent_string(value.get(name), name) for name in expected}
    )
    if (
        pin.schema != _AGENT_CONTROL_PLANE_PIN_SCHEMA
        or re.fullmatch(r"[0-9a-f]{40}", pin.source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", pin.source_tree) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", pin.archive_sha256) is None
    ):
        raise ValueError("accepted control-plane pin schema is invalid")
    return pin


def _parse_agent_invocation_binding(
    value: object,
) -> AgentImplementationInvocationBinding:
    expected = {
        "schema",
        "invocation_id",
        "logical_attempt_id",
        "task_id",
        "attempt",
        "task_revision_cid",
        "prompt_cid",
        "worktree_id",
        "workspace_path",
        "repository_cid",
        "baseline_commit",
        "effects",
        "scope_cid",
        "budget_cid",
        "resource_cid",
        "authority_cid",
        "route_id",
        "primary_provider_id",
        "primary_model_id",
        "fallback_provider_id",
        "fallback_model_id",
        "fallback_reasoning_effort",
        "fallback_implementer_identity",
        "reviewer_identity",
        "reviewer_provider",
        "profile_id",
        "profile_identity_did",
        "profile_lifecycle_anchor_id",
        "profile_lifecycle_generation",
        "profile_dir",
        "profile_lifecycle_dir",
        "issued_at_ms",
        "expires_at_ms",
        "provider_attempt_store",
        "provider_attempt_store_identity",
        "control_plane",
        "reviewer_signature",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("agent invocation binding fields are invalid")
    effects = value.get("effects")
    if not isinstance(effects, list) or not effects:
        raise ValueError("agent invocation effect bounds are invalid")
    normalized_effects = tuple(_agent_string(item, "effect") for item in effects)
    if tuple(sorted(normalized_effects)) != normalized_effects or len(
        set(normalized_effects)
    ) != len(normalized_effects):
        raise ValueError("agent invocation effect bounds are invalid")
    control_plane = _parse_agent_control_plane_pin(value.get("control_plane"))
    strings = {
        name: _agent_string(value.get(name), name)
        for name in expected
        if name
        not in {
            "effects",
            "control_plane",
            "profile_lifecycle_generation",
            "attempt",
            "issued_at_ms",
            "expires_at_ms",
        }
    }
    integers: dict[str, int] = {}
    for name in (
        "profile_lifecycle_generation",
        "attempt",
        "issued_at_ms",
        "expires_at_ms",
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"agent invocation {name} is invalid")
        integers[name] = item
    binding = AgentImplementationInvocationBinding(
        **strings,
        **integers,
        effects=normalized_effects,
        control_plane=control_plane,
    )
    if binding.schema != _AGENT_INVOCATION_BINDING_SCHEMA:
        raise ValueError("agent invocation binding schema is invalid")
    return binding


def resolve_agent_implementation_private_state_path(value: Path | str) -> Path:
    """Reject symlink components before normalizing signed private state."""

    expanded = Path(value).expanduser()
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        if component in {"", "."}:
            continue
        if component == "..":
            raise ValueError("signed private state path contains parent traversal")
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("signed private state path is unavailable") from exc
        if stat_module.S_ISLNK(metadata.st_mode):
            raise ValueError("signed private state path contains a symlink")
    return Path(os.path.abspath(os.fspath(candidate)))


def bind_agent_implementation_attempt_store(
    value: Path | str,
    *,
    create: bool = False,
    expected_identity: str = "",
) -> tuple[Path, str]:
    """Create/reopen one stable private CAS root and bind its inode identity."""

    path = resolve_agent_implementation_private_state_path(value)
    try:
        os.lstat(path)
    except FileNotFoundError:
        if not create:
            raise ValueError("provider attempt store is unavailable")
        created_identity = _agent_create_private_directory_chain(
            path,
            final_mode=0o700,
        )
        # A parent inserted during recursive creation must not survive the
        # second lexical component walk.
        if resolve_agent_implementation_private_state_path(path) != path:
            raise ValueError("provider attempt store changed during creation")
        created_path = os.lstat(path)
        if (
            created_path.st_dev,
            created_path.st_ino,
            created_path.st_mode,
            created_path.st_uid,
        ) != created_identity:
            raise ValueError("provider attempt store changed during creation")
    except OSError as exc:
        raise ValueError("provider attempt store is unavailable") from exc
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("provider attempt store no-follow checks are unavailable")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
        )
    except OSError as exc:
        raise ValueError("provider attempt store is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        final = os.lstat(path)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("provider attempt store changed") from exc
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
    )
    if (
        not stat_module.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat_module.S_IMODE(before.st_mode) != 0o700
        or not (identity(before) == identity(after) == identity(final))
    ):
        raise ValueError("provider attempt store is not an owned private directory")
    body = {
        "path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "uid": before.st_uid,
    }
    observed_identity = _content_addressed_mapping(
        body,
        identity_field="store_identity",
    )
    if expected_identity and observed_identity != expected_identity:
        raise ValueError("provider attempt store identity drifted")
    return path, observed_identity


def verify_agent_implementation_invocation_binding(
    binding: AgentImplementationInvocationBinding | Mapping[str, object],
    *,
    route: AgentImplementationRoutePlan,
    repo_root: Path | str,
    workspace: Path | str,
    expected_binding: Mapping[str, object] | None = None,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
    historical_effect_started_at_ms: int | None = None,
) -> AgentImplementationInvocationBinding:
    """Verify signature, exact route/authority equality, and accepted provenance."""

    parsed = (
        binding
        if isinstance(binding, AgentImplementationInvocationBinding)
        else _parse_agent_invocation_binding(binding)
    )
    authorization = route.authorization
    bounds = authorization.authority_bounds if authorization is not None else None
    historical = historical_effect_started_at_ms is not None
    if authorization is None or bounds is None:
        raise ValueError("signed invocation requires scoped route authorization")
    if (
        isinstance(now_ms, bool)
        or not isinstance(now_ms, int)
        or now_ms <= 0
        or isinstance(max_age_ms, bool)
        or not isinstance(max_age_ms, int)
        or not 0 < max_age_ms <= _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
        or (
            historical
            and (
                isinstance(historical_effect_started_at_ms, bool)
                or not isinstance(historical_effect_started_at_ms, int)
                or historical_effect_started_at_ms <= 0
                or historical_effect_started_at_ms != now_ms
            )
        )
        or parsed.attempt < 1
        or parsed.profile_lifecycle_generation < 1
        or parsed.issued_at_ms <= 0
        or parsed.issued_at_ms >= parsed.expires_at_ms
        or parsed.expires_at_ms - parsed.issued_at_ms
        > _AGENT_IMPLEMENTATION_MAX_FRESHNESS_MS
        or parsed.issued_at_ms
        > now_ms + _AGENT_IMPLEMENTATION_MAX_CLOCK_SKEW_MS
        or now_ms > parsed.expires_at_ms
        or now_ms - parsed.issued_at_ms > max_age_ms
    ):
        raise ValueError("signed invocation freshness is invalid")
    route_values = {
        name: getattr(route, name)
        for name in (
            "route_id",
            "primary_provider_id",
            "primary_model_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "fallback_implementer_identity",
        )
    }
    parsed_route_values = {
        name: getattr(parsed, name) for name in route_values
    }
    if (
        parsed_route_values != route_values
        or parsed.reviewer_identity != authorization.reviewer_identity
        or parsed.reviewer_provider != authorization.reviewer_provider
        or parsed.fallback_implementer_identity
        == parsed.reviewer_identity
        or parsed.reviewer_provider in {"codex", "openai"}
        or parsed.repository_cid != bounds.repository_cid
        or parsed.effects != bounds.effects
        or parsed.budget_cid != bounds.budget_cid
        or parsed.resource_cid != bounds.resource_cid
        or parsed.authority_cid != bounds.authority_cid
    ):
        raise ValueError("signed invocation does not exactly match route authority")
    try:
        workspace_root = resolve_agent_implementation_private_state_path(
            workspace
        ).resolve(strict=True)
        if not historical:
            observed_head = os.fsdecode(
                _agent_git_output(
                    workspace_root,
                    ("rev-parse", "--verify", "HEAD^{commit}"),
                )
            ).strip()
            _agent_git_output(
                workspace_root,
                (
                    "merge-base",
                    "--is-ancestor",
                    bounds.baseline_commit,
                    parsed.baseline_commit,
                ),
                maximum_bytes=64,
            )
            if observed_head != parsed.baseline_commit:
                raise ValueError(
                    "signed invocation baseline is not the live descendant"
                )
    except (OSError, ValueError) as exc:
        raise ValueError("signed invocation baseline is unavailable") from exc
    expected = dict(expected_binding or {})
    if expected:
        expected.pop("reviewer_signature", None)
        if expected != parsed.signed_payload():
            raise ValueError("signed invocation does not match the live invocation")
    # ``repo_root`` is the candidate repository used for authorization-tree
    # checks.  The accepted capsule is instead the already imported router's
    # source generation and must never be inferred from that candidate.
    Path(repo_root).expanduser().resolve(strict=True)
    candidate = Path(workspace).expanduser().resolve(strict=True)
    signed_workspace = Path(parsed.workspace_path).expanduser().resolve(strict=True)
    attempt_store, attempt_store_identity = (
        bind_agent_implementation_attempt_store(
            parsed.provider_attempt_store,
            expected_identity=parsed.provider_attempt_store_identity,
        )
    )
    if (
        signed_workspace != candidate
        or attempt_store.is_relative_to(candidate)
        or not attempt_store.is_absolute()
        or attempt_store_identity != parsed.provider_attempt_store_identity
    ):
        raise ValueError("signed invocation workspace/state binding is invalid")
    expected_worktree_id = _agent_dag_json_content_identity(
        {
            "workspace_path": str(candidate),
            "baseline_commit": parsed.baseline_commit,
        }
    )
    logical_body = {
        "task_id": parsed.task_id,
        "task_revision_cid": parsed.task_revision_cid,
        "attempt": parsed.attempt,
        "prompt_cid": parsed.prompt_cid,
        "worktree_id": parsed.worktree_id,
        "route_id": parsed.route_id,
    }
    expected_logical_attempt_id = _agent_dag_json_content_identity(logical_body)
    expected_invocation_id = _agent_dag_json_content_identity(
        {**logical_body, "logical_attempt_id": expected_logical_attempt_id}
    )
    if (
        parsed.worktree_id != expected_worktree_id
        or parsed.logical_attempt_id != expected_logical_attempt_id
        or parsed.invocation_id != expected_invocation_id
    ):
        raise ValueError("signed invocation content identities are invalid")
    pin = parsed.control_plane
    observed_pin = (
        pin
        if historical
        else build_agent_implementation_control_plane_pin(
            runner_path=pin.runner_path,
            capsule_root=pin.capsule_root,
        )
    )
    runner = Path(observed_pin.runner_path)
    capsule_root = Path(observed_pin.capsule_root)
    # The accepted capsule is the already-imported router generation.  The
    # candidate can only be the workspace argument; it can never be an import
    # root or runner source.
    if (
        observed_pin != pin
        or runner.is_relative_to(candidate)
        or capsule_root.is_relative_to(candidate)
    ):
        raise ValueError("accepted control-plane provenance drifted")
    _agent_verify_did_signature(
        identity_did=parsed.reviewer_identity,
        payload=parsed.signed_payload(),
        signature=parsed.reviewer_signature,
    )
    if historical:
        # Rotation, revocation, or operator cleanup after effect_started may
        # remove/replace live profile state.  Historical recovery accounts an
        # existing effect only; it never consults those paths to authorize a
        # new start.  Retain merely the signed lexical/outside-worktree bound.
        profile_directory = Path(parsed.profile_dir)
        lifecycle_directory = Path(parsed.profile_lifecycle_dir)
        if (
            not profile_directory.is_absolute()
            or ".." in profile_directory.parts
            or not lifecycle_directory.is_absolute()
            or ".." in lifecycle_directory.parts
            or profile_directory.is_relative_to(candidate)
            or lifecycle_directory.is_relative_to(candidate)
        ):
            raise ValueError("signed profile state paths are invalid")
        profile = None
    else:
        from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
            load_local_profile,
            resolve_local_profile_state_paths,
        )

        profile_directory = resolve_agent_implementation_private_state_path(
            parsed.profile_dir
        )
        lifecycle_directory = resolve_agent_implementation_private_state_path(
            parsed.profile_lifecycle_dir
        )
        current_profile_dir, current_lifecycle_dir = (
            resolve_local_profile_state_paths(
                profile_dir=profile_directory,
                lifecycle_dir=lifecycle_directory,
            )
        )
        if (
            current_profile_dir != profile_directory
            or current_lifecycle_dir != lifecycle_directory
            or profile_directory.is_relative_to(candidate)
            or lifecycle_directory.is_relative_to(candidate)
        ):
            raise ValueError("signed profile state paths are invalid")
        profile = load_local_profile(
            repository_cid=parsed.repository_cid,
            profile_dir=profile_directory,
            lifecycle_dir=lifecycle_directory,
        )
    if not historical and profile is not None and (
        profile.profile_id != parsed.profile_id
        or profile.identity_did != parsed.profile_identity_did
        or profile.identity_did != parsed.reviewer_identity
        or profile.lifecycle_anchor_id != parsed.profile_lifecycle_anchor_id
        or profile.lifecycle_generation
        != parsed.profile_lifecycle_generation
        or profile.lifecycle_root_path != str(lifecycle_directory)
        or profile.baseline_commit != bounds.baseline_commit
        or profile.effect_bounds != parsed.effects
        or profile.budget_cid != parsed.budget_cid
        or profile.resource_cid != parsed.resource_cid
        or profile.content_id != parsed.authority_cid
        or profile.route_id != parsed.route_id
        or profile.fallback_provider_id != parsed.fallback_provider_id
        or profile.fallback_model_id != parsed.fallback_model_id
        or profile.fallback_reasoning_effort
        != parsed.fallback_reasoning_effort
    ):
        raise ValueError("signed invocation profile lifecycle is no longer current")
    return parsed


def bind_agent_implementation_route_invocation(
    route: AgentImplementationRoutePlan,
    binding: AgentImplementationInvocationBinding | Mapping[str, object],
    *,
    repo_root: Path | str,
    workspace: Path | str,
    expected_binding: Mapping[str, object] | None = None,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
    historical_effect_started_at_ms: int | None = None,
) -> AgentImplementationRoutePlan:
    verified = verify_agent_implementation_invocation_binding(
        binding,
        route=route,
        repo_root=repo_root,
        workspace=workspace,
        expected_binding=expected_binding,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        historical_effect_started_at_ms=historical_effect_started_at_ms,
    )
    return replace(route, invocation_binding=verified)


def resolve_agent_implementation_route_binding(
    binding: Mapping[str, object],
    *,
    repo_root: Path | str,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
    historical_effect_started_at_ms: int | None = None,
) -> AgentImplementationRoutePlan:
    """Revalidate a scheduler-authored frozen plan at an effect boundary."""

    expected_fields = {
        *_AGENT_IMPLEMENTATION_ROUTE_FIELDS,
        "route_id",
        "authorization",
        "fallback_implementer_identity",
        "invocation_binding",
    }
    if set(binding) != expected_fields:
        raise ValueError("agent implementation route binding fields are invalid")
    authorization_raw = binding.get("authorization")
    authorization = None
    if authorization_raw is not None:
        if not isinstance(authorization_raw, Mapping) or set(
            authorization_raw
        ) != {
            "board_namespace",
            "artifact_path",
            "artifact_sha256",
            "authorization_kind",
            "source_head",
            "source_tree",
            "authorization_id",
            "reviewer_identity",
            "reviewer_provider",
            "reviewer_signature",
            "reviewer_profile_id",
            "reviewer_profile_content_id",
            "reviewer_lifecycle_anchor_id",
            "reviewer_lifecycle_generation",
            "reviewer_witness_path",
            "reviewer_witness_sha256",
            "lifecycle_root_identity_did",
            "lifecycle_witness_nonce",
            "lifecycle_root_pin_path",
            "lifecycle_root_pin_sha256",
            "authorized_at_ms",
            "fallback_implementer_identity",
            "authority_bounds",
        }:
            raise ValueError(
                "agent implementation route authorization binding is invalid"
            )
        authorization = load_agent_implementation_route_authorization(
            repo_root=repo_root,
            artifact_path=str(authorization_raw.get("artifact_path") or ""),
            board_namespace=str(
                authorization_raw.get("board_namespace") or ""
            ),
            expected_sha256=str(
                authorization_raw.get("artifact_sha256") or ""
            ),
            expected_authorization_id=str(
                authorization_raw.get("authorization_id") or ""
            ),
        )
        if authorization.as_dict() != dict(authorization_raw):
            raise ValueError(
                "agent implementation route authorization binding drifted"
            )
    plan = resolve_agent_implementation_route(
        **{
            field: str(binding.get(field) or "")
            for field in _AGENT_IMPLEMENTATION_ROUTE_FIELDS
        },
        authorization=authorization,
    )
    if binding.get("route_id") != plan.route_id:
        raise ValueError("agent implementation route identity drifted")
    implementer = str(binding.get("fallback_implementer_identity") or "").strip()
    if not implementer:
        raise ValueError("fallback implementer identity is required")
    if (
        authorization is not None
        and (
            implementer == authorization.reviewer_identity
            or implementer != authorization.fallback_implementer_identity
        )
    ):
        raise ValueError("fallback implementer/reviewer binding is invalid")
    plan = replace(plan, fallback_implementer_identity=implementer)
    invocation_raw = binding.get("invocation_binding")
    if invocation_raw is None:
        return plan
    return bind_agent_implementation_route_invocation(
        plan,
        invocation_raw,
        repo_root=repo_root,
        workspace=str(
            invocation_raw.get("workspace_path")
            if isinstance(invocation_raw, Mapping)
            else ""
        ),
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        historical_effect_started_at_ms=historical_effect_started_at_ms,
    )


def decide_agent_implementation_fallback(
    route: AgentImplementationRoutePlan,
    *,
    repo_root: Path | str,
    failure_receipt: Mapping[str, object],
    expected_nonce: str,
    expected_model: str,
    expected_probe_returncode: int,
    independent_quota_evidence: object | None = None,
    expected_invocation_binding: Mapping[str, object] | None = None,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
    historical_effect_started_at_ms: int | None = None,
) -> AgentImplementationFallbackDecision:
    """Decide the exceptional typed fallback for side-effecting agent work.

    The function has no ambient defaults and revalidates the explicit route
    binding against ``repo_root`` at each decision boundary. A caller must
    supply a canonical frozen plan and the actual nonce-bound receipt;
    caller-provided booleans/classes/hashes never create authority.
    Generic/untyped errors, overflowed evidence, and mixed auth diagnostics
    never authorize fallback.
    """

    if historical_effect_started_at_ms is None:
        canonical_route = resolve_agent_implementation_route_binding(
            route.as_binding_dict(),
            repo_root=repo_root,
            now_ms=now_ms,
            max_age_ms=max_age_ms,
        )
    else:
        # The effect has already crossed the once-only CAS boundary.  Its
        # embedded authorization/root/witness snapshot was independently
        # verified by ``parse_*_effect_authorization_context``; reopening a
        # later checkout here would make accounting depend on mutable bytes
        # and strand effects after a valid rotation or artifact transition.
        authorization = route.authorization
        if not _agent_route_authorization_is_sealed(authorization):
            raise ValueError("historical route authorization is not sealed")
        canonical_route = resolve_agent_implementation_route(
            **{
                field: getattr(route, field)
                for field in _AGENT_IMPLEMENTATION_ROUTE_FIELDS
            },
            authorization=authorization,
        )
        if canonical_route.route_id != route.route_id:
            raise ValueError("historical route identity drifted")
        canonical_route = replace(
            canonical_route,
            fallback_implementer_identity=route.fallback_implementer_identity,
        )
        if route.invocation_binding is not None:
            canonical_route = bind_agent_implementation_route_invocation(
                canonical_route,
                route.invocation_binding,
                repo_root=repo_root,
                workspace=route.invocation_binding.workspace_path,
                expected_binding=route.invocation_binding.signed_payload(),
                now_ms=now_ms,
                max_age_ms=max_age_ms,
                historical_effect_started_at_ms=(
                    historical_effect_started_at_ms
                ),
            )
    if canonical_route.route_id != route.route_id:
        raise ValueError("agent implementation route identity is invalid")
    reviewer = canonical_route.authorization
    invocation = canonical_route.invocation_binding

    def decision(**values: object) -> AgentImplementationFallbackDecision:
        """Attach the authoritative route/reviewer binding to every outcome."""
        return AgentImplementationFallbackDecision(
            **values,
            route_id=canonical_route.route_id,
            fallback_provider_id=canonical_route.fallback_provider_id,
            fallback_model_id=canonical_route.fallback_model_id,
            fallback_reasoning_effort=canonical_route.fallback_reasoning_effort,
            reviewer_identity=(reviewer.reviewer_identity if reviewer else ""),
            reviewer_provider=(reviewer.reviewer_provider if reviewer else ""),
            invocation_binding_id=(invocation.content_id if invocation else ""),
            control_plane_id=(
                invocation.control_plane.capsule_id if invocation else ""
            ),
        )
    if reviewer is not None:
        if invocation is None or expected_invocation_binding is None:
            return decision(
                authorized=False,
                requires_independent_quota_verification=False,
                reason_code="signed_invocation_required",
                verifier_status="not_run",
            )
        try:
            verify_agent_implementation_invocation_binding(
                invocation,
                route=canonical_route,
                repo_root=repo_root,
                workspace=invocation.workspace_path,
                expected_binding=expected_invocation_binding,
            now_ms=now_ms,
            max_age_ms=max_age_ms,
            historical_effect_started_at_ms=(
                historical_effect_started_at_ms
            ),
            )
        except (OSError, ValueError):
            return decision(
                authorized=False,
                requires_independent_quota_verification=False,
                reason_code="signed_invocation_mismatch",
                verifier_status="not_run",
            )
    receipt_valid = valid_agent_implementation_failure_receipt(
        failure_receipt,
        nonce=expected_nonce,
        model=expected_model,
        probe_returncode=expected_probe_returncode,
        now_ms=(now_ms if reviewer is not None else None),
        max_age_ms=(max_age_ms if reviewer is not None else None),
    )
    if not receipt_valid or failure_receipt.get("evidence_overflow") is True:
        return decision(
            authorized=False,
            requires_independent_quota_verification=False,
            reason_code="typed_failure_denied",
            verifier_status="not_run",
        )
    normalized_class = str(failure_receipt.get("failure_class") or "").strip()
    normalized_evidence = str(
        failure_receipt.get("evidence_sha256") or ""
    ).strip()
    if normalized_class == "authentication_unavailable":
        if (
            canonical_route.permits_authentication_unavailable
            and normalized_evidence
            in _AGENT_IMPLEMENTATION_DIRECT_AUTH_EVIDENCE
        ):
            return decision(
                authorized=True,
                requires_independent_quota_verification=False,
                reason_code="authentication_unavailable",
                verifier_status="not_required_exact_auth",
            )
        return decision(
            authorized=False,
            requires_independent_quota_verification=False,
            reason_code="authentication_fallback_not_in_route",
            verifier_status="not_run",
        )
    if normalized_class not in {"hard_quota_exhausted", "authentication"}:
        return decision(
            authorized=False,
            requires_independent_quota_verification=False,
            reason_code="failure_class_not_authorized",
            verifier_status="not_run",
        )
    if independent_quota_evidence is None:
        return decision(
            authorized=False,
            requires_independent_quota_verification=True,
            reason_code="independent_quota_verification_required",
            verifier_status="not_run",
        )
    valid_quota_evidence = _valid_agent_implementation_quota_evidence(
        independent_quota_evidence,
        failure_receipt=failure_receipt,
        invocation=invocation,
        now_ms=now_ms,
        max_age_ms=max_age_ms,
        require_current_lifecycle=(
            historical_effect_started_at_ms is None
        ),
    )
    if valid_quota_evidence:
        return decision(
            authorized=True,
            requires_independent_quota_verification=False,
            reason_code="quota_exhausted",
            verifier_status="confirmed_quota",
        )
    return decision(
        authorized=False,
        requires_independent_quota_verification=False,
        reason_code="independent_quota_not_confirmed",
        verifier_status="not_confirmed",
    )


