"""Signed planning-policy artifact (ASE3-024).

Independently versioned from :class:`SignedSupervisorProfile`. Owns bounded
planning route, retention, and replay policy only. Never grants provider
choice from prompt content and never mutates the supervisor profile schema.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Final, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
    ed25519_public_key_from_did,
)

PLANNING_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/signed-prompt-planning-policy@1"
)
PLANNING_POLICY_REQUIREMENT_ID: Final = (
    "prompt_v3.planning_policy.signed_artifact@1"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_TTL_MS: Final = 7 * 24 * 60 * 60 * 1000
_MAX_ATTEMPTS: Final = 8


class PlanningPolicyError(ValueError):
    """Raised when a planning-policy artifact is malformed or unauthorized."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_sha256(value: Any, name: str) -> str:
    text = str(value or "")
    if _SHA256_RE.fullmatch(text) is None:
        raise PlanningPolicyError(f"{name} must be sha256:<hex>")
    return text


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanningPolicyError(f"{name} must be a positive integer")
    if value > maximum:
        raise PlanningPolicyError(f"{name} exceeds maximum {maximum}")
    return value


@dataclass(frozen=True)
class SignedPromptPlanningPolicyArtifact:
    """Operator-signed planning bounds independent of SignedSupervisorProfile."""

    schema: str
    policy_id: str
    signer_identity_did: str
    signer_generation: int
    issued_at_ms: int
    expires_at_ms: int
    revoked: bool
    max_prompt_bytes: int
    max_planning_attempts: int
    retention_ttl_ms: int
    allow_provider_replay_on_unknown: bool
    allowed_planning_providers: tuple[str, ...]
    signature: str
    content_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != PLANNING_POLICY_SCHEMA:
            raise PlanningPolicyError("unsupported planning-policy schema")
        if not str(self.policy_id or "").strip():
            raise PlanningPolicyError("policy_id is required")
        if not str(self.signer_identity_did or "").startswith("did:key:"):
            raise PlanningPolicyError("signer_identity_did must be did:key")
        object.__setattr__(
            self,
            "signer_generation",
            _positive_int(self.signer_generation, "signer_generation", maximum=10**9),
        )
        if (
            isinstance(self.issued_at_ms, bool)
            or not isinstance(self.issued_at_ms, int)
            or self.issued_at_ms <= 0
        ):
            raise PlanningPolicyError("issued_at_ms is invalid")
        if (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms <= self.issued_at_ms
        ):
            raise PlanningPolicyError("expires_at_ms must be after issued_at_ms")
        object.__setattr__(self, "revoked", bool(self.revoked))
        object.__setattr__(
            self,
            "max_prompt_bytes",
            _positive_int(self.max_prompt_bytes, "max_prompt_bytes", maximum=10**7),
        )
        object.__setattr__(
            self,
            "max_planning_attempts",
            _positive_int(
                self.max_planning_attempts,
                "max_planning_attempts",
                maximum=_MAX_ATTEMPTS,
            ),
        )
        object.__setattr__(
            self,
            "retention_ttl_ms",
            _positive_int(
                self.retention_ttl_ms, "retention_ttl_ms", maximum=_MAX_TTL_MS
            ),
        )
        if self.allow_provider_replay_on_unknown:
            raise PlanningPolicyError(
                "allow_provider_replay_on_unknown must be false"
            )
        providers = tuple(
            str(item).strip().lower()
            for item in (self.allowed_planning_providers or ())
            if str(item or "").strip()
        )
        if not providers:
            raise PlanningPolicyError("allowed_planning_providers is required")
        object.__setattr__(self, "allowed_planning_providers", providers)
        if not str(self.signature or "").strip():
            raise PlanningPolicyError("signature is required")
        expected = self.compute_content_id()
        if self.content_id and self.content_id != expected:
            raise PlanningPolicyError("planning-policy content_id drifted")
        object.__setattr__(self, "content_id", expected)

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "signer_identity_did": self.signer_identity_did,
            "signer_generation": self.signer_generation,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "revoked": self.revoked,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_planning_attempts": self.max_planning_attempts,
            "retention_ttl_ms": self.retention_ttl_ms,
            "allow_provider_replay_on_unknown": self.allow_provider_replay_on_unknown,
            "allowed_planning_providers": list(self.allowed_planning_providers),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_payload()
        payload["signature"] = self.signature
        payload["content_id"] = self.content_id
        return payload

    def compute_content_id(self) -> str:
        body = self.unsigned_payload()
        body["signature"] = self.signature
        return "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()


def sign_prompt_planning_policy(
    *,
    private_key: Ed25519PrivateKey,
    policy_id: str,
    signer_generation: int = 1,
    issued_at_ms: int | None = None,
    expires_at_ms: int | None = None,
    max_prompt_bytes: int = 256_000,
    max_planning_attempts: int = 1,
    retention_ttl_ms: int = 3_600_000,
    allowed_planning_providers: tuple[str, ...] = ("grok",),
) -> SignedPromptPlanningPolicyArtifact:
    """Create and sign a planning-policy artifact with an Ed25519 key."""

    issued = int(issued_at_ms if issued_at_ms is not None else _now_ms())
    expires = int(
        expires_at_ms if expires_at_ms is not None else issued + retention_ttl_ms
    )
    did = ed25519_did_key(private_key.public_key())
    unsigned = {
        "schema": PLANNING_POLICY_SCHEMA,
        "policy_id": str(policy_id),
        "signer_identity_did": did,
        "signer_generation": int(signer_generation),
        "issued_at_ms": issued,
        "expires_at_ms": expires,
        "revoked": False,
        "max_prompt_bytes": int(max_prompt_bytes),
        "max_planning_attempts": int(max_planning_attempts),
        "retention_ttl_ms": int(retention_ttl_ms),
        "allow_provider_replay_on_unknown": False,
        "allowed_planning_providers": list(allowed_planning_providers),
    }
    signature = base64.b64encode(private_key.sign(_canonical(unsigned))).decode(
        "ascii"
    )
    return SignedPromptPlanningPolicyArtifact(
        **unsigned,
        signature=signature,
    )


def verify_prompt_planning_policy(
    artifact: SignedPromptPlanningPolicyArtifact | Mapping[str, Any],
    *,
    now_ms: int | None = None,
    expected_signer_did: str | None = None,
) -> SignedPromptPlanningPolicyArtifact:
    """Verify signature, expiry, and non-revocation of a planning policy."""

    if isinstance(artifact, SignedPromptPlanningPolicyArtifact):
        policy = artifact
    else:
        policy = SignedPromptPlanningPolicyArtifact(
            schema=str(artifact.get("schema") or ""),
            policy_id=str(artifact.get("policy_id") or ""),
            signer_identity_did=str(artifact.get("signer_identity_did") or ""),
            signer_generation=int(artifact.get("signer_generation") or 0),
            issued_at_ms=int(artifact.get("issued_at_ms") or 0),
            expires_at_ms=int(artifact.get("expires_at_ms") or 0),
            revoked=bool(artifact.get("revoked")),
            max_prompt_bytes=int(artifact.get("max_prompt_bytes") or 0),
            max_planning_attempts=int(artifact.get("max_planning_attempts") or 0),
            retention_ttl_ms=int(artifact.get("retention_ttl_ms") or 0),
            allow_provider_replay_on_unknown=bool(
                artifact.get("allow_provider_replay_on_unknown")
            ),
            allowed_planning_providers=tuple(
                artifact.get("allowed_planning_providers") or ()
            ),
            signature=str(artifact.get("signature") or ""),
            content_id=str(artifact.get("content_id") or ""),
        )

    if expected_signer_did and policy.signer_identity_did != expected_signer_did:
        raise PlanningPolicyError("planning-policy signer did not match")
    if policy.revoked:
        raise PlanningPolicyError("planning-policy is revoked")
    clock = int(now_ms if now_ms is not None else _now_ms())
    if clock < policy.issued_at_ms or clock >= policy.expires_at_ms:
        raise PlanningPolicyError("planning-policy is expired or not yet valid")

    public_key = ed25519_public_key_from_did(policy.signer_identity_did)
    try:
        public_key.verify(
            base64.b64decode(policy.signature.encode("ascii")),
            _canonical(policy.unsigned_payload()),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise PlanningPolicyError("planning-policy signature is invalid") from exc
    return policy


def revoke_prompt_planning_policy(
    artifact: SignedPromptPlanningPolicyArtifact,
    *,
    private_key: Ed25519PrivateKey,
) -> SignedPromptPlanningPolicyArtifact:
    """Return a re-signed revoked planning-policy artifact."""

    if ed25519_did_key(private_key.public_key()) != artifact.signer_identity_did:
        raise PlanningPolicyError("revocation signer must match original signer")
    unsigned = artifact.unsigned_payload()
    unsigned["revoked"] = True
    signature = base64.b64encode(private_key.sign(_canonical(unsigned))).decode(
        "ascii"
    )
    return SignedPromptPlanningPolicyArtifact(**unsigned, signature=signature)


__all__ = [
    "PLANNING_POLICY_REQUIREMENT_ID",
    "PLANNING_POLICY_SCHEMA",
    "PlanningPolicyError",
    "SignedPromptPlanningPolicyArtifact",
    "revoke_prompt_planning_policy",
    "sign_prompt_planning_policy",
    "verify_prompt_planning_policy",
]
