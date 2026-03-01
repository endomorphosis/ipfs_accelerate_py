"""UCAN-style delegation validation for MCP++ Profile C.

This module provides deterministic, dependency-light validation of delegation
chains for execution-time authorization checks in unified dispatch paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Tuple

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    HAVE_CRYPTO_ED25519 = True
except Exception:  # pragma: no cover - optional dependency guard
    InvalidSignature = Exception  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]
    HAVE_CRYPTO_ED25519 = False


@dataclass(frozen=True)
class UcanCapability:
    """Capability tuple for resource/ability checks."""

    resource: str
    ability: str

    def matches(self, *, resource: str, ability: str) -> bool:
        resource_ok = self.resource in {"*", resource}
        ability_ok = self.ability in {"*", ability}
        return resource_ok and ability_ok


@dataclass(frozen=True)
class UcanDelegation:
    """Delegation edge from issuer -> audience with capability set."""

    issuer: str
    audience: str
    capabilities: Tuple[UcanCapability, ...]
    expiry: float | None = None
    revoked: bool = False
    proof_cid: str = ""
    signature: str = ""
    caveats: Tuple[Dict[str, Any], ...] = ()

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.expiry is None:
            return False
        t = float(now if now is not None else time.time())
        return t > float(self.expiry)


@dataclass(frozen=True)
class UcanValidationResult:
    """Structured result for delegation chain validation."""

    allowed: bool
    reason: str
    chain_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "chain_length": self.chain_length,
        }


def _parse_capabilities(raw: Iterable[Dict[str, Any]]) -> Tuple[UcanCapability, ...]:
    caps: List[UcanCapability] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resource = str(item.get("resource", "") or "").strip() or "*"
        ability = str(item.get("ability", "") or "").strip() or "*"
        caps.append(UcanCapability(resource=resource, ability=ability))
    return tuple(caps)


def _canonical_delegation_payload(delegation: UcanDelegation) -> bytes:
    """Canonical payload bytes used for deterministic proof/signature checks."""
    payload = {
        "issuer": delegation.issuer,
        "audience": delegation.audience,
        "capabilities": [{"resource": c.resource, "ability": c.ability} for c in delegation.capabilities],
        "expiry": delegation.expiry,
        "caveats": list(delegation.caveats),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_delegation_proof_cid(delegation: UcanDelegation) -> str:
    """Compute deterministic proof CID for a delegation payload."""
    digest = hashlib.sha256(_canonical_delegation_payload(delegation)).hexdigest()
    return f"cidv1-sha256-{digest}"


def compute_delegation_signature(
    *,
    delegation: UcanDelegation,
    issuer_key_hint: str = "",
) -> str:
    """Compute deterministic signature token for a delegation.

    This is a dependency-light signature envelope checker for migration phases,
    not a replacement for full cryptographic proof systems.
    """
    proof_cid = compute_delegation_proof_cid(delegation)
    material = f"{delegation.issuer}|{proof_cid}|{issuer_key_hint}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return f"sigv1-sha256-{digest}"


def _b64_decode(value: str) -> bytes:
    """Decode base64/base64url with optional missing padding."""
    text = str(value or "").strip()
    if not text:
        return b""
    text = text.replace("-", "+").replace("_", "/")
    padding = "=" * ((4 - (len(text) % 4)) % 4)
    return base64.b64decode(text + padding)


def _b64_encode_urlsafe(raw: bytes) -> str:
    """Encode bytes as unpadded base64url text."""
    return base64.urlsafe_b64encode(bytes(raw)).decode("ascii").rstrip("=")


def _extract_ed25519_public_key_b64(value: Any) -> str:
    """Extract supported Ed25519 public key string from issuer key mapping value."""
    if isinstance(value, dict):
        alg = str(value.get("alg") or value.get("algorithm") or "").strip().lower()
        key = str(value.get("public_key") or value.get("key") or "").strip()
        if alg and alg not in {"ed25519", "eddsa"}:
            return ""
        if key.startswith("ed25519-pub:"):
            return key.split(":", 1)[1].strip()
        return key

    text = str(value or "").strip()
    if text.startswith("ed25519-pub:"):
        return text.split(":", 1)[1].strip()
    return text


def compute_delegation_signature_ed25519(
    *,
    delegation: UcanDelegation,
    private_key_b64: str,
) -> str:
    """Compute Ed25519 signature for delegation payload.

    Returns: `ed25519:<base64url(signature)>`
    """
    if not HAVE_CRYPTO_ED25519:
        raise RuntimeError("cryptography_ed25519_unavailable")

    key_bytes = _b64_decode(private_key_b64)
    if len(key_bytes) not in {32, 64}:
        raise ValueError("invalid_ed25519_private_key_length")

    if len(key_bytes) == 64:
        key_bytes = key_bytes[:32]

    signer = Ed25519PrivateKey.from_private_bytes(key_bytes)
    sig = signer.sign(_canonical_delegation_payload(delegation))
    return f"ed25519:{_b64_encode_urlsafe(sig)}"


def verify_delegation_signature_ed25519(
    *,
    delegation: UcanDelegation,
    signature: str,
    public_key_b64: str,
) -> bool:
    """Verify Ed25519 signature token for delegation payload."""
    if not HAVE_CRYPTO_ED25519:
        return False

    token = str(signature or "").strip()
    if not token.startswith("ed25519:"):
        return False

    sig_b64 = token.split(":", 1)[1].strip()
    if not sig_b64:
        return False

    pub = _extract_ed25519_public_key_b64(public_key_b64)
    pub_bytes = _b64_decode(pub)
    if len(pub_bytes) != 32:
        return False

    sig_bytes = _b64_decode(sig_b64)
    try:
        verifier = Ed25519PublicKey.from_public_bytes(pub_bytes)
        verifier.verify(sig_bytes, _canonical_delegation_payload(delegation))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def _caveats_allow(
    caveats: Tuple[Dict[str, Any], ...],
    *,
    resource: str,
    ability: str,
    now: float,
) -> bool:
    """Evaluate a minimal deterministic caveat subset.

    Supported caveats:
    - `not_before`: UNIX seconds
    - `not_after`: UNIX seconds
    - `resource_prefix`: required resource prefix
    - `ability`: required exact ability
    """
    for caveat in caveats:
        if not isinstance(caveat, dict):
            continue
        try:
            not_before = caveat.get("not_before")
            if not_before is not None and now < float(not_before):
                return False
        except Exception:
            return False
        try:
            not_after = caveat.get("not_after")
            if not_after is not None and now > float(not_after):
                return False
        except Exception:
            return False
        prefix = str(caveat.get("resource_prefix") or "").strip()
        if prefix and not str(resource or "").startswith(prefix):
            return False
        required_ability = str(caveat.get("ability") or "").strip()
        if required_ability and required_ability != str(ability or ""):
            return False
    return True


def parse_delegation_chain(raw_chain: Iterable[Dict[str, Any]]) -> List[UcanDelegation]:
    """Parse raw dict chain into typed delegation entries."""
    chain: List[UcanDelegation] = []
    for item in raw_chain:
        if not isinstance(item, dict):
            continue
        chain.append(
            UcanDelegation(
                issuer=str(item.get("issuer", "") or "").strip(),
                audience=str(item.get("audience", "") or "").strip(),
                capabilities=_parse_capabilities(item.get("capabilities", []) or []),
                expiry=float(item["expiry"]) if item.get("expiry") is not None else None,
                revoked=bool(item.get("revoked", False)),
                proof_cid=str(item.get("proof_cid", "") or "").strip(),
                signature=str(item.get("signature", "") or "").strip(),
                caveats=tuple(item.get("caveats", []) or []),
            )
        )
    return chain


def _covers(parent: UcanDelegation, child: UcanDelegation) -> bool:
    """Return True if child's capabilities are attenuated by parent's set."""
    if not child.capabilities:
        return False
    for child_cap in child.capabilities:
        ok = any(
            parent_cap.matches(resource=child_cap.resource, ability=child_cap.ability)
            for parent_cap in parent.capabilities
        )
        if not ok:
            return False
    return True


def validate_delegation_chain(
    *,
    chain: Iterable[UcanDelegation],
    resource: str,
    ability: str,
    actor: str = "",
    now: float | None = None,
    require_signatures: bool = False,
    issuer_public_keys: Dict[str, str] | None = None,
    revoked_proof_cids: Iterable[str] | None = None,
) -> UcanValidationResult:
    """Validate a root->leaf delegation chain for execution authorization."""
    parsed = list(chain)
    if not parsed:
        return UcanValidationResult(False, "missing_delegation_chain", 0)

    t = float(now if now is not None else time.time())
    revoked_set = {str(x or "").strip() for x in (revoked_proof_cids or []) if str(x or "").strip()}
    key_map = dict(issuer_public_keys or {})

    for idx, d in enumerate(parsed):
        if not d.issuer or not d.audience:
            return UcanValidationResult(False, f"invalid_principal_at_hop_{idx}", len(parsed))
        if d.revoked:
            return UcanValidationResult(False, f"revoked_at_hop_{idx}", len(parsed))
        if d.is_expired(now=t):
            return UcanValidationResult(False, f"expired_at_hop_{idx}", len(parsed))
        if d.proof_cid and d.proof_cid in revoked_set:
            return UcanValidationResult(False, f"revoked_proof_at_hop_{idx}", len(parsed))
        if not _caveats_allow(d.caveats, resource=resource, ability=ability, now=t):
            return UcanValidationResult(False, f"caveat_denied_at_hop_{idx}", len(parsed))
        if require_signatures:
            if not d.proof_cid or not d.signature:
                return UcanValidationResult(False, f"missing_signature_at_hop_{idx}", len(parsed))
            expected_proof = compute_delegation_proof_cid(d)
            if d.proof_cid != expected_proof:
                return UcanValidationResult(False, f"invalid_proof_cid_at_hop_{idx}", len(parsed))

            issuer_key = key_map.get(d.issuer, "")
            if str(d.signature or "").startswith("ed25519:"):
                if not HAVE_CRYPTO_ED25519:
                    return UcanValidationResult(False, f"cryptography_unavailable_at_hop_{idx}", len(parsed))
                if not verify_delegation_signature_ed25519(
                    delegation=d,
                    signature=d.signature,
                    public_key_b64=_extract_ed25519_public_key_b64(issuer_key),
                ):
                    return UcanValidationResult(False, f"invalid_signature_at_hop_{idx}", len(parsed))
            else:
                expected_sig = compute_delegation_signature(
                    delegation=d,
                    issuer_key_hint=str(issuer_key or ""),
                )
                if d.signature != expected_sig:
                    return UcanValidationResult(False, f"invalid_signature_at_hop_{idx}", len(parsed))

    # issuer/audience continuity: audience(i) == issuer(i+1)
    for idx in range(len(parsed) - 1):
        if parsed[idx].audience != parsed[idx + 1].issuer:
            return UcanValidationResult(False, f"broken_chain_at_hop_{idx}", len(parsed))

    # attenuation: each child must be subset of parent capabilities
    for idx in range(len(parsed) - 1):
        if not _covers(parsed[idx], parsed[idx + 1]):
            return UcanValidationResult(False, f"capability_escalation_at_hop_{idx+1}", len(parsed))

    leaf = parsed[-1]
    if actor and leaf.audience != actor:
        return UcanValidationResult(False, "actor_mismatch", len(parsed))

    granted = any(cap.matches(resource=resource, ability=ability) for cap in leaf.capabilities)
    if not granted:
        return UcanValidationResult(False, "capability_not_granted", len(parsed))

    return UcanValidationResult(True, "allowed", len(parsed))


def validate_raw_delegation_chain(
    *,
    raw_chain: Iterable[Dict[str, Any]],
    resource: str,
    ability: str,
    actor: str = "",
    now: float | None = None,
    require_signatures: bool = False,
    issuer_public_keys: Dict[str, str] | None = None,
    revoked_proof_cids: Iterable[str] | None = None,
) -> UcanValidationResult:
    """Convenience validator for raw dict chain payloads."""
    chain = parse_delegation_chain(raw_chain)
    return validate_delegation_chain(
        chain=chain,
        resource=resource,
        ability=ability,
        actor=actor,
        now=now,
        require_signatures=require_signatures,
        issuer_public_keys=issuer_public_keys,
        revoked_proof_cids=revoked_proof_cids,
    )


__all__ = [
    "UcanCapability",
    "UcanDelegation",
    "UcanValidationResult",
    "HAVE_CRYPTO_ED25519",
    "compute_delegation_proof_cid",
    "compute_delegation_signature",
    "compute_delegation_signature_ed25519",
    "parse_delegation_chain",
    "verify_delegation_signature_ed25519",
    "validate_delegation_chain",
    "validate_raw_delegation_chain",
]
