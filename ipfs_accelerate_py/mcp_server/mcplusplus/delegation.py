"""UCAN-style delegation validation for MCP++ Profile C.

This module provides deterministic, dependency-light validation of delegation
chains for execution-time authorization checks in unified dispatch paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import re
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
    proof_lineage: List[str]
    failure_hop: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "chain_length": self.chain_length,
            "proof_lineage": list(self.proof_lineage),
            "failure_hop": self.failure_hop,
        }


def _compute_proof_lineage(chain: Iterable[UcanDelegation]) -> list[str]:
    """Return deterministic proof lineage for a parsed delegation chain."""

    lineage: list[str] = []
    for delegation in chain:
        proof_cid = str(delegation.proof_cid or "").strip() or compute_delegation_proof_cid(delegation)
        lineage.append(proof_cid)
    return lineage


def _parse_capabilities(raw: Iterable[Dict[str, Any]]) -> Tuple[UcanCapability, ...]:
    caps: List[UcanCapability] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resource = str(item.get("resource", "") or "").strip() or "*"
        ability = str(item.get("ability", "") or "").strip() or "*"
        caps.append(UcanCapability(resource=resource, ability=ability))
    return tuple(caps)


def _parse_capabilities_from_token_payload(payload: Dict[str, Any]) -> Tuple[UcanCapability, ...]:
    """Parse capabilities from compact UCAN/JWT-style token payloads.

    Supported shapes:
    - `capabilities`: [{"resource": "...", "ability": "..."}, {"with": "...", "can": "..."}]
    - `att`: {"resource": {"ability": [...]}}
    """
    caps: List[UcanCapability] = []

    raw_caps = payload.get("capabilities")
    if isinstance(raw_caps, list):
        for item in raw_caps:
            if not isinstance(item, dict):
                continue
            resource = str(item.get("resource") or item.get("with") or "").strip() or "*"
            ability = str(item.get("ability") or item.get("can") or "").strip() or "*"
            caps.append(UcanCapability(resource=resource, ability=ability))

    att = payload.get("att")
    if isinstance(att, dict):
        for resource, grants in att.items():
            resource_text = str(resource or "").strip() or "*"
            if isinstance(grants, dict):
                for ability in grants.keys():
                    ability_text = str(ability or "").strip() or "*"
                    caps.append(UcanCapability(resource=resource_text, ability=ability_text))

    return tuple(caps)


def _decode_compact_token_payload(token: str) -> Dict[str, Any] | None:
    """Decode JWT/UCAN compact payload segment without validating signature."""
    text = str(token or "").strip()
    if not text or "." not in text:
        return None

    parts = text.split(".")
    if len(parts) < 2:
        return None

    payload_raw = parts[1]
    try:
        decoded = _b64_decode(payload_raw)
        payload = json.loads(decoded.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _extract_token_proof_cid(payload: Dict[str, Any]) -> str:
    """Extract parent proof CID-like linkage from token payload when present."""
    value = payload.get("prf")
    if isinstance(value, str):
        return str(value).strip()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return str(first).strip()
    return ""


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
        key = str(
            value.get("public_key")
            or value.get("public_key_b64")
            or value.get("public_key_base64")
            or value.get("did_key")
            or value.get("did")
            or value.get("key")
            or ""
        ).strip()
        if alg and alg not in {"ed25519", "eddsa"}:
            return ""
        if key.startswith("ed25519-pub:"):
            return key.split(":", 1)[1].strip()
        if key.startswith("did:key:"):
            key = _extract_ed25519_public_key_from_did_key(key)
        key_hex = str(value.get("public_key_hex") or "").strip()
        if key_hex:
            try:
                return _b64_encode_urlsafe(bytes.fromhex(key_hex))
            except ValueError:
                return ""
        return key

    text = str(value or "").strip()
    if text.startswith("ed25519-pub:"):
        return text.split(":", 1)[1].strip()
    if text.startswith("did:key:"):
        return _extract_ed25519_public_key_from_did_key(text)
    # Allow raw 32-byte hex public keys for interoperability.
    if len(text) == 64:
        try:
            return _b64_encode_urlsafe(bytes.fromhex(text))
        except ValueError:
            return ""
    return text


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58btc_decode(value: str) -> bytes:
    """Decode base58btc text into bytes."""
    text = str(value or "").strip()
    if not text:
        return b""

    acc = 0
    for ch in text:
        idx = _B58_ALPHABET.find(ch)
        if idx < 0:
            raise ValueError("invalid_base58btc_character")
        acc = (acc * 58) + idx

    raw = acc.to_bytes((acc.bit_length() + 7) // 8, "big") if acc else b""
    zeros = 0
    for ch in text:
        if ch != "1":
            break
        zeros += 1
    return (b"\x00" * zeros) + raw


def _extract_ed25519_public_key_from_did_key(value: str) -> str:
    """Extract raw Ed25519 public key bytes from `did:key:z...` and return b64url."""
    text = str(value or "").strip()
    if not text.startswith("did:key:"):
        return ""
    mb = text.split(":", 2)[-1].strip()
    if not mb.startswith("z"):
        return ""

    try:
        decoded = _base58btc_decode(mb[1:])
    except ValueError:
        return ""

    # did:key Ed25519 uses multicodec prefix 0xed01 + 32-byte public key.
    if len(decoded) >= 34 and decoded[0] == 0xED and decoded[1] == 0x01:
        return _b64_encode_urlsafe(decoded[2:34])

    return ""


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
    sig_b64 = ""
    if token.startswith("ed25519:"):
        sig_b64 = token.split(":", 1)[1].strip()
    elif token.startswith("ed25519-hex:"):
        try:
            sig_b64 = _b64_encode_urlsafe(bytes.fromhex(token.split(":", 1)[1].strip()))
        except ValueError:
            return False
    elif token.startswith("hex:"):
        try:
            sig_b64 = _b64_encode_urlsafe(bytes.fromhex(token.split(":", 1)[1].strip()))
        except ValueError:
            return False
    else:
        # Interop: treat raw 64-byte signature hex as Ed25519 signature bytes.
        if len(token) == 128:
            try:
                sig_b64 = _b64_encode_urlsafe(bytes.fromhex(token))
            except ValueError:
                return False
        else:
            return False

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
    actor: str,
    context_cids: Iterable[str] | None,
    now: float,
) -> bool:
    """Evaluate deterministic caveat subset.

    Supported caveats:
    - `not_before`: UNIX seconds
    - `not_after`: UNIX seconds
    - `resource_prefix`: required resource prefix
    - `resource_regex`: regex that must match resource
    - `ability`: required exact ability
    - `ability_in`: list/set of accepted abilities
    - `actor_equals`: required actor value
    - `actor_in`: list/set of accepted actors
    - `actor_regex`: regex that must match actor
    - `context_cids_all`: list of context CIDs that must all be present
    - `context_cids_any`: list of context CIDs where at least one must be present
    - `context_cids_none`: list of context CIDs that must not be present
    """
    context_set = {str(x or "").strip() for x in (context_cids or []) if str(x or "").strip()}

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
        regex = str(caveat.get("resource_regex") or "").strip()
        if regex:
            try:
                if re.fullmatch(regex, str(resource or "")) is None:
                    return False
            except re.error:
                return False
        required_ability = str(caveat.get("ability") or "").strip()
        if required_ability and required_ability != str(ability or ""):
            return False
        abilities = caveat.get("ability_in")
        if isinstance(abilities, (list, tuple, set)):
            allowed_abilities = {
                str(x or "").strip() for x in abilities if str(x or "").strip()
            }
            if allowed_abilities and str(ability or "") not in allowed_abilities:
                return False

        actor_equals = str(caveat.get("actor_equals") or "").strip()
        if actor_equals and actor_equals != str(actor or ""):
            return False

        actor_values = caveat.get("actor_in")
        if isinstance(actor_values, (list, tuple, set)):
            allowed_actors = {
                str(x or "").strip() for x in actor_values if str(x or "").strip()
            }
            if allowed_actors and str(actor or "") not in allowed_actors:
                return False

        actor_regex = str(caveat.get("actor_regex") or "").strip()
        if actor_regex:
            try:
                if re.fullmatch(actor_regex, str(actor or "")) is None:
                    return False
            except re.error:
                return False

        required_context = caveat.get("context_cids_all")
        if isinstance(required_context, (list, tuple, set)):
            required_context_set = {
                str(x or "").strip() for x in required_context if str(x or "").strip()
            }
            if required_context_set and not required_context_set.issubset(context_set):
                return False

        any_context = caveat.get("context_cids_any")
        if isinstance(any_context, (list, tuple, set)):
            any_context_set = {
                str(x or "").strip() for x in any_context if str(x or "").strip()
            }
            if any_context_set and context_set.isdisjoint(any_context_set):
                return False

        forbidden_context = caveat.get("context_cids_none")
        if isinstance(forbidden_context, (list, tuple, set)):
            forbidden_context_set = {
                str(x or "").strip() for x in forbidden_context if str(x or "").strip()
            }
            if forbidden_context_set and not context_set.isdisjoint(forbidden_context_set):
                return False
    return True


def parse_delegation_chain(raw_chain: Iterable[Dict[str, Any]]) -> List[UcanDelegation]:
    """Parse raw dict chain into typed delegation entries."""
    chain: List[UcanDelegation] = []
    for item in raw_chain:
        if not isinstance(item, dict):
            continue

        # Interop path: compact UCAN/JWT token envelope.
        token = str(item.get("token") or item.get("ucan") or item.get("jwt") or "").strip()
        if token:
            token_payload = _decode_compact_token_payload(token)
            if isinstance(token_payload, dict):
                issuer = str(item.get("issuer") or token_payload.get("iss") or "").strip()
                audience = str(item.get("audience") or token_payload.get("aud") or "").strip()
                expiry_raw = item.get("expiry")
                if expiry_raw is None:
                    expiry_raw = token_payload.get("exp")
                expiry: float | None = None
                if expiry_raw is not None:
                    try:
                        expiry = float(expiry_raw)
                    except (TypeError, ValueError):
                        expiry = None

                proof_cid = str(item.get("proof_cid") or "").strip() or _extract_token_proof_cid(token_payload)
                caveats_raw = item.get("caveats")
                if caveats_raw is None:
                    caveats_raw = token_payload.get("fct")
                caveats: tuple[Dict[str, Any], ...] = ()
                if isinstance(caveats_raw, (list, tuple)):
                    caveats = tuple(x for x in caveats_raw if isinstance(x, dict))

                chain.append(
                    UcanDelegation(
                        issuer=issuer,
                        audience=audience,
                        capabilities=_parse_capabilities_from_token_payload(token_payload),
                        expiry=expiry,
                        revoked=bool(item.get("revoked", False)),
                        proof_cid=proof_cid,
                        signature=str(item.get("signature", "") or "").strip(),
                        caveats=caveats,
                    )
                )
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
    context_cids: Iterable[str] | None = None,
) -> UcanValidationResult:
    """Validate a root->leaf delegation chain for execution authorization."""
    parsed = list(chain)
    if not parsed:
        return UcanValidationResult(False, "missing_delegation_chain", 0, [], None)

    t = float(now if now is not None else time.time())
    revoked_set = {str(x or "").strip() for x in (revoked_proof_cids or []) if str(x or "").strip()}
    key_map = dict(issuer_public_keys or {})
    proof_lineage = _compute_proof_lineage(parsed)

    def _result(allowed: bool, reason: str, failure_hop: int | None = None) -> UcanValidationResult:
        return UcanValidationResult(allowed, reason, len(parsed), list(proof_lineage), failure_hop)

    for idx, d in enumerate(parsed):
        if not d.issuer or not d.audience:
            return _result(False, f"invalid_principal_at_hop_{idx}", idx)
        if d.revoked:
            return _result(False, f"revoked_at_hop_{idx}", idx)
        if d.is_expired(now=t):
            return _result(False, f"expired_at_hop_{idx}", idx)
        if d.proof_cid and d.proof_cid in revoked_set:
            return _result(False, f"revoked_proof_at_hop_{idx}", idx)
        if not _caveats_allow(
            d.caveats,
            resource=resource,
            ability=ability,
            actor=actor,
            context_cids=context_cids,
            now=t,
        ):
            return _result(False, f"caveat_denied_at_hop_{idx}", idx)
        if require_signatures:
            if not d.proof_cid or not d.signature:
                return _result(False, f"missing_signature_at_hop_{idx}", idx)
            expected_proof = compute_delegation_proof_cid(d)
            if d.proof_cid != expected_proof:
                return _result(False, f"invalid_proof_cid_at_hop_{idx}", idx)

            issuer_key = key_map.get(d.issuer, "")
            sig_token = str(d.signature or "").strip()
            is_crypto_sig = (
                sig_token.startswith("ed25519:")
                or sig_token.startswith("ed25519-hex:")
                or sig_token.startswith("hex:")
                or len(sig_token) == 128
            )

            if is_crypto_sig:
                if not HAVE_CRYPTO_ED25519:
                    return _result(False, f"cryptography_unavailable_at_hop_{idx}", idx)
                if not verify_delegation_signature_ed25519(
                    delegation=d,
                    signature=sig_token,
                    public_key_b64=_extract_ed25519_public_key_b64(issuer_key),
                ):
                    return _result(False, f"invalid_signature_at_hop_{idx}", idx)
            else:
                expected_sig = compute_delegation_signature(
                    delegation=d,
                    issuer_key_hint=str(issuer_key or ""),
                )
                if d.signature != expected_sig:
                    return _result(False, f"invalid_signature_at_hop_{idx}", idx)

    # issuer/audience continuity: audience(i) == issuer(i+1)
    for idx in range(len(parsed) - 1):
        if parsed[idx].audience != parsed[idx + 1].issuer:
            return _result(False, f"broken_chain_at_hop_{idx}", idx + 1)

    # attenuation: each child must be subset of parent capabilities
    for idx in range(len(parsed) - 1):
        if not _covers(parsed[idx], parsed[idx + 1]):
            return _result(False, f"capability_escalation_at_hop_{idx+1}", idx + 1)

    leaf = parsed[-1]
    if actor and leaf.audience != actor:
        return _result(False, "actor_mismatch", len(parsed) - 1)

    granted = any(cap.matches(resource=resource, ability=ability) for cap in leaf.capabilities)
    if not granted:
        return _result(False, "capability_not_granted", len(parsed) - 1)

    return _result(True, "allowed", None)


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
    context_cids: Iterable[str] | None = None,
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
        context_cids=context_cids,
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
