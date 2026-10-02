"""Lossless datasets-to-supervisor cache *identity* correspondence.

Datasets owns ``CanonicalProofCacheKey@1`` semantics.  The supervisor's
``ProofCacheKey`` additionally names execution context.  Neither a provider
name nor a checker name is a verification receipt, and these two obligation
namespaces have no general, verified equivalence adapter.  Consequently this
bridge retains both identities but deliberately supplies no top-level
``obligation_id``: the existing FormalVerificationCache refuses receipt
admission for this identity-only envelope.  Single-flight coordination can use
its full key without acquiring proof authority.

The reviewed correspondence is exact field retention, not semantic inference:
all sixteen datasets dimensions live in the obligation's canonical payload;
all supervisor execution fields remain unchanged, and the original supervisor
obligation is nested intact.  In particular assumptions are not inferred from
premises, bounds from resource budgets, source from candidate trees,
translation from translator IDs, or checker from kernel markers.  A future
obligation-linking owner must separately establish those relationships before
positive proof reuse can be supported.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final

from .formal_verification_cache import ProofCacheKey
from .formal_verification_contracts import canonical_json

if TYPE_CHECKING:
    from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey


BRIDGE_SCHEMA: Final = "canonical-proof-cache-identity-bridge/v1"
MAX_BRIDGE_BYTES: Final = 1024 * 1024

# This closed versioned inventory is intentionally independent of owner runtime
# discovery: an added/removed native dimension needs a new bridge review.
IDENTITY_FIELDS: Final = (
    "source", "expression", "formalization", "slice", "obligation",
    "assumptions", "bounds", "translation", "provider", "environment",
    "policy", "schema", "checker", "network_policy", "evidence_kind",
    "authority_ceiling",
)


class CanonicalCacheBridgeError(ValueError):
    """The identity bridge is malformed or differs from the exact request."""


def _json(value: Any) -> str:
    try:
        encoded = canonical_json(value)
        if len(encoded.encode("utf-8")) > MAX_BRIDGE_BYTES:
            raise CanonicalCacheBridgeError("cache identity bridge exceeds byte limit")
        return encoded
    except (TypeError, ValueError, RecursionError) as error:
        if isinstance(error, CanonicalCacheBridgeError):
            raise
        raise CanonicalCacheBridgeError("cache identity bridge is not canonical JSON") from error


def _semantic(key: CanonicalProofCacheKey) -> CanonicalProofCacheKey:
    # Imports are local so the optional datasets dependency is needed only when
    # this cross-package API is called.  No structural/duck-typed fallback exists.
    from ipfs_datasets_py.logic.common.canonical_cache_key import (
        REQUIRED_IDENTITY_FIELDS,
        CanonicalProofCacheKey,
    )

    if type(key) is not CanonicalProofCacheKey:
        raise TypeError("an exact native CanonicalProofCacheKey is required")
    if tuple(REQUIRED_IDENTITY_FIELDS) != IDENTITY_FIELDS:
        raise CanonicalCacheBridgeError("native identity fields need a new bridge review")
    payload = key.to_dict()
    replay = CanonicalProofCacheKey.from_dict(payload)
    if _json(replay.to_dict()) != _json(payload):
        raise CanonicalCacheBridgeError("native canonical key did not replay exactly")
    return replay


def _execution(key: ProofCacheKey) -> ProofCacheKey:
    if type(key) is not ProofCacheKey:
        raise TypeError("an exact native supervisor ProofCacheKey is required")
    # Native keys contain mutable JSON values.  Reconstruct a detached snapshot
    # and require the complete closed native representation, without dropping
    # unknown fields or repairing malformed values.
    payload = json.loads(_json(key.to_dict()))
    replay = ProofCacheKey.from_dict(payload)
    if _json(replay.to_dict()) != _json(payload):
        raise CanonicalCacheBridgeError("native supervisor key did not replay exactly")
    return replay


def _correspondence() -> list[dict[str, str]]:
    return [
        {
            "datasets_field": name,
            "supervisor_field": f"obligation.canonical_key.{name}",
            "relation": "exact_identity_retention",
        }
        for name in IDENTITY_FIELDS
    ]


def bridge_canonical_proof_cache_key(
    semantic_key: CanonicalProofCacheKey, *, execution_key: ProofCacheKey,
) -> ProofCacheKey:
    """Retain both native keys without creating a proof obligation or receipt.

    ``execution_key`` is explicitly supplied context, not evidence that its
    source, premises or toolchain implement the datasets semantics.  The native
    canonical payload includes its schema/interface and optional source CID as
    well as all sixteen semantic fields.  No values are regenerated from IDs.
    """
    semantic = _semantic(semantic_key)
    execution = _execution(execution_key)
    if isinstance(execution.obligation, dict) and execution.obligation.get("schema") == BRIDGE_SCHEMA:
        raise CanonicalCacheBridgeError("nested identity bridges are not supported")
    payload = execution.to_dict()
    payload["obligation"] = {
        "schema": BRIDGE_SCHEMA,
        "canonical_key": semantic.to_dict(),
        "canonical_key_id": semantic.key_id,
        "execution_key_id": execution.key_id,
        "execution_obligation": execution.obligation,
        "field_correspondence": _correspondence(),
        "scope": {
            "identity_only": True,
            "semantic_equivalence_checked": False,
            "receipt_admission_supported": False,
            "proof_authority": False,
            "completion_authority": False,
        },
    }
    _json(payload)
    return ProofCacheKey.from_dict(payload)


def unbridge_canonical_proof_cache_key(
    bridged_key: ProofCacheKey,
    *,
    request: CanonicalProofCacheKey,
    expected_execution_key: ProofCacheKey,
) -> tuple[CanonicalProofCacheKey, ProofCacheKey]:
    """Recover both keys only after exact native request/context admission.

    This is cache *identity* admission, never proof or live-handle admission.
    Cross-environment requests raise the datasets owner's native error.  Every
    other semantic dimension and execution context must also match exactly.
    """
    from ipfs_datasets_py.logic.common.canonical_cache_key import (
        CanonicalProofCacheKey,
        admit_cache_hit,
    )

    requested = _semantic(request)
    expected = _execution(expected_execution_key)
    bridged = _execution(bridged_key)
    envelope = bridged.obligation
    required = {
        "schema", "canonical_key", "canonical_key_id", "execution_key_id",
        "execution_obligation", "field_correspondence", "scope",
    }
    if not isinstance(envelope, dict) or set(envelope) != required:
        raise CanonicalCacheBridgeError("identity bridge has missing or unknown fields")
    if envelope["schema"] != BRIDGE_SCHEMA or not isinstance(envelope["canonical_key"], dict):
        raise CanonicalCacheBridgeError("unsupported identity bridge payload")
    stored = CanonicalProofCacheKey.from_dict(envelope["canonical_key"])
    # The native loader permits omitted interface/schema and ignores a key_id
    # marker.  The bridge requires precisely native to_dict(), including the
    # actual interface and schema, so no ignored declaration survives replay.
    if _json(stored.to_dict()) != _json(envelope["canonical_key"]):
        raise CanonicalCacheBridgeError("canonical key payload is not exact native encoding")
    admit_cache_hit(stored, requested)
    recovered_payload = bridged.to_dict()
    recovered_payload["obligation"] = envelope["execution_obligation"]
    recovered = ProofCacheKey.from_dict(recovered_payload)
    if _json(recovered.to_dict()) != _json(expected.to_dict()):
        raise CanonicalCacheBridgeError("supervisor execution context differs from request")
    replay = bridge_canonical_proof_cache_key(stored, execution_key=recovered)
    if _json(replay.to_dict()) != _json(bridged.to_dict()):
        raise CanonicalCacheBridgeError("identity bridge did not replay exactly")
    return stored, recovered


__all__ = [
    "BRIDGE_SCHEMA", "IDENTITY_FIELDS", "MAX_BRIDGE_BYTES",
    "CanonicalCacheBridgeError", "bridge_canonical_proof_cache_key",
    "unbridge_canonical_proof_cache_key",
]
