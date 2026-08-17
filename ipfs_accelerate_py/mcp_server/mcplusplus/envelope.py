"""Runtime adapter for MCP++ ExecutionEnvelope@1 (RuntimeEnvelopeAdapter@1).

MCPP-034 / track envelope-accelerate: the accelerate MCP server can create,
content-address (``mcpp-jcs-v1``), persist by CID, and verify portable
execution envelopes without reimplementing the canonicalization or schema
suites owned by the shared MCP++ package.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Interface markers
# ---------------------------------------------------------------------------

INTERFACE = "RuntimeEnvelopeAdapter@1"
SCHEMA_ENVELOPE = "mcp++/execution/envelope@1"
CANONICALIZATION = "mcpp-jcs-v1"
TASK_ID = "MCPP-034"

_CID_RE = re.compile(r"^(Qm[1-9A-HJ-NP-Za-km-z]{44}|b[a-z2-7]{58,})$")
_DID_RE = re.compile(r"^did:[a-z0-9]+:[A-Za-z0-9._:%-]+(?:[/?#][^\x00]*)?$")

_ENVELOPE_REQUIRED = (
    "schema",
    "interface_cid",
    "input_cid",
    "intent_cid",
    "parents",
    "created_at_ms",
    "correlation_id",
    "requester",
    "authority",
)


class EnvelopeError(ValueError):
    """Fail-closed rejection while creating or verifying an envelope."""

    def __init__(self, reason_code: str, message: str, *, path: str = "") -> None:
        self.reason_code = reason_code
        self.path = path
        super().__init__(message if not path else f"{path}: {message}")


@dataclass
class EnvelopeVerificationResult:
    """Outcome of verifying one ExecutionEnvelope@1."""

    ok: bool
    cid: Optional[str] = None
    algorithm: str = CANONICALIZATION
    interface: str = INTERFACE
    schema: str = SCHEMA_ENVELOPE
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "cid": self.cid,
            "algorithm": self.algorithm,
            "interface": self.interface,
            "schema": self.schema,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class EmittedEnvelope:
    """Envelope plus its mcpp-jcs-v1 identity after mint."""

    envelope: Dict[str, Any]
    cid: str
    canonical_bytes: bytes
    algorithm: str = CANONICALIZATION

    def to_dict(self, *, include_canonical_bytes: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "envelope": copy.deepcopy(self.envelope),
            "cid": self.cid,
            "algorithm": self.algorithm,
            "byte_length": len(self.canonical_bytes),
        }
        if include_canonical_bytes:
            out["canonical_bytes_hex"] = self.canonical_bytes.hex()
        return out


# ---------------------------------------------------------------------------
# mcpp-jcs-v1 loader (reuse shared implementation; no local reimplementation)
# ---------------------------------------------------------------------------


def _mcplusplus_tests_py() -> Path:
    """Locate ``ipfs_accelerate_py/mcplusplus/tests-py`` for shared validators."""
    here = Path(__file__).resolve()
    # .../ipfs_accelerate_py/mcp_server/mcplusplus/envelope.py
    candidates = [
        here.parents[2] / "mcplusplus" / "tests-py",  # ipfs_accelerate_py/mcplusplus/tests-py
        here.parents[3] / "ipfs_accelerate_py" / "mcplusplus" / "tests-py",
    ]
    for parent in here.parents:
        cand = parent / "ipfs_accelerate_py" / "mcplusplus" / "tests-py"
        if cand not in candidates:
            candidates.append(cand)
        cand2 = parent / "mcplusplus" / "tests-py"
        if cand2 not in candidates:
            candidates.append(cand2)
    for path in candidates:
        if (path / "validators" / "canonical_jcs.py").is_file():
            return path
    raise EnvelopeError(
        "mcpp_jcs_unavailable",
        "shared mcpp-jcs-v1 implementation not found under mcplusplus/tests-py",
    )


def _load_canonical_jcs() -> Any:
    """Import the shared ``validators.canonical_jcs`` module (mcpp-jcs-v1)."""
    tests_py = _mcplusplus_tests_py()
    root = str(tests_py)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("validators.canonical_jcs")


def _load_envelope_validator() -> Any:
    """Import shared structural ``validate_envelope_v1`` when available."""
    tests_py = _mcplusplus_tests_py()
    root = str(tests_py)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("validators.envelope_profile_b")


# ---------------------------------------------------------------------------
# CID / DID helpers
# ---------------------------------------------------------------------------


def is_valid_cid(value: Any) -> bool:
    return isinstance(value, str) and bool(_CID_RE.match(value))


def is_valid_did(value: Any) -> bool:
    return isinstance(value, str) and bool(_DID_RE.match(value))


def _require_cid(value: Any, *, path: str) -> str:
    if not is_valid_cid(value):
        raise EnvelopeError("invalid_cid", f"invalid CID: {value!r}", path=path)
    return str(value)


def _require_did(value: Any, *, path: str) -> str:
    if not is_valid_did(value):
        raise EnvelopeError("invalid_did", f"invalid DID: {value!r}", path=path)
    return str(value)


def _as_cid_list(value: Any, *, path: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_require_cid(value, path=path)]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise EnvelopeError("type_error", "expected CID array", path=path)
    out: List[str] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        cid = _require_cid(item, path=f"{path}/{i}")
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _party(
    did: Any,
    *,
    path: str,
    key_id: Any = None,
    peer_id: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"did": _require_did(did, path=path)}
    if key_id is not None and key_id != "":
        out["key_id"] = str(key_id)
    if peer_id is not None and peer_id != "":
        out["peer_id"] = str(peer_id)
    return out


def _authority_block(
    *,
    proof_cids: Optional[Iterable[str]] = None,
    proof_cid: Optional[str] = None,
    delegation_cids: Optional[Iterable[str]] = None,
    resource: Optional[str] = None,
    ability: Optional[str] = None,
) -> Dict[str, Any]:
    proofs = _as_cid_list(list(proof_cids or []), path="/authority/proof_cids")
    primary: Optional[str] = None
    if proof_cid is not None and proof_cid != "":
        primary = _require_cid(proof_cid, path="/authority/proof_cid")
        if primary not in proofs:
            proofs = [primary, *proofs]
    elif proofs:
        primary = proofs[0]

    authority: Dict[str, Any] = {"proof_cids": proofs}
    if primary is not None:
        authority["proof_cid"] = primary
    else:
        # Empty proofs only for same-trust local execution (structural).
        authority["proof_cid"] = None
    if delegation_cids is not None:
        authority["delegation_cids"] = _as_cid_list(
            list(delegation_cids), path="/authority/delegation_cids"
        )
    if resource is not None:
        authority["resource"] = str(resource)
    if ability is not None:
        authority["ability"] = str(ability)
    return authority


# ---------------------------------------------------------------------------
# Create / canonicalize / CID
# ---------------------------------------------------------------------------


def create_envelope(
    *,
    interface_cid: str,
    input_cid: str,
    intent_cid: str,
    requester_did: str,
    correlation_id: str,
    method: Optional[str] = None,
    policy_cid: Optional[str] = None,
    decision_cid: Optional[str] = None,
    parents: Optional[Iterable[str]] = None,
    proof_cids: Optional[Iterable[str]] = None,
    proof_cid: Optional[str] = None,
    delegation_cids: Optional[Iterable[str]] = None,
    authority_resource: Optional[str] = None,
    authority_ability: Optional[str] = None,
    constraints: Optional[Mapping[str, Any]] = None,
    constraints_cid: Optional[str] = None,
    expected_output_schema_cid: Optional[str] = None,
    state_refs: Optional[Sequence[Mapping[str, Any]]] = None,
    audience_did: Optional[str] = None,
    requester_key_id: Optional[str] = None,
    requester_peer_id: Optional[str] = None,
    created_at_ms: Optional[int] = None,
    nonce: Optional[str] = None,
    metadata_cid: Optional[str] = None,
    profile_b_envelope_cid: Optional[str] = None,
    declared_side_effects: Optional[Sequence[str]] = None,
    deadline_ms: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structural ExecutionEnvelope@1 for the accelerate runtime.

    Does not re-encode Profile B/C/D/F/G documents; callers pass CID references.
    New mints always declare ``canonicalization: mcpp-jcs-v1``.
    """
    if not isinstance(correlation_id, str) or not (1 <= len(correlation_id) <= 128):
        raise EnvelopeError(
            "invalid_correlation_id",
            "correlation_id must be a string of length 1..128",
            path="/correlation_id",
        )
    if created_at_ms is None:
        created_at_ms = int(time.time() * 1000)
    if not isinstance(created_at_ms, int) or isinstance(created_at_ms, bool) or created_at_ms < 0:
        raise EnvelopeError(
            "invalid_created_at_ms",
            "created_at_ms must be a non-negative integer",
            path="/created_at_ms",
        )

    envelope: Dict[str, Any] = {
        "schema": SCHEMA_ENVELOPE,
        "interface_cid": _require_cid(interface_cid, path="/interface_cid"),
        "input_cid": _require_cid(input_cid, path="/input_cid"),
        "intent_cid": _require_cid(intent_cid, path="/intent_cid"),
        "parents": _as_cid_list(list(parents or []), path="/parents"),
        "created_at_ms": int(created_at_ms),
        "correlation_id": correlation_id,
        "requester": _party(
            requester_did,
            path="/requester/did",
            key_id=requester_key_id,
            peer_id=requester_peer_id,
        ),
        "authority": _authority_block(
            proof_cids=proof_cids,
            proof_cid=proof_cid,
            delegation_cids=delegation_cids,
            resource=authority_resource,
            ability=authority_ability,
        ),
        "canonicalization": CANONICALIZATION,
    }

    if method is not None:
        text = str(method)
        if not (1 <= len(text) <= 256):
            raise EnvelopeError("invalid_method", "method length must be 1..256", path="/method")
        envelope["method"] = text
    if policy_cid is not None and policy_cid != "":
        envelope["policy_cid"] = _require_cid(policy_cid, path="/policy_cid")
    if decision_cid is not None and decision_cid != "":
        envelope["decision_cid"] = _require_cid(decision_cid, path="/decision_cid")
    if constraints is not None:
        if not isinstance(constraints, Mapping):
            raise EnvelopeError("type_error", "constraints must be an object", path="/constraints")
        envelope["constraints"] = copy.deepcopy(dict(constraints))
    if constraints_cid is not None and constraints_cid != "":
        envelope["constraints_cid"] = _require_cid(constraints_cid, path="/constraints_cid")
    if expected_output_schema_cid is not None and expected_output_schema_cid != "":
        envelope["expected_output_schema_cid"] = _require_cid(
            expected_output_schema_cid, path="/expected_output_schema_cid"
        )
    if state_refs is not None:
        envelope["state_refs"] = [copy.deepcopy(dict(item)) for item in state_refs]
    else:
        envelope["state_refs"] = []
    if audience_did is not None and audience_did != "":
        envelope["audience"] = _party(audience_did, path="/audience/did")
    if nonce is not None:
        envelope["nonce"] = str(nonce)
    if metadata_cid is not None and metadata_cid != "":
        envelope["metadata_cid"] = _require_cid(metadata_cid, path="/metadata_cid")
    if profile_b_envelope_cid is not None and profile_b_envelope_cid != "":
        envelope["profile_b_envelope_cid"] = _require_cid(
            profile_b_envelope_cid, path="/profile_b_envelope_cid"
        )
    if declared_side_effects is not None:
        envelope["declared_side_effects"] = [str(x) for x in declared_side_effects]
    if deadline_ms is not None:
        if not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool) or deadline_ms < 0:
            raise EnvelopeError(
                "invalid_deadline_ms",
                "deadline_ms must be a non-negative integer",
                path="/deadline_ms",
            )
        envelope["deadline_ms"] = int(deadline_ms)
    if extra:
        for key, value in extra.items():
            if key in envelope:
                continue
            envelope[str(key)] = copy.deepcopy(value)

    return envelope


def canonicalize_envelope(envelope: Mapping[str, Any]) -> bytes:
    """Return ``mcpp-jcs-v1`` (RFC 8785 JCS) UTF-8 bytes for *envelope*."""
    if not isinstance(envelope, Mapping):
        raise EnvelopeError("type_error", "envelope must be an object")
    jcs = _load_canonical_jcs()
    if getattr(jcs, "ALGORITHM_ID", CANONICALIZATION) != CANONICALIZATION:
        raise EnvelopeError(
            "algorithm_mismatch",
            f"expected algorithm {CANONICALIZATION!r}, got {getattr(jcs, 'ALGORITHM_ID', None)!r}",
        )
    return jcs.canonicalize_bytes(dict(envelope))


def compute_envelope_cid(envelope: Mapping[str, Any]) -> str:
    """Content-address *envelope* under ``mcpp-jcs-v1`` (CIDv1 raw + sha2-256)."""
    if not isinstance(envelope, Mapping):
        raise EnvelopeError("type_error", "envelope must be an object")
    jcs = _load_canonical_jcs()
    return str(jcs.artifact_cid(dict(envelope)))


def envelope_identity(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Return algorithm, canonical bytes digest, and CID for *envelope*."""
    if not isinstance(envelope, Mapping):
        raise EnvelopeError("type_error", "envelope must be an object")
    jcs = _load_canonical_jcs()
    ident = jcs.identity(dict(envelope))
    if hasattr(ident, "as_dict"):
        return dict(ident.as_dict())
    return {
        "algorithm": getattr(ident, "algorithm", CANONICALIZATION),
        "canonical_utf8": getattr(ident, "canonical_utf8", None),
        "canonical_sha256": getattr(ident, "sha256", None),
        "cid": getattr(ident, "cid", None),
    }


def emit_envelope(**kwargs: Any) -> EmittedEnvelope:
    """Create an envelope and compute its ``mcpp-jcs-v1`` CID in one step."""
    envelope = create_envelope(**kwargs)
    jcs = _load_canonical_jcs()
    ident = jcs.identity(envelope)
    return EmittedEnvelope(
        envelope=envelope,
        cid=str(ident.cid),
        canonical_bytes=bytes(ident.canonical_bytes),
        algorithm=str(getattr(ident, "algorithm", CANONICALIZATION)),
    )


# ---------------------------------------------------------------------------
# Structural + CID verification
# ---------------------------------------------------------------------------


def _structural_errors(envelope: Mapping[str, Any]) -> List[str]:
    """Lightweight structural checks (fail-closed; mirrors shared required set)."""
    errors: List[str] = []
    if not isinstance(envelope, Mapping):
        return ["envelope must be an object"]

    if envelope.get("schema") != SCHEMA_ENVELOPE:
        errors.append(f"schema must be {SCHEMA_ENVELOPE!r}")

    for key in _ENVELOPE_REQUIRED:
        if key not in envelope:
            errors.append(f"missing required field: {key}")

    for key in (
        "interface_cid",
        "input_cid",
        "intent_cid",
        "policy_cid",
        "decision_cid",
        "constraints_cid",
        "expected_output_schema_cid",
        "metadata_cid",
        "profile_b_envelope_cid",
    ):
        if key in envelope and envelope[key] is not None and not is_valid_cid(envelope[key]):
            errors.append(f"invalid CID at /{key}")

    parents = envelope.get("parents")
    if "parents" in envelope:
        if not isinstance(parents, list):
            errors.append("parents must be an array")
        else:
            for i, parent in enumerate(parents):
                if not is_valid_cid(parent):
                    errors.append(f"invalid parent CID at /parents/{i}")

    created = envelope.get("created_at_ms")
    if "created_at_ms" in envelope and (
        not isinstance(created, int) or isinstance(created, bool) or created < 0
    ):
        errors.append("created_at_ms must be a non-negative integer")

    corr = envelope.get("correlation_id")
    if "correlation_id" in envelope and (
        not isinstance(corr, str) or not (1 <= len(corr) <= 128)
    ):
        errors.append("correlation_id must be a string of length 1..128")

    requester = envelope.get("requester")
    if "requester" in envelope:
        if not isinstance(requester, Mapping) or not is_valid_did(requester.get("did")):
            errors.append("requester.did must be a valid DID")

    authority = envelope.get("authority")
    if "authority" in envelope:
        if not isinstance(authority, Mapping):
            errors.append("authority must be an object")
        elif "proof_cids" not in authority:
            errors.append("authority.proof_cids is required")
        elif not isinstance(authority.get("proof_cids"), list):
            errors.append("authority.proof_cids must be an array")
        else:
            for i, cid in enumerate(authority["proof_cids"]):
                if not is_valid_cid(cid):
                    errors.append(f"invalid CID at /authority/proof_cids/{i}")
            if authority.get("proof_cid") is not None and not is_valid_cid(authority["proof_cid"]):
                errors.append("invalid CID at /authority/proof_cid")

    canon = envelope.get("canonicalization")
    if "canonicalization" in envelope and canon not in (None, CANONICALIZATION):
        errors.append(f"canonicalization must be {CANONICALIZATION!r} when present")

    return errors


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_cid: Optional[str] = None,
    use_shared_validator: bool = True,
) -> EnvelopeVerificationResult:
    """Verify structural shape and recompute the ``mcpp-jcs-v1`` CID.

    When *expected_cid* is provided, the recomputed CID must match exactly.
    """
    result = EnvelopeVerificationResult(ok=True)
    result.metadata["task_id"] = TASK_ID

    errors = _structural_errors(envelope)
    for err in errors:
        result.errors.append(err)

    if use_shared_validator:
        try:
            mod = _load_envelope_validator()
            shared = mod.validate_envelope_v1(dict(envelope))
            if hasattr(shared, "is_valid") and not shared.is_valid:
                for err in getattr(shared, "errors", []) or []:
                    text = str(err)
                    if text not in result.errors:
                        result.errors.append(text)
            for warn in getattr(shared, "warnings", []) or []:
                result.warnings.append(str(warn))
            result.metadata["shared_validator"] = "envelope_profile_b.validate_envelope_v1"
        except Exception as exc:  # pragma: no cover - optional shared path
            result.warnings.append(f"shared_validator_unavailable:{type(exc).__name__}")
            result.metadata["shared_validator_error"] = repr(exc)

    try:
        cid = compute_envelope_cid(envelope)
        result.cid = cid
        result.metadata["algorithm"] = CANONICALIZATION
        if expected_cid is not None:
            if not is_valid_cid(expected_cid):
                result.errors.append(f"invalid expected_cid: {expected_cid!r}")
            elif expected_cid != cid:
                result.errors.append(
                    f"cid_mismatch: expected {expected_cid!r}, recomputed {cid!r}"
                )
    except Exception as exc:
        result.errors.append(f"cid_compute_failed:{type(exc).__name__}:{exc}")

    result.ok = not result.errors
    return result


# ---------------------------------------------------------------------------
# Persistence by CID
# ---------------------------------------------------------------------------


class EnvelopeStore:
    """Thread-safe in-memory envelope store keyed by mcpp-jcs-v1 CID."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_cid: Dict[str, Dict[str, Any]] = {}

    def put(self, cid: str, envelope: Mapping[str, Any]) -> str:
        """Persist *envelope* under *cid* after verifying CID binding."""
        key = str(cid or "").strip()
        if not is_valid_cid(key):
            raise EnvelopeError("invalid_cid", f"invalid store CID: {cid!r}")
        if not isinstance(envelope, Mapping):
            raise EnvelopeError("type_error", "envelope must be an object")
        recomputed = compute_envelope_cid(envelope)
        if recomputed != key:
            raise EnvelopeError(
                "cid_mismatch",
                f"provided cid {key!r} does not match mcpp-jcs-v1 cid {recomputed!r}",
            )
        with self._lock:
            self._by_cid[key] = copy.deepcopy(dict(envelope))
        return key

    def put_envelope(self, envelope: Mapping[str, Any]) -> str:
        """Compute CID under mcpp-jcs-v1 and persist; return the CID."""
        cid = compute_envelope_cid(envelope)
        return self.put(cid, envelope)

    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        key = str(cid or "").strip()
        with self._lock:
            payload = self._by_cid.get(key)
            return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def has(self, cid: str) -> bool:
        key = str(cid or "").strip()
        with self._lock:
            return key in self._by_cid

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"envelope_count": int(len(self._by_cid))}

    def export_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                cid: copy.deepcopy(self._by_cid[cid]) for cid in sorted(self._by_cid.keys())
            }

    def verify_stored(self, cid: str) -> EnvelopeVerificationResult:
        """Load by CID and verify structure + recomputed mcpp-jcs-v1 CID."""
        payload = self.get(cid)
        if payload is None:
            return EnvelopeVerificationResult(
                ok=False,
                cid=None,
                errors=[f"envelope_not_found:{cid}"],
            )
        return verify_envelope(payload, expected_cid=str(cid))

    def save_json(self, file_path: str) -> int:
        target = Path(str(file_path)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.export_records()
        target.write_text(
            json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        return len(records)

    @classmethod
    def load_json(cls, file_path: str) -> "EnvelopeStore":
        store = cls()
        source = Path(str(file_path)).expanduser()
        if not source.exists():
            return store
        try:
            parsed = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return store
        if isinstance(parsed, dict):
            for cid, payload in parsed.items():
                if isinstance(payload, dict) and is_valid_cid(cid):
                    try:
                        store.put(str(cid), payload)
                    except EnvelopeError:
                        continue
        return store


# ---------------------------------------------------------------------------
# Adapter facade
# ---------------------------------------------------------------------------


class RuntimeEnvelopeAdapter:
    """Accelerate runtime facade: create → CID → persist → verify."""

    interface = INTERFACE
    algorithm = CANONICALIZATION
    schema = SCHEMA_ENVELOPE

    def __init__(self, store: Optional[EnvelopeStore] = None) -> None:
        self.store = store if store is not None else EnvelopeStore()

    def create(self, **kwargs: Any) -> Dict[str, Any]:
        return create_envelope(**kwargs)

    def emit(self, **kwargs: Any) -> EmittedEnvelope:
        return emit_envelope(**kwargs)

    def compute_cid(self, envelope: Mapping[str, Any]) -> str:
        return compute_envelope_cid(envelope)

    def canonicalize(self, envelope: Mapping[str, Any]) -> bytes:
        return canonicalize_envelope(envelope)

    def verify(
        self,
        envelope: Mapping[str, Any],
        *,
        expected_cid: Optional[str] = None,
    ) -> EnvelopeVerificationResult:
        return verify_envelope(envelope, expected_cid=expected_cid)

    def persist(self, envelope: Mapping[str, Any]) -> str:
        return self.store.put_envelope(envelope)

    def emit_and_persist(self, **kwargs: Any) -> EmittedEnvelope:
        emitted = self.emit(**kwargs)
        self.store.put(emitted.cid, emitted.envelope)
        return emitted

    def load(self, cid: str) -> Optional[Dict[str, Any]]:
        return self.store.get(cid)

    def verify_stored(self, cid: str) -> EnvelopeVerificationResult:
        return self.store.verify_stored(cid)


__all__ = [
    "INTERFACE",
    "SCHEMA_ENVELOPE",
    "CANONICALIZATION",
    "TASK_ID",
    "EnvelopeError",
    "EnvelopeVerificationResult",
    "EmittedEnvelope",
    "EnvelopeStore",
    "RuntimeEnvelopeAdapter",
    "is_valid_cid",
    "is_valid_did",
    "create_envelope",
    "canonicalize_envelope",
    "compute_envelope_cid",
    "envelope_identity",
    "emit_envelope",
    "verify_envelope",
]
