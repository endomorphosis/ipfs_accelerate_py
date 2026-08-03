"""Signed, fenced resume cache for legacy landed provider-leaf evidence.

The live cache is a same-host DuckDB coordination shard.  Provider calls run
outside its process-shared file lock under short, renewable per-key leases.
Only an exact approved leaf receipt signed by the operator's distinct legacy
review key is durable or reusable.  Validation receipts, review aggregates,
and final task attestations are deliberately not cached: every resumed review
gets a fresh run ID, fresh validations, and a fresh final attestation.

Parquet/IPLD snapshots are immutable, content-addressed replicas of signed
leaf records.  They never grant a lease, completion authority, or proof
authority and never replace the local DuckDB owner.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..entrypoints.verified_ipld_backend import (
    VerifiedIPLDBackend,
    admit_cid,
)
from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ..task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    initialize_duckdb_database,
    open_duckdb_connection,
    resolve_duckdb_path,
)
from .legacy_landed_attestation import (
    _read_private_key,
    legacy_landed_review_key_id,
)
from .legacy_landed_review import (
    LEGACY_LANDED_LEAF_DECISION_SCHEMA,
    LEGACY_LANDED_LEAF_REVIEW_RECEIPT_SCHEMA,
    LegacyLandedReviewError,
    LegacyLandedReviewPolicy,
    LegacyLeafReviewRequest,
    LegacyProviderInvoker,
    LegacyProviderPolicy,
    LegacyTaskPolicy,
    _leaf_review_request,
    _review_one_leaf,
)

LEGACY_LANDED_LEAF_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-cache-key@1"
)
LEGACY_LANDED_LEAF_CACHE_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-cache-record@1"
)
LEGACY_LANDED_LEAF_CACHE_RECORD_INTERFACE: Final = (
    "LegacyLandedLeafCacheRecord@1"
)
LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA_V1: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-cache-snapshot@1"
)
LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-leaf-cache-snapshot@2"
)
LEGACY_LANDED_LEAF_CACHE_SIGNATURE_ALGORITHM: Final = "Ed25519"
DEFAULT_LEAF_LEASE_SECONDS: Final = 360
DEFAULT_LEAF_WAIT_SECONDS: Final = 900.0
DEFAULT_POLL_SECONDS: Final = 0.02
MAX_SNAPSHOT_RECORDS: Final = 100_000
MAX_SNAPSHOT_MANIFEST_BYTES: Final = 32 * 1024 * 1024
MAX_SNAPSHOT_PARQUET_BYTES: Final = 512 * 1024 * 1024
MAX_SNAPSHOT_ROW_JSON_BYTES: Final = 256 * 1024

_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")


class LegacyLandedLeafCacheError(RuntimeError):
    """Fail-closed cache corruption, fencing, or timeout error."""


class LegacyLandedLeafCacheTimeout(LegacyLandedLeafCacheError, TimeoutError):
    """No signed leaf record arrived before the bounded deadline."""


def _strict_object(value: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = item
        return result

    if isinstance(value, Mapping):
        raw = canonical_json_bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise ValueError("canonical JSON object is required")
    parsed = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {item}")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError("canonical JSON object is required")
    return parsed


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _commit(value: Any, field: str) -> str:
    result = _text(value, field)
    if not _COMMIT_RE.fullmatch(result):
        raise ValueError(f"{field} must be a full Git object ID")
    return result


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    encoded = _text(value, "base64url value")
    return base64.b64decode(
        encoded + "=" * (-len(encoded) % 4),
        altchars=b"-_",
        validate=True,
    )


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheKey:
    """Closed identity of every input capable of changing one leaf review."""

    policy_id: str
    task_id: str
    canonical_task_key: str
    canonical_task_cid: str
    manifest_id: str
    manifest_merkle_root: str
    leaf_index: int
    leaf_id: str
    request_id: str
    request_cid: str
    role: str
    provider: str
    model: str
    current_head: str
    current_tree_id: str

    def __post_init__(self) -> None:
        for field in (
            "policy_id",
            "task_id",
            "canonical_task_key",
            "canonical_task_cid",
            "manifest_id",
            "manifest_merkle_root",
            "leaf_id",
            "request_id",
            "request_cid",
            "role",
            "provider",
            "model",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(
            self, "current_head", _commit(self.current_head, "current_head")
        )
        object.__setattr__(
            self,
            "current_tree_id",
            _commit(self.current_tree_id, "current_tree_id"),
        )
        if (
            isinstance(self.leaf_index, bool)
            or not isinstance(self.leaf_index, int)
            or self.leaf_index < 0
        ):
            raise ValueError("leaf_index must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEGACY_LANDED_LEAF_CACHE_KEY_SCHEMA,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "canonical_task_key": self.canonical_task_key,
            "canonical_task_cid": self.canonical_task_cid,
            "manifest_id": self.manifest_id,
            "manifest_merkle_root": self.manifest_merkle_root,
            "leaf_index": self.leaf_index,
            "leaf_id": self.leaf_id,
            "request_id": self.request_id,
            "request_cid": self.request_cid,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "current_head": self.current_head,
            "current_tree_id": self.current_tree_id,
        }

    @property
    def key_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> LegacyLandedLeafCacheKey:
        payload = _strict_object(value)
        fields = {
            "policy_id",
            "task_id",
            "canonical_task_key",
            "canonical_task_cid",
            "manifest_id",
            "manifest_merkle_root",
            "leaf_index",
            "leaf_id",
            "request_id",
            "request_cid",
            "role",
            "provider",
            "model",
            "current_head",
            "current_tree_id",
        }
        if payload.get("schema") != LEGACY_LANDED_LEAF_CACHE_KEY_SCHEMA:
            raise ValueError("legacy leaf cache key schema is invalid")
        if set(payload) != {"schema", *fields}:
            raise ValueError("legacy leaf cache key shape is invalid")
        return cls(**{field: payload[field] for field in fields})

    @classmethod
    def from_request(
        cls,
        *,
        policy: LegacyLandedReviewPolicy,
        task: LegacyTaskPolicy,
        manifest: Mapping[str, Any],
        leaf: Mapping[str, Any],
        provider: LegacyProviderPolicy,
        request: LegacyLeafReviewRequest,
    ) -> LegacyLandedLeafCacheKey:
        expected = _leaf_review_request(
            policy=policy,
            task=task,
            manifest=manifest,
            leaf=leaf,
            provider=provider,
        )
        if canonical_json_bytes(expected.to_dict()) != canonical_json_bytes(
            request.to_dict()
        ):
            raise ValueError("legacy leaf request differs from audited envelope")
        return cls(
            policy_id=policy.policy_id,
            task_id=task.task_id,
            canonical_task_key=task.canonical_task_key,
            canonical_task_cid=task.canonical_task_cid,
            manifest_id=_text(manifest.get("manifest_id"), "manifest_id"),
            manifest_merkle_root=_text(
                manifest.get("merkle_root"), "manifest_merkle_root"
            ),
            leaf_index=leaf.get("leaf_index"),
            leaf_id=_text(leaf.get("leaf_id"), "leaf_id"),
            request_id=request.request_id,
            request_cid=content_identity(request.to_dict()),
            role=provider.role,
            provider=provider.provider,
            model=provider.model,
            current_head=policy.current_head,
            current_tree_id=policy.current_tree_id,
        )


_LEAF_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "receipt_id",
        "review_run_id",
        "role",
        "request_id",
        "request_token_upper_bound",
        "manifest_id",
        "leaf_index",
        "leaf_id",
        "requested_provider",
        "requested_model",
        "effective_provider",
        "effective_model",
        "provider_chain",
        "fallback_used",
        "self_review",
        "supervisor_observed",
        "observation_id",
        "response",
        "response_id",
        "approved",
        "completion_authoritative",
        "proof_authoritative",
        # Present after cache integration; old cold records remain parseable.
        "provider_evidence_source",
        "provider_invoked_in_current_run",
        "provider_evidence_cache_record",
    }
)


def _verified_origin_leaf_receipt(
    key: LegacyLandedLeafCacheKey,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _strict_object(value)
    required = _LEAF_RECEIPT_FIELDS - {
        "provider_evidence_source",
        "provider_invoked_in_current_run",
        "provider_evidence_cache_record",
    }
    if not required.issubset(receipt) or set(receipt) - _LEAF_RECEIPT_FIELDS:
        raise ValueError("legacy cached leaf receipt shape is invalid")
    body = dict(receipt)
    receipt_id = _text(body.pop("receipt_id", ""), "receipt_id")
    if receipt_id != content_identity(body):
        raise ValueError("legacy cached leaf receipt identity is invalid")
    expected = {
        "schema": LEGACY_LANDED_LEAF_REVIEW_RECEIPT_SCHEMA,
        "role": key.role,
        "request_id": key.request_id,
        "manifest_id": key.manifest_id,
        "leaf_index": key.leaf_index,
        "leaf_id": key.leaf_id,
        "requested_provider": key.provider,
        "requested_model": key.model,
        "effective_provider": key.provider,
        "effective_model": key.model,
        "provider_chain": [key.provider],
        "fallback_used": False,
        "self_review": False,
        "supervisor_observed": True,
        "approved": True,
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise ValueError(f"legacy cached leaf receipt {field} mismatch")
    run_id = receipt.get("review_run_id")
    if not isinstance(run_id, str) or len(run_id) < 16:
        raise ValueError("legacy cached leaf review run is invalid")
    token_bound = receipt.get("request_token_upper_bound")
    if (
        isinstance(token_bound, bool)
        or not isinstance(token_bound, int)
        or token_bound < 1
        or token_bound > 4_096
    ):
        raise ValueError("legacy cached request bound is invalid")
    if not isinstance(receipt.get("observation_id"), str) or not receipt[
        "observation_id"
    ]:
        raise ValueError("legacy cached observation is missing")
    response = receipt.get("response")
    if not isinstance(response, Mapping):
        raise ValueError("legacy cached provider response is invalid")
    expected_response = {
        "schema": LEGACY_LANDED_LEAF_DECISION_SCHEMA,
        "decision": "approve",
        "manifest_id": key.manifest_id,
        "leaf_id": key.leaf_id,
        "findings": [],
    }
    if dict(response) != expected_response:
        raise ValueError("legacy cached provider response is not exact approval")
    if receipt.get("response_id") != content_identity(response):
        raise ValueError("legacy cached provider response identity is invalid")
    source = receipt.get("provider_evidence_source")
    if source not in (None, "fresh_provider"):
        raise ValueError("only a fresh provider receipt may seed the cache")
    if source == "fresh_provider" and (
        receipt.get("provider_invoked_in_current_run") is not True
        or receipt.get("provider_evidence_cache_record") is not None
    ):
        raise ValueError("fresh provider provenance is invalid")
    return receipt


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheRecord:
    key: LegacyLandedLeafCacheKey
    receipt: Mapping[str, Any]
    issuer_key_id: str
    issued_at_ms: int
    nonce: str
    signature: str
    record_id: str

    def unsigned_dict(self) -> dict[str, Any]:
        receipt = _verified_origin_leaf_receipt(self.key, self.receipt)
        return {
            "schema": LEGACY_LANDED_LEAF_CACHE_RECORD_SCHEMA,
            "interface": LEGACY_LANDED_LEAF_CACHE_RECORD_INTERFACE,
            "signature_algorithm": LEGACY_LANDED_LEAF_CACHE_SIGNATURE_ALGORITHM,
            "key_id": self.key.key_id,
            "key": self.key.to_dict(),
            "receipt_id": receipt["receipt_id"],
            "receipt_cid": content_identity(receipt),
            "receipt": receipt,
            "issuer_key_id": self.issuer_key_id,
            "issued_at_ms": self.issued_at_ms,
            "nonce": self.nonce,
            "provider_evidence_only": True,
            "validation_cached": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "record_id": self.record_id,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> LegacyLandedLeafCacheRecord:
        payload = _strict_object(value)
        expected = {
            "schema",
            "interface",
            "signature_algorithm",
            "key_id",
            "key",
            "receipt_id",
            "receipt_cid",
            "receipt",
            "issuer_key_id",
            "issued_at_ms",
            "nonce",
            "provider_evidence_only",
            "validation_cached",
            "completion_authoritative",
            "proof_authoritative",
            "record_id",
            "signature",
        }
        if set(payload) != expected:
            raise ValueError("legacy leaf cache record shape is invalid")
        if (
            payload.get("schema") != LEGACY_LANDED_LEAF_CACHE_RECORD_SCHEMA
            or payload.get("interface")
            != LEGACY_LANDED_LEAF_CACHE_RECORD_INTERFACE
            or payload.get("signature_algorithm")
            != LEGACY_LANDED_LEAF_CACHE_SIGNATURE_ALGORITHM
            or payload.get("provider_evidence_only") is not True
            or payload.get("validation_cached") is not False
            or payload.get("completion_authoritative") is not False
            or payload.get("proof_authoritative") is not False
        ):
            raise ValueError("legacy leaf cache record authority shape is invalid")
        if not isinstance(payload.get("key"), Mapping) or not isinstance(
            payload.get("receipt"), Mapping
        ):
            raise ValueError("legacy leaf cache record payload is invalid")
        key = LegacyLandedLeafCacheKey.from_dict(payload["key"])
        receipt = _verified_origin_leaf_receipt(key, payload["receipt"])
        if payload.get("key_id") != key.key_id:
            raise ValueError("legacy leaf cache key identity is invalid")
        if payload.get("receipt_id") != receipt["receipt_id"]:
            raise ValueError("legacy leaf cache receipt binding is invalid")
        if payload.get("receipt_cid") != content_identity(receipt):
            raise ValueError("legacy leaf cache receipt CID is invalid")
        issued = payload.get("issued_at_ms")
        if isinstance(issued, bool) or not isinstance(issued, int) or issued < 1:
            raise ValueError("legacy leaf cache issue time is invalid")
        return cls(
            key=key,
            receipt=receipt,
            issuer_key_id=_text(payload.get("issuer_key_id"), "issuer_key_id"),
            issued_at_ms=issued,
            nonce=_text(payload.get("nonce"), "nonce"),
            signature=_text(payload.get("signature"), "signature"),
            record_id=_text(payload.get("record_id"), "record_id"),
        )


class LegacyLandedLeafCacheAuthority:
    """Domain-separated signer backed by the strict legacy private-key loader."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key

    @classmethod
    def from_private_key_path(
        cls, path: str | Path
    ) -> LegacyLandedLeafCacheAuthority:
        return cls(Ed25519PrivateKey.from_private_bytes(_read_private_key(path)))

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes_raw()

    @property
    def issuer_key_id(self) -> str:
        return legacy_landed_review_key_id(self.public_key_bytes)

    def issue(
        self,
        key: LegacyLandedLeafCacheKey,
        receipt: Mapping[str, Any],
        *,
        issued_at_ms: int | None = None,
    ) -> LegacyLandedLeafCacheRecord:
        verified_receipt = _verified_origin_leaf_receipt(key, receipt)
        provisional = LegacyLandedLeafCacheRecord(
            key=key,
            receipt=verified_receipt,
            issuer_key_id=self.issuer_key_id,
            issued_at_ms=(
                int(time.time() * 1000)
                if issued_at_ms is None
                else int(issued_at_ms)
            ),
            nonce=secrets.token_urlsafe(24).rstrip("="),
            signature="pending",
            record_id="pending",
        )
        unsigned = provisional.unsigned_dict()
        signature = _b64(self._private_key.sign(canonical_json_bytes(unsigned)))
        record_id = content_identity({**unsigned, "signature": signature})
        return LegacyLandedLeafCacheRecord(
            key=key,
            receipt=verified_receipt,
            issuer_key_id=self.issuer_key_id,
            issued_at_ms=provisional.issued_at_ms,
            nonce=provisional.nonce,
            signature=signature,
            record_id=record_id,
        )


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheVerification:
    verified: bool
    reason_codes: tuple[str, ...]
    record: LegacyLandedLeafCacheRecord | None = None


def verify_legacy_landed_leaf_cache_record(
    value: LegacyLandedLeafCacheRecord | Mapping[str, Any],
    *,
    expected_key: LegacyLandedLeafCacheKey,
    trusted_public_keys: Mapping[str, bytes | str],
) -> LegacyLandedLeafCacheVerification:
    failures: list[str] = []
    try:
        record = (
            value
            if isinstance(value, LegacyLandedLeafCacheRecord)
            else LegacyLandedLeafCacheRecord.from_dict(value)
        )
        # Reparse typed values too; callers cannot bypass closed-shape checks.
        record = LegacyLandedLeafCacheRecord.from_dict(record.to_dict())
    except (TypeError, ValueError, json.JSONDecodeError):
        return LegacyLandedLeafCacheVerification(
            False, ("legacy_leaf_cache_record_malformed",)
        )
    if canonical_json_bytes(record.key.to_dict()) != canonical_json_bytes(
        expected_key.to_dict()
    ):
        failures.append("legacy_leaf_cache_key_mismatch")
    public_value = trusted_public_keys.get(record.issuer_key_id)
    if public_value is None:
        failures.append("legacy_leaf_cache_issuer_untrusted")
        public_bytes = b""
    else:
        try:
            public_bytes = (
                public_value
                if isinstance(public_value, bytes)
                else _unb64(public_value)
            )
            if legacy_landed_review_key_id(public_bytes) != record.issuer_key_id:
                raise ValueError("public key ID mismatch")
        except (TypeError, ValueError):
            failures.append("legacy_leaf_cache_public_key_invalid")
            public_bytes = b""
    unsigned = record.unsigned_dict()
    if record.record_id != content_identity(
        {**unsigned, "signature": record.signature}
    ):
        failures.append("legacy_leaf_cache_record_id_invalid")
    if public_bytes:
        try:
            Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                _unb64(record.signature), canonical_json_bytes(unsigned)
            )
        except (InvalidSignature, TypeError, ValueError):
            failures.append("legacy_leaf_cache_signature_invalid")
    reasons = tuple(dict.fromkeys(failures))
    return LegacyLandedLeafCacheVerification(not reasons, reasons, record)


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheLease:
    key_id: str
    owner_id: str
    token: str
    fencing_token: int
    acquired_at_ms: int
    expires_at_ms: int
    acquired: bool


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheReview:
    receipt: Mapping[str, Any]
    cache_hit: bool
    cache_record_id: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class LegacyLandedLeafCacheSnapshot:
    manifest_cid: str
    parquet_cid: str
    parquet_path: Path
    row_count: int
    manifest: Mapping[str, Any]


class LegacyLandedLeafResultCache:
    """Same-host DuckDB store for exact signed provider-leaf evidence."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        policy: LegacyLandedReviewPolicy,
        operator_key_path: str | Path,
        clock: Callable[[], float] = time.time,
        duckdb_timeout_seconds: int = 30,
    ) -> None:
        if not isinstance(policy, LegacyLandedReviewPolicy):
            raise TypeError("parsed legacy landed review policy is required")
        self.policy = policy
        self.path, self._legacy_path = resolve_duckdb_path(
            path,
            default_filename="legacy_landed_review_cache.duckdb",
            temporary_prefix="legacy-landed-review-cache-",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock
        self._duckdb_timeout_seconds = int(duckdb_timeout_seconds)
        if self._duckdb_timeout_seconds < 1:
            raise ValueError("duckdb_timeout_seconds must be positive")
        self._authority = LegacyLandedLeafCacheAuthority.from_private_key_path(
            operator_key_path
        )
        if self._authority.issuer_key_id != policy.issuer_key_id:
            raise ValueError("legacy leaf cache policy/key binding is invalid")
        self.trusted_public_keys = {
            self._authority.issuer_key_id: self._authority.public_key_bytes
        }
        self._initialize()

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _connect(self) -> DuckDBConnection:
        return open_duckdb_connection(
            self.path,
            timeout_seconds=self._duckdb_timeout_seconds,
        )

    def _require_policy_key(self, key: LegacyLandedLeafCacheKey) -> None:
        """Refuse lookup, leasing, signing, or import outside pinned policy."""

        if not isinstance(key, LegacyLandedLeafCacheKey):
            raise TypeError("typed legacy leaf cache key is required")
        if (
            key.policy_id != self.policy.policy_id
            or key.current_head != self.policy.current_head
            or key.current_tree_id != self.policy.current_tree_id
        ):
            raise LegacyLandedLeafCacheError(
                "legacy leaf cache key is outside the pinned policy fence"
            )
        try:
            task = self.policy.task(key.task_id)
        except (LegacyLandedReviewError, ValueError) as exc:
            raise LegacyLandedLeafCacheError(
                "legacy leaf cache task is outside the pinned policy"
            ) from exc
        if (
            key.canonical_task_key != task.canonical_task_key
            or key.canonical_task_cid != task.canonical_task_cid
        ):
            raise LegacyLandedLeafCacheError(
                "legacy leaf cache canonical task binding is invalid"
            )
        providers = tuple(
            item
            for item in (self.policy.grok, self.policy.codex)
            if item.role == key.role
        )
        if len(providers) != 1 or (
            key.provider != providers[0].provider
            or key.model != providers[0].model
        ):
            raise LegacyLandedLeafCacheError(
                "legacy leaf cache provider binding is invalid"
            )

    def _initialize(self) -> None:
        initialize_duckdb_database(
            self.path,
            legacy_sqlite_path=self._legacy_path,
            timeout_seconds=self._duckdb_timeout_seconds,
            table_names=(
                "legacy_landed_leaf_records",
                "legacy_landed_leaf_flights",
            ),
            schema_sql="""
                CREATE TABLE IF NOT EXISTS legacy_landed_leaf_records (
                    key_id TEXT PRIMARY KEY,
                    policy_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    canonical_task_cid TEXT NOT NULL,
                    manifest_id TEXT NOT NULL,
                    leaf_index BIGINT NOT NULL,
                    leaf_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    current_head TEXT NOT NULL,
                    current_tree_id TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    record_id TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL,
                    stored_at_ms BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS legacy_landed_leaf_task_idx
                    ON legacy_landed_leaf_records(
                        policy_id, task_id, manifest_id, leaf_index, role
                    );
                CREATE TABLE IF NOT EXISTS legacy_landed_leaf_flights (
                    key_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    fencing_token BIGINT NOT NULL,
                    acquired_at_ms BIGINT NOT NULL,
                    expires_at_ms BIGINT NOT NULL
                );
                """,
        )

    @staticmethod
    def key_for(
        *,
        policy: LegacyLandedReviewPolicy,
        task: LegacyTaskPolicy,
        manifest: Mapping[str, Any],
        leaf: Mapping[str, Any],
        provider: LegacyProviderPolicy,
        request: LegacyLeafReviewRequest,
    ) -> LegacyLandedLeafCacheKey:
        return LegacyLandedLeafCacheKey.from_request(
            policy=policy,
            task=task,
            manifest=manifest,
            leaf=leaf,
            provider=provider,
            request=request,
        )

    def _decode_row(
        self,
        row: DuckDBRow,
        *,
        expected_key: LegacyLandedLeafCacheKey,
    ) -> LegacyLandedLeafCacheRecord:
        self._require_policy_key(expected_key)
        try:
            key_payload = _strict_object(str(row["key_json"]))
            record_payload = _strict_object(str(row["record_json"]))
            stored_key = LegacyLandedLeafCacheKey.from_dict(key_payload)
            verification = verify_legacy_landed_leaf_cache_record(
                record_payload,
                expected_key=expected_key,
                trusted_public_keys=self.trusted_public_keys,
            )
            record = verification.record
            if not verification.verified or record is None:
                raise ValueError(
                    ",".join(verification.reason_codes)
                    or "record verification failed"
                )
            expected_columns = {
                "key_id": stored_key.key_id,
                "policy_id": stored_key.policy_id,
                "task_id": stored_key.task_id,
                "canonical_task_cid": stored_key.canonical_task_cid,
                "manifest_id": stored_key.manifest_id,
                "leaf_index": stored_key.leaf_index,
                "leaf_id": stored_key.leaf_id,
                "request_id": stored_key.request_id,
                "role": stored_key.role,
                "provider": stored_key.provider,
                "model": stored_key.model,
                "current_head": stored_key.current_head,
                "current_tree_id": stored_key.current_tree_id,
                "record_id": record.record_id,
            }
            if any(row[field] != value for field, value in expected_columns.items()):
                raise ValueError("denormalized cache columns differ")
            if canonical_json_bytes(stored_key.to_dict()) != canonical_json_bytes(
                expected_key.to_dict()
            ):
                raise ValueError("stored cache key differs")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LegacyLandedLeafCacheError(
                "legacy landed leaf cache record is poisoned"
            ) from exc
        return record

    def lookup(
        self, key: LegacyLandedLeafCacheKey
    ) -> LegacyLandedLeafCacheRecord | None:
        self._require_policy_key(key)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM legacy_landed_leaf_records WHERE key_id=?",
                (key.key_id,),
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else self._decode_row(row, expected_key=key)

    def acquire(
        self,
        key: LegacyLandedLeafCacheKey,
        *,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_LEAF_LEASE_SECONDS,
    ) -> LegacyLandedLeafCacheLease:
        self._require_policy_key(key)
        if isinstance(lease_seconds, bool) or lease_seconds < 1:
            raise ValueError("leaf cache lease_seconds must be positive")
        owner = str(
            owner_id
            or f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        ).strip()
        if not owner:
            raise ValueError("leaf cache owner_id is required")
        now = self._now_ms()
        token = uuid.uuid4().hex
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            completed = connection.execute(
                "SELECT 1 FROM legacy_landed_leaf_records WHERE key_id=?",
                (key.key_id,),
            ).fetchone()
            row = connection.execute(
                "SELECT * FROM legacy_landed_leaf_flights WHERE key_id=?",
                (key.key_id,),
            ).fetchone()
            if completed is not None:
                connection.commit()
                return LegacyLandedLeafCacheLease(
                    key.key_id, "completed", "", 0, now, now, False
                )
            if row is not None and int(row["expires_at_ms"]) > now:
                connection.commit()
                return LegacyLandedLeafCacheLease(
                    key.key_id,
                    str(row["owner_id"]),
                    str(row["token"]),
                    int(row["fencing_token"]),
                    int(row["acquired_at_ms"]),
                    int(row["expires_at_ms"]),
                    False,
                )
            fence = int(row["fencing_token"]) + 1 if row is not None else 1
            expires = now + int(lease_seconds) * 1000
            connection.execute(
                """
                INSERT INTO legacy_landed_leaf_flights(
                    key_id, owner_id, token, fencing_token,
                    acquired_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    token=excluded.token,
                    fencing_token=excluded.fencing_token,
                    acquired_at_ms=excluded.acquired_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (key.key_id, owner, token, fence, now, expires),
            )
            connection.commit()
            return LegacyLandedLeafCacheLease(
                key.key_id, owner, token, fence, now, expires, True
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew(
        self,
        lease: LegacyLandedLeafCacheLease,
        *,
        lease_seconds: int = DEFAULT_LEAF_LEASE_SECONDS,
    ) -> LegacyLandedLeafCacheLease:
        if not lease.acquired:
            raise LegacyLandedLeafCacheError("only a leaf lease owner can renew")
        now = self._now_ms()
        expires = now + int(lease_seconds) * 1000
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE legacy_landed_leaf_flights SET expires_at_ms=?
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                  AND expires_at_ms>?
                """,
                (
                    expires,
                    lease.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise LegacyLandedLeafCacheError("leaf cache lease was fenced")
        finally:
            connection.close()
        return LegacyLandedLeafCacheLease(
            lease.key_id,
            lease.owner_id,
            lease.token,
            lease.fencing_token,
            lease.acquired_at_ms,
            expires,
            True,
        )

    def release(self, lease: LegacyLandedLeafCacheLease) -> bool:
        if not lease.acquired:
            return False
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM legacy_landed_leaf_flights
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                """,
                (
                    lease.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                ),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    def put(
        self,
        key: LegacyLandedLeafCacheKey,
        receipt: Mapping[str, Any],
        *,
        lease: LegacyLandedLeafCacheLease,
    ) -> LegacyLandedLeafCacheRecord:
        """Sign and atomically insert one still-owned exact leaf result."""

        self._require_policy_key(key)
        record = self._authority.issue(
            key, receipt, issued_at_ms=max(1, self._now_ms())
        )
        now = self._now_ms()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                """
                SELECT 1 FROM legacy_landed_leaf_flights
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                  AND expires_at_ms>?
                """,
                (
                    key.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    now,
                ),
            ).fetchone()
            if not lease.acquired or owned is None:
                raise LegacyLandedLeafCacheError("leaf cache writer was fenced")
            existing = connection.execute(
                "SELECT * FROM legacy_landed_leaf_records WHERE key_id=?",
                (key.key_id,),
            ).fetchone()
            if existing is not None:
                decoded = self._decode_row(existing, expected_key=key)
                if canonical_json_bytes(decoded.to_dict()) != canonical_json_bytes(
                    record.to_dict()
                ):
                    raise LegacyLandedLeafCacheError(
                        "same-key legacy leaf cache record collision"
                    )
                connection.execute(
                    """
                    DELETE FROM legacy_landed_leaf_flights
                    WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                    """,
                    (
                        key.key_id,
                        lease.owner_id,
                        lease.token,
                        lease.fencing_token,
                    ),
                )
                connection.commit()
                return decoded
            connection.execute(
                """
                INSERT INTO legacy_landed_leaf_records(
                    key_id, policy_id, task_id, canonical_task_cid,
                    manifest_id, leaf_index, leaf_id, request_id, role,
                    provider, model, current_head, current_tree_id,
                    key_json, record_id, record_json, stored_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key.key_id,
                    key.policy_id,
                    key.task_id,
                    key.canonical_task_cid,
                    key.manifest_id,
                    key.leaf_index,
                    key.leaf_id,
                    key.request_id,
                    key.role,
                    key.provider,
                    key.model,
                    key.current_head,
                    key.current_tree_id,
                    canonical_json_bytes(key.to_dict()).decode("ascii"),
                    record.record_id,
                    canonical_json_bytes(record.to_dict()).decode("ascii"),
                    now,
                ),
            )
            deleted = connection.execute(
                """
                DELETE FROM legacy_landed_leaf_flights
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                """,
                (
                    key.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                ),
            )
            if deleted.rowcount != 1:
                raise LegacyLandedLeafCacheError(
                    "leaf cache lease changed during commit"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return record

    @staticmethod
    def rebind_cached_receipt(
        record: LegacyLandedLeafCacheRecord,
        *,
        review_run_id: str,
    ) -> dict[str, Any]:
        if not isinstance(review_run_id, str) or len(review_run_id) < 16:
            raise ValueError("fresh legacy review_run_id is required")
        body = dict(record.receipt)
        body.pop("receipt_id", None)
        body["review_run_id"] = review_run_id
        body["provider_evidence_source"] = "signed_cache"
        body["provider_invoked_in_current_run"] = False
        body["provider_evidence_cache_record"] = record.to_dict()
        return {**body, "receipt_id": content_identity(body)}

    def _heartbeat_lease(
        self,
        *,
        lease: LegacyLandedLeafCacheLease,
        stop: threading.Event,
        errors: list[BaseException],
        lease_seconds: int,
    ) -> None:
        interval = max(0.05, float(lease_seconds) / 3)
        while not stop.wait(interval):
            try:
                self.renew(lease, lease_seconds=lease_seconds)
            except BaseException as exc:
                errors.append(exc)
                return

    def review_leaf(
        self,
        *,
        task: LegacyTaskPolicy,
        manifest: Mapping[str, Any],
        leaf: Mapping[str, Any],
        provider: LegacyProviderPolicy,
        invoker: LegacyProviderInvoker,
        review_run_id: str,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_LEAF_LEASE_SECONDS,
        wait_timeout_seconds: float = DEFAULT_LEAF_WAIT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> LegacyLandedLeafCacheReview:
        """Return fresh or signed-resumed evidence without caching validation."""

        if wait_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("leaf cache wait bounds must be positive")
        request = _leaf_review_request(
            policy=self.policy,
            task=task,
            manifest=manifest,
            leaf=leaf,
            provider=provider,
        )
        key = self.key_for(
            policy=self.policy,
            task=task,
            manifest=manifest,
            leaf=leaf,
            provider=provider,
            request=request,
        )
        deadline = time.monotonic() + float(wait_timeout_seconds)
        while time.monotonic() < deadline:
            cached = self.lookup(key)
            if cached is not None:
                return LegacyLandedLeafCacheReview(
                    self.rebind_cached_receipt(
                        cached, review_run_id=review_run_id
                    ),
                    True,
                    cached.record_id,
                    0,
                )
            lease = self.acquire(
                key, owner_id=owner_id, lease_seconds=lease_seconds
            )
            if lease.acquired:
                heartbeat_stop = threading.Event()
                heartbeat_errors: list[BaseException] = []

                heartbeat_thread = threading.Thread(
                    target=self._heartbeat_lease,
                    kwargs={
                        "lease": lease,
                        "stop": heartbeat_stop,
                        "errors": heartbeat_errors,
                        "lease_seconds": lease_seconds,
                    },
                    name=f"legacy-leaf-cache-{lease.fencing_token}",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    receipt = _review_one_leaf(
                        request=request,
                        provider=provider,
                        invoker=invoker,
                        review_run_id=review_run_id,
                    )
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    if heartbeat_errors:
                        raise LegacyLandedLeafCacheError(
                            "legacy leaf cache heartbeat was fenced"
                        ) from heartbeat_errors[0]
                    record = self.put(key, receipt, lease=lease)
                    return LegacyLandedLeafCacheReview(
                        receipt,
                        False,
                        record.record_id,
                        lease.fencing_token,
                    )
                except BaseException:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    self.release(lease)
                    raise
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
            while time.monotonic() < deadline:
                cached = self.lookup(key)
                if cached is not None:
                    return LegacyLandedLeafCacheReview(
                        self.rebind_cached_receipt(
                            cached, review_run_id=review_run_id
                        ),
                        True,
                        cached.record_id,
                        lease.fencing_token,
                    )
                connection = self._connect()
                try:
                    active = connection.execute(
                        """
                        SELECT fencing_token, expires_at_ms
                        FROM legacy_landed_leaf_flights WHERE key_id=?
                        """,
                        (key.key_id,),
                    ).fetchone()
                finally:
                    connection.close()
                if (
                    active is None
                    or int(active["expires_at_ms"]) <= self._now_ms()
                    or int(active["fencing_token"]) != lease.fencing_token
                ):
                    break
                time.sleep(
                    min(
                        poll_interval_seconds,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
        raise LegacyLandedLeafCacheTimeout(
            f"timed out waiting for signed legacy leaf {key.key_id}"
        )

    def records(self, *, limit: int = MAX_SNAPSHOT_RECORDS) -> tuple[
        LegacyLandedLeafCacheRecord, ...
    ]:
        if not 1 <= int(limit) <= MAX_SNAPSHOT_RECORDS:
            raise ValueError("legacy leaf record limit is invalid")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM legacy_landed_leaf_records
                ORDER BY key_id LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        finally:
            connection.close()
        result: list[LegacyLandedLeafCacheRecord] = []
        for row in rows:
            key = LegacyLandedLeafCacheKey.from_dict(
                _strict_object(str(row["key_json"]))
            )
            result.append(self._decode_row(row, expected_key=key))
        return tuple(result)

    def export_snapshot(
        self,
        directory: str | Path,
        *,
        backend: VerifiedIPLDBackend,
        pin: bool = True,
    ) -> LegacyLandedLeafCacheSnapshot:
        """Export one immutable, non-authoritative Parquet/IPLD snapshot."""

        if type(pin) is not bool:
            raise TypeError("legacy cache snapshot pin must be a boolean")
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = root / f".legacy-leaf-{uuid.uuid4().hex}.parquet.tmp"
        connection = self._connect()
        try:
            count_row = connection.execute(
                "SELECT count(*) AS row_count FROM legacy_landed_leaf_records"
            ).fetchone()
            row_count = int(count_row["row_count"]) if count_row is not None else 0
            if row_count > MAX_SNAPSHOT_RECORDS:
                raise LegacyLandedLeafCacheError(
                    "legacy leaf cache snapshot record bound exceeded"
                )
            rows = connection.execute(
                "SELECT * FROM legacy_landed_leaf_records ORDER BY key_id"
            ).fetchall()
            records: list[LegacyLandedLeafCacheRecord] = []
            for row in rows:
                key = LegacyLandedLeafCacheKey.from_dict(
                    _strict_object(str(row["key_json"]))
                )
                records.append(self._decode_row(row, expected_key=key))
            if len(records) != row_count:
                raise LegacyLandedLeafCacheError(
                    "legacy leaf cache snapshot inventory changed"
                )
            connection.execute(
                """
                COPY (
                    SELECT key_id, key_json, record_id, record_json, stored_at_ms
                    FROM legacy_landed_leaf_records ORDER BY key_id
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                (str(temporary),),
            )
        finally:
            connection.close()
        try:
            with temporary.open("rb") as stream:
                parquet_bytes = stream.read()
                os.fsync(stream.fileno())
            if not 1 <= len(parquet_bytes) <= MAX_SNAPSHOT_PARQUET_BYTES:
                raise LegacyLandedLeafCacheError(
                    "legacy leaf cache snapshot Parquet byte bound exceeded"
                )
            put = backend.put_raw(parquet_bytes, pin=pin)
            parquet_cid = admit_cid(put.cid, codecs=("raw",))
            admitted_bytes, _raw_receipt = backend.get_raw(parquet_cid)
            if admitted_bytes != parquet_bytes:
                raise LegacyLandedLeafCacheError(
                    "legacy cache backend raw readback mismatch"
                )
            final_path = root / f"{parquet_cid}.parquet"
            try:
                # Hard-link publication is atomic and never replaces an
                # existing content-addressed path.
                os.link(temporary, final_path)
            except FileExistsError:
                if final_path.read_bytes() != parquet_bytes:
                    raise LegacyLandedLeafCacheError(
                        "existing snapshot path differs from its CID"
                    ) from None
                temporary.unlink()
            else:
                temporary.unlink()
                directory_fd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            manifest = {
                "schema": LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA,
                "policy_id": self.policy.policy_id,
                "current_head": self.policy.current_head,
                "current_tree_id": self.policy.current_tree_id,
                "row_count": len(records),
                "ordered_key_ids": [record.key.key_id for record in records],
                "ordered_record_ids": [record.record_id for record in records],
                "parquet_cid": parquet_cid,
                "parquet_codec": "raw",
                "parquet_byte_length": len(parquet_bytes),
                "replication_pin_requested": pin,
                "signed_records_only": True,
                "mutable_lock_or_lease_store": False,
                "completion_authoritative": False,
                "proof_authoritative": False,
            }
            manifest_put = backend.put_dag_json(manifest, pin=pin)
            manifest_cid = admit_cid(manifest_put.cid, codecs=("dag-json",))
            manifest_bytes, _manifest_receipt = backend.get_dag_json(
                manifest_cid
            )
            if canonical_json_bytes(_strict_object(manifest_bytes)) != (
                canonical_json_bytes(manifest)
            ):
                raise LegacyLandedLeafCacheError(
                    "legacy cache backend manifest readback mismatch"
                )
            return LegacyLandedLeafCacheSnapshot(
                manifest_cid,
                parquet_cid,
                final_path,
                len(records),
                manifest,
            )
        finally:
            if temporary.exists():
                temporary.unlink()

    def _insert_imported_records(
        self, records: tuple[LegacyLandedLeafCacheRecord, ...]
    ) -> int:
        """Admit a completely preverified snapshot in one transaction."""

        connection = self._connect()
        inserted = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                key = record.key
                self._require_policy_key(key)
                existing = connection.execute(
                    "SELECT * FROM legacy_landed_leaf_records WHERE key_id=?",
                    (key.key_id,),
                ).fetchone()
                if existing is not None:
                    decoded = self._decode_row(existing, expected_key=key)
                    if canonical_json_bytes(decoded.to_dict()) != (
                        canonical_json_bytes(record.to_dict())
                    ):
                        raise LegacyLandedLeafCacheError(
                            "same-key imported legacy leaf cache collision"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO legacy_landed_leaf_records(
                        key_id, policy_id, task_id, canonical_task_cid,
                        manifest_id, leaf_index, leaf_id, request_id, role,
                        provider, model, current_head, current_tree_id,
                        key_json, record_id, record_json, stored_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key.key_id,
                        key.policy_id,
                        key.task_id,
                        key.canonical_task_cid,
                        key.manifest_id,
                        key.leaf_index,
                        key.leaf_id,
                        key.request_id,
                        key.role,
                        key.provider,
                        key.model,
                        key.current_head,
                        key.current_tree_id,
                        canonical_json_bytes(key.to_dict()).decode("ascii"),
                        record.record_id,
                        canonical_json_bytes(record.to_dict()).decode("ascii"),
                        self._now_ms(),
                    ),
                )
                inserted += 1
            connection.commit()
            return inserted
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def import_snapshot(
        self,
        manifest_cid: str,
        *,
        backend: VerifiedIPLDBackend,
    ) -> int:
        """Rehydrate only exact signed records from a verified IPLD snapshot."""

        manifest_bytes, _manifest_receipt = backend.get_dag_json(manifest_cid)
        if len(manifest_bytes) > MAX_SNAPSHOT_MANIFEST_BYTES:
            raise LegacyLandedLeafCacheError(
                "legacy cache snapshot manifest exceeds its byte bound"
            )
        manifest = _strict_object(manifest_bytes)
        common_fields = {
            "schema",
            "policy_id",
            "current_head",
            "current_tree_id",
            "row_count",
            "ordered_key_ids",
            "ordered_record_ids",
            "parquet_cid",
            "parquet_codec",
            "parquet_byte_length",
            "signed_records_only",
            "mutable_lock_or_lease_store",
            "completion_authoritative",
            "proof_authoritative",
        }
        schema = manifest.get("schema")
        observed_fields = set(manifest)
        allowed_fields = (
            (common_fields, common_fields | {"replication_pin_requested"})
            if schema == LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA_V1
            else (common_fields | {"replication_pin_requested"},)
        )
        if observed_fields not in allowed_fields:
            raise LegacyLandedLeafCacheError("legacy cache snapshot shape is invalid")
        if (
            schema
            not in {
                LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA_V1,
                LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA,
            }
            or manifest.get("policy_id") != self.policy.policy_id
            or manifest.get("current_head") != self.policy.current_head
            or manifest.get("current_tree_id") != self.policy.current_tree_id
            or manifest.get("parquet_codec") != "raw"
            or (
                "replication_pin_requested" in manifest
                and type(manifest.get("replication_pin_requested")) is not bool
            )
            or manifest.get("signed_records_only") is not True
            or manifest.get("mutable_lock_or_lease_store") is not False
            or manifest.get("completion_authoritative") is not False
            or manifest.get("proof_authoritative") is not False
        ):
            raise LegacyLandedLeafCacheError(
                "legacy cache snapshot policy or authority binding is invalid"
            )
        row_count = manifest.get("row_count")
        key_ids = manifest.get("ordered_key_ids")
        record_ids = manifest.get("ordered_record_ids")
        parquet_byte_length = manifest.get("parquet_byte_length")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or not 0 <= row_count <= MAX_SNAPSHOT_RECORDS
            or not isinstance(key_ids, list)
            or not isinstance(record_ids, list)
            or len(key_ids) != row_count
            or len(record_ids) != row_count
            or any(
                not isinstance(item, str) or not 1 <= len(item) <= 256
                for item in (*key_ids, *record_ids)
            )
            or len(set(key_ids)) != row_count
            or len(set(record_ids)) != row_count
            or isinstance(parquet_byte_length, bool)
            or not isinstance(parquet_byte_length, int)
            or not 1 <= parquet_byte_length <= MAX_SNAPSHOT_PARQUET_BYTES
        ):
            raise LegacyLandedLeafCacheError(
                "legacy cache snapshot inventory bounds are invalid"
            )
        parquet_cid = admit_cid(manifest.get("parquet_cid"), codecs=("raw",))
        parquet_bytes, _parquet_receipt = backend.get_raw(parquet_cid)
        if len(parquet_bytes) != parquet_byte_length:
            raise LegacyLandedLeafCacheError(
                "legacy cache snapshot Parquet length mismatch"
            )
        temporary = self.path.parent / f".legacy-import-{uuid.uuid4().hex}.parquet"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(parquet_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT key_id, key_json, record_id, record_json
                    FROM read_parquet(?) ORDER BY key_id
                    """,
                    (str(temporary),),
                ).fetchall()
            finally:
                connection.close()
        finally:
            if temporary.exists():
                temporary.unlink()
        if (
            row_count != len(rows)
            or [str(row["key_id"]) for row in rows] != key_ids
            or [str(row["record_id"]) for row in rows] != record_ids
        ):
            raise LegacyLandedLeafCacheError(
                "legacy cache snapshot ordering or row count mismatch"
            )
        records: list[LegacyLandedLeafCacheRecord] = []
        for row in rows:
            if (
                len(str(row["key_json"]).encode("utf-8"))
                > MAX_SNAPSHOT_ROW_JSON_BYTES
                or len(str(row["record_json"]).encode("utf-8"))
                > MAX_SNAPSHOT_ROW_JSON_BYTES
            ):
                raise LegacyLandedLeafCacheError(
                    "legacy cache snapshot row exceeds its JSON bound"
                )
            key = LegacyLandedLeafCacheKey.from_dict(
                _strict_object(str(row["key_json"]))
            )
            self._require_policy_key(key)
            record = LegacyLandedLeafCacheRecord.from_dict(
                _strict_object(str(row["record_json"]))
            )
            if key.key_id != str(row["key_id"]) or record.record_id != str(
                row["record_id"]
            ):
                raise LegacyLandedLeafCacheError(
                    "legacy cache snapshot row identity mismatch"
                )
            verification = verify_legacy_landed_leaf_cache_record(
                record,
                expected_key=key,
                trusted_public_keys=self.trusted_public_keys,
            )
            if not verification.verified:
                raise LegacyLandedLeafCacheError(
                    "imported legacy leaf cache signature is invalid"
                )
            records.append(record)
        return self._insert_imported_records(tuple(records))


__all__ = [
    "DEFAULT_LEAF_LEASE_SECONDS",
    "DEFAULT_LEAF_WAIT_SECONDS",
    "LEGACY_LANDED_LEAF_CACHE_KEY_SCHEMA",
    "LEGACY_LANDED_LEAF_CACHE_RECORD_INTERFACE",
    "LEGACY_LANDED_LEAF_CACHE_RECORD_SCHEMA",
    "LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA",
    "LEGACY_LANDED_LEAF_CACHE_SNAPSHOT_SCHEMA_V1",
    "LegacyLandedLeafCacheAuthority",
    "LegacyLandedLeafCacheError",
    "LegacyLandedLeafCacheKey",
    "LegacyLandedLeafCacheLease",
    "LegacyLandedLeafCacheRecord",
    "LegacyLandedLeafCacheReview",
    "LegacyLandedLeafCacheSnapshot",
    "LegacyLandedLeafCacheTimeout",
    "LegacyLandedLeafCacheVerification",
    "LegacyLandedLeafResultCache",
    "verify_legacy_landed_leaf_cache_record",
]
