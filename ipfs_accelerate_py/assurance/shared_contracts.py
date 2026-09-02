"""Fail-closed PCPR-040 shared-contract catalog.

One normative source for the fourteen PCPR shared contracts named by the
plan. Each contract has one identity, an exact schema version,
canonicalization, unknown-field rejection, int64 and Unicode rules, and
CID rules. Datasets and Kit bind these identities; they do not remint
them.

This module is not a freeze (PCPR-002), not canonical-byte or CID vectors
(PCPR-041), not negative or cross-language vectors (PCPR-042), and not
cross-repository compatibility checks (PCPR-043). It is not release
authority: it does not write DuckDB or Quack state and never emits a
closed PCPR release outcome. Live claims require live evidence.
Simulated results are not live.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

INTERFACE: Final = "SharedContractCatalog@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/shared-contracts-catalog@1"
CONTRACT_INTERFACE_SUFFIX: Final = "@1"
PCPR_040_TASK_ID: Final = "PCPR-040"
PCPR_040_GOAL_ID: Final = "PCPR-G500"
PCPR_041_TASK_ID: Final = "PCPR-041"
PCPR_042_TASK_ID: Final = "PCPR-042"
PCPR_043_TASK_ID: Final = "PCPR-043"
PCPR_002_TASK_ID: Final = "PCPR-002"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/shared-contracts-catalog@1"

INT64_MIN: Final = -(2**63)
INT64_MAX: Final = (2**63) - 1
MAX_STRING_BYTES: Final = 4_096
MAX_ARRAY_ITEMS: Final = 256
MAX_RECORD_BYTES: Final = 1_048_576
MAX_OBJECT_KEYS: Final = 64

CID_VERSION: Final = 1
CID_BASE: Final = "base32"
CID_MULTIHASH: Final = "sha2-256"
CID_CODEC: Final = "dag-json"
CID_PATTERN: Final = r"^b[a-z2-7]+$"
_CID_RE: Final = re.compile(CID_PATTERN)
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_HEX64_RE: Final = re.compile(r"^[0-9a-fA-F]{64}$")
_QM_LIKE_RE: Final = re.compile(r"^[Qm][1-9A-HJ-NP-Za-km-z]{0,50}$")

UNKNOWN_FIELD_POLICY: Final = "reject"
UNICODE_NORMALIZATION: Final = "NFC"
KEY_ORDERING: Final = "utf8_code_unit_lexicographic"
FLOAT_POLICY: Final = "rejected"
CANONICAL_CODEC: Final = "dag-json"

AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "ipfs_accelerate_py",
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "portfolio",
    }
)

REQUIRED_CONTRACT_NAMES: Final[tuple[str, ...]] = (
    "SupervisorObjectiveIntent",
    "ObjectiveMaterializationReceipt",
    "SupervisorContextPack",
    "SemanticArtifactIdentity",
    "DurableArtifactReceipt",
    "ProofObligation",
    "ProofResult",
    "ProofAdmissionDecision",
    "ExecutionInvocation",
    "ExecutionReceipt",
    "SupervisorEvent",
    "TaskStateTransition",
    "ReleaseComponentManifest",
    "PortfolioCompatibilityManifest",
)

OWNER_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SupervisorContextPack": "ipfs_datasets_py/datasets-context-pack@1",
        "ProofObligation": (
            "ipfs_accelerate_py/agent-supervisor/code-proof-obligation@1"
        ),
        "ProofResult": "ipfs_accelerate_py/agent-supervisor/proof-receipt@1",
        "ExecutionInvocation": (
            "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-request@1"
        ),
        "ExecutionReceipt": (
            "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-result@1"
        ),
    }
)

OWNER_INTERFACES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SupervisorContextPack": "DatasetsContextPack@1",
        "ProofObligation": "CodeProofObligation@1",
        "ProofResult": "ProofReceipt@1",
        "ExecutionInvocation": "InvocationRequest@1",
        "ExecutionReceipt": "InvocationResult@1",
    }
)

CONTRACT_AUTHORITIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SupervisorObjectiveIntent": "ipfs_accelerate_py",
        "ObjectiveMaterializationReceipt": "ipfs_accelerate_py",
        "SupervisorContextPack": "ipfs_datasets_py",
        "SemanticArtifactIdentity": "ipfs_datasets_py",
        "DurableArtifactReceipt": "ipfs_kit_py",
        "ProofObligation": "ipfs_accelerate_py",
        "ProofResult": "ipfs_accelerate_py",
        "ProofAdmissionDecision": "ipfs_accelerate_py",
        "ExecutionInvocation": "ipfs_accelerate_py",
        "ExecutionReceipt": "ipfs_accelerate_py",
        "SupervisorEvent": "ipfs_accelerate_py",
        "TaskStateTransition": "ipfs_accelerate_py",
        "ReleaseComponentManifest": "portfolio",
        "PortfolioCompatibilityManifest": "portfolio",
    }
)


def shared_schema_id(name: str) -> str:
    kebab = _kebab(name)
    return f"pcpr/shared-contracts/{kebab}@1"


def shared_interface_id(name: str) -> str:
    return f"{name}{CONTRACT_INTERFACE_SUFFIX}"


def _kebab(name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            chars.append("-")
        chars.append(char.lower())
    return "".join(chars)


@dataclass(frozen=True)
class SharedContractSpec:
    """One catalog entry. Identity is the schema string."""

    name: str
    schema: str
    interface: str
    authority: str
    owner_schema: str
    owner_interface: str
    disposition: str
    unknown_field_policy: str
    freeze_task: str
    vector_task: str
    negative_vector_task: str
    compatibility_task: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "schema": self.schema,
            "interface": self.interface,
            "authority": self.authority,
            "owner_schema": self.owner_schema,
            "owner_interface": self.owner_interface,
            "disposition": self.disposition,
            "unknown_field_policy": self.unknown_field_policy,
            "freeze_task": self.freeze_task,
            "vector_task": self.vector_task,
            "negative_vector_task": self.negative_vector_task,
            "compatibility_task": self.compatibility_task,
            "normative_task": PCPR_040_TASK_ID,
        }


def _spec(name: str) -> SharedContractSpec:
    schema = shared_schema_id(name)
    return SharedContractSpec(
        name=name,
        schema=schema,
        interface=shared_interface_id(name),
        authority=CONTRACT_AUTHORITIES[name],
        owner_schema=OWNER_SCHEMAS.get(name, schema),
        owner_interface=OWNER_INTERFACES.get(name, shared_interface_id(name)),
        disposition="normative_stabilized_not_frozen",
        unknown_field_policy=UNKNOWN_FIELD_POLICY,
        freeze_task=PCPR_002_TASK_ID,
        vector_task=PCPR_041_TASK_ID,
        negative_vector_task=PCPR_042_TASK_ID,
        compatibility_task=PCPR_043_TASK_ID,
    )


CANONICAL_CONTRACT_CATALOG: Final[tuple[SharedContractSpec, ...]] = tuple(
    _spec(name) for name in REQUIRED_CONTRACT_NAMES
)
CONTRACT_BY_NAME: Final[Mapping[str, SharedContractSpec]] = MappingProxyType(
    {item.name: item for item in CANONICAL_CONTRACT_CATALOG}
)
CONTRACT_SCHEMA_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {item.name: item.schema for item in CANONICAL_CONTRACT_CATALOG}
)

CANONICALIZATION_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "codec": CANONICAL_CODEC,
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": True,
        "allow_nan": False,
        "floats": FLOAT_POLICY,
        "unicode_encoding": "utf-8",
        "unicode_normalization": UNICODE_NORMALIZATION,
        "key_ordering": KEY_ORDERING,
        "unknown_fields": UNKNOWN_FIELD_POLICY,
        "integer_domain": "int64",
        "integer_min": INT64_MIN,
        "integer_max": INT64_MAX,
        "max_string_bytes": MAX_STRING_BYTES,
        "max_array_items": MAX_ARRAY_ITEMS,
        "max_record_bytes": MAX_RECORD_BYTES,
        "cid": MappingProxyType(
            {
                "version": CID_VERSION,
                "base": CID_BASE,
                "multihash": CID_MULTIHASH,
                "codec": CID_CODEC,
                "pattern": CID_PATTERN,
                "reject": (
                    "raw_hex",
                    "cidv0_qm",
                    "truncated",
                    "uppercase",
                    "unpadded_mismatch",
                ),
            }
        ),
    }
)


class SharedContractError(ValueError):
    """Malformed shared-contract evidence or a forbidden authority claim."""


class SharedContractAdmissionError(SharedContractError):
    """A payload, field, or identity was rejected fail-closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Deterministic DAG-JSON bytes after NFC. Floats and NaN are rejected."""

    normalized = _canonicalize_value(value, path="$")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise SharedContractAdmissionError(
            "canonical record exceeds max_record_bytes"
        )
    return encoded


def content_identity(value: Any) -> str:
    """CIDv1 dag-json / sha2-256 identity (baguqeera…)."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def catalog_mapping() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_040_TASK_ID,
        "goal_id": PCPR_040_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "frozen": False,
        "freeze_task": PCPR_002_TASK_ID,
        "vector_task": PCPR_041_TASK_ID,
        "negative_vector_task": PCPR_042_TASK_ID,
        "compatibility_task": PCPR_043_TASK_ID,
        "unknown_field_policy": UNKNOWN_FIELD_POLICY,
        "canonicalization": dict(CANONICALIZATION_POLICY),
        "contracts": [item.to_mapping() for item in CANONICAL_CONTRACT_CATALOG],
    }


def catalog_cid() -> str:
    return content_identity(catalog_mapping())


def refuse_remint(name: str, schema: str) -> str:
    """Reject a second identity for a catalog contract."""

    spec = CONTRACT_BY_NAME.get(name)
    if spec is None:
        raise SharedContractAdmissionError(
            f"{name} is not a PCPR shared contract"
        )
    text = _nfc_text(schema, "schema")
    if text != spec.schema:
        raise SharedContractAdmissionError(
            f"{name} identity {text} remints {spec.schema}"
        )
    return text


def _nfc_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SharedContractAdmissionError(f"{name} must be a non-empty string")
    normalized = unicodedata.normalize(UNICODE_NORMALIZATION, value)
    encoded = normalized.encode("utf-8")
    if len(encoded) > MAX_STRING_BYTES:
        raise SharedContractAdmissionError(f"{name} exceeds max_string_bytes")
    if any(ord(char) > 0x10FFFF for char in normalized):
        raise SharedContractAdmissionError(f"{name} is not Unicode scalar")
    return normalized


def _int64(value: Any, name: str) -> int:
    if type(value) is bool or type(value) is not int:
        raise SharedContractAdmissionError(f"{name} must be an int64 integer")
    if value < INT64_MIN or value > INT64_MAX:
        raise SharedContractAdmissionError(f"{name} is outside the int64 domain")
    return value


def _canonicalize_value(value: Any, *, path: str) -> Any:
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return _int64(value, path)
    if type(value) is float:
        raise SharedContractAdmissionError(
            f"{path} floats are rejected by the shared-contract catalog"
        )
    if isinstance(value, str):
        return _nfc_text(value, path)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ARRAY_ITEMS:
            raise SharedContractAdmissionError(
                f"{path} exceeds max_array_items"
            )
        return [
            _canonicalize_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if len(value) > MAX_OBJECT_KEYS:
            raise SharedContractAdmissionError(
                f"{path} exceeds max_object_keys"
            )
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise SharedContractAdmissionError(
                    f"{path} map keys must be non-empty strings"
                )
            nfc_key = unicodedata.normalize(UNICODE_NORMALIZATION, key)
            if nfc_key in out:
                raise SharedContractAdmissionError(
                    f"{path} Unicode-equivalent keys collide after NFC"
                )
            out[nfc_key] = _canonicalize_value(item, path=f"{path}.{nfc_key}")
        return out
    raise SharedContractAdmissionError(
        f"{path} is not a DAG-JSON value: {type(value).__name__}"
    )


def _cid(value: Any, name: str) -> str:
    text = _nfc_text(value, name)
    if _HEX64_RE.fullmatch(text):
        raise SharedContractAdmissionError(
            f"{name} raw SHA-256 hex is not a CID"
        )
    if _QM_LIKE_RE.fullmatch(text):
        raise SharedContractAdmissionError(f"{name} CIDv0 Qm form is rejected")
    if text != text.lower():
        raise SharedContractAdmissionError(f"{name} CID must be lowercase")
    if _CID_RE.fullmatch(text) is None:
        raise SharedContractAdmissionError(
            f"{name} must be CIDv1 lowercase base32"
        )
    return text


def _git_oid(value: Any, name: str) -> str:
    text = _nfc_text(value, name)
    if _GIT_OID_RE.fullmatch(text) is None:
        raise SharedContractAdmissionError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _token(value: Any, name: str) -> str:
    text = _nfc_text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SharedContractAdmissionError(f"{name} is not an admitted token")
    return text


_CONTRACT_FIELDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "SupervisorObjectiveIntent": (
            "schema",
            "interface",
            "objective_id",
            "language",
            "idea_digest",
        ),
        "ObjectiveMaterializationReceipt": (
            "schema",
            "interface",
            "objective_id",
            "plan_id",
            "admitted",
            "current_tree",
        ),
        "SupervisorContextPack": (
            "schema",
            "interface",
            "task_id",
            "repository_state_cid",
            "scanned_tree_oid",
            "target_source_cid",
            "surrounding_source_cid",
            "test_source_cid",
        ),
        "SemanticArtifactIdentity": (
            "schema",
            "interface",
            "artifact_id",
            "ir_identity",
            "lineage_cid",
        ),
        "DurableArtifactReceipt": (
            "schema",
            "interface",
            "cid",
            "byte_length",
            "digest_hex",
            "durable",
        ),
        "ProofObligation": (
            "schema",
            "interface",
            "obligation_id",
            "statement",
            "logic_family",
        ),
        "ProofResult": (
            "schema",
            "interface",
            "obligation_id",
            "outcome",
            "evidence_kind",
        ),
        "ProofAdmissionDecision": (
            "schema",
            "interface",
            "obligation_id",
            "admitted",
            "reason",
        ),
        "ExecutionInvocation": (
            "schema",
            "interface",
            "invocation_id",
            "target",
            "tree_id",
        ),
        "ExecutionReceipt": (
            "schema",
            "interface",
            "invocation_id",
            "outcome",
            "evidence_kind",
        ),
        "SupervisorEvent": (
            "schema",
            "interface",
            "event_id",
            "event_type",
            "sequence",
        ),
        "TaskStateTransition": (
            "schema",
            "interface",
            "task_id",
            "from_state",
            "to_state",
            "fence",
        ),
        "ReleaseComponentManifest": (
            "schema",
            "interface",
            "component_id",
            "repository",
            "version",
        ),
        "PortfolioCompatibilityManifest": (
            "schema",
            "interface",
            "portfolio_id",
            "contract_catalog_cid",
            "component_ids",
        ),
    }
)

_CID_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_state_cid",
        "target_source_cid",
        "surrounding_source_cid",
        "test_source_cid",
        "lineage_cid",
        "cid",
        "contract_catalog_cid",
        "idea_digest",
    }
)
_OID_FIELDS: Final[frozenset[str]] = frozenset(
    {"scanned_tree_oid", "tree_id", "current_tree"}
)
_INT_FIELDS: Final[frozenset[str]] = frozenset(
    {"byte_length", "sequence", "fence"}
)
_BOOL_FIELDS: Final[frozenset[str]] = frozenset({"admitted", "durable"})
_LIST_FIELDS: Final[frozenset[str]] = frozenset({"component_ids"})


def contract_json_schema(name: str) -> dict[str, Any]:
    """Compact JSON Schema for one catalog contract."""

    spec = CONTRACT_BY_NAME.get(name)
    if spec is None:
        raise SharedContractError(f"{name} is not a PCPR shared contract")
    fields = _CONTRACT_FIELDS[name]
    properties: dict[str, Any] = {}
    for field in fields:
        if field == "schema":
            properties[field] = {"const": spec.schema, "type": "string"}
        elif field == "interface":
            properties[field] = {"const": spec.interface, "type": "string"}
        elif field in _CID_FIELDS:
            properties[field] = {"$ref": "#/$defs/cid"}
        elif field in _OID_FIELDS:
            properties[field] = {"$ref": "#/$defs/git_oid"}
        elif field in _INT_FIELDS:
            properties[field] = {"$ref": "#/$defs/int64"}
        elif field in _BOOL_FIELDS:
            properties[field] = {"type": "boolean"}
        elif field in _LIST_FIELDS:
            properties[field] = {
                "type": "array",
                "maxItems": MAX_ARRAY_ITEMS,
                "items": {"type": "string", "minLength": 1},
            }
        else:
            properties[field] = {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_STRING_BYTES,
            }
    return {
        "$id": spec.schema,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "title": spec.interface,
        "type": "object",
        "required": list(fields),
        "properties": properties,
        "x-interface": spec.interface,
        "x-schema-version": spec.schema,
        "x-authority": spec.authority,
        "x-owner-schema": spec.owner_schema,
        "x-unknown-field-policy": UNKNOWN_FIELD_POLICY,
        "x-canonicalization": dict(CANONICALIZATION_POLICY),
        "x-frozen": False,
        "x-normative-task": PCPR_040_TASK_ID,
        "x-vector-task": PCPR_041_TASK_ID,
        "$defs": {
            "cid": {
                "type": "string",
                "minLength": 8,
                "maxLength": 128,
                "pattern": CID_PATTERN,
            },
            "git_oid": {
                "type": "string",
                "minLength": 40,
                "maxLength": 40,
                "pattern": "^[0-9a-f]{40}$",
            },
            "int64": {
                "type": "integer",
                "minimum": INT64_MIN,
                "maximum": INT64_MAX,
            },
        },
    }


def admit_shared_contract(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Admit one identity envelope. Unknown fields fail closed."""

    spec = CONTRACT_BY_NAME.get(name)
    if spec is None:
        raise SharedContractAdmissionError(
            f"{name} is not a PCPR shared contract"
        )
    if not isinstance(payload, Mapping):
        raise SharedContractAdmissionError(f"{name} payload must be a mapping")
    fields = _CONTRACT_FIELDS[name]
    extra = set(payload) - set(fields)
    if extra:
        raise SharedContractAdmissionError(
            f"{name} unknown fields are rejected: {sorted(extra)}"
        )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise SharedContractAdmissionError(
            f"{name} missing required fields: {missing}"
        )
    admitted: dict[str, Any] = {}
    for field in fields:
        value = payload[field]
        if field == "schema":
            admitted[field] = refuse_remint(name, value)
        elif field == "interface":
            text = _nfc_text(value, field)
            if text != spec.interface:
                raise SharedContractAdmissionError(
                    f"{name} interface {text} remints {spec.interface}"
                )
            admitted[field] = text
        elif field in _CID_FIELDS:
            admitted[field] = _cid(value, field)
        elif field in _OID_FIELDS:
            admitted[field] = _git_oid(value, field)
        elif field in _INT_FIELDS:
            admitted[field] = _int64(value, field)
        elif field in _BOOL_FIELDS:
            if type(value) is not bool:
                raise SharedContractAdmissionError(
                    f"{field} must be a JSON boolean"
                )
            admitted[field] = value
        elif field in _LIST_FIELDS:
            if not isinstance(value, list):
                raise SharedContractAdmissionError(f"{field} must be an array")
            if len(value) > MAX_ARRAY_ITEMS:
                raise SharedContractAdmissionError(
                    f"{field} exceeds max_array_items"
                )
            admitted[field] = [_token(item, f"{field}[]") for item in value]
        else:
            admitted[field] = _token(value, field) if field.endswith("_id") or field in {
                "language",
                "logic_family",
                "outcome",
                "evidence_kind",
                "reason",
                "target",
                "event_type",
                "from_state",
                "to_state",
                "repository",
                "version",
                "ir_identity",
            } else _nfc_text(value, field)
    _ = canonical_json_bytes(admitted)
    return admitted


def identity_fixture(name: str) -> dict[str, Any]:
    """Compact identity envelope used to exercise admission, not a vector set."""

    spec = CONTRACT_BY_NAME[name]
    digest = content_identity({"fixture": name, "task_id": PCPR_040_TASK_ID})
    tree = "0" * 40
    base: dict[str, Any] = {
        "schema": spec.schema,
        "interface": spec.interface,
    }
    extras: dict[str, Any] = {
        "SupervisorObjectiveIntent": {
            "objective_id": "PCPR-G500",
            "language": "Python",
            "idea_digest": digest,
        },
        "ObjectiveMaterializationReceipt": {
            "objective_id": "PCPR-G500",
            "plan_id": "plan:pcpr-040",
            "admitted": True,
            "current_tree": tree,
        },
        "SupervisorContextPack": {
            "task_id": PCPR_040_TASK_ID,
            "repository_state_cid": digest,
            "scanned_tree_oid": tree,
            "target_source_cid": digest,
            "surrounding_source_cid": digest,
            "test_source_cid": digest,
        },
        "SemanticArtifactIdentity": {
            "artifact_id": "semantic:pcpr-040",
            "ir_identity": "ir-canonical-identity-v1",
            "lineage_cid": digest,
        },
        "DurableArtifactReceipt": {
            "cid": digest,
            "byte_length": 32,
            "digest_hex": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            "durable": True,
        },
        "ProofObligation": {
            "obligation_id": "obligation:pcpr-040",
            "statement": "shared-contract-identity-is-unique",
            "logic_family": "propositional",
        },
        "ProofResult": {
            "obligation_id": "obligation:pcpr-040",
            "outcome": "unavailable",
            "evidence_kind": "unavailable",
        },
        "ProofAdmissionDecision": {
            "obligation_id": "obligation:pcpr-040",
            "admitted": False,
            "reason": "live-proof-unavailable",
        },
        "ExecutionInvocation": {
            "invocation_id": "invocation:pcpr-040",
            "target": "shared-contract-catalog",
            "tree_id": tree,
        },
        "ExecutionReceipt": {
            "invocation_id": "invocation:pcpr-040",
            "outcome": "observed",
            "evidence_kind": "measured",
        },
        "SupervisorEvent": {
            "event_id": "event:pcpr-040",
            "event_type": "contract-stabilized",
            "sequence": 1,
        },
        "TaskStateTransition": {
            "task_id": PCPR_040_TASK_ID,
            "from_state": "ready",
            "to_state": "implemented",
            "fence": 1,
        },
        "ReleaseComponentManifest": {
            "component_id": "component:ipfs_accelerate_py",
            "repository": "ipfs_accelerate_py",
            "version": "0.0.0-rnd",
        },
        "PortfolioCompatibilityManifest": {
            "portfolio_id": "portfolio:pcpr-v1",
            "contract_catalog_cid": digest,
            "component_ids": [
                "component:ipfs_accelerate_py",
                "component:ipfs_datasets_py",
                "component:ipfs_kit_py",
            ],
        },
    }
    payload = dict(base)
    payload.update(extras[name])
    return admit_shared_contract(name, payload)


def owner_schema_is_binding(name: str) -> bool:
    spec = CONTRACT_BY_NAME[name]
    return spec.owner_schema != spec.schema or name in OWNER_SCHEMAS


__all__ = (
    "CANONICALIZATION_POLICY",
    "CANONICAL_CONTRACT_CATALOG",
    "CONTRACT_BY_NAME",
    "CONTRACT_SCHEMA_IDS",
    "INTERFACE",
    "OWNER_SCHEMAS",
    "PCPR_040_TASK_ID",
    "REQUIRED_CONTRACT_NAMES",
    "SCHEMA",
    "SharedContractAdmissionError",
    "SharedContractError",
    "admit_shared_contract",
    "canonical_json_bytes",
    "catalog_cid",
    "catalog_mapping",
    "content_identity",
    "contract_json_schema",
    "identity_fixture",
    "refuse_remint",
    "shared_interface_id",
    "shared_schema_id",
)
