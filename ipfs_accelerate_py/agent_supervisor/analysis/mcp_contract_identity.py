"""DCR-020 canonical, relocation-stable MCP contract identities.

The codec follows the local MCP++ Profile-H / proof-contract convention:
canonical DAG-JSON bytes and a CIDv1 ``dag-json`` / ``sha2-256`` multihash.
Claimed CIDs are evidence only; the canonical CID is always re-derived.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity

MCP_CONTRACT_IDENTITY_INTERFACE: Final = "McpContractIdentity@1"
MCP_CONTRACT_IDENTITY_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/mcp-contract-identity@1"
MCP_CONTRACT_SEMANTIC_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-semantic-key@1"
)
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "package",
    "operation",
    "direction",
    "schema_root",
    "profile",
    "transport",
    "runtime_instance",
)


class McpContractIdentityError(ValueError):
    """An identity input is incomplete, ambiguous, or not canonical JSON."""


class ClaimedMcpContractCidMismatch(McpContractIdentityError):
    """A claimed CID is malformed or differs from the recomputed CID."""


class McpContractDirection(Enum):
    REQUEST = "request"
    RESULT = "result"
    ERROR = "error"


@dataclass(frozen=True)
class McpContractIdentity:
    """Recomputed identity plus non-authorizing alias diagnostics."""

    semantic_key: Mapping[str, Any]
    semantic_cid: str
    declaration_cid: str
    declaration_digest: str
    alias_bindings: tuple[Mapping[str, Any], ...]
    alias_issues: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_IDENTITY_SCHEMA,
            "interface": MCP_CONTRACT_IDENTITY_INTERFACE,
            "semantic_key": dict(self.semantic_key),
            "semantic_cid": self.semantic_cid,
            "declaration_cid": self.declaration_cid,
            "declaration_digest": self.declaration_digest,
            "alias_bindings": [dict(item) for item in self.alias_bindings],
            "alias_issues": [dict(item) for item in self.alias_issues],
            "claimed_cid_trusted": False,
        }


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise McpContractIdentityError(f"{field} must be non-empty text")
    return value.strip()


def _authority_roots(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise McpContractIdentityError("authority_roots must be a non-empty mapping")
    normalized = {
        str(key).strip(): _required_text(item, "authority root") for key, item in value.items()
    }
    if "" in normalized:
        raise McpContractIdentityError("authority root names must be non-empty")
    return dict(sorted(normalized.items()))


def _aliases(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise McpContractIdentityError("aliases must be a sequence of non-empty text")
    return tuple(_required_text(item, "alias") for item in value)


def _semantic_key(declaration: Mapping[str, Any]) -> dict[str, Any]:
    key = {field: _required_text(declaration.get(field), field) for field in _REQUIRED_FIELDS}
    try:
        key["direction"] = McpContractDirection(key["direction"]).value
    except ValueError as exc:
        raise McpContractIdentityError("direction is not a closed MCP contract direction") from exc
    key["authority_roots"] = _authority_roots(declaration.get("authority_roots"))
    return {"schema": MCP_CONTRACT_SEMANTIC_KEY_SCHEMA, **key}


def _declaration_payload(
    declaration: Mapping[str, Any], semantic_key: Mapping[str, Any], source_bytes: bytes | None
) -> dict[str, Any]:
    schema = declaration.get("schema")
    if not isinstance(schema, Mapping):
        raise McpContractIdentityError("schema must be an object")
    payload = {
        "schema": MCP_CONTRACT_IDENTITY_SCHEMA,
        "semantic_key": dict(semantic_key),
        "contract_schema": dict(schema),
    }
    if source_bytes is not None:
        if not isinstance(source_bytes, bytes):
            raise McpContractIdentityError("source_bytes must be bytes when provided")
        payload["source_digest"] = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    return payload


def canonical_mcp_contract_identity(
    declaration: Mapping[str, Any], *, source_bytes: bytes | None = None, claimed_cid: str = ""
) -> McpContractIdentity:
    """Derive one semantic key and CID pair without trusting a claimed CID.

    ``source_bytes`` intentionally affects only the declaration CID/digest, not
    the semantic key: equivalent declarations at relocated paths converge,
    while changed source bytes remain separately attestable.
    """

    if not isinstance(declaration, Mapping):
        raise McpContractIdentityError("declaration must be a mapping")
    semantic_key = _semantic_key(declaration)
    payload = _declaration_payload(declaration, semantic_key, source_bytes)
    semantic_cid = content_identity(semantic_key)
    declaration_cid = content_identity(payload)
    claims = [claimed_cid]
    claims.extend(declaration.get(field, "") for field in ("claimed_cid", "cid", "contract_cid"))
    for claim in claims:
        if claim and (not isinstance(claim, str) or claim != declaration_cid):
            raise ClaimedMcpContractCidMismatch(
                "claimed MCP contract CID does not match canonical declaration CID"
            )
    aliases = _aliases(declaration.get("aliases"))
    seen: set[str] = set()
    bindings: list[Mapping[str, Any]] = []
    issues: list[Mapping[str, str]] = []
    for occurrence, alias in enumerate(aliases):
        bindings.append(
            {
                "alias": alias,
                "occurrence": occurrence,
                "alias_cid": content_identity(
                    {"semantic_cid": semantic_cid, "alias": alias, "occurrence": occurrence}
                ),
            }
        )
        if alias in seen:
            issues.append(
                {"kind": "duplicate_alias", "alias": alias, "occurrence": str(occurrence)}
            )
        seen.add(alias)
    canonical_bytes = canonical_json_bytes(payload)
    return McpContractIdentity(
        semantic_key=semantic_key,
        semantic_cid=semantic_cid,
        declaration_cid=declaration_cid,
        declaration_digest="sha256:" + hashlib.sha256(canonical_bytes).hexdigest(),
        alias_bindings=tuple(bindings),
        alias_issues=tuple(issues),
    )


__all__ = [
    "ClaimedMcpContractCidMismatch",
    "MCP_CONTRACT_IDENTITY_INTERFACE",
    "MCP_CONTRACT_IDENTITY_SCHEMA",
    "MCP_CONTRACT_SEMANTIC_KEY_SCHEMA",
    "McpContractDirection",
    "McpContractIdentity",
    "McpContractIdentityError",
    "canonical_mcp_contract_identity",
]
