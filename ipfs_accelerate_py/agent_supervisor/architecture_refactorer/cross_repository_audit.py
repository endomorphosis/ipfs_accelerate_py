"""Read-only cross-repository published-contract audit (PCAR-025).

Inspects the three pinned sibling published surfaces:

* ``ipfs_datasets_py`` semantic and content identities
* ``ipfs_kit_py`` storage and proof-seal authority
* MCP++ (``ipfs_accelerate_py/mcplusplus``) wire and profile schemas

and classifies local compatibility with the closed disposition vocabulary.
Sibling repositories remain read-only. Shared-contract changes are emitted as
proposal packets. Write, symlink, and submodule-escape attempts fail before
mutating I/O.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ArchitectureContractError,
    _closed_enum,
    _require_int,
    _require_text,
)

CROSS_REPOSITORY_AUDIT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.architecture-refactorer."
    "cross-repository-contract-audit@1"
)
CROSS_REPOSITORY_AUDIT_VERSION = 1
CROSS_REPOSITORY_AUDIT_EVIDENCE = "pcar/cross-repository-audit@1"
PROPOSAL_PACKET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/cross-repository-proposal-packet@1"
)
EXTRACTOR_IDENTITY = "pcar-025-cross-repository-audit"
TASK_ID = "PCAR-025"
WRITE_POLICY = "deny"
EFFECT_CLASS = "cross_repository_read_only"
BOOTSTRAP_CONTRACTS_PATH = (
    "docs/architecture/architecture_refactorer_inventory/"
    "cross_repository_contracts.bootstrap.json"
)
INVENTORY_RELATIVE_PATH = (
    "docs/architecture/architecture_refactorer_inventory/"
    "cross_repository_contract_audit.json"
)
SEALED_BASELINE_PATH = (
    "docs/architecture/architecture_refactorer_inventory/"
    "sealed_current_tree_baseline.json"
)

_UNKNOWN_FIELD_MESSAGE = "unknown cross-repository audit field"
_MISSING_FIELD_MESSAGE = "missing cross-repository audit field"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class CrossRepositoryAuditError(ArchitectureContractError):
    """Fail-closed cross-repository audit contract violation."""


class CrossRepositoryWriteError(CrossRepositoryAuditError):
    """Raised before mutating I/O when a sibling write is requested."""


class CrossRepositoryEscapeError(CrossRepositoryAuditError):
    """Raised before I/O when a path escapes the declared read-only scope."""


class ContractCompatibilityDisposition(str, Enum):
    """Closed local-compatibility vocabulary (PCAR-PLAN-R1 / bootstrap)."""

    COMPATIBLE = "compatible"
    ADAPTER_REQUIRED = "adapter_required"
    DUPLICATE_AUTHORITY = "duplicate_authority"
    SCHEMA_DRIFT = "schema_drift"
    VERSION_INCOMPATIBLE = "version_incompatible"
    UNAVAILABLE = "unavailable"


CLOSED_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in ContractCompatibilityDisposition
)
CLOSED_DISPOSITION_ORDER: tuple[str, ...] = (
    ContractCompatibilityDisposition.COMPATIBLE.value,
    ContractCompatibilityDisposition.ADAPTER_REQUIRED.value,
    ContractCompatibilityDisposition.DUPLICATE_AUTHORITY.value,
    ContractCompatibilityDisposition.SCHEMA_DRIFT.value,
    ContractCompatibilityDisposition.VERSION_INCOMPATIBLE.value,
    ContractCompatibilityDisposition.UNAVAILABLE.value,
)

_REPORT_FIELDS = frozenset(
    {
        "authority",
        "closed_dispositions",
        "effect_class",
        "evidence",
        "extractor_identity",
        "proposal_packets",
        "required_gitlinks",
        "schema",
        "scopes",
        "task_id",
        "version",
        "write_policy",
    }
)
_SCOPE_FIELDS = frozenset(
    {
        "checkout_head",
        "checkout_matches_pin",
        "comparison",
        "disposition",
        "gitlink_path",
        "local_consumers",
        "observed_gitlink",
        "published_concern",
        "published_contracts",
        "repository",
        "required_pin",
        "unavailable",
    }
)
_CONTRACT_FIELDS = frozenset({"path", "present"})
_COMPARISON_FIELDS = frozenset(
    {
        "adapter_bound",
        "consumption_kind",
        "gitlink_available",
        "local_authority_claim",
        "published_authority_claim",
        "published_present",
        "shared_schema_tokens",
    }
)
_PACKET_FIELDS = frozenset(
    {
        "disposition",
        "gitlink_path",
        "local_adapter_alternative",
        "packet_id",
        "published_concern",
        "requested_change",
        "schema",
        "sibling_write_permitted",
        "target_repository",
    }
)
def _require_object(payload: Any, *, name: str = "payload") -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes, bytearray)):
        raise _error(f"{name} must be an object")
    return payload


def _require_exact_audit_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    extra = sorted(set(payload) - allowed_fields)
    if extra:
        raise _error(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise _error(f"{_MISSING_FIELD_MESSAGE}: {missing}")


@dataclass(frozen=True)
class SiblingScopeSpec:
    """Declared published-interface inspection root for one sibling."""

    repository: str
    gitlink_path: str
    required_pin: str
    published_concern: str
    published_paths: tuple[str, ...]
    local_consumer_paths: tuple[str, ...]
    published_authority_claim: bool
    local_authority_claim: bool
    consumption_kind: str = "direct_import"
    adapter_tokens: tuple[str, ...] = ()
    published_schema_tokens: tuple[str, ...] = ()
    local_schema_tokens: tuple[str, ...] = ()
    published_version_token: str = "1"
    local_version_token: str = "1"


DEFAULT_REQUIRED_GITLINKS: dict[str, str] = {
    "ipfs_datasets_py": "66a02063496fd200f2372b3083e376f1978c6be1",
    "ipfs_kit_py": "2564aea1ae35061f2165872aff91e8a40801ab7e",
    "ipfs_accelerate_py/mcplusplus": "5ac0ab162f420264fd224073a5df3f2d7c054ae3",
}

DEFAULT_SCOPE_SPECS: tuple[SiblingScopeSpec, ...] = (
    SiblingScopeSpec(
        repository="ipfs_datasets_py",
        gitlink_path="ipfs_datasets_py",
        required_pin=DEFAULT_REQUIRED_GITLINKS["ipfs_datasets_py"],
        published_concern="semantic and content identities",
        published_paths=(
            "ipfs_datasets_py/logic/software_contracts/content.py",
            "ipfs_datasets_py/duckdb_control/contracts.py",
        ),
        local_consumer_paths=(
            "ipfs_accelerate_py/utils/cid_utils.py",
            "ipfs_accelerate_py/assurance/content_identity.py",
            "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/"
            "architecture_ir.py",
        ),
        published_authority_claim=True,
        local_authority_claim=True,
        consumption_kind="parallel_authority",
        adapter_tokens=(),
        published_schema_tokens=(
            "software-contract-cid-profile-v1",
            "ipfs_datasets_py/duckdb-control-contracts@1",
        ),
        local_schema_tokens=(
            "ipfs_accelerate_py/assurance/content-identity@1",
            "ipfs_accelerate_py/agent-supervisor/architecture-ir@1",
        ),
        published_version_token="1",
        local_version_token="1",
    ),
    SiblingScopeSpec(
        repository="ipfs_kit_py",
        gitlink_path="ipfs_kit_py",
        required_pin=DEFAULT_REQUIRED_GITLINKS["ipfs_kit_py"],
        published_concern="storage and proof-seal authority",
        published_paths=("ipfs_kit_py/proof_seal_store/contracts.py",),
        local_consumer_paths=(
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "sealer.py",
        ),
        published_authority_claim=True,
        local_authority_claim=False,
        consumption_kind="direct_import",
        adapter_tokens=(
            "from ipfs_kit_py.proof_seal_store.contracts import",
            "from ipfs_kit_py.proof_seal_store.local_store import",
        ),
        published_schema_tokens=(
            "ArtifactKind",
            "ProofSealStore",
        ),
        local_schema_tokens=("ArtifactKind", "ProofSealStore"),
        published_version_token="1",
        local_version_token="1",
    ),
    SiblingScopeSpec(
        repository="MCP++",
        gitlink_path="ipfs_accelerate_py/mcplusplus",
        required_pin=DEFAULT_REQUIRED_GITLINKS["ipfs_accelerate_py/mcplusplus"],
        published_concern="wire and profile schemas",
        published_paths=(
            "schemas/execution/execution-envelope-1.schema.json",
            "schemas/canonicalization/mcpp-jcs-v1.schema.json",
            "schemas/profile-h/1.0/artifacts.schema.json",
            "scripts/generate_schemas.py",
        ),
        local_consumer_paths=(
            "ipfs_accelerate_py/mcp_server/mcplusplus/envelope.py",
            "ipfs_accelerate_py/mcp_server/mcplusplus/a2a_adapter.py",
        ),
        published_authority_claim=True,
        local_authority_claim=False,
        consumption_kind="runtime_adapter",
        adapter_tokens=(
            "RuntimeEnvelopeAdapter@1",
            'SCHEMA_ENVELOPE = "mcp++/execution/envelope@1"',
        ),
        published_schema_tokens=(
            "mcp++/execution/envelope@1",
            "mcp++/canonicalization/mcpp-jcs-v1@1",
            "mcp++/profile-h/1.0/artifacts@1",
        ),
        local_schema_tokens=(
            "mcp++/execution/envelope@1",
            "mcp++/a2a/agent-extension@1",
        ),
        published_version_token="1",
        local_version_token="1",
    ),
)


def _error(
    message: str,
    *,
    error_type: type[CrossRepositoryAuditError] = CrossRepositoryAuditError,
) -> CrossRepositoryAuditError:
    return error_type(message)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise _error(f"{name} must be a boolean")
    return value


def canonical_audit_json(payload: Mapping[str, Any]) -> str:
    """Return the committed inventory encoding."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _posix_parts(relative: str, *, name: str = "path") -> tuple[str, ...]:
    text = _require_text(relative, name, error_type=CrossRepositoryAuditError).replace("\\", "/")
    parts = tuple(part for part in text.split("/") if part not in ("", "."))
    if not parts or text.startswith("/") or any(part == ".." for part in parts):
        raise _error(
            f"{name} must be a repository-relative path",
            error_type=CrossRepositoryEscapeError,
        )
    return parts


def normalize_relative_path(relative: str, *, name: str = "path") -> str:
    """Return a repository-relative POSIX path with no escape components."""

    return "/".join(_posix_parts(relative, name=name))


def logical_path_under(relative: str, prefix: str) -> bool:
    """Return whether ``relative`` is ``prefix`` or a descendant (no I/O)."""

    path = normalize_relative_path(relative)
    root = normalize_relative_path(prefix, name="prefix")
    return path == root or path.startswith(root + "/")


def sibling_gitlink_paths(specs: Sequence[SiblingScopeSpec] | None = None) -> tuple[str, ...]:
    selected = specs if specs is not None else DEFAULT_SCOPE_SPECS
    return tuple(spec.gitlink_path for spec in selected)


def classify_compatibility(
    *,
    published_present: bool,
    gitlink_available: bool,
    published_version: str,
    local_version: str,
    published_markers: Iterable[str],
    local_markers: Iterable[str],
    published_authority_claim: bool,
    local_authority_claim: bool,
    adapter_bound: bool,
    consumption_kind: str = "direct_import",
) -> ContractCompatibilityDisposition:
    """Return one closed disposition. Unavailable never becomes compatible."""

    if not published_present or not gitlink_available:
        return ContractCompatibilityDisposition.UNAVAILABLE
    published_major = _major_version(published_version)
    local_major = _major_version(local_version)
    if published_major is not None and local_major is not None and published_major != local_major:
        return ContractCompatibilityDisposition.VERSION_INCOMPATIBLE
    published_set = {item for item in published_markers if item}
    local_set = {item for item in local_markers if item}
    shared = published_set & local_set
    if consumption_kind == "parallel_authority" or (
        published_authority_claim and local_authority_claim and not adapter_bound
    ):
        return ContractCompatibilityDisposition.DUPLICATE_AUTHORITY
    if published_set and local_set and not shared:
        return ContractCompatibilityDisposition.SCHEMA_DRIFT
    if consumption_kind == "runtime_adapter":
        return ContractCompatibilityDisposition.ADAPTER_REQUIRED
    if consumption_kind == "direct_import" and adapter_bound:
        return ContractCompatibilityDisposition.COMPATIBLE
    return ContractCompatibilityDisposition.COMPATIBLE


def _major_version(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


@dataclass(frozen=True)
class ProposalPacket:
    """External shared-contract proposal that never writes a sibling."""

    packet_id: str
    target_repository: str
    gitlink_path: str
    published_concern: str
    disposition: ContractCompatibilityDisposition
    requested_change: str
    local_adapter_alternative: str
    sibling_write_permitted: bool = False
    schema: str = PROPOSAL_PACKET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "packet_id",
            _require_text(self.packet_id, "packet_id", error_type=CrossRepositoryAuditError),
        )
        object.__setattr__(
            self,
            "target_repository",
            _require_text(
                self.target_repository,
                "target_repository",
                error_type=CrossRepositoryAuditError,
            ),
        )
        object.__setattr__(
            self,
            "gitlink_path",
            normalize_relative_path(self.gitlink_path, name="gitlink_path"),
        )
        object.__setattr__(
            self,
            "published_concern",
            _require_text(
                self.published_concern,
                "published_concern",
                error_type=CrossRepositoryAuditError,
            ),
        )
        object.__setattr__(
            self,
            "disposition",
            _closed_enum(
                self.disposition,
                ContractCompatibilityDisposition,
                "disposition",
                error_type=CrossRepositoryAuditError,
            ),
        )
        object.__setattr__(
            self,
            "requested_change",
            _require_text(
                self.requested_change,
                "requested_change",
                error_type=CrossRepositoryAuditError,
            ),
        )
        object.__setattr__(
            self,
            "local_adapter_alternative",
            _require_text(
                self.local_adapter_alternative,
                "local_adapter_alternative",
                error_type=CrossRepositoryAuditError,
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _require_text(self.schema, "schema", error_type=CrossRepositoryAuditError),
        )
        if self.sibling_write_permitted is not False:
            raise _error("proposal packets cannot permit sibling writes")
        if self.schema != PROPOSAL_PACKET_SCHEMA:
            raise _error("unsupported proposal packet schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "gitlink_path": self.gitlink_path,
            "local_adapter_alternative": self.local_adapter_alternative,
            "packet_id": self.packet_id,
            "published_concern": self.published_concern,
            "requested_change": self.requested_change,
            "schema": self.schema,
            "sibling_write_permitted": False,
            "target_repository": self.target_repository,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProposalPacket":
        mapping = _require_object(payload, name="proposal packet")
        _require_exact_audit_fields(mapping, _PACKET_FIELDS)
        return cls(
            packet_id=mapping["packet_id"],
            target_repository=mapping["target_repository"],
            gitlink_path=mapping["gitlink_path"],
            published_concern=mapping["published_concern"],
            disposition=mapping["disposition"],
            requested_change=mapping["requested_change"],
            local_adapter_alternative=mapping["local_adapter_alternative"],
            sibling_write_permitted=mapping["sibling_write_permitted"],
            schema=mapping["schema"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class CrossRepositoryContractAudit:
    """Evidence-bound audit of the three sibling published contracts."""

    scopes: tuple[dict[str, Any], ...]
    proposal_packets: tuple[ProposalPacket, ...]
    required_gitlinks: dict[str, str]
    schema: str = CROSS_REPOSITORY_AUDIT_SCHEMA
    version: int = CROSS_REPOSITORY_AUDIT_VERSION
    evidence: str = CROSS_REPOSITORY_AUDIT_EVIDENCE
    task_id: str = TASK_ID
    authority: bool = False
    write_policy: str = WRITE_POLICY
    effect_class: str = EFFECT_CLASS
    extractor_identity: str = EXTRACTOR_IDENTITY
    closed_dispositions: tuple[str, ...] = CLOSED_DISPOSITION_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _require_text(self.schema, "schema", error_type=CrossRepositoryAuditError),
        )
        if self.schema != CROSS_REPOSITORY_AUDIT_SCHEMA:
            raise _error("unsupported cross-repository audit schema")
        object.__setattr__(
            self,
            "version",
            _require_int(self.version, "version", error_type=CrossRepositoryAuditError),
        )
        if self.version != CROSS_REPOSITORY_AUDIT_VERSION:
            raise _error("unsupported cross-repository audit version")
        object.__setattr__(
            self,
            "evidence",
            _require_text(self.evidence, "evidence", error_type=CrossRepositoryAuditError),
        )
        object.__setattr__(
            self,
            "task_id",
            _require_text(self.task_id, "task_id", error_type=CrossRepositoryAuditError),
        )
        if self.task_id != TASK_ID:
            raise _error("task_id must be PCAR-025")
        if self.authority is not False:
            raise _error("cross-repository audit is not architecture authority")
        object.__setattr__(
            self,
            "write_policy",
            _require_text(self.write_policy, "write_policy", error_type=CrossRepositoryAuditError),
        )
        if self.write_policy != WRITE_POLICY:
            raise _error("write_policy must be deny")
        object.__setattr__(
            self,
            "effect_class",
            _require_text(self.effect_class, "effect_class", error_type=CrossRepositoryAuditError),
        )
        if self.effect_class != EFFECT_CLASS:
            raise _error("effect_class must be cross_repository_read_only")
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=CrossRepositoryAuditError,
            ),
        )
        dispositions = tuple(
            _require_text(item, "closed_dispositions item", error_type=CrossRepositoryAuditError)
            for item in self.closed_dispositions
        )
        if set(dispositions) != CLOSED_DISPOSITIONS or dispositions != CLOSED_DISPOSITION_ORDER:
            raise _error("closed_dispositions must be the exact closed vocabulary")
        object.__setattr__(self, "closed_dispositions", dispositions)
        gitlinks = {
            normalize_relative_path(key, name="gitlink path"): _require_text(
                value, "gitlink pin", error_type=CrossRepositoryAuditError
            )
            for key, value in dict(self.required_gitlinks).items()
        }
        if not gitlinks:
            raise _error("required_gitlinks must include the three sibling pins")
        object.__setattr__(self, "required_gitlinks", gitlinks)
        scopes = tuple(self._validate_scope(item) for item in self.scopes)
        if not scopes:
            raise _error("scopes must cover the three sibling published contracts")
        object.__setattr__(self, "scopes", scopes)
        packets = tuple(
            item if isinstance(item, ProposalPacket) else ProposalPacket.from_mapping(item)
            for item in self.proposal_packets
        )
        object.__setattr__(self, "proposal_packets", packets)
        self._assert_scope_coverage()

    def _validate_scope(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        mapping = _require_object(payload, name="scope")
        _require_exact_audit_fields(mapping, _SCOPE_FIELDS)
        disposition = _closed_enum(
            mapping["disposition"],
            ContractCompatibilityDisposition,
            "disposition",
            error_type=CrossRepositoryAuditError,
        )
        unavailable = _require_bool(mapping["unavailable"], "unavailable")
        if unavailable and disposition is not ContractCompatibilityDisposition.UNAVAILABLE:
            raise _error("unavailable stays unavailable")
        if (
            disposition is ContractCompatibilityDisposition.UNAVAILABLE
            and not unavailable
        ):
            raise _error("unavailable disposition requires unavailable=true")
        published = tuple(self._validate_contract(item, "published_contracts") for item in mapping["published_contracts"])
        consumers = tuple(self._validate_contract(item, "local_consumers") for item in mapping["local_consumers"])
        comparison = _require_object(mapping["comparison"], name="comparison")
        _require_exact_audit_fields(comparison, _COMPARISON_FIELDS)
        return {
            "checkout_head": mapping["checkout_head"],
            "checkout_matches_pin": _require_bool(
                mapping["checkout_matches_pin"], "checkout_matches_pin"
            ),
            "comparison": dict(comparison),
            "disposition": disposition.value,
            "gitlink_path": normalize_relative_path(mapping["gitlink_path"], name="gitlink_path"),
            "local_consumers": [dict(item) for item in consumers],
            "observed_gitlink": mapping["observed_gitlink"],
            "published_concern": _require_text(
                mapping["published_concern"],
                "published_concern",
                error_type=CrossRepositoryAuditError,
            ),
            "published_contracts": [dict(item) for item in published],
            "repository": _require_text(
                mapping["repository"], "repository", error_type=CrossRepositoryAuditError
            ),
            "required_pin": _require_text(
                mapping["required_pin"], "required_pin", error_type=CrossRepositoryAuditError
            ),
            "unavailable": unavailable,
        }

    def _validate_contract(self, payload: Any, name: str) -> dict[str, Any]:
        mapping = _require_object(payload, name=name)
        _require_exact_audit_fields(mapping, _CONTRACT_FIELDS)
        return {
            "path": normalize_relative_path(mapping["path"], name=f"{name} path"),
            "present": _require_bool(mapping["present"], "present"),
        }

    def _assert_scope_coverage(self) -> None:
        if len(self.scopes) != 3:
            raise _error("audit requires exactly three sibling scopes")
        paths = [item["gitlink_path"] for item in self.scopes]
        if len(set(paths)) != 3:
            raise _error("sibling gitlink paths must be unique")
        dispositions = {item["disposition"] for item in self.scopes}
        extra = dispositions - CLOSED_DISPOSITIONS
        if extra:
            raise _error(f"unsupported dispositions: {sorted(extra)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": False,
            "closed_dispositions": list(self.closed_dispositions),
            "effect_class": self.effect_class,
            "evidence": self.evidence,
            "extractor_identity": self.extractor_identity,
            "proposal_packets": [packet.to_dict() for packet in self.proposal_packets],
            "required_gitlinks": dict(sorted(self.required_gitlinks.items())),
            "schema": self.schema,
            "scopes": [dict(item) for item in self.scopes],
            "task_id": self.task_id,
            "version": self.version,
            "write_policy": self.write_policy,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CrossRepositoryContractAudit":
        mapping = _require_object(payload, name="audit report")
        _require_exact_audit_fields(mapping, _REPORT_FIELDS)
        packets = tuple(
            ProposalPacket.from_mapping(item) for item in mapping["proposal_packets"]
        )
        gitlinks = mapping["required_gitlinks"]
        if not isinstance(gitlinks, Mapping):
            raise _error("required_gitlinks must be an object")
        return cls(
            scopes=tuple(mapping["scopes"]),
            proposal_packets=packets,
            required_gitlinks=dict(gitlinks),
            schema=mapping["schema"],
            version=mapping["version"],
            evidence=mapping["evidence"],
            task_id=mapping["task_id"],
            authority=mapping["authority"],
            write_policy=mapping["write_policy"],
            effect_class=mapping["effect_class"],
            extractor_identity=mapping["extractor_identity"],
            closed_dispositions=tuple(mapping["closed_dispositions"]),
        )

    from_dict = from_mapping


class CrossRepositoryContractAuditor:
    """Read-only auditor over pinned sibling published contracts."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        scopes: Sequence[SiblingScopeSpec] | None = None,
        require_git: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise _error("audit root must be a directory")
        self.scopes = tuple(scopes) if scopes is not None else DEFAULT_SCOPE_SPECS
        if len(self.scopes) != 3:
            raise _error("audit requires exactly three sibling scopes")
        self.require_git = require_git
        self._write_rejection_traces: list[dict[str, str]] = []

    @property
    def write_rejection_traces(self) -> tuple[dict[str, str], ...]:
        return tuple(self._write_rejection_traces)

    def sibling_prefixes(self) -> tuple[str, ...]:
        return sibling_gitlink_paths(self.scopes)

    def refuse_sibling_write(self, relative_path: str, *, reason: str = "sibling_write") -> None:
        """Fail closed before any mutating I/O."""

        normalized = normalize_relative_path(relative_path)
        trace = {"path": normalized, "reason": reason}
        for prefix in self.sibling_prefixes():
            if logical_path_under(normalized, prefix):
                self._write_rejection_traces.append({**trace, "reason": "sibling_write"})
                raise CrossRepositoryWriteError(
                    f"sibling write forbidden before I/O: {normalized}"
                )
        self._reject_escape(normalized, write=True)
        self._write_rejection_traces.append(trace)
        raise CrossRepositoryWriteError(f"cross-repository write forbidden before I/O: {normalized}")

    def read_published(self, relative_path: str) -> bytes:
        """Read one published path after containment and symlink checks."""

        normalized = normalize_relative_path(relative_path)
        physical = self._contained_physical_path(normalized, write=False)
        if not physical.is_file() or physical.is_symlink():
            raise CrossRepositoryEscapeError(f"published path is not a regular file: {normalized}")
        return physical.read_bytes()

    def write_published(self, relative_path: str, data: bytes) -> None:
        """Rejected sibling mutation surface. Never opens a write handle."""

        del data
        self.refuse_sibling_write(relative_path, reason="write_published")

    def propose_shared_change(
        self,
        *,
        target_repository: str,
        gitlink_path: str,
        published_concern: str,
        disposition: ContractCompatibilityDisposition | str,
        requested_change: str,
        local_adapter_alternative: str,
        packet_id: str,
    ) -> ProposalPacket:
        """Return a proposal packet after refusing the sibling write logically."""

        return self.emit_proposal_packet(
            target_repository=target_repository,
            gitlink_path=gitlink_path,
            published_concern=published_concern,
            disposition=disposition,
            requested_change=requested_change,
            local_adapter_alternative=local_adapter_alternative,
            packet_id=packet_id,
        )

    def emit_proposal_packet(
        self,
        *,
        target_repository: str,
        gitlink_path: str,
        published_concern: str,
        disposition: ContractCompatibilityDisposition | str,
        requested_change: str,
        local_adapter_alternative: str,
        packet_id: str,
    ) -> ProposalPacket:
        """Construct a proposal packet after logically refusing the sibling write."""

        normalized = normalize_relative_path(gitlink_path, name="gitlink_path")
        if not any(logical_path_under(normalized, prefix) or normalized == prefix for prefix in self.sibling_prefixes()):
            # Still a sibling-targeted packet; require it to match a declared gitlink.
            if normalized not in self.sibling_prefixes():
                raise CrossRepositoryWriteError(
                    f"proposal target is not a declared sibling gitlink: {normalized}"
                )
        self._write_rejection_traces.append(
            {"path": normalized, "reason": "proposal_packet_no_sibling_write"}
        )
        return ProposalPacket(
            packet_id=packet_id,
            target_repository=target_repository,
            gitlink_path=normalized,
            published_concern=published_concern,
            disposition=disposition,
            requested_change=requested_change,
            local_adapter_alternative=local_adapter_alternative,
        )

    def audit(self) -> CrossRepositoryContractAudit:
        """Compare pinned published contracts with local consumers."""

        gitlinks = {spec.gitlink_path: spec.required_pin for spec in self.scopes}
        scopes: list[dict[str, Any]] = []
        packets: list[ProposalPacket] = []
        for spec in self.scopes:
            record, packet = self._audit_scope(spec)
            scopes.append(record)
            if packet is not None:
                packets.append(packet)
        return CrossRepositoryContractAudit(
            scopes=tuple(scopes),
            proposal_packets=tuple(packets),
            required_gitlinks=gitlinks,
        )

    def write_inventory(self, destination: str | os.PathLike[str] | None = None) -> Path:
        """Write the local audit inventory only to the owned path."""

        relative = INVENTORY_RELATIVE_PATH
        if destination is not None:
            dest = Path(destination)
            if dest.is_absolute():
                try:
                    relative = dest.resolve().relative_to(self.root).as_posix()
                except ValueError as exc:
                    raise CrossRepositoryEscapeError(
                        "inventory destination escapes the audit root"
                    ) from exc
            else:
                relative = dest.as_posix()
        relative = normalize_relative_path(relative)
        if relative != INVENTORY_RELATIVE_PATH:
            raise CrossRepositoryWriteError(
                f"inventory writes are limited to {INVENTORY_RELATIVE_PATH}"
            )
        physical = self._contained_physical_path(relative, write=True, allow_local_owned=True)
        payload = self.audit().to_dict()
        encoded = canonical_audit_json(payload)
        physical.parent.mkdir(parents=True, exist_ok=True)
        physical.write_text(encoded, encoding="utf-8")
        return physical

    def _audit_scope(
        self, spec: SiblingScopeSpec
    ) -> tuple[dict[str, Any], ProposalPacket | None]:
        observed, checkout, matches, gitlink_available = self._gitlink_identity(spec)
        published_records = [
            self._inspect_file(spec.gitlink_path, path, sibling=True)
            for path in spec.published_paths
        ]
        local_records = [
            self._inspect_file("", path, sibling=False) for path in spec.local_consumer_paths
        ]
        published_present = gitlink_available and all(item["present"] for item in published_records)
        published_text = "\n".join(item.get("_text", "") for item in published_records)
        local_text = "\n".join(item.get("_text", "") for item in local_records)
        adapter_bound = bool(spec.adapter_tokens) and all(
            token in local_text for token in spec.adapter_tokens
        )
        published_version = spec.published_version_token
        local_version = spec.local_version_token
        shared_schema_tokens = [
            token
            for token in spec.published_schema_tokens
            if token in published_text and token in local_text
        ]
        disposition = classify_compatibility(
            published_present=published_present,
            gitlink_available=gitlink_available,
            published_version=published_version,
            local_version=local_version,
            published_markers=spec.published_schema_tokens,
            local_markers=tuple(shared_schema_tokens) + spec.local_schema_tokens,
            published_authority_claim=spec.published_authority_claim,
            local_authority_claim=spec.local_authority_claim,
            adapter_bound=adapter_bound,
            consumption_kind=spec.consumption_kind,
        )
        comparison = {
            "adapter_bound": adapter_bound,
            "consumption_kind": spec.consumption_kind,
            "gitlink_available": gitlink_available,
            "local_authority_claim": spec.local_authority_claim,
            "published_authority_claim": spec.published_authority_claim,
            "published_present": published_present,
            "shared_schema_tokens": shared_schema_tokens,
        }
        record = {
            "checkout_head": checkout,
            "checkout_matches_pin": matches,
            "comparison": comparison,
            "disposition": disposition.value,
            "gitlink_path": spec.gitlink_path,
            "local_consumers": [self._public_contract(item) for item in local_records],
            "observed_gitlink": observed,
            "published_concern": spec.published_concern,
            "published_contracts": [self._public_contract(item) for item in published_records],
            "repository": spec.repository,
            "required_pin": spec.required_pin,
            "unavailable": disposition is ContractCompatibilityDisposition.UNAVAILABLE,
        }
        packet = None
        if disposition in {
            ContractCompatibilityDisposition.DUPLICATE_AUTHORITY,
            ContractCompatibilityDisposition.SCHEMA_DRIFT,
            ContractCompatibilityDisposition.VERSION_INCOMPATIBLE,
        }:
            packet = self.emit_proposal_packet(
                packet_id=f"pcar-025-{_packet_slug(spec.repository)}-{disposition.value}",
                target_repository=spec.repository,
                gitlink_path=spec.gitlink_path,
                published_concern=spec.published_concern,
                disposition=disposition,
                requested_change=(
                    "Do not write the sibling gitlink. If a shared published-contract "
                    f"change is required for {spec.published_concern}, emit this packet "
                    "to the sibling owner instead of mutating the pinned checkout."
                ),
                local_adapter_alternative=(
                    "Keep a local adapter or consume the pinned published interface; "
                    "do not create a second production authority."
                ),
            )
        return record, packet

    def _gitlink_identity(
        self, spec: SiblingScopeSpec
    ) -> tuple[str | None, str | None, bool, bool]:
        checkout_dir = self.root / spec.gitlink_path
        observed = self._git_ls_tree_gitlink(spec.gitlink_path)
        checkout = self._git_checkout_head(checkout_dir)
        matches = bool(
            observed
            and checkout
            and observed == spec.required_pin
            and checkout == spec.required_pin
        )
        available = checkout_dir.is_dir() and any(
            (checkout_dir / path).is_file() for path in spec.published_paths
        )
        if self.require_git:
            available = bool(available and matches)
        return observed, checkout, matches, available

    def _inspect_file(self, gitlink_path: str, relative: str, *, sibling: bool) -> dict[str, Any]:
        if sibling:
            path = f"{gitlink_path}/{relative}".replace("//", "/")
        else:
            path = relative
        normalized = normalize_relative_path(path)
        record: dict[str, Any] = {
            "path": normalized,
            "present": False,
            "_text": "",
        }
        try:
            physical = self._contained_physical_path(normalized, write=False)
        except CrossRepositoryAuditError:
            return record
        if not physical.is_file() or physical.is_symlink():
            return record
        text = physical.read_bytes().decode("utf-8", errors="replace")
        record.update({"present": True, "_text": text})
        return record

    def _public_contract(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {"path": record["path"], "present": record["present"]}

    def _contained_physical_path(
        self,
        relative: str,
        *,
        write: bool,
        allow_local_owned: bool = False,
    ) -> Path:
        normalized = normalize_relative_path(relative)
        if write and not allow_local_owned:
            self.refuse_sibling_write(normalized)
        if write and allow_local_owned:
            if any(logical_path_under(normalized, prefix) for prefix in self.sibling_prefixes()):
                raise CrossRepositoryWriteError(
                    f"sibling write forbidden before I/O: {normalized}"
                )
            if normalized != INVENTORY_RELATIVE_PATH:
                raise CrossRepositoryWriteError(
                    f"inventory writes are limited to {INVENTORY_RELATIVE_PATH}"
                )
        return self._reject_escape(normalized, write=write)

    def _reject_escape(self, relative: str, *, write: bool) -> Path:
        parts = _posix_parts(relative)
        current = self.root
        root_resolved = self.root.resolve()
        for index, part in enumerate(parts):
            current = current / part
            try:
                st = current.lstat()
            except FileNotFoundError:
                if write:
                    remainder = "/".join(parts[: index + 1])
                    if any(logical_path_under(remainder, prefix) for prefix in self.sibling_prefixes()):
                        raise CrossRepositoryWriteError(
                            f"sibling write forbidden before I/O: {relative}"
                        )
                    continue
                if index == len(parts) - 1:
                    break
                continue
            if stat.S_ISLNK(st.st_mode):
                raise CrossRepositoryEscapeError(f"symlink escape rejected before I/O: {relative}")
            if stat.S_ISDIR(st.st_mode) and (current / ".git").exists() and index + 1 < len(parts):
                nested_rel = "/".join(parts[: index + 1])
                if nested_rel in self.sibling_prefixes() and write:
                    raise CrossRepositoryWriteError(
                        f"submodule escape rejected before I/O: {relative}"
                    )
        try:
            # Resolve only after symlink refusal so we never follow an escape.
            if current.exists() and not current.is_symlink():
                resolved = current.resolve()
                resolved.relative_to(root_resolved)
                for prefix in self.sibling_prefixes():
                    sibling = (self.root / prefix).resolve()
                    if write and _is_relative_to(resolved, sibling):
                        raise CrossRepositoryWriteError(
                            f"sibling write forbidden before I/O: {relative}"
                        )
        except ValueError as exc:
            raise CrossRepositoryEscapeError(
                f"path escapes audit root before I/O: {relative}"
            ) from exc
        return current

    def _git_ls_tree_gitlink(self, path: str) -> str | None:
        raw = self._git("ls-tree", "HEAD", path, cwd=self.root)
        if raw is None:
            return None
        parts = raw.split()
        if len(parts) < 3:
            return None
        mode, kind, sha = parts[0], parts[1], parts[2]
        if mode != "160000" or kind != "commit" or not _SHA1_RE.fullmatch(sha):
            return None
        return sha

    def _git_checkout_head(self, checkout: Path) -> str | None:
        if not checkout.is_dir():
            return None
        return self._git("rev-parse", "HEAD", cwd=checkout)

    def _git(self, *args: str, cwd: Path) -> str | None:
        if not self.require_git and not (cwd / ".git").exists() and cwd != self.root:
            return None
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _packet_slug(repository: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", repository.lower()).strip("-")


def audit_cross_repository_contracts(
    root: str | os.PathLike[str],
    *,
    scopes: Sequence[SiblingScopeSpec] | None = None,
    require_git: bool = True,
) -> CrossRepositoryContractAudit:
    """Run the read-only three-sibling published-contract audit."""

    return CrossRepositoryContractAuditor(
        root, scopes=scopes, require_git=require_git
    ).audit()





__all__ = [
    "CLOSED_DISPOSITIONS",
    "CLOSED_DISPOSITION_ORDER",
    "CROSS_REPOSITORY_AUDIT_EVIDENCE",
    "CROSS_REPOSITORY_AUDIT_SCHEMA",
    "CROSS_REPOSITORY_AUDIT_VERSION",
    "ContractCompatibilityDisposition",
    "CrossRepositoryAuditError",
    "CrossRepositoryContractAudit",
    "CrossRepositoryContractAuditor",
    "CrossRepositoryEscapeError",
    "CrossRepositoryWriteError",
    "DEFAULT_REQUIRED_GITLINKS",
    "DEFAULT_SCOPE_SPECS",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "INVENTORY_RELATIVE_PATH",
    "PROPOSAL_PACKET_SCHEMA",
    "ProposalPacket",
    "SiblingScopeSpec",
    "TASK_ID",
    "WRITE_POLICY",
    "audit_cross_repository_contracts",
    "canonical_audit_json",
    "classify_compatibility",
    "logical_path_under",
    "normalize_relative_path",
]
