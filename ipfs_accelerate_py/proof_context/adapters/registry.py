"""Closed, explicit registry for v0.1 coding-agent adapters (PCCE-035).

The registry is an allowlist over the four concrete proposal sources supported
by v0.1.  It intentionally has no entry points, import strings, environment
configuration, or default adapter selection.  A caller must name one supported
adapter and supply a closed configuration for it.  Registry construction only
returns a proposal adapter; it never applies, verifies, seals, or accepts a
proposal.  Those operations remain lifecycle responsibilities.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    ADAPTER_CONTRACT_CID,
    APPROVAL_AUTHORITY,
    CANONICAL_BRANCH_AUTHORITY,
    CodingAgentAdapter,
)
from ipfs_accelerate_py.proof_context.adapters.codex import (
    ADAPTER as CODEX_ADAPTER,
    CODEX_ADAPTER_CID,
    CodexAdapter,
    CodexMechanismProbe,
)
from ipfs_accelerate_py.proof_context.adapters.command import (
    ADAPTER as COMMAND_ADAPTER,
    CommandAdapter,
    CommandPolicy,
)
from ipfs_accelerate_py.proof_context.adapters.external_patch import (
    ADAPTER as EXTERNAL_PATCH_ADAPTER,
    ExternalPatch,
    ExternalPatchAdapter,
)
from ipfs_accelerate_py.proof_context.adapters.models import wire_canonical_utf8
from ipfs_accelerate_py.proof_context.adapters.replay import (
    ADAPTER as REPLAY_ADAPTER,
    ReplayAdapter,
    ReplayFixture,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    UnknownFieldError,
)

REGISTRY_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/adapter-registry@1"
CONFIGURATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.proof-context.v0.1/adapter-configuration@1"
)
REGISTRY_INTERFACE: Final[str] = "CodingAgentAdapterRegistry@0.1"

CODEX: Final[str] = "codex"
COMMAND: Final[str] = "command"
REPLAY: Final[str] = "replay"
EXTERNAL_PATCH: Final[str] = "external-patch"
ADAPTER_NAMES: Final[tuple[str, ...]] = (CODEX, COMMAND, REPLAY, EXTERNAL_PATCH)

_OPTION_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        CODEX: frozenset({"transport", "permit_live", "probe"}),
        COMMAND: frozenset({"policy"}),
        REPLAY: frozenset(
            {"fixtures", "selected_fixture_cid", "selected_response_artifact_cid", "adapter_id"}
        ),
        EXTERNAL_PATCH: frozenset({"patch", "declared_files"}),
    }
)


def _mint_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(wire_canonical_utf8(value).encode("utf-8")).digest()
    raw = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def _closed_mapping(value: Any, *, allowed: frozenset[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MalformedError(f"{field} must be a mapping")
    unknown = set(value).difference(allowed)
    if unknown:
        raise UnknownFieldError(f"{field} contains unsupported field {sorted(unknown)[0]!r}")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class AdapterConfiguration:
    """One explicit, closed configuration for a named adapter.

    Values that are already capability objects (for example ``CommandPolicy``
    or a recorded replay fixture) are accepted only in their adapter's own
    slot.  They are not interpreted as import paths, shell fragments, or
    provider credentials.
    """

    name: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in ADAPTER_NAMES:
            raise UnknownFieldError(f"unsupported adapter name {self.name!r}")
        object.__setattr__(
            self,
            "options",
            _closed_mapping(self.options, allowed=_OPTION_KEYS[self.name], field="options"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdapterConfiguration":
        payload = _closed_mapping(
            value,
            allowed=frozenset({"schema", "name", "options"}),
            field="adapter configuration",
        )
        if payload.get("schema") != CONFIGURATION_SCHEMA:
            raise MalformedError("adapter configuration has an unsupported schema")
        if not isinstance(payload.get("name"), str) or "options" not in payload:
            raise MalformedError("adapter configuration requires name and options")
        return cls(payload["name"], payload["options"])

    def to_mapping(self) -> Mapping[str, Any]:
        # Capability values can be non-wire runtime objects, so this is a
        # descriptor rather than a persistence format.
        return MappingProxyType(
            {"schema": CONFIGURATION_SCHEMA, "name": self.name, "options": dict(self.options)}
        )


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise MalformedError(f"{field} must be a boolean")
    return value


def _create_codex(options: Mapping[str, Any]) -> CodexAdapter:
    transport = options.get("transport")
    permit_live = options.get("permit_live", False)
    probe = options.get("probe")
    if not isinstance(permit_live, bool):
        _require_bool(permit_live, field="codex permit_live")
    if probe is not None and not isinstance(probe, CodexMechanismProbe):
        raise MalformedError("codex probe must be a CodexMechanismProbe")
    # ``transport`` is deliberately opaque: the concrete adapter validates its
    # command contract before use, which permits recorded test transports while
    # preserving the unavailable-by-default production path.
    return CodexAdapter(transport=transport, permit_live=permit_live, probe=probe)


def _create_command(options: Mapping[str, Any]) -> CommandAdapter:
    policy = options.get("policy")
    if not isinstance(policy, CommandPolicy):
        raise MalformedError("command adapter requires an explicit CommandPolicy")
    return CommandAdapter(policy)


def _create_replay(options: Mapping[str, Any]) -> ReplayAdapter:
    required = ("fixtures", "selected_fixture_cid", "selected_response_artifact_cid")
    if any(name not in options for name in required):
        raise MalformedError("replay adapter requires explicit fixtures and selectors")
    fixtures = options["fixtures"]
    if isinstance(fixtures, (str, bytes, bytearray)) or not isinstance(fixtures, Sequence):
        raise MalformedError("replay fixtures must be a sequence")
    if not all(isinstance(item, ReplayFixture) for item in fixtures):
        raise MalformedError("replay fixtures must contain ReplayFixture records")
    selectors = (options["selected_fixture_cid"], options["selected_response_artifact_cid"])
    if not all(isinstance(item, str) and item for item in selectors):
        raise MalformedError("replay selectors must be non-empty strings")
    adapter_id = options.get("adapter_id", REPLAY_ADAPTER)
    if not isinstance(adapter_id, str) or not adapter_id:
        raise MalformedError("replay adapter_id must be a non-empty string")
    return ReplayAdapter(
        tuple(fixtures),
        selected_fixture_cid=selectors[0],
        selected_response_artifact_cid=selectors[1],
        adapter_id=adapter_id,
    )


def _create_external_patch(options: Mapping[str, Any]) -> ExternalPatchAdapter:
    if "patch" not in options:
        raise MalformedError("external-patch adapter requires patch bytes")
    patch = options["patch"]
    declared_files = options.get("declared_files")
    if isinstance(patch, ExternalPatch):
        if declared_files is not None:
            raise MalformedError("declared_files is invalid with an ExternalPatch")
        return ExternalPatchAdapter(patch)
    if not isinstance(patch, (bytes, bytearray, memoryview)):
        raise MalformedError("external-patch patch must be exact bytes")
    if isinstance(declared_files, (str, bytes, bytearray)) or not isinstance(declared_files, Sequence):
        raise MalformedError("external-patch adapter requires declared_files")
    return ExternalPatchAdapter(bytes(patch), tuple(declared_files))


@dataclass(frozen=True)
class AdapterRegistry:
    """The closed v0.1 adapter factory; it has no lifecycle operations."""

    descriptor_cid: str = ""

    def __post_init__(self) -> None:
        if self.descriptor_cid not in {"", REGISTRY_DESCRIPTOR_CID}:
            raise BoundaryViolationError("adapter registry descriptor identity drifted")
        object.__setattr__(self, "descriptor_cid", REGISTRY_DESCRIPTOR_CID)

    @property
    def names(self) -> tuple[str, ...]:
        return ADAPTER_NAMES

    def create(self, configuration: AdapterConfiguration | Mapping[str, Any]) -> CodingAgentAdapter:
        config = (
            configuration
            if isinstance(configuration, AdapterConfiguration)
            else AdapterConfiguration.from_mapping(configuration)
        )
        if config.name == CODEX:
            return _create_codex(config.options)
        if config.name == COMMAND:
            return _create_command(config.options)
        if config.name == REPLAY:
            return _create_replay(config.options)
        if config.name == EXTERNAL_PATCH:
            return _create_external_patch(config.options)
        raise UnknownFieldError(f"unsupported adapter name {config.name!r}")


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": REGISTRY_SCHEMA,
        "interface": REGISTRY_INTERFACE,
        "adapter_contract_cid": ADAPTER_CONTRACT_CID,
        "names": ADAPTER_NAMES,
        "configuration_schema": CONFIGURATION_SCHEMA,
        "adapter_descriptors": {
            CODEX: {"adapter": CODEX_ADAPTER, "cid": CODEX_ADAPTER_CID},
            COMMAND: {"adapter": COMMAND_ADAPTER},
            REPLAY: {"adapter": REPLAY_ADAPTER},
            EXTERNAL_PATCH: {"adapter": EXTERNAL_PATCH_ADAPTER},
        },
        "configuration_options": {name: tuple(sorted(options)) for name, options in _OPTION_KEYS.items()},
        "dynamic_imports": False,
        "implicit_credential_discovery": False,
        "default_shell_execution": False,
        "approval_authority": APPROVAL_AUTHORITY,
        "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
        "lifecycle_operations": False,
    }
)
REGISTRY_DESCRIPTOR_CID: Final[str] = _mint_cid(_DESCRIPTOR_BODY)
REGISTRY_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": REGISTRY_DESCRIPTOR_CID}
)
DEFAULT_ADAPTER_REGISTRY: Final[AdapterRegistry] = AdapterRegistry()


def adapter_registry_descriptor() -> Mapping[str, Any]:
    return REGISTRY_DESCRIPTOR


def create_adapter(configuration: AdapterConfiguration | Mapping[str, Any]) -> CodingAgentAdapter:
    """Create one explicitly configured adapter from the closed global registry."""

    return DEFAULT_ADAPTER_REGISTRY.create(configuration)


__all__ = [
    "ADAPTER_NAMES",
    "AdapterConfiguration",
    "AdapterRegistry",
    "CODEX",
    "COMMAND",
    "CONFIGURATION_SCHEMA",
    "DEFAULT_ADAPTER_REGISTRY",
    "EXTERNAL_PATCH",
    "REGISTRY_DESCRIPTOR",
    "REGISTRY_DESCRIPTOR_CID",
    "REGISTRY_INTERFACE",
    "REGISTRY_SCHEMA",
    "REPLAY",
    "adapter_registry_descriptor",
    "create_adapter",
]
