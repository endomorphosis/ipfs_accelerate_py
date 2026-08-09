"""UI, ORB, IDL, mobile, and projection repair operators (DCR-045).

Interface: ``UIProjectionRepairOperators@1``

Evidence: ``dcr/ui-repair@1``

Implements the structural preview/inverse bodies for
:attr:`OperatorKind.REPAIR_UI_PROJECTION`.  The library synchronizes typed
projections (desktop, web, CLI, mobile, ORB, and MCP-IDL) from a closed
semantic UI/UX IR document and proves:

* every edited projection roundtrips to the same semantic IR identity; and
* every live UI action reaches the expected mediated MCP effect.

Conflict policy (fail-closed):

* Full ``ui_ux_ir`` authority is required on the semantic source.
* Bridge-only, prose-inferred, or missing target projections abstain.
* Operators remain proposal-only: they never grant write, proof, or semantic
  authority and never mutate production trees.

Predicted symbols: :class:`UiDescriptorOperator`, :class:`OrbBindingOperator`,
:class:`IdlProjectionOperator`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...analysis.mcp_contract_catalog import (
    MCP_IDL_INTERFACE,
    ORB_INTERFACE,
    UIIR_DOCUMENT_INTERFACE,
)
from ...proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .registry import (
    OperatorKind,
    build_default_operator_registry,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

UI_PROJECTION_REPAIR_OPERATORS_INTERFACE: Final[str] = "UIProjectionRepairOperators@1"
UI_REPAIR_EVIDENCE: Final[str] = "dcr/ui-repair@1"
UI_UX_IR_SCHEMA_VERSION: Final[str] = "ui-ux-ir/v1"
UI_PROJECTION_REPAIR_VERSION: Final[int] = 1

UI_PROJECTION_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-projection-repair@1"
)
UI_PROJECTION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-projection-repair-request@1"
)
UI_PROJECTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-projection-repair-receipt@1"
)
UI_SEMANTIC_IR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-semantic-ir@1"
)
UI_SURFACE_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-surface-projection@1"
)
UI_ACTION_MEDIATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-action-mediation@1"
)
UI_PROJECTION_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-projection-operator-vectors@1"
)

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_COLLECTION: Final[int] = 256
MAX_REASON_CODES: Final[int] = 32

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
)
_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:bafy|bagu|bafk|sha256:)[A-Za-z0-9:_-]{8,200}$"
)

_FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
        "handler_body",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class UiProjectionRepairError(ContractValidationError):
    """Malformed UI projection repair input or closed-boundary violation."""


class ProjectionSurface(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed projection surfaces synchronized by DCR-045 operators."""

    DESKTOP = "desktop"
    WEB = "web"
    CLI = "cli"
    MOBILE = "mobile"
    ORB = "orb"
    IDL = "idl"


class SourceAuthority(str, Enum):  # noqa: UP042
    """Authority retained on semantic IR and projection artifacts.

    Only ``production`` (full ``ui_ux_ir``) may authorize a repair preview.
    Bridge-only, prose-inferred, and missing targets abstain.
    """

    PRODUCTION = "production"
    FULL_UI_UX_IR = "full_ui_ux_ir"
    BRIDGE_ONLY = "bridge_only"
    PROSE_INFERRED = "prose_inferred"
    MISSING = "missing"
    FIXTURE = "fixture"

    @property
    def authorizes_semantic_source(self) -> bool:
        return self in {SourceAuthority.PRODUCTION, SourceAuthority.FULL_UI_UX_IR}

    @property
    def is_abstaining_target(self) -> bool:
        return self in {
            SourceAuthority.BRIDGE_ONLY,
            SourceAuthority.PROSE_INFERRED,
            SourceAuthority.MISSING,
        }


class RepairDisposition(str, Enum):  # noqa: UP042
    """Closed outcomes for one UI projection repair attempt."""

    PREVIEW_READY = "preview_ready"
    ALREADY_ALIGNED = "already_aligned"
    ABSTAIN = "abstain"
    REJECTED = "rejected"


class OperatorRole(str, Enum):  # noqa: UP042
    """Closed operator roles implementing REPAIR_UI_PROJECTION."""

    UI_DESCRIPTOR = "ui_descriptor"
    ORB_BINDING = "orb_binding"
    IDL_PROJECTION = "idl_projection"
    MOBILE_PROJECTION = "mobile_projection"
    SURFACE_SYNC = "surface_sync"


class MediationPathClass(str, Enum):  # noqa: UP042
    """Closed mediation path classes for live UI → MCP effects."""

    GOVERNED_MEDIATOR = "governed_mediator"
    TOOLS_CALL = "tools_call"
    TOOLS_DISPATCH = "tools_dispatch"
    ORB_IDL_BRIDGE = "orb_idl_bridge"
    DIRECT_PROXY = "direct_proxy"  # rejected: not a governed path


# Surfaces each operator role is allowed to repair.
_ROLE_SURFACES: Final[Mapping[OperatorRole, frozenset[ProjectionSurface]]] = (
    MappingProxyType(
        {
            OperatorRole.UI_DESCRIPTOR: frozenset(
                {
                    ProjectionSurface.DESKTOP,
                    ProjectionSurface.WEB,
                    ProjectionSurface.CLI,
                    ProjectionSurface.MOBILE,
                }
            ),
            OperatorRole.ORB_BINDING: frozenset({ProjectionSurface.ORB}),
            OperatorRole.IDL_PROJECTION: frozenset({ProjectionSurface.IDL}),
            OperatorRole.MOBILE_PROJECTION: frozenset({ProjectionSurface.MOBILE}),
            OperatorRole.SURFACE_SYNC: frozenset(set(ProjectionSurface)),
        }
    )
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    identifier: bool = False,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise UiProjectionRepairError(f"{name} must be a string")
    if required and not text:
        raise UiProjectionRepairError(f"{name} must not be empty")
    if "\x00" in text:
        raise UiProjectionRepairError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > maximum:
        raise UiProjectionRepairError(f"{name} exceeds its byte bound")
    if identifier and text and not _IDENTIFIER_RE.fullmatch(text):
        raise UiProjectionRepairError(f"{name} is not a stable identifier")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise UiProjectionRepairError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise UiProjectionRepairError(f"{name} must be one of: {allowed}") from exc


def _cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=MAX_ID_BYTES)
    if text and not _CID_RE.fullmatch(text):
        raise UiProjectionRepairError(f"{name} is not a content identity")
    return text


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    forbidden = sorted(
        key for key in payload if str(key).lower() in _FORBIDDEN_PAYLOAD_KEYS
    )
    if forbidden:
        raise UiProjectionRepairError(
            f"{label} contains forbidden fields: {', '.join(forbidden)}"
        )


def _tuple_of(
    values: Any,
    name: str,
    builder: Any,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
) -> tuple[Any, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise UiProjectionRepairError(f"{name} must be a sequence")
    else:
        items = values
    if required and not items:
        raise UiProjectionRepairError(f"{name} must not be empty")
    if len(items) > maximum:
        raise UiProjectionRepairError(f"{name} exceeds its item bound")
    return tuple(builder(item, f"{name}[{index}]") for index, item in enumerate(items))


def _string_tuple(
    values: Any,
    name: str,
    *,
    required: bool = False,
    identifier: bool = False,
) -> tuple[str, ...]:
    def _one(item: Any, label: str) -> str:
        return _text(item, label, identifier=identifier)

    return _tuple_of(values, name, _one, required=required)


# ---------------------------------------------------------------------------
# Semantic IR and projection models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UiActionBinding(CanonicalContract):
    """One UI action bound to a mediated MCP method effect."""

    SCHEMA: ClassVar[str] = UI_ACTION_MEDIATION_SCHEMA

    action_id: str
    label: str
    mcp_method: str
    interface_cid: str
    effect_id: str
    mediation_path: MediationPathClass = MediationPathClass.GOVERNED_MEDIATOR
    argument_schema_cid: str = ""
    result_schema_cid: str = ""
    confirmation_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_id", _text(self.action_id, "action_id", identifier=True)
        )
        object.__setattr__(self, "label", _text(self.label, "label"))
        object.__setattr__(
            self, "mcp_method", _text(self.mcp_method, "mcp_method", identifier=True)
        )
        object.__setattr__(
            self, "interface_cid", _cid(self.interface_cid, "interface_cid")
        )
        object.__setattr__(
            self, "effect_id", _text(self.effect_id, "effect_id", identifier=True)
        )
        object.__setattr__(
            self,
            "mediation_path",
            _enum(self.mediation_path, MediationPathClass, "mediation_path"),
        )
        object.__setattr__(
            self,
            "argument_schema_cid",
            _cid(self.argument_schema_cid, "argument_schema_cid", required=False),
        )
        object.__setattr__(
            self,
            "result_schema_cid",
            _cid(self.result_schema_cid, "result_schema_cid", required=False),
        )
        object.__setattr__(
            self,
            "confirmation_required",
            _bool(self.confirmation_required, "confirmation_required"),
        )
        if self.mediation_path is MediationPathClass.DIRECT_PROXY:
            raise UiProjectionRepairError(
                "direct_proxy mediation is rejected; only governed paths are admissible"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "mcp_method": self.mcp_method,
            "interface_cid": self.interface_cid,
            "effect_id": self.effect_id,
            "mediation_path": self.mediation_path.value,
            "argument_schema_cid": self.argument_schema_cid,
            "result_schema_cid": self.result_schema_cid,
            "confirmation_required": self.confirmation_required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UiActionBinding":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("action binding must be an object")
        _reject_forbidden_fields(payload, label="action binding")
        return cls(
            action_id=payload.get("action_id", ""),
            label=payload.get("label", ""),
            mcp_method=payload.get("mcp_method", ""),
            interface_cid=payload.get("interface_cid", ""),
            effect_id=payload.get("effect_id", ""),
            mediation_path=payload.get(
                "mediation_path", MediationPathClass.GOVERNED_MEDIATOR
            ),
            argument_schema_cid=payload.get("argument_schema_cid", ""),
            result_schema_cid=payload.get("result_schema_cid", ""),
            confirmation_required=payload.get("confirmation_required", False),
        )


@dataclass(frozen=True)
class UiComponentNode(CanonicalContract):
    """One semantic UI component retained in the closed IR."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/ui-component-node@1"

    component_id: str
    role: str
    purpose: str
    action_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "component_id", identifier=True),
        )
        object.__setattr__(self, "role", _text(self.role, "role", identifier=True))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose"))
        object.__setattr__(
            self,
            "action_ids",
            _string_tuple(self.action_ids, "action_ids", identifier=True),
        )
        object.__setattr__(
            self,
            "child_ids",
            _string_tuple(self.child_ids, "child_ids", identifier=True),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "role": self.role,
            "purpose": self.purpose,
            "action_ids": list(self.action_ids),
            "child_ids": list(self.child_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UiComponentNode":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("component must be an object")
        _reject_forbidden_fields(payload, label="component")
        return cls(
            component_id=payload.get("component_id", ""),
            role=payload.get("role", ""),
            purpose=payload.get("purpose", ""),
            action_ids=payload.get("action_ids") or (),
            child_ids=payload.get("child_ids") or (),
        )


@dataclass(frozen=True)
class UIIRSemanticDocument(CanonicalContract):
    """Closed semantic UI/UX IR document (``UIIRDocument`` projection authority).

    This is the DCR-045 semantic identity used for roundtrip proofs.  It is a
    closed repair-facing subset of ``ui-ux-ir/v1`` and never claims execution
    authority.
    """

    SCHEMA: ClassVar[str] = UI_SEMANTIC_IR_SCHEMA
    INTERFACE: ClassVar[str] = UIIR_DOCUMENT_INTERFACE

    document_id: str
    title: str
    components: tuple[UiComponentNode, ...]
    actions: tuple[UiActionBinding, ...]
    entry_components: tuple[str, ...]
    terminal_outcomes: tuple[str, ...]
    schema_version: str = UI_UX_IR_SCHEMA_VERSION
    authority: SourceAuthority = SourceAuthority.FULL_UI_UX_IR
    schema_cid: str = ""
    orb_interface_cid: str = ""
    idl_interface_cid: str = ""
    mcp_idl_interface: str = MCP_IDL_INTERFACE
    orb_interface: str = ORB_INTERFACE
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "document_id", identifier=True),
        )
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version"),
        )
        if self.schema_version != UI_UX_IR_SCHEMA_VERSION:
            raise UiProjectionRepairError(
                f"schema_version must be exactly {UI_UX_IR_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self, "authority", _enum(self.authority, SourceAuthority, "authority")
        )
        components = _tuple_of(
            self.components,
            "components",
            lambda item, label: (
                item
                if isinstance(item, UiComponentNode)
                else UiComponentNode.from_dict(item)
            ),
            required=True,
        )
        object.__setattr__(self, "components", components)
        actions = _tuple_of(
            self.actions,
            "actions",
            lambda item, label: (
                item
                if isinstance(item, UiActionBinding)
                else UiActionBinding.from_dict(item)
            ),
            required=True,
        )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "entry_components",
            _string_tuple(self.entry_components, "entry_components", required=True, identifier=True),
        )
        object.__setattr__(
            self,
            "terminal_outcomes",
            _string_tuple(
                self.terminal_outcomes, "terminal_outcomes", required=True, identifier=True
            ),
        )
        object.__setattr__(
            self, "schema_cid", _cid(self.schema_cid, "schema_cid", required=False)
        )
        object.__setattr__(
            self,
            "orb_interface_cid",
            _cid(self.orb_interface_cid, "orb_interface_cid", required=False),
        )
        object.__setattr__(
            self,
            "idl_interface_cid",
            _cid(self.idl_interface_cid, "idl_interface_cid", required=False),
        )
        object.__setattr__(
            self,
            "mcp_idl_interface",
            _text(self.mcp_idl_interface, "mcp_idl_interface"),
        )
        object.__setattr__(
            self, "orb_interface", _text(self.orb_interface, "orb_interface")
        )
        if self.mcp_idl_interface != MCP_IDL_INTERFACE:
            raise UiProjectionRepairError(
                f"mcp_idl_interface must be {MCP_IDL_INTERFACE}"
            )
        if self.orb_interface != ORB_INTERFACE:
            raise UiProjectionRepairError(f"orb_interface must be {ORB_INTERFACE}")
        object.__setattr__(
            self,
            "source_refs",
            _string_tuple(self.source_refs, "source_refs", identifier=True),
        )
        component_ids = {item.component_id for item in self.components}
        for entry in self.entry_components:
            if entry not in component_ids:
                raise UiProjectionRepairError(
                    f"entry_components references unknown component: {entry}"
                )
        action_ids = {item.action_id for item in self.actions}
        for component in self.components:
            for action_id in component.action_ids:
                if action_id not in action_ids:
                    raise UiProjectionRepairError(
                        f"component {component.component_id} references unknown action"
                    )
        # Derive missing identity CIDs from stable cores that do not embed the
        # CID fields themselves (avoids content_id circularity).
        if not self.idl_interface_cid:
            object.__setattr__(
                self,
                "idl_interface_cid",
                content_identity(
                    {
                        "interface": MCP_IDL_INTERFACE,
                        "document_id": self.document_id,
                        "methods": [item.mcp_method for item in self.actions],
                    }
                ),
            )
        if not self.orb_interface_cid:
            object.__setattr__(
                self,
                "orb_interface_cid",
                content_identity(
                    {
                        "interface": ORB_INTERFACE,
                        "document_id": self.document_id,
                        "actions": [item.action_id for item in self.actions],
                    }
                ),
            )
        if not self.schema_cid:
            object.__setattr__(
                self,
                "schema_cid",
                content_identity(
                    {
                        "interface": UIIR_DOCUMENT_INTERFACE,
                        "schema_version": self.schema_version,
                        "document_id": self.document_id,
                        "components": [item.to_dict() for item in self.components],
                        "actions": [item.to_dict() for item in self.actions],
                    }
                ),
            )

    @property
    def semantic_digest(self) -> str:
        return _digest(self.to_dict())

    def semantic_core_digest(self) -> str:
        """Digest of the semantic core used for projection roundtrip proofs."""

        return _digest(
            {
                "document_id": self.document_id,
                "components": [item.to_dict() for item in self.components],
                "actions": [item.to_dict() for item in self.actions],
                "entry_components": list(self.entry_components),
                "schema_version": self.schema_version,
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "title": self.title,
            "authority": self.authority.value,
            "schema_cid": self.schema_cid,
            "orb_interface_cid": self.orb_interface_cid,
            "idl_interface_cid": self.idl_interface_cid,
            "mcp_idl_interface": self.mcp_idl_interface,
            "orb_interface": self.orb_interface,
            "components": [item.to_dict() for item in self.components],
            "actions": [item.to_dict() for item in self.actions],
            "entry_components": list(self.entry_components),
            "terminal_outcomes": list(self.terminal_outcomes),
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UIIRSemanticDocument":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("semantic IR must be an object")
        _reject_forbidden_fields(payload, label="semantic IR")
        interface = payload.get("interface", UIIR_DOCUMENT_INTERFACE)
        if interface not in (None, "", UIIR_DOCUMENT_INTERFACE, "UIUXIR@1"):
            raise UiProjectionRepairError("unsupported semantic IR interface")
        return cls(
            document_id=payload.get("document_id", ""),
            title=payload.get("title", ""),
            components=payload.get("components") or (),
            actions=payload.get("actions") or (),
            entry_components=payload.get("entry_components") or (),
            terminal_outcomes=payload.get("terminal_outcomes") or (),
            schema_version=payload.get("schema_version", UI_UX_IR_SCHEMA_VERSION),
            authority=payload.get("authority", SourceAuthority.FULL_UI_UX_IR),
            schema_cid=payload.get("schema_cid", ""),
            orb_interface_cid=payload.get("orb_interface_cid", ""),
            idl_interface_cid=payload.get("idl_interface_cid", ""),
            mcp_idl_interface=payload.get("mcp_idl_interface", MCP_IDL_INTERFACE),
            orb_interface=payload.get("orb_interface", ORB_INTERFACE),
            source_refs=payload.get("source_refs") or (),
        )


@dataclass(frozen=True)
class SurfaceProjection(CanonicalContract):
    """One surface projection derived from (or repaired toward) semantic IR."""

    SCHEMA: ClassVar[str] = UI_SURFACE_PROJECTION_SCHEMA

    surface: ProjectionSurface
    projection_id: str
    document_id: str
    nodes: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    source_schema_cid: str
    target_schema_cid: str
    authority: SourceAuthority = SourceAuthority.PRODUCTION
    mcp_interface_cid: str = ""
    orb_interface_cid: str = ""
    semantic_digest: str = ""
    mediation_path: MediationPathClass = MediationPathClass.GOVERNED_MEDIATOR

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface", _enum(self.surface, ProjectionSurface, "surface")
        )
        object.__setattr__(
            self,
            "projection_id",
            _text(self.projection_id, "projection_id", identifier=True),
        )
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "document_id", identifier=True),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, SourceAuthority, "authority")
        )
        object.__setattr__(
            self, "source_schema_cid", _cid(self.source_schema_cid, "source_schema_cid")
        )
        object.__setattr__(
            self, "target_schema_cid", _cid(self.target_schema_cid, "target_schema_cid")
        )
        object.__setattr__(
            self,
            "mcp_interface_cid",
            _cid(self.mcp_interface_cid, "mcp_interface_cid", required=False),
        )
        object.__setattr__(
            self,
            "orb_interface_cid",
            _cid(self.orb_interface_cid, "orb_interface_cid", required=False),
        )
        object.__setattr__(
            self,
            "semantic_digest",
            _text(self.semantic_digest, "semantic_digest", required=False),
        )
        object.__setattr__(
            self,
            "mediation_path",
            _enum(self.mediation_path, MediationPathClass, "mediation_path"),
        )
        nodes = _tuple_of(
            self.nodes,
            "nodes",
            lambda item, label: _require_mapping(item, label),
            required=False,
        )
        object.__setattr__(self, "nodes", nodes)
        actions = _tuple_of(
            self.actions,
            "actions",
            lambda item, label: _require_mapping(item, label),
            required=False,
        )
        object.__setattr__(self, "actions", actions)
        if self.mediation_path is MediationPathClass.DIRECT_PROXY:
            raise UiProjectionRepairError(
                "surface projection cannot claim a direct_proxy mediation path"
            )

    @property
    def projection_digest(self) -> str:
        return _digest(self.to_dict())

    def _payload(self) -> dict[str, Any]:
        return {
            "surface": self.surface.value,
            "projection_id": self.projection_id,
            "document_id": self.document_id,
            "authority": self.authority.value,
            "source_schema_cid": self.source_schema_cid,
            "target_schema_cid": self.target_schema_cid,
            "mcp_interface_cid": self.mcp_interface_cid,
            "orb_interface_cid": self.orb_interface_cid,
            "semantic_digest": self.semantic_digest,
            "mediation_path": self.mediation_path.value,
            "nodes": [dict(item) for item in self.nodes],
            "actions": [dict(item) for item in self.actions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurfaceProjection":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("surface projection must be an object")
        _reject_forbidden_fields(payload, label="surface projection")
        return cls(
            surface=payload.get("surface", ProjectionSurface.DESKTOP),
            projection_id=payload.get("projection_id", ""),
            document_id=payload.get("document_id", ""),
            nodes=payload.get("nodes") or (),
            actions=payload.get("actions") or (),
            source_schema_cid=payload.get("source_schema_cid", ""),
            target_schema_cid=payload.get("target_schema_cid", ""),
            authority=payload.get("authority", SourceAuthority.PRODUCTION),
            mcp_interface_cid=payload.get("mcp_interface_cid", ""),
            orb_interface_cid=payload.get("orb_interface_cid", ""),
            semantic_digest=payload.get("semantic_digest", ""),
            mediation_path=payload.get(
                "mediation_path", MediationPathClass.GOVERNED_MEDIATOR
            ),
        )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UiProjectionRepairError(f"{name} must be an object")
    _reject_forbidden_fields(value, label=name)
    # Canonicalize nested mapping to plain dict with sorted JSON stability.
    return json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))


# ---------------------------------------------------------------------------
# Projection synthesis and roundtrip
# ---------------------------------------------------------------------------


def project_semantic_ir(
    document: UIIRSemanticDocument,
    surface: ProjectionSurface | str,
    *,
    projection_id: str | None = None,
) -> SurfaceProjection:
    """Project a full semantic IR document onto a closed surface.

    The projection is deterministic and lossless with respect to the semantic
    action/component identities required for roundtrip.  Presentation-only
    fields are surface-tagged but do not alter semantic identity.
    """

    if not isinstance(document, UIIRSemanticDocument):
        raise UiProjectionRepairError("document must be a UIIRSemanticDocument")
    surface_enum = _enum(surface, ProjectionSurface, "surface")
    if not document.authority.authorizes_semantic_source:
        raise UiProjectionRepairError(
            "project_semantic_ir requires full ui_ux_ir / production semantic authority"
        )

    entry_set = set(document.entry_components)
    nodes = tuple(
        {
            "component_id": component.component_id,
            "role": component.role,
            "purpose": component.purpose,
            "action_ids": list(component.action_ids),
            "child_ids": list(component.child_ids),
            "surface": surface_enum.value,
            "entry": component.component_id in entry_set,
        }
        for component in document.components
    )
    actions = tuple(
        {
            "action_id": action.action_id,
            "label": action.label,
            "mcp_method": action.mcp_method,
            "interface_cid": action.interface_cid,
            "effect_id": action.effect_id,
            "mediation_path": action.mediation_path.value,
            "argument_schema_cid": action.argument_schema_cid,
            "result_schema_cid": action.result_schema_cid,
            "confirmation_required": action.confirmation_required,
            "surface": surface_enum.value,
        }
        for action in document.actions
    )

    if surface_enum is ProjectionSurface.IDL:
        target_schema = document.idl_interface_cid
        mcp_cid = document.idl_interface_cid
        orb_cid = ""
        mediation = MediationPathClass.TOOLS_DISPATCH
    elif surface_enum is ProjectionSurface.ORB:
        target_schema = document.orb_interface_cid
        mcp_cid = document.idl_interface_cid
        orb_cid = document.orb_interface_cid
        mediation = MediationPathClass.ORB_IDL_BRIDGE
    elif surface_enum is ProjectionSurface.MOBILE:
        target_schema = content_identity(
            {
                "surface": "mobile",
                "document_id": document.document_id,
                "schema_cid": document.schema_cid,
            }
        )
        mcp_cid = document.idl_interface_cid
        orb_cid = document.orb_interface_cid
        mediation = MediationPathClass.GOVERNED_MEDIATOR
    else:
        target_schema = content_identity(
            {
                "surface": surface_enum.value,
                "document_id": document.document_id,
                "schema_cid": document.schema_cid,
            }
        )
        mcp_cid = document.idl_interface_cid
        orb_cid = document.orb_interface_cid
        mediation = MediationPathClass.GOVERNED_MEDIATOR

    pid = projection_id or f"projection:{surface_enum.value}:{document.document_id}"
    return SurfaceProjection(
        surface=surface_enum,
        projection_id=pid,
        document_id=document.document_id,
        nodes=nodes,
        actions=actions,
        source_schema_cid=document.schema_cid,
        target_schema_cid=target_schema,
        authority=SourceAuthority.PRODUCTION,
        mcp_interface_cid=mcp_cid,
        orb_interface_cid=orb_cid,
        semantic_digest=document.semantic_digest,
        mediation_path=mediation,
    )


def semantic_ir_from_projection(
    projection: SurfaceProjection,
    *,
    title: str = "",
    entry_components: Sequence[str] | None = None,
    terminal_outcomes: Sequence[str] = ("success", "failure"),
    authority: SourceAuthority = SourceAuthority.FULL_UI_UX_IR,
) -> UIIRSemanticDocument:
    """Reconstruct semantic IR from a surface projection (roundtrip inverse)."""

    if not isinstance(projection, SurfaceProjection):
        raise UiProjectionRepairError("projection must be a SurfaceProjection")
    if projection.authority.is_abstaining_target:
        raise UiProjectionRepairError(
            "cannot reconstruct semantic IR from an abstaining target projection"
        )
    if not projection.nodes or not projection.actions:
        raise UiProjectionRepairError(
            "projection is missing nodes/actions required for semantic reconstruction"
        )

    components = tuple(
        UiComponentNode(
            component_id=str(node["component_id"]),
            role=str(node["role"]),
            purpose=str(node["purpose"]),
            action_ids=tuple(node.get("action_ids") or ()),
            child_ids=tuple(node.get("child_ids") or ()),
        )
        for node in projection.nodes
    )
    actions = tuple(
        UiActionBinding(
            action_id=str(action["action_id"]),
            label=str(action["label"]),
            mcp_method=str(action["mcp_method"]),
            interface_cid=str(action["interface_cid"]),
            effect_id=str(action["effect_id"]),
            mediation_path=action.get(
                "mediation_path", MediationPathClass.GOVERNED_MEDIATOR
            ),
            argument_schema_cid=str(action.get("argument_schema_cid") or ""),
            result_schema_cid=str(action.get("result_schema_cid") or ""),
            confirmation_required=bool(action.get("confirmation_required", False)),
        )
        for action in projection.actions
    )
    if entry_components:
        entry = tuple(entry_components)
    else:
        # Prefer an explicit entry marker stamped onto projection nodes.
        stamped = tuple(
            str(node["component_id"])
            for node in projection.nodes
            if node.get("entry") is True
        )
        entry = stamped or (components[0].component_id,)
    return UIIRSemanticDocument(
        document_id=projection.document_id,
        title=title or projection.document_id,
        components=components,
        actions=actions,
        entry_components=entry,
        terminal_outcomes=tuple(terminal_outcomes) or ("success", "failure"),
        authority=authority,
        schema_cid=projection.source_schema_cid,
        orb_interface_cid=projection.orb_interface_cid,
        idl_interface_cid=projection.mcp_interface_cid or projection.target_schema_cid,
    )


def assert_semantic_roundtrip(
    document: UIIRSemanticDocument,
    surface: ProjectionSurface | str,
) -> SurfaceProjection:
    """Project then reconstruct; fail unless semantic digests match."""

    projection = project_semantic_ir(document, surface)
    restored = semantic_ir_from_projection(
        projection,
        title=document.title,
        entry_components=document.entry_components,
        terminal_outcomes=document.terminal_outcomes,
        authority=document.authority,
    )
    # Compare the semantic core (components + actions + document identity).
    if restored.semantic_core_digest() != document.semantic_core_digest():
        raise UiProjectionRepairError(
            "projection does not roundtrip to the same semantic IR"
        )
    if projection.semantic_digest != document.semantic_digest:
        # Projection must carry the source semantic digest for evidence.
        raise UiProjectionRepairError(
            "projection semantic_digest does not match source IR"
        )
    return projection


def projection_diff(
    current: SurfaceProjection | None,
    expected: SurfaceProjection,
) -> dict[str, Any]:
    """Return a closed, content-addressed projection diff."""

    if current is None:
        return {
            "kind": "missing_target",
            "expected_digest": expected.projection_digest,
            "current_digest": "",
            "changed_paths": ("projection",),
        }
    if current.surface is not expected.surface:
        raise UiProjectionRepairError("projection surfaces must match for diff")
    current_payload = current.to_dict()
    expected_payload = expected.to_dict()
    changed: list[str] = []
    for key in sorted(set(current_payload) | set(expected_payload)):
        if current_payload.get(key) != expected_payload.get(key):
            changed.append(key)
    return {
        "kind": "aligned" if not changed else "drift",
        "expected_digest": expected.projection_digest,
        "current_digest": current.projection_digest,
        "changed_paths": tuple(changed),
    }


# ---------------------------------------------------------------------------
# Live action mediation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveActionTrace(CanonicalContract):
    """Observed (or fixture) live UI action → MCP mediation transcript row."""

    SCHEMA: ClassVar[str] = UI_ACTION_MEDIATION_SCHEMA

    action_id: str
    mcp_method: str
    effect_id: str
    mediation_path: MediationPathClass
    terminal_state: str
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_id", _text(self.action_id, "action_id", identifier=True)
        )
        object.__setattr__(
            self, "mcp_method", _text(self.mcp_method, "mcp_method", identifier=True)
        )
        object.__setattr__(
            self, "effect_id", _text(self.effect_id, "effect_id", identifier=True)
        )
        object.__setattr__(
            self,
            "mediation_path",
            _enum(self.mediation_path, MediationPathClass, "mediation_path"),
        )
        if self.mediation_path is MediationPathClass.DIRECT_PROXY:
            raise UiProjectionRepairError(
                "direct_proxy mediation is rejected; only governed paths are admissible"
            )
        object.__setattr__(
            self,
            "terminal_state",
            _text(self.terminal_state, "terminal_state", identifier=True),
        )
        object.__setattr__(
            self, "receipt_cid", _cid(self.receipt_cid, "receipt_cid", required=False)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "mcp_method": self.mcp_method,
            "effect_id": self.effect_id,
            "mediation_path": self.mediation_path.value,
            "terminal_state": self.terminal_state,
            "receipt_cid": self.receipt_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiveActionTrace":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("live action trace must be an object")
        return cls(
            action_id=payload.get("action_id", ""),
            mcp_method=payload.get("mcp_method", ""),
            effect_id=payload.get("effect_id", ""),
            mediation_path=payload.get(
                "mediation_path", MediationPathClass.GOVERNED_MEDIATOR
            ),
            terminal_state=payload.get("terminal_state", ""),
            receipt_cid=payload.get("receipt_cid", ""),
        )


def verify_mediated_mcp_effects(
    document: UIIRSemanticDocument,
    traces: Sequence[LiveActionTrace | Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Prove each IR action reaches its expected mediated MCP effect.

    Returns ``(ok, reason_codes, evidence_rows)``.  Direct proxies and missing
    traces fail closed.
    """

    normalized: list[LiveActionTrace] = []
    for index, item in enumerate(traces):
        if isinstance(item, LiveActionTrace):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(LiveActionTrace.from_dict(item))
        else:
            raise UiProjectionRepairError(f"traces[{index}] must be an object")

    by_action = {trace.action_id: trace for trace in normalized}
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    ok = True
    for action in document.actions:
        trace = by_action.get(action.action_id)
        if trace is None:
            ok = False
            reasons.append(f"missing_live_trace:{action.action_id}")
            continue
        if trace.mcp_method != action.mcp_method:
            ok = False
            reasons.append(f"mcp_method_mismatch:{action.action_id}")
        if trace.effect_id != action.effect_id:
            ok = False
            reasons.append(f"effect_mismatch:{action.action_id}")
        if trace.mediation_path is MediationPathClass.DIRECT_PROXY:
            ok = False
            reasons.append(f"direct_proxy_rejected:{action.action_id}")
        if trace.mediation_path is not action.mediation_path:
            # Allow any governed path class that is not direct proxy when the
            # action declares a governed mediator.
            governed = {
                MediationPathClass.GOVERNED_MEDIATOR,
                MediationPathClass.TOOLS_CALL,
                MediationPathClass.TOOLS_DISPATCH,
                MediationPathClass.ORB_IDL_BRIDGE,
            }
            if (
                action.mediation_path in governed
                and trace.mediation_path in governed
            ):
                pass
            else:
                ok = False
                reasons.append(f"mediation_path_mismatch:{action.action_id}")
        if trace.terminal_state not in {"passed", "success", "ok"}:
            ok = False
            reasons.append(f"live_action_not_successful:{action.action_id}")
        evidence.append(
            {
                "action_id": action.action_id,
                "expected_effect_id": action.effect_id,
                "observed_effect_id": trace.effect_id,
                "mcp_method": trace.mcp_method,
                "mediation_path": trace.mediation_path.value,
                "terminal_state": trace.terminal_state,
                "receipt_cid": trace.receipt_cid,
            }
        )
    if ok and not reasons:
        reasons.append("mediated_mcp_effects_verified")
    return ok, tuple(reasons[:MAX_REASON_CODES]), tuple(evidence)


# ---------------------------------------------------------------------------
# Repair request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UiProjectionRepairRequest(CanonicalContract):
    """Closed input for one UI projection repair operator application."""

    SCHEMA: ClassVar[str] = UI_PROJECTION_REQUEST_SCHEMA

    semantic_ir: UIIRSemanticDocument
    surface: ProjectionSurface
    role: OperatorRole = OperatorRole.SURFACE_SYNC
    current_projection: SurfaceProjection | None = None
    live_traces: tuple[LiveActionTrace, ...] = ()
    require_live_mediation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_ir, UIIRSemanticDocument):
            if isinstance(self.semantic_ir, Mapping):
                object.__setattr__(
                    self, "semantic_ir", UIIRSemanticDocument.from_dict(self.semantic_ir)
                )
            else:
                raise UiProjectionRepairError("semantic_ir must be a UIIRSemanticDocument")
        object.__setattr__(
            self, "surface", _enum(self.surface, ProjectionSurface, "surface")
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        if self.current_projection is not None and not isinstance(
            self.current_projection, SurfaceProjection
        ):
            if isinstance(self.current_projection, Mapping):
                object.__setattr__(
                    self,
                    "current_projection",
                    SurfaceProjection.from_dict(self.current_projection),
                )
            else:
                raise UiProjectionRepairError(
                    "current_projection must be a SurfaceProjection or null"
                )
        traces = _tuple_of(
            self.live_traces,
            "live_traces",
            lambda item, label: (
                item if isinstance(item, LiveActionTrace) else LiveActionTrace.from_dict(item)
            ),
            required=False,
        )
        object.__setattr__(self, "live_traces", traces)
        object.__setattr__(
            self,
            "require_live_mediation",
            _bool(self.require_live_mediation, "require_live_mediation"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "semantic_ir": self.semantic_ir.to_dict(),
            "surface": self.surface.value,
            "role": self.role.value,
            "current_projection": (
                None
                if self.current_projection is None
                else self.current_projection.to_dict()
            ),
            "live_traces": [item.to_dict() for item in self.live_traces],
            "require_live_mediation": self.require_live_mediation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UiProjectionRepairRequest":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("repair request must be an object")
        _reject_forbidden_fields(payload, label="repair request")
        return cls(
            semantic_ir=payload.get("semantic_ir") or {},
            surface=payload.get("surface", ProjectionSurface.DESKTOP),
            role=payload.get("role", OperatorRole.SURFACE_SYNC),
            current_projection=payload.get("current_projection"),
            live_traces=payload.get("live_traces") or (),
            require_live_mediation=payload.get("require_live_mediation", True),
        )


@dataclass(frozen=True)
class UiProjectionRepairReceipt(CanonicalContract):
    """Non-authoritative preview/inverse receipt for one UI projection repair."""

    SCHEMA: ClassVar[str] = UI_PROJECTION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE

    disposition: RepairDisposition
    role: OperatorRole
    surface: ProjectionSurface
    operator_kind: str
    reason_codes: tuple[str, ...]
    source_schema_cid: str
    target_schema_cid: str
    semantic_digest: str
    expected_projection: SurfaceProjection | None = None
    preview_projection: SurfaceProjection | None = None
    inverse_projection: SurfaceProjection | None = None
    projection_diff: Mapping[str, Any] = MappingProxyType({})
    mediation_evidence: tuple[dict[str, Any], ...] = ()
    semantic_roundtrip_ok: bool = False
    live_mediation_ok: bool = False
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    evidence_id: str = UI_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", _enum(self.disposition, RepairDisposition, "disposition")
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "surface", _enum(self.surface, ProjectionSurface, "surface")
        )
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes", required=True),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise UiProjectionRepairError("reason_codes exceeds its item bound")
        object.__setattr__(
            self, "source_schema_cid", _cid(self.source_schema_cid, "source_schema_cid")
        )
        object.__setattr__(
            self,
            "target_schema_cid",
            _cid(self.target_schema_cid, "target_schema_cid", required=False),
        )
        object.__setattr__(
            self, "semantic_digest", _text(self.semantic_digest, "semantic_digest")
        )
        object.__setattr__(
            self, "semantic_roundtrip_ok", _bool(self.semantic_roundtrip_ok, "semantic_roundtrip_ok")
        )
        object.__setattr__(
            self, "live_mediation_ok", _bool(self.live_mediation_ok, "live_mediation_ok")
        )
        # Authority flags are sealed closed.
        for flag in (
            "proposal_only",
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
        ):
            current = getattr(self, flag)
            if flag == "proposal_only":
                if current is not True:
                    raise UiProjectionRepairError("receipts must remain proposal-only")
                object.__setattr__(self, flag, True)
            else:
                if current is not False:
                    raise UiProjectionRepairError(f"{flag} cannot be true on a repair receipt")
                object.__setattr__(self, flag, False)
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        if self.evidence_id != UI_REPAIR_EVIDENCE:
            raise UiProjectionRepairError(
                f"evidence_id must be exactly {UI_REPAIR_EVIDENCE}"
            )
        if self.projection_diff is None:
            object.__setattr__(self, "projection_diff", MappingProxyType({}))
        elif not isinstance(self.projection_diff, Mapping):
            raise UiProjectionRepairError("projection_diff must be an object")
        else:
            object.__setattr__(
                self, "projection_diff", MappingProxyType(dict(self.projection_diff))
            )

    @property
    def is_editable(self) -> bool:
        """Whether a preview edit was produced (still non-writing)."""

        return self.disposition is RepairDisposition.PREVIEW_READY

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "role": self.role.value,
            "surface": self.surface.value,
            "operator_kind": self.operator_kind,
            "reason_codes": list(self.reason_codes),
            "source_schema_cid": self.source_schema_cid,
            "target_schema_cid": self.target_schema_cid,
            "semantic_digest": self.semantic_digest,
            "expected_projection": (
                None
                if self.expected_projection is None
                else self.expected_projection.to_dict()
            ),
            "preview_projection": (
                None
                if self.preview_projection is None
                else self.preview_projection.to_dict()
            ),
            "inverse_projection": (
                None
                if self.inverse_projection is None
                else self.inverse_projection.to_dict()
            ),
            "projection_diff": dict(self.projection_diff),
            "mediation_evidence": [dict(item) for item in self.mediation_evidence],
            "semantic_roundtrip_ok": self.semantic_roundtrip_ok,
            "live_mediation_ok": self.live_mediation_ok,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": UI_PROJECTION_REPAIR_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UiProjectionRepairReceipt":
        if not isinstance(payload, Mapping):
            raise UiProjectionRepairError("repair receipt must be an object")
        _reject_forbidden_fields(payload, label="repair receipt")
        expected = payload.get("expected_projection")
        preview = payload.get("preview_projection")
        inverse = payload.get("inverse_projection")
        return cls(
            disposition=payload.get("disposition", RepairDisposition.REJECTED),
            role=payload.get("role", OperatorRole.SURFACE_SYNC),
            surface=payload.get("surface", ProjectionSurface.DESKTOP),
            operator_kind=payload.get("operator_kind", OperatorKind.REPAIR_UI_PROJECTION.value),
            reason_codes=payload.get("reason_codes") or ("rejected",),
            source_schema_cid=payload.get("source_schema_cid", ""),
            target_schema_cid=payload.get("target_schema_cid", ""),
            semantic_digest=payload.get("semantic_digest", ""),
            expected_projection=(
                None if expected is None else SurfaceProjection.from_dict(expected)
            ),
            preview_projection=(
                None if preview is None else SurfaceProjection.from_dict(preview)
            ),
            inverse_projection=(
                None if inverse is None else SurfaceProjection.from_dict(inverse)
            ),
            projection_diff=payload.get("projection_diff") or {},
            mediation_evidence=tuple(payload.get("mediation_evidence") or ()),
            semantic_roundtrip_ok=payload.get("semantic_roundtrip_ok", False),
            live_mediation_ok=payload.get("live_mediation_ok", False),
            proposal_only=payload.get("proposal_only", True),
            grants_write_authority=payload.get("grants_write_authority", False),
            grants_proof_authority=payload.get("grants_proof_authority", False),
            semantic_authority=payload.get("semantic_authority", False),
            evidence_id=payload.get("evidence_id", UI_REPAIR_EVIDENCE),
        )


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------


def _registry_descriptor():
    registry = build_default_operator_registry()
    return registry.require_known(OperatorKind.REPAIR_UI_PROJECTION)


def _abstain_receipt(
    request: UiProjectionRepairRequest,
    *,
    reasons: Sequence[str],
    role: OperatorRole,
) -> UiProjectionRepairReceipt:
    return UiProjectionRepairReceipt(
        disposition=RepairDisposition.ABSTAIN,
        role=role,
        surface=request.surface,
        operator_kind=OperatorKind.REPAIR_UI_PROJECTION.value,
        reason_codes=tuple(reasons) or ("abstain",),
        source_schema_cid=request.semantic_ir.schema_cid or content_identity({"empty": True}),
        target_schema_cid=(
            request.current_projection.target_schema_cid
            if request.current_projection is not None
            else content_identity({"missing_target": True})
        ),
        semantic_digest=request.semantic_ir.semantic_digest,
        expected_projection=None,
        preview_projection=None,
        inverse_projection=request.current_projection,
        projection_diff={"kind": "abstain", "changed_paths": ()},
        mediation_evidence=(),
        semantic_roundtrip_ok=False,
        live_mediation_ok=False,
    )


def _reject_receipt(
    request: UiProjectionRepairRequest,
    *,
    reasons: Sequence[str],
    role: OperatorRole,
) -> UiProjectionRepairReceipt:
    return UiProjectionRepairReceipt(
        disposition=RepairDisposition.REJECTED,
        role=role,
        surface=request.surface,
        operator_kind=OperatorKind.REPAIR_UI_PROJECTION.value,
        reason_codes=tuple(reasons) or ("rejected",),
        source_schema_cid=request.semantic_ir.schema_cid or content_identity({"empty": True}),
        target_schema_cid=(
            request.current_projection.target_schema_cid
            if request.current_projection is not None
            else content_identity({"missing_target": True})
        ),
        semantic_digest=request.semantic_ir.semantic_digest,
        expected_projection=None,
        preview_projection=None,
        inverse_projection=request.current_projection,
        projection_diff={"kind": "rejected", "changed_paths": ()},
        mediation_evidence=(),
        semantic_roundtrip_ok=False,
        live_mediation_ok=False,
    )


def _apply_repair(
    request: UiProjectionRepairRequest,
    *,
    role: OperatorRole,
) -> UiProjectionRepairReceipt:
    """Shared applicability + preview + roundtrip + mediation pipeline."""

    # Confirm the registry descriptor remains proposal-only metadata.
    descriptor = _registry_descriptor()
    if descriptor.kind is not OperatorKind.REPAIR_UI_PROJECTION:
        return _reject_receipt(
            request, reasons=("registry_kind_mismatch",), role=role
        )
    if descriptor.proposal_only is not True or descriptor.grants_write_authority:
        return _reject_receipt(
            request, reasons=("descriptor_authority_violation",), role=role
        )

    allowed = _ROLE_SURFACES[role]
    if request.surface not in allowed:
        return _reject_receipt(
            request,
            reasons=(f"surface_not_in_role:{request.surface.value}",),
            role=role,
        )

    semantic = request.semantic_ir
    if not semantic.authority.authorizes_semantic_source:
        return _abstain_receipt(
            request,
            reasons=(
                "semantic_source_not_full_ui_ux_ir",
                f"authority:{semantic.authority.value}",
            ),
            role=role,
        )

    current = request.current_projection
    if current is None:
        return _abstain_receipt(
            request,
            reasons=("missing_target_projection", "conflict_policy_abstain"),
            role=role,
        )
    if current.authority.is_abstaining_target:
        return _abstain_receipt(
            request,
            reasons=(
                f"target_authority_{current.authority.value}",
                "conflict_policy_abstain",
            ),
            role=role,
        )
    if current.surface is not request.surface:
        return _reject_receipt(
            request,
            reasons=("current_surface_mismatch",),
            role=role,
        )
    if current.document_id != semantic.document_id:
        return _reject_receipt(
            request,
            reasons=("document_id_mismatch",),
            role=role,
        )

    # Synthesize expected projection from full semantic IR and prove roundtrip.
    try:
        expected = assert_semantic_roundtrip(semantic, request.surface)
    except UiProjectionRepairError as exc:
        return _reject_receipt(
            request,
            reasons=("semantic_roundtrip_failed", str(exc)[:200]),
            role=role,
        )

    # Live mediation gate.
    live_ok = True
    mediation_reasons: tuple[str, ...] = ()
    mediation_evidence: tuple[dict[str, Any], ...] = ()
    if request.require_live_mediation:
        if not request.live_traces:
            return _reject_receipt(
                request,
                reasons=("live_mediation_traces_required",),
                role=role,
            )
        live_ok, mediation_reasons, mediation_evidence = verify_mediated_mcp_effects(
            semantic, request.live_traces
        )
        if not live_ok:
            return _reject_receipt(
                request,
                reasons=("live_mediation_failed", *mediation_reasons),
                role=role,
            )

    diff = projection_diff(current, expected)
    if diff["kind"] == "aligned":
        return UiProjectionRepairReceipt(
            disposition=RepairDisposition.ALREADY_ALIGNED,
            role=role,
            surface=request.surface,
            operator_kind=OperatorKind.REPAIR_UI_PROJECTION.value,
            reason_codes=("already_aligned", "semantic_roundtrip_ok", *mediation_reasons),
            source_schema_cid=semantic.schema_cid,
            target_schema_cid=expected.target_schema_cid,
            semantic_digest=semantic.semantic_digest,
            expected_projection=expected,
            preview_projection=expected,
            inverse_projection=current,
            projection_diff=diff,
            mediation_evidence=mediation_evidence,
            semantic_roundtrip_ok=True,
            live_mediation_ok=live_ok,
        )

    # Preview is the expected projection; inverse restores the current one.
    # Re-verify that the preview still roundtrips.
    restored = semantic_ir_from_projection(
        expected,
        title=semantic.title,
        entry_components=semantic.entry_components,
        terminal_outcomes=semantic.terminal_outcomes,
        authority=semantic.authority,
    )
    if restored.semantic_core_digest() != semantic.semantic_core_digest():
        return _reject_receipt(
            request,
            reasons=("preview_roundtrip_mismatch",),
            role=role,
        )

    return UiProjectionRepairReceipt(
        disposition=RepairDisposition.PREVIEW_READY,
        role=role,
        surface=request.surface,
        operator_kind=OperatorKind.REPAIR_UI_PROJECTION.value,
        reason_codes=(
            "preview_ready",
            "semantic_roundtrip_ok",
            "projection_drift",
            *mediation_reasons,
        ),
        source_schema_cid=semantic.schema_cid,
        target_schema_cid=expected.target_schema_cid,
        semantic_digest=semantic.semantic_digest,
        expected_projection=expected,
        preview_projection=expected,
        inverse_projection=current,
        projection_diff=diff,
        mediation_evidence=mediation_evidence,
        semantic_roundtrip_ok=True,
        live_mediation_ok=live_ok,
    )


class UiDescriptorOperator:
    """Synchronize desktop/web/CLI/mobile UI descriptor projections."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.UI_DESCRIPTOR
    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE

    def apply(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        return _apply_repair(request, role=self.ROLE)

    def preview(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: UiProjectionRepairReceipt) -> SurfaceProjection | None:
        if not isinstance(receipt, UiProjectionRepairReceipt):
            raise UiProjectionRepairError("receipt must be a UiProjectionRepairReceipt")
        return receipt.inverse_projection


class OrbBindingOperator:
    """Synchronize ORB bindings from semantic UI IR and MCP-IDL identity."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.ORB_BINDING
    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE

    def apply(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        # Force ORB surface when callers omit it incorrectly.
        if request.surface is not ProjectionSurface.ORB:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=ProjectionSurface.ORB,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        else:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=request.surface,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        return _apply_repair(request, role=self.ROLE)

    def preview(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: UiProjectionRepairReceipt) -> SurfaceProjection | None:
        if not isinstance(receipt, UiProjectionRepairReceipt):
            raise UiProjectionRepairError("receipt must be a UiProjectionRepairReceipt")
        return receipt.inverse_projection


class IdlProjectionOperator:
    """Synchronize MCP-IDL projections from semantic UI IR bindings."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.IDL_PROJECTION
    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE

    def apply(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        if request.surface is not ProjectionSurface.IDL:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=ProjectionSurface.IDL,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        else:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=request.surface,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        return _apply_repair(request, role=self.ROLE)

    def preview(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: UiProjectionRepairReceipt) -> SurfaceProjection | None:
        if not isinstance(receipt, UiProjectionRepairReceipt):
            raise UiProjectionRepairError("receipt must be a UiProjectionRepairReceipt")
        return receipt.inverse_projection


class MobileProjectionOperator:
    """Synchronize mobile companion projections from semantic UI IR."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.MOBILE_PROJECTION
    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE

    def apply(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        if request.surface is not ProjectionSurface.MOBILE:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=ProjectionSurface.MOBILE,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        else:
            request = UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=request.surface,
                role=self.ROLE,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        return _apply_repair(request, role=self.ROLE)

    def preview(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: UiProjectionRepairReceipt) -> SurfaceProjection | None:
        if not isinstance(receipt, UiProjectionRepairReceipt):
            raise UiProjectionRepairError("receipt must be a UiProjectionRepairReceipt")
        return receipt.inverse_projection


class UIProjectionRepairOperators:
    """Facade over the closed DCR-045 UI/ORB/IDL/mobile operator set."""

    INTERFACE: ClassVar[str] = UI_PROJECTION_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = UI_REPAIR_EVIDENCE

    def __init__(self) -> None:
        self.ui_descriptor = UiDescriptorOperator()
        self.orb_binding = OrbBindingOperator()
        self.idl_projection = IdlProjectionOperator()
        self.mobile_projection = MobileProjectionOperator()

    def for_role(self, role: OperatorRole | str) -> Any:
        role_enum = _enum(role, OperatorRole, "role")
        mapping = {
            OperatorRole.UI_DESCRIPTOR: self.ui_descriptor,
            OperatorRole.ORB_BINDING: self.orb_binding,
            OperatorRole.IDL_PROJECTION: self.idl_projection,
            OperatorRole.MOBILE_PROJECTION: self.mobile_projection,
            OperatorRole.SURFACE_SYNC: self.ui_descriptor,
        }
        return mapping[role_enum]

    def apply(self, request: UiProjectionRepairRequest) -> UiProjectionRepairReceipt:
        if not isinstance(request, UiProjectionRepairRequest):
            if isinstance(request, Mapping):
                request = UiProjectionRepairRequest.from_dict(request)
            else:
                raise UiProjectionRepairError("request must be a UiProjectionRepairRequest")
        # Route by explicit role, with surface-based defaults.
        role = request.role
        if role is OperatorRole.SURFACE_SYNC:
            if request.surface is ProjectionSurface.ORB:
                role = OperatorRole.ORB_BINDING
            elif request.surface is ProjectionSurface.IDL:
                role = OperatorRole.IDL_PROJECTION
            elif request.surface is ProjectionSurface.MOBILE:
                role = OperatorRole.MOBILE_PROJECTION
            else:
                role = OperatorRole.UI_DESCRIPTOR
        return self.for_role(role).apply(
            UiProjectionRepairRequest(
                semantic_ir=request.semantic_ir,
                surface=request.surface,
                role=role,
                current_projection=request.current_projection,
                live_traces=request.live_traces,
                require_live_mediation=request.require_live_mediation,
            )
        )

    def repair_all_surfaces(
        self,
        semantic_ir: UIIRSemanticDocument,
        projections: Mapping[str, SurfaceProjection | Mapping[str, Any] | None],
        live_traces: Sequence[LiveActionTrace | Mapping[str, Any]] = (),
    ) -> tuple[UiProjectionRepairReceipt, ...]:
        """Apply the appropriate operator to each declared surface projection."""

        receipts: list[UiProjectionRepairReceipt] = []
        for surface in ProjectionSurface:
            raw = projections.get(surface.value)
            if raw is None and surface.value not in projections:
                continue
            current: SurfaceProjection | None
            if raw is None:
                current = None
            elif isinstance(raw, SurfaceProjection):
                current = raw
            else:
                current = SurfaceProjection.from_dict(raw)
            request = UiProjectionRepairRequest(
                semantic_ir=semantic_ir,
                surface=surface,
                role=OperatorRole.SURFACE_SYNC,
                current_projection=current,
                live_traces=tuple(live_traces),
                require_live_mediation=True,
            )
            receipts.append(self.apply(request))
        return tuple(receipts)

    def to_artifact_dict(self) -> dict[str, Any]:
        descriptor = _registry_descriptor()
        payload = {
            "schema": UI_PROJECTION_ARTIFACT_SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": self.EVIDENCE_ID,
            "operator_kind": OperatorKind.REPAIR_UI_PROJECTION.value,
            "operator_id": descriptor.operator_id,
            "roles": [role.value for role in OperatorRole],
            "surfaces": [surface.value for surface in ProjectionSurface],
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": UI_PROJECTION_REPAIR_VERSION,
        }
        return {
            **payload,
            "artifact_digest": _digest(payload),
        }


def build_ui_projection_repair_operators() -> UIProjectionRepairOperators:
    """Return the sealed DCR-045 operator facade."""

    return UIProjectionRepairOperators()


def materialize_ui_projection_operator_vectors() -> dict[str, Any]:
    """Materialize the non-authoritative operator vector artifact payload."""

    return build_ui_projection_repair_operators().to_artifact_dict()


__all__ = (
    "UI_PROJECTION_REPAIR_OPERATORS_INTERFACE",
    "UI_REPAIR_EVIDENCE",
    "UI_UX_IR_SCHEMA_VERSION",
    "UIIR_DOCUMENT_INTERFACE",
    "MCP_IDL_INTERFACE",
    "ORB_INTERFACE",
    "UiProjectionRepairError",
    "ProjectionSurface",
    "SourceAuthority",
    "RepairDisposition",
    "OperatorRole",
    "MediationPathClass",
    "UiActionBinding",
    "UiComponentNode",
    "UIIRSemanticDocument",
    "SurfaceProjection",
    "LiveActionTrace",
    "UiProjectionRepairRequest",
    "UiProjectionRepairReceipt",
    "UiDescriptorOperator",
    "OrbBindingOperator",
    "IdlProjectionOperator",
    "MobileProjectionOperator",
    "UIProjectionRepairOperators",
    "project_semantic_ir",
    "semantic_ir_from_projection",
    "assert_semantic_roundtrip",
    "projection_diff",
    "verify_mediated_mcp_effects",
    "build_ui_projection_repair_operators",
    "materialize_ui_projection_operator_vectors",
)
