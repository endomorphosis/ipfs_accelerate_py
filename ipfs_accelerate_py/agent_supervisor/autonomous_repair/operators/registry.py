"""DCR-040 finite reviewed repair-operator registry (enumeration only).

Descriptors are data, not code: this module has no import loader, subprocess,
network client, filesystem write, or execution method.  Production activation
remains integration-pending until DCR-035's mandatory logic gate is current.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ...proof.formal_verification_contracts import content_identity

OPERATOR_REGISTRY_INTERFACE: Final[str] = "DeterministicRepairOperatorRegistry@1"
OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-operator-registry@1"
)
_OPERATOR_KINDS: Final[frozenset[str]] = frozenset(
    {"replace_exact_bytes", "rename_exact_symbol", "replace_unique_registration"}
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@/-]{0,383}$")
_ARGV = re.compile(r"^-?[A-Za-z0-9][A-Za-z0-9_.@/=-]{0,383}$")
_FORBIDDEN_TEXT = re.compile(r"[\n\r;|&`$<>]|\b(import|exec|eval|lambda|subprocess|shell)\b", re.I)


class RepairOperatorRegistryError(ValueError):
    """A descriptor or reviewed manifest is not closed deterministic data."""


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or _FORBIDDEN_TEXT.search(value)
    ):
        raise RepairOperatorRegistryError(f"{field} must be a closed identifier")
    return value


def _path(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _PATH.fullmatch(value)
        or value.startswith("/")
        or ".." in value.split("/")
        or _FORBIDDEN_TEXT.search(value)
    ):
        raise RepairOperatorRegistryError(f"{field} must be a bounded relative path")
    return value


def _identifiers(value: Any, field: str, *, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RepairOperatorRegistryError(f"{field} must be a sequence")
    result = tuple(_identifier(item, field) for item in value)
    if len(result) < minimum or len(set(result)) != len(result):
        raise RepairOperatorRegistryError(f"{field} must be non-empty and unique")
    return result


def _input_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "type",
        "required",
        "properties",
        "additional_properties",
    }:
        raise RepairOperatorRegistryError("input_schema must use the exact closed schema")
    if value.get("type") != "object" or value.get("additional_properties") is not False:
        raise RepairOperatorRegistryError("input_schema must be a closed object")
    required = _identifiers(value.get("required"), "input_schema.required")
    properties = value.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != set(required):
        raise RepairOperatorRegistryError("input_schema properties must exactly match required")
    normalized = {
        key: _identifier(item, "input_schema.property_type") for key, item in properties.items()
    }
    if any(item not in {"sha256", "path", "symbol", "cid"} for item in normalized.values()):
        raise RepairOperatorRegistryError("input_schema property type is not admitted")
    return {
        "type": "object",
        "required": list(required),
        "properties": dict(sorted(normalized.items())),
        "additional_properties": False,
    }


def _preview(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "fields"}:
        raise RepairOperatorRegistryError("preview must be a closed metadata template")
    if value.get("kind") != "metadata_only":
        raise RepairOperatorRegistryError("preview must be metadata_only")
    return {
        "kind": "metadata_only",
        "fields": list(_identifiers(value.get("fields"), "preview.fields")),
    }


def _inverse(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "binding"}:
        raise RepairOperatorRegistryError("inverse must be a closed reversible binding")
    if value.get("kind") != "restore_exact_before_bytes":
        raise RepairOperatorRegistryError("inverse kind is not admitted")
    return {
        "kind": "restore_exact_before_bytes",
        "binding": _identifier(value.get("binding"), "inverse.binding"),
    }


def _commands(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RepairOperatorRegistryError("validation_commands must be non-empty structured argv")
    commands: list[tuple[str, ...]] = []
    for command in value:
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
            raise RepairOperatorRegistryError("validation command must be non-empty argv")
        argv = tuple(_argv_token(token) for token in command)
        commands.append(argv)
    return tuple(commands)


def _argv_token(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _ARGV.fullmatch(value)
        or ".." in value.split("/")
        or _FORBIDDEN_TEXT.search(value)
    ):
        raise RepairOperatorRegistryError("validation command token is not closed argv")
    return value


@dataclass(frozen=True)
class OperatorDescriptor:
    operator_id: str
    kind: str
    input_schema: Mapping[str, Any]
    owner_root: str
    write_scope: tuple[str, ...]
    before_predicates: tuple[str, ...]
    after_predicates: tuple[str, ...]
    applicability_proofs: tuple[str, ...]
    preview: Mapping[str, Any]
    inverse: Mapping[str, str]
    validation_commands: tuple[tuple[str, ...], ...]

    @classmethod
    def from_mapping(cls, value: Any) -> OperatorDescriptor:
        if not isinstance(value, Mapping):
            raise RepairOperatorRegistryError("operator descriptor must be a mapping")
        required = {
            "operator_id",
            "kind",
            "input_schema",
            "owner_root",
            "write_scope",
            "before_predicates",
            "after_predicates",
            "applicability_proofs",
            "preview",
            "inverse",
            "validation_commands",
        }
        if set(value) != required:
            raise RepairOperatorRegistryError("operator descriptor fields are closed")
        kind = _identifier(value.get("kind"), "kind")
        if kind not in _OPERATOR_KINDS:
            raise RepairOperatorRegistryError("operator kind is not reviewed")
        return cls(
            operator_id=_identifier(value.get("operator_id"), "operator_id"),
            kind=kind,
            input_schema=_input_schema(value.get("input_schema")),
            owner_root=_identifier(value.get("owner_root"), "owner_root"),
            write_scope=tuple(
                sorted({_path(item, "write_scope") for item in value.get("write_scope", ())})
            ),
            before_predicates=_identifiers(value.get("before_predicates"), "before_predicates"),
            after_predicates=_identifiers(value.get("after_predicates"), "after_predicates"),
            applicability_proofs=_identifiers(
                value.get("applicability_proofs"), "applicability_proofs"
            ),
            preview=_preview(value.get("preview")),
            inverse=_inverse(value.get("inverse")),
            validation_commands=_commands(value.get("validation_commands")),
        )

    def __post_init__(self) -> None:
        if not self.write_scope:
            raise RepairOperatorRegistryError("write_scope must be non-empty")

    @property
    def descriptor_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "kind": self.kind,
            "input_schema": dict(self.input_schema),
            "owner_root": self.owner_root,
            "write_scope": list(self.write_scope),
            "before_predicates": list(self.before_predicates),
            "after_predicates": list(self.after_predicates),
            "applicability_proofs": list(self.applicability_proofs),
            "preview": dict(self.preview),
            "inverse": dict(self.inverse),
            "validation_commands": [list(command) for command in self.validation_commands],
        }

    def preview_input(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Return deterministic metadata only; it never renders or executes an edit."""
        if not isinstance(value, Mapping) or set(value) != set(self.input_schema["required"]):
            raise RepairOperatorRegistryError("preview input does not exactly match schema")
        for field, kind in self.input_schema["properties"].items():
            candidate = value[field]
            validator = _path if kind == "path" else _identifier
            validator(candidate, f"preview input {field}")
        return {
            "operator_id": self.operator_id,
            "descriptor_id": self.descriptor_id,
            "owner_root": self.owner_root,
            "write_scope": list(self.write_scope),
            "input_cid": content_identity(dict(sorted(value.items()))),
            "activation_status": "integration_pending_dcr035",
            "model_call_count": 0,
            "execution_authorized": False,
        }


class OperatorRegistry:
    """Manifest-pinned enumeration of finite reviewed descriptors only."""

    def __init__(
        self,
        descriptors: Sequence[OperatorDescriptor | Mapping[str, Any]],
        *,
        reviewed_manifest: Mapping[str, str],
    ) -> None:
        if not isinstance(reviewed_manifest, Mapping) or not reviewed_manifest:
            raise RepairOperatorRegistryError("reviewed_manifest is required")
        manifest = {
            _identifier(operator_id, "manifest operator_id"): _identifier(
                descriptor_id, "manifest descriptor_id"
            )
            for operator_id, descriptor_id in reviewed_manifest.items()
        }
        parsed = tuple(
            OperatorDescriptor.from_mapping(
                descriptor.to_dict() if isinstance(descriptor, OperatorDescriptor) else descriptor
            )
            for descriptor in descriptors
        )
        if not parsed or len({item.operator_id for item in parsed}) != len(parsed):
            raise RepairOperatorRegistryError("descriptors must be non-empty and unique")
        actual = {item.operator_id: item.descriptor_id for item in parsed}
        if actual != manifest:
            raise RepairOperatorRegistryError("descriptors do not exactly match reviewed manifest")
        self._descriptors = tuple(sorted(parsed, key=lambda item: item.operator_id))
        self._manifest = dict(sorted(manifest.items()))

    def enumerate(self) -> tuple[OperatorDescriptor, ...]:
        return self._descriptors

    def report(self) -> dict[str, Any]:
        payload = {
            "schema": OPERATOR_REGISTRY_SCHEMA,
            "interface": OPERATOR_REGISTRY_INTERFACE,
            "activation_status": "integration_pending_dcr035",
            "authoritative": False,
            "execution_authorized": False,
            "model_call_count": 0,
            "reviewed_manifest": dict(self._manifest),
            "operators": [
                {**descriptor.to_dict(), "descriptor_id": descriptor.descriptor_id}
                for descriptor in self._descriptors
            ],
        }
        return {**payload, "registry_cid": content_identity(payload)}


__all__ = [
    "OPERATOR_REGISTRY_INTERFACE",
    "OPERATOR_REGISTRY_SCHEMA",
    "OperatorDescriptor",
    "OperatorRegistry",
    "RepairOperatorRegistryError",
]
