"""Deployment-owned allowlisted action catalog.

Domain packs and voice response libraries may *reference* descriptor IDs.
They cannot widen the catalog or supply executable paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .contracts import RiskClass, SideEffectClass, content_digest


@dataclass(frozen=True)
class ActionDescriptor:
    """Operator-reviewed binding from a logical action to an adapter interface."""

    descriptor_id: str
    logical_action: str
    adapter: str
    risk_class: RiskClass = RiskClass.READ
    side_effect_class: SideEffectClass = SideEffectClass.NONE
    requires_confirmation: bool = True
    allowed_channels: tuple[str, ...] = ("voice", "chat", "test")
    allowed_tenants: tuple[str, ...] = ("*",)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.descriptor_id or not self.logical_action or not self.adapter:
            raise ValueError("descriptor_id, logical_action, and adapter are required")
        if self.adapter not in {"cli", "mcp", "python", "workflow", "supervisor", "human"}:
            raise ValueError(f"unsupported adapter {self.adapter!r}")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "descriptor_id": self.descriptor_id,
                "logical_action": self.logical_action,
                "adapter": self.adapter,
                "risk_class": self.risk_class.value,
                "side_effect_class": self.side_effect_class.value,
                "requires_confirmation": self.requires_confirmation,
                "allowed_channels": list(self.allowed_channels),
                "allowed_tenants": list(self.allowed_tenants),
                "metadata": dict(self.metadata),
            }
        )


class ActionCatalog:
    """In-memory, fail-closed catalog of reviewed descriptors."""

    def __init__(self, descriptors: list[ActionDescriptor] | None = None) -> None:
        self._by_id: dict[str, ActionDescriptor] = {}
        for descriptor in descriptors or ():
            self.register(descriptor)

    def register(self, descriptor: ActionDescriptor) -> None:
        if descriptor.descriptor_id in self._by_id:
            raise ValueError(f"duplicate descriptor_id {descriptor.descriptor_id!r}")
        self._by_id[descriptor.descriptor_id] = descriptor

    def get(self, descriptor_id: str) -> ActionDescriptor | None:
        return self._by_id.get(descriptor_id)

    def require(self, descriptor_id: str) -> ActionDescriptor:
        descriptor = self.get(descriptor_id)
        if descriptor is None:
            raise KeyError(f"unknown descriptor_id {descriptor_id!r}")
        return descriptor

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))


def resolve_reviewed_executable(path: str | Path) -> Path:
    """Resolve and pin an absolute executable identity.

    Relative paths, non-files, and non-executable targets are rejected so that
    domain packs can never supply a runnable path at proposal time.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("executable path must be absolute and operator-reviewed")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"executable is not a file: {resolved}")
    if not os_access_executable(resolved):
        raise ValueError(f"executable is not runnable: {resolved}")
    return resolved


def os_access_executable(path: Path) -> bool:
    import os

    return os.access(path, os.X_OK)
