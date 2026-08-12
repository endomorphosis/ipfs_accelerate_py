"""Deployment-owned 211-AI pilot ActionDescriptor catalog.

This module pins reviewed logical actions for the voice-action pilot. Domain
packs and Abby content may *reference* these descriptor IDs and logical action
names; they cannot widen the catalog or embed executable locators.

The JSON export under ``data/voice_action_dag/catalog/211ai-pilot-v1.json`` is
the durable, content-addressable snapshot of this catalog. It intentionally
contains no command/argv/executable/url/import_path fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .catalog import ActionCatalog, ActionDescriptor
from .contracts import RiskClass, SideEffectClass, content_digest
from .surface_exposure import SURFACE_EXPOSURE_CLASS, VOICE_CLIENT_SURFACE_ACTIONS

CATALOG_ID: str = "211ai-pilot-v1"
CATALOG_VERSION: str = "1.1"
CATALOG_SCHEMA: str = "voice-action/catalog@1"
POLICY_REVISION: str = "pilot-v1.1-surface-coverage"

# Stable pilot logical actions (board namespace voice-action-dag-abby-v1).
PILOT_LOGICAL_ACTIONS: tuple[str, ...] = (
    "handoff_live_agent",
    "open_app_surface",
    "open_wallet_documents",
    "read_calendar",
    "create_calendar_reminder",
    "read_provider_messages",
    "leave_provider_message",
    "open_service_detail",
    "schedule_service_callback",
    "escalate_safety",
)

# Keys that must never appear in the public catalog export (or descriptor metadata).
FORBIDDEN_LOCATOR_KEYS: frozenset[str] = frozenset(
    {
        "command",
        "argv",
        "executable",
        "cwd",
        "env",
        "shell",
        "import_path",
        "url",
        "webhook",
        "host",
        "port",
        "binary",
        "module",
        "entrypoint",
    }
)

_DEFAULT_CHANNELS: tuple[str, ...] = ("voice", "chat", "test")
_DEFAULT_TENANTS: tuple[str, ...] = ("*",)


def _descriptor_id(adapter: str, logical_action: str) -> str:
    return f"voice.{adapter}.{logical_action}.v1"


def _meta(**pairs: str) -> Mapping[str, str]:
    """Build string metadata and reject locator smuggling."""

    for key, value in pairs.items():
        lowered = key.lower()
        if lowered in FORBIDDEN_LOCATOR_KEYS or lowered.endswith("_path"):
            raise ValueError(f"descriptor metadata key {key!r} is not allowed")
        if not isinstance(value, str):
            raise TypeError(f"metadata value for {key!r} must be str")
    return dict(pairs)


def pilot_descriptors() -> tuple[ActionDescriptor, ...]:
    """Return the reviewed pilot descriptor set in a stable order."""

    return (
        ActionDescriptor(
            descriptor_id=_descriptor_id("human", "handoff_live_agent"),
            logical_action="handoff_live_agent",
            adapter="human",
            risk_class=RiskClass.HUMAN,
            side_effect_class=SideEffectClass.NETWORK,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="handoff",
                auth_required="false",
                confirmation_mode="explicit_or_policy",
                truthfulness="never_claim_transfer_without_receipt",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "open_app_surface"),
            logical_action="open_app_surface",
            adapter="python",
            risk_class=RiskClass.READ,
            side_effect_class=SideEffectClass.LOCAL_READ,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="app_surface",
                auth_required="false",
                confirmation_mode="explicit",
                # Authority plane enforces exposure classes; content never embeds allowlists.
                surface_gate="voice_navigable_or_voice_actionable",
                surface_arg="surface_id",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "open_wallet_documents"),
            logical_action="open_wallet_documents",
            adapter="python",
            risk_class=RiskClass.READ,
            side_effect_class=SideEffectClass.LOCAL_READ,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="wallet_documents",
                auth_required="false",
                confirmation_mode="explicit",
                surface_gate="voice_actionable_uploads",
                default_surface_id="uploads",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "read_calendar"),
            logical_action="read_calendar",
            adapter="python",
            risk_class=RiskClass.READ,
            side_effect_class=SideEffectClass.LOCAL_READ,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="calendar",
                auth_required="false",
                confirmation_mode="explicit",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "create_calendar_reminder"),
            logical_action="create_calendar_reminder",
            adapter="python",
            risk_class=RiskClass.WRITE,
            side_effect_class=SideEffectClass.LOCAL_WRITE,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="calendar",
                auth_required="true",
                confirmation_mode="explicit_plus_auth",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "read_provider_messages"),
            logical_action="read_provider_messages",
            adapter="python",
            risk_class=RiskClass.READ,
            side_effect_class=SideEffectClass.LOCAL_READ,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="messaging",
                auth_required="true",
                confirmation_mode="explicit_plus_auth",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "leave_provider_message"),
            logical_action="leave_provider_message",
            adapter="python",
            risk_class=RiskClass.WRITE,
            side_effect_class=SideEffectClass.EXTERNAL_MUTATION,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="messaging",
                auth_required="true",
                confirmation_mode="explicit_plus_auth",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("python", "open_service_detail"),
            logical_action="open_service_detail",
            adapter="python",
            risk_class=RiskClass.READ,
            side_effect_class=SideEffectClass.LOCAL_READ,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="service",
                auth_required="false",
                confirmation_mode="explicit",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("workflow", "schedule_service_callback"),
            logical_action="schedule_service_callback",
            adapter="workflow",
            risk_class=RiskClass.WRITE,
            side_effect_class=SideEffectClass.EXTERNAL_MUTATION,
            requires_confirmation=True,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="service",
                auth_required="true",
                confirmation_mode="explicit_plus_auth",
            ),
        ),
        ActionDescriptor(
            descriptor_id=_descriptor_id("human", "escalate_safety"),
            logical_action="escalate_safety",
            adapter="human",
            risk_class=RiskClass.HUMAN,
            side_effect_class=SideEffectClass.NETWORK,
            # Policy-driven safety overlay may auto-admit; never smuggles tools.
            requires_confirmation=False,
            allowed_channels=_DEFAULT_CHANNELS,
            allowed_tenants=_DEFAULT_TENANTS,
            metadata=_meta(
                family="safety",
                auth_required="false",
                confirmation_mode="policy_driven",
                truthfulness="never_claim_transfer_without_receipt",
            ),
        ),
    )


def logical_action_to_descriptor_id() -> Mapping[str, str]:
    """Map pilot logical actions to their reviewed descriptor IDs."""

    return {
        descriptor.logical_action: descriptor.descriptor_id
        for descriptor in pilot_descriptors()
    }


def build_pilot_catalog() -> ActionCatalog:
    """Construct an in-memory catalog of the pilot descriptors."""

    return ActionCatalog(list(pilot_descriptors()))


def descriptor_to_public_dict(
    descriptor: ActionDescriptor,
    *,
    include_digest: bool = True,
) -> dict[str, Any]:
    """Serialize a descriptor without executable locators."""

    payload: dict[str, Any] = {
        "adapter": descriptor.adapter,
        "allowed_channels": list(descriptor.allowed_channels),
        "allowed_tenants": list(descriptor.allowed_tenants),
        "descriptor_id": descriptor.descriptor_id,
        "logical_action": descriptor.logical_action,
        "metadata": {key: descriptor.metadata[key] for key in sorted(descriptor.metadata)},
        "requires_confirmation": descriptor.requires_confirmation,
        "risk_class": descriptor.risk_class.value,
        "side_effect_class": descriptor.side_effect_class.value,
    }
    if include_digest:
        payload["descriptor_digest"] = descriptor.digest
    assert_no_executable_locators(payload)
    return payload


def descriptor_from_public_dict(payload: Mapping[str, Any]) -> ActionDescriptor:
    """Rehydrate a descriptor from the public JSON shape."""

    assert_no_executable_locators(payload)
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("metadata must be string-to-string")
        lowered = key.lower()
        if lowered in FORBIDDEN_LOCATOR_KEYS or lowered.endswith("_path"):
            raise ValueError(f"descriptor metadata key {key!r} is not allowed")

    descriptor = ActionDescriptor(
        descriptor_id=str(payload["descriptor_id"]),
        logical_action=str(payload["logical_action"]),
        adapter=str(payload["adapter"]),
        risk_class=RiskClass(str(payload["risk_class"])),
        side_effect_class=SideEffectClass(str(payload["side_effect_class"])),
        requires_confirmation=bool(payload["requires_confirmation"]),
        allowed_channels=tuple(str(c) for c in payload.get("allowed_channels") or ()),
        allowed_tenants=tuple(str(t) for t in payload.get("allowed_tenants") or ()),
        metadata={str(k): str(v) for k, v in metadata.items()},
    )
    expected_digest = payload.get("descriptor_digest")
    if (
        expected_digest is not None
        and expected_digest != descriptor.digest
        and str(expected_digest) not in {"", "PENDING"}
    ):
        raise ValueError(
            f"descriptor_digest mismatch for {descriptor.descriptor_id!r}: "
            f"expected {expected_digest}, got {descriptor.digest}"
        )
    return descriptor


def catalog_digest(descriptors: tuple[ActionDescriptor, ...] | None = None) -> str:
    """Return a stable content digest for the pilot catalog.

    Descriptor order is normalized by ``descriptor_id`` so key reordering in
    intermediate maps cannot change the digest.
    """

    rows = list(descriptors if descriptors is not None else pilot_descriptors())
    rows.sort(key=lambda d: d.descriptor_id)
    return content_digest(
        {
            "catalog_id": CATALOG_ID,
            "version": CATALOG_VERSION,
            "schema": CATALOG_SCHEMA,
            "policy_revision": POLICY_REVISION,
            "descriptors": [
                {
                    "descriptor_id": d.descriptor_id,
                    "descriptor_digest": d.digest,
                    "logical_action": d.logical_action,
                }
                for d in rows
            ],
        }
    )


def export_surface_action_matrix() -> dict[str, Any]:
    """Export surface → exposure_class → logical actions (content-safe)."""

    surfaces: list[dict[str, Any]] = []
    for surface_id in sorted(SURFACE_EXPOSURE_CLASS):
        klass = SURFACE_EXPOSURE_CLASS[surface_id]
        actions = list(VOICE_CLIENT_SURFACE_ACTIONS.get(surface_id, ()))
        surfaces.append(
            {
                "surface_id": surface_id,
                "exposure_class": klass,
                "logical_actions": actions,
                "client_voice_open": klass in {"voice_navigable", "voice_actionable"},
            }
        )
    payload = {
        "schema": "voice-action/surface-action-matrix@1",
        "catalog_id": CATALOG_ID,
        "catalog_version": CATALOG_VERSION,
        "policy_revision": POLICY_REVISION,
        "surfaces": surfaces,
    }
    assert_no_executable_locators(payload)
    return payload


def export_pilot_catalog_dict(
    descriptors: tuple[ActionDescriptor, ...] | None = None,
    *,
    include_digests: bool = True,
) -> dict[str, Any]:
    """Export the pilot catalog as a JSON-serializable dict (no locators)."""

    rows = list(descriptors if descriptors is not None else pilot_descriptors())
    # Stable public order: sorted by descriptor_id for deterministic JSON.
    rows.sort(key=lambda d: d.descriptor_id)
    payload: dict[str, Any] = {
        "catalog_id": CATALOG_ID,
        "descriptors": [
            descriptor_to_public_dict(d, include_digest=include_digests) for d in rows
        ],
        "logical_actions": sorted(d.logical_action for d in rows),
        "policy_revision": POLICY_REVISION,
        "schema": CATALOG_SCHEMA,
        "surface_action_matrix": export_surface_action_matrix(),
        "version": CATALOG_VERSION,
    }
    if include_digests:
        payload["catalog_digest"] = catalog_digest(tuple(rows))
    assert_no_executable_locators(payload)
    return payload


def load_pilot_catalog_from_dict(payload: Mapping[str, Any]) -> ActionCatalog:
    """Load and validate a catalog export; unknown structure fails closed."""

    if not isinstance(payload, Mapping):
        raise TypeError("catalog payload must be a mapping")
    assert_no_executable_locators(payload)

    catalog_id = payload.get("catalog_id")
    if catalog_id != CATALOG_ID:
        raise ValueError(f"unexpected catalog_id {catalog_id!r}")

    raw_descriptors = payload.get("descriptors")
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        raise ValueError("catalog descriptors must be a non-empty list")

    descriptors = [descriptor_from_public_dict(row) for row in raw_descriptors]
    expected = payload.get("catalog_digest")
    computed = catalog_digest(tuple(descriptors))
    if (
        expected is not None
        and expected != computed
        and str(expected) not in {"", "PENDING"}
    ):
        raise ValueError(
            f"catalog_digest mismatch: expected {expected}, got {computed}"
        )
    return ActionCatalog(descriptors)


def assert_no_executable_locators(value: object, *, _path: str = "$") -> None:
    """Fail closed if any forbidden executable locator key appears."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in FORBIDDEN_LOCATOR_KEYS or lowered.endswith("_path"):
                raise ValueError(
                    f"forbidden executable locator key {key_text!r} at {_path}"
                )
            assert_no_executable_locators(child, _path=f"{_path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_executable_locators(child, _path=f"{_path}[{index}]")


def default_catalog_json_path(repository_root: Path | None = None) -> Path:
    """Resolve the checked-in pilot catalog JSON path."""

    if repository_root is None:
        # action_runtime/ -> package -> ipfs_accelerate_py/ -> monorepo root
        repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "data" / "voice_action_dag" / "catalog" / "211ai-pilot-v1.json"


def load_pilot_catalog_json(path: Path | None = None) -> ActionCatalog:
    """Load the durable JSON snapshot and return an ActionCatalog."""

    target = path or default_catalog_json_path()
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return load_pilot_catalog_from_dict(payload)


def render_pilot_catalog_json(
    descriptors: tuple[ActionDescriptor, ...] | None = None,
    *,
    include_digests: bool = True,
) -> str:
    """Render the pilot catalog as canonical pretty JSON (trailing newline)."""

    payload = export_pilot_catalog_dict(descriptors, include_digests=include_digests)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def write_pilot_catalog_json(
    path: Path | None = None,
    *,
    include_digests: bool = False,
) -> Path:
    """Write the durable pilot catalog JSON snapshot atomically-ish.

    Digests are computed in-process from descriptor fields; the checked-in
    durable snapshot omits them so the file stays free of dual-sourced hashes.
    """

    target = path or default_catalog_json_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_pilot_catalog_json(include_digests=include_digests)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


__all__ = [
    "CATALOG_ID",
    "CATALOG_SCHEMA",
    "export_surface_action_matrix",
    "CATALOG_VERSION",
    "FORBIDDEN_LOCATOR_KEYS",
    "PILOT_LOGICAL_ACTIONS",
    "POLICY_REVISION",
    "assert_no_executable_locators",
    "build_pilot_catalog",
    "catalog_digest",
    "default_catalog_json_path",
    "descriptor_from_public_dict",
    "descriptor_to_public_dict",
    "export_pilot_catalog_dict",
    "load_pilot_catalog_from_dict",
    "load_pilot_catalog_json",
    "logical_action_to_descriptor_id",
    "pilot_descriptors",
    "render_pilot_catalog_json",
    "write_pilot_catalog_json",
]
