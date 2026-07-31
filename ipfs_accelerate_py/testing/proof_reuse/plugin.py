"""Cold-import-safe pytest shell for proof-backed test reuse.

PTR-050 intentionally performs no candidate lookup, skip, receipt write,
capability probe, network access, or daemon startup.  Later plugin layers can
consume the immutable configuration and per-item metadata exposed here.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .config import PROOF_REUSE_MODES, ProofReuseConfig, ProofReuseMode

PLUGIN_NAME = "ipfs-proof-reuse"
CONFIG_ATTRIBUTE = "_ipfs_proof_reuse_config"
ITEM_METADATA_ATTRIBUTE = "_ipfs_proof_reuse_metadata"

MODE_OPTION = "--proof-reuse-mode"
REQUIRED_AUDIT_OPTION = "--proof-reuse-required-audit"
MODE_INI = "proof_reuse_mode"
REQUIRED_AUDIT_INI = "proof_reuse_required_audit"

DISABLED_MARKER = "proof_reuse_disabled"
EFFECTS_MARKER = "proof_reuse_effects"

_MARKER_DESCRIPTIONS = (
    (
        DISABLED_MARKER,
        "proof_reuse_disabled(reason=None): always execute this test; no "
        "proof-backed reuse lookup or write is permitted",
    ),
    (
        EFFECTS_MARKER,
        "proof_reuse_effects(*adapters): declare reviewed effect adapter names "
        "for proof-reuse dependency tracing",
    ),
)


@dataclass(frozen=True)
class ProofReuseItemMetadata:
    """Collection facts derived directly from one pytest item."""

    nodeid: str
    disabled: bool = False
    disabled_reason: str = ""
    effect_adapters: Tuple[str, ...] = ()


def pytest_addoption(parser: Any) -> None:
    """Register the cold shell's CLI and ini configuration."""

    group = parser.getgroup(
        "proof-reuse",
        "proof-backed reuse of exact pytest pass evidence",
    )
    group.addoption(
        MODE_OPTION,
        action="store",
        dest="proof_reuse_mode",
        choices=PROOF_REUSE_MODES,
        default=None,
        metavar="MODE",
        help=(
            "proof reuse mode: off, shadow, read, write, or readwrite "
            "(default: IPFS_TEST_PROOF_REUSE_MODE or off)"
        ),
    )
    group.addoption(
        REQUIRED_AUDIT_OPTION,
        action="store_true",
        dest="proof_reuse_required_audit",
        default=False,
        help=(
            "enable the separate CI required-audit policy; this is not a "
            "proof reuse mode"
        ),
    )
    parser.addini(
        MODE_INI,
        "proof reuse mode (off, shadow, read, write, or readwrite)",
        default="",
    )
    parser.addini(
        REQUIRED_AUDIT_INI,
        "enable the separate proof reuse required-audit CI policy",
        type="bool",
        default=False,
    )


def _getoption(config: Any, name: str, default: Any = None) -> Any:
    try:
        return config.getoption(name, default=default)
    except (AttributeError, TypeError, ValueError):
        return default


def _getini(config: Any, name: str, default: Any = None) -> Any:
    try:
        return config.getini(name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def get_proof_reuse_config(config: Any) -> ProofReuseConfig:
    """Return the resolved config, safely defaulting to off if unconfigured."""

    existing = getattr(config, CONFIG_ATTRIBUTE, None)
    if isinstance(existing, ProofReuseConfig):
        return existing
    return ProofReuseConfig()


def pytest_configure(config: Any) -> None:
    """Resolve pure configuration and register marker documentation."""

    for _marker_name, description in _MARKER_DESCRIPTIONS:
        config.addinivalue_line("markers", description)

    resolved = ProofReuseConfig.resolve(
        command_line_mode=_getoption(config, "proof_reuse_mode"),
        ini_mode=_getini(config, MODE_INI, ""),
        environ=os.environ,
        command_line_required_audit=_getoption(
            config,
            "proof_reuse_required_audit",
            False,
        ),
        ini_required_audit=_getini(config, REQUIRED_AUDIT_INI, False),
    )
    setattr(config, CONFIG_ATTRIBUTE, resolved)


def _bounded_marker_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip()[:512]


def _marker_reason(marker: Any) -> str:
    if marker is None:
        return ""
    reason = getattr(marker, "kwargs", {}).get("reason")
    if reason is None:
        args = getattr(marker, "args", ())
        reason = args[0] if args else ""
    return _bounded_marker_text(reason)


def _effect_adapters(item: Any) -> Tuple[str, ...]:
    adapters = []
    seen = set()
    try:
        markers: Iterable[Any] = item.iter_markers(name=EFFECTS_MARKER)
    except (AttributeError, TypeError):
        marker = item.get_closest_marker(EFFECTS_MARKER)
        markers = () if marker is None else (marker,)
    for marker in markers:
        raw_values = list(getattr(marker, "args", ()))
        keyword_values = getattr(marker, "kwargs", {}).get("adapters", ())
        if isinstance(keyword_values, str):
            raw_values.append(keyword_values)
        else:
            try:
                raw_values.extend(keyword_values)
            except TypeError:
                # Malformed marker metadata must never disrupt collection.
                pass
        for raw_value in raw_values:
            adapter = _bounded_marker_text(raw_value)
            if adapter and adapter not in seen:
                seen.add(adapter)
                adapters.append(adapter)
    return tuple(adapters)


def collect_item_metadata(item: Any) -> ProofReuseItemMetadata:
    """Build metadata from a direct collected node, without a path registry."""

    disabled_marker = item.get_closest_marker(DISABLED_MARKER)
    return ProofReuseItemMetadata(
        nodeid=str(getattr(item, "nodeid", ""))[:2048],
        disabled=disabled_marker is not None,
        disabled_reason=_marker_reason(disabled_marker),
        effect_adapters=_effect_adapters(item),
    )


def get_item_metadata(item: Any) -> Optional[ProofReuseItemMetadata]:
    metadata = getattr(item, ITEM_METADATA_ATTRIBUTE, None)
    if isinstance(metadata, ProofReuseItemMetadata):
        return metadata
    return None


def pytest_collection_modifyitems(config: Any, items: Iterable[Any]) -> None:
    """Attach shell metadata when enabled; off mode is strictly inert."""

    proof_config = get_proof_reuse_config(config)
    if proof_config.mode is ProofReuseMode.OFF:
        return
    for item in items:
        setattr(item, ITEM_METADATA_ATTRIBUTE, collect_item_metadata(item))


__all__ = [
    "CONFIG_ATTRIBUTE",
    "DISABLED_MARKER",
    "EFFECTS_MARKER",
    "ITEM_METADATA_ATTRIBUTE",
    "PLUGIN_NAME",
    "ProofReuseItemMetadata",
    "collect_item_metadata",
    "get_item_metadata",
    "get_proof_reuse_config",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
]
