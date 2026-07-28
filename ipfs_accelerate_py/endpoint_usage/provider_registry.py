"""Registered provider families for endpoint-usage observation adapters.

This registry is pure and offline: it maps stable family identifiers to
descriptor metadata used by :mod:`ipfs_accelerate_py.endpoint_usage.adapters`.
No network, process, credential, or model side effects occur on import or use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schema import ProtocolKind

PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID = (
    "requirement:provider-usage-adapter.v1"
)
ADAPTER_PARSER_VERSION = "1.0"
ADAPTER_REGISTRY_VERSION = "1.0"

MAX_ADAPTER_ALIASES = 32
MAX_CUSTOM_ADAPTERS = 64
MAX_DESCRIPTOR_STRING = 128
MAX_DESCRIPTION_BYTES = 256

_NAME = __import__("re").compile(r"^[a-z][a-z0-9._-]{0,63}$")


class AdapterFamily(str, Enum):
    """High-level observation shape families."""

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CLI = "cli"
    LOCAL = "local"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class AdapterError(ValueError):
    """Raised when adapter registration or resolution fails closed."""


@dataclass(frozen=True)
class ProviderAdapterDescriptor:
    """Bounded, secret-free descriptor for one registered adapter family."""

    family: AdapterFamily
    adapter_id: str
    aliases: Tuple[str, ...] = ()
    protocols: Tuple[ProtocolKind, ...] = ()
    description: str = ""
    supports_headers: bool = False
    supports_body_usage: bool = False
    supports_error_body: bool = False
    supports_cli_metadata: bool = False
    supports_local_capacity: bool = False
    default_window_ms: Optional[int] = 60_000

    def __post_init__(self) -> None:
        family = self.family
        if isinstance(family, str):
            try:
                family = AdapterFamily(family)
            except ValueError as exc:
                raise AdapterError("unknown adapter family") from exc
        object.__setattr__(self, "family", family)
        adapter_id = _require_name(self.adapter_id, "adapter_id")
        object.__setattr__(self, "adapter_id", adapter_id)
        aliases = _normalize_aliases(self.aliases)
        object.__setattr__(self, "aliases", aliases)
        protocols = _normalize_protocols(self.protocols)
        object.__setattr__(self, "protocols", protocols)
        description = self.description or ""
        if not isinstance(description, str):
            raise AdapterError("description must be a string")
        if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
            raise AdapterError("description exceeds bound")
        object.__setattr__(self, "description", description)
        for flag in (
            "supports_headers",
            "supports_body_usage",
            "supports_error_body",
            "supports_cli_metadata",
            "supports_local_capacity",
        ):
            value = getattr(self, flag)
            if not isinstance(value, bool):
                raise AdapterError("%s must be a boolean" % flag)
        window = self.default_window_ms
        if window is not None:
            if isinstance(window, bool) or not isinstance(window, int) or window < 0:
                raise AdapterError("default_window_ms must be a non-negative int")
        object.__setattr__(self, "default_window_ms", window)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family.value,
            "adapter_id": self.adapter_id,
            "aliases": list(self.aliases),
            "protocols": [item.value for item in self.protocols],
            "description": self.description,
            "supports_headers": self.supports_headers,
            "supports_body_usage": self.supports_body_usage,
            "supports_error_body": self.supports_error_body,
            "supports_cli_metadata": self.supports_cli_metadata,
            "supports_local_capacity": self.supports_local_capacity,
            "default_window_ms": self.default_window_ms,
        }


def _require_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterError("%s must be a non-empty string" % field)
    text = value.casefold().strip()
    if not _NAME.fullmatch(text):
        raise AdapterError("%s is not a canonical adapter name" % field)
    if len(text.encode("utf-8")) > MAX_DESCRIPTOR_STRING:
        raise AdapterError("%s exceeds bound" % field)
    return text


def _normalize_aliases(values: Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        raise AdapterError("aliases must be a sequence")
    if len(values) > MAX_ADAPTER_ALIASES:
        raise AdapterError("aliases exceeds maximum count")
    out = []
    seen = set()
    for item in values:
        name = _require_name(item, "alias")
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(sorted(out))


def _normalize_protocols(values: Any) -> Tuple[ProtocolKind, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        raise AdapterError("protocols must be a sequence")
    out = []
    seen = set()
    for item in values:
        if isinstance(item, ProtocolKind):
            protocol = item
        else:
            try:
                protocol = ProtocolKind(str(item).casefold())
            except ValueError as exc:
                raise AdapterError("unknown protocol kind") from exc
        if protocol.value in seen:
            continue
        seen.add(protocol.value)
        out.append(protocol)
    return tuple(sorted(out, key=lambda item: item.value))


def _builtin_descriptors() -> Tuple[ProviderAdapterDescriptor, ...]:
    return (
        ProviderAdapterDescriptor(
            family=AdapterFamily.OPENAI_COMPATIBLE,
            adapter_id="openai_compatible",
            aliases=(
                "openai",
                "xai",
                "openrouter",
                "vllm",
                "tgi",
                "together",
                "groq",
                "fireworks",
                "azure_openai",
                "compatible",
            ),
            protocols=(ProtocolKind.HTTP, ProtocolKind.HTTPS),
            description=(
                "OpenAI-compatible HTTP usage bodies and x-ratelimit headers"
            ),
            supports_headers=True,
            supports_body_usage=True,
            supports_error_body=True,
            default_window_ms=60_000,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.ANTHROPIC,
            adapter_id="anthropic",
            aliases=("claude", "anthropic_messages"),
            protocols=(ProtocolKind.HTTP, ProtocolKind.HTTPS),
            description="Anthropic-style usage and anthropic-ratelimit headers",
            supports_headers=True,
            supports_body_usage=True,
            supports_error_body=True,
            default_window_ms=60_000,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.HUGGINGFACE,
            adapter_id="huggingface",
            aliases=("hf", "hf_inference", "tei", "hf_tgi", "huggingface_hub"),
            protocols=(ProtocolKind.HTTP, ProtocolKind.HTTPS),
            description="Hugging Face Inference/TEI/TGI usage and error shapes",
            supports_headers=True,
            supports_body_usage=True,
            supports_error_body=True,
            default_window_ms=60_000,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.CLI,
            adapter_id="cli",
            aliases=(
                "codex",
                "copilot",
                "grok",
                "gemini",
                "goose",
                "mistral",
                "mistral_vibe",
                "claude_code",
            ),
            protocols=(ProtocolKind.CLI,),
            description="Structured CLI usage/reset metadata without raw payload retention",
            supports_cli_metadata=True,
            supports_error_body=True,
            default_window_ms=3_600_000,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.LOCAL,
            adapter_id="local",
            aliases=(
                "transformers",
                "llama_cpp",
                "llamacpp",
                "backend_manager",
                "local_runtime",
            ),
            protocols=(ProtocolKind.LOCAL, ProtocolKind.UNIX, ProtocolKind.HTTP),
            description="Local concurrency and memory capacity ceilings",
            supports_local_capacity=True,
            supports_body_usage=True,
            default_window_ms=None,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.CUSTOM,
            adapter_id="custom",
            aliases=("registered", "explicit"),
            protocols=tuple(ProtocolKind),
            description="Explicit custom adapter contract with registered field maps",
            supports_headers=True,
            supports_body_usage=True,
            supports_error_body=True,
            supports_cli_metadata=True,
            supports_local_capacity=True,
            default_window_ms=60_000,
        ),
        ProviderAdapterDescriptor(
            family=AdapterFamily.UNKNOWN,
            adapter_id="unknown",
            aliases=("generic", "fallback"),
            protocols=tuple(ProtocolKind),
            description="Fail-closed fallback that only retains restrictive cooldowns",
            supports_headers=True,
            supports_error_body=True,
            default_window_ms=60_000,
        ),
    )


_BUILTIN: Tuple[ProviderAdapterDescriptor, ...] = _builtin_descriptors()
_BY_ID: Dict[str, ProviderAdapterDescriptor] = {
    item.adapter_id: item for item in _BUILTIN
}
_BY_ALIAS: Dict[str, str] = {}
for _descriptor in _BUILTIN:
    _BY_ALIAS[_descriptor.adapter_id] = _descriptor.adapter_id
    for _alias in _descriptor.aliases:
        _BY_ALIAS[_alias] = _descriptor.adapter_id

_CUSTOM: Dict[str, ProviderAdapterDescriptor] = {}


def list_adapter_descriptors() -> Tuple[ProviderAdapterDescriptor, ...]:
    """Return built-in and custom descriptors in stable order."""

    custom = tuple(
        sorted(_CUSTOM.values(), key=lambda item: item.adapter_id)
    )
    return _BUILTIN + custom


def get_adapter_descriptor(adapter_id: str) -> ProviderAdapterDescriptor:
    """Resolve a descriptor by adapter id or alias."""

    key = _require_name(adapter_id, "adapter_id")
    if key in _CUSTOM:
        return _CUSTOM[key]
    resolved = _BY_ALIAS.get(key)
    if resolved is None:
        raise AdapterError("adapter is not registered: %s" % key)
    return _BY_ID[resolved]


def resolve_adapter_family(
    value: Any,
    *,
    protocol: Optional[Any] = None,
    default: AdapterFamily = AdapterFamily.UNKNOWN,
) -> AdapterFamily:
    """Map a provider name, adapter id, or alias to an :class:`AdapterFamily`.

    Unknown values resolve to ``default`` (normally :attr:`AdapterFamily.UNKNOWN`)
    rather than guessing a privileged parser.
    """

    if value is None or value == "":
        if protocol is not None:
            try:
                protocol_kind = (
                    protocol
                    if isinstance(protocol, ProtocolKind)
                    else ProtocolKind(str(protocol).casefold())
                )
            except ValueError:
                return default
            if protocol_kind is ProtocolKind.CLI:
                return AdapterFamily.CLI
            if protocol_kind is ProtocolKind.LOCAL:
                return AdapterFamily.LOCAL
        return default
    if isinstance(value, AdapterFamily):
        return value
    try:
        text = _require_name(str(value), "family")
    except AdapterError:
        return default
    if text in _CUSTOM:
        return AdapterFamily.CUSTOM
    resolved = _BY_ALIAS.get(text)
    if resolved is None:
        # Accept direct family enum values that are not adapter ids.
        try:
            return AdapterFamily(text)
        except ValueError:
            return default
    return _BY_ID[resolved].family


def register_custom_adapter(
    descriptor: ProviderAdapterDescriptor | Mapping[str, Any],
) -> ProviderAdapterDescriptor:
    """Register an explicit custom adapter descriptor for this process.

    Built-in ids cannot be overwritten. Custom registration is in-memory only
    and is intended for tests or explicit operator configuration, never remote
    discovery.
    """

    if len(_CUSTOM) >= MAX_CUSTOM_ADAPTERS and not (
        isinstance(descriptor, ProviderAdapterDescriptor)
        and descriptor.adapter_id in _CUSTOM
        or isinstance(descriptor, Mapping)
        and str(descriptor.get("adapter_id", "")).casefold() in _CUSTOM
    ):
        # Allow in-place replace of an existing custom id even at capacity.
        existing_id = None
        if isinstance(descriptor, ProviderAdapterDescriptor):
            existing_id = descriptor.adapter_id
        elif isinstance(descriptor, Mapping):
            existing_id = str(descriptor.get("adapter_id", "")).casefold()
        if existing_id not in _CUSTOM:
            raise AdapterError("custom adapter registry is full")

    if isinstance(descriptor, Mapping):
        descriptor = ProviderAdapterDescriptor(
            family=descriptor.get("family", AdapterFamily.CUSTOM),
            adapter_id=descriptor["adapter_id"],
            aliases=tuple(descriptor.get("aliases") or ()),
            protocols=tuple(descriptor.get("protocols") or ()),
            description=str(descriptor.get("description") or ""),
            supports_headers=bool(descriptor.get("supports_headers", True)),
            supports_body_usage=bool(descriptor.get("supports_body_usage", True)),
            supports_error_body=bool(descriptor.get("supports_error_body", True)),
            supports_cli_metadata=bool(
                descriptor.get("supports_cli_metadata", False)
            ),
            supports_local_capacity=bool(
                descriptor.get("supports_local_capacity", False)
            ),
            default_window_ms=descriptor.get("default_window_ms", 60_000),
        )
    if not isinstance(descriptor, ProviderAdapterDescriptor):
        raise AdapterError("descriptor must be ProviderAdapterDescriptor or mapping")
    if descriptor.adapter_id in _BY_ID:
        raise AdapterError("cannot overwrite a built-in adapter id")
    if descriptor.family is not AdapterFamily.CUSTOM:
        # Custom registrations stay in the custom family to avoid shadowing
        # privileged parsers via untrusted remote names.
        raise AdapterError("custom registrations must use family=custom")
    for alias in descriptor.aliases:
        owner = _BY_ALIAS.get(alias)
        if owner is not None and owner != descriptor.adapter_id:
            raise AdapterError("alias collides with registered adapter: %s" % alias)
        if alias in _CUSTOM and alias != descriptor.adapter_id:
            raise AdapterError("alias collides with custom adapter: %s" % alias)
    _CUSTOM[descriptor.adapter_id] = descriptor
    return descriptor


def unregister_custom_adapter(adapter_id: str) -> bool:
    """Remove a process-local custom adapter. Built-ins cannot be removed."""

    key = _require_name(adapter_id, "adapter_id")
    return _CUSTOM.pop(key, None) is not None


def clear_custom_adapters() -> None:
    """Drop all process-local custom adapters (tests / process reset)."""

    _CUSTOM.clear()


def adapter_capabilities(adapter_id: str) -> Mapping[str, bool]:
    """Return bounded capability flags for discovery endpoints."""

    descriptor = get_adapter_descriptor(adapter_id)
    return {
        "supports_headers": descriptor.supports_headers,
        "supports_body_usage": descriptor.supports_body_usage,
        "supports_error_body": descriptor.supports_error_body,
        "supports_cli_metadata": descriptor.supports_cli_metadata,
        "supports_local_capacity": descriptor.supports_local_capacity,
    }


def known_adapter_ids() -> Tuple[str, ...]:
    """Stable list of built-in adapter ids plus custom ids."""

    return tuple(item.adapter_id for item in list_adapter_descriptors())


def is_registered_adapter(value: Any) -> bool:
    """Return whether *value* names a built-in or custom adapter/alias."""

    if value is None:
        return False
    try:
        key = _require_name(str(value), "adapter_id")
    except AdapterError:
        return False
    return key in _CUSTOM or key in _BY_ALIAS


__all__ = [
    "ADAPTER_PARSER_VERSION",
    "ADAPTER_REGISTRY_VERSION",
    "AdapterError",
    "AdapterFamily",
    "PROVIDER_USAGE_ADAPTER_REQUIREMENT_ID",
    "ProviderAdapterDescriptor",
    "adapter_capabilities",
    "clear_custom_adapters",
    "get_adapter_descriptor",
    "is_registered_adapter",
    "known_adapter_ids",
    "list_adapter_descriptors",
    "register_custom_adapter",
    "resolve_adapter_family",
    "unregister_custom_adapter",
]
