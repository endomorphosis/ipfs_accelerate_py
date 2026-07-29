"""Lazy CLI provider registry with deterministic alias resolution.

Importing this module and listing providers never loads optional provider
implementations, never installs tools, and never starts processes. Factories
are retained and invoked only when a caller explicitly resolves a provider
instance.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .contracts import (
    CapabilitySupport,
    CLICapabilities,
    LLMProvider,
    ProviderFactory,
    ProviderSpec,
    _normalize_identifier,
)
from .errors import (
    ContractValidationError,
    ProviderLoadError,
    ProviderNotFoundError,
    RegistryCollisionError,
)


@dataclass(frozen=True)
class RegistryEntry:
    """Internal registry row: metadata plus optional lazy factory."""

    spec: ProviderSpec
    factory: Optional[ProviderFactory] = None


class CLIProviderRegistry:
    """Thread-safe registry mapping canonical names and aliases to specs.

    Collision policy is fail-closed: registering a name or alias that already
    maps to a different canonical provider raises :class:`RegistryCollisionError`
    unless ``replace=True``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, RegistryEntry] = {}
        self._index: dict[str, str] = {}

    def clear(self) -> None:
        """Remove all registrations (intended for tests)."""
        with self._lock:
            self._entries.clear()
            self._index.clear()

    def register(
        self,
        name: str,
        *,
        factory: Optional[ProviderFactory] = None,
        aliases: Sequence[str] = (),
        description: str = "",
        capabilities: CLICapabilities | Mapping[str, Any] | None = None,
        spec: ProviderSpec | Mapping[str, Any] | None = None,
        replace: bool = False,
        **spec_fields: Any,
    ) -> ProviderSpec:
        """Register a provider by metadata and optional factory.

        The factory is stored but not called. ``replace=False`` (default)
        rejects any name or alias collision with a different provider.
        """
        if factory is not None and not callable(factory):
            raise TypeError("provider factory must be callable or None")

        if spec is not None:
            if isinstance(spec, ProviderSpec):
                provider_spec = spec
            elif isinstance(spec, Mapping):
                provider_spec = ProviderSpec.from_dict(spec)
            else:
                raise ContractValidationError(
                    "spec must be a ProviderSpec or mapping"
                )
            canonical = _normalize_identifier(name, "name")
            if canonical not in provider_spec.all_names():
                # Allow register(name=...) to match the canonical name after
                # normalization; otherwise require membership in all_names.
                if canonical != provider_spec.name:
                    raise ContractValidationError(
                        "register name must match provider spec name or alias"
                    )
        else:
            if capabilities is None:
                caps = CLICapabilities.chat_defaults()
            elif isinstance(capabilities, CLICapabilities):
                caps = capabilities
            elif isinstance(capabilities, Mapping):
                caps = CLICapabilities.from_dict(capabilities)
            else:
                raise ContractValidationError(
                    "capabilities must be a CLICapabilities"
                )
            provider_spec = ProviderSpec(
                name=name,
                aliases=tuple(aliases),
                description=description,
                capabilities=caps,
                streaming=spec_fields.get(
                    "streaming", CapabilitySupport.UNKNOWN
                ),
                tools=spec_fields.get(
                    "tools", CapabilitySupport.NOT_SUPPORTED
                ),
                sessions=spec_fields.get(
                    "sessions", CapabilitySupport.NOT_SUPPORTED
                ),
                cancellation=spec_fields.get(
                    "cancellation", CapabilitySupport.SUPPORTED
                ),
                provider_override=spec_fields.get(
                    "provider_override", CapabilitySupport.SUPPORTED
                ),
                model_override=spec_fields.get(
                    "model_override", CapabilitySupport.SUPPORTED
                ),
                locality=str(spec_fields.get("locality") or "unknown"),
                metadata=spec_fields.get("metadata") or {},
            )

        canonical = provider_spec.name
        claimed = list(provider_spec.all_names())

        with self._lock:
            for claimed_name in claimed:
                existing = self._index.get(claimed_name)
                if existing is not None and existing != canonical:
                    raise RegistryCollisionError(
                        f"provider name/alias {claimed_name!r} collides with "
                        f"canonical provider {existing!r}",
                        details={
                            "name": claimed_name,
                            "canonical": existing,
                            "requested": canonical,
                        },
                    )
            if not replace and canonical in self._entries:
                raise RegistryCollisionError(
                    f"provider {canonical!r} is already registered",
                    details={"canonical": canonical},
                )

            # Drop prior index entries owned by this canonical when replacing.
            prior = self._entries.get(canonical)
            if prior is not None:
                for old_name in prior.spec.all_names():
                    if self._index.get(old_name) == canonical:
                        del self._index[old_name]

            entry = RegistryEntry(spec=provider_spec, factory=factory)
            self._entries[canonical] = entry
            for claimed_name in claimed:
                self._index[claimed_name] = canonical
            return provider_spec

    def unregister(self, name: str) -> None:
        """Remove a provider by name or alias (no-op if missing)."""
        try:
            canonical = self.resolve(name)
        except (ProviderNotFoundError, ContractValidationError):
            return
        with self._lock:
            entry = self._entries.pop(canonical, None)
            if entry is None:
                return
            for claimed_name in entry.spec.all_names():
                if self._index.get(claimed_name) == canonical:
                    del self._index[claimed_name]

    def resolve(self, name: str) -> str:
        """Resolve a name or alias to its canonical provider name."""
        key = _normalize_identifier(name, "name")
        with self._lock:
            canonical = self._index.get(key)
            if canonical is None:
                raise ProviderNotFoundError(key)
            return canonical

    def resolve_optional(self, name: str) -> Optional[str]:
        """Like :meth:`resolve` but returns ``None`` when unknown."""
        try:
            return self.resolve(name)
        except (ProviderNotFoundError, ContractValidationError):
            return None

    def get_spec(self, name: str) -> ProviderSpec:
        """Return provider metadata without invoking the factory."""
        canonical = self.resolve(name)
        with self._lock:
            return self._entries[canonical].spec

    def get_entry(self, name: str) -> RegistryEntry:
        """Return the registry entry (spec + factory) without calling factory."""
        canonical = self.resolve(name)
        with self._lock:
            return self._entries[canonical]

    def has_provider(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except (ProviderNotFoundError, ContractValidationError):
            return False

    def list_names(self) -> tuple[str, ...]:
        """Return sorted canonical provider names without loading factories."""
        with self._lock:
            return tuple(sorted(self._entries))

    def list_specs(self) -> tuple[ProviderSpec, ...]:
        """Return sorted provider specs without loading factories."""
        with self._lock:
            return tuple(
                self._entries[name].spec for name in sorted(self._entries)
            )

    def list_aliases(self) -> dict[str, str]:
        """Return a sorted mapping of alias/name → canonical name."""
        with self._lock:
            return {key: self._index[key] for key in sorted(self._index)}

    def create(self, name: str) -> LLMProvider:
        """Lazily invoke the registered factory for ``name``."""
        canonical = self.resolve(name)
        with self._lock:
            entry = self._entries[canonical]
            factory = entry.factory
        if factory is None:
            raise ProviderLoadError(
                canonical, f"provider {canonical!r} has no factory"
            )
        try:
            provider = factory()
        except Exception as exc:  # noqa: BLE001 - surface factory failures
            raise ProviderLoadError(
                canonical,
                f"provider {canonical!r} factory failed: {exc}",
                details={"error_type": type(exc).__name__},
            ) from exc
        if provider is None:
            raise ProviderLoadError(
                canonical, f"provider {canonical!r} factory returned None"
            )
        return provider

    def register_many(
        self,
        specs: Iterable[ProviderSpec | Mapping[str, Any]],
        *,
        factories: Mapping[str, ProviderFactory] | None = None,
        replace: bool = False,
    ) -> list[ProviderSpec]:
        """Register multiple specs; optional factories keyed by canonical name."""
        factories = factories or {}
        registered: list[ProviderSpec] = []
        for item in specs:
            if isinstance(item, ProviderSpec):
                spec = item
            elif isinstance(item, Mapping):
                spec = ProviderSpec.from_dict(item)
            else:
                raise ContractValidationError(
                    "specs must be ProviderSpec or mappings"
                )
            factory = factories.get(spec.name)
            registered.append(
                self.register(
                    spec.name, spec=spec, factory=factory, replace=replace
                )
            )
        return registered

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry metadata only (no factories, no process activity)."""
        return {
            "providers": [spec.to_dict() for spec in self.list_specs()],
        }


_DEFAULT_REGISTRY: Optional[CLIProviderRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.RLock()


def get_default_registry() -> CLIProviderRegistry:
    """Return the process-wide CLI provider registry."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = CLIProviderRegistry()
        return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Clear the default registry (tests only)."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is not None:
            _DEFAULT_REGISTRY.clear()
        _DEFAULT_REGISTRY = CLIProviderRegistry()


def register_provider(
    name: str,
    *,
    factory: Optional[ProviderFactory] = None,
    aliases: Sequence[str] = (),
    description: str = "",
    capabilities: CLICapabilities | Mapping[str, Any] | None = None,
    spec: ProviderSpec | Mapping[str, Any] | None = None,
    replace: bool = False,
    **spec_fields: Any,
) -> ProviderSpec:
    """Register into the default registry without loading the provider."""
    return get_default_registry().register(
        name,
        factory=factory,
        aliases=aliases,
        description=description,
        capabilities=capabilities,
        spec=spec,
        replace=replace,
        **spec_fields,
    )


def resolve_provider_name(name: str) -> str:
    """Resolve an alias or name via the default registry."""
    return get_default_registry().resolve(name)


def list_providers() -> tuple[ProviderSpec, ...]:
    """List registered provider specs without invoking factories."""
    return get_default_registry().list_specs()


def get_provider(name: str) -> LLMProvider:
    """Create a provider instance via the default registry factory."""
    return get_default_registry().create(name)


__all__ = [
    "RegistryEntry",
    "CLIProviderRegistry",
    "get_default_registry",
    "reset_default_registry",
    "register_provider",
    "resolve_provider_name",
    "list_providers",
    "get_provider",
]
