"""Deterministic orchestration for the canonical AI service catalog.

The catalog publishes immutable registry generations.  Source work is staged
outside the read lock and a completed generation is swapped in atomically, so
readers observe either the old snapshot or the new snapshot and never a
partially refreshed mix.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .registry import (
    CatalogRegistry,
    RegistryClaim,
    RegistryDiagnostic,
    RegistryError,
)
from .resolver import CatalogResolver, ResolutionRequest, ResolutionResult
from .schema import (
    MAX_SNAPSHOT_RECORDS,
    CatalogSnapshot,
    DeploymentDescriptor,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    RouterBinding,
)
from .snapshot import CatalogPage, paginate_snapshot, snapshot_records

DEFAULT_MAX_SOURCES = 64
DEFAULT_MAX_SOURCE_RECORDS = MAX_SNAPSHOT_RECORDS
DEFAULT_MAX_OUTPUT_RECORDS = MAX_SNAPSHOT_RECORDS
MAX_CATALOG_DIAGNOSTICS = 2_000
_SOURCE = re.compile(r"^[a-z0-9](?:[a-z0-9._/-]{0,126}[a-z0-9])?$")
_RECORD_TYPES = ("providers", "models", "deployments", "bindings")
_ID_FIELDS = {
    "providers": "provider_id",
    "models": "model_id",
    "deployments": "deployment_id",
    "bindings": "binding_id",
}


class CatalogSourceError(ValueError):
    """A source registration or source result is invalid."""


class RefreshPolicyError(PermissionError):
    """Explicit refresh was not authorized for a side-effecting source."""


def _source_name(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise CatalogSourceError("source name must be canonical text")
    normalized = value.casefold()
    if (
        len(normalized.encode("utf-8")) > 128
        or not _SOURCE.fullmatch(normalized)
        or "//" in normalized
        or ".." in normalized
    ):
        raise CatalogSourceError("source name must be canonical text")
    return normalized


def _bound(value: Any, field_name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            "%s must be between %d and %d" % (field_name, minimum, maximum)
        )
    return value


def _precedence(value: Any) -> int:
    return _bound(
        value, "precedence", minimum=-1_000_000, maximum=1_000_000
    )


def _safe_failure(error: BaseException) -> str:
    """Return bounded diagnostics without echoing source-controlled values."""

    name = type(error).__name__
    return ("source raised %s" % name)[:256]


def _result_snapshot(result: Any) -> CatalogSnapshot:
    if isinstance(result, CatalogSnapshot):
        return result
    snapshot = getattr(result, "snapshot", None)
    if isinstance(snapshot, CatalogSnapshot):
        return snapshot
    if isinstance(result, Mapping):
        if "snapshot" in result:
            value = result["snapshot"]
            return (
                value
                if isinstance(value, CatalogSnapshot)
                else CatalogSnapshot.from_dict(value)
            )
        return CatalogSnapshot.from_dict(result)
    raise CatalogSourceError("source must return a CatalogSnapshot or source result")


def _result_precedence(result: Any, fallback: Optional[int]) -> int:
    metadata = getattr(result, "metadata", None)
    value = getattr(metadata, "precedence", None)
    if value is None:
        value = getattr(result, "precedence", None)
    if value is None:
        value = fallback
    if value is None:
        raise CatalogSourceError("source precedence must be explicit")
    return _precedence(value)


def _result_source(result: Any, fallback: str) -> str:
    metadata = getattr(result, "metadata", None)
    value = getattr(metadata, "source", None)
    if value is None:
        value = getattr(result, "source", None)
    return fallback if value is None else _source_name(value)


def _result_diagnostics(result: Any) -> Tuple[Any, ...]:
    values = getattr(result, "diagnostics", ())
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        raise CatalogSourceError("source diagnostics must be a bounded array")
    if len(values) > MAX_CATALOG_DIAGNOSTICS:
        raise CatalogSourceError("source diagnostics exceed the catalog bound")
    return tuple(values)


def _diagnostic_code(value: Any) -> str:
    code = getattr(value, "code", "source_diagnostic")
    if not isinstance(code, str) or not re.fullmatch(
        r"[a-z][a-z0-9_.-]{0,63}", code
    ):
        return "source_diagnostic"
    return code


@dataclass(frozen=True)
class CatalogDiagnostic:
    """A bounded source or merge diagnostic exposed by the aggregate view."""

    code: str
    message: str
    source: Optional[str] = None
    record_type: Optional[str] = None
    record_id: Optional[str] = None
    field: Optional[str] = None
    winner_source: Optional[str] = None
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]{0,63}", self.code
        ):
            raise ValueError("diagnostic code is invalid")
        if not isinstance(self.message, str) or len(
            self.message.encode("utf-8")
        ) > 1024:
            raise ValueError("diagnostic message is invalid")
        if self.source is not None:
            object.__setattr__(self, "source", _source_name(self.source))
        if self.record_type is not None and self.record_type not in _RECORD_TYPES:
            raise ValueError("diagnostic record type is invalid")
        for name in ("record_id", "field"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value.encode("utf-8")) > 256
            ):
                raise ValueError("diagnostic %s is invalid" % name)
        if self.winner_source is not None:
            object.__setattr__(
                self, "winner_source", _source_name(self.winner_source)
            )

    @classmethod
    def from_registry(cls, value: RegistryDiagnostic) -> "CatalogDiagnostic":
        return cls(
            code=value.code,
            message=value.message,
            source=None,
            record_type=value.record_type,
            record_id=value.record_id,
            field=value.field,
            winner_source=value.winner_source,
            ambiguous=value.ambiguous,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "field": self.field,
            "winner_source": self.winner_source,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True)
class RefreshPolicy:
    """Authorization supplied to an explicit source refresh.

    A side-effecting source is allowed only when ``allow_side_effects`` is true
    and, when ``allowed_sources`` is non-empty, its canonical name is listed.
    The policy is deliberately data-only; it is not forwarded into a source.
    """

    allow_side_effects: bool = False
    allowed_sources: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.allow_side_effects, bool):
            raise ValueError("allow_side_effects must be boolean")
        values = tuple(sorted({_source_name(item) for item in self.allowed_sources}))
        if len(values) > DEFAULT_MAX_SOURCES:
            raise ValueError("refresh policy has too many sources")
        object.__setattr__(self, "allowed_sources", values)

    def allows(self, source: str, *, side_effecting: bool) -> bool:
        if not side_effecting:
            return True
        canonical = _source_name(source)
        return self.allow_side_effects and (
            not self.allowed_sources or canonical in self.allowed_sources
        )


@dataclass(frozen=True)
class SourceState:
    """Last published state and bounded diagnostics for one registered source."""

    name: str
    precedence: int
    side_effecting: bool
    loaded: bool = False
    healthy: bool = False
    revision: Optional[str] = None
    reported_source: Optional[str] = None
    record_count: int = 0
    last_error: Optional[str] = None
    diagnostics: Tuple[CatalogDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _source_name(self.name))
        object.__setattr__(self, "precedence", _precedence(self.precedence))
        if not isinstance(self.side_effecting, bool):
            raise ValueError("side_effecting must be boolean")
        if self.reported_source is not None:
            object.__setattr__(
                self, "reported_source", _source_name(self.reported_source)
            )
        if self.revision is not None and (
            not isinstance(self.revision, str)
            or len(self.revision.encode("utf-8")) > 512
        ):
            raise ValueError("source revision is invalid")
        if self.last_error is not None and (
            not isinstance(self.last_error, str)
            or len(self.last_error.encode("utf-8")) > 256
        ):
            raise ValueError("source error is invalid")
        if len(self.diagnostics) > MAX_CATALOG_DIAGNOSTICS or any(
            not isinstance(item, CatalogDiagnostic) for item in self.diagnostics
        ):
            raise ValueError("source diagnostics are invalid or excessive")
        _bound(
            self.record_count,
            "record_count",
            minimum=0,
            maximum=DEFAULT_MAX_SOURCE_RECORDS,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "precedence": self.precedence,
            "side_effecting": self.side_effecting,
            "loaded": self.loaded,
            "healthy": self.healthy,
            "revision": self.revision,
            "reported_source": self.reported_source,
            "record_count": self.record_count,
            "last_error": self.last_error,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class CatalogView:
    """One snapshot with the claims and diagnostics that produced it."""

    snapshot: CatalogSnapshot
    claims: Tuple[RegistryClaim, ...]
    source_states: Tuple[SourceState, ...]
    diagnostics: Tuple[CatalogDiagnostic, ...]

    @property
    def revision(self) -> str:
        return self.snapshot.revision  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "claims": [item.to_dict() for item in self.claims],
            "source_states": [item.to_dict() for item in self.source_states],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one explicit, atomically published refresh generation."""

    snapshot: CatalogSnapshot
    refreshed: Tuple[str, ...] = ()
    failed: Tuple[str, ...] = ()
    unchanged: Tuple[str, ...] = ()
    source_states: Tuple[SourceState, ...] = ()
    diagnostics: Tuple[CatalogDiagnostic, ...] = ()

    @property
    def revision(self) -> str:
        return self.snapshot.revision  # type: ignore[return-value]

    @property
    def partial(self) -> bool:
        return bool(self.failed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "refreshed": list(self.refreshed),
            "failed": list(self.failed),
            "unchanged": list(self.unchanged),
            "source_states": [item.to_dict() for item in self.source_states],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class CatalogHealth:
    """Aggregate source and tri-state record health without active probing."""

    snapshot_revision: str
    sources: Tuple[SourceState, ...]
    record_counts: Tuple[Tuple[str, int], ...]
    state_counts: Tuple[Tuple[str, int], ...]

    @property
    def healthy(self) -> bool:
        return all(item.healthy for item in self.sources)

    @property
    def partial(self) -> bool:
        return any(not item.healthy for item in self.sources)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_revision": self.snapshot_revision,
            "healthy": self.healthy,
            "partial": self.partial,
            "sources": [item.to_dict() for item in self.sources],
            "record_counts": dict(self.record_counts),
            "state_counts": dict(self.state_counts),
        }


@dataclass
class _SourceRegistration:
    name: str
    adapter: Any
    precedence: Optional[int]
    side_effecting: bool
    state: SourceState


@dataclass(frozen=True)
class _StagedSource:
    name: str
    snapshot: CatalogSnapshot
    precedence: int
    reported_source: str
    diagnostics: Tuple[CatalogDiagnostic, ...] = ()


def _source_loader(source: Any, *, refreshing: bool) -> Any:
    names = ("refresh", "load", "snapshot", "read") if refreshing else (
        "load",
        "snapshot",
        "read",
    )
    for name in names:
        candidate = getattr(source, name, None)
        if callable(candidate):
            return candidate()
    if callable(source):
        return source()
    raise CatalogSourceError("source has no supported load operation")


def _source_diagnostics(name: str, values: Tuple[Any, ...]) -> Tuple[CatalogDiagnostic, ...]:
    result = []
    for value in values:
        message = getattr(value, "message", "source reported a diagnostic")
        if not isinstance(message, str):
            message = "source reported a diagnostic"
        message = message[:1024]
        result.append(
            CatalogDiagnostic(
                code=_diagnostic_code(value),
                message=message,
                source=name,
                record_id=getattr(value, "source_record_id", None),
            )
        )
    return tuple(result)


def _record_selector(record: Any, value: Optional[str]) -> bool:
    if value is None:
        return True
    normalized = value.casefold()
    identity = next(
        (
            getattr(record, field)
            for field in _ID_FIELDS.values()
            if hasattr(record, field)
        ),
        None,
    )
    names = (getattr(record, "name", ""),) + tuple(
        getattr(record, "aliases", ())
    )
    return normalized == identity or normalized in names


def _capabilities(record: Any) -> Tuple[Any, ...]:
    return tuple(getattr(record, "capabilities", ()))


class AIServiceCatalog:
    """Aggregate source snapshots without taking ownership of invocation."""

    def __init__(
        self,
        sources: Any = (),
        *,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_source_records: int = DEFAULT_MAX_SOURCE_RECORDS,
        max_output_records: int = DEFAULT_MAX_OUTPUT_RECORDS,
    ) -> None:
        self.max_sources = _bound(
            max_sources, "max_sources", minimum=1, maximum=1_024
        )
        self.max_source_records = _bound(
            max_source_records,
            "max_source_records",
            minimum=0,
            maximum=MAX_SNAPSHOT_RECORDS,
        )
        self.max_output_records = _bound(
            max_output_records,
            "max_output_records",
            minimum=0,
            maximum=MAX_SNAPSHOT_RECORDS,
        )
        self._lock = threading.RLock()
        self._refresh_lock = threading.RLock()
        self._registry = CatalogRegistry()
        self._registry_view = self._registry.view()
        self._sources: Dict[str, _SourceRegistration] = {}
        self._resolver = CatalogResolver()
        for name, source in self._normalize_sources(sources):
            self.register_source(name, source)

    @staticmethod
    def _normalize_sources(sources: Any) -> Tuple[Tuple[str, Any], ...]:
        if sources is None:
            return ()
        if isinstance(sources, Mapping):
            return tuple(
                sorted(
                    ((_source_name(name), source) for name, source in sources.items()),
                    key=lambda item: item[0],
                )
            )
        if isinstance(sources, (str, bytes)):
            raise CatalogSourceError("sources must be adapters, not text")
        result = []
        for item in sources:
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
            ):
                name, source = item
            else:
                source = item
                name = getattr(source, "source", None)
                if name is None:
                    raise CatalogSourceError(
                        "unnamed source adapters must expose a source name"
                    )
            result.append((_source_name(name), source))
        return tuple(sorted(result, key=lambda item: item[0]))

    def _stage(
        self, registration: _SourceRegistration, *, refreshing: bool
    ) -> _StagedSource:
        result = _source_loader(registration.adapter, refreshing=refreshing)
        snapshot = _result_snapshot(result)
        count = len(snapshot_records(snapshot))
        if count > self.max_source_records:
            raise CatalogSourceError(
                "source exceeds maximum record count (%d > %d)"
                % (count, self.max_source_records)
            )
        precedence = (
            registration.precedence
            if registration.precedence is not None
            else _result_precedence(result, None)
        )
        reported_source = _result_source(result, registration.name)
        diagnostics = _source_diagnostics(
            registration.name, _result_diagnostics(result)
        )
        return _StagedSource(
            name=registration.name,
            snapshot=snapshot,
            precedence=precedence,
            reported_source=reported_source,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _records(staged: _StagedSource) -> Tuple[Any, ...]:
        return snapshot_records(staged.snapshot)

    def _publish(
        self,
        selected: Tuple[str, ...],
        staged: Mapping[str, _StagedSource],
        failures: Mapping[str, BaseException],
    ) -> RefreshResult:
        with self._lock:
            current_registry = self._registry
            current_view = self._registry_view
            registrations = {
                name: self._sources[name]
                for name in selected
                if name in self._sources
            }
        replacement_names = set(staged)
        next_registry = CatalogRegistry()
        try:
            for claim in current_registry.claims():
                if claim.source not in replacement_names:
                    next_registry.register(
                        claim.record,
                        source=claim.source,
                        precedence=claim.precedence,
                    )
            for name in sorted(staged):
                item = staged[name]
                next_registry.register_many(
                    self._records(item),
                    source=name,
                    precedence=item.precedence,
                )
            next_view = next_registry.view()
            output_count = len(snapshot_records(next_view.snapshot))
            if output_count > self.max_output_records:
                raise CatalogSourceError(
                    "aggregate output exceeds maximum record count (%d > %d)"
                    % (output_count, self.max_output_records)
                )
        except (RegistryError, TypeError, ValueError) as exc:
            # A generation-wide bound or merge construction failure publishes
            # nothing; every successfully staged source keeps its prior claim.
            failures = dict(failures)
            for name in staged:
                failures.setdefault(name, exc)
            staged = {}
            next_registry = current_registry
            next_view = current_view

        catalog_diagnostics = []
        for name in sorted(failures):
            catalog_diagnostics.append(
                CatalogDiagnostic(
                    code="source_refresh_failed",
                    message=_safe_failure(failures[name]),
                    source=name,
                )
            )
        for name in sorted(staged):
            catalog_diagnostics.extend(staged[name].diagnostics)
        catalog_diagnostics.extend(
            CatalogDiagnostic.from_registry(item)
            for item in next_view.diagnostics
        )
        diagnostics = tuple(catalog_diagnostics[:MAX_CATALOG_DIAGNOSTICS])

        with self._lock:
            # Source registration and refresh use the same serialization lock,
            # so these registrations cannot have been replaced while staged.
            for name, registration in registrations.items():
                if name in staged:
                    item = staged[name]
                    registration.precedence = item.precedence
                    registration.state = SourceState(
                        name=name,
                        precedence=item.precedence,
                        side_effecting=registration.side_effecting,
                        loaded=True,
                        healthy=True,
                        revision=item.snapshot.revision,
                        reported_source=item.reported_source,
                        record_count=len(self._records(item)),
                        diagnostics=item.diagnostics,
                    )
                elif name in failures:
                    old = registration.state
                    registration.state = SourceState(
                        name=name,
                        precedence=old.precedence,
                        side_effecting=registration.side_effecting,
                        loaded=old.loaded,
                        healthy=False,
                        revision=old.revision,
                        reported_source=old.reported_source,
                        record_count=old.record_count,
                        last_error=_safe_failure(failures[name]),
                        diagnostics=old.diagnostics,
                    )
            if staged:
                self._registry = next_registry
                self._registry_view = next_view
            states = self.source_states()
            snapshot = self._registry_view.snapshot

        # ``unchanged`` describes source revisions, not aggregate revisions.
        unchanged = tuple(
            name
            for name in selected
            if name in staged
            and staged[name].snapshot.revision
            == next(
                (
                    state.revision
                    for state in states
                    if state.name == name
                ),
                None,
            )
            and not self._source_revision_changed(
                current_registry, name, staged[name].snapshot
            )
        )
        return RefreshResult(
            snapshot=snapshot,
            refreshed=tuple(sorted(staged)),
            failed=tuple(sorted(failures)),
            unchanged=tuple(sorted(unchanged)),
            source_states=states,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _source_revision_changed(
        registry: CatalogRegistry, source: str, snapshot: CatalogSnapshot
    ) -> bool:
        old = tuple(claim.record for claim in registry.claims(source=source))
        return CatalogSnapshot(
            providers=tuple(
                item for item in old if isinstance(item, ProviderDescriptor)
            ),
            models=tuple(
                item for item in old if isinstance(item, ModelDescriptor)
            ),
            deployments=tuple(
                item for item in old if isinstance(item, DeploymentDescriptor)
            ),
            bindings=tuple(
                item for item in old if isinstance(item, RouterBinding)
            ),
        ).revision != snapshot.revision

    def register_source(
        self,
        name: Any,
        source: Any = None,
        *,
        precedence: Optional[int] = None,
        side_effecting: Optional[bool] = None,
        load: bool = True,
        strict: bool = False,
    ) -> SourceState:
        """Register a source and optionally publish its pure initial load."""

        if source is None:
            source = name
            name = getattr(source, "source", None)
            if name is None:
                raise CatalogSourceError("source name is required")
        canonical = _source_name(name)
        if precedence is not None:
            precedence = _precedence(precedence)
        inferred = getattr(source, "side_effecting", False)
        selected_side_effecting = inferred if side_effecting is None else side_effecting
        if not isinstance(selected_side_effecting, bool):
            raise CatalogSourceError("side_effecting must be boolean")
        initial_precedence = precedence
        if initial_precedence is None:
            candidate = getattr(source, "precedence", None)
            initial_precedence = 0 if candidate is None else _precedence(candidate)
        registration = _SourceRegistration(
            name=canonical,
            adapter=source,
            precedence=precedence,
            side_effecting=selected_side_effecting,
            state=SourceState(
                name=canonical,
                precedence=initial_precedence,
                side_effecting=selected_side_effecting,
            ),
        )
        with self._refresh_lock:
            with self._lock:
                if (
                    canonical not in self._sources
                    and len(self._sources) >= self.max_sources
                ):
                    raise CatalogSourceError("catalog source capacity exceeded")
                self._sources[canonical] = registration
            if load:
                result = self._refresh_selected((canonical,), refreshing=False)
                if strict and result.failed:
                    raise CatalogSourceError(
                        "source %s could not be loaded" % canonical
                    )
        return self.source_state(canonical)

    add_source = register_source

    def unregister_source(self, name: str) -> bool:
        """Atomically remove a source registration and all of its claims."""

        canonical = _source_name(name)
        with self._refresh_lock:
            with self._lock:
                if canonical not in self._sources:
                    return False
                del self._sources[canonical]
                old = self._registry
            replacement = CatalogRegistry()
            for claim in old.claims():
                if claim.source != canonical:
                    replacement.register(
                        claim.record,
                        source=claim.source,
                        precedence=claim.precedence,
                    )
            replacement_view = replacement.view()
            with self._lock:
                self._registry = replacement
                self._registry_view = replacement_view
        return True

    remove_source = unregister_source

    def _refresh_selected(
        self, selected: Tuple[str, ...], *, refreshing: bool
    ) -> RefreshResult:
        staged: Dict[str, _StagedSource] = {}
        failures: Dict[str, BaseException] = {}
        with self._lock:
            registrations = {
                name: self._sources[name]
                for name in selected
            }
        for name in selected:
            try:
                staged[name] = self._stage(
                    registrations[name], refreshing=refreshing
                )
            except Exception as exc:  # source isolation is an API guarantee
                failures[name] = exc
        return self._publish(selected, staged, failures)

    def refresh(
        self,
        sources: Any,
        *,
        policy: Optional[RefreshPolicy] = None,
        raise_on_error: bool = False,
    ) -> RefreshResult:
        """Explicitly refresh only the named sources.

        Safe sources need no policy.  Every side-effecting source is preflighted
        before any selected source runs, preventing a partially authorized
        refresh.
        """

        if isinstance(sources, str):
            names = (sources,)
        elif isinstance(sources, Sequence) and not isinstance(sources, (bytes, str)):
            names = tuple(sources)
        else:
            raise CatalogSourceError("refresh requires a bounded source-name array")
        if not names or len(names) > self.max_sources:
            raise CatalogSourceError("refresh requires one or more bounded sources")
        selected = tuple(sorted({_source_name(item) for item in names}))
        with self._refresh_lock:
            with self._lock:
                missing = tuple(name for name in selected if name not in self._sources)
                if missing:
                    raise CatalogSourceError(
                        "unknown catalog source: %s" % ", ".join(missing)
                    )
                denied = tuple(
                    name
                    for name in selected
                    if self._sources[name].side_effecting
                    and (
                        policy is None
                        or not policy.allows(name, side_effecting=True)
                    )
                )
            if denied:
                raise RefreshPolicyError(
                    "refresh policy does not authorize: %s" % ", ".join(denied)
                )
            result = self._refresh_selected(selected, refreshing=True)
        if raise_on_error and result.failed:
            raise CatalogSourceError(
                "catalog refresh failed for: %s" % ", ".join(result.failed)
            )
        return result

    refresh_sources = refresh

    def source_state(self, name: str) -> SourceState:
        canonical = _source_name(name)
        with self._lock:
            registration = self._sources.get(canonical)
            if registration is None:
                raise CatalogSourceError("unknown catalog source: %s" % canonical)
            return registration.state

    def source_states(self) -> Tuple[SourceState, ...]:
        with self._lock:
            return tuple(
                self._sources[name].state for name in sorted(self._sources)
            )

    list_sources = source_states

    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            return self._registry_view.snapshot

    @property
    def revision(self) -> str:
        return self.snapshot().revision  # type: ignore[return-value]

    def view(self) -> CatalogView:
        with self._lock:
            registry_view = self._registry_view
            states = tuple(
                self._sources[name].state for name in sorted(self._sources)
            )
        diagnostics = []
        for state in states:
            if state.last_error is not None:
                diagnostics.append(
                    CatalogDiagnostic(
                        code="source_refresh_failed",
                        message=state.last_error,
                        source=state.name,
                    )
                )
            diagnostics.extend(state.diagnostics)
        seen = {item.to_dict().__repr__() for item in diagnostics}
        for item in registry_view.diagnostics:
            converted = CatalogDiagnostic.from_registry(item)
            key = converted.to_dict().__repr__()
            if key not in seen:
                diagnostics.append(converted)
                seen.add(key)
        return CatalogView(
            snapshot=registry_view.snapshot,
            claims=registry_view.claims,
            source_states=states,
            diagnostics=tuple(diagnostics[:MAX_CATALOG_DIAGNOSTICS]),
        )

    def claims(
        self,
        record_id: Optional[str] = None,
        *,
        record_type: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> Tuple[RegistryClaim, ...]:
        with self._lock:
            registry = self._registry
        return registry.claims(
            record_id, record_type=record_type, source=source
        )

    def diagnostics(self) -> Tuple[CatalogDiagnostic, ...]:
        return self.view().diagnostics

    def get(
        self,
        identifier: str,
        *,
        record_type: Optional[Any] = None,
        snapshot: Optional[CatalogSnapshot] = None,
    ) -> Optional[Any]:
        selected = snapshot if snapshot is not None else self.snapshot()
        with self._lock:
            registry = self._registry
        return registry.get(
            identifier, record_type=record_type, snapshot=selected
        )

    def resolve(
        self,
        request: Optional[ResolutionRequest] = None,
        *,
        snapshot: Optional[CatalogSnapshot] = None,
        **constraints: Any,
    ) -> ResolutionResult:
        selected = snapshot if snapshot is not None else self.snapshot()
        return self._resolver.resolve(selected, request, **constraints)

    def list_records(
        self,
        record_type: str,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        operation: Optional[Any] = None,
        modality: Optional[Any] = None,
        state: Optional[Mapping[str, bool]] = None,
        labels: Optional[Mapping[str, str]] = None,
        snapshot: Optional[CatalogSnapshot] = None,
    ) -> CatalogPage:
        """Return a bounded deterministic page from one immutable snapshot."""

        if record_type not in _RECORD_TYPES:
            raise ValueError("unknown catalog record type: %s" % record_type)
        selected = snapshot if snapshot is not None else self.snapshot()
        provider_value = self._filter_selector(provider, "provider")
        model_value = self._filter_selector(model, "model")
        operation_value = None if operation is None else Operation(operation)
        modality_value = None if modality is None else Modality(modality)
        state_values = dict(state or {})
        label_values = {}
        for key, value in (labels or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("label filters must be text key/value pairs")
            canonical_key = key.strip().casefold()
            if (
                not canonical_key
                or len(canonical_key.encode("utf-8")) > 64
                or len(value.encode("utf-8")) > 256
            ):
                raise ValueError("label filters must be bounded")
            label_values[canonical_key] = value
        if len(state_values) > 6 or len(label_values) > 64:
            raise ValueError("catalog filters exceed the supported bound")
        unknown_states = set(state_values) - set(
            OperationalState.__dataclass_fields__  # type: ignore[attr-defined]
        )
        if unknown_states or any(
            not isinstance(value, bool) for value in state_values.values()
        ):
            raise ValueError("state filters must be known boolean state fields")
        providers_by_id = {
            item.provider_id: item for item in selected.providers
        }
        models_by_id = {item.model_id: item for item in selected.models}
        deployments_by_id = {
            item.deployment_id: item for item in selected.deployments
        }

        def predicate(record: Any) -> bool:
            record_provider = (
                record
                if record_type == "providers"
                else providers_by_id.get(getattr(record, "provider_id", None))
            )
            record_model = (
                record
                if record_type == "models"
                else models_by_id.get(getattr(record, "model_id", None))
            )
            if provider_value is not None and (
                record_provider is None
                or not _record_selector(record_provider, provider_value)
            ):
                return False
            if model_value is not None and (
                record_model is None
                or not _record_selector(record_model, model_value)
            ):
                return False
            effective_capabilities = list(_capabilities(record))
            if record_type == "bindings":
                deployment = deployments_by_id.get(
                    getattr(record, "deployment_id", None)
                )
                for related in (record_provider, record_model, deployment):
                    if related is not None:
                        effective_capabilities.extend(_capabilities(related))
            operations = tuple(getattr(record, "operations", ())) + tuple(
                item
                for capability in effective_capabilities
                for item in capability.operations
            )
            if operation_value is not None and operation_value not in operations:
                return False
            if modality_value is not None and not any(
                modality_value in capability.input_modalities
                or modality_value in capability.output_modalities
                for capability in effective_capabilities
            ):
                return False
            record_state = getattr(record, "state", OperationalState())
            if any(
                getattr(record_state, key) is not value
                for key, value in state_values.items()
            ):
                return False
            record_labels = dict(getattr(record, "labels", ()))
            return all(
                record_labels.get(key) == value
                for key, value in label_values.items()
            )

        query = {
            "provider": provider_value,
            "model": model_value,
            "operation": (
                None if operation_value is None else operation_value.value
            ),
            "modality": (
                None if modality_value is None else modality_value.value
            ),
            "state": dict(sorted(state_values.items())),
            "labels": dict(sorted(label_values.items())),
        }
        return paginate_snapshot(
            selected,
            record_type,
            limit=limit,
            cursor=cursor,
            predicate=predicate,
            query=query,
        )

    @staticmethod
    def _filter_selector(value: Optional[str], field_name: str) -> Optional[str]:
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.encode("utf-8")) > 256
        ):
            raise ValueError(
                "%s filter must be bounded non-empty text" % field_name
            )
        return value.strip().casefold()

    def list_providers(self, **kwargs: Any) -> CatalogPage:
        return self.list_records("providers", **kwargs)

    def list_services(self, **kwargs: Any) -> CatalogPage:
        return self.list_providers(**kwargs)

    def list_models(self, **kwargs: Any) -> CatalogPage:
        return self.list_records("models", **kwargs)

    def list_deployments(self, **kwargs: Any) -> CatalogPage:
        return self.list_records("deployments", **kwargs)

    def list_bindings(self, **kwargs: Any) -> CatalogPage:
        return self.list_records("bindings", **kwargs)

    def health(self, *, snapshot: Optional[CatalogSnapshot] = None) -> CatalogHealth:
        """Project already known state; this method never probes a source."""

        selected = snapshot if snapshot is not None else self.snapshot()
        records = snapshot_records(selected)
        counts = tuple(
            (record_type, len(snapshot_records(selected, record_type)))
            for record_type in _RECORD_TYPES
        )
        state_counts = {
            "healthy": 0,
            "unhealthy": 0,
            "unknown": 0,
        }
        for record in records:
            value = getattr(getattr(record, "state", None), "healthy", None)
            key = "healthy" if value is True else "unhealthy" if value is False else "unknown"
            state_counts[key] += 1
        return CatalogHealth(
            snapshot_revision=selected.revision,  # type: ignore[arg-type]
            sources=self.source_states(),
            record_counts=counts,
            state_counts=tuple(sorted(state_counts.items())),
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(snapshot_records(self.snapshot()))

    def __len__(self) -> int:
        return len(snapshot_records(self.snapshot()))


Catalog = AIServiceCatalog


__all__ = [
    "AIServiceCatalog",
    "Catalog",
    "CatalogDiagnostic",
    "CatalogHealth",
    "CatalogSourceError",
    "CatalogView",
    "DEFAULT_MAX_OUTPUT_RECORDS",
    "DEFAULT_MAX_SOURCE_RECORDS",
    "DEFAULT_MAX_SOURCES",
    "MAX_CATALOG_DIAGNOSTICS",
    "RefreshPolicy",
    "RefreshPolicyError",
    "RefreshResult",
    "SourceState",
]
