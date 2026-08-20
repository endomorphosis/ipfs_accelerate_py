"""Deterministic, thread-safe registration and source-aware record merging."""

from __future__ import annotations

import dataclasses
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Type

from .identity import canonical_json, content_cid
from .schema import (
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    ModelDescriptor,
    OperationalState,
    ProviderDescriptor,
    RouterBinding,
)
from .snapshot import create_snapshot, snapshot_records

MAX_REGISTRY_CLAIMS = 50_000
MAX_DIAGNOSTICS = 10_000
MAX_SOURCE_LENGTH = 128
_SOURCE = re.compile(r"^[a-z0-9](?:[a-z0-9._/-]{0,126}[a-z0-9])?$")

_KIND_BY_TYPE = {
    ProviderDescriptor: "providers",
    ModelDescriptor: "models",
    DeploymentDescriptor: "deployments",
    RouterBinding: "bindings",
}
_TYPE_BY_KIND = {value: key for key, value in _KIND_BY_TYPE.items()}
_ID_FIELD = {
    "providers": "provider_id",
    "models": "model_id",
    "deployments": "deployment_id",
    "bindings": "binding_id",
}


class RegistryError(ValueError):
    """Base class for invalid registry operations."""


class RegistryCapacityError(RegistryError):
    """A registration would exceed a defensive registry bound."""


class AmbiguousAliasError(RegistryError):
    """A name or alias maps to more than one canonical identity."""


def _kind(value: Any) -> str:
    if isinstance(value, type) and value in _KIND_BY_TYPE:
        return _KIND_BY_TYPE[value]
    if isinstance(value, str):
        normalized = value.strip().casefold()
        aliases = {
            "provider": "providers",
            "model": "models",
            "deployment": "deployments",
            "binding": "bindings",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in _TYPE_BY_KIND:
            return normalized
    raise RegistryError("unknown catalog record type: %r" % (value,))


def _record_kind(record: Any) -> str:
    kind = _KIND_BY_TYPE.get(type(record))
    if kind is None:
        raise TypeError("unsupported catalog record type: %s" % type(record).__name__)
    return kind


def _source(value: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise RegistryError("source must be a canonical non-empty name")
    value = value.casefold()
    if (
        len(value.encode("utf-8")) > MAX_SOURCE_LENGTH
        or not _SOURCE.fullmatch(value)
        or "//" in value
        or ".." in value
    ):
        raise RegistryError("source must be a canonical non-empty name")
    return value


def _precedence(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -1_000_000 <= value <= 1_000_000
    ):
        raise RegistryError("precedence must be between -1000000 and 1000000")
    return value


def _timestamp(value: Optional[Any]) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise RegistryError("at must be an RFC 3339 timestamp") from exc
    else:
        raise RegistryError("at must be an RFC 3339 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistryError("at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _infer_source(record: Any) -> str:
    sources = {item.source for item in record.provenance}
    return next(iter(sources)) if len(sources) == 1 else "direct"


@dataclass(frozen=True)
class RegistryClaim:
    """One source's current assertion about a canonical identity."""

    source: str
    precedence: int
    record: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "precedence", _precedence(self.precedence))
        _record_kind(self.record)

    @property
    def record_type(self) -> str:
        return _record_kind(self.record)

    @property
    def record_id(self) -> str:
        return getattr(self.record, _ID_FIELD[self.record_type])

    @property
    def cid(self) -> str:
        return self.record.cid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "precedence": self.precedence,
            "record_type": self.record_type,
            "record": self.record.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryClaim":
        if not isinstance(data, Mapping) or set(data) != {
            "source",
            "precedence",
            "record_type",
            "record",
        }:
            raise RegistryError("RegistryClaim has missing or unknown fields")
        kind = _kind(data["record_type"])
        return cls(
            source=data["source"],
            precedence=data["precedence"],
            record=_TYPE_BY_KIND[kind].from_dict(data["record"]),
        )


@dataclass(frozen=True)
class RegistryDiagnostic:
    """Bounded merge, staleness, and alias-collision evidence."""

    code: str
    record_type: str
    record_id: str
    message: str
    sources: Tuple[str, ...] = ()
    field: Optional[str] = None
    winner_source: Optional[str] = None
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.code):
            raise RegistryError("diagnostic code is invalid")
        object.__setattr__(self, "record_type", _kind(self.record_type))
        if not isinstance(self.record_id, str) or len(self.record_id) > 256:
            raise RegistryError("diagnostic record_id is invalid")
        if not isinstance(self.message, str) or len(self.message.encode("utf-8")) > 1024:
            raise RegistryError("diagnostic message is invalid")
        sources = tuple(sorted({_source(item) for item in self.sources}))
        if len(sources) > 64:
            raise RegistryError("diagnostic has too many sources")
        object.__setattr__(self, "sources", sources)
        if self.winner_source is not None:
            object.__setattr__(self, "winner_source", _source(self.winner_source))
        if not isinstance(self.ambiguous, bool):
            raise RegistryError("diagnostic ambiguous must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "message": self.message,
            "sources": list(self.sources),
            "field": self.field,
            "winner_source": self.winner_source,
            "ambiguous": self.ambiguous,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryDiagnostic":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if not isinstance(data, Mapping) or set(data) != set(fields):
            raise RegistryError("RegistryDiagnostic has missing or unknown fields")
        return cls(**dict(data))


@dataclass(frozen=True)
class RegistryView:
    """A snapshot plus the exact claims and diagnostics used to build it."""

    snapshot: CatalogSnapshot
    claims: Tuple[RegistryClaim, ...]
    diagnostics: Tuple[RegistryDiagnostic, ...]

    @property
    def revision(self) -> str:
        return self.snapshot.revision

    @property
    def cid(self) -> str:
        return self.snapshot.cid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "claims": [item.to_dict() for item in self.claims],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryView":
        if not isinstance(data, Mapping) or set(data) != {
            "snapshot",
            "claims",
            "diagnostics",
        }:
            raise RegistryError("RegistryView has missing or unknown fields")
        return cls(
            snapshot=CatalogSnapshot.from_dict(data["snapshot"]),
            claims=tuple(RegistryClaim.from_dict(item) for item in data["claims"]),
            diagnostics=tuple(RegistryDiagnostic.from_dict(item) for item in data["diagnostics"]),
        )


def _claim_sort_key(claim: RegistryClaim) -> Tuple[Any, ...]:
    return (
        claim.record_type,
        claim.record_id,
        -claim.precedence,
        claim.source,
        claim.cid,
    )


def _is_stale(claim: RegistryClaim, at: str) -> bool:
    provenance = tuple(item for item in claim.record.provenance if item.source == claim.source)
    if not provenance:
        return False
    # Multiple receipts from one source remain useful while any receipt is
    # unexpired or has no expiry.
    return all(item.expires_at is not None and item.expires_at <= at for item in provenance)


def _meaningful(value: Any) -> bool:
    return value is not None and value != "" and value != LifecycleState.UNKNOWN


def _canonical_key(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return canonical_json(value)


def _choose_value(
    claims: Sequence[RegistryClaim],
    field_name: str,
    diagnostics: list,
) -> Tuple[Any, bool]:
    values = [
        (claim, getattr(claim.record, field_name))
        for claim in claims
        if _meaningful(getattr(claim.record, field_name))
    ]
    if not values:
        return getattr(claims[0].record, field_name), False
    best = max(claim.precedence for claim, _ in values)
    leaders = [(claim, value) for claim, value in values if claim.precedence == best]
    leader_values = {_canonical_key(value) for _, value in leaders}
    record_id = claims[0].record_id
    record_type = claims[0].record_type
    if len(leader_values) > 1:
        diagnostics.append(
            RegistryDiagnostic(
                code="ambiguous_claim",
                record_type=record_type,
                record_id=record_id,
                field=field_name,
                sources=tuple(claim.source for claim, _ in leaders),
                message="equal-precedence sources disagree on %s" % field_name,
                ambiguous=True,
            )
        )
        return None, True
    winner, value = min(leaders, key=lambda pair: (pair[0].source, pair[0].cid))
    losers = [claim for claim, other in values if _canonical_key(other) != _canonical_key(value)]
    if losers:
        diagnostics.append(
            RegistryDiagnostic(
                code="precedence_conflict",
                record_type=record_type,
                record_id=record_id,
                field=field_name,
                sources=tuple([winner.source] + [item.source for item in losers]),
                winner_source=winner.source,
                message="source precedence selected %s" % winner.source,
            )
        )
    return value, False


def _merge_collections(claims: Sequence[RegistryClaim], field_name: str) -> Tuple[Any, ...]:
    values: Dict[str, Any] = {}
    for claim in claims:
        for item in getattr(claim.record, field_name):
            values[_canonical_key(item)] = item
    return tuple(values[key] for key in sorted(values))


def _merge_labels(
    claims: Sequence[RegistryClaim], diagnostics: list
) -> Tuple[Tuple[str, str], ...]:
    keys = sorted({key for claim in claims for key, _ in getattr(claim.record, "labels", ())})
    result = []
    for key in keys:
        values = []
        for claim in claims:
            labels = dict(claim.record.labels)
            if key in labels:
                values.append((claim, labels[key]))
        best = max(claim.precedence for claim, _ in values)
        leaders = [(claim, value) for claim, value in values if claim.precedence == best]
        distinct = {value for _, value in leaders}
        if len(distinct) > 1:
            diagnostics.append(
                RegistryDiagnostic(
                    code="ambiguous_claim",
                    record_type=claims[0].record_type,
                    record_id=claims[0].record_id,
                    field="labels.%s" % key,
                    sources=tuple(claim.source for claim, _ in leaders),
                    message="equal-precedence sources disagree on label %s" % key,
                    ambiguous=True,
                )
            )
            raise AmbiguousAliasError("ambiguous label claim")
        winner, value = min(leaders, key=lambda pair: pair[0].source)
        if any(other != value for _, other in values):
            diagnostics.append(
                RegistryDiagnostic(
                    code="precedence_conflict",
                    record_type=claims[0].record_type,
                    record_id=claims[0].record_id,
                    field="labels.%s" % key,
                    sources=tuple(claim.source for claim, _ in values),
                    winner_source=winner.source,
                    message="source precedence selected label from %s" % winner.source,
                )
            )
        result.append((key, value))
    return tuple(result)


def _merge_state(
    claims: Sequence[RegistryClaim], diagnostics: list
) -> Tuple[Optional[OperationalState], bool]:
    values = {}
    ambiguous = False
    for name in OperationalState.__dataclass_fields__:  # type: ignore[attr-defined]
        present = [
            (claim, getattr(claim.record.state, name))
            for claim in claims
            if getattr(claim.record.state, name) is not None
        ]
        if not present:
            values[name] = None
            continue
        best = max(claim.precedence for claim, _ in present)
        leaders = [(claim, value) for claim, value in present if claim.precedence == best]
        distinct = {value for _, value in leaders}
        if len(distinct) > 1:
            diagnostics.append(
                RegistryDiagnostic(
                    code="ambiguous_claim",
                    record_type=claims[0].record_type,
                    record_id=claims[0].record_id,
                    field="state.%s" % name,
                    sources=tuple(claim.source for claim, _ in leaders),
                    message="equal-precedence sources disagree on state.%s" % name,
                    ambiguous=True,
                )
            )
            ambiguous = True
            continue
        winner, value = min(leaders, key=lambda pair: pair[0].source)
        if any(other != value for _, other in present):
            diagnostics.append(
                RegistryDiagnostic(
                    code="precedence_conflict",
                    record_type=claims[0].record_type,
                    record_id=claims[0].record_id,
                    field="state.%s" % name,
                    sources=tuple(claim.source for claim, _ in present),
                    winner_source=winner.source,
                    message="source precedence selected state.%s from %s" % (name, winner.source),
                )
            )
        values[name] = value
    return (None if ambiguous else OperationalState(**values)), ambiguous


def _merge_claims(claims: Sequence[RegistryClaim], diagnostics: list) -> Optional[Any]:
    ordered = tuple(sorted(claims, key=_claim_sort_key))
    if len({claim.cid for claim in ordered}) == 1:
        return ordered[0].record

    base = ordered[0].record
    changes: Dict[str, Any] = {}
    ambiguous = False
    union_fields = {"provenance"}
    if isinstance(base, (ProviderDescriptor, ModelDescriptor)):
        union_fields.update(("aliases", "capabilities"))
    elif isinstance(base, DeploymentDescriptor):
        union_fields.add("capabilities")
    elif isinstance(base, RouterBinding):
        union_fields.add("operations")

    identity_fields = {
        "schema_version",
        "provider_id",
        "model_id",
        "deployment_id",
        "binding_id",
        "name",
        "endpoint_uri",
        "router",
    }
    for field_name in base.__dataclass_fields__:  # type: ignore[attr-defined]
        if field_name in identity_fields:
            continue
        if field_name in union_fields:
            changes[field_name] = _merge_collections(ordered, field_name)
        elif field_name == "labels":
            try:
                changes[field_name] = _merge_labels(ordered, diagnostics)
            except AmbiguousAliasError:
                ambiguous = True
        elif field_name == "state":
            state, state_ambiguous = _merge_state(ordered, diagnostics)
            ambiguous = ambiguous or state_ambiguous
            if state is not None:
                changes[field_name] = state
        elif field_name == "created_at":
            present = [
                claim.record.created_at for claim in ordered if claim.record.created_at is not None
            ]
            changes[field_name] = min(present) if present else None
        elif field_name == "updated_at":
            present = [
                claim.record.updated_at for claim in ordered if claim.record.updated_at is not None
            ]
            changes[field_name] = max(present) if present else None
        else:
            value, field_ambiguous = _choose_value(ordered, field_name, diagnostics)
            ambiguous = ambiguous or field_ambiguous
            if not field_ambiguous:
                changes[field_name] = value
    if ambiguous:
        return None
    try:
        return dataclasses.replace(base, **changes)
    except (TypeError, ValueError) as exc:
        diagnostics.append(
            RegistryDiagnostic(
                code="merge_invalid",
                record_type=ordered[0].record_type,
                record_id=ordered[0].record_id,
                sources=tuple(claim.source for claim in ordered),
                message=("merged record is invalid: %s" % exc)[:1024],
                ambiguous=True,
            )
        )
        return None


def _alias_diagnostics(records: Sequence[Any]) -> Tuple[RegistryDiagnostic, ...]:
    by_kind: Dict[str, Dict[str, set]] = {}
    for record in records:
        kind = _record_kind(record)
        if not hasattr(record, "name"):
            continue
        keys = (record.name,) + tuple(getattr(record, "aliases", ()))
        by_kind.setdefault(kind, {})
        for value in keys:
            by_kind[kind].setdefault(value, set()).add(getattr(record, _ID_FIELD[kind]))
    diagnostics = []
    for kind in sorted(by_kind):
        for alias in sorted(by_kind[kind]):
            identities = sorted(by_kind[kind][alias])
            if len(identities) > 1:
                diagnostics.append(
                    RegistryDiagnostic(
                        code="alias_collision",
                        record_type=kind,
                        record_id=alias,
                        message=(
                            "alias maps to multiple canonical identities: %s"
                            % ", ".join(identities)
                        )[:1024],
                        ambiguous=True,
                    )
                )
    return tuple(diagnostics)


class CatalogRegistry:
    """A copy-on-read registry with deterministic source precedence.

    Mutations hold a short re-entrant lock.  Snapshot construction copies
    immutable claims under that lock and performs merging afterwards, allowing
    concurrent readers and writers without exposing partial source refreshes.
    """

    def __init__(
        self,
        records: Iterable[Any] = (),
        *,
        source_precedence: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._claims: Dict[Tuple[str, str, str], RegistryClaim] = {}
        self._source_precedence = {
            _source(name): _precedence(value) for name, value in (source_precedence or {}).items()
        }
        if records:
            self.register_many(records)

    def _make_claim(
        self, record: Any, source: Optional[str], precedence: Optional[int]
    ) -> RegistryClaim:
        _record_kind(record)
        normalized_source = _source(source if source is not None else _infer_source(record))
        normalized_precedence = (
            self._source_precedence.get(normalized_source, 0)
            if precedence is None
            else _precedence(precedence)
        )
        return RegistryClaim(normalized_source, normalized_precedence, record)

    def register(
        self,
        record: Any,
        *,
        source: Optional[str] = None,
        precedence: Optional[int] = None,
    ) -> str:
        """Register or replace one source's claim and return its canonical ID."""

        claim = self._make_claim(record, source, precedence)
        key = (claim.record_type, claim.record_id, claim.source)
        with self._lock:
            if key not in self._claims and len(self._claims) >= MAX_REGISTRY_CLAIMS:
                raise RegistryCapacityError("registry claim capacity exceeded")
            self._claims[key] = claim
        return claim.record_id

    def register_many(
        self,
        records: Iterable[Any],
        *,
        source: Optional[str] = None,
        precedence: Optional[int] = None,
    ) -> Tuple[str, ...]:
        """Atomically register a bounded collection of claims."""

        claims = tuple(self._make_claim(item, source, precedence) for item in records)
        if len(claims) > MAX_REGISTRY_CLAIMS:
            raise RegistryCapacityError("registration batch exceeds claim capacity")
        with self._lock:
            new_keys = {
                (claim.record_type, claim.record_id, claim.source) for claim in claims
            } - set(self._claims)
            if len(self._claims) + len(new_keys) > MAX_REGISTRY_CLAIMS:
                raise RegistryCapacityError("registry claim capacity exceeded")
            for claim in claims:
                self._claims[(claim.record_type, claim.record_id, claim.source)] = claim
        return tuple(claim.record_id for claim in claims)

    def replace_source(
        self,
        source: str,
        records: Iterable[Any],
        *,
        precedence: Optional[int] = None,
    ) -> Tuple[str, ...]:
        """Atomically replace all claims from *source*."""

        normalized = _source(source)
        claims = tuple(self._make_claim(item, normalized, precedence) for item in records)
        if len(claims) > MAX_REGISTRY_CLAIMS:
            raise RegistryCapacityError("source exceeds claim capacity")
        with self._lock:
            retained = {
                key: claim for key, claim in self._claims.items() if claim.source != normalized
            }
            if len(retained) + len(claims) > MAX_REGISTRY_CLAIMS:
                raise RegistryCapacityError("registry claim capacity exceeded")
            for claim in claims:
                retained[(claim.record_type, claim.record_id, normalized)] = claim
            self._claims = retained
        return tuple(claim.record_id for claim in claims)

    def remove_source(self, source: str) -> int:
        normalized = _source(source)
        with self._lock:
            keys = [key for key, claim in self._claims.items() if claim.source == normalized]
            for key in keys:
                del self._claims[key]
        return len(keys)

    unregister_source = remove_source

    def unregister(
        self,
        record_id: str,
        *,
        record_type: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> int:
        normalized_kind = _kind(record_type) if record_type is not None else None
        normalized_source = _source(source) if source is not None else None
        with self._lock:
            keys = [
                key
                for key, claim in self._claims.items()
                if claim.record_id == record_id
                and (normalized_kind is None or claim.record_type == normalized_kind)
                and (normalized_source is None or claim.source == normalized_source)
            ]
            for key in keys:
                del self._claims[key]
        return len(keys)

    def claims(
        self,
        record_id: Optional[str] = None,
        *,
        record_type: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> Tuple[RegistryClaim, ...]:
        normalized_kind = _kind(record_type) if record_type is not None else None
        normalized_source = _source(source) if source is not None else None
        with self._lock:
            result = tuple(
                claim
                for claim in self._claims.values()
                if (record_id is None or claim.record_id == record_id)
                and (normalized_kind is None or claim.record_type == normalized_kind)
                and (normalized_source is None or claim.source == normalized_source)
            )
        return tuple(sorted(result, key=_claim_sort_key))

    def view(self, *, at: Optional[Any] = None, created_at: Optional[Any] = None) -> RegistryView:
        observed_at = _timestamp(at)
        claims = self.claims()
        diagnostics = []
        grouped: Dict[Tuple[str, str], list] = {}
        for claim in claims:
            if _is_stale(claim, observed_at):
                diagnostics.append(
                    RegistryDiagnostic(
                        code="stale_claim",
                        record_type=claim.record_type,
                        record_id=claim.record_id,
                        sources=(claim.source,),
                        message="claim expired before snapshot time",
                    )
                )
                continue
            grouped.setdefault((claim.record_type, claim.record_id), []).append(claim)

        merged = []
        for key in sorted(grouped):
            record = _merge_claims(grouped[key], diagnostics)
            if record is not None:
                merged.append(record)
        diagnostics.extend(_alias_diagnostics(merged))
        diagnostics = sorted(
            diagnostics,
            key=lambda item: (
                item.record_type,
                item.record_id,
                item.code,
                item.field or "",
                item.sources,
            ),
        )
        if len(diagnostics) > MAX_DIAGNOSTICS:
            diagnostics = diagnostics[: MAX_DIAGNOSTICS - 1]
            diagnostics.append(
                RegistryDiagnostic(
                    code="diagnostics_truncated",
                    record_type="providers",
                    record_id="registry",
                    message="registry diagnostics exceeded the configured bound",
                )
            )
        snapshot_time = observed_at if created_at is None else _timestamp(created_at)
        snapshot = create_snapshot(merged, created_at=snapshot_time)
        return RegistryView(snapshot, claims, tuple(diagnostics))

    def snapshot(
        self, *, at: Optional[Any] = None, created_at: Optional[Any] = None
    ) -> CatalogSnapshot:
        return self.view(at=at, created_at=created_at).snapshot

    def diagnostics(self, *, at: Optional[Any] = None) -> Tuple[RegistryDiagnostic, ...]:
        return self.view(at=at).diagnostics

    def get(
        self,
        identifier: str,
        *,
        record_type: Optional[Any] = None,
        snapshot: Optional[CatalogSnapshot] = None,
    ) -> Optional[Any]:
        """Get by stable ID, canonical name, or unambiguous alias."""

        if not isinstance(identifier, str) or not identifier:
            raise RegistryError("identifier must be a non-empty string")
        view = snapshot if snapshot is not None else self.snapshot()
        kinds = (
            (_kind(record_type),)
            if record_type is not None
            else ("providers", "models", "deployments", "bindings")
        )
        exact_matches = []
        alias_matches = []
        normalized = identifier.casefold()
        for kind in kinds:
            for record in snapshot_records(view, kind):
                record_id = getattr(record, _ID_FIELD[kind])
                names = (getattr(record, "name", ""),) + tuple(getattr(record, "aliases", ()))
                if identifier == record_id:
                    exact_matches.append(record)
                elif normalized in names:
                    alias_matches.append(record)
        matches = exact_matches if exact_matches else alias_matches
        unique = {
            (_record_kind(item), getattr(item, _ID_FIELD[_record_kind(item)])): item
            for item in matches
        }
        if len(unique) > 1:
            raise AmbiguousAliasError(
                "identifier %r maps to multiple canonical records" % identifier
            )
        return next(iter(unique.values())) if unique else None

    def resolve_alias(
        self,
        record_type: Any,
        name: str,
        *,
        snapshot: Optional[CatalogSnapshot] = None,
    ) -> Optional[str]:
        record = self.get(name, record_type=record_type, snapshot=snapshot)
        return None if record is None else getattr(record, _ID_FIELD[_kind(record_type)])

    @property
    def revision(self) -> str:
        return self.snapshot().revision

    def __iter__(self) -> Iterator[Any]:
        return iter(snapshot_records(self.snapshot()))

    def __len__(self) -> int:
        return len(snapshot_records(self.snapshot()))


# Common concise alias.
Registry = CatalogRegistry
AIServiceRegistry = CatalogRegistry


__all__ = [
    "AmbiguousAliasError",
    "AIServiceRegistry",
    "CatalogRegistry",
    "MAX_DIAGNOSTICS",
    "MAX_REGISTRY_CLAIMS",
    "Registry",
    "RegistryCapacityError",
    "RegistryClaim",
    "RegistryDiagnostic",
    "RegistryError",
    "RegistryView",
]
