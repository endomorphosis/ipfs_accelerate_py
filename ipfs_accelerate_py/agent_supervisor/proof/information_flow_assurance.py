"""IFA — Information-Flow Assurance (FACP-044).

Introduces a fail-closed security lattice from ``Public`` through
``CryptographicSecret`` and ``WitnessSecret``, explicit declassification that
binds policy / actor / destination / exact source / purpose, taint
propagation, public-channel redaction with canaries, and bounded two-run
noninterference suites for:

* browser/host nonauthority
* cross-tenant isolation
* prompt/authority separation
* credential nonleakage
* proof-witness noninterference

Public logs, receipts, browser payloads, and prompts never carry protected raw
values or host paths.  Evidence is explicitly bounded and non-authoritative for
universal noninterference claims beyond the checked kernel suites.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = "facp/information-flow@1"
EVIDENCE_SCHEMA: Final[str] = "facp/information-flow@1"
TWO_RUN_SCHEMA: Final[str] = "facp/two-run-noninterference@1"
DECLASSIFICATION_SCHEMA: Final[str] = "facp/declassification-permit@1"
PUBLIC_AUDIT_SCHEMA: Final[str] = "facp/public-channel-audit@1"
TASK_ID: Final[str] = "FACP-044"
GOAL_ID: Final[str] = "FACP-G420"
BUNDLE: Final[str] = "facp/static/information-flow"
ASSURER_VERSION: Final[str] = "ifa/v1"

REDACTED_VALUE: Final[str] = "<redacted>"
HOST_PATH_REDACTED: Final[str] = "<host-path-redacted>"
MISSING_VALUE: Final[str] = "<missing>"
CANARY_PREFIX: Final[str] = "ifa-canary:"

EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "Public",
    "Internal",
    "RepositoryPrivate",
    "TenantPrivate",
    "MatterConfidential",
    "Credential",
    "CryptographicSecret",
    "WitnessSecret",
    "browser-host",
    "tenant",
    "prompt-authority",
    "credential",
    "witness",
)

# Destinations that may never receive a declassification (browser cannot mint
# authority and must not become a sink for protected material).
FORBIDDEN_DECLASSIFICATION_DESTINATIONS: Final[frozenset[str]] = frozenset(
    {
        "browser",
        "browser_authority",
        "browser_consent",
        "browser_allow",
        "browser_dry_run",
        "prompt",
        "model",
        "ui",
        "public_log",
        "public_receipt",
        "public_prompt",
    }
)

_ABSOLUTE_HOST_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:/|~/|\\\\|[A-Za-z]:[\\/]|file:)"
)
_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "cryptographic_secret",
        "hidden_witness",
        "host_path",
        "password",
        "private_key",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
        "witness_secret",
        "matter_confidential",
        "tenant_private",
    }
)
_PATH_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "cwd",
        "directory",
        "file_path",
        "host_path",
        "path",
        "workdir",
        "worktree_path",
    }
)


class InformationFlowError(ContractValidationError):
    """Malformed IFA contract, illegal declassification, or public leak."""


class SecurityLabel(str, Enum):
    """Ordered security lattice (low -> high).

    Higher labels dominate lower ones under join.  ``flows_to`` is the
    lattice order: a value labeled ``src`` may flow to sink ``dst`` only when
    ``src.flows_to(dst)`` holds (i.e. ``src`` is at most as confidential as
    ``dst``).
    """

    PUBLIC = "Public"
    INTERNAL = "Internal"
    REPOSITORY_PRIVATE = "RepositoryPrivate"
    TENANT_PRIVATE = "TenantPrivate"
    MATTER_CONFIDENTIAL = "MatterConfidential"
    CREDENTIAL = "Credential"
    CRYPTOGRAPHIC_SECRET = "CryptographicSecret"
    WITNESS_SECRET = "WitnessSecret"

    @property
    def rank(self) -> int:
        return _LABEL_RANK[self]

    def flows_to(self, other: "SecurityLabel") -> bool:
        return self.rank <= other.rank

    def join(self, other: "SecurityLabel") -> "SecurityLabel":
        return self if self.rank >= other.rank else other

    def meet(self, other: "SecurityLabel") -> "SecurityLabel":
        return self if self.rank <= other.rank else other


_LABEL_RANK: Final[Mapping[SecurityLabel, int]] = MappingProxyType(
    {
        SecurityLabel.PUBLIC: 0,
        SecurityLabel.INTERNAL: 1,
        SecurityLabel.REPOSITORY_PRIVATE: 2,
        SecurityLabel.TENANT_PRIVATE: 3,
        SecurityLabel.MATTER_CONFIDENTIAL: 4,
        SecurityLabel.CREDENTIAL: 5,
        SecurityLabel.CRYPTOGRAPHIC_SECRET: 6,
        SecurityLabel.WITNESS_SECRET: 7,
    }
)

# Public channels may only carry labels at or below Internal without an
# explicit declassification permit targeting that channel.
PUBLIC_CHANNEL_CEILING: Final[SecurityLabel] = SecurityLabel.INTERNAL


class PublicChannel(str, Enum):
    """Surfaces that must never carry protected raw values or host paths."""

    LOG = "log"
    RECEIPT = "receipt"
    BROWSER = "browser"
    PROMPT = "prompt"


class TwoRunPropertyKind(str, Enum):
    """Critical bounded two-run hyperproperties owned by FACP-044."""

    BROWSER_HOST = "browser-host"
    TENANT = "tenant"
    PROMPT_AUTHORITY = "prompt-authority"
    CREDENTIAL = "credential"
    WITNESS = "witness"


class TwoRunVerdict(str, Enum):
    HOLDS = "holds"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"


class LeakKind(str, Enum):
    PROTECTED_RAW_VALUE = "protected_raw_value"
    HOST_PATH = "host_path"
    CANARY = "canary"
    FORBIDDEN_KEY = "forbidden_key"
    LABEL_CEILING = "label_ceiling"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise InformationFlowError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise InformationFlowError(f"{name} must not be empty")
    return result


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise InformationFlowError(f"unsupported {name}") from exc


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise InformationFlowError(f"{name} must be an object with string keys")
    try:
        result = _canonical_value(dict(value))
    except ContractValidationError as exc:
        raise InformationFlowError(str(exc)) from exc
    assert isinstance(result, dict)
    return result


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalized_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def looks_like_host_path(value: str) -> bool:
    """Return True when ``value`` looks like an absolute host filesystem path."""

    if not value or not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if _ABSOLUTE_HOST_PATH_RE.match(text):
        return True
    if text.startswith("file://"):
        return True
    return False


def key_looks_protected(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in _SECRET_KEY_MARKERS or lowered in _PATH_KEY_MARKERS:
        return True
    for marker in _SECRET_KEY_MARKERS:
        if marker in lowered:
            return True
    if lowered.endswith("_path") or lowered.endswith("_dir") or lowered.endswith(
        "_directory"
    ):
        return True
    return False


def label_for_key(name: str) -> SecurityLabel:
    """Infer a conservative label from a field name."""

    lowered = _normalized_key(name)
    if "witness" in lowered:
        return SecurityLabel.WITNESS_SECRET
    if any(
        token in lowered
        for token in (
            "private_key",
            "cryptographic",
            "signing_key",
            "seed_phrase",
        )
    ):
        return SecurityLabel.CRYPTOGRAPHIC_SECRET
    if any(
        token in lowered
        for token in (
            "api_key",
            "access_token",
            "refresh_token",
            "password",
            "credential",
            "session_token",
            "authorization",
            "cookie",
            "secret",
        )
    ):
        return SecurityLabel.CREDENTIAL
    if "matter" in lowered or "confidential" in lowered:
        return SecurityLabel.MATTER_CONFIDENTIAL
    if "tenant" in lowered and "private" in lowered:
        return SecurityLabel.TENANT_PRIVATE
    if "host_path" in lowered or looks_like_host_path(name):
        return SecurityLabel.REPOSITORY_PRIVATE
    if "repository" in lowered and "private" in lowered:
        return SecurityLabel.REPOSITORY_PRIVATE
    if lowered in {"internal", "lane_id", "worktree_id"}:
        return SecurityLabel.INTERNAL
    return SecurityLabel.PUBLIC


def join_labels(*labels: SecurityLabel) -> SecurityLabel:
    result = SecurityLabel.PUBLIC
    for label in labels:
        result = result.join(_enum(label, SecurityLabel, "security label"))
    return result


# ---------------------------------------------------------------------------
# Labeled values and taint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledValue:
    """A value paired with its security label.

    High values are comparison/digest-only in public projections.  The raw
    ``value`` is intentionally excluded from ``repr``, equality, and hashing so
    accidental logging cannot re-emit protected material.
    """

    label: SecurityLabel
    value: Any = field(repr=False, compare=False, hash=False)
    source_id: str = ""
    canary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "label", _enum(self.label, SecurityLabel, "security label")
        )
        object.__setattr__(
            self, "source_id", _text(self.source_id, "source_id", required=False)
        )
        object.__setattr__(self, "canary", _text(self.canary, "canary", required=False))

    @property
    def digest(self) -> str:
        return _digest(self.value)

    @property
    def is_public(self) -> bool:
        return self.label is SecurityLabel.PUBLIC

    def flows_to(self, sink: SecurityLabel) -> bool:
        return self.label.flows_to(sink)

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label.value,
            "digest": self.digest,
            "redacted": True,
        }
        if self.source_id:
            payload["source_id"] = self.source_id
        if self.label.flows_to(PUBLIC_CHANNEL_CEILING):
            # Only low labels may expose the raw value in a public projection.
            payload["value"] = self.value
            payload["redacted"] = False
        return _canonical_value(payload)

    def join(self, other: "LabeledValue") -> "LabeledValue":
        return LabeledValue(
            label=self.label.join(other.label),
            value={"left": self.digest, "right": other.digest},
            source_id=self.source_id or other.source_id,
            canary=self.canary or other.canary,
        )


def mint_canary(source_id: str, *, kind: str = "secret") -> str:
    """Mint a deterministic canary token for leak detection (never a real secret)."""

    source = _text(source_id, "source_id")
    kind_text = _text(kind, "kind")
    digest = hashlib.sha256(f"{kind_text}:{source}".encode("utf-8")).hexdigest()[:24]
    return f"{CANARY_PREFIX}{kind_text}:{digest}"


def labeled(
    value: Any,
    label: SecurityLabel | str,
    *,
    source_id: str = "",
    with_canary: bool = False,
) -> LabeledValue:
    resolved = _enum(label, SecurityLabel, "security label")
    canary = ""
    if with_canary and not resolved.flows_to(PUBLIC_CHANNEL_CEILING):
        canary = mint_canary(source_id or _digest(value), kind=resolved.value)
    return LabeledValue(
        label=resolved, value=value, source_id=source_id, canary=canary
    )


@dataclass
class TaintStore:
    """Path-keyed taint map with join-on-write semantics."""

    _entries: dict[str, LabeledValue] = field(default_factory=dict)

    def get(self, path: str) -> LabeledValue | None:
        return self._entries.get(_text(path, "path"))

    def label_of(self, path: str, default: SecurityLabel = SecurityLabel.PUBLIC) -> SecurityLabel:
        entry = self.get(path)
        return entry.label if entry is not None else default

    def write(self, path: str, value: LabeledValue) -> LabeledValue:
        key = _text(path, "path")
        if not isinstance(value, LabeledValue):
            raise InformationFlowError("taint write requires a LabeledValue")
        existing = self._entries.get(key)
        if existing is None:
            self._entries[key] = value
            return value
        joined = LabeledValue(
            label=existing.label.join(value.label),
            value=value.value,
            source_id=value.source_id or existing.source_id,
            canary=value.canary or existing.canary,
        )
        self._entries[key] = joined
        return joined

    def replace(self, path: str, value: LabeledValue) -> LabeledValue:
        """Overwrite a path after an explicit declassification (no join)."""

        key = _text(path, "path")
        if not isinstance(value, LabeledValue):
            raise InformationFlowError("taint replace requires a LabeledValue")
        self._entries[key] = value
        return value

    def propagate(self, sources: Sequence[str], destination: str) -> LabeledValue:
        dest = _text(destination, "destination")
        joined = SecurityLabel.PUBLIC
        canary = ""
        source_id = ""
        for source in sources:
            entry = self.get(source)
            if entry is None:
                continue
            joined = joined.join(entry.label)
            canary = canary or entry.canary
            source_id = source_id or entry.source_id
        result = LabeledValue(
            label=joined,
            value={"propagated_from": sorted(sources), "destination": dest},
            source_id=source_id,
            canary=canary,
        )
        self._entries[dest] = result
        return result

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            path: entry.to_public_dict()
            for path, entry in sorted(self._entries.items())
        }


# ---------------------------------------------------------------------------
# Declassification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclassificationPermit(CanonicalContract):
    """Explicit, repository-owned declassification binding.

    Every allowed declassification must bind policy, actor, destination, exact
    source, and purpose.  Browser authority destinations are rejected.
    """

    SCHEMA = DECLASSIFICATION_SCHEMA

    policy_id: str
    actor: str
    destination: str
    source: str
    purpose: str
    from_label: SecurityLabel
    to_label: SecurityLabel
    permit_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "actor", _text(self.actor, "actor"))
        destination = _text(self.destination, "destination")
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "source", _text(self.source, "source"))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose"))
        object.__setattr__(
            self, "from_label", _enum(self.from_label, SecurityLabel, "from_label")
        )
        object.__setattr__(
            self, "to_label", _enum(self.to_label, SecurityLabel, "to_label")
        )
        if self.to_label.rank > self.from_label.rank:
            raise InformationFlowError(
                "declassification cannot raise confidentiality"
            )
        if self.to_label is self.from_label:
            raise InformationFlowError(
                "declassification requires a strictly lower destination label"
            )
        dest_key = _normalized_key(destination)
        if (
            destination in FORBIDDEN_DECLASSIFICATION_DESTINATIONS
            or dest_key in {_normalized_key(item) for item in FORBIDDEN_DECLASSIFICATION_DESTINATIONS}
            or dest_key.startswith("browser")
            or dest_key.startswith("public_")
        ):
            raise InformationFlowError(
                "browser/public destinations cannot receive declassification"
            )
        permit_id = self.permit_id.strip() if isinstance(self.permit_id, str) else ""
        if not permit_id:
            permit_id = content_identity(self._binding_payload())
        object.__setattr__(self, "permit_id", permit_id)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "actor": self.actor,
            "destination": self.destination,
            "source": self.source,
            "purpose": self.purpose,
            "from_label": self.from_label.value,
            "to_label": self.to_label.value,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ifa_version": ASSURER_VERSION,
            "task_id": TASK_ID,
            "permit_id": self.permit_id,
            **self._binding_payload(),
            "bindings_complete": True,
        }

    def binds_source(self, source_id: str) -> bool:
        return self.source == _text(source_id, "source_id")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeclassificationPermit":
        if not isinstance(payload, Mapping):
            raise InformationFlowError("declassification permit must be an object")
        return cls(
            policy_id=payload.get("policy_id", ""),
            actor=payload.get("actor", ""),
            destination=payload.get("destination", ""),
            source=payload.get("source", ""),
            purpose=payload.get("purpose", ""),
            from_label=payload.get("from_label", ""),
            to_label=payload.get("to_label", ""),
            permit_id=str(payload.get("permit_id") or ""),
        )


def require_declassification_bindings(permit: DeclassificationPermit) -> None:
    """Fail closed unless every required binding field is present and non-empty."""

    if not isinstance(permit, DeclassificationPermit):
        raise InformationFlowError("declassification permit required")
    for name in ("policy_id", "actor", "destination", "source", "purpose"):
        if not getattr(permit, name):
            raise InformationFlowError(f"declassification missing binding: {name}")
    if permit.from_label.rank <= permit.to_label.rank:
        raise InformationFlowError("declassification must lower the label")


def declassify(
    value: LabeledValue,
    permit: DeclassificationPermit,
    *,
    expected_source: str | None = None,
) -> LabeledValue:
    """Lower a labeled value only when an exact declassification permit matches."""

    if not isinstance(value, LabeledValue):
        raise InformationFlowError("declassify requires a LabeledValue")
    require_declassification_bindings(permit)
    source = expected_source or value.source_id
    if not source:
        raise InformationFlowError("declassify requires an exact source id")
    if not permit.binds_source(source):
        raise InformationFlowError("declassification source does not match value")
    if value.label is not permit.from_label:
        raise InformationFlowError(
            "declassification from_label does not match value label"
        )
    # When the destination label may appear on a public channel, retain only a
    # digest so declassification cannot re-emit the protected raw value.
    if permit.to_label.flows_to(PUBLIC_CHANNEL_CEILING):
        released: Any = {
            "declassified": True,
            "source": source,
            "digest": value.digest,
            "permit_id": permit.permit_id,
        }
    else:
        released = value.value
    return LabeledValue(
        label=permit.to_label,
        value=released,
        source_id=source,
        canary=value.canary,
    )


# ---------------------------------------------------------------------------
# Public channel projection and audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakFinding(CanonicalContract):
    SCHEMA = ""

    kind: LeakKind
    path: str
    channel: PublicChannel
    detail: str
    digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, LeakKind, "leak kind"))
        object.__setattr__(self, "path", _text(self.path, "path"))
        object.__setattr__(
            self, "channel", _enum(self.channel, PublicChannel, "public channel")
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail"))
        object.__setattr__(
            self, "digest", _text(self.digest, "digest", required=False)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "channel": self.channel.value,
            "detail": self.detail,
            "digest": self.digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(self._payload())


@dataclass(frozen=True)
class PublicChannelAudit(CanonicalContract):
    SCHEMA = PUBLIC_AUDIT_SCHEMA

    channel: PublicChannel
    clean: bool
    findings: tuple[LeakFinding, ...]
    projected: Mapping[str, Any]
    canaries_searched: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "channel", _enum(self.channel, PublicChannel, "public channel")
        )
        if not isinstance(self.clean, bool):
            raise InformationFlowError("clean must be a boolean")
        findings = tuple(self.findings)
        if any(not isinstance(item, LeakFinding) for item in findings):
            raise InformationFlowError("findings must be LeakFinding values")
        object.__setattr__(self, "findings", findings)
        if self.clean and findings:
            raise InformationFlowError("clean audit cannot retain findings")
        if (not self.clean) and not findings:
            raise InformationFlowError("dirty audit requires findings")
        # Store a plain dict so audits remain JSON-serializable while still
        # frozen at the dataclass boundary.
        object.__setattr__(self, "projected", _mapping(self.projected, "projected"))
        object.__setattr__(
            self,
            "canaries_searched",
            tuple(
                sorted(
                    {
                        _text(item, "canary")
                        for item in (self.canaries_searched or ())
                    }
                )
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ifa_version": ASSURER_VERSION,
            "task_id": TASK_ID,
            "channel": self.channel.value,
            "clean": self.clean,
            "findings": [item.to_dict() for item in self.findings],
            "projected": dict(self.projected),
            "canaries_searched": list(self.canaries_searched),
            "contains_protected_raw": False if self.clean else True,
        }


def project_public_value(
    value: Any,
    *,
    channel: PublicChannel | str = PublicChannel.LOG,
    known_protected: Sequence[str] = (),
) -> Any:
    """Project a value for a public channel, redacting secrets and host paths."""

    channel_value = _enum(channel, PublicChannel, "public channel")
    protected = {item for item in known_protected if isinstance(item, str) and item}

    def _project(node: Any, path: str) -> Any:
        if isinstance(node, LabeledValue):
            if node.label.flows_to(PUBLIC_CHANNEL_CEILING):
                return _project(node.value, path)
            return {
                "label": node.label.value,
                "digest": node.digest,
                "redacted": True,
            }
        if isinstance(node, Mapping):
            out: dict[str, Any] = {}
            for key, item in node.items():
                name = str(key)
                child = f"{path}.{name}"
                projected = _project(item, child)
                if isinstance(item, str) and looks_like_host_path(item):
                    out[name] = HOST_PATH_REDACTED
                    continue
                if isinstance(projected, str) and looks_like_host_path(projected):
                    out[name] = HOST_PATH_REDACTED
                    continue
                if _normalized_key(name) in _PATH_KEY_MARKERS or _normalized_key(
                    name
                ).endswith(("_path", "_dir", "_directory")):
                    if isinstance(item, str) and item not in (
                        "",
                        REDACTED_VALUE,
                        HOST_PATH_REDACTED,
                    ):
                        # Path-shaped fields on public channels are always host-path redacted.
                        out[name] = HOST_PATH_REDACTED
                        continue
                if key_looks_protected(name):
                    out[name] = REDACTED_VALUE
                    continue
                out[name] = projected
            return out
        if isinstance(node, (list, tuple)):
            return [_project(item, f"{path}[{index}]") for index, item in enumerate(node)]
        if isinstance(node, str):
            if node in protected or node.startswith(CANARY_PREFIX):
                return REDACTED_VALUE
            if looks_like_host_path(node):
                return HOST_PATH_REDACTED
            return node
        if node is None or isinstance(node, (bool, int)):
            return node
        if isinstance(node, float):
            # Canonical contracts reject floats; stringify conservatively.
            return REDACTED_VALUE
        return REDACTED_VALUE

    _ = channel_value  # channel retained for call-site clarity / future policy
    return _project(value, "$")


def audit_public_payload(
    payload: Any,
    *,
    channel: PublicChannel | str,
    canaries: Sequence[str] = (),
    known_protected: Sequence[str] = (),
    labeled_ceiling: SecurityLabel = PUBLIC_CHANNEL_CEILING,
) -> PublicChannelAudit:
    """Audit a public payload for protected raw values, host paths, and canaries."""

    channel_value = _enum(channel, PublicChannel, "public channel")
    ceiling = _enum(labeled_ceiling, SecurityLabel, "labeled_ceiling")
    canary_set = tuple(
        sorted({_text(item, "canary") for item in canaries if item})
    )
    protected = {item for item in known_protected if isinstance(item, str) and item}
    findings: list[LeakFinding] = []

    if isinstance(payload, LabeledValue) and not payload.label.flows_to(ceiling):
        findings.append(
            LeakFinding(
                kind=LeakKind.LABEL_CEILING,
                path="$",
                channel=channel_value,
                detail=f"label {payload.label.value} exceeds public ceiling",
                digest=payload.digest,
            )
        )

    def _scan(node: Any, path: str) -> None:
        if isinstance(node, LabeledValue):
            if not node.label.flows_to(ceiling):
                findings.append(
                    LeakFinding(
                        kind=LeakKind.LABEL_CEILING,
                        path=path,
                        channel=channel_value,
                        detail=f"label {node.label.value} exceeds public ceiling",
                        digest=node.digest,
                    )
                )
            _scan(node.value, path)
            return
        if isinstance(node, Mapping):
            for key, item in node.items():
                name = str(key)
                child = f"{path}.{name}"
                if key_looks_protected(name) and item not in (
                    None,
                    "",
                    REDACTED_VALUE,
                    HOST_PATH_REDACTED,
                ):
                    if isinstance(item, str) and item in (
                        REDACTED_VALUE,
                        HOST_PATH_REDACTED,
                    ):
                        continue
                    # Raw protected field present on a public channel.
                    if isinstance(item, str) and looks_like_host_path(item):
                        kind = LeakKind.HOST_PATH
                    elif key_looks_protected(name):
                        kind = LeakKind.FORBIDDEN_KEY
                    else:
                        kind = LeakKind.PROTECTED_RAW_VALUE
                    findings.append(
                        LeakFinding(
                            kind=kind,
                            path=child,
                            channel=channel_value,
                            detail=f"protected field {name!r} present on {channel_value.value}",
                            digest=_digest(item) if not isinstance(item, (bytes, bytearray)) else "",
                        )
                    )
                _scan(item, child)
            return
        if isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                _scan(item, f"{path}[{index}]")
            return
        if isinstance(node, str):
            if node in canary_set or node.startswith(CANARY_PREFIX):
                findings.append(
                    LeakFinding(
                        kind=LeakKind.CANARY,
                        path=path,
                        channel=channel_value,
                        detail="canary token observed on public channel",
                        digest=_digest(node),
                    )
                )
            elif node in protected:
                findings.append(
                    LeakFinding(
                        kind=LeakKind.PROTECTED_RAW_VALUE,
                        path=path,
                        channel=channel_value,
                        detail="known protected raw value observed",
                        digest=_digest(node),
                    )
                )
            elif looks_like_host_path(node):
                findings.append(
                    LeakFinding(
                        kind=LeakKind.HOST_PATH,
                        path=path,
                        channel=channel_value,
                        detail="host path observed on public channel",
                        digest=_digest(node),
                    )
                )

    _scan(payload, "$")
    # Deduplicate by (kind, path, detail)
    unique: dict[tuple[str, str, str], LeakFinding] = {}
    for finding in findings:
        unique[(finding.kind.value, finding.path, finding.detail)] = finding
    ordered = tuple(
        sorted(unique.values(), key=lambda item: (item.kind.value, item.path, item.detail))
    )
    projected = project_public_value(
        payload, channel=channel_value, known_protected=tuple(protected)
    )
    return PublicChannelAudit(
        channel=channel_value,
        clean=not ordered,
        findings=ordered,
        projected=projected if isinstance(projected, Mapping) else {"value": projected},
        canaries_searched=canary_set,
    )


def assert_public_channels_clean(
    payloads: Mapping[str, Any],
    *,
    canaries: Sequence[str] = (),
    known_protected: Sequence[str] = (),
) -> tuple[PublicChannelAudit, ...]:
    """Audit log/receipt/browser/prompt payloads; raise on any leak."""

    audits: list[PublicChannelAudit] = []
    for name, payload in payloads.items():
        channel = _enum(name, PublicChannel, "public channel")
        audit = audit_public_payload(
            payload,
            channel=channel,
            canaries=canaries,
            known_protected=known_protected,
        )
        audits.append(audit)
        if not audit.clean:
            kinds = ", ".join(sorted({item.kind.value for item in audit.findings}))
            raise InformationFlowError(
                f"public {channel.value} channel leaked protected material ({kinds})"
            )
    return tuple(audits)


# ---------------------------------------------------------------------------
# Two-run noninterference suites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoRunTrace:
    """One execution used in a bounded two-run comparison.

    High inputs are excluded from public projections and equality.
    """

    trace_id: str
    public_inputs: Mapping[str, Any]
    observations: Mapping[str, Any]
    high_inputs: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _text(self.trace_id, "trace_id"))
        object.__setattr__(
            self, "public_inputs", _mapping(self.public_inputs, "public_inputs")
        )
        object.__setattr__(
            self, "observations", _mapping(self.observations, "observations")
        )
        object.__setattr__(
            self, "high_inputs", _mapping(self.high_inputs, "high_inputs")
        )

    @property
    def public_ref(self) -> str:
        return content_identity(
            {
                "trace_id": self.trace_id,
                "public_inputs_sha256": _digest(self.public_inputs),
                "observations_sha256": _digest(self.observations),
                "high_inputs_redacted": True,
            }
        )

    def to_public_dict(self) -> dict[str, Any]:
        return _canonical_value(
            {
                "trace_id": self.trace_id,
                "public_ref": self.public_ref,
                "high_inputs_redacted": True,
            }
        )


@dataclass(frozen=True)
class TwoRunModel(CanonicalContract):
    """Reviewed bounded two-run model for one critical IFA property."""

    SCHEMA = TWO_RUN_SCHEMA

    model_id: str
    kind: TwoRunPropertyKind
    low_input_fields: tuple[str, ...]
    observation_fields: tuple[str, ...]
    high_input_fields: tuple[str, ...]
    description: str
    authority_ceiling: str = "bounded_self_composition"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, TwoRunPropertyKind, "two-run kind")
        )
        object.__setattr__(
            self,
            "low_input_fields",
            tuple(_text(item, "low_input_fields") for item in self.low_input_fields),
        )
        observations = tuple(
            _text(item, "observation_fields") for item in self.observation_fields
        )
        if not observations:
            raise InformationFlowError("observation_fields must not be empty")
        object.__setattr__(self, "observation_fields", observations)
        highs = tuple(
            _text(item, "high_input_fields") for item in self.high_input_fields
        )
        if not highs:
            raise InformationFlowError("high_input_fields must not be empty")
        object.__setattr__(self, "high_input_fields", highs)
        if set(highs) & set(self.low_input_fields):
            raise InformationFlowError("high and low input fields must be disjoint")
        object.__setattr__(self, "description", _text(self.description, "description"))
        object.__setattr__(
            self,
            "authority_ceiling",
            _text(self.authority_ceiling, "authority_ceiling"),
        )
        if self.authority_ceiling not in {
            "bounded_self_composition",
            "non_authoritative_test",
        }:
            raise InformationFlowError(
                "IFA two-run evidence cannot claim universal noninterference authority"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ifa_version": ASSURER_VERSION,
            "task_id": TASK_ID,
            "model_id": self.model_id,
            "kind": self.kind.value,
            "low_input_fields": list(self.low_input_fields),
            "observation_fields": list(self.observation_fields),
            "high_input_fields": list(self.high_input_fields),
            "description": self.description,
            "authority_ceiling": self.authority_ceiling,
            "authoritative": False,
            "bounded": True,
        }


def _path_get(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return MISSING_VALUE
        current = current[component]
    return current


def _project_fields(mapping: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(_path_get(mapping, field) for field in fields)


@dataclass(frozen=True)
class TwoRunResult(CanonicalContract):
    SCHEMA = TWO_RUN_SCHEMA

    model_id: str
    kind: TwoRunPropertyKind
    verdict: TwoRunVerdict
    left_ref: str
    right_ref: str
    observation_fields: tuple[str, ...]
    differing_fields: tuple[str, ...]
    reason: str
    authoritative: bool = False
    bounded: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, TwoRunPropertyKind, "two-run kind")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, TwoRunVerdict, "verdict")
        )
        object.__setattr__(self, "left_ref", _text(self.left_ref, "left_ref"))
        object.__setattr__(self, "right_ref", _text(self.right_ref, "right_ref"))
        object.__setattr__(
            self,
            "observation_fields",
            tuple(_text(item, "observation_fields") for item in self.observation_fields),
        )
        object.__setattr__(
            self,
            "differing_fields",
            tuple(_text(item, "differing_fields") for item in self.differing_fields),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.authoritative:
            raise InformationFlowError(
                "IFA two-run results are bounded and non-authoritative"
            )
        if not self.bounded:
            raise InformationFlowError("IFA two-run results must declare bounded=True")
        if self.verdict is TwoRunVerdict.VIOLATED and not self.differing_fields:
            raise InformationFlowError("violated result requires differing fields")
        if self.verdict is TwoRunVerdict.HOLDS and self.differing_fields:
            raise InformationFlowError("holds result cannot list differing fields")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ifa_version": ASSURER_VERSION,
            "task_id": TASK_ID,
            "model_id": self.model_id,
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "observation_fields": list(self.observation_fields),
            "differing_fields": list(self.differing_fields),
            "reason": self.reason,
            "authoritative": False,
            "bounded": True,
            "contains_high_inputs": False,
        }


def compare_two_runs(
    model: TwoRunModel,
    left: TwoRunTrace,
    right: TwoRunTrace,
) -> TwoRunResult:
    """Compare two traces under a model; high inputs must differ, lows must match."""

    if not isinstance(model, TwoRunModel):
        raise InformationFlowError("model must be a TwoRunModel")
    if not isinstance(left, TwoRunTrace) or not isinstance(right, TwoRunTrace):
        raise InformationFlowError("two-run comparison requires TwoRunTrace values")

    left_low = _project_fields(left.public_inputs, model.low_input_fields)
    right_low = _project_fields(right.public_inputs, model.low_input_fields)
    if left_low != right_low:
        return TwoRunResult(
            model_id=model.model_id,
            kind=model.kind,
            verdict=TwoRunVerdict.INCONCLUSIVE,
            left_ref=left.public_ref,
            right_ref=right.public_ref,
            observation_fields=model.observation_fields,
            differing_fields=(),
            reason="low inputs differ; pair is outside the noninterference premise",
        )

    left_high = _project_fields(left.high_inputs, model.high_input_fields)
    right_high = _project_fields(right.high_inputs, model.high_input_fields)
    if left_high == right_high:
        return TwoRunResult(
            model_id=model.model_id,
            kind=model.kind,
            verdict=TwoRunVerdict.INCONCLUSIVE,
            left_ref=left.public_ref,
            right_ref=right.public_ref,
            observation_fields=model.observation_fields,
            differing_fields=(),
            reason="high inputs are identical; pair does not stress the property",
        )

    left_obs = _project_fields(left.observations, model.observation_fields)
    right_obs = _project_fields(right.observations, model.observation_fields)
    differing = tuple(
        field
        for field, left_value, right_value in zip(
            model.observation_fields, left_obs, right_obs
        )
        if left_value != right_value
    )
    if differing:
        return TwoRunResult(
            model_id=model.model_id,
            kind=model.kind,
            verdict=TwoRunVerdict.VIOLATED,
            left_ref=left.public_ref,
            right_ref=right.public_ref,
            observation_fields=model.observation_fields,
            differing_fields=differing,
            reason="low observations differ while low inputs match and high inputs vary",
        )
    return TwoRunResult(
        model_id=model.model_id,
        kind=model.kind,
        verdict=TwoRunVerdict.HOLDS,
        left_ref=left.public_ref,
        right_ref=right.public_ref,
        observation_fields=model.observation_fields,
        differing_fields=(),
        reason="bounded two-run pair preserves low observations",
    )


# ---------------------------------------------------------------------------
# Default models and hermetic kernels
# ---------------------------------------------------------------------------


def default_two_run_models() -> tuple[TwoRunModel, ...]:
    """Return the five critical FACP-044 two-run models."""

    return (
        TwoRunModel(
            model_id="ifa.browser-host@1",
            kind=TwoRunPropertyKind.BROWSER_HOST,
            low_input_fields=(
                "host.request.method",
                "host.request.resource",
                "host.request.argument_cid",
            ),
            observation_fields=("host.authorization_digest", "host.decision"),
            high_input_fields=(
                "browser.allow",
                "browser.consent",
                "browser.dry_run",
            ),
            description=(
                "Changing browser allow/consent/dry_run must not change host "
                "authorization inputs or decisions."
            ),
        ),
        TwoRunModel(
            model_id="ifa.tenant@1",
            kind=TwoRunPropertyKind.TENANT,
            low_input_fields=("tenant.id", "request.operation"),
            observation_fields=("tenant.view_digest", "response.status"),
            high_input_fields=("foreign_tenant.private",),
            description=(
                "Foreign-tenant private data must not affect this tenant's "
                "observations."
            ),
        ),
        TwoRunModel(
            model_id="ifa.prompt-authority@1",
            kind=TwoRunPropertyKind.PROMPT_AUTHORITY,
            low_input_fields=("request.operation", "request.argument_cid"),
            observation_fields=("authority.decision", "authority.digest"),
            high_input_fields=("prompt.text", "prompt.claimed_allow"),
            description=(
                "Prompt text and claimed allow flags must not grant authority "
                "or alter host authority decisions."
            ),
        ),
        TwoRunModel(
            model_id="ifa.credential@1",
            kind=TwoRunPropertyKind.CREDENTIAL,
            low_input_fields=("actor.ref", "request.operation"),
            observation_fields=("public.receipt_digest", "public.log_digest"),
            high_input_fields=("credential.secret",),
            description=(
                "Credential secrets must not appear in or alter public receipt "
                "or log digests across runs."
            ),
        ),
        TwoRunModel(
            model_id="ifa.witness@1",
            kind=TwoRunPropertyKind.WITNESS,
            low_input_fields=("proof.obligation_id", "proof.public_inputs_digest"),
            observation_fields=("receipt.public_digest", "receipt.status"),
            high_input_fields=("proof.witness",),
            description=(
                "Private proof witnesses must not affect public receipt digests "
                "or status."
            ),
        ),
    )


def model_for(kind: TwoRunPropertyKind | str) -> TwoRunModel:
    resolved = _enum(kind, TwoRunPropertyKind, "two-run kind")
    for model in default_two_run_models():
        if model.kind is resolved:
            return model
    raise InformationFlowError(f"no default model for {resolved.value}")


def _host_authorization_digest(
    method: str, resource: str, argument_cid: str
) -> str:
    return _digest(
        {
            "method": method,
            "resource": resource,
            "argument_cid": argument_cid,
            "issuer": "effect_admission_kernel",
        }
    )


def simulate_browser_host_kernel(
    *,
    method: str,
    resource: str,
    argument_cid: str,
    browser_allow: bool,
    browser_consent: bool,
    browser_dry_run: bool,
) -> TwoRunTrace:
    """Hermetic host kernel: browser authority fields are ignored."""

    _ = (browser_allow, browser_consent, browser_dry_run)
    decision = "admit_review"
    auth_digest = _host_authorization_digest(method, resource, argument_cid)
    return TwoRunTrace(
        trace_id=content_identity(
            {"kind": "browser-host", "auth": auth_digest, "decision": decision}
        )[:24],
        public_inputs={
            "host": {
                "request": {
                    "method": method,
                    "resource": resource,
                    "argument_cid": argument_cid,
                }
            }
        },
        observations={
            "host": {
                "authorization_digest": auth_digest,
                "decision": decision,
            }
        },
        high_inputs={
            "browser": {
                "allow": browser_allow,
                "consent": browser_consent,
                "dry_run": browser_dry_run,
            }
        },
    )


def simulate_tenant_kernel(
    *,
    tenant_id: str,
    operation: str,
    foreign_private: str,
) -> TwoRunTrace:
    view = _digest({"tenant": tenant_id, "operation": operation})
    return TwoRunTrace(
        trace_id=content_identity({"kind": "tenant", "view": view})[:24],
        public_inputs={"tenant": {"id": tenant_id}, "request": {"operation": operation}},
        observations={
            "tenant": {"view_digest": view},
            "response": {"status": "ok"},
        },
        high_inputs={"foreign_tenant": {"private": foreign_private}},
    )


def simulate_prompt_authority_kernel(
    *,
    operation: str,
    argument_cid: str,
    prompt_text: str,
    claimed_allow: bool,
) -> TwoRunTrace:
    _ = (prompt_text, claimed_allow)
    decision = "deny_prompt_authority"
    digest = _digest(
        {
            "operation": operation,
            "argument_cid": argument_cid,
            "issuer": "effect_admission_kernel",
        }
    )
    return TwoRunTrace(
        trace_id=content_identity({"kind": "prompt-authority", "digest": digest})[:24],
        public_inputs={
            "request": {"operation": operation, "argument_cid": argument_cid}
        },
        observations={
            "authority": {"decision": decision, "digest": digest},
        },
        high_inputs={
            "prompt": {"text": prompt_text, "claimed_allow": claimed_allow},
        },
    )


def simulate_credential_kernel(
    *,
    actor_ref: str,
    operation: str,
    credential_secret: str,
) -> TwoRunTrace:
    # Public digests intentionally exclude the credential material.
    receipt = _digest({"actor": actor_ref, "operation": operation, "channel": "receipt"})
    log = _digest({"actor": actor_ref, "operation": operation, "channel": "log"})
    return TwoRunTrace(
        trace_id=content_identity({"kind": "credential", "receipt": receipt})[:24],
        public_inputs={"actor": {"ref": actor_ref}, "request": {"operation": operation}},
        observations={
            "public": {"receipt_digest": receipt, "log_digest": log},
        },
        high_inputs={"credential": {"secret": credential_secret}},
    )


def simulate_witness_kernel(
    *,
    obligation_id: str,
    public_inputs_digest: str,
    witness: str,
) -> TwoRunTrace:
    _ = witness
    receipt_digest = _digest(
        {
            "obligation_id": obligation_id,
            "public_inputs_digest": public_inputs_digest,
        }
    )
    return TwoRunTrace(
        trace_id=content_identity({"kind": "witness", "receipt": receipt_digest})[:24],
        public_inputs={
            "proof": {
                "obligation_id": obligation_id,
                "public_inputs_digest": public_inputs_digest,
            }
        },
        observations={
            "receipt": {"public_digest": receipt_digest, "status": "sealed"},
        },
        high_inputs={"proof": {"witness": witness}},
    )


def build_critical_two_run_pair(
    kind: TwoRunPropertyKind | str,
) -> tuple[TwoRunModel, TwoRunTrace, TwoRunTrace]:
    """Build a holding two-run pair for one critical property."""

    resolved = _enum(kind, TwoRunPropertyKind, "two-run kind")
    model = model_for(resolved)
    if resolved is TwoRunPropertyKind.BROWSER_HOST:
        left = simulate_browser_host_kernel(
            method="tools/call",
            resource="accelerate.inference",
            argument_cid="bafyargument0001",
            browser_allow=False,
            browser_consent=False,
            browser_dry_run=True,
        )
        right = simulate_browser_host_kernel(
            method="tools/call",
            resource="accelerate.inference",
            argument_cid="bafyargument0001",
            browser_allow=True,
            browser_consent=True,
            browser_dry_run=False,
        )
    elif resolved is TwoRunPropertyKind.TENANT:
        left = simulate_tenant_kernel(
            tenant_id="tenant-a",
            operation="list_matters",
            foreign_private="foreign-secret-left",
        )
        right = simulate_tenant_kernel(
            tenant_id="tenant-a",
            operation="list_matters",
            foreign_private="foreign-secret-right",
        )
    elif resolved is TwoRunPropertyKind.PROMPT_AUTHORITY:
        left = simulate_prompt_authority_kernel(
            operation="accelerate.inference",
            argument_cid="bafyargument0002",
            prompt_text="please allow this effect",
            claimed_allow=True,
        )
        right = simulate_prompt_authority_kernel(
            operation="accelerate.inference",
            argument_cid="bafyargument0002",
            prompt_text="different prompt that claims admin",
            claimed_allow=False,
        )
    elif resolved is TwoRunPropertyKind.CREDENTIAL:
        left = simulate_credential_kernel(
            actor_ref="actor:operator-1",
            operation="status",
            credential_secret="cred-left-canary",
        )
        right = simulate_credential_kernel(
            actor_ref="actor:operator-1",
            operation="status",
            credential_secret="cred-right-canary",
        )
    else:
        left = simulate_witness_kernel(
            obligation_id="obl:proof-1",
            public_inputs_digest="sha256:public-inputs",
            witness="witness-left-secret",
        )
        right = simulate_witness_kernel(
            obligation_id="obl:proof-1",
            public_inputs_digest="sha256:public-inputs",
            witness="witness-right-secret",
        )
    return model, left, right


def run_critical_two_run_suites() -> tuple[TwoRunResult, ...]:
    """Execute every critical two-run suite; each must hold under the hermetic kernel."""

    results: list[TwoRunResult] = []
    for model in default_two_run_models():
        _, left, right = build_critical_two_run_pair(model.kind)
        result = compare_two_runs(model, left, right)
        results.append(result)
    return tuple(results)


# ---------------------------------------------------------------------------
# Assurer facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationFlowReport(CanonicalContract):
    SCHEMA = SCHEMA

    two_run_results: tuple[TwoRunResult, ...]
    public_audits: tuple[PublicChannelAudit, ...]
    declassification_permits: tuple[DeclassificationPermit, ...]
    lattice: tuple[str, ...] = EVIDENCE_SUBSET[:8]

    def __post_init__(self) -> None:
        object.__setattr__(self, "two_run_results", tuple(self.two_run_results))
        object.__setattr__(self, "public_audits", tuple(self.public_audits))
        object.__setattr__(
            self, "declassification_permits", tuple(self.declassification_permits)
        )
        object.__setattr__(
            self,
            "lattice",
            tuple(_text(item, "lattice") for item in self.lattice),
        )
        if any(result.verdict is not TwoRunVerdict.HOLDS for result in self.two_run_results):
            raise InformationFlowError("report requires every critical two-run suite to hold")
        if any(not audit.clean for audit in self.public_audits):
            raise InformationFlowError("report requires clean public channel audits")
        for permit in self.declassification_permits:
            require_declassification_bindings(permit)

    @property
    def all_suites_hold(self) -> bool:
        return all(item.verdict is TwoRunVerdict.HOLDS for item in self.two_run_results)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ifa_version": ASSURER_VERSION,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "lattice": list(self.lattice),
            "two_run_results": [item.to_dict() for item in self.two_run_results],
            "public_audits": [item.to_dict() for item in self.public_audits],
            "declassification_permits": [
                item.to_dict() for item in self.declassification_permits
            ],
            "authoritative": False,
            "bounded": True,
        }


@dataclass
class InformationFlowAssurer:
    """Facade for lattice ops, declassification, redaction, and two-run suites."""

    taint: TaintStore = field(default_factory=TaintStore)
    permits: list[DeclassificationPermit] = field(default_factory=list)

    def register_permit(self, permit: DeclassificationPermit) -> DeclassificationPermit:
        require_declassification_bindings(permit)
        self.permits.append(permit)
        return permit

    def label_and_store(
        self,
        path: str,
        value: Any,
        label: SecurityLabel | str,
        *,
        with_canary: bool = True,
    ) -> LabeledValue:
        entry = labeled(
            value, label, source_id=path, with_canary=with_canary
        )
        return self.taint.write(path, entry)

    def declassify_path(
        self,
        path: str,
        permit: DeclassificationPermit,
    ) -> LabeledValue:
        entry = self.taint.get(path)
        if entry is None:
            raise InformationFlowError(f"no labeled value at {path}")
        lowered = declassify(entry, permit, expected_source=path)
        return self.taint.replace(path, lowered)

    def run_assurance(
        self,
        *,
        public_payloads: Mapping[str, Any] | None = None,
        known_protected: Sequence[str] = (),
    ) -> InformationFlowReport:
        results = run_critical_two_run_suites()
        canaries = tuple(
            entry.canary
            for entry in self.taint._entries.values()
            if entry.canary
        )
        payloads = dict(public_payloads or {})
        for channel in PublicChannel:
            payloads.setdefault(channel.value, {})
        audits = assert_public_channels_clean(
            payloads,
            canaries=canaries,
            known_protected=known_protected,
        )
        return InformationFlowReport(
            two_run_results=results,
            public_audits=audits,
            declassification_permits=tuple(self.permits),
        )


def lattice_labels() -> tuple[SecurityLabel, ...]:
    return tuple(
        sorted(SecurityLabel, key=lambda item: item.rank)
    )


__all__ = [
    "ASSURER_VERSION",
    "BUNDLE",
    "CANARY_PREFIX",
    "DECLASSIFICATION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SUBSET",
    "FORBIDDEN_DECLASSIFICATION_DESTINATIONS",
    "GOAL_ID",
    "HOST_PATH_REDACTED",
    "PUBLIC_AUDIT_SCHEMA",
    "PUBLIC_CHANNEL_CEILING",
    "REDACTED_VALUE",
    "SCHEMA",
    "TASK_ID",
    "TWO_RUN_SCHEMA",
    "DeclassificationPermit",
    "InformationFlowAssurer",
    "InformationFlowError",
    "InformationFlowReport",
    "LabeledValue",
    "LeakFinding",
    "LeakKind",
    "PublicChannel",
    "PublicChannelAudit",
    "SecurityLabel",
    "TaintStore",
    "TwoRunModel",
    "TwoRunPropertyKind",
    "TwoRunResult",
    "TwoRunTrace",
    "TwoRunVerdict",
    "assert_public_channels_clean",
    "audit_public_payload",
    "build_critical_two_run_pair",
    "compare_two_runs",
    "declassify",
    "default_two_run_models",
    "join_labels",
    "key_looks_protected",
    "label_for_key",
    "labeled",
    "lattice_labels",
    "looks_like_host_path",
    "mint_canary",
    "model_for",
    "project_public_value",
    "require_declassification_bindings",
    "run_critical_two_run_suites",
    "simulate_browser_host_kernel",
    "simulate_credential_kernel",
    "simulate_prompt_authority_kernel",
    "simulate_tenant_kernel",
    "simulate_witness_kernel",
]
