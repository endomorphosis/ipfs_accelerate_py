"""Source-disclosure policy enforced before model calls (EAAEF-031).

Frozen, content-addressed ``SourceDisclosurePolicy@1`` and
``DisclosureDecision@1`` records bind confidentiality, exclusions, secret
scanning, provider allowlists, local-only classification, byte limits, and
exact ContextPack identity.  Missing policy fails closed.  A CID, prompt, or
imported history never grants disclosure.

Importing this module performs no I/O and opens no sockets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .external_principal import ExternalPrincipal

CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

SOURCE_DISCLOSURE_POLICY_INTERFACE: Final[str] = "SourceDisclosurePolicy@1"
DISCLOSURE_DECISION_INTERFACE: Final[str] = "DisclosureDecision@1"

SOURCE_DISCLOSURE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-disclosure-policy@1"
)
DISCLOSURE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/disclosure-decision@1"
)

MAX_ID_BYTES: Final[int] = 256
MAX_REASON_BYTES: Final[int] = 256
MAX_EXCLUSIONS: Final[int] = 256
MAX_PROVIDERS: Final[int] = 128
MAX_EXCLUSION_BYTES: Final[int] = 1024
DEFAULT_MAX_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_BYTES: Final[int] = 2_000_000

_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)
_POLICY_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)
_PROVIDER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)
_EXCLUSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_.@+=-][A-Za-z0-9_./@+=-]*$"
)

_LOCAL_PROVIDER_PREFIXES: Final[tuple[str, ...]] = (
    "local:",
    "local/",
    "sim:",
    "simulated:",
    "injected:",
    "hermetic:",
    "offline:",
    "stub:",
)

_API_KEY_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:api[_-]?key)\s*[:=]\s*\S{4,}"
)
_API_KEY_BYTES_RE: Final[re.Pattern[bytes]] = re.compile(
    br"(?i)(?:api[_-]?key)\s*[:=]\s*\S{4,}"
)
_BEARER_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
)
_BEARER_BYTES_RE: Final[re.Pattern[bytes]] = re.compile(
    br"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
)
_PEM_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_PEM_BYTES_RE: Final[re.Pattern[bytes]] = re.compile(
    br"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    br"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {"content_id", "cid", "identity", "canonical_id"}
)
_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        *_IDENTITY_KEYS,
    }
)

TEnum = TypeVar("TEnum", bound=Enum)


class ConfidentialityClass(str, Enum):
    """Closed confidentiality classes.  Unknown values fail closed."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

    @property
    def rank(self) -> int:
        return {
            ConfidentialityClass.PUBLIC: 0,
            ConfidentialityClass.INTERNAL: 1,
            ConfidentialityClass.CONFIDENTIAL: 2,
            ConfidentialityClass.SECRET: 3,
        }[self]


class ProviderLocality(str, Enum):
    LOCAL = "local"
    EXTERNAL = "external"


class DisclosureVerdict(str, Enum):
    PERMIT = "permit"
    DENY = "deny"


class SecretKind(str, Enum):
    API_KEY = "api_key"
    BEARER = "bearer"
    PRIVATE_KEY_PEM = "private_key_pem"


class DisclosurePolicyError(ContractValidationError):
    """Malformed, missing, or violated source-disclosure policy."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PolicyMissingError(DisclosurePolicyError):
    """No SourceDisclosurePolicy@1 was supplied."""


class SecretMaterialError(DisclosurePolicyError):
    """Secret-shaped material was present in a disclosure payload."""


class ContextPackIdentityError(DisclosurePolicyError):
    """ContextPack content identity was missing or did not match."""


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = MAX_ID_BYTES,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise DisclosurePolicyError(f"{name} must be a string", reason_code="malformed")
    else:
        result = value.strip()
    if required and not result:
        raise DisclosurePolicyError(f"{name} is required", reason_code="malformed")
    if "\x00" in result:
        raise DisclosurePolicyError(
            f"{name} must not contain NUL", reason_code="malformed"
        )
    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        raise DisclosurePolicyError(
            f"{name} exceeds {max_bytes} UTF-8 bytes", reason_code="bounds"
        )
    if result and pattern is not None and pattern.fullmatch(result) is None:
        raise DisclosurePolicyError(
            f"{name} is not a permitted identifier", reason_code="malformed"
        )
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DisclosurePolicyError(f"{name} must be a boolean", reason_code="malformed")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DisclosurePolicyError(
            f"{name} must be a non-negative integer", reason_code="malformed"
        )
    if value < 0:
        raise DisclosurePolicyError(
            f"{name} must be a non-negative integer", reason_code="malformed"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise DisclosurePolicyError(
            f"{name} must be a positive integer", reason_code="malformed"
        )
    return result


def _enum(value: Any, enum_type: type[TEnum], name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise DisclosurePolicyError(
            f"{name} must be one of: {allowed}", reason_code="malformed"
        ) from exc


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Sequence[str], *, artifact_name: str
) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise DisclosurePolicyError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload",
            reason_code="malformed",
        )


def _require_schema(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_interface: str,
    *,
    artifact_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise DisclosurePolicyError(
            f"{artifact_name} payload must be an object", reason_code="malformed"
        )
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        raise DisclosurePolicyError(
            f"unsupported {artifact_name} schema {schema!r}; expected {expected_schema}",
            reason_code="unsupported_version",
        )
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        raise DisclosurePolicyError(
            f"unsupported {artifact_name} interface {interface!r}; "
            f"expected {expected_interface}",
            reason_code="unsupported_version",
        )
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", CONTRACT_VERSION):
            raise DisclosurePolicyError(
                f"unsupported {artifact_name} contract version; rebuild with "
                f"{expected_interface}",
                reason_code="unsupported_version",
            )


def _claimed_identity(
    payload: Mapping[str, Any], actual: str, *, artifact_name: str
) -> None:
    for name in ("content_id", "cid", "identity", "canonical_id"):
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise DisclosurePolicyError(
                f"{artifact_name} content identity does not match payload",
                reason_code="identity_mismatch",
            )


def _unique_tokens(
    value: Any,
    name: str,
    *,
    max_items: int,
    max_bytes: int,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DisclosurePolicyError(
            f"{name} must be a sequence of strings", reason_code="malformed"
        )
    if len(value) > max_items:
        raise DisclosurePolicyError(
            f"{name} exceeds {max_items} items", reason_code="bounds"
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = _text(item, name, max_bytes=max_bytes, pattern=pattern)
        if token in seen:
            raise DisclosurePolicyError(
                f"{name} must not contain duplicates", reason_code="malformed"
            )
        seen.add(token)
        result.append(token)
    return tuple(result)


def _normalized_key(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def _kind_name(kind: SecretKind | str) -> str:
    if isinstance(kind, SecretKind):
        return kind.value
    return SecretKind(kind).value


_POLICY_FIELDS: Final[tuple[str, ...]] = (
    "policy_id",
    "confidentiality",
    "exclusions",
    "allowed_providers",
    "local_only",
    "max_bytes",
    "require_secret_scan",
)


@dataclass(frozen=True)
class SourceDisclosurePolicy(CanonicalContract):
    """Frozen content-addressed source-disclosure policy @1."""

    SCHEMA: ClassVar[str] = SOURCE_DISCLOSURE_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = SOURCE_DISCLOSURE_POLICY_INTERFACE

    policy_id: str
    confidentiality: ConfidentialityClass | str
    exclusions: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()
    local_only: bool = True
    max_bytes: int = DEFAULT_MAX_BYTES
    require_secret_scan: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", pattern=_POLICY_RE)
        )
        confidentiality = _enum(
            self.confidentiality, ConfidentialityClass, "confidentiality"
        )
        object.__setattr__(self, "confidentiality", confidentiality)
        object.__setattr__(
            self,
            "exclusions",
            _unique_tokens(
                self.exclusions,
                "exclusions",
                max_items=MAX_EXCLUSIONS,
                max_bytes=MAX_EXCLUSION_BYTES,
                pattern=_EXCLUSION_RE,
            ),
        )
        object.__setattr__(
            self,
            "allowed_providers",
            _unique_tokens(
                self.allowed_providers,
                "allowed_providers",
                max_items=MAX_PROVIDERS,
                max_bytes=MAX_ID_BYTES,
                pattern=_PROVIDER_RE,
            ),
        )
        object.__setattr__(self, "local_only", _bool(self.local_only, "local_only"))
        max_bytes = _positive_int(self.max_bytes, "max_bytes")
        if max_bytes > ABSOLUTE_MAX_BYTES:
            raise DisclosurePolicyError(
                f"max_bytes exceeds {ABSOLUTE_MAX_BYTES}", reason_code="bounds"
            )
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(
            self,
            "require_secret_scan",
            _bool(self.require_secret_scan, "require_secret_scan"),
        )
        if not self.require_secret_scan:
            raise DisclosurePolicyError(
                "require_secret_scan must be true", reason_code="malformed"
            )
        if (
            confidentiality in (ConfidentialityClass.CONFIDENTIAL, ConfidentialityClass.SECRET)
            and not self.local_only
        ):
            raise DisclosurePolicyError(
                "confidential and secret classes require local_only",
                reason_code="local_only",
            )

    def _payload(self) -> dict[str, Any]:
        confidentiality = self.confidentiality
        assert isinstance(confidentiality, ConfidentialityClass)
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "confidentiality": confidentiality.value,
            "exclusions": list(self.exclusions),
            "allowed_providers": list(self.allowed_providers),
            "local_only": self.local_only,
            "max_bytes": self.max_bytes,
            "require_secret_scan": self.require_secret_scan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceDisclosurePolicy":
        if not isinstance(payload, Mapping):
            raise DisclosurePolicyError(
                "source disclosure policy payload must be an object",
                reason_code="malformed",
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="source disclosure policy",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_POLICY_FIELDS)),
            artifact_name="source disclosure policy",
        )
        result = cls(
            policy_id=payload.get("policy_id", ""),
            confidentiality=payload.get("confidentiality", ""),
            exclusions=payload.get("exclusions", ()),
            allowed_providers=payload.get("allowed_providers", ()),
            local_only=payload.get("local_only", True),
            max_bytes=payload.get("max_bytes", DEFAULT_MAX_BYTES),
            require_secret_scan=payload.get("require_secret_scan", True),
        )
        _claimed_identity(
            payload, result.content_id, artifact_name="source disclosure policy"
        )
        return result


_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "policy_id",
    "policy_content_id",
    "context_pack_content_id",
    "provider_id",
    "provider_locality",
    "confidentiality",
    "payload_bytes",
    "verdict",
    "reason_code",
    "secret_kinds",
    "excluded_matches",
    "principal_content_id",
)


@dataclass(frozen=True)
class DisclosureDecision(CanonicalContract):
    """Frozen content-addressed disclosure decision bound to a ContextPack @1."""

    SCHEMA: ClassVar[str] = DISCLOSURE_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = DISCLOSURE_DECISION_INTERFACE

    policy_id: str
    policy_content_id: str
    context_pack_content_id: str
    provider_id: str
    provider_locality: ProviderLocality | str
    confidentiality: ConfidentialityClass | str
    payload_bytes: int
    verdict: DisclosureVerdict | str
    reason_code: str
    secret_kinds: tuple[str, ...] = ()
    excluded_matches: tuple[str, ...] = ()
    principal_content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", pattern=_POLICY_RE)
        )
        object.__setattr__(
            self,
            "policy_content_id",
            _text(self.policy_content_id, "policy_content_id", pattern=_ID_RE),
        )
        object.__setattr__(
            self,
            "context_pack_content_id",
            _text(
                self.context_pack_content_id,
                "context_pack_content_id",
                pattern=_ID_RE,
            ),
        )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, "provider_id", pattern=_PROVIDER_RE),
        )
        object.__setattr__(
            self,
            "provider_locality",
            _enum(self.provider_locality, ProviderLocality, "provider_locality"),
        )
        object.__setattr__(
            self,
            "confidentiality",
            _enum(self.confidentiality, ConfidentialityClass, "confidentiality"),
        )
        object.__setattr__(
            self, "payload_bytes", _nonnegative_int(self.payload_bytes, "payload_bytes")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, DisclosureVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(
                self.reason_code,
                "reason_code",
                max_bytes=MAX_REASON_BYTES,
                pattern=_ID_RE,
            ),
        )
        kinds: list[str] = []
        seen: set[str] = set()
        for item in self.secret_kinds:
            kind = _kind_name(_enum(item, SecretKind, "secret_kinds"))
            if kind in seen:
                raise DisclosurePolicyError(
                    "secret_kinds must not contain duplicates", reason_code="malformed"
                )
            seen.add(kind)
            kinds.append(kind)
        object.__setattr__(self, "secret_kinds", tuple(kinds))
        object.__setattr__(
            self,
            "excluded_matches",
            _unique_tokens(
                self.excluded_matches,
                "excluded_matches",
                max_items=MAX_EXCLUSIONS,
                max_bytes=MAX_EXCLUSION_BYTES,
                pattern=_EXCLUSION_RE,
            ),
        )
        object.__setattr__(
            self,
            "principal_content_id",
            _text(
                self.principal_content_id,
                "principal_content_id",
                required=False,
                pattern=_ID_RE,
            ),
        )

    @property
    def permitted(self) -> bool:
        return self.verdict is DisclosureVerdict.PERMIT

    def _payload(self) -> dict[str, Any]:
        locality = self.provider_locality
        assert isinstance(locality, ProviderLocality)
        confidentiality = self.confidentiality
        assert isinstance(confidentiality, ConfidentialityClass)
        verdict = self.verdict
        assert isinstance(verdict, DisclosureVerdict)
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "policy_content_id": self.policy_content_id,
            "context_pack_content_id": self.context_pack_content_id,
            "provider_id": self.provider_id,
            "provider_locality": locality.value,
            "confidentiality": confidentiality.value,
            "payload_bytes": self.payload_bytes,
            "verdict": verdict.value,
            "reason_code": self.reason_code,
            "secret_kinds": list(self.secret_kinds),
            "excluded_matches": list(self.excluded_matches),
            "principal_content_id": self.principal_content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DisclosureDecision":
        if not isinstance(payload, Mapping):
            raise DisclosurePolicyError(
                "disclosure decision payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="disclosure decision",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_DECISION_FIELDS)),
            artifact_name="disclosure decision",
        )
        result = cls(
            policy_id=payload.get("policy_id", ""),
            policy_content_id=payload.get("policy_content_id", ""),
            context_pack_content_id=payload.get("context_pack_content_id", ""),
            provider_id=payload.get("provider_id", ""),
            provider_locality=payload.get("provider_locality", ""),
            confidentiality=payload.get("confidentiality", ""),
            payload_bytes=payload.get("payload_bytes"),
            verdict=payload.get("verdict", ""),
            reason_code=payload.get("reason_code", ""),
            secret_kinds=tuple(payload.get("secret_kinds") or ()),
            excluded_matches=tuple(payload.get("excluded_matches") or ()),
            principal_content_id=payload.get("principal_content_id", ""),
        )
        _claimed_identity(
            payload, result.content_id, artifact_name="disclosure decision"
        )
        return result


def load_policy(
    source: SourceDisclosurePolicy | Mapping[str, Any] | str | Path | None,
) -> SourceDisclosurePolicy:
    """Load a frozen SourceDisclosurePolicy@1.  Missing policy fails closed."""

    if source is None:
        raise PolicyMissingError(
            "source disclosure policy is required", reason_code="policy_missing"
        )
    if isinstance(source, SourceDisclosurePolicy):
        return source
    if isinstance(source, Mapping):
        return SourceDisclosurePolicy.from_dict(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PolicyMissingError(
                "source disclosure policy is required", reason_code="policy_missing"
            ) from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DisclosurePolicyError(
                "source disclosure policy JSON is malformed", reason_code="malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DisclosurePolicyError(
                "source disclosure policy payload must be an object",
                reason_code="malformed",
            )
        return SourceDisclosurePolicy.from_dict(payload)
    raise PolicyMissingError(
        "source disclosure policy is required", reason_code="policy_missing"
    )


def classify_provider_locality(provider_id: str) -> ProviderLocality:
    """Classify a provider id as local or external.  Unknown is external."""

    pid = _text(provider_id, "provider_id", pattern=_PROVIDER_RE)
    lowered = pid.casefold()
    for prefix in _LOCAL_PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            return ProviderLocality.LOCAL
    return ProviderLocality.EXTERNAL


def _scan_text(text: str, found: set[str]) -> None:
    if _API_KEY_TEXT_RE.search(text):
        found.add(SecretKind.API_KEY.value)
    if _BEARER_TEXT_RE.search(text):
        found.add(SecretKind.BEARER.value)
    if _PEM_TEXT_RE.search(text):
        found.add(SecretKind.PRIVATE_KEY_PEM.value)


def _scan_bytes(data: bytes, found: set[str]) -> None:
    if _API_KEY_BYTES_RE.search(data):
        found.add(SecretKind.API_KEY.value)
    if _BEARER_BYTES_RE.search(data):
        found.add(SecretKind.BEARER.value)
    if _PEM_BYTES_RE.search(data):
        found.add(SecretKind.PRIVATE_KEY_PEM.value)


def _scan_walk(value: Any, found: set[str], *, depth: int = 0) -> None:
    if depth > 32:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _normalized_key(key) in {
                "api_key",
                "apikey",
            }:
                found.add(SecretKind.API_KEY.value)
            elif isinstance(key, str):
                _scan_text(key, found)
            _scan_walk(item, found, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _scan_walk(item, found, depth=depth + 1)
        return
    if isinstance(value, str):
        _scan_text(value, found)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        _scan_bytes(bytes(value), found)


def scan_secret_material(payload: Any) -> tuple[str, ...]:
    """Scan text or bytes for api_key, bearer, and private-key PEM shapes."""

    found: set[str] = set()
    if isinstance(payload, Path):
        try:
            data = payload.read_bytes()
        except OSError:
            return ()
        _scan_bytes(data, found)
        try:
            _scan_text(data.decode("utf-8"), found)
        except UnicodeDecodeError:
            _scan_text(data.decode("utf-8", errors="replace"), found)
        return tuple(sorted(found))
    _scan_walk(payload, found)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            _scan_text(bytes(payload).decode("utf-8"), found)
        except UnicodeDecodeError:
            pass
    return tuple(sorted(found))


def _encode_json(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DisclosurePolicyError(
            f"{name} is not a bounded JSON-compatible payload",
            reason_code="malformed",
        ) from exc


def _payload_bytes(payload: Any) -> tuple[Any, int, tuple[str, ...]]:
    paths: list[str] = []
    if isinstance(payload, Path):
        paths.append(payload.name)
        paths.append(payload.as_posix())
        try:
            data = payload.read_bytes()
        except OSError as exc:
            raise DisclosurePolicyError(
                "payload path could not be read", reason_code="malformed"
            ) from exc
        return data, len(data), tuple(paths)
    if isinstance(payload, str):
        return payload, len(payload.encode("utf-8")), ()
    if isinstance(payload, (bytes, bytearray, memoryview)):
        data = bytes(payload)
        return data, len(data), ()
    if isinstance(payload, Mapping):
        for key in payload:
            if isinstance(key, str):
                paths.append(key)
        encoded = _encode_json(dict(payload), "payload")
        return payload, len(encoded), tuple(paths)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        encoded = _encode_json(list(payload), "payload")
        return payload, len(encoded), ()
    raise DisclosurePolicyError(
        "payload must be text, bytes, a mapping, or a local path",
        reason_code="malformed",
    )


def _matches_exclusion(path: str, exclusion: str) -> bool:
    if path == exclusion:
        return True
    marker = exclusion.rstrip("/")
    if not marker:
        return False
    name = path.rsplit("/", 1)[-1]
    if name == marker:
        return True
    if path.startswith(marker + "/"):
        return True
    if exclusion.endswith("/") and f"/{marker}/" in f"/{path.rstrip('/')}/":
        return True
    return False


def _exclusion_hits(
    paths: Sequence[str], exclusions: Sequence[str]
) -> tuple[str, ...]:
    hits: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for exclusion in exclusions:
            if _matches_exclusion(path, exclusion) and exclusion not in seen:
                seen.add(exclusion)
                hits.append(exclusion)
    return tuple(hits)


def _pack_mapping(pack: Any) -> Mapping[str, Any] | None:
    if isinstance(pack, Mapping):
        return pack
    to_dict = getattr(pack, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def bind_context_pack_identity(
    context_pack: Any = None,
    context_pack_content_id: str | None = None,
) -> str:
    """Return the exact ContextPack content identity or fail closed."""

    claimed = context_pack_content_id
    if claimed is None and context_pack is None:
        raise ContextPackIdentityError(
            "ContextPack content identity is required",
            reason_code="identity_mismatch",
        )
    if isinstance(claimed, str):
        claimed = claimed.strip() or None
    elif claimed is not None:
        raise ContextPackIdentityError(
            "ContextPack content identity must be a string",
            reason_code="malformed",
        )

    actual: str | None = None
    if context_pack is not None:
        direct = getattr(context_pack, "content_id", None) or getattr(
            context_pack, "cid", None
        )
        payload = _pack_mapping(context_pack)
        if payload is not None:
            body = {key: value for key, value in payload.items() if key not in _IDENTITY_KEYS}
            actual = content_identity(body)
            for name in ("content_id", "cid", "identity", "canonical_id"):
                embedded = payload.get(name)
                if embedded not in (None, "") and embedded != actual:
                    raise ContextPackIdentityError(
                        "ContextPack content identity does not match payload",
                        reason_code="identity_mismatch",
                    )
        elif isinstance(direct, str) and direct.strip():
            actual = direct.strip()
        elif isinstance(context_pack, str) and context_pack.strip():
            actual = context_pack.strip()
        else:
            raise ContextPackIdentityError(
                "ContextPack content identity is required",
                reason_code="identity_mismatch",
            )

    if claimed and actual and claimed != actual:
        raise ContextPackIdentityError(
            "ContextPack content identity does not match payload",
            reason_code="identity_mismatch",
        )
    identity = actual or claimed
    if not identity:
        raise ContextPackIdentityError(
            "ContextPack content identity is required",
            reason_code="identity_mismatch",
        )
    return _text(identity, "context_pack_content_id", pattern=_ID_RE)


def _context_pack_exclusions(context_pack: Any) -> tuple[str, ...]:
    payload = _pack_mapping(context_pack)
    if payload is None:
        return ()
    raw = payload.get("exclusions") or ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token or token in seen:
                continue
            if _EXCLUSION_RE.fullmatch(token) is None:
                continue
            seen.add(token)
            result.append(token)
        return tuple(result)
    return ()


def _decision(
    policy: SourceDisclosurePolicy,
    *,
    context_pack_content_id: str,
    provider_id: str,
    provider_locality: ProviderLocality,
    confidentiality: ConfidentialityClass,
    payload_bytes: int,
    verdict: DisclosureVerdict,
    reason_code: str,
    secret_kinds: tuple[str, ...] = (),
    excluded_matches: tuple[str, ...] = (),
    principal_content_id: str = "",
) -> DisclosureDecision:
    return DisclosureDecision(
        policy_id=policy.policy_id,
        policy_content_id=policy.content_id,
        context_pack_content_id=context_pack_content_id,
        provider_id=provider_id,
        provider_locality=provider_locality,
        confidentiality=confidentiality,
        payload_bytes=payload_bytes,
        verdict=verdict,
        reason_code=reason_code,
        secret_kinds=secret_kinds,
        excluded_matches=excluded_matches,
        principal_content_id=principal_content_id,
    )


def evaluate_disclosure(
    policy: SourceDisclosurePolicy | Mapping[str, Any] | str | Path | None,
    *,
    payload: Any,
    provider_id: str,
    context_pack: Any = None,
    context_pack_content_id: str | None = None,
    confidentiality: ConfidentialityClass | str | None = None,
    paths: Sequence[str] | None = None,
    principal: ExternalPrincipal | None = None,
) -> DisclosureDecision:
    """Evaluate disclosure against SourceDisclosurePolicy@1.

    Missing policy fails closed.  A permit binds the exact ContextPack
    content identity and the policy content identity.
    """

    bound_policy = load_policy(policy)
    provider = _text(provider_id, "provider_id", pattern=_PROVIDER_RE)
    locality = classify_provider_locality(provider)
    pack_id = bind_context_pack_identity(context_pack, context_pack_content_id)
    material, size, payload_paths = _payload_bytes(payload)
    principal_id = ""
    if principal is not None:
        if not isinstance(principal, ExternalPrincipal):
            raise DisclosurePolicyError(
                "principal must be an ExternalPrincipal@1 record",
                reason_code="malformed",
            )
        if principal.disclosure_policy_id != bound_policy.policy_id:
            raise DisclosurePolicyError(
                "principal disclosure_policy_id does not match policy_id",
                reason_code="policy_mismatch",
            )
        principal_id = principal.content_id

    requested = (
        bound_policy.confidentiality
        if confidentiality is None
        else _enum(confidentiality, ConfidentialityClass, "confidentiality")
    )
    assert isinstance(requested, ConfidentialityClass)
    assert isinstance(bound_policy.confidentiality, ConfidentialityClass)

    extra_paths = tuple(paths or ())
    pack_exclusions = _context_pack_exclusions(context_pack)
    excluded = _exclusion_hits(
        payload_paths + extra_paths, bound_policy.exclusions + pack_exclusions
    )
    secrets = scan_secret_material(payload if isinstance(payload, Path) else material)

    def deny(reason: str, *, secret_kinds: tuple[str, ...] = (), excluded_matches: tuple[str, ...] = ()) -> DisclosureDecision:
        return _decision(
            bound_policy,
            context_pack_content_id=pack_id,
            provider_id=provider,
            provider_locality=locality,
            confidentiality=requested,
            payload_bytes=size,
            verdict=DisclosureVerdict.DENY,
            reason_code=reason,
            secret_kinds=secret_kinds,
            excluded_matches=excluded_matches,
            principal_content_id=principal_id,
        )

    if requested.rank > bound_policy.confidentiality.rank:
        return deny("confidentiality")
    if size > bound_policy.max_bytes:
        return deny("byte_limit")
    if provider not in bound_policy.allowed_providers:
        return deny("provider_not_allowlisted")
    if bound_policy.local_only and locality is ProviderLocality.EXTERNAL:
        return deny("local_only")
    if excluded:
        return deny("excluded", excluded_matches=excluded)
    if secrets:
        return deny("secret_material", secret_kinds=secrets)
    return _decision(
        bound_policy,
        context_pack_content_id=pack_id,
        provider_id=provider,
        provider_locality=locality,
        confidentiality=requested,
        payload_bytes=size,
        verdict=DisclosureVerdict.PERMIT,
        reason_code="bound",
        principal_content_id=principal_id,
    )


def admit_disclosure(
    policy: SourceDisclosurePolicy | Mapping[str, Any] | str | Path | None,
    *,
    payload: Any,
    provider_id: str,
    context_pack: Any = None,
    context_pack_content_id: str | None = None,
    confidentiality: ConfidentialityClass | str | None = None,
    paths: Sequence[str] | None = None,
    principal: ExternalPrincipal | None = None,
) -> DisclosureDecision:
    """Permit disclosure or raise a fail-closed typed error."""

    decision = evaluate_disclosure(
        policy,
        payload=payload,
        provider_id=provider_id,
        context_pack=context_pack,
        context_pack_content_id=context_pack_content_id,
        confidentiality=confidentiality,
        paths=paths,
        principal=principal,
    )
    if decision.verdict is DisclosureVerdict.PERMIT:
        return decision
    reason = decision.reason_code
    if reason == "secret_material":
        raise SecretMaterialError(
            "secret-shaped material cannot be disclosed",
            reason_code=reason,
        )
    if reason == "identity_mismatch":
        raise ContextPackIdentityError(
            "ContextPack content identity does not match payload",
            reason_code=reason,
        )
    raise DisclosurePolicyError(
        f"source disclosure denied ({reason})",
        reason_code=reason,
    )


__all__ = (
    "ABSOLUTE_MAX_BYTES",
    "CONTRACT_VERSION",
    "DEFAULT_MAX_BYTES",
    "DISCLOSURE_DECISION_INTERFACE",
    "DISCLOSURE_DECISION_SCHEMA",
    "ConfidentialityClass",
    "ContextPackIdentityError",
    "DisclosureDecision",
    "DisclosurePolicyError",
    "DisclosureVerdict",
    "PolicyMissingError",
    "ProviderLocality",
    "SCHEMA_VERSION",
    "SOURCE_DISCLOSURE_POLICY_INTERFACE",
    "SOURCE_DISCLOSURE_POLICY_SCHEMA",
    "SecretKind",
    "SecretMaterialError",
    "SourceDisclosurePolicy",
    "admit_disclosure",
    "bind_context_pack_identity",
    "classify_provider_locality",
    "evaluate_disclosure",
    "load_policy",
    "scan_secret_material",
)
