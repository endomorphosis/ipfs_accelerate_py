"""Effect-bound external-agent principal and capability decisions (EAAEF-030).

Frozen, content-addressed ``ExternalPrincipal@1`` and ``CapabilityDecision@1``
records bind a caller to one repository, run, exact effects, expiry, autonomy
and resource ceilings, disclosure/provider policy, and a bounded nonce.

A CID, imported history, prompt, payment, or commit never grants authority.
Unknown effects fail closed.  Expiry is required and must be strictly in the
future relative to the ``now_ms`` supplied at decision time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)

CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_PRINCIPAL_INTERFACE: Final[str] = "ExternalPrincipal@1"
CAPABILITY_DECISION_INTERFACE: Final[str] = "CapabilityDecision@1"
RESOURCE_CEILINGS_INTERFACE: Final[str] = "ResourceCeilings@1"

EXTERNAL_PRINCIPAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-principal@1"
)
CAPABILITY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/capability-decision@1"
)
RESOURCE_CEILINGS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/resource-ceilings@1"
)

MAX_ID_BYTES: Final[int] = 256
MAX_NONCE_BYTES: Final[int] = 64
MIN_NONCE_BYTES: Final[int] = 8
MAX_EFFECTS: Final[int] = 32
MAX_REASON_BYTES: Final[int] = 256

_DID_RE: Final[re.Pattern[str]] = re.compile(
    r"^did:[a-z0-9]+:[A-Za-z0-9._:%-]+$"
)
_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)
_NONCE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._:-]+$")
_POLICY_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)

TEnum = TypeVar("TEnum", bound=Enum)


class EffectName(str, Enum):
    """Closed allowlist of effect names that may be bound into a principal."""

    INSPECT_REPOSITORY = "inspect_repository"
    WRITE_SUPERVISOR_STATE = "write_supervisor_state"
    CREATE_ISOLATED_WORKTREE = "create_isolated_worktree"
    EDIT_ISOLATED_WORKTREE = "edit_isolated_worktree"
    RUN_VALIDATION = "run_validation"
    LAUNCH_LOCAL_PROCESS = "launch_local_process"
    SUBMIT_MERGE_PROPOSAL = "submit_merge_proposal"
    MERGE = "merge"
    PUSH = "push"
    DEPLOY = "deploy"
    DESTRUCTIVE_CLEANUP = "destructive_cleanup"
    INSTALL = "install"
    NETWORK = "network"
    SECRET = "secret"
    DISCLOSURE = "disclosure"
    PUBLICATION = "publication"


ALLOWED_EFFECTS: Final[frozenset[str]] = frozenset(item.value for item in EffectName)


class AutonomyCeiling(str, Enum):
    """Closed autonomy levels.  Unknown values fail closed."""

    PREVIEW = "preview"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


class AuthoritySource(str, Enum):
    """How a bind attempt claims to have obtained authority.

    Only an authenticated principal may grant effects.  A CID, imported
    history, prompt, payment, or commit is evidence or context, never authority.
    """

    AUTHENTICATED_PRINCIPAL = "authenticated_principal"
    CID = "cid"
    IMPORTED_HISTORY = "imported_history"
    PROMPT = "prompt"
    PAYMENT = "payment"
    COMMIT = "commit"


FORBIDDEN_AUTHORITY_SOURCES: Final[frozenset[AuthoritySource]] = frozenset(
    {
        AuthoritySource.CID,
        AuthoritySource.IMPORTED_HISTORY,
        AuthoritySource.PROMPT,
        AuthoritySource.PAYMENT,
        AuthoritySource.COMMIT,
    }
)

_AUTHORITY_REASON_CODES: Final[Mapping[AuthoritySource, str]] = {
    AuthoritySource.CID: "cid_is_not_authority",
    AuthoritySource.IMPORTED_HISTORY: "history_is_not_authority",
    AuthoritySource.PROMPT: "prompt_is_not_authority",
    AuthoritySource.PAYMENT: "payment_is_not_authority",
    AuthoritySource.COMMIT: "commit_is_not_authority",
}

_INFERENCE_ALIASES: Final[Mapping[str, AuthoritySource]] = {
    "cid": AuthoritySource.CID,
    "content_id": AuthoritySource.CID,
    "content_identity": AuthoritySource.CID,
    "history": AuthoritySource.IMPORTED_HISTORY,
    "imported_history": AuthoritySource.IMPORTED_HISTORY,
    "imported": AuthoritySource.IMPORTED_HISTORY,
    "prompt": AuthoritySource.PROMPT,
    "payment": AuthoritySource.PAYMENT,
    "commit": AuthoritySource.COMMIT,
}


class CapabilityVerdict(str, Enum):
    PERMIT = "permit"
    DENY = "deny"


class PrincipalAuthorityError(ContractValidationError):
    """Malformed, expired, or inferred external-agent authority."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PrincipalExpiryError(PrincipalAuthorityError):
    """Expiry is missing or not strictly in the future at decision time."""


class UnknownEffectError(PrincipalAuthorityError):
    """An effect name is outside the closed allowlist."""


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
        raise PrincipalAuthorityError(
            f"{name} must be a string", reason_code="malformed"
        )
    else:
        result = value.strip()
    if required and not result:
        raise PrincipalAuthorityError(f"{name} is required", reason_code="malformed")
    if "\x00" in result:
        raise PrincipalAuthorityError(
            f"{name} must not contain NUL", reason_code="malformed"
        )
    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        raise PrincipalAuthorityError(
            f"{name} exceeds {max_bytes} UTF-8 bytes", reason_code="bounds"
        )
    if result and pattern is not None and pattern.fullmatch(result) is None:
        raise PrincipalAuthorityError(
            f"{name} is not a permitted identifier", reason_code="malformed"
        )
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrincipalAuthorityError(
            f"{name} must be a non-negative integer", reason_code="malformed"
        )
    if value < 0:
        raise PrincipalAuthorityError(
            f"{name} must be a non-negative integer", reason_code="malformed"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise PrincipalAuthorityError(
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
        raise PrincipalAuthorityError(
            f"{name} must be one of: {allowed}", reason_code="malformed"
        ) from exc


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Sequence[str], *, artifact_name: str
) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise PrincipalAuthorityError(
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
        raise PrincipalAuthorityError(
            f"{artifact_name} payload must be an object", reason_code="malformed"
        )
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        raise PrincipalAuthorityError(
            f"unsupported {artifact_name} schema {schema!r}; expected {expected_schema}",
            reason_code="unsupported_version",
        )
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        raise PrincipalAuthorityError(
            f"unsupported {artifact_name} interface {interface!r}; "
            f"expected {expected_interface}",
            reason_code="unsupported_version",
        )
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", CONTRACT_VERSION):
            raise PrincipalAuthorityError(
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
            raise PrincipalAuthorityError(
                f"{artifact_name} content identity does not match payload",
                reason_code="identity_mismatch",
            )


_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "identity",
        "canonical_id",
    }
)


def _principal_id(value: Any) -> str:
    text = _text(value, "principal_id", pattern=_DID_RE)
    return text


def _policy_id(value: Any, name: str) -> str:
    return _text(value, name, pattern=_POLICY_RE)


def _record_id(value: Any, name: str) -> str:
    return _text(value, name, pattern=_ID_RE)


def _nonce(value: Any) -> str:
    text = _text(value, "nonce", max_bytes=MAX_NONCE_BYTES, pattern=_NONCE_RE)
    size = len(text.encode("utf-8"))
    if size < MIN_NONCE_BYTES:
        raise PrincipalAuthorityError(
            f"nonce must be at least {MIN_NONCE_BYTES} UTF-8 bytes",
            reason_code="nonce_out_of_bounds",
        )
    return text


def _effect_name(value: Any) -> str:
    if isinstance(value, EffectName):
        return value.value
    if not isinstance(value, str):
        raise UnknownEffectError(
            "effect names must be strings from the closed allowlist",
            reason_code="unknown_effect",
        )
    name = value.strip()
    if name not in ALLOWED_EFFECTS:
        raise UnknownEffectError(
            f"unknown effect {name!r} fails closed",
            reason_code="unknown_effect",
        )
    return name


def _effects(value: Any, name: str = "exact_effects") -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PrincipalAuthorityError(
            f"{name} must be a sequence of effect names", reason_code="malformed"
        )
    if len(value) > MAX_EFFECTS:
        raise PrincipalAuthorityError(
            f"{name} exceeds {MAX_EFFECTS} items", reason_code="bounds"
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        effect = _effect_name(item)
        if effect in seen:
            raise PrincipalAuthorityError(
                f"{name} must not contain duplicate effects", reason_code="malformed"
            )
        seen.add(effect)
        result.append(effect)
    if not result:
        raise PrincipalAuthorityError(
            f"{name} must not be empty", reason_code="malformed"
        )
    return tuple(result)


def _authority_source(value: Any) -> AuthoritySource:
    if isinstance(value, AuthoritySource):
        source = value
    else:
        text = _text(str(getattr(value, "value", value)), "authority_source")
        alias = _INFERENCE_ALIASES.get(text.lower().replace("-", "_"))
        if alias is not None:
            source = alias
        else:
            source = _enum(text, AuthoritySource, "authority_source")
    if source in FORBIDDEN_AUTHORITY_SOURCES:
        reason = _AUTHORITY_REASON_CODES[source]
        raise PrincipalAuthorityError(
            _authority_rejection_message(source), reason_code=reason
        )
    return source


def _authority_rejection_message(source: AuthoritySource) -> str:
    if source is AuthoritySource.CID:
        return "a CID is not authority"
    if source is AuthoritySource.IMPORTED_HISTORY:
        return "imported history is not authority"
    if source is AuthoritySource.PROMPT:
        return "a prompt is not authority"
    if source is AuthoritySource.PAYMENT:
        return "a payment is not authority"
    if source is AuthoritySource.COMMIT:
        return "a commit is not authority"
    return f"{source.value} is not authority"


_INFERENCE_KWARGS: Final[tuple[tuple[str, AuthoritySource], ...]] = (
    ("cid", AuthoritySource.CID),
    ("content_id", AuthoritySource.CID),
    ("content_identity", AuthoritySource.CID),
    ("imported_history", AuthoritySource.IMPORTED_HISTORY),
    ("history", AuthoritySource.IMPORTED_HISTORY),
    ("prompt", AuthoritySource.PROMPT),
    ("payment", AuthoritySource.PAYMENT),
    ("commit", AuthoritySource.COMMIT),
)


def _reject_inferred_authority(
    *,
    authority_source: Any,
    inferences: Mapping[str, Any],
) -> AuthoritySource:
    for name, source in _INFERENCE_KWARGS:
        if inferences.get(name) is not None:
            raise PrincipalAuthorityError(
                _authority_rejection_message(source),
                reason_code=_AUTHORITY_REASON_CODES[source],
            )
    return _authority_source(authority_source)


@dataclass(frozen=True)
class ResourceCeilings(CanonicalContract):
    """Integer CPU, RAM, disk and timeout ceilings bound into a principal."""

    SCHEMA: ClassVar[str] = RESOURCE_CEILINGS_SCHEMA
    INTERFACE: ClassVar[str] = RESOURCE_CEILINGS_INTERFACE

    cpu: int
    ram: int
    disk: int
    timeout: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", _nonnegative_int(self.cpu, "cpu"))
        object.__setattr__(self, "ram", _nonnegative_int(self.ram, "ram"))
        object.__setattr__(self, "disk", _nonnegative_int(self.disk, "disk"))
        object.__setattr__(self, "timeout", _nonnegative_int(self.timeout, "timeout"))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "cpu": self.cpu,
            "ram": self.ram,
            "disk": self.disk,
            "timeout": self.timeout,
        }

    def as_mapping(self) -> dict[str, int]:
        return {
            "cpu": self.cpu,
            "ram": self.ram,
            "disk": self.disk,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceCeilings":
        if isinstance(payload, ResourceCeilings):
            return payload
        if not isinstance(payload, Mapping):
            raise PrincipalAuthorityError(
                "resource_ceilings must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="resource ceilings",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union({"cpu", "ram", "disk", "timeout"})),
            artifact_name="resource ceilings",
        )
        result = cls(
            cpu=payload.get("cpu"),
            ram=payload.get("ram"),
            disk=payload.get("disk"),
            timeout=payload.get("timeout"),
        )
        _claimed_identity(payload, result.content_id, artifact_name="resource ceilings")
        return result


def _resource_ceilings(value: Any) -> ResourceCeilings:
    if isinstance(value, ResourceCeilings):
        return value
    if isinstance(value, Mapping):
        if "schema" in value or "interface" in value:
            return ResourceCeilings.from_dict(value)
        extra = set(value).difference({"cpu", "ram", "disk", "timeout"})
        if extra:
            raise PrincipalAuthorityError(
                "resource_ceilings contains unknown keys; rebuild its canonical payload",
                reason_code="malformed",
            )
        missing = {"cpu", "ram", "disk", "timeout"}.difference(value)
        if missing:
            raise PrincipalAuthorityError(
                "resource_ceilings must bind cpu, ram, disk and timeout",
                reason_code="malformed",
            )
        return ResourceCeilings(
            cpu=value["cpu"],
            ram=value["ram"],
            disk=value["disk"],
            timeout=value["timeout"],
        )
    raise PrincipalAuthorityError(
        "resource_ceilings must be a ResourceCeilings or mapping",
        reason_code="malformed",
    )


_PRINCIPAL_FIELDS: Final[tuple[str, ...]] = (
    "principal_id",
    "repository_id",
    "run_id",
    "exact_effects",
    "expires_at_ms",
    "autonomy_ceiling",
    "resource_ceilings",
    "disclosure_policy_id",
    "provider_policy_id",
    "nonce",
)


@dataclass(frozen=True)
class ExternalPrincipal(CanonicalContract):
    """Frozen content-addressed caller identity and effect-bound grant @1."""

    SCHEMA: ClassVar[str] = EXTERNAL_PRINCIPAL_SCHEMA
    INTERFACE: ClassVar[str] = EXTERNAL_PRINCIPAL_INTERFACE

    principal_id: str
    repository_id: str
    run_id: str
    exact_effects: tuple[str, ...]
    expires_at_ms: int
    autonomy_ceiling: AutonomyCeiling | str
    resource_ceilings: ResourceCeilings | Mapping[str, int]
    disclosure_policy_id: str
    provider_policy_id: str
    nonce: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_id", _principal_id(self.principal_id))
        object.__setattr__(self, "repository_id", _record_id(self.repository_id, "repository_id"))
        object.__setattr__(self, "run_id", _record_id(self.run_id, "run_id"))
        object.__setattr__(self, "exact_effects", _effects(self.exact_effects))
        object.__setattr__(
            self, "expires_at_ms", _positive_int(self.expires_at_ms, "expires_at_ms")
        )
        object.__setattr__(
            self,
            "autonomy_ceiling",
            _enum(self.autonomy_ceiling, AutonomyCeiling, "autonomy_ceiling"),
        )
        object.__setattr__(self, "resource_ceilings", _resource_ceilings(self.resource_ceilings))
        object.__setattr__(
            self,
            "disclosure_policy_id",
            _policy_id(self.disclosure_policy_id, "disclosure_policy_id"),
        )
        object.__setattr__(
            self,
            "provider_policy_id",
            _policy_id(self.provider_policy_id, "provider_policy_id"),
        )
        object.__setattr__(self, "nonce", _nonce(self.nonce))

    def _payload(self) -> dict[str, Any]:
        ceilings = self.resource_ceilings
        assert isinstance(ceilings, ResourceCeilings)
        ceiling = self.autonomy_ceiling
        assert isinstance(ceiling, AutonomyCeiling)
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "principal_id": self.principal_id,
            "repository_id": self.repository_id,
            "run_id": self.run_id,
            "exact_effects": list(self.exact_effects),
            "expires_at_ms": self.expires_at_ms,
            "autonomy_ceiling": ceiling.value,
            "resource_ceilings": ceilings.as_mapping(),
            "disclosure_policy_id": self.disclosure_policy_id,
            "provider_policy_id": self.provider_policy_id,
            "nonce": self.nonce,
        }

    def permits(self, effect: str | EffectName) -> bool:
        return _effect_name(effect) in self.exact_effects

    def bind(
        self,
        *,
        now_ms: int,
        requested_effects: Sequence[str | EffectName] | None = None,
        authority_source: str | AuthoritySource = AuthoritySource.AUTHENTICATED_PRINCIPAL,
        cid: Any = None,
        imported_history: Any = None,
        history: Any = None,
        prompt: Any = None,
        payment: Any = None,
        commit: Any = None,
        content_id: Any = None,
        content_identity: Any = None,
    ) -> "CapabilityDecision":
        return bind_capability(
            self,
            now_ms=now_ms,
            requested_effects=requested_effects,
            authority_source=authority_source,
            cid=cid,
            imported_history=imported_history,
            history=history,
            prompt=prompt,
            payment=payment,
            commit=commit,
            content_id=content_id,
            content_identity=content_identity,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalPrincipal":
        if not isinstance(payload, Mapping):
            raise PrincipalAuthorityError(
                "external principal payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="external principal",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_PRINCIPAL_FIELDS)),
            artifact_name="external principal",
        )
        result = cls(
            principal_id=payload.get("principal_id", ""),
            repository_id=payload.get("repository_id", ""),
            run_id=payload.get("run_id", ""),
            exact_effects=payload.get("exact_effects", ()),
            expires_at_ms=payload.get("expires_at_ms"),
            autonomy_ceiling=payload.get("autonomy_ceiling", ""),
            resource_ceilings=payload.get("resource_ceilings") or {},
            disclosure_policy_id=payload.get("disclosure_policy_id", ""),
            provider_policy_id=payload.get("provider_policy_id", ""),
            nonce=payload.get("nonce", ""),
        )
        _claimed_identity(payload, result.content_id, artifact_name="external principal")
        return result


_DECISION_FIELDS: Final[tuple[str, ...]] = _PRINCIPAL_FIELDS + (
    "principal_content_id",
    "granted_effects",
    "decided_at_ms",
    "verdict",
    "reason_code",
    "authority_source",
)


@dataclass(frozen=True)
class CapabilityDecision(CanonicalContract):
    """Frozen content-addressed effect-bound capability decision @1."""

    SCHEMA: ClassVar[str] = CAPABILITY_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = CAPABILITY_DECISION_INTERFACE

    principal_id: str
    repository_id: str
    run_id: str
    exact_effects: tuple[str, ...]
    expires_at_ms: int
    autonomy_ceiling: AutonomyCeiling | str
    resource_ceilings: ResourceCeilings | Mapping[str, int]
    disclosure_policy_id: str
    provider_policy_id: str
    nonce: str
    principal_content_id: str
    granted_effects: tuple[str, ...]
    decided_at_ms: int
    verdict: CapabilityVerdict | str = CapabilityVerdict.PERMIT
    reason_code: str = "bound"
    authority_source: AuthoritySource | str = AuthoritySource.AUTHENTICATED_PRINCIPAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_id", _principal_id(self.principal_id))
        object.__setattr__(self, "repository_id", _record_id(self.repository_id, "repository_id"))
        object.__setattr__(self, "run_id", _record_id(self.run_id, "run_id"))
        object.__setattr__(self, "exact_effects", _effects(self.exact_effects))
        object.__setattr__(
            self, "expires_at_ms", _positive_int(self.expires_at_ms, "expires_at_ms")
        )
        object.__setattr__(
            self,
            "autonomy_ceiling",
            _enum(self.autonomy_ceiling, AutonomyCeiling, "autonomy_ceiling"),
        )
        object.__setattr__(self, "resource_ceilings", _resource_ceilings(self.resource_ceilings))
        object.__setattr__(
            self,
            "disclosure_policy_id",
            _policy_id(self.disclosure_policy_id, "disclosure_policy_id"),
        )
        object.__setattr__(
            self,
            "provider_policy_id",
            _policy_id(self.provider_policy_id, "provider_policy_id"),
        )
        object.__setattr__(self, "nonce", _nonce(self.nonce))
        object.__setattr__(
            self,
            "principal_content_id",
            _text(self.principal_content_id, "principal_content_id", pattern=_ID_RE),
        )
        object.__setattr__(
            self, "granted_effects", _effects(self.granted_effects, "granted_effects")
        )
        extra = set(self.granted_effects).difference(self.exact_effects)
        if extra:
            raise PrincipalAuthorityError(
                "granted_effects must be a subset of exact_effects",
                reason_code="effect_not_granted",
            )
        object.__setattr__(
            self, "decided_at_ms", _nonnegative_int(self.decided_at_ms, "decided_at_ms")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, CapabilityVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", max_bytes=MAX_REASON_BYTES, pattern=_ID_RE),
        )
        source = _enum(self.authority_source, AuthoritySource, "authority_source")
        if source in FORBIDDEN_AUTHORITY_SOURCES:
            raise PrincipalAuthorityError(
                _authority_rejection_message(source),
                reason_code=_AUTHORITY_REASON_CODES[source],
            )
        object.__setattr__(self, "authority_source", source)

    @property
    def permitted(self) -> bool:
        return self.verdict is CapabilityVerdict.PERMIT

    def permits(self, effect: str | EffectName) -> bool:
        return self.permitted and _effect_name(effect) in self.granted_effects

    def _payload(self) -> dict[str, Any]:
        ceilings = self.resource_ceilings
        assert isinstance(ceilings, ResourceCeilings)
        ceiling = self.autonomy_ceiling
        assert isinstance(ceiling, AutonomyCeiling)
        verdict = self.verdict
        assert isinstance(verdict, CapabilityVerdict)
        source = self.authority_source
        assert isinstance(source, AuthoritySource)
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "principal_id": self.principal_id,
            "repository_id": self.repository_id,
            "run_id": self.run_id,
            "exact_effects": list(self.exact_effects),
            "expires_at_ms": self.expires_at_ms,
            "autonomy_ceiling": ceiling.value,
            "resource_ceilings": ceilings.as_mapping(),
            "disclosure_policy_id": self.disclosure_policy_id,
            "provider_policy_id": self.provider_policy_id,
            "nonce": self.nonce,
            "principal_content_id": self.principal_content_id,
            "granted_effects": list(self.granted_effects),
            "decided_at_ms": self.decided_at_ms,
            "verdict": verdict.value,
            "reason_code": self.reason_code,
            "authority_source": source.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityDecision":
        if not isinstance(payload, Mapping):
            raise PrincipalAuthorityError(
                "capability decision payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="capability decision",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_DECISION_FIELDS)),
            artifact_name="capability decision",
        )
        result = cls(
            principal_id=payload.get("principal_id", ""),
            repository_id=payload.get("repository_id", ""),
            run_id=payload.get("run_id", ""),
            exact_effects=payload.get("exact_effects", ()),
            expires_at_ms=payload.get("expires_at_ms"),
            autonomy_ceiling=payload.get("autonomy_ceiling", ""),
            resource_ceilings=payload.get("resource_ceilings") or {},
            disclosure_policy_id=payload.get("disclosure_policy_id", ""),
            provider_policy_id=payload.get("provider_policy_id", ""),
            nonce=payload.get("nonce", ""),
            principal_content_id=payload.get("principal_content_id", ""),
            granted_effects=payload.get("granted_effects", ()),
            decided_at_ms=payload.get("decided_at_ms"),
            verdict=payload.get("verdict", CapabilityVerdict.PERMIT),
            reason_code=payload.get("reason_code", "bound"),
            authority_source=payload.get(
                "authority_source", AuthoritySource.AUTHENTICATED_PRINCIPAL
            ),
        )
        _claimed_identity(payload, result.content_id, artifact_name="capability decision")
        return result


def bind_capability(
    principal: ExternalPrincipal,
    *,
    now_ms: int,
    requested_effects: Sequence[str | EffectName] | None = None,
    authority_source: str | AuthoritySource = AuthoritySource.AUTHENTICATED_PRINCIPAL,
    cid: Any = None,
    imported_history: Any = None,
    history: Any = None,
    prompt: Any = None,
    payment: Any = None,
    commit: Any = None,
    content_id: Any = None,
    content_identity: Any = None,
) -> CapabilityDecision:
    """Bind ``principal`` to an effect-bound decision at ``now_ms``.

    Authority is taken only from the authenticated principal record.  Passing a
    CID, imported history, prompt, payment, or commit as an inference channel
    is rejected.  Unknown effects fail closed.  ``expires_at_ms`` must be
    strictly greater than ``now_ms``.
    """

    if not isinstance(principal, ExternalPrincipal):
        raise PrincipalAuthorityError(
            "bind requires an ExternalPrincipal@1 record", reason_code="malformed"
        )
    source = _reject_inferred_authority(
        authority_source=authority_source,
        inferences={
            "cid": cid,
            "content_id": content_id,
            "content_identity": content_identity,
            "imported_history": imported_history,
            "history": history,
            "prompt": prompt,
            "payment": payment,
            "commit": commit,
        },
    )
    decided_at = _nonnegative_int(now_ms, "now_ms")
    if principal.expires_at_ms <= decided_at:
        raise PrincipalExpiryError(
            "expiry must be in the future relative to now_ms",
            reason_code="expired",
        )
    granted = (
        principal.exact_effects
        if requested_effects is None
        else _effects(requested_effects, "requested_effects")
    )
    extra = set(granted).difference(principal.exact_effects)
    if extra:
        raise PrincipalAuthorityError(
            "requested effects must be a subset of the bound exact_effects",
            reason_code="effect_not_granted",
        )
    return CapabilityDecision(
        principal_id=principal.principal_id,
        repository_id=principal.repository_id,
        run_id=principal.run_id,
        exact_effects=principal.exact_effects,
        expires_at_ms=principal.expires_at_ms,
        autonomy_ceiling=principal.autonomy_ceiling,
        resource_ceilings=principal.resource_ceilings,
        disclosure_policy_id=principal.disclosure_policy_id,
        provider_policy_id=principal.provider_policy_id,
        nonce=principal.nonce,
        principal_content_id=principal.content_id,
        granted_effects=granted,
        decided_at_ms=decided_at,
        verdict=CapabilityVerdict.PERMIT,
        reason_code="bound",
        authority_source=source,
    )


__all__ = (
    "ALLOWED_EFFECTS",
    "AuthoritySource",
    "AutonomyCeiling",
    "CAPABILITY_DECISION_INTERFACE",
    "CAPABILITY_DECISION_SCHEMA",
    "CONTRACT_VERSION",
    "CapabilityDecision",
    "CapabilityVerdict",
    "EXTERNAL_PRINCIPAL_INTERFACE",
    "EXTERNAL_PRINCIPAL_SCHEMA",
    "EffectName",
    "ExternalPrincipal",
    "FORBIDDEN_AUTHORITY_SOURCES",
    "MAX_NONCE_BYTES",
    "MIN_NONCE_BYTES",
    "PrincipalAuthorityError",
    "PrincipalExpiryError",
    "RESOURCE_CEILINGS_INTERFACE",
    "RESOURCE_CEILINGS_SCHEMA",
    "ResourceCeilings",
    "SCHEMA_VERSION",
    "UnknownEffectError",
    "bind_capability",
)
