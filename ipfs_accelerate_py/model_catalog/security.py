"""Fail-closed security primitives for federated AI catalog data.

The helpers in this module are deliberately transport-free.  DNS answers,
issuer keys, clocks, and authorization grants are supplied by the caller so
tests and policy evaluation never need a live network or ambient credentials.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

from .identity import canonical_json_bytes, is_secret_key, is_secret_value
from .schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

MAX_ERROR_MESSAGE_BYTES = 192
MAX_ADVERTISEMENT_BYTES = 65_536
MAX_ADVERTISEMENT_ITEMS = 512
MAX_ADVERTISEMENT_DEPTH = 8
MAX_REPLAY_ENTRIES = 16_384
MAX_URL_BYTES = 2_048
MAX_REDIRECTS = 5
SERVICE_NAMESPACE = "/mcppp/services/1.0.0"

_CAPABILITY = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,7}$")
_CID = re.compile(r"^b[a-z2-7]{20,200}$")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_NONCE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_OPERATION = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,7}$")
_URL_FIELD = re.compile(
    r"(?:^|[_-])(?:url|uri|endpoint|redirect|callback|webhook)(?:$|[_-])",
    re.IGNORECASE,
)
_URI_REFERENCE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)
_NUMERIC_HOST = re.compile(
    r"^(?:0x[0-9a-f]+|0[0-7]+|[0-9]+|[0-9.]+)$", re.IGNORECASE
)


class SecurityPolicyError(ValueError):
    """A public, bounded, non-reflective security policy failure."""

    def __init__(self, code: str, message: str) -> None:
        safe_code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{1,63}", code)
            else "policy_rejected"
        )
        safe_message = str(message).encode("utf-8")[:MAX_ERROR_MESSAGE_BYTES]
        super().__init__(safe_message.decode("utf-8", "ignore"))
        self.code = safe_code

    def to_dict(self) -> Dict[str, str]:
        return {"status": "rejected", "reason": self.code}


class AdvertisementVerificationError(SecurityPolicyError):
    """A catalog advertisement did not satisfy verification policy."""


class AuthorizationPolicyError(SecurityPolicyError):
    """An exact catalog capability was not granted."""


class URLPolicyError(SecurityPolicyError):
    """A URL or resolved address violated outbound network policy."""


class InputPolicyError(SecurityPolicyError):
    """Untrusted structured input exceeded its safe contract."""


def _reject(error_type: type[SecurityPolicyError], code: str, message: str) -> None:
    raise error_type(code, message)


@dataclass(frozen=True)
class InputLimits:
    """Finite traversal and allocation limits for one untrusted input class."""

    max_bytes: int
    max_depth: int
    max_items: int
    max_string_bytes: int
    max_binary_bytes: int = 0


RECORD_LIMITS = InputLimits(
    max_bytes=MAX_ADVERTISEMENT_BYTES,
    max_depth=MAX_ADVERTISEMENT_DEPTH,
    max_items=MAX_ADVERTISEMENT_ITEMS,
    max_string_bytes=8_192,
)
PAGE_LIMITS = InputLimits(
    max_bytes=1_048_576,
    max_depth=16,
    max_items=20_000,
    max_string_bytes=65_536,
)
MEDIA_LIMITS = InputLimits(
    max_bytes=16_777_216,
    max_depth=8,
    max_items=256,
    max_string_bytes=4_096,
    max_binary_bytes=16_777_216,
)
DIAGNOSTIC_LIMITS = InputLimits(
    max_bytes=16_384,
    max_depth=4,
    max_items=128,
    max_string_bytes=1_024,
)


def _safe_ip(value: str) -> ipaddress._BaseAddress:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        _reject(URLPolicyError, "dns_invalid", "DNS returned an invalid address")
    if (
        address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    ):
        _reject(URLPolicyError, "address_denied", "address range is not permitted")
    return address


def _canonical_host(host: str) -> str:
    if not isinstance(host, str) or not host or len(host.encode("utf-8")) > 253:
        _reject(URLPolicyError, "host_invalid", "URL host is invalid")
    if "%" in host or any(ord(char) < 33 for char in host):
        _reject(URLPolicyError, "host_invalid", "URL host is invalid")
    try:
        canonical = host.rstrip(".").encode("idna").decode("ascii").casefold()
    except (UnicodeError, ValueError):
        _reject(URLPolicyError, "host_invalid", "URL host is invalid")
    if not canonical or canonical in {"localhost", "localhost.localdomain"}:
        _reject(URLPolicyError, "host_denied", "URL host is not permitted")
    try:
        return str(ipaddress.ip_address(canonical))
    except ValueError:
        pass
    if _NUMERIC_HOST.fullmatch(canonical):
        _reject(URLPolicyError, "host_invalid", "ambiguous numeric hosts are forbidden")
    labels = canonical.split(".")
    if len(labels) < 2 or any(not _HOST_LABEL.fullmatch(label) for label in labels):
        _reject(URLPolicyError, "host_invalid", "URL host is invalid")
    return canonical


@dataclass(frozen=True)
class URLPolicy:
    """Scheme, authority, redirect, DNS, address, and allowlist policy.

    ``resolver`` must be an injected deterministic callable.  This class never
    calls ``socket.getaddrinfo`` or any other live network primitive.
    """

    allowed_hosts: Tuple[str, ...]
    allowed_schemes: Tuple[str, ...] = ("https",)
    allowed_ports: Tuple[int, ...] = (443,)
    resolver: Optional[Callable[[str], Iterable[str]]] = None
    require_dns_resolution: bool = True
    allow_query: bool = False
    allow_cross_host_redirects: bool = False
    max_redirects: int = MAX_REDIRECTS

    def __post_init__(self) -> None:
        schemes = tuple(sorted(set(str(item).casefold() for item in self.allowed_schemes)))
        if not schemes or any(item not in {"http", "https"} for item in schemes):
            raise ValueError("allowed_schemes must contain only http or https")
        ports = tuple(sorted(set(self.allowed_ports)))
        if (
            not ports
            or any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports)
        ):
            raise ValueError("allowed_ports must contain valid TCP ports")
        hosts = []
        for pattern in self.allowed_hosts:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("allowed_hosts contains an invalid pattern")
            wildcard = pattern.startswith("*.")
            host = _canonical_host(pattern[2:] if wildcard else pattern)
            hosts.append("*." + host if wildcard else host)
        if (
            isinstance(self.max_redirects, bool)
            or not isinstance(self.max_redirects, int)
            or not 0 <= self.max_redirects <= 20
        ):
            raise ValueError("max_redirects must be between 0 and 20")
        object.__setattr__(self, "allowed_schemes", schemes)
        object.__setattr__(self, "allowed_ports", ports)
        object.__setattr__(self, "allowed_hosts", tuple(sorted(set(hosts))))

    def _host_allowed(self, host: str) -> bool:
        for pattern in self.allowed_hosts:
            if pattern.startswith("*."):
                suffix = pattern[1:]
                if host.endswith(suffix) and host != pattern[2:]:
                    return True
            elif host == pattern:
                return True
        return False

    def validate(
        self,
        url: str,
        *,
        resolved_addresses: Optional[Iterable[str]] = None,
    ) -> str:
        """Validate and normalize one absolute outbound URL."""

        if (
            not isinstance(url, str)
            or not url
            or len(url.encode("utf-8")) > MAX_URL_BYTES
            or "\\" in url
            or any(ord(char) < 32 or ord(char) == 127 for char in url)
            or is_secret_value(url)
        ):
            _reject(URLPolicyError, "url_malformed", "URL is malformed")
        try:
            parts = urlsplit(url)
            explicit_port = parts.port
        except ValueError:
            _reject(URLPolicyError, "url_malformed", "URL is malformed")
        scheme = parts.scheme.casefold()
        if scheme not in self.allowed_schemes:
            _reject(URLPolicyError, "scheme_denied", "URL scheme is not permitted")
        if not parts.netloc or parts.hostname is None:
            _reject(URLPolicyError, "host_invalid", "URL must have an authority")
        if parts.username is not None or parts.password is not None:
            _reject(URLPolicyError, "userinfo_denied", "URL user information is forbidden")
        if parts.fragment:
            _reject(URLPolicyError, "fragment_denied", "URL fragments are forbidden")
        if parts.query and not self.allow_query:
            _reject(URLPolicyError, "query_denied", "URL queries are forbidden")
        if parts.query:
            for key, value in parse_qsl(parts.query, keep_blank_values=True):
                if is_secret_key(key) or is_secret_value(value):
                    _reject(URLPolicyError, "secret_input", "URL query contains credentials")

        host = _canonical_host(parts.hostname)
        if not self._host_allowed(host):
            _reject(URLPolicyError, "host_denied", "URL host is not allowlisted")
        port = explicit_port or (443 if scheme == "https" else 80)
        if port not in self.allowed_ports:
            _reject(URLPolicyError, "port_denied", "URL port is not permitted")

        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        answers = resolved_addresses
        if answers is None and literal is not None:
            answers = (str(literal),)
        if answers is None and self.resolver is not None:
            try:
                answers = self.resolver(host)
            except Exception:
                _reject(URLPolicyError, "dns_failed", "DNS resolution failed")
        if answers is None:
            if self.require_dns_resolution:
                _reject(URLPolicyError, "dns_required", "validated DNS answers are required")
        else:
            if isinstance(answers, (str, bytes)):
                answers = (str(answers),)
            try:
                bounded_answers = tuple(answers)
            except TypeError:
                _reject(URLPolicyError, "dns_invalid", "DNS answers are invalid")
            if not bounded_answers or len(bounded_answers) > 32:
                _reject(URLPolicyError, "dns_invalid", "DNS answers are invalid")
            for answer in bounded_answers:
                if not isinstance(answer, str):
                    _reject(URLPolicyError, "dns_invalid", "DNS answers are invalid")
                _safe_ip(answer)

        normalized_host = "[%s]" % host if ":" in host else host
        default_port = 443 if scheme == "https" else 80
        authority = normalized_host if port == default_port else "%s:%d" % (normalized_host, port)
        return urlunsplit((scheme, authority, parts.path or "/", parts.query, ""))

    def validate_redirect(
        self,
        source_url: str,
        target_url: str,
        *,
        redirect_count: int,
        resolved_addresses: Optional[Iterable[str]] = None,
    ) -> str:
        """Validate every redirect hop, including relative redirects."""

        if (
            isinstance(redirect_count, bool)
            or not isinstance(redirect_count, int)
            or redirect_count < 1
            or redirect_count > self.max_redirects
        ):
            _reject(URLPolicyError, "redirect_denied", "redirect limit exceeded")
        source = self.validate(source_url)
        target = self.validate(
            urljoin(source, target_url), resolved_addresses=resolved_addresses
        )
        if (
            not self.allow_cross_host_redirects
            and urlsplit(source).hostname != urlsplit(target).hostname
        ):
            _reject(URLPolicyError, "redirect_denied", "cross-host redirect is forbidden")
        return target


def _validate_value(
    value: Any,
    *,
    limits: InputLimits,
    url_policy: Optional[URLPolicy],
    allow_urls: bool,
) -> Any:
    seen: set[int] = set()
    item_count = 0
    byte_count = 0

    def account_bytes(amount: int) -> None:
        nonlocal byte_count
        byte_count += amount
        if byte_count > limits.max_bytes:
            _reject(InputPolicyError, "input_oversized", "input exceeds byte bound")

    def visit(item: Any, depth: int, field_name: str = "") -> None:
        nonlocal item_count
        if depth > limits.max_depth:
            _reject(InputPolicyError, "input_recursive", "input exceeds nesting bound")
        item_count += 1
        if item_count > limits.max_items:
            _reject(InputPolicyError, "input_oversized", "input exceeds item bound")
        if item is None or isinstance(item, bool):
            account_bytes(4)
            return
        if isinstance(item, int):
            if abs(item) > (1 << 63) - 1:
                _reject(InputPolicyError, "input_malformed", "integer exceeds bound")
            account_bytes(24)
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                _reject(InputPolicyError, "input_malformed", "non-finite number")
            account_bytes(32)
            return
        if isinstance(item, str):
            size = len(item.encode("utf-8"))
            if size > limits.max_string_bytes:
                _reject(InputPolicyError, "input_oversized", "string exceeds bound")
            if "\x00" in item or is_secret_value(item):
                _reject(InputPolicyError, "secret_input", "credential-shaped input")
            account_bytes(size + 2)
            network_field = bool(_URL_FIELD.search(field_name)) and (
                field_name.casefold() != "endpoint_protocol"
            )
            if _URI_REFERENCE.match(item) or network_field:
                if not allow_urls or url_policy is None:
                    _reject(InputPolicyError, "ssrf_input", "network location is forbidden")
                try:
                    url_policy.validate(item)
                except URLPolicyError as exc:
                    _reject(InputPolicyError, "ssrf_input", str(exc))
            return
        if isinstance(item, (bytes, bytearray, memoryview)):
            size = len(item)
            if limits.max_binary_bytes <= 0 or size > limits.max_binary_bytes:
                _reject(InputPolicyError, "input_oversized", "binary input exceeds bound")
            account_bytes(size)
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                _reject(InputPolicyError, "input_recursive", "recursive input is forbidden")
            seen.add(identity)
            try:
                for key, child in item.items():
                    if not isinstance(key, str):
                        _reject(InputPolicyError, "input_malformed", "object key must be text")
                    key_size = len(key.encode("utf-8"))
                    if key_size > 128:
                        _reject(InputPolicyError, "input_oversized", "object key exceeds bound")
                    if is_secret_key(key):
                        _reject(InputPolicyError, "secret_input", "credential field is forbidden")
                    account_bytes(key_size + 3)
                    visit(child, depth + 1, key)
            finally:
                seen.discard(identity)
            return
        if isinstance(item, Sequence):
            identity = id(item)
            if identity in seen:
                _reject(InputPolicyError, "input_recursive", "recursive input is forbidden")
            seen.add(identity)
            try:
                for child in item:
                    visit(child, depth + 1, field_name)
            finally:
                seen.discard(identity)
            return
        _reject(InputPolicyError, "input_malformed", "unsupported input type")

    visit(value, 0)
    return value


@dataclass(frozen=True)
class CatalogInputPolicy:
    """Separate finite contracts for records, pages, media, and diagnostics."""

    url_policy: Optional[URLPolicy] = None
    record_limits: InputLimits = RECORD_LIMITS
    page_limits: InputLimits = PAGE_LIMITS
    media_limits: InputLimits = MEDIA_LIMITS
    diagnostic_limits: InputLimits = DIAGNOSTIC_LIMITS

    def validate_record(self, value: Any, *, allow_urls: bool = False) -> Any:
        return _validate_value(
            value,
            limits=self.record_limits,
            url_policy=self.url_policy,
            allow_urls=allow_urls,
        )

    def validate_page(self, value: Any, *, allow_urls: bool = False) -> Any:
        return _validate_value(
            value,
            limits=self.page_limits,
            url_policy=self.url_policy,
            allow_urls=allow_urls,
        )

    def validate_media(self, value: Any) -> Any:
        return _validate_value(
            value,
            limits=self.media_limits,
            url_policy=self.url_policy,
            allow_urls=False,
        )

    def validate_diagnostic(self, value: Any) -> Any:
        return _validate_value(
            value,
            limits=self.diagnostic_limits,
            url_policy=None,
            allow_urls=False,
        )


class CatalogCapability(str, Enum):
    """Distinct abilities used by the federated catalog control plane."""

    READ = "catalog.read"
    REMOTE_REFRESH = "catalog.refresh.remote"
    HEALTH_PROBE = "catalog.health.probe"

    @classmethod
    def invoke(cls, operation: str) -> str:
        if not isinstance(operation, str) or not _OPERATION.fullmatch(operation):
            _reject(AuthorizationPolicyError, "capability_invalid", "operation is invalid")
        return "catalog.invoke.%s" % operation


@dataclass(frozen=True)
class CapabilityGrant:
    """A UCAN-equivalent exact resource/ability grant."""

    resource: str
    ability: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resource, str)
            or not self.resource
            or len(self.resource.encode("utf-8")) > 256
            or self.resource == "*"
        ):
            raise ValueError("capability resource must be explicit and bounded")
        if not isinstance(self.ability, str) or not _CAPABILITY.fullmatch(self.ability):
            raise ValueError("capability ability is invalid")


class CatalogAuthorizationPolicy:
    """Exact, deny-by-default capability policy keyed by authenticated actor."""

    def __init__(self, grants: Optional[Mapping[str, Iterable[CapabilityGrant]]] = None):
        self._grants: Dict[str, frozenset[CapabilityGrant]] = {}
        for actor, actor_grants in (grants or {}).items():
            if not isinstance(actor, str) or not actor or len(actor.encode("utf-8")) > 256:
                raise ValueError("authorization actor is invalid")
            bounded = tuple(actor_grants)
            if len(bounded) > 256 or any(not isinstance(item, CapabilityGrant) for item in bounded):
                raise ValueError("authorization grants are invalid or excessive")
            self._grants[actor] = frozenset(bounded)

    def is_authorized(self, actor: str, resource: str, ability: str) -> bool:
        try:
            requested = CapabilityGrant(resource=resource, ability=ability)
        except ValueError:
            return False
        return requested in self._grants.get(actor, frozenset())

    def require(self, actor: str, resource: str, ability: str) -> None:
        if not self.is_authorized(actor, resource, ability):
            _reject(
                AuthorizationPolicyError,
                "capability_denied",
                "required catalog capability was not granted",
            )


class ReplayCache:
    """Thread-safe, expiry-bound nonce cache with a hard memory ceiling."""

    def __init__(self, max_entries: int = MAX_REPLAY_ENTRIES) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= MAX_REPLAY_ENTRIES
        ):
            raise ValueError("max_entries is invalid")
        self.max_entries = max_entries
        self._entries: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def consume(self, issuer: str, nonce: str, expires_at: float, now: float) -> bool:
        key = (issuer, nonce)
        with self._lock:
            self._entries = {
                item: expiry
                for item, expiry in self._entries.items()
                if expiry > now
            }
            if key in self._entries:
                return False
            if len(self._entries) >= self.max_entries:
                # A replay cache must not make an old nonce reusable merely to
                # admit a new one.  Capacity pressure therefore fails closed.
                return False
            self._entries[key] = expires_at
        return True


@dataclass
class AdvertisementVerifier:
    """Verify trusted signed advertisements before catalog admission."""

    trusted_issuers: Mapping[str, bytes | str]
    expected_schema_version: str = SCHEMA_VERSION
    max_clock_skew: float = 30.0
    replay_window: float = 600.0
    max_lifetime: float = 600.0
    replay_cache: ReplayCache = field(default_factory=ReplayCache)
    clock: Callable[[], float] = time.time
    _keys: Dict[str, bytes] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.expected_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("expected_schema_version is unsupported")
        for value, name in (
            (self.max_clock_skew, "max_clock_skew"),
            (self.replay_window, "replay_window"),
            (self.max_lifetime, "max_lifetime"),
        ):
            minimum = 0 if name == "max_clock_skew" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < minimum
            ):
                raise ValueError("%s must be finite and positive" % name)
        keys: Dict[str, bytes] = {}
        for issuer, key in self.trusted_issuers.items():
            if not isinstance(issuer, str) or not issuer or len(issuer.encode("utf-8")) > 256:
                raise ValueError("trusted issuer is invalid")
            material = key.encode("utf-8") if isinstance(key, str) else key
            if not isinstance(material, bytes) or not material or len(material) > 4_096:
                raise ValueError("trusted issuer key is invalid")
            keys[issuer] = material
        self._keys = keys

    def verify(
        self,
        record: Any,
        *,
        now: Optional[float] = None,
        expected_catalog_cid: Optional[str] = None,
        consume_nonce: bool = True,
    ) -> Any:
        """Return *record* only after every policy check succeeds."""

        selected_now = self.clock() if now is None else now
        if (
            isinstance(selected_now, bool)
            or not isinstance(selected_now, (int, float))
            or not math.isfinite(selected_now)
        ):
            _reject(AdvertisementVerificationError, "clock_invalid", "verification clock is invalid")
        try:
            payload = record.to_dict()
        except Exception:
            _reject(AdvertisementVerificationError, "record_malformed", "advertisement is malformed")
        try:
            CatalogInputPolicy().validate_record(payload)
            canonical = canonical_json_bytes(payload)
        except Exception:
            _reject(AdvertisementVerificationError, "record_malformed", "advertisement is malformed")
        if len(canonical) > MAX_ADVERTISEMENT_BYTES:
            _reject(AdvertisementVerificationError, "record_oversized", "advertisement exceeds bound")

        issuer = getattr(record, "issuer", None)
        peer_id = getattr(record, "peer_id", None)
        if issuer not in self._keys:
            _reject(AdvertisementVerificationError, "issuer_untrusted", "advertisement issuer is not trusted")
        if not isinstance(peer_id, str) or issuer != peer_id:
            _reject(AdvertisementVerificationError, "issuer_mismatch", "issuer does not match peer identity")
        if getattr(record, "schema_version", None) != self.expected_schema_version:
            _reject(AdvertisementVerificationError, "schema_unsupported", "catalog schema version is unsupported")

        cid = getattr(record, "catalog_cid", None)
        revision = getattr(record, "catalog_revision", None)
        if (
            not isinstance(cid, str)
            or not _CID.fullmatch(cid)
            or cid != revision
            or (expected_catalog_cid is not None and cid != expected_catalog_cid)
        ):
            _reject(AdvertisementVerificationError, "catalog_cid_invalid", "catalog CID does not match revision")
        required_text = (
            "service_name",
            "service_id",
            "endpoint_protocol",
        )
        if any(
            not isinstance(getattr(record, name, None), str)
            or not getattr(record, name)
            for name in required_text
        ):
            _reject(AdvertisementVerificationError, "record_malformed", "required advertisement field is missing")
        identity_payload = json.dumps(
            {
                "namespace": SERVICE_NAMESPACE,
                "service_name": record.service_name,
                "issuer": issuer,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_service_id = "service_" + hashlib.sha256(identity_payload).hexdigest()
        if record.service_id != expected_service_id:
            _reject(
                AdvertisementVerificationError,
                "service_identity_invalid",
                "service identity does not match signed fields",
            )
        operations = getattr(record, "operation_summary", None)
        interfaces = getattr(record, "interface_cids", None)
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(item, str) or not _OPERATION.fullmatch(item) for item in operations)
            or not isinstance(interfaces, list)
            or not interfaces
            or any(not isinstance(item, str) or not item or len(item.encode("utf-8")) > 256 for item in interfaces)
        ):
            _reject(AdvertisementVerificationError, "record_malformed", "operation or interface summary is invalid")
        nonce = getattr(record, "nonce", None)
        if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
            _reject(AdvertisementVerificationError, "nonce_invalid", "advertisement nonce is invalid")

        issued_at = getattr(record, "issued_at", None)
        expires_at = getattr(record, "expires_at", None)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (issued_at, expires_at)
        ):
            _reject(AdvertisementVerificationError, "time_invalid", "advertisement time is invalid")
        issued = float(issued_at)
        expires = float(expires_at)
        if issued > selected_now + self.max_clock_skew:
            _reject(AdvertisementVerificationError, "issued_in_future", "advertisement exceeds clock skew")
        if expires <= issued or expires - issued > self.max_lifetime:
            _reject(AdvertisementVerificationError, "lifetime_invalid", "advertisement lifetime is invalid")
        if expires <= selected_now:
            _reject(AdvertisementVerificationError, "expired", "advertisement has expired")
        if issued < selected_now - self.replay_window:
            _reject(AdvertisementVerificationError, "stale", "advertisement is outside replay window")

        verifier = getattr(record, "verify_signature", None)
        if not callable(verifier) or not verifier(self._keys[issuer]):
            _reject(AdvertisementVerificationError, "signature_invalid", "advertisement signature is invalid")
        if consume_nonce and not self.replay_cache.consume(issuer, nonce, expires, float(selected_now)):
            _reject(AdvertisementVerificationError, "replayed", "advertisement nonce was already used")
        return record


__all__ = [
    "AdvertisementVerificationError",
    "AdvertisementVerifier",
    "AuthorizationPolicyError",
    "CapabilityGrant",
    "CatalogAuthorizationPolicy",
    "CatalogCapability",
    "CatalogInputPolicy",
    "DIAGNOSTIC_LIMITS",
    "InputLimits",
    "InputPolicyError",
    "MAX_ADVERTISEMENT_BYTES",
    "MAX_REDIRECTS",
    "MEDIA_LIMITS",
    "PAGE_LIMITS",
    "RECORD_LIMITS",
    "ReplayCache",
    "SecurityPolicyError",
    "URLPolicy",
    "URLPolicyError",
]
