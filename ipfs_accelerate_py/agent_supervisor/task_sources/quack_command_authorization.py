"""Effect-bound authorization for the bounded Quack command ingress.

``StateCommand@1`` is a fenced/idempotent storage command, not an identity or
authorization credential.  This module deliberately leaves that shared
contract unchanged and wraps it in a closed, independently signed envelope.
The sole local DuckDB owner verifies the envelope after transport and before
submitting the command.  Possession of a Quack token therefore never grants
mutation authority.

The module verifies signatures only.  It does not load private keys, sign
commands, or mint runtime authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..control.profile_authority import (
    LocalProfileTampered,
    verify_did_key_signature,
)
from .control_plane_contracts import CommandKind, StateCommand

AUTHORIZED_STATE_COMMAND_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
)
AUTHORIZED_STATE_COMMAND_INTERFACE: Final = "AuthorizedStateCommand@1"
QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-command-authorization-policy@1"
)
MAX_AUTHORIZATION_LIFETIME_MS: Final = 5 * 60 * 1000
MAX_CLOCK_SKEW_MS: Final = 5_000

_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "request_id",
        "submission_id",
        "ingress_slot",
        "principal_did",
        "approver_did",
        "authority_ref_cid",
        "board_namespace",
        "shard_id",
        "owner_principal_did",
        "lease_id",
        "scope_id",
        "effect",
        "issued_at_ms",
        "expires_at_ms",
        "deadline_ms",
        "one_use_nonce",
        "command_content_id",
        "command",
        "approver_signature",
        "envelope_cid",
    }
)
_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CID_RE: Final = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})\Z")
_COMPACT_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@+\-]{0,511}\Z")


class QuackCommandAuthorizationError(ValueError):
    """A transported command lacks exact effect-bound authority."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise QuackCommandAuthorizationError("authorized command is not canonical JSON") from exc


def _sha256_cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise QuackCommandAuthorizationError(f"{name} must be a non-empty string")
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise QuackCommandAuthorizationError(f"{name} is outside its byte bound")
    return value


def _compact(value: Any, name: str) -> str:
    text = _text(value, name)
    if _COMPACT_RE.fullmatch(text) is None:
        raise QuackCommandAuthorizationError(f"{name} is not a compact identifier")
    return text


def _did(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("did:key:z"):
        raise QuackCommandAuthorizationError(f"{name} must be an Ed25519 did:key")
    return text


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CID_RE.fullmatch(text) is None:
        raise QuackCommandAuthorizationError(f"{name} is not a content identity")
    return text


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QuackCommandAuthorizationError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class QuackCommandAuthorizationPolicy:
    """Host-pinned verification policy for one command-fabric owner."""

    board_namespace: str
    shard_id: str
    store_id: str
    authority_ref_cid: str
    owner_principal_did: str
    owner_generation: int
    fence_epoch: int
    trusted_approver_dids: frozenset[str]
    authorized_principal_dids: frozenset[str]
    allowed_command_kinds: frozenset[CommandKind]
    maximum_authorization_lifetime_ms: int = MAX_AUTHORIZATION_LIFETIME_MS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "board_namespace", _compact(self.board_namespace, "board_namespace")
        )
        object.__setattr__(self, "shard_id", _compact(self.shard_id, "shard_id"))
        object.__setattr__(self, "store_id", _compact(self.store_id, "store_id"))
        object.__setattr__(
            self,
            "authority_ref_cid",
            _cid(self.authority_ref_cid, "authority_ref_cid"),
        )
        object.__setattr__(
            self,
            "owner_principal_did",
            _did(self.owner_principal_did, "owner_principal_did"),
        )
        object.__setattr__(
            self,
            "owner_generation",
            _positive_integer(self.owner_generation, "owner_generation"),
        )
        object.__setattr__(self, "fence_epoch", _positive_integer(self.fence_epoch, "fence_epoch"))
        approvers = frozenset(
            _did(value, "trusted_approver_did") for value in self.trusted_approver_dids
        )
        principals = frozenset(
            _did(value, "authorized_principal_did") for value in self.authorized_principal_dids
        )
        kinds = frozenset(
            value if isinstance(value, CommandKind) else CommandKind(value)
            for value in self.allowed_command_kinds
        )
        if not approvers or not principals or not kinds:
            raise QuackCommandAuthorizationError("authorization policy sets must not be empty")
        if self.owner_principal_did in approvers:
            raise QuackCommandAuthorizationError(
                "the command owner cannot approve its own mutations"
            )
        if approvers.intersection(principals):
            raise QuackCommandAuthorizationError(
                "a command principal cannot approve its own mutation"
            )
        lifetime = _positive_integer(
            self.maximum_authorization_lifetime_ms,
            "maximum_authorization_lifetime_ms",
        )
        if lifetime > MAX_AUTHORIZATION_LIFETIME_MS:
            raise QuackCommandAuthorizationError(
                "maximum authorization lifetime exceeds the protocol bound"
            )
        object.__setattr__(self, "trusted_approver_dids", approvers)
        object.__setattr__(self, "authorized_principal_dids", principals)
        object.__setattr__(self, "allowed_command_kinds", kinds)
        object.__setattr__(self, "maximum_authorization_lifetime_ms", lifetime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA,
            "board_namespace": self.board_namespace,
            "shard_id": self.shard_id,
            "store_id": self.store_id,
            "authority_ref_cid": self.authority_ref_cid,
            "owner_principal_did": self.owner_principal_did,
            "owner_generation": self.owner_generation,
            "fence_epoch": self.fence_epoch,
            "trusted_approver_dids": sorted(self.trusted_approver_dids),
            "authorized_principal_dids": sorted(self.authorized_principal_dids),
            "allowed_command_kinds": sorted(
                value.value for value in self.allowed_command_kinds
            ),
            "maximum_authorization_lifetime_ms": (
                self.maximum_authorization_lifetime_ms
            ),
        }

    @property
    def policy_cid(self) -> str:
        return _sha256_cid(self.to_dict())


@dataclass(frozen=True)
class AuthorizedStateCommand:
    """Closed signed envelope transported through the append-only ingress."""

    request_id: str
    submission_id: str
    ingress_slot: int
    principal_did: str
    approver_did: str
    authority_ref_cid: str
    board_namespace: str
    shard_id: str
    owner_principal_did: str
    lease_id: str
    scope_id: str
    effect: str
    issued_at_ms: int
    expires_at_ms: int
    deadline_ms: int
    one_use_nonce: str
    command: StateCommand
    approver_signature: str
    envelope_cid: str

    SCHEMA: ClassVar[str] = AUTHORIZED_STATE_COMMAND_SCHEMA
    INTERFACE: ClassVar[str] = AUTHORIZED_STATE_COMMAND_INTERFACE

    def __post_init__(self) -> None:
        for name in ("request_id", "submission_id", "lease_id", "scope_id", "one_use_nonce"):
            object.__setattr__(self, name, _compact(getattr(self, name), name))
        object.__setattr__(self, "principal_did", _did(self.principal_did, "principal_did"))
        object.__setattr__(self, "approver_did", _did(self.approver_did, "approver_did"))
        object.__setattr__(
            self, "owner_principal_did", _did(self.owner_principal_did, "owner_principal_did")
        )
        object.__setattr__(
            self, "authority_ref_cid", _cid(self.authority_ref_cid, "authority_ref_cid")
        )
        object.__setattr__(
            self,
            "board_namespace",
            _compact(self.board_namespace, "board_namespace"),
        )
        object.__setattr__(self, "shard_id", _compact(self.shard_id, "shard_id"))
        object.__setattr__(self, "effect", _compact(self.effect, "effect"))
        object.__setattr__(
            self, "ingress_slot", _positive_integer(self.ingress_slot, "ingress_slot")
        )
        for name in ("issued_at_ms", "expires_at_ms", "deadline_ms"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if not isinstance(self.command, StateCommand):
            raise QuackCommandAuthorizationError("command must be StateCommand@1")
        signature = _text(self.approver_signature, "approver_signature", maximum=256)
        object.__setattr__(self, "approver_signature", signature)
        claimed = _text(self.envelope_cid, "envelope_cid")
        if _SHA256_RE.fullmatch(claimed) is None or claimed != self.expected_envelope_cid:
            raise QuackCommandAuthorizationError("authorized command envelope CID mismatch")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "request_id": self.request_id,
            "submission_id": self.submission_id,
            "ingress_slot": self.ingress_slot,
            "principal_did": self.principal_did,
            "approver_did": self.approver_did,
            "authority_ref_cid": self.authority_ref_cid,
            "board_namespace": self.board_namespace,
            "shard_id": self.shard_id,
            "owner_principal_did": self.owner_principal_did,
            "lease_id": self.lease_id,
            "scope_id": self.scope_id,
            "effect": self.effect,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "deadline_ms": self.deadline_ms,
            "one_use_nonce": self.one_use_nonce,
            "command_content_id": self.command.content_id,
            "command": self.command.to_record(),
        }

    def signed_payload(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "approver_signature": self.approver_signature}

    @property
    def expected_envelope_cid(self) -> str:
        return _sha256_cid(self.signed_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.signed_payload(), "envelope_cid": self.envelope_cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuthorizedStateCommand:
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise QuackCommandAuthorizationError(
                "authorized command must use the exact closed schema"
            )
        if value.get("schema") != cls.SCHEMA or value.get("interface") != cls.INTERFACE:
            raise QuackCommandAuthorizationError("authorized command schema is unsupported")
        command_value = value.get("command")
        if not isinstance(command_value, Mapping):
            raise QuackCommandAuthorizationError("command must be an object")
        try:
            command = StateCommand.from_dict(command_value)
        except Exception as exc:
            raise QuackCommandAuthorizationError("embedded StateCommand is invalid") from exc
        if value.get("command_content_id") != command.content_id:
            raise QuackCommandAuthorizationError("embedded command content identity mismatch")
        return cls(
            request_id=value.get("request_id"),
            submission_id=value.get("submission_id"),
            ingress_slot=value.get("ingress_slot"),
            principal_did=value.get("principal_did"),
            approver_did=value.get("approver_did"),
            authority_ref_cid=value.get("authority_ref_cid"),
            board_namespace=value.get("board_namespace"),
            shard_id=value.get("shard_id"),
            owner_principal_did=value.get("owner_principal_did"),
            lease_id=value.get("lease_id"),
            scope_id=value.get("scope_id"),
            effect=value.get("effect"),
            issued_at_ms=value.get("issued_at_ms"),
            expires_at_ms=value.get("expires_at_ms"),
            deadline_ms=value.get("deadline_ms"),
            one_use_nonce=value.get("one_use_nonce"),
            command=command,
            approver_signature=value.get("approver_signature"),
            envelope_cid=value.get("envelope_cid"),
        )


def authorized_state_command_signing_payload(
    *,
    request_id: str,
    submission_id: str,
    ingress_slot: int,
    principal_did: str,
    approver_did: str,
    authority_ref_cid: str,
    board_namespace: str,
    shard_id: str,
    owner_principal_did: str,
    lease_id: str,
    scope_id: str,
    effect: str,
    issued_at_ms: int,
    expires_at_ms: int,
    deadline_ms: int,
    one_use_nonce: str,
    command: StateCommand,
) -> Mapping[str, Any]:
    """Prepare canonical public data for an external independent signer."""

    if not isinstance(command, StateCommand):
        raise QuackCommandAuthorizationError("command must be StateCommand@1")
    payload = {
        "schema": AUTHORIZED_STATE_COMMAND_SCHEMA,
        "interface": AUTHORIZED_STATE_COMMAND_INTERFACE,
        "request_id": _compact(request_id, "request_id"),
        "submission_id": _compact(submission_id, "submission_id"),
        "ingress_slot": _positive_integer(ingress_slot, "ingress_slot"),
        "principal_did": _did(principal_did, "principal_did"),
        "approver_did": _did(approver_did, "approver_did"),
        "authority_ref_cid": _cid(authority_ref_cid, "authority_ref_cid"),
        "board_namespace": _compact(board_namespace, "board_namespace"),
        "shard_id": _compact(shard_id, "shard_id"),
        "owner_principal_did": _did(owner_principal_did, "owner_principal_did"),
        "lease_id": _compact(lease_id, "lease_id"),
        "scope_id": _compact(scope_id, "scope_id"),
        "effect": _compact(effect, "effect"),
        "issued_at_ms": _positive_integer(issued_at_ms, "issued_at_ms"),
        "expires_at_ms": _positive_integer(expires_at_ms, "expires_at_ms"),
        "deadline_ms": _positive_integer(deadline_ms, "deadline_ms"),
        "one_use_nonce": _compact(one_use_nonce, "one_use_nonce"),
        "command_content_id": command.content_id,
        "command": command.to_record(),
    }
    return MappingProxyType(payload)


def seal_authorized_state_command(
    prepared_payload: Mapping[str, Any], *, approver_signature: str
) -> AuthorizedStateCommand:
    """Join an external signature to prepared data without creating authority."""

    if not isinstance(prepared_payload, Mapping):
        raise QuackCommandAuthorizationError("prepared payload must be an object")
    body = dict(prepared_payload)
    if set(body) != _FIELDS - {"approver_signature", "envelope_cid"}:
        raise QuackCommandAuthorizationError("prepared payload has non-canonical fields")
    signed = {
        **body,
        "approver_signature": _text(approver_signature, "approver_signature", maximum=256),
    }
    return AuthorizedStateCommand.from_dict({**signed, "envelope_cid": _sha256_cid(signed)})


def verify_authorized_state_command(
    envelope: AuthorizedStateCommand,
    *,
    policy: QuackCommandAuthorizationPolicy,
    now_ms: int,
) -> None:
    """Verify exact authority, independence, freshness, fence, and signature."""

    if type(envelope) is not AuthorizedStateCommand:
        raise QuackCommandAuthorizationError("command envelope is untyped")
    if type(envelope.command) is not StateCommand:
        raise QuackCommandAuthorizationError("embedded command is untyped")
    if type(policy) is not QuackCommandAuthorizationPolicy:
        raise QuackCommandAuthorizationError("authorization policy is untyped")
    now = _positive_integer(now_ms, "now_ms")
    command = envelope.command
    expected_effect = f"control-plane/{command.command_kind.value}"
    if envelope.principal_did not in policy.authorized_principal_dids:
        raise QuackCommandAuthorizationError("command principal is not admitted")
    if envelope.approver_did not in policy.trusted_approver_dids:
        raise QuackCommandAuthorizationError("command approver is not trusted")
    if envelope.principal_did == envelope.approver_did:
        raise QuackCommandAuthorizationError("a principal cannot approve its own command")
    if envelope.owner_principal_did != policy.owner_principal_did:
        raise QuackCommandAuthorizationError("command owner identity mismatch")
    if envelope.approver_did == policy.owner_principal_did:
        raise QuackCommandAuthorizationError("the owner cannot approve its own command")
    if envelope.authority_ref_cid != policy.authority_ref_cid:
        raise QuackCommandAuthorizationError("command authority reference mismatch")
    if envelope.board_namespace != policy.board_namespace:
        raise QuackCommandAuthorizationError("command board namespace mismatch")
    if envelope.shard_id != policy.shard_id or command.store_id != policy.store_id:
        raise QuackCommandAuthorizationError("command shard/store identity mismatch")
    if command.command_kind not in policy.allowed_command_kinds:
        raise QuackCommandAuthorizationError("command kind is outside admitted effects")
    if envelope.effect != expected_effect:
        raise QuackCommandAuthorizationError("command effect does not match its operation")
    parameters = dict(command.parameters)
    semantic_bindings = {
        "request_id": envelope.request_id,
        "principal": envelope.principal_did,
        "authority_ref": envelope.authority_ref_cid,
        "lease_id": envelope.lease_id,
        "task_cid": envelope.scope_id,
        "deadline_ms": envelope.deadline_ms,
    }
    for name, expected in semantic_bindings.items():
        if name in parameters and parameters[name] != expected:
            raise QuackCommandAuthorizationError(f"command {name} contradicts its signed envelope")
    if (
        command.expected_generation != policy.owner_generation
        or command.fence_epoch != policy.fence_epoch
    ):
        raise QuackCommandAuthorizationError("command generation/fence is stale")
    if envelope.expires_at_ms - envelope.issued_at_ms > policy.maximum_authorization_lifetime_ms:
        raise QuackCommandAuthorizationError("command authorization lifetime is too broad")
    if envelope.issued_at_ms > now + MAX_CLOCK_SKEW_MS:
        raise QuackCommandAuthorizationError("command authorization is not yet valid")
    if now >= envelope.expires_at_ms or now >= envelope.deadline_ms:
        raise QuackCommandAuthorizationError("command authorization expired")
    if envelope.deadline_ms > envelope.expires_at_ms:
        raise QuackCommandAuthorizationError("command deadline exceeds authorization expiry")
    try:
        verify_did_key_signature(
            identity_did=envelope.approver_did,
            payload=envelope.unsigned_payload(),
            signature=envelope.approver_signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise QuackCommandAuthorizationError("command approval signature is invalid") from exc


__all__ = [
    "AUTHORIZED_STATE_COMMAND_INTERFACE",
    "AUTHORIZED_STATE_COMMAND_SCHEMA",
    "AuthorizedStateCommand",
    "MAX_AUTHORIZATION_LIFETIME_MS",
    "QUACK_COMMAND_AUTHORIZATION_POLICY_SCHEMA",
    "QuackCommandAuthorizationError",
    "QuackCommandAuthorizationPolicy",
    "authorized_state_command_signing_payload",
    "seal_authorized_state_command",
    "verify_authorized_state_command",
]
