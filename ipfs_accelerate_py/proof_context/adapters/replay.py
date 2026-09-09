"""Deterministic, offline recorded-response adapter (PCCE-033).

Replay fixtures retain the exact admitted *original* invocation, proposal, and
response bytes. Selecting a fixture requires its adapter, fixture, and response
identities in addition to the complete admitted request. A replay creates
copies of the original wire records whose only semantic changes are the
permanent ``replayed`` provenance label and removal of the live-only response
claim.

The adapter has no provider, process, network, or filesystem capability. The
returned log starts with a canonical replay-binding record so two selected
response artifacts remain distinguishable through the frozen ``AdapterResult``
contract without changing the preserved invocation or proposal identities.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    APPROVAL_AUTHORITY,
    CANONICAL_BRANCH_AUTHORITY,
    AdapterResult,
    CancellationToken,
    admit_adapter_result,
    bind_adapter_request,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    MAX_LOG_BYTES,
    MAX_PATCH_BYTES,
    MAX_PROVIDER_OUTPUT_BYTES,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    admit_bounded_log,
    admit_cid,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
)

ADAPTER: Final[str] = "ReplayAdapter@0.1"
FIXTURE_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/replay-fixture@1"
REPLAY_BINDING_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/replay-binding@1"
ORIGINAL_PROVENANCE: Final[str] = "live"
REPLAY_PROVENANCE: Final[str] = "replayed"

_FIXTURE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "fixture_cid",
        "adapter_id",
        "task_cid",
        "pack_cid",
        "route_cid",
        "response_artifact_cid",
        "response_artifact_base64",
        "original_invocation",
        "original_proposal",
        "patch_base64",
        "log_base64",
    }
)


def cid_for_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return the CIDv1 raw/sha2-256 identity of exact bytes."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise MalformedError("CID input must be exact bytes")
    digest = hashlib.sha256(bytes(value)).digest()
    raw = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def cid_for_record(record: Any) -> str:
    """Return the CID of a record's canonical UTF-8 representation."""

    if hasattr(record, "to_mapping"):
        value = dict(record.to_mapping())
    elif isinstance(record, Mapping):
        value = dict(record)
    else:
        raise MalformedError("replay record must be a wire mapping")
    return cid_for_bytes(wire_canonical_utf8(value).encode("utf-8"))


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _admit_exact_bytes(
    value: Any,
    *,
    field_name: str,
    max_bytes: int,
    allow_empty: bool,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise MalformedError(f"{field_name} must be exact bytes")
    admitted = bytes(value)
    if not admitted and not allow_empty:
        raise MalformedError(f"{field_name} must not be empty")
    if len(admitted) > max_bytes:
        raise BoundaryViolationError(f"{field_name} exceeds the frozen byte bound")
    return admitted


def _decode(
    value: Any,
    *,
    field_name: str,
    max_bytes: int,
    allow_empty: bool,
) -> bytes:
    """Decode one canonical RFC 4648 base64 value within its evidence bound."""

    if not isinstance(value, str):
        raise MalformedError(f"{field_name} must be canonical base64 text")
    max_encoded_bytes = 4 * ((max_bytes + 2) // 3)
    if len(value) > max_encoded_bytes:
        raise BoundaryViolationError(f"{field_name} exceeds the frozen byte bound")
    try:
        raw = value.encode("ascii")
        decoded = base64.b64decode(raw, validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise MalformedError(f"{field_name} is not canonical base64") from exc
    if _encode(decoded) != value:
        raise MalformedError(f"{field_name} is not canonical base64")
    return _admit_exact_bytes(
        decoded,
        field_name=field_name,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    )


def _replayed_invocation(original: CodingAgentInvocation) -> CodingAgentInvocation:
    payload = dict(original.to_mapping())
    payload["provenance"] = REPLAY_PROVENANCE
    payload.pop("response_artifact_cid", None)
    return CodingAgentInvocation.from_mapping(payload)


def _replayed_proposal(original: PatchProposal) -> PatchProposal:
    payload = dict(original.to_mapping())
    payload["provenance"] = REPLAY_PROVENANCE
    return PatchProposal.from_mapping(payload)


@dataclass(frozen=True)
class ReplayFixture:
    """One immutable, identity-bound original response and its replay inputs."""

    fixture_cid: str
    task_cid: str
    pack_cid: str
    route_cid: str
    response_artifact_cid: str
    response_artifact: bytes
    original_invocation: CodingAgentInvocation
    original_proposal: PatchProposal
    patch_bytes: bytes
    log_bytes: bytes
    adapter_id: str = ADAPTER

    def __post_init__(self) -> None:
        for name in (
            "fixture_cid",
            "task_cid",
            "pack_cid",
            "route_cid",
            "response_artifact_cid",
        ):
            object.__setattr__(
                self,
                name,
                admit_cid(getattr(self, name), field=name),
            )
        if self.adapter_id != ADAPTER:
            raise IdentityInconsistentError(
                "replay fixture adapter identity drifted",
                details={"field": "adapter_id"},
            )
        if not isinstance(self.original_invocation, CodingAgentInvocation):
            raise MalformedError("original_invocation must be an admitted wire record")
        if not isinstance(self.original_proposal, PatchProposal):
            raise MalformedError("original_proposal must be an admitted wire record")

        response = _admit_exact_bytes(
            self.response_artifact,
            field_name="response_artifact",
            max_bytes=MAX_PROVIDER_OUTPUT_BYTES,
            allow_empty=False,
        )
        patch = _admit_exact_bytes(
            self.patch_bytes,
            field_name="patch_bytes",
            max_bytes=MAX_PATCH_BYTES,
            allow_empty=False,
        )
        log = _admit_exact_bytes(
            self.log_bytes,
            field_name="log_bytes",
            max_bytes=MAX_LOG_BYTES,
            allow_empty=True,
        )
        object.__setattr__(self, "response_artifact", response)
        object.__setattr__(self, "patch_bytes", patch)
        object.__setattr__(self, "log_bytes", log)

        invocation = self.original_invocation
        proposal = self.original_proposal
        if (
            invocation.provenance != ORIGINAL_PROVENANCE
            or proposal.provenance != ORIGINAL_PROVENANCE
        ):
            raise BoundaryViolationError(
                "replay fixtures require exact live original invocation and proposal records"
            )
        if not invocation.usage_is_explicit():
            raise MalformedError("original invocation is missing recorded usage or cost")
        if invocation.response_artifact_cid is None:
            raise MalformedError("original invocation is missing response artifact evidence")
        if invocation.response_artifact_cid != self.response_artifact_cid:
            raise IdentityInconsistentError(
                "original invocation response artifact identity drifted",
                details={"field": "response_artifact_cid"},
            )
        if cid_for_bytes(response) != self.response_artifact_cid:
            raise IdentityInconsistentError(
                "replay response artifact CID does not verify",
                details={"field": "response_artifact_cid"},
            )
        if invocation.route_cid != self.route_cid:
            raise IdentityInconsistentError(
                "original invocation route identity drifted",
                details={"field": "route_cid"},
            )
        if proposal.invocation_cid != invocation.invocation_cid:
            raise IdentityInconsistentError(
                "original proposal invocation identity drifted",
                details={"field": "invocation_cid"},
            )
        if proposal.patch_cid is None:
            raise MalformedError("original proposal is missing patch artifact evidence")
        if proposal.patch_cid != cid_for_bytes(patch):
            raise IdentityInconsistentError(
                "replay patch CID does not verify",
                details={"field": "patch_cid"},
            )
        if self.fixture_cid != cid_for_bytes(wire_canonical_utf8(self._body()).encode("utf-8")):
            raise IdentityInconsistentError(
                "replay fixture CID does not verify",
                details={"field": "fixture_cid"},
            )

        # Prove at construction time that the binding plus exact recorded log
        # fits the contract's returned log channel.
        self.result_log_bytes()

    def _body(self) -> Mapping[str, Any]:
        return {
            "schema": FIXTURE_SCHEMA,
            "adapter_id": self.adapter_id,
            "task_cid": self.task_cid,
            "pack_cid": self.pack_cid,
            "route_cid": self.route_cid,
            "response_artifact_cid": self.response_artifact_cid,
            "response_artifact_base64": _encode(self.response_artifact),
            "original_invocation": dict(self.original_invocation.to_mapping()),
            "original_proposal": dict(self.original_proposal.to_mapping()),
            "patch_base64": _encode(self.patch_bytes),
            "log_base64": _encode(self.log_bytes),
        }

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType({"fixture_cid": self.fixture_cid, **self._body()})

    def replay_binding(self) -> Mapping[str, Any]:
        """Return the exact selector binding exposed by a replay result."""

        body = {
            "schema": REPLAY_BINDING_SCHEMA,
            "adapter_id": self.adapter_id,
            "fixture_cid": self.fixture_cid,
            "response_artifact_cid": self.response_artifact_cid,
            "task_cid": self.task_cid,
            "pack_cid": self.pack_cid,
            "route_cid": self.route_cid,
            "invocation_cid": self.original_invocation.invocation_cid,
            "proposal_cid": self.original_proposal.proposal_cid,
        }
        return MappingProxyType(
            {
                **body,
                "binding_cid": cid_for_bytes(wire_canonical_utf8(body).encode("utf-8")),
            }
        )

    def result_log_bytes(self) -> bytes:
        """Expose the binding, followed byte-for-byte by the recorded log."""

        binding = wire_canonical_utf8(dict(self.replay_binding())).encode("utf-8")
        return admit_bounded_log(binding + b"\n" + self.log_bytes)

    def replayed_records(self) -> tuple[CodingAgentInvocation, PatchProposal]:
        """Create replay-labeled copies while preserving source identities/usage."""

        return (
            _replayed_invocation(self.original_invocation),
            _replayed_proposal(self.original_proposal),
        )

    @classmethod
    def create(
        cls,
        *,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        response_artifact: bytes,
        original_invocation: CodingAgentInvocation,
        original_proposal: PatchProposal,
        patch_bytes: bytes,
        log_bytes: bytes,
    ) -> ReplayFixture:
        """Create a fixture from exact admitted original records and evidence."""

        bind_adapter_request(task, context_pack, route)
        if not isinstance(original_invocation, CodingAgentInvocation):
            raise MalformedError("original_invocation must be an admitted wire record")
        if not isinstance(original_proposal, PatchProposal):
            raise MalformedError("original_proposal must be an admitted wire record")
        response = _admit_exact_bytes(
            response_artifact,
            field_name="response_artifact",
            max_bytes=MAX_PROVIDER_OUTPUT_BYTES,
            allow_empty=False,
        )
        patch = _admit_exact_bytes(
            patch_bytes,
            field_name="patch_bytes",
            max_bytes=MAX_PATCH_BYTES,
            allow_empty=False,
        )
        log = _admit_exact_bytes(
            log_bytes,
            field_name="log_bytes",
            max_bytes=MAX_LOG_BYTES,
            allow_empty=True,
        )
        checks = (
            ("task_id", task.task_id, original_invocation.task_id),
            (
                "repository_state_cid",
                task.repository_state_cid,
                original_invocation.repository_state_cid,
            ),
            ("provider", route.provider, original_invocation.provider),
            ("model", route.model, original_invocation.model),
            (
                "revision",
                route.revision or "unspecified",
                original_invocation.revision,
            ),
            ("tier", route.tier, original_invocation.tier),
            ("route_cid", route.decision_cid, original_invocation.route_cid),
            ("task_id", task.task_id, original_proposal.task_id),
            (
                "repository_state_cid",
                task.repository_state_cid,
                original_proposal.repository_state_cid,
            ),
        )
        for field_name, expected, actual in checks:
            if expected != actual:
                raise IdentityInconsistentError(
                    "replay fixture source identity drifted",
                    details={"field": field_name},
                )
        body = {
            "schema": FIXTURE_SCHEMA,
            "adapter_id": ADAPTER,
            "task_cid": cid_for_record(task),
            "pack_cid": context_pack.pack_cid,
            "route_cid": route.decision_cid,
            "response_artifact_cid": cid_for_bytes(response),
            "response_artifact_base64": _encode(response),
            "original_invocation": dict(original_invocation.to_mapping()),
            "original_proposal": dict(original_proposal.to_mapping()),
            "patch_base64": _encode(patch),
            "log_base64": _encode(log),
        }
        return cls(
            fixture_cid=cid_for_bytes(wire_canonical_utf8(body).encode("utf-8")),
            task_cid=body["task_cid"],
            pack_cid=body["pack_cid"],
            route_cid=body["route_cid"],
            response_artifact_cid=body["response_artifact_cid"],
            response_artifact=response,
            original_invocation=original_invocation,
            original_proposal=original_proposal,
            patch_bytes=patch,
            log_bytes=log,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReplayFixture:
        if not isinstance(payload, Mapping) or set(payload) != _FIXTURE_FIELDS:
            raise MalformedError("replay fixture has an invalid closed field set")
        if payload.get("schema") != FIXTURE_SCHEMA:
            raise MalformedError("replay fixture schema is invalid")
        original_invocation = CodingAgentInvocation.from_mapping(payload["original_invocation"])
        original_proposal = PatchProposal.from_mapping(payload["original_proposal"])
        return cls(
            fixture_cid=payload["fixture_cid"],
            task_cid=payload["task_cid"],
            pack_cid=payload["pack_cid"],
            route_cid=payload["route_cid"],
            response_artifact_cid=payload["response_artifact_cid"],
            response_artifact=_decode(
                payload["response_artifact_base64"],
                field_name="response_artifact_base64",
                max_bytes=MAX_PROVIDER_OUTPUT_BYTES,
                allow_empty=False,
            ),
            original_invocation=original_invocation,
            original_proposal=original_proposal,
            patch_bytes=_decode(
                payload["patch_base64"],
                field_name="patch_base64",
                max_bytes=MAX_PATCH_BYTES,
                allow_empty=False,
            ),
            log_bytes=_decode(
                payload["log_base64"],
                field_name="log_base64",
                max_bytes=MAX_LOG_BYTES,
                allow_empty=True,
            ),
            adapter_id=payload["adapter_id"],
        )


@dataclass(frozen=True)
class ReplayAdapter:
    """Exact-selector in-memory lookup with no external-effect capability."""

    fixtures: Iterable[ReplayFixture]
    selected_fixture_cid: str
    selected_response_artifact_cid: str
    adapter_id: str = ADAPTER
    accepted: bool = field(init=False, default=False)
    approved: bool = field(init=False, default=False)
    approval_authority: bool = field(init=False, default=APPROVAL_AUTHORITY)
    canonical_branch_authority: bool = field(
        init=False,
        default=CANONICAL_BRANCH_AUTHORITY,
    )
    _fixture_index: Mapping[
        tuple[str, str, str, str, str, str],
        ReplayFixture,
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.adapter_id != ADAPTER:
            raise IdentityInconsistentError(
                "replay selector adapter identity drifted",
                details={"field": "adapter_id"},
            )
        object.__setattr__(
            self,
            "selected_fixture_cid",
            admit_cid(
                self.selected_fixture_cid,
                field="selected_fixture_cid",
            ),
        )
        object.__setattr__(
            self,
            "selected_response_artifact_cid",
            admit_cid(
                self.selected_response_artifact_cid,
                field="selected_response_artifact_cid",
            ),
        )

        fixtures = tuple(self.fixtures)
        object.__setattr__(self, "fixtures", fixtures)
        indexed: dict[
            tuple[str, str, str, str, str, str],
            ReplayFixture,
        ] = {}
        selector_matches = 0
        for fixture in fixtures:
            if not isinstance(fixture, ReplayFixture):
                raise MalformedError("replay adapter fixtures must be ReplayFixture records")
            key = (
                fixture.adapter_id,
                fixture.fixture_cid,
                fixture.response_artifact_cid,
                fixture.task_cid,
                fixture.pack_cid,
                fixture.route_cid,
            )
            if key in indexed:
                raise BoundaryViolationError("duplicate replay fixture identity")
            indexed[key] = fixture
            if key[:3] == (
                self.adapter_id,
                self.selected_fixture_cid,
                self.selected_response_artifact_cid,
            ):
                selector_matches += 1
        if selector_matches != 1:
            raise IdentityInconsistentError(
                "replay selector does not identify exactly one recorded response"
            )
        object.__setattr__(self, "_fixture_index", MappingProxyType(indexed))

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult:
        if cancellation is not None:
            cancellation.check()
        bind_adapter_request(task, context_pack, route)
        key = (
            self.adapter_id,
            self.selected_fixture_cid,
            self.selected_response_artifact_cid,
            cid_for_record(task),
            context_pack.pack_cid,
            route.decision_cid,
        )
        fixture = self._fixture_index.get(key)
        if fixture is None:
            raise IdentityInconsistentError(
                "no replay fixture matches the exact selector and request"
            )
        checks = (
            ("task_id", task.task_id, fixture.original_invocation.task_id),
            (
                "repository_state_cid",
                task.repository_state_cid,
                fixture.original_invocation.repository_state_cid,
            ),
            ("task_id", task.task_id, fixture.original_proposal.task_id),
            (
                "repository_state_cid",
                task.repository_state_cid,
                fixture.original_proposal.repository_state_cid,
            ),
            ("provider", route.provider, fixture.original_invocation.provider),
            ("model", route.model, fixture.original_invocation.model),
            (
                "revision",
                route.revision or "unspecified",
                fixture.original_invocation.revision,
            ),
            ("tier", route.tier, fixture.original_invocation.tier),
            (
                "route_cid",
                route.decision_cid,
                fixture.original_invocation.route_cid,
            ),
        )
        for field_name, expected, actual in checks:
            if expected != actual:
                raise IdentityInconsistentError(
                    "replay fixture identity drifted",
                    details={"field": field_name},
                )
        invocation, proposal = fixture.replayed_records()
        result = AdapterResult(
            proposal,
            invocation,
            patch_bytes=fixture.patch_bytes,
            log_bytes=fixture.result_log_bytes(),
        )
        return admit_adapter_result(
            task,
            context_pack,
            route,
            result,
            cancellation=cancellation,
        )


__all__ = [
    "ADAPTER",
    "FIXTURE_SCHEMA",
    "ORIGINAL_PROVENANCE",
    "REPLAY_BINDING_SCHEMA",
    "REPLAY_PROVENANCE",
    "ReplayAdapter",
    "ReplayFixture",
    "cid_for_bytes",
    "cid_for_record",
]
