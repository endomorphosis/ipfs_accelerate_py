"""SCA-111 bounded Grok implementation and Codex review routing tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    MAX_PROVIDER_PROMPT_BYTES,
    MAX_PROVIDER_PROMPT_TOKENS,
    MAX_PROVIDER_RESPONSE_BYTES,
    REDACTION_MARKER,
    ImplementationProviderRouter,
    ProviderBounds,
    ProviderQuotaError,
    ProviderQuotaLatch,
    ProviderReason,
    ProviderRole,
    RouteStatus,
    VerifiedGrokQuotaExhaustion,
    redact_provider_data,
    route_contract_packet,
)


SNAPSHOT = "git-tree:current"
PATH = "external/ipfs_accelerate/ipfs_accelerate_py/mcp/dispatch.py"


@dataclass(frozen=True)
class _Packet:
    packet_id: str = "packet:sca-111"
    snapshot_id: str = SNAPSHOT
    task_id: str = "SCA-111-fixture"
    implementable: bool = True
    payload: Mapping[str, Any] | None = None

    def assert_current(self, current_snapshot_id: str) -> None:
        if current_snapshot_id != self.snapshot_id:
            raise ValueError("stale")

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return self.payload or MappingProxyType(
            {
                "goal": {
                    "contract_ids": ["contract:repo.inspect"],
                    "obligation_ids": ["obligation:arguments"],
                    "counterexample": {
                        "data_label": "untrusted_repository_data",
                        "instruction_authority": False,
                        "value": {"expected": "string", "actual": "integer"},
                    },
                },
                "authority": {
                    "provider_semantic_authority": False,
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                },
                "scope": {
                    "read_paths": [PATH],
                    "write_paths": [PATH],
                },
                "acceptance": {
                    "validation_commands": ["python -m pytest test_contract.py -q"],
                    "reproof_commands": ["python -m proof.recheck obligation:arguments"],
                },
            }
        )


def _accept(proposal):
    return {"accepted": True, "reason_code": f"admitted:{proposal.role.value}"}


def _grok(request):
    assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
    return {
        "proposal": {
            "patch": f"diff --git a/{PATH} b/{PATH}\n",
            "declared_paths": [PATH],
        }
    }


def _codex(request):
    assert request["role"] == ProviderRole.CODEX_REVIEW.value
    assert "admitted_implementation_proposal" in request["provider_input"]
    assert request["provider_input"]["admitted_implementation_proposal"][
        "completion_authoritative"
    ] is False
    return {"decision": "approve", "findings": []}


def test_sequential_grok_then_codex_and_only_admitted_writer_can_mutate() -> None:
    events: list[str] = []

    def grok(request):
        events.append("grok")
        return _grok(request)

    def admit(proposal):
        events.append(f"admit:{proposal.role.value}")
        return True

    def codex(request):
        events.append("codex")
        assert events == [
            "grok",
            "admit:grok-implement",
            "codex",
        ]
        return _codex(request)

    writes = []

    def writer(proposal, lease_id):
        events.append("write")
        writes.append((proposal, lease_id))

    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=admit,
        writer=writer,
    )
    result = router.route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:swissknife:1",
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert result.admitted and result.write_performed
    assert events == [
        "grok",
        "admit:grok-implement",
        "codex",
        "admit:codex-independent-review",
        "write",
    ]
    assert len(writes) == 1
    assert writes[0][1] == "lease:swissknife:1"
    assert result.proof_authoritative is False
    assert result.completion_authoritative is False


def test_no_provider_receives_repository_path_corpus_or_expansion_bodies() -> None:
    seen = []

    def capture(request):
        seen.append(request.to_dict())
        assert "repository_root" not in request
        assert "workspace" not in request
        return {"proposal": {"patch": "bounded"}}

    router = ImplementationProviderRouter(
        grok_provider=capture,
        codex_provider=lambda request: (
            seen.append(request.to_dict())
            or {"decision": "approve", "findings": []}
        ),
        admission_gate=_accept,
    )
    result = router.route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.SUCCEEDED
    assert len(seen) == 2
    encoded = json.dumps(seen, sort_keys=True)
    assert "repository_root" not in encoded
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    assert set(seen[0]) == {
        "schema",
        "interface",
        "role",
        "packet_id",
        "snapshot_id",
        "task_id",
        "provider_input",
        "bounds",
        "response_contract",
        "authority",
    }
    assert seen[0]["authority"]["repository_write_allowed"] is False


@pytest.mark.parametrize(
    "broad_key",
    ["repository_corpus", "source_code", "ast_body", "workspace_path"],
)
def test_broad_context_is_rejected_before_any_provider_call(broad_key: str) -> None:
    calls = 0

    def forbidden(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    packet = _Packet(payload={"goal": {"slice": {broad_key: "broad"}}})
    result = ImplementationProviderRouter(
        grok_provider=forbidden,
        admission_gate=_accept,
    ).route(packet, current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.BROAD_CONTEXT_FORBIDDEN.value
    assert calls == 0


@pytest.mark.parametrize(
    "authority_attack",
    [
        {"completion_authoritative": True},
        {"receipt": {"proof_authoritative": True}},
        {"task_status": "complete"},
        {"proof_status": "proved"},
        {"mark_complete": 1},
    ],
)
def test_provider_cannot_change_proof_or_completion(
    authority_attack: Mapping[str, Any],
) -> None:
    writes = []
    result = ImplementationProviderRouter(
        grok_provider=lambda _request: authority_attack,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:1",
    )

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.PROVIDER_AUTHORITY_CLAIM.value
    assert result.proof_authoritative is False
    assert result.completion_authoritative is False
    assert writes == []


def test_review_repair_requires_a_further_review_and_never_writes() -> None:
    admissions = []
    writes = []

    def admit(proposal):
        admissions.append(proposal.role)
        return True

    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=lambda _request: {
            "decision": "repair",
            "proposal": {"patch": "codex repair", "declared_paths": [PATH]},
        },
        admission_gate=admit,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:one-writer",
    )

    assert admissions == [
        ProviderRole.GROK_IMPLEMENT,
        ProviderRole.CODEX_REVIEW,
    ]
    assert writes == []
    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.REVIEW_REJECTED.value
    assert result.selected_proposal is None
    assert not result.write_performed


def test_approve_with_findings_is_rejected_before_writer() -> None:
    writes = []
    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=lambda _request: {
            "decision": "approve",
            "findings": ["the proposal is unsafe"],
        },
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:contradictory-review",
    )

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.PROVIDER_RESPONSE_MALFORMED.value
    assert result.selected_proposal is None
    assert result.write_performed is False
    assert writes == []


def test_approve_without_findings_is_rejected_before_writer() -> None:
    writes = []
    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=lambda _request: {"decision": "approve"},
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:missing-review-findings",
    )

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.PROVIDER_RESPONSE_MALFORMED.value
    assert result.selected_proposal is None
    assert result.write_performed is False
    assert writes == []


def test_missing_admission_or_writer_lease_never_writes() -> None:
    writes = []
    no_gate = ImplementationProviderRouter(
        grok_provider=_grok,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:1",
    )
    assert no_gate.reason_code == ProviderReason.ADMISSION_REQUIRED.value

    no_lease = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(_Packet(), current_snapshot_id=SNAPSHOT, apply=True)
    assert no_lease.reason_code == ProviderReason.WRITER_LEASE_REQUIRED.value
    assert writes == []


def test_grok_quota_falls_back_locally_without_touching_codex_quota() -> None:
    calls = []
    router = ImplementationProviderRouter(
        grok_provider=lambda _request: calls.append("grok"),
        codex_provider=lambda _request: calls.append("codex"),
        deterministic_provider=lambda request: (
            calls.append(request.role.value)
            or {"proposal": {"patch": "deterministic"}}
        ),
        admission_gate=_accept,
        grok_quota=ProviderQuotaLatch(remaining_calls=0),
        codex_quota=ProviderQuotaLatch(remaining_calls=2),
    )
    result = router.route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.FALLBACK
    assert result.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value
    assert calls == [ProviderRole.DETERMINISTIC_LOCAL.value]
    assert router.codex_quota.remaining_calls == 2
    assert router.codex_quota.attempts == 0


def test_grok_quota_without_fallback_defers_with_typed_reason() -> None:
    result = ImplementationProviderRouter(
        grok_provider=_grok,
        admission_gate=_accept,
        grok_quota=0,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.DEFERRED
    assert result.deferred
    assert result.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value


def test_verified_grok_balance_exhaustion_routes_to_terra_pending_review() -> None:
    calls: list[str] = []
    writes: list[object] = []

    def exhausted_grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    def terra(request):
        calls.append("terra")
        assert request.role is ProviderRole.CODEX_QUOTA_IMPLEMENT
        assert "contract_packet" in request.payload
        return {
            "proposal": {
                "patch": "terra proposal",
                "declared_paths": [PATH],
            }
        }

    result = ImplementationProviderRouter(
        grok_provider=exhausted_grok,
        codex_implementation_fallback_provider=terra,
        codex_provider=lambda _request: calls.append("codex-review"),
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    ).route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:must-not-write",
    )

    assert calls == ["grok", "terra"]
    assert writes == []
    assert result.status is RouteStatus.DEFERRED
    assert result.reason_code == ProviderReason.NON_CODEX_REVIEW_REQUIRED.value
    assert result.provider == ProviderRole.CODEX_QUOTA_IMPLEMENT.value
    assert result.review_presence == "review_absent"
    assert result.provider_result_admitted is False
    assert result.selected_proposal is None
    assert result.implementation_proposal is not None
    assert result.implementation_proposal.admitted is True
    assert result.write_performed is False
    assert [item.role for item in result.attempts] == [
        ProviderRole.GROK_IMPLEMENT,
        ProviderRole.CODEX_QUOTA_IMPLEMENT,
    ]
    assert [item.role for item in result.review_chain] == [
        ProviderRole.GROK_IMPLEMENT.value,
        ProviderRole.CODEX_QUOTA_IMPLEMENT.value,
        ProviderRole.NON_CODEX_REVIEW.value,
    ]
    assert [item.status for item in result.review_chain] == [
        "failed",
        "succeeded",
        "absent",
    ]


def test_verified_grok_exhaustion_reuses_terra_after_latch_closes() -> None:
    calls: list[str] = []
    latch = ProviderQuotaLatch()

    def grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    def terra(_request):
        calls.append("terra")
        return {"proposal": {"patch": "terra proposal"}}

    router = ImplementationProviderRouter(
        grok_provider=grok,
        grok_quota=latch,
        codex_implementation_fallback_provider=terra,
        admission_gate=_accept,
    )
    results = [
        router.route(
            _Packet(packet_id=f"packet:repeated:{ordinal}"),
            current_snapshot_id=SNAPSHOT,
        )
        for ordinal in (1, 2)
    ]

    assert calls == ["grok", "terra", "terra"]
    assert latch.attempts == 1
    assert latch.exhausted is True
    assert (
        latch.reason_code
        == ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value
    )
    assert [result.status for result in results] == [
        RouteStatus.DEFERRED,
        RouteStatus.DEFERRED,
    ]
    assert [result.reason_code for result in results] == [
        ProviderReason.NON_CODEX_REVIEW_REQUIRED.value,
        ProviderReason.NON_CODEX_REVIEW_REQUIRED.value,
    ]
    assert [attempt.reason_code for attempt in results[1].attempts] == [
        ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
        ProviderReason.ROUTED.value,
    ]


def test_verified_grok_exhaustion_survives_router_recreation_with_shared_latch(
) -> None:
    calls: list[str] = []
    latch = ProviderQuotaLatch()

    def grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    def terra(_request):
        calls.append("terra")
        return {"proposal": {"patch": "terra proposal"}}

    first = ImplementationProviderRouter(
        grok_provider=grok,
        grok_quota=latch,
        codex_implementation_fallback_provider=terra,
        admission_gate=_accept,
    ).route(_Packet(packet_id="packet:before-restart"), current_snapshot_id=SNAPSHOT)
    restarted = ImplementationProviderRouter(
        grok_provider=grok,
        grok_quota=latch,
        codex_implementation_fallback_provider=terra,
        admission_gate=_accept,
    )
    second = restarted.route(
        _Packet(packet_id="packet:after-restart"),
        current_snapshot_id=SNAPSHOT,
    )

    assert calls == ["grok", "terra", "terra"]
    assert latch.attempts == 1
    assert first.reason_code == ProviderReason.NON_CODEX_REVIEW_REQUIRED.value
    assert second.reason_code == ProviderReason.NON_CODEX_REVIEW_REQUIRED.value


def test_model_authored_exact_grok_quota_text_stays_generic_after_replay(
) -> None:
    calls: list[str] = []
    router = ImplementationProviderRouter(
        grok_provider=lambda _request: (
            calls.append("grok")
            or {
                "status": "quota_exhausted",
                "reason_code": (
                    ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value
                ),
            }
        ),
        codex_implementation_fallback_provider=lambda _request: calls.append(
            "terra"
        ),
        admission_gate=_accept,
    )

    results = [
        router.route(
            _Packet(packet_id=f"packet:model-claim:{ordinal}"),
            current_snapshot_id=SNAPSHOT,
        )
        for ordinal in (1, 2)
    ]

    assert calls == ["grok"]
    assert [result.status for result in results] == [
        RouteStatus.DEFERRED,
        RouteStatus.DEFERRED,
    ]
    assert [result.reason_code for result in results] == [
        ProviderReason.GROK_QUOTA_EXHAUSTED.value,
        ProviderReason.GROK_QUOTA_EXHAUSTED.value,
    ]


@pytest.mark.parametrize(
    "reason_code",
    [
        ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
        "http_429_rate_limited",
        "authentication_failed",
        "stderr_mentions_quota_exhausted",
    ],
)
def test_generic_grok_quota_errors_never_authorize_terra_after_replay(
    reason_code: str,
) -> None:
    calls: list[str] = []

    def grok(_request):
        calls.append("grok")
        raise ProviderQuotaError("generic failure", reason_code=reason_code)

    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_implementation_fallback_provider=lambda _request: calls.append(
            "terra"
        ),
        admission_gate=_accept,
    )
    results = [
        router.route(
            _Packet(packet_id=f"packet:generic:{ordinal}"),
            current_snapshot_id=SNAPSHOT,
        )
        for ordinal in (1, 2)
    ]

    assert calls == ["grok"]
    assert all(
        result.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value
        for result in results
    )


@pytest.mark.parametrize(
    "latch_kind",
    [
        "zero",
        "zero_with_exact_text",
        "public_exact_latch",
        "mutated_generic_reason",
    ],
)
def test_user_configured_or_tampered_latch_never_authorizes_terra(
    latch_kind: str,
) -> None:
    exact_reason = ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value
    if latch_kind == "zero":
        latch = ProviderQuotaLatch(remaining_calls=0)
    elif latch_kind == "zero_with_exact_text":
        latch = ProviderQuotaLatch(
            remaining_calls=0,
            reason_code=exact_reason,
        )
    else:
        latch = ProviderQuotaLatch()
        latch.latch(
            exact_reason
            if latch_kind == "public_exact_latch"
            else "generic_quota"
        )
        if latch_kind == "mutated_generic_reason":
            latch.reason_code = exact_reason
    calls: list[str] = []
    router = ImplementationProviderRouter(
        grok_provider=lambda _request: calls.append("grok"),
        grok_quota=latch,
        codex_implementation_fallback_provider=lambda _request: calls.append(
            "terra"
        ),
        admission_gate=_accept,
    )

    first = router.route(_Packet(), current_snapshot_id=SNAPSHOT)
    second = router.route(
        _Packet(packet_id="packet:user-latch:2"),
        current_snapshot_id=SNAPSHOT,
    )

    assert calls == []
    assert first.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value
    assert second.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value


def test_public_relatch_revokes_verified_grok_exhaustion_provenance() -> None:
    calls: list[str] = []
    latch = ProviderQuotaLatch()

    def grok(_request):
        calls.append("grok")
        raise VerifiedGrokQuotaExhaustion()

    router = ImplementationProviderRouter(
        grok_provider=grok,
        grok_quota=latch,
        codex_implementation_fallback_provider=lambda _request: (
            calls.append("terra")
            or {"proposal": {"patch": "terra proposal"}}
        ),
        admission_gate=_accept,
    )
    first = router.route(_Packet(), current_snapshot_id=SNAPSHOT)
    latch.latch(ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value)
    second = router.route(
        _Packet(packet_id="packet:relatched"),
        current_snapshot_id=SNAPSHOT,
    )

    assert calls == ["grok", "terra"]
    assert first.reason_code == ProviderReason.NON_CODEX_REVIEW_REQUIRED.value
    assert second.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value


@pytest.mark.parametrize(
    "grok_result",
    [
        ProviderQuotaError(
            "unverified quota text",
            reason_code=ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
        ),
        {
            "status": "quota_exhausted",
            "reason_code": ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
        },
    ],
)
def test_unverified_or_model_authored_quota_claim_never_routes_to_terra(
    grok_result: object,
) -> None:
    calls: list[str] = []

    def grok(_request):
        calls.append("grok")
        if isinstance(grok_result, BaseException):
            raise grok_result
        return grok_result

    result = ImplementationProviderRouter(
        grok_provider=grok,
        codex_implementation_fallback_provider=lambda _request: calls.append(
            "terra"
        ),
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert calls == ["grok"]
    assert result.status is RouteStatus.DEFERRED
    assert result.reason_code == ProviderReason.GROK_QUOTA_EXHAUSTED.value


def test_codex_quota_preserves_grok_as_evidence_without_writing() -> None:
    writes = []
    router = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
        grok_quota=ProviderQuotaLatch(remaining_calls=3),
        codex_quota=ProviderQuotaLatch(remaining_calls=0),
    )
    result = router.route(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:1",
    )

    assert result.status is RouteStatus.FALLBACK
    assert result.reason_code == ProviderReason.CODEX_QUOTA_EXHAUSTED.value
    assert not result.write_performed
    assert writes == []
    assert router.grok_quota.attempts == 1
    assert router.codex_quota.attempts == 0


def test_runtime_quota_error_latches_only_the_failing_provider() -> None:
    router = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=lambda _request: (_ for _ in ()).throw(
            ProviderQuotaError("codex daily quota", reason_code="codex_daily_quota")
        ),
        admission_gate=_accept,
    )
    result = router.route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.FALLBACK
    assert result.reason_code == ProviderReason.CODEX_QUOTA_EXHAUSTED.value
    assert router.codex_quota.exhausted
    assert router.codex_quota.reason_code == "codex_daily_quota"
    assert not router.grok_quota.exhausted


def test_explicit_local_only_path_invokes_no_models() -> None:
    calls = []
    result = ImplementationProviderRouter(
        grok_provider=lambda _request: calls.append("grok"),
        codex_provider=lambda _request: calls.append("codex"),
        deterministic_provider=lambda _request: (
            calls.append("local") or {"proposal": {"patch": "local"}}
        ),
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT, local_only=True)

    assert result.status is RouteStatus.FALLBACK
    assert result.reason_code == ProviderReason.LOCAL_ONLY.value
    assert calls == ["local"]
    assert result.selected_proposal.role is ProviderRole.DETERMINISTIC_LOCAL


def test_stale_or_nonimplementable_packet_is_rejected_before_provider() -> None:
    calls = []
    router = ImplementationProviderRouter(
        grok_provider=lambda _request: calls.append(True),
        admission_gate=_accept,
    )
    stale = router.route(_Packet(), current_snapshot_id="git-tree:new")
    blocked = router.route(
        _Packet(implementable=False), current_snapshot_id=SNAPSHOT
    )

    assert stale.reason_code == ProviderReason.PACKET_STALE.value
    assert blocked.reason_code == ProviderReason.PACKET_NOT_IMPLEMENTABLE.value
    assert calls == []


def test_prompt_exact_byte_and_token_limits_are_inclusive() -> None:
    observed = {}

    def capture(request):
        observed["bytes"] = len(request.prompt)
        observed["tokens"] = request.prompt_tokens
        return {"proposal": {"patch": "x"}}

    generous = ImplementationProviderRouter(
        grok_provider=capture,
        admission_gate=_accept,
        bounds=ProviderBounds(
            max_prompt_bytes=MAX_PROVIDER_PROMPT_BYTES,
            max_prompt_tokens=MAX_PROVIDER_PROMPT_TOKENS,
        ),
        token_counter=lambda _prompt: 17,
    )
    first = generous.route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert first.status is RouteStatus.FALLBACK  # no Codex configured

    exact = ImplementationProviderRouter(
        grok_provider=capture,
        admission_gate=_accept,
        bounds=ProviderBounds(
            max_prompt_bytes=observed["bytes"],
            max_prompt_tokens=observed["tokens"],
        ),
        token_counter=lambda _prompt: observed["tokens"],
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert exact.status is RouteStatus.FALLBACK

    byte_over = ImplementationProviderRouter(
        grok_provider=capture,
        admission_gate=_accept,
        bounds=ProviderBounds(
            max_prompt_bytes=observed["bytes"] - 1,
            max_prompt_tokens=MAX_PROVIDER_PROMPT_TOKENS,
        ),
        token_counter=lambda _prompt: 17,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert byte_over.reason_code == ProviderReason.PROMPT_TOO_LARGE.value

    token_over = ImplementationProviderRouter(
        grok_provider=capture,
        admission_gate=_accept,
        bounds=ProviderBounds(
            max_prompt_bytes=MAX_PROVIDER_PROMPT_BYTES,
            max_prompt_tokens=16,
        ),
        token_counter=lambda _prompt: 17,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert token_over.reason_code == ProviderReason.PROMPT_TOKEN_BUDGET.value


def test_response_exact_utf8_byte_limit_is_inclusive() -> None:
    payload = {"proposal": {"patch": "é"}}
    exact_bytes = len(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    exact = ImplementationProviderRouter(
        grok_provider=lambda _request: payload,
        admission_gate=_accept,
        bounds=ProviderBounds(max_response_bytes=exact_bytes),
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert exact.status is RouteStatus.FALLBACK
    assert exact.implementation_proposal.response_bytes == exact_bytes

    over = ImplementationProviderRouter(
        grok_provider=lambda _request: payload,
        admission_gate=_accept,
        bounds=ProviderBounds(max_response_bytes=exact_bytes - 1),
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert over.status is RouteStatus.REJECTED
    assert over.reason_code == ProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value


def test_prompt_and_response_secrets_are_redacted_and_receipts_embed_neither() -> None:
    secret = "super-secret-value"
    seen = {}

    packet = _Packet(
        payload={
            "goal": {
                "counterexample": f"Authorization: Bearer {secret}",
                "api_key": secret,
            },
            "scope": {"read_paths": [PATH], "write_paths": [PATH]},
        }
    )

    def grok(request):
        seen["prompt"] = request.prompt.decode()
        return {
            "proposal": {"patch": "x"},
            "diagnostic": f"password={secret}",
            "credentials": secret,
        }

    result = ImplementationProviderRouter(
        grok_provider=grok,
        admission_gate=_accept,
    ).route(packet, current_snapshot_id=SNAPSHOT)

    assert secret not in seen["prompt"]
    assert REDACTION_MARKER in seen["prompt"]
    assert secret not in json.dumps(result.to_dict(), sort_keys=True)
    assert result.implementation_proposal.payload["credentials"] == REDACTION_MARKER
    assert result.implementation_proposal.payload["diagnostic"].endswith(
        REDACTION_MARKER
    )
    attempt = result.attempts[0].to_dict()
    assert attempt["prompt_embedded"] is False
    assert attempt["response_embedded"] is False


def test_redaction_key_matching_does_not_hide_nonsensitive_token_limits() -> None:
    redacted = redact_provider_data(
        {
            "access_token": "secret",
            "token": "another-secret",
            "max_input_tokens": 4096,
            "token_count": 12,
        }
    )
    assert redacted == {
        "access_token": REDACTION_MARKER,
        "token": REDACTION_MARKER,
        "max_input_tokens": 4096,
        "token_count": 12,
    }


def test_malformed_duplicate_json_and_oversized_output_are_typed() -> None:
    duplicate = ImplementationProviderRouter(
        grok_provider=lambda _request: '{"proposal":{},"proposal":{}}',
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert duplicate.reason_code == ProviderReason.PROVIDER_RESPONSE_MALFORMED.value

    oversized = ImplementationProviderRouter(
        grok_provider=lambda _request: {
            "proposal": {"patch": "x" * MAX_PROVIDER_RESPONSE_BYTES}
        },
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert oversized.reason_code == ProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value


def test_functional_facade_preserves_proposal_only_default() -> None:
    writes = []
    result = route_contract_packet(
        _Packet(),
        current_snapshot_id=SNAPSHOT,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda proposal, lease: writes.append((proposal, lease)),
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert result.admitted
    assert not result.write_performed
    assert writes == []


def test_route_receipt_has_provider_packet_review_chain_and_provider_receipt() -> None:
    """SCA-228: successful model-assisted routes emit nonempty receipt fields."""

    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.SUCCEEDED
    assert result.provider == ProviderRole.GROK_IMPLEMENT.value
    assert result.packet is not None
    assert result.packet.packet_id == "packet:sca-111"
    assert result.packet.packet_cid
    assert result.packet.packet_bytes > 0
    chain = result.review_chain
    assert len(chain) == 2
    assert chain[0].role == ProviderRole.GROK_IMPLEMENT.value
    assert chain[0].admitted is True
    assert chain[0].status == "succeeded"
    assert chain[1].role == ProviderRole.CODEX_REVIEW.value
    assert chain[1].admitted is True
    assert chain[1].status == "succeeded"
    receipt = result.provider_receipt
    assert receipt.receipt_id
    assert receipt.provider == ProviderRole.GROK_IMPLEMENT.value
    assert receipt.packet["packet_cid"] == result.packet.packet_cid
    assert receipt.review_presence == "independent_review"
    assert receipt.provider_result_admitted is True
    assert receipt.completion_authoritative is False
    assert receipt.proof_authoritative is False
    payload = result.to_dict()
    assert payload["provider"]
    assert payload["packet"]["packet_cid"]
    assert payload["review_chain"]
    assert payload["provider_receipt"]["receipt_id"]
    assert payload["completion_authoritative"] is False


def test_grok_cannot_self_review() -> None:
    """SCA-228: the same callable cannot implement and review."""

    def same_provider(request):
        if request["role"] == ProviderRole.GROK_IMPLEMENT.value:
            return {
                "proposal": {
                    "patch": f"diff --git a/{PATH} b/{PATH}\n",
                    "declared_paths": [PATH],
                }
            }
        return {"decision": "approve", "findings": []}

    result = ImplementationProviderRouter(
        grok_provider=same_provider,
        codex_provider=same_provider,
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.SELF_REVIEW_FORBIDDEN.value
    assert result.provider_result_admitted is False
    assert result.completion_authoritative is False
    assert result.packet is not None
    assert result.packet.packet_cid


def test_codex_receives_only_bounded_proposal_and_evidence_slice() -> None:
    """SCA-228: Codex never sees the full implementer contract packet body."""

    seen = {}

    def codex(request):
        seen["role"] = request["role"]
        seen["provider_input"] = request["provider_input"]
        assert "contract_packet" not in request["provider_input"]
        assert "admitted_implementation_proposal" in request["provider_input"]
        assert "evidence_slice" in request["provider_input"]
        slice_ = request["provider_input"]["evidence_slice"]
        assert "goal" not in slice_
        assert "counterexample" not in slice_
        # Goal bodies are reduced to identifiers only.
        assert set(slice_["goal_ids"]) <= {
            "contract_ids",
            "obligation_ids",
            "acceptance_ids",
            "claim_ids",
            "property_ids",
        }
        assert slice_["authority"]["completion_authoritative"] is False
        return {"decision": "approve", "findings": []}

    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=codex,
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.SUCCEEDED
    assert seen["role"] == ProviderRole.CODEX_REVIEW.value
    proposal = seen["provider_input"]["admitted_implementation_proposal"]
    assert proposal["role"] == ProviderRole.GROK_IMPLEMENT.value
    assert proposal["completion_authoritative"] is False


def test_absent_or_degraded_review_is_explicit_and_not_authoritative() -> None:
    """SCA-228: missing/degraded Codex review cannot satisfy completion."""

    absent = ImplementationProviderRouter(
        grok_provider=_grok,
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert absent.status is RouteStatus.FALLBACK
    assert absent.review_presence == "review_absent"
    assert absent.provider_result_admitted is False
    assert absent.completion_authoritative is False
    chain = absent.review_chain
    assert chain[-1].role == ProviderRole.CODEX_REVIEW.value
    assert chain[-1].status == "absent"
    assert chain[-1].admitted is False
    assert absent.provider_receipt.admission["independent_review"] is False
    assert absent.provider_receipt.completion_authoritative is False

    degraded = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=lambda _request: (_ for _ in ()).throw(
            RuntimeError("codex crashed")
        ),
        admission_gate=_accept,
    ).route(_Packet(), current_snapshot_id=SNAPSHOT)
    assert degraded.status is RouteStatus.FALLBACK
    assert degraded.review_presence == "review_degraded"
    assert degraded.provider_result_admitted is False
    assert degraded.completion_authoritative is False
    assert degraded.review_chain[-1].status == "degraded"
    assert degraded.provider_receipt.provider_result_admitted is False
