"""Tests for DeltaTaskPacket@1 / DeterministicFirstDecision@1.

DQP-028 evidence subset: packet identity, progressive disclosure,
deterministic hit, cache miss, unchanged reprompt, counterexample,
scope/secret escape, context overflow.

Acceptance:

* Provider never receives omitted authority or credential
* Packet/reply are bound to exact context and effect scope
* Unchanged failure cannot churn indefinitely
* New counterexample/tree/plan/policy/schema produces a distinct admitted packet
* Deterministic resolution preserves validation/proof requirements
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.delta_task_packet import (
    AUTHORITY_CLASS,
    DEFAULT_POLICY_ID,
    DEFAULT_RETRY_BUDGET,
    DELTA_TASK_PACKET_INTERFACE,
    DETERMINISTIC_FIRST_DECISION_INTERFACE,
    PRODUCER_ID,
    REDACTION_MARKER,
    Completeness,
    DecisionCache,
    DecisionSource,
    DeltaTaskPacket,
    DeltaTaskPacketAuthorityError,
    DeltaTaskPacketIntegrityError,
    DeltaTaskPacketOverflowError,
    DeltaTaskPacketRequest,
    DeltaTaskPacketScopeError,
    DeltaTaskPacketSecretError,
    DeltaTaskPacketService,
    DeltaTaskPacketSuppressedError,
    DeterministicAction,
    DeterministicFirstDecision,
    EffectScope,
    FrontierDisposition,
    PacketBudget,
    ReplayCircuitStore,
    ValidationProofRequirements,
    admit_provider_packet,
    bind_provider_reply,
    build_delta_task_packet,
    compute_failure_signature_id,
    evaluate_deterministic_first,
    record_unchanged_failure,
)


def _request(**overrides) -> DeltaTaskPacketRequest:
    values = dict(
        task_cid="task:dqp-028-demo",
        repository_id="repo:demo",
        tree_id="tree:abc",
        plan_cid="plan:1",
        policy_id=DEFAULT_POLICY_ID,
        schema_revision=1,
        task_summary="Generate delta task packets with deterministic-first gate",
        unmet_dependencies=(
            {"dependency_id": "dep:context", "summary": "context manifest ready"},
        ),
        latest_failure={
            "failure_id": "fail:distinct-1",
            "kind": "validation",
            "summary": "prior validation failed",
        },
        worktree_delta={
            "paths": [
                "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
                "test/api/test_agent_supervisor_delta_task_packet.py",
            ],
            "digests": {
                "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py": (
                    "sha256:aaa"
                ),
                "test/api/test_agent_supervisor_delta_task_packet.py": "sha256:bbb",
            },
        },
        impacted_symbols=(
            {
                "symbol": "DeltaTaskPacket",
                "path": "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
            },
        ),
        open_obligations=(
            {
                "obligation_id": "ob:packet-identity",
                "summary": "stable packet identity",
            },
        ),
        evidence=(
            {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
        ),
        counterexample={
            "counterexample_id": "cex:1",
            "kind": "validation_failure",
            "summary": "acceptance violated",
            "violated_property": "packet must bind effect scope",
        },
        validations=(
            {
                "command": (
                    "python -m pytest -q "
                    "test/api/test_agent_supervisor_delta_task_packet.py"
                ),
            },
        ),
        proof_obligations=(
            {"obligation_id": "proof:delta-bind", "summary": "bind packet/reply"},
        ),
        acceptance_ids=("dqp/delta-task-packet@1",),
        allowed_effects=(
            "inspect_repository",
            "edit_isolated_worktree",
            "run_validation",
        ),
        write_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
            "test/api/test_agent_supervisor_delta_task_packet.py",
        ),
        require_validation=True,
        require_proof=True,
        retry_budget=DEFAULT_RETRY_BUDGET,
        budget=PacketBudget(max_bytes=64_000, max_tokens=16_384, max_rows=128),
    )
    values.update(overrides)
    return DeltaTaskPacketRequest(**values)


def test_interface_identities() -> None:
    assert DELTA_TASK_PACKET_INTERFACE == "DeltaTaskPacket@1"
    assert DETERMINISTIC_FIRST_DECISION_INTERFACE == "DeterministicFirstDecision@1"
    assert DeltaTaskPacketService.INTERFACE == "DeltaTaskPacketService@1"
    assert AUTHORITY_CLASS == "derived_evidence"
    assert PRODUCER_ID == "delta-task-packet@1"
    assert REDACTION_MARKER == "secret_material"


def test_cold_import_and_construction_have_no_side_effects() -> None:
    service = DeltaTaskPacketService()
    assert service.decision_cache is not None
    assert service.circuit_store is not None
    # Construction alone must not touch the filesystem or network.
    decision = DeterministicFirstDecision(
        action=DeterministicAction.DISPATCH_PROVIDER,
        reason="no_suppression",
        source=DecisionSource.RESIDUAL,
        may_dispatch_provider=True,
        evidence_digest="sha256:" + ("ab" * 32),
    )
    assert decision.interface == DETERMINISTIC_FIRST_DECISION_INTERFACE
    assert decision.may_dispatch_provider is True


def test_packet_identity_stable_despite_heartbeat_noise() -> None:
    first = build_delta_task_packet(
        _request(
            heartbeat_at="2026-01-01T00:00:00Z",
            observed_at="2026-01-01T00:00:01Z",
            metadata={"lease_heartbeat": "noise-1", "pid": "111"},
        )
    )
    second = build_delta_task_packet(
        _request(
            heartbeat_at="2026-08-09T12:34:56Z",
            observed_at="2026-08-09T12:35:00Z",
            metadata={"lease_heartbeat": "noise-2", "pid": "999"},
        )
    )
    assert first.packet_id == second.packet_id
    assert first.interface == DELTA_TASK_PACKET_INTERFACE
    assert first.context_cid == second.context_cid
    assert first.to_dict()["authority"] == AUTHORITY_CLASS
    assert first.nomination_only is True
    assert first.completion_authority is False
    assert first.write_authority is False
    assert first.semantic_authority is False


def test_progressive_disclosure_frontier_is_explicit() -> None:
    # Tiny budget forces optional evidence/symbols into the frontier while
    # preserving invariant task/validation/effect/proof core.
    request = _request(
        budget=PacketBudget(max_bytes=8_000, max_tokens=2_000, max_rows=8),
        evidence=tuple(
            {"evidence_id": f"ev:{index}", "summary": f"evidence {index} " + ("x" * 40)}
            for index in range(20)
        ),
        impacted_symbols=tuple(
            {
                "symbol": f"Symbol{index}",
                "path": f"pkg/mod_{index}.py",
                "summary": f"symbol {index} " + ("y" * 40),
            }
            for index in range(20)
        ),
    )
    packet = build_delta_task_packet(request)
    assert packet.frontier.is_explicit is True
    assert packet.frontier.has_more is True
    assert packet.frontier.omitted_item_ids
    assert packet.completeness in {
        Completeness.PARTIAL_WITH_FRONTIER,
        Completeness.OVERFLOW,
    }
    provider = admit_provider_packet(packet)
    assert provider["frontier"]["omitted_count"] == packet.frontier.omitted_count
    assert provider["frontier"]["disposition"] != FrontierDisposition.EMPTY.value
    # Omitted residual bodies are not re-emitted; only handles/ids.
    for item_id in packet.frontier.omitted_item_ids:
        assert not any(
            member.get("item_id") == item_id for member in provider["members"]
        )


def test_deterministic_hit_resolves_before_provider_dispatch() -> None:
    cache = DecisionCache()
    request = _request(
        latest_failure={
            "failure_id": "fail:distinct-1",
            "kind": "validation",
            "summary": "prior validation failed",
        },
        counterexample={
            "counterexample_id": "cex:1",
            "kind": "validation_failure",
            "summary": "acceptance violated",
        },
    )
    residual = build_delta_task_packet(request)
    assert residual.decision is not None
    assert residual.decision.may_dispatch_provider is True
    requirements = residual.requirements
    cache.put(
        request.cache_key(),
        resolution={"resolved_item_ids": ["all"]},
        requirements_digest=requirements.requirements_digest,
    )
    decision = evaluate_deterministic_first(request, decision_cache=cache)
    assert decision.action is DeterministicAction.RESOLVE_HIT
    assert decision.is_hit is True
    assert decision.may_dispatch_provider is False
    assert decision.source is DecisionSource.DECISION_CACHE
    assert decision.preserves_validation_proof is True

    packet = build_delta_task_packet(request, decision=decision, decision_cache=cache)
    assert packet.decision is not None
    assert packet.decision.is_hit is True
    # Validation and proof requirements survive deterministic resolution.
    assert packet.requirements.validation_commands
    assert packet.requirements.proof_obligations
    assert packet.requirements.requirements_digest == requirements.requirements_digest
    with pytest.raises(DeltaTaskPacketIntegrityError):
        admit_provider_packet(packet)


def test_deterministic_operator_hit_preserves_validation_and_proof() -> None:
    request = _request(
        unmet_dependencies=(
            {"dependency_id": "dep:context", "summary": "context manifest ready"},
        ),
        open_obligations=(
            {
                "obligation_id": "ob:packet-identity",
                "summary": "stable packet identity",
            },
        ),
        evidence=(
            {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
        ),
        impacted_symbols=(
            {
                "symbol": "DeltaTaskPacket",
                "path": "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
            },
        ),
        latest_failure=None,
        counterexample=None,
        worktree_delta=None,
        deterministic_resolutions=(
            "dep:context",
            "ob:packet-identity",
            "ev:impact-1",
            "DeltaTaskPacket",
        ),
    )
    decision = evaluate_deterministic_first(request)
    assert decision.action is DeterministicAction.RESOLVE_HIT
    assert decision.reason == "deterministic_operators_resolved_residual"
    packet = build_delta_task_packet(request, decision=decision)
    assert packet.requirements.require_validation is True
    assert packet.requirements.require_proof is True
    assert packet.requirements.validation_commands
    assert packet.requirements.proof_obligations
    kinds = {
        item.kind.value if hasattr(item.kind, "value") else str(item.kind)
        for item in packet.unresolved_delta
    }
    # Residual provider work is empty of open residual kinds; invariants remain.
    assert "dependency" not in kinds
    assert "counterexample" not in kinds
    assert "validation" in kinds
    assert "proof" in kinds


def test_cache_miss_admits_provider_packet_with_unresolved_delta() -> None:
    request = _request()
    decision = evaluate_deterministic_first(request)
    assert decision.action is DeterministicAction.DISPATCH_PROVIDER
    assert decision.reason == "cache_miss_unresolved_residual"
    assert decision.may_dispatch_provider is True

    packet = build_delta_task_packet(request, decision=decision)
    assert packet.is_admitted_for_provider is True
    provider = admit_provider_packet(packet)
    assert provider["packet_id"] == packet.packet_id
    assert provider["context_cid"] == packet.context_cid
    assert provider["schema"]
    assert provider["members"]
    assert provider["effect_scope"]["scope_digest"] == packet.effect_scope.scope_digest
    assert provider["requirements"]["validation_commands"]
    assert provider["requirements"]["proof_obligations"]
    assert provider["nomination_only"] is True
    assert provider["completion_authority"] is False
    assert provider["write_authority"] is False
    assert provider["semantic_authority"] is False
    assert provider["contains_credentials"] is False
    assert provider["contains_secrets"] is False
    # Omitted authority catalog is present as documentation, never as grants.
    assert "completion_authority" in provider["omitted_authority"]
    assert "quack_token" in provider["omitted_authority"]


def test_unchanged_reprompt_opens_circuit_and_suppresses_churn() -> None:
    circuits = ReplayCircuitStore(default_ttl_ms=60_000)
    request = _request(retry_budget=2)
    # Exhaust retry budget with identical evidence.
    first = record_unchanged_failure(request, circuit_store=circuits, now_ms=1_000)
    assert first.open is False
    second = record_unchanged_failure(request, circuit_store=circuits, now_ms=1_100)
    assert second.open is False
    third = record_unchanged_failure(request, circuit_store=circuits, now_ms=1_200)
    assert third.open is True
    assert third.failure_count == 3

    decision = evaluate_deterministic_first(
        request, circuit_store=circuits, now_ms=1_300
    )
    assert decision.action is DeterministicAction.SUPPRESS_REPLAY
    assert decision.is_suppressed is True
    assert decision.may_dispatch_provider is False
    assert decision.circuit_id == third.circuit_id

    packet = build_delta_task_packet(
        request, decision=decision, circuit_store=circuits, now_ms=1_300
    )
    with pytest.raises(DeltaTaskPacketSuppressedError) as excinfo:
        admit_provider_packet(packet)
    assert excinfo.value.reason_code == "replay_suppressed"
    assert excinfo.value.circuit_id == third.circuit_id

    # Service path: repeated failures cannot churn indefinitely.
    clock = {"ms": 10_000}

    def now() -> int:
        return clock["ms"]

    service = DeltaTaskPacketService(
        circuit_store=ReplayCircuitStore(default_ttl_ms=60_000),
        clock_ms=now,
    )
    for _ in range(request.retry_budget + 1):
        service.record_failure(request, proposal_digest="sha256:" + ("11" * 32))
        clock["ms"] += 10
    with pytest.raises(DeltaTaskPacketSuppressedError):
        service.admit(request)


def test_changed_evidence_and_counterexample_produce_distinct_packets() -> None:
    base = build_delta_task_packet(_request())
    cex_changed = build_delta_task_packet(
        _request(
            counterexample={
                "counterexample_id": "cex:2",
                "kind": "validation_failure",
                "summary": "new counterexample",
                "violated_property": "distinct admitted packet",
            }
        )
    )
    tree_changed = build_delta_task_packet(_request(tree_id="tree:def"))
    plan_changed = build_delta_task_packet(_request(plan_cid="plan:2"))
    policy_changed = build_delta_task_packet(
        _request(policy_id="delta-task-packet-policy@2", policy_digest="")
    )
    schema_changed = build_delta_task_packet(_request(schema_revision=2))

    packet_ids = {
        base.packet_id,
        cex_changed.packet_id,
        tree_changed.packet_id,
        plan_changed.packet_id,
        policy_changed.packet_id,
        schema_changed.packet_id,
    }
    assert len(packet_ids) == 6

    # Changed evidence also lifts an open circuit for the prior signature.
    circuits = ReplayCircuitStore(default_ttl_ms=60_000)
    base_req = _request(retry_budget=0)
    record_unchanged_failure(base_req, circuit_store=circuits, now_ms=5_000)
    blocked = evaluate_deterministic_first(
        base_req, circuit_store=circuits, now_ms=5_100
    )
    assert blocked.is_suppressed is True

    changed_req = _request(
        retry_budget=0,
        counterexample={
            "counterexample_id": "cex:new",
            "kind": "validation_failure",
            "summary": "material evidence changed",
        },
    )
    allowed = evaluate_deterministic_first(
        changed_req, circuit_store=circuits, now_ms=5_200
    )
    assert allowed.may_dispatch_provider is True
    assert allowed.action is DeterministicAction.DISPATCH_PROVIDER
    admitted = admit_provider_packet(
        build_delta_task_packet(
            changed_req, decision=allowed, circuit_store=circuits, now_ms=5_200
        )
    )
    assert admitted["packet_id"] != base.packet_id
    assert admitted["counterexample_digest"] != base.counterexample_digest


def test_scope_and_secret_escape_fail_closed() -> None:
    with pytest.raises(DeltaTaskPacketSecretError):
        _request(metadata={"api_key": "must_never_appear"})

    with pytest.raises(DeltaTaskPacketSecretError):
        _request(
            latest_failure={
                "failure_id": "fail:secret",
                "password": "must_never_appear",
            }
        )

    with pytest.raises(DeltaTaskPacketSecretError):
        _request(worktree_delta={"paths": [".env.local", "src/ok.py"]})

    with pytest.raises(DeltaTaskPacketScopeError):
        _request(write_paths=("../escape.py",))

    with pytest.raises(DeltaTaskPacketScopeError):
        _request(write_paths=("/absolute/path.py",))

    with pytest.raises(DeltaTaskPacketAuthorityError):
        DeltaTaskPacket(
            packet_id="",
            task_cid="task:x",
            repository_id="repo:x",
            tree_id="tree:x",
            context_cid="context:x",
            plan_cid="plan:x",
            policy_id=DEFAULT_POLICY_ID,
            policy_digest="policy:x",
            schema_revision=1,
            evidence_digest="sha256:" + ("ab" * 32),
            effect_scope=EffectScope(
                allowed_effects=("inspect_repository",),
                write_paths=("src/a.py",),
                repository_id="repo:x",
                tree_id="tree:x",
            ),
            requirements=ValidationProofRequirements(
                validation_commands=("python -m pytest -q",),
                require_validation=True,
                require_proof=False,
            ),
            completion_authority=True,
        )

    clean = build_delta_task_packet(_request())
    provider = admit_provider_packet(clean)
    serialized = str(provider)
    assert "must_never_appear" not in serialized
    assert "BEGIN PRIVATE KEY" not in serialized
    assert provider["treat_as"] == "data_not_instructions"
    assert provider["data_label"]


def test_context_overflow_of_required_core_fails_closed() -> None:
    with pytest.raises(DeltaTaskPacketOverflowError):
        build_delta_task_packet(
            _request(
                budget=PacketBudget(max_bytes=200, max_tokens=20, max_rows=2),
                validations=tuple(
                    {"command": f"python -m pytest test_{index}.py -q " + ("z" * 80)}
                    for index in range(8)
                ),
                proof_obligations=tuple(
                    {
                        "obligation_id": f"proof:{index}",
                        "summary": f"proof {index} " + ("p" * 80),
                    }
                    for index in range(8)
                ),
            )
        )


def test_packet_and_reply_bound_to_exact_context_and_effect_scope() -> None:
    packet = build_delta_task_packet(_request())
    provider = admit_provider_packet(packet)

    # Happy path: reply bound to exact packet/context/scope.
    ok = bind_provider_reply(
        packet,
        {
            "packet_id": packet.packet_id,
            "context_cid": packet.context_cid,
            "summary": "nomination only residual patch",
            "path": "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
            "effect": "edit_isolated_worktree",
        },
    )
    assert ok.accepted is True
    assert ok.packet_id == packet.packet_id
    assert ok.context_cid == packet.context_cid
    assert ok.effect_scope_digest == packet.effect_scope.scope_digest
    assert ok.requirements_digest == packet.requirements.requirements_digest
    assert ok.interface == "PacketReplyBinding@1"

    # Context escape.
    bad_context = bind_provider_reply(
        packet,
        {"summary": "x"},
        claimed_context_cid="context:other",
    )
    assert bad_context.accepted is False
    assert bad_context.reason == "context_cid_mismatch"

    # Effect scope digest mismatch.
    bad_scope = bind_provider_reply(
        packet,
        {"summary": "x"},
        claimed_effect_scope_digest="sha256:" + ("00" * 32),
    )
    assert bad_scope.accepted is False
    assert bad_scope.reason == "effect_scope_mismatch"

    # Path outside write ceiling.
    bad_path = bind_provider_reply(
        packet,
        {
            "path": "docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md",
        },
    )
    assert bad_path.accepted is False
    assert bad_path.reason == "path_scope_escape"

    # Effect outside allowed set.
    bad_effect = bind_provider_reply(
        packet,
        {"effect": "merge"},
    )
    assert bad_effect.accepted is False
    assert bad_effect.reason == "effect_scope_escape"

    # Authority claim in reply rejected.
    bad_auth = bind_provider_reply(
        packet,
        {
            "summary": "claim completion",
            "completion_authority": True,
        },
    )
    assert bad_auth.accepted is False
    assert bad_auth.reason == "authority_claim_rejected"

    # Nested mismatched packet id rejected.
    bad_packet = bind_provider_reply(
        packet,
        {"packet_id": "packet:forged", "summary": "x"},
    )
    assert bad_packet.accepted is False
    assert bad_packet.reason == "reply_packet_mismatch"

    # Provider packet itself carries the binding anchors.
    assert provider["context_cid"] == packet.context_cid
    assert provider["effect_scope"]["write_paths"] == list(
        packet.effect_scope.write_paths
    )


def test_service_remember_resolution_then_hit() -> None:
    service = DeltaTaskPacketService()
    request = _request()
    first = service.evaluate(request)
    assert first.action is DeterministicAction.DISPATCH_PROVIDER

    packet = service.build(request)
    service.remember_resolution(
        request,
        resolved_item_ids=[item.item_id for item in packet.unresolved_delta],
    )
    second = service.evaluate(request)
    assert second.action is DeterministicAction.RESOLVE_HIT
    assert second.source is DecisionSource.DECISION_CACHE
    # Requirements digest mismatch invalidates cache.
    drifted = _request(
        validations=(
            {
                "command": (
                    "python -m pytest -q "
                    "test/api/test_agent_supervisor_delta_task_packet.py "
                    "--maxfail=1"
                ),
            },
        )
    )
    # Same semantic cache key components except validations change evidence + requirements.
    # Force same cache key by reusing evidence/context explicitly after building base key.
    # Validation change alters evidence_digest, so a new cache key is used — dispatch.
    drifted_decision = service.evaluate(drifted)
    assert drifted_decision.action is DeterministicAction.DISPATCH_PROVIDER


def test_provider_never_receives_omitted_authority_or_credential() -> None:
    packet = build_delta_task_packet(_request())
    provider = packet.provider_packet()
    for key in (
        "completion_authority",
        "write_authority",
        "semantic_authority",
    ):
        assert provider[key] is False
    assert provider["nomination_only"] is True
    # Credential-shaped keys must not appear as values (nested dicts are
    # walked; membership against a set only applies to hashable leaves).
    forbidden_keys = {
        "api_key",
        "password",
        "token",
        "credential",
        "credentials",
        "quack_token",
        "quack_credential",
    }
    forbidden_values = {"must_never_appear", "secret_material_value"}

    def _walk(node, *, path: str = "") -> None:
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                assert child_key not in forbidden_keys, path
                _walk(child_value, path=f"{path}.{child_key}")
            return
        if isinstance(node, (list, tuple)):
            for index, child_value in enumerate(node):
                _walk(child_value, path=f"{path}[{index}]")
            return
        if isinstance(node, (str, bytes, int, float, bool)) or node is None:
            assert node not in forbidden_values, path

    for key, value in provider.items():
        assert key not in forbidden_keys
        _walk(value, path=key)

    # Injecting authority into a raw reply path is rejected by binding.
    binding = bind_provider_reply(
        packet,
        {"write_authority": True, "summary": "nope"},
    )
    assert binding.accepted is False


def test_failure_signature_stable_for_unchanged_evidence() -> None:
    request = _request()
    first = compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest=request.evidence_digest,
        policy_id=request.policy_id,
    )
    second = compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest=request.evidence_digest,
        policy_id=request.policy_id,
    )
    assert first == second
    third = compute_failure_signature_id(
        task_cid=request.task_cid,
        evidence_digest="sha256:" + ("ff" * 32),
        policy_id=request.policy_id,
    )
    assert third != first


def test_effect_scope_permits_nested_write_paths() -> None:
    scope = EffectScope(
        allowed_effects=("edit_isolated_worktree", "run_validation"),
        write_paths=("ipfs_accelerate_py/agent_supervisor/prompt",),
        repository_id="repo:demo",
        tree_id="tree:abc",
    )
    assert scope.permits_path(
        "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py"
    )
    assert not scope.permits_path("test/api/test_agent_supervisor_delta_task_packet.py")
    assert scope.permits_effect("edit_isolated_worktree")
    assert not scope.permits_effect("merge")
