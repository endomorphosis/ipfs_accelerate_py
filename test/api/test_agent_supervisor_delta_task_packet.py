"""Tests for DeltaTaskPacket@1 / DeterministicFirstDecision@1.

DQP-028 evidence subset: packet identity, progressive disclosure, deterministic
hit, cache miss, unchanged reprompt, counterexample, scope/secret escape,
context overflow.

Acceptance:

* Provider never receives omitted authority or credential
* Packet/reply are bound to exact context and effect scope
* Unchanged failure cannot churn indefinitely
* New counterexample/tree/plan/policy/schema produces a distinct admitted packet
* Deterministic resolution preserves validation/proof requirements
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.context.database_context import (
    Completeness,
    ContextBudgetSpec,
    FrontierDisposition,
    TaskContextInput,
    build_database_context_manifest,
)
from ipfs_accelerate_py.agent_supervisor.prompt.delta_task_packet import (
    AUTHORITY_CLASS,
    DEFAULT_MAX_PACKET_BYTES,
    DEFAULT_POLICY_ID,
    DELTA_TASK_PACKET_INTERFACE,
    DELTA_TASK_PACKET_SCHEMA,
    DETERMINISTIC_FIRST_DECISION_INTERFACE,
    PRODUCER_ID,
    REDACTION_MARKER,
    DeltaTaskPacket,
    DeltaTaskPacketAuthorityError,
    DeltaTaskPacketOverflowError,
    DeltaTaskPacketScopeError,
    DeltaTaskPacketSecretError,
    DeterministicCacheEntry,
    DeterministicFirstDecision,
    DeterministicFirstDisposition,
    EffectScope,
    PacketAdmissionState,
    PacketBudgetSpec,
    build_delta_task_packet,
    compute_deterministic_cache_key,
    compute_evidence_digest,
    evaluate_deterministic_first,
    record_unchanged_failure,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_call_ledger import (
    duckdb_available,
    open_provider_call_ledger,
)


WRITE_PATHS = (
    "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
    "test/api/test_agent_supervisor_delta_task_packet.py",
)


def _request(**overrides) -> TaskContextInput:
    values = dict(
        task_cid="task:dqp-028-demo",
        repository_id="repo:demo",
        tree_id="tree:abc",
        policy_id=DEFAULT_POLICY_ID,
        task_revision="rev:1",
        plan_cid="plan:1",
        goal_cid="goal:context-economy",
        task_status="ready",
        task_summary="Generate bounded delta task packets",
        unmet_dependencies=(
            {"dependency_id": "dep:context", "summary": "context manifest ready"},
        ),
        latest_failure={
            "failure_id": "fail:distinct-1",
            "kind": "validation",
            "summary": "prior validation failed",
        },
        worktree_delta={
            "paths": list(WRITE_PATHS),
            "digests": {
                WRITE_PATHS[0]: "sha256:aaa",
                WRITE_PATHS[1]: "sha256:bbb",
            },
        },
        impacted_symbols=(
            {"symbol": "DeltaTaskPacket", "path": WRITE_PATHS[0]},
            {"symbol": "evaluate_deterministic_first", "path": WRITE_PATHS[0]},
        ),
        open_obligations=(
            {"obligation_id": "ob:packet-identity", "summary": "stable packet id"},
            {"obligation_id": "ob:replay-suppress", "summary": "suppress churn"},
        ),
        decisions=(
            {"decision_id": "dec:deterministic-first", "summary": "try cache first"},
        ),
        evidence=(
            {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
        ),
        validations=(
            {
                "command": (
                    "python -m pytest -q "
                    "test/api/test_agent_supervisor_delta_task_packet.py"
                ),
            },
        ),
        budget=ContextBudgetSpec(
            max_rows=64,
            max_bytes=64_000,
            max_tokens=8_192,
            page_size=32,
            page_offset=0,
        ),
    )
    values.update(overrides)
    return TaskContextInput(**values)


def _scope(**overrides) -> EffectScope:
    values = dict(
        write_paths=WRITE_PATHS,
        effect_ids=("effect:edit-packet",),
        read_paths=(),
    )
    values.update(overrides)
    return EffectScope(**values)


def test_interface_identities() -> None:
    assert DELTA_TASK_PACKET_INTERFACE == "DeltaTaskPacket@1"
    assert DETERMINISTIC_FIRST_DECISION_INTERFACE == "DeterministicFirstDecision@1"
    assert AUTHORITY_CLASS == "derived_evidence"
    assert REDACTION_MARKER == "secret_material"
    assert PRODUCER_ID == "delta-task-packet@1"
    assert DEFAULT_MAX_PACKET_BYTES == 34_000
    assert EffectScope.from_paths(WRITE_PATHS).write_paths == WRITE_PATHS


def test_cold_import_has_no_side_effects() -> None:
    # Importing the module must not open databases or touch the network.
    import ipfs_accelerate_py.agent_supervisor.prompt.delta_task_packet as mod

    assert mod.DELTA_TASK_PACKET_INTERFACE == "DeltaTaskPacket@1"
    assert callable(mod.build_delta_task_packet)
    assert callable(mod.evaluate_deterministic_first)


def test_packet_identity_is_content_addressed_and_stable() -> None:
    first = build_delta_task_packet(_request(), effect_scope=_scope())
    second = build_delta_task_packet(
        _request(
            heartbeat_at="2026-08-09T12:00:00Z",
            observed_at="2026-08-09T12:00:01Z",
            metadata={"lease_heartbeat": "noise", "pid": "999"},
        ),
        effect_scope=_scope(),
    )
    assert first.packet_id == second.packet_id
    assert first.content_id == first.packet_id
    assert first.interface == DELTA_TASK_PACKET_INTERFACE
    assert first.schema == DELTA_TASK_PACKET_SCHEMA
    assert first.to_dict()["authority"] == AUTHORITY_CLASS
    assert first.nomination_only is True
    assert first.semantic_authority is False
    assert first.write_authority is False
    assert first.completion_authority is False
    assert first.validation_commands
    assert first.effect_scope.write_paths == WRITE_PATHS


def test_progressive_disclosure_frontier_is_explicit() -> None:
    packet = build_delta_task_packet(
        _request(
            budget=ContextBudgetSpec(
                max_rows=64,
                max_bytes=64_000,
                max_tokens=8_192,
                page_size=2,
                page_offset=0,
            ),
            evidence=tuple(
                {"evidence_id": f"ev:{index}", "summary": f"evidence {index}"}
                for index in range(12)
            ),
        ),
        effect_scope=_scope(),
    )
    assert packet.frontier.is_explicit is True
    assert packet.frontier.has_more is True
    assert packet.frontier.omitted_member_ids
    assert packet.completeness in {
        Completeness.PARTIAL_WITH_FRONTIER,
        Completeness.OVERFLOW,
    }
    surface = packet.provider_surface()
    assert surface["frontier"]["omitted_count"] == len(
        packet.frontier.omitted_member_ids
    )
    assert surface["frontier"]["omitted_is_authority"] is False
    assert surface["frontier"]["disposition"] != FrontierDisposition.EMPTY.value
    # Omitted member payloads must not appear as authority on the provider surface.
    assert "omitted_member_ids" not in surface["frontier"] or surface[
        "frontier"
    ].get("omitted_is_authority") is False


def test_deterministic_hit_skips_provider_and_preserves_validation() -> None:
    request = _request()
    packet = build_delta_task_packet(request, effect_scope=_scope())
    cache_key = compute_deterministic_cache_key(
        task_cid=packet.task_cid,
        context_cid=packet.context_cid,
        tree_id=packet.tree_id,
        plan_cid=packet.plan_cid,
        policy_digest=packet.policy_digest,
        obligation_ids=packet.obligation_ids,
    )
    entry = DeterministicCacheEntry(
        cache_key=cache_key,
        resolution_digest="sha256:" + ("ab" * 32),
        validation_commands=packet.validation_commands,
        obligation_ids=packet.obligation_ids,
        proof_requirements=("proof:fixed-point", "proof:scope-bound"),
        summary="deterministic operator resolved known work",
    )
    decision = evaluate_deterministic_first(
        request,
        effect_scope=_scope(),
        proof_requirements=("proof:fixed-point", "proof:scope-bound"),
        deterministic_cache={cache_key: entry},
    )
    assert decision.interface == DETERMINISTIC_FIRST_DECISION_INTERFACE
    assert decision.is_deterministic is True
    assert decision.may_dispatch_provider is False
    assert decision.packet is None
    assert decision.admission_state is PacketAdmissionState.DETERMINISTIC
    assert decision.disposition in {
        DeterministicFirstDisposition.DETERMINISTIC_HIT,
        DeterministicFirstDisposition.CACHE_HIT,
    }
    # Validation and proof requirements are preserved, not dropped.
    assert decision.validation_commands == packet.validation_commands
    assert "proof:fixed-point" in decision.proof_requirements
    assert "proof:scope-bound" in decision.proof_requirements
    assert set(decision.obligation_ids) == set(packet.obligation_ids)


def test_cache_miss_admits_provider_packet() -> None:
    decision = evaluate_deterministic_first(
        _request(),
        effect_scope=_scope(),
        proof_requirements=("proof:fixed-point",),
        deterministic_cache={},
    )
    assert decision.disposition is DeterministicFirstDisposition.PROVIDER_ADMITTED
    assert decision.may_dispatch_provider is True
    assert decision.is_provider_admitted is True
    assert decision.admission_state is PacketAdmissionState.ADMITTED
    assert decision.packet is not None
    assert decision.packet.packet_id
    assert decision.reason == "cache_miss_provider_admitted"
    assert decision.validation_commands
    assert "proof:fixed-point" in decision.proof_requirements
    surface = decision.packet.provider_surface()
    assert surface["packet_id"] == decision.packet.packet_id
    assert surface["nomination_only"] is True
    assert surface["write_authority"] is False


def test_reply_binding_is_exact_to_context_and_effect_scope() -> None:
    packet = build_delta_task_packet(_request(), effect_scope=_scope())
    binding = packet.bind_reply(
        response_digest="sha256:" + ("cd" * 32),
        proposed_write_paths=[WRITE_PATHS[0]],
        outcome="proposed",
    )
    assert binding["packet_id"] == packet.packet_id
    assert binding["context_cid"] == packet.context_cid
    assert binding["tree_id"] == packet.tree_id
    assert binding["plan_cid"] == packet.plan_cid
    assert binding["policy_digest"] == packet.policy_digest
    assert binding["proposed_write_paths"] == [WRITE_PATHS[0]]
    assert binding["validation_commands"] == list(packet.validation_commands)
    assert binding["binding_id"]

    with pytest.raises(DeltaTaskPacketScopeError) as excinfo:
        packet.bind_reply(
            response_digest="sha256:" + ("ee" * 32),
            proposed_write_paths=["outside/forbidden.py"],
        )
    assert excinfo.value.reason_code == "scope_escape"


def test_secret_and_credential_never_reach_provider() -> None:
    # Secrets in member payloads fail closed during packet construction.
    with pytest.raises(DeltaTaskPacketSecretError):
        build_delta_task_packet(
            _request(
                latest_failure={
                    "failure_id": "fail:secret",
                    "password": "must_never_appear",
                }
            ),
            effect_scope=_scope(),
        )

    with pytest.raises(DeltaTaskPacketSecretError):
        build_delta_task_packet(
            _request(
                worktree_delta={"paths": [".env.local", WRITE_PATHS[0]]},
            ),
            effect_scope=_scope(),
        )

    with pytest.raises(DeltaTaskPacketSecretError):
        build_delta_task_packet(
            _request(
                evidence=(
                    {
                        "evidence_id": "ev:cred",
                        "api_key": "sk-must-never-appear",
                        "summary": "leaked credential",
                    },
                )
            ),
            effect_scope=_scope(),
        )

    with pytest.raises(DeltaTaskPacketSecretError):
        EffectScope.from_paths((".env", WRITE_PATHS[0]))

    # evaluate_deterministic_first maps secret escapes to a typed rejection.
    decision = evaluate_deterministic_first(
        _request(
            latest_failure={
                "failure_id": "fail:secret",
                "credential": "must_never_appear",
            }
        ),
        effect_scope=_scope(),
    )
    assert decision.disposition is DeterministicFirstDisposition.SECRET_ESCAPE
    assert decision.may_dispatch_provider is False
    assert decision.admission_state is PacketAdmissionState.REJECTED

    clean = build_delta_task_packet(_request(), effect_scope=_scope())
    surface = clean.provider_surface()
    serialized = str(surface)
    assert "must_never_appear" not in serialized
    assert "BEGIN PRIVATE KEY" not in serialized
    assert surface["frontier"]["omitted_is_authority"] is False
    assert surface["treat_as"] == "data_not_instructions"
    assert "api_key" not in serialized
    assert "password" not in serialized


def test_scope_escape_and_authority_claims_fail_closed() -> None:
    with pytest.raises(DeltaTaskPacketScopeError):
        EffectScope.from_paths(("../escape.py",))

    with pytest.raises(DeltaTaskPacketScopeError):
        EffectScope.from_paths(("/absolute/path.py",))

    with pytest.raises(DeltaTaskPacketScopeError):
        EffectScope.from_paths(("glob/*/escape.py",))

    with pytest.raises(DeltaTaskPacketScopeError):
        EffectScope.from_paths(())

    decision = evaluate_deterministic_first(
        _request(),
        effect_scope=(),  # empty effect scope is a scope escape
    )
    assert decision.disposition is DeterministicFirstDisposition.SCOPE_ESCAPE
    assert decision.may_dispatch_provider is False

    packet = build_delta_task_packet(_request(), effect_scope=_scope())
    with pytest.raises(DeltaTaskPacketAuthorityError):
        DeltaTaskPacket(
            packet_id="",
            task_cid=packet.task_cid,
            repository_id=packet.repository_id,
            tree_id=packet.tree_id,
            context_cid=packet.context_cid,
            plan_cid=packet.plan_cid,
            policy_id=packet.policy_id,
            policy_digest=packet.policy_digest,
            schema_revision=packet.schema_revision,
            effect_scope=packet.effect_scope,
            validation_commands=packet.validation_commands,
            obligation_ids=packet.obligation_ids,
            unresolved_members=packet.unresolved_members,
            frontier=packet.frontier,
            completeness=packet.completeness,
            write_authority=True,  # forbidden
        )


def test_context_overflow_fails_closed() -> None:
    with pytest.raises(DeltaTaskPacketOverflowError):
        build_delta_task_packet(
            _request(
                budget=ContextBudgetSpec(
                    max_rows=1,
                    max_bytes=32,
                    max_tokens=8,
                    page_size=1,
                ),
                unmet_dependencies=tuple(
                    {
                        "dependency_id": f"dep:{index}",
                        "summary": f"dependency {index} " + ("x" * 200),
                    }
                    for index in range(8)
                ),
                open_obligations=tuple(
                    {
                        "obligation_id": f"ob:{index}",
                        "summary": f"obligation {index} " + ("y" * 200),
                    }
                    for index in range(8)
                ),
                validations=tuple(
                    {"command": f"python -m pytest test_{index}.py -q"}
                    for index in range(8)
                ),
            ),
            effect_scope=_scope(),
        )

    decision = evaluate_deterministic_first(
        _request(
            budget=ContextBudgetSpec(
                max_rows=1,
                max_bytes=32,
                max_tokens=8,
                page_size=1,
            ),
            unmet_dependencies=tuple(
                {
                    "dependency_id": f"dep:{index}",
                    "summary": f"dependency {index} " + ("x" * 200),
                }
                for index in range(8)
            ),
            open_obligations=tuple(
                {
                    "obligation_id": f"ob:{index}",
                    "summary": f"obligation {index} " + ("y" * 200),
                }
                for index in range(8)
            ),
            validations=tuple(
                {"command": f"python -m pytest test_{index}.py -q"}
                for index in range(8)
            ),
        ),
        effect_scope=_scope(),
    )
    assert decision.disposition is DeterministicFirstDisposition.OVERFLOW
    assert decision.may_dispatch_provider is False


def test_new_counterexample_tree_plan_policy_schema_produce_distinct_packets() -> None:
    base = build_delta_task_packet(
        _request(),
        effect_scope=_scope(),
        counterexample_digest="cex:base",
    )

    by_cex = build_delta_task_packet(
        _request(),
        effect_scope=_scope(),
        counterexample_digest="cex:new",
    )
    by_tree = build_delta_task_packet(
        _request(tree_id="tree:changed"),
        effect_scope=_scope(),
        counterexample_digest="cex:base",
    )
    by_plan = build_delta_task_packet(
        _request(plan_cid="plan:changed"),
        effect_scope=_scope(),
        counterexample_digest="cex:base",
    )
    by_policy = build_delta_task_packet(
        _request(policy_id="policy:changed"),
        effect_scope=_scope(),
        counterexample_digest="cex:base",
    )
    by_schema = build_delta_task_packet(
        _request(schema_revision=2),
        effect_scope=_scope(),
        counterexample_digest="cex:base",
    )

    packet_ids = {
        base.packet_id,
        by_cex.packet_id,
        by_tree.packet_id,
        by_plan.packet_id,
        by_policy.packet_id,
        by_schema.packet_id,
    }
    assert len(packet_ids) == 6

    # Evidence digests used for replay admission also diverge.
    digests = {
        compute_evidence_digest(
            context_cid=item.context_cid,
            tree_id=item.tree_id,
            plan_cid=item.plan_cid,
            policy_digest=item.policy_digest,
            schema_revision=item.schema_revision,
            counterexample_digest=item.counterexample_digest,
            effect_scope=item.effect_scope,
            task_revision=item.task_revision,
        )
        for item in (base, by_cex, by_tree, by_plan, by_policy, by_schema)
    }
    assert len(digests) == 6


def test_delta_from_prior_manifest_is_bounded() -> None:
    prior = build_database_context_manifest(_request())
    packet = build_delta_task_packet(
        _request(
            task_revision="rev:2",
            evidence=(
                {"evidence_id": "ev:impact-1", "summary": "impact closure digest"},
                {"evidence_id": "ev:new-2", "summary": "new validation receipt"},
            ),
        ),
        effect_scope=_scope(),
        prior_manifest=prior,
    )
    assert packet.delta_id
    assert packet.from_manifest_cid == prior.manifest_cid
    # Delta packet still carries validation + scope, not a full repo dump.
    assert packet.validation_commands
    assert packet.effect_scope.write_paths
    kinds = {item["kind"] for item in packet.unresolved_members}
    # Only changed/added members should be present when a prior exists.
    assert "evidence" in kinds or packet.unresolved_members


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for ProviderCallLedger hermetic tests",
)
def test_unchanged_failure_is_replay_suppressed(tmp_path: Path) -> None:
    with open_provider_call_ledger(
        tmp_path / "provider_calls.duckdb",
        default_retry_budget=1,
    ) as ledger:
        request = _request()
        first = evaluate_deterministic_first(
            request,
            effect_scope=_scope(),
            counterexample_digest="cex:stable",
            ledger=ledger,
            attempt_id="attempt:1",
        )
        assert first.is_provider_admitted is True
        assert first.packet is not None

        # Exhaust retry policy against unchanged evidence.
        sig = record_unchanged_failure(
            ledger,
            packet=first.packet,
            attempt_id="attempt:1",
            retry_count=1,
            retry_budget=1,
        )
        assert sig.exhausted is True

        # Identical re-prompt is suppressed — no indefinite churn.
        second = evaluate_deterministic_first(
            request,
            effect_scope=_scope(),
            counterexample_digest="cex:stable",
            ledger=ledger,
            attempt_id="attempt:2",
        )
        assert second.disposition is DeterministicFirstDisposition.REPLAY_SUPPRESSED
        assert second.may_dispatch_provider is False
        assert second.is_suppressed is True
        assert second.admission_state is PacketAdmissionState.SUPPRESSED
        assert second.churn_decision is not None
        assert second.churn_decision.may_dispatch is False

        # Material counterexample change re-admits a distinct packet.
        third = evaluate_deterministic_first(
            request,
            effect_scope=_scope(),
            counterexample_digest="cex:new-material",
            ledger=ledger,
            attempt_id="attempt:3",
        )
        assert third.is_provider_admitted is True
        assert third.packet is not None
        assert third.packet.packet_id != first.packet.packet_id
        assert third.evidence_digest != second.evidence_digest


@pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for ProviderCallLedger hermetic tests",
)
def test_changed_tree_permits_new_call_after_suppression(tmp_path: Path) -> None:
    with open_provider_call_ledger(
        tmp_path / "provider_calls.duckdb",
        default_retry_budget=0,
    ) as ledger:
        base_req = _request(tree_id="tree:v1")
        admitted = evaluate_deterministic_first(
            base_req,
            effect_scope=_scope(),
            ledger=ledger,
            attempt_id="attempt:1",
        )
        assert admitted.packet is not None
        record_unchanged_failure(
            ledger,
            packet=admitted.packet,
            attempt_id="attempt:1",
            retry_count=0,
            retry_budget=0,
        )
        blocked = evaluate_deterministic_first(
            base_req,
            effect_scope=_scope(),
            ledger=ledger,
            attempt_id="attempt:2",
        )
        assert blocked.is_suppressed is True

        changed = evaluate_deterministic_first(
            _request(tree_id="tree:v2"),
            effect_scope=_scope(),
            ledger=ledger,
            attempt_id="attempt:3",
        )
        assert changed.is_provider_admitted is True
        assert changed.packet is not None
        assert changed.packet.tree_id == "tree:v2"
        assert changed.packet.packet_id != admitted.packet.packet_id


def test_decision_identity_is_stable() -> None:
    first = evaluate_deterministic_first(_request(), effect_scope=_scope())
    second = evaluate_deterministic_first(_request(), effect_scope=_scope())
    assert first.decision_id == second.decision_id
    assert first.to_dict()["interface"] == DETERMINISTIC_FIRST_DECISION_INTERFACE
    assert first.packet is not None
    assert first.packet.total_bytes <= first.packet.budget.max_bytes
    assert first.packet.total_tokens <= first.packet.budget.max_tokens


def test_packet_budget_spec_enforced() -> None:
    budget = PacketBudgetSpec(
        max_bytes=16_384,
        max_tokens=4_096,
        max_write_paths=2,
        max_obligations=8,
        max_validation_commands=8,
        max_delta_members=32,
    )
    packet = build_delta_task_packet(
        _request(
            budget=ContextBudgetSpec(
                max_rows=16,
                max_bytes=4_000,
                max_tokens=1_024,
                page_size=8,
            ),
            evidence=(),
            decisions=(),
            impacted_symbols=(),
        ),
        effect_scope=_scope(),
        budget=budget,
    )
    assert packet.total_bytes <= budget.max_bytes
    assert packet.total_tokens <= budget.max_tokens
    assert len(packet.effect_scope.write_paths) <= budget.max_write_paths
    assert packet.budget.max_write_paths == 2
