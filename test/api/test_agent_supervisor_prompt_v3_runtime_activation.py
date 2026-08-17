"""ASE3-026 protected runtime activation gate tests.

Covers pre-effect authorization (authorization_effect_observed:false), one
old+1 CAS/lease winner, fail-closed effect claims, and post-activation
observation binding. Authorization alone never proves the effect ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.monitor_runner import (
    DurableMonitorRunner,
)
from ipfs_accelerate_py.agent_supervisor.validation.protected_runtime_activation import (
    ACTIVATION_AUTHORIZATION_SCHEMA,
    POST_ACTIVATION_OBSERVATION_SCHEMA,
    ProcessJoinEvidence,
    ProtectedRuntimeActivationError,
    RefillActivationEvidence,
    RuntimeGenerationActivationStore,
    activation_authorization_id,
    build_activation_authorization,
    build_post_activation_observation,
    post_activation_observation_id,
    validate_activation_authorization,
    validate_post_activation_observation,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_adapters import (
    CurrentTreeResidualEvaluator,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller import (
    CompletionAuthorityDecision,
    ProductionRefillRuntime,
    RefillObservation,
    ResidualGap,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter import (
    ProductionRefillEventAdapter,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_store import (
    SIGNED_REFILL_POLICY_SCHEMA,
    RefillStore,
    SignedRefillPolicy,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_monitor import (
    ProcessEvidence,
    ReviewedHostNamespaceReconciler,
)
from ipfs_accelerate_py.agent_supervisor.validation import prompt_v3_convergence as conv


def _guardian() -> ReviewedHostNamespaceReconciler:
    return ReviewedHostNamespaceReconciler(
        guardian_identity="host-guardian",
        host_namespace="host.ns.v3",
        review_cid="bafyreview" + "a" * 50,
    )


def _auth(**overrides: object) -> dict:
    payload = build_activation_authorization(
        inactive_head="a" * 40,
        inactive_tree="b" * 40,
        branch="agent/prompt-self-improvement-v3",
        old_generation=0,
        lease_id="lease-gen-1",
        cas_token="cas-token-1",
        tree_id="tree-1",
        guardian_identity="host-guardian",
        host_namespace="host.ns.v3",
        review_cid="bafyreview" + "a" * 50,
        reviewer_identity="operator-reviewer",
        created_at="2026-08-10T12:00:00Z",
        expiry_at="2026-08-11T12:00:00Z",
    )
    payload.update(overrides)
    if "authorization_id" not in overrides:
        payload["authorization_id"] = activation_authorization_id(payload)
    return payload


def test_authorization_schema_and_pre_effect_invariant() -> None:
    payload = _auth()
    assert payload["schema"] == ACTIVATION_AUTHORIZATION_SCHEMA
    assert payload["authorization_effect_observed"] is False
    assert payload["receipt_phase"] == "pre_effect_authorization"
    assert payload["target_generation"] == 1
    errors = validate_activation_authorization(payload)
    assert errors == ()


def test_authorization_rejects_effect_observed_true() -> None:
    payload = _auth(authorization_effect_observed=True)
    payload["authorization_id"] = activation_authorization_id(payload)
    errors = validate_activation_authorization(payload)
    assert any("authorization_effect_observed" in e for e in errors)


def test_authorization_rejects_non_old_plus_one_generation() -> None:
    payload = _auth(target_generation=3)
    payload["authorization_id"] = activation_authorization_id(payload)
    errors = validate_activation_authorization(payload)
    assert any("old_generation + 1" in e for e in errors)


def test_authorization_rejects_already_active_flags() -> None:
    payload = _auth()
    flags = dict(payload["bounded_flags"])
    flags["already_active"] = True
    payload["bounded_flags"] = flags
    payload["authorization_id"] = activation_authorization_id(payload)
    errors = validate_activation_authorization(payload)
    assert any("already_active" in e for e in errors)


def test_authorization_rejects_codebase_refill_enabled() -> None:
    payload = _auth()
    flags = dict(payload["bounded_flags"])
    flags["codebase_refill_enabled"] = True
    payload["bounded_flags"] = flags
    payload["authorization_id"] = activation_authorization_id(payload)
    errors = validate_activation_authorization(payload)
    assert any("codebase_refill_enabled" in e for e in errors)


def test_authorization_rejects_identity_mismatch() -> None:
    payload = _auth()
    payload["authorization_id"] = "sha256:" + ("0" * 64)
    errors = validate_activation_authorization(payload)
    assert any("canonical identity mismatch" in e for e in errors)


def test_authorization_rejects_expired(now_ms: None = None) -> None:
    payload = _auth(
        created_at="2026-01-01T00:00:00Z",
        expiry_at="2026-01-01T01:00:00Z",
    )
    payload["authorization_id"] = activation_authorization_id(payload)
    # 2026-08-10 roughly
    future_ms = 1_786_348_800_000
    errors = validate_activation_authorization(payload, now_ms=future_ms)
    assert any("expired" in e for e in errors)


def test_cas_one_generation_winner_and_adoption(tmp_path: Path) -> None:
    store = RuntimeGenerationActivationStore(tmp_path / "gen")
    auth = _auth()
    state, adopted = store.consume_authorization(
        auth, guardian_identity="host-guardian", now_ms=1_000
    )
    assert adopted is False
    assert state.target_generation == 1
    assert state.refill_authorized is True
    assert state.monitor_authorized is True

    state2, adopted2 = store.consume_authorization(
        auth, guardian_identity="host-guardian", now_ms=2_000
    )
    assert adopted2 is True
    assert state2.lease_id == state.lease_id


def test_cas_rejects_non_guardian_consumer(tmp_path: Path) -> None:
    store = RuntimeGenerationActivationStore(tmp_path / "gen")
    auth = _auth()
    with pytest.raises(ProtectedRuntimeActivationError, match="guardian"):
        store.consume_authorization(auth, guardian_identity="cli-client")


def test_cas_rejects_conflicting_winner(tmp_path: Path) -> None:
    store = RuntimeGenerationActivationStore(tmp_path / "gen")
    auth1 = _auth()
    store.consume_authorization(auth1, guardian_identity="host-guardian")
    auth2 = _auth()
    cas = dict(auth2["cas_lease"])
    cas["lease_id"] = "lease-other"
    cas["cas_token"] = "cas-other"
    auth2["cas_lease"] = cas
    auth2["authorization_id"] = activation_authorization_id(auth2)
    with pytest.raises(ProtectedRuntimeActivationError, match="CAS lost"):
        store.consume_authorization(auth2, guardian_identity="host-guardian")


def test_post_activation_observation_binds_authorization_and_join(
    tmp_path: Path,
) -> None:
    auth = _auth()
    auth_raw = json.dumps(auth, sort_keys=True, separators=(",", ":")).encode("utf-8")

    guardian = _guardian()
    runner = DurableMonitorRunner(tmp_path / "mon", guardian=guardian)
    life = ProcessEvidence(
        role="lifecycle",
        process_cid="bafylife" + "b" * 51,
        process_birth_identity="life-birth-1",
        lease_id="life-lease-1",
        fencing_generation=1,
        heartbeat_at_ms=10_000,
        event_cursor="life-cursor:1",
        generation=1,
        healthy=True,
    )
    runner.start_or_adopt(
        run_id="run-1",
        requester="host-guardian",
        lifecycle=life,
        now_ms=10_000,
    )
    mon_state = runner.heartbeat("run-1", now_ms=12_000)
    assert mon_state is not None

    lifecycle = ProcessJoinEvidence(
        role="lifecycle",
        process_cid=life.process_cid,
        process_birth_identity=life.process_birth_identity,
        lease_id=life.lease_id,
        fencing_generation=life.fencing_generation,
        heartbeat_at_ms=12_000,
        event_cursor=life.event_cursor,
        generation=1,
        healthy=True,
    )
    monitor = ProcessJoinEvidence(
        role="monitor",
        process_cid=mon_state.process_cid,
        process_birth_identity=mon_state.process_birth_identity,
        lease_id=mon_state.lease_id,
        fencing_generation=mon_state.fencing_generation,
        heartbeat_at_ms=mon_state.heartbeat_at_ms,
        event_cursor=mon_state.event_cursor,
        generation=1,
        healthy=True,
    )

    # Activated refill saga to DISPATCHED/ADOPTED
    store = RefillStore(tmp_path / "refill")
    policy = SignedRefillPolicy(
        schema=SIGNED_REFILL_POLICY_SCHEMA,
        policy_cid="sha256:" + ("c" * 64),
        max_epochs=8,
        max_new_work_per_epoch=3,
        max_unchanged_epochs=2,
        activation_authorized=True,
        signer_identity_did="did:key:test",
    )
    gap = ResidualGap(
        "goal",
        "evidence",
        "scope-a",
        ("goal", "root"),
        0,
        {
            "priority": "P0",
            "track": "refill",
            "parallel_lane": "lane",
            "resource_class": "cpu",
        },
    )
    runtime = ProductionRefillRuntime(
        store=store,
        policy=policy,
        evaluator=CurrentTreeResidualEvaluator(
            required_tree_id="tree-1",
            residual_fn=lambda _obs: (gap,),
            completion_fn=lambda _obs: CompletionAuthorityDecision(False),
        ),
        event_adapter=ProductionRefillEventAdapter(),
    )
    receipt = runtime.run_once(
        RefillObservation("plan-root", 1, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-1",
    )
    assert receipt.dormant is False
    assert receipt.phase in {"DISPATCHED", "ADOPTED"} or receipt.disposition in {
        "dispatched",
        "adopted",
        "terminal",
    }

    refill = RefillActivationEvidence(
        plan_root_cid="plan-root",
        logical_attempt_id="attempt-1",
        phase=receipt.phase if receipt.phase in {"DISPATCHED", "ADOPTED"} else "DISPATCHED",
        epoch=max(int(getattr(receipt, "epoch", 0) or 0), 0),
        disposition=(
            receipt.disposition
            if receipt.disposition in {"dispatched", "adopted", "terminal"}
            else "dispatched"
        ),
        generation=1,
    )

    observation = build_post_activation_observation(
        authorization=auth,
        authorization_raw=auth_raw,
        lifecycle=lifecycle,
        monitor=monitor,
        refill=refill,
        reviewer_identity="operator-reviewer",
        created_at="2026-08-10T12:05:00Z",
    )
    assert observation["schema"] == POST_ACTIVATION_OBSERVATION_SCHEMA
    assert observation["observation_id"] == post_activation_observation_id(observation)
    errors = validate_post_activation_observation(
        observation,
        authorization=auth,
        authorization_sha256="sha256:"
        + __import__("hashlib").sha256(auth_raw).hexdigest(),
    )
    assert errors == (), errors


def test_observation_rejects_authorization_alone_claim() -> None:
    auth = _auth()
    auth_raw = b"{}"
    lifecycle = ProcessJoinEvidence(
        role="lifecycle",
        process_cid="bafylife" + "b" * 51,
        process_birth_identity="life-1",
        lease_id="ll",
        fencing_generation=1,
        heartbeat_at_ms=1,
        event_cursor="c1",
        generation=1,
        healthy=True,
    )
    monitor = ProcessJoinEvidence(
        role="monitor",
        process_cid="bafymon" + "c" * 52,
        process_birth_identity="mon-1",
        lease_id="ml",
        fencing_generation=1,
        heartbeat_at_ms=1,
        event_cursor="c2",
        generation=1,
        healthy=True,
    )
    refill = RefillActivationEvidence(
        plan_root_cid="p",
        logical_attempt_id="a",
        phase="DISPATCHED",
        epoch=0,
        disposition="dispatched",
        generation=1,
    )
    observation = build_post_activation_observation(
        authorization=auth,
        authorization_raw=auth_raw,
        lifecycle=lifecycle,
        monitor=monitor,
        refill=refill,
        reviewer_identity="operator-reviewer",
    )
    binding = dict(observation["authorization_binding"])
    binding["authorization_alone_proves_effect"] = True
    observation["authorization_binding"] = binding
    observation["observation_id"] = post_activation_observation_id(observation)
    errors = validate_post_activation_observation(observation, authorization=auth)
    assert any("authorization_alone_proves_effect" in e for e in errors)


def test_observation_requires_distinct_births() -> None:
    auth = _auth()
    life = ProcessJoinEvidence(
        role="lifecycle",
        process_cid="bafylife" + "b" * 51,
        process_birth_identity="same-birth",
        lease_id="ll",
        fencing_generation=1,
        heartbeat_at_ms=1,
        event_cursor="c1",
        generation=1,
        healthy=True,
    )
    mon = ProcessJoinEvidence(
        role="monitor",
        process_cid="bafymon" + "c" * 52,
        process_birth_identity="same-birth",
        lease_id="ml",
        fencing_generation=1,
        heartbeat_at_ms=1,
        event_cursor="c2",
        generation=1,
        healthy=True,
    )
    refill = RefillActivationEvidence(
        plan_root_cid="p",
        logical_attempt_id="a",
        phase="ADOPTED",
        epoch=0,
        disposition="adopted",
        generation=1,
    )
    observation = build_post_activation_observation(
        authorization=auth,
        authorization_raw=b"{}",
        lifecycle=life,
        monitor=mon,
        refill=refill,
        reviewer_identity="op",
    )
    errors = validate_post_activation_observation(observation, authorization=auth)
    assert any("distinct identities" in e for e in errors)


def test_convergence_loaders_fail_closed_on_garbage(tmp_path: Path) -> None:
    bad = tmp_path / conv.PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME
    bad.write_text("{}\n", encoding="utf-8")
    bad.chmod(0o600)
    with pytest.raises(ValueError, match="schema mismatch"):
        conv.load_protected_runtime_activation_authorization(bad)


def test_convergence_strict_validation_wrappers() -> None:
    payload = _auth()
    errors = conv.validate_protected_runtime_activation_authorization(payload)
    assert errors == ()


def test_unvalidated_receipt_present_fails_closed_in_plan_expansion(
    tmp_path: Path,
) -> None:
    """Garbage authorization receipt must not pass as reserved/unvalidated."""

    root = tmp_path / "convergence"
    root.mkdir()
    auth_path = root / conv.PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)
    # Also write empty observation so pair rule is not the only failure.
    obs_path = root / conv.PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_FILENAME
    obs_path.write_text("{}\n", encoding="utf-8")
    obs_path.chmod(0o600)

    tasks = conv._parse_taskboard_metadata(
        (Path(__file__).resolve().parents[2]
         / "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
        ).read_text(encoding="utf-8")
    )
    errors = conv._validate_program_plan_expansion(
        tasks=tasks,
        artifact_root=root,
        expected_statuses={},
    )
    assert any(
        "ASE3-026.authorization_receipt: strict load failed" in e
        or "ASE3-026.authorization_receipt:" in e
        for e in errors
    ), errors
    assert any(
        "ASE3-026.observation_receipt:" in e for e in errors
    ), errors


def test_dormant_refill_until_activation_authorization(tmp_path: Path) -> None:
    store = RefillStore(tmp_path / "refill")
    policy = SignedRefillPolicy(
        schema=SIGNED_REFILL_POLICY_SCHEMA,
        policy_cid="sha256:" + ("d" * 64),
        max_epochs=8,
        max_new_work_per_epoch=3,
        max_unchanged_epochs=2,
        activation_authorized=False,
        signer_identity_did="did:key:test",
    )
    runtime = ProductionRefillRuntime(
        store=store,
        policy=policy,
        evaluator=CurrentTreeResidualEvaluator(
            required_tree_id="tree-1",
            residual_fn=lambda _obs: (),
            completion_fn=lambda _obs: CompletionAuthorityDecision(False),
        ),
        event_adapter=ProductionRefillEventAdapter(),
    )
    receipt = runtime.run_once(
        RefillObservation("plan-root", 1, open_goals=1),
        tree_id="tree-1",
        logical_attempt_id="attempt-dormant",
    )
    assert receipt.dormant is True
    assert "ase3_026" in receipt.reason or "activation" in receipt.reason
