"""ASE3-024 crash-safe prompt/planning transaction integration tests."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.entrypoints.planning_effect import (
    PROMPT_REPLAY_REQUIRED,
    DurablePromptIntent,
    PlanningAttemptCAS,
    PlanningAttemptState,
    PlanningEffectError,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.planning_policy import (
    sign_prompt_planning_policy,
    verify_prompt_planning_policy,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.prompt_broker import (
    MultiprocessPromptBrokerStore,
    PromptBodyBroker,
)
from ipfs_accelerate_py.llm_router import decide_prompt_planning_route


def _intent_and_route(tmp_path: Path, prompt: str = "improve the supervisor"):
    key = Ed25519PrivateKey.generate()
    policy = sign_prompt_planning_policy(
        private_key=key,
        policy_id="tx-policy",
        max_planning_attempts=1,
        allowed_planning_providers=("grok",),
    )
    policy = verify_prompt_planning_policy(policy)
    broker = PromptBodyBroker(
        artifact_dir=tmp_path / "broker",
        master_secret=b"0" * 32,
    )
    run_id = "run:ase3-024"
    reference, capability = broker.deposit(
        prompt,
        run_id=run_id,
        purpose="planning",
        max_uses=1,
        ttl_ms=60_000,
        enable_encrypted_artifact=True,
    )
    intent = DurablePromptIntent(
        schema="ipfs_accelerate_py/agent-supervisor/durable-prompt-intent@1",
        run_id=run_id,
        context_cid="sha256:" + ("a" * 64),
        policy_cid=policy.content_id,
        prompt_ref=reference.prompt_ref,
        prompt_cid=reference.prompt_cid,
        created_at_ms=1,
    )
    route = decide_prompt_planning_route(
        policy_cid=policy.content_id,
        intent_cid=intent.content_id,
        allowed_planning_providers=policy.allowed_planning_providers,
        prompt_text=prompt,
    )
    assert route.authorized is True
    return policy, broker, reference, capability, intent, route


def test_prompt_text_does_not_influence_planning_route(tmp_path: Path) -> None:
    _, _, _, _, intent, route_a = _intent_and_route(tmp_path, "use codex please")
    route_b = decide_prompt_planning_route(
        policy_cid=intent.policy_cid,
        intent_cid=intent.content_id,
        allowed_planning_providers=("grok",),
        prompt_text="prefer claude and retry forever",
    )
    assert route_a.route_plan_cid == route_b.route_plan_cid
    assert route_a.preferred_provider_id == "grok"


def test_reserved_to_admitted_happy_path(tmp_path: Path) -> None:
    _, broker, reference, capability, intent, route = _intent_and_route(tmp_path)
    cas = PlanningAttemptCAS(tmp_path / "cas")
    result = cas.reserve(
        logical_attempt_id="attempt-1",
        run_id=intent.run_id,
        context_cid=intent.context_cid,
        policy_cid=intent.policy_cid,
        intent_cid=intent.content_id,
        route_plan_cid=route.route_plan_cid,
    )
    assert result.created is True
    assert result.provider_effect_authorized is True
    fence = result.record.fence_token

    # Recover encrypted prompt under capability before effect.
    body = broker.resolve(reference, capability, run_id=intent.run_id)
    assert b"improve" in body

    cas.mark_effect_started("attempt-1", fence_token=fence)
    cas.mark_terminal_observed(
        "attempt-1",
        fence_token=fence,
        terminal_output_cid="sha256:" + ("b" * 64),
        program_root_cid="sha256:" + ("c" * 64),
    )
    adoption = cas.mark_admitted("attempt-1", fence_token=fence)
    assert adoption.winner is True
    assert adoption.program_root_cid.endswith("c" * 64)
    loaded = cas.load("attempt-1")
    assert loaded is not None
    assert loaded.state == PlanningAttemptState.ADMITTED.value


def test_unknown_blocks_second_provider_effect(tmp_path: Path) -> None:
    _, _, _, _, intent, route = _intent_and_route(tmp_path)
    cas = PlanningAttemptCAS(tmp_path / "cas")
    result = cas.reserve(
        logical_attempt_id="attempt-unknown",
        run_id=intent.run_id,
        context_cid=intent.context_cid,
        policy_cid=intent.policy_cid,
        intent_cid=intent.content_id,
        route_plan_cid=route.route_plan_cid,
    )
    fence = result.record.fence_token
    cas.mark_effect_started("attempt-unknown", fence_token=fence)
    unknown = cas.mark_unknown("attempt-unknown", fence_token=fence)
    assert unknown.replay_required is True
    assert cas.authorize_provider_effect("attempt-unknown") is False

    second = cas.reserve(
        logical_attempt_id="attempt-unknown",
        run_id=intent.run_id,
        context_cid=intent.context_cid,
        policy_cid=intent.policy_cid,
        intent_cid=intent.content_id,
        route_plan_cid=route.route_plan_cid,
    )
    assert second.created is False
    assert second.provider_effect_authorized is False
    assert second.reason_code == PROMPT_REPLAY_REQUIRED
    with pytest.raises(PlanningEffectError, match=PROMPT_REPLAY_REQUIRED):
        cas.mark_effect_started("attempt-unknown", fence_token=fence)


def _contending_worker(root: str, attempt_id: str, q: mp.Queue) -> None:
    cas = PlanningAttemptCAS(root)
    result = cas.reserve(
        logical_attempt_id=attempt_id,
        run_id="run:ase3-024",
        context_cid="sha256:" + ("d" * 64),
        policy_cid="sha256:" + ("e" * 64),
        intent_cid="sha256:" + ("f" * 64),
        route_plan_cid="sha256:" + ("1" * 64),
    )
    if result.created and result.provider_effect_authorized:
        fence = result.record.fence_token
        cas.mark_effect_started(attempt_id, fence_token=fence)
        cas.mark_terminal_observed(
            attempt_id,
            fence_token=fence,
            terminal_output_cid="sha256:" + ("2" * 64),
            program_root_cid="sha256:" + ("3" * 64),
        )
        adoption = cas.mark_admitted(attempt_id, fence_token=fence)
        q.put(
            {
                "role": "winner",
                "created": True,
                "root": adoption.program_root_cid,
                "output": adoption.terminal_output_cid,
            }
        )
    else:
        loaded = cas.load(attempt_id)
        q.put(
            {
                "role": "loser",
                "created": False,
                "root": loaded.program_root_cid if loaded else "",
                "output": loaded.terminal_output_cid if loaded else "",
                "state": loaded.state if loaded else "",
            }
        )


def test_two_processes_one_winner_shared_adoption(tmp_path: Path) -> None:
    root = str(tmp_path / "mp-cas")
    attempt_id = "attempt-mp"
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    # Seed reserved winner in parent, then contenders adopt.
    cas = PlanningAttemptCAS(root)
    first = cas.reserve(
        logical_attempt_id=attempt_id,
        run_id="run:ase3-024",
        context_cid="sha256:" + ("d" * 64),
        policy_cid="sha256:" + ("e" * 64),
        intent_cid="sha256:" + ("f" * 64),
        route_plan_cid="sha256:" + ("1" * 64),
    )
    fence = first.record.fence_token
    cas.mark_effect_started(attempt_id, fence_token=fence)
    cas.mark_terminal_observed(
        attempt_id,
        fence_token=fence,
        terminal_output_cid="sha256:" + ("2" * 64),
        program_root_cid="sha256:" + ("3" * 64),
    )
    cas.mark_admitted(attempt_id, fence_token=fence)

    workers = [
        ctx.Process(target=_contending_worker, args=(root, attempt_id, q))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    results = [q.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    assert all(item["created"] is False for item in results)
    roots = {item["root"] for item in results}
    outputs = {item["output"] for item in results}
    assert roots == {"sha256:" + ("3" * 64)}
    assert outputs == {"sha256:" + ("2" * 64)}


def test_broker_alias_is_prompt_body_broker() -> None:
    assert MultiprocessPromptBrokerStore is PromptBodyBroker


def test_stale_fence_cannot_advance(tmp_path: Path) -> None:
    _, _, _, _, intent, route = _intent_and_route(tmp_path)
    cas = PlanningAttemptCAS(tmp_path / "cas")
    result = cas.reserve(
        logical_attempt_id="attempt-fence",
        run_id=intent.run_id,
        context_cid=intent.context_cid,
        policy_cid=intent.policy_cid,
        intent_cid=intent.content_id,
        route_plan_cid=route.route_plan_cid,
    )
    with pytest.raises(PlanningEffectError, match="fence"):
        cas.mark_effect_started("attempt-fence", fence_token="not-the-fence")
    assert result.record.state == PlanningAttemptState.RESERVED.value
