from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
    EAAEF_IDEMPOTENT_RESERVATION_SCHEMA,
    eaaef_reservation_id,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    external_agent_container_dispatcher as dispatch,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_FAILED,
    ATTEMPT_PHASE_PROVIDER,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bind_database_portal_execution_from_args,
)


def _cid(character: str) -> str:
    return "sha256:" + character * 64


def _packet() -> dispatch.ExternalAgentContainerWorkPacket:
    return dispatch.ExternalAgentContainerWorkPacket(
        task_id="EAAEF-010",
        task_cid=_cid("1"),
        attempt_id="attempt:eaaef:1",
        attempt_number=1,
        plan_revision_cid=_cid("2"),
        repository_tree="3" * 40,
        semantic_state_root=_cid("4"),
        worktree_id=_cid("5"),
        planned_container_id=_cid("6"),
        worker_principal_did="did:key:zworker",
        provider_principal_did="did:key:zprovider",
        provider="grok",
        model_route_cid=_cid("7"),
        container_profile_cid=_cid("8"),
        image_digest=_cid("9"),
        network_authorization_cid=_cid("a"),
        lease_id="lease:eaaef:1",
        fencing_token=7,
        fence_epoch=3,
        idempotency_key="eaaef:dispatch:1",
        effect_scope_cid=_cid("b"),
        gateway_binding_cid=_cid("c"),
    )


def _attempt() -> SimpleNamespace:
    packet = _packet()
    return SimpleNamespace(
        task_cid=packet.task_cid,
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        attempt_number=packet.attempt_number,
        lease_id=packet.lease_id,
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        owner_session_id="supervisor:1",
    )


def _seal(body: dict[str, object], field: str = "receipt_cid") -> dict[str, object]:
    return {**body, field: dispatch._content_id(body)}  # noqa: SLF001


def _qualification(packet: dispatch.ExternalAgentContainerWorkPacket) -> dict[str, object]:
    return {
        "status": "admitted",
        "dispatcher_interface": dispatch.EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
        "gateway_binding_cid": packet.gateway_binding_cid,
        "container_profile_cid": packet.container_profile_cid,
        "image_digest": packet.image_digest,
        "reservation_adapter_status": "qualified",
        "container_launcher_status": "qualified",
        "independent_verifier_status": "qualified",
        "host_source_isolation_status": "qualified",
        "qualification_receipt_cid": _cid("d"),
    }


def _proposal(
    packet: dispatch.ExternalAgentContainerWorkPacket,
) -> dict[str, object]:
    claim = dispatch.ExternalAgentContainerWorkerDispatcher._dispatch_claim(  # noqa: SLF001
        packet
    )
    return _seal(
        {
            "schema": dispatch.EXTERNAL_AGENT_CONTAINER_PROPOSAL_RECEIPT_SCHEMA,
            "interface": dispatch.EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "status": "proposal_ready",
            "claim_cid": claim["claim_cid"],
            "packet_cid": packet.packet_cid,
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "worker_principal_did": packet.worker_principal_did,
            "provider_principal_did": packet.provider_principal_did,
            "image_digest": packet.image_digest,
            "container_profile_cid": packet.container_profile_cid,
            "network_authorization_cid": packet.network_authorization_cid,
            "runtime_container_id": _cid("e"),
            "patch_artifact_cid": _cid("f"),
            "artifact_cids": [_cid("0")],
            "test_receipt_cids": [_cid("1")],
            "proof_receipt_cids": [_cid("2")],
            "host_source_mutated": False,
            "host_merge_attempted": False,
            "push_attempted": False,
        }
    )


def _verification(
    packet: dispatch.ExternalAgentContainerWorkPacket,
    proposal: dict[str, object],
    *,
    verifier: str = "did:key:zindependent",
) -> dict[str, object]:
    claim = dispatch.ExternalAgentContainerWorkerDispatcher._dispatch_claim(  # noqa: SLF001
        packet
    )
    return _seal(
        {
            "schema": dispatch.EXTERNAL_AGENT_CONTAINER_VERIFICATION_RECEIPT_SCHEMA,
            "interface": dispatch.EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "outcome": "passed",
            "claim_cid": claim["claim_cid"],
            "proposal_receipt_cid": proposal["receipt_cid"],
            "verifier_principal_did": verifier,
            "test_receipt_cids": list(proposal["test_receipt_cids"]),
            "proof_receipt_cids": list(proposal["proof_receipt_cids"]),
        }
    )


class _Owner:
    def __init__(self, *, deny: bool = False) -> None:
        self.deny = deny
        self.reservations: dict[str, dict[str, object]] = {}
        self.results: dict[str, dict[str, object]] = {}
        self.reserve_calls = 0
        self.commit_calls = 0

    def _reservation(
        self,
        claim_cid: str,
        *,
        outcome: str,
        accepted_result: dict[str, object] | None = None,
        reasons: list[str] | None = None,
    ) -> dict[str, object]:
        return _seal(
            {
                "schema": dispatch.EXTERNAL_AGENT_CONTAINER_DISPATCH_RESERVATION_SCHEMA,
                "claim_cid": claim_cid,
                "reservation_id": dispatch._content_id(  # noqa: SLF001
                    {"reservation": claim_cid}
                ),
                "outcome": outcome,
                "reason_codes": list(reasons or []),
                "accepted_result": accepted_result,
            }
        )

    def reserve_effect(self, **kwargs: object) -> dict[str, object]:
        self.reserve_calls += 1
        claim = dict(kwargs["claim"])
        claim_cid = str(claim["claim_cid"])
        if self.deny:
            return self._reservation(
                claim_cid,
                outcome="unavailable_fail_closed",
                reasons=["effect_reservation_before_external_effect_unqualified"],
            )
        prior = self.results.get(claim_cid)
        if prior is not None:
            return self._reservation(
                claim_cid,
                outcome="accepted_replay",
                accepted_result=prior,
            )
        if claim_cid in self.reservations:
            return self._reservation(claim_cid, outcome="in_flight_ambiguous")
        receipt = self._reservation(claim_cid, outcome="reserved_new")
        self.reservations[claim_cid] = receipt
        return receipt

    def commit_effect(self, **kwargs: object) -> dict[str, object]:
        self.commit_calls += 1
        claim = dict(kwargs["claim"])
        claim_cid = str(claim["claim_cid"])
        result = dict(kwargs["result"])
        prior = self.results.setdefault(claim_cid, result)
        if prior != result:
            raise RuntimeError("divergent accepted result")
        return dict(prior)


class _OuterReservationOwner:
    """Typed outer provider/effect reservation fixture for response loss."""

    def __init__(self, attempt: DatabaseTaskAttempt, *, kind: str) -> None:
        self.attempt = attempt
        self.kind = kind
        self.key = f"{kind}:{attempt.attempt_id}"
        self.record_id = eaaef_reservation_id(
            kind=kind,
            attempt_id=attempt.attempt_id,
            idempotency_key=self.key,
        )
        self.state = "existing_reserved_ambiguous"
        self.result: dict[str, object] = {}
        self.commit_calls = 0

    def get_idempotent_result(
        self,
        *,
        kind: str,
        attempt_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        assert (kind, attempt_id, idempotency_key) == (
            self.kind,
            self.attempt.attempt_id,
            self.key,
        )
        return {
            "schema": EAAEF_IDEMPOTENT_RESERVATION_SCHEMA,
            "kind": kind,
            "state": self.state,
            "record_id": self.record_id,
            "attempt_id": attempt_id,
            "task_cid": self.attempt.task_cid,
            "idempotency_key": idempotency_key,
            "result": dict(self.result),
        }

    def record_idempotent_result(self, **record: object) -> dict[str, object]:
        assert record["kind"] == self.kind
        assert record["record_id"] == self.record_id
        assert record["attempt_id"] == self.attempt.attempt_id
        assert record["task_cid"] == self.attempt.task_cid
        assert record["idempotency_key"] == self.key
        self.commit_calls += 1
        self.state = "committed"
        self.result = dict(record["result"])
        return dict(self.result)


def _response_loss_daemon(
    attempt: DatabaseTaskAttempt,
    owner: _OuterReservationOwner,
) -> DatabaseImplementationDaemon:
    """Create a no-I/O daemon shell exercising the real phase methods."""

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.authority_mode = "quack"
    daemon.require_real_execution = True
    daemon.owner_session_id = attempt.owner_session_id
    daemon._execution_repository = owner  # noqa: SLF001
    daemon._provider_fn = None  # noqa: SLF001
    daemon._effect_fn = None  # noqa: SLF001
    daemon._clock_ms = lambda: 1_900_000_000_000  # noqa: SLF001
    daemon.open = lambda: daemon
    daemon.get_attempt = lambda _attempt_id: attempt
    daemon._protect_attempt_write = lambda *_args, **_kwargs: None  # noqa: SLF001
    daemon._run_with_attempt_heartbeat = (  # noqa: SLF001
        lambda _attempt, callback: callback()
    )
    daemon.commit_phase = lambda current, phase, body=None: replace(
        current,
        committed_phase=phase,
        revision=current.revision + 1,
        body=dict(body or {}),
    )
    daemon._record_event = lambda *_args, **_kwargs: None  # noqa: SLF001
    return daemon

def _dispatcher(
    owner: _Owner,
    *,
    launcher,
    verifier=None,
    merge_observer=None,
    host_state: list[str] | None = None,
    qualification_guard=_qualification,
) -> dispatch.ExternalAgentContainerWorkerDispatcher:
    state = host_state if host_state is not None else ["host-tree:clean"]
    return dispatch.ExternalAgentContainerWorkerDispatcher(
        execution_repository=owner,
        packet_provider=lambda _attempt: _packet(),
        qualification_guard=qualification_guard,
        container_launcher=launcher,
        independent_verifier=(
            verifier
            if verifier is not None
            else lambda packet, proposal: _verification(packet, dict(proposal))
        ),
        merge_admission_observer=(
            merge_observer if merge_observer is not None else lambda *_args: None
        ),
        host_source_observer=lambda: state[0],
        now_ms=lambda: 1_900_000_000_000,
    )


def test_owner_denial_is_pre_effect_and_retry_does_not_call_container() -> None:
    owner = _Owner(deny=True)
    launches: list[str] = []
    dispatcher = _dispatcher(
        owner,
        launcher=lambda *_args: launches.append("called") or {},
    )

    for _attempt_index in range(2):
        with pytest.raises(
            dispatch.ExternalAgentContainerDispatchUnavailable,
            match="reservation was not admitted",
        ):
            dispatcher.run_provider(_attempt())

    assert owner.reserve_calls == 2
    assert owner.commit_calls == 0
    assert launches == []


def test_unqualified_profile_fails_before_owner_or_container() -> None:
    owner = _Owner()
    launches: list[str] = []
    dispatcher = _dispatcher(
        owner,
        launcher=lambda *_args: launches.append("called") or {},
        qualification_guard=lambda packet: {
            **_qualification(packet),
            "container_launcher_status": "unavailable_fail_closed",
        },
    )

    with pytest.raises(
        dispatch.ExternalAgentContainerDispatchUnavailable,
        match="qualification is absent",
    ):
        dispatcher.run_provider(_attempt())

    assert owner.reserve_calls == 0
    assert launches == []


def test_crash_after_reservation_is_ambiguous_and_never_relaunched() -> None:
    owner = _Owner()
    launches: list[str] = []

    def crash(*_args):
        launches.append("called")
        raise RuntimeError("container transport disappeared")

    dispatcher = _dispatcher(owner, launcher=crash)
    with pytest.raises(dispatch.ExternalAgentContainerDispatchAmbiguous):
        dispatcher.run_provider(_attempt())
    with pytest.raises(
        dispatch.ExternalAgentContainerDispatchAmbiguous,
        match="automatic replay is forbidden",
    ):
        dispatcher.run_provider(_attempt())

    assert owner.reserve_calls == 2
    assert owner.commit_calls == 0
    assert launches == ["called"]


def test_duplicate_attempt_replays_one_accepted_result_without_second_launch() -> None:
    owner = _Owner()
    launches: list[str] = []

    def launch(packet, _reservation):
        launches.append(packet.packet_cid)
        return _proposal(packet)

    dispatcher = _dispatcher(owner, launcher=launch)
    first = dispatcher.run_provider(_attempt())
    second = dispatcher.run_provider(_attempt())

    assert dict(first) == dict(second)
    assert first["accepted"] is True
    assert first["task_result_accepted"] is False
    assert first["merge_admitted"] is False
    assert owner.reserve_calls == 2
    assert owner.commit_calls == 1
    assert launches == [_packet().packet_cid]


def test_daemon_adopts_exact_inner_dispatch_replay_without_repeating_container() -> None:
    packet = _packet()
    attempt = DatabaseTaskAttempt(
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        task_cid=packet.task_cid,
        task_alias=packet.task_id,
        attempt_number=packet.attempt_number,
        owner_session_id="supervisor:1",
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        lease_id=packet.lease_id,
        committed_phase=ATTEMPT_PHASE_CONTEXT,
        status="running",
        started_at_ms=1_900_000_000_000,
        revision=2,
    )

    inner_owner = _Owner()
    launches: list[str] = []

    def launch(work_packet, _reservation):
        launches.append(work_packet.packet_cid)
        return _proposal(work_packet)

    dispatcher = _dispatcher(inner_owner, launcher=launch)
    accepted = dispatcher.run_provider(_attempt())
    assert launches == [packet.packet_cid]

    # Simulate response loss after the rich inner dispatch committed but
    # before the daemon could commit its outer provider projection.
    outer_provider = _OuterReservationOwner(attempt, kind="provider")
    daemon = _response_loss_daemon(attempt, outer_provider)
    updated, adopted, duplicated = daemon.run_provider(
        attempt,
        provider_fn=dispatcher.run_provider,
    )
    assert dict(adopted) == dict(accepted)
    assert duplicated is False
    assert updated.committed_phase == ATTEMPT_PHASE_PROVIDER
    assert outer_provider.state == "committed"
    assert outer_provider.commit_calls == 1
    assert launches == [packet.packet_cid]

    # If the rich inner reservation has no accepted receipt, exact re-entry
    # stops at ambiguity and cannot launch again or commit the outer record.
    ambiguous_owner = _Owner()
    ambiguous_launches: list[str] = []

    def crash_after_inner_reservation(*_args):
        ambiguous_launches.append("called")
        raise RuntimeError("lost container response")

    ambiguous_dispatcher = _dispatcher(
        ambiguous_owner,
        launcher=crash_after_inner_reservation,
    )
    with pytest.raises(dispatch.ExternalAgentContainerDispatchAmbiguous):
        ambiguous_dispatcher.run_provider(_attempt())
    ambiguous_outer = _OuterReservationOwner(attempt, kind="provider")
    ambiguous_daemon = _response_loss_daemon(attempt, ambiguous_outer)
    with pytest.raises(
        dispatch.ExternalAgentContainerDispatchAmbiguous,
        match="automatic replay is forbidden",
    ):
        ambiguous_daemon.run_provider(
            attempt,
            provider_fn=ambiguous_dispatcher.run_provider,
        )
    assert ambiguous_launches == ["called"]
    assert ambiguous_outer.commit_calls == 0

    # The exception is narrow: arbitrary callbacks are never invoked for an
    # ambiguous durable outer reservation.
    generic_calls: list[str] = []
    generic_outer = _OuterReservationOwner(attempt, kind="provider")
    generic_daemon = _response_loss_daemon(attempt, generic_outer)
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="refusing to repeat the external effect",
    ):
        generic_daemon.run_provider(
            attempt,
            provider_fn=lambda _attempt: generic_calls.append("called") or {},
        )
    assert generic_calls == []
    assert generic_outer.commit_calls == 0

    # The exact dispatcher effect method is pure and content-bound, so an
    # outer effect response loss safely recomputes the same receipt once.
    provider_attempt = replace(
        attempt,
        committed_phase=ATTEMPT_PHASE_PROVIDER,
        revision=3,
    )
    outer_effect = _OuterReservationOwner(provider_attempt, kind="effect")
    effect_daemon = _response_loss_daemon(provider_attempt, outer_effect)
    effect_attempt, effect, effect_duplicated = effect_daemon.run_effect(
        provider_attempt,
        accepted,
        effect_fn=dispatcher.apply_effect,
    )
    assert effect["status"] == "applied"
    assert effect_attempt.committed_phase == ATTEMPT_PHASE_EFFECT
    assert effect_duplicated is False
    assert outer_effect.state == "committed"
    assert outer_effect.commit_calls == 1
    assert launches == [packet.packet_cid]


def test_daemon_persists_one_canonical_dispatcher_validation_phase_and_replays_it() -> None:
    packet = _packet()
    owner = _Owner()
    admission: list[dict[str, object] | None] = [None]
    dispatcher = _dispatcher(
        owner,
        launcher=lambda work_packet, _reservation: _proposal(work_packet),
        merge_observer=lambda _packet, _effect: admission[0],
    )
    accepted = dispatcher.run_provider(_attempt())
    effect = dispatcher.apply_effect(_attempt(), accepted)
    claim = dispatch.ExternalAgentContainerWorkerDispatcher._dispatch_claim(  # noqa: SLF001
        packet
    )
    admission[0] = _seal(
        {
            "schema": dispatch.EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA,
            "interface": dispatch.EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "decision": "accepted",
            "delivery_mode": "reviewed_patch",
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "claim_cid": claim["claim_cid"],
            "accepted_result_receipt_id": effect["accepted_result_receipt_id"],
            "patch_artifact_cid": effect["patch_artifact_cid"],
            "reviewer_principal_did": "did:key:zhostreviewer",
            "effect_authority_cid": _cid("3"),
            "merge_commit": "",
        }
    )
    attempt = DatabaseTaskAttempt(
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        task_cid=packet.task_cid,
        task_alias=packet.task_id,
        attempt_number=packet.attempt_number,
        owner_session_id="supervisor:1",
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        lease_id=packet.lease_id,
        committed_phase=ATTEMPT_PHASE_EFFECT,
        status="running",
        started_at_ms=1_900_000_000_000,
        revision=4,
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.authority_mode = "quack"
    daemon.require_real_execution = True
    daemon._validation_fn = None  # noqa: SLF001
    daemon._protect_attempt_write = lambda *_args, **_kwargs: None  # noqa: SLF001
    daemon._run_with_attempt_heartbeat = (  # noqa: SLF001
        lambda _attempt, callback: callback()
    )
    daemon.provider_invocation_recorded = lambda *_args, **_kwargs: accepted
    daemon.effect_claim_recorded = lambda *_args, **_kwargs: effect
    canonical_bodies: list[dict[str, object]] = []
    completion_inputs: list[dict[str, object]] = []

    def commit_phase(current, phase, *, body=None):
        assert phase == "validation"
        canonical = {
            **dict(body or {}),
            "run_id": "validation-run:canonical",
            "result_id": "validation-result:canonical",
        }
        canonical_bodies.append(canonical)
        return replace(
            current,
            committed_phase="validation",
            revision=current.revision + 1,
            body=canonical,
        )

    def complete(current, *, validation_result):
        completion_inputs.append(dict(validation_result))
        return replace(
            current,
            committed_phase="complete",
            status="succeeded",
            revision=current.revision + 1,
        )

    daemon.commit_phase = commit_phase
    daemon.complete_attempt = complete
    daemon.phase_history = lambda _attempt_id: []
    result = daemon.resume_attempt(
        attempt,
        validation_fn=dispatcher.validate_effect,
    )
    assert result["status"] == "succeeded"
    assert completion_inputs == canonical_bodies
    assert owner.commit_calls == 1

    # A restart after validation but before completion consumes the exact
    # stored authority body; replay metadata cannot enter the receipt.
    stored_validation = replace(
        attempt,
        committed_phase="validation",
        revision=5,
        body=canonical_bodies[0],
    )
    daemon.phase_history = lambda _attempt_id: [
        {"phase": "validation", "body": canonical_bodies[0]}
    ]
    completion_inputs.clear()
    replayed = daemon.resume_attempt(stored_validation)
    assert replayed["status"] == "succeeded"
    assert completion_inputs == canonical_bodies
    assert "replayed" not in completion_inputs[0]


def test_daemon_emits_terminal_task_event_before_releasing_task_authority() -> None:
    packet = _packet()
    attempt = DatabaseTaskAttempt(
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        task_cid=packet.task_cid,
        task_alias=packet.task_id,
        attempt_number=packet.attempt_number,
        owner_session_id="supervisor:1",
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        lease_id=packet.lease_id,
        committed_phase="validation",
        status="running",
        started_at_ms=1_900_000_000_000,
        revision=5,
    )
    task = SimpleNamespace(
        task_cid=packet.task_cid,
        status="in_progress",
        revision=2,
    )
    task.to_dict = lambda: {
        "task_cid": task.task_cid,
        "status": task.status,
        "revision": task.revision,
    }
    claim = SimpleNamespace(claim_id=attempt.claim_id)
    ordering: list[str] = []

    class Coordinator:
        def prepare_task_completion(self, *_args, **_kwargs):
            ordering.append("prepare")
            return {"preparation_digest": _cid("4")}

        def complete_task_claim(self, *_args, **_kwargs):
            ordering.append("complete_claim")
            return {"status": "succeeded"}

        def settle_task_claim(self, *_args, **_kwargs):
            ordering.append("settle_claim")
            return {"state": "released"}

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.authority_mode = "quack"
    daemon.require_real_execution = True
    daemon.owner_session_id = attempt.owner_session_id
    daemon._clock_ms = lambda: 1_900_000_000_100  # noqa: SLF001
    daemon._task_source = SimpleNamespace(get=lambda _task_cid: task)  # noqa: SLF001
    daemon._coordinator = Coordinator()  # noqa: SLF001
    daemon.open = lambda: daemon
    daemon.get_attempt = lambda _attempt_id: attempt
    daemon._attempt_claim = lambda _attempt: claim  # noqa: SLF001
    daemon._protect_attempt_claim = lambda *_args, **_kwargs: None  # noqa: SLF001
    daemon._cas_task_status_database = (  # noqa: SLF001
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"schema": "test-control-cas"}
        )
    )
    daemon.commit_phase = lambda current, phase, body=None: (
        ordering.append("complete_phase")
        or replace(
            current,
            committed_phase=phase,
            status="succeeded",
            revision=current.revision + 1,
            body=dict(body or {}),
        )
    )
    daemon._record_event = (  # noqa: SLF001
        lambda *_args, **_kwargs: ordering.append("terminal_event")
    )
    completed = daemon.complete_attempt(
        attempt,
        validation_result={
            "outcome": "passed",
            "evidence_digest": _cid("5"),
            "argv": ["container-validation"],
            "body": {},
        },
    )
    assert completed.status == "succeeded"
    assert ordering == [
        "prepare",
        "complete_claim",
        "complete_phase",
        "terminal_event",
        "settle_claim",
    ]


def test_spoofed_eaaef_proxy_cannot_suppress_post_commit_event() -> None:
    packet = _packet()
    attempt = DatabaseTaskAttempt(
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        task_cid=packet.task_cid,
        task_alias=packet.task_id,
        attempt_number=packet.attempt_number,
        owner_session_id="supervisor:1",
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        lease_id=packet.lease_id,
        committed_phase=ATTEMPT_PHASE_CONTEXT,
        status="running",
        started_at_ms=1_900_000_000_000,
        revision=2,
    )
    body = {
        "reason": "retryable portal transport failure",
        "portal_retryable_failure": True,
    }

    class Repository:
        EAAEF_INTERFACE = "EAAEFBootstrapExecutionRepositoryProxy@2"
        commit_calls = 0

        @classmethod
        def commit_phase(cls, **values: object) -> dict[str, object]:
            cls.commit_calls += 1
            assert values["committed_phase"] == ATTEMPT_PHASE_FAILED
            return {
                **attempt.to_dict(),
                "committed_phase": ATTEMPT_PHASE_FAILED,
                "status": "failed",
                "finished_at_ms": values["finished_at_ms"],
                "revision": values["revision"],
                "body": dict(values["body"]),
            }

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.authority_mode = "quack"
    daemon.require_real_execution = True
    daemon._execution_repository = Repository()  # noqa: SLF001
    daemon._clock_ms = lambda: 1_900_000_000_100  # noqa: SLF001
    daemon.open = lambda: daemon
    daemon._protect_attempt_write = lambda *_args, **_kwargs: None  # noqa: SLF001
    events: list[str] = []
    daemon._record_event = (  # noqa: SLF001
        lambda *_args, **_kwargs: events.append("post-revocation-event")
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the exact EAAEF proxy",
    ):
        daemon.commit_phase(attempt, ATTEMPT_PHASE_FAILED, body=body)
    assert Repository.commit_calls == 0
    assert events == []


def test_dead_lane_recovery_normalizes_raw_proxy_lease_mappings() -> None:
    packet = _packet()
    attempt = DatabaseTaskAttempt(
        attempt_id=packet.attempt_id,
        claim_id="claim:control:1",
        task_cid=packet.task_cid,
        task_alias=packet.task_id,
        attempt_number=packet.attempt_number,
        owner_session_id="supervisor:dead-lane",
        fencing_token=packet.fencing_token,
        fence_epoch=packet.fence_epoch,
        lease_id=packet.lease_id,
        committed_phase=ATTEMPT_PHASE_CONTEXT,
        status="running",
        started_at_ms=1_899_999_000_000,
        revision=2,
    )
    claim = {
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "owner_session_id": attempt.owner_session_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "lease_id": attempt.lease_id,
        "state": "accepted",
        "expires_at_ms": 1_899_999_999_999,
    }
    snapshot = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-running-recovery-snapshot@1"
        ),
        "claim": claim,
        "preparation": None,
    }

    def exercise(raw_lease: dict[str, object]):
        class Coordinator:
            @staticmethod
            def expire_task_claim(*_args, **_kwargs):
                return dict(raw_lease)

        daemon = object.__new__(DatabaseImplementationDaemon)
        daemon.authority_mode = "quack"
        daemon._quack_command_gateway = object()  # noqa: SLF001
        daemon._clock_ms = lambda: 1_900_000_000_000  # noqa: SLF001
        daemon._coordinator = Coordinator()  # noqa: SLF001
        daemon._execution_repository = SimpleNamespace(  # noqa: SLF001
            list_expired_running_attempts=lambda **_kwargs: []
        )
        daemon._eaaef_running_recovery_snapshots = {  # noqa: SLF001
            attempt.attempt_id: snapshot
        }
        daemon.open = lambda: daemon
        daemon.list_running_attempts = lambda: [attempt]
        terminal_calls: list[dict[str, object]] = []
        daemon._commit_reconciled_attempt_terminal = (  # noqa: SLF001
            lambda _preparation, **kwargs: terminal_calls.append(dict(kwargs))
        )
        return daemon.reconcile_expired_running_attempts(), terminal_calls

    outcomes, terminal_calls = exercise({"state": "expired", "body": {}})
    assert outcomes[0]["lease_state"] == "expired"
    assert outcomes[0]["retry_required"] is True
    assert len(terminal_calls) == 1

    no_go = {
        "outcome": "mutation_not_admitted",
        "reason": "expired_attempt_has_provider_or_effect_reservation",
    }
    outcomes, terminal_calls = exercise(
        {"state": "expired", "body": {"recovery_no_go": no_go}}
    )
    assert outcomes == [
        {
            "task_cid": attempt.task_cid,
            "claim_id": attempt.claim_id,
            "attempt_id": attempt.attempt_id,
            "status": "blocked",
            "lease_state": "expired",
            "retry_required": False,
            "provider_evidence_reused": False,
            "effect_evidence_reused": False,
            "reason": "expired_attempt_has_provider_or_effect_reservation",
        }
    ]
    assert terminal_calls == []


def test_spoofed_eaaef_proxy_cannot_skip_mandatory_dead_lane_surface() -> None:
    class Coordinator:
        @staticmethod
        def expire_task_claim(*_args, **_kwargs):
            pytest.fail("spoofed EAAEF proxy reached claim expiry")

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.authority_mode = "quack"
    daemon._clock_ms = lambda: 1_900_000_000_000  # noqa: SLF001
    daemon._coordinator = Coordinator()  # noqa: SLF001
    daemon._execution_repository = SimpleNamespace(  # noqa: SLF001
        EAAEF_INTERFACE="EAAEFBootstrapExecutionRepositoryProxy@2",
        list_running_attempts=lambda **_kwargs: pytest.fail(
            "spoofed proxy was observed before exact-type validation"
        ),
    )
    daemon.open = lambda: daemon
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the exact EAAEF proxy",
    ):
        daemon.reconcile_expired_running_attempts()


def test_worker_cannot_approve_its_own_proposal() -> None:
    owner = _Owner()

    def launch(packet, _reservation):
        return _proposal(packet)

    dispatcher = _dispatcher(
        owner,
        launcher=launch,
        verifier=lambda packet, proposal: _verification(
            packet,
            dict(proposal),
            verifier=packet.worker_principal_did,
        ),
    )
    with pytest.raises(
        dispatch.ExternalAgentContainerDispatchError,
        match="independent passing verification",
    ):
        dispatcher.run_provider(_attempt())
    assert owner.commit_calls == 0


def test_host_source_mutation_is_rejected_and_never_committed() -> None:
    owner = _Owner()
    host_state = ["host-tree:clean"]

    def launch(packet, _reservation):
        host_state[0] = "host-tree:mutated"
        return _proposal(packet)

    dispatcher = _dispatcher(owner, launcher=launch, host_state=host_state)
    with pytest.raises(
        dispatch.ExternalAgentContainerDispatchError,
        match="forbidden host source mutation",
    ):
        dispatcher.run_provider(_attempt())

    assert owner.commit_calls == 0


def test_host_merge_admission_is_separate_from_worker_dispatch() -> None:
    owner = _Owner()
    admission: list[dict[str, object] | None] = [None]
    dispatcher = _dispatcher(
        owner,
        launcher=lambda packet, _reservation: _proposal(packet),
        merge_observer=lambda _packet, _effect: admission[0],
    )
    provider = dispatcher.run_provider(_attempt())
    effect = dispatcher.apply_effect(_attempt(), provider)

    assert effect["host_mutation_performed"] is False
    assert effect["merge_admitted"] is False
    with pytest.raises(dispatch.ExternalAgentContainerMergeAdmissionPending):
        dispatcher.validate_effect(_attempt(), effect)

    packet = _packet()
    claim = dispatch.ExternalAgentContainerWorkerDispatcher._dispatch_claim(  # noqa: SLF001
        packet
    )
    admission[0] = _seal(
        {
            "schema": dispatch.EXTERNAL_AGENT_HOST_MERGE_ADMISSION_SCHEMA,
            "interface": dispatch.EXTERNAL_AGENT_CONTAINER_WORKER_DISPATCHER_INTERFACE,
            "decision": "accepted",
            "delivery_mode": "reviewed_patch",
            "task_cid": packet.task_cid,
            "attempt_id": packet.attempt_id,
            "claim_cid": claim["claim_cid"],
            "accepted_result_receipt_id": effect["accepted_result_receipt_id"],
            "patch_artifact_cid": effect["patch_artifact_cid"],
            "reviewer_principal_did": "did:key:zhostreviewer",
            "effect_authority_cid": _cid("3"),
            "merge_commit": "",
        }
    )
    validation = dispatcher.validate_effect(_attempt(), effect)

    assert validation["outcome"] == "passed"
    assert validation["body"]["delivery_mode"] == "reviewed_patch"
    assert validation["body"]["control_claim_id"] == "claim:control:1"
    assert validation["body"]["admission_receipt"] == admission[0]


def test_eaaef_database_binding_never_falls_back_to_portal(tmp_path) -> None:
    portal_calls: list[str] = []

    class Portal:
        def __init__(self, **_kwargs):
            portal_calls.append("constructed")

    class Daemon:
        task_source = object()

        def bind_execution_callbacks(self, **_kwargs):
            pytest.fail("callbacks must not bind without a qualified dispatcher")

    parsed = SimpleNamespace(
        implement=True,
        worker_network_launch_authority_json='{"sealed":"authority"}',
    )
    with pytest.raises(
        RuntimeError,
        match="EAAEF container worker dispatch is unavailable_fail_closed",
    ):
        bind_database_portal_execution_from_args(
            Daemon(),
            parsed,
            repo_root=tmp_path,
            portal_daemon_class=Portal,
        )
    assert portal_calls == []


def test_source_qualification_truth_remains_fail_closed() -> None:
    evidence = dispatch.ExternalAgentContainerWorkerDispatcher.qualification_evidence()

    assert evidence["status"] == "unavailable_fail_closed"
    assert evidence["portal_fallback_allowed"] is False
    assert evidence["host_provider_allowed"] is False
    assert evidence["host_merge_inside_worker_allowed"] is False
    assert set(evidence["blockers"]) == set(
        dispatch.EXTERNAL_AGENT_CONTAINER_DISPATCH_BLOCKERS
    )
