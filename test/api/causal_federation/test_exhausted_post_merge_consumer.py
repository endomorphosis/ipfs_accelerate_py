"""The exhausted recovery receipt crosses the real Portal seed dispatch guard.

This uses real Git candidate ancestry and the real claim/seed validators.  The
independently qualified source projection and evidence are bounded fixtures;
final Portal projection acceptance is intercepted after seed admission.  This
therefore proves the consumer boundary, not native end-to-end task completion.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.exhausted_post_merge_recovery import (
    QUEUE_RECEIPT_SCHEMA,
    build_parameters,
    validate_parameters,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
)
from test.api.causal_federation.test_exhausted_post_merge_contract import (
    exhausted_fixture,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _callback_integration_recovery_evidence,
)
from test.api.test_agent_supervisor_merge_train import _git, _repo
from test.api.test_retained_callback_suffix import _seeded_dispatch_fixture


def _rehash(history: dict[str, Any]) -> None:
    history.pop("projection_cid", None)
    history["projection_cid"] = content_identity(history)


def _consumer(tmp_path: Path) -> SimpleNamespace:
    daemon, bridge, attempt, record, _seed, _receipt, history = (
        _seeded_dispatch_fixture(preflight_exhaustion=True)
    )
    task, prefix, prior_queue, transition = exhausted_fixture()
    source = daemon.get_attempt(transition["attempt_id"])
    assert source is not None
    repo = _repo(tmp_path)
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "base.txt").write_text("qualified retained candidate\n")
    _git(repo, "commit", "-am", "retained candidate")
    candidate = _git(repo, "rev-parse", "HEAD")

    evidence = _callback_integration_recovery_evidence(daemon, source)
    qualification = evidence["callback_requalification_receipt"]
    qualification.update(
        candidate_commit=candidate,
        integration_commit=candidate,
        current_target_commit=candidate,
        baseline_commit=baseline,
        validation=[{"task_id": task["task_alias"], "passed": True, "returncode": 0}],
    )
    qualification.pop("receipt_id")
    qualification["receipt_id"] = content_identity(qualification)
    evidence.update(
        candidate_commit=candidate,
        qualified_target_commit=candidate,
        callback_requalification_receipt_id=qualification["receipt_id"],
    )
    evidence.pop("evidence_id")
    evidence["evidence_id"] = daemon._database_portal_evidence_digest(evidence)
    seed = daemon._build_post_merge_completion_recovery_seed(
        attempt=source,
        task_revision=4,
        recovery_control_revision=11,
        evidence=evidence,
        qualified_target_commit=candidate,
        qualification_kind="callback_integration",
        qualification_receipt_id=qualification["receipt_id"],
        recovery_evidence_id=evidence["evidence_id"],
        terminal_reason=prefix["revisions"][3]["body"]["completion_receipt"]["reason"],
    )
    transition.update(
        candidate_commit=candidate,
        qualified_target_commit=candidate,
        source_integration_commit=candidate,
        callback_requalification_receipt_id=qualification["receipt_id"],
        callback_reconciliation_evidence_id=evidence["evidence_id"],
        post_merge_completion_recovery_seed=seed,
        queue_reason=("database_post_merge_declared_outputs_callback_integration:"
                      + seed["request_id"] + ":" + seed["qualification_receipt_id"]),
    )
    parameters = build_parameters(
        task=task,
        history=prefix,
        prior_queue=prior_queue,
        expected_control_receipt=task["body"]["completion_receipt"],
        transition_receipt=transition,
        now_ms=task["body"]["completion_receipt"]["execution_finished_at_ms"] + 1_000,
    )
    final = copy.deepcopy(validate_parameters(parameters)["final_transition_receipt"])
    assert final["queue_receipt"]["schema"] == QUEUE_RECEIPT_SCHEMA
    assert final["queue_receipt"]["source_identity"]["attempt_number"] == 1
    assert final["queue_receipt"]["current_identity"]["attempt_number"] == 3
    assert history["revisions"][:11] == prefix["revisions"]
    history["revisions"][11]["body"]["completion_receipt"] = final
    for row in history["revisions"][12:]:
        row["body"]["completion_receipt"]["post_merge_completion_recovery_seed"] = seed
    record.body = history["revisions"][-1]["body"]
    _rehash(history)

    binding = {
        "task_cid": task["task_cid"],
        "attempt_id": seed["queue_source_attempt_id"],
        "claim_id": seed["queue_source_claim_id"],
        "lease_id": seed["queue_source_lease_id"],
        "fencing_token": seed["queue_source_fencing_token"],
        "fence_epoch": seed["queue_source_fence_epoch"],
        "binding_id": seed["queue_source_binding_id"],
        "projection_immutable_digest": seed["queue_source_projection_immutable_digest"],
    }
    request = SimpleNamespace(
        task_id=task["task_alias"], commit_sha=candidate, metadata={},
    )
    bridge.task_source = SimpleNamespace(task_revision_history_projection=lambda _cid: history)
    bridge.merge_queue = SimpleNamespace(get=lambda _request_id: request)
    bridge.repository_root = repo
    bridge._owned_post_merge_recovery_projection = (
        lambda _request, **_kwargs: SimpleNamespace(binding=binding)
    )
    bridge._post_merge_recovery_evidence = lambda *_args, **_kwargs: evidence
    bridge._record_for_attempt = lambda *_args: record
    bridge._execution_route_binding = lambda **_kwargs: {}
    bridge._protected_preservation_seed_from_record = lambda **_kwargs: None
    bridge._post_commit_candidate_seed_from_record = lambda **_kwargs: None
    bridge._ensure_attempt_projection = lambda *_args: (None, None)
    bridge._verify_projection = lambda *_args: pytest.fail("ordinary provider path reached")
    accepted: list[dict[str, Any]] = []

    def accept(**kwargs: Any) -> dict[str, Any]:
        admitted = kwargs["seed"]
        accepted.append(admitted)
        return {"retained": True, "provider_dispatched": False, "seed_id": admitted["seed_id"]}

    bridge._accept_post_merge_completion_recovery_seed = accept
    return SimpleNamespace(
        bridge=bridge, attempt=attempt, record=record, seed=seed,
        history=history, predecessor=final, binding=binding, evidence=evidence,
        qualification=qualification, baseline=baseline, candidate=candidate,
        accepted=accepted,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "source_history",
        "history_gap",
        "admission_identity",
        "drop_seed",
        "queue_source_binding",
        "qualified_target",
        "qualification_evidence",
        "candidate_parent",
        "queue_current_identity",
        "queue_history_binding",
        "queue_schema_downgrade",
    ],
)
def test_exhausted_post_merge_receipt_survives_exact_consumer_dispatch(
    tmp_path: Path, mutation: str | None,
) -> None:
    case = _consumer(tmp_path)
    if mutation == "source_history":
        case.history["revisions"][3]["body"]["completion_receipt"]["claim_id"] = "claim:foreign"
    elif mutation == "history_gap":
        case.history["revisions"].pop(5)
    elif mutation == "admission_identity":
        case.record.body["completion_receipt"]["lease_id"] = "lease:foreign"
    elif mutation == "drop_seed":
        case.record.body["completion_receipt"].pop("post_merge_completion_recovery_seed")
    elif mutation == "queue_source_binding":
        case.binding["binding_id"] = "sha256:" + "f" * 64
    elif mutation == "qualified_target":
        case.evidence["qualified_target_commit"] = "f" * 40
    elif mutation == "qualification_evidence":
        case.evidence["evidence_id"] = "sha256:" + "f" * 64
    elif mutation == "candidate_parent":
        case.qualification["baseline_commit"] = case.candidate
    elif mutation == "queue_current_identity":
        case.predecessor["queue_receipt"]["current_identity"]["attempt_number"] = 999
    elif mutation == "queue_history_binding":
        case.predecessor["queue_receipt"]["history_projection_cid"] = "sha256:" + "f" * 64
    elif mutation == "queue_schema_downgrade":
        case.predecessor["queue_receipt"]["schema"] = "unrecognized-legacy-queue@1"
    _rehash(case.history)

    if mutation is None:
        admitted = case.bridge._post_merge_completion_recovery_seed_from_record(
            attempt=case.attempt, record=case.record,
        )
        assert admitted["baseline_commit"] == case.baseline
        assert admitted["candidate_commit"] == case.candidate
        assert admitted["seed_id"] == case.seed["seed_id"]
        assert case.bridge.run_provider(case.attempt) == {
            "retained": True,
            "provider_dispatched": False,
            "seed_id": case.seed["seed_id"],
        }
        assert len(case.accepted) == 1
        assert case.accepted[0]["recovery_evidence"] == case.evidence
    else:
        with pytest.raises(DatabasePortalBridgeError):
            case.bridge.run_provider(case.attempt)
        assert case.accepted == []
