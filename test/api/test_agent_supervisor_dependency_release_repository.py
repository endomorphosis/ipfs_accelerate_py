"""Source-only integration of metadata dependency resolution.

Ported from the preserved 20261114 release targets. Native/probe-specific
producer routes remain on the preservation branch and are not claimed here.
"""
from pathlib import Path
import hashlib
import json
import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import duckdb_available
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository, IntentRepository, IntentRepositoryConflictError, IntentRepositoryError,
    task_authority_spec_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.dependency_block_resolution import (
    catalogue_resolution_decision, dependency_release_decision,
    dependency_resolution_receipt, markdown_blocked_task_can_reopen,
)
pytestmark = pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
_BOILERPLATE = "Requires a reviewed current job packet, exact eligible inputs and fresh purpose-specific native admission."

def _manual_body(reason: str, *, stage: str = "validation") -> dict[str, object]:
    return {
        "completion": "manual",
        "execution_packet_required": True,
        "blocked_reason": reason,
        "resource_stage": stage,
        "required_stage_packets": [stage],
        "gold_or_teacher_admission_granted": False,
        "admitted": False,
        "formalized": False,
        "constitution_roundtrip_ok": False,
    }

def test_classifier_keeps_review_publication_and_unknown_reasons() -> None:
    boilerplate = _manual_body(_BOILERPLATE, stage="corpus")
    assert (
        catalogue_resolution_decision(
            status="blocked",
            body=boilerplate,
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "release"
    )
    review = _manual_body(
        "Requires actual independent legal/formal reviewer capabilities; "
        "agents may prepare packets but cannot invent reviews.",
        stage="semantic_review",
    )
    assert (
        catalogue_resolution_decision(
            status="blocked",
            body=review,
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "substantive_review_or_publication_hold"
    )
    publication = _manual_body(_BOILERPLATE, stage="publication")
    publication["required_stage_packets"] = ["publication"]
    assert (
        dependency_release_decision(
            status="blocked",
            body=publication,
            dependency_statuses=("completed",),
            active_block=False,
            criteria=(
                "No automatic runtime/default/gold/teacher promotion; "
                "preserve prior coherent releases and failure evidence.",
            ),
        )
        == "substantive_review_or_publication_hold"
    )
    assert (
        dependency_release_decision(
            status="blocked",
            body={"blocked_reason": "operator_reconciliation_required"},
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "unclassified_blocked_reason"
    )
    assert markdown_blocked_task_can_reopen(
        {"blocked reason": ""}, depends_on_count=1
    )
    assert markdown_blocked_task_can_reopen(
        {"blocked reason": _BOILERPLATE}, depends_on_count=1
    )
    assert not markdown_blocked_task_can_reopen(
        {"blocked reason": "operator_reconciliation_required"},
        depends_on_count=1,
    )
    assert not markdown_blocked_task_can_reopen({}, depends_on_count=0)
    receipt = dependency_resolution_receipt()
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False

def _seed(tmp_path: Path):
    repo = open_intent_repository(
        tmp_path / "control.duckdb",
        owner_id="owner:dependency-release",
        evidence_freshness_seconds=0,
    )
    repo.upsert_objective(
        objective_id="objective:scientific",
        objective_alias="scientific",
        title="Scientific",
    )
    repo.upsert_goal(
        goal_cid="goal:scientific",
        goal_alias="scientific-goal",
        title="Scientific goal",
        objective_id="objective:scientific",
    )
    repo.upsert_plan(
        plan_cid="plan:scientific",
        goal_cid="goal:scientific",
        plan_alias="scientific-plan",
        status="active",
    )
    return repo

def _task(
    repo,
    cid: str,
    alias: str,
    *,
    status: str,
    body: dict,
    dependencies=(),
    acceptance=None,
):
    repo.upsert_task(
        task_cid=cid,
        task_alias=alias,
        goal_cid="goal:scientific",
        plan_cid="plan:scientific",
        objective_id="objective:scientific",
        status=status,
        body=body,
        dependencies=list(dependencies),
        acceptance=acceptance
        or [
            {
                "criterion": "current validation evidence is already recorded",
                "evidence_kind": "validation",
            }
        ],
    )

def test_release_completes_only_recorded_evidence_and_leaves_holds(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:ready-evidence",
        "LIR-EVIDENCE",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:pending",
        "LIR-PENDING",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "agents may prepare packets but cannot invent reviews.",
            stage="semantic_review",
        ),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:open-dep",
        "LIR-WAITING",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:review"],
    )
    _bound_record(repo,
        task_cid="task:ready-evidence",
        evidence_kind="validation",
        digest="sha256:" + ("ab" * 32),
        body={"admitted": False, "formalized": False},
    )

    result = repo.release_dependency_satisfied_blocks()

    completed = {item["task_alias"] for item in result["completed"]}
    pending = {item["task_alias"] for item in result["pending"]}
    held = {item["task_alias"]: item["decision"] for item in result["held"]}
    assert completed == {"LIR-EVIDENCE"}
    assert pending == {"LIR-PENDING"}
    assert result["spawned"] == []
    assert held["LIR-REVIEW"] == "substantive_review_or_publication_hold"
    assert "LIR-WAITING" not in held
    assert "LIR-WAITING" not in pending
    evidence = repo.get_task("task:ready-evidence")
    assert evidence is not None
    assert evidence["status"] == "completed"
    receipt = evidence["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_from_current_evidence"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    assert evidence["body"]["gold_or_teacher_admission_granted"] is False
    assert repo.get_task("task:pending")["status"] == "blocked"
    assert repo.get_task("task:review")["status"] == "blocked"
    assert repo.get_task("task:open-dep")["status"] == "blocked"

    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert {item["task_alias"] for item in again["pending"]} == {"LIR-PENDING"}
    assert again["spawned"] == []
    assert repo.get_task("LIR-PENDING--missing-record") is None
    assert repo.get_task("task:pending")["dependencies"] == ("task:predecessor",)

def test_release_creates_subgoals_for_an_unresolved_high_level_goal(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body=_manual_body(
            "The high-level goal stays blocked until each clause is recorded."
        ),
        dependencies=["task:predecessor"],
        acceptance=(
            "Every accepted component retains complete material meaning.",
            "Input context and output capacities stay separate.",
        ),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert result["completed"] == []
    assert {item["task_alias"] for item in result["spawned"]} == {
        "LIR-GOAL--00-every-accepted-component-retains",
        "LIR-GOAL--01-input-context-output-capacities",
    }
    parent = repo.get_task("task:goal")
    assert parent["status"] == "blocked"
    assert parent["body"]["resolution_decomposed"] is True
    for alias in (
        "LIR-GOAL--00-every-accepted-component-retains",
        "LIR-GOAL--01-input-context-output-capacities",
    ):
        child = repo.get_task(alias)
        assert child["status"] == "blocked"
        assert child["body"]["resolution_subgoal"] is True
        assert child["body"]["admitted"] is False
        assert child["body"]["formalized"] is False
        assert child["body"]["gold_or_teacher_admission_granted"] is False
        assert child["body"]["constitution_roundtrip_ok"] is False
        assert child["body"]["completion_authority"] is False
        assert child["task_cid"] in parent["dependencies"]
    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []
    assert repo.get_task("task:goal")["status"] == "blocked"

def test_dependency_chain_requires_fresh_evidence_after_predecessor_revision(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:tail",
        "LIR-TAIL",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:mid"],
    )
    _task(
        repo,
        "task:mid",
        "LIR-MID",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "Requires actual independent review of the assembled interpretation.",
            stage="semantic_review",
        ),
        dependencies=["task:tail"],
    )
    for cid in ("task:tail", "task:mid"):
        _bound_record(repo,
            task_cid=cid,
            evidence_kind="validation",
            digest="sha256:" + ("cd" if cid.endswith("tail") else "ef") * 32,
            body={"admitted": False, "formalized": False},
        )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-MID"}
    assert repo.get_task("task:tail")["status"] == "blocked"
    assert {item["task_alias"] for item in result["pending"]} == {"LIR-TAIL"}
    # A producer must explicitly rebind the tail to the completed dependency.
    _bound_record(repo, task_cid="task:tail", evidence_kind="validation",
                  digest="sha256:" + "ab" * 32, body={"admitted": False, "formalized": False})
    refreshed = repo.release_dependency_satisfied_blocks()
    assert {item["task_alias"] for item in refreshed["completed"]} == {"LIR-TAIL"}
    assert {item["task_alias"] for item in refreshed["held"]} == {"LIR-REVIEW"}
    assert repo.get_task("task:review")["status"] == "blocked"
    for cid in ("task:mid", "task:tail"):
        receipt = repo.get_task(cid)["body"]["completion_receipt"]
        assert receipt["admitted"] is receipt["formalized"] is receipt["gold_or_teacher_admission_granted"] is False



def _bound_record(repo, *, task_cid, evidence_kind, digest, body):
    task = repo.get_task(task_cid)
    snapshot = [
        {"task_cid": dep["task_cid"], "revision": dep["revision"], "status": dep["status"],
         "spec_cid": task_authority_spec_cid(dep)}
        for dep in (repo.get_task(cid) for cid in task["dependencies"])
    ]
    return repo.record_evidence(task_cid=task_cid, evidence_kind=evidence_kind, digest=digest,
        body={**body, "task_authority_spec_cid": task_authority_spec_cid(task),
              "dependency_snapshot": snapshot})


def _gate(tmp_path, *, additions=None, identity=None, criterion=None):
    repo = _seed(tmp_path)
    _task(repo, "task:predecessor", "DONE", status="completed", body={})
    _task(repo, "task:gate", "GATE", status="blocked",
          body={**_manual_body(_BOILERPLATE), **(additions or {})},
          dependencies=["task:predecessor"],
          acceptance=criterion)
    if identity:
        task = repo.get_task("task:gate")
        repo.upsert_task(task_cid=task["task_cid"], task_alias=task["task_alias"],
                         goal_cid=task["goal_cid"], status="blocked", body=dict(task["body"]),
                         identity=identity, expected_revision=task["revision"])
    return repo


def _record(repo, *, bound=True, **extra):
    task = repo.get_task("task:gate")
    body = {**dependency_resolution_receipt(), **extra}
    if bound:
        body["task_authority_spec_cid"] = task_authority_spec_cid(task)
        body["dependency_snapshot"] = [
            {"task_cid": dep["task_cid"], "revision": dep["revision"], "status": dep["status"],
             "spec_cid": task_authority_spec_cid(dep)}
            for dep in (repo.get_task(cid) for cid in task["dependencies"])
        ]
        body["source_bindings"] = {
            section: {key: value for key, value in task[section].items() if key in {
                "source_binding", "source_sha256", "source_digest", "source_revision",
                "artifact_sha256", "artifact_digest", "interpretation_version",
                "dependency_version", "toolchain_profile"}}
            for section in ("body", "identity")
        }
        body["source_bindings"] = {key: value for key, value in body["source_bindings"].items() if value}
    return repo.record_evidence(task_cid=task["task_cid"], evidence_kind="validation",
                                digest="sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                                body=body)


def test_source_bound_evidence_completes_exact_current_specification(tmp_path):
    repo = _gate(tmp_path, additions={"source_sha256": "a" * 64, "interpretation_version": "pending-v1"})
    _record(repo, bound=True)
    original = repo.get_task("task:gate")
    result = repo.release_dependency_satisfied_blocks(expected_revisions={"task:gate": original["revision"]})
    assert [item["task_alias"] for item in result["completed"]] == ["GATE"]
    task = repo.get_task("task:gate")
    receipt = task["body"]["completion_receipt"]
    assert receipt["task_authority_spec_cid"] == task_authority_spec_cid(original)
    assert receipt["source_bindings"]["body"]["source_sha256"] == "a" * 64
    assert receipt["dependency_snapshot"][0]["task_cid"] == "task:predecessor"
    assert task["body"]["interpretation_version"] == "pending-v1"
    assert receipt["admitted"] is receipt["training_authorized"] is receipt["completion_authority"] is False
    with repo._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM completion_receipts WHERE task_cid='task:gate'").fetchone()[0] == 1


@pytest.mark.parametrize("field", ["source_sha256", "source_binding", "artifact_digest", "interpretation_version", "dependency_version", "toolchain_profile", "unrecognized_source_identity"])
def test_changed_source_artifact_or_version_refuses_old_evidence(tmp_path, field):
    repo = _gate(tmp_path, additions={field: "original"})
    _record(repo, bound=True)
    task = repo.get_task("task:gate")
    body = {**task["body"], field: "changed"}
    repo.upsert_task(task_cid=task["task_cid"], task_alias=task["task_alias"], goal_cid=task["goal_cid"], status="blocked",
                     body=body, expected_revision=task["revision"])
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"
    assert [item["task_alias"] for item in result["pending"]] == ["GATE"]


def test_an_associated_receipt_cannot_replace_source_bound_evidence(tmp_path):
    repo = _gate(tmp_path, additions={"source_digest": "retained-edition-2024"})
    _record(repo, bound=False)
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_changed_acceptance_cannot_reuse_an_old_bound_receipt(tmp_path):
    repo = _gate(tmp_path)
    _record(repo, bound=True)
    task = repo.get_task("task:gate")
    repo.upsert_task(task_cid=task["task_cid"], task_alias=task["task_alias"], goal_cid=task["goal_cid"], status="blocked",
                     body=dict(task["body"]), expected_revision=task["revision"],
                     acceptance=[{"criterion": "different obligation", "evidence_kind": "validation"}])
    assert repo.release_dependency_satisfied_blocks()["completed"] == []


@pytest.mark.parametrize("additions", [
    {"permission_expires_at_ms": 1}, {"permission": {"state": "expired"}},
    {"budget_exhausted": True}, {"retry_budget": 0},
    {"budgets": {"max_attempts": 0}}, {"source_lease": {"expires_at_ms": 1}},
])
def test_metadata_release_never_recovers_runtime_permission_or_budget(tmp_path, additions):
    repo = _gate(tmp_path, additions=additions)
    _record(repo)
    before = dict(repo.get_task("task:gate"))
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == result["spawned"] == []
    assert result["held"][0]["decision"] == "runtime_authority_hold"
    assert dict(repo.get_task("task:gate")) == before


def test_expired_persisted_lease_and_retry_counters_are_left_to_existing_owner(tmp_path):
    repo = _gate(tmp_path)
    _record(repo)
    repo.record_queue_backoff(task_cid="task:gate", delay_ms=0, reason="exhausted budget")
    with repo._connection() as connection:
        before = tuple(connection.execute("SELECT * FROM leases WHERE task_cid='task:gate'").fetchone().values())
    result = repo.release_dependency_satisfied_blocks()
    assert result["held"][0]["decision"] == "runtime_authority_hold"
    with repo._connection() as connection:
        after = tuple(connection.execute("SELECT * FROM leases WHERE task_cid='task:gate'").fetchone().values())
    assert after == before
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_active_block_stays_active_with_fresh_evidence(tmp_path):
    repo = _gate(tmp_path)
    _record(repo)
    repo.block_task(task_cid="task:gate", blocker_kind="operator", blocker_id="retain-hold", reason="human hold")
    before = dict(repo.get_task("task:gate"))
    result = repo.release_dependency_satisfied_blocks()
    assert result["held"][0]["decision"] == "active_block"
    assert dict(repo.get_task("task:gate")) == before
    with repo._connection() as connection:
        assert connection.execute("SELECT state FROM task_blocks WHERE task_cid='task:gate'").fetchone()[0] == "active"


@pytest.mark.parametrize("flag", ["admitted", "formalized", "completion_authority", "training_authorized", "native_admission", "reviewer_claimed"])
def test_evidence_claiming_authority_is_never_a_metadata_completion(tmp_path, flag):
    repo = _gate(tmp_path)
    _record(repo, **{flag: True})
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_stale_snapshot_refuses_all_changes_before_any_completion(tmp_path):
    repo = _gate(tmp_path)
    _record(repo)
    with pytest.raises(IntentRepositoryConflictError, match="snapshot CAS is stale"):
        repo.release_dependency_satisfied_blocks(expected_revisions={"task:predecessor": 999})
    assert repo.get_task("task:gate")["status"] == "blocked"
    with repo._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM completion_receipts").fetchone()[0] == 0


def test_child_identity_conflict_rolls_back_all_spawns_and_events(tmp_path):
    repo = _gate(tmp_path, criterion=["Every accepted component retains complete material meaning.",
                                     "Input context and output capacities stay separate."])
    _task(repo, "task:gate--01-input-context-output-capacities", "existing-child",
          status="blocked", body={"immutable": "existing identity"})
    with repo._connection() as connection:
        before = connection.execute("SELECT COUNT(*) FROM domain_events").fetchone()[0]
    with pytest.raises(IntentRepositoryConflictError, match="identity already exists"):
        repo.release_dependency_satisfied_blocks()
    assert repo.get_task("task:gate--00-every-accepted-component-retains") is None
    assert "resolution_decomposed" not in repo.get_task("task:gate")["body"]
    with repo._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM domain_events").fetchone()[0] == before


def test_changed_dependency_revision_invalidates_bound_evidence(tmp_path):
    repo = _gate(tmp_path, additions={"source_digest": "retained-edition-2024"})
    _record(repo, bound=True)
    predecessor = repo.get_task("task:predecessor")
    repo.upsert_task(task_cid=predecessor["task_cid"], task_alias=predecessor["task_alias"],
                     goal_cid=predecessor["goal_cid"], status="completed", body={"dependency_revision": "changed"},
                     expected_revision=predecessor["revision"])
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"


@pytest.mark.parametrize("created_at", ["2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00", "invalid"])
def test_stale_future_or_invalid_evidence_is_not_current(tmp_path, created_at):
    repo = _gate(tmp_path)
    repo.evidence_freshness_seconds = 60
    _record(repo)
    with repo._connection(write=True) as connection:
        connection.execute("UPDATE evidence_nodes SET created_at=? WHERE task_cid='task:gate'", [created_at])
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_an_unbound_kind_match_cannot_satisfy_another_bound_criterion(tmp_path):
    repo = _gate(tmp_path, additions={"source_digest": "retained-edition-2024"}, criterion=[
        {"criterion": "validation current", "evidence_kind": "validation"},
        {"criterion": "source correspondence current", "evidence_kind": "source_correspondence"},
    ])
    _record(repo, bound=True)
    # A duplicate digest with a different stored evidence identity/kind must
    # not smuggle an unbound criterion through the eligible-row filter.
    shared_digest = repo.current_evidence_for_task("task:gate")[0]["digest"]
    repo.record_evidence(task_cid="task:gate", evidence_kind="source_correspondence",
                         digest=shared_digest, body={"artifact_association": "only"})
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_bound_repository_requires_an_active_owner_transaction(tmp_path):
    repo = _gate(tmp_path)
    _record(repo)
    with repo._connection() as connection:
        borrowed = IntentRepository(bound_connection=connection)
        with pytest.raises(IntentRepositoryError, match="active owner transaction"):
            borrowed.release_dependency_satisfied_blocks()
    assert repo.get_task("task:gate")["status"] == "blocked"
    with repo._connection(write=True) as connection:
        borrowed = IntentRepository(bound_connection=connection)
        assert borrowed.release_dependency_satisfied_blocks()["completed"][0]["task_alias"] == "GATE"
    assert repo.get_task("task:gate")["status"] == "completed"


def test_unbound_metadata_evidence_is_pending_even_without_named_source_fields(tmp_path):
    repo = _gate(tmp_path)
    _record(repo, bound=False)
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert result["pending"][0]["task_alias"] == "GATE"
    assert "required:bound_current_evidence" in result["pending"][0]["missing"]


def test_bare_note_with_no_acceptance_is_not_an_automatic_completion(tmp_path):
    repo = _gate(tmp_path)
    task = repo.get_task("task:gate")
    repo.upsert_task(task_cid=task["task_cid"], task_alias=task["task_alias"], goal_cid=task["goal_cid"],
                     status="blocked", body=dict(task["body"]), acceptance=[], expected_revision=task["revision"])
    _bound_record(repo, task_cid="task:gate", evidence_kind="note", digest="sha256:" + "a" * 64, body={"note": "associated only"})
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert "required:explicit_acceptance" in result["pending"][0]["missing"]
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_bound_note_cannot_satisfy_an_explicit_validation_policy(tmp_path):
    repo = _gate(tmp_path)
    _bound_record(repo, task_cid="task:gate", evidence_kind="note", digest="sha256:" + "a" * 64, body={"note": "not validation"})
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert "kind:validation" in result["pending"][0]["missing"]
    assert repo.get_task("task:gate")["status"] == "blocked"


def test_bare_note_cannot_satisfy_an_untyped_acceptance_criterion(tmp_path):
    repo = _gate(tmp_path, criterion=["A producer must verify this criterion."])
    _bound_record(repo, task_cid="task:gate", evidence_kind="note", digest="sha256:" + "a" * 64, body={"note": "not validation"})
    assert repo.release_dependency_satisfied_blocks()["completed"] == []
    assert repo.get_task("task:gate")["status"] == "blocked"
