"""Focused tests for the pure campaign progress projector."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.campaign_progress import (
    CAMPAIGN_PROGRESS_CURRENT_MANIFEST_SCHEMA,
    CAMPAIGN_PROGRESS_REPORT_SCHEMA,
    COMPLETION_EVIDENCE_SCHEMA,
    INTENT_COMPLETION_PROJECTION_SCHEMA,
    NON_AUTHORITATIVE_BANNER,
    PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA,
    TYPED_COMPLETION_PROGRESS_REQUEST_SCHEMA,
    TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
    CampaignProgressDestinationError,
    CampaignProgressValidationError,
    CampaignProgressWriteError,
    build_program_qualification_disposition,
    render_campaign_progress,
    write_campaign_progress_outputs,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    StoreGeneration,
    content_identity,
)

TASKS = {
    "EAAEF-001": "task:cid:001",
    "EAAEF-002": "task:cid:002",
    "EAAEF-003": "task:cid:003",
    "EAAEF-004": "task:cid:004",
}


def _completion_receipt(
    task_cid: str,
    revision: int,
    *,
    completed_at: str = "2026-08-26T12:00:00Z",
    control_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    if control_receipt is None:
        control_receipt = {
            "outcome": "passed",
            "test_evidence": "sha256:" + ("1a" * 32),
        }
    evidence_digests = ["sha256:" + ("2b" * 32)]
    evidence_digest = content_identity(
        {
            "task_cid": task_cid,
            "revision": revision,
            "receipt": control_receipt,
            "evidence_digests": evidence_digests,
        }
    )
    receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": task_cid,
            "revision": revision,
            "evidence_digest": evidence_digest,
        }
    )
    return {
        "receipt_cid": receipt_cid,
        "task_cid": task_cid,
        "goal_cid": "goal:eaaef",
        "attempt_id": "attempt:001",
        "claim_cid": "claim:001",
        "fencing_token": 7,
        "completed_at": completed_at,
        "validation_run_id": "validation:001",
        "evidence_digest": evidence_digest,
        "body": {
            "schema": COMPLETION_EVIDENCE_SCHEMA,
            "receipt": control_receipt,
            "evidence_digests": evidence_digests,
            "revision": revision,
        },
    }


def _seal_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    material = {key: value for key, value in snapshot.items() if key != "projection_cid"}
    snapshot["projection_cid"] = content_identity(material)
    return snapshot


def _snapshot() -> dict[str, object]:
    return _seal_snapshot(
        {
            "schema": INTENT_COMPLETION_PROJECTION_SCHEMA,
            "event_watermark": 42,
            "task_states": [
                {"task_cid": "task:cid:001", "status": "completed", "revision": 3},
                {"task_cid": "task:cid:002", "status": "done", "revision": 2},
                {"task_cid": "task:cid:003", "status": "ready", "revision": 1},
                {"task_cid": "task:cid:004", "status": "skipped", "revision": 4},
            ],
            "completion_receipts": [_completion_receipt("task:cid:001", 3)],
        }
    )


def _seal_owner_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    material = {key: value for key, value in snapshot.items() if key != "snapshot_cid"}
    snapshot["snapshot_cid"] = content_identity(material)
    return snapshot


def _request_cid(owner_identity: dict[str, object]) -> str:
    return content_identity(
        {
            "schema": TYPED_COMPLETION_PROGRESS_REQUEST_SCHEMA,
            "task_cids": sorted(TASKS.values()),
            "expected_server_id": owner_identity["server_id"],
            "expected_process_birth_id": owner_identity["process_birth_id"],
            "expected_store_id": owner_identity["store_id"],
            "expected_database_uuid": owner_identity["database_uuid"],
            "expected_generation": owner_identity["generation"],
            "expected_fence_epoch": owner_identity["fence_epoch"],
        }
    )


def _owner_snapshot(
    completion_projection: dict[str, object] | None = None,
) -> dict[str, object]:
    owner_identity = {
        "server_id": "server:progress:001",
        "process_birth_id": "birth:progress:001",
        "store_id": "store:progress:001",
        "database_uuid": "12345678-1234-4678-9234-567812345678",
        "generation": 3,
        "fence_epoch": 7,
    }
    generation = StoreGeneration(
        store_id=owner_identity["store_id"],
        generation=owner_identity["generation"],
        schema_revision=14,
        fence_epoch=owner_identity["fence_epoch"],
        revision=19,
        database_uuid=owner_identity["database_uuid"],
        birth_id=owner_identity["process_birth_id"],
    )
    return _seal_owner_snapshot(
        {
            "schema": TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
            "request_cid": _request_cid(owner_identity),
            "owner_identity": owner_identity,
            "store_generation": generation.to_dict(),
            "completion_projection": completion_projection or _snapshot(),
        }
    )


def _qualification() -> dict[str, object]:
    return dict(
        build_program_qualification_disposition(
            program_id="EAAEF",
            status="blocked",
            blockers=("DuckLake qualification not run", "Live benchmark not run"),
            evidence_refs=("snapshot:current",),
        )
    )


def test_render_is_deterministic_and_separates_progress_claim_classes() -> None:
    snapshot = _owner_snapshot()
    qualification = _qualification()

    first = render_campaign_progress(
        snapshot,
        sealed_tasks=TASKS,
        qualification=qualification,
    )
    second = render_campaign_progress(
        copy.deepcopy(snapshot),
        sealed_tasks=dict(reversed(tuple(TASKS.items()))),
        qualification=copy.deepcopy(qualification),
    )

    assert first.json_text == second.json_text
    assert first.markdown_text == second.markdown_text
    report = first.to_dict()
    assert report["schema"] == CAMPAIGN_PROGRESS_REPORT_SCHEMA
    assert report["authoritative"] is False
    assert report["banner"] == NON_AUTHORITATIVE_BANNER
    assert report["operational_state"]["task_count"] == 4
    assert report["operational_state"]["operational_completion_count"] == 3
    assert report["operational_state"]["counts_by_status"] == {
        "completed": 1,
        "done": 1,
        "ready": 1,
        "skipped": 1,
    }
    backed = report["current_revision_normalized_receipt_backed_completions"]
    assert backed["count"] == 1
    assert [item["task_alias"] for item in backed["tasks"]] == ["EAAEF-001"]
    lacking = report["operational_completions_without_normalized_evidence"]
    assert lacking["count"] == 2
    assert [item["task_alias"] for item in lacking["tasks"]] == [
        "EAAEF-002",
        "EAAEF-004",
    ]
    assert report["program_qualification"]["qualified"] is False
    assert report["program_qualification"]["status"] == "blocked"
    assert report["source_snapshot"]["snapshot_cid"] == snapshot["snapshot_cid"]
    assert report["source_snapshot"]["owner_identity"] == snapshot["owner_identity"]
    assert report["source_snapshot"]["store_generation"] == snapshot["store_generation"]
    assert report["source_snapshot"]["completion_projection_cid"] == snapshot[
        "completion_projection"
    ]["projection_cid"]
    report_body = dict(report)
    report_cid = report_body.pop("report_cid")
    assert report_cid == content_identity(report_body)
    assert json.loads(first.json_text) == report
    assert first.json_text.endswith("\n")
    assert NON_AUTHORITATIVE_BANNER in first.markdown_text
    assert "## Operational state" in first.markdown_text
    assert "## Current-revision normalized receipt-backed completions" in (first.markdown_text)
    assert "## Operational completions lacking normalized evidence" in (first.markdown_text)
    assert "## Program qualification and blockers" in first.markdown_text


def test_render_accepts_canonical_empty_control_receipt() -> None:
    snapshot = _snapshot()
    receipts = snapshot["completion_receipts"]
    assert isinstance(receipts, list)
    receipts[0] = _completion_receipt(
        "task:cid:001",
        3,
        control_receipt={},
    )
    _seal_snapshot(snapshot)

    report = render_campaign_progress(
        _owner_snapshot(snapshot),
        sealed_tasks=TASKS,
        qualification=_qualification(),
    ).to_dict()

    backed = report["current_revision_normalized_receipt_backed_completions"]
    assert backed["count"] == 1
    assert backed["tasks"][0]["control_receipt_cid"] == content_identity({})


@pytest.mark.parametrize(
    "sealed_tasks",
    (
        {"EAAEF-001": "task:cid:001"},
        {**TASKS, "EAAEF-005": "task:cid:005"},
        {**TASKS, "EAAEF-005": "task:cid:001"},
    ),
)
def test_render_rejects_wrong_or_duplicate_sealed_populations(
    sealed_tasks: dict[str, str],
) -> None:
    with pytest.raises(CampaignProgressValidationError):
        render_campaign_progress(
            _owner_snapshot(),
            sealed_tasks=sealed_tasks,
            qualification=_qualification(),
        )


def test_render_rejects_duplicate_and_missing_snapshot_task_rows() -> None:
    duplicate = _snapshot()
    states = duplicate["task_states"]
    assert isinstance(states, list)
    states.insert(1, copy.deepcopy(states[0]))
    _seal_snapshot(duplicate)
    with pytest.raises(CampaignProgressValidationError, match="duplicate task states"):
        render_campaign_progress(
            _owner_snapshot(duplicate), sealed_tasks=TASKS, qualification=_qualification()
        )

    missing = _snapshot()
    missing_states = missing["task_states"]
    assert isinstance(missing_states, list)
    missing_states.pop()
    _seal_snapshot(missing)
    with pytest.raises(CampaignProgressValidationError, match="population"):
        render_campaign_progress(
            _owner_snapshot(missing), sealed_tasks=TASKS, qualification=_qualification()
        )


def test_render_rejects_unsupported_or_tampered_snapshot_projection() -> None:
    unsupported = _snapshot()
    unsupported["schema"] = "unsupported@2"
    _seal_snapshot(unsupported)
    with pytest.raises(CampaignProgressValidationError, match="unsupported"):
        render_campaign_progress(
            _owner_snapshot(unsupported), sealed_tasks=TASKS, qualification=_qualification()
        )

    tampered = _snapshot()
    tampered["event_watermark"] = 43
    with pytest.raises(CampaignProgressValidationError, match="projection CID"):
        render_campaign_progress(
            _owner_snapshot(tampered), sealed_tasks=TASKS, qualification=_qualification()
        )


def test_render_rejects_nested_receipt_identity_tampering() -> None:
    tampered = _snapshot()
    receipts = tampered["completion_receipts"]
    assert isinstance(receipts, list)
    receipt = receipts[0]
    assert isinstance(receipt, dict)
    body = receipt["body"]
    assert isinstance(body, dict)
    evidence = body["evidence_digests"]
    assert isinstance(evidence, list)
    evidence.append("sha256:" + ("3c" * 32))
    # Re-signing only the outer projection must not bless a forged nested
    # receipt identity.
    _seal_snapshot(tampered)

    with pytest.raises(CampaignProgressValidationError, match="evidence identity"):
        render_campaign_progress(
            _owner_snapshot(tampered), sealed_tasks=TASKS, qualification=_qualification()
        )


def test_render_rejects_stale_but_internally_resealed_receipt_revision() -> None:
    stale = _snapshot()
    receipts = stale["completion_receipts"]
    assert isinstance(receipts, list)
    replacement = _completion_receipt("task:cid:001", 2)
    receipts[0] = replacement
    _seal_snapshot(stale)

    with pytest.raises(CampaignProgressValidationError, match="stale"):
        render_campaign_progress(
            _owner_snapshot(stale), sealed_tasks=TASKS, qualification=_qualification()
        )


def test_render_rejects_malformed_nested_receipt_and_duplicate_receipts() -> None:
    malformed = _snapshot()
    receipts = malformed["completion_receipts"]
    assert isinstance(receipts, list)
    receipt = receipts[0]
    assert isinstance(receipt, dict)
    body = receipt["body"]
    assert isinstance(body, dict)
    body.pop("schema")
    _seal_snapshot(malformed)
    with pytest.raises(CampaignProgressValidationError, match="fields are not closed"):
        render_campaign_progress(
            _owner_snapshot(malformed), sealed_tasks=TASKS, qualification=_qualification()
        )

    duplicate = _snapshot()
    duplicate_receipts = duplicate["completion_receipts"]
    assert isinstance(duplicate_receipts, list)
    duplicate_receipts.append(copy.deepcopy(duplicate_receipts[0]))
    _seal_snapshot(duplicate)
    with pytest.raises(CampaignProgressValidationError, match="duplicate"):
        render_campaign_progress(
            _owner_snapshot(duplicate), sealed_tasks=TASKS, qualification=_qualification()
        )


def test_render_rejects_tampered_or_unsupported_qualification_disposition() -> None:
    tampered = _qualification()
    blockers = tampered["blockers"]
    assert isinstance(blockers, list)
    blockers[0] = "Different blocker"
    with pytest.raises(CampaignProgressValidationError, match="disposition CID"):
        render_campaign_progress(_owner_snapshot(), sealed_tasks=TASKS, qualification=tampered)

    unsupported = _qualification()
    unsupported["schema"] = PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA.replace("@1", "@2")
    with pytest.raises(CampaignProgressValidationError, match="unsupported"):
        render_campaign_progress(_owner_snapshot(), sealed_tasks=TASKS, qualification=unsupported)


def test_render_validates_and_binds_outer_owner_generation_and_fence() -> None:
    baseline_snapshot = _owner_snapshot()
    baseline = render_campaign_progress(
        baseline_snapshot, sealed_tasks=TASKS, qualification=_qualification()
    )

    resealed_request_tamper = copy.deepcopy(baseline_snapshot)
    resealed_request_tamper["request_cid"] = content_identity(
        {"forged": "different request"}
    )
    _seal_owner_snapshot(resealed_request_tamper)
    with pytest.raises(CampaignProgressValidationError, match="request CID"):
        render_campaign_progress(
            resealed_request_tamper,
            sealed_tasks=TASKS,
            qualification=_qualification(),
        )

    mismatched = copy.deepcopy(baseline_snapshot)
    owner = mismatched["owner_identity"]
    assert isinstance(owner, dict)
    owner["fence_epoch"] = 8
    mismatched["request_cid"] = _request_cid(owner)
    _seal_owner_snapshot(mismatched)
    with pytest.raises(CampaignProgressValidationError, match="identities differ"):
        render_campaign_progress(
            mismatched, sealed_tasks=TASKS, qualification=_qualification()
        )

    tampered_cid = copy.deepcopy(baseline_snapshot)
    generation = tampered_cid["store_generation"]
    assert isinstance(generation, dict)
    generation["revision"] = 20
    with pytest.raises(CampaignProgressValidationError, match="snapshot CID"):
        render_campaign_progress(
            tampered_cid, sealed_tasks=TASKS, qualification=_qualification()
        )

    advanced = copy.deepcopy(tampered_cid)
    _seal_owner_snapshot(advanced)
    advanced_rendering = render_campaign_progress(
        advanced, sealed_tasks=TASKS, qualification=_qualification()
    )
    assert advanced_rendering.report["report_cid"] != baseline.report["report_cid"]
    assert advanced_rendering.report["source_snapshot"]["store_generation"]["revision"] == 20


def _rendering_with_blocker(blocker: str) -> object:
    qualification = dict(
        build_program_qualification_disposition(
            program_id="EAAEF",
            status="blocked",
            blockers=(blocker,),
            evidence_refs=("snapshot:current",),
        )
    )
    return render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=qualification
    )


def test_atomic_writer_publishes_immutable_pair_and_one_current_manifest(
    tmp_path: Path,
) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )
    manifest = write_campaign_progress_outputs(
        rendering,
        repository_root=tmp_path,
        current_manifest_destination="generated/progress-current.json",
    )

    manifest_path = tmp_path / "generated/progress-current.json"
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert observed_manifest == dict(manifest)
    assert manifest["schema"] == CAMPAIGN_PROGRESS_CURRENT_MANIFEST_SCHEMA
    assert manifest["report_cid"] == rendering.report["report_cid"]
    assert manifest["owner_identity"] == rendering.report["source_snapshot"][
        "owner_identity"
    ]
    assert manifest["store_generation"] == rendering.report["source_snapshot"][
        "store_generation"
    ]
    json_path = tmp_path / manifest["artifacts"]["json"]["path"]
    markdown_path = tmp_path / manifest["artifacts"]["markdown"]["path"]
    assert json_path.name == f"progress-{rendering.report['report_cid']}.json"
    assert markdown_path.name == f"progress-{rendering.report['report_cid']}.md"
    assert json_path.read_text(encoding="utf-8") == rendering.json_text
    assert markdown_path.read_text(encoding="utf-8") == rendering.markdown_text
    assert (
        manifest["artifacts"]["json"]["sha256"]
        == "sha256:" + hashlib.sha256(rendering.json_text.encode("utf-8")).hexdigest()
    )
    assert (
        manifest["artifacts"]["markdown"]["sha256"]
        == "sha256:" + hashlib.sha256(rendering.markdown_text.encode("utf-8")).hexdigest()
    )
    assert not (tmp_path / "generated/progress-current.md").exists()
    repeated = write_campaign_progress_outputs(
        rendering,
        repository_root=tmp_path,
        current_manifest_destination="generated/progress-current.json",
    )
    assert dict(repeated) == dict(manifest)


@pytest.mark.parametrize(
    "current_manifest_destination",
    (
        "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json",
        "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md",
        "docs/architecture/agent_supervisor_causal_event_federation.todo.md",
    ),
)
def test_atomic_writer_refuses_all_canonical_board_paths_before_writing(
    tmp_path: Path,
    current_manifest_destination: str,
) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )

    with pytest.raises(CampaignProgressDestinationError, match="protected"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination=current_manifest_destination,
        )

    assert not (tmp_path / "generated").exists()


def test_atomic_writer_refuses_destination_outside_explicit_root(
    tmp_path: Path,
) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )

    with pytest.raises(CampaignProgressDestinationError, match="inside"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination=tmp_path.parent / "outside.json",
        )


def test_atomic_writer_rejects_lexical_final_and_parent_symlinks(tmp_path: Path) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    target = tmp_path / "target.json"
    target.write_text("sentinel", encoding="utf-8")
    (generated / "progress-current.json").symlink_to(target)
    with pytest.raises(CampaignProgressDestinationError, match="symlink"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination="generated/progress-current.json",
        )
    assert target.read_text(encoding="utf-8") == "sentinel"

    symlink_root = tmp_path / "parent-symlink-root"
    symlink_root.mkdir()
    (tmp_path / "linked").symlink_to(symlink_root, target_is_directory=True)
    with pytest.raises(CampaignProgressDestinationError, match="symlink|non-directory"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination="linked/progress-current.json",
        )
    assert list(symlink_root.iterdir()) == []


def test_atomic_writer_rejects_symlink_at_derived_immutable_name(tmp_path: Path) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    target = tmp_path / "artifact-target.json"
    target.write_text("sentinel", encoding="utf-8")
    immutable_name = f"progress-{rendering.report['report_cid']}.json"
    (generated / immutable_name).symlink_to(target)
    with pytest.raises(CampaignProgressDestinationError, match="symlink"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination="generated/progress-current.json",
        )
    assert target.read_text(encoding="utf-8") == "sentinel"
    assert not (generated / "progress-current.json").exists()


def test_atomic_writer_bounds_existing_immutable_artifact_read(tmp_path: Path) -> None:
    rendering = render_campaign_progress(
        _owner_snapshot(), sealed_tasks=TASKS, qualification=_qualification()
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    immutable_name = f"progress-{rendering.report['report_cid']}.json"
    oversized = generated / immutable_name
    oversized.write_bytes(b"x" * (len(rendering.json_text.encode("utf-8")) + 1))

    with pytest.raises(CampaignProgressWriteError, match="size differs"):
        write_campaign_progress_outputs(
            rendering,
            repository_root=tmp_path,
            current_manifest_destination="generated/progress-current.json",
        )

    assert not (generated / "progress-current.json").exists()


def test_manifest_replace_failure_preserves_previous_coherent_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _rendering_with_blocker("first blocker")
    second = _rendering_with_blocker("second blocker")
    manifest_path = tmp_path / "generated/progress-current.json"
    write_campaign_progress_outputs(
        first,
        repository_root=tmp_path,
        current_manifest_destination=manifest_path,
    )
    previous = manifest_path.read_bytes()

    def _fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("injected manifest replacement failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(CampaignProgressWriteError, match="publication failed"):
        write_campaign_progress_outputs(
            second,
            repository_root=tmp_path,
            current_manifest_destination=manifest_path,
        )
    assert manifest_path.read_bytes() == previous
    manifest = json.loads(previous)
    for artifact in manifest["artifacts"].values():
        assert (tmp_path / artifact["path"]).is_file()


def test_concurrent_publishers_leave_one_complete_coherent_manifest(tmp_path: Path) -> None:
    renderings = (
        _rendering_with_blocker("concurrent blocker alpha"),
        _rendering_with_blocker("concurrent blocker beta"),
    )

    def _publish(index: int) -> dict[str, object]:
        return dict(
            write_campaign_progress_outputs(
                renderings[index % 2],
                repository_root=tmp_path,
                current_manifest_destination="generated/progress-current.json",
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        manifests = list(executor.map(_publish, range(24)))

    current = json.loads(
        (tmp_path / "generated/progress-current.json").read_text(encoding="utf-8")
    )
    assert current in manifests
    assert current["report_cid"] in {
        rendering.report["report_cid"] for rendering in renderings
    }
    for artifact in current["artifacts"].values():
        payload = (tmp_path / artifact["path"]).read_bytes()
        assert len(payload) == artifact["size_bytes"]
        assert "sha256:" + hashlib.sha256(payload).hexdigest() == artifact["sha256"]
