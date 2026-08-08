"""ASE3-000 current-main convergence and historical-state isolation tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    prompt_v3_convergence as convergence_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.prompt_v3_convergence import (
    ARTIFACT_FILENAMES,
    BOARD_NAMESPACE,
    DEFAULT_ARTIFACT_ROOT,
    FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
    FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,
    FAILED_VALIDATION_EVENT_019_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
    FALSE_COMPLETION_RECOVERY_FILENAME,
    MANIFEST_FILENAME,
    MAX_EVIDENCE_SNAPSHOT_BYTES,
    OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
    POST_WAVE3_RESIDUAL_FILENAME,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
    SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
    CurrentMainBaseline,
    RescueDispositionReport,
    validate_convergence_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
)
TASKBOARD_PATH = REPO_ROOT / PROMPT_V3_TASKBOARD_RELATIVE_PATH
VALIDATOR_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "prompt_v3_convergence.py"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rebind_component_digest(root: Path, filename: str) -> None:
    manifest_path = root / MANIFEST_FILENAME
    manifest = _load(manifest_path)
    components = manifest["components"]
    assert isinstance(components, dict)
    components[filename] = "sha256:" + hashlib.sha256(
        (root / filename).read_bytes()
    ).hexdigest()
    _write(manifest_path, manifest)


def _recompute_event_id(event: dict[str, object]) -> str:
    body = dict(event)
    body.pop("event_id", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _portable_recovery_repository(
    tmp_path: Path,
    *,
    include_failed_candidate_parent: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    portable = tmp_path / "portable-repository"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO_ROOT), str(portable)],
        check=True,
        capture_output=True,
        text=True,
    )
    taskboard = portable / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    taskboard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASKBOARD_PATH, taskboard)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    failed = recovery["failed_attempt"]
    launch = incident["launch"]
    baseline = _load(root / "current_main_baseline.json")
    seed = baseline["integration_seed"]
    assert isinstance(failed, dict)
    assert isinstance(launch, dict)
    assert isinstance(seed, dict)
    command = [
        "git",
        "-c",
        "user.name=Portable Validation",
        "-c",
        "user.email=portable@example.invalid",
        "commit-tree",
        str(seed["tree"]),
        "-p",
        str(launch["launch_head"]),
    ]
    if include_failed_candidate_parent:
        command.extend(("-p", str(failed["implementation_commit"])))
    command.extend(("-m", "portable recovery descendant"))
    descendant = subprocess.run(
        command,
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "symbolic-ref", "HEAD", "refs/heads/portable-descendant"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "update-ref", "HEAD", descendant],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    return root, portable, taskboard


def test_checked_in_convergence_packet_is_valid_on_integration_checkout() -> None:
    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        repo_root=REPO_ROOT,
        check_repository=True,
    )

    assert report.valid is True, report.errors
    assert report.errors == ()
    assert set(report.checked_artifacts) == {*ARTIFACT_FILENAMES, MANIFEST_FILENAME}
    assert report.integration_seed_commit == "7d70a558e0f54a16a04b3a145fe3d43360cac4c5"


def test_rescue_population_is_complete_and_every_item_has_a_disposition() -> None:
    payload = _load(DEFAULT_ARTIFACT_ROOT / "rescue_artifact_dispositions.json")
    baseline = CurrentMainBaseline.from_dict(
        _load(DEFAULT_ARTIFACT_ROOT / "current_main_baseline.json")
    )
    report = RescueDispositionReport.from_dict(payload)

    assert report.validate(baseline) == ()
    assert len(report.commits) == 36
    assert len(report.files) == 35
    assert {item.disposition for item in (*report.commits, *report.files)} <= {
        "port",
        "rewrite",
        "superseded",
        "discard",
    }
    assert all(
        item.target_tasks
        for item in (*report.commits, *report.files)
        if item.disposition in {"port", "rewrite"}
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "false_completions.ASE3-006",
            "repair_task",
            "ASE3-027",
            "false_completions.ASE3-006",
        ),
        (
            "false_completions.ASE3-018",
            "repair_strict_shard",
            2,
            "false_completions.ASE3-018",
        ),
        (
            "failed_attempt",
            "merge_dispatched",
            True,
            "failed_attempt.merge_dispatched",
        ),
        (
            "disposition",
            "attempt_counter_mutation_authorized",
            True,
            "disposition.attempt_counter_mutation_authorized",
        ),
    ),
)
def test_false_completion_recovery_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_RECOVERY_FILENAME
    payload = _load(path)
    target: object = payload
    for component in section.split("."):
        assert isinstance(target, dict)
        target = target[component]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, FALSE_COMPLETION_RECOVERY_FILENAME)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("filename", "section", "field", "replacement", "error_fragment"),
    (
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            "",
            "task_id",
            "ASE3-018",
            "false_completion_merge_receipt.ASE3-006.task_id",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
            "merge_result.integration_commit_proof",
            "passed",
            False,
            "false_completion_merge_receipt.ASE3-018.integration_commit_proof.passed",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            "merge_result",
            "returncode",
            False,
            "false_completion_merge_receipt.ASE3-006.merge_result.returncode",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
            "merge_result.todo_update_result.protected_board_postcondition",
            "trusted",
            False,
            "false_completion_merge_receipt.ASE3-018."
            "protected_board_postcondition.trusted",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            "merge_result.todo_update_result.protected_board_postcondition."
            "release_proof",
            "clean",
            False,
            "false_completion_merge_receipt.ASE3-006."
            "protected_board_postcondition.release_proof.clean",
        ),
        (
            FAILED_VALIDATION_EVENT_019_FILENAME,
            "",
            "rescue_branch",
            "rescue/forged",
            "failed_validation_event.ASE3-019.event_id",
        ),
        (
            FAILED_VALIDATION_EVENT_019_FILENAME,
            "",
            "merge_dispatched",
            True,
            "failed_validation_event.ASE3-019.merge_dispatched",
        ),
    ),
)
def test_recovery_snapshot_tampering_fails_after_manifest_rebind(
    tmp_path: Path,
    filename: str,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / filename
    payload = _load(path)
    target: object = payload
    for component in filter(None, section.split(".")):
        assert isinstance(target, dict)
        target = target[component]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, filename)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "attempt_accounting",
            "attempt_restoration_authorized",
            True,
            "attempt_restoration_authorized",
        ),
        (
            "terminal_failure",
            "primary_provider_effect_dispatched",
            True,
            "primary_provider_effect_dispatched",
        ),
        (
            "terminal_failure",
            "implementation_runner_dispatched",
            False,
            "implementation_runner_dispatched",
        ),
        (
            "control_plane_provenance",
            "accepted_control_plane_required_for_salvage",
            False,
            "accepted_control_plane_required_for_salvage",
        ),
        (
            "operator_salvage_gate",
            "accepted_control_plane_required",
            False,
            "accepted_control_plane_required",
        ),
        (
            "operator_salvage_gate",
            "required_receipt_fields",
            [
                "schema",
                "created_at",
                "board_namespace",
                "task",
                "incident",
                "authority",
                "source_candidate",
                "salvage_base",
                "implementation",
                "merge",
                "validation",
                "review",
                "denials",
            ],
            "required_receipt_fields",
        ),
        (
            "task",
            "board_status",
            "completed",
            "task.board_status",
        ),
    ),
)
def test_attempt2_incident_tampering_fails_after_manifest_rebind(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    target = payload[section]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_attempt2_event_semantics_fail_even_after_identity_and_manifest_rebind(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    events = payload["events"]
    event_ids = payload["event_ids"]
    assert isinstance(events, dict)
    assert isinstance(event_ids, list)
    finished = events["implementation_finished"]
    assert isinstance(finished, dict)
    finished["provider_dispatched"] = False
    finished["event_id"] = _recompute_event_id(finished)
    event_ids[2] = finished["event_id"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "event_snapshot.provider_dispatched" in error for error in report.errors
    )


@pytest.mark.parametrize(
    ("event_name", "event_index", "field", "replacement", "error_fragment"),
    (
        (
            "prior_attempt_seeded",
            0,
            "applied",
            False,
            "events.prior_attempt_seeded.applied",
        ),
        (
            "implementation_started",
            1,
            "branch",
            "implementation/forged",
            "events.implementation_started.branch",
        ),
        (
            "implementation_shutdown_reconciled",
            3,
            "reconciled",
            False,
            "events.implementation_shutdown_reconciled.reconciled",
        ),
    ),
)
def test_attempt2_event_chain_semantics_fail_after_event_id_rebind(
    tmp_path: Path,
    event_name: str,
    event_index: int,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    events = payload["events"]
    event_ids = payload["event_ids"]
    assert isinstance(events, dict)
    assert isinstance(event_ids, list)
    event = events[event_name]
    assert isinstance(event, dict)
    event[field] = replacement
    event["event_id"] = _recompute_event_id(event)
    event_ids[event_index] = event["event_id"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_attempt2_event_bundle_order_is_exact_after_manifest_rebind(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    event_order = payload["event_order"]
    assert isinstance(event_order, list)
    event_order[0], event_order[1] = event_order[1], event_order[0]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("event_snapshot.event_order" in error for error in report.errors)


def test_attempt2_log_tampering_fails_after_manifest_rebind(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "agent implementation route binding fields are invalid",
            "forged terminal success",
            1,
        ),
        encoding="utf-8",
    )
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("log_snapshot" in error for error in report.errors)


def test_attempt2_log_uses_a_dedicated_eight_kibibyte_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME
    path.write_bytes(b"x" * (8 * 1024 + 1))

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("8192-byte evidence snapshot bound" in error for error in report.errors)


def test_recovery_snapshot_symlink_is_rejected_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    path.unlink()
    path.symlink_to(
        DEFAULT_ARTIFACT_ROOT / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    )

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(path.name in error for error in report.errors)


def test_evidence_snapshot_hardlink_is_rejected_before_parsing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    backing = root / "hardlink-backing.json"
    shutil.copy2(path, backing)
    path.unlink()
    os.link(backing, path)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("single-link evidence file" in error for error in report.errors)


def test_evidence_snapshot_size_bound_fails_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    path.write_bytes(b" " * (MAX_EVIDENCE_SNAPSHOT_BYTES + 1))

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("evidence snapshot bound" in error for error in report.errors)


def test_evidence_snapshot_descriptor_instability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    real_fstat = os.fstat
    fstat_calls = 0

    def unstable_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        observed = real_fstat(descriptor)
        if fstat_calls != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(convergence_module.os, "fstat", unstable_fstat)
    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("changed during bounded read" in error for error in report.errors)


def test_recovery_shard_fields_name_the_repair_and_retry_tasks() -> None:
    recovery = _load(DEFAULT_ARTIFACT_ROOT / FALSE_COMPLETION_RECOVERY_FILENAME)
    completions = recovery["false_completions"]
    failed = recovery["failed_attempt"]
    assert isinstance(completions, dict)
    assert isinstance(failed, dict)
    assert completions["ASE3-006"]["repair_strict_shard"] == 2
    assert completions["ASE3-018"]["repair_strict_shard"] == 0
    assert failed["retry_strict_shard"] == 1
    assert all("strict_shard" not in item for item in completions.values())
    assert "strict_shard" not in failed


def test_component_tampering_fails_closed_before_repository_checks(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    baseline_path = root / "current_main_baseline.json"
    baseline = _load(baseline_path)
    original = baseline["original_checkout"]
    assert isinstance(original, dict)
    original["dirty_entry_count"] = 0
    _write(baseline_path, baseline)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("dirty_entry_count" in error for error in report.errors)
    assert any("digest mismatch" in error for error in report.errors)


def test_rebound_historical_state_still_cannot_claim_v3_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "historical_state_contradictions.json"
    payload = _load(path)
    payload["authority"] = "completion-authority"
    payload["v3_completion_credit"] = True
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("authority: must be evidence-only" in error for error in report.errors)
    assert any("v3_completion_credit: must be false" in error for error in report.errors)


def test_rebound_post_wave3_residual_mapping_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / POST_WAVE3_RESIDUAL_FILENAME
    payload = _load(path)
    residuals = payload["residuals"]
    assert isinstance(residuals, list)
    record = next(
        item
        for item in residuals
        if isinstance(item, dict)
        and item.get("gap_id") == "trusted-context-canonical-composition"
    )
    record["target_task"] = "ASE3-019"
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(
        "trusted-context-canonical-composition.target_task: expected ASE3-018"
        in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "provider_incident",
            "attempt_consumed",
            True,
            "provider_incident.attempt_consumed: expected False",
        ),
        (
            "provider_incident",
            "fallback_dispatched",
            True,
            "provider_incident.fallback_dispatched: expected False",
        ),
        (
            "disposition",
            "completion_authority",
            True,
            "disposition.completion_authority: expected False",
        ),
        (
            "disposition",
            "gate_task",
            "ASE3-009",
            "disposition.gate_task: expected 'ASE3-008'",
        ),
    ),
)
def test_rebound_post_wave3_authority_and_provider_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / POST_WAVE3_RESIDUAL_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "authorization_source",
            "source_head",
            "0" * 40,
            "authorization_source.source_head: expected",
        ),
        (
            "authorization_source",
            "prospective_only",
            False,
            "authorization_source.prospective_only: expected True",
        ),
        (
            "route",
            "route_id",
            "global-ambient-route",
            (
                "route.route_id: expected 'agent-supervisor-prompt-v3-grok45-"
                "terra56-high-auth-or-hard-quota-v1'"
            ),
        ),
        (
            "route",
            "fallback_reasoning_effort",
            "medium",
            "route.fallback_reasoning_effort: expected 'high'",
        ),
        (
            "route",
            "allowed_trigger_classes",
            ["grok_authentication_unavailable", "rate_limit"],
            "route.allowed_trigger_classes: expected",
        ),
        (
            "ownership_contract",
            "canonical_route_plan_owner",
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
            (
                "ownership_contract.canonical_route_plan_owner: expected "
                "'ipfs_accelerate_py.llm_router'"
            ),
        ),
        (
            "ownership_contract",
            "typed_fallback_decision_owner",
            "implementation_daemon",
            (
                "ownership_contract.typed_fallback_decision_owner: expected "
                "'ipfs_accelerate_py.llm_router'"
            ),
        ),
        (
            "ownership_contract",
            "route_plan_and_decision_exports_required_before_bootstrap_dispatch",
            False,
            (
                "route_plan_and_decision_exports_required_before_bootstrap_dispatch: "
                "expected True"
            ),
        ),
        (
            "ownership_contract",
            "route_authority_binding_fields",
            ["board_namespace", "authorization_artifact_sha256"],
            "ownership_contract.route_authority_binding_fields: expected",
        ),
        (
            "ownership_contract",
            "verified_authority_binding_must_reach_terminal_outcome_and_daemon_accounting",
            False,
            (
                "verified_authority_binding_must_reach_terminal_outcome_and_daemon_accounting: "
                "expected True"
            ),
        ),
        (
            "ownership_contract",
            "ambient_six_field_route_profile_alone_authorizes_fallback",
            True,
            (
                "ambient_six_field_route_profile_alone_authorizes_fallback: expected "
                "False"
            ),
        ),
        (
            "ownership_contract",
            "runner_role",
            "route_policy_and_failure_classifier",
            (
                "ownership_contract.runner_role: expected "
                "'isolation_process_effect_and_terminal_outcome_emitter'"
            ),
        ),
        (
            "ownership_contract",
            "daemon_role",
            "provider_failure_reclassification",
            "ownership_contract.daemon_role: expected 'task_retry_accounting_only'",
        ),
        (
            "ownership_contract",
            "scheduler_role",
            "route_policy_owner",
            "ownership_contract.scheduler_role: expected 'route_profile_input_only'",
        ),
        (
            "ownership_contract",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed",
            True,
            (
                "duplicate_route_policy_or_failure_classification_outside_router_allowed: "
                "expected False"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "fallback_dispatch_scope",
            "once_per_host_forever",
            (
                "bootstrap_route_guarantees.fallback_dispatch_scope: expected "
                "'once_per_runner_same_daemon_attempt'"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "direct_auth_signal_allowlist",
            ["not signed in", "not authenticated", "forbidden"],
            "bootstrap_route_guarantees.direct_auth_signal_allowlist: expected",
        ),
        (
            "bootstrap_route_guarantees",
            "ambiguous_direct_auth_signals_denied",
            ["401", "403"],
            "ambiguous_direct_auth_signals_denied: expected",
        ),
        (
            "bootstrap_route_guarantees",
            "ambiguous_signal_may_continue_only_as_independently_confirmed_hard_quota",
            False,
            (
                "ambiguous_signal_may_continue_only_as_independently_confirmed_hard_quota: "
                "expected True"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "hard_quota_independent_confirmation_required",
            False,
            "hard_quota_independent_confirmation_required: expected True",
        ),
        (
            "bootstrap_route_guarantees",
            "explicit_codex_review_conflict_denied",
            False,
            "explicit_codex_review_conflict_denied: expected True",
        ),
        (
            "bootstrap_route_guarantees",
            "durable_cross_process_restart_reservation_present",
            True,
            "durable_cross_process_restart_reservation_present: expected False",
        ),
        (
            "bootstrap_route_guarantees",
            "full_signed_field_equality_present",
            True,
            "full_signed_field_equality_present: expected False",
        ),
        (
            "ase3_019_completion_requirements",
            "durable_cross_process_restart_once_only_cas_required",
            False,
            "durable_cross_process_restart_once_only_cas_required: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "auth_signal_policy_expansion_requires_signed_typed_policy",
            False,
            "auth_signal_policy_expansion_requires_signed_typed_policy: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "canonical_route_plan_and_typed_decision_must_remain_router_owned",
            False,
            (
                "canonical_route_plan_and_typed_decision_must_remain_router_owned: "
                "expected True"
            ),
        ),
        (
            "ase3_019_completion_requirements",
            "provider_capacity_attempt_restoration_must_remain_denied",
            False,
            (
                "provider_capacity_attempt_restoration_must_remain_denied: expected "
                "True"
            ),
        ),
        (
            "ase3_019_completion_requirements",
            "signed_reviewer_identity_and_provider_required",
            False,
            "signed_reviewer_identity_and_provider_required: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "fallback_implementer_and_reviewer_must_differ",
            False,
            "fallback_implementer_and_reviewer_must_differ: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "signed_equality_fields",
            ["invocation", "task", "prompt", "scope", "budget", "authority"],
            "ase3_019_completion_requirements.signed_equality_fields: expected",
        ),
        (
            "external_docker_boundary",
            "image_id",
            "sha256:" + "0" * 64,
            "external_docker_boundary.image_id: expected",
        ),
        (
            "external_docker_boundary",
            "workspace_is_only_writable_bind_mount",
            False,
            "workspace_is_only_writable_bind_mount: expected True",
        ),
        (
            "denials",
            "arbitrary_error_fallback_allowed",
            True,
            "denials.arbitrary_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "rate_limit_fallback_allowed",
            True,
            "denials.rate_limit_fallback_allowed: expected False",
        ),
        (
            "denials",
            "transport_error_fallback_allowed",
            True,
            "denials.transport_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "invalid_request_fallback_allowed",
            True,
            "denials.invalid_request_fallback_allowed: expected False",
        ),
        (
            "denials",
            "unknown_error_fallback_allowed",
            True,
            "denials.unknown_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "post_effect_fallback_allowed",
            True,
            "denials.post_effect_fallback_allowed: expected False",
        ),
        (
            "denials",
            "workspace_changed_before_fallback_allowed",
            True,
            "workspace_changed_before_fallback_allowed: expected False",
        ),
        (
            "denials",
            "attempt_counter_mutation_authorized",
            True,
            "attempt_counter_mutation_authorized: expected False",
        ),
        (
            "denials",
            "provider_capacity_attempt_restoration_allowed",
            True,
            "provider_capacity_attempt_restoration_allowed: expected False",
        ),
        (
            "denials",
            "legacy_objective_refill_authorized",
            True,
            "legacy_objective_refill_authorized: expected False",
        ),
        (
            "denials",
            "legacy_codebase_refill_authorized",
            True,
            "legacy_codebase_refill_authorized: expected False",
        ),
        (
            "historical_evidence",
            "post_wave3_residual_report_is_immutable",
            False,
            "post_wave3_residual_report_is_immutable: expected True",
        ),
    ),
)
def test_rebound_provider_fallback_authorization_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_ase3_019_cannot_downgrade_terra_reasoning_or_auth_fallback(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "exactly one concurrent or restarted worker automatically admits a "
        "matching pre-effect Codex `gpt-5.6-terra` fallback at `high` reasoning"
    )
    replacement = (
        "exactly one concurrent or restarted worker requires reauthentication "
        "before a Codex `gpt-5.6-terra` fallback at `medium` reasoning"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.acceptance: exact automatic "
        "auth/quota fallback contract required"
    ) in report.errors


def test_ase3_019_cannot_move_route_policy_outside_llm_router(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "Export an immutable canonical implementation route plan and typed "
        "fallback decision from `ipfs_accelerate_py.llm_router` as the sole "
        "provider-policy source"
    )
    replacement = (
        "Let the runner and daemon independently choose implementation routes "
        "and fallback decisions"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.effects: exact automatic "
        "auth/quota fallback contract required"
    ) in report.errors


def test_ase3_019_must_name_llm_router_and_its_dedicated_route_test(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "- Outputs: ipfs_accelerate_py/llm_router.py, "
    replacement = "- Outputs: "
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.outputs: exact "
        "llm_router-owned route surface required"
    ) in report.errors


@pytest.mark.parametrize(
    ("needle", "replacement", "error_fragment"),
    (
        (
            "- Repairs task: ASE3-006\n",
            "- Repairs task: ASE3-018\n",
            "ASE3-023.repairs_task",
        ),
        (
            "- Is schedulable: true\n- Review only: false\n- Priority: P0\n"
            "- Track: ambient-inference-production-repair\n",
            "- Is schedulable: false\n- Review only: false\n- Priority: P0\n"
            "- Track: ambient-inference-production-repair\n",
            "ASE3-027.is_schedulable",
        ),
        (
            "- Depends on: ASE3-006, ASE3-018, ASE3-019, ASE3-023, ASE3-027\n",
            "- Depends on: ASE3-006, ASE3-018, ASE3-019\n",
            "ASE3-022.depends_on",
        ),
        (
            "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
            "and once-only fallback\n",
            "## ASE3-019 Changed identity\n",
            "provider_fallback_task_contract.ASE3-019.title",
        ),
        (
            "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
            "and once-only fallback\n\n- Status: todo\n",
            "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
            "and once-only fallback\n\n- Status: completed\n",
            "provider_fallback_task_contract.ASE3-019.contract_sha256",
        ),
        (
            "Configured-board production launch consumes the compiled active plan",
            "Configured-board production launch may ignore the compiled active plan",
            "false_completion_repair_tasks.ASE3-023.contract_sha256",
        ),
        (
            "call the existing canonical target, state/run, profile, objective/task-source",
            "optionally bypass the canonical target, state/run, profile, objective/task-source",
            "false_completion_repair_tasks.ASE3-027.contract_sha256",
        ),
    ),
)
def test_false_completion_repair_task_contract_fails_closed(
    tmp_path: Path,
    needle: str,
    replacement: str,
    error_fragment: str,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_reload_gate_rejects_a_removed_blocked_reason(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "- Blocked reason: provider-attempt daemon reload boundary not yet accepted\n"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, "", 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "ASE3-022.blocked_reason: expected 'provider-attempt daemon reload "
        "boundary not yet accepted'" in error
        for error in report.errors
    )


def test_reload_gate_rejects_a_removed_ase3_021_dependency(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019, ASE3-022\n"
    replacement = "- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019\n"
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert "provider_attempt_reload_gate.ASE3-021.depends_on: missing ASE3-022" in (
        report.errors
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    (
        (
            "goal",
            "- Goal id: ASE3-G055\n- Outputs: ",
            "ASE3-022.goal_id: must be absent",
        ),
        (
            "outputs",
            "- Outputs: data/forged-reload-receipt.json",
            "ASE3-022.outputs: expected only",
        ),
        (
            "predicted",
            "- Predicted files: data/forged-reload-receipt.json",
            "ASE3-022.predicted_files: expected only",
        ),
    ),
)
def test_reload_gate_rejects_goal_enrollment_and_receipt_redirects(
    tmp_path: Path,
    field: str,
    replacement: str,
    error_fragment: str,
) -> None:
    taskboard_path = tmp_path / f"prompt-v3-{field}.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    if field == "goal":
        needle = f"- Outputs: {PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        replacement += PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    elif field == "outputs":
        needle = f"- Outputs: {PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
    else:
        needle = (
            "- Predicted files: "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_reload_gate_completion_requires_future_receipt_authority(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "## ASE3-022 Accept the provider-attempt daemon reload boundary\n\n"
        "- Status: blocked\n"
    )
    replacement = (
        "## ASE3-022 Accept the provider-attempt daemon reload boundary\n\n"
        "- Status: completed\n"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "ASE3-022.status: completion requires a strict reload receipt validator "
        "and convergence-manifest binding" in error
        for error in report.errors
    )


def test_reload_receipt_path_is_reserved_until_strictly_validated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    receipt.symlink_to(tmp_path / "missing-reload-receipt-target.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "receipt: present without a strict validator and convergence-manifest binding"
        in error
        for error in report.errors
    )


@pytest.mark.parametrize("receipt_kind", ("regular", "dangling-symlink"))
def test_operator_salvage_receipt_path_is_reserved_during_c1(
    tmp_path: Path,
    receipt_kind: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / OPERATOR_SALVAGE_RECEIPT_019_FILENAME
    if receipt_kind == "regular":
        receipt.write_text("{}\n", encoding="utf-8")
    else:
        receipt.symlink_to(tmp_path / "missing-salvage-receipt-target.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        OPERATOR_SALVAGE_RECEIPT_019_FILENAME in error
        and "present without a strict validator" in error
        for error in report.errors
    )


def test_reload_gate_c1_operator_salvage_contract_is_exact(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "mandatory accepted-control-plane provenance"
    assert text.count(needle) == 1
    taskboard_path.write_text(
        text.replace(needle, "optional ambient control-plane provenance", 1),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "ASE3-022.contract_sha256" in error for error in report.errors
    )


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "current_main_baseline.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            '  "board_namespace":',
            '  "schema": "duplicate-must-fail",\n  "board_namespace":',
            1,
        ),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("duplicate JSON key: schema" in error for error in report.errors)


def test_rebound_recorded_tree_must_match_the_git_object(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "current_main_baseline.json"
    payload = _load(path)
    upstream = payload["upstream_main"]
    assert isinstance(upstream, dict)
    upstream["tree"] = "0" * 40
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        repo_root=REPO_ROOT,
        check_repository=True,
    )

    assert report.valid is False
    assert "repository_binding.upstream_main.tree: Git identity mismatch" in report.errors


@pytest.mark.parametrize(
    ("field_path", "value", "error_fragment"),
    (
        (("goal_id",), "ASE3-G999", "goal_id: expected ASE3-G010"),
        (("created_at",), "not-a-timestamp", "created_at: expected UTC timestamp"),
        (
            ("integration_seed_commit",),
            "0" * 40,
            "integration_seed_commit: baseline mismatch",
        ),
        (
            ("integration_seed_tree",),
            "0" * 40,
            "integration_seed_tree: baseline mismatch",
        ),
        (
            ("population", "rescue_commits"),
            35,
            "population.rescue_commits: expected 36",
        ),
        (
            ("population", "rescue_changed_paths"),
            34,
            "population.rescue_changed_paths: expected 35",
        ),
        (("population", "v2_tasks"), 7, "population.v2_tasks: expected 8"),
        (
            ("population", "historical_contradictions"),
            4,
            "population.historical_contradictions: expected 5",
        ),
        (
            ("population", "v3_seed_tasks"),
            14,
            "population.v3_seed_tasks: expected 15",
        ),
        (
            ("population", "v3_seed_goals"),
            8,
            "population.v3_seed_goals: expected 9",
        ),
        (
            ("completion_rules", "historical_status_or_receipt_satisfies_v3"),
            True,
            "historical_status_or_receipt_satisfies_v3: expected False",
        ),
        (
            ("completion_rules", "branch_local_commit_satisfies_v3"),
            True,
            "branch_local_commit_satisfies_v3: expected False",
        ),
        (
            ("completion_rules", "queue_drain_satisfies_goal_completion"),
            True,
            "queue_drain_satisfies_goal_completion: expected False",
        ),
        (
            ("completion_rules", "current_tree_acceptance_required"),
            False,
            "current_tree_acceptance_required: expected True",
        ),
        (
            ("completion_rules", "forced_residual_scan_required"),
            False,
            "forced_residual_scan_required: expected True",
        ),
        (
            ("downstream_rules", "required_ancestor"),
            "0" * 40,
            "downstream_rules.required_ancestor: expected",
        ),
        (
            ("downstream_rules", "merge_target_branch"),
            "other",
            "downstream_rules.merge_target_branch: expected",
        ),
        (
            ("downstream_rules", "rescue_disposition_required_before_use"),
            False,
            "rescue_disposition_required_before_use: expected True",
        ),
        (
            ("downstream_rules", "fresh_validation_receipt_required_per_task"),
            False,
            "fresh_validation_receipt_required_per_task: expected True",
        ),
        (
            ("downstream_rules", "protected_source_checkout_may_be_modified"),
            True,
            "protected_source_checkout_may_be_modified: expected False",
        ),
    ),
)
def test_rebound_manifest_fields_fail_closed(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / MANIFEST_FILENAME
    payload = _load(path)
    block: dict[str, object] = payload
    for field in field_path[:-1]:
        child = block[field]
        assert isinstance(child, dict)
        block = child
    block[field_path[-1]] = value
    _write(path, payload)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "extra_key"),
    (
        ("population", "unreviewed_count"),
        ("completion_rules", "soft_completion_allowed"),
        ("downstream_rules", "unreviewed_effect_allowed"),
    ),
)
def test_manifest_policy_and_count_objects_reject_extra_keys(
    tmp_path: Path,
    section: str,
    extra_key: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / MANIFEST_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[extra_key] = True
    _write(path, payload)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(f"convergence_manifest.{section}: population mismatch" in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "worktree",
            "isolated_from_source_checkout",
            False,
            "isolated_from_source_checkout: must be true",
        ),
        ("worktree", "branch", "other", "worktree.branch: must equal"),
        (
            "protected_source_checkout",
            "modified_by_bootstrap",
            True,
            "modified_by_bootstrap: must be false",
        ),
        (
            "state_namespace",
            "fresh_for_board",
            False,
            "fresh_for_board: must be true",
        ),
        (
            "state_namespace",
            "historical_import_allowed",
            True,
            "historical_import_allowed: must be false",
        ),
        (
            "downstream_binding",
            "changed_revision_requires_fresh_validation",
            False,
            "changed_revision_requires_fresh_validation: must be true",
        ),
    ),
)
def test_rebound_critical_worktree_receipt_fields_fail_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "clean_integration_worktree_receipt.json"
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_rebound_rescue_disposition_rejects_unknown_target_task(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    files = payload["files"]
    assert isinstance(files, list)
    first_rewrite = next(
        item
        for item in files
        if isinstance(item, dict) and item.get("disposition") == "rewrite"
    )
    first_rewrite["target_tasks"] = ["ASE3-999"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("target_tasks: unknown task 'ASE3-999'" in error for error in report.errors)


@pytest.mark.parametrize("field", ("merge_base", "rescue_head", "current_seed"))
def test_rebound_rescue_top_level_identities_match_the_baseline(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    payload[field] = "0" * 40
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(
        f"rescue_artifact_dispositions.{field}: baseline mismatch" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("population", "mutation", "expected_fragment"),
    (
        ("commits", "replace-with-garbage", "commits[0]: expected object"),
        ("files", "replace-with-garbage", "files[0]: expected object"),
        ("commits", "append-extra-object", "commits: expected 36, got 37"),
        ("files", "append-extra-object", "files: expected 35, got 36"),
    ),
)
def test_rescue_populations_reject_non_objects_and_extra_elements(
    tmp_path: Path,
    population: str,
    mutation: str,
    expected_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    entries = payload[population]
    assert isinstance(entries, list)
    if mutation == "replace-with-garbage":
        entries[0] = "not-an-object"
    else:
        first = entries[0]
        assert isinstance(first, dict)
        entries.append(dict(first))
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(expected_fragment in error for error in report.errors)


def test_repository_validation_is_portable_to_an_alternate_descendant_worktree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)

    baseline_path = root / "current_main_baseline.json"
    baseline = _load(baseline_path)
    original = baseline["original_checkout"]
    seed = baseline["integration_seed"]
    assert isinstance(original, dict)
    assert isinstance(seed, dict)
    original["path"] = "/historical/source/checkout"
    _write(baseline_path, baseline)
    _rebind_component_digest(root, baseline_path.name)

    receipt_path = root / "clean_integration_worktree_receipt.json"
    receipt = _load(receipt_path)
    source = receipt["protected_source_checkout"]
    worktree = receipt["worktree"]
    assert isinstance(source, dict)
    assert isinstance(worktree, dict)
    source["path"] = original["path"]
    worktree["path"] = "/historical/integration/worktree"
    _write(receipt_path, receipt)
    _rebind_component_digest(root, receipt_path.name)

    portable = tmp_path / "portable-repository"
    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            str(REPO_ROOT),
            str(portable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    portable_taskboard_path = portable / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    portable_taskboard_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASKBOARD_PATH, portable_taskboard_path)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    launch = incident["launch"]
    assert isinstance(launch, dict)
    seed_tree = str(seed["tree"])
    descendant = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Portable Validation",
            "-c",
            "user.email=portable@example.invalid",
            "commit-tree",
                seed_tree,
                "-p",
                str(launch["launch_head"]),
            "-m",
            "portable descendant",
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "symbolic-ref", "HEAD", "refs/heads/portable-descendant"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "update-ref", "HEAD", descendant],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )

    assert subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "portable-descendant"
    assert Path(str(worktree["path"])).resolve() != portable.resolve()
    assert not Path(str(source["path"])).exists()

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=portable_taskboard_path,
    )

    assert report.valid is True, report.errors
    assert report.errors == ()


def test_recovery_requires_the_failed_candidate_rescue_ref(tmp_path: Path) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    failed = recovery["failed_attempt"]
    assert isinstance(failed, dict)
    rescue_branch = str(failed["rescue_branch"])
    for reference in (
        f"refs/heads/{rescue_branch}",
        f"refs/remotes/origin/{rescue_branch}",
    ):
        subprocess.run(
            ["git", "update-ref", "-d", reference],
            cwd=portable,
            check=True,
            capture_output=True,
            text=True,
        )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("ASE3-019.rescue_branch" in error for error in report.errors)


def test_recovery_requires_the_exact_attempt2_branch_ref(tmp_path: Path) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    prior_seed = incident["prior_attempt_seed"]
    assert isinstance(prior_seed, dict)
    branch = str(prior_seed["attempt_2_branch"])
    for reference in (
        f"refs/heads/{branch}",
        f"refs/remotes/origin/{branch}",
    ):
        subprocess.run(
            ["git", "update-ref", "-d", reference],
            cwd=portable,
            check=True,
            capture_output=True,
            text=True,
        )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("attempt_2_branch: exact ref unavailable" in error for error in report.errors)


def test_recovery_rejects_conflicting_exact_named_rescue_refs(
    tmp_path: Path,
) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    source = recovery["source"]
    failed = recovery["failed_attempt"]
    assert isinstance(source, dict)
    assert isinstance(failed, dict)
    rescue_branch = str(failed["rescue_branch"])
    subprocess.run(
        [
            "git",
            "update-ref",
            f"refs/remotes/origin/{rescue_branch}",
            str(failed["implementation_commit"]),
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "update-ref",
            f"refs/heads/{rescue_branch}",
            str(source["recovery_parent_head"]),
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("exact named refs disagree" in error for error in report.errors)


def test_recovery_rejects_a_head_containing_the_failed_candidate(
    tmp_path: Path,
) -> None:
    root, portable, taskboard = _portable_recovery_repository(
        tmp_path,
        include_failed_candidate_parent=True,
    )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        "ASE3-019.merge_dispatched: candidate is an ancestor of HEAD" in error
        for error in report.errors
    )


def test_scheduler_config_loads_and_binds_the_v3_board_structurally() -> None:
    board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)

    assert board.board_namespace == BOARD_NAMESPACE
    assert board.task_prefix == "ASE3-"
    assert board.max_lanes == 3
    assert board.strict_task_sharding is True
    assert board.merge_target_branch == "agent/prompt-self-improvement-v3"
    assert board.validator_path.endswith("prompt_v3_convergence.py")
    for filename in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME):
        relative = (
            "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
            + filename
        )
        assert relative in board.protected_paths


def test_check_all_cli_emits_the_sealed_preflight_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.validation.prompt_v3_convergence",
            "--check-all",
            "--repo-root",
            str(REPO_ROOT),
            "--artifacts-root",
            str(DEFAULT_ARTIFACT_ROOT),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_check_all_direct_file_entrypoint_matches_scheduler_execution() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--check-all",
            "--repo-root",
            str(REPO_ROOT),
            "--artifacts-root",
            str(DEFAULT_ARTIFACT_ROOT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert payload["valid"] is True
    assert payload["errors"] == []
