"""ASE3-000 current-main convergence and historical-state isolation tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.validation.prompt_v3_convergence import (
    ARTIFACT_FILENAMES,
    BOARD_NAMESPACE,
    DEFAULT_ARTIFACT_ROOT,
    MANIFEST_FILENAME,
    POST_WAVE3_RESIDUAL_FILENAME,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
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
    residual = _load(root / POST_WAVE3_RESIDUAL_FILENAME)
    residual_repository = residual["repository"]
    assert isinstance(residual_repository, dict)
    report_head = str(residual_repository["head"])
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
            report_head,
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
