"""Fail-closed tests for the operator-owned AAE supervisor controls."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control import profile_authority

from scripts import validate_adversarial_assurance_engine_board as validator
from scripts.ops.agent_supervisor import (
    adversarial_assurance_engine_scheduler as scheduler,
)

CONTROL_PATHS = (
    validator.PLAN_REL,
    validator.OBJECTIVES_REL,
    validator.TODO_REL,
    validator.SCHEDULER_REL,
    validator.PREREQUISITES_REL,
    validator.LAUNCHER_REL,
)


def _copy_controls(tmp_path: Path) -> Path:
    for relative in CONTROL_PATHS:
        source = validator.REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path


def _rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _json_mutation(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_current_controls_are_valid_and_exactly_projected() -> None:
    report = validator.validate()
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 64
    assert report["goal_count"] == 10
    assert report["initial_completed_task_ids"] == ["AAE-000"]
    assert report["initial_ready_task_ids"] == [
        "AAE-001",
        "AAE-002",
        "AAE-003",
        "AAE-004",
    ]
    assert report["initial_blocked_task_ids"] == ["AAE-006"]
    assert report["terminal_task_id"] == "AAE-063"


def test_blocked_prerequisite_is_truthful_but_not_release_authority() -> None:
    board = validator.validate(check_repository=False)
    release = validator.validate_prerequisites(check_repository=False)
    assert board["valid"] is True
    assert board["operator_gate"] == {
        "task_id": "AAE-006",
        "receipt_status": "blocked",
        "release_valid": False,
    }
    assert release["valid"] is False
    assert release["runtime_and_sealing_authorized"] is False


def test_monotonic_completed_progress_remains_valid(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(
        root / validator.TODO_REL,
        "## AAE-001 Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces\n\n- Status: todo",
        "## AAE-001 Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces\n\n- Status: completed",
    )
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is True, report["errors"]


def test_out_of_order_completion_fails_closed(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(
        root / validator.TODO_REL,
        "## AAE-005 Reconcile authority matrix, manifests, blind spots, and focused baselines\n\n- Status: todo",
        "## AAE-005 Reconcile authority matrix, manifests, blind spots, and focused baselines\n\n- Status: completed",
    )
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any("AAE-005 is completed before dependencies" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("relative", "old", "new", "expected"),
    (
        (
            validator.TODO_REL,
            "## AAE-063 Publish trust model",
            "## AAE-064 Publish trust model",
            "task IDs/order differ",
        ),
        (
            validator.TODO_REL,
            "- Depends on: AAE-056, AAE-057, AAE-058, AAE-061, AAE-062\n- Goal id: AAE-G090",
            "- Depends on: AAE-056\n- Goal id: AAE-G090",
            "dependencies differ",
        ),
        (
            validator.TODO_REL,
            "- Status: blocked\n- Blocked reason: SCG and IncrementalProofSealer",
            "- Status: todo\n- Blocked reason: SCG and IncrementalProofSealer",
            "AAE-006 status",
        ),
        (
            validator.TODO_REL,
            "- Is schedulable: false\n- Review only: false\n- Priority: P0\n- Track: prerequisite-release",
            "- Is schedulable: true\n- Review only: false\n- Priority: P0\n- Track: prerequisite-release",
            "AAE-006 schedulability differs",
        ),
        (
            validator.PLAN_REL,
            "The system used semantically targeted counterfactual mutations",
            "The system guessed",
            "prescribed bounded final claim",
        ),
    ),
)
def test_markdown_control_mutations_fail_closed(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(root / relative, old, new)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"]), report["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda payload: payload["protected_paths"].pop(),
            "protected_paths differ",
        ),
        (
            lambda payload: payload["source_binding"].update(
                {"ipfs_datasets_planning_revision": "0" * 40}
            ),
            "source_binding differs",
        ),
        (
            lambda payload: payload["bootstrap_source_binding"].update(
                {"ipfs_datasets_planning_revision": "1" * 40}
            ),
            "immutable bootstrap_source_binding differs",
        ),
        (
            lambda payload: payload["source_binding"].update(
                {"pin_generation": 1}
            ),
            "blocked scheduler source_binding differs from bootstrap pins",
        ),
        (
            lambda payload: payload["authority_policy"].update(
                {"mutation_score_proves_correctness": True}
            ),
            "authority doctrine differs",
        ),
        (
            lambda payload: payload["lanes"][0]["initial_task_ids"].append("AAE-001"),
            "wrong strict shard",
        ),
    ),
)
def test_scheduler_mutations_fail_closed(tmp_path: Path, mutation, expected: str) -> None:
    root = _copy_controls(tmp_path)
    _json_mutation(root / validator.SCHEDULER_REL, mutation)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"]), report["errors"]


def test_forged_completed_prerequisite_fails_closed(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["status"] = "completed"
        payload["runtime_and_sealing_authorized"] = True

    _json_mutation(root / validator.PREREQUISITES_REL, forge)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(
        "completed prerequisite receipt" in error for error in report["errors"]
    ), report["errors"]


def test_plausible_forged_completed_gate_lacks_bound_evidence_and_signature(
    tmp_path: Path,
) -> None:
    root = _copy_controls(tmp_path)
    todo_path = root / validator.TODO_REL
    todo_text = todo_path.read_text(encoding="utf-8")
    for index in range(1, 7):
        task_id = f"AAE-{index:03d}"
        todo_text, replacement_count = re.subn(
            rf"(## {task_id} [^\n]+\n\n- Status: )(?:todo|blocked)",
            r"\g<1>completed",
            todo_text,
            count=1,
        )
        assert replacement_count == 1
    todo_path.write_text(todo_text, encoding="utf-8")

    release_commit = "a" * 40
    fake_cid = "b" + "a" * 58
    scheduler_path = root / validator.SCHEDULER_REL

    def release_pins(payload: dict[str, object]) -> None:
        source = payload["source_binding"]
        assert isinstance(source, dict)
        source.update({"pin_state": "operator_released", "pin_generation": 1})

    _json_mutation(scheduler_path, release_pins)
    scheduler_payload = json.loads(scheduler_path.read_text(encoding="utf-8"))
    source = scheduler_payload["source_binding"]
    release_gitlinks = {
        "ipfs_datasets_py": source["ipfs_datasets_planning_revision"],
        "ipfs_kit_py": source["ipfs_kit_planning_revision"],
        "ipfs_accelerate_py/mcplusplus": source[
            "mcp_plus_plus_planning_revision"
        ],
    }
    evidence_names = (
        "scg_lifecycle",
        "scg_terminal",
        "incremental_proof_sealer_release",
        "datasets_baseline",
        "accelerate_baseline",
        "ipfs_kit_py_baseline",
        "mcp_plus_plus_baseline",
    )

    def forge_plausible_release(payload: dict[str, object]) -> None:
        payload.update(
            {
                "status": "completed",
                "controller": {
                    "repository": "endomorphosis/ipfs_accelerate_py",
                    "branch": validator.BRANCH,
                    "required_ancestor": validator.BASE_REVISION,
                    "pin_generation": 1,
                    "release_commit": release_commit,
                    "release_gitlinks": release_gitlinks,
                },
                "semantic_compression_governor": {
                    "observed_commit": release_commit,
                    "terminal_launch_commit": release_commit,
                    "observed_datasets_commit": release_gitlinks[
                        "ipfs_datasets_py"
                    ],
                    "terminal_receipt_valid": True,
                    "disposition": "terminal_completed",
                },
                "incremental_proof_sealer": {
                    "observed_commit": release_commit,
                    "terminal_receipt_valid": True,
                    "disposition": "released",
                    "api_bindings": {
                        name: "ipfs_accelerate_py.agent_supervisor.fake"
                        for name in (
                            "IncrementalProofSealer",
                            "FullCheckpointSeal",
                            "DeltaSeal",
                            "create_full_checkpoint",
                            "publish_full_checkpoint",
                            "build_delta_seal",
                            "publish_delta_seal",
                        )
                    },
                },
                "evidence_artifacts": {
                    name: {
                        "path": (
                            "docs/architecture/adversarial_assurance_inventory/"
                            f"prerequisite_evidence/{name}.json"
                        ),
                        "canonical_identity": fake_cid,
                    }
                    for name in evidence_names
                },
                "worker_may_complete": False,
                "runtime_and_sealing_authorized": True,
                "canonical_identity_scope": (
                    "entire completed receipt excluding canonical_identity and "
                    "authorization"
                ),
                "canonical_identity": fake_cid,
                "authorization": {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "adversarial-assurance-prerequisite-authorization@1"
                    ),
                    "identity_did": validator.OPERATOR_AUTHORITY_DID,
                    "audience": validator.BOARD_NAMESPACE,
                    "action": "complete:AAE-006",
                    "receipt_cid": fake_cid,
                    "pin_generation": 1,
                    "release_commit": release_commit,
                    "release_gitlinks": release_gitlinks,
                    "signature": "zPlausibleButForged",
                },
                "provenance": "operator release evidence",
            }
        )

    _json_mutation(root / validator.PREREQUISITES_REL, forge_plausible_release)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(
        "completed prerequisite receipt does not satisfy release validation" in error
        for error in report["errors"]
    ), report["errors"]
    release = validator.validate_prerequisites(root, check_repository=False)
    assert release["valid"] is False
    assert any(
        "evidence artifact is invalid" in error for error in release["errors"]
    ), release["errors"]
    assert any(
        "canonical identity differs" in error for error in release["errors"]
    ), release["errors"]
    assert any(
        "signature verification failed" in error for error in release["errors"]
    ), release["errors"]


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resign_completed_gate(bundle: SimpleNamespace) -> None:
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    unsigned.pop("canonical_identity", None)
    unsigned.pop("authorization", None)
    receipt_cid = bundle.cid_for_obj(unsigned)
    controller = receipt["controller"]
    authorization = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "adversarial-assurance-prerequisite-authorization@1"
        ),
        "identity_did": bundle.identity_did,
        "audience": validator.BOARD_NAMESPACE,
        "action": "complete:AAE-006",
        "receipt_cid": receipt_cid,
        "pin_generation": controller["pin_generation"],
        "release_commit": controller["release_commit"],
        "release_gitlinks": controller["release_gitlinks"],
    }
    authorization["signature"] = base64.b64encode(
        bundle.private_key.sign(_canonical_json_bytes(authorization))
    ).decode("ascii")
    receipt["canonical_identity"] = receipt_cid
    receipt["authorization"] = authorization
    _write_json(bundle.receipt_path, receipt)


def _rebind_evidence(bundle: SimpleNamespace, name: str) -> None:
    payload = json.loads(bundle.evidence_paths[name].read_text(encoding="utf-8"))
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    receipt["evidence_artifacts"][name]["canonical_identity"] = (
        bundle.cid_for_obj(payload)
    )
    _write_json(bundle.receipt_path, receipt)
    _resign_completed_gate(bundle)


@pytest.fixture
def valid_completed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    root = _copy_controls(tmp_path)
    todo_path = root / validator.TODO_REL
    todo_text = todo_path.read_text(encoding="utf-8")
    for index in range(1, 7):
        task_id = f"AAE-{index:03d}"
        todo_text, replacement_count = re.subn(
            rf"(## {task_id} [^\n]+\n\n- Status: )(?:todo|blocked)",
            r"\g<1>completed",
            todo_text,
            count=1,
        )
        assert replacement_count == 1
    todo_path.write_text(todo_text, encoding="utf-8")

    private_key = Ed25519PrivateKey.generate()
    identity_did = profile_authority.ed25519_did_key(private_key.public_key())
    monkeypatch.setattr(validator, "OPERATOR_AUTHORITY_DID", identity_did)

    scheduler_path = root / validator.SCHEDULER_REL
    scheduler_payload = json.loads(scheduler_path.read_text(encoding="utf-8"))
    source = scheduler_payload["source_binding"]
    source.update(
        {
            "pin_state": "operator_released",
            "pin_generation": 7,
            "ipfs_datasets_planning_revision": "b" * 40,
            "ipfs_kit_planning_revision": "c" * 40,
            "mcp_plus_plus_planning_revision": "d" * 40,
        }
    )
    scheduler_payload["prerequisite_gate"]["operator_authority_did"] = identity_did
    _write_json(scheduler_path, scheduler_payload)

    # Keep the fixture small while exercising the production CID and signature
    # implementations. The validator insists their module paths belong to the
    # supplied repository, so only those two authority imports are rebound.
    cid_errors: list[str] = []
    validator._canonical_cid(
        validator.REPO_ROOT,
        {"fixture": "authority-load"},
        noun="fixture authority",
        errors=cid_errors,
    )
    assert cid_errors == []
    content_authority = validator.importlib.import_module(
        "ipfs_datasets_py.logic.software_contracts.content"
    )
    cid_for_obj = content_authority.cid_for_obj
    actual_import_module = validator.importlib.import_module
    (root / "ipfs_datasets_py").mkdir()

    def fixture_authority_import(name: str, package: str | None = None):
        if name == "ipfs_datasets_py.logic.software_contracts.content":
            return SimpleNamespace(
                __file__=str(root / "ipfs_datasets_py/logic/software_contracts/content.py"),
                cid_for_obj=cid_for_obj,
            )
        if name == "ipfs_accelerate_py.agent_supervisor.control.profile_authority":
            return SimpleNamespace(
                __file__=str(
                    root
                    / "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py"
                ),
                verify_did_key_signature=(
                    profile_authority.verify_did_key_signature
                ),
            )
        return actual_import_module(name, package)

    monkeypatch.setattr(validator.importlib, "import_module", fixture_authority_import)

    required_api_symbols = {
        "IncrementalProofSealer",
        "FullCheckpointSeal",
        "DeltaSeal",
        "create_full_checkpoint",
        "publish_full_checkpoint",
        "build_delta_seal",
        "publish_delta_seal",
    }
    api_probe_calls: list[dict[str, object]] = []

    def probe_unreleased_current_tree_apis(
        repo_root: Path,
        bindings: object,
        errors: list[str],
    ) -> None:
        assert repo_root == root.resolve()
        assert isinstance(bindings, dict)
        assert set(bindings) == required_api_symbols
        assert all(
            str(module).startswith("ipfs_accelerate_py.agent_supervisor.")
            for module in bindings.values()
        )
        api_probe_calls.append(dict(bindings))

    monkeypatch.setattr(
        validator,
        "_probe_sealer_api_bindings",
        probe_unreleased_current_tree_apis,
    )

    release_commit = "a" * 40
    scg_launch_commit = "e" * 40
    sealer_commit = "f" * 40
    release_gitlinks = {
        "ipfs_datasets_py": source["ipfs_datasets_planning_revision"],
        "ipfs_kit_py": source["ipfs_kit_planning_revision"],
        "ipfs_accelerate_py/mcplusplus": source[
            "mcp_plus_plus_planning_revision"
        ],
    }
    configuration_root = cid_for_obj({"scg": "configuration"})
    lifecycle = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-compression-governor-lifecycle@1"
        ),
        "plan": {
            "source_head": scg_launch_commit,
            "expected_task_count": 49,
            "configuration_root": configuration_root,
        },
        "profile": {
            "run_id": "scg-release-run",
            "profile_id": "scg-release-profile",
            "configuration_root": configuration_root,
        },
    }
    terminal = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-compression-governor-terminal@1"
        ),
        "run_id": "scg-release-run",
        "profile_id": "scg-release-profile",
        "configuration_root": configuration_root,
        "expected_task_count": 49,
        "drained": True,
        "lane_evidence": [
            {
                "lane": lane,
                "terminal": True,
                "completed_count": 49,
                "blocked_count": 0,
                "ready_count": 0,
                "waiting_count": 0,
                "active_task_id": None,
            }
            for lane in range(3)
        ],
    }
    sealer_release = {
        "schema_version": "incremental-proof-sealer-release-validation@2",
        "runner_id": "protected-board-release-validation-runner@1",
        "terminal_gate": {
            "id": "terminal-board-gate",
            "capture_status": "completed",
            "exit_code": 0,
        },
        "validation_worktree_parent_revision": release_commit,
        "source_revisions": {
            "accelerate": sealer_commit,
            "datasets": release_gitlinks["ipfs_datasets_py"],
            "kit": release_gitlinks["ipfs_kit_py"],
        },
    }
    sealer_release["receipt_digest"] = "sha256:" + hashlib.sha256(
        _canonical_json_bytes(sealer_release)
    ).hexdigest()

    evidence_payloads: dict[str, dict[str, object]] = {
        "scg_lifecycle": lifecycle,
        "scg_terminal": terminal,
        "incremental_proof_sealer_release": sealer_release,
    }
    baseline_states = {
        "datasets": release_gitlinks["ipfs_datasets_py"],
        "accelerate": release_commit,
        "ipfs_kit_py": release_gitlinks["ipfs_kit_py"],
        "mcp_plus_plus": release_gitlinks["ipfs_accelerate_py/mcplusplus"],
    }
    for name, state_root in baseline_states.items():
        evidence_payloads[f"{name}_baseline"] = {
            "schema": (
                "ipfs_accelerate_py/adversarial-assurance/"
                "focused-baseline-receipt@1"
            ),
            "runner_id": "protected-aae-focused-baseline-runner@1",
            "repository": name,
            "repository_state_root": state_root,
            "started_at": "2026-08-13T12:00:00Z",
            "finished_at": "2026-08-13T12:00:01Z",
            "duration_ns": 1_000_000_000,
            "command_argv": ["python3", "-m", "pytest", f"test/{name}"],
            "returncode": 0,
            "terminal_status": "passed",
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "environment_identity": cid_for_obj({"environment": name}),
            "dependency_lock_identity": cid_for_obj({"dependencies": name}),
            "bounded_log_digest": "sha256:" + "1" * 64,
            "network_access": "disabled",
            "production_credentials_available": False,
        }

    evidence_paths: dict[str, Path] = {}
    evidence_bindings: dict[str, dict[str, str]] = {}
    for name, payload in evidence_payloads.items():
        relative = (
            validator.SEALER_RELEASE_RECEIPT_REL
            if name == "incremental_proof_sealer_release"
            else f"{validator.PREREQUISITE_EVIDENCE_PREFIX}/{name}.json"
        )
        path = root / relative
        _write_json(path, payload)
        evidence_paths[name] = path
        evidence_bindings[name] = {
            "path": relative,
            "canonical_identity": cid_for_obj(payload),
        }

    receipt_path = root / validator.PREREQUISITES_REL
    receipt = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "adversarial-assurance-prerequisite-receipt@1"
        ),
        "task_id": "AAE-006",
        "status": "completed",
        "observed_at": "2026-08-13T12:00:02Z",
        "controller": {
            "repository": "endomorphosis/ipfs_accelerate_py",
            "branch": validator.BRANCH,
            "required_ancestor": validator.BASE_REVISION,
            "pin_generation": 7,
            "release_commit": release_commit,
            "release_gitlinks": release_gitlinks,
        },
        "semantic_compression_governor": {
            "observed_commit": release_commit,
            "terminal_launch_commit": scg_launch_commit,
            "observed_datasets_commit": release_gitlinks["ipfs_datasets_py"],
            "disposition": "terminal_completed",
        },
        "incremental_proof_sealer": {
            "observed_commit": sealer_commit,
            "disposition": "released",
            "api_bindings": {
                symbol: "ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer"
                for symbol in required_api_symbols
            },
            "qualification_argv": [
                "python3",
                "scripts/validate_incremental_proof_sealer_board.py",
                "--run-release-validation",
            ],
        },
        "evidence_artifacts": evidence_bindings,
        "completion_requirements": [
            "SCG lifecycle is terminal and bound",
            "full-checkpoint and delta APIs are released",
            "clean recursive repository forest is pinned",
            "focused baselines are closed receipts",
            "operator review authorizes the exact evidence",
        ],
        "worker_may_complete": False,
        "runtime_and_sealing_authorized": True,
        "canonical_identity_scope": (
            "entire completed receipt excluding canonical_identity and authorization"
        ),
        "canonical_identity": None,
        "authorization": {},
        "provenance": "operator-approved release evidence",
        "baseline_qualification_argv": list(validator.FOCUSED_BASELINE_RUNNER),
    }
    _write_json(receipt_path, receipt)
    bundle = SimpleNamespace(
        root=root,
        receipt_path=receipt_path,
        evidence_paths=evidence_paths,
        cid_for_obj=cid_for_obj,
        private_key=private_key,
        identity_did=identity_did,
        api_probe_calls=api_probe_calls,
    )
    _resign_completed_gate(bundle)
    return bundle


def test_valid_completed_gate_is_relaunchable_and_cryptographically_bound(
    valid_completed_gate: SimpleNamespace,
) -> None:
    bundle = valid_completed_gate
    release = validator.validate_prerequisites(
        bundle.root,
        check_repository=False,
    )
    assert release["valid"] is True, release["errors"]
    board = validator.validate(bundle.root, check_repository=False)
    assert board["valid"] is True, board["errors"]
    assert release["runtime_and_sealing_authorized"] is True
    assert len(bundle.api_probe_calls) == 2


def test_completed_gate_rejects_altered_evidence_cid(
    valid_completed_gate: SimpleNamespace,
) -> None:
    bundle = valid_completed_gate
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    receipt["evidence_artifacts"]["scg_terminal"]["canonical_identity"] = (
        bundle.cid_for_obj({"not": "the terminal receipt"})
    )
    _write_json(bundle.receipt_path, receipt)
    _resign_completed_gate(bundle)
    release = validator.validate_prerequisites(bundle.root, check_repository=False)
    assert release["valid"] is False
    assert "scg_terminal evidence canonical identity differs" in release["errors"]


def test_completed_gate_rejects_invalid_operator_signature(
    valid_completed_gate: SimpleNamespace,
) -> None:
    bundle = valid_completed_gate
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    receipt["authorization"]["signature"] = base64.b64encode(b"0" * 64).decode(
        "ascii"
    )
    _write_json(bundle.receipt_path, receipt)
    release = validator.validate_prerequisites(bundle.root, check_repository=False)
    assert release["valid"] is False
    assert any(
        "operator release signature verification failed" in error
        for error in release["errors"]
    )


def test_completed_gate_rejects_controller_binding(
    valid_completed_gate: SimpleNamespace,
) -> None:
    bundle = valid_completed_gate
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    receipt["controller"]["repository"] = "attacker/fork"
    _write_json(bundle.receipt_path, receipt)
    _resign_completed_gate(bundle)
    release = validator.validate_prerequisites(bundle.root, check_repository=False)
    assert release["valid"] is False
    assert "prerequisite controller binding differs from active release pins" in (
        release["errors"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda payload: payload.update({"returncode": 1}),
            "accelerate baseline is not terminal passed",
        ),
        (
            lambda payload: payload.update(
                {"output_receipt_cid": "b" + "a" * 58}
            ),
            "accelerate baseline fields differ",
        ),
    ),
)
def test_completed_gate_rejects_baseline_field_or_output_binding(
    valid_completed_gate: SimpleNamespace,
    mutation,
    expected: str,
) -> None:
    bundle = valid_completed_gate
    path = bundle.evidence_paths["accelerate_baseline"]
    baseline = json.loads(path.read_text(encoding="utf-8"))
    mutation(baseline)
    _write_json(path, baseline)
    _rebind_evidence(bundle, "accelerate_baseline")
    release = validator.validate_prerequisites(bundle.root, check_repository=False)
    assert release["valid"] is False
    assert expected in release["errors"]


def test_completed_gate_rejects_proof_sealer_receipt_digest(
    valid_completed_gate: SimpleNamespace,
) -> None:
    bundle = valid_completed_gate
    path = bundle.evidence_paths["incremental_proof_sealer_release"]
    sealer_release = json.loads(path.read_text(encoding="utf-8"))
    sealer_release["receipt_digest"] = "sha256:" + "0" * 64
    _write_json(path, sealer_release)
    _rebind_evidence(bundle, "incremental_proof_sealer_release")
    release = validator.validate_prerequisites(bundle.root, check_repository=False)
    assert release["valid"] is False
    assert "IncrementalProofSealer release digest is invalid" in release["errors"]


def test_wrapper_preflight_checks_the_third_mcplusplus_gitlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    source = board.payload["source_binding"]
    nested_checks: set[str] = set()

    def completed(argv, *, stdout: str = "", returncode: int = 0):
        return scheduler.subprocess.CompletedProcess(
            argv,
            returncode,
            stdout=stdout,
            stderr="",
        )

    def fake_git(_board, *argv: str, timeout: float = 30.0):
        del timeout
        if argv[:2] == ("symbolic-ref", "--quiet"):
            return completed(argv, stdout=validator.BRANCH + "\n")
        if argv[:2] == ("rev-parse", "--verify"):
            return completed(argv, stdout="a" * 40 + "\n")
        if argv[:2] == ("ls-files", "-s"):
            relative = argv[-1]
            revision_field = {
                "ipfs_datasets_py": "ipfs_datasets_planning_revision",
                "ipfs_kit_py": "ipfs_kit_planning_revision",
                "ipfs_accelerate_py/mcplusplus": (
                    "mcp_plus_plus_planning_revision"
                ),
            }[relative]
            return completed(
                argv,
                stdout=f"160000 {source[revision_field]} 0\t{relative}\n",
            )
        return completed(argv)

    def fake_nested_run(argv, **_kwargs):
        nested = Path(argv[2])
        relative = nested.relative_to(board.repo_root).as_posix()
        nested_checks.add(relative)
        if argv[3:5] == ["rev-parse", "HEAD"]:
            revision_field = {
                "ipfs_datasets_py": "ipfs_datasets_planning_revision",
                "ipfs_kit_py": "ipfs_kit_planning_revision",
                "ipfs_accelerate_py/mcplusplus": (
                    "mcp_plus_plus_planning_revision"
                ),
            }[relative]
            observed = str(source[revision_field])
            if relative == "ipfs_accelerate_py/mcplusplus":
                observed = "0" * 40
            return completed(argv, stdout=observed + "\n")
        return completed(argv)

    monkeypatch.setattr(scheduler, "_git", fake_git)
    monkeypatch.setattr(scheduler.subprocess, "run", fake_nested_run)
    monkeypatch.setattr(
        scheduler,
        "_validator_report",
        lambda _board: {"valid": True, "task_count": 64},
    )
    monkeypatch.setattr(
        scheduler,
        "_validate_launch_plan",
        lambda _board, _plan: {"lane_count": 2},
    )
    monkeypatch.setattr(
        scheduler,
        "_collision_check",
        lambda _board, _plan: {"safe": True, "live_targets": []},
    )
    monkeypatch.setattr(
        scheduler,
        "_provider_preflight",
        lambda _board: {"grok_ready": True, "codex_ready": True},
    )

    report = scheduler.preflight(board)
    assert nested_checks == {
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py/mcplusplus",
    }
    assert report["checks"]["gitlinks_exact_and_clean"] is False
    assert report["checks"]["launch_admission"] is True
    assert report["launch_admission"]["required"] is False
    assert any("gitlinks are not exact" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("blocked_count", "blocked_task_ids", "expected_gate", "expected_blockers"),
    (
        (1, ["AAE-006"], True, []),
        (1, ["AAE-007"], False, ["blocked_tasks_present:1"]),
        (2, ["AAE-006", "AAE-007"], False, ["blocked_tasks_present:2"]),
        (1, ["AAE-006", "AAE-007"], False, ["blocked_tasks_present:1"]),
    ),
)
def test_wrapper_suppresses_only_the_exact_operator_gate_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_count: int,
    blocked_task_ids: list[str],
    expected_gate: bool,
    expected_blockers: list[str],
) -> None:
    board = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    status_path = tmp_path / "aae_lane_0_supervisor_status.json"
    task_path = tmp_path / "aae_lane_0_task_state.json"
    status_path.write_text("{}\n", encoding="utf-8")
    task_path.write_text("{}\n", encoding="utf-8")
    task_payload = {
        "task_count": 64,
        "completed_count": 1,
        "ready_count": 4,
        "selectable_ready_count": 2,
        "eligible_ready_count": 2,
        "external_reserved_count": 0,
        "waiting_count": 58,
        "blocked_count": blocked_count,
        "blocked_task_ids": blocked_task_ids,
        "active_task_id": "AAE-001",
        "implementation_in_progress": True,
        "last_progress_at": "2026-08-13T00:00:00Z",
        "selection_idle_reason": "",
    }

    monkeypatch.setattr(
        scheduler,
        "_read_runtime_json",
        lambda path: (
            task_payload if Path(path) == task_path else {"status": "running"}
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_file_age_seconds",
        lambda _path, *, now: 0.0,
    )
    adapter = SimpleNamespace(
        snapshot=lambda _profile: SimpleNamespace(members=(), roots=())
    )
    track = SimpleNamespace(name="aae-lane-0", supervisor_status_path=status_path)

    projection = scheduler._read_lane_projection(
        board,
        track,
        SimpleNamespace(profile_id="test-profile"),
        adapter=adapter,
        launched_at=0.0,
        expected_task_count=64,
        now=1.0,
    )
    assert projection["expected_operator_gate_blocked"] is expected_gate
    assert projection["blockers"] == expected_blockers


def _signed_launch_admission_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    board = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    private_key = Ed25519PrivateKey.generate()
    identity_did = profile_authority.ed25519_did_key(private_key.public_key())
    gate = board.payload["prerequisite_gate"]
    assert isinstance(gate, dict)
    gate["operator_authority_did"] = identity_did
    source_head = "a" * 40
    receipt_cid = scheduler._aae_cid(
        board,
        {"fixture": "prerequisite-receipt"},
        noun="test prerequisite receipt",
    )
    gitlinks = {
        "ipfs_datasets_py": "b" * 40,
        "ipfs_kit_py": "c" * 40,
        "ipfs_accelerate_py/mcplusplus": "d" * 40,
    }
    ledger_path = tmp_path / "git-common" / "launch-admissions.jsonl"
    lock_path = tmp_path / "git-common" / "launch-admissions.lock"
    admission_path = tmp_path / "external-launch-admission.json"

    monkeypatch.setattr(
        scheduler,
        "_aae_prerequisite_binding",
        lambda _board: {
            "required": True,
            "status": "completed",
            "prerequisite_receipt_cid": receipt_cid,
            "pin_generation": 1,
        },
    )
    monkeypatch.setattr(scheduler, "_source_head", lambda _board: source_head)
    monkeypatch.setattr(scheduler, "_aae_exact_gitlinks", lambda _board: gitlinks)
    def ledger_paths(_board, *, create: bool):
        if create:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
        return ledger_path, lock_path

    monkeypatch.setattr(scheduler, "_aae_ledger_paths", ledger_paths)
    monkeypatch.setenv(scheduler._AAE_ADMISSION_ENV, str(admission_path))

    unsigned = {
        "schema": scheduler._AAE_ADMISSION_SCHEMA,
        "identity_did": identity_did,
        "audience": validator.BOARD_NAMESPACE,
        "action": "launch:adversarial-assurance-engine-v1",
        "source_head": source_head,
        "prerequisite_receipt_cid": receipt_cid,
        "pin_generation": 1,
        "gitlinks": gitlinks,
        "launch_generation": 1,
        "previous_ledger_cid": None,
    }

    def write_signed(**changes: object) -> dict[str, object]:
        payload = {**unsigned, **changes}
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        payload["signature"] = base64.b64encode(
            private_key.sign(canonical)
        ).decode("ascii")
        admission_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload

    write_signed()
    return SimpleNamespace(
        board=board,
        ledger_path=ledger_path,
        lock_path=lock_path,
        admission_path=admission_path,
        source_head=source_head,
        receipt_cid=receipt_cid,
        gitlinks=gitlinks,
        write_signed=write_signed,
    )


def test_blocked_gate_requires_no_launch_admission_or_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    monkeypatch.delenv(scheduler._AAE_ADMISSION_ENV, raising=False)
    monkeypatch.setattr(
        scheduler,
        "_aae_prerequisite_binding",
        lambda _board: {"required": False, "status": "blocked"},
    )
    monkeypatch.setattr(
        scheduler,
        "_aae_ledger_paths",
        lambda *_args, **_kwargs: pytest.fail("blocked gate touched ledger"),
    )

    assert scheduler._aae_preflight_admission(board) == {
        "required": False,
        "valid": True,
        "status": "blocked",
    }
    assert scheduler._aae_consume_launch_admission(
        board,
        {"source_head": "unused"},
    )["consumed"] is False


def test_completed_gate_preflight_does_not_consume_but_launch_is_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)

    first = scheduler._aae_preflight_admission(fixture.board)
    second = scheduler._aae_preflight_admission(fixture.board)
    assert first["launch_generation"] == second["launch_generation"] == 1
    assert not fixture.ledger_path.exists()

    consumed = scheduler._aae_consume_launch_admission(
        fixture.board,
        {"source_head": fixture.source_head},
    )
    assert consumed["consumed"] is True
    assert consumed["launch_generation"] == 1
    entries = scheduler._aae_read_ledger(fixture.board, fixture.ledger_path)
    assert len(entries) == 1
    assert entries[0]["previous_ledger_cid"] is None
    assert entries[0]["entry_cid"] == consumed["entry_cid"]

    with pytest.raises(
        scheduler.AAESchedulerError,
        match="generation or previous ledger CID differs",
    ):
        scheduler._aae_consume_launch_admission(
            fixture.board,
            {"source_head": fixture.source_head},
        )


def test_changed_gate_requires_strictly_higher_pin_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)
    consumed = scheduler._aae_consume_launch_admission(
        fixture.board,
        {"source_head": fixture.source_head},
    )
    changed_receipt_cid = scheduler._aae_cid(
        fixture.board,
        {"fixture": "changed-prerequisite-receipt"},
        noun="changed test prerequisite receipt",
    )
    monkeypatch.setattr(
        scheduler,
        "_aae_prerequisite_binding",
        lambda _board: {
            "required": True,
            "status": "completed",
            "prerequisite_receipt_cid": changed_receipt_cid,
            "pin_generation": 1,
        },
    )
    fixture.write_signed(
        prerequisite_receipt_cid=changed_receipt_cid,
        launch_generation=2,
        previous_ledger_cid=consumed["entry_cid"],
    )

    with pytest.raises(
        scheduler.AAESchedulerError,
        match="changed gate or gitlinks require a higher pin generation",
    ):
        scheduler._aae_preflight_admission(fixture.board)


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"source_head": "e" * 40}, "source_head differs"),
        (
            {"prerequisite_receipt_cid": "b" + "a" * 58},
            "prerequisite_receipt_cid differs",
        ),
        ({"pin_generation": 2}, "pin_generation differs"),
        (
            {
                "gitlinks": {
                    "ipfs_datasets_py": "e" * 40,
                    "ipfs_kit_py": "c" * 40,
                    "ipfs_accelerate_py/mcplusplus": "d" * 40,
                }
            },
            "gitlinks differs",
        ),
        ({"previous_ledger_cid": "b" + "a" * 58}, "previous ledger CID differs"),
        ({"launch_generation": 2}, "generation or previous ledger CID differs"),
    ),
)
def test_completed_gate_admission_rejects_wrong_binding_or_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    expected: str,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)
    fixture.write_signed(**changes)

    with pytest.raises(scheduler.AAESchedulerError, match=expected):
        scheduler._aae_preflight_admission(fixture.board)


def test_completed_gate_rejects_malformed_existing_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)
    fixture.ledger_path.parent.mkdir(parents=True)
    fixture.ledger_path.write_text('{"schema":"truncated"}', encoding="utf-8")

    with pytest.raises(scheduler.AAESchedulerError, match="partial final entry"):
        scheduler._aae_preflight_admission(fixture.board)


def test_completed_gate_rejects_invalid_admission_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture.admission_path.read_text(encoding="utf-8"))
    payload["signature"] = base64.b64encode(b"not-an-ed25519-signature").decode(
        "ascii"
    )
    fixture.admission_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(scheduler.AAESchedulerError, match="signature verification"):
        scheduler._aae_preflight_admission(fixture.board)


def test_consumed_ledger_revalidates_historical_admission_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _signed_launch_admission_fixture(tmp_path, monkeypatch)
    scheduler._aae_consume_launch_admission(
        fixture.board,
        {"source_head": fixture.source_head},
    )
    entry = json.loads(fixture.ledger_path.read_text(encoding="utf-8"))
    entry["admission"]["signature"] = base64.b64encode(b"forged").decode(
        "ascii"
    )
    unsigned_entry = dict(entry)
    unsigned_entry.pop("entry_cid")
    entry["entry_cid"] = scheduler._aae_cid(
        fixture.board,
        unsigned_entry,
        noun="test forged ledger entry",
    )
    fixture.ledger_path.write_text(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(scheduler.AAESchedulerError, match="signature verification"):
        scheduler._aae_read_ledger(fixture.board, fixture.ledger_path)


def test_launch_admission_must_be_external_and_ledger_is_not_lifecycle_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    admission = repository / "admission.json"
    admission.write_text("{}\n", encoding="utf-8")
    board = replace(
        scheduler.load_board(
            validator.SCHEDULER_REL,
            repo_root=validator.REPO_ROOT,
        ),
        repo_root=repository,
    )
    monkeypatch.setenv(scheduler._AAE_ADMISSION_ENV, str(admission))
    with pytest.raises(scheduler.AAESchedulerError, match="external"):
        scheduler._aae_external_admission_path(board)

    actual = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    ledger_path, _lock_path = scheduler._aae_ledger_paths(actual, create=False)
    assert ledger_path.is_relative_to(scheduler._aae_git_common_dir(actual))
    assert not ledger_path.is_relative_to(actual.state_dir)


def test_run_launch_consumes_under_lifecycle_lock_before_process_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scheduler.load_board(
        validator.SCHEDULER_REL,
        repo_root=validator.REPO_ROOT,
    )
    board = replace(
        original,
        repo_root=tmp_path,
        config_path=tmp_path / "scheduler.json",
    )
    monkeypatch.setattr(scheduler, "_source_head", lambda _board: "a" * 40)
    plan = scheduler.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="aae-admission-lock-test",
        expected_task_count=64,
    )
    monkeypatch.setattr(
        scheduler,
        "_validate_launch_plan",
        lambda _board, _plan: {"lane_count": 2},
    )
    monkeypatch.setattr(
        scheduler,
        "_collision_check",
        lambda _board, _plan, *, adapter=None: {
            "safe": True,
            "live_targets": [],
        },
    )
    for name in plan["environment"]:
        monkeypatch.delenv(name, raising=False)
    events: list[str] = []

    def consume(_board, _plan):
        events.append("consume")
        competing = scheduler.os.open(
            board.lifecycle_lock_path,
            scheduler.os.O_RDWR,
        )
        try:
            with pytest.raises(BlockingIOError):
                scheduler.fcntl.flock(
                    competing,
                    scheduler.fcntl.LOCK_EX | scheduler.fcntl.LOCK_NB,
                )
        finally:
            scheduler.os.close(competing)

    monkeypatch.setattr(scheduler, "_aae_consume_launch_admission", consume)

    class Adapter:
        def snapshot(self, _profile):
            return SimpleNamespace(members=(), roots=())

        def launch(self, _profile, *, fencing_epoch: int):
            assert fencing_epoch == 0
            events.append("launch")
            return SimpleNamespace(to_dict=lambda: {})

    assert scheduler.run_launch(board, plan, adapter=Adapter()) == 0
    assert events == ["consume", "launch"]
