"""FVT-083 successor release-evidence fan-in and publication fencing."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor import release_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)
INTEGRATION_BRANCH = "agent/software-verification-prover-matrix"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "fvt083_successor_release_builder_test",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _event(
    *,
    sequence: int,
    previous_event_id: str,
    event_type: str,
    task_bound: bool,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": sequence,
        "previous_event_id": previous_event_id,
        "stream_id": "event-log:sha256:" + ("a" * 64),
        "snapshot_id": "event-log-snapshot:sha256:" + ("b" * 64),
        "type": event_type,
        "timestamp": f"2026-08-03T01:14:{sequence:02d}+00:00",
        **extra,
    }
    if task_bound:
        payload.update(
            {
                "task_id": release_evidence.TRUSTED_SUCCESSOR_TASK_ID,
                "canonical_task_cid": (
                    release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_CID
                ),
                "canonical_task_key": (
                    release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_KEY
                ),
            }
        )
    payload["event_id"] = release_evidence.content_digest(payload)
    return payload


def _authority_gates(node_id: str) -> list[dict[str, Any]]:
    return [
        {
            "depends_on": [node_id],
            "disposition": "pending",
            "gate": gate,
            "reason": "validation_passed_requires_independent_authority",
        }
        for gate in ("completion", "freshness", "merge", "proof", "semantic")
    ]


def _prepare_repository(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", INTEGRATION_BRANCH)
    _git(repo, "config", "user.name", "FVT-083 Test")
    _git(repo, "config", "user.email", "fvt083@example.invalid")
    exporter = repo / release_evidence.RELEASE_EVIDENCE_EXPORTER_RELATIVE
    exporter.parent.mkdir(parents=True)
    shutil.copy2(Path(release_evidence.__file__).resolve(), exporter)
    (repo / "implementation.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "implementation/fvt-083")
    (repo / "implementation.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "implementation.txt")
    _git(repo, "commit", "-m", "implement FVT-083")
    implementation = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", INTEGRATION_BRANCH)
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/fvt-083")
    merge = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", baseline)
    return {
        "repo": repo,
        "baseline": baseline,
        "implementation": implementation,
        "merge": merge,
    }


def _write_sources(
    setup: dict[str, Any],
    *,
    wrong_receipt_identity: bool = False,
    broken_chain: bool = False,
    wrong_tree_binding: bool = False,
    wrong_target_branch: bool = False,
) -> dict[str, Path]:
    repo = setup["repo"]
    state_dir = repo.parent / "supervisor-state"
    state_dir.mkdir(exist_ok=True)
    baseline = setup["baseline"]
    implementation = setup["implementation"]
    merge = setup["merge"]
    receipt_cid = (
        "baguqeera-wrong"
        if wrong_receipt_identity
        else release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_CID
    )
    receipt = {
        "schema": release_evidence.MEMBER_COMPLETION_RECEIPT_SCHEMA,
        "status": "succeeded",
        "task_id": release_evidence.TRUSTED_SUCCESSOR_TASK_ID,
        "canonical_task_cid": receipt_cid,
        "canonical_task_key": (
            release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_KEY
        ),
    }
    node_id = "1" * 64
    proposal_receipt_id = "2" * 64
    gates = _authority_gates(node_id)
    target_branch = (
        "attacker/untrusted"
        if wrong_target_branch
        else INTEGRATION_BRANCH
    )
    claimed_implementation = baseline if wrong_tree_binding else implementation

    events: list[dict[str, Any]] = []
    events.append(
        _event(
            sequence=1,
            previous_event_id="",
            event_type="merge_finished",
            task_bound=False,
            attempted=True,
            merged=True,
            returncode=0,
            merge_commit=merge,
            target_branch=target_branch,
        )
    )
    events.append(
        _event(
            sequence=2,
            previous_event_id=events[-1]["event_id"],
            event_type="todo_status_updated",
            task_bound=True,
            completion_receipts=[receipt],
        )
    )
    events.append(
        _event(
            sequence=3,
            previous_event_id=(
                "sha256:" + ("f" * 64)
                if broken_chain
                else events[-1]["event_id"]
            ),
            event_type="implementation_finished",
            task_bound=True,
            implementation_commit=claimed_implementation,
            baseline_ref=baseline,
            validation_result={
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "target_commit": baseline,
                "authoritative": False,
                "completion_authoritative": False,
                "code_proof_authoritative": False,
                "proof_authoritative": False,
                "freshness_authoritative": False,
                "authority_gates": gates,
                "candidate_binding": {
                    "verified": True,
                    "current_fingerprint": "sha256:" + ("3" * 64),
                    "expected_fingerprint": "sha256:" + ("3" * 64),
                },
                "proposal_gate": {
                    "attempted": True,
                    "accepted": True,
                    "receipt_id": proposal_receipt_id,
                    "repository_tree_id": baseline,
                    "reason_codes": [],
                },
                "coverage_errors": [],
                "validation_dag_receipt": {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "validation-dag-receipt@3"
                    ),
                    "objective_id": "FVT-G200",
                    "receipt_id": "4" * 64,
                    "graph_id": "5" * 64,
                    "proposal_receipt_id": proposal_receipt_id,
                    "repository_tree_id": baseline,
                    "passed": True,
                    "coverage_complete": True,
                    "uncovered_impact": False,
                    "completion_authoritative": False,
                    "code_proof_authoritative": False,
                    "proof_authoritative": False,
                    "authority_gates": gates,
                    "impact_graph": {"repository_tree_id": baseline},
                    "required_validation_ids": ["declared:fvt083"],
                    "selected_node_ids": [node_id],
                    "nodes": [
                        {
                            "node_id": node_id,
                            "validation_id": "declared:fvt083",
                            "selected": True,
                            "mandatory": True,
                            "disposition": "succeeded",
                            "returncode": 0,
                            "result_digest": "6" * 64,
                        }
                    ],
                },
            },
            merge_result={
                "attempted": True,
                "merged": True,
                "returncode": 0,
                "implementation_commit": claimed_implementation,
                "merge_commit": merge,
                "target_branch": target_branch,
                "integration_commit_proof": {
                    "passed": True,
                    "reasons": [],
                    "implementation_commit": claimed_implementation,
                    "integration_commit": merge,
                    "integration_ref": merge,
                    "target_branch": target_branch,
                },
                "post_merge_declared_output_invariant": {
                    "passed": True,
                    "mode": "repository_tree",
                    "repository_ref": merge,
                    "task_ids": [
                        release_evidence.TRUSTED_SUCCESSOR_TASK_ID
                    ],
                    "missing_outputs": [],
                    "unsafe_outputs": [],
                    "untracked_outputs": [],
                    "checks": [
                        {
                            "task_id": (
                                release_evidence.TRUSTED_SUCCESSOR_TASK_ID
                            ),
                            "repository_ref": merge,
                            "path": "implementation.txt",
                            "exists": True,
                            "tracked": True,
                        }
                    ],
                },
            },
        )
    )
    events.append(
        _event(
            sequence=4,
            previous_event_id=events[-1]["event_id"],
            event_type="task_completed",
            task_bound=True,
        )
    )

    state = {
        "implementation_in_progress": False,
        "last_implementation_task_id": (
            release_evidence.TRUSTED_SUCCESSOR_TASK_ID
        ),
        "last_implementation_task_cid": (
            release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_CID
        ),
        "last_implementation_commit": claimed_implementation,
        "last_merge_commit": merge,
        "task_statuses": {
            release_evidence.TRUSTED_SUCCESSOR_TASK_ID: "completed"
        },
        "task_identities": {
            release_evidence.TRUSTED_SUCCESSOR_TASK_ID: {
                "canonical_task_cid": (
                    release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_CID
                ),
                "canonical_task_key": (
                    release_evidence.TRUSTED_SUCCESSOR_CANONICAL_TASK_KEY
                ),
            }
        },
        "attempt": 3,
        "phase": "implementation",
    }
    paths = {
        "task_state": state_dir / "task_state.json",
        "event_log": state_dir / "events.jsonl",
        "event_manifest": state_dir / "events.jsonl.manifest.json",
        "lane_manifest": state_dir / "lane_manifest.json",
        "scheduler_snapshot": state_dir / "scheduler_snapshot.json",
    }
    paths["task_state"].write_text(json.dumps(state), encoding="utf-8")
    paths["event_log"].write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    paths["event_manifest"].write_text(
        json.dumps(
            {
                "stream_id": events[-1]["stream_id"],
                "snapshot_id": events[-1]["snapshot_id"],
                "last_event_id": events[-1]["event_id"],
            }
        ),
        encoding="utf-8",
    )
    paths["lane_manifest"].write_text(
        json.dumps(
            {
                "lane_id": "formal-verification-tactician/toolchain-release",
                "baseline_commit": baseline,
            }
        ),
        encoding="utf-8",
    )
    paths["scheduler_snapshot"].write_text(
        json.dumps({"phase": "implementation", "metrics": {"completed": 1}}),
        encoding="utf-8",
    )
    return paths


def _export(setup: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return release_evidence.export_release_evidence(
        task_state_path=paths["task_state"],
        event_log_path=paths["event_log"],
        event_manifest_path=paths["event_manifest"],
        lane_manifest_path=paths["lane_manifest"],
        scheduler_snapshot_path=paths["scheduler_snapshot"],
        repo_root=setup["repo"],
        metrics_module_present=True,
    )


def test_completed_schema_binds_provisionally_then_finalizes_after_push(
    builder,
    tmp_path: Path,
) -> None:
    setup = _prepare_repository(tmp_path)
    paths = _write_sources(setup)
    evidence = _export(setup, paths)
    verified = release_evidence.verify_release_evidence(
        evidence,
        repo_root=setup["repo"],
    )
    assert verified["valid"] is True

    provisional = builder.derive_supervisor_binding(
        evidence,
        repo_root=setup["repo"],
        integration_branch=INTEGRATION_BRANCH,
    )
    assert provisional["trusted_successor_task_id"] == "FVT-083"
    assert provisional["legacy_display_task_id"] == "FVT-053"
    assert provisional["provisional_bound"] is True
    assert provisional["publication_bound"] is False
    assert provisional["bound"] is False
    assert provisional["publication_phase"] == "provisional_merge"
    assert provisional["post_push_finalization_required"] is True
    assert provisional["validation_dag_bindings"][0][
        "supervisor_execution_authoritative"
    ] is False
    assert "merge_commit_not_published_to_origin_main" in provisional[
        "block_reasons"
    ]

    _git(
        setup["repo"],
        "update-ref",
        "refs/remotes/origin/main",
        setup["merge"],
    )
    evidence_path = tmp_path / "release-evidence.json"
    assert (
        release_evidence.main(
            [
                "--task-state",
                str(paths["task_state"]),
                "--event-log",
                str(paths["event_log"]),
                "--event-manifest",
                str(paths["event_manifest"]),
                "--lane-manifest",
                str(paths["lane_manifest"]),
                "--scheduler-snapshot",
                str(paths["scheduler_snapshot"]),
                "--repo-root",
                str(setup["repo"]),
                "--output",
                str(evidence_path),
            ]
        )
        == 0
    )
    final = builder.finalize_supervisor_release_evidence(
        evidence_path=evidence_path,
        repo_root=setup["repo"],
        integration_branch=INTEGRATION_BRANCH,
    )
    assert final["provisional_bound"] is True
    assert final["publication_bound"] is True
    assert final["bound"] is True
    assert final["publication_phase"] == "published_final"
    assert final["post_push_finalization_required"] is False


def test_identity_event_tree_target_and_publication_tampering_fail_closed(
    builder,
    tmp_path: Path,
) -> None:
    setup = _prepare_repository(tmp_path)
    paths = _write_sources(setup)
    evidence = _export(setup, paths)

    # Re-hashing a modified envelope cannot replace replay against the bound
    # source files.
    forged_event = copy.deepcopy(evidence)
    forged_event["snapshot"]["events"][2]["type"] = "attacker_finished"
    forged_event.pop("content_id")
    forged_event["content_id"] = release_evidence.content_digest(forged_event)
    replay_rejection = release_evidence.verify_release_evidence(
        forged_event,
        repo_root=setup["repo"],
    )
    assert replay_rejection["valid"] is False
    assert "release_evidence_source_replay_mismatch" in replay_rejection[
        "failures"
    ]

    wrong_identity = _export(
        setup,
        _write_sources(setup, wrong_receipt_identity=True),
    )
    identity_binding = builder.derive_supervisor_binding(
        wrong_identity,
        repo_root=setup["repo"],
    )
    assert identity_binding["provisional_bound"] is False
    assert "member_completion_receipt_missing" in identity_binding[
        "block_reasons"
    ]

    broken_chain = _export(
        setup,
        _write_sources(setup, broken_chain=True),
    )
    chain_rejection = release_evidence.verify_release_evidence(
        broken_chain,
        repo_root=setup["repo"],
    )
    assert chain_rejection["valid"] is False
    assert "event_chain_invalid" in chain_rejection["failures"]

    wrong_tree = _export(
        setup,
        _write_sources(setup, wrong_tree_binding=True),
    )
    tree_binding = builder.derive_supervisor_binding(
        wrong_tree,
        repo_root=setup["repo"],
    )
    assert tree_binding["provisional_bound"] is False
    assert "merge_commit_tree_evidence_missing" in tree_binding[
        "block_reasons"
    ]

    wrong_target = _export(
        setup,
        _write_sources(setup, wrong_target_branch=True),
    )
    target_binding = builder.derive_supervisor_binding(
        wrong_target,
        repo_root=setup["repo"],
    )
    assert target_binding["provisional_bound"] is False

    # A payload claim cannot bypass the independent origin/main ref check.
    final_claim = copy.deepcopy(_export(setup, _write_sources(setup)))
    final_claim["snapshot"]["publication"]["published"] = True
    final_claim.pop("content_id")
    final_claim["content_id"] = release_evidence.content_digest(final_claim)
    publication_binding = builder.derive_supervisor_binding(
        final_claim,
        repo_root=setup["repo"],
    )
    assert publication_binding["bound"] is False
    assert publication_binding["publication_bound"] is False


def test_role_aware_cli_requires_export_and_rejects_raw_inputs(
    builder,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as missing:
        builder.main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--role-aware",
                "--output",
                str(tmp_path / "completion.json"),
            ]
        )
    assert missing.value.code == 2

    state = tmp_path / "state.json"
    events = tmp_path / "events.jsonl"
    state.write_text("{}\n", encoding="utf-8")
    events.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as raw:
        builder.main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--role-aware",
                "--supervisor-task-state",
                str(state),
                "--supervisor-event-log",
                str(events),
            ]
        )
    assert raw.value.code == 2
