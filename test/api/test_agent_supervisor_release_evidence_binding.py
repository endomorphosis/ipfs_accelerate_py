"""FVT-G212: bind durable supervisor release evidence and enforce expected outputs.

Covers:

* exact ignored declared outputs may be force-staged path-by-path only
* proposals fail ``expected_output_ignored_or_unstaged`` when outputs are
  missing, protected, or still unstaged
* a regression commit carries both an ignored JSON deliverable and a tracked
  source change
* ``AgentSupervisorReleaseEvidence@1`` reads sources once, hashes raw bytes,
  binds identity/trees/events/receipts, never mutates live state, and never
  treats metrics-module presence as completion
* leased-lane durable completion fencing shares the member-receipt schema
* FVT-078 objective validation repair: exact-text discovery of
  ``objective validation repair`` without granting completion authority
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge import leased_lane
from ipfs_accelerate_py.agent_supervisor.release_evidence import (
    EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED,
    MEMBER_COMPLETION_RECEIPT_SCHEMA,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    RELEASE_EVIDENCE_BINDING_TEST,
    RELEASE_EVIDENCE_GOAL_ID,
    RELEASE_EVIDENCE_INTERFACE,
    RELEASE_EVIDENCE_SCHEMA,
    TRUSTED_SUCCESSOR_CANONICAL_TASK_CID,
    TRUSTED_SUCCESSOR_CANONICAL_TASK_KEY,
    TRUSTED_SUCCESSOR_TASK_ID,
    all_covered_evidence_terms,
    content_digest,
    export_release_evidence,
    objective_validation_repair_claim,
    objective_validation_repair_evidence_terms,
    release_evidence_domain_terms,
    sha256_bytes,
    verify_release_evidence,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {' '.join(args)} failed in {repo}:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Release Evidence Test")
    _git(path, "config", "user.email", "release-evidence@example.invalid")
    return path


def _daemon(
    repo: Path,
    *,
    implementation_protected_paths: tuple[str, ...] = (),
) -> TodoImplementationDaemon:
    state_dir = repo.parent / f".{repo.name}-release-evidence-state"
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="FVT-",
        worktree_pool_enabled=False,
        implementation_protected_paths=implementation_protected_paths,
    )


def _proposal_task(task_id: str, *outputs: str) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title=f"Produce {', '.join(outputs)}",
        status="todo",
        completion="manual",
        priority="P0",
        track="supervisor-integrity",
        outputs=list(outputs),
        validation=["python -m pytest"],
    )


def _event_id(body: dict) -> str:
    return content_digest({key: value for key, value in body.items() if key != "event_id"})


def _canonical_event(
    *,
    sequence: int,
    previous_event_id: str,
    stream_id: str,
    snapshot_id: str,
    event_type: str,
    task_id: str,
    canonical_task_cid: str,
    canonical_task_key: str,
    **extra: object,
) -> dict:
    body = {
        "sequence": sequence,
        "previous_event_id": previous_event_id,
        "stream_id": stream_id,
        "snapshot_id": snapshot_id,
        "type": event_type,
        "timestamp": f"2026-07-31T12:0{sequence}:00+00:00",
        "task_id": task_id,
        "canonical_task_cid": canonical_task_cid,
        "canonical_task_key": canonical_task_key,
        **extra,
    }
    body["event_id"] = _event_id(body)
    return body


def test_expected_output_force_stages_exact_ignored_json_and_tracked_source(
    tmp_path: Path,
) -> None:
    """Ignored JSON + tracked source both enter the commit; unrelated ignored does not."""

    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")

    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "deliverable.json").write_text(
        '{"certified": true}\n',
        encoding="utf-8",
    )
    (repo / "unrelated.json").write_text(
        '{"must_not_be_staged": true}\n',
        encoding="utf-8",
    )
    daemon = _daemon(repo)
    task = _proposal_task(
        "FVT-063-A",
        "implementation.py",
        "deliverable.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )
    assert proposal.accepted is True
    assert set(proposal.proposal.changed_paths) == {
        "deliverable.json",
        "implementation.py",
    }
    assert _git(repo, "diff", "--cached", "--name-only") == "deliverable.json"

    result = daemon._commit_worktree_changes(
        repo,
        task,
        1,
        baseline_ref=baseline,
    )
    assert result["committed"] is True
    candidate = result["commit"]
    names = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        candidate,
    ).splitlines()
    assert names == ["deliverable.json", "implementation.py"]
    assert _git(repo, "show", f"{candidate}:deliverable.json") == (
        '{"certified": true}'
    )
    assert _git(repo, "show", f"{candidate}:implementation.py") == "VALUE = 1"
    assert (
        _git(repo, "ls-tree", "--name-only", candidate, "--", "unrelated.json")
        == ""
    )


def test_expected_output_missing_fails_with_stable_reason(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    daemon = _daemon(repo)
    task = _proposal_task(
        "FVT-063-B",
        "implementation.py",
        "missing.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )
    assert proposal.accepted is False
    assert {
        finding.code.value for finding in proposal.findings
    } == {EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED}
    assert [finding.path for finding in proposal.findings] == ["missing.json"]

    result = daemon._commit_worktree_changes(
        repo,
        task,
        1,
        baseline_ref=baseline,
    )
    assert result["committed"] is False
    assert result["reason"] == EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED
    assert result["declared_output_invariant"]["missing_outputs"] == [
        {"task_id": "FVT-063-B", "path": "missing.json"}
    ]


def test_expected_output_never_force_stages_protected_ignored_path(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.json\n", encoding="utf-8")
    (repo / "implementation.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "implementation.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "protected.json").write_text('{"protected": true}\n', encoding="utf-8")
    daemon = _daemon(
        repo,
        implementation_protected_paths=("protected.json",),
    )
    task = _proposal_task(
        "FVT-063-C",
        "implementation.py",
        "protected.json",
    )

    proposal = daemon._validate_implementation_patch(
        repo,
        task,
        baseline_ref=baseline,
    )
    assert proposal.accepted is False
    finding_codes = {finding.code.value for finding in proposal.findings}
    assert EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED in finding_codes
    assert "protected.json" in {finding.path for finding in proposal.findings}
    # Protected declared ignored outputs must never enter the index.
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_export_release_evidence_binds_durable_sources_and_is_read_only(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path / "repo")
    # Ensure exporter path resolves for verify_release_evidence identity.
    exporter_src = Path(
        "ipfs_accelerate_py/agent_supervisor/release_evidence.py"
    ).resolve()
    assert exporter_src.is_file()

    state_dir = tmp_path / "lane" / "state"
    state_dir.mkdir(parents=True)
    task_id = TRUSTED_SUCCESSOR_TASK_ID
    canonical_task_cid = TRUSTED_SUCCESSOR_CANONICAL_TASK_CID
    canonical_task_key = TRUSTED_SUCCESSOR_CANONICAL_TASK_KEY
    stream_id = "event-log:sha256:abc"
    snapshot_id = "event-log-snapshot:sha256:abc"
    implementation_commit = "a" * 40
    merge_commit = "b" * 40

    task_state = {
        "active_task_id": "",
        "active_task_cid": "",
        "implementation_in_progress": False,
        "last_implementation_task_id": task_id,
        "last_implementation_task_cid": canonical_task_cid,
        "last_implementation_commit": implementation_commit,
        "last_merge_commit": merge_commit,
        "baseline_tree": "c" * 40,
        "merged_tree": "d" * 40,
        "gitlinks": {"ipfs_datasets_py": "e" * 40},
        "heartbeat_at": "2026-07-31T12:02:30+00:00",
        "task_statuses": {task_id: "completed"},
        "task_identities": {
            task_id: {
                "canonical_task_cid": canonical_task_cid,
                "canonical_task_key": canonical_task_key,
            }
        },
        "dependency_cids": ["bafydep0001"],
        "attempt": 2,
        "phase": "implementation",
    }
    task_state_path = state_dir / "task_state.json"
    task_state_bytes = json.dumps(task_state, sort_keys=True).encode("utf-8")
    task_state_path.write_bytes(task_state_bytes)

    receipt = {
        "schema": MEMBER_COMPLETION_RECEIPT_SCHEMA,
        "task_id": task_id,
        "canonical_task_cid": canonical_task_cid,
        "canonical_task_key": canonical_task_key,
        "status": "succeeded",
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
    }
    first = _canonical_event(
        sequence=1,
        previous_event_id="",
        stream_id=stream_id,
        snapshot_id=snapshot_id,
        event_type="implementation_started",
        task_id=task_id,
        canonical_task_cid=canonical_task_cid,
        canonical_task_key=canonical_task_key,
        attempt=2,
        phase="implementation",
    )
    second = _canonical_event(
        sequence=2,
        previous_event_id=first["event_id"],
        stream_id=stream_id,
        snapshot_id=snapshot_id,
        event_type="implementation_finished",
        task_id=task_id,
        canonical_task_cid=canonical_task_cid,
        canonical_task_key=canonical_task_key,
        implementation_commit=implementation_commit,
        attempt=2,
        phase="implementation",
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "target_commit": implementation_commit,
            "receipt_id": "validation:1",
        },
        merge_result={
            "merged": True,
            "implementation_commit": implementation_commit,
            "merge_commit": merge_commit,
            "target_branch": "main",
            "baseline_tree": "c" * 40,
            "merged_tree": "d" * 40,
            "gitlinks": {"ipfs_datasets_py": "e" * 40},
        },
        todo_update_result={
            "completion_receipts": [receipt],
        },
    )
    events_path = state_dir / "events.jsonl"
    events_text = json.dumps(first) + "\n" + json.dumps(second) + "\n"
    events_bytes = events_text.encode("utf-8")
    events_path.write_bytes(events_bytes)

    lane_manifest = {
        "lane_id": "supervisor-release-evidence",
        "baseline_tree": "c" * 40,
        "baseline_commit": "f" * 40,
        "gitlinks": {"ipfs_datasets_py": "e" * 40},
        "bundle_id": "formal-verification-tactician/supervisor-release-evidence",
    }
    lane_path = state_dir / "lane_manifest.json"
    lane_path.write_text(json.dumps(lane_manifest), encoding="utf-8")

    scheduler_snapshot = {
        "phase": "implementation",
        "metrics": {"tasks_completed": 1},
    }
    scheduler_path = state_dir / "scheduler_snapshot.json"
    scheduler_path.write_text(json.dumps(scheduler_snapshot), encoding="utf-8")

    bundle_metadata = {
        "bundle_id": "formal-verification-tactician/supervisor-release-evidence",
        "goal_id": RELEASE_EVIDENCE_GOAL_ID,
        "dependency_cids": ["bafydep0001"],
    }
    bundle_path = state_dir / "bundle_metadata.json"
    bundle_path.write_text(json.dumps(bundle_metadata), encoding="utf-8")

    task_metadata = {
        "task_id": task_id,
        "canonical_task_cid": canonical_task_cid,
        "canonical_task_key": canonical_task_key,
        "expected_outputs": [
            "ipfs_accelerate_py/agent_supervisor/release_evidence.py",
        ],
        "dependency_cids": ["bafydep0001"],
    }
    task_meta_path = state_dir / "task_metadata.json"
    task_meta_path.write_text(json.dumps(task_metadata), encoding="utf-8")

    event_manifest = {
        "stream_id": stream_id,
        "snapshot_id": snapshot_id,
        "last_event_id": second["event_id"],
    }
    manifest_path = state_dir / "event_manifest.json"
    manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")

    receipts_path = state_dir / "member_completion_receipts.json"
    receipts_path.write_text(
        json.dumps({"member_completion_receipts": [receipt]}),
        encoding="utf-8",
    )

    before_state = task_state_path.read_bytes()
    before_events = events_path.read_bytes()

    export = export_release_evidence(
        task_id=task_id,
        task_state_path=task_state_path,
        event_log_path=events_path,
        event_manifest_path=manifest_path,
        lane_manifest_path=lane_path,
        scheduler_snapshot_path=scheduler_path,
        bundle_metadata_path=bundle_path,
        task_metadata_path=task_meta_path,
        member_completion_receipts_path=receipts_path,
        repo_root=exporter_src.parents[2],
        metrics_module_present=True,
    )

    # Live sources must not be rewritten.
    assert task_state_path.read_bytes() == before_state
    assert events_path.read_bytes() == before_events

    assert export["schema"] == RELEASE_EVIDENCE_SCHEMA
    assert export["interface"] == RELEASE_EVIDENCE_INTERFACE
    assert export["goal_id"] == RELEASE_EVIDENCE_GOAL_ID
    assert export["completion_authoritative"] is False
    assert export["proof_authoritative"] is False
    assert "task_state" not in export
    assert "events" not in export
    assert "task_state_source" not in export

    snapshot = export["snapshot"]
    identity = snapshot["task_state"]["canonical_identity"]
    assert identity["canonical_task_cid"] == canonical_task_cid
    assert identity["canonical_task_key"] == canonical_task_key
    assert snapshot["dependency_cids"] == ["bafydep0001"]
    assert snapshot["trees"]["baseline_tree"] == "c" * 40
    assert snapshot["trees"]["merged_tree"] == "d" * 40
    assert snapshot["trees"]["gitlinks"]["ipfs_datasets_py"] == "e" * 40
    assert snapshot["trees"]["implementation_commit"] == implementation_commit
    assert snapshot["trees"]["merge_commit"] == merge_commit
    assert snapshot["attempt_phase"]["attempt"] == 2
    assert snapshot["attempt_phase"]["phase"] == "implementation"
    assert snapshot["event_chain"]["valid"] is True
    assert snapshot["event_chain"]["continuous"] is True
    assert snapshot["event_chain"]["event_count"] == 2
    assert len(snapshot["events"]) == 2
    assert snapshot["member_completion_receipts"]
    assert (
        snapshot["member_completion_receipts"][0]["schema"]
        == MEMBER_COMPLETION_RECEIPT_SCHEMA
    )
    assert snapshot["authority"]["metrics_module_present"] is True
    assert snapshot["authority"]["metrics_module_is_completion"] is False
    assert snapshot["authority"]["completion_bound"] is True
    assert snapshot["freshness"]["live_state_mutated"] is False
    assert snapshot["freshness"]["all_sources_read_once"] is True

    sources_by_key = {item["key"]: item for item in snapshot["sources"]}
    assert sources_by_key["task_state"]["sha256"] == sha256_bytes(task_state_bytes)
    assert sources_by_key["event_log"]["sha256"] == sha256_bytes(events_bytes)
    assert sources_by_key["task_state"]["read_once"] is True
    assert sources_by_key["event_log"]["mutated"] is False

    verified = verify_release_evidence(
        export,
        repo_root=exporter_src.parents[2],
    )
    assert verified["valid"] is True
    assert verified["failures"] == []
    assert verified["snapshot"]["task_id"] == task_id
    assert verified["exporter"]["bound"] is True


def test_verify_release_evidence_rejects_raw_state_and_metrics_as_completion(
    tmp_path: Path,
) -> None:
    exporter_src = Path(
        "ipfs_accelerate_py/agent_supervisor/release_evidence.py"
    ).resolve()
    repo_root = exporter_src.parents[2]

    export = export_release_evidence(
        task_id="FVT-053",
        task_state={
            "implementation_in_progress": False,
            "task_identities": {
                "FVT-053": {
                    "canonical_task_cid": "cid-1",
                    "canonical_task_key": "key-1",
                }
            },
            "task_statuses": {"FVT-053": "completed"},
            "last_implementation_task_id": "FVT-053",
            "last_implementation_task_cid": "cid-1",
            "last_implementation_commit": "a" * 40,
            "last_merge_commit": "b" * 40,
        },
        events=[
            _canonical_event(
                sequence=1,
                previous_event_id="",
                stream_id="stream",
                snapshot_id="snap",
                event_type="implementation_finished",
                task_id="FVT-053",
                canonical_task_cid="cid-1",
                canonical_task_key="key-1",
                implementation_commit="a" * 40,
                validation_result={
                    "attempted": True,
                    "passed": True,
                    "returncode": 0,
                    "target_commit": "a" * 40,
                },
                merge_result={
                    "merged": True,
                    "implementation_commit": "a" * 40,
                    "merge_commit": "b" * 40,
                },
                todo_update_result={
                    "completion_receipts": [
                        {
                            "schema": MEMBER_COMPLETION_RECEIPT_SCHEMA,
                            "task_id": "FVT-053",
                            "canonical_task_cid": "cid-1",
                            "canonical_task_key": "key-1",
                            "status": "succeeded",
                            "implementation_commit": "a" * 40,
                            "merge_commit": "b" * 40,
                        }
                    ]
                },
            )
        ],
        repo_root=repo_root,
        metrics_module_present=True,
    )
    assert export["snapshot"]["authority"]["metrics_module_is_completion"] is False

    # Raw supervisor state at the export root is not release evidence.
    raw = dict(export)
    raw["task_state"] = {"spoofed": True}
    raw.pop("content_id", None)
    raw["content_id"] = content_digest(raw)
    rejected = verify_release_evidence(raw, repo_root=repo_root)
    assert rejected["valid"] is False
    assert "raw_supervisor_state_is_not_release_evidence" in rejected["failures"]

    # Tampered content_id fails closed.
    tampered = dict(export)
    tampered["content_id"] = "sha256:" + ("0" * 64)
    rejected_id = verify_release_evidence(tampered, repo_root=repo_root)
    assert rejected_id["valid"] is False
    assert "content_id_mismatch" in rejected_id["failures"]

    # Metrics module may be present without granting completion authority.
    metrics_only = export_release_evidence(
        task_id="FVT-053",
        task_state={
            "implementation_in_progress": True,
            "task_identities": {
                "FVT-053": {
                    "canonical_task_cid": "cid-1",
                    "canonical_task_key": "key-1",
                }
            },
        },
        events=[],
        scheduler_snapshot={"metrics": {"present": True}},
        repo_root=repo_root,
        metrics_module_present=True,
    )
    assert metrics_only["snapshot"]["authority"]["completion_bound"] is False
    assert metrics_only["snapshot"]["authority"]["metrics_module_present"] is True
    assert (
        metrics_only["snapshot"]["member_completion_receipts"] == []
    )


def test_export_never_synthesizes_missing_member_completion_receipt(
    tmp_path: Path,
) -> None:
    exporter_src = Path(
        "ipfs_accelerate_py/agent_supervisor/release_evidence.py"
    ).resolve()
    export = export_release_evidence(
        task_id="FVT-053",
        task_state={
            "implementation_in_progress": False,
            "task_identities": {
                "FVT-053": {
                    "canonical_task_cid": "cid-1",
                    "canonical_task_key": "key-1",
                }
            },
            "task_statuses": {"FVT-053": "completed"},
        },
        events=[
            _canonical_event(
                sequence=1,
                previous_event_id="",
                stream_id="stream",
                snapshot_id="snap",
                event_type="implementation_finished",
                task_id="FVT-053",
                canonical_task_cid="cid-1",
                canonical_task_key="key-1",
                merge_result={"merged": True},
            )
        ],
        repo_root=exporter_src.parents[2],
    )
    assert export["snapshot"]["member_completion_receipts"] == []
    assert export["snapshot"]["authority"]["completion_bound"] is False


def test_leased_lane_shares_member_completion_receipt_schema() -> None:
    """Leased-lane fencing and G212 exports share one receipt schema constant."""

    assert (
        leased_lane._MEMBER_COMPLETION_RECEIPT_SCHEMA
        == MEMBER_COMPLETION_RECEIPT_SCHEMA
    )
    validated = leased_lane._validated_member_completion_receipts(
        [
            {
                "schema": MEMBER_COMPLETION_RECEIPT_SCHEMA,
                "task_id": "FVT-053",
                "canonical_task_cid": "cid-1",
                "status": "succeeded",
            }
        ],
        {"FVT-053": "cid-1"},
    )
    assert validated == [
        {"task_id": "FVT-053", "canonical_task_cid": "cid-1"}
    ]
    assert (
        leased_lane._validated_member_completion_receipts(
            [
                {
                    "schema": "wrong.schema@1",
                    "task_id": "FVT-053",
                    "canonical_task_cid": "cid-1",
                    "status": "succeeded",
                }
            ],
            {"FVT-053": "cid-1"},
        )
        is None
    )


def test_objective_validation_repair_evidence_term_discoverable() -> None:
    """FVT-G212 / FVT-078 objective validation repair: exact-text discovery key.

    Anchors the synthetic phrase ``objective validation repair`` so objective
    scans re-find the validation gate.  Domain evidence
    (``AgentSupervisorReleaseEvidence@1``, binding tests, member receipts)
    stays separate from the repair term.  The repair term never enters export
    content_id identity, completion authority, or proof authority.  Parent
    domain goal remains FVT-G212; the repair obligation is owned by FVT-078.
    """

    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
    assert OBJECTIVE_GOAL_ID == "FVT-G212"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "FVT-078"
    assert RELEASE_EVIDENCE_GOAL_ID == "FVT-G212"
    assert RELEASE_EVIDENCE_BINDING_TEST == (
        "test/api/test_agent_supervisor_release_evidence_binding.py"
    )
    assert objective_validation_repair_evidence_terms() == (
        "objective validation repair",
    )
    domain = release_evidence_domain_terms()
    assert RELEASE_EVIDENCE_INTERFACE in domain
    assert RELEASE_EVIDENCE_SCHEMA in domain
    assert RELEASE_EVIDENCE_BINDING_TEST in domain
    assert MEMBER_COMPLETION_RECEIPT_SCHEMA in domain
    assert "objective validation repair" not in domain
    assert all_covered_evidence_terms() == domain + (
        "objective validation repair",
    )
    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE in all_covered_evidence_terms()

    # Leased-lane predicted path re-exports the same discovery key.
    assert (
        leased_lane.OBJECTIVE_VALIDATION_REPAIR_EVIDENCE
        == "objective validation repair"
    )
    assert leased_lane.OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "FVT-078"

    claim = objective_validation_repair_claim()
    assert claim["evidence"] == "objective validation repair"
    assert claim["requirement_id"] == "objective validation repair"
    assert claim["goal_id"] == "FVT-G212"
    assert claim["task_id"] == "FVT-078"
    assert claim["interface"] == RELEASE_EVIDENCE_INTERFACE
    assert claim["completion_authoritative"] is False
    assert claim["proof_authoritative"] is False
    assert "objective validation repair" not in claim["domain_evidence_terms"]
    assert claim["repair_evidence_terms"] == ["objective validation repair"]
    assert "expected_output" in claim["validation"]
    assert "release_evidence" in claim["validation"]

    # Domain export identity must not absorb the synthetic repair term.
    export = export_release_evidence(
        task_id="FVT-053",
        task_state={
            "implementation_in_progress": False,
            "task_identities": {
                "FVT-053": {
                    "canonical_task_cid": "cid-1",
                    "canonical_task_key": "key-1",
                }
            },
        },
        events=[],
        metrics_module_present=True,
    )
    encoded = json.dumps(export, sort_keys=True)
    assert export["interface"] == RELEASE_EVIDENCE_INTERFACE
    assert export["goal_id"] == RELEASE_EVIDENCE_GOAL_ID
    assert export["completion_authoritative"] is False
    assert export["proof_authoritative"] is False
    assert "objective validation repair" not in encoded
    assert export["snapshot"]["authority"]["metrics_module_is_completion"] is False


def test_expected_output_preflight_compares_filesystem_proposal_and_stage(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("artifact.json\n", encoding="utf-8")
    (repo / "src.py").write_text("X = 0\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "src.py")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "src.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
    daemon = _daemon(repo)
    task = _proposal_task("FVT-063-D", "src.py", "artifact.json")

    preflight = daemon._prepare_proposal_expected_outputs(
        repo,
        task,
        baseline_ref=baseline,
        scope_paths=("src.py", "artifact.json"),
    )
    assert preflight["expected_paths"] == ["artifact.json", "src.py"]
    checks = {item["path"]: item for item in preflight["checks"]}
    assert checks["artifact.json"]["exists"] is True
    assert checks["artifact.json"]["ignored"] is True
    assert checks["artifact.json"]["force_stage_required"] is True
    assert checks["artifact.json"]["force_stage_succeeded"] is True
    assert checks["artifact.json"]["staged"] is True
    assert checks["src.py"]["exists"] is True
    assert checks["src.py"]["ignored"] is False
    assert checks["src.py"]["force_stage_required"] is False
    assert not any(item.get("issue") for item in preflight["checks"])
