"""Native checked scheduling lifecycle with attached semantic/world context.

An independently reviewed integer model supplies the semantics. Feasibility
checks are not kernel proofs, optimality guarantees or a Terminal-Bench score.
The native owner alone validates, publishes and completes the admitted task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from test.api.test_doctor_schedule_contract import isolated_schedule_resources  # noqa: F401


def _provider_free_runner(path: Path, marker: Path) -> None:
    """Trip both the owner test and spawned worker on any router invocation."""
    package_root = str(Path(__file__).resolve().parents[2])
    path.write_text(f'''import sys
from pathlib import Path
sys.path.insert(0, {package_root!r})
from ipfs_accelerate_py import llm_router
def forbidden(*args, **kwargs):
    Path({str(marker)!r}).write_text("unexpected provider call\\n")
    raise AssertionError("finite schedule qualification must not call a model")
for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
             "generate_text_mesh_batch", "get_llm_provider"):
    setattr(llm_router, name, forbidden)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import main
raise SystemExit(main())
''')


def test_signed_native_schedule_validates_publishes_completes_and_stops(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_data_contract import prepare_interval_schedule_candidate
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
    from test.api.test_intent_interval_schedule import _schedule_case
    from ipfs_datasets_py.logic.software_contracts.finite_interval_schedule import (
        FiniteIntervalScheduleContract, check_finite_interval_schedule, verify_finite_schedule_check,
    )

    capabilities = probe_quack_capabilities()
    if not capabilities.passes_health_check:
        pytest.skip(f"actual installed Quack required: {capabilities.reason_code}")
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "ambient-unrelated"))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("finite schedule qualification must not call a model")

    for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
                 "generate_text_mesh_batch", "get_llm_provider"):
        monkeypatch.setattr(llm_router, name, forbidden)

    case = _schedule_case(tmp_path / "fixture")
    repository = case["repository"]
    manifest = case["manifest"]["payload"]
    proposed = case["proposed"]
    task = proposed["graph"].tasks[0]
    task_cid = task.task_cid
    input_path = case["contract"]["reviewed_interval_schedule"]["input_path"]
    output_path = case["contract"]["reviewed_interval_schedule"]["output_path"]
    baseline = local._git(repository, "rev-parse", "HEAD")
    immutable = {path: (repository / path).read_bytes() for path in manifest["sources"]}
    check = list(task.validations[0].argv)
    assert subprocess.run(check, cwd=repository, capture_output=True, timeout=10).returncode != 0
    database = tmp_path / "intent.duckdb"
    with IntentRepository(database) as intent:
        admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
            requirement_bindings=proposed["requirement_bindings"])
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        before = intent.get_task(task_cid)
        candidate = prepare_interval_schedule_candidate(repository=repository, admission=admission,
            intent=intent, task_cid=task_cid, state=tmp_path / "doctor")
        assert candidate["status"] == "candidate_ready", candidate
        assert candidate["route"] == "doctor_contract_candidate"
        assert candidate["provider_calls"] == 0
        assert candidate["publication_authority"] is candidate["completion_authority"] is False
        assert candidate["check"]["check"]["evidence_kind"] == "finite_schedule_check"
        assert candidate["check"]["check"]["kernel_checked"] is False
        assert candidate["kernel_proved"] is candidate["whole_program_proved"] is False
        assert candidate["contract_index"]["hydrated"] is True
        assert candidate["contract_index"]["active_receipt_ids"] == []
        assert intent.get_task(task_cid) == before and before["status"] == "ready"
        assert not (repository / output_path).exists()
        assert local._git(repository, "rev-parse", "HEAD") == baseline
        artifact = Path(candidate["artifact"])
        payload = json.loads(artifact.read_bytes())
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == candidate["sha256"]
        assert payload["edits"][0]["effect"] == "create"
        assert payload["edits"][0]["before_sha256"] is None
        context = prepare_supervised_task_context(repository=repository, intent=intent,
            task_cid=task_cid, paths=sorted(manifest["sources"]),
            required_raw_paths=["input.json", "instruction.txt"],
            output=repository / ".runtime/schedule-context",
            semantic_program_paths=["public_check.py"],
            semantic_worker_query=case["source"], semantic_max_symbols=256)
        semantic = context["semantic"]
        assert semantic["ducklake"]["status"] == "projected"
        assert semantic["ducklake"]["stored_catalogs"] == 2
        assert semantic["ducklake"]["stored_links"] == 2
        assert semantic["ducklake"]["authoritative"] is False
        assert context["semantic_root_cid"] and context["world_snapshot_cid"]
        assert context["retrieval"] is None  # This path attaches semantic/world context, not a vector index.
        assert context["canonical_task_mutated"] is False
        bundle = write_task_context_bundle(repository=repository, prepared=[context],
            output=repository / ".runtime/schedule-context-bundle.json")
        assert intent.get_task(task_cid) == before

    marker = tmp_path / "unexpected-provider-call.txt"
    runner = tmp_path / "provider_free_candidate.py"
    _provider_free_runner(runner, marker)
    command = shlex.join([sys.executable, "-B", str(runner), "--artifact", candidate["artifact"],
        "--sha256", candidate["sha256"], "--task-cid", task_cid])
    worktrees = tmp_path / "allocated-worktrees"
    worktrees.mkdir(mode=0o750)
    with open_existing_native_owner(database=database, checkout=repository, state_dir=tmp_path / "owner",
            repository_id=manifest["repository_cid"], execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=owner.server, source=owner.source, implement=True, implementation_command=command,
            implementation_timeout_seconds=30, max_task_attempts=1, worker_worktree_root=worktrees,
            context_bundle=bundle, refresh_context_on_completion=True)
        try:
            started = runtime.start()
            assert started.succeeded, started.error
            deadline = time.monotonic() + 90
            while True:
                current = owner.source.get_task(task_cid)
                if current.status in {"completed", "failed", "blocked", "cancelled"}:
                    break
                assert time.monotonic() < deadline, (current.status, current.revision)
                assert runtime.process.snapshot(runtime.profile).members
                time.sleep(.25)
            assert current.status == "completed", (current.status, current.revision)
            assert current.revision >= 4
            assert runtime.bootstrap_receipts and not runtime.bootstrap_errors
            history = owner.source.task_revision_diagnostic_window(task_cid, current_revision=current.revision)
            admitted = [row for row in history["revisions"] if row["body"].get("completion_receipt", {}).get(
                "claim_phase_schema") == TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA]
            assert len(admitted) == 1
            claim = admitted[0]["body"]["completion_receipt"]
            assert claim["operation"] == "database_attempt_admitted" and claim["claim_process_attestation"]
            assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
            assert owner.source.get_task(task_cid).status == "completed"
            published = runtime.observe()["published_context"]
            assert published and all(row["status"] == "refreshed" for row in published)
            logs = list((runtime.state / "run/admitted_database_portal_attempts").rglob("*.log"))
            materialized = [json.loads(line) for path in logs for line in path.read_text().splitlines()
                if line.startswith('{"') and 'native-doctor-contract-candidate-materialization@1' in line]
            assert len(materialized) == 1
            assert materialized[0]["status"] == "candidate_materialized"
            assert materialized[0]["provider_calls"] == 0
            assert materialized[0]["writes"] == [{"path": output_path, "effect": "create",
                "after_sha256": payload["edits"][0]["after_sha256"], "write_mode": "exclusive_create"}]
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
            runtime.close()

    assert calls == [] and not marker.exists()
    assert subprocess.run(check, cwd=repository, capture_output=True, timeout=10).returncode == 0
    assert local._git(repository, "rev-parse", "HEAD") != baseline
    assert {path: (repository / path).read_bytes() for path in immutable} == immutable
    assert (repository / input_path).read_bytes() == immutable[input_path]
    assert local._git(repository, "diff", "--name-only", baseline, "HEAD").splitlines() == [output_path]
    assert hashlib.sha256((repository / output_path).read_bytes()).hexdigest() == payload["edits"][0]["after_sha256"]
    published_bytes = (repository / output_path).read_bytes()
    independently_checked = check_finite_interval_schedule(case["input_bytes"], published_bytes,
        FiniteIntervalScheduleContract())
    assert independently_checked["status"] == "checked"
    assert independently_checked["job_count"] == len(case["input_value"]["jobs"])
    assert independently_checked["kernel_checked"] is False
    verify_finite_schedule_check(candidate["check"]["check"], input_bytes=case["input_bytes"],
        output_bytes=published_bytes, contract=FiniteIntervalScheduleContract())
