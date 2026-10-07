"""Actual signed owner lifecycle for an indexed, source-proved package repair.

The signed public smoke is structural; the additional authored runtime check
observes repaired behavior. Neither one establishes general program semantics.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_symbolic_repair_pipeline import container_umask  # noqa: F401


def _provider_free_runner(path, marker):
    package_root = str(Path(__file__).resolve().parents[2])
    path.write_text(f'''import sys
from pathlib import Path
sys.path.insert(0, {package_root!r})
from ipfs_accelerate_py import llm_router
def forbidden(*args, **kwargs):
    Path({str(marker)!r}).write_text("unexpected provider call\\n")
    raise AssertionError("package alias qualification must not call a model")
for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
             "generate_text_mesh_batch", "get_llm_provider"):
    setattr(llm_router, name, forbidden)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import main
raise SystemExit(main())
''')


def test_signed_indexed_package_alias_validates_publishes_completes_and_stops(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
    from benchmarks.agent_supervisor.container_coding.test_terminal_package_alias_repair import prepared_package_alias
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA

    capabilities = probe_quack_capabilities()
    if not capabilities.passes_health_check:
        pytest.skip(f"actual installed Quack required: {capabilities.reason_code}")
    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "ambient-unrelated"))
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("package alias qualification must not call a model")

    for name in ("generate_text", "generate_text_batch", "generate_text_mesh",
                 "generate_text_mesh_batch", "get_llm_provider"):
        monkeypatch.setattr(llm_router, name, forbidden)

    case = prepared_package_alias(tmp_path / "fixture", nested=True)
    repository, state, admission = case["repository"], case["state"], case["admission"]
    task_cid = case["task_cid"]
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest, task = verified["manifest"], verified["graph"].tasks[0]
    baseline = local._git(repository, "rev-parse", "HEAD")
    immutable = {path: (repository / path).read_bytes() for path in manifest["sources"]
                 if path != case["selected"]}
    behavior = ["python3", "-B", "-c",
        "from pkg.sub.worker import answer; assert answer(3) == 6; assert answer(-2) == -4"]
    initial = subprocess.run(behavior, cwd=repository, capture_output=True, timeout=10)
    assert initial.returncode != 0 and b"NameError" in initial.stderr
    database = state / "intent.duckdb"
    with IntentRepository(database, install_schema=False) as intent:
        before = intent.get_task(task_cid)
    candidate = dispatch.prepare_terminal_doctor_dispatch(repository=repository, state=state,
        admission=admission, task_cid=task_cid)
    assert candidate["status"] == "candidate_ready", candidate
    assert candidate["route"] == "doctor_candidate" and candidate["provider_calls"] == 0
    assert candidate["publication_authority"] is candidate["completion_authority"] is False
    assert candidate["symbolic_capabilities"]["proof"]["local_contract_proof_reported"] is True
    assert candidate["symbolic_capabilities"]["proof"]["whole_program_verified"] is False
    with IntentRepository(database, install_schema=False) as intent:
        assert intent.get_task(task_cid) == before and before["status"] == "ready"
    assert (repository / case["selected"]).read_text() == case["source"]
    assert local._git(repository, "rev-parse", "HEAD") == baseline
    artifact = Path(candidate["artifact"])
    payload = json.loads(artifact.read_bytes())
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == candidate["sha256"]
    assert payload["proof_receipt_id"]

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
            context_bundle=case["context"]["context_bundle"], refresh_context_on_completion=True,
            published_retrieval_policy="lexical-tfidf-symbols@1")
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
            assert published and all(row["status"] == "refreshed"
                and row["retrieval_status"] == "current" for row in published)
            logs = list((runtime.state / "run/admitted_database_portal_attempts").rglob("*.log"))
            materialized = [json.loads(line) for path in logs for line in path.read_text().splitlines()
                if line.startswith('{"') and 'native-doctor-candidate-materialization@1' in line]
            assert len(materialized) == 1
            assert materialized[0]["status"] == "candidate_materialized"
            assert materialized[0]["provider_calls"] == 0
            assert materialized[0]["changed_paths"] == [case["selected"]]
            assert materialized[0]["source_after_sha256"] == payload["edits"][0]["after_sha256"]
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
            runtime.close()

    assert calls == [] and not marker.exists()
    assert subprocess.run(behavior, cwd=repository, capture_output=True, timeout=10).returncode == 0
    assert subprocess.run(list(task.validations[0].argv), cwd=repository,
                          capture_output=True, timeout=10).returncode == 0
    assert local._git(repository, "rev-parse", "HEAD") != baseline
    assert {path: (repository / path).read_bytes() for path in immutable} == immutable
    assert local._git(repository, "diff", "--name-only", baseline, "HEAD").splitlines() == [case["selected"]]
    assert (repository / case["selected"]).read_text() == case["after"]
