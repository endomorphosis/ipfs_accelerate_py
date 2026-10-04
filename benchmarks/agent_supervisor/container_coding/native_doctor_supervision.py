"""Provider-free native supervision of an automatically selected Doctor repair.

The task declaration is an authored qualification fixture. Selection, proof,
synthesis, candidate application, daemon validation/publication/completion and
post-stop semantic/world refresh use the actual implementations. This is not
a Terminal-Bench score or a claim of general repair or model planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def qualify(output: Path) -> dict:
    import duckdb
    from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify as index
    from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _native_diagnostics
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult
    from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import prepare_doctor_task_repair, execute_doctor_task_repair
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification requires a fresh exact directory")
    lean_pin = os.environ.get("DOCTOR_COMPOSITION_LEAN", "")
    if not shutil.which("z3") or (not lean_pin and not shutil.which("elan")):
        raise ValueError("actual installed Lean and Z3 are required")
    lean = Path(lean_pin or subprocess.check_output(["elan", "which", "lean"], text=True).strip())
    output.mkdir(parents=True)
    repository = output / "repository"
    repository.mkdir()
    source = "def process(amount):\n    return amount\n\ndef answer():\n    return process(count=2)\n"
    (repository / "answer.py").write_text(source)
    for args in (("init", "-q"), ("config", "user.name", "Native qualification"),
                 ("config", "user.email", "qualification@example.invalid"),
                 ("add", "answer.py"), ("commit", "-qm", "Public keyword mismatch")):
        subprocess.run(["git", "-C", str(repository), *args], check=True)
    with (repository / ".git/info/exclude").open("a") as stream:
        stream.write("\n.runtime/\n")
    baseline = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    candidate_ref = "refs/heads/doctor/automatic"
    subprocess.run(["git", "-C", str(repository), "update-ref", candidate_ref, baseline], check=True)
    check = ["python3", "-B", "-c", "from answer import answer; assert answer() == 2"]
    before = subprocess.run(check, cwd=repository, capture_output=True)
    if not before.returncode:
        raise RuntimeError("qualification must start with a failing public check")
    report = {"schema": "native-doctor-supervision-qualification@1", "qualified": False,
        "planning_source": "independently authored single-goal single-task declaration",
        "provider_calls": 0, "benchmark_result": False, "token_savings": None,
        "learned_embeddings": False, "observations": [], "baseline_commit": baseline,
        "initial_public_check_exit_code": before.returncode, "production_activation": False}
    started = time.monotonic()
    try:
        with IntentRepository(output / "intent.duckdb") as intent:
            declared = prepare_local_task(repository=repository, state=output / "policy", intent=intent,
                scope_paths=["answer.py"], output_path="answer.py", validation_argv=check,
                objective="Repair the direct local keyword call so answer() returns 2")
            cid = declared["task_cid"]
            vector = index(repository, output / "vectors", ["answer.py"], "answer process")
            with duckdb.connect(str(output / "vectors/vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
                snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
                    "SELECT payload FROM snapshots WHERE id=?", [vector["index_id"]]).fetchone()[0]))
            context = prepare_supervised_task_context(repository=repository, intent=intent, task_cid=cid,
                paths=["answer.py"], required_raw_paths=["answer.py"], output=repository / ".runtime/initial",
                code_vector_snapshot=snapshot, code_vector_result=CodeVectorSearchResult.from_dict(vector["hits"]),
                code_query_text=vector["query"])
            bundle = write_task_context_bundle(repository=repository, prepared=[context],
                output=repository / ".runtime/context-bundle.json")
            doctor = DeterministicDoctorRuntime(checkout_root=repository, index_root=output / "doctor-index",
                policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO))
            prepared = prepare_doctor_task_repair(runtime=doctor, intent=intent, admission=declared["admission"],
                task_cid=cid, state_root=output / "doctor", solver_executable=Path(shutil.which("z3")),
                kernel_executable=lean, candidate_ref=candidate_ref)
            candidate = execute_doctor_task_repair(prepared)
            _write(output / "doctor-result.json", candidate)
            if candidate["status"] != "candidate_ready":
                raise RuntimeError("automatic Doctor did not produce an admissible candidate")
            if (repository / "answer.py").read_text() != source or intent.get_task(cid)["status"] != "ready":
                raise RuntimeError("Doctor must leave canonical source and task pending")
            report.update(task_cid=cid, context_bundle=bundle, initial_context=context,
                vector_index_id=vector["index_id"], doctor_stages=candidate["stages"],
                doctor_handoff_sha256=candidate["handoff_sha256"],
                doctor_candidate_commit=candidate["handoff"]["candidate_commit"],
                pre_start_task_status="ready", canonical_unchanged_before_start=True)
        command = shlex.join([sys.executable, "-B", "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner",
            "--artifact", candidate["handoff_path"], "--sha256", candidate["handoff_sha256"], "--task-cid", cid])
        worktrees = output / "allocated-worktrees"
        worktrees.mkdir(mode=0o750)
        with open_existing_native_owner(database=output / "intent.duckdb", checkout=repository,
                state_dir=output / "owner", repository_id=declared["manifest"]["payload"]["repository_cid"],
                execution_routes={declared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
            runtime = AdmittedBenchmarkRuntime.create(output / "launch", admission=declared["admission"],
                server=owner.server, source=owner.source, implement=True, implementation_command=command,
                context_bundle=bundle, refresh_context_on_completion=True, max_task_attempts=1,
                published_retrieval_policy="lexical-tfidf-symbols@1",
                timeout_ms=30_000, lifetime_seconds=180, worker_worktree_root=worktrees)
            try:
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded":
                    raise RuntimeError("native START failed")
                deadline = time.monotonic() + 90
                previous = None
                while time.monotonic() < deadline:
                    task = owner.source.get_task(cid)
                    current = (task.status, task.revision)
                    if current != previous:
                        report["observations"].append({"status": task.status, "revision": task.revision,
                            "seconds": time.monotonic() - started})
                        _write(output / "progress.json", report)
                        previous = current
                    if task.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    if not runtime.process.snapshot(runtime.profile).members:
                        raise RuntimeError("native supervisor ended before task completion")
                    time.sleep(.25)
                report["task"] = {"status": task.status, "revision": task.revision}
                report["before_stop"] = runtime.observe()
            finally:
                try:
                    report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    report["bootstrap_receipts"] = runtime.bootstrap_receipts
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    if report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0:
                        refresh_started = time.monotonic()
                        report["after_stop"] = runtime.observe()
                        report["refresh_seconds"] = time.monotonic() - refresh_started
                finally:
                    runtime.close()
        after = subprocess.run(check, cwd=repository, capture_output=True)
        report["final_public_check_exit_code"] = after.returncode
        report["published_commit"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        report["published_source_sha256"] = hashlib.sha256((repository / "answer.py").read_bytes()).hexdigest()
        refreshed = report.get("after_stop", {}).get("published_context", [])
        report["qualified"] = bool(report.get("task", {}).get("status") == "completed"
            and report["remaining_processes"] == 0 and report["stop"]["status"] == "succeeded"
            and not after.returncode and report["published_commit"] != baseline
            and not report["bootstrap_errors"] and refreshed
            and all(row["status"] == "refreshed" and row["retrieval_status"] == "current" for row in refreshed))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
    finally:
        report["seconds"] = time.monotonic() - started
        _write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = qualify(args.output)
    print(json.dumps({key: result[key] for key in ("qualified", "seconds", "provider_calls")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
