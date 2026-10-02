"""Learned scalar contracts, checked repair proposals, and the native daemon.

This is a bounded qualification with independently authored task acceptance,
not Terminal-Bench or general automatic instruction decomposition. The repair
operator is selected from a live counterexample; its candidate must pass fresh
Security inference and Lake before the owner creates the native worker handoff.
Only the existing daemon's validation/publication gates can complete the task.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

INSTRUCTION = ("the runner must compute result; requires left > 0; "
    "ensures result = old(right) * old(left) and returned.")
SOURCE = "def derive(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"
DOMAINS = {name: dict(lower=-1, upper=1) for name in ("capacity", "threshold")}
PUBLIC_CHECK = ("from source import derive; "
    "observed = [derive(1, -1), derive(1, 0), derive(1, 1)]; "
    "assert all(type(value) is int for value in observed); "
    "assert observed == [-1, 0, 1], observed")


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(*, output: Path, intent_config: Path, security_config: Path, lake: Path) -> dict:
    import duckdb
    from .local_planning_qualification import prepare_local_task
    from .native_quack_qualification import open_existing_native_owner
    from .vector_index_preflight import qualify as index
    from .terminal_container_supervisor import _native_diagnostics
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import prepare_intent_384_selection
    from ipfs_accelerate_py.agent_supervisor.runtime.scalar_candidate_handoff import prepare_scalar_candidate_handoff
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification requires a fresh exact output directory")
    configs = [Path(value).resolve(strict=True) for value in (intent_config, security_config)]
    config_hashes = {str(path): _sha(path) for path in configs}
    intent_selection, security_selection = [json.loads(path.read_bytes()) for path in configs]
    weights = {row["checkpoint_path"]: _sha(Path(row["checkpoint_path"]))
        for row in (intent_selection, security_selection)}
    if any(weights[row["checkpoint_path"]] != row["checkpoint_sha256"] for row in (intent_selection, security_selection)):
        raise ValueError("exact checkpoint hashes required")
    if security_selection.get("schema") != "supervisor-security-source-program-384-config/v2":
        raise ValueError("explicit finite-state Security384 selection required")
    security_selection = deepcopy(security_selection)
    security_selection.update(finite_state_domains={"source.py": deepcopy(DOMAINS)}, lake=None)
    lake = Path(lake).resolve(strict=True)
    effects = dict(schema="supervisor-intent-code-effect-config/v2", contracts=[dict(
        id="declared-return", source_id="source.py", action_id="action",
        input_parameter_mapping={"left": "capacity", "right": "threshold"}, input_domains=deepcopy(DOMAINS))],
        lake=dict(executable=str(lake), timeout_seconds=60))
    output.mkdir(parents=True)
    _write(output / "inputs.json", dict(instruction=INSTRUCTION, source_text=SOURCE,
        intent_config=intent_selection, security_config=security_selection, effect_config=effects,
        public_validation=PUBLIC_CHECK, source_configs_sha256=config_hashes, checkpoint_sha256=weights))
    report = dict(schema="native-scalar-supervision-qualification/v1", qualified=False,
        provider_calls=0, provider_tokens=0, benchmark_result=False, token_savings=None,
        task_decomposition="independently authored local goal and task",
        repair_selection="live bounded Intent counterexample and fresh candidate inference/Lake",
        doctor_keyword_composition_used=False, production_activation=False,
        source_executed_for_public_validation=True, source_executed_during_symbolic_analysis=False,
        observations=[])
    started = time.monotonic()
    try:
        # Actual published inference precedes the independently authored graph.
        advice, selection, elapsed = prepare_intent_384_selection(instruction=INSTRUCTION, config_path=configs[0])
        _write(output / "intent-advice.json", advice)
        if advice["status"] != "semantic_candidate_advice":
            raise RuntimeError("controlled fixture did not produce supported learned Intent")
        report["preplanning"] = dict(selection=selection, status=advice["status"],
            seconds=elapsed / 1e9, before_goal_decomposition=True,
            checkpoint_sha256=advice["checkpoint_sha256"], numerical_replay_verified=advice["numerical_replay_verified"])
        repository = output / "repository"; repository.mkdir()
        (repository / "source.py").write_text(SOURCE)
        for args in (("init", "-q"), ("config", "user.name", "Scalar qualification"),
                ("config", "user.email", "qualification@example.invalid"), ("add", "source.py"),
                ("commit", "-qm", "Independent scalar acceptance fixture")):
            subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
        with (repository / ".git/info/exclude").open("a") as stream: stream.write("\n.runtime/\n")
        # The worker-readable handoff parent is owner-controlled even under a
        # group-writable development umask. Full advice stays in external state.
        (repository / ".runtime").mkdir(mode=0o755)
        baseline = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        # The native validation owner resolves this sealed launcher itself.
        check = ["python3", "-B", "-c", PUBLIC_CHECK]
        before = subprocess.run(check, cwd=repository, capture_output=True, timeout=10)
        report.update(baseline_commit=baseline, initial_public_check_exit_code=before.returncode)
        if not before.returncode:
            raise RuntimeError("qualification must start with failing independent acceptance")
        with IntentRepository(output / "intent.duckdb") as intent:
            declared = prepare_local_task(repository=repository, state=output / "policy", intent=intent,
                scope_paths=["source.py"], output_path="source.py", validation_argv=check, objective=INSTRUCTION)
            cid = declared["task_cid"]
            vector = index(repository, output / "vectors", ["source.py"], "derive capacity threshold")
            with duckdb.connect(str(output / "vectors/vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
                snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
                    "SELECT payload FROM snapshots WHERE id=?", [vector["index_id"]]).fetchone()[0]))
            context = prepare_supervised_task_context(repository=repository, intent=intent, task_cid=cid,
                paths=["source.py"], required_raw_paths=["source.py"], output=repository / ".runtime/initial",
                code_vector_snapshot=snapshot, code_vector_result=CodeVectorSearchResult.from_dict(vector["hits"]),
                code_query_text=vector["query"], security_source_program_config=security_selection,
                intent_code_effect_instruction=INSTRUCTION, intent_code_effect_intent_advice=advice,
                intent_code_effect_config=effects)
            baseline_effect = context["intent_code_effect_advice"]
            if (baseline_effect.get("live_build_verified") is not True
                    or baseline_effect["rows"][0]["effect_status"] != "refuted"):
                raise RuntimeError("initial task context did not verify the bounded counterexample")
            bundle = write_task_context_bundle(repository=repository, prepared=[context],
                output=repository / ".runtime/context-bundle.json")
            candidate = prepare_scalar_candidate_handoff(repository=repository, admission=declared["admission"],
                intent=intent, task_cid=cid, state=output / "scalar-repair", instruction=INSTRUCTION,
                intent_config=intent_selection, security_config=security_selection, effect_config=effects)
            _write(output / "candidate-result.json", candidate)
            if candidate["status"] != "candidate_ready":
                raise RuntimeError("bounded scalar repair did not produce a unique checked candidate")
            if (repository / "source.py").read_text() != SOURCE or intent.get_task(cid)["status"] != "ready":
                raise RuntimeError("candidate preparation changed canonical source or task state")
            report.update(task_cid=cid, context_bundle=bundle, initial_context=context,
                vector_index_id=vector["index_id"], candidate_handoff_sha256=candidate["handoff_sha256"],
                pre_start_task_status="ready", canonical_unchanged_before_start=True)
        command = shlex.join([sys.executable, "-B", "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.scalar_candidate_runner",
            "--artifact", candidate["handoff_path"], "--sha256", candidate["handoff_sha256"], "--task-cid", cid])
        worktrees = output / "allocated-worktrees"; worktrees.mkdir(mode=0o750)
        with open_existing_native_owner(database=output / "intent.duckdb", checkout=repository,
                state_dir=output / "owner", repository_id=declared["manifest"]["payload"]["repository_cid"],
                execution_routes={declared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
            runtime = AdmittedBenchmarkRuntime.create(output / "launch", admission=declared["admission"],
                server=owner.server, source=owner.source, implement=True, implementation_command=command,
                context_bundle=bundle, refresh_context_on_completion=True, max_task_attempts=1,
                published_retrieval_policy="lexical-tfidf-symbols@1", timeout_ms=30_000,
                lifetime_seconds=180, worker_worktree_root=worktrees)
            try:
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded": raise RuntimeError("native START failed")
                deadline = time.monotonic() + 90
                previous = None
                while time.monotonic() < deadline:
                    task = owner.source.get_task(cid)
                    current = (task.status, task.revision)
                    if current != previous:
                        report["observations"].append(dict(status=task.status, revision=task.revision,
                            seconds=time.monotonic() - started))
                        _write(output / "progress.json", report); previous = current
                    if task.status in {"completed", "failed", "blocked", "cancelled"}: break
                    if not runtime.process.snapshot(runtime.profile).members:
                        raise RuntimeError("native supervisor ended before task completion")
                    time.sleep(.25)
                report["task"] = dict(status=task.status, revision=task.revision)
                report["before_stop"] = runtime.observe()
            finally:
                try:
                    report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    report["bootstrap_errors"] = runtime.bootstrap_errors
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    if report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0:
                        report["after_stop"] = runtime.observe()
                finally:
                    runtime.close()
        after = subprocess.run(check, cwd=repository, capture_output=True, timeout=10)
        report["final_public_check_exit_code"] = after.returncode
        report["published_commit"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        report["published_source_sha256"] = _sha(repository / "source.py")
        report["published_source_matches_checked_candidate"] = (
            report["published_source_sha256"] == candidate["handoff"]["edit"]["after_sha256"])
        report["configuration_and_weights_unchanged"] = all(_sha(Path(path)) == digest
            for path, digest in {**config_hashes, **weights}.items())
        refreshed = report.get("after_stop", {}).get("published_context", [])
        report["qualified"] = bool(report.get("task", {}).get("status") == "completed"
            and report["remaining_processes"] == 0 and report["stop"]["status"] == "succeeded"
            and not after.returncode and report["published_commit"] != baseline
            and report["published_source_matches_checked_candidate"]
            and not report["bootstrap_errors"] and refreshed and report["configuration_and_weights_unchanged"]
            and all(row["status"] == "refreshed" and row["retrieval_status"] == "current" for row in refreshed))
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error)[:2048])
    finally:
        report["seconds"] = time.monotonic() - started
        _write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--intent-config", required=True, type=Path)
    parser.add_argument("--security-config", required=True, type=Path)
    parser.add_argument("--lake", required=True, type=Path)
    result = qualify(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("qualified", "seconds", "provider_calls")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
