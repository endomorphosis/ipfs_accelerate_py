"""Provider-free native supervision of an authored local header-contract task.

Exercises real scoped diagnostics, Lean/Z3, proof/world indexes, the allocated
worker, native validation/publication/completion, and post-STOP context refresh.
The task declaration and public regression check are qualification fixtures,
not a Terminal-Bench score or a whole-program security proof.
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
import traceback


SOURCE = '''def normalize_label(value):
    text = str(value)
    return text.title().replace('_', '-')

def normalize_payload(value):
    text = str(value)
    return text

class HeaderRecord:
    def __init__(self):
        self._fields = {}

    def put(self, label, payload):
        self._fields[normalize_label(label)] = [normalize_payload(payload)]

    @property
    def wire_headers(self):
        pairs = list(self._fields.items())
        return [(label, payload) for label, payloads in pairs for payload in payloads]

def application(environ, start_response):
    response = HeaderRecord()
    start_response('200 OK', response.wire_headers)
'''

PUBLIC_CHECK = '''from source import HeaderRecord, normalize_label, normalize_payload

def check():
    for value in ('', 'hello', 'éλ', 17):
        assert normalize_label(value) == str(value).title().replace('_', '-')
        assert normalize_payload(value) == str(value)
    response = HeaderRecord()
    response.put('x_header', 'accepted')
    assert response.wire_headers == [('X-Header', 'accepted')]
    for control in ('\\r', '\\n', '\\x00'):
        for prefix in ('', 'ordinary', 'éλ'):
            for helper in (normalize_label, normalize_payload):
                try:
                    helper(prefix + control + 'tail')
                except ValueError:
                    pass
                else:
                    raise AssertionError('forbidden converted input was accepted')
check()
'''


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def qualify(output: Path, *, security_checkpoint: dict | None = None,
            formula_decoder: dict | None = None, startup_timeout_ms: int = 30_000) -> dict:
    import duckdb
    from .local_planning_qualification import prepare_local_task
    from .native_quack_qualification import open_existing_native_owner
    from .vector_index_preflight import qualify as index
    from .terminal_container_supervisor import _native_diagnostics
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult
    from ipfs_accelerate_py.agent_supervisor.analysis.doctor_header_contracts import WsgiHeaderProtocolContract
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_header_workflow import prepare_header_contract_repair
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

    output = Path(output).absolute()
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import get_operation_catalog, Operation
    catalog_limit = min(get_operation_catalog().operation(operation).bounds.timeout_ms
                        for operation in (Operation.START, Operation.STOP))
    if type(startup_timeout_ms) is not int or not 2_000 <= startup_timeout_ms <= catalog_limit:
        raise ValueError("bounded explicit native startup timeout required")
    if formula_decoder is not None and security_checkpoint is None:
        raise ValueError("formula decoder qualification requires frozen security advice")
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification requires a fresh exact directory")
    lean_pin = os.environ.get("DOCTOR_COMPOSITION_LEAN", "")
    if not shutil.which("z3") or (not lean_pin and not shutil.which("elan")):
        raise ValueError("actual installed Lean and Z3 are required")
    lean = Path(lean_pin or subprocess.check_output(["elan", "which", "lean"], text=True).strip())
    output.mkdir(parents=True)
    repository = output / "repository"
    repository.mkdir()
    (repository / "source.py").write_text(SOURCE)
    (repository / "public_check.py").write_text(PUBLIC_CHECK)
    (repository / "docs").mkdir()
    (repository / "docs/example.rst").write_text('Authored public example:\npassword = "AUTHORED-DUMMY-NOT-A-CREDENTIAL"\n')
    security_advice = None
    security_preparation_seconds = 0.0
    if security_checkpoint is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import (
            prepare_security_advice, validate_security_advice,
        )
        security_started = time.monotonic()
        security_paths = ["source.py", "public_check.py"]
        source_hashes = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                         for name in security_paths}
        security_advice = prepare_security_advice(repository=repository,
            paths=security_paths, source_hashes=source_hashes, checkpoint=security_checkpoint,
            output=output / "frozen-security-initial", formula_decoder=formula_decoder,
            header_protocol=({"review_ref": "authored native header supervision SOURCE fixture",
                "callback_parameter": "start_response"} if formula_decoder is not None else None))
        summary = validate_security_advice(repository=repository, expected_receipt=security_advice)
        # The independently signed task and its raw context contain the actual
        # source-bound advice; model scores never change the reviewed contract.
        _write(repository / "frozen-security-advice.json", summary)
        (repository / "frozen-security-advice.json").chmod(0o444)
        security_preparation_seconds = time.monotonic() - security_started
    for args in (("init", "-q"), ("config", "user.name", "Native header qualification"),
                 ("config", "user.email", "qualification@example.invalid"),
                 ("add", "."), ("commit", "-qm", "Authored public header task")):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    with (repository / ".git/info/exclude").open("a") as stream:
        stream.write("\n.runtime/\n")
    (repository / ".runtime").mkdir(mode=0o755)
    baseline = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    check = ["python3", "-B", "public_check.py"]
    before = subprocess.run(check, cwd=repository, capture_output=True, timeout=10)
    if not before.returncode:
        raise RuntimeError("qualification must start with a failing public check")
    report = {"schema": "native-header-contract-supervision-qualification@1", "qualified": False,
        "planning_source": "independently authored single-goal single-task declaration",
        "provider_calls": 0, "benchmark_result": False, "token_savings": None,
        "learned_embeddings": False, "observations": [], "baseline_commit": baseline,
        "initial_public_check_exit_code": before.returncode, "production_activation": False,
        "whole_program_proved": False}
    report["startup_timeout_ms"] = startup_timeout_ms
    if security_advice is not None:
        report.update(frozen_security_initial=security_advice,
            frozen_security_preparation_seconds=security_preparation_seconds,
            frozen_security_instruction="candidate advice only; all source remains eligible for analysis")
    started = time.monotonic()
    try:
        with IntentRepository(output / "intent.duckdb") as intent:
            paths = ["source.py", "public_check.py"]
            objective = "Reject CR LF NUL after header conversion and preserve accepted normalization"
            if security_advice is not None:
                paths.append("frozen-security-advice.json")
                objective += (". Frozen security advice is declared read-only input for analysis ordering; "
                              "its scores grant no omission, proof or completion authority")
            declared = prepare_local_task(repository=repository, state=output / "policy", intent=intent,
                scope_paths=paths, output_path="source.py", validation_argv=check,
                objective=objective)
            cid = declared["task_cid"]
            vector = index(repository, output / "vectors", paths, "normalize label payload")
            with duckdb.connect(str(output / "vectors/vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
                snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
                    "SELECT payload FROM snapshots WHERE id=?", [vector["index_id"]]).fetchone()[0]))
            context = prepare_supervised_task_context(repository=repository, intent=intent, task_cid=cid,
                paths=paths, required_raw_paths=paths, output=repository / ".runtime/initial",
                code_vector_snapshot=snapshot, code_vector_result=CodeVectorSearchResult.from_dict(vector["hits"]),
                code_query_text=vector["query"])
            bundle = write_task_context_bundle(repository=repository, prepared=[context],
                output=repository / ".runtime/context-bundle.json")
            candidate = prepare_header_contract_repair(repository=repository, admission=declared["admission"],
                intent=intent, task_cid=cid, state=output / "doctor",
                protocol=WsgiHeaderProtocolContract("authored-qualification-reviewed-wsgi@1"),
                lean=lean, z3=Path(shutil.which("z3")))
            _write(output / "doctor-result.json", candidate)
            if candidate["status"] != "candidate_ready":
                raise RuntimeError("local contract Doctor did not produce a candidate")
            if (repository / "source.py").read_text() != SOURCE or intent.get_task(cid)["status"] != "ready":
                raise RuntimeError("Doctor must leave canonical source and task pending")
            report.update(task_cid=cid, context_bundle=bundle, initial_context=context,
                vector_index_id=vector["index_id"], doctor_proof=candidate["proof"],
                security_ir=candidate["security_ir"],
                contract_index=candidate["contract_index"], scoped_analysis=candidate["analysis"],
                doctor_artifact_sha256=candidate["sha256"],
                pre_start_task_status="ready", canonical_unchanged_before_start=True)
        command = shlex.join([sys.executable, "-B", "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner",
            "--artifact", candidate["artifact"], "--sha256", candidate["sha256"], "--task-cid", cid])
        worktrees = output / "allocated-worktrees"
        worktrees.mkdir(mode=0o750)
        with open_existing_native_owner(database=output / "intent.duckdb", checkout=repository,
                state_dir=output / "owner", repository_id=declared["manifest"]["payload"]["repository_cid"],
                execution_routes={declared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
            runtime = AdmittedBenchmarkRuntime.create(output / "launch", admission=declared["admission"],
                server=owner.server, source=owner.source, implement=True, implementation_command=command,
                context_bundle=bundle, refresh_context_on_completion=True, max_task_attempts=1,
                published_retrieval_policy="lexical-tfidf-symbols@1",
                timeout_ms=startup_timeout_ms, lifetime_seconds=300, worker_worktree_root=worktrees)
            try:
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded":
                    raise RuntimeError("native START failed")
                deadline, previous = time.monotonic() + 90, None
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
                    report["startup_health"] = {
                        "bootstrap_receipt_count": len(runtime.bootstrap_receipts),
                        "last_scope_mismatch": getattr(runtime.process, "last_scope_mismatch", ""),
                        "last_heartbeat_evidence": getattr(runtime.process, "last_heartbeat_evidence", {}),
                    }
                    report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    if report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0:
                        report["after_stop"] = runtime.observe()
                finally:
                    runtime.close()
        after = subprocess.run(check, cwd=repository, capture_output=True, timeout=10)
        report["final_public_check_exit_code"] = after.returncode
        report["published_commit"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        report["published_source_sha256"] = hashlib.sha256((repository / "source.py").read_bytes()).hexdigest()
        if security_advice is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import refresh_security_advice
            from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import load_security_checkpoint
            stale_rejected = False
            try:
                validate_security_advice(repository=repository, expected_receipt=security_advice)
            except ValueError:
                stale_rejected = True
            if not stale_rejected:
                raise RuntimeError("published source must invalidate its prior security observations")
            updated = refresh_security_advice(repository=repository, previous=security_advice,
                output=output / "frozen-security-refreshed")
            validate_security_advice(repository=repository, expected_receipt=updated)
            checked = load_security_checkpoint(Path(security_checkpoint["output"]),
                expected_manifest_sha256=security_checkpoint["manifest_sha256"])
            if checked["descriptor"] != security_checkpoint or updated["checkpoint"] != security_checkpoint:
                raise RuntimeError("source refresh must preserve the exact frozen checkpoint")
            if any(updated[key] != 0 for key in ("training_steps", "download_calls", "provider_calls")):
                raise RuntimeError("offline frozen refresh performed unexpected work")
            if formula_decoder is not None:
                from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import load_security_formula_decoder
                load_security_formula_decoder(formula_decoder)
                if security_advice["formula_decoder"] != formula_decoder or updated["formula_decoder"] != formula_decoder:
                    raise RuntimeError("source refresh must preserve the exact frozen formula decoder")
                if (updated["formalization"]["report_cid"] == security_advice["formalization"]["report_cid"]
                        or updated["record"]["formalization_plan_sha256"] == security_advice["record"]["formalization_plan_sha256"]):
                    raise RuntimeError("published source must refresh its formalization and proof-work plan")
                if any(not advice["hydration"]["ducklake_verified"] or advice["hydration"]["catalog_count"] != 6
                       for advice in (security_advice, updated)):
                    raise RuntimeError("formal advice must retain all six verified DuckLake catalogs")
                report.update(frozen_formula_decoder_unchanged=True,
                    frozen_formula_plan_refreshed=True, frozen_formula_ducklake_verified=True)
            report.update(frozen_security_refreshed=updated,
                frozen_security_prior_observation_rejected=True, frozen_security_checkpoint_unchanged=True,
                frozen_security_training_steps=0, frozen_security_download_calls=0)
        refreshed = report.get("after_stop", {}).get("published_context", [])
        report["qualified"] = bool(report.get("task", {}).get("status") == "completed"
            and report["remaining_processes"] == 0 and report["stop"]["status"] == "succeeded"
            and not after.returncode and report["published_commit"] != baseline
            and not report["bootstrap_errors"] and refreshed
            and all(row["status"] == "refreshed" and row["retrieval_status"] == "current" for row in refreshed))
    except Exception as error:
        # Avoid exporting exception messages that could contain source material.
        report["error"] = {"type": type(error).__name__, "frames": [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)]}
    finally:
        report["seconds"] = time.monotonic() - started
        _write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--security-checkpoint", type=Path)
    parser.add_argument("--security-checkpoint-manifest-sha256")
    parser.add_argument("--formula-decoder-descriptor", type=Path)
    parser.add_argument("--startup-timeout-ms", type=int, default=30_000)
    args = parser.parse_args()
    if bool(args.security_checkpoint) != bool(args.security_checkpoint_manifest_sha256):
        parser.error("security checkpoint requires both a package and its independently pinned manifest SHA256")
    checkpoint = None
    if args.security_checkpoint:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import load_security_checkpoint
        checkpoint = load_security_checkpoint(args.security_checkpoint.absolute(),
            expected_manifest_sha256=args.security_checkpoint_manifest_sha256)["descriptor"]
    decoder = None
    if args.formula_decoder_descriptor:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import load_security_formula_decoder
        decoder = json.loads(_read(args.formula_decoder_descriptor.absolute(), 32_000))
        load_security_formula_decoder(decoder)
    result = qualify(args.output, security_checkpoint=checkpoint, formula_decoder=decoder,
                     startup_timeout_ms=args.startup_timeout_ms)
    print(json.dumps({key: result[key] for key in ("qualified", "seconds", "provider_calls")}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
