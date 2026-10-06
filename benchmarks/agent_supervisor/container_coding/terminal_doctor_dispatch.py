"""Automatic bounded Doctor selection before the native benchmark owner starts.

Only the existing signed task is considered. Unsupported repairs retain native
Doctor residual proposals and use the ordinary model route. A committed repair
is exported as a candidate for the existing worker/Portal validation path; this
module cannot publish source or complete the task.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import (
    execute_doctor_task_repair, prepare_doctor_task_repair,
)
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy
from .terminal_symbolic_capabilities import assess_terminal_symbolic_capabilities


def _installed_provers() -> tuple[Path, Path]:
    """Discover installed tools only; never download a toolchain during a task."""
    z3 = Path(shutil.which("z3") or "/unavailable/z3")
    nominated = os.environ.get("DOCTOR_COMPOSITION_LEAN", "")
    if nominated:
        return z3, Path(nominated)
    elan = shutil.which("elan")
    if elan:
        try:
            found = subprocess.run([elan, "which", "lean"], capture_output=True, text=True,
                                   timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if found.returncode == 0 and found.stdout.strip():
                return z3, Path(found.stdout.strip())
    return z3, Path("/unavailable/lean")


def _publish_handoff(*, repository: Path, result: dict) -> Path:
    """Expose candidate bytes, leaving receipts and owner authority private."""
    source = Path(result["handoff_path"])
    if source.is_symlink() or not source.is_file():
        raise ValueError("Doctor candidate handoff must be a regular file")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != result["handoff_sha256"] or json.loads(raw) != result["handoff"]:
        raise ValueError("Doctor handoff changed before worker publication")
    handoff = result["handoff"]
    if (handoff["repository"] != str(repository) or handoff["provider_calls"] != 0
            or handoff["publication_authority"] is not False
            or handoff["completion_authority"] is not False):
        raise ValueError("Doctor handoff repository or candidate-only authority differs")
    current = repository
    for component in (".runtime", "doctor-handoffs"):
        current = current / component
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022
                or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("Doctor handoff directory must be owner-controlled and worker-readable")
    target = current / (digest + ".json")
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def prepare_terminal_doctor_dispatch(*, repository: Path, state: Path,
                                    admission: dict, task_cid: str,
                                    contract_profile: str | None = None) -> dict:
    """Select from real diagnostics/AST and run native gates without model calls.

    Must run while the intent database is file-backed, before the native owner
    process opens it. Admission or source drift remains a hard refusal, rather
    than a reason to execute an unbound fallback.
    """
    repository = Path(repository).absolute()
    state = Path(state).absolute()
    verified = verify_local_benchmark_admission(admission, initial=True)
    if (repository.resolve(strict=True) != repository
            or str(repository) != verified["manifest"]["repository"]
            or state.resolve(strict=True) != state or state.is_relative_to(repository)):
        raise ValueError("Doctor dispatch requires exact admitted repository and external state")
    if [task.task_cid for task in verified["graph"].tasks] != [task_cid]:
        raise ValueError("Doctor dispatch requires the single admitted benchmark task")
    if contract_profile not in {None, "wsgi-header-controls@1"}:
        raise ValueError("unknown reviewed local contract profile")
    solver, kernel = _installed_provers()
    contract_result = None
    if contract_profile is not None:
        from ipfs_accelerate_py.agent_supervisor.analysis.doctor_header_contracts import WsgiHeaderProtocolContract
        from ipfs_accelerate_py.agent_supervisor.runtime.doctor_header_workflow import (
            CweReportOutput, prepare_header_contract_repair,
        )
        with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
            contract_result = prepare_header_contract_repair(repository=repository, admission=admission,
                intent=intent, task_cid=task_cid, state=state / "doctor-header-workflow",
                protocol=WsgiHeaderProtocolContract(review_ref="public-task-wsgi-header-profile@1"),
                lean=kernel, z3=solver, report_output=CweReportOutput("report.jsonl", "/app"))
        if contract_result["status"] == "candidate_ready":
            dispatch = {"schema": "terminal-doctor-dispatch@1", "status": "candidate_ready",
                "task_cid": task_cid, "route": "doctor_contract_candidate", "provider_calls": 0,
                "analysis_status": "available", "completion_authority": False,
                "publication_authority": False, "reason_codes": [], "residual_successors": 0,
                "residual_work_proposals": 0, "artifact": contract_result["artifact"],
                "sha256": contract_result["sha256"], "contract_workflow": contract_result,
                "result_artifact": str(state / "doctor-header-workflow/result.json")}
            dispatch["symbolic_capabilities"] = assess_terminal_symbolic_capabilities(
                manifest=admission["manifest"], task_cid=task_cid,
                task_spec=verified["manifest"]["tasks"][0], doctor_result=contract_result,
                prover_paths={"lean": kernel, "z3": solver}, contract_profile=contract_profile)
            return dispatch
    candidate_ref = "refs/heads/doctor/terminal-" + uuid.uuid4().hex
    base = verified["manifest"]["baseline_commit"]
    subprocess.run(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                    "update-ref", candidate_ref, base, "0" * len(base)],
                   check=True, capture_output=True, timeout=10)
    runtime = DeterministicDoctorRuntime(checkout_root=repository, index_root=state / "doctor-index",
        policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO))
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        prepared = prepare_doctor_task_repair(runtime=runtime, intent=intent, admission=admission,
            task_cid=task_cid, state_root=state / "doctor-workflow", solver_executable=solver,
            kernel_executable=kernel, candidate_ref=candidate_ref)
        result = execute_doctor_task_repair(prepared)
        residual_context = None
        if result["status"] == "residual":
            from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import prepare_doctor_residual_context
            residual_context = prepare_doctor_residual_context(prepared=prepared, result=result)
    dispatch = {"schema": "terminal-doctor-dispatch@1", "status": result["status"],
        "task_cid": task_cid, "route": "model_router", "provider_calls": 0,
        "analysis_status": result.get("analysis_status"),
        "analysis_observation_cid": result.get("analysis_observation", {}).get("observation_cid"),
        "completion_authority": False, "publication_authority": False,
        "result_artifact": str(state / "doctor-workflow-result.json"),
        "reason_codes": result.get("reason_codes", []),
        "residual_successors": len(result.get("plan_refill", {}).get("successors", [])),
        "residual_work_proposals": len(result.get("plan_refill", {}).get("work_proposals", []))}
    if residual_context is not None:
        dispatch["residual_context"] = residual_context
    if contract_result is not None:
        dispatch["contract_workflow"] = contract_result
    if result["status"] == "candidate_ready":
        artifact = _publish_handoff(repository=repository, result=result)
        dispatch.update(route="doctor_candidate", artifact=str(artifact),
                        sha256=result["handoff_sha256"], candidate_commit=result["handoff"]["candidate_commit"])
    elif result["status"] != "residual":
        raise ValueError("Doctor returned an unsupported dispatch state")
    dispatch["symbolic_capabilities"] = assess_terminal_symbolic_capabilities(
        manifest=admission["manifest"], task_cid=task_cid,
        task_spec=verified["manifest"]["tasks"][0], doctor_result=result,
        prover_paths={"lean": kernel, "z3": solver}, contract_profile=contract_profile)
    with (state / "doctor-workflow-result.json").open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return dispatch


def implementation_argv(*, router: Path, model: str, reasoning: str, timeout: int,
                        semantic_repository: Path | None, doctor: dict | None,
                        semantic_transport_schema: str = "supervisor-semantic-router-input@1",
                        coding_reply_mode: str = "legacy") -> list[str]:
    """Closed choice that the admitted launch signs before any worker starts."""
    from .terminal_semantic_transport_policy import DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, validate_semantic_transport_schema
    validate_semantic_transport_schema(semantic_transport_schema)
    from .terminal_coding_reply_policy import DEFAULT_CODING_REPLY_MODE, validate_coding_reply_mode
    validate_coding_reply_mode(coding_reply_mode)
    if coding_reply_mode != DEFAULT_CODING_REPLY_MODE and (semantic_repository is None
            or doctor is not None and doctor.get("route") != "model_router"):
        raise ValueError("ordinary completion requires a semantic model-router coding route")
    if (semantic_transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA
            and (semantic_repository is None or doctor is not None and doctor.get("route") != "model_router")):
        raise ValueError("controller dictionary transport requires a semantic model-router coding route")
    if doctor is not None and doctor.get("route") == "doctor_contract_candidate":
        if doctor.get("status") != "candidate_ready" or doctor.get("provider_calls") != 0:
            raise ValueError("proved local contract candidate required")
        return [str(router), "--doctor-contract-artifact", doctor["artifact"],
                "--doctor-contract-sha256", doctor["sha256"], "--doctor-contract-task-cid", doctor["task_cid"]]
    if doctor is not None and doctor.get("route") == "doctor_candidate":
        if doctor.get("status") != "candidate_ready" or doctor.get("provider_calls") != 0:
            raise ValueError("committed provider-free Doctor candidate required")
        return [str(router), "--doctor-candidate-artifact", doctor["artifact"],
                "--doctor-candidate-sha256", doctor["sha256"], "--doctor-task-cid", doctor["task_cid"]]
    if doctor is not None and (doctor.get("route") != "model_router" or doctor.get("status") != "residual"):
        raise ValueError("Doctor dispatch is neither a candidate nor a residual fallback")
    return [str(router), "--model", model, "--reasoning-effort", reasoning,
            "--timeout", str(timeout), "--max-output-tokens", "4096",
            *(["--semantic-repository", str(semantic_repository)] if semantic_repository is not None else []),
            *(["--semantic-transport-schema", semantic_transport_schema]
              if semantic_transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA else []),
            *(["--coding-reply-mode", coding_reply_mode]
              if coding_reply_mode != DEFAULT_CODING_REPLY_MODE else []),
            *(["--doctor-residual-artifact", doctor["residual_context"]["artifact"],
               "--doctor-residual-sha256", doctor["residual_context"]["sha256"],
               "--doctor-residual-task-cid", doctor["residual_context"]["task_cid"]]
              if doctor is not None and doctor.get("residual_context") is not None else [])]
