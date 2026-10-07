"""Native public-task context for an explicitly administrative metadata census.

This provider-free preparation uses the real parser, signed admission, owner
materialization, deterministic indexing and pending-task prompt builder. It
does not plan with an LLM, implement, validate, settle or evaluate the task.
Use only a fresh owned task copy; operational prompt/state artifacts stay private.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from . import terminal_indexed_preparation as prep
from .local_live_planner import preflight_proposal
from .terminal_task_profile import SCHEMA as TASK_SCHEMA, validate_task_profile


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _administrative_graph(prepared):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        _select_evidence, parse_prompt_goal_graph,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptWorkflowRequest, DirectoryScanReceipt,
    )
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    scan = DirectoryScanReceipt.from_dict(prepared["scan"])
    config = prep._config(Path(prepared["repository"]),
        timeout_seconds=prepared["planner_timeout_seconds"], provider_profile=prepared.get("provider_profile"))
    proposal = preflight_proposal({"spec": prepared["spec"], "evidence": _select_evidence(request, scan, config)})
    proposal["root_goal_key"] = "TB-GOAL"
    proposal["goals"][0].update(goal_key="TB-GOAL", title="Public task administration",
                              objective="Fulfill the public instruction")
    proposal["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
                              title="Declared public task", objective="Fulfill the public instruction")
    proposal["tasks"][0].update(task_key=prepared["spec"]["task_key"], goal_key="TB-SUBGOAL",
        objective="Fulfill the public instruction",
        predicted_files=sorted(row["path"] for row in prepared["spec"]["outputs"]))
    return parse_prompt_goal_graph(json.dumps(proposal), request, scan, config=config,
                                   constraint_summaries=prepared["constraints"])


def _source384_metadata(path):
    if path is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_config import (
        MAX_CONFIG_BYTES, _regular_bytes, validate_source384_config,
    )
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _loads
    raw = _regular_bytes(path, MAX_CONFIG_BYTES)
    config = validate_source384_config(_loads(raw.decode("utf-8")))
    return {"config_sha256": _sha(raw), "config_bytes": len(raw), "schema": config["schema"],
        "checkpoint_sha256": config["checkpoint_sha256"], "embedding_revision": config["embedding_revision"],
        "declared_embedding_asset_count": len(config["embedding_assets"]),
        "config_shape_verified": True, "assets_verified": False, "used_for_index": False,
        "inference_performed": False, "training_steps": 0, "download_calls": 0}


def _private_prompt(path, prompt):
    with Path(path).open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(prompt)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)


def _doctor_residual_only(*, repository, state, admission, task_cid):
    """Prepare actual native diagnostics; reject a repair before execution."""
    from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import DoctorMode
    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import DeterministicDoctorRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_task_workflow import prepare_doctor_task_repair
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import prepare_doctor_residual_context
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import DeterministicDoctorPolicy
    from .terminal_doctor_dispatch import _installed_provers

    verified = local.verify_local_benchmark_admission(admission, initial=True)
    base = verified["manifest"]["baseline_commit"]
    candidate_ref = "refs/heads/doctor/metadata-census-" + uuid.uuid4().hex
    prep._git(repository, "-c", "core.hooksPath=/dev/null", "update-ref", candidate_ref, base, "0" * len(base))
    try:
        solver, kernel = _installed_provers()
        runtime = DeterministicDoctorRuntime(checkout_root=repository, index_root=state / "doctor-index",
            policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.REPORT_ONLY))
        with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
            prepared = prepare_doctor_task_repair(runtime=runtime, intent=intent, admission=admission,
                task_cid=task_cid, state_root=state / "doctor-workflow", solver_executable=solver,
                kernel_executable=kernel, candidate_ref=candidate_ref)
            if (prepared.inputs is not None or prepared.report.get("status") != "residual"
                    or prepared.report.get("provider_calls") != 0):
                raise ValueError("census refuses a prepared Doctor repair before native execution")
            residual = prepare_doctor_residual_context(prepared=prepared, result=dict(prepared.report))
            return {"status": "residual", "route": "model_router", "provider_calls": 0,
                "reason_codes": list(prepared.report.get("reason_codes", [])), "residual_context": residual,
                "repair_execution_permitted": False}
    finally:
        prep._git(repository, "-c", "core.hooksPath=/dev/null", "update-ref", "-d", candidate_ref, base)


@contextmanager
def prepare_terminal_metadata_preflight(
    *, repository: Path, instruction: Path, state: Path, task_profile: dict,
    resource_profile=None, source384_config: Path | None = None,
    include_doctor_context: bool = True,
):
    """Yield private census inputs while a native detached worktree is owned.

    Layout is a fresh owned setup directory with sibling ``repository`` and
    ``state`` children. The caller supplies only declared public input copies
    with one clean Git baseline. Both declared program/support bodies are
    processed by native owner APIs; no body belongs in the exported census.
    Source384 selection is recorded as metadata only. It cannot certify that
    this deterministic administrative context matches a live Source384 run.
    """
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _bounded_json
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalTask, TodoImplementationDaemon,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import managed_git_worktree
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import prepare_public_instruction_context
    from .indexed_doctor_lifecycle import _context
    import shlex

    if type(include_doctor_context) is not bool or type(task_profile) is not dict:
        raise ValueError("explicit public profile and Doctor context selection required")
    profile = validate_task_profile(json.loads(_bounded_json(task_profile, 65_536)))
    if profile["schema"] != TASK_SCHEMA:
        raise ValueError("metadata preflight supports one explicit public Python task profile")
    root, state = Path(repository).absolute(), Path(state).absolute()
    if (root.name != "repository" or state.name != "state" or root.parent != state.parent
            or root.resolve(strict=True) != root or state.resolve() != state or state.exists()):
        raise ValueError("fresh sibling repository/state setup required")
    owner = root.parent.stat()
    if owner.st_uid != os.geteuid() or stat.S_IMODE(owner.st_mode) & 0o022:
        raise ValueError("owned census setup directory required")
    git_root = root / ".git"
    if (git_root.is_symlink() or not git_root.is_dir() or git_root.stat().st_uid != os.geteuid()
            or prep._git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root)
            or prep._git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip() != str(git_root)):
        raise ValueError("fresh census copy must own its independent Git common directory")
    if prep._git(root, "status", "--porcelain=v1").strip():
        raise ValueError("clean copied public input baseline required")
    if prep._git(root, "rev-list", "--count", "HEAD").strip() != b"1":
        raise ValueError("fresh copied public input Git history required")
    tracked = set(prep._git(root, "ls-files", "-z").decode().split("\0")) - {""}
    if tracked != set(profile["input_paths"]):
        raise ValueError("copied baseline must contain exactly the declared public inputs")
    inputs_before = local._sources(root, profile["input_paths"], max_files=256)
    original_head = prep._git(root, "rev-parse", "HEAD").decode().strip()
    source384 = _source384_metadata(source384_config)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, resource_profile=resource_profile, disable_intent_autoencoder=True)
    state.chmod(0o700)
    graph = _administrative_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(state / "admission.json", admission)
    with IntentRepository(state / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        reference = intent.get_plan(materialized["plan_id"])["body"]["local_planning_receipt_ref"]
        if local.load_local_planning_receipt(reference, manifest=prepared["manifest"]) != admission["receipt"]:
            raise ValueError("native owner did not retain the exact signed planning receipt")
    indexed = prep.context(state=state, model_snapshot=None, model_revision="")
    # These newly owned advisory directories must satisfy the native worker
    # readability gate even when the caller uses a restrictive private umask.
    for runtime_dir in (root / ".runtime", root / ".runtime/doctor-residuals",
                        root / ".runtime/router-public-instruction"):
        runtime_dir.mkdir(mode=0o755, exist_ok=True)
        descriptor = os.open(runtime_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if os.fstat(descriptor).st_uid != os.geteuid():
                raise ValueError("fresh runtime artifact storage must remain owner controlled")
            os.fchmod(descriptor, 0o755)
        finally:
            os.close(descriptor)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    task_cids = materialized["task_cids"]
    if len(task_cids) != 1:
        raise ValueError("one independently declared census task required")
    task_cid = task_cids[0]
    doctor = None
    if include_doctor_context:
        doctor = _doctor_residual_only(repository=root, state=state, admission=admission, task_cid=task_cid)
    local.verify_local_benchmark_admission(admission, initial=True)
    instruction_context = prepare_public_instruction_context(repository=root, admission=admission,
        task_cid=task_cid, source_path=prep.INSTRUCTION,
        expected_source_sha256=verified["manifest"]["sources"][prep.INSTRUCTION]["sha256"])
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task = intent.get_task(task_cid)
        if task["status"] != "ready":
            raise ValueError("census task must remain pending and unclaimed")
        spec = prepared["spec"]
        portal = PortalTask(task_id=task["task_alias"], title=task["body"]["title"],
            canonical_task_cid=task_cid, status="ready", completion="manual", priority="P2", track="implementation",
            outputs=[row["path"] for row in spec["outputs"]],
            validation=[shlex.join(row["argv"]) for row in spec["validations"]],
            acceptance=" ; ".join(row["criterion"] for row in spec["acceptance"]),
            metadata={"database task cid": task_cid,
                "local planning contract cid": local.content_identity(task["body"][local.CONTRACT_KEY])})
        daemon = TodoImplementationDaemon(todo_path=root / ".runtime/metadata-preflight.todo.md",
            state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
            events_path=state / "worker/events.jsonl", repo_root=root, task_header_prefix="## TB-")
        daemon._world_intent_repository = intent
        daemon._task_context_nomination_bundle = indexed["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            retrieval, world = _context(prompt, "code-retrieval-context"), _context(prompt, "intent-world-context")
            if (retrieval["status"] != "current" or retrieval["index_id"] != indexed["index_id"]
                    or world["semantic_root_cid"] != indexed["semantic_root_cid"]):
                raise ValueError("native pending prompt did not consume current contexts")
        finally:
            daemon.close_event_runtime()
        current_task = intent.get_task(task_cid)
        contract, _, _, _ = local._contract(current_task["body"], task_cid)
        if contract["manifest"] != admission["manifest"] or current_task != task:
            raise ValueError("native census task contract or readiness changed")
    if local._sources(root, profile["input_paths"], max_files=256) != inputs_before:
        raise ValueError("copied public task inputs changed during census preparation")
    prompt_path = state / "worker-prompt.txt"
    _private_prompt(prompt_path, prompt)
    report = {"schema": "terminal-public-metadata-preflight@1", "observation_only": True,
        "graph_provenance": "explicitly-authored-administrative-plan", "planning_performed": False,
        "live_benchmark_planning_remains_enabled": True, "provider_calls": 0, "model_runs": 0,
        "model_runs_scope": "provider-and-pretrained-neural-inference-only",
        "implementation_executed": False, "validation_executed": False, "official_verifier_executed": False,
        "official_reward": None, "source384_equivalence_claimed": False, "source384_controls": source384,
        "index_configuration": "native-lexical-tfidf-symbols@1", "learned_embeddings": False,
        "native_lexical_vectorizer_fitting_performed": indexed["indexed_symbols"] > 0,
        "training_steps_scope": "neural-model-training-only",
        "source384_inference_performed": False, "training_steps": 0, "download_calls": 0,
        "public_input_bodies_processed_by_native_owner": True,
        "agent_evaluator_body_inspection": False, "parent_evaluator_body_inspection": False,
        "hidden_evaluator_reads": False, "solution_reads": False, "provider_response_reads": False,
        "existing_private_database_reads": 0, "fresh_native_owner_state_created": True,
        "original_copied_input_hashes_preserved": True, "original_copy_head": original_head,
        "source_count": len(verified["manifest"]["sources"]), "public_input_count": len(profile["input_paths"]),
        "graph_goal_count": len(graph.goals), "graph_task_count": len(graph.tasks),
        "manifest_cid": verified["receipt"]["manifest_cid"], "graph_cid": graph.content_id,
        "native_task_cid": task_cid, "native_task_revision": task["revision"],
        "semantic_root_cid": indexed["semantic_root_cid"], "index_id": indexed["index_id"],
        "indexed_symbol_count": indexed["indexed_symbols"], "full_capsule_count": indexed["full_capsules"],
        "native_fact_rows_replayed": indexed["native_fact_rows_replayed"],
        "worker_capsule_count": indexed["worker_capsules"], "native_prompt_sha256": _sha(prompt.encode()),
        "native_prompt_bytes": len(prompt.encode()), "public_instruction_artifact_sha256": instruction_context["sha256"],
        "doctor_residual_included": bool(doctor and doctor.get("residual_context")),
        "token_savings_measured": False, "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "publication_authority": False, "required_fact_omission_authority": False,
        "post_yield_admission_task_revalidated": False, "allocated_worktree_removed": False}
    options = {"purpose": "coding", "semantic_repository": root,
        "semantic_transport_schema": "supervisor-semantic-router-input@1",
        "semantic_metadata_view": "common-bindings@1", "census_only": True,
        "coding_reply_mode": "ordinary-completion@1",
        "public_instruction_artifact": Path(instruction_context["artifact"]),
        "public_instruction_sha256": instruction_context["sha256"], "public_instruction_task_cid": task_cid}
    if doctor and doctor.get("residual_context"):
        residual = doctor["residual_context"]
        options.update(doctor_residual_artifact=Path(residual["artifact"]),
            doctor_residual_sha256=residual["sha256"], doctor_residual_task_cid=task_cid)
    workspace = state / "allocated-worktree"
    with managed_git_worktree(repo_root=root, worktree_path=workspace,
            metadata_rel=".runtime/metadata-preflight.json", owner_rel=".runtime/metadata-preflight-owner.json") as session:
        if not session.ready:
            raise ValueError("native detached worktree allocation failed")
        common = ("rev-parse", "--path-format=absolute", "--git-common-dir")
        if (prep._git(root, *common) != prep._git(workspace, *common)
                or prep._git(root, "rev-parse", "HEAD") != prep._git(workspace, "rev-parse", "HEAD")):
            raise ValueError("census workspace belongs to a foreign repository or baseline")
        if local._sources(workspace, sorted(verified["manifest"]["sources"]), max_files=256) != verified["manifest"]["sources"]:
            raise ValueError("allocated census workspace input population differs")
        report["detached_same_common_dir_checked"] = True
        yield {"repository": root, "state": state, "workspace": workspace,
               "prompt_artifact": prompt_path, "runner_kwargs": options, "metadata": report}
        local.verify_local_benchmark_admission(admission, initial=True)
        with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
            final_task = intent.get_task(task_cid)
            final_contract, _, _, _ = local._contract(final_task["body"], task_cid)
            if final_task != task or final_task["status"] != "ready" or final_contract != contract:
                raise ValueError("native census owner task revision, contract or readiness changed")
        if (local._sources(root, profile["input_paths"], max_files=256) != inputs_before
                or local._sources(workspace, sorted(verified["manifest"]["sources"]), max_files=256)
                    != verified["manifest"]["sources"]
                or prep._git(root, *common) != prep._git(workspace, *common)
                or prep._git(root, "rev-parse", "HEAD") != prep._git(workspace, "rev-parse", "HEAD")):
            raise ValueError("census source population or allocated baseline changed")
        report["post_yield_admission_task_revalidated"] = True
    registered = prep._git(root, "worktree", "list", "--porcelain", "-z").decode().split("\0")
    if workspace.exists() or "worktree " + str(workspace) in registered:
        raise ValueError("native census allocated worktree cleanup incomplete")
    report["allocated_worktree_removed"] = True
