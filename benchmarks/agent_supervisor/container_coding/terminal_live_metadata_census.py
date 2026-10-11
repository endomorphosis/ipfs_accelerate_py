"""Observe an already planned native task without crossing coding dispatch.

The caller owns fresh preparation, Source384 inference, isolated LLM planning,
admission and admitted context construction. This helper never creates that
state or reads a provider response. Its public result contains counts, digests
and bounded provenance; the native prompt and operational artifacts stay private.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import uuid

from . import terminal_indexed_preparation as prep
from .terminal_metadata_preflight import _doctor_residual_only, _private_prompt


SCHEMA = "terminal-live-metadata-census-context@1"
MAX_METADATA_BYTES = 1_048_576


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode())


def _read_metadata(path: Path) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or info.st_size > MAX_METADATA_BYTES):
            raise ValueError("owner metadata must be a bounded owner-controlled regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("owner metadata exceeds its byte bound")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("owner metadata must be an object")
        return value
    finally:
        os.close(descriptor)


def _owned_directory(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022 or path.resolve(strict=True) != path):
        raise ValueError("canonical owner-controlled directory required")
    return info.st_dev, info.st_ino


def _pending_task(*, state, admission, verified):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

    tasks = verified["graph"].tasks
    if len(tasks) != 1:
        raise ValueError("one admitted pending census task required")
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        task = intent.get_task(tasks[0].task_cid)
        contract, _, _, _ = local._contract(task["body"], task["task_cid"])
        if (task["status"] != "ready" or task["task_alias"] != tasks[0].task_key
                or contract["manifest"] != admission["manifest"]
                or contract["graph_cid"] != verified["graph"].content_id
                or contract["task_spec"] != verified["manifest"]["tasks"][0]):
            raise ValueError("native census task readiness or admitted contract differs")
        plan = intent.get_plan(contract["plan_id"])
        reference = plan["body"]["local_planning_receipt_ref"]
        if local.load_local_planning_receipt(reference, manifest=admission["manifest"]) != admission["receipt"]:
            raise ValueError("native owner planning receipt differs from admission")
    return task, contract


def _owner_snapshot(*, state, require_source384):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_selection
    from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import load_intent_world_context
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from .terminal_initial_context import load_initial_context, _summaries

    database = state / "intent.duckdb"
    database_info = database.lstat()
    if (not stat.S_ISREG(database_info.st_mode) or database_info.st_uid != os.geteuid()
            or stat.S_IMODE(database_info.st_mode) & 0o022):
        raise ValueError("current native intent database must be an owner-controlled regular file")
    retained = {name: _read_metadata(state / (name + ".json")) for name in
                ("prepared", "admission", "planning-result", "context-result", "initial-context-result")}
    prepared = prep._load_prepared(state)
    prep._require_single_task_context(prepared)
    if prepared != retained["prepared"] or prepared.get("administrative_only"):
        raise ValueError("live census requires unchanged non-administrative preparation")
    admission, planning, indexed = (retained[name] for name in
                                    ("admission", "planning-result", "context-result"))
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    if admission["manifest"] != prepared["manifest"]:
        raise ValueError("live census admission differs from prepared manifest")
    root = Path(prepared["repository"])
    _owned_directory(root)
    task, contract = _pending_task(state=state, admission=admission, verified=verified)
    receipt = planning.get("provider_receipt") or {}
    provider = receipt.get("provider") or {}
    parsed = receipt.get("parse") or {}
    initial_binding = planning.get("initial_indexed_context") or {}
    if (planning.get("schema") != "terminal-indexed-planner-result@1"
            or planning.get("qualified") is not True or planning.get("provider_calls") != 1
            or planning.get("planning_strategy") not in {"direct", "intent_coverage"}
            or planning.get("task_cids") != [task["task_cid"]]
            or planning.get("goals") != len(verified["graph"].goals)
            or planning.get("tasks") != 1 or receipt.get("outcome") != "provider"
            or receipt.get("fallback", {}).get("used") is not False
            or receipt.get("plan_root_cid") != verified["graph"].content_id
            or receipt.get("request_cid") != PromptWorkflowRequest.from_dict(prepared["request"]).request_cid
            or receipt.get("scan_cid") != DirectoryScanReceipt.from_dict(prepared["scan"]).scan_cid
            or provider.get("attempted") is not True or provider.get("status") != "succeeded"
            or provider.get("provider_id") != prepared["provider"] or provider.get("model_id") != prepared["model"]
            or parsed.get("status") != "succeeded" or parsed.get("plan_root_cid") != verified["graph"].content_id
            or initial_binding.get("supplied_to_router") is not True
            or any(planning.get(key) != prepared[key] for key in ("provider", "model", "reasoning_effort"))):
        raise ValueError("persisted native provider planning provenance differs")
    initial = load_initial_context(state=state, prepared=prepared, require_empty_owner=False)
    # Reconstruct the historical planner projection's exact bytes. Its empty
    # world was current before admission; current admitted-world freshness is
    # independently checked below, so this does not certify an empty live DB.
    planning_summaries = _summaries(initial["descriptor"], initial["semantic"], initial["retrieval"],
        initial["capture"]["planning_context"], intent_freshness_checked=True)
    source384 = initial["descriptor"].get("source384_context")
    if require_source384 and source384 is None:
        raise ValueError("live census requires the selected current Source384 context")
    if (indexed.get("schema") != "terminal-indexed-context-preparation@1"
            or indexed.get("task_cid") != task["task_cid"] or indexed.get("initial_indexes_reused") is not True
            or indexed.get("new_embedding_calls") != 0 or indexed.get("provider_calls") != 0
            or indexed.get("initial_context_descriptor") != initial["receipt"]["descriptor"]
            or indexed.get("semantic_root_cid") != initial["semantic"]["semantic_root_cid"]
            or indexed.get("index_id") != initial["retrieval"]["index_id"]
            or initial_binding.get("descriptor") != initial["receipt"]["descriptor"]
            or initial_binding.get("semantic_root_cid") != initial["semantic"]["semantic_root_cid"]
            or initial_binding.get("index_id") != initial["retrieval"]["index_id"]
            or initial_binding.get("world_snapshot_cid") != initial["capture"]["snapshot"]["snapshot_cid"]
            or initial_binding.get("summary_sha256") != _identity(planning_summaries)):
        raise ValueError("admitted context differs from provider-planning initial context")
    bundle = indexed["context_bundle"]
    selection = load_task_context_selection(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=task["task_cid"], task_id=task["task_alias"])
    if selection.get("source384_context") != source384:
        raise ValueError("admitted task nomination differs from selected Source384 context")
    metadata = selection["metadata"]
    relative = Path(metadata["world context artifact"])
    artifact = root / relative
    if relative.is_absolute() or ".." in relative.parts or not artifact.resolve(strict=True).is_relative_to(root):
        raise ValueError("admitted world context escapes current repository")
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        world = load_intent_world_context(artifact=artifact,
            expected_sha256=metadata["world context sha256"], task_id=task["task_alias"],
            repository_id=metadata["world context repository"], intent=intent)
    if (world["world_snapshot_cid"] != indexed["world_snapshot_cid"]
            or world["semantic_root_cid"] != indexed["semantic_root_cid"]):
        raise ValueError("admitted world context differs from current indexed owner context")
    return {"retained": retained, "prepared": prepared, "admission": admission, "verified": verified,
            "task": task, "contract": contract, "indexed": indexed, "initial": initial, "source384": source384,
            "database_identity": (database_info.st_dev, database_info.st_ino)}


def _worker_readable_directory(path):
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    _owned_directory(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o005 != 0o005:
            os.fchmod(descriptor, 0o755)
    finally:
        os.close(descriptor)


@contextmanager
def prepare_terminal_live_metadata_census(*, state: Path, include_doctor_context: bool = True,
                                          require_source384: bool = True, worktree_root: Path | None = None,
                                          worker_writable: bool = False, worker_group_gid: int | None = None):
    """Yield existing live owner inputs and an owned detached census workspace.

    ``require_source384=False`` supports native authored lexical boundary tests;
    the output reports that selection honestly. Persisted provider provenance
    is checked against admission, but actual live LLM execution and isolation
    must be established by the caller's normal router custody evidence.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import prepare_public_instruction_context
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import run_command
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import managed_git_worktree
    from .indexed_doctor_lifecycle import _context

    if (type(include_doctor_context) is not bool or type(require_source384) is not bool
            or type(worker_writable) is not bool
            or worker_writable and (type(worker_group_gid) is not int or worker_group_gid < 0)
            or not worker_writable and worker_group_gid is not None):
        raise ValueError("explicit census context selections required")
    state = Path(state).absolute()
    state_identity = _owned_directory(state)
    observed = _owner_snapshot(state=state, require_source384=require_source384)
    prepared, admission, verified, task, indexed = (observed[key] for key in
        ("prepared", "admission", "verified", "task", "indexed"))
    root = Path(prepared["repository"])
    if state.is_relative_to(root) or root.is_relative_to(state):
        raise ValueError("live census state must remain outside its admitted repository")
    root_identity = _owned_directory(root)
    common_args = ("rev-parse", "--path-format=absolute", "--git-common-dir")
    common = prep._git(root, *common_args)
    common_path = Path(common.decode().strip())
    common_identity = _owned_directory(common_path)
    head = prep._git(root, "rev-parse", "HEAD")
    if head.decode().strip() != verified["manifest"]["baseline_commit"]:
        raise ValueError("live census repository HEAD differs from admission")
    private = state / "live-metadata-census"
    private.mkdir(mode=0o700)
    private_identity = _owned_directory(private)
    allocation_parent = private if worktree_root is None else Path(worktree_root).absolute()
    parent_identity = _owned_directory(allocation_parent)
    parent_group = allocation_parent.stat().st_gid
    if worktree_root is not None and (allocation_parent.is_relative_to(state)
            or state.is_relative_to(allocation_parent) or allocation_parent.is_relative_to(root)
            or root.is_relative_to(allocation_parent)):
        raise ValueError("explicit census worktree root must be outside owner state and admitted repository")
    if worker_writable and worktree_root is None:
        raise ValueError("worker-writable census requires an explicit external worktree root")
    for directory in (root / ".runtime", root / ".runtime/doctor-residuals",
                      root / ".runtime/router-public-instruction"):
        _worker_readable_directory(directory)
    doctor = (_doctor_residual_only(repository=root, state=state, admission=admission,
                task_cid=task["task_cid"]) if include_doctor_context else None)
    instruction = prepare_public_instruction_context(repository=root, admission=admission,
        task_cid=task["task_cid"], source_path=prep.INSTRUCTION,
        expected_source_sha256=verified["manifest"]["sources"][prep.INSTRUCTION]["sha256"])
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        spec = prepared["spec"]
        portal = PortalTask(task_id=task["task_alias"], title=task["body"]["title"],
            canonical_task_cid=task["task_cid"], status="ready", completion="manual", priority="P2",
            track="implementation", outputs=[row["path"] for row in spec["outputs"]],
            validation=[shlex.join(row["argv"]) for row in spec["validations"]],
            acceptance=" ; ".join(row["criterion"] for row in spec["acceptance"]),
            metadata={"database task cid": task["task_cid"],
                "local planning contract cid": local.content_identity(task["body"][local.CONTRACT_KEY])})
        daemon = TodoImplementationDaemon(todo_path=private / "pending.todo.md",
            state_path=private / "worker/tasks.json", strategy_path=private / "worker/strategy.json",
            events_path=private / "worker/events.jsonl", repo_root=root, task_header_prefix="## TB-")
        daemon._world_intent_repository = intent
        daemon._task_context_nomination_bundle = indexed["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal, attempt=1)
            retrieval, world = _context(prompt, "code-retrieval-context"), _context(prompt, "intent-world-context")
            if (retrieval["status"] != "current" or retrieval["index_id"] != indexed["index_id"]
                    or world["semantic_root_cid"] != indexed["semantic_root_cid"]
                    or world["world_snapshot_cid"] != indexed["world_snapshot_cid"]):
                raise ValueError("native pending prompt did not consume the admitted current context")
        finally:
            daemon.close_event_runtime()
    current = _owner_snapshot(state=state, require_source384=require_source384)
    if (current["database_identity"] != observed["database_identity"]
            or current["retained"] != observed["retained"] or current["task"] != task
            or current["contract"] != observed["contract"]):
        raise ValueError("live census owner changed during prompt preparation")
    prompt_path = private / "worker-prompt.txt"
    _private_prompt(prompt_path, prompt)
    source384 = observed["source384"]
    report = {"schema": SCHEMA, "observation_only": True,
        "graph_provenance": "persisted-native-provider-plan", "prior_planning_provider_calls": 1,
        "planning_performed_by_census": False, "live_llm_execution_verified_here": False,
        "isolation_verified_here": False, "provider_calls": 0, "model_runs": 0,
        "source384_present": source384 is not None, "source384_currentness_verified": source384 is not None,
        "source384_inference_performed_by_census": False, "source384_neural_inference_replayed": False,
        "source384_receipt_sha256": _identity(source384) if source384 else None,
        "source384_config_sha256": source384["config_sha256"] if source384 else None,
        "source384_checkpoint_sha256": source384["checkpoint_sha256"] if source384 else None,
        "source384_inference_sha256": source384["inference_sha256"] if source384 else None,
        "metadata_digest_encoding": "canonical-json-utf8@1",
        "prepared_metadata_sha256": _identity(prepared), "admission_metadata_sha256": _identity(admission),
        "planning_metadata_sha256": _identity(observed["retained"]["planning-result"]),
        "context_metadata_sha256": _identity(indexed), "native_task_cid_sha256": _identity(task["task_cid"]),
        "native_task_revision": task["revision"], "native_task_contract_sha256": _identity(observed["contract"]),
        "manifest_cid_sha256": _identity(verified["receipt"]["manifest_cid"]),
        "graph_cid_sha256": _identity(verified["graph"].content_id),
        "semantic_root_cid_sha256": _identity(indexed["semantic_root_cid"]),
        "index_id_sha256": _identity(indexed["index_id"]), "baseline_commit_sha256": _sha(head.strip()),
        "source_count": len(verified["manifest"]["sources"]),
        "graph_goal_count": len(verified["graph"].goals), "graph_task_count": 1,
        "indexed_symbol_count": indexed["indexed_symbols"], "full_capsule_count": indexed["full_capsules"],
        "worker_capsule_count": indexed["worker_capsules"], "native_fact_rows_replayed": indexed["native_fact_rows_replayed"],
        "native_prompt_sha256": _sha(prompt.encode()), "native_prompt_bytes": len(prompt.encode()),
        "public_instruction_artifact_sha256": instruction["sha256"],
        "doctor_residual_included": bool(doctor and doctor.get("residual_context")),
        "doctor_repair_execution_permitted": False, "implementation_executed": False,
        "validation_executed": False, "official_verifier_executed": False,
        "provider_response_reads": False, "hidden_evaluator_reads": False, "solution_reads": False,
        "owner_state_created_by_census": False, "existing_native_owner_state_observed": True,
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "publication_authority": False, "required_fact_omission_authority": False,
        "post_yield_admission_task_revalidated": False, "allocated_worktree_removed": False,
        "guarded_cleanup": True, "explicit_external_worktree_root": worktree_root is not None,
        "worker_writable_workspace_selected": worker_writable}
    options = {"purpose": "coding", "semantic_repository": root,
        "semantic_transport_schema": "supervisor-semantic-router-input@1",
        "semantic_metadata_view": "common-bindings@1", "census_only": True,
        "coding_reply_mode": "ordinary-completion@1", "public_instruction_artifact": Path(instruction["artifact"]),
        "public_instruction_sha256": instruction["sha256"], "public_instruction_task_cid": task["task_cid"]}
    if doctor and doctor.get("residual_context"):
        residual = doctor["residual_context"]
        options.update(doctor_residual_artifact=Path(residual["artifact"]),
            doctor_residual_sha256=residual["sha256"], doctor_residual_task_cid=task["task_cid"])
    workspace = allocation_parent / ("terminal_live_metadata_census_" + uuid.uuid4().hex)
    allocation = {}

    def check_workspace():
        try:
            workspace_info, git_info = workspace.lstat(), (workspace / ".git").lstat()
            workspace_mode, git_mode = stat.S_IMODE(workspace_info.st_mode), stat.S_IMODE(git_info.st_mode)
            original_modes = (workspace_mode == 0o755 and workspace_info.st_gid == allocation["initial_gid"]
                and git_mode == 0o644 and git_info.st_gid == allocation["initial_gid"])
            worker_modes = (worker_writable and workspace_mode == 0o3770
                and workspace_info.st_gid == worker_group_gid and git_mode == 0o440
                and git_info.st_gid == worker_group_gid)
            valid = (_owned_directory(state) == state_identity and _owned_directory(root) == root_identity
                and _owned_directory(private) == private_identity and _owned_directory(common_path) == common_identity
                and _owned_directory(allocation_parent) == parent_identity and allocation_parent.stat().st_gid == parent_group
                and stat.S_ISDIR(workspace_info.st_mode) and workspace.resolve(strict=True) == workspace
                and (workspace_info.st_dev, workspace_info.st_ino) == allocation["workspace_identity"]
                and workspace_info.st_uid == os.geteuid() and stat.S_ISREG(git_info.st_mode)
                and (git_info.st_dev, git_info.st_ino) == allocation["git_file_identity"]
                and git_info.st_uid == os.geteuid() and (original_modes or worker_modes)
                and prep._git(root, *common_args) == common and prep._git(workspace, *common_args) == common
                and _sha((workspace / ".git").read_bytes()) == allocation["git_file_sha256"])
        except (OSError, ValueError, KeyError):
            valid = False
        if not valid:
            raise ValueError("census workspace allocation ownership changed; cleanup refused")

    def guarded_command(argv, **kwargs):
        if tuple(argv[:3]) == ("git", "worktree", "remove"):
            if not allocation:
                raise ValueError("census unowned workspace cleanup refused")
            check_workspace()
        result = run_command(("git", "-c", "core.hooksPath=/dev/null", *argv[1:]), **kwargs)
        if tuple(argv[:3]) == ("git", "worktree", "remove") and not result.ok:
            # The native helper otherwise falls back to recursive deletion.
            # A failed Git removal never authorizes that wider cleanup here.
            raise ValueError("guarded census Git worktree removal failed; recursive cleanup refused")
        if tuple(argv[:3]) == ("git", "worktree", "add") and result.ok:
            descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(descriptor)
                if info.st_uid != os.geteuid():
                    raise ValueError("native census allocation owner differs")
                os.fchmod(descriptor, 0o755)
            finally:
                os.close(descriptor)
            git_file = workspace / ".git"
            descriptor = os.open(git_file, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                git_info = os.fstat(descriptor)
                if not stat.S_ISREG(git_info.st_mode) or git_info.st_uid != os.geteuid():
                    raise ValueError("native census allocation Git file owner differs")
                os.fchmod(descriptor, 0o644)
            finally:
                os.close(descriptor)
            allocation.update(workspace_identity=_owned_directory(workspace), initial_gid=info.st_gid,
                git_file_identity=(git_info.st_dev, git_info.st_ino),
                git_file_sha256=_sha(git_file.read_bytes()))
        return result

    with managed_git_worktree(repo_root=root, worktree_path=workspace,
            metadata_rel=".runtime/live-metadata-census.json", owner_rel=".runtime/live-metadata-census-owner.json",
            run_command_fn=guarded_command, prune_on_exit=False) as session:
        if not session.ready:
            raise ValueError("native census detached worktree allocation failed")
        check_workspace()
        if prep._git(workspace, "rev-parse", "HEAD") != head:
            raise ValueError("census workspace baseline differs")
        # Symbolic refs must be absent; rev-parse --abbrev-ref is defined for a
        # detached HEAD and does not turn the expected Git status into an error.
        if prep._git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip() != b"HEAD":
            raise ValueError("census workspace must have detached HEAD")
        if local._sources(workspace, sorted(verified["manifest"]["sources"]), max_files=256) != verified["manifest"]["sources"]:
            raise ValueError("census workspace source population differs")
        report["detached_same_common_dir_checked"] = True
        try:
            yield {"repository": root, "state": state, "workspace": workspace,
                   "prompt_artifact": prompt_path, "runner_kwargs": options, "metadata": report}
        finally:
            # A caller exception never skips current owner/source custody checks.
            final = _owner_snapshot(state=state, require_source384=require_source384)
            check_workspace()
            if (final["database_identity"] != observed["database_identity"]
                    or final["retained"] != observed["retained"] or final["task"] != task
                    or final["contract"] != observed["contract"]
                    or prep._git(root, "rev-parse", "HEAD") != head
                    or prep._git(workspace, "rev-parse", "HEAD") != head
                    or prep._git(workspace, "rev-parse", "--abbrev-ref", "HEAD").strip() != b"HEAD"
                    or local._sources(workspace, sorted(verified["manifest"]["sources"]), max_files=256)
                        != verified["manifest"]["sources"]):
                raise ValueError("census admission, task revision, contract, sources or baseline changed")
            report["post_yield_admission_task_revalidated"] = True
    registered = prep._git(root, "worktree", "list", "--porcelain", "-z").decode().split("\0")
    if workspace.exists() or "worktree " + str(workspace) in registered:
        raise ValueError("guarded census workspace cleanup incomplete")
    report["allocated_worktree_removed"] = True
