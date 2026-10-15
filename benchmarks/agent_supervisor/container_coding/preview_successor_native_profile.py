"""Fresh native, model-off preview inputs for the signed successor benchmark.

This is trusted benchmark owner wiring. Its private persisted planning source
is separate from the subsequently mutable execution queue. Every observer
rereads that source and owner policy/catalog records; source observations use
the real datasets index. Checker identities describe the actual signed public
checks and their scripts, never successful outcomes or formal proof authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

SCHEMA = "native-preview-successor-owner-profile@1"
COMPANION_SCHEMA = "native-preview-successor-companion-owner-profile@1"
MAX_FILE_BYTES = 1024 * 1024
MAX_OBSERVATIONS = 1024
_MIB = 1024 * 1024
_TASK_KEYS = ("SUCCESSOR-TYPE", "SUCCESSOR-FORMAT")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate native owner metadata key")
        result[key] = value
    return result


def _read_file(path, *, limit=MAX_FILE_BYTES):
    """Bound one regular canonical descriptor and reject replacement while read."""
    path = Path(path)
    _require(path.is_absolute() and path.resolve(strict=True) == path,
             "canonical native owner metadata path required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
                 "bounded regular native owner metadata required")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    named = path.stat(follow_symlinks=False)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                             value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    _require(len(raw) == before.st_size and len(raw) <= limit
             and identity(before) == identity(after) == identity(named),
             "native owner metadata changed while observed")
    return raw


def _write_file(path, raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_FILE_BYTES,
             "bounded native owner artifact bytes required")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    _require(_read_file(path) == raw, "persisted native owner artifact differs")


def native_preview_memory_mb(captured, *, memory_mb):
    """Reserve the native scan estimate within an exact caller envelope.

    The selected immutable snapshot supplies capture bounds, not its observed
    file count. This is an admission estimate, not a measured RSS guarantee.
    The full signed worker retains its separate 1024 MiB minimum caller budget.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits

    _require(type(memory_mb) is int and 256 <= memory_mb <= 4096,
             "bounded exact native preview memory envelope required")
    try:
        limits = CodebaseScanLimits(captured.max_entries, captured.max_file_bytes)
    except AttributeError as error:
        raise ValueError("captured native scan limits required") from error
    needed = 16 * limits.max_entries * limits.max_file_bytes
    derived = max(256, (needed + _MIB - 1) // _MIB)
    limits.validate_reservation(derived)
    _require(derived <= memory_mb,
             "captured scan exceeds the native preview memory envelope")
    return derived


def prepare_native_preview_successor(*, parent_admission, graph, index, registry,
        selection, root, completion, repository, owner_state_dir, scheduler=None,
        parent_lease=None, cancel_event=None, timeout_seconds=30.0,
        admission_timeout_seconds=30.0, memory_mb=1024):
    """Produce a genuine preview, signed companion and immutable-parent @7 child.

    The returned private ``bridge_inputs`` retains the fresh observers for
    installation, context creation and reservation. ``records`` is diagnostic;
    its observation list continues to grow on later uses and should be saved
    again after execution. No preparation, training, inference, provider calls,
    task installation or worker launch occurs here.
    """
    return _prepare_native_preview_successor(parent_admission=parent_admission, graph=graph,
        index=index, registry=registry, selection=selection, root=root, completion=completion,
        repository=repository, owner_state_dir=owner_state_dir, scheduler=scheduler,
        parent_lease=parent_lease, cancel_event=cancel_event, timeout_seconds=timeout_seconds,
        admission_timeout_seconds=admission_timeout_seconds, memory_mb=memory_mb, _worker_profile=True)


def prepare_native_preview_successor_companion(*, parent_admission, graph, index, registry,
        selection, root, completion, repository, owner_state_dir, scheduler=None,
        parent_lease=None, cancel_event=None, timeout_seconds=30.0,
        admission_timeout_seconds=30.0, memory_mb=1024):
    """Produce a genuine preview and signed companion for the original @6 parent.

    Private typed inputs retain fresh source and checker observers for a separate
    current companion verification. Diagnostic records grow on later observations.
    This entrypoint authors no worker manifest, admission or context and performs
    no task installation, worker launch, fitting or forward inference.
    """
    return _prepare_native_preview_successor(parent_admission=parent_admission, graph=graph,
        index=index, registry=registry, selection=selection, root=root, completion=completion,
        repository=repository, owner_state_dir=owner_state_dir, scheduler=scheduler,
        parent_lease=parent_lease, cancel_event=cancel_event, timeout_seconds=timeout_seconds,
        admission_timeout_seconds=admission_timeout_seconds, memory_mb=memory_mb, _worker_profile=False)


def _prepare_native_preview_successor(*, parent_admission, graph, index, registry,
        selection, root, completion, repository, owner_state_dir, scheduler=None,
        parent_lease=None, cancel_event=None, timeout_seconds=30.0,
        admission_timeout_seconds=30.0, memory_mb=1024, _worker_profile):
    """Share native owner inputs without reducing either public profile's checks."""
    _require(type(_worker_profile) is bool, "exact native preview preparation profile required")
    from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
        ProducerRule, TaskCandidate, TypedIntent, TypedPredicate, obligation_id_for_producer,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        DirtyTreePolicy, PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget, TaskSourceKind,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import (
        PROFILE, RepositoryPlanPreviewOwner, preview_repository_plan,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context import structural_codebase_context
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateMaterials, PlanCreatePreviewReceipt,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        codebase_preview_successor_admission as companion,
        codebase_preview_successor_worker_admission as worker,
        local_planning_admission as local,
    )
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError, ResourceLane, get_global_resource_scheduler,
    )

    _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 600 and type(admission_timeout_seconds) in {int, float}
             and math.isfinite(admission_timeout_seconds) and 0 <= admission_timeout_seconds <= 600
             and type(memory_mb) is int and 1024 <= memory_mb <= 4096,
             "bounded native preview successor resources required")
    _require(scheduler is None or parent_lease is None, "choose scheduler or a native parent lease")
    deadline = time.monotonic() + timeout_seconds

    def remaining(cap=600.0):
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("native preview successor cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("native preview successor preparation deadline exceeded")
        return min(cap, left)

    remaining()
    repository = Path(repository)
    _require(repository.is_absolute() and repository.resolve(strict=True) == repository
             and repository.is_dir(), "canonical native benchmark repository required")
    directory = Path(owner_state_dir)
    _require(directory.is_absolute() and not directory.exists()
             and directory.parent.resolve(strict=True) == directory.parent,
             "fresh canonical native owner state directory required")
    parent = local._plain(parent_admission)
    parent_wire = _wire(parent)
    verified = local.verify_local_benchmark_admission(parent)
    _require(verified["manifest"]["schema"] == local.SUCCESSOR_MANIFEST_SCHEMA
             and type(graph) is local.PromptGoalGraph
             and _wire(graph.to_dict()) == _wire(verified["graph"].to_dict()),
             "complete independently authored immutable @6 parent graph required")
    by_key = {task.task_key: task for task in graph.tasks}
    _require(set(by_key) == set(_TASK_KEYS) and len(graph.tasks) == 2 and len(graph.goals) == 1,
             "exact complete two-task native benchmark population required")
    tasks = tuple(by_key[key] for key in _TASK_KEYS)
    _require(not tasks[0].dependency_task_cids
             and tasks[1].dependency_task_cids == (tasks[0].task_cid,)
             and all(task.acceptance and task.validations for task in tasks),
             "original native dependency and acceptance population required")
    head = CodebaseHead.from_dict(selection.to_dict()["current_head"])
    _require(index.current(head.repository_id) == head, "selected native successor head is stale")
    native_manifest = index.load(head.manifest_cid)
    _require(graph.program_root == native_manifest.semantic_state.state_cid,
             "authored graph and selected native semantic root differ")
    try:
        captured = native_manifest.snapshot
        captured_limits = CodebaseScanLimits(captured.max_entries, captured.max_file_bytes)
    except AttributeError as error:
        raise ValueError("captured native scan limits required") from error
    preview_memory_mb = native_preview_memory_mb(captured_limits, memory_mb=memory_mb)
    directory.mkdir(mode=0o700)
    policy = local._plain(local.LOCAL_POLICY)
    policy_cid = content_identity(policy)
    validations = [{"task_cid": task.task_cid, "validations": [
        {name: local._plain(getattr(check, name)) for name in
         ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
        for check in task.validations]} for task in tasks]
    _require(all(check["policy_cid"] == policy_cid for row in validations for check in row["validations"]),
             "actual native validation policy differs from signed task policy")
    script_paths = sorted({check["argv"][2] for row in validations for check in row["validations"]
                           if check["argv"][:2] == ["python3", "-B"] and len(check["argv"]) == 3
                           and check["cwd"] == "."})
    _require(script_paths == ["check_offset.py", "check_type.py"]
             and sum(len(row["validations"]) for row in validations) == 2,
             "exact native authored public check argv required")
    runtime_dir = Path(local.__file__).resolve(strict=True).parent
    implementation_paths = {
        "local_planning_admission.py": runtime_dir / "local_planning_admission.py",
        "local_completion_bridge.py": runtime_dir / "local_completion_bridge.py",
        "candidate_execution.py": runtime_dir / "candidate_execution.py",
        "validation_runtime.py": runtime_dir.parent / "validation" / "validation_runtime.py",
    }

    def current_checker():
        return {"schema": "native-preview-public-validation-policy@1",
            "planning_policy": local._plain(local.LOCAL_POLICY), "task_validations": validations,
            "scripts": [{"path": path, "sha256": hashlib.sha256(_read_file(repository / path)).hexdigest()}
                        for path in script_paths],
            "implementation_sha256": {name: hashlib.sha256(_read_file(path)).hexdigest()
                                      for name, path in sorted(implementation_paths.items())},
            "outcomes_recorded": False, "checker_executed": False,
            "proof_authority": False, "scope": "native signed public validation identity only"}

    checker_record = current_checker()
    _write_file(directory / "checker-policy.json", _wire(checker_record))
    checker_policy_cid = content_identity(checker_record)
    source_lines = ["# Authored native successor planning source", "", graph.goals[0].objective, ""]
    for task in tasks:
        source_lines.extend([f"- [ ] {task.task_key}: {task.objective}",
            f"  Signed task CID: {task.task_cid}",
            *["  Acceptance: " + item.criterion for item in task.acceptance]])
    source_raw = ("\n".join(source_lines) + "\n").encode("utf-8")
    _write_file(directory / "TASKS.md", source_raw)
    source_record = {"schema": "native-preview-authored-task-source@1", "kind": "markdown",
        "source_path": "TASKS.md", "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "parent_admission_cid": content_identity(parent), "graph": graph.to_dict(),
        "scope": "immutable authored planning inputs; execution queue lifecycle is separate"}
    source_cid = content_identity(source_record)
    predicates = tuple(TypedPredicate("requirement:preview:" + kind, "authored_public_check_requirement",
        "codebase:increment", object_ref="acceptance:public-" + kind,
        provenance_refs=(source_cid,), validation_requirement_refs=("public-" + kind,))
        for kind in ("type", "offset"))
    intent = TypedIntent("intent:native-successor-public-checks", predicates, (source_cid,),
        current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule("producer:preview:" + kind, (predicate.predicate_id,),
        provenance_refs=(source_cid,)) for kind, predicate in zip(("type", "offset"), predicates))
    candidates = tuple(TaskCandidate("candidate:preview:" + kind,
        (obligation_id_for_producer(producer.producer_id, predicate.predicate_id),),
        producer_id=producer.producer_id,
        depends_on_candidate_ids=() if kind == "type" else ("candidate:preview:type",),
        provenance_refs=(source_cid,))
        for kind, producer, predicate in zip(("type", "offset"), producers, predicates))
    evaluation_policy = EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids,
        evidence_terms=intent.source_refs, allowed_scopes=("scope:repository",),
        available_resource_classes=("cpu",), require_validation=True, require_proof=False)
    metadata = {
        "task-source": source_record,
        "planning-policy": {"schema": "native-preview-planning-policy@1", "policy": policy,
            "checker_policy_cid": checker_policy_cid, "evaluation_policy": evaluation_policy.to_dict(profile_g=True)},
        "intent-ir": {"schema": "native-preview-intent-ir@1", "kind": "authored_typed_intent",
            "intent": intent.to_dict(), "trained_head_selected": False},
        "legal-ir": {"schema": "native-preview-external-ir-selection@1", "kind": "legal",
            "selected": False, "model_calls": 0, "proof_authority": False},
        "security-ir": {"schema": "native-preview-external-ir-selection@1", "kind": "security",
            "selected": False, "model_calls": 0, "proof_authority": False},
        "capability-catalog": {"schema": "native-preview-capability-catalog@1",
            "available": ["native_structural_source_observation", "typed_obligation_compilation",
                          "native_signed_public_validation"],
            "checker_policy_cid": checker_policy_cid, "behavioral_facts_supplied": 0},
        "provider-catalog": {"schema": "native-preview-provider-catalog@1", "providers": [],
            "model_off": True, "network": "disabled"},
        "usage-policy": {"schema": "native-preview-usage-policy@1", "model_calls": 0,
            "training_steps": 0, "structural_only": True, "production_activation": False},
        "configuration": {"schema": "native-preview-configuration@1", "repository_id": head.repository_id,
            "selected_head": head.to_dict(), "successor_selection_cid": selection.artifact_cid,
            "root_cid": root.artifact_cid, "completion_cid": completion.artifact_cid,
            "task_source_kind": "markdown", "profile": PROFILE, "max_tasks": 2,
            "max_model_calls": 0, "checker_policy_cid": checker_policy_cid},
    }
    for name, record in metadata.items():
        _write_file(directory / (name + ".json"), _wire(local._plain(record)))
    records = {"schema": SCHEMA if _worker_profile else COMPANION_SCHEMA, "owner_state_dir": str(directory),
        "capture_limits": {"max_entries": captured_limits.max_entries,
                           "max_file_bytes": captured_limits.max_file_bytes},
        "preview_memory_mb": preview_memory_mb, "signed_memory_mb": memory_mb,
        "owner_artifacts": [{"name": name + ".json", "cid": content_identity(record),
            "sha256": hashlib.sha256(_read_file(directory / (name + ".json"))).hexdigest()}
            for name, record in sorted(metadata.items())] + [
                {"name": "checker-policy.json", "cid": checker_policy_cid,
                 "sha256": hashlib.sha256(_read_file(directory / "checker-policy.json")).hexdigest()},
                {"name": "TASKS.md", "sha256": hashlib.sha256(source_raw).hexdigest()}],
        "checker_policy_cid": checker_policy_cid, "planning_source_cid": source_cid,
        "task_source_kind": "markdown", "execution_queue_root_observed": False,
        "native_source_observation": True, "application_observations_atomic": False,
        "checker_outcomes_observed_here": False, "source_semantics_verified": False,
        "proof_authority": False, "worker_launched_here": False, "production_admitted": False,
        "owner_observations": []}
    expected_fields = {name: set(record) for name, record in metadata.items()}
    expected_request_wire = None

    def read_records():
        values = {}
        for name, fields in expected_fields.items():
            raw = _read_file(directory / (name + ".json"))
            value = json.loads(raw, object_pairs_hook=_unique,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite native owner metadata")))
            _require(type(value) is dict and set(value) == fields and _wire(value) == raw
                     and value["schema"] == metadata[name]["schema"],
                     "closed canonical native owner metadata required")
            values[name] = value
        source = values["task-source"]
        _require(hashlib.sha256(_read_file(directory / "TASKS.md")).hexdigest() == source["source_sha256"],
                 "current authored planning source differs from its metadata")
        return values

    def log_observation(kind, value):
        observations = records["owner_observations"]
        _require(len(observations) < MAX_OBSERVATIONS, "native owner observation count exceeds bound")
        observations.append({"ordinal": len(observations), "kind": kind, **value})

    def observe_roots(typed_request, *, source_parent=None):
        if typed_request is not None:
            _require(expected_request_wire is not None
                     and type(typed_request) is PlanCreateRequest
                     and _wire(typed_request.to_dict()) == expected_request_wire,
                     "native roots observer received another request")
        values = read_records()
        # Later uses have their own bounded native source operation rather
        # than borrowing the already released preview orchestration lease.
        controls = {"scheduler": scheduler if source_parent is None else None,
            "parent_lease": source_parent if source_parent is not None else parent_lease,
            "cancel_event": cancel_event, "timeout_seconds": min(90.0, timeout_seconds),
            "admission_timeout_seconds": min(30.0, admission_timeout_seconds), "memory_mb": preview_memory_mb}
        with structural_codebase_context(index, repository, repository_id=head.repository_id,
                expected_head=head, **controls) as structural:
            _require(_wire(values) == _wire(read_records()), "owner roots changed during native source observation")
            roots = PlanAuthorityRoots(repository_id=structural.head.repository_id,
                repository_root_cid=structural.head.snapshot_cid, dirty_worktree_root=structural.head.snapshot_cid,
                task_source_id=content_identity({"schema": "native-preview-task-source-owner@1",
                    "repository_id": structural.head.repository_id, "source_path": "TASKS.md", "kind": "markdown"}),
                task_source_revision=content_identity(values["task-source"]),
                policy_root=content_identity(values["planning-policy"]),
                intent_ir_root=content_identity(values["intent-ir"]), legal_ir_root=content_identity(values["legal-ir"]),
                security_ir_root=content_identity(values["security-ir"]), program_root=structural.semantic_state_cid,
                capability_catalog_root=content_identity(values["capability-catalog"]),
                provider_catalog_root=content_identity(values["provider-catalog"]),
                usage_policy_root=content_identity(values["usage-policy"]),
                configuration_root=content_identity(values["configuration"]))
        _require(_wire(values) == _wire(read_records()), "owner roots changed during native source close")
        log_observation("roots", {"roots": roots.to_dict(), "structural_context_cid": structural.cid,
                                  "head": structural.to_dict()["head"], "memory_mb": preview_memory_mb})
        return roots

    def roots_observer(typed_request):
        return observe_roots(typed_request)

    def checker_observer(typed_request):
        _require(type(typed_request) is PlanCreateRequest
                 and _wire(typed_request.to_dict()) == expected_request_wire,
                 "native checker observer received another request")
        raw = _read_file(directory / "checker-policy.json")
        observed = current_checker()
        _require(_wire(observed) == raw and _read_file(directory / "checker-policy.json") == raw,
                 "current validation policy, scripts or checker implementation changed")
        cid = content_identity(observed)
        log_observation("checker", {"checker_policy_cid": cid, "policy_cid": content_identity(observed["planning_policy"]),
            "scripts": observed["scripts"], "checker_executed": False, "outcomes_recorded": False})
        return cid

    roots = observe_roots(None)
    scope_paths = tuple(sorted({path for task in tasks for path in task.scope_paths}))
    request = PlanCreateRequest(prompt_source_cid=source_cid, repository_id=head.repository_id,
        repository_root=str(repository), scope_paths=scope_paths,
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND, task_source_kind=TaskSourceKind.MARKDOWN,
        board_namespace="native-preview-successor", alias_prefix="NATIVE", roots=roots,
        budget=PlanRequestBudget(max_goals=1, max_tasks=2, max_graph_depth=2, max_ready_width=1,
            max_model_calls=0, max_provider_tokens=0, max_latency_ms=max(1, int(remaining(90) * 1000))),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=(), observe_roots=True,
        caller="principal:native-benchmark-owner",
        redacted_source_metadata={"authored_task_source_cid": source_cid,
            "immutable_parent_admission_cid": content_identity(parent)})
    expected_request_wire = _wire(request.to_dict())
    materials = PlanCreateMaterials(intent=intent, producers=producers, task_candidates=candidates,
        frozen_goal=FrozenPlanningGoal("goal:native-successor-preview", graph.goals[0].goal_cid,
            head.snapshot_cid, evaluation_policy),
        candidate_context={"domain": "native-authored-successor-public-checks",
            "task_metadata": {candidate.candidate_id: {"predicted_files": list(task.predicted_files),
                "scope_ids": ["scope:repository"], "resource_classes": ["cpu"]}
                for candidate, task in zip(candidates, tasks)}},
        extra={"root_observation_profile": PROFILE, "authored_task_source_cid": source_cid,
               "checker_policy_cid": checker_policy_cid})
    lease_request = {"lane": ResourceLane.ORCHESTRATION, "cpu_slots": 1, "memory_mb": preview_memory_mb,
        "child_process_slots": 1, "cancel_event": cancel_event,
        "timeout": min(admission_timeout_seconds, remaining()), "request_id": "native-preview:" + request.request_cid}
    if parent_lease is None:
        authority = scheduler if scheduler is not None else get_global_resource_scheduler()
        lease = authority.acquire(**lease_request)
    else:
        lease = parent_lease.acquire_child(**lease_request)
    with lease:
        preview_lease_record = lease.to_dict()
        _require(preview_lease_record["memory_mb"] == preview_memory_mb,
                 "native preview lease differs from the captured memory reservation")
        preview_signal = lease.combined_cancellation_signal(cancel_event)
        preview_owner = RepositoryPlanPreviewOwner(index=index, repository=repository,
            expected_head=head, parent_lease=lease, cancel_event=preview_signal,
            timeout_seconds=remaining(90), memory_mb=preview_memory_mb)
        preview_record = preview_repository_plan(owner=preview_owner, request=request, materials=materials,
            policy_observer=lambda typed_request: observe_roots(typed_request, source_parent=lease))
        remaining()
    preview = PlanCreatePreviewReceipt.from_dict(preview_record["preview"])
    candidate_bindings = [{"candidate_id": candidate.candidate_id, "task_cid": task.task_cid}
                          for candidate, task in zip(candidates, tasks)]
    requirement_bindings = [{"predicate_id": predicate.predicate_id, "task_cids": [task.task_cid],
        "acceptance_cids": sorted(item.content_id for item in task.acceptance)}
        for predicate, task in zip(predicates, tasks)]
    bridge_inputs = worker.PreviewSuccessorBridgeInputs(request=request, materials=materials, preview=preview,
        roots_observer=roots_observer, checker_observer=checker_observer)
    current = {"index": index, "registry": registry, "selection": selection, "root": root,
        "completion": completion, "repository": repository, "scheduler": scheduler,
        "parent_lease": parent_lease, "cancel_event": cancel_event,
        "admission_timeout_seconds": admission_timeout_seconds, "memory_mb": memory_mb}
    bridge = companion.author_current_preview_successor_bridge(**current, admission=parent,
        request=request, materials=materials, preview=preview, candidate_bindings=candidate_bindings,
        requirement_bindings=requirement_bindings, checker_policy_cid=checker_policy_cid,
        roots_observer=roots_observer, checker_observer=checker_observer, timeout_seconds=remaining())
    worker_result, worker_records = {}, {}
    if _worker_profile:
        manifest = worker.author_current_preview_successor_worker_manifest(**current, parent_admission=parent,
            bridge=bridge, bridge_inputs=bridge_inputs, timeout_seconds=remaining())
        admission = worker.admit_current_preview_successor_worker_plan(**current, manifest=manifest, graph=graph,
            bridge_inputs=bridge_inputs, timeout_seconds=remaining())
        worker_result = {"manifest": manifest, "admission": admission}
        worker_records = {"worker_manifest_cid": content_identity(manifest),
                          "worker_admission_cid": content_identity(admission)}
    else:
        # The worker route's next boundary already checks this same deadline.
        # A companion-only return must also reject cancellation or expiry.
        remaining()
    _require(_wire(parent) == parent_wire and _wire(local._plain(parent_admission)) == parent_wire,
             "original @6 parent admission changed during native preview preparation")
    records.update({"request_cid": request.request_cid, "input_snapshot_cid": preview.input_snapshot_cid,
        "preview_cid": preview.receipt_cid, "signed_bridge_cid": content_identity(bridge),
        "bridge_cid": bridge["payload"]["bridge_cid"], "parent_admission_cid": content_identity(parent),
        **worker_records,
        "candidate_bindings": candidate_bindings, "requirement_bindings": requirement_bindings,
        "administrator_task_cids": sorted(task.task_cid for task in tasks),
        "population": bridge["payload"]["population"], "preview_record": preview_record,
        "preview_lease": preview_lease_record, "preview_owner_memory_mb": preview_owner.memory_mb,
        "orchestration_lease_released": lease.released, "model_calls": 0, "training_steps": 0})
    if not _worker_profile:
        records["worker_profile_integrated"] = False
        remaining()
    return {**worker_result, "bridge_inputs": bridge_inputs,
        "request": request, "materials": materials, "preview": preview, "bridge": bridge,
        "checker_policy_cid": checker_policy_cid, "records": records}


__all__ = ["SCHEMA", "COMPANION_SCHEMA", "native_preview_memory_mb", "prepare_native_preview_successor",
           "prepare_native_preview_successor_companion"]
