"""Fresh native scan/query/model-off planning join qualification.

Only a new output namespace is accepted. Setup fits two private root epochs
and one same-head child epoch, then executes native conditional verification
and applicability. Subsequent joins, receiving validation and plan previews
reuse that exact state. Parent subprocess observations do not attest solver
executions inside native isolated phase workers. This finite 8D CPU fixture
does not activate production, prove runtime behavior or qualify crash recovery.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import sysconfig
import threading
import time

from benchmarks.agent_supervisor.container_coding import qualify_codebase_inventory_scan as base

SCHEMA = "codebase-inventory-evidence-join-qualification@1"
FIXTURE_HELPER_SHA256 = "d3f2ed54b99ab2d62d0892576db2375f17eebafccb2eb1ddda5a6691f128fd4a"
QUERY_HELPER_SHA256 = "4d5b994bf192a21bda003986aa45c247e9e2ffbc9db7751e0fe19b2a6d8472c5"
NEW_MODULES = (
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_evidence",
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_lineage",
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_replay",
    "ipfs_datasets_py.duckdb_control.codebase_verification_catalog",
    "ipfs_datasets_py.duckdb_control.codebase_verification_queries",
    "ipfs_datasets_py.duckdb_control.codebase_verification_projection",
    "ipfs_datasets_py.logic.software_contracts.codebase_verification",
    "ipfs_datasets_py.logic.software_contracts.codebase_applicability",
    "ipfs_datasets_py.logic.software_contracts.codebase_smt_execution",
    "ipfs_datasets_py.logic.software_contracts.codebase_smt_protocol",
    "ipfs_datasets_py.logic.software_contracts.codebase_smt_compat",
    "ipfs_accelerate_py.agent_supervisor.planning.codebase_inventory_evidence_context",
    "ipfs_accelerate_py.agent_supervisor.planning.conditional_codebase_evidence",
    "ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview",
    "ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context",
    "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler",
    "ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator",
    "ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts",
    "ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner",
    "ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class PostSetupFitAttempt(AssertionError):
    """A qualification failure, never an acceptable integrity refusal."""


def _error_chain(error):
    rows, seen = [], set()
    while error is not None and id(error) not in seen and len(rows) < 16:
        seen.add(id(error))
        rows.append(error)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return rows


def _installed_libraries_json():
    spec = importlib.util.find_spec("torch")
    require(spec is not None and spec.origin is not None, "installed numerical dependency required")
    libraries = list(dict.fromkeys((str(Path(sysconfig.get_path("purelib")).resolve()),
                                   str(Path(spec.origin).resolve().parent.parent))))
    require(all(Path(path).is_dir() for path in libraries), "installed numerical library directories required")
    return json.dumps(libraries)


def _inventory_launch(executable, argv, *, python, worker, max_workspace_bytes, libraries_json=None):
    """Recognize the native resource wrapper without removing its limits.

    This Linux qualification observes BoundedToolRunner's actual prlimit
    launcher. The raw Popen event remains in the audit; this only classifies
    its exact executable, core/file bounds and inner numerical worker argv.
    """
    require(type(argv) in (list, tuple) and type(max_workspace_bytes) is int
            and max_workspace_bytes > 0, "exact native launch arguments and workspace bound required")
    values = [os.fsdecode(item) for item in argv]
    limiter = Path("/usr/bin/prlimit").resolve()
    require(limiter.is_file() and Path(os.fsdecode(executable)).resolve() == limiter
            and len(values) == 9 and Path(values[0]).resolve() == limiter,
            "native inventory launch must retain its exact prlimit wrapper")
    require(values[1:4] == ["--core=0:0", f"--fsize={max_workspace_bytes}:{max_workspace_bytes}", "--"],
            "native inventory launch core/file limits differ")
    inner = values[4:]
    require(inner[1:3] == ["-I", "-B"] and Path(inner[0]).resolve() == Path(python).resolve()
            and Path(inner[3]).resolve() == Path(worker).resolve(),
            "post-setup scan launched an unexpected parent-observed subprocess")
    require(inner[4] == (_installed_libraries_json() if libraries_json is None else libraries_json),
            "native inventory worker installed library roots differ")
    return {"expected_inventory_worker": True, "native_limits_wrapper": "prlimit",
            "unwrapped_argv": inner, "max_workspace_bytes": max_workspace_bytes}


class JoinProcessAudit:
    """Bound main-parent observations; do not change native resource gates."""

    SCAN_SCOPES = frozenset({"complete_scan_join", "partial_scan_join", "empty_budget_scan_join",
        "inference_budget_scan_join", "canonical_key_scan_join", "epoch_drift_join",
        "closing_join_fault", "pre_cancelled_join"})

    def __init__(self, *, python=None, worker=None, max_events=32768, max_event_bytes=16 * 1024 * 1024):
        require(type(max_events) is int and 0 < max_events <= 32768
            and type(max_event_bytes) is int and 2 <= max_event_bytes <= 16 * 1024 * 1024,
            "bounded exact parent audit retention required")
        self.stage, self.only_git = "startup", False
        self.events, self.counts = [], {}
        self.python = sys.executable if python is None else python
        self.worker = worker
        self.max_events, self.max_event_bytes = max_events, max_event_bytes
        self.retained_event_bytes = self.event_bytes_high_water = 2
        self.event_count_high_water = 0
        self.retention_overflow = False
        self.launch_policy_failure = False

    def observe(self, event, arguments):
        if event != "subprocess.Popen":
            return
        executable, argv, cwd, _environment = arguments
        values = [os.fsdecode(item) for item in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
        name = Path(os.fsdecode(executable)).name
        kind = ("version" if name in {"z3", "cvc5"} and any(item in {"--version", "-version"} for item in values)
            else "solver_query" if name in {"z3", "cvc5"} else "git" if name == "git" else "other")
        key = self.stage + ":" + kind
        self.counts[key] = self.counts.get(key, 0) + 1
        row = {"stage": self.stage, "kind": kind, "executable": os.fsdecode(executable),
               "argv": values, "cwd": None if cwd is None else os.fsdecode(cwd)}
        size = len(base._wire(row)) + bool(self.events)
        if len(self.events) >= self.max_events or self.retained_event_bytes + size > self.max_event_bytes:
            self.retention_overflow = True
            raise AssertionError("parent process audit retention exceeded; qualification must fail")
        self.events.append(row)
        self.retained_event_bytes += size
        self.event_count_high_water = max(self.event_count_high_water, len(self.events))
        self.event_bytes_high_water = max(self.event_bytes_high_water, self.retained_event_bytes)
        if self.only_git and name != "git":
            self.launch_policy_failure = True
            raise AssertionError("lookup attempted non-Git subprocess: " + name)
        if self.stage in self.SCAN_SCOPES and name != "git":
            try:
                require(self.worker is not None, "exact inventory numerical worker required")
                classification = _inventory_launch(executable, values, python=self.python,
                    worker=self.worker, max_workspace_bytes=48 * 1024 * 1024)
            except Exception:
                self.launch_policy_failure = True
                raise
            classified = {**row, **classification}
            delta = len(base._wire(classified)) - len(base._wire(row))
            if self.retained_event_bytes + delta > self.max_event_bytes:
                self.retention_overflow = True
                raise AssertionError("classified parent audit retention exceeded; qualification must fail")
            row.update(classification)
            self.retained_event_bytes += delta
            self.event_bytes_high_water = max(self.event_bytes_high_water, self.retained_event_bytes)

    @contextmanager
    def scope(self, name, *, only_git=False):
        previous = self.stage, self.only_git
        self.stage, self.only_git = name, only_git
        try:
            yield
        finally:
            self.stage, self.only_git = previous

    def to_dict(self):
        return {"counts": self.counts, "events": self.events,
            "scope": "main_parent_process_Popen_events_only_not_native_worker_descendant_solver_observation",
            "limits": {"max_events": self.max_events, "max_serialized_event_bytes": self.max_event_bytes},
            "retained_events": len(self.events), "retained_serialized_event_bytes": self.retained_event_bytes,
            "event_count_high_water": self.event_count_high_water, "event_bytes_high_water": self.event_bytes_high_water,
            "retention_overflow": self.retention_overflow, "overflow_aborts_qualification": True,
            "launch_policy_failure": self.launch_policy_failure, "launch_policy_failure_aborts_qualification": True,
            "serialized_scope": "exact_canonical_JSON_event_array_including_classification_and_delimiters"}


def _query_helper():
    datasets = Path(importlib.import_module(base.PRODUCER_MODULES[0]).__file__).resolve().parents[3]
    path = datasets / "benchmarks/bench_codebase_evidence_queries.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == QUERY_HELPER_SHA256,
            "reviewed authored-intent/query fixture helper generation differs")
    spec = importlib.util.spec_from_file_location("_inventory_evidence_native_query_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _producer_inputs(output, query_helper_path):
    require(hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() == FIXTURE_HELPER_SHA256,
            "unchanged native inventory fixture helper differs")
    names = tuple(dict.fromkeys((*base.PRODUCER_MODULES, *NEW_MODULES)))
    selected = [(name, Path(importlib.import_module(name).__file__).resolve()) for name in names]
    datasets = selected[0][1].parents[3]
    accelerate = Path(__file__).resolve().parents[3]
    selected += [("join_qualification_harness", Path(__file__).resolve()),
                 ("unchanged_inventory_fixture_helper", Path(base.__file__).resolve()),
                 ("reviewed_native_query_fixture_helper", query_helper_path),
                 ("inventory_evidence_unit_controls", datasets / "tests/unit/logic/software_contracts/test_codebase_inventory_evidence.py"),
                 ("supervisor_inventory_evidence_unit_controls", accelerate / "test/api/test_codebase_inventory_evidence_context.py"),
                 ("native_plan_semantic_input_controls", accelerate / "test/api/test_plan_create_semantic_input_identity.py"),
                 ("native_limits_launch_unit_controls", accelerate / "test/benchmarks/test_codebase_inventory_evidence_join_launch_audit.py")]
    copies = output / "producers"
    copies.mkdir()
    rows = []
    for name, path in selected:
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024,
                "selected producer must be a bounded regular file: " + str(path))
        raw = path.read_bytes()
        copy = copies / (name + ".py")
        with copy.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    value = {"schema": "codebase-inventory-evidence-join-selected-inputs@1", "files": rows,
             "capture": "sequential_selected_source_copies", "execution_attestation": False,
             "scope": "listed_local_files_only_not_transitive_dependency_or_execution_attestation"}
    base._write(output / "generation-inputs.json", value)
    return value


def _source_sql(connection):
    tables = connection.execute("SELECT schema_name,table_name FROM duckdb_tables() WHERE database_name=current_database() AND schema_name IN ('codebase_control','codebase_verification_control','codebase_verification_query') ORDER BY schema_name,table_name LIMIT 32").fetchall()
    require(0 < len(tables) < 32, "bounded source/evidence table population differs")
    return {schema + "." + table: connection.execute("SELECT * FROM " + schema + "." + table + " ORDER BY ALL LIMIT 4097").fetchall()
            for schema, table in tables}


def _owners(index, registry, connection):
    return {**base._owners(index, registry), "source_and_evidence_sql": _source_sql(connection)}


@contextmanager
def _restore_sql(connection):
    """Restore only this fresh, private fault fixture, never a retained owner.

    Intentional committed fault injections and native epoch changes are undone
    for independent controls. This is benchmark restoration, not a production
    epoch rollback API or a crash recovery qualification.
    """
    original = _source_sql(connection)
    require(all(len(rows) <= 4096 for rows in original.values()), "fault SQL backup exceeds its bound")
    try:
        yield original
    finally:
        connection.execute("BEGIN TRANSACTION")
        try:
            for table, rows in original.items():
                connection.execute("DELETE FROM " + table)
                if rows:
                    placeholders = ",".join("?" for _ in rows[0])
                    connection.executemany("INSERT INTO " + table + " VALUES (" + placeholders + ")", rows)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        require(_source_sql(connection) == original, "private fault fixture SQL did not restore exactly")


@contextmanager
def _restore_artifact_population(root):
    original = {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}
    require(len(original) <= 256 and sum(map(len, original.values())) <= 32 * 1024 * 1024,
            "private artifact backup exceeds its bound")
    try:
        yield
    finally:
        for path in root.rglob("*"):
            if path.is_file() and path.relative_to(root).as_posix() not in original:
                path.unlink()
        for name, raw in original.items():
            path = root / name
            if not path.is_file() or path.read_bytes() != raw:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)


def _authored_plan(index, repository, head):
    from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
    from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
        ProducerRule, TaskCandidate, TypedIntent, TypedPredicate, obligation_id_for_producer)
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
    from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
        DirtyTreePolicy, PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget, TaskSourceKind, plan_revision_cid)
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import PlanCreateMaterials
    cid = lambda label: plan_revision_cid({"authored_fixture": label})
    roots = PlanAuthorityRoots(repository_id=head.repository_id, task_source_id="source:inventory-join-authored",
        **{name: cid(name) for name in ("repository_root_cid", "dirty_worktree_root", "task_source_revision",
            "policy_root", "intent_ir_root", "legal_ir_root", "security_ir_root", "program_root",
            "capability_catalog_root", "provider_catalog_root", "usage_policy_root", "configuration_root")})
    manifest = index.load(head.manifest_cid)
    roots = replace(roots, repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
                    program_root=manifest.semantic_state.state_cid)
    paths = tuple(f"source{number:02d}.py" for number in range(4))
    request = PlanCreateRequest(prompt_source_cid=cid("prompt"), repository_id=head.repository_id,
        repository_root=str(repository.resolve()), scope_paths=paths,
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND, task_source_kind=TaskSourceKind.BOTH,
        board_namespace="inventory-evidence-native", alias_prefix="JOIN", roots=roots, observe_roots=True,
        budget=PlanRequestBudget(max_model_calls=0, max_latency_ms=90000),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=())
    goals = tuple(TypedPredicate("goal:runtime:" + path, "runtime_exact_integer", path,
                                object_ref="requirement:" + path) for path in paths)
    intent = TypedIntent("intent:inventory-join-authored", goals, ("source:independent-authored-runtime-goals",),
                         current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule("producer:" + path, (goal.predicate_id,)) for path, goal in zip(paths, goals))
    tasks = tuple(TaskCandidate("task:" + path, (obligation_id_for_producer(producer.producer_id, goal.predicate_id),),
                               producer_id=producer.producer_id) for path, producer, goal in zip(paths, producers, goals))
    policy = EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids, evidence_terms=intent.source_refs,
        allowed_scopes=("scope:repository",), available_resource_classes=("cpu",),
        require_validation=True, require_proof=False)
    materials = PlanCreateMaterials(intent=intent, producers=producers, task_candidates=tasks,
        frozen_goal=FrozenPlanningGoal("goal:inventory-join", cid("authored-goal"), head.snapshot_cid, policy),
        candidate_context={"domain": "inventory-evidence-native", "repository_paths": list(paths),
            "task_metadata": {task.candidate_id: {"predicted_files": [path], "scope_ids": ["scope:repository"],
                "resource_classes": ["cpu"]} for task, path in zip(tasks, paths)}},
        extra={"authored_metadata": "runtime_goals_independent_of_conditional_evidence"})
    require(materials.current_facts == () and materials.current_roots is None,
            "authored plan must contain no observed source facts")
    return request, materials


def run(output, *, overall_seconds=300.0):
    require(type(overall_seconds) in (int, float) and math.isfinite(overall_seconds)
            and 0 < overall_seconds <= 600, "bounded finite overall qualification deadline required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    phases, attempts, producers, controls = [], [], [], []
    report = {"schema": SCHEMA, "qualified": False, "phases": phases, "setup_training_attempts": attempts,
        "native_evidence_producers": producers, "controls": controls,
        "scope": "fresh_fixed_8d_cpu_float64_structural_conditional_evidence_model_off_preview",
        "proof_authority": False, "source_execution_attested": False, "production_default_activated": False,
        "cuda_qualified": False, "gradient_synchronization_qualified": False,
        "fresh_process_reopen_qualified": False, "overall_deadline_seconds": overall_seconds,
        "post_setup_fit_attempt_count": 0}
    scheduler = registry = connection = deadline = None
    audit = None

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - began
        base._progress(output / "progress.json", report)

    def phase(name, operation):
        row = {"name": name, "status": "running"}
        phases.append(row)
        progress()
        print("Evidence join qualification: " + name, file=sys.stderr, flush=True)
        started = time.monotonic()
        try:
            value = operation()
            row["status"] = "completed"
            return value
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            progress()

    try:
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_evidence as join
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_scan as scanner
        from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
        from ipfs_datasets_py.logic.software_contracts.codebase_verification import verify_current_codebase_unit
        from ipfs_datasets_py.logic.software_contracts.codebase_applicability import verify_current_codebase_applicability
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
        from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec
        from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain
        from ipfs_datasets_py.logic.ir_core.protocols import ExecutionBounds
        from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
        from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationSelector
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            GlobalResourceScheduler, ResourceSchedulerConfig, ResourceSchedulerError, LeaseCancelledError)
        from ipfs_accelerate_py.agent_supervisor.planning import codebase_inventory_evidence_context as supervisor
        from ipfs_accelerate_py.agent_supervisor.planning import conditional_codebase_evidence as matcher
        from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import (
            RepositoryPlanPreviewOwner, preview_repository_plan)
        from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
            PlanCreateInputSnapshot, PlanCreatePreviewReceipt, freeze_plan_create_input_snapshot)
        helper, helper_path = _query_helper()
        selected = _producer_inputs(output, helper_path)
        report["generation_inputs"] = "generation-inputs.json"
        numerical = importlib.import_module("ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_feature_worker")
        audit = JoinProcessAudit(worker=numerical.__file__)
        sys.addaudithook(audit.observe)
        deadline = helper.Deadline(max(.001, overall_seconds - (time.monotonic() - began)), 120.0)
        report["genuine_host_resources_before"] = asdict(collect_proof_host_resources())
        scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
            state_path=output / "admission.json", lane_reservations={}, auto_renew_leases=True))
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        with audit.scope("fixture"):
            index, repository, head, registry, connection = phase("publish_complete_fixed_inventory",
                lambda: base._native_fixture(output, scheduler))
        catalog = CodebaseVerificationCatalog(index)
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
                      (("source00.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))]

        def options():
            return {**deadline.options(), "scheduler": scheduler, "memory_mb": 1024}

        def fit(kind, epochs, parent=None):
            attempt = {"kind": kind, "requested_epochs": epochs, "completed": False,
                       "actual_completed_epochs": None, "unknown_actual_epochs_on_failure": True}
            attempts.append(attempt)
            progress()
            before = 0 if parent is None else runtimes._read_candidate(
                registry, registry.get_version(parent))["state"]["completed_epochs"]
            record = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=kind, parent_version_id=parent,
                epochs=epochs, learning_rate=.002, seed=1729, **options())
            saved = runtimes._read_candidate(registry, registry.get_version(record.to_dict()["version_id"]))
            actual = saved["state"]["completed_epochs"] - before
            require(actual == epochs, "setup completed epoch delta differs")
            attempt.update(completed=True, actual_completed_epochs=actual, unknown_actual_epochs_on_failure=False,
                           cumulative_model_epochs=saved["state"]["completed_epochs"], version_id=record.to_dict()["version_id"])
            base._write(output / (kind + ".json"), record.to_dict())
            progress()
            return record

        with audit.scope("training_setup"):
            root = phase("fit_private_root_two_epochs", lambda: fit("root", 2))
            child = phase("fit_private_same_head_child_one_epoch", lambda: fit("child", 1, root.to_dict()["version_id"]))
        version = child.to_dict()["version_id"]
        report.update(setup_actual_completed_epochs=sum(a["actual_completed_epochs"] for a in attempts),
                      scanned_version_id=version, ancestry_version_ids=[version, root.to_dict()["version_id"]])
        identities = []
        native_cases = ((0, 1, "True", "recorded_conditional_proved"),
                        (1, 3, "True", "recorded_conditional_refuted"),
                        (2, 3, "False", "recorded_conditional_vacuous"),
                        (3, 4, "True", "recorded_conditional_proved"),
                        (3, 4, "True", "recorded_conditional_proved"))
        for number, (source_number, requested_offset, predicate, expected_status) in enumerate(native_cases):
            path = f"source{source_number:02d}.py"
            contract = ContractSpec("step", postconditions=(f"result == n + {requested_offset}",))
            domain = RequestedInputDomain("step", (predicate,), "domain:" + path)
            bounds = ExecutionBounds(timeout_ms=5000 + (number == 4), max_steps=100000,
                max_memory_bytes=128 * 1024 * 1024, max_output_bytes=256 * 1024)
            with audit.scope("native_verification_setup"):
                verification = phase(f"native_verify_{number:02d}_{path}", lambda path=path, contract=contract, bounds=bounds:
                    verify_current_codebase_unit(index, repository, expected_head=head, path=path,
                        contracts=[contract], bounds=bounds, **options()))
                applicability = phase(f"native_applicability_{number:02d}_{path}", lambda:
                    verify_current_codebase_applicability(index, repository, expected_head=head,
                        verification_cid=verification.artifact_cid, domains=[domain], **options()))
            with audit.scope("native_publish_setup", only_git=True):
                projection = phase(f"publish_conditional_{number:02d}_{path}", lambda:
                    catalog.publish(repository, expected_head=head, verification_cid=verification.artifact_cid,
                        applicability_cid=applicability.artifact_cid, operation_id=f"conditional:{number}", **options()))
            require(verification.observed_live is True and applicability.observed_live is True,
                    "native evidence setup did not return observed-live producer records")
            value = projection.to_dict()
            selected_contract = value["contracts"][0]
            identity = {"path": path, "contract_id": contract.contract_id, "contract_cid": cid_for_structured(contract.to_dict()),
                "domain_id": domain.domain_id, "domain_cid": cid_for_structured(domain.to_dict()),
                "verification_cid": verification.artifact_cid, "applicability_cid": applicability.artifact_cid,
                "projection_cid": projection.projection_cid, "key_id": selected_contract["canonical_keys"][0]["key_id"],
                "source_cid": value["source_binding"]["entry"]["source_cid"], "expected_status": expected_status}
            identities.append(identity)
            for kind, native in (("verification", verification), ("applicability", applicability)):
                base._write(output / f"native-{number:02d}-{kind}.json", native.to_dict())
            producers.append({**identity, "requested_bounds": bounds.to_dict(),
                "native_verification_record": f"native-{number:02d}-verification.json",
                "native_applicability_record": f"native-{number:02d}-applicability.json",
                "verification_process_observations": verification.to_dict()["process_observations"],
                "applicability_process_observations": applicability.to_dict()["process_observations"],
                "native_phase_receipts_retained": True, "observed_live_setup": True,
                "parent_audit_does_not_observe_nested_solver_processes": True})
            base._assert_clean(scheduler)
            progress()
        require(identities[3]["verification_cid"] != identities[4]["verification_cid"],
                "independent native verification configuration failed to create an ambiguous pair")

        def never_fit(*args, **kwargs):
            report["post_setup_fit_attempt_count"] += 1
            raise PostSetupFitAttempt("post-setup join attempted model fitting")

        baseline = _owners(index, registry, connection)
        base._write(output / "owners-before-joins.json", baseline)
        saved = runtimes._read_candidate(registry, registry.get_version(version))
        numerical = {"state_sha256": features.digest(saved["state"]), "completed_epochs": saved["state"]["completed_epochs"],
                     "adam_steps": [item["step"] for item in saved["state"]["adam"]]}
        report["numerical_before"] = numerical
        report["latent_width"] = saved["state"]["latent_width"]
        report["frozen_feature_columns"] = len(saved["feature_space"]["columns"])
        report["evidence_identities"] = identities
        join_args = {"expected_head": head, "registry": registry, "version_id": version, "verification_catalog": catalog}

        def scan_join(**changes):
            return join.scan_current_codebase_evidence(index, repository, **{**join_args, **options(), **changes})

        def validate(record, **changes):
            return join.validate_current_inventory_evidence(record, index, repository,
                registry=registry, verification_catalog=catalog, expected_head=head, **{**options(), **changes})

        def preserved():
            require(_owners(index, registry, connection) == baseline, "inference/query/preview changed source/model/evidence owners")
            require(base._verify_producer_inputs(output, selected)["current_and_retained_copies_unchanged"] is True,
                    "selected implementation generation changed")
            return base._assert_clean(scheduler)

        with base._patch(features, "train_projection_features", never_fit), \
                base._patch(runtimes.SourceBoundCodebaseFeatureRuntime, "train", never_fit), \
                base._patch(training, "train_current_codebase_features", never_fit):
            with audit.scope("receiving_and_matcher", only_git=True):
                matcher_results = []
                for number, identity in enumerate(identities):
                    source_number, requested_offset, predicate, expected_status = native_cases[number]
                    unit = helper.FixtureUnit(identity["path"], source_number + 1, requested_offset, predicate, expected_status)
                    contract = ContractSpec("step", postconditions=(f"result == n + {requested_offset}",))
                    domain = RequestedInputDomain("step", (predicate,), "domain:" + identity["path"])
                    text, document = helper.authored_intent(unit, contract, domain)
                    statements = tuple(replace(statement, arguments=tuple("step" if item == "increment" else item
                        for item in statement.arguments)) for statement in document.statements)
                    document = replace(document, statements=statements)
                    document.validate()
                    result = matcher.match_conditional_codebase_intent(catalog=catalog, index=index, repository=repository,
                        repository_id=head.repository_id, expected_head=head, intent_document=document,
                        source_text=text, path=identity["path"], contract=contract, domain=domain,
                        statement_id="mathematical-goal", verification_cid=identity["verification_cid"],
                        expected_key_id=identity["key_id"], **options())
                    require(result["status"] == expected_status, "native conditional matcher status differs")
                    require(result["current_facts"] == result["removed_task_ids"] == [], "historical evidence elevated runtime facts")
                    matcher_results.append(result)
                ambiguous = matcher.match_conditional_codebase_intent(catalog=catalog, index=index, repository=repository,
                    repository_id=head.repository_id, expected_head=head, intent_document=document,
                    source_text=text, path=identities[4]["path"], contract=contract, domain=domain,
                    statement_id="mathematical-goal", **options())
                require(ambiguous["status"] == "unknown" and
                    "ambiguous_exact_current_evidence_requires_explicit_selector" in ambiguous["reasons"]
                    and len(ambiguous["indexed_query_page"]["entries"]) == 2,
                    "two genuine source03 records did not retain explicit ambiguity")
                report["native_ambiguous_matcher_control"] = ambiguous
                report["native_matcher_controls"] = matcher_results

            complete_limits = join.CodebaseInventoryEvidenceLimits(page_size=1)
            with audit.scope("complete_scan_join"):
                complete = phase("complete_page_size_one_join", lambda: scan_join(limits=complete_limits))
            body = complete.to_dict()
            base._write(output / "complete-join.json", body)
            require(len(body["entries"]) == 28 and body["query"]["complete"] is True,
                    "complete join lost source membership or evidence pagination")
            require(len(body["query"]["pages"]) == 5, "page-size-one join did not traverse all five native evidence entries")
            require(complete.artifact_cid == cid_for_bytes(base._wire(body)), "joined raw record CID differs")
            require({item["query_entry"]["verification_cid"] for item in body["evidence"]}
                    == {item["verification_cid"] for item in identities}, "complete join omitted a native conditional record")
            require(sum(row["evidence_disposition"] == "matched_complete" for row in body["entries"]) == 4
                and sum(row["evidence_disposition"] == "no_exact_indexed_conditional_evidence" for row in body["entries"]) == 24,
                "complete join collapsed source/evidence membership or misclassified scoped absence")
            for item in body["evidence"]:
                if item["query_entry"]["path"] == "source01.py":
                    require(item["verification_status"] == "recorded_conditional_refuted",
                            "joined refutation status differs from native source01 evidence")
                if item["query_entry"]["path"] == "source02.py":
                    require(item["applicability_status"] == "empty_domain"
                            and item["applicability_summary"]["status"] == "empty_domain",
                            "joined native vacuity/applicability ledger was omitted")
            source03_key = next(row["source_key"] for row in body["scan"]["record"]["entries"] if row["path"] == "source03.py")
            source03_ledger = next(row for row in body["entries"] if row["source_key"] == source03_key)
            require(len(source03_ledger["evidence_entry_ids"]) == 2, "ambiguous source03 evidence pair was collapsed")
            preserved()
            with audit.scope("receiving_validation", only_git=True):
                validated = phase("validate_current_without_inference", lambda: validate(complete))
            require(validated.artifact_cid == complete.artifact_cid, "receiving validation changed immutable join")
            preserved()
            with audit.scope("partial_scan_join"):
                partial = phase("partial_max_pages_one_join", lambda:
                    scan_join(limits=replace(complete_limits, max_pages=1)))
            base._write(output / "partial-join.json", partial.to_dict())
            require(partial.to_dict()["query"]["complete"] is False and partial.to_dict()["query"]["next_cursor"] is not None,
                    "bounded join converted unqueried evidence to absence")
            require(sum(row["evidence_disposition"] == "unknown_budget" for row in partial.to_dict()["entries"]) == 27
                and not any(row["evidence_disposition"] == "no_exact_indexed_conditional_evidence"
                            for row in partial.to_dict()["entries"]), "partial traversal fabricated complete evidence absence")
            preserved()
            with audit.scope("empty_budget_scan_join"):
                empty = phase("one_byte_query_budget_zero_page_join", lambda:
                    scan_join(limits=replace(complete_limits, max_query_bytes=1)))
            empty_body = empty.to_dict()
            base._write(output / "empty-budget-join.json", empty_body)
            require(len(empty_body["entries"]) == 28 and empty_body["query"]["complete"] is False
                and empty_body["query"]["next_cursor"] is None and empty_body["query"]["pages"] == []
                and empty_body["evidence"] == [] and all(row["evidence_disposition"] == "unknown_budget"
                    and row["evidence_entry_ids"] == [] for row in empty_body["entries"]),
                "zero consumed pages fabricated evidence absence or lost complete source membership")
            with audit.scope("empty_budget_receiving_validation", only_git=True):
                empty_validated = phase("validate_current_zero_page_join", lambda: validate(empty))
            require(empty_validated.artifact_cid == empty.artifact_cid, "zero-page receiving validation selected another immutable root")
            preserved()
            with audit.scope("inference_budget_scan_join"):
                deferred = phase("inference_budget_four_join", lambda:
                    scan_join(limits=complete_limits, scan_limits=replace(scanner.CodebaseInventoryScanLimits(), max_inferred_rows=4)))
            base._write(output / "deferred-join.json", deferred.to_dict())
            require(any(row["path"] == "source03.py" and row["disposition"] == "deferred_budget"
                        for row in deferred.to_dict()["scan"]["record"]["entries"]),
                    "budget fixture did not defer its evidence-bearing source")
            require(next(row for row in deferred.to_dict()["entries"] if row["source_key"] == source03_key)
                    == source03_ledger, "feature budget hid independently indexed conditional source03 evidence")
            preserved()
            identity = identities[0]
            exact_selector = CodebaseVerificationSelector(path=identity["path"], contract_id=identity["contract_id"],
                expected_contract_cid=identity["contract_cid"], verification_cid=identity["verification_cid"],
                canonical_key_id=identity["key_id"], requested_domain_id=identity["domain_id"], requested_domain_cid=identity["domain_cid"])
            with audit.scope("canonical_key_scan_join"):
                exact = phase("exact_canonical_key_join", lambda: scan_join(limits=complete_limits, selector=exact_selector))
            base._write(output / "exact-key-join.json", exact.to_dict())
            require(exact.to_dict()["query"]["complete"] is True and len(exact.to_dict()["evidence"]) == 1
                and exact.to_dict()["evidence"][0]["query_entry"]["verification_cid"] == identity["verification_cid"],
                "complete canonical-key selector joined a different conditional record")
            preserved()

            request, materials = _authored_plan(index, repository, head)
            original_materials = deepcopy(materials.to_binding_dict())
            authored_snapshot = freeze_plan_create_input_snapshot(request, materials=materials)

            def preview(record, policy_observer=None):
                return supervisor.preview_current_inventory_evidence_plan(record, index, repository,
                    verification_catalog=catalog, registry=registry, request=request, materials=materials,
                    policy_observer=policy_observer or (lambda typed: typed.roots), **options())

            with audit.scope("native_plan_preview", only_git=True):
                baseline_preview = phase("preview_independent_authored_plan", lambda: preview_repository_plan(
                    owner=RepositoryPlanPreviewOwner(index=index, repository=repository.resolve(), expected_head=head,
                        scheduler=scheduler, cancel_event=deadline.cancelled,
                        timeout_seconds=min(90.0, deadline.options()["timeout_seconds"]), memory_mb=1024),
                    request=request, materials=materials, policy_observer=lambda typed: typed.roots))
                full_preview = phase("preview_complete_join", lambda: preview(complete))
                partial_preview = phase("preview_partial_join", lambda: preview(partial))
                empty_preview = phase("preview_zero_page_join", lambda: preview(empty))
            base._write(output / "independent-authored-plan-preview.json", baseline_preview)
            baseline_snapshot = PlanCreateInputSnapshot.from_dict(baseline_preview["input_snapshot"])
            for name, value in (("complete-plan-preview.json", full_preview), ("partial-plan-preview.json", partial_preview),
                                ("empty-budget-plan-preview.json", empty_preview)):
                base._write(output / name, value)
                require(value["declared_task_ids"] == [task.candidate_id for task in materials.task_candidates],
                        "advisory join removed or replaced an independently authored task")
                require(set(value["declared_requirement_ids"]) == set(materials.intent.goal_predicate_ids),
                        "advisory join lost runtime requirements")
                require(value["current_facts"] == value["removed_task_ids"] == [] and all(
                        flag is False for flag in value["authority"].values()), "preview elevated conditional authority")
                snapshot = PlanCreateInputSnapshot.from_dict(value["repository_preview"]["input_snapshot"])
                receipt = PlanCreatePreviewReceipt.from_dict(value["repository_preview"]["preview"])
                require(receipt.input_snapshot_cid == snapshot.snapshot_cid, "generic native preview snapshot CID differs")
                for field in ("intent", "producers", "task_candidates", "frozen_goal", "predicates", "current_facts"):
                    require(snapshot.material_binding["field_digests"][field]
                        == baseline_snapshot.material_binding["field_digests"][field]
                        == authored_snapshot.material_binding["field_digests"][field],
                        "joined advisory material changed independently authored " + field)
                stages = {item.stage.value: item for item in receipt.stage_results}
                require(stages["obligation"].passed is True and stages["candidate"].passed is True,
                        "generic native planner did not compile the declared obligation/task inputs")
                require(snapshot.snapshot_cid != baseline_snapshot.snapshot_cid,
                        "joined advisory references were omitted from generic native semantic input identity")
            require(materials.to_binding_dict() == original_materials, "preview mutated caller-owned plan materials")
            require(full_preview["repository_preview"]["input_snapshot"]["snapshot_cid"]
                    != partial_preview["repository_preview"]["input_snapshot"]["snapshot_cid"],
                    "generic plan input identity omitted joined advisory record")
            preserved()
            report["positive_join_records"] = {"complete": complete.artifact_cid, "partial": partial.artifact_cid,
                "zero_consumed_pages": empty.artifact_cid, "inference_budget_four": deferred.artifact_cid,
                "exact_canonical_key": exact.artifact_cid}
            report["positive_plan_preview"] = {"task_population_preserved": True, "runtime_requirements_residual": True,
                "current_facts_supplied": 0, "advisory_record_changes_input_snapshot": True, "materials_unchanged": True}

            def refuse(name, operation, details=None):
                error = None
                try:
                    operation()
                except (PostSetupFitAttempt, ResourceSchedulerError, TimeoutError) as caught:
                    if name != "pre_cancelled_join" or type(caught) is not LeaseCancelledError:
                        raise
                    error = caught
                except (ValueError, RuntimeError) as caught:
                    if name == "pre_cancelled_join":
                        raise AssertionError("pre-cancel control did not raise exact LeaseCancelledError") from caught
                    if any(isinstance(cause, (PostSetupFitAttempt, ResourceSchedulerError, TimeoutError))
                           for cause in _error_chain(caught)):
                        raise
                    error = caught
                if error is None:
                    raise AssertionError(name + " accepted its intentionally corrupted current input")
                require(not audit.retention_overflow and not audit.launch_policy_failure,
                        "masked parent audit failure cannot qualify an integrity refusal")
                controls.append({"name": name, "refused": True, "error_type": type(error).__name__,
                    "error": str(error), "failure_chain": [{"type": type(cause).__name__, "message": str(cause)}
                        for cause in _error_chain(error)], "details": details or {},
                    "resource_pressure_or_timeout_failure_accepted": False,
                    "expected_pre_cancellation_accepted": name == "pre_cancelled_join",
                    "resources_after": base._assert_clean(scheduler)})
                progress()

            # A new valid raw checksum does not certify typed membership or
            # fresh replay. Report the receiving stage where rejection occurs.
            def receiving_mutation(name, mutate, *, require_live_validation=False):
                value = deepcopy(body)
                mutate(value)
                raw = base._wire(value)
                details = {"raw_checksum_recomputed": True, "receiving_validation_invoked": False}
                def receive():
                    record = join.CodebaseInventoryEvidenceRecord(cid_for_bytes(raw), raw)
                    details["receiving_validation_invoked"] = True
                    return validate(record)
                with audit.scope("rehashed_receiving_control", only_git=True):
                    phase(name, lambda: refuse(name, receive, details))
                require(not require_live_validation or details["receiving_validation_invoked"] is True,
                        "status fault did not reach fresh native receiving replay")
                preserved()

            def changed_scan(value, field):
                scan_body = value["scan"]["record"]
                if field == "source":
                    next(row for row in scan_body["entries"] if row["path"] == "source00.py")["source_cid"] = identities[1]["source_cid"]
                else:
                    scan_body["model"]["state_sha256"] = "0" * 64
                value["scan"]["artifact_cid"] = cid_for_bytes(base._wire(scan_body))
            receiving_mutation("rehashed_scan_source_identity", lambda value: changed_scan(value, "source"))
            receiving_mutation("rehashed_scan_model_identity", lambda value: changed_scan(value, "model"), require_live_validation=True)
            receiving_mutation("rehashed_inventory_membership", lambda value: value["entries"].pop())
            receiving_mutation("rehashed_exact_selector", lambda value: value["selector"].__setitem__("path", "source19.py"))
            receiving_mutation("rehashed_page_epoch_type", lambda value:
                value["query"]["pages"][0]["page"].__setitem__("epoch", True))
            receiving_mutation("rehashed_continuation_cursor", lambda value:
                value["query"]["pages"][1]["page"]["start_cursor"].__setitem__("epoch", 999999))
            receiving_mutation("rehashed_recorded_conditional_status", lambda value:
                next(item for item in value["evidence"] if item["query_entry"]["path"] == "source00.py").__setitem__(
                    "verification_status", "recorded_conditional_refuted"), require_live_validation=True)

            with _restore_sql(connection):
                connection.execute("DELETE FROM codebase_verification_query.entries WHERE entry_id=(SELECT entry_id FROM codebase_verification_query.entries ORDER BY entry_id LIMIT 1)")
                with audit.scope("sql_membership_control", only_git=True):
                    phase("deleted_actual_normalized_membership", lambda:
                        refuse("deleted_actual_normalized_membership", lambda: validate(complete)))
            preserved()
            for kind, path in (("captured_source", index.artifacts.path_for(identities[0]["source_cid"], source=True)),
                               ("conditional_verification", index.artifacts.path_for(identities[0]["verification_cid"])),
                               ("conditional_projection", index.artifacts.path_for(identities[0]["projection_cid"]))):
                with base._restore_bytes(path) as original:
                    path.write_bytes(b"!" + original[1:])
                    with audit.scope("cas_integrity_control", only_git=True):
                        phase("tampered_" + kind, lambda kind=kind:
                            refuse("tampered_" + kind, lambda: validate(complete)))
                preserved()
            proof_path = index.artifacts.path_for(identities[0]["verification_cid"])
            with base._restore_bytes(proof_path):
                proof_path.unlink()
                with audit.scope("cas_missing_control", only_git=True):
                    phase("missing_conditional_verification", lambda:
                        refuse("missing_conditional_verification", lambda: validate(complete)))
            preserved()

            native_query = type(catalog).query_current
            for mode in ("append", "rebuild"):
                changed = {"genuine_native_page_returned": False, "native_same_head_change_committed": False}
                def drifting_query(owner, *args, **kwargs):
                    page = native_query(owner, *args, **kwargs)
                    if owner is catalog and not changed["native_same_head_change_committed"]:
                        changed["genuine_native_page_returned"] = True
                        if mode == "append":
                            catalog.publish(repository, expected_head=head, verification_cid=identities[0]["verification_cid"],
                                operation_id="fault:same-head-without-domain", **options())
                        else:
                            catalog.rebuild_current(repository, expected_head=head, **options())
                        changed["native_same_head_change_committed"] = True
                    return page
                with _restore_sql(connection), _restore_artifact_population(index.artifacts.root):
                    with base._patch(type(catalog), "query_current", drifting_query), audit.scope("epoch_drift_join"):
                        phase("same_head_" + mode + "_during_join", lambda mode=mode:
                            refuse("same_head_" + mode + "_during_join", lambda: scan_join(limits=complete_limits), changed))
                require(all(changed.values()), "epoch drift did not follow a genuine native query page")
                preserved()

            with _restore_sql(connection), _restore_artifact_population(index.artifacts.root):
                changed = {"genuine_generic_policy_observer_invoked": False, "native_rebuild_committed": False}
                def changed_policy(typed):
                    if not changed["native_rebuild_committed"]:
                        changed["genuine_generic_policy_observer_invoked"] = True
                        catalog.rebuild_current(repository, expected_head=head, **options())
                        changed["native_rebuild_committed"] = True
                    return typed.roots
                with audit.scope("epoch_drift_preview", only_git=True):
                    phase("same_head_rebuild_during_preview", lambda:
                        refuse("same_head_rebuild_during_preview", lambda: preview(complete, changed_policy), changed))
                require(all(changed.values()), "preview epoch control did not reach its real policy observation")
            preserved()

            native_model_fence = scanner._model_fence
            changed = {"genuine_native_model_fence_returned": False, "native_same_head_rebuild_committed": False}
            def drift_after_model_fence(*args, **kwargs):
                result = native_model_fence(*args, **kwargs)
                if not changed["native_same_head_rebuild_committed"]:
                    changed["genuine_native_model_fence_returned"] = True
                    catalog.rebuild_current(repository, expected_head=head, **options())
                    changed["native_same_head_rebuild_committed"] = True
                return result
            with _restore_sql(connection), _restore_artifact_population(index.artifacts.root):
                with base._patch(scanner, "_model_fence", drift_after_model_fence), audit.scope("late_evidence_epoch_control", only_git=True):
                    phase("same_head_rebuild_after_closing_model_fence", lambda:
                        refuse("same_head_rebuild_after_closing_model_fence", lambda: validate(complete), changed))
            require(all(changed.values()), "late epoch control did not follow a genuine native model fence")
            preserved()

            checkpoint = registry.artifact_path(registry.get_version(version)["artifact"])
            source_path = repository / "source00.py"
            for mode in ("source", "head", "checkpoint"):
                changed = {"genuine_terminal_native_page_returned": False, "fault_after_terminal_page": False}
                def closing_fault(owner, *args, **kwargs):
                    page = native_query(owner, *args, **kwargs)
                    if owner is catalog and page.complete and not changed["fault_after_terminal_page"]:
                        changed["genuine_terminal_native_page_returned"] = True
                        if mode == "source":
                            source_path.write_bytes(source_path.read_bytes() + b"# changed after evidence page\n")
                        elif mode == "head":
                            connection.execute("UPDATE codebase_control.heads SET generation=generation+1 WHERE repository_id=?", [head.repository_id])
                        else:
                            checkpoint.write_bytes(checkpoint.read_bytes() + b"\n")
                        changed["fault_after_terminal_page"] = True
                    return page
                with _restore_sql(connection), base._restore_bytes(source_path), base._restore_bytes(checkpoint):
                    with base._patch(type(catalog), "query_current", closing_fault), audit.scope("closing_join_fault"):
                        phase(mode + "_change_after_terminal_native_page", lambda mode=mode:
                            refuse(mode + "_change_after_terminal_native_page", lambda: scan_join(limits=complete_limits), changed))
                require(all(changed.values()), "closing fault did not follow a genuine terminal native evidence page")
                preserved()
            cancelled = threading.Event()
            cancelled.set()
            with audit.scope("pre_cancelled_join"):
                phase("pre_cancelled_join", lambda: refuse("pre_cancelled_join", lambda: scan_join(cancel_event=cancelled)))
            preserved()
            report["post_setup_training_executed"] = False

        final_saved = runtimes._read_candidate(registry, registry.get_version(version))
        final_numerical = {"state_sha256": features.digest(final_saved["state"]),
            "completed_epochs": final_saved["state"]["completed_epochs"],
            "adam_steps": [item["step"] for item in final_saved["state"]["adam"]]}
        require(final_numerical == numerical, "joined query or preview changed the frozen model state")
        report["numerical_after"] = final_numerical
        base._write(output / "owners-after-joins.json", _owners(index, registry, connection))
        report["final_owner_preservation"] = _owners(index, registry, connection) == baseline
        report["selected_inputs_ending"] = base._verify_producer_inputs(output, selected)
        report["final_resources"] = base._assert_clean(scheduler)
        report["parent_process_audit"] = audit.to_dict()
        report["genuine_host_resources_after"] = asdict(collect_proof_host_resources())
        require(report["final_owner_preservation"], "final owner export differs")
        require(report["post_setup_fit_attempt_count"] == 0, "post-setup fitting guard was invoked")
        require(not audit.retention_overflow and not audit.launch_policy_failure,
                "parent process audit overflow or unexpected launch invalidates qualification")
        require(all(row.get("expected_inventory_worker") is True for row in audit.events
            if row["stage"] in audit.SCAN_SCOPES and row["kind"] != "git"),
            "every observed scan subprocess must retain its classified native resource wrapper")
        deadline.options()
        report["qualified"] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
            failure_chain=[{"type": type(cause).__name__, "message": str(cause)} for cause in _error_chain(error)])
    finally:
        if deadline is not None:
            deadline.close()
        report["recorded_seconds"] = time.monotonic() - began
        report["known_actual_setup_epochs"] = sum(a["actual_completed_epochs"] or 0 for a in attempts)
        report["unknown_fitting_epochs"] = any(a["unknown_actual_epochs_on_failure"] for a in attempts)
        if audit is not None:
            report.setdefault("parent_process_audit", audit.to_dict())
        if scheduler is not None:
            try:
                report.setdefault("final_resources", base._assert_clean(scheduler))
            except BaseException as error:
                report["resource_cleanup_error"] = type(error).__name__ + ": " + str(error)
        if registry is not None:
            registry.close()
        if connection is not None:
            connection.close()
        progress()
        base._write(output / "result.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overall-seconds", type=float, default=300.0)
    args = parser.parse_args()
    require(0 < args.overall_seconds <= 600, "overall qualification deadline must be finite and <=600 seconds")
    result = run(args.output, overall_seconds=args.overall_seconds)
    print(json.dumps({"qualified": result["qualified"], "recorded_seconds": result["recorded_seconds"],
        "known_actual_setup_epochs": result["known_actual_setup_epochs"],
        "unknown_fitting_epochs": result["unknown_fitting_epochs"], "error": result.get("error")}, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 1)
