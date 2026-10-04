"""Authored capture -> actual Z3 -> typed operation -> native admission replay.

No public benchmark code, learned model, provider or completion claim. Native
resource admission uses an explicit isolated scheduler with authored telemetry;
these are mechanism controls, not a host-pressure qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test.api.test_intent_plan_coverage import _ledger
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as planning
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import build_intent_planning_materials
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as owner
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

PROGRAM = '''# Authored fixture; never imported by formalization or planning.
def stringify(raw, encoding='utf8', errors='strict'):
    if isinstance(raw, (bytes, bytearray)):
        return str(raw, encoding, errors)
    return '' if raw is None else str(raw)

def field_label(raw):
    label = stringify(raw)
    return label.title().replace('_', '-')

def field_payload(raw):
    payload = stringify(raw)
    return payload

class Reply:
    def put(self, label, payload):
        self._fields[field_label(label)] = [field_payload(payload)]

    @property
    def wire_fields(self):
        pairs = list(self._fields.items())
        return [(label, value) for label, values in pairs for value in values]

def application(environ, emit):
    reply = Reply()
    emit('200 OK', reply.wire_fields)
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def runtime_config(root, selection):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    return dict(schema="terminal-source384-config@2", mode="pinned_parent",
        checkpoint_path=str(root / "fixture-checkpoint-not-loaded.pt"), checkpoint_sha256="0" * 64,
        embedding_snapshot=str(root / "fixture-embedding-not-loaded"), embedding_revision=embedding.PINNED_REVISION,
        embedding_assets=[{"name": name, "sha256": sha, "bytes": size}
            for name, (size, sha) in sorted(embedding._PINNED_ASSETS.items())],
        training_steps=0, download_calls=0,
        header_applicability=dict(schema=owner.PROFILE_SCHEMA, selector_cid=cid_for_dag_json(selection),
            checker_profile=owner.CHECKER_PROFILE, solver_sha256=sha(Path(shutil.which("z3")).resolve().read_bytes())))


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    import duckdb
    from ipfs_datasets_py.logic.software_contracts import codebase_header_context as captured
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources

    z3 = shutil.which("z3")
    assert z3, "actual native Z3 is required for this qualification"
    config = resources.ResourceSchedulerConfig.for_proof_host(state_path=tmp_path / "resource.json",
        proof_resource_sampler=lambda: ProofHostResources(8, 16384, 16384),
        lane_reservations={}, auto_renew_leases=False, poll_interval_seconds=.005)
    scheduler = resources.GlobalResourceScheduler(config)
    monkeypatch.setattr(codebase_resources, "get_global_resource_scheduler", lambda: scheduler)
    text, ledger = _ledger(objects=("headers",))
    program = PROGRAM
    source_variant = getattr(request, "param", "unsafe")
    if source_variant == "guarded":
        reviewed = captured.contracts.WsgiHeaderProtocolContract("review:authored-header-protocol@1", "emit")
        program = captured.contracts.analyze_http_header_contracts(PROGRAM, protocol=reviewed).candidate.source
    elif source_variant == "unsupported":
        program = "def field_payload(raw):\n    return raw\n"
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "headers.py").write_text(program)
    (repo / "instruction.txt").write_text(text)
    (repo / "test_headers.py").write_text(
        "from headers import field_payload\n"
        "try:\n    field_payload('bad\\nvalue')\n"
        "except ValueError:\n    pass\nelse:\n    raise AssertionError('unsafe header accepted')\n")
    for args in (("init", "-q"), ("config", "user.name", "Authored fixture"),
                 ("config", "user.email", "fixture@example.invalid"),
                 ("add", "."), ("commit", "-qm", "authored input")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=repo, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    requirement = ledger["requirements"][0]
    native = ledger["source_report"]["candidates"][0]["candidate_intent_ir"]["statements"][0]
    outputs = [{"path": "headers.py", "effect": "modify", "media_type": "text/x-python"}]
    operation = {"operation_id": "operation:reviewed-header-guard", "task_key": "HEADER-TASK",
        "matchers": [{"requirement_id": requirement["requirement_id"],
            "native_document_sha256": requirement["native_document_sha256"],
            "statement_id": native["statement_id"], "predicate": native["predicate"],
            "arguments": native["arguments"], "modality": native["modality"]}],
        "outputs": outputs, "validation_keys": ["public-header-guard"], "dependency_operation_ids": []}
    selection = dict(schema=owner.SCHEMA, review_ref="review:authored-requirement-to-guard@1",
        operation_id=operation["operation_id"], operator_id=owner.OPERATOR, source_path="headers.py",
        protocol={"review_ref": "review:authored-header-protocol@1", "callback_parameter": "emit"},
        checker_profile=owner.CHECKER_PROFILE, **owner.FALSE)
    contract = {"schema": owner.CONTRACT_SCHEMA, "source_path": "instruction.txt", "ledger": ledger,
        "requirements": [{"requirement_id": requirement["requirement_id"], "outputs": outputs,
            "validation_keys": ["public-header-guard"], "dependency_requirement_ids": []}],
        "symbolic_operations": {"schema": "intent-symbolic-operation-contract@1",
            "ledger_sha256": ledger["ledger_sha256"], "review_ref": "review:authored-task-mapping@1",
            "interpretation_scope": "administrative_requirement_task_coverage", "operations": [operation],
            "semantic_alignment_verified": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False},
        "source_applicability": selection}
    specs = [{"task_key": "HEADER-TASK", "scope_paths": ["headers.py", "test_headers.py"],
        "outputs": outputs, "dependencies": [],
        "validations": [{"validation_key": "public-header-guard", "argv": [sys.executable, "test_headers.py"],
            "cwd": ".", "expected_exit_codes": [0], "policy_cid": content_identity(local.LOCAL_POLICY)}],
        "acceptance": [{"criterion_key": "header-guard", "criterion": "The authored unsafe-header check passes",
            "evidence_cids": [], "validation_keys": ["public-header-guard"]}]}]
    roots = {key: content_identity({"authored": key}) for key in ("request_cid", "scan_cid", "program_root")}
    def author(value):
        return local.author_local_benchmark_manifest(repository=repo, profile_dir=profile,
            lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots, intent_requirements=value)
    manifest = author(contract)
    signed_before_capture = wire(manifest)
    assert "captured_receipt" not in wire(contract).decode()
    store_root = tmp_path / "captured"; store_root.mkdir(mode=0o700)
    config = runtime_config(tmp_path, selection)
    config_path = tmp_path / "source-config.json"; config_path.write_bytes(wire(config))
    with duckdb.connect(str(store_root / "source.duckdb"), config={"threads": 1, "memory_limit": "128MB"}) as cx:
        store = DuckDBASTStore(connection=cx); cas = ImmutableCAS(store_root / "source-artifacts")
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=cas,
            catalog=CodebaseCatalog(store, cas))
        head = index.prepare_current(repo, repository_id="authored-header-baseline", operation_id="capture",
            expected_head=None, limits=CodebaseScanLimits(max_entries=8, max_file_bytes=1_000_000),
            scheduler=scheduler, memory_mb=512).head
        with captured._operation(scheduler=scheduler, parent_lease=None, cancel_event=None,
                timeout_seconds=30., memory_mb=1024) as (lease, signal, remaining):
            nomination = owner.prepare_runtime_nomination(index, head=head, output=store_root,
                config_path=config_path, config=config,
                intent_binding={"contract": contract, "manifest_cid": content_identity(manifest)},
                source_hashes={name: row["sha256"] for name, row in manifest["payload"]["sources"].items()},
                parent_lease=lease, remaining=remaining)
    assert wire(manifest) == signed_before_capture
    yield SimpleNamespace(repo=repo, root=tmp_path, store_root=store_root, contract=contract,
        manifest=manifest, author=author, scheduler=scheduler, selection=selection, nomination=nomination,
        config=config, config_path=config_path)
    snap = scheduler.snapshot()
    assert snap["active_lease_count"] == snap["waiting_request_count"] == 0


def test_real_checker_is_a_typed_precondition_and_cold_native_admission_replays(case, monkeypatch):
    c = case
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    def forbidden_observation(*args, **kwargs):
        raise AssertionError("captured applicability must not add live source scans")
    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", forbidden_observation)
    materials = build_intent_planning_materials(c.contract, manifest=c.manifest, source_applicability_nomination=c.nomination)
    assert len(materials.current_facts) == 1
    fact = materials.current_facts[0]
    assert fact.predicate.predicate_type == owner.PREDICATE
    assert fact.predicate.predicate_id in materials.producers[0].required_predicate_ids
    assert fact.predicate not in materials.intent.desired_predicates
    assert all(p.predicate_type == "administrative_requirement_task_coverage"
               for p in materials.intent.desired_predicates)
    first = planning.build_intent_symbolic_plan(c.contract, manifest=c.manifest, source_applicability_nomination=c.nomination)
    second = planning.build_intent_symbolic_plan(c.contract, manifest=json.loads(json.dumps(c.manifest)), source_applicability_nomination=c.nomination)
    assert second["receipt"] == first["receipt"]
    evidence = first["receipt"]["source_applicability"]
    assert evidence["solver_calls"] == 6
    assert evidence["native_source_observer_calls"] == evidence["provider_calls"] == 0
    assert all(evidence[k] is False for k in owner.FALSE)
    assert {row["solver_answer"] for row in evidence["checked_obligations"]} == {"sat", "unsat"}
    admission = local.admit_local_benchmark_plan(graph=first["graph"], manifest=c.manifest,
        requirement_bindings=first["requirement_bindings"], source_applicability_nomination=c.nomination)
    assert admission["receipt"]["payload"]["intent_symbolic_planning"] == first["receipt"]
    local.verify_local_benchmark_admission(admission)
    with IntentRepository(c.root / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(materialized["task_cids"][0])
        pending, _, _, _ = local._contract(task["body"], task["task_cid"])
        assert pending["pending_requirements"]
        assert all(row["phase"] == "post_execution" for row in pending["pending_requirements"])
        ref = intent.get_plan(materialized["plan_id"])["body"]["local_planning_receipt_ref"]
        assert local.load_local_planning_receipt(ref, manifest=c.manifest) == admission["receipt"]


@pytest.mark.parametrize("field", ["source_sha256", "solver_sha256", "operation_id", "operator_id", "proof_authority", "extra"])
def test_independent_binding_changes_cannot_nominate(case, field):
    bad = deepcopy(case.contract)
    value = bad["source_applicability"]
    value[field] = True if field == "proof_authority" else "0" * 64 if field.endswith("sha256") else "wrong"
    with pytest.raises(ValueError):
        manifest = case.author(bad)
        planning.build_intent_symbolic_plan(bad, manifest=manifest, source_applicability_nomination=case.nomination)


def test_owner_resigned_checked_marker_is_not_accepted(case):
    planned = planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=case.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=case.nomination)
    forged = deepcopy(admission)
    payload = forged["receipt"]["payload"]
    payload["intent_symbolic_planning"]["source_applicability"]["solver_calls"] = 0
    forged["receipt"] = local._signed(payload, case.manifest["payload"])
    with pytest.raises(local.LocalPlanningError, match="recomputed"):
        local.verify_local_benchmark_admission(forged)


def test_historical_checker_uses_captured_bytes_but_current_admission_still_rejects_drift(case):
    first = owner.checked_applicability(case.selection, manifest=case.manifest, nomination=case.nomination)
    planned = planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)
    (case.repo / "headers.py").write_text(PROGRAM + "\n# changed live source\n")
    assert owner.checked_applicability(case.selection, manifest=case.manifest, nomination=case.nomination) == first
    with pytest.raises(local.LocalPlanningError):
        local.admit_local_benchmark_plan(graph=planned["graph"], manifest=case.manifest,
            requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=case.nomination)


def test_missing_solver_blocks_without_fallback(case, monkeypatch):
    monkeypatch.setattr(owner.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="unavailable"):
        planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)


@pytest.mark.parametrize("case", ["guarded", "unsupported"], indirect=True)
def test_source_without_applicable_counterexample_cannot_select_repair(case):
    with pytest.raises(ValueError, match="model|candidate"):
        planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)


@pytest.mark.parametrize("field", ["protocol", "source_head", "producer", "artifact_cid"])
def test_owner_review_cannot_override_captured_native_identity(case, field):
    bad = deepcopy(case.nomination)
    receipt = bad["captured_receipt"]
    if field == "protocol":
        receipt[field]["review_ref"] = "different-review"
    elif field == "source_head":
        receipt[field]["generation"] += 1
    elif field == "producer":
        receipt[field]["owner"] = "0" * 64
    else:
        receipt[field] = cid_for_dag_json({"foreign": "artifact"})
    with pytest.raises((ValueError, KeyError, OSError)):
        planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=bad)


def test_injected_unknown_native_solver_answer_cannot_become_a_fact(case, monkeypatch):
    from ipfs_datasets_py.logic.security_ir import bounded_header_checker
    from ipfs_datasets_py.logic.backends.smt.differential import SmtRawSolverOutput
    # Error-path control only: the accepted end-to-end case above uses actual Z3.
    monkeypatch.setattr(bounded_header_checker, "bounded_header_runner", lambda *a, **k:
        lambda script, bounds: SmtRawSolverOutput(stdout="unknown\n", returncode=0,
            solver_version="injected-unknown-control"))
    with pytest.raises(local.LocalPlanningError, match="actual complete") as caught:
        local._replay_intent_symbolic_plan(case.contract, case.manifest,
            source_applicability_nomination=case.nomination)
    diagnostic = owner.project_header_checker_failure(caught.value)
    assert diagnostic["reason_codes"] == ["check_status"]
    assert diagnostic["status"] == "model_check_inconclusive_or_mismatch"
    assert diagnostic["result_status_counts"] == {"unknown": 6}
    assert diagnostic["expected_solver_calls"] == diagnostic["observed_solver_calls"] == 6
    assert diagnostic["execution_profile_matches"] and diagnostic["solver_identity_matches"]
    assert "injected-unknown-control" not in json.dumps(diagnostic)


@pytest.mark.parametrize("change", ["execution_profile", "solver_identity", "solver_call_count",
                                    "missing_counterexample", "query_expectation"])
def test_actual_checker_refusal_retains_exact_failed_gate_without_raw_outputs(case, monkeypatch, change):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    actual = header.check_header_semantics
    def changed(*args, **kwargs):
        result = actual(*args, **kwargs)
        assert result["status"] == "checked_local_model"
        if change == "execution_profile":
            result["execution_profile"] = "PRIVATE_PROFILE"
        elif change == "solver_identity":
            result["solver_executable_sha256"] = "PRIVATE_IDENTITY"
        elif change == "solver_call_count":
            result["solver_calls"] -= 1
        elif change == "missing_counterexample":
            for item in result["results"]:
                if item["kind"] == "unsafe_converted_input_accepted":
                    item["solver_answer"] = "unknown"
        else:
            result["results"][0]["matches_model_expectation"] = False
        for item in result["results"]:
            item["model_text"] = "PRIVATE_MODEL_BODY"
        return result
    monkeypatch.setattr(header, "check_header_semantics", changed)
    with pytest.raises(local.LocalPlanningError) as caught:
        local._replay_intent_symbolic_plan(case.contract, case.manifest,
            source_applicability_nomination=case.nomination)
    diagnostic = owner.project_header_checker_failure(caught.value)
    assert diagnostic["reason_codes"] == [change]
    assert diagnostic["result_count"] == 6
    assert "PRIVATE" not in json.dumps(diagnostic)


def test_checker_diagnostic_projection_caps_rows_and_detaches_metadata():
    checked = {"status": "model_check_inconclusive_or_mismatch", "solver_calls": 500,
        "results": [{"status": "unknown", "model_text": "PRIVATE"}] * 500}
    error = owner._checker_refusal(checked, expected_calls=500, solver_sha="expected",
        reasons=["check_status"], message="refused")
    result = owner.project_header_checker_failure(error)
    assert result["result_rows_truncated"] and result["result_count"] == 500
    assert result["result_status_counts"] == {"unknown": 64}
    result["reason_codes"].append("solver_call_count")
    assert error.header_checker_diagnostic["reason_codes"] == ["check_status"]
    assert "PRIVATE" not in json.dumps(result)


def test_expired_or_cancelled_checker_cannot_return_applicability(case):
    import threading
    event = threading.Event(); event.set()
    with pytest.raises(Exception, match="cancel"):
        owner.checked_applicability(case.selection, manifest=case.manifest, nomination=case.nomination,
            scheduler=case.scheduler, cancel_event=event)
    with pytest.raises(Exception, match="deadline|timeout|Timeout"):
        owner.checked_applicability(case.selection, manifest=case.manifest, nomination=case.nomination,
            scheduler=case.scheduler, timeout_seconds=.000001)


def test_materialization_inherits_expired_planner_budget_before_durable_receipt(case, monkeypatch):
    planned = planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=case.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=case.nomination)
    clock = [1000.]
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock[0])
    with IntentRepository(case.root / "intent-timeout.duckdb") as intent:
        with owner.applicability_budget(1.):
            clock[0] += 2.
            with pytest.raises(TimeoutError, match="aggregate"):
                local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        assert intent.get_task(planned["graph"].tasks[0].task_cid) is None
    assert not (case.root / "lifecycle/local-planning-receipts").exists()


def test_nested_budget_cannot_renew_and_restores_previous_scope(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock[0])
    with owner.applicability_budget(1.):
        clock[0] += 2.
        with owner.applicability_budget(45.):
            with pytest.raises(TimeoutError, match="aggregate"):
                owner.require_applicability_budget()
    owner.require_applicability_budget()



def test_cold_stored_and_pending_receipts_cannot_promote_security_goal(case):
    planned = planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest, source_applicability_nomination=case.nomination)
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=case.manifest,
        requirement_bindings=planned["requirement_bindings"], source_applicability_nomination=case.nomination)
    forged = deepcopy(admission["receipt"])
    forged["payload"]["intent_symbolic_planning"]["source_applicability"]["security_goal_satisfied"] = True
    forged = local._signed(forged["payload"], case.manifest["payload"])
    with pytest.raises(local.LocalPlanningError, match="symbolic selection"):
        local._store_local_planning_receipt(forged, manifest=case.manifest)
    with IntentRepository(case.root / "pending.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(materialized["task_cids"][0])
        body = deepcopy(task["body"])
        payload = body[local.CONTRACT_KEY]["payload"]
        payload["intent_plan"]["symbolic_planning"]["source_applicability"]["security_goal_satisfied"] = True
        body[local.CONTRACT_KEY] = local._signed(payload, case.manifest["payload"])
        with pytest.raises(local.LocalPlanningError, match="recomputed"):
            local._contract(body, task["task_cid"])


@pytest.mark.parametrize("gate", ["admit", "verify", "stored", "pending"])
def test_source_mutation_during_actual_solver_is_rejected_by_enclosing_gate(case, monkeypatch, gate):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    plan = planning.build_intent_symbolic_plan(case.contract, manifest=case.manifest,
        source_applicability_nomination=case.nomination)
    admission = local.admit_local_benchmark_plan(graph=plan["graph"], manifest=case.manifest,
        requirement_bindings=plan["requirement_bindings"], source_applicability_nomination=case.nomination)
    reference = local._store_local_planning_receipt(admission["receipt"], manifest=case.manifest)
    with IntentRepository(case.root / "mutation-intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(materialized["task_cids"][0])
    check = header.check_header_semantics
    calls = []
    def mutate(*args, **kwargs):
        result = check(*args, **kwargs)
        if not calls:
            path = case.repo / "headers.py"
            path.write_bytes(path.read_bytes() + b"\n# changed while external checker ran\n")
        calls.append(result["solver_calls"])
        return result
    monkeypatch.setattr(header, "check_header_semantics", mutate)
    with pytest.raises(ValueError):
        if gate == "admit":
            local.admit_local_benchmark_plan(graph=plan["graph"], manifest=case.manifest,
                requirement_bindings=plan["requirement_bindings"], source_applicability_nomination=case.nomination)
        elif gate == "verify":
            local.verify_local_benchmark_admission(admission, initial=True)
        elif gate == "stored":
            local.load_local_planning_receipt(reference, manifest=case.manifest)
        else:
            local._contract(task["body"], task["task_cid"])
    assert calls and calls[0] > 0


def test_normal_initialization_with_real_checkpoint_produces_nomination_before_planning(case, tmp_path, monkeypatch):
    """Real checkpoint/capture/solver; authored source and reviewed intent mapping."""
    import os
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
    from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_nomination
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    checkpoint, snapshot = os.environ.get("CODEBASE384_CHECKPOINT"), os.environ.get("CODEBASE384_EMBEDDING_SNAPSHOT")
    if not checkpoint or not snapshot:
        pytest.skip("explicit pinned checkpoint and cached GTE snapshot required")
    repo = tmp_path / "normal-app"; repo.mkdir()
    (repo / "bottle.py").write_text(PROGRAM)
    for args in (("init", "-q"), ("add", "."), ("-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "authored initial input")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    instruction = tmp_path / "normal-instruction.md"
    instruction.write_text("Reject control characters in HTTP headers and write report.jsonl with file_path and cwe_id fields.")
    contract = _requirements(instruction, symbolic=True)
    selection = dict(schema=owner.SCHEMA, review_ref="review:authored-normal-flow@1",
        operation_id=contract["symbolic_operations"]["operations"][0]["operation_id"],
        operator_id=owner.OPERATOR, source_path="bottle.py",
        protocol={"review_ref": "review:authored-header-protocol@1", "callback_parameter": "emit"},
        checker_profile=owner.CHECKER_PROFILE, **owner.FALSE)
    contract.update(schema=owner.CONTRACT_SCHEMA, source_applicability=selection)
    requirements = tmp_path / "normal-requirements.json"; requirements.write_bytes(wire(contract))
    state = tmp_path / "normal-state"
    prepared = prep.prepare(repository=repo, instruction=instruction, state=state,
        intent_requirement_contract=requirements)
    signed_before = wire(prepared["manifest"])
    assert not (state / "source384-context").exists()
    config = runtime_config(tmp_path, selection)
    copied = tmp_path / "normal-checkpoint.json"; shutil.copyfile(checkpoint, copied)
    config.update(checkpoint_path=str(copied), checkpoint_sha256=sha(copied.read_bytes()),
        embedding_snapshot=snapshot, embedding_assets=embedding._snapshot_assets(snapshot)[1])
    config_path = tmp_path / "normal-config.json"; config_path.write_bytes(wire(config))
    initialized = prep.initial_context(state=state, source384_config=config_path)
    receipt = initialized["source384_context"]
    nomination = receipt["source_applicability_nomination"]
    assert receipt["schema"] == "terminal-source384-repository-context@2"
    assert nomination["manifest_cid"] == content_identity(prepared["manifest"])
    assert nomination["captured_receipt"]["source_head"] == receipt["source_head"]
    assert wire(json.loads((state / "prepared.json").read_bytes())["manifest"]) == signed_before
    inference = json.loads((state / "source384-context/inference.json").read_bytes())
    assert inference["native_worker_executed"] and inference["report"]["output"]["model_loads"] == 1
    monkeypatch.setattr(units, "_worker", lambda *a, **k: pytest.fail("checkpoint replayed after initialization"))
    planned = prep.plan(state, timeout_seconds=90)
    assert planned["qualified"], planned
    assert planned["provider_calls"] == 0 and planned["goals"] == 2 and planned["tasks"] == 1
    assert planned["symbolic_planning"]["source_applicability_nomination"] == nomination
    context = prep.context(state=state)
    bundle = context["context_bundle"]
    payload = json.loads((repo / bundle["artifact"]).read_bytes()); task = payload["tasks"][0]
    assert load_task_context_nomination(repository=repo, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_id=task["task_id"], task_cid=task["task_cid"])
    assert not (repo / "report.jsonl").exists()
    assert (repo / "bottle.py").read_text() == PROGRAM
