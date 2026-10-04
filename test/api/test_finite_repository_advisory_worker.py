"""Structural controls for the additive advisory/native-worker experiment.

Upstream admission and proposal verifiers are replaced only in the bridge unit
fixture. These cases test the additional joins and do not qualify a signer,
trainer, prover, native owner, process origin, or worker execution boundary.
"""
from copy import deepcopy
from collections.abc import Mapping
import ast
import base64
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import finite_repository_advisory_worker_support as support
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import (
    OFFSET_STATEMENT_ID, TYPE_STATEMENT_ID,
)


def _context_identity(binding):
    binding["context_cid"] = cid_for_structured({key: value for key, value in binding.items()
                                              if key != "context_cid"})


def _bridge_identity(bridge):
    bridge["bridge_cid"] = cid_for_structured({key: value for key, value in bridge.items()
                                             if key != "bridge_cid"})


@pytest.fixture
def bridge_case(tmp_path, monkeypatch):
    """Authored upstream records; only the new structural join is exercised."""
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as finite
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate as proposal
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as worker

    calls = []

    def verify_admission(*, admission):
        calls.append("admission")
        return {"semantic_context": deepcopy(admission["semantic_context"])}

    def verify_review(*, admission, candidate):
        calls.append("reviewed")
        return deepcopy(candidate)

    def verify_generated(*, record):
        calls.append("generated")

    def verify_worker(candidate):
        calls.append("worker")

    monkeypatch.setattr(finite, "verify_finite_repository_admission", verify_admission)
    monkeypatch.setattr(proposal, "verify_finite_repository_candidate", verify_review)
    monkeypatch.setattr(proposal, "verify_generated_finite_repository_candidate", verify_generated)
    monkeypatch.setattr(worker, "_validate", verify_worker)

    before = b"def increment(n: int) -> int:\n    return n + 1\n"
    after = b"def increment(n: int) -> int:\n    return n + 2\n"
    replacement = tmp_path / "generated.py"
    replacement.write_bytes(after)
    head = {"schema": "codebase-head@1", "repository_id": "repository:authored-bridge-test",
            "generation": 1, "snapshot_cid": cid_for_structured({"snapshot": 1})}
    offset = cid_for_structured({"task": "FINITE-OFFSET"})
    kind = cid_for_structured({"task": "FINITE-TYPE"})
    population = sorted((offset, kind))
    semantic = {"head": head, "source_cid": cid_for_bytes(before),
                "administrator_task_cids": population,
                "native_task_bindings": {
                    TYPE_STATEMENT_ID: {"task_key": "FINITE-TYPE", "task_cid": kind},
                    OFFSET_STATEMENT_ID: {"task_key": "FINITE-OFFSET", "task_cid": offset}}}
    admission = {"semantic_context": semantic}
    reviewed = {"payload": {"task_cid": offset, "administrator_task_cids": population,
                            "after_cid": cid_for_bytes(after),
                            "after_sha256": hashlib.sha256(after).hexdigest()}}
    generated = {"parent_admission": deepcopy(admission), "reviewed_candidate": deepcopy(reviewed),
                 "head": deepcopy(head), "task_cid": offset,
                 "replacement_cid": cid_for_bytes(after),
                 "artifacts": {"replacement": {"path": str(replacement)}},
                 "result_cid": cid_for_structured({"synthetic_generated": 1}),
                 "training_steps": 0, "provider_calls": 0}
    worker_candidate = {"finite_admission": deepcopy(admission), "task_cid": offset,
                        "task_revision": 1, "semantic_context_cid": cid_for_structured(semantic),
                        "edit": {"after_bytes_base64": base64.b64encode(after).decode("ascii"),
                                 "after_sha256": hashlib.sha256(after).hexdigest()},
                        "candidate_cid": cid_for_structured({"synthetic_worker": 1}),
                        "training_steps": 0, "provider_calls": 0}
    model = {"schema": "supervisor-codebase-feature-context@1", "mode": "train",
             "model_enabled": True, "version_id": "sha256:authored-root-version",
             "latent_width": 8, "parameter_dtype": "float64", "actual_training_delta": 16,
             "authority": {"proof_authority": False, "execution_authority": False},
             "head": deepcopy(head)}
    _context_identity(model)
    arguments = {"admission": admission, "reviewed_candidate": reviewed,
                 "generated_candidate": generated, "worker_candidate": worker_candidate,
                 "replacement_bytes": after, "expected_model_binding": model,
                 "expected_task_cid": offset}
    return {"arguments": arguments, "calls": calls, "replacement": replacement,
            "upstream_modules": (finite, proposal, worker)}


def test_authored_inventory_keeps_training_controls_and_original_public_checks(tmp_path):
    repository = tmp_path / "repository"
    inventory = support.create_advisory_worker_sources(repository)
    expected = {"calc.py", "known_variant.py", "tune.py", "canary.py", "decoy.py",
                "support.py", "consumer.py", "unsupported.py", "check_type.py", "check_offset.py"}
    assert set(inventory) == expected
    assert {path.name for path in repository.glob("*.py")} == expected
    calc = ast.parse((repository / "calc.py").read_bytes()).body[0].body[0].value
    known = ast.parse((repository / "known_variant.py").read_bytes()).body[0].body[0].value
    assert isinstance(calc, ast.BinOp) and isinstance(calc.op, ast.Add) and calc.right.value == 1
    assert isinstance(known, ast.BinOp) and isinstance(known.op, ast.Add) and known.right.value == 2
    assert b"getattr" in (repository / "unsupported.py").read_bytes()
    assert b"increment" in (repository / "check_type.py").read_bytes()
    assert b"increment" in (repository / "check_offset.py").read_bytes()


def test_bridge_preserves_complete_context_without_creating_native_authority(bridge_case):
    arguments = bridge_case["arguments"]
    before = canonical_dag_json_bytes({key: value for key, value in arguments.items()
                                      if key != "replacement_bytes"})
    bridge = support.build_advisory_worker_bridge(**arguments)
    verified = support.validate_advisory_worker_bridge(bridge, **arguments)
    assert verified == bridge
    assert bridge_case["calls"] == ["admission", "reviewed", "generated", "worker"] * 2
    assert set(bridge) == support.BRIDGE_FIELDS
    assert bridge["scope"] == "advisory_generated_bytes_to_separately_admitted_native_worker_candidate"
    assert bridge["task_cid"] == arguments["expected_task_cid"]
    assert bridge["administrator_task_cids"] == arguments["admission"]["semantic_context"]["administrator_task_cids"]
    assert len(bridge["administrator_task_cids"]) == 2
    assert bridge["replacement_cid"] == cid_for_bytes(arguments["replacement_bytes"])
    assert bridge["model_version_id"] == arguments["expected_model_binding"]["version_id"]
    assert bridge["model_context_cid"] == arguments["expected_model_binding"]["context_cid"]
    assert all(bridge[key] is False for key in support.CLAIMS)
    assert bridge["training_steps"] == bridge["provider_calls"] == 0
    assert canonical_dag_json_bytes({key: value for key, value in arguments.items()
                                     if key != "replacement_bytes"}) == before


@pytest.mark.parametrize("field", sorted(support.CLAIMS))
@pytest.mark.parametrize("value", [True, 0])
def test_rehashed_bridge_cannot_gain_or_alias_authority(bridge_case, field, value):
    arguments = bridge_case["arguments"]
    bridge = support.build_advisory_worker_bridge(**arguments)
    bridge[field] = value
    _bridge_identity(bridge)
    with pytest.raises(ValueError, match="nonauthoritative"):
        support.validate_advisory_worker_bridge(bridge, **arguments)


@pytest.mark.parametrize("field", ["admission_cid", "reviewed_candidate_cid", "generated_result_cid",
    "worker_candidate_cid", "source_cid", "replacement_cid", "replacement_sha256", "task_cid",
    "task_revision", "head", "administrator_task_cids", "model_context_cid", "model_version_id",
    "model_binding", "training_steps", "provider_calls"])
def test_rehashed_bridge_cannot_rebind_native_identity_or_verified_model(bridge_case, field):
    arguments = bridge_case["arguments"]
    bridge = support.build_advisory_worker_bridge(**arguments)
    if field == "head":
        bridge[field]["generation"] += 1
    elif field == "administrator_task_cids":
        bridge[field] = bridge[field][:1]
    elif field == "model_binding":
        bridge[field]["version_id"] = "sha256:foreign-model"
        _context_identity(bridge[field])
    elif field == "task_revision":
        bridge[field] = True
    elif field in {"training_steps", "provider_calls"}:
        bridge[field] = False
    else:
        bridge[field] = "foreign-identity"
    _bridge_identity(bridge)
    with pytest.raises(ValueError):
        support.validate_advisory_worker_bridge(bridge, **arguments)


@pytest.mark.parametrize("field", ["schema", "scope", "unknown-field", "missing-field", "digest"])
def test_bridge_schema_and_material_digest_remain_closed(bridge_case, field):
    arguments = bridge_case["arguments"]
    bridge = support.build_advisory_worker_bridge(**arguments)
    if field == "unknown-field":
        bridge["claim_id"] = "invented-native-claim"
    elif field == "missing-field":
        del bridge["model_binding"]
    elif field == "digest":
        bridge["replacement_sha256"] = "0" * 64
    else:
        bridge[field] = "foreign-schema-or-scope"
    if field != "digest":
        _bridge_identity(bridge)
    with pytest.raises(ValueError):
        support.validate_advisory_worker_bridge(bridge, **arguments)


@pytest.mark.parametrize("control", ["review-task", "generated-task", "worker-task", "expected-task",
    "generated-head", "model-head", "generated-admission", "worker-admission", "generated-review",
    "population-omission", "raw-bytes", "retained-bytes", "worker-bytes", "replacement-cid",
    "worker-sha", "review-sha", "semantic-context", "revision-bool", "revision-zero",
    "generated-training", "generated-provider", "worker-training", "worker-provider"])
def test_proposal_conversion_rejects_cross_source_task_population_and_byte_mismatch(bridge_case, control):
    arguments = deepcopy(bridge_case["arguments"])
    review = arguments["reviewed_candidate"]["payload"]
    generated = arguments["generated_candidate"]
    worker = arguments["worker_candidate"]
    model = arguments["expected_model_binding"]
    if control == "review-task":
        review["task_cid"] = "foreign-task"
    elif control == "generated-task":
        generated["task_cid"] = "foreign-task"
    elif control == "worker-task":
        worker["task_cid"] = "foreign-task"
    elif control == "expected-task":
        arguments["expected_task_cid"] = "foreign-task"
    elif control == "generated-head":
        generated["head"]["generation"] += 1
    elif control == "model-head":
        model["head"]["generation"] += 1
        _context_identity(model)
    elif control == "generated-admission":
        generated["parent_admission"]["semantic_context"]["source_cid"] = "foreign-source"
    elif control == "worker-admission":
        worker["finite_admission"]["semantic_context"]["source_cid"] = "foreign-source"
    elif control == "generated-review":
        generated["reviewed_candidate"]["payload"]["after_cid"] = "foreign-source"
    elif control == "population-omission":
        review["administrator_task_cids"] = review["administrator_task_cids"][:1]
        generated["reviewed_candidate"] = deepcopy(arguments["reviewed_candidate"])
    elif control == "raw-bytes":
        arguments["replacement_bytes"] += b"# caller substituted bytes\n"
    elif control == "retained-bytes":
        bridge_case["replacement"].write_bytes(arguments["replacement_bytes"] + b"# retained drift\n")
    elif control == "worker-bytes":
        worker["edit"]["after_bytes_base64"] = base64.b64encode(b"foreign bytes").decode("ascii")
    elif control == "replacement-cid":
        generated["replacement_cid"] = "foreign-source"
    elif control == "worker-sha":
        worker["edit"]["after_sha256"] = "0" * 64
    elif control == "review-sha":
        review["after_sha256"] = "0" * 64
        generated["reviewed_candidate"] = deepcopy(arguments["reviewed_candidate"])
    elif control == "semantic-context":
        worker["semantic_context_cid"] = "foreign-context"
    elif control in {"revision-bool", "revision-zero"}:
        worker["task_revision"] = True if control == "revision-bool" else 0
    else:
        target = generated if control.startswith("generated-") else worker
        target["training_steps" if control.endswith("training") else "provider_calls"] = True
    with pytest.raises(ValueError):
        support.build_advisory_worker_bridge(**arguments)


@pytest.mark.parametrize("field,value", [("mode", "frozen"), ("model_enabled", 1),
    ("version_id", ""), ("latent_width", True), ("latent_width", 384),
    ("parameter_dtype", "float32"), ("actual_training_delta", True),
    ("actual_training_delta", 0), ("authority", {}), ("authority", {"proof_authority": 0}),
    ("authority", {"execution_authority": True}), ("schema", "foreign-model-context")])
def test_conversion_requires_exact_initial_verified_advisory_model_profile(bridge_case, field, value):
    arguments = deepcopy(bridge_case["arguments"])
    arguments["expected_model_binding"][field] = value
    _context_identity(arguments["expected_model_binding"])
    with pytest.raises(ValueError, match="model binding"):
        support.build_advisory_worker_bridge(**arguments)


@pytest.mark.parametrize("upstream", ["admission", "reviewed", "generated", "worker"])
def test_bridge_never_bypasses_an_upstream_verifier_refusal(bridge_case, monkeypatch, upstream):
    finite, proposal, worker = bridge_case["upstream_modules"]
    selected = {"admission": (finite, "verify_finite_repository_admission"),
                "reviewed": (proposal, "verify_finite_repository_candidate"),
                "generated": (proposal, "verify_generated_finite_repository_candidate"),
                "worker": (worker, "_validate")}[upstream]

    def refused(*args, **kwargs):
        raise ValueError("independent upstream refusal")

    monkeypatch.setattr(*selected, refused)
    with pytest.raises(ValueError, match="independent upstream refusal"):
        support.build_advisory_worker_bridge(**bridge_case["arguments"])


def _finite_preview(offset):
    """Authored projection input; this fixture performs no observation or fit."""
    satisfied = offset == 2
    domain = [-2, -1, 0, 1, 2]
    clauses = [{"statement_id": TYPE_STATEMENT_ID, "status": "bounded_observed_satisfied",
                "scope": "explicit_finite_domain_only", "reasons": []},
               {"statement_id": OFFSET_STATEMENT_ID,
                "status": "bounded_observed_satisfied" if satisfied else "requires_work",
                "scope": "explicit_finite_domain_only", "reasons": [] if satisfied else ["offset differs"]}]
    return {"match": {"source_cid": cid_for_bytes(f"offset={offset}".encode()),
                       "query": {"requirement_ids": [TYPE_STATEMENT_ID, OFFSET_STATEMENT_ID]},
                       "observation": {"domain_inputs": domain,
                           "observations": [{"input": value, "output": value + offset} for value in domain]},
                       "eligible_clause_ids": [TYPE_STATEMENT_ID, OFFSET_STATEMENT_ID] if satisfied else [TYPE_STATEMENT_ID],
                       "residual_clause_ids": [] if satisfied else [OFFSET_STATEMENT_ID],
                       "clause_results": clauses},
            "current_facts_count": 2 if satisfied else 1,
            "selected_task_ids": [] if satisfied else ["task:finite:offset"],
            "planning_model_calls": 0, "training_steps_during_preview": 0,
            **{key: False for key in ("source_semantics_verified", "proof_authority", "execution_authority",
                "completion_authority", "production_admitted", "worker_launched", "convergence_proved")}}


@pytest.mark.parametrize("offset", [1, 2])
def test_off_train_frozen_projection_keeps_both_finite_clauses_and_original_task_choice(offset):
    previews = [deepcopy(_finite_preview(offset)) for _ in range(3)]
    original = canonical_dag_json_bytes(previews)
    for mode, preview in zip(("model_off", "train", "frozen"), previews):
        preview["feature_context"] = {"mode": mode, "context_cid": "context:" + mode}
        preview["result_cid"] = "fresh-observation:" + mode
    projections = [support.finite_outcome_projection(preview) for preview in previews]
    assert projections[0] == projections[1] == projections[2]
    assert len(projections[0]["clause_results"]) == 2
    assert {row["statement_id"] for row in projections[0]["clause_results"]} == {TYPE_STATEMENT_ID, OFFSET_STATEMENT_ID}
    assert projections[0]["selected_task_ids"] == ([] if offset == 2 else ["task:finite:offset"])
    assert projections[0]["observations"] == _finite_preview(offset)["match"]["observation"]["observations"]
    for preview in previews:
        del preview["feature_context"]
        del preview["result_cid"]
    assert canonical_dag_json_bytes(previews) == original


@pytest.mark.parametrize("field", ["source_cid", "query", "domain_inputs", "observations",
    "eligible_clause_ids", "residual_clause_ids", "clause_results", "current_facts_count", "selected_task_ids"])
def test_projection_does_not_hide_changes_to_finite_requirements_or_observations(field):
    before = _finite_preview(1)
    after = deepcopy(before)
    if field in {"domain_inputs", "observations"}:
        after["match"]["observation"][field] = []
    elif field == "clause_results":
        after["match"][field][1]["status"] = "bounded_observed_satisfied"
    elif field in {"source_cid", "query", "eligible_clause_ids", "residual_clause_ids"}:
        after["match"][field] = "foreign" if field == "source_cid" else {} if field == "query" else []
    elif field == "current_facts_count":
        after[field] += 1
    else:
        after[field] = []
    assert support.finite_outcome_projection(after) != support.finite_outcome_projection(before)


@pytest.mark.parametrize("field,value", [("planning_model_calls", 1), ("planning_model_calls", False),
    ("training_steps_during_preview", 1), ("training_steps_during_preview", False),
    ("current_facts_count", True), ("selected_task_ids", ("task:finite:offset",)),
    ("source_semantics_verified", True), ("proof_authority", True), ("execution_authority", 0),
    ("completion_authority", True), ("production_admitted", True), ("worker_launched", True),
    ("convergence_proved", True)])
def test_projection_refuses_fitting_planner_calls_authority_and_type_aliases(field, value):
    preview = _finite_preview(1)
    preview[field] = value
    with pytest.raises(ValueError):
        support.finite_outcome_projection(preview)


@pytest.mark.parametrize("population", [[], [{"statement_id": TYPE_STATEMENT_ID}],
    [{"statement_id": TYPE_STATEMENT_ID}] * 3])
def test_projection_refuses_incomplete_original_clause_population(population):
    preview = _finite_preview(1)
    preview["match"]["clause_results"] = population
    with pytest.raises(ValueError):
        support.finite_outcome_projection(preview)


def test_metadata_keeps_every_family_full_bytes_and_repeated_occurrences():
    families = ("sources", "ast", "symbols", "kg", "vectors", "contracts", "compiled_logic",
                "current_facts", "native_publications", "signed_admissions", "native_tasks", "artifacts",
                "feature_contexts", "preview_invariants", "training", "lowering_proofs", "reviewed_candidates",
                "proposal_worker_bridges", "native_capacity", "processes", "worker_lifecycle", "controls")
    raw = b"\x00\xfffull source and checkpoint bytes\n"
    record = {"producer": {"source_bytes_base64": base64.b64encode(raw).decode("ascii"),
                            "disposition": "unsupported_unproved", "proven": False},
              "original_task_cids": ["type", "offset"]}
    values = {family: [deepcopy(record), deepcopy(record)] for family in families}
    records = {}
    support.append_metadata_occurrences(records, values)
    support.append_metadata_occurrences(records, values)
    assert set(records) == set(families)
    for family in families:
        assert [row["occurrence"] for row in records[family]] == [0, 1, 2, 3]
        assert all(set(row) == {"schema", "occurrence", "record"} for row in records[family])
        assert all(row["schema"] == support.OCCURRENCE_SCHEMA for row in records[family])
        assert all(row["record"] == record for row in records[family])
        assert all(base64.b64decode(row["record"]["producer"]["source_bytes_base64"]) == raw for row in records[family])
    values["ast"][0]["producer"]["proven"] = True
    records["ast"][0]["record"]["original_task_cids"].pop()
    assert records["ast"][1]["record"] == record
    assert all(row["record"] == record for row in records["native_tasks"])


@pytest.mark.parametrize("records,values", [([], {}), ({}, []), ({}, {1: []}),
    ({}, {"ast": ()}), ({"ast": {}}, {"ast": []})])
def test_metadata_occurrence_contract_rejects_family_shape_changes(records, values):
    with pytest.raises(ValueError):
        support.append_metadata_occurrences(records, values)


@pytest.mark.parametrize("empty", [False, True])
def test_native_evidence_snapshot_uses_real_cursor_named_rows_without_description(empty):
    from benchmarks.agent_supervisor.container_coding.finite_repository_advisory_worker_experiment import (
        _native_evidence_snapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBCursor, DuckDBRow

    queries, expected, actual_cursors = [], {}, []
    names = ("task_cid", "body_json", "revision", "validation_run_id")

    class UnderlyingDuckDB:
        description = [(name,) for name in names]

        def __init__(self, table):
            self.rows = [] if empty else [
                ("task:offset", json.dumps({"table": table, "claim": "actual-authored-offset"}), 4, "run:offset"),
                ("task:type", json.dumps({"table": table, "claim": "actual-authored-type"}), 4, None)]
            expected[table] = sorted([dict(zip(names, values)) for values in self.rows],
                                     key=lambda row: json.dumps(row, sort_keys=True))

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, query):
            queries.append(query)
            cursor = DuckDBCursor(UnderlyingDuckDB(query.split('"')[1]))
            assert not hasattr(cursor, "description")
            actual_cursors.append(cursor)
            return cursor

    row = DuckDBRow(names, ("actual-value", "actual-body", 4, None))
    assert isinstance(row, Mapping)
    assert list(row) == list(names) and row[0] == "actual-value"
    snapshot = _native_evidence_snapshot(SimpleNamespace(_connection=Connection(), _lock=threading.RLock()))
    assert snapshot["tables"] == expected
    assert set(expected) == {"objectives", "goals", "plans", "tasks", "task_dependencies", "task_outputs",
                             "task_validations", "task_acceptance", "task_attempts", "task_claims", "leases",
                             "fencing_epochs", "validation_runs", "validation_results", "completion_receipts", "merge_attempts"}
    assert len(queries) == len(expected) == 16
    assert len(actual_cursors) == 16 and all(not hasattr(cursor, "description") for cursor in actual_cursors)
    assert all(not cursor.fetchall() for cursor in actual_cursors)
    assert all(len(rows) == (0 if empty else 2) for rows in snapshot["tables"].values())
    assert all(row["revision"] == 4 and row["task_cid"].startswith("task:")
               for rows in snapshot["tables"].values() for row in rows)
    assert all(json.loads(row["body_json"])["table"] == table
               for table, rows in snapshot["tables"].items() for row in rows)


def test_native_evidence_snapshot_refuses_non_mapping_rows(monkeypatch):
    from benchmarks.agent_supervisor.container_coding.finite_repository_advisory_worker_experiment import (
        _native_evidence_snapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBCursor

    underlying = SimpleNamespace(description=[("task_cid",), ("body_json",)],
                                 fetchall=lambda: [("task:offset", "actual-body")])
    cursor = DuckDBCursor(underlying)
    assert not hasattr(cursor, "description")
    monkeypatch.setattr(cursor, "fetchall", lambda: [("task:offset", "actual-body")])
    connection = SimpleNamespace(execute=lambda query: cursor)
    with pytest.raises(ValueError, match="complete named row projection"):
        _native_evidence_snapshot(SimpleNamespace(_connection=connection, _lock=threading.RLock()))


def _checkpoint_accounting_case(tmp_path, reports):
    """Authored retained checkpoint summaries; no native training is invoked."""
    subject = object.__new__(support.AdvisoryWorkerSupport)
    subject.root = tmp_path
    subject.root_context = subject.child = None
    for number, epochs, selected_progress in reports:
        record = {"report": {"codebase_request_sha256": str(number) * 64,
                             "attempted_epochs": epochs,
                             "selected_epoch": selected_progress,
                             "codebase_worker_receipt": {"returncode": 0},
                             "codebase_provenance": {"head": {"generation": number},
                                                     "parent_version_id": None if number == 1 else "root-model"}}}
        raw = canonical_dag_json_bytes(record)
        root = tmp_path / "model-artifacts"
        root.mkdir(exist_ok=True)
        (root / hashlib.sha256(raw).hexdigest()).write_bytes(raw)
    return subject


def test_native_checkpoint_cost_survives_missing_finalized_context(tmp_path):
    subject = _checkpoint_accounting_case(tmp_path, [(1, 16, 5)])
    costs = subject._training_cost()
    assert costs["known_completed_context_epochs"] == 0
    assert costs["checkpoint_observed_epochs"] == costs["observed_epoch_lower_bound"] == 16
    assert subject.actual_training_epochs == 16
    assert len(costs["checkpoints"]) == 1 and costs["errors"] == []
    assert costs["unknown_unretained_training_attempts_excluded_from_exact_claim"] is True


def test_checkpoint_accounting_counts_distinct_root_child_attempts_not_selected_progress(tmp_path):
    subject = _checkpoint_accounting_case(tmp_path, [(1, 16, 5), (2, 16, 3)])
    subject.root_context = SimpleNamespace(actual_training_delta=16)
    costs = subject._training_cost()
    assert costs["known_completed_context_epochs"] == 16
    assert costs["checkpoint_observed_epochs"] == costs["observed_epoch_lower_bound"] == 32
    assert subject.actual_training_epochs == 32
    assert {row["head"]["generation"] for row in costs["checkpoints"]} == {1, 2}
    assert costs["errors"] == []


def test_failure_without_retained_checkpoint_has_zero_observed_lower_bound(tmp_path):
    subject = _checkpoint_accounting_case(tmp_path, [])
    costs = subject._training_cost()
    assert costs["known_completed_context_epochs"] == costs["checkpoint_observed_epochs"] == 0
    assert costs["observed_epoch_lower_bound"] == 0 and costs["checkpoints"] == []
    assert costs["unknown_unretained_training_attempts_excluded_from_exact_claim"] is True


@pytest.mark.parametrize("mismatch", [None, "task", "candidate"])
def test_worker_bridge_loads_full_public_handoff_from_the_descriptor(bridge_case, tmp_path, monkeypatch, mismatch):
    """Exercise the adapter seam with upstream/custody calls explicitly stubbed."""
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as worker

    arguments = bridge_case["arguments"]
    loaded = deepcopy(arguments["worker_candidate"])
    artifact = tmp_path / "public-handoff.json"
    artifact.write_bytes(canonical_dag_json_bytes(loaded))
    descriptor = {"artifact": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  "candidate_cid": loaded["candidate_cid"], "task_cid": loaded["task_cid"]}
    if mismatch == "task":
        loaded["task_cid"] = "foreign-task"
    elif mismatch == "candidate":
        loaded["candidate_cid"] = "foreign-candidate"
    loader_calls = []

    def load(*, artifact, expected_sha256):
        loader_calls.append((artifact, expected_sha256))
        return deepcopy(loaded)

    monkeypatch.setattr(worker, "load_finite_repository_candidate", load)
    monkeypatch.setattr(codebase_feature_context, "verify_current_context", lambda *args: None)
    monkeypatch.setattr(support, "_with_custody", lambda owner, operation: operation())
    monkeypatch.setattr(support, "_owned_file_pins", lambda *args: [])
    monkeypatch.setattr(support, "_pin", lambda path: {"path": str(path), "bytes": 0, "sha256": "authored-sidecar-pin"})
    subject = object.__new__(support.AdvisoryWorkerSupport)
    subject.output, subject.root = tmp_path, tmp_path / "advisory"
    subject.root.mkdir()
    subject.generated, subject.reviewed = arguments["generated_candidate"], arguments["reviewed_candidate"]
    subject.replacement, subject.bridge = arguments["replacement_bytes"], None
    subject.root_context = SimpleNamespace(material_binding=arguments["expected_model_binding"])
    subject.owner, subject.registry = object(), object()
    subject._phase = lambda name, operation: operation()
    options = {"admission": arguments["admission"], "candidate": descriptor,
               "task_cid": arguments["expected_task_cid"]}
    if mismatch:
        with pytest.raises(ValueError, match="loaded full public candidate"):
            subject.bind_worker_descriptor(**options)
        assert subject.bridge is None and not (subject.root / "worker-bridge.json").exists()
    else:
        bridge = subject.bind_worker_descriptor(**options)
        assert subject.worker_candidate == loaded
        assert subject.worker_descriptor == descriptor
        assert "finite_admission" in subject.worker_candidate
        assert "finite_admission" not in descriptor
        assert bridge["worker_candidate_cid"] == loaded["candidate_cid"]
        assert all(bridge[key] is False for key in support.CLAIMS)
    assert loader_calls == [(artifact, descriptor["sha256"])]
