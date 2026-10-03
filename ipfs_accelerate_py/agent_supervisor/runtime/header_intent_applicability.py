"""Reviewed repair applicability from actual checks of captured header models.

This opt-in profile discharges a narrow operation precondition, never the
requested security goal. Every call replays native captured source and runs the
checker. No serialized `checked` field or caller-provided facts are accepted.
The enclosing local manifest owner still verifies source currentness, signature,
allowed outputs and publication transitions independently.
"""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import time

from ..core.multiformats_identity import cid_for_dag_json

SCHEMA = "reviewed-header-operation-selector@1"
NOMINATION_SCHEMA = "source-header-applicability-nomination@1"
PROFILE_SCHEMA = "source-header-applicability-profile@1"
CHECKER_PROFILE = "native-leased-bounded-header-checker@1"
CONTRACT_SCHEMA = "intent-plan-requirement-contract@3"
PREDICATE = "reviewed_guard_has_checked_local_model_counterexample"
OPERATOR = "reviewed-http-header-control-guard@1"
_DEADLINE = ContextVar("reviewed_header_applicability_deadline", default=None)
FALSE = dict(proof_authority=False, execution_authority=False,
             mutation_authority=False, completion_authority=False,
             source_semantics_verified=False, security_goal_satisfied=False)


@contextmanager
def applicability_budget(timeout_seconds):
    """Thread/task-local remaining budget, inherited by every nested replay.

    This carries only a deadline, never a checked fact or authority token.
    Nested scopes cannot renew or extend an enclosing caller's deadline.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 300):
        raise ValueError("bounded applicability aggregate deadline required")
    deadline = time.monotonic() + timeout_seconds
    previous = _DEADLINE.get()
    if previous is not None:
        deadline = min(deadline, previous)
    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def require_applicability_budget():
    deadline = _DEADLINE.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("aggregate applicability deadline expired before publication")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_applicability_selection(value, *, operations):
    """Validate reviewed intent, excluding any not-yet-observed runtime capture."""
    from ..planning.intent_requirement_adapter import _json, _object, _path, _text
    from ipfs_datasets_py.logic.security_ir.doctor_header_contracts import WsgiHeaderProtocolContract
    value = _json(value)
    _object(value, {"schema", "review_ref", "operation_id", "operator_id", "source_path",
                    "protocol", "checker_profile", *FALSE}, "header applicability selector")
    _require(value["schema"] == SCHEMA and value["operator_id"] == OPERATOR
             and value["checker_profile"] == CHECKER_PROFILE, "explicit reviewed header profile required")
    _text(value["review_ref"], "independent applicability review")
    _require(all(value[k] is False for k in FALSE), "applicability grants no authority")
    _require(len(operations) == 1 and operations[0]["operation_id"] == value["operation_id"],
             "first header profile supports one exact reviewed operation")
    path = _path(value["source_path"])
    _require(any(row["path"] == path and row["effect"] == "modify" for row in operations[0]["outputs"]),
             "modeled source must be a reviewed modify output")
    _object(value["protocol"], {"review_ref", "callback_parameter"}, "reviewed header protocol")
    WsgiHeaderProtocolContract(**value["protocol"])
    return value


def validate_runtime_profile(value):
    _require(type(value) is dict and set(value) == {"schema", "selector_cid", "checker_profile", "solver_sha256"}
             and value["schema"] == PROFILE_SCHEMA and value["checker_profile"] == CHECKER_PROFILE
             and type(value["selector_cid"]) is str and 0 < len(value["selector_cid"]) <= 256
             and _digest(value["solver_sha256"]), "closed bounded header runtime profile required")
    return json.loads(_raw(value))


def resolve_header_checker(profile):
    """Verify the runtime-installed solver bytes without executing it or repinning."""
    from .source384_config import _regular_bytes
    profile = validate_runtime_profile(profile)
    selected = shutil.which("z3")
    _require(selected is not None, "native Z3 is unavailable; no operation is applicable")
    executable = Path(selected).resolve(strict=True)
    _require(os.access(executable, os.X_OK), "selected native solver is not executable")
    raw = _regular_bytes(executable, 128 * 1024**2)
    digest = _sha(raw)
    _require(digest == profile["solver_sha256"], "selected runtime solver identity changed")
    return dict(executable=str(executable), sha256=digest, bytes=len(raw))


def _contract_binding(contract, manifest_cid, source_hashes, config):
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract
    contract = validate_intent_requirement_contract(contract)
    _require(contract["schema"] == CONTRACT_SCHEMA, "reviewed selector contract required")
    selection = contract["source_applicability"]
    profile = validate_runtime_profile(config.get("header_applicability"))
    _require(config["schema"] == "terminal-source384-config@2"
             and profile["selector_cid"] == cid_for_dag_json(selection)
             and profile["checker_profile"] == selection["checker_profile"], "runtime profile differs from reviewed selector")
    _require(type(manifest_cid) is str and 0 < len(manifest_cid) <= 256
             and _digest(source_hashes.get(selection["source_path"])), "signed source binding required")
    return selection, profile


def prepare_runtime_nomination(index, *, head, output, config_path, config, intent_binding,
                               source_hashes, parent_lease, remaining):
    """Called only after native capture; no source scan, solver or neural call."""
    from ipfs_datasets_py.logic.software_contracts import codebase_header_context as captured
    from .source384_config import _regular_bytes
    _require(type(intent_binding) is dict and set(intent_binding) == {"contract", "manifest_cid"},
             "closed immutable intent binding required")
    selection, profile = _contract_binding(intent_binding["contract"], intent_binding["manifest_cid"], source_hashes, config)
    receipt = captured.prepare_captured_header_context(index, expected_head=head,
        paths=[selection["source_path"]], protocol=captured.contracts.WsgiHeaderProtocolContract(**selection["protocol"]),
        parent_lease=parent_lease, timeout_seconds=remaining(), memory_mb=512)
    nomination = dict(schema=NOMINATION_SCHEMA, selection_cid=cid_for_dag_json(selection),
        manifest_cid=intent_binding["manifest_cid"], source_path=selection["source_path"],
        source_sha256=source_hashes[selection["source_path"]], store_root=str(output), captured_receipt=receipt,
        config_path=str(config_path), config_sha256=_sha(_regular_bytes(config_path, 32768)),
        checker_profile=profile["checker_profile"], solver_sha256=profile["solver_sha256"], **FALSE)
    _require(len(_raw(nomination)) <= 32768, "bounded header nomination required")
    remaining()
    return nomination


def validate_nomination_binding(nomination, *, contract, manifest):
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract
    from ..proof.formal_verification_contracts import content_identity
    from ..planning.intent_requirement_adapter import _json, _object
    contract = validate_intent_requirement_contract(contract)
    _require(contract["schema"] == CONTRACT_SCHEMA, "nomination requires explicit header intent profile")
    selection = contract["source_applicability"]
    nomination = _json(nomination)
    _object(nomination, {"schema", "selection_cid", "manifest_cid", "source_path", "source_sha256", "store_root",
        "captured_receipt", "config_path", "config_sha256", "checker_profile", "solver_sha256", *FALSE}, "header runtime nomination")
    declared = manifest.get("payload", manifest)
    _require(nomination["schema"] == NOMINATION_SCHEMA and len(_raw(nomination)) <= 32768
        and nomination["selection_cid"] == cid_for_dag_json(selection)
        and nomination["manifest_cid"] == content_identity(manifest)
        and nomination["source_path"] == selection["source_path"]
        and nomination["source_sha256"] == declared["sources"].get(selection["source_path"], {}).get("sha256")
        and nomination["checker_profile"] == selection["checker_profile"]
        and all(nomination[k] is False for k in FALSE)
        and all(_digest(nomination[k]) for k in ("source_sha256", "config_sha256", "solver_sha256")),
        "runtime nomination differs from independently signed selector/source")
    receipt = nomination["captured_receipt"]
    _require(type(receipt) is dict and receipt.get("schema") == "codebase-captured-header-receipt@1"
        and receipt.get("paths") == [selection["source_path"]] and receipt.get("protocol") == selection["protocol"],
        "captured nomination differs from reviewed source/protocol")
    for key in ("store_root", "config_path"):
        value = nomination[key]
        _require(type(value) is str and 0 < len(value) <= 4096 and Path(value).is_absolute()
            and str(Path(value)) == value and ".." not in Path(value).parts and not any(c in value for c in "\n\r\0"),
            "canonical external runtime nomination path required")
    return nomination


def validate_captured_nomination(index, *, head, nomination, config, config_path, output,
                                  source_hashes, parent_lease, remaining):
    """Replay historical captured data inside the caller's existing live boundary."""
    from ipfs_datasets_py.logic.software_contracts import codebase_header_context as captured
    from .source384_config import _regular_bytes
    profile = validate_runtime_profile(config.get("header_applicability"))
    _require(nomination["selection_cid"] == profile["selector_cid"]
        and nomination["checker_profile"] == profile["checker_profile"]
        and nomination["solver_sha256"] == profile["solver_sha256"]
        and nomination["store_root"] == str(output) and nomination["config_path"] == str(config_path)
        and nomination["config_sha256"] == _sha(_regular_bytes(config_path, 32768))
        and nomination["source_sha256"] == source_hashes[nomination["source_path"]]
        and nomination["captured_receipt"]["source_head"] == head.to_dict(), "runtime captured nomination changed")
    receipt = nomination["captured_receipt"]
    captured.validate_captured_header_context(index, expected_head=head, receipt=receipt,
        paths=receipt["paths"], protocol=captured.contracts.WsgiHeaderProtocolContract(**receipt["protocol"]),
        parent_lease=parent_lease, timeout_seconds=remaining(), memory_mb=512)
    remaining()


def _store(nomination, repository):
    root = Path(nomination["store_root"])
    _require(root.resolve(strict=True) == root and root.is_dir()
             and not root.is_relative_to(repository) and not repository.is_relative_to(root),
             "captured store must be canonical and outside the worker repository")
    info = root.stat()
    _require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
             "captured applicability store must be private to its owner")
    database = root / "source.duckdb"
    artifacts = root / "source-artifacts"
    _require(database.resolve(strict=True) == database and database.is_file()
             and artifacts.resolve(strict=True) == artifacts and artifacts.is_dir(),
             "existing native index and immutable artifact paths required")
    return root, database, artifacts


def checked_applicability(selection, *, nomination, manifest, timeout_seconds=45.,
                          scheduler=None, parent_lease=None, cancel_event=None):
    """Replay immutable source and real solver before returning a narrow fact.

    Historical replay intentionally reads the captured bytes; current-source
    checks belong to the enclosing native admission/dispatch boundary. All
    checks happen again even when a prior checker report exists.
    """
    started = time.monotonic()
    _require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 45., "bounded applicability deadline required")
    deadline = started + timeout_seconds
    inherited = _DEADLINE.get()
    if inherited is not None:
        deadline = min(deadline, inherited)
    if deadline <= started:
        raise TimeoutError("aggregate applicability deadline expired before replay")
    from ipfs_datasets_py.logic.software_contracts import codebase_header_context as captured
    from ipfs_datasets_py.logic.software_contracts import codebase_ir, cache, duckdb_ast_store, duckdb_ingest
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseHead
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as contracts
    from ipfs_datasets_py.logic.security_ir import bounded_header_checker
    from ipfs_datasets_py.logic.backends import codebase_process
    from ipfs_datasets_py.logic.backends.z3 import compiler as z3_compiler
    from ipfs_datasets_py.logic.backends.smt import differential as smt_runner
    import duckdb

    declared = manifest.get("payload", manifest)
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract
    bound = validate_intent_requirement_contract(json.loads(declared["intent_requirements"]["contract_json"]))
    _require(bound["schema"] == CONTRACT_SCHEMA and bound["source_applicability"] == selection
             and declared["intent_requirements"]["contract_cid"] == cid_for_dag_json(bound),
             "applicability differs from independently signed requirement binding")
    nomination = validate_nomination_binding(nomination, contract=bound, manifest=manifest)
    name = selection["source_path"]
    source = declared["sources"].get(name)
    _require(source is not None and source["sha256"] == nomination["source_sha256"],
             "captured model source differs from independently signed input")
    repository = Path(declared["repository"]).resolve(strict=True)
    root, database, artifacts_path = _store(nomination, repository)
    solver_profile = dict(schema=PROFILE_SCHEMA, selector_cid=nomination["selection_cid"],
        checker_profile=nomination["checker_profile"], solver_sha256=nomination["solver_sha256"])
    solver = resolve_header_checker(solver_profile)
    executable = Path(solver["executable"])
    solver_sha = solver["sha256"]
    def pins():
        return {__name__: _sha(Path(__file__).read_bytes()),
            **{m.__name__: _sha(Path(m.__file__).read_bytes()) for m in
                (header, contracts, captured, z3_compiler, smt_runner, bounded_header_checker, codebase_process)}}
    from .source384_config import _regular_bytes, validate_source384_config
    def check_config():
        raw = _regular_bytes(nomination["config_path"], 32768)
        _require(_sha(raw) == nomination["config_sha256"], "runtime header configuration changed")
        config = validate_source384_config(json.loads(raw))
        _, profile = _contract_binding(bound, nomination["manifest_cid"],
            {name: source["sha256"]}, config)
        _require(resolve_header_checker(profile) == solver, "selected runtime checker differs")
    check_config()
    producer = pins()
    receipt = nomination["captured_receipt"]
    protocol = contracts.WsgiHeaderProtocolContract(**receipt["protocol"])
    expected_head = CodebaseHead.from_dict(receipt["source_head"])
    # The native lease reserves one CPU/process and bounds admission separately
    # from aggregate execution. Every solver receives the remaining deadline.
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("applicability deadline expired before resource admission")
    with captured._operation(scheduler=scheduler, parent_lease=parent_lease,
            cancel_event=cancel_event, timeout_seconds=left, memory_mb=1024
            ) as (lease, signal, remaining), duckdb.connect(str(database),
                config={"threads": 1, "memory_limit": "128MB"}) as connection:
        store = duckdb_ast_store.DuckDBASTStore(connection=connection)
        artifacts = cache.ImmutableCAS(artifacts_path)
        index = codebase_ir.RepositoryCodebaseIndex(
            ingestor=duckdb_ingest.DuckDBASTIngestor(store=store), artifacts=artifacts,
            catalog=CodebaseCatalog(store, artifacts))
        head, paths, protocol = captured._inputs(index, expected_head, receipt["paths"], protocol)
        captured._validate(index, head, paths, protocol, receipt, remaining)
        report = artifacts.get(receipt["artifact_cid"], expected_schema=captured.SCHEMA)
        row = next(item for item in report["modules"] if item["path"] == name)
        _require(row["disposition"] == "modeled" and row["source_sha256"] == source["sha256"],
                 "reviewed source does not have a bound header model")
        body = artifacts.get_bytes(row["source_cid"])
        _require(len(body) == row["bytes"] and _sha(body) == source["sha256"],
                 "captured bytes differ from signed source")
        analysis = contracts.analyze_http_header_contracts(body.decode("utf-8"), protocol=protocol)
        candidate = analysis.candidate
        _require(candidate is not None and candidate.operator_id == OPERATOR
                 and contracts.verify_header_candidate(body.decode("utf-8"), candidate),
                 "independently replayable exact guard candidate required")
        checked = header.check_header_semantics(row["derivation"], source_bytes=body,
            source_path=name, protocol=protocol, z3_executable=str(executable),
            timeout_seconds=remaining(), cancel_event=signal, parent_lease=lease)
        remaining()
        _require(checked["status"] == "checked_local_model"
                 and checked.get("execution_profile") == "native-leased-bounded-header-checker@1"
                 and checked["solver_executable_sha256"] == solver_sha
                 and checked["solver_calls"] == len(row["derivation"]["smt_targets"])
                 and checked["solver_calls"] > 0,
                 "actual complete local-model checking required")
        counterexamples = [item for item in checked["results"]
            if item["kind"] == "unsafe_converted_input_accepted" and item["solver_answer"] == "sat"]
        _require(counterexamples and all(item["matches_model_expectation"] for item in checked["results"]),
                 "guard operation requires a checked local-model counterexample")
        _require(index.current(head.repository_id) == head, "captured head changed during checking")
        _require(producer == pins() and resolve_header_checker(solver_profile) == solver,
                 "checker producer or solver changed during checking")
        _require(_store(nomination, repository) == (root, database, artifacts_path),
                 "selected store location changed")
        # Replay once more without scanning live source. This catches captured
        # CAS/head changes during the external solver boundary.
        captured._validate(index, head, paths, protocol, receipt, remaining)
        check_config()
        _require(producer == pins() and resolve_header_checker(solver_profile) == solver,
                 "checker producer or solver changed during final captured replay")
        evidence = dict(schema="checked-header-operation-applicability@1",
            selection_cid=cid_for_dag_json(selection), nomination_cid=cid_for_dag_json(nomination),
            artifact_cid=receipt["artifact_cid"],
            source_head=head.to_dict(), source_sha256=source["sha256"],
            operation_id=selection["operation_id"], operator_id=OPERATOR,
            candidate_sha256=candidate.after_sha256, exact_candidate_replayed=True,
            check_cid=checked["check_cid"], checker_producer=producer,
            solver_sha256=solver_sha, solver_calls=checked["solver_calls"],
            checker_execution_profile=checked["execution_profile"],
            checked_obligations=[{key: item[key] for key in ("symbol", "kind", "query_mode",
                "solver_answer", "expected_model_answer", "script_digest", "script_sha256",
                "compilation_id", "solver_version")} for item in checked["results"]],
            assumptions=checked["assumptions"], open_frontiers=checked["open_frontiers"],
            native_source_observer_calls=0, provider_calls=0, learned_formula_count=0, **FALSE)
        remaining()
        return evidence


def ground_materials(materials, *, selection, nomination, manifest, timeout_seconds=45.):
    """Execute the checker; then attach only an operation applicability atom."""
    from ..planning.obligation_graph_compiler import (
        TypedPredicate, ObservedFact, FactTruth, FactAuthority,
    )
    evidence = checked_applicability(selection, nomination=nomination, manifest=manifest, timeout_seconds=timeout_seconds)
    evidence_cid = cid_for_dag_json(evidence)
    operation_id = selection["operation_id"]
    producer = next(row for row in materials.producers
        if materials.operation_candidate_ids[operation_id] in row.task_candidate_ids)
    predicate = TypedPredicate(predicate_id="predicate:" + evidence_cid,
        predicate_type=PREDICATE, subject_ref=materials.contract_cid, object_ref=operation_id,
        provenance_refs=(evidence_cid, nomination["captured_receipt"]["artifact_cid"]),
        invalidation_selectors=producer.invalidation_selectors)
    fact = ObservedFact(fact_id="fact:" + evidence_cid, predicate=predicate,
        truth=FactTruth.TRUE, authority=FactAuthority.BOUNDED_OBSERVATION,
        provenance_refs=(evidence_cid,), current_root_id=materials.current_root_id,
        invalidation_selectors=producer.invalidation_selectors)
    producers = tuple(replace(row,
        required_predicate_ids=tuple(sorted((*row.required_predicate_ids, predicate.predicate_id))),
        provenance_refs=tuple(sorted((*row.provenance_refs, evidence_cid))))
        if row.producer_id == producer.producer_id else row for row in materials.producers)
    from ..planning.intent_requirement_adapter import _freeze
    return replace(materials, predicates=(*materials.predicates, predicate),
                   producers=producers, current_facts=(fact,), source_applicability=_freeze(evidence))
