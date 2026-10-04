"""Reviewed repair applicability from actual checks of captured header models.

This opt-in profile discharges a narrow operation precondition, never the
requested security goal. Every call replays native captured source and runs the
checker. No serialized `checked` field or caller-provided facts are accepted.
The enclosing local manifest owner still verifies source currentness, signature,
allowed outputs and publication transitions independently.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
_LOCAL_BUDGET = ContextVar("local_benchmark_applicability_budget", default=None)
FALSE = dict(proof_authority=False, execution_authority=False,
             mutation_authority=False, completion_authority=False,
             source_semantics_verified=False, security_goal_satisfied=False)


@dataclass(frozen=True)
class _LocalReplayBudget:
    deadline_monotonic: float


def applicability_replay_timeout(timeout_seconds=None):
    """Select a replay ceiling; an inherited deadline can only shorten it."""
    maximum = 120. if _LOCAL_BUDGET.get() is not None else 45.
    value = maximum if timeout_seconds is None else timeout_seconds
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= maximum):
        raise ValueError("bounded applicability deadline required")
    return float(value)


@contextmanager
def local_benchmark_applicability_budget(*, deadline_monotonic):
    """Explicit local-profile work deadline, carrying no proof authority."""
    now = time.monotonic()
    if (type(deadline_monotonic) not in (int, float)
            or not math.isfinite(deadline_monotonic)
            or not 0 < deadline_monotonic - now <= 900):
        raise ValueError("bounded local applicability work deadline required")
    with resume_applicability_budget(_LocalReplayBudget(float(deadline_monotonic))) as scope:
        yield scope


def capture_applicability_budget():
    """Capture the remaining local work scope for an explicit thread handoff."""
    scope = _LOCAL_BUDGET.get()
    if scope is None:
        return None
    inherited = _DEADLINE.get()
    return _LocalReplayBudget(min(scope.deadline_monotonic,
                                 inherited if inherited is not None else scope.deadline_monotonic))


@contextmanager
def resume_applicability_budget(scope, *, deadline_monotonic=None):
    """Resume a captured scope without renewing it, including across threads."""
    if scope is None:
        yield None
        return
    if type(scope) is not _LocalReplayBudget:
        raise ValueError("an exact captured applicability budget is required")
    if os.environ.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE") != "local-benchmark@1":
        raise ValueError("explicit local benchmark resource profile required")
    deadline = scope.deadline_monotonic
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline - time.monotonic() > 900):
        raise ValueError("bounded captured applicability deadline required")
    if deadline_monotonic is not None:
        if type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic):
            raise ValueError("finite enclosing applicability deadline required")
        deadline = min(deadline, deadline_monotonic)
    previous = _DEADLINE.get()
    local = _LOCAL_BUDGET.get()
    if previous is not None:
        deadline = min(deadline, previous)
    if local is not None:
        deadline = min(deadline, local.deadline_monotonic)
    if deadline <= time.monotonic():
        raise TimeoutError("aggregate applicability deadline expired before replay")
    captured = _LocalReplayBudget(deadline)
    deadline_token = _DEADLINE.set(deadline)
    local_token = _LOCAL_BUDGET.set(captured)
    try:
        yield captured
    finally:
        _LOCAL_BUDGET.reset(local_token)
        _DEADLINE.reset(deadline_token)


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


_CHECK_STATUSES = frozenset({"unsupported", "solver_unavailable", "checked_local_model",
    "model_check_inconclusive_or_mismatch", "invalid"})
_QUERY_STATUSES = frozenset({"proved", "disproved", "satisfiable", "unsatisfiable",
    "unknown", "error", "invalid"})
_CHECK_REFUSALS = frozenset({"check_status", "execution_profile", "solver_identity",
    "solver_call_count", "missing_counterexample", "query_expectation"})


def _bounded_count(value):
    return value if type(value) is int and 0 <= value <= 65535 else None


def _checker_refusal(checked, *, expected_calls, solver_sha, reasons, message):
    """Attach only bounded verdict metadata, never solver/model/source bodies."""
    results = checked.get("results")
    rows = results[:64] if type(results) is list else []
    counts = {}
    for row in rows:
        status = row.get("status") if type(row) is dict else None
        status = status if type(status) is str and status in _QUERY_STATUSES else "invalid"
        counts[status] = counts.get(status, 0) + 1
    status = checked.get("status")
    status = status if type(status) is str and status in _CHECK_STATUSES else "invalid"
    error = ValueError(message)
    error.header_checker_diagnostic = dict(schema="header-model-check-refusal@1",
        reason_codes=list(reasons), status=status,
        execution_profile_matches=checked.get("execution_profile") == CHECKER_PROFILE,
        solver_identity_matches=checked.get("solver_executable_sha256") == solver_sha,
        expected_solver_calls=_bounded_count(expected_calls),
        observed_solver_calls=_bounded_count(checked.get("solver_calls")),
        result_count=_bounded_count(len(results)) if type(results) is list else None,
        result_status_counts=counts, result_rows_truncated=type(results) is list and len(results) > 64)
    return error


def project_header_checker_failure(error):
    """Read closed diagnostics through at most eight explicit exception causes."""
    seen = set()
    for _ in range(8):
        if not isinstance(error, BaseException) or id(error) in seen:
            return None
        seen.add(id(error))
        try:
            value = vars(error).get("header_checker_diagnostic")
            cause = error.__cause__
        except Exception:
            return None
        if type(value) is dict:
            if (set(value) == {"schema", "phase", "reason"}
                    and value["schema"] == "bounded-header-checker-failure@1"
                    and type(value["phase"]) is str and value["phase"] in {"child_admission", "version_probe", "query", "child_release"}
                    and type(value["reason"]) is str and value["reason"] in {"admission_timeout", "cancelled", "deadline", "tool_refusal"}):
                return dict(value)
            if (set(value) == {"schema", "reason_codes", "status", "execution_profile_matches",
                    "solver_identity_matches", "expected_solver_calls", "observed_solver_calls",
                    "result_count", "result_status_counts", "result_rows_truncated"}
                    and value["schema"] == "header-model-check-refusal@1"
                    and type(value["reason_codes"]) is list and 0 < len(value["reason_codes"]) <= 6
                    and all(type(item) is str and item in _CHECK_REFUSALS for item in value["reason_codes"])
                    and type(value["status"]) is str and value["status"] in _CHECK_STATUSES
                    and all(type(value[key]) is bool for key in (
                        "execution_profile_matches", "solver_identity_matches", "result_rows_truncated"))
                    and all(value[key] is None or _bounded_count(value[key]) is not None for key in (
                        "expected_solver_calls", "observed_solver_calls", "result_count"))
                    and type(value["result_status_counts"]) is dict
                    and len(value["result_status_counts"]) <= len(_QUERY_STATUSES)
                    and all(type(key) is str and key in _QUERY_STATUSES and type(count) is int and 0 <= count <= 64
                            for key, count in value["result_status_counts"].items())
                    and sum(value["result_status_counts"].values()) <= 64):
                return {**value, "reason_codes": list(value["reason_codes"]),
                        "result_status_counts": dict(value["result_status_counts"])}
        error = cause
    return None


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


def checked_applicability(selection, *, nomination, manifest, timeout_seconds=None,
                          scheduler=None, parent_lease=None, cancel_event=None):
    """Replay immutable source and real solver before returning a narrow fact.

    Historical replay intentionally reads the captured bytes; current-source
    checks belong to the enclosing native admission/dispatch boundary. All
    checks happen again even when a prior checker report exists.
    """
    started = time.monotonic()
    timeout_seconds = applicability_replay_timeout(timeout_seconds)
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
        expected_calls = len(row["derivation"]["smt_targets"])
        failures = [reason for reason, accepted in (
            ("check_status", checked.get("status") == "checked_local_model"),
            ("execution_profile", checked.get("execution_profile") == CHECKER_PROFILE),
            ("solver_identity", checked.get("solver_executable_sha256") == solver_sha),
            ("solver_call_count", checked.get("solver_calls") == expected_calls and checked["solver_calls"] > 0),
        ) if not accepted]
        if failures:
            raise _checker_refusal(checked, expected_calls=expected_calls, solver_sha=solver_sha,
                reasons=failures, message="actual complete local-model checking required")
        counterexamples = [item for item in checked["results"]
            if item["kind"] == "unsafe_converted_input_accepted" and item["solver_answer"] == "sat"]
        failures = [reason for reason, accepted in (
            ("missing_counterexample", bool(counterexamples)),
            ("query_expectation", all(item["matches_model_expectation"] for item in checked["results"])),
        ) if not accepted]
        if failures:
            raise _checker_refusal(checked, expected_calls=expected_calls, solver_sha=solver_sha,
                reasons=failures, message="guard operation requires a checked local-model counterexample")
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


def ground_materials(materials, *, selection, nomination, manifest, timeout_seconds=None):
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
