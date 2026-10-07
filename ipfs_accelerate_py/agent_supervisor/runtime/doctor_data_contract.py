"""Signed finite data transformations and schedules through the candidate worker.

The shared datasets operators produce bytes and independently check finite
record correspondence or interval-schedule feasibility. Its finite-instance evidence is not a kernel theorem,
an interpretation of prose, or completion authority. The signed declaration
supplies the field mapping; model nominations and profile classifications do
not. Canonical source is never edited by this workflow.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath

from ..planning.intent_data_transform import validate_reviewed_data_transform
from ..proof.formal_verification_contracts import canonical_json, content_identity
from ..proof.proof_scope_index import (
    IndexedObligation, IndexedScopeRecord, ProofInputKind, ProofScopeBlobRecord,
    ProofScopeKey, build_proof_scope_index,
)
from ..semantic_state.program_world_database import ProgramWorldDatabase
from . import local_planning_admission as local
from .doctor_contract_candidate_runner import SCHEMA, publish_doctor_contract_candidate
from .doctor_scoped_analysis import build_scoped_doctor_analysis
from .supervisor_meta_index import SupervisorMetaIndex
from .terminal_source_partition import _read

WORKFLOW_SCHEMA = "supervisor-finite-data-contract-workflow@1"
REQUIREMENT_SCHEMA = "intent-plan-requirement-contract@4"
CHECK_SCOPE = (
    "Exact finite NDJSON record correspondence for the signed copy/rename rule: "
    "cardinality, order, multiplicity, scalar types and unrelated fields are preserved. "
    "This native check is not a kernel proof, prose-alignment proof or whole-program verification."
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _output_parent_available(repository, output_path, baseline):
    """The existing worker creates files, not new untracked directories."""
    relative = PurePosixPath(output_path).parent
    if str(relative) == ".":
        return True
    parent = repository / relative
    if parent.resolve() != parent or not parent.is_dir():
        return False
    return local._git(repository, "ls-tree", "-d", "--format=%(objecttype)",
                      baseline, "--", str(relative)) == "tree"


def _persist_index(*, scoped, task_id, declaration, check, state, check_scope=CHECK_SCOPE,
                   evidence_kind="finite_record_check"):
    """Hydrate observations without making a finite check an active proof."""
    state.mkdir(mode=0o700)
    schedule = evidence_kind == "finite_schedule_check"
    operation = "finite_interval_schedule" if schedule else "finite_data_contract"
    scopes, keys, scope_ids = [], [], []
    for name, digest in scoped.report["source_hashes"].items():
        key = ProofScopeKey(ProofInputKind.FILE, name)
        scope_id = content_identity({"path": name, "sha256": digest})
        scopes.append(ProofScopeBlobRecord(name, digest, (
            IndexedScopeRecord(scope_id, name, digest, (key,)),)))
        scope_ids.append(scope_id)
        keys.append(key)
    declaration_cid = content_identity(declaration)
    keys.extend((ProofScopeKey(ProofInputKind.POLICY, scoped.report["manifest_cid"]),
                 ProofScopeKey(ProofInputKind.PREMISE, declaration_cid)))
    obligation = IndexedObligation(declaration_cid, tuple(scope_ids), tuple(keys), (), {
        "scope": check_scope, "analysis_cid": scoped.report["analysis_cid"],
        "status": "finite_instance_checked" if check is not None else "unsupported",
        "kernel_proved": False})
    index = build_proof_scope_index(scope_blobs=scopes, obligations=(obligation,), receipts=(),
                                   root_id=scoped.report["source_tree_id"])
    index_body = index.to_dict()
    artifact = state / "contract-scope-index.json"
    artifact.write_text(json.dumps(index_body, sort_keys=True, indent=2) + "\n")
    record = {"task_id": task_id, "board": "doctor-contracts", "operation": operation,
        "analysis": scoped.report, "declaration": declaration, evidence_kind: check,
        "proof_index": index_body, "kernel_proved": False, "whole_program_proved": False,
        "proposal_only": True, "completion_authority": False}
    world = ProgramWorldDatabase(state / "contracts.duckdb", state / "contracts-lake")
    persisted = world.persist(record)
    hydrated = world.records_for_decision(task_id=task_id, operation=operation)
    if (hydrated["n"] != 1 or hydrated["records"][0]["payload"]["proof_index"] != index_body
            or hydrated["records"][0]["payload"][evidence_kind] != check):
        raise ValueError("finite contract world hydration differs")
    meta = SupervisorMetaIndex(state / "metadata.duckdb", state / "metadata-lake")
    catalog = meta.register_catalog(kind="world_model", locator_ref=str(state / "contracts.duckdb"),
        repository_id=scoped.snapshot.roots.repository_id,
        tree_id=scoped.report["source_tree_id"], project=False)
    meta.link_identity(subject_kind="task_id", subject_ref=task_id, catalog_id=catalog["catalog_id"],
        record_kind="world_model", record_ref=persisted["record_cid"], project=False)
    projection = meta.project_ducklake()
    return {"world_record": persisted, "metadata": projection, "artifact": str(artifact),
        "artifact_sha256": _sha(artifact.read_bytes()), "active_receipt_ids": [],
        "finite_check_recorded": check is not None, "hydrated": True,
        "completion_authority": False, "whole_program_proved": False}


SCHEDULE_CHECK_SCOPE = (
    "Exact finite integer-time feasibility for the signed jobs, resource availability and capacity: "
    "unique complete assignments, duration, release/deadline, horizon and half-open occupancy. "
    "An independent endpoint sweep checks the candidate; the SMT result alone grants no authority. "
    "Not a kernel theorem, optimality proof, prose-alignment proof or calendar/timezone interpretation."
)


def _prepare_contract_candidate(*, repository, admission, intent, task_cid, state, schedule):
    """Share source, state, index and staged-publication gates across closed operators."""
    if schedule:
        from ..planning.intent_interval_schedule import validate_reviewed_interval_schedule as validate
        from ipfs_datasets_py.logic.software_contracts.finite_interval_schedule import (
            FiniteIntervalScheduleContract as Contract, FiniteIntervalScheduleError as ContractError,
            solve_finite_interval_schedule as solve, check_finite_interval_schedule as check_candidate,
            verify_finite_schedule_check as verify_check,
        )
    else:
        from ipfs_datasets_py.logic.software_contracts.finite_record_projection import (
            FiniteRecordProjectionContract as Contract, FiniteRecordProjectionError as ContractError,
            synthesize_finite_record_projection as synthesize, check_finite_record_projection as check_candidate,
            verify_finite_record_check as verify_check,
        )
        validate = validate_reviewed_data_transform
    requirement_schema = "intent-plan-requirement-contract@5" if schedule else REQUIREMENT_SCHEMA
    selector_key = "reviewed_interval_schedule" if schedule else "reviewed_data_transform"
    check_scope = SCHEDULE_CHECK_SCOPE if schedule else CHECK_SCOPE
    evidence_kind = "finite_schedule_check" if schedule else "finite_record_check"
    media_type = "application/json" if schedule else "application/x-ndjson"
    prefix = "finite_schedule" if schedule else "finite_data"
    repository, state = Path(repository).absolute(), Path(state).absolute()
    if state.resolve() != state or state.is_relative_to(repository) or state.exists():
        raise ValueError("finite data workflow state must be a new external directory")
    scoped = build_scoped_doctor_analysis(repository=repository, admission=admission, task_cid=task_cid)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = verified["manifest"]
    task = intent.get_task(task_cid)
    if task is None or task["status"] not in {"ready", "in_progress"}:
        raise ValueError("finite data repair requires the current active admitted task")
    task_contract, _, _, _ = local._contract(task["body"], task_cid)
    spec = task_contract["task_spec"]
    if (task_contract["manifest_cid"] != scoped.report["manifest_cid"]
            or content_identity(spec) != scoped.report["task_spec_cid"]):
        raise ValueError("finite data task differs from independent admission")

    def assert_current():
        scoped.assert_current()
        if intent.get_task(task_cid) != task:
            raise ValueError("task revision changed during finite data repair")

    result = {"schema": "supervisor-finite-schedule-contract-workflow@1" if schedule else WORKFLOW_SCHEMA, "status": "residual", "repository": str(repository),
        "task_cid": task_cid, "task_revision": task["revision"], "analysis": scoped.report,
        "provider_calls": 0, "canonical_source_edits": 0, "proof_authority": False,
        "publication_authority": False, "completion_authority": False, "kernel_proved": False,
        "whole_program_proved": False, "evidence_kind": evidence_kind, "check_scope": check_scope,
        "reason_codes": ["doctor_task_data_contract_unavailable"]}
    requirements = (local.decode_intent_requirement_contract(manifest)
                    if "intent_requirements" in manifest else None)
    state.mkdir(parents=True, mode=0o700)
    if requirements is not None and requirements.get("schema") == requirement_schema:
        declaration = validate(requirements[selector_key],
            operations=requirements["symbolic_operations"]["operations"], manifest=manifest)
        contract = (Contract() if schedule else Contract(mode=declaration["mode"],
            source_field=declaration["source_field"], target_field=declaration["target_field"]))
        input_path, output_path = declaration["input_path"], declaration["output_path"]
        raw = _read(repository, input_path, manifest["sources"], 1_000_000)
        if scoped.sources.get(input_path) != raw:
            raise ValueError("finite data source differs from scoped analysis")
        result["declaration_cid"] = content_identity(declaration)
        result["operator"] = ("finite-integer-interval-schedule@1" if schedule
                              else "finite-ndjson-field-projection@1")
        check = None
        try:
            if not _output_parent_available(repository, output_path, manifest["baseline_commit"]):
                raise ContractError(prefix + "_output_parent_unavailable")
            if schedule:
                solved = solve(raw, contract)
                if (type(solved) is not dict or type(solved.get("status")) is not str
                        or solved["status"] not in {
                        "sat", "unsat", "unknown", "timeout", "unavailable", "cancelled", "error"}):
                    raise ValueError("schedule producer returned an invalid bounded status")
                result["solver_status"] = solved["status"]
                if solved["status"] != "sat":
                    if "output_bytes" in solved:
                        raise ValueError("non-SAT schedule result carries candidate bytes")
                    raise ContractError("finite_schedule_solver_" + solved["status"])
                after = solved.get("output_bytes")
            else:
                after = synthesize(raw, contract)
        except ContractError as error:
            admitted_reasons = {prefix + "_output_parent_unavailable"}
            if schedule:
                admitted_reasons.update("finite_schedule_solver_" + status for status in
                    ("unsat", "unknown", "timeout", "unavailable", "cancelled", "error"))
            result["reason_codes"] = [error.reason_code if error.reason_code in admitted_reasons
                else prefix + "_source_outside_reviewed_profile"]
        else:
            # A producer defect is not an unsupported task and must never hand
            # off a mismatching candidate, even if ordinary smoke would pass.
            check = check_candidate(raw, after, contract)
            assert_current()
            binding = {"schema": ("supervisor-finite-schedule-check-binding@1" if schedule
                                  else "supervisor-finite-data-check-binding@1"),
                "manifest_cid": scoped.report["manifest_cid"],
                "analysis_cid": scoped.report["analysis_cid"],
                "task_cid": task_cid, "task_revision": task["revision"],
                "task_spec_cid": scoped.report["task_spec_cid"],
                "requirement_contract_cid": manifest["intent_requirements"]["contract_cid"],
                "declaration_cid": result["declaration_cid"], "input_path": input_path,
                "output_path": output_path, "check": check,
                "workflow_sha256": _sha(Path(__file__).read_bytes()),
                "proof_authority": False, "publication_authority": False, "completion_authority": False}
            result["check"] = {**binding, "check_receipt_id": content_identity(binding)}
            bound_check_json = canonical_json(result["check"])
            result["reason_codes"] = []
        result["contract_index"] = _persist_index(scoped=scoped, task_id=task["task_alias"],
            declaration=declaration, check=result.get("check"), state=state / "index",
            check_scope=check_scope, evidence_kind=evidence_kind)
        assert_current()
        if check is not None:
            # Persistence returns observations, never replacement task/path or
            # receipt bindings. Compare canonical bytes (including bool types).
            if canonical_json(result["check"]) != bound_check_json:
                raise ValueError("finite check binding changed during index hydration")
            if _read(repository, input_path, manifest["sources"], 1_000_000) != raw:
                raise ValueError("finite data source changed before candidate handoff")
            verify_check(check, input_bytes=raw, output_bytes=after, contract=contract)
            assert_current()
            edit = {"path": output_path, "effect": "create", "before_sha256": None,
                    "after_sha256": _sha(after), "after_bytes_base64": base64.b64encode(after).decode()}
            if spec["outputs"] != [{"path": output_path, "effect": "create",
                                    "media_type": media_type}]:
                raise ValueError("finite operator must cover exactly the signed output")
            payload = {"schema": SCHEMA, "repository": str(repository),
                "baseline_commit": manifest["baseline_commit"], "task_cid": task_cid,
                "task_id": task["task_alias"], "task_revision": task["revision"],
                "manifest_cid": scoped.report["manifest_cid"],
                "proof_receipt_id": result["check"]["check_receipt_id"], "proof_scope": check_scope,
                "analysis_cid": scoped.report["analysis_cid"], "edits": [edit],
                "permitted_outputs": spec["outputs"], "provider_calls": 0,
                "publication_authority": False, "completion_authority": False}
            artifact, digest, cid = publish_doctor_contract_candidate(repository, payload)
            result.update(status="candidate_ready", route="doctor_contract_candidate",
                          artifact=str(artifact), sha256=digest, artifact_cid=cid)
    assert_current()
    (state / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def prepare_ndjson_contract_candidate(*, repository: Path, admission: dict, intent,
                                     task_cid: str, state: Path) -> dict:
    """Propose one created NDJSON output from the independently signed @4 rule."""
    return _prepare_contract_candidate(repository=repository, admission=admission, intent=intent,
        task_cid=task_cid, state=state, schedule=False)


def prepare_interval_schedule_candidate(*, repository: Path, admission: dict, intent,
                                        task_cid: str, state: Path) -> dict:
    """Propose a freshly checked finite schedule from the signed @5 declaration.

    Non-SAT solver reports stay residual. A mismatching independently checked
    witness or stale source/task binding is a hard refusal before staging.
    """
    return _prepare_contract_candidate(repository=repository, admission=admission, intent=intent,
        task_cid=task_cid, state=state, schedule=True)
