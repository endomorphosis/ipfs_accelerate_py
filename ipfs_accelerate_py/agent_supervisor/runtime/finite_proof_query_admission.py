"""Current proof-query context joined to complete finite local admission.

The new signature binds indexed conditional mathematics and independently owned
finite observations. Facts retain BOUNDED_OBSERVATION authority. Every original
administrator task and pending local acceptance check remains in the native
transaction. This profile does not extend the worker's execution authority.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import time

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_structured,
)

from ..planning import finite_proof_query_join as join
from ..planning.finite_integer_source_custody import (
    _read as read_custody_file, capture_source_custody,
)
from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ..task_sources.intent_repository import IntentRepository
from . import finite_repository_admission as finite
from . import local_planning_admission as local

SCHEMA = "finite-proof-query-admission-bundle@1"
RECEIPT_SCHEMA = "finite-proof-query-admission@1"
REFERENCE_SCHEMA = "finite-proof-query-admission-reference@1"
PROFILE = "finite-observation-with-current-proof-query@1"
MAX_BYTES = 16 * 1024**2
_FALSE = {name: False for name in (
    "proof_authority", "behavior_authority", "source_semantics_verified",
    "runtime_behavior_verified", "execution_authority", "completion_authority",
    "omission_authority", "mutation_authority", "worker_launched",
    "training_executed", "model_inference_executed",
)}
_POLICY = {
    "facts": "fresh_exact_finite_observation_only",
    "indexed_proofs": "historical_conditional_model_context_only",
    "query": "complete_singleton_full_key_current_inventory_and_epoch",
    "task_population": "complete_original_administrator_population",
    "completion": "existing_signed_pending_local_acceptance_checks",
    "model": "explicit-model-off@1",
    "worker_proof_query_fence": False,
}
_INDEXED_FIELDS = frozenset({
    "schema", "profile", "proof_query_closure", "proof_query_closure_cid", "match",
    "input_snapshot", "preview", "operation_catalog", "operation_catalog_cid",
    "requirement_ledger", "obligation_graph", "portfolio", "candidate_plan",
    "critique", "critic_evidence", "execution_plan", "planner_status",
    "declared_task_requirement_ids", "selected_task_ids", "current_facts_count",
    "removed_task_ids", "scope", "model_selection", "model_calls", "training_steps",
    "solver_calls", "producer", "authority", "result_cid",
}) | frozenset(join._AUTHORITY)


class FiniteProofQueryAdmissionError(ValueError):
    """The complete current proof/planning/admission join failed closed."""


def _need(condition, message):
    if not condition:
        raise FiniteProofQueryAdmissionError(message)


def _wire(value):
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= 350_000 and depth <= 40, "bounded exact admission JSON required")
        if type(item) is dict:
            _need(len(item) <= 350_000 - count and all(type(key) is str for key in item),
                  "bounded exact admission keys required")
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(len(item) <= 350_000 - count, "bounded exact admission arrays required")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int:
            _need(item.bit_length() <= 128, "bounded exact admission integer required")
        else:
            _need(type(item) in {bool, type(None)}, "inert exact admission JSON required")
        _need(text_bytes <= MAX_BYTES, "complete admission text byte bound exceeded")
    raw = canonical_dag_json_bytes(value)
    _need(len(raw) <= MAX_BYTES, "complete proof-query admission byte bound exceeded")
    return raw


def _plain(value):
    return json.loads(_wire(value))


def _same(left, right):
    return _wire(left) == _wire(right)


def _pins():
    names = (
        __name__, join.__name__, finite.__name__, local.__name__,
        "ipfs_datasets_py.duckdb_control.codebase_verification_catalog",
        "ipfs_datasets_py.duckdb_control.codebase_verification_queries",
    )
    return {"schema": "finite-proof-query-admission-implementation@1",
        "source_sha256": {name: hashlib.sha256(
            Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in names},
        "scope": "selected producer bytes; no process-origin attestation"}


def _budget(owner):
    _need(type(owner) is RepositoryPlanPreviewOwner, "exact live repository owner required")
    deadline = time.monotonic() + owner.timeout_seconds

    def checkpoint():
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            LeaseCancelledError, LeaseTimeoutError,
        )
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("finite proof-query admission cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("finite proof-query admission deadline exceeded")
        return left

    return checkpoint


def _current_owner(owner, checkpoint):
    return replace(owner, timeout_seconds=checkpoint())


def _query(owner, base, catalog):
    _, _, _, _, document, _, _ = finite._declaration(base["declaration"])
    return join.capture_current_finite_proof_query(owner=owner,
        verification_catalog=catalog, intent_document=document,
        source_text=base["declaration"]["payload"]["source_text"])


def _plan(owner, base, catalog, observer):
    payload, _, request, _, document, operations, _ = finite._declaration(base["declaration"])
    return join._preview_owned_finite_proof_join(owner=owner, request=request,
        intent_document=document, source_text=payload["source_text"],
        operation_catalog=operations, match=base["evidence"]["match"],
        verification_catalog=catalog, policy_observer=observer)


def _payload(base, indexed, semantic):
    _need(type(indexed) is dict and set(indexed) == _INDEXED_FIELDS
          and indexed.get("schema") == join.SCHEMA and indexed.get("profile") == join.PROFILE
          and indexed.get("result_cid") == cid_for_structured(
              {key: value for key, value in indexed.items() if key != "result_cid"}),
          "complete native indexed finite plan identity required")
    _need(_same(indexed["authority"], join._AUTHORITY)
          and all(indexed[name] is False for name in join._AUTHORITY)
          and indexed["match"] == base["evidence"]["match"],
          "indexed proposal must preserve the independently reconstructed finite match")
    return {
        "schema": RECEIPT_SCHEMA, "profile": PROFILE,
        "finite_admission_cid": cid_for_structured(base),
        "declaration_cid": cid_for_structured(base["declaration"]),
        "graph_cid": cid_for_structured(base["graph"]),
        "evidence_cid": cid_for_structured(base["evidence"]),
        "semantic_context_cid": cid_for_structured(semantic),
        "indexed_plan_cid": indexed["result_cid"],
        "input_snapshot_cid": indexed["input_snapshot"]["snapshot_cid"],
        "proof_query_closure_cid": indexed["proof_query_closure_cid"],
        "administrator_task_cids": semantic["administrator_task_cids"],
        "planning_permitted": bool(semantic["residual_requirement_ids"]),
        "no_work_review_only": not semantic["residual_requirement_ids"],
        "policy": _POLICY, "implementation": _pins(), **_FALSE,
    }


def _received(admission):
    """Authenticate the closed envelope and independently replay base facts."""
    value = _plain(admission)
    _need(type(value) is dict and set(value) ==
          {"schema", "finite_admission", "indexed_plan", "receipt"}
          and value["schema"] == SCHEMA, "exact proof-query admission bundle required")
    verified = finite.verify_finite_repository_admission(admission=value["finite_admission"])
    base = verified["admission"]
    _, profile, *_ = finite._declaration(base["declaration"])
    received = local._verify_signature(value["receipt"], profile)
    expected = _payload(base, value["indexed_plan"], verified["semantic_context"])
    _need(_same(received, expected), "signed proof-query admission does not bind its complete inputs")
    return value, verified


def _proof_file_fence(owner, closure, checkpoint):
    """Seal complete selected CAS bodies after native replay; no callbacks."""
    objects = (
        (closure["projection_cid"], closure["projection"]),
        (closure["verification_cid"], closure["verification"]),
        (closure["applicability_cid"], closure["applicability"]),
    )
    sealed, total = [], 0
    for cid, value in objects:
        _need(cid_for_structured(value) == cid, "selected complete proof object identity differs")
        path = owner.index.artifacts.path_for(cid).resolve(strict=True)
        _need(path == owner.index.artifacts.path_for(cid), "proof CAS path alias differs")
        witness, raw = read_custody_file(path, role="proof-query:" + cid,
            bound=MAX_BYTES, checkpoint=checkpoint)
        _need(raw == canonical_dag_json_bytes(value), "selected physical proof body differs")
        total += len(raw)
        _need(total <= MAX_BYTES, "selected physical proof closure exceeds its bound")
        sealed.append(witness)

    def fence():
        for original in sealed:
            current, _ = read_custody_file(original.path, role=original.role,
                bound=MAX_BYTES, checkpoint=checkpoint)
            _need(current == original, "selected physical proof object changed after native replay")
        checkpoint()

    fence()
    return fence


def _replay(owner, value, catalog, observer):
    """Rebuild every frozen service input, query, clause and reviewed effect."""
    current = _plan(owner, value["finite_admission"], catalog, observer)
    _need(_same(current, value["indexed_plan"]),
          "current proof query, complete planning input or reviewed effect differs")
    return current


def admit_finite_proof_query_plan(*, owner, declaration, graph,
        verification_catalog, output, policy_observer):
    """Observe and plan natively, then sign both complete admission records."""
    checkpoint = _budget(owner)
    # Missing or ambiguous evidence refuses before fresh repository execution.
    payload, _, _, _, document, _, _ = finite._declaration(declaration)
    initial = join.capture_current_finite_proof_query(owner=_current_owner(owner, checkpoint),
        verification_catalog=verification_catalog, intent_document=document,
        source_text=payload["source_text"])
    base = finite.admit_finite_repository_plan(owner=_current_owner(owner, checkpoint),
        declaration=declaration, graph=graph, output=output, policy_observer=policy_observer)
    verified = finite.verify_finite_repository_admission(admission=base)
    custody = capture_source_custody(owner, checkpoint)
    indexed = _plan(_current_owner(owner, checkpoint), base, verification_catalog, policy_observer)
    _need(_same(initial, indexed["proof_query_closure"]),
          "evidence inventory changed while producing finite admission")
    proof_fence = _proof_file_fence(owner, indexed["proof_query_closure"], checkpoint)
    receipt = _payload(base, indexed, verified["semantic_context"])
    value = _plain({"schema": SCHEMA, "finite_admission": base, "indexed_plan": indexed,
        "receipt": local._signed(receipt, payload["manifest"]["payload"])})
    _received(value)
    _replay(_current_owner(owner, checkpoint), value, verification_catalog, policy_observer)
    finite._artifact_fence(owner, base["evidence"], checkpoint)()
    custody.require_current(checkpoint)
    join._require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
        closure=indexed["proof_query_closure"], checkpoint=checkpoint)
    proof_fence()
    return value


def verify_current_finite_proof_query_admission(*, owner, admission,
        verification_catalog, output, policy_observer):
    """Reobserve Python/Lean semantics and the exact current native query."""
    checkpoint = _budget(owner)
    value, verified = _received(admission)
    closure = value["indexed_plan"]["proof_query_closure"]
    _need(_same(_query(_current_owner(owner, checkpoint), value["finite_admission"],
                      verification_catalog), closure), "proof query inventory or epoch is stale")
    proof_fence = _proof_file_fence(owner, closure, checkpoint)
    custody = capture_source_custody(owner, checkpoint)
    fresh = finite.verify_current_finite_repository_admission(
        owner=_current_owner(owner, checkpoint), admission=value["finite_admission"],
        output=output, policy_observer=policy_observer)
    _replay(_current_owner(owner, checkpoint), value, verification_catalog, policy_observer)
    _received(value)
    finite._artifact_fence(owner, value["finite_admission"]["evidence"], checkpoint)()
    custody.require_current(checkpoint)
    join._require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
        closure=closure, checkpoint=checkpoint)
    proof_fence()
    return {"schema": "finite-proof-query-current-verification@1",
        "admission_cid": cid_for_structured(value), "observed_current": True,
        "proof_query_closure_cid": value["indexed_plan"]["proof_query_closure_cid"],
        "fresh_evidence_cid": cid_for_structured(fresh["fresh_evidence"]),
        "semantic_context": verified["semantic_context"], **_FALSE}


def _store(value, manifest):
    # Existing immutable publication and descriptor/path checks; a new schema
    # prevents treating this additional context as an old execution reference.
    reference = finite._store(value, manifest)
    return {**reference, "schema": REFERENCE_SCHEMA}


def materialize_finite_proof_query_plan(*, owner, admission, intent,
        verification_catalog, output, policy_observer):
    """Commit all original tasks only after receiving replay inside SQL."""
    _need(type(intent) is IntentRepository and not intent.uses_bound_connection,
          "independently owned native intent transaction required")
    checkpoint = _budget(owner)
    value, verified = _received(admission)
    base = value["finite_admission"]
    _need(verified["receipt"]["planning_permitted"] is True and base["local_admission"] is not None,
          "no-work finite evidence remains review-only")
    closure = value["indexed_plan"]["proof_query_closure"]
    _need(_same(_query(_current_owner(owner, checkpoint), base, verification_catalog), closure),
          "proof query inventory or epoch is stale before native writes")
    proof_fence = _proof_file_fence(owner, closure, checkpoint)
    old_artifact_fence = finite._artifact_fence(owner, base["evidence"], checkpoint)
    evidence, semantic, custody, _, fresh_fence = finite._fresh(
        _current_owner(owner, checkpoint), base["declaration"], base["graph"], output, policy_observer)
    _need(_same(semantic, verified["semantic_context"]), "fresh finite semantics differ")
    _replay(_current_owner(owner, checkpoint), value, verification_catalog, policy_observer)
    manifest = base["declaration"]["payload"]["manifest"]["payload"]
    base_ref = finite._store(base, manifest)
    reference = _store(value, manifest)
    with intent._connection(write=True) as connection:
        with IntentRepository(bound_connection=connection, owner_id=intent.owner_id,
                              session_id=intent.session_id) as bound:
            result = local._materialize_local_benchmark_plan(admission=base["local_admission"], intent=bound)
            _need(result["task_cids"] == semantic["administrator_task_cids"],
                  "native commit omitted an original administrator task")
            plan = bound.get_plan(result["plan_id"])
            bound.upsert_plan(plan_cid=plan["plan_cid"], goal_cid=plan["goal_cid"],
                plan_alias=plan["plan_alias"], status=plan["status"], expected_revision=plan["revision"],
                body={**plan["body"], "finite_repository_admission_ref": base_ref,
                      "finite_proof_query_admission_ref": reference})
            # All signing/install callbacks have run. A refusal here rolls back
            # objectives, goals, plans, tasks and their native dependency rows.
            _received(value)
            _replay(_current_owner(owner, checkpoint), value, verification_catalog, policy_observer)
            finite._evidence(evidence, base["declaration"], base["graph"], checkpoint)
            for ref, body in ((base_ref, base), (reference, value)):
                _need(finite._read(ref["path"], MAX_BYTES, checkpoint) == _wire(body),
                      "retained signed admission changed before native commit")
            custody.require_current(checkpoint)
            fresh_fence()
            old_artifact_fence()
            join._require_frozen_proof_inventory(owner=owner, verification_catalog=verification_catalog,
                closure=closure, checkpoint=checkpoint)
            proof_fence()
    return {**result, "schema": "finite-proof-query-native-materialization@1",
        "finite_admission_ref": base_ref, "finite_proof_query_admission_ref": reference,
        "admission_cid": reference["admission_cid"],
        "fresh_evidence_cid": cid_for_structured(evidence),
        "administrator_task_population_preserved": True, "observed_current": True, **_FALSE}


__all__ = ["SCHEMA", "PROFILE", "FiniteProofQueryAdmissionError",
    "admit_finite_proof_query_plan", "verify_current_finite_proof_query_admission",
    "materialize_finite_proof_query_plan"]
