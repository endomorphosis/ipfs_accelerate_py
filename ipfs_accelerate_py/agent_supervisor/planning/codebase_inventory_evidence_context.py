"""Current inventory/evidence references at the structural planning seam.

Learned features and recorded conditional evidence remain advisory metadata.
The caller supplies the independently authored intent, producer declarations
and tasks. No row becomes an observed fact, runtime discharge or admission.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import CodeType, FunctionType
from typing import Any

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured, validate_cid

from ..prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreateMaterials, PlanCreateMode, PlanCreatePreviewReceipt,
    PlanCreateServiceError, freeze_plan_create_input_snapshot,
)
from .obligation_graph_compiler import ProducerRule, TaskCandidate, TypedIntent
from .plan_revision_contracts import PlanCreateRequest
from .repository_plan_preview import RepositoryPlanPreviewOwner, preview_repository_plan

SCHEMA = "supervisor-codebase-inventory-evidence-plan-preview@1"
MATERIAL_KEY = "codebase_inventory_evidence"
MAX_MATERIAL_BYTES = 8 * 1024 * 1024
_IMPORTED_SOURCE_PIN: str | None = None
_AUTHORITY_NAMES = frozenset({
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "authoritative_cache_eligible", "behavioral_satisfaction",
    "training_executed", "decoded_formulas_generated", "repository_code_executed",
    "source_execution_attested", "scan_execution_attested",
})
_FALSE = {name: False for name in sorted(_AUTHORITY_NAMES | {
    "production_admitted", "worker_launched", "omission_authority",
})}


class CodebaseInventoryEvidenceContextError(ValueError):
    """Current advisory references cannot be consumed by this proposal profile."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CodebaseInventoryEvidenceContextError(message)


def _wire(value: Any) -> bytes:
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= 100_000 and depth <= 32, "bounded inert material JSON required")
        if type(item) is dict:
            _require(len(item) <= 100_000 - nodes and all(type(key) is str for key in item),
                     "bounded exact material JSON keys required")
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _require(len(item) <= 100_000 - nodes, "material collection bound exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int:
            _require(item.bit_length() <= 128, "bounded material integer required")
        else:
            _require(type(item) in {bool, type(None)}, "float-free inert material JSON required")
        _require(text_bytes <= MAX_MATERIAL_BYTES, "material text bound exceeded")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise CodebaseInventoryEvidenceContextError("canonical inert material JSON required") from exc
    _require(len(raw) <= MAX_MATERIAL_BYTES, "material byte bound exceeded")
    return raw


def _cid(value: Any, *, raw: bool = False) -> None:
    _require(type(value) is str, "exact material CID required")
    try:
        validate_cid(value, codecs={"raw" if raw else "dag-json"})
    except (ValueError, TypeError) as exc:
        raise CodebaseInventoryEvidenceContextError("canonical material CID required") from exc


def _validate_refs(value: Any, *, artifact_cid: str, head: CodebaseHead) -> dict[str, Any]:
    """Detach the complete float-free ledger; native validation supplies custody."""
    value = json.loads(_wire(value))
    fields = {"schema", "artifact_cid", "head", "head_cid", "scan_artifact_cid",
              "membership_cid", "model", "coverage", "query", "entry_evidence", "authority"}
    _require(type(value) is dict and set(value) == fields
             and value["schema"] == "codebase-inventory-evidence-advisory-refs@1",
             "closed advisory reference fields required")
    _require(value["artifact_cid"] == artifact_cid and value["head"] == head.to_dict()
             and value["head_cid"] == cid_for_structured(head.to_dict()),
             "advisory record or complete head identity differs")
    for name in ("artifact_cid", "scan_artifact_cid"):
        _cid(value[name], raw=True)
    for name in ("head_cid", "membership_cid"):
        _cid(value[name])
    _require(type(value["authority"]) is dict and set(value["authority"]) == _AUTHORITY_NAMES
             and all(flag is False for flag in value["authority"].values()),
             "advisory references must retain exact false authority flags")
    model = value["model"]
    model_fields = {"version_id", "variant_id", "artifact_cid", "contract_sha256",
                    "state_sha256", "feature_space_sha256"}
    _require(type(model) is dict and set(model) == model_fields, "complete advisory model identity required")
    for name in ("version_id", "variant_id"):
        _require(type(model[name]) is str and 0 < len(model[name].encode("utf-8")) <= 512,
                 "bounded exact model identity required")
    _cid(model["artifact_cid"], raw=True)
    for name in ("contract_sha256", "state_sha256", "feature_space_sha256"):
        digest = model[name]
        _require(type(digest) is str and len(digest) == 64
                 and all(char in "0123456789abcdef" for char in digest), "exact model SHA256 required")
    coverage = value["coverage"]
    coverage_fields = {"inventory_entries", "inferred_rows", "evidence_entries", "evidence_matched_members",
                       "evidence_complete_absent_members", "evidence_unknown_members"}
    _require(type(coverage) is dict and set(coverage) == coverage_fields
             and all(type(count) is int and count >= 0 for count in coverage.values()),
             "complete exact advisory coverage counts required")
    rows = value["entry_evidence"]
    _require(type(rows) is list and len(rows) <= 256 and len(rows) == coverage["inventory_entries"]
             and coverage["inferred_rows"] <= len(rows)
             and sum(coverage[name] for name in ("evidence_matched_members", "evidence_complete_absent_members",
                                                "evidence_unknown_members")) == len(rows),
             "complete advisory member coverage required")
    ordered, seen, evidence_ids, dispositions = [], set(), set(), Counter()
    for row in rows:
        _require(type(row) is dict and set(row) == {"source_key", "entry_cid", "evidence_disposition", "evidence_entry_ids"},
                 "closed advisory member ledger required")
        key = row["source_key"]
        _require(type(key) is str and 0 < len(key.encode("utf-8")) <= 4096 and key not in seen,
                 "unique bounded advisory source key required")
        seen.add(key)
        _cid(row["entry_cid"])
        _require(type(row["evidence_disposition"]) is str and row["evidence_disposition"] in {
            "matched_complete", "no_exact_indexed_conditional_evidence", "matched_partial", "unknown_budget"},
                 "explicit member evidence disposition required")
        dispositions[row["evidence_disposition"]] += 1
        ids = row["evidence_entry_ids"]
        _require(type(ids) is list and len(ids) <= 4096 and all(type(identity) is str for identity in ids)
                 and len(set(ids)) == len(ids),
                 "bounded unique member evidence identities required")
        for identity in ids:
            _cid(identity)
            _require(identity not in evidence_ids, "evidence entry belongs to multiple source members")
            evidence_ids.add(identity)
        _require(bool(ids) is row["evidence_disposition"].startswith("matched_"),
                 "evidence disposition differs from exact member evidence identities")
        ordered.append({"source_key": key, "entry_cid": row["entry_cid"]})
    _require(cid_for_structured(ordered) == value["membership_cid"], "complete ordered membership identity differs")
    _require(coverage["evidence_entries"] == len(evidence_ids)
             and coverage["evidence_matched_members"] == dispositions["matched_complete"] + dispositions["matched_partial"]
             and coverage["evidence_complete_absent_members"] == dispositions["no_exact_indexed_conditional_evidence"]
             and coverage["evidence_unknown_members"] == dispositions["unknown_budget"],
             "coverage counts differ from complete member dispositions")
    query = value["query"]
    _require(type(query) is dict and set(query) == {"selector_cid", "inventory_cid", "epoch", "complete", "next_cursor", "page_cids"},
             "complete sealed advisory query identity required")
    for name in ("selector_cid", "inventory_cid"):
        _cid(query[name])
    _require(type(query["epoch"]) is int and 1 <= query["epoch"] < 2**63
             and type(query["complete"]) is bool, "sealed query epoch or completeness differs")
    _require(type(query["page_cids"]) is list and len(query["page_cids"]) <= 256,
             "bounded consumed query page ledger required")
    _require((query["complete"] and query["next_cursor"] is None and query["page_cids"])
             or (not query["complete"] and (query["next_cursor"] is not None or not query["page_cids"])),
             "sealed query completeness or bounded continuation differs")
    _require((query["complete"] and not dispositions["matched_partial"] and not dispositions["unknown_budget"])
             or (not query["complete"] and not dispositions["matched_complete"]
                 and not dispositions["no_exact_indexed_conditional_evidence"]),
             "incomplete evidence traversal cannot establish exact absence")
    if query["next_cursor"] is not None:
        from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationQueryCursor
        try:
            cursor = CodebaseVerificationQueryCursor.from_dict(query["next_cursor"])
        except (ValueError, TypeError, KeyError) as exc:
            raise CodebaseInventoryEvidenceContextError("native sealed continuation required") from exc
        _require(cursor.head_cid == value["head_cid"] and cursor.inventory_cid == query["inventory_cid"]
                 and cursor.selector_cid == query["selector_cid"] and cursor.epoch == query["epoch"],
                 "query continuation lost its inventory binding")
    for identity in query["page_cids"]:
        _cid(identity)
    return value


def _authored_materials(request: PlanCreateRequest, materials: PlanCreateMaterials):
    _require(type(request) is PlanCreateRequest and type(materials) is PlanCreateMaterials
             and type(materials.intent) is TypedIntent, "exact independently authored planning inputs required")
    _require(type(materials.extra) is dict and MATERIAL_KEY not in materials.extra
             and MATERIAL_KEY not in materials.candidate_context, "reserved inventory material key collision")
    _require(request.budget.max_model_calls == 0 and not materials.current_facts
             and materials.model_provider is None, "model-off planning with zero observed facts required")
    _require(materials.scan is None and materials.current_roots is None
             and materials.obligation_graph is None and materials.evidence_bundle is None
             and materials.admission_materials is None and not materials.evidence_adapters
             and not materials.evidence_queries and materials.parallel_request is None
             and materials.parallel_tasks is None, "structural preview cannot accept facts, static roots or injected stages")
    _require(type(materials.producers) is tuple and type(materials.task_candidates) is tuple
             and all(type(item) is ProducerRule for item in materials.producers)
             and all(type(item) is TaskCandidate for item in materials.task_candidates),
             "immutable independently authored producer and task declarations required")
    requirements = list(materials.intent.goal_predicate_ids)
    tasks = [item.candidate_id for item in materials.task_candidates]
    _require(0 < len(requirements) <= 16 and 0 < len(tasks) <= 16
             and len(set(requirements)) == len(requirements) and len(set(tasks)) == len(tasks),
             "bounded complete requirement and task population required")
    snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
    _require(snapshot.material_binding.get("reuse_supported") is True,
             "all authored planning inputs must have complete semantic bindings")
    return requirements, tasks, snapshot


def _check_preview(result: Any, request: PlanCreateRequest, materials: PlanCreateMaterials,
                   refs: dict[str, Any]) -> dict[str, Any]:
    """Replay native receipt identities and the exact injected material binding."""
    _require(type(result) is dict and result.get("schema") == "supervisor-repository-plan-preview@1",
             "native repository preview result required")
    _require(type(result.get("observed_facts_supplied")) is int and result["observed_facts_supplied"] == 0
             and type(result.get("model_calls")) is int and result["model_calls"] == 0
             and all(result.get(name) is False for name in (
                 "source_semantics_verified", "proof_authority", "production_admitted",
                 "worker_launched", "execution_authority", "completion_authority")),
             "repository preview acquired non-advisory authority")
    preview_raw, snapshot_raw = result.get("preview"), result.get("input_snapshot")
    _require(type(preview_raw) is dict and preview_raw.get("read_only") is True
             and preview_raw.get("wrote_effects") == [] and type(snapshot_raw) is dict,
             "read-only native preview receipt required")
    try:
        receipt = PlanCreatePreviewReceipt.from_dict(preview_raw)
        snapshot = PlanCreateInputSnapshot.from_dict(snapshot_raw)
    except (ValueError, TypeError, KeyError, PlanCreateServiceError) as exc:
        raise CodebaseInventoryEvidenceContextError("native preview receipt or snapshot identity differs") from exc
    context = result.get("structural_context")
    expected_context_head = {name: value for name, value in refs["head"].items() if name != "schema"}
    _require(type(context) is dict and _wire(context.get("head")) == _wire(expected_context_head)
             and result.get("structural_context_cid") == cid_for_structured(context),
             "repository preview structural context differs")
    from .structural_codebase_context import StructuralCodebaseContext
    try:
        native_context = StructuralCodebaseContext(CodebaseHead.from_dict(refs["head"]),
            context.get("semantic_state_cid"), context.get("coverage"))
    except (ValueError, TypeError) as exc:
        raise CodebaseInventoryEvidenceContextError("native structural context inventory differs") from exc
    _require(_wire(native_context.to_dict()) == _wire(context)
             and native_context.semantic_state_cid == request.roots.program_root,
             "repository preview context lost its structural authority ceiling")
    bound = replace(materials,
        scan={"scan_cid": result["structural_context_cid"], "structural_codebase": context},
        candidate_context={**materials.candidate_context,
            "structural_codebase_context_cid": result["structural_context_cid"], "structural_codebase": context},
        extra={**materials.extra, "structural_codebase_context_cid": result["structural_context_cid"],
            "structural_codebase": context, "root_observation_profile": "repository-live-root-observation@1"})
    expected = freeze_plan_create_input_snapshot(request, materials=bound)
    _require(snapshot == expected and receipt.input_snapshot_cid == expected.snapshot_cid
             and receipt.request_cid == request.request_cid and receipt.roots == request.roots
             and receipt.mode is PlanCreateMode.DETERMINISTIC and receipt.admitted is False,
             "native preview did not consume the complete inventory material")
    return result


def _producer_pin() -> dict[str, str]:
    global _IMPORTED_SOURCE_PIN
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    _require(len(raw) <= 256 * 1024, "bounded inventory consumer implementation required")
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    codes = {item.co_name: item for item in compiled.co_consts if type(item) is CodeType}
    for name, value in globals().copy().items():
        if type(value) is FunctionType and value.__module__ == __name__:
            _require(codes.get(name) == value.__code__, "loaded inventory consumer differs from installed source")
    digest = hashlib.sha256(raw).hexdigest()
    _require(_IMPORTED_SOURCE_PIN is None or _IMPORTED_SOURCE_PIN == digest,
             "inventory consumer implementation generation changed")
    _IMPORTED_SOURCE_PIN = digest
    return {"module": __name__, "sha256": digest}


def preview_current_inventory_evidence_plan(record, index, repository, *, verification_catalog,
        registry, materials: PlanCreateMaterials, request: PlanCreateRequest, policy_observer,
        scheduler=None, parent_lease=None, cancel_event=None, admission_timeout_seconds=30.0,
        timeout_seconds=120.0, memory_mb=1024) -> dict[str, Any]:
    """Validate current advisory evidence around a native model-off preview.

    One admission-inclusive cooperative deadline and one real datasets parent
    reservation cover both current validations and planning. Same-head evidence
    changes during callbacks invalidate the result at the closing validation.
    No fit, inference, solver, source execution or worker dispatch occurs here.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_inventory_evidence import (
        CodebaseInventoryEvidenceRecord, validate_current_inventory_evidence,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    _require(type(record) is CodebaseInventoryEvidenceRecord and callable(policy_observer),
             "exact native inventory evidence record and live policy observer required")
    _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 600, "bounded finite overall deadline required")
    _require(type(admission_timeout_seconds) in {int, float} and math.isfinite(admission_timeout_seconds)
             and admission_timeout_seconds >= 0, "finite nonnegative admission timeout required")
    _require(type(memory_mb) is int and 1024 <= memory_mb <= 4096, "bounded native inventory memory reservation required")
    _require(cancel_event is None or callable(getattr(cancel_event, "is_set", None)), "native cancellation signal required")
    started = time.monotonic()
    requirements, tasks, authored_snapshot = _authored_materials(request, materials)
    deadline = started + min(timeout_seconds, request.budget.max_latency_ms / 1000)
    original_bytes, artifact_cid = record._payload, record.artifact_cid
    refs_value = record.advisory_refs()
    head = CodebaseHead.from_dict(refs_value["head"])
    refs = _validate_refs(refs_value, artifact_cid=artifact_cid, head=head)
    refs_bytes = _wire(refs)
    refs_cid = cid_for_structured(refs)
    bound_materials = replace(materials, extra={**materials.extra, MATERIAL_KEY: refs})
    producer = _producer_pin()

    def remaining(signal=None):
        if (signal is not None and signal.is_set()) or (cancel_event is not None and cancel_event.is_set()):
            raise LeaseCancelledError("inventory evidence planning cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("inventory evidence planning deadline exceeded")
        return left

    def unchanged():
        _require(record._payload == original_bytes and record.artifact_cid == artifact_cid
                 and _wire(refs) == refs_bytes and _wire(record.advisory_refs()) == refs_bytes,
                 "immutable inventory material changed during planning")
        _require(freeze_plan_create_input_snapshot(request, materials=materials) == authored_snapshot,
                 "independently authored planning inputs changed during callbacks")
        _require(_producer_pin() == producer, "inventory consumer implementation changed during planning")

    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
            timeout_seconds=min(admission_timeout_seconds, remaining()), memory_mb=memory_mb) as lease:
        signal = lease.combined_cancellation_signal(cancel_event)

        def current():
            validated = validate_current_inventory_evidence(record, index, repository,
                verification_catalog=verification_catalog, registry=registry, expected_head=head,
                parent_lease=lease, cancel_event=signal,
                admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                timeout_seconds=remaining(signal), memory_mb=memory_mb)
            _require(type(validated) is CodebaseInventoryEvidenceRecord
                     and validated.artifact_cid == artifact_cid and validated._payload == original_bytes,
                     "current validation selected different inventory evidence")
            unchanged()
            remaining(signal)

        current()
        owner = RepositoryPlanPreviewOwner(index=index, repository=Path(repository), expected_head=head,
            parent_lease=lease, cancel_event=signal, timeout_seconds=min(90.0, remaining(signal)), memory_mb=memory_mb)

        def observe_policy(typed_request):
            _require(typed_request == request, "inventory preview changed the independently authored request")
            unchanged()
            remaining(signal)
            roots = policy_observer(typed_request)
            unchanged()
            remaining(signal)
            return roots

        preview = preview_repository_plan(owner=owner, request=request, materials=bound_materials,
                                          policy_observer=observe_policy)
        unchanged()
        _check_preview(preview, request, bound_materials, refs)
        current()
        result = {"schema": SCHEMA, MATERIAL_KEY: refs, "codebase_inventory_evidence_cid": refs_cid,
            "repository_preview": preview, "declared_requirement_ids": requirements, "declared_task_ids": tasks,
            "residual_requirements": [{"predicate_id": identity, "status": "runtime_behavior_unresolved"}
                                      for identity in requirements],
            "current_facts": [], "removed_task_ids": [], "training_steps": 0, "inference_calls": 0,
            "authority": dict(_FALSE), "producer": producer}
        _wire(result)
        result["result_cid"] = cid_for_structured(result)
        remaining(signal)
    remaining()
    return result


__all__ = ["SCHEMA", "MATERIAL_KEY", "CodebaseInventoryEvidenceContextError",
           "preview_current_inventory_evidence_plan"]
