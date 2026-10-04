"""Fresh structural source successors in independently authored planning inputs.

The complete source delta is advisory metadata. It creates no observed facts,
removes no tasks, and does not make predecessor proofs or learned rows eligible
under a successor head. Source publication, model selection, inference and
proof checking remain separate explicit operations. Receiving is sequential;
this adapter does not lock the checkout or attest process execution.
"""
from __future__ import annotations

from dataclasses import fields, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import CodeType, FunctionType, MappingProxyType
from typing import Any

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor import (
    CodebaseSourceDeltaRecord, validate_current_codebase_source_delta,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from ..prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreateMaterials, PlanCreateMode,
    PlanCreatePreviewReceipt, PlanCreateServiceError,
    freeze_plan_create_input_snapshot,
)
from .adaptive_planner import FrozenPlanningGoal
from .obligation_graph_compiler import (
    InvalidationSelector, InvalidationSelectorKind, PredicatePolarity, ProducerRule,
    SemanticSupport, TaskCandidate, TypedIntent, TypedPredicate,
)
from .plan_evaluator import EvidenceAwarePlanPolicy
from .plan_revision_contracts import (
    DirtyTreePolicy, FallbackPolicy, PlanAuthorityRoots, PlanCreateRequest,
    PlanRequestBudget, TaskSourceKind,
)
from .repository_plan_preview import RepositoryPlanPreviewOwner, preview_repository_plan
from .structural_codebase_context import StructuralCodebaseContext

SCHEMA = "supervisor-codebase-source-delta-plan-preview@1"
REFS_SCHEMA = "codebase-source-delta-advisory-refs@1"
MATERIAL_KEY = "codebase_source_delta_advisory"
MAX_MATERIAL_BYTES = 4 * 1024 * 1024
_IMPORTED_SOURCE_PIN: str | None = None
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "authoritative_cache_eligible", "behavioral_satisfaction",
    "training_executed", "decoded_formulas_generated", "repository_code_executed",
    "source_execution_attested", "scan_execution_attested",
)}
_REF_FIELDS = ("previous_head", "current_head", "previous_membership_cid",
    "current_membership_cid", "capture_policy", "coverage", "authority",
    "numerical_reuse", "model_advanced", "removal_scope", "physical_absence_verified")
_GUARD_RECORDS = frozenset({PlanCreateMaterials, PlanCreateRequest, PlanAuthorityRoots,
    PlanRequestBudget, FrozenPlanningGoal, EvidenceAwarePlanPolicy, TypedIntent,
    TypedPredicate, ProducerRule, TaskCandidate, InvalidationSelector})
_GUARD_ENUMS = frozenset({DirtyTreePolicy, FallbackPolicy, TaskSourceKind,
    PredicatePolarity, SemanticSupport, InvalidationSelectorKind})


class CodebaseSourceDeltaContextError(ValueError):
    """A source successor cannot supply this current proposal-only preview."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CodebaseSourceDeltaContextError(message)


def _wire(value: Any) -> bytes:
    """Bound plain, float-free material without invoking record converters."""
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
        raise CodebaseSourceDeltaContextError("canonical inert material JSON required") from exc
    _require(len(raw) <= MAX_MATERIAL_BYTES, "material byte bound exceeded")
    return raw


def _advisory_refs(record: CodebaseSourceDeltaRecord) -> dict[str, Any]:
    """Project complete body-free refs; intrinsic integrity is not freshness."""
    _require(type(record) is CodebaseSourceDeltaRecord, "exact native source delta required")
    # Reconstruct the native closed protocol instead of trusting detached caller
    # dictionaries or a custom to_dict method. This does not open an owner.
    sealed = CodebaseSourceDeltaRecord(record.artifact_cid, record._payload)
    value = json.loads(sealed._payload)
    refs = {"schema": REFS_SCHEMA, "artifact_cid": sealed.artifact_cid,
            **{name: value[name] for name in _REF_FIELDS}, "ledger": []}
    for row in value["ledger"]:
        refs["ledger"].append({
            **{name: row[name] for name in ("source_key", "classification",
                "source_bytes_comparison", "ast_identity_comparison")},
            "previous": None if row["previous"] is None else row["previous"]["member"],
            "current": None if row["current"] is None else row["current"]["member"],
        })
    _require(set(refs["authority"]) == set(_FALSE)
             and all(flag is False for flag in refs["authority"].values()),
             "source delta acquired non-advisory authority")
    return json.loads(_wire(refs))


def _plain_state(value: Any) -> bytes:
    """Read bounded native data without converters or arbitrary mapping hooks.

    This closing mutation guard deliberately admits a smaller input vocabulary
    than generic semantic-material conversion. Native frozen mappings must
    actually wrap a plain dict. Looking through the proxy with GC introspection
    avoids invoking a foreign backing mapping after the final source gate.
    Finite input floats are represented by exact hexadecimal strings only in
    this private guard, not in the public source or proof identity protocol.
    """
    nodes, text_bytes = 0, 0

    def visit(item, depth):
        nonlocal nodes, text_bytes
        nodes += 1
        _require(nodes <= 100_000 and depth <= 32, "bounded plain planning state required")
        kind = type(item)
        if kind is str:
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
            out = ["str", item]
        elif kind is int:
            _require(item.bit_length() <= 128, "bounded plain planning integer required")
            out = ["int", item]
        elif kind is bool or item is None:
            out = ["bool" if kind is bool else "null", item]
        elif kind is float:
            _require(math.isfinite(item), "finite plain planning input number required")
            out = ["float_hex", item.hex()]
        elif kind in {dict, MappingProxyType}:
            mapping = item
            if kind is MappingProxyType:
                referents = gc.get_referents(item)
                _require(len(referents) == 1 and type(referents[0]) is dict,
                         "native frozen mapping must wrap a plain dict")
                mapping = referents[0]
            _require(len(mapping) <= 100_000 - nodes and all(type(key) is str for key in dict.keys(mapping)),
                     "plain planning mapping keys required")
            keys = sorted(dict.keys(mapping))
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in keys)
            out = ["dict" if kind is dict else "frozen_dict",
                   [[key, visit(dict.__getitem__(mapping, key), depth + 1)] for key in keys]]
        elif kind in {list, tuple}:
            _require(len(item) <= 100_000 - nodes, "plain planning collection bound exceeded")
            out = ["list" if kind is list else "tuple", [visit(child, depth + 1) for child in item]]
        elif kind in _GUARD_ENUMS:
            out = ["enum", kind.__module__ + "." + kind.__qualname__,
                   visit(object.__getattribute__(item, "_value_"), depth + 1)]
        elif kind in _GUARD_RECORDS:
            # These exact native dataclasses use stored fields, not properties.
            state = object.__getattribute__(item, "__dict__")
            _require(type(state) is dict, "plain native planning record state required")
            names = {field.name for field in fields(kind)}
            _require(set(dict.keys(state)) == names, "native planning record has foreign or missing state")
            out = ["record", kind.__module__ + "." + kind.__qualname__,
                   [[name, visit(dict.__getitem__(state, name), depth + 1)] for name in sorted(names)]]
        else:
            raise CodebaseSourceDeltaContextError("unsupported foreign type in plain planning state")
        _require(text_bytes <= 4 * 1024 * 1024, "plain planning state text bound exceeded")
        return out

    raw = json.dumps(visit(value, 0), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    _require(len(raw) <= 16 * 1024 * 1024, "plain planning state byte bound exceeded")
    return raw


def _authored_inputs(request: PlanCreateRequest, materials: PlanCreateMaterials):
    _require(type(request) is PlanCreateRequest and type(materials) is PlanCreateMaterials
             and type(materials.intent) is TypedIntent,
             "exact independently authored planning inputs required")
    _require(type(materials.extra) is dict and type(materials.candidate_context) is dict
             and MATERIAL_KEY not in materials.extra and MATERIAL_KEY not in materials.candidate_context,
             "reserved source delta material key collision")
    _require(request.budget.max_model_calls == 0 and not materials.current_facts
             and materials.model_provider is None,
             "model-off planning with zero observed facts required")
    _require(materials.scan is None and materials.current_roots is None
             and materials.obligation_graph is None and materials.evidence_bundle is None
             and materials.admission_materials is None and not materials.evidence_adapters
             and not materials.evidence_queries and materials.parallel_request is None
             and materials.parallel_tasks is None and materials.query_plan is None
             and materials.workflow_request is None,
             "source delta preview cannot accept facts, static roots or injected stages")
    _require(type(materials.producers) is tuple and type(materials.task_candidates) is tuple
             and type(materials.predicates) is tuple
             and all(type(item) is ProducerRule for item in materials.producers)
             and all(type(item) is TaskCandidate for item in materials.task_candidates)
             and all(type(item) is TypedPredicate for item in materials.predicates),
             "immutable independently authored producer, task and predicate declarations required")
    requirements = list(materials.intent.goal_predicate_ids)
    tasks = [item.candidate_id for item in materials.task_candidates]
    _require(0 < len(requirements) <= 16 and 0 < len(tasks) <= 16
             and len(materials.producers) <= 32 and len(materials.predicates) <= 32
             and len(set(requirements)) == len(requirements) and len(set(tasks)) == len(tasks),
             "bounded complete requirement and task population required")
    snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
    _require(snapshot.material_binding.get("reuse_supported") is True,
             "all authored planning inputs must have complete semantic bindings")
    return requirements, tasks, snapshot


def _check_preview(result, request, materials, head):
    """Replay the exact native preview and its consumed material snapshot."""
    fields = {"schema", "preview", "input_snapshot", "structural_context", "structural_context_cid",
              "observed_facts_supplied", "model_calls", "source_semantics_verified", "proof_authority",
              "production_admitted", "worker_launched", "execution_authority", "completion_authority"}
    _require(type(result) is dict and set(result) == fields
             and result["schema"] == "supervisor-repository-plan-preview@1",
             "closed native repository preview result required")
    _require(type(result["observed_facts_supplied"]) is int and result["observed_facts_supplied"] == 0
             and type(result["model_calls"]) is int and result["model_calls"] == 0
             and all(result[name] is False for name in (
                 "source_semantics_verified", "proof_authority", "production_admitted",
                 "worker_launched", "execution_authority", "completion_authority")),
             "repository preview acquired non-advisory authority")
    _require(type(result["preview"]) is dict and result["preview"].get("read_only") is True
             and result["preview"].get("wrote_effects") == []
             and type(result["input_snapshot"]) is dict,
             "read-only native preview receipt required")
    try:
        receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
        snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    except (ValueError, TypeError, KeyError, PlanCreateServiceError) as exc:
        raise CodebaseSourceDeltaContextError("native preview receipt or snapshot identity differs") from exc
    context = result["structural_context"]
    _require(type(context) is dict
             and _wire(context.get("head")) == _wire({name: value for name, value in head.to_dict().items() if name != "schema"})
             and result["structural_context_cid"] == cid_for_structured(context),
             "repository preview structural context differs")
    try:
        native_context = StructuralCodebaseContext(head, context.get("semantic_state_cid"), context.get("coverage"))
    except (ValueError, TypeError) as exc:
        raise CodebaseSourceDeltaContextError("native structural context inventory differs") from exc
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
             "native preview did not consume the complete source delta material")
    _wire(result)


def _producer_pin() -> dict[str, str]:
    """Bind this additive consumer without changing any inherited generation."""
    global _IMPORTED_SOURCE_PIN
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    _require(len(raw) <= 256 * 1024, "bounded source delta consumer implementation required")
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    codes = {item.co_name: item for item in compiled.co_consts if type(item) is CodeType}
    for name, value in globals().copy().items():
        if type(value) is FunctionType and value.__module__ == __name__:
            _require(codes.get(name) == value.__code__, "loaded source delta consumer differs from installed source")
    digest = hashlib.sha256(raw).hexdigest()
    _require(_IMPORTED_SOURCE_PIN is None or _IMPORTED_SOURCE_PIN == digest,
             "source delta consumer implementation generation changed")
    _IMPORTED_SOURCE_PIN = digest
    return {"module": __name__, "sha256": digest,
            "scope": "listed_local_file_only_not_execution_attestation"}


def preview_current_source_delta_plan(record, index, repository, *,
        materials: PlanCreateMaterials, request: PlanCreateRequest, policy_observer,
        scheduler=None, parent_lease=None, cancel_event=None, admission_timeout_seconds=30.0,
        timeout_seconds=120.0, memory_mb=1024) -> dict[str, Any]:
    """Consume the complete current source successor in a model-off preview.

    One admission-inclusive cooperative deadline and real parent reservation
    cover receiving and planning. Result construction, record/material checks
    and consumer pin reads precede the final fresh receiver. This function never
    publishes a head, opens a model registry, checks proofs or launches workers.
    Every later planning or admission use requires a new receiving operation.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError

    started = time.monotonic()
    _require(type(record) is CodebaseSourceDeltaRecord and callable(policy_observer),
             "exact native source delta and live policy observer required")
    _require(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 600, "bounded finite overall deadline required")
    _require(type(admission_timeout_seconds) in {int, float} and math.isfinite(admission_timeout_seconds)
             and 0 <= admission_timeout_seconds <= 600, "bounded finite admission timeout required")
    _require(type(memory_mb) is int and 1024 <= memory_mb <= 4096,
             "bounded native source delta memory reservation required")
    _require(cancel_event is None or callable(getattr(cancel_event, "is_set", None)),
             "native cancellation signal required")
    requirements, tasks, authored_snapshot = _authored_inputs(request, materials)
    deadline = started + min(timeout_seconds, request.budget.max_latency_ms / 1000)
    original_bytes, artifact_cid = record._payload, record.artifact_cid
    refs = _advisory_refs(record)
    refs_bytes, refs_cid = _wire(refs), cid_for_structured(refs)
    head = CodebaseHead.from_dict(refs["current_head"])
    bound_materials = replace(materials, extra={**materials.extra, MATERIAL_KEY: refs})
    bound_snapshot = freeze_plan_create_input_snapshot(request, materials=bound_materials)
    producer = _producer_pin()
    authored_state = _plain_state([request, materials])

    def remaining(signal=None):
        if (signal is not None and signal.is_set()) or (cancel_event is not None and cancel_event.is_set()):
            raise LeaseCancelledError("source delta planning cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("source delta planning deadline exceeded")
        return left

    def unchanged():
        _require(record._payload == original_bytes and record.artifact_cid == artifact_cid
                 and _wire(refs) == refs_bytes and _wire(_advisory_refs(record)) == refs_bytes,
                 "immutable source delta material changed during planning")
        _require(freeze_plan_create_input_snapshot(request, materials=materials) == authored_snapshot
                 and freeze_plan_create_input_snapshot(request, materials=bound_materials) == bound_snapshot,
                 "independently authored planning inputs changed during callbacks")
        _require(_producer_pin() == producer, "source delta consumer implementation changed during planning")

    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
            timeout_seconds=min(admission_timeout_seconds, remaining()), memory_mb=memory_mb) as lease:
        signal = lease.combined_cancellation_signal(cancel_event)

        def current():
            validated = validate_current_codebase_source_delta(record, index, repository,
                parent_lease=lease, cancel_event=signal,
                admission_timeout_seconds=min(admission_timeout_seconds, remaining(signal)),
                timeout_seconds=remaining(signal), memory_mb=memory_mb)
            _require(type(validated) is CodebaseSourceDeltaRecord
                     and validated.artifact_cid == artifact_cid and validated._payload == original_bytes,
                     "current validation selected a different source delta")

        current()
        unchanged()
        owner = RepositoryPlanPreviewOwner(index=index, repository=Path(repository), expected_head=head,
            parent_lease=lease, cancel_event=signal, timeout_seconds=min(90.0, remaining(signal)), memory_mb=memory_mb)

        def observe_policy(typed_request):
            _require(typed_request == request, "source delta preview changed the independently authored request")
            unchanged()
            remaining(signal)
            roots = policy_observer(typed_request)
            unchanged()
            remaining(signal)
            return roots

        preview = preview_repository_plan(owner=owner, request=request, materials=bound_materials,
                                          policy_observer=observe_policy)
        _check_preview(preview, request, bound_materials, head)
        result = {"schema": SCHEMA, MATERIAL_KEY: refs, "source_delta_advisory_cid": refs_cid,
            "repository_preview": preview, "declared_requirement_ids": requirements, "declared_task_ids": tasks,
            "residual_requirements": [{"predicate_id": identity, "status": "runtime_behavior_unresolved"}
                                      for identity in requirements],
            "residual_task_ids": list(tasks), "current_facts": [], "removed_task_ids": [],
            "training_steps": 0, "inference_calls": 0, "source_property_solver_calls": 0,
            "numerical_reuse": False, "model_advanced": False, "physical_absence_verified": False,
            "production_admitted": False, "worker_launched": False,
            "authority": dict(_FALSE), "producer": producer}
        _wire(result)
        result["result_cid"] = cid_for_structured(result)
        # All native planner/policy and adapter record/material/producer reads
        # precede this last full receiving operation. No callback can turn a
        # catalog-head equality check into a replacement for fresh source/SQL.
        unchanged()
        remaining(signal)
        _require(_plain_state([request, materials]) == authored_state,
                 "independently authored plain planning inputs changed during callbacks")
        bound_state, result_state = _plain_state(bound_materials), _plain_state(result)
        current()
        # The final native receiver itself reads CAS and producer data. Those
        # reads may run controlled callbacks which alter previously consumed
        # materials or the result while leaving source intact. Check only plain
        # native data here: no converter, producer read or source-affecting
        # planning callback follows the last fresh receiving operation.
        _require(record.artifact_cid == artifact_cid and record._payload == original_bytes
                 and _plain_state([request, materials]) == authored_state
                 and _plain_state(bound_materials) == bound_state
                 and _plain_state(result) == result_state,
                 "plain planning material or result changed during final receiving")
    return result


__all__ = ["SCHEMA", "REFS_SCHEMA", "MATERIAL_KEY", "CodebaseSourceDeltaContextError",
           "preview_current_source_delta_plan"]
