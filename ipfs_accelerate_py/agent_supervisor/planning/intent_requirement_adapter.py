"""Checked administrative operations for atomic IntentIR requirement planning.

An operation maps an explicitly reviewed native atom to independently declared
task outputs and checks. Its symbolic effect means that a task is selected to
cover a requirement. It does not mean the requested software behavior already
exists, that source semantics are proved, or that executing the task succeeds.
The caller owns manifest signature, source observation, and execution policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from ..core.multiformats_identity import cid_for_dag_json
from ..proof.formal_verification_contracts import content_identity
from .adaptive_planner import FrozenPlanningGoal
from .obligation_graph_compiler import (
    InvalidationSelector, InvalidationSelectorKind, PredicatePolarity, ProducerRule,
    SemanticSupport, TaskCandidate, TypedIntent, TypedPredicate,
    obligation_id_for_producer,
)
from .plan_evaluator import EvidenceAwarePlanPolicy

SYMBOLIC_OPERATION_SCHEMA = "intent-symbolic-operation-contract@1"
INTENT_PLANNING_MATERIALS_SCHEMA = "intent-planning-materials@1"
INTERPRETATION_SCOPE = "administrative_requirement_task_coverage"
MAX_OPERATION_BYTES = 4 * 1024 * 1024
MAX_OPERATIONS = 256
MAX_MATCHERS = 4096
_AUTHORITY = {"semantic_alignment_verified": False, "proof_authority": False,
              "execution_authority": False, "completion_authority": False}


class IntentRequirementAdapterError(ValueError):
    """A declared operation cannot be checked against its independent inputs."""


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise IntentRequirementAdapterError("bounded JSON value required") from exc


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _json(value):
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 48 or count > 100_000:
            raise IntentRequirementAdapterError("operation JSON structure exceeds its bound")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise IntentRequirementAdapterError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) not in (str, int, bool, type(None)):
            raise IntentRequirementAdapterError("operation declarations need exact integer-only JSON types")
    raw = _wire(value)
    if len(raw) > MAX_OPERATION_BYTES:
        raise IntentRequirementAdapterError("operation JSON byte bound exceeded")
    return json.loads(raw)


def _object(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise IntentRequirementAdapterError(f"exact {name} fields required")
    return value


def _text(value, name, maximum=512):
    if (type(value) is not str or not value or value.strip() != value
            or len(value.encode("utf-8")) > maximum
            or any(char in value for char in "\r\n\0")):
        raise IntentRequirementAdapterError(f"bounded {name} required")
    return value


def _array(value, name, maximum=MAX_OPERATIONS):
    if type(value) is not list or len(value) > maximum:
        raise IntentRequirementAdapterError(f"bounded {name} array required")
    return value


def _strings(value, name):
    result = [_text(item, name) for item in _array(value, name)]
    if len(set(result)) != len(result):
        raise IntentRequirementAdapterError(f"duplicate {name}")
    return sorted(result)


def _path(value):
    value = _text(value, "source-relative path", 4096)
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or str(path) != value or value == "."
            or ".git" in path.parts or any(part.startswith(".runtime") for part in path.parts)):
        raise IntentRequirementAdapterError("canonical source-relative path required")
    return value


def _outputs(raw):
    result = []
    for item in _array(raw, "outputs"):
        _object(item, {"path", "effect", "media_type"}, "output")
        effect = _text(item["effect"], "output effect")
        if effect not in {"create", "write", "modify", "delete"}:
            raise IntentRequirementAdapterError("declared output effect required")
        result.append({"path": _path(item["path"]), "effect": effect,
                       "media_type": _text(item["media_type"], "media type")})
    if len({item["path"] for item in result}) != len(result):
        raise IntentRequirementAdapterError("duplicate output path")
    return sorted(result, key=lambda item: item["path"])


def _ledger_requirements(ledger):
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import validate_intent_requirement_ledger
    try:
        source_text = "".join(unit["text"] for unit in ledger["source_units"])
        validate_intent_requirement_ledger(ledger, source_text=source_text)
    except (TypeError, KeyError, ValueError) as exc:
        raise IntentRequirementAdapterError("invalid symbolic source ledger: " + str(exc)) from exc
    if any(unit["disposition"] == "unsupported" for unit in ledger["source_units"]):
        raise IntentRequirementAdapterError("symbolic planning does not support unresolved source units")
    requirements = {}
    for requirement in ledger["requirements"]:
        if (requirement["representation"] != "native_statement" or requirement["compound"] is not None
                or requirement["kind"] != "goal" or requirement["modality"] not in {"required", "intended"}):
            raise IntentRequirementAdapterError("symbolic planning supports atomic mandatory goal requirements only")
        requirements[requirement["requirement_id"]] = requirement
    if not requirements or len(requirements) > MAX_OPERATIONS:
        raise IntentRequirementAdapterError("bounded nonempty symbolic requirements required")
    return requirements


def _groundings(raw, requirement_ids):
    result = {}
    for item in _array(raw, "grounded requirements"):
        _object(item, {"requirement_id", "outputs", "validation_keys", "dependency_requirement_ids"}, "grounding")
        key = _text(item["requirement_id"], "requirement ID")
        if key not in requirement_ids or key in result:
            raise IntentRequirementAdapterError("unknown or duplicate grounded requirement")
        result[key] = {"requirement_id": key, "outputs": _outputs(item["outputs"]),
            "validation_keys": _strings(item["validation_keys"], "validation key"),
            "dependency_requirement_ids": _strings(item["dependency_requirement_ids"], "requirement dependency")}
    if set(result) != set(requirement_ids):
        raise IntentRequirementAdapterError("every symbolic requirement needs independent grounding")
    for key, item in result.items():
        if key in item["dependency_requirement_ids"] or not set(item["dependency_requirement_ids"]) <= set(result):
            raise IntentRequirementAdapterError("unknown or self requirement dependency")
    return result


def _native_statement(ledger, requirement):
    from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir

    report = ledger["source_report"]
    while report["schema"] == "source-document-autoencoder/v1":
        report = report.get("rich_intent") or report["intent"]
    candidate = next(item for item in report["candidates"]
                     if item["unit_id"] == requirement["source_unit_id"])
    if "candidate_intent_ir" in candidate:
        document = candidate["candidate_intent_ir"]
    else:
        unit = next(item for item in report["units"] if item["unit_id"] == candidate["unit_id"])
        document = unit["inference"]["logic"]["native_intent_ir"]
    native = decode_intent_ir(document)
    digest = hashlib.sha256(canonical_intent_ir_bytes(native)).hexdigest()
    if digest != requirement["native_document_sha256"] or len(requirement["statement_ids"]) != 1:
        raise IntentRequirementAdapterError("native requirement document identity differs")
    statement = next(item for item in native.statements if item.statement_id == requirement["statement_ids"][0])
    return {"requirement_id": requirement["requirement_id"], "native_document_sha256": digest,
        "statement_id": statement.statement_id, "predicate": statement.predicate,
        "arguments": list(statement.arguments), "modality": statement.modality.value}


def validate_symbolic_operations(symbolic, *, ledger, requirements):
    """Check exact native matchers and independent groundings, without a manifest.

    This function deliberately does not call the enclosing requirement-contract
    validator: that validator uses this function to check its v2 extension.
    """
    value = _json(_plain(symbolic))
    _object(value, {"schema", "ledger_sha256", "review_ref", "interpretation_scope", "operations", *_AUTHORITY},
            "symbolic operation contract")
    if (value["schema"] != SYMBOLIC_OPERATION_SCHEMA or value["interpretation_scope"] != INTERPRETATION_SCOPE
            or value["ledger_sha256"] != ledger.get("ledger_sha256")):
        raise IntentRequirementAdapterError("symbolic operation schema, interpretation scope or ledger differs")
    _text(value["review_ref"], "explicit operation review reference")
    if any(value[key] is not False for key in _AUTHORITY):
        raise IntentRequirementAdapterError("symbolic operations grant no semantic or execution authority")
    reqs = _ledger_requirements(ledger)
    groundings = _groundings(requirements, reqs)
    native_matchers = {key: _native_statement(ledger, requirement) for key, requirement in reqs.items()}
    operations, by_id, by_task, req_ops = [], {}, {}, {key: set() for key in reqs}
    matcher_count = 0
    for item in _array(value["operations"], "operations"):
        _object(item, {"operation_id", "task_key", "matchers", "outputs", "validation_keys",
                       "dependency_operation_ids"}, "operation")
        op_id, task_key = _text(item["operation_id"], "operation ID"), _text(item["task_key"], "task key")
        if op_id in by_id or task_key in by_task:
            raise IntentRequirementAdapterError("one unique operation per unique task key required")
        matchers, matched = [], set()
        for matcher in _array(item["matchers"], "native matchers"):
            _object(matcher, {"requirement_id", "native_document_sha256", "statement_id",
                              "predicate", "arguments", "modality"}, "native matcher")
            key = _text(matcher["requirement_id"], "matcher requirement ID")
            if key not in native_matchers or key in matched or matcher != native_matchers[key]:
                raise IntentRequirementAdapterError("operation matcher must equal its exact native atom")
            if req_ops[key]:
                raise IntentRequirementAdapterError("one unique operation per requirement required")
            if not matcher["predicate"]:
                raise IntentRequirementAdapterError("opaque native text cannot supply an operation predicate")
            matched.add(key)
            matchers.append(matcher)
        matcher_count += len(matchers)
        if not matchers or matcher_count > MAX_MATCHERS:
            raise IntentRequirementAdapterError("bounded nonempty native operation matchers required")
        outputs = _outputs(item["outputs"])
        expected_outputs = {row["path"]: row for key in matched for row in groundings[key]["outputs"]}
        if any(expected_outputs[row["path"]] != row for key in matched for row in groundings[key]["outputs"]):
            raise IntentRequirementAdapterError("groundings assign conflicting effects to an output path")
        expected_validations = sorted({key_ for key in matched for key_ in groundings[key]["validation_keys"]})
        validation_keys = _strings(item["validation_keys"], "operation validation key")
        if outputs != sorted(expected_outputs.values(), key=lambda row: row["path"]) or validation_keys != expected_validations:
            raise IntentRequirementAdapterError("operation outputs and checks must equal its reviewed groundings")
        if not outputs or not validation_keys:
            raise IntentRequirementAdapterError("symbolic operations need explicit outputs and validation checks")
        operation = {"operation_id": op_id, "task_key": task_key,
            "matchers": sorted(matchers, key=lambda row: row["requirement_id"]), "outputs": outputs,
            "validation_keys": validation_keys,
            "dependency_operation_ids": _strings(item["dependency_operation_ids"], "operation dependency")}
        operations.append(operation)
        by_id[op_id], by_task[task_key] = operation, op_id
        for key in matched:
            req_ops[key].add(op_id)
    if not operations or any(not ids for ids in req_ops.values()):
        raise IntentRequirementAdapterError("every symbolic requirement must have a declared operation")
    for operation in operations:
        op_id = operation["operation_id"]
        dependencies = set(operation["dependency_operation_ids"])
        if op_id in dependencies or not dependencies <= set(by_id):
            raise IntentRequirementAdapterError("unknown or self operation dependency")
        required_operations = {required_op for matcher in operation["matchers"]
            for key in groundings[matcher["requirement_id"]]["dependency_requirement_ids"]
            for required_op in req_ops[key]}
        if not required_operations <= dependencies:
            raise IntentRequirementAdapterError("operation drops grounded requirement ordering")
    active, visited = set(), set()
    def visit(key):
        if key in active:
            raise IntentRequirementAdapterError("operation dependency cycle")
        if key not in visited:
            active.add(key)
            for dependency in by_id[key]["dependency_operation_ids"]:
                visit(dependency)
            active.remove(key)
            visited.add(key)
    for key in by_id:
        visit(key)
    value["operations"] = sorted(operations, key=lambda item: item["operation_id"])
    return value


@dataclass(frozen=True)
class IntentPlanningMaterials:
    intent: TypedIntent
    producers: tuple[ProducerRule, ...]
    task_candidates: tuple[TaskCandidate, ...]
    predicates: tuple[TypedPredicate, ...]
    current_facts: tuple
    frozen_goal: FrozenPlanningGoal
    candidate_context: Mapping[str, Any]
    requirements: tuple[Mapping[str, Any], ...]
    operations: tuple[Mapping[str, Any], ...]
    operation_contract: Mapping[str, Any]
    operation_candidate_ids: Mapping[str, str]
    candidate_task_keys: Mapping[str, str]
    candidate_requirement_ids: Mapping[str, tuple[str, ...]]
    contract_cid: str
    manifest_cid: str
    current_root_id: str

    def to_dict(self):
        value = {"schema": INTENT_PLANNING_MATERIALS_SCHEMA,
            "contract_cid": self.contract_cid, "manifest_cid": self.manifest_cid,
            "current_root_id": self.current_root_id,
            "operation_contract_cid": cid_for_dag_json(_plain(self.operation_contract)),
            "intent": self.intent.to_dict(), "predicates": [item.to_dict() for item in self.predicates],
            "producers": [item.to_dict() for item in self.producers],
            "task_candidates": [item.to_dict() for item in self.task_candidates], "current_facts": [],
            "frozen_goal": self.frozen_goal.to_dict(), "candidate_context": _plain(self.candidate_context),
            "operation_candidate_ids": _plain(self.operation_candidate_ids),
            "candidate_task_keys": _plain(self.candidate_task_keys),
            "candidate_requirement_ids": _plain(self.candidate_requirement_ids),
            "requirement_ids": [row["requirement_id"] for row in self.requirements],
            "interpretation_scope": INTERPRETATION_SCOPE, "source_semantics_verified": False,
            "observed_source_facts": False, **_AUTHORITY}
        value["materials_cid"] = cid_for_dag_json(value)
        return value


def build_intent_planning_materials(contract, *, manifest):
    """Build pure proposal-tier compiler inputs from a supplied manifest envelope.

    The caller verifies the manifest's signature, source observations, profile,
    and any publication transition. This adapter binds the declared baseline
    and inert contract; it never observes the filesystem or creates facts.
    """
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract

    canonical = validate_intent_requirement_contract(contract)
    if "symbolic_operations" not in canonical:
        raise IntentRequirementAdapterError("symbolic planning requires an explicit v2 operation contract")
    if not isinstance(manifest, Mapping):
        raise IntentRequirementAdapterError("declared manifest mapping required")
    payload = manifest.get("payload", manifest)
    if not isinstance(payload, Mapping) or payload.get("schema") != "supervisor-local-benchmark-manifest@4":
        raise IntentRequirementAdapterError("local v4 intent manifest required")
    artifact = payload.get("intent_requirements")
    if (not isinstance(artifact, Mapping) or set(artifact) != {"schema", "contract_json", "contract_cid"}
            or artifact["schema"] != "supervisor-local-intent-requirement-artifact@1"
            or artifact["contract_json"] != _wire(canonical).decode("utf-8")
            or artifact["contract_cid"] != cid_for_dag_json(canonical)):
        raise IntentRequirementAdapterError("operation contract differs from signed manifest requirements")
    symbolic = validate_symbolic_operations(canonical["symbolic_operations"], ledger=canonical["ledger"],
                                           requirements=canonical["requirements"])
    specs = {}
    for spec in _array(payload.get("tasks"), "manifest tasks"):
        if type(spec) is not dict:
            raise IntentRequirementAdapterError("declared manifest task object required")
        key = _text(spec.get("task_key"), "manifest task key")
        if key in specs:
            raise IntentRequirementAdapterError("duplicate manifest task key")
        specs[key] = spec
    operations = symbolic["operations"]
    if set(specs) != {item["task_key"] for item in operations}:
        raise IntentRequirementAdapterError("operation task population differs from signed manifest")
    op_by_id = {item["operation_id"]: item for item in operations}
    for operation in operations:
        spec = specs[operation["task_key"]]
        validation_keys = sorted(_text(row.get("validation_key"), "manifest validation key")
                                 for row in _array(spec.get("validations"), "manifest validations"))
        acceptance_keys = {key for row in _array(spec.get("acceptance"), "manifest acceptance")
                           for key in _strings(row.get("validation_keys"), "acceptance validation key")}
        dependencies = _strings(spec.get("dependencies"), "manifest dependency")
        expected_dependencies = sorted(op_by_id[key]["task_key"] for key in operation["dependency_operation_ids"])
        if (operation["outputs"] != _outputs(spec.get("outputs"))
                or operation["validation_keys"] != validation_keys
                or acceptance_keys != set(validation_keys)
                or dependencies != expected_dependencies):
            raise IntentRequirementAdapterError("operation changes signed task output, validation, acceptance or dependency")

    contract_cid, manifest_cid = cid_for_dag_json(canonical), cid_for_dag_json(_plain(manifest))
    current_root_id = content_identity({"schema": "supervisor-local-source-tree@1", "sources": payload["sources"]})
    operation_cid = cid_for_dag_json(symbolic)
    reqs = {item["requirement_id"]: item for item in canonical["ledger"]["requirements"]}
    groundings = {item["requirement_id"]: item for item in canonical["requirements"]}
    selectors = (InvalidationSelector(selector_id="selector:intent-planning-baseline",
        kind=InvalidationSelectorKind.ROOT, value_ref=current_root_id,
        provenance_refs=(manifest_cid,)),)
    predicates = tuple(TypedPredicate(
        predicate_id="predicate:" + hashlib.sha256(_wire({"contract": contract_cid, "requirement": key})).hexdigest(),
        predicate_type=INTERPRETATION_SCOPE, subject_ref=contract_cid, object_ref=key,
        polarity=PredicatePolarity.POSITIVE, support=SemanticSupport.REVIEWED,
        provenance_refs=(operation_cid, "sha256:" + canonical["ledger"]["ledger_sha256"]),
        invalidation_selectors=selectors, validation_requirement_refs=tuple(groundings[key]["validation_keys"]))
        for key in sorted(reqs))
    predicate_by_req = {predicate.object_ref: predicate for predicate in predicates}
    intent = TypedIntent(intent_id="intent:" + contract_cid, desired_predicates=predicates,
        source_refs=(contract_cid, operation_cid), current_root_id=current_root_id,
        metadata={"interpretation_scope": INTERPRETATION_SCOPE, "source_semantics_verified": False,
                  "observed_source_facts": False, **_AUTHORITY})
    producer_ids = {item["operation_id"]: "producer:" + cid_for_dag_json(item) for item in operations}
    candidate_ids = {item["operation_id"]: "candidate:" + cid_for_dag_json(item) for item in operations}
    producers, candidates, task_metadata, task_keys, candidate_reqs = [], [], {}, {}, {}
    for operation in operations:
        op_id, task_key = operation["operation_id"], operation["task_key"]
        matched = sorted(matcher["requirement_id"] for matcher in operation["matchers"])
        effects = tuple(predicate_by_req[key].predicate_id for key in matched)
        required = tuple(sorted({predicate_by_req[matcher["requirement_id"]].predicate_id
            for dep in operation["dependency_operation_ids"] for matcher in op_by_id[dep]["matchers"]}))
        producer = ProducerRule(producer_id=producer_ids[op_id], effect_predicate_ids=effects,
            required_predicate_ids=required, provenance_refs=(operation_cid, manifest_cid),
            invalidation_selectors=selectors, validation_requirement_refs=tuple(operation["validation_keys"]),
            task_candidate_ids=(candidate_ids[op_id],), executable=True)
        producers.append(producer)
        candidates.append(TaskCandidate(candidate_id=candidate_ids[op_id],
            closes_obligation_ids=tuple(obligation_id_for_producer(producer.producer_id, key) for key in effects),
            producer_id=producer.producer_id,
            depends_on_candidate_ids=tuple(candidate_ids[key] for key in operation["dependency_operation_ids"]),
            provenance_refs=(operation_cid, manifest_cid)))
        spec = specs[task_key]
        paths = [row["path"] for row in operation["outputs"]]
        scopes = _strings(spec.get("scope_paths"), "signed task scope")
        if not set(paths) <= set(scopes):
            raise IntentRequirementAdapterError("signed task output is outside its declared scope")
        validation_commands = [json.dumps(row["argv"], separators=(",", ":")) for row in spec["validations"]]
        task_metadata[candidate_ids[op_id]] = {"predicted_files": paths, "scope_ids": scopes,
            "resource_classes": ["cpu-medium"], "validation_commands": validation_commands,
            "estimated_cost_millionths": 100_000, "estimated_tokens": 128,
            "estimated_runtime_milliseconds": 1000}
        task_keys[candidate_ids[op_id]], candidate_reqs[candidate_ids[op_id]] = task_key, tuple(matched)
    scope_population = tuple(sorted({scope for item in task_metadata.values() for scope in item["scope_ids"]}))
    acceptance = tuple(sorted({row["criterion_key"] for spec in specs.values() for row in spec["acceptance"]}))
    frozen_goal = FrozenPlanningGoal(goal_id=intent.intent_id, goal_content_id=contract_cid,
        repository_tree_id=current_root_id, policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=acceptance, evidence_terms=(contract_cid, operation_cid),
            trusted_assumptions=(), supported_semantics=(INTERPRETATION_SCOPE,),
            allowed_scopes=scope_population, available_resource_classes=("cpu-medium",),
            max_estimated_resource_cost=MAX_OPERATIONS, max_estimated_tokens=MAX_OPERATIONS * 128,
            max_estimated_runtime_seconds=MAX_OPERATIONS,
            require_validation=True, require_proof=False))
    context = {"domain": INTERPRETATION_SCOPE, "repository_paths": sorted({row["path"]
        for operation in operations for row in operation["outputs"]}), "task_metadata": task_metadata}
    return IntentPlanningMaterials(intent=intent, producers=tuple(producers), task_candidates=tuple(candidates),
        predicates=predicates, current_facts=(), frozen_goal=frozen_goal, candidate_context=_freeze(context),
        requirements=tuple(_freeze(row) for row in canonical["requirements"]),
        operations=tuple(_freeze(item) for item in operations), operation_contract=_freeze(symbolic),
        operation_candidate_ids=_freeze(candidate_ids), candidate_task_keys=_freeze(task_keys),
        candidate_requirement_ids=_freeze(candidate_reqs), contract_cid=contract_cid,
        manifest_cid=manifest_cid, current_root_id=current_root_id)


__all__ = ["SYMBOLIC_OPERATION_SCHEMA", "INTENT_PLANNING_MATERIALS_SCHEMA", "INTERPRETATION_SCOPE",
           "IntentRequirementAdapterError", "IntentPlanningMaterials", "validate_symbolic_operations",
           "build_intent_planning_materials"]
