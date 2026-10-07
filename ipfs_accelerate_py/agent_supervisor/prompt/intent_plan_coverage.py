"""Bounded requirement bindings for candidate prompt goal graphs.

Groundings are independently authored input. This checker compares declared
outputs, validation keys and ordering with those groundings; it does not infer
file effects from prose or certify the translation of prose into IntentIR.
The existing graph parser and local manifest still own their checks.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from ..core.multiformats_identity import cid_for_dag_json, validate_cid
from .prompt_workflow import PromptGoalGraph, PromptOutputRecord

INTENT_REQUIREMENT_CONTRACT_SCHEMA = "intent-plan-requirement-contract@1"
INTENT_SYMBOLIC_REQUIREMENT_CONTRACT_SCHEMA = "intent-plan-requirement-contract@2"
INTENT_HEADER_REQUIREMENT_CONTRACT_SCHEMA = "intent-plan-requirement-contract@3"
INTENT_DATA_REQUIREMENT_CONTRACT_SCHEMA = "intent-plan-requirement-contract@4"
INTENT_SYMBOLIC_CONTRACT_SCHEMAS = frozenset({
    INTENT_SYMBOLIC_REQUIREMENT_CONTRACT_SCHEMA, INTENT_HEADER_REQUIREMENT_CONTRACT_SCHEMA,
    INTENT_DATA_REQUIREMENT_CONTRACT_SCHEMA,
})
INTENT_PLAN_PROPOSAL_SCHEMA = "intent-plan-proposal@1"
INTENT_PLAN_PROVIDER_REQUEST_SCHEMA = "intent-plan-provider-request@1"
INTENT_PLAN_COVERAGE_RECEIPT_SCHEMA = "intent-plan-coverage-receipt@1"
MAX_INTENT_PLAN_BYTES = 4 * 1024 * 1024
MAX_INTENT_REQUIREMENTS = 256
MAX_INTENT_PLAN_DEPTH = 48
_AUTHORITY = {
    "semantic_alignment_verified": False,
    "execution_authority": False,
    "completion_authority": False,
    "proof_authority": False,
}


class IntentPlanCoverageError(ValueError):
    """A requirement contract or provider binding is malformed."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise IntentPlanCoverageError("bounded JSON value required") from exc


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntentPlanCoverageError("duplicate JSON object key")
        result[key] = value
    return result


def _bounded(value: Any, depth: int = 0) -> None:
    if depth > MAX_INTENT_PLAN_DEPTH:
        raise IntentPlanCoverageError("JSON depth bound exceeded")
    if isinstance(value, dict):
        for item in value.values():
            _bounded(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _bounded(item, depth + 1)


def _decode(text: str) -> Any:
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_INTENT_PLAN_BYTES:
        raise IntentPlanCoverageError("JSON byte bound exceeded")
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda _: _invalid_constant())
    except (ValueError, RecursionError) as exc:
        raise IntentPlanCoverageError("strict JSON required") from exc
    _bounded(value)
    return value


def _invalid_constant():
    raise IntentPlanCoverageError("nonfinite JSON value")


def _object(value: Any, keys: set[str], noun: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise IntentPlanCoverageError(f"exact {noun} fields required")
    return value


def _text(value: Any, noun: str, maximum: int = 256) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value.encode("utf-8")) > maximum
            or any(char in value for char in "\n\r\0")):
        raise IntentPlanCoverageError(f"bounded {noun} text required")
    return value


def _array(value: Any, noun: str, maximum: int = MAX_INTENT_REQUIREMENTS) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        raise IntentPlanCoverageError(f"bounded {noun} array required")
    return value


def _strings(value: Any, noun: str) -> list[str]:
    result = [_text(item, noun) for item in _array(value, noun)]
    if len(result) != len(set(result)):
        raise IntentPlanCoverageError(f"duplicate {noun}")
    return sorted(result)


def _path(value: Any) -> str:
    name = _text(value, "source-relative path", 4096)
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or str(path) != name
            or name == "." or ".git" in path.parts
            or any(part.startswith(".runtime") for part in path.parts)):
        raise IntentPlanCoverageError("canonical source-relative path required")
    return name


def _outputs(value: Any) -> list[dict]:
    result = []
    for raw in _array(value, "outputs"):
        item = _object(raw, {"path", "effect", "media_type"}, "output")
        try:
            output = PromptOutputRecord(path=_path(item["path"]),
                                        effect=item["effect"],
                                        media_type=_text(item["media_type"], "media_type"))
        except ValueError as exc:
            raise IntentPlanCoverageError("invalid grounded output") from exc
        result.append({"path": output.path, "effect": output.effect,
                       "media_type": output.media_type})
    if len({item["path"] for item in result}) != len(result):
        raise IntentPlanCoverageError("duplicate grounded output paths")
    return sorted(result, key=lambda item: (item["path"], item["effect"], item["media_type"]))


def _mode(requirement: Mapping[str, Any]) -> str:
    if (requirement["representation"] != "native_statement"
            or requirement["compound"] is not None or requirement["kind"] != "goal"):
        return "unsupported"
    modality = requirement["modality"]
    if modality in {"required", "intended"}:
        return "mandatory"
    if modality == "prohibited":
        return "prohibited"
    return "nonmandatory"


def validate_intent_requirement_contract(
    contract: Mapping[str, Any], *, source_text: str | None = None,
) -> dict:
    """Validate independent groundings and the embedded source-bound ledger.

    Supplying source_text additionally checks the exact immutable source. The
    runtime must supply it before accepting an independently signed contract.
    """
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import (
        validate_intent_requirement_ledger,
    )

    value = _decode(_json(contract))
    keys = {"schema", "source_path", "ledger", "requirements"}
    if value.get("schema") in INTENT_SYMBOLIC_CONTRACT_SCHEMAS:
        keys.add("symbolic_operations")
    if value.get("schema") == INTENT_HEADER_REQUIREMENT_CONTRACT_SCHEMA:
        keys.add("source_applicability")
    if value.get("schema") == INTENT_DATA_REQUIREMENT_CONTRACT_SCHEMA:
        keys.add("reviewed_data_transform")
    _object(value, keys, "intent contract")
    if value["schema"] not in {INTENT_REQUIREMENT_CONTRACT_SCHEMA, *INTENT_SYMBOLIC_CONTRACT_SCHEMAS}:
        raise IntentPlanCoverageError("unsupported intent contract schema")
    value["source_path"] = _path(value["source_path"])
    try:
        if source_text is None:
            # Units partition the exact original source, including whitespace
            # and unreported regions. The datasets validator verifies this
            # reconstruction against the ledger's full source identity.
            source_text = "".join(unit["text"] for unit in value["ledger"]["source_units"])
        ledger = validate_intent_requirement_ledger(value["ledger"], source_text=source_text)
    except (KeyError, TypeError, ValueError) as exc:
        raise IntentPlanCoverageError(f"invalid requirement ledger: {exc}") from exc
    value["ledger"] = ledger
    requirements = {item["requirement_id"]: item for item in ledger["requirements"]}
    if len(requirements) > MAX_INTENT_REQUIREMENTS:
        raise IntentPlanCoverageError("requirement count bound exceeded")
    specs = []
    for raw in _array(value["requirements"], "grounded requirements"):
        item = _object(raw, {"requirement_id", "outputs", "validation_keys",
                             "dependency_requirement_ids"}, "grounded requirement")
        key = _text(item["requirement_id"], "requirement_id")
        if key not in requirements:
            raise IntentPlanCoverageError("unknown grounded requirement")
        mode = _mode(requirements[key])
        if mode in {"unsupported", "nonmandatory"}:
            raise IntentPlanCoverageError("unsupported or nonmandatory requirement cannot authorize tasks")
        spec = {"requirement_id": key, "outputs": _outputs(item["outputs"]),
                "validation_keys": _strings(item["validation_keys"], "validation_keys"),
                "dependency_requirement_ids": _strings(item["dependency_requirement_ids"],
                                                       "dependency_requirement_ids")}
        if mode == "prohibited":
            if not spec["outputs"] or spec["validation_keys"] or spec["dependency_requirement_ids"]:
                raise IntentPlanCoverageError("prohibition needs explicit forbidden outputs only")
        elif not spec["outputs"] and not spec["validation_keys"]:
            raise IntentPlanCoverageError("mandatory requirement needs an output or validation grounding")
        specs.append(spec)
    if len({item["requirement_id"] for item in specs}) != len(specs):
        raise IntentPlanCoverageError("duplicate grounded requirements")
    spec_by_id = {item["requirement_id"]: item for item in specs}
    mandatory = {key for key, item in requirements.items() if _mode(item) == "mandatory"}
    if not mandatory <= set(spec_by_id):
        raise IntentPlanCoverageError("mandatory requirement lacks independent grounding")
    for spec in specs:
        dependencies = set(spec["dependency_requirement_ids"])
        if spec["requirement_id"] in dependencies or not dependencies <= mandatory:
            raise IntentPlanCoverageError("unknown, self or nonmandatory requirement dependency")
    visited, active = set(), set()
    def visit(key):
        if key in active:
            raise IntentPlanCoverageError("requirement dependency cycle")
        if key in visited:
            return
        active.add(key)
        for dependency in spec_by_id[key]["dependency_requirement_ids"]:
            visit(dependency)
        active.remove(key)
        visited.add(key)
    for key in spec_by_id:
        visit(key)
    value["requirements"] = sorted(specs, key=lambda item: item["requirement_id"])
    if value["schema"] in INTENT_SYMBOLIC_CONTRACT_SCHEMAS:
        from ..planning.intent_requirement_adapter import validate_symbolic_operations

        value["symbolic_operations"] = validate_symbolic_operations(
            value["symbolic_operations"], ledger=ledger, requirements=value["requirements"],
        )
    if value["schema"] == INTENT_HEADER_REQUIREMENT_CONTRACT_SCHEMA:
        from ..runtime.header_intent_applicability import validate_applicability_selection
        value["source_applicability"] = validate_applicability_selection(
            value["source_applicability"], operations=value["symbolic_operations"]["operations"])
    if value["schema"] == INTENT_DATA_REQUIREMENT_CONTRACT_SCHEMA:
        from ..planning.intent_data_transform import validate_reviewed_data_transform
        value["reviewed_data_transform"] = validate_reviewed_data_transform(
            value["reviewed_data_transform"], operations=value["symbolic_operations"]["operations"])
    return value


def build_intent_requirement_contract(
    *, source_path: str, ledger: Mapping[str, Any], requirements: list[dict],
    source_text: str | None = None,
    symbolic_operations: Mapping[str, Any] | None = None,
    reviewed_data_transform: Mapping[str, Any] | None = None,
) -> dict:
    """Build a contract from already authored groundings; no effects are inferred."""
    contract = {
        "schema": INTENT_REQUIREMENT_CONTRACT_SCHEMA,
        "source_path": source_path, "ledger": ledger, "requirements": requirements,
    }
    if symbolic_operations is not None:
        contract.update(schema=INTENT_SYMBOLIC_REQUIREMENT_CONTRACT_SCHEMA,
                        symbolic_operations=symbolic_operations)
    if reviewed_data_transform is not None:
        if symbolic_operations is None:
            raise IntentPlanCoverageError("reviewed data transformation requires explicit symbolic operations")
        contract.update(schema=INTENT_DATA_REQUIREMENT_CONTRACT_SCHEMA,
                        reviewed_data_transform=reviewed_data_transform)
    return validate_intent_requirement_contract(contract, source_text=source_text)


def _bindings(value: Any) -> list[dict]:
    result = []
    for raw in _array(value, "requirement_bindings"):
        item = _object(raw, {"requirement_id", "task_keys", "validation_keys"}, "requirement binding")
        tasks = _strings(item["task_keys"], "task_keys")
        if not tasks:
            raise IntentPlanCoverageError("requirement binding must identify tasks")
        result.append({"requirement_id": _text(item["requirement_id"], "requirement_id"),
                       "task_keys": tasks,
                       "validation_keys": _strings(item["validation_keys"], "validation_keys")})
    if len({item["requirement_id"] for item in result}) != len(result):
        raise IntentPlanCoverageError("duplicate requirement bindings")
    return sorted(result, key=lambda item: item["requirement_id"])


def parse_intent_plan_proposal(
    text: str, *, contract: Mapping[str, Any] | None = None,
) -> tuple[str, list[dict]]:
    """Unwrap strict JSON without changing the inner graph's content identity.

    The graph still requires parse_prompt_goal_graph. Callers providing the
    signed contract also reject a response bound to a different contract.
    """
    value = _object(_decode(text), {"schema", "contract_cid", "graph_proposal",
                                    "requirement_bindings"}, "intent plan proposal")
    if value["schema"] != INTENT_PLAN_PROPOSAL_SCHEMA:
        raise IntentPlanCoverageError("unsupported intent plan proposal schema")
    try:
        validate_cid(value["contract_cid"], codecs=("dag-json",))
    except ValueError as exc:
        raise IntentPlanCoverageError("canonical contract CID required") from exc
    if contract is not None:
        expected = cid_for_dag_json(validate_intent_requirement_contract(contract))
        if value["contract_cid"] != expected:
            raise IntentPlanCoverageError("provider contract identity is stale or mismatched")
    if not isinstance(value["graph_proposal"], dict):
        raise IntentPlanCoverageError("inner graph proposal must be an object")
    return _json(value["graph_proposal"]), _bindings(value["requirement_bindings"])


def build_intent_plan_provider_request(existing_json_prompt: str, contract: Mapping[str, Any]) -> str:
    """Wrap an existing planner request with explicit administrative groundings."""
    graph_request = _decode(existing_json_prompt)
    if not isinstance(graph_request, dict):
        raise IntentPlanCoverageError("existing graph provider request must be an object")
    canonical = validate_intent_requirement_contract(contract)
    result = {
        "schema": INTENT_PLAN_PROVIDER_REQUEST_SCHEMA,
        "graph_request": graph_request,
        "intent_contract": canonical,
        "contract_cid": cid_for_dag_json(canonical),
        "response_contract": {
            "schema": INTENT_PLAN_PROPOSAL_SCHEMA,
            "fields": ["schema", "contract_cid", "graph_proposal", "requirement_bindings"],
            "binding_fields": ["requirement_id", "task_keys", "validation_keys"],
            "instructions": (
                "Return one JSON object using the exact response fields. Copy contract_cid. "
                "graph_proposal must obey graph_request's graph proposal contract. Bind every "
                "mandatory requirement to task keys and its grounded validation keys. Match "
                "explicit output paths, effects and media types. Preserve requirement ordering. "
                "Permissions do not require execution; prohibitions forbid their grounded effects. "
                "Do not claim execution coverage for compound or unsupported requirements."
            ),
        },
        **_AUTHORITY,
    }
    return _json(_decode(_json(result)))


def check_intent_plan_coverage(
    contract: Mapping[str, Any], *, graph: PromptGoalGraph, bindings: list[dict],
) -> dict:
    """Compare canonical tasks with explicit groundings, retaining rejection evidence."""
    canonical = validate_intent_requirement_contract(contract)
    if not isinstance(graph, PromptGoalGraph):
        raise IntentPlanCoverageError("canonical PromptGoalGraph required")
    try:
        # Recheck graph edges, population and depth before ancestor traversal.
        graph = PromptGoalGraph.from_dict(graph.to_dict())
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise IntentPlanCoverageError("invalid bounded PromptGoalGraph") from exc
    ledger = canonical["ledger"]
    reqs = {item["requirement_id"]: item for item in ledger["requirements"]}
    specs = {item["requirement_id"]: item for item in canonical["requirements"]}
    rows = _bindings(bindings)
    by_req = {item["requirement_id"]: item for item in rows}
    tasks = {item.task_key: item for item in graph.tasks}
    for row in rows:
        key = row["requirement_id"]
        if key not in reqs or _mode(reqs[key]) != "mandatory":
            raise IntentPlanCoverageError("unknown or nonmandatory requirement binding")
        if not set(row["task_keys"]) <= set(tasks):
            raise IntentPlanCoverageError("binding identifies unknown task key")
        if set(row["validation_keys"]) != set(specs[key]["validation_keys"]):
            raise IntentPlanCoverageError("binding differs from grounded validation keys")
    mandatory = {key for key, item in reqs.items() if _mode(item) == "mandatory"}
    unsupported = sorted(key for key, item in reqs.items() if _mode(item) == "unsupported")
    uncovered = set(mandatory - set(by_req))
    errors = []
    def error(code, **details):
        errors.append({"code": code, **details})
    if not ledger["source_accounting_complete"]:
        error("incomplete_source_accounting")
    unsupported_units = sorted(item["unit_id"] for item in ledger["source_units"]
                               if item["disposition"] == "unsupported")
    if unsupported_units:
        error("unsupported_source_units", source_unit_ids=unsupported_units)
    if unsupported:
        error("unsupported_requirement_scope", requirement_ids=unsupported)
    def triple(output):
        if isinstance(output, dict):
            return output["path"], output["effect"], output["media_type"]
        return output.path, output.effect, output.media_type
    task_requirements = {key: [] for key in tasks}
    projections = []
    for row in rows:
        key = row["requirement_id"]
        spec = specs[key]
        bound = [tasks[task_key] for task_key in row["task_keys"]]
        available_outputs = {triple(output) for task in bound for output in task.outputs}
        missing_outputs = [output for output in spec["outputs"] if triple(output) not in available_outputs]
        linked_validation_keys = {
            validation_key for task in bound for criterion in task.acceptance
            for validation_key in criterion.validation_keys
        }
        missing_validations = sorted(set(spec["validation_keys"]) - linked_validation_keys)
        if missing_outputs or missing_validations:
            uncovered.add(key)
            error("missing_grounded_obligations", requirement_id=key,
                  outputs=missing_outputs, validation_keys=missing_validations)
        for task in bound:
            task_requirements[task.task_key].append(key)
            has_purpose = ({triple(output) for output in task.outputs}
                           & {triple(output) for output in spec["outputs"]}
                           or {validation.validation_key for validation in task.validations}
                           & set(spec["validation_keys"]))
            if not has_purpose:
                error("task_has_no_grounded_purpose", requirement_id=key, task_key=task.task_key)
        projections.append({**row, "task_cids": sorted(task.task_cid for task in bound)})
    for task_key, requirement_ids in task_requirements.items():
        task = tasks[task_key]
        if not requirement_ids:
            error("orphan_task", task_key=task_key)
        allowed_outputs = {triple(output) for key in requirement_ids for output in specs[key]["outputs"]}
        extra_outputs = [dict(path=output.path, effect=output.effect, media_type=output.media_type)
                         for output in task.outputs if triple(output) not in allowed_outputs]
        allowed_validations = {validation_key for key in requirement_ids
                               for validation_key in specs[key]["validation_keys"]}
        extra_validations = sorted(validation.validation_key for validation in task.validations
                                   if validation.validation_key not in allowed_validations)
        if extra_outputs or extra_validations:
            error("ungrounded_task_obligations", task_key=task_key,
                  outputs=extra_outputs, validation_keys=extra_validations)
    ancestors = {}
    tasks_by_cid = {task.task_cid: task for task in tasks.values()}
    def prior_tasks(cid):
        if cid not in ancestors:
            ancestors[cid] = set(tasks_by_cid[cid].dependency_task_cids)
            for dependency in tasks_by_cid[cid].dependency_task_cids:
                ancestors[cid].update(prior_tasks(dependency))
        return ancestors[cid]
    for key, row in by_req.items():
        for dependency in specs[key]["dependency_requirement_ids"]:
            if dependency not in by_req:
                uncovered.add(key)
                error("unbound_requirement_dependency", requirement_id=key, dependency_requirement_id=dependency)
                continue
            required_cids = {tasks[task_key].task_cid for task_key in by_req[dependency]["task_keys"]}
            for task_key in row["task_keys"]:
                if not required_cids <= prior_tasks(tasks[task_key].task_cid):
                    uncovered.add(key)
                    error("missing_requirement_ordering", requirement_id=key,
                          dependency_requirement_id=dependency, task_key=task_key)
    prohibited = []
    for key, requirement in reqs.items():
        if _mode(requirement) != "prohibited":
            continue
        if key not in specs:
            unsupported.append(key)
            error("ungrounded_prohibition", requirement_id=key)
            continue
        forbidden = {(output["path"], output["effect"]) for output in specs[key]["outputs"]}
        for task in tasks.values():
            for output in task.outputs:
                if (output.path, output.effect) in forbidden:
                    prohibited.append({"requirement_id": key, "task_key": task.task_key,
                                       "path": output.path, "effect": output.effect})
    result = {
        "schema": INTENT_PLAN_COVERAGE_RECEIPT_SCHEMA,
        "accepted": not (errors or uncovered or unsupported or prohibited),
        "contract_cid": cid_for_dag_json(canonical),
        "ledger_sha256": ledger["ledger_sha256"],
        "graph_cid": graph.plan_root_cid,
        "bindings": projections,
        "uncovered_requirement_ids": sorted(uncovered),
        "unsupported_requirement_ids": sorted(set(unsupported)),
        "prohibited_output_violations": prohibited,
        "errors": errors,
        "source_accounting_complete": ledger["source_accounting_complete"],
        "semantic_support_complete": ledger["semantic_support_complete"],
        **_AUTHORITY,
    }
    return _decode(_json(result))
