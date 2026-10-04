"""Signed finite repository context with the original local task population.

The bounded Python/Lean observations inform a separately versioned admission.
They never remove administrator tasks or upgrade a source/model claim to proof.
Actual tasks retain the existing signed local pending-completion contract.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as observation
from ipfs_datasets_py.logic.software_contracts import codebase_integer_model_lean as model
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
    IntegerOffsetContract, compile_integer_offset,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

from ..control.profile_authority import load_local_profile
from ..planning import finite_integer_capacity_preview as capacity_preview
from ..planning import finite_integer_codebase as matcher
from ..planning.finite_integer_plan_preview import (
    FiniteIntegerOperationCatalog, ReviewedFiniteIntegerOperation,
    finite_integer_intent_cid, finite_integer_prompt_cid,
)
from ..planning.finite_integer_source_custody import capture_source_custody
from ..planning.plan_revision_contracts import PlanCreateRequest
from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ..prompt.plan_create_service import PlanCreatePreviewReceipt
from ..prompt.prompt_workflow import PromptGoalGraph
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local

PROFILE = "finite-repository-fixed-administrator-population@1"
DECLARATION_SCHEMA = "supervisor-finite-repository-declaration@1"
ADMISSION_SCHEMA = "supervisor-finite-repository-admission@1"
REFERENCE_SCHEMA = "supervisor-finite-repository-admission-reference@1"
MAX_BYTES = 4 * 1024**2
_CAPACITY_SCHEMA = "finite-integer-capacity-bound-plan-preview@2"
_REQUIREMENTS = {matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID}
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "code_proof_authority", "production_admitted", "production_activation",
    "execution_authority", "completion_authority", "mutation_authority",
    "omission_authority", "worker_launched", "convergence_proved",
)}
_POLICY = {"schema": "finite-repository-fixed-population-policy@1",
    "preserve_complete_administrator_task_population": True,
    "finite_facts_are_context_only": True, "models": "off",
    "completion": "existing_local_pending_tests", **_FALSE}
_FEATURE_POLICY = {"schema": "finite-plan-immutable-advisory-feature-validation@1",
    "mode": "model_off", "numerical_verification_points": [],
    "between_verifications": "current_source_CAS_AST_and_all_frozen_context_lineage_bytes_at_every_fence",
    "final_order": "native_verification_then_detached_byte_closure",
    "behavior_authority": False, "proof_authority": False, "execution_authority": False}
_CAPACITY_FIELDS = {"schema", "profile", "original_finite_preview", "preview", "input_snapshot",
    "match", "operation_catalog", "operation_catalog_cid", "operational_model", "feature_context",
    "feature_validation_policy", "source_custody", "obligation_graph", "portfolio", "candidate_plan", "critique", "critic_evidence",
    "execution_plan", "capacity_observation", "capacity_compilation_request", "planner_status",
    "selected_task_ids", "declared_task_requirement_ids", "current_facts_count", "scope", "capacity_binding",
    "reservation_released_on_return", "capacity_scope", "training_steps_during_preview", "planning_model_calls",
    "result_cid", *capacity_preview._FALSE}
_MATCH_FIELDS = {"schema", "profile", "query", "status", "head", "current_root_id", "structural_context",
    "source_cid", "domain_cid", "domain_inputs", "typed_intent", "current_facts", "clause_results",
    "eligible_clause_ids", "residual_clause_ids", "eligible_requirements", "residual_requirements",
    "finite_counterexamples", "observation", "observation_cid", "semantic_alignment_verified",
    "alignment_scope", "scope", "unbounded_behavior_status", "removed_task_ids", "match_cid", *matcher._AUTHORITY}


class FiniteRepositoryAdmissionError(ValueError):
    """A closed finite declaration, evidence or native commit was refused."""


def _need(condition, message):
    if not condition:
        raise FiniteRepositoryAdmissionError(message)


def _plain(value):
    try:
        raw = canonical_dag_json_bytes(value)
        _need(len(raw) <= MAX_BYTES, "finite admission byte bound exceeded")
        return json.loads(raw)
    except (TypeError, ValueError, RecursionError) as error:
        if isinstance(error, FiniteRepositoryAdmissionError):
            raise
        raise FiniteRepositoryAdmissionError("bounded canonical finite record required") from error


def _same(left, right):
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _pins():
    modules = (local, matcher, capacity_preview, observation, model)
    rows = {item.__name__: hashlib.sha256(Path(item.__file__).read_bytes()).hexdigest()
            for item in modules}
    rows[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"schema": "finite-repository-admission-implementation@1",
            "source_sha256": rows, "scope": "selected producer bytes; no process-origin attestation"}


def _profile(manifest):
    declared = manifest.get("payload", {})
    local._validate_local_manifest_declarations(declared)
    _need(_same(declared["policy"], local.LOCAL_POLICY), "exact local authority policy booleans required")
    _need(declared["schema"] != local.INTENT_MANIFEST_SCHEMA,
          "first finite fixed-population profile supports local manifests 1, 2 and 3")
    profile = load_local_profile(repository_cid=declared["repository_cid"],
        profile_dir=Path(declared["profile_dir"]), lifecycle_dir=Path(declared["lifecycle_dir"]))
    local._verify_signature(manifest, profile)
    _need(profile.content_id == declared["profile_content_id"]
          and all(profile.allows(cap) for cap in
                  ("read", "edit", "test", "isolated_worktree", "write_worktree")),
          "active profile differs from the exact local declaration")
    root = Path(declared["repository"])
    _need(not any(Path(declared[key]).is_relative_to(root)
                  for key in ("profile_dir", "lifecycle_dir")),
          "finite owner keys must remain outside the worker repository")
    return profile


def _catalog(value):
    _need(type(value) is dict and set(value) == {"schema", "profile", "operations", "authority"},
          "complete exact finite operation catalog required")
    try:
        result = FiniteIntegerOperationCatalog(tuple(ReviewedFiniteIntegerOperation(**row)
                                                     for row in value["operations"]))
    except (TypeError, ValueError, KeyError) as error:
        raise FiniteRepositoryAdmissionError("invalid finite operation catalog") from error
    _need(result.to_dict() == value, "finite operation catalog changed")
    return result


def _declaration(envelope):
    envelope = _plain(envelope)
    _need(type(envelope) is dict and set(envelope) == {"payload", "binding"},
          "exact signed finite declaration envelope required")
    payload = envelope["payload"]
    fields = {"schema", "profile", "manifest", "head", "request", "intent_json",
              "source_text", "operation_catalog", "task_bindings", "tool_policy",
              "policy", "implementation"}
    _need(capacity_preview.SCHEMA == _CAPACITY_SCHEMA
          and type(payload) is dict and set(payload) == fields
          and payload["schema"] == DECLARATION_SCHEMA and payload["profile"] == PROFILE
          and _same(payload["policy"], _POLICY) and _same(payload["implementation"], _pins()),
          "finite declaration schema, exact policy or producer bytes differ")
    profile = _profile(payload["manifest"])
    local._verify_signature(envelope, profile)
    declared = payload["manifest"]["payload"]
    request = PlanCreateRequest.from_dict(payload["request"])
    head = CodebaseHead.from_dict(payload["head"])
    document = decode_intent_ir(payload["intent_json"])
    _need(canonical_intent_ir_bytes(document).decode() == payload["intent_json"],
          "complete canonical native intent bytes required")
    catalog = _catalog(payload["operation_catalog"])
    query = matcher.prepare_finite_integer_query(intent_document=document, source_text=payload["source_text"])
    _need(query["supported"] is True and set(query["requirement_ids"]) == _REQUIREMENTS,
          "complete supported two-clause finite instruction required")
    contract = IntegerOffsetContract.from_dict(query["contract"])
    _need(request.repository_root == declared["repository"]
          and request.repository_id == head.repository_id
          and request.roots.repository_root_cid == head.snapshot_cid
          and request.roots.dirty_worktree_root == head.snapshot_cid
          and request.prompt_source_cid == finite_integer_prompt_cid(payload["source_text"])
          and request.roots.intent_ir_root == finite_integer_intent_cid(document)
          and request.roots.capability_catalog_root == catalog.cid
          and request.scope_paths == (contract.path,)
          and request.budget.max_model_calls == 0
          and declared["planning_roots"]["request_cid"] == request.request_cid
          and declared["planning_roots"]["program_root"] == request.roots.program_root,
          "finite prompt, IR, request roots, model-off policy or local source differ")
    bindings = payload["task_bindings"]
    specs = {item["task_key"]: item for item in declared["tasks"]}
    _need(type(bindings) is dict and set(bindings) == _REQUIREMENTS
          and all(type(key) is str and key in specs for key in bindings.values())
          and len(set(bindings.values())) == 2 and len(specs) >= 2,
          "both finite clauses must bind distinct original administrator tasks")
    for operation in catalog.operations:
        _need((operation.path, operation.function_name, operation.parameter)
              == (contract.path, contract.function_name, contract.parameter),
              "reviewed operation retargets the finite source contract")
        spec = specs[bindings[operation.requirement_id]]
        _need(contract.path in spec["scope_paths"]
              and {item["path"] for item in spec["outputs"]} == {contract.path}
              and all(item["effect"] == "modify" for item in spec["outputs"]),
              "finite mapped task must preserve its exact declared source modification")
    observation._policy(payload["tool_policy"])
    return payload, profile, request, head, document, catalog, query


def author_finite_repository_declaration(*, owner, manifest, request, intent_document,
        source_text, operation_catalog, tool_policy, task_bindings):
    """Sign complete administrator declarations before finite/model planning.

    The original manifest supplies every task, output and acceptance command.
    Facts, previews and candidate task subsets cannot enter this API.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner and type(request) is PlanCreateRequest
          and type(intent_document) is IntentIRDocument
          and type(operation_catalog) is FiniteIntegerOperationCatalog,
          "exact native owner, request, intent and catalog required")
    manifest, tool_policy, task_bindings = _plain(manifest), _plain(tool_policy), _plain(task_bindings)
    intent_json = canonical_intent_ir_bytes(intent_document).decode()
    source_text = matcher._text(source_text)
    custody = capture_source_custody(owner)
    local._manifest(manifest, initial=True)
    _need(owner.repository == Path(manifest["payload"]["repository"]), "owner repository differs")
    payload = _plain({"schema": DECLARATION_SCHEMA, "profile": PROFILE,
        "manifest": manifest, "head": owner.expected_head.to_dict(), "request": request.to_dict(),
        "intent_json": intent_json, "source_text": source_text,
        "operation_catalog": operation_catalog.to_dict(), "task_bindings": task_bindings,
        "tool_policy": tool_policy, "policy": _POLICY, "implementation": _pins()})
    envelope = local._signed(payload, manifest["payload"])
    _declaration(envelope)
    custody.require_current()
    return _plain(envelope)


def _read(path, maximum, checkpoint=lambda: None):
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
          "canonical regular finite artifact path required")
    chunks, size = [], 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum,
              "finite artifact byte bound exceeded")
        for block in iter(lambda: stream.read(64 * 1024), b""):
            checkpoint()
            size += len(block)
            _need(size <= maximum, "finite artifact grew beyond its bound")
            chunks.append(block)
        after, current = os.fstat(stream.fileno()), path.stat(follow_symlinks=False)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    _need(identity(before) == identity(after) == identity(current)
          and path.resolve(strict=True) == path and not path.is_symlink(),
          "finite artifact descriptor or path changed")
    return b"".join(chunks)


def _artifacts(record, checkpoint=lambda: None):
    output = Path(record["output"])
    _need(output.is_absolute() and output.resolve(strict=True) == output
          and output.is_dir() and not output.is_symlink(), "canonical finite evidence directory required")
    raws = {}
    _need(type(record["artifacts"]) is dict and 1 <= len(record["artifacts"]) <= 16,
          "complete bounded finite artifact population required")
    for name, row in record["artifacts"].items():
        _need(type(row) is dict and set(row) == {"path", "sha256", "size_bytes", "cid"}
              and type(row["size_bytes"]) is int and 0 <= row["size_bytes"] <= 64 * 1024**2
              and Path(row["path"]).parent == output, "closed finite artifact descriptor required")
        raw = _read(row["path"], row["size_bytes"], checkpoint)
        _need(len(raw) == row["size_bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
              and cid_for_bytes(raw) == row["cid"], "finite artifact bytes differ")
        raws[name] = raw
    _need(_read(output / "result.json", MAX_BYTES, checkpoint) == canonical_dag_json_bytes(record),
          "complete finite result artifact differs")
    return raws


def _artifact_fence(owner, evidence, checkpoint):
    """Freeze expected identities, then check without application callbacks."""
    rows = {}
    def add(path, size, digest):
        path = Path(path)
        expected = (size, digest)
        _need(path not in rows or rows[path] == expected, "conflicting finite artifact identities")
        rows[path] = expected
    def literal(path, raw):
        add(path, len(raw), hashlib.sha256(raw).hexdigest())
    observed, operational = evidence["match"]["observation"], evidence["operational_model"]
    for record in (observed, operational):
        for row in record["artifacts"].values():
            add(row["path"], row["size_bytes"], row["sha256"])
            if record is operational:
                add(owner.index.artifacts.path_for(row["cid"], source=True), row["size_bytes"], row["sha256"])
        raw = canonical_dag_json_bytes(record)
        literal(Path(record["output"]) / "result.json", raw)
        literal(owner.index.artifacts.path_for(cid_for_structured(record)), raw)
    source = observed["artifacts"]["source"]
    add(owner.index.artifacts.path_for(source["cid"], source=True), source["size_bytes"], source["sha256"])
    for name in ("python", "lean"):
        row = observed["tool_policy"][name]
        add(row["path"], row["size_bytes"], row["sha256"])
    def fence():
        for path, (size, digest) in rows.items():
            raw = _read(path, size, checkpoint)
            _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
                  "owned finite artifact bytes changed after application callbacks")
        checkpoint()
    return fence


def _evidence(evidence, declaration, graph, checkpoint=lambda: None):
    """Rebuild the clause/fact partition from the complete retained domain rows."""
    payload, profile, request, head, document, catalog, query = _declaration(declaration)
    value = _plain(evidence)
    _need(capacity_preview.SCHEMA == _CAPACITY_SCHEMA and set(value) == _CAPACITY_FIELDS
          and value.get("schema") == _CAPACITY_SCHEMA
          and value["profile"] == "finite-integer-live-reserved-capacity@2"
          and value["scope"] == "explicit_finite_domain_only"
          and value["capacity_scope"] == "feasibility_during_live_reservation_only"
          and value.get("result_cid") == cid_for_structured({key: item for key, item in value.items()
                                                           if key != "result_cid"})
          and all(value.get(name) is False for name in capacity_preview._FALSE)
          and type(value.get("planning_model_calls")) is int and value["planning_model_calls"] == 0
          and type(value.get("training_steps_during_preview")) is int and value["training_steps_during_preview"] == 0
          and _same(value.get("feature_context"), {"schema": "finite-plan-feature-selection@1",
              "mode": "model_off", "head": head.to_dict(), "behavior_authority": False, "proof_authority": False})
          and _same(value["feature_validation_policy"], _FEATURE_POLICY)
          and value.get("reservation_released_on_return") is True
          and value.get("operation_catalog") == catalog.to_dict()
          and value.get("operation_catalog_cid") == catalog.cid,
          "complete model-off finite evidence or exact authority booleans differ")
    preview = PlanCreatePreviewReceipt.from_dict(value["preview"])
    _need(preview.admitted is False and preview.read_only is True and preview.wrote_effects == ()
          and preview.mode.value == "deterministic" and preview.request_cid == request.request_cid,
          "finite preview must remain the exact deterministic review-only request")
    match = value["match"]
    _need(set(match) == _MATCH_FIELDS and match["schema"] == matcher.SCHEMA and match["profile"] == matcher.PROFILE
          and match["scope"] == "explicit_finite_domain_only"
          and match["unbounded_behavior_status"] == "unresolved" and match["removed_task_ids"] == []
          and match["semantic_alignment_verified"] is query["semantic_alignment_verified"]
          and match["alignment_scope"] == query["alignment_scope"]
          and match.get("match_cid") == cid_for_structured({key: item for key, item in match.items()
                                                       if key != "match_cid"})
          and match.get("query") == query and match.get("head") == head.to_dict()
          and match.get("current_root_id") == head.snapshot_cid
          and all(match.get(name) is False for name in matcher._AUTHORITY),
          "finite match exact source/query/authority identity differs")
    contract = IntegerOffsetContract.from_dict(query["contract"])
    observed = observation.validate_finite_integer_observation(match["observation"],
        expected_head=head, contract=contract, inputs=query["domain_inputs"], tool_policy=payload["tool_policy"])
    _need(observed["status"] == "observed", "successful complete native finite observation required")
    _need(match["observation_cid"] == observed["result_cid"] and match["domain_cid"] == query["domain_cid"]
          and _same(match["domain_inputs"], query["domain_inputs"]), "finite observation domain identity differs")
    raw = _artifacts(observed, checkpoint)["source"]
    compiled = compile_integer_offset(raw, contract, revision="snapshot:" + head.snapshot_cid)
    _need(observed["compiled_cid"] == compiled.cid and observed["source_cid"] == match["source_cid"],
          "independent finite source lowering differs")
    typed = matcher._typed(query, head, observed["source_cid"])
    _need(_same(match["typed_intent"], typed.to_dict()), "typed finite clause bindings differ")
    predicates = {row.predicate_id: row for row in typed.desired_predicates}
    eligible, residual, facts, clauses, examples = [], [], [], [], []
    for requirement in query["requirement_ids"]:
        predicate = predicates[typed.metadata["requirement_predicate_ids"][requirement]]
        satisfied = (observed["type_clause_satisfied"] if requirement == matcher.TYPE_STATEMENT_ID
                     else all(row["output"] == row["input"] + contract.offset for row in observed["observations"]))
        status = "bounded_observed_satisfied" if satisfied else "finite_counterexample"
        reasons = [] if satisfied else ["explicit_domain_observation_contradicts_requested_offset"]
        clauses.append({"statement_id": requirement, "predicate_id": predicate.predicate_id,
                        "status": status, "reasons": reasons, "scope": "explicit_finite_domain_only"})
        if satisfied:
            eligible.append(requirement)
            key = {"schema": "supervisor-finite-integer-fact-binding@1", "requirement_id": requirement,
                "predicate_id": predicate.predicate_id, "head": head.to_dict(),
                "source_cid": observed["source_cid"], "domain_cid": query["domain_cid"],
                "observation_cid": observed["result_cid"], "trace_cid": observed["trace_cid"]}
            refs = (query["query_cid"], cid_for_structured(head.to_dict()), head.snapshot_cid,
                observed["source_cid"], query["domain_cid"], observed["trace_cid"], observed["result_cid"],
                cid_for_structured(observed["lean_certificate"]), observed["compiled_cid"], observed["tool_policy_cid"])
            facts.append(matcher.ObservedFact(fact_id="finite-observation:" + cid_for_structured(key),
                predicate=predicate, truth=matcher.FactTruth.TRUE, authority=matcher.FactAuthority.BOUNDED_OBSERVATION,
                provenance_refs=refs, current_root_id=head.snapshot_cid,
                invalidation_selectors=predicate.invalidation_selectors).to_dict())
        else:
            residual.append(requirement)
            examples.extend({"statement_id": requirement, "input": row["input"],
                "observed_output": row["output"], "expected_output": row["input"] + contract.offset}
                for row in observed["observations"] if row["output"] != row["input"] + contract.offset)
    _need(_same(match["current_facts"], facts) and _same(match["clause_results"], clauses)
          and match["eligible_clause_ids"] == match["eligible_requirements"] == eligible
          and match["residual_clause_ids"] == residual and match["finite_counterexamples"] == examples
          and match["residual_requirements"] == [row for row in clauses if row["statement_id"] in residual]
          and type(value["current_facts_count"]) is int and value["current_facts_count"] == len(facts),
          "finite facts or complete residual partition differ")
    selected = [operation.task_id for operation in catalog.operations if operation.requirement_id in residual]
    _need(set(value["selected_task_ids"]) == set(selected)
          and len(value["selected_task_ids"]) == len(selected), "reviewed finite selected task partition differs")
    operational = value["operational_model"]
    _need(set(operational) == model._RECORD_FIELDS and operational["schema"] == model.SCHEMA
          and operational["profile"] == model.PROFILE and operational["scope"] == model.SCOPE
          and all(operational[name] is False for name in model._FALSE)
          and operational["head"] == head.to_dict() and operational["source_cid"] == observed["source_cid"]
          and operational["manifest_cid"] == head.manifest_cid and operational["source_path"] == contract.path
          and operational["source_sha256"] == hashlib.sha256(raw).hexdigest()
          and operational["contract"] == contract.to_dict() and operational["contract_cid"] == contract.cid
          and _same(operational["tool_policy"], payload["tool_policy"])
          and operational["tool_policy_cid"] == payload["tool_policy"]["policy_cid"]
          and operational["result_cid"] == cid_for_structured({key: item for key, item in operational.items()
                                                               if key != "result_cid"}),
          "operational model source/domain/authority binding differs")
    model_raws = _artifacts(operational, checkpoint)
    _need(set(model_raws) == set(model._NAMES), "complete exact operational model artifacts required")
    implementation = model._implementation()
    translation = model._translation(compiled, head, implementation)
    lean_text, names = model._lean(translation)
    certificate = operational["lean_certificate"]
    _need(type(certificate) is dict and set(certificate) == model._CERTIFICATE_FIELDS
          and certificate["schema"] == "codebase-integer-operational-model-certificate@1"
          and certificate["scope"] == model.SCOPE and certificate["claim"] == model._CERTIFICATE_CLAIM
          and certificate["translation_cid"] == operational["translation_cid"]
          and certificate["source_cid"] == cid_for_bytes(lean_text)
          and certificate["olean_cid"] == operational["artifacts"]["lean_olean"]["cid"]
          and _same(certificate["tool"], payload["tool_policy"]["lean"])
          and certificate["dependency_scope"] == payload["tool_policy"]["dependency_scope"],
          "operational model certificate scope and exact bindings differ")
    for process, arguments in ((certificate["version_process"], ["--version"]),
                               (certificate["process"], model._ARGS)):
        flags = ("timed_out", "cancelled", "unavailable", "output_truncated", "workspace_limit_exceeded",
                 "process_tree_terminated", "resource_exhausted")
        _need(type(process) is dict and set(process) == set(observed["python_process"])
              and process["command"] == [payload["tool_policy"]["lean"]["path"], *arguments]
              and type(process["returncode"]) is int and process["returncode"] == 0
              and type(process["elapsed_ms"]) is int and process["elapsed_ms"] >= 0
              and process["workspace_cleaned"] is True and all(process[name] is False for name in flags)
              and all(type(process[name]) is str for name in
                      ("interface_version", "stdout", "stderr", "termination_reason", "error"))
              and process["stderr"] == process["error"] == "",
              "operational model requires successful exact native process records")
        limits = process["limits"]
        policy_limits = payload["tool_policy"]["process_limits"]
        _need(type(limits) is dict and set(limits) == set(observed["python_process"]["limits"])
              and all(type(item) is int and item > 0 for item in limits.values())
              and limits["timeout_ms"] <= 300000 and limits["cpu_seconds"] <= 300
              and limits["address_space_bytes"] == policy_limits["lean"]["address_space_bytes"]
              and limits["resident_memory_bytes"] == policy_limits["lean"]["resident_memory_bytes"]
              and all(limits[name] == policy_limits[name] for name in
                      ("max_input_bytes", "max_output_bytes", "max_workspace_bytes", "max_output_files")),
              "operational model process limits differ from the sealed tool policy")
    _need(certificate["version_process"]["stdout"].startswith("Lean (version ")
          and certificate["process"]["stdout"] == "", "operational model native compiler output differs")
    expected_model = {"source": raw, "compiled": canonical_dag_json_bytes(compiled.to_dict()),
        "translation": canonical_dag_json_bytes(translation), "frontend": canonical_dag_json_bytes(implementation),
        "tool_policy": canonical_dag_json_bytes(payload["tool_policy"]), "lean_source": lean_text,
        "lean_certificate": canonical_dag_json_bytes(certificate),
        "lean_process": canonical_dag_json_bytes({"version_process": certificate["version_process"],
                                                   "process": certificate["process"]})}
    _need(all(model_raws[name] == item for name, item in expected_model.items()),
          "operational model artifacts do not replay from the independent source projection")
    _need(model_raws["source"] == raw and operational["compiled_cid"] == compiled.cid
          and operational["translation"] == translation
          and operational["translation_cid"] == cid_for_structured(translation)
          and model_raws["lean_source"] == lean_text and operational["lean_certificate"]["theorems"] == names
          and operational["source_identity_proved"] is True and operational["kernel_checked_model"] is True
          and operational["requested_model_theorem_proved"] is (compiled.body_offset == contract.offset)
          and operational["status"] == ("model_proved" if compiled.body_offset == contract.offset else "model_refuted")
          and _same(operational["model_counterexample"], None if compiled.body_offset == contract.offset
                    else {"input": 0, "model_output": compiled.body_offset, "required_output": contract.offset}),
          "independent operational model projection differs")
    graph = PromptGoalGraph.from_dict(graph)
    # Replay all signed scopes, outputs, dependencies and acceptance checks in
    # every branch, including a review-only no-work finite observation.
    local._planning_payload(graph, payload["manifest"], payload["manifest"]["payload"],
                            profile, payload["manifest"]["payload"]["sources"])
    _need(graph.request_cid == request.request_cid and graph.program_root == request.roots.program_root,
          "original native graph differs from finite request roots")
    specs = payload["manifest"]["payload"]["tasks"]
    tasks = {task.task_key: task for task in graph.tasks}
    _need(set(tasks) == {spec["task_key"] for spec in specs},
          "finite admission must preserve every original administrator task")
    native = {requirement: {"task_key": key, "task_cid": tasks[key].task_cid}
              for requirement, key in payload["task_bindings"].items()}
    semantic = {"schema": "finite-repository-context-projection@1", "head": head.to_dict(),
        "source_cid": observed["source_cid"], "query": query, "typed_intent": typed.to_dict(),
        "domain_inputs": query["domain_inputs"], "observations": observed["observations"],
        "eligible_requirement_ids": eligible, "residual_requirement_ids": residual,
        "finite_selected_task_ids": value["selected_task_ids"], "operation_catalog_cid": catalog.cid,
        "native_task_bindings": native, "administrator_task_cids": sorted(task.task_cid for task in graph.tasks),
        "model_translation_cid": operational["translation_cid"], "model_status": operational["status"],
        "task_population_preserved": True, "finite_facts_are_context_only": True, **_FALSE}
    checkpoint()
    return _plain(semantic)


def _receipt_payload(declaration, graph, evidence, local_admission, semantic):
    permitted = bool(semantic["residual_requirement_ids"])
    return {"schema": ADMISSION_SCHEMA, "profile": PROFILE,
        "declaration_cid": cid_for_structured(declaration), "graph_cid": cid_for_structured(graph),
        "evidence_cid": cid_for_structured(evidence), "semantic_context": semantic,
        "semantic_context_cid": cid_for_structured(semantic),
        "local_admission_cid": cid_for_structured(local_admission) if local_admission is not None else None,
        "planning_permitted": permitted, "admission_scope": "original_signed_local_task_population",
        "no_work_review_only": not permitted, "policy": _POLICY, "implementation": _pins(), **_FALSE}


def verify_finite_repository_admission(*, admission):
    """Historical signed/artifact integrity; never a current-source grant."""
    admission = _plain(admission)
    _need(type(admission) is dict and set(admission) ==
          {"declaration", "graph", "evidence", "local_admission", "receipt"}, "exact finite admission bundle required")
    payload, profile, *_ = _declaration(admission["declaration"])
    graph = PromptGoalGraph.from_dict(admission["graph"])
    semantic = _evidence(admission["evidence"], admission["declaration"], admission["graph"])
    received = local._verify_signature(admission["receipt"], profile)
    if semantic["residual_requirement_ids"]:
        old = admission["local_admission"]
        local._require_admission_fields(old)
        _need(old["manifest"] == payload["manifest"] and old["graph"] == graph.to_dict(),
              "original local signed admission or complete graph differs")
        old_receipt = local._verify_signature(old["receipt"], profile)
        expected_old = local._plain(local._planning_payload(graph, old["manifest"], payload["manifest"]["payload"],
            profile, payload["manifest"]["payload"]["sources"]))
        _need(_same(old_receipt, expected_old), "local pending admission does not independently replay")
    else:
        _need(admission["local_admission"] is None, "no-work finite evidence cannot grant local admission")
    expected = _receipt_payload(admission["declaration"], admission["graph"], admission["evidence"],
                                admission["local_admission"], semantic)
    _need(_same(received, expected), "signed finite admission does not match recomputed complete contract")
    # All native/signature callbacks finish before these detached physical reads.
    for record in (admission["evidence"]["match"]["observation"], admission["evidence"]["operational_model"]):
        _artifacts(record)
    return {"schema": "finite-repository-historical-verification@1", "admission": admission,
            "receipt": received, "semantic_context": semantic, "observed_current": False, **_FALSE}


def _fresh(owner, declaration, graph, output, policy_observer):
    payload, _, request, head, document, catalog, _ = _declaration(declaration)
    _need(type(owner) is RepositoryPlanPreviewOwner and owner.expected_head == head
          and owner.repository == Path(payload["manifest"]["payload"]["repository"])
          and callable(policy_observer), "exact current native owner and live policy observer required")
    deadline = time.monotonic() + min(owner.timeout_seconds, request.budget.max_latency_ms / 1000)
    def checkpoint():
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            LeaseCancelledError, LeaseTimeoutError,
        )
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("finite repository admission cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("finite repository admission deadline exceeded")
        return left
    checkpoint()
    local._manifest(payload["manifest"], initial=True)
    custody = capture_source_custody(owner, checkpoint)
    evidence = capacity_preview.preview_capacity_bound_finite_integer_plan(
        owner=replace(owner, timeout_seconds=checkpoint()), request=request, intent_document=document,
        source_text=payload["source_text"], operation_catalog=catalog, output=output,
        tool_policy=payload["tool_policy"], policy_observer=policy_observer)
    semantic = _evidence(evidence, declaration, graph, checkpoint)
    custody.require_current(checkpoint)
    fence = _artifact_fence(owner, evidence, checkpoint)
    fence()
    return evidence, semantic, custody, checkpoint, fence


def admit_finite_repository_plan(*, owner, declaration, graph, output, policy_observer):
    """Fresh owned finite evaluation joined to every original pending task."""
    _need(type(graph) is PromptGoalGraph, "exact complete native prompt goal graph required")
    declaration = _plain(declaration)
    graph_wire = _plain(graph.to_dict())
    graph = PromptGoalGraph.from_dict(graph_wire)
    payload, *_ = _declaration(declaration)
    evidence, semantic, custody, checkpoint, fence = _fresh(owner, declaration, graph_wire, output, policy_observer)
    old = (local._plain(local.admit_local_benchmark_plan(graph=graph, manifest=payload["manifest"]))
           if semantic["residual_requirement_ids"] else None)
    receipt = _receipt_payload(declaration, graph_wire, evidence, old, semantic)
    admission = _plain({"declaration": declaration, "graph": graph_wire, "evidence": evidence,
        "local_admission": old, "receipt": local._signed(receipt, payload["manifest"]["payload"])})
    verify_finite_repository_admission(admission=admission)
    custody.require_current(checkpoint)
    fence()
    return admission


def verify_current_finite_repository_admission(*, owner, admission, output, policy_observer):
    """Reobserve source and native tools; signed historical facts are not inputs."""
    verified = verify_finite_repository_admission(admission=admission)
    frozen = verified["admission"]
    evidence, semantic, custody, checkpoint, fence = _fresh(owner, frozen["declaration"], frozen["graph"],
                                                    output, policy_observer)
    _need(_same(semantic, verified["semantic_context"]), "fresh finite source/context partition differs")
    if frozen["local_admission"] is not None:
        local.verify_local_benchmark_admission(frozen["local_admission"])
    verify_finite_repository_admission(admission=frozen)
    _evidence(evidence, frozen["declaration"], frozen["graph"], checkpoint)
    custody.require_current(checkpoint)
    fence()
    return {"schema": "finite-repository-current-verification@1", "admission_cid": cid_for_structured(frozen),
        "fresh_evidence": evidence, "semantic_context": semantic, "observed_current": True, **_FALSE}


def _store(admission, manifest):
    raw = canonical_dag_json_bytes(admission)
    digest = hashlib.sha256(raw).hexdigest()
    path = local._receipt_artifact_path(manifest, digest, create=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    _need(_read(path, MAX_BYTES) == raw, "immutable finite admission artifact differs")
    return {"schema": REFERENCE_SCHEMA, "path": str(path), "sha256": digest, "bytes": len(raw),
            "admission_cid": cid_for_structured(admission), **_FALSE}


def materialize_finite_repository_plan(*, owner, admission, intent, output, policy_observer):
    """Atomically commit the complete old task population with fresh context.

    No worker launches here. Signed pending acceptance and completion guards
    remain the existing local owner's authority. Detached source and retained
    artifact fences run after all observer/signing/materialization callbacks.
    """
    _need(type(intent) is IntentRepository and not intent.uses_bound_connection,
          "independently owned native intent transaction required")
    verified = verify_finite_repository_admission(admission=admission)
    frozen = verified["admission"]
    _need(verified["receipt"]["planning_permitted"] is True and frozen["local_admission"] is not None,
          "no-work finite preview remains review-only; no empty grant or task omission")
    evidence, semantic, custody, checkpoint, fence = _fresh(owner, frozen["declaration"], frozen["graph"],
                                                    output, policy_observer)
    _need(_same(semantic, verified["semantic_context"]), "fresh finite source/context partition differs")
    manifest = frozen["declaration"]["payload"]["manifest"]["payload"]
    reference = _store(frozen, manifest)
    with intent._connection(write=True) as connection:
        with IntentRepository(bound_connection=connection, owner_id=intent.owner_id,
                              session_id=intent.session_id) as bound:
            result = local._materialize_local_benchmark_plan(admission=frozen["local_admission"], intent=bound)
            _need(result["task_cids"] == semantic["administrator_task_cids"],
                  "native commit omitted an original administrator task")
            plan = bound.get_plan(result["plan_id"])
            bound.upsert_plan(plan_cid=plan["plan_cid"], goal_cid=plan["goal_cid"],
                plan_alias=plan["plan_alias"], status=plan["status"], expected_revision=plan["revision"],
                body={**plan["body"], "finite_repository_admission_ref": reference})
            verify_finite_repository_admission(admission=frozen)
            _evidence(evidence, frozen["declaration"], frozen["graph"], checkpoint)
            _need(_read(reference["path"], MAX_BYTES, checkpoint) == canonical_dag_json_bytes(frozen),
                  "retained finite admission changed before native commit")
            custody.require_current(checkpoint)
            fence()
    return {**result, "schema": "finite-repository-native-materialization@1",
        "finite_admission_cid": reference["admission_cid"], "finite_admission_ref": reference,
        "fresh_evidence_cid": cid_for_structured(evidence), "administrator_task_population_preserved": True,
        "observed_current": True, **_FALSE}


__all__ = ["PROFILE", "DECLARATION_SCHEMA", "ADMISSION_SCHEMA", "REFERENCE_SCHEMA",
    "FiniteRepositoryAdmissionError", "author_finite_repository_declaration", "admit_finite_repository_plan",
    "verify_finite_repository_admission", "verify_current_finite_repository_admission",
    "materialize_finite_repository_plan"]
