"""Public, signed proof-query history accompanying an unchanged finite candidate.

The receiving worker checks embedded bytes and public DID signatures only.
Owner paths in original signed records remain opaque. Native inventory freshness,
claiming, publication and completion remain the owner's separate dispatch gates.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

from . import finite_repository_candidate_runner as finite_candidate
from .doctor_candidate_runner import _directory, _sha, _unique

SCHEMA = "supervisor-finite-proof-query-worker-context@1"
PROFILE = "public-finite-proof-query-history@1"
MAX_BYTES = 16 * 1024**2
DESCRIPTOR_FIELDS = frozenset({
    "artifact", "sha256", "context_cid", "finite_proof_query_admission_cid",
    "finite_proof_query_closure_cid", "finite_repository_candidate_cid",
    "task_cid", "task_revision",
})
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "publication_authority", "completion_authority",
    "task_omission_authority", "current_inventory_attested", "training_executed",
    "model_inference_executed",
)}
FIELDS = frozenset({
    "schema", "profile", "admission", "finite_proof_query_admission_cid",
    "finite_proof_query_closure_cid", "finite_repository_candidate_cid",
    "finite_admission_cid", "semantic_context_cid", "task_cid", "task_id",
    "task_revision", "original_instruction", "original_clause_ids", "edit",
    "provider_calls", "training_steps", "context_cid", *_FALSE,
})
_INDEXED_FIELDS = frozenset({
    "schema", "profile", "proof_query_closure", "proof_query_closure_cid", "match",
    "input_snapshot", "preview", "operation_catalog", "operation_catalog_cid",
    "requirement_ledger", "obligation_graph", "portfolio", "candidate_plan",
    "critique", "critic_evidence", "execution_plan", "planner_status",
    "declared_task_requirement_ids", "selected_task_ids", "current_facts_count",
    "removed_task_ids", "scope", "model_selection", "model_calls", "training_steps",
    "solver_calls", "producer", "authority", "result_cid",
})
_RECEIPT_POLICY = {
    "facts": "fresh_exact_finite_observation_only",
    "indexed_proofs": "historical_conditional_model_context_only",
    "query": "complete_singleton_full_key_current_inventory_and_epoch",
    "task_population": "complete_original_administrator_population",
    "completion": "existing_signed_pending_local_acceptance_checks",
    "model": "explicit-model-off@1", "worker_proof_query_fence": False,
}
_PAGE_FIELDS = frozenset({
    "schema", "head", "head_cid", "inventory_cid", "epoch", "selector", "selector_cid",
    "entries", "complete", "start_cursor", "next_cursor", "authority", "page_cid",
})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    """Bound inert JSON before traversing signed public records."""
    pending, count, size = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= 350_000 and depth <= 48, "bounded public proof-query JSON required")
        if type(item) is dict:
            _need(all(type(key) is str for key in item), "exact public JSON keys required")
            size += sum(len(key.encode("utf-8")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            size += len(item.encode("utf-8"))
        elif type(item) is int:
            _need(item.bit_length() <= 128, "bounded public integer required")
        else:
            _need(type(item) in {bool, type(None)}, "inert exact public scalars required")
        _need(size <= MAX_BYTES, "public proof-query text bound exceeded")
    raw = canonical_dag_json_bytes(value)
    _need(len(raw) <= MAX_BYTES, "public proof-query context exceeds its byte bound")
    return raw


def _same(left, right):
    return _wire(left) == _wire(right)


def _facts(match, declaration, semantic, before):
    """Rebuild finite facts from embedded source and retained exact integer rows."""
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
        IntegerOffsetContract, compile_integer_offset,
    )
    from ..planning import finite_integer_codebase as matcher

    document = decode_intent_ir(declaration["intent_json"])
    query = matcher.prepare_finite_integer_query(intent_document=document,
        source_text=declaration["source_text"])
    head = CodebaseHead.from_dict(declaration["head"])
    contract = IntegerOffsetContract.from_dict(query["contract"])
    observed = match["observation"]
    compiled = compile_integer_offset(before, contract, revision="snapshot:" + head.snapshot_cid, mirror=False)
    expected_rows = [{"input": n, "input_type": "int", "output": n + compiled.body_offset,
                      "output_type": "int"} for n in query["domain_inputs"]]
    _need(query["supported"] is True and _same(query, match["query"])
          and _same(query, semantic["query"]) and _same(head.to_dict(), match["head"])
          and _same(head.to_dict(), semantic["head"])
          and match["current_root_id"] == head.snapshot_cid
          and match["removed_task_ids"] == []
          and match["match_cid"] == cid_for_structured({k: v for k, v in match.items() if k != "match_cid"})
          and all(match[name] is False for name in matcher._AUTHORITY)
          and observed["status"] == "observed"
          and all(observed[name] is False for name in (
              "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
              "behavior_authority", "execution_authority", "completion_authority", "mutation_authority"))
          and observed["result_cid"] == cid_for_structured({k: v for k, v in observed.items() if k != "result_cid"})
          and observed["source_cid"] == match["source_cid"] == semantic["source_cid"] == cid_for_bytes(before)
          and observed["source_sha256"] == _sha(before)
          and observed["compiled_cid"] == compiled.cid
          and _same(observed["observations"], expected_rows)
          and _same(semantic["observations"], expected_rows)
          and _same(observed["head"], head.to_dict())
          and _same(observed["contract"], contract.to_dict())
          and observed["contract_cid"] == contract.cid
          and _same(observed["domain_inputs"], query["domain_inputs"])
          and observed["domain_cid"] == match["domain_cid"] == query["domain_cid"]
          and _same(observed["tool_policy"], declaration["tool_policy"])
          and observed["trace_cid"] == cid_for_structured(observed["trace"])
          and _same(observed["trace"]["observations"], expected_rows)
          and observed["type_clause_satisfied"] is True
          and observed["offset_clause_satisfied"] is (compiled.body_offset == contract.offset),
          "embedded finite source, domain, observation or authority differs")
    typed = matcher._typed(query, head, observed["source_cid"])
    _need(_same(match["typed_intent"], typed.to_dict())
          and _same(semantic["typed_intent"], typed.to_dict()), "public typed finite clauses differ")
    predicates = {item.predicate_id: item for item in typed.desired_predicates}
    facts, clauses, eligible, residual, examples = [], [], [], [], []
    for requirement in query["requirement_ids"]:
        predicate = predicates[typed.metadata["requirement_predicate_ids"][requirement]]
        satisfied = requirement == matcher.TYPE_STATEMENT_ID or compiled.body_offset == contract.offset
        clauses.append({"statement_id": requirement, "predicate_id": predicate.predicate_id,
            "status": "bounded_observed_satisfied" if satisfied else "finite_counterexample",
            "reasons": [] if satisfied else ["explicit_domain_observation_contradicts_requested_offset"],
            "scope": "explicit_finite_domain_only"})
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
                predicate=predicate, truth=matcher.FactTruth.TRUE,
                authority=matcher.FactAuthority.BOUNDED_OBSERVATION, provenance_refs=refs,
                current_root_id=head.snapshot_cid, invalidation_selectors=predicate.invalidation_selectors).to_dict())
        else:
            residual.append(requirement)
            examples.extend({"statement_id": requirement, "input": row["input"],
                "observed_output": row["output"], "expected_output": row["input"] + contract.offset}
                for row in expected_rows if row["output"] != row["input"] + contract.offset)
    _need(_same(match["current_facts"], facts) and _same(match["clause_results"], clauses)
          and match["eligible_clause_ids"] == match["eligible_requirements"] == eligible
          and match["residual_clause_ids"] == residual
          and _same(match["residual_requirements"], [row for row in clauses if row["statement_id"] in residual])
          and _same(match["finite_counterexamples"], examples)
          and semantic["eligible_requirement_ids"] == eligible
          and semantic["residual_requirement_ids"] == residual
          and eligible == [matcher.TYPE_STATEMENT_ID] and residual == [matcher.OFFSET_STATEMENT_ID],
          "exact public bounded facts and residual partition differ")
    return document, query, head


def _proof(closure, document, source_text, query, head, match):
    """Replay native historical objects, all keys and complete singleton pages."""
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationProjection
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import (
        CodebaseVerificationQueryRequest, CodebaseVerificationSelector,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_verification import CodebaseVerificationRecord
    from ipfs_datasets_py.logic.software_contracts.codebase_applicability import CodebaseApplicabilityRecord
    from ..planning import finite_proof_query_join as join
    from ..planning import conditional_codebase_evidence as conditional

    _need(type(closure) is dict and set(closure) == join._CLOSURE_FIELDS
          and closure["schema"] == join.CLOSURE_SCHEMA and closure["profile"] == join.PROFILE
          and closure["closure_cid"] == cid_for_structured({k: v for k, v in closure.items() if k != "closure_cid"})
          and _same(closure["head"], head.to_dict()) and closure["head_cid"] == cid_for_structured(head.to_dict())
          and closure["current_root_id"] == head.snapshot_cid
          and _same(closure["finite_query"], query) and _same(closure["authority"], join._AUTHORITY)
          and _same(closure["model_selection"], join._MODEL_OFF)
          and closure["historical_execution_attested"] is False
          and closure["current_facts"] == closure["removed_task_ids"] == [],
          "complete public historical proof closure differs")
    contract, domain = join.derive_finite_proof_spec(intent_document=document,
        source_text=source_text)
    _need(_same(closure["contract"], contract.to_dict()) and closure["contract_cid"] == cid_for_structured(contract.to_dict())
          and _same(closure["domain"], domain.to_dict()) and closure["domain_cid"] == cid_for_structured(domain.to_dict()),
          "public whole authored contract or domain differs")
    bridge = {"schema": "finite-input-to-conditional-domain-bridge@1", "finite_query_cid": query["query_cid"],
        "finite_domain_cid": query["domain_cid"], "domain_inputs": query["domain_inputs"],
        "parameter": query["contract"]["parameter"], "requested_domain_cid": closure["domain_cid"],
        "predicates": list(domain.predicates)}
    _need(_same(bridge, closure["domain_bridge"]), "public complete finite/domain bridge differs")
    verification = CodebaseVerificationRecord(closure["verification_cid"], _wire(closure["verification"]))
    applicability = CodebaseApplicabilityRecord(closure["applicability_cid"], _wire(closure["applicability"]))
    projection = CodebaseVerificationProjection(closure["projection_cid"], _wire(closure["projection"]), verification, applicability)
    request = {"path": query["contract"]["path"], "contract": contract.to_dict(), "contract_cid": closure["contract_cid"],
        "domain": domain.to_dict(), "domain_cid": closure["domain_cid"]}
    status, reasons, summary = conditional._indexed_evidence(projection, head, request)
    _need(not reasons and status == closure["conditional_status"]
          and status in {"recorded_conditional_proved", "recorded_conditional_refuted"}
          and _same(summary, closure["conditional_evidence_summary"])
          and closure["verification"]["source_binding"]["entry"]["source_cid"] == match["source_cid"]
          and closure["verification"]["source_binding"]["content_sha256"] == match["observation"]["source_sha256"]
          and closure["verification"]["requested_contracts"] == [contract.to_dict()]
          and closure["applicability"]["requested_domains"] == [domain.to_dict()],
          "public native conditional source or full obligation context differs")
    membership = join._key_membership(closure["projection"], closure["verification"], closure["applicability"], contract.contract_id)
    _need(len(membership) == 5 and _same(membership, closure["canonical_key_membership"]),
          "all five complete canonical proof keys required")
    classes = [row["differential"]["classification"] for row in closure["applicability"]["checks"]]
    _need(classes[:3] == ["agree_satisfiable", "agree_satisfiable", "agree_proved"]
          and classes[3:] in (["agree_proved"], ["agree_disproved"])
          and (classes[3] == "agree_proved") is match["observation"]["offset_clause_satisfied"],
          "public satisfiable premises, full domain coverage or property verdict differs")
    selected = next(row for row in closure["projection"]["contracts"] if row["contract_id"] == contract.contract_id)
    entry = {"entry_id": cid_for_structured({"projection_cid": closure["projection_cid"], "contract_id": contract.contract_id}),
        "contract_id": contract.contract_id, "projection_cid": closure["projection_cid"],
        "verification_cid": closure["verification_cid"], "applicability_cid": closure["applicability_cid"],
        "path": closure["projection"]["path"], "contract_cid": selected["contract_cid"],
        "domain_id": selected["domain_id"], "domain_cid": selected["domain_cid"],
        "canonical_key_ids": sorted({row["key_id"] for row in membership})}
    pages = []
    for phase in ("discovery_query", "exact_query"):
        wrapper = closure[phase]
        selector = CodebaseVerificationSelector(path=query["contract"]["path"], contract_id=contract.contract_id,
            expected_contract_cid=closure["contract_cid"], requested_domain_id=domain.domain_id,
            requested_domain_cid=closure["domain_cid"], **({"verification_cid": closure["verification_cid"],
                "canonical_key_id": membership[0]["key_id"]} if phase == "exact_query" else {}))
        page = wrapper["page"]
        _need(set(wrapper) == {"request", "page"}
              and type(page) is dict and set(page) == _PAGE_FIELDS
              and _same(wrapper["request"], CodebaseVerificationQueryRequest(selector, 2).to_dict())
              and page["schema"] == "codebase-verification-query-page@1"
              and _same(page["selector"], selector.to_dict()) and page["selector_cid"] == selector.cid
              and page["complete"] is True and page["start_cursor"] is page["next_cursor"] is None
              and type(page["epoch"]) is int and page["epoch"] >= 1
              and _same(page["head"], head.to_dict()) and page["head_cid"] == closure["head_cid"]
              and _same(page["entries"], [entry])
              and page["page_cid"] == cid_for_structured({k: v for k, v in page.items() if k != "page_cid"})
              and page["authority"]["historical_conditional_evidence"] is True
              and all(page["authority"][name] is False for name in (
                  "kernel_checked", "source_runtime_semantics_verified", "behavioral_satisfaction",
                  "authoritative_cache_eligible", "admission_authority", "completion_authority")),
              "public complete singleton proof query differs")
        pages.append(page)
    _need(pages[0]["epoch"] == pages[1]["epoch"] and pages[0]["inventory_cid"] == pages[1]["inventory_cid"],
          "public discovery/exact inventory generations differ")


def verify_finite_proof_query_worker_context(*, context, candidate):
    """Pure public historical verification; no owner keys, paths or callbacks."""
    from ..planning import finite_proof_query_join as join
    from ..planning import finite_integer_plan_preview as adapter
    from ..planning.plan_revision_contracts import PlanCreateRequest
    from ..planning.structural_codebase_context import StructuralCodebaseContext
    from ..prompt.plan_create_service import freeze_plan_create_input_snapshot

    value = json.loads(_wire(context), object_pairs_hook=_unique)
    candidate = json.loads(_wire(candidate), object_pairs_hook=_unique)
    base, manifest, graph, semantic, before, _ = finite_candidate._validate(candidate, mirror=False)
    signer = base["declaration"]["binding"]
    payload = finite_candidate._public(value, signer, mirror=False)
    _need(type(payload) is dict and set(payload) == FIELDS and payload["schema"] == SCHEMA
          and payload["profile"] == PROFILE and all(payload[name] is False for name in _FALSE)
          and type(payload["provider_calls"]) is int and payload["provider_calls"] == 0
          and type(payload["training_steps"]) is int and payload["training_steps"] == 0
          and type(payload["task_revision"]) is int and payload["task_revision"] >= 1
          and payload["context_cid"] == cid_for_structured({k: v for k, v in payload.items() if k != "context_cid"}),
          "closed signed public context identity or authority differs")
    admission = payload["admission"]
    _need(type(admission) is dict and set(admission) == {"schema", "finite_admission", "indexed_plan", "receipt"}
          and admission["schema"] == "finite-proof-query-admission-bundle@1"
          and _same(admission["finite_admission"], base)
          and payload["finite_proof_query_admission_cid"] == cid_for_structured(admission)
          and payload["finite_repository_candidate_cid"] == candidate["candidate_cid"]
          and payload["finite_admission_cid"] == candidate["finite_admission_cid"]
          and payload["semantic_context_cid"] == candidate["semantic_context_cid"]
          and all(_same(payload[key], candidate[key]) for key in ("task_cid", "task_id", "task_revision", "edit"))
          and payload["original_instruction"] == candidate["original_prompt"]
          and _same(payload["original_clause_ids"], candidate["original_clause_ids"]),
          "public context belongs to a different candidate, task, revision or admission")
    indexed = admission["indexed_plan"]
    _need(type(indexed) is dict and set(indexed) == _INDEXED_FIELDS | frozenset(join._AUTHORITY)
          and indexed["schema"] == join.SCHEMA and indexed["profile"] == join.PROFILE
          and indexed["result_cid"] == cid_for_structured({k: v for k, v in indexed.items() if k != "result_cid"})
          and _same(indexed["match"], base["evidence"]["match"])
          and _same(indexed["authority"], join._AUTHORITY)
          and all(indexed[name] is False for name in join._AUTHORITY)
          and _same(indexed["model_selection"], join._MODEL_OFF)
          and all(type(indexed[name]) is int and indexed[name] == 0 for name in ("model_calls", "training_steps", "solver_calls"))
          and indexed["removed_task_ids"] == [], "complete public model-off indexed plan differs")
    receipt = finite_candidate._public(admission["receipt"], signer, mirror=False)
    required = {"schema", "profile", "finite_admission_cid", "declaration_cid", "graph_cid", "evidence_cid",
        "semantic_context_cid", "indexed_plan_cid", "input_snapshot_cid", "proof_query_closure_cid",
        "administrator_task_cids", "planning_permitted", "no_work_review_only", "policy", "implementation"}
    false_names = ("proof_authority", "behavior_authority", "source_semantics_verified", "runtime_behavior_verified",
        "execution_authority", "completion_authority", "omission_authority", "mutation_authority", "worker_launched",
        "training_executed", "model_inference_executed")
    _need(set(receipt) == required | set(false_names)
          and receipt["schema"] == "finite-proof-query-admission@1"
          and receipt["profile"] == "finite-observation-with-current-proof-query@1"
          and all(receipt[name] is False for name in false_names)
          and receipt["planning_permitted"] is True and receipt["no_work_review_only"] is False
          and _same(receipt["policy"], _RECEIPT_POLICY)
          and receipt["finite_admission_cid"] == cid_for_structured(base)
          and all(receipt[name + "_cid"] == cid_for_structured(base[name]) for name in ("declaration", "graph", "evidence"))
          and receipt["semantic_context_cid"] == cid_for_structured(semantic)
          and receipt["indexed_plan_cid"] == indexed["result_cid"]
          and receipt["input_snapshot_cid"] == indexed["input_snapshot"]["snapshot_cid"]
          and receipt["proof_query_closure_cid"] == indexed["proof_query_closure_cid"]
          and _same(receipt["administrator_task_cids"], semantic["administrator_task_cids"]),
          "public proof-query signature does not bind the full finite/indexed inputs")
    declaration = base["declaration"]["payload"]
    document, query, head = _facts(indexed["match"], declaration, semantic, before)
    closure = indexed["proof_query_closure"]
    _need(payload["finite_proof_query_closure_cid"] == indexed["proof_query_closure_cid"] == closure["closure_cid"],
          "public complete proof closure pin differs")
    _proof(closure, document, declaration["source_text"], query, head, indexed["match"])
    context_record = indexed["match"]["structural_context"]
    structural = StructuralCodebaseContext(head, context_record["semantic_state_cid"], context_record["coverage"])
    _need(_same(structural.to_dict(), context_record), "public structural context differs")
    operations = adapter.FiniteIntegerOperationCatalog(tuple(
        adapter.ReviewedFiniteIntegerOperation(**row) for row in declaration["operation_catalog"]["operations"]))
    _need(_same(operations.to_dict(), declaration["operation_catalog"])
          and _same(indexed["operation_catalog"], operations.to_dict())
          and indexed["operation_catalog_cid"] == operations.cid, "public whole operation catalog differs")
    ledger = [{**row, "predicate_id": indexed["match"]["typed_intent"]["metadata"]["requirement_predicate_ids"][row["statement_id"]],
        "fact_ids": [fact["fact_id"] for fact in indexed["match"]["current_facts"]
                     if fact["predicate"]["predicate_id"] == row["predicate_id"]]}
        for row in indexed["match"]["clause_results"]]
    selected = [operation.task_id for operation in operations.operations
        if operation.requirement_id in semantic["residual_requirement_ids"]]
    _need(_same(ledger, indexed["requirement_ledger"])
          and _same(indexed["declared_task_requirement_ids"],
              {operation.task_id: operation.requirement_id for operation in operations.operations})
          and _same(indexed["selected_task_ids"], selected)
          and _same(indexed["match"]["domain_inputs"], query["domain_inputs"])
          and indexed["planner_status"] == "selected", "public full requirement ledger or selected residual differs")
    request = PlanCreateRequest.from_dict(declaration["request"])
    materials, _ = adapter._materials(indexed["match"], operations, structural)
    materials = replace(materials, extra={**materials.extra, join.MATERIAL_KEY: closure,
        join.MATERIAL_KEY + "_cid": closure["closure_cid"], "finite_proof_query_profile": join.PROFILE})
    _need(_same(freeze_plan_create_input_snapshot(request, materials=materials).to_dict(), indexed["input_snapshot"])
          and len(graph.tasks) == 2
          and sorted(task.task_cid for task in graph.tasks) == semantic["administrator_task_cids"]
          and type(indexed["current_facts_count"]) is int and indexed["current_facts_count"] == 1,
          "public frozen full planning input, original population or fact count differs")
    return value


def _read_context(artifact):
    path = Path(artifact).absolute()
    _need(path.resolve(strict=True) == path, "exact non-symlink proof context artifact required")
    parent = _directory(path.parent)
    try:
        _need(not os.fstat(parent).st_mode & 0o022, "proof context directory must forbid other writers")
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and not before.st_mode & 0o222 and before.st_size <= MAX_BYTES,
                  "bounded readonly single-link proof context required")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(fd)
            _need(len(raw) <= MAX_BYTES and all(getattr(before, key) == getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
                "proof context changed during immutable read")
        finally:
            os.close(fd)
    finally:
        os.close(parent)
    return path, raw


def load_finite_proof_query_worker_context(*, artifact, expected_sha256, expected_context_cid, candidate):
    """Read pinned public context; embedded owner artifact paths are not opened."""
    path, raw = _read_context(artifact)
    _need(type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
          and _sha(raw) == expected_sha256, "pinned proof context digest differs")
    value = json.loads(raw, object_pairs_hook=_unique)
    _need(_wire(value) == raw, "canonical proof context bytes required")
    verified = verify_finite_proof_query_worker_context(context=value, candidate=candidate)
    _need(verified["payload"]["context_cid"] == expected_context_cid
          and not path.is_relative_to(Path(candidate["repository"])), "proof context identity or location differs")
    return verified


def validate_finite_proof_query_worker_descriptor(*, descriptor, candidate):
    _need(type(descriptor) is dict and set(descriptor) == DESCRIPTOR_FIELDS,
          "closed proof-query worker descriptor required")
    value = load_finite_proof_query_worker_context(artifact=descriptor["artifact"],
        expected_sha256=descriptor["sha256"], expected_context_cid=descriptor["context_cid"], candidate=candidate)
    payload = value["payload"]
    expected = {"artifact": descriptor["artifact"], "sha256": descriptor["sha256"],
        **{name: payload[name] for name in DESCRIPTOR_FIELDS - {"artifact", "sha256"}}}
    _need(_same(expected, descriptor), "proof context descriptor task or complete body binding differs")
    return value


def author_finite_proof_query_worker_context(*, owner, admission, candidate_descriptor, output):
    """Owner-only signing; authenticate originals without altering signed fields."""
    from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from . import finite_proof_query_admission as proof_admission
    from . import local_planning_admission as local

    _need(type(owner) is RepositoryPlanPreviewOwner, "exact proof-query context owner required")
    _need(type(candidate_descriptor) is dict and set(candidate_descriptor) == {
        "artifact", "sha256", "candidate_cid", "finite_admission_cid", "semantic_context_cid",
        "task_cid", "task_id", "task_revision", "before_sha256", "after_sha256"},
        "closed original finite candidate descriptor required")
    value, verified = proof_admission._received(admission)
    candidate = finite_candidate.load_finite_repository_candidate(artifact=Path(candidate_descriptor["artifact"]),
        expected_sha256=candidate_descriptor["sha256"], mirror=False)
    _need(_same(candidate["finite_admission"], value["finite_admission"])
          and _same(owner.expected_head.to_dict(), verified["semantic_context"]["head"])
          and all(_same(candidate_descriptor[key], candidate[key]) for key in (
              "candidate_cid", "finite_admission_cid", "semantic_context_cid", "task_cid", "task_id", "task_revision"))
          and all(candidate_descriptor[name + "_sha256"] == candidate["edit"][name + "_sha256"]
              for name in ("before", "after")),
          "owner context and complete original finite candidate differ")
    payload = {"schema": SCHEMA, "profile": PROFILE, "admission": value,
        "finite_proof_query_admission_cid": cid_for_structured(value),
        "finite_proof_query_closure_cid": value["indexed_plan"]["proof_query_closure_cid"],
        "finite_repository_candidate_cid": candidate["candidate_cid"],
        **{key: candidate[key] for key in ("finite_admission_cid", "semantic_context_cid", "task_cid", "task_id", "task_revision", "edit")},
        "original_instruction": candidate["original_prompt"], "original_clause_ids": candidate["original_clause_ids"],
        "provider_calls": 0, "training_steps": 0, **_FALSE}
    payload["context_cid"] = cid_for_structured(payload)
    envelope = local._signed(payload, candidate["finite_admission"]["declaration"]["payload"]["manifest"]["payload"])
    verify_finite_proof_query_worker_context(context=envelope, candidate=candidate)
    path = Path(output).absolute()
    _need(not path.is_relative_to(Path(candidate["repository"])), "proof context must remain outside the repository")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _need(path.parent.resolve(strict=True) == path.parent, "canonical proof context output directory required")
    raw = _wire(envelope)
    parent = _directory(path.parent)
    try:
        _need(not os.fstat(parent).st_mode & 0o022, "proof context output directory forbids other writers")
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fchmod(fd, 0o444)
                os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(parent)
    finally:
        os.close(parent)
    descriptor = {"artifact": str(path), "sha256": _sha(raw),
        **{key: payload[key] for key in DESCRIPTOR_FIELDS - {"artifact", "sha256"}}}
    proof_admission._received(value)
    _need(_same(candidate, finite_candidate.load_finite_repository_candidate(
        artifact=Path(candidate_descriptor["artifact"]), expected_sha256=candidate_descriptor["sha256"], mirror=False)),
        "finite candidate changed during proof context signing")
    validate_finite_proof_query_worker_descriptor(descriptor=descriptor, candidate=candidate)
    return descriptor


def extend_candidate_binding(*, binding, context):
    """Append exactly three public pins to the unchanged native finite command."""
    _need(type(binding) is dict and set(binding) == {"descriptor", "argv", "implementation_command"},
          "exact original finite candidate launch binding required")
    descriptor = binding["descriptor"]
    candidate = finite_candidate.load_finite_repository_candidate(artifact=Path(descriptor["artifact"]),
        expected_sha256=descriptor["sha256"], mirror=False)
    validate_finite_proof_query_worker_descriptor(descriptor=context, candidate=candidate)
    original = ["/opt/ipfs-supervisor/bin/owner-worker", "--finite-repository-artifact", descriptor["artifact"],
        "--finite-repository-sha256", descriptor["sha256"], "--finite-repository-task-cid", candidate["task_cid"]]
    _need(_same(binding["argv"], original) and binding["implementation_command"] == shlex.join(original),
          "immutable original finite command differs")
    argv = [*original, "--finite-proof-query-context", context["artifact"],
        "--finite-proof-query-sha256", context["sha256"], "--finite-proof-query-context-cid", context["context_cid"]]
    return {"descriptor": json.loads(_wire(descriptor)), "worker_context": json.loads(_wire(context)),
        "argv": argv, "implementation_command": shlex.join(argv)}


def materialize_finite_proof_query_candidate(*, artifact, expected_sha256, task_cid, context,
        context_sha256, context_cid, prompt, workspace):
    candidate = finite_candidate.load_finite_repository_candidate(artifact=Path(artifact), expected_sha256=expected_sha256, mirror=False)
    _need(candidate["task_cid"] == task_cid, "proof context worker task differs")
    original = load_finite_proof_query_worker_context(artifact=context, expected_sha256=context_sha256,
        expected_context_cid=context_cid, candidate=candidate)
    _need(not Path(context).absolute().is_relative_to(Path(workspace).absolute()),
          "proof context must remain outside the allocated worktree")
    result = finite_candidate.materialize_finite_repository_candidate(artifact=Path(artifact),
        expected_sha256=expected_sha256, task_cid=task_cid, prompt=prompt, workspace=Path(workspace), mirror=False)
    _need(_same(original, load_finite_proof_query_worker_context(artifact=context,
        expected_sha256=context_sha256, expected_context_cid=context_cid, candidate=candidate)),
        "proof context changed during finite materialization")
    return {**result, "schema": "native-finite-proof-query-candidate-materialization@1",
        "proof_query_context_cid": context_cid,
        "finite_proof_query_admission_cid": original["payload"]["finite_proof_query_admission_cid"],
        "finite_proof_query_closure_cid": original["payload"]["finite_proof_query_closure_cid"],
        "current_inventory_attested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-cid", required=True)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--context-cid", required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_finite_proof_query_candidate(artifact=args.artifact,
            expected_sha256=args.sha256, task_cid=args.task_cid, context=args.context,
            context_sha256=args.context_sha256, context_cid=args.context_cid,
            prompt=sys.stdin.buffer.read(256_001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps({"schema": "native-finite-proof-query-candidate-materialization@1",
            "status": "refused", "error_type": type(error).__name__, **_FALSE}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "PROFILE", "FIELDS", "DESCRIPTOR_FIELDS",
    "author_finite_proof_query_worker_context", "verify_finite_proof_query_worker_context",
    "load_finite_proof_query_worker_context", "validate_finite_proof_query_worker_descriptor",
    "extend_candidate_binding", "materialize_finite_proof_query_candidate", "main"]
