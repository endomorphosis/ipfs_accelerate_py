"""Current, indexed conditional mathematical evidence for explicit native intent.

This lookup-only profile joins a reviewed authored Int/Bool atom to an exact
source head, contract and requested input domain. Review labels describe the
authored interpretation; they do not attest review custody or Python behavior.
Stored process observations remain recorded claims. All runtime requirements
remain residual, and this module cannot admit a plan or omit work.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time
from types import CodeType, FunctionType
from typing import Any

SCHEMA = "supervisor-conditional-codebase-evidence@1"
QUERY_SCHEMA = "supervisor-conditional-codebase-query@1"
PROFILE = "typed-straight-line-int-bool-linear@1"
PREDICATE = "conditional_int_bool_property"
MAX_JSON_BYTES = 2 * 1024 * 1024
_IMPORTED_SOURCE_PIN: str | None = None
_AUTHORITY = {
    "semantic_alignment_verified": False,
    "source_semantics_verified": False,
    "runtime_behavior_verified": False,
    "kernel_checked": False,
    "proof_authority": False,
    "execution_authority": False,
    "completion_authority": False,
    "mutation_authority": False,
    "admission_authority": False,
    "authoritative_cache_eligible": False,
    "behavioral_satisfaction": False,
}


class ConditionalCodebaseEvidenceError(ValueError):
    """A native authored query or indexed binding is malformed or inconsistent."""


def _json(value: Any) -> Any:
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 40_000 or depth > 32:
            raise ConditionalCodebaseEvidenceError("bounded inert query JSON required")
        if type(item) is dict:
            if len(item) > 40_000 - count or any(type(key) is not str for key in item):
                raise ConditionalCodebaseEvidenceError("bounded string-keyed query objects required")
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) > 40_000 - count:
                raise ConditionalCodebaseEvidenceError("query collection limit exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_JSON_BYTES:
                raise ConditionalCodebaseEvidenceError("query string limit exceeded")
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int:
            if item.bit_length() > 128:
                raise ConditionalCodebaseEvidenceError("query integer limit exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ConditionalCodebaseEvidenceError("finite query numbers required")
        elif type(item) not in {bool, type(None)}:
            raise ConditionalCodebaseEvidenceError("exact inert JSON values required")
        if text_bytes > MAX_JSON_BYTES:
            raise ConditionalCodebaseEvidenceError("query text limit exceeded")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ConditionalCodebaseEvidenceError("finite bounded query JSON required") from error
    if len(raw) > MAX_JSON_BYTES:
        raise ConditionalCodebaseEvidenceError("query byte limit exceeded")
    return json.loads(raw)


def _producer_pin() -> dict[str, str]:
    global _IMPORTED_SOURCE_PIN
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    if len(raw) > 256 * 1024:
        raise ConditionalCodebaseEvidenceError("matcher implementation size limit exceeded")
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    codes = {item.co_name: item for item in compiled.co_consts if type(item) is CodeType}
    for name, value in globals().copy().items():
        if type(value) is FunctionType and value.__module__ == __name__:
            if codes.get(name) != value.__code__:
                raise ConditionalCodebaseEvidenceError("loaded matcher differs from installed source")
    digest = hashlib.sha256(raw).hexdigest()
    if _IMPORTED_SOURCE_PIN is not None and _IMPORTED_SOURCE_PIN != digest:
        raise ConditionalCodebaseEvidenceError("matcher implementation generation changed")
    _IMPORTED_SOURCE_PIN = digest
    return {"module": __name__, "sha256": digest, "profile": PROFILE}


def _owned_request(path: Any, contract: Any, domain: Any) -> tuple[Any, Any, str, str]:
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec
    from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain

    if (type(path) is not str or not path or len(path.encode("utf-8")) > 4096
            or PurePosixPath(path).is_absolute() or PurePosixPath(path).as_posix() != path
            or ".." in PurePosixPath(path).parts or not path.endswith(".py")):
        raise ConditionalCodebaseEvidenceError("canonical captured relative Python path required")
    if type(contract) is not ContractSpec or type(domain) is not RequestedInputDomain:
        raise ConditionalCodebaseEvidenceError("exact native ContractSpec and RequestedInputDomain required")
    raw_contract = _json(contract.to_dict())
    raw_domain = _json(domain.to_dict())
    try:
        owned_contract = ContractSpec(**raw_contract)
        owned_domain = RequestedInputDomain.from_dict(raw_domain)
    except (ValueError, TypeError) as error:
        raise ConditionalCodebaseEvidenceError("invalid native contract or domain") from error
    if (owned_contract.to_dict() != raw_contract or owned_domain.to_dict() != raw_domain
            or owned_contract.function_name != owned_domain.function_name):
        raise ConditionalCodebaseEvidenceError("native request fields or function binding differ")
    conditions = owned_contract.preconditions + owned_contract.postconditions + owned_domain.predicates
    if (not owned_contract.postconditions or len(conditions) > 64
            or any(len(item.encode("utf-8")) > 16 * 1024 for item in conditions)):
        raise ConditionalCodebaseEvidenceError("bounded explicit postcondition and input domain required")
    return owned_contract, owned_domain, cid_for_structured(raw_contract), cid_for_structured(raw_domain)


def prepare_conditional_codebase_query(
    *, intent_document: Any, source_text: str, path: str, contract: Any,
    domain: Any, statement_id: str, source_identity: Any = None,
) -> dict[str, Any]:
    """Bind a reviewed native mathematical atom; perform no scans or solving.

    The atom's ordered arguments are ``(path, function_name, contract_cid,
    domain_cid, PROFILE)``. CIDs identify the exact authored ContractSpec and
    RequestedInputDomain dictionaries, not a lowered ProgramContract. Every
    source/clause remains in the returned ledger, including unselected goals.
    Review status is an interpretation label, never an admission credential.
    """
    from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, NodeGrounding, ReviewStatus, StatementKind,
    )
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    contract, domain, contract_cid, domain_cid = _owned_request(path, contract, domain)
    if type(source_text) is not str or not source_text or len(source_text) > 128 * 1024:
        raise ConditionalCodebaseEvidenceError("bounded exact original intent text required")
    try:
        raw_source = source_text.encode("utf-8")
    except UnicodeError as error:
        raise ConditionalCodebaseEvidenceError("valid UTF-8 intent text required") from error
    if len(raw_source) > 128 * 1024:
        raise ConditionalCodebaseEvidenceError("intent source byte limit exceeded")
    if type(statement_id) is not str or not statement_id:
        raise ConditionalCodebaseEvidenceError("one explicit native statement ID required")
    raw_document = intent_document.to_dict() if type(intent_document) is IntentIRDocument else intent_document
    try:
        native = decode_intent_ir(_json(raw_document))
        native.validate()
    except (ValueError, TypeError, KeyError) as error:
        raise ConditionalCodebaseEvidenceError("valid exact native IntentIR required") from error
    if len(native.sources) > 64 or len(native.statements) > 256 or len(native.actions) > 256:
        raise ConditionalCodebaseEvidenceError("native intent inventory limit exceeded")
    selected = next((item for item in native.statements if item.statement_id == statement_id), None)
    if selected is None:
        raise ConditionalCodebaseEvidenceError("selected native statement does not exist")
    identities = [item.to_dict() for item in sorted(native.sources, key=lambda item: item.ref_id)]
    if source_identity is not None and _json(source_identity) != identities:
        raise ConditionalCodebaseEvidenceError("complete native source identity ledger differs")
    digest = hashlib.sha256(raw_source).hexdigest()
    source_by_ref = {item.ref_id: item for item in native.sources}
    for item in native.sources:
        if (item.content_sha256 != digest or item.span is None
                or not 0 <= item.span.start_char < item.span.end_char <= len(source_text)):
            raise ConditionalCodebaseEvidenceError("source digest or nonempty clause span differs from original text")
    refs = [source_by_ref[ref] for ref in selected.source_ref_ids]
    if not refs:
        raise ConditionalCodebaseEvidenceError("selected mathematical atom requires original source references")
    expected_arguments = [path, contract.function_name, contract_cid, domain_cid, PROFILE]
    reasons = []
    if (native.intent_kind is not IntentKind.DECLARATIVE or native.actions or native.control_edges
            or native.entry_action_ids or native.terminal_action_ids):
        reasons.append("declarative_native_mathematical_intent_required")
    if (selected.kind is not StatementKind.GOAL or selected.modality is not IntentModality.REQUIRED
            or selected.grounding is not NodeGrounding.GROUNDED):
        reasons.append("grounded_required_mathematical_goal_required")
    if (selected.review_status is not ReviewStatus.HUMAN_REVIEWED
            or any(ref.review_status is not ReviewStatus.HUMAN_REVIEWED for ref in refs)):
        reasons.append("explicit_reviewed_native_interpretation_required")
    if selected.predicate != PREDICATE or list(selected.arguments) != expected_arguments:
        reasons.append("exact_native_property_contract_domain_binding_required")
    # Native IntentIR confidence/metadata may contain finite JSON floats. Keep
    # its complete native canonical bytes as an explicit string in the stricter
    # software-contract CAS envelope; do not discard or reinterpret them.
    native_bytes = canonical_intent_ir_bytes(native)
    ledger = [{"reference_json": json.dumps(item.to_dict(), sort_keys=True,
                   separators=(",", ":"), ensure_ascii=False, allow_nan=False),
               "original_text": source_text[item.span.start_char:item.span.end_char]}
              for item in sorted(native.sources, key=lambda item: item.ref_id)]
    query = {
        "schema": QUERY_SCHEMA, "profile": PROFILE, "path": path,
        "contract": contract.to_dict(), "contract_cid": contract_cid,
        "domain": domain.to_dict(), "domain_cid": domain_cid,
        "intent_document_json": native_bytes.decode("utf-8"), "intent_source_text": source_text,
        "intent_source_sha256": digest, "intent_source_ledger": ledger,
        "native_document_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "statement_id": statement_id,
        "statement": {"statement_id": selected.statement_id, "predicate": selected.predicate,
                      "arguments": list(selected.arguments), "kind": selected.kind.value,
                      "modality": selected.modality.value, "source_ref_ids": list(selected.source_ref_ids)},
        "requirement_ids": sorted(item.statement_id for item in native.statements),
        "supported": not reasons, "reasons": reasons,
        "interpretation_scope": "explicit_reviewed_native_mathematical_atom_only",
        "review_custody_verified": False, "free_text_semantics_verified": False,
        "producer": _producer_pin(), **_AUTHORITY,
    }
    query["query_cid"] = cid_for_structured(query)
    return _json(query)


def _indexed_evidence(projection: Any, head: Any, query: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationProjection
    from ipfs_datasets_py.logic.software_contracts.codebase_verification import CodebaseVerificationRecord
    from ipfs_datasets_py.logic.software_contracts.codebase_applicability import CodebaseApplicabilityRecord
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    if type(projection) is not CodebaseVerificationProjection or type(projection.verification) is not CodebaseVerificationRecord:
        raise ConditionalCodebaseEvidenceError("native indexed projection and replay record required")
    value = projection.to_dict()
    parent = projection.verification.to_dict()
    if (cid_for_structured(value) != projection.projection_cid
            or cid_for_structured(parent) != projection.verification.artifact_cid
            or value["verification_cid"] != projection.verification.artifact_cid
            or projection.verification.observed_live is not False
            or value["head"] != head.to_dict() or value["path"] != query["path"]
            or parent["source_binding"]["head"] != head.to_dict()
            or parent["source_binding"]["path"] != query["path"]
            or value["source_binding"] != parent["source_binding"]):
        raise ConditionalCodebaseEvidenceError("indexed evidence lost its exact content/head/source binding")
    requested = [item for item in parent["requested_contracts"] if item["contract_id"] == query["contract"]["contract_id"]]
    selected = [item for item in value["contracts"] if item["contract_id"] == query["contract"]["contract_id"]]
    if (requested != [query["contract"]] or len(selected) != 1
            or selected[0]["requested_contract"] != query["contract"]
            or selected[0]["contract_cid"] != query["contract_cid"]):
        raise ConditionalCodebaseEvidenceError("indexed authored contract differs from the native query")
    selected = selected[0]
    authority_names = ("kernel_checked", "source_runtime_semantics_verified", "behavioral_satisfaction",
                       "authoritative_cache_eligible", "admission_authority", "completion_authority")
    if any(parent["authority"].get(name) is not False for name in authority_names):
        raise ConditionalCodebaseEvidenceError("recorded conditional evidence claims elevated authority")
    result = parent["pipeline_result"]
    solved = [item for item in result["obligation_results"]
              if item["vc_obligation"]["parent_contract_id"] == query["contract"]["contract_id"]]
    parent_status = ("recorded_conditional_refuted" if any(item["verdict_classification"] == "agree_disproved" for item in solved)
                     else "recorded_conditional_proved" if solved and all(
                         item["verdict_classification"] == "agree_proved" for item in solved)
                     and result["status"] == "success" else "unknown")
    summary = {
        "projection_cid": projection.projection_cid,
        "verification_cid": projection.verification.artifact_cid,
        "applicability_cid": value["applicability_cid"],
        "head_cid": cid_for_structured(head.to_dict()),
        "source_binding": parent["source_binding"], "contract_cid": query["contract_cid"],
        "lowered_contract_cid": selected["lowered_contract_cid"],
        "domain_cid": query["domain_cid"], "canonical_keys": selected["canonical_keys"],
        "parent_contract_status": parent_status,
        "recorded_execution_attested": False, "historical_records_observed_live": False,
    }
    if projection.applicability is None:
        return "unknown", ["requested_domain_applicability_not_indexed"], summary
    if type(projection.applicability) is not CodebaseApplicabilityRecord:
        raise ConditionalCodebaseEvidenceError("native linked applicability replay record required")
    app = projection.applicability.to_dict()
    if (cid_for_structured(app) != projection.applicability.artifact_cid
            or value["applicability_cid"] != projection.applicability.artifact_cid
            or app["verification_cid"] != projection.verification.artifact_cid
            or app["source_binding"] != parent["source_binding"]
            or projection.applicability.observed_live is not False
            or any(app["authority"].get(name) is not False for name in authority_names)):
        raise ConditionalCodebaseEvidenceError("linked applicability content/source/authority differs")
    domains = [item for item in app["requested_domains"] if item["function_name"] == query["contract"]["function_name"]]
    if (domains != [query["domain"]] or selected["domain_id"] != query["domain"]["domain_id"]
            or selected["domain_cid"] != query["domain_cid"]):
        raise ConditionalCodebaseEvidenceError("indexed requested domain differs from the complete authored predicates")
    rows = [item for item in app["contract_results"]
            if item["parent_contract_id"] == query["contract"]["contract_id"]]
    checks = [item for item in app["checks"]
              if item["parent_contract_id"] == query["contract"]["contract_id"]]
    kinds = ["premises_satisfiable", "domain_satisfiable", "domain_implies_preconditions", "property_on_requested_domain"]
    binding = {"parent_contract_id": query["contract"]["contract_id"],
        "contract_cid": selected["lowered_contract_cid"], "function_name": query["contract"]["function_name"],
        "domain_id": query["domain"]["domain_id"], "domain_cid": query["domain_cid"]}
    if (len(rows) != 1 or [item["kind"] for item in checks] != kinds
            or any(any(item.get(key) != expected for key, expected in binding.items()) for item in [*rows, *checks])):
        raise ConditionalCodebaseEvidenceError("selected applicability contract/domain/check inventory differs")
    row = rows[0]
    classes = [item["differential"]["classification"] for item in checks]
    gates = classes[:3] == ["agree_satisfiable", "agree_satisfiable", "agree_proved"]
    if (row["classifications"] != classes or row["model_applicability_established"] is not gates
            or row["domain_property_classification"] != classes[3]
            or row["conditional_proved"] is not (gates and classes[3] == "agree_proved")
            or row["conditional_refuted"] is not (gates and classes[3] == "agree_disproved")):
        raise ConditionalCodebaseEvidenceError("recorded applicability summary does not bind its native checks")
    summary["recorded_domain_checks"] = {"contract_result": row,
        "check_obligation_ids": [item["obligation"]["smt_obligation"]["obligation_id"] for item in checks],
        "canonical_keys": [key for check, key in zip(app["checks"], app["canonical_keys"])
                           if check["parent_contract_id"] == binding["parent_contract_id"]]}
    if "disagree" in classes:
        return "unknown", ["recorded_native_solver_disagreement"], summary
    if row["status"] in {"inconsistent_premises", "empty_domain"}:
        return "recorded_conditional_vacuous", ["recorded_" + row["status"]], summary
    if row["status"] == "domain_not_covered":
        return "recorded_conditional_domain_not_covered", ["requested_domain_not_within_contract_preconditions"], summary
    if gates:
        if row["conditional_proved"]:
            return "recorded_conditional_proved", [], summary
        if row["conditional_refuted"]:
            return "recorded_conditional_refuted", [], summary
        return "recorded_conditional_applicable", ["requested_domain_property_inconclusive"], summary
    return "unknown", ["recorded_applicability_" + row["status"]], summary


def match_conditional_codebase_intent(
    *, catalog: Any, index: Any, repository: Any, repository_id: str, expected_head: Any,
    intent_document: Any, source_text: str, path: str, contract: Any, domain: Any,
    statement_id: str, source_identity: Any = None, verification_cid: str | None = None,
    expected_key_id: str | None = None, scheduler: Any = None, parent_lease: Any = None,
    cancel_event: Any = None, admission_timeout_seconds: float = 30.0,
    timeout_seconds: float = 120.0, memory_mb: int = 512,
) -> dict[str, Any]:
    """Resolve exact current indexed claims; never train, solve, admit or dispatch.

    Producers explicitly verify the source, check domain applicability and publish
    those CIDs before this call. One bounded page distinguishes a unique exact
    projection from missing, ambiguous or incomplete evidence. Only a complete
    singleton is classified. Lookup replays native captured-source identities and
    recorded solver output parsing, not solver processes. Entry/exit source checks
    do not lock the checkout; later admission needs its own current observation.
    """
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import (
        CodebaseVerificationSelector, CodebaseVerificationQueryPage, CodebaseVerificationQueryEntry,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
    from .structural_codebase_context import run_with_structural_codebase_context

    if (type(index) is not RepositoryCodebaseIndex or type(catalog) is not CodebaseVerificationCatalog
            or catalog.index is not index or type(expected_head) is not CodebaseHead):
        raise ConditionalCodebaseEvidenceError("same native index/catalog and exact current head required")
    head = CodebaseHead.from_dict(expected_head.to_dict())
    if head.repository_id != repository_id:
        raise ConditionalCodebaseEvidenceError("expected head does not bind this repository view")
    query = prepare_conditional_codebase_query(intent_document=intent_document, source_text=source_text,
        path=path, contract=contract, domain=domain, statement_id=statement_id, source_identity=source_identity)
    if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConditionalCodebaseEvidenceError("finite positive overall timeout required")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise ConditionalCodebaseEvidenceError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("conditional intent lookup cancelled")
        value = deadline - time.monotonic()
        if value <= 0:
            raise LeaseTimeoutError("conditional intent lookup deadline exceeded")
        return value

    def build(context: Any) -> dict[str, Any]:
        status, evidence, reasons = "unknown", None, list(query["reasons"])
        indexed_query_page = None
        if query["supported"]:
            selector = CodebaseVerificationSelector(path=path,
                contract_id=query["contract"]["contract_id"], verification_cid=verification_cid,
                expected_contract_cid=query["contract_cid"], canonical_key_id=expected_key_id,
                requested_domain_id=query["domain"]["domain_id"], requested_domain_cid=query["domain_cid"])
            page = catalog.query_current(repository, expected_head=head, selector=selector,
                page_size=2, cursor=None,
                scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
                admission_timeout_seconds=min(admission_timeout_seconds, remaining()),
                timeout_seconds=remaining(), memory_mb=memory_mb)
            remaining()
            if (type(page) is not CodebaseVerificationQueryPage
                    or type(page.selector) is not CodebaseVerificationSelector
                    or page.selector.to_dict() != selector.to_dict() or page.selector.cid != selector.cid
                    or type(page.head) is not CodebaseHead or page.head.to_dict() != head.to_dict()
                    or type(page.entries) is not tuple or len(page.entries) > 2
                    or any(type(entry) is not CodebaseVerificationQueryEntry for entry in page.entries)
                    or page.start_cursor is not None
                    or type(page.complete) is not bool or page.complete != (page.next_cursor is None)):
                raise ConditionalCodebaseEvidenceError("indexed query page differs from the exact bounded native selector/head")
            indexed_query_page = _json(page.to_dict())
            page_body = {key: value for key, value in indexed_query_page.items() if key != "page_cid"}
            page_authority = indexed_query_page.get("authority")
            if (indexed_query_page.get("schema") != "codebase-verification-query-page@1"
                    or indexed_query_page.get("selector") != selector.to_dict()
                    or indexed_query_page.get("selector_cid") != selector.cid
                    or indexed_query_page.get("head") != head.to_dict()
                    or indexed_query_page.get("head_cid") != cid_for_structured(head.to_dict())
                    or indexed_query_page.get("inventory_cid") != page.inventory_cid
                    or indexed_query_page.get("epoch") != page.epoch
                    or indexed_query_page.get("complete") is not page.complete
                    or "start_cursor" not in indexed_query_page or indexed_query_page["start_cursor"] is not None
                    or indexed_query_page.get("page_cid") != page.page_cid
                    or cid_for_structured(page_body) != page.page_cid
                    or type(page_authority) is not dict
                    or page_authority.get("historical_conditional_evidence") is not True
                    or any(page_authority.get(name) is not False for name in (
                        "kernel_checked", "source_runtime_semantics_verified", "behavioral_satisfaction",
                        "authoritative_cache_eligible", "admission_authority", "completion_authority"))):
                raise ConditionalCodebaseEvidenceError("indexed query receipt lost its content/head/selector binding")
            if len(page.entries) >= 2:
                reasons.append("ambiguous_exact_current_evidence_requires_explicit_selector")
            elif not page.complete:
                reasons.append("incomplete_exact_current_evidence_query")
            elif not page.entries:
                reasons.append("no_exact_current_indexed_evidence")
            else:
                entry = page.entries[0]
                if entry.contract_id != query["contract"]["contract_id"]:
                    raise ConditionalCodebaseEvidenceError("indexed query entry differs from the selected authored contract")
                status, selected_reasons, evidence = _indexed_evidence(entry.projection, head, query)
                reasons.extend(selected_reasons)
        if _producer_pin() != query["producer"]:
            raise ConditionalCodebaseEvidenceError("matcher generation changed during lookup")
        result = {
            "schema": SCHEMA, "profile": PROFILE, "status": status,
            "query": query, "head": head.to_dict(), "head_cid": cid_for_structured(head.to_dict()),
            "current_root_id": head.snapshot_cid, "structural_context": context.to_dict(),
            "evidence": evidence, "reasons": reasons, "indexed_query_page": indexed_query_page,
            "selection": {"verification_cid": verification_cid, "expected_key_id": expected_key_id},
            "scope": "recorded_conditional_mathematics_on_requested_domain_only",
            "producer": query["producer"], "recorded_execution_attested": False,
            "current_facts": [], "current_behavioral_facts": [],
            "eligible_requirements": [], "behavioral_satisfied_requirements": [],
            "runtime_refutations": [], "removed_task_ids": [],
            "residual_requirements": [{"statement_id": item, "status": "runtime_behavior_unresolved",
                "selected_for_mathematical_lookup": item == statement_id}
                for item in query["requirement_ids"]],
            **_AUTHORITY,
        }
        result["match_cid"] = cid_for_structured(result)
        return _json(result)

    return run_with_structural_codebase_context(index, repository, build,
        repository_id=repository_id, expected_head=head, scheduler=scheduler, parent_lease=parent_lease,
        cancel_event=cancel_event, admission_timeout_seconds=admission_timeout_seconds,
        timeout_seconds=remaining(), memory_mb=memory_mb)


__all__ = ["SCHEMA", "QUERY_SCHEMA", "PROFILE", "PREDICATE", "ConditionalCodebaseEvidenceError",
           "prepare_conditional_codebase_query", "match_conditional_codebase_intent"]
