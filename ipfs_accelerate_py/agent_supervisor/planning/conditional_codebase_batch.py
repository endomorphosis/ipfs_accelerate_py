"""Bounded, ordered conditional intent lookups under one evidence inventory.

Only explicit native mathematical requests are matched. Every runtime
requirement remains unresolved; historical evidence grants no task authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import CodeType, FunctionType
from typing import Any

from . import conditional_codebase_evidence as single

SCHEMA = "supervisor-conditional-codebase-batch@1"
MAX_REQUIREMENTS = 32
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
_REQUIRED = frozenset({"requirement_id", "intent_document", "source_text", "path",
                       "contract", "domain", "statement_id"})
_OPTIONAL = frozenset({"source_identity", "verification_cid", "expected_key_id"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_IMPORTED_SOURCE_PIN: str | None = None


class ConditionalCodebaseBatchError(ValueError):
    """An explicit batch or returned inventory does not preserve its bindings."""


def _producer_pin() -> dict[str, Any]:
    global _IMPORTED_SOURCE_PIN
    path = Path(__file__).resolve()
    with path.open("rb") as source:
        raw = source.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ConditionalCodebaseBatchError("batch implementation size limit exceeded")
    compiled = compile(raw, str(path), "exec", dont_inherit=True)
    codes = {item.co_name: item for item in compiled.co_consts if type(item) is CodeType}
    for name, value in globals().copy().items():
        if type(value) is FunctionType and value.__module__ == __name__:
            if codes.get(name) != value.__code__:
                raise ConditionalCodebaseBatchError("loaded batch matcher differs from installed source")
    digest = hashlib.sha256(raw).hexdigest()
    if _IMPORTED_SOURCE_PIN is not None and _IMPORTED_SOURCE_PIN != digest:
        raise ConditionalCodebaseBatchError("batch matcher implementation generation changed")
    _IMPORTED_SOURCE_PIN = digest
    return {"module": __name__, "sha256": digest, "profile": single.PROFILE,
            "matcher": single._producer_pin()}


def _bytes(value: Any, maximum: int) -> bytes:
    # Only owned output from the strict single-request decoder reaches this
    # encoder; public envelope objects are never serialized directly.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(raw) > maximum:
        raise ConditionalCodebaseBatchError("aggregate conditional batch byte limit exceeded")
    return raw


def _requests(requirements: Any, producer: dict[str, Any], checkpoint) -> tuple:
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationSelector
    from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument

    if type(requirements) is not tuple or not 1 <= len(requirements) <= MAX_REQUIREMENTS:
        raise ConditionalCodebaseBatchError("one to 32 explicit requirement envelopes in an exact tuple required")
    snapshots, seen, raw_retained = [], set(), 0
    for item in requirements:
        checkpoint()
        if (type(item) is not dict or any(type(key) is not str for key in item)
                or not _REQUIRED <= item.keys() or not item.keys() <= _REQUIRED | _OPTIONAL):
            raise ConditionalCodebaseBatchError("exact conditional requirement envelope fields required")
        item = dict(item)
        identity = item["requirement_id"]
        if type(identity) is not str or _ID.fullmatch(identity) is None or identity in seen:
            raise ConditionalCodebaseBatchError("unique bounded requirement IDs required")
        seen.add(identity)
        contract, domain, contract_cid, domain_cid = single._owned_request(
            item["path"], item["contract"], item["domain"])
        document = item["intent_document"]
        document = document.to_dict() if type(document) is IntentIRDocument else document
        # Snapshot the entire authored inventory as exact bounded inert JSON
        # before native intent decoding. Later caller changes cannot alter a
        # not-yet-decoded row, and aggregate raw input limits apply first.
        inert = single._json({"requirement_id": identity, "intent_document": document,
            "source_text": item["source_text"], "path": item["path"],
            "contract": contract.to_dict(), "domain": domain.to_dict(),
            "statement_id": item["statement_id"], "source_identity": item.get("source_identity"),
            "verification_cid": item.get("verification_cid"), "expected_key_id": item.get("expected_key_id")})
        raw_retained += len(_bytes(inert, MAX_REQUEST_BYTES))
        if raw_retained > MAX_REQUEST_BYTES:
            raise ConditionalCodebaseBatchError("aggregate conditional request byte limit exceeded")
        selection = {"verification_cid": inert["verification_cid"], "expected_key_id": inert["expected_key_id"]}
        # Optional selectors are checked even for unsupported native atoms.
        selector = CodebaseVerificationSelector(path=inert["path"], contract_id=contract.contract_id,
            verification_cid=selection["verification_cid"], expected_contract_cid=contract_cid,
            canonical_key_id=selection["expected_key_id"], requested_domain_id=domain.domain_id,
            requested_domain_cid=domain_cid)
        snapshots.append((inert, contract, domain, selector, selection))
        checkpoint()
    rows, retained = [], 0
    for item, contract, domain, selector, selection in snapshots:
        checkpoint()
        query = single.prepare_conditional_codebase_query(
            intent_document=item["intent_document"], source_text=item["source_text"],
            path=item["path"], contract=contract, domain=domain,
            statement_id=item["statement_id"], source_identity=item["source_identity"])
        checkpoint()
        if query["producer"] != producer["matcher"]:
            raise ConditionalCodebaseBatchError("matcher generation changed while preparing the batch")
        retained += len(_bytes({"requirement_id": item["requirement_id"], "query": query,
                               "selection": selection}, MAX_REQUEST_BYTES))
        if retained > MAX_REQUEST_BYTES:
            raise ConditionalCodebaseBatchError("aggregate conditional request byte limit exceeded")
        rows.append((item["requirement_id"], query, selector, selection))
    return tuple(rows)


def _page(page: Any, *, selector: Any, head: Any) -> dict[str, Any]:
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import (
        CodebaseVerificationSelector, CodebaseVerificationQueryPage, CodebaseVerificationQueryEntry,
    )
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    if (type(page) is not CodebaseVerificationQueryPage
            or type(page.selector) is not CodebaseVerificationSelector
            or page.selector.to_dict() != selector.to_dict() or page.selector.cid != selector.cid
            or type(page.head) is not CodebaseHead or page.head.to_dict() != head.to_dict()
            or type(page.entries) is not tuple or len(page.entries) > 2
            or any(type(entry) is not CodebaseVerificationQueryEntry for entry in page.entries)
            or page.start_cursor is not None
            or type(page.complete) is not bool or page.complete != (page.next_cursor is None)):
        raise ConditionalCodebaseBatchError("batch page differs from its exact bounded native selector/head")
    value = single._json(page.to_dict())
    body = {key: item for key, item in value.items() if key != "page_cid"}
    authority = value.get("authority")
    if (value.get("schema") != "codebase-verification-query-page@1"
            or value.get("selector") != selector.to_dict() or value.get("selector_cid") != selector.cid
            or value.get("head") != head.to_dict()
            or value.get("head_cid") != cid_for_structured(head.to_dict())
            or value.get("inventory_cid") != page.inventory_cid or value.get("epoch") != page.epoch
            or value.get("complete") is not page.complete
            or "start_cursor" not in value or value["start_cursor"] is not None
            or value.get("entries") != [entry.to_dict() for entry in page.entries]
            or value.get("next_cursor") != (None if page.next_cursor is None else page.next_cursor.to_dict())
            or value.get("page_cid") != page.page_cid or cid_for_structured(body) != page.page_cid
            or type(authority) is not dict or authority.get("historical_conditional_evidence") is not True
            or any(authority.get(name) is not False for name in (
                "kernel_checked", "source_runtime_semantics_verified", "behavioral_satisfaction",
                "authoritative_cache_eligible", "admission_authority", "completion_authority"))):
        raise ConditionalCodebaseBatchError("batch page receipt lost its content/head/selector binding")
    return value


def _match(*, query, selection, page, receipt, head, context) -> dict[str, Any]:
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    status, evidence, reasons = "unknown", None, list(query["reasons"])
    if page is not None:
        if len(page.entries) >= 2:
            reasons.append("ambiguous_exact_current_evidence_requires_explicit_selector")
        elif not page.complete:
            reasons.append("incomplete_exact_current_evidence_query")
        elif not page.entries:
            reasons.append("no_exact_current_indexed_evidence")
        else:
            entry = page.entries[0]
            if entry.contract_id != query["contract"]["contract_id"]:
                raise ConditionalCodebaseBatchError("batch entry differs from selected authored contract")
            status, selected_reasons, evidence = single._indexed_evidence(entry.projection, head, query)
            reasons.extend(selected_reasons)
    result = {
        "schema": single.SCHEMA, "profile": single.PROFILE, "status": status,
        "query": query, "head": head.to_dict(), "head_cid": cid_for_structured(head.to_dict()),
        "current_root_id": head.snapshot_cid, "structural_context": context.to_dict(),
        "evidence": evidence, "reasons": reasons, "indexed_query_page": receipt,
        "selection": selection, "scope": "recorded_conditional_mathematics_on_requested_domain_only",
        "producer": query["producer"], "recorded_execution_attested": False,
        "current_facts": [], "current_behavioral_facts": [], "eligible_requirements": [],
        "behavioral_satisfied_requirements": [], "runtime_refutations": [], "removed_task_ids": [],
        "residual_requirements": [{"statement_id": item, "status": "runtime_behavior_unresolved",
            "selected_for_mathematical_lookup": item == query["statement_id"]}
            for item in query["requirement_ids"]], **single._AUTHORITY,
    }
    result["match_cid"] = cid_for_structured(result)
    return single._json(result)


def match_conditional_codebase_requirements(
    *, catalog: Any, index: Any, repository: Any, repository_id: str, expected_head: Any,
    requirements: Any, scheduler: Any = None, parent_lease: Any = None, cancel_event: Any = None,
    admission_timeout_seconds: float = 30.0, timeout_seconds: float = 120.0, memory_mb: int = 512,
) -> dict[str, Any]:
    """Match 1–32 explicit envelopes in one current, all-or-nothing lookup.

    Required fields are requirement_id, intent_document, source_text, path,
    contract, domain and statement_id. Optional fields are source_identity,
    verification_cid and expected_key_id. Duplicate mathematical requests keep
    separate unique requirement IDs. Unsupported requests remain in the ledger.
    A failed final source observation withholds every result. No lookup trains,
    invokes solvers, grants admission or changes runtime task requirements.
    """
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationQueryRequest
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
    from .structural_codebase_context import run_with_structural_codebase_context

    if (type(index) is not RepositoryCodebaseIndex or type(catalog) is not CodebaseVerificationCatalog
            or catalog.index is not index or type(expected_head) is not CodebaseHead):
        raise ConditionalCodebaseBatchError("same native index/catalog and exact current head required")
    head = CodebaseHead.from_dict(expected_head.to_dict())
    if type(repository_id) is not str or head.repository_id != repository_id:
        raise ConditionalCodebaseBatchError("expected head does not bind this repository view")
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 600):
        raise ConditionalCodebaseBatchError("finite positive overall timeout of at most 600 seconds required")
    if (type(admission_timeout_seconds) not in {int, float}
            or not math.isfinite(admission_timeout_seconds) or admission_timeout_seconds < 0):
        raise ConditionalCodebaseBatchError("finite nonnegative admission timeout required")
    if type(memory_mb) is not int or not 64 <= memory_mb <= 4096:
        raise ConditionalCodebaseBatchError("bounded integer memory reservation of 64–4096 MiB required")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise ConditionalCodebaseBatchError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("conditional requirement batch cancelled")
        value = deadline - time.monotonic()
        if value <= 0:
            raise LeaseTimeoutError("conditional requirement batch deadline exceeded")
        return value

    remaining()
    producer = _producer_pin()
    rows = _requests(requirements, producer, remaining)
    remaining()

    def build(context: Any) -> dict[str, Any]:
        supported = tuple(row for row in rows if row[1]["supported"])
        pages = ()
        if supported:
            duration = remaining()
            pages = catalog.query_many_current(repository, expected_head=head,
                requests=tuple(CodebaseVerificationQueryRequest(row[2], page_size=2, cursor=None) for row in supported),
                scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
                admission_timeout_seconds=min(admission_timeout_seconds, duration),
                timeout_seconds=duration, memory_mb=memory_mb)
            remaining()
        if type(pages) is not tuple or len(pages) != len(supported):
            raise ConditionalCodebaseBatchError("batch pages must cover the exact ordered supported inventory")
        selected, inventory = {}, None
        for row, page in zip(supported, pages):
            remaining()
            receipt = _page(page, selector=row[2], head=head)
            binding = {"inventory_cid": page.inventory_cid, "epoch": page.epoch}
            if inventory is not None and binding != inventory:
                raise ConditionalCodebaseBatchError("batch pages came from different evidence inventories")
            inventory = binding
            selected[row[0]] = (page, receipt)
        matches, residuals, retained = [], [], 0
        for identity, query, selector, selection in rows:
            remaining()
            page, receipt = selected.get(identity, (None, None))
            match = _match(query=query, selection=selection, page=page, receipt=receipt,
                           head=head, context=context)
            retained += len(_bytes(match, MAX_RESULT_BYTES))
            if retained > MAX_RESULT_BYTES:
                raise ConditionalCodebaseBatchError("aggregate conditional result byte limit exceeded")
            matches.append({"requirement_id": identity, "match": match})
            residuals.extend({"requirement_id": identity, **item} for item in match["residual_requirements"])
        if _producer_pin() != producer:
            raise ConditionalCodebaseBatchError("batch matcher generation changed during lookup")
        result = {"schema": SCHEMA, "profile": single.PROFILE,
            "requested_requirement_ids": [row[0] for row in rows], "matches": matches,
            "head": head.to_dict(), "head_cid": cid_for_structured(head.to_dict()),
            "structural_context": context.to_dict(), "inventory": inventory,
            "complete_input_ledger": True, "residual_requirements": residuals,
            "scope": "recorded_conditional_mathematics_on_requested_domain_only",
            "producer": producer, "recorded_execution_attested": False,
            "current_facts": [], "current_behavioral_facts": [], "eligible_requirements": [],
            "behavioral_satisfied_requirements": [], "runtime_refutations": [], "removed_task_ids": [],
            **single._AUTHORITY}
        result["batch_cid"] = cid_for_structured(result)
        remaining()
        return json.loads(_bytes(result, MAX_RESULT_BYTES))

    result = run_with_structural_codebase_context(index, repository, build,
        repository_id=repository_id, expected_head=head, scheduler=scheduler, parent_lease=parent_lease,
        cancel_event=cancel_event, admission_timeout_seconds=admission_timeout_seconds,
        timeout_seconds=remaining(), memory_mb=memory_mb)
    remaining()
    if _producer_pin() != producer:
        raise ConditionalCodebaseBatchError("batch matcher generation changed during final observation")
    remaining()
    return result


__all__ = ["SCHEMA", "MAX_REQUIREMENTS", "MAX_REQUEST_BYTES", "MAX_RESULT_BYTES",
           "ConditionalCodebaseBatchError", "match_conditional_codebase_requirements"]
