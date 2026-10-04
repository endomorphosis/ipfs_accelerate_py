"""Complete bounded ledgers of explicit integer requirements and fresh checks.

Inputs are separately authored native atoms, optionally built from exact CNL
sentences. This is not a claim to have interpreted an arbitrary full prompt.
Every requirement remains in the ledger; conditional solver outcomes do not
create behavioral facts, remove tasks, or authorize runtime execution.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured

from .checked_integer_codebase import (
    STATEMENT_ID, build_integer_offset_intent, prepare_integer_offset_query,
)
from .structural_codebase_context import run_with_structural_codebase_context

SCHEMA = "supervisor-checked-integer-requirements@1"
LEDGER_SCHEMA = "supervisor-integer-requirement-ledger@1"
MAX_REQUIREMENTS = 32
MAX_LEDGER_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 1024 * 1024
_PROFILE = "python-integer-offset@1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_FIELDS = frozenset({"requirement_id", "intent_document", "source_text", "statement_id"})
_AUTHORITY = {
    "source_semantics_verified": False, "runtime_behavior_verified": False,
    "behavior_authority": False, "proof_authority": False,
    "execution_authority": False, "completion_authority": False,
    "mutation_authority": False, "behavioral_satisfaction": False,
}


class CheckedIntegerBatchError(ValueError):
    """A batch input or owner result violates the complete ledger contract."""


def _inputs(values):
    if type(values) not in {list, tuple} or not 1 <= len(values) <= MAX_REQUIREMENTS:
        raise CheckedIntegerBatchError("one to 32 explicit requirement inputs required")
    return values


def _copy(value, maximum):
    raw = canonical_dag_json_bytes(value)
    if len(raw) > maximum:
        raise CheckedIntegerBatchError("bounded integer requirement artifact required")
    return json.loads(raw)


def build_integer_offset_requirements(
    sentences: list[str] | tuple[str, ...], *, source_revision: str = "authored:1",
) -> list[dict[str, Any]]:
    """Build one native required goal per exact CNL sentence, keeping duplicates.

    IDs are deterministic ordinal identities within this explicitly supplied
    list. Callers needing persistent external requirement IDs can author the
    same closed envelopes directly.
    """
    inputs = []
    for ordinal, sentence in enumerate(_inputs(sentences)):
        identity = f"requirement:{ordinal:04d}"
        native = build_integer_offset_intent(sentence, source_id=identity, source_revision=source_revision)
        inputs.append({"requirement_id": identity, "intent_document": native.to_dict(),
                       "source_text": sentence, "statement_id": STATEMENT_ID})
    return inputs


def prepare_integer_requirement_ledger(requirements) -> dict[str, Any]:
    """Validate every authored envelope without observation, proving or training.

    Valid unsupported atoms remain present. If an input document has additional
    native statements, its existing single-atom profile abstains and all of its
    native statement IDs remain residual in this complete input ledger.
    """
    rows, seen = [], set()
    for item in _inputs(requirements):
        if type(item) is not dict or frozenset(item) not in {_FIELDS, _FIELDS | {"source_identity"}}:
            raise CheckedIntegerBatchError("exact requirement envelope fields required")
        identity = item["requirement_id"]
        if type(identity) is not str or _ID.fullmatch(identity) is None or identity in seen:
            raise CheckedIntegerBatchError("unique bounded requirement IDs required")
        seen.add(identity)
        query = prepare_integer_offset_query(
            intent_document=item["intent_document"], source_text=item["source_text"],
            statement_id=item["statement_id"], source_identity=item.get("source_identity"),
        )
        rows.append({"requirement_id": identity, "query": query,
                     "native_requirement_ids": query["requirement_ids"]})
    ledger = {"schema": LEDGER_SCHEMA, "profile": _PROFILE,
              "input_scope": "explicit_requirement_envelopes_only", "requirements": rows,
              "requirement_count": len(rows), "complete_input_ledger": True, **_AUTHORITY}
    ledger = _copy(ledger, MAX_LEDGER_BYTES)
    ledger["ledger_cid"] = cid_for_structured(ledger)
    return ledger


def _receipt(result, *, head, contract):
    if type(result) is not dict:
        raise CheckedIntegerBatchError("native per-contract receipt required")
    result = _copy(result, MAX_RECEIPT_BYTES)
    if (result.get("schema") != "codebase-integer-verification@1"
            or result.get("profile") != _PROFILE
            or canonical_dag_json_bytes(result.get("head")) != canonical_dag_json_bytes(head.to_dict())
            or canonical_dag_json_bytes(result.get("contract")) != canonical_dag_json_bytes(contract.to_dict())
            or result.get("contract_cid") != contract.cid
            or result.get("kernel_checked") is not False or result.get("behavior_authority") is not False
            or result.get("execution_authority") is not False or result.get("completion_authority") is not False):
        raise CheckedIntegerBatchError("owner receipt contract, head or authority differs")
    outcome = result.get("status")
    if outcome not in {"proved", "refuted", "unknown", "timeout", "unavailable", "disagreement", "unsupported", "error"}:
        raise CheckedIntegerBatchError("supported owner receipt outcome required")
    if outcome in {"proved", "refuted"} and result.get("solver_replayed") is not True:
        raise CheckedIntegerBatchError("conditional evidence requires fresh native replay")
    return result


def match_checked_integer_requirements(
    *, index, repository, repository_id: str, requirements, expected_head=None,
    scheduler=None, parent_lease=None, cancel_event=None, admission_timeout_seconds: float = 30.0,
    timeout_seconds: float = 120.0, max_workers: int | None = None, memory_mb: int | None = None,
    evidence_index=None, cache=None,
) -> dict[str, Any]:
    """Resolve a complete explicit ledger through one bounded owner batch call.

    Every supported contract is submitted, including duplicates; the owner may
    deduplicate work while returning one ordered result per input. Historical
    index hits never replace its native replay. Current-source/head drift or a
    failed completion observation withholds the entire derived result.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    ledger = prepare_integer_requirement_ledger(requirements)
    if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
        raise CheckedIntegerBatchError("finite positive overall timeout of at most 600 seconds required")
    if (type(admission_timeout_seconds) not in {int, float} or not math.isfinite(admission_timeout_seconds)
            or admission_timeout_seconds < 0):
        raise CheckedIntegerBatchError("finite nonnegative admission timeout required")
    if max_workers is not None and (type(max_workers) is not int or not 1 <= max_workers <= MAX_REQUIREMENTS):
        raise CheckedIntegerBatchError("max_workers must be a positive bounded integer")
    if memory_mb is not None and (type(memory_mb) is not int or memory_mb < 1024):
        raise CheckedIntegerBatchError("memory_mb must be an integer of at least 1024 MiB")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise CheckedIntegerBatchError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("integer requirement batch cancelled")
        value = deadline - time.monotonic()
        if value <= 0:
            raise LeaseTimeoutError("integer requirement batch deadline exceeded")
        return value

    def build(context):
        supported = [row for row in ledger["requirements"] if row["query"]["supported"]]
        contracts = [IntegerOffsetContract.from_dict(row["query"]["contract"]) for row in supported]
        batch, verified = None, {}
        if contracts:
            from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

            duration = remaining()
            batch = CodebaseIntegerVerifier(index, cache=cache).verify_many(
                repository, expected_head=context.head, contracts=contracts,
                scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
                admission_timeout_seconds=min(admission_timeout_seconds, duration), timeout_seconds=duration,
                max_workers=max_workers, memory_mb=memory_mb, evidence_index=evidence_index,
            )
            remaining()
            if (type(batch) is not dict or batch.get("schema") != "codebase-integer-batch@1"
                    or batch.get("profile") != _PROFILE
                    or canonical_dag_json_bytes(batch.get("head")) != canonical_dag_json_bytes(context.head.to_dict())
                    or canonical_dag_json_bytes(batch.get("requested_contracts"))
                    != canonical_dag_json_bytes([contract.to_dict() for contract in contracts])
                    or any(batch.get(name) is not False for name in ("kernel_checked", "behavior_authority",
                                                                   "execution_authority", "completion_authority"))
                    or type(batch.get("results")) is not list or len(batch["results"]) != len(contracts)):
                raise CheckedIntegerBatchError("owner batch must cover the exact ordered contract inventory and head")
            for row, contract, receipt in zip(supported, contracts, batch["results"]):
                verified[row["requirement_id"]] = _receipt(receipt, head=context.head, contract=contract)
            batch = _copy(batch, MAX_REQUIREMENTS * MAX_RECEIPT_BYTES + MAX_LEDGER_BYTES)
        results, residuals = [], []
        counts = {"matched": 0, "refuted": 0, "unknown": 0}
        for row in ledger["requirements"]:
            identity, query = row["requirement_id"], row["query"]
            receipt = verified.get(identity)
            status, reasons = "unknown", list(query["reasons"])
            if receipt is not None:
                outcome = receipt["status"]
                if outcome == "proved":
                    status = "conditional_contract_matched"
                elif outcome == "refuted":
                    status = "conditional_contract_refuted"
                else:
                    reasons.append("checked_contract_" + outcome)
            counts[{"conditional_contract_matched": "matched", "conditional_contract_refuted": "refuted"}.get(status, "unknown")] += 1
            results.append({"requirement_id": identity, "native_requirement_ids": row["native_requirement_ids"],
                            "query_cid": query["query_cid"], "status": status, "reasons": reasons,
                            "checked_result_cid": cid_for_structured(receipt) if receipt is not None else None,
                            "receipt_cid": receipt.get("receipt_cid") if receipt is not None else None,
                            "runtime_behavior_status": "unresolved", **_AUTHORITY})
            residuals.extend({"requirement_id": identity, "statement_id": statement_id,
                              "status": "runtime_behavior_unresolved"}
                             for statement_id in row["native_requirement_ids"])
        result = {
            "schema": SCHEMA, "profile": _PROFILE,
            "status": "conditional_results_available" if counts["matched"] or counts["refuted"] else "unknown",
            "ledger": ledger, "structural_context": context.to_dict(),
            "verification_batch": batch, "verification_batch_cid": cid_for_structured(batch) if batch is not None else None,
            "requirement_results": results, "conditional_summary": {"total": len(results), **counts},
            "residual_requirements": residuals,
            "conditional_residual_requirement_ids": [row["requirement_id"] for row in results
                                                       if row["status"] != "conditional_contract_matched"],
            "current_facts": [], "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
            "runtime_refutations": [], "removed_task_ids": [],
            "scope": "declared_integer_contracts_only", **_AUTHORITY,
        }
        remaining()
        result["match_cid"] = cid_for_structured(result)
        return _copy(result, MAX_REQUIREMENTS * MAX_RECEIPT_BYTES + 3 * MAX_LEDGER_BYTES)

    return run_with_structural_codebase_context(
        index, repository, build, repository_id=repository_id, expected_head=expected_head,
        scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
        admission_timeout_seconds=admission_timeout_seconds, timeout_seconds=remaining(), memory_mb=512,
    )


__all__ = ["SCHEMA", "LEDGER_SCHEMA", "MAX_REQUIREMENTS", "CheckedIntegerBatchError",
           "build_integer_offset_requirements", "prepare_integer_requirement_ledger",
           "match_checked_integer_requirements"]
