"""Versioned behavioral matching for an explicit finite source profile.

The v2 envelope joins complete semantic inventory, exact native discovery and
checked history to the existing fresh finite matcher. Discovery and cached
absence cannot establish a property. Other instruction meanings remain open.
"""
from __future__ import annotations

import math
import time

from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live import verify_policy_current
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import load_codebase_semantic_manifest
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from . import finite_integer_codebase as finite
from ..proof.finite_checked_cache import FiniteCheckedCache

SCHEMA = "intent-codebase-match@2"
PROFILE = "finite-integer-behavior-with-native-discovery@1"


def match_behavioral_intent(*, catalog, checked_cache, repository, expected_head,
        semantic_manifest_cid, intent_document, source_text, output, tool_policy,
        scheduler=None, parent_lease=None, cancel_event=None, timeout_seconds=180,
        memory_mb=1024):
    """Resolve the full supported instruction; preserve every other requirement.

    This route is model-off. A manifest nomination must exist for the exact
    selected contract before checked history or a fresh observation is consumed.
    Both requirement clauses remain in the result and signed task population.
    No stored or caller-provided match can replace this owner invocation.
    """
    if (type(catalog) is not IntentCodebaseCatalog or type(checked_cache) is not FiniteCheckedCache
            or checked_cache.artifacts is not catalog.index.artifacts):
        raise ValueError("native discovery, source and checked-cache owners must share exact artifacts")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise ValueError("bounded behavioral matching deadline required")
    if type(memory_mb) is not int or memory_mb < 1024:
        raise ValueError("behavioral matching requires at least 1024 MiB")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise ValueError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds
    def remaining():
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("behavioral matching cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("behavioral matching deadline expired")
        return duration

    index = catalog.index
    query = finite.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    manifest = load_codebase_semantic_manifest(index, semantic_manifest_cid)
    if manifest["source_head"] != expected_head.to_dict():
        raise ValueError("behavioral semantic inventory belongs to another source generation")
    resources = dict(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event)
    def observe():
        verify_policy_current(index, repository, expected_head=expected_head,
            receipt_cid=manifest["policy_receipt_cid"], timeout_seconds=remaining(), **resources)
    observe()
    discovery, cache_result = None, None
    supported = query["supported"]
    if supported:
        contract = IntegerOffsetContract.from_dict(query["contract"])
        discovery = catalog.lookup(repository, expected_head=expected_head,
            policy_receipt_cid=manifest["policy_receipt_cid"], path=contract.path,
            contract_cid=contract.cid, timeout_seconds=remaining(), **resources)
        nominees = [row for row in discovery["records"]
                    if row["record"]["semantic_manifest_cid"] == semantic_manifest_cid]
        if not nominees:
            # Missing discovery evidence must not imply that a requirement is
            # satisfied, or trigger unrecorded fallback to an unrelated model.
            raise ValueError("exact semantic manifest/contract nomination is unavailable")
        if any(row["record"]["model"] is not None for row in nominees):
            raise ValueError("this model-off profile cannot reinterpret a model-associated nomination")
        selected = [row for row in manifest["units"] if row["path"] == contract.path
                    and row["declared_contract_cid"] == contract.cid]
        if len(selected) != 1 or selected[0]["model_status"] != "source_bound_model":
            raise ValueError("nomination lacks a supported exact source/contract model")
        inputs = dict(index=index, repository=repository, expected_head=expected_head,
            contract=contract, inputs=query["domain_inputs"], tool_policy=tool_policy, **resources)
        cache_result = checked_cache.lookup(owner_inputs=inputs, timeout_seconds=remaining())
        if cache_result["status"] == "miss":
            cache_result = checked_cache.check_and_store(owner_inputs=inputs, timeout_seconds=remaining())
    match = finite.match_finite_integer_intent(index=index, repository=repository,
        repository_id=expected_head.repository_id, expected_head=expected_head,
        intent_document=intent_document, source_text=source_text, output=output,
        tool_policy=tool_policy, timeout_seconds=remaining(), memory_mb=memory_mb, **resources)
    if match.get("match_cid") != cid_for_structured({key: value for key, value in match.items() if key != "match_cid"}):
        raise ValueError("fresh matcher envelope identity differs")
    if match.get("head") != expected_head.to_dict() or match.get("current_root_id") != expected_head.snapshot_cid:
        raise ValueError("fresh matcher source generation differs")
    if match["query"] != query:
        raise ValueError("fresh matcher consumed a different full instruction")
    if supported and match["observation"] is not None and match["observation"]["status"] == "observed":
        expected_status = "positive" if match["observation"]["offset_clause_satisfied"] else "refuted"
        if cache_result["status"] != expected_status:
            raise ValueError("checked history and fresh source behavior disagree")
    requirements = query["requirement_ids"]
    rows = match["clause_results"]
    if (sorted(row["statement_id"] for row in rows) != requirements
            or sorted(match["eligible_clause_ids"] + match["residual_clause_ids"]) != requirements):
        raise ValueError("behavioral matching dropped or duplicated a requirement")
    observe()
    observed = match["observation"] is not None and match["observation"]["status"] == "observed"
    result = dict(schema=SCHEMA, profile=PROFILE, source_head=expected_head.to_dict(),
        semantic_manifest_cid=semantic_manifest_cid, complete_inventory=manifest["coverage"],
        unsupported_units=[dict(path=row["path"], disposition=row["model_status"])
                           for row in manifest["units"] if row["model_status"] != "source_bound_model"],
        model={"enabled": False, "identity": "explicit-model-off@1"},
        interpretation={"supported": supported, "full_instruction_alignment": query["semantic_alignment_verified"],
            "profile": finite.CNL_PROFILE, "unresolved_reasons": query["reasons"],
            "symbol_resolution": "exact_captured_path_function_parameter" if supported else "unresolved",
            "quantifier": "all_enumerated_nonempty_integer_inputs" if supported else "unresolved",
            "polarity": "required_positive_properties" if supported else "unresolved",
            "guards": "no_conditional_scope_in_selected_profile" if supported else "unresolved",
            "effects": "closed_pure_integer_expression_under_declared_assumptions" if supported else "unresolved",
            "theorem_domain": query["domain_inputs"], "source_statements": query["statements"]},
        discovery=discovery, checked_cache=None if cache_result is None else {
            key: cache_result[key] for key in ("record_cid", "request_key", "status", "scope", "positive_reuse_eligible")},
        status=match["status"], requirement_results=rows, current_facts=match["current_facts"],
        satisfied_requirements=match["eligible_clause_ids"], residual_requirements=match["residual_clause_ids"],
        counterexamples=match["finite_counterexamples"], fresh_match=match,
        assurance="complete_finite_runtime_observation_and_kernel_checked_model_table" if observed else "unresolved",
        whole_program_verified=False, proof_cache_grants_source_authority=False,
        execution_authority=False, completion_authority=False, reduced_task_population_authorized=False)
    result["match_cid"] = cid_for_structured(result)
    remaining()
    return result


__all__ = ["match_behavioral_intent", "SCHEMA", "PROFILE"]
