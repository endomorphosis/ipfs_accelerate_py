"""Explicit finite integer intents checked by fresh owner-run observations.

This separate profile describes two properties of a captured function on an
explicit finite input set. Its facts do not describe arbitrary calls, prove
Python equivalence, authorize a task, or change the conditional @1 matcher.
The native observer is called here; callers cannot supply a positive receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
from ipfs_datasets_py.logic.intent_ir.schema import (
    IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
    ReviewStatus, SourceRef, SourceSpan, StatementKind,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)

from .obligation_graph_compiler import (
    FactAuthority, FactTruth, InvalidationSelector, InvalidationSelectorKind,
    ObservedFact, SemanticSupport, TypedIntent, TypedPredicate,
)
from .structural_codebase_context import run_with_structural_codebase_context

SCHEMA = "supervisor-finite-integer-intent@1"
QUERY_SCHEMA = "supervisor-finite-integer-query@1"
CNL_PROFILE = "finite-integer-intent-sentences@1"
PROFILE = "python-integer-offset-finite@1"
TYPE_STATEMENT_ID = "finite-integer-type-goal"
OFFSET_STATEMENT_ID = "finite-integer-offset-goal"
TYPE_PREDICATE = "finite_integer_exact_type"
OFFSET_PREDICATE = "finite_integer_offset"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_TARGET = (r"(?P<path>[A-Za-z0-9_./-]{1,512})::"
           r"(?P<function>[A-Za-z_][A-Za-z0-9_]{0,127})\("
           r"(?P<parameter>[A-Za-z_][A-Za-z0-9_]{0,127})\)")
_TYPE_SENTENCE = re.compile(
    r"Under python-integer-offset-finite@1, " + _TARGET
    + r" must return an exact int for inputs (?P<inputs>\[[^\]\r\n]{1,1024}\])\."
)
_OFFSET_SENTENCE = re.compile(
    r"Under python-integer-offset-finite@1, " + _TARGET
    + r" must return (?P=parameter) (?P<sign>[+-]) (?P<offset>0|[1-9][0-9]{0,19})"
    + r" for inputs (?P<inputs>\[[^\]\r\n]{1,1024}\])\."
)
_AUTHORITY = {
    "source_semantics_verified": False, "runtime_behavior_verified": False,
    "behavior_authority": False, "proof_authority": False,
    "execution_authority": False, "completion_authority": False,
    "mutation_authority": False,
}


class FiniteIntegerIntentError(ValueError):
    """Input or observation cannot support this exact finite intent profile."""


def _json(value):
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 80_000 or depth > 32:
            raise FiniteIntegerIntentError("bounded finite-intent JSON required")
        if type(item) is dict:
            if len(item) > 80_000 - count or any(type(key) is not str for key in item):
                raise FiniteIntegerIntentError("exact string JSON keys required")
            text_bytes += sum(len(key.encode("utf-8", errors="surrogatepass")) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) > 80_000 - count:
                raise FiniteIntegerIntentError("finite-intent collection bound exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int and item.bit_length() > 128:
            raise FiniteIntegerIntentError("finite-intent integer bound exceeded")
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise FiniteIntegerIntentError("exact inert JSON required")
        if text_bytes > _MAX_JSON_BYTES:
            raise FiniteIntegerIntentError("finite-intent text bound exceeded")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise FiniteIntegerIntentError("finite JSON required") from exc
    if len(raw) > _MAX_JSON_BYTES:
        raise FiniteIntegerIntentError("finite-intent byte bound exceeded")
    return json.loads(raw)


def _text(source_text):
    if type(source_text) is not str or not source_text or len(source_text) > 4096:
        raise FiniteIntegerIntentError("nonempty intent source within 4096 bytes required")
    try:
        if len(source_text.encode("utf-8")) > 4096:
            raise FiniteIntegerIntentError("intent source exceeds 4096 bytes")
    except UnicodeError as exc:
        raise FiniteIntegerIntentError("valid UTF-8 intent source required") from exc
    return source_text


def _parse(source_text):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract

    parts = _text(source_text).split("\n")
    if len(parts) != 2:
        raise FiniteIntegerIntentError("exactly two finite-integer sentences required")
    type_match, offset_match = _TYPE_SENTENCE.fullmatch(parts[0]), _OFFSET_SENTENCE.fullmatch(parts[1])
    if type_match is None or offset_match is None:
        raise FiniteIntegerIntentError("exact finite-integer sentence syntax required")
    if any(type_match[name] != offset_match[name] for name in ("path", "function", "parameter", "inputs")):
        raise FiniteIntegerIntentError("both finite clauses must have the same target and input domain")
    try:
        inputs = json.loads(type_match["inputs"])
    except (ValueError, RecursionError) as exc:
        raise FiniteIntegerIntentError("canonical finite integer input array required") from exc
    if (type(inputs) is not list or not 1 <= len(inputs) <= 32
            or any(type(item) is not int or abs(item) > 2**31 for item in inputs)
            or inputs != sorted(set(inputs))
            or json.dumps(inputs, separators=(",", ":")) != type_match["inputs"]):
        raise FiniteIntegerIntentError("sorted unique nonempty exact integer inputs within profile bounds required")
    magnitude = int(offset_match["offset"])
    if offset_match["sign"] == "-" and magnitude == 0:
        raise FiniteIntegerIntentError("negative zero offset is not canonical")
    try:
        contract = IntegerOffsetContract(
            type_match["path"], type_match["function"], type_match["parameter"],
            magnitude if offset_match["sign"] == "+" else -magnitude,
        )
    except (ValueError, TypeError) as exc:
        raise FiniteIntegerIntentError("finite sentence target or offset is outside the native contract profile") from exc
    return contract, inputs, parts


def _arguments(contract, inputs, *, offset):
    common = [contract.path, contract.function_name, contract.parameter,
              json.dumps(inputs, separators=(",", ":")), PROFILE]
    return common + ([str(contract.offset)] if offset else ["exact_int"])


def build_finite_integer_intent(
    source_text: str, *, source_id: str = "finite-integer:request", source_revision: str = "authored:1",
) -> IntentIRDocument:
    """Build native IR from the complete two-clause canonical finite CNL source."""
    contract, inputs, clauses = _parse(source_text)
    digest = hashlib.sha256(source_text.encode()).hexdigest()
    sources, statements = [], []
    for position, (statement_id, predicate, label) in enumerate((
        (TYPE_STATEMENT_ID, TYPE_PREDICATE, "type"),
        (OFFSET_STATEMENT_ID, OFFSET_PREDICATE, "offset"),
    )):
        start = 0 if position == 0 else len(clauses[0]) + 1
        source = SourceRef(
            ref_id="source:finite-integer-" + label, source_uri="intent:finite-integer",
            source_id=source_id, source_revision=source_revision, content_sha256=digest,
            review_status=ReviewStatus.UNREVIEWED, span=SourceSpan(start, start + len(clauses[position])),
        )
        sources.append(source)
        statements.append(IntentStatement(
            statement_id=statement_id, kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
            normalized_text=clauses[position], source_ref_ids=(source.ref_id,), predicate=predicate,
            arguments=tuple(_arguments(contract, inputs, offset=position == 1)),
            grounding=NodeGrounding.GROUNDED, review_status=ReviewStatus.UNREVIEWED,
        ))
    native = IntentIRDocument(
        document_id="intent:finite-integer", title="Explicit finite integer observations",
        intent_kind=IntentKind.DECLARATIVE, sources=tuple(sources), statements=tuple(statements),
    )
    native.validate()
    return native


def prepare_finite_integer_query(*, intent_document, source_text: str, source_identity=None) -> dict[str, Any]:
    """Rebuild full CNL and native selectors; valid unsupported intent stays open.

    ``source_identity`` is, when supplied, the complete sorted native source-ref
    dictionaries. Provenance failure is an error; caller annotations cannot
    turn unsupported syntax or a different native meaning into finite facts.
    """
    _text(source_text)
    try:
        if type(intent_document) is IntentIRDocument:
            intent_document.validate()
            raw = intent_document.to_dict()
        else:
            raw = intent_document
        native = decode_intent_ir(_json(raw))
        native.validate()
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise FiniteIntegerIntentError("valid exact native IntentIR required") from exc
    if len(native.statements) > 64 or len(native.sources) > 16 or len(native.actions) > 64:
        raise FiniteIntegerIntentError("native intent collection bound exceeded")
    identities = [item.to_dict() for item in sorted(native.sources, key=lambda item: item.ref_id)]
    if source_identity is not None and _json(source_identity) != identities:
        raise FiniteIntegerIntentError("complete native intent source identity differs")
    digest = hashlib.sha256(source_text.encode()).hexdigest()
    for source in native.sources:
        if (source.content_sha256 != digest or source.span is None
                or not 0 <= source.span.start_char < source.span.end_char <= len(source_text)):
            raise FiniteIntegerIntentError("native source digest or clause span differs from full instruction")
    reasons, contract, inputs, rebuilt = [], None, None, None
    try:
        contract, inputs, _ = _parse(source_text)
        first = native.sources[0]
        rebuilt = build_finite_integer_intent(source_text, source_id=first.source_id,
                                             source_revision=first.source_revision)
    except FiniteIntegerIntentError:
        reasons.append("unsupported_full_finite_cnl_source")
    if rebuilt is not None and native.to_dict() != rebuilt.to_dict():
        reasons.append("complete_native_document_or_clause_selector_differs_from_rebuilt_cnl")
    supported = not reasons
    result = {
        "schema": QUERY_SCHEMA, "profile": PROFILE, "cnl_profile": CNL_PROFILE,
        "native_document_sha256": hashlib.sha256(canonical_intent_ir_bytes(native)).hexdigest(),
        "intent_source_sha256": digest, "intent_sources": identities,
        "statements": [{"statement_id": item.statement_id, "predicate": item.predicate,
                        "arguments": list(item.arguments), "kind": item.kind.value,
                        "modality": item.modality.value, "normalized_text": item.normalized_text,
                        "source_ref_ids": list(item.source_ref_ids)}
                       for item in sorted(native.statements, key=lambda item: item.statement_id)],
        "requirement_ids": sorted(item.statement_id for item in native.statements),
        "contract": contract.to_dict() if supported else None,
        "contract_cid": contract.cid if supported else None,
        "domain_inputs": inputs if supported else None,
        "domain_cid": cid_for_structured({"schema": "codebase-finite-integer-domain@1",
                                          "profile": PROFILE, "inputs": inputs}) if supported else None,
        "supported": supported, "reasons": reasons,
        "semantic_alignment_verified": supported,
        "alignment_scope": CNL_PROFILE if supported else "unsupported_full_instruction",
        **_AUTHORITY,
    }
    result["query_cid"] = cid_for_structured(result)
    return _json(result)


def _typed(query, head, source_cid):
    root = head.snapshot_cid
    head_cid = cid_for_structured(head.to_dict())
    refs = (query["query_cid"], head_cid, root, query["intent_source_sha256"])
    desired = []
    for statement in query["statements"]:
        requirement = statement["statement_id"]
        kind = statement["predicate"] if query["supported"] else "unsupported_finite_integer_requirement"
        binding = {"schema": "supervisor-finite-integer-predicate-binding@1", "profile": PROFILE,
                   "requirement_id": requirement, "query_cid": query["query_cid"],
                   "head": head.to_dict(), "source_cid": source_cid,
                   "contract": query["contract"], "domain_cid": query["domain_cid"],
                   "domain_inputs": query["domain_inputs"], "predicate": kind}
        binding_cid = cid_for_structured(binding)
        selectors = (
            InvalidationSelector("root:" + binding_cid, InvalidationSelectorKind.ROOT, root, refs),
            InvalidationSelector("domain:" + binding_cid, InvalidationSelectorKind.POLICY, query["query_cid"], refs),
        )
        if source_cid:
            selectors += (InvalidationSelector("source:" + binding_cid, InvalidationSelectorKind.EVIDENCE, source_cid, refs),)
        if query["contract"]:
            selectors += (InvalidationSelector("path:" + binding_cid, InvalidationSelectorKind.PATH,
                                               query["contract"]["path"], refs),)
        desired.append(TypedPredicate(
            predicate_id="finite-requirement:" + binding_cid,
            predicate_type=kind + "_under_python_integer_offset_finite_v1",
            subject_ref=head_cid, object_ref=binding_cid, property_id=kind,
            support=SemanticSupport.REVIEWED if query["supported"] else SemanticSupport.UNSUPPORTED,
            provenance_refs=refs + ((source_cid,) if source_cid else ()),
            invalidation_selectors=selectors,
        ))
    return TypedIntent(
        intent_id="finite-intent:" + query["query_cid"], desired_predicates=tuple(desired),
        source_refs=refs, current_root_id=root,
        metadata={"schema": "supervisor-finite-integer-intent-binding@1", "profile": PROFILE,
                  "query_cid": query["query_cid"], "head": head.to_dict(), "source_cid": source_cid,
                  "domain_cid": query["domain_cid"], "domain_inputs": query["domain_inputs"],
                  "requirement_predicate_ids": {item["statement_id"]: predicate.predicate_id
                                                for item, predicate in zip(query["statements"], desired)},
                  "scope": "explicit_finite_domain_only", **_AUTHORITY},
    )


def _check_observation(*, observation, index, head, contract, inputs, tool_policy, output):
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import (
        validate_finite_integer_observation,
    )
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import compile_integer_offset

    checked = _json(observation)
    checked = validate_finite_integer_observation(checked, expected_head=head, contract=contract,
                                                  inputs=inputs, tool_policy=tool_policy)
    if (canonical_dag_json_bytes(checked["head"]) != canonical_dag_json_bytes(head.to_dict())
            or canonical_dag_json_bytes(checked["contract"]) != canonical_dag_json_bytes(contract.to_dict())
            or checked["contract_cid"] != contract.cid
            or canonical_dag_json_bytes(checked["domain_inputs"]) != canonical_dag_json_bytes(inputs)
            or canonical_dag_json_bytes(checked["tool_policy"]) != canonical_dag_json_bytes(tool_policy)
            or checked["profile"] != PROFILE or checked["output"] != str(output)
            or any(checked[name] is not False for name in _AUTHORITY)):
        raise FiniteIntegerIntentError("owner observation domain, source, contract or authority differs")
    manifest = index.load(head.manifest_cid)
    if manifest.snapshot.snapshot_cid != head.snapshot_cid:
        raise FiniteIntegerIntentError("owner observation manifest and source root differ")
    leaf = next((entry for entry in manifest.snapshot.entries if entry.path == contract.path), None)
    if checked["source_cid"] != (leaf.source_cid if leaf is not None and not leaf.is_opaque else None):
        raise FiniteIntegerIntentError("finite observation does not bind current captured source")
    if checked["status"] == "observed":
        if leaf is None or leaf.is_opaque or checked["source_cid"] != leaf.source_cid:
            raise FiniteIntegerIntentError("finite observation does not bind current captured source")
        source = index.artifacts.get_bytes(leaf.source_cid)
        compiled = compile_integer_offset(source, contract, revision="snapshot:" + head.snapshot_cid)
        if (cid_for_bytes(source) != checked["source_cid"]
                or hashlib.sha256(source).hexdigest() != checked["source_sha256"]
                or compiled.cid != checked["compiled_cid"]):
            raise FiniteIntegerIntentError("finite observation source or independent lowering differs")
        rows = checked["observations"]
        if (type(rows) is not list or len(rows) != len(inputs)
                or any(type(row) is not dict or set(row) != {"input", "output", "input_type", "output_type"}
                       or type(row["input"]) is not int or type(row["output"]) is not int
                       or row["input_type"] != "int" or row["output_type"] != "int"
                       for row in rows)
                or [row["input"] for row in rows] != inputs
                or checked["type_clause_satisfied"] is not True
                or checked["offset_clause_satisfied"] is not all(row["output"] == row["input"] + contract.offset for row in rows)
                or checked["runtime_observation_coverage_complete"] is not True
                or checked["kernel_checked_model_table"] is not True):
            raise FiniteIntegerIntentError("complete exact finite observation rows and checked table required")
    return checked


def match_finite_integer_intent(
    *, index, repository, repository_id: str, intent_document, source_text: str,
    expected_head, output, tool_policy, source_identity=None, scheduler=None,
    parent_lease=None, cancel_event=None, admission_timeout_seconds: float = 30.0,
    timeout_seconds: float = 120.0, memory_mb: int = 1024,
) -> dict[str, Any]:
    """Return finite facts only after a fresh native run and live exit fence.

    The output must be a fresh canonical absolute directory. A stored receipt,
    cache row, caller flag or supplied evidence cannot bypass the native call.
    All requirement clauses remain in ``typed_intent`` even when satisfied.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    query = prepare_finite_integer_query(intent_document=intent_document, source_text=source_text,
                                         source_identity=source_identity)
    if type(expected_head) is not CodebaseHead or expected_head.repository_id != repository_id:
        raise FiniteIntegerIntentError("explicit canonical owner head for this repository identity required")
    tool_policy = _json(tool_policy)
    if type(output) is not str and not isinstance(output, Path):
        raise FiniteIntegerIntentError("fresh canonical absolute observation output required")
    target = Path(output)
    if not target.is_absolute() or target != target.resolve() or target.exists() or target.is_symlink():
        raise FiniteIntegerIntentError("fresh canonical absolute observation output required")
    repository_root = Path(repository).resolve(strict=True)
    if target == repository_root or repository_root in target.parents:
        raise FiniteIntegerIntentError("finite observation output must be outside the tested repository")
    if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise FiniteIntegerIntentError("finite positive overall timeout within 300 seconds required")
    if type(admission_timeout_seconds) not in {int, float} or not math.isfinite(admission_timeout_seconds) or admission_timeout_seconds < 0:
        raise FiniteIntegerIntentError("finite nonnegative admission timeout required")
    if type(memory_mb) is not int or memory_mb < 1024:
        raise FiniteIntegerIntentError("finite observation requires at least 1024 MiB")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise FiniteIntegerIntentError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("finite integer intent observation cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("finite integer intent observation deadline exceeded")
        return duration

    def build(context):
        observation = None
        source_cid = None
        if query["supported"]:
            from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import (
                observe_finite_integer_source,
            )
            contract = IntegerOffsetContract.from_dict(query["contract"])
            duration = remaining()
            observation = observe_finite_integer_source(
                index=index, repository=repository, expected_head=context.head, contract=contract,
                inputs=query["domain_inputs"], output=target, tool_policy=tool_policy,
                scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
                admission_timeout_seconds=min(admission_timeout_seconds, duration),
                timeout_seconds=duration, memory_mb=memory_mb,
            )
            remaining()
            observation = _check_observation(observation=observation, index=index, head=context.head,
                                             contract=contract, inputs=query["domain_inputs"],
                                             tool_policy=tool_policy, output=target)
            if observation["domain_cid"] != query["domain_cid"]:
                raise FiniteIntegerIntentError("native finite input-domain identity differs")
            source_cid = observation["source_cid"]
        typed = _typed(query, context.head, source_cid)
        # TypedIntent canonicalizes predicates by their content-derived IDs.
        # Their order need not match the independently sorted clause inventory.
        # Preserve the explicit clause binding when constructing facts/results.
        predicates_by_id = {item.predicate_id: item for item in typed.desired_predicates}
        predicates = {requirement: predicates_by_id[predicate_id]
                      for requirement, predicate_id in typed.metadata["requirement_predicate_ids"].items()}
        clauses, facts, eligible, residual, counterexamples = [], [], [], [], []
        for requirement in query["requirement_ids"]:
            status = "open"
            reason = list(query["reasons"])
            if observation is not None and observation["status"] == "observed":
                satisfied = observation["type_clause_satisfied"] if requirement == TYPE_STATEMENT_ID else observation["offset_clause_satisfied"]
                status = "bounded_observed_satisfied" if satisfied else "finite_counterexample"
                if not satisfied:
                    reason.append("explicit_domain_observation_contradicts_requested_offset")
                    counterexamples.extend({"statement_id": requirement, "input": row["input"],
                                            "observed_output": row["output"],
                                            "expected_output": row["input"] + contract.offset}
                                           for row in observation["observations"]
                                           if row["output"] != row["input"] + contract.offset)
            elif observation is not None:
                reason.append("native_finite_observation_" + observation["status"])
            if status == "bounded_observed_satisfied":
                predicate = predicates[requirement]
                provenance = (query["query_cid"], cid_for_structured(context.head.to_dict()),
                              context.head.snapshot_cid, source_cid, query["domain_cid"],
                              observation["trace_cid"], observation["result_cid"],
                              cid_for_structured(observation["lean_certificate"]),
                              observation["compiled_cid"], observation["tool_policy_cid"])
                fact_key = {"schema": "supervisor-finite-integer-fact-binding@1", "requirement_id": requirement,
                            "predicate_id": predicate.predicate_id, "head": context.head.to_dict(),
                            "source_cid": source_cid, "domain_cid": query["domain_cid"],
                            "observation_cid": observation["result_cid"], "trace_cid": observation["trace_cid"]}
                fact = ObservedFact(
                    fact_id="finite-observation:" + cid_for_structured(fact_key), predicate=predicate,
                    truth=FactTruth.TRUE, authority=FactAuthority.BOUNDED_OBSERVATION,
                    provenance_refs=provenance, current_root_id=context.head.snapshot_cid,
                    invalidation_selectors=predicate.invalidation_selectors,
                )
                facts.append(fact.to_dict())
                eligible.append(requirement)
            else:
                residual.append(requirement)
            clauses.append({"statement_id": requirement, "predicate_id": predicates[requirement].predicate_id,
                            "status": status, "reasons": reason, "scope": "explicit_finite_domain_only"})
        remaining()
        result = {
            "schema": SCHEMA, "profile": PROFILE, "query": query,
            "status": "bounded_observed_satisfied" if not residual else "finite_counterexample" if counterexamples else "open",
            "head": context.head.to_dict(), "current_root_id": context.head.snapshot_cid,
            "structural_context": context.to_dict(), "source_cid": source_cid,
            "domain_cid": query["domain_cid"], "domain_inputs": query["domain_inputs"],
            "typed_intent": typed.to_dict(), "current_facts": facts,
            "clause_results": clauses, "eligible_clause_ids": eligible, "residual_clause_ids": residual,
            "eligible_requirements": eligible,
            "residual_requirements": [item for item in clauses if item["statement_id"] in residual],
            "finite_counterexamples": counterexamples, "observation": observation,
            "observation_cid": observation["result_cid"] if observation else None,
            "semantic_alignment_verified": query["semantic_alignment_verified"],
            "alignment_scope": query["alignment_scope"], "scope": "explicit_finite_domain_only",
            "unbounded_behavior_status": "unresolved", "removed_task_ids": [], **_AUTHORITY,
        }
        result["match_cid"] = cid_for_structured(result)
        return _json(result)

    return run_with_structural_codebase_context(
        index, repository, build, repository_id=repository_id, expected_head=expected_head,
        scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
        admission_timeout_seconds=admission_timeout_seconds, timeout_seconds=remaining(), memory_mb=memory_mb,
    )


__all__ = ["SCHEMA", "QUERY_SCHEMA", "CNL_PROFILE", "PROFILE", "TYPE_STATEMENT_ID", "OFFSET_STATEMENT_ID",
           "TYPE_PREDICATE", "OFFSET_PREDICATE", "FiniteIntegerIntentError", "build_finite_integer_intent",
           "prepare_finite_integer_query", "match_finite_integer_intent"]
