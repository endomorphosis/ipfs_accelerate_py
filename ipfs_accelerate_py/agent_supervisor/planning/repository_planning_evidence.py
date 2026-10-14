"""Complete reviewed domain evidence for a current repository planning context.

This finite profile retains the full native Intent requirement contract, never
a selection of its mandatory rows. Optional SecurityIR evidence is recomputed
by its existing scoped owner; unsupported analyses and omitted sources remain
visible. UI/UX evidence is explicitly unavailable in this profile. Declarations
and source bindings confer no proof, mutation, or completion authority.

Bounds are 32 complete Intent ledger rows, 60 KiB per complete domain body,
8,192 JSON values and depth 48 before native validation, and 64 SecurityIR
analysis rows and scoped sources. A cooperative 30-second deadline includes resource admission
(256 MiB reservation, at most 10 seconds). These are bounded input/admission
controls, not a hard RSS or CPU sandbox. Every overflow refuses the entire
result; no required domain body is shortened.
"""
from __future__ import annotations

import hashlib
import json
import time

from ..context.context_contracts import ContextReference, ContextTier
from ..prompt.intent_plan_coverage import validate_intent_requirement_contract
from ..runtime.repository_source_session import RepositorySourceSession
from ..runtime.supervisor_meta_index import public_replay_without_metadata

MAX_REQUIREMENT_ROWS = 32
MAX_BODY_BYTES = 60 * 1024
MAX_JSON_VALUES = 8192
MAX_JSON_DEPTH = 48
_FALSE = dict(proof_authority=False, execution_authority=False,
              mutation_authority=False, completion_authority=False,
              semantic_alignment_verified=False)


class RepositoryPlanningEvidenceError(ValueError):
    """A bounded required population lost its native or current source binding."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryPlanningEvidenceError(code)


def _json(value):
    """Bound the input before serialization and reject instead of abbreviating."""
    count, text_bytes = 0, 0

    def visit(item, depth=0):
        nonlocal count, text_bytes
        count += 1
        _require(count <= MAX_JSON_VALUES and depth <= MAX_JSON_DEPTH,
                 "planning_evidence_structure_limit")
        if type(item) is str:
            text_bytes += len(item.encode("utf-8"))
            _require(text_bytes <= MAX_BODY_BYTES, "planning_evidence_body_limit")
        elif type(item) is dict:
            for key, member in item.items():
                _require(type(key) is str, "invalid_planning_evidence_json")
                visit(key, depth + 1)
                visit(member, depth + 1)
        elif type(item) in (list, tuple):
            for member in item:
                visit(member, depth + 1)
        else:
            _require(item is None or type(item) in (bool, int, float),
                     "invalid_planning_evidence_json")

    visit(value)
    try:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RepositoryPlanningEvidenceError("invalid_planning_evidence_json") from exc
    _require(len(body.encode("utf-8")) <= MAX_BODY_BYTES, "planning_evidence_body_limit")
    return body


def _binding(session, generation):
    status = session.status_summary()
    _require(type(generation) is int and generation == status["generation"],
             "stale_planning_evidence_generation")
    _require(status["source_current"] is True, "stale_planning_evidence_source")
    return {"session_id": status["session_id"], "generation": generation,
            "repository_id": status["datasets_head"]["repository_id"],
            "tree_id": "sha256:" + status["source_tree_sha256"],
            "semantic_state_cid": status["semantic_state_cid"]}


def _read(session, generation, path, binding):
    value = session.read(generation=generation, path=path, representation="original")
    _require(value["encoding"] == "utf-8"
             and value["session_id"] == binding["session_id"]
             and value["generation"] == binding["generation"]
             and value["datasets_head"]["repository_id"] == binding["repository_id"]
             and "sha256:" + value["source_tree_sha256"] == binding["tree_id"]
             and value["semantic_state_cid"] == binding["semantic_state_cid"],
             "planning_evidence_source_binding_differs")
    raw = value["content"].encode("utf-8")
    _require(hashlib.sha256(raw).hexdigest() == value["original_sha256"],
             "planning_evidence_source_digest_differs")
    return value["content"], value["original_sha256"]


def _reference(domain, body, binding, *, path="", coverage_ids=(), source_sha256=""):
    raw = body.encode("utf-8")
    return ContextReference(
        reference_id="planning-required:" + domain, kind="repository-planning-" + domain,
        tier=ContextTier.INVARIANT, referenced_content_id="sha256:" + hashlib.sha256(raw).hexdigest(),
        repository_id=binding["repository_id"], tree_id=binding["tree_id"],
        path=path, summary=body, byte_count=len(raw), metadata=dict(
            required=True, coverage_ids=("planning-domain:" + domain, *coverage_ids),
            source_sha256=source_sha256, generation=binding["generation"],
            evidence_population="complete_bounded_domain", **_FALSE))


def _unavailable(domain, reason):
    return _json(dict(schema="repository-planning-domain-status@1", domain=domain,
        status="unavailable", reason=reason, source_semantics_verified=False,
        whole_repository_analysis=False, **_FALSE))


def build_repository_planning_evidence(*, session, generation, intent_contract,
                                       security_scope=None, security_analyses=None):
    """Return three complete required references bound to the current session.

    ``intent_contract`` is independently reviewed input, validated against its
    exact current original source. ``security_scope`` and ``security_analyses``
    must be supplied together and are passed to the existing native compiler;
    arbitrary saved reports are not accepted. No learned IR is synthesized and
    no target source, provider, solver, or publication operation is executed.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources

    _require(isinstance(session, RepositorySourceSession), "invalid_planning_evidence_session")
    _require((security_scope is None) == (security_analyses is None),
             "incomplete_planning_security_inputs")
    deadline = time.monotonic() + 30.0
    # Copy bounded JSON before native validation, so caller mutation cannot
    # change the source path or full ledger between validation and inclusion.
    contract = json.loads(_json(intent_contract))
    _require(type(contract) is dict and type(contract.get("source_path")) is str,
             "invalid_planning_intent_contract")
    ledger = contract.get("ledger")
    _require(type(ledger) is dict and type(ledger.get("requirements")) is list,
             "invalid_planning_intent_contract")
    _require(len(ledger["requirements"]) <= MAX_REQUIREMENT_ROWS,
             "planning_intent_population_limit")
    with public_replay_without_metadata(), acquire_codebase_resources(
            scheduler=getattr(session._index, "_scheduler", None),
            timeout_seconds=10.0, memory_mb=256) as lease:
        def checkpoint():
            _require(not lease.cancelled, "planning_evidence_cancelled")
            _require(time.monotonic() < deadline, "planning_evidence_timeout")

        checkpoint()
        binding = _binding(session, generation)
        source, digest = _read(session, generation, contract["source_path"], binding)
        try:
            contract = validate_intent_requirement_contract(contract, source_text=source)
        except (TypeError, ValueError) as exc:
            raise RepositoryPlanningEvidenceError("invalid_planning_intent_contract") from exc
        ledger = contract["ledger"]
        _require(ledger["source_report_schema"] == "intent-reviewed-source-report@1",
                 "reviewed_planning_intent_required")
        rows = ledger["requirements"]
        _require(len(rows) <= MAX_REQUIREMENT_ROWS, "planning_intent_population_limit")
        requirement_ids = tuple(row["requirement_id"] for row in rows)
        intent = _reference("intent_ir", _json(contract), binding, path=contract["source_path"],
                            coverage_ids=requirement_ids, source_sha256=digest)
        checkpoint()
        if security_scope is None:
            security_body = _unavailable("security_ir", "no_current_scoped_security_owner_supplied")
        else:
            from ..runtime.doctor_scoped_analysis import ScopedDoctorAnalysis
            from ..runtime.doctor_security_ir import compile_header_security_ir

            _require(type(security_scope) is ScopedDoctorAnalysis
                     and security_scope.repository == session.workspace,
                     "planning_security_scope_differs")
            analyses = json.loads(_json(security_analyses))
            _require(type(analyses) is list and len(analyses) <= 64,
                     "planning_security_population_limit")
            security_scope.assert_current()
            observed = security_scope.report
            _require(set(observed["source_hashes"]) == set(security_scope.sources),
                     "planning_security_population_differs")
            _require(len(observed["source_hashes"]) <= 64, "planning_security_population_limit")
            for path, expected_digest in observed["source_hashes"].items():
                checkpoint()
                text, actual_digest = _read(session, generation, path, binding)
                _require(actual_digest == expected_digest
                         and text.encode("utf-8") == security_scope.sources[path],
                         "planning_security_source_differs")
            checkpoint()
            compiled = compile_header_security_ir(scoped=security_scope, analyses=analyses)
            security_body = _json(compiled.report)
            security_scope.assert_current()
        security = _reference("security_ir", security_body, binding)
        ui = _reference("ui_ux_ir", _unavailable("ui_ux_ir",
                         "current_repository_ui_ux_owner_not_connected"), binding)
        checkpoint()
        _require(_binding(session, generation) == binding, "planning_evidence_source_binding_differs")
        checkpoint()
        return intent, security, ui


__all__ = ["RepositoryPlanningEvidenceError", "build_repository_planning_evidence"]
