"""Closed local intent/residual queries over the fresh native evidence owner.

The complete finite-profile requirement population is returned atomically.
These queries never accept a saved match, injected fact, or omission authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument
from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from .repository_code_evidence import RepositoryCodeEvidence, RepositoryCodeEvidenceError, FALSE

SCHEMA = "repository-typed-intent-evidence@1"
MAX_INPUT_BYTES = 64 * 1024


def _require(value, message):
    if not value:
        raise RepositoryCodeEvidenceError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _object(raw):
    _require(type(raw) is str and 0 < len(raw.encode()) <= MAX_INPUT_BYTES, "bounded canonical JSON required")
    value = json.loads(raw)
    _require(type(value) is dict and _wire(value) == raw, "exact canonical object required")
    return value


@dataclass(frozen=True)
class IntentResolutionQuery:
    source_text: str
    intent_document_json: str
    tool_policy_json: str
    max_requirements: int = 16

    def __post_init__(self):
        _require(type(self.source_text) is str and 0 < len(self.source_text.encode()) <= MAX_INPUT_BYTES,
                 "bounded complete source instruction required")
        document = _object(self.intent_document_json)
        _require(decode_intent_ir(document).to_dict() == document, "exact native Intent document required")
        _object(self.tool_policy_json)
        _require(type(self.max_requirements) is int and 1 <= self.max_requirements <= 16,
                 "bounded complete requirement population required")
        _require(0 < len(document["statements"]) <= self.max_requirements,
                 "complete native statement population exceeds query bound")

    @classmethod
    def from_native(cls, *, source_text, intent_document, tool_policy, max_requirements=16):
        _require(type(intent_document) is IntentIRDocument, "exact native Intent owner type required")
        return cls(source_text, _wire(intent_document.to_dict()), _wire(tool_policy), max_requirements)

    def to_dict(self):
        return dict(kind="intent_resolution", source_text=self.source_text,
            intent_document_json=self.intent_document_json, tool_policy_json=self.tool_policy_json,
            max_requirements=self.max_requirements)


@dataclass(frozen=True)
class ResidualObligationsQuery(IntentResolutionQuery):
    def to_dict(self):
        return {**super().to_dict(), "kind": "residual_obligations"}


class RepositoryIntentEvidence:
    def __init__(self, plane):
        _require(type(plane) is RepositoryCodeEvidence, "exact current native evidence facade required")
        self.plane = plane

    def _query(self, query, expected_type, *, repository, expected_head,
               semantic_manifest_cid, output, **resources):
        _require(type(query) is expected_type, "exact closed query type required")
        # Reconstruct even frozen instances: no hidden caller-mutated fields.
        query = expected_type(**{key: value for key, value in query.to_dict().items()
                                 if key != "kind"})
        pin = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        native = self.plane.resolve_intent(repository=repository, expected_head=expected_head,
            semantic_manifest_cid=semantic_manifest_cid, source_text=query.source_text,
            intent_document=_object(query.intent_document_json), tool_policy=_object(query.tool_policy_json),
            output=output, **resources)
        ids = native["complete_requirement_ids"]
        all_rows = native["match"]["requirement_results"]
        residual = native["residual_obligations"]
        _require(0 < len(ids) <= query.max_requirements and len(ids) == len(set(ids))
                 and sorted(row["statement_id"] for row in all_rows) == ids,
                 "complete requirement population exceeds query bound or differs")
        _require(native["selection_complete"] is True and not native["reduced_task_population_authorized"]
                 and native["consumed_match_cid"] == native["match"]["match_cid"]
                 and native["result_cid"] == cid_for_structured({k:v for k,v in native.items() if k != "result_cid"}),
                 "native consumed evidence commitments differ")
        _require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == pin, "typed query producer changed")
        root = dict(schema=SCHEMA, query=query.to_dict(), source_head=expected_head.to_dict(),
            semantic_manifest_cid=semantic_manifest_cid, producer_sha256=pin,
            native_result_cid=native["result_cid"], consumed_match_cid=native["consumed_match_cid"],
            consumed_record_commitments=native["consumed_record_commitments"])
        result = dict(schema=SCHEMA, query=query.to_dict(), root=cid_for_structured(root), root_material=root,
            rows=all_rows if expected_type is IntentResolutionQuery else residual,
            complete_requirement_ids=ids, residual_requirement_ids=sorted(row["statement_id"] for row in residual),
            complete_native_result=native, selection_complete=True, next_cursor=None,
            consumed_commitments_are_not_proof_capabilities=True, reduced_task_population_authorized=False, **FALSE)
        result["result_cid"] = cid_for_structured(result)
        return result

    def resolve_intent(self, *, query, **arguments):
        return self._query(query, IntentResolutionQuery, **arguments)

    def residual_obligations(self, *, query, **arguments):
        return self._query(query, ResidualObligationsQuery, **arguments)


__all__ = ["IntentResolutionQuery", "ResidualObligationsQuery", "RepositoryIntentEvidence"]
