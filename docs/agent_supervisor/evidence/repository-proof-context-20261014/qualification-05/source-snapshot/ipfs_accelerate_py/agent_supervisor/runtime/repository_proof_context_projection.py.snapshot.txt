"""Pure bounded projection of a conditional local header proof into context.

This validates canonical correspondences, not admission or currentness. The
caller must independently admit the original proof/cache/source owners and
bracket this operation with their live checks. No owner, cache, filesystem,
model, verifier or execution is constructed or consulted here.
"""
from __future__ import annotations

import json
import re

from ..context.context_contracts import ContextReference, ContextTier
from ..proof.deterministic_doctor_hammer import DoctorReviewedTheorem
from ..proof.doctor_proof_cache import (
    DoctorProofCacheGate, DoctorProofCacheKey, _contains_private_material,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel, EvidenceFreshness, LocalProofScope, ProofReceipt,
    ProofVerdict, ScopedProofReceipt, canonical_json, content_identity,
)
from .doctor_contract_key_custody import (
    HEADER_SOURCE_PREMISE_SCHEMA, _source_observations, restore_header_source_premise,
)

CONDITIONS_SCHEMA = "supervisor-repository-header-proof-conditions@1"
PROOF_SCHEMA = "supervisor-repository-header-proof-handle@1"
SOURCE_BINDING_SCHEMA = "supervisor-repository-proof-source-binding@1"
CONDITIONS_PREFIX = "repository-proof-conditions:"
PROOF_PREFIX = "repository-proof:"
MAX_BODY_BYTES = 32 * 1024
MAX_PUBLIC_PROJECTION_BYTES = 256 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_FIELDS = {
    "schema", "source_correspondence", "session_id", "generation",
    "session_repository_id", "proof_repository_id", "signed_manifest_cid",
    "task_cid", "scope_id", "local_source_tree_id", "source_tree_sha256",
    "package_manifest_sha256", "semantic_state_cid", "native_head",
    "file_count", "directory_count", "source_bytes", "scoped_file_count",
    "omitted_source_count", "full_signed_population_joined",
    "view_metadata_separately_bound", "git_tree_equivalence_verified",
    "proof_authority", "execution_authority", "mutation_authority",
    "completion_authority", "omission_authority", "whole_program_proved",
}
_FALSE_FIELDS = (
    "proof_authority", "execution_authority", "mutation_authority",
    "completion_authority", "omission_authority", "whole_program_proved",
)
_RECEIPT_METADATA = {
    "schema", "full_key_id", "native_proof_cid", "native_authority_entry_cid",
    "native_custody_receipt_cid", "generic_receipt_implementations",
    "source_premise_profile_cid", "open_frontiers",
    "security_ir_obligations_discharged", "completion_authority",
    "execution_authority", "mutation_authority", "whole_program_proved",
    "resource_enforcement_verified",
}
_CONTRACT_FIELDS = {
    "schema", "operator_id", "source_sha256", "protocol", "normative_reference",
    "forbidden_codepoints", "error_type", "accepted_behavior", "proof_scope",
    "assumptions", "open_frontiers", "normalizers", "proof_obligations",
}
_SECURITY_FIELDS = {"security_ir_declaration_cid", "security_ir_formalization_cid"}


class RepositoryProofContextProjectionError(ValueError):
    """A public local-proof projection lost its exact bounded conditions."""


def _require(value, message):
    if not value:
        raise RepositoryProofContextProjectionError(message)


def _json(value, maximum=MAX_PUBLIC_PROJECTION_BYTES):
    rendered = canonical_json(value)
    _require(len(rendered.encode("utf-8")) <= maximum, "public proof projection exceeds byte bound")
    return rendered


def _pairs(rows):
    result = {}
    for key, value in rows:
        _require(key not in result, "duplicate public proof projection field")
        result[key] = value
    return result


def _decode(raw):
    _require(type(raw) is bytes and len(raw) <= MAX_PUBLIC_PROJECTION_BYTES,
             "bounded canonical public proof preimage required")
    value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            RepositoryProofContextProjectionError("nonfinite public proof projection")))
    _require(type(value) is dict and _json(value).encode("utf-8") == raw,
             "exact canonical public proof preimage required")
    return value


def _text(value):
    return (type(value) is str and value and value == value.strip()
            and len(value.encode("utf-8")) <= 4096
            and all(ord(character) >= 32 for character in value))


def _strings(values, *, nonempty=False):
    _require(type(values) in (list, tuple) and len(values) <= 256
             and (bool(values) or not nonempty) and all(_text(value) for value in values)
             and len(values) == len(set(values)), "complete bounded public condition identities required")
    return tuple(values)


def _source_binding(value):
    _require(type(value) is dict and set(value) == _SOURCE_FIELDS,
             "closed public repository proof source observation required")
    value = json.loads(_json(value), object_pairs_hook=_pairs)
    _require(value["schema"] == SOURCE_BINDING_SCHEMA
             and value["source_correspondence"] == "complete-signed-regular-file-population@1"
             and value["full_signed_population_joined"] is True
             and value["view_metadata_separately_bound"] is True
             and value["git_tree_equivalence_verified"] is False
             and all(value[name] is False for name in _FALSE_FIELDS),
             "source observation must retain its correspondence and authority boundaries")
    for field in ("source_tree_sha256", "package_manifest_sha256"):
        _require(type(value[field]) is str and _SHA256.fullmatch(value[field]) is not None,
                 "exact session source digest required")
    for field in ("session_id", "session_repository_id", "proof_repository_id",
                  "signed_manifest_cid", "task_cid", "scope_id", "local_source_tree_id",
                  "semantic_state_cid"):
        _require(_text(value[field]), "bounded source namespace identity required")
    _require(type(value["generation"]) is int and 1 <= value["generation"] <= 8
             and type(value["native_head"]) is dict,
             "exact source generation and native head observation required")
    for field in ("file_count", "directory_count", "source_bytes", "scoped_file_count",
                  "omitted_source_count"):
        _require(type(value[field]) is int and value[field] >= 0,
                 "exact nonnegative source population count required")
    _require(value["file_count"] > 0 and value["scoped_file_count"] > 0
             and value["scoped_file_count"] + value["omitted_source_count"] == value["file_count"]
             and not _contains_private_material(value),
             "complete public signed source population required")
    return value


def _premise(binding):
    value = _decode(binding.canonical_bytes)
    _require(value.get("schema") == HEADER_SOURCE_PREMISE_SCHEMA
             and type(value.get("contract_fields")) is dict,
             "reviewed public header source premise required")
    fields = value["contract_fields"]
    security = set(fields) & _SECURITY_FIELDS
    _require(set(fields) == _CONTRACT_FIELDS | security
             and (not security or security == _SECURITY_FIELDS)
             and fields["schema"] == "supervisor-http-header-local-contract@1"
             and type(fields["source_sha256"]) is str
             and _SHA256.fullmatch(fields["source_sha256"]) is not None,
             "closed complete public header source contract required")
    _require(all(_text(fields[name]) for name in (
        "operator_id", "normative_reference", "error_type", "accepted_behavior", "proof_scope"))
        and all(_text(fields[name]) for name in security)
        and fields["forbidden_codepoints"] == [0, 10, 13]
        and all(type(value) is int for value in fields["forbidden_codepoints"]),
        "exact public header contract vocabulary required")
    _strings(fields["assumptions"], nonempty=True)
    _strings(fields["open_frontiers"], nonempty=True)
    _strings(fields["proof_obligations"], nonempty=True)
    _source_observations(value.get("ast_source_observations"), protocol=fields["protocol"],
        normalizers=fields["normalizers"], maximum_line=(1 << 31) - 1)
    restored = restore_header_source_premise(value,
        expected_contract=dict(fields, witnesses=value["ast_source_observations"]))
    _require(binding.logical_id == value["original_contract_cid"] == content_identity(restored)
             and binding.cid == content_identity(value),
             "source premise differs from its original contract or public artifact identity")
    return value


def project_repository_header_proof(*, key, receipt, source_binding):
    """Return mandatory conditions and an optional handle, without admission.

The complete theorem and public source premises are model-visible data in the
mandatory reference. Original proof namespaces survive beside the session
namespace. Conditions are never truncated; an insufficient compiler budget
must refuse them. A digest-only proof handle cannot remove those conditions.
"""
    try:
        _require(type(key) is DoctorProofCacheKey and type(receipt) is ScopedProofReceipt
                 and type(receipt.scope) is LocalProofScope,
                 "exact native key and conditional local receipt required")
        key_body = _json(key.to_dict())
        restored_key = DoctorProofCacheKey.from_dict(json.loads(key_body))
        _require(_json(restored_key.to_dict()) == key_body and not key.contains_private_material,
                 "canonical public complete Doctor key required")
        restored_receipt = ProofReceipt.from_dict(json.loads(_json(receipt.to_dict())))
        _require(type(restored_receipt) is ScopedProofReceipt
                 and _json(restored_receipt.to_dict()) == _json(receipt.to_dict())
                 and receipt.verdict is receipt.authoritative_verdict is ProofVerdict.PROVED
                 and receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
                 and key.required_assurance is AssuranceLevel.KERNEL_VERIFIED
                 and receipt.freshness is EvidenceFreshness.CURRENT
                 and not DoctorProofCacheGate._receipt_reasons(key, receipt)
                 and receipt.satisfies_completion() is False,
                 "current exact kernel-checked local receipt correspondence required")
        source = _source_binding(source_binding)
        theorem_body = _decode(key.goal.canonical_bytes)
        theorem = DoctorReviewedTheorem.from_dict(theorem_body)
        scope = receipt.scope
        _require(_json(theorem.to_dict()) == _json(theorem_body)
                 and key.goal.logical_id == theorem.theorem_id == receipt.obligation_id
                 and key.goal.cid == theorem.content_id == scope.theorem_cid
                 and scope.property_id == theorem.property_id
                 and scope.consequence_ref == theorem.consequence_ref
                 and scope.premise_ids == theorem.premise_ids == receipt.premise_ids
                 and scope.assumption_ids == theorem.assumption_ids
                 and receipt.ast_scope_ids == (key.ast.logical_id,)
                 and receipt.translator_id == theorem.translator_id
                 and receipt.toolchain_id == theorem.toolchain_id
                 and receipt.policy_id == theorem.policy_id
                 and receipt.repository_id == theorem.roots.repository_id == source["proof_repository_id"]
                 and key.tree.logical_id == key.candidate_tree.logical_id == source["scope_id"]
                 and theorem.roots.tree_id == source["scope_id"]
                 and key.forest.logical_id == theorem.roots.forest_id == source["scope_id"]
                 and key.overlay.logical_id == theorem.roots.overlay_id == source["scope_id"]
                 and key.graph.logical_id == theorem.roots.graph_id
                 and key.corpus.logical_id == theorem.roots.corpus_id
                 and key.environment.logical_id == theorem.roots.environment_id,
                 "complete local theorem scope and source namespace correspondences required")
        _require(1 <= len(key.premises) <= 32, "complete bounded header source premises required")
        premises = [_premise(binding) for binding in key.premises]
        _require(tuple(binding.logical_id for binding in key.premises) == scope.premise_ids,
                 "all public source premises must match the local scope")
        metadata = json.loads(_json(receipt.metadata))
        _require(set(metadata) == _RECEIPT_METADATA
                 and metadata["schema"] == "supervisor-doctor-header-cache-bridge@1"
                 and metadata["full_key_id"] == key.key_id
                 and metadata["source_premise_profile_cid"] == key.premises[0].cid
                 and all(metadata[name] is False for name in (
                     "security_ir_obligations_discharged", "completion_authority",
                     "execution_authority", "mutation_authority", "whole_program_proved",
                     "resource_enforcement_verified")),
                 "complete local receipt conditions and authority limits required")
        implementations = metadata["generic_receipt_implementations"]
        _require(type(implementations) is dict
                 and set(implementations) == {"bridge_sha256", "receipt_contracts_sha256"}
                 and all(type(value) is str and _SHA256.fullmatch(value) is not None
                         for value in implementations.values())
                 and all(_text(metadata[name]) for name in (
                     "native_proof_cid", "native_authority_entry_cid", "native_custody_receipt_cid")),
                 "closed native receipt implementation and custody identities required")
        frontiers = _strings(metadata["open_frontiers"], nonempty=True)
        _require(all(set(premise["contract_fields"]["open_frontiers"]) <= set(frontiers)
                     for premise in premises), "declared source frontiers cannot be omitted")
        security_present = any(_SECURITY_FIELDS <= set(premise["contract_fields"]) for premise in premises)
        _require(bool(scope.residual_obligation_ids) == security_present,
                 "SecurityIR declarations must retain their declared residual obligations")
        _json(dict(source_binding=source, theorem=theorem_body, source_premises=premises))
        conditions_id = CONDITIONS_PREFIX + key.key_id
        required_body = dict(schema=CONDITIONS_SCHEMA, key_id=key.key_id,
            receipt_id=receipt.receipt_id, original_proof_repository_id=receipt.repository_id,
            original_proof_scope_cid=key.tree.logical_id, source_binding=source,
            local_scope=scope.to_dict(), public_theorem=theorem_body,
            public_source_premises=premises, open_frontiers=list(frontiers),
            security_ir_obligations_discharged=False, conditional_local_only=True,
            projection_is_admission=False, **{name: False for name in _FALSE_FIELDS})
        required_json = _json(required_body, MAX_BODY_BYTES)
        common = dict(repository_id=source["session_repository_id"],
                      tree_id="sha256:" + source["source_tree_sha256"])
        required = ContextReference(reference_id=conditions_id,
            kind="conditional-local-proof-conditions", tier=ContextTier.INVARIANT,
            referenced_content_id=content_identity(required_body), summary=required_json,
            byte_count=len(required_json.encode("utf-8")), **common,
            metadata=dict(required=True, coverage_ids=("coverage:" + conditions_id,),
                conditional_local_only=True, projection_is_admission=False,
                proof_authority=False, execution_authority=False, completion_authority=False))
        optional_body = dict(schema=PROOF_SCHEMA, key_id=key.key_id,
            receipt_id=receipt.receipt_id, theorem_cid=scope.theorem_cid,
            conditions_reference_id=conditions_id, conditional_local_only=True,
            projection_is_admission=False, **{name: False for name in _FALSE_FIELDS})
        optional_json = _json(optional_body, MAX_BODY_BYTES)
        optional = ContextReference(reference_id=PROOF_PREFIX + key.key_id,
            kind="conditional-local-proof-handle", tier=ContextTier.EVIDENCE,
            referenced_content_id=receipt.receipt_id, summary=optional_json,
            byte_count=len(optional_json.encode("utf-8")), **common,
            metadata=dict(required=False, digest_only=True, conditions_reference_id=conditions_id,
                conditional_local_only=True, projection_is_admission=False,
                proof_authority=False, execution_authority=False, completion_authority=False))
        return required, optional
    except RepositoryProofContextProjectionError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError) as error:
        raise RepositoryProofContextProjectionError("invalid complete public local-proof projection") from error


__all__ = ["RepositoryProofContextProjectionError", "project_repository_header_proof",
           "CONDITIONS_SCHEMA", "PROOF_SCHEMA", "CONDITIONS_PREFIX", "PROOF_PREFIX",
           "MAX_BODY_BYTES", "MAX_PUBLIC_PROJECTION_BYTES"]
