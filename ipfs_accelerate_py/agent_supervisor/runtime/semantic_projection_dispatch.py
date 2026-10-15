"""Project verified semantic evidence at the existing router boundary.

The canonical worker artifact and executable source stay unchanged. This
adapter replaces only the semantic evidence references in an already compiled
provider payload; task, authority, scope, acceptance, unrelated evidence and
literal suffixes retain their original values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding

from . import semantic_router_translation as codec
from .semantic_context_projection import (
    SemanticProjectionBindings,
    _encoded,
    build_semantic_context_projection,
    semantic_context_projection_references,
)
from .semantic_decision_packet import (
    SemanticDecisionPolicy,
    build_semantic_decision_packet,
    expand_semantic_decision_packet,
    semantic_decision_packet_references,
)


@dataclass(frozen=True)
class SemanticRouterProjection:
    projected_prompt: str
    receipt_json: str

    @property
    def receipt(self) -> dict:
        return json.loads(self.receipt_json)


def project_semantic_router_prompt(
    *, prompt: str, repository: Path,
    decision_policy: SemanticDecisionPolicy | None = None,
    retrieval_query_binding: SemanticRetrievalQueryBinding | None = None,
    chunk_bytes: int = 6144,
) -> SemanticRouterProjection:
    """Verify the complete original before constructing a derived role view.

    This is a provider-free representation operation. Selection still requires
    the existing router's bound counter over the complete rendered message.
    The producer verifier is repeated after construction to reject changed
    parent artifacts or sources. Expansion is provided by the projection API,
    with an explicit resolver bound to the current native dispatch loader.

    An independently supplied decision_policy opts into capsule body selection.
    That packet is a library/offline boundary only: the current CLI codec does
    not admit it without a provider tool bridge and independent policy custody.
    """
    if decision_policy is not None and type(decision_policy) is not SemanticDecisionPolicy:
        raise ValueError("explicit trusted decision policy required")
    if retrieval_query_binding is not None:
        from .semantic_retrieval_query_reference import (
            SemanticRetrievalQueryBinding,
            build_retrieval_query_reference,
            load_live_retrieval_query_context,
            restore_retrieval_query_reference,
            retrieval_query_reference_references,
            validate_native_retrieval_references,
        )
        if (type(retrieval_query_binding) is not SemanticRetrievalQueryBinding
                or decision_policy is None or decision_policy.task_text_source_path is None):
            raise ValueError("retrieval query binding requires explicit @2 semantic text policy")
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root:
        raise codec.SemanticTranslationError("semantic repository must be canonical")
    wire, suffix, positions, references, text, artifact = codec._prompt_parts(prompt)
    semantic, _ = codec._verified_semantic(
        root, artifact=artifact, text=text, task_id=wire["objective_id"], current=True,
    )
    if semantic.get("schema") != "supervisor-semantic-worker-context@1":
        raise codec.SemanticTranslationError("projection requires the complete native worker artifact")
    bindings = SemanticProjectionBindings(
        artifact=artifact,
        expected_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        task_id=wire["objective_id"],
        semantic_root_cid=semantic["semantic_root_cid"],
        scope_cid=semantic["scope_cid"],
        repository_id=wire["repository_id"],
        tree_id=wire["tree_id"],
    )
    coverage = tuple(sorted({item for reference in references for item in reference.coverage_ids}))
    if decision_policy is None:
        view = build_semantic_context_projection(original_text=text, bindings=bindings)
        projected_references = semantic_context_projection_references(
            view=view, original_text=text, bindings=bindings, coverage_ids=coverage,
            chunk_bytes=chunk_bytes,
        )
    else:
        view = build_semantic_decision_packet(
            original_text=text, bindings=bindings, policy=decision_policy,
        )
        projected_references = semantic_decision_packet_references(
            view=view, original_text=text, bindings=bindings, policy=decision_policy,
            coverage_ids=coverage, chunk_bytes=chunk_bytes,
        )
    if not projected_references or any(not reference.required for reference in projected_references):
        raise codec.SemanticTranslationError("projected decision evidence must remain required")
    if not set(coverage) <= {
        item for reference in projected_references for item in reference.coverage_ids
    }:
        raise codec.SemanticTranslationError("projection changed required evidence coverage")
    retrieval_receipt = None
    retrieval_positions, retrieval_references, retrieval_projected = (), (), ()
    if retrieval_query_binding is not None:
        if any(reference.repository_id != wire["repository_id"] or reference.tree_id != wire["tree_id"]
               for reference in references):
            raise codec.SemanticTranslationError("native semantic reference repository/tree binding differs")
        retrieval_text = load_live_retrieval_query_context(repository=root, binding=retrieval_query_binding,
            semantic_original_text=text, semantic_bindings=bindings)
        retrieval_positions, retrieval_references = validate_native_retrieval_references(
            wire=wire, binding=retrieval_query_binding, native_text=retrieval_text)
        retrieval_coverage = tuple(sorted({item for reference in retrieval_references for item in reference.coverage_ids}))
        retrieval_arguments = dict(native_text=retrieval_text, binding=retrieval_query_binding,
            semantic_view=view, semantic_original_text=text, semantic_bindings=bindings, policy=decision_policy)
        retrieval_view = build_retrieval_query_reference(**retrieval_arguments)
        retrieval_projected = retrieval_query_reference_references(view=retrieval_view, **retrieval_arguments,
            coverage_ids=retrieval_coverage, chunk_bytes=chunk_bytes)
        if not retrieval_projected or any(not reference.required for reference in retrieval_projected):
            raise codec.SemanticTranslationError("projected retrieval evidence must remain required")
        if not set(retrieval_coverage) <= {item for reference in retrieval_projected for item in reference.coverage_ids}:
            raise codec.SemanticTranslationError("retrieval projection changed required evidence coverage")
        restored = restore_retrieval_query_reference(view=retrieval_view, **retrieval_arguments)
        inverse_positions, inverse_references = validate_native_retrieval_references(
            wire=wire, binding=retrieval_query_binding, native_text=restored)
        if inverse_positions != retrieval_positions or codec._json([row.to_record() for row in inverse_references]) != codec._json(
                [row.to_record() for row in retrieval_references]):
            raise codec.SemanticTranslationError("retrieval projection changed original chunk records or order")
        retrieval_receipt = {
            "schema": "supervisor-code-retrieval-query-reference-receipt@1",
            "binding": retrieval_query_binding.to_dict(),
            "native_context_sha256": hashlib.sha256(retrieval_text.encode()).hexdigest(),
            "view_sha256": hashlib.sha256(_encoded(retrieval_view)).hexdigest(),
            "native_reference_records_sha256": hashlib.sha256(codec._json(
                [row.to_record() for row in retrieval_references]).encode()).hexdigest(),
            "native_reference_positions": list(retrieval_positions),
            "required_coverage_ids": list(retrieval_coverage),
            "target_semantic_packet_sha256": hashlib.sha256(codec._json(view).encode()).hexdigest(),
            "exact_native_context_and_reference_inverse": True,
            "source_freshness_verified": True, "historical_replay": False,
            "execution_authority": False, "completion_authority": False,
        }
    selected_positions = set(positions)
    selected_positions.update(retrieval_positions)
    evidence = []
    for index, reference in enumerate(wire["evidence"]):
        if index == positions[0]:
            evidence.extend(reference.to_record() for reference in projected_references)
        if retrieval_positions and index == retrieval_positions[0]:
            evidence.extend(reference.to_record() for reference in retrieval_projected)
        if index not in selected_positions:
            evidence.append(reference)
    if retrieval_query_binding is not None:
        identities = [row.get("reference_id") if type(row) is dict else None for row in evidence]
        if (any(type(identity) is not str or not identity.strip() for identity in identities)
                or len(identities) != len(set(identities))):
            raise codec.SemanticTranslationError("projected retrieval evidence reference identities are ambiguous")
    projected_prompt = codec._json({**wire, "evidence": evidence}) + suffix
    if len(projected_prompt.encode("utf-8")) > codec.MAX_BYTES:
        raise codec.SemanticTranslationError("projected provider payload exceeds translation bound")
    final, _ = codec._verified_semantic(
        root, artifact=artifact, text=text, task_id=wire["objective_id"], current=True,
    )
    if codec._json(final) != codec._json(semantic):
        raise codec.SemanticTranslationError("semantic parent changed during projection")
    if retrieval_query_binding is not None and load_live_retrieval_query_context(
            repository=root, binding=retrieval_query_binding, semantic_original_text=text,
            semantic_bindings=bindings) != retrieval_text:
        raise codec.SemanticTranslationError("retrieval parent changed during projection")
    receipt = {
        "schema": "supervisor-semantic-router-projection@1",
        "projection_revision": view["projection_revision"],
        "original_artifact": artifact,
        "original_artifact_sha256": bindings.expected_sha256,
        "native_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "projected_prompt_sha256": hashlib.sha256(projected_prompt.encode("utf-8")).hexdigest(),
        "native_prompt_bytes": len(prompt.encode("utf-8")),
        "projected_prompt_bytes": len(projected_prompt.encode("utf-8")),
        "required_coverage_ids": list(coverage),
        "deferred_reference_count": len(view["deferred_references"]),
        "task_core_and_literal_suffix_preserved": True,
        "original_artifact_unchanged": True,
        "source_freshness_checked": True,
        "provider_called": False,
        "execution_authority": False,
        "completion_authority": False,
        "total_session_token_savings": None,
    }
    if decision_policy is not None:
        receipt.update(
            selection_policy_sha256=hashlib.sha256(codec._json(decision_policy.to_dict()).encode()).hexdigest(),
            selection_summary=view["selection_summary"],
            provider_expansion_registered=False,
        )
    if retrieval_receipt is not None:
        receipt["retrieval_query_reference"] = retrieval_receipt
    return SemanticRouterProjection(projected_prompt, codec._json(receipt))


def expand_semantic_router_decision_packet(
    *, repository: Path, view: dict, reference_id: str,
    bindings: SemanticProjectionBindings, policy: SemanticDecisionPolicy,
    count_tokens, max_bytes: int = 16384, max_tokens: int = 4096, cancelled=None,
    capsule_ids: tuple[str, ...] | None = None,
):
    """Bind packet expansion to native producer and current source verification.

    A future provider bridge must retain bindings/policy independently and pass
    through this boundary. A bare artifact reader is insufficient.
    """
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root:
        raise codec.SemanticTranslationError("semantic repository must be canonical")

    def resolve_original(*, artifact, expected_sha256, task_id):
        raw = codec._read(root, artifact)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise codec.SemanticTranslationError("decision packet original artifact digest differs")
        text = raw.decode()
        codec._verified_semantic(root, artifact=artifact, text=text, task_id=task_id, current=True)
        return text

    return expand_semantic_decision_packet(
        view=view, reference_id=reference_id, bindings=bindings, policy=policy,
        resolve_original=resolve_original, count_tokens=count_tokens,
        max_bytes=max_bytes, max_tokens=max_tokens, cancelled=cancelled, capsule_ids=capsule_ids,
    )
