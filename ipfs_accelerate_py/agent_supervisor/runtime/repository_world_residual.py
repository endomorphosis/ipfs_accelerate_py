"""Focus optional repository context using an existing sealed residual request.

All original task invariants and required reference bodies remain in the issued
worker packet. Native lexical retrieval can choose different optional handles;
that selection establishes neither equivalence nor resolved work. This helper
does not dispatch, choose a cheaper representation, create a world authority,
or change the source, task, policy, compiler, or its existing resource admission.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json

from ..context.context_compiler import ContextCompileResult
from ..planning.residual_llm_packet import ResidualLlmPacket
from ..procedure_compiler.world_model import RepositoryWorldState
from .repository_context_compiler import PREFIX, RepositoryContextAdapter
from .repository_residual_context import prepare_repository_residual_context
from .repository_world_packet import (
    ARTIFACT_SCHEMA, PreparedRepositoryWorldPacket, _ISSUANCE_LOCK, _ISSUED,
    prepare_repository_world_packet,
)
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "repository-world-residual-focus@1"
MAX_QUERY_BYTES = 4096


class RepositoryWorldResidualError(ValueError):
    """The current baseline cannot support a bounded optional-context focus."""


def _need(value, reason):
    if not value:
        raise RepositoryWorldResidualError(reason)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _wire(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _custody(baseline):
    _need(type(baseline) is PreparedRepositoryWorldPacket, "exact_issued_baseline_required")
    refs = PreparedRepositoryWorldPacket._seal(baseline)
    with _ISSUANCE_LOCK:
        issued = _ISSUED.get(baseline)
    _need(issued is not None and len(issued) == len(refs)
        and all(first is second for first, second in zip(issued, refs)), "baseline_packet_custody_changed")
    _need(not set(baseline.__dict__) & {"receipt", "_seal", "validate_current", "require_provider_dispatch"},
        "baseline_packet_method_shadowed")
    PreparedRepositoryWorldPacket.require_provider_dispatch(baseline)


def _current(baseline, *, task_text):
    _custody(baseline)
    return PreparedRepositoryWorldPacket.validate_current(baseline,
        workspace=baseline._workspace, task_text=task_text)


def _references(capsule):
    selected = {reference.reference_id: reference for reference in capsule.evidence}
    _need(len(selected) == len(capsule.evidence), "duplicate_selected_context_reference")
    return selected


def _population(references):
    # Reference hashes/CIDs identify the exact population without exporting
    # paths, summaries, source text, questions, or arbitrary caller labels.
    return [dict(reference_id_sha256=_sha(identity), reference_content_id=reference.content_id)
        for identity, reference in sorted(references.items())]


@dataclass(frozen=True)
class PreparedRepositoryWorldResidualRequest:
    """Issued candidate plus a detached historical preparation observation.

    The candidate packet's existing live owner checks govern later dispatch.
    This wrapper is not a provider, source lease, proof, or freshness grant.
    """

    packet: PreparedRepositoryWorldPacket
    compilation: ContextCompileResult = field(repr=False)
    _receipt_json: str = field(repr=False)

    def receipt(self):
        return json.loads(self._receipt_json)


def prepare_repository_world_residual_request(*, baseline, residual_packet, query,
        unresolved_question, limit=8, max_bytes=32768):
    """Recompile advisory optional evidence while retaining full required input.

    ``baseline`` supplies the existing issued adapter, generation, task recipe,
    native intent observation, and every full required reference. The caller
    explicitly chooses the residual query/question; lexical selection is not a
    proof that omitted optional material is irrelevant. No automatic selection
    is made from byte counts or estimates, and no provider is invoked here.
    """
    _need(type(baseline) is PreparedRepositoryWorldPacket, "exact_issued_baseline_required")
    _need(type(residual_packet) is ResidualLlmPacket, "exact_residual_packet_required")
    _need(type(query) is str and query.strip() and len(query.encode()) <= MAX_QUERY_BYTES,
        "bounded_residual_query_required")
    _need(type(unresolved_question) is str and unresolved_question.strip()
        and len(unresolved_question.encode()) <= 1024, "bounded_residual_question_required")
    _need(type(limit) is int and 1 <= limit <= 32, "bounded_residual_query_limit_required")
    _need(type(max_bytes) is int and 1024 <= max_bytes <= 65536, "bounded_residual_query_bytes_required")
    _custody(baseline)
    # Inspect only an invocation-local issued record. Historical replay cannot
    # restore this baseline, even if all its serialized hashes still match.
    material = json.loads(baseline._replay_material_json)
    _need(material.get("schema") == ARTIFACT_SCHEMA and type(material.get("recipe")) is dict,
        "baseline_packet_recipe_unavailable")
    recipe = material["recipe"]
    _need("residual_context" not in recipe and "residual_packet" not in recipe,
        "already_residual_baseline_refused")
    task_text = recipe["task_text"]
    with public_replay_without_metadata():
        before = _current(baseline, task_text=task_text)
        adapter, capsule = baseline._adapter, baseline._capsule
        _need(type(adapter) is RepositoryContextAdapter, "exact_baseline_context_adapter_required")
        # Reject a foreign residual before any new optional-context query.
        residual_context = prepare_repository_residual_context(residual_packet=residual_packet,
            expected_task_id=recipe["task_id"], repository_id=capsule.repository_id, tree_id=capsule.tree_id,
            policy_id=capsule.policy_id, policy_revision=capsule.policy_revision,
            write_paths=tuple(recipe["write_paths"]), validation_commands=tuple(recipe["validation_commands"]),
            acceptance_ids=tuple(recipe["acceptance_ids"]))
        # The original packet can contain nested mutable mappings. Keep only
        # the independently validated detached snapshot across compilation.
        residual_snapshot = ResidualLlmPacket.from_dict(residual_context.packet)
        selected = _references(capsule)
        required = {identity: reference for identity, reference in selected.items() if reference.required}
        # Reuse available caller-authored optional evidence as optional input.
        # Native handles are queried again through the original owner instead
        # of being smuggled in as caller evidence under their reserved prefix.
        optional = {reference.reference_id: reference for reference in
            (*capsule.evidence, *capsule.expansion_references)
            if not reference.required and not reference.reference_id.startswith(PREFIX)}
        authority = _plain(capsule.authority)
        compiled = adapter.compile(generation=baseline._generation, query=query,
            unresolved_question=unresolved_question, required_evidence=tuple(required.values()),
            required_reference_ids=tuple(required),
            required_coverage_ids=tuple(sorted({coverage for reference in required.values()
                for coverage in reference.coverage_ids})),
            objective_id=capsule.objective_id, objective_revision=capsule.objective_revision,
            policy_id=capsule.policy_id, policy_revision=capsule.policy_revision,
            goal=_plain(capsule.goal), authority=authority["task"],
            scope=_plain(capsule.scope), acceptance=_plain(capsule.acceptance),
            optional_evidence=tuple(optional.values()), limit=limit, max_bytes=max_bytes)
        after_capsule = compiled.capsule
        after_selected = _references(after_capsule)
        after_required = {identity: reference for identity, reference in after_selected.items() if reference.required}
        _need(set(after_required) == set(required) and all(
            after_required[identity].canonical_bytes() == reference.canonical_bytes()
            for identity, reference in required.items()), "residual_required_reference_changed")
        _need(all(_wire(getattr(after_capsule, name)) == _wire(getattr(capsule, name))
            for name in ("repository_id", "tree_id", "objective_id", "objective_revision",
                         "policy_id", "policy_revision", "goal", "scope", "acceptance"))
            and _wire(after_capsule.authority["task"]) == _wire(authority["task"]),
            "residual_task_invariants_changed")
        arguments = dict(recipe)
        arguments["current_world"] = (None if recipe["current_world"] is None
            else RepositoryWorldState.from_dict(recipe["current_world"]))
        for name in ("unresolved_obligations", "validation_commands", "proof_obligations",
                     "acceptance_ids", "allowed_effects", "write_paths"):
            arguments[name] = tuple(arguments[name])
        candidate = prepare_repository_world_packet(adapter=adapter, compilation=compiled,
            intent_world_observation=baseline._intent_world_observation,
            residual_packet=residual_snapshot, **arguments)
        old_optional = {identity: reference for identity, reference in selected.items() if not reference.required}
        new_optional = {identity: reference for identity, reference in after_selected.items() if not reference.required}
        omitted = {identity: reference for identity, reference in old_optional.items() if identity not in new_optional}
        native_omitted = {identity: reference for identity, reference in omitted.items() if identity.startswith(PREFIX)}
        added = {identity: reference for identity, reference in new_optional.items() if identity not in old_optional}
        changed = {identity: reference for identity, reference in new_optional.items()
            if identity in old_optional and reference.canonical_bytes() != old_optional[identity].canonical_bytes()}
        query_population = after_capsule.authority["repository_context"]
        receipt = dict(schema=SCHEMA, baseline_prompt_sha256=before["prompt_sha256"],
            baseline_prompt_bytes=before["prompt_bytes"], candidate_prompt_sha256=_sha(candidate.prompt),
            candidate_prompt_bytes=len(candidate.prompt.encode()),
            baseline_context_cid=capsule.capsule_id, candidate_context_cid=after_capsule.capsule_id,
            residual_packet_id=residual_snapshot.packet_id, generation=baseline._generation,
            source_binding=_plain(before["source_binding"]),
            task_text_sha256=before["task_text_sha256"], query_sha256=_sha(query),
            unresolved_question_sha256=_sha(unresolved_question), query_limit=limit, query_max_bytes=max_bytes,
            required_reference_count=len(required), required_references_preserved=True,
            required_population_sha256=_sha(_wire(_population(required))),
            full_task_invariants_preserved=True, required_omission_authority=False,
            baseline_optional_selected_count=len(old_optional), candidate_optional_selected_count=len(new_optional),
            omitted_baseline_optional_count=len(omitted), omitted_baseline_native_optional_count=len(native_omitted),
            added_optional_count=len(added), rebound_optional_count=len(changed),
            baseline_optional_selected=_population(old_optional), candidate_optional_selected=_population(new_optional),
            omitted_baseline_optional=_population(omitted), added_optional=_population(added),
            rebound_optional=_population(changed), compiler_omitted_reference_count=len(after_capsule.omitted_reference_ids),
            query_population_count=query_population["population_count"], query_matching_count=query_population["matching_count"],
            query_selected_count=query_population["selected_count"], query_omitted_count=query_population["omitted_count"],
            query_truncated=query_population["truncated"], unqueried_native_reference_ids_exported=False,
            optional_selection_basis="existing_source_query_and_context_compiler_advisory_policy",
            optional_omission_equivalence_verified=False, resolved_work_claimed=False,
            world_projection_status=before["world_projection_status"],
            independent_world_owner_freshness="not_revalidated", world_authority_created=False,
            full_repository_coverage_verified=False, provider_retained_parent=False,
            provider_calls=0, provider_input_tokens=None, token_savings_established=False,
            default_adoption_authorized=False, proof_authority=False,
            execution_authority=False, completion_authority=False)
        result = PreparedRepositoryWorldResidualRequest(candidate, compiled, _wire(receipt))
        # All rendering/serialization precedes these last native fences. They
        # remain cooperative observations rather than a lock across execution.
        _current(baseline, task_text=task_text)
        _current(candidate, task_text=task_text)
        return result
