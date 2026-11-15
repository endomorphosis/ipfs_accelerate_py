"""Preserve a pinned initial implementation context in a fresh source packet.

The parent and its independently supplied pins are caller custody inputs, not
proof of an owner, signature, task claim, or worktree allocation. This helper
checks exact context preservation and current local source ownership only. It
does not deserialize live packet authority, offer a worker CLI, or establish
current intent ownership. Authenticated dispatch remains the caller's job.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
from pathlib import PurePosixPath

from ..context.context_compiler import render_context_capsule
from ..context.context_contracts import ContextCapsule
from .repository_context_compiler import RepositoryContextAdapter, MAX_REFERENCES, PREFIX
from .repository_world_packet import (
    MAX_BYTES, PreparedRepositoryWorldPacket, prepare_repository_world_packet,
    repository_world_task_binding,
)
from .supervisor_meta_index import public_replay_without_metadata


SCHEMA = "repository-world-parent-binding@1"
PROVENANCE_KEY = "world_parent_context"
EFFECTS = ("read_source", "write_source", "run_validation")
_RESERVED_KINDS = {"source-formalization-context", "repository-intent-world"}


class RepositoryWorldParentError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _need(value, code):
    if not value:
        raise RepositoryWorldParentError(code)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value, name, maximum=4096):
    _need(type(value) is str and value and value.strip() == value and "\0" not in value
          and len(value.encode("utf-8")) <= maximum, "invalid_parent_" + name)
    return value


def _texts(value, name, *, required=False):
    _need(type(value) is tuple and len(value) <= 64 and (value or not required),
          "invalid_parent_" + name)
    for item in value:
        _text(item, name)
    _need(len(value) == len(set(value)), "duplicate_parent_" + name)
    return value


def _paths(value, name, *, required=False, writable=False):
    paths = _texts(value, name, required=required)
    for value in paths:
        path = PurePosixPath(value)
        _need(not path.is_absolute() and path.as_posix() == value
              and all(part not in {"", ".", ".."} for part in value.split("/"))
              and not any(char in value for char in "\\:*?[]\r\n")
              and (not writable or ".git" not in path.parts),
              "invalid_parent_" + name)
    return paths


def _overlap(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _budget(parent, max_bytes):
    declared = parent.authority.get("implementation_context_budget")
    fields = {"source", "max_provider_input_bytes", "task_max_input_tokens",
              "provider_context_window", "supervisor_max_input_tokens",
              "reserved_output_tokens", "reserved_tool_tokens"}
    _need(isinstance(declared, Mapping) and set(declared) == fields
          and declared["source"] in {"task_metadata", "supervisor_default"},
          "invalid_parent_budget")
    for key in fields - {"source"}:
        value = declared[key]
        nullable = key in {"max_provider_input_bytes", "task_max_input_tokens", "provider_context_window"}
        minimum = 0 if key.startswith("reserved_") else 1
        _need(value is None and nullable or type(value) is int and value >= minimum,
              "invalid_parent_budget")
    # The capsule already carries its compiler's negotiated input ceiling.
    # Reserves apply to a total provider window, not to that input ceiling.
    tokens = min(parent.budget.max_input_tokens, declared["supervisor_max_input_tokens"])
    if declared["task_max_input_tokens"] is not None:
        tokens = min(tokens, declared["task_max_input_tokens"])
    if declared["provider_context_window"] is not None:
        tokens = min(tokens, declared["provider_context_window"]
            - declared["reserved_output_tokens"] - declared["reserved_tool_tokens"])
    byte_limit = min(max_bytes, parent.budget.max_serialized_bytes)
    if declared["max_provider_input_bytes"] is not None:
        byte_limit = min(byte_limit, declared["max_provider_input_bytes"])
    _need(tokens > 0 and byte_limit >= 1024, "parent_budget_exhausted")
    return tokens, byte_limit


def compile_implementation_world_packet(
    *, adapter: RepositoryContextAdapter, parent: ContextCapsule,
    rendered_prompt: str, expected_parent_cid: str, task_text: str,
    expected_task_text_sha256: str, query: str, unresolved_question: str,
    max_bytes: int = MAX_BYTES,
) -> PreparedRepositoryWorldPacket:
    """Compile an exact pinned initial parent through a current source session.

    All selected parent evidence must survive, even optional evidence. Original
    reference identities must explicitly name this parent repository and tree;
    blank or partial bindings are unsupported in this first profile. Only those
    outer identities are rebound to the worker's fresh source session. Full goal,
    scope, acceptance, and existing authority values remain unchanged. Budget
    checks use this adapter's declared counter; the parent's tokenizer is not
    reconstructed. The runner owns final advisory, schema, and provider-token
    accounting, and the parent's timeout policy is preserved without executing it.
    """
    _need(type(adapter) is RepositoryContextAdapter and type(parent) is ContextCapsule,
          "invalid_parent_owners")
    _need(type(max_bytes) is int and 1024 <= max_bytes <= MAX_BYTES,
          "invalid_parent_packet_limit")
    _text(query, "query")
    _text(unresolved_question, "question", maximum=1024)
    with public_replay_without_metadata():
        _need(len(parent.canonical_bytes()) <= MAX_BYTES, "parent_context_byte_limit")
        # Revalidate nested contracts before accepting even an in-memory object.
        parent = ContextCapsule.from_dict(parent.to_record())
        _need(type(expected_parent_cid) is str and parent.capsule_id == expected_parent_cid,
              "parent_context_pin_mismatch")
        _need(type(rendered_prompt) is str and len(rendered_prompt.encode("utf-8")) <= MAX_BYTES
              and rendered_prompt == render_context_capsule(parent), "parent_prompt_mismatch")
        _need(type(task_text) is str and len(task_text.encode("utf-8")) <= max_bytes
              and type(expected_task_text_sha256) is str
              and _sha(task_text) == expected_task_text_sha256, "parent_task_text_pin_mismatch")
        _need(not parent.parent_capsule_id and not parent.expansion_references
              and not parent.omissions and parent.truncated is False,
              "unsupported_parent_retry_or_omission")
        _need(parent.caller == "agent-supervisor:implementation-daemon"
              and parent.stage == "implementation"
              and parent.policy_id == "policy:implementation-daemon",
              "unsupported_parent_origin")
        _need(all(isinstance(value, Mapping) for value in (
            parent.goal, parent.authority, parent.scope, parent.acceptance)), "invalid_parent_core")
        _need(parent.goal.get("task_id") == parent.objective_id
              and type(parent.goal.get("attempt")) is int and parent.goal["attempt"] == 1,
              "unsupported_parent_task_or_attempt")
        _need("world_task_packet" not in parent.authority and PROVENANCE_KEY not in parent.authority,
              "parent_binding_collision")
        edit = parent.authority.get("edit_policy")
        _need(isinstance(edit, Mapping), "invalid_parent_edit_policy")
        paths = _paths(parent.scope.get("allowed_edit_paths"), "write_paths", required=True, writable=True)
        _need(paths == _paths(edit.get("allowed_paths"), "edit_paths", required=True, writable=True),
              "parent_write_scope_mismatch")
        protected = _paths(parent.scope.get("protected_edit_paths"), "protected_paths")
        _need(protected == _paths(edit.get("protected_paths"), "edit_protected_paths"),
              "parent_protected_scope_mismatch")
        _need(not any(_overlap(path, forbidden) for path in paths for forbidden in protected),
              "parent_protected_scope_overlap")
        criteria = _text(parent.acceptance.get("criteria"), "criteria")
        _need(criteria != "none listed", "missing_parent_criteria")
        validations = _texts(parent.acceptance.get("validation_commands"), "validations", required=True)
        _need(parent.acceptance.get("all_expected_outputs_required") is True,
              "invalid_parent_output_acceptance")
        token_limit, byte_limit = _budget(parent, max_bytes)
        task_binding = repository_world_task_binding(parent.objective_id, task_text, EFFECTS, paths)
        _need(len(task_text.encode("utf-8")) <= byte_limit, "parent_task_text_byte_limit")
        _need(len(parent.evidence) <= MAX_REFERENCES, "parent_reference_limit")
        _need(all(reference.kind not in _RESERVED_KINDS
                  and not reference.reference_id.startswith(PREFIX)
                  for reference in parent.evidence), "unsupported_parent_evidence")
        _need(all(reference.repository_id == parent.repository_id
                  and reference.tree_id == parent.tree_id for reference in parent.evidence),
              "parent_reference_binding_mismatch")

        status = adapter.session.status_summary()
        _need(status["source_current"] is True, "parent_source_not_current")
        generation = status["generation"]
        repository_id = status["datasets_head"]["repository_id"]
        tree_id = "sha256:" + status["source_tree_sha256"]
        selected = tuple(replace(reference, repository_id=repository_id, tree_id=tree_id)
                         for reference in parent.evidence)
        required = tuple(reference for reference in selected if reference.required)
        optional = tuple(reference for reference in selected if not reference.required)
        provenance = dict(schema=SCHEMA, parent_capsule_id=parent.capsule_id,
            parent_prompt_sha256=_sha(rendered_prompt), parent_prompt_bytes=len(rendered_prompt.encode("utf-8")),
            parent_repository_id=parent.repository_id, parent_tree_id=parent.tree_id,
            parent_caller=parent.caller, parent_stage=parent.stage,
            parent_budget_cid=parent.budget.content_id, input_token_limit=token_limit,
            packet_byte_limit=byte_limit, token_estimator=adapter.compiler.estimator.name,
            parent_tokenizer_reconstructed=False, final_provider_input_measured=False,
            owner_authority_verified=False, allocation_verified=False, intent_freshness_verified=False,
            parent_evidence_currentness_verified=False)
        authority = dict(_plain(parent.authority), world_task_packet=task_binding,
                         **{PROVENANCE_KEY: provenance})
        compiled = adapter.compile(generation=generation, query=query,
            unresolved_question=unresolved_question, required_evidence=required,
            required_reference_ids=tuple(reference.reference_id for reference in required),
            required_coverage_ids=tuple(sorted({key for reference in required for key in reference.coverage_ids})),
            optional_evidence=optional, objective_id=parent.objective_id,
            objective_revision=parent.objective_revision, policy_id=parent.policy_id,
            policy_revision=parent.policy_revision, goal=_plain(parent.goal), authority=authority,
            scope=_plain(parent.scope), acceptance=_plain(parent.acceptance))
        retained = {reference.reference_id: reference for reference in compiled.capsule.evidence}
        _need(all(retained.get(reference.reference_id) == reference for reference in selected),
              "parent_selected_evidence_changed")
        _need(all(_plain(getattr(compiled.capsule, name)) == _plain(getattr(parent, name))
                  for name in ("goal", "scope", "acceptance"))
              and _plain(compiled.capsule.authority["task"]) == authority,
              "parent_core_changed")
        _need(compiled.capsule.input_tokens <= token_limit
              and len(render_context_capsule(compiled.capsule).encode("utf-8")) <= byte_limit,
              "parent_compiled_budget_exceeded")
        packet = prepare_repository_world_packet(adapter=adapter, compilation=compiled,
            generation=generation, task_id=parent.objective_id, task_text=task_text,
            unresolved_obligations=(criteria,), validation_commands=validations,
            allowed_effects=EFFECTS, write_paths=paths, max_bytes=byte_limit)
        _need(adapter.compiler.estimator.count(packet.prompt) <= token_limit,
              "parent_packet_token_limit")
        return packet
