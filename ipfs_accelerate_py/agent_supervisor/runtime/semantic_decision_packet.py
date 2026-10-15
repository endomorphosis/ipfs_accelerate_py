"""Explicit selection over a caller-authenticated semantic worker artifact.

The trusted caller declares a complete required/optional capsule partition.
Only unselected optional bodies and estimator telemetry leave the decision
packet. The opt-in @2 view also references two exact required text duplicates
inside the packet; their literal targets remain immediately available.
This API neither infers optionality nor registers a provider expansion tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..context.context_compiler import build_text_context_references
from ..context.context_contracts import ContextReference, ContextTier
from .semantic_context_projection import (
    MAX_ARTIFACT_BYTES, MAX_EXPANSION_BYTES, SELECTORS,
    SemanticProjectionBindings, _bound, _encoded, _original, _sha, _structure,
    _telemetry, _text,
)

SCHEMA = "supervisor-semantic-decision-packet@1"
PROJECTION_REVISION = "explicit-capsule-selection@1"
ARCHIVE_SCHEMA = "supervisor-semantic-decision-archive@1"
TEXT_PROJECTION_REVISION = "explicit-capsule-selection+exact-task-text-references@2"
POLICY_LIST_FIELDS = frozenset({
    "required_capsule_ids", "optional_capsule_ids", "selected_optional_capsule_ids",
})


@dataclass(frozen=True)
class SemanticDecisionPolicy:
    """Trusted caller nomination using native capsule_cid identities.

    Required/optional classifications are assertions by this caller, not proof
    derived from retrieval scores, capsule confidence or raw-source flags.
    Verification and expansion require this independently held policy again.
    """

    required_capsule_ids: tuple[str, ...]
    optional_capsule_ids: tuple[str, ...]
    selected_optional_capsule_ids: tuple[str, ...] = ()
    task_text_source_path: str | None = None

    def __post_init__(self):
        for name in POLICY_LIST_FIELDS:
            values = getattr(self, name)
            if type(values) is not tuple or len(values) > 1024:
                raise ValueError("bounded explicit capsule identity tuples required")
            for value in values:
                _text(value, "capsule identity")
            if len(set(values)) != len(values):
                raise ValueError("duplicate policy capsule identity")
        if set(self.required_capsule_ids) & set(self.optional_capsule_ids):
            raise ValueError("required and optional capsule identities overlap")
        if not set(self.selected_optional_capsule_ids) <= set(self.optional_capsule_ids):
            raise ValueError("selected optional capsule identity is outside policy")
        if self.task_text_source_path is not None:
            value = self.task_text_source_path
            if (type(value) is not str or not value.strip() or not 1 <= len(value.encode()) <= 4096
                    or any(char in value for char in ("\x00", "\n", "\r"))
                    or PurePosixPath(value).is_absolute() or str(PurePosixPath(value)) != value
                    or any(part in {"", ".", ".."} for part in value.split("/"))):
                raise ValueError("canonical relative task text source path required")

    @property
    def schema_revision(self) -> str:
        return "1" if self.task_text_source_path is None else "2"

    def to_dict(self) -> dict:
        value = {name: list(getattr(self, name)) for name in (
            "required_capsule_ids", "optional_capsule_ids", "selected_optional_capsule_ids",
        )}
        if self.task_text_source_path is not None:
            value["task_text_source_path"] = self.task_text_source_path
        return value

    @classmethod
    def from_dict(cls, value: dict) -> SemanticDecisionPolicy:
        if type(value) is not dict or set(value) not in (
                POLICY_LIST_FIELDS, POLICY_LIST_FIELDS | {"task_text_source_path"}):
            raise ValueError("exact semantic decision selection policy fields required")
        if any(type(value[name]) is not list for name in POLICY_LIST_FIELDS):
            raise ValueError("semantic decision selection policy lists required")
        if "task_text_source_path" in value and type(value["task_text_source_path"]) is not str:
            raise ValueError("explicit task text source path required")
        return cls(**{name: tuple(value[name]) for name in POLICY_LIST_FIELDS},
                   task_text_source_path=value.get("task_text_source_path"))


def _task_text_references(required, policy):
    """Reference only exact duplicates, preserving two distinct literal roots."""
    path = policy.task_text_source_path
    if path is None:
        return []
    instruction = required["raw_sources"].get(path)
    manifest = required["manifest"].get(path)
    objective = required.get("objective")
    projection = required.get("worker_projection")
    packed = required["pack"]["pack"]
    if (path not in required["required_raw_paths"] or type(instruction) is not str
            or not instruction.strip() or type(manifest) is not dict
            or manifest.get("sha256") != _sha(instruction.encode())
            or type(objective) is not str or not objective.strip()
            or type(packed.get("objective")) is not str or packed["objective"] != objective
            or type(projection) is not dict or type(projection.get("query")) is not str
            or projection["query"] != instruction):
        raise ValueError("task text projection requires exact duplicate strings and bound required literal source")
    references = [
        {"field_selector": ["worker_projection", "query"],
         "target_selector": ["raw_sources", path], "sha256": _sha(instruction.encode())},
        {"field_selector": ["pack", "pack", "objective"],
         "target_selector": ["objective"], "sha256": _sha(objective.encode())},
    ]
    del projection["query"]
    del packed["objective"]
    return references


def _material(original_text, bindings, policy, max_original_bytes):
    if type(policy) is not SemanticDecisionPolicy:
        raise ValueError("independent trusted decision policy required")
    required = _original(original_text, bindings, max_original_bytes)
    capsules, admissions = required["capsules"], required["admissions"]
    if len(capsules) != len(admissions) or len(capsules) > 1024:
        raise ValueError("one matching admission per native capsule required")
    identities, symbols = [], []
    for capsule, admission in zip(capsules, admissions, strict=True):
        if (type(capsule) is not dict or type(admission) is not dict
                or type(admission.get("ref")) is not dict):
            raise ValueError("typed capsule and matching admission required")
        identity, symbol = capsule.get("capsule_cid"), capsule.get("stable_symbol_id")
        _text(identity, "native capsule identity")
        _text(symbol, "native capsule symbol")
        if (admission["ref"].get("capsule_cid") != identity
                or admission["ref"].get("stable_symbol_id") != symbol):
            raise ValueError("native capsule and admission identities differ")
        identities.append(identity)
        symbols.append(symbol)
    if len(set(identities)) != len(identities) or len(set(symbols)) != len(symbols):
        raise ValueError("duplicate native capsule identity or symbol")
    if set(identities) != set(policy.required_capsule_ids) | set(policy.optional_capsule_ids):
        raise ValueError("policy must completely partition the native capsule inventory")

    archive = []
    for selector in SELECTORS:
        value = _telemetry(required, selector)
        if value is not None:
            archive.append({"field_selector": list(selector), "value": value})
            node = required
            for field in selector[:-1]:
                node = node[field]
            del node[selector[-1]]
    retained = set(policy.required_capsule_ids) | set(policy.selected_optional_capsule_ids)
    selected = []
    for index, capsule in enumerate(capsules):
        if capsule["capsule_cid"] in retained:
            selected.append(capsule)
        else:
            archive.append({"field_selector": ["capsules", index], "value": capsule})
    required["capsules"] = selected
    return required, archive, len(capsules)


def _expanded(handle, body):
    return ContextReference(
        reference_id=handle.reference_id, kind=handle.kind, tier=ContextTier.EVIDENCE,
        referenced_content_id=handle.referenced_content_id,
        repository_id=handle.repository_id, tree_id=handle.tree_id, path=handle.path,
        summary=body.decode(), byte_count=len(body), token_count=0,
        metadata=dict(handle.metadata),
    )


def build_semantic_decision_packet(
    *, original_text: str, bindings: SemanticProjectionBindings,
    policy: SemanticDecisionPolicy, max_original_bytes: int = MAX_ARTIFACT_BYTES,
    max_view_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """Select bodies only after full producer/source verification by the caller."""
    _bound(max_original_bytes, MAX_ARTIFACT_BYTES, "original byte budget")
    _bound(max_view_bytes, MAX_ARTIFACT_BYTES, "view byte budget")
    required, archive, count = _material(original_text, bindings, policy, max_original_bytes)
    canonical_coverage = _sha(_encoded(required))
    text_references = _task_text_references(required, policy)
    coverage = (_sha(_encoded({"required_context": required, "required_text_references": text_references}))
                if text_references else canonical_coverage)
    revision = TEXT_PROJECTION_REVISION if text_references else PROJECTION_REVISION
    deferred = []
    if archive:
        body = _encoded({"schema": ARCHIVE_SCHEMA, "fields": archive})
        digest = _sha(body)
        handle = ContextReference(
            reference_id="semantic-decision-archive:" + digest,
            kind="semantic-decision-archive", tier=ContextTier.EXPANSION,
            referenced_content_id="sha256:" + digest,
            repository_id=bindings.repository_id, tree_id=bindings.tree_id,
            path=bindings.artifact,
            summary=("Explicitly optional capsule bodies and estimator telemetry. "
                     "Request capsule_ids for a subset; otherwise the whole archive expands. "
                     "Deferred facts are unavailable until expanded. Read exact current "
                     "repository source before editing; capsules do not replace source."),
            byte_count=len(body), token_count=0,
            metadata={
                "required": False, "selector_policy": revision,
                "parent_sha256": bindings.expected_sha256, "task_id": bindings.task_id,
                "semantic_root_cid": bindings.semantic_root_cid, "scope_cid": bindings.scope_cid,
                "policy_sha256": _sha(_encoded(policy.to_dict())),
                "archived_sha256": digest, "required_coverage_sha256": coverage,
                "coverage_ids": [],
            },
        )
        # Never emit a handle which cannot be returned under the hard bound.
        if len(_encoded(_expanded(handle, body).to_dict())) > MAX_EXPANSION_BYTES:
            raise ValueError("complete decision archive expansion exceeds byte bound")
        deferred.append(json.loads(_encoded(handle.to_dict())))
    view = {
        "schema": "supervisor-semantic-decision-packet@" + policy.schema_revision,
        "projection_revision": revision,
        "origin": bindings.to_dict(), "selection_policy": policy.to_dict(),
        "selection_summary": {
            "parent_capsule_count": count,
            "retained_capsule_count": len(required["capsules"]),
            "deferred_capsule_count": count - len(required["capsules"]),
            "parent_worker_projection_preserved": not bool(text_references),
        },
        "required_context": required, "required_coverage_sha256": coverage,
        "deferred_references": deferred,
        "execution_authority": False, "completion_authority": False,
        "semantic_equivalence_claimed": False,
    }
    if text_references:
        view.update(
            required_text_references=text_references,
            canonical_required_coverage_sha256=canonical_coverage,
            required_text_reference_rules=(
                "Each field_selector remains required. Its exact string is already present at "
                "target_selector within required_context; use that literal without expansion. "
                "The public instruction and objective are distinct literal targets."
            ),
        )
        view["selection_summary"]["parent_worker_projection_reconstructable"] = True
    if len(_encoded(view)) > max_view_bytes:
        raise ValueError("decision packet exceeds byte bound; required material cannot be truncated")
    return view


def verify_semantic_decision_packet(
    *, view: dict, original_text: str, bindings: SemanticProjectionBindings,
    policy: SemanticDecisionPolicy, max_original_bytes: int = MAX_ARTIFACT_BYTES,
    max_view_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict:
    """Regenerate against an independent expected policy and verified parent."""
    _bound(max_original_bytes, MAX_ARTIFACT_BYTES, "original byte budget")
    _bound(max_view_bytes, MAX_ARTIFACT_BYTES, "view byte budget")
    _structure(view, max_view_bytes)
    if len(_encoded(view)) > max_view_bytes:
        raise ValueError("decision packet exceeds byte bound")
    expected = build_semantic_decision_packet(
        original_text=original_text, bindings=bindings, policy=policy,
        max_original_bytes=max_original_bytes, max_view_bytes=max_view_bytes,
    )
    if _encoded(view) != _encoded(expected):
        raise ValueError("decision packet differs from required coverage or trusted policy")
    return expected


def restore_semantic_decision_required_context(
    *, view: dict, original_text: str, bindings: SemanticProjectionBindings,
    policy: SemanticDecisionPolicy,
) -> dict:
    """Restore selected required fields only, after independent parent verification.

    Optional bodies and estimator telemetry remain governed by expansion. This
    does not reconstruct a repository, or grant source or task authority.
    """
    verified = verify_semantic_decision_packet(
        view=view, original_text=original_text, bindings=bindings, policy=policy,
    )
    required = json.loads(_encoded(verified["required_context"]))
    for reference in verified.get("required_text_references", []):
        target = required
        for field in reference["target_selector"]:
            target = target[field]
        if type(target) is not str or _sha(target.encode()) != reference["sha256"]:
            raise ValueError("required literal text target differs")
        node = required
        for field in reference["field_selector"][:-1]:
            node = node[field]
        node[reference["field_selector"][-1]] = target
    expected, _, _ = _material(original_text, bindings, policy, MAX_ARTIFACT_BYTES)
    if _encoded(required) != _encoded(expected):
        raise ValueError("required text restoration differs from selected native context")
    return required


def semantic_decision_packet_references(
    *, view: dict, original_text: str, bindings: SemanticProjectionBindings,
    policy: SemanticDecisionPolicy, coverage_ids: Sequence[str] = (), chunk_bytes: int = 6144,
) -> tuple[ContextReference, ...]:
    _bound(chunk_bytes, 16384, "decision packet chunk byte budget")
    verified = verify_semantic_decision_packet(
        view=view, original_text=original_text, bindings=bindings, policy=policy,
    )
    if type(coverage_ids) not in {list, tuple} or any(type(item) is not str for item in coverage_ids):
        raise ValueError("explicit coverage identifier sequence required")
    return build_text_context_references(
        _encoded(verified).decode(), reference_prefix="semantic-decision-packet",
        kind="semantic-context", path=bindings.artifact,
        repository_id=bindings.repository_id, tree_id=bindings.tree_id,
        required=True, chunk_bytes=chunk_bytes,
        coverage_ids=(*coverage_ids, "semantic-decision-required:" + verified["required_coverage_sha256"]),
    )


def expand_semantic_decision_packet(
    *, view: dict, reference_id: str, bindings: SemanticProjectionBindings,
    policy: SemanticDecisionPolicy, resolve_original: Callable[..., str],
    count_tokens: Callable[[str], int], max_bytes: int = 16384, max_tokens: int = 4096,
    cancelled: Callable[[], bool] | None = None,
    capsule_ids: tuple[str, ...] | None = None,
) -> ContextReference:
    """Return bounded exact archive bytes, with fresh verification at both ends.

    The resolver must authenticate the native producer and current source, not
    merely read the retained file. The counter sees the complete returned
    descriptor with token_count=0. This is not provider/session usage telemetry.
    A nonempty capsule_ids tuple requests only those deferred bodies, excluding
    telemetry. None returns the complete archive. No selection changes policy.
    """
    _bound(max_bytes, MAX_EXPANSION_BYTES, "expansion byte budget")
    _bound(max_tokens, 128000, "expansion token budget")
    _text(reference_id, "expansion reference identity")
    if (type(bindings) is not SemanticProjectionBindings or type(policy) is not SemanticDecisionPolicy
            or not callable(resolve_original) or not callable(count_tokens)
            or (cancelled is not None and not callable(cancelled))):
        raise ValueError("trusted bindings, policy, current resolver and counter required")
    if capsule_ids is not None:
        if type(capsule_ids) is not tuple or not 1 <= len(capsule_ids) <= 1024:
            raise ValueError("nonempty bounded capsule expansion tuple required")
        for identity in capsule_ids:
            _text(identity, "requested capsule identity")
        deferred_ids = set(policy.optional_capsule_ids) - set(policy.selected_optional_capsule_ids)
        if len(set(capsule_ids)) != len(capsule_ids) or not set(capsule_ids) <= deferred_ids:
            raise ValueError("requested capsule identities must be unique deferred optional bodies")

    def check_cancelled():
        if cancelled is not None:
            state = cancelled()
            if type(state) is not bool or state:
                raise ValueError("decision packet expansion cancelled or unavailable")

    check_cancelled()
    _structure(view, MAX_ARTIFACT_BYTES)
    if len(_encoded(view)) > MAX_ARTIFACT_BYTES:
        raise ValueError("decision packet exceeds byte bound")
    if (type(view) is not dict or view.get("selection_policy") != policy.to_dict()
            or view.get("origin") != bindings.to_dict()):
        raise ValueError("decision packet trusted policy or parent binding differs")
    request = {"artifact": bindings.artifact, "expected_sha256": bindings.expected_sha256,
               "task_id": bindings.task_id}
    original_text = resolve_original(**request)
    verified = verify_semantic_decision_packet(
        view=view, original_text=original_text, bindings=bindings, policy=policy,
    )
    handles = verified["deferred_references"]
    if len(handles) != 1 or handles[0]["reference_id"] != reference_id:
        raise ValueError("decision archive expansion handle is absent or differs")
    handle = ContextReference.from_dict(handles[0])
    if capsule_ids is None and handle.byte_count > max_bytes:
        raise ValueError("decision archive exceeds byte bound before materialization")
    _, archive, _ = _material(original_text, bindings, policy, MAX_ARTIFACT_BYTES)
    body = _encoded({"schema": ARCHIVE_SCHEMA, "fields": archive})
    if len(body) != handle.byte_count or _sha(body) != handle.metadata["archived_sha256"]:
        raise ValueError("decision archive size or digest differs")
    if capsule_ids is None:
        expanded = _expanded(handle, body)
    else:
        requested = set(capsule_ids)
        selected = [row for row in archive if row["field_selector"][0] == "capsules"
                    and row["value"]["capsule_cid"] in requested]
        if len(selected) != len(capsule_ids):
            raise ValueError("requested capsule bodies differ from verified archive")
        body = _encoded({"schema": ARCHIVE_SCHEMA, "fields": selected})
        digest = _sha(body)
        expanded = ContextReference(
            reference_id=handle.reference_id + ":capsule-subset:" + digest,
            kind=handle.kind, tier=ContextTier.EVIDENCE,
            referenced_content_id="sha256:" + digest,
            repository_id=handle.repository_id, tree_id=handle.tree_id, path=handle.path,
            summary=body.decode(), byte_count=len(body), token_count=0,
            metadata={**handle.metadata, "expanded_sha256": digest, "complete_archive": False,
                      "expanded_capsule_ids": [row["value"]["capsule_cid"] for row in selected]},
        )
    wire = _encoded(expanded.to_dict()).decode()
    if len(wire.encode()) > max_bytes:
        raise ValueError("complete expanded decision reference exceeds byte bound")
    tokens = count_tokens(wire)
    if type(tokens) is not int or not 1 <= tokens <= max_tokens:
        raise ValueError("complete expanded decision reference exceeds typed token bound")
    check_cancelled()
    closing_text = resolve_original(**request)
    verify_semantic_decision_packet(
        view=view, original_text=closing_text, bindings=bindings, policy=policy,
    )
    if closing_text != original_text:
        raise ValueError("decision parent changed during expansion")
    check_cancelled()
    return expanded
