"""Obligation-first context capsules for codebase-proof work (CBP-060).

Builds :class:`ContextCapsule` inputs from :class:`CodeProofQuery` so
implementation agents receive:

* invariant core — task/acceptance ids, open obligations, assumptions,
  counterexamples, dependency/AST slice, specification handles, failure traces
* satisfied proofs as digest/handle only
* optional source/evidence VoI-ranked with expansion handles
* untrusted repository text labeled as data (cannot inject instructions)
* solver traces excluded by default
* required claim/acceptance coverage never deferred as optional
* token budget + omitted-handle manifest for audit
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .code_proof_query import (
    ClaimQueryHit,
    CodeProofQuery,
    CodeProofQueryResult,
    ProofDeltaResult,
    build_code_proof_query,
)
from .code_claim_contracts import ClaimStatus, CodeClaimRecord
from .code_proof_obligations import CodeProofObligationCompilation
from ..context.context_compiler import (
    ContextCompileResult,
    ContextCompiler,
    ContextDeltaResult,
    compile_context_capsule,
    compile_context_delta,
    reconstruct_context,
)
from ..context.context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextReference,
    ContextTier,
)
from .formal_verification_contracts import content_identity


CODE_PROOF_CONTEXT_INTERFACE: Final = "CodeProofContext@1"
CODE_PROOF_CONTEXT_VERSION: Final = "1"
CODE_PROOF_CONTEXT_DELTA_INTERFACE: Final = "CodeProofContextDelta@1"
UNTRUSTED_DATA_LABEL: Final = "untrusted_repository_data"


class CodeProofContextError(ValueError):
    """Code-proof context profile input is malformed."""


def _tokens_for(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 23) // 24)


def _ref(
    *,
    reference_id: str,
    kind: str,
    tier: ContextTier,
    content: Mapping[str, Any] | str,
    repository_id: str,
    tree_id: str,
    path: str = "",
    summary: str = "",
    required: bool = False,
    priority: int = 0,
    metadata: Mapping[str, Any] | None = None,
    untrusted_data: bool = False,
) -> ContextReference:
    if isinstance(content, str):
        body: Any = content
        payload = content
    else:
        body = dict(content)
        payload = body
    digest = content_identity(payload)
    text = summary or str(payload)[:240]
    meta = {
        "required": bool(required),
        "priority": int(priority),
        "coverage_ids": (f"coverage:{reference_id}",),
        "code_proof_context": True,
    }
    if untrusted_data:
        meta["data_label"] = UNTRUSTED_DATA_LABEL
        meta["instruction_injection"] = False
        meta["treat_as"] = "data_not_instructions"
    if metadata:
        meta.update(dict(metadata))
    return ContextReference(
        reference_id=reference_id,
        kind=kind,
        tier=tier,
        referenced_content_id=digest
        if str(digest).startswith("sha256:") or ":" in str(digest)
        else f"sha256:{digest}",
        repository_id=repository_id,
        tree_id=tree_id,
        path=path,
        summary=text[:500],
        token_count=_tokens_for(text),
        byte_count=len(text.encode("utf-8")),
        metadata=meta,
    )


@dataclass(frozen=True)
class CodeProofContextRequest:
    """Inputs for an obligation-first capsule."""

    repository_id: str
    tree_id: str
    task_id: str
    acceptance_ids: tuple[str, ...]
    objective_id: str = "CBP-G060"
    objective_revision: str = "cbp-context@1"
    policy_id: str = "policy:code-proof"
    policy_revision: str = "sha256:policy-code-proof"
    caller: str = "supervisor:code-proof"
    stage: str = "implementation"
    query: CodeProofQuery | None = None
    claims: tuple[CodeClaimRecord, ...] = ()
    compilation: CodeProofObligationCompilation | None = None
    changed_paths: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    specification_handles: tuple[str, ...] = ()
    failure_traces: tuple[Mapping[str, Any], ...] = ()
    optional_source_snippets: tuple[Mapping[str, Any], ...] = ()
    include_solver_traces: bool = False
    budget: ContextBudget | None = None
    goal_summary: str = "Discharge open code-proof obligations without bulk source"

    def __post_init__(self) -> None:
        if not str(self.task_id or "").strip():
            raise CodeProofContextError("task_id is required")
        if not self.acceptance_ids:
            raise CodeProofContextError("acceptance_ids must be non-empty")
        object.__setattr__(
            self,
            "acceptance_ids",
            tuple(str(a).strip() for a in self.acceptance_ids if str(a).strip()),
        )


@dataclass(frozen=True)
class CodeProofContextCapsule:
    """Audit wrapper around a compiled ContextCompileResult."""

    task_id: str
    acceptance_ids: tuple[str, ...]
    open_obligation_ids: tuple[str, ...]
    satisfied_receipt_handles: tuple[str, ...]
    counterexample_ids: tuple[str, ...]
    expansion_handle_ids: tuple[str, ...]
    omitted_handles: tuple[str, ...]
    token_budget: Mapping[str, Any]
    compile_result: ContextCompileResult
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def capsule(self):
        return self.compile_result.capsule

    @property
    def capsule_id(self) -> str:
        return content_identity(
            {
                "interface": CODE_PROOF_CONTEXT_INTERFACE,
                "task_id": self.task_id,
                "acceptance_ids": list(self.acceptance_ids),
                "open_obligation_ids": list(self.open_obligation_ids),
                "satisfied_receipt_handles": list(self.satisfied_receipt_handles),
                "counterexample_ids": list(self.counterexample_ids),
                "expansion_handle_ids": list(self.expansion_handle_ids),
                "omitted_handles": list(self.omitted_handles),
                "context_capsule_id": getattr(self.capsule, "capsule_id", ""),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": CODE_PROOF_CONTEXT_INTERFACE,
            "version": CODE_PROOF_CONTEXT_VERSION,
            "task_id": self.task_id,
            "acceptance_ids": list(self.acceptance_ids),
            "open_obligation_ids": list(self.open_obligation_ids),
            "satisfied_receipt_handles": list(self.satisfied_receipt_handles),
            "counterexample_ids": list(self.counterexample_ids),
            "expansion_handle_ids": list(self.expansion_handle_ids),
            "omitted_handles": list(self.omitted_handles),
            "token_budget": dict(self.token_budget),
            "capsule_id": self.capsule_id,
            "context_capsule_id": getattr(self.capsule, "capsule_id", ""),
            "input_tokens": getattr(self.capsule, "input_tokens", 0),
            "truncated": getattr(self.capsule, "truncated", False),
            "metadata": dict(self.metadata),
        }


def _default_budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=4_000,
        reserved_output_tokens=800,
        reserved_tool_tokens=200,
        max_items=64,
        max_item_bytes=16_384,
        max_serialized_bytes=512_000,
        max_depth=10,
        max_text_bytes=16_384,
    )


def _resolve_query(request: CodeProofContextRequest) -> CodeProofQuery:
    if request.query is not None:
        return request.query
    return build_code_proof_query(
        claims=request.claims,
        compilation=request.compilation,
    )


def build_code_proof_context_references(
    request: CodeProofContextRequest,
) -> tuple[tuple[ContextReference, ...], dict[str, Any]]:
    """Build invariant + optional evidence references for the capsule."""

    query = _resolve_query(request)
    repo = request.repository_id
    tree = request.tree_id
    refs: list[ContextReference] = []

    # --- invariant core ---
    refs.append(
        _ref(
            reference_id=f"task:{request.task_id}",
            kind="task_identity",
            tier=ContextTier.INVARIANT,
            content={
                "task_id": request.task_id,
                "acceptance_ids": list(request.acceptance_ids),
            },
            repository_id=repo,
            tree_id=tree,
            summary=f"task {request.task_id}",
            required=True,
            priority=0,
            metadata={"acceptance_ids": list(request.acceptance_ids)},
        )
    )
    refs.append(
        _ref(
            reference_id="acceptance:criteria",
            kind="acceptance",
            tier=ContextTier.INVARIANT,
            content={"acceptance_ids": list(request.acceptance_ids)},
            repository_id=repo,
            tree_id=tree,
            summary="acceptance criteria ids",
            required=True,
            priority=0,
        )
    )

    open_hits = query.properties_open().hits
    open_ids: list[str] = []
    for hit in open_hits:
        oid = hit.obligation_ids[0] if hit.obligation_ids else hit.claim_id
        open_ids.append(oid)
        refs.append(
            _ref(
                reference_id=f"obligation:open:{hit.property_id}",
                kind="open_obligation",
                tier=ContextTier.INVARIANT,
                content={
                    "property_id": hit.property_id,
                    "claim_id": hit.claim_id,
                    "obligation_ids": list(hit.obligation_ids),
                    "status": hit.status.value,
                    "reason_codes": list(hit.reason_codes),
                },
                repository_id=repo,
                tree_id=tree,
                summary=f"open {hit.property_id}",
                required=True,
                priority=1,
                metadata={
                    "claim_id": hit.claim_id,
                    "property_id": hit.property_id,
                    "coverage_ids": (f"coverage:claim:{hit.property_id}",),
                },
            )
        )

    # Assumption ids from open claims / compilation
    assumption_ids: set[str] = set()
    if request.compilation is not None:
        assumption_ids.update(request.compilation.assumption_ids)
    for claim in request.claims:
        assumption_ids.update(claim.assumption_ids or ())
    if assumption_ids:
        refs.append(
            _ref(
                reference_id="assumptions:bound",
                kind="assumptions",
                tier=ContextTier.INVARIANT,
                content={"assumption_ids": sorted(assumption_ids)},
                repository_id=repo,
                tree_id=tree,
                summary="bound assumption ids",
                required=True,
                priority=2,
            )
        )

    counterexamples = query.counterexamples().hits
    cex_ids: list[str] = []
    for hit in counterexamples:
        cex_ids.append(hit.claim_id)
        refs.append(
            _ref(
                reference_id=f"counterexample:{hit.property_id}",
                kind="counterexample",
                tier=ContextTier.INVARIANT,
                content={
                    "property_id": hit.property_id,
                    "claim_id": hit.claim_id,
                    "counterexample": hit.counterexample or {},
                    "reason_codes": list(hit.reason_codes),
                },
                repository_id=repo,
                tree_id=tree,
                summary=f"counterexample {hit.property_id}",
                required=True,
                priority=2,
            )
        )

    # Dependency / AST slice (handles only — no bulk source)
    if request.changed_paths or request.changed_symbols:
        refs.append(
            _ref(
                reference_id="slice:dependency-ast",
                kind="dependency_ast_slice",
                tier=ContextTier.INVARIANT,
                content={
                    "changed_paths": list(request.changed_paths),
                    "changed_symbols": list(request.changed_symbols),
                },
                repository_id=repo,
                tree_id=tree,
                summary="changed dependency/AST slice handles",
                required=True,
                priority=3,
            )
        )

    for handle in request.specification_handles:
        refs.append(
            _ref(
                reference_id=f"spec:{handle}",
                kind="specification_handle",
                tier=ContextTier.INVARIANT,
                content={"handle": handle},
                repository_id=repo,
                tree_id=tree,
                summary=f"spec {handle}",
                required=True,
                priority=3,
                metadata={"specification_handle": handle},
            )
        )

    for index, trace in enumerate(request.failure_traces):
        refs.append(
            _ref(
                reference_id=f"failure-trace:{index}",
                kind="failure_trace",
                tier=ContextTier.INVARIANT,
                content=dict(trace),
                repository_id=repo,
                tree_id=tree,
                summary=str(trace.get("summary") or f"failure trace {index}")[:200],
                required=True,
                priority=4,
                metadata={"bounded_failure_trace": True},
            )
        )

    # Satisfied proofs — digest/handle only
    satisfied_handles: list[str] = []
    for hit in query.properties_satisfied().hits:
        handle = hit.receipt_id or hit.claim_id
        satisfied_handles.append(handle)
        refs.append(
            _ref(
                reference_id=f"satisfied:{hit.property_id}",
                kind="satisfied_proof_handle",
                tier=ContextTier.EVIDENCE,
                content={
                    "property_id": hit.property_id,
                    "claim_id": hit.claim_id,
                    "receipt_id": hit.receipt_id,
                    "digest_only": True,
                },
                repository_id=repo,
                tree_id=tree,
                summary=f"satisfied handle {handle[:48]}",
                required=False,
                priority=10,
                metadata={"digest_only": True, "no_body": True},
            )
        )

    # Optional source snippets — untrusted data, VoI ranked via priority
    expansion_ids: list[str] = []
    for index, snippet in enumerate(request.optional_source_snippets):
        path = str(snippet.get("path") or f"snippet:{index}")
        # Never put raw large source in invariant tier.
        body = str(snippet.get("text") or snippet.get("summary") or path)[:400]
        ref = _ref(
            reference_id=f"source-optional:{index}",
            kind="optional_source",
            tier=ContextTier.EVIDENCE,
            content={
                "path": path,
                "preview": body,
                "handle": snippet.get("handle") or path,
            },
            repository_id=repo,
            tree_id=tree,
            path=path,
            summary=f"optional source {path}",
            required=False,
            priority=20 + index,
            untrusted_data=True,
            metadata={
                "voi_rank": index,
                "expansion_candidate": True,
            },
        )
        expansion_ids.append(ref.reference_id)
        refs.append(ref)

    # Solver traces excluded by default
    if request.include_solver_traces:
        refs.append(
            _ref(
                reference_id="solver-trace:optional",
                kind="solver_trace",
                tier=ContextTier.SUGGESTION,
                content={"note": "solver trace explicitly requested"},
                repository_id=repo,
                tree_id=tree,
                summary="solver trace (explicit)",
                required=False,
                priority=50,
            )
        )

    manifest = {
        "open_obligation_ids": open_ids,
        "satisfied_receipt_handles": satisfied_handles,
        "counterexample_ids": cex_ids,
        "expansion_handle_ids": expansion_ids,
        "solver_traces_excluded_by_default": not request.include_solver_traces,
        "untrusted_data_label": UNTRUSTED_DATA_LABEL,
    }
    return tuple(refs), manifest


def compile_code_proof_context_capsule(
    request: CodeProofContextRequest,
    *,
    tokenizer: Any | None = None,
    provider_context_window: int | None = None,
) -> CodeProofContextCapsule:
    """Compile an obligation-first :class:`ContextCapsule` for CBP agents."""

    if not isinstance(request, CodeProofContextRequest):
        raise CodeProofContextError("request must be a CodeProofContextRequest")

    budget = request.budget or _default_budget()
    evidence, manifest = build_code_proof_context_references(request)

    goal = {
        "id": request.objective_id,
        "task_id": request.task_id,
        "summary": request.goal_summary,
        "open_obligations": list(manifest["open_obligation_ids"]),
    }
    authority = {
        "mode": "implementation",
        "allowed_paths": list(request.changed_paths) or ["*"],
        "untrusted_repository_text_is_data": True,
    }
    scope = {
        "paths": list(request.changed_paths),
        "symbols": list(request.changed_symbols),
        "specification_handles": list(request.specification_handles),
    }
    acceptance = {
        "criteria": list(request.acceptance_ids),
        "required_claim_coverage": list(manifest["open_obligation_ids"]),
        "cannot_defer_required_claims": True,
    }

    result = compile_context_capsule(
        budget,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_id=request.objective_id,
        objective_revision=request.objective_revision,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        caller=request.caller,
        stage=request.stage,
        goal=goal,
        authority=authority,
        scope=scope,
        acceptance=acceptance,
        evidence=evidence,
        tokenizer=tokenizer or (lambda text: _tokens_for(str(text))),
        provider_context_window=provider_context_window,
    )

    omitted = tuple(getattr(result.capsule, "omissions", ()) or ())
    expansion_handles = tuple(
        ref.reference_id
        for ref in (getattr(result.capsule, "expansion_references", ()) or ())
    )
    # Prefer expansion candidates that were omitted under budget.
    if not expansion_handles:
        expansion_handles = tuple(manifest.get("expansion_handle_ids") or ())

    token_budget = {
        "max_input_tokens": budget.max_input_tokens,
        "reserved_output_tokens": budget.reserved_output_tokens,
        "reserved_tool_tokens": budget.reserved_tool_tokens,
        "input_tokens": getattr(result.capsule, "input_tokens", 0),
        "effective_input_limit": getattr(
            getattr(result, "budget_resolution", None),
            "effective_input_limit",
            budget.max_input_tokens,
        ),
    }

    return CodeProofContextCapsule(
        task_id=request.task_id,
        acceptance_ids=request.acceptance_ids,
        open_obligation_ids=tuple(manifest["open_obligation_ids"]),
        satisfied_receipt_handles=tuple(manifest["satisfied_receipt_handles"]),
        counterexample_ids=tuple(manifest["counterexample_ids"]),
        expansion_handle_ids=expansion_handles,
        omitted_handles=omitted,
        token_budget=token_budget,
        compile_result=result,
        metadata={
            "interface": CODE_PROOF_CONTEXT_INTERFACE,
            "version": CODE_PROOF_CONTEXT_VERSION,
            "solver_traces_excluded_by_default": manifest[
                "solver_traces_excluded_by_default"
            ],
            "untrusted_data_label": UNTRUSTED_DATA_LABEL,
        },
    )


# ---------------------------------------------------------------------------
# CBP-070: proof_delta-driven retry contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeProofContextDeltaCapsule:
    """Parent-bound retry capsule carrying only proof_delta evidence."""

    parent_capsule_id: str
    task_id: str
    delta: ProofDeltaResult
    reopened_property_ids: tuple[str, ...]
    still_valid_property_ids: tuple[str, ...]
    cold_input_tokens: int
    retry_input_tokens: int
    delta_result: ContextDeltaResult
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def delta_capsule(self):
        return self.delta_result.delta_capsule

    @property
    def token_reduction_ratio(self) -> float:
        if self.cold_input_tokens <= 0:
            return 0.0
        return 1.0 - (self.retry_input_tokens / float(self.cold_input_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": CODE_PROOF_CONTEXT_DELTA_INTERFACE,
            "parent_capsule_id": self.parent_capsule_id,
            "task_id": self.task_id,
            "delta": self.delta.to_dict(),
            "reopened_property_ids": list(self.reopened_property_ids),
            "still_valid_property_ids": list(self.still_valid_property_ids),
            "cold_input_tokens": self.cold_input_tokens,
            "retry_input_tokens": self.retry_input_tokens,
            "token_reduction_ratio_millis": int(
                round(self.token_reduction_ratio * 1_000_000)
            ),
            "metadata": dict(self.metadata),
        }


def _property_ids_from_query(query: CodeProofQuery) -> set[str]:
    return {hit.property_id for hit in query.hits}


def _delta_evidence_for_reopened(
    *,
    child_query: CodeProofQuery,
    reopened_property_ids: Sequence[str],
    repository_id: str,
    tree_id: str,
    delta: ProofDeltaResult,
) -> tuple[ContextReference, ...]:
    """Build only the delta evidence for reopened/invalidated properties."""

    reopened = set(reopened_property_ids)
    hits_by_prop = {hit.property_id: hit for hit in child_query.hits}
    refs: list[ContextReference] = []

    # Always include a compact proof_delta summary (required for retry).
    refs.append(
        _ref(
            reference_id="proof-delta:summary",
            kind="proof_delta",
            tier=ContextTier.INVARIANT,
            content={
                "parent_tree_id": delta.parent_tree_id,
                "child_tree_id": delta.child_tree_id,
                "entry_count": len(delta.entries),
                "reopened_property_ids": sorted(reopened),
                "entries": [entry.to_dict() for entry in delta.entries],
            },
            repository_id=repository_id,
            tree_id=tree_id,
            summary=f"proof_delta {len(delta.entries)} entries",
            required=True,
            priority=0,
            metadata={"proof_delta_only": True},
        )
    )

    for property_id in sorted(reopened):
        hit = hits_by_prop.get(property_id)
        if hit is None:
            refs.append(
                _ref(
                    reference_id=f"delta:missing:{property_id}",
                    kind="reopened_obligation",
                    tier=ContextTier.INVARIANT,
                    content={
                        "property_id": property_id,
                        "status": "missing_on_child",
                    },
                    repository_id=repository_id,
                    tree_id=tree_id,
                    summary=f"reopened missing {property_id}",
                    required=True,
                    priority=1,
                )
            )
            continue
        refs.append(
            _ref(
                reference_id=f"delta:open:{property_id}",
                kind="reopened_obligation",
                tier=ContextTier.INVARIANT,
                content={
                    "property_id": hit.property_id,
                    "claim_id": hit.claim_id,
                    "obligation_ids": list(hit.obligation_ids),
                    "status": hit.status.value,
                    "reason_codes": list(hit.reason_codes),
                },
                repository_id=repository_id,
                tree_id=tree_id,
                summary=f"reopened {property_id}",
                required=True,
                priority=1,
                metadata={
                    "claim_id": hit.claim_id,
                    "property_id": property_id,
                    "coverage_ids": (f"coverage:claim:{property_id}",),
                },
            )
        )
        if hit.counterexample is not None or hit.status is ClaimStatus.REFUTED:
            refs.append(
                _ref(
                    reference_id=f"delta:cex:{property_id}",
                    kind="counterexample",
                    tier=ContextTier.INVARIANT,
                    content={
                        "property_id": property_id,
                        "counterexample": hit.counterexample or {},
                    },
                    repository_id=repository_id,
                    tree_id=tree_id,
                    summary=f"delta counterexample {property_id}",
                    required=True,
                    priority=2,
                )
            )
    return tuple(refs)


def compile_code_proof_context_delta(
    parent: CodeProofContextCapsule,
    child_request: CodeProofContextRequest,
    *,
    parent_query: CodeProofQuery,
    tokenizer: Any | None = None,
    provider_context_window: int | None = None,
) -> CodeProofContextDeltaCapsule:
    """Compile a proof_delta-only retry context bound to a parent capsule.

    Still-valid cached obligations are **not** re-opened in the delta unless
    :meth:`CodeProofQuery.proof_delta` reports an invalidation reason.
    Parent-bound reconstruction preserves the invariant core.
    """

    if not isinstance(parent, CodeProofContextCapsule):
        raise CodeProofContextError("parent must be a CodeProofContextCapsule")
    if not isinstance(child_request, CodeProofContextRequest):
        raise CodeProofContextError("child_request must be a CodeProofContextRequest")
    if not isinstance(parent_query, CodeProofQuery):
        raise CodeProofContextError("parent_query must be a CodeProofQuery")

    parent_capsule: ContextCapsule = parent.capsule
    child_query = _resolve_query(child_request)

    delta = child_query.proof_delta(parent_query)
    reopened = tuple(sorted({entry.property_id for entry in delta.entries}))
    parent_props = _property_ids_from_query(parent_query)
    still_valid = tuple(sorted(parent_props - set(reopened)))

    # Still-valid properties must not appear as reopened open-obligation refs.
    # ContextDeltaCapsule forbids changing immutable tree identity on evidence
    # rows — keep parent tree_id on references; child tree is recorded inside
    # proof_delta payload content.
    delta_updates = _delta_evidence_for_reopened(
        child_query=child_query,
        reopened_property_ids=reopened,
        repository_id=parent_capsule.repository_id or child_request.repository_id,
        tree_id=parent_capsule.tree_id,
        delta=delta,
    )
    if not delta_updates:
        raise CodeProofContextError("proof_delta produced no retry evidence")

    # compile_delta requires every parent-required reference id to remain present
    # (and still required). Unchanged required refs are listed but not re-sent
    # as "changed" payloads when content identity matches.
    parent_required = {
        ref.reference_id: ref
        for ref in parent_capsule.evidence
        if ref.required
    }
    candidates: dict[str, ContextReference] = dict(parent_required)
    for ref in delta_updates:
        candidates[ref.reference_id] = ref
    delta_evidence = tuple(
        candidates[key] for key in sorted(candidates)
    )

    budget = child_request.budget or _default_budget()
    delta_result = compile_context_delta(
        budget,
        parent_capsule,
        evidence=delta_evidence,
        stage=child_request.stage or parent_capsule.stage,
        tokenizer=tokenizer or (lambda text: _tokens_for(str(text))),
        provider_context_window=provider_context_window,
    )

    cold_tokens = int(
        parent.token_budget.get("input_tokens") or parent_capsule.input_tokens
    )
    # Prefer delta transmission tokens (not full reconstructed size).
    receipt = getattr(delta_result, "receipt", None)
    if receipt is not None and getattr(receipt, "delta_tokens", None) is not None:
        retry_tokens = int(receipt.delta_tokens)
    else:
        retry_tokens = int(
            sum(
                int(getattr(ref, "token_count", 0) or 0)
                for ref in delta_result.delta_capsule.evidence
            )
        )

    reconstructed = reconstruct_context(parent_capsule, delta_result.delta_capsule)
    if reconstructed.objective_id != parent_capsule.objective_id:
        raise CodeProofContextError("delta reconstruction lost objective core")
    if reconstructed.policy_id != parent_capsule.policy_id:
        raise CodeProofContextError("delta reconstruction lost policy core")
    if reconstructed.goal != parent_capsule.goal:
        raise CodeProofContextError("delta reconstruction lost goal core")

    return CodeProofContextDeltaCapsule(
        parent_capsule_id=str(parent_capsule.capsule_id),
        task_id=child_request.task_id or parent.task_id,
        delta=delta,
        reopened_property_ids=reopened,
        still_valid_property_ids=still_valid,
        cold_input_tokens=cold_tokens,
        retry_input_tokens=retry_tokens,
        delta_result=delta_result,
        metadata={
            "interface": CODE_PROOF_CONTEXT_DELTA_INTERFACE,
            "proof_delta_only": True,
            "still_valid_not_reopened": True,
            "cache_reuse_expected_for_still_valid": True,
        },
    )


__all__ = [
    "CODE_PROOF_CONTEXT_INTERFACE",
    "CODE_PROOF_CONTEXT_VERSION",
    "CODE_PROOF_CONTEXT_DELTA_INTERFACE",
    "UNTRUSTED_DATA_LABEL",
    "CodeProofContextError",
    "CodeProofContextRequest",
    "CodeProofContextCapsule",
    "CodeProofContextDeltaCapsule",
    "build_code_proof_context_references",
    "compile_code_proof_context_capsule",
    "compile_code_proof_context_delta",
]
