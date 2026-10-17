"""Bind existing context compilation to one current repository source session.

Required evidence is a complete caller-reviewed population, never inferred from
lexical retrieval. Native capsules stay advisory and retain their datasets CIDs.
This adapter owns only bounded, invocation-local parent bindings; it creates no
catalog, artifact store, proof, training job or model invocation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import threading
import time

from ..context.context_compiler import (
    ContextCompilationError, ContextCompiler, EvidenceExpansionRequest,
    ContextExpansionCancelled,
)
from ..context.context_contracts import ContextCapsule, ContextReference, ContextTier
from .repository_source_session import RepositorySourceSession
from .supervisor_meta_index import public_replay_without_metadata

MAX_REFERENCES = 32
MAX_CONTEXT_BYTES = 262144
MAX_PARENTS = 8
PREFIX = "repository-symbol:"
IDENTITY_FIELDS = ("generation", "source_tree_sha256", "semantic_state_cid",
                   "semantic_root_cid", "capsule_index_cid")
FALSE_FIELDS = ("proof_authority", "execution_authority", "completion_authority",
                "semantic_equivalence_claimed")


class RepositoryContextError(ContextCompilationError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryContextError(code)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _ids(values, *, maximum=MAX_REFERENCES):
    _require(type(values) in (tuple, list) and len(values) <= maximum
             and all(type(v) is str and 0 < len(v.encode("utf-8")) <= 1024 for v in values),
             "invalid_context_ids")
    _require(len(values) == len(set(values)), "duplicate_context_ids")
    return tuple(sorted(values))


def _references(values):
    _require(type(values) in (tuple, list) and len(values) <= MAX_REFERENCES
             and all(isinstance(v, ContextReference) for v in values), "invalid_context_references")
    by_id = {}
    for value in values:
        _require(value.reference_id not in by_id or by_id[value.reference_id] == value,
                 "conflicting_context_reference")
        by_id[value.reference_id] = value
    _require(sum(len(v.canonical_bytes()) for v in by_id.values()) <= MAX_CONTEXT_BYTES,
             "context_reference_bytes")
    return by_id


def _cancel(cancelled):
    _require(cancelled is None or type(cancelled) is bool or callable(cancelled)
             or callable(getattr(cancelled, "is_set", None)), "invalid_context_cancellation")
    value = (cancelled if cancelled is None or type(cancelled) is bool else
             cancelled() if callable(cancelled) else cancelled.is_set())
    if value:
        raise ContextExpansionCancelled("repository context expansion cancelled")


@dataclass(frozen=True)
class _Parent:
    capsule: ContextCapsule
    query_arguments: dict
    query: dict
    binding: dict
    required: dict
    handles: dict
    question: str


class RepositoryContextAdapter:
    """Compile, expand and retry using existing compiler and native owners.

The caller supplies authoritative required IDs/coverage along with every full
required reference. Missing bodies or insufficient budgets refuse compilation.
Selected optional handles are model-visible; compiler-omitted handles remain
supervisor-side until an explicitly charged discovery or expansion is sent.
"""

    def __init__(self, session: RepositorySourceSession, compiler: ContextCompiler):
        _require(isinstance(session, RepositorySourceSession) and isinstance(compiler, ContextCompiler),
                 "invalid_context_owners")
        _require(compiler.effective_budget.max_items <= 128
                 and compiler.effective_budget.max_serialized_bytes <= MAX_CONTEXT_BYTES,
                 "repository_context_budget_limit")
        self.session, self.compiler = session, compiler
        self._parents = {}
        self._lock = threading.RLock()

    def _observe(self, arguments):
        query = self.session.context_query(**arguments)
        status = self.session.status_summary()
        _require(status["source_current"] is True and status["generation"] == query["generation"]
                 and status["source_tree_sha256"] == query["source_tree_sha256"]
                 and status["semantic_state_cid"] == query["semantic_state_cid"], "context_source_changed")
        binding = {key: query[key] for key in IDENTITY_FIELDS}
        binding.update(session_id=status["session_id"], repository_id=status["datasets_head"]["repository_id"])
        return query, binding

    def _current(self, state):
        query, binding = self._observe(state.query_arguments)
        _require(binding == state.binding and query == state.query, "context_source_changed")

    def _compile(self, method, *args, **arguments):
        from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources

        # Native fences release their own leases before compiler admission.
        # Borrow the index's parent when supplied, so compilation does not
        # acquire an independent root beneath an admitted caller's envelope.
        from .repository_source_index import NativeRepositoryIndex

        index = self.session._index
        deadline = time.monotonic() + 30.0
        inherited = getattr(self.session, "_deadline_monotonic", None)
        if inherited is not None:
            deadline = min(deadline, inherited)
        if isinstance(index, NativeRepositoryIndex) and index._deadline_monotonic is not None:
            deadline = min(deadline, index._deadline_monotonic)
        controls = (NativeRepositoryIndex._resource_options(index) if isinstance(index, NativeRepositoryIndex)
                    else dict(scheduler=getattr(index, "_scheduler", None)))
        remaining = deadline - time.monotonic()
        _require(remaining > 0, "context_compilation_timeout")
        with acquire_codebase_resources(**controls,
                                        timeout_seconds=min(10.0, remaining), memory_mb=256) as lease:
            cancelled = lease.combined_cancellation_signal(controls.get("cancel_event"))
            _require(not cancelled.is_set(), "context_compilation_cancelled")
            _require(time.monotonic() < deadline, "context_compilation_timeout")
            result = getattr(self.compiler, method)(*args, **arguments)
            _require(not cancelled.is_set(), "context_compilation_cancelled")
            _require(time.monotonic() < deadline, "context_compilation_timeout")
            return result

    def _save(self, capsule, state):
        _require(len(capsule.canonical_bytes()) <= MAX_CONTEXT_BYTES, "repository_context_byte_limit")
        _require(capsule.capsule_id in self._parents or len(self._parents) < MAX_PARENTS,
                 "repository_context_parent_limit")
        required = {v.reference_id: v for v in capsule.evidence if v.required}
        _require(required == state.required, "required_context_changed")
        # Parent bookkeeping must not publish a result that crossed the
        # inherited deadline while canonicalizing or copying its evidence.
        from .repository_source_index import NativeRepositoryIndex

        def checkpoint():
            self.session._checkpoint()
            if isinstance(self.session._index, NativeRepositoryIndex):
                NativeRepositoryIndex._resource_options(self.session._index)

        replacement = replace(state, capsule=capsule)
        checkpoint()
        previous = self._parents.get(capsule.capsule_id)
        self._parents[capsule.capsule_id] = replacement
        try:
            checkpoint()
        except BaseException:
            if previous is None:
                self._parents.pop(capsule.capsule_id, None)
            else:
                self._parents[capsule.capsule_id] = previous
            raise

    def _parent(self, parent, generation):
        _require(isinstance(parent, ContextCapsule), "invalid_context_parent")
        state = self._parents.get(parent.capsule_id)
        _require(state is not None and parent == state.capsule, "unknown_context_parent")
        _require(type(generation) is int and generation == state.binding["generation"],
                 "stale_context_generation")
        self._current(state)
        return state

    def compile(self, *, generation, query, unresolved_question, required_evidence,
                required_reference_ids, required_coverage_ids, objective_id,
                objective_revision, policy_id, policy_revision, goal, authority,
                scope, acceptance, limit=8, max_bytes=32768, optional_evidence=()):
        """Preserve reviewed mandatory bodies and add bounded native handles."""
        with self._lock, public_replay_without_metadata():
            _require(type(unresolved_question) is str and unresolved_question.strip()
                     and len(unresolved_question.encode("utf-8")) <= 1024, "invalid_context_question")
            question = unresolved_question.strip()
            required = _references(required_evidence)
            ids = _ids(required_reference_ids)
            coverage = _ids(required_coverage_ids, maximum=64)
            _require(tuple(sorted(required)) == ids and all(v.required for v in required.values()),
                     "incomplete_required_context")
            _require(all(v.summary.strip() for v in required.values()), "required_context_body_missing")
            _require(tuple(sorted({c for v in required.values() for c in v.coverage_ids})) == coverage,
                     "incomplete_required_coverage")
            _require(not any(v.reference_id.startswith(PREFIX) for v in required.values()),
                     "reserved_context_reference")
            optional = _references(optional_evidence)
            _require(not set(optional).intersection(required)
                     and all(not v.required for v in optional.values()), "invalid_optional_context")
            _require(not any(v.reference_id.startswith(PREFIX) for v in optional.values()),
                     "reserved_context_reference")
            _require(sum(len(v.canonical_bytes()) for v in (*required.values(), *optional.values()))
                     <= MAX_CONTEXT_BYTES, "context_reference_bytes")
            _require(len(self._parents) < MAX_PARENTS, "repository_context_parent_limit")
            arguments = dict(generation=generation, query=query, limit=limit, max_bytes=max_bytes)
            selected, binding = self._observe(arguments)
            tree = "sha256:" + binding["source_tree_sha256"]
            handles = {}
            for row in selected["candidates"]:
                ref = ContextReference(reference_id=PREFIX + row["symbol_id"],
                    kind="repository-semantic-capsule", tier=ContextTier.EVIDENCE,
                    referenced_content_id=row["capsule_cid"], repository_id=binding["repository_id"],
                    tree_id=tree, path=row["path"], summary=row["name"], metadata=dict(
                        source_sha256=row["source_sha256"], source_cid=row["source_cid"],
                        confidence=row["confidence"], raw_source_required=row["raw_source_required"],
                        unresolved_questions=(question,), question_bound_expansion=True,
                        authority="structural_only", proof_authority=False))
                _require(ref.reference_id not in handles, "duplicate_native_context")
                handles[ref.reference_id] = ref
            provenance = dict(binding, required_reference_ids=ids, required_coverage_ids=coverage,
                **{key: selected[key] for key in ("query", "ranking", "population_count", "matching_count",
                                                "selected_count", "omitted_count", "truncated")})
            result = self._compile("compile", repository_id=binding["repository_id"], tree_id=tree,
                objective_id=objective_id, objective_revision=objective_revision,
                policy_id=policy_id, policy_revision=policy_revision,
                caller="repository-context-adapter", stage="planning", goal=goal,
                authority=dict(task=authority, repository_context=provenance,
                               proof_authority=False, execution_authority=False, completion_authority=False),
                scope=scope, acceptance=acceptance,
                evidence=(*required.values(), *optional.values(), *handles.values()))
            state = _Parent(result.capsule, arguments, selected, binding, required, handles, question)
            self._current(state)
            self._save(result.capsule, state)
            return result

    def validate_current(self, parent, *, generation):
        """Check an issued context at a consumer boundary without expanding it.

        This is a cooperative freshness fence, not a lock across provider work.
        Consumers must repeat it after an asynchronous or queued call returns.
        """
        with self._lock, public_replay_without_metadata():
            self._parent(parent, generation)

    def expand_for_question(self, parent, request, *, generation, max_bytes=65536, cancelled=None):
        """Resolve exact native CIDs into a question-bound compiler delta."""
        from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
        from ..semantic_state.capsules import CapsuleAdmission

        with self._lock, public_replay_without_metadata():
            _cancel(cancelled)
            _require(isinstance(request, EvidenceExpansionRequest), "invalid_context_expansion")
            requested = _ids(request.reference_ids)
            state = self._parent(parent, generation)
            _require(request.unresolved_question == state.question, "unbound_context_question")
            visible = {v.reference_id: v for v in (*parent.evidence, *parent.expansion_references)}
            _require(all(i in state.handles and i in visible for i in requested), "unknown_context_reference")
            symbols = [i.removeprefix(PREFIX) for i in requested]
            expanded = self.session.capsule_expand(generation=generation,
                semantic_root_cid=state.binding["semantic_root_cid"], stable_symbol_ids=symbols, max_bytes=max_bytes)
            _require(all(expanded.get(k) == state.binding[k] for k in IDENTITY_FIELDS)
                     and expanded.get("authority") == "structural_only"
                     and all(expanded.get(k) is False for k in FALSE_FIELDS)
                     and expanded["requested_stable_symbol_ids"] == symbols
                     and len(expanded["capsules"]) == len(expanded["admissions"]) == len(symbols),
                     "invalid_context_expansion_binding")
            replacements, sources = {}, {}
            for identity, raw, admission_raw in zip(requested, expanded["capsules"], expanded["admissions"]):
                _cancel(cancelled)
                handle = state.handles[identity]
                capsule, admission = SemanticCapsule.from_dict(raw), CapsuleAdmission.from_dict(admission_raw)
                _require(capsule.capsule_cid == handle.referenced_content_id == admission.ref.capsule_cid
                    and capsule.stable_symbol_id == identity.removeprefix(PREFIX) == admission.ref.stable_symbol_id
                    and capsule.source_cid == handle.metadata["source_cid"] == admission.ref.source_cid
                    and capsule.version_cid == admission.ref.version_cid
                    and capsule.source_slice_path == handle.path
                    and admission.ref.semantic_state_root_cid == state.binding["semantic_root_cid"]
                    and admission.ref.confidence == handle.metadata["confidence"]
                    and str(getattr(capsule.confidence, "value", capsule.confidence)) == admission.ref.confidence
                    and admission.requires_raw_source == handle.metadata["raw_source_required"]
                    and admission.freshness == "fresh", "invalid_context_native_capsule")
                sources[handle.path] = dict(source_sha256=handle.metadata["source_sha256"],
                                            source_cid=handle.metadata["source_cid"])
                body = _json(dict(capsule=raw, admission=admission_raw,
                    source=sources[handle.path], raw_source_policy=expanded["raw_source_policy"],
                    authority="structural_only", proof_authority=False))
                replacements[identity] = ContextReference(reference_id=identity, kind=handle.kind,
                    tier=ContextTier.EVIDENCE, referenced_content_id=handle.referenced_content_id,
                    repository_id=handle.repository_id, tree_id=handle.tree_id, path=handle.path,
                    summary=body, byte_count=len(body.encode("utf-8")), metadata=dict(handle.metadata,
                        repository_capsule_expanded=True, expansion_question=request.unresolved_question))
            _require(sources == expanded["raw_source_fetch_required"], "invalid_context_source_bindings")
            evidence = {v.reference_id: v for v in parent.evidence}
            evidence.update(replacements)
            result = self._compile("compile_delta", parent, evidence=tuple(evidence.values()),
                                                  requested_reference_ids=requested)
            _cancel(cancelled)
            self._current(state)
            self._save(result.reconstructed_capsule, state)
            return result

    def retry(self, parent, *, generation, evidence, requested_reference_ids=()):
        """Send only changed/explicitly requested diagnostics through a delta."""
        with self._lock, public_replay_without_metadata():
            state = self._parent(parent, generation)
            updates = _references(evidence)
            requested = _ids(requested_reference_ids)
            retained = {v.reference_id: v for v in parent.evidence}
            for identity, value in updates.items():
                _require(not identity.startswith(PREFIX) or value == retained.get(identity),
                         "native_context_requires_expansion")
                _require(not value.required or value == state.required.get(identity), "required_context_changed")
                _require(identity not in state.required or value == state.required[identity], "required_context_changed")
            retained.update(updates)
            result = self._compile("compile_delta", parent, evidence=tuple(retained.values()),
                                                  requested_reference_ids=requested)
            self._current(state)
            self._save(result.reconstructed_capsule, state)
            return result

    def close(self):
        """Forget local parents without closing the caller-owned source session."""
        with self._lock:
            self._parents.clear()
