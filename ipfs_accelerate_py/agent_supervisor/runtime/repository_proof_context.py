"""Current local proof-cache evidence for existing repository contexts.

Only a live header admission and independently reviewed complete native key are
supported. Generic cache hits, serialized observations and Git-tree key aliases
are not admissions. Complete local conditions stay mandatory. Every consumer
boundary rechecks original source and native owners; checks are sequential,
not a writer lease. Existing native gate bookkeeping may write its owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import threading

from ..context.context_compiler import (
    CalibratedTokenEstimator, ContextCompiler, EvidenceValuePolicy,
    EvidenceExpansionRequest, ExclusionReason, _as_expansion,
)
from ..context.context_contracts import (
    ContextBudget, ContextBudgetResolution, ContextCapsule, ContextReference,
    canonical_context_json_bytes,
)
from ..proof.doctor_proof_cache import DoctorProofCacheGate, DoctorProofCacheKey
from ..proof.formal_verification_contracts import canonical_json
from .doctor_header_cache_bridge import HeaderProofCacheAdmission
from .repository_context_compiler import (
    MAX_CONTEXT_BYTES, MAX_PARENTS, MAX_REFERENCES, RepositoryContextAdapter,
    RepositoryContextError,
)
from .repository_proof_context_projection import project_repository_header_proof
from .repository_proof_source_binding import RepositoryProofSourceBinding
from .repository_source_index import NativeRepositoryIndex
from .repository_source_session import RepositorySourceSession

_PREFIXES = ("repository-proof:", "repository-proof-conditions:")


class RepositoryProofContextError(RepositoryContextError):
    """No checked proof context is returned; the obligation stays unresolved."""

    proof_status = "unresolved"
    execution_authority = False
    completion_authority = False


def _require(value, code):
    if not value:
        raise RepositoryProofContextError(code)


def _body(value):
    raw = canonical_json(value).encode("utf-8")
    _require(len(raw) <= MAX_CONTEXT_BYTES, "proof_context_binding_bytes")
    return raw


def _references(values):
    _require(type(values) in (tuple, list) and len(values) <= MAX_REFERENCES
             and all(type(value) is ContextReference for value in values),
             "invalid_proof_context_references")
    _require(len({value.reference_id for value in values}) == len(values),
             "duplicate_proof_context_reference")
    return tuple(values)


def _arguments(arguments):
    """Snapshot caller bytes and construct private compiler inputs."""
    required = _references(arguments.get("required_evidence"))
    optional = _references(arguments.get("optional_evidence", ()))
    public = dict(arguments, required_evidence=[v.to_dict() for v in required],
                  optional_evidence=[v.to_dict() for v in optional])
    raw = canonical_context_json_bytes(public)
    _require(len(raw) <= MAX_CONTEXT_BYTES, "proof_context_argument_bytes")
    private = json.loads(raw)
    for name in ("required_evidence", "optional_evidence"):
        private[name] = tuple(ContextReference.from_dict(v) for v in private[name])
    return raw, private


@dataclass(frozen=True)
class _Parent:
    capsule: ContextCapsule
    canonical_bytes: bytes
    source_bytes: bytes
    conditions: ContextReference
    proof: ContextReference


class RepositoryProofContextAdapter:
    """Wrap one existing adapter with one live, scoped header proof admission.

    Expected profiles and identities are independently reviewed host inputs,
    not authentication by themselves. Source correspondence and native proof
    stage checks remain independent. This creates no database, native owner,
    provider, solver, decoder or training job. The caller owns all supplied
    owners. Ordinary adapter methods do not substitute for this wrapper's
    proof-aware validation and rendering methods.
    """

    def __init__(self, *, context_adapter, gate, admission, source_binding,
                 scoped_analysis, expected_key, expected_receipt_id,
                 expected_cache_owner_path_sha256, generation,
                 expected_manifest_cid, expected_task_cid, expected_scope_id):
        _require(type(context_adapter) is RepositoryContextAdapter
                 and type(context_adapter.session) is RepositorySourceSession
                 and type(context_adapter.session._index) is NativeRepositoryIndex
                 and type(context_adapter.compiler) is ContextCompiler,
                 "original_repository_context_owners_required")
        _require(type(gate) is DoctorProofCacheGate
                 and type(admission) is HeaderProofCacheAdmission
                 and type(source_binding) is RepositoryProofSourceBinding
                 and type(expected_key) is DoctorProofCacheKey,
                 "original_native_proof_admission_required")
        _require(type(generation) is int and generation > 0
                 and type(expected_receipt_id) is str and expected_receipt_id,
                 "independent_proof_context_identities_required")
        self.context_adapter, self.gate = context_adapter, gate
        self.admission, self.source_binding = admission, source_binding
        self.scoped_analysis, self.expected_key = scoped_analysis, expected_key
        self._expected_receipt_id = expected_receipt_id
        self._expected_cache_owner_path_sha256 = expected_cache_owner_path_sha256
        self._source_arguments = dict(
            session=context_adapter.session, scoped_analysis=scoped_analysis,
            generation=generation, expected_manifest_cid=expected_manifest_cid,
            expected_task_cid=expected_task_cid, expected_scope_id=expected_scope_id)
        self._generation = generation
        self._key_bytes = _body(expected_key.to_dict())
        self._receipt_bytes = _body(admission.receipt.to_dict())
        self._source_argument_values = tuple(self._source_arguments.items())
        self._pins = (
            context_adapter, context_adapter.session, context_adapter.session._index,
            context_adapter.compiler, gate, admission, admission._guard,
            source_binding, scoped_analysis, expected_key)
        compiler = context_adapter.compiler
        _require(type(compiler.budget) is ContextBudget
                 and type(compiler.effective_budget) is ContextBudget
                 and type(compiler.budget_resolution) is ContextBudgetResolution
                 and type(compiler.estimator) is CalibratedTokenEstimator
                 and type(compiler.value_policy) is EvidenceValuePolicy,
                 "reviewed_context_compiler_policy_required")
        self._compiler_pins = (compiler.budget, compiler.effective_budget,
            compiler.budget_resolution, compiler.estimator, compiler.value_policy,
            compiler.estimator._tokenizer)
        self._compiler_bytes = self._compiler_state()
        self._identity_pins = (generation, expected_receipt_id,
                               expected_cache_owner_path_sha256)
        self._parents = {}
        self._mutex = threading.RLock()
        self._active = self._closed = False
        with self._operation():
            self._source_bytes = _body(self._check())

    def _compiler_state(self):
        compiler = self.context_adapter.compiler
        return _body(dict(budget=asdict(compiler.budget),
            effective_budget=asdict(compiler.effective_budget),
            budget_resolution=asdict(compiler.budget_resolution),
            effective_input_limit=compiler.effective_input_limit,
            provider_max_input_bytes=compiler.provider_max_input_bytes,
            value_policy=asdict(compiler.value_policy),
            chars_per_token=compiler.estimator._chars_per_token,
            calibration_samples=compiler.estimator._samples))

    @contextmanager
    def _operation(self):
        # No bridge lock is held while native owners, schedulers, CAS or caller
        # callbacks run. Concurrent and recursive entry refuse immediately.
        with self._mutex:
            _require(not self._active, "proof_context_reentry")
            _require(not self._closed, "proof_context_closed")
            self._active = True
        try:
            yield
        finally:
            with self._mutex:
                self._active = False

    def _original(self):
        adapter, session, index, compiler, gate, admission, guard, source, scoped, key = self._pins
        _require(self.context_adapter is adapter and adapter.session is session
                 and session._index is index and adapter.compiler is compiler
                 and self.gate is gate and self.admission is admission
                 and admission._guard is guard and self.source_binding is source
                 and self.scoped_analysis is scoped and self.expected_key is key
                 and type(adapter) is RepositoryContextAdapter
                 and type(session) is RepositorySourceSession
                 and type(index) is NativeRepositoryIndex
                 and type(compiler) is ContextCompiler
                 and type(gate) is DoctorProofCacheGate
                 and type(admission) is HeaderProofCacheAdmission
                 and type(source) is RepositoryProofSourceBinding,
                 "proof_context_owner_substituted")
        _require(all(value is original for value, original in zip(
            (compiler.budget, compiler.effective_budget, compiler.budget_resolution,
             compiler.estimator, compiler.value_policy, compiler.estimator._tokenizer),
            self._compiler_pins)) and self._compiler_state() == self._compiler_bytes,
            "proof_context_compiler_policy_changed")
        _require((self._generation, self._expected_receipt_id,
                  self._expected_cache_owner_path_sha256) == self._identity_pins
                 and tuple(self._source_arguments.items()) == self._source_argument_values
                 and _body(key.to_dict()) == self._key_bytes
                 and _body(admission.key.to_dict()) == self._key_bytes
                 and _body(admission.receipt.to_dict()) == self._receipt_bytes,
                 "proof_context_binding_changed")
        return adapter, admission, source

    def _check(self):
        adapter, admission, source = self._original()
        try:
            source.assert_bound(**self._source_arguments)
            admission.assert_bound(gate=self.gate, scoped_analysis=self.scoped_analysis,
                expected_key_id=self.expected_key.key_id,
                expected_receipt_id=self._expected_receipt_id,
                expected_owner_path_sha256=self._expected_cache_owner_path_sha256)
            source.assert_current()
            admission.assert_current()
            result = admission.revalidate_for_render()
            _require(_body(result.key.to_dict()) == self._key_bytes
                     and result.receipt is not None
                     and _body(result.receipt.to_dict()) == self._receipt_bytes,
                     "proof_context_native_receipt_changed")
            source.assert_current()
            observed = source.observation()
            _require(observed["generation"] == self._generation
                     and observed["session_id"] == adapter.session.session_id
                     and observed["session_repository_id"] == adapter.session._index.repository_id,
                     "proof_context_source_changed")
            if hasattr(self, "_source_bytes"):
                _require(_body(observed) == self._source_bytes, "proof_context_source_changed")
            # The final native source observation can run SQL/CAS/resource
            # callbacks. Close proof eligibility afterwards, then use a full
            # filesystem fence with no new native callback or owner lookup.
            result = admission.revalidate_for_render()
            _require(_body(result.key.to_dict()) == self._key_bytes
                     and result.receipt is not None
                     and _body(result.receipt.to_dict()) == self._receipt_bytes,
                     "proof_context_native_receipt_changed")
            source.assert_bound(**self._source_arguments)
            source.assert_source_current()
            self._original()
            return observed
        except RepositoryProofContextError:
            raise
        except Exception as exc:
            raise RepositoryProofContextError("current_local_proof_context_refused") from exc

    def _project(self, observed):
        conditions, proof = project_repository_header_proof(
            key=self.expected_key, receipt=self.admission.receipt,
            source_binding=observed)
        _require(type(conditions) is ContextReference and conditions.required
                 and type(proof) is ContextReference and not proof.required,
                 "invalid_local_proof_context_projection")
        return conditions, proof

    def _generation_check(self, generation):
        _require(type(generation) is int and generation == self._generation,
                 "stale_proof_context_generation")

    def _save(self, capsule, observed, conditions, proof, required_bytes):
        _require(type(capsule) is ContextCapsule, "invalid_proof_context_parent")
        raw = capsule.canonical_bytes()
        _require(len(raw) <= MAX_CONTEXT_BYTES and
                 (capsule.capsule_id in self._parents or len(self._parents) < MAX_PARENTS),
                 "proof_context_parent_limit")
        refs = {v.reference_id: v for v in (*capsule.evidence, *capsule.expansion_references)}
        deferred = refs.get(proof.reference_id)
        if deferred != proof and deferred is not None:
            reason = deferred.metadata.get("selection_exclusion_reason")
            _require(reason in {value.value for value in ExclusionReason}
                     and deferred == _as_expansion(proof, exclusion_reason=reason),
                     "local_proof_handle_changed")
        _require(refs.get(conditions.reference_id) == conditions
                 and conditions in capsule.evidence
                 and deferred is not None,
                 "local_proof_conditions_changed")
        emitted = {v.reference_id: v.canonical_bytes() for v in capsule.evidence if v.required}
        _require(emitted == required_bytes, "required_proof_context_changed")
        binding = capsule.authority["repository_context"]
        _require(binding["generation"] == observed["generation"]
                 and binding["source_tree_sha256"] == observed["source_tree_sha256"]
                 and binding["semantic_state_cid"] == observed["semantic_state_cid"]
                 and capsule.repository_id == observed["session_repository_id"]
                 and capsule.tree_id == "sha256:" + observed["source_tree_sha256"],
                 "proof_context_parent_source_changed")
        self._parents[capsule.capsule_id] = _Parent(capsule, raw, _body(observed), conditions, proof)

    def _parent(self, parent, generation):
        self._generation_check(generation)
        _require(type(parent) is ContextCapsule, "invalid_proof_context_parent")
        state = self._parents.get(parent.capsule_id)
        _require(state is not None and parent is state.capsule
                 and parent.canonical_bytes() == state.canonical_bytes,
                 "unknown_or_changed_proof_context_parent")
        return state

    def compile(self, **arguments):
        """Add complete mandatory local conditions and an optional proof digest."""
        with self._operation():
            input_bytes, private = _arguments(arguments)
            self._generation_check(arguments.get("generation"))
            observed = self._check()
            conditions, proof = self._project(observed)
            conditions_bytes, proof_bytes = conditions.canonical_bytes(), proof.canonical_bytes()
            required = _references(private.get("required_evidence"))
            optional = _references(private.get("optional_evidence", ()))
            _require(not any(v.reference_id.startswith(_PREFIXES) for v in (*required, *optional)),
                     "reserved_local_proof_context_reference")
            ids, coverage = private.get("required_reference_ids"), private.get("required_coverage_ids")
            _require(type(ids) in (tuple, list) and type(coverage) in (tuple, list),
                     "invalid_proof_context_requirement_ids")
            # Validate the caller's complete population before adding ours;
            # the wrapper must not silently repair a missing declaration.
            _require(tuple(sorted(ids)) == tuple(sorted(v.reference_id for v in required))
                     and tuple(sorted(coverage)) == tuple(sorted({c for v in required for c in v.coverage_ids})),
                     "incomplete_proof_context_requirements")
            required_bytes = {v.reference_id: v.canonical_bytes() for v in (*required, conditions)}
            expected_core = canonical_context_json_bytes({name: private[name] for name in
                ("goal", "authority", "scope", "acceptance", "objective_id", "objective_revision",
                 "policy_id", "policy_revision")})
            options = dict(private,
                required_evidence=(*required, conditions),
                optional_evidence=(*optional, proof),
                required_reference_ids=(*ids, conditions.reference_id),
                required_coverage_ids=(*coverage, *conditions.coverage_ids))
            result = self.context_adapter.compile(**options)
            self._check()
            self.context_adapter.validate_current(result.capsule, generation=self._generation)
            self._check()
            _require(_arguments(arguments)[0] == input_bytes
                     and conditions.canonical_bytes() == conditions_bytes
                     and proof.canonical_bytes() == proof_bytes,
                     "proof_context_caller_inputs_changed")
            _require(canonical_context_json_bytes(dict(
                goal=result.capsule.goal, authority=result.capsule.authority["task"],
                scope=result.capsule.scope, acceptance=result.capsule.acceptance,
                objective_id=result.capsule.objective_id,
                objective_revision=result.capsule.objective_revision,
                policy_id=result.capsule.policy_id, policy_revision=result.capsule.policy_revision))
                == expected_core, "proof_context_core_changed")
            self._save(result.capsule, observed, conditions, proof, required_bytes)
            return result

    def validate_current(self, parent, *, generation):
        """Recheck source, live proof owners and an issued parent at a consumer boundary."""
        with self._operation():
            state = self._parent(parent, generation)
            self._check()
            self.context_adapter.validate_current(parent, generation=generation)
            self._check()
            _require(parent.canonical_bytes() == state.canonical_bytes,
                     "proof_context_parent_changed")

    def expand_for_question(self, parent, request, *, generation, max_bytes=65536, cancelled=None):
        """Use existing question-bound source expansion with proof fences."""
        with self._operation():
            state = self._parent(parent, generation)
            _require(type(request) is EvidenceExpansionRequest, "invalid_proof_context_expansion")
            request_bytes = canonical_context_json_bytes(asdict(request))
            private_request = EvidenceExpansionRequest(**json.loads(request_bytes))
            condition_bytes, proof_bytes = state.conditions.canonical_bytes(), state.proof.canonical_bytes()
            required_bytes = {v.reference_id: v.canonical_bytes() for v in parent.evidence if v.required}
            observed = self._check()
            result = self.context_adapter.expand_for_question(
                parent, private_request, generation=generation, max_bytes=max_bytes, cancelled=cancelled)
            self._check()
            self.context_adapter.validate_current(result.reconstructed_capsule, generation=generation)
            self._check()
            _require(parent.canonical_bytes() == state.canonical_bytes
                     and canonical_context_json_bytes(asdict(request)) == request_bytes
                     and state.conditions.canonical_bytes() == condition_bytes
                     and state.proof.canonical_bytes() == proof_bytes,
                     "proof_context_expansion_inputs_changed")
            self._save(result.reconstructed_capsule, observed, state.conditions, state.proof, required_bytes)
            return result

    def retry(self, parent, *, generation, evidence, requested_reference_ids=()):
        """Preserve required conditions while applying ordinary context deltas."""
        with self._operation():
            state = self._parent(parent, generation)
            caller_updates = _references(evidence)
            updates_bytes = canonical_context_json_bytes([v.to_dict() for v in caller_updates])
            requested_bytes = canonical_context_json_bytes(requested_reference_ids)
            requested = tuple(json.loads(requested_bytes))
            condition_bytes, proof_bytes = state.conditions.canonical_bytes(), state.proof.canonical_bytes()
            required_bytes = {v.reference_id: v.canonical_bytes() for v in parent.evidence if v.required}
            observed = self._check()
            updates = tuple(ContextReference.from_dict(v) for v in json.loads(updates_bytes))
            _require(not any(v.reference_id.startswith(_PREFIXES) for v in updates),
                     "local_proof_projection_not_caller_replaceable")
            _require(type(requested_reference_ids) in (tuple, list),
                     "invalid_proof_context_requirement_ids")
            if state.proof.reference_id in requested:
                updates = (*updates, state.proof)
            result = self.context_adapter.retry(parent, generation=generation,
                evidence=updates, requested_reference_ids=requested)
            self._check()
            self.context_adapter.validate_current(result.reconstructed_capsule, generation=generation)
            self._check()
            _require(parent.canonical_bytes() == state.canonical_bytes
                     and canonical_context_json_bytes([v.to_dict() for v in _references(evidence)]) == updates_bytes
                     and canonical_context_json_bytes(requested_reference_ids) == requested_bytes
                     and state.conditions.canonical_bytes() == condition_bytes
                     and state.proof.canonical_bytes() == proof_bytes,
                     "proof_context_retry_inputs_changed")
            self._save(result.reconstructed_capsule, observed, state.conditions, state.proof, required_bytes)
            return result

    def render(self, parent, *, generation):
        """Return bounded canonical provider-input bytes after closing proof checks.

        Framing, provider usage and complete session cost remain the caller's
        accounting responsibility. Repeat validation after asynchronous work.
        """
        with self._operation():
            state = self._parent(parent, generation)
            self._check()
            self.context_adapter.validate_current(parent, generation=generation)
            raw = canonical_context_json_bytes(parent.provider_input_payload)
            _require(len(raw) <= MAX_CONTEXT_BYTES, "proof_context_render_bytes")
            self._check()
            self.context_adapter.validate_current(parent, generation=generation)
            self._check()
            _require(parent.canonical_bytes() == state.canonical_bytes,
                     "proof_context_parent_changed")
            return raw

    def close(self):
        """Retire only this wrapper's parents; supplied owners remain caller-owned."""
        with self._operation():
            self._parents.clear()
            self._closed = True


__all__ = ["RepositoryProofContextAdapter", "RepositoryProofContextError"]
