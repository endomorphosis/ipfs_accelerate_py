"""Closed no-effect refusal at the new finite advisory launch boundary.

The boundary creates a private token before Popen. The native lifecycle must
then prove a fresh empty tree and persist a failed saga before a separate token
can make the control transaction terminal. These are cooperative process-local
capabilities, not authenticated process-origin or arbitrary-writer guarantees.
"""
from __future__ import annotations

import json

from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes
from .control_plane import BackendConflictError

_BOUNDARY_SEAL = object()
_VERIFIED_SEAL = object()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _wire(value):
    raw = canonical_dag_json_bytes(value)
    _need(len(raw) <= 16 * 1024, "bounded before-Popen refusal material required")
    return raw


class BeforePopenRefusal(BackendConflictError):
    """Private observation of the exact unlaunched advisory runtime."""

    def __init__(self, seal, *, message, binding):
        _need(seal is _BOUNDARY_SEAL, "before-Popen refusal needs its native boundary")
        super().__init__(message)
        self._seal = seal
        self._binding = _wire(binding)


class VerifiedBeforePopenRefusal(BackendConflictError):
    """Native saga has persisted failure after proving no process effect."""

    def __init__(self, seal, *, message, proof):
        _need(seal is _VERIFIED_SEAL, "verified refusal needs its native lifecycle")
        super().__init__(message)
        self._seal = seal
        self._proof = _wire(proof)


def _profile_binding(profile):
    return {key: getattr(profile, key) for key in (
        "profile_id", "target_id", "run_id", "configuration_root", "repository_root",
        "state_root", "run_root")}


def refuse_finite_advisory_before_popen(runtime, error):
    """Called only when the actual final fence raises before Popen is called."""
    from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ..runtime.finite_repository_execution import (
        ADVISORY_PROFILE, ADVISORY_SCHEMA, FrozenFiniteRepositoryExecutionScope,
    )
    _need(type(runtime) is AdmittedBenchmarkRuntime and isinstance(error, Exception),
          "exact finite advisory runtime and original refusal required")
    scope = runtime.finite_execution_scope
    _need(type(scope) is FrozenFiniteRepositoryExecutionScope and scope._runtime is runtime
          and not scope._spawned and not runtime._children,
          "only the exact native unlaunched finite scope can refuse without Popen")
    scope._active(allow_cancelled=True)
    envelope = scope.to_dict()
    payload = envelope["payload"]
    _need(payload["schema"] == ADVISORY_SCHEMA and payload["profile"] == ADVISORY_PROFILE
          and scope._advisory_closure is not None
          and runtime.manifest["finite_execution_scope"] == envelope
          and runtime.manifest["run_id"] == runtime.profile.run_id
          and runtime.manifest_id == runtime.profile.configuration_root
          and runtime.server is scope._server and runtime.source is scope._source
          and runtime.lease.resource_id == runtime.run_id
          and runtime.lease.owner_session_id == runtime.local_profile.identity_did,
          "no-effect refusal must match the signed native profile and live lease binding")
    protected = runtime.coordinator.protect_write(runtime.lease,
        expected_fencing_token=runtime.lease.fencing_token,
        expected_fence_epoch=runtime.lease.fence_epoch)
    _need(protected.resource_id == runtime.run_id
          and protected.owner_session_id == runtime.local_profile.identity_did,
          "before-Popen refusal run lease changed owner")
    binding = {**_profile_binding(runtime.profile), "lease_id": runtime.lease.lease_id,
        "fencing_epoch": runtime.lease.fence_epoch,
        "finite_scope_lease_id": scope.parent_lease.lease_id,
        "production_activated": False, "authenticated_process_origin": False,
        "atomicity_attested": False}
    return BeforePopenRefusal(_BOUNDARY_SEAL,
        message="native process refused before Popen: " + str(error)[:4096], binding=binding)


def refuse_finite_proof_query_before_popen(runtime, error):
    """Called only when the actual final fence raises before Popen is called."""
    from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ..runtime.finite_repository_execution import FrozenFiniteRepositoryExecutionScope
    from ..runtime.finite_proof_query_execution import (
        PROFILE, EXECUTION_SCHEMA, FrozenFiniteProofQueryExecutionClosure,
    )
    _need(type(runtime) is AdmittedBenchmarkRuntime and isinstance(error, Exception),
          "exact finite proof-query runtime and original refusal required")
    scope = runtime.finite_execution_scope
    _need(type(scope) is FrozenFiniteRepositoryExecutionScope and scope._runtime is runtime
          and not scope._spawned and not runtime._children,
          "only the exact native unlaunched finite scope can refuse without Popen")
    scope._active(allow_cancelled=True)
    envelope = scope.to_dict()
    payload = envelope["payload"]
    _need(payload["schema"] == EXECUTION_SCHEMA and payload["profile"] == PROFILE
          and type(scope._proof_query_closure) is FrozenFiniteProofQueryExecutionClosure
          and scope._advisory_closure is None
          and runtime.manifest["finite_execution_scope"] == envelope
          and runtime.manifest["run_id"] == runtime.profile.run_id
          and runtime.manifest_id == runtime.profile.configuration_root
          and runtime.server is scope._server and runtime.source is scope._source
          and runtime.lease.resource_id == runtime.run_id
          and runtime.lease.owner_session_id == runtime.local_profile.identity_did,
          "no-effect refusal must match the signed native profile and live lease binding")
    protected = runtime.coordinator.protect_write(runtime.lease,
        expected_fencing_token=runtime.lease.fencing_token,
        expected_fence_epoch=runtime.lease.fence_epoch)
    _need(protected.resource_id == runtime.run_id
          and protected.owner_session_id == runtime.local_profile.identity_did,
          "before-Popen refusal run lease changed owner")
    binding = {**_profile_binding(runtime.profile), "lease_id": runtime.lease.lease_id,
        "fencing_epoch": runtime.lease.fence_epoch,
        "finite_scope_lease_id": scope.parent_lease.lease_id,
        "production_activated": False, "authenticated_process_origin": False,
        "atomicity_attested": False}
    return BeforePopenRefusal(_BOUNDARY_SEAL,
        message="native process refused before Popen: " + str(error)[:4096], binding=binding)


def _boundary_matches(error, profile, intent):
    if type(error) is not BeforePopenRefusal or getattr(error, "_seal", None) is not _BOUNDARY_SEAL:
        return False
    try:
        binding = json.loads(error._binding)
        return (_wire(binding) == error._binding
            and set(binding) == set(_profile_binding(profile)) | {
                "lease_id", "fencing_epoch", "finite_scope_lease_id", "production_activated",
                "authenticated_process_origin", "atomicity_attested"}
            and all(binding[key] == value for key, value in _profile_binding(profile).items())
            and binding["lease_id"] == intent.lease_id
            and type(binding["fencing_epoch"]) is int and binding["fencing_epoch"] == intent.fencing_epoch
            and type(binding["finite_scope_lease_id"]) is str and bool(binding["finite_scope_lease_id"])
            and all(binding[key] is False for key in (
                "production_activated", "authenticated_process_origin", "atomicity_attested")))
    except (ValueError, TypeError, AttributeError, KeyError):
        return False


def verify_lifecycle_before_popen_refusal(error, *, profile, state, empty_tree):
    """Issue the second token after the native failed journal was persisted."""
    from .lifecycle_orchestrator import LifecycleAction, LifecycleSagaPhase
    _need(_boundary_matches(error, profile, state.intent)
          and state.intent.action is LifecycleAction.START
          and state.phase is LifecycleSagaPhase.FAILED
          and state.failure_code == "refused_before_popen_without_process_effect"
          and state.old_tree is None and state.new_tree is None and not state.old_tree_fenced
          and not state.observed_effects and not state.compensation and state.receipt is None
          and empty_tree.profile_id == profile.profile_id and empty_tree.run_id == profile.run_id
          and not empty_tree.members,
          "native failed START and fresh empty process tree required for no-effect refusal")
    proof = {"schema": "native-before-popen-no-effect-refusal@1",
        "boundary": json.loads(error._binding), "request_id": state.intent.request_id,
        "operation": "start", "repository_id": state.intent.repository_id,
        "tree_id": state.intent.tree_id, "objective_id": state.intent.objective_id,
        "objective_revision": state.intent.objective_revision, "policy_id": state.intent.policy_id,
        "policy_revision": state.intent.policy_revision, "caller": state.intent.caller,
        "idempotency_key": state.intent.idempotency_key, "lease_id": state.intent.lease_id,
        "fencing_epoch": state.intent.fencing_epoch, "target_id": state.intent.target_id,
        "transition_id": state.intent.transition_id, "journal_revision": state.revision,
        "saga_phase": state.phase.value, "empty_tree": empty_tree.to_dict(),
        "applied_effect_ids": []}
    return VerifiedBeforePopenRefusal(_VERIFIED_SEAL, message=str(error), proof=proof)


def verified_refusal_for_request(error, request):
    """Only the exact second-stage token can reject a control mutation."""
    if type(error) is not VerifiedBeforePopenRefusal or getattr(error, "_seal", None) is not _VERIFIED_SEAL:
        return False
    try:
        proof = json.loads(error._proof)
        fields = ("request_id", "repository_id", "tree_id", "objective_id", "objective_revision",
                  "policy_id", "policy_revision", "caller", "idempotency_key", "lease_id", "fencing_epoch")
        return (_wire(proof) == error._proof and proof["schema"] == "native-before-popen-no-effect-refusal@1"
            and request.operation.value == proof["operation"] == "start"
            and all(proof[key] == getattr(request, key) for key in fields)
            and proof["saga_phase"] == "failed" and proof["applied_effect_ids"] == []
            and proof["empty_tree"]["members"] == [])
    except (ValueError, TypeError, AttributeError, KeyError):
        return False


__all__ = ["BeforePopenRefusal", "VerifiedBeforePopenRefusal",
           "refuse_finite_advisory_before_popen", "refuse_finite_proof_query_before_popen",
           "verify_lifecycle_before_popen_refusal",
           "verified_refusal_for_request"]
