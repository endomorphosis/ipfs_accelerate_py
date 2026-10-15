"""Actual current producer prepare-only checks with explicit authored owners.

Preparation tokenizes and verifies the complete native context without model
dispatch, worktree effects, attempt reservations or acceptance promotion.
Source/capsule/tokenizer operations are real; launch and model compatibility
remain authored fixtures and grant no benchmark, model or Lean qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.semantic_state import harness as harness_module
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import ModelRoute
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessCancelled,
    HarnessLoopError,
    HarnessRootConflict,
    PreparedModelContext,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from test.api.semantic_state.test_harness import MemoryDurablePort, _routing
from test.api.semantic_state.test_owner_context_materializer import (  # noqa: F401 -- register actual producer fixtures
    SOURCE_NAMES,
    current,
    decoded_capsule,
    dispatch,
)


def state_of(case) -> dict[str, Any]:
    """Actual mutable state that preparation must not publish or reserve."""
    return copy.deepcopy({
        "objects": case.durable._objects,
        "roots": case.durable._roots,
        "put_order": case.durable.put_order,
        "cas_calls": case.durable.cas_calls,
        "terminal_attempts": case.harness._terminal_attempts,
        "provider_calls": case.provider.calls,
    })


@pytest.fixture
def guarded(dispatch, monkeypatch):  # noqa: F811 -- imported native fixture
    effects = []

    def forbidden(*args, **kwargs):
        effects.append((args, kwargs))
        raise AssertionError("prepare-only crossed a dispatch/persistence/worktree boundary")

    class NoAttemptOperations:
        def __getattr__(self, name):
            return forbidden

    # Leave source, capsule, tokenizer and root reads real. Only forbidden
    # effect boundaries are tripwires; a swallowed attempt is also recorded.
    dispatch.harness.terminal_attempt_index = NoAttemptOperations()
    monkeypatch.setattr(dispatch.durable, "put", forbidden)
    monkeypatch.setattr(dispatch.durable, "compare_and_swap_root", forbidden)
    monkeypatch.setattr(harness_module, "create_isolated_worktree", forbidden)
    monkeypatch.setattr(harness_module, "invoke_model", forbidden)
    return dispatch, effects


def refused(guarded, request=None, *, exception=HarnessLoopError):
    case, effects = guarded
    before = state_of(case)
    with pytest.raises(exception) as error:
        case.harness.prepare_model_context(request or case.request)
    assert state_of(case) == before
    assert effects == []
    return error.value


def test_prepare_contains_native_sources_capsule_and_actual_complete_count(guarded):
    case, effects = guarded
    before = state_of(case)
    prepared = case.harness.prepare_model_context(case.request)
    assert isinstance(prepared, PreparedModelContext)
    assert prepared.provider_id == case.provider.provider_id
    assert prepared.route == case.request.routing_decision.route
    assert prepared.mode == case.harness.policy.mode
    assert prepared.context_pack_cid == cid_for_payload(case.request.context_pack.to_dict())
    assert prepared.base_tree == case.request.base_tree
    assert prepared.materializer_revision_cid == case.port.revision_cid
    assert prepared.observed_root == case.root
    assert prepared.input_tokens == case.current.tokenizer.count(prepared.prompt) > 0
    packet = json.loads(prepared.prompt)
    assert packet["objective"] == case.request.objective
    assert packet["scope"] == case.request.scope.to_dict()
    material = packet["materialized_context"]
    for kind, path in SOURCE_NAMES.items():
        assert material["required_sources"][kind]["text"].encode() == case.current.sources[path]
        assert material["required_sources"][kind]["source_cid"] == case.current.source_cids[kind]
    view = case.current.provider.open_verified_view(case.current.bundle.root.root_cid, case.current.get_block)
    capsule = view.capsule(case.current.dependency_symbol)
    assert decoded_capsule(material["dependency_capsules"][0]) == capsule.to_dict()
    compact = prepared.to_dict()
    assert compact["schema"] == "ipfs-accelerate.prepared-model-context@1"
    assert compact["prompt_sha256"] == hashlib.sha256(prepared.prompt.encode()).hexdigest()
    assert compact["prompt_bytes"] == len(prepared.prompt.encode())
    assert compact["input_tokens"] == prepared.input_tokens
    assert compact["observed_root"] == case.root.to_dict()
    assert "prompt" not in compact and "café" not in json.dumps(compact, ensure_ascii=False)
    for flag in (
        "execution_authority", "semantic_acceptance_authority",
        "tokenizer_compatibility_authority", "completion_authority",
        "provider_invoked", "attempt_reserved",
    ):
        assert compact[flag] is False
    assert prepared.to_dict(include_prompt=True) == {**compact, "prompt": prepared.prompt}
    assert state_of(case) == before
    assert effects == []


def test_each_preparation_freshly_reads_sources_and_does_not_publish_history(guarded):
    case, effects = guarded
    reads = []
    original = case.current.read_source_blob

    def captured(cid):
        reads.append(cid)
        return original(cid)

    case.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=captured)
    before = state_of(case)
    first = case.harness.prepare_model_context(case.request)
    first_reads = len(reads)
    second = case.harness.prepare_model_context(case.request)
    assert first.prompt == second.prompt
    assert first.input_tokens == second.input_tokens
    assert first_reads >= 12
    assert len(reads) >= first_reads + 12
    assert state_of(case) == before
    assert effects == []


def test_corrupt_current_source_refuses_preparation_without_effects(guarded):
    case, _ = guarded
    cid = case.current.source_cids["target_source"]
    case.current.raw[cid] += b"# differs from the admitted producer blob\n"
    refused(guarded)


def test_revoked_owner_refuses_preparation_without_effects(guarded):
    case, _ = guarded
    case.authority.active = False
    refused(guarded)


def test_changed_actual_tokenizer_asset_refuses_preparation_without_effects(guarded):
    case, _ = guarded
    asset = case.current.artifact.parent / case.current.descriptor["asset_path"]
    asset.write_bytes(asset.read_bytes() + b" ")
    refused(guarded)


def test_missing_materializer_does_not_fall_back_to_objective_or_estimates(guarded):
    case, _ = guarded
    case.harness.context_materializer = None
    refused(guarded)


@pytest.mark.parametrize("route", [ModelRoute.DETERMINISTIC_ONLY.value, ModelRoute.HUMAN_REVIEW_REQUIRED.value])
def test_non_model_route_cannot_prepare_a_provider_prompt(guarded, route):
    case, _ = guarded
    request = replace(case.request, routing_decision=_routing(route))
    refused(guarded, request)


def test_actual_measured_one_token_overflow_refuses_preparation(guarded):
    case, _ = guarded
    prepared = case.harness.prepare_model_context(case.request)
    budget = ContextBudget()
    window = prepared.input_tokens + budget.reserved_output_tokens + budget.reserved_tool_tokens - 1
    assert window < case.provider.capability_spec.max_context_tokens
    object.__setattr__(case.provider, "spec", replace(case.provider.capability_spec, max_context_tokens=window))
    refused(guarded)


def test_unavailable_provider_refuses_preparation_without_reservation(guarded):
    case, _ = guarded
    object.__setattr__(case.provider, "spec", replace(case.provider.capability_spec, available=False))
    refused(guarded)


def test_changed_harness_policy_refuses_preparation_without_effects(guarded):
    case, _ = guarded
    case.harness.policy = replace(case.harness.policy, require_full_suite_oracle=True)
    refused(guarded)


def test_cancelled_preparation_never_reads_or_dispatches_context(guarded):
    case, _ = guarded
    token = CancellationToken("explicit-authored-prepare-cancellation")
    assert token.cancel(cancellation_id=token.cancellation_id, reason="authored cancellation")
    request = replace(case.request, cancellation=token)
    admits = case.authority.admit_calls
    refused(guarded, request, exception=HarnessCancelled)
    assert case.authority.admit_calls == admits


def test_stale_expected_durable_root_refuses_before_source_admission(guarded):
    case, _ = guarded
    case.durable._roots[case.request.repository_id] = replace(case.root, generation=case.root.generation + 1)
    admits = case.authority.admit_calls
    refused(guarded)
    assert case.authority.admit_calls == admits


def test_missing_explicit_tree_does_not_create_worktree_to_discover_it(guarded):
    case, _ = guarded
    request = replace(case.request, base_tree=None)
    refused(guarded, request)


def test_root_change_during_real_source_reads_is_not_a_preparation_admission(guarded):
    case, effects = guarded
    original = case.current.read_source_blob
    external_root = replace(case.root, generation=case.root.generation + 1)
    changed = False

    def read_with_external_root_change(cid):
        nonlocal changed
        raw = original(cid)
        if not changed:
            changed = True
            # Explicit simulated external owner update; the preparation must
            # observe it without publishing any additional state itself.
            case.durable._roots[case.request.repository_id] = external_root
        return raw

    case.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=read_with_external_root_change)
    before = state_of(case)
    with pytest.raises(HarnessRootConflict):
        case.harness.prepare_model_context(case.request)
    assert changed
    expected = copy.deepcopy(before)
    expected["roots"][case.request.repository_id] = external_root
    assert state_of(case) == expected
    assert effects == []


def test_provider_configuration_change_during_actual_tokenization_refuses(guarded, monkeypatch):
    case, _ = guarded
    actual = case.current.tokenizer.count
    window = case.provider.capability_spec.max_context_tokens

    def counted_with_changed_configuration(self, prompt):
        measured = actual(prompt)
        # Retain real measured counts and decrease the declared window while
        # still leaving this small prompt within it. Closing configuration
        # validation must refuse the changed declaration independently.
        object.__setattr__(case.provider, "spec", replace(
            case.provider.capability_spec, max_context_tokens=window - 1,
        ))
        return measured

    monkeypatch.setattr(type(case.current.tokenizer), "count", counted_with_changed_configuration)
    error = refused(guarded)
    assert error.reason_code == "context_configuration_changed"


def test_cancellation_during_actual_source_read_refuses_at_closing_gate(guarded):
    case, _ = guarded
    token = CancellationToken("explicit-authored-prepare-closing-cancellation")
    request = replace(case.request, cancellation=token)
    original = case.current.read_source_blob
    reads = []

    def read_and_cancel(cid):
        reads.append(cid)
        raw = original(cid)
        assert token.cancel(cancellation_id=token.cancellation_id, reason="cancel during source observation")
        return raw

    case.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=read_and_cancel)
    refused(guarded, request, exception=HarnessCancelled)
    assert reads


def test_oversized_owner_source_reader_is_refused_without_truncation(guarded):
    case, _ = guarded
    cid = case.current.source_cids["target_source"]
    original = case.current.raw[cid]
    # This is an explicit malformed owner-reader fault, not an admitted large
    # source. The existing byte ceiling must refuse before hashing or slicing.
    case.current.raw[cid] = original + b"x" * (ContextBudget().max_text_bytes + 1)
    refused(guarded)
    assert case.current.raw[cid] == original + b"x" * (ContextBudget().max_text_bytes + 1)


def test_opening_root_read_cannot_swap_durable_owner_even_with_identical_root(guarded, monkeypatch):
    case, effects = guarded
    other_owner = MemoryDurablePort()
    other_owner._roots = dict(case.durable._roots)
    original_read = case.durable.read_root
    other_read = other_owner.read_root
    reads = []

    def opening_read(repository_id):
        reads.append("original-owner")
        observed = original_read(repository_id)
        case.harness.durable = other_owner
        return observed

    def read_other(repository_id):
        reads.append("replacement-owner")
        return other_read(repository_id)

    monkeypatch.setattr(case.durable, "read_root", opening_read)
    monkeypatch.setattr(other_owner, "read_root", read_other)
    before = state_of(case)
    admits = case.authority.admit_calls
    with pytest.raises(HarnessLoopError):
        case.harness.prepare_model_context(case.request)
    assert reads == ["original-owner"]
    assert case.authority.admit_calls == admits
    assert state_of(case) == before
    assert other_owner._objects == {} and other_owner.put_order == [] and other_owner.cas_calls == []
    assert effects == []


def test_opening_root_read_cannot_swap_materializer_before_configuration_capture(guarded, monkeypatch):
    case, effects = guarded
    original_read = case.durable.read_root
    calls = []

    class ReplacementMaterializer:
        # Equal declared revision cannot authorize a different callback object.
        revision_cid = case.port.revision_cid

        def materialize(self, **bindings):
            calls.append("materialize")
            return case.port.materialize(**bindings)

        def verify(self, **bindings):
            calls.append("verify")
            return case.port.verify(**bindings)

        def count_tokens(self, *args, **bindings):
            calls.append("count")
            return case.port.count_tokens(*args, **bindings)

    replacement = ReplacementMaterializer()

    def opening_read(repository_id):
        observed = original_read(repository_id)
        case.harness.context_materializer = replacement
        return observed

    monkeypatch.setattr(case.durable, "read_root", opening_read)
    before = state_of(case)
    admits = case.authority.admit_calls
    with pytest.raises(HarnessLoopError):
        case.harness.prepare_model_context(case.request)
    assert calls == [] and case.authority.admit_calls == admits
    assert state_of(case) == before
    assert effects == []


def test_cancellation_inside_opening_root_read_prevents_all_materializer_callbacks(guarded, monkeypatch):
    case, effects = guarded
    original_read = case.durable.read_root
    token = CancellationToken("explicit-opening-read-cancellation")
    request = replace(case.request, cancellation=token)
    reads = []
    materializer_calls = []

    def forbidden_materializer(*args, **kwargs):
        materializer_calls.append((args, kwargs))
        raise AssertionError("opening-read cancellation must precede every materializer callback")

    for operation in ("materialize", "verify", "count_tokens"):
        monkeypatch.setattr(type(case.port), operation, forbidden_materializer)

    def opening_read(repository_id):
        reads.append(repository_id)
        observed = original_read(repository_id)
        assert token.cancel(cancellation_id=token.cancellation_id, reason="cancel during opening root observation")
        return observed

    monkeypatch.setattr(case.durable, "read_root", opening_read)
    before = state_of(case)
    admits, verifies = case.authority.admit_calls, case.authority.verify_calls
    with pytest.raises(HarnessCancelled):
        case.harness.prepare_model_context(request)
    assert reads == [request.repository_id]
    assert materializer_calls == []
    assert case.authority.admit_calls == admits and case.authority.verify_calls == verifies
    assert state_of(case) == before
    assert effects == []


@pytest.mark.parametrize("fault", ["materializer_identity", "cancellation"])
def test_closing_capability_getter_cannot_hide_replacement_or_cancellation(guarded, monkeypatch, fault):
    case, effects = guarded
    token = CancellationToken("explicit-closing-capability-cancellation")
    request = replace(case.request, cancellation=token)
    actual_count = case.port.count_tokens
    actual_root = case.durable.read_root
    spec = case.provider.capability_spec
    state = {"counted": False, "window_checked": False, "closing_root_seen": False, "fired": False}
    events = []

    class ReplacementMaterializer:
        revision_cid = case.port.revision_cid

        def materialize(self, **bindings):
            raise AssertionError("replacement is not authorized to read sources")

        def verify(self, **bindings):
            raise AssertionError("replacement is not authorized to verify")

        def count_tokens(self, *args, **bindings):
            raise AssertionError("replacement is not authorized to tokenize")

    replacement = ReplacementMaterializer()

    class ObservedCapability:
        @property
        def max_context_tokens(self):
            if state["counted"]:
                state["window_checked"] = True
                events.append("measured_context_window")
            return spec.max_context_tokens

        def supports(self, capability):
            return spec.supports(capability)

        def to_dict(self):
            # Configuration bytes remain identical. Getter effects, not a
            # differing declared capability value, must cause the refusal.
            return spec.to_dict()

    capability = ObservedCapability()

    class ProviderProxy:
        provider_id = case.provider.provider_id

        @property
        def capability_spec(self):
            if state["closing_root_seen"] and not state["fired"]:
                state["fired"] = True
                events.append("closing_configuration_getter")
                if fault == "materializer_identity":
                    case.harness.context_materializer = replacement
                else:
                    assert token.cancel(
                        cancellation_id=token.cancellation_id,
                        reason="cancel during closing capability observation",
                    )
            return capability

        def is_available(self):
            return case.provider.is_available()

        def generate(self, *args, **kwargs):
            raise AssertionError("prepare-only must never generate")

    case.harness.providers = (ProviderProxy(),)

    def count_and_arm(self, *args, **kwargs):
        measured = actual_count(*args, **kwargs)
        assert measured > 0
        state["counted"] = True
        events.append("actual_tokenization")
        return measured

    def observe_closing_root(repository_id):
        observed = actual_root(repository_id)
        if state["window_checked"]:
            state["closing_root_seen"] = True
            events.append("closing_root_observation")
        return observed

    monkeypatch.setattr(type(case.port), "count_tokens", count_and_arm)
    monkeypatch.setattr(case.durable, "read_root", observe_closing_root)
    before = state_of(case)
    expected_error = HarnessCancelled if fault == "cancellation" else HarnessLoopError
    with pytest.raises(expected_error):
        case.harness.prepare_model_context(request)
    assert state["fired"]
    assert (
        events.index("actual_tokenization") < events.index("measured_context_window")
        < events.index("closing_root_observation") < events.index("closing_configuration_getter")
    )
    assert state_of(case) == before
    assert effects == []
