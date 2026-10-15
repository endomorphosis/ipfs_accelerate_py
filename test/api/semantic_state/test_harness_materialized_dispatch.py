"""No-model checks of the materialized semantic harness dispatch boundary.

The injected port/provider are authored fixtures. They do not establish real
producer freshness, provider-tokenizer fidelity or benchmark qualification.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessDisposition,
    ModelRoute,
)
from test.api.semantic_state.test_harness import (
    MemoryContextMaterializer,
    _accept_request,
    _bootstrap,
    _cid,
    _harness,
    _pack,
    _provider,
    _routing,
    _simple_patch,
)


class CapturingMaterializer(MemoryContextMaterializer):
    def __init__(self) -> None:
        self.prompt = ""
        self.verified_bindings: dict[str, Any] = {}
        self.count_bindings: dict[str, Any] = {}
        self.tokens: Any = 1234
        self.admitted = True
        self.mutate = lambda material: material

    def materialize(self, **bindings: Any) -> dict[str, Any]:
        material = dict(super().materialize(**bindings))
        material["dependency_capsules"] = [
            {
                "capsule_cid": cid,
                "capsule": {"exports": ["dependency_read"], "effects": ["read_only"]},
            }
            for cid in bindings["pack"].dependency_capsule_cids
        ]
        material["references"] = [
            {"cid": bindings["pack"].delta_cid, "kind": "source_delta"},
        ]
        return self.mutate(material)

    def verify(self, *, prompt: str, **bindings: Any) -> bool:
        self.prompt = prompt
        self.verified_bindings = bindings
        return self.admitted

    def count_tokens(self, prompt: str, **bindings: Any) -> Any:
        assert prompt == self.prompt
        self.count_bindings = bindings
        return self.tokens


def setup_dispatch(port: Any):
    provider = _provider(patch_text=_simple_patch())
    durable, harness = _harness(providers=(provider,), context_materializer=port)
    root = _bootstrap(harness)
    request = _accept_request(
        root,
        routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value),
        context_pack=_pack(
            dependency_capsule_cids=[_cid("selected-capsule")],
            assumptions=["conservative dependency; source effects not independently proved"],
            exclusions=["unchanged unrelated dependency omitted; use exact handle for expansion"],
        ),
    )
    return provider, durable, harness, root, request


def test_actual_dispatched_prompt_contains_source_facts_core_and_references():
    port = CapturingMaterializer()
    provider, durable, harness, root, request = setup_dispatch(port)
    result = harness.run(request)
    # Development dispatch stays observational; actual source in a prompt
    # does not relax the existing simulation/receipt promotion gates.
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "simulation_never_verifies_or_promotes" in result.result.reasons
    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert prompt == port.prompt
    assert prompt != request.objective
    packet = json.loads(prompt)
    assert packet["objective"] == request.objective
    assert packet["scope"] == request.scope.to_dict()
    assert packet["constraints"]["policy_cid"] == request.policy_cid
    assert packet["constraints"]["policy"] == harness.policy.to_dict()
    assert packet["context_pack"]["assumptions"] == list(request.context_pack.assumptions)
    assert packet["context_pack"]["exclusions"] == list(request.context_pack.exclusions)
    material = packet["materialized_context"]
    assert material["required_sources"]["target_source"]["text"] == "VALUE = 1\n"
    assert material["required_sources"]["surrounding_source"]["text"] == "# surrounding fixture\n"
    assert material["required_sources"]["test_source"]["text"] == "assert VALUE == 2\n"
    assert material["dependency_capsules"][0]["capsule"]["effects"] == ["read_only"]
    assert material["references"][0]["cid"] == request.context_pack.delta_cid
    assert packet["token_accounting_scope"] == "canonical_materialized_prompt_only"
    assert port.verified_bindings["request"] is request
    assert port.verified_bindings["base_tree"] == request.base_tree
    assert port.verified_bindings["provider_id"] == provider.provider_id
    assert port.count_bindings == {
        "provider_id": provider.provider_id,
        "route": ModelRoute.SMALL_LOCAL_MODEL.value,
        "mode": harness.policy.mode,
    }
    assert durable.read_root(request.repository_id) == root


def test_missing_materializer_never_falls_back_to_objective_or_calls_provider():
    provider, durable, harness, root, request = setup_dispatch(None)
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "context_materializer_unavailable" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


@pytest.mark.parametrize(
    "fault",
    [
        "missing_source",
        "wrong_source",
        "descriptor_source",
        "missing_capsule",
        "extra_capsule",
        "descriptor_capsule",
        "oversized_text",
    ],
)
def test_invalid_material_is_refused_before_provider_dispatch(fault):
    port = CapturingMaterializer()

    def mutate(material):
        if fault == "missing_source":
            del material["required_sources"]["test_source"]
        elif fault == "wrong_source":
            material["required_sources"]["target_source"]["source_cid"] = _cid("different")
        elif fault == "descriptor_source":
            del material["required_sources"]["target_source"]["text"]
        elif fault == "missing_capsule":
            material["dependency_capsules"] = []
        elif fault == "extra_capsule":
            material["dependency_capsules"].append(
                {
                    "capsule_cid": _cid("unselected"),
                    "capsule": {"exports": ["unselected"]},
                }
            )
        elif fault == "descriptor_capsule":
            material["dependency_capsules"][0]["capsule"] = {}
        elif fault == "oversized_text":
            material["required_sources"]["test_source"]["text"] = "x" * 8193
        return material

    port.mutate = mutate
    provider, durable, harness, root, request = setup_dispatch(port)
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "invalid_model_context" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root
    assert port.prompt == ""


@pytest.mark.parametrize("tokens", [True, False, 1.5, -1, 0, {}, 4097, 9000])
def test_invalid_or_over_limit_complete_input_count_never_dispatches(tokens):
    port = CapturingMaterializer()
    port.tokens = tokens
    provider, durable, harness, root, request = setup_dispatch(port)
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "model_context_overflow" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


def test_admission_refusal_does_not_call_provider_or_advance_root():
    port = CapturingMaterializer()
    port.admitted = False
    provider, durable, harness, root, request = setup_dispatch(port)
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "model_context_not_admitted" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


def test_missing_required_source_at_authority_boundary_is_unavailable():
    class MissingSource(CapturingMaterializer):
        def materialize(self, **bindings):
            raise FileNotFoundError("current source unavailable")

    provider, durable, harness, root, request = setup_dispatch(MissingSource())
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "context_materialization_unavailable" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


def test_exact_measured_count_replaces_descriptor_hint_for_dispatch():
    port = CapturingMaterializer()
    provider, _, harness, _, request = setup_dispatch(port)
    decision = request.routing_decision
    prompt, measured, selected = harness._materialize_model_prompt(
        request,
        request.context_pack,
        _cid("pack"),
        request.base_tree,
        decision,
    )
    assert measured.inputs.context_tokens == 1234
    assert decision.inputs.context_tokens == 100
    assert selected is provider
    assert prompt == port.prompt


def test_precomputed_gateway_result_cannot_attest_to_newly_rendered_prompt():
    provider, durable, harness, root, request = setup_dispatch(CapturingMaterializer())
    harness.gateway_result = object()
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "unbound_gateway_context" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


def test_deterministic_patch_path_does_not_require_materializer():
    _, harness = _harness(context_materializer=None)
    root = _bootstrap(harness)
    result = harness.run(_accept_request(root))
    assert result.result.disposition == HarnessDisposition.ACCEPTED.value


def test_context_authority_refuses_stale_source_without_dispatch():
    class StaleSource(CapturingMaterializer):
        def verify(self, **bindings):
            raise ValueError("producer current tree differs")

    provider, durable, harness, root, request = setup_dispatch(StaleSource())
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "model_context_not_admitted" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root


def test_test_imports_use_current_accelerate_and_datasets_checkouts():
    import ipfs_accelerate_py.agent_supervisor.semantic_state.harness as harness_module

    import ipfs_datasets_py

    current_a = Path(__file__).resolve().parents[3]
    current_p = current_a.parent / "ipfs_datasets"
    assert Path(harness_module.__file__).resolve().is_relative_to(current_a)
    assert Path(ipfs_datasets_py.__file__).resolve().is_relative_to(current_p)


def test_oversized_material_is_refused_before_canonical_encoding(monkeypatch):
    import ipfs_accelerate_py.agent_supervisor.semantic_state.harness as harness_module

    port = CapturingMaterializer()

    def large_references(material):
        material["references"] = [{"text": "x" * 8192} for _ in range(40)]
        return material

    port.mutate = large_references
    _, _, harness, _, request = setup_dispatch(port)

    def must_not_encode(_value):
        pytest.fail("oversized material reached the canonical encoder")

    monkeypatch.setattr(harness_module, "canonical_context_json_bytes", must_not_encode)
    with pytest.raises(harness_module.HarnessLoopError, match="transport bounds"):
        harness._materialize_model_prompt(
            request,
            request.context_pack,
            _cid("pack"),
            request.base_tree,
            request.routing_decision,
        )


@pytest.mark.parametrize("source_text", ["", "\n", " \t\n"])
def test_producer_admitted_empty_source_is_byte_exact_in_actual_prompt(source_text):
    port = CapturingMaterializer()

    def admitted_empty(material):
        material["required_sources"]["test_source"]["text"] = source_text
        return material

    port.mutate = admitted_empty
    provider, _, harness, _, request = setup_dispatch(port)
    harness.run(request)
    assert len(provider.calls) == 1
    packet = json.loads(provider.calls[0]["prompt"])
    assert packet["materialized_context"]["required_sources"]["test_source"]["text"] == source_text


def test_same_input_replay_is_historical_without_new_dispatch_or_cas():
    port = CapturingMaterializer()
    provider, durable, harness, _, request = setup_dispatch(port)
    first = harness.run(request)
    initial_cas = len(durable.cas_calls)
    second = harness.run(request)
    assert len(provider.calls) == 1
    assert len(durable.cas_calls) == initial_cas
    assert second.historical_replay is True
    assert second.to_dict()["historical_replay"] is True
    assert "historical_attempt_replay" in second.result.reasons
    assert "context_freshness_not_revalidated" in second.result.reasons
    assert second.attempt_identity == first.attempt_identity
    assert second.result.disposition == first.result.disposition
    assert first.historical_replay is False


@pytest.mark.parametrize(
    "changed",
    [
        "objective",
        "pack",
        "tree",
        "policy",
        "provider",
        "revision",
        "source",
        "scope",
    ],
)
def test_changed_bindings_require_new_attempt_instead_of_replaying_success(changed):
    port = CapturingMaterializer()
    provider, durable, harness, root, request = setup_dispatch(port)
    first = harness.run(request)
    second_request = request
    if changed == "objective":
        second_request = replace(request, objective="different objective")
    elif changed == "pack":
        second_request = replace(
            request, context_pack=replace(request.context_pack, assumptions=("changed assumption",))
        )
    elif changed == "tree":
        second_request = replace(request, base_tree=_cid("different-tree"))
    elif changed == "policy":
        second_request = replace(request, policy_cid=_cid("different-policy"))
    elif changed == "provider":
        harness.providers = (
            _provider(patch_text=_simple_patch(), provider_id="different-provider"),
        )
    elif changed == "revision":
        port.revision_cid = _cid("different-materializer-provider-revision")
    elif changed == "source":
        second_request = replace(request, visible_sources={"pkg/target.py": "VALUE = 3\n"})
    elif changed == "scope":
        second_request = replace(
            request, scope=replace(request.scope, task_owned_paths=("pkg/sub/",))
        )
    calls = len(provider.calls)
    cas = len(durable.cas_calls)
    replay = harness.run(second_request)
    assert replay.result.disposition == HarnessDisposition.REJECTED.value
    assert "attempt_context_changed" in replay.result.reasons
    assert "new_attempt_key_required" in replay.result.reasons
    assert not replay.historical_replay
    assert len(provider.calls) == calls
    assert len(durable.cas_calls) == cas
    assert durable.read_root(request.repository_id) == root
    record = harness._terminal_attempts[first.attempt_identity]
    assert record["outcome"]["result"] == first.result.to_dict()


def test_old_durable_key_only_terminal_row_does_not_satisfy_bound_replay():
    provider, durable, harness, root, request = setup_dispatch(CapturingMaterializer())
    first = harness.run(request)
    old = dict(harness._terminal_attempts[first.attempt_identity])
    old.pop("replay_context_cid")
    durable._objects[first.attempt_identity] = old
    harness._terminal_attempts.clear()
    calls = len(provider.calls)
    cas = len(durable.cas_calls)
    replay = harness.run(request)
    assert replay.result.disposition == HarnessDisposition.REJECTED.value
    assert "attempt_context_changed" in replay.result.reasons
    assert len(provider.calls) == calls
    assert len(durable.cas_calls) == cas
    assert durable.read_root(request.repository_id) == root


def test_provider_dispatch_requires_a_declared_materializer_revision():
    port = CapturingMaterializer()
    port.revision_cid = None
    provider, durable, harness, root, request = setup_dispatch(port)
    result = harness.run(request)
    assert result.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "context_materializer_unavailable" in result.result.reasons
    assert provider.calls == []
    assert durable.read_root(request.repository_id) == root
