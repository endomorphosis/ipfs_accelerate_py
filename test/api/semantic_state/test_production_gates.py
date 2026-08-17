"""SCH-015 production simulation / degraded / OFF / fallback gates.

``sim:`` / ``degraded:`` reservations, OFF / SIMULATED / DEGRADED phases, fallback
reason codes, and unadmitted replay must never report production verification
or root-commit authority.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    ContextPack,
    HarnessDisposition,
    HarnessMode,
    ModelRoute,
    RootRef,
    SemanticStateRootManifest,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import RootConflict
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessPolicy,
    HarnessRequest,
    SemanticCompressionHarness,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    InjectedModelProvider,
    ModelCapability,
    ProductionProviderGate,
    ProviderCapabilitySpec,
    invoke_model,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.receipts import (
    ADMISSION_SIMULATED,
    PROVIDER_MODE_PRODUCTION,
    PROVIDER_MODE_SIMULATED,
    PROOF_STATUS_PASSED,
    SimulatedReceiptError,
    admit_receipt,
    compile_verification_receipt,
    receipt_may_promote_root,
    receipt_may_verify,
    ReceiptBindings,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
    route_model,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    ProviderBinding,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gateway_result(**overrides: Any) -> SimpleNamespace:
    payload = {
        "phase": "settled",
        "final_status": "committed",
        "granted": True,
        "reservation_id": "resv:prod-1",
        "provider_id": "provider-alpha",
        "reason_codes": (),
        "replayed": False,
        "coordination_state": "available",
        "mode": "enforce",
        "attribution": {
            "attribution_id": "attr:1",
            "provider_id": "provider-alpha",
            "scope_id": "scope_alpha",
        },
        "supervisor_receipt_id": "receipt:prod-1",
        "receipt": {"receipt_id": "receipt:prod-1"},
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _production_gate(**overrides: Any) -> ProductionProviderGate:
    payload = {
        "expected_provider_id": "provider-alpha",
        "coordinator_present": True,
        "invoker_present": True,
        "admitted_production_receipt_ids": ("receipt:prod-1",),
    }
    payload.update(overrides)
    return ProductionProviderGate(**payload)


def _assert_not_production_verification(verdict: Any) -> None:
    assert verdict.admitted is False or verdict.can_verify is False
    assert verdict.can_verify is False
    assert verdict.can_commit is False


class MemoryDurablePort:
    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        self._objects[expected_cid] = dict(artifact)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        import json

        return json.dumps(
            self._objects[cid], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def has(self, cid: str) -> bool:
        return cid in self._objects

    def read_root(self, repository_id: str) -> RootRef | None:
        return self._roots.get(repository_id)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        current = self._roots.get(repository_id)
        body = self.get(new_root_cid)
        manifest = {k: v for k, v in body.items() if k != "schema"}
        disposition = manifest.get("acceptance_disposition")
        if expected is None:
            if current is not None:
                raise RootConflict("root already exists")
            if disposition != AcceptanceDisposition.BOOTSTRAP.value:
                raise RootConflict("initial CAS requires bootstrap disposition")
            ref = RootRef(root_cid=new_root_cid, generation=1)
            self._roots[repository_id] = ref
            return ref
        if current is None:
            raise RootConflict("expected root missing")
        if (
            current.root_cid != expected.root_cid
            or current.generation != expected.generation
        ):
            raise RootConflict("expected root token mismatch")
        if disposition != AcceptanceDisposition.ACCEPTED.value:
            raise RootConflict("only accepted manifests may advance the root")
        if current.root_cid == new_root_cid:
            return current
        ref = RootRef(root_cid=new_root_cid, generation=current.generation + 1)
        self._roots[repository_id] = ref
        return ref

    def recover(self) -> Mapping[str, Any]:
        return {"ok": True}


def _scope() -> PatchScope:
    return PatchScope.from_dict(
        {
            "allowed_paths": ("pkg/",),
            "effect_paths": ("pkg/target.py",),
            "task_owned_paths": ("pkg/",),
        }
    )


def _pack() -> ContextPack:
    return ContextPack.from_dict(
        {
            "objective": "production gate",
            "target_source_cid": _cid("target-src"),
            "surrounding_source_cid": _cid("surround-src"),
            "test_source_cid": _cid("test-src"),
            "dependency_capsule_cids": [],
            "obligation_cids": [],
            "counterexample_cids": [],
            "delta_cid": _cid("pack-delta"),
            "interface_cids": [],
            "assumptions": [],
            "exclusions": [],
            "token_totals": {"total": 80, "target": 20},
            "estimator_version": "sch-test-estimator@1",
            "risk": RiskClass.LOW.value,
            "route": ModelRoute.SMALL_LOCAL_MODEL.value,
            "escalation_recommendation": "none",
        }
    )


def _simple_patch() -> str:
    return textwrap.dedent(
        """\
        diff --git a/pkg/target.py b/pkg/target.py
        --- a/pkg/target.py
        +++ b/pkg/target.py
        @@ -1 +1 @@
        -VALUE = 1
        +VALUE = 2
        """
    )


def _routing(route: str = ModelRoute.SMALL_LOCAL_MODEL.value) -> RoutingDecision:
    inputs = RoutingInputs.from_dict(
        {
            "context_tokens": 2_000,
            "lowest_confidence": ConfidenceClass.HEURISTIC.value,
            "risk": RiskClass.LOW.value,
            "dependency_cone_size": 3,
            "unresolved_obligations": 0,
            "prior_repair_failures": 0,
            "available_proofs": 0,
            "prior_route_failed": False,
        }
    )
    return RoutingDecision(
        route=route,
        reason_codes=(route,),
        explanation=f"route {route}",
        requires_provider=True,
        halt_before_dispatch=False,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=ModelRoutingPolicy.default(),
    )


def _env_cids() -> dict[str, str]:
    return {
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
    }


def _provider(
    *,
    patch_text: str,
    provider_id: str = "prod-provider",
    simulated: bool = False,
) -> InjectedModelProvider:
    calls: list[dict[str, Any]] = []

    def generate_fn(prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        calls.append({"prompt": prompt, **kwargs})
        return {
            "provider_id": provider_id,
            "status": "ok",
            "simulated": simulated,
            "patch_text": patch_text,
        }

    spec = ProviderCapabilitySpec.from_dict(
        {
            "provider_id": provider_id,
            "capabilities": [
                ModelCapability.SMALL_LOCAL.value,
                ModelCapability.MEDIUM.value,
                ModelCapability.FRONTIER.value,
            ],
            "max_context_tokens": 200_000,
            "modality": "text",
            "available": True,
        }
    )
    provider = InjectedModelProvider(spec=spec, generate_fn=generate_fn)
    object.__setattr__(provider, "calls", calls)  # type: ignore[attr-defined]
    return provider


def _bootstrap(harness: SemanticCompressionHarness) -> RootRef:
    env = _env_cids()
    outcome = harness.bootstrap_scan(
        HarnessRequest(
            repository_id="repo:prod",
            task_id="task-boot",
            objective="bootstrap",
            scope=_scope(),
            context_pack=_pack(),
            bootstrap_tree_cid=_cid("base-tree"),
            **env,
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    return outcome.result.current_root


def _accept_request(root: RootRef, **overrides: Any) -> HarnessRequest:
    env = _env_cids()
    payload: dict[str, Any] = {
        "repository_id": "repo:prod",
        "task_id": "task-patch",
        "objective": "production path",
        "scope": _scope(),
        "expected_root": root,
        "context_pack": _pack(),
        "routing_decision": _routing(),
        "patch_text": None,
        "base_tree": _cid("base-tree"),
        "changed_symbol_ids": ("pkg.target.VALUE",),
        "obligation_cids": (_cid("obligation-a"),),
        "visible_sources": {"pkg/target.py": "VALUE = 1\n"},
        "attempt_key": "prod-1",
        **env,
    }
    payload.update(overrides)
    return HarnessRequest(**payload)


def _receipt_bindings(**overrides: Any) -> ReceiptBindings:
    payload: dict[str, Any] = {
        "pre_tree_cid": _cid("pre-tree"),
        "post_tree_cid": _cid("post-tree"),
        "datasets_state_cid": _cid("datasets-state"),
        "datasets_semantic_state_root_cid": _cid("datasets-root"),
        "capsule_index_cid": _cid("capsule-index"),
        "delta_cid": _cid("delta"),
        "selection_cid": _cid("selection"),
        "previous_semantic_state_root_cid": _cid("prev-root"),
        "current_semantic_state_root_cid": _cid("curr-root"),
        "command_identity": "sch-cmd:prod:1",
        "toolchain_cid": _cid("toolchain"),
        "dependency_lock_cid": _cid("lock"),
        "config_cid": _cid("config"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
        "provider_mode": PROVIDER_MODE_PRODUCTION,
        "proof_outcomes": [
            {"proof_id": "proof.a", "status": PROOF_STATUS_PASSED},
        ],
        "output_artifact_cids": [_cid("out-a")],
        "event_parent_cid": _cid("event-parent"),
    }
    payload.update(overrides)
    return ReceiptBindings.from_dict(payload)


def _current_from(bindings: ReceiptBindings) -> dict[str, Any]:
    data = bindings.to_dict()
    return {
        "pre_tree_cid": data["pre_tree_cid"],
        "post_tree_cid": data["post_tree_cid"],
        "datasets_state_cid": data["datasets_state_cid"],
        "datasets_semantic_state_root_cid": data["datasets_semantic_state_root_cid"],
        "capsule_index_cid": data["capsule_index_cid"],
        "delta_cid": data["delta_cid"],
        "selection_cid": data["selection_cid"],
        "previous_semantic_state_root_cid": data["previous_semantic_state_root_cid"],
        "current_semantic_state_root_cid": data["current_semantic_state_root_cid"],
        "command_identity": data["command_identity"],
        "toolchain_cid": data["toolchain_cid"],
        "dependency_lock_cid": data["dependency_lock_cid"],
        "config_cid": data["config_cid"],
        "policy_cid": data["policy_cid"],
        "interface_cid": data["interface_cid"],
        "provider_mode": data["provider_mode"],
    }


# ---------------------------------------------------------------------------
# Predicted symbol: production rejects simulation matrix
# ---------------------------------------------------------------------------


def test_production_rejects_simulation() -> None:
    """Canonical forbidden production paths never get verify/commit authority."""

    gate = _production_gate()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"reservation_id": "sim:local"}, "sim"),
        ({"reservation_id": "degraded:local"}, "degraded"),
        ({"mode": "off"}, "off"),
        ({"mode": "OFF"}, "off"),
        ({"mode": "assist"}, "assist"),
        ({"phase": "simulated"}, "simulated"),
        ({"phase": "SIMULATED"}, "simulated"),
        ({"phase": "degraded"}, "degraded"),
        ({"phase": "DEGRADED"}, "degraded"),
        ({"phase": "off"}, "off"),
        ({"coordination_state": "simulated"}, "simulated"),
        ({"coordination_state": "unavailable"}, "unavailable"),
        ({"reason_codes": ("local_fallback_used",)}, "fallback"),
        ({"reason_codes": ("cross_provider_fallback",)}, "fallback"),
        ({"reason_codes": ("allow_local_fallback",)}, "fallback"),
        ({"reason_codes": ("degraded_capacity",)}, "degraded"),
        (
            {"replayed": True, "supervisor_receipt_id": "receipt:unknown"},
            "unadmitted",
        ),
        ({"granted": False}, "not_granted"),
        ({"provider_id": "other-provider"}, "mismatch"),
        ({"attribution": None}, "attribution"),
    ]

    for override, label in cases:
        verdict = gate.evaluate(
            _gateway_result(**override),
            mode=HarnessMode.PRODUCTION,
        )
        _assert_not_production_verification(verdict)
        assert "production_rejected" in verdict.reason_codes or verdict.admitted is False, (
            label,
            verdict.reason_codes,
        )


def test_production_admits_only_enforce_available_real_path() -> None:
    gate = _production_gate()
    verdict = gate.evaluate(
        _gateway_result(),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is True
    assert verdict.can_verify is True
    assert verdict.can_commit is True
    assert verdict.simulated is False
    assert "production_admitted" in verdict.reason_codes

    # Admitted replay with a known production receipt is allowed.
    replay = gate.evaluate(
        _gateway_result(replayed=True, supervisor_receipt_id="receipt:prod-1"),
        mode=HarnessMode.PRODUCTION,
    )
    assert replay.admitted is True
    assert replay.can_commit is True


def test_development_simulation_never_verifies_or_commits() -> None:
    gate = _production_gate()
    decision = route_model(
        RoutingInputs.from_dict(
            {
                "context_tokens": 2_000,
                "lowest_confidence": ConfidenceClass.HEURISTIC.value,
                "risk": RiskClass.LOW.value,
                "dependency_cone_size": 3,
                "unresolved_obligations": 0,
                "prior_repair_failures": 0,
                "available_proofs": 0,
                "prior_route_failed": False,
            }
        )
    )
    provider = _provider(patch_text=_simple_patch())
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.DEVELOPMENT,
        prompt="dev-sim",
    )
    assert result.simulated is True
    assert result.exit_code == 0
    assert result.gate is not None
    assert result.gate.can_verify is False
    assert result.gate.can_commit is False
    assert result.gate.simulated is True

    # Direct development gateway evaluation is observational only.
    verdict = gate.evaluate(
        _gateway_result(reservation_id="sim:dev"),
        mode=HarnessMode.DEVELOPMENT,
    )
    assert verdict.can_verify is False
    assert verdict.can_commit is False


def test_invoke_model_production_rejects_sim_and_degraded_reservations() -> None:
    decision = route_model(
        RoutingInputs.from_dict(
            {
                "context_tokens": 2_000,
                "lowest_confidence": ConfidenceClass.HEURISTIC.value,
                "risk": RiskClass.LOW.value,
                "dependency_cone_size": 3,
                "unresolved_obligations": 0,
                "prior_repair_failures": 0,
                "available_proofs": 0,
                "prior_route_failed": False,
            }
        )
    )
    provider = _provider(patch_text=_simple_patch())

    for reservation_id in ("sim:bad", "degraded:bad"):
        rejected = invoke_model(
            decision=decision,
            providers=[provider],
            mode=HarnessMode.PRODUCTION,
            gateway_result=_gateway_result(reservation_id=reservation_id),
            coordinator_present=True,
            invoker_present=True,
        )
        assert rejected.status == "rejected"
        assert rejected.exit_code == 1
        assert rejected.gate is not None
        assert rejected.gate.can_verify is False
        assert rejected.gate.can_commit is False


def test_simulated_receipt_never_reports_production_verification() -> None:
    bindings = _receipt_bindings()
    receipt = compile_verification_receipt(
        bindings,
        exit_code=0,
        stages_passed=True,
        simulated=True,
        store=False,
    )
    assert receipt.simulated is True
    assert receipt.acceptance_eligible is False
    admission = admit_receipt(receipt, current=_current_from(bindings))
    assert admission.admission == ADMISSION_SIMULATED
    assert admission.can_verify is False
    assert admission.can_promote_root is False
    assert receipt_may_verify(admission) is False
    assert receipt_may_promote_root(admission) is False
    with pytest.raises(SimulatedReceiptError):
        admit_receipt(receipt, current=_current_from(bindings), raise_on_reject=True)

    # Provider mode simulated forces simulation even without explicit flag.
    sim_bindings = _receipt_bindings(provider_mode=PROVIDER_MODE_SIMULATED)
    sim_receipt = compile_verification_receipt(
        sim_bindings, exit_code=0, stages_passed=True, store=False
    )
    assert sim_receipt.simulated is True
    assert sim_receipt.acceptance_eligible is False


def test_production_simulated_provider_never_promotes_root() -> None:
    provider = _provider(patch_text=_simple_patch(), simulated=True)
    port = MemoryDurablePort()
    harness = SemanticCompressionHarness(
        durable=port,
        providers=(provider,),
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)
    outcome = harness.run(
        _accept_request(
            root,
            routing_decision=_routing(),
            patch_text=None,
            attempt_key="prod-sim",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert port.read_root("repo:prod") == root
    assert outcome.simulated is True or "simulated" in " ".join(outcome.result.reasons)
    assert outcome.accepted_manifest_cid is None


def test_production_real_provider_can_accept() -> None:
    provider = _provider(patch_text=_simple_patch(), simulated=False)
    port = MemoryDurablePort()
    harness = SemanticCompressionHarness(
        durable=port,
        providers=(provider,),
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)
    outcome = harness.run(
        _accept_request(
            root,
            routing_decision=_routing(),
            patch_text=None,
            attempt_key="prod-real",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert port.read_root("repo:prod") == outcome.result.current_root
    assert outcome.result.current_root.generation == 2
    body = {
        k: v
        for k, v in port.get(outcome.result.current_root.root_cid).items()
        if k != "schema"
    }
    manifest = SemanticStateRootManifest.from_dict(body)
    assert manifest.acceptance_disposition == AcceptanceDisposition.ACCEPTED.value


def test_scheduling_contract_rejects_simulated_production_reservation() -> None:
    """ProviderBinding / work result contracts refuse sim: under production."""

    from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError

    with pytest.raises(HarnessError, match="sim:|degraded:"):
        ProviderBinding.from_dict(
            {
                "provider_id": "provider-alpha",
                "reservation_id": "sim:local",
                "mode": HarnessMode.PRODUCTION.value,
                "simulated": False,
            }
        )

    with pytest.raises(HarnessError, match="sim:|degraded:"):
        ProviderBinding.from_dict(
            {
                "provider_id": "provider-alpha",
                "reservation_id": "degraded:local",
                "mode": HarnessMode.PRODUCTION.value,
                "simulated": False,
            }
        )

    with pytest.raises(HarnessError, match="cannot be simulated"):
        ProviderBinding.from_dict(
            {
                "provider_id": "provider-alpha",
                "reservation_id": "resv:real-1",
                "mode": HarnessMode.PRODUCTION.value,
                "simulated": True,
            }
        )


def test_unadmitted_replay_never_production_verifies() -> None:
    gate = _production_gate(admitted_production_receipt_ids=("receipt:known",))
    verdict = gate.evaluate(
        _gateway_result(
            replayed=True,
            supervisor_receipt_id="receipt:forged",
        ),
        mode=HarnessMode.PRODUCTION,
    )
    _assert_not_production_verification(verdict)
    assert "unadmitted_replay" in verdict.reason_codes

    # Empty admitted set rejects all replays.
    empty_gate = _production_gate(admitted_production_receipt_ids=())
    verdict2 = empty_gate.evaluate(
        _gateway_result(replayed=True, supervisor_receipt_id="receipt:prod-1"),
        mode=HarnessMode.PRODUCTION,
    )
    _assert_not_production_verification(verdict2)
    assert "unadmitted_replay" in verdict2.reason_codes


def test_off_mode_and_degraded_phase_matrix() -> None:
    gate = _production_gate()
    forbidden = [
        {"mode": "off", "phase": "settled"},
        {"mode": "observe", "phase": "settled"},
        {"mode": "shadow", "phase": "settled"},
        {"mode": "enforce", "phase": "off"},
        {"mode": "enforce", "phase": "simulated"},
        {"mode": "enforce", "phase": "degraded"},
        {"mode": "enforce", "phase": "denied"},
        {"mode": "enforce", "phase": "failed"},
        {
            "mode": "enforce",
            "phase": "settled",
            "reason_codes": ("simulated_fallback",),
        },
    ]
    for override in forbidden:
        verdict = gate.evaluate(
            _gateway_result(**override),
            mode=HarnessMode.PRODUCTION,
        )
        _assert_not_production_verification(verdict)
        assert verdict.can_verify is False
        assert verdict.can_commit is False
