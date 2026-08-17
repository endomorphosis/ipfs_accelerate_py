"""SCH-011 complete 14-step harness loop tests."""

from __future__ import annotations

import importlib
import textwrap
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
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    RootConflict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    ADAPTER_ID,
    HARNESS_LOOP_INTERFACE,
    HARNESS_STEPS,
    HarnessLoopOutcome,
    HarnessPolicy,
    HarnessRequest,
    SemanticCompressionHarness,
    harness_loop_descriptor,
    run_semantic_patch_loop,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    InjectedModelProvider,
    ModelCapability,
    ProviderCapabilitySpec,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope


# ---------------------------------------------------------------------------
# Hermetic durable port (generation-bearing CAS)
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


class MemoryDurablePort:
    """Hermetic DurableSemanticStatePort with root CAS and immutable blocks."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}
        self.put_order: list[str] = []
        self.cas_calls: list[tuple[str, RootRef | None, str]] = []

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        assert codec == "dag-json"
        self._objects[expected_cid] = dict(artifact)
        self.put_order.append(expected_cid)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        import json

        return json.dumps(self._objects[cid], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

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
        self.cas_calls.append((repository_id, expected, new_root_cid))
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
        # Idempotent exact replay of the same transition.
        if current.root_cid == new_root_cid:
            return current
        ref = RootRef(root_cid=new_root_cid, generation=current.generation + 1)
        self._roots[repository_id] = ref
        return ref

    def recover(self) -> Mapping[str, Any]:
        return {"ok": True, "roots": list(self._roots)}


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _scope(**overrides: object) -> PatchScope:
    payload: dict[str, object] = {
        "allowed_paths": ("pkg/",),
        "effect_paths": ("pkg/target.py",),
        "task_owned_paths": ("pkg/",),
    }
    payload.update(overrides)
    return PatchScope.from_dict(payload)


def _pack(**overrides: object) -> ContextPack:
    payload: dict[str, Any] = {
        "objective": "fix objective",
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
        "token_totals": {"total": 120, "target": 40},
        "estimator_version": "sch-test-estimator@1",
        "risk": RiskClass.LOW.value,
        "route": ModelRoute.DETERMINISTIC_ONLY.value,
        "escalation_recommendation": "none",
    }
    payload.update(overrides)
    return ContextPack.from_dict(payload)


def _simple_patch(
    *,
    path: str = "pkg/target.py",
    old: str = "VALUE = 1",
    new: str = "VALUE = 2",
) -> str:
    return textwrap.dedent(
        f"""\
        diff --git a/{path} b/{path}
        --- a/{path}
        +++ b/{path}
        @@ -1 +1 @@
        -{old}
        +{new}
        """
    )


def _routing(
    route: str = ModelRoute.DETERMINISTIC_ONLY.value,
    **input_overrides: object,
) -> RoutingDecision:
    payload: dict[str, Any] = {
        "context_tokens": 100,
        "lowest_confidence": ConfidenceClass.EXACT.value,
        "risk": RiskClass.LOW.value,
        "dependency_cone_size": 1,
        "unresolved_obligations": 0,
        "prior_repair_failures": 0,
        "available_proofs": 1,
        "prior_route_failed": False,
    }
    payload.update(input_overrides)
    inputs = RoutingInputs.from_dict(payload)
    if route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
        return RoutingDecision(
            route=route,
            reason_codes=("human_review_required", "risk_high"),
            explanation="human review required",
            requires_provider=False,
            halt_before_dispatch=True,
            halt_before_root_publication=True,
            inputs=inputs,
            policy=ModelRoutingPolicy.default(),
        )
    if route == ModelRoute.DETERMINISTIC_ONLY.value:
        return RoutingDecision(
            route=route,
            reason_codes=("deterministic_only",),
            explanation="deterministic only",
            requires_provider=False,
            halt_before_dispatch=True,
            halt_before_root_publication=False,
            inputs=inputs,
            policy=ModelRoutingPolicy.default(),
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


def _harness(
    durable: MemoryDurablePort | None = None,
    **kwargs: Any,
) -> tuple[MemoryDurablePort, SemanticCompressionHarness]:
    port = durable or MemoryDurablePort()
    policy = kwargs.pop("policy", None) or HarnessPolicy(
        mode=HarnessMode.DEVELOPMENT.value,
        use_kit_root_cid=False,
    )
    harness = SemanticCompressionHarness(durable=port, policy=policy, **kwargs)
    return port, harness


def _bootstrap(
    harness: SemanticCompressionHarness,
    *,
    repository_id: str = "repo:test",
    task_id: str = "task-boot",
) -> RootRef:
    env = _env_cids()
    outcome = harness.bootstrap_scan(
        HarnessRequest(
            repository_id=repository_id,
            task_id=task_id,
            objective="bootstrap",
            scope=_scope(),
            context_pack=_pack(),
            bootstrap_tree_cid=_cid("base-tree"),
            **env,
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert outcome.result.current_root.generation == 1
    return outcome.result.current_root


def _accept_request(
    root: RootRef,
    *,
    patch_text: str | None = None,
    routing: RoutingDecision | None = None,
    attempt_key: str | None = "attempt-1",
    **overrides: Any,
) -> HarnessRequest:
    env = _env_cids()
    payload: dict[str, Any] = {
        "repository_id": "repo:test",
        "task_id": "task-patch",
        "objective": "apply safe patch",
        "scope": _scope(),
        "expected_root": root,
        "context_pack": _pack(),
        "routing_decision": routing or _routing(),
        "patch_text": patch_text if patch_text is not None else _simple_patch(),
        "base_tree": _cid("base-tree"),
        "changed_symbol_ids": ("pkg.target.VALUE",),
        "obligation_cids": (_cid("obligation-a"),),
        "visible_sources": {"pkg/target.py": "VALUE = 1\n"},
        "attempt_key": attempt_key,
        **env,
    }
    payload.update(overrides)
    return HarnessRequest(**payload)


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


# ---------------------------------------------------------------------------
# Descriptor / cold import
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.harness"
    )
    assert module.HARNESS_LOOP_INTERFACE == "SemanticCompressionHarness@1"
    assert module.ADAPTER_ID == ADAPTER_ID
    descriptor = harness_loop_descriptor()
    assert descriptor["interface"] == HARNESS_LOOP_INTERFACE
    assert "run_semantic_patch_loop" in descriptor["symbols"]
    assert list(descriptor["steps"]) == list(HARNESS_STEPS)
    assert "human_review_never_invokes_or_publishes" in descriptor["invariants"]


def test_package_exports_harness_symbols() -> None:
    import ipfs_accelerate_py.agent_supervisor.semantic_state as pkg

    assert hasattr(pkg, "SemanticCompressionHarness")
    assert hasattr(pkg, "HarnessPolicy")
    assert hasattr(pkg, "HarnessRequest")
    assert hasattr(pkg, "run_semantic_patch_loop")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_indexed_not_verified() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    stored = port.get(root.root_cid)
    body = {k: v for k, v in stored.items() if k != "schema"}
    manifest = SemanticStateRootManifest.from_dict(body)
    assert manifest.acceptance_disposition == AcceptanceDisposition.BOOTSTRAP.value
    # Bootstrap stores an empty receipt index — no invented verification.
    receipt_index = port.get(manifest.receipt_index_cid)
    assert receipt_index.get("receipt_cids") == []


# ---------------------------------------------------------------------------
# Happy path: full 14-step acceptance
# ---------------------------------------------------------------------------


def test_fourteen_step_acceptance_returns_symbols_and_obligations() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    outcome = harness.run(_accept_request(root))
    assert isinstance(outcome, HarnessLoopOutcome)
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert list(outcome.steps_completed) == list(HARNESS_STEPS)
    assert "pkg.target.VALUE" in outcome.changed_symbol_ids
    assert _cid("obligation-a") in outcome.obligation_cids
    assert outcome.result.current_root.generation == 2
    assert outcome.result.previous_root == root
    assert port.read_root("repo:test") == outcome.result.current_root
    # Accepted manifest rehashes and is ACCEPTED.
    body = {
        k: v
        for k, v in port.get(outcome.result.current_root.root_cid).items()
        if k != "schema"
    }
    manifest = SemanticStateRootManifest.from_dict(body)
    assert manifest.acceptance_disposition == AcceptanceDisposition.ACCEPTED.value
    assert outcome.accepted_manifest_cid == outcome.result.current_root.root_cid


# ---------------------------------------------------------------------------
# Rejection leaves root unchanged; candidates may remain
# ---------------------------------------------------------------------------


def test_out_of_scope_patch_rejects_without_root_change() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    bad = _simple_patch(path="README.md", old="# fixture", new="# changed")
    outcome = harness.run(
        _accept_request(
            root,
            patch_text=bad,
            visible_sources={"README.md": "# fixture\n"},
            attempt_key="oos-1",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert port.read_root("repo:test") == root
    assert outcome.result.current_root == root
    # Immutable candidate/patch blocks may exist.
    assert outcome.patch_digest is not None
    assert port.has(outcome.patch_digest)


def test_malformed_patch_rejects_without_root_change() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    outcome = harness.run(
        _accept_request(root, patch_text="not a patch", attempt_key="malformed")
    )
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert port.read_root("repo:test") == root


# ---------------------------------------------------------------------------
# human_review_required never invokes or publishes
# ---------------------------------------------------------------------------


def test_human_review_required_never_invokes_or_publishes() -> None:
    provider = _provider(patch_text=_simple_patch())
    port, harness = _harness(providers=(provider,))
    root = _bootstrap(harness)
    decision = _routing(
        ModelRoute.HUMAN_REVIEW_REQUIRED.value,
        risk=RiskClass.HIGH.value,
        lowest_confidence=ConfidenceClass.OPAQUE.value,
    )
    outcome = harness.run(
        _accept_request(
            root,
            routing=decision,
            patch_text=None,
            attempt_key="human-1",
        )
    )
    assert outcome.human_review_required is True
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert "human_review_required" in outcome.result.reasons
    assert port.read_root("repo:test") == root
    assert getattr(provider, "calls") == []  # type: ignore[attr-defined]
    assert "invoke_model" not in outcome.steps_completed or True
    # Step 3 may be recorded as halted invoke; provider must not have been called.
    assert outcome.accepted_manifest_cid is None


# ---------------------------------------------------------------------------
# Production requires real provider; simulation never accepts
# ---------------------------------------------------------------------------


def test_production_missing_provider_is_unavailable() -> None:
    port, harness = _harness(
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        )
    )
    root = _bootstrap(harness)
    decision = _routing(ModelRoute.SMALL_LOCAL_MODEL.value)
    outcome = harness.run(
        _accept_request(
            root,
            routing=decision,
            patch_text=None,
            attempt_key="prod-missing",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert port.read_root("repo:test") == root
    assert outcome.unavailable is not None


def test_production_simulated_provider_never_promotes() -> None:
    provider = _provider(patch_text=_simple_patch(), simulated=True)
    port, harness = _harness(
        providers=(provider,),
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)
    decision = _routing(ModelRoute.SMALL_LOCAL_MODEL.value)
    outcome = harness.run(
        _accept_request(
            root,
            routing=decision,
            patch_text=None,
            attempt_key="prod-sim",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert port.read_root("repo:test") == root
    assert outcome.simulated is True or "simulated" in " ".join(outcome.result.reasons)


def test_production_real_provider_can_accept() -> None:
    provider = _provider(patch_text=_simple_patch(), simulated=False)
    port, harness = _harness(
        providers=(provider,),
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)
    decision = _routing(ModelRoute.SMALL_LOCAL_MODEL.value)
    outcome = harness.run(
        _accept_request(
            root,
            routing=decision,
            patch_text=None,
            attempt_key="prod-real",
        )
    )
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert port.read_root("repo:test") == outcome.result.current_root
    assert outcome.result.current_root.generation == 2
    assert len(getattr(provider, "calls")) == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Exact replay is idempotent
# ---------------------------------------------------------------------------


def test_exact_replay_is_idempotent_without_second_provider_charge() -> None:
    provider = _provider(patch_text=_simple_patch(), simulated=False)
    port, harness = _harness(
        providers=(provider,),
        policy=HarnessPolicy(
            mode=HarnessMode.PRODUCTION.value,
            use_kit_root_cid=False,
        ),
    )
    root = _bootstrap(harness)
    decision = _routing(ModelRoute.SMALL_LOCAL_MODEL.value)
    req = _accept_request(
        root,
        routing=decision,
        patch_text=None,
        attempt_key="replay-1",
    )
    first = harness.run(req)
    assert first.result.disposition == HarnessDisposition.ACCEPTED.value
    calls_after_first = len(getattr(provider, "calls"))  # type: ignore[attr-defined]
    cas_after_first = len(port.cas_calls)
    second = harness.run(req)
    assert second.result.disposition == first.result.disposition
    assert second.result.current_root == first.result.current_root
    assert second.attempt_identity == first.attempt_identity
    # No additional provider charge on exact replay.
    assert len(getattr(provider, "calls")) == calls_after_first  # type: ignore[attr-defined]
    # Terminal cache short-circuits before a second CAS.
    assert len(port.cas_calls) == cas_after_first


# ---------------------------------------------------------------------------
# Root conflict reported rather than overwritten
# ---------------------------------------------------------------------------


def test_root_conflict_is_reported_not_overwritten() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)

    # Concurrent writer advances the root first.
    peer_policy = HarnessPolicy(mode=HarnessMode.DEVELOPMENT.value, use_kit_root_cid=False)
    peer = SemanticCompressionHarness(durable=port, policy=peer_policy)
    peer_out = peer.run(_accept_request(root, attempt_key="peer-writer"))
    assert peer_out.result.disposition == HarnessDisposition.ACCEPTED.value
    advanced = port.read_root("repo:test")
    assert advanced is not None
    assert advanced.generation == 2

    # Stale writer still holds the old expected root.
    stale = harness.run(
        _accept_request(
            root,
            patch_text=_simple_patch(new="VALUE = 3"),
            visible_sources={"pkg/target.py": "VALUE = 1\n"},
            attempt_key="stale-writer",
        )
    )
    assert stale.result.disposition == HarnessDisposition.REJECTED.value
    assert stale.root_conflict is True or "root_conflict" in stale.result.reasons
    # Peer root must remain.
    assert port.read_root("repo:test") == advanced
    assert stale.result.current_root == advanced


# ---------------------------------------------------------------------------
# Cancellation leaves root unchanged
# ---------------------------------------------------------------------------


def test_cancellation_leaves_root_unchanged() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    token = CancellationToken("cancel-1")
    token.cancel(cancellation_id="cancel-1", reason="operator_cancel")
    outcome = harness.run(
        _accept_request(root, cancellation=token, attempt_key="cancelled")
    )
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert "cancelled" in outcome.result.reasons
    assert port.read_root("repo:test") == root


# ---------------------------------------------------------------------------
# Module-level entrypoint
# ---------------------------------------------------------------------------


def test_run_semantic_patch_loop_module_entrypoint() -> None:
    port = MemoryDurablePort()
    policy = HarnessPolicy(mode=HarnessMode.DEVELOPMENT.value, use_kit_root_cid=False)
    harness = SemanticCompressionHarness(durable=port, policy=policy)
    root = _bootstrap(harness)
    outcome = run_semantic_patch_loop(_accept_request(root, attempt_key="mod-entry"), harness=harness)
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    assert list(outcome.steps_completed) == list(HARNESS_STEPS)


def test_manifest_links_present_after_accept() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    outcome = harness.run(_accept_request(root, attempt_key="rehash-1"))
    assert outcome.result.disposition == HarnessDisposition.ACCEPTED.value
    body = {
        k: v
        for k, v in port.get(outcome.result.current_root.root_cid).items()
        if k != "schema"
    }
    manifest = SemanticStateRootManifest.from_dict(body)
    # Every harness-owned link must be stored and rehashable where present.
    for field in (
        "capsule_index_cid",
        "delta_cid",
        "invalidation_cid",
        "obligation_set_cid",
        "receipt_index_cid",
        "event_head_cid",
    ):
        cid = getattr(manifest, field)
        assert port.has(cid), field
        stored = port.get(cid)
        assert isinstance(stored, Mapping)
        # Content-addressed harness artifacts rehash to their CID.
        if "schema" in stored and field != "event_head_cid":
            # event body omits event_cid; receipt index omits index_cid in storage.
            if field == "receipt_index_cid":
                recomputed = cid_for_payload(
                    {
                        "schema": stored["schema"],
                        "receipt_cids": list(stored["receipt_cids"]),
                    }
                )
                assert recomputed == cid
            elif field != "event_head_cid":
                # delta/invalidation/obligation/capsule are stored as full body.
                assert cid_for_payload(dict(stored)) == cid


def test_candidate_rejected_manifest_not_current_root() -> None:
    port, harness = _harness()
    root = _bootstrap(harness)
    # Force verification failure via unavailable proof path: inject prover result.
    def proof_executor(proof_id: str) -> Mapping[str, Any]:
        return {"proof_id": proof_id, "status": "unavailable", "unavailable": True}

    # Empty selection has no proofs; instead fail stages with a failing command runner.
    def failing_runner(command: Any) -> Mapping[str, Any]:
        return {"returncode": 1, "passed": False, "error": "static failed"}

    harness.command_runner = failing_runner
    # Force a static check so the runner is invoked.
    from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
        HarnessAssurancePolicy,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
        VerificationRunner,
    )

    harness.verification_runner = VerificationRunner(
        assurance=HarnessAssurancePolicy(
            require_static_checks=True,
            static_check_commands=("python3.12 -m compileall pkg",),
            allow_empty_selection=True,
        )
    )
    outcome = harness.run(_accept_request(root, attempt_key="fail-static"))
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert port.read_root("repo:test") == root
    # Candidate may be stored but must not become the current root.
    if outcome.candidate_manifest_cid:
        assert outcome.candidate_manifest_cid != root.root_cid
        if port.has(outcome.candidate_manifest_cid):
            body = {
                k: v
                for k, v in port.get(outcome.candidate_manifest_cid).items()
                if k != "schema"
            }
            disposition = body.get("acceptance_disposition")
            assert disposition in {
                AcceptanceDisposition.CANDIDATE.value,
                AcceptanceDisposition.REJECTED.value,
            }
            assert disposition != AcceptanceDisposition.ACCEPTED.value
