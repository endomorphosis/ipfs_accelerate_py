"""Production Hammer coordination and reconstruction (LPR-012)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    NativeGoalDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicNativeGoalBinding,
    SemanticRoundTripReceipt,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    HAMMER_IMPORT_ISOLATION,
    HAMMER_IMPORT_ISOLATION_HARDENED,
    HammerSupervisorPolicy,
    IpfsDatasetsLogicProvider,
    IsolatedHammerLoader,
    get_isolated_hammer_loader,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    CodeProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.tactician_hammer_coordinator import (
    COORDINATION_OUTCOMES,
    CoordinationConclusiveness,
    CountermodelValidator,
    HammerCoordinationOutcome,
    PremiseSelectorMode,
    TacticianHammerCoordinator,
    create_tactician_hammer_coordinator,
    map_provider_status_to_outcome,
)
from ipfs_accelerate_py.agent_supervisor.validation.hammer_native_execution_gate import (
    NativeExecutionAuthorizationGate,
    NativeExecutionLane,
    NativeExecutionOperation,
    NativeExecutionPermit,
    ResourceEnforcementReport,
    ResourceEnforcementStrength,
    ResourcePolicySlice,
    probe_resource_enforcement,
)


def _lock(*, solvers=("z3",)):
    return {
        "lock_id": "hammer-environment:sha256:fixture",
        "itp": "lean",
        "itp_version": "4.19.0",
        "kernel_command_template": "lean {source}",
        "solver_versions": {solver: f"{solver}-pinned" for solver in solvers},
        "executable_paths": {
            "lean": "/opt/pinned/bin/lean",
            **{solver: f"/opt/pinned/bin/{solver}" for solver in solvers},
        },
        "os_info": "linux-x86_64-pinned",
        "container_digest": "sha256:environment",
    }


def _policy(**overrides):
    values = {
        "allowed_solvers": ("z3",),
        "timeout_ms": 20_000,
        "cpu_time_ms": 12_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_premises": 4,
        "max_parallel_processes": 2,
        "network_allowed": False,
        "environment_lock": _lock(),
        "fallback_checks": ("pytest:provider-fallback",),
    }
    values.update(overrides)
    return HammerSupervisorPolicy(**values)


def _obligation(**overrides):
    values = {
        "repository_id": "repo",
        "repository_tree_id": "tree:candidate",
        "ast_scope_ids": ("src/state.py::advance",),
        "statement": "(assert (not bad_transition))",
        "premise_ids": ("premise:relation", "premise:state"),
        "template_id": "legal-state-transitions",
        "template_version": "1.0.0",
        "template_semantic_hash": "sha256:template",
        "invariant_class": "state_transition",
        "task_id": "LPR-012",
        "fallback_checks": ("pytest:state-transitions",),
        "metadata": {
            "translation_family": "smtlib2",
            "statement_format": "smtlib2",
            "corpus_revision": "corpus:reviewed",
            "upstream_receipt_ids": ["receipt:obligation"],
            "goal_id": "goal:reviewed",
            "accepted_plan_id": "plan:reviewed",
            "assumptions_digest": "assumptions:reviewed",
            "scope_set_id": "scope-set:reviewed",
            "effect_scope_map": {
                "effect:advance": ["src/state.py::advance"],
            },
            "code_proof_toolchain_id": "toolchain:reviewed",
            "translation_map_id": "translation-map:exact",
        },
    }
    values.update(overrides)
    return CodeProofObligation(**values)


def _premises():
    return [
        {
            "premise_id": "premise:state",
            "statement": "The current state is ready.",
            "receipt_id": "receipt:state",
            "content_digest": "sha256:state",
        },
        {
            "premise_id": "premise:relation",
            "statement": "Ready may transition only to running.",
            "upstream_receipt_ids": ["receipt:relation"],
            "content_digest": "sha256:relation",
        },
    ]


def _roots(**overrides):
    values = {
        "repository_id": "repository:one",
        "objective_id": "objective:one",
        "trace_id": "trace:one",
        "change_id": "change:one",
        "consumer_id": "consumer:one",
        "forest_id": "forest:one",
        "tree_id": "tree:candidate",
        "overlay_id": "overlay:one",
        "graph_id": "graph:one",
        "index_id": "index:one",
        "corpus_id": "corpus:reviewed",
        "model_id": "model:none",
        "translator_id": "translator:one",
        "toolchain_id": "toolchain:reviewed",
        "policy_id": "policy:one",
        "environment_id": "hammer-environment:sha256:fixture",
    }
    values.update(overrides)
    return ProgramLogicAuthorityRoots(**values)


def _round_trip(
    *,
    obligation_id: str = "obligation:logic-ir",
    disposition: NativeGoalDisposition = NativeGoalDisposition.ROUND_TRIP_OK,
) -> SemanticRoundTripReceipt:
    return SemanticRoundTripReceipt(
        receipt_id="roundtrip:one",
        logic_ir_claim_id=obligation_id,
        native_statement_id="native-stmt:one",
        equivalence_method="statement_equivalence",
        disposition=disposition,
    )


def _native_binding(**overrides):
    roots = overrides.pop("roots", _roots())
    values = {
        "roots": roots,
        "binding_id": "binding:one",
        "logic_ir_obligation_id": "obligation:logic-ir",
        "premise_ids": ("premise:relation", "premise:state"),
        "native_itp_id": "itp:lean",
        "goal_snapshot_id": "goal-snapshot:exact",
        "native_theorem_source_id": "native-src:theorem",
        "proof_hole_id": "hole:single",
        "kernel_id": "kernel:lean4",
        "semantic_round_trip": _round_trip(),
        "disposition": NativeGoalDisposition.ROUND_TRIP_OK,
        "import_ids": ("import:Prelude",),
        "invalidation_refs": ("tree:candidate", "toolchain:reviewed"),
    }
    values.update(overrides)
    return ProgramLogicNativeGoalBinding(**values)


def _permit(**overrides):
    values = {
        "permit_id": "permit:coord",
        "operations": (
            NativeExecutionOperation.PORTFOLIO,
            NativeExecutionOperation.SOLVER,
            NativeExecutionOperation.RECONSTRUCTION,
            NativeExecutionOperation.KERNEL,
            NativeExecutionOperation.COUNTERMODEL_REPLAY,
        ),
        "environment_lock_id": "hammer-environment:sha256:fixture",
        "lane": NativeExecutionLane.SUPERVISED,
        "allowed_solvers": ("z3",),
    }
    values.update(overrides)
    return NativeExecutionPermit(**values)


def _candidate_runner(status="candidate"):
    def runner(invocation):
        attempt_id = f"{invocation.bundle.request_id}:translation:z3:0"
        base = {
            "request_id": invocation.bundle.request_id,
            "status": status,
            "attempts": [
                {
                    "attempt_id": attempt_id,
                    "request_id": invocation.bundle.request_id,
                    "translation_id": invocation.translations[0].translation_id,
                    "solver_name": "z3",
                }
            ],
        }
        if status == "candidate":
            base["proof_candidate"] = {
                "candidate_id": "candidate:1",
                "request_id": invocation.bundle.request_id,
                "solver_attempt_id": attempt_id,
                "premise_ids": ["premise:relation"],
            }
        if status == "counterexample":
            base["counterexample"] = {
                "solver_countermodel_id": "solver-cm:1",
                "raw_diagnostic_refs": ["diag:model"],
            }
        return base

    return runner


def _coordinator(tmp_path, *, portfolio_runner=None, policy=None, **gate_kwargs):
    policy = policy or _policy()
    provider = IpfsDatasetsLogicProvider(
        policy,
        portfolio_runner=portfolio_runner or _candidate_runner(),
    )
    gate = NativeExecutionAuthorizationGate(
        default_permit=NativeExecutionPermit.disabled(),
        supervisor_policy=ResourcePolicySlice(
            allowed_solvers=tuple(policy.allowed_solvers),
            timeout_ms=policy.timeout_ms,
            cpu_time_ms=policy.cpu_time_ms,
            memory_bytes=policy.memory_bytes,
            max_premises=policy.max_premises,
            max_parallel_processes=policy.max_parallel_processes,
            network_allowed=policy.network_allowed,
            native_execution_allowed=True,
        ),
        provider_policy=ResourcePolicySlice(
            allowed_solvers=tuple(policy.allowed_solvers),
            timeout_ms=policy.timeout_ms,
            cpu_time_ms=policy.cpu_time_ms,
            memory_bytes=policy.memory_bytes,
            max_premises=policy.max_premises,
            max_parallel_processes=policy.max_parallel_processes,
            network_allowed=policy.network_allowed,
        ),
        resource_enforcement=gate_kwargs.pop(
            "resource_enforcement", probe_resource_enforcement()
        ),
        **gate_kwargs,
    )
    return TacticianHammerCoordinator(
        provider=provider,
        gate=gate,
        receipt_store_dir=tmp_path / "receipts",
    )


def test_import_isolation_hardened_without_home_or_prefix_mutation():
    import inspect

    from ipfs_accelerate_py.agent_supervisor.integrations import (
        ipfs_datasets_logic_provider as logic_provider,
    )

    assert HAMMER_IMPORT_ISOLATION == HAMMER_IMPORT_ISOLATION_HARDENED
    loader = get_isolated_hammer_loader()
    report = loader.isolation_report()
    assert report["import_isolation"] == HAMMER_IMPORT_ISOLATION_HARDENED
    assert report["mutates_home"] is False
    assert report["mutates_sys_prefix"] is False
    assert report["concurrency_safe"] is True

    # Adapter _load_hammer must not itself rewrite HOME/sys.prefix (LPR-012).
    source = inspect.getsource(logic_provider._load_hammer)
    assert 'os.environ["HOME"]' not in source
    assert "sys.prefix =" not in source
    loader_source = inspect.getsource(IsolatedHammerLoader.load)
    # The load path may restore globals after transitive imports, but must not
    # publish a temporary HOME swap of its own.
    assert 'os.environ["HOME"] = import_root' not in loader_source
    assert "os.environ[\"HOME\"] = import_root" not in loader_source

    original_home = os.environ.get("HOME")
    original_prefix = __import__("sys").prefix

    # First import is serialized under the loader lock; complete it before
    # concurrent observation so the hot path is a cache hit with no global swap.
    module = loader.load()
    assert module is not None
    assert os.environ.get("HOME") == original_home
    assert __import__("sys").prefix == original_prefix

    observed = {"home": [], "prefix": []}
    ready = threading.Event()
    stop = threading.Event()

    def watcher():
        ready.set()
        while not stop.is_set():
            observed["home"].append(os.environ.get("HOME"))
            observed["prefix"].append(__import__("sys").prefix)
            # Busy-sample so a short concurrent window still leaves traces.
            for _ in range(50):
                if stop.is_set():
                    break

    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    try:
        # Concurrent loads must not expose process-global HOME/sys.prefix swaps.
        for _ in range(20):
            assert loader.load() is module
            assert get_isolated_hammer_loader().load() is module
            observed["home"].append(os.environ.get("HOME"))
            observed["prefix"].append(__import__("sys").prefix)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert os.environ.get("HOME") == original_home
    assert __import__("sys").prefix == original_prefix
    assert observed["home"], "watcher should sample during concurrent loads"
    assert all(h == original_home for h in observed["home"])
    assert all(p == original_prefix for p in observed["prefix"])


def test_default_gate_denies_without_permit(tmp_path):
    coord = _coordinator(tmp_path)
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        environment_lock=_lock(),
        permit=None,
        persist=False,
    )
    assert receipt.outcome is HammerCoordinationOutcome.POLICY_DENIED
    assert receipt.conclusiveness is CoordinationConclusiveness.NON_CONCLUSIVE
    assert receipt.proof_success is False
    assert "native_execution_disabled_by_default" in receipt.reason_codes


def test_coordinate_candidate_is_non_conclusive_and_persists_receipt(tmp_path):
    coord = _coordinator(tmp_path)
    binding = _native_binding()
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        translation_map_id="translation-map:exact",
        native_goal_binding=binding,
        roots=_roots(),
        persist=True,
    )
    assert receipt.outcome is HammerCoordinationOutcome.CANDIDATE
    assert receipt.conclusiveness is CoordinationConclusiveness.NON_CONCLUSIVE
    assert receipt.proof_success is False
    assert receipt.kernel_checked is False
    assert receipt.translation_map_id == "translation-map:exact"
    assert receipt.native_goal_binding_id == "binding:one"
    assert receipt.selector_mode is PremiseSelectorMode.DETERMINISTIC
    assert receipt.import_isolation == HAMMER_IMPORT_ISOLATION_HARDENED
    assert receipt.receipt_binding is not None
    assert receipt.gate_decision["authorized"] is True
    assert receipt.policy_intersection["network_false_is_metadata_unless_os_isolation"] is True
    path = Path(receipt.receipt_binding["persisted_path"])
    assert path.is_file()
    assert "candidate" in path.read_text(encoding="utf-8")


def test_verified_requires_matching_native_kernel_reconstruction(tmp_path):
    # Provider candidate without kernel reconstruction cannot become verified
    # via status spoofing on the portfolio runner.
    def spoofed(invocation):
        attempt_id = f"{invocation.bundle.request_id}:translation:z3:0"
        return {
            "request_id": invocation.bundle.request_id,
            "status": "verified",  # untrusted if no kernel path
            "attempts": [
                {
                    "attempt_id": attempt_id,
                    "request_id": invocation.bundle.request_id,
                    "translation_id": invocation.translations[0].translation_id,
                    "solver_name": "z3",
                }
            ],
            "proof_candidate": {
                "candidate_id": "candidate:spoof",
                "request_id": invocation.bundle.request_id,
                "solver_attempt_id": attempt_id,
                "premise_ids": ["premise:relation"],
            },
        }

    # adapt_hammer_result rejects candidate+non-candidate status mismatch;
    # "verified" with a candidate raises.  Use plain verified without candidate.
    def spoofed_verified(invocation):
        return {
            "request_id": invocation.bundle.request_id,
            "status": "verified",
            "attempts": [],
            "proof_candidate": None,
        }

    coord = _coordinator(tmp_path, portfolio_runner=spoofed_verified)
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        persist=False,
    )
    # Without kernel_checked + kernel_verified assurance, not conclusive proof.
    assert receipt.outcome is not HammerCoordinationOutcome.VERIFIED or (
        receipt.proof_success and receipt.kernel_checked
    )
    if receipt.outcome is HammerCoordinationOutcome.VERIFIED:
        assert receipt.kernel_checked is True
        assert receipt.proof_success is True
    else:
        assert receipt.conclusiveness is not CoordinationConclusiveness.CONCLUSIVE_PROOF


def test_raw_countermodel_is_diagnostic_until_replay(tmp_path):
    coord = _coordinator(
        tmp_path, portfolio_runner=_candidate_runner("counterexample")
    )
    # Raw only → diagnostic, non-conclusive for rejection authority.
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        translation_map_id="translation-map:exact",
        roots=_roots(),
        countermodel_raw={
            "solver_countermodel_id": "solver-cm:raw",
            "raw_diagnostic_refs": ["diag:assignment"],
        },
        persist=False,
    )
    assert receipt.outcome is HammerCoordinationOutcome.COUNTEREXAMPLE
    assert receipt.conclusiveness is CoordinationConclusiveness.DIAGNOSTIC
    assert receipt.countermodel_validation is not None
    assert (
        receipt.countermodel_validation["disposition"]
        == CountermodelDisposition.DIAGNOSTIC_ONLY.value
    )

    # Deterministic LogicIR replay → validated refutation.
    validated = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        translation_map_id="translation-map:exact",
        roots=_roots(),
        countermodel_raw={
            "solver_countermodel_id": "solver-cm:raw",
            "raw_diagnostic_refs": ["diag:assignment"],
        },
        countermodel_replay={
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:logic-ir",
        },
        persist=False,
    )
    assert validated.outcome is HammerCoordinationOutcome.COUNTEREXAMPLE
    assert (
        validated.conclusiveness
        is CoordinationConclusiveness.CONCLUSIVE_REFUTATION
    )
    assert (
        validated.countermodel_validation["disposition"]
        == CountermodelDisposition.VALIDATED.value
    )


def test_stale_cross_root_timeout_denial_unavailable_are_non_conclusive(tmp_path):
    coord = _coordinator(tmp_path)

    stale = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        expected_tree_id="tree:other",
        persist=False,
    )
    assert stale.outcome is HammerCoordinationOutcome.STALE
    assert stale.conclusiveness is CoordinationConclusiveness.NON_CONCLUSIVE

    denied = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        environment_lock=_lock(),
        persist=False,
    )
    assert denied.outcome is HammerCoordinationOutcome.POLICY_DENIED
    assert denied.conclusiveness is CoordinationConclusiveness.NON_CONCLUSIVE

    def time_out(_invocation):
        raise TimeoutError("fixture deadline")

    timed = _coordinator(tmp_path, portfolio_runner=time_out)
    timeout_receipt = timed.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        persist=False,
    )
    assert timeout_receipt.outcome is HammerCoordinationOutcome.TIMEOUT
    assert (
        timeout_receipt.conclusiveness
        is CoordinationConclusiveness.NON_CONCLUSIVE
    )


def test_learned_selector_is_opt_in_pinned_and_ranking_only(tmp_path):
    coord = _coordinator(tmp_path)
    # Opt-in without digest is denied.
    denied = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        selector_mode=PremiseSelectorMode.LEARNED_RANKING_ONLY,
        learned_selector_model_digest="",
        persist=False,
    )
    assert denied.outcome is HammerCoordinationOutcome.POLICY_DENIED
    assert "learned_selector_requires_pinned_model_digest" in denied.reason_codes

    # Deterministic remains the default.
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        persist=False,
    )
    assert receipt.selector_mode is PremiseSelectorMode.DETERMINISTIC


def test_autonomous_lane_blocked_when_cpu_memory_unenforced(tmp_path):
    weak = ResourceEnforcementReport(
        platform="windows",
        cpu_enforcement=ResourceEnforcementStrength.UNSUPPORTED,
        memory_enforcement=ResourceEnforcementStrength.UNSUPPORTED,
        process_isolation=ResourceEnforcementStrength.PARTIAL,
    )
    coord = _coordinator(tmp_path, resource_enforcement=weak)
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(lane=NativeExecutionLane.AUTONOMOUS),
        environment_lock=_lock(),
        persist=False,
    )
    assert receipt.outcome is HammerCoordinationOutcome.POLICY_DENIED
    assert any(
        "autonomous_lane" in code or "unenforced" in code or "cpu" in code
        for code in receipt.reason_codes
    )


def test_outcome_vocabulary_is_exact():
    assert COORDINATION_OUTCOMES == {
        "verified",
        "candidate",
        "counterexample",
        "timeout",
        "unsupported",
        "unavailable",
        "policy_denied",
        "unknown",
        "stale",
        "error",
    }
    assert (
        map_provider_status_to_outcome(
            "verified",
            proof_success=True,
            kernel_checked=True,
            authoritative_assurance="kernel_verified",
        )
        is HammerCoordinationOutcome.VERIFIED
    )
    assert (
        map_provider_status_to_outcome("timed_out")
        is HammerCoordinationOutcome.TIMEOUT
    )
    assert (
        map_provider_status_to_outcome("policy_denied")
        is HammerCoordinationOutcome.POLICY_DENIED
    )


def test_cancellation_cleans_temps_without_leak(tmp_path):
    coord = _coordinator(tmp_path)
    owned = coord._owned_tempdir("cancel-test-")
    assert Path(owned).is_dir()
    coord.cancel()
    assert coord.cancelled is True
    assert not Path(owned).exists()
    # Subsequent coordinate is non-conclusive cancelled.
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        persist=False,
    )
    assert receipt.cancelled is True
    assert receipt.conclusiveness is CoordinationConclusiveness.NON_CONCLUSIVE


def test_countermodel_validator_separates_diagnostic_and_validated():
    validator = CountermodelValidator()
    roots = _roots()
    diagnostic = validator.validate(
        roots=roots,
        solver_countermodel_id="solver-cm:1",
        translation_map_id="translation-map:exact",
        originating_logic_ir_id="obligation:logic-ir",
        raw_diagnostic_refs=("diag:model",),
        invalidation_refs=("tree:candidate",),
    )
    assert diagnostic.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY
    assert diagnostic.may_reject_hypothesis is False

    validated = validator.validate(
        roots=roots,
        solver_countermodel_id="solver-cm:1",
        translation_map_id="translation-map:exact",
        originating_logic_ir_id="obligation:logic-ir",
        raw_diagnostic_refs=("diag:model",),
        replay_result={
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:1",
        },
        invalidation_refs=("tree:candidate",),
    )
    assert validated.disposition is CountermodelDisposition.VALIDATED
    assert validated.may_reject_hypothesis is True

    by_negation = validator.validate(
        roots=roots,
        solver_countermodel_id="solver-cm:1",
        translation_map_id="translation-map:exact",
        originating_logic_ir_id="obligation:logic-ir",
        proof_of_negation_id="proof-neg:1",
        invalidation_refs=("tree:candidate",),
    )
    assert by_negation.disposition is CountermodelDisposition.VALIDATED


def test_provider_declares_hardened_isolation_and_extends_not_bypasses():
    provider = IpfsDatasetsLogicProvider(_policy())
    caps = provider.capabilities().to_dict()
    assert caps["metadata"]["import_isolation"] == HAMMER_IMPORT_ISOLATION_HARDENED
    assert caps["metadata"]["deterministic_selector_default"] is True
    assert caps["metadata"]["learned_selector_default"] is False
    # Factory still wires the production provider type.
    coord = create_tactician_hammer_coordinator(
        policy=_policy(),
        permit=_permit(),
        portfolio_runner=_candidate_runner(),
    )
    assert isinstance(coord.provider, IpfsDatasetsLogicProvider)
    assert isinstance(coord.loader, IsolatedHammerLoader)


def test_extends_provider_with_translation_map_in_prove_path(tmp_path):
    captured = []

    def runner(invocation):
        captured.append(invocation)
        attempt_id = f"{invocation.bundle.request_id}:translation:z3:0"
        return {
            "request_id": invocation.bundle.request_id,
            "status": "candidate",
            "attempts": [
                {
                    "attempt_id": attempt_id,
                    "request_id": invocation.bundle.request_id,
                    "translation_id": invocation.translations[0].translation_id,
                    "solver_name": "z3",
                }
            ],
            "proof_candidate": {
                "candidate_id": "candidate:map",
                "request_id": invocation.bundle.request_id,
                "solver_attempt_id": attempt_id,
                "premise_ids": ["premise:relation"],
            },
        }

    coord = _coordinator(tmp_path, portfolio_runner=runner)
    receipt = coord.coordinate(
        obligation=_obligation(),
        premises=_premises(),
        permit=_permit(),
        environment_lock=_lock(),
        translation_map_id="translation-map:exact",
        native_goal_binding=_native_binding(),
        roots=_roots(),
        persist=False,
    )
    assert receipt.outcome is HammerCoordinationOutcome.CANDIDATE
    assert receipt.provider_result.get("translation_map_id") == "translation-map:exact"
    assert receipt.provider_result.get("native_goal_binding") is not None
    assert len(captured) == 1
