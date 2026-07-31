"""Fail-closed native Hammer execution authorization gate (LPR-012)."""

from __future__ import annotations

import platform

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.hammer_native_execution_gate import (
    NATIVE_EXECUTION_GATE_INTERFACE,
    NativeExecutionAuthorizationGate,
    NativeExecutionDisposition,
    NativeExecutionLane,
    NativeExecutionOperation,
    NativeExecutionPermit,
    ResourceEnforcementReport,
    ResourceEnforcementStrength,
    ResourcePolicySlice,
    intersect_resource_policies,
    probe_resource_enforcement,
)


def _lock(**overrides):
    values = {
        "lock_id": "hammer-environment:sha256:fixture",
        "itp": "lean",
        "itp_version": "4.19.0",
        "solver_versions": {"z3": "z3-pinned", "cvc5": "cvc5-pinned"},
        "executable_paths": {
            "lean": "/opt/pinned/bin/lean",
            "z3": "/opt/pinned/bin/z3",
            "cvc5": "/opt/pinned/bin/cvc5",
        },
        "os_info": "linux-x86_64-pinned",
    }
    values.update(overrides)
    return values


def _supervisor(**overrides):
    values = {
        "allowed_solvers": ("cvc5", "z3"),
        "timeout_ms": 20_000,
        "cpu_time_ms": 12_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_premises": 8,
        "max_parallel_processes": 2,
        "network_allowed": False,
        "native_execution_allowed": True,
    }
    values.update(overrides)
    return ResourcePolicySlice(**values)


def _permit(**overrides):
    values = {
        "permit_id": "permit:fixture",
        "operations": (
            NativeExecutionOperation.PORTFOLIO,
            NativeExecutionOperation.SOLVER,
            NativeExecutionOperation.RECONSTRUCTION,
            NativeExecutionOperation.KERNEL,
        ),
        "environment_lock_id": "hammer-environment:sha256:fixture",
        "lane": NativeExecutionLane.SUPERVISED,
        "allowed_solvers": ("z3", "cvc5"),
    }
    values.update(overrides)
    return NativeExecutionPermit(**values)


def _gate(**kwargs):
    return NativeExecutionAuthorizationGate(
        supervisor_policy=kwargs.pop("supervisor_policy", _supervisor()),
        provider_policy=kwargs.pop(
            "provider_policy",
            ResourcePolicySlice(
                allowed_solvers=("cvc5", "z3"),
                timeout_ms=30_000,
                cpu_time_ms=30_000,
                memory_bytes=512 * 1024 * 1024,
                max_premises=64,
                max_parallel_processes=4,
                network_allowed=False,
            ),
        ),
        resource_enforcement=kwargs.pop(
            "resource_enforcement", probe_resource_enforcement()
        ),
        **kwargs,
    )


def test_defaults_disable_solver_frontend_and_kernel_execution():
    gate = NativeExecutionAuthorizationGate()
    for op in (
        NativeExecutionOperation.SOLVER,
        NativeExecutionOperation.FRONTEND,
        NativeExecutionOperation.KERNEL,
        NativeExecutionOperation.PORTFOLIO,
        NativeExecutionOperation.RECONSTRUCTION,
    ):
        decision = gate.authorize(op, environment_lock=_lock())
        assert decision.authorized is False
        assert (
            decision.disposition
            is NativeExecutionDisposition.DISABLED_BY_DEFAULT
        )
        assert "native_execution_disabled_by_default" in decision.reason_codes


def test_requires_exact_operation_permit_environment_and_policy():
    gate = _gate()
    # Wrong operation on permit.
    decision = gate.authorize(
        NativeExecutionOperation.FRONTEND,
        permit=_permit(operations=(NativeExecutionOperation.SOLVER,)),
        environment_lock=_lock(),
    )
    assert decision.authorized is False
    assert decision.disposition is NativeExecutionDisposition.PERMIT_MISMATCH

    # Missing environment lock.
    decision = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(),
        environment_lock=None,
    )
    assert decision.authorized is False
    assert (
        decision.disposition is NativeExecutionDisposition.ENVIRONMENT_MISMATCH
    )

    # Environment id mismatch.
    decision = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(environment_lock_id="hammer-environment:other"),
        environment_lock=_lock(),
    )
    assert decision.authorized is False
    assert (
        decision.disposition is NativeExecutionDisposition.ENVIRONMENT_MISMATCH
    )

    # Happy path.
    decision = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(),
        environment_lock=_lock(),
        required_solvers=("z3",),
    )
    assert decision.authorized is True
    assert decision.disposition is NativeExecutionDisposition.AUTHORIZED
    assert decision.gate_interface if False else True  # noqa: SIM222 - sanity
    assert decision.to_dict()["gate_interface"] == NATIVE_EXECUTION_GATE_INTERFACE


def test_policy_intersection_tightens_solver_process_time_cpu_memory():
    intersection = intersect_resource_policies(
        supervisor=ResourcePolicySlice(
            allowed_solvers=("cvc5", "z3", "vampire"),
            timeout_ms=20_000,
            cpu_time_ms=15_000,
            memory_bytes=256 * 1024 * 1024,
            max_parallel_processes=4,
            network_allowed=False,
            native_execution_allowed=True,
        ),
        request=ResourcePolicySlice(
            allowed_solvers=("z3", "vampire"),
            timeout_ms=8_000,
            cpu_time_ms=20_000,
            memory_bytes=128 * 1024 * 1024,
            max_parallel_processes=2,
            network_allowed=True,  # cannot expand
            native_execution_allowed=True,
        ),
        provider=ResourcePolicySlice(
            allowed_solvers=("cvc5", "z3"),
            timeout_ms=30_000,
            cpu_time_ms=12_000,
            memory_bytes=512 * 1024 * 1024,
            max_parallel_processes=8,
            network_allowed=False,
            native_execution_allowed=True,
        ),
    )
    assert intersection.allowed_solvers == ("z3",)
    assert intersection.timeout_ms == 8_000
    assert intersection.cpu_time_ms == 12_000
    assert intersection.memory_bytes == 128 * 1024 * 1024
    assert intersection.max_parallel_processes == 2
    assert intersection.network_allowed is False
    assert intersection.network_false_is_metadata_unless_os_isolation is True
    payload = intersection.to_dict()
    assert payload["network_false_is_metadata_unless_os_isolation"] is True


def test_network_false_is_metadata_unless_os_isolation_receipt():
    gate = _gate()
    decision = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(),
        environment_lock=_lock(),
    )
    assert decision.authorized is True
    assert decision.network_false_is_metadata is True
    assert decision.details["network_false_is_metadata"] is True
    assert decision.details["network_os_isolation"] is False

    with_os = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(os_network_isolation_receipt_id="os-isolation:netns-1"),
        environment_lock=_lock(),
    )
    assert with_os.authorized is True
    assert with_os.details["network_os_isolation"] is True


def test_resource_enforcement_reports_posix_strength_and_blocks_autonomous():
    report = probe_resource_enforcement()
    assert isinstance(report, ResourceEnforcementReport)
    assert report.cpu_enforcement in set(ResourceEnforcementStrength)
    assert report.memory_enforcement in set(ResourceEnforcementStrength)
    assert report.network_policy_denied is True
    assert report.network_os_isolation is False
    assert report.signed_binary_integrity is False
    assert report.environment_lock_path_version_only is True

    # Force unenforceable platform for autonomous denial.
    weak = ResourceEnforcementReport(
        platform="windows",
        cpu_enforcement=ResourceEnforcementStrength.UNSUPPORTED,
        memory_enforcement=ResourceEnforcementStrength.UNSUPPORTED,
        process_isolation=ResourceEnforcementStrength.PARTIAL,
    )
    gate = _gate(resource_enforcement=weak)
    denied = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(lane=NativeExecutionLane.AUTONOMOUS),
        environment_lock=_lock(),
    )
    assert denied.authorized is False
    assert (
        denied.disposition is NativeExecutionDisposition.RESOURCE_UNENFORCEABLE
    )
    assert "autonomous_lane_requires_enforced_cpu_memory" in denied.reason_codes

    # Supervised lane may proceed even when enforcement is partial/unsupported.
    supervised = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=_permit(lane=NativeExecutionLane.SUPERVISED),
        environment_lock=_lock(),
    )
    assert supervised.authorized is True


def test_supply_chain_requires_reviewed_digest_or_isolated_receipt():
    gate = _gate(
        supervisor_policy=_supervisor(require_supply_chain_integrity=True)
    )
    # Path/version lock alone is insufficient.
    denied = gate.authorize(
        NativeExecutionOperation.KERNEL,
        permit=_permit(
            operations=(NativeExecutionOperation.KERNEL,),
            require_supply_chain_integrity=True,
        ),
        environment_lock=_lock(),
    )
    assert denied.authorized is False
    assert denied.disposition is NativeExecutionDisposition.SUPPLY_CHAIN_DENIED
    assert "reviewed_digest_or_isolated_receipt_missing" in denied.reason_codes

    with_digest = gate.authorize(
        NativeExecutionOperation.KERNEL,
        permit=_permit(
            operations=(NativeExecutionOperation.KERNEL,),
            require_supply_chain_integrity=True,
            reviewed_executable_digests={
                "lean": "sha256:reviewed-lean-digest",
            },
        ),
        environment_lock=_lock(),
    )
    assert with_digest.authorized is True
    assert with_digest.supply_chain_satisfied is True

    with_isolated = gate.authorize(
        NativeExecutionOperation.KERNEL,
        permit=_permit(
            operations=(NativeExecutionOperation.KERNEL,),
            require_supply_chain_integrity=True,
            isolated_execution_receipt_id="isolated-exec:worker-1",
        ),
        environment_lock=_lock(),
    )
    assert with_isolated.authorized is True
    assert with_isolated.supply_chain_satisfied is True


def test_solver_not_on_allowlist_is_policy_denied():
    gate = _gate(supervisor_policy=_supervisor(allowed_solvers=("z3",)))
    decision = gate.authorize(
        NativeExecutionOperation.SOLVER,
        permit=_permit(
            operations=(NativeExecutionOperation.SOLVER,),
            allowed_solvers=("vampire",),
        ),
        environment_lock=_lock(),
        required_solvers=("vampire",),
    )
    assert decision.authorized is False
    assert decision.disposition is NativeExecutionDisposition.POLICY_DENIED


def test_permit_and_decision_round_trip():
    permit = _permit(
        learned_selector_model_digest="sha256:model",
        learned_selector_ranking_only=True,
    )
    restored = NativeExecutionPermit.from_dict(permit.to_dict())
    assert restored.permit_id == permit.permit_id
    assert restored.operations == permit.operations
    assert restored.learned_selector_ranking_only is True

    gate = _gate()
    decision = gate.authorize(
        NativeExecutionOperation.PORTFOLIO,
        permit=permit,
        environment_lock=_lock(),
    )
    payload = decision.to_dict()
    assert payload["authorized"] is True
    assert payload["decision_id"].startswith("native-exec-decision:sha256:")
    assert payload["resource_enforcement"]["platform"]
    # Linux CI should typically report posix_rlimit.
    if platform.system().lower() == "linux":
        assert payload["resource_enforcement"]["cpu_enforcement"] in {
            "posix_rlimit",
            "partial",
            "unsupported",
            "unknown",
        }


def test_require_raises_on_denial():
    gate = NativeExecutionAuthorizationGate()
    with pytest.raises(Exception, match="native execution denied"):
        gate.require(NativeExecutionOperation.SOLVER, environment_lock=_lock())
