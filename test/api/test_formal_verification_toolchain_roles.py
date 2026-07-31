"""Executable contract for role-aware toolchain authority (FVT-037 / FVT-G100).

Validates ``FormalVerificationToolRole@1`` and ``RoleAwarePromotionPolicy@1``:

* every matrix entry has exactly one closed role and authority ceiling;
* Java, Maude, and OPAM are support only;
* Leanstral, autoencoder, SymAI, ErgoAI, and Hammer are advisor/candidate only
  until independent reconstruction;
* external Souffle/SecPAL are shadow checkers;
* in-process Datalog/SecPAL have authorization-only authority;
* Runtime MTL has finite-trace authority;
* state and hyperproperty tools have bounded authority;
* Lean/Rocq/Isabelle have kernel authority;
* ZKP has attestation authority only;
* support, advisor, or shadow presence alone can never satisfy a
  certified-authority requirement;
* semantic lanes are pre-registered for independent ownership.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES_MODULE_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "toolchain_roles.py"
)
CERT_ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"

INTERFACE = "FormalVerificationToolRole@1"
PROMOTION_INTERFACE = "RoleAwarePromotionPolicy@1"
GOAL_ID = "FVT-G100"
TASK_ID = "FVT-037"

SUPPORT_ONLY = frozenset({"java", "maude", "opam"})
ADVISOR_OR_CANDIDATE = frozenset(
    {"symbolicai", "ergoai", "leanstral", "autoencoder", "hammer"}
)
SHADOW_CHECKERS = frozenset({"souffle", "secpal"})
AUTHORIZATION_ONLY = frozenset({"datalog-authorization", "secpal-authorization"})
FINITE_TRACE = frozenset({"runtime-mtl", "runtime-mtl-external"})
BOUNDED_STATE = frozenset({"apalache", "tlc"})
BOUNDED_HYPER = frozenset({"hyperltl", "autohyper", "mchyper"})
KERNEL = frozenset({"lean", "coq", "isabelle"})
ATTESTATION = frozenset({"zkp-circuit"})

REQUIRED_LANES = frozenset(
    {
        "smt",
        "tla",
        "datalog_secpal",
        "protocol",
        "hyperltl",
        "atp",
        "hammer",
        "kernel",
        "runtime_mtl",
        "attestation",
        "support",
    }
)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    # Ensure ipfs_datasets_py package root is importable.
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def roles_mod():
    return _load_module(ROLES_MODULE_PATH, "toolchain_roles_under_test")


@pytest.fixture(scope="module")
def cert_roles_mod(roles_mod):
    # roles_mod must load first so package imports resolve consistently.
    assert roles_mod is not None
    return _load_module(CERT_ROLES_PATH, "certification_roles_under_test")


@pytest.fixture
def policy(roles_mod):
    roles_mod.reset_default_policy()
    return roles_mod.default_promotion_policy()


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert ROLES_MODULE_PATH.is_file()
    assert CERT_ROLES_PATH.is_file()
    assert Path(__file__).is_file()


def test_interfaces_and_goal_identity(roles_mod, cert_roles_mod) -> None:
    assert roles_mod.FORMAL_VERIFICATION_TOOL_ROLE_INTERFACE == INTERFACE
    assert roles_mod.ROLE_AWARE_PROMOTION_POLICY_INTERFACE == PROMOTION_INTERFACE
    assert roles_mod.GOAL_ID == GOAL_ID
    assert roles_mod.TASK_ID == TASK_ID
    assert cert_roles_mod.INTERFACE == INTERFACE
    assert cert_roles_mod.PROMOTION_INTERFACE == PROMOTION_INTERFACE
    assert cert_roles_mod.GOAL_ID == GOAL_ID
    assert roles_mod.role_matrix_side_effect_free_on_import() is True


# ---------------------------------------------------------------------------
# Matrix structure: exactly one closed role + ceiling per tool
# ---------------------------------------------------------------------------


def test_every_matrix_entry_has_exactly_one_closed_role_and_ceiling(
    policy, roles_mod
) -> None:
    policy.assert_matrix_invariants()
    tool_ids = policy.list_tool_ids()
    assert tool_ids, "role matrix must not be empty"
    seen: set[str] = set()
    for tool_id in tool_ids:
        assert tool_id not in seen
        seen.add(tool_id)
        assignment = policy.get_role(tool_id)
        assert isinstance(assignment.role, roles_mod.ToolRole)
        assert isinstance(
            assignment.authority_ceiling, roles_mod.ToolchainAuthorityCeiling
        )
        # Closed vocabularies — no free-form strings.
        assert assignment.role.value in {item.value for item in roles_mod.ToolRole}
        assert assignment.authority_ceiling.value in {
            item.value for item in roles_mod.ToolchainAuthorityCeiling
        }
        payload = assignment.to_dict()
        assert payload["role"] == assignment.role.value
        assert payload["authority_ceiling"] == assignment.authority_ceiling.value
        assert payload["can_satisfy_certified_authority"] is (
            assignment.can_satisfy_certified_authority
        )
        # Round-trip preserves closed enums.
        restored = roles_mod.FormalVerificationToolRole.from_dict(payload)
        assert restored.tool_id == assignment.tool_id
        assert restored.role is assignment.role
        assert restored.authority_ceiling is assignment.authority_ceiling


def test_matrix_covers_required_acceptance_tools(policy) -> None:
    tool_ids = set(policy.list_tool_ids())
    required = (
        SUPPORT_ONLY
        | ADVISOR_OR_CANDIDATE
        | SHADOW_CHECKERS
        | AUTHORIZATION_ONLY
        | FINITE_TRACE
        | BOUNDED_STATE
        | BOUNDED_HYPER
        | KERNEL
        | ATTESTATION
        | {"z3", "cvc5", "vampire", "eprover", "tamarin", "proverif"}
    )
    missing = required - tool_ids
    assert not missing, f"missing matrix tools: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Acceptance-specific role and ceiling assignments
# ---------------------------------------------------------------------------


def test_java_maude_opam_are_support_only(policy, roles_mod) -> None:
    for tool_id in SUPPORT_ONLY:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.SUPPORT
        assert (
            assignment.authority_ceiling is roles_mod.ToolchainAuthorityCeiling.NONE
        )
        assert assignment.can_satisfy_certified_authority is False


def test_advisors_and_candidates_until_reconstruction(policy, roles_mod) -> None:
    for tool_id in ADVISOR_OR_CANDIDATE:
        assignment = policy.get_role(tool_id)
        assert assignment.role in {
            roles_mod.ToolRole.ADVISOR,
            roles_mod.ToolRole.CANDIDATE,
        }
        assert assignment.authority_ceiling in {
            roles_mod.ToolchainAuthorityCeiling.ADVISORY,
            roles_mod.ToolchainAuthorityCeiling.CANDIDATE,
            roles_mod.ToolchainAuthorityCeiling.NONE,
        }
        assert assignment.can_satisfy_certified_authority is False
        assert assignment.independent_reconstruction_required is True


def test_external_souffle_secpal_are_shadow_checkers(policy, roles_mod) -> None:
    for tool_id in SHADOW_CHECKERS:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.SHADOW
        assert assignment.can_satisfy_certified_authority is False


def test_in_process_datalog_secpal_authorization_only(policy, roles_mod) -> None:
    for tool_id in AUTHORIZATION_ONLY:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.AUTHORITY
        assert (
            assignment.authority_ceiling
            is roles_mod.ToolchainAuthorityCeiling.AUTHORIZATION
        )
        assert assignment.can_satisfy_certified_authority is True


def test_runtime_mtl_finite_trace_authority(policy, roles_mod) -> None:
    for tool_id in FINITE_TRACE:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.AUTHORITY
        assert (
            assignment.authority_ceiling
            is roles_mod.ToolchainAuthorityCeiling.FINITE_TRACE
        )
        assert assignment.can_satisfy_certified_authority is True


def test_state_and_hyperproperty_bounded_authority(policy, roles_mod) -> None:
    for tool_id in BOUNDED_STATE | BOUNDED_HYPER:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.AUTHORITY
        assert (
            assignment.authority_ceiling
            is roles_mod.ToolchainAuthorityCeiling.BOUNDED
        )
        assert assignment.can_satisfy_certified_authority is True


def test_lean_rocq_isabelle_kernel_authority(policy, roles_mod) -> None:
    for tool_id in KERNEL:
        assignment = policy.get_role(tool_id)
        assert assignment.role is roles_mod.ToolRole.AUTHORITY
        assert (
            assignment.authority_ceiling
            is roles_mod.ToolchainAuthorityCeiling.KERNEL
        )
        assert assignment.can_satisfy_certified_authority is True


def test_zkp_attestation_authority_only(policy, roles_mod) -> None:
    assignment = policy.get_role("zkp-circuit")
    assert assignment.role is roles_mod.ToolRole.AUTHORITY
    assert (
        assignment.authority_ceiling
        is roles_mod.ToolchainAuthorityCeiling.ATTESTATION
    )
    assert assignment.can_satisfy_certified_authority is True


# ---------------------------------------------------------------------------
# Promotion policy: non-certifying presence can never promote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_id",
    sorted(SUPPORT_ONLY | ADVISOR_OR_CANDIDATE | SHADOW_CHECKERS),
)
def test_support_advisor_shadow_presence_never_certifies(
    policy, tool_id: str
) -> None:
    decision = policy.evaluate_promotion(
        tool_id,
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
        independent_reconstruction=True,
    )
    assert decision.allowed is False
    assert decision.can_satisfy_certified_authority is False
    assert "cannot_satisfy_certified_authority_requirement" in decision.reason_codes


def test_authority_tool_promotes_only_with_hermetic_certificate(policy) -> None:
    denied = policy.evaluate_promotion(
        "lean",
        present=True,
        usable=True,
        production_certified=False,
        hermetic_certificate=False,
    )
    assert denied.allowed is False
    assert denied.can_satisfy_certified_authority is True
    assert "production_certification_required" in denied.reason_codes

    allowed = policy.evaluate_promotion(
        "lean",
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
    )
    assert allowed.allowed is True
    assert "role_aware_authority_satisfied" in allowed.reason_codes


def test_role_can_satisfy_helper_matches_matrix(roles_mod) -> None:
    assert (
        roles_mod.role_can_satisfy_certified_authority(
            roles_mod.ToolRole.SUPPORT,
            roles_mod.ToolchainAuthorityCeiling.NONE,
        )
        is False
    )
    assert (
        roles_mod.role_can_satisfy_certified_authority(
            roles_mod.ToolRole.ADVISOR,
            roles_mod.ToolchainAuthorityCeiling.ADVISORY,
        )
        is False
    )
    assert (
        roles_mod.role_can_satisfy_certified_authority(
            roles_mod.ToolRole.SHADOW,
            roles_mod.ToolchainAuthorityCeiling.NONE,
        )
        is False
    )
    assert (
        roles_mod.role_can_satisfy_certified_authority(
            roles_mod.ToolRole.AUTHORITY,
            roles_mod.ToolchainAuthorityCeiling.KERNEL,
        )
        is True
    )


def test_inconsistent_authority_role_rejected(roles_mod) -> None:
    with pytest.raises(roles_mod.ToolchainRoleError):
        roles_mod.FormalVerificationToolRole(
            tool_id="bad-support",
            role=roles_mod.ToolRole.SUPPORT,
            authority_ceiling=roles_mod.ToolchainAuthorityCeiling.KERNEL,
            lane_ids=("support",),
        )
    with pytest.raises(roles_mod.ToolchainRoleError):
        roles_mod.FormalVerificationToolRole(
            tool_id="bad-advisor",
            role=roles_mod.ToolRole.ADVISOR,
            authority_ceiling=roles_mod.ToolchainAuthorityCeiling.KERNEL,
            lane_ids=("hammer",),
        )


# ---------------------------------------------------------------------------
# Semantic lane registration / independent ownership
# ---------------------------------------------------------------------------


def test_semantic_lanes_pre_registered(policy) -> None:
    lane_ids = set(policy.list_lane_ids())
    assert REQUIRED_LANES <= lane_ids
    for lane_id in REQUIRED_LANES:
        lane = policy.lanes[lane_id]
        assert lane.owner_module.startswith("tools.logic.certification.")
        assert lane.handler_id
        assert set(lane.authority_tool_ids).issubset(lane.tool_ids)


def test_hammer_and_support_lanes_have_empty_authority_sets(policy) -> None:
    assert policy.lanes["hammer"].authority_tool_ids == ()
    assert policy.lanes["support"].authority_tool_ids == ()


def test_certification_surface_registers_lane_handlers(cert_roles_mod) -> None:
    policy = cert_roles_mod.build_role_aware_policy()
    registered = set(policy.registered_handler_ids())
    assert REQUIRED_LANES <= registered
    # Placeholder handlers are fail-closed (never certify).
    result = policy.get_lane_handler("kernel")()
    assert result["certified"] is False
    assert "lane_handler_pending_implementation" in result["reason_codes"]


def test_bind_lane_handler_replaces_placeholder(cert_roles_mod) -> None:
    policy = cert_roles_mod.build_role_aware_policy()

    def _real_kernel_handler(**kwargs: Any) -> dict[str, Any]:
        return {"certified": True, "lane_id": "kernel", "kwargs": kwargs}

    cert_roles_mod.bind_lane_handler(
        "kernel", _real_kernel_handler, policy=policy, replace=True
    )
    outcome = policy.get_lane_handler("kernel")(sample=True)
    assert outcome["certified"] is True
    assert outcome["kwargs"]["sample"] is True


def test_authority_boundary_report(cert_roles_mod) -> None:
    report = cert_roles_mod.authority_boundary_report()
    assert report["interface"] == PROMOTION_INTERFACE
    assert report["goal_id"] == GOAL_ID
    assert set(report["support_only"]) == SUPPORT_ONLY
    assert set(report["advisor_or_candidate_only"]) == ADVISOR_OR_CANDIDATE
    assert set(report["shadow_checkers"]) == SHADOW_CHECKERS
    assert set(report["authorization_only"]) == AUTHORIZATION_ONLY
    assert set(report["finite_trace_authority"]) == FINITE_TRACE
    assert BOUNDED_STATE | BOUNDED_HYPER <= set(report["bounded_authority"])
    assert set(report["kernel_authority"]) == KERNEL
    assert set(report["attestation_authority_only"]) == ATTESTATION
    assert report["policy"]["support_advisor_shadow_presence_cannot_certify"] is True
    for evaluation in report["non_certifying_presence_evaluations"]:
        assert evaluation["allowed"] is False


def test_role_matrix_receipt_is_serializable(cert_roles_mod) -> None:
    receipt = cert_roles_mod.build_role_matrix_receipt()
    assert receipt["interface"] == PROMOTION_INTERFACE
    assert receipt["goal_id"] == GOAL_ID
    assert isinstance(receipt["roles"], list) and receipt["roles"]
    assert isinstance(receipt["lanes"], list) and receipt["lanes"]
    assert "authority_boundary" in receipt
    # Ensure JSON-friendly primitives only.
    import json

    encoded = json.dumps(receipt)
    assert INTERFACE in encoded or PROMOTION_INTERFACE in encoded


def test_package_import_path_matches_module(roles_mod) -> None:
    """Prefer package import when the datasets tree is on sys.path."""

    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    if str(datasets_root) not in sys.path:
        sys.path.insert(0, str(datasets_root))
    from ipfs_datasets_py.logic.backends import toolchain_roles as pkg

    assert (
        pkg.FORMAL_VERIFICATION_TOOL_ROLE_INTERFACE
        == roles_mod.FORMAL_VERIFICATION_TOOL_ROLE_INTERFACE
    )
    pkg.reset_default_policy()
    assignment = pkg.get_tool_role("lean")
    assert assignment.authority_ceiling is pkg.ToolchainAuthorityCeiling.KERNEL
