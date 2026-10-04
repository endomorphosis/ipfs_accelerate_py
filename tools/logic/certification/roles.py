#!/usr/bin/env python3
"""Role-aware toolchain authority and promotion certification surface.

``FormalVerificationToolRole@1`` / ``RoleAwarePromotionPolicy@1`` (FVT-G100 / FVT-037).

Owns the certification-side view of the closed role matrix and pre-registers
per-lane handlers so later semantic certification tasks bind under
``tools.logic.certification.<lane>`` without editing the central multi-prover
certifier or the generated certificate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    FORMAL_VERIFICATION_TOOL_ROLE_INTERFACE,
    GOAL_ID,
    PROGRAM,
    PROMOTION_POLICY_SCHEMA,
    ROLE_AWARE_PROMOTION_POLICY_INTERFACE,
    TASK_ID,
    FormalVerificationToolRole,
    PromotionDecision,
    RoleAwarePromotionPolicy,
    SemanticLane,
    ToolRole,
    ToolchainAuthorityCeiling,
    ToolchainRoleError,
    can_satisfy_certified_authority_requirement,
    default_policy,
    default_promotion_policy,
    evaluate_role_aware_promotion,
    get_tool_role,
    list_semantic_lanes,
    list_tool_roles,
    reset_default_policy,
    role_can_satisfy_certified_authority,
    tools_by_authority_ceiling,
    tools_by_role,
)

INTERFACE: Final = FORMAL_VERIFICATION_TOOL_ROLE_INTERFACE
PROMOTION_INTERFACE: Final = ROLE_AWARE_PROMOTION_POLICY_INTERFACE
CERTIFICATION_SURFACE: Final = "tools.logic.certification.roles"

# Pre-declared handler slots. Later tasks replace the placeholder with a real
# semantic certifier without touching certify_formal_verification_toolchains.py.
LANE_HANDLER_OWNERS: Final[Mapping[str, str]] = {
    "smt": "tools.logic.certification.smt",
    "tla": "tools.logic.certification.state_model",
    "datalog_secpal": "tools.logic.certification.authorization",
    "protocol": "tools.logic.certification.tamarin",
    "hyperltl": "tools.logic.certification.hyperproperty",
    "atp": "tools.logic.certification.atp",
    "hammer": "tools.logic.certification.advisors",
    "kernel": "tools.logic.certification.lean",
    "runtime_mtl": "tools.logic.certification.runtime_mtl",
    "attestation": "tools.logic.certification.zkp",
    "support": "tools.logic.certification.roles",
}


@dataclass(frozen=True, slots=True)
class LaneHandlerPlaceholder:
    """Fail-closed placeholder until a lane-specific certifier is implemented."""

    lane_id: str
    owner_module: str
    handler_id: str
    status: str = "registered_pending_implementation"

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "owner_module": self.owner_module,
            "handler_id": self.handler_id,
            "status": self.status,
            "certified": False,
            "reason_codes": ["lane_handler_pending_implementation"],
            "args_received": bool(args) or bool(kwargs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "owner_module": self.owner_module,
            "handler_id": self.handler_id,
            "status": self.status,
        }


def build_role_aware_policy(
    *,
    register_placeholders: bool = True,
) -> RoleAwarePromotionPolicy:
    """Return the canonical policy with independently owned lane handlers bound."""

    policy = default_promotion_policy()
    if register_placeholders:
        for lane_id, lane in policy.lanes.items():
            owner = LANE_HANDLER_OWNERS.get(lane_id, lane.owner_module)
            placeholder = LaneHandlerPlaceholder(
                lane_id=lane_id,
                owner_module=owner,
                handler_id=lane.handler_id,
            )
            policy.register_lane_handler(lane_id, placeholder, replace=True)
    policy.assert_matrix_invariants()
    return policy


def bind_lane_handler(
    lane_id: str,
    handler: Callable[..., Any],
    *,
    policy: RoleAwarePromotionPolicy | None = None,
    replace: bool = True,
) -> RoleAwarePromotionPolicy:
    """Attach a real semantic certifier to a pre-registered lane."""

    target = policy if policy is not None else build_role_aware_policy()
    target.register_lane_handler(lane_id, handler, replace=replace)
    return target


def authority_boundary_report(
    policy: RoleAwarePromotionPolicy | None = None,
) -> dict[str, Any]:
    """Machine-readable boundary report for certification and tests."""

    target = policy if policy is not None else build_role_aware_policy()
    support = [item.tool_id for item in tools_by_role(ToolRole.SUPPORT)]
    advisors = [
        item.tool_id
        for item in tools_by_role(ToolRole.ADVISOR)
    ] + [item.tool_id for item in tools_by_role(ToolRole.CANDIDATE)]
    shadows = [item.tool_id for item in tools_by_role(ToolRole.SHADOW)]
    kernels = [
        item.tool_id
        for item in tools_by_authority_ceiling(ToolchainAuthorityCeiling.KERNEL)
    ]
    authorization = [
        item.tool_id
        for item in tools_by_authority_ceiling(
            ToolchainAuthorityCeiling.AUTHORIZATION
        )
    ]
    finite_trace = [
        item.tool_id
        for item in tools_by_authority_ceiling(
            ToolchainAuthorityCeiling.FINITE_TRACE
        )
    ]
    bounded = [
        item.tool_id
        for item in tools_by_authority_ceiling(ToolchainAuthorityCeiling.BOUNDED)
    ]
    attestation = [
        item.tool_id
        for item in tools_by_authority_ceiling(
            ToolchainAuthorityCeiling.ATTESTATION
        )
    ]

    non_certifying_presence_blocked: list[dict[str, Any]] = []
    for tool_id in support + advisors + shadows:
        decision = target.evaluate_promotion(
            tool_id,
            present=True,
            usable=True,
            production_certified=True,
            hermetic_certificate=True,
            independent_reconstruction=True,
        )
        non_certifying_presence_blocked.append(decision.to_dict())
        if decision.allowed:
            raise ToolchainRoleError(
                f"non-certifying tool {tool_id!r} incorrectly allowed promotion"
            )

    return {
        "schema_version": PROMOTION_POLICY_SCHEMA,
        "interface": PROMOTION_INTERFACE,
        "role_interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "certification_surface": CERTIFICATION_SURFACE,
        "support_only": sorted(set(support)),
        "advisor_or_candidate_only": sorted(set(advisors)),
        "shadow_checkers": sorted(set(shadows)),
        "authorization_only": sorted(authorization),
        "finite_trace_authority": sorted(finite_trace),
        "bounded_authority": sorted(bounded),
        "kernel_authority": sorted(kernels),
        "attestation_authority_only": sorted(attestation),
        "lane_handlers": {
            lane_id: {
                "owner_module": LANE_HANDLER_OWNERS.get(
                    lane_id, target.lanes[lane_id].owner_module
                ),
                "handler_id": target.lanes[lane_id].handler_id,
                "registered": lane_id in target.registered_handler_ids(),
                "authority_tool_ids": list(target.lanes[lane_id].authority_tool_ids),
            }
            for lane_id in target.list_lane_ids()
        },
        "non_certifying_presence_evaluations": non_certifying_presence_blocked,
        "policy": {
            "support_advisor_shadow_presence_cannot_certify": True,
            "availability_is_not_authority": True,
            "exactly_one_role_and_ceiling_per_tool": True,
            "lanes_independently_owned": True,
            "central_certifier_not_required_for_lane_binding": True,
        },
    }


def build_role_matrix_receipt(
    policy: RoleAwarePromotionPolicy | None = None,
) -> dict[str, Any]:
    """Full matrix receipt for operators and downstream certification lanes."""

    target = policy if policy is not None else build_role_aware_policy()
    payload = target.to_dict()
    payload["authority_boundary"] = authority_boundary_report(target)
    payload["certification_surface"] = CERTIFICATION_SURFACE
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the role-aware toolchain authority matrix and promotion policy "
            f"({INTERFACE} / {PROMOTION_INTERFACE})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full matrix receipt as JSON",
    )
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="Print only the authority-boundary report",
    )
    args = parser.parse_args(argv)

    policy = build_role_aware_policy()
    if args.boundary_only:
        payload = authority_boundary_report(policy)
    else:
        payload = build_role_matrix_receipt(policy)

    if args.json or args.boundary_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        roles = list_tool_roles()
        print(f"{INTERFACE} / {PROMOTION_INTERFACE}")
        print(f"goal={GOAL_ID} task={TASK_ID} tools={len(roles)} lanes={len(list_semantic_lanes())}")
        for item in roles:
            flag = "certifying" if item.can_satisfy_certified_authority else "non-certifying"
            print(
                f"  {item.tool_id:24} role={item.role.value:10} "
                f"ceiling={item.authority_ceiling.value:16} {flag}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "PROMOTION_INTERFACE",
    "CERTIFICATION_SURFACE",
    "LANE_HANDLER_OWNERS",
    "LaneHandlerPlaceholder",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "FormalVerificationToolRole",
    "PromotionDecision",
    "RoleAwarePromotionPolicy",
    "SemanticLane",
    "ToolRole",
    "ToolchainAuthorityCeiling",
    "ToolchainRoleError",
    "build_role_aware_policy",
    "bind_lane_handler",
    "authority_boundary_report",
    "build_role_matrix_receipt",
    "can_satisfy_certified_authority_requirement",
    "default_policy",
    "default_promotion_policy",
    "evaluate_role_aware_promotion",
    "get_tool_role",
    "list_semantic_lanes",
    "list_tool_roles",
    "reset_default_policy",
    "role_can_satisfy_certified_authority",
    "tools_by_authority_ceiling",
    "tools_by_role",
    "main",
]
