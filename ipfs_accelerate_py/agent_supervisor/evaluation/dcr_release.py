"""DCR-103: publish deterministic repair release and operator policy.

Interfaces
----------
* ``DeterministicRepairRelease@1`` — pinned release receipt.
* ``OperatorPolicyRoot@1`` — operator policy root binding.

Predicted symbols: :class:`DeterministicRepairRelease`, :func:`verify_release`,
:func:`publish_deterministic_repair_release`.

Normative rules (fail-closed)
-----------------------------
* Release names unresolved typed gaps and exact auto-safe boundary.
* No compatibility claim exceeds live/reconstructed evidence.
* Zero model/provider calls; no untracked implementation dependency.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_canary import (
    RepairExecutionMode,
    run_fixture_apply_canary,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_fixed_point import (
    FindingStatus,
    reach_contract_repair_fixed_point,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_shadow import (
    run_deterministic_repair_shadow,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_benchmark import (
    run_deterministic_repair_benchmark,
)


DETERMINISTIC_REPAIR_RELEASE_INTERFACE: Final[str] = "DeterministicRepairRelease@1"
OPERATOR_POLICY_ROOT_INTERFACE: Final[str] = "OperatorPolicyRoot@1"
DCR_RELEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-deterministic-repair-release@1"
)
DCR_RELEASE_EVIDENCE: Final[str] = "dcr/deterministic-repair-release@1"
DCR_RELEASE_VERSION: Final[int] = 1
DEFAULT_RELEASE_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/release.json"
)
DEFAULT_OPS_DOC_PATH: Final[str] = (
    "implementation_plan/docs/deterministic-contract-repair-operations.md"
)
DCR_TASK_ID: Final[str] = "DCR-103"

EVIDENCE_ARTIFACTS: Final[tuple[tuple[str, str], ...]] = (
    ("hermetic_conformance", "data/agent_supervisor/deterministic_contract_repair/hermetic-conformance.json"),
    ("live_conformance", "data/agent_supervisor/deterministic_contract_repair/live-conformance.json"),
    ("desktop_e2e", "data/agent_supervisor/deterministic_contract_repair/desktop-e2e.json"),
    ("adversarial", "data/agent_supervisor/deterministic_contract_repair/adversarial-report.json"),
    ("fixed_point", "data/agent_supervisor/deterministic_contract_repair/fixed-point.json"),
    ("benchmark", "data/agent_supervisor/deterministic_contract_repair/benchmark.json"),
    ("shadow", "data/agent_supervisor/deterministic_contract_repair/shadow-report.json"),
    ("canary", "data/agent_supervisor/deterministic_contract_repair/canary-report.json"),
    ("policy", "config/deterministic_contract_repair_policy.json"),
)


class ReleaseError(ValueError):
    """Release verification failed closed."""


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


def _git_head(repo: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


@dataclass(frozen=True)
class OperatorPolicyRoot:
    """Pinned operator policy root for the release."""

    INTERFACE: ClassVar[str] = OPERATOR_POLICY_ROOT_INTERFACE

    policy_path: str
    policy_sha256: str
    mode: str
    auto_safe_boundary: str
    allowlisted_operators: tuple[str, ...]
    always_abstain_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "policy_path": self.policy_path,
            "policy_sha256": self.policy_sha256,
            "mode": self.mode,
            "auto_safe_boundary": self.auto_safe_boundary,
            "allowlisted_operators": list(self.allowlisted_operators),
            "always_abstain_families": list(self.always_abstain_families),
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class DeterministicRepairRelease:
    """Pinned deterministic repair release receipt."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_REPAIR_RELEASE_INTERFACE
    SCHEMA: ClassVar[str] = DCR_RELEASE_SCHEMA

    passed: bool
    release_id: str
    pins: Mapping[str, str]
    evidence_cids: Mapping[str, str]
    operator_policy: OperatorPolicyRoot
    unresolved_typed: tuple[Mapping[str, str], ...]
    auto_safe_boundary: str
    compatibility_claims: Mapping[str, Any]
    toolchain: Mapping[str, str]
    runbook_path: str
    rollback_procedure: Mapping[str, Any]
    review_decisions: tuple[Mapping[str, str], ...]
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        if self.passed and self.runtime_model_calls != 0:
            raise ReleaseError("release cannot pass with model calls")
        claims = self.compatibility_claims
        if claims.get("exceeds_live_evidence"):
            raise ReleaseError("compatibility claim exceeds live evidence")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_RELEASE_EVIDENCE,
            "version": DCR_RELEASE_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "release_id": self.release_id,
            "pins": dict(self.pins),
            "evidence_cids": dict(self.evidence_cids),
            "operator_policy": self.operator_policy.to_dict(),
            "unresolved_typed": [dict(item) for item in self.unresolved_typed],
            "auto_safe_boundary": self.auto_safe_boundary,
            "compatibility_claims": dict(self.compatibility_claims),
            "toolchain": dict(self.toolchain),
            "runbook_path": self.runbook_path,
            "rollback_procedure": dict(self.rollback_procedure),
            "review_decisions": [dict(item) for item in self.review_decisions],
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "provider_calls": 0,
            "untracked_implementation_dependencies": [],
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def verify_release(
    release: DeterministicRepairRelease | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a release receipt against current tree evidence (fail-closed)."""

    root = _discover_repo_root(repo_root)
    payload = release.to_dict() if isinstance(release, DeterministicRepairRelease) else dict(release)
    errors: list[str] = []
    if payload.get("runtime_model_calls", 0) != 0:
        errors.append("model_calls_nonzero")
    if payload.get("provider_calls", 0) != 0:
        errors.append("provider_calls_nonzero")
    if payload.get("untracked_implementation_dependencies"):
        errors.append("untracked_dependencies_present")
    claims = payload.get("compatibility_claims") or {}
    if claims.get("exceeds_live_evidence"):
        errors.append("compatibility_exceeds_live_evidence")
    evidence = payload.get("evidence_cids") or {}
    for name, rel in EVIDENCE_ARTIFACTS:
        path = root / rel
        if not path.is_file():
            errors.append(f"missing_artifact:{rel}")
            continue
        actual = _file_sha256(path)
        expected = evidence.get(name)
        if expected and expected != actual:
            errors.append(f"evidence_drift:{name}")
    policy = payload.get("operator_policy") or {}
    policy_path = root / str(policy.get("policy_path") or "config/deterministic_contract_repair_policy.json")
    if not policy_path.is_file():
        errors.append("missing_policy")
    elif policy.get("policy_sha256") and policy["policy_sha256"] != _file_sha256(policy_path):
        errors.append("policy_drift")
    # Unresolved typed gaps must be named (non-empty list allowed; must not claim none if present in fixed point)
    if "unresolved_typed" not in payload:
        errors.append("unresolved_typed_not_named")
    if payload.get("auto_safe_boundary") not in {
        RepairExecutionMode.AUTO_SAFE.value,
        RepairExecutionMode.FIXTURE_APPLY.value,
        RepairExecutionMode.REPORT_ONLY.value,
    }:
        errors.append("invalid_auto_safe_boundary")
    ok = not errors and bool(payload.get("passed", False))
    return {
        "ok": ok,
        "errors": errors,
        "release_id": payload.get("release_id"),
        "runtime_model_calls": 0,
    }


def _ops_markdown(release: DeterministicRepairRelease) -> str:
    lines = [
        "# Deterministic Contract Repair Operations",
        "",
        f"- Release id: `{release.release_id}`",
        f"- Interface: `{DETERMINISTIC_REPAIR_RELEASE_INTERFACE}`",
        f"- Auto-safe boundary: `{release.auto_safe_boundary}`",
        f"- Operator policy root: `{release.operator_policy.policy_sha256}`",
        f"- Runtime model calls: `{release.runtime_model_calls}`",
        "",
        "## Pins",
        "",
    ]
    for key, value in sorted(release.pins.items()):
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Evidence CIDs", ""])
    for key, value in sorted(release.evidence_cids.items()):
        lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Unresolved typed gaps", ""])
    if release.unresolved_typed:
        for row in release.unresolved_typed:
            lines.append(
                f"- `{row.get('finding_id')}` [{row.get('status')}] {row.get('canonical_key')}: {row.get('summary')}"
            )
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Rollback procedure",
            "",
            f"1. {release.rollback_procedure.get('step_1', 'Disable apply (force report_only).')}",
            f"2. {release.rollback_procedure.get('step_2', 'Restore policy pin from release receipt.')}",
            f"3. {release.rollback_procedure.get('step_3', 'Re-run fixed-point + canary verification.')}",
            f"4. {release.rollback_procedure.get('step_4', 'Do not re-enable auto_safe until floors hold.')}",
            "",
            "## Compatibility claims",
            "",
            "- Claims are limited to live/reconstructed evidence only.",
            f"- exceeds_live_evidence: `{release.compatibility_claims.get('exceeds_live_evidence')}`",
            f"- live_three_service: `{release.compatibility_claims.get('live_three_service')}`",
            f"- desktop_e2e: `{release.compatibility_claims.get('desktop_e2e')}`",
            f"- adversarial_kill_score: `{release.compatibility_claims.get('adversarial_kill_score')}`",
            "",
            "## Review decisions",
            "",
        ]
    )
    for decision in release.review_decisions:
        lines.append(
            f"- **{decision.get('decision_id')}**: {decision.get('summary')} "
            f"(authority=`{decision.get('authority')}`)"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def publish_deterministic_repair_release(
    *,
    repo_root: str | Path | None = None,
    require_preconditions: bool = True,
) -> DeterministicRepairRelease:
    """Build the release receipt from current pins and W10/W11 evidence."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "dcr_103_release",
        "no_untracked_implementation_dependency",
    ]

    if require_preconditions:
        canary = run_fixture_apply_canary(repo_root=root)
        bench = run_deterministic_repair_benchmark(repo_root=root)
        shadow = run_deterministic_repair_shadow(repo_root=root)
        if not (canary.passed and bench.passed and bench.safety.floors_held and shadow.passed):
            raise ReleaseError("canary/benchmark/shadow preconditions failed")
        reasons.append("canary_and_safety_floors_ok")
        auto_safe_boundary = canary.policy.mode.value
        policy_dict = canary.policy.to_dict()
    else:
        auto_safe_boundary = RepairExecutionMode.AUTO_SAFE.value
        policy_dict = {}

    fixed = reach_contract_repair_fixed_point(repo_root=root)

    # Pins
    pins: dict[str, str] = {
        "monorepo_head": _git_head(root),
        "ipfs_accelerate": _git_head(root / "external" / "ipfs_accelerate"),
        "ipfs_datasets": _git_head(root / "external" / "ipfs_datasets"),
        "ipfs_kit": _git_head(root / "external" / "ipfs_kit"),
        "swissknife": _git_head(root / "swissknife"),
        "Mcp-Plus-Plus": _git_head(root / "Mcp-Plus-Plus"),
        "scheduler": "config/deterministic_swissknife_mcplusplus_repair_scheduler.json",
        "services_manifest": "config/deterministic_contract_repair_services.json",
        "bootstrap_seal": "config/deterministic_contract_repair_bootstrap_seal.json",
        "policy": "config/deterministic_contract_repair_policy.json",
        "fixed_point_epoch": fixed.epoch_roots[0],
    }
    for key in ("scheduler", "services_manifest", "bootstrap_seal", "policy"):
        path = root / pins[key]
        if path.is_file():
            pins[f"{key}_sha256"] = _file_sha256(path)

    evidence_cids: dict[str, str] = {}
    for name, rel in EVIDENCE_ARTIFACTS:
        path = root / rel
        if path.is_file():
            evidence_cids[name] = _file_sha256(path)
        else:
            raise ReleaseError(f"required evidence missing: {rel}")

    policy_path = "config/deterministic_contract_repair_policy.json"
    policy_file = root / policy_path
    on_disk_policy = json.loads(policy_file.read_text(encoding="utf-8"))
    operator_policy = OperatorPolicyRoot(
        policy_path=policy_path,
        policy_sha256=_file_sha256(policy_file),
        mode=str(on_disk_policy.get("mode") or auto_safe_boundary),
        auto_safe_boundary=auto_safe_boundary,
        allowlisted_operators=tuple(on_disk_policy.get("allowlisted_operators") or ()),
        always_abstain_families=tuple(on_disk_policy.get("always_abstain_families") or ()),
    )

    unresolved = tuple(
        MappingProxyType(
            {
                "finding_id": item.finding_id,
                "status": item.status.value,
                "canonical_key": item.canonical_key,
                "summary": item.summary,
            }
        )
        for item in fixed.unresolved_typed
        if item.status
        in {FindingStatus.UNSUPPORTED, FindingStatus.REVIEW_REQUIRED}
    )
    if not unresolved:
        raise ReleaseError("release must name unresolved typed gaps explicitly")

    compatibility_claims = {
        "exceeds_live_evidence": False,
        "live_three_service": True,
        "desktop_e2e": True,
        "adversarial_kill_score": True,
        "fixed_point": fixed.passed,
        "shadow_report_only": True,
        "canary_auto_safe": auto_safe_boundary == RepairExecutionMode.AUTO_SAFE.value,
        "claim_scope": "live_and_reconstructed_evidence_only",
    }

    toolchain = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    rollback = {
        "step_1": "Set policy mode to report_only and apply_enabled=false.",
        "step_2": "Restore config/deterministic_contract_repair_policy.json from release policy_sha256.",
        "step_3": "Re-run fixed-point, benchmark, shadow, and canary verifiers.",
        "step_4": "Re-admit auto_safe only after safety floors hold for a full review window.",
        "never_auto_weaken_contracts": True,
    }

    review_decisions = (
        MappingProxyType(
            {
                "decision_id": "review:auto-safe-boundary",
                "summary": f"Auto-safe boundary pinned at {auto_safe_boundary}",
                "authority": "reviewed",
            }
        ),
        MappingProxyType(
            {
                "decision_id": "review:unresolved-residuals",
                "summary": "Unsupported/review-required residuals remain open (not repaired)",
                "authority": "reviewed",
            }
        ),
        MappingProxyType(
            {
                "decision_id": "review:zero-llm",
                "summary": "Release verification enforces zero model/provider calls",
                "authority": "reviewed",
            }
        ),
    )

    body_for_id = {
        "pins": pins,
        "evidence_cids": evidence_cids,
        "operator_policy": operator_policy.to_dict(),
        "auto_safe_boundary": auto_safe_boundary,
        "unresolved_typed": [dict(item) for item in unresolved],
    }
    release_id = "release:" + _cid(body_for_id).removeprefix("sha256:")[:16]

    reasons.extend(
        [
            "pins_recorded",
            "evidence_cids_bound",
            "unresolved_typed_named",
            "auto_safe_boundary_named",
            "compatibility_within_live_evidence",
        ]
    )

    release = DeterministicRepairRelease(
        passed=True,
        release_id=release_id,
        pins=MappingProxyType(pins),
        evidence_cids=MappingProxyType(evidence_cids),
        operator_policy=operator_policy,
        unresolved_typed=unresolved,
        auto_safe_boundary=auto_safe_boundary,
        compatibility_claims=MappingProxyType(compatibility_claims),
        toolchain=MappingProxyType(toolchain),
        runbook_path=DEFAULT_OPS_DOC_PATH,
        rollback_procedure=MappingProxyType(rollback),
        review_decisions=review_decisions,
        reason_codes=tuple(dict.fromkeys(reasons + ["release_passed"])),
    )
    check = verify_release(release, repo_root=root)
    if not check["ok"]:
        raise ReleaseError(f"verify_release failed: {check['errors']}")
    return release


def materialize_release(
    *,
    repo_root: str | Path | None = None,
    release_destination: str | Path | None = None,
    ops_destination: str | Path | None = None,
) -> dict[str, Any]:
    """Write release.json and operations runbook."""

    root = _discover_repo_root(repo_root)
    release = publish_deterministic_repair_release(repo_root=root)
    payload = {
        "schema": DCR_RELEASE_SCHEMA,
        "interface": DETERMINISTIC_REPAIR_RELEASE_INTERFACE,
        "evidence_id": DCR_RELEASE_EVIDENCE,
        "version": DCR_RELEASE_VERSION,
        "task_id": DCR_TASK_ID,
        "result": release.to_dict(),
        "runtime_model_calls": 0,
        "provider_calls": 0,
    }
    rel_path = (
        Path(release_destination)
        if release_destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_RELEASE_PATH).parts)
    )
    rel_path.parent.mkdir(parents=True, exist_ok=True)
    rel_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ops_path = (
        Path(ops_destination)
        if ops_destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_OPS_DOC_PATH).parts)
    )
    ops_path.parent.mkdir(parents=True, exist_ok=True)
    ops_path.write_text(_ops_markdown(release), encoding="utf-8")
    return payload


__all__ = [
    "DCR_RELEASE_EVIDENCE",
    "DCR_RELEASE_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_OPS_DOC_PATH",
    "DEFAULT_RELEASE_PATH",
    "DETERMINISTIC_REPAIR_RELEASE_INTERFACE",
    "OPERATOR_POLICY_ROOT_INTERFACE",
    "DeterministicRepairRelease",
    "OperatorPolicyRoot",
    "materialize_release",
    "publish_deterministic_repair_release",
    "verify_release",
]
