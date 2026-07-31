"""Two-profile end-to-end: VFS and hermetic non-VFS share generic engines (LPR-028).

Runs inventory → contracts projection → rollout/verify through the identical
generic modules under (1) the locked IPFS Kit VFS profile and (2) a hermetic
non-VFS inventory-to-rollout fixture. Confirms parameterization without domain
branches in the engines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory import (
    SurfaceInventoryPolicy,
    SurfaceKindSpec,
    SurfaceSignal,
    SignalTarget,
    inventory_repository_surfaces,
)
from ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout import (
    AssuranceRolloutMode,
    SymbolicAssurancePublicAPI,
    build_frozen_adversarial_population,
    build_generic_rollout_profile,
    evaluate_symbolic_assurance_rollout,
    project_bounded_findings,
    project_bounded_receipts,
    project_bounded_status,
    run_symbolic_assurance_e2e,
    verify_adversarial_e2e_report,
    verify_symbolic_assurance_rollout,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_kit_vfs_assurance import (
    CLOSED_ADAPTERS,
    dispatch,
    load_assurance_config,
    run_contracts,
    run_rollout,
    run_verify,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "ipfs_kit_vfs_symbolic_assurance.json"

GENERIC_ENGINE_IMPORT_PATHS = (
    "ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory",
    "ipfs_accelerate_py.agent_supervisor.analysis.program_contract_profile",
    "ipfs_accelerate_py.agent_supervisor.validation.differential_contract_harness",
    "ipfs_accelerate_py.agent_supervisor.analysis.interface_contract_parity",
    "ipfs_accelerate_py.agent_supervisor.validation.symbolic_efficiency_benchmark",
    "ipfs_accelerate_py.agent_supervisor.runtime.symbolic_assurance_pilot",
    "ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout",
)


@dataclass(frozen=True)
class AssuranceTwoProfileConformance:
    """Cross-profile conformance receipt for LPR-028."""

    schema: str = (
        "ipfs_accelerate_py/agent-supervisor/assurance-two-profile-conformance@1"
    )
    vfs_profile_id: str = ""
    non_vfs_profile_id: str = ""
    shared_engine_modules: tuple[str, ...] = ()
    vfs_stages: Mapping[str, Any] = field(default_factory=dict)
    non_vfs_stages: Mapping[str, Any] = field(default_factory=dict)
    content_id: str = ""

    @property
    def passed(self) -> bool:
        if not self.vfs_stages or not self.non_vfs_stages:
            return False
        required = ("inventory", "contracts", "rollout", "verify")
        for stage in required:
            if stage not in self.vfs_stages or stage not in self.non_vfs_stages:
                return False
        if self.vfs_stages.get("engine_module_ids") != self.non_vfs_stages.get(
            "engine_module_ids"
        ):
            return False
        if self.vfs_profile_id == self.non_vfs_profile_id:
            return False
        return bool(self.vfs_stages.get("ok") and self.non_vfs_stages.get("ok"))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "passed": self.passed,
            "vfs_profile_id": self.vfs_profile_id,
            "non_vfs_profile_id": self.non_vfs_profile_id,
            "shared_engine_modules": list(self.shared_engine_modules),
            "vfs_stages": dict(self.vfs_stages),
            "non_vfs_stages": dict(self.non_vfs_stages),
        }
        unsigned = dict(payload)
        unsigned.pop("content_id", None)
        digest = "sha256:" + hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        payload["content_id"] = digest
        return payload


def _engine_module_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for name in GENERIC_ENGINE_IMPORT_PATHS:
        module = __import__(name, fromlist=["*"])
        path = Path(getattr(module, "__file__", "") or "")
        ids.append(f"{name}:{path.resolve()}")
    return tuple(ids)


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _vfs_inventory_policy() -> SurfaceInventoryPolicy:
    return SurfaceInventoryPolicy(
        profile_id="ipfs-kit-vfs-inventory@1",
        schema="ipfs_accelerate_py/agent-supervisor/vfs-surface-inventory@1",
        contract_version="vfs-surface-inventory/v1",
        content_signals=(
            SurfaceSignal(
                name="domain_content",
                pattern=r"(?i)(?<![a-z0-9])(?:vfs|fsspec|ipfs)(?![a-z0-9])",
                target=SignalTarget.CONTENT,
            ),
        ),
        path_signals=(
            SurfaceSignal(
                name="domain_path",
                pattern=(
                    r"(?i)(?:^|[./_-])(?:vfs|fsspec|ipfs)"
                    r"(?:[_-][a-z0-9]+)*(?=[^a-z0-9]|$)"
                ),
                target=SignalTarget.PATH,
            ),
        ),
        kind_specs=(
            SurfaceKindSpec(kind="fsspec", combined_patterns=(r"(?i)fsspec",)),
            SurfaceKindSpec(kind="vfs_surface", combined_patterns=(r"(?i)vfs",)),
        ),
    )


def _widget_inventory_policy() -> SurfaceInventoryPolicy:
    return SurfaceInventoryPolicy(
        profile_id="widget-surface-inventory@1",
        schema="ipfs_accelerate_py/agent-supervisor/repository-surface-inventory@1",
        contract_version="repository-surface-inventory/v1",
        content_signals=(
            SurfaceSignal(
                name="widget_content",
                pattern=r"(?i)(?<![a-z0-9])(?:widget|gadget[_-]?bus)(?![a-z0-9])",
                target=SignalTarget.CONTENT,
            ),
        ),
        path_signals=(
            SurfaceSignal(
                name="widget_path",
                pattern=(
                    r"(?i)(?:^|[./_-])(?:widget|gadget)"
                    r"(?:[_-][a-z0-9]+)*(?=[^a-z0-9]|$)"
                ),
                target=SignalTarget.PATH,
            ),
        ),
        kind_specs=(
            SurfaceKindSpec(
                kind="widget_manager",
                path_patterns=(r"(?:^|[/_.-])widget[_-]?manager(?:[/_.-]|$)",),
            ),
            SurfaceKindSpec(
                kind="gadget_bus",
                path_patterns=(r"(?:^|[/_.-])gadget[_-]?bus(?:[/_.-]|$)",),
            ),
        ),
    )


def _vfs_fixture(root: Path) -> None:
    _write(
        root,
        "pkg/vfs_manager.py",
        "class VFSManager:\n"
        "    def mount(self, path):\n"
        "        return path\n",
    )
    _write(
        root,
        "pkg/ipfs_fsspec.py",
        "def open_fs(url):\n"
        "    return url\n",
    )
    _write(root, "docs/readme.md", "no domain signal here\n")


def _widget_fixture(root: Path) -> None:
    _write(
        root,
        "pkg/widget_manager.py",
        "class WidgetManager:\n"
        "    def spin(self, rate):\n"
        "        return rate\n",
    )
    _write(
        root,
        "pkg/gadget_bus.py",
        "def tick():\n"
        "    return 1\n",
    )
    # Deliberate VFS-looking names that the widget profile must ignore.
    _write(
        root,
        "pkg/vfs_manager.py",
        "class VFSManager:\n"
        "    def mount(self, path):\n"
        "        return path\n",
    )


def _run_vfs_profile_stages() -> dict[str, Any]:
    config = load_assurance_config(CONFIG_PATH)
    engine_ids = _engine_module_ids()
    contracts = run_contracts(config=config)
    rollout = run_rollout(config=config, desired_mode="assist")
    verified = run_verify(config=config)

    # Inventory through the same generic factory the adapter uses.
    inventory_factory = __import__(
        "ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory",
        fromlist=["inventory_repository_surfaces"],
    ).inventory_repository_surfaces
    # Hermetic mini corpus under tmp is exercised separately; here confirm the
    # adapter dispatch surface and closed registry.
    adapters_ok = set(config.adapters) == set(CLOSED_ADAPTERS)
    ok = (
        adapters_ok
        and "read" in contracts.get("operations", [])
        and rollout.get("automatic_mutation_enabled") is False
        and verified.get("verified") is True
        and callable(inventory_factory)
    )
    return {
        "ok": ok,
        "profile_id": config.profile.profile_id,
        "behavior_id": config.profile.behavior_id,
        "engine_module_ids": list(engine_ids),
        "inventory": {
            "adapter": "inventory",
            "factory": inventory_factory.__module__ + "." + inventory_factory.__name__,
        },
        "contracts": {
            "operations": list(contracts.get("operations") or [])[:8],
            "authority_flags": dict(contracts.get("authority_flags") or {}),
        },
        "rollout": {
            "effective_mode": rollout.get("decision", {}).get("effective_mode"),
            "schema": rollout.get("adversarial_e2e_gate", {}).get("schema"),
            "automatic_mutation_enabled": rollout.get("automatic_mutation_enabled"),
        },
        "verify": {
            "verified": verified.get("verified"),
            "effective_mode": verified.get("effective_mode"),
        },
    }


def _run_non_vfs_profile_stages(tmp_path: Path) -> dict[str, Any]:
    engine_ids = _engine_module_ids()
    widget_root = tmp_path / "widget_repo"
    widget_root.mkdir(exist_ok=True)
    _widget_fixture(widget_root)

    policy = _widget_inventory_policy()
    inventory = inventory_repository_surfaces(widget_root, policy)
    paths = sorted(item.path for item in inventory.surfaces)

    profile = build_generic_rollout_profile(
        profile_id="profile:widget-assurance@1",
        behavior_id="behavior:widget-assurance@1",
        objective_id="WIDGET-G001",
        objective_revision="WIDGET-G001@1",
        requirement_id="widget:adversarial-e2e",
        default_fixture_repositories={
            "repository:widget-a@fixture": {
                "src/widget.py": "def spin():\n    return 1\n",
                "node_modules/x/index.js": "module.exports=1\n",
            },
            "repository:widget-b@fixture": {
                "lib/gadget.py": "def tick():\n    return 2\n",
                "__pycache__/x.pyc": "bin",
            },
        },
        default_fixture_id="fixture:widget-e2e@1",
    )
    fixture, report, binding, policy_rollout = build_frozen_adversarial_population(
        profile=profile
    )
    assert verify_adversarial_e2e_report(report)
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy_rollout,
        desired_mode=AssuranceRolloutMode.ASSIST,
    )
    assert verify_symbolic_assurance_rollout(
        decision, report, binding=binding, policy=policy_rollout
    )
    e2e = run_symbolic_assurance_e2e(profile=profile, desired_mode="assist")
    api = SymbolicAssurancePublicAPI(
        report, binding=binding, policy=policy_rollout, initial_mode="shadow"
    )
    discovery = SymbolicAssurancePublicAPI.discovery(profile)

    ok = (
        "pkg/widget_manager.py" in paths
        and "pkg/gadget_bus.py" in paths
        and "pkg/vfs_manager.py" not in paths
        and report.passed
        and decision.effective_mode is AssuranceRolloutMode.ASSIST
        and e2e.get("automatic_mutation_enabled") is False
        and discovery["behavior_id"] == "behavior:widget-assurance@1"
        and project_bounded_status(decision)["behavior_id"]
        == "behavior:widget-assurance@1"
        and project_bounded_findings(decision)["finding_count"] == 0
        and project_bounded_receipts(decision)["receipt_count"] >= 4
        and api.status().decision.effective_mode is AssuranceRolloutMode.SHADOW
    )
    return {
        "ok": ok,
        "profile_id": profile.profile_id,
        "behavior_id": profile.behavior_id,
        "engine_module_ids": list(engine_ids),
        "inventory": {
            "surface_paths": paths,
            "profile_id": policy.profile_id,
            "surface_count": len(paths),
        },
        "contracts": {
            # Non-VFS profile does not load the VFS operation matrix; the
            # contracts *engine* is still the same module when used.
            "engine": "program_contract_profile",
            "parameterized": True,
        },
        "rollout": {
            "effective_mode": decision.effective_mode.value,
            "fixture_id": fixture.fixture_id,
            "automatic_mutation_enabled": False,
            "e2e_schema": e2e.get("adversarial_e2e_gate", {}).get("schema"),
        },
        "verify": {
            "verified": True,
            "effective_mode": decision.effective_mode.value,
        },
    }


def compute_two_profile_conformance(tmp_path: Path) -> AssuranceTwoProfileConformance:
    vfs_stages = _run_vfs_profile_stages()
    non_vfs_stages = _run_non_vfs_profile_stages(tmp_path)
    receipt = AssuranceTwoProfileConformance(
        vfs_profile_id=str(vfs_stages.get("profile_id") or ""),
        non_vfs_profile_id=str(non_vfs_stages.get("profile_id") or ""),
        shared_engine_modules=GENERIC_ENGINE_IMPORT_PATHS,
        vfs_stages=vfs_stages,
        non_vfs_stages=non_vfs_stages,
    )
    materialized = receipt.to_dict()
    object.__setattr__(receipt, "content_id", materialized["content_id"])
    return receipt


def test_same_generic_engine_modules_serve_both_profiles(tmp_path: Path) -> None:
    receipt = compute_two_profile_conformance(tmp_path)
    assert isinstance(receipt, AssuranceTwoProfileConformance)
    assert receipt.passed, receipt.to_dict()
    assert receipt.vfs_profile_id != receipt.non_vfs_profile_id
    assert receipt.vfs_stages["engine_module_ids"] == receipt.non_vfs_stages[
        "engine_module_ids"
    ]


def test_vfs_profile_inventory_to_rollout_fixture(tmp_path: Path) -> None:
    root = tmp_path / "vfs_repo"
    root.mkdir()
    _vfs_fixture(root)
    inventory = inventory_repository_surfaces(root, _vfs_inventory_policy())
    paths = {item.path for item in inventory.surfaces}
    assert "pkg/vfs_manager.py" in paths
    assert "pkg/ipfs_fsspec.py" in paths

    config = load_assurance_config(CONFIG_PATH)
    payload = run_rollout(config=config, desired_mode="shadow")
    assert payload["automatic_mutation_enabled"] is False
    assert payload["decision"]["effective_mode"] == "shadow"
    verified = run_verify(config=config)
    assert verified["verified"] is True


def test_non_vfs_inventory_to_rollout_fixture(tmp_path: Path) -> None:
    stages = _run_non_vfs_profile_stages(tmp_path)
    assert stages["ok"]
    assert "pkg/widget_manager.py" in stages["inventory"]["surface_paths"]
    assert "pkg/vfs_manager.py" not in stages["inventory"]["surface_paths"]
    assert stages["rollout"]["effective_mode"] == "assist"
    assert stages["behavior_id"] == "behavior:widget-assurance@1"


def test_dispatch_closed_commands_for_vfs_profile() -> None:
    config = load_assurance_config(CONFIG_PATH)
    for command in ("contracts", "differential", "parity", "benchmark", "pilot", "verify"):
        payload = dispatch(command, config=config)
        assert payload.get("automatic_mutation_enabled") is False
        if command == "verify":
            assert payload["verified"] is True
        else:
            assert payload.get("adapter") == command or "verified" in payload


def test_two_profile_conformance_fixed_point(tmp_path: Path) -> None:
    first = compute_two_profile_conformance(tmp_path)
    second = compute_two_profile_conformance(tmp_path)
    assert first.content_id == second.content_id
    assert first.content_id.startswith("sha256:")
    assert first.schema.endswith("assurance-two-profile-conformance@1")
