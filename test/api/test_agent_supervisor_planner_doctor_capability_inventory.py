"""Focused contracts for the PDR-001 shipped-vs-wired inventory."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.planner_doctor_capability_inventory import (
    INVENTORY_AUTHORIZES_MUTATION,
    INVENTORY_IS_COMPLETION_EVIDENCE,
    INVENTORY_IS_PROOF_EVIDENCE,
    PACKAGE_PRESENCE_IS_CAPABILITY,
    CapabilityAvailability,
    DefaultWiringState,
    PlannerDoctorCapabilityInventory,
    PlannerDoctorInventoryIntegrityError,
    PlannerDoctorInventoryValidationError,
    ToolHealthState,
    build_planner_doctor_capability_inventory,
    discover_planner_doctor_inventory_schemas,
    inventory_content_id,
    replay_planner_doctor_capability_inventory,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "analysis"
    / "planner_doctor_capability_inventory.py"
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "PDR Test",
            "GIT_AUTHOR_EMAIL": "pdr@example.invalid",
            "GIT_COMMITTER_NAME": "PDR Test",
            "GIT_COMMITTER_EMAIL": "pdr@example.invalid",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    return result.stdout.strip()


def _write(root: Path, relative: str, value: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _init_repository(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "PDR Test")
    _git(root, "config", "user.email", "pdr@example.invalid")


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def _minimal_board(*, pdr_001_status: str = "pending") -> str:
    return f"""# PDR fixture

## PDR-000 Bootstrap

- Status: completed
- Depends on:
- Goal id: PDR-G000

## PDR-001 Inventory

- Status: {pdr_001_status}
- Depends on: PDR-000
- Goal id: PDR-G010
"""


def _minimal_objectives() -> str:
    return """# PDR fixture goals

## PDR-G000 Root

- Status: active
- Parent:

## PDR-G010 Foundation

- Status: active
- Parent: PDR-G000
"""


def _minimal_scheduler() -> str:
    return json.dumps(
        {
            "schema": "test.scheduler@1",
            "doctor": {
                "default_mode": "report_only",
                "enabled_at_bootstrap": False,
                "mutation_authorized": False,
                "allow_llm": False,
                "allow_network": False,
            },
            "planner": {"default_mode": "shadow"},
            "derived_refill": {
                "enabled_at_bootstrap": False,
                "enabled_after_task": "PDR-081",
            },
            "benchmark": {
                "live_evidence_required": True,
                "synthetic_evidence_may_promote": False,
                "skipped_checks_may_promote": False,
                "concurrency_sweep": [1, 2, 4],
            },
            "rollout": {"initial_mode": "shadow", "automatic_enabled": False},
        },
        sort_keys=True,
    )


def _seed_audited_sources(root: Path) -> None:
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
        """class PromptSupervisorService:
    def __init__(
        self,
        *,
        optional_analysis=None,
        admission_request_factory=None,
    ):
        self.optional_analysis = optional_analysis
        self.admission_request_factory = admission_request_factory
""",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/control/"
        "deterministic_doctor_service.py",
        """class DeterministicDoctorService:
    def __init__(self, *, backends=None):
        self.backends = backends or ()
""",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/analysis/"
        "doctor_repository_diagnostics.py",
        'DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA = "doctor-evidence-snapshot@1"\n'
        "class DoctorEvidenceSnapshot: pass\n",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/analysis/"
        "deterministic_doctor_contracts.py",
        'DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA = '
        '"deterministic-doctor/evidence-snapshot@1"\n'
        "class DoctorEvidenceSnapshot: pass\n",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/planning/"
        "deterministic_doctor_transaction.py",
        """def _default_static_applicator(request):
    return request

def _default_restore(checkpoint):
    return True
""",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "deterministic_doctor_fixed_point.py",
        """def _default_restore(checkpoint):
    return True
""",
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "deterministic_doctor_benchmark.py",
        'FIXTURE_KIND = "synthetic"\n',
    )
    _write(
        root,
        "ipfs_accelerate_py/agent_supervisor/self_improvement/"
        "supervisor_v2_benchmark.py",
        'PROVIDER_ID = "provider:deterministic-fixture@1"\n',
    )
    _write(
        root,
        "ipfs_accelerate_py/__init__.py",
        "from .hf_space_inference import HFSpaceClient\n",
    )
    _write(root, "ipfs_accelerate_py/hf_space_inference.py", "import requests\n")
    _write(root, "sentinel.txt", "PRIVATE_SOURCE_BODY_MUST_NOT_BE_SERIALIZED\n")


def _make_nested_submodule_fixture(tmp_path: Path) -> tuple[Path, str]:
    grandchild = tmp_path / "grandchild"
    _init_repository(grandchild)
    _write(grandchild, "grandchild.py", "VALUE = 1\n")
    _commit_all(grandchild, "grandchild")

    child = tmp_path / "child"
    _init_repository(child)
    _write(child, "child.py", "VALUE = 2\n")
    _git(
        child,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        os.fspath(grandchild),
        "vendor/grandchild",
    )
    _commit_all(child, "child with nested gitlink")

    root = tmp_path / "root"
    _init_repository(root)
    _seed_audited_sources(root)
    _write(
        root,
        "docs/architecture/"
        "agent_supervisor_proof_directed_planner_doctor.todo.md",
        _minimal_board(),
    )
    _write(
        root,
        "docs/architecture/"
        "agent_supervisor_proof_directed_planner_doctor.objectives.md",
        _minimal_objectives(),
    )
    _write(
        root,
        "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
        _minimal_scheduler(),
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        os.fspath(child),
        "vendor/child",
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "-q",
    )
    baseline = _commit_all(root, "audited baseline")
    _write(root, "live-overlay.txt", "overlay-one\n")
    return root, baseline


def _available_z3_probe() -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "planner-doctor-tool-health@1"
        ),
        "tool_id": "z3",
        "probe_id": "test.metadata",
        "health": "available",
        "capability_ids": ["doctor.pinned_proof_authority"],
        "version": "test-version",
        "reason_codes": [],
        "metadata_only": True,
        "network_used": False,
        "process_started": False,
    }


@pytest.fixture()
def inventory_fixture(
    tmp_path: Path,
) -> tuple[Path, str, PlannerDoctorCapabilityInventory]:
    root, baseline = _make_nested_submodule_fixture(tmp_path)
    inventory = build_planner_doctor_capability_inventory(
        root,
        audited_ref=baseline,
        tool_probes=(_available_z3_probe,),
    )
    return root, baseline, inventory


def test_inventory_binds_baseline_live_overlay_recursive_gitlinks_and_status(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    _root, baseline, inventory = inventory_fixture

    assert inventory.audited_baseline.commit == baseline
    assert inventory.audited_baseline.dirty is False
    assert inventory.current_checkout.commit == baseline
    assert inventory.current_checkout.dirty is True
    assert (
        inventory.audited_baseline.dirty_overlay_cid
        != inventory.current_checkout.dirty_overlay_cid
    )
    assert inventory.audited_baseline.gitlink_closure_complete is True
    assert [item.path for item in inventory.audited_baseline.gitlinks] == [
        "vendor/child",
        "vendor/child/vendor/grandchild",
    ]
    assert [item.depth for item in inventory.audited_baseline.gitlinks] == [0, 1]
    assert all(item.resolved for item in inventory.audited_baseline.gitlinks)

    task_status = {
        item.item_id: item.status for item in inventory.control_status.tasks
    }
    goal_status = {
        item.item_id: item.status for item in inventory.control_status.goals
    }
    assert task_status == {"PDR-000": "completed", "PDR-001": "pending"}
    assert goal_status == {"PDR-G000": "active", "PDR-G010": "active"}
    assert inventory.control_status.completed_task_count == 1


def test_module_presence_does_not_imply_default_wiring(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    _root, _baseline, inventory = inventory_fixture

    analysis = inventory.capability("prompt.repository_analysis")
    admission = inventory.capability("prompt.independent_plan_admission")
    parallel = inventory.capability("planner.parallel_execution_plan")
    transaction = inventory.capability("doctor.live_transaction")
    benchmark = inventory.capability("benchmark.live_paired_runner")

    assert analysis.availability is CapabilityAvailability.SHIPPED
    assert analysis.default_wiring is DefaultWiringState.UNWIRED
    assert admission.availability is CapabilityAvailability.SHIPPED
    assert admission.default_wiring is DefaultWiringState.UNWIRED
    assert parallel.availability is CapabilityAvailability.MISSING
    assert parallel.default_wiring is DefaultWiringState.UNWIRED
    assert transaction.default_wiring is DefaultWiringState.UNSAFE_STUB
    assert benchmark.default_wiring is DefaultWiringState.SYNTHETIC_ONLY
    assert "self_improvement.derived_refill" in inventory.gap_capability_ids

    assert INVENTORY_IS_PROOF_EVIDENCE is False
    assert INVENTORY_IS_COMPLETION_EVIDENCE is False
    assert INVENTORY_AUTHORIZES_MUTATION is False
    assert PACKAGE_PRESENCE_IS_CAPABILITY is False


def test_content_identity_replay_is_order_independent_and_tamper_evident(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    _root, _baseline, inventory = inventory_fixture
    record = inventory.to_record()

    replayed = replay_planner_doctor_capability_inventory(record)
    assert replayed.content_id == inventory.content_id
    assert replay_planner_doctor_capability_inventory(
        inventory.to_json()
    ).content_id == inventory.content_id

    reordered = PlannerDoctorCapabilityInventory(
        audited_baseline=inventory.audited_baseline,
        current_checkout=inventory.current_checkout,
        control_status=inventory.control_status,
        capabilities=tuple(reversed(inventory.capabilities)),
        artifacts=tuple(reversed(inventory.artifacts)),
        configurations=tuple(reversed(inventory.configurations)),
        tool_health=tuple(reversed(inventory.tool_health)),
    )
    assert reordered.content_id == inventory.content_id

    tampered = copy.deepcopy(record)
    tampered["capabilities"][0]["reason_codes"].append("forged_reason")
    with pytest.raises(
        PlannerDoctorInventoryIntegrityError,
        match="content identity mismatch",
    ):
        replay_planner_doctor_capability_inventory(tampered)

    nested_tamper = copy.deepcopy(record)
    nested_tamper["current_checkout"]["tree"] = "0" * 40
    with pytest.raises(
        PlannerDoctorInventoryIntegrityError,
        match="repository revision identity mismatch",
    ):
        replay_planner_doctor_capability_inventory(nested_tamper)

    unsupported = copy.deepcopy(record)
    unsupported["provider_claimed_complete"] = True
    with pytest.raises(
        PlannerDoctorInventoryValidationError,
        match="unsupported fields",
    ):
        replay_planner_doctor_capability_inventory(unsupported)

    without_identity = copy.deepcopy(record)
    without_identity.pop("content_id")
    with pytest.raises(
        PlannerDoctorInventoryIntegrityError,
        match="requires content_id",
    ):
        replay_planner_doctor_capability_inventory(without_identity)


def test_overlay_task_and_config_drift_change_inventory_identity(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    root, baseline, first = inventory_fixture

    _write(root, "live-overlay.txt", "overlay-two\n")
    overlay_changed = build_planner_doctor_capability_inventory(
        root, audited_ref=baseline
    )
    assert (
        overlay_changed.current_checkout.dirty_overlay_cid
        != first.current_checkout.dirty_overlay_cid
    )
    assert overlay_changed.content_id != first.content_id

    _write(
        root,
        "docs/architecture/"
        "agent_supervisor_proof_directed_planner_doctor.todo.md",
        _minimal_board(pdr_001_status="completed"),
    )
    task_changed = build_planner_doctor_capability_inventory(
        root, audited_ref=baseline
    )
    assert (
        task_changed.control_status.taskboard_blob_cid
        != overlay_changed.control_status.taskboard_blob_cid
    )
    assert task_changed.content_id != overlay_changed.content_id

    scheduler_path = (
        root
        / "config"
        / "agent_supervisor_proof_directed_planner_doctor_scheduler.json"
    )
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    scheduler["derived_refill"]["enabled_at_bootstrap"] = True
    scheduler_path.write_text(json.dumps(scheduler, sort_keys=True), encoding="utf-8")
    config_changed = build_planner_doctor_capability_inventory(
        root, audited_ref=baseline
    )
    assert (
        config_changed.capability(
            "self_improvement.derived_refill"
        ).default_wiring
        is DefaultWiringState.WIRED
    )
    assert config_changed.content_id != task_changed.content_id


def test_tool_health_is_injected_metadata_only_and_non_authoritative(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    root, baseline, inventory = inventory_fixture
    tools = {item.tool_id: item for item in inventory.tool_health}

    assert tools["z3"].health is ToolHealthState.AVAILABLE
    assert tools["z3"].version == "test-version"
    assert tools["lean"].health is ToolHealthState.NOT_PROBED
    assert inventory.capability(
        "doctor.pinned_proof_authority"
    ).default_wiring is not DefaultWiringState.WIRED

    invalid_probe = {
        **_available_z3_probe(),
        "tool_id": "unsafe_probe",
        "network_used": True,
    }
    with pytest.raises(
        PlannerDoctorInventoryValidationError,
        match="metadata-only",
    ):
        build_planner_doctor_capability_inventory(
            root,
            audited_ref=baseline,
            tool_probes=(invalid_probe,),
        )


def test_inventory_record_contains_no_source_bodies_private_values_or_host_paths(
    inventory_fixture: tuple[Path, str, PlannerDoctorCapabilityInventory],
) -> None:
    root, _baseline, inventory = inventory_fixture
    serialized = inventory.to_json()

    assert "PRIVATE_SOURCE_BODY_MUST_NOT_BE_SERIALIZED" not in serialized
    assert os.fspath(root) not in serialized
    assert str(root.parent) not in serialized
    assert "requests_import_reachable" in serialized
    assert '"content_id":"b' in serialized


def test_private_configuration_keys_fail_closed(tmp_path: Path) -> None:
    root, baseline = _make_nested_submodule_fixture(tmp_path)
    config = (
        root
        / "config"
        / "agent_supervisor_proof_directed_planner_doctor_scheduler.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["provider"] = {"api_key": "do-not-serialize"}
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PlannerDoctorInventoryValidationError,
        match="private material",
    ):
        build_planner_doctor_capability_inventory(root, audited_ref=baseline)


def test_schema_discovery_and_cid_are_deterministic() -> None:
    schemas = discover_planner_doctor_inventory_schemas()
    assert len(schemas) == len(set(schemas))
    assert all(value.endswith("@1") for value in schemas)
    assert inventory_content_id({"b": 2, "a": 1}) == inventory_content_id(
        {"a": 1, "b": 2}
    )
    with pytest.raises(
        PlannerDoctorInventoryValidationError,
        match="cannot contain floats",
    ):
        inventory_content_id({"unsafe": 1.5})


def test_module_file_is_cold_importable_without_optional_providers() -> None:
    script = f"""
import builtins
import runpy

forbidden = {{
    "aiohttp", "duckdb", "httpx", "neo4j", "requests", "torch",
    "transformers", "urllib3",
}}
original = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError("optional provider imported: " + name)
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
runpy.run_path({os.fspath(MODULE_PATH)!r}, run_name="pdr_inventory_cold")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
