"""Load and structural tests for prompt-only entrypoints v2 plan and rollout."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "data" / "agent_supervisor" / "prompt_only_entrypoints_v2"
PLAN_PATH = V2_ROOT / "plan"
ROLLOUT_PATH = V2_ROOT / "rollout"

REQUIRED_PLAN_MARKERS = (
    "task_id: ASE2-008",
    "canonical_root: data/agent_supervisor/prompt_only_entrypoints_v2",
    "legacy_root: data/agent_supervisor/prompt_only_entrypoints",
    "coordinator_id: agent-supervisor-prompt-only-v2-cutover",
    "completion_authoritative: false",
    "proof_authoritative: false",
    "rollback",
    "cutover",
    "validate_trigger",
    "activate_trigger",
    "rollback_trigger",
)

REQUIRED_ROLLOUT_KEYS = (
    "schema",
    "task_id",
    "track",
    "version",
    "active",
    "state",
    "coordinator",
    "roots",
    "baseline",
    "declared_effect_paths",
    "cutover",
    "rollback",
    "rejection_policy",
    "validation_commands",
    "process_ids",
    "effect_ids",
)

REJECTION_POLICY_KEYS = (
    "unknown_dependency",
    "duplicate_id",
    "cycle",
    "predicted_file_conflict",
    "stale_completion",
    "unauthorized_effect",
    "prompt_secret_leak",
    "duplicate_process_effect",
    "mutable_replica_authority",
    "provider_route_drift",
    "transport_mismatch",
)

SECRET_LEAK_PATTERNS = (
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)secret\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _read_text(path: Path) -> str:
    assert path.is_file(), f"missing required artifact: {path}"
    return path.read_text(encoding="utf-8")


def _load_rollout() -> dict:
    raw = _read_text(ROLLOUT_PATH)
    data = json.loads(raw)
    assert isinstance(data, dict), "rollout must be a JSON object"
    return data


def test_plan_exists_and_is_nonempty():
    text = _read_text(PLAN_PATH)
    assert text.strip(), "plan must be non-empty"
    assert "ASE2-008" in text


def test_plan_names_roots_coordinator_authority_and_triggers():
    text = _read_text(PLAN_PATH)
    for marker in REQUIRED_PLAN_MARKERS:
        assert marker in text, f"plan missing required marker: {marker}"


def test_rollout_exists_and_parses():
    data = _load_rollout()
    for key in REQUIRED_ROLLOUT_KEYS:
        assert key in data, f"rollout missing key: {key}"


def test_rollout_task_and_version():
    data = _load_rollout()
    assert data["task_id"] == "ASE2-008"
    assert data["track"] == "verification-rollout"
    assert data["version"] == 2
    assert data["active"] is False
    assert data["state"] in {"staged", "validated", "materialized"}


def test_rollout_roots_and_coordinator():
    data = _load_rollout()
    roots = data["roots"]
    assert roots["canonical"] == "data/agent_supervisor/prompt_only_entrypoints_v2"
    assert roots["legacy"] == "data/agent_supervisor/prompt_only_entrypoints"
    assert roots["plan"] == "data/agent_supervisor/prompt_only_entrypoints_v2/plan"
    assert roots["rollout"] == "data/agent_supervisor/prompt_only_entrypoints_v2/rollout"

    coord = data["coordinator"]
    assert coord["id"] == "agent-supervisor-prompt-only-v2-cutover"
    assert coord["completion_authoritative"] is False
    assert coord["proof_authoritative"] is False
    assert coord.get("repository_write_allowed") is False


def test_cutover_and_rollback_name_exact_fields():
    data = _load_rollout()
    for section_name in ("cutover", "rollback"):
        section = data[section_name]
        assert isinstance(section, dict)
        assert section.get("name"), f"{section_name} must name the operation"
        assert section.get("from_root"), f"{section_name} must name from_root"
        assert section.get("to_root"), f"{section_name} must name to_root"
        assert section.get("coordinator_id") == "agent-supervisor-prompt-only-v2-cutover"
        assert section.get("authority"), f"{section_name} must name authority"
        assert section.get("trigger"), f"{section_name} must name trigger"

    assert data["cutover"]["to_root"] == data["roots"]["canonical"]
    assert data["rollback"]["from_root"] == data["roots"]["canonical"]
    assert data["rollback"]["to_root"] == data["roots"]["legacy"]


def test_rejection_policy_covers_acceptance_criteria():
    data = _load_rollout()
    policy = data["rejection_policy"]
    for key in REJECTION_POLICY_KEYS:
        assert key in policy, f"rejection_policy missing {key}"
        assert str(policy[key]).lower() in {"reject", "denied", "fail"}


def test_no_duplicate_process_or_effect_ids():
    data = _load_rollout()
    process_ids = data["process_ids"]
    effect_ids = data["effect_ids"]
    assert isinstance(process_ids, list) and process_ids
    assert isinstance(effect_ids, list) and effect_ids
    assert len(process_ids) == len(set(process_ids)), "duplicate process_ids"
    assert len(effect_ids) == len(set(effect_ids)), "duplicate effect_ids"


def test_declared_effect_paths_match_scope():
    data = _load_rollout()
    expected = {
        "data/agent_supervisor/prompt_only_entrypoints_v2/plan",
        "data/agent_supervisor/prompt_only_entrypoints_v2/rollout",
        "test/api/test_agent_supervisor_prompt_only_v2_load.py",
    }
    declared = set(data["declared_effect_paths"])
    assert declared == expected
    # unauthorized effects: nothing outside declared set
    for path in declared:
        assert not path.startswith("/"), "absolute effect paths unauthorized"
        assert ".." not in Path(path).parts, "path traversal unauthorized"


def test_no_prompt_or_secret_leak_in_artifacts():
    for path in (PLAN_PATH, ROLLOUT_PATH):
        text = _read_text(path)
        for pattern in SECRET_LEAK_PATTERNS:
            assert pattern.search(text) is None, f"secret-like content in {path}"


def test_baseline_snapshot_binding():
    data = _load_rollout()
    baseline = data["baseline"]
    assert baseline["snapshot_id"] == "git-commit:8f6bdbe7046687284e17f165115d96a4e6a5f3b7"
    assert baseline["baseline_commit"] == "8f6bdbe7046687284e17f165115d96a4e6a5f3b7"


def test_validation_commands_include_load_suite():
    data = _load_rollout()
    commands = data["validation_commands"]
    assert isinstance(commands, list) and commands
    joined = " ".join(commands)
    assert "test_agent_supervisor_prompt_only_v2_load.py" in joined
    assert "pytest" in joined


def test_no_cycle_in_root_references():
    data = _load_rollout()
    roots = data["roots"]
    # canonical and legacy must differ; cutover edges must not form a self-loop
    assert roots["canonical"] != roots["legacy"]
    assert data["cutover"]["from_root"] != data["cutover"]["to_root"]
    assert data["rollback"]["from_root"] != data["rollback"]["to_root"]


def test_plan_and_rollout_coordinator_align():
    plan = _read_text(PLAN_PATH)
    data = _load_rollout()
    assert data["coordinator"]["id"] in plan
    assert "agent-supervisor-prompt-only-v2-cutover" in plan
