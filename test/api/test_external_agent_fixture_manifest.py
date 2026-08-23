"""EAAEF-140: fixture manifest covers client families and hostile inputs."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FAMILIES = {
    "codex",
    "claude_code",
    "gemini_cli",
    "generic_mcp",
    "repository",
}
REQUIRED_KINDS = {
    "visible_history",
    "truncated_history",
    "branched_history",
    "forgery",
    "failure",
    "dirty_worktree",
    "submodule",
    "lfs",
    "unsupported_language",
    "malicious",
    "large",
    "budget",
}

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "external_agent_handoff"
    / "manifest.json"
)


def test_manifest_covers_required_families_and_hostile_inputs() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"].endswith("external-agent-fixture-manifest@1")
    fixtures = payload["fixtures"]
    families = {item["family"] for item in fixtures}
    kinds = {item["kind"] for item in fixtures}
    assert REQUIRED_FAMILIES <= families
    assert REQUIRED_KINDS <= kinds
    ids = [item["id"] for item in fixtures]
    assert len(ids) == len(set(ids))
    for item in fixtures:
        assert item["path"]
        assert item["id"]
