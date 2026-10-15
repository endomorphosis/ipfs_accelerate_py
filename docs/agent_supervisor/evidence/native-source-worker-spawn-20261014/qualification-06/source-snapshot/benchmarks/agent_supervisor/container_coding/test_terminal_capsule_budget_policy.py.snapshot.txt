"""Explicit advisory selection limits with unchanged legacy defaults."""
import hashlib
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_capsule_selection_policy as policy


def _inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "bytes": (root / name).stat().st_size} for name in policy.BUDGET_SOURCE_PATHS]}


def test_legacy_budget_is_implicit_and_preserves_existing_native_default():
    assert policy.capsule_budget_limit("legacy") == 8
    assert policy.capsule_budget_options("legacy") == policy.capsule_budget_observation("legacy") == {}
    assert policy.validate_capsule_budget("legacy", arm="no-index") == "legacy"
    policy.require_capsule_budget_archive(None, "legacy")


@pytest.mark.parametrize("value,limit", [("one-capsule@1", 1), ("two-capsules@1", 2)])
def test_opt_in_budget_only_selects_a_maximum_and_grants_no_authority(value, limit):
    assert policy.validate_capsule_budget(value, arm="full") == value
    assert policy.capsule_budget_limit(value) == limit
    assert policy.capsule_budget_options(value) == {"capsule_budget": value}
    assert policy.capsule_budget_observation(value) == {
        "capsule_budget": value, "capsule_budget_max_capsules": limit,
        "capsule_budget_token_savings_claimed": False, "capsule_budget_authority": False}
    with pytest.raises(ValueError, match="full indexed"):
        policy.validate_capsule_budget(value, arm="no-index")
    policy.require_capsule_budget_archive(_inventory(), value)


@pytest.mark.parametrize("value", [None, True, False, 0, 1, 2, 8, [], {}, "", "one", "1", "legacy ", "one-capsule@2"])
def test_invalid_budget_refuses_in_all_policy_entrypoints(value):
    for function in (policy.validate_capsule_budget, policy.capsule_budget_limit,
                     policy.capsule_budget_options, policy.capsule_budget_observation):
        with pytest.raises(ValueError, match="supported capsule budget"):
            function(value)


@pytest.mark.parametrize("relative", policy.BUDGET_SOURCE_PATHS)
@pytest.mark.parametrize("mutation", ["missing", "stale", "duplicate", "wrong_size"])
def test_budget_archive_requires_exact_selection_reuse_and_refresh_sources(relative, mutation):
    manifest = _inventory()
    row = next(r for r in manifest["files"] if r["path"] == "source/" + relative)
    if mutation == "missing":
        manifest["files"].remove(row)
    elif mutation == "duplicate":
        manifest["files"].append(dict(row))
    elif mutation == "wrong_size":
        row["bytes"] += 1
    else:
        row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime"):
        policy.require_capsule_budget_archive(manifest, "one-capsule@1")


@pytest.mark.parametrize("manifest", [None, {}, {"files": {}}, {"files": []}])
def test_budget_archive_rejects_absent_inventory(manifest):
    with pytest.raises(ValueError, match="newly qualified runtime"):
        policy.require_capsule_budget_archive(manifest, "two-capsules@1")
