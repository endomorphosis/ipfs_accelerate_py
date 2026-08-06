"""Golden and fail-closed tests for the 211-AI pilot ActionDescriptor catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    CATALOG_ID,
    FORBIDDEN_LOCATOR_KEYS,
    PILOT_LOGICAL_ACTIONS,
    assert_no_executable_locators,
    build_pilot_catalog,
    catalog_digest,
    default_catalog_json_path,
    descriptor_from_public_dict,
    descriptor_to_public_dict,
    export_pilot_catalog_dict,
    load_pilot_catalog_from_dict,
    load_pilot_catalog_json,
    logical_action_to_descriptor_id,
    pilot_descriptors,
    render_pilot_catalog_json,
)
from ipfs_accelerate_py.action_runtime.contracts import RiskClass, SideEffectClass

# Monorepo root: ipfs_accelerate_py/test -> ipfs_accelerate_py -> monorepo
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_JSON = _REPO_ROOT / "data" / "voice_action_dag" / "catalog" / "211ai-pilot-v1.json"


def test_pilot_logical_actions_complete() -> None:
    expected = set(PILOT_LOGICAL_ACTIONS)
    observed = {d.logical_action for d in pilot_descriptors()}
    assert observed == expected
    for name in (
        "handoff_live_agent",
        "open_app_surface",
        "open_wallet_documents",
        "read_calendar",
        "create_calendar_reminder",
        "read_provider_messages",
        "leave_provider_message",
        "open_service_detail",
        "schedule_service_callback",
        "escalate_safety",
    ):
        assert name in observed


def test_build_pilot_catalog_registers_all_and_fail_closed_unknown() -> None:
    catalog = build_pilot_catalog()
    mapping = logical_action_to_descriptor_id()
    assert set(mapping) == set(PILOT_LOGICAL_ACTIONS)
    for logical, descriptor_id in mapping.items():
        descriptor = catalog.require(descriptor_id)
        assert descriptor.logical_action == logical
    assert catalog.get("voice.python.not_a_real_action.v1") is None
    with pytest.raises(KeyError, match="unknown descriptor_id"):
        catalog.require("voice.python.not_a_real_action.v1")


def test_risk_and_confirmation_matrix() -> None:
    by_action = {d.logical_action: d for d in pilot_descriptors()}

    assert by_action["handoff_live_agent"].risk_class is RiskClass.HUMAN
    assert by_action["handoff_live_agent"].adapter == "human"
    assert by_action["handoff_live_agent"].requires_confirmation is True

    assert by_action["open_app_surface"].risk_class is RiskClass.READ
    assert by_action["open_app_surface"].adapter == "python"
    assert by_action["open_app_surface"].requires_confirmation is True

    assert by_action["create_calendar_reminder"].risk_class is RiskClass.WRITE
    assert by_action["create_calendar_reminder"].side_effect_class is SideEffectClass.LOCAL_WRITE
    assert by_action["create_calendar_reminder"].metadata["auth_required"] == "true"

    assert by_action["leave_provider_message"].risk_class is RiskClass.WRITE
    assert (
        by_action["leave_provider_message"].side_effect_class
        is SideEffectClass.EXTERNAL_MUTATION
    )

    assert by_action["schedule_service_callback"].adapter == "workflow"
    assert by_action["schedule_service_callback"].risk_class is RiskClass.WRITE

    assert by_action["escalate_safety"].risk_class is RiskClass.HUMAN
    assert by_action["escalate_safety"].requires_confirmation is False
    assert by_action["escalate_safety"].metadata["confirmation_mode"] == "policy_driven"


def test_catalog_digest_stable_under_key_reordering() -> None:
    baseline = catalog_digest()
    again = catalog_digest()
    assert baseline == again
    assert len(baseline) == 64
    assert all(c in "0123456789abcdef" for c in baseline)

    payload_a = export_pilot_catalog_dict()
    reversed_rows = tuple(reversed(pilot_descriptors()))
    payload_b = export_pilot_catalog_dict(reversed_rows)
    assert payload_a["catalog_digest"] == payload_b["catalog_digest"] == baseline

    # Key order / descriptor list order must not change the digest after load.
    shuffled = {
        "version": payload_a["version"],
        "catalog_digest": payload_a["catalog_digest"],
        "schema": payload_a["schema"],
        "descriptors": list(reversed(payload_a["descriptors"])),
        "logical_actions": list(reversed(payload_a["logical_actions"])),
        "policy_revision": payload_a["policy_revision"],
        "catalog_id": payload_a["catalog_id"],
    }
    catalog = load_pilot_catalog_from_dict(shuffled)
    recomputed = catalog_digest(
        tuple(catalog.require(descriptor_id) for descriptor_id in catalog.list_ids())
    )
    assert recomputed == baseline


def test_public_export_has_no_executable_locators() -> None:
    payload = export_pilot_catalog_dict()
    assert_no_executable_locators(payload)
    text = json.dumps(payload)
    for banned in FORBIDDEN_LOCATOR_KEYS:
        assert f'"{banned}"' not in text


def test_descriptor_rejects_locator_metadata() -> None:
    good = descriptor_to_public_dict(pilot_descriptors()[0])
    bad = dict(good)
    bad["metadata"] = {**dict(good["metadata"]), "executable": "/usr/bin/true"}
    with pytest.raises(ValueError, match="forbidden executable locator|not allowed"):
        assert_no_executable_locators(bad)
    with pytest.raises(ValueError, match="forbidden executable locator|not allowed"):
        descriptor_from_public_dict(bad)


def test_checked_in_json_matches_python_export() -> None:
    assert _CATALOG_JSON.is_file(), f"missing catalog snapshot: {_CATALOG_JSON}"
    on_disk = _CATALOG_JSON.read_text(encoding="utf-8")
    disk_payload = json.loads(on_disk)

    # Durable snapshot is structural (no digests); digests are computed in-process.
    python_export = export_pilot_catalog_dict(include_digests=False)
    assert disk_payload == python_export
    assert json.loads(render_pilot_catalog_json(include_digests=False)) == python_export
    # Re-render is byte-stable for a fixed pilot set.
    assert render_pilot_catalog_json(include_digests=False) == render_pilot_catalog_json(
        include_digests=False
    )

    assert default_catalog_json_path(_REPO_ROOT) == _CATALOG_JSON

    loaded = load_pilot_catalog_json(_CATALOG_JSON)
    mapping = logical_action_to_descriptor_id()
    for logical, descriptor_id in mapping.items():
        assert loaded.require(descriptor_id).logical_action == logical

    assert disk_payload["catalog_id"] == CATALOG_ID
    assert "catalog_digest" not in disk_payload
    assert set(disk_payload["logical_actions"]) == set(PILOT_LOGICAL_ACTIONS)
    assert_no_executable_locators(disk_payload)

    # Digests from the loaded snapshot match the in-process pilot catalog.
    loaded_rows = tuple(
        loaded.require(descriptor_id) for descriptor_id in loaded.list_ids()
    )
    assert catalog_digest(loaded_rows) == catalog_digest()


def test_malformed_catalog_payload_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected catalog_id"):
        load_pilot_catalog_from_dict({"catalog_id": "other", "descriptors": []})

    good = export_pilot_catalog_dict()
    tampered = dict(good)
    tampered["catalog_digest"] = "0" * 64
    with pytest.raises(ValueError, match="catalog_digest mismatch"):
        load_pilot_catalog_from_dict(tampered)

    broken_row = dict(good["descriptors"][0])
    broken_row["descriptor_digest"] = "f" * 64
    with pytest.raises(ValueError, match="descriptor_digest mismatch"):
        descriptor_from_public_dict(broken_row)


def test_json_contains_no_executable_locators_on_disk() -> None:
    text = _CATALOG_JSON.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert_no_executable_locators(payload)
    lowered = text.lower()
    for banned in (
        '"command"',
        '"argv"',
        '"executable"',
        '"import_path"',
        '"cwd"',
        '"shell"',
    ):
        assert banned not in lowered
