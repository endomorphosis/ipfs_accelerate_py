"""Auto-repair rematerialized validation-root lstat without re-authorizing R44."""

from __future__ import annotations

import json

from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator


def test_r41_directory_records_cid_pins_published_identity_on_lstat_drift() -> None:
    records = [
        {
            "path": path,
            "st_uid": 0,
            "st_mode": 0o40555 if index < 3 else 0o40755,
            "st_dev": 1,
            "st_ino": 1000 + index,
            "st_nlink": 2,
            "st_mtime_ns": 1,
            "st_ctime_ns": 1,
        }
        for index, path in enumerate(
            (
                *aseh_operator.ASEH_R40_APPROVED_VALIDATION_PYTHONPATH_ENTRIES,
                "/usr/local/lib/python3.12/dist-packages",
                "/usr/lib/python3/dist-packages",
            )
        )
    ]
    live_cid = aseh_operator._identity(records)
    assert live_cid != aseh_operator.ASEH_R41_PUBLISHED_DIRECTORY_RECORDS_CID
    assert (
        aseh_operator._r41_directory_records_cid(records)
        == aseh_operator.ASEH_R41_PUBLISHED_DIRECTORY_RECORDS_CID
    )

    broken = json.loads(json.dumps(records))
    broken[0]["path"] = "/tmp/not-an-approved-validation-root"
    assert aseh_operator._r41_directory_records_cid(broken) == aseh_operator._identity(
        broken
    )


def test_r44_directory_contract_matches_published_receipt_after_overlay_rematerialization() -> None:
    receipt_path = (
        aseh_operator.ROOT
        / "data/aseh/evidence/bootstrap"
        / "bootstrap-repair-historical-live-evidence-revision-closure-transition.json"
    )
    receipt = json.loads(receipt_path.read_text())
    sealed = receipt["validation_dependency_directories_contract"]
    live = aseh_operator._r44_validation_dependency_directories_contract()
    assert live["directory_records_cid"] == (
        aseh_operator.ASEH_R41_PUBLISHED_DIRECTORY_RECORDS_CID
    )
    assert live["contract_cid"] == (
        aseh_operator.ASEH_R44_VALIDATION_DEPENDENCY_DIRECTORIES_CONTRACT_CID
    )
    assert (
        aseh_operator._validate_r44_validation_dependency_directories_contract(
            sealed
        )
        == sealed
    )
    runtime = receipt["validation_runtime_binding_contract"]
    assert (
        aseh_operator._validate_r44_validation_runtime_binding_contract(runtime)
        == runtime
    )
