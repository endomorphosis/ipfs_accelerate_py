from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation import eaaef_host_admission
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_bootstrap_admission as bootstrap_admission,
)

ROOT = Path(__file__).resolve().parents[2]
ISSUER_PATH = ROOT / "scripts/issue_eaaef_admission_bundle.py"


def _load_issuer() -> object:
    specification = importlib.util.spec_from_file_location(
        "eaaef_admission_bundle_single_capture_test",
        ISSUER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_raw_private_key(
    path: Path,
    key: Ed25519PrivateKey,
) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _prebootstrap_statement(
    *,
    source_head: str,
    source_tree: str,
    board_cid: str,
    materialization_cid: str,
) -> dict[str, object]:
    issued_at_ms = int(time.time() * 1000)
    value: dict[str, object] = {
        field: "" for field in bootstrap_admission._STATEMENT_FIELDS
    }
    value.update(
        {
            "schema": bootstrap_admission.EAAEF_BOOTSTRAP_ADMISSION_STATEMENT_SCHEMA,
            "task_id": "EAAEF-000",
            "board_namespace": (
                "external-agent-autonomous-execution-fabric-v1"
            ),
            "decision": "no_go",
            "outcome": "mutation_not_admitted",
            "blockers": ["EAAEF-191 host admission bundle pending"],
            "board_cid": board_cid,
            "source_head": source_head,
            "source_tree": source_tree,
            "materialization_receipt_cid": materialization_cid,
            "materialization_store_generation": "eaaef-test-run-v1",
            "materialization_database_program_binding_cid": (
                "sha256:" + "6" * 64
            ),
            "materialization_bootstrap_profile_cid": "sha256:" + "7" * 64,
            "materialization_operational_profile_cid": "sha256:" + "8" * 64,
            "provider_qualification_expires_at_ms": 0,
            "provider_maximum_parallel_workers": 0,
            "provider_maximum_parallel_containers": 0,
            "provider_task_dispatch_admitted": False,
            "quack_qualification_expires_at_ms": 0,
            "quack_epoch": 0,
            "quack_fence": 0,
            "authority": dict(bootstrap_admission._EXPECTED_AUTHORITY),
            "one_use_nonce": "single-capture-prebootstrap",
            "issued_at_ms": issued_at_ms,
            "expires_at_ms": issued_at_ms + 3_600_000,
        }
    )
    value.pop("statement_cid", None)
    value["statement_cid"] = bootstrap_admission._cid(value)
    return value


def _child_receipt(
    *,
    task_id: str,
    filename: str,
    capture: int,
    source_head: str,
    source_tree: str,
    board_namespace: str,
    board_cid: str,
    bootstrap_statement: Mapping[str, object],
) -> dict[str, object]:
    decision = {
        "EAAEF-180": "inventory",
        "EAAEF-181": "bound_unadmitted",
    }.get(task_id, "admitted")
    receipt: dict[str, object] = {
        "schema": eaaef_host_admission.RECEIPT_SCHEMA,
        "task_id": task_id,
        "receipt_name": filename,
        "decision": decision,
        "process_started": False,
        "supervisor_process_started": False,
        "self_signed": False,
        "independent_signatures": [],
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
        # Every hypothetical collection produces a different child identity.
        "evidence": {
            "volatile_capture": capture,
            "bootstrap_admission_statement": (
                dict(bootstrap_statement) if task_id == "EAAEF-180" else None
            ),
            "items": (
                [
                    {
                        "blocker": "EAAEF-191 host admission bundle pending",
                        "class": "host_gated_external_authority",
                        "closing_task_ids": ["EAAEF-191"],
                    }
                ]
                if task_id == "EAAEF-180"
                else None
            ),
        },
    }
    receipt["receipt_cid"] = eaaef_host_admission.cid(receipt)
    return receipt


def test_issue_finalizes_one_capture_without_recollecting_volatile_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_issuer()
    process_attempts: list[str] = []

    def reject_process(*_args: object, **_kwargs: object) -> None:
        process_attempts.append("attempted")
        raise AssertionError("admission-bundle finalization attempted a process")

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    monkeypatch.setattr(os, "fork", reject_process)
    monkeypatch.setattr(os, "posix_spawn", reject_process)
    monkeypatch.setattr(os, "posix_spawnp", reject_process)
    monkeypatch.setattr(os, "system", reject_process)
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    signatures_path = receipt_dir / "admission_bundle.signatures.json"
    operator_key = Ed25519PrivateKey.generate()
    reviewer_key = Ed25519PrivateKey.generate()
    operator_key_path = tmp_path / "operator.key"
    reviewer_key_path = tmp_path / "reviewer.key"
    _write_raw_private_key(operator_key_path, operator_key)
    _write_raw_private_key(reviewer_key_path, reviewer_key)
    operator_did = ed25519_did_key(operator_key.public_key())
    reviewer_did = ed25519_did_key(reviewer_key.public_key())

    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(issuer, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(issuer, "BUNDLE_SIGNATURES_PATH", signatures_path)
    monkeypatch.setattr(issuer, "OPERATOR_KEY", operator_key_path)
    monkeypatch.setattr(issuer, "LIFECYCLE_KEY", reviewer_key_path)
    monkeypatch.setattr(issuer, "TRUSTED_OPERATOR_DIDS", (operator_did,))
    monkeypatch.setattr(
        issuer,
        "TRUSTED_SECURITY_REVIEWER_DIDS",
        (reviewer_did,),
    )
    monkeypatch.setattr(
        issuer,
        "lifecycle_root_identity_did",
        lambda: reviewer_did,
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "TRUSTED_OPERATOR_DIDS",
        (operator_did,),
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "TRUSTED_SECURITY_REVIEWER_DIDS",
        (reviewer_did,),
    )

    source_head = "1" * 40
    source_tree = "2" * 40
    board_namespace = "external-agent-autonomous-execution-fabric-v1"
    board_cid = "sha256:" + "3" * 64
    materialization_cid = "sha256:" + "5" * 64
    bootstrap_statement = _prebootstrap_statement(
        source_head=source_head,
        source_tree=source_tree,
        board_cid=board_cid,
        materialization_cid=materialization_cid,
    )
    bootstrap_cid = str(bootstrap_statement["statement_cid"])
    collection_count = 0
    captured_child_cids: list[dict[str, str]] = []

    def collect_once() -> dict[str, object]:
        nonlocal collection_count
        collection_count += 1
        child_cids: dict[str, str] = {}
        child_decisions: dict[str, str] = {}
        for task_id, filename in issuer.RECEIPT_FILES.items():
            if task_id == "EAAEF-191":
                continue
            receipt = _child_receipt(
                task_id=task_id,
                filename=filename,
                capture=collection_count,
                source_head=source_head,
                source_tree=source_tree,
                board_namespace=board_namespace,
                board_cid=board_cid,
                bootstrap_statement=bootstrap_statement,
            )
            (receipt_dir / filename).write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            child_cids[task_id] = str(receipt["receipt_cid"])
            child_decisions[task_id] = str(receipt["decision"])
        captured_child_cids.append(child_cids)
        unsigned_bundle: dict[str, object] = {
            "schema": eaaef_host_admission.BUNDLE_SCHEMA,
            "task_id": "EAAEF-191",
            "receipt_name": issuer.RECEIPT_FILES["EAAEF-191"],
            "decision": "no_go",
            "process_started": False,
            "supervisor_process_started": False,
            "self_signed": False,
            "independent_signatures": [],
            "source_head": source_head,
            "source_tree": source_tree,
            "board_namespace": board_namespace,
            "board_cid": board_cid,
            "evidence": {
                "child_receipt_cids": child_cids,
                "launch_plan_allowed": False,
                "bootstrap_admission_statement_cid": bootstrap_cid,
                "materialization_receipt_cid": materialization_cid,
                "independent_operator_signature": "",
                "independent_security_reviewer_signature": "",
                "operator_did": "",
                "security_reviewer_did": "",
                "independent_signature_present": False,
                "prospective_supervisor_signature_rejected": True,
                "inventory_open_host_gated": [
                    "EAAEF-191 host admission bundle pending"
                ],
            },
        }
        unsigned_bundle["receipt_cid"] = eaaef_host_admission.cid(
            unsigned_bundle
        )
        (receipt_dir / issuer.RECEIPT_FILES["EAAEF-191"]).write_text(
            json.dumps(unsigned_bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "decisions": {**child_decisions, "EAAEF-191": "no_go"},
            "process_started": False,
            "configured_board_launch": False,
            "live_launch_allowed": False,
        }

    signed_reviews: list[dict[str, object]] = []
    original_review = issuer.admission_bundle_review_payload

    def capture_signed_review(**arguments: object) -> dict[str, object]:
        signed_reviews.append(
            {
                **arguments,
                "child_receipt_cids": dict(
                    arguments["child_receipt_cids"]  # type: ignore[arg-type]
                ),
            }
        )
        return original_review(**arguments)

    monkeypatch.setattr(issuer, "collect_and_write", collect_once)
    monkeypatch.setattr(
        issuer,
        "admission_bundle_review_payload",
        capture_signed_review,
    )

    result = issuer.issue()

    assert collection_count == 1
    assert len(captured_child_cids) == 1
    assert len(signed_reviews) == 1
    signed_child_cids = signed_reviews[0]["child_receipt_cids"]
    assert isinstance(signed_child_cids, Mapping)
    assert dict(signed_child_cids) == captured_child_cids[0]

    final_bundle = json.loads(
        (receipt_dir / issuer.RECEIPT_FILES["EAAEF-191"]).read_text(
            encoding="utf-8"
        )
    )
    assert final_bundle["evidence"]["child_receipt_cids"] == dict(
        signed_child_cids
    )
    for task_id, filename in issuer.RECEIPT_FILES.items():
        receipt = json.loads((receipt_dir / filename).read_text(encoding="utf-8"))
        assert receipt["process_started"] is False
        assert receipt["supervisor_process_started"] is False
        if task_id != "EAAEF-191":
            assert receipt["receipt_cid"] == signed_child_cids[task_id]

    hypothetical_second_cids = {
        task_id: str(
            _child_receipt(
                task_id=task_id,
                filename=filename,
                capture=2,
                source_head=source_head,
                source_tree=source_tree,
                board_namespace=board_namespace,
                board_cid=board_cid,
                bootstrap_statement=bootstrap_statement,
            )["receipt_cid"]
        )
        for task_id, filename in issuer.RECEIPT_FILES.items()
        if task_id != "EAAEF-191"
    }
    assert hypothetical_second_cids.keys() == signed_child_cids.keys()
    assert all(
        hypothetical_second_cids[task_id] != signed_child_cids[task_id]
        for task_id in hypothetical_second_cids
    )
    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    assert signatures["supervisor_signed"] is False
    assert signatures["configured_board_launch"] is False
    assert result["decision"] == "admitted"
    assert result["configured_board_launch"] == "false"
    assert process_attempts == []
