from __future__ import annotations

import base64
import importlib.util
import json
import os
import stat
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
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
    monkeypatch.setattr(issuer, "_require_clean_source_checkout", lambda: None)
    process_attempts: list[str] = []

    def reject_process(*_args: object, **_kwargs: object) -> None:
        process_attempts.append("attempted")
        raise AssertionError("admission-bundle finalization attempted a process")

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    monkeypatch.setattr(os, "fork", reject_process)
    monkeypatch.setattr(os, "posix_spawn", reject_process)
    monkeypatch.setattr(os, "posix_spawnp", reject_process)
    monkeypatch.setattr(os, "system", reject_process)
    tmp_path.chmod(0o700)
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir(mode=0o700)
    receipt_dir.chmod(0o700)
    operator_key = Ed25519PrivateKey.generate()
    reviewer_key = Ed25519PrivateKey.generate()
    operator_did = ed25519_did_key(operator_key.public_key())
    reviewer_did = ed25519_did_key(reviewer_key.public_key())

    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(issuer, "RECEIPT_DIR", receipt_dir)
    authority_root = tmp_path.parent / f"{tmp_path.name}-authority"
    final_dir = authority_root / "host-admission"
    monkeypatch.setattr(
        issuer, "AUTHORITY_ROOT_OVERRIDE", authority_root
    )
    monkeypatch.setattr(eaaef_host_admission, "ROOT", tmp_path)
    monkeypatch.setattr(eaaef_host_admission, "RECEIPT_DIR", receipt_dir)
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
    observed_identity = {
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
    }
    monkeypatch.setattr(issuer, "_source_identity", lambda: dict(observed_identity))
    bundle_path, signatures_path = (
        eaaef_host_admission.source_addressed_admission_bundle_paths(
            final_dir=final_dir,
            source_head=source_head,
        )
    )
    bootstrap_statement = _prebootstrap_statement(
        source_head=source_head,
        source_tree=source_tree,
        board_cid=board_cid,
        materialization_cid=materialization_cid,
    )
    bootstrap_cid = str(bootstrap_statement["statement_cid"])
    collection_count = 0
    captured_child_cids: list[dict[str, str]] = []

    def collect_once() -> dict[str, dict[str, object]]:
        nonlocal collection_count
        collection_count += 1
        receipts: dict[str, dict[str, object]] = {}
        child_cids: dict[str, str] = {}
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
            receipts[task_id] = receipt
            child_cids[task_id] = str(receipt["receipt_cid"])
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
        receipts["EAAEF-191"] = unsigned_bundle
        return receipts

    monkeypatch.setattr(issuer, "collect_host_admission_receipts", collect_once)
    monkeypatch.setattr(
        issuer,
        "materialize_host_evidence",
        lambda: {"decisions": {"capture": "test"}},
    )

    prepared_result = issuer.prepare()
    prepared = prepared_result["prepared_review"]
    review = prepared_result["review"]
    drift_path = receipt_dir / issuer.RECEIPT_FILES["EAAEF-182"]
    original_child = drift_path.read_bytes()
    drifted_child = json.loads(original_child)
    drifted_child["evidence"]["volatile_capture"] = 99
    drifted_child.pop("receipt_cid")
    drifted_child["receipt_cid"] = eaaef_host_admission.cid(drifted_child)
    drift_path.write_text(
        json.dumps(drifted_child, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="bundle child identities differ"):
        issuer.publish(prepared_review=prepared, signatures={})
    drift_path.write_bytes(original_child)
    operator_signature = base64.b64encode(
        operator_key.sign(issuer._canonical(review))
    ).decode("ascii")
    reviewer_payload = {
        **review,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
    }
    reviewer_signature = base64.b64encode(
        reviewer_key.sign(issuer._canonical(reviewer_payload))
    ).decode("ascii")
    signatures = {
        "schema": eaaef_host_admission.BUNDLE_SIGNATURES_SCHEMA,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
        "security_reviewer_did": reviewer_did,
        "security_reviewer_signature": reviewer_signature,
        "payload_sha256": eaaef_host_admission.cid(review),
        "supervisor_signed": False,
        "configured_board_launch": False,
        "decision": prepared_result["decision"],
    }
    observed_identity["source_head"] = "9" * 40
    with pytest.raises(RuntimeError, match="prepared source or board is not current"):
        issuer.publish(prepared_review=prepared, signatures=signatures)
    assert not bundle_path.exists()
    assert not signatures_path.exists()
    observed_identity["source_head"] = source_head
    real_verifier = issuer.verify_admission_bundle_receipt
    monkeypatch.setattr(
        issuer,
        "verify_admission_bundle_receipt",
        lambda **_kwargs: {
            "admitted": False,
            "decision": "admitted",
            "target_decision": "admitted",
            "blockers": ["forced final-verification failure"],
        },
    )
    with pytest.raises(RuntimeError, match="did not verify"):
        issuer.publish(prepared_review=prepared, signatures=signatures)
    assert not bundle_path.exists()
    assert not signatures_path.exists()
    monkeypatch.setattr(issuer, "verify_admission_bundle_receipt", real_verifier)
    real_time = eaaef_host_admission.time.time
    monkeypatch.setattr(
        eaaef_host_admission.time,
        "time",
        lambda: (int(bootstrap_statement["expires_at_ms"]) + 1) / 1000,
    )
    with pytest.raises(RuntimeError, match="pre-bootstrap statement differs"):
        issuer.publish(prepared_review=prepared, signatures=signatures)
    assert not bundle_path.exists()
    assert not signatures_path.exists()
    monkeypatch.setattr(eaaef_host_admission.time, "time", real_time)
    result = issuer.publish(
        prepared_review=prepared,
        signatures=signatures,
    )

    assert collection_count == 1
    assert len(captured_child_cids) == 1
    signed_child_cids = review["child_receipt_cids"]
    assert isinstance(signed_child_cids, Mapping)
    assert dict(signed_child_cids) == captured_child_cids[0]

    final_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert final_bundle["evidence"]["child_receipt_cids"] == dict(
        signed_child_cids
    )
    for task_id, _filename in issuer.RECEIPT_FILES.items():
        receipt_path = (
            bundle_path
            if task_id == "EAAEF-191"
            else eaaef_host_admission.source_addressed_child_receipt_path(
                final_dir=final_dir,
                source_head=source_head,
                task_id=task_id,
            )
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["process_started"] is False
        assert receipt["supervisor_process_started"] is False
        if task_id != "EAAEF-191":
            assert receipt["receipt_cid"] == signed_child_cids[task_id]
            assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400

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
    assert stat.S_IMODE(bundle_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(signatures_path.stat().st_mode) == 0o400
    assert result["decision"] == "admitted"
    assert result["configured_board_launch"] is False
    repeated = issuer.publish(
        prepared_review=prepared,
        signatures=signatures,
    )
    assert repeated["bundle_created"] is False
    assert repeated["signatures_created"] is False
    assert collection_count == 1
    assert process_attempts == []

    for task_id, filename in issuer.RECEIPT_FILES.items():
        if task_id == "EAAEF-191":
            continue
        changed = _child_receipt(
            task_id=task_id,
            filename=filename,
            capture=2,
            source_head=source_head,
            source_tree=source_tree,
            board_namespace=board_namespace,
            board_cid=board_cid,
            bootstrap_statement=bootstrap_statement,
        )
        (receipt_dir / filename).write_text(
            json.dumps(changed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    immutable_verification = eaaef_host_admission.verify_admission_bundle_receipt(
        receipt_dir=receipt_dir,
        expected_source_head=source_head,
        expected_source_tree=source_tree,
        expected_board_namespace=board_namespace,
        expected_board_cid=board_cid,
        require_source_addressed=True,
        final_dir=final_dir,
        final_root=authority_root,
        include_verified_artifacts=True,
    )
    assert immutable_verification["admitted"] is True
    assert immutable_verification["blockers"] == []
    historical_verification = eaaef_host_admission.verify_admission_bundle_receipt(
        receipt_dir=receipt_dir,
        expected_source_head=source_head,
        expected_source_tree=source_tree,
        expected_board_namespace=board_namespace,
        expected_board_cid=board_cid,
        prebootstrap_statement_now_ms=(
            int(bootstrap_statement["expires_at_ms"]) + 1
        ),
        require_source_addressed=True,
        final_dir=final_dir,
        final_root=authority_root,
    )
    assert historical_verification["admitted"] is True
    assert historical_verification["blockers"] == []
    assert (
        immutable_verification["verified_artifacts"]["EAAEF-182"][
            "evidence"
        ]["volatile_capture"]
        == 1
    )


def test_publish_refuses_to_consume_final_authority_for_a_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_issuer()
    monkeypatch.setattr(issuer, "_require_clean_source_checkout", lambda: None)
    monkeypatch.setattr(
        issuer,
        "_validate_prepared_review",
        lambda _prepared: {
            "review": {"capture": "exact"},
            "bundle_template": {},
        },
    )
    monkeypatch.setattr(issuer, "_load_current_child_receipts", lambda: {})
    monkeypatch.setattr(
        issuer,
        "_review_components",
        lambda **_kwargs: {
            "review": {"capture": "exact"},
            "decision": "no_go",
        },
    )
    monkeypatch.setattr(
        issuer,
        "_require_current_identity",
        lambda _components: ("head", "tree", "board"),
    )

    with pytest.raises(RuntimeError, match="no-go evidence cannot consume"):
        issuer.publish(prepared_review={}, signatures={})

    assert not (tmp_path / "final").exists()


def test_source_cleanliness_rejects_non_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issuer = _load_issuer()
    receipt_dir = (
        tmp_path
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "receipts/host_admission"
    )
    monkeypatch.setattr(issuer, "ROOT", tmp_path)
    monkeypatch.setattr(issuer, "RECEIPT_DIR", receipt_dir)
    observed: list[list[str]] = []

    def dirty(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(argv)
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=" M ipfs_accelerate_py/agent_supervisor/runtime/runner.py\n",
            stderr="",
        )

    monkeypatch.setattr(issuer.subprocess, "run", dirty)

    with pytest.raises(RuntimeError, match="non-receipt changes"):
        issuer._require_clean_source_checkout()
    assert observed and any(
        item.startswith(":(top,exclude,literal)")
        for item in observed[0]
    )
