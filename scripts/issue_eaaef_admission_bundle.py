#!/usr/bin/env python3
"""Prepare and publish the separately reviewed EAAEF-191 admission bundle.

Preparation captures the host evidence once and emits the exact review object.
It never reads a reviewer key. Publication accepts externally produced
operator and security-reviewer signatures, revalidates every captured child,
and creates the two final artifacts without replacing either one.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.validation.eaaef_authority_registry import (
    EAAEFAuthorityNotFound,
    EAAEFAuthorityRegistry,
    EAAEFAuthorityRegistryError,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    BUNDLE_SCHEMA,
    RECEIPT_DIR,
    RECEIPT_FILES,
    RECEIPT_SCHEMA,
    _source_identity,
    admission_bundle_review_payload,
    admission_bundle_target_decision,
    cid,
    collect_host_admission_receipts,
    materialize_host_evidence,
    source_addressed_admission_bundle_logical_paths,
    source_addressed_child_receipt_logical_path,
    verify_admission_bundle_receipt,
    verify_admission_bundle_signatures_payload,
    verify_prebootstrap_admission_statement,
    write_host_admission_receipts,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
    ExternalAgentBootstrapAdmissionError,
)

PREPARED_REVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-admission-bundle-prepared-review@1"
)
_PREPARED_REVIEW_FIELDS = frozenset(
    {"schema", "review", "bundle_template", "prepared_cid"}
)
_CHILD_TASK_IDS = tuple(
    task_id for task_id in RECEIPT_FILES if task_id != "EAAEF-191"
)
AUTHORITY_ROOT_OVERRIDE: Path | None = None


def _require_clean_source_checkout() -> None:
    """Reject source drift while allowing only generated receipt staging."""

    allowed = tuple(
        RECEIPT_DIR / filename for filename in RECEIPT_FILES.values()
    ) + (RECEIPT_DIR / "admission_bundle.signatures.json",)
    pathspecs = [
        ".",
        *(
            ":(top,exclude,literal)" + path.relative_to(ROOT).as_posix()
            for path in allowed
        ),
    ]
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError("EAAEF-191 source cleanliness is unavailable")
    if completed.stdout.strip():
        raise RuntimeError("EAAEF-191 source checkout has non-receipt changes")
    index = subprocess.run(
        ["/usr/bin/git", "ls-files", "-v", "-z", "--", "."],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if index.returncode != 0:
        raise RuntimeError("EAAEF-191 source index state is unavailable")
    entries = (entry for entry in index.stdout.split("\0") if entry)
    if any(entry[0].islower() or entry.startswith("S ") for entry in entries):
        raise RuntimeError(
            "EAAEF-191 source index hides assume-unchanged or skip-worktree paths"
        )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_ceremony_artifact(path: str | Path, *, noun: str) -> dict[str, Any]:
    """Load one reviewer-supplied object without following its final link."""

    selected = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise RuntimeError(f"{noun} is unavailable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size <= 0
            or before.st_size > 4 * 1024 * 1024
        ):
            raise RuntimeError(f"{noun} is unsafe")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        pathname = os.lstat(selected)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            len(raw) != before.st_size
            or stat.S_ISLNK(pathname.st_mode)
            or tuple(getattr(before, name) for name in stable_fields)
            != tuple(getattr(after, name) for name in stable_fields)
            or (before.st_dev, before.st_ino, before.st_size)
            != (pathname.st_dev, pathname.st_ino, pathname.st_size)
        ):
            raise RuntimeError(f"{noun} changed while read")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{noun} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{noun} is not an object")
    return payload


def _load_current_child_receipts() -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for task_id in _CHILD_TASK_IDS:
        path = RECEIPT_DIR / RECEIPT_FILES[task_id]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{task_id} child receipt is unavailable") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{task_id} child receipt is not an object")
        receipts[task_id] = payload
    return receipts


def _review_components(
    *,
    child_receipts: Mapping[str, Mapping[str, Any]],
    bundle_template: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the exact review from canonical current child receipts."""

    if set(child_receipts) != set(_CHILD_TASK_IDS):
        raise RuntimeError("EAAEF-191 child receipt set differs")
    template_body = {
        key: value for key, value in bundle_template.items() if key != "receipt_cid"
    }
    if (
        bundle_template.get("schema") != BUNDLE_SCHEMA
        or bundle_template.get("task_id") != "EAAEF-191"
        or bundle_template.get("receipt_name") != RECEIPT_FILES["EAAEF-191"]
        or bundle_template.get("receipt_cid") != cid(template_body)
        or bundle_template.get("process_started") is not False
        or bundle_template.get("supervisor_process_started") is not False
        or bundle_template.get("self_signed") is not False
    ):
        raise RuntimeError("EAAEF-191 bundle template identity differs")
    source_head = str(bundle_template.get("source_head") or "")
    source_tree = str(bundle_template.get("source_tree") or "")
    board_namespace = str(bundle_template.get("board_namespace") or "")
    board_cid = str(bundle_template.get("board_cid") or "")
    child_decisions: dict[str, str] = {}
    child_receipt_cids: dict[str, str] = {}
    for task_id in _CHILD_TASK_IDS:
        receipt = child_receipts[task_id]
        body = {key: value for key, value in receipt.items() if key != "receipt_cid"}
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("task_id") != task_id
            or receipt.get("receipt_name") != RECEIPT_FILES[task_id]
            or receipt.get("receipt_cid") != cid(body)
            or receipt.get("source_head") != source_head
            or receipt.get("source_tree") != source_tree
            or receipt.get("board_namespace") != board_namespace
            or receipt.get("board_cid") != board_cid
            or receipt.get("process_started") is not False
            or receipt.get("supervisor_process_started") is not False
            or receipt.get("self_signed") is not False
        ):
            raise RuntimeError(f"{task_id} child receipt identity differs")
        child_decisions[task_id] = str(receipt.get("decision") or "")
        child_receipt_cids[task_id] = str(receipt.get("receipt_cid") or "")

    evidence = bundle_template.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("EAAEF-191 bundle template evidence differs")
    raw_template_children = evidence.get("child_receipt_cids")
    if not isinstance(raw_template_children, Mapping) or {
        str(key): str(value) for key, value in raw_template_children.items()
    } != child_receipt_cids:
        raise RuntimeError("EAAEF-191 bundle child identities differ")
    bootstrap_evidence = child_receipts["EAAEF-180"].get("evidence")
    if not isinstance(bootstrap_evidence, Mapping):
        raise RuntimeError("EAAEF-191 bootstrap evidence differs")
    inventory_items = bootstrap_evidence.get("items")
    if not isinstance(inventory_items, list):
        raise RuntimeError("EAAEF-191 blocker inventory differs")
    inventory_blockers = {
        str(item.get("blocker") or "")
        for item in inventory_items
        if isinstance(item, Mapping)
    }
    open_host_gates = [
        str(item.get("blocker") or "")
        for item in inventory_items
        if isinstance(item, Mapping)
        and item.get("class") == "host_gated_external_authority"
    ]
    if list(evidence.get("inventory_open_host_gated") or ()) != open_host_gates:
        raise RuntimeError("EAAEF-191 open host-gate inventory differs")
    bootstrap_statement = bootstrap_evidence.get("bootstrap_admission_statement")
    bootstrap_cid = str(evidence.get("bootstrap_admission_statement_cid") or "")
    materialization_cid = str(evidence.get("materialization_receipt_cid") or "")
    try:
        verified_statement = verify_prebootstrap_admission_statement(
            statement=bootstrap_statement,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            expected_board_namespace=board_namespace,
            expected_board_cid=board_cid,
            expected_materialization_receipt_cid=materialization_cid,
        )
    except (ExternalAgentBootstrapAdmissionError, TypeError, ValueError) as exc:
        raise RuntimeError("EAAEF-191 pre-bootstrap statement differs") from exc
    if (
        verified_statement.get("statement_cid") != bootstrap_cid
        or not set(verified_statement.get("blockers") or ()).issubset(
            inventory_blockers
        )
    ):
        raise RuntimeError("EAAEF-191 pre-bootstrap statement identity differs")
    decision = admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_preflight_valid=True,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    )
    review = admission_bundle_review_payload(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision=decision,
        launch_plan_allowed=False,
        source_head=source_head,
        source_tree=source_tree,
        board_namespace=board_namespace,
        board_cid=board_cid,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
        inventory_open_host_gated=open_host_gates,
    )
    return {
        "review": review,
        "decision": decision,
        "child_decisions": child_decisions,
        "child_receipt_cids": child_receipt_cids,
        "source_head": source_head,
        "source_tree": source_tree,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
        "bootstrap_admission_statement_cid": bootstrap_cid,
        "materialization_receipt_cid": materialization_cid,
        "inventory_open_host_gated": open_host_gates,
    }


def _require_current_identity(components: Mapping[str, Any]) -> dict[str, str]:
    current = _source_identity()
    expected = {
        field: str(components.get(field) or "")
        for field in ("source_head", "source_tree", "board_namespace", "board_cid")
    }
    if current != expected:
        raise RuntimeError("EAAEF-191 prepared source or board is not current")
    return current


def prepare() -> dict[str, Any]:
    """Capture once and return the unsigned object for external review."""

    _require_clean_source_checkout()
    identity_before = _source_identity()
    materialize = materialize_host_evidence()
    receipts = collect_host_admission_receipts()
    write_host_admission_receipts(receipts, task_ids=_CHILD_TASK_IDS)
    bundle_template = dict(receipts["EAAEF-191"])
    components = _review_components(
        child_receipts={
            task_id: dict(receipts[task_id]) for task_id in _CHILD_TASK_IDS
        },
        bundle_template=bundle_template,
    )
    if _require_current_identity(components) != identity_before:
        raise RuntimeError("EAAEF-191 source changed during preparation")
    _require_clean_source_checkout()
    prepared: dict[str, Any] = {
        "schema": PREPARED_REVIEW_SCHEMA,
        "review": components["review"],
        "bundle_template": bundle_template,
    }
    prepared["prepared_cid"] = cid(prepared)
    return {
        "prepared_review": prepared,
        "review": components["review"],
        "decision": components["decision"],
        "child_receipt_cids": components["child_receipt_cids"],
        "host_evidence": materialize.get("decisions"),
        "published": False,
        "independent_signature_present": False,
        "process_started": False,
        "configured_board_launch": False,
    }


def _validate_prepared_review(prepared: object) -> dict[str, Any]:
    if not isinstance(prepared, Mapping):
        raise RuntimeError("prepared EAAEF-191 review is not an object")
    value = dict(prepared)
    body = {key: item for key, item in value.items() if key != "prepared_cid"}
    if (
        set(value) != _PREPARED_REVIEW_FIELDS
        or value.get("schema") != PREPARED_REVIEW_SCHEMA
        or value.get("prepared_cid") != cid(body)
        or not isinstance(value.get("review"), Mapping)
        or not isinstance(value.get("bundle_template"), Mapping)
    ):
        raise RuntimeError("prepared EAAEF-191 review identity differs")
    return value


def publish(
    *,
    prepared_review: Mapping[str, Any],
    signatures: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate, verify two external reviewers, and publish create-once."""

    _require_clean_source_checkout()
    prepared = _validate_prepared_review(prepared_review)
    child_receipts = _load_current_child_receipts()
    components = _review_components(
        child_receipts=child_receipts,
        bundle_template=dict(prepared["bundle_template"]),
    )
    if _canonical(components["review"]) != _canonical(prepared["review"]):
        raise RuntimeError("prepared EAAEF-191 review differs from current evidence")
    identity_before = _require_current_identity(components)
    if components["decision"] != "admitted":
        raise RuntimeError("EAAEF-191 no-go evidence cannot consume final authority")
    signature_artifact = dict(signatures)
    verified_signatures = verify_admission_bundle_signatures_payload(
        signature_artifact,
        child_decisions=components["child_decisions"],
        child_receipt_cids=components["child_receipt_cids"],
        decision=components["decision"],
        launch_plan_allowed=False,
        source_head=components["source_head"],
        source_tree=components["source_tree"],
        board_namespace=components["board_namespace"],
        board_cid=components["board_cid"],
        bootstrap_admission_statement_cid=components[
            "bootstrap_admission_statement_cid"
        ],
        materialization_receipt_cid=components["materialization_receipt_cid"],
        inventory_open_host_gated=components["inventory_open_host_gated"],
    )
    if not all(verified_signatures.values()):
        raise RuntimeError("separate EAAEF-191 reviewer signatures are invalid")
    bundle_template = dict(prepared["bundle_template"])
    evidence = dict(bundle_template["evidence"])
    evidence.update(
        {
            **verified_signatures,
            "child_receipt_cids": components["child_receipt_cids"],
            "independent_signature_present": True,
        }
    )
    bundle = {
        **bundle_template,
        "decision": components["decision"],
        "evidence": evidence,
    }
    bundle.pop("receipt_cid", None)
    bundle["receipt_cid"] = cid(bundle)
    bundle_logical, signatures_logical = (
        source_addressed_admission_bundle_logical_paths(
            source_head=components["source_head"],
        )
    )
    child_artifacts = tuple(
        (
            source_addressed_child_receipt_logical_path(
                source_head=components["source_head"],
                task_id=task_id,
            ),
            child_receipts[task_id],
        )
        for task_id in _CHILD_TASK_IDS
    )
    if _require_current_identity(components) != identity_before:
        raise RuntimeError("EAAEF-191 source changed before publication")
    source_artifacts = {
        **{task_id: child_receipts[task_id] for task_id in _CHILD_TASK_IDS},
        "EAAEF-191": bundle,
        "EAAEF-191.signatures": signature_artifact,
    }
    verification = verify_admission_bundle_receipt(
        receipt_dir=RECEIPT_DIR,
        expected_source_head=components["source_head"],
        expected_source_tree=components["source_tree"],
        expected_board_namespace=components["board_namespace"],
        expected_board_cid=components["board_cid"],
        require_source_addressed=True,
        source_artifacts=source_artifacts,
    )
    if (
        verification.get("admitted") is not True
        or verification.get("decision") != "admitted"
        or verification.get("target_decision") != "admitted"
        or verification.get("blockers") != []
    ):
        raise RuntimeError(
            "final EAAEF-191 bundle did not verify: "
            + json.dumps(verification, sort_keys=True)
        )
    try:
        registry = EAAEFAuthorityRegistry(
            repo_root=ROOT,
            authority_root=AUTHORITY_ROOT_OVERRIDE,
        )
        with registry.ceremony():
            _require_clean_source_checkout()
            # The persisted verifier intentionally treats the pre-bootstrap
            # no-go as a historical ordering fact.  Establish that fact while
            # it is still current at the create-once ceremony boundary.
            live_components = _review_components(
                child_receipts=child_receipts,
                bundle_template=dict(prepared["bundle_template"]),
            )
            if _canonical(live_components["review"]) != _canonical(
                prepared["review"]
            ):
                raise RuntimeError(
                    "prepared EAAEF-191 review expired before publication"
                )
            child_preexisting: dict[Path, bool] = {}
            for child_path, _child_receipt in child_artifacts:
                try:
                    registry.read_json(child_path)
                except EAAEFAuthorityNotFound:
                    child_preexisting[child_path] = False
                else:
                    child_preexisting[child_path] = True
            try:
                registry.read_json(signatures_logical)
            except EAAEFAuthorityNotFound:
                signatures_created = True
            else:
                signatures_created = False
            try:
                registry.read_json(bundle_logical)
            except EAAEFAuthorityNotFound:
                bundle_created = True
            else:
                bundle_created = False
            for child_path, child_receipt in child_artifacts:
                registry.publish_json(child_path, child_receipt)
            registry.publish_json(signatures_logical, signature_artifact)
            if _require_current_identity(components) != identity_before:
                raise RuntimeError("EAAEF-191 source changed before final commit")
            registry.publish_json(bundle_logical, bundle)
            if _require_current_identity(components) != identity_before:
                raise RuntimeError("EAAEF-191 source changed during publication")
    except EAAEFAuthorityRegistryError as exc:
        raise RuntimeError(f"EAAEF-191 authority registry rejected publication: {exc}") from exc
    bundle_path = registry.physical_path(bundle_logical)
    signatures_path = registry.physical_path(signatures_logical)
    child_snapshots_created = sum(
        not child_preexisting[path] for path, _payload in child_artifacts
    )
    return {
        "operator_did": verified_signatures["operator_did"],
        "security_reviewer_did": verified_signatures["security_reviewer_did"],
        "payload_sha256": cid(components["review"]),
        "bundle_path": str(bundle_path),
        "signatures_path": str(signatures_path),
        "decision": components["decision"],
        "published": True,
        "bundle_created": bundle_created,
        "signatures_created": signatures_created,
        "child_snapshots_created": child_snapshots_created,
        "independent_signature_present": True,
        "configured_board_launch": False,
        "process_started": False,
    }


def issue(
    *,
    prepared_review: Mapping[str, Any],
    signatures: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible programmatic name for the publish phase."""

    return publish(prepared_review=prepared_review, signatures=signatures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--prepared", type=Path, required=True)
    publish_parser.add_argument("--signatures", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare()
    else:
        result = publish(
            prepared_review=_load_ceremony_artifact(
                arguments.prepared,
                noun="prepared EAAEF-191 review",
            ),
            signatures=_load_ceremony_artifact(
                arguments.signatures,
                noun="EAAEF-191 signature artifact",
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
