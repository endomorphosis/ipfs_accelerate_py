"""Seal four authored artifacts before a coordinator receipt.

The receipt commit hash is refused when it is placed inside the hashed body.
Candidate, reviewed, replay-qualified, and promoted selectors each return one
distinct artifact. No remote upload, Hub readback, or ModelManager catalog
write runs.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from .dependency_block_resolution import EXPECTED_RELEASE_COORDINATOR_PROBE

_cached: dict[str, object] | None = None


def measure_release_coordinator_probe() -> dict[str, object] | None:
    """Return the locked seal result, or None when the coordinator check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_RELEASE_COORDINATOR_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_release_coordinator_v1 as coordinator

    source = Path(coordinator.__file__).read_text(encoding="utf-8")
    if "huggingface.co" in source or "requests." in source or "urllib" in source:
        raise RuntimeError("coordinator source reaches a remote publication")
    root = Path(tempfile.mkdtemp(prefix="lir2101-"))
    try:
        unpublished_refused = False
        try:
            coordinator.seal_coordinator_receipt(root, [{
                "relative_path": "missing.bin",
                "sha256": "0" * 64,
                "byte_length": 1,
                "status": "candidate",
                "published": True,
            }])
        except coordinator.ReleaseCoordinatorError:
            unpublished_refused = True
        if (root / coordinator.RECEIPT_NAME).exists():
            raise RuntimeError("receipt written before referenced artifacts")
        payloads = {
            "candidate": b"candidate-artifact",
            "reviewed": b"reviewed-artifact",
            "replay-qualified": b"replay-qualified-artifact",
            "promoted": b"promoted-artifact",
        }
        artifacts = [
            coordinator.publish_referenced_artifact(
                root, status + ".bin", payload, status
            )
            for status, payload in payloads.items()
        ]
        self_listed = False
        try:
            coordinator.seal_coordinator_receipt(root, [
                *artifacts,
                {
                    "relative_path": coordinator.RECEIPT_NAME,
                    "sha256": "ab" * 32,
                    "byte_length": 1,
                    "status": "candidate",
                    "published": True,
                },
            ])
        except coordinator.ReleaseCoordinatorError:
            self_listed = True
        if (root / coordinator.RECEIPT_NAME).exists():
            raise RuntimeError("self-listed receipt was sealed")
        receipt = coordinator.seal_coordinator_receipt(root, artifacts)
        commit = coordinator.coordinator_commit_sha256(receipt)
        receipt_text = (root / coordinator.RECEIPT_NAME).read_text(encoding="utf-8")
        if commit in receipt_text:
            raise RuntimeError("commit hash stored inside the receipt")
        poisoned = dict(receipt)
        poisoned["commit_sha256"] = commit
        self_refused = False
        try:
            coordinator.coordinator_commit_sha256(poisoned)
        except coordinator.ReleaseCoordinatorError:
            self_refused = True
        selected = {
            status: coordinator.select_model_manager_status(receipt["artifacts"], status)
            for status in coordinator.STATUSES
        }
        paths = [set() for _ in coordinator.STATUSES]
        for index, status in enumerate(coordinator.STATUSES):
            paths[index] = {item["relative_path"] for item in selected[status]}
            if any(item["status"] != status for item in selected[status]):
                raise RuntimeError("selector returned another status")
        if any(paths[index] & paths[other] for index in range(4) for other in range(index + 1, 4)):
            raise RuntimeError("selectors overlap")
        if set().union(*paths) != {status + ".bin" for status in coordinator.STATUSES}:
            raise RuntimeError("selectors dropped an artifact")
        collapsed = False
        try:
            coordinator.select_model_manager_status(receipt["artifacts"], "candidate|promoted")
        except coordinator.ReleaseCoordinatorError:
            collapsed = True
        alias_record = {"relative_path": "alias.bin", "status": "reviewed", "selector_alias": "promoted"}
        try:
            coordinator.select_model_manager_status([alias_record], "promoted")
        except coordinator.ReleaseCoordinatorError:
            pass
        else:
            collapsed = False
        if receipt["remote_upload_executed"] is not False or receipt["hub_readback_executed"] is not False:
            raise RuntimeError("remote publication ran")
        if receipt["model_manager_catalog_mutated"] is not False:
            raise RuntimeError("ModelManager catalog mutated")
        if receipt["legal_acceptance_claimed"] is not False or receipt["federal_corpus_acceptance_claimed"] is not False:
            raise RuntimeError("seal claimed legal acceptance")
        for artifact in artifacts:
            path = root / str(artifact["relative_path"])
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
                raise RuntimeError("artifact bytes changed after publication")
        return {
            "admitted": False,
            "artifact_count": len(artifacts),
            "candidate_selector_count": len(selected["candidate"]),
            "federal_corpus_acceptance_claimed": False,
            "formalized": False,
            "hub_readback_executed": False,
            "legal_acceptance_claimed": False,
            "legal_meaning_established": False,
            "model_manager_catalog_mutated": False,
            "model_manager_selectors_disjoint": True,
            "promoted_selector_count": len(selected["promoted"]),
            "proof_authority": False,
            "receipt_lists_itself": not self_listed,
            "referenced_artifacts_published_before_receipt": unpublished_refused and receipt_text.startswith("{"),
            "remote_upload_executed": False,
            "replay_qualified_selector_count": len(selected["replay-qualified"]),
            "reviewed_selector_count": len(selected["reviewed"]),
            "selector_collapses_statuses": not collapsed,
            "self_referential_commit_hash_refused": self_refused,
            "source_semantics_verified": False,
            "statuses": list(coordinator.STATUSES),
            "unpublished_artifact_refused": unpublished_refused,
        }
    finally:
        shutil.rmtree(root)
