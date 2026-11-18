"""Measure separate storage roles without moving a cap.

Budget-expanded lineage, temporary replay, retained evidence, and publication
overlap stay distinct. The native grant stays observational. Foreign assets
stay preserved. A mismatch returns None.
"""

from __future__ import annotations

from pathlib import Path

from .dependency_block_resolution import EXPECTED_STORAGE_CONCURRENCY_PROBE

_BOUNDARY = (
    "Budget-expanded lineage, temporary replay, retained evidence and publication overlap stay separate.",
    "A native grant stays purpose-specific to observation.",
    "Foreign claims and assets stay preserved.",
    "Ledger and cap values are not bypassed.",
)
_cached: dict[str, object] | None = None


def measure_storage_concurrency_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_STORAGE_CONCURRENCY_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _forbidden(text: str) -> bool:
    return "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_storage_concurrency_audit_v1 as audit

    root = Path("/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py")
    audit_text = (root / "logic/autoformal/legal_storage_concurrency_audit_v1.py").read_text(
        encoding="utf-8"
    )
    runtime_text = (
        root / "optimizers/logic_theorem_optimizer/autoencoder_owned_runtime.py"
    ).read_text(encoding="utf-8")
    feed_text = (root / "logic/autoformal/span_cache_feed.py").read_text(encoding="utf-8")
    if _forbidden(audit_text):
        raise RuntimeError("storage concurrency source reaches training or a remote service")
    for sentence in _BOUNDARY:
        if sentence not in audit_text or sentence not in runtime_text or sentence not in feed_text:
            raise RuntimeError("storage concurrency boundary is missing")
    runtime_head = runtime_text.split("STORAGE_CONCURRENCY_BOUNDARY", 1)[0]
    feed_head = feed_text.split("STORAGE_CONCURRENCY_BOUNDARY", 1)[0]
    for sentence in (
        "foreign runtime directory UID",
        "runtime byte limit exceeded",
        "ledger_changed=False",
        "resource_admission=False",
    ):
        if sentence not in runtime_head:
            raise RuntimeError("owned runtime cap contract changed")
    if "class OwnedRuntimeDirectory" not in runtime_head or "def observe" not in runtime_head:
        raise RuntimeError("storage concurrency boundary replaced the runtime contract")
    for sentence in (
        "foreign or incomplete feed database",
        "not a version3 full-report/reference replay or a new source claim.",
        "poll download budget exhausted",
    ):
        if sentence not in feed_head:
            raise RuntimeError("feed preservation contract changed")
    if "class SpanCacheFeed" not in feed_head:
        raise RuntimeError("storage concurrency boundary replaced the feed contract")
    overlap_refused = False
    try:
        audit.audit_storage_concurrency(refusal="overlap")
    except audit.StorageConcurrencyAuditError as error:
        overlap_refused = "publication overlap replaced retained evidence" in str(error)
    grant_refused = False
    try:
        audit.audit_storage_concurrency(refusal="grant")
    except audit.StorageConcurrencyAuditError as error:
        grant_refused = "native grant left its purpose" in str(error)
    foreign_removal_refused = False
    try:
        audit.audit_storage_concurrency(refusal="foreign")
    except audit.StorageConcurrencyAuditError as error:
        foreign_removal_refused = "foreign asset was not preserved" in str(error)
    cap_bypass_refused = False
    try:
        audit.audit_storage_concurrency(refusal="cap")
    except audit.StorageConcurrencyAuditError as error:
        cap_bypass_refused = "ledger or cap value was bypassed" in str(error)
    recorded = audit.audit_storage_concurrency()
    if (
        not overlap_refused
        or not grant_refused
        or not foreign_removal_refused
        or not cap_bypass_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("role_count") != 4
        or recorded.get("roles_distinct") is not True
        or recorded.get("budget_expanded_lineage") != "budget-expanded-lineage"
        or recorded.get("temporary_replay") != "temporary-replay"
        or recorded.get("retained_evidence") != "retained-evidence"
        or recorded.get("publication_overlap") != "publication-overlap"
        or recorded.get("publication_overlap_replaced_retained") is not False
        or recorded.get("native_grant_purpose") != "observation-only"
        or recorded.get("native_grant_purpose_specific") is not True
        or recorded.get("resource_admission") is not False
        or recorded.get("ledger_changed") is not False
        or recorded.get("files_deleted") is not False
        or recorded.get("foreign_asset_disposition") != "preserved"
        or recorded.get("foreign_assets_preserved") is not True
        or recorded.get("external_file_cap") != 134217728
        or recorded.get("ledger_cap") != 16777216
        or recorded.get("cap_bypassed") is not False
        or recorded.get("ledger_bypassed") is not False
        or recorded.get("decoder_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("storage concurrency audit did not keep the named boundary")
    return {
        "admitted": False,
        "authored_fixture_only": True,
        "budget_expanded_lineage": "budget-expanded-lineage",
        "cap_bypassed": False,
        "cap_bypass_refused": True,
        "decoder_executed": False,
        "external_file_cap": 134217728,
        "federal_corpus_scored": False,
        "files_deleted": False,
        "foreign_asset_disposition": "preserved",
        "foreign_assets_preserved": True,
        "foreign_removal_refused": True,
        "formalized": False,
        "grant_refused": True,
        "ledger_bypassed": False,
        "ledger_cap": 16777216,
        "ledger_changed": False,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "model_executed": False,
        "native_grant_purpose": "observation-only",
        "native_grant_purpose_specific": True,
        "overlap_refused": True,
        "publication_overlap": "publication-overlap",
        "publication_overlap_replaced_retained": False,
        "resource_admission": False,
        "retained_evidence": "retained-evidence",
        "role_count": 4,
        "roles_distinct": True,
        "schema": audit.SCHEMA,
        "temporary_replay": "temporary-replay",
        "training_executed": False,
        "worker_launched": False,
    }
