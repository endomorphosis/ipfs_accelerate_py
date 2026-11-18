"""Measure target-identity invalidation without supplying a teacher.

Two authored targets produce different split, experiment, and teacher
identities. Semantic distillation scope stays off. Engineering retention
stays a separate label. A mismatch returns None.
"""

from __future__ import annotations

from pathlib import Path

from .dependency_block_resolution import EXPECTED_TEACHER_IDENTITY_PROBE

_BOUNDARY = (
    "Changed targets invalidate split, experiment and teacher identities.",
    "Only an independently qualified teacher supplies semantic distillation scope.",
    "Engineering retention stays a separate label.",
    "No independently qualified teacher was supplied.",
)
_cached: dict[str, object] | None = None


def measure_teacher_identity_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_TEACHER_IDENTITY_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _forbidden(text: str) -> bool:
    return "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_teacher_identity_audit_v1 as audit

    root = Path("/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py")
    audit_text = (root / "logic/autoformal/legal_teacher_identity_audit_v1.py").read_text(
        encoding="utf-8"
    )
    queue_text = (root / "logic/autoformal/supervisor_queue.py").read_text(encoding="utf-8")
    example_text = (
        root / "logic/formalization/autoencoder/legal_relational_decoder_example_v1.py"
    ).read_text(encoding="utf-8")
    function_text = queue_text.split("def repair_target_identities", 1)[1]
    for text in (audit_text, function_text, example_text):
        if _forbidden(text):
            raise RuntimeError("teacher identity source reaches training or a remote service")
    for sentence in _BOUNDARY:
        if sentence not in audit_text or sentence not in queue_text or sentence not in example_text:
            raise RuntimeError("teacher identity boundary is missing")
    if "def repair_issue_identity" not in queue_text.split("def repair_target_identities", 1)[0]:
        raise RuntimeError("teacher identity function replaced the repair contract")
    example_head = example_text.split("TEACHER_IDENTITY_BOUNDARY", 1)[0]
    if "def prepare_relational_decoder_example_v1" not in example_head:
        raise RuntimeError("teacher identity boundary replaced the example contract")
    if "LABEL_MASK_BOUNDARY" not in example_head:
        raise RuntimeError("teacher identity boundary replaced the label-mask contract")
    preserved_identity_refused = False
    try:
        audit.audit_teacher_boundaries(refusal="preserved-identity")
    except audit.TeacherIdentityAuditError as error:
        preserved_identity_refused = "changed targets kept an identity" in str(error)
    qualified_teacher_refused = False
    try:
        audit.audit_teacher_boundaries(refusal="qualified-teacher")
    except audit.TeacherIdentityAuditError as error:
        qualified_teacher_refused = "independently qualified teacher was invented" in str(error)
    retention_scope_refused = False
    try:
        audit.audit_teacher_boundaries(refusal="retention-scope")
    except audit.TeacherIdentityAuditError as error:
        retention_scope_refused = (
            "engineering retention supplied distillation scope" in str(error)
        )
    recorded = audit.audit_teacher_boundaries()
    if (
        not preserved_identity_refused
        or not qualified_teacher_refused
        or not retention_scope_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("target_count") != 2
        or recorded.get("split_identity_changed") is not True
        or recorded.get("experiment_identity_changed") is not True
        or recorded.get("teacher_identity_changed") is not True
        or recorded.get("unchanged_target_identities_stable") is not True
        or recorded.get("semantic_distillation_scope_supplied") is not False
        or recorded.get("independently_qualified_teacher_supplied") is not False
        or recorded.get("engineering_retention") != "engineering-retention"
        or recorded.get("engineering_retention_separately_labeled") is not True
        or recorded.get("retention_identity_unchanged") is not True
        or recorded.get("decoder_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("teacher_promoted") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("teacher identity audit did not keep the named boundary")
    return {
        "admitted": False,
        "authored_fixture_only": True,
        "changed_experiment_identity": recorded["changed_experiment_identity"],
        "changed_split_identity": recorded["changed_split_identity"],
        "changed_target_sha256": recorded["changed_target_sha256"],
        "changed_teacher_identity": recorded["changed_teacher_identity"],
        "decoder_executed": False,
        "engineering_retention": "engineering-retention",
        "engineering_retention_separately_labeled": True,
        "experiment_identity": recorded["experiment_identity"],
        "experiment_identity_changed": True,
        "federal_corpus_scored": False,
        "formalized": False,
        "independently_qualified_teacher_supplied": False,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "model_executed": False,
        "preserved_identity_refused": True,
        "qualified_teacher_refused": True,
        "retention_identity_unchanged": True,
        "retention_scope_refused": True,
        "schema": "legal-teacher-identity-audit/v1",
        "semantic_distillation_scope_supplied": False,
        "split_identity": recorded["split_identity"],
        "split_identity_changed": True,
        "target_count": 2,
        "target_sha256": recorded["target_sha256"],
        "teacher_identity": recorded["teacher_identity"],
        "teacher_identity_changed": True,
        "teacher_promoted": False,
        "training_executed": False,
        "unchanged_target_identities_stable": True,
        "worker_launched": False,
    }
