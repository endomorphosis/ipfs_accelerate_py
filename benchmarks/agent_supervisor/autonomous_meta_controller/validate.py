#!/usr/bin/env python3
"""Validate the immutable APMC benchmark inputs without running a model."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "baseline_manifest.json"
CASES = ROOT / "cases.json"
PROGRAM_ID = "agent-supervisor-autonomous-meta-controller-v1"
CASE_IDS = tuple(f"APMC-B{index:03d}" for index in range(1, 17))
REQUIRED_FAMILIES = frozenset(
    {
        "semantic_compression",
        "adversarial_assurance",
        "proof_sealer_verification",
        "low_risk_maintenance",
        "context_omission",
        "provider_unavailable",
        "repeated_retry",
        "human_escalation",
        "stale_cache_changed_tree",
        "conflicting_plan",
        "deterministic_repair",
    }
)
ACTIONS = frozenset(
    {
        "NO_OP",
        "READ_CACHED_RECEIPT",
        "RUN_LOCAL_STATIC_ANALYSIS",
        "RUN_INCREMENTAL_INDEX_QUERY",
        "RUN_GRAPH_RETRIEVAL",
        "EXPAND_CONTEXT_REFERENCE",
        "RUN_SCHEMA_VALIDATION",
        "RUN_TYPE_CHECK",
        "RUN_SELECTED_TEST",
        "RUN_FULL_VALIDATION",
        "RUN_SMT_OR_PROVER",
        "CALL_LOCAL_SMALL_MODEL",
        "CALL_REMOTE_STANDARD_MODEL",
        "CALL_REMOTE_STRONG_MODEL",
        "REQUEST_HUMAN_DECISION",
        "GENERATE_BOUNDED_REPAIR",
        "REPLAN_AFFECTED_SUFFIX",
        "QUARANTINE_TASK",
    }
)


def _load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must contain an object")
    return result


def validate() -> dict[str, Any]:
    manifest = _load(MANIFEST)
    corpus = _load(CASES)
    raw_cases = corpus.get("cases")
    cases = raw_cases if isinstance(raw_cases, list) else []
    errors: list[str] = []
    digest = hashlib.sha256(CASES.read_bytes()).hexdigest()

    if manifest.get("program_id") != PROGRAM_ID or corpus.get("program_id") != PROGRAM_ID:
        errors.append("program identity mismatch")
    if corpus.get("seed") != 20260820 or corpus.get("frozen") is not True:
        errors.append("corpus is not frozen to seed 20260820")
    ids = [item.get("case_id") for item in cases if isinstance(item, dict)]
    if tuple(ids) != CASE_IDS:
        errors.append("case population/order mismatch")
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate case IDs: {duplicates}")
    families = {
        str(item.get("source_family"))
        for item in cases
        if isinstance(item, dict)
    }
    missing_families = sorted(REQUIRED_FAMILIES - families)
    if missing_families:
        errors.append(f"missing source families: {missing_families}")
    malformed = [
        item.get("case_id")
        for item in cases
        if not isinstance(item, dict)
        or item.get("expected_action") not in ACTIONS
        or not isinstance(item.get("model_call_forbidden"), bool)
        or not str(item.get("fixture") or "").strip()
        or len(str(item.get("fixture") or "").encode("utf-8")) > 2048
    ]
    if malformed:
        errors.append(f"malformed cases: {malformed}")
    expected = manifest.get("corpus") if isinstance(manifest.get("corpus"), dict) else {}
    if expected.get("sha256") != digest:
        errors.append("corpus sha256 mismatch")
    if expected.get("case_count") != len(cases) or expected.get("seed") != corpus.get("seed"):
        errors.append("manifest corpus count/seed mismatch")
    measurements = manifest.get("measurements")
    if not isinstance(measurements, dict) or measurements.get("status") != "not_run":
        errors.append("bootstrap measurements must remain explicitly not_run")
    if manifest.get("promotion_eligible") is not False:
        errors.append("unexecuted bootstrap may not be promotion eligible")

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/benchmark-validation@1",
        "valid": not errors,
        "program_id": PROGRAM_ID,
        "case_count": len(cases),
        "corpus_sha256": digest,
        "source_families": sorted(families),
        "measurement_status": measurements.get("status") if isinstance(measurements, dict) else None,
        "promotion_eligible": False,
        "errors": errors,
    }


def main() -> int:
    try:
        report = validate()
    except Exception as exc:
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/benchmark-validation@1",
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
