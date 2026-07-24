"""Bounded proof-scheduler observability and queryable receipt projections.

The proof scheduler's durable database is an execution/audit store.  It may
contain provider-owned metadata, proof terms, counterexamples, or diagnostic
transcripts and is consequently the wrong interface for dashboards and
planning policy.  This module creates a deliberately smaller, versioned
projection:

* every metric carries the same seven canonical dimensions;
* records contain identifiers, verdicts, counts, timings, and resource
  measurements, never proof bodies or hidden witnesses; and
* the JSON document can be materialized as normalized DuckDB tables by
  :mod:`artifact_store`.

Inputs may be typed supervisor contracts or plain mappings.  This makes the
projection useful at the live scheduler boundary and when rebuilding metrics
from retained receipts after a restart.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final


PROOF_METRICS_SCHEMA_VERSION: Final = 1
PROOF_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.proof-metrics@1"
)
UNKNOWN_METRIC_DIMENSION: Final = "unknown"
PROOF_METRIC_DIMENSIONS: Final = (
    "goal_cid",
    "subgoal_cid",
    "task_cid",
    "repository_tree_id",
    "provider_id",
    "template_id",
    "resource_class",
)
PROOF_LATENCY_FIELDS: Final = (
    "queue_latency_ms",
    "solver_latency_ms",
    "kernel_latency_ms",
    "model_latency_ms",
    "validation_latency_ms",
    "merge_latency_ms",
    "cancellation_latency_ms",
    "cache_latency_ms",
)
ASSURANCE_LEVELS: Final = (
    "unverified",
    "candidate",
    "solver_checked",
    "kernel_verified",
    "attested",
)

# Every quality rate has additive numerator/denominator counters beside it.
# Rates are recomputed independently for each dimensional row and again from
# the snapshot-wide counter totals; consumers combining selected rows can use
# those counters instead of taking a mathematically invalid average of rates.
PROOF_OPERATIONAL_COUNT_FIELDS: Final = (
    "availability_check_count",
    "availability_success_count",
    "availability_failure_count",
    "schema_validation_count",
    "schema_acceptance_count",
    "schema_rejection_count",
    "proof_closure_count",
    "fallback_count",
    "repair_attempt_count",
    "repair_convergence_count",
    "repair_exhaustion_count",
    "input_token_count",
    "output_token_count",
    "token_count",
    "unsupported_semantics_count",
    "false_completion_prevention_count",
)
PROOF_RATE_FIELDS: Final = (
    "availability_rate",
    "schema_acceptance_rate",
    "proof_closure_rate",
    "fallback_rate",
    "repair_convergence_rate",
    "cache_hit_rate",
)

# These keys are intentionally rejected even when nested in caller-owned
# metadata.  The public projection below mostly uses an allow-list; this
# deny-list also protects extension fields accepted by ``safe_public_value``.
_PRIVATE_KEY_PARTS: Final = (
    "transcript",
    "witness",
    "proof_term",
    "proof_body",
    "raw_proof",
    "private_premise",
    "source_excerpt",
)
_PRIVATE_KEYS: Final = frozenset(
    {
        "stdout",
        "stderr",
        "prompt",
        "response",
        "completion",
        "statement",
        "proof_log",
        "model_output",
        "raw_output",
        "hidden",
        "secret",
        "api_key",
        "access_token",
        "refresh_token",
    }
)
MAX_PUBLIC_TEXT_BYTES: Final = 2_048
MAX_PUBLIC_SEQUENCE_ITEMS: Final = 128
MAX_PUBLIC_MAPPING_ITEMS: Final = 128
MAX_PUBLIC_DEPTH: Final = 4

_PROOF_SNAPSHOT_TABLE_FIELDS: Final = {
    "obligations": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "obligation_id",
            "plan_id",
            "invariant_class",
            "required_assurance",
            "status",
            "ast_scope_ids",
            "premise_count",
            "fallback_check_count",
        )
    ),
    "attempts": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "attempt_id",
            "plan_id",
            "step_id",
            "obligation_id",
            "stage",
            "status",
            "started_at",
            "finished_at",
            "duration_ms",
            "input_count",
            "output_count",
            "evidence_count",
            "error_code",
            "claimed_assurance",
            "authoritative_assurance",
            "cpu_milliseconds",
            "memory_peak_bytes",
            "input_token_count",
            "output_token_count",
            "token_count",
        )
    ),
    "receipts": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "receipt_id",
            "plan_id",
            "attempt_id",
            "obligation_id",
            "repository_id",
            "verdict",
            "assurance",
            "authoritative",
            "freshness",
            "policy_id",
            "translator_id",
            "solver_id",
            "kernel_id",
            "toolchain_id",
            "theorem_registry_id",
            "started_at",
            "finished_at",
            "duration_ms",
            "scope_count",
            "premise_count",
            "evidence_count",
            "assurance_reason_codes",
        )
    ),
    "dependencies": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "plan_id",
            "source_step_id",
            "target_step_id",
            "obligation_id",
            "dependency_kind",
            "satisfied",
        )
    ),
    "cache_outcomes": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "cache_key",
            "obligation_id",
            "receipt_id",
            "outcome",
            "lookup_latency_ms",
            "required_assurance",
            "actual_assurance",
            "fresh",
            "reason_codes",
            "observed_at",
        )
    ),
    "resource_samples": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "observed_at_ms",
            "cpu_percent",
            "memory_percent",
            "disk_percent",
            "memory_used_bytes",
            "memory_available_bytes",
            "disk_used_bytes",
            "disk_available_bytes",
            "active_workers",
            "available_worker_capacity",
            "provider_latency_ms",
            "provider_quota_remaining",
            "provider_token_budget_remaining",
        )
    ),
    "assurance_counts": frozenset(
        (
            *PROOF_METRIC_DIMENSIONS,
            "assurance",
            "receipt_count",
            "authoritative_count",
        )
    ),
}
_PROOF_METRIC_COUNT_FIELDS: Final = (
    "obligation_count",
    "attempt_count",
    "successful_attempt_count",
    "failed_attempt_count",
    "receipt_count",
    "authoritative_receipt_count",
    "dependency_count",
    "cache_hit_count",
    "cache_miss_count",
    "cache_rejection_count",
    "resource_sample_count",
    "cancellation_count",
    *PROOF_OPERATIONAL_COUNT_FIELDS,
)
_PROOF_METRIC_ROW_FIELDS: Final = frozenset(
    (
        *PROOF_METRIC_DIMENSIONS,
        *_PROOF_METRIC_COUNT_FIELDS,
        *PROOF_RATE_FIELDS,
        *PROOF_LATENCY_FIELDS,
        *(field.removesuffix("_ms") + "_seconds" for field in PROOF_LATENCY_FIELDS),
    )
)
_PROOF_SNAPSHOT_TOP_LEVEL_FIELDS: Final = frozenset(
    (
        "schema",
        "schema_version",
        "generated_at",
        "snapshot_id",
        "authoritative",
        "bounded",
        "contains_hidden_witnesses",
        "contains_proof_transcripts",
        "plan_id",
        "plan_ids",
        *_PROOF_SNAPSHOT_TABLE_FIELDS,
        "metrics",
        "latency_metrics",
        "totals",
        "source_counts",
        "query_store",
    )
)


def _utc_iso(value: datetime | str | None = None) -> str:
    if isinstance(value, str) and value.strip():
        parsed = _parse_time(value)
        if parsed is not None:
            return parsed.isoformat()
    current = value if isinstance(value, datetime) else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            # Values which look like epoch milliseconds are common in
            # resource and cache telemetry.
            number = float(value)
            if abs(number) > 10_000_000_000:
                number /= 1000.0
            parsed = datetime.fromtimestamp(number, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_record", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return dict(result)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return dict(attributes)
    return {}


def _records(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _text(value: Any, default: str = "") -> str:
    if isinstance(value, Enum):
        value = value.value
    if value in (None, ""):
        return default
    return str(value).strip() or default


def _first(sources: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for source in sources:
        identity = source.get("identity")
        nested = identity if isinstance(identity, Mapping) else {}
        for candidate in (nested, source):
            for name in names:
                if candidate.get(name) not in (None, ""):
                    return candidate[name]
    return None


def _integer(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _ratio(numerator: Any, denominator: Any) -> float:
    """Return a bounded ratio, using zero for an empty population."""

    bottom = _integer(denominator)
    if bottom <= 0:
        return 0.0
    return round(min(1.0, _integer(numerator) / bottom), 6)


def _token_usage(record: Mapping[str, Any]) -> tuple[int, int, int]:
    """Normalize common provider token accounting without double counting."""

    usage = (
        record.get("resource_usage")
        if isinstance(record.get("resource_usage"), Mapping)
        else {}
    )
    token_usage = (
        record.get("usage")
        if isinstance(record.get("usage"), Mapping)
        else {}
    )
    sources = (record, usage, token_usage)

    def first(names: Sequence[str]) -> int:
        for source in sources:
            for name in names:
                if source.get(name) not in (None, ""):
                    return _integer(source[name])
        return 0

    input_count = first(
        (
            "input_token_count",
            "input_tokens",
            "prompt_token_count",
            "prompt_tokens",
        )
    )
    output_count = first(
        (
            "output_token_count",
            "output_tokens",
            "completion_token_count",
            "completion_tokens",
            "generated_tokens",
        )
    )
    reported_total = first(("token_count", "total_tokens", "tokens"))
    # Some providers report only a total, while others report input/output and
    # a total.  max() preserves either shape without adding the total twice.
    total = max(reported_total, input_count + output_count)
    return input_count, output_count, total


def _limit_integer(value: Any) -> int:
    """Preserve ``-1`` as the resource scheduler's unknown-limit sentinel."""

    if value in (None, "") or isinstance(value, bool):
        return -1
    try:
        return max(-1, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return -1


def _boolean(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "hit", "accepted", "fresh"}:
            return True
        if normalized in {"false", "no", "0", "miss", "rejected", "stale"}:
            return False
    return bool(value)


def _strings(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        else (value,)
    )
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result[:MAX_PUBLIC_SEQUENCE_ITEMS]


def _private_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _PRIVATE_KEYS or any(
        part in normalized for part in _PRIVATE_KEY_PARTS
    )


def safe_public_value(value: Any, *, _depth: int = 0) -> Any:
    """Return a bounded JSON value with private proof material removed.

    The function is public so integrations can apply exactly the same policy
    before attaching optional operator annotations.
    """

    if _depth >= MAX_PUBLIC_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, (bytes, bytearray)):
        return {
            "omitted": True,
            "byte_count": len(value),
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
        }
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) <= MAX_PUBLIC_TEXT_BYTES:
            return value
        return {
            "omitted": True,
            "byte_count": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item))[:MAX_PUBLIC_MAPPING_ITEMS]:
            if _private_key(key):
                continue
            projected = safe_public_value(value[key], _depth=_depth + 1)
            if projected is not None:
                result[str(key)] = projected
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            projected
            for item in value[:MAX_PUBLIC_SEQUENCE_ITEMS]
            if (projected := safe_public_value(item, _depth=_depth + 1)) is not None
        ]
    return _text(value)


def validate_public_projection(value: Any, *, _depth: int = 0) -> None:
    """Reject private or unbounded fields in an already-projected snapshot."""

    if _depth > MAX_PUBLIC_DEPTH + 2:
        raise ValueError("proof metrics projection exceeds its nesting bound")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_PUBLIC_TEXT_BYTES:
            raise ValueError("proof metrics projection contains unbounded text")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_PUBLIC_MAPPING_ITEMS * 4:
            raise ValueError("proof metrics projection contains an oversized mapping")
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized not in {
                "contains_hidden_witnesses",
                "contains_proof_transcripts",
            } and _private_key(key):
                raise ValueError(
                    f"private proof material is not queryable: {key}"
                )
            validate_public_projection(item, _depth=_depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        # Top-level tables are allowed to exceed the per-cell sequence bound;
        # their query results remain bounded by artifact_store.MAX_QUERY_ROWS.
        for item in value:
            validate_public_projection(item, _depth=_depth + 1)


def normalize_proof_metric_identity(
    value: Mapping[str, Any] | Any | None = None,
    defaults: Mapping[str, Any] | Any | None = None,
) -> dict[str, str]:
    """Return the seven mandatory canonical proof metric dimensions."""

    raw = _record(value)
    base = _record(defaults)
    sources = (raw, base)
    goal = _text(
        _first(
            sources,
            ("goal_cid", "canonical_goal_cid", "canonical_goal_id", "goal_id", "goal"),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    subgoal = _text(
        _first(
            sources,
            (
                "subgoal_cid",
                "canonical_subgoal_cid",
                "canonical_subgoal_id",
                "subgoal_id",
                "subgoal",
            ),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    task = _text(
        _first(
            sources,
            (
                "task_cid",
                "canonical_task_cid",
                "canonical_task_id",
                "task_id",
                "task",
            ),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    tree = _text(
        _first(
            sources,
            (
                "repository_tree_id",
                "tree_id",
                "canonical_tree_id",
                "candidate_tree_id",
                "git_tree_id",
            ),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    provider = _text(
        _first(
            sources,
            (
                "provider_id",
                "canonical_provider_id",
                "effective_provider_name",
                "provider",
                "solver_id",
                "kernel_id",
            ),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    template = _text(
        _first(
            sources,
            (
                "template_id",
                "canonical_template_id",
                "obligation_template_id",
                "template",
            ),
        ),
        UNKNOWN_METRIC_DIMENSION,
    )
    resource = _text(
        _first(
            sources,
            (
                "resource_class",
                "canonical_resource_class",
                "resource_pool",
                "worker_class",
            ),
        ),
        "",
    )
    stage = _text(_first(sources, ("stage", "proof_stage", "phase")))
    if resource or stage:
        from .resource_scheduler import normalize_resource_class

        resource = normalize_resource_class(resource, stage=stage)
    resource = resource or UNKNOWN_METRIC_DIMENSION
    return {
        "goal_cid": goal,
        "subgoal_cid": subgoal,
        "task_cid": task,
        "repository_tree_id": tree,
        "provider_id": provider,
        "template_id": template,
        "resource_class": resource,
        # Discoverable aliases help old and new clients agree on which fields
        # are canonical without multiplying grouping dimensions.
        "canonical_goal_id": goal,
        "canonical_subgoal_id": subgoal,
        "canonical_task_id": task,
        "tree_id": tree,
        "canonical_tree_id": tree,
        "canonical_provider_id": provider,
        "canonical_template_id": template,
        "canonical_resource_class": resource,
    }


def _dimension_key(identity: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_text(identity.get(name), UNKNOWN_METRIC_DIMENSION) for name in PROOF_METRIC_DIMENSIONS)


def _duration_ms(record: Mapping[str, Any], *, prefix: str = "") -> int:
    names = (
        f"{prefix}_latency_ms" if prefix else "latency_ms",
        f"{prefix}_duration_ms" if prefix else "duration_ms",
        "elapsed_ms",
    )
    for name in names:
        if record.get(name) not in (None, ""):
            return _integer(record[name])
    seconds_names = (
        f"{prefix}_latency_seconds" if prefix else "latency_seconds",
        f"{prefix}_duration_seconds" if prefix else "duration_seconds",
        "elapsed_seconds",
        "duration",
    )
    for name in seconds_names:
        if record.get(name) not in (None, ""):
            return int(round(_number(record[name]) * 1000.0))
    start = _parse_time(
        record.get("started_at")
        or record.get("queued_at")
        or record.get("created_at")
    )
    finish = _parse_time(
        record.get("finished_at")
        or record.get("completed_at")
        or record.get("updated_at")
    )
    if start is None or finish is None or finish < start:
        return 0
    return int(round((finish - start).total_seconds() * 1000.0))


def _assurance(value: Any) -> str:
    normalized = _text(value, "unverified").lower().replace("-", "_")
    aliases = {
        "solver": "solver_checked",
        "kernel": "kernel_verified",
        "verified": "kernel_verified",
        "attestation": "attested",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ASSURANCE_LEVELS else "unverified"


def _base_metrics(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: identity[name] for name in PROOF_METRIC_DIMENSIONS}
    result.update(
        {
            "obligation_count": 0,
            "attempt_count": 0,
            "successful_attempt_count": 0,
            "failed_attempt_count": 0,
            "receipt_count": 0,
            "authoritative_receipt_count": 0,
            "dependency_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "cache_rejection_count": 0,
            "resource_sample_count": 0,
            "cancellation_count": 0,
            **{field: 0 for field in PROOF_OPERATIONAL_COUNT_FIELDS},
            **{field: 0.0 for field in PROOF_RATE_FIELDS},
            **{field: 0 for field in PROOF_LATENCY_FIELDS},
        }
    )
    return result


def _dedupe_rows(
    rows: Iterable[dict[str, Any]], key_fields: Sequence[str]
) -> list[dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        key = tuple(_text(row.get(name)) for name in key_fields)
        # Empty synthetic identities should not collapse unrelated legacy
        # records; stable source order supplies a bounded local discriminator.
        if not any(key):
            key = (*key, str(ordinal))
        result.setdefault(key, row)
    return [result[key] for key in sorted(result)]


def _public_obligation(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "obligation_id": _text(
            record.get("obligation_id") or record.get("content_id")
        ),
        "plan_id": _text(record.get("plan_id")),
        "invariant_class": _text(record.get("invariant_class")),
        "required_assurance": _assurance(record.get("required_assurance")),
        "status": _text(record.get("status"), "planned").lower(),
        "ast_scope_ids": _strings(record.get("ast_scope_ids") or record.get("scope_ids")),
        "premise_count": len(_strings(record.get("premise_ids"))),
        "fallback_check_count": len(_records(record.get("fallback_checks"))),
    }


def _public_attempt(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    resource_usage = (
        record.get("resource_usage")
        if isinstance(record.get("resource_usage"), Mapping)
        else {}
    )
    stage = _text(record.get("stage"), "unknown").lower()
    input_tokens, output_tokens, total_tokens = _token_usage(record)
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "attempt_id": _text(record.get("attempt_id") or record.get("content_id")),
        "plan_id": _text(record.get("plan_id")),
        "step_id": _text(record.get("step_id")),
        "obligation_id": _text(record.get("obligation_id")),
        "stage": stage,
        "status": _text(record.get("status"), "unknown").lower(),
        "started_at": _text(record.get("started_at")),
        "finished_at": _text(record.get("finished_at")),
        "duration_ms": _duration_ms(record),
        "input_count": len(_strings(record.get("input_ids"))),
        "output_count": len(_strings(record.get("output_ids"))),
        "evidence_count": len(_records(record.get("evidence"))),
        "error_code": _text(record.get("error_code")),
        "claimed_assurance": _assurance(
            record.get("provider_claimed_assurance")
            or record.get("claimed_assurance")
        ),
        "authoritative_assurance": _assurance(
            record.get("authoritative_assurance")
        ),
        "cpu_milliseconds": _integer(
            resource_usage.get("cpu_milliseconds")
            or resource_usage.get("cpu_ms")
        ),
        "memory_peak_bytes": _integer(
            resource_usage.get("memory_peak_bytes")
            or resource_usage.get("peak_memory_bytes")
        ),
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "token_count": total_tokens,
    }


def _public_receipt(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    # Provider verdicts and claimed assurance are not authoritative.  Typed
    # ProofReceipt records publish independently-derived ``authoritative_*``
    # fields; legacy mappings without them fail closed.
    assurance = _assurance(record.get("authoritative_assurance"))
    verdict = _text(
        record.get("authoritative_verdict"),
        "inconclusive",
    ).lower()
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "receipt_id": _text(record.get("receipt_id") or record.get("content_id")),
        "plan_id": _text(record.get("plan_id")),
        "attempt_id": _text(record.get("attempt_id")),
        "obligation_id": _text(record.get("obligation_id")),
        "repository_id": _text(record.get("repository_id")),
        "verdict": verdict,
        "assurance": assurance,
        "authoritative": verdict in {"proved", "disproved"},
        "freshness": _text(record.get("freshness"), "unknown").lower(),
        "policy_id": _text(record.get("policy_id")),
        "translator_id": _text(record.get("translator_id")),
        "solver_id": _text(record.get("solver_id")),
        "kernel_id": _text(record.get("kernel_id")),
        "toolchain_id": _text(record.get("toolchain_id")),
        "theorem_registry_id": _text(record.get("theorem_registry_id")),
        "started_at": _text(record.get("started_at")),
        "finished_at": _text(record.get("finished_at")),
        "duration_ms": _duration_ms(record),
        "scope_count": len(_strings(record.get("ast_scope_ids"))),
        "premise_count": len(_strings(record.get("premise_ids"))),
        "evidence_count": len(_records(record.get("evidence"))),
        "assurance_reason_codes": _strings(record.get("assurance_reason_codes")),
    }


def _public_dependency(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "plan_id": _text(record.get("plan_id")),
        "source_step_id": _text(
            record.get("source_step_id")
            or record.get("step_id")
            or record.get("source")
        ),
        "target_step_id": _text(
            record.get("target_step_id")
            or record.get("depends_on_step_id")
            or record.get("dependency_id")
            or record.get("target")
        ),
        "obligation_id": _text(record.get("obligation_id")),
        "dependency_kind": _text(
            record.get("dependency_kind") or record.get("edge_kind"),
            "requires",
        ),
        "satisfied": (
            None
            if record.get("satisfied") is None
            else _boolean(record.get("satisfied"))
        ),
    }


def _public_cache_outcome(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    key_value = record.get("key")
    cache_key = _text(record.get("cache_key"))
    if not cache_key and key_value is not None:
        cache_key = _text(
            getattr(key_value, "key_id", "")
            or _record(key_value).get("key_id")
            or _record(key_value).get("cache_key_id")
        )
    entry_value = record.get("entry")
    entry = _record(entry_value)
    receipt_value = record.get("receipt") or entry.get("receipt")
    receipt_record = _record(receipt_value)
    raw_outcome = _text(
        record.get("outcome")
        or record.get("status")
        or ("hit" if record.get("cache_hit") is True else "")
    ).lower()
    if raw_outcome not in {"hit", "miss", "rejected", "error", "bypass"}:
        raw_outcome = "miss"
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "cache_key": cache_key,
        "obligation_id": _text(record.get("obligation_id")),
        "receipt_id": _text(
            record.get("receipt_id")
            or receipt_record.get("receipt_id")
            or receipt_record.get("content_id")
        ),
        "outcome": raw_outcome,
        "lookup_latency_ms": _duration_ms(record, prefix="cache"),
        "required_assurance": _assurance(record.get("required_assurance")),
        "actual_assurance": _assurance(
            record.get("actual_assurance") or record.get("assurance")
        ),
        "fresh": (
            None if record.get("fresh") is None else _boolean(record.get("fresh"))
        ),
        "reason_codes": _strings(
            record.get("reason_codes") or record.get("rejection_reasons")
        ),
        "observed_at": _text(
            record.get("observed_at") or record.get("timestamp")
        ),
    }


def _public_resource_sample(
    record: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    return {
        **{name: identity[name] for name in PROOF_METRIC_DIMENSIONS},
        "observed_at_ms": _integer(
            record.get("observed_at_ms")
            or record.get("timestamp_ms")
            or record.get("measured_at_ms")
        ),
        "cpu_percent": _integer(record.get("cpu_percent")),
        "memory_percent": _integer(record.get("memory_percent")),
        "disk_percent": _integer(record.get("disk_percent")),
        "memory_used_bytes": _integer(
            record.get("memory_used_bytes")
            or (
                _integer(record.get("memory_total_bytes"))
                - _integer(record.get("memory_available_bytes"))
            )
        ),
        "memory_available_bytes": _integer(record.get("memory_available_bytes")),
        "disk_used_bytes": _integer(
            record.get("disk_used_bytes")
            or (
                _integer(record.get("disk_total_bytes"))
                - _integer(record.get("disk_available_bytes"))
            )
        ),
        "disk_available_bytes": _integer(record.get("disk_available_bytes")),
        "active_workers": _integer(
            record.get("active_workers") or record.get("occupied_worker_capacity")
        ),
        "available_worker_capacity": _integer(
            record.get("available_worker_capacity")
        ),
        "provider_latency_ms": _integer(record.get("latency_ms")),
        "provider_quota_remaining": _limit_integer(record.get("quota_remaining")),
        "provider_token_budget_remaining": _limit_integer(
            record.get("token_budget_remaining")
        ),
    }


def _add_operational_observation(
    metric: dict[str, Any],
    record: Mapping[str, Any],
    *,
    include_tokens: bool,
) -> None:
    """Reduce one public operational observation into additive counters.

    Explicit ``*_count`` values take precedence over boolean/status inference
    for the same metric family.  This permits both individual lifecycle events
    and already-batched provider telemetry without multiplying observations.
    """

    kind = _text(
        record.get("metric")
        or record.get("phase")
        or record.get("type")
        or record.get("event_type")
    ).lower()

    def add_first(field: str, aliases: Sequence[str]) -> bool:
        for name in (field, *aliases):
            if record.get(name) not in (None, ""):
                metric[field] += _integer(record[name])
                return True
        return False

    availability_explicit = (
        add_first(
            "availability_check_count",
            ("capability_check_count", "availability_probe_count", "route_probe_count"),
        ),
        add_first(
            "availability_success_count",
            ("available_count", "availability_available_count"),
        ),
        add_first(
            "availability_failure_count",
            ("unavailable_count", "availability_unavailable_count"),
        ),
    )
    if not any(availability_explicit):
        for name in (
            "available",
            "availability",
            "route_available",
            "capability_available",
        ):
            if record.get(name) is not None:
                metric["availability_check_count"] += 1
                if _boolean(record[name]):
                    metric["availability_success_count"] += 1
                else:
                    metric["availability_failure_count"] += 1
                break
        else:
            status = _text(record.get("status")).lower()
            if any(token in kind for token in ("availability", "capability", "route_probe")):
                if status in {"available", "ready", "healthy", "success", "succeeded"}:
                    metric["availability_check_count"] += 1
                    metric["availability_success_count"] += 1
                elif status in {"unavailable", "not_ready", "unhealthy", "failed", "error"}:
                    metric["availability_check_count"] += 1
                    metric["availability_failure_count"] += 1

    schema_explicit = (
        add_first(
            "schema_validation_count",
            ("schema_check_count", "schema_attempt_count"),
        ),
        add_first(
            "schema_acceptance_count",
            ("schema_accepted_count", "schema_valid_count"),
        ),
        add_first(
            "schema_rejection_count",
            ("schema_rejected_count", "schema_invalid_count"),
        ),
    )
    if not any(schema_explicit):
        for name in (
            "schema_accepted",
            "schema_acceptance",
            "schema_valid",
        ):
            if record.get(name) is not None:
                metric["schema_validation_count"] += 1
                if _boolean(record[name]):
                    metric["schema_acceptance_count"] += 1
                else:
                    metric["schema_rejection_count"] += 1
                break

    if not add_first(
        "proof_closure_count", ("closed_proof_count", "proof_closed_count")
    ):
        for name in ("proof_closed", "proof_closure", "authoritative_proof_closed"):
            if record.get(name) is not None:
                metric["proof_closure_count"] += int(_boolean(record[name]))
                break

    if not add_first(
        "fallback_count",
        ("deterministic_fallback_count", "fallback_used_count"),
    ):
        for name in ("used_fallback", "fallback_used", "deterministic_fallback"):
            if record.get(name) is not None:
                metric["fallback_count"] += int(_boolean(record[name]))
                break

    add_first(
        "repair_attempt_count",
        ("repair_attempts", "repair_count", "repair_round_count"),
    )
    if not add_first(
        "repair_convergence_count",
        ("repair_converged_count", "converged_repair_count"),
    ):
        for name in ("repair_converged", "repair_convergence"):
            if record.get(name) is not None:
                metric["repair_convergence_count"] += int(_boolean(record[name]))
                break
    if not add_first(
        "repair_exhaustion_count",
        ("repair_exhausted_count", "exhausted_repair_count"),
    ):
        if record.get("repair_exhausted") is not None:
            metric["repair_exhaustion_count"] += int(
                _boolean(record["repair_exhausted"])
            )
    # A singular repair-attempt marker represents one observation even when
    # its integer value is a one-based round ordinal.
    if (
        not any(
            record.get(name) not in (None, "")
            for name in (
                "repair_attempt_count",
                "repair_attempts",
                "repair_count",
                "repair_round_count",
            )
        )
        and any(
            record.get(name) not in (None, "", False)
            for name in ("repair_attempt", "repair_attempted", "repair_round")
        )
    ):
        metric["repair_attempt_count"] += 1

    if not add_first(
        "unsupported_semantics_count",
        ("unsupported_semantic_count",),
    ):
        semantics = record.get("unsupported_semantics")
        if isinstance(semantics, Mapping):
            metric["unsupported_semantics_count"] += len(semantics)
        elif isinstance(semantics, Sequence) and not isinstance(
            semantics, (str, bytes, bytearray)
        ):
            metric["unsupported_semantics_count"] += len(semantics)
        elif semantics not in (None, "", False):
            metric["unsupported_semantics_count"] += 1
        elif _text(record.get("status")).lower() == "unsupported":
            metric["unsupported_semantics_count"] += 1

    if not add_first(
        "false_completion_prevention_count",
        ("false_completion_prevented_count", "prevented_false_completion_count"),
    ):
        for name in (
            "false_completion_prevented",
            "prevented_false_completion",
            "completion_prevented",
        ):
            if record.get(name) is not None:
                metric["false_completion_prevention_count"] += int(
                    _boolean(record[name])
                )
                break

    if include_tokens:
        input_tokens, output_tokens, total_tokens = _token_usage(record)
        metric["input_token_count"] += input_tokens
        metric["output_token_count"] += output_tokens
        metric["token_count"] += total_tokens


def _extract_plan(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_object = getattr(value, "plan", None)
    snapshot_object = getattr(value, "snapshot", None)
    if plan_object is not None:
        return _record(plan_object), _record(snapshot_object)
    record = _record(value)
    if "plan" in record and isinstance(record["plan"], Mapping):
        return dict(record["plan"]), dict(record.get("snapshot") or {})
    return record, {}


def _validate_snapshot_shape(payload: Mapping[str, Any]) -> None:
    unknown_top_level = set(payload) - _PROOF_SNAPSHOT_TOP_LEVEL_FIELDS
    if unknown_top_level:
        raise ValueError(
            "proof metrics snapshot contains unsupported fields: "
            + ", ".join(sorted(str(name) for name in unknown_top_level))
        )
    if payload.get("bounded") is not True:
        raise ValueError("proof metrics snapshot must be marked bounded")
    if payload.get("contains_hidden_witnesses") is not False:
        raise ValueError("proof metrics snapshot cannot contain hidden witnesses")
    if payload.get("contains_proof_transcripts") is not False:
        raise ValueError("proof metrics snapshot cannot contain proof transcripts")

    for table_name, allowed_fields in _PROOF_SNAPSHOT_TABLE_FIELDS.items():
        rows = payload.get(table_name)
        if not isinstance(rows, list):
            raise ValueError(f"proof metrics {table_name} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"proof metrics {table_name} rows must be objects")
            unknown = set(row) - allowed_fields
            if unknown:
                raise ValueError(
                    f"proof metrics {table_name} contains unsupported fields: "
                    + ", ".join(sorted(str(name) for name in unknown))
                )
    for table_name in ("metrics", "latency_metrics"):
        rows = payload.get(table_name)
        if not isinstance(rows, list):
            raise ValueError(f"proof metrics {table_name} must be a list")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) - _PROOF_METRIC_ROW_FIELDS:
                raise ValueError(
                    f"proof metrics {table_name} contains unsupported row fields"
                )

    totals = payload.get("totals")
    allowed_totals = {
        *_PROOF_METRIC_COUNT_FIELDS,
        *PROOF_RATE_FIELDS,
        *PROOF_LATENCY_FIELDS,
        "assurance_counts",
    }
    if not isinstance(totals, Mapping) or set(totals) - allowed_totals:
        raise ValueError("proof metrics totals contains unsupported fields")
    assurance_totals = totals.get("assurance_counts")
    if not isinstance(assurance_totals, Mapping) or set(assurance_totals) - set(
        ASSURANCE_LEVELS
    ):
        raise ValueError("proof metrics totals assurance counts are invalid")

    source_counts = payload.get("source_counts")
    if not isinstance(source_counts, Mapping) or set(source_counts) - {
        *_PROOF_SNAPSHOT_TABLE_FIELDS,
        "events",
    }:
        raise ValueError("proof metrics source counts contain unsupported fields")
    query_store = payload.get("query_store")
    if query_store is not None and (
        not isinstance(query_store, Mapping)
        or set(query_store)
        - {"schema", "artifact_kind", "duckdb_path", "catalog_table"}
    ):
        raise ValueError("proof metrics query store contains unsupported fields")


@dataclass(frozen=True)
class ProofMetricsSnapshot(Mapping[str, Any]):
    """Immutable mapping wrapper used by JSON and artifact-store writers."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(
            json.dumps(dict(self.payload), sort_keys=True, separators=(",", ":"))
        )
        if copied.get("schema") != PROOF_METRICS_SCHEMA:
            raise ValueError("invalid proof metrics schema")
        _validate_snapshot_shape(copied)
        validate_public_projection(copied)
        object.__setattr__(self, "payload", copied)

    @property
    def snapshot_id(self) -> str:
        return str(self.payload.get("snapshot_id") or "")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, sort_keys=True))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


def build_proof_metrics_snapshot(
    plan: Any | None = None,
    *,
    plans: Iterable[Any] = (),
    obligations: Iterable[Any] = (),
    attempts: Iterable[Any] = (),
    receipts: Iterable[Any] = (),
    dependencies: Iterable[Any] = (),
    cache_outcomes: Iterable[Any] = (),
    resource_samples: Iterable[Any] = (),
    events: Iterable[Any] = (),
    scheduler_events: Iterable[Any] = (),
    identity: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    generated_at: datetime | str | None = None,
    now: datetime | str | None = None,
) -> ProofMetricsSnapshot:
    """Build one deterministic proof observability snapshot.

    ``plan`` may also be a :class:`ProofScheduleResult`; its typed attempts and
    receipts are discovered automatically.  Explicit iterables are additive.
    """

    supplied_plans = tuple(plans)
    plan_inputs = (
        ((plan,) if plan is not None else ())
        + supplied_plans
    )
    primary_plan = plan_inputs[0] if plan_inputs else None
    plan_record, schedule_record = _extract_plan(primary_plan)
    plan_ids: list[str] = []
    if len(plan_inputs) > 1:
        merged_steps: list[dict[str, Any]] = []
        merged_obligation_ids: list[str] = []
        merged_attempts: list[dict[str, Any]] = []
        merged_receipts: list[dict[str, Any]] = []
        for plan_input in plan_inputs:
            current_plan, current_schedule = _extract_plan(plan_input)
            current_metadata = (
                current_plan.get("metadata")
                if isinstance(current_plan.get("metadata"), Mapping)
                else {}
            )
            current_plan_id = _text(
                current_plan.get("plan_id") or current_plan.get("content_id")
            )
            if current_plan_id and current_plan_id not in plan_ids:
                plan_ids.append(current_plan_id)
            plan_identity_fields = {
                "plan_id": current_plan_id,
                "task_id": current_plan.get("task_id")
                or current_metadata.get("task_id"),
                "repository_tree_id": current_plan.get("repository_tree_id"),
                "goal_cid": current_metadata.get("goal_cid")
                or current_metadata.get("canonical_goal_cid")
                or current_metadata.get("goal_id"),
                "subgoal_cid": current_metadata.get("subgoal_cid")
                or current_metadata.get("canonical_subgoal_cid")
                or current_metadata.get("subgoal_id"),
            }
            for value in _records(current_plan.get("steps")):
                step = _record(value)
                if step:
                    for name, field_value in plan_identity_fields.items():
                        if field_value not in (None, ""):
                            step.setdefault(name, field_value)
                    merged_steps.append(step)
            merged_obligation_ids.extend(
                _strings(current_plan.get("obligation_ids"))
            )
            for target, values in (
                (merged_attempts, current_schedule.get("attempts")),
                (merged_receipts, current_schedule.get("receipts")),
            ):
                for value in _records(values):
                    record = _record(value)
                    if record:
                        for name, field_value in plan_identity_fields.items():
                            if field_value not in (None, ""):
                                record.setdefault(name, field_value)
                        target.append(record)
        plan_record = {
            **plan_record,
            "steps": merged_steps,
            "obligation_ids": merged_obligation_ids,
        }
        schedule_record = {
            **schedule_record,
            "attempts": merged_attempts,
            "receipts": merged_receipts,
        }
    else:
        only_plan_id = _text(
            plan_record.get("plan_id") or plan_record.get("content_id")
        )
        if only_plan_id:
            plan_ids.append(only_plan_id)
    plan_metadata = (
        plan_record.get("metadata")
        if isinstance(plan_record.get("metadata"), Mapping)
        else {}
    )
    base_identity_record = {
        **dict(defaults or {}),
        **dict(identity or {}),
        **dict(plan_metadata),
        "task_id": plan_record.get("task_id")
        or plan_metadata.get("task_id")
        or dict(identity or {}).get("task_id"),
        "repository_tree_id": plan_record.get("repository_tree_id")
        or dict(identity or {}).get("repository_tree_id"),
    }
    base_identity = normalize_proof_metric_identity(base_identity_record)
    plan_id = _text(plan_record.get("plan_id") or plan_record.get("content_id"))

    step_records = [
        _record(item) for item in _records(plan_record.get("steps")) if _record(item)
    ]
    step_by_key = {
        (_text(item.get("plan_id") or plan_id), _text(item.get("step_id"))): item
        for item in step_records
    }
    steps_by_id: dict[str, list[dict[str, Any]]] = {}
    for item in step_records:
        steps_by_id.setdefault(_text(item.get("step_id")), []).append(item)
    obligation_hints: dict[str, dict[str, Any]] = {}
    for step in step_records:
        metadata = step.get("metadata") if isinstance(step.get("metadata"), Mapping) else {}
        obligation_hints.setdefault(_text(step.get("obligation_id")), {}).update(
            {
                "plan_id": step.get("plan_id") or plan_id,
                "goal_cid": step.get("goal_cid"),
                "subgoal_cid": step.get("subgoal_cid"),
                "task_cid": step.get("task_cid") or step.get("task_id"),
                "repository_tree_id": step.get("repository_tree_id"),
                "provider_id": step.get("provider_id"),
                "template_id": metadata.get("template_id")
                or metadata.get("obligation_template_id"),
                "resource_class": step.get("resource_class"),
                "required_assurance": step.get("required_assurance"),
            }
        )

    raw_obligations = [_record(item) for item in obligations if _record(item)]
    for obligation in raw_obligations:
        obligation_id = _text(
            obligation.get("obligation_id") or obligation.get("content_id")
        )
        hint = obligation_hints.setdefault(obligation_id, {})
        for name in (
            "goal_cid",
            "subgoal_cid",
            "task_cid",
            "repository_tree_id",
            "provider_id",
            "template_id",
            "resource_class",
            "required_assurance",
        ):
            if obligation.get(name) not in (None, ""):
                hint[name] = obligation[name]
    present_obligations = {
        _text(item.get("obligation_id") or item.get("content_id"))
        for item in raw_obligations
    }
    for obligation_id in _strings(plan_record.get("obligation_ids")):
        if obligation_id not in present_obligations:
            raw_obligations.append(
                {"obligation_id": obligation_id, **obligation_hints.get(obligation_id, {})}
            )

    explicit_attempts = [_record(item) for item in attempts if _record(item)]
    explicit_attempts.extend(
        _record(item)
        for item in _records(
            schedule_record.get("attempts")
            or getattr(getattr(primary_plan, "snapshot", None), "attempts", ())
        )
        if _record(item)
    )
    # Durable scheduler state contains a RUNNING record followed by a terminal
    # record for the same execution.  Prefer the terminal record so throughput
    # and latency do not double-count that state transition.
    attempts_by_execution: dict[tuple[str, ...], dict[str, Any]] = {}
    terminal_statuses = {
        "succeeded", "failed", "unsupported", "unavailable", "timed_out",
        "cancelled", "canceled", "blocked",
    }
    for item in sorted(
        explicit_attempts,
        key=lambda record: json.dumps(record, sort_keys=True, default=str),
    ):
        started_at = _text(item.get("started_at"))
        key = (
            _text(item.get("plan_id")),
            _text(item.get("step_id")),
            _text(item.get("provider_id")),
            started_at or _text(item.get("attempt_id")),
        )
        previous = attempts_by_execution.get(key)
        status = _text(item.get("status")).lower()
        previous_status = _text((previous or {}).get("status")).lower()
        if (
            previous is None
            or status in terminal_statuses
            or previous_status not in terminal_statuses
        ):
            attempts_by_execution[key] = item
    explicit_attempts = [
        attempts_by_execution[key] for key in sorted(attempts_by_execution)
    ]
    explicit_receipts = [_record(item) for item in receipts if _record(item)]
    explicit_receipts.extend(
        _record(item)
        for item in _records(
            schedule_record.get("receipts")
            or getattr(getattr(primary_plan, "snapshot", None), "receipts", ())
        )
        if _record(item)
    )

    raw_dependencies = [_record(item) for item in dependencies if _record(item)]
    for step in step_records:
        for dependency_id in _strings(step.get("depends_on")):
            raw_dependencies.append(
                {
                    "plan_id": step.get("plan_id") or plan_id,
                    "source_step_id": _text(step.get("step_id")),
                    "target_step_id": dependency_id,
                    "obligation_id": _text(step.get("obligation_id")),
                    "dependency_kind": "requires",
                    **step,
                }
            )

    def identity_for(record: Mapping[str, Any]) -> dict[str, str]:
        record_plan_id = _text(record.get("plan_id") or plan_id)
        step_id = _text(record.get("step_id"))
        step = step_by_key.get((record_plan_id, step_id), {})
        if not step and len(steps_by_id.get(step_id, ())) == 1:
            step = steps_by_id[step_id][0]
        hint = obligation_hints.get(_text(record.get("obligation_id")), {})
        return normalize_proof_metric_identity(
            {
                **dict(hint),
                **dict(step),
                **dict(
                    step.get("metadata")
                    if isinstance(step.get("metadata"), Mapping)
                    else {}
                ),
                **dict(record),
            },
            base_identity,
        )

    obligation_rows = _dedupe_rows(
        (
            _public_obligation(item, identity_for(item))
            for item in raw_obligations
        ),
        ("obligation_id",),
    )
    attempt_rows = [
        _public_attempt(item, identity_for(item)) for item in explicit_attempts
    ]
    receipt_rows = _dedupe_rows(
        (
            _public_receipt(item, identity_for(item))
            for item in explicit_receipts
        ),
        ("receipt_id",),
    )
    dependency_rows = _dedupe_rows(
        (
            _public_dependency(item, identity_for(item))
            for item in raw_dependencies
        ),
        ("plan_id", "source_step_id", "target_step_id", "dependency_kind"),
    )
    normalized_cache_records: list[dict[str, Any]] = []
    for item in cache_outcomes:
        record = _record(item)
        if not record:
            continue
        key_record = _record(record.get("key"))
        entry_record = _record(record.get("entry"))
        receipt_record = _record(
            record.get("receipt") or entry_record.get("receipt")
        )
        if key_record:
            record.setdefault("obligation_id", key_record.get("obligation"))
            record.setdefault(
                "repository_tree_id", key_record.get("candidate_tree")
            )
            record.setdefault("provider_id", key_record.get("solver"))
            record.setdefault("resource_class", "cpu-proof-solver")
        if receipt_record:
            record.setdefault(
                "actual_assurance",
                receipt_record.get("authoritative_assurance")
                or receipt_record.get("assurance"),
            )
            record.setdefault(
                "receipt_id",
                receipt_record.get("receipt_id")
                or receipt_record.get("content_id"),
            )
        normalized_cache_records.append(record)
    cache_rows = sorted(
        (
            _public_cache_outcome(record, identity_for(record))
            for record in normalized_cache_records
        ),
        key=lambda row: json.dumps(row, sort_keys=True),
    )
    resource_rows = sorted(
        (
            _public_resource_sample(record, identity_for(record))
            for item in resource_samples
            if (record := _record(item))
        ),
        key=lambda row: json.dumps(row, sort_keys=True),
    )

    metric_groups: dict[tuple[str, ...], dict[str, Any]] = {}

    def metrics_for(row: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_proof_metric_identity(row, base_identity)
        key = _dimension_key(normalized)
        if key not in metric_groups:
            metric_groups[key] = _base_metrics(normalized)
        return metric_groups[key]

    for row in obligation_rows:
        metrics_for(row)["obligation_count"] += 1
    for raw_attempt, row in zip(explicit_attempts, attempt_rows):
        metric = metrics_for(row)
        metric["attempt_count"] += 1
        status = _text(row.get("status")).lower()
        if status in {"succeeded", "success", "completed", "passed"}:
            metric["successful_attempt_count"] += 1
        elif status in {"failed", "error", "timed_out", "unsupported", "blocked"}:
            metric["failed_attempt_count"] += 1
        duration = _integer(row.get("duration_ms"))
        stage = _text(row.get("stage")).lower()
        latency_field = {
            "solve": "solver_latency_ms",
            "solver": "solver_latency_ms",
            "reconstruct": "kernel_latency_ms",
            "kernel_verify": "kernel_latency_ms",
            "kernel": "kernel_latency_ms",
            "model_draft": "model_latency_ms",
            "model": "model_latency_ms",
            "validate": "validation_latency_ms",
            "validation": "validation_latency_ms",
            "persist": "merge_latency_ms",
            "merge": "merge_latency_ms",
        }.get(stage)
        if latency_field:
            metric[latency_field] += duration
        if status in {"cancelled", "canceled"}:
            metric["cancellation_count"] += 1
            metric["cancellation_latency_ms"] += duration
        metric["input_token_count"] += _integer(row.get("input_token_count"))
        metric["output_token_count"] += _integer(row.get("output_token_count"))
        metric["token_count"] += _integer(row.get("token_count"))
        _add_operational_observation(
            metric,
            raw_attempt,
            include_tokens=False,
        )
    closed_obligations: set[tuple[str, ...]] = set()
    for row in receipt_rows:
        metric = metrics_for(row)
        metric["receipt_count"] += 1
        if row["authoritative"]:
            metric["authoritative_receipt_count"] += 1
        if row["authoritative"] and row["verdict"] == "proved":
            closure_key = (
                *_dimension_key(row),
                _text(row.get("obligation_id") or row.get("receipt_id")),
            )
            if closure_key not in closed_obligations:
                closed_obligations.add(closure_key)
                metric["proof_closure_count"] += 1
    observed_receipt_claims: set[tuple[str, ...]] = set()
    for raw_receipt in explicit_receipts:
        claim_key = (
            _text(raw_receipt.get("receipt_id") or raw_receipt.get("content_id")),
            _text(raw_receipt.get("attempt_id")),
            _text(raw_receipt.get("obligation_id")),
        )
        if claim_key in observed_receipt_claims:
            continue
        observed_receipt_claims.add(claim_key)
        metric = metrics_for(identity_for(raw_receipt))
        claimed_verdict = _text(
            raw_receipt.get("claimed_verdict")
            or raw_receipt.get("provider_verdict")
            or raw_receipt.get("verdict")
        ).lower()
        authoritative_verdict = _text(
            raw_receipt.get("authoritative_verdict"),
            "inconclusive",
        ).lower()
        previous_prevention_count = metric["false_completion_prevention_count"]
        _add_operational_observation(
            metric,
            raw_receipt,
            include_tokens=False,
        )
        if claimed_verdict in {"proved", "verified", "complete", "completed"} and (
            authoritative_verdict not in {"proved", "verified"}
        ) and metric["false_completion_prevention_count"] == previous_prevention_count:
            metric["false_completion_prevention_count"] += 1
    for row in dependency_rows:
        metrics_for(row)["dependency_count"] += 1
    for row in cache_rows:
        metric = metrics_for(row)
        outcome = row["outcome"]
        if outcome == "hit":
            metric["cache_hit_count"] += 1
        elif outcome == "rejected":
            metric["cache_rejection_count"] += 1
        else:
            metric["cache_miss_count"] += 1
        metric["cache_latency_ms"] += _integer(row["lookup_latency_ms"])
    for row in resource_rows:
        metrics_for(row)["resource_sample_count"] += 1

    event_records = (*tuple(events), *tuple(scheduler_events))
    # Event records supply queue, merge, cancellation, and any timings which
    # are not represented by a ProofAttempt contract.
    for item in event_records:
        record = _record(item)
        if not record:
            continue
        metric = metrics_for(identity_for(record))
        kind = _text(
            record.get("metric")
            or record.get("phase")
            or record.get("type")
            or record.get("event_type")
        ).lower()
        duration = _duration_ms(record)
        explicit_latency = False
        for field in PROOF_LATENCY_FIELDS:
            if record.get(field) not in (None, ""):
                metric[field] += _integer(record[field])
                explicit_latency = True
                continue
            seconds_field = field.removesuffix("_ms") + "_seconds"
            if record.get(seconds_field) not in (None, ""):
                metric[field] += int(
                    round(_number(record[seconds_field]) * 1000)
                )
                explicit_latency = True
        if explicit_latency:
            if "cancel" in kind:
                metric["cancellation_count"] += 1
            _add_operational_observation(metric, record, include_tokens=True)
            continue
        for prefix, field in (
            ("queue", "queue_latency_ms"),
            ("solver", "solver_latency_ms"),
            ("solve", "solver_latency_ms"),
            ("kernel", "kernel_latency_ms"),
            ("model", "model_latency_ms"),
            ("validation", "validation_latency_ms"),
            ("validate", "validation_latency_ms"),
            ("merge", "merge_latency_ms"),
            ("cancel", "cancellation_latency_ms"),
            ("cache", "cache_latency_ms"),
        ):
            if prefix in kind:
                metric[field] += duration
                break
        if "cancel" in kind:
            metric["cancellation_count"] += 1
        _add_operational_observation(metric, record, include_tokens=True)

    metrics = [metric_groups[key] for key in sorted(metric_groups)]
    for row in metrics:
        for field in PROOF_LATENCY_FIELDS:
            row[field.removesuffix("_ms") + "_seconds"] = round(
                row[field] / 1000.0, 6
            )
        closure_population = (
            row["obligation_count"]
            if row["obligation_count"] > 0
            else row["receipt_count"]
        )
        row.update(
            {
                "availability_rate": _ratio(
                    row["availability_success_count"],
                    row["availability_check_count"],
                ),
                "schema_acceptance_rate": _ratio(
                    row["schema_acceptance_count"],
                    row["schema_validation_count"],
                ),
                "proof_closure_rate": _ratio(
                    row["proof_closure_count"],
                    closure_population,
                ),
                "fallback_rate": _ratio(
                    row["fallback_count"],
                    row["attempt_count"],
                ),
                "repair_convergence_rate": _ratio(
                    row["repair_convergence_count"],
                    row["repair_attempt_count"],
                ),
                "cache_hit_rate": _ratio(
                    row["cache_hit_count"],
                    row["cache_hit_count"]
                    + row["cache_miss_count"]
                    + row["cache_rejection_count"],
                ),
            }
        )

    assurance_rows_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in receipt_rows:
        identity_key = _dimension_key(row)
        assurance = _assurance(row.get("assurance"))
        key = (*identity_key, assurance)
        if key not in assurance_rows_by_key:
            assurance_rows_by_key[key] = {
                **{name: row[name] for name in PROOF_METRIC_DIMENSIONS},
                "assurance": assurance,
                "receipt_count": 0,
                "authoritative_count": 0,
            }
        assurance_rows_by_key[key]["receipt_count"] += 1
        assurance_rows_by_key[key]["authoritative_count"] += int(
            bool(row.get("authoritative"))
        )
    # Zero-fill all assurance levels for every observed metric identity so SQL
    # dashboards can compare dimensions without an outer dimension table.
    for metric in metrics:
        identity_key = _dimension_key(metric)
        for assurance in ASSURANCE_LEVELS:
            key = (*identity_key, assurance)
            assurance_rows_by_key.setdefault(
                key,
                {
                    **{
                        name: metric[name]
                        for name in PROOF_METRIC_DIMENSIONS
                    },
                    "assurance": assurance,
                    "receipt_count": 0,
                    "authoritative_count": 0,
                },
            )
    assurance_counts = [
        assurance_rows_by_key[key] for key in sorted(assurance_rows_by_key)
    ]

    totals = {
        key: sum(_integer(row.get(key)) for row in metrics)
        for key in (*_PROOF_METRIC_COUNT_FIELDS, *PROOF_LATENCY_FIELDS)
    }
    closure_population = (
        totals["obligation_count"]
        if totals["obligation_count"] > 0
        else totals["receipt_count"]
    )
    totals.update(
        {
            "availability_rate": _ratio(
                totals["availability_success_count"],
                totals["availability_check_count"],
            ),
            "schema_acceptance_rate": _ratio(
                totals["schema_acceptance_count"],
                totals["schema_validation_count"],
            ),
            "proof_closure_rate": _ratio(
                totals["proof_closure_count"],
                closure_population,
            ),
            "fallback_rate": _ratio(
                totals["fallback_count"],
                totals["attempt_count"],
            ),
            "repair_convergence_rate": _ratio(
                totals["repair_convergence_count"],
                totals["repair_attempt_count"],
            ),
            "cache_hit_rate": _ratio(
                totals["cache_hit_count"],
                totals["cache_hit_count"]
                + totals["cache_miss_count"]
                + totals["cache_rejection_count"],
            ),
        }
    )
    totals["assurance_counts"] = {
        level: sum(
            row["receipt_count"]
            for row in assurance_counts
            if row["assurance"] == level
        )
        for level in ASSURANCE_LEVELS
    }

    material = {
        "schema": PROOF_METRICS_SCHEMA,
        "plan_id": plan_id,
        "plan_ids": sorted(plan_ids),
        "obligations": obligation_rows,
        "attempts": attempt_rows,
        "receipts": receipt_rows,
        "dependencies": dependency_rows,
        "cache_outcomes": cache_rows,
        "resource_samples": resource_rows,
        "assurance_counts": assurance_counts,
        "metrics": metrics,
        "latency_metrics": metrics,
        "totals": totals,
        "source_counts": {
            "obligations": len(obligation_rows),
            "attempts": len(attempt_rows),
            "receipts": len(receipt_rows),
            "dependencies": len(dependency_rows),
            "cache_outcomes": len(cache_rows),
            "resource_samples": len(resource_rows),
            "events": len(event_records),
        },
    }
    snapshot_id = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProofMetricsSnapshot(
        {
            **material,
            "schema_version": PROOF_METRICS_SCHEMA_VERSION,
            "generated_at": _utc_iso(generated_at if generated_at is not None else now),
            "snapshot_id": snapshot_id,
            "authoritative": True,
            "bounded": True,
            "contains_hidden_witnesses": False,
            "contains_proof_transcripts": False,
        }
    )


# Compatibility-oriented descriptive aliases.
proof_metrics_snapshot = build_proof_metrics_snapshot
derive_proof_metrics = build_proof_metrics_snapshot
build_proof_metrics = build_proof_metrics_snapshot


def write_proof_metrics_snapshot(
    path: Path | str,
    snapshot: ProofMetricsSnapshot | Mapping[str, Any],
    *,
    queryable: bool = True,
) -> Path:
    """Atomically write JSON and, by default, its DuckDB query sidecar."""

    target = Path(path)
    payload = (
        snapshot.to_dict()
        if isinstance(snapshot, ProofMetricsSnapshot)
        else ProofMetricsSnapshot(dict(snapshot)).to_dict()
    )
    if queryable:
        from .artifact_store import write_proof_metrics_artifact

        write_proof_metrics_artifact(target, payload)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


persist_proof_metrics = write_proof_metrics_snapshot


def read_proof_metrics_snapshot(path: Path | str) -> ProofMetricsSnapshot | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema") != PROOF_METRICS_SCHEMA:
        return None
    try:
        return ProofMetricsSnapshot(payload)
    except (TypeError, ValueError):
        return None


def query_proof_metrics(path: Path | str, **query: Any) -> dict[str, Any]:
    """Execute one bounded, read-only query against a proof metrics artifact."""

    from .artifact_store import PROOF_METRICS_KIND, query_artifact

    supplied_kind = query.pop("kind", PROOF_METRICS_KIND)
    if supplied_kind != PROOF_METRICS_KIND:
        raise ValueError("proof metrics queries require proof_metrics kind")
    return query_artifact(path, kind=PROOF_METRICS_KIND, **query)


__all__ = [
    "ASSURANCE_LEVELS",
    "MAX_PUBLIC_DEPTH",
    "MAX_PUBLIC_MAPPING_ITEMS",
    "MAX_PUBLIC_SEQUENCE_ITEMS",
    "MAX_PUBLIC_TEXT_BYTES",
    "PROOF_LATENCY_FIELDS",
    "PROOF_METRIC_DIMENSIONS",
    "PROOF_OPERATIONAL_COUNT_FIELDS",
    "PROOF_RATE_FIELDS",
    "PROOF_METRICS_SCHEMA",
    "PROOF_METRICS_SCHEMA_VERSION",
    "ProofMetricsSnapshot",
    "UNKNOWN_METRIC_DIMENSION",
    "build_proof_metrics_snapshot",
    "build_proof_metrics",
    "derive_proof_metrics",
    "normalize_proof_metric_identity",
    "persist_proof_metrics",
    "proof_metrics_snapshot",
    "query_proof_metrics",
    "read_proof_metrics_snapshot",
    "safe_public_value",
    "validate_public_projection",
    "write_proof_metrics_snapshot",
]
