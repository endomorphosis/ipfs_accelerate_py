"""Closed, source-bound, fail-closed checks for the CASF-043 report."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Final

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORT_RELATIVE_PATH = (
    "docs/architecture/causal_event_federation_inventory/final_qualification_report.json"
)
MARKDOWN_RELATIVE_PATH = (
    "docs/architecture/causal_event_federation_inventory/final_qualification_report.md"
)
SUITE_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/manifest.json"
REPORT_PATH = ROOT / REPORT_RELATIVE_PATH
MARKDOWN_PATH = ROOT / MARKDOWN_RELATIVE_PATH
SUITE_PATH = ROOT / SUITE_RELATIVE_PATH

STARTING_REVISION = "84a056e41e48a81d4484be43840196578d6c87da"
STARTING_TREE = "40f0771e77d394ac91d92cc1edb02f7860f6131b"
INPUT_REVISION = "5796d3f78b77b2b6c1c59a2b74c86020a0b141ae"
INPUT_TREE = "14b36ca1f21bfd03dd4b88a7866a0c1a40059249"
SUITE_COMPONENT_REVISION = "75f9487ff051ce5defd6171d7b41dd8127a0d59f"
SUITE_COMPONENT_TREE = "4e31236de005816686e68a336adb1a7fe679e6fa"
SUITE_PROJECTION_REVISION = "edeb276d83713315864d38eb81dc09b519f5360e"
SUITE_PROJECTION_TREE = "1853e8297c44b7b5ee67f8a015fd67107670eb2a"
SUITE_BLOB = "ed4682d69c60509f960f71023c6a38838fdf88f4"
SUITE_RAW_SHA256 = "0d6280f6dc982dae824f24fe6de4ef245afa4f9e30b7ad18ceb48bea71ff3646"
SUITE_ID = "sha256:aa916a4418b4345e90c75b7955f5d0bdefde657158c5fb4f2834ae6cfea3eb0b"

_TRUSTED_GIT: Final = Path("/usr/bin/git")
_TRUSTED_GIT_EXEC_PATH: Final = Path("/usr/lib/git-core")
_TRUSTED_PROCESS_PATH: Final = "/usr/bin:/bin"
_MAX_REPORT_BYTES: Final = 256 * 1024
_MAX_MARKDOWN_BYTES: Final = 256 * 1024
_MAX_BOUND_BLOB_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 14
_MAX_JSON_ITEMS: Final = 4096
_MAX_TEXT_BYTES: Final = 16 * 1024
_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_REF_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SECRET_KEY_RE = re.compile(
    r"^(?:api[_-]?key|access[_-]?token|password|secret|credential|private[_-]?key|authorization)$",
    re.IGNORECASE,
)

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "report_id",
        "artifact_type",
        "artifact_version",
        "program_id",
        "repository_id",
        "root_goal_id",
        "tranche_goal_id",
        "goal_id",
        "task_id",
        "report_kind",
        "status",
        "authority",
        "starting_baseline",
        "qualification_input_snapshot",
        "final_tree_binding",
        "source_bindings",
        "control_plane",
        "capabilities",
        "population_and_concurrency",
        "benchmark_suite",
        "evidence_disposition",
        "result_coverage",
        "safety_gates",
        "claims",
        "qualification",
        "residual_gaps",
        "rollback",
        "provenance",
        "nonclaims",
    }
)
SOURCE_PATHS = (
    "docs/architecture/causal_event_federation_inventory/starting_tree.json",
    "docs/architecture/causal_event_federation_inventory/authorities.json",
    "docs/architecture/causal_event_federation_inventory/capability_snapshot.json",
    "docs/architecture/causal_event_federation_inventory/README.md",
    "docs/architecture/agent_supervisor_causal_event_federation.objectives.md",
    "docs/architecture/agent_supervisor_causal_event_federation.todo.md",
    "scripts/validate_agent_supervisor_causal_event_federation_board.py",
    "config/agent_supervisor_causal_event_federation_scheduler.json",
    "ipfs_accelerate_py/agent_supervisor/federation/promotion.py",
    "test/api/causal_federation/test_promotion.py",
    SUITE_RELATIVE_PATH,
)
BENCHMARK_TASKS = ("CASF-038", "CASF-039", "CASF-040", "CASF-041")
BENCHMARK_SCHEMAS = (
    "casf/idle-benchmark@1",
    "casf/parallel-benchmark@1",
    "casf/load-benchmark@1",
    "casf/token-benchmark@1",
)
BENCHMARK_REASONS = (
    "typed_quack_live_endpoint_not_supplied",
    "qualified_twelve_supervisor_live_capacity_not_admitted",
    "qualified_256_agent_bounded_load_live_capacity_not_admitted",
    "qualified_cross_supervisor_token_live_capacity_not_admitted",
)
PROMOTION_REASON_CODES = frozenset(
    {
        "missing:casf_030_accepted_producer_provenance",
        "missing:casf_030_full_qualification_identity_binding",
        "missing:casf_030_state_owner_provenance",
        "missing:casf_032_accepted_producer_provenance",
        "missing:casf_032_full_qualification_identity_binding",
        "missing:casf_032_state_owner_provenance",
        "missing:casf_033_accepted_producer_provenance",
        "missing:casf_033_full_qualification_identity_binding",
        "missing:casf_033_state_owner_provenance",
        "blocked:casf_034_current_state_owner_capability_unattested",
        "missing:casf_035_control_parity_report_decoder",
        "missing:casf_036_formal_report_decoder",
        "blocked:casf_037_local_qualification_unavailable",
        "unavailable:casf_038_live_not_run",
        "unavailable:casf_039_live_not_run",
        "unavailable:casf_040_live_not_run",
        "unavailable:casf_041_live_not_run",
    }
)
CORE_REASON_CODES = (
    "accepted_current_generation_qualification_identity_unavailable",
    "accepted_current_tree_state_owner_attestation_unavailable",
    "missing:casf_030_accepted_producer_provenance",
    "missing:casf_030_full_qualification_identity_binding",
    "missing:casf_030_state_owner_provenance",
    "missing:casf_033_accepted_producer_provenance",
    "missing:casf_033_full_qualification_identity_binding",
    "missing:casf_033_state_owner_provenance",
    "blocked:casf_034_current_state_owner_capability_unattested",
    "missing:casf_035_control_parity_report_decoder",
    "missing:casf_036_formal_report_decoder",
    "blocked:casf_037_local_qualification_unavailable",
    "unavailable:casf_038_live_not_run",
    "unavailable:casf_039_live_not_run",
    "unavailable:casf_040_live_not_run",
    "unavailable:casf_041_live_not_run",
    "benchmark_result_artifacts_absent",
    "accepted_conjunctive_promotion_decision_unavailable",
)
GAP_IDS = (
    "CASF-043-FINAL-TREE-ACCEPTANCE",
    "CASF-043-LIVE-TYPED-QUACK-STATE-OWNER",
    "CASF-043-CASF-030-PROVENANCE",
    "CASF-043-CASF-032-DUCKLAKE-PROVENANCE",
    "CASF-043-CASF-033-PROVENANCE",
    "CASF-043-CASF-034-STATE-OWNER-CAPABILITY",
    "CASF-043-CASF-035-CONTROL-PARITY-DECODER",
    "CASF-043-CASF-036-FORMAL-DECODER",
    "CASF-043-CASF-037-LOCAL-QUALIFICATION",
    "CASF-043-CASF-038-IDLE-NOT-RUN",
    "CASF-043-CASF-039-PARALLEL-NOT-RUN",
    "CASF-043-CASF-040-LOAD-NOT-RUN",
    "CASF-043-CASF-041-TOKEN-NOT-RUN",
    "CASF-043-BENCHMARK-RESULT-ARTIFACTS",
    "CASF-043-CONJUNCTIVE-PROMOTION-DECISION",
    "CASF-043-ROLLBACK-VERIFICATION",
)
RESULT_FIELDS = frozenset(
    {
        "task_and_dedup",
        "causal_graph_and_abstraction",
        "interventions_and_independence",
        "events_outbox_dead_letters_and_wakeups",
        "idle_behavior",
        "parallel_throughput_and_merge",
        "model_context_and_token_efficiency",
        "proof_and_validation",
        "ducklake_projection",
        "recovery_and_failures",
    }
)
SAFETY_FIELDS = frozenset(
    {
        "direct_multi_process_file_mutation",
        "store_ambiguity",
        "event_loss",
        "duplicate_committed_effects",
        "stale_fence_completion",
        "unauthorized_creation",
        "tenant_leakage",
        "agent_sql",
        "secret_leakage",
        "causal_notification_loss",
        "nomination_or_stale_map_authority",
        "cycle_or_shard_corruption",
        "forbidden_idle_activity",
        "replay_idempotency",
        "ownership_effect_or_merge_corruption",
        "reduced_assurance",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "event_driven",
        "causally_coordinated",
        "multi_supervisor",
        "parallel",
        "contention_free",
        "token_efficient",
        "production_ready",
        "duckdb_quack_qualified",
        "ducklake_quack_qualified",
        "ducklake_promotion_qualified",
        "exactly_once_network_delivery",
    }
)
NONCLAIMS = (
    "This report is not a federation-completion receipt, accepted qualification identity, policy decision, release authority, or promotion decision.",
    "Repository bindings verify bytes and lineage only; they do not verify a live control plane, population, event transport, safety gate, benchmark result, or provider execution.",
    "No task-board status, source module, test name, process exit, quiet queue, model statement, historical receipt, metric, or DuckLake projection is current-tree qualification evidence.",
    "No direct DuckDB access, Quack-to-file fallback, DuckLake scheduling authority, model-created authority, model-created policy permission, or model-created completion is authorized.",
    "Not qualified and quarantine recommended do not assert that a safety gate failed, that quarantine was applied, or that production state changed.",
    "Exactly-once network delivery is not claimed; only separately verified fenced idempotent authoritative effects could have an exactly-once scope.",
)


class QualificationReportError(ValueError):
    """The report or its repository binding is not exact and fail closed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise QualificationReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> Any:
    raise QualificationReportError(f"non-finite JSON number: {value}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualificationReportError("value is not canonical JSON") from exc


def _safe_relative_path(value: Any, name: str) -> str:
    text = _text(value, name, maximum=512)
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(
            part in {"", ".", ".."} or _PATH_SEGMENT_RE.fullmatch(part) is None
            for part in candidate.parts
        )
    ):
        raise QualificationReportError(f"{name} must be a closed repository-relative path")
    return text


def _text(value: Any, name: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum
        or "\x00" in value
    ):
        raise QualificationReportError(f"{name} must be bounded nonempty exact text")
    return value


def _oid(value: Any, name: str) -> str:
    text = _text(value, name, maximum=40)
    if _OID_RE.fullmatch(text) is None:
        raise QualificationReportError(f"{name} must be a lowercase Git object id")
    return text


def _sha(value: Any, name: str) -> str:
    text = _text(value, name, maximum=64)
    if _SHA_RE.fullmatch(text) is None:
        raise QualificationReportError(f"{name} must be lowercase SHA-256 hex")
    return text


def _content_ref(value: Any, name: str) -> str:
    text = _text(value, name, maximum=71)
    if _CONTENT_REF_RE.fullmatch(text) is None:
        raise QualificationReportError(f"{name} must be a sha256 content reference")
    return text


def _exact_dict(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise QualificationReportError(f"{name} has missing or unknown fields")
    return value


def _exact_list(value: Any, name: str, *, maximum: int = 256) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise QualificationReportError(f"{name} must be a bounded exact JSON array")
    return value


def _validate_json_limits(value: Any, *, depth: int = 0) -> int:
    if depth > _MAX_JSON_DEPTH:
        raise QualificationReportError("JSON nesting is too deep")
    if value is None or type(value) in {bool, int}:
        return 1
    if type(value) is float:
        raise QualificationReportError("floating-point values are not admitted")
    if type(value) is str:
        if len(value.encode("utf-8")) > _MAX_TEXT_BYTES or "\x00" in value:
            raise QualificationReportError("JSON text is unbounded or contains NUL")
        return 1
    if type(value) is list:
        total = 1 + sum(_validate_json_limits(item, depth=depth + 1) for item in value)
    elif type(value) is dict:
        total = 1
        for key, item in value.items():
            if type(key) is not str or _SECRET_KEY_RE.fullmatch(key):
                raise QualificationReportError("secret-shaped or non-text JSON key")
            total += _validate_json_limits(item, depth=depth + 1)
    else:
        raise QualificationReportError("non-JSON runtime type")
    if total > _MAX_JSON_ITEMS:
        raise QualificationReportError("JSON container count exceeds the bound")
    return total


def _read_regular_bytes(root: Path, relative_path: str, *, maximum: int) -> bytes:
    relative = _safe_relative_path(relative_path, "repository path")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(no_follow) is not int or type(directory) is not int:
        raise QualificationReportError("platform cannot enforce no-follow reads")
    common = no_follow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(root, os.O_RDONLY | directory | common)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | directory | common, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], os.O_RDONLY | common, dir_fd=descriptor)
    except OSError as exc:
        raise QualificationReportError(f"cannot open exact regular file: {relative}") from exc
    finally:
        os.close(descriptor)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise QualificationReportError(f"{relative} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(file_descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(payload) > maximum or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise QualificationReportError(f"{relative} changed during its bounded read")
    return payload


def _decode_json_bytes(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except QualificationReportError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationReportError(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise QualificationReportError(f"{name} must contain one JSON object")
    _validate_json_limits(value)
    return value


def _read_report(root: Path = ROOT) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_bytes(root, REPORT_RELATIVE_PATH, maximum=_MAX_REPORT_BYTES)
    return _decode_json_bytes(payload, "qualification report"), payload


def _binding(value: Any, name: str, *, schema: bool = False) -> dict[str, Any]:
    fields = {"path", "git_blob_oid", "raw_sha256"} | ({"schema"} if schema else set())
    item = _exact_dict(value, frozenset(fields), name)
    _safe_relative_path(item["path"], f"{name}.path")
    _oid(item["git_blob_oid"], f"{name}.git_blob_oid")
    _sha(item["raw_sha256"], f"{name}.raw_sha256")
    if schema:
        _text(item["schema"], f"{name}.schema", maximum=256)
    return item


def _validate_suite(value: Any) -> dict[str, Any]:
    suite_fields = frozenset(
        {
            "schema",
            "suite_id",
            "program_id",
            "root_goal_id",
            "tranche_goal_id",
            "goal_id",
            "component_snapshot",
            "matrix_binding",
            "components",
            "result_artifacts",
            "suite_status",
            "authoritative",
            "scheduling_authority",
            "qualification_authority",
            "completion_authority",
            "ducklake_authoritative",
            "promotion_eligible",
            "release_eligible",
            "blockers",
            "nonclaims",
        }
    )
    suite = _exact_dict(value, suite_fields, "benchmark suite manifest")
    if (
        suite["schema"] != "casf/benchmark-suite@1"
        or suite["suite_id"] != SUITE_ID
        or suite["program_id"] != "agent-supervisor-causal-event-federation-v1"
        or suite["root_goal_id"] != "CASF-G000"
        or suite["tranche_goal_id"] != "CASF-G040"
        or suite["goal_id"] != "CASF-G043"
    ):
        raise QualificationReportError("benchmark suite identity changed")
    snapshot = _exact_dict(
        suite["component_snapshot"], frozenset({"revision", "tree_id"}), "suite snapshot"
    )
    if snapshot != {"revision": SUITE_COMPONENT_REVISION, "tree_id": SUITE_COMPONENT_TREE}:
        raise QualificationReportError("benchmark component snapshot changed")
    _binding(suite["matrix_binding"], "matrix binding", schema=True)
    components = _exact_list(suite["components"], "benchmark components", maximum=4)
    if len(components) != 4 or tuple(item.get("task_id") for item in components) != BENCHMARK_TASKS:
        raise QualificationReportError("benchmark tasks must be unique, ordered, and complete")
    component_fields = frozenset(
        {
            "task_id",
            "benchmark_id",
            "manifest_state",
            "manifest_binding",
            "runner_binding",
            "result_schema",
            "availability",
            "execution_status",
            "reason_code",
            "result_ref",
            "metrics_omitted",
            "qualified",
            "promotion_eligible",
        }
    )
    for index, component_value in enumerate(components):
        component = _exact_dict(component_value, component_fields, f"component {index}")
        _text(component["benchmark_id"], f"component {index}.benchmark_id", maximum=128)
        _binding(component["manifest_binding"], f"component {index}.manifest", schema=True)
        _binding(component["runner_binding"], f"component {index}.runner")
        expected_state = "specification_only" if index == 0 else "capability_unavailable"
        if (
            component["task_id"] != BENCHMARK_TASKS[index]
            or component["manifest_state"] != expected_state
            or component["result_schema"] != BENCHMARK_SCHEMAS[index]
            or component["availability"] != "unavailable"
            or component["execution_status"] != "not_run"
            or component["reason_code"] != BENCHMARK_REASONS[index]
            or component["result_ref"] is not None
            or component["metrics_omitted"] is not True
            or component["qualified"] is not False
            or component["promotion_eligible"] is not False
        ):
            raise QualificationReportError(f"{BENCHMARK_TASKS[index]} was overstated")
    if suite["result_artifacts"] != [] or suite["suite_status"] != "not_run":
        raise QualificationReportError("benchmark results were invented")
    false_fields = (
        "authoritative",
        "scheduling_authority",
        "qualification_authority",
        "completion_authority",
        "ducklake_authoritative",
        "promotion_eligible",
        "release_eligible",
    )
    if any(suite[field] is not False for field in false_fields):
        raise QualificationReportError("benchmark suite created authority")
    if suite["blockers"] != [f"unavailable:casf_0{index}_live_not_run" for index in range(38, 42)]:
        raise QualificationReportError("benchmark blocker coverage changed")
    expected_suite_id = (
        "sha256:"
        + hashlib.sha256(
            _canonical_bytes({key: item for key, item in suite.items() if key != "suite_id"})
        ).hexdigest()
    )
    if suite["suite_id"] != expected_suite_id:
        raise QualificationReportError("benchmark suite content identity is invalid")
    for item in _exact_list(suite["nonclaims"], "suite nonclaims", maximum=8):
        _text(item, "suite nonclaim")
    return suite


def _validate_report(value: Any) -> dict[str, Any]:
    _validate_json_limits(value)
    report = _exact_dict(value, TOP_LEVEL_FIELDS, "qualification report")
    if (
        report["schema"] != "casf/qualification-report@1"
        or report["artifact_type"] != "casf_final_qualification_report"
        or type(report["artifact_version"]) is not int
        or report["artifact_version"] != 1
        or report["program_id"] != "agent-supervisor-causal-event-federation-v1"
        or report["repository_id"] != "endomorphosis/ipfs_accelerate_py"
        or report["root_goal_id"] != "CASF-G000"
        or report["tranche_goal_id"] != "CASF-G040"
        or report["goal_id"] != "CASF-G043"
        or report["task_id"] != "CASF-043"
        or report["report_kind"] != "repository_qualification_observation"
        or report["status"] != "not_qualified"
    ):
        raise QualificationReportError("qualification report identity or status changed")
    claimed_id = _content_ref(report["report_id"], "report_id")
    expected_id = (
        "sha256:"
        + hashlib.sha256(
            _canonical_bytes({key: item for key, item in report.items() if key != "report_id"})
        ).hexdigest()
    )
    if claimed_id != expected_id:
        raise QualificationReportError("report content identity is invalid")

    authority = _exact_dict(
        report["authority"],
        frozenset(
            {
                "authoritative",
                "qualification_authority",
                "completion_authority",
                "promotion_authority",
                "scheduling_authority",
                "release_authority",
            }
        ),
        "authority",
    )
    if any(item is not False for item in authority.values()):
        raise QualificationReportError("report must not create authority")
    baseline = _exact_dict(
        report["starting_baseline"],
        frozenset({"revision", "tree_id", "source_path"}),
        "starting baseline",
    )
    if baseline != {
        "revision": STARTING_REVISION,
        "tree_id": STARTING_TREE,
        "source_path": SOURCE_PATHS[0],
    }:
        raise QualificationReportError("starting baseline changed")
    source = _exact_dict(
        report["qualification_input_snapshot"],
        frozenset(
            {
                "revision",
                "tree_id",
                "snapshot_kind",
                "report_artifacts_present",
                "accepted_dependency_receipt_ref",
                "state_owner_attestation_ref",
            }
        ),
        "qualification input",
    )
    if source != {
        "revision": INPUT_REVISION,
        "tree_id": INPUT_TREE,
        "snapshot_kind": "pre_report_component_snapshot",
        "report_artifacts_present": False,
        "accepted_dependency_receipt_ref": None,
        "state_owner_attestation_ref": None,
    }:
        raise QualificationReportError("qualification input snapshot changed")
    final = _exact_dict(
        report["final_tree_binding"],
        frozenset(
            {
                "status",
                "revision",
                "tree_id",
                "control_plane_generation",
                "schema_fingerprint",
                "qualification_identity_ref",
                "validation_receipt_ref",
                "acceptance_receipt_ref",
                "final_result_identity",
            }
        ),
        "final tree binding",
    )
    if final["status"] != "pending_post_merge_state_owner_acceptance" or any(
        final[field] is not None for field in final if field != "status"
    ):
        raise QualificationReportError("pending final-tree evidence was invented")

    bindings = _exact_list(report["source_bindings"], "source bindings", maximum=len(SOURCE_PATHS))
    if len(bindings) != len(SOURCE_PATHS):
        raise QualificationReportError("source binding coverage is incomplete")
    for index, binding_value in enumerate(bindings):
        binding = _exact_dict(
            binding_value,
            frozenset({"path", "git_blob_oid", "raw_sha256", "byte_count"}),
            f"source binding {index}",
        )
        if binding["path"] != SOURCE_PATHS[index]:
            raise QualificationReportError("source bindings must be unique, ordered, and complete")
        _safe_relative_path(binding["path"], f"source binding {index}.path")
        _oid(binding["git_blob_oid"], f"source binding {index}.git_blob_oid")
        _sha(binding["raw_sha256"], f"source binding {index}.raw_sha256")
        if (
            type(binding["byte_count"]) is not int
            or not 0 < binding["byte_count"] <= _MAX_BOUND_BLOB_BYTES
        ):
            raise QualificationReportError("source binding byte count is invalid")

    control = _exact_dict(
        report["control_plane"],
        frozenset(
            {
                "status",
                "generation_id",
                "schema_fingerprint",
                "state_owner_receipt_ref",
                "reason_code",
            }
        ),
        "control plane",
    )
    if control != {
        "status": "not_observed",
        "generation_id": None,
        "schema_fingerprint": None,
        "state_owner_receipt_ref": None,
        "reason_code": "accepted_current_tree_state_owner_attestation_unavailable",
    }:
        raise QualificationReportError("control-plane evidence was overstated")
    capabilities = _exact_dict(
        report["capabilities"],
        frozenset({"duckdb_quack", "typed_quack_event_wait", "ducklake_quack", "live_scale"}),
        "capabilities",
    )
    capability_fields = frozenset(
        {
            "status",
            "evidence_ref",
            "blocks_core_qualification",
            "blocks_ducklake_promotion",
            "reason_code",
        }
    )
    expected_capabilities = {
        "duckdb_quack": (
            "not_qualified",
            True,
            True,
            "accepted_single_owner_typed_quack_receipt_unavailable",
        ),
        "typed_quack_event_wait": (
            "not_qualified",
            True,
            True,
            "remote_no_lost_wakeup_qualification_unavailable",
        ),
        "ducklake_quack": (
            "not_qualified",
            False,
            True,
            "accepted_non_authoritative_projection_receipts_unavailable",
        ),
        "live_scale": ("unavailable", True, True, "frozen_live_profiles_not_run"),
    }
    for name, expected in expected_capabilities.items():
        item = _exact_dict(capabilities[name], capability_fields, f"capability {name}")
        if (
            item["status"],
            item["blocks_core_qualification"],
            item["blocks_ducklake_promotion"],
            item["reason_code"],
        ) != expected or item["evidence_ref"] is not None:
            raise QualificationReportError(f"{name} capability was overstated")
    population = _exact_dict(
        report["population_and_concurrency"],
        frozenset(
            {
                "status",
                "federation_population",
                "supervisor_processes",
                "registered_logical_agents",
                "maximum_concurrent_subagents",
                "qualified_live_population",
                "contention_free_operation_qualified",
                "reason_code",
            }
        ),
        "population",
    )
    if (
        population["status"] != "not_observed"
        or any(
            population[field] is not None
            for field in (
                "federation_population",
                "supervisor_processes",
                "registered_logical_agents",
                "maximum_concurrent_subagents",
            )
        )
        or population["qualified_live_population"] is not False
        or population["contention_free_operation_qualified"] is not False
        or population["reason_code"]
        != "current_generation_fence_bound_population_attestation_unavailable"
    ):
        raise QualificationReportError("population or contention state was overstated")

    suite_wrapper = _exact_dict(
        report["benchmark_suite"],
        frozenset(
            {"path", "projection_binding", "relationship_to_qualification_input", "manifest"}
        ),
        "benchmark suite wrapper",
    )
    if suite_wrapper["path"] != SUITE_RELATIVE_PATH:
        raise QualificationReportError("benchmark suite path changed")
    projection = _exact_dict(
        suite_wrapper["projection_binding"],
        frozenset({"revision", "tree_id", "git_blob_oid", "raw_sha256"}),
        "suite projection binding",
    )
    if projection != {
        "revision": SUITE_PROJECTION_REVISION,
        "tree_id": SUITE_PROJECTION_TREE,
        "git_blob_oid": SUITE_BLOB,
        "raw_sha256": SUITE_RAW_SHA256,
    }:
        raise QualificationReportError("suite projection binding changed")
    relationship = _exact_dict(
        suite_wrapper["relationship_to_qualification_input"],
        frozenset(
            {
                "component_snapshot_is_ancestor",
                "suite_projection_is_ancestor",
                "all_bound_inputs_exact_at_qualification_input",
            }
        ),
        "suite relationship",
    )
    if any(item is not True for item in relationship.values()):
        raise QualificationReportError("suite ancestry or byte equality is not established")
    _validate_suite(suite_wrapper["manifest"])

    disposition = _exact_dict(
        report["evidence_disposition"],
        frozenset({"verified", "failed", "skipped", "not_run", "missing_or_unaccepted"}),
        "evidence disposition",
    )
    expected_verified = [
        {
            "evidence_id": "qualification_input_source_bindings",
            "scope": "repository_objects_at_qualification_input",
        },
        {
            "evidence_id": "benchmark_suite_specification_bindings",
            "scope": "component_and_projection_ancestry_plus_exact_blobs",
        },
    ]
    expected_skipped = [
        {
            "evidence_id": "live_runtime_and_database_observation",
            "reason_code": "repository_only_report_has_no_runtime_authority",
        },
        {"evidence_id": "ducklake_live_profile", "reason_code": "optional_profile_not_admitted"},
    ]
    expected_missing = [
        "casf/qualification-identity@1",
        "casf/promotion-evidence-bundle@1",
        "casf/promotion-decision@1",
        "casf/promotion-decision-validation@1",
        "casf/federation-completion-receipt@1",
        "casf/fixed-point@1 accepted-producer provenance",
        "casf/drift-report@1 accepted-producer provenance",
        "casf/control-parity@1 decoder",
        "casf/formal-model-report@1 decoder",
    ]
    if (
        disposition["verified"] != expected_verified
        or disposition["failed"] != []
        or disposition["skipped"] != expected_skipped
        or disposition["not_run"] != list(BENCHMARK_SCHEMAS)
        or disposition["missing_or_unaccepted"] != expected_missing
    ):
        raise QualificationReportError("evidence disposition is incomplete or inaccurate")
    coverage = _exact_dict(report["result_coverage"], RESULT_FIELDS, "result coverage")
    if any(item != "not_qualified" for item in coverage.values()):
        raise QualificationReportError("result coverage was overstated")
    safety = _exact_dict(report["safety_gates"], SAFETY_FIELDS, "safety gates")
    if any(item != "unverified" for item in safety.values()):
        raise QualificationReportError("safety gate evidence was invented")
    claims = _exact_dict(report["claims"], CLAIM_FIELDS, "claims")
    if any(item is not False for item in claims.values()):
        raise QualificationReportError("unsupported qualification claim")

    qualification = _exact_dict(
        report["qualification"],
        frozenset(
            {
                "status",
                "profiles",
                "promotion_eligible",
                "release_eligible",
                "disposition",
                "disposition_authoritative",
                "quarantine_applied",
                "promotion_applied",
                "rollback_applied",
                "production_state_changed",
                "authority_created",
                "completion_created",
                "upstream_reverification_required",
                "promotion_decision_ref",
                "reason_codes",
            }
        ),
        "qualification",
    )
    false_fields = (
        "promotion_eligible",
        "release_eligible",
        "disposition_authoritative",
        "quarantine_applied",
        "promotion_applied",
        "rollback_applied",
        "production_state_changed",
        "authority_created",
        "completion_created",
    )
    if (
        qualification["status"] != "not_qualified"
        or qualification["disposition"] != "quarantine_recommended"
        or any(qualification[field] is not False for field in false_fields)
        or qualification["upstream_reverification_required"] is not True
        or qualification["promotion_decision_ref"] is not None
        or qualification["reason_codes"] != list(CORE_REASON_CODES)
    ):
        raise QualificationReportError("qualification disposition was overstated")
    profiles = _exact_dict(
        qualification["profiles"], frozenset({"duckdb_quack", "ducklake_quack"}), "profiles"
    )
    if profiles != {
        "duckdb_quack": {
            "status": "not_qualified",
            "single_state_owner_required": True,
            "contention_free_operation_qualified": False,
            "decision_ref": None,
        },
        "ducklake_quack": {
            "status": "not_qualified",
            "authoritative": False,
            "scheduling_authority": False,
            "blocks_core_qualification": False,
            "decision_ref": None,
        },
    }:
        raise QualificationReportError("qualification profiles changed")

    gaps = _exact_list(report["residual_gaps"], "residual gaps", maximum=len(GAP_IDS))
    if len(gaps) != len(GAP_IDS) or tuple(item.get("gap_id") for item in gaps) != GAP_IDS:
        raise QualificationReportError("residual gaps must be unique, ordered, and complete")
    all_gap_reasons: set[str] = set()
    for index, gap_value in enumerate(gaps):
        gap = _exact_dict(
            gap_value,
            frozenset({"gap_id", "status", "scope", "reason_codes", "required_resolution"}),
            f"residual gap {index}",
        )
        if gap["status"] not in {"blocking_core", "blocking_ducklake_profile", "blocking_release"}:
            raise QualificationReportError("residual gap status is unknown")
        _text(gap["scope"], f"gap {index}.scope")
        _text(gap["required_resolution"], f"gap {index}.required_resolution")
        reasons = _exact_list(gap["reason_codes"], f"gap {index}.reason_codes", maximum=8)
        if not reasons or len(set(reasons)) != len(reasons):
            raise QualificationReportError("gap reason codes must be nonempty and unique")
        for reason in reasons:
            all_gap_reasons.add(_text(reason, f"gap {index}.reason_code", maximum=256))
    if not set(CORE_REASON_CODES).issubset(all_gap_reasons) or not PROMOTION_REASON_CODES.issubset(
        all_gap_reasons
    ):
        raise QualificationReportError("promotion blockers are missing from residual gaps")

    rollback = _exact_dict(
        report["rollback"],
        frozenset(
            {
                "status",
                "scope",
                "target",
                "applied",
                "executable",
                "decision_ref",
                "required_authority",
                "history_rewrite_permitted",
            }
        ),
        "rollback",
    )
    if (
        rollback["status"] != "not_authorized"
        or rollback["scope"]
        != [REPORT_RELATIVE_PATH, MARKDOWN_RELATIVE_PATH, __file__.split(str(ROOT) + os.sep, 1)[1]]
        or rollback["target"] != {"revision": INPUT_REVISION, "tree_id": INPUT_TREE}
        or rollback["applied"] is not False
        or rollback["executable"] is not False
        or rollback["decision_ref"] is not None
        or rollback["history_rewrite_permitted"] is not False
    ):
        raise QualificationReportError(
            "rollback boundary is not the inert owned-artifact predecessor"
        )
    _text(rollback["required_authority"], "rollback.required_authority")
    provenance = _exact_dict(
        report["provenance"],
        frozenset(
            {
                "evidence_source",
                "source_binding_method",
                "qualification_database_queried",
                "qualification_runtime_contacted",
                "qualification_network_used",
                "qualification_provider_output_admitted",
                "model_output_admitted_as_evidence",
                "task_board_status_admitted_as_evidence",
                "ducklake_projection_admitted_as_authority",
            }
        ),
        "provenance",
    )
    if (
        provenance["evidence_source"] != "repository_only"
        or provenance["source_binding_method"] != "git_blob_oid_plus_raw_sha256"
        or any(
            item is not False
            for key, item in provenance.items()
            if key not in {"evidence_source", "source_binding_method"}
        )
    ):
        raise QualificationReportError("report provenance was overstated")
    if tuple(report["nonclaims"]) != NONCLAIMS or any(
        type(item) is not str for item in report["nonclaims"]
    ):
        raise QualificationReportError("nonclaims must be exact and complete")
    return report


def _render_markdown(report: dict[str, Any], report_bytes: bytes) -> str:
    suite = report["benchmark_suite"]["manifest"]
    source_rows = "\n".join(
        f"| `{item['path']}` | `{item['git_blob_oid']}` | `{item['raw_sha256']}` |"
        for item in report["source_bindings"]
    )
    benchmark_rows = "\n".join(
        f"| {item['task_id']} | `{item['result_schema']}` | {item['availability']} | "
        f"{item['execution_status']} | {str(item['metrics_omitted']).lower()} | {item['result_ref']} |"
        for item in suite["components"]
    ).replace(" | None |", " | null |")
    gap_rows = "\n".join(
        f"{index}. **{gap['gap_id']}** ({gap['status']}): {gap['required_resolution']} "
        f"Reason codes: `{', '.join(gap['reason_codes'])}`."
        for index, gap in enumerate(report["residual_gaps"], start=1)
    )
    digest = hashlib.sha256(report_bytes).hexdigest()
    return (
        f"""# CASF final qualification and residual-gap report

Machine report: `final_qualification_report.json`
Schema: `casf/qualification-report@1`
Report ID: `{report["report_id"]}`
Machine-report raw SHA-256: `{digest}`

## Disposition

**Not qualified — quarantine recommended, not applied.** This repository-only
report is non-authoritative. It is not a qualification identity, completion
receipt, accepted gate decision, release authority, or production-state change.

The DuckDB + Quack core profile and the DuckLake + Quack profile are both
`not_qualified`. DuckLake remains non-authoritative and does not block the core
profile, but its missing receipts do block DuckLake promotion. Contention-free
operation is not qualified because no current-generation, fence-bound,
single-state-owner population attestation was admitted.

## Identity boundary

| Boundary | Revision | Tree | Status |
|---|---|---|---|
| Sealed program baseline | `{STARTING_REVISION}` | `{STARTING_TREE}` | historical baseline |
| Qualification input | `{INPUT_REVISION}` | `{INPUT_TREE}` | pre-report component snapshot |
| Final merged tree | `null` | `null` | pending post-merge state-owner acceptance |

The input snapshot is an ancestor, not a claim about the report commit or the
eventual landing tree. The final revision, tree, generation, schema fingerprint,
qualification identity, validation receipt, acceptance receipt, and result
identity all remain null until accepted producers and an independent verifier
create them.

## Exact repository source bindings

| Path | Git blob | Raw SHA-256 |
|---|---|---|
{source_rows}

These bindings describe repository objects at the qualification input. They
prove byte identity and lineage only; they are not live or execution evidence.

## Benchmark suite

Suite ID: `{suite["suite_id"]}`

Suite projection: `{SUITE_PROJECTION_REVISION}` / `{SUITE_PROJECTION_TREE}`

Component snapshot: `{SUITE_COMPONENT_REVISION}` / `{SUITE_COMPONENT_TREE}`

Suite manifest blob/raw SHA-256: `{SUITE_BLOB}` / `{SUITE_RAW_SHA256}`

Both suite snapshots are ancestors of the qualification input, and every bound
matrix, manifest, and runner blob is byte-identical there. This establishes a
current specification lineage, not benchmark execution or qualification.

| Task | Result schema | Availability | Execution | Metrics omitted | Result ref |
|---|---|---|---|---|---|
{benchmark_rows}

The suite has no result artifacts, no metrics, and no scheduling,
qualification, completion, promotion, release, or DuckLake authority.

## Evidence disposition

- Verified: qualification-input repository bindings and benchmark specification bindings.
- Failed: none recorded; unexecuted gates are not reported as failures.
- Skipped: live runtime/database observation and the optional DuckLake live profile.
- Not run: idle, parallel, load, and token benchmark result schemas.
- Missing or unaccepted: current qualification identity, evidence bundle,
  promotion decision and validation, completion receipt, accepted CASF-030 and
  CASF-033 producer provenance, and CASF-035/CASF-036 decoders.

Every product result area is `not_qualified`, and every non-compensable safety
gate is `unverified` for the qualification input and pending final-tree
identity. This is evidence absence, not evidence that a gate failed.

## Residual gaps

{gap_rows}

## Rollback boundary

Rollback is not authorized, executable, or applied. Its scope is only the
three CASF-043 owned artifacts, and its target is the pre-report input
`{INPUT_REVISION}` / `{INPUT_TREE}`. A verified, current-fence typed state-owner
decision is required; history rewriting is prohibited.

## Nonclaims

"""
        + "\n".join(f"- {item}" for item in NONCLAIMS)
        + "\n"
    )


def _trusted_git_assets() -> tuple[str, str]:
    try:
        executable = _TRUSTED_GIT.lstat()
        exec_path = _TRUSTED_GIT_EXEC_PATH.lstat()
    except OSError as exc:
        raise QualificationReportError("trusted Git installation is unavailable") from exc
    if (
        not stat.S_ISREG(executable.st_mode)
        or executable.st_uid != 0
        or executable.st_mode & 0o022
        or not executable.st_mode & 0o111
    ):
        raise QualificationReportError("trusted Git executable is not protected")
    if not stat.S_ISDIR(exec_path.st_mode) or exec_path.st_uid != 0 or exec_path.st_mode & 0o022:
        raise QualificationReportError("trusted Git exec path is not protected")
    return str(_TRUSTED_GIT), str(_TRUSTED_GIT_EXEC_PATH)


def _git_environment(exec_path: str) -> dict[str, str]:
    return {
        "HOME": "/",
        "PATH": _TRUSTED_PROCESS_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_EXEC_PATH": exec_path,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_GRAFT_FILE": os.devnull,
    }


def _git_result(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    executable, exec_path = _trusted_git_assets()
    try:
        return subprocess.run(
            [
                executable,
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(root),
                *args,
            ],
            check=check,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=20,
            env=_git_environment(exec_path),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise QualificationReportError("trusted Git repository observation failed") from exc


def _git_bytes(root: Path, *args: str) -> bytes:
    return _git_result(root, *args).stdout


def _git_text(root: Path, *args: str) -> str:
    try:
        return _git_bytes(root, *args).decode("utf-8", errors="strict").rstrip("\n")
    except UnicodeError as exc:
        raise QualificationReportError("Git returned non-UTF-8 identity output") from exc


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = _git_result(root, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode != 0:
        raise QualificationReportError(f"{ancestor} is not an ancestor of {descendant}")


def _blob_at(root: Path, revision: str, relative_path: str) -> tuple[str, bytes]:
    _oid(revision, "revision")
    path = _safe_relative_path(relative_path, "bound path")
    raw = _git_bytes(root, "ls-tree", "-z", revision, "--", path)
    if not raw.endswith(b"\x00") or raw.count(b"\x00") != 1:
        raise QualificationReportError(f"{path} is not exactly one tracked object")
    try:
        metadata, observed_path = raw[:-1].split(b"\t", 1)
        mode, kind, oid_bytes = metadata.split(b" ", 2)
        observed = observed_path.decode("utf-8", errors="strict")
        oid = oid_bytes.decode("ascii", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise QualificationReportError("malformed Git tree response") from exc
    if mode != b"100644" or kind != b"blob" or observed != path or _OID_RE.fullmatch(oid) is None:
        raise QualificationReportError(f"{path} is not an exact regular tracked blob")
    payload = _git_bytes(root, "cat-file", "blob", oid)
    if len(payload) > _MAX_BOUND_BLOB_BYTES:
        raise QualificationReportError(f"{path} exceeds its bound")
    return oid, payload


def _verify_binding_at(root: Path, revision: str, binding: dict[str, Any]) -> bytes:
    oid, payload = _blob_at(root, revision, binding["path"])
    if (
        oid != binding["git_blob_oid"]
        or hashlib.sha256(payload).hexdigest() != binding["raw_sha256"]
        or ("byte_count" in binding and len(payload) != binding["byte_count"])
    ):
        raise QualificationReportError(f"source binding differs: {binding['path']}")
    return payload


def _verify_worktree_matches_head(root: Path, relative_path: str, maximum: int) -> bytes:
    working = _read_regular_bytes(root, relative_path, maximum=maximum)
    head = _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{commit}"), "HEAD")
    _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{tree}"), "HEAD tree")
    _oid_at_head, committed = _blob_at(root, head, relative_path)
    if working != committed:
        raise QualificationReportError(f"working bytes differ from tracked HEAD: {relative_path}")
    return working


def _suite_bindings(suite: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    result = [suite["matrix_binding"]]
    for component in suite["components"]:
        result.extend((component["manifest_binding"], component["runner_binding"]))
    return tuple(result)


def _validate_repository(root: Path = ROOT) -> dict[str, Any]:
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root.resolve(strict=True):
        raise QualificationReportError("Git top level differs from the report repository")
    first_head = _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{commit}"), "HEAD")
    first_tree = _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{tree}"), "HEAD tree")
    if _git_text(root, "rev-parse", "--verify", f"{STARTING_REVISION}^{{tree}}") != STARTING_TREE:
        raise QualificationReportError("starting tree binding changed")
    if _git_text(root, "rev-parse", "--verify", f"{INPUT_REVISION}^{{tree}}") != INPUT_TREE:
        raise QualificationReportError("qualification input tree binding changed")
    _require_ancestor(root, STARTING_REVISION, INPUT_REVISION)
    _require_ancestor(root, INPUT_REVISION, first_head)

    report_bytes = _verify_worktree_matches_head(root, REPORT_RELATIVE_PATH, _MAX_REPORT_BYTES)
    markdown_bytes = _verify_worktree_matches_head(
        root, MARKDOWN_RELATIVE_PATH, _MAX_MARKDOWN_BYTES
    )
    report = _validate_report(_decode_json_bytes(report_bytes, "qualification report"))
    for binding in report["source_bindings"]:
        _verify_binding_at(root, INPUT_REVISION, binding)

    wrapper = report["benchmark_suite"]
    suite = wrapper["manifest"]
    if (
        _git_text(root, "rev-parse", "--verify", f"{SUITE_COMPONENT_REVISION}^{{tree}}")
        != SUITE_COMPONENT_TREE
    ):
        raise QualificationReportError("suite component tree changed")
    if (
        _git_text(root, "rev-parse", "--verify", f"{SUITE_PROJECTION_REVISION}^{{tree}}")
        != SUITE_PROJECTION_TREE
    ):
        raise QualificationReportError("suite projection tree changed")
    _require_ancestor(root, SUITE_COMPONENT_REVISION, INPUT_REVISION)
    _require_ancestor(root, SUITE_PROJECTION_REVISION, INPUT_REVISION)
    suite_oid, suite_projection_bytes = _blob_at(
        root, SUITE_PROJECTION_REVISION, SUITE_RELATIVE_PATH
    )
    if (
        suite_oid != SUITE_BLOB
        or hashlib.sha256(suite_projection_bytes).hexdigest() != SUITE_RAW_SHA256
    ):
        raise QualificationReportError("suite projection object changed")
    current_suite_bytes = _verify_worktree_matches_head(
        root, SUITE_RELATIVE_PATH, _MAX_REPORT_BYTES
    )
    current_suite = _decode_json_bytes(current_suite_bytes, "benchmark suite")
    if current_suite != suite or suite_projection_bytes != current_suite_bytes:
        raise QualificationReportError("embedded, projected, and current suite bytes differ")
    for binding in _suite_bindings(suite):
        component_bytes = _verify_binding_at(root, SUITE_COMPONENT_REVISION, binding)
        input_bytes = _verify_binding_at(root, INPUT_REVISION, binding)
        current_oid, current_bytes = _blob_at(root, first_head, binding["path"])
        if (
            current_oid != binding["git_blob_oid"]
            or component_bytes != input_bytes
            or input_bytes != current_bytes
        ):
            raise QualificationReportError(f"suite input drifted: {binding['path']}")

    promotion_binding = next(
        item
        for item in report["source_bindings"]
        if item["path"] == "ipfs_accelerate_py/agent_supervisor/federation/promotion.py"
    )
    promotion_source = _verify_binding_at(root, INPUT_REVISION, promotion_binding)
    if any(reason.encode("utf-8") not in promotion_source for reason in PROMOTION_REASON_CODES):
        raise QualificationReportError(
            "reported residual blockers differ from the bound promotion gate"
        )
    if markdown_bytes.decode("utf-8", errors="strict") != _render_markdown(report, report_bytes):
        raise QualificationReportError("Markdown is not the exact deterministic report projection")

    second_head = _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{commit}"), "HEAD")
    second_tree = _oid(_git_text(root, "rev-parse", "--verify", "HEAD^{tree}"), "HEAD tree")
    if (first_head, first_tree) != (second_head, second_tree):
        raise QualificationReportError("repository identity changed during validation")
    return report


def test_report_schema_identity_and_truthful_disposition() -> None:
    report, _payload = _read_report()
    assert _validate_report(report) == report


def test_exact_repository_suite_and_markdown_bindings() -> None:
    report = _validate_repository()
    assert report["status"] == "not_qualified"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"artifact_version": True}),
        lambda value: value.update({"status": "qualified"}),
        lambda value: value["claims"].update({"production_ready": True}),
        lambda value: value["authority"].update({"completion_authority": True}),
        lambda value: value["final_tree_binding"].update({"revision": "f" * 40}),
        lambda value: value["control_plane"].update({"generation_id": 7}),
        lambda value: value["population_and_concurrency"].update({"supervisor_processes": 12}),
        lambda value: value["benchmark_suite"]["manifest"]["components"].__setitem__(
            3, deepcopy(value["benchmark_suite"]["manifest"]["components"][0])
        ),
        lambda value: value["benchmark_suite"]["manifest"]["components"][0].update(
            {"metrics_omitted": False}
        ),
        lambda value: value["benchmark_suite"]["manifest"].update(
            {"result_artifacts": ["sha256:" + "1" * 64]}
        ),
        lambda value: value["qualification"].update({"quarantine_applied": True}),
        lambda value: value["qualification"]["profiles"]["ducklake_quack"].update(
            {"authoritative": True}
        ),
        lambda value: value["rollback"].update(
            {"target": {"revision": STARTING_REVISION, "tree_id": STARTING_TREE}}
        ),
        lambda value: value["residual_gaps"].pop(2),
        lambda value: value["nonclaims"].__setitem__(0, "Production ready."),
        lambda value: value.update({"api_key": "forbidden"}),
    ],
)
def test_mutations_cannot_create_authority_or_hide_required_gaps(mutate: Any) -> None:
    report, _payload = _read_report()
    mutate(report)
    with pytest.raises(QualificationReportError):
        _validate_report(report)


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"schema":"one","schema":"two"}', "duplicate JSON key"),
        (b'{"value":NaN}', "non-finite JSON number"),
        (b'{"value":Infinity}', "non-finite JSON number"),
        (b'{"value":1.5}', "floating-point values"),
        (b'{"value":{"api_key":"secret"}}', "secret-shaped"),
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_float_and_secret_shapes(
    payload: bytes, match: str
) -> None:
    with pytest.raises(QualificationReportError, match=match):
        value = _decode_json_bytes(payload, "adversarial fixture")
        _validate_json_limits(value)


def test_bounded_reader_rejects_symlinks_and_oversized_inputs(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "target.json"
    target.write_text("{}", encoding="utf-8")
    (root / "linked.json").symlink_to(target)
    with pytest.raises(QualificationReportError, match="cannot open exact regular file"):
        _read_regular_bytes(root, "linked.json", maximum=100)
    (root / "large.json").write_bytes(b"x" * 101)
    with pytest.raises(QualificationReportError, match="bounded regular file"):
        _read_regular_bytes(root, "large.json", maximum=100)


def test_git_ignores_path_loader_config_replace_and_alternate_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-ran"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    poison = {
        "PATH": str(fake_bin),
        "GIT_EXEC_PATH": str(fake_bin),
        "GIT_DIR": str(tmp_path / "evil.git"),
        "GIT_WORK_TREE": str(tmp_path),
        "GIT_INDEX_FILE": str(tmp_path / "evil-index"),
        "GIT_OBJECT_DIRECTORY": str(tmp_path / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(tmp_path / "objects"),
        "GIT_REPLACE_REF_BASE": "refs/evil/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.bare",
        "GIT_CONFIG_VALUE_0": "true",
        "LD_PRELOAD": str(tmp_path / "loader.so"),
        "BASH_ENV": str(fake_git),
    }
    for key, item in poison.items():
        monkeypatch.setenv(key, item)
    assert _OID_RE.fullmatch(_git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}"))
    assert not marker.exists()
    environment = _git_environment(str(_TRUSTED_GIT_EXEC_PATH))
    assert environment["PATH"] == _TRUSTED_PROCESS_PATH
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not (set(poison) - {"PATH", "GIT_EXEC_PATH"}) & set(environment)


@pytest.mark.parametrize("kind", ["missing", "directory", "file", "symlink"])
def test_unprotected_git_executable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    candidate = tmp_path / kind
    if kind == "directory":
        candidate.mkdir()
    elif kind == "file":
        candidate.write_bytes(b"not executable")
        candidate.chmod(0o644)
    elif kind == "symlink":
        candidate.symlink_to("/usr/bin/git")
    monkeypatch.setattr(sys.modules[__name__], "_TRUSTED_GIT", candidate)
    with pytest.raises(QualificationReportError, match="trusted Git"):
        _trusted_git_assets()


def _run_fixture_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env={"HOME": "/", "PATH": _TRUSTED_PROCESS_PATH, "LANG": "C", "LC_ALL": "C"},
    ).stdout.strip()


def _minimal_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _run_fixture_git(root, "init", "-q")
    (root / "bound.txt").write_text("one\n", encoding="utf-8")
    _run_fixture_git(root, "add", "bound.txt")
    _run_fixture_git(
        root,
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf@example.invalid",
        "commit",
        "-q",
        "-m",
        "one",
    )
    return root


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_index_flags_cannot_substitute_working_bytes(tmp_path: Path, flag: str) -> None:
    root = _minimal_repository(tmp_path)
    _run_fixture_git(root, "update-index", flag, "bound.txt")
    (root / "bound.txt").write_text("tampered\n", encoding="utf-8")
    assert _run_fixture_git(root, "status", "--porcelain=v1") == ""
    with pytest.raises(QualificationReportError, match="working bytes differ"):
        _verify_worktree_matches_head(root, "bound.txt", 100)


def test_internal_replace_ref_cannot_redirect_bound_tree(tmp_path: Path) -> None:
    root = _minimal_repository(tmp_path)
    original_commit = _run_fixture_git(root, "rev-parse", "HEAD")
    original_tree = _run_fixture_git(root, "rev-parse", "HEAD^{tree}")
    (root / "bound.txt").write_text("two\n", encoding="utf-8")
    _run_fixture_git(root, "add", "bound.txt")
    _run_fixture_git(
        root,
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf@example.invalid",
        "commit",
        "-q",
        "-m",
        "two",
    )
    replacement = _run_fixture_git(root, "rev-parse", "HEAD")
    _run_fixture_git(root, "replace", original_commit, replacement)
    _run_fixture_git(root, "checkout", "-q", original_commit)
    assert _git_text(root, "rev-parse", "HEAD^{tree}") == original_tree


def test_two_pass_repository_identity_detects_toctou(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _git_text
    head_reads = 0

    def racing_git(root: Path, *args: str) -> str:
        nonlocal head_reads
        value = original(root, *args)
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            head_reads += 1
            if head_reads >= 4:
                return "f" * 40
        return value

    monkeypatch.setattr(sys.modules[__name__], "_git_text", racing_git)
    with pytest.raises(QualificationReportError, match="changed during validation"):
        _validate_repository(ROOT)


def test_markdown_rejects_appended_positive_claims() -> None:
    report, payload = _read_report()
    markdown = _render_markdown(_validate_report(report), payload)
    assert "production ready" not in markdown.lower()
    assert markdown + "\nProduction ready: true.\n" != markdown
