"""Sealed trusted pre-validation action for state-laws live cohorts.

This module is the accelerator-owned authority boundary for SLR-009..SLR-021.
It never accepts a provider-authored command.  The only admissible action is a
fixed invocation of ``run_state_laws_reindex_cohort.py --execute-live`` for the
cohort statically assigned to the canonical task.

The adjacent sandbox runs the candidate with ``--network none`` and exposes a
loopback proxy backed by a read-only Unix socket.  A separate pinned proxy
container owns DNS and ordinary bridge egress and admits only exact official
catalog hosts.  The outer receipt joins the sandbox, image, namespace, audit,
source projection, and live-manifest identities before it can complete.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:  # pragma: no cover - exercised by the sealed wrapper
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import terminate_pid_tree
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import atomic_write_json
else:
    from ..todo_daemon.core import terminate_pid_tree
    from ..todo_daemon.engine import atomic_write_json


TRUSTED_PREVALIDATION_ACTION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/trusted-prevalidation-action-policy@1"
)
TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/trusted-prevalidation-action-receipt@1"
)
SANDBOX_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-laws-live-cohort-sandbox-policy@1"
)
SANDBOX_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-laws-live-cohort-sandbox-receipt@1"
)
SANDBOX_ALLOWLIST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-laws-official-egress-allowlist@1"
)
HANDLER = "state_laws_live_cohort_v1"
PHASE = "accepted_proposal_before_hermetic_validation"
RUNNER_PATH = "scripts/ops/legal_data/run_state_laws_reindex_cohort.py"
WRAPPER_PATH = (
    "ipfs_accelerate_py/agent_supervisor/trusted_actions/state_laws_live_cohort_v1.py"
)
SANDBOX_POLICY_PATH = (
    "ipfs_accelerate_py/agent_supervisor/trusted_actions/"
    "state_laws_live_cohort_v1.policy.json"
)
EVIDENCE_ENVIRONMENT_VARIABLE = "STATE_LAWS_REINDEX_EVIDENCE_ROOT"
RECEIPT_ENVIRONMENT_VARIABLE = "STATE_LAWS_TRUSTED_ACTION_RECEIPT"
RECEIPT_SHA256_ENVIRONMENT_VARIABLE = "STATE_LAWS_TRUSTED_ACTION_RECEIPT_SHA256"
SOURCE_PROJECTION_MODE = "cohort-scoped-explicit-source-projection@1"
PRE_ACTION_VALIDATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-laws-pre-action-validation@1"
)
PRE_ACTION_VALIDATION_HANDLER = "state_laws_cohort_pytest_v1"
TRUSTED_STATIC_PRE_ACTION_GATE = "state_laws_adapter_evidence_contract_v1"
ADAPTER_EVIDENCE_OWNERSHIP = "cohort-adapter-native-stage-associate-frontier-replay@1"
OFFICIAL_SOURCE_CATALOG_PATH = "data/legal/state_laws/official_source_catalog.json"
SOURCE_CLOSURE_POLICY_REGISTRY_PATH = (
    "data/legal/state_laws/source_closure_policy_registry.json"
)
SOURCE_CLOSURE_POLICY_REGISTRY_SCHEMA = "state-laws-source-closure-policy-registry-v1"
SANDBOX_MODE = "sealed_official_egress_v1"
SANDBOX_CAPABILITY = "official_domain_egress_allowlist_v1"
CURRENT_RUN_SCHEMA = "state-laws-reindex-current-run-v1"
LIVE_RUN_SCHEMA = "state-laws-reindex-live-run-v1"
LIVE_MODE = "live_uncapped"
LIVE_CORPUS_ROOT = "corpus"
SOURCE_UNIT_SCHEMA = "state-laws-source-unit-v1"
SOURCE_UNIT_PROVENANCE_SCHEMA = "state-laws-source-unit-provenance-v1"
MAX_PARALLEL = 2
EXIT_SANDBOX_UNAVAILABLE = 78
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SEALED_GIT_PATH = Path("/usr/bin/git")
SEALED_GIT_SHA256 = "aa6540695d076182256dd6e96c8b302e4d56381e3000bbfd5c71bbdfe94a4942"

STATE_LAWS_TASK_COHORTS: Mapping[str, str] = {
    f"SLR-{number:03d}": chr(ord("A") + number - 9) for number in range(9, 22)
}

COHORT_JURISDICTIONS: Mapping[str, tuple[str, ...]] = {
    "A": ("AL", "AK", "AZ", "AR"),
    "B": ("CA", "CO", "CT", "DE"),
    "C": ("FL", "GA", "HI", "ID"),
    "D": ("IL", "IN", "IA", "KS"),
    "E": ("KY", "LA", "ME", "MD"),
    "F": ("MA", "MI", "MN", "MS"),
    "G": ("MO", "MT", "NE", "NV"),
    "H": ("NH", "NJ", "NM", "NY"),
    "I": ("NC", "ND", "OH", "OK"),
    "J": ("OR", "PA", "RI", "SC"),
    "K": ("SD", "TN", "TX", "UT"),
    "L": ("VT", "VA", "WA", "WV"),
    "M": ("WI", "WY", "DC"),
}

SCRUBBED_ENVIRONMENT_NAMES = (
    "STATE_SCRAPER_MAX_STATUTES",
    "STATE_SCRAPER_BOUNDED_DIRECT_ONLY",
    "STATE_SCRAPER_GLOBAL_BOUNDED_ENV",
    "STATE_SCRAPER_CODE_TIMEOUT_SECONDS",
    "STATE_SCRAPER_FETCH_TIMEOUT_SECONDS",
    "STATE_SCRAPER_HARD_TIMEOUT_SECONDS",
)

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "handler",
        "phase",
        "required_for_task_completion",
        "receipt_schema",
        "task_cohorts",
        "runner_path",
        "evidence_root_environment_variable",
        "validation_receipt_environment_variable",
        "validation_receipt_sha256_environment_variable",
        "evidence_root",
        "receipt_root",
        "max_parallel",
        "action_timeout_seconds",
        "progress_timeout_seconds",
        "trusted_static_pre_action_gate",
        "adapter_evidence_ownership",
        "adapter_contract_catalog_binding_required",
        "adapter_contract_closed_mode_schema_required",
        "adapter_contract_reviewed_policy_digest_required",
        "source_closure_policy_registry_path",
        "source_closure_policy_registry_schema",
        "adapter_evidence_api_signature_and_dataflow_required",
        "source_projection",
        "pre_action_validation",
        "receipt_bindings",
        "retry",
        "sandbox",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "mode",
        "shared_paths",
        "cohort_paths",
        "unrelated_cohort_paths_invalidate",
        "shared_path_changes_invalidate",
        "path_sha256_map_required",
        "digest_required",
    }
)
_SANDBOX_FIELDS = frozenset(
    {
        "mode",
        "required_capability",
        "wrapper_path",
        "wrapper_sha256",
        "policy_path",
        "policy_sha256",
        "unavailable_is_failure",
    }
)
_RETRY_FIELDS = frozenset({"max_attempts", "resume_exact_fingerprint_only"})
_PRE_ACTION_VALIDATION_FIELDS = frozenset(
    {"schema", "handler", "network", "proposal_bound"}
)
_REQUIRED_RECEIPT_BINDINGS = (
    "canonical_task_cid",
    "proposal_id",
    "proposal_receipt_id",
    "diff_digest",
    "baseline_id",
    "repository_tree_id",
    "proposal_action_diff_digest",
    "candidate_source_fingerprint",
    "cohort",
    "policy_digest",
    "argv_digest",
    "wrapper_argv_digest",
    "source_projection_digest",
    "baseline_commit",
    "baseline_regular_files_digest",
    "baseline_regular_file_count",
    "baseline_regular_file_bytes",
    "baseline_excluded_nonregular_count",
    "baseline_excluded_nonregular_digest",
    "candidate_workspace_snapshot_digest",
    "git_client_sha256",
    "cleanup_service",
    "cleanup",
    "cleanup_claim_scope",
    "startup_sweep_removed_container_ids_sha256",
    "startup_sweep_removed_lease_ids_sha256",
    "sandbox_policy_sha256",
    "sandbox_receipt_path",
    "sandbox_receipt_sha256",
    "runner_image_id",
    "preflight_runner_image_id",
    "runner_dependency_manifest_sha256",
    "proxy_image_id",
    "forwarder_image_id",
    "proxy_dependency_manifest_sha256",
    "proxy_code_sha256",
    "container_launcher_sha256",
    "playwright_smoke_receipt_sha256",
    "playwright_smoke",
    "sandbox_runtime_sha256",
    "dockerfile_sha256",
    "proxy_dockerfile_sha256",
    "chromium_policy_sha256",
    "docker_client_sha256",
    "candidate_environment_digest",
    "catalog_sha256",
    "allowlist_digest",
    "runner_network_namespace_sha256",
    "forwarder_network_namespace_sha256",
    "proxy_network_namespace_sha256",
    "proxy_audit_sha256",
    "proxy_audit_path",
    "proxy_audit_event_count",
    "proxy_audit_allowed_count",
    "proxy_audit_rejected_count",
    "proxy_audit_failed_count",
    "proxy_audit_preflight_allowed_count",
    "proxy_audit_acquisition_allowed_count",
    "proxy_acquisition_host_metrics_digest",
    "proxy_acquisition_host_metrics",
    "evidence_ledger_live_official_hosts_digest",
    "evidence_ledger_live_official_hosts",
    "authority_claim_scope",
    "trusted_response_content_binding",
    "run_id",
    "current_pointer_sha256",
    "manifest_sha256",
)
_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "handler",
        "phase",
        "status",
        "task_id",
        "attempt",
        *_REQUIRED_RECEIPT_BINDINGS,
        "runner_argv",
        "environment_names",
        "scrubbed_environment",
        "source_projection",
        "started_at",
        "heartbeat_at",
        "finished_at",
        "returncode",
        "wrapper_argv_digest",
        "elapsed_seconds",
        "reason",
        "blocker",
        "proof_authoritative",
        "completion_authoritative",
        "reused_from_receipt_id",
        "rebound_at",
        "receipt_id",
    }
)


class TrustedPrevalidationActionError(ValueError):
    """A policy, binding, or sealed-action invariant failed."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    text = str(value or "")
    if not text.endswith("Z"):
        raise TrustedPrevalidationActionError(f"{field} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrustedPrevalidationActionError(
            f"{field} is not a UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise TrustedPrevalidationActionError(f"{field} is not UTC")
    return parsed


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_action_progress_marker(
    *,
    log_path: Path,
    cohort_root: Path,
) -> tuple[int, int, int, int, int]:
    """Return a cheap progress tuple that includes nested live-corpus writes.

    Official scrapes often stay silent on stdout for long stretches while they
    append under ``<run-id>/corpus/``. Watching only the impl log, the
    current-run pointer, and the cohort directory mtime treats that work as
    idle and SIGKILLs the container after ``progress_timeout_seconds``.
    """

    log_size = log_path.stat().st_size if log_path.exists() else 0
    pointer = cohort_root / "current_run.json"
    pointer_mtime = int(pointer.stat().st_mtime_ns) if pointer.exists() else 0
    root_mtime = int(cohort_root.stat().st_mtime_ns) if cohort_root.exists() else 0
    nested_mtime = 0
    nested_size = 0
    if cohort_root.is_dir():
        try:
            children = list(cohort_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir() or child.name == "archived":
                continue
            for rel in (
                "live_run_manifest.json",
                "corpus/state_refresh_progress.json",
            ):
                path = child / rel
                try:
                    stat = path.stat()
                except OSError:
                    continue
                nested_mtime = max(nested_mtime, int(stat.st_mtime_ns))
                nested_size += int(stat.st_size)
    return (log_size, pointer_mtime, root_mtime, nested_mtime, nested_size)


def _content_id(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustedPrevalidationActionError(f"{field} must be an object")
    return value


def _closed_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TrustedPrevalidationActionError(
            f"{field} contains unsupported fields: {', '.join(unknown)}"
        )


def _required_text(value: Mapping[str, Any], name: str, *, field: str = "") -> str:
    raw = value.get(name)
    if not isinstance(raw, str) or not raw.strip():
        prefix = f"{field}." if field else ""
        raise TrustedPrevalidationActionError(
            f"{prefix}{name} must be a nonempty string"
        )
    text = raw.strip()
    if "\x00" in text or "\n" in text or "\r" in text:
        raise TrustedPrevalidationActionError(f"{name} must be one line")
    return text


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TrustedPrevalidationActionError(f"{field} must be a relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or path.as_posix() in {".", ".."}
        or ".." in path.parts
        or "\x00" in normalized
    ):
        raise TrustedPrevalidationActionError(f"{field} is unsafe")
    return path.as_posix()


def _relative_paths(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrustedPrevalidationActionError(f"{field} must be a nonempty list")
    result = tuple(
        _relative_path(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise TrustedPrevalidationActionError(f"{field} contains duplicates")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrustedPrevalidationActionError(f"{field} must be a positive integer")
    return value


def _positive_seconds(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TrustedPrevalidationActionError(f"{field} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrustedPrevalidationActionError(
            f"{field} must be finite and positive"
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise TrustedPrevalidationActionError(f"{field} must be finite and positive")
    return result


def _contained(root: Path, relative: str, *, field: str) -> Path:
    root = root.resolve()
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrustedPrevalidationActionError(f"{field} escapes repository") from exc
    if candidate.is_symlink():
        raise TrustedPrevalidationActionError(f"{field} must not be a symlink")
    return candidate


@dataclass(frozen=True)
class SourceProjectionPolicy:
    shared_paths: tuple[str, ...]
    cohort_paths: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class SandboxPolicy:
    wrapper_path: str
    wrapper_sha256: str
    policy_path: str
    policy_sha256: str


@dataclass(frozen=True)
class TrustedPrevalidationActionPolicy:
    payload: Mapping[str, Any]
    policy_digest: str
    evidence_root: str
    receipt_root: str
    action_timeout_seconds: float
    progress_timeout_seconds: float
    max_attempts: int
    trusted_static_pre_action_gate: str
    adapter_evidence_ownership: str
    source_closure_policy_registry_path: str
    source_closure_policy_registry_schema: str
    source_projection: SourceProjectionPolicy
    sandbox: SandboxPolicy

    def cohort_for_task(self, task_id: str) -> str | None:
        return STATE_LAWS_TASK_COHORTS.get(str(task_id).strip())


def parse_state_laws_live_cohort_policy(
    raw: Mapping[str, Any],
) -> TrustedPrevalidationActionPolicy:
    """Parse the closed state-laws action policy from a scheduler config."""

    payload = dict(_mapping(raw, field="trusted_prevalidation_action"))
    _closed_fields(payload, _POLICY_FIELDS, field="trusted_prevalidation_action")
    exact_scalars = {
        "schema": TRUSTED_PREVALIDATION_ACTION_POLICY_SCHEMA,
        "handler": HANDLER,
        "phase": PHASE,
        "receipt_schema": TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA,
        "runner_path": RUNNER_PATH,
        "evidence_root_environment_variable": EVIDENCE_ENVIRONMENT_VARIABLE,
        "validation_receipt_environment_variable": RECEIPT_ENVIRONMENT_VARIABLE,
        "validation_receipt_sha256_environment_variable": (
            RECEIPT_SHA256_ENVIRONMENT_VARIABLE
        ),
        "trusted_static_pre_action_gate": TRUSTED_STATIC_PRE_ACTION_GATE,
        "adapter_evidence_ownership": ADAPTER_EVIDENCE_OWNERSHIP,
        "source_closure_policy_registry_path": (SOURCE_CLOSURE_POLICY_REGISTRY_PATH),
        "source_closure_policy_registry_schema": (
            SOURCE_CLOSURE_POLICY_REGISTRY_SCHEMA
        ),
    }
    for name, expected in exact_scalars.items():
        if payload.get(name) != expected:
            raise TrustedPrevalidationActionError(
                f"trusted_prevalidation_action.{name} must equal {expected!r}"
            )
    if payload.get("required_for_task_completion") is not True:
        raise TrustedPrevalidationActionError(
            "trusted_prevalidation_action.required_for_task_completion must be true"
        )
    for name in (
        "adapter_contract_catalog_binding_required",
        "adapter_contract_closed_mode_schema_required",
        "adapter_contract_reviewed_policy_digest_required",
        "adapter_evidence_api_signature_and_dataflow_required",
    ):
        if payload.get(name) is not True:
            raise TrustedPrevalidationActionError(
                f"trusted_prevalidation_action.{name} must be true"
            )
    if payload.get("max_parallel") != MAX_PARALLEL:
        raise TrustedPrevalidationActionError(
            f"trusted_prevalidation_action.max_parallel must equal {MAX_PARALLEL}"
        )
    if dict(_mapping(payload.get("task_cohorts"), field="task_cohorts")) != dict(
        STATE_LAWS_TASK_COHORTS
    ):
        raise TrustedPrevalidationActionError(
            "trusted_prevalidation_action.task_cohorts must be the exact "
            "SLR-009..SLR-021 to A..M mapping"
        )

    evidence_root = _relative_path(
        payload.get("evidence_root"), field="trusted_prevalidation_action.evidence_root"
    )
    receipt_root = _relative_path(
        payload.get("receipt_root"), field="trusted_prevalidation_action.receipt_root"
    )
    evidence_parts = PurePosixPath(evidence_root).parts
    if PurePosixPath(receipt_root).parts[: len(evidence_parts)] != evidence_parts:
        raise TrustedPrevalidationActionError(
            "receipt_root must be under evidence_root"
        )

    source = dict(_mapping(payload.get("source_projection"), field="source_projection"))
    _closed_fields(source, _SOURCE_FIELDS, field="source_projection")
    exact_source = {
        "mode": SOURCE_PROJECTION_MODE,
        "unrelated_cohort_paths_invalidate": False,
        "shared_path_changes_invalidate": True,
        "path_sha256_map_required": True,
        "digest_required": True,
    }
    for name, expected in exact_source.items():
        if source.get(name) != expected:
            raise TrustedPrevalidationActionError(
                f"source_projection.{name} must equal {expected!r}"
            )
    shared_paths = _relative_paths(
        source.get("shared_paths"), field="source_projection.shared_paths"
    )
    if shared_paths[:2] != (
        OFFICIAL_SOURCE_CATALOG_PATH,
        SOURCE_CLOSURE_POLICY_REGISTRY_PATH,
    ):
        raise TrustedPrevalidationActionError(
            "source_projection.shared_paths must begin with the sealed official "
            "catalog and source-closure policy registry"
        )
    cohort_raw = _mapping(
        source.get("cohort_paths"), field="source_projection.cohort_paths"
    )
    if tuple(sorted(cohort_raw)) != tuple(STATE_LAWS_TASK_COHORTS.values()):
        raise TrustedPrevalidationActionError(
            "source_projection.cohort_paths must contain exactly A..M"
        )
    cohort_paths = {
        cohort: _relative_paths(
            cohort_raw[cohort], field=f"source_projection.cohort_paths.{cohort}"
        )
        for cohort in sorted(cohort_raw)
    }
    all_cohort_paths = [path for paths in cohort_paths.values() for path in paths]
    if len(all_cohort_paths) != len(set(all_cohort_paths)):
        raise TrustedPrevalidationActionError(
            "source_projection cohort paths must be file-disjoint"
        )
    if set(shared_paths) & set(all_cohort_paths):
        raise TrustedPrevalidationActionError(
            "source_projection shared and cohort paths must be disjoint"
        )

    pre_action = dict(
        _mapping(
            payload.get("pre_action_validation"),
            field="pre_action_validation",
        )
    )
    _closed_fields(
        pre_action,
        _PRE_ACTION_VALIDATION_FIELDS,
        field="pre_action_validation",
    )
    if pre_action != {
        "schema": PRE_ACTION_VALIDATION_SCHEMA,
        "handler": PRE_ACTION_VALIDATION_HANDLER,
        "network": "disabled",
        "proposal_bound": True,
    }:
        raise TrustedPrevalidationActionError(
            "pre_action_validation must equal the sealed cohort-pytest policy"
        )

    retry = dict(_mapping(payload.get("retry"), field="retry"))
    _closed_fields(retry, _RETRY_FIELDS, field="retry")
    max_attempts = _positive_int(retry.get("max_attempts"), field="retry.max_attempts")
    if retry.get("resume_exact_fingerprint_only") is not True:
        raise TrustedPrevalidationActionError(
            "retry.resume_exact_fingerprint_only must be true"
        )

    sandbox = dict(_mapping(payload.get("sandbox"), field="sandbox"))
    _closed_fields(sandbox, _SANDBOX_FIELDS, field="sandbox")
    if sandbox.get("mode") != SANDBOX_MODE:
        raise TrustedPrevalidationActionError(
            f"sandbox.mode must equal {SANDBOX_MODE!r}"
        )
    if sandbox.get("required_capability") != SANDBOX_CAPABILITY:
        raise TrustedPrevalidationActionError(
            f"sandbox.required_capability must equal {SANDBOX_CAPABILITY!r}"
        )
    if sandbox.get("unavailable_is_failure") is not True:
        raise TrustedPrevalidationActionError(
            "sandbox.unavailable_is_failure must be true"
        )
    wrapper_path = _relative_path(
        sandbox.get("wrapper_path"), field="sandbox.wrapper_path"
    )
    policy_path = _relative_path(
        sandbox.get("policy_path"), field="sandbox.policy_path"
    )
    if wrapper_path != WRAPPER_PATH or policy_path != SANDBOX_POLICY_PATH:
        raise TrustedPrevalidationActionError(
            "sandbox wrapper/policy paths are not sealed"
        )
    wrapper_sha256 = str(sandbox.get("wrapper_sha256") or "")
    policy_sha256 = str(sandbox.get("policy_sha256") or "")
    if (
        HEX64.fullmatch(wrapper_sha256) is None
        or HEX64.fullmatch(policy_sha256) is None
    ):
        raise TrustedPrevalidationActionError(
            "sandbox digests must be lowercase SHA-256"
        )

    bindings = payload.get("receipt_bindings")
    if not isinstance(bindings, list) or tuple(bindings) != _REQUIRED_RECEIPT_BINDINGS:
        raise TrustedPrevalidationActionError(
            "receipt_bindings must equal the sealed ordered binding list"
        )

    return TrustedPrevalidationActionPolicy(
        payload=payload,
        policy_digest=_content_id(payload),
        evidence_root=evidence_root,
        receipt_root=receipt_root,
        action_timeout_seconds=_positive_seconds(
            payload.get("action_timeout_seconds"), field="action_timeout_seconds"
        ),
        progress_timeout_seconds=_positive_seconds(
            payload.get("progress_timeout_seconds"), field="progress_timeout_seconds"
        ),
        max_attempts=max_attempts,
        trusted_static_pre_action_gate=TRUSTED_STATIC_PRE_ACTION_GATE,
        adapter_evidence_ownership=ADAPTER_EVIDENCE_OWNERSHIP,
        source_closure_policy_registry_path=(SOURCE_CLOSURE_POLICY_REGISTRY_PATH),
        source_closure_policy_registry_schema=(SOURCE_CLOSURE_POLICY_REGISTRY_SCHEMA),
        source_projection=SourceProjectionPolicy(
            shared_paths=shared_paths,
            cohort_paths=cohort_paths,
        ),
        sandbox=SandboxPolicy(
            wrapper_path=wrapper_path,
            wrapper_sha256=wrapper_sha256,
            policy_path=policy_path,
            policy_sha256=policy_sha256,
        ),
    )


def load_state_laws_live_cohort_policy(
    scheduler_config_path: Path | str,
    *,
    expected_policy_digest: str = "",
) -> TrustedPrevalidationActionPolicy | None:
    """Load the action key from a duplicate-key-rejecting scheduler JSON."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrustedPrevalidationActionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    path = Path(scheduler_config_path).resolve()
    try:
        root = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except TrustedPrevalidationActionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustedPrevalidationActionError(
            f"trusted action scheduler config is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    root = _mapping(root, field="scheduler config")
    raw = root.get("trusted_prevalidation_action")
    if raw is None:
        return None
    result = parse_state_laws_live_cohort_policy(
        _mapping(raw, field="trusted_prevalidation_action")
    )
    expected = str(expected_policy_digest or "").strip()
    if expected and result.policy_digest != expected:
        raise TrustedPrevalidationActionError("trusted action policy digest mismatch")
    return result


def _verify_regular_file(path: Path, expected_sha256: str, *, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise TrustedPrevalidationActionError(f"{field} is missing or unsafe")
    if _sha256_file(path) != expected_sha256:
        raise TrustedPrevalidationActionError(f"{field} SHA-256 mismatch")


def _sandbox_capability(
    *,
    repo_root: Path,
    policy: TrustedPrevalidationActionPolicy,
) -> tuple[Path, Path]:
    del repo_root  # Sandbox assets are pinned to this accelerator checkout.
    accelerator_root = Path(__file__).resolve().parents[3]
    wrapper = _contained(
        accelerator_root,
        policy.sandbox.wrapper_path,
        field="sandbox.wrapper_path",
    )
    sandbox_policy_path = _contained(
        accelerator_root,
        policy.sandbox.policy_path,
        field="sandbox.policy_path",
    )
    _verify_regular_file(
        wrapper, policy.sandbox.wrapper_sha256, field="sandbox.wrapper_path"
    )
    _verify_regular_file(
        sandbox_policy_path,
        policy.sandbox.policy_sha256,
        field="sandbox.policy_path",
    )
    try:
        sandbox_policy = json.loads(sandbox_policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustedPrevalidationActionError("sandbox policy is unreadable") from exc
    sandbox_policy = _mapping(sandbox_policy, field="sandbox policy")
    if (
        sandbox_policy.get("schema") != SANDBOX_POLICY_SCHEMA
        or sandbox_policy.get("handler") != HANDLER
    ):
        raise TrustedPrevalidationActionError("sandbox policy identity mismatch")
    if sandbox_policy.get("available") is not True:
        reason = str(sandbox_policy.get("unavailable_reason") or "sandbox_unavailable")
        raise TrustedPrevalidationActionError(reason)
    network = _mapping(sandbox_policy.get("network"), field="sandbox policy network")
    if (
        network.get("mode") != SANDBOX_CAPABILITY
        or network.get("default") != "deny"
        or network.get("direct_egress") is not False
        or network.get("redirect_targets_rechecked") is not True
        or network.get("dns_rebinding_protected") is not True
    ):
        raise TrustedPrevalidationActionError(
            "sandbox official-egress capability is incomplete"
        )
    return wrapper, sandbox_policy_path


def accepted_proposal_source_projection(
    *,
    workspace_path: Path,
    cohort: str,
    policy: TrustedPrevalidationActionPolicy,
    proposal_validation: Any,
    proposal_action_diff_digest: str,
    required_changed_paths: Sequence[str] = (),
    allowed_extra_changed_paths: Sequence[str] = (),
) -> tuple[dict[str, str], str, dict[str, bytes]]:
    """Bind current bytes to immutable baseline plus accepted diff sources."""

    workspace = Path(workspace_path).resolve(strict=True)
    git_metadata = SEALED_GIT_PATH.stat()
    if (
        SEALED_GIT_PATH.is_symlink()
        or not SEALED_GIT_PATH.is_file()
        or git_metadata.st_uid != 0
        or stat.S_IMODE(git_metadata.st_mode) & 0o022
        or _sha256_file(SEALED_GIT_PATH) != SEALED_GIT_SHA256
    ):
        raise TrustedPrevalidationActionError("sealed Git client drifted")
    proposal = getattr(proposal_validation, "proposal", None)
    if (
        proposal is None
        or not bool(getattr(proposal_validation, "accepted", False))
        or not str(proposal_action_diff_digest or "")
    ):
        raise TrustedPrevalidationActionError(
            "accepted proposal is required for source projection"
        )
    baseline_commit = str(getattr(proposal, "baseline_id", "") or "")
    if re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None:
        raise TrustedPrevalidationActionError(
            "accepted proposal baseline commit is invalid"
        )
    git_environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "HOME": "/nonexistent/ipfs-accelerate-proposal-projection",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }

    def run_git(
        *arguments: str, text: bool = False
    ) -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(
                [
                    str(SEALED_GIT_PATH),
                    "--no-replace-objects",
                    "-C",
                    str(workspace),
                    *arguments,
                ],
                env=git_environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=text,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TrustedPrevalidationActionError(
                "sealed baseline inspection failed"
            ) from exc

    commit_identity = run_git(
        "rev-parse", "--verify", f"{baseline_commit}^{{commit}}", text=True
    )
    if (
        commit_identity.returncode != 0
        or commit_identity.stdout.strip() != baseline_commit
    ):
        raise TrustedPrevalidationActionError(
            "accepted proposal baseline is not the exact commit"
        )
    projection_paths = {
        *policy.source_projection.shared_paths,
        *policy.source_projection.cohort_paths[cohort],
    }
    required = {
        _relative_path(path, field="required_changed_paths")
        for path in required_changed_paths
    }
    allowed_extra = {
        _relative_path(path, field="allowed_extra_changed_paths")
        for path in allowed_extra_changed_paths
    }
    allowed_changes = projection_paths | allowed_extra
    report_path = f"docs/reports/state_laws_reindex/cohort_{cohort.lower()}.json"
    changed_paths = {
        _relative_path(str(path), field="proposal.changed_paths")
        for path in tuple(getattr(proposal, "changed_paths", ()) or ())
        if str(path or "").strip()
    }
    executable_changes = changed_paths - {report_path}
    if not required.issubset(executable_changes):
        raise TrustedPrevalidationActionError(
            "accepted proposal lacks a required executable path"
        )
    unexpected = executable_changes - allowed_changes
    if unexpected:
        raise TrustedPrevalidationActionError(
            "accepted proposal contains an unsealed executable path: "
            + ", ".join(sorted(unexpected)[:12])
        )
    accepted_sources: dict[str, bytes] = {}

    def baseline_entry(relative: str) -> tuple[bytes, str] | None:
        identity = run_git("ls-tree", baseline_commit, "--", relative, text=True)
        records = [line for line in identity.stdout.splitlines() if line]
        if identity.returncode != 0 or len(records) > 1:
            raise TrustedPrevalidationActionError(
                f"baseline source projection identity is invalid: {relative}"
            )
        if not records:
            return None
        metadata, separator, observed_path = records[0].partition("\t")
        fields = metadata.split()
        if (
            not separator
            or observed_path != relative
            or len(fields) != 3
            or fields[0] not in {"100644", "100755"}
            or fields[1] != "blob"
            or re.fullmatch(r"[0-9a-f]{40}", fields[2]) is None
        ):
            raise TrustedPrevalidationActionError(
                f"baseline source projection mode is unsafe: {relative}"
            )
        completed = run_git("show", f"{baseline_commit}:{relative}")
        if completed.returncode != 0:
            raise TrustedPrevalidationActionError(
                f"baseline source projection path is unavailable: {relative}"
            )
        return bytes(completed.stdout), fields[2]

    def git_blob_identity(content: bytes) -> str:
        return hashlib.sha1(
            b"blob " + str(len(content)).encode("ascii") + b"\0" + content,
            usedforsecurity=False,
        ).hexdigest()

    for entry in tuple(getattr(proposal, "candidate_diff", ()) or ()):
        old_path = str(getattr(entry, "old_path", "") or "").strip("/")
        new_path = str(getattr(entry, "new_path", "") or "").strip("/")
        paths = {path for path in (old_path, new_path) if path}
        if report_path in paths:
            continue
        change_kind = str(
            getattr(getattr(entry, "change_kind", None), "value", "")
            or getattr(entry, "change_kind", "")
            or ""
        )
        if (
            bool(getattr(entry, "binary", False))
            or not new_path
            or change_kind not in {"add", "modify"}
            or (change_kind == "add" and old_path)
            or (change_kind == "modify" and old_path != new_path)
            or new_path not in allowed_changes
            or new_path in accepted_sources
        ):
            raise TrustedPrevalidationActionError(
                "accepted executable diff must be unique same-path/add text"
            )
        after_source = getattr(entry, "after_source", None)
        if after_source is None:
            to_dict = getattr(entry, "to_dict", None)
            payload = to_dict(include_sources=True) if callable(to_dict) else {}
            after_source = payload.get("after_source")
        if not isinstance(after_source, str):
            raise TrustedPrevalidationActionError(
                "accepted executable diff lacks exact after_source"
            )
        before_source = getattr(entry, "before_source", None)
        if before_source is None:
            to_dict = getattr(entry, "to_dict", None)
            payload = to_dict(include_sources=True) if callable(to_dict) else {}
            before_source = payload.get("before_source")
        before_blob_id = str(getattr(entry, "before_blob_id", "") or "")
        after_blob_id = str(getattr(entry, "after_blob_id", "") or "")
        after_bytes = after_source.encode("utf-8", errors="surrogatepass")
        baseline = baseline_entry(new_path)
        if change_kind == "add":
            if baseline is not None or before_source is not None or before_blob_id:
                raise TrustedPrevalidationActionError(
                    "accepted addition conflicts with the baseline"
                )
        else:
            if baseline is None or not isinstance(before_source, str):
                raise TrustedPrevalidationActionError(
                    "accepted modification lacks the baseline source"
                )
            before_bytes = before_source.encode("utf-8", errors="surrogatepass")
            if before_bytes != baseline[0] or before_blob_id != baseline[1]:
                raise TrustedPrevalidationActionError(
                    "accepted modification before identity differs from baseline"
                )
        if after_blob_id != git_blob_identity(after_bytes):
            raise TrustedPrevalidationActionError(
                "accepted executable diff after blob identity is invalid"
            )
        accepted_sources[new_path] = after_bytes
    if set(accepted_sources) != executable_changes:
        raise TrustedPrevalidationActionError(
            "accepted changed paths differ from source-bound diff entries"
        )

    expected_projection_bytes: dict[str, bytes] = {}
    for relative in sorted(projection_paths):
        expected = accepted_sources.get(relative)
        if expected is None:
            baseline = baseline_entry(relative)
            if baseline is None:
                raise TrustedPrevalidationActionError(
                    f"baseline source projection path is unavailable: {relative}"
                )
            expected = baseline[0]
        expected_projection_bytes[relative] = expected
    for relative, expected in {
        **expected_projection_bytes,
        **accepted_sources,
    }.items():
        candidate = _contained(
            workspace,
            relative,
            field=f"accepted proposal source {relative}",
        )
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.read_bytes() != expected
        ):
            raise TrustedPrevalidationActionError(
                "workspace bytes differ from accepted proposal: " + relative
            )
    projection = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in sorted(expected_projection_bytes.items())
    }
    digest = _content_id(
        {
            "schema": "state-laws-reindex-live-source-projection-v1",
            "cohort_id": cohort,
            "files": projection,
        }
    )
    return projection, digest, dict(sorted(accepted_sources.items()))


def _proposal_binding(proposal_validation: Any, *, cohort: str) -> dict[str, str]:
    proposal = getattr(proposal_validation, "proposal", None)
    receipt = getattr(proposal_validation, "receipt", None)
    if (
        proposal is None
        or receipt is None
        or not bool(getattr(proposal_validation, "accepted", False))
    ):
        raise TrustedPrevalidationActionError("accepted proposal receipt is required")
    result = {
        "proposal_id": str(getattr(proposal, "proposal_id", "") or ""),
        "diff_digest": str(getattr(proposal, "diff_digest", "") or ""),
        "proposal_receipt_id": str(getattr(receipt, "receipt_id", "") or ""),
        "repository_tree_id": str(getattr(proposal, "repository_tree_id", "") or ""),
        "baseline_id": str(getattr(proposal, "baseline_id", "") or ""),
    }
    if any(not value for value in result.values()):
        raise TrustedPrevalidationActionError("proposal binding is incomplete")
    report_path = f"docs/reports/state_laws_reindex/cohort_{cohort.lower()}.json"
    action_entries: list[Mapping[str, Any]] = []
    for entry in tuple(getattr(proposal, "candidate_diff", ()) or ()):
        paths = {
            str(getattr(entry, "old_path", "") or "").strip("/"),
            str(getattr(entry, "new_path", "") or "").strip("/"),
        }
        if report_path in paths:
            continue
        to_dict = getattr(entry, "to_dict", None)
        if not callable(to_dict):
            raise TrustedPrevalidationActionError(
                "proposal candidate diff is not source-bindable"
            )
        action_entries.append(to_dict(include_sources=True))
    result["proposal_action_diff_digest"] = _content_id(action_entries)
    return result


def _task_metadata_binding(task: Any, cohort: str) -> None:
    metadata = {
        str(key).strip().lower().replace("_", " ").replace("-", " "): str(
            value or ""
        ).strip()
        for key, value in dict(getattr(task, "metadata", {}) or {}).items()
    }
    expected = {
        "trusted prevalidation action": HANDLER,
        "trusted action cohort": cohort,
        "trusted action required": "true",
        "trusted action receipt schema": TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA,
        "trusted action source projection": SOURCE_PROJECTION_MODE,
        "trusted static pre action gate": ("state_laws_adapter_evidence_contract_v1"),
        "adapter evidence ownership": (
            "cohort-adapter-native-stage-associate-frontier-replay@1"
        ),
        "pre action validation": pre_action_validation_command(cohort),
        "validation materialized outputs": (
            f"docs/reports/state_laws_reindex/cohort_{cohort.lower()}.json"
        ),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise TrustedPrevalidationActionError(
                f"task metadata {name!r} must equal {value!r}"
            )


def pre_action_validation_command(cohort: str) -> str:
    """Return the sole accelerator-authored command allowed before acquisition."""

    normalized = str(cohort or "").strip().upper()
    if normalized not in STATE_LAWS_TASK_COHORTS.values():
        raise TrustedPrevalidationActionError("pre-action cohort is invalid")
    return (
        "python -m pytest "
        f"tests/integration/legal_scrapers/test_state_laws_cohort_{normalized.lower()}.py "
        "-q"
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrustedPrevalidationActionError(
            f"required JSON is missing or unsafe: {path.name}"
        )
    try:

        def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise TrustedPrevalidationActionError(
                        f"duplicate JSON key in {path.name}: {key}"
                    )
                result[key] = item
            return result

        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except TrustedPrevalidationActionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustedPrevalidationActionError(f"invalid JSON: {path.name}") from exc
    return _mapping(value, field=path.name)


def _ledger_url_host(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrustedPrevalidationActionError(
            f"live evidence {field} must be an exact HTTP(S) URL"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise TrustedPrevalidationActionError(
            f"live evidence {field} has an invalid authority"
        ) from exc
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    default_port = 443 if scheme == "https" else 80
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, default_port}
    ):
        raise TrustedPrevalidationActionError(
            f"live evidence {field} must be credential-free HTTP(S) on its default port"
        )
    return host


def _live_official_ledger_binding(
    *,
    manifest: Mapping[str, Any],
    run_root: Path,
    cohort: str,
    acquisition_host_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    jurisdictions = COHORT_JURISDICTIONS.get(cohort)
    if jurisdictions is None or manifest.get("jurisdictions") != list(jurisdictions):
        raise TrustedPrevalidationActionError(
            "live manifest jurisdictions do not match the sealed cohort"
        )
    if manifest.get("corpus_root") != LIVE_CORPUS_ROOT:
        raise TrustedPrevalidationActionError(
            "live manifest corpus_root does not match the sealed runner"
        )
    artifacts = _mapping(manifest.get("artifacts"), field="live manifest artifacts")

    normalized_metrics: dict[str, Mapping[str, Any]] = {}
    for raw_host, raw_metric in acquisition_host_metrics.items():
        host = str(raw_host)
        if not host or host != host.lower().rstrip("."):
            raise TrustedPrevalidationActionError(
                "proxy acquisition metric host is not normalized"
            )
        metric = _mapping(raw_metric, field=f"proxy acquisition metric {host}")
        if set(metric) != {"request_count", "upstream_bytes", "downstream_bytes"}:
            raise TrustedPrevalidationActionError(
                "proxy acquisition metric fields are not exact"
            )
        if any(
            isinstance(metric.get(name), bool)
            or not isinstance(metric.get(name), int)
            or int(metric[name]) < 0
            for name in ("request_count", "upstream_bytes", "downstream_bytes")
        ):
            raise TrustedPrevalidationActionError(
                "proxy acquisition metric values are invalid"
            )
        normalized_metrics[host] = metric

    live_hosts: set[str] = set()
    for state in jurisdictions:
        relative = f"{LIVE_CORPUS_ROOT}/evidence/STATE-{state}-source-units.jsonl"
        artifact = _mapping(
            artifacts.get(relative), field=f"live manifest artifact {relative}"
        )
        if set(artifact) != {"sha256", "size"}:
            raise TrustedPrevalidationActionError(
                f"live evidence artifact fields are not exact: {relative}"
            )
        expected_sha256 = str(artifact.get("sha256") or "")
        expected_size = artifact.get("size")
        if (
            HEX64.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
            or expected_size > 2 * 1024 * 1024 * 1024
        ):
            raise TrustedPrevalidationActionError(
                f"live evidence artifact identity is invalid: {relative}"
            )
        ledger_path = _contained(run_root, relative, field=f"ledger {state}")
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise TrustedPrevalidationActionError(
                f"live evidence ledger is missing or unsafe: {state}"
            )
        if (
            ledger_path.stat().st_size != expected_size
            or _sha256_file(ledger_path) != expected_sha256
        ):
            raise TrustedPrevalidationActionError(
                f"live evidence ledger artifact binding mismatch: {state}"
            )

        state_live_record_count = 0
        try:
            with ledger_path.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if len(raw_line) > 8 * 1024 * 1024:
                        raise TrustedPrevalidationActionError(
                            f"live evidence ledger line is too large: {state}:{line_number}"
                        )
                    if not raw_line.strip():
                        continue

                    def reject_duplicates(
                        pairs: Sequence[tuple[str, Any]],
                        state_code: str = state,
                        ledger_line_number: int = line_number,
                    ) -> dict[str, Any]:
                        result: dict[str, Any] = {}
                        for key, item in pairs:
                            if key in result:
                                raise TrustedPrevalidationActionError(
                                    "duplicate JSON key in live evidence ledger: "
                                    f"{state_code}:{ledger_line_number}:{key}"
                                )
                            result[key] = item
                        return result

                    try:
                        record = json.loads(
                            raw_line,
                            object_pairs_hook=reject_duplicates,
                        )
                    except TrustedPrevalidationActionError:
                        raise
                    except json.JSONDecodeError as exc:
                        raise TrustedPrevalidationActionError(
                            f"invalid live evidence ledger JSON: {state}:{line_number}"
                        ) from exc
                    item = _mapping(
                        record, field=f"live evidence ledger {state}:{line_number}"
                    )
                    if (
                        item.get("schema") != SOURCE_UNIT_SCHEMA
                        or item.get("state_code") != state
                    ):
                        raise TrustedPrevalidationActionError(
                            f"live evidence ledger identity mismatch: {state}:{line_number}"
                        )
                    provenance = _mapping(
                        item.get("provenance"),
                        field=f"live evidence provenance {state}:{line_number}",
                    )
                    if provenance.get("live_official") is not True:
                        continue
                    if set(provenance) != {
                        "schema",
                        "provider",
                        "acquisition_mode",
                        "requested_url",
                        "final_url",
                        "live_official",
                    } or (
                        provenance.get("schema") != SOURCE_UNIT_PROVENANCE_SCHEMA
                        or provenance.get("acquisition_mode") != "live_official"
                        or not isinstance(provenance.get("provider"), str)
                        or not provenance.get("provider")
                    ):
                        raise TrustedPrevalidationActionError(
                            f"live evidence provenance is invalid: {state}:{line_number}"
                        )
                    for name, value in (
                        ("source_url", item.get("source_url")),
                        ("requested_url", provenance.get("requested_url")),
                        ("final_url", provenance.get("final_url")),
                    ):
                        live_hosts.add(
                            _ledger_url_host(
                                value,
                                field=f"{state}:{line_number}:{name}",
                            )
                        )
                    state_live_record_count += 1
        except UnicodeDecodeError as exc:
            raise TrustedPrevalidationActionError(
                f"live evidence ledger is not UTF-8: {state}"
            ) from exc
        if state_live_record_count < 1:
            raise TrustedPrevalidationActionError(
                f"live evidence ledger has no live-official source unit: {state}"
            )

    ordered_hosts = sorted(live_hosts)
    for host in ordered_hosts:
        metrics = normalized_metrics.get(host)
        if (
            metrics is None
            or int(metrics.get("request_count") or 0) < 1
            or int(metrics.get("downstream_bytes") or 0) < 1
        ):
            raise TrustedPrevalidationActionError(
                f"live evidence ledger host is absent from acquisition traffic: {host}"
            )
    return {
        "evidence_ledger_live_official_hosts_digest": _content_id(ordered_hosts),
        "evidence_ledger_live_official_hosts": ordered_hosts,
    }


def _sandbox_receipt_binding(
    *,
    sandbox_receipt_path: Path,
    sandbox_policy_path: Path,
    expected_policy_sha256: str,
    cohort: str,
    source_projection_digest: str,
    baseline_commit: str,
) -> dict[str, Any]:
    receipt = dict(_read_json(sandbox_receipt_path))
    exact_fields = {
        "schema",
        "handler",
        "status",
        "cohort",
        "policy_sha256",
        "runner_image",
        "proxy_image",
        "proxy_script_sha256",
        "container_launcher_sha256",
        "dockerfile_sha256",
        "proxy_dockerfile_sha256",
        "chromium_policy_sha256",
        "sandbox_runtime_sha256",
        "docker_client_sha256",
        "candidate_environment_digest",
        "catalog_sha256",
        "allowlist_digest",
        "allowlist_exact_hosts",
        "source_projection_digest",
        "baseline_commit",
        "baseline_regular_files_digest",
        "baseline_regular_file_count",
        "baseline_regular_file_bytes",
        "baseline_excluded_nonregular_count",
        "baseline_excluded_nonregular_digest",
        "candidate_workspace_snapshot_digest",
        "git_client_sha256",
        "playwright_instrumentation_digest",
        "playwright_launch_count",
        "playwright_helper_contract",
        "playwright_smoke",
        "preflight_runner_namespace",
        "runner_namespace",
        "forwarder_namespace",
        "proxy_namespace",
        "transport",
        "authority_claim_scope",
        "trusted_response_content_binding",
        "proxy_orderly_stop",
        "forwarder_orderly_stop",
        "proxy_audit_path",
        "proxy_audit",
        "runner_returncode",
        "cleanup",
        "cleanup_service",
        "startup_sweep_removed_container_ids_sha256",
        "startup_sweep_removed_lease_ids_sha256",
        "cleanup_claim_scope",
        "started_at",
        "finished_at",
        "failure",
    }
    if set(receipt) != exact_fields:
        raise TrustedPrevalidationActionError("sandbox receipt fields are not exact")
    if (
        receipt.get("schema") != SANDBOX_RECEIPT_SCHEMA
        or receipt.get("handler") != HANDLER
        or receipt.get("status") != "completed"
        or receipt.get("cohort") != cohort
        or receipt.get("policy_sha256") != expected_policy_sha256
        or receipt.get("source_projection_digest") != source_projection_digest
        or receipt.get("baseline_commit") != baseline_commit
        or receipt.get("runner_returncode") != 0
        or receipt.get("failure") != ""
        or receipt.get("proxy_orderly_stop") is not True
        or receipt.get("transport")
        != "isolated_forwarder_loopback_to_readonly_unix_socket_v1"
        or receipt.get("trusted_response_content_binding") is not False
        or receipt.get("authority_claim_scope")
        != "exact_host_sni_resolved_public_peer_timing_and_byte_counts_only"
        or receipt.get("forwarder_orderly_stop") is not True
    ):
        raise TrustedPrevalidationActionError("sandbox terminal binding mismatch")
    sandbox_started_at = _parse_utc_timestamp(
        receipt.get("started_at"), field="sandbox.started_at"
    )
    sandbox_finished_at = _parse_utc_timestamp(
        receipt.get("finished_at"), field="sandbox.finished_at"
    )
    if sandbox_finished_at < sandbox_started_at:
        raise TrustedPrevalidationActionError("sandbox timestamps are inverted")
    sandbox_policy = dict(_read_json(sandbox_policy_path))
    docker_policy = _mapping(sandbox_policy.get("docker"), field="sandbox.docker")
    assets_policy = _mapping(sandbox_policy.get("assets"), field="sandbox.assets")
    network_policy = _mapping(sandbox_policy.get("network"), field="sandbox.network")
    process_policy = _mapping(sandbox_policy.get("process"), field="sandbox.process")
    resources_policy = _mapping(
        sandbox_policy.get("resources"), field="sandbox.resources"
    )
    baseline_snapshot = {
        name: receipt.get(name)
        for name in (
            "baseline_commit",
            "baseline_regular_files_digest",
            "baseline_regular_file_count",
            "baseline_regular_file_bytes",
            "baseline_excluded_nonregular_count",
            "baseline_excluded_nonregular_digest",
        )
    }
    if (
        any(
            HEX64.fullmatch(str(baseline_snapshot[name] or "")) is None
            for name in (
                "baseline_regular_files_digest",
                "baseline_excluded_nonregular_digest",
            )
        )
        or any(
            isinstance(baseline_snapshot[name], bool)
            or not isinstance(baseline_snapshot[name], int)
            or baseline_snapshot[name] < 0
            for name in (
                "baseline_regular_file_count",
                "baseline_regular_file_bytes",
                "baseline_excluded_nonregular_count",
            )
        )
        or int(baseline_snapshot["baseline_regular_file_count"] or 0) < 1
        or receipt.get("git_client_sha256") != process_policy.get("git_client_sha256")
        or receipt.get("candidate_workspace_snapshot_digest")
        != _content_id(
            {
                **baseline_snapshot,
                "source_projection_digest": source_projection_digest,
            }
        )
    ):
        raise TrustedPrevalidationActionError(
            "sandbox baseline workspace snapshot binding mismatch"
        )
    cleanup_service = dict(
        _mapping(receipt.get("cleanup_service"), field="sandbox.cleanup_service")
    )
    cleanup_service_fields = {
        "unit",
        "main_pid",
        "main_pid_starttime",
        "control_group_sha256",
        "properties_digest",
        "systemd_run_sha256",
        "systemctl_sha256",
        "cleanup_python_sha256",
        "cleanup_env_sha256",
        "runtime_root",
        "independent_cgroup",
    }
    unit = str(cleanup_service.get("unit") or "")
    unit_match = re.fullmatch(
        r"ipfs-accelerate-slr-cleanup-([0-9a-f]{24})\.service",
        unit,
    )
    token = unit_match.group(1) if unit_match else ""
    runtime_root = str(cleanup_service.get("runtime_root") or "")
    lease_root = f"{runtime_root}/ipfs-accelerate-slr-action-{token}"
    cleanup_properties = [
        "Type=exec",
        "KillMode=control-group",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "RestrictAddressFamilies=AF_UNIX",
        "MemoryMax=134217728",
        "TasksMax=32",
        "UMask=0077",
        f"ReadWritePaths={lease_root}",
        f"WorkingDirectory={lease_root}",
    ]
    if (
        set(cleanup_service) != cleanup_service_fields
        or unit_match is None
        or runtime_root != process_policy.get("runtime_root")
        or cleanup_service.get("systemd_run_sha256")
        != process_policy.get("systemd_run_sha256")
        or cleanup_service.get("systemctl_sha256")
        != process_policy.get("systemctl_sha256")
        or cleanup_service.get("cleanup_python_sha256")
        != process_policy.get("cleanup_python_sha256")
        or cleanup_service.get("cleanup_env_sha256")
        != process_policy.get("cleanup_env_sha256")
        or cleanup_service.get("independent_cgroup") is not True
        or isinstance(cleanup_service.get("main_pid"), bool)
        or not isinstance(cleanup_service.get("main_pid"), int)
        or cleanup_service["main_pid"] < 1
        or not str(cleanup_service.get("main_pid_starttime") or "").isdigit()
        or HEX64.fullmatch(str(cleanup_service.get("control_group_sha256") or ""))
        is None
        or cleanup_service.get("properties_digest") != _content_id(cleanup_properties)
    ):
        raise TrustedPrevalidationActionError(
            "sandbox independent cleanup service binding mismatch"
        )
    cleanup = dict(_mapping(receipt.get("cleanup"), field="sandbox.cleanup"))
    if cleanup != {
        "runner_verification_removed": True,
        "proxy_verification_removed": True,
        "runner_removed": True,
        "forwarder_removed": True,
        "proxy_removed": True,
        "remaining_owned_container_count": 0,
        "complete": True,
    } or receipt.get("cleanup_claim_scope") != (
        "independent_user_transient_service_plus_dead_owner_exact_label_sweep"
    ):
        raise TrustedPrevalidationActionError("sandbox cleanup proof is incomplete")
    for field in (
        "startup_sweep_removed_container_ids_sha256",
        "startup_sweep_removed_lease_ids_sha256",
    ):
        values = receipt.get(field)
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(HEX64.fullmatch(str(value)) is None for value in values)
        ):
            raise TrustedPrevalidationActionError(f"sandbox {field} is invalid")

    def image_binding(name: str) -> tuple[str, str]:
        observed = dict(_mapping(receipt.get(name), field=name))
        expected = dict(_mapping(docker_policy.get(name), field=f"docker.{name}"))
        if observed != {
            "image_id": expected.get("image_id"),
            "os": expected.get("os"),
            "architecture": expected.get("architecture"),
            "dependency_manifest_mode": expected.get("dependency_manifest_mode"),
            "dependency_manifest_sha256": expected.get("dependency_manifest_sha256"),
            "labels": dict(
                sorted(
                    _mapping(expected.get("required_labels"), field="labels").items()
                )
            ),
        }:
            raise TrustedPrevalidationActionError(f"sandbox {name} identity mismatch")
        return str(observed["image_id"]), str(observed["dependency_manifest_sha256"])

    runner_image_id, runner_manifest = image_binding("runner_image")
    proxy_image_id, proxy_manifest = image_binding("proxy_image")
    asset_receipt_fields = {
        "proxy_script_sha256": "proxy_script_sha256",
        "container_launcher_sha256": "container_launcher_sha256",
        "sandbox_runtime_sha256": "sandbox_runtime_sha256",
        "dockerfile_sha256": "dockerfile_sha256",
        "proxy_dockerfile_sha256": "proxy_dockerfile_sha256",
        "chromium_policy_sha256": "chromium_policy_sha256",
    }
    for receipt_name, policy_name in asset_receipt_fields.items():
        if receipt.get(receipt_name) != assets_policy.get(policy_name):
            raise TrustedPrevalidationActionError(
                f"sandbox asset binding mismatch: {receipt_name}"
            )
    expected_exact_hosts = _mapping(
        network_policy.get("cohort_exact_hosts"),
        field="network.cohort_exact_hosts",
    ).get(cohort)
    expected_allowlist_digest = _content_id(
        {
            "schema": SANDBOX_ALLOWLIST_SCHEMA,
            "cohort": cohort,
            "hostname_match": "exact",
            "domains": expected_exact_hosts,
        }
    )
    if (
        receipt.get("docker_client_sha256") != docker_policy.get("client_sha256")
        or receipt.get("candidate_environment_digest")
        != process_policy.get("candidate_environment_digest")
        or receipt.get("catalog_sha256") != network_policy.get("catalog_sha256")
        or receipt.get("allowlist_exact_hosts") != expected_exact_hosts
        or receipt.get("allowlist_digest") != expected_allowlist_digest
    ):
        raise TrustedPrevalidationActionError("sandbox Docker/catalog binding mismatch")
    playwright_smoke = dict(
        _mapping(receipt.get("playwright_smoke"), field="sandbox.playwright_smoke")
    )
    smoke_fields = {
        "schema",
        "cohort",
        "attempts",
        "successful_requested_url",
        "successful_final_url",
        "http_status",
        "sha256",
        "size",
    }
    attempts = playwright_smoke.get("attempts")
    allowed_exact_hosts = set(receipt.get("allowlist_exact_hosts") or ())
    if (
        set(playwright_smoke) != smoke_fields
        or playwright_smoke.get("schema") != "state-laws-playwright-smoke-v1"
        or playwright_smoke.get("cohort") != cohort
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= 8
    ):
        raise TrustedPrevalidationActionError(
            "sandbox Playwright smoke receipt is invalid"
        )
    normalized_attempts: list[dict[str, Any]] = []
    completed_requested_urls: set[str] = set()
    active_requested_url = ""
    active_requested_count = 0
    for index, raw_attempt in enumerate(attempts):
        attempt = dict(_mapping(raw_attempt, field="sandbox Playwright smoke attempt"))
        status = str(attempt.get("status") or "")
        expected_attempt_fields = (
            {"requested_url", "status", "final_url", "http_status"}
            if status == "response"
            else {"requested_url", "status"}
        )
        try:
            requested_url = str(attempt.get("requested_url") or "")
            requested_host = _ledger_url_host(
                requested_url, field="Playwright requested_url"
            )
        except TrustedPrevalidationActionError as exc:
            raise TrustedPrevalidationActionError(
                "sandbox Playwright requested URL is invalid"
            ) from exc
        if requested_url != active_requested_url:
            if active_requested_url:
                completed_requested_urls.add(active_requested_url)
            active_requested_url = requested_url
            active_requested_count = 0
        active_requested_count += 1
        if (
            set(attempt) != expected_attempt_fields
            or status not in {"response", "no_response", "transport_error"}
            or urlsplit(requested_url).scheme != "https"
            or requested_host not in allowed_exact_hosts
            or requested_url in completed_requested_urls
            or active_requested_count > 2
            or (status == "response") != (index == len(attempts) - 1)
        ):
            raise TrustedPrevalidationActionError(
                "sandbox Playwright smoke attempt is invalid"
            )
        if status == "response":
            final_url = str(attempt.get("final_url") or "")
            final_host = _ledger_url_host(final_url, field="Playwright final_url")
            http_status = attempt.get("http_status")
            if (
                urlsplit(final_url).scheme != "https"
                or final_host not in allowed_exact_hosts
                or isinstance(http_status, bool)
                or not isinstance(http_status, int)
                or not 100 <= http_status <= 599
            ):
                raise TrustedPrevalidationActionError(
                    "sandbox Playwright smoke response is invalid"
                )
        normalized_attempts.append(attempt)
    smoke_success = normalized_attempts[-1]
    raw_smoke = {
        name: playwright_smoke[name] for name in smoke_fields - {"sha256", "size"}
    }
    smoke_bytes = _canonical_bytes(raw_smoke) + b"\n"
    if (
        playwright_smoke.get("successful_requested_url")
        != smoke_success.get("requested_url")
        or playwright_smoke.get("successful_final_url")
        != smoke_success.get("final_url")
        or playwright_smoke.get("http_status") != smoke_success.get("http_status")
        or playwright_smoke.get("sha256") != hashlib.sha256(smoke_bytes).hexdigest()
        or playwright_smoke.get("size") != len(smoke_bytes)
    ):
        raise TrustedPrevalidationActionError(
            "sandbox Playwright smoke success binding mismatch"
        )

    runner_mount_destinations = sorted(
        {
            "/workspace",
            f"/evidence/cohort_{cohort}",
            "/opt/ipfs-accelerate/state_laws_live_cohort_container_v1.py",
            "/etc/chromium/policies/managed",
        }
    )
    namespace_expectations = {
        "preflight_runner_namespace": (
            "none_shared_forwarder",
            "runner",
            runner_mount_destinations,
        ),
        "runner_namespace": (
            "none_shared_forwarder",
            "runner",
            runner_mount_destinations,
        ),
        "forwarder_namespace": (
            "none",
            "forwarder",
            [
                "/opt/ipfs-accelerate/state_laws_official_egress_proxy_v1.py",
                "/run/state-laws-proxy",
            ],
        ),
        "proxy_namespace": (
            "bridge",
            "proxy",
            [
                "/opt/ipfs-accelerate/state_laws_official_egress_proxy_v1.py",
                "/run/audit",
                "/run/policy/allowlist.json",
                "/run/state-laws-proxy",
            ],
        ),
    }

    def namespace_binding(name: str) -> str:
        namespace = dict(_mapping(receipt.get(name), field=name))
        mode, resource_name, mount_destinations = namespace_expectations[name]
        resource = _mapping(
            resources_policy.get(resource_name),
            field=f"sandbox.resources.{resource_name}",
        )
        digest = str(namespace.get("network_namespace_sha256") or "")
        if (
            set(namespace)
            != {
                "container_id",
                "network_mode",
                "network_namespace_sha256",
                "published_ports",
                "read_only_root",
                "candidate_user",
                "capabilities",
                "no_new_privileges",
                "mount_destinations",
                "nofile",
                "pids_limit",
            }
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(namespace.get("container_id") or "")
            )
            is None
            or namespace.get("network_mode") != mode
            or namespace.get("published_ports") is not False
            or namespace.get("read_only_root") is not True
            or namespace.get("candidate_user") != "1000:1000"
            or namespace.get("capabilities") != "none"
            or namespace.get("no_new_privileges") is not True
            or namespace.get("mount_destinations") != sorted(mount_destinations)
            or namespace.get("nofile") != resource.get("nofile")
            or namespace.get("pids_limit") != resource.get("pids")
            or HEX64.fullmatch(digest) is None
        ):
            raise TrustedPrevalidationActionError(
                f"sandbox namespace binding mismatch: {name}"
            )
        return digest

    runner_namespace = namespace_binding("runner_namespace")
    preflight_runner_namespace = namespace_binding("preflight_runner_namespace")
    forwarder_namespace = namespace_binding("forwarder_namespace")
    if not (runner_namespace == preflight_runner_namespace == forwarder_namespace):
        raise TrustedPrevalidationActionError(
            "runners did not share the sealed forwarder network namespace"
        )
    proxy_namespace = namespace_binding("proxy_namespace")
    if proxy_namespace == forwarder_namespace:
        raise TrustedPrevalidationActionError(
            "proxy unexpectedly shared the network-none namespace"
        )
    cleanup = _mapping(receipt.get("cleanup"), field="sandbox.cleanup")
    if cleanup.get("complete") is not True:
        raise TrustedPrevalidationActionError("sandbox cleanup did not complete")
    proxy_audit = dict(
        _mapping(receipt.get("proxy_audit"), field="sandbox.proxy_audit")
    )
    audit_fields = {
        "event_count",
        "allowed_count",
        "rejected_count",
        "failed_count",
        "ready_count",
        "terminal_count",
        "terminal_status",
        "first_event",
        "last_event",
        "allowed_hosts",
        "allowed_host_metrics",
        "sha256",
        "size",
        "preflight_allowed_count",
        "acquisition_allowed_count",
        "acquisition_host_metrics",
        "acquisition_host_metrics_digest",
    }
    audit_integer_fields = {
        "event_count",
        "allowed_count",
        "rejected_count",
        "failed_count",
        "ready_count",
        "terminal_count",
        "size",
        "preflight_allowed_count",
        "acquisition_allowed_count",
    }
    if set(proxy_audit) != audit_fields or any(
        isinstance(proxy_audit.get(field), bool)
        or not isinstance(proxy_audit.get(field), int)
        or int(proxy_audit[field]) < 0
        for field in audit_integer_fields
    ):
        raise TrustedPrevalidationActionError(
            "sandbox proxy audit summary is not closed"
        )
    acquisition_host_metrics = _mapping(
        proxy_audit.get("acquisition_host_metrics"),
        field="sandbox.proxy_audit.acquisition_host_metrics",
    )
    allowed_host_metrics = _mapping(
        proxy_audit.get("allowed_host_metrics"),
        field="sandbox.proxy_audit.allowed_host_metrics",
    )
    acquisition_host_metrics_digest = _content_id(
        dict(sorted(acquisition_host_metrics.items()))
    )
    acquisition_request_count = 0
    for host, raw_metrics in acquisition_host_metrics.items():
        try:
            normalized_host = str(host)
            metrics = dict(_mapping(raw_metrics, field=f"proxy metrics {host}"))
            values = tuple(
                metrics[name]
                for name in ("request_count", "upstream_bytes", "downstream_bytes")
            )
        except KeyError as exc:
            raise TrustedPrevalidationActionError(
                "sandbox proxy host metrics are incomplete"
            ) from exc
        if (
            set(metrics) != {"request_count", "upstream_bytes", "downstream_bytes"}
            or not normalized_host
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            or values[0] < 1
        ):
            raise TrustedPrevalidationActionError(
                "sandbox proxy host metrics are invalid"
            )
        acquisition_request_count += values[0]
    allowed_request_count = 0
    for host, raw_metrics in allowed_host_metrics.items():
        metrics = dict(_mapping(raw_metrics, field=f"allowed proxy metrics {host}"))
        if set(metrics) != {"request_count", "upstream_bytes", "downstream_bytes"}:
            raise TrustedPrevalidationActionError(
                "sandbox allowed proxy host metrics are incomplete"
            )
        values = tuple(
            metrics[name]
            for name in ("request_count", "upstream_bytes", "downstream_bytes")
        )
        if (
            host not in allowed_exact_hosts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            or values[0] < 1
        ):
            raise TrustedPrevalidationActionError(
                "sandbox allowed proxy host metrics are invalid"
            )
        acquisition_metrics = acquisition_host_metrics.get(host)
        if acquisition_metrics is not None and any(
            int(acquisition_metrics[field]) > int(metrics[field])
            for field in ("request_count", "upstream_bytes", "downstream_bytes")
        ):
            raise TrustedPrevalidationActionError(
                "sandbox acquisition metrics exceed the full proxy audit"
            )
        allowed_request_count += values[0]
    audit_path = sandbox_receipt_path.with_name("proxy-audit.jsonl")
    if (
        receipt.get("proxy_audit_path") != str(audit_path)
        or audit_path.is_symlink()
        or not audit_path.is_file()
        or proxy_audit.get("sha256") != _sha256_file(audit_path)
        or proxy_audit.get("size") != audit_path.stat().st_size
        or int(proxy_audit.get("acquisition_allowed_count") or 0) < 1
        or int(proxy_audit.get("terminal_count") or 0) != 1
        or proxy_audit.get("terminal_status") != "signal_stop"
        or proxy_audit.get("ready_count") != 1
        or proxy_audit.get("first_event") != "proxy_ready"
        or proxy_audit.get("last_event") != "proxy_terminal"
        or proxy_audit.get("event_count")
        != (
            int(proxy_audit["allowed_count"])
            + int(proxy_audit["rejected_count"])
            + int(proxy_audit["failed_count"])
            + int(proxy_audit["ready_count"])
            + int(proxy_audit["terminal_count"])
        )
        or proxy_audit.get("allowed_count")
        != int(proxy_audit["preflight_allowed_count"])
        + int(proxy_audit["acquisition_allowed_count"])
        or int(proxy_audit["preflight_allowed_count"]) < 1
        or proxy_audit.get("acquisition_host_metrics_digest")
        != acquisition_host_metrics_digest
        or not acquisition_host_metrics
        or not set(acquisition_host_metrics).issubset(allowed_exact_hosts)
        or proxy_audit.get("allowed_hosts") != sorted(allowed_host_metrics)
        or allowed_request_count != int(proxy_audit["allowed_count"])
        or acquisition_request_count
        != int(proxy_audit.get("acquisition_allowed_count") or 0)
    ):
        raise TrustedPrevalidationActionError("sandbox proxy audit binding mismatch")
    sandbox_receipt_sha256 = _sha256_file(sandbox_receipt_path)
    return {
        "sandbox_policy_sha256": expected_policy_sha256,
        "sandbox_receipt_path": str(sandbox_receipt_path),
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
        **baseline_snapshot,
        "candidate_workspace_snapshot_digest": str(
            receipt["candidate_workspace_snapshot_digest"]
        ),
        "git_client_sha256": str(receipt["git_client_sha256"]),
        "cleanup_service": cleanup_service,
        "cleanup": cleanup,
        "cleanup_claim_scope": str(receipt["cleanup_claim_scope"]),
        "startup_sweep_removed_container_ids_sha256": list(
            receipt["startup_sweep_removed_container_ids_sha256"]
        ),
        "startup_sweep_removed_lease_ids_sha256": list(
            receipt["startup_sweep_removed_lease_ids_sha256"]
        ),
        "runner_image_id": runner_image_id,
        "preflight_runner_image_id": runner_image_id,
        "runner_dependency_manifest_sha256": runner_manifest,
        "proxy_image_id": proxy_image_id,
        "forwarder_image_id": proxy_image_id,
        "proxy_dependency_manifest_sha256": proxy_manifest,
        "proxy_code_sha256": str(receipt["proxy_script_sha256"]),
        "container_launcher_sha256": str(receipt["container_launcher_sha256"]),
        "playwright_smoke_receipt_sha256": str(playwright_smoke["sha256"]),
        "playwright_smoke": playwright_smoke,
        "sandbox_runtime_sha256": str(receipt["sandbox_runtime_sha256"]),
        "dockerfile_sha256": str(receipt["dockerfile_sha256"]),
        "proxy_dockerfile_sha256": str(receipt["proxy_dockerfile_sha256"]),
        "chromium_policy_sha256": str(receipt["chromium_policy_sha256"]),
        "docker_client_sha256": str(receipt["docker_client_sha256"]),
        "candidate_environment_digest": str(receipt["candidate_environment_digest"]),
        "catalog_sha256": str(receipt["catalog_sha256"]),
        "allowlist_digest": str(receipt["allowlist_digest"]),
        "runner_network_namespace_sha256": runner_namespace,
        "forwarder_network_namespace_sha256": forwarder_namespace,
        "proxy_network_namespace_sha256": proxy_namespace,
        "proxy_audit_sha256": str(proxy_audit["sha256"]),
        "proxy_audit_path": str(audit_path),
        "proxy_audit_event_count": int(proxy_audit["event_count"]),
        "proxy_audit_allowed_count": int(proxy_audit["allowed_count"]),
        "proxy_audit_rejected_count": int(proxy_audit["rejected_count"]),
        "proxy_audit_failed_count": int(proxy_audit["failed_count"]),
        "proxy_audit_preflight_allowed_count": int(
            proxy_audit["preflight_allowed_count"]
        ),
        "proxy_audit_acquisition_allowed_count": int(
            proxy_audit["acquisition_allowed_count"]
        ),
        "proxy_acquisition_host_metrics_digest": acquisition_host_metrics_digest,
        "proxy_acquisition_host_metrics": dict(
            sorted(acquisition_host_metrics.items())
        ),
        "authority_claim_scope": str(receipt["authority_claim_scope"]),
        "trusted_response_content_binding": False,
    }


def _live_evidence_binding(
    *,
    evidence_root: Path,
    cohort: str,
    expected_source_projection: Mapping[str, str],
    expected_source_digest: str,
    acquisition_host_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_root = evidence_root / f"cohort_{cohort}"
    pointer_path = cohort_root / "current_run.json"
    pointer = _read_json(pointer_path)
    if set(pointer) != {"schema", "run_id", "manifest_path", "manifest_sha256"}:
        raise TrustedPrevalidationActionError(
            "current-run pointer fields are not exact"
        )
    if pointer.get("schema") != CURRENT_RUN_SCHEMA:
        raise TrustedPrevalidationActionError("current-run pointer schema mismatch")
    run_id = str(pointer.get("run_id") or "")
    manifest_sha256 = str(pointer.get("manifest_sha256") or "")
    if HEX64.fullmatch(run_id) is None or HEX64.fullmatch(manifest_sha256) is None:
        raise TrustedPrevalidationActionError("current-run pointer digests are invalid")
    manifest_relative = _relative_path(
        pointer.get("manifest_path"), field="manifest_path"
    )
    if manifest_relative != f"{run_id}/live_run_manifest.json":
        raise TrustedPrevalidationActionError(
            "current-run pointer path/run_id mismatch"
        )
    manifest_path = _contained(cohort_root, manifest_relative, field="manifest_path")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise TrustedPrevalidationActionError("current-run manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    exact = {
        "schema": LIVE_RUN_SCHEMA,
        "mode": LIVE_MODE,
        "status": "candidate_complete",
        "run_id": run_id,
        "cohort_id": cohort,
        "run_fingerprint": run_id,
        "source_tree_digest": expected_source_digest,
    }
    for name, expected in exact.items():
        if manifest.get(name) != expected:
            raise TrustedPrevalidationActionError(
                f"live manifest {name} does not match the sealed action"
            )
    source_projection = _mapping(
        manifest.get("source_projection"), field="source_projection"
    )
    if source_projection.get(
        "schema"
    ) != "state-laws-reindex-live-source-projection-v1" or source_projection.get(
        "files"
    ) != dict(sorted(expected_source_projection.items())):
        raise TrustedPrevalidationActionError(
            "live manifest source projection mismatch"
        )
    refresh = _mapping(manifest.get("refresh"), field="refresh")
    if refresh.get("returncode") != 0:
        raise TrustedPrevalidationActionError("live refresh did not exit successfully")
    ledger_binding = _live_official_ledger_binding(
        manifest=manifest,
        run_root=manifest_path.parent,
        cohort=cohort,
        acquisition_host_metrics=acquisition_host_metrics,
    )
    return {
        "run_id": run_id,
        "current_pointer_sha256": _sha256_file(pointer_path),
        "manifest_sha256": manifest_sha256,
        **ledger_binding,
    }


def _exact_environment(evidence_root: Path) -> dict[str, str]:
    # Values are constants or supervisor-owned paths.  No ambient credential,
    # proxy, home, Python path, or provider variable crosses the action boundary.
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        EVIDENCE_ENVIRONMENT_VARIABLE: str(evidence_root),
        "TZ": "UTC",
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("receipt_id", None)
    body["receipt_id"] = _content_id(body)
    atomic_write_json(path, body)
    return body


def _terminal_receipt_valid(
    path: Path,
    *,
    bindings: Mapping[str, Any],
    evidence_root: Path,
    cohort: str,
    source_projection: Mapping[str, str],
    source_digest: str,
    sandbox_policy_path: Path,
    sandbox_policy_sha256: str,
) -> dict[str, Any] | None:
    try:
        receipt = dict(_read_json(path))
        if set(receipt) != _TERMINAL_RECEIPT_FIELDS:
            return None
        claimed = str(receipt.pop("receipt_id") or "")
        if claimed != _content_id(receipt):
            return None
        receipt["receipt_id"] = claimed
        if (
            receipt.get("schema") != TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA
            or receipt.get("handler") != HANDLER
            or receipt.get("phase") != PHASE
            or receipt.get("status") != "completed"
            or receipt.get("reason")
            not in {
                "trusted_prevalidation_action_completed",
                "trusted_prevalidation_action_receipt_reused",
            }
            or receipt.get("blocker") != ""
            or receipt.get("returncode") != 0
            or receipt.get("proof_authoritative") is not False
            or receipt.get("completion_authoritative") is not False
        ):
            return None
        elapsed_seconds = receipt.get("elapsed_seconds")
        if (
            not isinstance(receipt.get("runner_argv"), list)
            or any(not isinstance(value, str) for value in receipt["runner_argv"])
            or _content_id(receipt["runner_argv"]) != bindings.get("argv_digest")
            or receipt.get("environment_names")
            != sorted(_exact_environment(evidence_root))
            or receipt.get("scrubbed_environment") != list(SCRUBBED_ENVIRONMENT_NAMES)
            or receipt.get("source_projection")
            != dict(sorted(source_projection.items()))
            or HEX64.fullmatch(str(receipt.get("wrapper_argv_digest") or "")) is None
            or isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or elapsed_seconds < 0
        ):
            return None
        started_at = _parse_utc_timestamp(
            receipt.get("started_at"), field="receipt.started_at"
        )
        heartbeat_at = _parse_utc_timestamp(
            receipt.get("heartbeat_at"), field="receipt.heartbeat_at"
        )
        finished_at = _parse_utc_timestamp(
            receipt.get("finished_at"), field="receipt.finished_at"
        )
        if not started_at <= heartbeat_at <= finished_at:
            return None
        rebound_at_raw = receipt.get("rebound_at")
        if rebound_at_raw is not None:
            rebound_at = _parse_utc_timestamp(
                rebound_at_raw, field="receipt.rebound_at"
            )
            if rebound_at < finished_at:
                return None
        # Proposal IDs legitimately change when hermetic certification writes
        # only the declared cohort report.  The action execution identity is the
        # task/policy/argv/cohort source projection; a reused receipt is rebound
        # to the newly accepted proposal below.
        proposal_specific = {
            "proposal_id",
            "diff_digest",
            "proposal_receipt_id",
            "repository_tree_id",
        }
        for name, value in bindings.items():
            if name in proposal_specific:
                continue
            if receipt.get(name) != value:
                return None
        sandbox = _sandbox_receipt_binding(
            sandbox_receipt_path=path.with_name("sandbox-receipt.json"),
            sandbox_policy_path=sandbox_policy_path,
            expected_policy_sha256=sandbox_policy_sha256,
            cohort=cohort,
            source_projection_digest=source_digest,
            baseline_commit=str(bindings["baseline_id"]),
        )
        if any(receipt.get(name) != value for name, value in sandbox.items()):
            return None
        evidence = _live_evidence_binding(
            evidence_root=evidence_root,
            cohort=cohort,
            expected_source_projection=source_projection,
            expected_source_digest=source_digest,
            acquisition_host_metrics=sandbox["proxy_acquisition_host_metrics"],
        )
        if any(receipt.get(name) != value for name, value in evidence.items()):
            return None
        return receipt
    except (OSError, TrustedPrevalidationActionError, ValueError):
        return None


def _execute_state_laws_live_cohort_action_locked(
    *,
    policy: TrustedPrevalidationActionPolicy,
    repo_root: Path,
    workspace_path: Path,
    task: Any,
    canonical_task_cid: str,
    proposal_validation: Any,
    attempt: int,
    log_path: Path,
    cancel_requested: Callable[[], bool | str] | None = None,
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
    source_projection_builder: Callable[
        ..., tuple[dict[str, str], str, dict[str, bytes]]
    ] = accepted_proposal_source_projection,
    sandbox_capability: Callable[..., tuple[Path, Path]] = _sandbox_capability,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run or idempotently reuse one proposal-bound trusted acquisition."""

    root = Path(repo_root).resolve()
    workspace = Path(workspace_path).resolve()
    task_id = str(getattr(task, "task_id", "") or "").strip()
    cohort = policy.cohort_for_task(task_id)
    if cohort is None:
        return {"required": False, "passed": True, "attempted": False}
    if not canonical_task_cid:
        raise TrustedPrevalidationActionError("canonical task CID is required")
    if isinstance(attempt, bool) or int(attempt) < 1:
        raise TrustedPrevalidationActionError("trusted action attempt must be positive")
    if int(attempt) > policy.max_attempts:
        return {
            "required": True,
            "attempted": False,
            "passed": False,
            "reason": "trusted_prevalidation_action_retry_budget_exhausted",
            "task_id": task_id,
            "cohort": cohort,
            "attempt": int(attempt),
            "max_attempts": policy.max_attempts,
        }
    _task_metadata_binding(task, cohort)
    proposal = _proposal_binding(proposal_validation, cohort=cohort)
    cohort_test_path = (
        f"tests/integration/legal_scrapers/test_state_laws_cohort_{cohort.lower()}.py"
    )
    source_projection, source_digest, _accepted_sources = source_projection_builder(
        workspace_path=workspace,
        cohort=cohort,
        policy=policy,
        proposal_validation=proposal_validation,
        proposal_action_diff_digest=proposal["proposal_action_diff_digest"],
        required_changed_paths=(cohort_test_path,),
        allowed_extra_changed_paths=(cohort_test_path,),
    )
    evidence_root = _contained(root, policy.evidence_root, field="evidence_root")
    receipt_root = _contained(root, policy.receipt_root, field="receipt_root")
    cohort_root = evidence_root / f"cohort_{cohort}"
    runner = _contained(workspace, RUNNER_PATH, field="runner_path")
    if runner.is_symlink() or not runner.is_file():
        raise TrustedPrevalidationActionError("live cohort runner is missing or unsafe")

    runner_argv = [
        str(Path(sys.executable).resolve()),
        str(runner.resolve()),
        "--cohort",
        cohort,
        "--execute-live",
        "--max-parallel",
        str(MAX_PARALLEL),
    ]
    argv_digest = _content_id(runner_argv)
    candidate_source_fingerprint = _content_id(
        {
            "schema": SOURCE_PROJECTION_MODE,
            "baseline_id": proposal["baseline_id"],
            "proposal_action_diff_digest": proposal["proposal_action_diff_digest"],
            "source_projection_digest": source_digest,
        }
    )
    bindings: dict[str, Any] = {
        "canonical_task_cid": str(canonical_task_cid),
        **proposal,
        "candidate_source_fingerprint": candidate_source_fingerprint,
        "cohort": cohort,
        "policy_digest": policy.policy_digest,
        "argv_digest": argv_digest,
        "source_projection_digest": source_digest,
    }
    action_key = _content_id(
        {
            "handler": HANDLER,
            "task_id": task_id,
            "canonical_task_cid": canonical_task_cid,
            "cohort": cohort,
            "policy_digest": policy.policy_digest,
            "source_projection_digest": source_digest,
            "candidate_source_fingerprint": candidate_source_fingerprint,
        }
    )
    action_dir = receipt_root / task_id / action_key
    action_dir.mkdir(parents=True, exist_ok=True)
    lease_path = action_dir / "action.lock"
    receipt_path = action_dir / "receipt.json"
    sandbox_policy_for_reuse = _contained(
        Path(__file__).resolve().parents[3],
        policy.sandbox.policy_path,
        field="sandbox.policy_path",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with lease_path.open("a+b") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "required": True,
                "attempted": False,
                "passed": False,
                "reason": "trusted_prevalidation_action_lease_busy",
                "task_id": task_id,
                "cohort": cohort,
                "source_projection_digest": source_digest,
            }

        reused = _terminal_receipt_valid(
            receipt_path,
            bindings=bindings,
            evidence_root=evidence_root,
            cohort=cohort,
            source_projection=source_projection,
            source_digest=source_digest,
            sandbox_policy_path=sandbox_policy_for_reuse,
            sandbox_policy_sha256=policy.sandbox.policy_sha256,
        )
        if reused is not None:
            prior_receipt_id = str(reused.get("receipt_id") or "")
            rebound = dict(reused)
            rebound.update(
                {
                    **proposal,
                    "reason": "trusted_prevalidation_action_receipt_reused",
                    "reused_from_receipt_id": prior_receipt_id,
                    "rebound_at": _utc_now(),
                }
            )
            rebound = _write_receipt(receipt_path, rebound)
            return {
                "required": True,
                "attempted": False,
                "reused": True,
                "passed": True,
                "reason": "trusted_prevalidation_action_receipt_reused",
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha256_file(receipt_path),
                **rebound,
            }

        started_at = _utc_now()
        running: dict[str, Any] = {
            "schema": TRUSTED_PREVALIDATION_ACTION_RECEIPT_SCHEMA,
            "handler": HANDLER,
            "phase": PHASE,
            "status": "running",
            "task_id": task_id,
            "attempt": int(attempt),
            **bindings,
            "runner_argv": runner_argv,
            "environment_names": sorted(_exact_environment(evidence_root)),
            "scrubbed_environment": list(SCRUBBED_ENVIRONMENT_NAMES),
            "source_projection": dict(sorted(source_projection.items())),
            "started_at": started_at,
            "heartbeat_at": started_at,
            "finished_at": None,
            "returncode": None,
            "wrapper_argv_digest": "",
            "elapsed_seconds": 0.0,
            "reason": "",
            "blocker": "",
            "reused_from_receipt_id": "",
            "rebound_at": None,
            "run_id": "",
            "current_pointer_sha256": "",
            "manifest_sha256": "",
            "sandbox_policy_sha256": policy.sandbox.policy_sha256,
            "sandbox_receipt_path": "",
            "sandbox_receipt_sha256": "",
            "baseline_commit": proposal["baseline_id"],
            "baseline_regular_files_digest": "",
            "baseline_regular_file_count": 0,
            "baseline_regular_file_bytes": 0,
            "baseline_excluded_nonregular_count": 0,
            "baseline_excluded_nonregular_digest": "",
            "candidate_workspace_snapshot_digest": "",
            "git_client_sha256": "",
            "cleanup_service": {},
            "cleanup": {},
            "cleanup_claim_scope": "",
            "startup_sweep_removed_container_ids_sha256": [],
            "startup_sweep_removed_lease_ids_sha256": [],
            "runner_image_id": "",
            "preflight_runner_image_id": "",
            "runner_dependency_manifest_sha256": "",
            "proxy_image_id": "",
            "forwarder_image_id": "",
            "proxy_dependency_manifest_sha256": "",
            "proxy_code_sha256": "",
            "container_launcher_sha256": "",
            "playwright_smoke_receipt_sha256": "",
            "playwright_smoke": {},
            "sandbox_runtime_sha256": "",
            "dockerfile_sha256": "",
            "proxy_dockerfile_sha256": "",
            "chromium_policy_sha256": "",
            "docker_client_sha256": "",
            "candidate_environment_digest": "",
            "catalog_sha256": "",
            "allowlist_digest": "",
            "runner_network_namespace_sha256": "",
            "forwarder_network_namespace_sha256": "",
            "proxy_network_namespace_sha256": "",
            "proxy_audit_sha256": "",
            "proxy_audit_path": "",
            "proxy_audit_event_count": 0,
            "proxy_audit_allowed_count": 0,
            "proxy_audit_rejected_count": 0,
            "proxy_audit_failed_count": 0,
            "proxy_audit_preflight_allowed_count": 0,
            "proxy_audit_acquisition_allowed_count": 0,
            "proxy_acquisition_host_metrics_digest": "",
            "proxy_acquisition_host_metrics": {},
            "evidence_ledger_live_official_hosts_digest": "",
            "evidence_ledger_live_official_hosts": [],
            "authority_claim_scope": "",
            "trusted_response_content_binding": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        running = _write_receipt(receipt_path, running)
        if heartbeat is not None:
            try:
                heartbeat(running)
            except Exception:  # noqa: BLE001,S110 - telemetry cannot break the action
                pass

        report_path = _contained(
            workspace,
            f"docs/reports/state_laws_reindex/cohort_{cohort.lower()}.json",
            field="validation materialized output",
        )
        if report_path.exists() or report_path.is_symlink():
            failed = dict(running)
            failed.update(
                {
                    "status": "failed",
                    "reason": "validation_materialized_output_present_before_action",
                    "heartbeat_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "returncode": 1,
                }
            )
            failed = _write_receipt(receipt_path, failed)
            return {
                "required": True,
                "attempted": False,
                "passed": False,
                "receipt_path": str(receipt_path),
                **failed,
            }

        try:
            wrapper, sandbox_policy_path = sandbox_capability(
                repo_root=root,
                policy=policy,
            )
        except (OSError, TrustedPrevalidationActionError) as exc:
            failed = dict(running)
            failed.update(
                {
                    "status": "failed",
                    "reason": "sandbox_unavailable",
                    "blocker": str(exc),
                    "heartbeat_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "returncode": EXIT_SANDBOX_UNAVAILABLE,
                }
            )
            failed = _write_receipt(receipt_path, failed)
            return {
                "required": True,
                "attempted": False,
                "passed": False,
                "reason": "sandbox_unavailable",
                "blocker": str(exc),
                "receipt_path": str(receipt_path),
                **failed,
            }

        source_projection_manifest_path = action_dir / "source-projection.json"
        source_projection_manifest = {
            "schema": "state-laws-reindex-live-source-projection-v1",
            "cohort_id": cohort,
            "files": dict(sorted(source_projection.items())),
            "digest": source_digest,
        }
        atomic_write_json(
            source_projection_manifest_path,
            source_projection_manifest,
        )
        sandbox_receipt_path = action_dir / "sandbox-receipt.json"
        wrapper_argv = [
            str(Path(sys.executable).resolve()),
            str(wrapper.resolve()),
            "--policy",
            str(sandbox_policy_path.resolve()),
            "--policy-sha256",
            policy.sandbox.policy_sha256,
            "--worktree",
            str(workspace),
            "--evidence-root",
            str(evidence_root),
            "--cohort",
            cohort,
            "--runner",
            str(runner.resolve()),
            "--max-parallel",
            str(MAX_PARALLEL),
            "--sandbox-receipt",
            str(sandbox_receipt_path),
            "--source-projection-manifest",
            str(source_projection_manifest_path),
            "--source-projection-digest",
            source_digest,
            "--baseline-id",
            proposal["baseline_id"],
        ]
        running["wrapper_argv_digest"] = _content_id(wrapper_argv)
        running = _write_receipt(receipt_path, running)

        process: subprocess.Popen[Any] | None = None
        started = monotonic()
        last_progress = started
        heartbeat_due = started
        progress_marker: tuple[int, int, int, int, int] | None = None
        returncode = 1
        failure_reason = "trusted_prevalidation_action_failed"
        try:
            with log_path.open("ab") as output:
                process = popen(
                    wrapper_argv,
                    cwd=str(root),
                    env=_exact_environment(evidence_root),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    now = monotonic()
                    cancellation = cancel_requested() if cancel_requested else False
                    if cancellation:
                        failure_reason = (
                            str(cancellation)
                            if isinstance(cancellation, str)
                            else "cancellation_requested"
                        )
                        raise InterruptedError(failure_reason)
                    if now - started >= policy.action_timeout_seconds:
                        failure_reason = "trusted_prevalidation_action_hard_timeout"
                        raise TimeoutError(failure_reason)
                    try:
                        marker = _live_action_progress_marker(
                            log_path=log_path,
                            cohort_root=cohort_root,
                        )
                    except OSError:
                        marker = (0, 0, 0, 0, 0)
                    if marker != progress_marker:
                        progress_marker = marker
                        last_progress = now
                    if now - last_progress >= policy.progress_timeout_seconds:
                        failure_reason = "trusted_prevalidation_action_progress_timeout"
                        raise TimeoutError(failure_reason)
                    if now >= heartbeat_due:
                        running["heartbeat_at"] = _utc_now()
                        running["elapsed_seconds"] = round(max(0.0, now - started), 3)
                        running = _write_receipt(receipt_path, running)
                        if heartbeat is not None:
                            try:
                                heartbeat(running)
                            except Exception:  # noqa: BLE001,S110 - best-effort telemetry
                                pass
                        heartbeat_due = now + min(
                            30.0, policy.progress_timeout_seconds / 3.0
                        )
                    sleep(0.25)
                returncode = int(process.wait())
        except (InterruptedError, TimeoutError, OSError) as exc:
            if process is not None and process.poll() is None:
                terminate_pid_tree(
                    process.pid,
                    grace_seconds=5.0,
                    freeze_first=True,
                    require_gone=True,
                    owned_process_group_id=process.pid,
                )
                process.wait()
            if isinstance(exc, OSError):
                failure_reason = "trusted_prevalidation_action_launch_failed"
                returncode = 127
        finally:
            if process is not None and process.poll() is None:
                terminate_pid_tree(
                    process.pid,
                    grace_seconds=5.0,
                    freeze_first=True,
                    require_gone=True,
                    owned_process_group_id=process.pid,
                )
                process.wait()

        evidence: dict[str, Any] = {}
        sandbox_evidence: dict[str, Any] = {}
        if returncode == 0:
            try:
                sandbox_evidence = _sandbox_receipt_binding(
                    sandbox_receipt_path=sandbox_receipt_path,
                    sandbox_policy_path=sandbox_policy_path,
                    expected_policy_sha256=policy.sandbox.policy_sha256,
                    cohort=cohort,
                    source_projection_digest=source_digest,
                    baseline_commit=proposal["baseline_id"],
                )
                evidence = _live_evidence_binding(
                    evidence_root=evidence_root,
                    cohort=cohort,
                    expected_source_projection=source_projection,
                    expected_source_digest=source_digest,
                    acquisition_host_metrics=sandbox_evidence[
                        "proxy_acquisition_host_metrics"
                    ],
                )
            except TrustedPrevalidationActionError as exc:
                failure_reason = "trusted_prevalidation_action_evidence_invalid"
                running["blocker"] = str(exc)
                returncode = 1
        status = "completed" if returncode == 0 and evidence else "failed"
        terminal = dict(running)
        terminal.update(
            {
                "status": status,
                "reason": (
                    "trusted_prevalidation_action_completed"
                    if status == "completed"
                    else failure_reason
                ),
                "heartbeat_at": _utc_now(),
                "finished_at": _utc_now(),
                "returncode": int(returncode),
                **sandbox_evidence,
                **evidence,
            }
        )
        terminal = _write_receipt(receipt_path, terminal)
        if heartbeat is not None:
            try:
                heartbeat(terminal)
            except Exception:  # noqa: BLE001,S110 - telemetry cannot alter verdict
                pass
        return {
            "required": True,
            "attempted": True,
            "reused": False,
            "passed": status == "completed",
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
            **terminal,
        }


def execute_state_laws_live_cohort_action(
    *,
    policy: TrustedPrevalidationActionPolicy,
    repo_root: Path,
    workspace_path: Path,
    task: Any,
    canonical_task_cid: str,
    proposal_validation: Any,
    attempt: int,
    log_path: Path,
    cancel_requested: Callable[[], bool | str] | None = None,
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
    source_projection_builder: Callable[
        ..., tuple[dict[str, str], str, dict[str, bytes]]
    ] = accepted_proposal_source_projection,
    sandbox_capability: Callable[..., tuple[Path, Path]] = _sandbox_capability,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Serialize the shared durable evidence pointer for one exact cohort."""

    task_id = str(getattr(task, "task_id", "") or "").strip()
    cohort = policy.cohort_for_task(task_id)
    if cohort is None:
        return _execute_state_laws_live_cohort_action_locked(
            policy=policy,
            repo_root=repo_root,
            workspace_path=workspace_path,
            task=task,
            canonical_task_cid=canonical_task_cid,
            proposal_validation=proposal_validation,
            attempt=attempt,
            log_path=log_path,
            cancel_requested=cancel_requested,
            heartbeat=heartbeat,
            source_projection_builder=source_projection_builder,
            sandbox_capability=sandbox_capability,
            popen=popen,
            monotonic=monotonic,
            sleep=sleep,
        )
    root = Path(repo_root).resolve()
    receipt_root = _contained(root, policy.receipt_root, field="receipt_root")
    receipt_root.mkdir(parents=True, exist_ok=True)
    lock_path = receipt_root / f"cohort_{cohort}.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise TrustedPrevalidationActionError(
            "trusted cohort acquisition lock is unsafe"
        ) from exc
    with os.fdopen(descriptor, "a+b", closefd=True) as cohort_lock:
        metadata = os.fstat(cohort_lock.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise TrustedPrevalidationActionError(
                "trusted cohort acquisition lock identity drifted"
            )
        try:
            fcntl.flock(cohort_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "required": True,
                "attempted": False,
                "passed": False,
                "reason": "trusted_prevalidation_action_cohort_lease_busy",
                "task_id": task_id,
                "cohort": cohort,
            }
        return _execute_state_laws_live_cohort_action_locked(
            policy=policy,
            repo_root=repo_root,
            workspace_path=workspace_path,
            task=task,
            canonical_task_cid=canonical_task_cid,
            proposal_validation=proposal_validation,
            attempt=attempt,
            log_path=log_path,
            cancel_requested=cancel_requested,
            heartbeat=heartbeat,
            source_projection_builder=source_projection_builder,
            sandbox_capability=sandbox_capability,
            popen=popen,
            monotonic=monotonic,
            sleep=sleep,
        )


def _wrapper_main(argv: Sequence[str] | None = None) -> int:
    """Enter the sealed network-none/Docker trusted acquisition sandbox."""

    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--cohort", required=True, choices=tuple("ABCDEFGHIJKLM"))
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument(
        "--max-parallel", required=True, type=int, choices=(MAX_PARALLEL,)
    )
    parser.add_argument("--sandbox-receipt", required=True, type=Path)
    parser.add_argument("--source-projection-manifest", required=True, type=Path)
    parser.add_argument("--source-projection-digest", required=True)
    parser.add_argument("--baseline-id", required=True)
    args = parser.parse_args(argv)
    if args.policy.is_symlink() or not args.policy.is_file():
        return EXIT_SANDBOX_UNAVAILABLE
    if _sha256_file(args.policy) != args.policy_sha256:
        return EXIT_SANDBOX_UNAVAILABLE
    sibling_root = Path(__file__).resolve().parent
    if str(sibling_root) not in sys.path:
        sys.path.insert(0, str(sibling_root))
    try:
        from state_laws_live_cohort_sandbox_v1 import (  # type: ignore[import-not-found]
            SandboxRuntimeError,
            run_sandbox_action,
        )

        return run_sandbox_action(
            policy_path=args.policy,
            policy_sha256=args.policy_sha256,
            worktree=args.worktree,
            evidence_root=args.evidence_root,
            cohort=args.cohort,
            runner=args.runner,
            max_parallel=args.max_parallel,
            sandbox_receipt_path=args.sandbox_receipt,
            source_projection_manifest_path=args.source_projection_manifest,
            source_projection_digest=args.source_projection_digest,
            baseline_commit=args.baseline_id,
        )
    except (OSError, SandboxRuntimeError) as exc:
        print(f"trusted action sandbox blocked: {exc}", file=sys.stderr)
        return EXIT_SANDBOX_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(_wrapper_main())
