#!/usr/bin/env python3
"""Materialize or verify the append-only LGCVF single-user R&D closeout.

The predecessor closeout remains a truthful partial/non-production result.
This layer verifies that protected predecessor, authenticates the two R&D
authority dispositions, validates S003 against the exact qualification suite,
and derives only task implementation completion.  Objective, release, and
production authority remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_successor_resolution import (
    DERIVED_STATES,
    EXPECTED_DISPOSITIONS,
    EXPECTED_TASK_IDS,
    LgcvfSuccessorResolutionError,
)
from scripts.resolve_lgcvf_r_and_d_successors import (
    BENCHMARK_PATH,
    EXTERNAL_RECEIPT_PATH,
    PLAN_PATH,
    PREDECESSOR_PATH,
    PRODUCTION_RECEIPT_PATH,
    QUALIFICATION_PATH,
    RELEASE_PATH,
    RESOLUTION_PATH,
    ResolutionCommandError,
    _admit_output,
    _write_guarded,
    current_source_revisions,
    load_trust_policy,
    verify_current_successor_resolution,
)
from scripts.resolve_lgcvf_r_and_d_successors import (
    _load_object as _load_strict_object,
)

DATA_DIR: Final[Path] = RESOLUTION_PATH.parent
OUTPUT_PATH: Final[Path] = DATA_DIR / "r_and_d_terminal_closeout.json"
TRUST_PATH: Final[Path] = ROOT / "config/lgcvf_r_and_d_authority_trust.json"
PUBLIC_KEY_PATH: Final[Path] = ROOT / "config/lgcvf_r_and_d_authority_public_key.pem"
IMPLEMENTATION_PATH: Final[Path] = (
    ROOT
    / "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md"
)
PREDECESSOR_VALIDATOR_PATH: Final[Path] = (
    ROOT
    / "scripts/validate_logic_governed_compositional_verification_fabric_closeout.py"
)
PREDECESSOR_TIMEOUT_SECONDS: Final[int] = 4_200
TERMINAL_SCHEMA: Final[str] = "lgcvf-r-and-d-terminal-closeout@1"
MANDATORY_LIMITATIONS: Final[tuple[str, ...]] = (
    "Single-user self-verification is not independent third-party qualification.",
    "The benchmark remains partial because warm-cache model-call displacement was not evaluated.",
    "Production authorization is explicitly declined for this R&D-only scope.",
)
_CID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^baguqeera[a-z2-7]{52}$")
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")


class TerminalCloseoutError(RuntimeError):
    """The terminal closeout is absent, stale, malformed, or authority-raising."""


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    return _load_strict_object(path, label=label)


def _sha256_file(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TerminalCloseoutError(f"closeout input is unreadable: {path}") from exc


def _validate_predecessor_closeout() -> dict[str, Any]:
    try:
        resolved = PREDECESSOR_VALIDATOR_PATH.resolve(strict=True)
    except OSError as exc:
        raise TerminalCloseoutError(
            f"predecessor validator is unavailable: {exc}"
        ) from exc
    if resolved != PREDECESSOR_VALIDATOR_PATH.absolute() or not resolved.is_file():
        raise TerminalCloseoutError(
            "predecessor validator is not an exact regular file"
        )
    try:
        completed = subprocess.run(
            (sys.executable, str(resolved), "implementation", "--check"),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=PREDECESSOR_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TerminalCloseoutError(
            f"predecessor protected replay failed: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2_000:]
        raise TerminalCloseoutError(
            f"predecessor closeout rejected current evidence: {detail}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TerminalCloseoutError(
            "predecessor validator emitted invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise TerminalCloseoutError("predecessor validation root is not an object")
    _require_predecessor_validation(result)
    return result


def _require_predecessor_validation(value: Mapping[str, Any]) -> None:
    expected_states = {
        "valid": True,
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    if any(value.get(field) is not wanted for field, wanted in expected_states.items()):
        raise TerminalCloseoutError("predecessor closeout state differs")
    _predecessor_authority_cid(value)


def _predecessor_authority_cid(value: Mapping[str, Any]) -> str:
    """Bind the stable predecessor authority projection, not fresh replay IDs."""

    fields = (
        "schema",
        "valid",
        "report_sha256",
        "successor_tasks_cid",
        "benchmark_cid",
        "benchmark_authority_cid",
        "qualification_cid",
        "qualification_authority_cid",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    projection = {field: value.get(field) for field in fields}
    if any(item is None for item in projection.values()):
        raise TerminalCloseoutError(
            "predecessor stable authority projection is incomplete"
        )
    return content_identity(projection)


def _input_snapshots() -> dict[Path, bytes]:
    paths = (
        PLAN_PATH,
        QUALIFICATION_PATH,
        BENCHMARK_PATH,
        PREDECESSOR_PATH,
        EXTERNAL_RECEIPT_PATH,
        PRODUCTION_RECEIPT_PATH,
        RESOLUTION_PATH,
        RELEASE_PATH,
        IMPLEMENTATION_PATH,
        TRUST_PATH,
        PUBLIC_KEY_PATH,
    )
    snapshots: dict[Path, bytes] = {}
    for path in paths:
        try:
            snapshots[path] = path.read_bytes()
        except OSError as exc:
            raise TerminalCloseoutError(
                f"terminal input is unreadable: {path}"
            ) from exc
    return snapshots


def _require_snapshots_unchanged(snapshots: Mapping[Path, bytes]) -> None:
    for path, expected in snapshots.items():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise TerminalCloseoutError(f"terminal input disappeared: {path}") from exc
        if observed != expected:
            raise TerminalCloseoutError(f"terminal input changed during replay: {path}")


def build_terminal_closeout(
    *,
    predecessor_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the exact current terminal R&D closeout object."""

    snapshots = _input_snapshots()
    source_before = current_source_revisions().to_dict()
    validation = (
        _validate_predecessor_closeout()
        if predecessor_validation is None
        else dict(predecessor_validation)
    )
    _require_predecessor_validation(validation)
    successor_check = verify_current_successor_resolution()
    predecessor = _load_object(PREDECESSOR_PATH, label="successor predecessor")
    qualification = _load_object(QUALIFICATION_PATH, label="qualification result")
    benchmark = _load_object(BENCHMARK_PATH, label="benchmark result")
    external = _load_object(EXTERNAL_RECEIPT_PATH, label="external R&D receipt")
    production = _load_object(
        PRODUCTION_RECEIPT_PATH, label="production-declined receipt"
    )
    resolution = _load_object(RESOLUTION_PATH, label="successor resolution")
    trust = load_trust_policy()
    sources = current_source_revisions().to_dict()
    if sources != source_before:
        raise TerminalCloseoutError("semantic source roots changed during replay")
    if benchmark.get("overall_disposition") != "partial":
        raise TerminalCloseoutError(
            "terminal R&D closeout requires the truthful partial benchmark"
        )
    tasks = resolution.get("tasks")
    if not isinstance(tasks, list):
        raise TerminalCloseoutError("successor resolution tasks are absent")
    task_records: list[dict[str, Any]] = []
    for task_id, raw in zip(EXPECTED_TASK_IDS, tasks, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != task_id:
            raise TerminalCloseoutError("successor resolution task order differs")
        evidence = raw.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TerminalCloseoutError(f"{task_id} resolution evidence is absent")
        record = {
            "task_id": task_id,
            "disposition": EXPECTED_DISPOSITIONS[task_id],
            "predecessor_task_cid": raw.get("predecessor_task_cid"),
            "task_resolution_cid": raw.get("task_resolution_cid"),
            "evidence_cid": evidence.get("evidence_cid"),
        }
        task_records.append(record)
    closeout: dict[str, Any] = {
        "schema": TERMINAL_SCHEMA,
        "authority_scope": "research_and_development_only",
        "trust_model": "self_signed_single_user_r_and_d",
        "trust_key_id": trust.key_id,
        "verifier_identity": trust.identity,
        "verifier_role": trust.role,
        "third_party_independence_claimed": False,
        "plan_cid": predecessor.get("plan_cid"),
        "qualification_result_cid": qualification.get("result_cid"),
        "qualification_checkout_fingerprint_cid": qualification.get(
            "checkout_fingerprint_cid"
        ),
        "benchmark_report_cid": benchmark.get("report_cid"),
        "predecessor_release_report_sha256": _sha256_file(RELEASE_PATH),
        "predecessor_implementation_report_sha256": _sha256_file(IMPLEMENTATION_PATH),
        "predecessor_implementation_authority_cid": _predecessor_authority_cid(
            validation
        ),
        "predecessor_successor_tasks_cid": predecessor.get("successor_tasks_cid"),
        "external_qualification_receipt_cid": external.get("receipt_cid"),
        "production_authorization_receipt_cid": production.get("receipt_cid"),
        "successor_resolution_cid": resolution.get("resolution_cid"),
        "successor_resolution_check_cid": successor_check.get("check_cid"),
        "source_revisions": sources,
        "resolved_tasks": task_records,
        "task_implementation_complete": True,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "limitations": list(MANDATORY_LIMITATIONS),
    }
    closeout["closeout_cid"] = content_identity(closeout)
    _require_snapshots_unchanged(snapshots)
    validate_terminal_closeout(closeout, expected=closeout)
    return closeout


def validate_terminal_closeout(
    value: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None
) -> str:
    closeout = dict(value)
    expected_fields = {
        "schema",
        "authority_scope",
        "trust_model",
        "trust_key_id",
        "verifier_identity",
        "verifier_role",
        "third_party_independence_claimed",
        "plan_cid",
        "qualification_result_cid",
        "qualification_checkout_fingerprint_cid",
        "benchmark_report_cid",
        "predecessor_release_report_sha256",
        "predecessor_implementation_report_sha256",
        "predecessor_implementation_authority_cid",
        "predecessor_successor_tasks_cid",
        "external_qualification_receipt_cid",
        "production_authorization_receipt_cid",
        "successor_resolution_cid",
        "successor_resolution_check_cid",
        "source_revisions",
        "resolved_tasks",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "limitations",
        "closeout_cid",
    }
    if set(closeout) != expected_fields:
        raise TerminalCloseoutError("terminal closeout fields differ")
    if closeout.get("schema") != TERMINAL_SCHEMA:
        raise TerminalCloseoutError("terminal closeout schema differs")
    body = {key: item for key, item in closeout.items() if key != "closeout_cid"}
    if closeout.get("closeout_cid") != content_identity(body):
        raise TerminalCloseoutError("terminal closeout content identity differs")
    exact_states = {
        "task_implementation_complete": True,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "third_party_independence_claimed": False,
    }
    if any(closeout.get(field) is not wanted for field, wanted in exact_states.items()):
        raise TerminalCloseoutError("terminal closeout raises or loses authority state")
    if closeout.get("limitations") != list(MANDATORY_LIMITATIONS):
        raise TerminalCloseoutError("terminal closeout limitations differ")
    if (
        closeout.get("authority_scope") != "research_and_development_only"
        or closeout.get("trust_model") != "self_signed_single_user_r_and_d"
    ):
        raise TerminalCloseoutError("terminal closeout trust scope differs")
    cid_fields = (
        "trust_key_id",
        "plan_cid",
        "qualification_result_cid",
        "qualification_checkout_fingerprint_cid",
        "benchmark_report_cid",
        "predecessor_implementation_authority_cid",
        "predecessor_successor_tasks_cid",
        "external_qualification_receipt_cid",
        "production_authorization_receipt_cid",
        "successor_resolution_cid",
        "successor_resolution_check_cid",
        "closeout_cid",
    )
    if any(
        not isinstance(closeout.get(field), str)
        or _CID_PATTERN.fullmatch(str(closeout[field])) is None
        for field in cid_fields
    ):
        raise TerminalCloseoutError("terminal closeout CID binding differs")
    if any(
        not isinstance(closeout.get(field), str)
        or _SHA256_PATTERN.fullmatch(str(closeout[field])) is None
        for field in (
            "predecessor_release_report_sha256",
            "predecessor_implementation_report_sha256",
        )
    ):
        raise TerminalCloseoutError("terminal closeout report hash differs")
    source_revisions = closeout.get("source_revisions")
    if not isinstance(source_revisions, Mapping):
        raise TerminalCloseoutError("terminal closeout source revisions are absent")
    accelerator = source_revisions.get("ipfs_accelerate_py")
    datasets = source_revisions.get("ipfs_datasets_py")
    if (
        not isinstance(accelerator, Mapping)
        or set(accelerator) != {"head", "tree"}
        or not isinstance(datasets, Mapping)
        or set(datasets) != {"head", "tree", "gitlink"}
        or any(
            not isinstance(value, str) or _GIT_PATTERN.fullmatch(value) is None
            for value in (*accelerator.values(), *datasets.values())
        )
        or datasets.get("head") != datasets.get("gitlink")
    ):
        raise TerminalCloseoutError("terminal closeout source revisions differ")
    resolved = closeout.get("resolved_tasks")
    if not isinstance(resolved, list) or len(resolved) != len(EXPECTED_TASK_IDS):
        raise TerminalCloseoutError(
            "terminal closeout resolved task population differs"
        )
    expected_task_fields = {
        "task_id",
        "disposition",
        "predecessor_task_cid",
        "task_resolution_cid",
        "evidence_cid",
    }
    for task_id, task in zip(EXPECTED_TASK_IDS, resolved, strict=True):
        if (
            not isinstance(task, Mapping)
            or set(task) != expected_task_fields
            or task.get("task_id") != task_id
            or task.get("disposition") != EXPECTED_DISPOSITIONS[task_id]
            or any(
                not isinstance(task.get(field), str)
                or _CID_PATTERN.fullmatch(str(task[field])) is None
                for field in (
                    "predecessor_task_cid",
                    "task_resolution_cid",
                    "evidence_cid",
                )
            )
        ):
            raise TerminalCloseoutError(f"terminal closeout task differs: {task_id}")
    if expected is not None and closeout != dict(expected):
        raise TerminalCloseoutError(
            "terminal closeout differs from current reconstruction"
        )
    return str(closeout["closeout_cid"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        reconstructed = build_terminal_closeout()
        if args.check:
            stored = _load_object(OUTPUT_PATH, label="terminal R&D closeout")
            closeout_cid = validate_terminal_closeout(stored, expected=reconstructed)
        else:
            admitted = _admit_output(OUTPUT_PATH, reconstructed)
            _write_guarded(
                OUTPUT_PATH,
                reconstructed,
                admitted_previous=admitted,
            )
            closeout_cid = str(reconstructed["closeout_cid"])
        result = {
            "schema": "lgcvf-r-and-d-terminal-closeout-validation@1",
            "valid": True,
            "closeout_cid": closeout_cid,
            "resolved_task_ids": list(EXPECTED_TASK_IDS),
            **DERIVED_STATES,
            "test_qualification_complete": True,
        }
        result["validation_cid"] = content_identity(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
        TerminalCloseoutError,
        ResolutionCommandError,
        LgcvfSuccessorResolutionError,
    ) as exc:
        print(
            json.dumps(
                {"valid": False, "error": type(exc).__name__, "reason": str(exc)}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
