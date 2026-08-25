#!/usr/bin/env python3
"""Verify the fail-closed PCCE v0.1 release workflow contract.

The default command is a local, read-only contract audit.  It proves that the
declared workflow contains the complete required job graph, immutable action
pins, bounded evidence uploads, and no error-swallowing escape hatch.  Local
contract success is intentionally distinct from release qualification.

External GitHub run, log, and ruleset evidence is checked only when explicitly
supplied.  ``--require-qualified`` fails with exit code 5 while any required
authority or qualification input is unavailable; it never turns local YAML
inspection into a CI-pass claim.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
ACCELERATOR_ROOT = SCRIPT_PATH.parents[2]
OUTER_ROOT = SCRIPT_PATH.parents[4]

TASK_ID = "PCCE-080"
LIVE_TASK_CID = "baguqeerax65k5eln5awrgysj33pt6gsatg2po2leftgrmg25m3f47ojhnnga"
BOARD_NAMESPACE = "proof-carrying-context-engine-v0.1"
SCHEMA_PREFIX = "lift_coding.proof-carrying-context-engine"
MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}.required-ci-jobs@1"
JOB_EVIDENCE_SCHEMA = f"{SCHEMA_PREFIX}.ci-job-evidence@1"
DEFAULT_WORKFLOW = ACCELERATOR_ROOT / ".github/workflows/proof-context-v0.1.yml"
DEFAULT_MANIFEST = OUTER_ROOT / "artifacts/proof_carrying_context_engine/ci/required_jobs.json"
EXIT_EVIDENCE_ERROR = 2
EXIT_NOT_QUALIFIED = 5

CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_ACTION = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
DOWNLOAD_ACTION = "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
UPLOAD_ACTION = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"

JOB_ORDER = (
    "release-contract",
    "clean-install",
    "unit-integration",
    "security",
    "benchmark-smoke",
    "receipt-seal",
    "container",
    "dependency-license",
    "release-gate",
)
UPSTREAM_JOBS = JOB_ORDER[:-1]
JOB_NAMES = {
    "release-contract": "PCCE / release contract",
    "clean-install": "PCCE / clean install",
    "unit-integration": "PCCE / unit and integration",
    "security": "PCCE / security",
    "benchmark-smoke": "PCCE / benchmark smoke",
    "receipt-seal": "PCCE / receipt and seal verification",
    "container": "PCCE / supported container",
    "dependency-license": "PCCE / dependency and license",
    "release-gate": "PCCE / required release gate",
}
JOB_RUNNERS = {
    "release-contract": "ubuntu-24.04",
    "clean-install": "ubuntu-24.04-arm",
    "unit-integration": "ubuntu-24.04-arm",
    "security": "ubuntu-24.04-arm",
    "benchmark-smoke": "ubuntu-24.04-arm",
    "receipt-seal": "ubuntu-24.04",
    "container": "ubuntu-24.04-arm",
    "dependency-license": "ubuntu-24.04",
    "release-gate": "ubuntu-24.04",
}
JOB_MARKERS = {
    "release-contract": (
        '--manifest "$PCCE_REQUIRED_JOB_MANIFEST"',
        '--evidence-root "$PCCE_EVIDENCE_ROOT"',
        "--require-qualified",
    ),
    "clean-install": (
        "scripts/proof_context/test_clean_install.py",
        "artifact_hashes.json",
        "--require-qualified",
    ),
    "unit-integration": (
        "--require-hashes",
        "evaluation.txt",
        "test/proof_context/test_v01_dependencies.py",
        "test/proof_context/test_runtime_integration.py",
        "test/proof_context/test_example_repository.py",
    ),
    "security": (
        "test/proof_context/security/test_threat_model.py",
        "test/proof_context/security/test_sandbox.py",
        "test/proof_context/security/test_adversarial_patch_and_agent.py",
        "test/proof_context/security/test_adversarial_concurrency.py",
        "test_isolation.py",
        "test_trust_admission.py",
        "--require-category security",
    ),
    "benchmark-smoke": (
        "test_configurations_ab.py",
        "test_configurations_cd.py",
        "test_metrics.py",
        "--require-category benchmark",
    ),
    "receipt-seal": ("--verify-receipts", "--require-qualified"),
    "container": (
        "docker buildx build",
        "--network none",
        "docker/proof-context/Dockerfile",
    ),
    "dependency-license": ("--audit-dependencies", "--require-qualified"),
    "release-gate": ('--aggregate-needs "$PCCE_NEEDS_JSON"',),
}

EXPECTED_GLOBAL_ENV = {
    "PCCE_ARTIFACT_MANIFEST_SHA256": (
        "b5b38995520aedd3392a205173182dcb07bc43361a5825b53639b985cb460ade"
    ),
    "PCCE_DATASETS_COMMIT": "817a665897467b44e6b6fc413d757d9ade719d4d",
    "PCCE_EVIDENCE_ARTIFACT_NAME": "proof-context-pcce080-inputs",
    "PCCE_EVIDENCE_REPOSITORY": "endomorphosis/lift_coding",
    "PCCE_KIT_COMMIT": "f591ce1404df68bc597031141e483ae6e58b4dbb",
    "PCCE_LOCK_INPUTS_SHA256": ("2042dd244262bd26522c0fd7a0ff9fff049b5c17077d0b3fc6982e47aedd91fd"),
    "PCCE_MCP_COMMIT": "0ed2b23d13371a6cae25e5f328a10152e5d1da11",
}

FORBIDDEN_TEXT_PATTERNS = {
    "continue-on-error": re.compile(r"(?m)^\s*continue-on-error\s*:"),
    "disabled-fail-fast": re.compile(r"(?m)^\s*fail-fast\s*:\s*false\s*$"),
    "error-swallow-or-true": re.compile(r"\|\|\s*(?:true|:)\b"),
    "error-swallow-semicolon": re.compile(r";\s*true\b"),
    "set-plus-e": re.compile(r"(?m)^\s*set\s+\+e\s*$"),
    "pytest-deselect": re.compile(r"(?:--deselect|--ignore|--continue-on-collection-errors)"),
    "pytest-filter": re.compile(r"(?:^|\s)-k(?:\s|=)"),
    "mutable-action-ref": re.compile(r"(?m)^\s*uses:\s*[^\s@]+@(?:v\d+|main|master)\s*$"),
}

ACTION_REFERENCE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
JOB_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    """Raised when local workflow or manifest evidence is inconsistent."""


class EvidenceUnavailable(ContractError):
    """Raised when a structurally valid gate lacks required external authority."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_cid_v1_from_sha256(sha256: str) -> str:
    if HEX_SHA256.fullmatch(sha256) is None:
        raise ContractError(f"invalid SHA-256 for raw CID: {sha256!r}")
    payload = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    cid = "b" + base64.b32encode(payload).decode("ascii").lower().rstrip("=")
    _verify_raw_cid_v1(cid, sha256)
    return cid


def _verify_raw_cid_v1(cid: str, sha256: str) -> None:
    if not isinstance(cid, str) or not cid.startswith("b") or cid != cid.lower():
        raise ContractError(f"raw CIDv1 is not canonical base32-lower: {cid!r}")
    if HEX_SHA256.fullmatch(sha256) is None:
        raise ContractError(f"invalid expected SHA-256: {sha256!r}")
    encoded = cid[1:].upper()
    encoded += "=" * ((8 - len(encoded) % 8) % 8)
    try:
        payload = base64.b32decode(encoded, casefold=False)
    except (ValueError, binascii.Error) as exc:
        raise ContractError(f"raw CIDv1 cannot be decoded: {cid!r}") from exc
    expected = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    if payload != expected:
        raise ContractError(f"raw CIDv1 does not bind expected SHA-256: {cid!r}")


def _descriptor(path: str, value: bytes) -> dict[str, Any]:
    sha256 = _sha256_bytes(value)
    return {
        "path": path,
        "sha256": sha256,
        "cid_v1_raw": _raw_cid_v1_from_sha256(sha256),
        "size": len(value),
    }


def _load_canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    if raw != _canonical_json_bytes(value):
        raise ContractError(f"JSON is not canonical sorted PCCE encoding: {path}")
    return value, raw


def _workflow_lines(path: Path) -> tuple[list[str], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read UTF-8 workflow {path}: {exc}") from exc
    if not text.endswith("\n") or "\t" in text:
        raise ContractError("workflow must be LF-terminated UTF-8 with no tabs")
    lines = text.splitlines()
    if any(line.rstrip() != line for line in lines):
        raise ContractError("workflow contains trailing whitespace")
    if any(
        re.search(pattern, text)
        for pattern in (r"(?m)^---\s*$", r"(?m)(?:^|\s)[&*][A-Za-z_][A-Za-z0-9_-]*")
    ):
        raise ContractError("workflow must not use multi-document YAML or anchors/aliases")
    return lines, raw


def _top_level_index(lines: list[str], key: str) -> int:
    needle = f"{key}:"
    matches = [index for index, line in enumerate(lines) if line == needle]
    if len(matches) != 1:
        raise ContractError(f"workflow requires exactly one top-level {needle}")
    return matches[0]


def _top_level_block(lines: list[str], key: str) -> list[str]:
    start = _top_level_index(lines, key) + 1
    end = len(lines)
    for index in range(start, len(lines)):
        line = lines[index]
        if line and not line.startswith(" ") and re.fullmatch(r"[A-Za-z0-9_-]+:", line):
            end = index
            break
    return lines[start:end]


def _simple_mapping(block: list[str], indent: int) -> dict[str, str]:
    prefix = " " * indent
    result: dict[str, str] = {}
    pattern = re.compile(rf"{re.escape(prefix)}([A-Za-z0-9_-]+):\s*(.*)")
    for line in block:
        match = pattern.fullmatch(line)
        if match is None:
            continue
        key, value = match.groups()
        if key in result:
            raise ContractError(f"duplicate workflow mapping key: {key}")
        result[key] = value.strip().strip('"')
    return result


def _job_blocks(lines: list[str]) -> dict[str, list[str]]:
    block = _top_level_block(lines, "jobs")
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(block):
        match = re.fullmatch(r"  ([a-z][a-z0-9-]*):", line)
        if match is not None:
            starts.append((index, match.group(1)))
    if len({job_id for _, job_id in starts}) != len(starts):
        raise ContractError("workflow contains duplicate job IDs")
    result: dict[str, list[str]] = {}
    for position, (start, job_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(block)
        result[job_id] = block[start + 1 : end]
    return result


def _steps(job: list[str]) -> list[list[str]]:
    starts: list[int] = []
    for index, line in enumerate(job):
        if re.fullmatch(r"      - name:\s+.+", line):
            starts.append(index)
    result: list[list[str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(job)
        result.append(job[start:end])
    return result


def _step_value(step: list[str], key: str) -> str | None:
    pattern = re.compile(rf"        {re.escape(key)}:\s*(.+)")
    values = [
        match.group(1).split(" #", maxsplit=1)[0].strip()
        for line in step
        if (match := pattern.fullmatch(line))
    ]
    if len(values) > 1:
        raise ContractError(f"step contains duplicate {key} values")
    return values[0] if values else None


def _step_name(step: list[str]) -> str:
    match = re.fullmatch(r"      - name:\s+(.+)", step[0])
    if match is None:
        raise ContractError("malformed named workflow step")
    return match.group(1)


def _run_body(step: list[str]) -> str | None:
    for index, line in enumerate(step):
        if line == "        run: |":
            body: list[str] = []
            for child in step[index + 1 :]:
                if child and not child.startswith("          "):
                    break
                body.append(child[10:] if child.startswith("          ") else "")
            return "\n".join(body).rstrip() + "\n"
    return None


def _job_value(job: list[str], key: str) -> str | None:
    pattern = re.compile(rf"    {re.escape(key)}:\s*(.+)")
    values = [match.group(1).strip() for line in job if (match := pattern.fullmatch(line))]
    if len(values) > 1:
        raise ContractError(f"job contains duplicate {key} values")
    return values[0] if values else None


def _job_needs(job: list[str]) -> list[str]:
    try:
        start = job.index("    needs:") + 1
    except ValueError:
        return []
    result: list[str] = []
    for line in job[start:]:
        match = re.fullmatch(r"      - ([a-z][a-z0-9-]*)", line)
        if match is not None:
            result.append(match.group(1))
            continue
        if line and not line.startswith("      "):
            break
    return result


def _validate_triggers(lines: list[str]) -> None:
    block = _top_level_block(lines, "on")
    triggers = {
        match.group(1)
        for line in block
        if (match := re.fullmatch(r"  ([a-z_]+):", line)) is not None
    }
    expected = {"merge_group", "pull_request", "push", "workflow_dispatch"}
    if triggers != expected:
        raise ContractError(f"workflow trigger set differs: {sorted(triggers)!r}")
    joined = "\n".join(block)
    for marker in (
        "evidence_run_id:",
        "outer_evidence_commit:",
        "required: true",
        "branches:",
        "- main",
    ):
        if marker not in joined:
            raise ContractError(f"workflow trigger input marker is absent: {marker}")


def _validate_permissions_and_environment(lines: list[str]) -> None:
    permissions = _simple_mapping(_top_level_block(lines, "permissions"), 2)
    if permissions != {"actions": "read", "contents": "read"}:
        raise ContractError(f"workflow permissions differ: {permissions!r}")
    concurrency = _simple_mapping(_top_level_block(lines, "concurrency"), 2)
    if concurrency.get("cancel-in-progress") != "false" or not concurrency.get("group"):
        raise ContractError("workflow concurrency must retain evidence-producing runs")
    environment = _simple_mapping(_top_level_block(lines, "env"), 2)
    for key, expected in EXPECTED_GLOBAL_ENV.items():
        if environment.get(key) != expected:
            raise ContractError(f"workflow immutable environment binding drifted: {key}")
    if environment.get("PCCE_EVIDENCE_RUN_ID") != "${{ github.event.inputs.evidence_run_id }}":
        raise ContractError("workflow evidence run must come from one explicit dispatch input")
    if environment.get("PCCE_OUTER_EVIDENCE_COMMIT") != (
        "${{ github.event.inputs.outer_evidence_commit }}"
    ):
        raise ContractError("workflow outer commit must come from one explicit dispatch input")


def _validate_action_step(step: list[str]) -> None:
    uses = _step_value(step, "uses")
    if uses is None:
        return
    if ACTION_REFERENCE.fullmatch(uses) is None:
        raise ContractError(f"action is not pinned to one full commit: {uses!r}")


def _validate_candidate_checkout(job_id: str, steps: list[list[str]]) -> None:
    matches = [step for step in steps if _step_name(step) == "Check out the exact candidate"]
    if len(matches) != 1 or _step_value(matches[0], "uses") != CHECKOUT_ACTION:
        raise ContractError(f"{job_id} must contain one exact candidate checkout")
    text = "\n".join(matches[0])
    for marker in (
        "fetch-depth: 1",
        "persist-credentials: false",
        "ref: ${{ github.sha }}",
    ):
        if marker not in text:
            raise ContractError(f"{job_id} candidate checkout lacks {marker!r}")


def _validate_upload(job_id: str, steps: list[list[str]]) -> None:
    uploads = [step for step in steps if _step_value(step, "uses") == UPLOAD_ACTION]
    if len(uploads) != 1:
        raise ContractError(f"{job_id} must have exactly one bounded evidence upload")
    upload = uploads[0]
    if _step_value(upload, "if") != "${{ always() }}":
        raise ContractError(f"{job_id} evidence upload must execute after failures")
    text = "\n".join(upload)
    for marker in (
        "if-no-files-found: error",
        "retention-days: 30",
        "${{ env.PCCE_JOB_EVIDENCE }}",
    ):
        if marker not in text:
            raise ContractError(f"{job_id} upload lacks {marker!r}")


def _validate_runs(job_id: str, steps: list[list[str]]) -> str:
    bodies = [body for step in steps if (body := _run_body(step)) is not None]
    if not bodies:
        raise ContractError(f"{job_id} has no fail-closed command steps")
    for body in bodies:
        nonempty = [line for line in body.splitlines() if line.strip()]
        if not nonempty or nonempty[0].strip() != "set -euo pipefail":
            raise ContractError(f"{job_id} command step does not start fail-closed")
    combined = "\n".join(bodies)
    for marker in (
        "--write-job-evidence",
        "--phase started",
        "--phase completed",
        '--head-sha "${{ github.sha }}"',
    ):
        if marker not in combined:
            raise ContractError(f"{job_id} bounded evidence marker is absent: {marker}")
    for marker in JOB_MARKERS[job_id]:
        if marker not in combined:
            raise ContractError(f"{job_id} required command marker is absent: {marker}")
    if "pip install" in combined:
        for marker in ("--no-index", "--only-binary=:all:", "--require-hashes"):
            if marker not in combined:
                raise ContractError(f"{job_id} pip invocation lacks {marker}")
        if re.search(r"(?m)pip install[^\n]*(?:\s-e\s|--editable|\s\.\s*$)", combined):
            raise ContractError(f"{job_id} attempts a source/editable installation")
    return combined


def _validate_job(job_id: str, job: list[str]) -> dict[str, Any]:
    if _job_value(job, "name") != JOB_NAMES[job_id]:
        raise ContractError(f"{job_id} check name drifted")
    if _job_value(job, "runs-on") != JOB_RUNNERS[job_id]:
        raise ContractError(f"{job_id} runner drifted")
    timeout = _job_value(job, "timeout-minutes")
    if timeout is None or not timeout.isdigit() or not 1 <= int(timeout) <= 90:
        raise ContractError(f"{job_id} timeout is absent or unbounded")
    condition = _job_value(job, "if")
    if job_id == "release-gate":
        if condition != "${{ always() }}" or _job_needs(job) != list(UPSTREAM_JOBS):
            raise ContractError("release-gate must aggregate every required upstream job")
    elif condition is not None or _job_needs(job):
        raise ContractError(f"{job_id} must not be conditional or dependent")
    steps = _steps(job)
    if not steps:
        raise ContractError(f"{job_id} has no named steps")
    for step in steps:
        _validate_action_step(step)
    _validate_candidate_checkout(job_id, steps)
    setup = [step for step in steps if _step_value(step, "uses") == SETUP_PYTHON_ACTION]
    if len(setup) != 1 or "python-version: 3.12.3" not in "\n".join(setup[0]):
        raise ContractError(f"{job_id} must provision exact CPython 3.12.3")
    downloads = [step for step in steps if _step_value(step, "uses") == DOWNLOAD_ACTION]
    if (job_id == "release-gate" and downloads) or (
        job_id != "release-gate" and len(downloads) != 1
    ):
        raise ContractError(f"{job_id} exact evidence download count drifted")
    _validate_upload(job_id, steps)
    combined = _validate_runs(job_id, steps)
    return {
        "job_id": job_id,
        "check_name": JOB_NAMES[job_id],
        "runner": JOB_RUNNERS[job_id],
        "needs": _job_needs(job),
        "command_sha256": _sha256_bytes(combined.encode("utf-8")),
        "step_count": len(steps),
    }


def validate_workflow(path: Path) -> dict[str, Any]:
    """Return the exact local workflow contract after fail-closed validation."""
    lines, raw = _workflow_lines(path)
    text = raw.decode("utf-8")
    if "name: Proof-context v0.1 required release gate" not in lines:
        raise ContractError("workflow name drifted")
    for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
        if pattern.search(text):
            raise ContractError(f"workflow contains prohibited construct: {label}")
    _validate_triggers(lines)
    _validate_permissions_and_environment(lines)
    jobs = _job_blocks(lines)
    if tuple(jobs) != JOB_ORDER:
        raise ContractError(f"required job order/set differs: {tuple(jobs)!r}")
    records = [_validate_job(job_id, jobs[job_id]) for job_id in JOB_ORDER]
    descriptor = _descriptor(
        "external/ipfs_accelerate/.github/workflows/proof-context-v0.1.yml",
        raw,
    )
    return {
        "workflow": descriptor,
        "required_job_count": len(records),
        "jobs": records,
        "error_swallow_audit": "passed",
        "skip_audit": "passed",
        "immutable_action_pin_audit": "passed",
        "bounded_evidence_upload_audit": "passed",
    }


def _validate_file_identity(record: Any, path: Path, expected_path: str) -> None:
    if not isinstance(record, dict):
        raise ContractError(f"manifest file identity is absent: {expected_path}")
    if not path.is_file():
        raise ContractError(f"manifest-bound file is absent: {path}")
    expected = _descriptor(expected_path, path.read_bytes())
    if record != expected:
        raise ContractError(f"manifest file identity drifted: {expected_path}")


def validate_manifest(path: Path, workflow_contract: dict[str, Any]) -> dict[str, Any]:
    manifest, raw = _load_canonical_object(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ContractError("required-job manifest schema is unsupported")
    if manifest.get("task_id") != TASK_ID or manifest.get("board_namespace") != BOARD_NAMESPACE:
        raise ContractError("required-job manifest task authority drifted")
    authority = manifest.get("task_authority")
    if not isinstance(authority, dict) or authority.get("live_task_cid") != LIVE_TASK_CID:
        raise ContractError("required-job manifest does not bind the live PCCE-080 task CID")
    jobs = manifest.get("required_jobs")
    if not isinstance(jobs, list) or [item.get("job_id") for item in jobs] != list(JOB_ORDER):
        raise ContractError("required-job manifest job set/order drifted")
    qualification = manifest.get("qualification")
    no_go_qualification = {
        "decision": "NO-GO",
        "external_ci_authority_available": False,
        "local_contract_verified": True,
        "release_qualified": False,
        "waivers": [],
    }
    go_qualification = {
        "decision": "GO",
        "external_ci_authority_available": True,
        "local_contract_verified": True,
        "release_qualified": True,
        "waivers": [],
    }
    if qualification not in (no_go_qualification, go_qualification):
        raise ContractError("required-job manifest qualification is not fail-closed")
    is_no_go = qualification == no_go_qualification
    for expected, observed in zip(workflow_contract["jobs"], jobs, strict=True):
        if observed.get("check_name") != expected["check_name"]:
            raise ContractError(f"manifest check name drifted: {expected['job_id']}")
        if observed.get("needs") != expected["needs"]:
            raise ContractError(f"manifest dependency graph drifted: {expected['job_id']}")
        if observed.get("required") is not True:
            raise ContractError(f"manifest job is not required: {expected['job_id']}")
        external = observed.get("external_result")
        if not isinstance(external, dict):
            raise ContractError(f"manifest external result is absent: {expected['job_id']}")
        if is_no_go:
            if external != {
                "check_run_id": None,
                "conclusion": None,
                "log_cid": None,
                "log_sha256": None,
                "status": "unavailable-not-run",
                "url": None,
            }:
                raise ContractError(f"NO-GO manifest result drifted: {expected['job_id']}")
        else:
            if (
                not isinstance(external.get("check_run_id"), int)
                or external["check_run_id"] <= 0
                or external.get("conclusion") != "success"
                or external.get("status") != "completed"
                or not str(external.get("url", "")).startswith("https://github.com/")
            ):
                raise ContractError(f"GO manifest job authority is invalid: {expected['job_id']}")
            log_sha256 = external.get("log_sha256")
            log_cid = external.get("log_cid")
            if not isinstance(log_sha256, str) or not isinstance(log_cid, str):
                raise ContractError(f"GO manifest job log identity is absent: {expected['job_id']}")
            _verify_raw_cid_v1(log_cid, log_sha256)
    identities = manifest.get("file_identities")
    if not isinstance(identities, dict):
        raise ContractError("required-job manifest file identities are absent")
    if identities.get("workflow") != workflow_contract["workflow"]:
        raise ContractError("required-job manifest workflow digest drifted")
    _validate_file_identity(
        identities.get("verifier"),
        SCRIPT_PATH,
        "external/ipfs_accelerate/scripts/proof_context/verify_release_ci.py",
    )
    test_path = ACCELERATOR_ROOT / "test/proof_context/test_release_ci_contract.py"
    _validate_file_identity(
        identities.get("contract_test"),
        test_path,
        "external/ipfs_accelerate/test/proof_context/test_release_ci_contract.py",
    )
    external_authority = manifest.get("external_authority")
    if not isinstance(external_authority, dict):
        raise ContractError("external CI authority disposition is absent")
    authority_fields = (
        "branch_ruleset_id",
        "branch_ruleset_url",
        "current_head_sha",
        "workflow_run_id",
        "workflow_run_url",
    )
    if is_no_go:
        if any(external_authority.get(key) is not None for key in authority_fields):
            raise ContractError("NO-GO manifest contains an unobserved CI authority")
        if external_authority.get("status") != "unavailable-not-observed":
            raise ContractError("external CI authority status is not explicit")
    else:
        if any(not external_authority.get(key) for key in authority_fields):
            raise ContractError("GO manifest external CI authority is incomplete")
        if external_authority.get("status") != "observed-and-verified":
            raise ContractError("GO manifest external CI authority status is invalid")
        if not HEX_COMMIT.fullmatch(str(external_authority.get("current_head_sha", ""))):
            raise ContractError("GO manifest current-head identity is invalid")
        if not isinstance(external_authority.get("dependency_license_findings"), list):
            raise ContractError("GO manifest dependency/license findings are absent")
    descriptor = _descriptor(
        "artifacts/proof_carrying_context_engine/ci/required_jobs.json",
        raw,
    )
    return {"manifest": descriptor, "value": manifest}


def _qualification_paths(evidence_root: Path, category: str) -> tuple[Path, ...]:
    artifact_root = evidence_root / "artifacts/proof_carrying_context_engine"
    if category == "benchmark":
        return (artifact_root / "benchmark/qualification.json",)
    if category == "security":
        return (artifact_root / "security/qualification.json",)
    raise ContractError(f"unsupported qualification category: {category}")


def _require_category(evidence_root: Path, category: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in _qualification_paths(evidence_root, category):
        try:
            value, raw = _load_canonical_object(path)
        except ContractError as exc:
            raise EvidenceUnavailable(f"{category} qualification is unavailable: {exc}") from exc
        decision = str(value.get("decision", value.get("status", ""))).upper()
        if (
            decision not in {"GO", "QUALIFIED", "PASSED"}
            or value.get("release_qualified") is not True
        ):
            raise EvidenceUnavailable(f"{category} qualification is not release-qualified")
        records.append(_descriptor(path.as_posix(), raw))
    return {"category": category, "records": records, "qualified": True}


def _verify_receipts(evidence_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    required = manifest.get("required_predecessor_receipts")
    if not isinstance(required, list) or required != ["PCCE-068", "PCCE-076"]:
        raise ContractError("manifest predecessor receipt set drifted")
    receipt_root = evidence_root / "artifacts/proof_carrying_context_engine/receipts"
    records: list[dict[str, Any]] = []
    for task_id in required:
        path = receipt_root / f"{task_id}.json"
        try:
            value, raw = _load_canonical_object(path)
        except ContractError as exc:
            raise EvidenceUnavailable(f"required receipt {task_id} is unavailable: {exc}") from exc
        if value.get("task_id") != task_id or value.get("status") != "completed":
            raise ContractError(f"required receipt identity/status drifted: {task_id}")
        if not value.get("artifact_identity"):
            raise ContractError(f"required receipt lacks artifact identity: {task_id}")
        records.append(_descriptor(path.as_posix(), raw))
    return {"receipt_count": len(records), "records": records, "verified": True}


def _audit_dependencies(evidence_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifact_root = evidence_root / "artifacts/proof_carrying_context_engine"
    sbom_path = artifact_root / "environment/sbom.spdx.json"
    locks_path = artifact_root / "environment/dependency_locks.json"
    try:
        sbom, sbom_raw = _load_canonical_object(sbom_path)
        locks, locks_raw = _load_canonical_object(locks_path)
    except ContractError as exc:
        raise EvidenceUnavailable(f"dependency evidence is unavailable: {exc}") from exc
    if not str(sbom.get("spdxVersion", "")).startswith("SPDX-"):
        raise ContractError("SBOM is not an SPDX document")
    if not isinstance(locks.get("profiles"), dict):
        raise ContractError("dependency lock manifest lacks profiles")
    external = manifest.get("external_authority", {})
    findings = external.get("dependency_license_findings")
    if findings is None:
        raise EvidenceUnavailable("external dependency/license findings were not observed")
    if not isinstance(findings, list):
        raise ContractError("dependency/license findings must be a bounded list")
    return {
        "sbom": _descriptor(sbom_path.as_posix(), sbom_raw),
        "locks": _descriptor(locks_path.as_posix(), locks_raw),
        "finding_count": len(findings),
    }


def _require_external_qualification(manifest: dict[str, Any]) -> dict[str, Any]:
    qualification = manifest.get("qualification", {})
    external = manifest.get("external_authority", {})
    if qualification.get("decision") != "GO" or qualification.get("release_qualified") is not True:
        raise EvidenceUnavailable("current-head CI qualification is NO-GO")
    required = (
        "branch_ruleset_id",
        "branch_ruleset_url",
        "current_head_sha",
        "workflow_run_id",
        "workflow_run_url",
    )
    if any(not external.get(key) for key in required):
        raise EvidenceUnavailable("current-head run or required-check ruleset authority is absent")
    for job in manifest.get("required_jobs", []):
        result = job.get("external_result", {})
        if result.get("conclusion") != "success":
            raise EvidenceUnavailable(f"required job is not successful: {job.get('job_id')}")
        if not result.get("url") or not result.get("log_sha256") or not result.get("log_cid"):
            raise EvidenceUnavailable(
                f"required job log authority is incomplete: {job.get('job_id')}"
            )
        _verify_raw_cid_v1(result["log_cid"], result["log_sha256"])
    return {"qualified": True, "workflow_run_id": external["workflow_run_id"]}


def _check_inputs(evidence_run_id: str | None, outer_commit: str | None) -> dict[str, Any]:
    if evidence_run_id is None or re.fullmatch(r"[1-9][0-9]*", evidence_run_id) is None:
        raise EvidenceUnavailable("one explicit numeric evidence workflow run ID is required")
    if outer_commit is None or HEX_COMMIT.fullmatch(outer_commit) is None:
        raise EvidenceUnavailable("one exact 40-hex outer evidence commit is required")
    return {
        "evidence_run_id": evidence_run_id,
        "outer_evidence_commit": outer_commit,
        "input_shape": "passed",
        "authority_observed": False,
    }


def _write_job_evidence(
    path: Path, job_id: str | None, phase: str | None, head_sha: str | None
) -> dict[str, Any]:
    if job_id not in JOB_ORDER:
        raise ContractError(f"job evidence has unknown job ID: {job_id!r}")
    if phase not in {"started", "completed"}:
        raise ContractError(f"job evidence phase is invalid: {phase!r}")
    if head_sha is None or HEX_COMMIT.fullmatch(head_sha) is None:
        raise ContractError("job evidence requires one exact 40-hex head SHA")
    if path.is_symlink() or path.parent.is_symlink():
        raise ContractError("job evidence path must not be a symlink")
    value = {
        "schema": JOB_EVIDENCE_SCHEMA,
        "task_id": TASK_ID,
        "job_id": job_id,
        "check_name": JOB_NAMES[job_id],
        "head_sha": head_sha,
        "phase": phase,
        "local_command_sequence_completed": phase == "completed",
        "release_qualification_claimed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))
    return value


def _aggregate_needs(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"aggregate needs JSON is malformed: {exc}") from exc
    if not isinstance(value, dict) or set(value) != set(UPSTREAM_JOBS):
        raise ContractError("aggregate needs set differs from all eight required upstream jobs")
    results: dict[str, str] = {}
    for job_id in UPSTREAM_JOBS:
        record = value.get(job_id)
        if not isinstance(record, dict) or not isinstance(record.get("result"), str):
            raise ContractError(f"aggregate needs result is malformed: {job_id}")
        results[job_id] = record["result"]
    failed = sorted(job_id for job_id, result in results.items() if result != "success")
    if failed:
        raise EvidenceUnavailable(f"required upstream jobs are not successful: {failed}")
    return {"required_upstream_jobs": len(results), "results": results, "qualified": True}


def _resolve(path: str | None, default: Path) -> Path:
    value = Path(path) if path else default
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", metavar="PATH")
    parser.add_argument("--manifest", metavar="PATH")
    parser.add_argument("--evidence-root", metavar="PATH")
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--require-category", choices=("benchmark", "security"))
    parser.add_argument("--verify-receipts", action="store_true")
    parser.add_argument("--audit-dependencies", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--evidence-run-id")
    parser.add_argument("--outer-commit")
    parser.add_argument("--write-job-evidence", metavar="PATH")
    parser.add_argument("--job-id")
    parser.add_argument("--phase", choices=("started", "completed"))
    parser.add_argument("--head-sha")
    parser.add_argument("--aggregate-needs", metavar="JSON")
    return parser


def _default_audit(args: argparse.Namespace) -> dict[str, Any]:
    workflow_path = _resolve(args.workflow, DEFAULT_WORKFLOW)
    manifest_path = _resolve(args.manifest, DEFAULT_MANIFEST)
    workflow = validate_workflow(workflow_path)
    manifest_record = validate_manifest(manifest_path, workflow)
    manifest = manifest_record["value"]
    result: dict[str, Any] = {
        "task_id": TASK_ID,
        "live_task_cid": LIVE_TASK_CID,
        "local_contract": "passed",
        "required_job_count": len(JOB_ORDER),
        "workflow": workflow["workflow"],
        "required_job_manifest": manifest_record["manifest"],
        "qualification": manifest["qualification"],
    }
    if args.evidence_root:
        evidence_root = _resolve(args.evidence_root, Path.cwd())
        if args.require_category:
            result[args.require_category] = _require_category(evidence_root, args.require_category)
        if args.verify_receipts:
            result["receipts"] = _verify_receipts(evidence_root, manifest)
        if args.audit_dependencies:
            result["dependencies"] = _audit_dependencies(evidence_root, manifest)
    elif args.require_category or args.verify_receipts or args.audit_dependencies:
        raise EvidenceUnavailable("the requested external evidence root was not supplied")
    if args.require_qualified:
        result["external_qualification"] = _require_external_qualification(manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        operational_modes = sum(
            value is not None or flag
            for value, flag in (
                (args.write_job_evidence, False),
                (args.aggregate_needs, False),
                (None, args.check_inputs),
            )
        )
        if operational_modes > 1:
            raise ContractError("operational verifier modes are mutually exclusive")
        if args.write_job_evidence is not None:
            result = _write_job_evidence(
                _resolve(args.write_job_evidence, Path.cwd()),
                args.job_id,
                args.phase,
                args.head_sha,
            )
        elif args.aggregate_needs is not None:
            result = _aggregate_needs(args.aggregate_needs)
        elif args.check_inputs:
            result = _check_inputs(args.evidence_run_id, args.outer_commit)
        else:
            result = _default_audit(args)
    except EvidenceUnavailable as exc:
        print(f"PCCE-080 CI not qualified: {exc}", file=sys.stderr)
        return EXIT_NOT_QUALIFIED
    except ContractError as exc:
        print(f"PCCE-080 CI contract error: {exc}", file=sys.stderr)
        return EXIT_EVIDENCE_ERROR
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
