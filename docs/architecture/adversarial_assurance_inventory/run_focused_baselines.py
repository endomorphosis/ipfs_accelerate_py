#!/usr/bin/env python3
"""Protected AAE focused-baseline runner.

Supports only ``--current-tree`` execution and ``--verify-bundle`` read-only
verification. Receipts are written to one of the two reviewed output roots.
Canonical identities are recomputed from the datasets CID authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RUNNER_ID = "protected-aae-focused-baseline-runner@1"
RECEIPT_SCHEMA = "ipfs_accelerate_py/adversarial-assurance/focused-baseline-receipt@1"
REPORT_SCHEMA = "ipfs_accelerate_py/adversarial-assurance/focused-baseline-verification@1"
INVENTORY_REL = "docs/architecture/adversarial_assurance_inventory"
BASELINE_RECEIPTS_REL = f"{INVENTORY_REL}/baseline_receipts"
PREREQUISITE_EVIDENCE_REL = f"{INVENTORY_REL}/prerequisite_evidence"
ALLOWED_OUTPUT_ROOTS = (BASELINE_RECEIPTS_REL, PREREQUISITE_EVIDENCE_REL)
BOUNDED_LOG_BYTES = 64 * 1024
PRODUCTION_CREDENTIAL_MARKERS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "ACCESS_KEY",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PYTEST_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|xfailed|xpassed|error|errors)"
)
PYTEST_OVERRIDES = (
    "--override-ini=addopts=--import-mode=importlib",
    "--override-ini=log_cli=false",
    "--color=no",
    "-q",
    "--tb=line",
)
RECEIPT_FIELDS = (
    "schema",
    "runner_id",
    "repository",
    "repository_state_root",
    "started_at",
    "finished_at",
    "duration_ns",
    "command_argv",
    "returncode",
    "terminal_status",
    "passed",
    "failed",
    "skipped",
    "environment_identity",
    "dependency_lock_identity",
    "bounded_log_digest",
    "network_access",
    "production_credentials_available",
)
def _pytest_argv(*targets: str) -> tuple[str, ...]:
    return ("python3", "-m", "pytest", *PYTEST_OVERRIDES, *targets)


SUITE_SPECS = (
    {
        "name": "datasets",
        "binding": "datasets_baseline",
        "gitlink": "ipfs_datasets_py",
        "argv": _pytest_argv(
            "ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_index",
            "ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_state",
            "ipfs_datasets_py/tests/unit/logic/software_contracts/test_content_identity.py",
        ),
    },
    {
        "name": "accelerate",
        "binding": "accelerate_baseline",
        "gitlink": None,
        "argv": _pytest_argv(
            "test/api/test_agent_supervisor_incremental_verification_planner.py",
            "test/api/test_agent_supervisor_verification_receipt_cache.py",
            "test/api/test_agent_supervisor_verification_executor.py",
            "test/api/test_agent_supervisor_verification_selection.py",
            "test/api/semantic_governor",
        ),
    },
    {
        "name": "ipfs_kit_py",
        "binding": "ipfs_kit_py_baseline",
        "gitlink": "ipfs_kit_py",
        "argv": _pytest_argv(
            "ipfs_kit_py/tests/test_coordination_storage.py",
            "ipfs_kit_py/tests/test_semantic_state_root_contracts.py",
            "ipfs_kit_py/tests/test_semantic_state_root_cas.py",
            "ipfs_kit_py/tests/test_semantic_state_root_adapter.py",
            "ipfs_kit_py/tests/test_semantic_state_root_recovery.py",
            "ipfs_kit_py/tests/test_semantic_state_root_acceptance.py",
            "ipfs_kit_py/tests/test_semantic_state_root_performance.py",
            "ipfs_kit_py/tests/test_semantic_state_root_import_safety.py",
            "ipfs_kit_py/tests/semantic_governor_store",
            "ipfs_kit_py/tests/test_proof_certificate_store.py",
        ),
    },
    {
        "name": "mcp_plus_plus",
        "binding": "mcp_plus_plus_baseline",
        "gitlink": "ipfs_accelerate_py/mcplusplus",
        "argv": _pytest_argv(
            "test/api/semantic_state/test_wire.py",
            "ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py",
            "ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py",
            "ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py",
            "ipfs_accelerate_py/mcp/tests/test_profile_g_transport.py",
            "ipfs_accelerate_py/mcplusplus/tests-py/integration/test_conformance_vectors.py",
        ),
    },
)


class BaselineRunnerError(RuntimeError):
    """Fail-closed runner error."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise BaselineRunnerError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.returncode}"
        )
    return result.stdout.strip()


def _cid_for_obj(root: Path, value: object, *, noun: str) -> str:
    module_root = root / "ipfs_datasets_py"
    inserted = False
    try:
        if not module_root.is_dir():
            raise FileNotFoundError(module_root)
        module_text = str(module_root)
        if module_text not in sys.path:
            sys.path.insert(0, module_text)
            inserted = True
        module = importlib.import_module(
            "ipfs_datasets_py.logic.software_contracts.content"
        )
        authority_path = Path(str(module.__file__ or "")).resolve()
        if module_root.resolve() not in authority_path.parents:
            raise BaselineRunnerError(
                "content identity was imported from another tree"
            )
        identity = str(module.cid_for_obj(value))
    except BaselineRunnerError:
        raise
    except Exception as exc:
        raise BaselineRunnerError(
            f"{noun} canonical identity could not be computed: {type(exc).__name__}"
        ) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(str(module_root))
            except ValueError:
                pass
    if re.fullmatch(r"b[a-z2-7]{20,}", identity) is None:
        raise BaselineRunnerError(f"{noun} canonical identity is not a CID")
    return identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _bounded_log_digest(text: str) -> str:
    payload = text.encode("utf-8", "replace")[:BOUNDED_LOG_BYTES]
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _production_credentials_present(environment: Mapping[str, str]) -> bool:
    for name, value in environment.items():
        upper = name.upper()
        if any(marker in upper for marker in PRODUCTION_CREDENTIAL_MARKERS) and value:
            return True
    return False


def _child_environment(root: Path) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(marker in upper for marker in PRODUCTION_CREDENTIAL_MARKERS):
            continue
        if upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY"}:
            continue
        cleaned[name] = value
    pythonpath = [
        str(root / "ipfs_kit_py"),
        str(root / "ipfs_datasets_py"),
        str(root),
    ]
    existing = cleaned.get("PYTHONPATH", "")
    cleaned["PYTHONPATH"] = os.pathsep.join(
        pythonpath + ([existing] if existing else [])
    )
    cleaned["NO_NETWORK"] = "1"
    return cleaned


def _parse_pytest_counts(output: str) -> tuple[int, int, int]:
    passed = failed = skipped = 0
    found = False
    cleaned = ANSI_RE.sub("", output)
    for match in PYTEST_SUMMARY_RE.finditer(cleaned):
        found = True
        count = int(match.group("count"))
        kind = match.group("kind")
        if kind == "passed":
            passed = count
        elif kind == "failed":
            failed = count
        elif kind == "skipped":
            skipped = count
        elif kind in {"error", "errors"}:
            failed += count
    if not found:
        excerpt = cleaned.strip().replace("\n", " ")[-400:]
        raise BaselineRunnerError(
            "pytest did not emit a closed summary"
            + (f": {excerpt}" if excerpt else "")
        )
    return passed, failed, skipped


def _state_root(root: Path, gitlink: str | None) -> str:
    if gitlink:
        return _git(root, "rev-parse", f"HEAD:{gitlink}").lower()
    return _git(root, "rev-parse", "HEAD").lower()


def _environment_identity(root: Path, environment: Mapping[str, str]) -> str:
    return _cid_for_obj(
        root,
        {
            "schema": "ipfs_accelerate_py/adversarial-assurance/environment-identity@1",
            "python": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "platform": sys.platform,
            "cwd": str(root),
            "network_access": "disabled",
            "allowlisted_env": {
                name: "set"
                for name in sorted(environment)
                if name
                in {
                    "PYTHONPATH",
                    "NO_NETWORK",
                    "PYTHONDONTWRITEBYTECODE",
                    "TZ",
                    "LANG",
                    "LC_ALL",
                }
            },
        },
        noun="environment identity",
    )


def _dependency_lock_identity(root: Path) -> str:
    return _cid_for_obj(
        root,
        {
            "schema": "ipfs_accelerate_py/adversarial-assurance/dependency-lock-identity@1",
            "accelerate": _git(root, "rev-parse", "HEAD").lower(),
            "ipfs_datasets_py": _git(root, "rev-parse", "HEAD:ipfs_datasets_py").lower(),
            "ipfs_kit_py": _git(root, "rev-parse", "HEAD:ipfs_kit_py").lower(),
            "mcp_plus_plus": _git(
                root, "rev-parse", "HEAD:ipfs_accelerate_py/mcplusplus"
            ).lower(),
        },
        noun="dependency lock identity",
    )


def _resolve_output_dir(root: Path, raw: str | None, *, default: str) -> Path:
    relative = (raw or default).strip().replace("\\", "/")
    while relative.startswith("./"):
        relative = relative[2:]
    if relative not in ALLOWED_OUTPUT_ROOTS:
        raise BaselineRunnerError(
            "output directory is not one of the two reviewed AAE roots"
        )
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise BaselineRunnerError("output directory escaped the repository")
    return path


def _receipt_path(output_dir: Path, binding: str) -> Path:
    return output_dir / f"{binding}.json"


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineRunnerError(f"{path} is not readable JSON: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise BaselineRunnerError(f"{path} is not a JSON object")
    return payload


def _validate_receipt(
    payload: Mapping[str, object],
    *,
    name: str,
    expected_state: str,
    expected_argv: tuple[str, ...],
    require_green: bool,
) -> list[str]:
    errors: list[str] = []
    if tuple(payload) != RECEIPT_FIELDS:
        errors.append(f"{name} baseline fields differ")
    if payload.get("schema") != RECEIPT_SCHEMA:
        errors.append(f"{name} baseline schema differs")
    if payload.get("runner_id") != RUNNER_ID:
        errors.append(f"{name} baseline runner differs")
    if payload.get("repository") != name:
        errors.append(f"{name} baseline repository identity differs")
    if payload.get("repository_state_root") != expected_state:
        errors.append(f"{name} baseline repository-state root differs")
    timestamps: list[datetime] = []
    for field in ("started_at", "finished_at"):
        raw = str(payload.get(field) or "")
        try:
            timestamps.append(_parse_utc(raw))
        except ValueError:
            errors.append(f"{name} baseline {field} is not canonical UTC")
    if len(timestamps) == 2 and timestamps[1] < timestamps[0]:
        errors.append(f"{name} baseline time interval is reversed")
    if not isinstance(payload.get("duration_ns"), int) or int(payload.get("duration_ns") or 0) <= 0:
        errors.append(f"{name} baseline duration is not positive")
    argv = payload.get("command_argv")
    if argv != list(expected_argv):
        errors.append(f"{name} baseline argv differs")
    if not isinstance(payload.get("passed"), int) or int(payload.get("passed") or 0) <= 0:
        errors.append(f"{name} baseline has no positive pass count")
    if not isinstance(payload.get("failed"), int) or int(payload.get("failed") or 0) < 0:
        errors.append(f"{name} baseline failed count is invalid")
    if not isinstance(payload.get("skipped"), int) or int(payload.get("skipped") or 0) < 0:
        errors.append(f"{name} baseline skipped count is invalid")
    if require_green:
        if payload.get("returncode") != 0 or payload.get("terminal_status") != "passed":
            errors.append(f"{name} baseline is not terminal passed")
        if payload.get("failed") != 0:
            errors.append(f"{name} baseline is not green")
    else:
        if payload.get("terminal_status") not in {"passed", "failed"}:
            errors.append(f"{name} baseline terminal status is not closed")
        if payload.get("returncode") not in {0, 1}:
            errors.append(f"{name} baseline returncode is not a closed pytest status")
    for field in ("environment_identity", "dependency_lock_identity"):
        if re.fullmatch(r"b[a-z2-7]{20,}", str(payload.get(field) or "")) is None:
            errors.append(f"{name} baseline {field} is not canonical")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(payload.get("bounded_log_digest") or "")) is None:
        errors.append(f"{name} baseline log digest is invalid")
    if payload.get("network_access") != "disabled":
        errors.append(f"{name} baseline network policy is not disabled")
    if payload.get("production_credentials_available") is not False:
        errors.append(f"{name} baseline exposed production credentials")
    return errors


def _emit_report(
    *,
    valid: bool,
    source_head: str,
    receipt_bindings: dict[str, dict[str, str]],
    errors: list[str],
) -> int:
    report = {
        "schema": REPORT_SCHEMA,
        "valid": valid,
        "runner": RUNNER_ID,
        "source_head": source_head,
        "receipt_bindings": receipt_bindings,
        "errors": errors,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    sys.stdout.write("\n")
    return 0 if valid else 1


def _run_suite(
    root: Path,
    spec: Mapping[str, object],
    *,
    environment: Mapping[str, str],
    environment_identity: str,
    dependency_lock_identity: str,
) -> dict[str, object]:
    argv = tuple(str(item) for item in spec["argv"])  # type: ignore[index]
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        list(argv),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env=dict(environment),
        timeout=3600.0,
    )
    finished = datetime.now(timezone.utc)
    combined = (result.stdout or "") + ("\n" if result.stderr else "") + (result.stderr or "")
    passed, failed, skipped = _parse_pytest_counts(combined)
    duration_ns = max(1, int((finished - started).total_seconds() * 1_000_000_000))
    terminal = "passed" if result.returncode == 0 and failed == 0 else "failed"
    return {
        "schema": RECEIPT_SCHEMA,
        "runner_id": RUNNER_ID,
        "repository": spec["name"],
        "repository_state_root": _state_root(root, spec.get("gitlink")),  # type: ignore[arg-type]
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_ns": duration_ns,
        "command_argv": list(argv),
        "returncode": int(result.returncode),
        "terminal_status": terminal,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "environment_identity": environment_identity,
        "dependency_lock_identity": dependency_lock_identity,
        "bounded_log_digest": _bounded_log_digest(combined),
        "network_access": "disabled",
        "production_credentials_available": False,
    }


def run_current_tree(root: Path, output_dir: Path) -> int:
    if _production_credentials_present(os.environ):
        # Strip from children; refuse to claim they were available.
        pass
    environment = _child_environment(root)
    environment_identity = _environment_identity(root, environment)
    dependency_lock_identity = _dependency_lock_identity(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for spec in SUITE_SPECS:
        receipt = _run_suite(
            root,
            spec,
            environment=environment,
            environment_identity=environment_identity,
            dependency_lock_identity=dependency_lock_identity,
        )
        errors.extend(
            _validate_receipt(
                receipt,
                name=str(spec["name"]),
                expected_state=str(receipt["repository_state_root"]),
                expected_argv=tuple(str(item) for item in spec["argv"]),  # type: ignore[index]
                require_green=False,
            )
        )
        path = _receipt_path(output_dir, str(spec["binding"]))
        ordered = {field: receipt[field] for field in RECEIPT_FIELDS}
        path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
        bindings[str(spec["binding"])] = {
            "path": _posix(path, root),
            "canonical_identity": _cid_for_obj(root, receipt, noun=str(spec["binding"])),
        }
    return _emit_report(
        valid=not errors,
        source_head=_git(root, "rev-parse", "HEAD").lower(),
        receipt_bindings=bindings,
        errors=errors,
    )


def verify_bundle(root: Path, output_dir: Path, *, require_green: bool) -> int:
    errors: list[str] = []
    bindings: dict[str, dict[str, str]] = {}
    source_head = _git(root, "rev-parse", "HEAD").lower()
    for spec in SUITE_SPECS:
        name = str(spec["name"])
        binding = str(spec["binding"])
        path = _receipt_path(output_dir, binding)
        if not path.is_file():
            errors.append(f"{binding} receipt is absent")
            continue
        payload = _load_json(path)
        expected_state = _state_root(root, spec.get("gitlink"))  # type: ignore[arg-type]
        errors.extend(
            _validate_receipt(
                payload,
                name=name,
                expected_state=expected_state,
                expected_argv=tuple(str(item) for item in spec["argv"]),  # type: ignore[index]
                require_green=require_green,
            )
        )
        recomputed = _cid_for_obj(root, payload, noun=binding)
        bindings[binding] = {
            "path": _posix(path, root),
            "canonical_identity": recomputed,
        }
    if _posix(output_dir, root) == PREREQUISITE_EVIDENCE_REL and not require_green:
        errors.append("prerequisite evidence verification must be green")
    return _emit_report(
        valid=not errors,
        source_head=source_head,
        receipt_bindings=bindings,
        errors=errors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--current-tree", action="store_true")
    mode.add_argument("--verify-bundle", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = _repo_root()
    try:
        if args.current_tree:
            output_dir = _resolve_output_dir(
                root, args.output_dir, default=BASELINE_RECEIPTS_REL
            )
            return run_current_tree(root, output_dir)
        output_dir = _resolve_output_dir(
            root, args.output_dir, default=PREREQUISITE_EVIDENCE_REL
        )
        require_green = _posix(output_dir, root) == PREREQUISITE_EVIDENCE_REL
        return verify_bundle(root, output_dir, require_green=require_green)
    except BaselineRunnerError as exc:
        return _emit_report(
            valid=False,
            source_head="",
            receipt_bindings={},
            errors=[str(exc)],
        )


if __name__ == "__main__":
    raise SystemExit(main())
