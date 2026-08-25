#!/usr/bin/env python3
"""Create-once EAAEF-000 bootstrap admission for the current source HEAD.

Diagnoses without starting a supervisor. Publishes only when the live
materialization receipt, provider/container qualification, Quack owner
qualification, and owner-only parent chain all bind this HEAD. Does not
rematerialize a new store generation.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64
import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_der_private_key
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
    ExternalAgentBootstrapAdmissionError,
    _canonical_bytes,
    _open_secure_publication_parent,
    _publication_parent_is_stable,
    assemble_external_agent_bootstrap_admission,
    external_agent_bootstrap_admission_relative_path,
    prepare_external_agent_bootstrap_admission,
    publish_external_agent_bootstrap_admission,
)

from ipfs_accelerate_py import agent_implementation_route as routes

CONFIG_PATH = ROOT / "config/external_agent_autonomous_execution_fabric_scheduler.json"
CURSOR_PATH = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
AUTHORITY_DIR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "authority"
)
PRINCIPAL_DIR = AUTHORITY_DIR / "runtime-principals"
_PRINCIPAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-runtime-principal-secret@1"
)
_IDENTITY_BLOCKERS = frozenset(
    {
        "materialization_source_or_board_mismatch",
        "materialization_source_tree_mismatch",
        "materialization_board_cid_mismatch",
        "materialization_generation_cursor_invalid",
        "immutable publication parent is not an owner-only directory",
        "immutable publication parent is unavailable",
        "bootstrap admission receipt already exists",
    }
)
_INPUT_BUNDLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission-inputs@2"
)
_INPUT_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "source_head",
        "source_tree",
        "provider_container_qualification_cid",
        "quack_owner_qualification_cid",
        "route_binding",
        "image_qualification",
        "container_profile",
        "input_cid",
    }
)
_AUTHORITY_ARTIFACT_PREFIXES = {
    "provider_container": "provider-container-qualification--",
    "quack_owner": "quack-owner-qualification--",
    "admission_inputs": "bootstrap-admission-inputs--",
}
_GENERATION_CURSOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-store-generation-cursor@1"
)
_GENERATION_RE = re.compile(r"^eaaef-run-v(?P<number>\d+)$")
_RUNTIME_REGISTRY_RE = re.compile(
    r"^data/agent_supervisor/external_agent_autonomous_execution_fabric/"
    r"run-v(?P<number>\d+)/registry$"
)
_EXPLICIT_QUALIFICATION_TRUST_FIELDS = (
    "trusted_provider_signer_dids",
    "trusted_image_reviewer_dids",
    "trusted_container_profile_reviewer_dids",
    "trusted_quack_reviewer_dids",
)
_QUALIFICATION_TRUST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-qualification-trust@1"
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _principal_did(role: str) -> str:
    if role not in {"worker", "provider", "quack_owner"}:
        raise RuntimeError("runtime principal role is invalid")
    parent = os.lstat(PRINCIPAL_DIR)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise RuntimeError("runtime principal directory is unsafe")
    path = PRINCIPAL_DIR / f"{role}.json"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > 65_536
        ):
            raise RuntimeError(f"{role} principal file is unsafe")
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        pathname = os.lstat(path)
        if (
            len(raw) != before.st_size
            or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
            or (before.st_dev, before.st_ino) != (pathname.st_dev, pathname.st_ino)
            or stat.S_ISLNK(pathname.st_mode)
        ):
            raise RuntimeError(f"{role} principal file changed while read")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{role} principal is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema", "role", "did", "private_key_pkcs8_der_b64"}
        or payload.get("schema") != _PRINCIPAL_SCHEMA
        or payload.get("role") != role
    ):
        raise RuntimeError(f"{role} principal binding is invalid")
    did = str(payload.get("did") or "")
    try:
        encoded_key = base64.b64decode(
            str(payload.get("private_key_pkcs8_der_b64") or ""),
            validate=True,
        )
        key = load_der_private_key(encoded_key, password=None)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{role} principal private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey) or ed25519_did_key(key.public_key()) != did:
        raise RuntimeError(f"{role} principal DID does not match its private key")
    return did


def _parent_mode_blockers(relative: Path) -> list[str]:
    blockers: list[str] = []
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError:
            blockers.append("immutable publication parent is unavailable")
            return blockers
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or mode & 0o077
        ):
            blockers.append(
                "immutable publication parent is not an owner-only directory"
            )
            return blockers
    return blockers


def _authority_artifact_name(
    kind: str,
    source_head: str,
    source_tree: str,
) -> str:
    return (
        f"{_AUTHORITY_ARTIFACT_PREFIXES[kind]}"
        f"{source_head}--{source_tree}.json"
    )


def _qualification_paths(source_head: str, source_tree: str) -> dict[str, Any]:
    names = [
        path.name
        for path in AUTHORITY_DIR.iterdir()
        if path.is_file() and not path.is_symlink()
    ]
    candidates = {
        "provider_container": sorted(
            name
            for name in names
            if name.startswith("provider-container-qualification--")
        ),
        "quack_owner": sorted(
            name for name in names if name.startswith("quack-owner-qualification--")
        ),
    }
    expected = {
        kind: _authority_artifact_name(kind, source_head, source_tree)
        for kind in _AUTHORITY_ARTIFACT_PREFIXES
    }
    return {
        **candidates,
        "expected": expected,
        "available": {
            kind: name in names for kind, name in expected.items()
        },
    }


def _load_authority_artifact(name: str) -> dict[str, Any]:
    """Read one owner-only authority object through an anchored parent walk."""

    path = AUTHORITY_DIR / name
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"authority artifact escapes the repository: {name}") from exc
    if relative.name != name or Path(name).name != name:
        raise RuntimeError(f"authority artifact name is unsafe: {name}")
    try:
        root_fd, parent_fd, identities = _open_secure_publication_parent(
            ROOT,
            relative,
        )
    except ExternalAgentBootstrapAdmissionError as exc:
        raise RuntimeError(f"authority artifact parent is unsafe: {name}") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        os.close(root_fd)
        raise RuntimeError(f"authority artifact is unsafe: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o077
            or before.st_size <= 0
            or before.st_size > 2 * 1024 * 1024
        ):
            raise RuntimeError(f"authority artifact is unsafe: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(2 * 1024 * 1024 + 1)
        after = os.fstat(descriptor)
        pathname = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or identity(before) != identity(pathname)
            or not _publication_parent_is_stable(root_fd, identities, parent_fd)
        ):
            raise RuntimeError(f"authority artifact changed while read: {name}")
    finally:
        os.close(descriptor)
        os.close(parent_fd)
        os.close(root_fd)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"authority artifact is invalid JSON: {name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"authority artifact is not an object: {name}")
    return payload


def _explicit_reviewer_trust(
    qualification_trust: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Require distinct tracked trust roles; never infer one role from another."""

    if (
        set(qualification_trust)
        != {"schema", *_EXPLICIT_QUALIFICATION_TRUST_FIELDS}
        or qualification_trust.get("schema") != _QUALIFICATION_TRUST_SCHEMA
    ):
        raise RuntimeError(
            "distinct bootstrap qualification reviewer trust is unavailable"
        )
    trust: dict[str, tuple[str, ...]] = {
        field: tuple(qualification_trust.get(field) or ())
        for field in _EXPLICIT_QUALIFICATION_TRUST_FIELDS
    }
    if any(
        not values
        or any(
            not isinstance(item, str) or not item.startswith("did:key:z")
            for item in values
        )
        or len(values) != len(set(values))
        for values in trust.values()
    ):
        raise RuntimeError(
            "distinct bootstrap qualification reviewer trust is unavailable"
        )
    roles = tuple(trust.values())
    if any(
        set(roles[left]).intersection(roles[right])
        for left in range(len(roles))
        for right in range(left + 1, len(roles))
    ):
        raise RuntimeError(
            "bootstrap qualification reviewer trust roles overlap"
        )
    return trust


def _admission_expiry_ms(
    *,
    now_ms: int,
    admission_inputs: dict[str, Any],
) -> int:
    """Bound the admission to every loaded qualification and invocation."""

    candidates = [now_ms + 3_600_000]
    for artifact_name in (
        "provider_container_qualification",
        "image_qualification",
        "container_profile",
    ):
        artifact = admission_inputs.get(artifact_name)
        expiry = artifact.get("expires_at_ms") if isinstance(artifact, dict) else None
        if isinstance(expiry, int) and not isinstance(expiry, bool) and expiry > now_ms:
            candidates.append(expiry)
    quack = admission_inputs.get("quack_owner_qualification")
    qualification = quack.get("qualification") if isinstance(quack, dict) else None
    quack_expiry = (
        qualification.get("expires_at_ms")
        if isinstance(qualification, dict)
        else None
    )
    if (
        isinstance(quack_expiry, int)
        and not isinstance(quack_expiry, bool)
        and quack_expiry > now_ms
    ):
        candidates.append(quack_expiry)
    route = admission_inputs.get("route_plan")
    invocation = getattr(route, "invocation_binding", None)
    invocation_expiry = getattr(invocation, "expires_at_ms", None)
    if (
        isinstance(invocation_expiry, int)
        and not isinstance(invocation_expiry, bool)
        and invocation_expiry > now_ms
    ):
        candidates.append(invocation_expiry)
    return min(candidates)


def _load_admission_inputs(
    *,
    source_head: str,
    source_tree: str,
    now_ms: int,
    qualification_trust: dict[str, Any],
) -> dict[str, Any]:
    """Load and revalidate the exact source-addressed admission inputs."""

    provider = _load_authority_artifact(
        _authority_artifact_name("provider_container", source_head, source_tree)
    )
    quack = _load_authority_artifact(
        _authority_artifact_name("quack_owner", source_head, source_tree)
    )
    bundle = _load_authority_artifact(
        _authority_artifact_name("admission_inputs", source_head, source_tree)
    )
    body = dict(bundle)
    input_cid = str(body.pop("input_cid", ""))
    expected_cid = "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if (
        set(bundle) != _INPUT_BUNDLE_FIELDS
        or bundle.get("schema") != _INPUT_BUNDLE_SCHEMA
        or bundle.get("source_head") != source_head
        or bundle.get("source_tree") != source_tree
        or bundle.get("provider_container_qualification_cid")
        != provider.get("qualification_cid")
        or bundle.get("quack_owner_qualification_cid")
        != quack.get("receipt_cid")
        or input_cid != expected_cid
    ):
        raise RuntimeError("bootstrap admission input bundle is not source-bound")
    provider_source_matches = (
        provider.get("source_head") == source_head
        and provider.get("source_tree") == source_tree
    )
    quack_source = quack.get("source")
    quack_source_matches = isinstance(quack_source, dict) and (
        quack_source.get("commit") == source_head
        and quack_source.get("tree") == source_tree
    )
    if not provider_source_matches or not quack_source_matches:
        raise RuntimeError("bootstrap qualifications are stale for this source")
    route_binding = bundle.get("route_binding")
    image = bundle.get("image_qualification")
    profile = bundle.get("container_profile")
    if (
        not isinstance(route_binding, dict)
        or not isinstance(image, dict)
        or not isinstance(profile, dict)
    ):
        raise RuntimeError("bootstrap admission input bundle is incomplete")
    route_plan = routes.resolve_agent_implementation_route_binding(
        route_binding,
        repo_root=ROOT,
        now_ms=now_ms,
        max_age_ms=5 * 60 * 1000,
    )
    trust = _explicit_reviewer_trust(qualification_trust)
    return {
        "provider_container_qualification": provider,
        "route_plan": route_plan,
        "image_qualification": image,
        "container_profile": profile,
        "quack_owner_qualification": quack,
        **trust,
    }


def _receipt_path(config: dict[str, Any] | None = None) -> Path:
    """Resolve the exact materializer generation without accepting path input."""

    scheduler = dict(config or _load(CONFIG_PATH))
    program = scheduler.get("bootstrap_database_program")
    if not isinstance(program, dict):
        raise RuntimeError("bootstrap database program is unavailable")
    configured = str(program.get("store_generation") or "")
    registry = str(program.get("runtime_registry_path") or "")
    configured_match = _GENERATION_RE.fullmatch(configured)
    registry_match = _RUNTIME_REGISTRY_RE.fullmatch(registry)
    if (
        configured_match is None
        or registry_match is None
        or configured_match.group("number") != registry_match.group("number")
    ):
        raise RuntimeError("configured materialization generation is invalid")
    active = configured
    if CURSOR_PATH.exists():
        if not CURSOR_PATH.is_file() or CURSOR_PATH.is_symlink():
            raise RuntimeError("materialization generation cursor is unsafe")
        cursor = _load(CURSOR_PATH)
        candidate = str(cursor.get("active_generation") or "")
        candidate_match = _GENERATION_RE.fullmatch(candidate)
        if (
            cursor.get("schema") != _GENERATION_CURSOR_SCHEMA
            or cursor.get("configured_generation") != configured
            or cursor.get("process_started") is not False
            or candidate_match is None
            or int(candidate_match.group("number"))
            < int(configured_match.group("number"))
        ):
            raise RuntimeError("materialization generation cursor is invalid")
        active = candidate
    active_number = _GENERATION_RE.fullmatch(active)
    if active_number is None:
        raise RuntimeError("active materialization generation is invalid")
    active_registry = registry.replace(
        f"run-v{configured_match.group('number')}",
        f"run-v{active_number.group('number')}",
        1,
    )
    if _RUNTIME_REGISTRY_RE.fullmatch(active_registry) is None:
        raise RuntimeError("active materialization registry is invalid")
    return ROOT / active_registry / "bootstrap-materialization.json"


def diagnose() -> dict[str, Any]:
    """Read-only blockers for create-once publication at current HEAD."""

    source_head = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    relative = external_agent_bootstrap_admission_relative_path(source_head)
    target = ROOT / relative
    receipt_error = ""
    try:
        receipt_path = _receipt_path()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        receipt_error = str(exc)
        receipt_path = ROOT / ".invalid-eaaef-materialization-receipt"
    board_path = (
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "task_board.json"
    )
    receipt = _load(receipt_path) if receipt_path.is_file() else {}
    board = _load(board_path) if board_path.is_file() else {}
    qualifications = _qualification_paths(source_head, source_tree)
    blockers: list[str] = []
    if receipt_error:
        blockers.append("materialization_generation_cursor_invalid")
    if str(receipt.get("source_head") or "") != source_head:
        blockers.append("materialization_source_or_board_mismatch")
    if str(receipt.get("source_tree") or "") != source_tree:
        blockers.append("materialization_source_tree_mismatch")
    if str((receipt.get("board_validation") or {}).get("board_cid") or "") != str(
        board.get("board_cid") or ""
    ):
        blockers.append("materialization_board_cid_mismatch")
    if not qualifications["available"]["provider_container"]:
        blockers.append("provider_container_qualification_missing")
    if not qualifications["available"]["quack_owner"]:
        blockers.append("quack_owner_qualification_missing")
    if not qualifications["available"]["admission_inputs"]:
        blockers.append("bootstrap_admission_inputs_missing")
    try:
        scheduler = _load(CONFIG_PATH)
        _explicit_reviewer_trust(
            dict(scheduler.get("bootstrap_qualification_trust") or {})
        )
    except (OSError, TypeError, ValueError, RuntimeError):
        blockers.append("bootstrap_qualification_trust_missing_or_overlapping")
    blockers.extend(_parent_mode_blockers(relative))
    if target.is_file():
        blockers.append("bootstrap admission receipt already exists")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission-issue@1",
        "source_head": source_head,
        "source_tree": source_tree,
        "materialization_source_head": str(receipt.get("source_head") or ""),
        "materialization_source_tree": str(receipt.get("source_tree") or ""),
        "materialization_receipt_cid": str(receipt.get("receipt_cid") or ""),
        "materialization_store_generation": str(
            (
                (receipt.get("database_program_bindings") or {}).get("bootstrap") or {}
            ).get("store_generation")
            or ""
        ),
        "board_cid": str(board.get("board_cid") or ""),
        "relative_path": relative.as_posix(),
        "exists": target.is_file(),
        "qualifications": qualifications,
        "materialization_child_adapter_status": str(
            (receipt.get("database_program_bindings") or {}).get(
                "operational_child_adapter_status"
            )
            or ""
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "identity_blockers": [
            item
            for item in dict.fromkeys(blockers)
            if item in _IDENTITY_BLOCKERS
        ],
        "would_publish": not blockers,
        "would_publish_nogo": not any(
            item in _IDENTITY_BLOCKERS for item in blockers
        )
        and not target.is_file(),
        "published": False,
        "process_started": False,
        "configured_board_launch": False,
        "rematerialize": False,
    }


def issue(
    *,
    operator_approval: dict[str, Any] | None = None,
    security_approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = diagnose()
    if report.get("identity_blockers"):
        report["published"] = False
        return report
    config = _load(CONFIG_PATH)
    board = _load(
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "task_board.json"
    )
    receipt = _load(_receipt_path())
    now_ms = int(time.time() * 1000)
    live_seal = config.get("configured_board_live_seal") or {}
    trusted_operator = tuple(live_seal.get("trusted_operator_dids") or ())
    trusted_security = tuple(live_seal.get("trusted_security_reviewer_dids") or ())
    try:
        admission_inputs = _load_admission_inputs(
            source_head=str(report["source_head"]),
            source_tree=str(report["source_tree"]),
            now_ms=now_ms,
            qualification_trust=dict(
                config.get("bootstrap_qualification_trust") or {}
            ),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        report["blockers"] = list(
            dict.fromkeys([*report["blockers"], str(exc)])
        )
        report["would_publish"] = False
        report["published"] = False
        return report
    expires_at_ms = _admission_expiry_ms(
        now_ms=now_ms,
        admission_inputs=admission_inputs,
    )
    try:
        statement = prepare_external_agent_bootstrap_admission(
            board=board,
            materialization_receipt=receipt,
            **admission_inputs,
            expected_worker_principal_did=_principal_did("worker"),
            expected_provider_principal_did=_principal_did("provider"),
            expected_source_commit=str(report["source_head"]),
            expected_source_tree=str(report["source_tree"]),
            one_use_nonce=os.urandom(24).hex(),
            issued_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
        )
    except ExternalAgentBootstrapAdmissionError as exc:
        report["blockers"] = list(dict.fromkeys([*report["blockers"], str(exc)]))
        report["would_publish"] = False
        report["published"] = False
        return report
    report["statement_decision"] = str(statement.get("decision") or "")
    report["statement_cid"] = str(statement.get("statement_cid") or "")
    report["prepared_statement"] = statement
    if statement.get("decision") != "admitted":
        report["blockers"] = list(
            dict.fromkeys(
                [
                    *report["blockers"],
                    *(str(item) for item in statement.get("blockers") or ()),
                    "bootstrap admission statement is a typed no-go",
                ]
            )
        )
        report["would_publish"] = False
        report["published"] = False
        return report
    if operator_approval is None or security_approval is None:
        report["blockers"] = list(
            dict.fromkeys(
                [
                    *report["blockers"],
                    "separate operator and security approval artifacts are required",
                ]
            )
        )
        report["would_publish"] = False
        report["published"] = False
        return report
    receipt_out = assemble_external_agent_bootstrap_admission(
        statement,
        operator_approval=operator_approval,
        security_approval=security_approval,
        trusted_operator_dids=trusted_operator,
        trusted_security_reviewer_dids=trusted_security,
        now_ms=now_ms,
    )
    verification = publish_external_agent_bootstrap_admission(
        ROOT,
        receipt_out,
        trusted_operator_dids=trusted_operator,
        trusted_security_reviewer_dids=trusted_security,
        now_ms=now_ms,
    )
    report["published"] = True
    report["statement_decision"] = statement.get("decision")
    report["receipt_cid"] = verification.get("receipt_cid")
    report["relative_path"] = external_agent_bootstrap_admission_relative_path(
        str(report["source_head"])
    ).as_posix()
    report["process_started"] = False
    report["configured_board_launch"] = False
    return report


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    if command not in {"diagnose", "issue"}:
        raise SystemExit("usage: issue_eaaef_bootstrap_admission.py [diagnose|issue]")
    payload = diagnose() if command == "diagnose" else issue()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("published") else 1


if __name__ == "__main__":
    raise SystemExit(main())
