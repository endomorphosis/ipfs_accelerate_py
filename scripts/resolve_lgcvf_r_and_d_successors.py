#!/usr/bin/env python3
"""Issue or verify the sole-user LGCVF R&D successor resolution.

Issuance requires an explicit Ed25519 private-key path.  The private key is
never generated, copied, or retained by this program, and a path inside the
repository is rejected.  Verification uses only the repository-pinned public
key and current content-addressed evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_r_and_d_authority import (
    LGCVF_BASE64URL_ENCODING,
    LGCVF_ED25519_ALGORITHM,
    LGCVF_EXTERNAL_R_AND_D_DISPOSITION,
    LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2,
    LGCVF_PRODUCTION_DECLINED_DISPOSITION,
    LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2,
    LGCVF_R_AND_D_AUTHORITY_SCOPE,
    LGCVF_R_AND_D_SIGNATURE_DOMAIN,
    LGCVF_R_AND_D_TRUST_MODEL,
    LgcvfAuthorityBindings,
    LgcvfRAndDTrustPolicy,
    LgcvfSourceRevisions,
    load_lgcvf_r_and_d_trust_policy,
    validate_lgcvf_external_r_and_d_receipt,
    validate_lgcvf_production_declined_r_and_d_receipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_successor_resolution import (
    AUTHORITY_VALIDATION_SCHEMA,
    EXPECTED_DISPOSITIONS,
    LgcvfSuccessorResolutionError,
    build_successor_resolution,
    validate_successor_resolution,
)

DATA_DIR: Final[Path] = (
    ROOT / "data/agent_supervisor/logic_governed_compositional_verification_fabric"
)
PLAN_PATH: Final[Path] = DATA_DIR / "formal_work_plan.json"
QUALIFICATION_PATH: Final[Path] = DATA_DIR / "independent_qualification_result.json"
BENCHMARK_PATH: Final[Path] = DATA_DIR / "benchmark_result.json"
PREDECESSOR_PATH: Final[Path] = DATA_DIR / "successor_tasks.json"
EXTERNAL_RECEIPT_PATH: Final[Path] = (
    DATA_DIR / "external_qualification_r_and_d_receipt.v2.json"
)
PRODUCTION_RECEIPT_PATH: Final[Path] = (
    DATA_DIR / "production_authorization_r_and_d_receipt.v2.json"
)
RESOLUTION_PATH: Final[Path] = DATA_DIR / "successor_resolution.json"
RELEASE_PATH: Final[Path] = (
    ROOT
    / "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md"
)
TRUST_PATH: Final[Path] = ROOT / "config/lgcvf_r_and_d_authority_trust.json"
PUBLIC_KEY_PATH: Final[Path] = ROOT / "config/lgcvf_r_and_d_authority_public_key.pem"
_EVIDENCE_ONLY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_r_and_d_receipt.v2.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/independent_qualification_result.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_r_and_d_receipt.v2.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/r_and_d_terminal_closeout.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_resolution.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_tasks.json",
        "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md",
        "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md",
        "docs/architecture/lgcvf_current_roots_packet.json",
    }
)
_MAX_JSON_BYTES: Final[int] = 16 * 1024 * 1024


class ResolutionCommandError(RuntimeError):
    """Issuance or verification inputs are invalid, stale, or unsafe."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise ResolutionCommandError(f"{label} exceeds its bounded JSON size")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResolutionCommandError(f"{label} is invalid strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionCommandError(f"{label} root is not an object")
    return value


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ResolutionCommandError(f"{label} is unreadable: {exc}") from exc
    return _decode_object(encoded, label=label)


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str, root: Path = ROOT) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ResolutionCommandError(
            f"Git observation failed: {(completed.stderr or completed.stdout).strip()[-1000:]}"
        )
    return completed.stdout.strip()


def _git_path_set(root: Path, *arguments: str) -> set[str]:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ResolutionCommandError(
            "Git worktree query failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1_000:]
        )
    try:
        return {item.decode("utf-8") for item in completed.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise ResolutionCommandError("Git worktree path is not UTF-8") from exc


def _assert_no_uncommitted_source() -> None:
    changed = _git_path_set(ROOT, "diff", "--name-only", "-z", "HEAD")
    untracked = _git_path_set(ROOT, "ls-files", "--others", "--exclude-standard", "-z")
    semantic = (changed | untracked) - _EVIDENCE_ONLY_PATHS
    if semantic:
        raise ResolutionCommandError(
            "uncommitted accelerator source differs: " + min(semantic)
        )
    datasets_root = ROOT / "ipfs_datasets_py"
    datasets_changed = _git_path_set(
        datasets_root, "diff", "--name-only", "-z", "HEAD"
    ) | _git_path_set(datasets_root, "ls-files", "--others", "--exclude-standard", "-z")
    if datasets_changed:
        raise ResolutionCommandError(
            "uncommitted datasets source differs: " + min(datasets_changed)
        )


def _input_snapshots() -> dict[Path, bytes]:
    paths = (
        PLAN_PATH,
        QUALIFICATION_PATH,
        BENCHMARK_PATH,
        PREDECESSOR_PATH,
        RELEASE_PATH,
        TRUST_PATH,
        PUBLIC_KEY_PATH,
    )
    snapshots: dict[Path, bytes] = {}
    for path in paths:
        try:
            snapshots[path] = path.read_bytes()
        except OSError as exc:
            raise ResolutionCommandError(
                f"authority input is unreadable: {path}"
            ) from exc
    return snapshots


def _require_snapshots_unchanged(snapshots: Mapping[Path, bytes]) -> None:
    for path, expected in snapshots.items():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise ResolutionCommandError(
                f"authority input disappeared: {path}"
            ) from exc
        if observed != expected:
            raise ResolutionCommandError(
                f"authority input changed during validation: {path}"
            )


def _semantic_source_revision() -> tuple[str, str]:
    revisions = _git("rev-list", "--first-parent", "HEAD").splitlines()
    for revision in revisions[:100_000]:
        lineage = _git("rev-list", "--parents", "-n", "1", revision).split()
        if not lineage or lineage[0] != revision:
            raise ResolutionCommandError("Git source lineage is invalid")
        if len(lineage) > 1:
            changed = _git(
                "diff", "--no-ext-diff", "--name-only", lineage[1], revision
            ).splitlines()
        else:
            changed = _git(
                "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", revision
            ).splitlines()
        if set(changed) - _EVIDENCE_ONLY_PATHS:
            tree = _git("rev-parse", f"{revision}^{{tree}}")
            return revision, tree
    raise ResolutionCommandError(
        "repository history contains no semantic source revision"
    )


def current_source_revisions() -> LgcvfSourceRevisions:
    accelerator_head, accelerator_tree = _semantic_source_revision()
    entry = _git("ls-tree", accelerator_head, "--", "ipfs_datasets_py").split()
    if len(entry) < 3 or entry[0] != "160000" or entry[1] != "commit":
        raise ResolutionCommandError(
            "source revision does not contain a datasets gitlink"
        )
    datasets_gitlink = entry[2]
    datasets_head = _git("rev-parse", "HEAD", root=ROOT / "ipfs_datasets_py")
    datasets_tree = _git("rev-parse", "HEAD^{tree}", root=ROOT / "ipfs_datasets_py")
    if datasets_head != datasets_gitlink:
        raise ResolutionCommandError(
            "datasets checkout differs from the source gitlink"
        )
    return LgcvfSourceRevisions(
        accelerator_head=accelerator_head,
        accelerator_tree=accelerator_tree,
        datasets_head=datasets_head,
        datasets_tree=datasets_tree,
        datasets_gitlink=datasets_gitlink,
    )


def load_trust_policy() -> LgcvfRAndDTrustPolicy:
    return load_lgcvf_r_and_d_trust_policy(ROOT)


def _bindings(
    qualification: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    source_revisions: LgcvfSourceRevisions,
) -> LgcvfAuthorityBindings:
    plan = _load_object(PLAN_PATH, label="formal plan")
    plan_cid = plan.get("plan_cid")
    if not isinstance(plan_cid, str):
        raise ResolutionCommandError("formal plan CID is absent")
    return LgcvfAuthorityBindings(
        plan_cid=plan_cid,
        qualification_result_cid=str(qualification.get("result_cid") or ""),
        qualification_checkout_fingerprint_cid=str(
            qualification.get("checkout_fingerprint_cid") or ""
        ),
        benchmark_report_cid=str(benchmark.get("report_cid") or ""),
        release_report_sha256=_sha256_file(RELEASE_PATH),
        source_revisions=source_revisions,
    )


def _sign_payload(
    payload: Mapping[str, Any], private_key: Ed25519PrivateKey
) -> dict[str, Any]:
    body = dict(payload)
    payload_bytes = canonical_json_bytes(body)
    receipt = body | {
        "payload_cid": content_identity(body),
        "signature": {
            "algorithm": LGCVF_ED25519_ALGORITHM,
            "encoding": LGCVF_BASE64URL_ENCODING,
            "value": _b64url(
                private_key.sign(LGCVF_R_AND_D_SIGNATURE_DOMAIN + payload_bytes)
            ),
        },
    }
    receipt["receipt_cid"] = content_identity(receipt)
    return receipt


def _signer(trust: LgcvfRAndDTrustPolicy) -> dict[str, Any]:
    return {
        "identity": trust.identity,
        "role": trust.role,
        "key_id": trust.key_id,
        "public_key_base64url": trust.public_key_base64url,
    }


def _issue_receipts(
    *,
    trust: LgcvfRAndDTrustPolicy,
    expected: LgcvfAuthorityBindings,
    private_key: Ed25519PrivateKey,
    issued_at: str,
    expires_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    external_payload = {
        "schema": LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2,
        "receipt_kind": "external_qualification_r_and_d",
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "issuer": _signer(trust),
        "third_party_independence_claimed": False,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "plan_cid": expected.plan_cid,
        "qualification_result_cid": expected.qualification_result_cid,
        "qualification_checkout_fingerprint_cid": expected.qualification_checkout_fingerprint_cid,
        "benchmark_report_cid": expected.benchmark_report_cid,
        "source_revisions": expected.source_revisions.to_dict(),
        "cohorts": {
            "live_local_model_execution": "unavailable",
            "live_remote_model_execution": "unavailable",
            "production_authoritative_evidence": "unavailable",
        },
        "provider_disclosure_policy": "No model or remote provider was invoked; only the bound hermetic R&D cohort was judged.",
        "multi_writer": {
            "quack_qualified": False,
            "disposition": "unavailable",
            "notes": "The bound evidence does not qualify a production multi-writer deployment.",
        },
        "disposition": LGCVF_EXTERNAL_R_AND_D_DISPOSITION,
        "release_qualified": False,
        "production_authorized": False,
        "limitations": [
            "The signer is the sole R&D user and candidate owner, not an independent third party.",
            "The receipt records self-verification only and grants no release qualification.",
            "Live model, remote-provider, and production-authoritative cohorts were not exercised.",
        ],
    }
    external = _sign_payload(external_payload, private_key)
    validated_external = validate_lgcvf_external_r_and_d_receipt(
        external, trust=trust, expected=expected
    )
    production_payload = {
        "schema": LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2,
        "receipt_kind": "production_authorization_r_and_d",
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "operator": _signer(trust),
        "issued_at": issued_at,
        "expires_at": expires_at,
        "plan_cid": expected.plan_cid,
        "qualification_result_cid": expected.qualification_result_cid,
        "qualification_checkout_fingerprint_cid": expected.qualification_checkout_fingerprint_cid,
        "benchmark_report_cid": expected.benchmark_report_cid,
        "external_qualification_receipt_cid": validated_external.receipt_cid,
        "external_qualification_payload_cid": validated_external.payload_cid,
        "release_report_sha256": expected.release_report_sha256,
        "source_revisions": expected.source_revisions.to_dict(),
        "scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "lgswf_006_reused": False,
        "depends_on_lgcvf_121": True,
        "depends_on_lgcvf_122": True,
        "disposition": LGCVF_PRODUCTION_DECLINED_DISPOSITION,
        "release_qualified": False,
        "production_authorized": False,
        "limitations": [
            "Production authorization is explicitly declined for this single-user R&D checkout.",
            "The partial benchmark and self-verification receipt are not release qualification.",
            "No public, shared, remote, or production deployment is authorized.",
        ],
    }
    production = _sign_payload(production_payload, private_key)
    validate_lgcvf_production_declined_r_and_d_receipt(
        production,
        external_receipt=external,
        trust=trust,
        expected=expected,
    )
    return external, production


def _authority_validator(
    *,
    trust: LgcvfRAndDTrustPolicy,
    expected: LgcvfAuthorityBindings,
    external: Mapping[str, Any],
    validation_time: datetime | None = None,
):
    expected_sources = expected.source_revisions.to_dict()

    def validate(
        *,
        task_id: str,
        disposition: str,
        receipt: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            context.get("task_id") != task_id
            or context.get("disposition") != disposition
            or context.get("plan_cid") != expected.plan_cid
            or context.get("qualification_result_cid")
            != expected.qualification_result_cid
            or context.get("benchmark_report_cid") != expected.benchmark_report_cid
            or context.get("source_roots") != expected_sources
        ):
            raise ResolutionCommandError(
                "successor authority context is stale or foreign"
            )
        if task_id == "LGCVF-S001" and disposition == EXPECTED_DISPOSITIONS[task_id]:
            validated = validate_lgcvf_external_r_and_d_receipt(
                receipt,
                trust=trust,
                expected=expected,
                now=validation_time,
            )
        elif task_id == "LGCVF-S002" and disposition == EXPECTED_DISPOSITIONS[task_id]:
            validated = validate_lgcvf_production_declined_r_and_d_receipt(
                receipt,
                external_receipt=external,
                trust=trust,
                expected=expected,
                now=validation_time,
            )
        else:
            raise ResolutionCommandError("unsupported authority task or disposition")
        verdict: dict[str, Any] = {
            "schema": AUTHORITY_VALIDATION_SCHEMA,
            "valid": True,
            "signed": True,
            "task_id": task_id,
            "disposition": disposition,
            "receipt_cid": validated.receipt_cid,
            "release_qualified": False,
            "production_authorized": False,
            "context_cid": context.get("context_cid"),
        }
        verdict["validation_cid"] = content_identity(verdict)
        return verdict

    return validate


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _admit_output(path: Path, value: Mapping[str, Any]) -> bytes | None:
    """Admit only an absent output or byte-identical append-only replay."""

    wanted = _encoded_json(value)
    try:
        observed = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResolutionCommandError(f"cannot inspect output {path}: {exc}") from exc
    if observed == wanted:
        return observed
    raise ResolutionCommandError(f"append-only output already differs: {path}")


def _write_guarded(
    path: Path,
    value: Mapping[str, Any],
    *,
    admitted_previous: bytes | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encoded_json(value)
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    except OSError as exc:
        raise ResolutionCommandError(f"cannot recheck output {path}: {exc}") from exc
    if current == encoded:
        return
    if current != admitted_previous:
        raise ResolutionCommandError(f"output changed during issuance: {path}")
    if admitted_previous is not None:
        raise ResolutionCommandError(f"append-only output already differs: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ResolutionCommandError(
                f"append-only output appeared during issuance: {path}"
            ) from exc
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_private_key(path: Path, trust: LgcvfRAndDTrustPolicy) -> Ed25519PrivateKey:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ResolutionCommandError("private key must remain outside the repository")
    try:
        metadata = resolved.stat()
    except OSError as exc:
        raise ResolutionCommandError(f"private key cannot be inspected: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or metadata.st_size < 32
        or metadata.st_size > 16 * 1024
    ):
        raise ResolutionCommandError(
            "private key must be an owner-only regular file of bounded size"
        )
    try:
        key = serialization.load_pem_private_key(resolved.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise ResolutionCommandError(f"private key is unreadable: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ResolutionCommandError("private key is not Ed25519")
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if public != trust.public_key:
        raise ResolutionCommandError("private key does not match the pinned public key")
    return key


def _parse_time(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ResolutionCommandError(f"{label} is not RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResolutionCommandError(f"{label} lacks a UTC offset")
    return parsed.astimezone(timezone.utc)


def verify_current_successor_resolution() -> dict[str, Any]:
    _assert_no_uncommitted_source()
    snapshots = _input_snapshots()
    source_before = current_source_revisions()
    trust = load_trust_policy()
    qualification = _load_object(QUALIFICATION_PATH, label="qualification result")
    benchmark = _load_object(BENCHMARK_PATH, label="benchmark result")
    predecessor = _load_object(PREDECESSOR_PATH, label="successor predecessor")
    external = _load_object(EXTERNAL_RECEIPT_PATH, label="external R&D receipt")
    production = _load_object(
        PRODUCTION_RECEIPT_PATH, label="production-declined receipt"
    )
    resolution = _load_object(RESOLUTION_PATH, label="successor resolution")
    snapshots.update(
        {
            EXTERNAL_RECEIPT_PATH: EXTERNAL_RECEIPT_PATH.read_bytes(),
            PRODUCTION_RECEIPT_PATH: PRODUCTION_RECEIPT_PATH.read_bytes(),
            RESOLUTION_PATH: RESOLUTION_PATH.read_bytes(),
        }
    )
    sources = current_source_revisions()
    if sources != source_before:
        raise ResolutionCommandError("semantic source roots changed during validation")
    expected = _bindings(qualification, benchmark, sources)
    archival_time = _parse_time(
        str(production.get("issued_at") or ""),
        label="production issued_at",
    )
    authority_validator = _authority_validator(
        trust=trust,
        expected=expected,
        external=external,
        validation_time=archival_time,
    )
    validation = validate_successor_resolution(
        resolution,
        predecessor=predecessor,
        qualification=qualification,
        benchmark=benchmark,
        expected_source_roots=sources.to_dict(),
        authority_receipts={"LGCVF-S001": external, "LGCVF-S002": production},
        authority_validator=authority_validator,
    )
    result = {
        **validation,
        "schema": "lgcvf-r-and-d-successor-check@1",
        "external_qualification_receipt_cid": external["receipt_cid"],
        "production_authorization_receipt_cid": production["receipt_cid"],
        "trust_key_id": trust.key_id,
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "third_party_independence_claimed": False,
        "archival_validation_time": archival_time.isoformat().replace("+00:00", "Z"),
    }
    result["check_cid"] = content_identity(result)
    _require_snapshots_unchanged(snapshots)
    _assert_no_uncommitted_source()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--issue", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument(
        "--private-key",
        type=Path,
        help="owner-only unencrypted Ed25519 PEM outside the repository",
    )
    parser.add_argument("--issued-at")
    parser.add_argument("--expires-at")
    args = parser.parse_args(argv)
    try:
        if args.check:
            if args.private_key is not None or args.issued_at or args.expires_at:
                parser.error("issuance arguments require --issue")
            result = verify_current_successor_resolution()
        else:
            if args.private_key is None:
                parser.error("--issue requires --private-key")
            _assert_no_uncommitted_source()
            snapshots = _input_snapshots()
            trust = load_trust_policy()
            qualification = _load_object(
                QUALIFICATION_PATH, label="qualification result"
            )
            benchmark = _load_object(BENCHMARK_PATH, label="benchmark result")
            predecessor = _load_object(PREDECESSOR_PATH, label="successor predecessor")
            sources = current_source_revisions()
            expected = _bindings(qualification, benchmark, sources)
            private_key = _load_private_key(args.private_key, trust)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            issued = (
                _parse_time(args.issued_at, label="issued_at")
                if args.issued_at
                else now
            )
            expires = (
                _parse_time(args.expires_at, label="expires_at")
                if args.expires_at
                else issued + timedelta(days=365)
            )
            if not issued <= now < expires:
                raise ResolutionCommandError(
                    "receipt validity does not contain the current time"
                )
            issued_at = issued.isoformat().replace("+00:00", "Z")
            expires_at = expires.isoformat().replace("+00:00", "Z")
            external, production = _issue_receipts(
                trust=trust,
                expected=expected,
                private_key=private_key,
                issued_at=issued_at,
                expires_at=expires_at,
            )
            authority_validator = _authority_validator(
                trust=trust, expected=expected, external=external
            )
            resolution = build_successor_resolution(
                predecessor=predecessor,
                qualification=qualification,
                benchmark=benchmark,
                source_roots=sources.to_dict(),
                authority_receipts={"LGCVF-S001": external, "LGCVF-S002": production},
                authority_validator=authority_validator,
            )
            external_previous = _admit_output(EXTERNAL_RECEIPT_PATH, external)
            production_previous = _admit_output(PRODUCTION_RECEIPT_PATH, production)
            resolution_previous = _admit_output(RESOLUTION_PATH, resolution)
            _require_snapshots_unchanged(snapshots)
            if current_source_revisions() != sources:
                raise ResolutionCommandError(
                    "semantic source roots changed during issuance"
                )
            _assert_no_uncommitted_source()
            _write_guarded(
                EXTERNAL_RECEIPT_PATH,
                external,
                admitted_previous=external_previous,
            )
            _write_guarded(
                PRODUCTION_RECEIPT_PATH,
                production,
                admitted_previous=production_previous,
            )
            _write_guarded(
                RESOLUTION_PATH,
                resolution,
                admitted_previous=resolution_previous,
            )
            result = {
                "schema": "lgcvf-r-and-d-successor-issuance@1",
                "issued_at": issued_at,
                "expires_at": expires_at,
                "external_qualification_receipt_cid": external["receipt_cid"],
                "production_authorization_receipt_cid": production["receipt_cid"],
                "successor_resolution_cid": resolution["resolution_cid"],
                "task_implementation_complete": True,
                "objective_complete": False,
                "release_qualified": False,
                "production_authorized": False,
            }
            result["issuance_cid"] = content_identity(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        ValueError,
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
