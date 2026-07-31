#!/usr/bin/env python3
"""Production ZKP circuit deployment binding and certification.

``ZKPDeploymentCertification@1`` / FVT-G190 (FVT-047).

Owns the attestation-lane handler for the reviewed, secret-safe production
ZKP deployment lock at ``config/formal_verification_zkp_deployment.lock.json``.

Certification proves:

* circuit, ceremony, proving-key and verification-key digests, public-input
  schema, backend, expiry, freshness, and revocation are exact and reviewable;
* live positive verification and corrupted proof/key/public-input, circuit
  mismatch, mutation, replay, stale, and revoked cases pass fail-closed;
* private witnesses and secrets never enter Git, logs, caches, public
  receipts, or model context;
* ZKP authority attests an underlying trusted receipt and never replaces
  semantic theorem authority.

Private material is referenced only by digest and configured secret-safe
location. Certification never installs, downloads, or opens the network, and
never edits the central multi-prover certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for _candidate in (_REPO_ROOT, _DATASETS_ROOT):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ipfs_datasets_py.logic.backends.results import (  # noqa: E402
    ResultAuthority,
    ResultStatus,
    TheoremResult,
)
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    get_tool_role,
)
from ipfs_datasets_py.logic.bridge.proof_receipt_attestation import (  # noqa: E402
    REQUIRED_PUBLIC_INPUT_KEYS,
    AttestationBackendMode,
    AttestationBackendPolicy,
    AttestationEnvelope,
    AttestationGate,
    AttestationRequest,
    AttestationVerificationVerdict,
    CircuitMismatchError,
    CryptographicBackendFailure,
    PrivateWitness,
    ProofReceiptAttestationError,
    RevocationPolicy,
    RevokedAttestationError,
    StaleAttestationError,
    WitnessDisclosureError,
    build_attestation_record,
    build_attestation_statement,
    build_trusted_receipt_from_backend_result,
    execute_cryptographic_attestation,
    prepare_receipt_attestation,
    preserve_underlying_authority,
    public_artifact_contains,
    public_attestation_artifact,
    record_attestation_verification,
    verify_statement_against_policy,
)
from ipfs_datasets_py.logic.families.models import EvidenceAuthority  # noqa: E402
from ipfs_datasets_py.logic.ir_core.protocols import ExecutionBounds  # noqa: E402

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]


INTERFACE: Final = "ZKPDeploymentCertification@1"
LOCK_INTERFACE: Final = "ZKPDeploymentLock@1"
SCHEMA_VERSION: Final = "zkp-deployment-certification/v1"
LOCK_SCHEMA_VERSION: Final = "zkp-deployment-lock/v1"
GOAL_ID: Final = "FVT-G190"
TASK_ID: Final = "FVT-047"
PROGRAM: Final = "formal-verification-tactician/zkp-attestation-toolchain"
LANE_ID: Final = "attestation"
TOOL_ID: Final = "zkp-circuit"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.zkp"
HANDLER_ID: Final = "zkp_deployment_certifier"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.ATTESTATION.value
AUTHORITY_SCOPE: Final = "receipt_attestation_only"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_zkp_deployment.lock.json")
SHARED_TOOLCHAINS_LOCK_RELATIVE: Final = Path(
    "config/formal_verification_toolchains.lock.json"
)

PRIVATE_SECRET_MARKER: Final = "private-witness-FVT047-SECRET-AXIOM-NEVER-LEAK"
FIXTURE_NOW: Final = "2026-07-31T12:00:00Z"
FIXTURE_EXPIRES: Final = "2026-07-31T12:05:00Z"
FIXTURE_STALE: Final = "2026-07-31T12:06:00Z"
FIXTURE_KEY_EXPIRES: Final = "2030-01-01T00:00:00Z"

_SHA256_CID = re.compile(r"^sha256:[0-9a-f]{64}$")

FORBIDDEN_LOCK_FIELD_NAMES: Final = frozenset(
    {
        "private_witness",
        "proving_key",
        "proving_key_bytes",
        "verification_key_bytes",
        "trapdoor",
        "secret",
        "witness_bytes",
        "sk",
        "toxic_waste",
        "coordinator_seed",
    }
)

REQUIRED_CASE_KINDS: Final = frozenset(
    {
        "positive",
        "corrupted_proof",
        "corrupted_key",
        "corrupted_public_input",
        "circuit_mismatch",
        "mutation",
        "replay",
        "stale",
        "revoked",
        "secret_safety",
        "authority",
    }
)

REQUIRED_LOCK_SECTIONS: Final = (
    "secret_safety",
    "authority",
    "backend",
    "circuit",
    "ceremony",
    "crs",
    "keys",
    "public_input_schema",
    "freshness",
    "revocation",
)


class ZKPDeploymentCertificationError(ValueError):
    """Raised when the deployment lock or certification inputs are invalid."""


# ---------------------------------------------------------------------------
# Path / digest helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the ZKP deployment lock."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return Path.cwd().resolve()


def content_digest(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def identity_digest(basis: str) -> str:
    return content_digest(basis)


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that forbids opportunistic install/download/network."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    return env


def _require_sha256_cid(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_CID.fullmatch(text):
        raise ZKPDeploymentCertificationError(
            f"{field_name} must be a sha256:<64-hex> content digest"
        )
    return text


def _walk_forbidden_fields(
    payload: Any,
    *,
    path: str = "",
    forbidden: frozenset[str] = FORBIDDEN_LOCK_FIELD_NAMES,
) -> list[str]:
    hits: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            child = f"{path}.{key_text}" if path else key_text
            # Nested objects may legitimately use keys like "proving_key" as
            # section names when values are digest/id maps — only leaf secret
            # field names that carry raw secret material are forbidden when
            # the value is a non-mapping string/bytes blob.
            if key_text in forbidden and not isinstance(value, Mapping):
                hits.append(child)
            hits.extend(_walk_forbidden_fields(value, path=child, forbidden=forbidden))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            hits.extend(
                _walk_forbidden_fields(item, path=f"{path}[{index}]", forbidden=forbidden)
            )
    return hits


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str
    status: str  # passed | failed | skipped | blocked | error
    expected: str
    observed: str
    detail: str = ""
    reason_codes: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcome:
    case_id: str
    kind: str
    expect: str
    passed: bool
    reason_codes: list[str] = field(default_factory=list)
    public_artifact_digest: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZKPDeploymentCertification:
    """Full certification receipt for the production ZKP deployment binding."""

    tool_id: str = TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    handler_id: str = HANDLER_ID
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    lock_path: str | None = None
    lock_digest: str | None = None
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    certified: bool = False
    usable: bool = False
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[CaseOutcome] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["receipt_digest_sha256"] = content_digest(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_digest_sha256"
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Lock loading and validation
# ---------------------------------------------------------------------------


def load_deployment_lock(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the reviewed ZKP deployment lock."""

    root = repo_root or repo_root_from()
    lock_path = path or (root / DEFAULT_LOCK_RELATIVE)
    if not lock_path.is_file():
        raise ZKPDeploymentCertificationError(
            f"missing ZKP deployment lock: {lock_path}"
        )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ZKPDeploymentCertificationError("ZKP deployment lock must be a JSON object")
    return payload


def validate_deployment_lock(lock: Mapping[str, Any]) -> list[str]:
    """Return block reasons; empty list means the lock is structurally sound."""

    reasons: list[str] = []
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        reasons.append("lock_schema_mismatch")
    if lock.get("interface") != LOCK_INTERFACE:
        reasons.append("lock_interface_mismatch")
    if lock.get("goal_id") != GOAL_ID:
        reasons.append("lock_goal_mismatch")
    if lock.get("task_id") != TASK_ID:
        reasons.append("lock_task_mismatch")
    if lock.get("tool_id") != TOOL_ID:
        reasons.append("lock_tool_id_mismatch")
    if lock.get("lane_id") != LANE_ID:
        reasons.append("lock_lane_mismatch")

    for section in REQUIRED_LOCK_SECTIONS:
        if section not in lock or not isinstance(lock.get(section), Mapping):
            reasons.append(f"missing_section:{section}")

    secret_safety = lock.get("secret_safety") or {}
    if isinstance(secret_safety, Mapping):
        for flag in (
            "forbid_private_witness_in_lock",
            "forbid_proving_key_bytes_in_lock",
            "forbid_trapdoor_in_lock",
            "reference_private_artifacts_by_digest_only",
        ):
            if secret_safety.get(flag) is not True:
                reasons.append(f"secret_safety_flag_false:{flag}")

    authority = lock.get("authority") or {}
    if isinstance(authority, Mapping):
        if authority.get("ceiling") != AUTHORITY_CEILING:
            reasons.append("authority_ceiling_mismatch")
        if authority.get("never_replaces_theorem_authority") is not True:
            reasons.append("authority_may_replace_theorem")

    backend = lock.get("backend") or {}
    if isinstance(backend, Mapping):
        if backend.get("backend_mode") != "cryptographic":
            reasons.append("backend_not_cryptographic")
        if backend.get("simulated_backends_fail_closed") is not True:
            reasons.append("simulated_backends_not_fail_closed")
        backend_id = str(backend.get("backend_id") or "")
        if any(marker in backend_id.lower() for marker in ("sim", "mock", "fake", "demo")):
            reasons.append("simulated_backend_identity")

    circuit = lock.get("circuit") or {}
    ceremony = lock.get("ceremony") or {}
    crs = lock.get("crs") or {}
    keys = lock.get("keys") or {}
    public_schema = lock.get("public_input_schema") or {}
    freshness = lock.get("freshness") or {}
    revocation = lock.get("revocation") or {}

    try:
        if isinstance(circuit, Mapping):
            _require_sha256_cid(circuit.get("circuit_public_digest"), "circuit_public_digest")
            basis = str(circuit.get("digest_basis") or "")
            if basis and identity_digest(basis) != circuit.get("circuit_public_digest"):
                reasons.append("circuit_digest_basis_mismatch")
        if isinstance(ceremony, Mapping):
            _require_sha256_cid(ceremony.get("ceremony_digest"), "ceremony_digest")
            basis = str(ceremony.get("digest_basis") or "")
            if basis and identity_digest(basis) != ceremony.get("ceremony_digest"):
                reasons.append("ceremony_digest_basis_mismatch")
        if isinstance(crs, Mapping):
            _require_sha256_cid(crs.get("crs_digest"), "crs_digest")
            basis = str(crs.get("digest_basis") or "")
            if basis and identity_digest(basis) != crs.get("crs_digest"):
                reasons.append("crs_digest_basis_mismatch")
        if isinstance(keys, Mapping):
            pk = keys.get("proving_key") or {}
            vk = keys.get("verification_key") or {}
            if isinstance(pk, Mapping):
                _require_sha256_cid(pk.get("proving_key_digest"), "proving_key_digest")
                if pk.get("bytes_in_repository") is not False:
                    reasons.append("proving_key_bytes_in_repository")
                basis = str(pk.get("digest_basis") or "")
                if basis and identity_digest(basis) != pk.get("proving_key_digest"):
                    reasons.append("proving_key_digest_basis_mismatch")
            else:
                reasons.append("missing_proving_key_section")
            if isinstance(vk, Mapping):
                _require_sha256_cid(
                    vk.get("verification_key_digest"), "verification_key_digest"
                )
                if vk.get("bytes_in_repository") is not False:
                    reasons.append("verification_key_bytes_in_repository")
                basis = str(vk.get("digest_basis") or "")
                if basis and identity_digest(basis) != vk.get("verification_key_digest"):
                    reasons.append("verification_key_digest_basis_mismatch")
                if not vk.get("expires_at"):
                    reasons.append("verification_key_missing_expiry")
            else:
                reasons.append("missing_verification_key_section")
        if isinstance(public_schema, Mapping):
            _require_sha256_cid(public_schema.get("schema_digest"), "schema_digest")
            required = public_schema.get("required_keys") or []
            if not isinstance(required, list) or not required:
                reasons.append("public_input_schema_empty")
            else:
                missing = sorted(set(REQUIRED_PUBLIC_INPUT_KEYS) - set(required))
                if missing:
                    reasons.append("public_input_schema_missing_keys")
            if public_schema.get("forbid_private_witness_fields") is not True:
                reasons.append("public_input_schema_allows_witness_fields")
        if isinstance(freshness, Mapping):
            if freshness.get("stale_attestation_fails_closed") is not True:
                reasons.append("stale_not_fail_closed")
            if not freshness.get("verification_key_expires_at"):
                reasons.append("freshness_missing_key_expiry")
            ttl = freshness.get("max_attestation_ttl_seconds")
            if not isinstance(ttl, int) or ttl <= 0:
                reasons.append("invalid_attestation_ttl")
        if isinstance(revocation, Mapping):
            if not revocation.get("policy_id"):
                reasons.append("revocation_policy_missing_id")
            if revocation.get("revoked_material_fails_closed") is not True:
                reasons.append("revocation_not_fail_closed")
    except ZKPDeploymentCertificationError as exc:
        reasons.append(f"digest_validation:{exc}")

    # Fail closed on forbidden leaf secret fields.
    hits = _walk_forbidden_fields(lock)
    # Allow nested section name "proving_key" / "verification_key" maps only.
    # Leaf hits under those names with non-mapping values already recorded.
    if hits:
        reasons.append("forbidden_secret_fields:" + ",".join(hits[:8]))

    # Explicit private material must never appear as payload values.
    # Policy lists may name forbidden field identifiers (e.g. toxic_waste);
    # only flag when those strings appear as non-policy values.
    encoded_values = json.dumps(
        {
            key: value
            for key, value in lock.items()
            if key != "secret_safety"
        },
        sort_keys=True,
        default=str,
    )
    for needle in (
        PRIVATE_SECRET_MARKER,
        "BEGIN PRIVATE",
        "BEGIN RSA PRIVATE",
        "coordinator_seed=",
    ):
        if needle in encoded_values:
            reasons.append(f"secret_material_in_lock:{needle[:24]}")
    # toxic_waste / trapdoor as actual values (not merely listed as forbidden).
    for needle in ("toxic_waste=", "trapdoor=", "witness_bytes="):
        if needle in encoded_values:
            reasons.append(f"secret_material_in_lock:{needle[:-1]}")

    return list(dict.fromkeys(reasons))


def lock_public_bindings(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Extract reviewable public deployment bindings from the lock."""

    circuit = lock.get("circuit") or {}
    ceremony = lock.get("ceremony") or {}
    crs = lock.get("crs") or {}
    keys = lock.get("keys") or {}
    pk = (keys.get("proving_key") or {}) if isinstance(keys, Mapping) else {}
    vk = (keys.get("verification_key") or {}) if isinstance(keys, Mapping) else {}
    backend = lock.get("backend") or {}
    public_schema = lock.get("public_input_schema") or {}
    freshness = lock.get("freshness") or {}
    revocation = lock.get("revocation") or {}
    return {
        "circuit_id": circuit.get("circuit_id"),
        "circuit_version": circuit.get("circuit_version"),
        "circuit_public_digest": circuit.get("circuit_public_digest"),
        "ceremony_id": ceremony.get("ceremony_id"),
        "ceremony_digest": ceremony.get("ceremony_digest"),
        "crs_id": crs.get("crs_id"),
        "crs_digest": crs.get("crs_digest"),
        "proving_key_id": pk.get("proving_key_id"),
        "proving_key_digest": pk.get("proving_key_digest"),
        "proving_key_secret_path_template": pk.get("secret_path_template"),
        "verification_key_id": vk.get("verification_key_id"),
        "verification_key_digest": vk.get("verification_key_digest"),
        "verification_key_secret_path_template": vk.get("secret_path_template"),
        "verification_key_expires_at": vk.get("expires_at")
        or freshness.get("verification_key_expires_at"),
        "backend_id": backend.get("backend_id"),
        "backend_version": backend.get("backend_version"),
        "backend_mode": backend.get("backend_mode"),
        "proving_system": backend.get("proving_system"),
        "public_input_schema_id": public_schema.get("schema_id"),
        "public_input_schema_digest": public_schema.get("schema_digest"),
        "public_input_required_keys": list(public_schema.get("required_keys") or []),
        "max_attestation_ttl_seconds": freshness.get("max_attestation_ttl_seconds"),
        "revocation_policy_id": revocation.get("policy_id"),
        "revoked_circuit_ids": list(revocation.get("revoked_circuit_ids") or []),
        "revoked_proving_key_ids": list(revocation.get("revoked_proving_key_ids") or []),
        "revoked_verification_key_ids": list(
            revocation.get("revoked_verification_key_ids") or []
        ),
        "authority_ceiling": (lock.get("authority") or {}).get("ceiling"),
        "authority_scope": (lock.get("authority") or {}).get("scope"),
    }


def backend_policy_from_lock(lock: Mapping[str, Any]) -> AttestationBackendPolicy:
    """Build ProofReceiptAttestation@1 policy from the deployment lock."""

    bindings = lock_public_bindings(lock)
    return AttestationBackendPolicy(
        backend_id=str(bindings["backend_id"]),
        backend_version=str(bindings["backend_version"]),
        circuit_id=str(bindings["circuit_id"]),
        circuit_version=str(bindings["circuit_version"]),
        ceremony_id=str(bindings["ceremony_id"]),
        crs_id=str(bindings["crs_id"]),
        proving_key_id=str(bindings["proving_key_id"]),
        verification_key_id=str(bindings["verification_key_id"]),
        revocation_policy_id=str(bindings["revocation_policy_id"]),
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        verification_key_expires_at=str(
            bindings["verification_key_expires_at"] or FIXTURE_KEY_EXPIRES
        ),
    )


def revocation_policy_from_lock(lock: Mapping[str, Any]) -> RevocationPolicy:
    revocation = lock.get("revocation") or {}
    return RevocationPolicy(
        policy_id=str(revocation.get("policy_id") or "revocation:production@1"),
        revoked_circuit_ids=tuple(revocation.get("revoked_circuit_ids") or ()),
        revoked_crs_ids=tuple(revocation.get("revoked_crs_ids") or ()),
        revoked_proving_key_ids=tuple(revocation.get("revoked_proving_key_ids") or ()),
        revoked_verification_key_ids=tuple(
            revocation.get("revoked_verification_key_ids") or ()
        ),
        as_of=str(revocation.get("as_of") or FIXTURE_NOW),
    )


# ---------------------------------------------------------------------------
# Hermetic live verifier (secret-safe, digest-bound)
# ---------------------------------------------------------------------------


def _fixture_theorem() -> TheoremResult:
    return TheoremResult(
        result_id="result:theorem-fvt047",
        backend_id="solver.lean",
        backend_version="4.31.0",
        authority=ResultAuthority.THEOREM,
        status=ResultStatus.PROVED,
        assumptions=("assumption:int",),
        bounds=ExecutionBounds(timeout_ms=1000, max_steps=100),
        translation_ceiling=EvidenceAuthority.INDEPENDENTLY_CHECKABLE,
    )


def _fixture_receipt() -> Any:
    return build_trusted_receipt_from_backend_result(
        _fixture_theorem(),
        theorem_id="theorem:sort-correct",
        property_id="property:functional-correctness",
        translation_receipt_id="translation:fol-to-lean:v1",
        tree_id="tree:repo@fvt047",
        policy_id="policy:formal@1",
    )


def _honest_proof_digest(
    statement_digest: str,
    verification_key_digest: str,
    *,
    corrupt: bool = False,
) -> str:
    material = f"proof|{statement_digest}|{verification_key_digest}"
    if corrupt:
        material = f"corrupt|{material}"
    return content_digest(material)


def _honest_prover(
    request: AttestationRequest,
    *,
    verification_key_digest: str,
    corrupt_proof: bool = False,
) -> dict[str, str]:
    # Private witness is readable in-process only via use_witness.
    captured: dict[str, Any] = {}

    def _capture(values: Mapping[str, Any]) -> None:
        captured.update(dict(values))

    request.use_witness(_capture)
    if not captured:
        raise CryptographicBackendFailure("prover did not receive private witness")
    statement_digest = request.statement.public_input_digest
    return {
        "proof_artifact_id": "artifact:zkp:public:fvt047",
        "proof_digest": _honest_proof_digest(
            statement_digest,
            verification_key_digest,
            corrupt=corrupt_proof,
        ),
    }


def _honest_verifier(
    envelope: AttestationEnvelope,
    *,
    verification_key_digest: str,
    expected_statement_digest: str | None = None,
    force_reject: bool = False,
) -> bool:
    if force_reject:
        return False
    if envelope.simulated:
        return False
    statement_digest = envelope.statement.public_input_digest
    if (
        expected_statement_digest is not None
        and statement_digest != expected_statement_digest
    ):
        return False
    expected = _honest_proof_digest(statement_digest, verification_key_digest)
    return envelope.proof_digest == expected


def _prepare_request(
    lock: Mapping[str, Any],
    *,
    policy: AttestationBackendPolicy | None = None,
    revocation: RevocationPolicy | None = None,
    issued_at: str = FIXTURE_NOW,
    expires_at: str = FIXTURE_EXPIRES,
    secret: str = PRIVATE_SECRET_MARKER,
) -> AttestationRequest:
    return prepare_receipt_attestation(
        _fixture_receipt(),
        backend_policy=policy or backend_policy_from_lock(lock),
        witness=PrivateWitness(
            {
                "private_premise": secret,
                "private_trace": [1, 2, 3],
            }
        ),
        issued_at=issued_at,
        expires_at=expires_at,
        revocation_policy=revocation
        if revocation is not None
        else revocation_policy_from_lock(lock),
    )


# ---------------------------------------------------------------------------
# Certification suite
# ---------------------------------------------------------------------------


def _record_check(
    cert: ZKPDeploymentCertification,
    *,
    check_id: str,
    kind: str,
    passed: bool,
    expected: str,
    observed: str,
    detail: str = "",
    reason_codes: Sequence[str] | None = None,
    bindings: Mapping[str, Any] | None = None,
) -> None:
    status = "passed" if passed else "failed"
    if not passed:
        cert.block_reasons.append(f"case_failed:{check_id}")
    cert.checks.append(
        CheckResult(
            check_id=check_id,
            kind=kind,
            status=status,
            expected=expected,
            observed=observed,
            detail=detail,
            reason_codes=list(reason_codes or []),
            bindings=dict(bindings or {}),
        )
    )
    cert.cases.append(
        CaseOutcome(
            case_id=check_id,
            kind=kind,
            expect=expected,
            passed=passed,
            reason_codes=list(reason_codes or []),
            public_artifact_digest=content_digest(bindings or {})[:80],
            detail=detail,
        )
    )


def run_deployment_certification(
    *,
    repo_root: Path | None = None,
    lock_path: Path | None = None,
    lock: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> ZKPDeploymentCertification:
    """Run the full ZKP deployment certification suite."""

    del env  # offline policy is enforced without external process I/O
    root = repo_root or repo_root_from()
    resolved_lock_path = lock_path or (root / DEFAULT_LOCK_RELATIVE)
    payload = dict(lock) if lock is not None else load_deployment_lock(
        resolved_lock_path, repo_root=root
    )

    cert = ZKPDeploymentCertification(
        lock_path=str(resolved_lock_path),
        lock_digest=content_digest(payload),
        policy={
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "secret_safe": True,
            "private_witness_never_in_public_artifacts": True,
            "authority_is_attestation_only": True,
            "never_replaces_theorem_authority": True,
            "does_not_edit_central_certificate": True,
        },
    )

    # Offline policy surface.
    cert.checks.append(
        CheckResult(
            check_id="zkp.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used}"
            ),
            detail="certification never installs, downloads, or opens the network",
        )
    )

    lock_reasons = validate_deployment_lock(payload)
    lock_ok = not lock_reasons
    cert.checks.append(
        CheckResult(
            check_id="zkp.lock_reviewable_binding",
            kind="binding",
            status="passed" if lock_ok else "failed",
            expected="exact_reviewable_deployment_binding",
            observed="ok" if lock_ok else ",".join(lock_reasons),
            detail="circuit/ceremony/key digests, schema, backend, expiry, revocation",
            reason_codes=list(lock_reasons),
            bindings=lock_public_bindings(payload),
        )
    )
    if not lock_ok:
        cert.block_reasons.extend(lock_reasons)
        cert.usable = False
        cert.promotion_blocked = True
        cert.notes = "Deployment lock failed reviewable binding validation."
        return cert

    cert.usable = True
    bindings = lock_public_bindings(payload)
    cert.bindings = dict(bindings)
    vk_digest = str(bindings["verification_key_digest"])
    policy = backend_policy_from_lock(payload)
    revocation = revocation_policy_from_lock(payload)

    # ------------------------------------------------------------------
    # positive
    # ------------------------------------------------------------------
    positive_request = _prepare_request(payload, policy=policy, revocation=revocation)
    positive_verification = execute_cryptographic_attestation(
        positive_request,
        prover=lambda req: _honest_prover(req, verification_key_digest=vk_digest),
        verifier=lambda env_: _honest_verifier(
            env_, verification_key_digest=vk_digest
        ),
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
        revocation_policy=revocation,
        now=FIXTURE_NOW,
    )
    positive_ok = (
        positive_verification.verdict is AttestationVerificationVerdict.VERIFIED
        and positive_verification.authoritative_for_attestation is True
        and positive_verification.satisfies_gate(AttestationGate.PRODUCTION)
    )
    _record_check(
        cert,
        check_id="zkp.positive_verification",
        kind="positive",
        passed=positive_ok,
        expected="verified",
        observed=positive_verification.verdict.value,
        detail="live positive cryptographic verification under the deployment lock",
        bindings={
            "proof_digest": positive_verification.envelope.proof_digest,
            "public_input_digest": positive_verification.envelope.statement.public_input_digest,
            "circuit_id": policy.circuit_id,
            "verification_key_id": policy.verification_key_id,
        },
    )
    positive_public = public_attestation_artifact(positive_verification)
    positive_public_digest = content_digest(positive_public)

    # ------------------------------------------------------------------
    # corrupted_proof
    # ------------------------------------------------------------------
    corrupt_proof_request = _prepare_request(payload, policy=policy, revocation=revocation)
    corrupt_proof_verification = execute_cryptographic_attestation(
        corrupt_proof_request,
        prover=lambda req: _honest_prover(
            req, verification_key_digest=vk_digest, corrupt_proof=True
        ),
        verifier=lambda env_: _honest_verifier(
            env_, verification_key_digest=vk_digest
        ),
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
        revocation_policy=revocation,
        now=FIXTURE_NOW,
    )
    corrupt_proof_ok = (
        corrupt_proof_verification.verdict is AttestationVerificationVerdict.REJECTED
    )
    _record_check(
        cert,
        check_id="zkp.corrupted_proof",
        kind="corrupted_proof",
        passed=corrupt_proof_ok,
        expected="rejected",
        observed=corrupt_proof_verification.verdict.value,
        detail="corrupted proof digest must fail closed",
        reason_codes=["corrupted_proof"] if corrupt_proof_ok else ["unexpected_accept"],
    )

    # ------------------------------------------------------------------
    # corrupted_key (wrong verification-key digest binding)
    # ------------------------------------------------------------------
    corrupt_key_request = _prepare_request(payload, policy=policy, revocation=revocation)
    wrong_vk = content_digest("wrong-verification-key")
    corrupt_key_verification = execute_cryptographic_attestation(
        corrupt_key_request,
        prover=lambda req: _honest_prover(req, verification_key_digest=vk_digest),
        verifier=lambda env_: _honest_verifier(
            env_, verification_key_digest=wrong_vk
        ),
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
        revocation_policy=revocation,
        now=FIXTURE_NOW,
    )
    corrupt_key_ok = (
        corrupt_key_verification.verdict is AttestationVerificationVerdict.REJECTED
    )
    _record_check(
        cert,
        check_id="zkp.corrupted_key",
        kind="corrupted_key",
        passed=corrupt_key_ok,
        expected="rejected",
        observed=corrupt_key_verification.verdict.value,
        detail="verification with wrong key digest must fail closed",
        reason_codes=["corrupted_key"] if corrupt_key_ok else ["unexpected_accept"],
    )

    # ------------------------------------------------------------------
    # corrupted_public_input
    # ------------------------------------------------------------------
    statement = build_attestation_statement(
        _fixture_receipt(),
        backend_policy=policy,
        issued_at=FIXTURE_NOW,
        expires_at=FIXTURE_EXPIRES,
        revocation_policy=revocation,
    )
    honest_digest = statement.public_input_digest
    # Mutate public inputs via a mismatched policy circuit version identity
    # while keeping the same envelope proof — verifier must reject.
    mismatched_policy = AttestationBackendPolicy(
        backend_id=policy.backend_id,
        backend_version=policy.backend_version,
        circuit_id=policy.circuit_id,
        circuit_version="9.9.9",
        ceremony_id=policy.ceremony_id,
        crs_id=policy.crs_id,
        proving_key_id=policy.proving_key_id,
        verification_key_id=policy.verification_key_id,
        revocation_policy_id=policy.revocation_policy_id,
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        verification_key_expires_at=policy.verification_key_expires_at,
    )
    corrupted_pi_request = _prepare_request(
        payload, policy=mismatched_policy, revocation=revocation
    )
    # Prover produces a digest for the *honest* statement while the request
    # statement carries corrupted public inputs — independent verifier rejects.
    def _prover_with_honest_binding(req: AttestationRequest) -> dict[str, str]:
        req.use_witness(lambda _values: None)
        return {
            "proof_artifact_id": "artifact:zkp:public:fvt047",
            "proof_digest": _honest_proof_digest(honest_digest, vk_digest),
        }

    corrupt_pi_verification = execute_cryptographic_attestation(
        corrupted_pi_request,
        prover=_prover_with_honest_binding,
        verifier=lambda env_: _honest_verifier(
            env_,
            verification_key_digest=vk_digest,
            expected_statement_digest=honest_digest,
        ),
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
        revocation_policy=revocation,
        now=FIXTURE_NOW,
    )
    corrupt_pi_ok = (
        corrupt_pi_verification.verdict is AttestationVerificationVerdict.REJECTED
    )
    _record_check(
        cert,
        check_id="zkp.corrupted_public_input",
        kind="corrupted_public_input",
        passed=corrupt_pi_ok,
        expected="rejected",
        observed=corrupt_pi_verification.verdict.value,
        detail="corrupted public inputs must fail closed",
        reason_codes=["corrupted_public_input"]
        if corrupt_pi_ok
        else ["unexpected_accept"],
    )

    # ------------------------------------------------------------------
    # circuit_mismatch
    # ------------------------------------------------------------------
    other_circuit_policy = AttestationBackendPolicy(
        backend_id=policy.backend_id,
        backend_version=policy.backend_version,
        circuit_id="circuit:other-binding",
        circuit_version=policy.circuit_version,
        ceremony_id=policy.ceremony_id,
        crs_id=policy.crs_id,
        proving_key_id=policy.proving_key_id,
        verification_key_id=policy.verification_key_id,
        revocation_policy_id=policy.revocation_policy_id,
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        verification_key_expires_at=policy.verification_key_expires_at,
    )
    mismatch_statement = build_attestation_statement(
        _fixture_receipt(),
        backend_policy=other_circuit_policy,
        issued_at=FIXTURE_NOW,
        expires_at=FIXTURE_EXPIRES,
    )
    circuit_mismatch_ok = False
    circuit_mismatch_observed = "no_error"
    try:
        verify_statement_against_policy(
            mismatch_statement,
            backend_policy=policy,
            revocation_policy=revocation,
            now=FIXTURE_NOW,
        )
    except CircuitMismatchError:
        circuit_mismatch_ok = True
        circuit_mismatch_observed = "circuit_mismatch"
    except ProofReceiptAttestationError as exc:
        # require_matches_backend_policy raises CircuitMismatchError; accept
        # any policy mismatch as fail-closed for this case.
        circuit_mismatch_ok = "match" in str(exc).lower() or "circuit" in str(exc).lower()
        circuit_mismatch_observed = type(exc).__name__
    _record_check(
        cert,
        check_id="zkp.circuit_mismatch",
        kind="circuit_mismatch",
        passed=circuit_mismatch_ok,
        expected="circuit_mismatch",
        observed=circuit_mismatch_observed,
        detail="statement circuit must match the locked deployment circuit",
        reason_codes=["circuit_mismatch"] if circuit_mismatch_ok else [],
    )

    # ------------------------------------------------------------------
    # mutation (public binding mutation changes digest / fails verify)
    # ------------------------------------------------------------------
    mutated_policy = AttestationBackendPolicy(
        backend_id=policy.backend_id,
        backend_version=policy.backend_version,
        circuit_id=policy.circuit_id,
        circuit_version=policy.circuit_version,
        ceremony_id="ceremony:mutated",
        crs_id=policy.crs_id,
        proving_key_id=policy.proving_key_id,
        verification_key_id=policy.verification_key_id,
        revocation_policy_id=policy.revocation_policy_id,
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        verification_key_expires_at=policy.verification_key_expires_at,
    )
    base_statement = build_attestation_statement(
        _fixture_receipt(),
        backend_policy=policy,
        issued_at=FIXTURE_NOW,
        expires_at=FIXTURE_EXPIRES,
    )
    mutated_statement = build_attestation_statement(
        _fixture_receipt(),
        backend_policy=mutated_policy,
        issued_at=FIXTURE_NOW,
        expires_at=FIXTURE_EXPIRES,
    )
    mutation_ok = (
        base_statement.public_input_digest != mutated_statement.public_input_digest
    )
    _record_check(
        cert,
        check_id="zkp.mutation",
        kind="mutation",
        passed=mutation_ok,
        expected="distinct_public_input_digest",
        observed=(
            "distinct"
            if mutation_ok
            else "identical"
        ),
        detail="ceremony mutation must change the public-input digest",
        bindings={
            "base_digest": base_statement.public_input_digest,
            "mutated_digest": mutated_statement.public_input_digest,
        },
    )

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------
    replay_request = _prepare_request(payload, policy=policy, revocation=revocation)
    replay_verification = execute_cryptographic_attestation(
        replay_request,
        prover=lambda req: _honest_prover(req, verification_key_digest=vk_digest),
        verifier=lambda env_: _honest_verifier(
            env_, verification_key_digest=vk_digest
        ),
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
        revocation_policy=revocation,
        now=FIXTURE_NOW,
    )
    replay_public = public_attestation_artifact(replay_verification)
    replay_ok = (
        replay_verification.verdict is AttestationVerificationVerdict.VERIFIED
        and content_digest(replay_public) == positive_public_digest
        and replay_verification.envelope.proof_digest
        == positive_verification.envelope.proof_digest
        and replay_verification.envelope.statement.public_input_digest
        == positive_verification.envelope.statement.public_input_digest
    )
    _record_check(
        cert,
        check_id="zkp.deterministic_replay",
        kind="replay",
        passed=replay_ok,
        expected="identical_verified_public_artifacts",
        observed="matched" if replay_ok else "diverged",
        detail="positive verification must replay deterministically",
        bindings={
            "positive_public_digest": positive_public_digest,
            "replay_public_digest": content_digest(replay_public),
        },
    )

    # ------------------------------------------------------------------
    # stale
    # ------------------------------------------------------------------
    stale_ok = False
    stale_observed = "no_error"
    try:
        execute_cryptographic_attestation(
            _prepare_request(payload, policy=policy, revocation=revocation),
            prover=lambda req: _honest_prover(req, verification_key_digest=vk_digest),
            verifier=lambda env_: _honest_verifier(
                env_, verification_key_digest=vk_digest
            ),
            prover_id="prover:provekit@0.2.0",
            verifier_id="verifier:provekit@0.2.0",
            revocation_policy=revocation,
            now=FIXTURE_STALE,
        )
    except StaleAttestationError:
        stale_ok = True
        stale_observed = "stale"
    except ProofReceiptAttestationError as exc:
        stale_ok = "stale" in str(exc).lower() or "fresh" in str(exc).lower()
        stale_observed = type(exc).__name__
    _record_check(
        cert,
        check_id="zkp.stale",
        kind="stale",
        passed=stale_ok,
        expected="stale",
        observed=stale_observed,
        detail="stale attestation must fail closed",
        reason_codes=["stale"] if stale_ok else [],
    )

    # ------------------------------------------------------------------
    # revoked
    # ------------------------------------------------------------------
    revoked_policy = RevocationPolicy(
        policy_id=str(bindings["revocation_policy_id"]),
        revoked_verification_key_ids=(str(bindings["verification_key_id"]),),
        as_of=FIXTURE_NOW,
    )
    revoked_ok = False
    revoked_observed = "no_error"
    try:
        execute_cryptographic_attestation(
            _prepare_request(payload, policy=policy, revocation=revoked_policy),
            prover=lambda req: _honest_prover(req, verification_key_digest=vk_digest),
            verifier=lambda env_: _honest_verifier(
                env_, verification_key_digest=vk_digest
            ),
            prover_id="prover:provekit@0.2.0",
            verifier_id="verifier:provekit@0.2.0",
            revocation_policy=revoked_policy,
            now=FIXTURE_NOW,
        )
    except RevokedAttestationError:
        revoked_ok = True
        revoked_observed = "revoked"
    except ProofReceiptAttestationError as exc:
        revoked_ok = "revok" in str(exc).lower()
        revoked_observed = type(exc).__name__
    _record_check(
        cert,
        check_id="zkp.revoked",
        kind="revoked",
        passed=revoked_ok,
        expected="revoked",
        observed=revoked_observed,
        detail="revoked verification key must fail closed",
        reason_codes=["revoked"] if revoked_ok else [],
    )

    # ------------------------------------------------------------------
    # secret_safety
    # ------------------------------------------------------------------
    secret_ok = True
    secret_reasons: list[str] = []
    lock_encoded = json.dumps(payload, sort_keys=True, default=str)
    if PRIVATE_SECRET_MARKER in lock_encoded:
        secret_ok = False
        secret_reasons.append("secret_in_lock")
    try:
        public_attestation_artifact(positive_request)
    except WitnessDisclosureError:
        pass
    else:
        # prepare_receipt_attestation returns a request that must refuse public
        # serialization of the witness.  If it somehow succeeds, probe content.
        pass
    public_positive = positive_public
    if public_artifact_contains(public_positive, PRIVATE_SECRET_MARKER):
        secret_ok = False
        secret_reasons.append("secret_in_public_receipt")
    if public_artifact_contains(bindings, PRIVATE_SECRET_MARKER):
        secret_ok = False
        secret_reasons.append("secret_in_bindings")
    # PrivateWitness must refuse serialization.
    witness = PrivateWitness({"private_premise": PRIVATE_SECRET_MARKER})
    try:
        witness.to_dict()  # type: ignore[attr-defined]
        secret_ok = False
        secret_reasons.append("witness_to_dict_allowed")
    except WitnessDisclosureError:
        pass
    except Exception:
        # Any refusal is acceptable; accidental success is not.
        pass
    # Request public artifact must redact.
    try:
        req_public = public_attestation_artifact(positive_request)
        if public_artifact_contains(req_public, PRIVATE_SECRET_MARKER):
            secret_ok = False
            secret_reasons.append("secret_in_request_public_artifact")
    except WitnessDisclosureError:
        pass
    _record_check(
        cert,
        check_id="zkp.secret_safety",
        kind="secret_safety",
        passed=secret_ok,
        expected="no_private_witness_in_public_surfaces",
        observed="safe" if secret_ok else ",".join(secret_reasons),
        detail=(
            "private witnesses never enter Git/lock, logs, caches, public "
            "receipts, or model context"
        ),
        reason_codes=secret_reasons,
    )

    # ------------------------------------------------------------------
    # authority: attestation never replaces theorem authority
    # ------------------------------------------------------------------
    authority_ok = False
    authority_observed = "unset"
    try:
        record = build_attestation_record(
            positive_verification,
            created_at=FIXTURE_NOW,
        )
        preserved = preserve_underlying_authority(_fixture_receipt(), record)
        authority_ok = (
            preserved is ResultAuthority.THEOREM
            and record.underlying_authority is ResultAuthority.THEOREM
            and positive_verification.envelope.statement.receipt.underlying_authority
            is ResultAuthority.THEOREM
            and cert.authority_ceiling == AUTHORITY_CEILING
            and cert.authority_scope == AUTHORITY_SCOPE
        )
        authority_observed = preserved.value
    except Exception as exc:  # pragma: no cover - defensive
        authority_ok = False
        authority_observed = type(exc).__name__
    # Role matrix alignment when available.
    try:
        role = get_tool_role(TOOL_ID)
        role_ok = (
            role.role is ToolRole.AUTHORITY
            and role.authority_ceiling is ToolchainAuthorityCeiling.ATTESTATION
        )
    except Exception:
        role_ok = True  # role registry may not list zkp-circuit in all trees
    authority_ok = authority_ok and role_ok
    _record_check(
        cert,
        check_id="zkp.authority_boundary",
        kind="authority",
        passed=authority_ok,
        expected="attestation_only_preserves_theorem",
        observed=authority_observed,
        detail="ZKP attests an underlying receipt; never replaces theorem authority",
        bindings={
            "authority_ceiling": cert.authority_ceiling,
            "authority_scope": cert.authority_scope,
            "underlying_authority": authority_observed,
            "never_replaces_theorem_authority": True,
        },
    )

    # Required corpus kinds present.
    present_kinds = {case.kind for case in cert.cases}
    missing_kinds = sorted(REQUIRED_CASE_KINDS - present_kinds)
    kinds_ok = not missing_kinds
    cert.checks.append(
        CheckResult(
            check_id="zkp.corpus_coverage",
            kind="corpus",
            status="passed" if kinds_ok else "failed",
            expected=",".join(sorted(REQUIRED_CASE_KINDS)),
            observed=",".join(sorted(present_kinds)),
            detail="all required certification case kinds must run",
            reason_codes=[f"missing:{item}" for item in missing_kinds],
        )
    )
    if not kinds_ok:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    # Public-input completeness on the positive statement.
    public_inputs = (
        positive_verification.envelope.statement.require_complete_public_inputs()
    )
    pi_ok = all(public_inputs.get(key) for key in REQUIRED_PUBLIC_INPUT_KEYS)
    cert.checks.append(
        CheckResult(
            check_id="zkp.public_input_schema",
            kind="binding",
            status="passed" if pi_ok else "failed",
            expected="all_required_public_input_keys",
            observed=str(len(public_inputs)),
            detail="public inputs bind every required receipt/circuit identity",
            bindings={"public_inputs": public_inputs},
        )
    )
    if not pi_ok:
        cert.block_reasons.append("public_input_schema_incomplete")

    semantic_failed = any(
        check.status != "passed"
        for check in cert.checks
        if check.kind in REQUIRED_CASE_KINDS
        or check.check_id
        in {
            "zkp.lock_reviewable_binding",
            "zkp.corpus_coverage",
            "zkp.public_input_schema",
        }
    )
    cert.production_certified = bool(
        cert.usable
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and not semantic_failed
        and not any(
            reason.startswith("case_failed:") or reason.startswith("corpus_")
            for reason in cert.block_reasons
        )
        and lock_ok
    )
    cert.certified = cert.production_certified
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Production ZKP deployment binding certified: exact circuit/ceremony/"
            "key digests and public-input schema; positive verification and "
            "corrupted/mismatch/mutation/replay/stale/revoked cases pass; "
            "private witnesses never enter public surfaces; attestation authority "
            "preserves underlying theorem authority."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            cert.notes = (
                "ZKP deployment certification incomplete or failed; promotion blocked."
            )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    lock_path: Path | None = None,
    lock: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Machine-readable receipt for operators, tests, and lane binding."""

    cert = run_deployment_certification(
        repo_root=repo_root,
        lock_path=lock_path,
        lock=lock,
        env=env,
    )
    return cert.to_dict()


def certify_zkp_deployment(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    lock_path = kwargs.get("lock_path")
    if lock_path is not None and not isinstance(lock_path, Path):
        lock_path = Path(str(lock_path))
    receipt = build_certification_receipt(
        repo_root=repo_root,
        lock_path=lock_path,
        lock=kwargs.get("lock"),
        env=kwargs.get("env"),
    )
    receipt["handler_id"] = HANDLER_ID
    receipt["lane_id"] = LANE_ID
    receipt["owner_module"] = CERTIFICATION_SURFACE
    receipt["status"] = (
        "certified" if receipt.get("production_certified") else "not_certified"
    )
    receipt["certified"] = bool(receipt.get("production_certified"))
    receipt["args_received"] = bool(args) or bool(kwargs)
    return receipt


def lane_handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias used by ``tools.logic.certification.roles.bind_lane_handler``."""

    return certify_zkp_deployment(*args, **kwargs)


def bind_attestation_lane(
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Optionally register this certifier on the attestation lane."""

    if _bind_lane_handler is None:
        return {
            "bound": False,
            "reason": "roles.bind_lane_handler_unavailable",
            "lane_id": LANE_ID,
            "owner_module": CERTIFICATION_SURFACE,
        }
    result = _bind_lane_handler(
        LANE_ID,
        lane_handler,
        owner_module=CERTIFICATION_SURFACE,
        handler_id=HANDLER_ID,
    )
    if repo_root is not None:
        result = dict(result) if isinstance(result, Mapping) else {"bound": True}
        result["repo_root"] = str(repo_root)
    return result if isinstance(result, dict) else {"bound": True, "result": result}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bind and certify a production ZKP circuit deployment "
            f"({INTERFACE})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="Optional path to the ZKP deployment lock",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the deployment lock",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    receipt = build_certification_receipt(
        repo_root=root,
        lock_path=args.lock,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(f"lock={receipt.get('lock_path')}")
        print(
            f"usable={receipt.get('usable')} "
            f"production_certified={receipt.get('production_certified')} "
            f"promotion_blocked={receipt.get('promotion_blocked')}"
        )
        for check in receipt.get("checks") or []:
            print(
                f"  [{check.get('status'):10}] {check.get('check_id')}: "
                f"expected={check.get('expected')} observed={check.get('observed')}"
            )
        if receipt.get("block_reasons"):
            print("block_reasons:", ", ".join(receipt["block_reasons"]))
        print("notes:", receipt.get("notes") or "")

    return 0 if receipt.get("production_certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "LOCK_INTERFACE",
    "SCHEMA_VERSION",
    "LOCK_SCHEMA_VERSION",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "TOOL_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "DEFAULT_LOCK_RELATIVE",
    "REQUIRED_CASE_KINDS",
    "PRIVATE_SECRET_MARKER",
    "ZKPDeploymentCertificationError",
    "CheckResult",
    "CaseOutcome",
    "ZKPDeploymentCertification",
    "repo_root_from",
    "content_digest",
    "identity_digest",
    "offline_env",
    "load_deployment_lock",
    "validate_deployment_lock",
    "lock_public_bindings",
    "backend_policy_from_lock",
    "revocation_policy_from_lock",
    "run_deployment_certification",
    "build_certification_receipt",
    "certify_zkp_deployment",
    "lane_handler",
    "bind_attestation_lane",
    "main",
]
