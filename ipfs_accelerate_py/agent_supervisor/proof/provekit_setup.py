"""Exact ProveKit executable/setup/circuit/verifier identity and self-test gate.

SCA-G180 / SCAEV180PROOFREADY requires real ZK readiness to be content-addressed
and fail closed:

* absent ``provekit-cli`` or setup artifacts emit typed unavailable status;
* configured executable, setup, circuit, and verifier identities are digests;
* positive and negative self-tests are mandatory for production eligibility; and
* only an already kernel-verified approved receipt predicate may be attested.

Simulated ZK and hash-commitment placeholders remain non-attested.  This module
never installs tools and never claims arbitrary function-call or source-code
correctness from a ZK receipt alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .solver_readiness import (
    SCAEV180PROOFREADY,
    SCAEV180PROOFREADY_COVERAGE,
    SCAEV180PROOFREADY_EVIDENCE,
)


PROVEKIT_SETUP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provekit-setup@1"
)
PROVEKIT_SETUP_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provekit-setup-report@1"
)
PROVEKIT_ATTESTATION_ELIGIBILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provekit-attestation-eligibility@1"
)
PROVEKIT_SETUP_VERSION: Final = 1

# SCAEV180PROOFREADY / SCAEV180PROOFREADY_EVIDENCE / SCAEV180PROOFREADY_COVERAGE
# are re-exported from solver_readiness for AST/evidence scanners.

APPROVED_VERIFIED_RECEIPT_PREDICATE: Final = (
    "kernel_verified_receipt_predicate@1"
)
APPROVED_VERIFIED_RECEIPT_PREDICATES: Final = frozenset(
    {
        APPROVED_VERIFIED_RECEIPT_PREDICATE,
        "approved_verified_receipt_predicate",
        "kernel-verified-receipt",
        "kernel_verified_receipt",
    }
)

DEFAULT_PROVEKIT_EXECUTABLE_NAMES: Final = ("provekit-cli", "provekit")
DEFAULT_PROVER_KEY_PATTERNS: Final = ("*.pkp", "*prover*.key", "prover_key*")
DEFAULT_VERIFIER_KEY_PATTERNS: Final = ("*.pkv", "*verifier*.key", "verifier_key*")


class ProveKitSetupStatus(str, Enum):
    """Truthful ProveKit readiness states."""

    UNAVAILABLE = "unavailable"
    CONFIGURED = "configured"
    AVAILABLE = "available"
    VERIFIED = "verified"
    SIMULATED = "simulated"
    DEGRADED = "degraded"


class ProveKitSelfTestCase(str, Enum):
    """Mandatory positive and negative canaries for real ZK."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    MALFORMED_PROOF = "malformed_proof"
    WITNESS_NO_LEAK = "witness_no_leak"


REQUIRED_PROVEKIT_SELF_TESTS: Final = tuple(ProveKitSelfTestCase)


class ProveKitSelfTestVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_RUN = "not_run"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest_bytes(data: bytes, *, prefix: str) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(data).hexdigest()}"


def _digest_json(value: Any, *, prefix: str) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"), prefix=prefix)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _match_any(path: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    return None


@dataclass(frozen=True)
class ContentAddressedIdentity:
    """Content-addressed pin for one ProveKit surface artifact."""

    surface: str
    path: str
    digest: str
    present: bool
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "path": self.path,
            "digest": self.digest,
            "present": self.present,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ProveKitSelfTestResult:
    """One bounded positive/negative ProveKit canary result."""

    case: ProveKitSelfTestCase
    verdict: ProveKitSelfTestVerdict
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case",
            ProveKitSelfTestCase(str(getattr(self.case, "value", self.case))),
        )
        object.__setattr__(
            self,
            "verdict",
            ProveKitSelfTestVerdict(
                str(getattr(self.verdict, "value", self.verdict))
            ),
        )

    @property
    def passed(self) -> bool:
        return self.verdict is ProveKitSelfTestVerdict.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.value,
            "verdict": self.verdict.value,
            "reason_code": self.reason_code,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ProveKitSetupConfig:
    """Explicit paths and names; nothing is installed implicitly."""

    executable_path: str | None = None
    executable_names: tuple[str, ...] = DEFAULT_PROVEKIT_EXECUTABLE_NAMES
    executable_env_names: tuple[str, ...] = (
        "IPFS_DATASETS_PROVEKIT_BINARY",
        "PROVEKIT_CLI",
    )
    artifacts_path: str | None = None
    artifacts_env_names: tuple[str, ...] = (
        "IPFS_DATASETS_PROVEKIT_ARTIFACTS_DIR",
        "PROVEKIT_ARTIFACTS_DIR",
    )
    circuit_id: str = "circuit:provekit-receipt-binding"
    circuit_version: str = "1.0.0"
    public_input_schema_id: str = "schema:provekit-public-inputs"
    public_input_schema_version: str = "1.0.0"
    simulated: bool = False
    backend_version: str = "0.0.0"


@dataclass(frozen=True)
class ProveKitSetupReceipt:
    """Capability/setup receipt consumed by real datasets ZK attestation."""

    status: ProveKitSetupStatus
    configured: bool
    available: bool
    production_eligible: bool
    executable: ContentAddressedIdentity
    setup: ContentAddressedIdentity
    circuit: ContentAddressedIdentity
    verifier: ContentAddressedIdentity
    prover: ContentAddressedIdentity
    self_tests: tuple[ProveKitSelfTestResult, ...]
    circuit_id: str
    circuit_version: str
    public_input_schema_id: str
    public_input_schema_version: str
    backend_version: str
    simulated: bool
    reason_code: str
    reason: str
    proof_attempted: bool = field(default=False, init=False)
    proof_success: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            ProveKitSetupStatus(str(getattr(self.status, "value", self.status))),
        )
        if self.simulated and self.production_eligible:
            raise ValueError("simulated ProveKit cannot be production eligible")
        if self.production_eligible and self.status is not ProveKitSetupStatus.VERIFIED:
            raise ValueError(
                "production eligibility requires verified ProveKit setup status"
            )
        if self.available and not self.configured:
            raise ValueError("available ProveKit setup must also be configured")
        if self.proof_attempted or self.proof_success:
            raise ValueError("ProveKit setup receipts never attempt or succeed at proof")

    @property
    def setup_identity(self) -> str:
        return _digest_json(
            {
                "schema": PROVEKIT_SETUP_SCHEMA,
                "version": PROVEKIT_SETUP_VERSION,
                "status": self.status.value,
                "configured": self.configured,
                "available": self.available,
                "production_eligible": self.production_eligible,
                "executable": self.executable.to_dict(),
                "setup": self.setup.to_dict(),
                "circuit": self.circuit.to_dict(),
                "verifier": self.verifier.to_dict(),
                "prover": self.prover.to_dict(),
                "self_tests": [item.to_dict() for item in self.self_tests],
                "circuit_id": self.circuit_id,
                "circuit_version": self.circuit_version,
                "public_input_schema_id": self.public_input_schema_id,
                "public_input_schema_version": self.public_input_schema_version,
                "backend_version": self.backend_version,
                "simulated": self.simulated,
                "reason_code": self.reason_code,
            },
            prefix="provekit-setup",
        )

    @property
    def policy_material(self) -> dict[str, Any]:
        """Pinned identity set for attestation backend policy construction."""

        return {
            "backend_id": "backend:provekit" if not self.simulated else "backend:simulated",
            "backend_version": self.backend_version,
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "public_input_schema_id": self.public_input_schema_id,
            "public_input_schema_version": self.public_input_schema_version,
            "verification_key_id": self.verifier.digest or "vk:missing",
            "verification_key_version": self.verifier.digest or "missing",
            "executable_digest": self.executable.digest,
            "setup_digest": self.setup.digest,
            "circuit_digest": self.circuit.digest,
            "prover_digest": self.prover.digest,
            "simulated": self.simulated,
            "setup_identity": self.setup_identity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVEKIT_SETUP_SCHEMA,
            "version": PROVEKIT_SETUP_VERSION,
            "status": self.status.value,
            "configured": self.configured,
            "available": self.available,
            "production_eligible": self.production_eligible,
            "executable": self.executable.to_dict(),
            "setup": self.setup.to_dict(),
            "circuit": self.circuit.to_dict(),
            "verifier": self.verifier.to_dict(),
            "prover": self.prover.to_dict(),
            "self_tests": [item.to_dict() for item in self.self_tests],
            "circuit_id": self.circuit_id,
            "circuit_version": self.circuit_version,
            "public_input_schema_id": self.public_input_schema_id,
            "public_input_schema_version": self.public_input_schema_version,
            "backend_version": self.backend_version,
            "simulated": self.simulated,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "proof_attempted": False,
            "proof_success": False,
            "setup_identity": self.setup_identity,
            "policy_material": self.policy_material,
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
        }


@dataclass(frozen=True)
class ProveKitAttestationEligibility:
    """Gate: only approved verified-receipt predicates may be attested."""

    eligible: bool
    predicate_id: str
    setup: ProveKitSetupReceipt
    kernel_verified: bool
    kernel_receipt_id: str
    reason_code: str
    reason: str
    attested: bool = False

    def __post_init__(self) -> None:
        if self.attested and not self.eligible:
            raise ValueError("ineligible ProveKit predicates cannot be attested")
        if self.eligible and self.setup.simulated:
            raise ValueError("simulated ProveKit is never attestation-eligible")
        if self.eligible and not self.setup.production_eligible:
            raise ValueError(
                "attestation eligibility requires production-eligible ProveKit setup"
            )
        if self.eligible and not self.kernel_verified:
            raise ValueError(
                "attestation eligibility requires an already kernel-verified receipt"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVEKIT_ATTESTATION_ELIGIBILITY_SCHEMA,
            "eligible": self.eligible,
            "predicate_id": self.predicate_id,
            "setup": self.setup.to_dict(),
            "kernel_verified": self.kernel_verified,
            "kernel_receipt_id": self.kernel_receipt_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "attested": self.attested,
            "approved_predicate": APPROVED_VERIFIED_RECEIPT_PREDICATE,
            "evidence": {
                "requirement_ids": [SCAEV180PROOFREADY],
                "coverage": list(SCAEV180PROOFREADY_COVERAGE),
            },
        }


Which = Callable[[str], str | None]
Environ = Mapping[str, str]
SelfTestCallback = Callable[[ProveKitSelfTestCase], bool]


def _resolve_executable(
    config: ProveKitSetupConfig,
    *,
    which: Which,
    environ: Environ,
) -> ContentAddressedIdentity:
    candidates: list[str] = []
    if config.executable_path:
        candidates.append(str(config.executable_path))
    for env_name in config.executable_env_names:
        value = str(environ.get(env_name, "") or "").strip()
        if value:
            candidates.append(value)
    for name in config.executable_names:
        found = which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            digest = f"provekit-executable:sha256:{_sha256_file(path)}"
        except OSError as exc:
            return ContentAddressedIdentity(
                surface="executable",
                path=str(path),
                digest="",
                present=False,
                reason_code=f"executable_unreadable:{type(exc).__name__}",
            )
        return ContentAddressedIdentity(
            surface="executable",
            path=str(path),
            digest=digest,
            present=True,
        )

    return ContentAddressedIdentity(
        surface="executable",
        path="",
        digest="",
        present=False,
        reason_code="provekit_executable_missing",
    )


def _resolve_artifacts_dir(
    config: ProveKitSetupConfig,
    *,
    environ: Environ,
) -> Path | None:
    if config.artifacts_path:
        path = Path(config.artifacts_path)
        return path if path.is_dir() else path
    for env_name in config.artifacts_env_names:
        value = str(environ.get(env_name, "") or "").strip()
        if value:
            return Path(value)
    return None


def _identity_for_path(
    surface: str,
    path: Path | None,
    *,
    missing_code: str,
) -> ContentAddressedIdentity:
    if path is None:
        return ContentAddressedIdentity(
            surface=surface,
            path="",
            digest="",
            present=False,
            reason_code=missing_code,
        )
    if not path.exists():
        return ContentAddressedIdentity(
            surface=surface,
            path=str(path),
            digest="",
            present=False,
            reason_code=missing_code,
        )
    try:
        if path.is_dir():
            entries = sorted(
                f"{child.name}:{_sha256_file(child) if child.is_file() else 'dir'}"
                for child in path.iterdir()
            )
            digest = _digest_json(
                {"path": str(path), "entries": entries},
                prefix=f"provekit-{surface}",
            )
        else:
            digest = f"provekit-{surface}:sha256:{_sha256_file(path)}"
    except OSError as exc:
        return ContentAddressedIdentity(
            surface=surface,
            path=str(path),
            digest="",
            present=False,
            reason_code=f"{surface}_unreadable:{type(exc).__name__}",
        )
    return ContentAddressedIdentity(
        surface=surface,
        path=str(path),
        digest=digest,
        present=True,
    )


def _run_self_tests(
    callbacks: Mapping[ProveKitSelfTestCase, SelfTestCallback] | None,
) -> tuple[ProveKitSelfTestResult, ...]:
    results: list[ProveKitSelfTestResult] = []
    for case in REQUIRED_PROVEKIT_SELF_TESTS:
        if callbacks is None or case not in callbacks:
            results.append(
                ProveKitSelfTestResult(
                    case=case,
                    verdict=ProveKitSelfTestVerdict.NOT_RUN,
                    reason_code="self_test_not_supplied",
                )
            )
            continue
        callback = callbacks[case]
        try:
            passed = bool(callback(case))
        except Exception:
            results.append(
                ProveKitSelfTestResult(
                    case=case,
                    verdict=ProveKitSelfTestVerdict.ERROR,
                    reason_code="self_test_raised",
                )
            )
            continue
        results.append(
            ProveKitSelfTestResult(
                case=case,
                verdict=(
                    ProveKitSelfTestVerdict.PASSED
                    if passed
                    else ProveKitSelfTestVerdict.FAILED
                ),
                reason_code="" if passed else "self_test_failed",
            )
        )
    return tuple(results)


def probe_provekit_setup(
    config: ProveKitSetupConfig | None = None,
    *,
    which: Which | None = None,
    environ: Environ | None = None,
    self_tests: Mapping[ProveKitSelfTestCase | str, SelfTestCallback] | None = None,
) -> ProveKitSetupReceipt:
    """Discover ProveKit executable/setup/circuit/verifier identities.

    Discovery is metadata-only plus optional injected self-test callbacks.  It
    never installs binaries and never generates cryptographic proofs.
    """

    checked = config or ProveKitSetupConfig()
    which_fn = which or shutil.which
    env = environ if environ is not None else os.environ

    if checked.simulated:
        missing = ContentAddressedIdentity(
            surface="simulated",
            path="",
            digest="",
            present=False,
            reason_code="simulated_non_cryptographic",
        )
        return ProveKitSetupReceipt(
            status=ProveKitSetupStatus.SIMULATED,
            configured=True,
            available=False,
            production_eligible=False,
            executable=missing,
            setup=missing,
            circuit=ContentAddressedIdentity(
                surface="circuit",
                path="",
                digest="",
                present=False,
                reason_code="simulated_non_cryptographic",
            ),
            verifier=ContentAddressedIdentity(
                surface="verifier",
                path="",
                digest="",
                present=False,
                reason_code="simulated_non_cryptographic",
            ),
            prover=ContentAddressedIdentity(
                surface="prover",
                path="",
                digest="",
                present=False,
                reason_code="simulated_non_cryptographic",
            ),
            self_tests=tuple(
                ProveKitSelfTestResult(
                    case=case,
                    verdict=ProveKitSelfTestVerdict.NOT_RUN,
                    reason_code="simulated_backend",
                )
                for case in REQUIRED_PROVEKIT_SELF_TESTS
            ),
            circuit_id=checked.circuit_id,
            circuit_version=checked.circuit_version,
            public_input_schema_id=checked.public_input_schema_id,
            public_input_schema_version=checked.public_input_schema_version,
            backend_version=checked.backend_version,
            simulated=True,
            reason_code="simulated_non_attested",
            reason=(
                "simulated/hash-commitment ProveKit is discoverable only as a "
                "non-cryptographic lane and cannot be attested"
            ),
        )

    executable = _resolve_executable(checked, which=which_fn, environ=env)
    artifacts_dir = _resolve_artifacts_dir(checked, environ=env)

    setup = _identity_for_path(
        "setup",
        artifacts_dir,
        missing_code="provekit_setup_artifacts_missing",
    )
    circuit_path: Path | None = None
    prover_path: Path | None = None
    verifier_path: Path | None = None
    if artifacts_dir is not None and artifacts_dir.is_dir():
        manifest = artifacts_dir / "manifest.json"
        circuit_path = manifest if manifest.is_file() else artifacts_dir
        prover_path = _match_any(artifacts_dir, DEFAULT_PROVER_KEY_PATTERNS)
        verifier_path = _match_any(artifacts_dir, DEFAULT_VERIFIER_KEY_PATTERNS)

    circuit = _identity_for_path(
        "circuit",
        circuit_path,
        missing_code="provekit_circuit_identity_missing",
    )
    prover = _identity_for_path(
        "prover",
        prover_path,
        missing_code="provekit_prover_key_missing",
    )
    verifier = _identity_for_path(
        "verifier",
        verifier_path,
        missing_code="provekit_verifier_key_missing",
    )

    normalized_tests: dict[ProveKitSelfTestCase, SelfTestCallback] | None = None
    if self_tests is not None:
        normalized_tests = {
            ProveKitSelfTestCase(str(getattr(case, "value", case))): callback
            for case, callback in self_tests.items()
        }
    test_results = _run_self_tests(normalized_tests)

    configured = bool(
        executable.present
        or setup.present
        or circuit.present
        or prover.present
        or verifier.present
    )
    surfaces_ready = (
        executable.present
        and setup.present
        and circuit.present
        and prover.present
        and verifier.present
    )
    tests_complete = all(
        result.verdict is not ProveKitSelfTestVerdict.NOT_RUN
        for result in test_results
    )
    tests_passed = all(result.passed for result in test_results) and tests_complete

    if not configured:
        status = ProveKitSetupStatus.UNAVAILABLE
        reason_code = "provekit_unconfigured"
        reason = (
            "ProveKit executable and setup artifacts are not configured; "
            "real ZK remains typed unavailable"
        )
        available = False
        production_eligible = False
    elif not surfaces_ready:
        status = ProveKitSetupStatus.DEGRADED if setup.present or executable.present else ProveKitSetupStatus.UNAVAILABLE
        missing = [
            name
            for name, identity in (
                ("executable", executable),
                ("setup", setup),
                ("circuit", circuit),
                ("prover", prover),
                ("verifier", verifier),
            )
            if not identity.present
        ]
        reason_code = "provekit_surface_incomplete"
        reason = (
            "ProveKit requires executable, setup, circuit, prover, and verifier "
            f"identities; missing: {', '.join(missing)}"
        )
        available = False
        production_eligible = False
        if executable.present and setup.present and not (
            circuit.present and prover.present and verifier.present
        ):
            status = ProveKitSetupStatus.CONFIGURED
    elif not tests_complete:
        status = ProveKitSetupStatus.AVAILABLE
        reason_code = "provekit_self_tests_required"
        reason = (
            "ProveKit executable/setup/circuit/verifier identities are present, "
            "but mandatory positive/negative self-tests have not all been run"
        )
        available = True
        production_eligible = False
    elif not tests_passed:
        status = ProveKitSetupStatus.DEGRADED
        failed = [
            result.case.value
            for result in test_results
            if not result.passed
        ]
        reason_code = "provekit_self_test_failed"
        reason = (
            "ProveKit failed mandatory self-tests: " + ", ".join(failed)
        )
        available = True
        production_eligible = False
    else:
        status = ProveKitSetupStatus.VERIFIED
        reason_code = "provekit_production_eligible"
        reason = (
            "ProveKit executable, setup, circuit, and verifier identities are "
            "content-addressed and mandatory self-tests passed"
        )
        available = True
        production_eligible = True

    return ProveKitSetupReceipt(
        status=status,
        configured=configured,
        available=available,
        production_eligible=production_eligible,
        executable=executable,
        setup=setup,
        circuit=circuit,
        verifier=verifier,
        prover=prover,
        self_tests=test_results,
        circuit_id=checked.circuit_id,
        circuit_version=checked.circuit_version,
        public_input_schema_id=checked.public_input_schema_id,
        public_input_schema_version=checked.public_input_schema_version,
        backend_version=checked.backend_version,
        simulated=False,
        reason_code=reason_code,
        reason=reason,
    )


def evaluate_provekit_attestation_eligibility(
    setup: ProveKitSetupReceipt,
    *,
    predicate_id: str,
    kernel_verified: bool,
    kernel_receipt_id: str = "",
) -> ProveKitAttestationEligibility:
    """Allow attestation only for an approved kernel-verified receipt predicate.

    Simulated backends, incomplete setup, failed self-tests, unapproved
    predicates, and receipts that are not already kernel-verified all fail
    closed with ``eligible=False`` and ``attested=False``.
    """

    if not isinstance(setup, ProveKitSetupReceipt):
        raise ValueError("setup must be a ProveKitSetupReceipt")
    predicate = str(predicate_id or "").strip()
    receipt = str(kernel_receipt_id or "").strip()

    if setup.simulated:
        return ProveKitAttestationEligibility(
            eligible=False,
            predicate_id=predicate,
            setup=setup,
            kernel_verified=False,
            kernel_receipt_id="",
            reason_code="simulated_non_attested",
            reason="simulated/hash-commitment ZK cannot be attested",
            attested=False,
        )
    if not setup.production_eligible:
        return ProveKitAttestationEligibility(
            eligible=False,
            predicate_id=predicate,
            setup=setup,
            kernel_verified=bool(kernel_verified),
            kernel_receipt_id=receipt,
            reason_code="setup_not_production_eligible",
            reason=(
                "ProveKit setup is not production eligible: " + setup.reason
            ),
            attested=False,
        )
    if predicate not in APPROVED_VERIFIED_RECEIPT_PREDICATES:
        return ProveKitAttestationEligibility(
            eligible=False,
            predicate_id=predicate,
            setup=setup,
            kernel_verified=bool(kernel_verified),
            kernel_receipt_id=receipt,
            reason_code="predicate_not_approved",
            reason=(
                "real ZK attests only an approved verified-receipt predicate; "
                f"got {predicate!r}"
            ),
            attested=False,
        )
    if not kernel_verified:
        return ProveKitAttestationEligibility(
            eligible=False,
            predicate_id=predicate,
            setup=setup,
            kernel_verified=False,
            kernel_receipt_id=receipt,
            reason_code="kernel_verification_required",
            reason=(
                "only an already kernel-verified approved receipt predicate is "
                "eligible for ProveKit attestation"
            ),
            attested=False,
        )
    if not receipt:
        return ProveKitAttestationEligibility(
            eligible=False,
            predicate_id=predicate,
            setup=setup,
            kernel_verified=True,
            kernel_receipt_id="",
            reason_code="kernel_receipt_missing",
            reason="kernel-verified attestation requires a kernel receipt id",
            attested=False,
        )
    return ProveKitAttestationEligibility(
        eligible=True,
        predicate_id=predicate,
        setup=setup,
        kernel_verified=True,
        kernel_receipt_id=receipt,
        reason_code="eligible_verified_receipt_predicate",
        reason=(
            "ProveKit setup is production eligible and the predicate is an "
            "approved kernel-verified receipt"
        ),
        attested=False,
    )


def build_provekit_setup_report(
    receipt: ProveKitSetupReceipt,
) -> dict[str, Any]:
    """Public report envelope for operators and production composition."""

    return {
        "schema": PROVEKIT_SETUP_REPORT_SCHEMA,
        "version": PROVEKIT_SETUP_VERSION,
        "receipt": receipt.to_dict(),
        "production_eligible": receipt.production_eligible,
        "status": receipt.status.value,
        "setup_identity": receipt.setup_identity,
        "evidence": {
            "requirement_ids": [SCAEV180PROOFREADY],
            "coverage": list(SCAEV180PROOFREADY_COVERAGE),
        },
    }


__all__ = [
    "APPROVED_VERIFIED_RECEIPT_PREDICATE",
    "APPROVED_VERIFIED_RECEIPT_PREDICATES",
    "ContentAddressedIdentity",
    "DEFAULT_PROVEKIT_EXECUTABLE_NAMES",
    "PROVEKIT_ATTESTATION_ELIGIBILITY_SCHEMA",
    "PROVEKIT_SETUP_REPORT_SCHEMA",
    "PROVEKIT_SETUP_SCHEMA",
    "PROVEKIT_SETUP_VERSION",
    "ProveKitAttestationEligibility",
    "ProveKitSelfTestCase",
    "ProveKitSelfTestResult",
    "ProveKitSelfTestVerdict",
    "ProveKitSetupConfig",
    "ProveKitSetupReceipt",
    "ProveKitSetupStatus",
    "REQUIRED_PROVEKIT_SELF_TESTS",
    "SCAEV180PROOFREADY",
    "SCAEV180PROOFREADY_COVERAGE",
    "SCAEV180PROOFREADY_EVIDENCE",
    "build_provekit_setup_report",
    "evaluate_provekit_attestation_eligibility",
    "probe_provekit_setup",
]
