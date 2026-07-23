"""Independent kernel reconstruction boundary for formal proof receipts.

Hammer portfolio results and model-generated proof text are deliberately
untrusted inputs.  This module is the supervisor-owned adapter from the
``ReconstructionRecord`` / ``ReconstructionEvidence`` pair emitted by
``ipfs_datasets_py.logic.hammers`` to the canonical proof contracts.

The adapter repeats all trust-relevant checks instead of trusting a provider's
``verified`` status:

* request, candidate, theorem, target ITP, environment lock, command, and
  output digests must agree;
* the checked source must be the expected theorem and contain neither proof
  escape hatches nor forbidden declarations;
* timeout, unavailable kernel, non-zero exit, corrupt evidence, and the
  target-specific Lean/Coq/Isabelle acceptance checks all fail closed; and
* only the evidence emitted by a successful mapping can derive ``PROVED`` and
  ``KERNEL_VERIFIED``.

The mature Hammer package is imported only by :class:`IndependentKernelVerifier`
when live reconstruction is requested.  Mapping persisted records has no
optional package import and accepts either upstream objects or their serialized
mapping form.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    content_identity,
    derive_assurance,
    derive_verdict,
)


KERNEL_VERIFICATION_SCHEMA_VERSION: Final = 1
KERNEL_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/kernel-verification@1"
)
SUPPORTED_RECONSTRUCTION_SCHEMA_VERSIONS: Final = frozenset({"1.0.0"})
SUPPORTED_KERNEL_TARGETS: Final = ("lean", "coq", "isabelle")
DEFAULT_MAX_CHECKED_SOURCE_BYTES: Final = 4 * 1024 * 1024
DEFAULT_MAX_KERNEL_OUTPUT_BYTES: Final = 4 * 1024 * 1024


class KernelVerificationError(ContractValidationError):
    """Raised when the caller supplies an invalid verification contract."""


class KernelTarget(str, Enum):
    """Allowed native target kernels."""

    UNKNOWN = "unknown"
    LEAN = "lean"
    COQ = "coq"
    ROCQ = "coq"
    ISABELLE = "isabelle"


class KernelVerificationStatus(str, Enum):
    """Outcome of independently evaluating a reconstruction packet."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class KernelFailureCode(str, Enum):
    """Stable, fail-closed reconstruction failure vocabulary."""

    NONE = ""
    KERNEL_UNAVAILABLE = "kernel_unavailable"
    UNSUPPORTED_KERNEL = "unsupported_kernel"
    KERNEL_TIMED_OUT = "kernel_timed_out"
    KERNEL_REJECTED = "kernel_rejected"
    BINDING_MISMATCH = "binding_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    STATEMENT_MISMATCH = "statement_mismatch"
    FORBIDDEN_DECLARATION = "forbidden_declaration"
    INCOMPLETE_PROOF = "incomplete_proof"
    CORRUPT_EVIDENCE = "corrupt_evidence"
    MALFORMED_RECONSTRUCTION = "malformed_reconstruction"


@dataclass(frozen=True)
class KernelVerificationPolicy:
    """Supervisor policy applied after the upstream reconstruction check."""

    allowed_targets: tuple[KernelTarget | str, ...] = (
        KernelTarget.LEAN,
        KernelTarget.COQ,
        KernelTarget.ISABELLE,
    )
    require_environment_lock: bool = True
    require_exact_statement: bool = True
    require_output_digest: bool = True
    max_checked_source_bytes: int = DEFAULT_MAX_CHECKED_SOURCE_BYTES
    max_kernel_output_bytes: int = DEFAULT_MAX_KERNEL_OUTPUT_BYTES

    def __post_init__(self) -> None:
        targets = tuple(
            sorted(
                {
                    _target(item, field_name="allowed_targets")
                    for item in self.allowed_targets
                },
                key=lambda item: item.value,
            )
        )
        if not targets or KernelTarget.UNKNOWN in targets:
            raise KernelVerificationError(
                "allowed_targets must contain supported kernels"
            )
        object.__setattr__(self, "allowed_targets", targets)
        for name in (
            "require_environment_lock",
            "require_exact_statement",
            "require_output_digest",
        ):
            if not isinstance(getattr(self, name), bool):
                raise KernelVerificationError(f"{name} must be a boolean")
        for name in ("max_checked_source_bytes", "max_kernel_output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise KernelVerificationError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class KernelVerificationBindings:
    """Exact semantic/execution identities expected by the supervisor."""

    obligation_id: str
    request_id: str
    candidate_id: str
    kernel_id: str
    toolchain_id: str
    expected_statement: str = ""
    expected_statement_digest: str = ""
    expected_checked_source_digest: str = ""
    expected_native_source: str = ""

    def __post_init__(self) -> None:
        for name in (
            "obligation_id",
            "request_id",
            "candidate_id",
            "kernel_id",
            "toolchain_id",
        ):
            value = _text(getattr(self, name), field_name=name, required=True)
            object.__setattr__(self, name, value)
        for name in (
            "expected_statement",
            "expected_statement_digest",
            "expected_checked_source_digest",
            "expected_native_source",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )


@dataclass(frozen=True)
class KernelVerificationResult:
    """Auditable mapping result whose verdict and assurance are projections."""

    target: KernelTarget
    status: KernelVerificationStatus
    failure_code: KernelFailureCode
    reason_codes: tuple[str, ...]
    obligation_id: str
    request_id: str
    candidate_id: str
    reconstruction_id: str
    kernel_id: str
    toolchain_id: str
    environment_lock_id: str
    checked_source_digest: str
    kernel_output_digest: str
    evidence: ProofEvidence
    provider_status: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _target(self.target, field_name="target"))
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, KernelVerificationStatus, "status"),
        )
        object.__setattr__(
            self,
            "failure_code",
            _enum_value(self.failure_code, KernelFailureCode, "failure_code"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                sorted(
                    {
                        _text(item, field_name="reason_codes", required=True)
                        for item in self.reason_codes
                    }
                )
            ),
        )
        for name in (
            "obligation_id",
            "request_id",
            "candidate_id",
            "kernel_id",
            "toolchain_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        for name in (
            "reconstruction_id",
            "environment_lock_id",
            "checked_source_digest",
            "kernel_output_digest",
            "provider_status",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        if not isinstance(self.evidence, ProofEvidence):
            raise KernelVerificationError("evidence must be ProofEvidence")
        if not isinstance(self.diagnostics, Mapping):
            raise KernelVerificationError("diagnostics must be a mapping")
        diagnostics = _json_safe(self.diagnostics, field_name="diagnostics")
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.status is KernelVerificationStatus.ACCEPTED:
            if self.failure_code is not KernelFailureCode.NONE:
                raise KernelVerificationError(
                    "accepted verification cannot carry a failure code"
                )
            if self.evidence.verdict is not EvidenceVerdict.ACCEPTED:
                raise KernelVerificationError(
                    "accepted verification requires accepted evidence"
                )
        elif self.failure_code is KernelFailureCode.NONE:
            raise KernelVerificationError(
                "non-accepted verification requires a failure code"
            )

    @property
    def verdict(self) -> ProofVerdict:
        return derive_verdict(
            (self.evidence,),
            obligation_id=self.obligation_id,
            kernel_id=self.kernel_id,
        )

    @property
    def authoritative_verdict(self) -> ProofVerdict:
        return self.verdict

    @property
    def assurance(self) -> AssuranceLevel:
        return derive_assurance(
            (self.evidence,),
            obligation_id=self.obligation_id,
            kernel_id=self.kernel_id,
        )

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        return self.assurance

    @property
    def accepted(self) -> bool:
        return (
            self.status is KernelVerificationStatus.ACCEPTED
            and self.verdict is ProofVerdict.PROVED
            and self.assurance is AssuranceLevel.KERNEL_VERIFIED
        )

    @property
    def verification_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def kernel_receipt_id(self) -> str:
        return self.verification_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": KERNEL_VERIFICATION_SCHEMA,
            "schema_version": KERNEL_VERIFICATION_SCHEMA_VERSION,
            "target": self.target.value,
            "status": self.status.value,
            "failure_code": self.failure_code.value,
            "reason_codes": list(self.reason_codes),
            "obligation_id": self.obligation_id,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "reconstruction_id": self.reconstruction_id,
            "kernel_id": self.kernel_id,
            "toolchain_id": self.toolchain_id,
            "environment_lock_id": self.environment_lock_id,
            "checked_source_digest": self.checked_source_digest,
            "kernel_output_digest": self.kernel_output_digest,
            "evidence": self.evidence.to_dict(),
            "provider_status": self.provider_status,
            "authoritative_verdict": self.verdict.value,
            "authoritative_assurance": self.assurance.value,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KernelVerificationResult":
        if not isinstance(payload, Mapping):
            raise KernelVerificationError(
                "kernel verification payload must be a mapping"
            )
        if payload.get("schema") not in (None, KERNEL_VERIFICATION_SCHEMA):
            raise KernelVerificationError("unsupported kernel verification schema")
        result = cls(
            target=payload.get("target", KernelTarget.UNKNOWN),
            status=payload.get("status", KernelVerificationStatus.ERROR),
            failure_code=payload.get(
                "failure_code", KernelFailureCode.MALFORMED_RECONSTRUCTION
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            obligation_id=payload.get("obligation_id", ""),
            request_id=payload.get("request_id", ""),
            candidate_id=payload.get("candidate_id", ""),
            reconstruction_id=payload.get("reconstruction_id", ""),
            kernel_id=payload.get("kernel_id", ""),
            toolchain_id=payload.get("toolchain_id", ""),
            environment_lock_id=payload.get("environment_lock_id", ""),
            checked_source_digest=payload.get("checked_source_digest", ""),
            kernel_output_digest=payload.get("kernel_output_digest", ""),
            evidence=ProofEvidence.from_dict(payload.get("evidence") or {}),
            provider_status=payload.get("provider_status", ""),
            diagnostics=payload.get("diagnostics") or {},
        )
        claimed_verdict = payload.get("authoritative_verdict")
        if claimed_verdict and str(claimed_verdict) != result.verdict.value:
            raise KernelVerificationError(
                "kernel verification authoritative verdict does not match evidence"
            )
        claimed_assurance = payload.get("authoritative_assurance")
        if claimed_assurance and str(claimed_assurance) != result.assurance.value:
            raise KernelVerificationError(
                "kernel verification authoritative assurance does not match evidence"
            )
        return result

    def build_receipt(self, **receipt_fields: Any) -> ProofReceipt:
        """Build a receipt whose verdict is derived from this result."""

        return build_kernel_verified_receipt(self, **receipt_fields)


def _enum_value(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise KernelVerificationError(f"unsupported {field_name}: {raw!r}") from exc


def _target(value: Any, *, field_name: str) -> KernelTarget:
    raw = str(getattr(value, "value", value) or "").strip().lower()
    if raw == "rocq":
        raw = "coq"
    try:
        return KernelTarget(raw)
    except ValueError as exc:
        raise KernelVerificationError(
            f"{field_name} must be one of: {', '.join(SUPPORTED_KERNEL_TARGETS)}"
        ) from exc


def _text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise KernelVerificationError(f"{field_name} must be a string")
    if required and not result:
        raise KernelVerificationError(f"{field_name} is required")
    return result


def _record(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = value
    else:
        validator = getattr(value, "validate", None)
        if callable(validator):
            validator()
        converter = getattr(value, "to_dict", None)
        if not callable(converter):
            raise KernelVerificationError(
                f"{field_name} must be a mapping or expose to_dict()"
            )
        raw = converter()
    if not isinstance(raw, Mapping):
        raise KernelVerificationError(f"{field_name} must serialize to an object")
    return dict(raw)


def _json_safe(value: Any, *, field_name: str) -> Any:
    """Return the integer/string JSON subset used by proof contracts."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value, field_name=field_name)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise KernelVerificationError(f"{field_name} keys must be strings")
        return {
            key: _json_safe(item, field_name=field_name)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item, field_name=field_name) for item in value]
    raise KernelVerificationError(
        f"{field_name} contains unsupported {type(value).__name__}"
    )


def _upstream_content_digest(payload: Any) -> str:
    """Compute Hammer's digest without importing its package eagerly."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    try:
        from multiformats import CID, multihash  # type: ignore

        return str(CID("base32", 1, "raw", multihash.wrap(digest, "sha2-256")))
    except (ImportError, TypeError, ValueError):
        return "sha256:" + digest.hex()


def _digest_matches(claimed: str, payload: Any) -> bool:
    if not claimed:
        return False
    actual = _upstream_content_digest(payload)
    if claimed == actual:
        return True
    # Cross-environment compatibility: persisted records may use the plain
    # sha256 spelling when ``multiformats`` was absent at production time.
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return claimed == "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalized_statement(value: str) -> str:
    return " ".join(value.split())


_LEAN_DECLARATION = re.compile(
    r"\b(?:theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_'.]*\s*(.*?)(?::=|\bby\b)",
    re.DOTALL,
)
_COQ_DECLARATION = re.compile(
    r"\b(?:Theorem|Lemma|Example|Fact|Remark|Corollary|Proposition)\s+"
    r"[A-Za-z_][A-Za-z0-9_']*\s*:\s*(.*?)\.\s*(?:Proof\.)?",
    re.DOTALL,
)
_ISABELLE_DECLARATION = re.compile(
    r"\b(?:theorem|lemma|corollary|proposition)\s+"
    r"(?:[A-Za-z_][A-Za-z0-9_']*\s*:)?\s*(.*?)(?=\b(?:by|proof)\b)",
    re.DOTALL,
)


def _extract_statement(source: str, target: KernelTarget) -> str:
    pattern = {
        KernelTarget.LEAN: _LEAN_DECLARATION,
        KernelTarget.COQ: _COQ_DECLARATION,
        KernelTarget.ISABELLE: _ISABELLE_DECLARATION,
    }[target]
    match = pattern.search(source)
    if not match:
        return ""
    statement = _normalized_statement(match.group(1))
    if target is KernelTarget.LEAN:
        statement = statement.removeprefix(":").strip()
    if target is KernelTarget.ISABELLE and len(statement) >= 2:
        if statement[0] == statement[-1] and statement[0] in {'"', "'"}:
            statement = statement[1:-1].strip()
    return statement


_FORBIDDEN_DECLARATIONS = {
    KernelTarget.LEAN: (
        re.compile(r"(?im)^\s*(?:axiom|constant|unsafe\s+(?:def|theorem))\b"),
        re.compile(r"(?im)^\s*set_option\s+(?:autoImplicit|warningAsError)\s+false\b"),
    ),
    KernelTarget.COQ: (
        re.compile(r"(?im)^\s*(?:Axiom|Axioms|Parameter|Parameters)\b"),
        re.compile(r"(?im)^\s*Unset\s+Guard\s+Checking\s*\."),
    ),
    KernelTarget.ISABELLE: (
        re.compile(r"(?im)^\s*(?:axiomatization|axioms|oracle)\b"),
        re.compile(r"(?im)^\s*declare\s+\[\[quick_and_dirty\s*=\s*true\]\]"),
    ),
}

_INCOMPLETE_PROOFS = {
    KernelTarget.LEAN: re.compile(r"\b(?:sorry|admit)\b|\bsorryAx\b", re.IGNORECASE),
    KernelTarget.COQ: re.compile(r"\b(?:admit|admitted)\b", re.IGNORECASE),
    KernelTarget.ISABELLE: re.compile(
        r"\b(?:sorry|oops|skip_proof)\b", re.IGNORECASE
    ),
}


def _source_failure(source: str, target: KernelTarget) -> tuple[KernelFailureCode, str] | None:
    if _INCOMPLETE_PROOFS[target].search(source):
        return (
            KernelFailureCode.INCOMPLETE_PROOF,
            "checked source contains a sorry/admit/incomplete-proof escape hatch",
        )
    for pattern in _FORBIDDEN_DECLARATIONS[target]:
        match = pattern.search(source)
        if match:
            declaration = " ".join(match.group(0).split())[:160]
            return (
                KernelFailureCode.FORBIDDEN_DECLARATION,
                f"checked source contains forbidden declaration: {declaration}",
            )
    return None


def _target_output_failure(
    target: KernelTarget,
    *,
    stdout: str,
    stderr: str,
) -> str:
    combined = stdout + "\n" + stderr
    if target is KernelTarget.LEAN:
        messages = []
        for line in stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping):
                messages.append(item)
        if any(item.get("severity") == "error" for item in messages):
            return "Lean emitted an error diagnostic"
        axiom_messages = [
            str(item.get("data") or "")
            for item in messages
            if "depends on axioms" in str(item.get("data") or "")
            or "does not depend on any axioms" in str(item.get("data") or "")
        ]
        if not axiom_messages:
            return "Lean emitted no #print axioms result"
        if any("sorryAx" in message for message in axiom_messages):
            return "Lean axiom report contains sorryAx"
    elif target is KernelTarget.COQ:
        if "Error:" in combined:
            return "Coq/Rocq emitted an error diagnostic"
        if "Closed under the global context" not in stdout:
            return "Coq/Rocq did not confirm a closed global context"
    else:
        if re.search(r"(?m)^\*\*\*\s+", combined) or "Failed" in combined:
            return "Isabelle emitted a failure diagnostic"
        if re.search(r"\bsorry\b", combined, re.IGNORECASE):
            return "Isabelle output references sorry"
    return ""


def _failed_result(
    *,
    target: KernelTarget,
    status: KernelVerificationStatus,
    failure_code: KernelFailureCode,
    reason: str,
    bindings: KernelVerificationBindings,
    record: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    environment_lock: Mapping[str, Any],
    provider_status: str,
) -> KernelVerificationResult:
    reconstruction_id = str(record.get("reconstruction_id") or "").strip()
    environment_lock_id = str(
        record.get("environment_lock_id")
        or environment_lock.get("lock_id")
        or ""
    ).strip()
    checked_source_digest = str(
        evidence_record.get("checked_source_digest") or ""
    ).strip()
    output_digest = str(
        evidence_record.get("raw_output_digest")
        or record.get("kernel_output_digest")
        or ""
    ).strip()
    artifact_payload = {
        "failure_code": failure_code.value,
        "reason": reason,
        "reconstruction_id": reconstruction_id,
        "request_id": bindings.request_id,
        "candidate_id": bindings.candidate_id,
        "checked_source_digest": checked_source_digest,
        "kernel_output_digest": output_digest,
    }
    proof_evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=(
            EvidenceVerdict.UNSUPPORTED
            if status is KernelVerificationStatus.UNAVAILABLE
            else EvidenceVerdict.REJECTED
            if status in {
                KernelVerificationStatus.REJECTED,
                KernelVerificationStatus.TIMED_OUT,
            }
            else EvidenceVerdict.ERROR
        ),
        artifact_id=content_identity(artifact_payload),
        subject_id=bindings.obligation_id,
        verifier_id=bindings.kernel_id,
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata={
            "failure_code": failure_code.value,
            "reconstruction_id": reconstruction_id,
            "target": target.value,
            "bindings_verified": False,
        },
    )
    return KernelVerificationResult(
        target=target,
        status=status,
        failure_code=failure_code,
        reason_codes=(failure_code.value,),
        obligation_id=bindings.obligation_id,
        request_id=bindings.request_id,
        candidate_id=bindings.candidate_id,
        reconstruction_id=reconstruction_id,
        kernel_id=bindings.kernel_id,
        toolchain_id=bindings.toolchain_id,
        environment_lock_id=environment_lock_id,
        checked_source_digest=checked_source_digest,
        kernel_output_digest=output_digest,
        evidence=proof_evidence,
        provider_status=provider_status,
        diagnostics={"message": reason[:4096]},
    )


def verify_kernel_reconstruction(
    reconstruction_record: Any,
    reconstruction_evidence: Any,
    environment_lock: Any,
    *,
    bindings: KernelVerificationBindings | None = None,
    obligation: CodeProofObligation | Mapping[str, Any] | None = None,
    request: Any = None,
    obligation_id: str = "",
    request_id: str = "",
    candidate_id: str = "",
    kernel_id: str = "",
    toolchain_id: str = "",
    expected_statement: str = "",
    expected_theorem_statement: str = "",
    expected_statement_digest: str = "",
    expected_checked_source_digest: str = "",
    expected_native_source: str = "",
    provider_status: str = "",
    independent: bool = False,
    policy: KernelVerificationPolicy | None = None,
) -> KernelVerificationResult:
    """Map a Hammer reconstruction packet into authoritative proof evidence.

    All expected bindings are required.  They can be supplied through
    ``bindings`` or explicit keyword arguments; ``obligation`` and ``request``
    additionally validate that the Hammer request names the exact canonical
    obligation and unchanged theorem statement.  The caller must explicitly
    set ``independent=True`` after obtaining the packet from a supervisor-owned
    kernel execution boundary; provider-originated packets fail closed.
    """

    verification_policy = policy or KernelVerificationPolicy()
    if not isinstance(verification_policy, KernelVerificationPolicy):
        raise KernelVerificationError("policy must be KernelVerificationPolicy")

    record = _record(reconstruction_record, field_name="reconstruction_record")
    evidence_record = _record(
        reconstruction_evidence, field_name="reconstruction_evidence"
    )
    lock_record = _record(environment_lock, field_name="environment_lock")
    request_record = _record(request, field_name="request") if request is not None else {}

    obligation_record: Mapping[str, Any] = {}
    if obligation is not None:
        obligation_record = _record(obligation, field_name="obligation")
        if isinstance(obligation, CodeProofObligation):
            obligation_id = obligation_id or obligation.obligation_id
            expected_statement = expected_statement or obligation.statement
        else:
            obligation_id = obligation_id or str(
                obligation_record.get("obligation_id")
                or obligation_record.get("content_id")
                or ""
            )
            expected_statement = expected_statement or str(
                obligation_record.get("statement") or ""
            )

    if request_record:
        request_id = request_id or str(request_record.get("request_id") or "")
        if not obligation_id:
            obligation_id = str(
                request_record.get("theorem_id")
                or request_record.get("obligation_id")
                or ""
            )
        expected_statement = expected_statement or str(
            request_record.get("goal_statement") or ""
        )

    if bindings is None:
        bindings = KernelVerificationBindings(
            obligation_id=obligation_id,
            request_id=request_id,
            candidate_id=candidate_id,
            kernel_id=kernel_id,
            toolchain_id=toolchain_id,
            expected_statement=expected_theorem_statement or expected_statement,
            expected_statement_digest=expected_statement_digest,
            expected_checked_source_digest=expected_checked_source_digest,
            expected_native_source=expected_native_source,
        )
    elif not isinstance(bindings, KernelVerificationBindings):
        raise KernelVerificationError(
            "bindings must be KernelVerificationBindings"
        )
    elif any(
        (
            expected_theorem_statement,
            expected_statement_digest,
            expected_checked_source_digest,
            expected_native_source,
        )
    ):
        bindings = KernelVerificationBindings(
            obligation_id=bindings.obligation_id,
            request_id=bindings.request_id,
            candidate_id=bindings.candidate_id,
            kernel_id=bindings.kernel_id,
            toolchain_id=bindings.toolchain_id,
            expected_statement=(
                expected_theorem_statement
                or expected_statement
                or bindings.expected_statement
            ),
            expected_statement_digest=(
                expected_statement_digest or bindings.expected_statement_digest
            ),
            expected_checked_source_digest=(
                expected_checked_source_digest
                or bindings.expected_checked_source_digest
            ),
            expected_native_source=(
                expected_native_source or bindings.expected_native_source
            ),
        )

    raw_target = record.get("target_itp") or evidence_record.get("itp")
    try:
        target = _target(raw_target, field_name="target_itp")
    except KernelVerificationError:
        target = KernelTarget.UNKNOWN
        return _failed_result(
            target=target,
            status=KernelVerificationStatus.UNAVAILABLE,
            failure_code=KernelFailureCode.UNSUPPORTED_KERNEL,
            reason=f"unsupported target kernel {raw_target!r}",
            bindings=bindings,
            record=record,
            evidence_record=evidence_record,
            environment_lock=lock_record,
            provider_status=provider_status,
        )

    def fail(
        code: KernelFailureCode,
        reason: str,
        *,
        status: KernelVerificationStatus = KernelVerificationStatus.ERROR,
    ) -> KernelVerificationResult:
        return _failed_result(
            target=target,
            status=status,
            failure_code=code,
            reason=reason,
            bindings=bindings,
            record=record,
            evidence_record=evidence_record,
            environment_lock=lock_record,
            provider_status=_text(provider_status, field_name="provider_status"),
        )

    if (
        target is KernelTarget.UNKNOWN
        or target not in verification_policy.allowed_targets
    ):
        return fail(
            KernelFailureCode.UNSUPPORTED_KERNEL,
            f"target kernel {target.value} is not allowed by policy",
            status=KernelVerificationStatus.UNAVAILABLE,
        )
    if not independent:
        return fail(
            KernelFailureCode.BINDING_MISMATCH,
            "reconstruction did not execute at an independent kernel boundary",
        )

    schema = str(record.get("schema_version") or "")
    evidence_schema = str(evidence_record.get("schema_version") or "")
    lock_schema = str(lock_record.get("schema_version") or "")
    if (
        schema not in SUPPORTED_RECONSTRUCTION_SCHEMA_VERSIONS
        or evidence_schema not in SUPPORTED_RECONSTRUCTION_SCHEMA_VERSIONS
        or (
            verification_policy.require_environment_lock
            and lock_schema not in SUPPORTED_RECONSTRUCTION_SCHEMA_VERSIONS
        )
    ):
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "reconstruction, evidence, or environment lock schema is unsupported",
        )

    for name, expected in (
        ("request_id", bindings.request_id),
        ("candidate_id", bindings.candidate_id),
    ):
        if str(record.get(name) or "") != expected:
            return fail(
                KernelFailureCode.BINDING_MISMATCH,
                f"reconstruction {name} does not match the expected binding",
            )
        if str(evidence_record.get(name) or "") != expected:
            return fail(
                KernelFailureCode.BINDING_MISMATCH,
                f"reconstruction evidence {name} does not match the expected binding",
            )
    if str(evidence_record.get("reconstruction_id") or "") != str(
        record.get("reconstruction_id") or ""
    ):
        return fail(
            KernelFailureCode.BINDING_MISMATCH,
            "record and evidence reconstruction_id values do not match",
        )

    evidence_target = str(
        getattr(evidence_record.get("itp"), "value", evidence_record.get("itp"))
        or ""
    ).lower()
    lock_target = str(
        getattr(lock_record.get("itp"), "value", lock_record.get("itp")) or ""
    ).lower()
    if evidence_target == "rocq":
        evidence_target = "coq"
    if lock_target == "rocq":
        lock_target = "coq"
    if evidence_target != target.value or (
        verification_policy.require_environment_lock
        and lock_target != target.value
    ):
        return fail(
            KernelFailureCode.ENVIRONMENT_MISMATCH,
            "target ITP does not agree across reconstruction evidence and lock",
        )

    environment_lock_id = str(record.get("environment_lock_id") or "")
    if verification_policy.require_environment_lock and (
        not environment_lock_id
        or environment_lock_id != str(lock_record.get("lock_id") or "")
    ):
        return fail(
            KernelFailureCode.ENVIRONMENT_MISMATCH,
            "reconstruction environment lock identity does not match",
        )
    if not str(lock_record.get("itp_version") or "").strip():
        return fail(
            KernelFailureCode.ENVIRONMENT_MISMATCH,
            "environment lock has no exact ITP version",
        )

    if request_record:
        if str(request_record.get("request_id") or "") != bindings.request_id:
            return fail(
                KernelFailureCode.BINDING_MISMATCH,
                "Hammer request identity does not match the expected request",
            )
        theorem_id = str(
            request_record.get("theorem_id")
            or request_record.get("obligation_id")
            or ""
        )
        if theorem_id != bindings.obligation_id:
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "Hammer request theorem identity does not match the obligation",
            )
        request_statement = _normalized_statement(
            str(request_record.get("goal_statement") or "")
        )
        if bindings.expected_statement and request_statement != _normalized_statement(
            bindings.expected_statement
        ):
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "Hammer request changed the expected theorem statement",
            )

    checked_source = evidence_record.get("checked_source")
    proof_text = evidence_record.get("reconstructed_proof_text")
    stdout = evidence_record.get("stdout")
    stderr = evidence_record.get("stderr")
    command = evidence_record.get("command")
    if not isinstance(checked_source, str) or not checked_source:
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "checked_source must be a non-empty string",
        )
    if not isinstance(proof_text, str) or not proof_text.strip():
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "reconstructed_proof_text must be a non-empty string",
        )
    if checked_source.count(proof_text) != 1:
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "reconstructed proof text must occur exactly once in checked source",
        )
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "kernel stdout and stderr must be strings",
        )
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "kernel command must be an argv sequence",
        )
    if not command or not all(isinstance(item, str) and item for item in command):
        return fail(
            KernelFailureCode.MALFORMED_RECONSTRUCTION,
            "kernel command must contain non-empty argv strings",
        )
    if len(checked_source.encode("utf-8")) > verification_policy.max_checked_source_bytes:
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "checked source exceeds the policy byte limit",
        )
    if (
        len(stdout.encode("utf-8")) + len(stderr.encode("utf-8"))
        > verification_policy.max_kernel_output_bytes
    ):
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "kernel output exceeds the policy byte limit",
        )

    claimed_source_digest = str(
        evidence_record.get("checked_source_digest") or ""
    )
    if not _digest_matches(
        claimed_source_digest, {"checked_source": checked_source}
    ):
        return fail(
            KernelFailureCode.DIGEST_MISMATCH,
            "checked source digest does not match the submitted source",
        )
    raw_output_payload = {"stdout": stdout, "stderr": stderr}
    claimed_raw_digest = str(evidence_record.get("raw_output_digest") or "")
    if verification_policy.require_output_digest and not _digest_matches(
        claimed_raw_digest, raw_output_payload
    ):
        return fail(
            KernelFailureCode.DIGEST_MISMATCH,
            "raw kernel output digest does not match stdout/stderr",
        )
    if str(record.get("kernel_output_digest") or "") != claimed_raw_digest:
        return fail(
            KernelFailureCode.DIGEST_MISMATCH,
            "record and evidence kernel output digests do not match",
        )
    if bindings.expected_checked_source_digest and (
        claimed_source_digest != bindings.expected_checked_source_digest
    ):
        return fail(
            KernelFailureCode.STATEMENT_MISMATCH,
            "checked source digest does not match the expected theorem artifact",
        )
    if str(record.get("kernel_command") or "") != " ".join(command):
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "record kernel command does not match evidence argv",
        )

    source_problem = _source_failure(checked_source, target)
    if source_problem is not None:
        return fail(*source_problem)

    extracted_statement = _extract_statement(checked_source, target)
    if verification_policy.require_exact_statement:
        if not extracted_statement:
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "could not extract a theorem statement from checked source",
            )
        expected_native_statement = (
            _extract_statement(bindings.expected_native_source, target)
            if bindings.expected_native_source
            else ""
        )
        if expected_native_statement and extracted_statement != expected_native_statement:
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "checked source theorem statement differs from native source",
            )
        if (
            bindings.expected_statement
            and not expected_native_statement
            and not bindings.expected_statement_digest
            and not bindings.expected_checked_source_digest
            and extracted_statement
            != _normalized_statement(bindings.expected_statement)
        ):
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "checked source theorem statement differs from expected statement",
            )
        if not (
            request_record
            or expected_native_statement
            or bindings.expected_statement
            or bindings.expected_statement_digest
            or bindings.expected_checked_source_digest
        ):
            return fail(
                KernelFailureCode.STATEMENT_MISMATCH,
                "no exact theorem statement binding was supplied",
            )
        if bindings.expected_statement_digest:
            actual_digest = "sha256:" + hashlib.sha256(
                extracted_statement.encode("utf-8")
            ).hexdigest()
            if actual_digest != bindings.expected_statement_digest:
                return fail(
                    KernelFailureCode.STATEMENT_MISMATCH,
                    "checked source theorem statement digest does not match",
                )

    if evidence_record.get("timed_out") is True:
        return fail(
            KernelFailureCode.KERNEL_TIMED_OUT,
            "kernel reconstruction timed out",
            status=KernelVerificationStatus.TIMED_OUT,
        )
    returncode = evidence_record.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "kernel returncode must be an integer",
        )
    if returncode != 0:
        return fail(
            KernelFailureCode.KERNEL_REJECTED,
            f"kernel exited with non-zero status {returncode}",
            status=KernelVerificationStatus.REJECTED,
        )
    if record.get("finished_at") in (None, ""):
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "accepted reconstruction has no completion timestamp",
        )

    output_failure = _target_output_failure(
        target, stdout=stdout, stderr=stderr
    )
    if output_failure:
        code = (
            KernelFailureCode.INCOMPLETE_PROOF
            if "sorry" in output_failure or "closed global context" in output_failure
            else KernelFailureCode.KERNEL_REJECTED
        )
        return fail(code, output_failure, status=KernelVerificationStatus.REJECTED)

    if record.get("kernel_accepted") is not True:
        failure_reason = str(record.get("failure_reason") or "kernel rejected proof")
        return fail(
            KernelFailureCode.KERNEL_REJECTED,
            failure_reason,
            status=KernelVerificationStatus.REJECTED,
        )
    if record.get("failure_reason") not in (None, ""):
        return fail(
            KernelFailureCode.CORRUPT_EVIDENCE,
            "accepted reconstruction unexpectedly contains a failure reason",
        )

    reconstruction_id = str(record.get("reconstruction_id") or "")
    artifact_payload = {
        "schema": KERNEL_VERIFICATION_SCHEMA,
        "target": target.value,
        "obligation_id": bindings.obligation_id,
        "request_id": bindings.request_id,
        "candidate_id": bindings.candidate_id,
        "reconstruction_id": reconstruction_id,
        "environment_lock_id": environment_lock_id,
        "checked_source_digest": claimed_source_digest,
        "kernel_output_digest": claimed_raw_digest,
        "kernel_id": bindings.kernel_id,
        "toolchain_id": bindings.toolchain_id,
    }
    artifact_id = content_identity(artifact_payload)
    proof_evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=artifact_id,
        subject_id=bindings.obligation_id,
        verifier_id=bindings.kernel_id,
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata={
            "bindings_verified": True,
            "candidate_id": bindings.candidate_id,
            "checked_source_digest": claimed_source_digest,
            "environment_lock_id": environment_lock_id,
            "kernel_output_digest": claimed_raw_digest,
            "reconstruction_id": reconstruction_id,
            "statement_verified": True,
            "target": target.value,
            "toolchain_id": bindings.toolchain_id,
        },
    )
    return KernelVerificationResult(
        target=target,
        status=KernelVerificationStatus.ACCEPTED,
        failure_code=KernelFailureCode.NONE,
        reason_codes=("independent_kernel_acceptance",),
        obligation_id=bindings.obligation_id,
        request_id=bindings.request_id,
        candidate_id=bindings.candidate_id,
        reconstruction_id=reconstruction_id,
        kernel_id=bindings.kernel_id,
        toolchain_id=bindings.toolchain_id,
        environment_lock_id=environment_lock_id,
        checked_source_digest=claimed_source_digest,
        kernel_output_digest=claimed_raw_digest,
        evidence=proof_evidence,
        provider_status=_text(provider_status, field_name="provider_status"),
        diagnostics={
            "kernel_returncode": returncode,
            "kernel_timed_out": False,
            "target": target.value,
        },
    )


def kernel_unavailable_result(
    *,
    target: KernelTarget | str,
    bindings: KernelVerificationBindings,
    reason: str = "target kernel is unavailable",
    provider_status: str = "",
) -> KernelVerificationResult:
    """Create typed fail-closed evidence when reconstruction cannot run."""

    normalized_target = _target(target, field_name="target")
    return _failed_result(
        target=normalized_target,
        status=KernelVerificationStatus.UNAVAILABLE,
        failure_code=KernelFailureCode.KERNEL_UNAVAILABLE,
        reason=_text(reason, field_name="reason", required=True),
        bindings=bindings,
        record={},
        evidence_record={},
        environment_lock={},
        provider_status=_text(provider_status, field_name="provider_status"),
    )


def build_kernel_verified_receipt(
    verification: KernelVerificationResult,
    *,
    obligation: CodeProofObligation | None = None,
    plan_id: str,
    attempt_id: str,
    repository_id: str = "",
    repository_tree_id: str = "",
    ast_scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    translator_id: str,
    solver_id: str,
    policy_id: str,
    resource_budget: ResourceBudget | Mapping[str, Any],
    provider_id: str = "",
    provider_claimed_assurance: AssuranceLevel | str = AssuranceLevel.UNVERIFIED,
    theorem_registry_id: str = "",
    started_at: str = "",
    finished_at: str = "",
    resource_usage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProofReceipt:
    """Build a receipt with a verdict derived solely from reconstruction.

    A negative result can be persisted for diagnostics, but can never become
    kernel verified.  ``provider_claimed_assurance`` is retained only for
    audit and is not consulted when deriving either verdict or assurance.
    """

    if not isinstance(verification, KernelVerificationResult):
        raise KernelVerificationError(
            "verification must be KernelVerificationResult"
        )
    if obligation is not None:
        if not isinstance(obligation, CodeProofObligation):
            raise KernelVerificationError(
                "obligation must be CodeProofObligation"
            )
        if obligation.obligation_id != verification.obligation_id:
            raise KernelVerificationError(
                "obligation does not match kernel verification"
            )
        repository_id = repository_id or obligation.repository_id
        repository_tree_id = (
            repository_tree_id or obligation.repository_tree_id
        )
        ast_scope_ids = tuple(ast_scope_ids) or obligation.ast_scope_ids
        premise_ids = tuple(premise_ids) or obligation.premise_ids

    receipt_metadata = dict(metadata or {})
    receipt_metadata.update(
        {
            "kernel_failure_code": verification.failure_code.value,
            "kernel_verification_id": verification.verification_id,
            "reconstruction_id": verification.reconstruction_id,
            "reconstruction_status": verification.status.value,
        }
    )
    return ProofReceipt(
        obligation_id=verification.obligation_id,
        plan_id=plan_id,
        attempt_id=attempt_id,
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        ast_scope_ids=tuple(ast_scope_ids),
        premise_ids=tuple(premise_ids),
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=verification.kernel_id,
        toolchain_id=verification.toolchain_id,
        theorem_registry_id=theorem_registry_id,
        policy_id=policy_id,
        resource_budget=(
            resource_budget
            if isinstance(resource_budget, ResourceBudget)
            else ResourceBudget.from_dict(resource_budget)
        ),
        verdict=verification.verdict,
        evidence=(verification.evidence,),
        provider_id=provider_id,
        provider_claimed_assurance=provider_claimed_assurance,
        freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id=(
            verification.kernel_receipt_id if verification.accepted else ""
        ),
        started_at=started_at,
        finished_at=finished_at,
        resource_usage=resource_usage or {},
        metadata=receipt_metadata,
    )


class IndependentKernelVerifier:
    """Policy-bound facade for record mapping and live Hammer reconstruction."""

    def __init__(self, policy: KernelVerificationPolicy | None = None) -> None:
        self.policy = policy or KernelVerificationPolicy()
        if not isinstance(self.policy, KernelVerificationPolicy):
            raise KernelVerificationError(
                "policy must be KernelVerificationPolicy"
            )

    def verify(self, *args: Any, **kwargs: Any) -> KernelVerificationResult:
        kwargs.setdefault("policy", self.policy)
        return verify_kernel_reconstruction(*args, **kwargs)

    def reconstruct_and_verify(
        self,
        *,
        request: Any,
        candidate: Any,
        goal_snapshot: Any,
        native_source: str,
        bindings: KernelVerificationBindings,
        environment_lock: Any = None,
        timeout: float | None = None,
        provider_status: str = "",
    ) -> KernelVerificationResult:
        """Run the local Hammer reconstructor and immediately map its records."""

        try:
            from ipfs_datasets_py.logic.hammers.reconstruction import (
                KernelUnavailableError,
                reconstruct_candidate,
            )
        except ImportError as exc:
            return kernel_unavailable_result(
                target=getattr(request, "itp", "lean"),
                bindings=bindings,
                reason=f"Hammer reconstruction package unavailable: {exc}",
                provider_status=provider_status,
            )
        try:
            record, evidence, lock = reconstruct_candidate(
                request=request,
                candidate=candidate,
                goal_snapshot=goal_snapshot,
                native_source=native_source,
                environment_lock=environment_lock,
                timeout=timeout,
            )
        except KernelUnavailableError as exc:
            return kernel_unavailable_result(
                target=getattr(request, "itp", "lean"),
                bindings=bindings,
                reason=str(exc),
                provider_status=provider_status,
            )
        return self.verify(
            record,
            evidence,
            lock,
            bindings=bindings,
            request=request,
            expected_native_source=native_source,
            provider_status=provider_status,
            independent=True,
        )


# Compatibility spellings for integrations which focus on mapping or verdicts.
KernelVerifier = IndependentKernelVerifier
KernelReconstructionMapper = IndependentKernelVerifier
KernelReconstructionResult = KernelVerificationResult
map_reconstruction_record = verify_kernel_reconstruction
map_kernel_reconstruction = verify_kernel_reconstruction
map_hammer_reconstruction = verify_kernel_reconstruction
evaluate_reconstruction = verify_kernel_reconstruction
derive_reconstruction_verdict = derive_verdict
derive_kernel_verdict = derive_verdict
create_kernel_receipt = build_kernel_verified_receipt


__all__ = [
    "DEFAULT_MAX_CHECKED_SOURCE_BYTES",
    "DEFAULT_MAX_KERNEL_OUTPUT_BYTES",
    "KERNEL_VERIFICATION_SCHEMA",
    "KERNEL_VERIFICATION_SCHEMA_VERSION",
    "SUPPORTED_KERNEL_TARGETS",
    "SUPPORTED_RECONSTRUCTION_SCHEMA_VERSIONS",
    "IndependentKernelVerifier",
    "KernelFailureCode",
    "KernelReconstructionResult",
    "KernelReconstructionMapper",
    "KernelTarget",
    "KernelVerificationBindings",
    "KernelVerificationError",
    "KernelVerificationPolicy",
    "KernelVerificationResult",
    "KernelVerificationStatus",
    "KernelVerifier",
    "build_kernel_verified_receipt",
    "create_kernel_receipt",
    "derive_reconstruction_verdict",
    "derive_kernel_verdict",
    "evaluate_reconstruction",
    "kernel_unavailable_result",
    "map_kernel_reconstruction",
    "map_hammer_reconstruction",
    "map_reconstruction_record",
    "verify_kernel_reconstruction",
]
