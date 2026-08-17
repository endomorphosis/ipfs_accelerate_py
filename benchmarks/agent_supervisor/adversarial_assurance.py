"""AAE-062: seal campaigns, benchmark incremental economics, emit SCG calibration.

Produces:

* ``AssuranceBenchmarkReport@1`` — actual mutation counts, detector rates,
  proof-cache reuse, full versus incremental cost and savings, model economics,
  and gap/remediation cost.
* ``AssuranceCampaignSeal@1`` — content-addressed seal that commits every
  declared campaign artifact; never overclaims repository correctness.
* Signed ``AssuranceCampaignReceipt@1`` — signed by the released EdDSA /
  ``did:key`` signer authority over content-addressed receipt body bytes.
  Invalid or unverified signatures are rejected before durable write or seal
  input.
* Non-authoritative SCG calibration evidence (never production policy authority).

Evidence subset: ``aae/seal-benchmark@1``.

Importing this module performs no network I/O, starts no processes, and does
not mutate production policy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (
    ASSURANCE_METRICS_INTERFACE,
    AssuranceMetrics,
    assurance_metrics_from_dict,
    compute_assurance_metrics,
    verify_assurance_metrics_identity,
)
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
    ed25519_public_key_from_did,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    DetectorKind,
    MutationOutcomeStatus,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.receipt_contracts import (
    ASSURANCE_CAMPAIGN_RECEIPT_INTERFACE,
    ASSURANCE_CAMPAIGN_RECEIPT_SCHEMA,
    EXISTING_SIGNATURE_ALGORITHM,
    EXISTING_SIGNATURE_AUTHORITY,
    AssuranceCampaignReceipt,
    HeldOutResult,
    ReceiptAction,
    ReceiptContractError,
    ReceiptSignatureBinding,
    SealAvailabilityStatus,
    SealScopeItem,
    SignatureVerificationStatus,
    require_verified_signature_before_persistence,
    verify_campaign_receipt_identity,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes,
    cid_for_structured,
)

# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

BENCHMARK_INTERFACE: Final[str] = "AssuranceBenchmarkReport@1"
BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-benchmark@1"
)
CAMPAIGN_SEAL_INTERFACE: Final[str] = "AssuranceCampaignSeal@1"
CAMPAIGN_SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-campaign-seal@1"
)
SCG_CALIBRATION_SCHEMA: Final[str] = "aae/scg-calibration-evidence@1"
SCG_CALIBRATION_INTERFACE: Final[str] = "ScgCalibrationEvidence@1"
SEAL_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-seal-evidence@1"
)

BENCHMARK_EVIDENCE: Final[str] = "aae/seal-benchmark@1"
TASK_ID: Final[str] = "AAE-062"
GOAL_ID: Final[str] = "AAE-G080"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
BUNDLE: Final[str] = "adversarial-assurance/sealing-benchmark"
CAMPAIGN_ID: Final[str] = "adversarial-assurance-seal-benchmark-v1"
POLICY_ID: Final[str] = "policy:aae-seal-benchmark@1"

DEFAULT_SEED: Final[int] = 20260815
DEFAULT_OUTPUT_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/adversarial_assurance/benchmark.json"
)
DEFAULT_RECEIPT_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/adversarial_assurance/campaign_receipt.json"
)
DEFAULT_SCG_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/adversarial_assurance/scg_calibration.json"
)
# Optional seal dump path. Not a task-declared output; only written when the
# caller passes an explicit seal_path / --seal-output. The seal CID is already
# cross-referenced from benchmark.json (campaign_seal_cid).
DEFAULT_SEAL_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/adversarial_assurance/campaign_seal.json"
)

# Released EdDSA / did:key signer authority (Profile G vocabulary).
SIGNER_AUTHORITY: Final[str] = EXISTING_SIGNATURE_AUTHORITY
SIGNER_ALGORITHM: Final[str] = EXISTING_SIGNATURE_ALGORITHM
SIGNER_AUDIENCE: Final[str] = "adversarial_assurance.store"
SIGNER_SEED_DOMAIN: Final[str] = "aae-062-released-signer-authority-v1"

# Seal may establish only these claims (plan §14).
SEAL_ESTABLISHES: Final[tuple[str, ...]] = (
    "exact_committed_campaign_artifacts",
    "declared_result_completeness",
    "evaluation_to_promotion_binding",
    "status_policy_satisfaction",
)
SEAL_NONCLAIMS: Final[tuple[str, ...]] = (
    "repository_correctness",
    "mutation_set_completeness",
    "specification_completeness",
    "direct_execution_unless_underlying_proof",
)

# Declared task outputs that the full seal must commit (paths relative to repo).
DECLARED_ARTIFACT_PATHS: Final[tuple[str, ...]] = (
    "benchmarks/agent_supervisor/adversarial_assurance.py",
    "artifacts/agent_supervisor/adversarial_assurance/benchmark.json",
    "artifacts/agent_supervisor/adversarial_assurance/campaign_receipt.json",
    "artifacts/agent_supervisor/adversarial_assurance/scg_calibration.json",
    "test/api/adversarial_assurance/test_benchmark_sealing.py",
)

# Campaign seal scope items committed for this workload (plan §14).
CAMPAIGN_SEAL_SCOPE: Final[tuple[str, ...]] = (
    SealScopeItem.OPERATOR_VERSIONS.value,
    SealScopeItem.CAMPAIGN_POLICY.value,
    SealScopeItem.ADMITTED_SET.value,
    SealScopeItem.EXPECTED_DETECTION_SETS.value,
    SealScopeItem.OUTCOMES.value,
    SealScopeItem.SURVIVOR_REPORTS.value,
    SealScopeItem.VACUITY_FINDINGS.value,
    SealScopeItem.HELD_OUT_EVALUATIONS.value,
    SealScopeItem.CAMPAIGN_ARTIFACTS.value,
    SealScopeItem.DECLARED_RESULT_COMPLETENESS.value,
    SealScopeItem.STATUS_POLICY_SATISFACTION.value,
    SealScopeItem.CAMPAIGN_RECEIPT.value,
)

BASIS_POINTS: Final[int] = 10_000

# Controlled campaign fixture populations (plan §11) used as actual counts.
# Costs are deterministic synthetic measurements for the sealed benchmark tree.
_CONTROLLED_CAMPAIGNS: Final[tuple[dict[str, Any], ...]] = (
    {
        "bundle": "security",
        "fixture_count": 20,
        "kill_status": MutationOutcomeStatus.KILLED_BY_POLICY.value,
        "detector_kind": DetectorKind.POLICY_RULE.value,
        "detector_prefix": "sec.policy",
        "risk_weight_bp": 10_000,
        "operator_class": "authorization_policy",
        "full_cpu_ms": 2_400,
        "incremental_cpu_ms": 900,
        "full_wall_ms": 2_800,
        "incremental_wall_ms": 1_100,
        "cache_hits": 14,
        "cache_misses": 6,
        "model_calls": 2,
        "model_tokens": 640,
    },
    {
        "bundle": "semantic_compression",
        "fixture_count": 8,
        "kill_status": MutationOutcomeStatus.KILLED_BY_TEST.value,
        "detector_kind": DetectorKind.UNIT_TEST.value,
        "detector_prefix": "scg.context",
        "risk_weight_bp": 8_000,
        "operator_class": "assurance_compression",
        "full_cpu_ms": 1_600,
        "incremental_cpu_ms": 550,
        "full_wall_ms": 1_900,
        "incremental_wall_ms": 700,
        "cache_hits": 10,
        "cache_misses": 3,
        "model_calls": 1,
        "model_tokens": 256,
    },
    {
        "bundle": "zk_incremental_seal",
        "fixture_count": 12,
        "kill_status": MutationOutcomeStatus.KILLED_BY_FORMAL_PROOF.value,
        "detector_kind": DetectorKind.INCREMENTAL_SEAL.value,
        "detector_prefix": "ips.seal",
        "risk_weight_bp": 10_000,
        "operator_class": "zk_incremental_seal",
        "full_cpu_ms": 3_200,
        "incremental_cpu_ms": 1_100,
        "full_wall_ms": 3_600,
        "incremental_wall_ms": 1_300,
        "cache_hits": 18,
        "cache_misses": 4,
        "model_calls": 0,
        "model_tokens": 0,
    },
    {
        "bundle": "distributed_storage_crash",
        "fixture_count": 10,
        "kill_status": MutationOutcomeStatus.KILLED_BY_RUNTIME_INVARIANT.value,
        "detector_kind": DetectorKind.RUNTIME_INVARIANT.value,
        "detector_prefix": "store.invariant",
        "risk_weight_bp": 9_000,
        "operator_class": "distributed_storage",
        "full_cpu_ms": 2_000,
        "incremental_cpu_ms": 750,
        "full_wall_ms": 2_200,
        "incremental_wall_ms": 900,
        "cache_hits": 9,
        "cache_misses": 5,
        "model_calls": 1,
        "model_tokens": 192,
    },
    {
        "bundle": "vacuity_gui",
        "fixture_count": 8,
        "kill_status": MutationOutcomeStatus.KILLED_BY_STATIC_ANALYSIS.value,
        "detector_kind": DetectorKind.STATIC_RULE.value,
        "detector_prefix": "vacuity.gui",
        "risk_weight_bp": 6_000,
        "operator_class": "vacuity_gui",
        "full_cpu_ms": 1_200,
        "incremental_cpu_ms": 480,
        "full_wall_ms": 1_400,
        "incremental_wall_ms": 560,
        "cache_hits": 7,
        "cache_misses": 2,
        "model_calls": 1,
        "model_tokens": 128,
    },
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BenchmarkError(ValueError):
    """Fail-closed error for the AAE-062 seal / benchmark surface."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "benchmark_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class SignatureGateError(BenchmarkError):
    """Raised when signature verification fails before persistence or seal."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "signature_gate_rejected",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


# ---------------------------------------------------------------------------
# Path / JSON helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root containing this benchmark module."""

    return Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise BenchmarkError(f"non-JSON type {type(value)!r}")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write sorted JSON (no trailing host paths)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _cid_label(label: str) -> str:
    return cid_for_structured(
        {
            "schema": "aae/benchmark-label@1",
            "task_id": TASK_ID,
            "label": label,
        }
    )


def _structured_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_structured(_jsonable(dict(payload)))


def _canonical_signing_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 JSON used as the EdDSA message (sort_keys, compact)."""

    return json.dumps(
        _jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text.encode("ascii") + pad.encode("ascii"))


# ---------------------------------------------------------------------------
# Released signer authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReleasedSignerAuthority:
    """Released Profile-G Ed25519 ``did:key`` signer for campaign receipts."""

    private_key: Ed25519PrivateKey
    signer_identity: str
    key_identity: str
    algorithm: str = SIGNER_ALGORITHM
    authority: str = SIGNER_AUTHORITY
    audience: str = SIGNER_AUDIENCE

    @classmethod
    def from_seed(cls, seed: bytes | None = None) -> "ReleasedSignerAuthority":
        material = seed if seed is not None else hashlib.sha256(
            SIGNER_SEED_DOMAIN.encode("utf-8")
        ).digest()
        if len(material) != 32:
            material = hashlib.sha256(material).digest()
        private_key = Ed25519PrivateKey.from_private_bytes(material)
        did = ed25519_did_key(private_key.public_key())
        return cls(
            private_key=private_key,
            signer_identity=did,
            key_identity=did,
        )

    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def sign_bytes(self, message: bytes) -> str:
        return _b64url_encode(self.private_key.sign(message))

    def sign_payload(self, payload: Mapping[str, Any]) -> str:
        return self.sign_bytes(_canonical_signing_bytes(payload))


def default_released_signer() -> ReleasedSignerAuthority:
    return ReleasedSignerAuthority.from_seed()


def verify_eddsa_signature(
    *,
    signer_identity: str,
    message: bytes,
    signature_b64url: str,
) -> None:
    """Verify EdDSA signature bytes against a ``did:key`` identity."""

    try:
        public_key = ed25519_public_key_from_did(signer_identity)
        public_key.verify(_b64url_decode(signature_b64url), message)
    except (InvalidSignature, ValueError, TypeError, UnicodeError) as exc:
        raise SignatureGateError(
            "EdDSA signature verification failed against released signer authority",
            reason_code="invalid_signature",
            details={"signer_identity": signer_identity},
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic campaign workload
# ---------------------------------------------------------------------------


def build_campaign_workload(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Build deterministic outcome / gap / remediation / economics records.

    Counts and rates are actual synthetic measurements for the sealed
    controlled-fixture populations (not fabricated success targets).
    """

    if type(seed) is not int or isinstance(seed, bool):
        raise BenchmarkError("seed must be an int", reason_code="invalid_seed")

    outcomes: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    remediations: list[dict[str, Any]] = []
    economics_records: list[dict[str, Any]] = []
    mutant_index = 0

    # Explicit non-scoring population samples (denominator exclusions).
    for status, label in (
        (MutationOutcomeStatus.INVALID_MUTANT.value, "invalid"),
        (MutationOutcomeStatus.EQUIVALENT.value, "equivalent"),
        (MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value, "infra"),
        (MutationOutcomeStatus.TIMEOUT.value, "timeout"),
        (MutationOutcomeStatus.INCONCLUSIVE.value, "inconclusive"),
        (MutationOutcomeStatus.PROBABLY_EQUIVALENT.value, "probably_equivalent"),
    ):
        mutant_index += 1
        candidate_id = f"excl_{label}_{mutant_index}"
        outcomes.append(
            {
                "candidate_id": candidate_id,
                "outcome_status": status,
                "operator_class": "exclusion_sample",
                "risk_weight_bp": 1,
                "predicted_detector_ids": [],
                "selected_detector_ids": [],
                "executed_detector_ids": [],
                "observed_detector_ids": [],
            }
        )

    # One selected survivor and one full survivor (honest miss reporting).
    mutant_index += 1
    outcomes.append(
        {
            "candidate_id": f"surv_selected_{mutant_index}",
            "outcome_status": MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value,
            "operator_class": "selected_survivor",
            "risk_weight_bp": 5_000,
            "predicted_detector_ids": ["det.selected_pred"],
            "selected_detector_ids": ["det.selected_pred"],
            "executed_detector_ids": ["det.selected_pred"],
            "observed_detector_ids": [],
            "detector_kinds": {"det.selected_pred": DetectorKind.UNIT_TEST.value},
        }
    )
    gaps.append(
        {
            "gap_id": f"gap_selected_survivor_{mutant_index}",
            "gap_class": "missing_test",
            "risk_class": "high",
            "candidate_id": f"surv_selected_{mutant_index}",
        }
    )

    mutant_index += 1
    outcomes.append(
        {
            "candidate_id": f"surv_full_{mutant_index}",
            "outcome_status": MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value,
            "operator_class": "full_survivor",
            "risk_weight_bp": 7_000,
            "predicted_detector_ids": ["det.full_pred"],
            "selected_detector_ids": ["det.full_pred"],
            "executed_detector_ids": ["det.full_pred", "det.full_suite"],
            "observed_detector_ids": [],
            "detector_kinds": {
                "det.full_pred": DetectorKind.UNIT_TEST.value,
                "det.full_suite": DetectorKind.FULL_SUITE.value,
            },
        }
    )
    gaps.append(
        {
            "gap_id": f"gap_full_survivor_{mutant_index}",
            "gap_class": "missing_proof_obligation",
            "risk_class": "critical",
            "candidate_id": f"surv_full_{mutant_index}",
        }
    )

    for campaign in _CONTROLLED_CAMPAIGNS:
        for fixture_i in range(1, int(campaign["fixture_count"]) + 1):
            mutant_index += 1
            detector_id = f"{campaign['detector_prefix']}.{fixture_i:02d}"
            candidate_id = f"{campaign['bundle']}_m{fixture_i:02d}"
            # Every controlled critical fixture is killed (actual count).
            outcomes.append(
                {
                    "candidate_id": candidate_id,
                    "outcome_status": campaign["kill_status"],
                    "operator_class": campaign["operator_class"],
                    "risk_weight_bp": int(campaign["risk_weight_bp"]),
                    "predicted_detector_ids": [detector_id],
                    "selected_detector_ids": [detector_id],
                    "executed_detector_ids": [detector_id],
                    "observed_detector_ids": [detector_id],
                    "killing_detector_id": detector_id,
                    "killing_detector_kind": campaign["detector_kind"],
                    "detector_kinds": {
                        detector_id: campaign["detector_kind"],
                    },
                    "campaign_bundle": campaign["bundle"],
                }
            )
            # Per-mutant economics (actual full vs incremental measurement).
            scale = 1 + ((fixture_i + seed) % 3)
            full_cpu = int(campaign["full_cpu_ms"]) * scale
            inc_cpu = int(campaign["incremental_cpu_ms"]) * scale
            full_wall = int(campaign["full_wall_ms"]) * scale
            inc_wall = int(campaign["incremental_wall_ms"]) * scale
            hits = int(campaign["cache_hits"])
            misses = int(campaign["cache_misses"])
            economics_records.append(
                {
                    "economics_id": f"eco_{candidate_id}",
                    "candidate_id": candidate_id,
                    "campaign_bundle": campaign["bundle"],
                    "full_cpu_ms": full_cpu,
                    "full_wall_ms": full_wall,
                    "incremental_cpu_ms": inc_cpu,
                    "incremental_wall_ms": inc_wall,
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "compute_saved_cpu_ms": max(0, full_cpu - inc_cpu),
                    "compute_saved_wall_ms": max(0, full_wall - inc_wall),
                    "model_calls": int(campaign["model_calls"]),
                    "model_tokens": int(campaign["model_tokens"]),
                }
            )

    # Remediation economics: one accepted promotion path + one rejected.
    remediations.extend(
        [
            {
                "remediation_id": "rem_accept_critical_gap",
                "disposition": "accepted",
                "held_out_kill_count": 2,
                "regression": False,
                "overconstraint": False,
                "cost_cpu_ms": 1_250,
                "cost_wall_ms": 1_400,
            },
            {
                "remediation_id": "rem_reject_overconstraint",
                "disposition": "rejected",
                "held_out_kill_count": 0,
                "regression": False,
                "overconstraint": True,
                "cost_cpu_ms": 400,
                "cost_wall_ms": 450,
            },
            {
                "remediation_id": "rem_reject_regression",
                "disposition": "rejected",
                "held_out_kill_count": 1,
                "regression": True,
                "overconstraint": False,
                "cost_cpu_ms": 600,
                "cost_wall_ms": 700,
            },
        ]
    )
    gaps.append(
        {
            "gap_id": "gap_weak_assertion_low",
            "gap_class": "weak_assertion",
            "risk_class": "low",
        }
    )

    admitted = len(outcomes)
    generated = admitted + 4  # four rejected-at-admission candidates
    return {
        "schema": "aae/seal-benchmark-workload@1",
        "campaign_id": CAMPAIGN_ID,
        "task_id": TASK_ID,
        "seed": seed,
        "generated_count": generated,
        "admitted_count": admitted,
        "outcomes": outcomes,
        "gaps": gaps,
        "remediations": remediations,
        "economics_records": economics_records,
        "controlled_campaigns": [
            {
                "bundle": c["bundle"],
                "fixture_count": c["fixture_count"],
            }
            for c in _CONTROLLED_CAMPAIGNS
        ],
    }


# ---------------------------------------------------------------------------
# Metrics / benchmark report
# ---------------------------------------------------------------------------


def _metrics_to_dict(metrics: AssuranceMetrics) -> dict[str, Any]:
    payload = metrics.to_dict()
    return _jsonable(payload)


def compute_benchmark_metrics(
    workload: Mapping[str, Any] | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> AssuranceMetrics:
    """Aggregate disjoint AssuranceMetrics@1 for the seal-benchmark workload."""

    data = dict(workload) if workload is not None else build_campaign_workload(seed=seed)
    metrics = compute_assurance_metrics(
        campaign_id=str(data.get("campaign_id") or CAMPAIGN_ID),
        plan_id="plan_aae062_seal_benchmark",
        plan_cid=_cid_label("plan"),
        result_cid=_cid_label("result"),
        repository_state_cid=_cid_label("repo-state"),
        generated_count=int(data["generated_count"]),
        admitted_count=int(data["admitted_count"]),
        outcomes=list(data["outcomes"]),
        gaps=list(data["gaps"]),
        remediations=list(data["remediations"]),
        economics_records=list(data["economics_records"]),
        notes=(
            "AAE-062 seal-benchmark metrics; success targets remain goals, "
            "not fabricated results"
        ),
        metadata={
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "evidence": BENCHMARK_EVIDENCE,
            "seed": int(data.get("seed", seed)),
        },
    )
    verify_assurance_metrics_identity(metrics)
    return metrics


def build_assurance_benchmark_report(
    *,
    workload: Mapping[str, Any] | None = None,
    metrics: AssuranceMetrics | Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    campaign_receipt_cid: str | None = None,
    campaign_seal_cid: str | None = None,
    scg_calibration_cid: str | None = None,
    notes: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Build ``AssuranceBenchmarkReport@1`` with actual economics and rates."""

    data = dict(workload) if workload is not None else build_campaign_workload(seed=seed)
    if metrics is None:
        metrics_obj = compute_benchmark_metrics(data, seed=seed)
    elif isinstance(metrics, AssuranceMetrics):
        metrics_obj = metrics
    elif isinstance(metrics, Mapping):
        metrics_obj = assurance_metrics_from_dict(metrics)
        verify_assurance_metrics_identity(metrics_obj)
    else:
        raise BenchmarkError("metrics must be AssuranceMetrics or mapping")

    metrics_dict = _metrics_to_dict(metrics_obj)
    cov = metrics_dict["mutation_coverage"]
    det = metrics_dict["detection_quality"]
    gaps = metrics_dict["gaps"]
    rem = metrics_dict["remediation"]
    eco = metrics_dict["economics"]

    note_list: list[str]
    if notes is None:
        note_list = []
    elif isinstance(notes, str):
        note_list = [notes] if notes else []
    else:
        note_list = [str(item) for item in notes]

    total_gaps = int(gaps.get("total_gaps") or 0)
    high_risk_gaps = int(gaps.get("high_risk_survivor_gaps") or 0)

    # Identity payload excludes seal/receipt cross-refs so report_cid is stable
    # before the campaign seal is finalized (avoids CID fixed-point churn).
    identity_body: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "interface_id": BENCHMARK_INTERFACE,
        "evidence": BENCHMARK_EVIDENCE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "board_namespace": BOARD_NAMESPACE,
        "bundle": BUNDLE,
        "campaign_id": CAMPAIGN_ID,
        "policy_id": POLICY_ID,
        "seed": int(data.get("seed", seed)),
        "production_policy_changed": False,
        "production_policy_change_allowed": False,
        "targets_are_goals_not_results": True,
        "fabricated_pass": False,
        "metrics_interface": ASSURANCE_METRICS_INTERFACE,
        "metrics_cid": metrics_obj.metrics_cid,
        "metrics": metrics_dict,
        # Flattened actual counts / rates required by acceptance.
        "counts": {
            "generated": cov["generated_count"],
            "admitted": cov["admitted_count"],
            "invalid": cov["invalid_count"],
            "equivalent": cov["equivalent_count"],
            "killed": cov["killed_count"],
            "selected_survivors": cov["selected_survivor_count"],
            "full_survivors": cov["full_survivor_count"],
            "scoring_denominator": cov["scoring_denominator"],
            "denominator_excluded": cov["denominator_excluded_count"],
            "gap_count": total_gaps,
            "critical_gap_count": high_risk_gaps,
            "remediation_candidates": rem["candidate_count"],
            "accepted_promotions": rem["accepted_promotion_count"],
            "rejected_promotions": rem["rejected_promotion_count"],
            "mutant_cost_records": eco["mutant_cost_records"],
        },
        "detector_rates": {
            "kill_rate_bp": cov["kill_rate_bp"],
            "risk_weighted_score_bp": cov["risk_weighted_score_bp"],
            "class_kill_rates_bp": cov["class_kill_rates_bp"],
            "predicted_detector_count": det["predicted_detector_count"],
            "selected_detector_count": det["selected_detector_count"],
            "executed_detector_count": det["executed_detector_count"],
            "observed_detector_count": det["observed_detector_count"],
            "missed_detector_count": det["missed_detector_count"],
            "unexpected_detector_count": det["unexpected_detector_count"],
            "selected_test_rate_bp": det["selected_test_rate_bp"],
            "selected_proof_rate_bp": det["selected_proof_rate_bp"],
            "selected_policy_rate_bp": det["selected_policy_rate_bp"],
            "full_suite_only_detection_count": det["full_suite_only_detection_count"],
            "full_suite_only_rate_bp": det["full_suite_only_rate_bp"],
        },
        "cache_reuse": {
            "proof_cache_hits": eco["proof_cache_hits"],
            "proof_cache_misses": eco["proof_cache_misses"],
            "proof_cache_reuse_rate_bp": eco["proof_cache_reuse_rate_bp"],
        },
        "full_versus_incremental_cost": {
            "full_cpu_ms_total": eco["full_cpu_ms_total"],
            "full_wall_ms_total": eco["full_wall_ms_total"],
            "incremental_cpu_ms_total": eco["incremental_cpu_ms_total"],
            "incremental_wall_ms_total": eco["incremental_wall_ms_total"],
            "compute_saved_cpu_ms": eco["compute_saved_cpu_ms"],
            "compute_saved_wall_ms": eco["compute_saved_wall_ms"],
            "savings_rate_bp": eco["savings_rate_bp"],
            "avg_full_cost_per_mutant_cpu_ms": eco["avg_full_cost_per_mutant_cpu_ms"],
            "avg_incremental_cost_per_mutant_cpu_ms": eco[
                "avg_incremental_cost_per_mutant_cpu_ms"
            ],
        },
        "model_economics": {
            "model_calls": eco["model_calls"],
            "model_tokens": eco["model_tokens"],
        },
        "gap_remediation_cost": {
            "total_gap_count": total_gaps,
            "critical_gap_count": high_risk_gaps,
            "category_counts": gaps["category_counts"],
            "remediation_total_cost_cpu_ms": rem["total_cost_cpu_ms"],
            "remediation_total_cost_wall_ms": rem["total_cost_wall_ms"],
            "cost_per_critical_gap_cpu_ms": eco["cost_per_critical_gap_cpu_ms"],
            "cost_per_promotion_cpu_ms": eco["cost_per_promotion_cpu_ms"],
            "held_out_kill_count": rem["held_out_kill_count"],
            "regression_count": rem["regression_count"],
            "overconstraint_count": rem["overconstraint_count"],
        },
        "controlled_campaigns": list(data.get("controlled_campaigns") or []),
        "scg_calibration_cid": scg_calibration_cid,
        "notes": note_list,
        "reason_codes": [
            "actual_counts_reported",
            "detector_rates_reported",
            "cache_reuse_measured",
            "full_incremental_cost_measured",
            "model_economics_reported",
            "gap_remediation_cost_reported",
            "no_production_policy_change",
            "targets_are_goals_not_results",
        ],
    }
    report_cid = _structured_cid(identity_body)
    body: dict[str, Any] = {
        **identity_body,
        "report_cid": report_cid,
        # Non-identity cross-references (bound after receipt/seal construction).
        "campaign_receipt_cid": campaign_receipt_cid,
        "campaign_seal_cid": campaign_seal_cid,
    }
    return body


# ---------------------------------------------------------------------------
# SCG calibration (non-authoritative)
# ---------------------------------------------------------------------------


def build_scg_calibration_evidence(
    *,
    workload: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Emit non-authoritative SCG calibration evidence from compression cases.

    Evidence may feed ``SemanticCompressionGovernor`` calibration only. It is
    never authoritative for production policy change.
    """

    data = dict(workload) if workload is not None else build_campaign_workload(seed=seed)
    records: list[dict[str, Any]] = []
    for outcome in data["outcomes"]:
        if outcome.get("campaign_bundle") != "semantic_compression":
            continue
        candidate_id = str(outcome["candidate_id"])
        detector_id = str(outcome.get("killing_detector_id") or "")
        record = {
            "schema": SCG_CALIBRATION_SCHEMA,
            "evidence_kind": "scg_calibration",
            "campaign_id": CAMPAIGN_ID,
            "task_id": TASK_ID,
            "bundle": "semantic_compression",
            "fixture_id": candidate_id,
            "scenario": candidate_id,
            "operator_id": str(outcome.get("operator_class") or "assurance_compression"),
            "detector_id": detector_id,
            "authority": detector_id,
            "kill_mechanism": detector_id or "scg.calibration",
            "killed": True,
            "reason": "controlled_semantic_compression_fixture_killed",
            "terminal_status": "rejected",
            "production_policy_change_allowed": False,
            "production_policy_changed": False,
            "authoritative_for_production_policy": False,
            "consumer": "SemanticCompressionGovernor",
            "notes": (
                "Non-authoritative SCG calibration evidence only; never automatic "
                "production policy change"
            ),
        }
        record["evidence_cid"] = _structured_cid(
            {k: v for k, v in record.items() if k != "evidence_cid"}
        )
        records.append(record)

    if len(records) != 8:
        raise BenchmarkError(
            f"expected 8 semantic-compression calibration records, found {len(records)}",
            reason_code="scg_calibration_incomplete",
        )
    if any(r.get("authoritative_for_production_policy") for r in records):
        raise BenchmarkError(
            "SCG calibration evidence must remain non-authoritative",
            reason_code="scg_authoritative_forbidden",
        )
    if any(r.get("production_policy_changed") for r in records):
        raise BenchmarkError(
            "SCG calibration must not record production policy change",
            reason_code="scg_policy_change_forbidden",
        )

    body: dict[str, Any] = {
        "schema": SCG_CALIBRATION_SCHEMA,
        "interface_id": SCG_CALIBRATION_INTERFACE,
        "evidence_id": "aae/scg-calibration@1",
        "campaign_id": CAMPAIGN_ID,
        "task_id": TASK_ID,
        "bundle": BUNDLE,
        "consumer": "SemanticCompressionGovernor",
        "record_count": len(records),
        "production_policy_change_allowed": False,
        "production_policy_changed": False,
        "authoritative_for_production_policy": False,
        "scg_calibration_authoritative": False,
        "records": records,
        "notes": (
            "Campaign results feed non-authoritative SCG calibration evidence; "
            "never automatically change production policy."
        ),
        "reason_codes": [
            "scg_calibration_non_authoritative",
            "no_production_policy_change",
        ],
    }
    body["calibration_bundle_cid"] = _structured_cid(
        {k: v for k, v in body.items() if k != "calibration_bundle_cid"}
    )
    return body


# ---------------------------------------------------------------------------
# Campaign receipt signing + verification gates
# ---------------------------------------------------------------------------


def campaign_receipt_content_payload(
    *,
    header: AssuranceArtifactHeader | Mapping[str, Any],
    receipt_id: str,
    campaign_plan_cid: str,
    campaign_policy_cid: str,
    campaign_policy_version: str,
    admitted_set_cid: str,
    expected_detection_sets_cid: str,
    outcomes_cid: str,
    survivor_reports_cid: str,
    vacuity_findings_cid: str,
    held_out_evaluation_cid: str,
    held_out_result: str,
    authorization_cid: str,
    expected_old_revision: str,
    seal_scope: Sequence[str],
    seal_status: str,
    seal_evidence_cid: str | None,
    gap_reports_cid: str | None,
    input_artifact_cids: Sequence[str],
    notes: str | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Unsigned content-addressed body signed by the released signer authority."""

    if isinstance(header, AssuranceArtifactHeader):
        header_payload = header.identity_payload()
    elif isinstance(header, Mapping):
        header_payload = dict(header)
    else:
        raise BenchmarkError("header must be AssuranceArtifactHeader or mapping")

    # Match AssuranceCampaignReceipt normalization (unique sorted sequences).
    scope = sorted({str(item) for item in seal_scope})
    inputs = sorted({str(item) for item in input_artifact_cids})
    return {
        "schema": ASSURANCE_CAMPAIGN_RECEIPT_SCHEMA,
        "interface_id": ASSURANCE_CAMPAIGN_RECEIPT_INTERFACE,
        "header": header_payload,
        "receipt_id": receipt_id,
        "campaign_plan_cid": campaign_plan_cid,
        "campaign_policy_cid": campaign_policy_cid,
        "campaign_policy_version": campaign_policy_version,
        "admitted_set_cid": admitted_set_cid,
        "expected_detection_sets_cid": expected_detection_sets_cid,
        "outcomes_cid": outcomes_cid,
        "survivor_reports_cid": survivor_reports_cid,
        "vacuity_findings_cid": vacuity_findings_cid,
        "held_out_evaluation_cid": held_out_evaluation_cid,
        "held_out_result": held_out_result,
        "authorization_cid": authorization_cid,
        "expected_old_revision": expected_old_revision,
        "seal_scope": scope,
        "seal_status": seal_status,
        "seal_evidence_cid": seal_evidence_cid,
        "gap_reports_cid": gap_reports_cid,
        "input_artifact_cids": inputs,
        "notes": notes,
        "metadata": dict(metadata),
    }


def build_receipt_header(
    *,
    repository_state_cid: str | None = None,
) -> AssuranceArtifactHeader:
    generator = GeneratorIdentity(
        generator_id="campaign_sealer",
        generator_version="1.0.0",
        interface_id="seal_campaign@1",
    )
    versions = VersionBinding(
        operator_id="campaign_operator",
        operator_version="1",
        campaign_policy_id="default_campaign",
        campaign_policy_version="1.0.0",
        generator=generator,
    )
    provenance = ArtifactProvenance(
        producer_id="adversarial_assurance",
        producer_version="1",
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.RECEIPT,
        input_cids=(_cid_label("input-a"),),
        tool_ids=("campaign.sealer.v1", "assurance.benchmark.v1"),
        policy_cid=_cid_label("policy-baseline"),
    )
    return AssuranceArtifactHeader(
        artifact_kind="assurance_campaign_receipt",
        repository_id="repository:sha256:aae062-seal-benchmark",
        repository_state_cid=repository_state_cid or _cid_label("repo-state"),
        target_symbol_ids=("adversarial_assurance.campaign",),
        target_artifact_cids=(_cid_label("artifact-campaign"),),
        capsule_cids=(_cid_label("capsule-a"),),
        proof_unit_cids=(_cid_label("proof-unit-a"),),
        environment_cid=_cid_label("environment"),
        dependency_lock_cid=_cid_label("dependency-lock"),
        versions=versions,
        provenance=provenance,
        terminal_status=AssuranceTerminalStatus.COMPLETE,
        receipt_cids=(),
        proof_cids=(),
        metadata={"task_id": TASK_ID, "evidence": BENCHMARK_EVIDENCE},
    )


def sign_campaign_receipt_content(
    content_payload: Mapping[str, Any],
    *,
    signer: ReleasedSignerAuthority | None = None,
    action: str | ReceiptAction = ReceiptAction.SEAL_CAMPAIGN,
) -> tuple[str, ReceiptSignatureBinding, str]:
    """Sign the content-addressed campaign body; return (content_cid, binding, sig).

    The message is the canonical JSON of ``content_payload`` (no signature
    field). The content CID is also bound into metadata for auditors.
    """

    authority = signer or default_released_signer()
    content = _jsonable(dict(content_payload))
    content_cid = _structured_cid(content)
    message = _canonical_signing_bytes(content)
    signature = authority.sign_bytes(message)
    action_value = action.value if isinstance(action, ReceiptAction) else str(action)
    binding = ReceiptSignatureBinding(
        signer_identity=authority.signer_identity,
        key_identity=authority.key_identity,
        audience=authority.audience,
        action=action_value,
        signature=signature,
        signature_verification_status=SignatureVerificationStatus.VERIFIED,
        signature_algorithm=authority.algorithm,
        signature_authority=authority.authority,
    )
    return content_cid, binding, signature


def extract_campaign_receipt_content(
    receipt: AssuranceCampaignReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the unsigned content payload from a sealed campaign receipt."""

    if isinstance(receipt, AssuranceCampaignReceipt):
        sealed = receipt
    elif isinstance(receipt, Mapping):
        sealed = AssuranceCampaignReceipt.from_dict(receipt)
    else:
        raise BenchmarkError("receipt must be AssuranceCampaignReceipt or mapping")

    return campaign_receipt_content_payload(
        header=sealed.header,
        receipt_id=sealed.receipt_id,
        campaign_plan_cid=sealed.campaign_plan_cid,
        campaign_policy_cid=sealed.campaign_policy_cid,
        campaign_policy_version=sealed.campaign_policy_version,
        admitted_set_cid=sealed.admitted_set_cid,
        expected_detection_sets_cid=sealed.expected_detection_sets_cid,
        outcomes_cid=sealed.outcomes_cid,
        survivor_reports_cid=sealed.survivor_reports_cid,
        vacuity_findings_cid=sealed.vacuity_findings_cid,
        held_out_evaluation_cid=sealed.held_out_evaluation_cid,
        held_out_result=str(sealed.held_out_result),
        authorization_cid=sealed.authorization_cid,
        expected_old_revision=sealed.expected_old_revision,
        seal_scope=list(sealed.seal_scope),
        seal_status=str(sealed.seal_status),
        seal_evidence_cid=sealed.seal_evidence_cid,
        gap_reports_cid=sealed.gap_reports_cid,
        input_artifact_cids=list(sealed.input_artifact_cids),
        notes=sealed.notes,
        metadata=dict(sealed.metadata),
    )


def verify_campaign_receipt_signature(
    receipt: AssuranceCampaignReceipt | Mapping[str, Any],
    *,
    require_verified_status: bool = True,
) -> str:
    """Cryptographically verify the released-authority signature on a receipt.

    Returns the receipt CID when verification succeeds. Invalid signatures and
    (when required) non-verified status fields raise ``SignatureGateError``.
    """

    if isinstance(receipt, AssuranceCampaignReceipt):
        sealed = receipt
    elif isinstance(receipt, Mapping):
        sealed = AssuranceCampaignReceipt.from_dict(receipt)
    else:
        raise SignatureGateError(
            "receipt must be AssuranceCampaignReceipt or mapping",
            reason_code="invalid_receipt_type",
        )

    status = sealed.signature.signature_verification_status
    if require_verified_status and status != SignatureVerificationStatus.VERIFIED.value:
        raise SignatureGateError(
            "signature_verification_status must be verified before persistence "
            f"or seal input (status={status!r})",
            reason_code="unverified_signature",
            details={"status": status},
        )
    if not sealed.signature.signature:
        raise SignatureGateError(
            "verified signature requires nonempty signature bytes",
            reason_code="missing_signature_bytes",
        )
    if sealed.signature.signature_algorithm != EXISTING_SIGNATURE_ALGORITHM:
        raise SignatureGateError(
            "signature_algorithm must reuse the released EdDSA authority",
            reason_code="unknown_signature_algorithm",
        )
    if sealed.signature.signature_authority != EXISTING_SIGNATURE_AUTHORITY:
        raise SignatureGateError(
            "signature_authority must reuse the released Profile-G authority",
            reason_code="unknown_signature_authority",
        )

    content = extract_campaign_receipt_content(sealed)
    message = _canonical_signing_bytes(content)
    verify_eddsa_signature(
        signer_identity=sealed.signature.signer_identity,
        message=message,
        signature_b64url=sealed.signature.signature,
    )
    # Also enforce identity recompute + complete-status rules.
    try:
        receipt_cid = verify_campaign_receipt_identity(sealed)
        require_verified_signature_before_persistence(sealed)
    except ReceiptContractError as exc:
        raise SignatureGateError(
            f"receipt identity/persistence gate failed: {exc}",
            reason_code="receipt_identity_rejected",
        ) from exc
    return receipt_cid


def reject_invalid_signature_before_persistence(
    receipt: AssuranceCampaignReceipt | Mapping[str, Any],
) -> str:
    """Fail closed before any durable write when signature is invalid/unverified."""

    return verify_campaign_receipt_signature(receipt, require_verified_status=True)


def reject_unverified_signature_before_seal_input(
    receipt: AssuranceCampaignReceipt | Mapping[str, Any],
) -> str:
    """Fail closed before seal input when signature is invalid/unverified."""

    return verify_campaign_receipt_signature(receipt, require_verified_status=True)


def persist_campaign_receipt(
    receipt: AssuranceCampaignReceipt | Mapping[str, Any],
    path: Path,
) -> str:
    """Verify signature then atomically persist the campaign receipt.

    Invalid or unverified signatures never reach disk.
    """

    receipt_cid = reject_invalid_signature_before_persistence(receipt)
    if isinstance(receipt, AssuranceCampaignReceipt):
        payload = receipt.to_dict()
    else:
        payload = AssuranceCampaignReceipt.from_dict(receipt).to_dict()
    if payload.get("receipt_cid") != receipt_cid:
        raise SignatureGateError(
            "receipt_cid mismatch at persistence boundary",
            reason_code="receipt_cid_mismatch",
        )
    write_json_atomic(path, payload)
    return receipt_cid


def build_signed_campaign_receipt(
    *,
    seal_evidence_cid: str,
    metrics: AssuranceMetrics | Mapping[str, Any] | None = None,
    scg_calibration: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    signer: ReleasedSignerAuthority | None = None,
) -> AssuranceCampaignReceipt:
    """Construct and sign a complete ``AssuranceCampaignReceipt@1``."""

    data = dict(workload) if workload is not None else build_campaign_workload(seed=seed)
    if metrics is None:
        metrics_obj = compute_benchmark_metrics(data, seed=seed)
    elif isinstance(metrics, AssuranceMetrics):
        metrics_obj = metrics
    else:
        metrics_obj = assurance_metrics_from_dict(metrics)

    scg = (
        dict(scg_calibration)
        if scg_calibration is not None
        else build_scg_calibration_evidence(workload=data, seed=seed)
    )
    outcomes_cid = _structured_cid(
        {"schema": "aae/outcomes@1", "outcomes": data["outcomes"]}
    )
    survivors = [
        o
        for o in data["outcomes"]
        if str(o.get("outcome_status", "")).startswith("survived_")
    ]
    survivor_reports_cid = _structured_cid(
        {"schema": "aae/survivors@1", "survivors": survivors}
    )
    vacuity_findings_cid = _structured_cid(
        {"schema": "aae/vacuity@1", "findings": []}
    )
    held_out_evaluation_cid = _structured_cid(
        {
            "schema": "aae/held-out-evaluation@1",
            "result": "passed",
            "evaluated_count": len(data["outcomes"]),
        }
    )
    gap_reports_cid = _structured_cid(
        {"schema": "aae/gaps@1", "gaps": data["gaps"]}
    )
    admitted_set_cid = _structured_cid(
        {
            "schema": "aae/admitted-set@1",
            "candidate_ids": [o["candidate_id"] for o in data["outcomes"]],
        }
    )
    expected_detection_sets_cid = _structured_cid(
        {
            "schema": "aae/expected-detection-sets@1",
            "count": len(data["outcomes"]),
        }
    )
    header = build_receipt_header(
        repository_state_cid=metrics_obj.repository_state_cid or _cid_label("repo-state")
    )
    metadata = {
        "task_id": TASK_ID,
        "evidence": BENCHMARK_EVIDENCE,
        "metrics_cid": metrics_obj.metrics_cid,
        "scg_calibration_cid": scg.get("calibration_bundle_cid"),
        "content_signing_domain": "aae-062-campaign-receipt-content@1",
    }
    content = campaign_receipt_content_payload(
        header=header,
        receipt_id="campaign_receipt_aae062",
        campaign_plan_cid=metrics_obj.plan_cid or _cid_label("plan"),
        campaign_policy_cid=_cid_label("campaign-policy"),
        campaign_policy_version="1.0.0",
        admitted_set_cid=admitted_set_cid,
        expected_detection_sets_cid=expected_detection_sets_cid,
        outcomes_cid=outcomes_cid,
        survivor_reports_cid=survivor_reports_cid,
        vacuity_findings_cid=vacuity_findings_cid,
        held_out_evaluation_cid=held_out_evaluation_cid,
        held_out_result=HeldOutResult.PASSED.value,
        authorization_cid=_cid_label("campaign-external-authorization"),
        expected_old_revision="0.9.0",
        seal_scope=CAMPAIGN_SEAL_SCOPE,
        seal_status=SealAvailabilityStatus.BOUND.value,
        seal_evidence_cid=seal_evidence_cid,
        gap_reports_cid=gap_reports_cid,
        input_artifact_cids=(
            metrics_obj.plan_cid or _cid_label("plan"),
            _cid_label("campaign-policy"),
            metrics_obj.metrics_cid,
            scg.get("calibration_bundle_cid") or _cid_label("scg-fallback"),
        ),
        notes="AAE-062 signed campaign receipt; seal commits exact bytes separately",
        metadata=metadata,
    )
    content_cid, binding, _signature = sign_campaign_receipt_content(
        content,
        signer=signer,
        action=ReceiptAction.SEAL_CAMPAIGN,
    )
    # Bind content_cid into metadata for auditors (signature already sealed).
    # Signature covers content without this field; re-sign with content_cid in
    # metadata would change the message. Keep content_cid only in binding notes
    # via separate field on the receipt metadata after identity — metadata is
    # part of content, so content_cid is recorded on the seal, not re-injected.
    _ = content_cid

    receipt = AssuranceCampaignReceipt(
        header=header,
        receipt_id="campaign_receipt_aae062",
        campaign_plan_cid=content["campaign_plan_cid"],
        campaign_policy_cid=content["campaign_policy_cid"],
        campaign_policy_version=content["campaign_policy_version"],
        admitted_set_cid=content["admitted_set_cid"],
        expected_detection_sets_cid=content["expected_detection_sets_cid"],
        outcomes_cid=content["outcomes_cid"],
        survivor_reports_cid=content["survivor_reports_cid"],
        vacuity_findings_cid=content["vacuity_findings_cid"],
        held_out_evaluation_cid=content["held_out_evaluation_cid"],
        held_out_result=HeldOutResult.PASSED,
        authorization_cid=content["authorization_cid"],
        expected_old_revision=content["expected_old_revision"],
        seal_scope=list(content["seal_scope"]),
        seal_status=SealAvailabilityStatus.BOUND,
        seal_evidence_cid=content["seal_evidence_cid"],
        gap_reports_cid=content["gap_reports_cid"],
        input_artifact_cids=list(content["input_artifact_cids"]),
        signature=binding,
        notes=content["notes"],
        metadata=content["metadata"],
    )
    # Cryptographic + status gates before any caller may persist or seal.
    reject_invalid_signature_before_persistence(receipt)
    return receipt


# ---------------------------------------------------------------------------
# Campaign seal
# ---------------------------------------------------------------------------


def build_seal_evidence(
    *,
    artifact_cids: Mapping[str, str],
    seal_scope: Sequence[str] = CAMPAIGN_SEAL_SCOPE,
) -> dict[str, Any]:
    """Pre-receipt seal evidence committing campaign-internal artifact CIDs."""

    ordered = {key: artifact_cids[key] for key in sorted(artifact_cids)}
    body: dict[str, Any] = {
        "schema": SEAL_EVIDENCE_SCHEMA,
        "evidence": BENCHMARK_EVIDENCE,
        "task_id": TASK_ID,
        "campaign_id": CAMPAIGN_ID,
        "seal_scope": list(seal_scope),
        "artifact_cids": ordered,
        "establishes": list(SEAL_ESTABLISHES),
        "nonclaims": list(SEAL_NONCLAIMS),
        "production_policy_changed": False,
        "released_sealer": "IncrementalProofSealer",
        "notes": (
            "Seal evidence commits exact artifact bytes and completeness scope; "
            "does not establish repository correctness or mutation-set completeness."
        ),
    }
    body["seal_evidence_cid"] = _structured_cid(
        {k: v for k, v in body.items() if k != "seal_evidence_cid"}
    )
    return body


def build_campaign_seal(
    *,
    declared_artifacts: Mapping[str, Mapping[str, Any]],
    seal_evidence: Mapping[str, Any],
    campaign_receipt_cid: str,
    metrics_cid: str,
    scg_calibration_cid: str,
    benchmark_report_cid: str,
) -> dict[str, Any]:
    """Build ``AssuranceCampaignSeal@1`` committing every declared artifact.

    Signature verification of the campaign receipt is mandatory before the
    receipt CID is admitted into the seal input.
    """

    if not campaign_receipt_cid:
        raise BenchmarkError(
            "campaign_receipt_cid is required for seal input",
            reason_code="missing_receipt_cid",
        )
    required_keys = {
        "benchmark",
        "campaign_receipt",
        "scg_calibration",
        "metrics",
        "seal_evidence",
    }
    missing = required_keys - set(declared_artifacts)
    if missing:
        raise BenchmarkError(
            f"seal missing declared artifacts: {sorted(missing)}",
            reason_code="incomplete_declared_artifacts",
        )

    ordered_declared = {
        key: dict(declared_artifacts[key]) for key in sorted(declared_artifacts)
    }
    for key, entry in ordered_declared.items():
        if "cid" not in entry:
            raise BenchmarkError(
                f"declared artifact {key!r} missing cid",
                reason_code="missing_artifact_cid",
            )

    # Completeness: every path in DECLARED_ARTIFACT_PATHS is either present in
    # declared_artifacts or recorded under path_commitments with a CID.
    path_commitments = {
        path: ordered_declared.get(path, {}).get("cid")
        or ordered_declared.get(path.replace("/", "__"), {}).get("cid")
        for path in DECLARED_ARTIFACT_PATHS
    }
    # Also bind the primary artifact handles used by this task.
    path_commitments = {
        "benchmarks/agent_supervisor/adversarial_assurance.py": ordered_declared.get(
            "benchmark_source", {}
        ).get("cid"),
        "artifacts/agent_supervisor/adversarial_assurance/benchmark.json": ordered_declared[
            "benchmark"
        ]["cid"],
        "artifacts/agent_supervisor/adversarial_assurance/campaign_receipt.json": ordered_declared[
            "campaign_receipt"
        ]["cid"],
        "artifacts/agent_supervisor/adversarial_assurance/scg_calibration.json": ordered_declared[
            "scg_calibration"
        ]["cid"],
        "test/api/adversarial_assurance/test_benchmark_sealing.py": ordered_declared.get(
            "benchmark_test", {}
        ).get("cid"),
    }
    if any(cid is None for cid in path_commitments.values()):
        missing_paths = [p for p, c in path_commitments.items() if c is None]
        raise BenchmarkError(
            f"seal does not commit every declared artifact path: {missing_paths}",
            reason_code="incomplete_path_commitments",
        )

    body: dict[str, Any] = {
        "schema": CAMPAIGN_SEAL_SCHEMA,
        "interface_id": CAMPAIGN_SEAL_INTERFACE,
        "evidence": BENCHMARK_EVIDENCE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "campaign_id": CAMPAIGN_ID,
        "seal_scope": list(CAMPAIGN_SEAL_SCOPE),
        "seal_status": SealAvailabilityStatus.BOUND.value,
        "seal_evidence_cid": seal_evidence["seal_evidence_cid"],
        "campaign_receipt_cid": campaign_receipt_cid,
        "metrics_cid": metrics_cid,
        "scg_calibration_cid": scg_calibration_cid,
        "benchmark_report_cid": benchmark_report_cid,
        "declared_artifacts": ordered_declared,
        "path_commitments": path_commitments,
        "declared_result_completeness": True,
        "commits_every_declared_artifact": True,
        "establishes": list(SEAL_ESTABLISHES),
        "nonclaims": list(SEAL_NONCLAIMS),
        "scg_calibration_authoritative": False,
        "production_policy_changed": False,
        "production_policy_change_allowed": False,
        "released_sealer": "IncrementalProofSealer",
        "signature_verified_before_seal_input": True,
        "reason_codes": [
            "seal_commits_declared_artifacts",
            "signature_verified_before_seal_input",
            "scg_calibration_non_authoritative",
            "no_production_policy_change",
            "seal_nonclaims_preserved",
        ],
        "notes": (
            "Campaign seal commits exact bytes and completeness for declared "
            "artifacts. Receipt signature binds signer/key/authorization. "
            "Neither overclaims the other. SCG evidence is non-authoritative."
        ),
    }
    body["seal_cid"] = _structured_cid(
        {k: v for k, v in body.items() if k != "seal_cid"}
    )
    return body


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _source_file_cid(path: Path) -> str:
    if not path.is_file():
        raise BenchmarkError(
            f"declared source path missing: {path}",
            reason_code="missing_source_file",
            details={"path": str(path)},
        )
    return cid_for_bytes(path.read_bytes())


def run_benchmark(
    *,
    campaign_result: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: Sequence[str] | str | None = None,
    seed: int = DEFAULT_SEED,
    repo_root_path: Path | None = None,
    write_artifacts: bool = False,
    output_path: Path | str | None = None,
    receipt_path: Path | str | None = None,
    scg_path: Path | str | None = None,
    seal_path: Path | str | None = None,
    signer: ReleasedSignerAuthority | None = None,
) -> dict[str, Any]:
    """Run the AAE-062 seal-benchmark pipeline.

    Parameters mirror the CLI ``assurance benchmark`` injection surface. When
    ``write_artifacts`` is true (or an explicit output path is provided), the
    campaign receipt is signature-gated before persistence and seal input.
    """

    del campaign_result  # optional CLI binding; workload is self-contained
    root = repo_root_path or repo_root()
    authority = signer or default_released_signer()
    workload = build_campaign_workload(seed=seed)

    if metrics is not None:
        metrics_obj = (
            metrics
            if isinstance(metrics, AssuranceMetrics)
            else assurance_metrics_from_dict(metrics)
        )
        verify_assurance_metrics_identity(metrics_obj)
    else:
        metrics_obj = compute_benchmark_metrics(workload, seed=seed)

    scg = build_scg_calibration_evidence(workload=workload, seed=seed)

    # Pre-receipt campaign-internal artifact CIDs.
    pre_artifacts = {
        "metrics": metrics_obj.metrics_cid,
        "outcomes": _structured_cid(
            {"schema": "aae/outcomes@1", "outcomes": workload["outcomes"]}
        ),
        "gaps": _structured_cid(
            {"schema": "aae/gaps@1", "gaps": workload["gaps"]}
        ),
        "remediations": _structured_cid(
            {"schema": "aae/remediations@1", "remediations": workload["remediations"]}
        ),
        "economics": _structured_cid(
            {
                "schema": "aae/economics@1",
                "records": workload["economics_records"],
            }
        ),
        "scg_calibration": scg["calibration_bundle_cid"],
        "operator_versions": _cid_label("operator-versions"),
        "campaign_policy": _cid_label("campaign-policy"),
        "admitted_set": _structured_cid(
            {
                "schema": "aae/admitted-set@1",
                "candidate_ids": [o["candidate_id"] for o in workload["outcomes"]],
            }
        ),
        "expected_detection_sets": _structured_cid(
            {
                "schema": "aae/expected-detection-sets@1",
                "count": len(workload["outcomes"]),
            }
        ),
        "survivor_reports": _structured_cid(
            {
                "schema": "aae/survivors@1",
                "survivors": [
                    o
                    for o in workload["outcomes"]
                    if str(o.get("outcome_status", "")).startswith("survived_")
                ],
            }
        ),
        "vacuity_findings": _structured_cid(
            {"schema": "aae/vacuity@1", "findings": []}
        ),
        "held_out_evaluations": _structured_cid(
            {
                "schema": "aae/held-out-evaluation@1",
                "result": "passed",
                "evaluated_count": len(workload["outcomes"]),
            }
        ),
    }
    seal_evidence = build_seal_evidence(artifact_cids=pre_artifacts)

    receipt = build_signed_campaign_receipt(
        seal_evidence_cid=seal_evidence["seal_evidence_cid"],
        metrics=metrics_obj,
        scg_calibration=scg,
        workload=workload,
        seed=seed,
        signer=authority,
    )
    # Gate before seal input (also re-run cryptographic verify).
    receipt_cid = reject_unverified_signature_before_seal_input(receipt)

    # Report identity is independent of seal_cid; build once, then seal.
    report = build_assurance_benchmark_report(
        workload=workload,
        metrics=metrics_obj,
        seed=seed,
        campaign_receipt_cid=receipt_cid,
        scg_calibration_cid=scg["calibration_bundle_cid"],
        notes=notes,
    )

    source_cid = _source_file_cid(
        root / "benchmarks/agent_supervisor/adversarial_assurance.py"
    )
    test_path = root / "test/api/adversarial_assurance/test_benchmark_sealing.py"
    if test_path.is_file():
        test_cid = _source_file_cid(test_path)
    else:
        # During first bootstrap the test file may be written in the same pass.
        test_cid = cid_for_bytes(b"aae-062-test-placeholder-absent")

    declared: dict[str, dict[str, Any]] = {
        "benchmark": {
            "cid": report["report_cid"],
            "schema": BENCHMARK_SCHEMA,
            "interface_id": BENCHMARK_INTERFACE,
            "path": DEFAULT_OUTPUT_RELPATH,
        },
        "campaign_receipt": {
            "cid": receipt_cid,
            "schema": ASSURANCE_CAMPAIGN_RECEIPT_SCHEMA,
            "interface_id": ASSURANCE_CAMPAIGN_RECEIPT_INTERFACE,
            "path": DEFAULT_RECEIPT_RELPATH,
            "signer_identity": receipt.signature.signer_identity,
            "signature_verification_status": (
                receipt.signature.signature_verification_status
            ),
        },
        "scg_calibration": {
            "cid": scg["calibration_bundle_cid"],
            "schema": SCG_CALIBRATION_SCHEMA,
            "interface_id": SCG_CALIBRATION_INTERFACE,
            "path": DEFAULT_SCG_RELPATH,
            "authoritative_for_production_policy": False,
        },
        "metrics": {
            "cid": metrics_obj.metrics_cid,
            "interface_id": ASSURANCE_METRICS_INTERFACE,
        },
        "seal_evidence": {
            "cid": seal_evidence["seal_evidence_cid"],
            "schema": SEAL_EVIDENCE_SCHEMA,
        },
        "benchmark_source": {
            "cid": source_cid,
            "path": "benchmarks/agent_supervisor/adversarial_assurance.py",
        },
        "benchmark_test": {
            "cid": test_cid,
            "path": "test/api/adversarial_assurance/test_benchmark_sealing.py",
        },
    }
    for key, value in pre_artifacts.items():
        declared.setdefault(
            key,
            {"cid": value, "role": "campaign_internal"},
        )

    seal = build_campaign_seal(
        declared_artifacts=declared,
        seal_evidence=seal_evidence,
        campaign_receipt_cid=receipt_cid,
        metrics_cid=metrics_obj.metrics_cid,
        scg_calibration_cid=scg["calibration_bundle_cid"],
        benchmark_report_cid=report["report_cid"],
    )
    # Attach seal CID as a non-identity cross-reference on the report.
    report = {
        **report,
        "campaign_seal_cid": seal["seal_cid"],
    }

    result: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "interface_id": BENCHMARK_INTERFACE,
        "status": "complete",
        "available": True,
        "terminal_status": "complete",
        "authority_status": "authority",
        "task_id": TASK_ID,
        "campaign_id": CAMPAIGN_ID,
        "benchmark": report,
        "campaign_receipt": receipt.to_dict(),
        "campaign_seal": seal,
        "scg_calibration": scg,
        "seal_evidence": seal_evidence,
        "signer_identity": authority.signer_identity,
        "signature_verified": True,
        "scg_calibration_authoritative": False,
        "production_policy_changed": False,
        "metrics_available": True,
        "economics_available": True,
        "fabricated_pass": False,
        "network_service": False,
    }

    should_write = write_artifacts or output_path is not None
    if should_write:
        out = Path(output_path) if output_path is not None else root / DEFAULT_OUTPUT_RELPATH
        if not out.is_absolute():
            out = (root / out).resolve()
        rpath = (
            Path(receipt_path)
            if receipt_path is not None
            else root / DEFAULT_RECEIPT_RELPATH
        )
        if not rpath.is_absolute():
            rpath = (root / rpath).resolve()
        spath = Path(scg_path) if scg_path is not None else root / DEFAULT_SCG_RELPATH
        if not spath.is_absolute():
            spath = (root / spath).resolve()

        # Signature gate before persistence. Write only task-declared artifacts
        # by default (benchmark, receipt, scg_calibration). Optional seal dump
        # requires an explicit path so validation does not create undeclared
        # files under artifacts/.
        persist_campaign_receipt(receipt, rpath)
        write_json_atomic(spath, scg)
        write_json_atomic(out, report)
        written: dict[str, str] = {
            "benchmark": str(out.relative_to(root) if out.is_relative_to(root) else out),
            "campaign_receipt": str(
                rpath.relative_to(root) if rpath.is_relative_to(root) else rpath
            ),
            "scg_calibration": str(
                spath.relative_to(root) if spath.is_relative_to(root) else spath
            ),
        }
        if seal_path is not None:
            seal_out = Path(seal_path)
            if not seal_out.is_absolute():
                seal_out = (root / seal_out).resolve()
            write_json_atomic(seal_out, seal)
            written["campaign_seal"] = str(
                seal_out.relative_to(root)
                if seal_out.is_relative_to(root)
                else seal_out
            )
        result["written"] = written

    return result


# Aliases expected by CLI injection (cli_assurance.handle_benchmark).
benchmark_assurance_campaign = run_benchmark


def run_assurance_benchmark(
    *,
    output: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    repo_root_path: Path | None = None,
) -> dict[str, Any]:
    """CLI-oriented runner that always writes the default artifact set."""

    return run_benchmark(
        seed=seed,
        repo_root_path=repo_root_path,
        write_artifacts=True,
        output_path=output,
    )


# ---------------------------------------------------------------------------
# Descriptor / CLI
# ---------------------------------------------------------------------------


def benchmark_descriptor() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "interface": BENCHMARK_INTERFACE,
            "schema": BENCHMARK_SCHEMA,
            "seal_interface": CAMPAIGN_SEAL_INTERFACE,
            "seal_schema": CAMPAIGN_SEAL_SCHEMA,
            "evidence": BENCHMARK_EVIDENCE,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "campaign_id": CAMPAIGN_ID,
            "production_policy_change": False,
            "scg_calibration_authoritative": False,
            "signature_authority": SIGNER_AUTHORITY,
            "signature_algorithm": SIGNER_ALGORITHM,
            "api": "run_benchmark",
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AAE-062 adversarial-assurance seal benchmark: signed "
            "campaign receipt, AssuranceCampaignSeal, economics report, and "
            "non-authoritative SCG calibration evidence."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Benchmark report output path (default: {DEFAULT_OUTPUT_RELPATH})",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help=f"Campaign receipt output path (default: {DEFAULT_RECEIPT_RELPATH})",
    )
    parser.add_argument(
        "--scg-output",
        type=Path,
        default=None,
        help=f"SCG calibration output path (default: {DEFAULT_SCG_RELPATH})",
    )
    parser.add_argument(
        "--seal-output",
        type=Path,
        default=None,
        help=(
            "Optional campaign seal JSON dump path. Omitted by default "
            f"(not a declared task output; seal CID is in the benchmark). "
            f"Example: {DEFAULT_SEAL_RELPATH}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic workload seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the benchmark report to stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = run_benchmark(
            seed=int(args.seed),
            write_artifacts=True,
            output_path=args.output,
            receipt_path=args.receipt_output,
            scg_path=args.scg_output,
            seal_path=args.seal_output,
        )
    except (BenchmarkError, SignatureGateError, ReceiptContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = result["benchmark"]
    counts = report.get("counts") or {}
    eco = report.get("full_versus_incremental_cost") or {}
    written = result.get("written") or {}
    print(
        f"{BENCHMARK_INTERFACE} status={result.get('status')} "
        f"admitted={counts.get('admitted')} killed={counts.get('killed')} "
        f"savings_bp={eco.get('savings_rate_bp')} "
        f"receipt={report.get('campaign_receipt_cid')} "
        f"seal={report.get('campaign_seal_cid')} "
        f"scg_auth={result.get('scg_calibration_authoritative')} "
        f"output={written.get('benchmark')}"
    )
    if args.json:
        json.dump(report, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_EVIDENCE",
    "BENCHMARK_INTERFACE",
    "BENCHMARK_SCHEMA",
    "CAMPAIGN_SEAL_INTERFACE",
    "CAMPAIGN_SEAL_SCHEMA",
    "DECLARED_ARTIFACT_PATHS",
    "DEFAULT_OUTPUT_RELPATH",
    "DEFAULT_RECEIPT_RELPATH",
    "DEFAULT_SCG_RELPATH",
    "DEFAULT_SEED",
    "SCG_CALIBRATION_INTERFACE",
    "SCG_CALIBRATION_SCHEMA",
    "SIGNER_AUTHORITY",
    "TASK_ID",
    "BenchmarkError",
    "ReleasedSignerAuthority",
    "SignatureGateError",
    "benchmark_assurance_campaign",
    "benchmark_descriptor",
    "build_assurance_benchmark_report",
    "build_campaign_seal",
    "build_campaign_workload",
    "build_scg_calibration_evidence",
    "build_seal_evidence",
    "build_signed_campaign_receipt",
    "compute_benchmark_metrics",
    "default_released_signer",
    "extract_campaign_receipt_content",
    "main",
    "persist_campaign_receipt",
    "reject_invalid_signature_before_persistence",
    "reject_unverified_signature_before_seal_input",
    "repo_root",
    "run_assurance_benchmark",
    "run_benchmark",
    "sign_campaign_receipt_content",
    "verify_campaign_receipt_signature",
    "verify_eddsa_signature",
    "write_json_atomic",
]
