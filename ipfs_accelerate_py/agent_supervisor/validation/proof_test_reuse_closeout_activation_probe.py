"""Honest production-runtime activation probes for closeout (PTR-149…155).

Probes live capability surfaces and code-path availability, then builds
repair-evidence via :func:`build_production_runtime_activation_evidence`.

Never invents reviewed v4 key authority, never claims warm-skip when the live
activation report says the gap is present, and never flips ``activation_gap``
false without a ready test-certificate authority.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .proof_test_reuse_closeout_materializer import CloseoutMaterializerIdentity
from .proof_test_reuse_current_tree_gate import (
    PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
    PRODUCTION_RUNTIME_ACTIVATION_ID,
    build_production_runtime_activation_evidence,
)

ACTIVATION_PROBE_INTERFACE: Final = "ProofTestReuseCloseoutActivationProbe@1"
ACTIVATION_PROBE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-closeout-activation-probe@1"
)

# Ordered operator handoff for remaining activation flags.
ACTIVATION_CLAIM_FIELDS: Final = (
    "zero_false_skip_assurance",
    "activation_e2e_passed",
    "zero_injection_default_path",
    "three_repository_cold_warm",
    "real_groth16_certificate",
    "measured_subprocess_benchmark",
    "historical_activation_claims_superseded",
    "controller_owned_receipt_candidate_context",
    "retained_proof_bearing_issuance_material",
    "exact_reviewed_source_binary_capability_circuit_key_identities",
    "locally_verified_current_v4_certificate",
    "supervisor_healthy",
    "activation_gap",
    "passed",
    "authority",
)


@dataclass(slots=True)
class ActivationClaimAssessment:
    """One activation claim with probe evidence and operator action."""

    field: str
    observed: bool
    proven: bool
    detail: str
    operator_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "observed": self.observed,
            "proven": self.proven,
            "detail": self.detail,
            "operator_action": self.operator_action,
        }


@dataclass(slots=True)
class CloseoutActivationProbeReport:
    """Structured activation-gap handoff for operators and the gate materializer."""

    schema: str = ACTIVATION_PROBE_SCHEMA
    interface: str = ACTIVATION_PROBE_INTERFACE
    authority: bool = False
    activation_gap_present: bool = True
    live_report: dict[str, Any] = field(default_factory=dict)
    claims: tuple[ActivationClaimAssessment, ...] = ()
    repair_evidence: dict[str, Any] = field(default_factory=dict)
    remaining_operator_actions: tuple[str, ...] = ()
    notes: tuple[str, ...] = (
        "This probe never authorizes production warm-skip.",
        "activation_gap remains true until reviewed v4 keys/manifest are ready.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "authority": self.authority,
            "activation_gap_present": self.activation_gap_present,
            "live_report": dict(self.live_report),
            "claims": [item.to_dict() for item in self.claims],
            "repair_evidence_summary": {
                "passed": self.repair_evidence.get("passed"),
                "authority": self.repair_evidence.get("authority"),
                "activation_gap": self.repair_evidence.get("activation_gap"),
                "activation_gap_present": self.repair_evidence.get(
                    "activation_gap_present"
                ),
                "repair_id": self.repair_evidence.get("repair_id"),
                "producer_task_id": self.repair_evidence.get("producer_task_id"),
            },
            "remaining_operator_actions": list(self.remaining_operator_actions),
            "notes": list(self.notes),
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _closeout_composition_cache_root() -> Path:
    """Durable cache for closeout composition probes (candidate + cert stores).

    Keeps probe material out of the source tree while allowing ordinary default
    composition to materialize typed stores under a known state root.
    """

    return (
        Path.home()
        / ".local"
        / "state"
        / "ipfs_accelerate_py"
        / "proof-backed-test-reuse-v1"
        / "runtime"
        / "closeout-composition-cache"
    )


def _probe_live_activation_report(
    *,
    repo_root: Path | str | None,
) -> dict[str, Any]:
    """Compose ordinary defaults with mode + cache_root, then inventory handles.

    Prior path called ``proof_reuse_runtime_activation_report(root_path=...)``
    without ``mode`` or ``cache_root``. That left identity / candidate store /
    revalidator / current-context unconfigured even though
    ``compose_default_proof_reuse_services`` can construct them when both are
    supplied. This probe uses SHADOW (observe, no warm-skip) so composition
    readiness is measured without inventing production skip authority.
    """

    try:
        from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
        from ipfs_accelerate_py.testing.proof_reuse.reporting import (
            proof_reuse_runtime_activation_report,
        )
    except Exception as exc:
        return {
            "source": "import_failed",
            "reason_code": f"{type(exc).__name__}",
            "activation_gap_present": True,
            "test_certificate_authority_ready": False,
            "native_groth16_ready": False,
            "ordinary_warm_skip_path_complete": False,
            "ordinary_default_composition_usable": False,
            "activation_blocker_codes": ["activation_probe_import_failed"],
        }

    roots: list[Path] = []
    if repo_root is not None:
        root = Path(repo_root)
        roots.append(root)
        accel = root / "external" / "ipfs_accelerate"
        if accel.is_dir():
            roots.append(accel)
    try:
        # Prefer accelerate package root when available.
        import ipfs_accelerate_py

        pkg = Path(ipfs_accelerate_py.__file__).resolve().parent.parent
        roots.append(pkg)
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen_roots: set[str] = set()
    ordered_roots: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        ordered_roots.append(root)

    cache_root = _closeout_composition_cache_root()
    try:
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except Exception:
        pass

    last: dict[str, Any] = {
        "activation_gap_present": True,
        "ordinary_default_composition_usable": False,
    }
    best: dict[str, Any] | None = None
    best_score = -1
    for root in ordered_roots or [None]:  # type: ignore[list-item]
        try:
            report = proof_reuse_runtime_activation_report(
                mode=ProofReuseMode.SHADOW,
                root_path=root,
                cache_root=cache_root,
                compose_if_missing=True,
            )
            last = report.to_dict()
            last["composition_probe"] = {
                "mode": "shadow",
                "cache_root": str(cache_root)[:256],
                "root_path": str(root)[:256] if root is not None else "",
                "identity_required": True,
                "stores_required": True,
            }
            # Score: prefer usable composition, then fewer blockers.
            usable = 1 if last.get("ordinary_default_composition_usable") else 0
            blockers = last.get("activation_blocker_codes") or ()
            score = usable * 100 - len(tuple(blockers))
            if score > best_score:
                best_score = score
                best = last
            if usable:
                return last
        except Exception as exc:
            last = {
                "source": "probe_failed",
                "reason_code": f"{type(exc).__name__}:{exc}"[:96],
                "activation_gap_present": True,
                "ordinary_default_composition_usable": False,
            }
    return best if best is not None else last


def _probe_module(path: str) -> tuple[bool, str]:
    try:
        __import__(path)
        return True, "import_ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"[:120]


def _probe_fixture_discover() -> tuple[bool, str]:
    try:
        from pathlib import Path as _Path
        import importlib.util
        import sys

        candidates = [
            _Path(__file__).resolve().parents[3]
            / "test"
            / "api"
            / "proof_reuse_real_groth16_fixture.py",
            _Path(__file__).resolve().parents[4]
            / "external"
            / "ipfs_accelerate"
            / "test"
            / "api"
            / "proof_reuse_real_groth16_fixture.py",
        ]
        fixture_path = next((p for p in candidates if p.is_file()), None)
        if fixture_path is None:
            return False, "fixture_module_missing"
        name = "proof_reuse_real_groth16_fixture_activation_probe"
        spec = importlib.util.spec_from_file_location(name, fixture_path)
        if spec is None or spec.loader is None:
            return False, "fixture_spec_missing"
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        discover = getattr(module, "RealGroth16TestPassFixture", None)
        if discover is None:
            return False, "RealGroth16TestPassFixture_missing"
        try:
            fixture = discover.discover()
        except Exception as exc:
            return False, f"discover_failed:{type(exc).__name__}:{exc}"[:120]
        if fixture is None:
            return False, "fixture_not_discovered"
        return True, f"fixture_discovered:{getattr(fixture, 'name', 'ok')}"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"[:120]


def assess_activation_claims(
    *,
    identity: CloseoutMaterializerIdentity,
    live_report: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    supervisor_healthy: bool = True,
) -> tuple[ActivationClaimAssessment, ...]:
    """Map live probes to activation claim assessments (proven vs not)."""

    gap_present = bool(live_report.get("activation_gap_present"))
    cert_ready = bool(live_report.get("test_certificate_authority_ready"))
    native_ready = bool(live_report.get("native_groth16_ready"))
    warm_complete = bool(live_report.get("ordinary_warm_skip_path_complete"))
    composition_usable = bool(live_report.get("ordinary_default_composition_usable"))
    gap = live_report.get("activation_gap") if isinstance(live_report.get("activation_gap"), Mapping) else {}
    blockers = tuple(live_report.get("activation_blocker_codes") or ())

    # False-skip free MODE=off receipts retained for this tree.
    zero_false = True
    receipt_count = 0
    for row in validation_receipts:
        if not isinstance(row, Mapping):
            continue
        if row.get("passed") is not True:
            continue
        if _text(row.get("git_commit_id")) not in {"", identity.git_commit_id}:
            continue
        receipt_count += 1
        skipped = row.get("skipped_count", 0)
        if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped != 0:
            zero_false = False
            break
    if receipt_count == 0:
        zero_false = False

    controller_ok, controller_detail = _probe_module(
        "ipfs_accelerate_py.testing.proof_reuse.candidate_publication"
    )
    issuance_ok, issuance_detail = _probe_module(
        "ipfs_accelerate_py.testing.proof_reuse.receipt"
    )
    fixture_ok, fixture_detail = _probe_fixture_discover()

    # Proven only when live surfaces actually authorize the claim.
    claims: list[ActivationClaimAssessment] = [
        ActivationClaimAssessment(
            field="zero_false_skip_assurance",
            observed=receipt_count > 0,
            proven=zero_false and receipt_count > 0,
            detail=(
                f"mode_off_passed_receipts={receipt_count}; zero_false_skips={zero_false}"
            ),
            operator_action=""
            if zero_false and receipt_count > 0
            else "retain MODE=off validation receipts with skipped_count=0",
        ),
        ActivationClaimAssessment(
            field="activation_e2e_passed",
            observed=composition_usable or warm_complete,
            proven=False,  # never claim e2e without full activation path
            detail=(
                f"composition_usable={composition_usable}; "
                f"warm_complete={warm_complete}; gap={gap_present}"
            ),
            operator_action=(
                "run production runtime activation e2e with reviewed keys "
                f"(blockers={','.join(blockers[:6]) or 'none'})"
            ),
        ),
        ActivationClaimAssessment(
            field="zero_injection_default_path",
            observed=composition_usable,
            proven=bool(composition_usable and not gap_present),
            detail=(
                f"ordinary_default_composition_usable={composition_usable}; "
                f"gap={gap_present}; "
                f"blockers={','.join(blockers[:8]) or 'none'}"
            ),
            operator_action=(
                ""
                if composition_usable and not gap_present
                else (
                    # Composition may already be fully wired; remaining work is
                    # reviewed authority / gap closeout, not identity wiring.
                    "close activation gap so ordinary default composition can authorize warm-skip"
                    if composition_usable
                    else "wire production identity providers into ordinary default composition"
                )
            ),
        ),
        ActivationClaimAssessment(
            field="three_repository_cold_warm",
            observed=fixture_ok,
            proven=False,
            detail=fixture_detail,
            operator_action=(
                "measure three-repository cold/warm path with RealGroth16 fixture"
            ),
        ),
        ActivationClaimAssessment(
            field="real_groth16_certificate",
            observed=native_ready or cert_ready,
            proven=bool(cert_ready and native_ready and not gap_present),
            detail=(
                f"native_ready={native_ready}; cert_ready={cert_ready}; "
                f"gap_reason={_text(gap.get('reason_code'))}"
            ),
            operator_action=(
                "install reviewed v4 keys/manifest and native Groth16 readiness "
                f"(reason={_text(gap.get('reason_code') or live_report.get('reason_code'))})"
            ),
        ),
        ActivationClaimAssessment(
            field="measured_subprocess_benchmark",
            observed=fixture_ok,
            proven=False,
            detail=f"fixture={fixture_detail}; api=run_subprocess_proof_reuse_benchmark",
            operator_action=(
                "run run_subprocess_proof_reuse_benchmark with reviewed fixture pins"
            ),
        ),
        ActivationClaimAssessment(
            field="historical_activation_claims_superseded",
            observed=True,
            proven=True,
            detail="PTR-149 production activation supersedes historical pseudo claims",
            operator_action="",
        ),
        ActivationClaimAssessment(
            field="controller_owned_receipt_candidate_context",
            observed=controller_ok,
            proven=False,
            detail=controller_detail,
            operator_action=(
                "publish controller-owned v2 candidate context for the current tree"
            ),
        ),
        ActivationClaimAssessment(
            field="retained_proof_bearing_issuance_material",
            observed=issuance_ok,
            proven=False,
            detail=issuance_detail,
            operator_action=(
                "retain proof-bearing issuance material across lazy real issuer path"
            ),
        ),
        ActivationClaimAssessment(
            field="exact_reviewed_source_binary_capability_circuit_key_identities",
            observed=cert_ready,
            proven=bool(cert_ready and not gap_present),
            detail=(
                f"cert_ready={cert_ready}; "
                f"artifact_bindings="
                f"{(live_report.get('test_certificate_authority') or {}).get('artifact_bindings')}"
            ),
            operator_action=(
                "pin exact reviewed source/binary/capability/circuit/key identities"
            ),
        ),
        ActivationClaimAssessment(
            field="locally_verified_current_v4_certificate",
            observed=cert_ready,
            proven=bool(cert_ready and not gap_present),
            detail=f"test_certificate_authority_ready={cert_ready}",
            operator_action="produce and locally verify a current v4 certificate",
        ),
        ActivationClaimAssessment(
            field="supervisor_healthy",
            observed=True,
            proven=bool(supervisor_healthy),
            detail=f"supervisor_healthy={supervisor_healthy}",
            operator_action="" if supervisor_healthy else "restore supervisor health",
        ),
        ActivationClaimAssessment(
            field="activation_gap",
            observed=True,
            proven=gap_present,  # "proven" means the gap is confirmed present
            detail=(
                f"present={gap_present}; reason={_text(gap.get('reason_code'))}; "
                f"closeout_authorized={gap.get('closeout_authorized')}"
            ),
            operator_action=(
                "close activation gap with reviewed v4 keys/manifest allowlist"
                if gap_present
                else ""
            ),
        ),
    ]
    return tuple(claims)


def produce_closeout_activation_probe(
    identity: CloseoutMaterializerIdentity,
    *,
    objective_completion_tree_id: str,
    repo_root: Path | str | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    supervisor_healthy: bool = True,
    now_ms: int | None = None,
    freshness_seconds: float = 3_600.0,
    evidence_cid: str = "repair:production-runtime-activation-probe",
    attempt_heavy_measurements: bool | None = None,
) -> CloseoutActivationProbeReport:
    """Probe live activation and build honest repair evidence for PTR-122."""

    import os

    observed = int(now_ms if now_ms is not None else time.time() * 1000)
    fresh_until = observed + int(float(freshness_seconds) * 1000)
    live = _probe_live_activation_report(repo_root=repo_root)

    # Fixture discovery + optional heavy measurements (fail-closed skips).
    # Enable heavy runners when explicitly requested, or after local setup opt-in.
    if attempt_heavy_measurements is None:
        attempt_heavy_measurements = str(
            os.environ.get("PTR_CLOSEOUT_HEAVY_MEASUREMENTS", "")
        ).strip().lower() in {"1", "true", "yes", "on", "auto"}
        if not attempt_heavy_measurements and str(
            os.environ.get("PTR_CLOSEOUT_LOCAL_SETUP", "")
        ).strip().lower() in {"1", "true", "yes", "on"}:
            # Local setup implies we should try measurements once keys exist.
            attempt_heavy_measurements = True
    measurement_report: dict[str, Any] = {}
    measurement_claims: dict[str, bool] = {}
    try:
        from .proof_test_reuse_closeout_activation_measurements import (
            run_closeout_activation_measurements,
        )

        measurements = run_closeout_activation_measurements(
            attempt_heavy_measurements=bool(attempt_heavy_measurements),
            require_available_fixture=True,
        )
        measurement_report = measurements.to_dict()
        measurement_claims = dict(measurements.claims_supported)
    except Exception as exc:
        measurement_report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    claims = assess_activation_claims(
        identity=identity,
        live_report=live,
        validation_receipts=validation_receipts,
        supervisor_healthy=supervisor_healthy,
    )
    # Overlay measurement-supported claims without inventing production authority.
    # Local operational v4 keys can prove measured cold/warm paths, but never
    # clear the activation gap or production certificate claims while the live
    # report still says reviewed authority is absent.
    gap_still = bool(live.get("activation_gap_present"))
    # Claims that still require reviewed production ceremony even when local
    # operational measurements succeed. zero_injection_default_path is driven by
    # the live composition probe (mode+cache_root), not fixture measurements —
    # keep its assess_activation_claims result so usable composition is observed.
    production_sensitive = {
        "exact_reviewed_source_binary_capability_circuit_key_identities",
        "activation_e2e_passed",
    }
    # Local operational measurements (may use non-ceremony keys).
    measurement_only = {
        "measured_subprocess_benchmark",
        "three_repository_cold_warm",
        "locally_verified_current_v4_certificate",
        "retained_proof_bearing_issuance_material",
        "real_groth16_certificate",
    }
    if measurement_claims:
        updated: list[ActivationClaimAssessment] = []
        fixture = (
            (measurement_report.get("fixture") or {})
            if isinstance(measurement_report, Mapping)
            else {}
        )
        composition_usable_meas = bool(
            measurement_claims.get("ordinary_default_composition_usable")
            or live.get("ordinary_default_composition_usable")
        )
        for claim in claims:
            if claim.field in measurement_only:
                proven = bool(measurement_claims.get(claim.field))
                updated.append(
                    ActivationClaimAssessment(
                        field=claim.field,
                        observed=True,
                        proven=proven,
                        detail=(
                            f"fixture_available={fixture.get('available')}; "
                            f"reason={fixture.get('reason')}; "
                            f"measurement_supported={measurement_claims.get(claim.field)}; "
                            f"local_operational_keys_not_production={gap_still}"
                        ),
                        operator_action=""
                        if proven
                        else claim.operator_action,
                    )
                )
            elif claim.field in production_sensitive:
                # Measured local keys do not satisfy production e2e / reviewed
                # identity pins while activation_gap remains present.
                proven = bool(measurement_claims.get(claim.field)) and not gap_still
                updated.append(
                    ActivationClaimAssessment(
                        field=claim.field,
                        observed=bool(
                            fixture.get("binary_path")
                            or fixture.get("available")
                            or composition_usable_meas
                        ),
                        proven=proven,
                        detail=(
                            f"fixture_available={fixture.get('available')}; "
                            f"reason={fixture.get('reason')}; "
                            f"binary_present={measurement_claims.get('fixture_binary_present')}; "
                            f"keys_present={measurement_claims.get('fixture_keys_present')}; "
                            f"composition_usable={composition_usable_meas}; "
                            f"activation_gap={gap_still}"
                        ),
                        operator_action=claim.operator_action if not proven else "",
                    )
                )
            elif claim.field == "zero_injection_default_path":
                # Prefer live composition; measurements may reinforce observed.
                observed = bool(claim.observed or composition_usable_meas)
                proven = bool(observed and not gap_still)
                updated.append(
                    ActivationClaimAssessment(
                        field=claim.field,
                        observed=observed,
                        proven=proven,
                        detail=(
                            f"{claim.detail}; "
                            f"measurement_composition_usable="
                            f"{measurement_claims.get('ordinary_default_composition_usable')}"
                        ),
                        operator_action="" if proven else claim.operator_action,
                    )
                )
            else:
                updated.append(claim)
        claims = tuple(updated)
    by_field = {item.field: item for item in claims}
    gap_present = bool(live.get("activation_gap_present")) or bool(
        by_field.get("activation_gap") and by_field["activation_gap"].proven
    )

    def _proven(name: str) -> bool:
        item = by_field.get(name)
        return bool(item and item.proven)

    repair = build_production_runtime_activation_evidence(
        repository_id=identity.repository_id,
        tree_id=identity.git_tree_id,
        commit_id=identity.git_commit_id,
        gitlink_state_cid=identity.gitlink_state_cid,
        repository_forest_cid=identity.repository_forest_cid,
        capability_cid=identity.capability_cid,
        verifying_key_cid=identity.verifying_key_cid,
        circuit_cid=identity.circuit_cid,
        policy_cid=identity.policy_cid,
        objective_completion_tree_id=objective_completion_tree_id,
        observed_at_ms=observed - 1_000,
        fresh_until_ms=fresh_until,
        evidence_cid=evidence_cid,
        false_skips=0 if _proven("zero_false_skip_assurance") else 1,
        zero_false_skip_assurance=_proven("zero_false_skip_assurance"),
        activation_e2e_passed=_proven("activation_e2e_passed"),
        zero_injection_default_path=_proven("zero_injection_default_path"),
        three_repository_cold_warm=_proven("three_repository_cold_warm"),
        real_groth16_certificate=_proven("real_groth16_certificate"),
        measured_subprocess_benchmark=_proven("measured_subprocess_benchmark"),
        historical_activation_claims_superseded=_proven(
            "historical_activation_claims_superseded"
        ),
        controller_owned_receipt_candidate_context=_proven(
            "controller_owned_receipt_candidate_context"
        ),
        retained_proof_bearing_issuance_material=_proven(
            "retained_proof_bearing_issuance_material"
        ),
        exact_reviewed_source_binary_capability_circuit_key_identities=_proven(
            "exact_reviewed_source_binary_capability_circuit_key_identities"
        ),
        locally_verified_current_v4_certificate=_proven(
            "locally_verified_current_v4_certificate"
        ),
        supervisor_healthy=_proven("supervisor_healthy"),
        activation_gap=gap_present,
        activation_gap_present=gap_present,
        # Structural markers: not injected/pseudo when we are reporting truthfully.
        injected=False,
        pseudo_certificate=False,
        synthetic_timing=False,
        service_injection=False,
        structural_only_verification=False,
        # force builder to recompute passed from claims
        passed=False,
    )
    # Gate requires authority=authoritative for non-gap path; while gap is
    # present we intentionally keep authority non-authoritative so skip is
    # never granted. When gap clears and all claims prove, builder already
    # sets authoritative + passed.
    if gap_present:
        repair["authority"] = "none"
        repair["passed"] = False
        repair["activation_gap"] = True
        repair["activation_gap_present"] = True
    else:
        # Only promote authority when the live report says no gap and claims prove.
        all_positive = all(
            _proven(name)
            for name in (
                "zero_false_skip_assurance",
                "activation_e2e_passed",
                "zero_injection_default_path",
                "three_repository_cold_warm",
                "real_groth16_certificate",
                "measured_subprocess_benchmark",
                "controller_owned_receipt_candidate_context",
                "retained_proof_bearing_issuance_material",
                "exact_reviewed_source_binary_capability_circuit_key_identities",
                "locally_verified_current_v4_certificate",
                "supervisor_healthy",
            )
        )
        if all_positive:
            repair["authority"] = "authoritative"
            repair["passed"] = True
            repair["activation_gap"] = False
            repair["activation_gap_present"] = False
        else:
            repair["authority"] = "none"
            repair["passed"] = False

    repair["repair_id"] = PRODUCTION_RUNTIME_ACTIVATION_ID
    repair["requirement_id"] = PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT
    repair["probe_schema"] = ACTIVATION_PROBE_SCHEMA
    repair["live_activation_blocker_codes"] = list(
        live.get("activation_blocker_codes") or ()
    )[:32]
    repair["live_activation_gap_reason"] = _text(
        (live.get("activation_gap") or {}).get("reason_code")
        if isinstance(live.get("activation_gap"), Mapping)
        else live.get("reason_code")
    )

    remaining = tuple(
        item.operator_action
        for item in claims
        if item.operator_action and not item.proven
    )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for action in remaining:
        if action not in seen:
            seen.add(action)
            ordered.append(action)

    live_out = dict(live)
    live_out["measurements"] = measurement_report
    return CloseoutActivationProbeReport(
        activation_gap_present=gap_present,
        live_report=live_out,
        claims=claims,
        repair_evidence=repair,
        remaining_operator_actions=tuple(ordered),
    )


__all__ = [
    "ACTIVATION_CLAIM_FIELDS",
    "ACTIVATION_PROBE_INTERFACE",
    "ACTIVATION_PROBE_SCHEMA",
    "ActivationClaimAssessment",
    "CloseoutActivationProbeReport",
    "assess_activation_claims",
    "produce_closeout_activation_probe",
]
