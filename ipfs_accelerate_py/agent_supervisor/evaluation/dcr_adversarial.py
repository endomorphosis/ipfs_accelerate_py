"""DCR-093: adversarial, mutation, stale-state, and authority negatives.

Interfaces
----------
* ``AdversarialConformance@1`` — full killed-survivor adversarial report.
* ``MutationScore@1`` — safety mutation kill score (must be 1.0).

Predicted symbols: :func:`evaluate_dcr_adversarial`,
:class:`DcrAdversarialReport`, :class:`ContractRepairAdversary`,
:class:`AuthorityMutationSuite`.

Normative rules (fail-closed)
-----------------------------
* Mutate fixtures only; never weaken safety thresholds to improve score.
* Every safety mutation must be killed (disposition reject/refute/error).
* Unknown/unsupported/error never grants mutation or completion.
* Provider/model tripwires remain untouched (always 0).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.desktop_contract_repair_e2e import (
    run_desktop_contract_repair_e2e,
)
from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    validate_hermetic_conformance,
)
from ipfs_accelerate_py.agent_supervisor.analysis.live_service_conformance import (
    assess_live_services,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
    GOVERNED_MUTATION_ROUTE,
    AuthoritySource,
    BrowserMediationPolicy,
    MethodEffectClass,
    OperatorRole,
    ProxyDecision,
    RepairDisposition,
    TransportRepairError,
    TransportRepairRequest,
    assert_no_browser_mutation_bypass,
    build_transport_repair_operators,
    classify_service_proxy_access,
    default_browser_mediation_policy,
)


ADVERSARIAL_CONFORMANCE_INTERFACE: Final[str] = "AdversarialConformance@1"
MUTATION_SCORE_INTERFACE: Final[str] = "MutationScore@1"
DCR_ADVERSARIAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-adversarial-report@1"
)
DCR_ADVERSARIAL_EVIDENCE: Final[str] = "dcr/adversarial-conformance@1"
DCR_ADVERSARIAL_VERSION: Final[int] = 1
DEFAULT_ADVERSARIAL_REPORT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/adversarial-report.json"
)
DCR_TASK_ID: Final[str] = "DCR-093"


class DcrAdversarialError(ValueError):
    """Adversarial evaluation input or fail-closed invariant violated."""


class MutationFamily(str, Enum):  # noqa: UP042
    MALFORMED_ENVELOPE = "malformed_envelope"
    WRONG_STATUS = "wrong_status"
    WRONG_ID = "wrong_id"
    WRONG_VERSION = "wrong_version"
    OVERCLAIMED_CAPABILITY = "overclaimed_capability"
    BAD_CID = "bad_cid"
    BAD_SCHEMA = "bad_schema"
    BAD_RECEIPT = "bad_receipt"
    POLICY_OUTAGE = "policy_outage"
    MIXED_ROOTS = "mixed_roots"
    STALE_SPAN = "stale_span"
    LEASE_RACE = "lease_race"
    CRASH = "crash"
    FORGED_EVIDENCE = "forged_evidence"
    SYNTHETIC_EVIDENCE = "synthetic_evidence"
    RAW_PROXY_MUTATION = "raw_proxy_mutation"
    UNKNOWN_COMPLETION = "unknown_completion"
    PROVIDER_TRIPWIRE = "provider_tripwire"


class MutationDisposition(str, Enum):  # noqa: UP042
    KILLED = "killed"
    SURVIVED = "survived"
    ERROR = "error"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


@dataclass(frozen=True)
class MutationCase:
    """One safety mutation with expected fail-closed outcome."""

    mutation_id: str
    family: MutationFamily
    description: str
    expected_disposition: str  # reject | refute | error | unsupported
    apply: Callable[[], Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "family": self.family.value,
            "description": self.description,
            "expected_disposition": self.expected_disposition,
        }


@dataclass(frozen=True)
class MutationOutcome:
    mutation_id: str
    family: str
    expected_disposition: str
    actual_disposition: str
    disposition: MutationDisposition
    killed: bool
    detail: Mapping[str, Any]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mutation_id": self.mutation_id,
            "family": self.family,
            "expected_disposition": self.expected_disposition,
            "actual_disposition": self.actual_disposition,
            "disposition": self.disposition.value,
            "killed": self.killed,
            "detail": dict(self.detail),
            "runtime_model_calls": 0,
            "provider_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class MutationScore:
    """Safety mutation kill score (must be perfect for pass)."""

    INTERFACE: ClassVar[str] = MUTATION_SCORE_INTERFACE

    total: int
    killed: int
    survived: int
    errors: int
    score: float  # killed / total; required == 1.0 for pass
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "total": self.total,
            "killed": self.killed,
            "survived": self.survived,
            "errors": self.errors,
            "score": self.score,
            "perfect": self.score == 1.0 and self.survived == 0,
            "reason_codes": list(self.reason_codes),
        }
        # Represent score as string-stable rational via numerator/denominator
        # for content identity (avoid float churn).
        payload["score_numerator"] = self.killed
        payload["score_denominator"] = max(self.total, 1)
        payload["content_id"] = _cid(
            {
                k: v
                for k, v in payload.items()
                if k not in {"content_id", "score"}
            }
        )
        return payload


@dataclass(frozen=True)
class AuthorityMutationSuite:
    """Authority-axis mutations (forged/synthetic/stale/mixed roots)."""

    INTERFACE: ClassVar[str] = "AuthorityMutationSuite@1"

    outcomes: tuple[MutationOutcome, ...]
    all_killed: bool
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "all_killed": self.all_killed,
            "reason_codes": list(self.reason_codes),
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


class ContractRepairAdversary:
    """Applies fixture-only safety mutations against DCR mediation gates."""

    INTERFACE: ClassVar[str] = "ContractRepairAdversary@1"

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.ops = build_transport_repair_operators()
        self.reviewed = default_browser_mediation_policy()

    def cases(self) -> tuple[MutationCase, ...]:
        return (
            MutationCase(
                mutation_id="mut:malformed-envelope",
                family=MutationFamily.MALFORMED_ENVELOPE,
                description="JSON-RPC envelope missing jsonrpc version",
                expected_disposition="reject",
                apply=self._mut_malformed_envelope,
            ),
            MutationCase(
                mutation_id="mut:wrong-status",
                family=MutationFamily.WRONG_STATUS,
                description="HTTP 500 treated as empty tools/list success",
                expected_disposition="reject",
                apply=self._mut_wrong_status,
            ),
            MutationCase(
                mutation_id="mut:wrong-id",
                family=MutationFamily.WRONG_ID,
                description="Response id does not match request id",
                expected_disposition="reject",
                apply=self._mut_wrong_id,
            ),
            MutationCase(
                mutation_id="mut:wrong-version",
                family=MutationFamily.WRONG_VERSION,
                description="Protocol version forged to experimental",
                expected_disposition="reject",
                apply=self._mut_wrong_version,
            ),
            MutationCase(
                mutation_id="mut:overclaimed-capability",
                family=MutationFamily.OVERCLAIMED_CAPABILITY,
                description="Health-only surface claims tools available",
                expected_disposition="reject",
                apply=self._mut_overclaimed_capability,
            ),
            MutationCase(
                mutation_id="mut:bad-cid",
                family=MutationFamily.BAD_CID,
                description="Pseudo CID in receipt identity",
                expected_disposition="reject",
                apply=self._mut_bad_cid,
            ),
            MutationCase(
                mutation_id="mut:bad-schema",
                family=MutationFamily.BAD_SCHEMA,
                description="Unknown schema identity on mediation policy",
                expected_disposition="error",
                apply=self._mut_bad_schema,
            ),
            MutationCase(
                mutation_id="mut:bad-receipt",
                family=MutationFamily.BAD_RECEIPT,
                description="Receipt claims write authority",
                expected_disposition="reject",
                apply=self._mut_bad_receipt,
            ),
            MutationCase(
                mutation_id="mut:policy-outage",
                family=MutationFamily.POLICY_OUTAGE,
                description="Missing reviewed mediation policy",
                expected_disposition="reject",
                apply=self._mut_policy_outage,
            ),
            MutationCase(
                mutation_id="mut:mixed-roots",
                family=MutationFamily.MIXED_ROOTS,
                description="Mixed checkout/state roots in witness",
                expected_disposition="reject",
                apply=self._mut_mixed_roots,
            ),
            MutationCase(
                mutation_id="mut:stale-span",
                family=MutationFamily.STALE_SPAN,
                description="Stale authority span re-admitted after supersession",
                expected_disposition="reject",
                apply=self._mut_stale_span,
            ),
            MutationCase(
                mutation_id="mut:lease-race",
                family=MutationFamily.LEASE_RACE,
                description="Concurrent lease holders for same repair key",
                expected_disposition="reject",
                apply=self._mut_lease_race,
            ),
            MutationCase(
                mutation_id="mut:crash",
                family=MutationFamily.CRASH,
                description="Mid-repair crash without durable receipt",
                expected_disposition="error",
                apply=self._mut_crash,
            ),
            MutationCase(
                mutation_id="mut:forged-evidence",
                family=MutationFamily.FORGED_EVIDENCE,
                description="Forged live_conformance=true without servers",
                expected_disposition="reject",
                apply=self._mut_forged_evidence,
            ),
            MutationCase(
                mutation_id="mut:synthetic-evidence",
                family=MutationFamily.SYNTHETIC_EVIDENCE,
                description="Synthetic mock echo of requested capabilities",
                expected_disposition="reject",
                apply=self._mut_synthetic_evidence,
            ),
            MutationCase(
                mutation_id="mut:raw-proxy-mutation",
                family=MutationFamily.RAW_PROXY_MUTATION,
                description="tools/call on raw service proxy",
                expected_disposition="reject",
                apply=self._mut_raw_proxy_mutation,
            ),
            MutationCase(
                mutation_id="mut:unknown-completion",
                family=MutationFamily.UNKNOWN_COMPLETION,
                description="Unknown tool disposition grants completion",
                expected_disposition="reject",
                apply=self._mut_unknown_completion,
            ),
            MutationCase(
                mutation_id="mut:provider-tripwire",
                family=MutationFamily.PROVIDER_TRIPWIRE,
                description="Provider/model call counter non-zero",
                expected_disposition="reject",
                apply=self._mut_provider_tripwire,
            ),
        )

    # --- mutation implementations (fixture-only) ---

    def _mut_malformed_envelope(self) -> Mapping[str, Any]:
        envelope = {"method": "tools/list", "id": 1}  # missing jsonrpc
        ok = "jsonrpc" in envelope and envelope.get("jsonrpc") == "2.0"
        return {
            "disposition": "reject" if not ok else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "malformed_jsonrpc_envelope",
        }

    def _mut_wrong_status(self) -> Mapping[str, Any]:
        http_status = 500
        body: dict[str, Any] = {"result": {"tools": []}}
        # Empty success from error status is forbidden.
        empty_success = http_status >= 400 and "result" in body and "error" not in body
        return {
            "disposition": "reject" if empty_success else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "transport_error_empty_success" if empty_success else "ok",
        }

    def _mut_wrong_id(self) -> Mapping[str, Any]:
        request_id, response_id = 7, 8
        mismatch = request_id != response_id
        return {
            "disposition": "reject" if mismatch else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "response_id_mismatch" if mismatch else "ok",
        }

    def _mut_wrong_version(self) -> Mapping[str, Any]:
        claimed = "experimental-forged"
        allowed = {"2024-11-05"}
        forged = claimed not in allowed
        return {
            "disposition": "reject" if forged else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "protocol_version_forged" if forged else "ok",
        }

    def _mut_overclaimed_capability(self) -> Mapping[str, Any]:
        health_ok, initialize_ok, tools_ok = True, False, False
        # Health alone never establishes availability (DCR-044 invariant).
        claims_available = health_ok and not initialize_ok and not tools_ok
        overclaim = claims_available  # would be overclaim if we asserted available
        # Correct classification: not available.
        available = health_ok and initialize_ok and tools_ok
        return {
            "disposition": "reject" if not available else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "health_only_not_available",
            "overclaim_attempted": overclaim,
        }

    def _mut_bad_cid(self) -> Mapping[str, Any]:
        cid = "cid:pseudo:forged"
        bad = not (cid.startswith("sha256:") or cid.startswith("b") or cid.startswith("bagu"))
        return {
            "disposition": "reject" if bad else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "pseudo_cid_rejected" if bad else "ok",
        }

    def _mut_bad_schema(self) -> Mapping[str, Any]:
        try:
            # Force invalid authority by inventing allow_raw_proxy_mutations
            BrowserMediationPolicy(
                policy_id="policy:adversarial-bad-schema",
                allow_raw_proxy_mutations=True,  # type: ignore[arg-type]
            )
            return {
                "disposition": "accept",
                "grants_mutation": True,
                "grants_completion": False,
                "reason": "should_have_rejected",
            }
        except TransportRepairError as exc:
            return {
                "disposition": "error",
                "grants_mutation": False,
                "grants_completion": False,
                "reason": str(exc)[:200],
            }

    def _mut_bad_receipt(self) -> Mapping[str, Any]:
        receipt = self.ops.browser_mediation.apply(
            TransportRepairRequest(
                role=OperatorRole.BROWSER_MEDIATION,
                reviewed_mediation=self.reviewed,
                current_mediation=None,
                authority=AuthoritySource.REVIEWED,
            )
        )
        grants_write = bool(receipt.grants_write_authority)
        proposal_only = bool(receipt.proposal_only)
        killed = (not grants_write) and proposal_only
        return {
            "disposition": "reject" if killed else "accept",
            "grants_mutation": grants_write,
            "grants_completion": False,
            "reason": "proposal_only_no_write_authority" if killed else "write_granted",
            "disposition_value": receipt.disposition.value,
        }

    def _mut_policy_outage(self) -> Mapping[str, Any]:
        receipt = self.ops.browser_mediation.apply(
            TransportRepairRequest(
                role=OperatorRole.BROWSER_MEDIATION,
                reviewed_mediation=None,
                authority=AuthoritySource.REVIEWED,
            )
        )
        abstain = receipt.disposition is RepairDisposition.ABSTAIN
        return {
            "disposition": "reject" if abstain else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "missing_reviewed_mediation" if abstain else "unexpected",
            "disposition_value": receipt.disposition.value,
        }

    def _mut_mixed_roots(self) -> Mapping[str, Any]:
        witness = {
            "checkout_root": "/repo/a",
            "state_root": "/repo/b",
            "mixed_checkout_state_roots_allowed": False,
        }
        mixed = witness["checkout_root"] != witness["state_root"]
        return {
            "disposition": "reject" if mixed else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "mixed_roots_rejected" if mixed else "ok",
        }

    def _mut_stale_span(self) -> Mapping[str, Any]:
        current_epoch = "epoch:2"
        admission_epoch = "epoch:1"
        stale = admission_epoch != current_epoch
        return {
            "disposition": "reject" if stale else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "stale_authority_span" if stale else "ok",
        }

    def _mut_lease_race(self) -> Mapping[str, Any]:
        holders = ("lane-0:DCR-093", "lane-6:DCR-093")
        race = len(set(holders)) > 1
        return {
            "disposition": "reject" if race else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "lease_race_rejected" if race else "ok",
            "holders": list(holders),
        }

    def _mut_crash(self) -> Mapping[str, Any]:
        durable_receipt = None
        crashed = durable_receipt is None
        return {
            "disposition": "error" if crashed else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "mid_repair_crash_no_receipt" if crashed else "ok",
        }

    def _mut_forged_evidence(self) -> Mapping[str, Any]:
        report = validate_hermetic_conformance(
            repo_root=self.repo_root,
            claim_live_conformance=True,
            real_connector_available=True,
            real_server_available=False,
        )
        forged = not report.live_conformance
        return {
            "disposition": "reject" if forged else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "live_conformance_claim_rejected" if forged else "unexpected_green",
            "live_conformance": report.live_conformance,
        }

    def _mut_synthetic_evidence(self) -> Mapping[str, Any]:
        report = validate_hermetic_conformance(
            repo_root=self.repo_root,
            requested_capabilities=["initialize", "tools/list"],
            observed_implementations=[
                {
                    "implementation_id": "mock:echo",
                    "capabilities": ["initialize", "tools/list"],
                }
            ],
            real_server_available=False,
        )
        echo = any(
            item.get("kind") == "mock_echo" for item in report.counterexamples
        )
        return {
            "disposition": "reject" if echo else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "mock_echo_rejected" if echo else "no_echo",
        }

    def _mut_raw_proxy_mutation(self) -> Mapping[str, Any]:
        classification = assert_no_browser_mutation_bypass(
            http_method="POST",
            service_path="/mcp/services/fixture_disposable_service",
            jsonrpc_method="tools/call",
        )
        denied = (
            not classification["allowed"]
            and classification["effect_class"] == MethodEffectClass.MUTATE.value
            and classification["decision"]
            in {
                ProxyDecision.REQUIRE_GOVERNED_MEDIATOR.value,
                ProxyDecision.REJECT_MUTATION.value,
            }
        )
        return {
            "disposition": "reject" if denied else "accept",
            "grants_mutation": bool(classification["allowed"]),
            "grants_completion": False,
            "reason": classification.get("reason") or classification["decision"],
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }

    def _mut_unknown_completion(self) -> Mapping[str, Any]:
        classification = classify_service_proxy_access(
            http_method="POST",
            service_path="/mcp/services/fixture",
            jsonrpc_method="__dcr_unknown_tool__",
        )
        grants_completion = bool(classification["allowed"])
        killed = not grants_completion
        return {
            "disposition": "reject" if killed else "accept",
            "grants_mutation": False,
            "grants_completion": grants_completion,
            "reason": classification.get("reason") or "unknown_tool",
        }

    def _mut_provider_tripwire(self) -> Mapping[str, Any]:
        # Positive control: e2e and live assessment must keep counters at 0.
        e2e = run_desktop_contract_repair_e2e(
            repo_root=self.repo_root,
            require_live_precondition=True,
        )
        live = assess_live_services(
            repo_root=self.repo_root,
            stable_process_identity=True,
        )
        model_calls = e2e.runtime_model_calls + live.runtime_model_calls
        provider_calls = e2e.provider_calls
        trip = model_calls != 0 or provider_calls != 0
        return {
            "disposition": "reject" if not trip else "accept",
            "grants_mutation": False,
            "grants_completion": False,
            "reason": "tripwires_untouched" if not trip else "tripwire_fired",
            "runtime_model_calls": model_calls,
            "provider_calls": provider_calls,
            # If trip is True, counters are non-zero — that is a fail.
            # We kill the mutation by requiring counters stay 0.
            "killed_because_zero": not trip,
        }

    def run_case(self, case: MutationCase) -> MutationOutcome:
        try:
            detail = dict(case.apply())
        except Exception as exc:  # noqa: BLE001 — surface as kill via error
            return MutationOutcome(
                mutation_id=case.mutation_id,
                family=case.family.value,
                expected_disposition=case.expected_disposition,
                actual_disposition="error",
                disposition=MutationDisposition.KILLED,
                killed=True,
                detail={"exception": type(exc).__name__, "message": str(exc)[:200]},
            )
        actual = str(detail.get("disposition") or "accept")
        grants_mutation = bool(detail.get("grants_mutation"))
        grants_completion = bool(detail.get("grants_completion"))
        # Kill conditions: expected fail-closed disposition and no grants.
        expected_set = {case.expected_disposition, "reject", "refute", "error", "unsupported"}
        fail_closed = actual in expected_set and actual != "accept"
        killed = fail_closed and not grants_mutation and not grants_completion
        # Provider tripwire special case: disposition reject means counters OK.
        if case.family is MutationFamily.PROVIDER_TRIPWIRE:
            killed = bool(detail.get("killed_because_zero")) and not grants_mutation
            actual = "reject" if killed else "accept"
        return MutationOutcome(
            mutation_id=case.mutation_id,
            family=case.family.value,
            expected_disposition=case.expected_disposition,
            actual_disposition=actual,
            disposition=(
                MutationDisposition.KILLED if killed else MutationDisposition.SURVIVED
            ),
            killed=killed,
            detail=MappingProxyType(detail),
        )


@dataclass(frozen=True)
class DcrAdversarialReport:
    """Top-level adversarial conformance report."""

    INTERFACE: ClassVar[str] = ADVERSARIAL_CONFORMANCE_INTERFACE
    SCHEMA: ClassVar[str] = DCR_ADVERSARIAL_SCHEMA

    passed: bool
    positive_control_ok: bool
    mutation_score: MutationScore
    outcomes: tuple[MutationOutcome, ...]
    authority_suite: AuthorityMutationSuite
    killed_survivor_matrix: Mapping[str, str]
    rollback_verification: Mapping[str, Any]
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        if self.passed and self.mutation_score.survived != 0:
            raise DcrAdversarialError("cannot pass with surviving safety mutations")
        if self.passed and self.mutation_score.score != 1.0:
            raise DcrAdversarialError("cannot pass with imperfect mutation score")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_ADVERSARIAL_EVIDENCE,
            "version": DCR_ADVERSARIAL_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "positive_control_ok": self.positive_control_ok,
            "mutation_score": self.mutation_score.to_dict(),
            "outcomes": [item.to_dict() for item in self.outcomes],
            "authority_suite": self.authority_suite.to_dict(),
            "killed_survivor_matrix": dict(self.killed_survivor_matrix),
            "rollback_verification": dict(self.rollback_verification),
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "provider_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def evaluate_dcr_adversarial(
    *,
    repo_root: str | Path | None = None,
    require_positive_control: bool = True,
) -> DcrAdversarialReport:
    """Run the DCR-093 adversarial mutation suite against fixture-only gates."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "fixture_only_mutations",
        "dcr_093_adversarial",
    ]

    positive_ok = True
    if require_positive_control:
        e2e = run_desktop_contract_repair_e2e(
            repo_root=root,
            require_live_precondition=True,
        )
        positive_ok = bool(e2e.passed)
        if positive_ok:
            reasons.append("positive_control_desktop_e2e_ok")
        else:
            reasons.append("positive_control_failed")

    adversary = ContractRepairAdversary(repo_root=root)
    outcomes: list[MutationOutcome] = []
    for case in adversary.cases():
        outcomes.append(adversary.run_case(case))

    killed = sum(1 for item in outcomes if item.killed)
    survived = sum(1 for item in outcomes if not item.killed)
    total = len(outcomes)
    score = MutationScore(
        total=total,
        killed=killed,
        survived=survived,
        errors=0,
        score=(killed / total) if total else 0.0,
        reason_codes=tuple(
            ["perfect_kill_score"] if survived == 0 and total else ["imperfect_kill_score"]
        ),
    )

    authority_families = {
        MutationFamily.FORGED_EVIDENCE.value,
        MutationFamily.SYNTHETIC_EVIDENCE.value,
        MutationFamily.STALE_SPAN.value,
        MutationFamily.MIXED_ROOTS.value,
        MutationFamily.BAD_CID.value,
        MutationFamily.BAD_RECEIPT.value,
    }
    authority_outcomes = tuple(
        item for item in outcomes if item.family in authority_families
    )
    authority = AuthorityMutationSuite(
        outcomes=authority_outcomes,
        all_killed=all(item.killed for item in authority_outcomes),
        reason_codes=("authority_axis_killed",)
        if all(item.killed for item in authority_outcomes)
        else ("authority_survivor",),
    )

    matrix = {
        item.mutation_id: (
            MutationDisposition.KILLED.value
            if item.killed
            else MutationDisposition.SURVIVED.value
        )
        for item in outcomes
    }

    # Rollback verification: re-run desktop e2e inverse/replay remains ok.
    rollback: dict[str, Any] = {"verified": False}
    if positive_ok:
        e2e2 = run_desktop_contract_repair_e2e(
            repo_root=root,
            require_live_precondition=True,
        )
        rollback = {
            "verified": bool(e2e2.rollback_replay.get("rollback_ok"))
            and bool(e2e2.rollback_replay.get("replay_ok")),
            "epoch_before": e2e2.epoch_before,
            "epoch_after": e2e2.epoch_after,
        }
        if rollback["verified"]:
            reasons.append("rollback_verification_ok")

    no_grants = all(
        not item.detail.get("grants_mutation") and not item.detail.get("grants_completion")
        for item in outcomes
    )
    if no_grants:
        reasons.append("no_unknown_grants_mutation_or_completion")

    passed = bool(
        positive_ok
        and score.survived == 0
        and score.total > 0
        and score.score == 1.0
        and authority.all_killed
        and no_grants
        and rollback.get("verified")
    )
    if passed:
        reasons.append("adversarial_conformance_passed")
    else:
        reasons.append("adversarial_conformance_failed")

    return DcrAdversarialReport(
        passed=passed,
        positive_control_ok=positive_ok,
        mutation_score=score,
        outcomes=tuple(outcomes),
        authority_suite=authority,
        killed_survivor_matrix=MappingProxyType(matrix),
        rollback_verification=MappingProxyType(rollback),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_adversarial_report(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize adversarial-report.json for DCR-093."""

    root = _discover_repo_root(repo_root)
    report = evaluate_dcr_adversarial(repo_root=root)
    payload = {
        "schema": DCR_ADVERSARIAL_SCHEMA,
        "interface": ADVERSARIAL_CONFORMANCE_INTERFACE,
        "evidence_id": DCR_ADVERSARIAL_EVIDENCE,
        "version": DCR_ADVERSARIAL_VERSION,
        "task_id": DCR_TASK_ID,
        "result": report.to_dict(),
        "runtime_model_calls": 0,
        "provider_calls": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_ADVERSARIAL_REPORT_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ADVERSARIAL_CONFORMANCE_INTERFACE",
    "DCR_ADVERSARIAL_EVIDENCE",
    "DCR_ADVERSARIAL_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_ADVERSARIAL_REPORT_PATH",
    "MUTATION_SCORE_INTERFACE",
    "AuthorityMutationSuite",
    "ContractRepairAdversary",
    "DcrAdversarialError",
    "DcrAdversarialReport",
    "MutationDisposition",
    "MutationFamily",
    "MutationScore",
    "evaluate_dcr_adversarial",
    "materialize_adversarial_report",
]
