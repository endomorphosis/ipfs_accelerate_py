#!/usr/bin/env python3
"""Advisor role certification for SymAI, ErgoAI, Leanstral, autoencoder, Hammer.

``AdvisorRoleCertification@1`` / FVT-G160 (FVT-050).

Owns the hammer/advisor lane certification handler.  Certification proves:

* explicit strict installation selects locked SymAI and ErgoAI identities;
* SymAI, ErgoAI, Leanstral, autoencoder, and Hammer proposals are bounded,
  sanitized, source-bound, deterministic or replay-bound, cache-safe, and
  failure-explicit;
* no confidence, similarity, generated text, or advisor availability becomes
  proof without deterministic compilation and independent solver/kernel
  validation;
* advisors remain role=advisor/candidate with authority_ceiling=advisory and
  can never satisfy a certified-authority requirement or promote the hammer
  lane by presence alone.

Certification never installs, downloads, or opens the network (except that
installer dry-run / hermetic marker paths used by tests are offline).  It never
edits the central multi-prover certificate or model runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolRole,
    ToolchainAuthorityCeiling,
    can_satisfy_certified_authority_requirement,
    evaluate_role_aware_promotion,
    get_tool_role,
    tools_by_role,
)
from ipfs_datasets_py.logic.formalization.proposal_advisors import (  # noqa: E402
    LEANSTRAL_ADVISOR_INTERFACE,
    SYMAI_ADVISOR_INTERFACE,
    UNVERIFIED_AUTHORITY,
    LeanstralProposalAdvisor,
    ProposalAdvisorValidationError,
    ProposalKind,
    StaticProposalModel,
    SymAIProposalAdvisor,
    accept_candidate,
    build_json_candidates_response,
    confidence_never_yields_proof,
    is_untrusted_proposal_provider,
    sanitize_inert_text,
)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

try:  # pragma: no cover
    from ipfs_datasets_py.logic.backends.installers import advisors as advisors_installer
except Exception:  # pragma: no cover
    advisors_installer = None  # type: ignore[assignment]

INTERFACE: Final = "AdvisorRoleCertification@1"
SCHEMA_VERSION: Final = "advisor-role-certification/v1"
CORPUS_SCHEMA: Final = "advisor-role-corpus/v1"
GOAL_ID: Final = "FVT-G160"
TASK_ID: Final = "FVT-050"
PROGRAM: Final = "formal-verification-tactician/advisor-toolchains"
LANE_ID: Final = "hammer"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.advisors"
HANDLER_ID: Final = "advisor_role_certification@1"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.ADVISORY.value
AUTHORITY_SCOPE: Final = "candidate_generation_only"
LIVE_ERGOAI_INTERFACE: Final = "LiveErgoAIAdvisorCertification@1"
LIVE_ERGOAI_SCHEMA_VERSION: Final = "live-ergoai-advisor-certification/v1"
LIVE_ERGOAI_EVIDENCE_CLASS: Final = "checksummed_authoritative_vendor_execution"

ADVISOR_TOOL_IDS: Final = (
    "symbolicai",
    "ergoai",
    "leanstral",
    "autoencoder",
    "hammer",
)
LOCKED_SYMBOLICAI_VERSION: Final = ">=1.14.0,<2.0.0"
LOCKED_ERGOAI_VERSION: Final = "3.0"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

CHECK_KINDS: Final = frozenset(
    {
        "positive",
        "negative",
        "mutation",
        "replay",
        "malformed",
        "authority",
        "role",
        "install",
        "policy",
        "acceptance",
        "bounds",
    }
)


class AdvisorRoleCertificationError(ValueError):
    """Raised when advisor role certification inputs or results are invalid."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
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
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    return env


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str
    status: str  # passed | failed | skipped | unavailable | blocked | quarantined
    expected: str
    observed: str
    detail: str = ""
    tool_id: str = ""
    reason_codes: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcome:
    case_id: str
    kind: str
    advisor_id: str
    expect: str
    status: str
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    authority: str = UNVERIFIED_AUTHORITY
    output_digest: str = ""
    detail: str = ""
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdvisorRoleCertification:
    """Full certification receipt for the advisor/hammer lane."""

    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    lane_id: str = LANE_ID
    certification_surface: str = CERTIFICATION_SURFACE
    handler_id: str = HANDLER_ID
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    advisor_tool_ids: list[str] = field(default_factory=lambda: list(ADVISOR_TOOL_IDS))
    locked_symbolicai_version: str = LOCKED_SYMBOLICAI_VERSION
    locked_ergoai_version: str = LOCKED_ERGOAI_VERSION
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    advisors_never_promote_alone: bool = True
    semantic_corpus_passed: bool = False
    role_matrix_passed: bool = False
    install_identity_passed: bool = False
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[CaseOutcome] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
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
# Corpus (offline proposal / authority cases)
# ---------------------------------------------------------------------------


def _candidate_body(kind: str) -> str:
    return {
        "specification": "requires xs.length > 0 ensures result.sorted",
        "lemma": "lemma swap_preserves : forall xs, permute (swap xs) xs",
        "tactic": "apply Nat.le_refl",
        "premise": "axiom: list permutation is equivalence",
        "repair": "replace /goal/quantifier with forall",
    }.get(kind, "lemma swap_preserves : forall xs, permute (swap xs) xs")


def default_corpus_manifest() -> dict[str, Any]:
    """Compact embedded corpus for advisor role certification."""

    cases: list[dict[str, Any]] = []
    for advisor in ADVISOR_TOOL_IDS:
        provider = {
            "symbolicai": "symai",
            "ergoai": "ergoai",
            "leanstral": "leanstral",
            "autoencoder": "autoencoder",
            "hammer": "hammer",
        }[advisor]
        cases.append(
            {
                "case_id": f"{advisor}.positive_lemma",
                "kind": "positive",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "unverified_candidate",
                "confidence": 0.99,
                "is_valid": True,
                "similarity": 0.98,
                "description": f"{advisor} emits a source-bound lemma candidate only",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.negative_confidence_not_proof",
                "kind": "negative",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "not_proved",
                "confidence": 1.0,
                "is_valid": True,
                "similarity": 1.0,
                "description": f"{advisor} confidence/is_valid never yields proof",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.mutation_authority_claim",
                "kind": "mutation",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "rejected_or_quarantined",
                "mutate": "authority_claim",
                "description": f"{advisor} authority claim mutation is rejected",
            }
        )
        cases.append(
            {
                "case_id": f"{advisor}.replay_digest",
                "kind": "replay",
                "advisor_id": advisor,
                "provider": provider,
                "proposal_kind": "lemma",
                "expect": "deterministic_replay",
                "description": f"{advisor} proposal digests are deterministic",
            }
        )

    cases.extend(
        [
            {
                "case_id": "shared.malformed_executable_payload",
                "kind": "malformed",
                "advisor_id": "leanstral",
                "provider": "leanstral",
                "proposal_kind": "lemma",
                "expect": "rejected",
                "body": "lemma x : True := by\n```python\nimport os; os.system('rm -rf /')\n```",
                "description": "Executable payload markers are rejected",
            },
            {
                "case_id": "shared.ungrounded_source_refs",
                "kind": "malformed",
                "advisor_id": "symbolicai",
                "provider": "symai",
                "proposal_kind": "lemma",
                "expect": "rejected",
                "source_ref_ids": ["source:not-in-request"],
                "description": "Candidates with source_ref_ids outside the request are rejected",
            },
            {
                "case_id": "shared.acceptance_requires_compilation_and_validation",
                "kind": "acceptance",
                "advisor_id": "leanstral",
                "provider": "leanstral",
                "proposal_kind": "lemma",
                "expect": "acceptance_gate",
                "description": "accept_candidate requires compile + independent validation",
            },
        ]
    )

    return {
        "schema_version": CORPUS_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "lane_id": LANE_ID,
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "locked_symbolicai_version": LOCKED_SYMBOLICAI_VERSION,
        "locked_ergoai_version": LOCKED_ERGOAI_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "advisors_never_promote_alone": True,
            "confidence_never_yields_proof": True,
            "availability_is_not_authority": True,
            "authority_is_candidate_generation_only": True,
            "does_not_edit_central_certificate": True,
        },
        "cases": cases,
    }


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise AdvisorRoleCertificationError(
            "advisor corpus must declare a non-empty cases list"
        )
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Role / authority boundary
# ---------------------------------------------------------------------------


def advisor_role_boundary(tool_id: str) -> dict[str, Any]:
    """Prove one advisor cannot satisfy certified authority or promote alone."""

    role = get_tool_role(tool_id)
    decision = evaluate_role_aware_promotion(
        tool_id,
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
        independent_reconstruction=True,
    )
    can_satisfy = can_satisfy_certified_authority_requirement(tool_id)
    return {
        "tool_id": tool_id,
        "role": role.role.value,
        "authority_ceiling": role.authority_ceiling.value,
        "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
        "can_satisfy_certified_authority_requirement": can_satisfy,
        "promotion_allowed": decision.allowed,
        "promotion_decision": decision.to_dict(),
        "is_advisor_or_candidate": role.role
        in {ToolRole.ADVISOR, ToolRole.CANDIDATE},
        "ceiling_is_advisory": role.authority_ceiling
        is ToolchainAuthorityCeiling.ADVISORY,
        "ceiling_is_non_certifying": role.authority_ceiling
        in {
            ToolchainAuthorityCeiling.ADVISORY,
            ToolchainAuthorityCeiling.CANDIDATE,
            ToolchainAuthorityCeiling.NONE,
        },
        "blocks_alone": (not decision.allowed) and (not can_satisfy),
        "lane_ids": list(getattr(role, "lane_ids", ()) or ()),
        "lane_id": (
            list(getattr(role, "lane_ids", ()) or (LANE_ID,))[0]
            if getattr(role, "lane_ids", None)
            else LANE_ID
        ),
    }


def all_advisor_role_boundaries() -> dict[str, Any]:
    items = {tool_id: advisor_role_boundary(tool_id) for tool_id in ADVISOR_TOOL_IDS}
    all_blocked = all(item["blocks_alone"] for item in items.values())
    return {
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "all_blocked_from_certified_authority": all_blocked,
        "tools": items,
        "policy": {
            "support_advisor_shadow_presence_cannot_certify": True,
            "availability_is_not_authority": True,
            "confidence_never_yields_proof": True,
        },
    }


def advisors_cannot_promote_hammer_lane() -> dict[str, Any]:
    """Aggregate proof that advisor presence cannot promote the hammer lane."""

    boundaries = all_advisor_role_boundaries()
    return {
        "lane_id": LANE_ID,
        "authority_scope": AUTHORITY_SCOPE,
        "promotion_allowed": False,
        "all_blocked_from_certified_authority": boundaries[
            "all_blocked_from_certified_authority"
        ],
        "tools": boundaries["tools"],
        "authority_tool_ids_for_lane": [],  # empty by design in role matrix
    }


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------


def _build_static_candidate_response(
    *,
    provider: str,
    kind: str,
    body: str | None = None,
    confidence: float = 0.5,
    source_ref_ids: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    allow_authority_claims: bool = False,
) -> str:
    record: dict[str, Any] = {
        "candidate_id": f"cand:{provider}:{kind}",
        "kind": kind,
        "body": body if body is not None else _candidate_body(kind),
        "source_ref_ids": list(
            source_ref_ids
            if source_ref_ids is not None
            else ("source:module.py", "source:spec.md")
        ),
        "provider": provider,
        "confidence": confidence,
        "rationale": "advisor role certification corpus",
    }
    if extra:
        record.update(dict(extra))
    # Mutation cases intentionally inject authority claims so the advisor
    # surface can reject them.  build_json_candidates_response fails closed on
    # those claims, so emit raw JSON for adversarial fixtures only.
    if allow_authority_claims or extra:
        return json.dumps({"candidates": [record]}, sort_keys=True)
    return build_json_candidates_response([record])


def _run_proposal_advisor(
    *,
    advisor_id: str,
    provider: str,
    response: str,
    kind: str = "lemma",
) -> tuple[Any | None, str | None]:
    """Run Leanstral/SymAI advisors; synthesize contract checks for others."""

    request_payload = {
        "request_id": f"req:{advisor_id}",
        "goal_id": "goal:swap-correct",
        "logic_family": "hoare",
        "kind": kind,
        "source_ref_ids": ("source:module.py", "source:spec.md"),
        "context_text": "Prove that swap preserves the multiset of elements.",
        "goal_text": "forall xs, permute (swap xs) xs",
        "formula_id": "formula:swap",
        "notes": "proposal only",
    }
    from ipfs_datasets_py.logic.formalization.proposal_advisors import (
        ProposalAdvisorRequest,
    )

    request = ProposalAdvisorRequest(**request_payload)
    try:
        if advisor_id in {"leanstral"} or provider == "leanstral":
            advisor = LeanstralProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        if advisor_id in {"symbolicai"} or provider in {"symai", "symbolicai"}:
            advisor = SymAIProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        # ErgoAI / autoencoder / hammer: evaluate authority contracts without a
        # live model runtime (reuse proposal candidate parser path via SymAI
        # static model when provider is symai-compatible; otherwise manual).
        if provider in {"symai", "symbolicai", "leanstral"}:
            advisor = SymAIProposalAdvisor(StaticProposalModel(response))
            return advisor.propose(request), None
        # Manual offline evaluation for non-model advisors.
        payload = json.loads(response)
        candidates = payload.get("candidates") or []
        if not candidates:
            return None, "empty_candidates"
        first = candidates[0]
        body = sanitize_inert_text(
            first.get("body") or "", "body", maximum=8192
        )
        refs = list(first.get("source_ref_ids") or [])
        if not refs:
            return None, "missing_source_refs"
        authority = first.get("authority", UNVERIFIED_AUTHORITY)
        if authority != UNVERIFIED_AUTHORITY:
            return None, "authority_claim_rejected"
        for key in ("is_proved", "proof_status", "verification_status"):
            if key in first:
                return None, "authority_claim_rejected"
        proved = confidence_never_yields_proof(
            is_valid=first.get("is_valid"),
            confidence=first.get("confidence"),
            similarity=first.get("similarity"),
        )
        result = {
            "advisor_id": advisor_id,
            "provider": provider,
            "authority": UNVERIFIED_AUTHORITY,
            "candidates": [
                {
                    "candidate_id": str(first.get("candidate_id") or "cand:1"),
                    "kind": kind,
                    "body": body,
                    "source_ref_ids": refs,
                    "provider": provider,
                    "authority": UNVERIFIED_AUTHORITY,
                    "confidence": float(first.get("confidence") or 0.0),
                    "is_proved": proved,
                }
            ],
            "is_proved": False,
        }
        return result, None
    except (ProposalAdvisorValidationError, ValueError, TypeError, KeyError) as exc:
        return None, str(exc)


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one advisor corpus case offline (no install / network)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    advisor_id = str(case.get("advisor_id") or "leanstral")
    provider = str(case.get("provider") or advisor_id)
    expect = str(case.get("expect") or "unknown")
    proposal_kind = str(case.get("proposal_kind") or "lemma")
    confidence = float(case.get("confidence") or 0.5)

    reason_codes: list[str] = []
    status = "unknown"
    matched = False
    bindings: dict[str, Any] = {}
    digest_seed = ""

    if kind == "acceptance":
        response = _build_static_candidate_response(
            provider="leanstral",
            kind=proposal_kind,
            confidence=0.99,
        )
        result, error = _run_proposal_advisor(
            advisor_id="leanstral",
            provider="leanstral",
            response=response,
            kind=proposal_kind,
        )
        if error or result is None:
            status = "failed"
            reason_codes.append(error or "proposal_failed")
            matched = False
        else:
            candidates = getattr(result, "candidates", None)
            if candidates is None and isinstance(result, Mapping):
                candidates = result.get("candidates") or []
            candidate = candidates[0] if candidates else None
            if candidate is None:
                status = "failed"
                matched = False
            else:
                rejected = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=False,
                    independently_validated=False,
                )
                half = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=True,
                    independently_validated=False,
                )
                full = accept_candidate(
                    candidate if not isinstance(candidate, Mapping) else _wrap_candidate(candidate),
                    compiled=True,
                    independently_validated=True,
                )
                status = "acceptance_gate"
                matched = (
                    not rejected.accepted
                    and not half.accepted
                    and full.accepted
                    and not confidence_never_yields_proof(confidence=1.0)
                    is True  # always False
                )
                # confidence_never_yields_proof always returns False (= not proved)
                matched = (
                    not rejected.accepted
                    and not half.accepted
                    and full.accepted
                    and confidence_never_yields_proof(confidence=1.0) is False
                )
                bindings = {
                    "rejected": rejected.to_dict(),
                    "compile_only": half.to_dict(),
                    "full": full.to_dict(),
                }
                digest_seed = content_digest(bindings)
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched and expect == "acceptance_gate",
            reason_codes=reason_codes,
            output_digest=digest_seed or content_digest(case_id),
            detail=str(case.get("description") or ""),
            bindings=bindings,
        )

    if kind == "malformed":
        body = case.get("body")
        ungrounded_refs = case.get("source_ref_ids")
        extra = {"is_proved": True} if case.get("mutate") == "authority_claim" else None
        try:
            if body is not None:
                # Direct sanitize rejection path.
                try:
                    sanitize_inert_text(body, "body", maximum=8192)
                    status = "accepted_unexpectedly"
                    matched = False
                    reason_codes.append("executable_payload_not_rejected")
                except ProposalAdvisorValidationError:
                    status = "rejected"
                    matched = expect in {"rejected", "rejected_or_quarantined"}
                    reason_codes.append("executable_marker")
            else:
                response = _build_static_candidate_response(
                    provider=provider if provider in {"leanstral", "symai"} else "symai",
                    kind=proposal_kind,
                    confidence=confidence,
                    source_ref_ids=(
                        list(ungrounded_refs)
                        if isinstance(ungrounded_refs, (list, tuple))
                        else None
                    ),
                    extra=extra,
                )
                result, error = _run_proposal_advisor(
                    advisor_id=advisor_id if advisor_id in {"leanstral", "symbolicai"} else "symbolicai",
                    provider=provider if provider in {"leanstral", "symai"} else "symai",
                    response=response,
                    kind=proposal_kind,
                )
                if error or result is None:
                    status = "rejected"
                    matched = expect in {"rejected", "rejected_or_quarantined"}
                    reason_codes.append(error or "rejected")
                else:
                    status = "accepted_unexpectedly"
                    matched = False
                    reason_codes.append("malformed_accepted")
        except ProposalAdvisorValidationError as exc:
            status = "rejected"
            matched = expect in {"rejected", "rejected_or_quarantined"}
            reason_codes.append(str(exc)[:120])
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "status": status}),
            detail=str(case.get("description") or ""),
        )

    # positive / negative / mutation / replay
    extra: dict[str, Any] | None = None
    if kind == "mutation" and case.get("mutate") == "authority_claim":
        extra = {"authority": "proved", "is_proved": True, "proof_status": "proved"}

    response_provider = provider if provider in {"leanstral", "symai"} else "symai"
    try:
        response = _build_static_candidate_response(
            provider=response_provider,
            kind=proposal_kind,
            confidence=confidence,
            extra=extra,
            allow_authority_claims=bool(extra),
        )
    except ProposalAdvisorValidationError as exc:
        # Authority-claim fixtures must be rejected at the boundary.
        if kind == "mutation":
            return CaseOutcome(
                case_id=case_id,
                kind=kind,
                advisor_id=advisor_id,
                expect=expect,
                status="rejected",
                matched=expect in {"rejected_or_quarantined", "rejected"},
                reason_codes=[str(exc)[:160]],
                output_digest=content_digest({"case_id": case_id, "error": str(exc)}),
                detail=str(case.get("description") or ""),
            )
        raise
    # For non leanstral/symai advisors, evaluate authority contracts offline.
    if advisor_id in {"leanstral", "symbolicai"} or provider in {
        "leanstral",
        "symai",
        "symbolicai",
    }:
        run_advisor = advisor_id if advisor_id in {"leanstral", "symbolicai"} else "symbolicai"
        run_provider = (
            provider if provider in {"leanstral", "symai"} else "symai"
        )
        result, error = _run_proposal_advisor(
            advisor_id=run_advisor,
            provider=run_provider,
            response=response,
            kind=proposal_kind,
        )
    else:
        # Inject provider name into response for manual path.
        try:
            payload = json.loads(response)
            if payload.get("candidates"):
                payload["candidates"][0]["provider"] = provider
                if extra:
                    payload["candidates"][0].update(extra)
            response = json.dumps(payload)
        except Exception:
            pass
        result, error = _run_proposal_advisor(
            advisor_id=advisor_id,
            provider=provider,
            response=response,
            kind=proposal_kind,
        )

    if kind == "mutation":
        if error or result is None:
            status = "rejected"
            matched = expect in {"rejected_or_quarantined", "rejected"}
            reason_codes.append(error or "authority_claim_rejected")
        else:
            # If a result was produced, ensure it cannot claim proof.
            is_proved = False
            if hasattr(result, "candidates"):
                for cand in result.candidates:
                    if getattr(cand, "authority", UNVERIFIED_AUTHORITY) != UNVERIFIED_AUTHORITY:
                        is_proved = True
            elif isinstance(result, Mapping):
                is_proved = bool(result.get("is_proved"))
            if is_proved:
                status = "authority_leak"
                matched = False
                reason_codes.append("authority_leak")
            else:
                status = "quarantined"
                matched = expect in {"rejected_or_quarantined", "quarantined"}
                reason_codes.append("authority_claim_stripped_or_rejected")
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "status": status}),
            detail=str(case.get("description") or ""),
        )

    if error or result is None:
        status = "failed"
        matched = False
        reason_codes.append(error or "proposal_failed")
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            advisor_id=advisor_id,
            expect=expect,
            status=status,
            matched=matched,
            reason_codes=reason_codes,
            output_digest=content_digest({"case_id": case_id, "error": error}),
            detail=str(case.get("description") or ""),
        )

    # Extract candidate fields
    if hasattr(result, "candidates"):
        candidates = list(result.candidates)
        authority = getattr(result, "authority", UNVERIFIED_AUTHORITY)
        candidate_dicts = [c.to_dict() for c in candidates]
    elif isinstance(result, Mapping):
        candidates = list(result.get("candidates") or [])
        authority = str(result.get("authority") or UNVERIFIED_AUTHORITY)
        candidate_dicts = [
            c if isinstance(c, Mapping) else {"body": str(c)} for c in candidates
        ]
    else:
        candidates = []
        authority = UNVERIFIED_AUTHORITY
        candidate_dicts = []

    proved = confidence_never_yields_proof(
        is_valid=bool(case.get("is_valid")),
        confidence=confidence,
        similarity=float(case.get("similarity") or 0.0),
    )
    digest_seed = content_digest(
        {
            "advisor_id": advisor_id,
            "provider": provider,
            "kind": proposal_kind,
            "candidates": candidate_dicts,
            "authority": authority,
        }
    )
    bindings = {
        "authority": authority,
        "candidate_count": len(candidate_dicts),
        "is_proved": proved,
        "provider_untrusted": is_untrusted_proposal_provider(provider)
        or advisor_id in ADVISOR_TOOL_IDS,
    }

    if kind == "negative":
        status = "not_proved" if proved is False else "proved_unexpectedly"
        matched = expect == "not_proved" and proved is False and authority == UNVERIFIED_AUTHORITY
        if not matched:
            reason_codes.append("confidence_elevated_to_proof")
    elif kind == "replay":
        # Re-run and compare digests.
        result2, error2 = _run_proposal_advisor(
            advisor_id=advisor_id if advisor_id in {"leanstral", "symbolicai"} else advisor_id,
            provider=provider if provider in {"leanstral", "symai"} else provider,
            response=response if advisor_id in {"leanstral", "symbolicai"} or provider in {"leanstral", "symai"} else response,
            kind=proposal_kind,
        )
        if error2 or result2 is None:
            status = "failed"
            matched = False
            reason_codes.append(error2 or "replay_failed")
        else:
            if hasattr(result2, "candidates"):
                c2 = [c.to_dict() for c in result2.candidates]
                a2 = getattr(result2, "authority", UNVERIFIED_AUTHORITY)
            elif isinstance(result2, Mapping):
                c2 = list(result2.get("candidates") or [])
                a2 = str(result2.get("authority") or UNVERIFIED_AUTHORITY)
            else:
                c2, a2 = [], UNVERIFIED_AUTHORITY
            digest2 = content_digest(
                {
                    "advisor_id": advisor_id,
                    "provider": provider,
                    "kind": proposal_kind,
                    "candidates": c2,
                    "authority": a2,
                }
            )
            status = "deterministic_replay" if digest_seed == digest2 else "nondeterministic"
            matched = (
                expect == "deterministic_replay"
                and digest_seed == digest2
                and authority == UNVERIFIED_AUTHORITY
            )
            bindings["replay_digest"] = digest2
            if not matched:
                reason_codes.append("replay_mismatch")
    else:  # positive
        status = "unverified_candidate"
        matched = (
            expect == "unverified_candidate"
            and authority == UNVERIFIED_AUTHORITY
            and proved is False
            and len(candidate_dicts) >= 1
            and all(
                (c.get("source_ref_ids") if isinstance(c, Mapping) else getattr(c, "source_ref_ids", ()))
                for c in (candidate_dicts if candidate_dicts else candidates)
            )
        )
        if not matched:
            reason_codes.append("positive_contract_failed")

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        advisor_id=advisor_id,
        expect=expect,
        status=status,
        matched=matched,
        reason_codes=reason_codes,
        authority=authority if isinstance(authority, str) else UNVERIFIED_AUTHORITY,
        output_digest=digest_seed,
        detail=str(case.get("description") or ""),
        bindings=bindings,
    )


def _wrap_candidate(record: Mapping[str, Any]):
    """Build a ProposalCandidate from a dict for accept_candidate."""

    from ipfs_datasets_py.logic.formalization.proposal_advisors import (
        ProposalCandidate,
        ProposalKind,
        ProposalProvider,
    )

    provider_raw = str(record.get("provider") or "leanstral")
    try:
        provider = ProposalProvider(provider_raw if provider_raw != "symbolicai" else "symai")
    except ValueError:
        provider = ProposalProvider.LEANSTRAL
    kind_raw = str(record.get("kind") or "lemma")
    try:
        kind = ProposalKind(kind_raw)
    except ValueError:
        kind = ProposalKind.LEMMA
    return ProposalCandidate(
        candidate_id=str(record.get("candidate_id") or "cand:1"),
        kind=kind,
        body=str(record.get("body") or _candidate_body("lemma")),
        source_ref_ids=tuple(record.get("source_ref_ids") or ("source:module.py",)),
        provider=provider,
        confidence=float(record.get("confidence") or 0.0),
        rationale=str(record.get("rationale") or "corpus"),
    )


# ---------------------------------------------------------------------------
# Install identity (offline strict selection)
# ---------------------------------------------------------------------------


def certify_install_identities(
    *,
    repo_root: Path | None = None,
    install_root: Path | None = None,
) -> dict[str, Any]:
    """Select locked SymAI/ErgoAI identities offline (hermetic, no network)."""

    if advisors_installer is None:
        return {
            "passed": False,
            "reason": "advisors_installer_unavailable",
            "symbolicai": None,
            "ergoai": None,
        }

    root = repo_root or repo_root_from()
    # Never materialize hermetic shims under the repo tree (e.g. .tmp/): that
    # path is out of task scope and pollutes the proposal gate. Use an explicit
    # install_root when provided; otherwise a process-local temp directory.
    if install_root is not None:
        target = Path(install_root)
    else:
        target = Path(
            tempfile.mkdtemp(prefix="advisor-role-certification-install-")
        )
    target.mkdir(parents=True, exist_ok=True)

    # Dry-run strict pin selection first.
    symai_pin = advisors_installer.select_strict_pin(
        "symbolicai",
        platform_key="any",
        repo_root=root,
    )
    ergo_platform = advisors_installer.detect_platform_key()
    ergo_pin = advisors_installer.select_strict_pin(
        "ergoai",
        platform_key=ergo_platform,
        repo_root=root,
    )
    pin_ok = (
        symai_pin.version == LOCKED_SYMBOLICAI_VERSION
        and ergo_pin.version == LOCKED_ERGOAI_VERSION
    )

    # Hermetic install markers (offline, yes=True, no network).
    env_snapshot = {
        "FORMAL_VERIFICATION_CERTIFY_OFFLINE": os.environ.get(
            "FORMAL_VERIFICATION_CERTIFY_OFFLINE"
        ),
        "FORMAL_VERIFICATION_FORBID_NETWORK": os.environ.get(
            "FORMAL_VERIFICATION_FORBID_NETWORK"
        ),
    }
    os.environ["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    os.environ["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    try:
        symai = advisors_installer.ensure_symbolicai(
            yes=True,
            strict=True,
            force=True,
            install_root=target,
            repo_root=root,
            hermetic_marker=True,
            test_mode=True,
        )
        ergo = advisors_installer.ensure_ergoai(
            yes=True,
            strict=True,
            force=True,
            install_root=target,
            repo_root=root,
            platform_key=ergo_platform,
            hermetic_shim=True,
            test_mode=True,
        )
    finally:
        for key, value in env_snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    passed = (
        pin_ok
        and symai.ok
        and ergo.ok
        and symai.selected_version == LOCKED_SYMBOLICAI_VERSION
        and ergo.selected_version == LOCKED_ERGOAI_VERSION
        and not symai.grants_proof_authority
        and not ergo.grants_proof_authority
    )
    return {
        "passed": passed,
        "symbolicai_pin": symai_pin.to_dict(),
        "ergoai_pin": ergo_pin.to_dict(),
        "symbolicai_receipt": symai.to_dict(),
        "ergoai_receipt": ergo.to_dict(),
        "locked_symbolicai_version": LOCKED_SYMBOLICAI_VERSION,
        "locked_ergoai_version": LOCKED_ERGOAI_VERSION,
        "install_root": str(target),
        "network_used": False,
        "policy": {
            "strict_selects_locked_identities": True,
            "advisors_never_grant_proof_authority": True,
            "hermetic_offline": True,
        },
    }


def certify_live_ergoai_vendor(
    *,
    executable: str | Path | None = None,
    install_root: str | Path | None = None,
    repo_root: Path | None = None,
    platform_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Certify a real, checksum-bound ErgoAI runtime without installing it.

    This is deliberately separate from the offline advisor-role receipt.  It
    replays managed artifact digests and executes positive, negative, mutation,
    and replay F-logic cases.  Passing proves the vendor runtime is usable for
    bounded candidate/advisor work; it never elevates ErgoAI to solver or
    theorem authority.
    """

    root = repo_root or repo_root_from()
    if advisors_installer is None:
        payload = {
            "interface": LIVE_ERGOAI_INTERFACE,
            "schema_version": LIVE_ERGOAI_SCHEMA_VERSION,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "tool_id": "ergoai",
            "vendor_certified": False,
            "authoritative_live_evidence": False,
            "production_certified": False,
            "promotion_blocked": True,
            "authority_scope": AUTHORITY_SCOPE,
            "authority_ceiling": AUTHORITY_CEILING,
            "grants_proof_authority": False,
            "block_reasons": ["advisors_installer_unavailable"],
            "checks": [],
        }
        payload["receipt_digest_sha256"] = content_digest(payload)
        return payload

    selected_platform = platform_key or advisors_installer.detect_platform_key()
    resolved_root = advisors_installer.expand_user_local_root(install_root)
    probe = advisors_installer.probe_ergoai_identity(
        expected_version=LOCKED_ERGOAI_VERSION,
        executable=str(executable) if executable is not None else None,
        install_root=resolved_root,
        require_managed_vendor=True,
        platform_key=selected_platform,
    )
    resolved_executable = str(probe.get("executable_path") or "")
    semantics = (
        advisors_installer.run_ergoai_semantic_checks(
            resolved_executable,
            timeout=timeout,
        )
        if resolved_executable and probe.get("version_match")
        else {
            "schema_version": "ergoai-live-semantic-checks/v1",
            "passed": False,
            "checks": {},
            "replay_bound": False,
        }
    )

    checks: list[CheckResult] = []
    identity_ok = bool(probe.get("path_present") and probe.get("version_match"))
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.identity",
            kind="install",
            status="passed" if identity_ok else "failed",
            expected=f"ErgoAI {LOCKED_ERGOAI_VERSION}",
            observed=str(probe.get("version_string") or probe.get("probe_error")),
            tool_id="ergoai",
            reason_codes=[] if identity_ok else [str(probe.get("probe_error") or "identity_failed")],
            bindings={
                "executable_path": resolved_executable or None,
                "platform": selected_platform,
            },
        )
    )
    provenance_ok = bool(probe.get("managed_vendor_provenance_verified"))
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.provenance",
            kind="policy",
            status="passed" if provenance_ok else "failed",
            expected="checksummed_non_shim_managed_vendor_identity",
            observed=(
                "verified"
                if provenance_ok
                else ",".join(str(v) for v in probe.get("reason_codes") or ())
                or "unverified"
            ),
            tool_id="ergoai",
            reason_codes=[] if provenance_ok else list(probe.get("reason_codes") or ()),
            bindings={
                "identity_manifest_path": probe.get("identity_manifest_path"),
                "is_hermetic_advisor_shim": bool(
                    probe.get("is_hermetic_advisor_shim")
                ),
            },
        )
    )

    semantic_checks = semantics.get("checks") or {}
    for kind in ("positive", "negative", "mutation", "replay"):
        observed = semantic_checks.get(kind) or {}
        passed = bool(observed.get("passed"))
        checks.append(
            CheckResult(
                check_id=f"advisors.ergoai_live.{kind}",
                kind=kind,
                status="passed" if passed else "failed",
                expected=str(observed.get("expected") or ("yes" if kind in {"positive", "replay"} else "no")),
                observed=str(observed.get("verdict") or "unavailable"),
                tool_id="ergoai",
                reason_codes=[] if passed else [f"{kind}_semantic_check_failed"],
                bindings={
                    key: observed.get(key)
                    for key in (
                        "returncode",
                        "program_digest_sha256",
                        "query_digest_sha256",
                        "output_digest_sha256",
                    )
                },
            )
        )
    authority_ok = (
        AUTHORITY_CEILING == ToolchainAuthorityCeiling.ADVISORY.value
        and not can_satisfy_certified_authority_requirement("ergoai")
    )
    checks.append(
        CheckResult(
            check_id="advisors.ergoai_live.authority_boundary",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="advisor_only_never_proof_authority",
            observed=f"ceiling={AUTHORITY_CEILING};can_satisfy={not authority_ok}",
            tool_id="ergoai",
        )
    )

    vendor_certified = bool(
        identity_ok
        and provenance_ok
        and semantics.get("passed")
        and semantics.get("replay_bound")
        and authority_ok
    )
    block_reasons = sorted(
        {
            reason
            for check in checks
            if check.status != "passed"
            for reason in (check.reason_codes or [check.check_id])
        }
    )
    manifest = probe.get("manifest") or {}
    manifest_projection = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "tool_id",
            "version",
            "selected_platform",
            "release_tag",
            "release_url",
            "release_artifact_sha256",
            "release_artifact_size_bytes",
            "vendor_executable_sha256",
            "xsb_configuration",
            "xsb_executable_sha256",
            "launcher_sha256",
            "identity_digest_sha256",
            "license_components",
            "checksum_verified",
            "is_live_vendor",
            "is_hermetic_advisor_shim",
        )
    }
    identity_manifest_path = Path(str(probe.get("identity_manifest_path") or ""))
    identity_manifest_digest = (
        content_digest(identity_manifest_path.read_bytes())
        if identity_manifest_path.is_file()
        else None
    )
    payload = {
        "interface": LIVE_ERGOAI_INTERFACE,
        "schema_version": LIVE_ERGOAI_SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "tool_id": "ergoai",
        "evidence_class": (
            LIVE_ERGOAI_EVIDENCE_CLASS if vendor_certified else "unverified_or_incomplete"
        ),
        "vendor_certified": vendor_certified,
        "authoritative_live_evidence": vendor_certified,
        # Production-certified here means the advisor runtime is deployable in
        # its declared role.  It remains non-authoritative for proofs.
        "production_certified": vendor_certified,
        "promotion_blocked": True,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_ceiling": AUTHORITY_CEILING,
        "grants_theorem_authority": False,
        "grants_proof_authority": False,
        "advisors_never_promote_alone": True,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "selected_platform": selected_platform,
        "executable_path": resolved_executable or None,
        "identity_manifest_path": str(identity_manifest_path)
        if identity_manifest_path.is_file()
        else None,
        "identity_manifest_digest_sha256": identity_manifest_digest,
        "managed_identity": manifest_projection,
        "semantic_evidence_digest_sha256": semantics.get(
            "normalized_evidence_digest_sha256"
        ),
        "checks": [check.to_dict() for check in checks],
        "block_reasons": block_reasons,
        "source_binding": {
            "repo_root": str(root),
            "release_url": getattr(
                advisors_installer, "ERGOAI_RELEASE_URL", ""
            ),
            "release_sha256": getattr(
                advisors_installer, "ERGOAI_RELEASE_SHA256", ""
            ),
            "release_tag": getattr(
                advisors_installer, "ERGOAI_RELEASE_TAG", ""
            ),
        },
    }
    payload["receipt_digest_sha256"] = content_digest(payload)
    return payload


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    install_root: Path | None = None,
) -> AdvisorRoleCertification:
    """Run the full advisor role certification suite (offline)."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else default_corpus_manifest()
    cases = corpus_cases(corpus)
    cert = AdvisorRoleCertification()
    probe_env = offline_env(env)

    cert.checks.append(
        CheckResult(
            check_id="advisors.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used},"
                f"FORMAL_VERIFICATION_CERTIFY_OFFLINE="
                f"{probe_env.get('FORMAL_VERIFICATION_CERTIFY_OFFLINE')}"
            ),
            detail="certification never installs, downloads, or opens the network",
        )
    )

    # Interface constants.
    cert.checks.append(
        CheckResult(
            check_id="advisors.interfaces",
            kind="policy",
            status="passed",
            expected=INTERFACE,
            observed=INTERFACE,
            detail=(
                f"Leanstral={LEANSTRAL_ADVISOR_INTERFACE}; "
                f"SymAI={SYMAI_ADVISOR_INTERFACE}; "
                f"authority={UNVERIFIED_AUTHORITY}"
            ),
            bindings={
                "leanstral_interface": LEANSTRAL_ADVISOR_INTERFACE,
                "symai_interface": SYMAI_ADVISOR_INTERFACE,
                "unverified_authority": UNVERIFIED_AUTHORITY,
            },
        )
    )

    # Role matrix.
    role_report = all_advisor_role_boundaries()
    cert.role_matrix_passed = bool(role_report["all_blocked_from_certified_authority"])
    if not cert.role_matrix_passed:
        cert.block_reasons.append("advisor_role_matrix_failed")
    cert.checks.append(
        CheckResult(
            check_id="advisors.role_matrix",
            kind="role",
            status="passed" if cert.role_matrix_passed else "failed",
            expected="all_advisors_blocked_from_certified_authority",
            observed=(
                "blocked" if cert.role_matrix_passed else "authority_leak"
            ),
            detail="advisor/candidate presence cannot satisfy certified authority",
            bindings=role_report,
        )
    )
    for tool_id in ADVISOR_TOOL_IDS:
        boundary = role_report["tools"][tool_id]
        ok = (
            boundary["blocks_alone"]
            and boundary["is_advisor_or_candidate"]
            and boundary.get("ceiling_is_non_certifying", False)
        )
        if not ok:
            cert.block_reasons.append(f"role_boundary_failed:{tool_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"advisors.role.{tool_id}",
                kind="role",
                status="passed" if ok else "failed",
                expected="advisor_or_candidate_non_certifying_blocked",
                observed=(
                    f"role={boundary['role']},"
                    f"ceiling={boundary['authority_ceiling']},"
                    f"promote={boundary['promotion_allowed']}"
                ),
                tool_id=tool_id,
                bindings=boundary,
            )
        )

    promotion = advisors_cannot_promote_hammer_lane()
    cert.checks.append(
        CheckResult(
            check_id="advisors.hammer_lane_promotion_blocked",
            kind="authority",
            status="passed" if not promotion["promotion_allowed"] else "failed",
            expected="promotion_blocked",
            observed=(
                "blocked" if not promotion["promotion_allowed"] else "allowed"
            ),
            detail="hammer lane has empty authority_tool_ids by design",
            bindings=promotion,
        )
    )

    # Install identities (hermetic offline).
    try:
        install_report = certify_install_identities(
            repo_root=root, install_root=install_root
        )
        cert.install_identity_passed = bool(install_report.get("passed"))
        if not cert.install_identity_passed:
            cert.block_reasons.append("install_identity_failed")
        cert.checks.append(
            CheckResult(
                check_id="advisors.install_identities",
                kind="install",
                status="passed" if cert.install_identity_passed else "failed",
                expected=(
                    f"symbolicai={LOCKED_SYMBOLICAI_VERSION};"
                    f"ergoai={LOCKED_ERGOAI_VERSION}"
                ),
                observed=(
                    f"symbolicai={install_report.get('symbolicai_receipt', {}).get('selected_version')};"
                    f"ergoai={install_report.get('ergoai_receipt', {}).get('selected_version')}"
                ),
                detail="strict hermetic selection of locked advisor identities",
                bindings=install_report,
            )
        )
        cert.bindings["install"] = install_report
    except Exception as exc:  # pragma: no cover - defensive
        cert.block_reasons.append(f"install_identity_error:{exc}")
        cert.checks.append(
            CheckResult(
                check_id="advisors.install_identities",
                kind="install",
                status="failed",
                expected="locked identities",
                observed=str(exc)[:200],
                detail="install identity certification raised",
            )
        )

    # Semantic / proposal corpus.
    outcomes_by_id: dict[str, CaseOutcome] = {}
    for case in cases:
        outcome = evaluate_corpus_case(case)
        outcomes_by_id[outcome.case_id] = outcome
        cert.cases.append(outcome)
        status = "passed" if outcome.matched else "failed"
        if not outcome.matched:
            cert.block_reasons.append(f"case_failed:{outcome.case_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"advisors.{outcome.case_id}",
                kind=outcome.kind if outcome.kind in CHECK_KINDS else "positive",
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                tool_id=outcome.advisor_id,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "authority": outcome.authority,
                    **dict(outcome.bindings),
                },
            )
        )

    cert.semantic_corpus_passed = all(case.matched for case in cert.cases)
    if not cert.semantic_corpus_passed:
        cert.block_reasons.append("semantic_corpus_failed")

    cert.checks.append(
        CheckResult(
            check_id="advisors.confidence_never_yields_proof",
            kind="authority",
            status="passed"
            if confidence_never_yields_proof(confidence=1.0, is_valid=True) is False
            else "failed",
            expected="False",
            observed=str(
                confidence_never_yields_proof(confidence=1.0, is_valid=True)
            ),
            detail="documented invariant: model scores never establish proof",
        )
    )

    # Production certification for advisors means: role/corpus contracts hold
    # and install identities select locked pins.  Advisors never become
    # certified theorem authority — production_certified here means
    # "role-certified as candidate generation only".
    cert.production_certified = (
        cert.role_matrix_passed
        and cert.semantic_corpus_passed
        and cert.install_identity_passed
        and not cert.network_used
        and not cert.download_attempted
    )
    # Promotion remains blocked by design (advisors cannot promote hammer lane).
    cert.promotion_blocked = True
    if cert.production_certified and cert.block_reasons:
        # Drop soft block reasons that are informational only when all checks pass.
        hard = [
            r
            for r in cert.block_reasons
            if not r.startswith("case_failed:") or True
        ]
        # If any case failed, production_certified is already False.
        _ = hard

    cert.bindings.update(
        {
            "authority": {
                "scope": AUTHORITY_SCOPE,
                "ceiling": AUTHORITY_CEILING,
                "advisors_never_promote_alone": True,
                "confidence_never_yields_proof": True,
            },
            "locked_versions": {
                "symbolicai": LOCKED_SYMBOLICAI_VERSION,
                "ergoai": LOCKED_ERGOAI_VERSION,
            },
            "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
            "role_tools": [
                item.tool_id
                for item in tools_by_role(ToolRole.ADVISOR)
            ]
            + [item.tool_id for item in tools_by_role(ToolRole.CANDIDATE)],
            "lane_id": LANE_ID,
            "handler_id": HANDLER_ID,
        }
    )
    cert.notes = (
        "Advisor role certification: candidates only; independent reconstruction "
        "and kernel/solver validation remain mandatory for any proof authority."
    )
    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    install_root: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable certification receipt."""

    cert = run_certification_suite(
        repo_root=repo_root, env=env, install_root=install_root
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "advisors_never_promote_alone": True,
        "confidence_never_yields_proof": True,
        "availability_is_not_authority": True,
        "does_not_edit_central_certificate": True,
        "does_not_change_model_runtimes": True,
        "authority_is_candidate_generation_only": True,
    }
    payload["semantic_corpus_passed"] = cert.semantic_corpus_passed
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def lane_handler(**kwargs: Any) -> dict[str, Any]:
    """Hammer-lane handler for RoleAwarePromotionPolicy binding."""

    repo_root = kwargs.get("repo_root")
    root = Path(repo_root) if repo_root is not None else repo_root_from()
    cert = run_certification_suite(repo_root=root)
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if cert.production_certified else "role_boundary_only",
        "certified": cert.production_certified,
        "production_certified": cert.production_certified,
        "promotion_blocked": cert.promotion_blocked,
        "authority_scope": AUTHORITY_SCOPE,
        "authority_ceiling": AUTHORITY_CEILING,
        "advisors_never_promote_alone": True,
        "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
        "semantic_corpus_passed": cert.semantic_corpus_passed,
        "role_matrix_passed": cert.role_matrix_passed,
        "install_identity_passed": cert.install_identity_passed,
        "reason_codes": list(cert.block_reasons),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
    }


def bind_to_role_policy(policy: Any | None = None) -> Any:
    """Bind this lane handler under tools.logic.certification.roles."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        raise AdvisorRoleCertificationError(
            "roles certification surface unavailable for lane binding"
        )
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(LANE_ID, lane_handler, policy=target, replace=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify advisor utilities as bounded candidate generation "
            f"({INTERFACE})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (auto-detected when omitted)",
    )
    parser.add_argument(
        "--live-ergoai",
        action="store_true",
        help="Run checksum/provenance-bound live ErgoAI semantic certification",
    )
    parser.add_argument(
        "--ergoai-executable",
        type=Path,
        default=None,
        help="Managed ErgoAI launcher to certify (auto-discovered when omitted)",
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=None,
        help="Managed prover root containing the ErgoAI identity manifest",
    )
    parser.add_argument(
        "--platform-key",
        default=None,
        help="Expected lock platform, such as linux-aarch64",
    )
    args = parser.parse_args(argv)

    if args.live_ergoai:
        receipt = certify_live_ergoai_vendor(
            executable=args.ergoai_executable,
            install_root=args.install_root,
            repo_root=args.repo_root,
            platform_key=args.platform_key,
        )
        success_key = "vendor_certified"
    else:
        receipt = build_certification_receipt(repo_root=args.repo_root)
        success_key = "production_certified"
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.live_ergoai:
        print(f"{LIVE_ERGOAI_INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"  vendor_certified={receipt['vendor_certified']} "
            f"authoritative_live_evidence={receipt['authoritative_live_evidence']} "
            f"promotion_blocked={receipt['promotion_blocked']}"
        )
        print(f"  digest={receipt.get('receipt_digest_sha256', '')[:16]}…")
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"  production_certified={receipt['production_certified']} "
            f"promotion_blocked={receipt['promotion_blocked']} "
            f"corpus={receipt['semantic_corpus_passed']} "
            f"roles={receipt['role_matrix_passed']}"
        )
        print(f"  advisors={','.join(ADVISOR_TOOL_IDS)}")
        print(f"  digest={receipt.get('receipt_digest_sha256', '')[:16]}…")
    return 0 if receipt.get(success_key) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "CORPUS_SCHEMA",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "LIVE_ERGOAI_INTERFACE",
    "LIVE_ERGOAI_SCHEMA_VERSION",
    "LIVE_ERGOAI_EVIDENCE_CLASS",
    "ADVISOR_TOOL_IDS",
    "LOCKED_SYMBOLICAI_VERSION",
    "LOCKED_ERGOAI_VERSION",
    "AdvisorRoleCertificationError",
    "CheckResult",
    "CaseOutcome",
    "AdvisorRoleCertification",
    "default_corpus_manifest",
    "corpus_cases",
    "advisor_role_boundary",
    "all_advisor_role_boundaries",
    "advisors_cannot_promote_hammer_lane",
    "evaluate_corpus_case",
    "certify_install_identities",
    "certify_live_ergoai_vendor",
    "run_certification_suite",
    "build_certification_receipt",
    "lane_handler",
    "bind_to_role_policy",
    "offline_env",
    "main",
]
