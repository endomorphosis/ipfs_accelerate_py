"""Fail-closed PCPR-043 cross-repository compatibility evaluation.

One explicit supported combination binds Accelerate, Datasets, and Kit.
Incompatible combinations fail identically. This evaluator is not a
freeze, not a remint, not the signed PCPR-056 lock, and not a closed
PCPR release. It does not write DuckDB or Quack state.

Live claims require live evidence. Simulated results stay Simulated.
Missing supervisor promotion and CUDA re-qualification stay typed
unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.cross_repository_compatibility import (
    INCOMPATIBLE_CATEGORIES,
    INCOMPATIBLE_RECIPES,
    INTERFACE as FLOOR_INTERFACE,
    PINNED_041_VECTOR_DOCUMENT_CID,
    PINNED_042_NEGATIVE_DOCUMENT_CID,
    PINNED_CATALOG_CID,
    PINNED_COMPATIBILITY_DOCUMENT_CID,
    PINNED_VECTOR_CIDS,
    REQUIRED_REPOSITORIES,
    SCHEMA as FLOOR_SCHEMA,
    SUPPORTED_COMBINATION_ID,
    CrossRepositoryCompatibilityError,
    admit_combination,
    compatibility_document,
    compatibility_document_cid,
    evaluate_incompatible_combinations,
    refuse_compatibility_remint,
    supported_snapshot,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_AUTHORITIES,
    PCPR_002_TASK_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    catalog_cid,
    content_identity,
)
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)


COMPATIBILITY_INTERFACE: Final = "CrossRepositoryCompatibilityQualification@1"
COMPATIBILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "cross-repository-compatibility-qualification@1"
)
COMPATIBILITY_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "cross-repository-compatibility-qualification-verdict@1"
)

PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_017_TASK_ID: Final = "PCPR-017"
PCPR_027_TASK_ID: Final = "PCPR-027"
PCPR_039_TASK_ID: Final = "PCPR-039"
PCPR_040_GOAL_ID: Final = "PCPR-G500"
PCPR_041_GOAL_ID: Final = "PCPR-G510"
PCPR_042_GOAL_ID: Final = "PCPR-G520"
PCPR_043_GOAL_ID: Final = "PCPR-G520"
PCPR_056_TASK_ID: Final = "PCPR-056"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)

PCPR_017_VERDICT_CID: Final = (
    "baguqeeratvkodnxdxcudgddqaxoohvglwyq34grr7zupm6pbnwqk4zrp2ubq"
)
PCPR_027_VERDICT_CID: Final = (
    "baguqeera2yepzxux4cwbuzzk2t2oz4ju6n4hkwv4gqq2o6o6jsxua55loo4a"
)
PCPR_039_VERDICT_CID: Final = (
    "baguqeeraejqvwpevn5lawzy7pie7cz2atrp62ijbwj4dqx4fvqrxgbr32tla"
)
PCPR_040_VERDICT_CID: Final = (
    "baguqeeraxym6kuq5ljps7vwbvrkyobtfhrrpbquhbgrcny4viyfxlwversna"
)
PCPR_041_VERDICT_CID: Final = (
    "baguqeerafd3yeezarpuf6aicpdycn4sombbpcgfzoekttwytqjpw4pi7qtuq"
)
PCPR_042_VERDICT_CID: Final = (
    "baguqeera5ucb4kihzmf3gs5rnckjuf2khjq6a7vxogcsxmbetjg663yvlblq"
)

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)
PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)

CURRENT_HEAD_OUTER_COMMIT: Final = (
    "192f22a5dc7b4943d486d526a8fa5eb48c63e60b"
)
CURRENT_HEAD_OUTER_TREE: Final = "ad9bb7f17f809fedf07bbce8d439f5a48cec9e8a"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '5cd226debb8e6a526e916b9362031473ea5c66bf' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "a46795bc0cedb889ad63dadff4bc6e5ded6f2233"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = (
    "cd28ad6357441a1c2987bc3199e686ece84de125"
)
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = (
    "d7dd25b4bfe006cc69cc37962d6d002aed451d19"
)
CURRENT_HEAD_DATASETS_TREE: Final = "d45cac1a401ff15776be96d2dcbee20d8019c899"
CURRENT_HEAD_DATASETS_ORIGIN_MAIN: Final = (
    "f49afc579c22856849ca9f739435e5820003384f"
)
CURRENT_HEAD_KIT_COMMIT: Final = "4df10ebf91e29167ac03263bcbb30af8b69676a4"
CURRENT_HEAD_KIT_TREE: Final = "c9a9f3240f6390cb3ad260901a0ed214f42331fb"
CURRENT_HEAD_KIT_ORIGIN_MAIN: Final = (
    "b6c65ba732733d7e33852713ba18aa3b12235668"
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_cross_repository_compatibility.py",
)

FLOOR_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/cross_repository_compatibility.py"
)
DATASETS_BINDING_RELPATH: Final = (
    "external/ipfs_datasets/ipfs_datasets_py/assurance/"
    "cross_repository_compatibility.py"
)
KIT_BINDING_RELPATH: Final = (
    "external/ipfs_kit/ipfs_kit_py/assurance/cross_repository_compatibility.py"
)

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "compatibility_module",
        "one_supported_combination",
        "nine_incompatible_categories",
        "eighteen_incompatible_combinations_reject",
        "fail_identically_under_reorder",
        "catalog_identity_not_reminted",
        "vector_cids_not_reminted",
        "negative_document_not_reminted",
        "datasets_binding_does_not_remint",
        "kit_binding_does_not_remint",
        "ownership_boundaries_held",
        "freeze_not_claimed",
        "lock_not_claimed",
        "production_authorized_not_granted",
    }
)


class CrossRepositoryCompatibilityQualificationError(ValueError):
    """Malformed compatibility evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class CompatibilityProbe:
    """One measured or typed-unavailable compatibility observation."""

    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CompatibilityVerdict:
    """Fail-closed PCPR-043 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    catalog_cid: str
    vector_document_cid: str
    negative_document_cid: str
    compatibility_document_cid: str
    supported_combination_count: int
    incompatible_count: int
    categories_complete: bool
    datasets_binding_aligned: bool
    kit_binding_aligned: bool
    simulated_results_represented_as_live: bool
    live_supervisor_qualified: bool
    live_supervisor_evidence_kind: str
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    production_authorized: bool
    this_task_created_competing_authority: bool
    probes: tuple[CompatibilityProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "catalog_cid": self.catalog_cid,
            "vector_document_cid": self.vector_document_cid,
            "negative_document_cid": self.negative_document_cid,
            "compatibility_document_cid": self.compatibility_document_cid,
            "supported_combination_count": self.supported_combination_count,
            "incompatible_count": self.incompatible_count,
            "categories_complete": self.categories_complete,
            "datasets_binding_aligned": self.datasets_binding_aligned,
            "kit_binding_aligned": self.kit_binding_aligned,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_supervisor_qualified": self.live_supervisor_qualified,
            "live_supervisor_evidence_kind": self.live_supervisor_evidence_kind,
            "live_cuda_qualified": self.live_cuda_qualified,
            "live_cuda_evidence_kind": self.live_cuda_evidence_kind,
            "production_authorized": self.production_authorized,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CrossRepositoryCompatibilityQualificationError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CrossRepositoryCompatibilityQualificationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise CrossRepositoryCompatibilityQualificationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CrossRepositoryCompatibilityQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise CrossRepositoryCompatibilityQualificationError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> CompatibilityProbe:
    return CompatibilityProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=False,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _binding_aligned(source: str | None, relpath: str) -> CompatibilityProbe:
    if source is None:
        return _probe(
            "binding_source",
            present=None,
            evidence_kind="unavailable",
            reason=f"{relpath} is not present and is not recorded as empty.",
            details={"relpath": relpath},
        )
    sibling = False
    if relpath.endswith("ipfs_kit_py/assurance/cross_repository_compatibility.py"):
        sibling = "from ipfs_accelerate_py" in source or "from ipfs_datasets_py" in source
    if relpath.endswith(
        "ipfs_datasets_py/assurance/cross_repository_compatibility.py"
    ):
        sibling = "from ipfs_accelerate_py" in source or "from ipfs_kit_py" in source
    missing_categories = [
        category for category in INCOMPATIBLE_CATEGORIES if category not in source
    ]
    missing_repos = [
        repo for repo in REQUIRED_REPOSITORIES if repo not in source
    ]
    ok = (
        PINNED_COMPATIBILITY_DOCUMENT_CID in source
        and PINNED_CATALOG_CID in source
        and PINNED_041_VECTOR_DOCUMENT_CID in source
        and PINNED_042_NEGATIVE_DOCUMENT_CID in source
        and SUPPORTED_COMBINATION_ID in source
        and "refuse_compatibility_remint" in source
        and PCPR_056_TASK_ID in source
        and not missing_categories
        and not missing_repos
        and not sibling
        and "JavaScript" in source
        and "3.12" in source
    )
    return _probe(
        "binding",
        present=ok,
        evidence_kind="measured",
        reason=(
            "Binding pins the compatibility document CID and refuses remint."
            if ok
            else "Binding is missing identities, remints, or imports a sibling."
        ),
        details={
            "relpath": relpath,
            "missing_categories": missing_categories,
            "missing_repositories": missing_repos,
            "refuse_compatibility_remint": "refuse_compatibility_remint" in source,
            "sibling_import": sibling,
        },
    )


def current_head_compatibility_probes(
    *,
    accelerate_root: Path | None = None,
    portfolio_root: Path | None = None,
) -> tuple[CompatibilityProbe, ...]:
    """Measured current-tree probes. Missing trees stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[CompatibilityProbe] = []
    if root is None or not root.is_dir():
        return (
            _probe(
                "accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Accelerate source tree is not present and is not recorded as empty."
                ),
            ),
        )

    floor_src = _read_source(root, FLOOR_MODULE_RELPATH)
    module_ok = bool(floor_src) and all(
        category in (floor_src or "") for category in INCOMPATIBLE_CATEGORIES
    ) and FLOOR_SCHEMA in (floor_src or "") and PINNED_CATALOG_CID in (
        floor_src or ""
    ) and SUPPORTED_COMBINATION_ID in (floor_src or "")
    probes.append(
        _probe(
            "compatibility_module",
            present=module_ok,
            evidence_kind="measured" if floor_src is not None else "unavailable",
            reason=(
                "Cross-repository compatibility module declares one supported combination."
                if module_ok
                else "Cross-repository compatibility module is missing or incomplete."
            ),
            details={"relpath": FLOOR_MODULE_RELPATH},
        )
    )

    supported_ok = False
    categories_ok = False
    incompatibles_ok = False
    reorder_ok = False
    details: dict[str, Any] = {}
    try:
        admit_combination(supported_snapshot())
        supported_ok = True
        rows = evaluate_incompatible_combinations()
        observed = {item["category"] for item in rows}
        categories_ok = observed == set(INCOMPATIBLE_CATEGORIES)
        incompatibles_ok = (
            len(rows) == len(INCOMPATIBLE_RECIPES)
            and all(item["rejected"] is True for item in rows)
        )
        reorder_ok = any(item["identical_reversed"] for item in rows)
        details = {
            "count": len(rows),
            "categories": sorted(observed),
            "supported_combination_id": SUPPORTED_COMBINATION_ID,
        }
        supported_reason = (
            "The explicit Accelerate+Datasets+Kit Python 3.12 combination is admitted."
            if supported_ok
            else "The supported combination was not admitted."
        )
        incompatibles_reason = (
            "Eighteen incompatible recipes reject with the declared kinds."
            if incompatibles_ok
            else "An incompatible combination was admitted or drifted."
        )
        categories_reason = (
            "Missing-repository, reminted, sibling-import, cross-version, mixed, "
            "partial, unsupported-Python, authority-boundary, and claimed-early "
            "categories are present."
            if categories_ok
            else "A required incompatible category is missing."
        )
        reorder_reason = (
            "Reminted-catalog rejection is identical under key reordering."
            if reorder_ok
            else "Reordered remint rejection drifted."
        )
    except CrossRepositoryCompatibilityError as exc:
        supported_ok = False
        categories_ok = False
        incompatibles_ok = False
        reorder_ok = False
        supported_reason = str(exc)
        incompatibles_reason = str(exc)
        categories_reason = str(exc)
        reorder_reason = str(exc)
    probes.append(
        _probe(
            "one_supported_combination",
            present=supported_ok,
            evidence_kind="measured",
            reason=supported_reason,
            details={"supported_combination_id": SUPPORTED_COMBINATION_ID},
        )
    )
    probes.append(
        _probe(
            "nine_incompatible_categories",
            present=categories_ok,
            evidence_kind="measured",
            reason=categories_reason,
            details=details,
        )
    )
    probes.append(
        _probe(
            "eighteen_incompatible_combinations_reject",
            present=incompatibles_ok,
            evidence_kind="measured",
            reason=incompatibles_reason,
            details={"count": len(INCOMPATIBLE_RECIPES)},
        )
    )
    probes.append(
        _probe(
            "fail_identically_under_reorder",
            present=reorder_ok,
            evidence_kind="measured",
            reason=reorder_reason,
        )
    )

    catalog_ok = catalog_cid() == PINNED_CATALOG_CID
    probes.append(
        _probe(
            "catalog_identity_not_reminted",
            present=catalog_ok,
            evidence_kind="measured",
            reason=(
                "PCPR-040 catalog CID is reused and is not reminted."
                if catalog_ok
                else "The shared-contract catalog CID was reminted."
            ),
            details={"catalog_cid": PINNED_CATALOG_CID},
        )
    )

    vectors_ok = False
    vectors_reason = "Vector CIDs reminted."
    try:
        snapshot = supported_snapshot()
        vectors_ok = snapshot["vector_cids"] == dict(PINNED_VECTOR_CIDS)
        document_cid = compatibility_document_cid()
        vectors_ok = vectors_ok and document_cid == PINNED_COMPATIBILITY_DOCUMENT_CID
        vectors_reason = (
            "PCPR-041 vector CIDs are reused and are not reminted."
            if vectors_ok
            else "A PCPR-041 vector CID or the PCPR-043 document CID reminted."
        )
    except CrossRepositoryCompatibilityError as exc:
        vectors_ok = False
        vectors_reason = str(exc)
    probes.append(
        _probe(
            "vector_cids_not_reminted",
            present=vectors_ok,
            evidence_kind="measured",
            reason=vectors_reason,
            details={
                "compatibility_document_cid": PINNED_COMPATIBILITY_DOCUMENT_CID,
                "vector_document_cid": PINNED_041_VECTOR_DOCUMENT_CID,
            },
        )
    )

    negative_ok = PINNED_042_NEGATIVE_DOCUMENT_CID == (
        supported_snapshot()["negative_document_cid"]
    )
    probes.append(
        _probe(
            "negative_document_not_reminted",
            present=negative_ok,
            evidence_kind="measured",
            reason=(
                "PCPR-042 negative document CID is reused and is not reminted."
                if negative_ok
                else "The PCPR-042 negative document CID was reminted."
            ),
            details={"negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID},
        )
    )

    portfolio = portfolio_root or discover_portfolio_root()
    datasets_src = None
    kit_src = None
    if portfolio is None:
        probes.append(
            _probe(
                "datasets_binding_does_not_remint",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Portfolio tree is not present; Datasets compatibility binding stays typed unavailable."
                ),
            )
        )
        probes.append(
            _probe(
                "kit_binding_does_not_remint",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Portfolio tree is not present; Kit compatibility binding stays typed unavailable."
                ),
            )
        )
    else:
        datasets_src = _read_source(portfolio, DATASETS_BINDING_RELPATH)
        datasets_probe = _binding_aligned(datasets_src, DATASETS_BINDING_RELPATH)
        probes.append(
            CompatibilityProbe(
                probe_id="datasets_binding_does_not_remint",
                present=datasets_probe.present,
                evidence_kind=datasets_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Datasets binding pins the compatibility document CID and refuses remint."
                    if datasets_probe.present is True
                    else datasets_probe.reason
                ),
                details=datasets_probe.details,
            )
        )
        kit_src = _read_source(portfolio, KIT_BINDING_RELPATH)
        kit_probe = _binding_aligned(kit_src, KIT_BINDING_RELPATH)
        probes.append(
            CompatibilityProbe(
                probe_id="kit_binding_does_not_remint",
                present=kit_probe.present,
                evidence_kind=kit_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Kit binding pins the compatibility document CID and refuses remint."
                    if kit_probe.present is True
                    else kit_probe.reason
                ),
                details=kit_probe.details,
            )
        )

    ownership_ok = (
        CONTRACT_AUTHORITIES.get("SupervisorContextPack") == "ipfs_datasets_py"
        and CONTRACT_AUTHORITIES.get("SemanticArtifactIdentity") == "ipfs_datasets_py"
        and CONTRACT_AUTHORITIES.get("DurableArtifactReceipt") == "ipfs_kit_py"
        and bool(datasets_src)
        and "SupervisorContextPack" in (datasets_src or "")
        and "SemanticArtifactIdentity" in (datasets_src or "")
        and bool(kit_src)
        and "DurableArtifactReceipt" in (kit_src or "")
        and "semantic_authority" in (kit_src or "")
        and "task_completion_authority" in (kit_src or "")
    )
    probes.append(
        _probe(
            "ownership_boundaries_held",
            present=ownership_ok,
            evidence_kind="measured" if datasets_src is not None and kit_src is not None else "unavailable",
            reason=(
                "Datasets owns ContextPack and semantic identity; Kit owns durable receipts and does not claim semantic, proof, or task authority."
                if ownership_ok
                else "Ownership boundaries drifted or a sibling claimed the wrong authority."
            ),
            details={
                "datasets_owned": ["SupervisorContextPack", "SemanticArtifactIdentity"],
                "kit_owned": ["DurableArtifactReceipt"],
            },
        )
    )

    probes.append(
        _probe(
            "freeze_not_claimed",
            present=True,
            evidence_kind="measured",
            reason=(
                "PCPR-043 publishes cross-repository compatibility checks. Freeze remains PCPR-002 and is not claimed."
            ),
            details={"freeze_task": PCPR_002_TASK_ID, "frozen": False},
        )
    )
    probes.append(
        _probe(
            "lock_not_claimed",
            present=True,
            evidence_kind="measured",
            reason=(
                "The signed portfolio compatibility lock remains PCPR-056 and is not claimed here."
            ),
            details={"deferred_task_id": PCPR_056_TASK_ID, "lock": False},
        )
    )
    probes.append(
        _probe(
            "production_authorized_not_granted",
            present=True,
            evidence_kind="measured",
            reason=(
                "Cross-repository compatibility checks do not grant production_authorized or a freeze."
            ),
        )
    )
    probes.append(
        _probe(
            "live_supervisor_qualification",
            present=None,
            evidence_kind="unavailable",
            reason=(
                "PCPR-001 live qualification remains typed unavailable and is not "
                "recorded as False or passing."
            ),
            details={"deferred_task_id": PCPR_001_TASK_ID},
        )
    )
    probes.append(
        _probe(
            "live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            reason=(
                "This task does not re-qualify live CUDA. Missing CUDA re-qualification "
                "stays typed unavailable and is not recorded as False or passing."
            ),
            details={"deferred_task_id": "PCPR-038"},
        )
    )
    return tuple(probes)


def qualify_cross_repository_compatibility(
    *,
    probes: Sequence[CompatibilityProbe],
    duckdb_or_quack_state_written: bool = False,
    simulated_results_represented_as_live: bool = False,
    live_supervisor_qualified: bool = False,
    live_cuda_qualified: bool = False,
    production_authorized: bool = False,
    contracts_frozen: bool = False,
    this_task_created_competing_authority: bool = False,
) -> CompatibilityVerdict:
    """Fail-closed compatibility verdict. Never a closed release."""

    if duckdb_or_quack_state_written:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise CrossRepositoryCompatibilityQualificationError(
            "simulated compatibility results cannot be represented as live"
        )
    if live_supervisor_qualified or live_cuda_qualified:
        raise CrossRepositoryCompatibilityQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if production_authorized:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot grant production_authorized"
        )
    if contracts_frozen:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot freeze contracts"
        )
    if this_task_created_competing_authority:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot mint a competing authority"
        )

    normalized: list[CompatibilityProbe] = []
    for item in probes:
        if not isinstance(item, CompatibilityProbe):
            raise CrossRepositoryCompatibilityQualificationError(
                "probes must be CompatibilityProbe values"
            )
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise CrossRepositoryCompatibilityQualificationError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise CrossRepositoryCompatibilityQualificationError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise CrossRepositoryCompatibilityQualificationError(
                f"{item.probe_id} evidence_kind {item.evidence_kind} is not admitted here"
            )
        normalized.append(item)

    by_id = {item.probe_id: item for item in normalized}
    blockers: list[str] = []

    def _require(probe_id: str, blocker: str) -> bool:
        item = by_id.get(probe_id)
        ok = bool(item and item.present is True)
        if not ok:
            blockers.append(blocker)
        return ok

    categories_ok = _require(
        "nine_incompatible_categories", "incompatible_categories_incomplete"
    )
    datasets_ok = _require(
        "datasets_binding_does_not_remint", "datasets_binding_remints"
    )
    kit_ok = _require("kit_binding_does_not_remint", "kit_binding_remints")
    for probe_id, blocker in (
        ("compatibility_module", "compatibility_module_missing"),
        ("one_supported_combination", "supported_combination_missing"),
        ("eighteen_incompatible_combinations_reject", "incompatible_combination_admitted"),
        ("fail_identically_under_reorder", "reorder_reject_drifted"),
        ("catalog_identity_not_reminted", "catalog_reminted"),
        ("vector_cids_not_reminted", "vector_cid_reminted"),
        ("negative_document_not_reminted", "negative_document_reminted"),
        ("ownership_boundaries_held", "ownership_boundary_drifted"),
        ("freeze_not_claimed", "freeze_claimed"),
        ("lock_not_claimed", "lock_claimed_early"),
        ("production_authorized_not_granted", "production_authorized_claimed"),
    ):
        _require(probe_id, blocker)

    unavailable_count = sum(
        1 for item in normalized if item.evidence_kind == "unavailable"
    )
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise CrossRepositoryCompatibilityQualificationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not mint a closed release outcome"
        )

    document_cid = PINNED_COMPATIBILITY_DOCUMENT_CID
    try:
        document_cid = compatibility_document_cid()
    except CrossRepositoryCompatibilityError:
        blockers.append("compatibility_document_reminted")
        promotion_status = "typed_blocked"

    payload = {
        "schema": COMPATIBILITY_VERDICT_SCHEMA,
        "interface": COMPATIBILITY_INTERFACE,
        "task_id": PCPR_043_TASK_ID,
        "goal_id": PCPR_043_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": PINNED_041_VECTOR_DOCUMENT_CID,
        "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
        "compatibility_document_cid": document_cid,
        "supported_combination_count": 1,
        "incompatible_count": len(INCOMPATIBLE_RECIPES),
        "categories_complete": categories_ok,
        "datasets_binding_aligned": datasets_ok,
        "kit_binding_aligned": kit_ok,
        "simulated_results_represented_as_live": False,
        "live_supervisor_qualified": False,
        "live_supervisor_evidence_kind": "unavailable",
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "production_authorized": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return CompatibilityVerdict(
        schema=COMPATIBILITY_VERDICT_SCHEMA,
        interface=COMPATIBILITY_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        catalog_cid=PINNED_CATALOG_CID,
        vector_document_cid=PINNED_041_VECTOR_DOCUMENT_CID,
        negative_document_cid=PINNED_042_NEGATIVE_DOCUMENT_CID,
        compatibility_document_cid=document_cid,
        supported_combination_count=1,
        incompatible_count=len(INCOMPATIBLE_RECIPES),
        categories_complete=categories_ok,
        datasets_binding_aligned=datasets_ok,
        kit_binding_aligned=kit_ok,
        simulated_results_represented_as_live=False,
        live_supervisor_qualified=False,
        live_supervisor_evidence_kind="unavailable",
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        production_authorized=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_compatibility() -> CompatibilityVerdict:
    """Ordinary current-head PCPR-043 evaluation: honest R&D compatibility."""

    return qualify_cross_repository_compatibility(
        probes=current_head_compatibility_probes()
    )


# Pinned after the current-head evaluator is measured. Drift means the
# outer receipt must be regenerated from this evaluator.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeralan5iwoqbexusnfqfywx3zukpimdrv6usiu2flqznbj5s7yculsa"
)


def pcpr_043_receipt_promotion(verdict: CompatibilityVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CrossRepositoryCompatibilityQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise CrossRepositoryCompatibilityQualificationError(
            "promotion_status is not an admitted PCPR-043 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot promote the supervisor"
        )
    if verdict.contracts_frozen:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot freeze contracts"
        )
    if verdict.live_supervisor_qualified or verdict.live_cuda_qualified:
        raise CrossRepositoryCompatibilityQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if not verdict.categories_complete:
        raise CrossRepositoryCompatibilityQualificationError(
            "incompatible categories must be complete"
        )
    return {
        "schema": COMPATIBILITY_VERDICT_SCHEMA,
        "interface": COMPATIBILITY_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": verdict.catalog_cid,
        "vector_document_cid": verdict.vector_document_cid,
        "negative_document_cid": verdict.negative_document_cid,
        "compatibility_document_cid": verdict.compatibility_document_cid,
        "supported_combination_count": verdict.supported_combination_count,
        "incompatible_count": verdict.incompatible_count,
        "categories_complete": True,
        "datasets_binding_aligned": verdict.datasets_binding_aligned,
        "kit_binding_aligned": verdict.kit_binding_aligned,
        "simulated_results_represented_as_live": False,
        "live_supervisor_qualified": False,
        "live_supervisor_evidence_kind": "unavailable",
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "production_authorized": False,
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_043_receipt_promotion() -> dict[str, Any]:
    return pcpr_043_receipt_promotion(qualify_current_head_compatibility())


def pcpr_043_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "identities_not_reminted": True,
        "catalog_not_reminted": True,
        "contracts_not_frozen": True,
        "lock_not_claimed": True,
        "live_supervisor_not_claimed": True,
        "live_cuda_not_reclaimed": True,
        "production_authorized_not_granted": True,
        "unsupported_combinations_not_admitted": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_043_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_compatibility()
    promotion = pcpr_043_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    document = compatibility_document()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_042_TASK_ID,
            "goal_id": PCPR_042_GOAL_ID,
            "promotion_status": "rnd_non_promoted",
            "negative_cross_language_vector_verdict_cid": PCPR_042_VERDICT_CID,
            "canonical_byte_cid_vector_verdict_cid": PCPR_041_VERDICT_CID,
            "shared_contract_verdict_cid": PCPR_040_VERDICT_CID,
            "datasets_solver_task_id": PCPR_017_TASK_ID,
            "datasets_solver_verdict_cid": PCPR_017_VERDICT_CID,
            "kit_support_matrix_task_id": PCPR_027_TASK_ID,
            "kit_support_matrix_verdict_cid": PCPR_027_VERDICT_CID,
            "model_provider_task_id": PCPR_039_TASK_ID,
            "model_provider_verdict_cid": PCPR_039_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": (
                qualification.promotion_status
            ),
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-042 remains rnd_non_promoted R&D. This task publishes "
                "cross-repository compatibility checks on that catalog. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "cross_repository_compatibility": {
            "schema": FLOOR_SCHEMA,
            "interface": FLOOR_INTERFACE,
            "floor_task_id": PCPR_043_TASK_ID,
            "floor_goal_id": PCPR_043_GOAL_ID,
            "catalog_cid": PINNED_CATALOG_CID,
            "vector_document_cid": PINNED_041_VECTOR_DOCUMENT_CID,
            "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
            "compatibility_document_cid": PINNED_COMPATIBILITY_DOCUMENT_CID,
            "frozen": False,
            "freeze_task": PCPR_002_TASK_ID,
            "normative_task": PCPR_040_TASK_ID,
            "positive_vector_task": PCPR_041_TASK_ID,
            "negative_vector_task": PCPR_042_TASK_ID,
            "lock_task": PCPR_056_TASK_ID,
            "lock": False,
            "languages": ["Python", "JavaScript"],
            "typescript_compiler": "unavailable",
            "python": "3.12",
            "supported_combination_id": SUPPORTED_COMBINATION_ID,
            "supported_combination_count": 1,
            "incompatible_categories": list(INCOMPATIBLE_CATEGORIES),
            "incompatible_count": document["incompatible_count"],
            "repositories": list(REQUIRED_REPOSITORIES),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_043_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_043_current_tree_binding(
    *,
    outer_commit: str,
    outer_tree: str,
    outer_subject: str,
    origin_main: str,
    origin_main_is_ancestor: bool,
    accelerator_pre_change_commit: str,
    accelerator_pre_change_tree: str,
    accelerator_gitlink: str,
    accelerator_origin_main: str,
    accelerator_origin_main_is_ancestor: bool,
    datasets_commit: str,
    datasets_tree: str,
    datasets_gitlink: str,
    datasets_origin_main: str,
    datasets_origin_main_is_ancestor: bool,
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
    kit_origin_main: str,
    kit_origin_main_is_ancestor: bool,
) -> dict[str, Any]:
    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(
        accelerator_pre_change_commit, "accelerator_pre_change_commit"
    )
    accel_tree = _git_object_id(
        accelerator_pre_change_tree, "accelerator_pre_change_tree"
    )
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(
        accelerator_origin_main, "accelerator_origin_main"
    )
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise CrossRepositoryCompatibilityQualificationError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    datasets_origin = _git_object_id(
        datasets_origin_main, "datasets_origin_main"
    )
    _require_ancestor(
        datasets_origin_main_is_ancestor, "datasets_origin_main_is_ancestor"
    )
    if datasets != datasets_link:
        raise CrossRepositoryCompatibilityQualificationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    kit_origin = _git_object_id(kit_origin_main, "kit_origin_main")
    _require_ancestor(kit_origin_main_is_ancestor, "kit_origin_main_is_ancestor")
    if kit != kit_link:
        raise CrossRepositoryCompatibilityQualificationError(
            "kit_commit must equal kit_gitlink"
        )
    _reject_closed_release_value(subject, "outer_subject")
    return {
        "outer_repository": "endomorphosis/lift_coding",
        "owning_repository_for_receipts": "ipfs_accelerate_py",
        "outer_commit": outer,
        "outer_tree": tree,
        "outer_subject": subject,
        "origin_main": origin,
        "origin_main_is_ancestor": True,
        "accelerator_pre_change_commit": accel,
        "accelerator_pre_change_tree": accel_tree,
        "accelerator_gitlink": accel_link,
        "accelerator_origin_main": accel_origin,
        "accelerator_origin_main_is_ancestor": True,
        "accelerator_post_change_commit": "pending nested commit after admission",
        "accelerator_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "datasets_origin_main": datasets_origin,
        "datasets_origin_main_is_ancestor": True,
        "datasets_post_change_commit": "pending nested commit after admission",
        "datasets_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "kit_origin_main": kit_origin,
        "kit_origin_main_is_ancestor": True,
        "kit_post_change_commit": "pending nested commit after admission",
        "kit_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "evidence_kind": "measured",
    }


def current_head_pcpr_043_current_tree_binding() -> dict[str, Any]:
    return pcpr_043_current_tree_binding(
        outer_commit=CURRENT_HEAD_OUTER_COMMIT,
        outer_tree=CURRENT_HEAD_OUTER_TREE,
        outer_subject=CURRENT_HEAD_OUTER_SUBJECT,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        accelerator_pre_change_commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_pre_change_tree=CURRENT_HEAD_ACCELERATOR_TREE,
        accelerator_gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
        accelerator_origin_main_is_ancestor=True,
        datasets_commit=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_tree=CURRENT_HEAD_DATASETS_TREE,
        datasets_gitlink=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_origin_main=CURRENT_HEAD_DATASETS_ORIGIN_MAIN,
        datasets_origin_main_is_ancestor=True,
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
        kit_origin_main=CURRENT_HEAD_KIT_ORIGIN_MAIN,
        kit_origin_main_is_ancestor=True,
    )


def validate_pcpr_043_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-043 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise CrossRepositoryCompatibilityQualificationError(
            "outer receipt must be a mapping"
        )
    if payload.get("task_id") != PCPR_043_TASK_ID:
        raise CrossRepositoryCompatibilityQualificationError(
            "outer receipt task_id must be PCPR-043"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict must be a mapping"
        )
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot freeze contracts"
        )
    if verdict_section.get("live_supervisor_qualified") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "this task cannot claim live supervisor qualification"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise CrossRepositoryCompatibilityQualificationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-043 status"
        )
    if promotion_status == "supervisor_promoted":
        raise CrossRepositoryCompatibilityQualificationError(
            "cross-repository compatibility cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise CrossRepositoryCompatibilityQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise CrossRepositoryCompatibilityQualificationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_043_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise CrossRepositoryCompatibilityQualificationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise CrossRepositoryCompatibilityQualificationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_043_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "COMPATIBILITY_INTERFACE",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "PCPR_043_GOAL_ID",
    "PCPR_043_TASK_ID",
    "CrossRepositoryCompatibilityQualificationError",
    "current_head_compatibility_probes",
    "current_head_pcpr_043_current_tree_binding",
    "current_head_pcpr_043_receipt_promotion",
    "current_head_pcpr_043_receipt_sections",
    "qualify_cross_repository_compatibility",
    "qualify_current_head_compatibility",
    "refuse_compatibility_remint",
    "validate_pcpr_043_outer_receipt",
)
