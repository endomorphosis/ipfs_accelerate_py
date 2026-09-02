"""Fail-closed PCPR-042 negative and cross-language vector evaluation.

Invalid, stale, unknown, out-of-bound, reordered, reminted, and
cross-version inputs fail identically. Python and JavaScript mint the
same CIDs for PCPR-041 positive fixtures. TypeScript compilation stays
typed unavailable. This evaluator is not a freeze, not a remint, and not
a closed PCPR release. It does not write DuckDB or Quack state.

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

from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    PINNED_CONTRACT_VECTORS,
    PINNED_NFC_CID,
)
from ipfs_accelerate_py.assurance.negative_cross_language_vectors import (
    INTERFACE as FLOOR_INTERFACE,
    JAVASCRIPT_MODULE_RELPATH,
    NEGATIVE_CATEGORIES,
    NEGATIVE_RECIPES,
    PINNED_CATALOG_CID,
    PINNED_VECTOR_DOCUMENT_CID,
    SCHEMA as FLOOR_SCHEMA,
    SUPPORTED_LANGUAGES,
    TYPESCRIPT_SOURCE_RELPATH,
    NegativeCrossLanguageVectorError,
    cross_language_cid_vectors,
    evaluate_negative_vectors,
    javascript_agrees_with_python,
    negative_cross_language_vector_document,
    refuse_negative_remint,
    run_javascript_vectors,
    vector_document_cid,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
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


VECTOR_INTERFACE: Final = "NegativeCrossLanguageVectorQualification@1"
VECTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "negative-cross-language-vector-qualification@1"
)
VECTOR_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "negative-cross-language-vector-qualification-verdict@1"
)

PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_017_TASK_ID: Final = "PCPR-017"
PCPR_027_TASK_ID: Final = "PCPR-027"
PCPR_039_TASK_ID: Final = "PCPR-039"
PCPR_040_GOAL_ID: Final = "PCPR-G500"
PCPR_041_GOAL_ID: Final = "PCPR-G510"
PCPR_042_GOAL_ID: Final = "PCPR-G520"
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
    "170b5e1691e13efe9e9f1d774fc0e07053f983cb"
)
CURRENT_HEAD_OUTER_TREE: Final = "c9c29d54c183f74efd75e8999ad41db01e821ff4"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '16cb711ebee57481f20d77894319cc3e5099e826' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "ca2c7a7a501e600b4cf251a5f1ab80860142d983"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = (
    "b52bbe311c6274665fb1c8c80398b03c60287cce"
)
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = (
    "39ac4a292c5ceb6bac88ae69bbd2f126c2fd29f8"
)
CURRENT_HEAD_DATASETS_TREE: Final = "78b8720612d6955ab4d284879a537ef4c6cbef44"
CURRENT_HEAD_DATASETS_ORIGIN_MAIN: Final = (
    "f49afc579c22856849ca9f739435e5820003384f"
)
CURRENT_HEAD_KIT_COMMIT: Final = "b0749b558c42b336b2b194b9101ba6e41c148df0"
CURRENT_HEAD_KIT_TREE: Final = "2c12a8361c04d872a697e2a2e7a75dd70df18d01"
CURRENT_HEAD_KIT_ORIGIN_MAIN: Final = (
    "b6c65ba732733d7e33852713ba18aa3b12235668"
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_negative_cross_language_vectors.py",
)

FLOOR_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/negative_cross_language_vectors.py"
)
DATASETS_BINDING_RELPATH: Final = (
    "external/ipfs_datasets/ipfs_datasets_py/assurance/"
    "negative_cross_language_vectors.py"
)
KIT_BINDING_RELPATH: Final = (
    "external/ipfs_kit/ipfs_kit_py/assurance/negative_cross_language_vectors.py"
)
NODE_PATH: Final = "/usr/bin/node"
TSC_PATH: Final = "/usr/bin/tsc"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "negative_vector_module",
        "seven_negative_categories",
        "eighteen_negative_vectors_reject",
        "fail_identically_under_reorder",
        "catalog_identity_not_reminted",
        "positive_cid_vectors_not_reminted",
        "javascript_encoder_agrees",
        "typescript_source_present",
        "datasets_binding_does_not_remint",
        "kit_binding_does_not_remint",
        "freeze_not_claimed",
        "cross_repository_compatibility_deferred",
        "production_authorized_not_granted",
    }
)


class NegativeCrossLanguageVectorQualificationError(ValueError):
    """Malformed vector evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class VectorProbe:
    """One measured or typed-unavailable vector observation."""

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
class VectorVerdict:
    """Fail-closed PCPR-042 decision. Never a closed release."""

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
    negative_count: int
    cross_language_count: int
    categories_complete: bool
    javascript_agreed: bool
    datasets_binding_aligned: bool
    kit_binding_aligned: bool
    simulated_results_represented_as_live: bool
    live_supervisor_qualified: bool
    live_supervisor_evidence_kind: str
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    production_authorized: bool
    this_task_created_competing_authority: bool
    probes: tuple[VectorProbe, ...]
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
            "negative_count": self.negative_count,
            "cross_language_count": self.cross_language_count,
            "categories_complete": self.categories_complete,
            "javascript_agreed": self.javascript_agreed,
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
        raise NegativeCrossLanguageVectorQualificationError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise NegativeCrossLanguageVectorQualificationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise NegativeCrossLanguageVectorQualificationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise NegativeCrossLanguageVectorQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise NegativeCrossLanguageVectorQualificationError(f"{name} must be true")


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
) -> VectorProbe:
    return VectorProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=False,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _binding_aligned(source: str | None, relpath: str) -> VectorProbe:
    if source is None:
        return _probe(
            "binding_source",
            present=None,
            evidence_kind="unavailable",
            reason=f"{relpath} is not present and is not recorded as empty.",
            details={"relpath": relpath},
        )
    sibling = "from ipfs_accelerate_py" in source or "from ipfs_datasets_py" in source
    if relpath.endswith("ipfs_kit_py/assurance/negative_cross_language_vectors.py"):
        sibling = "from ipfs_accelerate_py" in source or "from ipfs_datasets_py" in source
    if relpath.endswith(
        "ipfs_datasets_py/assurance/negative_cross_language_vectors.py"
    ):
        sibling = "from ipfs_accelerate_py" in source or "from ipfs_kit_py" in source
    missing_categories = [
        category for category in NEGATIVE_CATEGORIES if category not in source
    ]
    ok = (
        PINNED_VECTOR_DOCUMENT_CID in source
        and PINNED_CATALOG_CID in source
        and "refuse_negative_remint" in source
        and not missing_categories
        and not sibling
        and "JavaScript" in source
    )
    return _probe(
        "binding",
        present=ok,
        evidence_kind="measured",
        reason=(
            "Binding pins the negative/cross-language document CID and refuses remint."
            if ok
            else "Binding is missing identities, remints, or imports a sibling."
        ),
        details={
            "relpath": relpath,
            "missing_categories": missing_categories,
            "refuse_negative_remint": "refuse_negative_remint" in source,
            "sibling_import": sibling,
        },
    )


def current_head_vector_probes(
    *,
    accelerate_root: Path | None = None,
    portfolio_root: Path | None = None,
) -> tuple[VectorProbe, ...]:
    """Measured current-tree probes. Missing trees stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[VectorProbe] = []
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
        category in (floor_src or "") for category in NEGATIVE_CATEGORIES
    ) and FLOOR_SCHEMA in (floor_src or "") and PINNED_CATALOG_CID in (
        floor_src or ""
    )
    probes.append(
        _probe(
            "negative_vector_module",
            present=module_ok,
            evidence_kind="measured" if floor_src is not None else "unavailable",
            reason=(
                "Negative and cross-language vector module declares seven categories."
                if module_ok
                else "Negative and cross-language vector module is missing or incomplete."
            ),
            details={"relpath": FLOOR_MODULE_RELPATH},
        )
    )

    negatives_ok = False
    categories_ok = False
    reorder_ok = False
    details: dict[str, Any] = {}
    try:
        negatives = evaluate_negative_vectors()
        observed = {item["category"] for item in negatives}
        categories_ok = observed == set(NEGATIVE_CATEGORIES)
        negatives_ok = (
            len(negatives) == len(NEGATIVE_RECIPES)
            and all(item["rejected"] is True for item in negatives)
        )
        reorder_ok = any(item["identical_reversed"] for item in negatives)
        details = {
            "count": len(negatives),
            "categories": sorted(observed),
        }
        negatives_reason = (
            "Eighteen negative recipes reject with the declared kinds."
            if negatives_ok
            else "A negative recipe was admitted or drifted."
        )
        categories_reason = (
            "Invalid, stale, unknown, out-of-bound, reordered, reminted, and "
            "cross-version categories are present."
            if categories_ok
            else "A required negative category is missing."
        )
        reorder_reason = (
            "Unknown-field rejection is identical under key reordering."
            if reorder_ok
            else "Reordered unknown-field rejection drifted."
        )
    except NegativeCrossLanguageVectorError as exc:
        negatives_ok = False
        categories_ok = False
        reorder_ok = False
        negatives_reason = str(exc)
        categories_reason = str(exc)
        reorder_reason = str(exc)
    probes.append(
        _probe(
            "seven_negative_categories",
            present=categories_ok,
            evidence_kind="measured",
            reason=categories_reason,
            details=details,
        )
    )
    probes.append(
        _probe(
            "eighteen_negative_vectors_reject",
            present=negatives_ok,
            evidence_kind="measured",
            reason=negatives_reason,
            details={"count": len(NEGATIVE_RECIPES)},
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
    catalog_reason = (
        "PCPR-040 catalog CID is reused and is not reminted."
        if catalog_ok
        else "The shared-contract catalog CID was reminted."
    )
    probes.append(
        _probe(
            "catalog_identity_not_reminted",
            present=catalog_ok,
            evidence_kind="measured",
            reason=catalog_reason,
            details={"catalog_cid": PINNED_CATALOG_CID},
        )
    )

    positive_ok = False
    positive_reason = "Positive CID vectors reminted."
    try:
        rows = cross_language_cid_vectors()
        by_id = {item["id"]: item for item in rows}
        positive_ok = (
            by_id["python_js.intent"]["cid"]
            == PINNED_CONTRACT_VECTORS["SupervisorObjectiveIntent"]["cid"]
            and by_id["python_js.nfc"]["cid"] == PINNED_NFC_CID
            and by_id["python_js.key_order"]["cid"]
            == by_id["python_js.intent"]["cid"]
            and vector_document_cid() == PINNED_VECTOR_DOCUMENT_CID
        )
        positive_reason = (
            "PCPR-041 positive CIDs are reused and are not reminted."
            if positive_ok
            else "A PCPR-041 positive CID or the PCPR-042 document CID reminted."
        )
    except NegativeCrossLanguageVectorError as exc:
        positive_ok = False
        positive_reason = str(exc)
    probes.append(
        _probe(
            "positive_cid_vectors_not_reminted",
            present=positive_ok,
            evidence_kind="measured",
            reason=positive_reason,
            details={"vector_document_cid": PINNED_VECTOR_DOCUMENT_CID},
        )
    )

    js_src = _read_source(root, JAVASCRIPT_MODULE_RELPATH)
    node = Path(NODE_PATH)
    if js_src is None:
        probes.append(
            _probe(
                "javascript_encoder_agrees",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "JavaScript encoder source is not present and is not recorded as empty."
                ),
                details={"relpath": JAVASCRIPT_MODULE_RELPATH},
            )
        )
    elif not node.is_file():
        probes.append(
            _probe(
                "javascript_encoder_agrees",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "/usr/bin/node is absent in the sealed PATH. Live JavaScript "
                    "execution stays typed unavailable and is not recorded as False "
                    "or passing."
                ),
                details={"relpath": JAVASCRIPT_MODULE_RELPATH, "node": NODE_PATH},
            )
        )
    else:
        js_ok = False
        js_reason = "JavaScript encoder disagreed with Python."
        try:
            report = run_javascript_vectors(node=NODE_PATH)
            javascript_agrees_with_python(report)
            js_ok = (
                report.get("language") == "JavaScript"
                and report.get("simulated") is False
            )
            js_reason = (
                "Sealed-environment JavaScript encoder agrees with Python CIDs and reject kinds."
                if js_ok
                else "JavaScript encoder report is incomplete."
            )
        except (NegativeCrossLanguageVectorError, FileNotFoundError, OSError, ValueError) as exc:
            js_ok = False
            js_reason = str(exc)
        probes.append(
            _probe(
                "javascript_encoder_agrees",
                present=js_ok,
                evidence_kind="measured",
                reason=js_reason,
                details={
                    "relpath": JAVASCRIPT_MODULE_RELPATH,
                    "node": NODE_PATH,
                    "languages": list(SUPPORTED_LANGUAGES),
                },
            )
        )

    ts_src = _read_source(root, TYPESCRIPT_SOURCE_RELPATH)
    ts_ok = bool(ts_src) and all(
        recipe_id in (ts_src or "")
        for recipe_id in (
            "invalid.non_mapping",
            "stale.zero_tree",
            "unknown.extra_field",
            "out_of_bound.int64",
            "reordered.nfc_key_collision",
            "reminted.schema",
            "cross_version.schema_v2",
            PINNED_VECTOR_DOCUMENT_CID,
        )
    )
    probes.append(
        _probe(
            "typescript_source_present",
            present=ts_ok,
            evidence_kind="measured" if ts_src is not None else "unavailable",
            reason=(
                "TypeScript companion source pins recipe identities. Live tsc stays typed unavailable."
                if ts_ok
                else "TypeScript companion source is missing or incomplete."
            ),
            details={"relpath": TYPESCRIPT_SOURCE_RELPATH},
        )
    )
    tsc = Path(TSC_PATH)
    probes.append(
        _probe(
            "typescript_compiler",
            present=None,
            evidence_kind="unavailable",
            reason=(
                "tsc is absent from the sealed PATH. TypeScript compilation stays "
                "typed unavailable and is not recorded as False or passing."
                if not tsc.is_file()
                else "tsc is present but this task does not treat compiled TypeScript as live authority."
            ),
            details={"tsc": TSC_PATH, "present_on_path": tsc.is_file()},
        )
    )

    portfolio = portfolio_root or discover_portfolio_root()
    if portfolio is None:
        probes.append(
            _probe(
                "datasets_binding_does_not_remint",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Portfolio tree is not present; Datasets vector binding stays typed unavailable."
                ),
            )
        )
        probes.append(
            _probe(
                "kit_binding_does_not_remint",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Portfolio tree is not present; Kit vector binding stays typed unavailable."
                ),
            )
        )
    else:
        datasets_src = _read_source(portfolio, DATASETS_BINDING_RELPATH)
        datasets_probe = _binding_aligned(datasets_src, DATASETS_BINDING_RELPATH)
        probes.append(
            VectorProbe(
                probe_id="datasets_binding_does_not_remint",
                present=datasets_probe.present,
                evidence_kind=datasets_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Datasets binding pins the negative/cross-language document CID and refuses remint."
                    if datasets_probe.present is True
                    else datasets_probe.reason
                ),
                details=datasets_probe.details,
            )
        )
        kit_src = _read_source(portfolio, KIT_BINDING_RELPATH)
        kit_probe = _binding_aligned(kit_src, KIT_BINDING_RELPATH)
        probes.append(
            VectorProbe(
                probe_id="kit_binding_does_not_remint",
                present=kit_probe.present,
                evidence_kind=kit_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Kit binding pins the negative/cross-language document CID and refuses remint."
                    if kit_probe.present is True
                    else kit_probe.reason
                ),
                details=kit_probe.details,
            )
        )

    probes.append(
        _probe(
            "freeze_not_claimed",
            present=True,
            evidence_kind="measured",
            reason=(
                "PCPR-042 publishes negative and cross-language vectors. Freeze remains PCPR-002 and is not claimed."
            ),
            details={"freeze_task": PCPR_002_TASK_ID, "frozen": False},
        )
    )
    probes.append(
        _probe(
            "cross_repository_compatibility_deferred",
            present=True,
            evidence_kind="measured",
            reason=(
                "Cross-repository compatibility checks remain PCPR-043 and are not claimed here."
            ),
            details={"deferred_task_id": PCPR_043_TASK_ID},
        )
    )
    probes.append(
        _probe(
            "production_authorized_not_granted",
            present=True,
            evidence_kind="measured",
            reason=(
                "Negative and cross-language vectors do not grant production_authorized or a freeze."
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


def qualify_negative_cross_language_vectors(
    *,
    probes: Sequence[VectorProbe],
    duckdb_or_quack_state_written: bool = False,
    simulated_results_represented_as_live: bool = False,
    live_supervisor_qualified: bool = False,
    live_cuda_qualified: bool = False,
    production_authorized: bool = False,
    contracts_frozen: bool = False,
    this_task_created_competing_authority: bool = False,
) -> VectorVerdict:
    """Fail-closed vector verdict. Never a closed release."""

    if duckdb_or_quack_state_written:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise NegativeCrossLanguageVectorQualificationError(
            "simulated vector results cannot be represented as live"
        )
    if live_supervisor_qualified or live_cuda_qualified:
        raise NegativeCrossLanguageVectorQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if production_authorized:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot grant production_authorized"
        )
    if contracts_frozen:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot freeze contracts"
        )
    if this_task_created_competing_authority:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot mint a competing authority"
        )

    normalized: list[VectorProbe] = []
    for item in probes:
        if not isinstance(item, VectorProbe):
            raise NegativeCrossLanguageVectorQualificationError(
                "probes must be VectorProbe values"
            )
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise NegativeCrossLanguageVectorQualificationError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise NegativeCrossLanguageVectorQualificationError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise NegativeCrossLanguageVectorQualificationError(
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

    categories_ok = _require("seven_negative_categories", "negative_categories_incomplete")
    js_ok = _require("javascript_encoder_agrees", "javascript_encoder_disagreed")
    datasets_ok = _require(
        "datasets_binding_does_not_remint", "datasets_binding_remints"
    )
    kit_ok = _require("kit_binding_does_not_remint", "kit_binding_remints")
    for probe_id, blocker in (
        ("negative_vector_module", "vector_module_missing"),
        ("eighteen_negative_vectors_reject", "negative_vector_admitted"),
        ("fail_identically_under_reorder", "reorder_reject_drifted"),
        ("catalog_identity_not_reminted", "catalog_reminted"),
        ("positive_cid_vectors_not_reminted", "positive_cid_reminted"),
        ("typescript_source_present", "typescript_source_missing"),
        ("freeze_not_claimed", "freeze_claimed"),
        ("cross_repository_compatibility_deferred", "compatibility_claimed_early"),
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
        raise NegativeCrossLanguageVectorQualificationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not mint a closed release outcome"
        )

    document_cid = PINNED_VECTOR_DOCUMENT_CID
    try:
        document_cid = vector_document_cid()
    except NegativeCrossLanguageVectorError:
        blockers.append("vector_document_reminted")
        promotion_status = "typed_blocked"

    payload = {
        "schema": VECTOR_VERDICT_SCHEMA,
        "interface": VECTOR_INTERFACE,
        "task_id": PCPR_042_TASK_ID,
        "goal_id": PCPR_042_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": document_cid,
        "negative_count": len(NEGATIVE_RECIPES),
        "cross_language_count": 7,
        "categories_complete": categories_ok,
        "javascript_agreed": js_ok,
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
    return VectorVerdict(
        schema=VECTOR_VERDICT_SCHEMA,
        interface=VECTOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        catalog_cid=PINNED_CATALOG_CID,
        vector_document_cid=document_cid,
        negative_count=len(NEGATIVE_RECIPES),
        cross_language_count=7,
        categories_complete=categories_ok,
        javascript_agreed=js_ok,
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


def qualify_current_head_vectors() -> VectorVerdict:
    """Ordinary current-head PCPR-042 evaluation: honest R&D vectors."""

    return qualify_negative_cross_language_vectors(probes=current_head_vector_probes())


# Pinned after the current-head evaluator is measured. Drift means the
# outer receipt must be regenerated from this evaluator.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera5ucb4kihzmf3gs5rnckjuf2khjq6a7vxogcsxmbetjg663yvlblq"
)


def pcpr_042_receipt_promotion(verdict: VectorVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vector completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise NegativeCrossLanguageVectorQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise NegativeCrossLanguageVectorQualificationError(
            "promotion_status is not an admitted PCPR-042 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot promote the supervisor"
        )
    if verdict.contracts_frozen:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot freeze contracts"
        )
    if verdict.live_supervisor_qualified or verdict.live_cuda_qualified:
        raise NegativeCrossLanguageVectorQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if not verdict.categories_complete:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative categories must be complete"
        )
    return {
        "schema": VECTOR_VERDICT_SCHEMA,
        "interface": VECTOR_INTERFACE,
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
        "negative_count": verdict.negative_count,
        "cross_language_count": verdict.cross_language_count,
        "categories_complete": True,
        "javascript_agreed": verdict.javascript_agreed,
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


def current_head_pcpr_042_receipt_promotion() -> dict[str, Any]:
    return pcpr_042_receipt_promotion(qualify_current_head_vectors())


def pcpr_042_receipt_negative_results() -> dict[str, Any]:
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
        "compatibility_not_claimed": True,
        "live_supervisor_not_claimed": True,
        "live_cuda_not_reclaimed": True,
        "production_authorized_not_granted": True,
        "typescript_compiler_not_claimed": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_042_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_vectors()
    promotion = pcpr_042_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    document = negative_cross_language_vector_document()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_041_TASK_ID,
            "goal_id": PCPR_041_GOAL_ID,
            "promotion_status": "rnd_non_promoted",
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
                "PCPR-041 remains rnd_non_promoted R&D. This task publishes "
                "negative and cross-language vectors on that catalog. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "negative_and_cross_language_vectors": {
            "schema": FLOOR_SCHEMA,
            "interface": FLOOR_INTERFACE,
            "floor_task_id": PCPR_042_TASK_ID,
            "floor_goal_id": PCPR_042_GOAL_ID,
            "catalog_cid": PINNED_CATALOG_CID,
            "vector_document_cid": PINNED_VECTOR_DOCUMENT_CID,
            "frozen": False,
            "freeze_task": PCPR_002_TASK_ID,
            "normative_task": PCPR_040_TASK_ID,
            "positive_vector_task": PCPR_041_TASK_ID,
            "compatibility_task": PCPR_043_TASK_ID,
            "languages": list(SUPPORTED_LANGUAGES),
            "typescript_compiler": "unavailable",
            "negative_categories": list(NEGATIVE_CATEGORIES),
            "negative_count": document["negative_count"],
            "cross_language_count": document["cross_language_count"],
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_042_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_042_current_tree_binding(
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
        raise NegativeCrossLanguageVectorQualificationError(
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
        raise NegativeCrossLanguageVectorQualificationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    kit_origin = _git_object_id(kit_origin_main, "kit_origin_main")
    _require_ancestor(kit_origin_main_is_ancestor, "kit_origin_main_is_ancestor")
    if kit != kit_link:
        raise NegativeCrossLanguageVectorQualificationError(
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


def current_head_pcpr_042_current_tree_binding() -> dict[str, Any]:
    return pcpr_042_current_tree_binding(
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


def validate_pcpr_042_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-042 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise NegativeCrossLanguageVectorQualificationError(
            "outer receipt must be a mapping"
        )
    if payload.get("task_id") != PCPR_042_TASK_ID:
        raise NegativeCrossLanguageVectorQualificationError(
            "outer receipt task_id must be PCPR-042"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise NegativeCrossLanguageVectorQualificationError(
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
        raise NegativeCrossLanguageVectorQualificationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot freeze contracts"
        )
    if verdict_section.get("live_supervisor_qualified") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "this task cannot claim live supervisor qualification"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise NegativeCrossLanguageVectorQualificationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise NegativeCrossLanguageVectorQualificationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-042 status"
        )
    if promotion_status == "supervisor_promoted":
        raise NegativeCrossLanguageVectorQualificationError(
            "negative and cross-language vectors cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise NegativeCrossLanguageVectorQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise NegativeCrossLanguageVectorQualificationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_042_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise NegativeCrossLanguageVectorQualificationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise NegativeCrossLanguageVectorQualificationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise NegativeCrossLanguageVectorQualificationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_042_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "PCPR_042_GOAL_ID",
    "PCPR_042_TASK_ID",
    "VECTOR_INTERFACE",
    "NegativeCrossLanguageVectorQualificationError",
    "current_head_pcpr_042_current_tree_binding",
    "current_head_pcpr_042_receipt_promotion",
    "current_head_pcpr_042_receipt_sections",
    "current_head_vector_probes",
    "qualify_current_head_vectors",
    "qualify_negative_cross_language_vectors",
    "refuse_negative_remint",
    "validate_pcpr_042_outer_receipt",
)
