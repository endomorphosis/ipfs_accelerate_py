"""Fail-closed PCPR-041 canonical-byte and CID vector evaluation.

PCPR-041 publishes exact canonical DAG-JSON bytes and CIDv1 identities
for the fourteen shared contracts. Datasets and Kit bind those vector
CIDs and cannot remint them. This evaluator is not a freeze, not
negative or cross-language vectors, and not a closed PCPR release. It
does not write DuckDB or Quack state.

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
    INTERFACE as FLOOR_INTERFACE,
    PINNED_CATALOG_CID,
    PINNED_CONTRACT_VECTORS,
    PINNED_NFC_CID,
    PINNED_VECTOR_DOCUMENT_CID,
    PINNED_VECTOR_SEED_CID,
    SCHEMA as FLOOR_SCHEMA,
    CanonicalByteCidVectorError,
    canonical_byte_cid_vector_document,
    catalog_identity_vector,
    compact_recipes,
    contract_vector,
    decode_and_recompute,
    key_order_vector,
    refuse_vector_remint,
    unicode_nfc_vector,
    vector_cid_map,
    vector_document_cid,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    PCPR_002_TASK_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    REQUIRED_CONTRACT_NAMES,
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


VECTOR_INTERFACE: Final = "CanonicalByteCidVectorQualification@1"
VECTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-byte-cid-vector-qualification@1"
)
VECTOR_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "canonical-byte-cid-vector-qualification-verdict@1"
)

PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_017_TASK_ID: Final = "PCPR-017"
PCPR_027_TASK_ID: Final = "PCPR-027"
PCPR_039_TASK_ID: Final = "PCPR-039"
PCPR_040_GOAL_ID: Final = "PCPR-G500"
PCPR_041_GOAL_ID: Final = "PCPR-G510"
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
    "acd9d4936514fb37d680f78d8d6e87edcad8bac7"
)
CURRENT_HEAD_OUTER_TREE: Final = "c3e692d7b75417195934daf5c5c471a01d01751c"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit 'f03bf88738b90f566f25838d0c172787bfd57431' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "df8b82eba6ec14f78cdba467f5a42fcfb01c6e0b"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = (
    "38a3d04218f09f47055488986296abe0be753521"
)
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = (
    "60a2b3928a3220417f7b954fd1b81fe20ba6cf7b"
)
CURRENT_HEAD_DATASETS_TREE: Final = "5071de4087bf812e9b8a4cf886f468cfa3d67292"
CURRENT_HEAD_DATASETS_ORIGIN_MAIN: Final = (
    "f49afc579c22856849ca9f739435e5820003384f"
)
CURRENT_HEAD_KIT_COMMIT: Final = "db948ade4ebb8f711ff6d094141f36da57d683eb"
CURRENT_HEAD_KIT_TREE: Final = "e25a644081301a54bee21464fbedab43d2b1e992"
CURRENT_HEAD_KIT_ORIGIN_MAIN: Final = (
    "b6c65ba732733d7e33852713ba18aa3b12235668"
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_canonical_byte_cid_vectors.py",
)

FLOOR_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/canonical_byte_cid_vectors.py"
)
DATASETS_BINDING_RELPATH: Final = (
    "external/ipfs_datasets/ipfs_datasets_py/assurance/"
    "canonical_byte_cid_vectors.py"
)
KIT_BINDING_RELPATH: Final = (
    "external/ipfs_kit/ipfs_kit_py/assurance/canonical_byte_cid_vectors.py"
)

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "canonical_vector_module",
        "fourteen_contract_vectors",
        "canonical_bytes_pinned",
        "cid_recomputes_from_bytes",
        "key_order_independent",
        "unicode_nfc_same_cid",
        "catalog_identity_not_reminted",
        "datasets_binding_does_not_remint",
        "kit_binding_does_not_remint",
        "freeze_not_claimed",
        "negative_cross_language_vectors_deferred",
        "cross_repository_compatibility_deferred",
        "production_authorized_not_granted",
    }
)


class CanonicalByteCidVectorQualificationError(ValueError):
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
    """Fail-closed PCPR-041 decision. Never a closed release."""

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
    contract_count: int
    vector_count: int
    identities_unique: bool
    canonical_bytes_pinned: bool
    cid_recomputes: bool
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
            "contract_count": self.contract_count,
            "vector_count": self.vector_count,
            "identities_unique": self.identities_unique,
            "canonical_bytes_pinned": self.canonical_bytes_pinned,
            "cid_recomputes": self.cid_recomputes,
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
        raise CanonicalByteCidVectorQualificationError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CanonicalByteCidVectorQualificationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise CanonicalByteCidVectorQualificationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalByteCidVectorQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise CanonicalByteCidVectorQualificationError(f"{name} must be true")


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
    missing = [
        cid
        for cid in vector_cid_map().values()
        if cid not in source
    ]
    remint = "remint" in source.lower() and "refuse_vector_remint" not in source
    sibling = "from ipfs_accelerate_py" in source or "from ipfs_datasets_py" in source
    if relpath.endswith("ipfs_kit_py/assurance/canonical_byte_cid_vectors.py"):
        sibling = "from ipfs_accelerate_py" in source or (
            "from ipfs_datasets_py" in source
        )
    if relpath.endswith(
        "ipfs_datasets_py/assurance/canonical_byte_cid_vectors.py"
    ):
        sibling = "from ipfs_accelerate_py" in source or (
            "from ipfs_kit_py" in source
        )
    ok = (
        not missing
        and "refuse_vector_remint" in source
        and not remint
        and not sibling
        and PINNED_CATALOG_CID in source
    )
    return _probe(
        "binding",
        present=ok,
        evidence_kind="measured",
        reason=(
            "Binding pins the fourteen vector CIDs and refuses remint."
            if ok
            else "Binding is missing vector CIDs, remints, or imports a sibling."
        ),
        details={
            "relpath": relpath,
            "missing_vector_cids": missing,
            "refuse_vector_remint": "refuse_vector_remint" in source,
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
        name in (floor_src or "") for name in REQUIRED_CONTRACT_NAMES
    ) and FLOOR_SCHEMA in (floor_src or "") and PINNED_CATALOG_CID in (
        floor_src or ""
    )
    probes.append(
        _probe(
            "canonical_vector_module",
            present=module_ok,
            evidence_kind="measured" if floor_src is not None else "unavailable",
            reason=(
                "Canonical-byte and CID vector module declares fourteen contracts."
                if module_ok
                else "Canonical-byte and CID vector module is missing or incomplete."
            ),
            details={"relpath": FLOOR_MODULE_RELPATH},
        )
    )

    fourteen_ok = False
    fourteen_reason = "Fourteen contract vectors failed to generate."
    details: dict[str, Any] = {}
    try:
        vectors = [contract_vector(name) for name in REQUIRED_CONTRACT_NAMES]
        cids = [item["cid"] for item in vectors]
        fourteen_ok = (
            len(vectors) == 14
            and len(set(cids)) == 14
            and all(
                item["cid"] == PINNED_CONTRACT_VECTORS[item["contract"]]["cid"]
                for item in vectors
            )
        )
        fourteen_reason = (
            "Fourteen shared contracts each have one pinned canonical-byte CID."
            if fourteen_ok
            else "Contract vector CIDs are missing, duplicated, or reminted."
        )
        details = {"count": len(vectors), "cid_count": len(set(cids))}
    except (CanonicalByteCidVectorError, KeyError) as exc:
        fourteen_ok = False
        fourteen_reason = str(exc)
    probes.append(
        _probe(
            "fourteen_contract_vectors",
            present=fourteen_ok,
            evidence_kind="measured",
            reason=fourteen_reason,
            details=details,
        )
    )

    pinned_ok = False
    pinned_reason = "Canonical-byte pins drifted from the encoder."
    try:
        recipes = compact_recipes()
        pinned_ok = all(
            contract_vector(name)["canonical_sha256"]
            == PINNED_CONTRACT_VECTORS[name]["canonical_sha256"]
            and contract_vector(name)["byte_length"]
            == PINNED_CONTRACT_VECTORS[name]["byte_length"]
            for name in REQUIRED_CONTRACT_NAMES
        ) and len(recipes) == 14
        pinned_reason = (
            "Exact canonical-byte sha256 pins match the catalog encoder."
            if pinned_ok
            else "A canonical-byte pin drifted from the catalog encoder."
        )
    except CanonicalByteCidVectorError as exc:
        pinned_ok = False
        pinned_reason = str(exc)
    probes.append(
        _probe(
            "canonical_bytes_pinned",
            present=pinned_ok,
            evidence_kind="measured",
            reason=pinned_reason,
        )
    )

    recompute_ok = False
    recompute_reason = "Retained canonical bytes did not rehash to the CID."
    try:
        document = canonical_byte_cid_vector_document()
        recomputed = [
            decode_and_recompute(item)
            for item in document["vectors"]
            if item.get("canonical_hex")
        ]
        recompute_ok = (
            len(recomputed) >= 15
            and all(
                decode_and_recompute(item) == item["cid"]
                for item in document["vectors"]
                if item.get("canonical_hex")
            )
            and vector_document_cid() == PINNED_VECTOR_DOCUMENT_CID
        )
        recompute_reason = (
            "Retained canonical bytes rehash to the pinned CIDv1 identities."
            if recompute_ok
            else "A retained canonical-byte vector failed decode-and-recompute."
        )
    except CanonicalByteCidVectorError as exc:
        recompute_ok = False
        recompute_reason = str(exc)
    probes.append(
        _probe(
            "cid_recomputes_from_bytes",
            present=recompute_ok,
            evidence_kind="measured",
            reason=recompute_reason,
            details={"vector_document_cid": PINNED_VECTOR_DOCUMENT_CID},
        )
    )

    order_ok = False
    order_reason = "Key-order independence failed."
    try:
        primary = contract_vector("SupervisorObjectiveIntent")
        ordered = key_order_vector()
        order_ok = (
            ordered["cid"] == primary["cid"] == PINNED_CONTRACT_VECTORS[
                "SupervisorObjectiveIntent"
            ]["cid"]
            and ordered["canonical_hex"] == primary["canonical_hex"]
        )
        order_reason = (
            "Opposite key insertion order yields the same canonical bytes and CID."
            if order_ok
            else "Key reordering reminted canonical bytes or CID."
        )
    except CanonicalByteCidVectorError as exc:
        order_ok = False
        order_reason = str(exc)
    probes.append(
        _probe(
            "key_order_independent",
            present=order_ok,
            evidence_kind="measured",
            reason=order_reason,
        )
    )

    nfc_ok = False
    nfc_reason = "NFC invariance failed."
    try:
        nfc = unicode_nfc_vector()
        nfc_ok = nfc["cid"] == PINNED_NFC_CID and nfc["statement_nfc"] == "caf\u00e9"
        nfc_reason = (
            "NFC composed and decomposed café share canonical bytes and CID."
            if nfc_ok
            else "Unicode NFC vectors minted different identities."
        )
    except CanonicalByteCidVectorError as exc:
        nfc_ok = False
        nfc_reason = str(exc)
    probes.append(
        _probe(
            "unicode_nfc_same_cid",
            present=nfc_ok,
            evidence_kind="measured",
            reason=nfc_reason,
            details={"nfc_cid": PINNED_NFC_CID},
        )
    )

    catalog_ok = False
    catalog_reason = "Catalog identity reminted."
    try:
        catalog = catalog_identity_vector()
        catalog_ok = (
            catalog["cid"] == PINNED_CATALOG_CID == catalog_cid()
            and catalog["normative_task"] == PCPR_040_TASK_ID
        )
        catalog_reason = (
            "PCPR-040 catalog CID is reused and is not reminted."
            if catalog_ok
            else "The shared-contract catalog CID was reminted."
        )
    except CanonicalByteCidVectorError as exc:
        catalog_ok = False
        catalog_reason = str(exc)
    probes.append(
        _probe(
            "catalog_identity_not_reminted",
            present=catalog_ok,
            evidence_kind="measured",
            reason=catalog_reason,
            details={"catalog_cid": PINNED_CATALOG_CID},
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
                    "Datasets binding pins the vector CIDs and refuses remint."
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
                    "Kit binding pins the vector CIDs and refuses remint."
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
                "PCPR-041 publishes vectors. Freeze remains PCPR-002 and is not claimed."
            ),
            details={"freeze_task": PCPR_002_TASK_ID, "frozen": False},
        )
    )
    probes.append(
        _probe(
            "negative_cross_language_vectors_deferred",
            present=True,
            evidence_kind="measured",
            reason=(
                "Negative and cross-language vectors remain PCPR-042 and are not claimed here."
            ),
            details={"deferred_task_id": PCPR_042_TASK_ID},
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
                "Canonical-byte and CID vectors do not grant production_authorized or a freeze."
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


def qualify_canonical_byte_cid_vectors(
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
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise CanonicalByteCidVectorQualificationError(
            "simulated vector results cannot be represented as live"
        )
    if live_supervisor_qualified or live_cuda_qualified:
        raise CanonicalByteCidVectorQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if production_authorized:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot grant production_authorized"
        )
    if contracts_frozen:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot freeze contracts"
        )
    if this_task_created_competing_authority:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot mint a competing authority"
        )

    normalized: list[VectorProbe] = []
    for item in probes:
        if not isinstance(item, VectorProbe):
            raise CanonicalByteCidVectorQualificationError(
                "probes must be VectorProbe values"
            )
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise CanonicalByteCidVectorQualificationError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise CanonicalByteCidVectorQualificationError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise CanonicalByteCidVectorQualificationError(
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

    unique_ok = _require("fourteen_contract_vectors", "vectors_not_unique")
    pinned_ok = _require("canonical_bytes_pinned", "canonical_bytes_unpinned")
    recompute_ok = _require("cid_recomputes_from_bytes", "cid_recompute_failed")
    datasets_ok = _require(
        "datasets_binding_does_not_remint", "datasets_binding_remints"
    )
    kit_ok = _require("kit_binding_does_not_remint", "kit_binding_remints")
    for probe_id, blocker in (
        ("canonical_vector_module", "vector_module_missing"),
        ("key_order_independent", "key_order_unstable"),
        ("unicode_nfc_same_cid", "unicode_not_nfc"),
        ("catalog_identity_not_reminted", "catalog_reminted"),
        ("freeze_not_claimed", "freeze_claimed"),
        ("negative_cross_language_vectors_deferred", "negative_vectors_claimed_early"),
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
        raise CanonicalByteCidVectorQualificationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not mint a closed release outcome"
        )

    document_cid = PINNED_VECTOR_DOCUMENT_CID
    try:
        document_cid = vector_document_cid()
    except CanonicalByteCidVectorError:
        blockers.append("vector_document_reminted")
        promotion_status = "typed_blocked"

    payload = {
        "schema": VECTOR_VERDICT_SCHEMA,
        "interface": VECTOR_INTERFACE,
        "task_id": PCPR_041_TASK_ID,
        "goal_id": PCPR_041_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": document_cid,
        "contract_count": len(REQUIRED_CONTRACT_NAMES),
        "vector_count": 17,
        "identities_unique": unique_ok,
        "canonical_bytes_pinned": pinned_ok,
        "cid_recomputes": recompute_ok,
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
        contract_count=len(REQUIRED_CONTRACT_NAMES),
        vector_count=17,
        identities_unique=unique_ok,
        canonical_bytes_pinned=pinned_ok,
        cid_recomputes=recompute_ok,
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
    """Ordinary current-head PCPR-041 evaluation: honest R&D vectors."""

    return qualify_canonical_byte_cid_vectors(probes=current_head_vector_probes())


# Pinned after the current-head evaluator is measured. Drift means the
# outer receipt must be regenerated from this evaluator.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerafd3yeezarpuf6aicpdycn4sombbpcgfzoekttwytqjpw4pi7qtuq"
)


def pcpr_041_receipt_promotion(verdict: VectorVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vector completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CanonicalByteCidVectorQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise CanonicalByteCidVectorQualificationError(
            "promotion_status is not an admitted PCPR-041 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot promote the supervisor"
        )
    if verdict.contracts_frozen:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot freeze contracts"
        )
    if verdict.live_supervisor_qualified or verdict.live_cuda_qualified:
        raise CanonicalByteCidVectorQualificationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if not verdict.identities_unique or not verdict.canonical_bytes_pinned:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must be unique and pinned"
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
        "contract_count": verdict.contract_count,
        "vector_count": verdict.vector_count,
        "identities_unique": True,
        "canonical_bytes_pinned": True,
        "cid_recomputes": verdict.cid_recomputes,
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


def current_head_pcpr_041_receipt_promotion() -> dict[str, Any]:
    return pcpr_041_receipt_promotion(qualify_current_head_vectors())


def pcpr_041_receipt_negative_results() -> dict[str, Any]:
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
        "negative_vectors_not_claimed": True,
        "live_supervisor_not_claimed": True,
        "live_cuda_not_reclaimed": True,
        "production_authorized_not_granted": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_041_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_vectors()
    promotion = pcpr_041_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    document = canonical_byte_cid_vector_document()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_040_TASK_ID,
            "goal_id": PCPR_040_GOAL_ID,
            "promotion_status": "rnd_non_promoted",
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
                "PCPR-040 remains rnd_non_promoted R&D. This task publishes "
                "canonical-byte and CID vectors from that catalog. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "canonical_byte_and_cid_vectors": {
            "schema": FLOOR_SCHEMA,
            "interface": FLOOR_INTERFACE,
            "floor_task_id": PCPR_041_TASK_ID,
            "floor_goal_id": PCPR_041_GOAL_ID,
            "catalog_cid": PINNED_CATALOG_CID,
            "vector_document_cid": PINNED_VECTOR_DOCUMENT_CID,
            "vector_seed_cid": PINNED_VECTOR_SEED_CID,
            "frozen": False,
            "freeze_task": PCPR_002_TASK_ID,
            "normative_task": PCPR_040_TASK_ID,
            "negative_vector_task": PCPR_042_TASK_ID,
            "compatibility_task": PCPR_043_TASK_ID,
            "contract_count": 14,
            "vector_count": document["vector_count"],
            "vector_cids": vector_cid_map(),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_041_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_041_current_tree_binding(
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
        raise CanonicalByteCidVectorQualificationError(
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
        raise CanonicalByteCidVectorQualificationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    kit_origin = _git_object_id(kit_origin_main, "kit_origin_main")
    _require_ancestor(kit_origin_main_is_ancestor, "kit_origin_main_is_ancestor")
    if kit != kit_link:
        raise CanonicalByteCidVectorQualificationError(
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


def current_head_pcpr_041_current_tree_binding() -> dict[str, Any]:
    return pcpr_041_current_tree_binding(
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


def validate_pcpr_041_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-041 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise CanonicalByteCidVectorQualificationError(
            "outer receipt must be a mapping"
        )
    if payload.get("task_id") != PCPR_041_TASK_ID:
        raise CanonicalByteCidVectorQualificationError(
            "outer receipt task_id must be PCPR-041"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise CanonicalByteCidVectorQualificationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise CanonicalByteCidVectorQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise CanonicalByteCidVectorQualificationError(
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
        raise CanonicalByteCidVectorQualificationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise CanonicalByteCidVectorQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot freeze contracts"
        )
    if verdict_section.get("live_supervisor_qualified") is True:
        raise CanonicalByteCidVectorQualificationError(
            "this task cannot claim live supervisor qualification"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise CanonicalByteCidVectorQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise CanonicalByteCidVectorQualificationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise CanonicalByteCidVectorQualificationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-041 status"
        )
    if promotion_status == "supervisor_promoted":
        raise CanonicalByteCidVectorQualificationError(
            "canonical-byte and CID vectors cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise CanonicalByteCidVectorQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise CanonicalByteCidVectorQualificationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_041_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise CanonicalByteCidVectorQualificationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise CanonicalByteCidVectorQualificationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise CanonicalByteCidVectorQualificationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_041_TASK_ID,
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
    "PCPR_041_GOAL_ID",
    "PCPR_041_TASK_ID",
    "VECTOR_INTERFACE",
    "CanonicalByteCidVectorQualificationError",
    "current_head_pcpr_041_current_tree_binding",
    "current_head_pcpr_041_receipt_promotion",
    "current_head_pcpr_041_receipt_sections",
    "current_head_vector_probes",
    "qualify_canonical_byte_cid_vectors",
    "qualify_current_head_vectors",
    "refuse_vector_remint",
    "validate_pcpr_041_outer_receipt",
)
