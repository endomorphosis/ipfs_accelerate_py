"""Fail-closed PCPR-040 shared-contract stabilization.

PCPR-040 publishes one normative catalog for the fourteen shared
contracts. Datasets and Kit bind those identities and cannot remint
them. This evaluator is not a freeze, not canonical-byte or CID vectors,
not negative or cross-language vectors, and not a closed PCPR release.
It does not write DuckDB or Quack state.

Live claims require live evidence. Simulated results stay Simulated.
Missing supervisor promotion, CUDA re-qualification, and solver evidence
stay typed unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.shared_contracts import (
    CANONICAL_CONTRACT_CATALOG,
    CONTRACT_SCHEMA_IDS,
    INTERFACE as FLOOR_INTERFACE,
    OWNER_SCHEMAS,
    PCPR_002_TASK_ID,
    PCPR_040_GOAL_ID,
    PCPR_040_TASK_ID,
    PCPR_041_TASK_ID,
    PCPR_042_TASK_ID,
    PCPR_043_TASK_ID,
    REQUIRED_CONTRACT_NAMES,
    SCHEMA as FLOOR_SCHEMA,
    SharedContractAdmissionError,
    SharedContractError,
    admit_shared_contract,
    canonical_json_bytes,
    catalog_cid,
    catalog_mapping,
    content_identity,
    contract_json_schema,
    identity_fixture,
    refuse_remint,
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


STABILIZATION_INTERFACE: Final = "SharedContractStabilization@1"
STABILIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/shared-contract-stabilization@1"
)
STABILIZATION_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "shared-contract-stabilization-verdict@1"
)

PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_017_TASK_ID: Final = "PCPR-017"
PCPR_027_TASK_ID: Final = "PCPR-027"
PCPR_039_TASK_ID: Final = "PCPR-039"
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
    "4370706843bb3bd9bad31b5c53ced98efcfbf183"
)
CURRENT_HEAD_OUTER_TREE: Final = "ec56382d728fb144f3efb2a76bc4b072d5747607"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit 'c6ebe027e42991b658f839a66be7e6f1c9b653be' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "0fd4ef11c63f5a76e57cabea38e6e8eb37770995"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = (
    "f58cd5ce5a3dff5c86f691bfef31ea142d73370d"
)
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = (
    "1635856f512c8f417577091ffd33f2834de0d62d"
)
CURRENT_HEAD_DATASETS_TREE: Final = "ebe30c3cce6d4f2c74bf6a3be8bbb07bb8c625dd"
CURRENT_HEAD_DATASETS_ORIGIN_MAIN: Final = (
    "f49afc579c22856849ca9f739435e5820003384f"
)
CURRENT_HEAD_KIT_COMMIT: Final = "93ce8c123adb5ab85b6735618a08099c4d03db09"
CURRENT_HEAD_KIT_TREE: Final = "753ad198c8616051d5c214190a9fc51b0cc8216b"
CURRENT_HEAD_KIT_ORIGIN_MAIN: Final = (
    "b6c65ba732733d7e33852713ba18aa3b12235668"
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_shared_contract_stabilization.py",
)

FLOOR_MODULE_RELPATH: Final = "ipfs_accelerate_py/assurance/shared_contracts.py"
DATASETS_BINDING_RELPATH: Final = (
    "external/ipfs_datasets/ipfs_datasets_py/assurance/shared_contracts.py"
)
KIT_BINDING_RELPATH: Final = (
    "external/ipfs_kit/ipfs_kit_py/assurance/shared_contracts.py"
)

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "canonical_catalog_module",
        "fourteen_unique_identities",
        "schema_versions_declared",
        "unknown_fields_rejected",
        "canonicalization_deterministic",
        "cid_rules_closed",
        "unicode_nfc_identity",
        "integer_limits_int64",
        "floats_rejected",
        "owner_schemas_not_reminted",
        "datasets_binding_does_not_remint",
        "kit_binding_does_not_remint",
        "freeze_not_claimed",
        "canonical_byte_vectors_deferred",
        "negative_cross_language_vectors_deferred",
        "cross_repository_compatibility_deferred",
        "production_authorized_not_granted",
    }
)


class SharedContractStabilizationError(ValueError):
    """Malformed stabilization evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class StabilizationProbe:
    """One measured or typed-unavailable shared-contract observation."""

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
class StabilizationVerdict:
    """Fail-closed PCPR-040 decision. Never a closed release."""

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
    contract_count: int
    identities_unique: bool
    unknown_fields_rejected: bool
    canonicalization_deterministic: bool
    datasets_binding_aligned: bool
    kit_binding_aligned: bool
    simulated_results_represented_as_live: bool
    live_supervisor_qualified: bool
    live_supervisor_evidence_kind: str
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    production_authorized: bool
    this_task_created_competing_authority: bool
    probes: tuple[StabilizationProbe, ...]
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
            "contract_count": self.contract_count,
            "identities_unique": self.identities_unique,
            "unknown_fields_rejected": self.unknown_fields_rejected,
            "canonicalization_deterministic": self.canonicalization_deterministic,
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
        raise SharedContractStabilizationError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise SharedContractStabilizationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise SharedContractStabilizationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise SharedContractStabilizationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise SharedContractStabilizationError(f"{name} must be true")


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
) -> StabilizationProbe:
    return StabilizationProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=False,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _binding_aligned(source: str | None, relpath: str) -> StabilizationProbe:
    if source is None:
        return _probe(
            "binding_source",
            present=None,
            evidence_kind="unavailable",
            reason=f"{relpath} is not present and is not recorded as empty.",
            details={"relpath": relpath},
        )
    missing = [
        schema
        for schema in CONTRACT_SCHEMA_IDS.values()
        if schema not in source
    ]
    remint = "remint" in source.lower() and "refuse_remint" not in source
    sibling = "from ipfs_accelerate_py" in source or "from ipfs_datasets_py" in source
    if relpath.endswith("ipfs_kit_py/assurance/shared_contracts.py"):
        sibling = "from ipfs_accelerate_py" in source or (
            "from ipfs_datasets_py" in source
        )
    if relpath.endswith("ipfs_datasets_py/assurance/shared_contracts.py"):
        sibling = "from ipfs_accelerate_py" in source or (
            "from ipfs_kit_py" in source
        )
    ok = not missing and "refuse_remint" in source and not remint and not sibling
    return _probe(
        "binding",
        present=ok,
        evidence_kind="measured",
        reason=(
            "Binding pins the fourteen shared identities and refuses remint."
            if ok
            else "Binding is missing identities, remints, or imports a sibling."
        ),
        details={
            "relpath": relpath,
            "missing_schema_ids": missing,
            "refuse_remint": "refuse_remint" in source,
            "sibling_import": sibling,
        },
    )


def current_head_stabilization_probes(
    *,
    accelerate_root: Path | None = None,
    portfolio_root: Path | None = None,
) -> tuple[StabilizationProbe, ...]:
    """Measured current-tree probes. Missing trees stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[StabilizationProbe] = []
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
    catalog_ok = bool(floor_src) and all(
        name in (floor_src or "") for name in REQUIRED_CONTRACT_NAMES
    ) and FLOOR_SCHEMA in (floor_src or "")
    probes.append(
        _probe(
            "canonical_catalog_module",
            present=catalog_ok,
            evidence_kind="measured" if floor_src is not None else "unavailable",
            reason=(
                "Canonical shared-contract catalog declares the fourteen contracts."
                if catalog_ok
                else "Canonical shared-contract catalog is missing or incomplete."
            ),
            details={"relpath": FLOOR_MODULE_RELPATH},
        )
    )

    names = [item.name for item in CANONICAL_CONTRACT_CATALOG]
    schemas = [item.schema for item in CANONICAL_CONTRACT_CATALOG]
    unique = (
        names == list(REQUIRED_CONTRACT_NAMES)
        and len(set(names)) == 14
        and len(set(schemas)) == 14
        and all(item.schema.startswith("pcpr/shared-contracts/") for item in CANONICAL_CONTRACT_CATALOG)
        and all(item.disposition == "normative_stabilized_not_frozen" for item in CANONICAL_CONTRACT_CATALOG)
    )
    probes.append(
        _probe(
            "fourteen_unique_identities",
            present=unique,
            evidence_kind="measured",
            reason=(
                "Fourteen shared contracts each have one schema identity."
                if unique
                else "Shared-contract identities are missing, duplicated, or reminted."
            ),
            details={"count": len(names), "schema_count": len(set(schemas))},
        )
    )

    versions_ok = all(
        contract_json_schema(name)["x-schema-version"] == CONTRACT_SCHEMA_IDS[name]
        and contract_json_schema(name)["additionalProperties"] is False
        for name in REQUIRED_CONTRACT_NAMES
    )
    probes.append(
        _probe(
            "schema_versions_declared",
            present=versions_ok,
            evidence_kind="measured",
            reason=(
                "Each contract schema declares one version and rejects unknown fields."
                if versions_ok
                else "A contract schema is missing its version or unknown-field policy."
            ),
        )
    )

    unknown_ok = True
    unknown_reason = "Unknown fields are rejected fail-closed."
    try:
        fixture = identity_fixture("SupervisorObjectiveIntent")
        forged = dict(fixture)
        forged["promoted"] = True
        admit_shared_contract("SupervisorObjectiveIntent", forged)
        unknown_ok = False
        unknown_reason = "Unknown fields were admitted."
    except SharedContractAdmissionError:
        unknown_ok = True
    probes.append(
        _probe(
            "unknown_fields_rejected",
            present=unknown_ok,
            evidence_kind="measured",
            reason=unknown_reason,
        )
    )

    first = canonical_json_bytes(catalog_mapping())
    second = canonical_json_bytes(catalog_mapping())
    reordered = dict(reversed(list(catalog_mapping().items())))
    third = canonical_json_bytes(reordered)
    canon_ok = first == second == third and catalog_cid().startswith("b")
    probes.append(
        _probe(
            "canonicalization_deterministic",
            present=canon_ok,
            evidence_kind="measured",
            reason=(
                "Catalog canonical bytes are deterministic under key reordering."
                if canon_ok
                else "Catalog canonicalization is not deterministic."
            ),
            details={"catalog_cid": catalog_cid()},
        )
    )

    cid_ok = True
    cid_reason = "CID rules reject hex, Qm, and uppercase identifiers."
    try:
        refuse_remint("ProofObligation", CONTRACT_SCHEMA_IDS["ProofObligation"])
        admit_shared_contract(
            "DurableArtifactReceipt",
            {
                **{
                    key: value
                    for key, value in identity_fixture("DurableArtifactReceipt").items()
                    if key != "cid"
                },
                "cid": "0" * 64,
            },
        )
        cid_ok = False
        cid_reason = "Raw hex was admitted as a CID."
    except SharedContractAdmissionError:
        cid_ok = True
    probes.append(
        _probe(
            "cid_rules_closed",
            present=cid_ok,
            evidence_kind="measured",
            reason=cid_reason,
        )
    )

    cafe = "caf\u00e9"
    nfc_ok = True
    try:
        composed = admit_shared_contract(
            "ProofObligation",
            {
                **identity_fixture("ProofObligation"),
                "statement": cafe,
            },
        )
        decomposed = admit_shared_contract(
            "ProofObligation",
            {
                **identity_fixture("ProofObligation"),
                "statement": "cafe\u0301",
            },
        )
        nfc_ok = composed["statement"] == decomposed["statement"] == cafe
    except SharedContractAdmissionError:
        nfc_ok = False
    probes.append(
        _probe(
            "unicode_nfc_identity",
            present=nfc_ok,
            evidence_kind="measured",
            reason=(
                "Identity strings normalize to NFC before canonicalization."
                if nfc_ok
                else "Unicode identity strings are not NFC-stable."
            ),
        )
    )

    int_ok = True
    try:
        admit_shared_contract(
            "SupervisorEvent",
            {**identity_fixture("SupervisorEvent"), "sequence": 2**63},
        )
        int_ok = False
    except SharedContractAdmissionError:
        int_ok = True
    probes.append(
        _probe(
            "integer_limits_int64",
            present=int_ok,
            evidence_kind="measured",
            reason=(
                "Integer fields reject values outside the int64 domain."
                if int_ok
                else "Integer overflow was admitted."
            ),
        )
    )

    float_ok = True
    try:
        admit_shared_contract(
            "SupervisorEvent",
            {**identity_fixture("SupervisorEvent"), "sequence": 1.5},
        )
        float_ok = False
    except SharedContractAdmissionError:
        float_ok = True
    probes.append(
        _probe(
            "floats_rejected",
            present=float_ok,
            evidence_kind="measured",
            reason=(
                "Floating-point values cannot mint a shared-contract identity."
                if float_ok
                else "A float was admitted as a shared-contract field."
            ),
        )
    )

    owner_ok = all(
        OWNER_SCHEMAS[name] != CONTRACT_SCHEMA_IDS[name]
        or name in {"SemanticArtifactIdentity"}
        for name in OWNER_SCHEMAS
    ) and OWNER_SCHEMAS["SupervisorContextPack"] == (
        "ipfs_datasets_py/datasets-context-pack@1"
    ) and OWNER_SCHEMAS["ProofObligation"] == (
        "ipfs_accelerate_py/agent-supervisor/code-proof-obligation@1"
    )
    # Owner schemas are bindings, not a second identity for the shared contract.
    owner_ok = (
        OWNER_SCHEMAS["SupervisorContextPack"]
        == "ipfs_datasets_py/datasets-context-pack@1"
        and OWNER_SCHEMAS["ProofObligation"]
        == "ipfs_accelerate_py/agent-supervisor/code-proof-obligation@1"
        and OWNER_SCHEMAS["ProofResult"]
        == "ipfs_accelerate_py/agent-supervisor/proof-receipt@1"
        and OWNER_SCHEMAS["ExecutionInvocation"]
        == "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-request@1"
        and OWNER_SCHEMAS["ExecutionReceipt"]
        == "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-result@1"
        and CONTRACT_SCHEMA_IDS["SupervisorContextPack"]
        != OWNER_SCHEMAS["SupervisorContextPack"]
        and CONTRACT_SCHEMA_IDS["ProofObligation"]
        != OWNER_SCHEMAS["ProofObligation"]
    )
    probes.append(
        _probe(
            "owner_schemas_not_reminted",
            present=owner_ok,
            evidence_kind="measured",
            reason=(
                "Existing owner schemas remain bindings and are not reminted."
                if owner_ok
                else "An owner schema was reminted as the shared identity."
            ),
            details=dict(OWNER_SCHEMAS),
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
                    "Portfolio tree is not present; Datasets binding stays typed unavailable."
                ),
            )
        )
        probes.append(
            _probe(
                "kit_binding_does_not_remint",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Portfolio tree is not present; Kit binding stays typed unavailable."
                ),
            )
        )
    else:
        datasets_src = _read_source(portfolio, DATASETS_BINDING_RELPATH)
        datasets_probe = _binding_aligned(datasets_src, DATASETS_BINDING_RELPATH)
        probes.append(
            StabilizationProbe(
                probe_id="datasets_binding_does_not_remint",
                present=datasets_probe.present,
                evidence_kind=datasets_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Datasets binding pins the catalog identities and refuses remint."
                    if datasets_probe.present is True
                    else datasets_probe.reason
                ),
                details=datasets_probe.details,
            )
        )
        kit_src = _read_source(portfolio, KIT_BINDING_RELPATH)
        kit_probe = _binding_aligned(kit_src, KIT_BINDING_RELPATH)
        probes.append(
            StabilizationProbe(
                probe_id="kit_binding_does_not_remint",
                present=kit_probe.present,
                evidence_kind=kit_probe.evidence_kind,
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Kit binding pins the catalog identities and refuses remint."
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
                "PCPR-040 stabilizes identities. Freeze remains PCPR-002 and is not claimed."
            ),
            details={"freeze_task": PCPR_002_TASK_ID, "frozen": False},
        )
    )
    probes.append(
        _probe(
            "canonical_byte_vectors_deferred",
            present=True,
            evidence_kind="measured",
            reason=(
                "Exact canonical-byte and CID vectors remain PCPR-041 and are not claimed here."
            ),
            details={"deferred_task_id": PCPR_041_TASK_ID},
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
                "Shared-contract stabilization does not grant production_authorized or a freeze."
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


def qualify_shared_contract_stabilization(
    *,
    probes: Sequence[StabilizationProbe],
    duckdb_or_quack_state_written: bool = False,
    simulated_results_represented_as_live: bool = False,
    live_supervisor_qualified: bool = False,
    live_cuda_qualified: bool = False,
    production_authorized: bool = False,
    contracts_frozen: bool = False,
    this_task_created_competing_authority: bool = False,
) -> StabilizationVerdict:
    """Fail-closed stabilization verdict. Never a closed release."""

    if duckdb_or_quack_state_written:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise SharedContractStabilizationError(
            "simulated shared-contract results cannot be represented as live"
        )
    if live_supervisor_qualified or live_cuda_qualified:
        raise SharedContractStabilizationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if production_authorized:
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot grant production_authorized"
        )
    if contracts_frozen:
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot freeze contracts"
        )
    if this_task_created_competing_authority:
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot mint a competing authority"
        )

    normalized: list[StabilizationProbe] = []
    for item in probes:
        if not isinstance(item, StabilizationProbe):
            raise SharedContractStabilizationError(
                "probes must be StabilizationProbe values"
            )
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise SharedContractStabilizationError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise SharedContractStabilizationError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise SharedContractStabilizationError(
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

    catalog_ok = _require("canonical_catalog_module", "catalog_missing")
    unique_ok = _require("fourteen_unique_identities", "identities_not_unique")
    versions_ok = _require("schema_versions_declared", "schema_version_missing")
    unknown_ok = _require("unknown_fields_rejected", "unknown_fields_admitted")
    canon_ok = _require(
        "canonicalization_deterministic", "canonicalization_unstable"
    )
    cid_ok = _require("cid_rules_closed", "cid_rules_open")
    nfc_ok = _require("unicode_nfc_identity", "unicode_not_nfc")
    int_ok = _require("integer_limits_int64", "integer_domain_open")
    float_ok = _require("floats_rejected", "floats_admitted")
    owner_ok = _require("owner_schemas_not_reminted", "owner_schema_reminted")
    datasets_ok = _require(
        "datasets_binding_does_not_remint", "datasets_binding_remints"
    )
    kit_ok = _require("kit_binding_does_not_remint", "kit_binding_remints")
    freeze_ok = _require("freeze_not_claimed", "freeze_claimed")
    vectors_ok = _require(
        "canonical_byte_vectors_deferred", "vectors_claimed_early"
    )
    negative_ok = _require(
        "negative_cross_language_vectors_deferred",
        "negative_vectors_claimed_early",
    )
    compat_ok = _require(
        "cross_repository_compatibility_deferred",
        "compatibility_claimed_early",
    )
    prod_ok = _require(
        "production_authorized_not_granted", "production_authorized_claimed"
    )
    del (
        catalog_ok,
        versions_ok,
        cid_ok,
        nfc_ok,
        int_ok,
        float_ok,
        owner_ok,
        freeze_ok,
        vectors_ok,
        negative_ok,
        compat_ok,
        prod_ok,
    )

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
        raise SharedContractStabilizationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not mint a closed release outcome"
        )

    payload = {
        "schema": STABILIZATION_VERDICT_SCHEMA,
        "interface": STABILIZATION_INTERFACE,
        "task_id": PCPR_040_TASK_ID,
        "goal_id": PCPR_040_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": catalog_cid(),
        "contract_count": len(REQUIRED_CONTRACT_NAMES),
        "identities_unique": unique_ok,
        "unknown_fields_rejected": unknown_ok,
        "canonicalization_deterministic": canon_ok,
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
    return StabilizationVerdict(
        schema=STABILIZATION_VERDICT_SCHEMA,
        interface=STABILIZATION_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        catalog_cid=catalog_cid(),
        contract_count=len(REQUIRED_CONTRACT_NAMES),
        identities_unique=unique_ok,
        unknown_fields_rejected=unknown_ok,
        canonicalization_deterministic=canon_ok,
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


def qualify_current_head_stabilization() -> StabilizationVerdict:
    """Ordinary current-head PCPR-040 evaluation: honest R&D catalog."""

    return qualify_shared_contract_stabilization(
        probes=current_head_stabilization_probes()
    )


# Pinned after the current-head evaluator is measured. Drift means the
# outer receipt must be regenerated from this evaluator.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraxym6kuq5ljps7vwbvrkyobtfhrrpbquhbgrcny4viyfxlwversna"
)


def pcpr_040_receipt_promotion(verdict: StabilizationVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise SharedContractStabilizationError(
            "shared-contract stabilization completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise SharedContractStabilizationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise SharedContractStabilizationError(
            "promotion_status is not an admitted PCPR-040 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot promote the supervisor"
        )
    if verdict.contracts_frozen:
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot freeze contracts"
        )
    if verdict.live_supervisor_qualified or verdict.live_cuda_qualified:
        raise SharedContractStabilizationError(
            "this task cannot claim live supervisor or CUDA qualification"
        )
    if not verdict.identities_unique or not verdict.unknown_fields_rejected:
        raise SharedContractStabilizationError(
            "shared-contract identities must be unique and unknown fields rejected"
        )
    return {
        "schema": STABILIZATION_VERDICT_SCHEMA,
        "interface": STABILIZATION_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "catalog_cid": verdict.catalog_cid,
        "contract_count": verdict.contract_count,
        "identities_unique": True,
        "unknown_fields_rejected": True,
        "canonicalization_deterministic": verdict.canonicalization_deterministic,
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


def current_head_pcpr_040_receipt_promotion() -> dict[str, Any]:
    return pcpr_040_receipt_promotion(qualify_current_head_stabilization())


def pcpr_040_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "unknown_fields_rejected": True,
        "floats_rejected": True,
        "raw_hex_cid_rejected": True,
        "identities_not_reminted": True,
        "contracts_not_frozen": True,
        "canonical_byte_vectors_not_claimed": True,
        "live_supervisor_not_claimed": True,
        "live_cuda_not_reclaimed": True,
        "production_authorized_not_granted": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_040_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_stabilization()
    promotion = pcpr_040_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_039_TASK_ID,
            "goal_id": "PCPR-G430",
            "promotion_status": "rnd_non_promoted",
            "model_provider_verdict_cid": PCPR_039_VERDICT_CID,
            "datasets_solver_task_id": PCPR_017_TASK_ID,
            "datasets_solver_verdict_cid": PCPR_017_VERDICT_CID,
            "kit_support_matrix_task_id": PCPR_027_TASK_ID,
            "kit_support_matrix_verdict_cid": PCPR_027_VERDICT_CID,
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
                "PCPR-017, PCPR-027, and PCPR-039 remain rnd_non_promoted R&D. "
                "This task stabilizes shared-contract identities. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "shared_contracts": {
            "schema": FLOOR_SCHEMA,
            "interface": FLOOR_INTERFACE,
            "floor_task_id": PCPR_040_TASK_ID,
            "floor_goal_id": PCPR_040_GOAL_ID,
            "catalog_cid": verdict.catalog_cid,
            "frozen": False,
            "freeze_task": PCPR_002_TASK_ID,
            "vector_task": PCPR_041_TASK_ID,
            "negative_vector_task": PCPR_042_TASK_ID,
            "compatibility_task": PCPR_043_TASK_ID,
            "contract_count": verdict.contract_count,
            "contracts": [
                item.to_mapping() for item in CANONICAL_CONTRACT_CATALOG
            ],
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_040_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_040_current_tree_binding(
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
        raise SharedContractStabilizationError(
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
        raise SharedContractStabilizationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    kit_origin = _git_object_id(kit_origin_main, "kit_origin_main")
    _require_ancestor(kit_origin_main_is_ancestor, "kit_origin_main_is_ancestor")
    if kit != kit_link:
        raise SharedContractStabilizationError(
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


def current_head_pcpr_040_current_tree_binding() -> dict[str, Any]:
    return pcpr_040_current_tree_binding(
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


def validate_pcpr_040_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-040 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise SharedContractStabilizationError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_040_TASK_ID:
        raise SharedContractStabilizationError(
            "outer receipt task_id must be PCPR-040"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise SharedContractStabilizationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise SharedContractStabilizationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise SharedContractStabilizationError(
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
        raise SharedContractStabilizationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise SharedContractStabilizationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise SharedContractStabilizationError(
            "shared-contract stabilization must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot freeze contracts"
        )
    if verdict_section.get("live_supervisor_qualified") is True:
        raise SharedContractStabilizationError(
            "this task cannot claim live supervisor qualification"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise SharedContractStabilizationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise SharedContractStabilizationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise SharedContractStabilizationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-040 status"
        )
    if promotion_status == "supervisor_promoted":
        raise SharedContractStabilizationError(
            "shared-contract stabilization cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise SharedContractStabilizationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise SharedContractStabilizationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_040_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise SharedContractStabilizationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise SharedContractStabilizationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise SharedContractStabilizationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_040_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }
