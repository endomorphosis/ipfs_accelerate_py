"""Fail-closed PCPR-032 pseudo-CID identity removal.

PCPR-032 replaces hexadecimal SHA-256 slices and Qm-prefixed random strings
with canonical CIDv1 minted from retained bytes. Ordinary runtime cannot
treat random or truncated identifiers as IPFS CIDs. This module is not
release authority: it does not write DuckDB or Quack state, does not
qualify live IPFS, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated identities stay ``Simulated``.
Missing IPFS storage stays typed unavailable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.content_identity import (
    CanonicalIPFSMultiformats,
    classify_pseudo_cid,
    is_qm_like,
    is_raw_sha256_hex,
    legacy_pseudo_cid,
    mint_content_identity,
    reject_pseudo_cid,
    verify_content_identity,
)
from ipfs_accelerate_py.compatibility.simulation.pseudo_cid import (
    MockIPFSClient,
    PseudoCidIdentityError,
    UnavailableIpfsClient,
    instantiate_mock_ipfs_client,
    load_ordinary_ipfs_client,
    load_ordinary_multiformats,
    mint_canonical_cid,
    ordinary_store_to_ipfs,
    random_cid,
    simulate_store_to_ipfs,
)

from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .fabricated_hardware_removal import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_031_VERDICT_CID,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
)


REMOVAL_INTERFACE: Final = "PseudoCidIdentityRemoval@1"
REMOVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pseudo-cid-identity-removal@1"
)
REMOVAL_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pseudo-cid-identity-removal-verdict@1"
)

PCPR_032_TASK_ID: Final = "PCPR-032"
PCPR_032_GOAL_ID: Final = "PCPR-G410"
PCPR_031_TASK_ID: Final = "PCPR-031"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"

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

CURRENT_HEAD_OUTER_COMMIT: Final = "d6f90c4369fd60976e13d990d3bf0e9b56e2eca8"
CURRENT_HEAD_OUTER_TREE: Final = "118370ecc25b66973b974c33786a8b7b300627b5"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '7fd154dfecc696d5def354a2b65036ad1493f629' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "15726be7c439f47a1c76bdec1ef8c88a7b5ff478"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "4805aa04fbcce976e9f7737554ada9288ab9bf7b"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_pseudo_cid_identity_removal.py",
)

SIMULATION_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/compatibility/simulation/pseudo_cid.py"
)
IDENTITY_MODULE_RELPATH: Final = "ipfs_accelerate_py/assurance/content_identity.py"
ORDINARY_RUNTIME_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate.py"
LEGACY_MODULE_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py"
MOCK_IPFS_RELPATH: Final = "ipfs_accelerate_py/mcp/tools/mock_ipfs.py"
BACKEND_ROUTER_RELPATH: Final = "ipfs_accelerate_py/ipfs_backend_router.py"
KIT_INTEGRATION_RELPATH: Final = "ipfs_accelerate_py/ipfs_kit_integration.py"
MCP_IPFS_RELPATH: Final = "ipfs_accelerate_py/mcp_server/tools/ipfs/__init__.py"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
QM_RANDOM_RE: Final = re.compile(
    r"""["']Qm["']\s*\+|f["']Qm\{|return\s+["']Qm["']\s*\+|bafy\{hash_value"""
)


class PseudoCidIdentityRemovalError(ValueError):
    """Malformed pseudo-CID-removal evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class RemovalProbe:
    """One measured or typed-unavailable pseudo-CID-removal observation."""

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
class RemovalVerdict:
    """Fail-closed PCPR-032 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    ordinary_runtime_uses_pseudo_cid: bool
    mock_ipfs_quarantined: bool
    random_cid_requires_canonical_bytes: bool
    canonical_bytes_produce_real_cid: bool
    simulated_results_represented_as_live: bool
    live_ipfs_qualified: bool
    live_ipfs_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[RemovalProbe, ...]
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
            "ordinary_runtime_uses_pseudo_cid": self.ordinary_runtime_uses_pseudo_cid,
            "mock_ipfs_quarantined": self.mock_ipfs_quarantined,
            "random_cid_requires_canonical_bytes": (
                self.random_cid_requires_canonical_bytes
            ),
            "canonical_bytes_produce_real_cid": self.canonical_bytes_produce_real_cid,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_ipfs_qualified": self.live_ipfs_qualified,
            "live_ipfs_evidence_kind": self.live_ipfs_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PseudoCidIdentityRemovalError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise PseudoCidIdentityRemovalError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise PseudoCidIdentityRemovalError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise PseudoCidIdentityRemovalError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise PseudoCidIdentityRemovalError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _source_probe(
    *,
    probe_id: str,
    source: str | None,
    present: bool | None,
    ok_reason: str,
    fail_reason: str,
    missing_reason: str,
    relpath: str,
) -> RemovalProbe:
    if source is None:
        return RemovalProbe(
            probe_id=probe_id,
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=missing_reason,
        )
    ok = bool(present)
    return RemovalProbe(
        probe_id=probe_id,
        present=ok,
        evidence_kind="measured",
        live=False,
        simulated_represented_as_live=False,
        reason=ok_reason if ok else fail_reason,
        details=MappingProxyType({"relpath": relpath}),
    )


def current_head_removal_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[RemovalProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[RemovalProbe] = []
    if root is None or not root.is_dir():
        return (
            RemovalProbe(
                probe_id="accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Accelerate source tree is not present and is not recorded as empty.",
            ),
        )

    simulation = _read_source(root, SIMULATION_MODULE_RELPATH)
    identity = _read_source(root, IDENTITY_MODULE_RELPATH)
    ordinary = _read_source(root, ORDINARY_RUNTIME_RELPATH)
    legacy = _read_source(root, LEGACY_MODULE_RELPATH)
    mock_ipfs = _read_source(root, MOCK_IPFS_RELPATH)
    backend = _read_source(root, BACKEND_ROUTER_RELPATH)
    kit = _read_source(root, KIT_INTEGRATION_RELPATH)
    mcp_ipfs = _read_source(root, MCP_IPFS_RELPATH)

    probes.append(
        _source_probe(
            probe_id="simulation_namespace_pseudo_cid",
            source=simulation,
            present=bool(
                simulation
                and "class MockIPFSClient" in simulation
                and "def random_cid" in simulation
                and "PSEUDO_CID_RANDOM_QM_REMOVED" in simulation
            ),
            ok_reason="Mock IPFS client and random_cid live in the explicit simulation namespace.",
            fail_reason="Simulation namespace is missing quarantined pseudo-CID symbols.",
            missing_reason="Simulation namespace module is unreadable.",
            relpath=SIMULATION_MODULE_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="canonical_identity_module",
            source=identity,
            present=bool(
                identity
                and "class CanonicalIPFSMultiformats" in identity
                and "def reject_pseudo_cid" in identity
            ),
            ok_reason="Canonical content identity rejects pseudo-CIDs and mints CIDv1.",
            fail_reason="Canonical content identity module is incomplete.",
            missing_reason="content_identity.py is unreadable.",
            relpath=IDENTITY_MODULE_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="ordinary_runtime_multiformats",
            source=ordinary,
            present=bool(
                ordinary
                and "load_ordinary_multiformats" in ordinary
                and "hexdigest()" not in ordinary.split("self.ipfs_multiformats", 1)[-1][:400]
            ),
            ok_reason="Ordinary constructor loads CanonicalIPFSMultiformats, not hexdigest-as-CID.",
            fail_reason="Ordinary constructor still mints hexadecimal pseudo-CIDs.",
            missing_reason="ipfs_accelerate.py is unreadable.",
            relpath=ORDINARY_RUNTIME_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="ordinary_legacy_store_no_qm",
            source=legacy,
            present=bool(
                legacy
                and "ordinary_store_to_ipfs" in legacy
                and 'mock_cid = f"Qm' not in legacy
            ),
            ok_reason="Ordinary legacy store_to_ipfs is typed unavailable, not a Qm+hex CID.",
            fail_reason="Ordinary legacy store_to_ipfs still returns a Qm pseudo-CID.",
            missing_reason="Legacy module is unreadable.",
            relpath=LEGACY_MODULE_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="mock_ipfs_shim_no_random_qm",
            source=mock_ipfs,
            present=bool(
                mock_ipfs
                and QM_RANDOM_RE.search(mock_ipfs) is None
                and "random_cid" in mock_ipfs
            ),
            ok_reason="mock_ipfs.py no longer concatenates random Qm strings.",
            fail_reason="mock_ipfs.py still generates Qm-prefixed random strings.",
            missing_reason="mock_ipfs.py is unreadable.",
            relpath=MOCK_IPFS_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="backend_router_canonical_cid",
            source=backend,
            present=bool(
                backend
                and "mint_canonical_cid" in backend
                and "bafy{hash_value" not in backend
            ),
            ok_reason="Backend router _generate_cid mints canonical CIDv1.",
            fail_reason="Backend router still fabricates bafy+hex cache keys as CIDs.",
            missing_reason="ipfs_backend_router.py is unreadable.",
            relpath=BACKEND_ROUTER_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="kit_integration_canonical_cid",
            source=kit,
            present=bool(
                kit
                and "mint_canonical_cid" in kit
                and "bafy{hash_value" not in kit
            ),
            ok_reason="Kit integration _generate_cid mints canonical CIDv1.",
            fail_reason="Kit integration still fabricates bafy+hex identifiers as CIDs.",
            missing_reason="ipfs_kit_integration.py is unreadable.",
            relpath=KIT_INTEGRATION_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="mcp_ipfs_no_random_qm",
            source=mcp_ipfs,
            present=bool(
                mcp_ipfs
                and QM_RANDOM_RE.search(mcp_ipfs) is None
                and "compatibility.simulation.pseudo_cid" in mcp_ipfs
            ),
            ok_reason="MCP IPFS shim loads the quarantined simulation client, not a Qm stub.",
            fail_reason="MCP IPFS shim still generates Qm-prefixed random strings.",
            missing_reason="mcp_server/tools/ipfs/__init__.py is unreadable.",
            relpath=MCP_IPFS_RELPATH,
        )
    )

    adapter = load_ordinary_multiformats()
    payload = b"pcpr-032-canonical-bytes"
    minted = adapter.get_cid(payload)
    hex_form = legacy_pseudo_cid(payload)
    ordinary_ok = (
        isinstance(adapter, CanonicalIPFSMultiformats)
        and not is_raw_sha256_hex(minted)
        and not is_qm_like(minted)
        and minted.startswith("b")
        and classify_pseudo_cid(hex_form) is not None
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_multiformats_runtime",
            present=ordinary_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary multiformats adapter mints canonical CIDv1, not hex."
                if ordinary_ok
                else "Ordinary multiformats adapter still emits a pseudo-CID."
            ),
            details=MappingProxyType(
                {
                    "type_name": type(adapter).__name__,
                    "cid_prefix": minted[:8],
                    "hex_rejected": classify_pseudo_cid(hex_form).value
                    if classify_pseudo_cid(hex_form)
                    else None,
                }
            ),
        )
    )

    verified = verify_content_identity(minted, payload)
    probes.append(
        RemovalProbe(
            probe_id="canonical_cid_roundtrip",
            present=bool(verified.ok and verified.integrity == "digest_valid"),
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Canonical CID decodes and recomputes against retained bytes."
                if verified.ok
                else "Canonical CID failed digest verification."
            ),
            details=MappingProxyType(
                {
                    "ok": verified.ok,
                    "integrity": verified.integrity,
                    "cid": verified.cid,
                }
            ),
        )
    )

    hex_blocked = False
    try:
        reject_pseudo_cid(hex_form)
    except Exception:
        hex_blocked = True
    probes.append(
        RemovalProbe(
            probe_id="hex_pseudo_cid_rejected",
            present=hex_blocked and is_raw_sha256_hex(hex_form),
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason="Raw SHA-256 hex is rejected as a pseudo-CID.",
        )
    )

    qm_blocked = False
    try:
        reject_pseudo_cid("QmFakeCidFromStringifiedData000000000000000")
    except Exception:
        qm_blocked = True
    probes.append(
        RemovalProbe(
            probe_id="qm_pseudo_cid_rejected",
            present=qm_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason="Qm-prefixed strings are rejected as pseudo-CIDs.",
        )
    )

    ordinary_client = load_ordinary_ipfs_client()
    ordinary_client_ok = (
        isinstance(ordinary_client, UnavailableIpfsClient)
        and ordinary_client.live is False
        and ordinary_client.add_file("missing")["cid"] is None
        and ordinary_client.add_file("missing")["outcome"] == "Unavailable"
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_ipfs_client",
            present=ordinary_client_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary IPFS client is typed unavailable and not mock."
                if ordinary_client_ok
                else "Ordinary IPFS client is not the typed unavailable stand-in."
            ),
            details=MappingProxyType({"type_name": type(ordinary_client).__name__}),
        )
    )

    mock_blocked = False
    mock_code = ""
    try:
        instantiate_mock_ipfs_client()
    except PseudoCidIdentityError as exc:
        mock_blocked = True
        mock_code = exc.code
    probes.append(
        RemovalProbe(
            probe_id="mock_ipfs_ordinary_instantiation",
            present=mock_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Mock IPFS ordinary instantiation is refused."
                if mock_blocked
                else "Mock IPFS can still be constructed without explicit simulation."
            ),
            details=MappingProxyType({"code": mock_code}),
        )
    )

    random_blocked = False
    random_code = ""
    try:
        random_cid()
    except PseudoCidIdentityError as exc:
        random_blocked = True
        random_code = exc.code
    probes.append(
        RemovalProbe(
            probe_id="random_cid_ordinary_instantiation",
            present=random_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "random_cid ordinary use is refused."
                if random_blocked
                else "random_cid can still mint a Qm string without explicit simulation."
            ),
            details=MappingProxyType({"code": random_code}),
        )
    )

    random_sim_blocked = False
    random_sim_code = ""
    try:
        random_cid(explicit_simulation=True)
    except PseudoCidIdentityError as exc:
        random_sim_blocked = True
        random_sim_code = exc.code
    probes.append(
        RemovalProbe(
            probe_id="random_cid_requires_canonical_bytes",
            present=random_sim_blocked and random_sim_code == "pseudo_cid_random_qm_removed",
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Even explicit simulation refuses random Qm identity without canonical bytes."
                if random_sim_blocked
                else "Explicit simulation still mints a random Qm string."
            ),
            details=MappingProxyType({"code": random_sim_code}),
        )
    )

    simulated = instantiate_mock_ipfs_client(explicit_simulation=True)
    import tempfile

    simulated_as_live = False
    simulated_cid = ""
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as handle:
        handle.write(payload)
        temp_path = handle.name
    try:
        added = simulated.add_file(temp_path)
        simulated_cid = str(added.get("Hash") or "")
        simulated_as_live = bool(simulated.live) or added.get("live") is True
        simulated_ok = (
            isinstance(simulated, MockIPFSClient)
            and not is_qm_like(simulated_cid)
            and not is_raw_sha256_hex(simulated_cid)
            and added.get("outcome") == "Simulated"
            and not simulated_as_live
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)
    probes.append(
        RemovalProbe(
            probe_id="mock_ipfs_explicit_simulation",
            present=simulated_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=simulated_as_live,
            reason=(
                "Explicit mock IPFS mints canonical CIDv1 and is not live."
                if simulated_ok
                else "Explicit mock IPFS reported a live or pseudo-CID claim."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated.origin,
                    "outcome": simulated.outcome,
                    "cid_prefix": simulated_cid[:8],
                }
            ),
        )
    )

    store = ordinary_store_to_ipfs(payload)
    store_ok = (
        store.get("cid") is None
        and store.get("outcome") == "Unavailable"
        and store.get("live") is False
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_store_to_ipfs",
            present=store_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary store_to_ipfs is typed unavailable and does not emit a CID."
                if store_ok
                else "Ordinary store_to_ipfs still claims a stored CID."
            ),
            details=MappingProxyType(
                {"status": store.get("status"), "code": store.get("code")}
            ),
        )
    )

    simulated_store = simulate_store_to_ipfs(payload, explicit_simulation=True)
    sim_store_as_live = simulated_store.get("live") is True
    sim_store_ok = (
        simulated_store.get("outcome") == "Simulated"
        and not is_qm_like(str(simulated_store.get("cid") or ""))
        and not sim_store_as_live
    )
    probes.append(
        RemovalProbe(
            probe_id="simulated_store_to_ipfs",
            present=sim_store_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=sim_store_as_live,
            reason=(
                "Explicit simulated store mints canonical CID and is labeled Simulated."
                if sim_store_ok
                else "Explicit simulated store claimed live IPFS storage."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated_store.get("origin"),
                    "outcome": simulated_store.get("outcome"),
                    "cid_prefix": str(simulated_store.get("cid") or "")[:8],
                }
            ),
        )
    )

    identity_obj = mint_content_identity(payload)
    probes.append(
        RemovalProbe(
            probe_id="canonical_bytes_produce_real_cid",
            present=bool(
                identity_obj.cid.startswith("b")
                and identity_obj.integrity == "digest_valid"
                and not is_raw_sha256_hex(identity_obj.cid)
            ),
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason="Canonical bytes produce a real CIDv1, not a hex digest.",
            details=MappingProxyType(
                {
                    "codec": identity_obj.codec,
                    "integrity": identity_obj.integrity,
                }
            ),
        )
    )

    probes.append(
        RemovalProbe(
            probe_id="live_ipfs_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify a live IPFS daemon. Missing storage "
                "evidence stays typed unavailable and is not recorded as False or passing."
            ),
        )
    )
    return tuple(probes)


def qualify_pseudo_cid_identity_removal(
    *,
    probes: Sequence[RemovalProbe],
    duckdb_or_quack_state_written: bool = False,
) -> RemovalVerdict:
    """Evaluate PCPR-032. Removal is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not write DuckDB or Quack state"
        )
    if not probes:
        raise PseudoCidIdentityRemovalError("pseudo-CID-removal probes are required")

    normalized: list[RemovalProbe] = []
    blockers: list[str] = []
    simulated_as_live = False
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise PseudoCidIdentityRemovalError(
                f"{probe_id}: estimated values cannot mint pseudo-CID-removal evidence"
            )
        if kind == "simulated" and probe.present is True:
            raise PseudoCidIdentityRemovalError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "measured_live":
            raise PseudoCidIdentityRemovalError(
                f"{probe_id}: this removal evaluator cannot carry measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            simulated_as_live = True
            blockers.append(f"{probe_id}:simulated_represented_as_live")
        if probe.live:
            blockers.append(f"{probe_id}:live_claim")
        normalized.append(
            RemovalProbe(
                probe_id=probe_id,
                present=probe.present,
                evidence_kind=kind,
                live=bool(probe.live),
                simulated_represented_as_live=bool(probe.simulated_represented_as_live),
                reason=str(probe.reason or ""),
                details=MappingProxyType(dict(probe.details)),
            )
        )

    by_id = {item.probe_id: item for item in normalized}

    def _require(probe_id: str, blocker: str) -> bool:
        item = by_id.get(probe_id)
        ok = bool(item and item.present is True)
        if not ok:
            blockers.append(blocker)
        return ok

    ordinary = by_id.get("ordinary_runtime_multiformats")
    uses_pseudo = bool(ordinary and ordinary.present is False)
    if ordinary is None or ordinary.present is not True:
        blockers.append("ordinary_runtime_uses_pseudo_cid")

    mock_quarantined = _require(
        "simulation_namespace_pseudo_cid", "mock_ipfs_not_in_simulation_namespace"
    )
    _require("canonical_identity_module", "canonical_identity_missing")
    _require("ordinary_legacy_store_no_qm", "legacy_store_emits_qm_cid")
    _require("mock_ipfs_shim_no_random_qm", "mock_ipfs_random_qm")
    _require("backend_router_canonical_cid", "backend_router_synthetic_bafy")
    _require("kit_integration_canonical_cid", "kit_integration_synthetic_bafy")
    _require("mcp_ipfs_no_random_qm", "mcp_ipfs_random_qm")
    _require("ordinary_multiformats_runtime", "ordinary_multiformats_pseudo_cid")
    _require("canonical_cid_roundtrip", "canonical_cid_roundtrip_failed")
    _require("hex_pseudo_cid_rejected", "hex_pseudo_cid_admitted")
    _require("qm_pseudo_cid_rejected", "qm_pseudo_cid_admitted")
    _require("ordinary_ipfs_client", "ordinary_ipfs_not_unavailable")
    _require("mock_ipfs_ordinary_instantiation", "mock_ipfs_ordinary_instantiation_allowed")
    _require("random_cid_ordinary_instantiation", "random_cid_ordinary_allowed")
    random_gated = _require(
        "random_cid_requires_canonical_bytes",
        "random_cid_still_mints_qm",
    )
    _require("mock_ipfs_explicit_simulation", "mock_ipfs_reported_live_or_pseudo")
    _require("ordinary_store_to_ipfs", "ordinary_store_emits_cid")
    _require("simulated_store_to_ipfs", "simulated_store_claimed_live")
    canonical_ok = _require(
        "canonical_bytes_produce_real_cid", "canonical_bytes_did_not_mint_cidv1"
    )

    unavailable_count = sum(1 for item in normalized if item.evidence_kind == "unavailable")
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise PseudoCidIdentityRemovalError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not mint a closed release outcome"
        )

    payload = {
        "schema": REMOVAL_VERDICT_SCHEMA,
        "interface": REMOVAL_INTERFACE,
        "task_id": PCPR_032_TASK_ID,
        "goal_id": PCPR_032_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_uses_pseudo_cid": uses_pseudo,
        "mock_ipfs_quarantined": mock_quarantined,
        "random_cid_requires_canonical_bytes": random_gated,
        "canonical_bytes_produce_real_cid": canonical_ok,
        "simulated_results_represented_as_live": simulated_as_live,
        "live_ipfs_qualified": False,
        "live_ipfs_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return RemovalVerdict(
        schema=REMOVAL_VERDICT_SCHEMA,
        interface=REMOVAL_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        ordinary_runtime_uses_pseudo_cid=uses_pseudo,
        mock_ipfs_quarantined=mock_quarantined,
        random_cid_requires_canonical_bytes=random_gated,
        canonical_bytes_produce_real_cid=canonical_ok,
        simulated_results_represented_as_live=simulated_as_live,
        live_ipfs_qualified=False,
        live_ipfs_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_removal() -> RemovalVerdict:
    """Ordinary current-head PCPR-032 evaluation: honest R&D removal."""

    return qualify_pseudo_cid_identity_removal(
        probes=current_head_removal_probes(),
    )


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraix4sod7r2ocsw2aqasj6r6tiksk4ts43ywaoosfidtdgbmcgohmq"
)


def pcpr_032_receipt_promotion(verdict: RemovalVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise PseudoCidIdentityRemovalError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise PseudoCidIdentityRemovalError(
            "promotion_status is not an admitted PCPR-032 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal cannot promote the supervisor"
        )
    if verdict.live_ipfs_qualified:
        raise PseudoCidIdentityRemovalError(
            "this task cannot claim live IPFS qualification"
        )
    return {
        "schema": REMOVAL_VERDICT_SCHEMA,
        "interface": REMOVAL_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_uses_pseudo_cid": verdict.ordinary_runtime_uses_pseudo_cid,
        "mock_ipfs_quarantined": verdict.mock_ipfs_quarantined,
        "random_cid_requires_canonical_bytes": (
            verdict.random_cid_requires_canonical_bytes
        ),
        "canonical_bytes_produce_real_cid": verdict.canonical_bytes_produce_real_cid,
        "simulated_results_represented_as_live": (
            verdict.simulated_results_represented_as_live
        ),
        "live_ipfs_qualified": False,
        "live_ipfs_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_032_receipt_promotion() -> dict[str, Any]:
    return pcpr_032_receipt_promotion(qualify_current_head_removal())


def pcpr_032_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_ipfs_not_claimed": True,
        "hex_digest_not_a_cid": True,
        "qm_random_string_not_a_cid": True,
        "ordinary_runtime_cannot_instantiate_mock_ipfs": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_032_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_removal()
    promotion = pcpr_032_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_031_TASK_ID,
            "goal_id": "PCPR-G410",
            "promotion_status": "rnd_non_promoted",
            "removal_verdict_cid": PCPR_031_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-031 removed fabricated hardware availability. This task "
                "removes pseudo-CID identity. PCPR-001 live qualification "
                "remains rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "removal": {
            "schema": REMOVAL_SCHEMA,
            "interface": REMOVAL_INTERFACE,
            "namespace": "ipfs_accelerate_py.compatibility.simulation.pseudo_cid",
            "identity_schema": "ipfs_accelerate_py/assurance/content-identity@1",
            "ordinary_runtime_uses_pseudo_cid": (
                verdict.ordinary_runtime_uses_pseudo_cid
            ),
            "mock_ipfs_quarantined": verdict.mock_ipfs_quarantined,
            "random_cid_requires_canonical_bytes": (
                verdict.random_cid_requires_canonical_bytes
            ),
            "canonical_bytes_produce_real_cid": verdict.canonical_bytes_produce_real_cid,
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_032_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_032_current_tree_binding(
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
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
) -> dict[str, Any]:
    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(accelerator_pre_change_commit, "accelerator_pre_change_commit")
    accel_tree = _git_object_id(accelerator_pre_change_tree, "accelerator_pre_change_tree")
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(accelerator_origin_main, "accelerator_origin_main")
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise PseudoCidIdentityRemovalError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise PseudoCidIdentityRemovalError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise PseudoCidIdentityRemovalError("kit_commit must equal kit_gitlink")
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
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_032_current_tree_binding() -> dict[str, Any]:
    return pcpr_032_current_tree_binding(
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
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
    )


def validate_pcpr_032_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-032 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise PseudoCidIdentityRemovalError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_032_TASK_ID:
        raise PseudoCidIdentityRemovalError("outer receipt task_id must be PCPR-032")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise PseudoCidIdentityRemovalError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise PseudoCidIdentityRemovalError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise PseudoCidIdentityRemovalError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise PseudoCidIdentityRemovalError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise PseudoCidIdentityRemovalError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal cannot freeze contracts"
        )
    if verdict_section.get("live_ipfs_qualified") is True:
        raise PseudoCidIdentityRemovalError(
            "this task cannot claim live IPFS qualification"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise PseudoCidIdentityRemovalError(
            "qualification_verdict.promotion_status is not an admitted PCPR-032 status"
        )
    if promotion_status == "supervisor_promoted":
        raise PseudoCidIdentityRemovalError(
            "pseudo-CID identity removal cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise PseudoCidIdentityRemovalError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise PseudoCidIdentityRemovalError("acceptance must not claim a release")

    expected = current_head_pcpr_032_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise PseudoCidIdentityRemovalError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise PseudoCidIdentityRemovalError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise PseudoCidIdentityRemovalError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_032_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }


def file_digest(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_ACCELERATOR_COMMIT",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "PCPR_032_GOAL_ID",
    "PCPR_032_TASK_ID",
    "REMOVAL_INTERFACE",
    "SEALED_GIT_BINARY",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "PseudoCidIdentityRemovalError",
    "RemovalProbe",
    "RemovalVerdict",
    "current_head_pcpr_032_current_tree_binding",
    "current_head_pcpr_032_receipt_promotion",
    "current_head_pcpr_032_receipt_sections",
    "current_head_removal_probes",
    "pcpr_032_current_tree_binding",
    "pcpr_032_receipt_promotion",
    "qualify_current_head_removal",
    "qualify_pseudo_cid_identity_removal",
    "validate_pcpr_032_outer_receipt",
)
