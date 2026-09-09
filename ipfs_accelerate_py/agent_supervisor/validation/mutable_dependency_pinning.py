"""Fail-closed PCPR-035 mutable-dependency pinning.

PCPR-035 replaces mutable Git branches with immutable commits in
Accelerate qualified requirement profiles and runtime VCS specs.
This module is not release authority: it does not write DuckDB or
Quack state, does not install the pinned packages, and never emits a
closed PCPR release outcome.

Live claims require live evidence. Simulated results stay Simulated.
Missing remotes, installs, and environments stay typed unavailable.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.mutable_dependency_pins import (
    COMMIT_RE,
    EXCLUDED_RELPATH_PREFIXES,
    INTERFACE as PIN_INTERFACE,
    PINS,
    QUALIFIED_PROFILE_RELPATHS,
    RUNTIME_SPEC_RELPATHS,
    SCHEMA as PIN_SCHEMA,
    TASK_ID as PIN_TASK_ID,
    mutable_git_references,
    pep508_spec,
)
from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .capability_ladder_consolidation import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_034_VERDICT_CID,
)
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)
from .source_seal_and_supervisor_baseline import SEALED_GIT_BINARY


PINNING_EVALUATOR_INTERFACE: Final = "MutableDependencyPinning@1"
PINNING_EVALUATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mutable-dependency-pinning@1"
)
PINNING_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mutable-dependency-pinning-verdict@1"
)

PCPR_035_TASK_ID: Final = "PCPR-035"
PCPR_035_GOAL_ID: Final = "PCPR-G420"
PCPR_034_TASK_ID: Final = "PCPR-034"
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

CURRENT_HEAD_OUTER_COMMIT: Final = "b96ef13ab327da245f4cf84e378a7fd89856631e"
CURRENT_HEAD_OUTER_TREE: Final = "e774965af04aeee69cfcbccbd6b6c40a8ec630f1"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '623dac8778c4930fe0e34f26f9dbbb13ae28904e' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "d9268f23fcda1cb6d3321960550f5d39709e806f"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "811f72effbe907d692d189e838cae58deb0bb463"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "d54a6494f50b5982758f8a6d2aafe24ccc7523b8"
CURRENT_HEAD_DATASETS_TREE: Final = "f7f82b0ccddc3ccb2628f04f26dd17f644ee53a6"
CURRENT_HEAD_KIT_COMMIT: Final = "a4fd25d944e6aaf25fb51e054a3f62bbcbea544d"
CURRENT_HEAD_KIT_TREE: Final = "caff018b9390cc55cdcb483d1c94f31b3ee93754"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_mutable_dependency_pinning.py",
)

PIN_MODULE_RELPATH: Final = "ipfs_accelerate_py/assurance/mutable_dependency_pins.py"
PROOF_CONTEXT_INPUTS_RELPATH: Final = "packaging/proof_context/locks/inputs.json"
KIT_INACTIVE_LIBP2P_FULL: Final = (
    "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main ; extra == \\\"full\\\""
)
KIT_INACTIVE_LIBP2P_LIBP2P: Final = (
    "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main ; extra == \\\"libp2p\\\""
)

SCAN_RELPATHS: Final[tuple[str, ...]] = (
    *QUALIFIED_PROFILE_RELPATHS,
    *RUNTIME_SPEC_RELPATHS,
)

GITLINK_EXPECTATIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("ipfs_kit_py", "portfolio", "external/ipfs_kit"),
    ("ipfs_datasets_py", "portfolio", "external/ipfs_datasets"),
    ("ipfs_transformers_py", "accelerate", "ipfs_transformers_py"),
    ("ipfs_model_manager_py", "accelerate", "ipfs_model_manager_py"),
)


class MutableDependencyPinningError(ValueError):
    """Malformed pinning evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class PinningProbe:
    """One measured or typed-unavailable pinning observation."""

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
class PinningVerdict:
    """Fail-closed PCPR-035 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    qualified_profiles_contain_mutable_branch: bool
    vcs_refs_are_immutable_commits: bool
    simulated_results_represented_as_live: bool
    live_vcs_install_qualified: bool
    live_vcs_install_evidence_kind: str
    live_libp2p_qualified: bool
    live_libp2p_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[PinningProbe, ...]
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
            "qualified_profiles_contain_mutable_branch": (
                self.qualified_profiles_contain_mutable_branch
            ),
            "vcs_refs_are_immutable_commits": self.vcs_refs_are_immutable_commits,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_vcs_install_qualified": self.live_vcs_install_qualified,
            "live_vcs_install_evidence_kind": self.live_vcs_install_evidence_kind,
            "live_libp2p_qualified": self.live_libp2p_qualified,
            "live_libp2p_evidence_kind": self.live_libp2p_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MutableDependencyPinningError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise MutableDependencyPinningError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise MutableDependencyPinningError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise MutableDependencyPinningError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise MutableDependencyPinningError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _git_gitlink(repo: Path | None, relpath: str) -> str | None:
    if repo is None or not repo.is_dir():
        return None
    try:
        completed = subprocess.run(
            [SEALED_GIT_BINARY, "-C", str(repo), "ls-tree", "HEAD", relpath],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = line[0].split()
    if len(parts) < 3 or parts[0] != "160000" or parts[1] != "commit":
        return None
    sha = parts[2]
    return sha if COMMIT_RE.fullmatch(sha) else None


def _scan_relpaths(
    root: Path, relpaths: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    hits: list[dict[str, Any]] = []
    for relpath in relpaths:
        if any(relpath.startswith(prefix) for prefix in EXCLUDED_RELPATH_PREFIXES):
            continue
        source = _read_source(root, relpath)
        if source is None:
            hits.append(
                {
                    "relpath": relpath,
                    "present": None,
                    "mutable": None,
                    "refs": (),
                }
            )
            continue
        mutable = mutable_git_references(source)
        hits.append(
            {
                "relpath": relpath,
                "present": True,
                "mutable": bool(mutable),
                "refs": tuple(
                    {
                        "distribution": item["distribution"],
                        "ref": item["ref"],
                        "url": item["url"],
                    }
                    for item in mutable
                ),
            }
        )
    return tuple(hits)


def _contains_pin(source: str | None, name: str) -> bool:
    if source is None:
        return False
    pin = PINS[name]
    token = f"{pin.git_url}@{pin.commit}"
    return token in source


def current_head_pinning_probes(
    *,
    accelerate_root: Path | None = None,
    portfolio_root: Path | None = None,
) -> tuple[PinningProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    portfolio = portfolio_root or discover_portfolio_root()
    probes: list[PinningProbe] = []
    if root is None or not root.is_dir():
        return (
            PinningProbe(
                probe_id="accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Accelerate source tree is not present and is not recorded as empty.",
            ),
        )

    pin_src = _read_source(root, PIN_MODULE_RELPATH)
    pin_ok = bool(pin_src) and all(
        COMMIT_RE.fullmatch(pin.commit) is not None and pin.live is False
        for pin in PINS.values()
    )
    probes.append(
        PinningProbe(
            probe_id="canonical_pin_table",
            present=None if pin_src is None else pin_ok,
            evidence_kind="unavailable" if pin_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Canonical pin table declares 40-character commits and no live claim."
                if pin_ok
                else "Canonical pin table is missing or still claims live qualification."
            ),
            details=MappingProxyType(
                {
                    "relpath": PIN_MODULE_RELPATH,
                    "pin_names": list(PINS),
                }
            ),
        )
    )

    pyproject = _read_source(root, "pyproject.toml")
    pyproject_ok = (
        _contains_pin(pyproject, "ipfs_transformers_py")
        and _contains_pin(pyproject, "ipfs_model_manager_py")
        and _contains_pin(pyproject, "libp2p")
        and not mutable_git_references(pyproject or "")
    )
    probes.append(
        PinningProbe(
            probe_id="pyproject_qualified_extras_pinned",
            present=None if pyproject is None else pyproject_ok,
            evidence_kind="unavailable" if pyproject is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "pyproject extras full/all/mcp-p2p/libp2p use immutable commits."
                if pyproject_ok
                else "pyproject extras still declare a mutable VCS branch."
            ),
            details=MappingProxyType({"relpath": "pyproject.toml"}),
        )
    )

    hf_src = _read_source(root, "requirements-hf-server.txt")
    hf_ok = (
        _contains_pin(hf_src, "ipfs_kit_py")
        and _contains_pin(hf_src, "ipfs_datasets_py")
        and not mutable_git_references(hf_src or "")
    )
    probes.append(
        PinningProbe(
            probe_id="hf_server_requirements_pinned",
            present=None if hf_src is None else hf_ok,
            evidence_kind="unavailable" if hf_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "HF-server requirements pin kit and datasets to immutable commits."
                if hf_ok
                else "HF-server requirements still declare a mutable VCS branch."
            ),
            details=MappingProxyType({"relpath": "requirements-hf-server.txt"}),
        )
    )

    scan = _scan_relpaths(root, SCAN_RELPATHS)
    missing = [item["relpath"] for item in scan if item["present"] is None]
    mutable_hits = [item for item in scan if item["mutable"] is True]
    profiles_ok = not missing and not mutable_hits
    probes.append(
        PinningProbe(
            probe_id="qualified_profiles_no_mutable_branch",
            present=None if missing and not scan else profiles_ok,
            evidence_kind="unavailable" if missing and len(missing) == len(scan) else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Qualified requirement profiles and runtime VCS specs contain no mutable branch."
                if profiles_ok
                else "A qualified profile or runtime spec still tracks a mutable VCS branch."
            ),
            details=MappingProxyType(
                {
                    "mutable_relpaths": [item["relpath"] for item in mutable_hits],
                    "missing_relpaths": missing,
                    "scanned": list(SCAN_RELPATHS),
                }
            ),
        )
    )

    runtime_src = _read_source(
        root, "ipfs_accelerate_py/mcplusplus_module/p2p/libp2p_runtime.py"
    )
    runtime_ok = _contains_pin(runtime_src, "libp2p") and (
        runtime_src is None or "PY_LIBP2P_PINNED_SPEC" in runtime_src
    )
    probes.append(
        PinningProbe(
            probe_id="runtime_libp2p_spec_pinned",
            present=None if runtime_src is None else runtime_ok,
            evidence_kind="unavailable" if runtime_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "MCP++ libp2p runtime spec is an immutable commit, not @main."
                if runtime_ok
                else "MCP++ libp2p runtime spec still tracks a mutable branch."
            ),
            details=MappingProxyType(
                {
                    "relpath": "ipfs_accelerate_py/mcplusplus_module/p2p/libp2p_runtime.py",
                    "pep508": pep508_spec("libp2p"),
                }
            ),
        )
    )

    llama_src = _read_source(
        root, "ipfs_accelerate_py/worker/skillset/libllama/requirements.txt"
    )
    llama_ok = _contains_pin(llama_src, "gguf-py") and not mutable_git_references(
        llama_src or ""
    )
    probes.append(
        PinningProbe(
            probe_id="llama_cpp_gguf_pin_is_immutable_commit",
            present=None if llama_src is None else llama_ok,
            evidence_kind="unavailable" if llama_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "llama.cpp gguf-py subdirectory is pinned to an immutable commit."
                if llama_ok
                else "llama.cpp gguf-py still tracks a mutable default branch."
            ),
            details=MappingProxyType(
                {
                    "relpath": "ipfs_accelerate_py/worker/skillset/libllama/requirements.txt",
                    "pin_kind": PINS["gguf-py"].pin_kind,
                }
            ),
        )
    )

    for name, owner, relpath in GITLINK_EXPECTATIONS:
        pin = PINS[name]
        repo = portfolio if owner == "portfolio" else root
        observed = _git_gitlink(repo, relpath)
        if observed is None:
            probes.append(
                PinningProbe(
                    probe_id=f"{name}_gitlink",
                    present=None,
                    evidence_kind="unavailable",
                    live=False,
                    simulated_represented_as_live=False,
                    reason=(
                        f"{name} gitlink could not be read and is not recorded as matching."
                    ),
                    details=MappingProxyType(
                        {
                            "owner": owner,
                            "relpath": relpath,
                            "pin_commit": pin.commit,
                            "pin_kind": pin.pin_kind,
                        }
                    ),
                )
            )
            continue
        matched = observed == pin.commit
        probes.append(
            PinningProbe(
                probe_id=f"{name}_gitlink",
                present=matched,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    f"{name} pin equals the {owner} gitlink."
                    if matched
                    else f"{name} pin drifted from the {owner} gitlink."
                ),
                details=MappingProxyType(
                    {
                        "owner": owner,
                        "relpath": relpath,
                        "pin_commit": pin.commit,
                        "gitlink": observed,
                        "pin_kind": pin.pin_kind,
                    }
                ),
            )
        )

    libp2p_pin = PINS["libp2p"]
    libp2p_ok = COMMIT_RE.fullmatch(libp2p_pin.commit) is not None
    probes.append(
        PinningProbe(
            probe_id="libp2p_declared_freeze",
            present=libp2p_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "libp2p is an in-tree declared freeze of a 40-character commit. "
                "Sealed validation does not re-fetch the remote and does not claim "
                "a live libp2p install."
            ),
            details=MappingProxyType(
                {
                    "commit": libp2p_pin.commit,
                    "pin_kind": libp2p_pin.pin_kind,
                    "pep508": libp2p_pin.pep508(),
                }
            ),
        )
    )

    proof_src = _read_source(root, PROOF_CONTEXT_INPUTS_RELPATH)
    proof_ok = bool(proof_src) and (
        KIT_INACTIVE_LIBP2P_FULL in proof_src
        and KIT_INACTIVE_LIBP2P_LIBP2P in proof_src
    )
    probes.append(
        PinningProbe(
            probe_id="proof_context_kit_inactive_declarations_retained",
            present=None if proof_src is None else proof_ok,
            evidence_kind="unavailable" if proof_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Kit Core Metadata inactive libp2p@main declarations remain recorded, "
                "not rewritten as Accelerate pins."
                if proof_ok
                else "Proof-context kit inactive declarations drifted."
            ),
            details=MappingProxyType({"relpath": PROOF_CONTEXT_INPUTS_RELPATH}),
        )
    )

    probes.append(
        PinningProbe(
            probe_id="live_vcs_install_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not install or qualify the pinned packages. Missing "
                "live VCS-install evidence stays typed unavailable and is not recorded "
                "as False or passing."
            ),
            details=MappingProxyType({}),
        )
    )
    probes.append(
        PinningProbe(
            probe_id="live_libp2p_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify live libp2p. A declared commit pin is not "
                "a live P2P, CUDA, CPU, or provider qualification."
            ),
            details=MappingProxyType({}),
        )
    )
    return tuple(probes)


def qualify_mutable_dependency_pinning(
    *,
    probes: Sequence[PinningProbe],
    duckdb_or_quack_state_written: bool = False,
    simulated_results_represented_as_live: bool = False,
    live_vcs_install_qualified: bool = False,
    live_libp2p_qualified: bool = False,
    this_task_created_competing_authority: bool = False,
) -> PinningVerdict:
    """Fail-closed pinning verdict. Never a closed release."""

    if duckdb_or_quack_state_written:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise MutableDependencyPinningError(
            "simulated pinning results cannot be represented as live"
        )
    if live_vcs_install_qualified or live_libp2p_qualified:
        raise MutableDependencyPinningError(
            "this task cannot claim live VCS-install or libp2p qualification"
        )
    if this_task_created_competing_authority:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning cannot mint a competing authority"
        )

    normalized: list[PinningProbe] = []
    for item in probes:
        if not isinstance(item, PinningProbe):
            raise MutableDependencyPinningError("probes must be PinningProbe values")
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise MutableDependencyPinningError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise MutableDependencyPinningError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise MutableDependencyPinningError(
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

    table_ok = _require("canonical_pin_table", "pin_table_incomplete")
    extras_ok = _require(
        "pyproject_qualified_extras_pinned", "pyproject_mutable_branch"
    )
    hf_ok = _require(
        "hf_server_requirements_pinned", "hf_server_mutable_branch"
    )
    profiles_ok = _require(
        "qualified_profiles_no_mutable_branch", "qualified_profile_mutable_branch"
    )
    runtime_ok = _require(
        "runtime_libp2p_spec_pinned", "runtime_libp2p_mutable_branch"
    )
    llama_ok = _require(
        "llama_cpp_gguf_pin_is_immutable_commit", "llama_cpp_unpinned"
    )
    _require("ipfs_kit_py_gitlink", "kit_pin_gitlink_mismatch")
    _require("ipfs_datasets_py_gitlink", "datasets_pin_gitlink_mismatch")
    _require(
        "ipfs_transformers_py_gitlink", "transformers_pin_gitlink_mismatch"
    )
    _require(
        "ipfs_model_manager_py_gitlink", "model_manager_pin_gitlink_mismatch"
    )
    _require("libp2p_declared_freeze", "libp2p_pin_not_immutable")
    _require(
        "proof_context_kit_inactive_declarations_retained",
        "proof_context_kit_declarations_rewritten",
    )

    immutable = table_ok and extras_ok and hf_ok and profiles_ok and runtime_ok and llama_ok
    mutable_remaining = not profiles_ok

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
        raise MutableDependencyPinningError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not mint a closed release outcome"
        )

    payload = {
        "schema": PINNING_VERDICT_SCHEMA,
        "interface": PINNING_EVALUATOR_INTERFACE,
        "task_id": PCPR_035_TASK_ID,
        "goal_id": PCPR_035_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "qualified_profiles_contain_mutable_branch": mutable_remaining,
        "vcs_refs_are_immutable_commits": immutable,
        "simulated_results_represented_as_live": False,
        "live_vcs_install_qualified": False,
        "live_vcs_install_evidence_kind": "unavailable",
        "live_libp2p_qualified": False,
        "live_libp2p_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return PinningVerdict(
        schema=PINNING_VERDICT_SCHEMA,
        interface=PINNING_EVALUATOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        qualified_profiles_contain_mutable_branch=mutable_remaining,
        vcs_refs_are_immutable_commits=immutable,
        simulated_results_represented_as_live=False,
        live_vcs_install_qualified=False,
        live_vcs_install_evidence_kind="unavailable",
        live_libp2p_qualified=False,
        live_libp2p_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_pinning() -> PinningVerdict:
    """Ordinary current-head PCPR-035 evaluation: honest R&D pinning."""

    return qualify_mutable_dependency_pinning(probes=current_head_pinning_probes())


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraglv6z75m4jv2zena6bco62zitwsk4cdep6fqrtfrvped3cyc4naa"
)


def pcpr_035_receipt_promotion(verdict: PinningVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise MutableDependencyPinningError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise MutableDependencyPinningError(
            "promotion_status is not an admitted PCPR-035 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise MutableDependencyPinningError(
            "mutable-dependency pinning cannot promote the supervisor"
        )
    if verdict.live_vcs_install_qualified or verdict.live_libp2p_qualified:
        raise MutableDependencyPinningError(
            "this task cannot claim live VCS-install or libp2p qualification"
        )
    if verdict.qualified_profiles_contain_mutable_branch:
        raise MutableDependencyPinningError(
            "qualified profiles must not contain a mutable VCS branch"
        )
    return {
        "schema": PINNING_VERDICT_SCHEMA,
        "interface": PINNING_EVALUATOR_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "qualified_profiles_contain_mutable_branch": False,
        "vcs_refs_are_immutable_commits": verdict.vcs_refs_are_immutable_commits,
        "simulated_results_represented_as_live": False,
        "live_vcs_install_qualified": False,
        "live_vcs_install_evidence_kind": "unavailable",
        "live_libp2p_qualified": False,
        "live_libp2p_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_035_receipt_promotion() -> dict[str, Any]:
    return pcpr_035_receipt_promotion(qualify_current_head_pinning())


def pcpr_035_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_vcs_install_not_claimed": True,
        "live_libp2p_not_claimed": True,
        "mutable_branch_not_retained_in_qualified_profiles": True,
        "declared_freeze_not_live_qualification": True,
        "kit_core_metadata_not_rewritten": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_035_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_pinning()
    promotion = pcpr_035_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_034_TASK_ID,
            "goal_id": "PCPR-G410",
            "promotion_status": "rnd_non_promoted",
            "removal_verdict_cid": PCPR_034_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-034 consolidated the capability ladder. This task pins mutable "
                "Git dependencies. PCPR-001 live qualification remains "
                "rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "pinning": {
            "schema": PINNING_EVALUATOR_SCHEMA,
            "interface": PINNING_EVALUATOR_INTERFACE,
            "pin_schema": PIN_SCHEMA,
            "pin_interface": PIN_INTERFACE,
            "pin_task_id": PIN_TASK_ID,
            "qualified_profiles_contain_mutable_branch": False,
            "vcs_refs_are_immutable_commits": verdict.vcs_refs_are_immutable_commits,
            "pins": {name: pin.to_mapping() for name, pin in PINS.items()},
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_035_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_035_current_tree_binding(
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
        raise MutableDependencyPinningError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise MutableDependencyPinningError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise MutableDependencyPinningError("kit_commit must equal kit_gitlink")
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


def current_head_pcpr_035_current_tree_binding() -> dict[str, Any]:
    return pcpr_035_current_tree_binding(
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


def validate_pcpr_035_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-035 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise MutableDependencyPinningError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_035_TASK_ID:
        raise MutableDependencyPinningError("outer receipt task_id must be PCPR-035")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise MutableDependencyPinningError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise MutableDependencyPinningError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise MutableDependencyPinningError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise MutableDependencyPinningError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise MutableDependencyPinningError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise MutableDependencyPinningError(
            "mutable-dependency pinning cannot freeze contracts"
        )
    if verdict_section.get("live_vcs_install_qualified") is True:
        raise MutableDependencyPinningError(
            "this task cannot claim live VCS-install qualification"
        )
    if verdict_section.get("live_libp2p_qualified") is True:
        raise MutableDependencyPinningError(
            "this task cannot claim live libp2p qualification"
        )
    if verdict_section.get("qualified_profiles_contain_mutable_branch") is True:
        raise MutableDependencyPinningError(
            "qualified profiles must not contain a mutable VCS branch"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise MutableDependencyPinningError(
            "qualification_verdict.promotion_status is not an admitted PCPR-035 status"
        )
    if promotion_status == "supervisor_promoted":
        raise MutableDependencyPinningError(
            "mutable-dependency pinning cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise MutableDependencyPinningError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise MutableDependencyPinningError("acceptance must not claim a release")

    expected = current_head_pcpr_035_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise MutableDependencyPinningError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise MutableDependencyPinningError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise MutableDependencyPinningError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_035_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }
