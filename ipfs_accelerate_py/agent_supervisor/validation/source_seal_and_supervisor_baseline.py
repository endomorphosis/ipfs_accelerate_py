"""Fail-closed PCPR source-seal and supervisor-baseline recorder.

PCPR-000 records the exact current-head portfolio and the three source
repositories, the immutable bootstrap-input identities, the current policy
payloads, and the current public supervisor-contract catalog.  This module is
not release authority: it does not write DuckDB or Quack state, does not freeze
contracts (PCPR-002), does not qualify the supervisor (PCPR-001), and never
emits a closed PCPR release outcome.

Live claims require measured evidence.  Recursive sibling submodules are not
source authority.  Missing origin/main, missing repositories, or unreadable
bootstrap files stay typed unavailable.  A gitlink mismatch without an
explicit isolated-source binding is typed blocked.  Simulation cannot mint a
clean seal.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


SOURCE_SEAL_INTERFACE: Final = "SourceSealAndSupervisorBaseline@1"
SOURCE_SEAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/source-seal-and-supervisor-baseline@1"
)
SOURCE_SEAL_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/source-seal-and-supervisor-baseline-verdict@1"
)

PCPR_000_TASK_ID: Final = "PCPR-000"
PCPR_000_GOAL_ID: Final = "PCPR-G110"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"
REQUIRED_CAMPAIGN_BRANCH: Final = (
    "agent/proof-carrying-platform-qualification-and-release-v1"
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
BINDING_KINDS: Final[frozenset[str]] = frozenset(
    {
        "gitlink",
        "isolated_source",
        "isolated_implementation_worktree",
        "unavailable",
        "blocked",
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
CLOSED_RELEASE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "release_candidate_qualified",
        "non_promoted_supervisor_unqualified",
        "non_promoted_import_or_false_success",
        "non_promoted_live_storage_gap",
        "non_promoted_live_compute_gap",
        "non_promoted_solver_gap",
        "non_promoted_packaging_gap",
        "non_promoted_dependency_reproducibility",
        "non_promoted_security_failure",
        "non_promoted_interoperability_gap",
        "non_promoted_reference_workflow_failure",
        "non_promoted_unmeasured",
        "non_promoted_operator_gate_required",
    }
)

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
SEALED_GIT_BINARY: Final = "/usr/bin/git"
SEALED_PYTHON: Final = "/usr/bin/python3.12"
SEALED_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

SOURCE_REPOSITORIES: Final[tuple[tuple[str, str], ...]] = (
    ("ipfs_accelerate_py", "external/ipfs_accelerate"),
    ("ipfs_datasets_py", "external/ipfs_datasets"),
    ("ipfs_kit_py", "external/ipfs_kit"),
)
SOURCE_AUTHORITY_PATHS: Final[frozenset[str]] = frozenset(
    path for _name, path in SOURCE_REPOSITORIES
)
SOURCE_AUTHORITY_COUNT: Final = 3

IMMUTABLE_BOOTSTRAP_INPUTS: Final[tuple[str, ...]] = (
    "docs/architecture/PROOF_CARRYING_PLATFORM_QUALIFICATION_AND_RELEASE_V1_PLAN.md",
    "docs/architecture/proof_carrying_platform_qualification_and_release_v1.objectives.md",
    "docs/architecture/proof_carrying_platform_qualification_and_release_v1.todo.md",
    "config/proof_carrying_platform_qualification_and_release_v1_supervisor.json",
    "scripts/generate_proof_carrying_platform_qualification_and_release_board.py",
    "scripts/validate_proof_carrying_platform_qualification_and_release_board.py",
    "scripts/run_agent_supervisor_proof_carrying_platform_qualification_and_release.py",
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
)

POLICY_BASELINE_KEYS: Final[tuple[str, ...]] = (
    "authority_policy",
    "budget_policy",
    "refill_policy",
    "source_binding",
    "hard_zero_invariants",
    "closed_release_outcomes",
    "operational_control_plane",
    "objective_submission",
    "initial_projection",
)

# Shared contracts named by the PCPR plan.  Recording them here is a
# baseline inventory, not a freeze (PCPR-002) and not normative
# stabilization (PCPR-040).
CANONICAL_CONTRACT_CATALOG: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "name": "SupervisorObjectiveIntent",
            "schema": "not_yet_normative",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ObjectiveMaterializationReceipt",
            "schema": "not_yet_normative",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "SupervisorContextPack",
            "schema": "ipfs_datasets_py.proof_context.context_pack",
            "authority": "ipfs_datasets_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "SemanticArtifactIdentity",
            "schema": "not_yet_normative",
            "authority": "ipfs_datasets_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "DurableArtifactReceipt",
            "schema": "not_yet_normative",
            "authority": "ipfs_kit_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ProofObligation",
            "schema": "ipfs_accelerate_py/agent-supervisor/code-proof-obligation@1",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ProofResult",
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-receipt@1",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ProofAdmissionDecision",
            "schema": "not_yet_normative",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ExecutionInvocation",
            "schema": "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-request@1",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ExecutionReceipt",
            "schema": "ipfs_accelerate_py/agent-supervisor/entrypoints/invocation-result@1",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "SupervisorEvent",
            "schema": "not_yet_normative",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "TaskStateTransition",
            "schema": "not_yet_normative",
            "authority": "ipfs_accelerate_py",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "ReleaseComponentManifest",
            "schema": "not_yet_normative",
            "authority": "portfolio",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
    MappingProxyType(
        {
            "name": "PortfolioCompatibilityManifest",
            "schema": "not_yet_normative",
            "authority": "portfolio",
            "disposition": "baseline_recorded_not_frozen",
            "normative_task": "PCPR-040",
            "freeze_task": "PCPR-002",
        }
    ),
)

SUPERVISOR_BASELINE_SURFACES: Final[tuple[str, ...]] = (
    "objective",
    "task",
    "event",
    "contextpack",
    "state_machine",
    "receipt",
)

PLANNING_SOURCE_FOREST: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "ipfs_accelerate_py": MappingProxyType(
            {
                "path": "external/ipfs_accelerate",
                "commit": "0bc65cad008b96a563978bfe53e3b19a1481b491",
                "tree": "48be2c0391e4f3081701a291e67b0d39975b0abe",
                "origin_main": "f8c2f633fa6a781b822176fd63e1a229f96b581c",
            }
        ),
        "ipfs_datasets_py": MappingProxyType(
            {
                "path": "external/ipfs_datasets",
                "commit": "f49afc579c22856849ca9f739435e5820003384f",
                "tree": "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b",
                "origin_main": "f49afc579c22856849ca9f739435e5820003384f",
            }
        ),
        "ipfs_kit_py": MappingProxyType(
            {
                "path": "external/ipfs_kit",
                "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
                "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
                "origin_main": "b6c65ba732733d7e33852713ba18aa3b12235668",
            }
        ),
    }
)
PLANNING_OUTER_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"

# Measured current-head identities for this isolated implementation worktree.
# Post-change Accelerate commit/tree remain pending the admitted nested commit.
CURRENT_HEAD_OUTER_COMMIT: Final = "41dd92e823ecc10be2925f92ef7ce07c494c78ce"
CURRENT_HEAD_OUTER_TREE: Final = "6de8abf6205711391b5a067ebdd11ec5c1f536c8"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "fix(pcpr): reseal accelerate planning pins to the current gitlink"
)
CURRENT_HEAD_OUTER_BRANCH: Final = (
    "implementation/pcpr-000-c12208769e34-attempt-1-1788186940"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = PLANNING_OUTER_ORIGIN_MAIN
CURRENT_HEAD_FIRST_PARENT: Final = "21ee07e8402bbf99ea77b7c877f9b61790853d0b"
CURRENT_HEAD_PORTFOLIO_GITLINK_DIGEST: Final = (
    "10443ff9b3cb82728f43df26373a8c087616f38d7cb446a801f32226deb978f0"
)
CURRENT_HEAD_PORTFOLIO_GITLINK_COUNT: Final = 8
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "0bc65cad008b96a563978bfe53e3b19a1481b491"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "48be2c0391e4f3081701a291e67b0d39975b0abe"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_ACCELERATOR_RECURSIVE_GITLINK_DIGEST: Final = (
    "5c66dfd35fea62b100b6f3f82a440ce44cdbd0846f08697b8f537f3cfa0a242a"
)
CURRENT_HEAD_ACCELERATOR_RECURSIVE_GITLINK_COUNT: Final = 10
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_DATASETS_RECURSIVE_GITLINK_DIGEST: Final = (
    "e32cc04bcc451c823b2cd47b63761ed391d18798accc284db43de02c83acbfe6"
)
CURRENT_HEAD_DATASETS_RECURSIVE_GITLINK_COUNT: Final = 10
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"
CURRENT_HEAD_KIT_RECURSIVE_GITLINK_DIGEST: Final = (
    "cc901e4cabfc3d004a9d51b836744eeebbfa830a226dfb4b583c758f02e63835"
)
CURRENT_HEAD_KIT_RECURSIVE_GITLINK_COUNT: Final = 11

CURRENT_HEAD_BOOTSTRAP_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "docs/architecture/PROOF_CARRYING_PLATFORM_QUALIFICATION_AND_RELEASE_V1_PLAN.md": (
            "14a954145b43a8bb112cbfa2976b53d5f82ca3d16fad9669b7616c15411f896c"
        ),
        "docs/architecture/proof_carrying_platform_qualification_and_release_v1.objectives.md": (
            "aadcde1450101f4a9c1ad1402240b5f33b027fba4fca4494765b61b418a3ed4f"
        ),
        "docs/architecture/proof_carrying_platform_qualification_and_release_v1.todo.md": (
            "7f59b35a3643aa94da8b04136029da7c7e2f6eeaa2837a9caa077714ea7102dc"
        ),
        "config/proof_carrying_platform_qualification_and_release_v1_supervisor.json": (
            "c90403fe66226aa70601dbbd92c36d8a66af5f40666c609ff95cc4a8f9d0c955"
        ),
        "scripts/generate_proof_carrying_platform_qualification_and_release_board.py": (
            "6bbf12aaeaaa8912ccbd351e89160f3e381c43d845a1c12771669991642dea6e"
        ),
        "scripts/validate_proof_carrying_platform_qualification_and_release_board.py": (
            "57701d90ba9d9d6477a603de3ebb76391ee4de08a536e2b98667a0ed803bccfd"
        ),
        "scripts/run_agent_supervisor_proof_carrying_platform_qualification_and_release.py": (
            "a6d24acb00d1dcc14468b791302c34b861b1682cc5478ff4c8108d610dc85055"
        ),
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py": (
            "96e675c5030378db25a2b630c234ad5faa040b4758eb7f6d06873329c288c6dc"
        ),
    }
)
CURRENT_HEAD_BOOTSTRAP_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "docs/architecture/PROOF_CARRYING_PLATFORM_QUALIFICATION_AND_RELEASE_V1_PLAN.md": 35810,
        "docs/architecture/proof_carrying_platform_qualification_and_release_v1.objectives.md": 35267,
        "docs/architecture/proof_carrying_platform_qualification_and_release_v1.todo.md": 21468,
        "config/proof_carrying_platform_qualification_and_release_v1_supervisor.json": 14294,
        "scripts/generate_proof_carrying_platform_qualification_and_release_board.py": 44409,
        "scripts/validate_proof_carrying_platform_qualification_and_release_board.py": 22133,
        "scripts/run_agent_supervisor_proof_carrying_platform_qualification_and_release.py": 91483,
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py": 939,
    }
)
CURRENT_HEAD_POLICY_CIDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "authority_policy": (
            "baguqeerairccawduzmr7iactge37yt7nfhm7zetldupsiw5uwr6mrl52iboq"
        ),
        "budget_policy": (
            "baguqeeraslbwlr6pdmzdojf52v4r2alj47kq2nalr5duuuysfwgvcfwivgla"
        ),
        "refill_policy": (
            "baguqeerasdbkl52utrlsb33wllwghdbtfx5wefdzu52qd4y72pelqo66mxna"
        ),
        "source_binding": (
            "baguqeerajq4vxhiolojsxbjqisfgvonztj4qyfctxsj5pvj4tdzrsuntrshq"
        ),
        "hard_zero_invariants": (
            "baguqeeraunkbkxcrd64sluvw6exipghhtqhg4knrh4syxcnoiw73bzqt4vzq"
        ),
        "closed_release_outcomes": (
            "baguqeeral2dkojytoeynbbebedrw5v7usvzxt43oa3mz4qbu546xulapzpxq"
        ),
        "operational_control_plane": (
            "baguqeerajicfyjt2cxv4hu346jkiubkcfi3zteznwdusqy2o2dtylgnjjl7a"
        ),
        "objective_submission": (
            "baguqeeraywj7dz4dtssioeqtadp3zbvwprqyhlmnjfmzhnpdb7ty37cryyfa"
        ),
        "initial_projection": (
            "baguqeeraap7mrdwjq4qwpm6i5aetwfzjmsb5ogu2ek3rtzusqo6ct3zx6inq"
        ),
    }
)
CURRENT_HEAD_POLICY_BUNDLE_CID: Final = (
    "baguqeerau5ns73vihqq6u46yex3og66q536se4fkvbtlzd77ctdq7ys7qsua"
)

# Pinned identity of the ordinary clean-gitlink current-head baseline
# verdict.  Drift means the default payload changed and the outer receipt
# must be regenerated from this evaluator.
CURRENT_HEAD_CLEAN_FOREST_VERDICT_CID: Final = (
    "baguqeerakcgqdihfy4v5gp6z3st52lc35yek54gunvt2mnldp6vybklfdodq"
)


class SourceSealError(ValueError):
    """Malformed source-seal evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class RepositoryObservation:
    """Measured or typed-unavailable observation of one Git worktree."""

    repository: str
    path: str
    commit: str | None
    tree: str | None
    origin_main: str | None
    origin_main_is_ancestor: bool | None
    gitlink: str | None
    clean: bool | None
    exact_toplevel: bool | None
    evidence_kind: str
    reason: str = ""
    branch: str = ""
    recursive_gitlink_count: int | None = None
    recursive_gitlink_digest: str | None = None
    isolated_source_binding: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "path": self.path,
            "commit": self.commit,
            "tree": self.tree,
            "origin_main": self.origin_main,
            "origin_main_is_ancestor": self.origin_main_is_ancestor,
            "gitlink": self.gitlink,
            "clean": self.clean,
            "exact_toplevel": self.exact_toplevel,
            "evidence_kind": self.evidence_kind,
            "reason": self.reason,
            "branch": self.branch,
            "recursive_gitlink_count": self.recursive_gitlink_count,
            "recursive_gitlink_digest": self.recursive_gitlink_digest,
            "isolated_source_binding": self.isolated_source_binding,
        }


@dataclass(frozen=True)
class BootstrapArtifact:
    """Content identity of one immutable bootstrap-input file."""

    path: str
    sha256: str | None
    bytes: int | None
    evidence_kind: str
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "evidence_kind": self.evidence_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyBaseline:
    """Content identity of one scheduler policy object."""

    key: str
    content_cid: str | None
    evidence_kind: str
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content_cid": self.content_cid,
            "evidence_kind": self.evidence_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvaluatedRepository:
    """Fail-closed evaluation of one source or portfolio observation."""

    repository: str
    path: str
    binding_kind: str
    source_authority: bool
    planning_pin_match: bool | None
    changed_source_seal: bool | None
    blockers: tuple[str, ...]
    observation: RepositoryObservation

    def to_mapping(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "path": self.path,
            "binding_kind": self.binding_kind,
            "source_authority": self.source_authority,
            "planning_pin_match": self.planning_pin_match,
            "changed_source_seal": self.changed_source_seal,
            "blockers": list(self.blockers),
            "observation": self.observation.to_mapping(),
        }


@dataclass(frozen=True)
class SourceSealVerdict:
    """Fail-closed PCPR-000 source-seal and baseline decision."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    changed_source_seal_requires_fresh_inventory: bool
    source_authority_count: int
    portfolio: EvaluatedRepository
    sources: tuple[EvaluatedRepository, ...]
    bootstrap_artifacts: tuple[BootstrapArtifact, ...]
    policies: tuple[PolicyBaseline, ...]
    contract_catalog: tuple[Mapping[str, str], ...]
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
            "changed_source_seal_requires_fresh_inventory": (
                self.changed_source_seal_requires_fresh_inventory
            ),
            "source_authority_count": self.source_authority_count,
            "portfolio": self.portfolio.to_mapping(),
            "sources": [item.to_mapping() for item in self.sources],
            "bootstrap_artifacts": [item.to_mapping() for item in self.bootstrap_artifacts],
            "policies": [item.to_mapping() for item in self.policies],
            "contract_catalog": [dict(item) for item in self.contract_catalog],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceSealError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise SourceSealError(f"{name} is not an admitted evidence kind")
    return kind


def _optional_commit(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise SourceSealError(f"{name} must be a 40-character lowercase commit")
    return text


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name)
    if SHA256_RE.fullmatch(text) is None:
        raise SourceSealError(f"{name} must be a 64-character lowercase sha256")
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise SourceSealError(f"{name} must not be a closed PCPR release outcome")


def _normalize_observation(item: RepositoryObservation) -> RepositoryObservation:
    repository = _text(item.repository, "repository")
    path = _text(item.path, "path")
    kind = _kind(item.evidence_kind, f"{repository}.evidence_kind")
    if kind == "simulated" and item.clean is True:
        raise SourceSealError("simulated cleanliness cannot mint a source seal")
    commit = _optional_commit(item.commit, f"{repository}.commit") if item.commit else None
    tree = _optional_commit(item.tree, f"{repository}.tree") if item.tree else None
    origin_main = (
        _optional_commit(item.origin_main, f"{repository}.origin_main")
        if item.origin_main
        else None
    )
    gitlink = _optional_commit(item.gitlink, f"{repository}.gitlink") if item.gitlink else None
    if item.recursive_gitlink_digest not in (None, ""):
        _optional_sha256(
            item.recursive_gitlink_digest, f"{repository}.recursive_gitlink_digest"
        )
    if item.isolated_source_binding:
        _text(item.isolated_source_binding, f"{repository}.isolated_source_binding")
    return RepositoryObservation(
        repository=repository,
        path=path,
        commit=commit,
        tree=tree,
        origin_main=origin_main,
        origin_main_is_ancestor=item.origin_main_is_ancestor,
        gitlink=gitlink,
        clean=item.clean,
        exact_toplevel=item.exact_toplevel,
        evidence_kind=kind,
        reason=str(item.reason or ""),
        branch=str(item.branch or ""),
        recursive_gitlink_count=item.recursive_gitlink_count,
        recursive_gitlink_digest=item.recursive_gitlink_digest or None,
        isolated_source_binding=str(item.isolated_source_binding or ""),
    )


def _evaluate_source(item: RepositoryObservation) -> EvaluatedRepository:
    observation = _normalize_observation(item)
    blockers: list[str] = []
    planning = PLANNING_SOURCE_FOREST.get(observation.repository)
    if observation.path not in SOURCE_AUTHORITY_PATHS:
        raise SourceSealError(
            f"{observation.repository} path is not a PCPR source-authority repository"
        )
    if planning is not None and observation.path != planning["path"]:
        raise SourceSealError(f"{observation.repository} path does not match the planning pin")

    if observation.evidence_kind == "unavailable" or observation.commit is None:
        binding = "unavailable"
        blockers.append(f"{observation.repository}:unavailable")
        pin_match: bool | None = None
        changed: bool | None = None
    else:
        if observation.exact_toplevel is False:
            blockers.append(f"{observation.repository}:not_exact_toplevel")
        if observation.origin_main is None or observation.origin_main_is_ancestor is None:
            blockers.append(f"{observation.repository}:origin_main_unavailable")
        gitlink_match = (
            observation.gitlink is not None and observation.commit == observation.gitlink
        )
        if gitlink_match:
            binding = "gitlink"
        elif observation.isolated_source_binding:
            binding = "isolated_source"
        else:
            binding = "blocked"
            blockers.append(
                f"{observation.repository}:gitlink_mismatch_without_isolated_source_binding"
            )
        if observation.origin_main_is_ancestor is False:
            blockers.append(f"{observation.repository}:origin_main_not_ancestor")
            binding = "blocked"
        if planning is None:
            pin_match = None
            changed = None
        else:
            pin_match = (
                observation.commit == planning["commit"]
                and observation.tree == planning["tree"]
            )
            changed = not pin_match

    if binding not in BINDING_KINDS:
        raise SourceSealError(f"{observation.repository} binding_kind is not admitted")
    return EvaluatedRepository(
        repository=observation.repository,
        path=observation.path,
        binding_kind=binding,
        source_authority=True,
        planning_pin_match=pin_match,
        changed_source_seal=changed,
        blockers=tuple(dict.fromkeys(blockers)),
        observation=observation,
    )


def _evaluate_portfolio(item: RepositoryObservation) -> EvaluatedRepository:
    observation = _normalize_observation(item)
    blockers: list[str] = []
    if observation.evidence_kind == "unavailable" or observation.commit is None:
        binding = "unavailable"
        blockers.append("portfolio:unavailable")
    else:
        if observation.origin_main is None or observation.origin_main_is_ancestor is None:
            blockers.append("portfolio:origin_main_unavailable")
        if observation.branch == REQUIRED_CAMPAIGN_BRANCH:
            binding = "gitlink"
        elif observation.isolated_source_binding or observation.branch.startswith(
            "implementation/"
        ):
            binding = "isolated_implementation_worktree"
        else:
            binding = "isolated_source"
        if observation.origin_main_is_ancestor is False:
            blockers.append("portfolio:origin_main_not_ancestor")
            binding = "blocked"
    return EvaluatedRepository(
        repository=observation.repository,
        path=observation.path,
        binding_kind=binding,
        source_authority=False,
        planning_pin_match=None,
        changed_source_seal=None,
        blockers=tuple(dict.fromkeys(blockers)),
        observation=observation,
    )


def _normalize_bootstrap(
    artifacts: Sequence[BootstrapArtifact],
) -> tuple[BootstrapArtifact, ...]:
    seen: set[str] = set()
    normalized: list[BootstrapArtifact] = []
    for path in IMMUTABLE_BOOTSTRAP_INPUTS:
        match = next((item for item in artifacts if item.path == path), None)
        if match is None:
            normalized.append(
                BootstrapArtifact(
                    path=path,
                    sha256=None,
                    bytes=None,
                    evidence_kind="unavailable",
                    reason="Immutable bootstrap input was not observed.",
                )
            )
            continue
        kind = _kind(match.evidence_kind, f"{path}.evidence_kind")
        digest = _optional_sha256(match.sha256, f"{path}.sha256") if match.sha256 else None
        if kind == "measured" and digest is None:
            raise SourceSealError(f"{path} measured bootstrap artifact needs sha256")
        if path in seen:
            raise SourceSealError(f"duplicate bootstrap artifact: {path}")
        seen.add(path)
        normalized.append(
            BootstrapArtifact(
                path=path,
                sha256=digest,
                bytes=match.bytes,
                evidence_kind=kind,
                reason=str(match.reason or ""),
            )
        )
    extra = [item.path for item in artifacts if item.path not in IMMUTABLE_BOOTSTRAP_INPUTS]
    if extra:
        raise SourceSealError(f"undeclared bootstrap artifact: {extra[0]}")
    return tuple(normalized)


def _normalize_policies(policies: Sequence[PolicyBaseline]) -> tuple[PolicyBaseline, ...]:
    by_key = {item.key: item for item in policies}
    normalized: list[PolicyBaseline] = []
    for key in POLICY_BASELINE_KEYS:
        item = by_key.get(key)
        if item is None:
            normalized.append(
                PolicyBaseline(
                    key=key,
                    content_cid=None,
                    evidence_kind="unavailable",
                    reason="Scheduler policy object was not observed.",
                )
            )
            continue
        kind = _kind(item.evidence_kind, f"{key}.evidence_kind")
        cid = str(item.content_cid).strip() if item.content_cid else None
        if kind == "measured":
            if not cid or not cid.startswith("baguqeera"):
                raise SourceSealError(f"{key} measured policy needs a CIDv1 identity")
        normalized.append(
            PolicyBaseline(
                key=key,
                content_cid=cid,
                evidence_kind=kind,
                reason=str(item.reason or ""),
            )
        )
    extra = [item.key for item in policies if item.key not in POLICY_BASELINE_KEYS]
    if extra:
        raise SourceSealError(f"undeclared policy baseline key: {extra[0]}")
    return tuple(normalized)


def policy_content_cid(payload: Any) -> str:
    """Return the CIDv1 identity of one scheduler policy object."""

    return content_identity(payload)


def seal_source_forest_and_supervisor_baseline(
    *,
    portfolio: RepositoryObservation,
    sources: Sequence[RepositoryObservation],
    bootstrap_artifacts: Sequence[BootstrapArtifact],
    policies: Sequence[PolicyBaseline],
    duckdb_or_quack_state_written: bool = False,
) -> SourceSealVerdict:
    """Evaluate the PCPR-000 source seal.  Promotion is fail-closed R&D."""

    if duckdb_or_quack_state_written:
        raise SourceSealError("source seal must not write DuckDB or Quack state")

    evaluated_portfolio = _evaluate_portfolio(portfolio)
    if len(sources) != SOURCE_AUTHORITY_COUNT:
        raise SourceSealError("exactly three source-authority repositories are required")
    expected_names = [name for name, _path in SOURCE_REPOSITORIES]
    observed_names = [item.repository for item in sources]
    if observed_names != expected_names:
        raise SourceSealError("source repositories must appear in the sealed order")

    evaluated_sources = tuple(_evaluate_source(item) for item in sources)
    bootstrap = _normalize_bootstrap(bootstrap_artifacts)
    policy_records = _normalize_policies(policies)

    blockers: list[str] = list(evaluated_portfolio.blockers)
    for source in evaluated_sources:
        blockers.extend(source.blockers)
    if any(item.evidence_kind == "unavailable" for item in bootstrap):
        blockers.append("bootstrap:unavailable")
    if any(item.evidence_kind == "unavailable" for item in policy_records):
        blockers.append("policies:unavailable")
    unique_blockers = tuple(dict.fromkeys(blockers))

    changed = any(item.changed_source_seal is True for item in evaluated_sources)
    evaluated_all = (evaluated_portfolio, *evaluated_sources)
    missing = any(item.binding_kind == "unavailable" for item in evaluated_all)
    hard_block = any(item.binding_kind == "blocked" for item in evaluated_all)

    if hard_block:
        promotion_status = "typed_blocked"
    elif missing:
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise SourceSealError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise SourceSealError("source seal must not mint a closed release outcome")

    payload = {
        "schema": SOURCE_SEAL_VERDICT_SCHEMA,
        "interface": SOURCE_SEAL_INTERFACE,
        "task_id": PCPR_000_TASK_ID,
        "goal_id": PCPR_000_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "changed_source_seal_requires_fresh_inventory": changed,
        "source_authority_count": SOURCE_AUTHORITY_COUNT,
        "portfolio": evaluated_portfolio.to_mapping(),
        "sources": [item.to_mapping() for item in evaluated_sources],
        "bootstrap_artifacts": [item.to_mapping() for item in bootstrap],
        "policies": [item.to_mapping() for item in policy_records],
        "contract_catalog": [dict(item) for item in CANONICAL_CONTRACT_CATALOG],
        "blockers": list(unique_blockers),
    }
    verdict_cid = content_identity(payload)
    return SourceSealVerdict(
        schema=SOURCE_SEAL_VERDICT_SCHEMA,
        interface=SOURCE_SEAL_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        changed_source_seal_requires_fresh_inventory=changed,
        source_authority_count=SOURCE_AUTHORITY_COUNT,
        portfolio=evaluated_portfolio,
        sources=evaluated_sources,
        bootstrap_artifacts=bootstrap,
        policies=policy_records,
        contract_catalog=CANONICAL_CONTRACT_CATALOG,
        blockers=unique_blockers,
        verdict_cid=verdict_cid,
    )


def current_head_portfolio_observation() -> RepositoryObservation:
    """Measured isolated-worktree observation of the current outer HEAD."""

    return RepositoryObservation(
        repository="portfolio",
        path=".",
        commit=CURRENT_HEAD_OUTER_COMMIT,
        tree=CURRENT_HEAD_OUTER_TREE,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        gitlink=None,
        clean=True,
        exact_toplevel=True,
        evidence_kind="measured",
        reason="Isolated implementation worktree of the sealed PCPR portfolio.",
        branch=CURRENT_HEAD_OUTER_BRANCH,
        recursive_gitlink_count=CURRENT_HEAD_PORTFOLIO_GITLINK_COUNT,
        recursive_gitlink_digest=CURRENT_HEAD_PORTFOLIO_GITLINK_DIGEST,
        isolated_source_binding="isolated_implementation_worktree",
    )


def current_head_source_observations() -> tuple[RepositoryObservation, ...]:
    """Measured gitlink-bound observations of the three source repositories."""

    return (
        RepositoryObservation(
            repository="ipfs_accelerate_py",
            path="external/ipfs_accelerate",
            commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
            tree=CURRENT_HEAD_ACCELERATOR_TREE,
            origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
            origin_main_is_ancestor=True,
            gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
            clean=True,
            exact_toplevel=True,
            evidence_kind="measured",
            reason="Nested HEAD equals the portfolio gitlink.",
            recursive_gitlink_count=CURRENT_HEAD_ACCELERATOR_RECURSIVE_GITLINK_COUNT,
            recursive_gitlink_digest=CURRENT_HEAD_ACCELERATOR_RECURSIVE_GITLINK_DIGEST,
        ),
        RepositoryObservation(
            repository="ipfs_datasets_py",
            path="external/ipfs_datasets",
            commit=CURRENT_HEAD_DATASETS_COMMIT,
            tree=CURRENT_HEAD_DATASETS_TREE,
            origin_main=CURRENT_HEAD_DATASETS_COMMIT,
            origin_main_is_ancestor=True,
            gitlink=CURRENT_HEAD_DATASETS_COMMIT,
            clean=True,
            exact_toplevel=True,
            evidence_kind="measured",
            reason="Nested HEAD equals the portfolio gitlink.",
            recursive_gitlink_count=CURRENT_HEAD_DATASETS_RECURSIVE_GITLINK_COUNT,
            recursive_gitlink_digest=CURRENT_HEAD_DATASETS_RECURSIVE_GITLINK_DIGEST,
        ),
        RepositoryObservation(
            repository="ipfs_kit_py",
            path="external/ipfs_kit",
            commit=CURRENT_HEAD_KIT_COMMIT,
            tree=CURRENT_HEAD_KIT_TREE,
            origin_main=CURRENT_HEAD_KIT_COMMIT,
            origin_main_is_ancestor=True,
            gitlink=CURRENT_HEAD_KIT_COMMIT,
            clean=True,
            exact_toplevel=True,
            evidence_kind="measured",
            reason="Nested HEAD equals the portfolio gitlink.",
            recursive_gitlink_count=CURRENT_HEAD_KIT_RECURSIVE_GITLINK_COUNT,
            recursive_gitlink_digest=CURRENT_HEAD_KIT_RECURSIVE_GITLINK_DIGEST,
        ),
    )


def current_head_bootstrap_artifacts() -> tuple[BootstrapArtifact, ...]:
    """Measured hashes of the immutable bootstrap inputs at current HEAD."""

    return tuple(
        BootstrapArtifact(
            path=path,
            sha256=CURRENT_HEAD_BOOTSTRAP_SHA256[path],
            bytes=CURRENT_HEAD_BOOTSTRAP_BYTES[path],
            evidence_kind="measured",
            reason="Committed bootstrap input; worker must not modify it.",
        )
        for path in IMMUTABLE_BOOTSTRAP_INPUTS
    )


def current_head_policies() -> tuple[PolicyBaseline, ...]:
    """Measured CIDs of the scheduler policy objects at current HEAD."""

    return tuple(
        PolicyBaseline(
            key=key,
            content_cid=CURRENT_HEAD_POLICY_CIDS[key],
            evidence_kind="measured",
            reason="Observed from the immutable supervisor scheduler config.",
        )
        for key in POLICY_BASELINE_KEYS
    )


def qualify_current_head_source_seal() -> SourceSealVerdict:
    """Ordinary current-head PCPR-000 evaluation: clean gitlink forest, R&D."""

    return seal_source_forest_and_supervisor_baseline(
        portfolio=current_head_portfolio_observation(),
        sources=current_head_source_observations(),
        bootstrap_artifacts=current_head_bootstrap_artifacts(),
        policies=current_head_policies(),
    )


def pcpr_000_receipt_promotion(verdict: SourceSealVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields.  Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise SourceSealError("source seal must not mint a closed release outcome")
    if verdict.release_claim:
        raise SourceSealError("source seal must not claim a PCPR release")
    if verdict.completion_authoritative:
        raise SourceSealError("source-seal completion is not authoritative")
    if verdict.contracts_frozen:
        raise SourceSealError("PCPR-000 records a baseline; it does not freeze contracts")
    if verdict.duckdb_or_quack_state_written:
        raise SourceSealError("source seal must not write DuckDB or Quack state")
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise SourceSealError("promotion_status must not be a closed release outcome")
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise SourceSealError("promotion_status is not an admitted PCPR-000 status")
    if verdict.promotion_status == "supervisor_promoted":
        raise SourceSealError("PCPR-000 cannot promote the supervisor")
    return {
        "schema": SOURCE_SEAL_VERDICT_SCHEMA,
        "interface": SOURCE_SEAL_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "changed_source_seal_requires_fresh_inventory": (
            verdict.changed_source_seal_requires_fresh_inventory
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_000_receipt_promotion() -> dict[str, Any]:
    """Fail-closed promotion section for the ordinary clean-forest case."""

    return pcpr_000_receipt_promotion(qualify_current_head_source_seal())


def pcpr_000_receipt_source_forest(verdict: SourceSealVerdict) -> dict[str, Any]:
    """Outer-receipt source-forest section."""

    return {
        "schema": "proof-carrying-platform-source-forest@1",
        "source_authority_count": verdict.source_authority_count,
        "recursive_sibling_submodules_are_source_authority": False,
        "portfolio": verdict.portfolio.to_mapping(),
        "repositories": [item.to_mapping() for item in verdict.sources],
        "planning_pins": {
            name: dict(pin) for name, pin in PLANNING_SOURCE_FOREST.items()
        },
        "changed_source_seal_requires_fresh_inventory": (
            verdict.changed_source_seal_requires_fresh_inventory
        ),
        "evidence_kind": "measured",
    }


def pcpr_000_receipt_contracts(verdict: SourceSealVerdict) -> dict[str, Any]:
    """Outer-receipt contract-baseline section.  Not a freeze."""

    return {
        "frozen": False,
        "freeze_task": "PCPR-002",
        "normative_stabilization_task": "PCPR-040",
        "catalog": [dict(item) for item in verdict.contract_catalog],
        "evidence_kind": "measured",
    }


def pcpr_000_receipt_policies(verdict: SourceSealVerdict) -> dict[str, Any]:
    """Outer-receipt policy-baseline section."""

    return {
        "bundle_cid": CURRENT_HEAD_POLICY_BUNDLE_CID
        if all(item.evidence_kind == "measured" for item in verdict.policies)
        and all(
            item.content_cid == CURRENT_HEAD_POLICY_CIDS[item.key]
            for item in verdict.policies
        )
        else None,
        "policies": [item.to_mapping() for item in verdict.policies],
        "markdown_is_bootstrap_only": True,
        "duckdb_authoritative_after_materialization": True,
        "quack_exclusive_owner_transport": True,
        "ducklake_non_authoritative": True,
        "evidence_kind": "measured",
    }


def pcpr_000_receipt_supervisor_baseline(verdict: SourceSealVerdict) -> dict[str, Any]:
    """Outer-receipt supervisor-baseline section.  Not qualification."""

    return {
        "surfaces": list(SUPERVISOR_BASELINE_SURFACES),
        "frozen": False,
        "supervisor_promoted": False,
        "qualification_task": "PCPR-001",
        "freeze_task": "PCPR-002",
        "direct_objective_handoff_task": "PCPR-004",
        "board_projection_records_pcpr_004_complete": True,
        "board_projection_is_not_duckdb_authority": True,
        "duckdb_task_state": "unavailable",
        "duckdb_task_state_reason": (
            "This worker did not query or write DuckDB. Bootstrap "
            "initial_projection is not live task authority."
        ),
        "promotion_status": verdict.promotion_status,
        "evidence_kind": "measured",
    }


def pcpr_000_receipt_negative_results() -> dict[str, Any]:
    """Fixed negative results for the PCPR-000 R&D baseline."""

    return {
        "simulated_clean_cannot_seal": True,
        "recursive_sibling_submodules_are_not_source_authority": True,
        "gitlink_mismatch_without_isolated_binding_is_blocked": True,
        "missing_origin_main_is_unavailable_not_zero": True,
        "contracts_not_frozen": True,
        "supervisor_not_promoted": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_000_receipt_sections() -> dict[str, Any]:
    """Promotion, forest, contract, policy, and negative sections."""

    verdict = qualify_current_head_source_seal()
    promotion = pcpr_000_receipt_promotion(verdict)
    return {
        "qualification_verdict": promotion,
        "source_forest": pcpr_000_receipt_source_forest(verdict),
        "contract_baseline": pcpr_000_receipt_contracts(verdict),
        "policy_baseline": pcpr_000_receipt_policies(verdict),
        "supervisor_baseline": pcpr_000_receipt_supervisor_baseline(verdict),
        "negative_results": pcpr_000_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
    }


def pcpr_000_current_tree_binding(
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
    """Measured current-tree identities for a PCPR-000 outer receipt."""

    outer = _optional_commit(outer_commit, "outer_commit")
    tree = _optional_commit(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _optional_commit(origin_main, "origin_main")
    if origin_main_is_ancestor is not True:
        raise SourceSealError("origin_main_is_ancestor must be true for a measured binding")
    accel = _optional_commit(accelerator_pre_change_commit, "accelerator_pre_change_commit")
    accel_tree = _optional_commit(
        accelerator_pre_change_tree, "accelerator_pre_change_tree"
    )
    accel_link = _optional_commit(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _optional_commit(
        accelerator_origin_main, "accelerator_origin_main"
    )
    if accelerator_origin_main_is_ancestor is not True:
        raise SourceSealError(
            "accelerator_origin_main_is_ancestor must be true for a measured binding"
        )
    if accel != accel_link:
        raise SourceSealError("accelerator_pre_change_commit must equal accelerator_gitlink")
    datasets = _optional_commit(datasets_commit, "datasets_commit")
    datasets_tree_id = _optional_commit(datasets_tree, "datasets_tree")
    datasets_link = _optional_commit(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise SourceSealError("datasets_commit must equal datasets_gitlink")
    kit = _optional_commit(kit_commit, "kit_commit")
    kit_tree_id = _optional_commit(kit_tree, "kit_tree")
    kit_link = _optional_commit(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise SourceSealError("kit_commit must equal kit_gitlink")
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
        "accelerator_post_change_tree": "dirty-worktree; exact CID after accepted nested commit",
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_000_current_tree_binding() -> dict[str, Any]:
    """Measured current-tree binding for this isolated worktree."""

    return pcpr_000_current_tree_binding(
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


def validate_pcpr_000_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-000 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise SourceSealError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_000_TASK_ID:
        raise SourceSealError("outer receipt task_id must be PCPR-000")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise SourceSealError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise SourceSealError("outer receipt completion is not authoritative")

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise SourceSealError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise SourceSealError("qualification_verdict.closed_release_outcome must be null")
    if verdict_section.get("release_claim") is True:
        raise SourceSealError("qualification_verdict must not claim a release")
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise SourceSealError("source seal must not write DuckDB or Quack state")
    if verdict_section.get("contracts_frozen") is True:
        raise SourceSealError("PCPR-000 must not freeze contracts")
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise SourceSealError(
            "qualification_verdict.promotion_status is not an admitted PCPR-000 status"
        )
    if promotion_status == "supervisor_promoted":
        raise SourceSealError("PCPR-000 cannot promote the supervisor")

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise SourceSealError("acceptance.closed_release_outcome must be null")
        if acceptance.get("release_claim") is True:
            raise SourceSealError("acceptance must not claim a release")
        if acceptance.get("promotion_status") not in {None, promotion_status}:
            raise SourceSealError(
                "acceptance.promotion_status must match qualification_verdict"
            )

    binding = payload.get("current_tree_binding")
    if isinstance(binding, Mapping):
        if binding.get("origin_main_is_ancestor") is not True:
            raise SourceSealError("current_tree_binding.origin_main_is_ancestor must be true")
        _reject_closed_release_value(binding.get("outer_subject"), "outer_subject")

    forest = payload.get("source_forest")
    if isinstance(forest, Mapping):
        if forest.get("recursive_sibling_submodules_are_source_authority") is True:
            raise SourceSealError(
                "recursive sibling submodules are not source authority"
            )
        if forest.get("source_authority_count") not in (None, SOURCE_AUTHORITY_COUNT):
            raise SourceSealError("source_authority_count must be 3")

    expected = current_head_pcpr_000_receipt_promotion()
    if promotion_status == "rnd_non_promoted":
        if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
            raise SourceSealError(
                "clean-forest qualification_verdict.verdict_cid must match the evaluator"
            )
        if expected["promotion_status"] == "rnd_non_promoted" and promotion_status not in {
            "rnd_non_promoted",
            "typed_unavailable",
            "typed_blocked",
            "supervisor_non_promoted",
        }:
            raise SourceSealError("promotion_status must be an honest non-promotion")
    return {
        "valid": True,
        "task_id": PCPR_000_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: Path,
    git_binary: str,
    timeout_seconds: int = 30,
) -> tuple[int, str]:
    result = subprocess.run(
        [git_binary, *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    stdout = str(result.stdout or "").strip()
    return int(result.returncode), stdout


def gitlink_digest(entries: Sequence[Mapping[str, str]]) -> str:
    """Return the sha256 of the canonical gitlink inventory."""

    payload = [
        {"commit": str(item["commit"]), "path": str(item["path"])}
        for item in sorted(entries, key=lambda row: str(row["path"]))
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observe_git_repository(
    root: Path,
    *,
    repository: str,
    relative: str,
    gitlink: str | None = None,
    git_binary: str = SEALED_GIT_BINARY,
    isolated_source_binding: str = "",
) -> RepositoryObservation:
    """Observe one Git worktree.  Failures become typed unavailable."""

    path = Path(root)
    if not path.is_dir():
        return RepositoryObservation(
            repository=repository,
            path=relative,
            commit=None,
            tree=None,
            origin_main=None,
            origin_main_is_ancestor=None,
            gitlink=gitlink,
            clean=None,
            exact_toplevel=False,
            evidence_kind="unavailable",
            reason="Repository directory is not present.",
            isolated_source_binding=isolated_source_binding,
        )
    try:
        top_code, toplevel = _run_git(
            ["rev-parse", "--show-toplevel"], cwd=path, git_binary=git_binary
        )
        head_code, head = _run_git(["rev-parse", "HEAD"], cwd=path, git_binary=git_binary)
        tree_code, tree = _run_git(
            ["rev-parse", "HEAD^{tree}"], cwd=path, git_binary=git_binary
        )
        status_code, status = _run_git(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            cwd=path,
            git_binary=git_binary,
        )
        branch_code, branch = _run_git(
            ["branch", "--show-current"], cwd=path, git_binary=git_binary
        )
        origin_code, origin_main = _run_git(
            ["rev-parse", "origin/main"], cwd=path, git_binary=git_binary
        )
        ls_code, ls_tree = _run_git(
            ["ls-tree", "-r", "HEAD"], cwd=path, git_binary=git_binary
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RepositoryObservation(
            repository=repository,
            path=relative,
            commit=None,
            tree=None,
            origin_main=None,
            origin_main_is_ancestor=None,
            gitlink=gitlink,
            clean=None,
            exact_toplevel=None,
            evidence_kind="unavailable",
            reason=f"Git observation failed: {type(exc).__name__}",
            isolated_source_binding=isolated_source_binding,
        )
    if top_code != 0 or head_code != 0 or tree_code != 0 or status_code != 0:
        return RepositoryObservation(
            repository=repository,
            path=relative,
            commit=None,
            tree=None,
            origin_main=None,
            origin_main_is_ancestor=None,
            gitlink=gitlink,
            clean=None,
            exact_toplevel=None,
            evidence_kind="unavailable",
            reason="Git identity commands failed.",
            isolated_source_binding=isolated_source_binding,
        )
    exact_top = Path(toplevel).resolve() == path.resolve()
    origin = origin_main if origin_code == 0 and COMMIT_RE.fullmatch(origin_main) else None
    ancestor: bool | None
    if origin is None:
        ancestor = None
    else:
        ancestor_code, _unused = _run_git(
            ["merge-base", "--is-ancestor", origin, head],
            cwd=path,
            git_binary=git_binary,
        )
        ancestor = ancestor_code == 0
    gitlink_entries: list[dict[str, str]] = []
    if ls_code == 0:
        for line in ls_tree.splitlines():
            if "\t" not in line:
                continue
            meta, entry_path = line.split("\t", 1)
            parts = meta.split()
            if len(parts) >= 3 and parts[0] == "160000":
                gitlink_entries.append({"path": entry_path, "commit": parts[2]})
    digest = gitlink_digest(gitlink_entries) if gitlink_entries or ls_code == 0 else None
    count = len(gitlink_entries) if digest is not None else None
    return RepositoryObservation(
        repository=repository,
        path=relative,
        commit=head if COMMIT_RE.fullmatch(head) else None,
        tree=tree if COMMIT_RE.fullmatch(tree) else None,
        origin_main=origin,
        origin_main_is_ancestor=ancestor,
        gitlink=gitlink,
        clean=not bool(status),
        exact_toplevel=exact_top,
        evidence_kind="measured",
        reason="Observed from the named Git worktree.",
        branch=branch if branch_code == 0 else "",
        recursive_gitlink_count=count,
        recursive_gitlink_digest=digest,
        isolated_source_binding=isolated_source_binding,
    )


def observe_source_forest(
    portfolio_root: Path,
    *,
    git_binary: str = SEALED_GIT_BINARY,
) -> tuple[RepositoryObservation, tuple[RepositoryObservation, ...]]:
    """Observe the portfolio and the three source-authority repositories."""

    root = Path(portfolio_root)
    portfolio_ls_code, portfolio_ls = _run_git(
        ["ls-tree", "HEAD", "--", *[path for _name, path in SOURCE_REPOSITORIES]],
        cwd=root,
        git_binary=git_binary,
    )
    gitlinks: dict[str, str] = {}
    if portfolio_ls_code == 0:
        for line in portfolio_ls.splitlines():
            if "\t" not in line:
                continue
            meta, entry_path = line.split("\t", 1)
            parts = meta.split()
            if len(parts) >= 3 and parts[0] == "160000":
                gitlinks[entry_path] = parts[2]
    portfolio = observe_git_repository(
        root,
        repository="portfolio",
        relative=".",
        git_binary=git_binary,
        isolated_source_binding="isolated_implementation_worktree",
    )
    sources = tuple(
        observe_git_repository(
            root / relative,
            repository=name,
            relative=relative,
            gitlink=gitlinks.get(relative),
            git_binary=git_binary,
        )
        for name, relative in SOURCE_REPOSITORIES
    )
    return portfolio, sources
