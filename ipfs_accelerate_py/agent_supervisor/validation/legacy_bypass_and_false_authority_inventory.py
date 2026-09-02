"""Fail-closed PCPR legacy-bypass and false-authority inventory.

PCPR-003 inventories every plan-named bypass, mock, simulation,
pseudo-identity, fallback, and false-authority path class across the
three source-authority repositories.  Each class records source
identity, authority analysis, disposition, migration impact, and
current-tree probe evidence.  This module is not release authority: it
does not write DuckDB or Quack state, does not freeze contracts, does
not prohibit competing authorities as a live freeze, and never emits a
closed PCPR release outcome.

Static source probes are ``measured``.  They are not ``measured_live``.
Missing repositories or files stay typed unavailable.  Simulated or
estimated observations cannot mint presence, absence, or live
prohibition.  Completeness is relative to the closed PCPR catalog, not
a proof that unlisted runtime bypasses are absent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import (
    CLOSED_RELEASE_OUTCOMES,
    COMPETING_AUTHORITY_PROHIBITIONS,
    COMPETING_AUTHORITY_SCOPES,
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    LIVE_SATISFYING_KIND,
    CompetingAuthorityObservation,
)
from .direct_objective_event_driven_qualification import (
    qualify_current_head_without_live_campaign,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
    SOURCE_REPOSITORIES,
)


INVENTORY_INTERFACE: Final = "LegacyBypassAndFalseAuthorityInventory@1"
INVENTORY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-bypass-and-false-authority-inventory@1"
)
INVENTORY_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "legacy-bypass-and-false-authority-inventory-verdict@1"
)

PCPR_003_TASK_ID: Final = "PCPR-003"
PCPR_003_GOAL_ID: Final = "PCPR-G130"
PCPR_000_TASK_ID: Final = "PCPR-000"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_002_TASK_ID: Final = "PCPR-002"
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
PATH_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "auto_install",
        "bypass",
        "competing_authority",
        "fallback",
        "false_authority",
        "mock",
        "pseudo_identity",
        "sibling_coupling",
        "simulation",
    }
)
PATH_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "keep_internal_not_public_requirement",
        "qualify_or_typed_unavailable",
        "quarantine",
        "remove",
        "replace_with_capability_ladder",
        "replace_with_canonical_cid",
        "replace_with_quack_fence",
        "replace_with_typed_outcome",
    }
)
SAMPLE_HIT_LIMIT: Final = 3
MAX_PROBE_FILE_BYTES: Final = 1_048_576

# Measured current-head identities for this isolated implementation worktree.
CURRENT_HEAD_OUTER_COMMIT: Final = "9b710b36e1db60c9d4f827d9c7d34d7775d64666"
CURRENT_HEAD_OUTER_TREE: Final = "4d88d5cd57f5675eb805c1c1061989bffeb60a1b"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "fix(pcpr): reseal accelerate planning pins to leftover-wait recovery"
)
CURRENT_HEAD_OUTER_BRANCH: Final = (
    "implementation/pcpr-003-5c42a24f2f56-attempt-1-1788210693"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_FIRST_PARENT: Final = "97df816c1755e35593f350832c383c048e12febc"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "d2bb921548f63f92a5e198ab3c90515b7973c806"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "9ab1f12ddb60e01ce5ffb49334b937320e45956a"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_source_seal_and_supervisor_baseline.py",
    "test/api/test_agent_supervisor_direct_objective_event_driven_qualification.py",
    "test/api/test_agent_supervisor_canonical_supervisor_contract_freeze.py",
    "test/api/test_agent_supervisor_legacy_bypass_and_false_authority_inventory.py",
)

REPO_ROOT_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ipfs_accelerate_py": "external/ipfs_accelerate",
        "ipfs_datasets_py": "external/ipfs_datasets",
        "ipfs_kit_py": "external/ipfs_kit",
        "portfolio": ".",
    }
)


class LegacyBypassInventoryError(ValueError):
    """Malformed inventory evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class PathClassRecipe:
    """Closed catalog entry for one plan-named false-authority class."""

    path_id: str
    category: str
    repository: str
    remediating_task: str
    disposition: str
    authority_claim: str
    migration_impact: str
    relpaths: tuple[str, ...]
    needles: tuple[str, ...]


def _recipe(
    path_id: str,
    *,
    category: str,
    repository: str,
    remediating_task: str,
    disposition: str,
    authority_claim: str,
    migration_impact: str,
    relpaths: tuple[str, ...],
    needles: tuple[str, ...],
) -> PathClassRecipe:
    return PathClassRecipe(
        path_id=path_id,
        category=category,
        repository=repository,
        remediating_task=remediating_task,
        disposition=disposition,
        authority_claim=authority_claim,
        migration_impact=migration_impact,
        relpaths=relpaths,
        needles=needles,
    )


# Closed PCPR-G130 catalog.  Completeness is relative to this list.
PATH_CLASS_CATALOG: Final[tuple[PathClassRecipe, ...]] = (
    _recipe(
        "datasets_import_time_auto_install",
        category="auto_install",
        repository="ipfs_datasets_py",
        remediating_task="PCPR-010",
        disposition="remove",
        authority_claim=(
            "Package import may construct an installer and attempt optional "
            "dependency installation, violating inert import."
        ),
        migration_impact=(
            "Callers that rely on import-time pip/network side effects must "
            "switch to an explicit CLI or operator install action."
        ),
        relpaths=(
            "ipfs_datasets_py/__init__.py",
            "ipfs_datasets_py/auto_installer.py",
            "ipfs_datasets_py/ipfs_backend_router.py",
        ),
        needles=(
            "ensure_repo_installer_current",
            "auto_install",
            "ensure_module",
            "DependencyInstaller",
        ),
    ),
    _recipe(
        "datasets_false_success_fallbacks",
        category="fallback",
        repository="ipfs_datasets_py",
        remediating_task="PCPR-011",
        disposition="replace_with_typed_outcome",
        authority_claim=(
            "Fallback and stub surfaces return status success without a real "
            "download, upload, or transport effect."
        ),
        migration_impact=(
            "Callers treating status success as observed effect must handle "
            "typed unavailable, simulated, or failed outcomes."
        ),
        relpaths=(
            "ipfs_datasets_py/__init__.py",
            "ipfs_datasets_py/p2p_networking/libp2p_kit.py",
            "ipfs_datasets_py/p2p_networking/libp2p_kit_stub.py",
        ),
        needles=(
            '_FallbackIPFSDatasets',
            '"status": "success"',
            "Stub implementation",
        ),
    ),
    _recipe(
        "datasets_untyped_none_fallbacks",
        category="fallback",
        repository="ipfs_datasets_py",
        remediating_task="PCPR-012",
        disposition="replace_with_typed_outcome",
        authority_claim=(
            "Public load/dataset symbols may be None or empty-success stubs "
            "when optional dependencies are absent."
        ),
        migration_impact=(
            "Public surfaces need stable, beta, experimental, simulation-only, "
            "or unavailable typing instead of silent None."
        ),
        relpaths=("ipfs_datasets_py/__init__.py",),
        needles=("load_dataset = None", "HAVE_IPFS_DATASETS = False"),
    ),
    _recipe(
        "datasets_logic_provider_v1_freeform",
        category="false_authority",
        repository="ipfs_datasets_py",
        remediating_task="PCPR-013",
        disposition="quarantine",
        authority_claim=(
            "LogicProviderProtocol version 1 remains importable; free-form "
            "payloads must not mint executable BackendRequest values."
        ),
        migration_impact=(
            "A version-1 adapter must be explicit and fail-closed; v2 or its "
            "successor becomes canonical."
        ),
        relpaths=(
            "ipfs_datasets_py/logic/backends/provider.py",
            "ipfs_datasets_py/logic/backends/protocol_v2.py",
        ),
        needles=(
            "LOGIC_PROVIDER_PROTOCOL_VERSION",
            "class LogicProvider",
            "LogicProviderProtocolV2",
        ),
    ),
    _recipe(
        "datasets_solver_live_claims",
        category="false_authority",
        repository="ipfs_datasets_py",
        remediating_task="PCPR-017",
        disposition="qualify_or_typed_unavailable",
        authority_claim=(
            "Docs and extras advertise Z3, cvc5, Lean, and Coq without this "
            "campaign's live solver qualification."
        ),
        migration_impact=(
            "Missing solvers must remain typed unavailable; README claims "
            "follow the qualification matrix."
        ),
        relpaths=("README.md", "pyproject.toml"),
        needles=("ipfs-datasets-install-provers", "z3-solver", "cvc5", "lean", "coq"),
    ),
    _recipe(
        "kit_unqualified_live_storage_claims",
        category="false_authority",
        repository="ipfs_kit_py",
        remediating_task="PCPR-020",
        disposition="qualify_or_typed_unavailable",
        authority_claim=(
            "Public README claims a production-ready toolkit without the "
            "PCPR live local/IPFS qualification matrix."
        ),
        migration_impact=(
            "Support-matrix generation (PCPR-027) must replace free-text "
            "production-ready language."
        ),
        relpaths=("README.md",),
        needles=("production-ready",),
    ),
    _recipe(
        "kit_iroh_live_claims",
        category="false_authority",
        repository="ipfs_kit_py",
        remediating_task="PCPR-022",
        disposition="qualify_or_typed_unavailable",
        authority_claim=(
            "Iroh CLI and docs are published as installable live support "
            "without a PCPR live Iroh qualification receipt."
        ),
        migration_impact=(
            "Iroh must be qualified, experimental, unavailable, or unsupported; "
            "live-support claims are removed otherwise."
        ),
        relpaths=("docs/QUICK_REFERENCE.md", "docs/DOCUMENTATION_INDEX.md"),
        needles=("ipfs-kit-iroh", "Iroh integration", "iroh"),
    ),
    _recipe(
        "kit_sibling_test_tree_coupling",
        category="sibling_coupling",
        repository="ipfs_kit_py",
        remediating_task="PCPR-025",
        disposition="remove",
        authority_claim=(
            "sitecustomize and benchmarks inject the source tree onto "
            "sys.path, coupling installed-package tests to a sibling checkout."
        ),
        migration_impact=(
            "Installed packages must use packaged vectors; sibling tests/ "
            "directories cannot be required."
        ),
        relpaths=("sitecustomize.py", "benchmarks/runtime_readiness/run.py"),
        needles=("sys.path.insert", "tests/runtime_readiness"),
    ),
    _recipe(
        "accelerate_legacy_mock_coordinator",
        category="mock",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-030",
        disposition="quarantine",
        authority_claim=(
            "Ordinary runtime constructors still substitute MockWorker and a "
            "legacy workflow coordinator that can run without a live backend."
        ),
        migration_impact=(
            "Move mock coordinators behind an explicit compatibility or "
            "simulation namespace; ordinary runtime cannot instantiate them."
        ),
        relpaths=(
            "ipfs_accelerate_py/compatibility/simulation/legacy_mock_coordinator.py",
            "ipfs_accelerate_py/ipfs_accelerate.py",
            "ipfs_accelerate_py/datasets_integration/workflow.py",
            "ipfs_accelerate_py/llm_router.py",
        ),
        needles=("MockWorker", "WorkflowCoordinator", "_MockProvider"),
    ),
    _recipe(
        "accelerate_fabricated_hardware",
        category="simulation",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-031",
        disposition="replace_with_capability_ladder",
        authority_claim=(
            "Mock hardware detection and CUDA mock implementations can report "
            "availability without declared/installed/detected/canary/qualified "
            "ladder evidence."
        ),
        migration_impact=(
            "Device visibility, adapter presence, or test defaults never imply "
            "production_authorized."
        ),
        relpaths=(
            "ipfs_accelerate_py/compatibility/simulation/fabricated_hardware.py",
            "ipfs_accelerate_py/assurance/hardware_capability_ladder.py",
            "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
            "ipfs_accelerate_py/worker/cuda_utils.py",
            "ipfs_accelerate_py/hf_model_server/hardware/detector.py",
            "ipfs_accelerate_py/kit/hardware_kit.py",
        ),
        needles=(
            "instantiate_mock_hardware_detection",
            "create_simulated_cuda_implementation",
            "production_authorized",
            "device_visibility_is_not_qualification",
        ),
    ),
    _recipe(
        "accelerate_pseudo_cid",
        category="pseudo_identity",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-032",
        disposition="replace_with_canonical_cid",
        authority_claim=(
            "Hexadecimal SHA-256 slices and Qm-prefixed strings are returned "
            "as IPFS CIDs."
        ),
        migration_impact=(
            "Callers and caches must use canonical bytes and real CID "
            "generation; hex strings are not CIDs."
        ),
        relpaths=(
            "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
            "ipfs_accelerate_py/mcp/tools/mock_ipfs.py",
            "ipfs_accelerate_py/ipfs_backend_router.py",
        ),
        needles=("mock_cid = f\"Qm", "random_cid", "_generate_cid"),
    ),
    _recipe(
        "accelerate_fabricated_endpoint_success",
        category="mock",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-033",
        disposition="quarantine",
        authority_claim=(
            "Endpoint registration and MCP inference can fall back to mock "
            "handlers or hardware tests that report passed without a live probe."
        ),
        migration_impact=(
            "A registered endpoint is live only after backend, model, canary, "
            "timeout, cancellation, and resource admission evidence."
        ),
        relpaths=(
            "ipfs_accelerate_py/ipfs_accelerate.py",
            "ipfs_accelerate_py/mcp/inference_tools.py",
            "ipfs_accelerate_py/mcp_server/tools/hardware_tools/native_hardware_tools.py",
        ),
        needles=(
            "_create_mock_handler",
            "_mock_inference",
            "_test_hardware_fallback",
        ),
    ),
    _recipe(
        "accelerate_capability_ladder_gap",
        category="false_authority",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-034",
        disposition="replace_with_capability_ladder",
        authority_claim=(
            "HardwareDetector.available is a boolean detection flag, not the "
            "declared/installed/detected/canary_passed/qualified/"
            "production_authorized ladder."
        ),
        migration_impact=(
            "Production actions must require production_authorized; detection "
            "alone is not qualification."
        ),
        relpaths=("ipfs_accelerate_py/hf_model_server/hardware/detector.py",),
        needles=("class HardwareDetector", "available: bool", "_detect_cuda"),
    ),
    _recipe(
        "accelerate_mutable_git_branch_deps",
        category="bypass",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-035",
        disposition="remove",
        authority_claim=(
            "pyproject extras and requirements pin sibling packages to mutable "
            "Git @main rather than immutable commits or signed artifacts."
        ),
        migration_impact=(
            "Qualified releases cannot depend on a mutable Git branch."
        ),
        relpaths=("pyproject.toml", "requirements-hf-server.txt"),
        needles=("git+https://", "@main"),
    ),
    _recipe(
        "accelerate_python_floor_metadata_drift",
        category="false_authority",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-036",
        disposition="remove",
        authority_claim=(
            "packaging metadata advertises Python >=3.8 while this campaign's "
            "canonical interpreter is Python 3.12."
        ),
        migration_impact=(
            "Code, metadata, and CI must agree on the real Python floor."
        ),
        relpaths=("pyproject.toml",),
        needles=('requires-python = ">=3.8"', "Programming Language :: Python :: 3.8"),
    ),
    _recipe(
        "accelerate_legacy_compatibility_module",
        category="mock",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-030",
        disposition="quarantine",
        authority_claim=(
            "ipfs_accelerate_py_legacy.py remains a compatibility surface with "
            "mock IPFS store, mock providers, and mock hardware."
        ),
        migration_impact=(
            "Compatibility namespace must be explicit; it cannot be the "
            "ordinary runtime."
        ),
        relpaths=("ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",),
        needles=(
            "Mock implementation",
            "_create_mock_handler",
            "store_to_ipfs",
        ),
    ),
    _recipe(
        "accelerate_direct_duckdb_task_state",
        category="competing_authority",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-003",
        disposition="replace_with_quack_fence",
        authority_claim=(
            "Supervisor stores still open DuckDB connections for task, lease, "
            "and coordination state; workers are forbidden from writing those "
            "tables directly, and Quack is the exclusive authenticated owner "
            "transport once live."
        ),
        migration_impact=(
            "All task-state mutations must pass through Quack-fenced owner "
            "APIs; this inventory does not itself write DuckDB or Quack."
        ),
        relpaths=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
            "ipfs_accelerate_py/workflow_manager.py",
        ),
        needles=("connect_duckdb_with_policy", "import duckdb", "duckdb.connect"),
    ),
    _recipe(
        "simulation_represented_as_live",
        category="simulation",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-030",
        disposition="quarantine",
        authority_claim=(
            "Simulation, fixture, and mock origins still exist on production-"
            "reachable paths and must not satisfy live capability predicates."
        ),
        migration_impact=(
            "Explicit simulation receipts must say simulation; PCAR-015 taint "
            "rules remain in force and are not a PCPR live qualification."
        ),
        relpaths=(
            "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/legacy_paths.py",
            "ipfs_accelerate_py/ipfs_accelerate.py",
        ),
        needles=(
            "INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE",
            "MockWorker",
            "OriginTaint.SIMULATION",
        ),
    ),
    _recipe(
        "client_supplied_policy_allow",
        category="bypass",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-083",
        disposition="remove",
        authority_claim=(
            "Client-supplied policy allow or confirmation must never be "
            "authoritative; this class inventories remaining allow-shaped "
            "client inputs."
        ),
        migration_impact=(
            "External clients cannot bypass confirmation, leases, fencing, or "
            "policy pointers."
        ),
        relpaths=(
            "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py",
            "ipfs_accelerate_py/agent_supervisor/autonomy/policy_evaluation.py",
        ),
        needles=(
            "client_policy_allow",
            "policy_allow_is_authoritative",
            "client confirmation is authoritative",
        ),
    ),
    _recipe(
        "prebuilt_complete_launch_plan_injection",
        category="bypass",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-004",
        disposition="keep_internal_not_public_requirement",
        authority_claim=(
            "CompleteLaunchPlan remains an internal runtime type.  Public "
            "Supervisor.run no longer requires caller injection after "
            "PCPR-004; leftover injection would restore the bootstrap bypass."
        ),
        migration_impact=(
            "Do not reintroduce caller CompleteLaunchPlan injection on the "
            "public prompt facade."
        ),
        relpaths=(
            "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py",
        ),
        needles=(
            "CompleteLaunchPlan injection",
            "complete_plan must be a CompleteLaunchPlan",
        ),
    ),
    _recipe(
        "markdown_as_live_authority",
        category="competing_authority",
        repository="portfolio",
        remediating_task="PCPR-004",
        disposition="replace_with_typed_outcome",
        authority_claim=(
            "The immutable Markdown board remains the operator-authorized "
            "campaign projection until DuckDB materialization; Markdown is "
            "bootstrap-only and cannot be live task authority."
        ),
        migration_impact=(
            "After successful materialization DuckDB is authoritative and "
            "Quack is the exclusive authenticated state-owner transport."
        ),
        relpaths=(
            "docs/architecture/proof_carrying_platform_qualification_and_release_v1.todo.md",
        ),
        needles=(
            "DuckDB becomes authoritative after materialization",
            "tracked Markdown board",
        ),
    ),
    _recipe(
        "model_completion_as_terminal_authority",
        category="competing_authority",
        repository="ipfs_accelerate_py",
        remediating_task="PCPR-001",
        disposition="remove",
        authority_claim=(
            "Model and worker output is candidate evidence only; independent "
            "current-tree validation owns terminalization."
        ),
        migration_impact=(
            "No task may complete solely from an LLM assertion; receipts stay "
            "completion_authoritative false."
        ),
        relpaths=(
            "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py",
            "ipfs_accelerate_py/agent_supervisor/autonomy/policy_evaluation.py",
        ),
        needles=(
            "llm_assertion_completes_task",
            "model_output_is_terminal_authority",
            "worker_completion_authoritative = True",
        ),
    ),
)

REQUIRED_PATH_IDS: Final[tuple[str, ...]] = tuple(
    item.path_id for item in PATH_CLASS_CATALOG
)
CATALOG_BY_ID: Final[Mapping[str, PathClassRecipe]] = MappingProxyType(
    {item.path_id: item for item in PATH_CLASS_CATALOG}
)

# Path classes whose measured hits count as repository-scope presence for a
# PCPR-002 competing-authority name.  Names omitted here stay typed
# unavailable at repository scope: static absence is not a live prohibition.
COMPETING_AUTHORITY_PATH_IDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "direct_duckdb_writes": ("accelerate_direct_duckdb_task_state",),
        "markdown_as_live_authority": ("markdown_as_live_authority",),
        "model_completion_authority": ("model_completion_as_terminal_authority",),
        "new_task_database": ("accelerate_direct_duckdb_task_state",),
    }
)


@dataclass(frozen=True)
class PathObservation:
    """Current-tree observation of one catalog class."""

    path_id: str
    category: str
    repository: str
    remediating_task: str
    disposition: str
    authority_claim: str
    migration_impact: str
    present: bool | None
    evidence_kind: str
    hit_count: int
    sample_hits: tuple[Mapping[str, Any], ...]
    source_identity: Mapping[str, Any]
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "category": self.category,
            "repository": self.repository,
            "remediating_task": self.remediating_task,
            "disposition": self.disposition,
            "authority_analysis": self.authority_claim,
            "migration_impact": self.migration_impact,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "hit_count": self.hit_count,
            "sample_hits": [dict(item) for item in self.sample_hits],
            "source_identity": dict(self.source_identity),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InventoryVerdict:
    """Fail-closed PCPR-003 inventory decision."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    competing_authorities_prohibited: bool
    live_runtime_inventory: bool
    duckdb_or_quack_state_written: bool
    catalog_complete: bool
    measured_present_count: int
    measured_absent_count: int
    unavailable_count: int
    this_task_created_competing_authority: bool
    paths: tuple[PathObservation, ...]
    competing_authorities: tuple[CompetingAuthorityObservation, ...]
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
            "competing_authorities_prohibited": self.competing_authorities_prohibited,
            "live_runtime_inventory": self.live_runtime_inventory,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "catalog_complete": self.catalog_complete,
            "measured_present_count": self.measured_present_count,
            "measured_absent_count": self.measured_absent_count,
            "unavailable_count": self.unavailable_count,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "paths": [item.to_mapping() for item in self.paths],
            "competing_authorities": [
                item.to_mapping() for item in self.competing_authorities
            ],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LegacyBypassInventoryError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise LegacyBypassInventoryError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise LegacyBypassInventoryError(
            f"{name} must be a lowercase 40-character git object id"
        )
    return text


def _require_ancestor(flag: Any, name: str) -> bool:
    if flag is not True:
        raise LegacyBypassInventoryError(
            f"{name} must be true; a non-ancestor origin/main cannot bind current-head evidence"
        )
    return True


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise LegacyBypassInventoryError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_unique(ids: Sequence[str], population: Sequence[str], name: str) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise LegacyBypassInventoryError(f"duplicate {name}: {item}")
        seen.add(item)
    extra = [item for item in ids if item not in population]
    if extra:
        raise LegacyBypassInventoryError(f"unknown {name}: {extra[0]}")
    missing = [item for item in population if item not in seen]
    if missing:
        raise LegacyBypassInventoryError(f"missing {name}: {missing[0]}")


def _normalize_hits(
    hits: Sequence[Mapping[str, Any]], path_id: str
) -> tuple[Mapping[str, Any], ...]:
    normalized: list[Mapping[str, Any]] = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            raise LegacyBypassInventoryError(f"{path_id}.sample_hits[{index}] must be a mapping")
        relpath = _text(hit.get("relpath"), f"{path_id}.sample_hits[{index}].relpath")
        line = hit.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise LegacyBypassInventoryError(
                f"{path_id}.sample_hits[{index}].line must be a positive integer"
            )
        needle = _text(hit.get("needle"), f"{path_id}.sample_hits[{index}].needle")
        normalized.append(
            MappingProxyType({"relpath": relpath, "line": line, "needle": needle})
        )
        if len(normalized) >= SAMPLE_HIT_LIMIT:
            break
    return tuple(normalized)


def _normalize_paths(records: Sequence[PathObservation]) -> tuple[PathObservation, ...]:
    normalized: list[PathObservation] = []
    for record in records:
        path_id = _text(record.path_id, "path_id")
        recipe = CATALOG_BY_ID.get(path_id)
        if recipe is None:
            raise LegacyBypassInventoryError(f"unknown path_id: {path_id}")
        category = _text(record.category, f"{path_id}.category")
        if category not in PATH_CATEGORIES:
            raise LegacyBypassInventoryError(f"{path_id}.category is not admitted")
        if category != recipe.category:
            raise LegacyBypassInventoryError(
                f"{path_id}.category must remain {recipe.category}"
            )
        repository = _text(record.repository, f"{path_id}.repository")
        if repository != recipe.repository:
            raise LegacyBypassInventoryError(
                f"{path_id}.repository must remain {recipe.repository}"
            )
        disposition = _text(record.disposition, f"{path_id}.disposition")
        if disposition not in PATH_DISPOSITIONS:
            raise LegacyBypassInventoryError(f"{path_id}.disposition is not admitted")
        kind = _kind(record.evidence_kind, f"{path_id}.evidence_kind")
        if kind == "estimated":
            raise LegacyBypassInventoryError(
                f"{path_id}: estimated inventory values cannot mint presence or absence"
            )
        present = record.present
        if present is not None and not isinstance(present, bool):
            raise LegacyBypassInventoryError(
                f"{path_id}.present must be a boolean or null"
            )
        if kind == "simulated" and present is True:
            raise LegacyBypassInventoryError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "simulated" and present is False:
            raise LegacyBypassInventoryError(
                "simulated absence cannot certify that a false-authority path is gone"
            )
        if kind == "unavailable" and present is not None:
            raise LegacyBypassInventoryError(
                f"{path_id} unavailable evidence must not record a boolean present"
            )
        if kind == LIVE_SATISFYING_KIND:
            raise LegacyBypassInventoryError(
                f"{path_id}: static inventory cannot carry measured_live evidence"
            )
        hit_count = record.hit_count
        if isinstance(hit_count, bool) or not isinstance(hit_count, int) or hit_count < 0:
            raise LegacyBypassInventoryError(
                f"{path_id}.hit_count must be a non-negative integer"
            )
        if kind == "unavailable" and hit_count != 0:
            raise LegacyBypassInventoryError(
                f"{path_id} unavailable evidence cannot record hits"
            )
        if present is True and hit_count < 1:
            raise LegacyBypassInventoryError(
                f"{path_id} present paths require at least one measured hit"
            )
        if present is False and hit_count != 0:
            raise LegacyBypassInventoryError(
                f"{path_id} absent paths cannot record hits"
            )
        if not isinstance(record.source_identity, Mapping):
            raise LegacyBypassInventoryError(f"{path_id}.source_identity must be a mapping")
        normalized.append(
            PathObservation(
                path_id=path_id,
                category=category,
                repository=repository,
                remediating_task=_text(
                    record.remediating_task, f"{path_id}.remediating_task"
                ),
                disposition=disposition,
                authority_claim=_text(
                    record.authority_claim, f"{path_id}.authority_claim"
                ),
                migration_impact=_text(
                    record.migration_impact, f"{path_id}.migration_impact"
                ),
                present=present,
                evidence_kind=kind,
                hit_count=hit_count,
                sample_hits=_normalize_hits(record.sample_hits, path_id),
                source_identity=MappingProxyType(dict(record.source_identity)),
                reason=str(record.reason or ""),
            )
        )
    ids = [item.path_id for item in normalized]
    _require_unique(ids, REQUIRED_PATH_IDS, "path_id")
    by_id = {item.path_id: item for item in normalized}
    return tuple(by_id[name] for name in REQUIRED_PATH_IDS)


def _normalize_competing(
    records: Sequence[CompetingAuthorityObservation],
) -> tuple[CompetingAuthorityObservation, ...]:
    normalized: list[CompetingAuthorityObservation] = []
    for record in records:
        name = _text(record.name, "competing_authority.name")
        kind = _kind(record.evidence_kind, f"{name}.evidence_kind")
        scope = _text(record.scope, f"{name}.scope")
        if scope not in COMPETING_AUTHORITY_SCOPES:
            raise LegacyBypassInventoryError(
                f"{name}.scope is not an admitted competing-authority scope"
            )
        present = record.present
        if present is not None and not isinstance(present, bool):
            raise LegacyBypassInventoryError(
                f"{name}.present must be a boolean or null"
            )
        if kind == "simulated" and present is False:
            raise LegacyBypassInventoryError(
                "simulated absence cannot certify that competing authorities are prohibited"
            )
        if kind == "simulated" and present is True:
            raise LegacyBypassInventoryError(
                "simulated competing-authority presence cannot be represented as live"
            )
        if kind == "unavailable" and present is not None:
            raise LegacyBypassInventoryError(
                f"{name} unavailable competing-authority evidence must not record a boolean"
            )
        if kind == LIVE_SATISFYING_KIND:
            raise LegacyBypassInventoryError(
                f"{name}: static inventory cannot carry measured_live competing-authority evidence"
            )
        if kind == "estimated":
            raise LegacyBypassInventoryError(
                f"{name}: estimated competing-authority values are not admitted"
            )
        normalized.append(
            CompetingAuthorityObservation(
                name=name,
                present=present,
                evidence_kind=kind,
                scope=scope,
                reason=str(record.reason or ""),
            )
        )
    names = [item.name for item in normalized]
    _require_unique(names, COMPETING_AUTHORITY_PROHIBITIONS, "competing authority")
    by_id = {item.name: item for item in normalized}
    return tuple(by_id[name] for name in COMPETING_AUTHORITY_PROHIBITIONS)


def inventory_legacy_bypass_and_false_authority_paths(
    *,
    paths: Sequence[PathObservation],
    competing_authorities: Sequence[CompetingAuthorityObservation],
    duckdb_or_quack_state_written: bool = False,
    live_runtime_inventory: bool = False,
) -> InventoryVerdict:
    """Evaluate PCPR-003.  Inventory is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not write DuckDB or Quack state"
        )
    if live_runtime_inventory:
        raise LegacyBypassInventoryError(
            "static source inventory cannot claim a live runtime inventory"
        )

    normalized_paths = _normalize_paths(paths)
    normalized_competing = _normalize_competing(competing_authorities)

    blockers: list[str] = []
    measured_present = sum(
        1
        for item in normalized_paths
        if item.present is True and item.evidence_kind == "measured"
    )
    measured_absent = sum(
        1
        for item in normalized_paths
        if item.present is False and item.evidence_kind in {"measured", "measured_hermetic"}
    )
    unavailable = sum(
        1 for item in normalized_paths if item.evidence_kind == "unavailable"
    )
    this_task_created = any(
        item.scope == "this_task" and item.present is True
        for item in normalized_competing
    )
    if this_task_created:
        blockers.append("this_task_created_competing_authority")

    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable == len(normalized_paths):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise LegacyBypassInventoryError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not mint a closed release outcome"
        )

    payload = {
        "schema": INVENTORY_VERDICT_SCHEMA,
        "interface": INVENTORY_INTERFACE,
        "task_id": PCPR_003_TASK_ID,
        "goal_id": PCPR_003_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "competing_authorities_prohibited": False,
        "live_runtime_inventory": False,
        "duckdb_or_quack_state_written": False,
        "catalog_complete": True,
        "measured_present_count": measured_present,
        "measured_absent_count": measured_absent,
        "unavailable_count": unavailable,
        "this_task_created_competing_authority": this_task_created,
        "path_ids": [item.path_id for item in normalized_paths],
        "path_present": [item.present for item in normalized_paths],
        "path_evidence_kinds": [item.evidence_kind for item in normalized_paths],
        "competing_authorities": [item.to_mapping() for item in normalized_competing],
        "blockers": list(dict.fromkeys(blockers)),
    }
    verdict_cid = content_identity(payload)
    return InventoryVerdict(
        schema=INVENTORY_VERDICT_SCHEMA,
        interface=INVENTORY_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        competing_authorities_prohibited=False,
        live_runtime_inventory=False,
        duckdb_or_quack_state_written=False,
        catalog_complete=True,
        measured_present_count=measured_present,
        measured_absent_count=measured_absent,
        unavailable_count=unavailable,
        this_task_created_competing_authority=this_task_created,
        paths=normalized_paths,
        competing_authorities=normalized_competing,
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=verdict_cid,
    )


def discover_portfolio_root(start: Path | None = None) -> Path | None:
    """Locate the portfolio worktree that contains the three source gitlinks."""

    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        marker = (
            candidate
            / "artifacts"
            / "proof_carrying_platform_qualification_and_release"
        )
        accel = candidate / "external" / "ipfs_accelerate"
        if marker.is_dir() and accel.is_dir():
            return candidate
    return None


def discover_accelerate_root(start: Path | None = None) -> Path | None:
    """Locate the Accelerate source checkout from this module."""

    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "ipfs_accelerate_py").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    return None


def _repo_root_for(
    repository: str,
    *,
    portfolio_root: Path | None,
    accelerate_root: Path | None,
) -> Path | None:
    if repository == "ipfs_accelerate_py":
        if portfolio_root is not None:
            nested = portfolio_root / "external" / "ipfs_accelerate"
            if nested.is_dir():
                return nested
        return accelerate_root
    if repository == "portfolio":
        return portfolio_root
    if portfolio_root is None:
        return None
    relative = REPO_ROOT_KEYS.get(repository)
    if relative is None:
        return None
    path = portfolio_root / relative
    return path if path.is_dir() else None


def _probe_file(path: Path, needles: Sequence[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    try:
        size = path.stat().st_size
    except OSError:
        return hits
    if size > MAX_PROBE_FILE_BYTES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:MAX_PROBE_FILE_BYTES]
        except OSError:
            return hits
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return hits
    for line_number, line in enumerate(text.splitlines(), start=1):
        for needle in needles:
            if needle and needle in line:
                hits.append(
                    {"relpath": path.name, "line": line_number, "needle": needle}
                )
                if len(hits) >= SAMPLE_HIT_LIMIT:
                    return hits
    return hits


def _file_digest(path: Path) -> tuple[str | None, int | None]:
    try:
        payload = path.read_bytes()
    except OSError:
        return (None, None)
    return (hashlib.sha256(payload).hexdigest(), len(payload))


def scan_path_catalog(
    *,
    portfolio_root: Path | None = None,
    accelerate_root: Path | None = None,
    source_commits: Mapping[str, str] | None = None,
    source_trees: Mapping[str, str] | None = None,
) -> tuple[PathObservation, ...]:
    """Probe named catalog files.  Missing trees stay typed unavailable."""

    commits = source_commits or {}
    trees = source_trees or {}
    observations: list[PathObservation] = []
    for recipe in PATH_CLASS_CATALOG:
        root = _repo_root_for(
            recipe.repository,
            portfolio_root=portfolio_root,
            accelerate_root=accelerate_root,
        )
        if root is None or not root.is_dir():
            observations.append(
                PathObservation(
                    path_id=recipe.path_id,
                    category=recipe.category,
                    repository=recipe.repository,
                    remediating_task=recipe.remediating_task,
                    disposition=recipe.disposition,
                    authority_claim=recipe.authority_claim,
                    migration_impact=recipe.migration_impact,
                    present=None,
                    evidence_kind="unavailable",
                    hit_count=0,
                    sample_hits=(),
                    source_identity=MappingProxyType(
                        {
                            "repository": recipe.repository,
                            "commit": commits.get(recipe.repository),
                            "tree": trees.get(recipe.repository),
                            "relpaths": list(recipe.relpaths),
                        }
                    ),
                    reason=(
                        f"{recipe.repository} source tree is not present in this "
                        "environment and is not represented as empty or passing."
                    ),
                )
            )
            continue
        hits: list[dict[str, Any]] = []
        existing: list[str] = []
        primary_sha: str | None = None
        primary_bytes: int | None = None
        for relpath in recipe.relpaths:
            candidate = root / relpath
            if not candidate.is_file():
                continue
            existing.append(relpath)
            digest, nbytes = _file_digest(candidate)
            if primary_sha is None:
                primary_sha = digest
                primary_bytes = nbytes
            file_hits = _probe_file(candidate, recipe.needles)
            for hit in file_hits:
                hits.append(
                    {
                        "relpath": relpath,
                        "line": hit["line"],
                        "needle": hit["needle"],
                    }
                )
            if len(hits) >= SAMPLE_HIT_LIMIT:
                hits = hits[:SAMPLE_HIT_LIMIT]
                break
        if not existing:
            observations.append(
                PathObservation(
                    path_id=recipe.path_id,
                    category=recipe.category,
                    repository=recipe.repository,
                    remediating_task=recipe.remediating_task,
                    disposition=recipe.disposition,
                    authority_claim=recipe.authority_claim,
                    migration_impact=recipe.migration_impact,
                    present=None,
                    evidence_kind="unavailable",
                    hit_count=0,
                    sample_hits=(),
                    source_identity=MappingProxyType(
                        {
                            "repository": recipe.repository,
                            "commit": commits.get(recipe.repository),
                            "tree": trees.get(recipe.repository),
                            "relpaths": list(recipe.relpaths),
                        }
                    ),
                    reason="Nominated probe files are not present; absence is not recorded as zero hits.",
                )
            )
            continue
        present = bool(hits)
        observations.append(
            PathObservation(
                path_id=recipe.path_id,
                category=recipe.category,
                repository=recipe.repository,
                remediating_task=recipe.remediating_task,
                disposition=recipe.disposition,
                authority_claim=recipe.authority_claim,
                migration_impact=recipe.migration_impact,
                present=present,
                evidence_kind="measured",
                hit_count=len(hits),
                sample_hits=tuple(MappingProxyType(dict(item)) for item in hits),
                source_identity=MappingProxyType(
                    {
                        "repository": recipe.repository,
                        "commit": commits.get(recipe.repository),
                        "tree": trees.get(recipe.repository),
                        "relpaths": existing,
                        "primary_relpath": existing[0],
                        "primary_sha256": primary_sha,
                        "primary_bytes": primary_bytes,
                    }
                ),
                reason=(
                    "Static source probe of named catalog files. Not a live "
                    "runtime campaign and not a closed release."
                    if present
                    else (
                        "Named probe files were readable and contained none of "
                        "the catalog needles. This is not a live prohibition."
                    )
                ),
            )
        )
    return tuple(observations)


def competing_authorities_from_paths(
    paths: Sequence[PathObservation],
) -> tuple[CompetingAuthorityObservation, ...]:
    """Map catalog probes onto the PCPR-002 competing-authority names."""

    by_id = {item.path_id: item for item in paths}
    observations: list[CompetingAuthorityObservation] = []
    for name in COMPETING_AUTHORITY_PROHIBITIONS:
        path_ids = COMPETING_AUTHORITY_PATH_IDS.get(name, ())
        if not path_ids:
            observations.append(
                CompetingAuthorityObservation(
                    name=name,
                    present=None,
                    evidence_kind="unavailable",
                    scope="repository",
                    reason=(
                        "Static source inventory cannot prove repository-wide "
                        f"absence of {name} as a live prohibition. This worker "
                        "did not create one."
                    ),
                )
            )
            continue
        matched = [by_id[item] for item in path_ids if item in by_id]
        if any(item.evidence_kind == "unavailable" for item in matched) and not any(
            item.present is True for item in matched
        ):
            observations.append(
                CompetingAuthorityObservation(
                    name=name,
                    present=None,
                    evidence_kind="unavailable",
                    scope="repository",
                    reason=(
                        f"{name} probe trees were not present; presence is not "
                        "recorded as false or zero."
                    ),
                )
            )
            continue
        present = any(item.present is True for item in matched)
        observations.append(
            CompetingAuthorityObservation(
                name=name,
                present=present,
                evidence_kind="measured",
                scope="repository",
                reason=(
                    f"Static catalog probe for {', '.join(path_ids)}. "
                    "Not a live freeze prohibition."
                ),
            )
        )
    return tuple(observations)


def current_head_path_observations() -> tuple[PathObservation, ...]:
    """Scan the current worktree.  Missing siblings stay typed unavailable."""

    return scan_path_catalog(
        portfolio_root=discover_portfolio_root(),
        accelerate_root=discover_accelerate_root(),
        source_commits={
            "ipfs_accelerate_py": CURRENT_HEAD_ACCELERATOR_COMMIT,
            "ipfs_datasets_py": CURRENT_HEAD_DATASETS_COMMIT,
            "ipfs_kit_py": CURRENT_HEAD_KIT_COMMIT,
            "portfolio": CURRENT_HEAD_OUTER_COMMIT,
        },
        source_trees={
            "ipfs_accelerate_py": CURRENT_HEAD_ACCELERATOR_TREE,
            "ipfs_datasets_py": CURRENT_HEAD_DATASETS_TREE,
            "ipfs_kit_py": CURRENT_HEAD_KIT_TREE,
            "portfolio": CURRENT_HEAD_OUTER_TREE,
        },
    )


def current_head_competing_authorities() -> tuple[CompetingAuthorityObservation, ...]:
    """Repository-scope competing-authority rows from the current-head scan."""

    return competing_authorities_from_paths(current_head_path_observations())


def qualify_current_head_inventory() -> InventoryVerdict:
    """Ordinary current-head PCPR-003 evaluation: honest R&D inventory."""

    paths = current_head_path_observations()
    return inventory_legacy_bypass_and_false_authority_paths(
        paths=paths,
        competing_authorities=competing_authorities_from_paths(paths),
    )


# Pinned identity of the ordinary current-head inventory verdict.  Drift
# means the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraossnqpwfmgxwzhx4bmaaqdg2oiumoqujgxh2uz3dhzconolfepqa"
)


def pcpr_003_receipt_promotion(verdict: InventoryVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields.  Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not write DuckDB or Quack state"
        )
    if verdict.live_runtime_inventory:
        raise LegacyBypassInventoryError(
            "static inventory cannot claim live runtime inventory"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise LegacyBypassInventoryError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise LegacyBypassInventoryError(
            "promotion_status is not an admitted PCPR-003 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory cannot promote the supervisor"
        )
    return {
        "schema": INVENTORY_VERDICT_SCHEMA,
        "interface": INVENTORY_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "competing_authorities_prohibited": False,
        "live_runtime_inventory": False,
        "duckdb_or_quack_state_written": False,
        "catalog_complete": verdict.catalog_complete,
        "measured_present_count": verdict.measured_present_count,
        "measured_absent_count": verdict.measured_absent_count,
        "unavailable_count": verdict.unavailable_count,
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_003_receipt_promotion() -> dict[str, Any]:
    """Fail-closed promotion section for the ordinary current-head inventory."""

    return pcpr_003_receipt_promotion(qualify_current_head_inventory())


def pcpr_003_receipt_paths(verdict: InventoryVerdict) -> dict[str, Any]:
    """Outer-receipt path catalog with source identity and disposition."""

    return {
        "schema": INVENTORY_SCHEMA,
        "catalog_complete_relative_to_closed_pcpr_list": True,
        "exhaustive_of_unlisted_runtime_bypasses": False,
        "required_path_ids": list(REQUIRED_PATH_IDS),
        "catalog": [item.to_mapping() for item in verdict.paths],
        "evidence_kind": "measured",
    }


def pcpr_003_receipt_competing_authorities(verdict: InventoryVerdict) -> dict[str, Any]:
    """Outer-receipt competing-authority section.  Not a live freeze."""

    return {
        "prohibited_by_inventory": False,
        "inventory_task": PCPR_003_TASK_ID,
        "freeze_task": PCPR_002_TASK_ID,
        "this_task_created_competing_authority": False,
        "repository_live_inventory": False,
        "repository_static_inventory": True,
        "observations": [item.to_mapping() for item in verdict.competing_authorities],
        "evidence_kind": "measured",
    }


def pcpr_003_receipt_negative_results() -> dict[str, Any]:
    """Fixed negative results for the PCPR-003 R&D inventory."""

    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_absence_cannot_prohibit": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "static_inventory_cannot_freeze_contracts": True,
        "static_inventory_cannot_prohibit_competing_authorities_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_003_receipt_sections() -> dict[str, Any]:
    """Promotion, path, competing-authority, and negative sections."""

    verdict = qualify_current_head_inventory()
    promotion = pcpr_003_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_001_TASK_ID,
            "goal_id": "PCPR-G120",
            "promotion_status": qualification.promotion_status,
            "supervisor_disposition": qualification.supervisor_disposition,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": qualification.verdict_cid,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-001 current-head qualification remains rnd_non_promoted; "
                "this inventory does not promote the supervisor."
            ),
            "evidence_kind": "measured",
        },
        "path_inventory": pcpr_003_receipt_paths(verdict),
        "competing_authorities": pcpr_003_receipt_competing_authorities(verdict),
        "negative_results": pcpr_003_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
        "live_runtime_inventory": False,
    }


def pcpr_003_current_tree_binding(
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
    """Measured current-tree identities for a PCPR-003 outer receipt."""

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
        raise LegacyBypassInventoryError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise LegacyBypassInventoryError("datasets_commit must equal datasets_gitlink")
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise LegacyBypassInventoryError("kit_commit must equal kit_gitlink")
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


def current_head_pcpr_003_current_tree_binding() -> dict[str, Any]:
    """Measured current-tree binding for this isolated worktree."""

    return pcpr_003_current_tree_binding(
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


def validate_pcpr_003_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-003 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise LegacyBypassInventoryError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_003_TASK_ID:
        raise LegacyBypassInventoryError("outer receipt task_id must be PCPR-003")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise LegacyBypassInventoryError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise LegacyBypassInventoryError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise LegacyBypassInventoryError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise LegacyBypassInventoryError(
            "qualification_verdict.closed_release_outcome must be null"
        )
    if verdict_section.get("release_claim") is True:
        raise LegacyBypassInventoryError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory must not write DuckDB or Quack state"
        )
    if verdict_section.get("live_runtime_inventory") is True:
        raise LegacyBypassInventoryError(
            "qualification_verdict must not claim live runtime inventory"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory cannot freeze contracts"
        )
    if verdict_section.get("competing_authorities_prohibited") is True:
        raise LegacyBypassInventoryError(
            "static inventory cannot prohibit competing authorities as a live freeze"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise LegacyBypassInventoryError(
            "qualification_verdict.promotion_status is not an admitted PCPR-003 status"
        )
    if promotion_status == "supervisor_promoted":
        raise LegacyBypassInventoryError(
            "legacy-bypass inventory cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise LegacyBypassInventoryError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise LegacyBypassInventoryError("acceptance must not claim a release")
        if acceptance.get("promotion_status") not in {None, promotion_status}:
            raise LegacyBypassInventoryError(
                "acceptance.promotion_status must match qualification_verdict"
            )

    binding = payload.get("current_tree_binding")
    if isinstance(binding, Mapping):
        if binding.get("evidence_kind") != "measured":
            raise LegacyBypassInventoryError(
                "current_tree_binding.evidence_kind must be measured"
            )
        if binding.get("origin_main_is_ancestor") is not True:
            raise LegacyBypassInventoryError(
                "current_tree_binding.origin_main_is_ancestor must be true"
            )
        _reject_closed_release_value(binding.get("outer_subject"), "outer_subject")
        for name in (
            "outer_commit",
            "outer_tree",
            "origin_main",
            "accelerator_pre_change_commit",
            "accelerator_pre_change_tree",
            "accelerator_gitlink",
            "accelerator_origin_main",
        ):
            _git_object_id(binding.get(name), f"current_tree_binding.{name}")

    expected = current_head_pcpr_003_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise LegacyBypassInventoryError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise LegacyBypassInventoryError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise LegacyBypassInventoryError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    prerequisite = payload.get("qualification_prerequisite")
    if isinstance(prerequisite, Mapping):
        if prerequisite.get("task_id") not in {None, PCPR_001_TASK_ID}:
            raise LegacyBypassInventoryError(
                "qualification_prerequisite.task_id must be PCPR-001"
            )
        if prerequisite.get("verdict_cid") not in {
            None,
            CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
        }:
            raise LegacyBypassInventoryError(
                "qualification_prerequisite.verdict_cid must match PCPR-001"
            )
        _reject_closed_release_value(
            prerequisite.get("promotion_status"),
            "qualification_prerequisite.promotion_status",
        )

    return {
        "valid": True,
        "task_id": PCPR_003_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "COMPETING_AUTHORITY_PROHIBITIONS",
    "CURRENT_HEAD_ACCELERATOR_COMMIT",
    "CURRENT_HEAD_DATASETS_COMMIT",
    "CURRENT_HEAD_KIT_COMMIT",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "CURRENT_HEAD_UNAVAILABLE_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INVENTORY_INTERFACE",
    "PATH_CLASS_CATALOG",
    "PCPR_003_GOAL_ID",
    "PCPR_003_TASK_ID",
    "REQUIRED_PATH_IDS",
    "SEALED_GIT_BINARY",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "SOURCE_REPOSITORIES",
    "InventoryVerdict",
    "LegacyBypassInventoryError",
    "PathObservation",
    "competing_authorities_from_paths",
    "current_head_competing_authorities",
    "current_head_path_observations",
    "current_head_pcpr_003_current_tree_binding",
    "current_head_pcpr_003_receipt_promotion",
    "current_head_pcpr_003_receipt_sections",
    "discover_accelerate_root",
    "discover_portfolio_root",
    "inventory_legacy_bypass_and_false_authority_paths",
    "pcpr_003_current_tree_binding",
    "pcpr_003_receipt_promotion",
    "qualify_current_head_inventory",
    "scan_path_catalog",
    "validate_pcpr_003_outer_receipt",
)
