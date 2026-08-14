#!/usr/bin/env python3
"""Deterministic fixture repository history and proof-graph generator (IPS-045).

Builds tiny, byte-stable repository histories and reason-labeled proof graphs
covering every required mutation class.  Recipes stay compact; expected
invalidation closures and full-fallback decisions are derived from the
datasets semantic authorities so consumers never invent product rules.

Rules:

* two clean generations are byte-identical (no wall clock, host, or env);
* every scenario binds explicit parents, changed-artifact provenance, direct
  and transitive unit closures, aggregate effect, and full-fallback decision;
* simulated proof modes are labeled and never production seal success;
* re-run this module to refresh ``fixture_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

# Allow ``python test/fixtures/.../generate_fixture_history.py`` from a clean
# checkout when the nested datasets package is not already on sys.path.
# Prefer the repository root so the outer ``ipfs_datasets_py`` bootstrap package
# (which extends __path__ to the inner package) is used, matching conftest.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from ipfs_datasets_py.logic.zkp.incremental_sealing.dependency_graph import (  # noqa: E402
    DependencyEdgeType,
    DependencyNodeKind,
    ProofDependencyGraph,
    sample_reason,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (  # noqa: E402
    ProofMode,
    SealStatus,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.identity import (  # noqa: E402
    ABSENCE_TOKEN,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.invalidation import (  # noqa: E402
    compute_invalidation_closure,
    sample_invalidation_policy,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.repository_diff import (  # noqa: E402
    ChangeClass,
    PathClassificationPolicy,
    diff_repository_states,
    sample_artifact,
    sample_repository_state,
)

# ---------------------------------------------------------------------------
# Corpus identity
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = (
    "ipfs_accelerate_py/incremental-proof-sealer/fixture-history@1"
)
CORPUS_ID: Final[str] = "incremental-proof-sealer-fixture-corpus-v1"
EVIDENCE_SUBSET: Final[str] = "ips/fixture-corpus@1"
DESCRIPTION: Final[str] = (
    "Hermetic deterministic repository histories and proof graphs for "
    "incremental-proof-sealer positive invalidation, reuse, branch, merge, "
    "and rollback conformance. Simulated evidence is labeled and never "
    "models production seal success."
)
REPOSITORY_ID: Final[str] = "repo/incremental-proof-sealer-fixture"

# Closed scenario catalogue required by IPS-045 acceptance.
REQUIRED_SCENARIOS: Final[tuple[str, ...]] = (
    "selector",
    "fixture",
    "configuration",
    "network_policy",
    "policy",
    "lock",
    "tool",
    "schema",
    "canonicalization",
    "checked_spec",
    "source",
    "test",
    "circuit",
    "key",
    "documentation",
    "branch",
    "merge",
    "rollback",
)

# Additional corpus scenarios (not in the required list but still generated).
AUXILIARY_SCENARIOS: Final[tuple[str, ...]] = (
    "genesis",
    "source_interface",
    "unrelated_source",
    "simulated_evidence",
)

GENESIS_REVISION: Final[str] = "rev-0000000000000000000000000000000000000000"

# Base inventory paths (tiny multi-module fixture repository).
PATH_SOURCE_A: Final[str] = "pkg/mod_a.py"
PATH_SOURCE_B: Final[str] = "pkg/mod_b.py"
PATH_INTERFACE: Final[str] = "pkg/api.py"
PATH_TEST: Final[str] = "tests/test_mod_a.py"
PATH_FIXTURE: Final[str] = "tests/fixtures/data.json"
PATH_LOCK: Final[str] = "poetry.lock"
PATH_CONFIG: Final[str] = "config/app.toml"
PATH_TOOL: Final[str] = "tools/prover_version.json"
PATH_CIRCUIT: Final[str] = "circuits/prove.circom"
PATH_PROVING_KEY: Final[str] = "keys/pk_main.pkey"
PATH_VERIFICATION_KEY: Final[str] = "keys/vk_main.vkey"
PATH_SELECTOR: Final[str] = "selectors/unit.json"
PATH_POLICY: Final[str] = "policy/verification_policy.json"
PATH_NETWORK_POLICY: Final[str] = "policy/network_policy.json"
PATH_CANONICALIZATION: Final[str] = "meta/canonicalization.json"
PATH_PROOF_SCHEMA: Final[str] = "meta/proof_schema.json"
PATH_DOCS: Final[str] = "docs/guide.md"
PATH_CHECKED_SPEC: Final[str] = "docs/checked/soundness.md"
PATH_ENVIRONMENT: Final[str] = "environment/trust_policy.json"

BASE_PATHS: Final[tuple[str, ...]] = (
    PATH_SOURCE_A,
    PATH_SOURCE_B,
    PATH_INTERFACE,
    PATH_TEST,
    PATH_FIXTURE,
    PATH_LOCK,
    PATH_CONFIG,
    PATH_TOOL,
    PATH_CIRCUIT,
    PATH_PROVING_KEY,
    PATH_VERIFICATION_KEY,
    PATH_SELECTOR,
    PATH_POLICY,
    PATH_NETWORK_POLICY,
    PATH_CANONICALIZATION,
    PATH_PROOF_SCHEMA,
    PATH_DOCS,
    PATH_CHECKED_SPEC,
    PATH_ENVIRONMENT,
)

# Graph node ids.
NODE_ARTIFACT_A: Final[str] = "artifact/pkg/mod_a.py"
NODE_ARTIFACT_B: Final[str] = "artifact/pkg/mod_b.py"
NODE_INTERFACE: Final[str] = "artifact/pkg/api.py"
NODE_FIXTURE: Final[str] = "fixture/tests/fixtures/data.json"
NODE_CONFIG: Final[str] = "config/config/app.toml"
NODE_TOOL: Final[str] = "config/tools/prover_version.json"
NODE_LOCK: Final[str] = "config/poetry.lock"
NODE_SELECTOR: Final[str] = "config/selectors/unit.json"
NODE_POLICY: Final[str] = "policy/verification"
NODE_NETWORK: Final[str] = "policy/network"
NODE_SCHEMA: Final[str] = "schema/proof"
NODE_CANON: Final[str] = "config/meta/canonicalization.json"
NODE_CIRCUIT: Final[str] = "config/circuits/prove.circom"
NODE_VK: Final[str] = "config/keys/vk_main.vkey"
NODE_PK: Final[str] = "config/keys/pk_main.pkey"
NODE_CHECKED: Final[str] = "artifact/docs/checked/soundness.md"
NODE_UNIT_STATIC: Final[str] = "unit/static-mod-a"
NODE_UNIT_TEST: Final[str] = "unit/test-mod-a"
NODE_UNIT_FORMAL: Final[str] = "unit/formal-mod-a"
NODE_UNIT_B: Final[str] = "unit/static-mod-b"
NODE_AGG_MAIN: Final[str] = "aggregate/main"
NODE_AGG_B: Final[str] = "aggregate/mod-b"

KNOWN_UNITS: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            NODE_UNIT_STATIC,
            NODE_UNIT_TEST,
            NODE_UNIT_FORMAL,
            NODE_UNIT_B,
            NODE_AGG_MAIN,
            NODE_AGG_B,
        )
    )
)

PATH_TO_NODE_IDS: Final[dict[str, tuple[str, ...]]] = {
    PATH_SOURCE_A: (NODE_ARTIFACT_A,),
    PATH_SOURCE_B: (NODE_ARTIFACT_B,),
    PATH_INTERFACE: (NODE_INTERFACE,),
    PATH_TEST: (NODE_UNIT_TEST,),
    PATH_FIXTURE: (NODE_FIXTURE,),
    PATH_LOCK: (NODE_LOCK,),
    PATH_CONFIG: (NODE_CONFIG,),
    PATH_TOOL: (NODE_TOOL,),
    PATH_CIRCUIT: (NODE_CIRCUIT,),
    PATH_PROVING_KEY: (NODE_PK,),
    PATH_VERIFICATION_KEY: (NODE_VK,),
    PATH_SELECTOR: (NODE_SELECTOR,),
    PATH_POLICY: (NODE_POLICY,),
    PATH_NETWORK_POLICY: (NODE_NETWORK,),
    PATH_CANONICALIZATION: (NODE_CANON,),
    PATH_PROOF_SCHEMA: (NODE_SCHEMA,),
    PATH_CHECKED_SPEC: (NODE_CHECKED,),
    # Ordinary documentation intentionally unmapped so it never seeds.
}


def _content_label(path: str, version: str) -> str:
    return f"{path}@{version}"


def _revision_for(scenario: str, *, index: int) -> str:
    """Stable 40-char revision id derived only from scenario identity.

    Avoids host/time entropy.  Uses a padded scenario token rather than a
    cryptographic digest so the compact checked-in catalog remains easy to
    audit and regenerate without bulk golden dumps.
    """

    token = f"{index:02d}-{scenario}".replace("_", "-")
    body = (token + ("0" * 40))[:40]
    return f"rev-{body}"


def _sha256_hex(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def classification_policy() -> PathClassificationPolicy:
    return PathClassificationPolicy(
        interface_paths=(PATH_INTERFACE,),
        checked_specification_paths=(PATH_CHECKED_SPEC,),
        generated_input_paths=(),
        class_overrides=(
            # Tool/prover version is configuration-class for classification
            # (no dedicated ChangeClass) but maps to its own graph seed.
            (PATH_TOOL, ChangeClass.CONFIGURATION.value),
            (PATH_PROOF_SCHEMA, ChangeClass.CANONICALIZATION.value),
        ),
        treat_unknown_as_ambiguous=True,
        treat_dependency_lock_as_full_fallback=True,
    )


def build_proof_graph() -> ProofDependencyGraph:
    """Tiny multi-module reason-labeled dependency graph for the corpus."""

    graph = ProofDependencyGraph()
    nodes: tuple[tuple[str, DependencyNodeKind, str], ...] = (
        (NODE_ARTIFACT_A, DependencyNodeKind.ARTIFACT, PATH_SOURCE_A),
        (NODE_ARTIFACT_B, DependencyNodeKind.ARTIFACT, PATH_SOURCE_B),
        (NODE_INTERFACE, DependencyNodeKind.ARTIFACT, PATH_INTERFACE),
        (NODE_FIXTURE, DependencyNodeKind.FIXTURE, PATH_FIXTURE),
        (NODE_CONFIG, DependencyNodeKind.CONFIG, PATH_CONFIG),
        (NODE_TOOL, DependencyNodeKind.CONFIG, PATH_TOOL),
        (NODE_LOCK, DependencyNodeKind.CONFIG, PATH_LOCK),
        (NODE_SELECTOR, DependencyNodeKind.CONFIG, PATH_SELECTOR),
        (NODE_POLICY, DependencyNodeKind.POLICY, PATH_POLICY),
        (NODE_NETWORK, DependencyNodeKind.POLICY, PATH_NETWORK_POLICY),
        (NODE_SCHEMA, DependencyNodeKind.SCHEMA, PATH_PROOF_SCHEMA),
        (NODE_CANON, DependencyNodeKind.CONFIG, PATH_CANONICALIZATION),
        (NODE_CIRCUIT, DependencyNodeKind.CONFIG, PATH_CIRCUIT),
        (NODE_VK, DependencyNodeKind.CONFIG, PATH_VERIFICATION_KEY),
        (NODE_PK, DependencyNodeKind.CONFIG, PATH_PROVING_KEY),
        (NODE_CHECKED, DependencyNodeKind.ARTIFACT, PATH_CHECKED_SPEC),
        (NODE_UNIT_STATIC, DependencyNodeKind.UNIT, "static-mod-a"),
        (NODE_UNIT_TEST, DependencyNodeKind.UNIT, "test-mod-a"),
        (NODE_UNIT_FORMAL, DependencyNodeKind.UNIT, "formal-mod-a"),
        (NODE_UNIT_B, DependencyNodeKind.UNIT, "static-mod-b"),
        (NODE_AGG_MAIN, DependencyNodeKind.AGGREGATE, "aggregate-main"),
        (NODE_AGG_B, DependencyNodeKind.AGGREGATE, "aggregate-mod-b"),
    )
    for node_id, kind, label in nodes:
        graph.add_node(node_id, kind, label=label)

    def edge(src: str, dst: str, edge_type: DependencyEdgeType, reason: str) -> None:
        graph.add_edge(src, dst, edge_type, sample_reason(reason))

    # Module A chain: artifact -> static -> formal -> aggregate; test parallel.
    edge(NODE_ARTIFACT_A, NODE_UNIT_STATIC, DependencyEdgeType.SOURCE_DEPENDS_ON, "a-static")
    edge(NODE_ARTIFACT_A, NODE_UNIT_TEST, DependencyEdgeType.TEST_COVERS, "a-test")
    edge(NODE_INTERFACE, NODE_UNIT_STATIC, DependencyEdgeType.IMPORTS, "api-static")
    edge(NODE_INTERFACE, NODE_UNIT_TEST, DependencyEdgeType.TEST_COVERS, "api-test")
    edge(NODE_UNIT_STATIC, NODE_UNIT_FORMAL, DependencyEdgeType.PROOF_DEPENDS_ON, "static-formal")
    edge(NODE_UNIT_TEST, NODE_UNIT_FORMAL, DependencyEdgeType.PROOF_DEPENDS_ON, "test-formal")
    edge(NODE_FIXTURE, NODE_UNIT_TEST, DependencyEdgeType.FIXTURE_DEPENDS_ON, "fixture-test")
    edge(NODE_CONFIG, NODE_UNIT_TEST, DependencyEdgeType.CONFIG_DEPENDS_ON, "config-test")
    edge(NODE_TOOL, NODE_UNIT_TEST, DependencyEdgeType.CONFIG_DEPENDS_ON, "tool-test")
    edge(NODE_TOOL, NODE_UNIT_STATIC, DependencyEdgeType.CONFIG_DEPENDS_ON, "tool-static")
    edge(NODE_LOCK, NODE_UNIT_STATIC, DependencyEdgeType.CONFIG_DEPENDS_ON, "lock-static")
    edge(NODE_LOCK, NODE_UNIT_TEST, DependencyEdgeType.CONFIG_DEPENDS_ON, "lock-test")
    edge(NODE_SELECTOR, NODE_UNIT_TEST, DependencyEdgeType.CONFIG_DEPENDS_ON, "selector-test")
    edge(NODE_POLICY, NODE_UNIT_FORMAL, DependencyEdgeType.CONFIG_DEPENDS_ON, "policy-formal")
    edge(NODE_NETWORK, NODE_UNIT_TEST, DependencyEdgeType.CONFIG_DEPENDS_ON, "net-test")
    edge(NODE_SCHEMA, NODE_UNIT_STATIC, DependencyEdgeType.SCHEMA_DEPENDS_ON, "schema-static")
    edge(NODE_SCHEMA, NODE_UNIT_FORMAL, DependencyEdgeType.SCHEMA_DEPENDS_ON, "schema-formal")
    edge(NODE_CANON, NODE_UNIT_FORMAL, DependencyEdgeType.CONFIG_DEPENDS_ON, "canon-formal")
    edge(NODE_CIRCUIT, NODE_UNIT_FORMAL, DependencyEdgeType.CONFIG_DEPENDS_ON, "circuit-formal")
    edge(NODE_VK, NODE_UNIT_FORMAL, DependencyEdgeType.CONFIG_DEPENDS_ON, "vk-formal")
    edge(NODE_PK, NODE_UNIT_FORMAL, DependencyEdgeType.CONFIG_DEPENDS_ON, "pk-formal")
    edge(NODE_CHECKED, NODE_UNIT_FORMAL, DependencyEdgeType.PROOF_DEPENDS_ON, "checked-formal")
    edge(NODE_UNIT_FORMAL, NODE_AGG_MAIN, DependencyEdgeType.AGGREGATE_CONTAINS, "formal-agg")
    edge(NODE_UNIT_TEST, NODE_AGG_MAIN, DependencyEdgeType.AGGREGATE_CONTAINS, "test-agg")
    edge(NODE_UNIT_STATIC, NODE_AGG_MAIN, DependencyEdgeType.AGGREGATE_CONTAINS, "static-agg")

    # Independent module B (unrelated reuse target).
    edge(NODE_ARTIFACT_B, NODE_UNIT_B, DependencyEdgeType.SOURCE_DEPENDS_ON, "b-static")
    edge(NODE_UNIT_B, NODE_AGG_B, DependencyEdgeType.AGGREGATE_CONTAINS, "b-agg")
    return graph


def base_inventory(*, version: str = "v1") -> tuple[Any, ...]:
    return tuple(
        sample_artifact(path, label=_content_label(path, version))
        for path in BASE_PATHS
    )


def mutate_inventory(
    inventory: Sequence[Any],
    *,
    path: str,
    version: str,
) -> tuple[Any, ...]:
    """Return a new inventory with one path content version bumped."""

    updated: list[Any] = []
    found = False
    for item in inventory:
        if item.path == path:
            updated.append(sample_artifact(path, label=_content_label(path, version)))
            found = True
        else:
            updated.append(item)
    if not found:
        raise ValueError(f"path {path!r} is not in the base inventory")
    return tuple(updated)


def restore_path_version(
    inventory: Sequence[Any],
    *,
    path: str,
    version: str,
) -> tuple[Any, ...]:
    return mutate_inventory(inventory, path=path, version=version)


# ---------------------------------------------------------------------------
# Scenario recipes (compact; expectations are derived, not hand-authored dumps)
# ---------------------------------------------------------------------------


def _recipe(
    scenario: str,
    *,
    path: str | None,
    family: str,
    description: str,
    invalidation_policy: Mapping[str, Any] | None = None,
    is_merge: bool = False,
    merge_resolved: bool = True,
    extra_parents: Sequence[str] = (),
    proof_mode: str = ProofMode.INTEGRITY_ONLY.value,
    production_success_allowed: bool = True,
    aggregate_effect: str = "invalidate_affected_aggregates",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "family": family,
        "description": description,
        "changed_path": path,
        "invalidation_policy": dict(invalidation_policy or {}),
        "is_merge": is_merge,
        "merge_resolved": merge_resolved,
        "extra_parents": list(extra_parents),
        "proof_mode": proof_mode,
        "production_success_allowed": production_success_allowed,
        "aggregate_effect": aggregate_effect,
        "notes": notes,
    }


SCENARIO_RECIPES: Final[tuple[dict[str, Any], ...]] = (
    _recipe(
        "genesis",
        path=None,
        family="lifecycle",
        description="Initial repository full checkpoint (no parent seal).",
        invalidation_policy={"is_genesis": True},
        aggregate_effect="build_full_forest",
        notes="Genesis forces full fallback; every unit is newly proved.",
    ),
    _recipe(
        "source",
        path=PATH_SOURCE_A,
        family="source",
        description="Localized private source implementation edit on module A.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "source_interface",
        path=PATH_INTERFACE,
        family="source",
        description="Public interface edit invalidates interface dependents.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "unrelated_source",
        path=PATH_SOURCE_B,
        family="source",
        description="Independent module B edit must not invalidate module A units.",
        aggregate_effect="update_mod_b_aggregate_only",
    ),
    _recipe(
        "test",
        path=PATH_TEST,
        family="test",
        description="Unit-test source edit invalidates the test unit and aggregates.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "fixture",
        path=PATH_FIXTURE,
        family="fixture",
        description="Fixture edit invalidates bound test units.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "selector",
        path=PATH_SELECTOR,
        family="selector",
        description="Test-selector change recomputes required manifest membership.",
        aggregate_effect="recompute_manifest_and_aggregates",
    ),
    _recipe(
        "configuration",
        path=PATH_CONFIG,
        family="configuration",
        description="Relevant configuration edit invalidates bound execution units.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "network_policy",
        path=PATH_NETWORK_POLICY,
        family="policy",
        description="Network-policy change invalidates network-bound units.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "policy",
        path=PATH_POLICY,
        family="policy",
        description="Verification-policy change recomputes required units.",
        aggregate_effect="recompute_manifest_and_aggregates",
    ),
    _recipe(
        "lock",
        path=PATH_LOCK,
        family="lock",
        description="Dependency-lock upgrade under policy forces full checkpoint.",
        invalidation_policy={"treat_dependency_lock_as_full_fallback": True},
        aggregate_effect="full_forest_rebuild",
    ),
    _recipe(
        "tool",
        path=PATH_TOOL,
        family="tool",
        description="Tool/prover version identity change invalidates bound units.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "schema",
        path=PATH_PROOF_SCHEMA,
        family="schema",
        description="Proof-schema change requires full checkpoint.",
        invalidation_policy={"proof_schema_changed": True},
        aggregate_effect="full_forest_rebuild",
    ),
    _recipe(
        "canonicalization",
        path=PATH_CANONICALIZATION,
        family="schema",
        description="Canonicalization version change requires full checkpoint.",
        invalidation_policy={"canonicalization_changed": True},
        aggregate_effect="full_forest_rebuild",
    ),
    _recipe(
        "checked_spec",
        path=PATH_CHECKED_SPEC,
        family="documentation",
        description="Checked-specification document edit invalidates bound formal units.",
        aggregate_effect="update_main_aggregate_only",
    ),
    _recipe(
        "documentation",
        path=PATH_DOCS,
        family="documentation",
        description="Ordinary documentation edit preserves execution proofs.",
        aggregate_effect="none_near_total_reuse",
    ),
    _recipe(
        "circuit",
        path=PATH_CIRCUIT,
        family="circuit_key",
        description="Circuit version change forces full checkpoint.",
        aggregate_effect="full_forest_rebuild",
    ),
    _recipe(
        "key",
        path=PATH_VERIFICATION_KEY,
        family="circuit_key",
        description="Verification-key change forces full checkpoint.",
        aggregate_effect="full_forest_rebuild",
    ),
    _recipe(
        "branch",
        path=PATH_SOURCE_A,
        family="lifecycle",
        description="Parent-bound branch-A edit from genesis.",
        aggregate_effect="update_main_aggregate_only",
        notes="Branch identity is the revision id; parent remains genesis.",
    ),
    _recipe(
        "merge",
        path=PATH_SOURCE_B,
        family="lifecycle",
        description="Merge of branch A and branch B with both parents bound.",
        is_merge=True,
        merge_resolved=True,
        aggregate_effect="bounded_incremental_or_honest_fallback",
        notes="Parents are both branch tips; selected parent is the earlier tip.",
    ),
    _recipe(
        "rollback",
        path=PATH_SOURCE_A,
        family="lifecycle",
        description=(
            "Rollback of source bytes to the genesis content under a new "
            "parent-bound transition (never a seal replay)."
        ),
        aggregate_effect="update_main_aggregate_only",
        notes="Content may match an earlier root; the transition is new.",
    ),
    _recipe(
        "simulated_evidence",
        path=PATH_SOURCE_A,
        family="adversarial_label",
        description=(
            "Simulated proving labeled for rejection/non-production plumbing only."
        ),
        proof_mode=ProofMode.SIMULATED.value,
        production_success_allowed=False,
        aggregate_effect="reject_as_simulated_only",
        notes=(
            "proof_mode=simulated forces seal outcome simulated_only under "
            "production policy; never sealed_full or sealed_incremental."
        ),
    ),
)


def _aggregate_effect_for(
    recipe: Mapping[str, Any],
    *,
    affected_aggregates: Sequence[str],
    full_fallback_required: bool,
    docs_only: bool,
) -> dict[str, Any]:
    if recipe["proof_mode"] == ProofMode.SIMULATED.value:
        effect = "reject_as_simulated_only"
        seal_outcome = SealStatus.SIMULATED_ONLY.value
    elif full_fallback_required or recipe.get("invalidation_policy", {}).get(
        "is_genesis"
    ):
        effect = "full_forest_rebuild"
        seal_outcome = SealStatus.SEALED_FULL.value
    elif docs_only:
        effect = "none_near_total_reuse"
        seal_outcome = SealStatus.SEALED_INCREMENTAL.value
    else:
        effect = str(recipe["aggregate_effect"])
        seal_outcome = SealStatus.SEALED_INCREMENTAL.value
    return {
        "effect": effect,
        "affected_aggregate_ids": list(affected_aggregates),
        "expected_seal_outcome": seal_outcome,
        "production_success_allowed": bool(recipe["production_success_allowed"]),
    }


def _parents_for(
    scenario: str,
    *,
    genesis_revision: str,
    branch_revision: str | None,
    source_revision: str | None,
    branch_b_revision: str | None,
) -> list[str]:
    if scenario == "genesis":
        return []
    if scenario == "merge":
        parents = [p for p in (branch_revision, branch_b_revision) if p]
        return sorted(parents)
    if scenario == "rollback":
        # Parent is the post-source tip so rollback is a new forward transition.
        return [source_revision or genesis_revision]
    if scenario == "branch":
        return [genesis_revision]
    return [genesis_revision]


def materialize_history() -> dict[str, Any]:
    """Generate the complete deterministic fixture history and graph corpus."""

    graph = build_proof_graph()
    policy = classification_policy()
    base = base_inventory(version="v1")
    genesis_state = sample_repository_state(
        repository_id=REPOSITORY_ID,
        revision=GENESIS_REVISION,
        tree_label=f"{CORPUS_ID}:tree:genesis",
        parent_revision_ids=(),
    )

    scenarios: list[dict[str, Any]] = []
    # Inventory snapshots keyed by revision for rollback/merge composition.
    inventories: dict[str, tuple[Any, ...]] = {GENESIS_REVISION: base}
    states: dict[str, Any] = {GENESIS_REVISION: genesis_state}

    # Pre-assign deterministic revisions so merge parents sort stably.
    scenario_revisions: dict[str, str] = {}
    for index, recipe in enumerate(SCENARIO_RECIPES, start=1):
        scenario_revisions[str(recipe["scenario"])] = _revision_for(
            str(recipe["scenario"]), index=index
        )

    for index, recipe in enumerate(SCENARIO_RECIPES, start=1):
        scenario = str(recipe["scenario"])
        revision = scenario_revisions[scenario]
        parents = _parents_for(
            scenario,
            genesis_revision=GENESIS_REVISION,
            branch_revision=scenario_revisions.get("branch"),
            source_revision=scenario_revisions.get("source"),
            branch_b_revision=scenario_revisions.get("unrelated_source"),
        )

        # Parent inventory: merge starts from branch inventory; others from
        # the first (or only) parent tip, defaulting to genesis.
        if scenario == "genesis":
            parent_revision = ABSENCE_TOKEN
            parent_inventory = base
            new_inventory = base
            parent_state = None
            new_state = genesis_state
            repository_diff = None
            inv_policy = sample_invalidation_policy(
                **dict(recipe.get("invalidation_policy") or {})
            )
            closure = compute_invalidation_closure(
                graph,
                known_unit_ids=KNOWN_UNITS,
                policy=inv_policy,
            )
            changed_artifacts_payload: list[dict[str, Any]] = []
            changed_commitment = ABSENCE_TOKEN
            change_classes: list[str] = []
        else:
            if scenario == "merge":
                # Compose inventory: start from genesis, apply both branch edits.
                parent_revision = parents[0]
                composed = base
                composed = mutate_inventory(
                    composed, path=PATH_SOURCE_A, version="branch-a"
                )
                composed = mutate_inventory(
                    composed, path=PATH_SOURCE_B, version="branch-b"
                )
                parent_inventory = inventories[GENESIS_REVISION]
                new_inventory = composed
                selected_parent = parents[0]
                parent_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=parent_revision,
                    tree_label=f"{CORPUS_ID}:tree:{parent_revision}",
                    parent_revision_ids=(GENESIS_REVISION,),
                )
                new_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=revision,
                    tree_label=f"{CORPUS_ID}:tree:{revision}",
                    parent_revision_ids=tuple(parents),
                )
                repository_diff = diff_repository_states(
                    parent_state,
                    new_state,
                    old_artifacts=parent_inventory,
                    new_artifacts=new_inventory,
                    policy=policy,
                    inventory_complete=True,
                    selected_parent_revision=selected_parent,
                    merge_resolved=bool(recipe["merge_resolved"]),
                )
            elif scenario == "rollback":
                parent_revision = parents[0]
                parent_inventory = inventories.get(
                    scenario_revisions.get("source", GENESIS_REVISION),
                    mutate_inventory(base, path=PATH_SOURCE_A, version="v2"),
                )
                # Restore module A bytes to genesis content under a new revision.
                new_inventory = restore_path_version(
                    parent_inventory, path=PATH_SOURCE_A, version="v1"
                )
                parent_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=parent_revision,
                    tree_label=f"{CORPUS_ID}:tree:{parent_revision}",
                    parent_revision_ids=(GENESIS_REVISION,),
                )
                new_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=revision,
                    tree_label=f"{CORPUS_ID}:tree:{revision}",
                    parent_revision_ids=tuple(parents),
                )
                repository_diff = diff_repository_states(
                    parent_state,
                    new_state,
                    old_artifacts=parent_inventory,
                    new_artifacts=new_inventory,
                    policy=policy,
                    inventory_complete=True,
                )
            elif scenario == "branch":
                parent_revision = GENESIS_REVISION
                parent_inventory = inventories[GENESIS_REVISION]
                new_inventory = mutate_inventory(
                    parent_inventory, path=PATH_SOURCE_A, version="branch-a"
                )
                parent_state = states[GENESIS_REVISION]
                new_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=revision,
                    tree_label=f"{CORPUS_ID}:tree:{revision}",
                    parent_revision_ids=(GENESIS_REVISION,),
                )
                repository_diff = diff_repository_states(
                    parent_state,
                    new_state,
                    old_artifacts=parent_inventory,
                    new_artifacts=new_inventory,
                    policy=policy,
                    inventory_complete=True,
                )
            else:
                parent_revision = GENESIS_REVISION
                parent_inventory = inventories[GENESIS_REVISION]
                changed_path = recipe["changed_path"]
                assert changed_path is not None
                version_tag = f"{scenario}-v2"
                new_inventory = mutate_inventory(
                    parent_inventory, path=str(changed_path), version=version_tag
                )
                parent_state = states[GENESIS_REVISION]
                new_state = sample_repository_state(
                    repository_id=REPOSITORY_ID,
                    revision=revision,
                    tree_label=f"{CORPUS_ID}:tree:{revision}",
                    parent_revision_ids=(GENESIS_REVISION,),
                )
                repository_diff = diff_repository_states(
                    parent_state,
                    new_state,
                    old_artifacts=parent_inventory,
                    new_artifacts=new_inventory,
                    policy=policy,
                    inventory_complete=True,
                )

            inv_policy = sample_invalidation_policy(
                **dict(recipe.get("invalidation_policy") or {})
            )
            # Lock recipe also sets classification treat_dependency_lock flag
            # via the shared PathClassificationPolicy above.
            closure = compute_invalidation_closure(
                graph,
                repository_diff=repository_diff,
                path_to_node_ids=PATH_TO_NODE_IDS,
                known_unit_ids=KNOWN_UNITS,
                policy=inv_policy,
            )
            changed_artifacts_payload = [
                item.to_canonical() for item in repository_diff.changed_artifacts
            ]
            changed_commitment = repository_diff.changed_artifact_commitment
            change_classes = list(repository_diff.change_classes_present)

        inventories[revision] = new_inventory
        states[revision] = new_state

        # Direct unit closure: seed units, or one-hop unit dependents of seeds.
        # Transitive = remaining invalidated units.
        seed_units = sorted(set(closure.seed_node_ids) & set(KNOWN_UNITS))
        if seed_units:
            direct_invalidated = seed_units
        else:
            immediate: set[str] = set()
            for seed in closure.seed_node_ids:
                if not graph.has_node(seed):
                    continue
                for edge in graph.edges():
                    if edge.from_id == seed and edge.to_id in KNOWN_UNITS:
                        if edge.to_id in closure.invalidated_unit_ids:
                            immediate.add(edge.to_id)
            direct_invalidated = sorted(immediate)

        # Under full fallback every known unit is invalidated; treat the whole
        # set as direct so the partition remains complete and non-empty.
        if closure.full_fallback.required and not direct_invalidated:
            direct_invalidated = list(closure.invalidated_unit_ids)

        transitive_invalidated = sorted(
            set(closure.invalidated_unit_ids) - set(direct_invalidated)
        )

        fallback = closure.full_fallback
        aggregate = _aggregate_effect_for(
            recipe,
            affected_aggregates=closure.affected_aggregate_ids,
            full_fallback_required=fallback.required,
            docs_only=closure.docs_only,
        )

        # Simulated evidence never claims production seal success.
        if recipe["proof_mode"] == ProofMode.SIMULATED.value:
            assert aggregate["production_success_allowed"] is False
            assert aggregate["expected_seal_outcome"] == SealStatus.SIMULATED_ONLY.value

        scenario_record = {
            "id": f"{index:02d}-{scenario}",
            "scenario": scenario,
            "family": recipe["family"],
            "description": recipe["description"],
            "notes": recipe["notes"],
            "revision": revision,
            "parents": {
                "parent_revision_ids": list(parents),
                "selected_parent_revision": (
                    parents[0] if parents else ABSENCE_TOKEN
                ),
                "parent_seal_binding": (
                    "genesis"
                    if scenario == "genesis"
                    else "explicit_parent_revision"
                ),
                "is_merge": bool(recipe["is_merge"]),
                "merge_resolved": bool(recipe["merge_resolved"]),
            },
            "changed_artifact_provenance": {
                "changed_path": recipe["changed_path"],
                "change_classes": change_classes,
                "changed_artifact_commitment": changed_commitment,
                "changed_artifacts": changed_artifacts_payload,
                "diff_complete": (
                    True if repository_diff is None else repository_diff.complete
                ),
                "inventory_complete": True,
            },
            "expected_unit_closure": {
                "direct_invalidated_unit_ids": direct_invalidated,
                "transitive_invalidated_unit_ids": transitive_invalidated,
                "invalidated_unit_ids": list(closure.invalidated_unit_ids),
                "preserved_unit_ids": list(closure.preserved_unit_ids),
                "added_unit_ids": list(closure.added_unit_ids),
                "removed_unit_ids": list(closure.removed_unit_ids),
                "seed_node_ids": list(closure.seed_node_ids),
                "closure_node_ids": list(closure.closure_node_ids),
                "docs_only": closure.docs_only,
                "complete": closure.complete,
            },
            "aggregate_effect": aggregate,
            "full_fallback_decision": fallback.to_canonical(),
            "proof_mode": recipe["proof_mode"],
            "production_success_allowed": bool(
                recipe["production_success_allowed"]
            ),
            "authority": {
                "expectation_sources": [
                    "reviewed_spec",
                    "datasets_invalidation_engine",
                    "datasets_repository_diff",
                ],
                "implementation_observation_authoritative": False,
                "simulated_authoritative_for_production": False,
                "wall_clock_forbidden": True,
                "host_environment_forbidden": True,
            },
        }
        scenarios.append(scenario_record)

    graph_canonical = graph.to_canonical()
    graph_cid = graph.graph_cid()

    history = {
        "repository_id": REPOSITORY_ID,
        "genesis_revision": GENESIS_REVISION,
        "scenario_order": [item["scenario"] for item in scenarios],
        "revision_index": {
            item["scenario"]: item["revision"] for item in scenarios
        },
        "base_paths": list(BASE_PATHS),
        "known_unit_ids": list(KNOWN_UNITS),
        "path_to_node_ids": {
            path: list(nodes) for path, nodes in sorted(PATH_TO_NODE_IDS.items())
        },
    }

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "evidence_subset": EVIDENCE_SUBSET,
        "description": DESCRIPTION,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "auxiliary_scenarios": list(AUXILIARY_SCENARIOS),
        "classification_policy_cid": policy.policy_cid(),
        "proof_graph": {
            "graph_cid": graph_cid,
            "dependency_graph_schema_version": graph_canonical[
                "dependency_graph_schema_version"
            ],
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
            "nodes": graph_canonical["nodes"],
            "edges": graph_canonical["edges"],
        },
        "history": history,
        "scenarios": scenarios,
        "simulated_production_guard": {
            "rule": (
                "No fixture may model proof_mode=simulated as production "
                "sealed_full or sealed_incremental success."
            ),
            "simulated_scenario_ids": [
                item["id"]
                for item in scenarios
                if item["proof_mode"] == ProofMode.SIMULATED.value
            ],
            "forbidden_seal_outcomes_for_simulated": [
                SealStatus.SEALED_FULL.value,
                SealStatus.SEALED_INCREMENTAL.value,
            ],
            "required_seal_outcome_for_simulated": SealStatus.SIMULATED_ONLY.value,
        },
    }
    manifest["corpus_content_id"] = "sha256:" + _sha256_hex(manifest)
    return manifest


def _expected_full_fallback_required(recipe: Mapping[str, Any]) -> bool:
    """Recipe-level full-fallback expectation (closed policy axes only)."""

    policy = dict(recipe.get("invalidation_policy") or {})
    if policy.get("is_genesis"):
        return True
    if policy.get("proof_schema_changed"):
        return True
    if policy.get("canonicalization_changed"):
        return True
    if policy.get("treat_dependency_lock_as_full_fallback") and recipe.get(
        "changed_path"
    ) == PATH_LOCK:
        return True
    path = recipe.get("changed_path")
    if path in {
        PATH_CIRCUIT,
        PATH_PROVING_KEY,
        PATH_VERIFICATION_KEY,
        PATH_CANONICALIZATION,
        PATH_PROOF_SCHEMA,
        PATH_ENVIRONMENT,
    }:
        return True
    if path == PATH_LOCK:
        # Classification policy treats dependency lock as full fallback.
        return True
    return False


def _parent_model(scenario: str) -> dict[str, Any]:
    if scenario == "genesis":
        return {
            "kind": "none",
            "parent_count": 0,
            "is_merge": False,
            "selected_parent": "none",
        }
    if scenario == "merge":
        return {
            "kind": "multi_parent_merge",
            "parent_count": 2,
            "is_merge": True,
            "selected_parent": "earliest_sorted_branch_tip",
        }
    if scenario == "rollback":
        return {
            "kind": "source_tip",
            "parent_count": 1,
            "is_merge": False,
            "selected_parent": "source_revision",
        }
    if scenario == "branch":
        return {
            "kind": "genesis",
            "parent_count": 1,
            "is_merge": False,
            "selected_parent": "genesis",
        }
    return {
        "kind": "genesis",
        "parent_count": 1,
        "is_merge": False,
        "selected_parent": "genesis",
    }


def build_compact_manifest() -> dict[str, Any]:
    """Compact checked-in catalog (recipes only; no bulk envelopes).

    Full parent/provenance/closure/fallback records are produced by
    :func:`materialize_history`.  This catalog is the durable fixture index and
    still lists explicit parent, provenance, closure, aggregate, and
    full-fallback fields for every scenario.
    """

    cases: list[dict[str, Any]] = []
    for index, recipe in enumerate(SCENARIO_RECIPES, start=1):
        scenario = str(recipe["scenario"])
        full_fallback = _expected_full_fallback_required(recipe)
        cases.append(
            {
                "id": f"{index:02d}-{scenario}",
                "scenario": scenario,
                "family": recipe["family"],
                "description": recipe["description"],
                "notes": recipe["notes"],
                "revision": _revision_for(scenario, index=index),
                "parents": _parent_model(scenario),
                "changed_artifact_provenance": {
                    "changed_path": recipe["changed_path"],
                    "provenance_required": scenario != "genesis",
                },
                "expected_unit_closure": {
                    "direct_field": "direct_invalidated_unit_ids",
                    "transitive_field": "transitive_invalidated_unit_ids",
                    "derived_by": "compute_invalidation_closure",
                },
                "aggregate_effect": {
                    "declared_effect": recipe["aggregate_effect"],
                    "production_success_allowed": bool(
                        recipe["production_success_allowed"]
                    ),
                },
                "full_fallback_decision": {
                    "required": full_fallback,
                    "policy_overrides": dict(
                        recipe.get("invalidation_policy") or {}
                    ),
                },
                "proof_mode": recipe["proof_mode"],
                "production_success_allowed": bool(
                    recipe["production_success_allowed"]
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "evidence_subset": EVIDENCE_SUBSET,
        "description": DESCRIPTION,
        "manifest_kind": "compact_recipe_catalog",
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "auxiliary_scenarios": list(AUXILIARY_SCENARIOS),
        "repository_id": REPOSITORY_ID,
        "genesis_revision": GENESIS_REVISION,
        "base_paths": list(BASE_PATHS),
        "known_unit_ids": list(KNOWN_UNITS),
        "path_to_node_ids": {
            path: list(nodes) for path, nodes in sorted(PATH_TO_NODE_IDS.items())
        },
        "cases": cases,
        "simulated_production_guard": {
            "rule": (
                "No fixture may model proof_mode=simulated as production "
                "sealed_full or sealed_incremental success."
            ),
            "simulated_scenario_ids": [
                case["id"]
                for case in cases
                if case["proof_mode"] == ProofMode.SIMULATED.value
            ],
            "forbidden_seal_outcomes_for_simulated": [
                SealStatus.SEALED_FULL.value,
                SealStatus.SEALED_INCREMENTAL.value,
            ],
            "required_seal_outcome_for_simulated": SealStatus.SIMULATED_ONLY.value,
        },
        "materialization": {
            "entry_point": "materialize_history",
            "byte_identical_generations_required": True,
            "full_scenario_fields": [
                "parents",
                "changed_artifact_provenance",
                "expected_unit_closure",
                "aggregate_effect",
                "full_fallback_decision",
            ],
        },
    }
    # Stable catalog identity from ordered scenario ids (no bulk envelope hash).
    catalog_fingerprint = "|".join(
        [
            CORPUS_ID,
            SCHEMA,
            *(case["id"] for case in cases),
        ]
    )
    payload["catalog_fingerprint"] = catalog_fingerprint
    payload["corpus_content_id"] = (
        f"catalog:{CORPUS_ID}:cases={len(cases)}"
    )
    return payload


def canonical_manifest_bytes(manifest: Mapping[str, Any] | None = None) -> bytes:
    """Serialize a full materialization to canonical UTF-8 JSON."""

    payload = materialize_history() if manifest is None else dict(manifest)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_compact_manifest_bytes(
    manifest: Mapping[str, Any] | None = None,
) -> bytes:
    payload = build_compact_manifest() if manifest is None else dict(manifest)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def generate_twice_byte_identical() -> tuple[bytes, bytes]:
    """Return two independent clean full generations for equality checks."""

    first = canonical_manifest_bytes()
    second = canonical_manifest_bytes()
    return first, second


def _compact_manifest_text(manifest: Mapping[str, Any] | None = None) -> str:
    """Pretty-print the compact catalog with stable insertion-order keys."""

    payload = build_compact_manifest() if manifest is None else dict(manifest)
    # Insertion order from :func:`build_compact_manifest` is the durable layout.
    # Avoid sort_keys so checked-in bytes stay reviewable and rewrite-stable.
    text = json.dumps(
        payload,
        sort_keys=False,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_manifest(path: Path | None = None) -> Path:
    """Write the compact checked-in ``fixture_manifest.json`` catalog."""

    target = path or (Path(__file__).resolve().parent / "fixture_manifest.json")
    text = _compact_manifest_text()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Only rewrite when content changes so validation imports stay side-effect free
    # when the checked-in catalog is already synchronized.
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    # Drop accidental undeclared scratch files from earlier generator runs.
    for scratch_name in ("_run_generate.py",):
        scratch = target.parent / scratch_name
        if scratch.is_file():
            try:
                scratch.unlink()
            except OSError:
                pass
    return target


def load_checked_in_manifest(path: Path | None = None) -> dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture_manifest.json")
    return json.loads(target.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args in ([], ["write"], ["--write"], ["materialize"], ["--materialize"]):
        out = write_manifest()
        print(f"wrote {out}")
        return 0
    if args == ["--check-identical"]:
        a, b = generate_twice_byte_identical()
        if a != b:
            print("generations differ", file=sys.stderr)
            return 1
        print(f"byte-identical generations ({len(a)} bytes)")
        return 0
    print(
        "usage: generate_fixture_history.py "
        "[--write|--materialize|--check-identical]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
