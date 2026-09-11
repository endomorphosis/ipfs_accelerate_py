#!/usr/bin/env python3
"""Sealed bootstrap specification for the SPAR configured-board program.

This module contains operator-owned declarative data and deterministic renderers
only.  It does not schedule work, mutate source, create an alternate task store,
or decide completion.  The existing configured-board scheduler and
DatabaseTaskSource remain the execution and task authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


ROOT: Final = Path(__file__).resolve().parents[3]
PROGRAM: Final = "semantic-preserving-autonomous-remodularization-v1"
PLAN_REVISION: Final = "SPAR-PLAN-R1"
BASE_REVISION: Final = "e3c9831d4465d0e9f1aba336994a385541611895"
BASE_TREE: Final = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
BRANCH: Final = "codex/semantic-preserving-autonomous-remodularization-v1"
OBSERVED_AT: Final = "2026-08-28T16:53:33Z"


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    title: str
    parent: str
    depends_on: tuple[str, ...]
    track: str
    goal: str
    completion: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    goal_id: str
    owner: str
    track: str
    dependencies: tuple[str, ...]
    objective: str
    paths: tuple[str, ...]
    symbols: str
    validation: str
    effect_class: str = "bounded_internal_implementation"
    public_api: str = "internal or compatibility-preserving only"
    state_impact: str = "No new authority; immutable evidence or explicitly fenced state only."
    autonomy: str = "Tier B"


GOALS: Final = (
    GoalSpec("SPAR-G000", "Build, self-host, and qualify semantic-preserving autonomous remodularization", "", (), "spar-root", "Deliver the accepted cross-repository compiler, supervisor consumer, capstone, and release evidence without weakening semantic, proof, storage, or task authority.", "Every mandatory child goal is accepted against one verified final root; all safety floors, seals, receipts, capstone checks, and fixed-point conditions pass."),
    GoalSpec("SPAR-G010", "Freeze authorities, identities, risks, and baseline evidence", "SPAR-G000", (), "baseline", "Freeze exact source and reuse decisions before implementation.", "SPAR-G011 through SPAR-G013 are accepted with no unclassified authority or preservation gap."),
    GoalSpec("SPAR-G011", "Reconcile predecessor systems and source authorities", "SPAR-G010", (), "baseline", "Bind current repositories, predecessor evidence, dirty-work preservation, and canonical owners.", "The repository forest, predecessor dispositions, capability facts, and preservation fingerprints verify."),
    GoalSpec("SPAR-G012", "Define symbol/module identity and projection contracts", "SPAR-G010", ("SPAR-G011",), "identity", "Define distinct exact semantic identities and advisory projections.", "Closed records, canonical encodings, strict rejection, and identity-separation vectors pass."),
    GoalSpec("SPAR-G013", "Define semantic, compatibility, and acceptance authority", "SPAR-G010", ("SPAR-G011",), "authority", "Separate facts, observations, proofs, policy, compatibility, and accepted transitions.", "No vector, model, task status, test, proof candidate, or worker can self-authorize a transition."),
    GoalSpec("SPAR-G020", "Build the function/module semantic and graph model", "SPAR-G000", ("SPAR-G010",), "semantic-model", "Extend current semantic-state and graph authorities with refactoring views.", "All capsule, graph, state, initialization, compatibility, frontier, and retrieval contracts verify."),
    GoalSpec("SPAR-G021", "Build function/class/block/module capsules", "SPAR-G020", (), "capsules", "Represent functions, methods, classes, callsites, top-level blocks, modules, and packages.", "Every eligible artifact has exact identities, provenance, freshness, and typed uncertainty."),
    GoalSpec("SPAR-G022", "Build program, state, initialization, and compatibility graphs", "SPAR-G020", ("SPAR-G021",), "graphs", "Project typed static, may, runtime, state, initialization, and compatibility facts.", "Cycles, dynamic frontiers, state owners, ordering, resources, consumers, tests, and proofs remain explicit."),
    GoalSpec("SPAR-G023", "Build multi-view projections and exact retrieval", "SPAR-G020", ("SPAR-G021",), "retrieval", "Persist model-pinned advisory projections resolved to exact current capsules.", "Corruption, staleness, capability absence, model mismatch, and non-authority behavior fail closed."),
    GoalSpec("SPAR-G030", "Plan coherent partitions and module boundaries", "SPAR-G000", ("SPAR-G020",), "partitioning", "Generate and compare constrained partitions and complete boundary migrations.", "Admitted candidates satisfy every hard constraint and retain reproducible score/constraint receipts."),
    GoalSpec("SPAR-G031", "Condense cycles and infer state ownership", "SPAR-G030", (), "partition-foundations", "Build the exact SCC condensation DAG and state-owner constraints.", "SCC identity, dynamic conservative edges, oversized-cycle gaps, and state uniqueness verify."),
    GoalSpec("SPAR-G032", "Generate and rank constrained partition candidates", "SPAR-G030", ("SPAR-G031",), "partition-candidates", "Compare multiple deterministic and advisory candidate generators.", "Soft scores never override a hard constraint; rejected comparisons remain negative evidence."),
    GoalSpec("SPAR-G033", "Synthesize boundary contracts, APIs, and migration plans", "SPAR-G030", ("SPAR-G032",), "boundaries", "Produce assume/guarantee contracts, target APIs, façades, and migrations.", "Every cut edge, consumer, state/resource owner, initialization obligation, and compatibility promise has a disposition."),
    GoalSpec("SPAR-G040", "Execute bounded transformations transactionally", "SPAR-G000", ("SPAR-G030",), "transformation", "Compile deterministic packets and apply bounded extraction waves in fenced worktrees.", "Exact preimages, allowed effects, rollback, source maps, ordering, state, and compatibility are preserved."),
    GoalSpec("SPAR-G041", "Compile deterministic transformation packets", "SPAR-G040", (), "packets", "Bind exact moves, rewrites, adapters, expected deltas, validation, and rollback.", "No packet has unrestricted scope or unsupported syntax and every dry-run is deterministic."),
    GoalSpec("SPAR-G042", "Preserve imports, initialization, state, and public compatibility", "SPAR-G040", ("SPAR-G041",), "compatibility", "Apply supported CST transforms and explicit compatibility mechanisms.", "Imports, state ownership, ordering, registries, resources, binding, introspection, and serialization verify or migrate explicitly."),
    GoalSpec("SPAR-G043", "Apply extraction waves with rollback and fixed-point control", "SPAR-G040", ("SPAR-G042",), "waves", "Execute bounded packets through current lease, fence, worktree, VFS, and merge mechanisms.", "Accepted waves are transactional and failed waves restore exact preimages while retaining negative evidence."),
    GoalSpec("SPAR-G050", "Prove and validate semantic preservation", "SPAR-G000", ("SPAR-G040",), "validation", "Independently validate each concrete old/new transformation under a declared observation profile.", "Evidence classes remain separate and the required conjunction passes without overstating coverage."),
    GoalSpec("SPAR-G051", "Select tests/proofs and validate translation", "SPAR-G050", (), "translation-validation", "Reuse current selection and validate syntax, imports, contracts, proofs, and tests.", "Selections bind current roots and uncertainty triggers the declared full fallback."),
    GoalSpec("SPAR-G052", "Compare traces, effects, mutation, and compatibility", "SPAR-G050", ("SPAR-G051",), "runtime-validation", "Compare paired behavior, effects, state, imports, compatibility, mutants, and adversarial cases.", "Required trace dimensions agree, critical mutants die, and unsupported dimensions remain blockers."),
    GoalSpec("SPAR-G053", "Integrate Hammer, e-graphs, interpolation, and CEGIS", "SPAR-G050", ("SPAR-G051",), "proof-synthesis", "Lower finite obligations into current proof and bounded synthesis authorities.", "Proofs reconstruct, countermodels replay, unknown remains unknown, and synthesized candidates re-enter validation."),
    GoalSpec("SPAR-G060", "Increase autonomy and reduce model dependence", "SPAR-G000", ("SPAR-G050",), "reuse", "Reuse exact states and compile repeated accepted waves into proof-carrying procedures.", "Exact/procedural routes precede models and context receipts justify every included or omitted item."),
    GoalSpec("SPAR-G061", "Build exact/procedural/refactor memory", "SPAR-G060", (), "memory", "Store exact accepted and rejected transition evidence with freshness-bound reuse keys.", "Similarity yields context only and stale or mismatched evidence revokes reuse."),
    GoalSpec("SPAR-G062", "Compile repeated refactorings into procedures", "SPAR-G060", ("SPAR-G061",), "procedures", "Qualify anti-unified extraction procedures through the current procedure authority.", "No procedure self-certifies and at least one held-out/adversarial qualification route exists."),
    GoalSpec("SPAR-G063", "Minimize context and route only residuals to models", "SPAR-G060", ("SPAR-G061",), "context", "Call a general model only for one named unresolved typed residual.", "Repeated identical failures do not retry without evidence and model output cannot weaken validation."),
    GoalSpec("SPAR-G070", "Self-host supervisor planning and repository integration", "SPAR-G000", ("SPAR-G060",), "self-hosting", "Make the current supervisor a first-class consumer and fixed-point planner.", "Goals, tasks, repairs, roots, VFS transitions, recovery, and rollout receipts use existing authorities."),
    GoalSpec("SPAR-G071", "Create goals, subgoals, tasks, repair work, and fixed-point refill", "SPAR-G070", (), "goal-refinement", "Compile exact monolith findings into bounded dependent work and evidence-driven repair.", "No generic fix prompt or repeated identical retry is emitted and every task has bounded scope."),
    GoalSpec("SPAR-G072", "Integrate VFS, world roots, cross-repository changes, and recovery", "SPAR-G070", ("SPAR-G071",), "root-integration", "Publish accepted transitions through existing durable VFS/outbox/CAS/recovery and gitlink ownership.", "Crashes, stale writers, root conflicts, and cross-repository integration recover without overwrite."),
    GoalSpec("SPAR-G073", "Activate progressive dogfooding", "SPAR-G070", ("SPAR-G072",), "rollout", "Advance bootstrap, shadow_plan, shadow_apply, guarded, and required only through sealed gates.", "Each mode enforces its mutation/merge/receipt/autonomy contract and workers cannot change rollout."),
    GoalSpec("SPAR-G080", "Qualify scale, safety, benchmark evidence, and release", "SPAR-G000", ("SPAR-G070",), "qualification", "Freeze and run scale, adversarial, capstone, benchmark, and release qualification.", "Safety floors remain zero, denominators retain failures, and final evidence verifies transitively."),
    GoalSpec("SPAR-G081", "Run adversarial/security/chaos qualification", "SPAR-G080", (), "adversarial", "Exercise controlled scale fixtures, acceptance matrices, security, privacy, chaos, and recovery.", "No unauthorized, stale, simulated, similarity-only, self-authorized, or critical-defect result is promoted."),
    GoalSpec("SPAR-G082", "Run scale/capstone benchmarks and publish release evidence", "SPAR-G080", ("SPAR-G081",), "release", "Run the self-hosted capstone, benchmark ablations, and publish migration/release/limitation evidence.", "The capstone, later procedure reuse, all seals, required receipts, and final root verify."),
)


TASK_ROWS: Final = (
    (0, "Freeze authorities, architecture, board, seals, benchmark, and scheduler", "SPAR-G011", "ipfs_accelerate_py", "operator-bootstrap", (), "Inventory exact repositories, predecessors, authorities, dirty state, capabilities, dynamic risks, and baselines; seal the reviewed plan, board, scheduler, validators, materializer, and benchmark before delegation.", ("docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md", "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md", "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md", "docs/architecture/semantic_preserving_autonomous_remodularization_inventory", "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json", "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json", "scripts/validate_semantic_preserving_remodularization_dependencies.py", "scripts/validate_semantic_preserving_remodularization_board.py", "scripts/materialize_semantic_preserving_remodularization_program.py", "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py"), "SPAR operator control bundle and seal", "python3 scripts/validate_semantic_preserving_remodularization_dependencies.py --check-all && python3 scripts/validate_semantic_preserving_remodularization_board.py --check-all", "operator_control", "none", "Immutable control inputs and operator completion evidence only.", "operator"),
    (1, "Audit current semantic-state, repair, graph, proof, VFS, storage, codemod, and supervisor capabilities", "SPAR-G011", "ipfs_accelerate_py", "capability-audit", ("SPAR-000",), "Produce a current-tree overlap/gap matrix locating exact owners and classifying every required capability by verified evidence status.", ("docs/architecture/semantic_preserving_autonomous_remodularization_inventory/verified_capability_matrix.json", "test/api/semantic_refactoring/test_capability_matrix.py"), "VerifiedCapabilityMatrix and current-tree probes", "python3 -m pytest -q test/api/semantic_refactoring/test_capability_matrix.py"),
    (2, "Define function, method, class, callsite, block, module, and package capsule contracts", "SPAR-G021", "ipfs_datasets_py", "capsule-contracts", ("SPAR-001",), "Extend datasets semantic authority with closed deterministic versioned capsules, strict canonical encoding, immutable nested values, and provider-free exports.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/capsules.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_capsules.py"), "FunctionSemanticCapsule and related @1 capsule types", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_capsules.py"),
    (3, "Define implementation, binding, contract, effect, state, dependency, behavior, and validation identities", "SPAR-G012", "ipfs_datasets_py", "identity-contracts", ("SPAR-001",), "Separate location-independent implementation identity from binding and compatibility identity, preserving existing @1 readers and excluding observational metadata.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/identities.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_identities.py"), "SemanticArtifactIdentitySet@1 and golden move vectors", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_identities.py"),
    (4, "Define top-level initialization, registration, resource, and public compatibility contracts", "SPAR-G013", "ipfs_datasets_py", "compatibility-contracts", ("SPAR-001",), "Make top-level blocks, import effects, registries, decorators, resources, serialization, introspection, CLI/plugin, and patch-target obligations first-class and explicit when unsupported.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/compatibility.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_compatibility_contracts.py"), "InitializationBlock and PublicCompatibilityObligation contracts", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_compatibility_contracts.py"),
    (5, "Implement datasets multi-view function and module projection contracts", "SPAR-G012", "ipfs_datasets_py", "projection-contracts", ("SPAR-001",), "Add exact model-pinned advisory projection records for all declared semantic views plus deterministic structural fingerprints when neural capability is unavailable.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/projections.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_projections.py"), "SemanticProjection@1 and ProjectionUnavailable@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_projections.py"),
    (6, "Implement kit verified projection storage, index manifests, and exact resolver", "SPAR-G023", "ipfs_kit_py", "projection-storage", ("SPAR-002", "SPAR-003", "SPAR-004", "SPAR-005"), "Reuse kit block/vector authorities to verify projection bytes, generations, manifests, corruption, staleness, rebuilds, and exact current-capsule resolution.", ("ipfs_kit_py/ipfs_kit_py/semantic_refactoring/projection_store.py", "ipfs_kit_py/tests/test_semantic_refactoring_projection_store.py"), "VerifiedProjectionStore adapter and exact resolver", "python3 -m pytest -q ipfs_kit_py/tests/test_semantic_refactoring_projection_store.py"),
    (7, "Build the typed static program and refactoring graph", "SPAR-G021", "ipfs_datasets_py", "program-graph", ("SPAR-002", "SPAR-003", "SPAR-004", "SPAR-005"), "Extend the current program graph with typed semantic/refactoring nodes and edges bound to exact tree, analyzer, environment, and unresolved frontier.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/program_graph.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"), "SemanticRefactoringGraphView@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"),
    (8, "Implement dynamic Python frontier and runtime evidence adapters", "SPAR-G022", "ipfs_datasets_py", "dynamic-frontier", ("SPAR-002", "SPAR-003", "SPAR-004", "SPAR-005"), "Detect and classify reflection, dynamic imports/dispatch, monkeypatching, registration, framework dynamics, FFI, generated code, and bounded hermetic runtime evidence without hiding unknowns.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/dynamic_frontier.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_dynamic_frontier.py"), "DynamicPythonFrontier@1 and typed findings", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_dynamic_frontier.py"),
    (9, "Infer state ownership, aliasing, lifecycle, lock, and transaction structure", "SPAR-G022", "ipfs_datasets_py", "state-ownership", ("SPAR-007", "SPAR-008"), "Build exact/conservative read-write summaries, alias sets, owner candidates, lifecycle and synchronization relations, and reject duplicated mutable state.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/state_ownership.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_state_ownership.py"), "StateOwnershipGraph@1 and StateExtractionCandidate@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_state_ownership.py"),
    (10, "Build initialization-order and import-time effect graph", "SPAR-G022", "ipfs_datasets_py", "initialization", ("SPAR-007", "SPAR-008"), "Partition top-level execution into content-addressed blocks, infer ordering/cycles/effects, synthesize explicit initialization candidates, and bind observation profiles.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/initialization.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_initialization.py"), "InitializationOrderGraph@1 and InitializationStateMachine@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_initialization.py"),
    (11, "Build public API and compatibility inventory", "SPAR-G022", "ipfs_datasets_py", "public-compatibility", ("SPAR-002", "SPAR-003", "SPAR-004", "SPAR-005"), "Inventory every declared import/API/binding/serialization/introspection/CLI/plugin/registration/documentation/patch obligation and disposition each consumer.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/public_compatibility.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_public_compatibility.py"), "PublicCompatibilityInventory@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_public_compatibility.py"),
    (12, "Compute hard-dependency SCCs and extraction condensation DAG", "SPAR-G031", "ipfs_datasets_py", "scc-condensation", ("SPAR-007", "SPAR-008"), "Compute deterministic SCCs under a versioned hard-edge policy, preserve conservative edges, build the extraction DAG, and support incremental invalidation.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/scc.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_scc.py"), "SCCSnapshot@1 and ExtractionCondensationDAG@1", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_scc.py"),
    (13, "Implement deterministic candidate partition generators", "SPAR-G032", "ipfs_accelerate_py", "partition-generation", ("SPAR-009", "SPAR-010", "SPAR-011", "SPAR-012"), "Generate multiple deterministic candidates from SCC, state, contract, tests/proofs, graph, and co-change evidence; keep projection clustering advisory.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_generators.py", "test/api/semantic_refactoring/test_partition_generators.py"), "ProgramPartitionCandidate@1 generators", "python3 -m pytest -q test/api/semantic_refactoring/test_partition_generators.py"),
    (14, "Implement multi-objective partition comparison and policy", "SPAR-G032", "ipfs_accelerate_py", "partition-policy", ("SPAR-013",), "Evaluate complete reproducible objective breakdowns and fail closed on SCC, state, consumer, compatibility, ordering, resource, frontier, proof, or transaction constraints.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_policy.py", "test/api/semantic_refactoring/test_partition_policy.py"), "PartitionObjectiveProfile and comparison receipt", "python3 -m pytest -q test/api/semantic_refactoring/test_partition_policy.py"),
    (15, "Implement analogous-refactor retrieval and optional partition ranker", "SPAR-G032", "ipfs_accelerate_py", "partition-retrieval", ("SPAR-006", "SPAR-013"), "Retrieve exact/lexical/graph/vector prior refactors and optionally reorder already-admitted candidates without granting authority or hiding violations.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_retrieval.py", "test/api/semantic_refactoring/test_partition_retrieval.py"), "AnalogousRefactorRetriever and advisory ranker", "python3 -m pytest -q test/api/semantic_refactoring/test_partition_retrieval.py"),
    (16, "Synthesize module-boundary assume/guarantee contracts", "SPAR-G033", "ipfs_datasets_py", "boundary-contracts", ("SPAR-014",), "Create explicit assumptions, guarantees, effects, exceptions, state/resource ownership, initialization, concurrency, authorization, serialization, versioning, and proof obligations for every cut edge.", ("ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/boundary_contracts.py", "ipfs_datasets_py/tests/unit/semantic_refactoring/test_boundary_contracts.py"), "ModuleBoundaryContract@1 family", "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_boundary_contracts.py"),
    (17, "Synthesize target module APIs and dependency direction", "SPAR-G033", "ipfs_accelerate_py", "target-api", ("SPAR-014",), "Propose public/private exports, protocols, adapters, state-owner interfaces, responsibility statements, and cycle-free dependency directions.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/target_api.py", "test/api/semantic_refactoring/test_target_api.py"), "TargetModuleAPIPlan@1", "python3 -m pytest -q test/api/semantic_refactoring/test_target_api.py"),
    (18, "Plan compatibility façades and migrations", "SPAR-G033", "ipfs_accelerate_py", "facade-planning", ("SPAR-011", "SPAR-014"), "Generate explicit façade, re-export, wrapper, deprecation, CLI/plugin, registry, serialization, introspection, traceback, and patch-target migration plans for every consumer.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/facade_planner.py", "test/api/semantic_refactoring/test_facade_planner.py"), "CompatibilityFacadePlan@1", "python3 -m pytest -q test/api/semantic_refactoring/test_facade_planner.py"),
    (19, "Compile exact refactor transformation packets", "SPAR-G041", "ipfs_accelerate_py", "packet-compilation", ("SPAR-016", "SPAR-017", "SPAR-018"), "Bind exact preimages, bounded moves/rewrites/adapters/façade edits, expected deltas, allowed paths/effects, lease/fence, validation, and rollback.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/transformation_packet.py", "test/api/semantic_refactoring/test_transformation_packet.py"), "RefactorTransformationPacket@1", "python3 -m pytest -q test/api/semantic_refactoring/test_transformation_packet.py"),
    (20, "Implement or adapt CST-preserving move and extraction codemods", "SPAR-G041", "ipfs_accelerate_py", "cst-transform", ("SPAR-019",), "Use the current best available CST/codemod capability for deterministic bounded moves with comments/source maps preserved and typed refusal for unsupported constructs.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/codemod.py", "test/api/semantic_refactoring/test_codemod.py"), "CSTExtractionCodemod adapter", "python3 -m pytest -q test/api/semantic_refactoring/test_codemod.py"),
    (21, "Implement import, re-export, and callsite rewrite transformations", "SPAR-G041", "ipfs_accelerate_py", "import-transform", ("SPAR-019",), "Rewrite exact symbol imports/callsites, create authorized exports, verify preimages, and reject new cycles or undispositioned consumers.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/import_rewriter.py", "test/api/semantic_refactoring/test_import_rewriter.py"), "ImportRewrite and ReexportPlan executor", "python3 -m pytest -q test/api/semantic_refactoring/test_import_rewriter.py"),
    (22, "Implement explicit state-object and boundary-adapter transformations", "SPAR-G042", "ipfs_accelerate_py", "state-transform", ("SPAR-009", "SPAR-019"), "Move state owners and synthesize explicit state objects, protocols, adapters, or injection only with complete ownership/lifecycle/synchronization obligations.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/state_transform.py", "test/api/semantic_refactoring/test_state_transform.py"), "ExplicitStateObjectPlan executor", "python3 -m pytest -q test/api/semantic_refactoring/test_state_transform.py"),
    (23, "Preserve initialization, decorators, registries, CLI, plugins, and resource lifecycles", "SPAR-G042", "ipfs_accelerate_py", "initialization-transform", ("SPAR-010", "SPAR-018", "SPAR-019"), "Apply deterministic transformations or adapters for order-sensitive initialization, registrations, commands/routes/plugins, signals, atexit, and resources with required trace validation.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/initialization_transform.py", "test/api/semantic_refactoring/test_initialization_transform.py"), "InitializationRewritePlan executor", "python3 -m pytest -q test/api/semantic_refactoring/test_initialization_transform.py"),
    (24, "Preserve binding, introspection, and serialization compatibility", "SPAR-G042", "ipfs_accelerate_py", "binding-compatibility", ("SPAR-011", "SPAR-018", "SPAR-019"), "Implement exact wrappers/migrations for signatures, annotations, module/qualname, pickle, reflection, tracebacks, docs, and patch targets; classify intentional incompatibility.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/binding_compatibility.py", "test/api/semantic_refactoring/test_binding_compatibility.py"), "BindingCompatibilityAdapter", "python3 -m pytest -q test/api/semantic_refactoring/test_binding_compatibility.py"),
    (25, "Implement checkpointed transactional extraction waves", "SPAR-G043", "ipfs_accelerate_py", "transactional-waves", ("SPAR-020", "SPAR-021", "SPAR-022", "SPAR-023", "SPAR-024"), "Apply one bounded packet at a time in isolated fenced worktrees with exact before hashes, dependency order, rollback, VFS mutation receipts, and effect auditing.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/extraction_wave.py", "test/api/semantic_refactoring/test_extraction_wave.py"), "ExtractionWave executor and RollbackPlan", "python3 -m pytest -q test/api/semantic_refactoring/test_extraction_wave.py"),
    (26, "Integrate datasets test/proof selection and full-suite fallback", "SPAR-G051", "ipfs_accelerate_py", "selection-adapter", ("SPAR-007", "SPAR-008", "SPAR-011", "SPAR-019"), "Consume current datasets-owned selection bound to roots/obligations/uncertainty and enforce raw-source plus full-suite fallback where required.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/selection_adapter.py", "test/api/semantic_refactoring/test_selection_adapter.py"), "RefactorValidationSelectionAdapter", "python3 -m pytest -q test/api/semantic_refactoring/test_selection_adapter.py"),
    (27, "Implement translation-validation contracts and orchestrator", "SPAR-G051", "ipfs_accelerate_py", "translation-validation", ("SPAR-025", "SPAR-026"), "Validate each concrete original/candidate transformation across syntax, imports, APIs, types/effects, contracts, tests, proofs, traces, state, compatibility, and resources without collapsing evidence classes.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/translation_validation.py", "test/api/semantic_refactoring/test_translation_validation.py"), "TranslationValidationRequest/Result and RefactorEquivalenceClaim", "python3 -m pytest -q test/api/semantic_refactoring/test_translation_validation.py"),
    (28, "Implement differential execution and import/runtime trace comparison", "SPAR-G052", "ipfs_accelerate_py", "differential-execution", ("SPAR-025", "SPAR-026"), "Run hermetic paired old/new workflows and compare outputs, exceptions, effects, import events, registrations, state transitions, resources, and declared trace projections.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/differential.py", "test/api/semantic_refactoring/test_differential.py"), "DifferentialExecutionReceipt@1", "python3 -m pytest -q test/api/semantic_refactoring/test_differential.py"),
    (29, "Implement property, metamorphic, mutation, and adversarial validation", "SPAR-G052", "ipfs_accelerate_py", "mutation-validation", ("SPAR-025", "SPAR-026"), "Reuse/generate bounded relations and mutants for moved boundaries/façades and block acceptance on critical survivors or unknown required dynamics.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/adversarial_validation.py", "test/api/semantic_refactoring/test_adversarial_validation.py"), "RefactorMutationAndAdversarialValidator", "python3 -m pytest -q test/api/semantic_refactoring/test_adversarial_validation.py"),
    (30, "Integrate Tactician and production Hammer for boundary obligations", "SPAR-G053", "ipfs_accelerate_py", "tactician-hammer", ("SPAR-016", "SPAR-025", "SPAR-026"), "Build content-addressed premise corpora, lower and decompose finite obligations, run bounded proof/countermodel search, reconstruct proofs, and replay supported countermodels.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/proof_adapter.py", "test/api/semantic_refactoring/test_proof_adapter.py"), "TacticianHammer remodularization adapter", "python3 -m pytest -q test/api/semantic_refactoring/test_proof_adapter.py"),
    (31, "Integrate e-graphs, equality saturation, interpolation, and abstraction refinement", "SPAR-G053", "ipfs_accelerate_py", "proof-normalization", ("SPAR-027", "SPAR-028", "SPAR-030"), "Use sound proof-backed normalization and supported interpolation/refinement for suitable expressions, adapters, imports, state projections, and smaller boundary summaries.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/proof_normalization.py", "test/api/semantic_refactoring/test_proof_normalization.py"), "RefactorProofNormalizationAdapter", "python3 -m pytest -q test/api/semantic_refactoring/test_proof_normalization.py"),
    (32, "Implement bounded CEGIS/CEGAR boundary and adapter synthesis", "SPAR-G053", "ipfs_accelerate_py", "bounded-synthesis", ("SPAR-027", "SPAR-028", "SPAR-030"), "Synthesize small adapters, guards, protocols, state mappings, or initialization repairs from explicit examples/counterexamples/obligations, then re-enter full validation.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/bounded_synthesis.py", "test/api/semantic_refactoring/test_bounded_synthesis.py"), "BoundaryAdapterSynthesizer", "python3 -m pytest -q test/api/semantic_refactoring/test_bounded_synthesis.py"),
    (33, "Implement exact refactor-state, transition, proof, and procedure reuse", "SPAR-G061", "ipfs_accelerate_py", "exact-reuse", ("SPAR-025", "SPAR-027", "SPAR-030"), "Build exact freshness-bound reuse keys over trees, state, partitions, policy, environment, toolchain, obligations, validation, and procedure version; retain negative episodes.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/refactor_memory.py", "test/api/semantic_refactoring/test_refactor_memory.py"), "RefactorTransition@1 and exact reuse decision", "python3 -m pytest -q test/api/semantic_refactoring/test_refactor_memory.py"),
    (34, "Compile accepted refactor waves into proof-carrying procedures", "SPAR-G062", "ipfs_accelerate_py", "procedure-compilation", ("SPAR-031", "SPAR-032", "SPAR-033"), "Normalize accepted trajectories, anti-unify plans, infer preconditions/effects/rollback, preserve holes, qualify held-out/adversarial cases, and promote through the existing procedure authority.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/procedure_adapter.py", "test/api/semantic_refactoring/test_procedure_adapter.py"), "ProofCarryingProcedure refactor adapter", "python3 -m pytest -q test/api/semantic_refactoring/test_procedure_adapter.py"),
    (35, "Integrate minimal semantic context and residual model routing", "SPAR-G063", "ipfs_accelerate_py", "context-routing", ("SPAR-033",), "Extend the current ContextCompiler/compression harness with named unresolved questions, exact affected slices, contracts, counterexamples, evidence, analogous refactors, and allowed effects.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/context_adapter.py", "test/api/semantic_refactoring/test_context_adapter.py"), "SemanticRefactorContextAdapter and route receipt", "python3 -m pytest -q test/api/semantic_refactoring/test_context_adapter.py"),
    (36, "Implement monolith opportunity detection and goal compilation", "SPAR-G071", "ipfs_accelerate_py", "opportunity-detection", ("SPAR-007", "SPAR-008", "SPAR-009", "SPAR-010", "SPAR-011", "SPAR-014"), "Detect oversized or structurally overloaded modules under a versioned policy and compile exact findings, risks, autonomy, and acceptance into durable goals.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/opportunity_detector.py", "test/api/semantic_refactoring/test_opportunity_detector.py"), "MonolithOpportunityDetector and GoalCompiler", "python3 -m pytest -q test/api/semantic_refactoring/test_opportunity_detector.py"),
    (37, "Implement subgoal refinement, extraction-wave task synthesis, and backlog repair", "SPAR-G071", "ipfs_accelerate_py", "task-synthesis", ("SPAR-019", "SPAR-035", "SPAR-036"), "Refine findings into bounded inventory/extraction/state/boundary/façade/validation/repair/rescan/retirement tasks with exact scopes, evidence, rollback, and deduplicated retries.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/task_compiler.py", "test/api/semantic_refactoring/test_task_compiler.py"), "PartitionSubgoalRefiner and ExtractionWaveTaskCompiler", "python3 -m pytest -q test/api/semantic_refactoring/test_task_compiler.py"),
    (38, "Implement fixed-point remodularization controller", "SPAR-G071", "ipfs_accelerate_py", "fixed-point", ("SPAR-027", "SPAR-037"), "After every accepted/rejected wave rebuild semantic state, graph, SCCs, contracts, frontier, partitions, and selections until fixed-point or a typed terminal.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/fixed_point.py", "test/api/semantic_refactoring/test_fixed_point.py"), "FixedPointRemodularizationController", "python3 -m pytest -q test/api/semantic_refactoring/test_fixed_point.py"),
    (39, "Integrate semantic world roots, VFS outbox, recovery, and cross-repository updates", "SPAR-G072", "ipfs_accelerate_py", "root-integration", ("SPAR-025", "SPAR-033"), "Persist packets/projections/receipts/transitions through kit authorities and integrate generation CAS, recovery, stale-writer rejection, gitlinks, and explicit cross-repository ownership.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/world_root_adapter.py", "test/api/semantic_refactoring/test_world_root_adapter.py"), "SemanticRefactorWorldRootAdapter", "python3 -m pytest -q test/api/semantic_refactoring/test_world_root_adapter.py"),
    (40, "Activate shadow_plan", "SPAR-G073", "ipfs_accelerate_py", "shadow-plan", ("SPAR-038", "SPAR-039"), "Run complete analysis and planning without source mutation or routing influence; compare and retain false/unsafe candidates.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/rollout.py", "test/api/semantic_refactoring/test_shadow_plan.py"), "shadow_plan rollout gate", "python3 -m pytest -q test/api/semantic_refactoring/test_shadow_plan.py"),
    (41, "Activate shadow_apply", "SPAR-G073", "ipfs_accelerate_py", "shadow-apply", ("SPAR-040",), "Apply bounded packets only in disposable isolated worktrees, run full validation, publish hypothetical transitions, and prove no merge/root promotion.", ("test/api/semantic_refactoring/test_shadow_apply.py",), "shadow_apply rollout receipt", "python3 -m pytest -q test/api/semantic_refactoring/test_shadow_apply.py"),
    (42, "Activate and qualify guarded", "SPAR-G073", "ipfs_accelerate_py", "guarded", ("SPAR-041",), "Permit only qualified Tier A and selected Tier B waves through current merge gates; require approval above that ceiling and reject vector/model-only authority.", ("test/api/semantic_refactoring/test_guarded_rollout.py",), "guarded rollout qualification", "python3 -m pytest -q test/api/semantic_refactoring/test_guarded_rollout.py"),
    (43, "Activate required dogfooding", "SPAR-G073", "ipfs_accelerate_py", "required", ("SPAR-042",), "Require every subsequent program task to consume refactoring context and emit exact pre/post roots, partition, boundary, packet, route, validation, and transition receipts.", ("test/api/semantic_refactoring/test_required_rollout.py",), "required rollout gate", "python3 -m pytest -q test/api/semantic_refactoring/test_required_rollout.py"),
    (44, "Add typed Python/CLI/MCP control and diagnostics", "SPAR-G071", "ipfs_accelerate_py", "control-surface", ("SPAR-035", "SPAR-043"), "Extend the current typed Python service and thin CLI/MCP adapters with deterministic non-authoritative diagnostics and narrow authorized operations; MCP never shells out.", ("ipfs_accelerate_py/agent_supervisor/semantic_refactoring/service.py", "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/cli.py", "test/api/semantic_refactoring/test_control_surface.py"), "SemanticRefactoringService and thin adapters", "python3 -m pytest -q test/api/semantic_refactoring/test_control_surface.py"),
    (45, "Build controlled 5k/20k/100k-LOC scale fixtures and benchmark corpus", "SPAR-G081", "ipfs_accelerate_py", "benchmark-corpus", ("SPAR-043",), "Create rights-admitted frozen synthetic/controlled monoliths spanning state, initialization, dynamics, registries, async, exceptions, resources, APIs, and seeded defects with sealed splits.", ("benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json", "benchmarks/agent_supervisor/semantic_refactoring/fixtures", "test/api/semantic_refactoring/test_benchmark_corpus.py"), "Frozen scale corpus and split manifest", "python3 -m pytest -q test/api/semantic_refactoring/test_benchmark_corpus.py", "benchmark_fixture_generation"),
    (46, "Run end-to-end acceptance matrix", "SPAR-G081", "ipfs_accelerate_py", "acceptance-matrix", ("SPAR-044", "SPAR-045"), "Exercise inventory through extraction, façade, validation, transition, restart, reuse, replan, rollback, corruption, staleness, unsafe splits, failed proofs, and root conflicts.", ("test/api/semantic_refactoring/test_end_to_end_acceptance.py", "benchmarks/agent_supervisor/semantic_refactoring/acceptance_matrix.json"), "End-to-end acceptance matrix receipt", "python3 -m pytest -q test/api/semantic_refactoring/test_end_to_end_acceptance.py"),
    (47, "Benchmark autonomy, context reduction, partition quality, and semantic assurance", "SPAR-G081", "ipfs_accelerate_py", "benchmark", ("SPAR-046",), "Run the preregistered ablation ladder with unchanged denominators, safety gates, provider/tokenizer criteria, and honest failed/escalated/rejected/unavailable results.", ("benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json", "test/api/semantic_refactoring/test_benchmark_report.py"), "Preregistered ablation report", "python3 -m pytest -q test/api/semantic_refactoring/test_benchmark_report.py", "benchmark_execution"),
    (48, "Run adversarial, security, privacy, chaos, and recovery qualification", "SPAR-G081", "ipfs_accelerate_py", "adversarial-qualification", ("SPAR-047",), "Attack prompt injection, forged identity, poisoned vectors, frontier hiding, evidence weakening, state/order/compatibility breaks, leakage, cancellation, crashes, and concurrent publication.", ("benchmarks/agent_supervisor/semantic_refactoring/adversarial_report.json", "test/api/semantic_refactoring/test_adversarial_qualification.py"), "Adversarial/security/chaos qualification receipt", "python3 -m pytest -q test/api/semantic_refactoring/test_adversarial_qualification.py", "adversarial_qualification", "none", "No accepted state mutation outside current fenced authorities.", "Tier D"),
    (49, "Perform required-mode self-hosted capstone remodularization", "SPAR-G082", "ipfs_accelerate_py", "capstone", ("SPAR-048",), "Have the supervisor decompose a bounded real oversized module through verified waves, compile a reusable procedure, and run a later lower/no-LLM procedure wave without top-level completion bypass.", ("benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json", "test/api/semantic_refactoring/test_capstone_receipts.py"), "Required-mode capstone roots and transitions", "python3 -m pytest -q test/api/semantic_refactoring/test_capstone_receipts.py", "guarded_source_transformation", "compatibility façade only; no unaccepted API break", "Accepted transitions only through VFS/CAS/merge authority.", "Tier C"),
    (50, "Publish release, migration, compatibility, benchmark, and limitation report", "SPAR-G082", "ipfs_accelerate_py", "release", ("SPAR-049",), "Verify all seals/tasks/roots, focused/regression suites, identity preservation, model/context changes, unsupported dynamics, scale denominators, capstone, migration, façade retirement, and transitive final root.", ("docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json", "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_FINAL_REPORT.md", "test/api/semantic_refactoring/test_release_gate.py"), "Release and limitation reports", "python3 -m pytest -q test/api/semantic_refactoring/test_release_gate.py", "release_evidence"),
)


def _task(row: tuple[Any, ...]) -> TaskSpec:
    padded = (*row, "bounded_internal_implementation", "internal or compatibility-preserving only", "No new authority; immutable evidence or explicitly fenced state only.", "Tier B")
    index, title, goal, owner, track, deps, objective, paths, symbols, validation, effect, public_api, state, autonomy = padded[:14]
    return TaskSpec(f"SPAR-{index:03d}", title, goal, owner, track, tuple(deps), objective, tuple(paths), symbols, validation, effect, public_api, state, autonomy)


TASKS: Final = tuple(_task(row) for row in TASK_ROWS)

WAVES: Final = (
    ("W0", ("SPAR-000",)), ("W1", ("SPAR-001",)),
    ("W2", ("SPAR-002", "SPAR-003", "SPAR-004", "SPAR-005")),
    ("W3", ("SPAR-006", "SPAR-007", "SPAR-008", "SPAR-011")),
    ("W4", ("SPAR-009", "SPAR-010", "SPAR-012")),
    ("W5", ("SPAR-013",)), ("W6", ("SPAR-014", "SPAR-015")),
    ("W7", ("SPAR-016", "SPAR-017", "SPAR-018")),
    ("W8", ("SPAR-019",)),
    ("W9", ("SPAR-020", "SPAR-021", "SPAR-022", "SPAR-023", "SPAR-024")),
    ("W10", ("SPAR-025", "SPAR-026")),
    ("W11", ("SPAR-027", "SPAR-028", "SPAR-029", "SPAR-030")),
    ("W12", ("SPAR-031", "SPAR-032", "SPAR-033")),
    ("W13", ("SPAR-034", "SPAR-035")),
    ("W14", ("SPAR-036", "SPAR-037")),
    ("W15", ("SPAR-038", "SPAR-039")),
    ("W16", ("SPAR-040",)), ("W17", ("SPAR-041",)),
    ("W18", ("SPAR-042",)), ("W19", ("SPAR-043",)),
    ("W20", ("SPAR-044", "SPAR-045")), ("W21", ("SPAR-046",)),
    ("W22", ("SPAR-047",)), ("W23", ("SPAR-048",)),
    ("W24", ("SPAR-049",)), ("W25", ("SPAR-050",)),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _csv(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _goal_parent(goal_id: str) -> str:
    return next(goal.parent for goal in GOALS if goal.goal_id == goal_id)


def render_objectives() -> str:
    tree = """SPAR-G000  Build, self-host, and qualify semantic-preserving autonomous remodularization
|-- SPAR-G010  Freeze authorities, identities, risks, and baseline evidence
|   |-- SPAR-G011  Reconcile predecessor systems and source authorities
|   |-- SPAR-G012  Define symbol/module identity and projection contracts
|   `-- SPAR-G013  Define semantic, compatibility, and acceptance authority
|-- SPAR-G020  Build the function/module semantic and graph model
|   |-- SPAR-G021  Build function/class/block/module capsules
|   |-- SPAR-G022  Build program, state, initialization, and compatibility graphs
|   `-- SPAR-G023  Build multi-view projections and exact retrieval
|-- SPAR-G030  Plan coherent partitions and module boundaries
|   |-- SPAR-G031  Condense cycles and infer state ownership
|   |-- SPAR-G032  Generate and rank constrained partition candidates
|   `-- SPAR-G033  Synthesize boundary contracts, APIs, and migration plans
|-- SPAR-G040  Execute bounded transformations transactionally
|   |-- SPAR-G041  Compile deterministic transformation packets
|   |-- SPAR-G042  Preserve imports, initialization, state, and public compatibility
|   `-- SPAR-G043  Apply extraction waves with rollback and fixed-point control
|-- SPAR-G050  Prove and validate semantic preservation
|   |-- SPAR-G051  Select tests/proofs and validate translation
|   |-- SPAR-G052  Compare traces, effects, mutation, and compatibility
|   `-- SPAR-G053  Integrate Hammer, e-graphs, interpolation, and CEGIS
|-- SPAR-G060  Increase autonomy and reduce model dependence
|   |-- SPAR-G061  Build exact/procedural/refactor memory
|   |-- SPAR-G062  Compile repeated refactorings into procedures
|   `-- SPAR-G063  Minimize context and route only residuals to models
|-- SPAR-G070  Self-host supervisor planning and repository integration
|   |-- SPAR-G071  Create goals, subgoals, tasks, repair work, and fixed-point refill
|   |-- SPAR-G072  Integrate VFS, world roots, cross-repository changes, and recovery
|   `-- SPAR-G073  Activate progressive dogfooding
`-- SPAR-G080  Qualify scale, safety, benchmark evidence, and release
    |-- SPAR-G081  Run adversarial/security/chaos qualification
    `-- SPAR-G082  Run scale/capstone benchmarks and publish release evidence"""
    out = ["# Semantic-Preserving Autonomous Remodularization objectives", "", f"Board namespace: `{PROGRAM}`. Plan revision: `{PLAN_REVISION}`.", "", "DuckDB/DatabaseTaskSource is operational goal/task/completion authority through the exclusive Quack owner. DuckLake and this Markdown are non-authoritative projections.", "", "```text", tree, "```", ""]
    task_groups = {goal.goal_id: [task.task_id for task in TASKS if task.goal_id == goal.goal_id] for goal in GOALS}
    for goal in GOALS:
        parent_line = f"- Parent: {goal.parent}" if goal.parent else "- Parent:"
        dependency_line = f"- Depends on: {_csv(goal.depends_on)}" if goal.depends_on else "- Depends on:"
        grouped_tasks = tuple(task_groups[goal.goal_id])
        gap_line = f"- Gap task: {_csv(grouped_tasks)}" if grouped_tasks else "- Gap task:"
        out.extend((f"## {goal.goal_id} {goal.title}", "", "- Status: active", parent_line, dependency_line, "- Priority: P0", f"- Track: {goal.track}", f"- Goal: {goal.goal}", f"- Completion contract: {goal.completion}", "- Evidence: exact current-tree receipts; accepted child goal/task receipts; content-addressed proof/test/trace/transition evidence", "- Acceptance criteria: current-tree; authority-separated; fail-closed; no-self-authorization; fixed-point", "- Outputs: declared child task outputs and compact accepted root manifests", "- Validation: python3 scripts/validate_semantic_preserving_remodularization_board.py --check-all", "- Acceptance: Markdown status, DuckLake projection, model output, vector score, test pass alone, or worker claim is never completion authority.", gap_line, ""))
    return "\n".join(out)


def render_taskboard() -> str:
    out = ["# Semantic-Preserving Autonomous Remodularization task board", "", f"Executable sealed bootstrap projection for `{PROGRAM}` using plan revision `{PLAN_REVISION}`.", "DuckDB is authoritative for goals, tasks, dependencies, attempts, leases, fencing, CAS, evidence and completion. Quack is the exclusive loopback state-owner transport. DuckLake is optional non-authoritative history/analytics. Markdown cannot change runtime status.", "", "`SPAR-000` is operator-only. It is sealed and completed by the materializer after current-tree dependency/board validation; no implementation worker may claim or complete it. All ordinary tasks begin `todo`; dependencies project as waiting rather than blocked.", "", "## Parallel waves", "", "```text"]
    out.extend(f"{wave:<4} " + " | ".join(tasks) for wave, tasks in WAVES)
    out.extend(("```", "", "Only tasks in the same wave with disjoint exact scopes may execute concurrently. Unknown overlap, shared exports/registries/gitlinks/facades/configuration, or overlapping SCCs serialize under one integration owner.", ""))
    for task in TASKS:
        operator = task.task_id == "SPAR-000"
        parent = _goal_parent(task.goal_id)
        owned = _csv(task.paths)
        protected = "all operator control paths" if operator else "sealed SPAR controls, release floors, task/completion authority, and out-of-scope tests/policies"
        out.extend((
            f"## {task.task_id} {task.title}", "",
            f"- Stable task ID: {task.task_id}", "- Status: todo",
            f"- Completion: {'operator' if operator else 'auto'}",
            f"- Is schedulable: {'false' if operator else 'true'}",
            f"- Review only: {'true' if operator else 'false'}",
            "- Priority: P0", f"- Track: {task.track}", f"- Goal id: {task.goal_id}",
            f"- Parent goal ID: {parent}", f"- Subgoal ID: {task.goal_id}",
            f"- Owning repository: {task.owner}", f"- Board namespace: {PROGRAM}",
            f"- Base revision: {BASE_REVISION}", f"- Base repository tree: {BASE_TREE}", f"- Base plan revision: {PLAN_REVISION}",
            f"- Objective: {task.objective}", f"- Depends on: {_csv(task.dependencies)}" if task.dependencies else "- Depends on:",
            f"- Owned paths: {owned}", f"- Predicted files: {owned}", f"- Predicted symbols: {task.symbols}",
            "- Read scope: Exact accepted predecessors, current semantic-state/program-graph slices, declared source/test/proof inputs, and current capability receipts only.",
            f"- Write scope: {owned}; isolated leased/fenced worktree only; nested repository writes only when `{task.owner}` owns the task and gitlink integration is explicit.",
            "- External effect scope: Network denied for ordinary tests; no implicit install/model download; no protected-branch, credential, production, or undeclared repository effect.",
            "- Authority impact: Extends the named current authority through versioned contracts/adapters; never creates a competing task, graph, identity, VFS, proof, context, scheduler, vector, merge, or state authority.",
            f"- Effect class: {task.effect_class}", f"- Public API impact: {task.public_api}", f"- State impact: {task.state_impact}",
            "- Preconditions: All dependencies accepted against current roots; exact preimages; current provider/capability probe; required raw source available; no unresolved scope conflict.",
            "- Declared effects: Write owned paths; run declared hermetic validation; emit content-addressed receipts; request merges only through current authority.",
            "- Permitted effects: Deterministic bounded source/test/evidence changes in owned paths, isolated subprocess validation, and current-authority state commands through Quack.",
            f"- Prohibited effects: Edit {protected}; direct multi-process DuckDB; unrestricted diff; hidden dynamic frontier; vector/model authority; self-approval; fabricated evidence; implicit network/install.",
            "- Resource class: cpu-standard-local-proof", "- Timeout: 14400 seconds; maximum 21600 seconds",
            "- Provider role: implementation proposal; independent supervisor validation and merge authority remain separate",
            "- Context budget: input_tokens=36000; output_tokens=12000; exact/procedural prefix first; residual-only fallback",
            "- Token budget: input_tokens=36000; output_tokens=12000",
            "- Resource demand: cpu_ms=7200000; cpu_concurrency=2; ram_mib=4096; gpu_memory_mib=0; gpu_compute_class=none; disk_mib=4096; disk_bandwidth_mib_s=100; network=deny; network_bandwidth_kib_s=0; subprocesses=16; worktree_slots=1; provider_quota_units=1; provider_concurrency=1; prover_class=local; prover_concurrency=1; exclusive_keys=" + task.track + "; merge_slots=1; persistence_kib_s=2048",
            "- Model-route class: exact reuse, verified procedure, deterministic analysis/transform, proof/synthesis, specialist ranking, then one named residual general-model question",
            "- No-model route: exact receipt reuse -> verified procedure -> AST/CST/graph/state/effect/contract analysis -> proof search -> deterministic transform -> independent validation",
            "- Model fallback: one typed unresolved residual with bounded semantic slice; output is proposal-only and cannot remove a gate",
            f"- Autonomy tier: {task.autonomy}", "- Rollout mode: bootstrap until SPAR-040; then the explicitly accepted progressive mode",
            f"- Parallel lane: {task.track}", f"- Concurrency group: {task.track}",
            "- Conflict policy: Exact path/symbol/SCC/state-owner overlap serializes; shared exports, registries, gitlinks, façades, scheduler controls, and unknown scopes require an integration owner.",
            "- Lease and fencing: One exact task/attempt/base-tree/plan-root lease, token, epoch, fence, expiry, CAS and idempotency binding; stale holders cannot write, validate, merge, settle, or accept.",
            "- Interfaces: Existing datasets semantic contracts, kit storage/VFS contracts, accelerator supervisor/runtime contracts, and versioned narrow SPAR adapters only.",
            "- Acceptance subset: exact-current-tree; declared-effects; independent-validation; rollback; authority-separation; no-safety-floor-regression",
            "- Completion contract: Declared current-tree validation, evidence, compatibility, proof/trace requirements, merge/post-merge checks, and affected fixed-point conditions pass; unsupported required behavior is a typed terminal, never success.",
            f"- Validation: {task.validation}",
            "- Proof requirements: Exact source/tree/environment bindings; current obligation roots; independent reconstruction or explicitly bounded observation; no transfer across stale toolchain/profile/tree.",
            "- Rollback: Reject the candidate, restore exact preimages or discard the isolated worktree, retain negative evidence, release leases/fences, and do not advance accepted roots.",
            "- Required evidence: exact inputs/preimages; route/context receipt; diff/effect audit; test/proof/trace/compatibility receipts as applicable; merge/post-merge/root receipt",
            "- Evidence: Separate static facts, may-facts, runtime observations, specifications, tests, proof candidates, reconstructed proofs, countermodels, vector candidates, model hypotheses, decisions, and accepted transitions.",
            "- Final result identity: pending; only current authority derives it after validation, merge, and post-merge acceptance",
            f"- Outputs: {owned}",
            "- Raw-source requirements: Every directly edited source/test and every uncertain or dynamically affected region; vectors/projections cannot suppress raw-source fallback.",
            "- Protected paths: plan, objectives, board, dependency seal, authority matrix, benchmark preregistration, rollout baseline/policy, scheduler, materializer, validators, and release safety floors",
            "- Limitations: General Python equivalence is not claimed; evidence is bounded by the declared observation/proof profile and unresolved dynamics lower autonomy.",
            "- Capability blockers: none at bootstrap; newly observed unavailability remains typed and only blocks dependent work.", ""
        ))
    return "\n".join(out)


def render_plan() -> str:
    return f"""# Semantic-Preserving Autonomous Remodularization Plan

Plan revision: `{PLAN_REVISION}`
Board namespace: `{PROGRAM}`
Root goal: `SPAR-G000`
Task range: `SPAR-000` through `SPAR-050`

## Outcome

Build and qualify one semantic-preserving remodularization compiler across the authoritative `ipfs_datasets_py`, `ipfs_kit_py`, and `ipfs_accelerate_py` copies. It progressively decomposes oversized Python modules in bounded waves while preserving or explicitly migrating public contracts, state/resource ownership, initialization order, effects, exceptions, imports, registrations, serialization, introspection, and required runtime observations.

The top-level session owns only the sealed controls and genuine launch-blocker repairs. After `SPAR-000`, the existing `ipfs_accelerate_py.agent_supervisor` owns ordinary implementation through the current DuckDB/Quack task authority, leased/fenced worktrees, validation, merge queue, receipts, and completion gates.

## Exact bootstrap and preservation boundary

The execution checkout is the isolated worktree `/home/barberb/lift_coding/.worktrees/{PROGRAM}` on `{BRANCH}`, based on accelerator `{BASE_REVISION}` / tree `{BASE_TREE}`. Pinned gitlinks are datasets `41bbe7ede20294944cccb77f22072351a29e6902`, kit `80bbdc3443e560b9bf40339c864a32689ccad8ef`, and MCP++ `31096be86103f29faef80a01e03d09b1ad7345c6`.

The dirty `/home/barberb/lift_coding` superproject and its dirty embedded accelerator/datasets checkouts are preservation-only evidence. They are never reset, cleaned, stashed, force-checked out, submodule-updated, or used as clean completion evidence. Exact fingerprints live in `repository_baseline.json`.

## Reconciled authorities and predecessor reuse

- `ipfs_datasets_py` owns semantic capsules, identities, contracts, effects, state footprints, dependency/program graph meaning, dynamic frontiers, compatibility obligations, claims, proof obligations, and test/proof selection.
- `ipfs_kit_py` owns verified bytes/CIDs, immutable blocks, projection/index manifests, exact resolution, transition history, VFS mutations/outbox, WAL/recovery, and generation-bearing root CAS.
- `ipfs_accelerate_py` owns opportunity detection, goal/task refinement, route/context decisions, partition orchestration, transformation packets, fenced execution, validation orchestration, merge/operational acceptance, rollout, memory, and metrics.
- DuckDB/`DatabaseTaskSource@1` is transactional task/goal/lease/fence/CAS/evidence/completion authority. Exactly one Quack owner is the live multi-process boundary. DuckLake is rebuildable, non-authoritative history/analytics and never gates readiness or completion.

Current-tree semantic-state, `ArchitectureIR`, program graph, PCAR boundary/operator/state/public-surface components, Tactician/Hammer, procedure compiler, `ContextCompiler`, vector/hybrid retrieval, VFS/CAS/WAL, worktrees, leases, fences, merge queues, and DuckLake projection are reused. New work is a versioned narrow adapter or verified extension, never a second framework.

Historical SAWM, proof-gated repair/Tactician-Hammer, VFS assurance, semantic-state, semantic-compression, proof-carrying-procedure, and proof-carrying-architecture-refactorer branches are evidence candidates only. Names or Markdown completion never establish current authority; exact source/receipt lookup and freshness validation are required.

## Normative reasoning and compilation loop

```text
exact state and receipt reuse
  -> verified refactoring procedure
  -> deterministic syntax and graph analysis
  -> state/effect/initialization/compatibility constraints
  -> abstract interpretation and proof search
  -> deterministic transformation or bounded synthesis
  -> vector or learned candidate ranking
  -> minimal-context general LLM for one unresolved residual
  -> independent translation validation
  -> accepted content-addressed refactor transition
```

The compiler loop is monolith finding -> exact semantic capsules -> typed graph -> SCC condensation -> hard constraints/dynamic frontier -> multiple candidate partitions -> assume/guarantee boundaries -> façade/migration -> exact CST packet -> isolated wave -> proofs/tests/traces/mutation/compatibility -> accepted transition -> rescan/fixed point -> proof-carrying procedure extraction.

Vectors may nominate clusters, names, analogous refactors, state-owner candidates, or procedures. They never prove cohesion/equivalence/ownership, remove an edge or test/proof obligation, suppress raw source or dynamic uncertainty, authorize mutation, or establish completion.

## Identity and evidence model

Capsules distinguish source, CST, AST, location-independent implementation IR, symbol binding, interface contract, effect, state footprint, dependency slice, behavior summary, initialization dependency, public compatibility, validation profile, provenance, semantic-state root, and aggregate capsule identity. Moving code may preserve implementation/contract identity while changing binding/compatibility identity; that delta is explicit.

Top-level blocks, callsites, registrations, state owners, and resources are first-class. Projection identity binds subject, view, exact model/tokenizer/preprocessor/profile, dimension, metric, dtype/byte order/quantization, vector bytes, privacy, and availability. Nonfinite or mismatched vectors fail; unavailable neural capability is typed; deterministic structural analysis remains usable.

Evidence classes stay separate: exact static fact, conservative may-fact, runtime observation, reviewed specification, test, proof candidate, reconstructed proof, countermodel, replayed counterexample, vector candidate, model hypothesis, human/policy decision, and accepted transition.

## Graph, partition, and boundary policy

One typed logical multigraph covers repositories/packages/modules/top-level blocks/classes/functions/methods/callsites/basic blocks/variables/state owners/registries/decorators/resources/locks/transactions/configuration/contracts/claims/proofs/tests/external boundaries/aliases/procedures. Immutable Merkle snapshots remain physically acyclic.

Hard constraints include recursion SCCs, shared mutable state and unique owners, initialization/registration/decorator order, public imports/serialization, lock/transaction/resource ownership, security/FFI/framework boundaries, dependency direction, and required unresolved dynamic edges. Soft call/data/contract/test/proof/co-change/lexical/vector/trace signals only score candidates that already pass hard constraints.

Every cut edge receives explicit assume/guarantee inputs, outputs, conditions, invariants, exceptions, allowed/forbidden effects, state/resource owner, initialization, authorization, concurrency/atomicity, serialization, versioning, and proof obligations. Incomplete authoritative contracts cause retrieval/proof/abstention or review, not guessed axioms.

## Transform and compatibility policy

Packets bind exact repository/tree/environment/graph/partition/source preimages, moves, destinations, import/callsite rewrites, state/boundary adapters, façade edits, expected graph/semantic deltas, path/effect scopes, lease/fence, validation, and rollback. Deterministic CST transforms precede code generation and refuse unsupported constructs.

The original module remains a compatibility façade until every consumer is dispositioned. Validation covers import paths/star exports/module attributes/signatures/annotations/defaults/decorators/exceptions/CLI/plugins/registries/module/qualname/pickle/introspection/tracebacks/docs/configuration and patch targets. Import-time eagerness/order/resource lifetime never changes silently.

## Independent assurance

Each concrete `P -> P'` transformation binds an observation profile and combines parse/import/API/type/effect/contract/test/proof/property/metamorphic/differential/import-trace/workflow-trace/exception/state/serialization/introspection/CLI/plugin/registry/mutation/resource/performance evidence as required. A test proves only tested executions; a theorem proves only its encoded model; traces prove only observations.

Tactician selects bounded routes through premise retrieval/decomposition, Hammer proofs/countermodels, native reconstruction/replay, e-graphs, interpolation, abstraction refinement, and CEGIS/CEGAR. Contradictory premises yield conflict/abstention. Every synthesized candidate re-enters independent validation.

## Supervisor control plane and parallel execution

`SPAR-000` is completed only by the operator materializer after validators and focused bootstrap tests pass. `SPAR-001` then becomes the sole ready ordinary task. The 26 dependency waves in the taskboard expose disjoint work concurrently across three lanes; exact path/symbol/SCC/state-owner overlap and shared integration points serialize.

Runtime isolation uses `data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1`, DuckDB `control.duckdb`, exclusive Quack `127.0.0.1:46731`, separate lane/worktree/merge/log/evidence roots, and a non-authoritative DuckLake catalog/data root. Provider capability is re-probed at launch; secrets remain environment-only. Health requires a live process-birth-bound Quack owner plus actual query, coordinator/lane liveness, valid leases/fences/worktrees, no restart loop, no durable blockers, and an admitted state transition or typed dependency-idle frontier.

## Progressive rollout

`bootstrap` builds contracts/stores/planners/validators without autonomous source mutation. `shadow_plan` plans only. `shadow_apply` mutates disposable worktrees and never merges. `guarded` admits qualified Tier A/selected Tier B through current gates and requires approval above them. `required` blocks task dispatch/completion without pre/post roots, graph/partition/boundary/packet/context/route/validation/transition receipts. Workers cannot self-approve or change rollout.

## Benchmark preregistration and release floors

Freeze 5k/20k/100k LOC controlled monoliths plus a safely admitted real current-tree module before tuning. Compare manual/LLM-heavy and current-supervisor baselines through exact graphs, deterministic partitioning, translation validation, proof guidance, retrieval/ranker, procedures, guarded, and required modes. Retain failed, rejected, escalated, unavailable, and reviewed waves in denominators.

Targets are >=30% lower median general-LLM input context per accepted wave, >=25% fewer general-LLM calls per accepted wave, >=80% Tier A waves without general LLM, no loss of required coverage, and one later promoted-procedure wave with no general LLM. Missing efficiency targets prohibit promotion; they never relax safety.

Required zero floors: false completion, unauthorized mutation, similarity-only/stale/simulated admission, test/proof weakening, silent state duplication, unaccepted public API break, root-conflict overwrite, critical seeded defect escape, and rollback failure.

## Honest terminal and completion

Stop only at accepted fixed point/policy target, a typed external capability/human-review/unsupported terminal, or the final accepted program state. Completion requires every mandatory task accepted or explicitly permitted terminal, passing validators/seals, required-mode receipts/roots, a passing self-hosted capstone, no safety-floor violation, settled leases/merge queue/blocking obligations, and transitive final-root verification. The final report must state exact task states, merges, evidence, limitations, unsupported dynamics, benchmark denominators, context/model changes, migration/façade guidance, and identity preservation.
"""


def repository_baseline() -> dict[str, Any]:
    return {
        "schema": "spar/repository-baseline@1", "observed_at": OBSERVED_AT,
        "execution_root": f"/home/barberb/lift_coding/.worktrees/{PROGRAM}",
        "authoritative_repositories": [
            {"repository": "ipfs_accelerate_py", "root": f"/home/barberb/lift_coding/.worktrees/{PROGRAM}", "origin": "https://github.com/endomorphosis/ipfs_accelerate_py.git", "branch": BRANCH, "head": BASE_REVISION, "tree": BASE_TREE, "package_version": "0.0.45", "public_import_root": "ipfs_accelerate_py", "dirty_entries": 0},
            {"repository": "ipfs_datasets_py", "root": f"/home/barberb/lift_coding/.worktrees/{PROGRAM}/ipfs_datasets_py", "origin": "https://github.com/endomorphosis/ipfs_datasets_py.git", "branch": "detached-gitlink", "head": "41bbe7ede20294944cccb77f22072351a29e6902", "tree": "ed2edab3ffbba25e17b5a59aba1b9d2dd37cf06d", "package_version": "0.2.0", "public_import_root": "ipfs_datasets_py", "dirty_entries": 0},
            {"repository": "ipfs_kit_py", "root": f"/home/barberb/lift_coding/.worktrees/{PROGRAM}/ipfs_kit_py", "origin": "https://github.com/endomorphosis/ipfs_kit_py.git", "branch": "detached-gitlink", "head": "80bbdc3443e560b9bf40339c864a32689ccad8ef", "tree": "543ccd2bd78994d4e3d55cd44993ae6ca65e4c28", "package_version": "current-gitlink", "public_import_root": "ipfs_kit_py", "dirty_entries": 0},
        ],
        "auxiliary_gitlinks": [{"path": "ipfs_accelerate_py/mcplusplus", "head": "31096be86103f29faef80a01e03d09b1ad7345c6", "tree": "61776431577f0c276546f25e4178a47a64007180", "access": "initialized read-only prerequisite unless an explicit task owns it"}],
        "environment": {"python": "3.12.3", "implementation": "CPython", "executable": "/home/barberb/.local/bin/python", "pyproject_sha256": "26a8ed30df2d8cb08a0853ecd56d81077ab52ba571775df7ffa58494d8988c40", "requirements_sha256": "db63abfc0624140dfb23b96d8d024d48a83013626fc227a56c4c26ad6682cc01", "gitmodules_sha256": "9b5856970fc61b9c18f7d272c1ed5b9322e5e7df08a8f74a100da2e33852bf90"},
        "preserved_user_work": {
            "superproject": {"root": "/home/barberb/lift_coding", "branch": "chore/fmt-check-main", "head": "8601408d6406681a14a9488d31d1cd9a16164649", "tree": "e35dfaebbd229d7e7de988b2e0345444d9284221", "status_entries": 585915, "status_bytes": 114104426, "status_sha256": "f0a737302e4455c55e60edc28403da4f339e9b64811cc0b0dcf0c87e2d9629b3", "disposition": "untouched preservation boundary"},
            "embedded_accelerator": {"root": "/home/barberb/lift_coding/external/ipfs_accelerate", "head": "5d79bbf8bb61e027fbda76920580cb9d4fd919c1", "tree": "c5ddd043de58349aafac0f9af03c1b018313623d", "status_entries": 6, "status_sha256": "c261a97a9d4d91da96af386077e5cdf9c1532273614a73911d8c07a063913798", "disposition": "never used as clean evidence or mutated by SPAR"},
            "embedded_datasets": {"root": "/home/barberb/lift_coding/external/ipfs_datasets", "head": "ac82107e246b30e35a2bbdcf75e01370d22350c6", "tree": "2b3d892dd1c31fb6b8a3eebdb88616d411c49a47", "status_entries": 31, "status_sha256": "a1b4619c21ab5c19d90aa666b48d10d7f5034a2abe2ce57e0f854892429d9c83", "disposition": "never used as clean evidence or mutated by SPAR"},
        },
        "network_install_policy": "no implicit install, model download, or ordinary-test network access",
    }


def authority_matrix() -> dict[str, Any]:
    rows = [
        ("semantic capsules, identities, contracts, graphs, proofs, selection", "ipfs_datasets_py", "formal semantic authority"),
        ("verified bytes/CIDs, blocks, projection storage, VFS, WAL, CAS, recovery", "ipfs_kit_py", "storage and retrieval authority"),
        ("planning, context, routes, worktrees, validation, merge, acceptance", "ipfs_accelerate_py", "operational refactoring authority"),
        ("goals, tasks, dependencies, lifecycle, leases, fences, CAS, evidence, completion", "DuckDB/DatabaseTaskSource@1 via Quack", "live operational authority"),
        ("history and benchmark projection", "DuckLake", "non-authoritative rebuildable projection"),
        ("candidate similarity/ranking", "existing vector/hybrid indexes", "advisory=false authority"),
        ("model-generated hypotheses/diffs", "current model router", "proposal-only"),
    ]
    return {"schema": "spar/authority-matrix@1", "program": PROGRAM, "rules": [{"concern": a, "owner": b, "authority": c, "duplicate_authority_forbidden": True} for a, b, c in rows], "worker_control_paths_mutable": False}


def overlap_gap_matrix() -> dict[str, Any]:
    predecessors = [
        ("semantic-addressed-world-model-v1", "agent/semantic-addressed-world-model-v1", "9c92490e245a", "reuse candidate identities/roots only after current-store lookup and freshness"),
        ("proof-gated-contract-repair and Tactician/Hammer", "agent/proof-gated-contract-repair", "bcd8fc4efb6e", "reuse current landed proof adapters; historical task titles are not completion"),
        ("VFS symbolic assurance", "integration/ipfs-kit-fuse-vfs-main-20260810", "b6ce7c9688fa", "reuse current VFS/WAL/CAS interfaces and verify exact source"),
        ("semantic-state harness", "feat/semantic-state-harness", "6feb0a71138d", "reuse current semantic-state producer/view and receipts"),
        ("semantic-compression governor", "agent/semantic-compression-governor-v1", "485edc0871c5", "reuse current ContextCompiler/compression authority"),
        ("proof-carrying procedure compiler", "codex/proof-carrying-procedure-compiler-v1", "d426d838b0ef", "reuse current procedure compiler/registry/verifier"),
        ("proof-carrying architecture refactorer", "codex/proof-carrying-architecture-refactorer-v1", "0374c408469c", "reuse landed ArchitectureIR, graphs, boundaries, state/public surface, operators, scheduler patterns"),
    ]
    gaps = [
        "no landed semantic-refactoring capsule family with all declared identity dimensions",
        "no explicit initialization-order/happens-before refactor graph relation",
        "no deterministic source-mutation executor behind PCAR candidates; must use current fenced workers",
        "no arbitrary-Python extraction translation-validation receipt",
        "no SPAR accepted-wave/fixed-point contract composed from current roots and merge receipts",
    ]
    return {"schema": "spar/overlap-gap-matrix@1", "program": PROGRAM, "predecessors": [{"program": p, "branch": b, "historical_revision": r, "disposition": d, "current_tree_verification_required": True} for p, b, r, d in predecessors], "current_tree_reuse": ["agent_supervisor.semantic_state", "agent_supervisor.architecture_refactorer", "analysis.program_graph", "analysis.code_symbol_vector_index", "proof.tactician_hammer_coordinator", "procedure_compiler", "context.context_compiler", "merge worktree/lease/fence/queue", "task_sources DatabaseTaskSource/Quack", "kit VFS/WAL/CAS", "DuckLake projections"], "gaps": gaps, "supersession_policy": "preserve accepted identity/evidence; add only versioned adapter, missing integration, migration, qualification, or successor task"}


def interface_inventory() -> dict[str, Any]:
    items = [
        ("semantic_state", "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state", "reuse"),
        ("semantic_state_operational_adapter", "ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py", "reuse"),
        ("architecture_refactorer", "ipfs_accelerate_py/agent_supervisor/architecture_refactorer", "reuse_and_versioned_extension"),
        ("program_graph", "ipfs_accelerate_py/agent_supervisor/analysis/program_graph.py", "reuse_query_adapter"),
        ("vector_retrieval", "ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py", "reuse_advisory_only"),
        ("tactician_hammer", "ipfs_accelerate_py/agent_supervisor/proof/tactician_hammer_coordinator.py", "reuse_lowering_adapter"),
        ("procedures", "ipfs_accelerate_py/agent_supervisor/procedure_compiler", "reuse"),
        ("context", "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py", "reuse_ordering_adapter"),
        ("vfs_wal_cas", "ipfs_kit_py/ipfs_kit_py/core/vfs", "reuse"),
        ("worktree_lifecycle", "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py", "reuse"),
        ("duckdb_quack", "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py", "reuse_authority"),
        ("ducklake", "ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py", "reuse_non_authoritative"),
    ]
    return {"schema": "spar/interface-inventory@1", "observed_tree": BASE_TREE, "interfaces": [{"concern": a, "path": b, "disposition": c, "current_tree_probe_required_by": "SPAR-001"} for a, b, c in items]}


def identity_inventory() -> dict[str, Any]:
    fields = ("stable_symbol_id", "source_cid", "cst_cid", "ast_cid", "implementation_ir_cid", "symbol_binding_cid", "interface_contract_cid", "effect_summary_cid", "state_footprint_cid", "dependency_slice_cid", "behavior_summary_cid", "initialization_dependency_cid", "public_compatibility_cid", "validation_profile_cid", "provenance_cid", "semantic_state_root_cid", "function_capsule_cid")
    capsules = ("FunctionSemanticCapsule@1", "MethodSemanticCapsule@1", "ClassSemanticCapsule@1", "TopLevelBlockCapsule@1", "ModuleSemanticCapsule@1", "PackageSemanticCapsule@1", "CallsiteSemanticCapsule@1", "StateOwnerCapsule@1", "RegistrationCapsule@1", "ResourceLifecycleCapsule@1")
    return {"schema": "spar/identity-inventory@1", "status": "contract_planned_current_identity_reuse_required", "identity_fields": list(fields), "capsule_types": list(capsules), "rules": ["CID identifies exact canonical bytes under declared codec/profile, not universal meaning", "move may preserve implementation/contract identity while changing binding/compatibility identity", "timestamps, process IDs, local paths, and model output are excluded from semantic identity", "accepted @1 payloads are not rewritten in place"]}


def dynamic_risk_inventory() -> dict[str, Any]:
    risks = ("getattr/setattr/delattr", "globals/locals", "eval/exec/compile", "dynamic import/importlib", "metaclasses/descriptors", "decorator/class-decorator side effects", "monkeypatch/pytest monkeypatch", "plugins/entry points/registries", "singledispatch/multimethod", "dependency injection/callbacks/higher-order", "closures/nonlocals", "contextvars/thread/task locals", "signals/atexit", "module __getattr__/__dir__", "serialization/pickle", "inspect/getsource/signature", "__module__/__qualname__", "traceback/log path expectations", "relative/circular imports", "native extensions/FFI", "generated code", "ORM/model registration", "web route registration", "CLI command registration", "environment-dependent imports")
    return {"schema": "spar/dynamic-python-risk-inventory@1", "baseline_status": "planned_detection_not_safe_by_default", "risks": [{"kind": risk, "status": "unknown_until_current_module_analysis", "authority": "ipfs_datasets_py semantic facts", "coverage": "SPAR-008 plus task-specific runtime validation", "required_validation": "typed static finding and bounded observation where policy permits", "migration_policy": "preserve or explicitly migrate", "autonomy_tier": "Tier D; Tier E when opaque"} for risk in risks], "rule": "unknown widens the dynamic frontier and lowers autonomy"}


def benchmark_preregistration() -> dict[str, Any]:
    floors = ("false_task_completions", "unauthorized_mutations", "similarity_only_authoritative_reuse", "stale_authoritative_reuse", "simulated_as_live_admission", "test_or_proof_weakening", "silent_state_duplication", "unaccepted_public_api_break", "root_conflict_overwrite", "critical_seeded_defect_escape", "rollback_failure")
    return {"schema": "spar/benchmark-preregistration@1", "sealed_before_tuning": True, "profiles": [{"name": "small", "target_loc": 5000}, {"name": "medium", "target_loc": 20000}, {"name": "large", "target_loc": 100000}, {"name": "real_current_tree", "target_loc": None}], "ablation_ladder": ["manual_or_general_llm_heavy", "current_supervisor", "exact_capsules_graph", "deterministic_scc_state_contract_partition", "translation_validation", "tactician_hammer", "vector_graph_retrieval", "optional_ranker", "procedure_reuse", "guarded", "required"], "promotion_targets": {"median_general_llm_input_context_reduction": 0.30, "general_llm_calls_per_accepted_wave_reduction": 0.25, "tier_a_no_general_llm_fraction": 0.80, "required_coverage_loss": 0, "later_promoted_procedure_no_general_llm_waves": 1}, "zero_safety_floors": {name: 0 for name in floors}, "denominator_policy": "include failed, escalated, rejected, unavailable, and human-reviewed waves", "failed_efficiency_target": "non-promotion; safety floors remain unchanged"}


def rollout_baseline() -> dict[str, Any]:
    return {"schema": "spar/rollout-baseline@1", "current_mode": "bootstrap", "worker_may_change_mode": False, "modes": {"bootstrap": {"source_mutation": False, "merge": False, "gate_task": "SPAR-000"}, "shadow_plan": {"source_mutation": False, "merge": False, "gate_task": "SPAR-040"}, "shadow_apply": {"source_mutation": "disposable_worktree_only", "merge": False, "gate_task": "SPAR-041"}, "guarded": {"source_mutation": "Tier A and qualified Tier B", "merge": "current authority gates", "gate_task": "SPAR-042"}, "required": {"source_mutation": "receipt-bound by autonomy tier", "merge": "current authority gates", "gate_task": "SPAR-043"}}, "receipt_floor": ["pre_world_root_cid", "program_graph_snapshot_cid", "partition_candidate_cid", "boundary_contract_set_cid", "transformation_packet_cid", "context_receipt_cid", "route_decision_cid", "validation_receipt_cids", "refactor_transition_cid", "post_world_root_cid", "expected_root_generation", "resulting_root_generation", "rollout_mode"]}


def _control_documents() -> dict[str, bytes]:
    docs = {
        "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md": render_plan().encode(),
        "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md": render_objectives().encode(),
        "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md": render_taskboard().encode(),
    }
    inventories = {"repository_baseline.json": repository_baseline(), "authority_matrix.json": authority_matrix(), "overlap_gap_matrix.json": overlap_gap_matrix(), "interface_inventory.json": interface_inventory(), "identity_inventory.json": identity_inventory(), "dynamic_python_risk_inventory.json": dynamic_risk_inventory(), "benchmark_preregistration.json": benchmark_preregistration(), "rollout_baseline.json": rollout_baseline()}
    for name, payload in inventories.items():
        docs[f"docs/architecture/semantic_preserving_autonomous_remodularization_inventory/{name}"] = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    benchmark_prereg = benchmark_preregistration()
    docs["benchmarks/agent_supervisor/semantic_refactoring/preregistration.json"] = json.dumps(benchmark_prereg, indent=2, sort_keys=True).encode() + b"\n"
    for relative in (
        ".gitignore",
        "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
        "scripts/validate_semantic_preserving_remodularization_dependencies.py",
        "scripts/validate_semantic_preserving_remodularization_board.py",
        "scripts/materialize_semantic_preserving_remodularization_program.py",
        "test/api/semantic_refactoring/test_bootstrap_controls.py",
    ):
        path = ROOT / relative
        if path.is_file():
            docs[relative] = path.read_bytes()
    return docs


def dependency_seal(documents: dict[str, bytes]) -> dict[str, Any]:
    edges = sorted((dependency, task.task_id) for task in TASKS for dependency in task.dependencies)
    bootstrap_runtime_paths = (
        "pyproject.toml",
        "scripts/ops/agent_supervisor/spar_legacy_capture.py",
        "scripts/ops/agent_supervisor/spar_legacy_import_plan.py",
        "scripts/ops/agent_supervisor/spar_legacy_observation.py",
        "scripts/ops/agent_supervisor/spar_legacy_origin.py",
        "scripts/ops/agent_supervisor/spar_legacy_launch_transition.py",
        "scripts/ops/agent_supervisor/spar_merge_owner.py",
        "scripts/ops/agent_supervisor/spar_merge_owner_handoff.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/state_owner_bootstrap.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/launch_source_amendment.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/typed_database_task_source.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/spar_closeout_profile.py",
        "ipfs_accelerate_py/agent_supervisor/semantic_state/kit_source_forest.py",
        "ipfs_accelerate_py/agent_supervisor/semantic_state/durable_state.py",
        "ipfs_accelerate_py/agent_supervisor/semantic_state/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/closeout_snapshot.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/completion_projection_repair.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_kernel.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_provider_gate.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
    )
    seal = {
        "schema": "spar/dependency-source-seal@1", "program": PROGRAM, "plan_revision": PLAN_REVISION,
        "source_binding": {"accelerator": {"commit": BASE_REVISION, "tree": BASE_TREE}, "datasets": {"commit": "41bbe7ede20294944cccb77f22072351a29e6902", "tree": "ed2edab3ffbba25e17b5a59aba1b9d2dd37cf06d"}, "kit": {"commit": "80bbdc3443e560b9bf40339c864a32689ccad8ef", "tree": "543ccd2bd78994d4e3d55cd44993ae6ca65e4c28"}, "mcp_plus_plus": {"commit": "31096be86103f29faef80a01e03d09b1ad7345c6", "tree": "61776431577f0c276546f25e4178a47a64007180"}},
        "control_file_sha256": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(documents.items())},
        "bootstrap_runtime_file_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in bootstrap_runtime_paths
        },
        "task_count": len(TASKS), "goal_count": len(GOALS), "dependency_count": len(edges), "dependency_edges": [list(edge) for edge in edges], "dependency_root_cid": identity(edges),
        "predecessor_reuse_candidates": {"sawm_program_definition": "sha256:f581af1f2234c127231b47bb1bb8d42910b984dcd1bece9a6303949ac1ba0b72", "sawm_plan_root": "sha256:d9481937430405ff6a512e779b14b7ce676de45d277c65d3763ebe49445ba914", "sawm_source_binding": "sha256:81d0fbd74e48c214b5334432cfa0f55e5d6b51a6f5096fa85e323512cf11fb77", "sawm_migration_receipt": "sha256:7a3800b6ba8cdf722d846bb2fd0bf505b78114cd659f806f7eafe7df81ec71b8"},
        "rules": ["historical identities require current-store lookup, byte verification, source binding, and freshness", "Markdown is not completion authority", "DuckLake and vectors are non-authoritative", "SPAR-000 is operator-only"],
    }
    seal["seal_cid"] = identity(seal)
    return seal


def scheduler_config(seal: dict[str, Any]) -> dict[str, Any]:
    runtime = "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
    protected = list(_control_documents()) + ["config/semantic_preserving_autonomous_remodularization_dependencies.seal.json", "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"]
    hierarchy = {goal.goal_id: [child.goal_id for child in GOALS if child.parent == goal.goal_id] for goal in GOALS if any(child.parent == goal.goal_id for child in GOALS)}
    groups = {goal.goal_id: [task.task_id for task in TASKS if task.goal_id == goal.goal_id] for goal in GOALS if any(task.goal_id == goal.goal_id for task in TASKS)}
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.semantic-preserving-autonomous-remodularization.scheduler_config@1", "program_identifier": PROGRAM,
        "taskboard_path": "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md", "objectives_path": "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md", "plan_path": "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md", "validator_path": "scripts/validate_semantic_preserving_remodularization_board.py",
        "dependency_validator_path": "scripts/validate_semantic_preserving_remodularization_dependencies.py", "materializer_path": "scripts/materialize_semantic_preserving_remodularization_program.py", "dependency_seal_path": "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json", "dependency_seal_cid": seal["seal_cid"],
        "task_prefix": "SPAR-", "goal_prefix": "SPAR-G", "board_namespace": PROGRAM, "accepted_plan_revision_alias": PLAN_REVISION, "merge_target_branch": BRANCH,
        "source_binding": {"accelerator_required_ancestor": BASE_REVISION, "accelerator_planning_revision": BASE_REVISION, "accelerator_planning_tree": BASE_TREE, "accelerator_required_branch": BRANCH, "bootstrap_task_source": "duckdb", "require_clean_checkout_at_launch": True, "record_repository_revision_at_launch": True, "require_initialized_gitlinks": True, "require_superproject_gitlink_equals_nested_head": True, "require_clean_nested_worktree_at_task_start": True, "record_recursive_repository_forest_at_launch": True, "changed_revision_requires_fresh_inventory_and_baseline": True, "planning_revision_is_runtime_completion_evidence": False, "datasets_submodule_path": "ipfs_datasets_py", "datasets_planning_revision": "41bbe7ede20294944cccb77f22072351a29e6902", "kit_submodule_path": "ipfs_kit_py", "kit_planning_revision": "80bbdc3443e560b9bf40339c864a32689ccad8ef", "mcp_plus_plus_submodule_path": "ipfs_accelerate_py/mcplusplus", "mcp_plus_plus_planning_revision": "31096be86103f29faef80a01e03d09b1ad7345c6"},
        "initial_projection": {"task_count": len(TASKS), "task_dependency_count": seal["dependency_count"], "task_dependency_root_cid": seal["dependency_root_cid"], "completed_task_ids": ["SPAR-000"], "ready_task_ids": ["SPAR-001"], "blocked_task_ids": [], "terminal_task_id": "SPAR-050", "goal_count": len(GOALS), "root_goal_id": "SPAR-G000"},
        "database_program": {"authority_mode": "quack", "task_source_kind": "duckdb", "endpoint_secret_handle": "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "quack_endpoint": "quack:127.0.0.1:46731", "store_id": f"{runtime}/control.duckdb", "store_generation": "1", "schema_revision": "1", "event_store_path": f"{runtime}/events", "runtime_registry_path": f"{runtime}/registry", "worktree_root": f"{runtime}/worktrees", "export_profile": "spar-v1", "failover_policy": "fail_closed", "explicit_legacy": False, "claim_policy": {"schema": "ipfs_accelerate_py/agent-supervisor/database-claim-policy@1", "task_prefix": "SPAR-", "task_shard_count": 3, "strict_task_sharding": True, "idle_lane_work_stealing": "virgin-transfer"}},
        "operational_control_plane": {"name": "DuckDB + Quack + DuckLake", "duckdb_role": "authoritative transactional goals, tasks, lifecycle, schema, CAS, fencing, receipts and projection cursor", "quack_role": "mandatory multi-reader/multi-writer transport and exclusive fenced state owner", "ducklake_role": "non-authoritative history and benchmark analytics", "canonical_gateway": "StateRepository/QuackStateRepository with StateCommand transactions", "direct_multi_process_duckdb_file_open_permitted": False, "automatic_file_fallback_permitted": False, "outage_policy": "fail_closed", "markdown_is_authority": False},
        "authoritative_board_projection_repair": {"mode": "sealed_bootstrap_projection", "automatic_repair_before_launch": True, "allowed_drift": "supervisor_generated_guardrail_suffix_only", "bootstrap_receipt_path": f"{runtime}/evidence/bootstrap/bootstrap-materialization.json", "repair_receipt_path": f"{runtime}/evidence/control-plane/board-projection-repair.json", "canonical_authority": "DuckDB/DatabaseTaskSource@1 over Quack", "canonical_block_mutation_permitted": False, "markdown_task_mutation_permitted": False, "suppressed_finding_disposition": "typed_runtime_telemetry_only"},
        "launch_source_amendment_policy": {"required": True, "schema": "ipfs_accelerate_py/agent-supervisor/launch-source-amendment@1", "authoritative_store": "DuckDB/PlanRevisionRepository@1 over Quack", "append_mode": "exact_plan_revision_cas", "task_history_policy": "preserve_immutable_task_cids_and_receipts", "attempt_source_policy": "exact_launch_forest_and_worktree_preimage", "completion_policy": "current_tree_requalification_required", "cli_json_is_authority": False, "filesystem_projection_is_authority": False},
        "ducklake_projection_program": {"mode": "enabled_non_authoritative", "authority": False, "scheduling_prerequisite": False, "acceptance_prerequisite": False, "completion_prerequisite": False, "source": "DuckDB event/outbox snapshots", "catalog_path": f"{runtime}/ducklake/catalog.duckdb", "data_path": f"{runtime}/ducklake/data", "outage_policy": "typed_unavailable_continue_core", "network_access": "deny", "extension_install_policy": "no_implicit_install"},
        "max_lanes": 3, "strict_task_sharding": True, "idle_lane_work_stealing": "virgin-transfer", "exit_when_all_tracks_terminal": True, "objective_refill_enabled": True, "codebase_refill_enabled": True, "objective_goal_refinement_enabled": True, "refill_policy": {"derived_refill": {"min_open_tasks": 2, "max_tasks_per_epoch": 3, "max_open_tasks": 64, "cooldown_seconds": 300}, "source": "accepted current semantic-state findings only", "generic_fix_prompts_permitted": False, "identical_retry_without_new_evidence_permitted": False}, "retry_budget_guardrail_enabled": True, "dependency_guardrail_enabled": True, "reconciliation_guardrail_enabled": True,
        "poll_interval_seconds": 5, "daemon_interval_seconds": 20, "check_interval_seconds": 10, "stale_seconds": 1800, "watchdog_startup_grace_seconds": 300, "max_restarts": 4, "max_task_attempts": 4, "implementation_retry_budget": 4, "validation_retry_budget": 4, "merge_retry_budget": 4, "implementation_timeout_seconds": 14400, "implementation_max_timeout_seconds": 21600, "implementation_log_stall_seconds": 1200,
        "worktree_submodule_paths": ["ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py/mcplusplus"], "protected_paths": sorted(protected),
        "runtime_paths": {"root": runtime, "state": f"{runtime}/state", "worktrees": f"{runtime}/worktrees", "merge_queue": f"{runtime}/merge-queue", "logs": f"{runtime}/logs", "evidence": f"{runtime}/evidence", "quack_owner": f"{runtime}/quack-owner"},
        "lanes": [{"index": 0, "name": "spar-lane-0", "strict_shard_remainder": 0, "initial_task_ids": ["SPAR-001"], "initial_focus": "semantic-and-identity"}, {"index": 1, "name": "spar-lane-1", "strict_shard_remainder": 1, "initial_task_ids": [], "initial_focus": "storage-transformation-and-validation"}, {"index": 2, "name": "spar-lane-2", "strict_shard_remainder": 2, "initial_task_ids": [], "initial_focus": "proof-rollout-and-qualification"}],
        "goal_hierarchy": hierarchy, "task_groups": groups, "waves": [{"id": wave, "task_ids": list(tasks)} for wave, tasks in WAVES],
        "provider": {"primary_provider_id": "grok_cli", "primary_model_id": "grok-4.6", "fallback_provider_id": "codex", "fallback_model_id": "gpt-5.6-terra", "fallback_trigger": "primary_quota_exhausted", "fallback_reasoning_effort": "high", "merge_resolver_mode": "disabled_until_sealed_residual", "max_concurrency": 3, "capability_probe_required_at_launch": True, "secrets_from_environment_only": True, "secrets_in_argv_prompts_logs_or_receipts": False},
        "authority_policy": {"duckdb_transactional_authority": True, "quack_exclusive_state_owner_transport": True, "ducklake_projection_authority": False, "ducklake_projection_is_scheduling_prerequisite": False, "ducklake_projection_is_acceptance_prerequisite": False, "markdown_is_completion_authority": False, "task_board_status_is_completion_evidence": False, "direct_multi_process_duckdb_file_open_permitted": False, "automatic_quack_to_file_fallback": False, "provider_or_model_claim_is_completion_authority": False, "vector_similarity_is_authority": False, "one_authoritative_store_per_mutable_semantic_fact": True, "deterministic_current_tree_admission_required": True, "exact_receipt_identity_required": True, "unknown_ownership_disposition": "typed_blocker", "cross_repository_writes_require_explicit_owner_and_gitlink_receipt": True, "candidate_self_promotion": False, "procedure_self_authorization": False, "worker_self_approval": False},
        "autonomy_ceiling": {"candidate_may_raise_ceiling": False, "tier_a": "deterministic validation", "tier_b": "full declared validation", "tier_c": "guarded stronger trace/proof gates", "tier_d": "approval unless qualified procedure", "tier_e": "abstain", "public_api_state_provider_receipt_or_cross_package_migration_mode": "proposal_only"},
        "completion_policy": {"terminal_task_id": "SPAR-050", "all_task_dependencies_terminal_required": True, "goal_completion_contracts_required": True, "current_tree_required": True, "active_mutating_claims_empty_required": True, "merge_queue_settled_required": True, "blocking_obligations_empty_required": True, "required_receipts_and_seals_verify": True, "required_mode_roots_and_receipts_required": True, "self_hosted_capstone_required": True, "safety_floors_noncompensable": True, "non_success_terminals_never_report_success": True, "ducklake_outage_cannot_block_core_completion": True, "final_report_required": True},
    }


def render_payloads() -> dict[str, bytes]:
    payloads = _control_documents()
    seal = dependency_seal(payloads)
    payloads["config/semantic_preserving_autonomous_remodularization_dependencies.seal.json"] = json.dumps(seal, indent=2, sort_keys=True).encode() + b"\n"
    payloads["config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"] = json.dumps(scheduler_config(seal), indent=2, sort_keys=True).encode() + b"\n"
    return payloads


def render(root: Path, *, check: bool) -> dict[str, Any]:
    mismatches: list[str] = []
    written: list[str] = []
    for relative, data in sorted(render_payloads().items()):
        path = root / relative
        if path.is_file() and path.read_bytes() == data:
            continue
        if check:
            mismatches.append(relative)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(relative)
    return {"schema": "spar/control-render@1", "valid": not mismatches, "check": check, "written": written, "mismatches": mismatches, "task_count": len(TASKS), "goal_count": len(GOALS), "dependency_count": sum(len(task.dependencies) for task in TASKS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "check", "summary"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "summary":
        result = {"schema": "spar/control-summary@1", "program": PROGRAM, "plan_revision": PLAN_REVISION, "task_count": len(TASKS), "goal_count": len(GOALS), "waves": len(WAVES), "dependency_count": sum(len(task.dependencies) for task in TASKS)}
    else:
        result = render(args.repo_root.resolve(), check=args.command == "check")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
