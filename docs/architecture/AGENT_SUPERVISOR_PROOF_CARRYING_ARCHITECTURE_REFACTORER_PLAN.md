# Proof-Carrying Architecture Refactorer plan

Program identifier: `agent-supervisor-proof-carrying-architecture-refactorer-v1`

Primary subsystem: `ProofCarryingArchitectureRefactorer`

Task prefix: `PCAR-`

Root objective: `PCAR-G000`

Board namespace: `agent-supervisor-proof-carrying-architecture-refactorer-v1`

Plan revision: `PCAR-PLAN-R1`

## 1. Purpose and decision rule

This program reduces the architecture that the supervisor must repeatedly
reconstruct while preserving observable behavior, authority, effects, public
contracts, proof obligations, release evidence, and exact rollback. It extends
the current supervisor; it does not replace its semantic index, context
compiler, planners, proof system, worktree manager, control service, provider
router, or durable task authority.

Every admitted change must move a bounded concern toward:

```text
one canonical authority
  -> explicit typed adapters and projections
  -> a bounded dependency cone
  -> a compact semantic capsule
  -> selected current-tree validation and proof
  -> a signed refactor receipt with rollback
```

Entropy, token, file, and symbol reductions are prioritization signals only.
They never establish equivalence, safety, ownership, dead code, or promotion.

## 2. Sealed starting point

The planning baseline was inspected on 2026-08-21 UTC.

| Fact | Exact value |
|---|---|
| Repository | `https://github.com/endomorphosis/ipfs_accelerate_py` |
| Clean worktree | `/home/barberb/lift_coding/.worktrees/proof-carrying-architecture-refactorer-v1` |
| Branch | `codex/proof-carrying-architecture-refactorer-v1` |
| `origin/main` commit | `bbf7f68799072c2b81f7d96eac91f2df3c4b3952` |
| Starting tree | `a698da9e4b54e2929adacb613bc61ba3e72eed58` |
| Python | `3.12.3` |
| Package | `0.0.45` |
| DuckDB | `1.5.5` |
| Operation catalog | `control-operation-catalog@2`, 35 operations |
| Operation catalog CID | `baguqeeradphx3pal7n2brjzpoa3l6tyjb5xrh7ekwbyhhrxxtyqqkxztffua` |
| Required protected-branch check | `documentation-gates` |

The live branch-protection query found force pushes and deletion disabled. The
primary repository checkout is dirty and is excluded from all PCAR writes. The
clean PCAR worktree is the only campaign checkout.

Pinned read-only sibling contracts are initialized at their superproject
gitlinks: `ipfs_datasets_py` `480a1666f144ad606fcb3cacb66e59775f28d0d1`,
`ipfs_kit_py` `2564aea1ae35061f2165872aff91e8a40801ab7e`, and
`ipfs_accelerate_py/mcplusplus`
`5ac0ab162f420264fd224073a5df3f2d7c054ae3`. PCAR may inspect these
published contracts but may not write any sibling repository.

## 3. Prerequisite disposition

Every status below binds current-tree source and tests in the compact
prerequisite inventory. Planning documents, old receipts, and similarly named
classes are not implementation evidence.

| Prerequisite | Disposition | Program response |
|---|---|---|
| `SemanticCompressionHarness` | `available` | Reuse its current public interface. |
| `SemanticCompressionGovernor` | `available_with_caveats` | Qualify against the initialized datasets gitlink. |
| `AdversarialAssuranceEngine` | `available_with_caveats` | Reuse `AssuranceCampaignApi@1`; retain an exact-symbol blocker. |
| `IncrementalVerificationPlanner` | `available` | Reuse its selection and receipt contracts. |
| `IncrementalProofSealer` | `available_with_caveats` | Qualify with pinned datasets proof types. |
| `AdaptivePlanner` | `available` | Extend through typed candidates, not a second planner. |
| `ContextCompiler` | `available` | Reuse for frozen before/after context compilation. |
| `SupervisorControlService` | `available` | Add typed operations and project them to Python, CLI, and MCP. |
| `AutonomousMetaController` | `missing` | Record `prerequisite.autonomous_meta_controller.current_tree_missing`; continue independent work. |
| `ProofCarryingProcedureCompiler` | `missing` | Record `prerequisite.proof_carrying_procedure_compiler.current_tree_missing`; expose a narrow future adapter only. |

Missing optional capabilities cannot become simulated success and do not block
independent ArchitectureIR, analysis, validation, or planning work.

## 4. Operational authority

The campaign uses three deliberately unequal storage roles:

```text
DuckDB
  authoritative transactional goals, tasks, dependencies, attempts,
  leases, fencing, CAS state, receipts, and completion records

Quack
  authenticated loopback multi-reader/multi-writer transport and the
  exclusive mutation/state-owner boundary over that DuckDB authority

DuckLake
  optional non-authoritative history and benchmark projection only
```

Markdown is an immutable bootstrap and human projection. It is never task or
completion authority after materialization. DuckLake cannot grant authority,
gate readiness, acceptance, completion, or release, and an unavailable
DuckLake projection cannot turn otherwise valid work into success or failure.
Quack outage fails closed; there is no automatic direct-file multi-writer
fallback.

## 5. Non-compensable invariants

The following invariants are hard gates:

```text
NoAuthorityWeakening
NoEffectExpansion
NoHiddenBehaviorChange
NoSimulatedAsLive
NoValidationReduction
NoProofObligationLoss
NoPublicContractBreakWithoutVersionedMigration
NoStaleEvidencePromotion
NoUnboundedRefactor
NoProcedureSelfAuthorization
NoArchitectureCandidateSelfPromotion
NoCrossRepositoryWrite
NoSecretOrPrivateDataLeak
NoFalseCompletion
```

No score or efficiency improvement compensates for a violation. Unknown
ownership, multiple production authorities without formal arbitration,
unresolved contract interpretation, state dual authority, heuristic critical
equivalence, scope escape, missing rollback, or non-independent validation
rejects a candidate before execution.

## 6. Explicit non-goals

PCAR will not build another agent framework, semantic index, context compiler,
task identity, worktree manager, proof cache, provider router, receipt
hierarchy, theorem prover, dashboard, or model provider. It will not synthesize
arbitrary shell policy, rewrite the repository, autonomously modify keys or
security policy, reduce validation, release a high-risk change, store private
chain-of-thought, mutate sibling repositories, or commit large generated
graphs.

## 7. Canonical architecture model

`ArchitectureIR` is the one machine-readable architecture authority. It covers
repositories, packages, modules, files, symbols, interfaces, schemas,
operations, effects, authority/policy/state/storage/execution owners, receipts,
tests, proofs, artifacts, entrypoints, Python/CLI/MCP surfaces, providers,
legacy adapters, simulations, and generated projections.

Closed node kinds:

```text
RepositoryNode PackageNode ModuleNode FileNode SymbolNode InterfaceNode
SchemaNode OperationNode EffectNode AuthorityNode PolicyNode StateNode
ReceiptNode TestNode ProofNode ProviderNode EntrypointNode ArtifactNode
CompatibilityNode SimulationNode GeneratedNode
```

Closed edge kinds:

```text
contains imports calls constructs reads writes mutates authorizes
evaluates_policy confirms executes observes persists serializes deserializes
generates tests proves invalidates implements adapts reexports duplicates
shadows supersedes deprecates fallbacks_to
```

Every fact carries source span, extractor identity, confidence, freshness,
repository tree, and content identity. Confidence is one of `exact`,
`conservative`, `heuristic`, or `opaque`. Heuristic and opaque facts may widen
analysis but cannot prove equivalence, ownership, safe removal, or dead code.

## 8. Architecture truth and entropy

`AuthorityOwnershipGraph` resolves each concern to exactly one canonical
owner, zero or more typed adapters/projections, explicitly quarantined legacy
and simulation paths, and no unknown production owner. Initial concerns are:

```text
content identity operation identity provider capability provider selection
execution result task identity objective identity policy decision authorization
confirmation lease and fencing state persistence proof verification test evidence
completion evidence release qualification
```

Required findings include duplicate provider decisions, receipt producers and
state owners; compatibility or tool-dispatch bypasses; simulation-to-production
flow; Python/CLI/MCP divergence; re-export authorities; and tests that validate
obsolete rather than canonical behavior.

`SemanticEntropyReport` keeps independent, versioned dimensions:

```text
AuthorityMultiplicity ImplementationDuplication PublicSurfaceArea
DependencyConeSize DynamicDispatchUncertainty StateOwnershipAmbiguity
EffectOpacity CompatibilityBurden ValidationAmplification CacheFragmentation
SchemaDrift ReceiptDrift DocumentationDrift MergeConflictDensity ContextBurden
```

The report retains numerators, denominators, evidence identities, uncertainty,
and the frozen task corpus. Change amplification is reported alongside these
dimensions. A composite ranking may be derived, but the dimensions remain
independently auditable.

## 9. Duplicate discovery and bounded normalization

`SemanticDuplicateDetector` combines AST, call graph, effect signature, schema,
error vocabulary, test/proof overlap, bounded runtime traces,
anti-unification, and supported e-graph normalization. Text similarity alone
never classifies a duplicate.

Closed classifications are `exact_duplicate`, `alpha_equivalent`,
`adapter_duplicate`, `behaviorally_equivalent`, `partially_overlapping`,
`legacy_superseded`, `simulation_only`, `false_positive`, and `unknown`.
Every candidate records canonical and alternative implementations, common and
differing behavior/effects/authority, coverage, migration, risk, and predicted
context reduction.

Equality saturation is limited to reviewed domains: operation descriptors,
effect expressions, policy predicates, result normalization, schema
projections, dependency/validation expressions, and path-independent pure
helpers. Each rewrite is proved, solver-validated, translation-validated, or
explicitly heuristic. Extraction balances semantic size, runtime, dependency
count, effect opacity, public surface, context cost, and validation cost.

## 10. Contracts and boundaries

`ContractCandidateExtractor` mines candidate-tier input/output schemas,
pre/postconditions, effects, frame conditions, errors, idempotency,
reversibility, authority, policy, confirmation, resource limits, freshness,
and observation requirements from types, schemas, tests, runtime checks,
proofs, accepted receipts, negative tests, mutants, and explicitly
authoritative documentation.

Candidates are compared with every existing public contract, test, proof,
production receipt, negative example, and authoritative document. Conflict
emits typed `ContractAmbiguity`; repetition in code or tests is not promoted
to a requirement.

`InterfaceBoundarySynthesizer` proposes stable boundaries around provider
capability/selection, execution requests/outcomes, analysis/context, proof and
verification scheduling, task/objective state, control operations,
receipt/evidence queries, legacy compatibility, and simulations. A proposal
names its canonical owner, allowed callers/effects, state owner, migration
adapters, deprecations, tests, proofs, rollback, and predicted context and cone
reductions.

## 11. Closed refactor grammar

Only the following initial operators are admissible:

```text
EXTRACT_MODULE EXTRACT_INTERFACE EXTRACT_PURE_FUNCTION MOVE_STATE_TO_OWNER
INTRODUCE_DEPENDENCY_INVERSION REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE
GENERATE_ADAPTER GENERATE_COMPATIBILITY_SHIM QUARANTINE_LEGACY_PATH
QUARANTINE_SIMULATION_PATH REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME
REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY
CONSOLIDATE_ERROR_VOCABULARY CONSOLIDATE_RECEIPT_PRODUCER
CONSOLIDATE_CAPABILITY_AUTHORITY REMOVE_CONFIRMED_DEAD_CODE
SPLIT_MONOLITH_BY_AUTHORITY MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY
DEPRECATE_PUBLIC_SYMBOL REMOVE_DEPRECATED_SYMBOL_AFTER_GATE
```

Each declaration contains preconditions, target kinds, expected effects,
authority/API/state impact, migration, rollback, validation, proof obligations,
and maximum scope. Arbitrary refactor scripts cannot be promoted.

## 12. State, public surface, and quarantine

`StateOwnershipModel` classifies DuckDB tables, JSON and Markdown files,
in-memory registries, events, caches, worktree metadata, leases, provider/goal/
task/completion/receipt state as `authoritative`, `materialized_projection`,
`cache`, `historical_event`, `fixture`, `legacy`, or `unknown`. Every mutable
semantic fact has exactly one authoritative store. Migration uses bounded
snapshot, dual-read/shadow comparison, formally controlled dual-write,
cutover, validation, read-only legacy, and retirement phases; it never leaves
indefinite dual authority.

`PublicSurfaceManifest` classifies exports as `stable`, `provisional`,
`internal`, `compatibility`, `deprecated`, `simulation`, `test_only`, or
`accidentally_public`. Stable records name owner, versioned schema, effects,
errors, authority, tests, proofs, and consumers. Python, CLI, and MCP are typed
projections of the canonical operation catalog, with lazy side-effect-free
imports. Removal requires prior deprecation, consumer migration, replacement,
compatibility satisfaction, negative import tests, and release notes.

Legacy, compatibility, fixtures, and simulations are inventoried as
production-reachable, test-only, compatibility-only, dead, or unknown.
Noncanonical behavior moves behind explicit compatibility/simulation/fixture
namespaces. Static and dynamic flow proofs enforce that no value from those
namespaces can satisfy production capability, execution-success, proof,
completion, or release predicates. Static reachability alone never proves dead
code where dynamic loading is possible.

## 13. Proof-preserving pipeline

```text
finding -> bounded candidate -> candidate contract/effects -> isolated worktree
-> declarative operator -> static/type checks -> differential behavior
-> effect/authority comparison -> selected tests/proofs -> adversarial assurance
-> context/architecture benchmark -> merge preflight -> current-tree validation
-> signed receipt
```

Differential validation covers valid, invalid and boundary inputs; exceptions;
side effects; state transitions; receipts; performance; cancellation; timeouts;
and restart behavior. Only versioned, contract-admitted differences are
allowed. Generated changes preserve the original and receive an independent
equivalence/refinement obligation. Arbitrary Python uses differential,
property, bounded symbolic, type/effect, mutation, and human review rather than
claims of complete symbolic equivalence.

The future procedure-compiler adapter preserves task-family, authority, state,
effect, and validation boundaries. Until the compiler exists on the current
tree, declarative operator families remain usable without recreating it.

## 14. Planner and autonomous ceiling

`ArchitectureRefactorPlanner` ranks recurring benefit (context tokens,
dependency cone, authority ambiguity, validation amplification, merge
conflicts, public surface, procedure reuse, runtime/memory) against
implementation, validation, migration, risk, rollback, and consumer impact.
It preserves valid prefixes and replans only the affected suffix after a
counterexample.

With complete current-policy validation, low-risk pure extraction, generated
projection regeneration, lazy imports, internal adapters, closed outcomes,
confirmed dead internal code, fixture relocation, and simulation relocation
may execute automatically. Public API/state/provider/receipt/legacy/mutable
state changes remain proposal or pull-request only. Authorization, policy,
security, payment, wire protocol, key, release-authority, legal, and financial
changes always require human approval. No candidate can raise its ceiling or
promote itself.

## 15. Context benchmark and drift

Each candidate runs on the same frozen task corpus and records files, symbols,
interfaces, raw expansions, dependency hops, authorities, tokens, prefix reuse,
selected tests/proofs, and duration before and after. A reduction is invalid if
evidence coverage or safety falls, retries/validation/intervention rise
materially, or dependencies become opaque.

`ArchitectureDriftMonitor` compares the exact current graph with the admitted
architecture root and deduplicates unchanged findings. It reacts to new public
symbols, authority paths, mutable stores, cycles, simulation flows, duplicate
contracts/receipt producers/state owners, CLI/MCP mismatch, opaque dependency,
and dependency/context growth. It emits a minimal delta, invalidates only
affected evidence, opens a bounded finding, and remains idle on an unchanged
tree.

## 16. Control surface

Read operations:

```text
architecture.status architecture.graph architecture.authorities
architecture.public_surface architecture.state_owners architecture.legacy_paths
architecture.simulation_paths architecture.duplicates architecture.entropy
architecture.dependency_cone architecture.context_burden architecture.drift
architecture.refactor_candidates architecture.refactor_history
```

Mutation operations:

```text
architecture.analyze architecture.plan_refactor architecture.run_shadow
architecture.apply_candidate architecture.cancel architecture.promote
architecture.rollback architecture.quarantine architecture.deprecate
```

Every mutation is authorized, idempotent, dry-run capable, bound to exact
repository/tree/scope, leased and fenced, and audited. `ipfs-accelerate agent
architecture ...` and the MCP supervisor category call the same typed service;
MCP never shells out to CLI strings.

## 17. Goal tree, dependency waves, and conflict policy

The executable hierarchy and complete task contracts are in the objective heap
and task board. The root has four tranches and nested parallel subgoals:

```text
PCAR-G000
|-- PCAR-G010 architecture truth
|   |-- PCAR-G011 baseline and inventory
|   |-- PCAR-G012 IR, graph, entropy, and context
|   `-- PCAR-G013 canonical authority
|-- PCAR-G020 refactor candidates
|   |-- PCAR-G021 duplicate/contract/boundary/operator discovery
|   `-- PCAR-G022 surface/state/quarantine
|-- PCAR-G030 verified refactoring
|   |-- PCAR-G031 execution, comparison, and procedure adapter
|   `-- PCAR-G032 autonomy, drift, and read-only audit
`-- PCAR-G040 qualification
    |-- PCAR-G041 generated projections and public controls
    `-- PCAR-G042 benchmark, adversarial assurance, gates, and report
```

`PCAR-000` is the sole initial ready task. All other tasks begin `todo`; unmet
dependencies make them waiting, not blocked. Exact output scopes are disjoint
within each wave. Shared package exports and public control/CLI/MCP files are
owned only by their serialized integration tasks. Opaque scope serializes.
Plan, objectives, board, scheduler configuration, validators, and bootstrap
operator are protected from model workers.

Broad autonomous refactoring cannot begin until authority ownership, state
ownership, the operator grammar, differential validation, effect comparison,
and rollback are accepted.

## 18. Tranches and required outputs

1. **Architecture truth (`PCAR-000..007`)** seals the source and prerequisite
   baseline, inventories packages/entrypoints/authorities/stores, implements
   ArchitectureIR extraction, entropy/cone/context metrics, the ownership
   graph, and duplicate-authority findings. This complete tranche is the first
   mandatory implementation target.
2. **Refactor candidates (`PCAR-008..016`)** implements semantic duplicate
   discovery, bounded e-graphs, contract extraction, boundary synthesis,
   operator grammar, public surface, state ownership, inventories, and
   quarantine proofs.
3. **Verified refactoring (`PCAR-017..025`)** implements isolated execution,
   differential/effect/authority/translation validation, the future procedure
   adapter, planner, bounded executor, drift monitor, and read-only sibling
   audit.
4. **Qualification (`PCAR-026..031`)** generates compact current-tree
   projections, extends controls, runs the frozen benchmark and adversarial
   campaign, applies promotion/rollback gates, and emits the final exact-tree
   qualification/residual-gap report.

Large graph and benchmark bodies are managed artifacts addressed by CID.
Only compact manifests, schemas, fixtures, references, and reports are
committed.

## 19. Test and evidence program

Hermetic tests live under `test/api/architecture_refactorer/` and cover IR
round trips and unknown fields; identity; graph, import/call/effect edges;
authority and duplicate findings; false duplicates; e-graph rules; contract
ambiguity; boundaries; state conflicts; public surface; compatibility,
simulation and legacy flow; operator bounds; translation and differential
behavior/effect/error/receipt comparison; restart/rollback; drift and
deduplication; context and validation amplification; autonomy and
self-promotion; scope/symlink/submodule escape; test deletion/validation
weakening; forged/stale receipts/roots; cross-repository writes; and idle
stability. Network and paid providers are never required at collection. Live
provider tests use a separate marker.

Current prerequisite qualification commands are recorded in
`architecture_refactorer_inventory/qualified_tests.json`; `PCAR-000` must run
and bind their actual current-tree results before any claim that a prerequisite
is qualified.

## 20. Promotion gates

Safety requires zero undeclared behavioral differences, effect expansion,
authority weakening, state loss, receipt incompatibility, stale proof reuse,
or false completion.

For the selected initial subsystem, all targeted concerns must have one
canonical authority, unknown mutable-state ownership must be zero, production
simulation reachability must be zero, and direct canonical-control bypasses
must be zero.

Against the exact frozen baseline, promotion also requires at least 30% lower
median context tokens, 25% lower affected symbols, 20% lower affected files,
25% fewer raw-source expansions, 20% higher eligible stable-prefix reuse, and
20% lower validation amplification, with no loss of required test/proof
coverage or increase in escaped seeded defects or regressions. It requires 25%
more eligible architecture work through verified procedures, 40% fewer remote
model calls on repeated families, no added low-risk human intervention, and
zero autonomous high-risk refactors.

Counts of public symbols, modules, cycles, duplicates, legacy paths, and state
owners are reported but never optimized blindly.

## 21. Launch, recovery, and completion truth

Bootstrap order is fixed:

1. validate the committed plan/objectives/board/config;
2. materialize goals, edges, tasks, dependencies, outputs, acceptance, and
   validation records into a fresh DuckDB namespace;
3. create only an optional DuckLake history projection of the bootstrap event;
4. start the authenticated loopback Quack exclusive state owner;
5. independently probe the endpoint and exact store identity;
6. run configured-board preflight and deterministic dry-run;
7. launch implementation workers with bounded lanes;
8. verify from DuckDB authority that `blocked_task_ids` is empty and the first
   task is claimed or making progress.

The supervisor continues independent ready work after a typed capability
blocker. Repeated unchanged drift produces no new work. Restart reconstructs
from DuckDB records and receipts; it never trusts a process dictionary or
Markdown status. Failed attempts and counterexamples remain evidence. Rollback
uses the isolated task branch/worktree and exact pre-change tree. Final claims
must bind the actual merged current tree, tests/proofs run and not run,
architecture root, comparisons, interventions, blockers, eligibility, and
rollback target.

No report may claim simplified architecture, production readiness, behavior
preservation, authority consolidation, or token efficiency without exact
current-tree evidence for that specific statement.
