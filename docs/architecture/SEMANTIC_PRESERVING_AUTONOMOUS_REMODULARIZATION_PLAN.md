# Semantic-Preserving Autonomous Remodularization Plan

Plan revision: `SPAR-PLAN-R1`
Board namespace: `semantic-preserving-autonomous-remodularization-v1`
Root goal: `SPAR-G000`
Task range: `SPAR-000` through `SPAR-050`

## Outcome

Build and qualify one semantic-preserving remodularization compiler across the authoritative `ipfs_datasets_py`, `ipfs_kit_py`, and `ipfs_accelerate_py` copies. It progressively decomposes oversized Python modules in bounded waves while preserving or explicitly migrating public contracts, state/resource ownership, initialization order, effects, exceptions, imports, registrations, serialization, introspection, and required runtime observations.

The top-level session owns only the sealed controls and genuine launch-blocker repairs. After `SPAR-000`, the existing `ipfs_accelerate_py.agent_supervisor` owns ordinary implementation through the current DuckDB/Quack task authority, leased/fenced worktrees, validation, merge queue, receipts, and completion gates.

## Exact bootstrap and preservation boundary

The execution checkout is the isolated worktree `/home/barberb/lift_coding/.worktrees/semantic-preserving-autonomous-remodularization-v1` on `codex/semantic-preserving-autonomous-remodularization-v1`, based on accelerator `e3c9831d4465d0e9f1aba336994a385541611895` / tree `fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051`. Pinned gitlinks are datasets `41bbe7ede20294944cccb77f22072351a29e6902`, kit `80bbdc3443e560b9bf40339c864a32689ccad8ef`, and MCP++ `31096be86103f29faef80a01e03d09b1ad7345c6`.

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
