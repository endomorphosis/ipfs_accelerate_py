# Semantic-Addressed Neuro-Symbolic Program World Model objective graph

Durable goal hierarchy for plan revision `SAWM-PLAN-R2`, board namespace
`semantic-addressed-world-model-v1`, task prefix `SAWM-`, and goal prefix
`SAWM-G`. The authoritative task/event projection lives in the current
DuckDB + Quack control plane. DuckLake is a non-authoritative history/query
projection. Markdown never completes a goal or task.

Goal completion is a separate evidence-backed reconciliation decision. A
worker, model, retrieved result, prediction, procedure, proof candidate, or
root candidate may not approve itself. Missing, stale, corrupt, simulated,
similarity-only, unauthorized, privacy-incompatible, or environment-incompatible
evidence fails closed.

## Exact hierarchy

```text
SAWM-G000  Deliver and self-host the semantic-addressed
           neuro-symbolic program world model

|-- SAWM-G010  Freeze authorities, identities, and program controls
|   |-- SAWM-G011  Seal current repositories and capability authorities
|   |-- SAWM-G012  Reconcile landed systems and identify exact gaps
|   `-- SAWM-G013  Establish identity, relation, and authority contracts
|
|-- SAWM-G020  Build canonical static and dynamic program-world semantics
|   |-- SAWM-G021  Build the typed static program graph
|   |-- SAWM-G022  Build runtime state, event, and trace semantics
|   `-- SAWM-G023  Build abstraction, equivalence, and invalidation semantics
|
|-- SAWM-G030  Build verified Merkle-DAG storage and retrieval
|   |-- SAWM-G031  Store semantic objects and model-pinned projections
|   |-- SAWM-G032  Persist graph snapshots, transitions, and world roots
|   `-- SAWM-G033  Qualify recovery, VFS, replication, and concurrency
|
|-- SAWM-G040  Increase symbolic autonomy and reusable synthesis
|   |-- SAWM-G041  Implement exact and proof-backed reuse
|   |-- SAWM-G042  Integrate Tactician, Hammer, e-graphs, and CEGIS
|   `-- SAWM-G043  Compile accepted trajectories into procedures
|
|-- SAWM-G050  Add calibrated residual transition intelligence
|   |-- SAWM-G051  Build rights-admitted trace corpora and benchmarks
|   |-- SAWM-G052  Implement next-call edge completion
|   |-- SAWM-G053  Implement next-event, inverse-trace, and repair prediction
|   `-- SAWM-G054  Evaluate graph-sequence and structured model families
|
|-- SAWM-G060  Integrate with supervisor planning and context minimization
|   |-- SAWM-G061  Integrate context, memory, routing, and cognitive budgets
|   |-- SAWM-G062  Activate progressive self-hosting and required dogfooding
|   `-- SAWM-G063  Integrate causal federation and distributed execution
|
`-- SAWM-G070  Qualify safety, efficiency, self-hosting, and release
    |-- SAWM-G071  Run adversarial, corruption, recovery, and security gates
    `-- SAWM-G072  Benchmark, perform capstone, and publish release evidence
```

## Goal records

## SAWM-G000 Deliver and self-host the semantic-addressed neuro-symbolic program world model

- Stable goal ID: SAWM-G000
- Status: active
- Parent: none
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Child goal IDs JSON: ["SAWM-G010","SAWM-G020","SAWM-G030","SAWM-G040","SAWM-G050","SAWM-G060","SAWM-G070"]
- Fib priority: 1000
- Priority: P0
- Track: program
- Bundle: semantic-addressed-world-model/program
- Parallel lane: integration
- Resource class: coordinator
- Goal: Deliver the verified semantic-addressed program-world loop, progressively minimize general-LLM calls and context, self-host the supervisor in required mode, and preserve every proof, validation, authority, freshness, safety, scope, and completion guarantee.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-001 through SAWM-044
- Evidence: sawm/final-world-root@1, sawm/required-capstone@1, sawm/release-report@1
- Evidence requirements JSON: ["sawm/final-world-root@1","sawm/required-capstone@1","sawm/release-report@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: canonical program-world semantics, durable semantic-world storage, operational reuse and inference, required-mode capstone, and release evidence
- Predicted files: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py, test/api/semantic_world, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py","ipfs_kit_py","ipfs_accelerate_py","test/api/semantic_world","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: SemanticAddressedProgramWorld@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Every mandatory descendant goal is evidence-reconciled; the final generation-bearing world root verifies transitively; safety floors remain zero; no stale, simulated, similarity-only, or self-authorized result is promoted.
- Gap tasks: SAWM-001 through SAWM-044
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G010 Freeze authorities, identities, and program controls

- Stable goal ID: SAWM-G010
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on:
- Dependencies JSON: []
- Child goal IDs JSON: ["SAWM-G011","SAWM-G012","SAWM-G013"]
- Fib priority: 377
- Priority: P0
- Track: control
- Bundle: semantic-addressed-world-model/control
- Parallel lane: control
- Resource class: coordinator
- Goal: Freeze exact current source and dependency authority, reconcile prior SAWM history, and establish immutable R2 identities, relations, authority boundaries, task controls, and typed domain adapters.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-000, SAWM-001, SAWM-002, SAWM-003, SAWM-004
- Evidence: sawm/program-seal@1, sawm/authority-overlap@1, sawm/identity-conformance@1
- Evidence requirements JSON: ["sawm/program-seal@1","sawm/authority-overlap@1","sawm/identity-conformance@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: operator controls, authority/gap evidence, identity and relation contracts
- Predicted files: docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md, docs/architecture/semantic_addressed_world_model.objectives.md, docs/architecture/semantic_addressed_world_model.todo.md, docs/architecture/semantic_addressed_world_model_inventory, config/semantic_addressed_world_model_dependencies.seal.json, config/agent_supervisor_semantic_addressed_world_model_scheduler.json
- Predicted files JSON: ["docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md","docs/architecture/semantic_addressed_world_model.objectives.md","docs/architecture/semantic_addressed_world_model.todo.md","docs/architecture/semantic_addressed_world_model_inventory","config/semantic_addressed_world_model_dependencies.seal.json","config/agent_supervisor_semantic_addressed_world_model_scheduler.json"]
- Interfaces: SAWMProgramSeal@1, SemanticIdentityProfiles@1, ScopedSemanticRelation@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: The R2 board is sealed and materialized without rewriting history; every claimed owner and capability is current-tree verified or explicitly unavailable.
- Gap tasks: SAWM-000, SAWM-001, SAWM-002, SAWM-003, SAWM-004
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G011 Seal current repositories and capability authorities

- Stable goal ID: SAWM-G011
- Status: active
- Parent: SAWM-G010
- Parent goal IDs JSON: ["SAWM-G010"]
- Depends on:
- Dependencies JSON: []
- Child goal IDs JSON: []
- Fib priority: 233
- Priority: P0
- Track: authority-seal
- Bundle: semantic-addressed-world-model/authority-seal
- Parallel lane: control
- Resource class: cpu-small
- Goal: Bind the program to exact repository, tree, gitlink, dependency, environment, package, task-store, DuckDB, Quack, and DuckLake observations while preserving dirty user state.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-000, SAWM-001
- Evidence: sawm/source-dependency-seal@1, sawm/repository-baseline@1
- Evidence requirements JSON: ["sawm/source-dependency-seal@1","sawm/repository-baseline@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: sealed controls and current-tree authority evidence
- Predicted files: docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md, docs/architecture/semantic_addressed_world_model.objectives.md, docs/architecture/semantic_addressed_world_model.todo.md, docs/architecture/semantic_addressed_world_model_inventory, config/semantic_addressed_world_model_dependencies.seal.json, config/agent_supervisor_semantic_addressed_world_model_scheduler.json
- Predicted files JSON: ["docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md","docs/architecture/semantic_addressed_world_model.objectives.md","docs/architecture/semantic_addressed_world_model.todo.md","docs/architecture/semantic_addressed_world_model_inventory","config/semantic_addressed_world_model_dependencies.seal.json","config/agent_supervisor_semantic_addressed_world_model_scheduler.json"]
- Interfaces: SAWMSourceSeal@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: No stale copy, installed package, DuckLake projection, Markdown status, or dirty checkout is mistaken for current operational authority.
- Gap tasks: SAWM-000, SAWM-001
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G012 Reconcile landed systems and identify exact gaps

- Stable goal ID: SAWM-G012
- Status: active
- Parent: SAWM-G010
- Parent goal IDs JSON: ["SAWM-G010"]
- Depends on: SAWM-G011
- Dependencies JSON: ["SAWM-G011"]
- Child goal IDs JSON: []
- Fib priority: 144
- Priority: P0
- Track: reconciliation
- Bundle: semantic-addressed-world-model/reconciliation
- Parallel lane: inventory
- Resource class: cpu-medium
- Goal: Classify every requested capability as available, caveated, partial, stale, incompatible, missing, duplicate non-authoritative, or historical only, with exact source and test evidence.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-001
- Evidence: sawm/authority-overlap@1, sawm/capability-gap@1
- Evidence requirements JSON: ["sawm/authority-overlap@1","sawm/capability-gap@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: execution-time authority and overlap receipt
- Predicted files: docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md, docs/architecture/semantic_addressed_world_model.objectives.md, docs/architecture/semantic_addressed_world_model.todo.md, docs/architecture/semantic_addressed_world_model_inventory, config/semantic_addressed_world_model_dependencies.seal.json, config/agent_supervisor_semantic_addressed_world_model_scheduler.json
- Predicted files JSON: ["docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md","docs/architecture/semantic_addressed_world_model.objectives.md","docs/architecture/semantic_addressed_world_model.todo.md","docs/architecture/semantic_addressed_world_model_inventory","config/semantic_addressed_world_model_dependencies.seal.json","config/agent_supervisor_semantic_addressed_world_model_scheduler.json"]
- Interfaces: SemanticWorldAuthorityMatrix@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Every desired capability maps to one landed owner or a verified gap; names, plans, fixtures, and historical artifacts alone establish nothing.
- Gap tasks: SAWM-001
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G013 Establish identity, relation, and authority contracts

- Stable goal ID: SAWM-G013
- Status: active
- Parent: SAWM-G010
- Parent goal IDs JSON: ["SAWM-G010"]
- Depends on: SAWM-G012
- Dependencies JSON: ["SAWM-G012"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: identity-authority
- Bundle: semantic-addressed-world-model/identity-authority
- Parallel lane: datasets-contracts
- Resource class: cpu-medium
- Goal: Keep exact identity, semantic projections, scoped relations, and admission authority distinct through closed versioned datasets contracts and adapters.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-002, SAWM-003, SAWM-004
- Evidence: sawm/identity-conformance@1, sawm/relation-conformance@1, sawm/domain-adapter-conformance@1
- Evidence requirements JSON: ["sawm/identity-conformance@1","sawm/relation-conformance@1","sawm/domain-adapter-conformance@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: datasets identity profiles, scoped relation claims, typed domain adapters
- Predicted files: docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md, docs/architecture/semantic_addressed_world_model.objectives.md, docs/architecture/semantic_addressed_world_model.todo.md, docs/architecture/semantic_addressed_world_model_inventory, config/semantic_addressed_world_model_dependencies.seal.json, config/agent_supervisor_semantic_addressed_world_model_scheduler.json
- Predicted files JSON: ["docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md","docs/architecture/semantic_addressed_world_model.objectives.md","docs/architecture/semantic_addressed_world_model.todo.md","docs/architecture/semantic_addressed_world_model_inventory","config/semantic_addressed_world_model_dependencies.seal.json","config/agent_supervisor_semantic_addressed_world_model_scheduler.json"]
- Interfaces: SemanticObjectEnvelope@1, ScopedSemanticRelation@1, SemanticDomainAdapter@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Embedding changes affect projections only; relation scope and invalidators are retained; unavailable domains return typed unavailability; existing @1 identities remain unchanged.
- Gap tasks: SAWM-002, SAWM-003, SAWM-004
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G020 Build canonical static and dynamic program-world semantics

- Stable goal ID: SAWM-G020
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G010
- Dependencies JSON: ["SAWM-G010"]
- Child goal IDs JSON: ["SAWM-G021","SAWM-G022","SAWM-G023"]
- Fib priority: 233
- Priority: P0
- Track: program-semantics
- Bundle: semantic-addressed-world-model/program-semantics
- Parallel lane: datasets
- Resource class: cpu-large
- Goal: Define deterministic static program graphs, runtime states, events, traces, abstractions, transitions, and multi-view projections under datasets semantic authority.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-005 through SAWM-010, SAWM-038
- Evidence: sawm/static-program-world@1, sawm/dynamic-program-world@1, sawm/abstraction-soundness@1
- Evidence requirements JSON: ["sawm/static-program-world@1","sawm/dynamic-program-world@1","sawm/abstraction-soundness@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: canonical static and dynamic program-world contracts and construction
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state","ipfs_datasets_py/ipfs_datasets_py/logic/software_verification"]
- Interfaces: ProgramGraphSnapshot@1, ProgramExecutionState@1, ProgramTransitionQuery@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Physical blocks remain acyclic while logical cycles are representable; incomplete dynamic behavior stays explicit; raw and abstract identities are distinct and fail closed.
- Gap tasks: SAWM-005 through SAWM-010, SAWM-038
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G021 Build the typed static program graph

- Stable goal ID: SAWM-G021
- Status: active
- Parent: SAWM-G020
- Parent goal IDs JSON: ["SAWM-G020"]
- Depends on: SAWM-G013
- Dependencies JSON: ["SAWM-G013"]
- Child goal IDs JSON: []
- Fib priority: 144
- Priority: P0
- Track: static-graph
- Bundle: semantic-addressed-world-model/static-graph
- Parallel lane: datasets-static
- Resource class: cpu-large
- Goal: Represent and deterministically construct current-tree nodes, edges, snapshots, deltas, successor sets, and unresolved dynamic frontiers with precise invalidation.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-005, SAWM-006, SAWM-038
- Evidence: sawm/program-graph-contracts@1, sawm/program-graph-build@1
- Evidence requirements JSON: ["sawm/program-graph-contracts@1","sawm/program-graph-build@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: static graph contracts, builder, multi-view projections
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state","ipfs_datasets_py/ipfs_datasets_py/logic/software_verification"]
- Interfaces: ProgramGraphNode@1, ProgramGraphEdge@1, ProgramGraphSnapshot@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Graph construction binds exact source/environment, canonicalizes ordering, preserves cycles and unchanged subroots, and never treats unknown behavior as absent.
- Gap tasks: SAWM-005, SAWM-006, SAWM-038
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G022 Build runtime state, event, and trace semantics

- Stable goal ID: SAWM-G022
- Status: active
- Parent: SAWM-G020
- Parent goal IDs JSON: ["SAWM-G020"]
- Depends on: SAWM-G013
- Dependencies JSON: ["SAWM-G013"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: execution-semantics
- Bundle: semantic-addressed-world-model/execution-semantics
- Parallel lane: datasets-tracing
- Resource class: cpu-large
- Goal: Define and collect bounded hermetic Python execution states, frames, events, traces, observations, environment bindings, privacy profiles, and unavailable dimensions.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-007, SAWM-008
- Evidence: sawm/execution-contracts@1, sawm/hermetic-trace@1
- Evidence requirements JSON: ["sawm/execution-contracts@1","sawm/hermetic-trace@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: dynamic execution contracts and Python tracing adapter
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state","ipfs_datasets_py/ipfs_datasets_py/logic/software_verification"]
- Interfaces: ProgramExecutionState@1, ProgramEvent@1, ExecutionTrace@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Every event binds exact code and environment; redaction is explicit; raw trace bodies remain private; cancellation emits no accepted transition.
- Gap tasks: SAWM-007, SAWM-008
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G023 Build abstraction, equivalence, and invalidation semantics

- Stable goal ID: SAWM-G023
- Status: active
- Parent: SAWM-G020
- Parent goal IDs JSON: ["SAWM-G020"]
- Depends on: SAWM-G021, SAWM-G022
- Dependencies JSON: ["SAWM-G021","SAWM-G022"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: abstraction-transition
- Bundle: semantic-addressed-world-model/abstraction-transition
- Parallel lane: datasets-abstraction
- Resource class: cpu-large
- Goal: Define sound versioned abstractions and proposal-only transition contracts for next-call, next-event, inverse-trace, and repair queries.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-009, SAWM-010
- Evidence: sawm/state-abstraction@1, sawm/program-transition-contracts@1
- Evidence requirements JSON: ["sawm/state-abstraction@1","sawm/program-transition-contracts@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: abstraction profiles, receipts, transition and admission contracts
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state","ipfs_datasets_py/ipfs_datasets_py/logic/software_verification"]
- Interfaces: StateAbstractionProfile@1, ProgramTransitionPrediction@1, ProgramTransitionAdmission@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Relevant transition distinctions survive abstraction, unsafe merges fail closed, and predictions can never establish proof, authorization, observation, reuse, or completion.
- Gap tasks: SAWM-009, SAWM-010
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G030 Build verified Merkle-DAG storage and retrieval

- Stable goal ID: SAWM-G030
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G010
- Dependencies JSON: ["SAWM-G010"]
- Child goal IDs JSON: ["SAWM-G031","SAWM-G032","SAWM-G033"]
- Fib priority: 233
- Priority: P0
- Track: durability
- Bundle: semantic-addressed-world-model/durability
- Parallel lane: kit
- Resource class: io-large
- Goal: Compose kit CID/IPLD/CAR/CAS/WAL/VFS/vector/GraphRAG authorities into verified immutable program-world persistence, advisory projection retrieval, histories, roots, recovery, and concurrency.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-011 through SAWM-014
- Evidence: sawm/verified-semantic-store@1, sawm/world-root-cas@1, sawm/storage-recovery@1
- Evidence requirements JSON: ["sawm/verified-semantic-store@1","sawm/world-root-cas@1","sawm/storage-recovery@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: kit semantic-world storage facade, projection resolver, graph history, root CAS and recovery
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_world_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/semantic_world_store"]
- Interfaces: SemanticWorldStore@1, ProjectionResolver@1, WorldRootRepository@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Store-before-reference, exact CID verification, generation CAS, corruption failure, idempotent replay, and advisory-only ANN behavior are current-tree tested.
- Gap tasks: SAWM-011 through SAWM-014
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G031 Store semantic objects and model-pinned projections

- Stable goal ID: SAWM-G031
- Status: active
- Parent: SAWM-G030
- Parent goal IDs JSON: ["SAWM-G030"]
- Depends on: SAWM-G013
- Dependencies JSON: ["SAWM-G013"]
- Child goal IDs JSON: []
- Fib priority: 144
- Priority: P0
- Track: immutable-storage
- Bundle: semantic-addressed-world-model/immutable-storage
- Parallel lane: kit-artifacts
- Resource class: io-medium
- Goal: Persist verified semantic blocks and explicit deterministic projection records, then search rebuildable indexes that always resolve candidates to exact immutable objects.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-011, SAWM-012
- Evidence: sawm/verified-block-store@1, sawm/projection-resolution@1
- Evidence requirements JSON: ["sawm/verified-block-store@1","sawm/projection-resolution@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: verified store facade, projection records, index manifests, exact resolver
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_world_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/semantic_world_store"]
- Interfaces: VerifiedSemanticStore@1, ProjectionRecord@1, ProjectionIndexManifest@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: NaN, infinity, byte-order ambiguity, unpinned models, dimension mismatch, corrupt CIDs, and authoritative ANN claims fail closed.
- Gap tasks: SAWM-011, SAWM-012
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G032 Persist graph snapshots, transitions, and world roots

- Stable goal ID: SAWM-G032
- Status: active
- Parent: SAWM-G030
- Parent goal IDs JSON: ["SAWM-G030"]
- Depends on: SAWM-G031
- Dependencies JSON: ["SAWM-G031"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: world-history
- Bundle: semantic-addressed-world-model/world-history
- Parallel lane: kit-history
- Resource class: io-large
- Goal: Persist immutable graph snapshots, trace segments, transition evidence, admission receipts, world snapshots, and generation-bearing root manifests.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-013
- Evidence: sawm/graph-history@1, sawm/world-root-history@1
- Evidence requirements JSON: ["sawm/graph-history@1","sawm/world-root-history@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: logical graph history and immutable world-root manifests
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_world_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/semantic_world_store"]
- Interfaces: ProgramGraphHistory@1, SemanticWorldRoot@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Logical cycles and repeated states preserve event history, every reference resolves to prior stored bytes, and root history is immutable.
- Gap tasks: SAWM-013
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G033 Qualify recovery, VFS, replication, and concurrency

- Stable goal ID: SAWM-G033
- Status: active
- Parent: SAWM-G030
- Parent goal IDs JSON: ["SAWM-G030"]
- Depends on: SAWM-G032
- Dependencies JSON: ["SAWM-G032"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: storage-assurance
- Bundle: semantic-addressed-world-model/storage-assurance
- Parallel lane: kit-recovery
- Resource class: io-large
- Goal: Qualify stale-writer rejection, ABA-safe generation CAS, WAL replay, restart, VFS post-commit outbox, corruption detection, optional replication, and concurrent writers.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-014
- Evidence: sawm/root-cas-concurrency@1, sawm/vfs-outbox@1, sawm/recovery@1
- Evidence requirements JSON: ["sawm/root-cas-concurrency@1","sawm/vfs-outbox@1","sawm/recovery@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: recovery, VFS outbox, replication and concurrency adapters/tests
- Predicted files: ipfs_kit_py/ipfs_kit_py/semantic_world_store
- Predicted files JSON: ["ipfs_kit_py/ipfs_kit_py/semantic_world_store"]
- Interfaces: SemanticWorldRootCAS@1, SemanticOutbox@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: One expected generation admits at most one successor; restart retains one valid visible root; file mutation and replication never imply semantic or supervisor acceptance.
- Gap tasks: SAWM-014
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G040 Increase symbolic autonomy and reusable synthesis

- Stable goal ID: SAWM-G040
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G020, SAWM-G030
- Dependencies JSON: ["SAWM-G020","SAWM-G030"]
- Child goal IDs JSON: ["SAWM-G041","SAWM-G042","SAWM-G043"]
- Fib priority: 233
- Priority: P0
- Track: symbolic-autonomy
- Bundle: semantic-addressed-world-model/symbolic-autonomy
- Parallel lane: accelerate-symbolic
- Resource class: cpu-large
- Goal: Run exact reuse, static and abstract analysis, proof search, normalization, deterministic synthesis, and verified procedures before any model residual.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-016 through SAWM-021
- Evidence: sawm/exact-reuse@1, sawm/symbolic-pruning@1, sawm/procedure-promotion@1
- Evidence requirements JSON: ["sawm/exact-reuse@1","sawm/symbolic-pruning@1","sawm/procedure-promotion@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: accelerator reuse gates, symbolic pipeline integrations, procedure extraction and deterministic repair
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/proof, ipfs_accelerate_py/agent_supervisor/autonomous_repair, ipfs_accelerate_py/agent_supervisor/self_improvement
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/analysis","ipfs_accelerate_py/agent_supervisor/proof","ipfs_accelerate_py/agent_supervisor/autonomous_repair","ipfs_accelerate_py/agent_supervisor/self_improvement"]
- Interfaces: ProgramWorldReuseGate@1, ProgramWorldTactician@1, ProgramWorldProcedureBridge@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Every stage records selection/skips; contradictions abstain; no model is called when authoritative deterministic evidence resolves the decision.
- Gap tasks: SAWM-016 through SAWM-021
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G041 Implement exact and proof-backed reuse

- Stable goal ID: SAWM-G041
- Status: active
- Parent: SAWM-G040
- Parent goal IDs JSON: ["SAWM-G040"]
- Depends on: SAWM-G023, SAWM-G033
- Dependencies JSON: ["SAWM-G023","SAWM-G033"]
- Child goal IDs JSON: []
- Fib priority: 144
- Priority: P0
- Track: verified-reuse
- Bundle: semantic-addressed-world-model/verified-reuse
- Parallel lane: accelerate-reuse
- Resource class: cpu-medium
- Goal: Require exact state, goal, policy, environment, toolchain, obligation, procedure, validation, and freshness bindings before reuse, then conservatively generate and prune successors.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-016, SAWM-017
- Evidence: sawm/reuse-admission@1, sawm/static-successors@1
- Evidence requirements JSON: ["sawm/reuse-admission@1","sawm/static-successors@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: reuse gates, static successor generation and symbolic pruning
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/proof, ipfs_accelerate_py/agent_supervisor/autonomous_repair, ipfs_accelerate_py/agent_supervisor/self_improvement
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/analysis","ipfs_accelerate_py/agent_supervisor/proof","ipfs_accelerate_py/agent_supervisor/autonomous_repair","ipfs_accelerate_py/agent_supervisor/self_improvement"]
- Interfaces: ProgramWorldReuseDecision@1, StaticSuccessorPlanner@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Similarity alone is context-only; unknown widens; stale bindings revoke; proof caches still require fresh admission and raw-source fallback remains available.
- Gap tasks: SAWM-016, SAWM-017
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G042 Integrate Tactician, Hammer, e-graphs, and CEGIS

- Stable goal ID: SAWM-G042
- Status: active
- Parent: SAWM-G040
- Parent goal IDs JSON: ["SAWM-G040"]
- Depends on: SAWM-G041
- Dependencies JSON: ["SAWM-G041"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: proof-synthesis
- Bundle: semantic-addressed-world-model/proof-synthesis
- Parallel lane: accelerate-proof
- Resource class: prover-medium
- Goal: Compile finite proof/countermodel obligations, reconstruct or replay natively, normalize soundly, and synthesize bounded repairs from counterexamples.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-018, SAWM-019, SAWM-021
- Evidence: sawm/tactician-hammer@1, sawm/egraph-normalization@1, sawm/cegis-repair@1
- Evidence requirements JSON: ["sawm/tactician-hammer@1","sawm/egraph-normalization@1","sawm/cegis-repair@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: proof pipeline adapters, relation promotion, bounded deterministic repair
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/proof, ipfs_accelerate_py/agent_supervisor/autonomous_repair, ipfs_accelerate_py/agent_supervisor/self_improvement
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/analysis","ipfs_accelerate_py/agent_supervisor/proof","ipfs_accelerate_py/agent_supervisor/autonomous_repair","ipfs_accelerate_py/agent_supervisor/self_improvement"]
- Interfaces: ProgramWorldProofSearch@1, ProgramWorldEGraph@1, ProgramWorldCEGIS@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: No proof or equivalence is model-created; contradictions produce conflict; repair candidates preserve tests, authority paths, scope, effects, and current-tree gates.
- Gap tasks: SAWM-018, SAWM-019, SAWM-021
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G043 Compile accepted trajectories into procedures

- Stable goal ID: SAWM-G043
- Status: active
- Parent: SAWM-G040
- Parent goal IDs JSON: ["SAWM-G040"]
- Depends on: SAWM-G042
- Dependencies JSON: ["SAWM-G042"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: procedural-memory
- Bundle: semantic-addressed-world-model/procedural-memory
- Parallel lane: accelerate-procedure
- Resource class: cpu-large
- Goal: Normalize accepted trajectories, anti-unify repeated families, synthesize bounded procedures, verify preconditions/effects/rollback/holes, and promote through current procedure authority.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-020
- Evidence: sawm/trajectory-normalization@1, sawm/procedure-candidate@1
- Evidence requirements JSON: ["sawm/trajectory-normalization@1","sawm/procedure-candidate@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: procedure compiler bridge and trajectory normalizer
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/proof, ipfs_accelerate_py/agent_supervisor/autonomous_repair, ipfs_accelerate_py/agent_supervisor/self_improvement
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/analysis","ipfs_accelerate_py/agent_supervisor/proof","ipfs_accelerate_py/agent_supervisor/autonomous_repair","ipfs_accelerate_py/agent_supervisor/self_improvement"]
- Interfaces: ProgramWorldProcedureCompiler@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Procedures never self-certify; stale procedures stop; typed holes retain validation; qualifying future tasks route through verified procedures before general models.
- Gap tasks: SAWM-020
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G050 Add calibrated residual transition intelligence

- Stable goal ID: SAWM-G050
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G020, SAWM-G030, SAWM-G040
- Dependencies JSON: ["SAWM-G020","SAWM-G030","SAWM-G040"]
- Child goal IDs JSON: ["SAWM-G051","SAWM-G052","SAWM-G053","SAWM-G054"]
- Fib priority: 377
- Priority: P0
- Track: residual-intelligence
- Bundle: semantic-addressed-world-model/residual-intelligence
- Parallel lane: accelerate-ml
- Resource class: accelerator-optional
- Goal: Build rights-admitted corpora and frozen benchmarks, then qualify the smallest candidate-only specialists for call targets, events, inverse traces, repairs, and graph deltas.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-023 through SAWM-031
- Evidence: sawm/trace-corpus@1, sawm/transition-benchmark@1, sawm/model-calibration@1
- Evidence requirements JSON: ["sawm/trace-corpus@1","sawm/transition-benchmark@1","sawm/model-calibration@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: corpora, baselines, calibrated specialists, experimental graph-sequence models and serving adapters
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus, ipfs_accelerate_py/agent_supervisor/evaluation, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus","ipfs_accelerate_py/agent_supervisor/evaluation","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: ProgramTransitionSpecialist@1, TransitionCalibration@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Ground truth is admitted static/runtime/proof/test/human evidence; OOD abstains; unavailable corpora, checkpoints, or hardware remain typed; complex models promote only on held-out benefit.
- Gap tasks: SAWM-023 through SAWM-031
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G051 Build rights-admitted trace corpora and benchmarks

- Stable goal ID: SAWM-G051
- Status: active
- Parent: SAWM-G050
- Parent goal IDs JSON: ["SAWM-G050"]
- Depends on: SAWM-G023, SAWM-G033
- Dependencies JSON: ["SAWM-G023","SAWM-G033"]
- Child goal IDs JSON: []
- Fib priority: 144
- Priority: P0
- Track: corpus-benchmark
- Bundle: semantic-addressed-world-model/corpus-benchmark
- Parallel lane: corpus
- Resource class: cpu-large
- Goal: Materialize rights/privacy/lineage admitted traces and leakage-safe partitions, then freeze controlled next-call cases and deterministic baselines.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-023, SAWM-024
- Evidence: sawm/corpus-admission@1, sawm/split-leakage@1, sawm/next-call-baseline@1
- Evidence requirements JSON: ["sawm/corpus-admission@1","sawm/split-leakage@1","sawm/next-call-baseline@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: trace corpus contracts/fixtures and frozen next-call benchmark
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus, ipfs_accelerate_py/agent_supervisor/evaluation, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus","ipfs_accelerate_py/agent_supervisor/evaluation","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: ExecutionTraceCorpus@1, NextCallBenchmark@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Private reasoning, credentials, hidden tests, unknown-rights source, and unadmitted traces are absent; all failures/escalations remain in benchmark denominators.
- Gap tasks: SAWM-023, SAWM-024
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G052 Implement next-call edge completion

- Stable goal ID: SAWM-G052
- Status: active
- Parent: SAWM-G050
- Parent goal IDs JSON: ["SAWM-G050"]
- Depends on: SAWM-G041, SAWM-G051
- Dependencies JSON: ["SAWM-G041","SAWM-G051"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: call-ranking
- Bundle: semantic-addressed-world-model/call-ranking
- Parallel lane: call-specialist
- Resource class: accelerator-optional
- Goal: Rank only exact existing static call-target candidates plus explicit unknown, preserving static recall, current-tree resolution, calibration, coverage, and OOD abstention.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-025
- Evidence: sawm/call-ranking@1, sawm/call-calibration@1
- Evidence requirements JSON: ["sawm/call-ranking@1","sawm/call-calibration@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: proposal-only call-target ranking specialist
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus, ipfs_accelerate_py/agent_supervisor/evaluation, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus","ipfs_accelerate_py/agent_supervisor/evaluation","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: CallTargetRanker@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: The ranker cannot erase a static possibility or unresolved frontier; stale/wrong-scope targets fail resolution; top-k, MRR, calibration, and abstention are reproducible.
- Gap tasks: SAWM-025
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G053 Implement next-event, inverse-trace, and repair prediction

- Stable goal ID: SAWM-G053
- Status: active
- Parent: SAWM-G050
- Parent goal IDs JSON: ["SAWM-G050"]
- Depends on: SAWM-G041, SAWM-G051
- Dependencies JSON: ["SAWM-G041","SAWM-G051"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: structured-prediction
- Bundle: semantic-addressed-world-model/structured-prediction
- Parallel lane: transition-specialists
- Resource class: accelerator-optional
- Goal: Predict closed-grammar next events/state deltas, set-valued predecessors, and bounded repair operators/graph deltas without upgrading simulation into observation or authority.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-026, SAWM-027, SAWM-028
- Evidence: sawm/next-event@1, sawm/inverse-trace@1, sawm/repair-prediction@1
- Evidence requirements JSON: ["sawm/next-event@1","sawm/inverse-trace@1","sawm/repair-prediction@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: structured transition, inverse and repair specialists
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus, ipfs_accelerate_py/agent_supervisor/evaluation, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus","ipfs_accelerate_py/agent_supervisor/evaluation","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: NextEventPredictor@1, InverseTracePredictor@1, RepairOperatorRanker@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Observed execution remains ground truth; multiple inverse answers are supported; deterministic procedures precede model repairs; calibration drift invalidates autonomous use.
- Gap tasks: SAWM-026, SAWM-027, SAWM-028
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G054 Evaluate graph-sequence and structured model families

- Stable goal ID: SAWM-G054
- Status: active
- Parent: SAWM-G050
- Parent goal IDs JSON: ["SAWM-G050"]
- Depends on: SAWM-G052, SAWM-G053
- Dependencies JSON: ["SAWM-G052","SAWM-G053"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: model-evaluation
- Bundle: semantic-addressed-world-model/model-evaluation
- Parallel lane: model-evaluation
- Resource class: accelerator-optional
- Goal: Compare deterministic, retrieval, linear, heterogeneous graph, transformer, TAGSeq-style, structured-decoder, and general-code baselines, then admit and serve only economical qualified specialists.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-029, SAWM-030, SAWM-031
- Evidence: sawm/model-ablation@1, sawm/checkpoint-lineage@1, sawm/serving-qualification@1
- Evidence requirements JSON: ["sawm/model-ablation@1","sawm/checkpoint-lineage@1","sawm/serving-qualification@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: graph-sequence experiments, corpus/training/calibration integration and serving adapters
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus, ipfs_accelerate_py/agent_supervisor/evaluation, benchmarks/agent_supervisor/semantic_addressed_world_model
- Predicted files JSON: ["ipfs_datasets_py/ipfs_datasets_py/logic/proof_corpus","ipfs_accelerate_py/agent_supervisor/evaluation","benchmarks/agent_supervisor/semantic_addressed_world_model"]
- Interfaces: GraphSequenceExperiment@1, ResidualProgramWorldFamily@1, ProgramWorldModelServer@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: No external TAGSeq production dependency or mandatory runtime model exists without license/security/reproducibility review and measured held-out benefit over simpler baselines.
- Gap tasks: SAWM-029, SAWM-030, SAWM-031
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G060 Integrate with supervisor planning and context minimization

- Stable goal ID: SAWM-G060
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G040, SAWM-G050
- Dependencies JSON: ["SAWM-G040","SAWM-G050"]
- Child goal IDs JSON: ["SAWM-G061","SAWM-G062","SAWM-G063"]
- Fib priority: 233
- Priority: P0
- Track: supervisor-integration
- Bundle: semantic-addressed-world-model/supervisor-integration
- Parallel lane: accelerate-runtime
- Resource class: coordinator
- Goal: Consume datasets identities and kit retrieval through existing accelerator planning, context, routing, cognitive budgets, rollout, event, distributed execution, and typed control surfaces.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-015, SAWM-022, SAWM-032 through SAWM-039
- Evidence: sawm/context-receipt@1, sawm/cognitive-route@1, sawm/dogfood-rollout@1
- Evidence requirements JSON: ["sawm/context-receipt@1","sawm/cognitive-route@1","sawm/dogfood-rollout@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: operational adapters, context capsules, meta-controller, federation, rollout modes, projections and diagnostics
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state, ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_state","ipfs_accelerate_py/agent_supervisor/context","ipfs_accelerate_py/agent_supervisor/planning","ipfs_accelerate_py/agent_supervisor/runtime","ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state"]
- Interfaces: SemanticWorldOperationalAdapter@1, ProgramWorldContext@1, ProgramWorldDogfoodGate@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: The existing ContextCompiler, decision runtime, provider/resource scheduler, event/outbox, worktree, lease/fence, merge, CLI, and MCP authorities are extended rather than duplicated.
- Gap tasks: SAWM-015, SAWM-022, SAWM-032 through SAWM-039
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G061 Integrate context, memory, routing, and cognitive budgets

- Stable goal ID: SAWM-G061
- Status: active
- Parent: SAWM-G060
- Parent goal IDs JSON: ["SAWM-G060"]
- Depends on: SAWM-G041, SAWM-G054
- Dependencies JSON: ["SAWM-G041","SAWM-G054"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: context-routing
- Bundle: semantic-addressed-world-model/context-routing
- Parallel lane: accelerate-context
- Resource class: cpu-large
- Goal: Assemble smallest-sufficient decision contexts and typed memory views, route the inference cascade under validation reserves, and expose direct typed operations with advisory neural labeling.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-015, SAWM-022, SAWM-032, SAWM-038, SAWM-039
- Evidence: sawm/context-capsule@1, sawm/meta-controller@1, sawm/control-surface@1
- Evidence requirements JSON: ["sawm/context-capsule@1","sawm/meta-controller@1","sawm/control-surface@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: adapters, context integration, meta-controller, multi-view projections, service/CLI/MCP diagnostics
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state, ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_state","ipfs_accelerate_py/agent_supervisor/context","ipfs_accelerate_py/agent_supervisor/planning","ipfs_accelerate_py/agent_supervisor/runtime","ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state"]
- Interfaces: ProgramWorldContextReceipt@1, ProgramWorldMetaController@1, SemanticWorldService@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Every inclusion/omission and escalation is explained; no decision-irrelevant model call occurs; raw source/tests/proofs/policy remain when required; MCP calls typed Python directly.
- Gap tasks: SAWM-015, SAWM-022, SAWM-032, SAWM-038, SAWM-039
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G062 Activate progressive self-hosting and required dogfooding

- Stable goal ID: SAWM-G062
- Status: active
- Parent: SAWM-G060
- Parent goal IDs JSON: ["SAWM-G060"]
- Depends on: SAWM-G061
- Dependencies JSON: ["SAWM-G061"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: dogfood-rollout
- Bundle: semantic-addressed-world-model/dogfood-rollout
- Parallel lane: rollout
- Resource class: coordinator
- Goal: Advance bootstrap to shadow-write, shadow-read, guarded, and required only through closed fail-closed gates and independently admitted evidence.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-034, SAWM-035, SAWM-036, SAWM-037
- Evidence: sawm/shadow-write@1, sawm/shadow-read@1, sawm/guarded-qualification@1, sawm/required-activation@1
- Evidence requirements JSON: ["sawm/shadow-write@1","sawm/shadow-read@1","sawm/guarded-qualification@1","sawm/required-activation@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: closed rollout implementations and completion gates
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state, ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_state","ipfs_accelerate_py/agent_supervisor/context","ipfs_accelerate_py/agent_supervisor/planning","ipfs_accelerate_py/agent_supervisor/runtime","ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state"]
- Interfaces: SemanticWorldRollout@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Mode order cannot skip; neural candidates never suppress source or validation; required dispatch and completion fail closed without independently verified pre/post roots and receipts.
- Gap tasks: SAWM-034, SAWM-035, SAWM-036, SAWM-037
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G063 Integrate causal federation and distributed execution

- Stable goal ID: SAWM-G063
- Status: active
- Parent: SAWM-G060
- Parent goal IDs JSON: ["SAWM-G060"]
- Depends on: SAWM-G061
- Dependencies JSON: ["SAWM-G061"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: federation
- Bundle: semantic-addressed-world-model/federation
- Parallel lane: event-federation
- Resource class: distributed-coordinator
- Goal: Publish semantic, proof, retrieval, trace, transition, and invalidation updates through current event/outbox and causal-frontier authorities, waking only affected supervisors.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-033
- Evidence: sawm/causal-event-federation@1
- Evidence requirements JSON: ["sawm/causal-event-federation@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: causal federation adapter and affected-supervisor tests
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state, ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/semantic_state","ipfs_accelerate_py/agent_supervisor/context","ipfs_accelerate_py/agent_supervisor/planning","ipfs_accelerate_py/agent_supervisor/runtime","ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/semantic_state"]
- Interfaces: ProgramWorldCausalFederation@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: No second event plane, state owner, or full-scan loop is introduced; fenced, causal, idempotent updates survive restart and stale workers cannot publish.
- Gap tasks: SAWM-033
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G070 Qualify safety, efficiency, self-hosting, and release

- Stable goal ID: SAWM-G070
- Status: active
- Parent: SAWM-G000
- Parent goal IDs JSON: ["SAWM-G000"]
- Depends on: SAWM-G060
- Dependencies JSON: ["SAWM-G060"]
- Child goal IDs JSON: ["SAWM-G071","SAWM-G072"]
- Fib priority: 144
- Priority: P0
- Track: qualification-release
- Bundle: semantic-addressed-world-model/qualification-release
- Parallel lane: release
- Resource class: coordinator
- Goal: Run controlled end-to-end, ablation, adversarial, and required-mode self-hosted qualification, then publish exact current-tree release, migration, rollback, blocker, limitation, and root evidence.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-040 through SAWM-044
- Evidence: sawm/e2e-acceptance@1, sawm/efficiency-ablation@1, sawm/adversarial-assurance@1, sawm/self-host-capstone@1, sawm/release-report@1
- Evidence requirements JSON: ["sawm/e2e-acceptance@1","sawm/efficiency-ablation@1","sawm/adversarial-assurance@1","sawm/self-host-capstone@1","sawm/release-report@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: acceptance matrix, benchmarks, adversarial evidence, capstone and release reports
- Predicted files: test/api/semantic_world, benchmarks/agent_supervisor/semantic_addressed_world_model, docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md
- Predicted files JSON: ["test/api/semantic_world","benchmarks/agent_supervisor/semantic_addressed_world_model","docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md"]
- Interfaces: SemanticWorldReleaseQualification@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: False completions, unauthorized mutations, similarity-only reuse, stale reuse, simulated-as-live admission, model-created authority/proof/completion, hidden validation reduction, unsafe abstraction, and root overwrite conflicts are all zero.
- Gap tasks: SAWM-040 through SAWM-044
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G071 Run adversarial, corruption, recovery, and security gates

- Stable goal ID: SAWM-G071
- Status: active
- Parent: SAWM-G070
- Parent goal IDs JSON: ["SAWM-G070"]
- Depends on: SAWM-G033, SAWM-G062, SAWM-G063
- Dependencies JSON: ["SAWM-G033","SAWM-G062","SAWM-G063"]
- Child goal IDs JSON: []
- Fib priority: 89
- Priority: P0
- Track: assurance
- Bundle: semantic-addressed-world-model/assurance
- Parallel lane: assurance
- Resource class: security-review
- Goal: Exercise the full acceptance scenario and negative variants for identity, graph, retrieval, trace, prediction, repair, procedure, privacy, import, storage, VFS, concurrency, recovery, provider, and authority failures.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-040, SAWM-042
- Evidence: sawm/e2e-negative-matrix@1, sawm/security-qualification@1
- Evidence requirements JSON: ["sawm/e2e-negative-matrix@1","sawm/security-qualification@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: end-to-end and adversarial test evidence
- Predicted files: test/api/semantic_world, benchmarks/agent_supervisor/semantic_addressed_world_model, docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md
- Predicted files JSON: ["test/api/semantic_world","benchmarks/agent_supervisor/semantic_addressed_world_model","docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md"]
- Interfaces: SemanticWorldAdversarialCampaign@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: All seeded critical defects are detected; corrupt/stale/forged/poisoned/cancelled/private/model-created evidence fails closed and cannot enter a root.
- Gap tasks: SAWM-040, SAWM-042
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## SAWM-G072 Benchmark, perform capstone, and publish release evidence

- Stable goal ID: SAWM-G072
- Status: active
- Parent: SAWM-G070
- Parent goal IDs JSON: ["SAWM-G070"]
- Depends on: SAWM-G071
- Dependencies JSON: ["SAWM-G071"]
- Child goal IDs JSON: []
- Fib priority: 55
- Priority: P0
- Track: benchmark-capstone-release
- Bundle: semantic-addressed-world-model/benchmark-capstone-release
- Parallel lane: release
- Resource class: coordinator
- Goal: Run frozen paired ablations with honest denominators, self-host one bounded enhancement in required mode, and publish machine/human release evidence with final transitive root verification.
- Refinement: Descendants and producing tasks own disjoint implementation slices; shared exports, registries, gitlinks, controls, schemas, CLI registration, and release aggregation have explicit integration owners.
- Producing tasks: SAWM-041, SAWM-043, SAWM-044
- Evidence: sawm/paired-ablation@1, sawm/required-capstone@1, sawm/final-world-root@1
- Evidence requirements JSON: ["sawm/paired-ablation@1","sawm/required-capstone@1","sawm/final-world-root@1"]
- Evidence criteria: Current-tree source, tests, schemas, environment, policy, authority binding, and exact immutable receipts must all agree; a title, similar class name, historical branch, fixture, report, projection, prediction, or Markdown status is insufficient.
- Outputs: benchmark report, capstone transition/root evidence, release and migration report
- Predicted files: test/api/semantic_world, benchmarks/agent_supervisor/semantic_addressed_world_model, docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md
- Predicted files JSON: ["test/api/semantic_world","benchmarks/agent_supervisor/semantic_addressed_world_model","docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md"]
- Interfaces: SemanticWorldBenchmark@1, RequiredModeCapstone@1
- Validation: PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. /home/barberb/.local/bin/python scripts/validate_semantic_addressed_world_model_board.py
- Acceptance: Efficiency misses block promotion rather than weakening safety; the top-level session does not implement/complete the capstone; release is withheld unless every mandatory terminal and final root is verified.
- Gap tasks: SAWM-041, SAWM-043, SAWM-044
- Rollout constraint: bootstrap -> shadow_write -> shadow_read -> guarded -> required; no mode may be skipped and required evidence is independently admitted.
- Authority constraint: Datasets owns semantic/formal meaning; kit owns verified storage/retrieval/durable roots; accelerate owns operational planning, execution, routing, and acceptance.
- Conflict policy: Unknown write overlap serializes. No descendant may modify SAWM-000 controls, weaken an existing authority, duplicate a landed subsystem, or accept its own result.

## Goal reconciliation invariants

- The supplied hierarchy contains one root and every named descendant shown above.
- Task status is not goal status; accepted task receipts are inputs to independent goal reconciliation.
- A goal remains active while any mandatory child evidence is missing, stale, blocked without an allowed terminal, or not bound to the current source/environment/policy.
- Required-mode evidence includes verified pre/post roots, context, reuse, prediction-consideration, proof/test/validation, transition, generation, and independent admission receipts.
- The final root must resolve and verify transitively; all release safety floors remain zero.
