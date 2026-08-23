# Logic-Governed Compositional Verification Fabric Task Board

Executable projection of the LGCVF objective heap. Task prefix: `## LGCVF-`.
Board namespace: `logic-governed-compositional-verification-fabric-v1`.

This is plan revision 2. Its immediate immutable predecessor is LGCVF revision
1, `baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q`;
the original LGSWF ancestor remains
`sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e`.
It does not rewrite either predecessor, the LGSWF board, or any manual seal.
Automatic work must not edit this board, its plan/objectives, its validator,
scheduler policy, benchmark/qualification judges, or predecessor evidence.

Construction statuses record the evidence available when this revision was
created. `completed` means task implementation evidence exists; it does not
mean its parent objective, release, or production is complete. The operational
DuckDB repository owns all later task transitions.

## LGCVF-001 Establish exact current-tree and capability truth

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: audit
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G010
- Depends on:
- Owning repository: ipfs_accelerate_py
- Outputs: docs/architecture/logic_governed_compositional_verification/current_tree_capability_gap_matrix.json, docs/architecture/logic_governed_compositional_verification/CURRENT_TREE_CAPABILITY_GAP_MATRIX.md, docs/architecture/logic_governed_compositional_verification/TRUST_AND_LIMITATIONS.md
- Predicted files: docs/architecture/logic_governed_compositional_verification/current_tree_capability_gap_matrix.json, docs/architecture/logic_governed_compositional_verification/CURRENT_TREE_CAPABILITY_GAP_MATRIX.md, docs/architecture/logic_governed_compositional_verification/TRUST_AND_LIMITATIONS.md
- Validation: python -m json.tool docs/architecture/logic_governed_compositional_verification/current_tree_capability_gap_matrix.json
- Acceptance: Exact superproject/datasets HEADs and trees, gitlink topology, bounded overlays, tools, capabilities, board roots, observed gaps, owners, assurance, risks, benchmarks, and acceptance evidence are recorded without treating package presence as live capability.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: audit
- Conflict policy: Read all required surfaces; write audit artifacts only; preserve user work and foreign gitlinks.
- Required evidence: topology and revision receipt, capability probes, JSON/Markdown gap-matrix agreement

## LGCVF-002 Materialize the immutable successor plan projections

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: plan-control
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G010
- Depends on: LGCVF-001
- Owning repository: ipfs_accelerate_py
- Outputs: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md, docs/architecture/logic_governed_compositional_verification_fabric.objectives.md, docs/architecture/logic_governed_compositional_verification_fabric.todo.md, data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json
- Predicted files: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md, docs/architecture/logic_governed_compositional_verification_fabric.objectives.md, docs/architecture/logic_governed_compositional_verification_fabric.todo.md, data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json
- Validation: python scripts/validate_logic_governed_compositional_verification_fabric_plan.py --check-all
- Acceptance: Human, objective, daemon, and FormalWorkPlan projections agree on 13 sufficient subgoals, 27 tasks, dependencies, namespace, ancestry, authority boundary, and non-release state.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: plan-control
- Conflict policy: Operator-owned successor control files; never outputs of later automatic tasks.
- Required evidence: validator receipt and FormalWorkPlan content identity

## LGCVF-010 Extend the canonical datasets compositional-contract IR

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: contracts
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G020
- Depends on: LGCVF-002
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_contracts/compositional.py, tests/unit/logic/software_contracts/test_compositional_contract.py
- Predicted files: ipfs_datasets_py/logic/software_contracts/compositional.py, tests/unit/logic/software_contracts/test_compositional_contract.py
- Validation: python -m pytest -q tests/unit/logic/software_contracts/test_compositional_contract.py
- Acceptance: Canonical typed contracts cover required identities, formula references, effects, interference, exceptional behavior, limitations, invalidators, and evidence; v1 records remain readable and unknown semantics remain opaque.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-contracts
- Conflict policy: Datasets semantic authority only; no duplicate contract registry, formula language, or receipt hierarchy.
- Required evidence: deterministic round trip, compatibility and rejection tests

## LGCVF-020 Implement the minimal conservative Python abstract interpreter

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: abstract-interpretation
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G030
- Depends on: LGCVF-010
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/abstract_interpretation.py, tests/unit/logic/software_verification/test_abstract_interpretation.py
- Predicted files: ipfs_datasets_py/logic/software_verification/abstract_interpretation.py, tests/unit/logic/software_verification/test_abstract_interpretation.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_abstract_interpretation.py
- Acceptance: The product domain has lattice/fixpoint/widening/narrowing behavior, interprocedural summaries, exceptions/effects, source provenance, budgets, and conservative opaque handling for unsupported dynamic constructs.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-abstract
- Conflict policy: Extend existing analysis strategy/verification IR; do not introduce a second semantic index or graph.
- Required evidence: lattice, monotonicity, convergence, exception, summary and opaque-fallback tests

## LGCVF-030 Implement typed assume-guarantee discharge

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: discharge
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G040
- Depends on: LGCVF-010, LGCVF-020
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/assume_guarantee.py, tests/unit/logic/software_verification/test_assume_guarantee.py
- Predicted files: ipfs_datasets_py/logic/software_verification/assume_guarantee.py, tests/unit/logic/software_verification/test_assume_guarantee.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_assume_guarantee.py
- Acceptance: Exact component edges generate assumption, guarantee, invariant, exceptional, effect, and interference obligations; failing consumers localize counterexamples; SCCs require independent inductive closure or reject.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-discharge
- Conflict policy: Reuse existing contracts, concurrency IR, obligations, graphs, and solver evidence.
- Required evidence: success, weak/missing assumption, interference, exception, stale-root and cycle tests

## LGCVF-040 Implement exact incremental invalidation and reuse planning

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: incremental-state
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G050
- Depends on: LGCVF-020, LGCVF-030
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/incremental_verification.py, tests/unit/logic/software_verification/test_incremental_verification.py
- Predicted files: ipfs_datasets_py/logic/software_verification/incremental_verification.py, tests/unit/logic/software_verification/test_incremental_verification.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_incremental_verification.py
- Acceptance: Changed identities, direct/reverse/contract/SCC closure, invalidated states/sessions/capsules, minimal checks, dynamic frontier, and reused evidence are derived from exact bindings; stale keys reject without assurance upgrade.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-invalidation
- Conflict policy: Join existing index/state/cache/snapshot APIs; large bodies stay content-addressed.
- Required evidence: delta, closure, stale-key, consumer invalidation and unaffected-reuse tests

## LGCVF-050 Implement provider-neutral incremental SMT sessions

- Status: completed
- Completion: auto
- Is schedulable: false
- Review only: false
- Priority: P0
- Track: incremental-smt
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G060
- Depends on: LGCVF-030, LGCVF-040
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/backends/smt/incremental.py, tests/unit/logic/backends/test_incremental_smt.py
- Predicted files: ipfs_datasets_py/logic/backends/smt/incremental.py, tests/unit/logic/backends/test_incremental_smt.py
- Validation: python -m pytest -q tests/unit/logic/backends/test_incremental_smt.py
- Acceptance: Stable named assertions, push/pop/assumptions, model/core/statistics, cancellation/close, typed outcomes, exact fingerprints, and replay manifests work without treating session reuse as proof authority.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-smt
- Conflict policy: Optional providers lazy-load; no network or installation; unsupported capabilities remain typed.
- Required evidence: session, core mapping, timeout/cancel, crash/replay, stale/toolchain and unsupported tests

## LGCVF-051 Complete public API adapters and bounded solver differentials

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: incremental-smt-api
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G060
- Depends on: LGCVF-050
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/verification_api.py, tests/unit/logic/test_compositional_verification_public_api.py
- Predicted files: ipfs_datasets_py/logic/verification_api.py, tests/unit/logic/test_compositional_verification_public_api.py
- Validation: python -m pytest -q tests/unit/logic/test_compositional_verification_public_api.py
- Acceptance: Existing verification API exposes checked thin adapters for abstract analysis, contracts, discharge, invalidation, sessions, and interpolation; bounded common-fragment Z3/CVC5 results agree or return a typed discrepancy.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-public-api
- Conflict policy: Adapter-only public surface; no transport semantics or duplicate implementation.
- Required evidence: API round trip, cold-import and differential receipts

## LGCVF-060 Qualify and independently validate Craig interpolation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: interpolation
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G070
- Depends on: LGCVF-050
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/backends/smt/interpolation.py, tests/unit/logic/backends/test_interpolation.py
- Predicted files: ipfs_datasets_py/logic/backends/smt/interpolation.py, tests/unit/logic/backends/test_interpolation.py
- Validation: python -m pytest -q tests/unit/logic/backends/test_interpolation.py
- Acceptance: Exact provider/theory support is probed; admitted interpolants pass A-implies-I, I-and-B-unsat, shared-vocabulary, identity, and bounds checks; unavailable providers produce typed fallback authority.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-interpolation
- Conflict policy: Solver availability is not interpolation support; never fabricate an interpolant.
- Required evidence: valid, invalid vocabulary/implication, unavailable and core-fallback tests

## LGCVF-061 Implement the bounded interpolation/core-driven CEGAR loop

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: cegar
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G070
- Depends on: LGCVF-060
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/cegar.py, tests/unit/logic/software_verification/test_cegar.py
- Predicted files: ipfs_datasets_py/logic/software_verification/cegar.py, tests/unit/logic/software_verification/test_cegar.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_cegar.py
- Acceptance: Spurious traces refine with validated interpolants/cores/reviewed predicates, real traces remain counterexamples, and every run terminates proved/disproved/unknown/timeout/unavailable/budget-exhausted.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-cegar
- Conflict policy: Extend existing counterexample-guided services; no parallel model checker.
- Required evidence: spurious/real trace preservation and budget tests

## LGCVF-070 Emit stage-addressed translation-validation receipts

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: translation-receipts
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G080
- Depends on: LGCVF-040, LGCVF-050
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/formalization/translation_receipts.py, tests/unit/logic/formalization/test_translation_receipts.py
- Predicted files: ipfs_datasets_py/logic/formalization/translation_receipts.py, tests/unit/logic/formalization/test_translation_receipts.py
- Validation: python -m pytest -q tests/unit/logic/formalization/test_translation_receipts.py
- Acceptance: Every compilation stage binds input/output/compiler/source maps/subset/losses/assumptions/obligations/validation/replay/bounds/evidence class; unsupported losses cap downstream authority.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-translation
- Conflict policy: Extend existing formalization/translations/backends and generic receipt primitives.
- Required evidence: source-map, unsupported/loss, replay, stale-proof and reconstruction tests

## LGCVF-071 Slice and replay unchanged translation and proof obligations

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: obligation-slicing
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G080
- Depends on: LGCVF-070
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/obligation_slicing.py, tests/unit/logic/software_verification/test_obligation_slicing.py
- Predicted files: ipfs_datasets_py/logic/software_verification/obligation_slicing.py, tests/unit/logic/software_verification/test_obligation_slicing.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_obligation_slicing.py
- Acceptance: Local mutations invalidate exactly affected translation stages, theorem dependencies, and obligations while independently revalidating reusable evidence.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-obligation-slicing
- Conflict policy: Reuse proof/cache/dependency identities; no second proof cache.
- Required evidence: theorem-granularity invalidation and unchanged-stage replay tests

## LGCVF-080 Audit and harden the existing equality-saturation path

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: egraph
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G090
- Depends on: LGCVF-030, LGCVF-050
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, test/api/test_agent_supervisor_program_repair_egraph.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, test/api/test_agent_supervisor_program_repair_egraph.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_repair_egraph.py
- Acceptance: Existing equality mode has typed e-classes, congruence/rebuild, reviewed side conditions/provenance, bounded saturation, extraction cost/replay, and independent equivalence/effect checks, or records each unavailable feature truthfully.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-egraph
- Conflict policy: Extend ProgramRepairSynthesizer@1; no second e-graph/synthesizer.
- Required evidence: congruence, side-condition, budget, extraction, replay and invalid-rewrite tests

## LGCVF-081 Integrate counterevidence-refined bounded synthesis

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: synthesis
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G090
- Depends on: LGCVF-060, LGCVF-080
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py, ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, test/api/test_agent_supervisor_program_repair_cegis.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py, ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, test/api/test_agent_supervisor_program_repair_cegis.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_program_repair_cegis.py
- Acceptance: Reviewed grammars/operators use counterexamples, cores, failed assumptions, and validated interpolants to refine search; no candidate adds undeclared imports, dependencies, files, authority, effects, or behavior.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-synthesis
- Conflict policy: Model residual syntax remains proposal-only and separately named.
- Required evidence: CEGIS refinement, effect/security restriction and zero-model tests

## LGCVF-090 Build and independently verify proof-carrying artifacts

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: proof-artifact
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G100
- Depends on: LGCVF-061, LGCVF-071, LGCVF-081
- Owning repository: ipfs_datasets_py
- Outputs: ipfs_datasets_py/logic/software_verification/proof_carrying_artifact.py, tests/unit/logic/software_verification/test_proof_carrying_artifact.py
- Predicted files: ipfs_datasets_py/logic/software_verification/proof_carrying_artifact.py, tests/unit/logic/software_verification/test_proof_carrying_artifact.py
- Validation: python -m pytest -q tests/unit/logic/software_verification/test_proof_carrying_artifact.py
- Acceptance: Bundle references exact semantic, contract, abstract, proof/test/static/security/policy/authority lineage; verifier rebuilds identities and compact checks and rejects producer flags, forged CIDs, stale roots, and missing mandatory evidence.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: datasets-proof-artifact
- Conflict policy: Reuse an existing equivalent bundle/receipt/CID authority when found.
- Required evidence: independent replay, forged/stale/omission rejection tests

## LGCVF-091 Compile mandatory-coverage proof-carrying context

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: proof-context
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G100
- Depends on: LGCVF-090
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py, test/api/test_agent_supervisor_proof_carrying_context.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py, test/api/test_agent_supervisor_proof_carrying_context.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_proof_carrying_context.py
- Acceptance: Context optimization minimizes cost subject to complete affected interfaces, open assumptions/obligations, policy, allowed effects and validation; proof handles compress satisfied evidence without exposing secrets or dropping critical source.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-proof-context
- Conflict policy: Extend current ContextPack/bundle optimizer; no model router or context compiler duplicate.
- Required evidence: exact/conservative/opaque capsule, stale, omission, dynamic, handle and injection tests

## LGCVF-100 Integrate semantic discharge into Planner/Doctor fixed points

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: planner-doctor
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G110
- Depends on: LGCVF-090, LGCVF-091
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py, test/api/test_agent_supervisor_lgcvf_planner_doctor.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py, test/api/test_agent_supervisor_lgcvf_planner_doctor.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_lgcvf_planner_doctor.py
- Acceptance: Plan admission/completion, impact, selected checks, repair and fixed-point validation consume current discharge/invalidation evidence; unsat cores/counterexamples/interpolants create minimal successors; missing coverage blocks.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-planner-doctor
- Conflict policy: Existing Planner/Doctor is the sole supervisor; preserve immutable plan ancestry.
- Required evidence: admission/completion/fixed-point/second-order/oscillation tests

## LGCVF-101 Persist typed operational references and restart state

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: persistence
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G110
- Depends on: LGCVF-100
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py, test/api/test_agent_supervisor_lgcvf_persistence.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py, test/api/test_agent_supervisor_lgcvf_persistence.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_lgcvf_persistence.py
- Acceptance: Typed repositories persist append-only/CID references with CAS, leases, fences, operation IDs, outbox cursors and restart reconciliation; single-writer enforcement remains truthful without qualified Quack.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-persistence
- Conflict policy: No agent SQL, new DuckDB abstraction, or operational fields in datasets SemanticStateRoot.
- Required evidence: restart, stale-worker, duplicate completion, fence, single-writer and outbox tests

## LGCVF-102 Expose one semantic service through Python, CLI, and MCP

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: api-projections
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G110
- Depends on: LGCVF-100
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_lgcvf_transport_parity.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_lgcvf_transport_parity.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_lgcvf_transport_parity.py
- Acceptance: Capability/snapshot/impact/contracts/abstract/discharge/verify/prove/counterexample/interpolate/synthesize/repair/context/benchmark/explain/replay operations share one typed service; mutation defaults to preview.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: accelerator-api
- Conflict policy: Wrappers perform no independent semantics and create no MCP++ profile.
- Required evidence: Python/CLI/MCP parity and preview-no-write tests

## LGCVF-110 Execute the complete hermetic Python vertical slice

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vertical-slice
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G120
- Depends on: LGCVF-051, LGCVF-061, LGCVF-071, LGCVF-081, LGCVF-101, LGCVF-102
- Owning repository: ipfs_accelerate_py
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/compositional_verification_vertical.py, test/fixtures/agent_supervisor/compositional_verification, test/api/test_agent_supervisor_compositional_verification_vertical.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/compositional_verification_vertical.py, test/fixtures/agent_supervisor/compositional_verification, test/api/test_agent_supervisor_compositional_verification_vertical.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_compositional_verification_vertical.py
- Acceptance: Public APIs execute all 22 required stages with a real isolated mutation/repair, current discharge and fixed point, unaffected evidence reuse, independently verified final artifact, final context, zero model calls/imports, and token/reuse metrics.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: vertical-driver
- Conflict policy: Fixture evidence is hermetic-development only; candidate cannot edit validation/oracles.
- Required evidence: machine trace, receipt identities, exact rollback and model-call/import counters

## LGCVF-111 Complete focused unit/property/differential/metamorphic tests

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: focused-tests
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G120
- Depends on: LGCVF-110
- Owning repository: ipfs_accelerate_py
- Outputs: test/api/test_agent_supervisor_lgcvf_focused_qualification.py
- Predicted files: test/api/test_agent_supervisor_lgcvf_focused_qualification.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_lgcvf_focused_qualification.py
- Acceptance: Every minimum abstract/discharge/SMT/interpolation/compilation/synthesis/capsule/context/supervisor requirement has a non-skipped executable test or a typed unavailable outcome that does not count as pass.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: qualification-focused
- Conflict policy: The candidate test is a deliverable, not its own completion authority; LGCVF-113 re-executes it with protected independent suites and never weakens current policies, thresholds or fixtures.
- Required evidence: test manifest and exact pass/fail/typed-unavailable counts

## LGCVF-112 Complete adversarial authority and rollback qualification

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adversarial-tests
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G120
- Depends on: LGCVF-110
- Owning repository: ipfs_accelerate_py
- Outputs: test/api/test_agent_supervisor_lgcvf_adversarial.py
- Predicted files: test/api/test_agent_supervisor_lgcvf_adversarial.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_lgcvf_adversarial.py
- Acceptance: Forgery, staleness, prompt injection, judge mutation, gitlink drift, lease/fence mismatch, duplicate completion, unchanged residual, oscillation, second-order findings, real-byte mutation, and exact rollback fail closed.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: qualification-adversarial
- Conflict policy: The candidate may implement this declared test deliverable, but cannot author the protected LGCVF-113 qualification judge or any protected control input; this test cannot certify itself.
- Required evidence: adversarial detection and rollback receipts

## LGCVF-113 Independently qualify the focused and adversarial test deliverables

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: independent-qualification
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G120
- Depends on: LGCVF-111, LGCVF-112
- Owning repository: ipfs_accelerate_py
- Outputs: data/agent_supervisor/logic_governed_compositional_verification_fabric/independent_qualification_result.json
- Predicted files: data/agent_supervisor/logic_governed_compositional_verification_fabric/independent_qualification_result.json
- Validation: python scripts/qualify_logic_governed_compositional_verification_fabric.py --check
- Acceptance: A pre-existing protected verifier re-executes both candidate suites plus a fixed content-bound manifest of semantic, proof, supervisor, authority, rollback, and fixed-point suites; it rejects skips, typed unavailable counted as pass, missing judges, drifted manifests, and any production claim, then emits a hermetic-only receipt.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: qualification-independent
- Conflict policy: Candidate work may emit only the result artifact; it cannot edit the protected verifier, manifest, test authorities, scheduler, plan, benchmark oracle, or acceptance policy.
- Required evidence: exact command/source identities, pass/fail/skip/unavailable counts, artifact CID, cohort label, and completion_authority=false outside hermetic qualification

## LGCVF-120 Run the preregistered paired hermetic benchmark

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: benchmark
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G130
- Depends on: LGCVF-111, LGCVF-112, LGCVF-113
- Owning repository: ipfs_accelerate_py
- Outputs: scripts/benchmark_lgcvf_symbolic_displacement.py, data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json
- Predicted files: scripts/benchmark_lgcvf_symbolic_displacement.py, data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json
- Validation: python scripts/benchmark_lgcvf_symbolic_displacement.py --check
- Acceptance: Baseline/challenger use identical roots, policies, seeds, budgets and independent oracles across required task classes; quality, omissions, reuse, context/models, resources, time/cost and rollback metrics are measured without hard-coded success, and the protected check rejects completion unless the current LGCVF-113 qualification receipt reconstructs.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: benchmark-hermetic
- Conflict policy: Benchmark manifest/oracles are protected; cohorts remain separate.
- Required evidence: paired machine result, cohort labels, preregistered threshold disposition

## LGCVF-121 Qualify live external/provider evidence

- Status: blocked
- Completion: external-authority
- Is schedulable: false
- Review only: true
- Blocked reason: blocked_external_authority; protected verifier/provider configuration and production-authoritative cohort are unavailable in this hermetic run
- Priority: P0
- Track: external-qualification
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G130
- Depends on: LGCVF-120
- Owning repository: ipfs_accelerate_py
- Outputs: data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json
- Predicted files: data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json
- Validation: python -m json.tool data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json
- Acceptance: Only an independently authorized verifier can bind live local/remote/production evidence, provider disclosure policy, and qualified multi-writer capability; typed unavailable is retained until then.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: external-authority
- Conflict policy: Never install, network-probe, forge, or self-author external qualification.
- Required evidence: independent qualification identity and scope

## LGCVF-122 Issue the evidence-based release disposition

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-disposition
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G130
- Depends on: LGCVF-120
- Owning repository: ipfs_accelerate_py
- Outputs: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md
- Predicted files: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md
- Validation: python scripts/validate_logic_governed_compositional_verification_fabric_closeout.py release --check
- Acceptance: Report says go/partial/no-go from exact cohort evidence, identifies external/manual blockers, and distinguishes implementation, tests, objective state, release qualification and production authorization.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: release-report
- Conflict policy: May report no-go with typed blockers; cannot authorize release or production.
- Required evidence: benchmark root, test roots, blocker list and threshold comparison

## LGCVF-123 Operator production authorization

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: blocked_manual; independent human/operator authorization has not been issued
- Priority: P0
- Track: production-authorization
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G130
- Depends on: LGCVF-121, LGCVF-122
- Owning repository: ipfs_accelerate_py
- Outputs: data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json
- Predicted files: data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json
- Validation: python -m json.tool data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json
- Acceptance: Only an exact current independently authorized manual receipt may transition this task; no model, fixture, test, CID, task state, or supervisor run can self-authorize it.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: operator-only
- Conflict policy: Operator-only; never automatically complete or fabricate, and never reuse LGSWF-006 as this authority.
- Required evidence: current manual authority, scope, expiry and exact plan/source roots

## LGCVF-124 Publish the final implementation report and minimal successors

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: final-report
- Goal id: LGCVF-G000
- Parent goal ID: LGCVF-G000
- Subgoal ID: LGCVF-G130
- Depends on: LGCVF-120, LGCVF-122
- Owning repository: ipfs_accelerate_py
- Outputs: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md, data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_tasks.json
- Predicted files: docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md, data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_tasks.json
- Validation: python scripts/validate_logic_governed_compositional_verification_fabric_closeout.py implementation --check
- Acceptance: Report includes exact revisions/topology, reused capabilities, verified gaps, authority decisions, changed files by repository, APIs, exact tests, vertical receipt identities, benchmarks, model/context displacement, risks/blockers, and machine-executable successors, with five completion/qualification states separated.
- Board namespace: logic-governed-compositional-verification-fabric-v1
- Parallel lane: final-report
- Conflict policy: Append evidence-based successors; do not rewrite accepted plan/task history or claim blocked production authority.
- Required evidence: report section validator and successor-plan content identity
