# Current-tree capability and gap matrix

This is the human-readable projection of
`current_tree_capability_gap_matrix.json`. The JSON file is the machine-readable
record; row identifiers and status values are identical here.

## Source truth and interpretation

- Audit checkpoint: 2026-08-18 UTC.
- Accelerator worktree: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-lgswf-actual-head`.
- Accelerator baseline: commit `12c4e8387de4986d38d69534f3d74864e7bb15c1`, tree
  `96504e7744d741784bc8076456f4b169dd665ab5`.
- `ipfs_datasets_py` topology: Git submodule/gitlink, not a subtree or a
  physically flattened repository.
- Datasets gitlink/nested commit:
  `480a1666f144ad606fcb3cacb66e59775f28d0d1`; nested tree:
  `6fdc81fabb04a86683d0f26200636fa8f61fd25c`.
- A bounded uncommitted datasets overlay contains candidate P1-P5 work and a
  locally qualified interpolation adapter. It is not part of either recorded
  tree, not an accepted revision, and not production authority.
- The live LGSWF `run-actual-v6` database reports 47 of 47 tasks completed.
  The root objective and 16 goals remain open. The accepted qualification is
  only `research_demo`. Continuous Quack multi-writer operation and live
  DuckLake remain no-go. Task implementation, test success, objective
  completion, release qualification, and production authorization are
  independent predicates.
- The durable predecessor plan is
  `sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e`.
  This audit does not alter that plan, its completed history, its accepted
  receipts, or its quarantined attempts.

The only status values used are `implemented_and_integrated`,
`implemented_not_wired`, `partial`, `declared_only`, `stale`,
`unavailable_optional`, `unavailable_required`, `missing`, `superseded`,
`blocked_manual`, and `blocked_external_authority`.

Datasets owns semantic meaning: program facts, contracts, abstract states,
formal artifacts, proof obligations, semantic state/capsules, evidence, and
confidence. Accelerator owns operational planning, routing, leases, fences,
worktrees, task/objective state, model invocation, benchmarking, rollback, and
operational DuckDB records.

## Status overview

| ID | Capability | Declared | Observed | Owner |
|---|---|---|---|---|
| P0-TOPOLOGY | Exact repository topology and recursive source identity | implemented_and_integrated | implemented_and_integrated | Git; accelerator admission |
| P0-PLAN-TRUTH | Durable LGSWF plan/task/objective reconciliation | implemented_and_integrated | partial | accelerator |
| P0-SEMANTIC-STATE | Semantic index/state and exact invalidation base | implemented_and_integrated | implemented_and_integrated | datasets |
| P0-SCANNER-REFERENCES | Scanner contract/proof reference population | partial | partial | datasets |
| P0-CAPABILITY-DISCOVERY | Solver/prover/interpolation capability truth | implemented_and_integrated | partial | datasets/accelerator |
| P0-EXTERNAL-EVIDENCE | Protected external-evidence verification | implemented_not_wired | unavailable_required | accelerator/external authority |
| P0-QUACK | Continuous Quack multi-writer control plane | blocked_external_authority | unavailable_required | accelerator |
| P0-DUCKLAKE | DuckLake analytics/history projection | unavailable_optional | unavailable_optional | datasets projection only |
| P1-CONTRACT-KERNEL | Canonical compositional contract kernel | partial | implemented_not_wired | datasets |
| P2-ABSTRACT-INTERPRETATION | Conservative Python abstract interpretation | declared_only | implemented_not_wired | datasets |
| P3-ASSUME-GUARANTEE | Assume-guarantee graph/discharge | declared_only | implemented_not_wired | datasets |
| P4-INCREMENTAL-VERIFICATION | Incremental semantic/verification state | partial | implemented_not_wired | datasets/accelerator |
| P5-INCREMENTAL-SMT | Reusable incremental SMT | declared_only | implemented_not_wired | datasets |
| P6-INTERPOLATION-CEGAR | Craig interpolation and CEGAR | declared_only | partial | datasets/accelerator |
| P7-PROOF-PRODUCING-COMPILATION | Staged proof/translation validation | partial | partial | datasets |
| P8-EGRAPH-SYNTHESIS | E-graphs and bounded synthesis | implemented_and_integrated | partial | accelerator proposals; datasets validation |
| P9-PROOF-ENGINEERING | Tactician/Hammer/cache/kernel integration | implemented_and_integrated | partial | datasets/accelerator |
| P10-PROOF-CARRYING | Proof-carrying code and context | partial | partial | datasets/accelerator |
| P11-CAPSULES | Capsules as checkable summaries | implemented_and_integrated | partial | datasets |
| P12-SUPERVISOR | Planner/Doctor self-supervision | implemented_and_integrated | partial | accelerator |
| P13-MODEL-MINIMIZATION | Constrained context/model minimization | implemented_and_integrated | partial | accelerator |
| PERSISTENCE-TYPED-REPOSITORIES | Typed DuckDB repositories | implemented_and_integrated | partial | accelerator |
| PUBLIC-PYTHON-API | Additive Python semantic API | partial | implemented_not_wired | datasets |
| PUBLIC-CLI-MCP | Shared CLI/MCP projections | missing | missing | transport only |
| VERTICAL-SLICE | Hermetic Python repair slice | missing | missing | datasets/accelerator |
| PAIRED-BENCHMARK | Baseline/challenger evaluation | partial | partial | accelerator/oracle |
| RELEASE-AUTHORIZATION | Independent release/production disposition | blocked_external_authority | blocked_external_authority | human/operator/external authority |

## Detailed rows

### P0-TOPOLOGY — exact repository topology and recursive source identity

- Existing source path: `.gitmodules`, `ipfs_datasets_py`, and
  `ipfs_accelerate_py/agent_supervisor/runtime/configured_board.py`.
- Existing public interface: Git HEAD/tree/gitlink identity and configured-board
  repository/submodule preflight.
- Current tests: configured-board schema and submodule-drift tests.
- Declared status: `implemented_and_integrated`.
- Observed status: `implemented_and_integrated`.
- Authority owner: Git repositories; accelerator validates operational
  admission.
- Assurance level: exact content/topology identity, not semantic correctness.
- Installed/available/conformant/reconstructable: yes/yes/yes/yes. The exact
  accelerator and nested datasets identities were read directly.
- Missing integration: the P1-P5 overlay is not a datasets commit and the
  accelerator gitlink has not been intentionally advanced.
- Proposed extension: validate and commit datasets first, then update only the
  accelerator gitlink in a clean intentional revision.
- Dependency: none.
- Risk: history flattening, stale gitlink admission, or unrelated dirty work.
- Benchmark: reconstruct HEAD, tree, gitlink and bounded overlays before and
  after each mutation.
- Acceptance evidence: exact revisions, mode-160000 gitlink, bounded overlay
  manifest and drift rejection.

### P0-PLAN-TRUTH — durable LGSWF reconciliation

- Existing source path: the LGSWF plan/taskboard and `run-actual-v6` data.
- Existing public interface: `DatabaseTaskSource`, objective repositories and
  the configured-board scheduler.
- Current tests: configured-board, materialization, objective and completion
  admission suites plus live read-only reconciliation.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: accelerator operational control plane.
- Assurance level: durable operational evidence, `research_demo` only.
- Installed/available/conformant/reconstructable: yes/yes/yes/yes. The database
  reports 47/47 tasks while the root objective and 16 goals remain open.
- Missing integration: the exhausted board neither closes the objective nor
  covers the newly verified compositional gaps.
- Proposed extension: create an immutable LGCVF successor bound to predecessor
  CID and receipts, preserving every historical status.
- Dependency: P0-TOPOLOGY.
- Risk: equating task count with objective completion or rewriting history.
- Benchmark: independently reconcile task, goal and objective predicates on
  one current root.
- Acceptance evidence: predecessor CID/ancestry, row-by-row revision rationale,
  47/47 count and open-objective report.

### P0-SEMANTIC-STATE — semantic index/state and invalidation base

- Existing source path: datasets `software_contracts/semantic_index` and
  `software_contracts/semantic_state`.
- Existing public interface: `scan_repository`, `diff_repository_states`,
  `calculate_invalidation`, `build_semantic_state`, and
  `verify_semantic_state`.
- Current tests: datasets semantic-index and semantic-state unit suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `implemented_and_integrated`.
- Authority owner: datasets.
- Assurance level: exact or conservative according to scanner confidence and
  validity bindings.
- Installed/available/conformant/reconstructable: yes/yes/yes/yes. Real Python
  scanning, deltas, Merkle state, persistence and invalidation are tested.
- Missing integration: the base frontier lacks some contract, abstract-state,
  session and proof-obligation edges.
- Proposed extension: add typed selectors/adapters; do not create another
  graph, index or semantic root.
- Dependency: P0-TOPOLOGY.
- Risk: treating a conservative/dynamic frontier as exact.
- Benchmark: seeded impact precision/recall and unaffected-evidence reuse.
- Acceptance evidence: fresh root verification, exact delta, conservative
  dynamic frontier, reverse closure and re-admitted cache keys.

### P0-SCANNER-REFERENCES — scanner contract/proof population

- Existing source path: semantic-index `scanner.py` and `python_analysis.py`.
- Existing public interface: `scan_repository`.
- Current tests: scanner and Python-analysis tests.
- Declared status: `partial`.
- Observed status: `partial`.
- Authority owner: datasets.
- Assurance level: exact syntax/selected relations, incomplete contract/proof
  coverage.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: production scans do not comprehensively populate
  contract, obligation, assumption/guarantee and proof-dependency references.
- Proposed extension: populate typed versioned references and opaque markers
  for unsupported semantics.
- Dependency: P0-SEMANTIC-STATE and P1-CONTRACT-KERNEL.
- Risk: false-minimal frontiers and unsafe capsule/proof reuse.
- Benchmark: mutation-corpus edge recall and critical-omission detection.
- Acceptance evidence: real scan records plus adversarial omission rejection.

### P0-CAPABILITY-DISCOVERY — solver/prover/interpolation truth

- Existing source path: datasets backends and verification API plus accelerator
  analysis strategy registry.
- Existing public interface: `list_providers`, `provider_capabilities`, and
  `probe_provider`.
- Current tests: verification API and backend capability tests.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: datasets capability semantics; accelerator routing.
- Assurance level: observed local capability, not production authorization.
- Installed/available/conformant/reconstructable: yes/yes/no/yes. Python 3.12.3,
  Z3 4.15.4, cvc5 1.3.3, Lean 4.33 and DuckDB 1.5.2 were observed. cvc5
  interpolation is locally qualified and independently checked, but not
  production-qualified.
- Missing integration: no exact session/interpolation negotiation by theory,
  API and provider version.
- Proposed extension: cold exact probes with typed unavailable/unsupported
  outcomes; do not install or use the network.
- Dependency: P0-TOPOLOGY.
- Risk: treating package presence or generic SMT support as qualification.
- Benchmark: hermetic probe and replay under exact fingerprints.
- Acceptance evidence: provider-specific conformance receipts and fail-closed
  unavailable results.

### P0-EXTERNAL-EVIDENCE — protected external verification

- Existing source path: accelerator `core/external_completion.py` and
  `planning/proof_carrying_planner.py`.
- Existing public interface: external-completion and proof-carrying admission.
- Current tests: corresponding admission unit suites.
- Declared status: `implemented_not_wired`.
- Observed status: `unavailable_required`.
- Authority owner: accelerator policy and independent external authority.
- Assurance level: fail closed without verifier identity/evidence.
- Installed/available/conformant/reconstructable: yes/no/no/no.
- Missing integration: no current independently authorized verifier binding.
- Proposed extension: prepare replayable candidate evidence and retain the
  block until separately qualified.
- Dependency: P9-PROOF-ENGINEERING and P10-PROOF-CARRYING.
- Risk: self-authored evidence being accepted as independent.
- Benchmark: forged, stale, self-signed and wrong-root rejection.
- Acceptance evidence: independent identity, policy/root binding, replay and
  authorized disposition.

### P0-QUACK — continuous multi-writer control plane

- Existing source path: accelerator runtime and the LGSWF qualification report.
- Existing public interface: `StateRepository`, `QuackStateRepository`,
  `QuackStateClient`, and `QuackStateServer`.
- Current tests: capability, repository, canary, chaos and release suites.
- Declared status: `blocked_external_authority`.
- Observed status: `unavailable_required`.
- Authority owner: accelerator.
- Assurance level: loopback research observations only; production no-go.
- Installed/available/conformant/reconstructable: yes/no/no/no.
- Missing integration: no qualified exclusive state-owner chain for continuous
  multi-process mutation.
- Proposed extension: retain exactly one embedded DuckDB writer until a
  separate Quack qualification succeeds.
- Dependency: P0-PLAN-TRUTH.
- Risk: split brain and unsafe direct multi-process DuckDB writes.
- Benchmark: fencing/failover, duplicate acceptance, remote readiness and
  direct-file-bypass suites.
- Acceptance evidence: exact profile and independent multi-writer release
  chain; otherwise unavailable.

### P0-DUCKLAKE — analytics/history projection

- Existing source path: LGSWF qualification report and datasets projection
  modules.
- Existing public interface: typed capability/projection contracts.
- Current tests: contract and hermetic projection tests.
- Declared status: `unavailable_optional`.
- Observed status: `unavailable_optional`.
- Authority owner: datasets projection implementation; never semantic or
  operational authority.
- Assurance level: not started for the qualified run.
- Installed/available/conformant/reconstructable: no/no/no/no.
- Missing integration: no live catalog, security, recovery or release receipt.
- Proposed extension: optional read-only projection after control-plane
  admission.
- Dependency: P0-QUACK.
- Risk: projection state being treated as scheduling/semantic authority.
- Benchmark: parity, replay, outage recovery and zero authority crossover.
- Acceptance evidence: a separate live-projection release receipt.

### P1-CONTRACT-KERNEL — canonical compositional contracts

- Existing source path: datasets software contracts, concurrency IR and the
  overlay `software_contracts/compositional.py`.
- Existing public interface: `CallableContract`, `ProgramContract`,
  `RelyGuaranteeContract`, `CompositionalContract@1`, and
  `compile_component_contract`.
- Current tests: contract suite and overlay compositional-contract tests.
- Declared status: `partial`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets.
- Assurance level: typed/content-addressed; legacy prose remains opaque.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; overlay only.
- Missing integration: no accepted revision, production population, capsule
  projection or accelerator completion gate.
- Proposed extension: validate and commit datasets, then wire references
  through adapters.
- Dependency: P0-SEMANTIC-STATE.
- Risk: prose lowered to true or a duplicate contract authority.
- Benchmark: v1 compatibility, closed fields, opaque semantics and identity.
- Acceptance evidence: focused tests and independent lowering on an accepted
  revision.

### P2-ABSTRACT-INTERPRETATION — conservative Python analysis

- Existing source path: overlay
  `software_verification/abstract_interpretation.py`.
- Existing public interface: `AbstractDomain`,
  `PythonAbstractInterpreter@1`, and `analyze_abstract_state`.
- Current tests: overlay abstract-interpretation unit/property cases.
- Declared status: `declared_only`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets.
- Assurance level: exact for supported transfers; conservative/opaque
  otherwise.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; overlay only.
- Missing integration: no accepted revision, production scan, VC lowering or
  supervisor admission.
- Proposed extension: land the minimal product domain and add reviewed domains
  through its registry.
- Dependency: P1-CONTRACT-KERNEL.
- Risk: unsound transfers or ignored dynamic Python behavior.
- Benchmark: lattice laws, monotonicity, convergence, exceptions,
  interprocedural summaries and opaque cases.
- Acceptance evidence: property/adversarial results, source/analyzer identity,
  convergence and dynamic frontier.

### P3-ASSUME-GUARANTEE — compositional discharge

- Existing source path: datasets concurrency IR and overlay
  `software_verification/assume_guarantee.py`.
- Existing public interface: `RelyGuaranteeContract`,
  `ComponentCompositionGraph`, and `discharge_assume_guarantee`.
- Current tests: overlay assume-guarantee suite.
- Declared status: `declared_only`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets.
- Assurance level: solver-checked bounded supported formulas; typed
  counterexample/unknown otherwise.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; overlay only.
- Missing integration: no obligation graph, plan admission, capsule,
  fixed-point or completion wiring.
- Proposed extension: admit datasets receipts through accelerator adapters and
  block on uncovered assumptions.
- Dependency: P1-CONTRACT-KERNEL and P5-INCREMENTAL-SMT.
- Risk: circular citation, weak guarantees, stale roots or string semantics.
- Benchmark: success, weak/missing assumptions, interference, exceptions,
  cycle checks, stale roots and localization.
- Acceptance evidence: replayable closed receipt with roots, graph,
  obligations, solver evidence, residuals and bounds.

### P4-INCREMENTAL-VERIFICATION — exact invalidation and reuse

- Existing source path: datasets semantic index, overlay
  `incremental_verification.py`, and accelerator reasoning snapshot.
- Existing public interface: `calculate_invalidation`,
  `plan_incremental_verification`, and
  `build_repository_reasoning_snapshot`.
- Current tests: semantic-index and overlay incremental-verification suites.
- Declared status: `partial`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets owns semantic invalidation; accelerator scheduling
  and references.
- Assurance level: exact key re-admission plus conservative closure.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; overlay not
  wired.
- Missing integration: no operational persistence across all state/session/
  capsule/proof/test decisions.
- Proposed extension: connect the typed plan to existing snapshot, planner and
  repositories, storing bodies by existing CIDs.
- Dependency: P0-SCANNER-REFERENCES and P1-P3.
- Risk: stale reuse, cache assurance upgrade or path-only session reuse.
- Benchmark: impact precision/recall and unaffected-obligation reuse.
- Acceptance evidence: exact delta/closures, reused/invalidated receipts,
  canonical keys and dynamic frontier.

### P5-INCREMENTAL-SMT — reusable sessions

- Existing source path: datasets SMT backends and overlay `smt/incremental.py`.
- Existing public interface: `IncrementalSmtSession`,
  `open_incremental_smt_session`, push/pop, assumption checking, unsat cores,
  and replay manifests.
- Current tests: overlay incremental-SMT suite.
- Declared status: `declared_only`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets solver-evidence authority.
- Assurance level: solver-checked; reuse is memoization only.
- Installed/available/conformant/reconstructable: yes/yes/no/yes. Lazy Z3
  4.15.4 named assertions/cores/replay pass focused local tests.
- Missing integration: no accepted facade, persistent replay repository,
  differential session suite or production resource qualification.
- Proposed extension: land the provider-neutral contract and bind replay
  identities to P4.
- Dependency: P0-CAPABILITY-DISCOVERY.
- Risk: reuse across changed semantics, invalid core/model or resource escape.
- Benchmark: push/pop, assumptions, core mapping, cancellation, replay, stale
  fingerprints and bounded differential checks.
- Acceptance evidence: session fingerprint, assertion/source map,
  independently validated evidence and replay manifest.

### P6-INTERPOLATION-CEGAR — interpolation and refinement

- Existing source path: overlay `smt/interpolation.py` and accelerator
  counterexample-guided tactician.
- Existing public interface: `compute_and_validate_interpolant` and
  `CounterexampleGuidedTactician`.
- Current tests: overlay interpolation and tactician tests.
- Declared status: `declared_only`.
- Observed status: `partial`.
- Authority owner: datasets interpolation evidence; accelerator orchestration.
- Assurance level: locally solver-checked interpolants; incomplete CEGAR.
- Installed/available/conformant/reconstructable: yes/yes/no/yes. cvc5 1.3.3
  is locally qualified for bounded QF_LIA, with fresh Z3 checks; this is not
  production qualification.
- Missing integration: no accepted theory matrix or complete CEGAR loop.
- Proposed extension: land validation, typed fallbacks and bounded CEGAR
  outcomes.
- Dependency: P2 and P5.
- Risk: fabricated interpolants, illegal vocabulary or unsupported theories.
- Benchmark: implication/unsatisfiability/vocabulary checks and spurious/real
  counterexample budgets.
- Acceptance evidence: exact partition/provider identity, independent checks,
  restrictions and typed fallback.

### P7-PROOF-PRODUCING-COMPILATION — staged validation

- Existing source path: datasets formalization, translations, backends and
  verification API.
- Existing public interface: `compile_verification_artifact` and translation/
  provider receipts.
- Current tests: formalization, translation, backend and API suites.
- Declared status: `partial`.
- Observed status: `partial`.
- Authority owner: datasets.
- Assurance level: translation validation and solver/kernel evidence only
  where explicitly present.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: no complete source-map/loss/preservation chain or
  obligation slicing across all stages.
- Proposed extension: version existing receipts by stage and slice unchanged
  obligations without claiming a verified compiler.
- Dependency: P1, P4 and P5.
- Risk: downstream authority exceeding partial upstream translation.
- Benchmark: source maps, unsupported syntax, losses, stale proofs and
  differential backends.
- Acceptance evidence: content-addressed stage chain with independent
  validators and explicit losses.

### P8-EGRAPH-SYNTHESIS — reviewed rewrite and repair search

- Existing source path: accelerator `program_repair_synthesis.py` and
  `repair_operator_registry.py`.
- Existing public interface: `ProgramRepairSynthesizer@1`,
  `DeclaredEqualityTheory`, registry and CEGIS mode.
- Current tests: repair synthesis and operator-registry suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: accelerator proposes; datasets validates semantics.
- Assurance level: proposal-only bounded search.
- Installed/available/conformant/reconstructable: yes/yes/no/yes. Current terms
  are opaque strings with bounded rewriting, not true typed e-classes and
  congruence closure.
- Missing integration: typed e-classes/rebuild, semantic side conditions,
  extraction costs and contract-driven refinement.
- Proposed extension: extend only the existing synthesizer/registry.
- Dependency: P3, P6 and P7.
- Risk: effect-changing/invalid rewrites receiving authority.
- Benchmark: congruence, side conditions, saturation, replay, effect rejection
  and CEGIS refinement.
- Acceptance evidence: reviewed rules, bounded extraction and independent
  contract/proof/security/test checks.

### P9-PROOF-ENGINEERING — proof search, cache and kernels

- Existing source path: accelerator proof modules plus datasets hammers and
  proof repository.
- Existing public interface: tactician, proof-directed retrieval,
  `CanonicalProofCacheKey` and provider/kernel adapters.
- Current tests: proof/tactician, hammer, repository and kernel suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: datasets theorem meaning; accelerator search/routing.
- Assurance level: candidate, solver and kernel classes stay distinct.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: no closed theorem-granularity dependency/replay path.
- Proposed extension: wire exact statement/import/toolchain identities while
  preserving evidence classes.
- Dependency: P4 and P7.
- Risk: candidate text or learned ranking treated as theorem truth.
- Benchmark: replay, minimization, invalidation, stale environments and
  cross-backend checks.
- Acceptance evidence: kernel receipt where available; otherwise an exact
  lower class and limitations.

### P10-PROOF-CARRYING — artifact and context bundles

- Existing source path: accelerator proof-carrying planner, semantic context
  pack and Planner/Doctor context.
- Existing public interface: proof-carrying plan artifacts, `ContextPack` and
  `compile_minimal_context`.
- Current tests: proof-carrying planner and context tests.
- Declared status: `partial`.
- Observed status: `partial`.
- Authority owner: datasets semantic/proof references; accelerator bundle and
  disclosure policy.
- Assurance level: content-addressed references plus independent checks where
  implemented.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: a full independent PCC verifier and full recursive
  gitlink/state/contract/replay lineage.
- Proposed extension: version an existing equivalent; do not add a generic
  receipt hierarchy.
- Dependency: P3, P7 and P9.
- Risk: producer pass fields, self-hashes or CIDs treated as correctness.
- Benchmark: forged roots, stale assumptions, omitted obligations,
  unauthorized disclosure and replay.
- Acceptance evidence: independent reconstruction and typed disposition.

### P11-CAPSULES — checkable semantic summaries

- Existing source path: datasets semantic-state capsules and accelerator
  capsule admission.
- Existing public interface: `SemanticCapsule`, compilation and admission.
- Current tests: datasets capsule and accelerator admission suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: datasets; accelerator consumes references.
- Assurance level: exact/conservative/heuristic/opaque remain distinct.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: general summary-sufficiency and consumer-strength
  discharge.
- Proposed extension: add typed references/obligations to the existing capsule.
- Dependency: P1, P2, P3 and P6.
- Risk: heuristic or opaque summary replacing critical raw source.
- Benchmark: substitution classes, caveats, freshness, dynamic expansion and
  omission cases.
- Acceptance evidence: fresh bindings and discharged sufficiency/consumer
  obligations, with raw fallback.

### P12-SUPERVISOR — symbolic self-supervision

- Existing source path: accelerator planning/runtime/objectives and
  `deterministic_doctor_live_fixed_point.py`.
- Existing public interface: goal obligation compilation, formal replanning,
  fixed-point validation and configured-board scheduling.
- Current tests: planner/Doctor, objective/task, lease/fence, worktree and
  rollback suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: accelerator.
- Assurance level: deterministic policy plus independently reproduced evidence
  where wired.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: P1-P6 receipts are not completion blockers throughout
  the public path.
- Proposed extension: consume datasets receipts at existing seams; retain one
  supervisor and one task authority.
- Dependency: P3, P4, P10 and P11.
- Risk: task count/command success replacing obligation closure.
- Benchmark: second-order findings, oscillation, residual termination,
  rollback, stale leases and protected judges.
- Acceptance evidence: current-root closure, zero-model deterministic receipt,
  independent checks and fixed-point trace.

### P13-MODEL-MINIMIZATION — constrained context and routing

- Existing source path: Planner/Doctor context, proof retrieval, semantic
  governor and semantic-state benchmark.
- Existing public interface: `compile_minimal_context`, proof retrieval and
  route/benchmark APIs.
- Current tests: context, retrieval, governor and benchmark suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: accelerator.
- Assurance level: deterministic selection separated from heuristic ranking.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: the optimizer does not yet cover all P1-P11 evidence,
  assumptions and dynamic frontier constraints.
- Proposed extension: extend existing bundle optimization with typed hard
  constraints and escalation reasons.
- Dependency: P4, P10 and P11.
- Risk: critical omission, unauthorized disclosure or size-only escalation.
- Benchmark: paired token/call/tier/closure/expansion/omission metrics.
- Acceptance evidence: all mandatory obligations represented and exclusions
  justified with exact accounting.

### PERSISTENCE-TYPED-REPOSITORIES — operational records

- Existing source path: accelerator runtime, task sources and integrations.
- Existing public interface: `DatabaseTaskSource`, typed repositories,
  lease/fence/CAS/outbox APIs.
- Current tests: restart, reconciliation, single-writer, fence and outbox suites.
- Declared status: `implemented_and_integrated`.
- Observed status: `partial`.
- Authority owner: accelerator.
- Assurance level: transactional records referencing immutable datasets
  artifacts.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; embedded
  one-writer mode is the truthful current mode.
- Missing integration: repositories for all new edge/state/session/core/
  interpolant/fixed-point references.
- Proposed extension: version existing repositories, keep large bodies in the
  existing object store, and forbid agent raw SQL.
- Dependency: P4 and P12.
- Risk: duplicate authority, body duplication or unsafe writers.
- Benchmark: restart, duplicate operation, stale fence, outbox and one-writer
  enforcement.
- Acceptance evidence: typed repository tests and CAS/lease/fence/replay
  receipts.

### PUBLIC-PYTHON-API — additive semantic facade

- Existing source path: datasets `logic/verification_api.py`.
- Existing public interface: `LogicVerificationAPI`, `STABLE_OPERATIONS`, and
  overlay `COMPOSITIONAL_VERIFICATION_OPERATIONS`.
- Current tests: verification API and focused P1-P6 suites.
- Declared status: `partial`.
- Observed status: `implemented_not_wired`.
- Authority owner: datasets.
- Assurance level: typed domain artifacts; legacy operation parity preserved.
- Installed/available/conformant/reconstructable: yes/yes/no/yes; overlay only.
- Missing integration: no accepted revision or accelerator/CLI/MCP callers.
- Proposed extension: land lazy additive wrappers without altering the legacy
  stable operation set.
- Dependency: P1-P6.
- Risk: wrapper-local semantics or broken MCP parity.
- Benchmark: import safety, operation discovery, compatibility and zero
  provider import on deterministic paths.
- Acceptance evidence: facade tests and canonical typed results from real
  implementations.

### PUBLIC-CLI-MCP — transport projections

- Existing source path: datasets logic CLI/MCP server and accelerator
  integrations.
- Existing public interface: logic CLI, `LogicVerificationMCP`, integrations.
- Current tests: CLI and MCP parity/conformance suites.
- Declared status: `missing`.
- Observed status: `missing`.
- Authority owner: transport only.
- Assurance level: not implemented for the new operations.
- Installed/available/conformant/reconstructable: yes/no/no/yes.
- Missing integration: requested compositional command family.
- Proposed extension: project the same accepted Python operations; mutations
  default to preview.
- Dependency: PUBLIC-PYTHON-API and P12.
- Risk: transport-local semantics or write authority.
- Benchmark: Python/CLI/MCP identity parity and preview-default tests.
- Acceptance evidence: identical canonical results and no transport-local
  discharge logic.

### VERTICAL-SLICE — hermetic Python repair flow

- Existing source path: accelerator validation and datasets fixtures.
- Existing public interface: current semantic state/capsule APIs plus planned
  additive compositional API.
- Current tests: no full 22-step integration test at this audit checkpoint.
- Declared status: `missing`.
- Observed status: `missing`.
- Authority owner: datasets evidence; accelerator isolated execution.
- Assurance level: not established.
- Installed/available/conformant/reconstructable: no/no/no/yes.
- Missing integration: fixture, mutation, invalidation/reuse, isolated repair,
  fixed point, artifact, context and zero-model receipt.
- Proposed extension: execute the smallest complete fixture through public
  APIs and existing repair/Doctor seams.
- Dependency: P1-P5, P10 and P12.
- Risk: prebuilt passing receipts, self-validation or production extrapolation.
- Benchmark: all required 22 steps, exact reuse, zero model calls and context
  displacement.
- Acceptance evidence: reproducible command, real isolated byte change,
  independent checks and hermetic trace identities.

### PAIRED-BENCHMARK — baseline versus challenger

- Existing source path: Planner/Doctor live benchmark and semantic-state
  benchmark.
- Existing public interface: `PlannerDoctorLiveBenchmark` and `run_benchmark`.
- Current tests: existing benchmark suites.
- Declared status: `partial`.
- Observed status: `partial`.
- Authority owner: accelerator orchestration and an independent oracle.
- Assurance level: existing simulated/hermetic measurements, not an LGCVF
  production cohort.
- Installed/available/conformant/reconstructable: yes/yes/no/yes.
- Missing integration: identical-corpus LGCVF baseline/challenger result across
  all requested task classes.
- Proposed extension: extend existing harnesses and keep evidence cohorts
  separate.
- Dependency: VERTICAL-SLICE and P13.
- Risk: hard-coded targets, workload drift, fixture-only production claims or
  silent skips.
- Benchmark: all requested safety, correctness, reuse, context/model, resource,
  cost and rollback metrics.
- Acceptance evidence: machine-readable paired result, comparability manifest,
  no required skips and truthful no-go/partial outcomes.

### RELEASE-AUTHORIZATION — independent disposition

- Existing source path: LGSWF qualification and external-completion admission.
- Existing public interface: release evidence and external receipt admission.
- Current tests: release, authority and adversarial suites.
- Declared status: `blocked_external_authority`.
- Observed status: `blocked_external_authority`.
- Authority owner: independent human/operator and configured external authority.
- Assurance level: `research_demo` only.
- Installed/available/conformant/reconstructable: yes/no/no/no.
- Missing integration: no independently reviewed LGCVF release manifest or
  production authorization.
- Proposed extension: prepare candidate evidence while leaving all external
  gates blocked.
- Dependency: PAIRED-BENCHMARK and P0-EXTERNAL-EVIDENCE.
- Risk: conflating implementation, tests, objective completion, release and
  production authorization.
- Benchmark: independent release-manifest replay and every safety floor.
- Acceptance evidence: a separate authorized disposition. Never fabricate
  LGSWF-006 or any operator-only seal.
