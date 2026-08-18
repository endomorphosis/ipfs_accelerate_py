# Logic-Governed Compositional Verification Fabric

Status: active research-and-development successor; not release-qualified and
not production-authorized.

Program namespace:
`logic-governed-compositional-verification-fabric-v1` (`LGCVF-`).
Predecessor: the immutable LGSWF plan
`sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e`.
This document adds a successor revision. It does not amend the protected LGSWF
board, its completed records, quarantined attempts, or manual seals.

## 1. Purpose and completion boundary

LGCVF closes the verified gaps between datasets-owned software semantics and
accelerator-owned orchestration. The target is one content-addressed,
incremental, deterministic-first loop that derives the smallest affected
semantic/verification frontier, reuses fresh evidence, discharges
compositional contracts, performs bounded deterministic repairs, and sends
only an irreducible residual to a model.

The root objective is not complete when implementation commands or tests pass.
It requires current semantic roots, independently reproduced obligation
evidence, adversarial tests, a paired benchmark, and explicit dispositions for
all mandatory unknowns. Release qualification and production authorization are
separate later decisions; neither is implied by this plan or its fixture.

## 2. Construction source truth

The plan was constructed against these immutable starting identities:

| Authority | Topology | HEAD | Tree |
| --- | --- | --- | --- |
| accelerator | superproject | `12c4e8387de4986d38d69534f3d74864e7bb15c1` | `96504e7744d741784bc8076456f4b169dd665ab5` |
| datasets | initialized Git submodule/gitlink | `480a1666f144ad606fcb3cacb66e59775f28d0d1` | `6fdc81fabb04a86683d0f26200636fa8f61fd25c` |

The datasets checkout is not flattened. Datasets changes must be validated and
committed in that repository before an intentional accelerator gitlink update.
The pre-existing dirty `mcplusplus` and `ipfs_kit_py` gitlinks are outside this
program and must not enter its commits.

Observed construction-time assumptions are conservative:

- the persisted LGSWF run has 47/47 tasks completed, including independently
  accepted manual tasks, while its broader objective and production state are
  not thereby complete;
- Z3 and CVC5 are locally available, but capability presence is not production
  conformance or proof authority;
- CVC5 interpolation is locally usable on a bounded supported fragment and an
  interpolant still requires independent implication, unsatisfiability, and
  vocabulary checks;
- Quack is installed but not loaded/qualified in the current profile, so the
  operational control plane remains one explicit DuckDB writer;
- model/provider and protected external-evidence authority are unavailable for
  this hermetic development run; no package is installed and no network probe
  is used to change that fact.

Every later execution must re-observe exact heads, recursive gitlinks, dirty
overlays, configuration, toolchains, policies, and capability receipts. A
construction baseline is not a freshness waiver.

## 3. Authority and trust invariants

1. `ipfs_datasets_py.logic` owns AST/symbol facts, relationships, contracts,
   abstract states, formulas, proof obligations, translation/prover evidence,
   semantic capsules, confidence classes, and semantic identities.
2. `ipfs_accelerate_py.agent_supervisor` owns goals/tasks, planning, resource
   and provider routing, worktrees, leases, fences, mutation permits,
   operational DuckDB state, validation, rollback, benchmarking, and model
   invocation.
3. Existing semantic-state, capsule, graph, proof-cache, plan, model-router,
   receipt, CID, object-store, DuckDB, and MCP++ authorities are extended
   through checked adapters; no parallel `v2` authority is permitted.
4. Models, candidates, fixtures, tests, receipts, hashes, CIDs, and task states
   cannot certify themselves. Independent replay establishes only its declared
   evidence class.
5. Unknown, stale, unavailable, unsupported, opaque, truncated, or ambiguous
   mandatory evidence blocks, broadens, or escalates. It never becomes `true`.
6. Candidate, syntax-checked, bounded/model-checked, solver-checked,
   kernel-verified, runtime-observed, and cryptographically attested evidence
   remain distinct.
7. Deterministic mode imports and invokes no model, embedding, remote-provider,
   or network client. Escalation follows exact reuse, static/symbolic analysis,
   abstract interpretation, contract discharge, reviewed synthesis, then
   small/medium/frontier model and human review.
8. Mutation requires current roots, exact allowed paths/effects, permit, lease,
   fence, isolated worktree or VFS overlay, checkpoint, independent
   postconditions, and exact rollback.
9. Operational state never enters a datasets semantic root. Accelerator records
   typed references to immutable datasets evidence.
10. LGCVF-123 and any equivalent operator-only decision remain
    `blocked_manual` until an authorized person acts. No LGSWF seal is
    regenerated or inferred.

## 4. Goal hierarchy and dependency spine

`LGCVF-G000` is refined into the following sufficient subgoals. A sufficient
refinement is a child-to-parent obligation, not an assertion that the child is
equivalent to the root goal.

```text
LGCVF-G000  Deterministic-first compositional verification control loop
├── G010 P0  current-tree truth and immutable plan reconciliation
├── G020 P1  canonical typed compositional contracts
├── G030 P2  conservative abstract interpretation
├── G040 P3  assume-guarantee composition
├── G050 P4  exact incremental semantic/verification invalidation
├── G060 P5  reusable incremental SMT
├── G070 P6  validated interpolation and CEGAR
├── G080 P7  translation receipts and obligation slicing
├── G090 P8  reviewed equality saturation and synthesis
├── G100 P9  proof-carrying artifacts and contexts
├── G110 P10 Planner/Doctor integration and deterministic routing
├── G120 P11 complete Python vertical slice and adversarial qualification
└── G130 P12 paired benchmark, release evidence, and successors
```

The normative dependencies live in the `FormalWorkPlan@1` artifact and the
task board. The broad order is P0→P1→P2→P3→P4→P5, after which P6, P7, and P8
can proceed in independent conflict-safe lanes. P9 joins their evidence; P10
integrates it operationally; P11 and P12 close the development evidence.

## 5. Technical program

### P0 — observe, classify, and reconcile

Record exact repository and recursive gitlink identities, bounded dirty
overlays, Python/tool versions, solver/prover capability states, task/objective
roots, protected-plan state, and live-vs-declared implementation evidence.
Publish the closed-status gap matrix in JSON and Markdown. Retain prior PDR,
SCG, LFV, LGSWF, and PGIR outcomes and explain successor ancestry without
rewriting history.

### P1 — canonical compositional contract kernel

Extend datasets' existing software-contract IR through a checked schema and v1
adapter. A contract binds component and source identities, configuration and
toolchain roots, typed pre/normal/exceptional postconditions, invariants,
assumptions, guarantees, rely/guarantee relations, effects, footprints,
interference, progress/fairness, policy obligations, invalidators, limitations,
and evidence references. Prose stays annotation-only. Unsupported legacy
semantics adapt to explicit opaque references, never tautologies.

### P2 — conservative abstract interpretation

Provide the reviewed domain protocol, bottom/top/order/equality/join/meet,
widening/narrowing, worklist fixed point, bounded path/context sensitivity,
exceptional flow, interprocedural summaries, provenance, and source maps. The
first product is constants × intervals × nullness × exception state × effects.
Reflection, dynamic import, monkey patching, native extensions, runtime code
generation, opaque callbacks, and uncontrolled I/O widen or force fallback.

### P3 — assume-guarantee discharge

Build exact component graphs and generate obligations that neighbors establish
assumptions, each component establishes its guarantees, consumers receive
sufficient guarantees, system invariants survive composition, interference is
compatible, and exceptional/resource behavior is covered. SCCs require an
explicit induction hypothesis and independent closure check or are rejected.
Discharge receipts bind roots, edges, obligations, evidence, counterexamples,
bounds, replay data, and uncovered assumptions.

### P4 — incremental semantic and verification protocol

Join the existing datasets semantic index/state with accelerator snapshots,
planning, caches, and typed operational repositories. Track imports, calls,
types, flow, control, aliasing, effects, exceptions, contracts,
assumption/guarantee edges, invariants, configuration, schemas, build inputs,
tests, proofs, policies, interfaces, concurrency, resources, and transport
projections. A mutation yields exact changed identities, reverse/SCC/contract
closure, invalidated states/sessions/capsules, selected checks, dynamic
frontier, and reused unaffected evidence. Cache keys bind every semantic,
environment, policy, toolchain, abstraction, and authority input.

### P5 — reusable incremental SMT

Expose provider-neutral sessions with stable declarations/assertions, named
source mappings, push/pop, assumptions, model/core/proof/statistics access,
cancellation, close, and replay manifests. Reuse requires exact declaration,
theory, translation, toolchain, policy, and assumption fingerprints. Results
are typed SAT/UNSAT/unknown/timeout/unsupported/unavailable; models and cores
are validated and session reuse never raises assurance.

### P6 — interpolation and CEGAR

Negotiate exact provider/theory support. Validate every interpolant by checking
`A ⇒ I`, `I ∧ B` unsatisfiable, shared vocabulary, provider/version, theory,
bounds, and partition identities. When unavailable, report and use validated
unsat-core, proof-tree, weakest-precondition, or reviewed-predicate refinement.
CEGAR terminates under fixed budgets with one typed disposition.

### P7 — proof-producing or translation-validating compilation

Address source→AST→normalized AST→CFG→SSA/data flow→contract/effect
IR→VC→family IR→backend stages. Each receipt names inputs, outputs, compiler,
source maps, supported subset, losses, assumptions, obligations, validation,
counterexamples, replay, bounds, and evidence class. Slice obligations so
unchanged stages replay; downstream authority never exceeds upstream semantic
preservation.

### P8 — equality saturation and bounded synthesis

Audit and extend only `ProgramRepairSynthesizer@1`, its reviewed operator
registry, equality mode, and CEGIS path. Require typed e-classes, congruence
closure/rebuild, side conditions, bounded saturation, extraction cost, replay,
and independent equivalence checks where missing. Reviewed grammars support
enumeration, constraints, CEGIS/SyGuS-style search, templates, holes,
invariants/contracts, adapters/serializers, and proof holes without adding
undeclared effects, imports, dependencies, or files.

### P9 — proof-carrying artifact and context

Extend an existing equivalent if present. The artifact binds recursive source
identities, delta, semantic/contract/abstract roots, obligations, translation,
proof/model/test/static/security receipts, residuals, policy/toolchains,
authority, allowed effects, and invalidators. An independent verifier rebuilds
identities and replays compact checks. Context optimization minimizes tokens,
cost, latency, disclosure, and verification overhead subject to complete
mandatory coverage; satisfied proof bodies remain opaque handles.

### P10 — Planner/Doctor self-supervision

Compile goals/subgoals/tasks and current facts into obligations, deterministic
baseline plans, proof-feasible parallel waves, and minimal successor tasks.
Persist append-only typed operational references via existing repositories,
CAS, leases, fences, operation IDs, outbox cursors, and restart reconciliation.
Python, CLI, and MCP remain projections of one semantic service. Preview is the
default for mutations. Meta-contracts enforce zero model calls in deterministic
mode, judge protection, current evidence, budget adherence, and rollback.

### P11 — first complete vertical slice

Run the hermetic Python A→B→C/config/test fixture only through public APIs:
identity, scan, abstract states, contracts, initial discharge, incremental SMT,
counterexample/core/interpolant when qualified, capsules/context, A mutation,
exact invalidation and unaffected reuse, deterministic repair in isolation,
affected-only replay, live fixed point, independently verified artifact, final
context, zero model calls, and token/work-reuse metrics. Add unit, property,
differential, metamorphic, adversarial, integration, and benchmark tests. A
required test cannot silently skip and count as success.

### P12 — paired benchmark and evidence-based disposition

Run baseline and challenger on identical roots, toolchains, policies, seeds,
budgets, and independent oracles across local, cross-module, exception,
schema/configuration/API, security, concurrency, proof, refactor, opaque Python,
and warm-cache tasks. Preserve simulated, hermetic, live-local-model,
live-remote-model, and production-authoritative cohorts separately. Thresholds
are preregistered targets, never output constants. A miss yields a truthful
partial/no-go and minimal successor tasks.

## 6. Parallel waves and conflict policy

| Wave | Runnable tasks | Rule |
| --- | --- | --- |
| W0 | LGCVF-001, then LGCVF-002 | observation and protected successor construction only |
| W1 | LGCVF-010→020→030→040→050 | dependency spine; datasets is validated before gitlink update |
| W2 | LGCVF-051, LGCVF-060, LGCVF-070, LGCVF-080 | disjoint solver, interpolation, translation, and repair lanes after P5 |
| W3 | LGCVF-061, LGCVF-071, LGCVF-081 | refinement/slicing/synthesis hardening |
| W4 | LGCVF-090→091 and LGCVF-100→(101 ‖ 102) | join semantic evidence, then integrate context/persistence projections |
| W5 | LGCVF-110→(111 ‖ 112) | vertical execution, then focused and adversarial qualification |
| W6 | LGCVF-120→(121 ‖ 122), then LGCVF-124 | benchmark, explicit blocker/release disposition, final report |
| Operator | LGCVF-123 | never schedulable; requires independent human authority |

Lane labels are hints only. The compiled dependency/conflict/resource plan is
authority. Until Quack is qualified, one DuckDB writer serializes operational
mutations even when analysis work is otherwise parallel-safe.

## 7. Admission, completion, and stop conditions

A task may start only from a fresh snapshot with satisfied dependencies,
non-overlapping writes, complete resource/authority bounds, and exact lease and
fence. It may settle only after its promised guarantee and acceptance
obligations are independently reproduced on the current tree. Cache reuse must
re-derive freshness and evidence class.

Stop and roll back on stale authority, unsupported mandatory semantics,
invalid evidence lineage, failed postconditions, changed protected inputs,
oscillation, unchanged residual, exhaustion, lease/fence mismatch, or inability
to restore exact bytes. Preserve counterexamples and failed/quarantined
evidence. Never weaken tests, policies, thresholds, seals, or holdouts.

## 8. Evaluation gates

Development targets are zero accepted authority/security/rollback regressions,
zero accepted critical omissions, ≥95% seeded critical-omission detection,
≥50% median model-input reduction, ≥50% warm-cache model-call reduction,
nonzero deterministic closure (development target ≥25% on symbolically
closable tasks), ≥80% unaffected proof/test reuse for local changes, and no
accepted-patch-quality loss. Targets are evaluated, not presumed.

Task implementation, test success, objective completion, release
qualification, and production authorization must be reported as five separate
facts. This plan currently authorizes implementation and hermetic evaluation
only.

## 9. Canonical projections

- Objective hierarchy:
  `docs/architecture/logic_governed_compositional_verification_fabric.objectives.md`
- Supervisor task board:
  `docs/architecture/logic_governed_compositional_verification_fabric.todo.md`
- Formal plan:
  `data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json`
- Generator:
  `scripts/emit_logic_governed_compositional_verification_fabric_plan.py`
- Validator:
  `scripts/validate_logic_governed_compositional_verification_fabric_plan.py`

The formal plan is a typed statement of intended work. Its content ID proves
canonical content identity only; validator success proves internal plan/board
consistency only. Neither proves code, release readiness, or authorization.
