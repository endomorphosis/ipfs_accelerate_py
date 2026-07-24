# Agent Supervisor Formal Planning and Prover Matrix Plan

## Status and relationship to G11

This plan extends, rather than replaces, the proof-aware supervisor program in
`AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md`.

- `G11` builds code-proof contracts, AST obligations, evidence graphs, Hammer
  integration, kernel reconstruction, trusted caching, Leanstral shadow
  assistance, ZKP receipt attestation, proof-aware merge gates, and shared CPU
  scheduling.
- `G12` formally represents the work the supervisor intends to perform, checks
  that plan before model dispatch, executes property-specific prover families,
  monitors the resulting trace, and replans from typed counterexamples.

The two goals use one trust model and one evidence graph. G12 must not create a
second proof cache, scheduler, goal lifecycle, or assurance vocabulary.

## Objective

Use the logic capabilities in `ipfs_datasets_py` to:

1. Compile goals, subgoals, tasks, AST scopes, policies, and evidence
   requirements into a deterministic formal work plan.
2. Detect contradictory, impossible, unauthorized, unsafe, or non-terminating
   plans before sending work to Codex or another model.
3. Give models a bounded, proof-carrying plan capsule so they do not have to
   rediscover repository-wide dependencies in their context window.
4. Route each property to a prover with matching semantics and record the exact
   translation, finite bounds, assumptions, toolchain, and assurance.
5. Formally check supported generated-code obligations and use tests and review
   for unsupported Python behavior.
6. Monitor real daemon and lane events against the accepted plan and create
   bounded repair tasks from counterexamples.
7. Preserve implementation throughput with one CPU, memory, process, and model
   admission budget across all serial and parallel lanes.

## Execution architecture

The first executable slice of this plan is organized as a sequence of typed
representation changes. Intent becomes a formal work plan; the plan becomes a
property-specific proof DAG; the observed execution becomes a conformance
trace; and typed counterexamples become bounded repair candidates:

- `formal_plan_conformance.py` binds an accepted plan to canonical execution
  events and evaluates transition order, authority, evidence freshness, and
  invalidation. The same evidence model drives goal completion and reopening.
- `formal_replanner.py` turns typed counterexamples into bounded repair
  candidates (`add_dependency`, `split_effects`, `tighten_authority`,
  `add_evidence`, `constrain_scope`, `add_premise`, `change_resource_bounds`,
  or `human_review`). Candidates are semantically deduplicated and emitted as
  bounded repair packets for model assistance.
- `multi_prover_resources.py` provides one resource manager for every prover
  family and nested child process. It supports dependency-closed bundle slices,
  deterministic receipt caching, adaptive portfolio width, process-group
  cancellation, and explicit CPU, memory, process, wall-time, and diagnostic
  limits.
- `formal_planning_adversarial.py` verifies plan/evidence identity bindings,
  provider conformance, cache age and digest, property-specific evidence
  classes, and public-output redaction. It fails closed on unknown, cancelled,
  timed-out, forged, contradictory, or insufficient evidence.
- `proof_carrying_planner.py` persists a resumable DAG for compile, bounded
  verification, implementation, scope verification, merge, runtime monitor,
  and repair. JSON and DuckDB projections are replayable; completion requires
  accepted merges, an accepted runtime trace, and the configured authoritative
  assurance.
- `formal_planning_metrics.py` and `formal_planning_rollout.py` define the
  benchmark report and promotion policy. Samples are explicitly `cold`,
  `warm`, or `parallel`; reports exclude raw context, proof bodies,
  counterexample bodies, and private witnesses.

This is an executable integration slice, not a blanket verification claim.
Unsupported providers continue through deterministic fallback validation, and
provider output remains candidate evidence until the relevant conformance and
assurance checks pass.

The important invariant is that these stages are not interchangeable. A
counterexample is an observation, not an instruction; a repair candidate is a
proposal, not a changed objective; and a proof receipt is authoritative only
for its exact obligation, tree, assumptions, translation, toolchain, and
bounds. This preserves soundness when the model, cache, provider, or
repository changes.

### Rollout policy defaults

Promotion evaluates context reduction, pre-dispatch defect detection, proof
support, counterexample quality, warm-cache reuse, queue latency, CPU
saturation, memory peak, and accepted-task throughput. The default thresholds
are:

| Mode | Context reduction | Proof support | Warm reuse | Throughput ratio | Minimum assurance |
| --- | ---: | ---: | ---: | ---: | --- |
| `shadow` | 5% | 40% | 20% | 0.65 | `candidate` |
| `canary` | 15% | 65% | 45% | 0.80 | `solver_checked` |
| `enforcement` | 25% | 80% | 60% | 0.90 | `kernel_verified` |

The policy also caps queue latency at 750/400/250 ms, CPU saturation at
95/90/85%, and peak memory at 4/3/2 GiB for shadow/canary/enforcement. Each
mode requires cold and warm samples by default. A lane that is unavailable,
low-value, incomplete, or below its property-specific assurance remains
diagnostic or shadow-only; aggregate benchmark success cannot promote it.

### Recommended operating sequence for the new workflow

1. Compile and adversarially admit the plan, retaining its tree, policy,
   provider, cache, and resource identities.
2. Run the proof-carrying DAG under the shared resource manager. Persist every
   decision, evidence record, lease, and partial receipt.
3. Compare observed events with the accepted plan before merge and completion.
4. Feed bounded counterexamples to the replanner; accept only candidates that
   improve the targeted finding without exceeding repair budgets.
5. Evaluate the pinned cold/warm/parallel matrix before changing rollout mode.
6. Replay the durable artifact after restart and verify that the same plan,
   evidence, and assurance decision is obtained.

## Examined Logic Surfaces

The assessment covered these implementation families:

| Family | Representative implementation |
| --- | --- |
| DCEC and CEC | `logic/CEC/native/dcec_core.py`, `prover_core.py`, `shadow_prover.py`, `cec_proof_cache.py`, `inference_rules/` |
| TDFOL | `logic/TDFOL/tdfol_core.py`, `tdfol_parser.py`, `tdfol_prover.py`, `tdfol_proof_cache.py`, `formula_dependency_graph.py`, `strategies/` |
| DCEC/TDFOL bridges | `logic/integration/bridges/tdfol_cec_bridge.py`, `logic/bridge/cec_dcec.py`, `logic/bridge/fol_tdfol.py` |
| ATP, SMT, and ITP routing | `logic/external_provers/`, `logic/CEC/provers/`, `logic/hammers/` |
| Leanstral | `logic/modal/leanstral*.py`, `logic/modal/lean_runtime.py` |
| Frame logic and graphs | `logic/bridge/modal_frame_logic.py`, `logic/modal/kg_bridge.py`, `logic/knowledge_graphs/` |
| Proof caches | `logic/common/proof_cache.py`, `logic/hammers/proof_cache.py`, DCEC and TDFOL cache adapters |
| ZKP | `logic/zkp/`, `logic/bridge/zkp_attestation.py`, DCEC, TDFOL, and frame-logic ZKP adapters |
| Security-model workflows | `logic/security_models/crypto_exchange/`, including AST extraction, SMT runners, TLA/Apalache reports, runtime MTL, receipts, and release policy |
| Matrix definition | `docs/security_verification/prover_matrix.md` |

## Capability findings as an evidence architecture

Capability discovery is a routing boundary, not a proof boundary. A provider
is usable only when its executable, bindings, model, translation profile, and
conformance fixtures satisfy the requested operation. Missing or degraded
capabilities remain explicit inputs to planning and scheduling. They do not
become false success, and they do not make the package import fail.

The evidence graph is an architectural index rather than a transcript of
implementation activity. Traversal must be deterministic, content addressed,
bounded, and safe to rebuild. A graph edge can narrow context or identify a
dependency, but it cannot by itself establish code correctness.

### Strong reusable foundations

- TDFOL has a typed formula AST, parsers, temporal and deontic rules, proof
  results, dependency graphs, caches, countermodels, and conversion paths.
- Native DCEC represents deontic, cognitive, event, and temporal concepts that
  fit supervisor actors, intentions, authority, tasks, and execution events.
- Hammer already separates premise selection, portfolio execution, candidate
  evidence, reconstruction, receipts, cache trust, and release policy.
- Lean and Coq frontends provide a path to small trusted checking kernels when
  the exact theorem and assumptions are preserved.
- Leanstral has useful batching, cache, audit, and bounded prompt machinery,
  but remains an untrusted proposal source.
- The security-model package demonstrates deterministic IRs, content
  identities, AST extraction, differential SMT checks, TLA generation,
  Apalache execution receipts, runtime MTL, counterexample reports, evidence
  review, and fail-closed release policy.
- Existing JSON and DuckDB supervisor stores can hold plans, matrix entries,
  graph records, receipts, traces, and metrics without placing entire datasets
  in a model prompt.

### Important soundness and maturity gaps

- Native DCEC and TDFOL proof success is not equivalent to acceptance by a
  small independently checked kernel. It can check plans in shadow or bounded
  policy modes, but cannot by itself mark generated Python as verified.
- The TDFOL CEC delegate still contains an "integration in progress" unknown
  path. A bridge being importable is not evidence that all formulas preserve
  semantics end to end.
- The CEC Z3 adapter encodes modal operators with uninterpreted functions.
  Such an encoding needs a declared abstraction contract and cannot silently
  stand in for full deontic, cognitive, or temporal semantics.
- Legacy CEC deontic rules currently show API drift, including obsolete result
  enum use and an undefined operator reference. Those paths must be
  quarantined until conformance tests pass.
- Some legacy proof-cache tests assert wall-clock speedup ratios on
  millisecond-scale operations. Cache correctness and identity tests are
  useful; timing ratios are not sound evidence.
- Source presence, an installer, executable discovery, or a version response
  does not establish that a prover can translate and verify a supervisor
  property.
- Frame logic and knowledge graphs provide structure, retrieval, lineage, and
  scoped relations. They are not general Python verification kernels.
- A simulated ZKP verifies serialization behavior only. It cannot provide
  authoritative attestation.
- Formal verification of arbitrary Python remains out of scope. Only reviewed
  obligation templates with explicit semantics may enter enforcement.

## Prover Matrix Assessment

The repository matrix names seven prover families. The supervisor should
convert this static matrix into an executable conformance registry.

| Matrix family | Current repository evidence | Supervisor use | Maximum initial assurance |
| --- | --- | --- | --- |
| Z3/CVC5 | Python and CLI adapters, SMT-LIB compilers, differential tests; local CLIs available | Finite constraints, resource invariants, bounded transition relations, countermodels | `solver_candidate` until the translation profile is conformant; kernel reconstruction where available |
| TLA+/TLC/Apalache | Domain-specific TLA generators and Apalache reports; pinned Apalache installer and local executable | Concurrent leases, retries, merges, refill, fencing, and capacity state machines | `bounded_model_checked` with exact model and bounds |
| Datalog/SecPAL style | Matrix target; no general supervisor lane found | Claim, delegation, revocation, merge authority, and override policy | `checked` reference policy until an external lane passes conformance |
| Tamarin/ProVerif | Pinned installers and domain-specific protocol workflow/tests; local executables available | Claim grants, fencing freshness, replay resistance, receipt binding, attestation exchange | `protocol_checked` only for exact versioned models and queries |
| HyperLTL/AutoHyper/MCHyper | Matrix target; no executable supervisor integration found | Prompt, worktree, cache, log, provider, and witness noninterference | `unsupported` until end-to-end adapters and fixtures exist |
| Lean/Coq | Existing frontends, reconstructors, runtime checks, and local kernels | Final checking of exact obligations and reconstructed candidates | `kernel_verified` when unsafe escapes and theorem substitution are rejected |
| Runtime MTL | A domain-specific Python monitor exists | Live task, lease, proof, validation, merge, cancellation, and deadline traces | `runtime_checked`; absence of a violation is not a proof |

Additional local prover families must be registered with the same rules:

- DCEC and TDFOL for typed formal planning and temporal-deontic reasoning.
- Vampire and E for first-order candidates.
- Isabelle where its frontend and reconstructor pass executable conformance.
- Native modal tableaux and ShadowProver as bounded or candidate lanes.
- Hammer as the portfolio and reconstruction coordinator.
- Leanstral as an LLM proposal lane only.
- ZKP backends as optional attestation over an already trusted receipt.

### Local tool snapshot

At assessment time the host exposed:

- Z3 `4.16.0`
- cvc5 `1.3.2`
- Lean `4.31.0`
- Rocq/Coq `9.1.1`
- Apalache `0.58.3`
- Tamarin `1.12.0`
- ProVerif `2.05`

The Python `z3` package was not importable even though the Z3 CLI was present.
TLC, Datalog/SecPAL, HyperLTL, AutoHyper, and MCHyper executables were not
discovered on the normal path. Capability status must therefore be
operation-specific, not a single boolean per family.

## Conformance Baseline

A 403-test representative run covered TDFOL core and proving, TDFOL caches,
temporal-deontic rules, the DCEC bridge, DCEC core and proving, event calculus,
CEC caches and prover manager, runtime MTL, Apalache/TLA workflows, and protocol
lanes.

- 355 passed.
- 23 skipped, primarily unavailable optional prover paths.
- 25 failed.
- Two failures were timing-ratio assertions in the CEC proof-cache suite.
- The remaining failures were concentrated in the legacy CEC deontic
  inference-rule layer and exposed API and representation drift.

This baseline is not a package-wide verification result. It is evidence that
provider conformance and quarantine must precede supervisor enforcement.

Materializing G12 also exposed a false-positive cycle in the supervisor's
iterative strongly-connected-component traversal. A converging acyclic graph
(`A -> B`, `A -> C`, `B -> C`) could be reported as cyclic because sibling
nodes were marked visited before traversal. The traversal and its regression
coverage were corrected; rebuilding the current G12 index now produces five
bundles, 20 tasks, and no dependency-repair evidence. This is an implementation
prerequisite for trusting later formal-plan DAG checks.

## Trust Model

G12 extends the G11 trust lattice with property-specific checked states:

1. `unverified`: model text, proposed formula, undocumented adapter output.
2. `schema_checked`: deterministic shape and identity validation.
3. `plan_checked`: the plan passed configured bounded DCEC/TDFOL checks.
4. `solver_candidate`: an SMT or ATP lane returned a candidate result.
5. `bounded_model_checked`: a model checker found no violation within recorded
   finite bounds.
6. `runtime_checked`: an observed finite trace had no configured violation.
7. `protocol_checked`: a versioned protocol model and query passed an accepted
   Tamarin or ProVerif lane.
8. `kernel_verified`: an accepted Lean, Coq, or Isabelle kernel checked the
   exact obligation under the exact assumptions.
9. `attested`: a production cryptographic backend bound a trusted receipt to
   its public statement.

These states are not a simple numeric ladder. A protocol result cannot satisfy
a code-functional theorem, a runtime trace cannot establish all future
behavior, and a bounded model result cannot be reported without its bounds.
Policy selects required evidence by property class.

## Formal Work-Plan Model

### Core records

`FormalWorkPlan` should contain:

- repository, base tree, candidate tree, objective, goal, subgoal, and task
  identities;
- actors and roles: supervisor, scheduler, worker, reviewer, prover, kernel,
  merge authority, and operator;
- resources and scopes: files, AST symbols, worktrees, leases, CPU, memory,
  providers, and proof artifacts;
- task preconditions, effects, acceptance criteria, fallback checks, risks,
  costs, deadlines, dependencies, and conflict edges;
- events and fluents;
- deontic norms and temporal constraints;
- proof obligations and required assurance by property class;
- allowed repair operations and bounded refinement policy;
- canonical JSON, content identity, schema version, and provenance.

### Frame and graph projection

Frame logic supplies a bounded structural view for modal planning. Versioned
worlds represent plan, candidate-tree, execution-epoch, and evidence states;
reviewed accessibility relations represent permitted transitions, knowledge
visibility, and temporal successors. The projection is derived from canonical
plan and evidence-graph records, records omitted edges and finite bounds, and
is queried by identity rather than regenerated from model text.

The knowledge graph supplies lineage, relevance, dependency closure, premise
selection, and compact counterexample neighborhoods. Frame reachability and
graph connectivity remain structural evidence only; neither is promoted into
a proof of generated Python.

### DCEC projection

DCEC should model agency and norms. The initial reviewed vocabulary includes:

- `Goal(supervisor, objective)`
- `Intends(worker, implement(task))`
- `Knows(supervisor, dependency_complete(task, dependency))`
- `Happens(claim(worker, task, lease), time)`
- `Initiates(claim(...), leased(task, worker, epoch), time)`
- `Terminates(expire(lease), leased(task, worker, epoch), time)`
- `Obligated(worker, produce_evidence(task))`
- `Permitted(merge_authority, merge(task))`
- `Forbidden(worker, publish(task, stale_epoch))`
- `Forbidden(merge_authority, merge(task, missing_required_evidence))`

The vocabulary is fixed and versioned. LLM-generated DCEC is a suggestion until
it maps to a reviewed predicate and passes validation.

### TDFOL projection

TDFOL should model quantified dependencies, temporal constraints, and goal
satisfaction. Initial properties include:

- Every started task has completed dependencies.
- At most one accepted live lease exists for a canonical task and epoch.
- A stale fencing token can never publish or merge.
- A task that starts eventually becomes completed, blocked, failed, cancelled,
  or superseded within configured bounds.
- A protected merge is always preceded by fresh required validation and proof
  evidence for the candidate tree.
- Cancellation or authority revocation prohibits subsequent publication.
- Resource reservations never exceed capacity.
- A goal becomes verified only after every mandatory criterion has fresh
  property-appropriate evidence.

TDFOL proof explanations and dependency graphs should identify the minimal
premises needed for each plan decision and context capsule.

### Plan verification result

Plan verification returns separate fields:

- parse and schema status;
- dependency and conflict-graph status;
- DCEC consistency status;
- TDFOL safety and liveness status;
- bounded model-check status and exact bounds;
- authority-policy status;
- unsupported semantics;
- counterexamples and unsat cores;
- required implementation and code-proof obligations;
- authoritative decision derived from policy.

## Architecture

```text
goals + taskboard + policy + AST + evidence
                    |
                    v
        deterministic FormalWorkPlan compiler
                    |
        +-----------+-----------+
        |                       |
        v                       v
 DCEC/TDFOL plan checks    evidence/KG query
        |                       |
        +-------- counterexamples
                    |
                    v
       property-specific proof-plan DAG
                    |
  +---------+-------+--------+---------+---------+
  |         |       |        |         |         |
 SMT/ATP  TLA/MC  auth   protocols  hyperprops  runtime
  |         |       |        |         |         |
  +---------+-------+--------+---------+---------+
                    |
       reconstruction / policy derivation
                    |
         bounded formal plan capsule
                    |
              Codex / Leanstral
                    |
       candidate tree + validation + proof
                    |
             merge and goal gate
                    |
          runtime trace conformance
                    |
        typed counterexample -> replan
```

## Context-Reduction Strategy

The formal planner should perform deterministic work before model dispatch:

1. Resolve the task and dependency closure from DuckDB.
2. Select changed and required AST symbols by graph edge and blob identity.
3. Select reviewed invariant and plan templates.
4. Reuse fresh trusted premise and proof receipts.
5. Run bounded plan checks and attach only minimal proof explanations,
   counterexamples, or unsat cores.
6. Emit one `FormalPlanContextCapsule` with strict row, hop, byte, token, and
   source limits.
7. Require the model response to bind the task, plan, tree, and policy CIDs.

The model should not receive full AST dumps, complete knowledge graphs, raw
prover transcripts, hidden witnesses, unrelated files, or all prior attempts.

## Translation and Routing Policy

Every translation edge declares:

- source and target logic;
- semantic profile and version;
- exact, equisatisfiable, bounded, conservative, lossy, or heuristic class;
- supported operators and sorts;
- dropped or abstracted information;
- finite domains and temporal bounds;
- soundness direction;
- round-trip and mutation fixtures;
- maximum assurance the result may receive.

Routing is by property, not by tool availability:

| Property | Primary lane | Independent evidence |
| --- | --- | --- |
| Arithmetic, finite capacity, acyclicity | Z3/CVC5 | Differential solver and reference predicate |
| Concurrent scheduler transitions | TLA+/TLC/Apalache | Runtime trace and targeted tests |
| Agent intent, norms, temporal duties | DCEC/TDFOL | Bounded model checks or kernel theorem where required |
| Delegation and authorization | Datalog/SecPAL style | Reference policy evaluator and negative fixtures |
| Claim and receipt protocols | Tamarin/ProVerif | Protocol attack fixtures and runtime correspondence |
| Information-flow noninterference | HyperLTL family | Bounded self-composition tests |
| Exact mathematical obligation | Hammer plus Lean/Coq/Isabelle | Kernel acceptance |
| Observed temporal behavior | Runtime MTL | Replay and mutation fixtures |
| Receipt privacy and portability | Production ZKP backend | Public-input and witness no-leak tests |

## Counterexample-Guided Replanning

All failure families normalize into `FormalCounterexample`:

- violated property and property class;
- task, plan, tree, scope, and policy identities;
- assumptions, finite bounds, and observation policy;
- minimal state trace, model, unsat core, attack trace, hypertrace, or kernel
  diagnostic;
- affected graph nodes and invalidated evidence;
- redaction and confidentiality disposition;
- deterministic repair classes.

Allowed repair classes initially include:

- add or correct a dependency;
- split a task whose effects cannot be modeled atomically;
- tighten an authority, lease, fencing, or override rule;
- add a reviewed obligation template or fallback test;
- constrain an AST scope or model bound;
- add a missing premise or evidence dependency;
- reduce portfolio width or adjust a resource class;
- request scoped human semantic review.

Each repair is recompiled and rechecked before taskboard admission. Refinement
depth, duplicate identities, token budget, retry count, and an explicit progress
measure prevent infinite autonomous replanning.

## Evidence Graph and Storage

Add these node types to the G11 evidence graph:

- formal plan, actor, role, event type, fluent, norm, temporal property;
- translation profile, model, finite bound, property class, matrix entry;
- state trace, attack trace, hypertrace, runtime violation, plan conformance;
- repair candidate and supersession receipt.

Add these edges:

- `plans`, `requires_before`, `permits`, `forbids`, `initiates`, `terminates`;
- `modeled_by`, `translated_by`, `checked_with`, `bounded_by`;
- `conforms_to`, `violates`, `counterexample_to`, `repaired_by`, `supersedes`.

Canonical records are persisted to paired JSON and DuckDB. The model-facing
query API returns bounded projections, never an entire table or graph.

## Cache Policy

Plan and prover cache keys must include:

- plan, task, tree, AST scope, policy, property, and obligation identities;
- normalized formulas, models, assumptions, domains, and finite bounds;
- translation profile and conformance fixture set;
- prover, kernel, model, circuit, and runtime versions;
- resource and timeout policy where it changes semantics;
- trust requirement and freshness policy.

The cache stores unsuccessful and counterexample results when safe, but no
cached provider field may assert authoritative assurance. Assurance is derived
again on every read. Single-flight ownership spans serial and parallel
supervisors.

## CPU and Parallelism

One top-level scheduler accounts for:

- `cpu-translate`
- `cpu-smt`
- `cpu-atp`
- `cpu-itp-kernel`
- `cpu-jvm-model-check`
- `cpu-protocol`
- `cpu-hyperproperty`
- `cpu-runtime-monitor`
- `cpu-validation`
- `llm-implementation`
- `llm-proof-proposal`
- `io-artifact`

Independent plan and proof DAG nodes may run in parallel. Each child process
receives a bounded lease for cores, threads, memory, time, and output. Nested
solver and BLAS thread counts are capped. A conclusive counterexample cancels
redundant portfolio attempts. Process-group cancellation and restart must
release capacity and preserve a partial receipt.

Bundle admission operates on dependency-closed ready-member slices. A blocked
later member of a goal or subgoal bundle must not idle an earlier ready task,
and a mixed-readiness lane must not disable task-level dependency enforcement.
The bundle identity remains stable while each admitted slice records its member
set, dependency frontier, and resource lease.

Tool installation and network access are separate operator-controlled actions,
not implicit proof execution.

## ZKP Role

ZKP can attest:

- the plan, tree, obligation, policy, and trusted receipt identities;
- a supported derivation trace or knowledge-of-axioms statement;
- public matrix and toolchain identities;
- that a private witness satisfied the versioned circuit.

ZKP cannot establish that an unsound translation represents Python correctly.
A simulated backend cannot satisfy enforcement. Witnesses, private source,
credentials, and raw model context never enter public inputs or logs.

## Rollout

### Stage 0: Capability truth and quarantine

Implement executable matrix entries and conformance fixtures. Quarantine the
currently failing CEC deontic paths. No planning or merge behavior changes.

### Stage 1: Formal plan compilation in shadow mode

Compile existing tasks into DCEC/TDFOL plans, record unsupported semantics, and
measure context capsules. Codex dispatch remains unchanged.

### Stage 2: Plan consistency and bounded state models

Reject only malformed or cyclic plans. Run DCEC/TDFOL and TLA/Apalache checks
advisorially and normalize counterexamples.

### Stage 3: Proof-carrying model dispatch

Use bounded plan capsules for selected low-risk tasks. Compare model tokens,
accepted patches, and repair rates against control tasks.

### Stage 4: Runtime conformance and automatic repair

Monitor supervisor traces and append semantically deduplicated repair tasks.
Reopen goals when required plan or evidence invariants are violated.

### Stage 5: Specialized security lanes

Canary authorization, protocol, and hyperproperty checks only after their
end-to-end conformance fixtures pass.

### Stage 6: Property-specific enforcement

Require formal plan and code evidence for a small set of reviewed supervisor
invariants. Expand only from measured mutation detection, false-block rate,
context reduction, cache value, and CPU throughput.

## Metrics

Track:

- plan compile, support, contradiction, counterexample, and repair counts;
- tasks rejected before model dispatch and model tokens avoided;
- context bytes and tokens by capsule section;
- property, translator, prover, kernel, bounds, and assurance;
- solver disagreement and conformance-quarantine rates;
- cold and warm cache behavior without timing-ratio correctness claims;
- plan critical-path accuracy and downstream unlock value;
- runtime violations, reopen events, and repair convergence;
- queue latency, CPU saturation, child process count, memory, cancellation,
  single-flight, and accepted implementation throughput;
- security leakage mutations and protocol or hyperproperty counterexamples.

## Executable Backlog

The root refactor supervisor appends:

- Goal `G12`: Formally plan supervisor work across the prover matrix.
- `G12.S1`, tasks `REF-275` through `REF-278`: DCEC/TDFOL formal work plans and
  bounded model context.
- `G12.S2`, tasks `REF-279` through `REF-282`: executable matrix,
  conformance, routing, receipts, and caches.
- `G12.S3`, tasks `REF-283` through `REF-286`: TLA state models,
  Datalog/SecPAL authority, Tamarin/ProVerif protocols, and HyperLTL
  information flow.
- `G12.S4`, tasks `REF-287` through `REF-290`: runtime MTL,
  counterexample graph, bounded replanning, and goal conformance.
- `G12.S5`, tasks `REF-291` through `REF-294`: shared resources,
  adversarial validation, end-to-end execution, benchmarks, and rollout.

G12 has parallel roots after the early G11 capability and contract tasks. Its
specialized prover lanes converge only after translation conformance and trust
policy are available.

## Exit Criteria

Before any G12 property enters enforcement:

- its vocabulary and translation profile have reviewed semantics;
- executable conformance and negative fixtures pass;
- all finite bounds and assumptions are visible;
- unavailable, timeout, unknown, and disagreement outcomes fail closed;
- the authoritative decision is derived from evidence, not provider text;
- stale cache, changed model, changed tree, changed policy, and changed
  toolchain mutations are rejected;
- paired JSON and DuckDB projections are equivalent;
- context capsules satisfy configured limits and exclude secrets;
- serial and parallel execution stays within shared resource capacity;
- an end-to-end restart test reproduces the same plan and evidence verdict;
- measured context reduction and defect detection justify rollout cost.
