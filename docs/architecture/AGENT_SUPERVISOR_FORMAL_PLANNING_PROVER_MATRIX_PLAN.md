# Agent Supervisor Formal Planning and Prover Matrix Plan

## Status and decision

This document is the reviewed architecture contract for `REF-275` and the
foundation for the G12 formal-planning track (`REF-276` through `REF-294`).
It extends, rather than replaces, the proof-evidence contracts introduced by
`REF-245`.

The central decision is:

> The supervisor compiles trusted, typed supervisor records into a canonical
> `FormalWorkPlan` before asking a language model to implement work. It never
> obtains authoritative formulas by interpreting free-form model text.

The plan contract describes intended work. A proof that this description is
internally consistent is evidence about the plan only. It is not evidence that
an implementation followed the plan, that generated Python is correct, or that
a repository tree satisfies a code invariant.

The initial wire contracts are version 1. A schema version changes whenever an
operator, field, finite-trace interpretation, normalization rule, or identity
rule changes meaning. Additive provider support does not change plan semantics,
but every provider and translator profile remains independently versioned and
bound into its execution receipt.

## Objectives

The architecture has five goals:

1. Give actors, goals, subgoals, tasks, events, state, norms, temporal
   constraints, and evidence requirements one deterministic representation.
2. Supply a small, reviewed DCEC vocabulary for cognitive, deontic,
   delegation, and execution claims without using an NLP-to-logic path.
3. Supply a small, reviewed TDFOL vocabulary for dependencies, deadlines,
   safety, liveness, and goal satisfaction over explicit finite traces.
4. Derive a versioned, bounded frame-logic and evidence-graph projection for
   navigation and model-context selection.
5. Keep plan consistency, execution conformance, and generated-code assurance
   as three separately evaluated levels with no implicit promotion between
   them.

This design is intentionally narrower than “formalize arbitrary software.” Its
subject is the supervisor's own canonical work records and reviewed predicate
templates.

## Relationship to the REF-245 trust model

`formal_verification_contracts.py` remains the authority for proof attempts,
proof receipts, resource budgets, and the ordered code-proof assurance lattice:

`unverified < candidate < solver_checked < kernel_verified < attested`

The formal-planning layer reuses its canonical JSON and content-identity rules
and may reference `ProofReceipt` CIDs as evidence. It does not redefine that
lattice or manufacture one of its levels.

The following boundaries are normative:

| Artifact or result | What it can establish | What it cannot establish |
| --- | --- | --- |
| Schema-valid `FormalWorkPlan` | The record is canonical and structurally valid | Consistency, conformance, or code correctness |
| DCEC/TDFOL consistency result | A reviewed finite abstraction has no found contradiction, or has a countermodel | That an execution occurred or a patch is correct |
| Frame-logic projection | A deterministic bounded view and accessibility graph | A proof of a formula or code property |
| Plan-conformance receipt | Observed canonical events conform to selected plan transitions within stated bounds | Correctness of event-producing code |
| Solver/ATP result | A candidate result under the exact translation and assumptions | Kernel assurance unless reconstructed and checked |
| Model-checker success | The generated finite model satisfies properties for the recorded state/trace bounds | Unbounded correctness or source-code correctness |
| `CodeProofObligation` receipt | The exact changed AST scope satisfies the exact reviewed code template at its derived REF-245 assurance | Conformance of unrelated work or arbitrary code correctness |
| Cryptographic attestation | A genuine verifier binds public inputs to an already qualified receipt | Stronger semantics than the attested receipt |

A code-proof requirement can cite a plan element, but it must still be compiled
as a `CodeProofObligation` against the candidate repository tree and changed
AST scopes. A plan consistency receipt is never accepted in its place.

## Non-goals

Version 1 does not:

- translate natural-language goals, prompts, model rationales, commit messages,
  or source comments directly into proof formulas;
- claim that DCEC, TDFOL, F-logic, graph reachability, or an LLM response is a
  sound semantics for arbitrary Python;
- treat provider discovery, executable presence, documentation, installer
  success, or a smoke test as proof success;
- infer actor authority from an actor name, provider identity, branch name, or
  possession of a worktree;
- assume closed-world negation except for predicates whose reviewed profile
  explicitly declares it;
- prove unbounded liveness from a finite trace;
- hide unsupported fields, dropped quantifiers, missing times, lossy
  translations, or incomplete exploration;
- let Leanstral or another model alter a theorem, plan identity, policy,
  trusted premise, or evidence verdict;
- let evidence-graph connectivity, path count, embedding similarity, or
  GraphRAG ranking satisfy a proof or completion gate.

## Architecture

```text
canonical objective/taskboard/AST/policy/evidence records
                         |
                         v
        deterministic formal-plan compiler (REF-276)
                         |
            +------------+-------------+
            |                          |
            v                          v
  FormalWorkPlan @ CID          bounded frame/evidence view
            |                          |
            v                          +--> bounded model capsule
  DCEC + TDFOL obligation set
            |
            v
  consistency validator (REF-277)
            |
        result + bounds + countermodel
            |
            +-------> implementation dispatch (REF-278)
            |                   |
            |                   v
            |           candidate tree + events
            |                   |
            +----------+--------+
                       |
          +------------+-------------+
          |                          |
          v                          v
 plan-conformance evaluation   code-proof/validation policy
          |                          |
          +------------+-------------+
                       |
            goal-completion gate (REF-290)
```

The compiler, validator, provider adapters, event monitor, and completion gate
are separate components. They exchange content-addressed records instead of
shared mutable formula objects.

## Canonical work-plan contract

### Top-level record

`FormalWorkPlan` is an immutable, content-addressed record. Its canonical
payload contains:

- a schema identifier and contract version;
- a vocabulary profile identifier and version;
- source record CIDs and an optional repository/base-tree CID;
- finite bounds and the abstraction policy;
- actors;
- goals and subgoals;
- tasks and dependency/conflict edges;
- event declarations;
- fluent declarations;
- preconditions and effects;
- norms;
- temporal constraints;
- evidence requirements;
- frame-projection configuration;
- typed formulas assembled from the reviewed vocabulary;
- declared unsupported or abstracted source fields;
- metadata limited to strict canonical JSON.

Every collection with set semantics is normalized, deduplicated, and sorted by
its semantic key. Every collection with sequence semantics preserves order and
declares that fact in its field contract. A list is never ambiguously treated
as sometimes ordered and sometimes unordered.

### Entity records

Each entity has a semantic key local to the plan and a deterministic content
identity. Human-readable labels are annotations, not logical identifiers.

| Record | Required semantics |
| --- | --- |
| `Actor` | Stable actor key, actor kind, principal/claimant reference, declared capabilities, authority/policy references |
| `Goal` | Goal key, required satisfaction predicate, owning actor, initial/terminal status set, acceptance and evidence references |
| `Subgoal` | Parent goal key plus its own satisfaction predicate, dependencies, and evidence references |
| `Task` | Goal/subgoal membership, assigned or eligible actors, dependency keys, resource class, expected events, precondition/effect references, terminal statuses |
| `Event` | Event key, event kind, actor, task, logical time, optional observed timestamp/epoch, parameters, and provenance |
| `Fluent` | Fluent key, typed arguments, value domain, initial value or explicit unknown state, and persistence policy |
| `Precondition` | Reviewed formula that must hold at an explicitly named transition point |
| `Effect` | Initiate, terminate, assign, or emit operation with an explicit transition point and target |
| `Norm` | Modal kind, bearer, action or formula, activation condition, scope, issuer/authority, validity interval, and discharge/violation event |
| `TemporalConstraint` | Reviewed temporal operator, subject references, time/step bounds, and trace interpretation |
| `EvidenceRequirement` | Covered goal/task/criterion, evidence kind, source/tree/scope bindings, freshness rule, minimum authoritative assurance, and fallback policy |

References are by canonical keys or CIDs, never by object address or display
text. Unknown references, duplicate semantic keys, self-dependencies,
unbounded recursive values, illegal effect targets, and conflicting initial
values make the contract invalid.

### Typed values and formulas

The contract uses a closed typed formula AST, not strings:

- terms: `symbol`, `variable`, `boolean`, `integer`, `time`, `duration`,
  `actor_ref`, `goal_ref`, `task_ref`, `event_ref`, `fluent_ref`,
  `evidence_ref`, `tree_ref`, and reviewed function application;
- atoms: a reviewed predicate identifier plus arity- and sort-checked terms;
- connectives: `not`, `and`, `or`, `implies`, and `iff`;
- quantifiers: `forall` and `exists` over a declared finite domain;
- modal/temporal applications: only operators registered by the selected
  vocabulary profile.

Raw formula source may be retained as non-authoritative diagnostic metadata,
but it is excluded from formula construction and never reparsed to recover
semantics. A formula node carries its operator ID, typed operands, profile
version, and canonical identity.

Formula validation rejects:

- an unknown operator or predicate;
- wrong arity or operand sort;
- a free variable;
- a quantifier without a finite domain bound;
- a time outside the plan horizon;
- an unsupported nesting combination;
- an undeclared entity reference;
- an arbitrary function, axiom, import, code fragment, or provider expression;
- a claimed formula CID that does not match canonical content.

### Preconditions and effects

Preconditions and effects define a transition relation over trace states.
For task `x` at transition step `i`:

- all task and policy preconditions are evaluated in state `S[i]`;
- `execute(actor, x)` is the transition event;
- effects are evaluated atomically to produce `S[i+1]`;
- simultaneous conflicting assignments to the same fluent are invalid unless a
  reviewed conflict-resolution profile explicitly defines their order;
- fluents not affected by the transition persist only when declared inertial;
- unknown initial fluents remain unknown and cannot be assumed false.

This makes effects deterministic enough for finite checking. It does not assert
that repository code actually implements the transition.

### Source provenance and abstraction log

The compiler must bind every formal element to one or more canonical source
records:

- objective, goal, subgoal, taskboard, dependency, lease, and resource CIDs;
- policy and rollout-mode CIDs;
- repository/base-tree and AST symbol CIDs;
- acceptance criterion and validation-command CIDs;
- evidence and contradiction CIDs.

Fields that cannot be mapped have an explicit abstraction entry containing the
source CID, source field path, status (`unsupported`, `ignored_by_profile`,
`bounded`, or `approximated`), reason, and affected formula IDs. An
unsupported authority, effect, temporal bound, or evidence requirement is an
invalid/unsupported plan outcome; it is never silently omitted.

### Deterministic identity

Identity follows the REF-245 canonical boundary:

1. Values are strict DAG-JSON-compatible values. Floating-point values,
   non-string map keys, NaN/infinity, Python objects, and unordered
   serialization are rejected.
2. Unicode and textual identifiers are normalized by the contract's declared
   rule; insignificant leading/trailing whitespace is not identity-bearing.
3. Maps are key-sorted. Set-like tuples are normalized and sorted. Ordered
   traces and effect sequences preserve order.
4. Derived identities are omitted from their own hash input.
5. The schema, vocabulary profile/version, source CIDs, bounds, formulas,
   abstraction log, and frame-projection profile all participate in the plan
   CID.
6. Content identity is CIDv1/DAG-JSON/sha2-256, matching REF-245.
7. A supplied identity is verified after decoding and cannot override the
   derived identity.

Semantically equivalent JSON and DuckDB records must compile to the same
canonical plan bytes. A semantic change to an operator profile, bound,
premise, source tree, policy, evidence requirement, or abstraction changes the
plan CID and invalidates dependent consistency/conformance evidence.

## Reviewed logic vocabulary

### Vocabulary registry

`formal_logic_vocabulary.py` owns a closed registry of reviewed predicates and
operators. A vocabulary entry records:

- stable operator/predicate ID and profile version;
- logic family;
- arity and operand/result sorts;
- canonical serialization name;
- finite-trace or possible-world interpretation;
- permitted nesting and quantification;
- translation targets and the semantic class of each translation (`exact`,
  `equisatisfiable`, `bounded_abstraction`, `conservative_approximation`, or
  `heuristic`);
- assumptions and closed/open-world policy;
- positive, negative, mutation, and countermodel fixtures;
- maximum assurance permitted for native and reconstructed results.

Only code review can add a vocabulary entry or mapping. Provider-discovered
symbols, model output, source text, and arbitrary strings cannot extend the
registry at runtime.

### Common event-calculus core

The DCEC and TDFOL views share the following typed, reviewed core:

- `happens(event, time)`;
- `initiates(event, fluent, time)`;
- `terminates(event, fluent, time)`;
- `holds_at(fluent, time)`;
- `before(time_a, time_b)` and `at_or_before(time_a, time_b)`;
- `actor_of(event, actor)` and `task_of(event, task)`;
- `depends_on(task, predecessor)`;
- `ready(task, time)`;
- `terminal(task, time)`;
- `goal_satisfied(goal, time)`;
- `evidence_satisfied(requirement, time)`.

These atoms are produced from canonical fields. They are not extracted from an
English sentence.

## DCEC profile

### Scope

The reviewed DCEC profile models cognitive state, norms, delegation, and
execution events needed by the supervisor. It is a deliberately bounded subset
of Deontic Cognitive Event Calculus.

The version 1 modalities are:

| Canonical operator | Typed reading |
| --- | --- |
| `belief(actor, time, formula)` | At `time`, `formula` is in the actor's declared/observed belief state |
| `knowledge(actor, time, formula)` | At `time`, authoritative evidence in the modeled world supports `formula` for the actor |
| `intention(actor, time, formula)` | At `time`, the actor is committed to pursue `formula` |
| `obligation(actor, activation_time, deadline, formula)` | The bearer must make `formula` true after activation and no later than the deadline |
| `permission(actor, interval, formula)` | Policy does not forbid the actor from making `formula` true in the interval; this is not authority or executability |
| `prohibition(actor, interval, formula)` | The actor must not make `formula` true in the interval |
| `delegation(delegator, delegatee, interval, action_scope)` | The delegator grants the delegatee the named, bounded action scope subject to policy |
| `execution(actor, task, event, time)` | The canonical event says the actor executed the task transition at `time` |

Belief, knowledge, and intention are distinct. Belief does not imply knowledge.
Intention does not imply execution. Permission does not imply obligation,
capability, readiness, or proof. Delegation does not imply the delegator owned
the authority, and therefore a separate authority/policy premise is required.
Execution is an event claim and does not imply successful effects unless the
transition model and observed events support them.

### Reviewed DCEC rules

The initial rule templates are limited to supervisor-relevant cases:

- knowledge may use only evidence explicitly classified as authoritative for
  the modeled proposition and current at the referenced time;
- obligation activation and discharge use explicit events and deadlines;
- prohibition conflicts with permission or obligation only when bearer,
  action/formula, scope, world, and time interval overlap under the profile;
- delegation requires a modeled `may_delegate` premise, bounded action scope,
  maximum depth, validity interval, fencing/lease context when applicable, and
  no matching revocation;
- execution requires the actor, task, event, and time to match canonical event
  records;
- a delegated action cannot escape its path, task, capability, evidence,
  expiry, or policy scope;
- a revoked or expired delegation cannot authorize later execution;
- evidence or a successful execution event may discharge a norm only through a
  reviewed discharge rule.

No general necessitation, introspection, omniscience, or belief-to-knowledge
rule is assumed. If a provider uses additional modal axioms, the translator
profile and receipt must identify them, and the result cannot satisfy this
profile until conformance fixtures approve those axioms.

### Free-form input boundary

The following are prohibited inputs to the authoritative DCEC compiler:

- model prose or chain-of-thought;
- Eng-DCEC/GF output derived from user/model prose;
- NLP entity or relation extraction;
- an unreviewed CEC string parser result;
- formula-shaped text from a provider response;
- an operator guessed from a label such as “must,” “knows,” or “plans.”

Those paths may be evaluated in an isolated research/shadow lane, stored as
unverified proposals, and compared with the typed compiler. They cannot modify
the plan, select a norm, or satisfy a gate.

## TDFOL finite-trace profile

### Trace model

A supervisor trace is a finite sequence:

`T = <S[0], E[0], S[1], E[1], ..., E[n-1], S[n]>`

where:

- `n` is explicitly bounded by `max_trace_steps`;
- each `S[i]` is a finite valuation of declared fluents;
- each `E[i]` is an ordered, bounded set of canonical events at step `i`;
- logical time is an integer step or an explicitly normalized integer time
  unit;
- actor, goal, task, event, fluent, and evidence domains are finite and
  enumerated in the plan;
- restart epochs and out-of-order tolerances are explicit rather than inferred.

The bounds record at least maximum trace steps, actors, goals, tasks, events,
fluents, worlds, accessibility edges, quantifier-domain size, graph hops,
graph rows, graph bytes, source excerpts, formula depth, and provider resource
budget. A result that omits its bounds is malformed.

### Reviewed temporal predicates

The initial profile defines:

- `dependency_order(predecessor, task)`: an execution of `task` is legal only
  after `predecessor` is terminal with required evidence;
- `deadline(subject, start, end)`: satisfaction/discharge must occur at a step
  `i` where `start <= i <= end`;
- `always(formula)`: `formula` holds at every state in the enumerated trace;
- `eventually(formula)`: `formula` holds at some state in the enumerated trace;
- `until(left, right)`: within the trace, `left` holds until a state satisfying
  `right`; if no such state is reached, the result is incomplete/violated
  according to the declared strong/weak form;
- `next(formula)`: `formula` holds at the next state, if one exists;
- `liveness(formula, bound)`: `formula` becomes true within the stated future
  step bound;
- `safety(formula)`: no state in the enumerated trace violates `formula`;
- `goal_satisfaction(goal)`: the goal's reviewed satisfaction predicate and
  all mandatory evidence requirements hold at a terminal step.

Deadline bounds are inclusive and normalized to the trace clock. Wall-clock
timestamps alone are not compared until a versioned clock/epoch normalization
maps them into logical time.

### Initial supervisor properties

The profile can express:

- no task executes before all hard dependencies are satisfied;
- no actor acts outside current authority, lease, fencing, or delegation scope;
- at most one accepted live lease owns a task in a state;
- revocation, cancellation, or expiration precedes any prohibited later action;
- proof-required merge events occur only after current required evidence;
- retries and repairs remain below explicit bounds;
- every admitted task reaches an allowed terminal state within a finite bound,
  or the validator reports an incomplete liveness horizon;
- forbidden merge, completion, evidence, and resource-capacity states are never
  reached;
- a goal is satisfied only when every required subgoal, criterion, terminal
  task, and evidence requirement is satisfied.

Safety violations are witnessed by a finite prefix. Finite liveness success is
worded as “satisfied within bound N,” never as unbounded eventual progress.
Failure to find a liveness witness within a truncated horizon is distinct from
a proof that no witness exists.

### Finite result vocabulary

Validation results remain distinct:

- `consistent`: all requested supported checks pass for the recorded bounds;
- `inconsistent`: a contradiction or countermodel is found;
- `violated`: a concrete trace violates a requested property;
- `unsupported`: a formula, mapping, or provider semantic is unsupported;
- `inconclusive`: search completed without a permitted conclusion;
- `incomplete`: finite horizon/domain truncation prevents the requested claim;
- `timed_out`, `cancelled`, `resource_exhausted`, `unavailable`, and `error`.

Only the first state is positive plan-check evidence, and it always retains the
formula set, assumptions, translator profile, provider, and finite bounds.

## Versioned frame-logic projection

### Purpose

The frame projection has two uses:

1. give modal formulas explicit bounded worlds and accessibility relations; and
2. select a small, reproducible neighborhood of relevant evidence for
   validators and model capsules.

It is not an independent source of plan facts and is not a code prover.

### Worlds and accessibility

Projection profile version 1 derives worlds only from canonical plan states:

- initial world;
- one world for each bounded task/event transition state;
- optional countermodel and repair-candidate worlds;
- observed execution worlds used by conformance evaluation.

Each world records its state/trace index, source plan CID, fluent valuation
digest, event references, evidence snapshot references, and whether it is
planned, observed, counterfactual, or countermodel.

Accessibility edges are typed and derived, for example:

- `temporal_next`;
- `dependency_possible`;
- `actor_belief_accessible`;
- `actor_knowledge_accessible`;
- `policy_permitted`;
- `delegation_scope`;
- `counterfactual_repair`;
- `observed_correspondence`.

An edge declares its source record CIDs and derivation rule. The profile never
constructs accessibility from text similarity. It records whether a modal
relation is reflexive, serial, symmetric, transitive, or Euclidean only when
the selected reviewed profile explicitly requires that property.

### Evidence-graph neighborhood

The root set is a sorted collection of exact plan, task, goal, formula,
obligation, tree, symbol, policy, evidence, or counterexample CIDs. Expansion
uses an allowlisted edge-type sequence and deterministic ordering. Limits are
applied before materializing a prompt:

- maximum roots;
- maximum hops;
- maximum nodes and edges;
- maximum rows per node/edge type;
- maximum total serialized bytes and model tokens;
- maximum source excerpts and bytes per excerpt;
- maximum counterexamples and diagnostics;
- source-class and confidentiality allowlists.

Every projection records roots, edge allowlist, limits, truncation reasons,
omitted counts, projection version, source graph CID, and result CID. Changing
one of these changes the projection identity.

Reachability means only that records are connected by a declared evidence or
dependency edge. In particular:

> `task -> changes -> symbol -> covered_by -> obligation -> proved_by -> receipt`
> does not prove that the task's generated code is correct.

The receipt must independently bind the exact candidate tree, AST scope,
template, premises, translator, toolchain, policy, freshness, and required
assurance. A graph edge may locate that receipt; it cannot upgrade or validate
it. Likewise, an embedding hit, inferred alias, path existence, or high graph
score cannot create `proved_by`, `validated_by`, `conforms_to`, or
`satisfies_goal` authority edges.

## Three assurance levels that must not collapse

### Level A: plan consistency

Question: Is the canonical intended-work model satisfiable and free from the
requested dependency, temporal, deontic, and finite-state contradictions under
the recorded assumptions and bounds?

Inputs are the accepted plan CID, reviewed vocabulary/profile CIDs, finite
bounds, and current policy. Outputs include a typed verdict, formulas checked,
assumptions, countermodel/trace when available, provider attempts, and
translation/conformance identities.

This level says nothing about actual execution or source code.

### Level B: plan conformance

Question: Do canonical observed events and evidence match the accepted plan's
authorized transitions and required effects within the recorded observation
and timing bounds?

Inputs are the accepted plan, event log/trace, actor/lease/fencing records,
evidence receipts, policy, and projection identity. It distinguishes skipped,
reordered, unauthorized, failed, overridden, superseded, duplicated, and
unobserved transitions.

This level does not establish that event-producing code is correct, that
unobserved behavior is absent, or that a patch satisfies a code invariant.

### Level C: generated-code assurance

Question: Does the exact candidate tree and AST scope satisfy configured
reviewed code obligations and deterministic validation policy?

This level uses REF-245 `CodeProofObligation`, `ProofPlan`, `ProofReceipt`,
tests/static checks, reconstruction/model-checker receipts, and optional real
attestation. It binds source/tree and toolchain identities.

### Non-promotion invariant

No function may convert a Level A or Level B success into a REF-245
`solver_checked`, `kernel_verified`, or `attested` code result. Completion may
require evidence from all three levels, but it evaluates them as separate
requirements. Overrides remain visible and scoped; they never rewrite an
underlying verdict.

## Prover capability matrix

### Matrix state model

Documentation and package discovery are not executable evidence. The runtime
matrix introduced by REF-279 records each route through these monotonic
readiness states:

1. `absent`;
2. `discovered`;
3. `versioned`;
4. `smoke_tested`;
5. `translation_conformant`;
6. `reconstruction_capable`;
7. `authoritative_for` a named property/profile.

States are property-specific. A Z3 executable may be authoritative for a
reviewed finite SMT constraint profile while unsupported for DCEC modal
semantics. A Lean installation may check a reconstructed theorem but is not
automatically a conformant translator. A provider row can be degraded for one
operation and healthy for another.

Capability reports continue to use the implemented
`available`/`degraded`/`unavailable`/`disabled` health vocabulary for route
discovery. Matrix readiness is a later, stronger artifact. Neither artifact is
a proof receipt.

### Property-specific routing matrix

| Property/profile | Primary routes | Independent check or promotion requirement | Safe fallback |
| --- | --- | --- | --- |
| Finite scalar/set constraints, dependency DAG, capacities | Z3, CVC5 | Translation conformance fixtures; differential/mutation cases; optional Lean/Coq reconstruction for stronger policy | Deterministic Python reference predicate and focused tests; otherwise unsupported |
| DCEC typed plan consistency | Reviewed DCEC/CEC provider, ShadowProver where conformant | Exact typed-AST translation, modal/deontic fixture suite, explicit axiom profile; reconstruct exact supported fragment for stronger claim | Reference bounded evaluator; report unsupported operators, never parse prose |
| TDFOL finite dependency/deadline checks | TDFOL provider, Z3/CVC5 for an approved finite encoding | Round-trip and differential fixtures preserve times, bounds, quantifiers, actors, norms, and trace semantics | Reference finite-trace evaluator; bounded result only |
| Premise selection and first-order candidate proof | Hammer with Vampire, E, Z3, or CVC5 | Exact theorem and premise binding; Lean/Coq/Isabelle reconstruction when policy requires kernel assurance | Retain solver candidate/counterexample; deterministic validation |
| State-machine safety/liveness | TLA+/TLC, Apalache | Generated-model identity, config and bound receipt, golden/counterexample fixtures | Bounded Python state exploration; no unbounded claim |
| Authority, delegation, revocation | Datalog/SecPAL-style evaluator | Reference-evaluator agreement, negative/revocation/confused-deputy fixtures | In-process reviewed policy evaluator; no authority inferred from absence of denial |
| Claim/fencing/receipt/attestation protocol | Tamarin, ProVerif | Exact model/query identity, attack fixtures, correspondence/secrecy semantics | Deterministic protocol tests; unsupported formal result |
| Cross-lane information-flow hyperproperties | HyperLTL tools, AutoHyper, MCHyper | Observation-policy binding, redacted hypertrace fixtures, bounded/self-composition conformance | Non-authoritative bounded self-composition tests |
| Runtime temporal conformance | Runtime MTL monitor | Trace-clock and out-of-order profile, gap/restart/duplicate fixtures | Deterministic event validator; absence of violations is not proof |
| Kernel checking | Lean, Coq, Isabelle | Trusted checker accepts the exact reconstructed statement without admits, unsafe axioms, or theorem substitution | Keep lower-assurance candidate and run required tests |
| Model proof proposal/decomposition | Leanstral or another configured LLM | Always pass output to schema validation and a separate trusted checker; fixed theorem and bounded context | Do not use proposal; deterministic route continues |
| Frame-logic queries | ErgoAI/F-logic for a conformant structural profile | Projection/version binding and positive/negative query fixtures | In-memory bounded graph projection; never code-proof authority |
| Evidence navigation | JSON/DuckDB graph queries | Projection-equivalence tests and exact root/edge/bound identity | Canonical JSON scan within the same hard limits |
| Attestation | Real Groth16 or ProveKit backend | Healthy cryptographic backend, circuit/version, verification key, golden/negative vectors, witness no-leak tests | No attestation; simulated backend is serialization-only |

### Existing capability-provider alignment

The implemented discovery report currently includes `hammer`, `tdfol`,
`external_provers`, `lean`, `leanstral`, `frame_logic`,
`knowledge_graphs`, and `zkp_backends`. It separately reports provider,
executable, package, model, circuit, and optional-dependency health. These rows
are retained as the first routing gate.

The G12 matrix expands coverage to Z3, CVC5, TLA+/TLC, Apalache,
Datalog/SecPAL, Tamarin, ProVerif, HyperLTL/AutoHyper/MCHyper, Coq, Isabelle,
runtime MTL, native DCEC, Hammer/Vampire/E, ShadowProver, Leanstral, and real
ZKP backends. New routes enter only through lazy provider discovery or the
bounded subprocess protocol. The supervisor remains importable and useful when
the parent `ipfs_datasets_py` logic package or any optional executable is
absent.

### Failure and fallback policy

Provider outcomes are not booleans. The matrix and receipt preserve:

- unsupported property/operator/profile;
- unavailable package, executable, model, circuit, key, or optional dependency;
- malformed request/response;
- translation non-conformance or quarantine;
- solver disagreement;
- proved, disproved, counterexample, unknown, or incomplete exploration;
- reconstruction rejection;
- timeout, cancellation, resource exhaustion, provider crash, and network
  denial;
- stale cache, stale capability snapshot, or toolchain drift.

Fallback is selected by policy before execution. A fallback never silently
lowers a protected requirement or changes enforcement to shadow. In shadow
mode it may record deterministic tests or a reference-evaluator result. In
enforcement mode an unmet required assurance fails closed unless a durable,
scope-bounded, expiring operator override explicitly permits progress. The
override does not alter the original verdict.

## Translation and provider trust boundary

Each translation contract binds:

- source plan/formula/profile CID;
- target language and semantic profile;
- translator implementation/version/hash;
- exactness class;
- assumptions and intentionally abstracted features;
- finite domains, times, traces, worlds, and resource bounds;
- target text/model digest;
- conformance fixture-set CID;
- permitted result/assurance class.

REF-280 quarantines a path if it drops or changes an actor, time, quantifier,
modal operator, deadline, norm scope, source premise, bound, or polarity. Known
CEC/DCEC API drift and timing-sensitive cache behavior remain degraded until
the same profile passes deterministic conformance fixtures.

Provider requests use the REF-246 operations (`capability`, `translate`,
`prove`, `reconstruct`, `verify`, and `attest`) through lazy in-process
discovery or bounded subprocess JSON. Requests bind resource budgets,
deadlines, network policy, and unpredictable correlation IDs. Provider output
is untrusted until decoded, size-limited, matched to the request, and evaluated
under the appropriate receipt rules.

## Validation strategy

### Contract tests

The REF-275 focused suite must cover:

- complete `FormalWorkPlan` round trips;
- deterministic identities under map and set-like input reordering;
- identity changes for every semantic field, source CID, profile, and bound;
- claimed-identity mismatch;
- floats, non-string keys, malformed JSON, duplicate keys, unknown fields
  where closed schemas require rejection, and unbounded nesting;
- duplicate/missing references, dependency cycles, self-dependencies,
  conflicting effects, missing initial semantics, and invalid times;
- formula sort, arity, scope, finite-domain, and operator validation;
- DCEC belief/knowledge/intention distinctions and deontic conflicts;
- permission not implying obligation, authority, readiness, or success;
- delegation scope, expiry, depth, revocation, and execution-event binding;
- TDFOL dependency order, deadline inclusivity, safety violation, finite
  liveness witness, incomplete horizon, and goal satisfaction;
- frame-world/accessibility determinism and neighborhood truncation;
- explicit rejection of free-form/model text as an authoritative formula;
- explicit proof-level separation and inability to promote plan evidence into
  code-proof assurance.

### Provider and translation tests

Subsequent matrix suites add:

- bounded capability self-tests with executable/package/model/translator and
  fixture identities;
- round-trip, differential, metamorphic, mutation, and negative translation
  fixtures;
- fake executable/version, malicious output, theorem substitution, dropped
  modal operand, and stale toolchain cases;
- solver disagreement, unknown, incomplete exploration, model-checker bound,
  protocol false-positive, and hypertrace leakage cases;
- reconstruction rejection of `sorry`, `admit`, unsafe declarations, added
  axioms, hidden premises, and changed theorems;
- simulation never satisfying cryptographic attestation;
- JSON/DuckDB matrix and receipt projection equivalence.

### Runtime and end-to-end tests

The end-to-end workflow must seed:

- a consistent plan and successful bounded execution;
- a deontic contradiction before model dispatch;
- an authorization or stale-fencing violation;
- a finite safety counterexample;
- an incomplete liveness horizon;
- a provider outage and policy-selected fallback;
- a code-proof failure after plan-consistency success;
- a runtime conformance failure after code-proof success;
- invalidation and deterministic repair after a source/policy/plan change;
- restart and replay from paired JSON/DuckDB artifacts.

## Security requirements

### Untrusted input and formula injection

All source records, provider responses, cached artifacts, and model outputs are
untrusted until validated. The contract accepts formulas only as typed nodes
from the reviewed registry. It never evaluates Python, Prolog, TPTP, SMT-LIB,
Lean, shell, template syntax, or provider expressions embedded in labels or
metadata.

Generated target-language files live in isolated temporary directories with
bounded filenames and no implicit imports. Provider commands are explicit
argv arrays, not shell strings. Network access defaults to denied. Subprocess
stdout/stderr and diagnostic metadata are size-limited and digested.

### Authority and replay

Actor IDs are not credentials. Authority comes from current signed/canonical
policy, lease, delegation, fencing, and revocation records. Receipts bind the
plan, tree, task, policy, provider, toolchain, translator, bounds, and
correlation identities. Stale epochs, expired delegations, replayed events,
cross-task cache entries, and mismatched trees fail closed.

### Data isolation

Frame/evidence projections enforce source-class and confidentiality allowlists.
They exclude credentials, hidden ZKP witnesses, unrelated lane/worktree data,
unbounded source, raw model reasoning, and raw prover transcripts. Hypertrace
and counterexample persistence applies redaction and byte limits before
writing JSON, DuckDB, logs, or prompts.

### Denial of service

Canonical decoding has limits on bytes, collection sizes, formula depth,
quantifiers, domains, worlds, accessibility edges, trace length, graph
expansion, and diagnostics. Solvers and checkers use wall time, CPU time,
memory, disk, process, output, model-token, and provider-quota budgets. Timeout
and cancellation terminate process groups and retain bounded failure receipts.

## Performance and scheduling

Formal planning should reduce model work without creating nested
oversubscription. All translation, solver, kernel, model-checker, protocol,
hyperproperty, runtime-monitor, LLM, and artifact work enters one top-level
resource scheduler.

Relevant resource classes are:

- plan compilation and projection;
- SMT and ATP solving;
- ITP reconstruction/kernel checking;
- JVM model checking;
- protocol verification;
- hyperproperty checking;
- runtime temporal monitoring;
- LLM proposal/inference;
- test/static validation;
- JSON/DuckDB/artifact I/O.

Independent nodes of the plan/proof DAG may run concurrently. Dependency and
conflict edges remain authoritative. Single-flight keys include every semantic
and trust dimension. A deterministic current cache hit may bypass execution
after freshness and scope validation, while a stale or lower-assurance hit is
retained only for audit.

Measurements are partitioned by property class, profile, translator, provider,
kernel, finite bounds, rollout mode, task risk, and authoritative assurance:

- compilation, projection, queue, solve, reconstruction, validation, merge,
  and persistence latency;
- CPU, memory, disk, child-process, provider-quota, and cancellation use;
- cache hits, stale rejections, single-flight savings, and portfolio
  cancellation;
- context rows, hops, bytes, excerpts, and model tokens;
- defects found before model dispatch;
- accepted-task throughput, false blocks, overrides, and reopen events.

An unavailable or low-value prover lane remains advisory. Rollout expands only
when shadow/canary evidence demonstrates useful defect detection or context
reduction within documented throughput limits.

## Persistence and invalidation

Plans, projections, capability matrices, attempts, receipts, counterexamples,
conformance results, and rollout decisions use paired canonical JSON and
DuckDB projections. JSON is the portable content record; DuckDB is a query
projection and cannot change semantics or identity.

Cache and receipt keys include:

- plan/formula/property/profile identities;
- source tree and AST scope identities where applicable;
- assumptions and premises;
- finite bounds and abstraction log;
- translator and conformance fixture-set identities;
- provider, solver, kernel, model, and toolchain versions;
- policy and resource-budget identities;
- required assurance and freshness rules.

A change to a goal, task, dependency, norm, effect, temporal constraint,
authority record, plan/profile version, tree/AST scope, premise, policy,
counterexample, translator, toolchain, finite bound, or required evidence
invalidates affected downstream records. Invalidation is transitive and
targeted: unrelated plans and obligations remain reusable.

## Downstream implementation roadmap

### G12.S1 — canonical planning

- `REF-275`: land the canonical work-plan and vocabulary contracts in this
  document, `formal_planning_contracts.py`, and
  `formal_logic_vocabulary.py`.
- `REF-276`: compile objective, taskboard, lease/resource, AST, acceptance,
  evidence, and policy records into plans while retaining every source CID and
  abstraction.
- `REF-277`: check dependency, temporal, deontic, authority, lease/fencing,
  evidence, transition, safety, and bounded-liveness properties.
- `REF-278`: build proof-carrying, strictly bounded task capsules for Codex and
  Leanstral; model responses bind plan/task CIDs and remain non-authoritative.

### G12.S2 — executable matrix and portfolios

- `REF-279`: materialize the property-specific runtime matrix and bounded
  self-tests.
- `REF-280`: classify translations, conformance-test them, and quarantine
  legacy/drifting paths.
- `REF-281`: route properties through appropriate multi-prover portfolios and
  retain disagreement/unknown/failure attempts.
- `REF-282`: persist conformance-bound receipts, caches, single-flight state,
  and JSON/DuckDB matrix projections.

### G12.S3 — specialized models

- `REF-283`: generate finite TLA+ models and run TLC/Apalache with explicit
  state and trace bounds.
- `REF-284`: formalize authority, delegation, revocation, leases, fencing,
  promotion authority, and overrides using a reference evaluator plus
  Datalog/SecPAL-style routes.
- `REF-285`: model claims, fencing, receipts, and attestation protocols with
  Tamarin/ProVerif.
- `REF-286`: check cross-lane prompt, log, witness, provider, and cache
  information-flow hyperproperties.

### G12.S4 — runtime evidence and repair

- `REF-287`: monitor canonical supervisor traces with versioned runtime MTL.
- `REF-288`: normalize solver models, unsat cores, DCEC/TDFOL contradictions,
  TLA traces, protocol attacks, hypertraces, kernel failures, and MTL
  violations into a redacted counterexample graph.
- `REF-289`: apply bounded typed repair rules, recompile, and recheck before
  taskboard admission.
- `REF-290`: compare observed events to accepted plans and require separate,
  current plan, conformance, code, validation, protocol, and runtime evidence
  for goal completion.

### G12.S5 — shared resources, adversarial assurance, and rollout

- `REF-291`: admit every route under one resource/process budget.
- `REF-292`: mutate every identity, trust boundary, formula, model, bound,
  receipt, trace, cache, and assurance label.
- `REF-293`: exercise the restartable compile/check/dispatch/verify/merge/
  monitor/repair workflow.
- `REF-294`: benchmark and gate rollout by context reduction, defect detection,
  assurance, resource use, and accepted-task throughput.

## Review checklist and acceptance

The REF-275 foundation is accepted when all of the following are true:

- `FormalWorkPlan` records actors, goals, subgoals, tasks, events, fluents,
  preconditions, effects, norms, temporal constraints, evidence requirements,
  source provenance, abstraction, finite bounds, and deterministic identities.
- Formula construction is closed over the reviewed typed vocabulary; no
  free-form model or NLP text is an authoritative formula source.
- DCEC explicitly distinguishes belief, knowledge, intention, obligation,
  permission, prohibition, delegation, and execution.
- TDFOL explicitly defines dependency ordering, inclusive deadlines, finite
  safety/liveness, incomplete horizons, and goal satisfaction over bounded
  supervisor traces.
- The frame profile versions worlds, accessibility relations, and evidence
  neighborhood bounds.
- The document and implementation state unambiguously that graph reachability
  and context relevance are not code proof.
- Plan consistency, plan conformance, and generated-code assurance have
  separate result types and promotion rules.
- A plan proof cannot construct or upgrade authoritative REF-245 code-proof
  assurance.
- Provider discovery, executable presence, smoke tests, model output, solver
  candidates, bounded results without their bounds, stale caches, and
  simulated ZKP never masquerade as stronger evidence.
- Unsupported semantics and every fallback are explicit, policy-selected, and
  auditable.

The focused validation command is:

```bash
PYTHONPATH=ipfs_datasets_py/ipfs_accelerate_py \
python -m pytest \
  ipfs_datasets_py/ipfs_accelerate_py/test/api/test_agent_supervisor_formal_planning_contracts.py \
  -q
```

