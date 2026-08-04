# Formal Verification Tactician and Production Readiness Plan

## Status

Proposed follow-on implementation program for the completed Logic Formal
Verification Expansion (`LFV-G000`). This document records the 2026-07-30
current-tree audit, the remaining production-readiness work, and the design
for a goal-directed proof tactician that can turn an explained end goal into
verified missing-proof obligations.

The executable projection is
`docs/architecture/formal_verification_tactician_readiness.todo.md`. The
canonical goal heap is
`docs/architecture/formal_verification_tactician_readiness.objectives.md`.

## Outcome

The finished platform must let a caller:

1. describe a desired software property or end state in prose or as a typed
   formal goal;
2. inspect and confirm the exact formal interpretation, assumptions, scope,
   bounds, and assurance target;
3. regress that target through program or transition semantics into a
   source-bound AND/OR graph of proof obligations;
4. identify the smallest useful set of missing invariants, contracts, lemmas,
   premises, refinement relations, evidence, semantics, or implementation
   changes;
5. generate and rank candidate ways to close those gaps using the existing
   autoencoder, Leanstral, SymAI, Hammer, proof corpus, caches, and provider
   matrix without granting those advisors proof authority;
6. validate every selected candidate with the appropriate solver, model
   checker, proof kernel, monitor, or independent replay;
7. return either a verified chain to the goal, an explicitly bounded result,
   or an honest `unknown`/`unsupported`/`unavailable` result; and
8. produce minimized, replayable, source-mapped counterexamples whose stated
   minimization guarantee is actually established.

This is a soundness and productization program. It does not replace the
existing logic-family platform, legal evidence tactician, goal-development
provider, formal replanner, or proof-carrying planner. It connects and hardens
them behind one typed goal-directed workflow.

## Audit of completed work

### What is complete and should be preserved

The LFV program delivered all 41 executable goals in 17 supervisor bundles.
Its current-tree completion receipt binds the parent repository, the
`ipfs_datasets_py` gitlink, capability matrix, conformance corpus, examples
manifest, rollout policy, and zero recorded proof-authority boundary
violations.

The resulting foundation is substantial:

- a shared, source-grounded software-verification IR for properties, state,
  traces, programs, heaps, concurrency, refinement, authorization,
  cryptographic protocols, and hyperproperties;
- typed provider, result, translation, cache, receipt, attestation, and
  authority contracts;
- Z3/CVC5, TLA-family, authorization, protocol, hyperproperty, ATP, Hammer,
  Lean/Rocq/Isabelle, runtime-MTL, advisor, cache, corpus, and ZKP adapter
  surfaces;
- a stable Python verification facade plus datasets-side CLI and MCP
  operations;
- supervisor routing, proof scheduling, bounded context, proof-directed
  retrieval, counterexample storage, replanning, and proof-carrying workflow
  machinery;
- explicit separation of model/advisor proposals, bounded solver evidence,
  kernel checking, runtime observation, and receipt attestation; and
- good bounded-process, cache, disagreement, redaction, and canonical identity
  primitives.

A focused readiness run passed 74 integration/API/CLI/MCP/example tests and
skipped 14 optional-tool cases. Live Z3 and CVC5 checks passed. These are good
implementation-level results and establish a real SMT foundation.

### What “LFV complete” does not yet prove

The LFV receipt certifies repository artifacts and their declared contracts.
It is not a production deployment certificate for every external prover.
The capability inventory intentionally uses environment-neutral
`runtime_probed` states and many conformance tests use injected or offline
runners. The follow-on program must preserve that honesty and add a separate,
machine-specific certification result.

Current external-tool evidence is uneven. Z3 and CVC5 are usable in the audit
environment. Java is present. `lean` and `lake` are discoverable, but the
default elan selection names Lean 4.32.2 while only toolchains through 4.32.1
are installed, so three real reconstruction checks attempted a network
download and failed. TLA, protocol, hyperproperty, ATP, and several kernel
tools are absent. Executable presence alone therefore cannot mean usable.

The datasets checkout is clean and its HEAD matches the parent gitlink, but
its local `origin/main` is behind that HEAD. This is a publication/alignment
gate, not a semantic implementation failure. It must be reported until the
intended child history is published; the gitlink must not be rewritten merely
to silence the check.

## Immediate soundness and safety findings

The audit found three P0 trust-boundary defects that precede feature expansion.

### Receipt verification accepts untrusted mappings

`ipfs_datasets_py.logic.verification_api.LogicVerificationAPI.verify_receipt`
currently accepts an empty mapping, arbitrary unrelated data, and a forged
kernel-authority mapping as a successful structurally valid receipt. The
fallback hashes a payload but does not require a closed receipt schema or
validate content identity, source/property bindings, assumptions, bounds,
tool identity, authority, freshness, or independent verification.

The fix must reject unknown, incomplete, forged, stale, cross-property, and
cross-authority receipts. Attestation preparation must not be reported as
proof or as successful authoritative verification.

### Public counterexample explanation leaks raw material

The stable datasets API returns the supplied counterexample in `result.raw`
and again in a witness payload. A direct audit probe preserved a
`hidden_witness` field and token-like values. This bypasses the safer bounded
normalizer in
`ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples`.

All Python, CLI, MCP, supervisor, and model-context projections must use one
canonical redacted envelope. Raw provider output may be retained only in a
private, access-controlled content-addressed artifact store and referenced by
digest and retention policy.

### Structural repair is treated as counterexample closure

`ipfs_accelerate_py.agent_supervisor.planning.formal_replanner` compiles and
structurally validates a repair, checks that the mutation appears to address
the counterexample, and then records the open-counterexample count as changing
from one to zero. It does not rerun the originating verifier.

A repair proposal must leave the counterexample open until a fresh verifier
receipt, bound to the repaired tree, exact property, assumptions, tool,
policy, and bounds, shows that the witness no longer applies. If the verifier
is unavailable, the result remains open or unknown.

## Remaining platform-readiness gaps

### Missing source-to-solver vertical slice

Source adapters can emit program/request shells, verification-condition
generation emits `VerificationConditionSet`, and the SMT compiler consumes
`SmtObligation`, but no production bridge connects the entire sequence. The
platform needs an executable source snapshot → typed program/contracts →
verification conditions → SMT obligations → Z3/CVC5 → replayable
source-mapped witness/receipt path. At least one buggy/fixed program pair must
exercise the path without manually injecting the expected counterexample.

### Provider surfaces are broader than the executable public API

The default stable backend registry currently exposes only Z3 and CVC5.
Portfolio planning does not execute the planned portfolio. TLA,
authorization, protocol, hyperproperty, kernel, ATP, Hammer, and runtime
adapters exist as mostly disconnected leaves. The parent repository MCP also
still exposes older TDFOL/CEC-oriented tools rather than the new verification
operations.

Every provider must be registered lazily through the common protocol, report
absence distinctly, execute only within declared resource and trust policy,
and quarantine disagreement. Python, CLI, datasets MCP, and parent MCP must
have schema and invocation parity.

### Tool isolation is inconsistent

Most new adapters use the bounded tool runner, but some SMT and differential
paths still call `subprocess.run` directly. Every execution and version probe
must share the same argument-array, workspace, timeout, process-tree, output,
memory/CPU, cancellation, redaction, and cleanup policy.

### Packaging and toolchains are not release-qualified

Several new package directories lack `__init__.py`, while legacy
`find_packages()` packaging may omit them. There is no clean wheel/sdist
verification test for the new API. The TypeScript package declares and tests
different `dist` paths. External tools are not pinned in a hermetic offline
environment, as illustrated by the Lean shim mismatch.

The release gate must install built artifacts into empty environments, import
and run every stable surface, validate npm layout, and probe exact offline
tool identities without downloads during verification.

### Examples and readiness metrics are partly synthetic

The examples manifest references seven source paths that are not checked in.
Tests use inline source and manually injected negative witnesses. Benchmark
outcome distributions and some hard-zero security counters are synthesized
rather than derived from live run receipts.

Runnable sources, mutation pairs, generated witnesses, and actual outcome
receipts must replace synthetic readiness claims. Synthetic fixtures remain
useful for deterministic unit tests, but must be labeled as such.

### Source-language coverage is narrow

Python covers a bounded statement subset. JavaScript/TypeScript extraction is
partial and can retain opaque bodies. Rust, Go, Java, C, C++, and WASM do not
have production frontends. Each frontend needs a declared semantic profile,
typed parser, unsupported-feature diagnostics, source spans, and coverage
gates. Adding a language must not imply support for its entire semantics.

## Proof-tactician audit

### Existing components and their correct roles

`processors/legal_data/proof_tactician.py` is a useful legal evidence-search
router. It orders local docket documents, indexes, authorities, legal parsers,
web search, and archives. It does not construct or solve a formal proof graph,
and its legal keyword-based gap focus must remain a domain adapter rather than
becoming the general software proof tactician.

The supervisor already has valuable components:

- the plan evaluator prunes and ranks already supplied complete AND/OR
  alternatives;
- proof context, proof-directed retrieval, and the proof scheduler provide
  bounded, pinned evidence and dependency/resource execution;
- the proof-carrying planner supplies a restartable execution backbone;
- Leanstral goal development safely proposes a decomposition of an already
  formal frozen goal and cannot mutate its formula or assumptions;
- Hammer, proof corpus, caches, autoencoder, SymAI, and retrieval utilities can
  propose or find candidate steps; and
- the formal counterexample layer supplies typed kinds, bounded storage,
  redaction, identities, graph/capsule projection, and repair classes.

These pieces should be composed, not duplicated.

### Missing goal-directed behavior

There is no general workflow that takes an informal end state, produces
candidate formal interpretations, regresses the selected target through
program or transition semantics, discovers missing proofs, validates them,
and extracts a minimal usable proof plan.

Current VC generation fails on missing loop invariants and records unmodeled
calls as unsupported effects instead of returning source-bound proof holes.
Legacy CEC “backward” search delegates to forward rule application, lemma
generation relies on simplified string equality, and TDFOL has no trusted
general backward strategy. These paths must be labeled experimental unless a
typed inversion, unification, abduction, and proof-reconstruction
implementation validates their output.

There is no general weakest-sufficient-assumption finder, interpolation,
Houdini, SyGuS/CHC candidate pipeline, missing-contract/invariant/lemma finder,
or trusted validation loop.

### Counterexample minimization is currently syntactic

The parent counterexample normalizer bounds collections, canonicalizes cores,
removes adjacent stutter, and preserves a trace prefix/suffix. It then marks
every result `minimized=True`. This is useful prompt minimization, but it is
not semantic minimization: no violation oracle is rerun after each removal and
no subset, local, or global minimality guarantee is proved.

The contract must replace a Boolean minimization claim with:

- `none`;
- `normalized`;
- `bounded`;
- `locally_minimal`; or
- `globally_minimal`.

It must record the oracle, exact snapshot/property/assumptions/bounds, budget,
algorithm/version, reduction log or certificate, and replay result. Budget
exhaustion must remain explicit.

## Goal-directed tactician architecture

The core flow is:

```text
Prose/typed request
  → EndGoalSpec candidates + ambiguity report
  → caller-confirmed FormalGoal
  → source/transition model + translation receipt
  → backward regression / weakest preconditions
  → typed AND/OR ProofObligationGraph
  → missing-proof and missing-semantics leaves
  → candidate synthesis/retrieval portfolio
  → trusted candidate validation
  → ranked ProofPlan alternatives
  → proof-carrying execution
  → verifier receipts or canonical counterexamples
  → counterexample-guided refinement
```

### EndGoalSpec

The closed, content-addressed end-goal contract includes:

- repository tree and source snapshot;
- caller text and phrase-to-clause provenance;
- actors, state variables, initial/current state, transitions, environment,
  and allowed interference;
- target state or property and its property class;
- quantifiers and the distinction among existential reachability, universal
  reachability, inevitability/liveness, invariance/safety, termination, and
  refinement;
- trusted assumptions, assumptions that must themselves be proved, and
  explicitly hypothetical assumptions;
- source and observation scope;
- logic family, backend requirements, assurance target, finite bounds, and
  resource policy;
- ambiguity candidates, unsupported semantics, and translation loss; and
- acceptance evidence and expected receipt classes.

Material ambiguity never resolves silently. For example, “the system reaches
ready” may mean that some execution can reach `ready`, every execution
eventually reaches it, or `ready` is an invariant after initialization. The
API returns controlled-English renderings and semantic diffs and requires a
selection or clarification.

### Typed proof holes

The VC and model-compilation layers emit typed holes for:

- loop invariants and termination variants;
- callee preconditions, postconditions, exceptional contracts, and summaries;
- frame, alias, ownership, and separation conditions;
- concurrency rely/guarantee and linearization obligations;
- state-machine invariants and refinement mappings;
- temporal fairness and progress premises;
- protocol trust, freshness, secrecy, and authentication premises;
- information-flow relations and observation policies;
- bridge lemmas and translation preservation obligations;
- missing source facts or evidence;
- unsupported language or logic semantics;
- unavailable tool or reconstruction authority; and
- required implementation changes when the stated goal is false of the
  current program.

Every hole includes a source span, reason, expected proof/evidence authority,
dependencies, and a machine-readable validation recipe.

### Backward proof graph

The tactician constructs a bounded, cycle-safe AND/OR graph:

- AND nodes represent jointly required obligations;
- OR nodes represent alternative proof rules, invariants, backends, or repair
  paths;
- regression uses weakest preconditions, transition preimages, temporal
  progression/regression, rule inversion, and typed unification;
- subsumption, strongly connected components, and explicit budgets prevent
  unbounded recursion;
- every edge names its inference rule and reconstruction method; and
- every apparently solved leaf references evidence with sufficient authority.

The graph can conclude that an end goal is false, inconsistent, unsupported,
or unknown. It must never manufacture a favorable premise merely because that
would entail the target.

### Abduction and candidate generation

Abduction seeks weak, relevant, non-circular missing conditions under a
declared finite theory and budget. Candidate sources include:

- exact proof-corpus and cache matches;
- Hammer and premise-selection retrieval;
- reviewed invariant, contract, frame, temporal, and refinement templates;
- Houdini-style candidate elimination;
- SMT unsat cores, interpolation where supported, and model-derived guards;
- CHC/PDR/IC3 and SyGuS candidates where conformant providers exist;
- legal evidence retrieval for legal-domain premises; and
- autoencoder, Leanstral, SymAI, and other learned proposal/ranking sources.

Learned output is always a candidate. A candidate is admitted only after
parse/type checks, exact source binding, consistency and non-vacuity checks,
non-circularity, relevance, and solver/model-checker/kernel replay. New
environment assumptions remain reviewable obligations and carry an assumption
cost; they are not silently incorporated as proofs.

### Candidate validation and ranking

Validation records the exact tree, target, assumptions, provider/version,
policy, resource bounds, translation receipt, proof/witness artifact, and
checker result. For small finite cases, deletion or core checks establish
local/subset minimality. Otherwise the result says `bounded` or `unknown`.

Complete proof-plan alternatives are hard-pruned for invalid or insufficient
authority and ranked by:

- amount of the goal graph discharged;
- downstream unlock and critical-path reduction;
- proof and assumption authority;
- number and risk of new assumptions;
- source/semantic coverage;
- expected proof cost and provider availability;
- cached independently validated evidence;
- counterexample quality and replayability; and
- fallback and recovery quality.

### Counterexample workbench

Backend-specific semantic reducers must include:

- SMT model projection, don't-care analysis, and MUS/QuickXplain-style
  assumption/core minimization;
- shortest violating state-machine prefix or lasso, with stutter and causal
  slicing;
- earliest runtime temporal violation and event slice;
- protocol dependency/attack-role slice;
- earliest hypertrace observation divergence and observed-field reduction;
- source-aware contract/VC slices; and
- kernel failure classification bound to theorem and artifact identities.

Every attempted reduction reruns the same violation oracle. Explanations show
decoded values, expected-versus-actual state, first violated condition or
divergence, source/AST spans, causal chain, relevant assumptions and bounds,
and separately labeled repair hypotheses. Model-generated prose may summarize
only verified facts.

### Counterexample-guided proof development

The CEGIS/CEGAR loop is:

1. normalize and replay a fresh counterexample;
2. refine the proof graph or candidate invariant/contract/lemma;
3. validate the candidate independently;
4. rerun the exact originating verifier on the repaired tree/property;
5. close the counterexample only when a fresh bound receipt succeeds; and
6. retain disagreement, timeout, unsupported, or changed-bound results as
   open/unknown.

## Public operations

Python, CLI, datasets MCP, and parent MCP expose equivalent typed operations:

- `formalize_goal`;
- `compare_goal_interpretations`;
- `find_missing_proofs`;
- `plan_proof`;
- `validate_proof_plan`;
- `execute_proof_plan`;
- `explain_counterexample`;
- `minimize_counterexample`;
- `replay_counterexample`; and
- `proof_plan_status`.

Existing `check`, `monitor`, `run_portfolio`, receipt, advisor, provider probe,
and installer operations remain compatible. The legal proof tactician remains
a legal evidence-source adapter with a compatibility import.

## Execution waves and parallelism

### Wave 0: fail-closed trust boundary

Build the live readiness baseline, harden receipt/attestation validation,
remove raw public counterexamples, require verifier-backed repair closure, and
move every backend and version probe onto the common bounded lifecycle.

### Wave 1: executable production foundation

Build the source-to-VC-to-SMT vertical slice, lazy executable provider matrix,
clean package/toolchain qualification, checked-in examples, and semantic
frontend profiles. These tasks can proceed alongside the new tactician
contracts after Wave 0 contracts stabilize.

### Wave 2: end-goal formalization

Define the contracts, extract bounded interpretations, expose ambiguity, lower
the selected goal into shared IR, and only then invoke Leanstral or other
decomposition advisors with immutable identifiers.

### Wave 3: missing-proof discovery

Emit typed holes, build the backward obligation graph, implement bounded
abduction and candidate portfolios, validate candidates, rank complete
alternatives, and connect proof execution, retrieval, caches, Hammer, kernels,
and advisor proposals.

### Wave 4: semantic counterexamples and refinement

Implement backend reducers, exact replay, source-aware explanation,
cross-provider semantic equivalence/disagreement, and verifier-backed CEGIS.
This wave shares contracts with Wave 3 but reducer implementations can proceed
in parallel.

### Wave 5: product and release qualification

Expose API/CLI/MCP parity, make supervisor runs restartable, run hermetic
real-tool, adversarial, metamorphic, mutation, differential, packaging, and
performance suites, publish docs and migration guidance, then issue separate
implementation and deployment receipts.

## Test corpus

At minimum, the golden corpus includes:

1. a loop with a missing inductive invariant;
2. a caller with a missing callee contract or frame;
3. a distributed lease/state-machine safety property;
4. a bounded liveness/end-state goal with fairness ambiguity;
5. an impossible target with a minimal inconsistent core;
6. an SMT resource invariant with a projected countermodel;
7. a runtime MTL violation with a shortest prefix;
8. a protocol attack with a dependency slice;
9. a noninterference failure with paired hypertraces;
10. a proof-kernel rejection bound to an exact theorem/artifact;
11. a bridge lemma required across logic translations; and
12. a legal-domain premise that delegates evidence search to the existing
    legal tactician.

Each case has a solvable, mutated, unsupported, and where meaningful
unsatisfiable variant. Public cases may draw from reviewed SMT-LIB,
SV-COMP-style reachability/termination, TLA examples, protocol suites, and
HyperLTL examples, subject to their licenses and checked-in provenance.

## Hard acceptance gates

The program cannot claim production readiness unless:

- malformed, forged, stale, cross-bound, and cross-authority receipts are
  rejected;
- no public/model projection exposes raw source, stdout, secrets, private
  witnesses, credentials, or hidden channels;
- no counterexample closes without a fresh matching verifier receipt;
- every counterexample declares a truthful minimization guarantee and exact
  replay scope;
- every claimed minimized witness still violates the property after each
  accepted reduction;
- every selected proof step has adequate independent authority;
- learned advisors, caches, monitors, bounded checks, tests, and ZKP
  attestations never receive theorem authority by implication;
- the source-to-solver path generates its own witness and source-bound receipt;
- portfolio execution quarantines unresolved disagreement;
- all stable imports and public operations work from clean built packages;
- optional tool absence and offline toolchain mismatch fail explicitly without
  installing or accessing the network;
- real-tool certification is derived from actual run receipts, never
  hardcoded/synthetic outcome counters;
- existing legal, DCEC/TDFOL/CEC, logic API, CLI, and MCP compatibility suites
  remain green; and
- a completion receipt binds the exact parent tree, datasets gitlink,
  toolchain identities, corpus, tests, metrics, rollout policy, and all child
  goal receipts.

## Promotion and rollback

New goal-directed behavior moves through `off`, `shadow`, `assist`,
`auto_safe`, and property-specific `enforced` modes. Shadow mode records
candidate quality without changing plans. Assist mode requires review.
`auto_safe` can admit only deterministic, independently validated candidates
within an allowlisted property/provider/authority policy.

Hard-zero rollback signals are false proof, false counterexample closure,
receipt authority escalation, secret/private-witness leakage, source-binding
mismatch, unresolved disagreement represented as success, or fabricated
readiness evidence. Rollback disables the affected property/provider pair,
quarantines its receipts and cache entries, preserves artifacts for audit, and
falls back to the last reviewed deterministic path.

## Completion definition

The follow-on program is complete when all executable `FVT-*` goals have
current-tree evidence; the golden and real-tool matrices pass at their
declared availability; the end-goal workflow returns a validated proof chain
or honest bounded failure for every corpus case; counterexamples are safe,
truthfully minimized, replayable, and source mapped; public surfaces and
supervisor restart behavior are conformant; and separate implementation and
deployment-certification receipts disclose every remaining unavailable tool,
unsupported semantic fragment, finite bound, and assurance ceiling.
