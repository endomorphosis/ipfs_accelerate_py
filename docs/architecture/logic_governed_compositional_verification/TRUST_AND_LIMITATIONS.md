# Logic-governed compositional verification: trust and limitations

## Scope of this statement

This trust statement binds the audit baseline, not a future branch name or a
working-directory path:

- accelerator commit `12c4e8387de4986d38d69534f3d74864e7bb15c1`, tree
  `96504e7744d741784bc8076456f4b169dd665ab5`;
- datasets Git submodule/gitlink commit
  `480a1666f144ad606fcb3cacb66e59775f28d0d1`, nested tree
  `6fdc81fabb04a86683d0f26200636fa8f61fd25c`;
- a separately identified, bounded, uncommitted datasets P1-P5 overlay.

The overlay is candidate implementation evidence. It is not contained in the
recorded source trees, does not update the accelerator gitlink, and cannot
authorize a task, release, merge, deployment, or production mode. The complete
capability classification is in `current_tree_capability_gap_matrix.json` and
its Markdown projection.

The authority boundary is binding. Datasets owns semantic meaning and semantic
evidence. Accelerator owns operational coordination and records references to
that evidence. Neither side gains the other's authority through a CID,
adapter, transport wrapper, solver answer, model answer, or task status.

## Evidence classes at the audit checkpoint

| Class | What is present | What it does not establish |
|---|---|---|
| Formally/kernel proved | Existing repository facilities can carry kernel receipts for supported Lean/other kernel paths. Lean 4.33 is installed locally. No new P1-P5 overlay result was admitted as a kernel theorem during this audit. | Tool installation, generated proof text, a passing solver result, or a test is not a kernel proof. No blanket correctness claim applies to the repository, compiler, analyzer, supervisor, or vertical slice. |
| Solver checked | The overlay incremental-SMT adapter executes named bounded Z3 4.15.4 checks. The overlay interpolation adapter obtains a cvc5 1.3.3 QF_LIA interpolant and uses fresh Z3 sessions to check both `A implies I` and unsatisfiability of `I and B`; shared vocabulary is checked structurally. | Solver checking is limited to the encoded formulas, theories, assumptions, bounds, provider versions and translation authority. It is not a proof of unencoded Python semantics, whole-program correctness, termination, security, or production fitness. |
| Abstractly interpreted | The overlay implements a product of constants, integer intervals, nullness, possible exceptions and effects, with join/meet, widening/narrowing, a worklist, source provenance and local interprocedural summaries. | It is not a complete Python semantics. It does not make opaque or unsupported behavior exact and it does not itself discharge contracts. |
| Bounded/model checked | Worklist iterations, call strings, solver time, SMT theories, interpolation, rewrite/synthesis search, fixed-point loops and fixture executions are or must be explicitly budgeted. | Exhaustion, timeout, unknown, unsupported and unavailable are not passes. A bounded result cannot be promoted silently to unbounded correctness. |
| Heuristic | Existing premise retrieval, learned/ranked tactic selection, semantic retrieval, context ranking and heuristic/opaque capsules may nominate work or source. | A heuristic never raises semantic confidence, establishes a theorem, excludes critical source, satisfies an assumption, or authorizes a mutation. |
| Test observed | At this checkpoint the six focused overlay test files executed 47 tests with 47 passes, no skips, under Python 3.12.3 and pytest 9.1.1. Existing semantic-index/state, verification, Planner/Doctor and operational suites provide additional prior observations. | Tests observe the cases and environment executed. Fixtures, mocks, expected values, generated tests and test success cannot certify themselves or establish objective/release/production completion. |
| Content addressed | Git commits/trees/gitlinks and canonical CIDs bind exact bytes or canonical records. Overlay contracts, receipts, analysis results and replay manifests define content identities where implemented. | A CID proves identity only. It does not prove semantic correctness, freshness, authority, independence, completeness or authorization. Self-hashes and producer `passed` fields are never sufficient. |
| Runtime observed | LGSWF `run-actual-v6` durably reports 47/47 tasks completed and accepted qualification `research_demo`; one-writer/loopback behavior was observed in its declared scope. | Those observations do not close the open root objective or 16 open goals and do not authorize continuous multi-writer Quack, live DuckLake, LGCVF release, or production use. |

## What is formally proved

Nothing newly implemented in the bounded P1-P5 overlay is claimed as a
kernel-accepted theorem by this audit. Where existing datasets proof APIs return
a kernel receipt, the claim is limited to the exact theorem statement,
assumptions, imported theorem set, translator, kernel and toolchain identities
named by that receipt. Candidate proof text and solver proof objects remain
lower assurance unless reconstructed and checked by the declared kernel.

The new compositional contract schema is a typed representation, not a proof.
The assume-guarantee engine generates and checks bounded obligations over its
supported lowering. Its receipt cannot prove more than the translation and
solver evidence recorded within it. A rejected circular component graph is a
safe refusal, not a proof that no valid invariant exists.

No ordinary source-to-backend code generation stage may be described as
proof-producing unless it emits independently checkable preservation evidence.
The current repository has useful translation receipts and solver/kernel
adapters, but the audit found no complete stage-by-stage preservation chain for
the requested source-to-backend pipeline.

## What is solver checked

The locally exercised overlay supports a bounded structured SMT fragment. Its
incremental Z3 session has stable named assertions and source/obligation
mappings, push/pop, assumption checks, models, unsat cores, typed outcomes and
an immutable replay manifest. The session fingerprint binds relevant provider,
translator, theory, policy, environment, seed and resource configuration.
Mutable process identity or repository path is not a cache key.

The initial Craig interpolation adapter is qualified locally only for the
declared QF_LIA fragment exposed through cvc5 1.3.3. An admitted interpolant
must:

- be computed for the exact A/B partition;
- use only the structural shared vocabulary;
- pass a fresh check of `A implies I`;
- pass a fresh check that `I and B` is unsatisfiable; and
- retain provider, theory, partition and limitation identities.

The independent validation currently uses Z3, so this is solver diversity,
not kernel reconstruction. An unsupported theory, absent interpolation API,
provider error, invalid vocabulary, failed implication or failed
unsatisfiability check yields a typed non-success. cvc5 interpolation is
qualified locally for development evidence; it is not integrated into an
accepted datasets revision and is not production-qualified.

Solver answers do not cover Python reflection, dynamic dispatch, environment
behavior, native extensions, concurrency, resource limits, security policy or
other facts unless those semantics are explicitly and soundly encoded. Model
and unsat-core validation is bounded to the supported formula representation.
Parametric user-defined sorts, proof-object reconstruction, process-isolated
resource control, crash recovery and a full cross-solver session differential
matrix remain incomplete.

## What is abstractly interpreted

The overlay Python interpreter implements a conservative initial product
domain:

`constant × integer interval × nullness × exception state × effect summary`.

It includes bottom/top, partial order, equality, join, meet, widening,
narrowing, bounded fixpoint iteration, source mapping, exception flow and
local interprocedural summaries. Results bind source and analyzer identities,
supported and unsupported constructs, iteration/convergence data, assumptions,
program-point states, derived candidates and open obligations.

The supported transfer subset may justify exact facts only for the semantics it
actually models. Aliasing, descriptors, metaclasses, native extensions,
dynamic imports/code execution, reflection, opaque callbacks, uncontrolled
subprocess/I/O and other unsupported behavior widen to conservative or opaque
state and create an explicit review/full-source frontier. An opaque value is
never translated to `true`, and a proposed invariant or contract remains a
candidate until independently discharged.

Additional requested domains—taint, resources, typestate, authorization,
locks/atomicity and collection shapes—are not part of the admitted minimal
product merely because the architecture names them.

## What is bounded

Bounds are semantic limitations and must travel with receipts. At this
checkpoint they include, depending on the operation:

- abstract worklist iteration, widening threshold, narrowing iterations and
  call-string depth;
- supported Python syntax/transfer functions and local interprocedural scope;
- QF_LIA interpolation and the structured SMT term subset;
- incremental solver timeout, deterministic seed and configured memory field;
- component graph/SCC algorithms and the formulas that can be lowered;
- existing repair operator, equality rewrite and CEGIS candidate/search
  budgets;
- supervisor retry, fixed-point, context and provider budgets; and
- hermetic fixture/test scope.

The current equality-rewrite implementation uses bounded rewriting over opaque
string terms. It is not yet a full typed e-graph with e-classes, congruence
closure and rebuild. Extracted repair candidates remain proposal-only and need
independent semantic, effect, security, proof, test and fixed-point checks.

No timeout, cancellation, unsupported construct, non-convergence, budget
exhaustion, unavailable provider or solver `unknown` can be converted to a
successful disposition.

## What is heuristic

Heuristic or learned logic may rank premises, tactics, source slices, repair
candidates or model contexts. It is advisory. It cannot:

- establish theorem truth;
- strengthen an assurance class;
- satisfy a contract assumption or completion predicate;
- exclude an affected interface, open obligation or dynamic frontier;
- authorize source disclosure, mutation, merge or release; or
- certify its own benchmark or generated test.

Heuristic and opaque semantic capsules are retrieval hints and force raw-source
or broader-verification fallback when mandatory facts are not otherwise
covered. Historical success, a model summary, a docstring or a model-generated
proof cannot change capsule confidence.

## What tests observed

The focused command executed at this audit checkpoint was:

```text
PYTHONPATH=. pytest -q \
  tests/unit/logic/software_contracts/test_compositional_contract.py \
  tests/unit/logic/software_verification/test_abstract_interpretation.py \
  tests/unit/logic/software_verification/test_assume_guarantee.py \
  tests/unit/logic/software_verification/test_incremental_verification.py \
  tests/unit/logic/backends/test_incremental_smt.py \
  tests/unit/logic/backends/test_interpolation.py
```

Observed result: 47 collected, 47 passed, no skipped tests, 0.58 seconds. This
is local hermetic development evidence over an uncommitted overlay. It does not
mean that the overlay is integrated, that all requested adversarial/integration
requirements are complete, or that a release is qualified.

The earlier focused current-tree datasets audit recorded 272 passes with no
skips in 7.14 seconds across the selected semantic/formal baseline. That is
pre-existing implementation evidence, not evidence that the new P1-P5 path was
wired into the accelerator.

Tests still required before objective completion include full public-API
vertical-slice execution, exact mutation/reuse/rollback, zero-provider-import
and zero-model-call proofs, stale/crash/resource paths, protected judge/plan
adversarial cases, Python/CLI/MCP parity and the paired benchmark. A required
integration test that skips cannot count as a pass.

## What is content addressed

The following can bind exact content when reconstructed with their declared
canonical encoders:

- Git blob, tree, commit and recursive gitlink identities;
- semantic repository/state roots and Merkle nodes;
- canonical logic terms, contracts, obligations, capsules and evidence
  references;
- solver assertion sets and replay manifests;
- translation/proof/test receipts; and
- operational manifests that refer to immutable bodies by CID.

Every cache hit must independently reconstruct its full key and re-derive its
evidence class. Relevant source/tree/gitlink roots, parser/analyzer/translator
versions, contracts/policies, environment/configuration, solver/prover/kernel,
domain configuration, assumptions, bounds, target and authority roots must be
fresh. A cache hit cannot upgrade assurance.

Operational attempts, clocks, leases, fences, task state, provider output and
model routing do not belong in the datasets semantic root. They remain
accelerator-owned transactional records referencing datasets-owned evidence.

## What remains unavailable or incomplete

The following are not currently admitted capabilities for the target system:

- a qualified protected external evidence verifier for LGCVF release;
- continuous Quack-backed multi-writer mutation and its complete exact-profile
  state-owner/repository/remote-readiness/zero-bypass chain;
- live DuckLake projection qualification;
- an accepted/wired datasets revision containing the P1-P5 overlay;
- comprehensive production scanner population of contract/proof references;
- a complete interpolation-driven CEGAR loop and theory matrix;
- a true typed e-graph with congruence closure/rebuild;
- stage-by-stage proof-producing or fully translation-validating compilation;
- end-to-end solver-to-kernel reconstruction;
- a canonical full proof-carrying-code verifier spanning all requested roots;
- general capsule summary-sufficiency and consumer-strength discharge;
- CLI/MCP projections for the new API;
- the required full public-API vertical slice and paired challenger benchmark;
  and
- any LGCVF release or production authorization.

These limitations produce typed unavailable, partial, candidate or blocked
outcomes. They are not silently replaced by mocks, fixtures, package presence,
task status, documentation statements or model claims.

## What requires human or operator authorization

The following remain outside the authority of implementation code, tests,
models, candidate patches and this audit:

- accepting any operator-only/manual seal;
- changing protected plan, benchmark, oracle, policy or release evidence;
- authorizing continuous multi-writer Quack or live DuckLake;
- accepting an external evidence-verifier identity and its trust policy;
- promoting research/demo evidence to an internal, external or production
  qualification;
- merging to a protected branch, publishing, deploying or enabling automatic
  promotion; and
- overriding a failed, unknown, stale, unavailable, unsupported, ambiguous or
  incomplete mandatory check.

The live operational database records LGSWF-006 as accepted through its
existing trusted manual-seal history. This audit preserves that durable fact;
it does not recreate, fabricate or retroactively endorse the seal. No new
manual LGCVF task may be marked complete by the supervisor or by locally
generated evidence.

All source writes still require exact current roots, allowed paths/effects,
mutation permit, lease, fence, isolation, checkpoint, independent postcondition
checks and exact rollback. Until Quack is independently qualified, only one
explicit embedded DuckDB writer is permitted. Logical plan waves may express
parallel independence, but they do not authorize multiple direct DuckDB
writers.

## Claim boundary

At this checkpoint the truthful claim is:

> The current repositories contain substantial reusable semantic,
> verification, planning and operational infrastructure, and a bounded
> uncommitted datasets overlay implements focused P1-P5 primitives with 47
> locally passing tests. The overlay is not yet integrated. LGSWF remains
> `research_demo`; its completed tasks do not close its open objectives.
> Continuous Quack, live DuckLake, LGCVF release and production operation remain
> unauthorized.

Anything stronger requires the missing integration, end-to-end execution,
paired measurements, independent evidence replay and explicit authority listed
above.
