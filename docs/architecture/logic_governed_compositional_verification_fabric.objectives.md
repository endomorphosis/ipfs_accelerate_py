# Logic-Governed Compositional Verification Fabric Objective Heap

Program prefix: `LGCVF`. Root goal: `LGCVF-G000`. Board namespace:
`logic-governed-compositional-verification-fabric-v1`. This is an additive
successor to LGSWF plan
`sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e`;
the predecessor's plan, task history, quarantined attempts, and manual
acceptances are immutable inputs.

Normative plan:
[`LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md`](./LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md).
Executable board:
[`logic_governed_compositional_verification_fabric.todo.md`](./logic_governed_compositional_verification_fabric.todo.md).
Typed projection:
`data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json`.

## Program invariants

- Datasets owns semantic meaning and evidence classifications; accelerator owns
  operational coordination and references datasets artifacts by identity.
- Existing semantic state, capsules, graphs, proof cache, stores, routers, CID
  implementation, and public APIs are extended, never duplicated.
- Exact and conservative fresh summaries may substitute for source;
  heuristic/opaque summaries cannot exclude mandatory source.
- Models and candidates only propose. Independent deterministic checks admit.
- Unknown, stale, unavailable, unsupported, or ambiguous mandatory evidence is
  a blocker, not success.
- Deterministic mode makes zero model, embedding, network, and remote-provider
  calls and, where testable, imports none of those providers.
- One explicit DuckDB writer remains until a separately qualified Quack state
  owner exists.
- No automatic task edits this objective heap, task board, normative plan,
  validator, scheduler policy, benchmark oracle, or protected predecessor.
- Task completion, objective completion, release qualification, and production
  authorization are different state transitions with different authorities.

## LGCVF-G000 Deterministic-first compositional verification control loop

- Status: active
- Parent:
- Depends on: LGCVF-G130
- Priority: P0
- Track: integration
- Direct child goals: LGCVF-G010, LGCVF-G020, LGCVF-G030, LGCVF-G040, LGCVF-G050, LGCVF-G060, LGCVF-G070, LGCVF-G080, LGCVF-G090, LGCVF-G100, LGCVF-G110, LGCVF-G120, LGCVF-G130
- Producing tasks: all tasks in the companion LGCVF board
- Goal: Deliver one content-addressed control loop that closes goals with deterministic semantic and verification evidence before model escalation and independently validates or exactly rolls back every mutation.
- Evidence: current evidence for every child goal, final report, trust/limitations report, paired benchmark, and explicit blocker dispositions
- Validation: `python scripts/validate_logic_governed_compositional_verification_fabric_plan.py --check-all`
- Acceptance: Every child is satisfied with fresh independent evidence or is represented by an explicit non-success blocker; the vertical slice and benchmark are reproducible; release and production are not inferred.
- Conflict policy: Root is tracking-only. It cannot rewrite predecessor history or semantic authority.

## LGCVF-G010 P0 current-tree truth and immutable reconciliation

- Status: completed
- Parent: LGCVF-G000
- Depends on:
- Priority: P0
- Track: audit
- Producing tasks: LGCVF-001, LGCVF-002
- Goal: Record exact topology, revisions, overlays, capabilities, boards, declared/live gaps, and successor ancestry before implementation.
- Evidence: JSON/Markdown gap matrix, trust inventory, topology/revision records, validated successor projections
- Validation: `python scripts/validate_logic_governed_compositional_verification_fabric_plan.py --check-all`
- Acceptance: Datasets is classified as an initialized gitlink rather than flattened; closed statuses are evidence-backed; completed PDR/SCG and PGIR no-go are preserved; LGSWF bytes/history are unchanged.
- Conflict policy: Read protected predecessors; write successor/audit artifacts only.

## LGCVF-G020 P1 canonical compositional contract kernel

- Status: active
- Parent: LGCVF-G000
- Depends on: LGCVF-G010
- Priority: P0
- Track: contracts
- Producing tasks: LGCVF-010
- Goal: Extend datasets' existing software-contract IR with one typed compositional contract and conservative v1 adapters.
- Evidence: contract schema/content identities, round-trip/backward-compatibility tests, opaque-unknown tests
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/test_compositional_contract.py`
- Acceptance: Typed semantic references cover assumptions, guarantees, normal/exceptional behavior, effects, interference, limitations, invalidators, and evidence; prose comparison and unknown-to-true translation are impossible.
- Conflict policy: Datasets owns this semantic type. Accelerator may hold checked references only.

## LGCVF-G030 P2 conservative abstract interpretation

- Status: active
- Parent: LGCVF-G000
- Depends on: LGCVF-G020
- Priority: P0
- Track: abstract-interpretation
- Producing tasks: LGCVF-020
- Goal: Build the minimal sound Python product-domain fixed point with exceptional/effect flow, provenance, summaries, and opaque dynamic fallback.
- Evidence: lattice/property tests, widening convergence, narrowing, exception/interprocedural/dynamic tests, analyzer receipts
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_verification/test_abstract_interpretation.py`
- Acceptance: Supported syntax is explicit; unsupported dynamic behavior widens or falls back; generated invariants remain candidates unless their semantic support justifies more.
- Conflict policy: Add reviewed domains through the existing registry; do not create a second analyzer authority.

## LGCVF-G040 P3 assume-guarantee composition

- Status: active
- Parent: LGCVF-G000
- Depends on: LGCVF-G020, LGCVF-G030
- Priority: P0
- Track: discharge
- Producing tasks: LGCVF-030
- Goal: Discharge component assumptions and guarantees over exact semantic, call, effect, task, and concurrency graphs with explicit SCC closure.
- Evidence: typed discharge receipts, successful/failing/cycle/interference/exception/staleness tests, localized counterexamples
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_verification/test_assume_guarantee.py`
- Acceptance: A cycle cannot prove itself; every consumed assumption and independent closure check is bound to current semantic/contract roots; task completion can depend on discharge.
- Conflict policy: Reuse RelyGuaranteeContract, concurrency IR, graph, obligation, and proof services.

## LGCVF-G050 P4 exact incremental semantic and verification state

- Status: active
- Parent: LGCVF-G000
- Depends on: LGCVF-G030, LGCVF-G040
- Priority: P0
- Track: incremental-state
- Producing tasks: LGCVF-040
- Goal: Compute the minimal changed and reverse semantic/contract/SCC frontier, invalidations, selected checks, dynamic frontier, and unaffected evidence reuse.
- Evidence: exact delta and cache-key receipts, dependency-change tests, stale-root rejection, gitlink/config/schema/test/proof invalidation tests
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_verification/test_incremental_verification.py`
- Acceptance: Every semantic input appears in the cache key; a hit revalidates identity/assurance; large bodies remain CID-addressed while DuckDB stores typed operational references.
- Conflict policy: Join existing index/state/snapshot/cache/store APIs; no new root, graph, or raw-SQL authority.

## LGCVF-G060 P5 reusable incremental SMT

- Status: active
- Parent: LGCVF-G000
- Depends on: LGCVF-G040, LGCVF-G050
- Priority: P0
- Track: incremental-smt
- Producing tasks: LGCVF-050, LGCVF-051
- Goal: Provide exact-fingerprint provider-neutral sessions, stable named assertions, replay, bounded execution, and independently validated models/cores.
- Evidence: Z3/CVC5 capability receipts, session/replay manifests, push/pop/core/stale/crash/timeout/differential tests
- Validation: `python -m pytest -q ipfs_datasets_py/tests/unit/logic/backends/test_incremental_smt.py`
- Acceptance: Mutable processes are never reused by path alone; unsupported/unavailable/timeout stay typed; incremental reuse does not create proof authority.
- Conflict policy: Extend existing SMT backend/public verification API with lazy optional adapters.

## LGCVF-G070 P6 validated interpolation and CEGAR

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G060
- Priority: P1
- Track: cegar
- Producing tasks: LGCVF-060, LGCVF-061
- Goal: Negotiate exact interpolation support, independently validate admitted interpolants, and drive a bounded CEGAR loop with truthful fallbacks.
- Evidence: capability and validation receipts; valid/invalid-vocabulary/invalid-implication/unavailable/fallback/spurious/real/budget tests
- Validation: focused interpolation and CEGAR suites named in the task board
- Acceptance: No interpolant is fabricated; each refinement and terminal disposition binds partitions, vocabulary, theories, provider, bounds, and source identities.
- Conflict policy: Provider capability is not semantic proof; use existing solver/proof services.

## LGCVF-G080 P7 translation receipts and obligation slicing

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G050, LGCVF-G060
- Priority: P1
- Track: proof-compilation
- Producing tasks: LGCVF-070, LGCVF-071
- Goal: Make each source-to-backend stage content-addressed, source-mapped, loss-explicit, replayable, and incrementally sliceable.
- Evidence: stage receipts, source-map/loss/unsupported/stale-proof/reconstruction/dependency tests
- Validation: focused formalization/translation/obligation-slicing suites
- Acceptance: Ordinary generation is never called proof-producing without checkable evidence; downstream assurance cannot exceed a partial upstream translation.
- Conflict policy: Extend current formalization, translation, VC, proof-evidence, and kernel adapters.

## LGCVF-G090 P8 reviewed equality saturation and synthesis

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G040, LGCVF-G060
- Priority: P1
- Track: synthesis
- Producing tasks: LGCVF-080, LGCVF-081
- Goal: Fill verified gaps in the existing repair synthesizer/equality/CEGIS implementation and use counterevidence to bound repair search.
- Evidence: e-class/congruence/rebuild/side-condition/budget/extraction/replay/equivalence/effect/CEGIS tests
- Validation: focused `ProgramRepairSynthesizer@1` and reviewed-operator suites
- Acceptance: Extracted repairs remain candidates and cannot add undeclared files/imports/dependencies/effects; independent translation, contract, security, test, and fixed-point checks admit them.
- Conflict policy: No second synthesizer or model-generated syntax authority.

## LGCVF-G100 P9 proof-carrying artifact and context

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G070, LGCVF-G080, LGCVF-G090
- Priority: P1
- Track: proof-carrying
- Producing tasks: LGCVF-090, LGCVF-091
- Goal: Bind semantic/proof/test/policy/authority lineage into an independently verified artifact and a minimal mandatory-coverage context.
- Evidence: verifier replay, forged/stale/omission/dynamic/prompt-injection tests, context token accounting
- Validation: artifact-verifier and context-compiler suites in the board
- Acceptance: Producer `passed=true` and CID identity are insufficient; context cannot omit open assumptions, obligations, affected interfaces, policy, or validation.
- Conflict policy: Extend existing capsule, context-pack, evidence-bundle, and receipt types if equivalent types exist.

## LGCVF-G110 P10 Planner/Doctor integration and deterministic routing

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G050, LGCVF-G100
- Priority: P1
- Track: supervisor-integration
- Producing tasks: LGCVF-100, LGCVF-101, LGCVF-102
- Goal: Make obligations, discharges, invalidation, context optimization, deterministic repair, typed persistence, and API/CLI/MCP projections part of the existing Planner/Doctor fixed point.
- Evidence: plan-admission, completion, lease/fence/restart/outbox/single-writer/rollback/fixed-point/model-route tests
- Validation: focused Planner/Doctor, control-plane, API/CLI/MCP, and deterministic-import suites
- Acceptance: Deterministic baseline precedes model proposals; writes are isolated and reversible; escalation has a typed reason; transports do not implement semantics.
- Conflict policy: Accelerator stores operational records and datasets evidence references only; use typed repository APIs, never agent SQL.

## LGCVF-G120 P11 complete vertical slice and adversarial qualification

- Status: planned
- Parent: LGCVF-G000
- Depends on: LGCVF-G060, LGCVF-G070, LGCVF-G080, LGCVF-G090, LGCVF-G100, LGCVF-G110
- Priority: P0
- Track: vertical-slice
- Producing tasks: LGCVF-110, LGCVF-111, LGCVF-112
- Goal: Execute and independently verify all 22 required fixture stages through public APIs with zero model use and exact unaffected-evidence reuse.
- Evidence: machine-readable trace/receipt identities, selected/full test receipts, fixed-point receipt, rollback/adversarial results, token/reuse metrics
- Validation: reproducible demo plus focused and adversarial integration commands in the task board
- Acceptance: Real bytes mutate and repair in isolation; only affected semantics/checks replay; unaffected evidence is fresh/reused; the final artifact verifier reproduces acceptance; required tests do not silently skip.
- Conflict policy: Fixture and test artifacts cannot certify production; candidate work cannot edit its judge.

## LGCVF-G130 P12 paired benchmark, release evidence, and successors

- Status: blocked_external_authority
- Parent: LGCVF-G000
- Depends on: LGCVF-G120
- Priority: P0
- Track: evaluation-release
- Producing tasks: LGCVF-120, LGCVF-121, LGCVF-122, LGCVF-123, LGCVF-124
- Goal: Measure baseline/challenger quality, safety, reuse, context/model displacement, resources, time, and cost; issue a truthful go/partial/no-go; preserve human production authority.
- Evidence: preregistered benchmark manifest, paired machine results, cohort separation, external qualification receipt or typed unavailable, final report and minimal successors
- Validation: benchmark and final-report validators named in the board
- Acceptance: Targets are measured rather than hard-coded; production claims use production-authoritative evidence; LGCVF-123 remains blocked until independently authorized.
- Conflict policy: Benchmark policies, holdouts, expectations, and external qualification are protected and cannot be authored by candidate work.
