# Logic-governed compositional verification: implementation report

Report date: 2026-08-18 UTC. Status: implementation checkpoint for a hermetic
research/development vertical slice. This report is not a release or production
authorization.

The normative successor plan is
[`LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md`](../LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md).
The machine-readable capability audit and benchmark are
[`current_tree_capability_gap_matrix.json`](current_tree_capability_gap_matrix.json)
and
[`benchmark_result.json`](../../../data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json).
Trust claims are bounded by [`TRUST_AND_LIMITATIONS.md`](TRUST_AND_LIMITATIONS.md).

## Executive disposition

| Predicate | Disposition | Evidence boundary |
| --- | --- | --- |
| Task implementation | **Partial program implemented** | P0 and the P1-P5 vertical spine are implemented; several P6-P12 production-scale tasks remain. Static task-board status is not treated as execution truth. |
| Focused test success | **Passed in the named local suites** | 52 datasets tests and 41 accelerator tests passed with no skips in the commands recorded below. No whole-repository pass is claimed. |
| Root objective completion | **Open** | The one-fixture slice does not satisfy the representative-suite, proof-compilation, full CEGAR, e-graph, transport, external-verification, or operational qualification obligations. |
| Release qualification | **Not qualified** | The paired benchmark disposition is `partial`; context reduction and representative task-class coverage miss their targets. |
| Production authorization | **Not authorized** | LGCVF-123 remains `blocked_manual`; no fixture, test, model, CID, supervisor, or implementation author can issue that authority. |

## A. Exact source revisions and repository topology

The audit and successor were constructed from:

| Repository authority | Identity | Tree | Topology/state |
| --- | --- | --- | --- |
| `ipfs_accelerate_py` superproject baseline | `12c4e8387de4986d38d69534f3d74864e7bb15c1` | `96504e7744d741784bc8076456f4b169dd665ab5` | Immutable LGSWF actual-v1 source checkpoint. |
| `ipfs_datasets_py` construction baseline | `480a1666f144ad606fcb3cacb66e59775f28d0d1` | `6fdc81fabb04a86683d0f26200636fa8f61fd25c` | Mode-160000 Git submodule/gitlink. |
| `ipfs_datasets_py` vertical-kernel revision | `11fa6e8d37c4d90b8d4905572f96ee95b9443448` | `08976767b5e6d915aac0512b49bdda63cfe749e6` | Nested commit `feat(logic): add compositional verification vertical kernel`. |
| `ipfs_datasets_py` launch-validation revision | `cf15d517e9ad9ab67e4bd6be5c621a620bcde387` | `ead65305f261affbe8089bbd759bda32bde6668b` | Nested commit `chore(validation): bind LGCVF scoped dependencies`; adds only a local optional validation profile and exact task-bound contract entries, with no installation or network action. |
| `ipfs_datasets_py` supervisor-produced revision | `c54c1db9a5653f0d76ddbacc23fa2251eb2240b8` | `c58d53ff857802bb4d8a3c612be799a096850801` | LGCVF-051 public-adapter and bounded differential hardening, independently validated before the gitlink advanced. |
| `ipfs_datasets_py` post-merge audit revision | `253a9d89048250e1edf9461df7b82c0bacd4cfa1` | `c22cc0eac453e4c605855e607d5eb6dbc6dd9216` | Removes one unused test import and fixes one import group after a fresh 13/13 gate; no semantic code changed. |
| `ipfs_accelerate_py` first accepted supervisor merge | `62faf5e6a6858147bf27fc0cc09cd97835edf7a0` | `5eea09cdf94ff04175d4b76752b9723b38b9524d` | Merge of accelerator candidate `828d922ae0b155c1dbebcde210b29e294a2b597b`; gitlink now names `c54c1db9...`. |

Datasets is a Git submodule/gitlink, not a subtree, nested untracked history, or
physically flattened package. Datasets-owned semantics were committed and
tested in that repository before the accelerator gitlink was advanced in the
working tree. Earlier unrelated dirty `ipfs_accelerate_py/mcplusplus` and
`ipfs_kit_py` gitlinks were excluded from every LGCVF commit. The authoritative
LGCVF checkout and its datasets submodule were clean at each accepted launch.

The predecessor LGSWF plan remains immutable at
`sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e`.
Its `run-actual-v6` task store reports 47/47 tasks completed, but its root
objective and 16 goals remain open and its qualification is `research_demo`.
Those facts justified an additive successor rather than editing historical
statuses. The successor FormalWorkPlan content identity is
`baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q`.

## B. Pre-existing implemented capabilities reused

The current repositories already provided material implementation; absence
from a suggested path was not treated as absence of behavior. Reused
capabilities include:

- datasets semantic indexing, exact repository deltas, invalidation,
  semantic-state Merkle roots, persistence, and datasets-owned capsules;
- software-contract, concurrency/rely-guarantee, logic-family, translation,
  solver, prover, Hammer, Tactician, and proof-evidence types;
- accelerator repository reasoning snapshots, obligation graph compilation,
  Planner/Doctor context and fixed-point validation;
- `ProgramRepairSynthesizer@1`, its reviewed operator registry, bounded
  equality rewrite mode, and CEGIS path;
- proof-directed retrieval, counterexample normalization, semantic governor,
  verification planning, objectives/task sources, and typed operational
  repositories;
- worktree isolation, exact byte/tree checks, mutation scopes, leases, fences,
  checkpoint/rollback contracts, and Doctor transactions; and
- the existing canonical CID/serialization, assurance distinctions, plan
  authority, model router, DuckDB state API, and object/capsule stores.

No new authority was introduced merely to obtain the requested names.

## C. Verified gaps and implemented closure

The audit matrix contains 27 evidence-backed rows using the required closed
status vocabulary. It is an immutable construction checkpoint, so its
`implemented_not_wired` descriptions are not retroactively rewritten after
implementation. This run closed the smallest useful spine as follows:

| Area | Implemented closure | Remaining boundary |
| --- | --- | --- |
| P1 contracts | `CompositionalContract@1`, typed clauses, roots, effects, interference, limitations, authority/evidence, and conservative v1 adapters | Broader language/schema population and complete production scanning remain. |
| P2 abstract interpretation | Constant × integer interval × nullness × exception × effects product; lattice operations, widening/narrowing, worklist, Python transfers, local summaries, provenance, opaque dynamic fallback | Taint, resources, typestate, authorization, locks, collection shape, richer alias/context sensitivity remain. |
| P3 assume-guarantee | Exact composition graph, edge obligations, SCC rejection or explicit inductive closure, Z3 checks, counterexamples, core/replay receipt | General theorem proving, concurrency theories, and wider contract lowerings remain. |
| P4 incremental verification | Existing semantic delta plus reverse dependency, contract and SCC closure; abstract/contract/session/capsule/proof/test invalidation; exact cache revalidation and unaffected reuse | Complete scanner edges and production-scale precision/recall qualification remain. |
| P5 incremental SMT | Lazy provider-neutral Z3 session, stable named assertions, push/pop, assumptions, model/core, typed results, exact fingerprints, replay manifest, cancellation/close | Process isolation/crash reconstruction, full differential matrix, user sorts and proof reconstruction remain. |
| P6 interpolation subset | cvc5 QF_LIA interpolation with fresh Z3 checks of implication, unsatisfiability and shared vocabulary | Full CEGAR loop and theory/provider matrix remain; unavailable capability must stay typed. |
| Vertical integration | Checked accelerator bridge, deterministic public fixture runner, isolated fault/repair, CEGIS, fixed point, context, artifact producer and independent verifier | The artifact is fixture-scoped, not the complete general proof-carrying-code authority requested by the root objective. |

Confirmed unclosed gaps include stage-by-stage proof-producing or
translation-validating compilation, solver-to-kernel reconstruction, a true
typed e-graph with congruence closure/rebuild, comprehensive CEGAR, general
capsule summary-sufficiency discharge, shared Python/CLI/MCP projection,
representative paired benchmarks, protected external evidence, qualified
Quack multi-writer operation, and production authorization.

## D. Architecture decisions and authority preservation

1. Datasets owns every new semantic type, analysis result, formula, contract,
   discharge receipt, solver/interpolation result and evidence classification.
   Accelerator calls these through a lazy, schema-checked adapter and retains
   operational references only.
2. Existing semantic index/state/capsules are extended. No second repository
   graph, semantic root, capsule compiler, proof cache, plan store, task
   authority, model router, CID implementation, DuckDB abstraction, object
   store, supervisor, or receipt hierarchy was created.
3. Legacy `CallableContract` and `RelyGuaranteeContract` records are adapted
   conservatively. Unsupported prose becomes explicit opaque semantics, never
   Boolean `true` and never string-equality discharge.
4. Incremental SMT is memoization/replay under an exact fingerprint, not new
   proof authority. Solver statistics do not perturb semantic receipt identity;
   transcripts still retain them as observations.
5. cvc5 is only an interpolation producer for the observed QF_LIA fragment.
   Fresh Z3 sessions independently validate the two logical conditions and
   vocabulary restriction. This is solver-checked evidence, not a kernel proof.
6. Repair uses the existing reviewed operator/CEGIS path. It proposes a
   bounded candidate; a fenced isolated Doctor transaction, rescanning,
   redischarge, tests, artifact replay and fixed point admit it.
7. The deterministic fixture imports or calls no model/provider during its
   route. Unsupported dynamic semantics widen or force broader source/review;
   they do not silently disappear.
8. Quack is not qualified for this profile. Operational execution remains one
   explicit DuckDB writer; parallel plan waves do not imply parallel direct
   database writers.

## E. Files changed by repository

### `ipfs_datasets_py` commit `11fa6e8d...`

- `ipfs_datasets_py/logic/software_contracts/compositional.py`
- `ipfs_datasets_py/logic/software_verification/abstract_interpretation.py`
- `ipfs_datasets_py/logic/software_verification/assume_guarantee.py`
- `ipfs_datasets_py/logic/software_verification/incremental_verification.py`
- `ipfs_datasets_py/logic/backends/smt/incremental.py`
- `ipfs_datasets_py/logic/backends/smt/interpolation.py`
- `ipfs_datasets_py/logic/verification_api.py`
- seven focused test modules under `tests/unit/logic/`.

### Accelerator integration

- checked semantic bridge:
  `agent_supervisor/semantic_state/datasets_adapter.py`;
- restored cold Hammer compatibility boundary:
  `agent_supervisor/integrations/ipfs_datasets_logic_provider.py`;
- public fixture runner:
  `agent_supervisor/validation/compositional_verification_vertical.py`;
- A→B→C/config/tests/unaffected fixture under
  `test/fixtures/agent_supervisor/compositional_verification/`;
- adapter, vertical, successor-plan and benchmark tests under `test/api/`;
- successor plan, objective heap, task board, generator, validator and typed
  FormalWorkPlan JSON;
- JSON/Markdown gap matrix, trust statement, vertical/paired summaries and this
  report; and
- `scripts/benchmark_lgcvf_symbolic_displacement.py` plus the admitted machine
  `benchmark_result.json` and its narrow `.gitignore` exception.
- the existing configured-board scheduler profile, checked one-writer
  control-plane materializer, and their focused conformance tests;
- fail-closed nested validation/output/CID projection, scoped dependency
  contract v3, stale-CAS repair, durable retry deferral, and provider-route
  compatibility fixes; and
- LGCVF-051's accepted nested datasets commit `c54c1db9...` and intentional
  accelerator gitlink merge `62faf5e6...`.

## F. Public interfaces added or extended

Datasets' existing `VerificationAPI` now exposes lazy additive operations:

- `analyze_abstract_state(...)`;
- `compile_component_contract(...)`;
- `discharge_assume_guarantee(...)`;
- `plan_incremental_verification(...)`;
- `open_incremental_smt_session(...)`; and
- `compute_and_validate_interpolant(...)`.

Supervisor-completed LGCVF-051 additionally exposes the checked bounded
`run_z3_cvc5_differential(...)` facade and strengthens result/interface/schema
admission for the compositional operations without moving semantic ownership
into accelerator.

Accelerator's existing semantic-state provider forwards those six operations
without reinterpreting returned semantic objects. It pins interface/schema
identities and fails closed on wrong schemas or unavailable datasets imports.

The reproducible vertical command is:

```bash
PYTHONPATH=.:ipfs_datasets_py python -m \
  ipfs_accelerate_py.agent_supervisor.validation.compositional_verification_vertical \
  --output /tmp/lgcvf-vertical.json \
  --benchmark-output /tmp/lgcvf-pair.json
```

The standalone benchmark command and reconstruction check are:

```bash
PYTHONPATH=.:ipfs_datasets_py \
  python scripts/benchmark_lgcvf_symbolic_displacement.py
PYTHONPATH=.:ipfs_datasets_py \
  python scripts/benchmark_lgcvf_symbolic_displacement.py --check
```

No duplicate CLI or MCP semantic implementation was added. The general
shared CLI/MCP projection remains a successor task.

## G. Tests and exact results

Datasets focused command (native datasets pytest configuration):

```text
52 collected; 52 passed; 0 skipped; 0.69 s
```

It covers compositional contracts, abstract interpretation,
assume-guarantee, incremental verification, incremental SMT, interpolation and
the public verification facade. `ruff` passed for the new datasets files, and
focused mypy with explicit package bases/follow-imports skipped passed. A
pre-existing datasets harness issue makes `-p no:cacheprovider` incompatible
with its `tests/conftest.py`; native pytest is the successful product run and
the harness issue was not hidden as a skip.

Accelerator focused command over the datasets adapter, vertical slice,
successor plan, standalone benchmark and existing deterministic Hammer suite:

```text
41 collected; 41 passed; 0 skipped; 15.69 s
```

The standalone benchmark test alone reports 3/3 passed, and `ruff` reports no
issues for its script/test. The real benchmark write completed and its fresh
`--check` reconstruction completed. No required integration test silently
skipped. These focused passes do not claim the entire repository is green.

Additional operational hardening suites run before launch covered the
successor/control projection, database Portal bridge, scoped dependency
contract v3, retry deferral, Grok route/container shape, and exact nested
authority. Named focused aggregates included 104/104 Portal/route tests,
75/75 scoped-contract compatibility tests, 30/30 database-daemon tests, and
15/15 container-helper tests. The independently executed LGCVF-051 board gate
then collected 13 tests and passed 13/13 in 0.62 s. These are local focused
results, not a whole-repository qualification.

## H. Vertical-slice trace and receipt identities

The checked summary in `vertical_slice_result.json` records:

- fixture base commit `ba85c7a2a70c79e7d7a71e261a700be38220fd49`, tree
  `bdc04162de9ec638145bf3a0f68526ef9cffd5eb`;
- vertical result
  `baguqeeraqri3lovzgesdyyb22bx7rv3te5bqqp4pydxbfjqi5tb6jvsvhzyq`;
- proof-carrying artifact
  `baguqeerazpi242wbviqoospasofhwkep4ki5nwlgr5tj2wp7cdy65qsjwfea`;
- baseline/final contract root
  `bafkreiftn4bfwpm3trqimycdivg27c7a64gsum3rzm3zfufscfxf557tva`;
- fault contract root
  `bafkreibrinvpj2sstpc3jwpcfbokepc54m57zjon7g4ox3x5c7fbflvwmq`;
- final semantic root
  `baguqeerajmau5fsvnvbexglpfwkjlkuetrep6c6zlvzmb5rpzdmwkttnffxq`;
- incremental solver receipt
  `bafkreiadvgiqymo6lloxxmlndla4mo42k4bofiqqndodaz5ebzwkbqdduq`;
  and
- independently validated interpolant receipt
  `bafkreihgoyq76uzpvesx367oldgkvp6rsht2ya3hw5crxste5rbtxdy4e4`.

The observed dispositions were baseline `proved`, injected fault `disproved`,
repaired final `proved`, and live fixed point `complete`. The changed A abstract
state was invalidated, while the unaffected module proof evidence was
revalidated and reused. The minimal named unsat core was
`consumer-upper-bound` plus `producer-lower-bound`. CEGIS selected reviewed
candidate `candidate:constant:10`; the isolated transaction committed only
`pkg/module_a.py`. Selected and full tests passed after repair. Independent
artifact replay reconstructed source/state/contract identities, redischarge,
and the selected-test receipt rather than trusting `passed=true`.

Fresh test observations contain elapsed-time-bearing receipt CIDs, so their
artifact/result CIDs can legitimately differ on replay. The benchmark therefore
separates those execution CIDs from the reproducible semantic projection.

## I. Benchmark metrics

Machine result:
`data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json`.
Current report CID:
`baguqeera5eunnikkby5xg4kkkrwrwp7wmo4x26u6xvxerx4lp2lbzhkrdqma`.
Stable reconstructed projection CID:
`baguqeeratddf3jpln5olhturxvhiqbk67foxoqiuq3rpnm23jxg2vocwcjqq`.

| Metric | Baseline | Challenger/result |
| --- | ---: | ---: |
| Raw fixture bytes | 1,545 | typed semantic/context route |
| Context tokens | 387 | 549 |
| Selected tests | 3 | 2 |
| Model calls | 0 | 0 |
| Abstract states reused | not measured | 1 |
| Capsule reuse count | not measured | 13 |
| Unaffected proof/test reuse | not measured | 10,000 bps |
| Deterministic closures | not measured | 1 |
| Solver replay manifests | not measured | 1 |
| Safety-floor violations | — | 0 |
| Critical omissions accepted | — | 0 |

Threshold disposition is `partial`. Safety, accepted critical omissions,
accepted-patch quality, deterministic-route share and unaffected reuse meet
their fixture targets. Median context reduction is 0 bps against a 5,000 bps
target because the tiny typed context is larger than this tiny raw fixture.
Only 3 of 12 required benchmark task classes are observed. Warm-cache model
call reduction is `not_evaluated` because both paths make zero model calls.
Wall-time/resource/cost comparison is not measured separately for the raw
baseline. A target miss remains a result and does not make the command fail.

The evidence cohort is only `hermetic_local_execution`. Simulated,
live-local-model, live-remote-model and production-authoritative evidence are
explicitly excluded and never aggregated.

## J. Model and context displacement achieved

The fixture proves one deterministic closure with zero model invocations and
no newly imported provider modules during the route. It demonstrates exact
semantic/proof reuse and avoids a model entirely, but it does **not** establish
model-call reduction against a nonzero baseline. The typed context is 549
tokens versus the 387-token raw fixture estimate, so no context-token reduction
is claimed. The meaningful positive displacement evidence is 100% reuse of the
seeded unaffected proof/test obligation and no model call for a deliberately
symbolically closable repair.

Supervisor execution is reported separately from that deterministic fixture.
LGCVF-051 required one bounded Grok 4.6 invocation after the analytical kernel
returned `no_analytical_close`; it therefore does not count toward the
deterministic-only displacement result. The invocation was isolated, used a
sealed read/edit/search tool set, and its output became admissible only after
the independently configured validation and merge gates passed.

## K. Remaining risks and production blockers

- The Python abstract semantics and SMT lowering cover a bounded subset;
  reflection, native/dynamic behavior, uncontrolled callbacks/I/O and broader
  theories require conservative fallback.
- Solver evidence is not a kernel theorem. Full translation preservation and
  reconstruction are absent.
- Interpolation is locally qualified only for the observed cvc5 QF_LIA API and
  independently Z3-checked fragment.
- Current equality repair is a reviewed bounded rewrite/CEGIS route, not a
  full typed e-graph.
- The fixture artifact verifier is not yet a general proof-carrying-code
  verifier across all policy/security/supply-chain roots.
- Scanner edge completeness, impact recall and capsule-sufficiency need
  adversarial representative qualification.
- The paired baseline is a raw-context/test-selection projection, not a
  separately timed/resource-metered implementation run.
- A qualified protected external verifier is unavailable. LGCVF-121 remains
  `blocked_external_authority`.
- Continuous Quack multi-writer and live DuckLake are not qualified. One
  embedded DuckDB writer is the truthful operational mode.
- LGCVF-123 is operator-only and remains `blocked_manual`. No LGSWF seal is
  copied or fabricated.
- Run-v16's LGCVF-060 model trace stopped advancing for approximately three
  minutes at a provider-response boundary. The operator stopped that attempt,
  preserved its trace, terminalized the exact dead-owner worktree lifecycle,
  expired its fenced coordination authority, and revised the task to
  `retrying`; no partial result was admitted. This is operational evidence for
  bounded stall handling, not task success.
- No merge, push, publication, automatic promotion, release, or production
  action is authorized by this report.

## L. Next minimal machine-executable tasks

The canonical goal/subgoal/task graph is the validated LGCVF FormalWorkPlan,
not this prose list. The smallest remaining tasks are:

1. `LGCVF-060`/`LGCVF-061`: retain the qualified interpolation subset and add a
   budgeted CEGAR loop with unsat-core/weakest-precondition fallbacks.
2. `LGCVF-070`/`LGCVF-071`: stage-address translation-validation receipts and
   obligation slicing.
3. `LGCVF-080`/`LGCVF-081`: add only the missing typed e-class, congruence,
   rebuild, side-condition and extraction checks to the existing synthesizer.
4. `LGCVF-090`/`LGCVF-091`: generalize proof-carrying artifact verification
   and summary-sufficiency/consumer-strength discharge.
5. `LGCVF-101`/`LGCVF-102`: expose typed operational references and shared
   transport projections without raw SQL or duplicate semantics.
6. `LGCVF-111`/`LGCVF-112`: complete property/adversarial cases, including
   stale authority, protected judge mutation, oscillation, second-order
   findings, exact rollback and gitlink drift.
7. `LGCVF-120`: expand the paired benchmark to all 12 task classes with
   separately executed baseline/challenger time, resources and costs. Preserve
   the current `partial` result as history.
8. `LGCVF-121`: remain externally blocked until an independently authorized
   verifier exists; do not install, probe the network or self-author evidence.
9. `LGCVF-122`: issue an evidence-based no-go/partial/go release disposition.
    `LGCVF-123` remains manual regardless of that result.

Validation entrypoint:

```bash
python scripts/validate_logic_governed_compositional_verification_fabric_plan.py --check-all
```

## Supervisor launch and liveness evidence

The existing `ipfs_accelerate_py.agent_supervisor` was launched directly from
the clean LGCVF branch using the public configured-board loader, preflight,
common-argument compiler, environment projector, and implementation-supervisor
entry point. No second supervisor or scheduler authority was created.

- Scheduler profile:
  `agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json`;
  plan CID `baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q`.
- Preflight: all ten configured checks passed before run-v16 materialization.
  Quack remained unqualified, so the launch used embedded DuckDB, one writer,
  one lane, strict sharding, and no installation or network capability probe.
- Store/generation/schema: `run-v16/control.duckdb`, `lgcvf-run-v16`,
  `datasets-authoritative-operational-v1`. Population root
  `baguqeerakwtl3e44d6qcaccpfkmjgptqppvjhslxjcp5b5lvexvtmgb3iyeq`;
  bootstrap materialization receipt
  `baguqeerawchbjmccz53efetlk3iipecdzfu6etlfqbqoeuvyqkoh6nlrvnpa`.
- Initial read-only verification root:
  `baguqeeraysr3vukwjgpmcdn2km2sncehq2m5esfu5gcdbq54oyjnt7hq37ha`;
  it observed 7 completed, 18 todo, 2 protected blocked tasks and zero
  attempts, claims, leases, provider invocations, or effects.
- Bounded live interval: 2026-08-18T21:23:29Z through
  2026-08-18T21:38:47Z. Supervisor PID 2723374 and daemon PID 2731911 stayed
  live with restart count zero and `control_plane_reload_deferred=false`.
- LGCVF-051 acquired its own claim/attempt/fence and isolated nested worktree.
  Its scoped dependency preflight admitted exactly the declared datasets root,
  `lgcvf-validation` extra and validation command. The provider trace grew,
  made real byte changes only in the two declared datasets files, and the
  independent board test passed 13/13.
- The supervisor committed datasets revision `c54c1db9...`, advanced the
  accelerator gitlink through candidate `828d922ae...` and merge
  `62faf5e6...`, released the worktree lease, emitted `merge_finished`,
  `task_completed`, and selected LGCVF-060. This proves productive progress,
  not merely process liveness.
- A subsequent independent focused audit retained the 13/13 result, cleaned
  two test-import lint findings in datasets revision `253a9d890...`, and
  intentionally advanced the configured gitlink before restart.
- LGCVF-060 acquired a distinct claim, fence, worktree and provider process.
  When its trace stopped advancing, the operator did not trust the generic
  active-worker flag: the generation was stopped, its exact container was
  terminated, its partial result was rejected, and typed reconciliation moved
  the task from `in_progress` to `retrying`.
- Final quiesced read-only verification root:
  `baguqeerapeqyrdbeyof6jnk4htau4sgsmohonx62bdvxuuqyagf27jdasoma`.
  It reconstructed 27 tasks with 8 logical completions, LGCVF-060 retrying,
  zero active claims/attempts/fenced leases, and unchanged stores. Historical
  execution rows and the partial trace remain preserved.

This evidence establishes a productive bounded research/development launch and
safe stall reconciliation. It does not complete the root objective, qualify
multi-writer operation, authorize release/production, or clear LGCVF-121 and
LGCVF-123.
