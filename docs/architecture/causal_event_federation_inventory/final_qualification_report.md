# CASF final qualification and residual-gap report

Machine report: `final_qualification_report.json`
Schema: `casf/qualification-report@1`
Machine-report SHA-256: `17e3e59d5b2e3fcfb38e673202e89d943828f6f5d5b5a8421dd40cfbd44e1c1c`

## Disposition

**Blocked — quarantine required.** This is a non-authoritative current-tree
observation, not a completion receipt, policy decision, release manifest, or
promotion decision. It creates no authority, state change, completion, or
rollback action.

Promotion and release are both ineligible. All required gates are conjunctive:
missing exact current-tree evidence blocks the relevant claim rather than
allowing a weaker claim to stand in for it.

## Snapshot boundary

| Snapshot | Revision | Tree |
|---|---|---|
| Sealed starting baseline | `84a056e41e48a81d4484be43840196578d6c87da` | `40f0771e77d394ac91d92cc1edb02f7860f6131b` |
| Observed predecessor component tree | `5796d3f78b77b2b6c1c59a2b74c86020a0b141ae` | `14b36ca1f21bfd03dd4b88a7866a0c1a40059249` |

The observed component tree precedes this report artifact. A committed report
cannot embed its own final Git tree identity, so it does not claim exact merged
tree qualification. A registered state-owner producer must create a
qualification identity for the actual merged tree and an independent verifier
must validate it before any qualification or promotion claim is available.

No current state-owner receipt supplied a control-plane generation or schema
fingerprint. The report deliberately records both as unobserved rather than
opening DuckDB or guessing from source files.

## Capability and population state

| Area | Disposition |
|---|---|
| Typed state owner | Not live qualified; no authenticated typed Quack receipt is bound to the observed tree. |
| Quack event wait | Owner-local hermetic evidence only; it is not a remote multi-supervisor qualification. |
| DuckLake | Optional and non-authoritative; no exact-snapshot promotion receipt was supplied. |
| Live scale | Unavailable; frozen scale and token profiles have not run. |
| Federation, supervisor, agent, and concurrency population | Not observed from a current generation and fence-bound state-owner attestation. |

## Benchmark evidence

The benchmark-suite manifest is non-authoritative, not run, and bound to an
older component snapshot (`75f9487ff051ce5defd6171d7b41dd8127a0d59f` /
`4e31236de005816686e68a336adb1a7fe679e6fa`), so it cannot qualify the
observed tree.

| Task | Artifact | Result |
|---|---|---|
| CASF-038 | `casf/idle-benchmark@1` | unavailable / not run |
| CASF-039 | `casf/parallel-benchmark@1` | unavailable / not run |
| CASF-040 | `casf/load-benchmark@1` | unavailable / not run |
| CASF-041 | `casf/token-benchmark@1` | unavailable / not run |

There are no current-tree benchmark result receipts or model-execution receipts
in this report. Frozen recipes, thresholds, source code, test names, and
historical artifacts are not substituted for those receipts.

## Safety and product coverage

The following areas are **not currently qualified**: task/deduplication;
causal graph and abstraction; interventions and independence; events, outbox,
dead letters, and wakeups; idle behavior; parallel throughput and merge;
model/context/token efficiency; proof and validation; DuckLake projection; and
recovery/failures.

Every non-compensable safety gate is unverified for this exact qualification
identity: direct multi-process file mutation, store ambiguity, event loss,
duplicate effects, stale-fence completion, unauthorized creation, tenant
leakage, agent SQL, secret leakage, causal notification loss,
nomination/stale-map authority, cycle/shard corruption, forbidden idle activity,
replay idempotency, ownership/effect/merge corruption, and reduced assurance.

This is an evidence absence, not a claim that any gate passed or failed.

## Claims withheld

The report makes no claim that the system is event driven, causally coordinated,
multi-supervisor, parallel, token efficient, production ready, DuckLake
promotion qualified, or capable of exactly-once network delivery.

## Residual gaps

1. **CASF-043-EXACT-MERGED-TREE-IDENTITY — exact merged-tree qualification identity.** A registered state-owner
   producer and independent verifier must bind the actual merged tree, schema,
   generation, policy, capability, task, attempt, assignment, worktree, and
   fence.
2. **CASF-043-LIVE-TYPED-QUACK — authenticated typed Quack live event wait.** It needs no-lost-wakeup
   evidence; polling or owner-local evidence cannot satisfy this requirement.
3. **CASF-043-SCALE-BENCHMARKS-NOT-RUN — scale benchmarks.** The admitted real-process 12-supervisor and 256-agent
   profiles must run against the exact tree with retained content-addressed
   results and every zero-tolerance gate satisfied.
4. **CASF-043-TOKEN-BENCHMARK-NOT-RUN — token benchmark.** The frozen same-population baseline and 12-supervisor
   comparison must run with receipt-backed identities and meet targets without
   reduced assurance.
5. **CASF-043-CONJUNCTIVE-GATE-DECISION — conjunctive gate decision.** Complete evidence must be evaluated by the
   registered promotion gate and independently validated. This report cannot
   manufacture, apply, or authorize that decision.

## Rollback boundary

The only recorded rollback target is the sealed starting baseline above. This
report authorizes no rollback. Any rollback requires a registered typed
state-owner decision bound to a verified predecessor qualification identity;
history rewriting is prohibited.

## Non-authority boundary

Task-board state, process exit, quiet queues, models, historical receipts,
metrics, retrieval output, and DuckLake projections are not completion or
promotion authority. The report also authorizes no direct DuckDB access,
Quack-to-file fallback, DuckLake scheduling authority, model-created authority,
model-created policy permission, or model-created completion.
