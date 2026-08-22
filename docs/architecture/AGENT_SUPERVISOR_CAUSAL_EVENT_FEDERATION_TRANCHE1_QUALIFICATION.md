# CASF Tranche-1 pre-launch qualification

Program `agent-supervisor-causal-event-federation-v1`, root objective
`CASF-G000`, branch `codex/causal-event-supervisor-federation-v1`.

This is a pre-launch qualification record, not a completion or promotion
receipt. It separates implementation and hermetic-test evidence from facts
that must be observed after a real launch. A `pending` or `not executed` cell
must not be converted to zero, pass, or unavailable without a typed producer
receipt bound to the final tree and control-plane generation.

## Decision and scope

- Qualification level: **pre-launch Tranche-1 implementation evidence only**.
- Promotion eligibility: **false**.
- Starting commit and rollback target:
  `84a056e41e48a81d4484be43840196578d6c87da`.
- Ending commit, final tree, live generation, and post-launch receipt bundle:
  **pending**.
- Board truth: 44 tasks are `todo`, zero tasks have an authoritative completion
  receipt, and CASF-000 is the sole dependency-ready task. Source and tests for
  Tranche 1 do not promote CASF-000..012 to completed while their final-result
  identities remain pending.
- Residual implementation program: CASF-013..043. Acceptance receipts for
  CASF-000..012 also remain outstanding.

The closed contracts, normalized migration, registries, authenticated local
bootstrap gateway, atomic outbox, bounded subscription/router path, typed
state-owner wait, and backpressure/dead-letter implementation have local
implementation evidence. Local bootstrap authentication uses a sealed HMAC
profile plus scoped typed grants; a live federation admission receipt is still
pending. The exact server-owned typed event-wait path has focused qualification
evidence. That narrow result does **not** qualify the federation as
event-driven, causally coordinated, multi-supervisor, parallel,
high-concurrency, token-efficient, production-ready, or DuckLake-promoted.

DuckDB remains the intended authoritative transactional store behind one
exclusive Quack state owner. DuckLake remains optional, non-authoritative, and
unimplemented until CASF-031/032; its absence cannot supply or block scheduling
authority.

## Final-report field ledger

| Required field | Current disposition |
| --- | --- |
| Starting commit | Verified: `84a056e41e48a81d4484be43840196578d6c87da`. |
| Ending commit | Pending final-tree/post-launch receipt. |
| Control-plane generation | Pending live owner receipt. |
| Schema fingerprint | Pending final-tree materialization receipt; migration implementation exists. |
| Quack capability and environment | Local compatible preflight and typed-owner implementation evidence exist; exact launched version, extension fingerprint, endpoint identity, and process birth are pending. Remote Quack is unqualified. |
| DuckLake capability and projection state | Typed unavailable/not executed; non-authoritative; CASF-031/032 outstanding. |
| Federations created | Pending live bootstrap receipt; hermetic creation is not counted as live. |
| Supervisors created | Pending live bootstrap receipt; target launch is one bounded coordinator, not a multi-supervisor qualification. |
| Logical agents registered | Pending live bootstrap receipt. |
| Maximum active agents | Pending live observation; no 64-slot claim. |
| Tasks completed | **0 authoritative completions** in the sealed all-`todo` board. |
| Duplicate work suppressed | Not executed/measured. |
| Causal graph identities | Not available; CASF-013 onward. |
| Abstraction maps | Not available; CASF-015 onward. |
| Intervention validation | Not executed; CASF-015 onward. |
| Events committed | Pending live transaction/outbox receipt. |
| Events delivered | Pending live delivery/acknowledgement receipt. |
| Duplicate deliveries | Pending replay measurement; do not infer zero. |
| Lost events | Pending bounded replay/recovery evidence; do not infer zero. |
| Dead letters | Pending live counter/receipt. |
| Supervisor wakeups | Pending live wait/cursor evidence. |
| Unnecessary wakeups | Not measured; causal frontier is residual work. |
| Idle query count | Pending live idle observation. |
| Idle CPU and wakeup observations | Pending live no-event interval; focused tests qualify only the exact wait path. |
| Parallel throughput | Not executed; parallel/high-concurrency admission is closed. |
| Merge conflicts | Not executed/measured. |
| Model calls | Pending live receipt; no model-free idle claim before observation. |
| Remote model calls | Pending live receipt; remote provider execution is not admitted by the bootstrap profile. |
| Input and output tokens | Pending live accounting receipts. |
| Context reuse | Not executed/measured. |
| Proof-cache reuse | Not executed/measured. |
| Validation reuse | Not executed/measured. |
| DuckLake projection lag | Not applicable while projection is typed unavailable; no numeric zero is asserted. |
| Failures and recovery | Negative-path implementation tests exist; live crash/reconnect/replay recovery is not executed for this report. |
| Tests and models executed | Focused hermetic contract, schema, registry, trigger, event/outbox/wait/router, bootstrap, operator, and boundary tests have implementation evidence; exact final-tree test manifest is pending. No foundational model-check result is claimed. |
| Tests and models not executed | CASF-013..043 causal/semantic/parallel/DuckLake/control/formal/chaos/scale/token suites; 12-supervisor, 256-agent, 64-slot, 1,000-task, and 100,000-delivery qualifications; remote/multi-host qualification. |
| Safety-gate results | Partial implementation evidence only. Authorization, closed-contract, state-owner, atomicity, idempotency, fencing, and bounded-wait negative paths are tested locally; conjunctive live, causal, scale, recovery, tenant, secret, idle, and promotion gates remain pending or not executed. |
| Remaining blockers | Canonical CASF-000..012 producer receipts; real launch/health receipt; CASF-013..043; causal frontier and recovery before concurrency; formal/chaos and frozen benchmarks; DuckLake-specific gates for DuckLake promotion. |
| Qualification level | `prelaunch_tranche_1_implementation_evidence_only`. |
| Promotion eligibility | `false` for event-driven federation, causal coordination, multi-supervisor, parallel/high-concurrency, token efficiency, production, and DuckLake promotion. |
| Rollback target | Verified starting commit `84a056e41e48a81d4484be43840196578d6c87da`; no history rewrite. |

## Post-launch fill requirements

The post-launch amendment must bind the exact ending commit/tree, schema
fingerprint, store generation, Quack extension and process birth, authenticated
federation/supervisor/subagent identities, event/outbox/delivery/acknowledgement
identities, durable cursor, current lease/fence, health classification, and
idle observations. It must preserve pending/not-executed values when a producer
did not run. A quiet queue, live process, board status, or passing hermetic test
alone is not completion or fixed-point evidence.

The machine-readable sibling is
`AGENT_SUPERVISOR_CAUSAL_EVENT_FEDERATION_TRANCHE1_QUALIFICATION.yaml`.
