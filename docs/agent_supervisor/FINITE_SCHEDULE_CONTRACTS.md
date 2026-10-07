# Reviewed finite interval scheduling

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Developers extending symbolic task coverage

**Sources:** `planning/intent_interval_schedule.py`, `runtime/doctor_data_contract.py`,
and `ipfs_datasets_py.logic.software_contracts.finite_interval_schedule`

**Last verified:** 2026-10-07

The supervisor can propose one JSON schedule from an explicitly reviewed finite
integer-time model. Shared `ipfs_datasets_py` code compiles the signed job and
resource constraints to bounded QF_LIA, invokes resource-admitted Z3 and checks
the resulting assignment independently. The supervisor replays the finite check
and retains ordinary signed admission, staged execution, validation and
publication gates.

## Declaring the operation

Pass `reviewed_interval_schedule=selector` and explicit `symbolic_operations`
to `build_intent_requirement_contract`. This produces
`intent-plan-requirement-contract@5`; older versions retain their meanings.
The reviewed selector has these exact fields:

```json
{
  "schema": "reviewed-integer-interval-schedule@1",
  "review_ref": "review:authored-schedule@1",
  "operation_id": "operation:schedule",
  "input_path": "input.json",
  "output_path": "output.json",
  "constraint_policy": "finite-half-open-capacity@1",
  "validation_key": "public-schedule-check",
  "semantic_alignment_verified": false,
  "proof_authority": false,
  "execution_authority": false,
  "publication_authority": false,
  "completion_authority": false
}
```

The input must be signed, immutable and distinct from the absent output. One
independently declared task must own one JSON create output and one validation,
with no dependencies. The operation, scope and validation must agree with the
signed declarations. Output directories must already exist in the baseline.
This is an authored interpretation of the instruction; the selector does not
prove that it captures every intended constraint.

The input schema is `finite-interval-schedule-input@1`, containing
`horizon_start`, `horizon_end`, explicit `resources` and `jobs`. Each resource has
an ID, positive capacity and ascending disjoint availability windows. Each job
has an ID, resource, duration, release, deadline and demand. The output schema
`finite-interval-schedule-witness@1` contains ordered assignments with exact
job IDs, starts and ends.

Times are abstract nonnegative integer ticks and intervals are half-open.
A job must fit inside one explicit resource window, even if adjacent windows
touch. Resource demand sums must stay within capacity. The shared contract
fixes bounds at 32 jobs, 16 resources, 16 windows per resource, integer magnitude
1,000,000,000, JSON depth five and 262,144 bytes per document. Duplicate keys,
extra fields, ambiguous identities, floats and Boolean numbers are refused.

## Candidate and index lifecycle

The benchmark's existing `--intent-requirement-contract` input accepts @5.
`prepare_terminal_doctor_dispatch` selects the scheduling route from the signed
contract. Direct native consumers call `prepare_interval_schedule_candidate`
from `runtime.doctor_data_contract` while the Intent database is file-backed.
The state directory must be fresh and outside the target repository.

Only a checked SAT witness can become a candidate. UNSAT, unknown, timeout,
unavailable, cancelled and error remain distinct residual outcomes. UNSAT is a
solver observation, not an independently verified impossibility proof. A forged
SAT witness, changed source/task or altered receipt is refused before staging.
Missing scheduling declarations leave the existing router route available.

The shared checker uses an endpoint sweep independently of the compiler's
capacity-at-start constraints. The supervisor binds its fresh receipt to the
manifest, instruction requirements, task revision, source analysis and output,
hydrates DuckDB/DuckLake observations, then repeats source and receipt checks.
The proof index records finite-check scope with no active kernel-proof receipts.
Canonical sources and task state remain unchanged during candidate preparation.

The existing candidate worker creates the sole output in an allocated worktree.
Native validation, proposal checks, publication and task completion still run.
Its legacy `proof_receipt_id` field carries the bound finite-check receipt;
`proof_scope` explicitly describes its limits. Capability reporting says
`finite_schedule_check`, `kernel_proof=false` and `optimality_verified=false`.
Neither a solver model nor an index row grants authority.

The shared solver uses the existing resource-admitted transport, including
configuration-generation checks, bounded subprocesses and cleanup. Local
qualification uses isolated `local-benchmark@1` scheduler ledgers. Concurrent
changes to another scheduler's configuration must retain a refusal; tests do
not weaken that check or reset another process's ledger.

Initial lexical retrieval can have no symbol-name overlap with the instruction.
It retains the complete hydrated index and emits a replayable empty result for
an exact zero cosine query. Publication uses the same rule; source freshness,
nonzero query normalization and vocabulary pins remain enforced.

## Scope and validation

Authored checks cover real SAT/UNSAT, capacity above one, boundary-touching
intervals, malformed solver models, stale receipts, source/task drift and
producer/index tampering. Public harness tests cover @5 selection, indexed
planning, context reuse and candidate synthesis. The separate native fixture
attaches semantic/world context, validates with an independently authored
checker, publishes, completes, refreshes context and stops with no tracked
process members. It does not build an initial vector retrieval index.

The public profile's structural smoke and the separate native fixture's
semantic checker have different scopes. Intent interpretation remains authored.
No new autoencoder/embedding inference, kernel theorem, optimum, Docker reward,
Terminal-Bench score or matched token saving is claimed. UTC/ICS, recurrence,
timezone conversion, preemption, multiple outputs and implicit preferences
remain outside this profile.

See the [qualification record](evidence/finite-schedule-20261007/README.md).
