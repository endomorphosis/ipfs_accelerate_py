# Reviewed bounded task-population admission

`runtime.intent_population_admission` is an opt-in administrative profile for
conditional, alternative and prohibited work, including zero tasks. It preserves
the complete independently signed task universe, original instruction, parsed
rich Intent atoms, reviewed atom/task/property groundings and explicitly allowed
task sets. The older nonempty graph and local admission contracts are unchanged.

This profile does not infer those groundings from free text. A reviewer supplies
them and the existing owner signs them before evidence is collected. The actor
must be `agent`, bound explicitly to that owner's native identity. The supported
syntax is one atom, two-atom AND/OR, or one initial-state conditional; temporal
ordering and permissions/recommendations remain unsupported. At most 16 original
tasks and 32 distinct authorized subsets are admitted. Every original requirement
and potential task must have a grounding. Prohibitions name exact future output
effects and include every original task that declares such an effect.

`author_population_policy` authors that independent policy.
`prepare_population_admission` checks a selected allowed subset against it and
freshly evaluates each declared guard/property through the native finite
Python/Lean checked-cache owner. Unknown guards remain unresolved. A successful
old receipt or caller-provided Boolean never supplies current truth. Selected
tasks must retain their complete task-dependency closure. Selected task bodies,
outputs, validations and acceptance stay unchanged when new native identity
references are assigned. The signed decision retains omitted branches and their
coverage dispositions instead of deleting them from the original requirement
population.

The coverage claim is administrative coverage under independently reviewed
groundings. It is not proof that arbitrary text means the supplied finite
property, proof of future task execution, or general Python equivalence. Proof,
execution, completion, free-text-semantics and source-equivalence authority flags
remain false. Selected native tasks retain the ordinary independently admitted
execution/validation gates. A zero-task decision creates only an inert proposed
native plan with no active head and no worker or completed task.

## Source preservation and finite scope

Finite requests use the existing self-contained `IntegerOffsetContract` profile
and a complete declared list of 1–32 distinct integers. If a current positive
property substitutes for omitted tasks, selected outputs cannot write its entire
source file. The native local contract also forbids changes outside declared
outputs. This is sufficient only for that closed standalone function profile;
imports and cross-file behavior are unsupported by its native checker. It is not
a generic dependency-aware frame proof. A conditional guard is explicitly a
property of the initial source snapshot and is not promoted to a permanent
invariant.

## Durable choice and interrupted replies

A deterministic choice identity binds the complete signed policy and exact native
source head. The existing Intent owner stores one proposed choice plan and its
parent goal. Their transaction also owns all selected task materialization. A
different independently allowed subset cannot subsequently accumulate a second
population under that same policy/source choice. Losing concurrent transactions
must retry or refuse; they cannot commit tasks without their unique choice row.
No new database, proof head or cross-owner transaction is introduced.

A private locked state directory contains an immutable retry-request locator.
It binds the choice, selected set, native owner roots, Intent database identity
and producers. Each retry still reobserves source and performs fresh native
checks. The saved choice is accepted only if those checks and the independently
recomputed admission have the same meaning. The existing successor helper
verifies exact untouched objective, goals, plan, unique active head, stored signed
planning receipt, ready task revisions, contracts, dependencies, outputs,
validations and acceptance. Changed or partial native populations refuse replay.

The original signed decision remains immutable on replay; fresh observations are
returned separately. `decision_replayed` states which path occurred, and
`saved_observations_used_as_proof` remains false. This handles a lost reply after
the native transaction without incrementing task revisions or changing the
selected set. Fresh-process reconstruction uses the existing file-backed owners.

Public decision publication uses the existing `.runtime/repository-finite-handoffs`
directory. A complete readonly temporary is linked atomically to its content name
and the directory is flushed. Exact existing bytes can be replayed. Interrupted
staging can leave an unreferenced complete temporary, but no partial content-named
artifact. Corrupted, writable or substituted existing artifacts refuse recovery.
Filesystem publication and native database commit remain separate operations.

## Qualification boundary

The tests exercise native file-backed Git/source/CAS/Intent owners, actual finite
Python/Lean checks, alternative accumulation refusal, property-source preservation,
conditional truth, prohibition, zero-task decisions, transaction rollback,
lost replies, readonly publication and fresh-process recovery. Controlled-host
runs inject only the foundation resource sampler; they do not replace proof
results. Live default-owner runs and any host resource refusals are reported
separately in the retained evidence.

This bounded profile does not complete TIP-010 (general task discovery) or TIP-011
(general multiple-task execution, per-task context and final aggregation). Those
prerequisites remain open in
[`terminal_bench_intent_planning.todo.md`](../architecture/terminal_bench_intent_planning.todo.md).
RPI-021 production closure must therefore remain open even when this profile's
local controls pass. See the exact run/provenance record in
[`evidence/intent-population-20261002/qualification.json`](evidence/intent-population-20261002/qualification.json).
