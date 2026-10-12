# Current header proof admissions in the Terminal census

The Terminal host can explicitly census scoped header proofs through their bound
native admission handles. This connects the [header bridge](header_proof_cache_bridge.md)
to the existing signed task/source/context lifecycle. It preserves the local
proof's assumptions and residual obligations, with every completion, execution,
publication, replacement and omission authority flag false.

This API observes an independently admitted key/receipt population. It does not
automatically discover a proof owner, generate a proof, change a worker prompt
or establish live Terminal coverage or token savings.

## Supply the complete host population

Use the existing `TerminalFormalStatusOwner` with its five original inputs:
current `scoped_analysis`, existing `gate`, complete immutable `keys`, independent
`owner_path_sha256` and independent `expected_key_ids`. Add both optional inputs
for scoped header reuse:

| Input | Required shape |
| --- | --- |
| `header_proof_admissions` | Exact tuple of native `HeaderProofCacheAdmission` handles |
| `expected_receipt_ids` | Independent tuple of `(key_id, generic_receipt_id)` pairs |

The host supplies expected identities from its admitted inventory; the interface
does not infer them from the received handles. Both populations must cover every
requested key, without duplicates, omissions, foreign handles or mixed partial
admissions. An explicitly admitted empty population differs from absent inputs.
The existing bounded key and canonical-byte limits remain enforced.

Pass the owner as `formal_status_owner` with
`include_formal_status_census=True`. The container host forwards the object
locally. Objects, paths, raw IDs, theorem bodies and private seal keys are not
passed to the coding worker. Existing defaults and the native worker input remain
unchanged. A raw Terminal host gate/key tuple cannot count a scoped `@2` receipt
without this typed binding; ordinary legacy `@1` receipts retain their behavior.

Proof creation/import is separate host work. The bridge performs fresh native
kernel replay before returning its admission handle. The Terminal census uses
the already admitted handles and executes no kernel. After a host restart,
explicitly reconnect the original owners, rebuild current preparation and
reimport with fresh native replay before supplying the new handles.

The [durable header inventory](header_proof_inventory.md) retains the admitted
population in the original sealed owner. Its explicit loader requires independent
current preparations and expected inventory/owner/task identities, preflights
every entry before replay, and returns the complete freshly imported handle set.

## Check the exact proof as well as the task

Before proof-owner filesystem observation or callbacks, Terminal validates the
complete public keys and independent key/receipt populations. Each exact handle
must join the original gate, complete key bytes, expected receipt, path digest and
the exact `ScopedDoctorAnalysis` object. This validation follows the existing
initial Terminal preparation/context checks; it is not a promise of zero I/O.

The signed Terminal task, source manifest and context are independently replayed.
Scope roots remain distinct from source-tree and Git-commit identities. Pure
header joins repeat around the host currentness callback. Original handles,
expected populations, analysis and gate remain retained across context entry and
exit, so substituting an equivalent-looking host object cannot replace them.

The generic census checks each returned positive receipt against the independently
expected generic ID, after canonical decoding and assurance/verdict rederivation.
A consistently substituted lookup/render receipt refuses, even if it matches the
same native key. A missing expected positive receipt refuses the bound population;
it does not become a measured zero.

Currentness probes call each handle's native custody/source guard. Rendering uses
the bound handle's stage method. After the final generic denial-state consultation,
the complete guard runs again, including combined native quarantine/tombstone
state and final source/owner checks. Original native SQLite/HMAC lineage, generic
cache/state owner objects, file identities, implementations, executable pins,
profiles, bounds and deterministic receipt bindings remain checked. Recursive
same-handle currentness or stage operations refuse.

The bound nested census uses `supervisor-formal-status-census@2`. It exports a
digest of the independent receipt population and operational counts, retaining
the local scope in the native receipt. Legacy census bindings omit the new
optional field when absent, preserving their canonical payload and digest.

## Retain private lifecycle control

The bound Terminal lifecycle retains its original nested report and private
observation state. Editing the exported status or authority/currentness fields
cannot skip the final native observation. Replacing the nested report refuses
and invalidates the original escaped reference. Fresh results update that original
dictionary in place. Initially refused populations retain their refusal rather
than accepting caller-supplied counts.

On native proof drift, the nested observation becomes refused and its accepted
count becomes null. The outer task/source/context binding can remain current when
only proof custody changed; these are separate observations. Task/source drift,
caller exceptions and cleanup failures preserve the existing lifecycle guards
and invalidate the retained bound proof observation. Required facts remain inline.

## Qualification and measurement limits

Authored tests prepare a real Terminal task with header source before admission,
use an authored router proposal and prover transcripts only during setup, then
forbid prover execution and new proof-owner construction during the census.
They exercise real native SQLite/HMAC and DuckDB owners, two observation passes,
unchanged prompt/worker input, exact receipt population and drift/refusal controls.
Separate bridge regressions execute actual installed Lean/Z3 on authored source.
Neither evidence class establishes live Terminal receipt coverage or a
whole-program theorem.

Counters describe direct adapter operations. `header_admission_current_checks`
counts explicit guard calls; bound rendering performs additional internal checks.
Lookup/render counters do not enumerate all source hashing, lineage work or SQL
transactions, and no counter substitutes for provider-token usage. Canonical
error invalidation restores the initial nested observation's counters; outer
cumulative totals may already include a later completed pass. Refused nested
counters therefore do not guarantee the latest attempted operation costs. Sequential
checks grant no atomic lease or guarantee of future eligibility. Native gate
bookkeeping may write its existing owner. Requested resource bounds are not OS
enforcement, and receipt byte caps do not bound native lineage-prefix work.

Next, admit an actual proof population from the current task inventory and census
accepted receipts plus dependency-complete repeated status. First match the task's
source and property to a reviewed
producer: this handle covers scoped header lemmas. The previously measured
`largest-eigenval` task needs its own reviewed numerical proof admission; a header
handle cannot establish its coverage. Preserve an explicit unavailable result
when no eligible task-bound producer exists. Measure native lookup latency and
lineage cost. Qualify bounded source/receipt expansion, retain property,
verdict, assurance, assumptions, dependency digests, residuals and unresolved
frontiers, then census the complete coding input before a matched trial comparing
total native planning/coding/lookup/repair tokens and reward.

This integration reuses existing model assets. It changes no embeddings,
checkpoints, training or Hugging Face repositories. LegalIR, SecurityIR, CodebaseIR
and IntentIR retain their separate family/schema/version/decoder-task inventories
and parallel 8D/384D/768D paths.
