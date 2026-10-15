# Source worker construction checks, 2026-10-14

The existing source broker now captures original owner/source/CAS witnesses and
copied request, limit, path and bound inputs under SERVER, then releases SERVER
for canonical reader initialization and explicit source-reference replay. A
retained-CAS callback can therefore enter the native coordinator without the
old lock-order reversal. Original binding and owner bytes remain the admission
witnesses; callback mutations cannot supply replacement witnesses.

After replay, native task/allocation/nomination fencing runs outside SERVER.
Detached original custody, selected references, complete canonical/allocated
source and protected boundary/parent checks close before a source proxy grant
is issued. The genuine typed proxy client opens outside SERVER, then the same
fences and source checks, including its exact active grant, close again before
the public socket, descriptor or serving thread is published. Runtime proxy
checks remain mandatory; there is no grant-free or partial-descriptor fallback.

The [existing broker](../../../../ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py)
implements this split and shares detached custody checks with the already
qualified launch/delivery paths. The [channel guide](../../source_lookup_worker.md#construct-the-source-session-before-publishing-endpoints)
and [improvement plan](../../reversible_codebase_views_plan.md#source-session-construction-2026-10-14)
keep this constructor boundary distinct from privileged native source-query
transactions, permission delegation and the actual coding-child handoff.

## Controls and qualification

The [unpatched control](baseline-01/commands.json) executed one new case against
published base `e21d6e7b41f14f3994b0fd8556bfcd341d9e888c`. It deliberately
failed with `broker construction replay must release SERVER` before attempting
native reentry, without deadlocking the fixture. Its one failure, zero setup
errors/skips and 24.167 seconds are retained independently of passing counts.
The baseline source snapshot has the initial 26-case test file; two post-client
cleanup cases were appended afterward, before full qualification.

The [first full attempt](qualification-01/commands.json) collected 194 cases
but stopped at its first new positive case with a missing `json` import before
any proxy grant. Its raw failure, JUnit, logs and source snapshots remain
unchanged. The [import correction](import-correction.json) retains the exact
one-import source diff. Runtime behavior and tests were otherwise unchanged for
that correction.

The [selected full record](qualification-02/commands.json) passed all 194
distinct cases in 1346.696 seconds, with zero failures, errors or skips. It
executes the exact [nine-suite roster](targets.json), with the 28 new constructor
cases plus the previous eight full source, delivery, launch, nomination and
allocation suites. Each attempt retains one pytest invocation, exact argv/environment
updates, interpreter/distribution/platform observations, raw outputs, and
production/test/runner snapshots. Prior counts are not aggregated.

The [listed inventory](source-inventory.json) covers 275 dependencies: 224
accelerator and 51 datasets sources. It is not a complete import/toolchain
closure. Actual fixed datasets and kit checkouts remain
`4b50ae4aac65da9166bd3b24608f32ba7e9fb23d` and
`40c98c71b95be750509ffdb21e7e5494d91b4e79`. No decoder, embedding or model
asset is regenerated. Source bytes and execution HEAD/index guards are checked
before and after; shared reference changes are separate unattributed
observations. Tests use unique short private temporary roots, offline settings,
one numerical thread, explicit AST sealing and scrubbed inherited private-owner
environment names. Runtime stores and credential values are not published.

The [incoming compatibility review](incoming-compatibility.json) examines
accelerator main `c91c42512c6c0f25af2cef61b33d2f2b31ea6032`, whose sole
change advances the embedded datasets pin to
`8fd82826e8ee935270457acfeee185054eb44aed`. All 51 listed datasets files
match that revision's Git bytes. The incoming pin is preserved separately from
the actual fixed `4b50ae` test checkout. This does not qualify the incoming
native-engine changes or establish complete dependency coverage.

The [parallel survey](parallel-survey.json) retains the dated review of relevant
peer work and published gaps. The local documentation receipt is
`documentation-checks.json`, with exclusive raw log `documentation-workflow.log`.
The [preflight reproduction](documentation-preflight-01/commands.json) retains
the checker's initial reference-order refusal and its original source snapshot.
The checker now reconstructs observations in the runner's ROOT/datasets/kit
order after JSON loading; fixed HEAD/index and source checks are unchanged.
The release checker independently recounts raw evidence and executes the four
exact reviewed documentation workflow blocks. The final `manifest.json` seals
the phase without hashing itself.

## What the new cases establish

The [28-case suite](../../../../test/api/test_source_worker_construction.py)
uses genuine Git/CAS, native task/claim/grants, Portal allocation/lifecycle and
typed source connections, with authored listener/deployment UID metadata. It
targets the actual canonical initialization load and subsequent refs load,
asserts SERVER is released before reentering genuine native fences, and checks
stable exact source references and owner-side retrieval.

At each load, 12 mutation cases cover selected/omitted originals in both roots,
task body, daemon grant, native claim, allocation lifecycle, a genuine alternate
source binding/store on the same CAS, boundary bytes and protected parent mode.
Pre-grant refusal leaves no proxy, client, listener, descriptor or thread. The
two post-client cases change omitted canonical source or revoke the exact newly
issued proxy. They require exact proxy revocation and transport closure before
endpoints, while preserving the original daemon grant and consumed source
reader/task/allocation markers. Source restoration in fixture cleanup does not
rebind or promote a refused session.

The new cases launch no child and admit no worker peer. Existing selected suites
also launch and reap real inert same-owner Python children. This is no separate
worker-UID, provider/model, target-program, encoder/trainer, solver, Docker or
official-task qualification. Proof, execution, completion and whole-session
token-saving authority remain false. The checks are sequential and provide no
filesystem writer lease.

Initial native reader binding and privileged `source.get` replay still use
their own transaction contracts; this constructor phase establishes no universal
retained-CAS callback safety. The [query-transaction follow-up](native-query-transaction-followup.json)
is a proposal, with no implementation or executed tests. It retains one-use,
grant-bound continuations and byte accounting, and stages cursor publication
until closing source, task and grant checks finish. Initial binding remains a
separate boundary. Preserve the separately planned writable
delegation, authentic native daemon/wrapper/child provenance and signed compiled
context joins. Keep CodebaseIR, IntentIR, SecurityIR, LegalIR and UI/UXIR assets
separate by family, schema/version, decoder task and 8D/384D/768D size, and gate
teacher promotion on actual encoder conditioning and fresh semantic/text
reconstruction. This source-custody change supplies no learned reconstruction
or token-saving result.
