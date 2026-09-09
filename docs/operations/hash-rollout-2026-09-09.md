# Hashing rollout — 2026-09-09

This records deployment separately from implementation. The earlier
`resource-aware-sealing.md` describes the initial repair and benchmark; its
original "not deployed" status predates this rollout.

## Scope and safety

Four accelerator Git registries contain 184 registrations and 49 different
nine-file source signatures. These include administrative aliases, missing
initializations, quarantined evidence and generated attempt worktrees. They
are not 184 independent running supervisors. Locked, quarantined, frozen and
historical attempt sources were preserved rather than rewriting their evidence.

Only ASEH was running during the initial process census. PCTDD has enabled
ensure/watchdog timers; its services were inactive. No active SAWM, DOEP or PCCE
supervisor was found. Saved `ready` files for older processes were not treated
as proof of live service health.

The selected ASEH, PCTDD, SAWM and DOEP programs have separate DuckDB stores.
Shared resource admission spans updated callers using the same host `/tmp` and
UID. Shared **digest observations** require a common, authenticated typed owner;
the rollout does not silently federate those separate authorities.

## ASEH: full source and owner upgrade

Source root:
`/home/barberb/lift_coding/.worktrees/agent-supervisor-efficiency-and-state-hardening-v1`.

The repair was committed as `f174800dc13ab312f25241695a1a6ab009c4c845` and installed
with merge `7f9a2d7489edfc6002621ff446effe4684ffbd14`. Its first parent is the
previous accepted head `73f9cbda4fe006482ad9d751cf08b6701fda11e1`; its second parent
is the repair. This preserves the existing R45 `sealed_line_descendant`
two-parent admission route. No authorization receipt was invented, no historical
proof replay was claimed, and the database was not rematerialized.

The existing cron-launch flock was held while stopping the exact idle launcher
(PID 2914930, birth ticks 32632841), after checking that its only descendants
were Git observations. Its supported SIGTERM path was used; no provider task
was terminated. The accepted worktree remained clean after the merge.

A pre-existing startup blocker required recovery: owner PID 243206 was dead,
but its marker, two typed sockets and token remained. Under the canonical owner,
migration, intent and database locks, exact inode/ownership/process-birth checks
and refused-listener probes were repeated. Only these four entries were renamed:

- `data/aseh/.control.duckdb.state-owner.json`
- `data/aseh/q/typed-state-owner.sock`
- `data/aseh/q/typed-state-owner-grants.sock`
- `data/aseh/q/typed-state-owner.token`

They remain recoverable in
`data/aseh/q/hash-rollout-stale-owner-5wcxrgjm/` in that worktree. The database,
WAL, canonical locks, saved status and historical receipts were not moved or
deleted. Normal startup subsequently produced its own genuine observations.

The upgraded cron-started capsule was observed at source head `7f9a2d...`,
capsule SHA-256 `ef3839cc5185454c9381320e914a178a1ef0e9feeb9ff597a79fcac425aab8c5`.
Its manifest included `_hash_resources.py`, `shared_hashing.py`,
`hash_observations.py` and SQL migration 0004, all materialized mode 0400.
A fresh physical owner then published generation 111, **schema revision 4**,
PID 3137400 with birth ticks 32696726. TCP port 41487 and both typed sockets
were listening. These were fresh live checks, not the old schema-3 status file.

Whole-supervisor health subsequently passed: the refreshed owner-launch receipt
bound runtime head `7f9a2d...`; all four lanes published that source and fresh
heartbeats. At 01:35 UTC, live health reported `healthy=true`, authenticated
owner/broker, one active task and no failure/stall. A real Grok provider child
was observed. This is a successful full ASEH deployment, not only a database
startup check.

The checkout's new `_hash_resources.py` initially inherited mode 0664 from Git's
collaborative umask. It was hardened to 0644 before capsule admission. A follow-up
fix extends the existing preparation helper so future checkouts do this
automatically. SQL package data already uses exact Git blobs, so it does not
need a blanket source-permission policy change.

A second follow-up adds a single-entry, PID-bound bootstrap Git observation
before the typed owner exists. It checks the full metadata witness on every
call, uses the same non-renewing 24-hour policy, and bypasses itself whenever
the shared owner is configured. Strict/TTL-zero calls still read bytes. Its
26 tests and the 10 hashing-entrypoint tests passed. This follow-up and the
automatic permission fix are prepared on `codex/resource-aware-sealing` but
**not merged into the active source while the Grok task is running**. They
require a later idle admission/restart window; no automatic restart job was
installed.

The ignored cron launcher retains nice 10, best-effort I/O priority 7, CPU
affinity 0–3 and one-thread native-library defaults. Its actual child output now
goes to its existing per-launch log, allowing startup failures to be diagnosed.

## Older base/PCTDD sources: resource-only backports

The base `external/ipfs_accelerate` and PCTDD G9's
`.worktrees/pctdd-g9-orphan-recovery/external/ipfs_accelerate` already contained
uncommitted hashing work. Those edits were preserved. Their old hashing helpers
were backed up verbatim to:

- `.backups/hash-rollout-20260909/base-hash_pressure.py`
- `.backups/hash-rollout-20260909/pctdd-g9-hash_pressure.py`

Both received the common resource implementation and compatible facades,
including explicit capsule membership and module-origin checks for the new
package-root helper. PCTDD's pure SHA/HMAC verification uses shared worker slots;
callbacks and aggregation stay outside those slots to avoid recursive-lock
deadlocks. Base tests: 10 passed. PCTDD tests: 26 passed, including the existing
verification acceptance suite.

These are **uncommitted source integrations**, not a claim that PCTDD's accepted
sealed runtime has been upgraded. Its unrelated large dirty worktree was not
committed, its index was not changed, and its old owner lacks the new typed hash
service. The shared-cache SQL/client were not copied into incompatible owners.

## PCTDD automatic launch protection

Four user-service drop-ins were installed as
`/home/barberb/.config/systemd/user/<unit>.service.d/50-hash-resource-budget.conf`:

- `ipfs-accelerate-pctdd-ensure`
- `ipfs-accelerate-pctdd-g9-watchdog`
- `g9-live-quack-owner`
- `pctdd-g9-quack-owner`

Each supplies nice 10, best-effort disk priority 7, CPUs 0–3, one-thread native
pool defaults, a two-worker hash ceiling and 86400-second cache policy.
`systemd-analyze --user verify` and daemon reload succeeded; effective properties
were checked. None of the four services was started, stopped or restarted.
Cache environment settings have no effect until compatible code is installed.

## Other selected programs

DOEP and PCCE received committed resource-only backports through compatible
clean source descendants with matching outer accelerator gitlinks, without
changing protected planning/configuration receipts. Their original workspace
fingerprint encoding is retained. They do not gain the ASEH hash-owner API
merely by gaining resource admission.

| Program | Accelerator commit | Outer gitlink commit |
| --- | --- | --- |
| DOEP | `c539908e0a54b17248c003d7560a103655db5909` | `77bd271db4594436b4cceb2c5286e16657131076` |
| PCCE r6 | `07e0e24f52563623645daf5653803bc6f2798493` | `520cbd15389981de72592480273275e111b01e1d` |

Both selected accelerator and outer worktrees are clean, on their required
outer branches. Focused preparation tests passed (49 DOEP, 48 PCCE); both
existing sealed-runner startup checks passed. Deployed-source focused reruns
passed 48 DOEP and 47 PCCE cases, excluding the already exercised isolated
capsule case. PCCE's older assertion that optional `multiformats` is absent also
fails on its unchanged baseline because `python -I` retains global site packages;
the new stdlib-only resource-capsule check uses `-I -S` and passes.

All nine changed/new deployed runtime Python files were narrowed from mode 0664
to 0644, and each matching capsule file verifier accepted them. Their operators
do not automatically perform this source hardening: future collaborative-umask
checkouts need `umask 022` or equivalent targeted permission correction. Git does
not encode the group-write difference. No whole DOEP/PCCE supervisor or database
was started to claim live readiness.

DOEP's zero-byte index lock dated August 31 had no Git process or open holder.
After exact metadata rechecks it was preserved, rather than deleted, at
`/home/barberb/lift_coding/.git/worktrees/portfolio/index.lock.pre-hash-resource-rollout-20260909`.
Only the new accelerator gitlink was staged in the outer commit. No other index
or lock recovery was performed.

SAWM's configuration explicitly requires a new operator source seal for a changed
revision, and protects both route and Grok runner source. Its existing retained
successor is not a generic repair authorization. Its launchable source remains
unchanged; a separately admitted source-successor update is still required.

## Authorized continuation: fresh observations after 04:37 UTC

The operator authorized preparing the SAWM successor seal and a clean PCTDD
deployment alongside its unfinished checkout. The earlier process census must
not be reused as a current idle/startup authorization:

- ASEH remains at `7f9a2d...`, with a newer owner since the initial deployment.
  Fresh generation 115/schema 4 health admitted its owner, broker
  and scheduler, with one active task and a live Grok provider. The bootstrap
  follow-up is committed as `91e5fb438d3e2b57dfc7431579de466b702e1687` but remains
  undeployed. No supported finish-current-task pause was found; SIGTERM would
  interrupt work, so neither the running source nor its process was changed.
- PCTDD now has a live watchdog, state owner, master and lanes. Its accepted
  runtime, launcher paths, store and unfinished checkout are not being replaced
  while those processes are active. Clean preparation uses immutable revisions
  separately from that checkout.
- SAWM's selected checkout was initially clean at `3a3e4dabe`, then another
  writer modified `scripts/materialize_semantic_addressed_world_model_program.py`
  during inspection. This rollout did not make that edit. Source/seal preparation
  must remain separate until that concurrent sealing work is reconciled.

A live pressure probe returned `(1, "cgroup_cpu_pressure")` and a one-worker
ceiling. The 78 bootstrap-cache, hashing-entrypoint, resource-budget and worker-
slot regression cases passed again (one Python multithreaded-fork deprecation
warning). These are verification results, not evidence that the pending follow-up
has been deployed or that unrelated Grok sessions are resource-admitted.

### SAWM clean source preparation

Preparation root: `.worktrees/sawm-resource-hashing-budget`, branch
`codex/sawm-resource-hashing-budget`, commit
`1a625cbc300dd5392c1423a942277e082009c430`, based on immutable `3a3e4dabe`.
The seven-file change adds the same resource budget and focused regression
tests, admits the helper into the capsule/origin checks, and wraps the unchanged
workspace fingerprint byte stream. Forty-nine focused tests passed, including
isolated resource-capsule and sealed-runner checks. The preparation is clean;
its runtime Python files are mode 0644.

The production source-capsule materialize/seal/verify APIs also passed against
that exact preparation commit and tree `a2af866df222c497d1f5329972730459a4bad52b`.
The generated capsule CID is
`sha256:950a091dfd1e1470e7c17a306336382973f4989c028162aa41abb02de498e9c3`;
its archive digest is
`sha256:7ee92e24d55019e1e275cca44a174d0920f004286d29a083efe54ae1021ba86b`.
The real write-sealed descriptor was verified and closed. This capsule was not
installed into the selected supervisor or presented as an operator receipt.

This is a source preparation, **not an operator-admitted successor deployment**.
The selected SAWM materializer, operator and both board/dependency validators
were being changed concurrently by another writer. Those changes were not
overwritten, committed or included in this preparation. Its new operator source
seal must be created against their stable, reviewed authority; an ordinary
source-capsule hash cannot substitute for that seal.

### Native thread observations

The active PCTDD watchdog has the installed nice-10, best-effort I/O-7 and CPU
0–3 service policy. The sampled owner and master also had nice 10 and affinity
0–3. The owner showed 149 threads, but an initial sample found 147 waiting on
futexes, one on accept and the main thread sleeping. A later approximately
67-second sample consumed about 21 CPU-seconds across all its threads, not
149 continuously busy hashing workers.

An isolated probe with CPU affinity 0–3 and native one-thread defaults increased
from one thread to 20 on importing DuckDB alone; an explicit one-thread database
connection did not remove that ambient pool. Thread counts therefore must not
be conflated with the active hash-worker budget. Two temporary Quack transport
client connections also lacked explicit DuckDB thread settings; bounding those
connections is a separate preventive source change, not a claim to eliminate
all library-created threads or identify every source of current CPU pressure.

Quack's `v1.5-variegata` HTTP server source creates a fixed 128-worker pool and
one listener; its comments explain that undersizing a per-keepalive-connection
pool can deadlock catalog and scan clients. That is consistent with the observed
149-thread composition, rather than evidence of 149 hash workers. No ad hoc
native HTTP-pool reduction was made. See the
[versioned upstream implementation](https://raw.githubusercontent.com/duckdb/duckdb-quack/v1.5-variegata/src/quack_http_server.cpp).

A separate isolated comparison did demonstrate the temporary-client issue:
after importing DuckDB, an uncapped connection increased the process from 20
to 39 threads and reported SQL `threads=20`; a capped connection retained 20
OS threads and reported SQL `threads=1`. The pending ASEH repair and clean
PCTDD preparation therefore set `config={"threads": "1"}` at creation of their
temporary Quack attach and live-query clients. This avoids creating that
additional hardware-sized query pool; it does not remove DuckDB's import-time
pool or Quack's HTTP workers.

The ASEH temporary-client changes passed 11 new focused tests and 103 existing
transport/server tests, including real default-transport readiness. They remain
preparation changes until explicitly admitted into a new live source capsule.

At 04:47 UTC, the earlier ASEH process snapshot was no longer current: generation
115 and its provider had exited, and cron had started a new launcher/owner
(generation 116) at the existing deployed source. Its old 04:39 status file was
unhealthy with an `authoritative_board_stuck` failure and cannot establish the
new owner's readiness or task liveness. No current health claim should be inferred
from the earlier successful 04:37 observation.

At approximately 04:49, generation 116 had reached owner readiness and new lane
wrappers were starting. Fresh zero-claim/zero-worker admission was unavailable;
the bootstrap optimization and temporary-client caps remain on the repair
branch (through `10966d27d`), not on the running source. No restart was performed
by this continuation.

### PCTDD clean source preparation

Preparation root: `.worktrees/pctdd-resource-aware-sealing-prep`, with its own
clean `external/ipfs_accelerate` checkout. The selected immutable baselines are
outer `3a6d685f6355d68b9adc897f1c93412fe87a7045` and accelerator
`0fccbf887b8ba25dff4adae7893e774a25895823`. Accelerator commit
`4d38f10188b8837acc2a1b61197d0c1d8e53e1a4` contains only the resource backport,
verification-slot integration, two temporary DuckDB client caps and their tests.
The unrelated unfinished G9 changes were not imported or committed. SHA/HMAC
verification and the workspace fingerprint encoding are preserved; verification
callbacks remain outside shared worker slots.

Outer preparation commit `c661f7851b444ee9bdbebe042b38a1fabf9e3575` records only
the new accelerator gitlink and explicit no-start/admission notes. Both clean
preparation worktrees passed final status checks. Focused tests: **107 passed**
(19 rollout, 17 worker-slot, 10 temporary-client, 9 proof-verification and 52
DuckDB-policy cases). An additional broad run had 93 passed and 20 failed; one
representative provider-route fixture failure reproduced on exact pre-patch
code, and a disposable Docker probe failed cleanup verification. Both exact
test containers were subsequently confirmed absent. The broad run is not
claimed as a passing deployment gate.

The existing copied configuration still names the active G9 runtime path and
endpoint. Its dependency validator requires exact sealed source/tree/gitlink
and clean-source witnesses; the old G8-to-G9 successor tool is not a generic
live-G9 upgrade procedure. Consequently this preparation must not be launched,
resumed or resealed as if those old references authorize the new checkout.
No live PCTDD route, owner, database or source was replaced.

Detailed admission requirements are in
`.worktrees/pctdd-resource-aware-sealing-prep/docs/operations/pctdd-resource-aware-sealing-preparation.md`.
The clean PCTDD and SAWM ports are resource controls, not shared digest caching;
the 24-hour authenticated observation service remains an ASEH integration.

## Continuation after 05:07 UTC

Fresh ASEH checks still found deployed source `7f9a2d...`, generation 116/schema
4 and an active Grok attempt. The supervisor has an infinite configured runtime;
the task timeout is four hours, so a prompt natural exit is not guaranteed.
A bounded waiter was queued on the existing cron launch lock, without sending
signals or changing the launcher, to catch a natural exit if one occurred.
The initial candidate merge preview was conflict-free and matched the repair
tree exactly. Source changes still require the established two-parent merge
and genuine startup admission after all live work has stopped.

An actual isolated capsule preflight exposed a fresh-checkout permission gap:
the preparation helper hardened its handwritten list and supervisor package,
but omitted required package-root utilities and `scripts/ops` entrypoints that
had inherited group-write permission. The seal correctly rejected those files.
The fix must use the capsule's authoritative source inventory and preserve its
ownership/no-follow checks; this is not grounds to relax capsule validation.

### Updated SAWM preparation, not a live successor

The other writer committed M70 and further generation-48 repairs, then advanced
the selected checkout again while this rollout was inspecting it. Its Quack
owner is now running. Neither a transient clean status nor the user's request
to continue was treated as evidence that the other writer had paused.

The resource patch was refreshed separately onto immutable source
`9e4804c5866f8b22aa77d095d13c7f78bab06490`, retaining the earlier preparation.
The new branch is `codex/sawm-resource-hashing-m70`, commit
`62ac6f52dc87cc5d29909f161bedbf6dcc5c0f33`, tree
`2c8f8b945d6847d53720381c5356decf2e95541e`. All 49 focused tests passed again,
including the sealed runner. Production capsule sealing and verification passed:

- Capsule CID: `sha256:e5a414bc4e7ec9aaede3028b54ae3cb35c9e87b4e21081de9306eeb930a514ca`.
- Archive SHA-256: `sha256:9bd44fa59ed98aa471d80694c6b55d3654e367e7c652b80f5ac2e2908ed919ca`.

This is still a source preparation, not operator admission. M70 explicitly
authorizes a restart/generation transition; it cannot serve as a zero-generation
resource amendment. A stable reviewed source and a new narrowly scoped
source-only authority/adapter remain necessary. No selected source, owner,
configuration, database or service was changed by this continuation.

### PCTDD admission clarified without mutations

The real read-only prepared-source preflight returned
`owner_management.owner_state_dir is outside the repository`, before creating
state. Existing G9 `resume` can admit a clean descendant source through its
current-tree checks and authenticated canonical task authority; it is not a hot
reload and returns `already_running` for a healthy master. Thus no speculative
G10 migration or new diagnostic CLI was added.

The supported next step is to reconcile the scoped resource commit into canonical
G9 source while preserving its unfinished overlay, bind the exact outer gitlink,
pass current-source controls, and coordinate normal resumption. Retargeting the
prepared configuration to borrow the live owner's path or replaying G8-to-G9
would not supply that admission. At 05:18, the live owner/master remained present
and lane 0 had started agentic maintenance after replacing a termination-blocked
supervisor. This was not an idle state. Both preparation worktrees stayed clean.

### Capsule permission correction

The preparation helper now takes its Python members directly from the capsule
source inventory and changes mode through anchored no-follow descriptors with
inode/mode/owner/link checks. SQL checkout permissions and unrelated files are
left alone; foreign-owned, hardlinked and nonregular files are not modified.
The capsule's subsequent strict byte and ownership admission is unchanged.
Thirteen new permission regressions and two existing cases passed; an independent
combined run of permission, bootstrap-cache, hashing-entrypoint and temporary
Quack-client tests passed all **60 cases**. The latter run emitted only the
known fork deprecation warning and an unregistered timeout-marker warning from
disabling optional pytest plugin autoload. No live source was edited.
