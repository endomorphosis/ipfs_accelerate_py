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
