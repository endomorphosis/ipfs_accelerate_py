# Native Quack fleet topology

`DatabaseProgramConfig` now defaults new program selections to DuckDB through
Quack, with fail-closed transport. Endpoint, secret handle, store identity,
generation and schema binding remain required. Existing explicitly selected
legacy/bootstrap profiles retain their meaning.

The fleet topology binds existing boards to two independent native state owners:

- `aggregate_control`: DuckDB + Quack control for federation and observational
  DuckLake history receipts. The existing history API is
  `DuckLakeHistoryProjection@1` / `DuckLakeProjectionStore@1`.
- `derived_coordination`: a separate DuckDB + Quack endpoint for derived AST CID,
  content-hash and state-root reference coordination. Git remains source-byte
  authority and `ipfs_datasets_py` remains semantic truth authority.

This module implements native Quack federation and observation aggregation.
[QuackLake](https://github.com/tobilg/quacklake) is a separate DuckLake catalog
service; this local aggregation owner is not that product. Connecting a real
QuackLake catalog requires a separate admitted endpoint and catalog connector.
History aggregation never completes tasks, steals leases or substitutes for
native acceptance.

The optional [QuackLake catalog connector](quacklake_catalog.md) exports admitted
fleet observations to an existing catalog. It runs separately from both native
owners, so an unavailable catalog cannot block board scheduling or state reads.

Compile the actual installed fleet and render native user services:

```sh
python3 -m ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_topology \
  --config config/agent_supervisor_quack_fleet_topology.json \
  --inventory "$HOME/.config/ipfs-taskboard-watchdog/inventory.json" \
  --state-root "$HOME/.local/state/ipfs-quack-fleet" \
  --code-root /absolute/immutable/release \
  --write-unit-dir /tmp/quack-fleet-units
```

The output retains exact sealed board selections. A missing native database or
configuration registers an unavailable source and does not recreate it from JSON.
Each new owner uses its own endpoint, database and owner-state directory.
The default dedicated ports are 27841 and 27842, below the usual Linux client
ephemeral range. Check the host range when overriding them: an outgoing client
can otherwise borrow a stopped owner's port and delay restart until TIME_WAIT
expires. The native owner retains its normal bind checks and restart backoff. Source
endpoint collisions, overlaps and path escapes are refused. Add more inventory
entries to expand the source population; up to 4096 sources are admitted per
compiled topology, independently of a board's lane count.

The generated units launch the existing native state-owner CLI with automatic
restart and `--deny-legacy-board-unstall`. They retain the normal capability,
process-birth, generation and isolation gates. If an installation requires a
native isolation receipt, provide that exact owner's receipt using
`owners.<role>.isolation_receipt_path`; a receipt from another owner is refused.
Rendering does not enable services or imply that capability gates have passed.

`attach_typed_instance` uses `QuackStateClient` and `TypedStateOwnerConnection`
with an owner-issued, scoped grant. It never opens a database filename on a
transport or authorization failure. The compiler does not issue credentials.
The native owner selects its current generation; a topology's initial-generation
binding is not evidence that an owner is currently ready.

Deployment wiring is separate from runtime qualification. The receipt accurately
reports `runtime_qualified: false` until the existing federation registration,
source-read grants, projection receipt commands and derived-reference consumers
have been admitted. In particular, an empty running owner alone is not a working
history aggregator or semantic index.

The implemented observation service is
`scripts/ops/agent_supervisor/quack_fleet_aggregate.py`. It owns the aggregate
Quack instance, polls each board through its existing admitted native operator
adapter, and writes a content-addressed artifact plus history receipt through a
scoped native typed command. The records use the existing `artifacts` and
`federation_receipts` tables. No new database implementation or alternate task
store is introduced.

A missing, stopped, denied or expired source becomes typed unavailable. Its last
admitted observation remains available as history. The current view checks age
again when queried; DOEP monitor receipts retain their shorter native deadline.
Blocked native boards can still provide valid observations. Unauthenticated
watchdog, daemon and Markdown projections never become admitted source samples.

The owner publishes a separate mode-0600 read credential. Independent processes
receive short-lived grants bound to their kernel peer and may query the two
closed observation reads; they cannot write observations or task state:

```sh
python3 /absolute/immutable/release/scripts/ops/agent_supervisor/quack_fleet_aggregate.py \
  --deployment "$HOME/.local/state/ipfs-quack-fleet/deployment.json" \
  --inventory "$HOME/.config/ipfs-taskboard-watchdog/inventory.json" --query
```

The JSON written to `aggregate-view.json` is an export of that typed query, not
source or completion authority. The CLI above queries the live control owner.

DuckLake archival publication is still a separate optional integration: source
event ranges must be admitted before its existing range-projection API can
publish history. This change implements the native Quack aggregation boundary
and durable observational history in the control owner; it does not claim that
a QuackLake catalog is deployed or connected, or that DuckLake archival export
is already running. History availability is never a scheduling prerequisite.

Native source reads have independent polling deadlines. A slow or unavailable
source does not delay publishing completed reads or polling healthy sources
again. At most one read per source is in flight; oldest due sources share a
bounded worker pool. `--source-workers` controls that pool (default 16, range
1–256). Results from removed or changed inventory bindings are discarded, and
reader errors become typed unavailable observations. Native receipt deadlines
remain enforced when reading the current view. Only successful control writes
refresh the observer's progress deadline.

These defaults do not automatically migrate legacy merge queues. In particular,
`MergeQueue` and `DatabaseMergeQueue` still have callers using
`open_duckdb_connection` with Quack preferred and file fallback allowed. Migrate
those existing queues through their native stopped-owner/source-requalification
controls, with a real Quack owner and qualified data transfer. Setting a global
require-Quack flag without provisioning that authority would strand the queue.

Dedicated-owner health monitoring uses `runtime.quack_fleet_health`. Its native
read-only probe must complete an authenticated typed generation query within
five seconds and match the still-live published process birth and database
identity. A status file saying `ready` alone cannot pass. Two consecutive failures
request restart of only the two closed dedicated-owner service names, with a
five-minute cooldown and at most three requests per hour. `HOLD` or
`OPERATOR_STOP` beside `deployment.json` disables these automatic requests.
The aggregate worker also has an independent 600-second progress deadline;
source query failures are retained as unavailable observations, while a stopped
or hung worker makes the owner process exit for systemd recovery. Repeated
control-write failures do not refresh that progress deadline.


The aggregate observation writer reuses one native client and one grant bound to
its exact owner identity and process birth. It renews that grant through the
native owner before expiry, including before a transport reconnect. A failed
write closes the connection; a later poll may reconnect with the same live
grant. Expiry or owner replacement retires the cached binding, and each cycle
attempts at most one renewal and one attach. This keeps ordinary polling from growing
the owner's grant registry or revocation history. Observer shutdown closes its
client and revokes the single remaining grant. No source admission, receipt TTL,
or semantic acceptance policy changes with this lifecycle repair.
