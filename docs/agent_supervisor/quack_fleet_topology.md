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

The user-facing term QuackLake describes this composition; this repository does
not contain a separate QuackLake database implementation. History aggregation
never completes tasks, steals leases or substitutes for native acceptance.

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
Each new owner uses its own endpoint, database and owner-state directory. Source
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
