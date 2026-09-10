# DuckLake fleet history

Fleet aggregation uses the official [DuckLake extension](https://duckdb.org/docs/lts/core_extensions/ducklake)
with a local DuckDB metadata catalog and Parquet data. The native DuckDB + Quack
control owner authenticates and aggregates independent supervisor observations.
A separate bounded worker archives samples from its read-only typed API into
DuckLake. The derived AST/hash/state owner remains a separate Quack instance.
This deployment needs no external catalog service or cloud credentials.

Run one export after the installed DuckLake extension has been qualified:

```sh
python3 scripts/ops/agent_supervisor/ducklake_fleet_export.py \
  --history-root "$HOME/.local/state/ipfs-quack-fleet/ducklake-history" \
  --deployment "$HOME/.local/state/ipfs-quack-fleet/deployment.json" \
  --inventory "$HOME/.config/ipfs-taskboard-watchdog/inventory.json"
```

The dedicated history directory must be absolute, owned by the current user,
and private (mode 0700). It contains `metadata.ducklake`, `parquet/`, and
`history.lock`. The exporter loads the installed extension without installing
or replacing extensions in a running owner's cache. It disables data inlining
so observations are written to Parquet. Local qualification used DuckDB 1.5.5;
the documentation link's LTS channel does not request a runtime downgrade.

`fleet_lake.fleet_source_observations` stores source ID, native observation CID,
observation time, availability, canonical payload and `completion_authority=false`.
Replay under the single writer lock skips previously committed CIDs. Source
unavailability is retained explicitly and never gains an old receipt as a current
one. Historical availability describes that observation's time; consumers must
query the live control plane for current admission and freshness.

The worker samples the latest observation for each registered source. This is
observational history, not a complete source event-log export. It neither changes
taskboard state nor proves task acceptance, goal settlement, or merge readiness.
Existing source-range projection APIs still require their own admitted receipts.

Every catalog reader and writer takes the same nonblocking local lock. An
overlapping exporter fails without opening the catalog; process exit releases
the lock. A failure after commit can be replayed safely by CID. Never delete
the lock, metadata, or WAL to clear contention. For bounded inspection:

```sh
python3 scripts/ops/agent_supervisor/ducklake_fleet_export.py \
  --history-root "$HOME/.local/state/ipfs-quack-fleet/ducklake-history" --inspect
```

Inspection uses a read-only attachment and never initializes missing history.
The optional `--status-file` output is diagnostic and is never read as authority.

For recurring export, publish a tested immutable source release and point
`~/.local/lib/ipfs-quack-fleet/ducklake-export-current` to it. Copy the two
`deploy/systemd/user/ipfs-ducklake-fleet-export.*` templates into
`~/.config/systemd/user/`, reload user units, then enable the timer:

```sh
systemctl --user daemon-reload
systemctl --user enable --now ipfs-ducklake-fleet-export.timer
systemctl --user start ipfs-ducklake-fleet-export.service
```

The timer retries every two minutes. The service has a 90-second deadline,
512 MiB memory limit, 128-task ceiling and a process-group cleanup boundary.
Numerical worker pools are limited to one thread. The task ceiling accommodates
the native reader subprocess and extension loading; a 32-task unit aborted
during live qualification. DuckDB connection thread limits are supplied at
creation rather than after initialization. Native aggregate
reads have a 45-second deadline and a 16 MiB output limit. A hung history worker
can therefore be terminated without stopping either Quack owner or a board's
supervisors. Fleet `HOLD` and `OPERATOR_STOP` files suppress scheduled export.

On installations that enabled the earlier optional QuackLake service, disable
`ipfs-quacklake-catalog-export.timer` when selecting this local DuckLake deployment.
The optional cloud connector remains available separately; it is not a dependency.
