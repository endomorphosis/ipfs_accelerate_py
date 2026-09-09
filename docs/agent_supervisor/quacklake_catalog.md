# QuackLake catalog export

The connector targets [tobilg/quacklake](https://github.com/tobilg/quacklake), a
DuckLake catalog served over Quack. Its catalog metadata runs in Cloudflare
Durable Objects; table data lives in R2. The local fleet control plane remains a
separate DuckDB + Quack owner. This connector creates no cloud resources and
requires an existing catalog endpoint, catalog JWT, and scoped R2 credentials.

Copy `config/agent_supervisor_quacklake_catalog.example.json` to
`~/.config/ipfs-taskboard-watchdog/quacklake-catalog.json` and replace every
example value with the catalog-assigned endpoint and data path. Credential paths
must be absolute. The JWT file contains only the token. The R2 credential file
contains JSON with `access_key_id`, `secret_access_key`, and `endpoint`, plus
optional `session_token`. The endpoint is the account's
`<account>.r2.cloudflarestorage.com` hostname. Both credential files must be
regular files owned by the current user, without group or other permissions;
symlinks are rejected. Keep credentials outside source control.

```sh
python3 scripts/ops/agent_supervisor/quacklake_catalog_export.py \
  --config "$HOME/.config/ipfs-taskboard-watchdog/quacklake-catalog.json" \
  --deployment "$HOME/.local/state/ipfs-quack-fleet/deployment.json" \
  --inventory "$HOME/.config/ipfs-taskboard-watchdog/inventory.json" \
  --check-config
```

This checks configuration structure only. It neither validates credentials nor
proves remote connectivity. Removing `--check-config` queries the live native
aggregate with an owner-issued read grant and exports its current source
observations. It does not read the cached aggregate JSON or taskboard files.
Unavailable sources remain explicitly unavailable; historical observations
never confer completion authority. Exported CIDs and JSON bytes match the native
observation contract, including Unicode and bounded payload admission.

Install and qualify compatible `quack`, `ducklake`, and `httpfs` extensions before
connecting. The upstream getting-started guide currently requires fixes from
`core_nightly` for Quack and DuckLake. The connector only loads installed
extensions; it never downloads or force-replaces the cache used by running
native owners. Use an isolated extension cache when qualifying different builds.
Local tests prove SQL projection/replay, transaction rollback and credential
handling. Remote compatibility remains unverified until a real catalog is
connected successfully.

The resulting `fleet_lake.fleet_source_observations` table stores native
observation CIDs, source IDs, observation timestamps, availability, payload JSON,
and an always-false completion-authority column. Repeating a batch through the
same designated exporter skips existing CIDs. QuackLake currently lacks
cross-session transaction conflict detection, so run one exporter for this table
per catalog. There is no distributed uniqueness or exactly-once guarantee.
Consumers must assess observation age and native receipt deadlines; a historical
`available` row is not evidence of current source availability.

User-unit templates are in `deploy/systemd/user/ipfs-quacklake-catalog-export.*`.
Install an immutable source release and point
`~/.local/lib/ipfs-quack-fleet/quacklake-export-current` to it before copying the
units into `~/.config/systemd/user/`. After `systemctl --user daemon-reload`,
enable `ipfs-quacklake-catalog-export.timer`. It attempts export every two minutes
only when catalog configuration exists. `HOLD` and `OPERATOR_STOP` beside the
native fleet deployment suppress export. A local file lock prevents overlapping
unit executions. Each attempt has a 90-second deadline and its own process group
and memory limit. Export failures never restart source boards or native owners.
Logs report only exception classes because driver exception text can contain
secret SQL.
