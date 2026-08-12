# Control-plane SQL migrations

**Owner module:** `ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations`

This directory is the only admitted source of control-plane schema DDL for
`control.duckdb`. Runtime modules must not invent tables. The checksum-bound
migration catalog loads ordered SQL files from here and the runner applies them
transactionally with receipts.

## Filename contract

Each migration file must match:

```text
NNNN_slug.sql
```

| Part | Rule |
| --- | --- |
| `NNNN` | Zero-padded positive integer version, contiguous from `0001` |
| `slug` | Lowercase snake_case description (`[a-z0-9_]+`) |
| extension | Exactly `.sql` |

Examples:

- `0001_control_plane.sql`
- `0002_task_projection.sql`

Files that do not match the pattern are refused at catalog load time. Gaps
(`0001` then `0003`) and duplicate versions are refused.

## What belongs in a migration

- `CREATE TABLE` / `CREATE VIEW` / constraints for control-plane domains
- Additive `ALTER TABLE` changes that preserve existing identities
- Seed rows required for schema contracts (not operator secrets)

What does **not** belong here:

- Runtime ad-hoc DDL from daemons or providers
- Credential material or provider tokens
- One-off operator repair scripts without a versioned checksum

## Checksums and receipts

The runner records, for every applied version:

- migration version and `migration_id` (`NNNN_slug`)
- SQL content checksum (`sha256:…`)
- application and DuckDB tool versions
- start/end timestamps and outcome
- post-apply schema fingerprint (canonical information-schema identity)

Replay of an already-applied migration is safe only when the catalog checksum
still matches the receipt. Altered SQL is **drift** and fails closed.

## Application rules

1. Migrations apply in strict ascending order with no gaps.
2. Each migration runs inside a single transaction; failure rolls back SQL.
3. Only one owner may apply migrations at a time (migration ownership lease).
4. Downgrades are refused.
5. Partial application markers refuse further apply until repaired.
6. Runtime connections opened through the runner refuse ad-hoc DDL unless an
   explicit compatibility path is enabled.

## Empty-to-latest equivalence

Applying the full catalog to two independent empty databases must yield the
same schema fingerprint. Upgrade paths that apply `1..N` incrementally must
match a single empty-to-latest apply of the same catalog.

## Adding a migration (later schema tasks)

1. Add the next contiguous `NNNN_slug.sql` file in this directory.
2. Keep statements deterministic and free of environment-specific paths.
3. Prefer explicit column types, UTC timestamps, revision/epoch fields, and
   primary keys that preserve task CIDs and lease semantics.
4. Extend tests under `test/api/test_agent_supervisor_control_plane_migrations.py`
   and domain schema tests as appropriate.
5. Do not edit historical SQL files after they have been applied in shared
   environments; ship a new version instead.

## Bootstrap bookkeeping

The runner installs these tables before domain SQL runs:

| Table | Role |
| --- | --- |
| `control_plane_metadata` | Head schema version, fingerprints, ownership lease |
| `schema_migrations` | Successful apply receipts |
| `schema_migration_attempts` | Attempt audit including failures |

Domain SQL (for example `0001_control_plane.sql`) lands in later foundation
tasks and is loaded automatically once present.
