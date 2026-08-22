# External Agent Fabric deployment (EAAEF-163)

Supported profiles: local supervised, detached single-host, multi-container single-host.
Remote multi-host stays disabled until authenticated Quack failover/partition/security gates pass.

## Schema migration

Apply control-plane migrations only through the exclusive Quack owner. Clients never ATTACH the operational DuckDB for UPDATE.

## Quack owner backup and restore

Backup the owner's private DuckDB file while the owner is fenced. Restore requires a later epoch/fence. Shared tokens are not authority.

## DuckLake recovery

DuckLake is history only. Loss or lag cannot grant claims, leases, fences, or merge authority. Rebuild from the DuckDB outbox cursor.

## Upgrade and rollback

Upgrade from immutable image digests and schema fingerprints. Rollback to the previous digest and schema revision with an exact compatibility check. Mutable tags are not authority.
