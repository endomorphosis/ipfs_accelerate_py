# External Agent Fabric security review (EAAEF-173)

Unresolved critical findings force no-go.

Boundaries audited:

- Trust: imported history is provenance, never authority; workers cannot self-approve
- File/path: no `..`, no host paths, hooks disabled, symlink escape refused
- Import/startup: fail-closed bootstrap; missing signed artifacts are typed no-go
- Environment: no Docker socket; rootless preferred; rootful fallback needs independent admission
- Credential: opaque handles, 0600 token vault, no transcript secrets
- Quack owner: one private DuckDB writer; remote UPDATE/arbitrary SQL refused
- DuckDB: 1.5.5; no silent 1.5.2 substitution
- DuckLake: history/analytics only; never claim/lease/fence/merge authority
- Container: nonroot, read-only root, cap-drop, no-new-privileges, network deny by default
- Receipts: content-addressed; forged identities rejected
