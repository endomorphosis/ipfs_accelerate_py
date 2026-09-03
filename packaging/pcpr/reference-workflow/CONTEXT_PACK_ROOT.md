# PCPR-062 Accelerate ContextPack storage binding

These files bind Accelerate to the Datasets-owned DatasetsContextPack@1
identity and the Kit-owned hermetic current root. Accelerate is a
consumer. It does not remint either identity, does not store bytes, and
does not execute the deterministic-first route.

- `cpython312/reference.context-pack.root.json` binds
  SupervisorContextPack to the Datasets owner identity and the Kit
  current root. Exact commit and tree are bound by the PCPR-062
  receipt `current_tree_binding`.
- `platform-context-pack-root-catalog.json` observes sibling Datasets
  and Kit storage documents when present. Sibling source is never
  required.
- Deterministic execution remains PCPR-063.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-062-operator-live-context-pack-root`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
