# PCPR-061 Accelerate semantic ContextPack binding

These files bind Accelerate to the Datasets-owned DatasetsContextPack@1
identity for the PCPR-060 reference idea. Accelerate is a consumer. It
does not remint the owner pack CID, does not store bytes, and does not
execute the deterministic-first route.

- `cpython312/reference.context-pack.json` binds SupervisorContextPack
  to the Datasets owner identity. Exact commit and tree are bound by
  the PCPR-061 receipt `current_tree_binding`.
- `platform-context-pack-catalog.json` observes sibling Datasets and
  Kit pack documents when present. Sibling source is never required.
- Durable storage remains PCPR-062. Deterministic execution remains
  PCPR-063.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-061-operator-live-context-pack-admission`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
