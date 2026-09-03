# PCPR-063 Accelerate deterministic-first route

These files record the Accelerate-owned hermetic deterministic-first
route against the PCPR-061 DatasetsContextPack@1 identity and the
PCPR-062 Kit current root. Accelerate owns execution. It does not
remint either identity, does not produce a bounded patch, and does
not run selected tests as live.

- `cpython312/reference.deterministic-first-route.json` is the declared
  route document. Exact commit and tree are bound by the PCPR-063
  receipt `current_tree_binding`.
- `platform-deterministic-first-route-catalog.json` observes sibling
  Datasets and Kit route bindings when present. Sibling source is
  never required.
- Escalation order is exact receipt, AST and dependency analysis,
  schema/type/static checks, selected tests, incremental prover,
  local small specialist, medium model, frontier model, human
  decision. A model assertion cannot complete work.
- Bounded patch remains PCPR-064. Selected tests and proofs remain
  PCPR-065.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-063-operator-live-deterministic-route`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
