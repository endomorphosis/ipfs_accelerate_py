# PCPR-064 Accelerate bounded PatchPlan

These files record the Accelerate-owned hermetic bounded PatchPlan
against the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit
current root, and the PCPR-063 deterministic-first route. Accelerate
owns execution. It does not remint those identities and does not run
selected tests as live.

- `cpython312/reference.bounded-patch.json` is the declared PatchPlan
  document. Exact commit and tree are bound by the PCPR-064 receipt
  `current_tree_binding`.
- `platform-bounded-patch-catalog.json` observes sibling Datasets and
  Kit patch bindings when present. Sibling source is never required.
- The hermetic patch adds a fail-closed PatchPlan surface to
  LogicProviderProtocol@2 without reminting the protocol identity or
  adding a protocol operation.
- Selected tests and proofs remain PCPR-065. Incomplete selection
  requires full validation.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-064-operator-live-bounded-patch`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
