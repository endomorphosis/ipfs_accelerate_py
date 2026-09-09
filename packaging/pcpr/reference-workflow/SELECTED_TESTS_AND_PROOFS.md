# PCPR-065 Accelerate selected tests and proofs

These files record the Accelerate-owned hermetic selected-tests-and-proofs
run against the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit
current root, the PCPR-063 deterministic-first route, and the PCPR-064
bounded PatchPlan. Accelerate owns execution. It does not remint those
identities and does not claim live selected tests or a live prover.

- `cpython312/reference.selected-tests-and-proofs.json` is the declared
  run document. Exact commit and tree are bound by the PCPR-065 receipt
  `current_tree_binding`.
- `platform-selected-tests-and-proofs-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- The PatchPlan selection is incomplete, so full validation is required
  and is run hermetically. Missing incremental provers stay typed
  unavailable. Model stages are not invoked.
- Unrelated-state-change reuse remains PCPR-066. Incomplete selection
  cannot count as selected-test-sufficient release evidence.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-065-operator-live-selected-tests-and-proofs`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
