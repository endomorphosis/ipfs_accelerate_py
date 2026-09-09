# PCPR-068 Accelerate relevant interface change

These files record the Accelerate-owned hermetic relevant interface
change against the PCPR-061 DatasetsContextPack@1 identity, the
PCPR-062 Kit current root, the PCPR-063 deterministic-first route, the
PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs
run, the PCPR-066 unrelated change, and the PCPR-067 eligible reuse.
Accelerate owns change-event classification. It does not remint those
identities and does not claim live reuse.

- `cpython312/reference.relevant-interface-change.json` is the declared
  change document. Exact commit and tree are bound by the PCPR-068
  receipt `current_tree_binding`.
- `platform-relevant-interface-change-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- The change adds successor protocol operation `impact`. The impacted
  cone is nonempty. LogicProviderProtocol@2 is not reminted. Stale
  rejection and PlanDelta remain PCPR-069.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-068-operator-live-relevant-interface-change`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
