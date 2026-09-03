# PCPR-069 Accelerate stale rejection and PlanDelta

These files record the Accelerate-owned hermetic stale rejection and
PlanDelta against the PCPR-061 DatasetsContextPack@1 identity, the
PCPR-062 Kit current root, the PCPR-063 deterministic-first route, the
PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs
run, the PCPR-066 unrelated change, the PCPR-067 eligible reuse, and
the PCPR-068 relevant interface change. Accelerate owns stale-evidence
rejection and PlanDelta production. It does not remint those identities
and does not claim live PlanDelta admission.

- `cpython312/reference.stale-rejection.json` is the declared
  stale-rejection and PlanDelta document. Exact commit and tree are
  bound by the PCPR-069 receipt `current_tree_binding`.
- `platform-stale-rejection-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- Stale ContextPack and protocol-test identities are rejected.
  Unaffected solver-qualification completion is preserved. The plan
  epoch increments. The hermetic PlanDelta refills only the impacted
  cone within 12 tasks and 20 epochs. Restart recovery remains
  PCPR-070.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-069-operator-live-stale-rejection-and-plan-delta`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
