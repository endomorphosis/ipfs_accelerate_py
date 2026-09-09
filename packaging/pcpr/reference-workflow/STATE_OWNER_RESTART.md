# PCPR-070 Accelerate authoritative state-owner restart

These files record the Accelerate-owned hermetic restart of the
authoritative Quack state owner against the PCPR-061 DatasetsContextPack@1
identity, the PCPR-062 Kit current root, the PCPR-063 deterministic-first
route, the PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs
run, the PCPR-066 unrelated change, the PCPR-067 eligible reuse, the
PCPR-068 relevant interface change, and the PCPR-069 stale rejection and
PlanDelta. Accelerate owns owner restart and durable reconstruction. It
does not remint those identities and does not claim live Quack restart.

- `cpython312/reference.state-owner-restart.json` is the declared
  restart document. Exact commit and tree are bound by the PCPR-070
  receipt `current_tree_binding`.
- `platform-state-owner-restart-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- Owner state is reconstructed from durable records, not memory.
  Duplicate restart effects are rejected. Unexpired leases and current
  fences are retained. Stale leases and fences expire. Continuation
  proceeds without database edits. Recovery and idempotency remain
  PCPR-071.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-070-operator-live-authoritative-state-owner-restart`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
