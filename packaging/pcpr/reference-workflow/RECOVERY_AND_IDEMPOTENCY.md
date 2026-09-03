# PCPR-071 Accelerate recovery and idempotency

These files record the Accelerate-owned hermetic recovery and
idempotency demonstration against the PCPR-070 authoritative state-owner
restart, the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit
current root, the PCPR-063 deterministic-first route, the PCPR-064
bounded PatchPlan, the PCPR-065 selected-tests-and-proofs run, the
PCPR-066 unrelated change, the PCPR-067 eligible reuse, the PCPR-068
relevant interface change, and the PCPR-069 stale rejection and
PlanDelta. Accelerate owns recovery continuation and duplicate-effect
rejection. It does not remint those identities and does not claim live
Quack recovery.

- `cpython312/reference.recovery-and-idempotency.json` is the declared
  recovery document. Exact commit and tree are bound by the PCPR-071
  receipt `current_tree_binding`.
- `platform-recovery-and-idempotency-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Recovery continues from durable reconstruction using the retained
  lease and current fence. Replaying the same recovery command yields
  the same recovery CID and is rejected as a duplicate effect. Stale
  leases and fences cannot authorize recovery. Owner generation stays
  at 2. Plan epoch stays at 2. Continuation proceeds without database
  edits. The final receipt chain remains PCPR-072.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-071-operator-live-recovery-and-idempotency`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
