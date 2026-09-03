# PCPR-072 Accelerate final proof-carrying receipt chain

These files record the Accelerate-owned hermetic final proof-carrying
receipt chain against the PCPR-071 recovery, the PCPR-070 authoritative
state-owner restart, the PCPR-061 DatasetsContextPack@1 identity, the
PCPR-062 Kit current root, the PCPR-063 deterministic-first route, the
PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs run,
the PCPR-066 unrelated change, the PCPR-067 eligible reuse, the
PCPR-068 relevant interface change, and the PCPR-069 stale rejection
and PlanDelta. Accelerate owns independent semantic validation,
hermetic artifact storage, and the hermetic ExecutionReceipt. It does
not remint those identities and does not claim live Quack chain
emission.

- `cpython312/reference.final-receipt-chain.json` is the declared chain
  document. Exact commit and tree are bound by the PCPR-072 receipt
  `current_tree_binding`.
- `platform-final-receipt-chain-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- The chain links objective, context, epoch, task, proof, storage,
  event, compatibility, and result identities. Independent semantic
  validation is hermetic. A hermetic ExecutionReceipt is emitted. Chain
  artifacts are stored hermetically, not as live IPFS. Replaying the
  same chain command yields the same chain CID and is rejected as a
  duplicate effect. Stale leases and fences cannot authorize the chain.
  Owner generation stays at 2. Plan epoch stays at 2. Continuation
  proceeds without database edits. External-client demonstration
  remains PCPR-080.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-072-operator-live-final-receipt-chain`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
