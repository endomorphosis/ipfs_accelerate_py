# Proof-context v0.1 example repository

This directory is a small, credential-free ordinary Python repository. It is
designed to be copied into a new Git repository or cloned from a local seed;
it does not search for sibling checkouts, contact a provider, or use the
network. The walkthrough consumes the installed
`ipfs_accelerate_py.proof_context` runtime and never acts as runtime or
acceptance authority itself.

From a fresh clone, run:

```bash
python walkthrough.py \
  --repository . \
  --state-dir /tmp/pcce-example-state \
  --output /tmp/pcce-example-transcript.json
python -m json.tool /tmp/pcce-example-transcript.json
```

The repository must already have a clean initial commit. The walkthrough then
executes and records these bounded operations:

1. bind the exact Git commit and tree;
2. scan and create a compressed context pack;
3. exercise explicit context expansion and report before/compressed/expanded
   token counts using the documented `whitespace-v1` estimator;
4. submit a path-traversal patch and require a live governed
   `boundary_violation` rejection;
5. apply the good local patch in a disposable validation worktree and run only
   the selected tests;
6. submit those exact bytes through the production-mode proof-context runtime;
7. reuse the selected-test receipt only after exact candidate-tree equality,
   and independently observe exact verification-receipt reuse from the runtime;
8. require live assurance, a real incremental seal CID, and a successful final
   report without changing the canonical clone.

The emitted JSON contains the exact fixture commit/tree, every operation's
status and artifact identity, the selected-test command and receipt, the bad
patch reason, the accepted patch identity, proof-reuse identities, and the
final seal CID. Any missing identity, failed test, simulated provenance,
unexpected mutation, unsafe reuse key, or unsealed result terminates without a
successful transcript.
