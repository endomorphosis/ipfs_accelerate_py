# CASF frozen benchmark matrix

This directory defines the immutable comparison envelope for CASF-038 through
CASF-041. It contains benchmark specifications, not qualification results.
Results are stored as content-addressed artifacts and referenced by compact
receipts only after the current merged tree is exercised.

The benchmark target is 12 independent supervisor processes, 256 registered
logical subagents, no more than 64 concurrently executing subagents, at least
1,000 bounded tasks, and at least 100,000 event deliveries including replay.
An in-process object graph cannot satisfy the process-qualified profile.

The single-supervisor baseline and federation candidate must use the same host,
task population, providers, budgets, tests, proofs, merge policy, and acceptance
criteria. Any missing identity, stale telemetry, lost event, duplicate effect,
reduced assurance, or simulated-as-live observation invalidates the comparison.

See `matrix.yaml` for frozen identities, metrics, targets, nonclaims, and the
artifact/receipt requirements.
