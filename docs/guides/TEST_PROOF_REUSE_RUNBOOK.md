# Test proof reuse rollout runbook

This runbook controls promotion and rollback of proof-backed pytest result
reuse. The system is fail-closed: the unpromoted default is `off`, a held
promotion leaves the current stage unchanged, and incomplete safety telemetry
causes an active stage to roll back.

The rollout decision is evidence, not authority. An operator-controlled
deployment must verify the decision and apply the corresponding pytest mode.
Never treat a benchmark receipt, a sampled match, or a rollout decision by
itself as permission to expand test or signer authority.

## Stages

Promotion is one adjacent stage at a time:

| Stage | Eligible pytest mode | Purpose |
| --- | --- | --- |
| `off` | `off` | No proof lookup, skip, or write. This is always the default. |
| `shadow` | `shadow` | Verify and predict, but run every test normally. |
| `read` | `read` for explicitly eligible tests; `off` otherwise | Permit verified skips after clean forced-rerun evidence. |
| `opt_in_readwrite` | `readwrite` only for eligible, explicitly opted-in jobs; otherwise `read` or `off` | Admit writes only from controlled CI issuers. |
| `eligible_default` | `read` for eligible tests; writes remain opt-in | Make reads eligible by default only after the current-tree and all-repository gates pass. |

The policy cannot jump a stage. `eligible_default` never grants implicit write
authority. The policy's `mode_for()` and `config_for()` methods only narrow a
reviewed promotion; they cannot promote a stage.

Pytest accepts the concrete modes `off`, `shadow`, `read`, `write`, and
`readwrite` through `--proof-reuse-mode` or
`IPFS_TEST_PROOF_REUSE_MODE`. Keep the environment unset or explicitly `off`
until an accepted rollout decision has been applied:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest
```

## Bootstrap and lazy dependencies

The accelerator, datasets, and kit repositories discover the same pytest
plugin through their `pytest11` entry point or a source-checkout root
`conftest.py` fallback. Direct node selection uses the collected pytest item;
there is no test-file registry. Importing any package or the plugin does not
install dependencies, initialize storage, create a cache, probe a prover,
start a daemon, or modify `sys.path`.

In an enabled mode, automatic item assembly runs for every non-disabled
collected test. Optional CID, certificate-verifier, cache, and provider modules
are resolved only after assembly produces an exact lookup request, or after a
complete pass produces a publication intent. Missing services therefore leave
ordinary test execution unchanged and are not imported merely because
`shadow`, `read`, or `write` was configured.

Managed environments can inject capabilities before collection with:

- `set_proof_reuse_dependency_installer(config, installer)` for a controlled
  installer;
- `set_proof_reuse_service_resolver(config, resolver)` or
  `set_proof_reuse_services(...)` for cache and verifier services; and
- `set_proof_reuse_identity_services(config, services)` for current forest,
  AST, component, policy, and runtime-evidence providers.

The built-in pip installer is disabled by default. To permit its closed
allowlist on first actual use, set
`IPFS_TEST_PROOF_REUSE_AUTO_INSTALL=1`. Override the local store location with
`IPFS_TEST_PROOF_REUSE_CACHE_DIR`; otherwise the enabled session uses
`.pytest_cache/proof-reuse` below its pytest root. An install, import, cache,
provider, verifier, or Groth16 failure always results in `RUN`.

Automatic collection cannot safely claim that a prior runtime trace is current.
Tests receive a typed `RUN` result unless the identity service supplies fresh
controlled-preflight evidence bound to the exact node, repository forest,
static trace, component root, and runtime policy. Even then, identity assembly
only creates a lookup request; the local verifier remains the sole authority
that can return `SKIP`.

## Promotion procedure

Create a new `ProofReusePromotionEvidence` packet for exactly one transition.
Do not copy evidence from an earlier tree, repository, policy revision, or
transition. Call `ProofReuseRolloutPolicy.evaluate_promotion()` with the
independently observed current repository ID, tree ID, and time. Apply the
target stage only when `decision.promoted` is true and every serialized gate is
passed.

Every promotion requires all of the following:

1. The requested transition is adjacent and matches the evidence packet's
   current and target stages.
2. The target is present in the reviewed policy's `approved_stages`.
3. `observed_at` is timezone-aware, no older than
   `max_evidence_age_seconds`, and no farther in the future than
   `max_future_skew_seconds`.
4. The evidence policy ID and revision match the active reviewed policy.
5. The evidence repository and tree match values obtained from the deployment,
   not values copied from the evidence.
6. A non-empty operator approval ID identifies this exact promotion.
7. The benchmark receipt uses `BenchmarkReceipt@1`, declares
   `ProofReuseBenchmark@1` and `ProofReuseMetrics@1`, reports zero false
   admissions, and has every benchmark gate passed.
8. The metrics snapshot uses `ProofReuseMetrics@1` and contains aggregate
   counts.
9. Mutation and degradation suites each report zero false skips.
10. Authority contradictions are zero, corruption has not spiked, stale keys
    are zero, and key and revocation monitoring are healthy.

Promotion to `read` or later also requires at least `min_forced_reruns`
completed forced reruns. Every selected sample must be completed, with zero
false skips, zero unexplained mismatches, and zero authority contradictions.

Promotion to `opt_in_readwrite` or later additionally requires
`controlled_issuer=True`. Promotion to `eligible_default` additionally
requires all three explicit conditions: the policy allows eligible-default,
the gate passed on the current tree, and the benchmark/current-tree gate passed
for every required repository.

Any absent, malformed, stale, contradicted, or failed value holds the current
stage. A held decision is not partially successful: resolve the failed
`reason_codes`, generate fresh evidence, and evaluate again.

## Forced-rerun sampling

`ForcedRerunSampler` deterministically selects candidates by basis points:
`100` is 1%, `1_000` is 10%, and `10_000` is 100%. Use a reviewed seed scoped
to the sampling window. All workers must use the same seed and sample rate so
the same execution identity receives the same decision.

For each selected reuse candidate:

1. Record the verifier's predicted outcome.
2. Bypass the reuse skip and execute the real test.
3. Pass the predicted and actual outcomes to
   `compare_predicted_actual()`.
4. Aggregate observations with `ForcedRerunSummary.from_observations()`.
5. Feed the aggregate summary both into the next promotion packet and into
   continuous rollback monitoring.

The controlled outcomes are `pass`, `fail`, `error`, and `skip`. A predicted
`pass` followed by any non-pass actual result is a false skip. Any unequal
prediction and actual outcome without a reviewed explanation code is an
unexplained mismatch. Explained mismatches may be investigated without
blocking promotion only when they are not false skips or authority
contradictions.

Do not emit node IDs, test names, repository paths, raw execution identities,
proof bytes, keys, signatures, or endpoints. The sampler emits a one-way
`sha256:` sample ID, and `ForcedRerunSummary` is aggregate-only. Persist the
canonical evidence and decision IDs so an audit can prove exactly which
immutable packet was evaluated.

## Automatic rollback

Evaluate current `ProofReuseSafetySignals` continuously and before applying
any promotion. Missing signal values are unsafe and are treated as an
unexplained mismatch.

| Trigger | Required response |
| --- | --- |
| Any false skip | Immediately set the rollout stage and pytest mode to `off`. |
| Any authority contradiction | Immediately set the rollout stage and pytest mode to `off`. |
| Any stale key | Immediately set the rollout stage and pytest mode to `off`. |
| Corruption spike | Roll an active read/write/default stage back to `shadow`; if already in `shadow`, set it to `off`. |
| Any unexplained mismatch | Roll an active read/write/default stage back to `shadow`; if already in `shadow`, set it to `off`. |
| Missing or incomplete safety monitoring | Treat as an unexplained mismatch and apply the same rollback. |

If multiple triggers occur, the stricter target wins. `off` remains `off`.
Apply the returned `effective_stage` immediately; incident review is not a
reason to delay rollback. Disable write credentials and stop publishing new
receipts whenever the effective stage no longer permits writes.

After rollback:

1. Preserve the aggregate metrics, safety signals, evidence ID, decision ID,
   active policy ID/revision, repository ID, tree ID, and time window.
2. Quarantine suspect receipts or keys without deleting audit evidence.
3. Re-run affected tests without reuse and investigate verifier, authority,
   revocation, corruption, and provider behavior.
4. Keep the pytest mode at the rollback target while investigating.
5. Repair the cause and repeat every adjacent promotion with new benchmark,
   forced-rerun, monitoring, tree-binding, and operator-approval evidence.

There is no automatic restoration to the previous stage. Evidence captured
before the incident or for another tree is not reusable for re-promotion.

## Operator check

Run the focused gate with reuse explicitly disabled:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off \
  python3 -m pytest external/ipfs_accelerate/test/api/test_proof_reuse_rollout.py -q
```

Before ending a rollout window, confirm that the deployed pytest mode agrees
with the latest accepted promotion or rollback decision. If that cannot be
established, set `IPFS_TEST_PROOF_REUSE_MODE=off`.
