# Semantic Compression Harness Benchmark Results

- Interface: `SemanticStateBenchmark@1`
- Bundle: `sch/benchmark@1`
- Corpus: `semantic-state-benchmark-corpus-v1`
- Fixture corpus: `semantic-state-controlled-repo-v1`
- Task count: **40**
- Tokenizer / estimator: `sch-fixture/token-estimator@1` / `semantic-state-token-estimator-v1`
- Deterministic digest (observational fields stripped): `sha256:15bddb87fcf7af223caaf43f579bbc6e38342356ec6d7ec7acc2c4f541823dd6`

## Gates

- `all_production_eligible_false`: **PASS**
- `median_reduction_at_least_30_percent`: **PASS**
- `no_model_receipts`: **PASS**
- `no_production_root_advanced`: **PASS**
- `task_count_is_40`: **PASS**
- `zero_controlled_false_negatives`: **PASS**
- `zero_coverage_omissions`: **PASS**
- `zero_simulated_admissions`: **PASS**
- `zero_stale_admissions`: **PASS**

## Overall context reduction

- Median reduction: **58.90%**
- Mean reduction: **52.28%**
- Range: 4.68% … 68.43%

## Reduction by task type

| Category | Count | Median reduction | Mean reduction |
|---|---:|---:|---:|
| `api_adapter` | 6 | 58.46% | 57.37% |
| `multi_file_refactor` | 6 | 52.81% | 48.79% |
| `rejection_or_escalation` | 6 | 47.09% | 41.55% |
| `schema_migration` | 6 | 58.14% | 57.47% |
| `small_bug_fix` | 10 | 59.08% | 49.34% |
| `test_repair` | 6 | 61.96% | 61.12% |

## Selection precision / recall

- Overall precision: **36.22%** (3622 bp)
- Overall recall: **100.00%** (10000 bp)
- Controlled false negatives: **0**
- False positives (extras kept visible): **81**
- Coverage omissions: **0**
- Stale admissions: **0**
- Simulated admissions: **0**
- Production-eligible true rows: **0**

## Failures and outcomes

### Failure kinds

- `escalated`: 2
- `none`: 34
- `rejected`: 4

### Candidate verification outcomes

- `escalate`: 2
- `pass`: 34
- `reject`: 4

### Production acceptance (never accepted for oracle/replay)

- `blocked`: 2
- `not_applicable`: 34
- `rejected`: 4

### Route distribution (measured)

- `deterministic_only`: 10
- `frontier_model`: 1
- `human_review_required`: 13
- `medium_model`: 16

### Fallback distribution

- `both`: 1
- `full_pytest`: 8
- `none`: 31

Uncertainty declared on **39** tasks (tasks with non-`none_declared` uncertainty).

## Per-task rows

| Task | Category | Baseline | Semantic | Reduction | Precision bp | Recall bp | FN | Route | Outcome | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `sch-bench-01-core-add-body-fix` | `small_bug_fix` | 1438 | 592 | 58.8% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-02-core-process-pipeline` | `small_bug_fix` | 1438 | 587 | 59.2% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-03-exception-behavior-fix` | `small_bug_fix` | 1446 | 538 | 62.8% | 5000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-04-security-side-effect` | `small_bug_fix` | 1450 | 555 | 61.7% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-05-deleted-legacy-helper` | `small_bug_fix` | 1403 | 548 | 60.9% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-06-monkey-patch-surface` | `small_bug_fix` | 1461 | 1379 | 5.6% | 1000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-07-dynamic-import-name` | `small_bug_fix` | 1441 | 1360 | 5.6% | 1000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-08-generated-bindings` | `small_bug_fix` | 1432 | 604 | 57.8% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-09-core-multiply-guard` | `small_bug_fix` | 1438 | 590 | 59.0% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-10-formatting-noise-adjacent` | `small_bug_fix` | 1435 | 546 | 62.0% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-11-fixture-sample-record` | `test_repair` | 1432 | 513 | 64.2% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-12-pytest-ini-markers` | `test_repair` | 1446 | 727 | 49.7% | 1000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-13-test-core-expectation` | `test_repair` | 1438 | 454 | 68.4% | 10000 | 10000 | 0 | `deterministic_only` | `pass` | False |
| `sch-bench-14-test-api-signature-assert` | `test_repair` | 1457 | 633 | 56.6% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-15-test-schema-roundtrip` | `test_repair` | 1438 | 579 | 59.7% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-16-test-security-assert` | `test_repair` | 1446 | 461 | 68.1% | 5000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-17-api-fetch-signature` | `api_adapter` | 1457 | 634 | 56.5% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-18-cross-module-call-core` | `api_adapter` | 1435 | 718 | 50.0% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-19-mcp-client-adapter` | `api_adapter` | 1466 | 580 | 60.4% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-20-mcp-ping-contract` | `api_adapter` | 1466 | 579 | 60.5% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-21-api-default-param` | `api_adapter` | 1457 | 635 | 56.4% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-22-adapter-endpoint-binding` | `api_adapter` | 1466 | 580 | 60.4% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-23-user-record-field-add` | `schema_migration` | 1438 | 578 | 59.8% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-24-user-record-dump-shape` | `schema_migration` | 1438 | 578 | 59.8% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-25-lockfile-pytest-pin` | `schema_migration` | 1432 | 705 | 50.8% | 1000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-26-policy-admission-mode` | `schema_migration` | 1441 | 474 | 67.1% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-27-requirements-txt-pin` | `schema_migration` | 1432 | 704 | 50.8% | 1000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-28-schema-fixture-co-migration` | `schema_migration` | 1438 | 626 | 56.5% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-29-rename-process-symbol` | `multi_file_refactor` | 1439 | 939 | 34.7% | 2000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-30-split-core-api-edge` | `multi_file_refactor` | 1435 | 719 | 49.9% | 10000 | 10000 | 0 | `medium_model` | `pass` | False |
| `sch-bench-31-adapter-interface-split` | `multi_file_refactor` | 1466 | 579 | 60.5% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-32-schema-test-co-refactor` | `multi_file_refactor` | 1438 | 628 | 56.3% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-33-security-api-co-refactor` | `multi_file_refactor` | 1450 | 642 | 55.7% | 10000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-34-delete-and-rename-wave` | `multi_file_refactor` | 1403 | 904 | 35.6% | 1000 | 10000 | 0 | `human_review_required` | `pass` | False |
| `sch-bench-35-stale-receipt-reject` | `rejection_or_escalation` | 1438 | 590 | 59.0% | 10000 | 10000 | 0 | `deterministic_only` | `reject` | False |
| `sch-bench-36-out-of-scope-patch-reject` | `rejection_or_escalation` | 1432 | 576 | 59.8% | 10000 | 10000 | 0 | `deterministic_only` | `reject` | False |
| `sch-bench-37-opaque-native-frontier` | `rejection_or_escalation` | 1433 | 1366 | 4.7% | 1000 | 10000 | 0 | `frontier_model` | `escalate` | False |
| `sch-bench-38-post-scan-source-race` | `rejection_or_escalation` | 1443 | 994 | 31.1% | 10000 | 10000 | 0 | `deterministic_only` | `reject` | False |
| `sch-bench-39-failed-cas-reject` | `rejection_or_escalation` | 1437 | 581 | 59.6% | 10000 | 10000 | 0 | `deterministic_only` | `reject` | False |
| `sch-bench-40-concurrent-human-review` | `rejection_or_escalation` | 1437 | 931 | 35.2% | 2000 | 10000 | 0 | `human_review_required` | `escalate` | False |

## Notes

- Checked-in candidates are oracle/replay fixtures only (production_eligible=false).
- Wall-clock latencies are observational and excluded from --check equality.
- Both context modes use the same pinned tokenizer/estimator and hard coverage policy.
- Required target/test/opaque source is never omitted to improve reduction.
- Failed and escalated tasks remain in the denominator.
- Stale and simulated admissions are never counted as production accepted.

Observational run wall-clock: 14.71 ms (excluded from `--check`).
