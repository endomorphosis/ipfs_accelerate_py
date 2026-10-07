# Compact transport increases token use in the qualified pilot

The corrected `largest-eigenval` pair passed all 168 comparison gates and 46
independent accounting checks. Both tasks earned official reward 1 and completed
native shutdown with zero remaining processes. The compact transport used
**80,125 more tokens (32.4%)**. Keep transport `@1` as the default; this result
supports no token-saving claim for `@2`.

| Corrected arm | Planning tokens | Coding tokens | Total tokens | Reward |
| --- | ---: | ---: | ---: | ---: |
| Baseline `@1` | 25,243 | 221,749 | 246,992 | 1 |
| Compact `@2` | 25,192 | 301,925 | 327,117 | 1 |

LLM planning and coding both remain enabled through `llm_router`. Each arm uses
one planning and one coding CLI session, the same `ordinary-completion@1` reply
contract, Codex 0.160.0, `gpt-6.1-sol` with high reasoning, 5 CPUs, 16 GiB,
cache policy `source384-native-aarch64-dontneed@2`, hash seed 0 and zero retries.
Both native coding-input audits verified the complete task prompt, including
the fixed 506-byte reply instruction. The generation schema is 215 bytes.
The acknowledgment grants no execution, completion, settlement or retry authority.

The complete initial coding input was 59,731 bytes for baseline and 57,160 for
compact. Their native inputs were 62,443 and 61,172 bytes, respectively. Different
generated plans therefore contribute to the observed 2,571-byte input reduction.
Coding token-count records were 6 and 8; those records are not API-call counts.
Initial byte reduction did not reduce complete native session tokens in this pair.
Cached input is already included in input totals, and reasoning output is already
included in output totals. This single pair does not establish repeatable or
causal effects.

The [qualified result](qualified-pair-result.json),
[unchanged-producer comparison](qualified-pair-comparison.json),
[independent review](qualified-pair-review.json) and
[full publication report](qualified-pair.md) bind the outcomes and accounting.
The corrected runtime is frozen at
`7a870ac52aefe798eef4d48319f168e62d840ae8`; archive SHA-256 is
`ea6a6806d62264f14801c2ad5c8f34a72a9478112930bb4320c1083784778797`.
Its [preflight](corrected-runtime-preflight.json) passed 245 actual archive checks
and 100 prepared-pair checks. The snapshot also contains concurrent worker,
lifecycle and custody work; paired arms use that same immutable snapshot.

The [initial reply-contract pair](initial-pair.md) also passed both tasks but
remains unqualified: its auditor dropped the recorded provider during subprocess
projection, causing `InvalidReceipt` before reconstruction. Its 659,361-token
cost and failed audits are retained. The baseline context was unavailable after
cleanup; restoring metadata alone cannot establish reconstruction.
The [diagnosis](audit-defect-diagnosis.json),
[fix qualification](ipc-fix-qualification.json) and
[independent fix review](ipc-fix-review.json) show the conditional provider
propagation and 37 passing tests, including the real audit subprocess. Missing
and foreign providers still fail; legacy IPC shapes and native authority checks
are preserved. The previous failed benchmark attempts also retain their costs.

The corrected pair costs **574,109 tokens**. Across the seven retained comparison
attempts, the campaign costs **1,885,347 tokens in 13 native CLI sessions**. This
campaign total covers the listed comparison attempts, not every project run.
No model weights changed or training ran in this experiment.

The [updated improvement backlog](next-treatment-backlog.json) and its
[independent review](next-treatment-review.json) retain planning. P0 is an opt-in
reference to acceptance criteria already inline in the typed task contract,
with exact reconstruction and an independent planning-input audit. Its expected
impact is bounded: coding accounts for 89.78% of the corrected baseline tokens.
Subsequent experiments prioritize typed DuckDB/Quack metadata, native formal
status cards, qualified bounded owner lookups and checked DuckLake history
deltas. The generic Quack owner gateway remains unqualified. Missing, stale,
inconclusive, unsupported, error and cancelled evidence cannot become proof or
omission authority. The 8D, 384D, 768D and possible Leanstral embeddings can rank
facts; semantic completeness and proof require their separate native checks.
Each treatment needs complete native token, lookup, repair, reward and cleanup
accounting. No further provider experiment is included here.

The [artifact manifest](artifact-manifest.json) binds the published evidence.
