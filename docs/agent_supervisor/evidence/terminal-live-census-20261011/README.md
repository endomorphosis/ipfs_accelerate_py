# Live Source384 input census, 2026-10-11

The [successful closed observation](actual-live-census-02.json) measures the
coding model prompt after fresh Source384 preparation, an actual isolated
planning session through `llm_router`, and native signed admission. It stops
before coding, active Doctor repair, implementation START, validation and the
official verifier. Planning remains enabled.

| Observation | Result |
| --- | --- |
| Original coding model prompt | 59,336 UTF-8 bytes |
| Selected common-bindings prompt | 58,454 UTF-8 bytes |
| Reduction | 882 bytes, 1.48645% |
| Named byte proxy | 14,834 to 14,614; 220 fewer units |
| Exact restoration and complete bounds | Passed; both below 256,000 bytes |
| Current fact/full capsule/worker capsule counts | 4 / 8 / 4 |
| Planning sessions / coding provider calls | 1 / 0 |
| Native planning input / output / total tokens | 23,707 / 1,734 / 25,441 |
| Cached input / cache write / reasoning output | 0 / 0 / 179 |
| Source384 inference / initial context seconds | 18.8251 / 23.7847 |
| Owned setup/run/cleanup seconds, successful attempt | 555.3065 |
| Official reward / task completion | Unmeasured / false |

The proxy is `utf8-bytes-ceil-div4@1`, not a provider tokenizer. Cached input
belongs to native input; reasoning output belongs to native output. Usage is
counted once per invocation rather than summing cumulative events. All six
categories were observed, but billing and monetary cost were not independently
verified. The coding output schema is separately bound at 215 bytes. CLI system
prompt and output-schema tokens are outside the input-byte census.

The native Source384 encoder used fresh GTE-small inference. Retrieval reused
the initial four-symbol lexical TF-IDF index without new admitted embedding
calls. Neural inference and lexical retrieval are distinct; this observation
does not establish learned-vector retrieval. No training or new model weights
were produced.

The retained [first attempt](actual-live-census-01.json) refused the exact 20-GiB
profile in setup-cache prerequisites before container start, provider dispatch
or inference. A narrow cache allowlist fix preserved the original populations,
file guards and resource observation. A new frozen bundle was built for the
successful retry. The first attempt's wall time is unknown, so complete
wall-time and cost totals across both attempts remain unknown.

## Runtime and qualification boundaries

The [source freeze](source-freeze-03.public.json) binds 24 runtime bodies, base source
HEAD `036ce8679b8f9d926ed0ac95dc03743673a4fb16`, datasets HEAD
`5c69724316a3365bdb22023ba5608fced96d568a`, and kit HEAD
`8ac00bb1974b1378e12091d546d8425af34049e0`. It includes the uncommitted runtime
changes by exact body hash. Later source publication or concurrent merges do
not change the runtime actually exercised by this observation.

[Archive readiness](archive-readiness-03.json) records the exact source,
provider, reply and worker-capability checks before the live run. It is
preparation evidence, without an execution claim. The
[owner-boundary review](owner-boundary-source-review-03.public.json) is a source review
that required cache regression qualification before retry. Those regressions
subsequently passed as recorded in the qualification ledger.

The [final native qualification](final-native-qualification-03.json) records
**138 passed, zero failed, errored or skipped**, with 138 fresh mandatory AST
source seals and zero completion-authority seals. It covers authored controls
and current native owner fixtures using the deployed default global-metadata
configuration. The live helper body is included in the test import closure;
23 of the 24 frozen runtime bodies intersect that source catalog and all match.
That intersection is not a claim of exhaustive execution coverage.

The [earlier qualification ledger](test-qualification-02.public.json) preserves
194 driver/runner passes, 192 cache/resource/archive passes, earlier individual
suites and historical source transitions. Counts overlap and must not be added
as unique tests. Historical manual AST/hash checks are distinct from mandatory
native pytest sealing. An earlier collection error and a deliberately
interrupted optional-global-mirror run remain recorded as historical partial
results. They do not qualify the final current runtime. The final 138-test run
supersedes the earlier pending current-datasets integration gate.

The [independent actual-result review](actual-live-census-review-01.public.json)
checks the closed arithmetic, token categories, exact digest joins, source
bindings, explicit limitations and cleanup. It does not inspect private owner
state or independently repeat the native execution. Actual runtime observations
record schema custody, retained owner planning receipt equality, post-yield
owner/world/task/source revalidation, guarded worktree removal, worker cleanup
and two exact-container absence observations.

This directory contains closed metadata and authored-test output only. Private
runtime prompts, catalogs, credentials, provider traces, owner databases,
container state, orchestration scripts, archives and weights are excluded.
[The artifact manifest](artifact-manifest.json) binds the published bodies by
bytes and SHA-256; it excludes its own body. Public metadata projections replace
local operational paths with named aliases. They retain the original artifact
digest and historical scope; run receipts refer to those original digests,
not the projected body. Original private metadata remains retained locally.

## Decision

This completes the genuine planned-input census gate. The measured 1.49%
reduction does not establish complete-run token savings or benchmark success.
The [updated plan](../../terminal_token_metadata_plan.md) prioritizes current
accepted formal-status reuse and bounded owner lookup, qualified against current
sources and unchanged obligations before another complete-token comparison.
Conditional spectral lemmas do not certify the whole eigenvalue task. Existing
negative complete-token comparison results remain relevant; planning stays in
the evaluation and every retained attempt contributes to usage accounting.
