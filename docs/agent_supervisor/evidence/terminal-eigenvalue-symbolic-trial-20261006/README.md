# Largest-eigenval: completed symbolic planning trial

The fresh official trial **largest-eigenval__CkAh6qM** received **reward 1.0**.
Its native symbolic planner covered eight authored requirement atoms and stored
one task with **zero planning-provider calls**. One coding session ran through
`llm_router` using `codex_cli`, `gpt-6.1-sol`, high reasoning and CLI 0.160.0.
Its complete observed usage was **255,951 tokens**. Native task completion,
STOP, cleanup exit zero and zero remaining processes were recorded; read-only
Docker inspection confirmed the exact owned container was removed.

| Retained trial | Official reward | CLI/router sessions | Observed tokens | Agent seconds |
| --- | ---: | ---: | ---: | ---: |
| Historical regular Codex | 0.0 | 1 | 161,750 | 288.11 |
| Historical supervisor, direct planning | 1.0 | 2 | 291,432 | 336.36 |
| Fresh supervisor, symbolic planning | 1.0 | 1 | 255,951 | 182.40 |

These are three individual observations. The fresh symbolic run used **35,481
fewer tokens (12.17%)** than the historical successful direct supervisor, and
its agent execution was **153.96 seconds shorter (45.77%)**. The historical
planning session accounts for **23,766** of those tokens; coding usage differed
by another **11,715**. Removing the planning session is demonstrated. The extra
coding-token and time differences are descriptive, not a causal advantage.
Regular Codex already had one session; this treatment removes the supervisor's
extra planning session. The earlier failed native trial's reward is unchanged.

The [bounded comparison](comparison.json) binds canonical receipt and official
result bytes, source-defined counter owners, the signed administrative admission
and the historical observations. Complete usage comprises **251,752 input
tokens** (including **217,472 cached input**) and **4,199 output tokens**.
Cached input is not added again; reasoning-output counters are also not added
again. Six native token-count records were observed inside the one coding
session. Sessions and token-count records are not individual internal API-call
counts. Dollar cost is unknown and no billing total is claimed.
The [Harbor aggregate readback](harbor-token-readback.json) independently agrees
with the native session's input, cached-input and output counters, and records
one completed trial, zero errors and zero retries.

Native planning took **4.18 seconds**, compared with 80.37 seconds in the
historical direct run. The supervisor's broader planning phase was **4.65
seconds** versus 80.97; the phase includes additional overhead. Coding-provider
time was **102.21 seconds** versus 186.79. Fresh setup took **223.58 seconds**,
and the full outer invocation took **428.35 seconds**. A separate cleanup
duration was not instrumented; cleanup status is available, not a measured
duration.

## What was held fixed

The original successful runtime archive was reused **without rebuilding**:
`a9f25619d230725ebe9f0f784d74feaf9191905ab78216fcedaee45a95c0cbcc`.
Its source is frozen at `8dab684ef69f87a88c3dac1585eea07a5bf29a24`, datasets
at `987cf856b2b902aa68c4587bb492b19b932b5d30`, and kit at
`a9b98beac1ef14278b4adf4f2289cd509530fb86`. The
[independent archive preflight](independent-archive-preflight.json) rehashed the
whole archive and checked eleven selected module bodies against the manifest
and frozen checkouts. The [independent native qualification review](independent-frozen-qualification-review.json)
confirmed the contract's eight requirements, native replay/admission and one
stored task under the frozen source, with zero providers.

The intended treatment was the explicit
`intent-plan-requirement-contract@2`, which selects `intent_symbolic`. Task
inputs, public profile, runtime archive, Source384 checkpoint/configuration,
provider identity, setup-cache policy and declared limits match the historical
direct arm. Actual Docker inspection recorded **five CPUs, 16 GiB RAM and
32 GiB combined RAM/swap**. The same 960-second outer limit, 840-second work
window, cleanup reserve and one coding attempt were retained. No historical
solution was reused. The model generated a fresh candidate from the original
public task, and the original benchmark inputs remained unchanged.

Strict configuration equality remains **false**: Harbor serialized the same
nine retry-exclusion names in a different order. Both configurations disable
retries, and the normalized comparison otherwise matches after removing output
identity and the intentional contract treatment. The first preparation gate
refused execution on that mismatch. Its failed gate is retained; the subsequent
explicit behavior-equivalent comparison permitted the same immutable
preparation to execute once, without editing it. See
[controls preflight](controls-preflight.json) and
[original strict gate](controls-preflight-before-retry-normalization.json).

The task/container image fingerprints differ from the historical arm, and
package resolution, time, host load, provider state and caches were not
controlled. This is not a randomized or repeated efficiency estimate, and the
same source archive does not make every environment byte identical.

## Scope of the symbolic result

The eight requirement atoms are agent-authored interpretation candidates.
Source accounting is complete and administrative coverage is accepted, but
source semantics, semantic alignment, proof, execution and completion
authority remain false in the planning contract/receipts. Its structural smoke
does not prove an eigensolver. Actual numerical correctness and speed are
supported by this official finite benchmark, not by a universal floating-point
or complex arithmetic theorem. The broader
[diagnosis and numerical-operator backlog](../terminal-eigenvalue-symbolic-diagnosis-20261005/README.md)
remains applicable.

The same Source384 checkpoint was consumed for advisory source context with
zero training and downloads. Its receipt binds an inference report and records
four source-contract-unsupported nominations. The candidates have no semantic
or numerical proof authority. The frozen consumer deliberately records
`neural_inference_replayed=False`: validating/reloading the retained inference
does not rerun the neural worker. The initial reporting producer incorrectly
expected that flag to be true; the corrected producer binds the exact consumer
source and checks the no-replay invariant. Neither runtime nor numerical
counters changed. The earlier report is retained privately.

The [completed metadata review](completed-metadata-review.json) directly checks
official results, complete native usage, signed-admission coverage, exact
archive identity, original inputs and lifecycle observations. It passed 21
checks. The primary agent completed that review; final delegated review turns
failed with usage-limit errors. The earlier independent archive and frozen
native reviews completed successfully and remain separate evidence. This
metadata review does not replay admission signatures or execute the verifier.

## Retained producers and publication boundaries

[run_matched_symbolic.py](run_matched_symbolic.py) records structured
prepare/execute commands, their exits and matched controls before the one
official execution. [inspect_symbolic_trial.py](inspect_symbolic_trial.py)
projects bounded metadata from exact source-bound counter owners. It never
opens model transcripts, hidden test/reference source or solution bodies.
Runtime setup logs, raw receipts, full source-bearing contracts and admission
envelopes stay in ignored local artifacts. The published subset contains
reviewed counters, hashes, provenance and producer scripts. Benchmark sources
and solutions are excluded from training; no model weights were created.

Published producer commands require the retained private local artifacts and
original repository layout; this folder is not a standalone runtime bundle.
The [container cleanup observation](container-cleanup.json),
[frozen native qualification](frozen-source-qualification.json),
[invocation provenance](invocation-provenance.json) and
[documentation gates](documentation-gates.json) supply the relevant readbacks.
The original shared workspace's source and index were preserved.
