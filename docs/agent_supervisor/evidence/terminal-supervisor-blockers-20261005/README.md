# Terminal-Bench supervisor recovery and remaining blockers

The fresh full indexed supervisor run passed the original `largest-eigenval`
verifier with reward **1.0**. Native START and STOP succeeded, the task reached
completed revision 4, cleanup returned zero, and no worker processes remained.
This is one development-profile trial, not a full-suite score.

| Fresh trial | CLI | Official reward | Agent seconds | Setup seconds | Observed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| [02](trial-02-summary.json) | 0.158.0 | 0.0 | 24.48 | 231.58 | Unknown |
| [03](trial-03-summary.json) | 0.160.0 | 1.0 | 336.36 | 213.21 | 291432 |

Both use `gpt-6.1-sol`, high reasoning, through `ipfs_accelerate_py.llm_router`.
Trial 02 stopped during planning. An isolated compatibility check using the same
host account, model and prompt found that CLI 0.158 rejected the model with HTTP
400 while CLI 0.160 returned successfully. The runtime pin now selects 0.160
consistently for preparation, planning, deployment and the native baseline.
New preparations reject incompatible archives; historical collection preserves
the original model and CLI instead of relabelling old results.

The successful run contains two provider sessions: planning took 74.62 seconds
and coding took 186.79 seconds. Observed input tokens total 285558, including
244992 cached input tokens; output tokens total 5874. Cached input is already
included and must not be added again. Total invocation time was 570.62 seconds,
including setup and verification overhead. Billing cost remains unknown. Missing
usage in trial 02 remains unknown, not zero. These runs also differ in source and
cache policy; they are not a controlled estimate of CLI performance or savings.

The integrated run indexed four symbols, produced eight full capsules and four
worker capsules, and loaded the frozen SecurityIR Source384 checkpoint
`2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`.
Actual inference covered two Python program files in 8.93 seconds, with the three
harness support files retained separately. Training steps were zero. Planning
qualified two goal records and one task. IntentIR was fail-open without a selected
checkpoint. The Doctor reported residual work and selected `model_router`;
this does not demonstrate symbolic-only synthesis, all four IR checkpoints, or
formal proof of the task's natural-language requirements.

The frozen Docker sources are accelerate
`8dab684ef69f87a88c3dac1585eea07a5bf29a24`, datasets
`987cf856b2b902aa68c4587bb492b19b932b5d30`, and kit
`a9b98beac1ef14278b4adf4f2289cd509530fb86`. The accelerate merge includes concurrent
checkpoint, staged patch and ordinary supervisor lifecycle fixes. The independent
[archive audit](build-03-audit.json) verified all 27342 members against their
hashes, sizes, modes, ownership and clean source trees. Container setup verified
the newly pinned native CLI files under cache policy
`source384-native-aarch64-dontneed@2`. The historical `@1` policy retains its
original CLI 0.158 hashes. Cache advice grants no admission authority.

The selected profile remains five CPUs, 16 GiB, 840 seconds of supervisor work
and 60 seconds reserved for cleanup, with a 960-second outer agent timeout.
Provider calls remain capped at 300 seconds; the successful coding call stayed
within that limit. There was one worker and one attempt per trial. Original task
inputs stayed unchanged. No benchmark training, hidden-test or solution bodies,
credentials, or model responses are published here.

Additional fixes retain closed provider and semantic response failure codes
without exporting raw error text or weakening rejection behavior. Strict ablation
flag validation now runs before archive access. The merged
[qualification](final-qualification.json) passed **560 tests**, with seven skips
and no failures, using a fresh DuckDB AST seal and unchanged source hashes.
Six skips require root container deployment tests; one requires a separately
selected published IntentIR action checkpoint. A pytest timeout-marker warning
remains. Earlier failing and repaired runs are retained separately and are not
summed. These host tests were not a hermetic Python environment; Docker sources
were independently frozen and audited. Raw local logs/XML remain available by
their recorded hashes.

The full suite is still unfinished. [Readiness evidence](final-preflight.json)
identifies the remaining blockers:

- **85 tasks lack trials under this indexed methodology.** Their rewards are
  unknown. The older three-task pilot used a different model/source profile;
  its results cannot be pooled with this retry into a matched suite score.
- Exact task profiles still need coverage for other languages and data formats,
  build/install/training stages, generated artifacts, services/VMs, Git state,
  external data and paths outside `/app`.
- Empty and zero-symbol retrieval exists, but selected Source384 inference still
  needs an independently bound empty-source abstention. Generic planning remains
  one task, and reviewed symbolic repair operators cover a narrow Python subset.
- Grok host readiness is not container qualification. The full benchmark still
  needs an immutable Grok route, isolated authentication transport and native
  usage accounting. This run used an explicit Codex comparison profile.
- Native task budgets differ from the capped development profile. All 89 native
  agent time allowances sum to 41.54 hours before setup and verification; this is
  an allowance sum, not a runtime forecast. Matched native and no-index baselines
  have not been rerun, so no token-efficiency advantage is claimed.

Host memory, disk and Docker availability did not block this retry. Leanstral was
already inactive; this work did not stop or restart it. The separate
[budget audit](implementation-cap-audit.json) records how a future explicitly
selected larger provider-call window could preserve signed limits and cleanup;
no such budget change was made in these trials.

The trial summaries are bounded observations, not replacement canonical receipts.
They bind original receipts, frozen producers and manifests by SHA-256. The
versioned inspector sources are retained. Producer commands contain historical
local paths; this evidence directory is not a self-contained runtime archive.
