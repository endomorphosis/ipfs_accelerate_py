# Full-supervisor instruction deduplication — 2026-10-12

Both fresh `largest-eigenval` trials passed the official Terminal Bench verifier.
Instruction deduplication used **309,813 versus 311,776 native tokens**, an
observed reduction of **1,963 (0.63%)**. The difference is principally cached
input. This single sequential pair, with independently generated plans, does
not establish repeatable savings or a billing advantage.

| Measurement | Full baseline | Full with instruction dedup |
| --- | ---: | ---: |
| Official reward | 1 | 1 |
| Planning tokens | 25,461 | 25,612 |
| Coding tokens | 286,315 | 284,201 |
| Total tokens / official pass | **311,776** | **309,813** |
| Cached input, already included in total | 236,800 | 234,624 |
| Uncached input | 69,141 | 69,109 |
| Output, including reasoning | 5,835 | 6,080 |
| Agent setup, seconds | 341.36 | 993.44 |
| Agent execution, seconds | 249.83 | 283.82 |
| Official verification, seconds | 7.47 | 11.49 |
| Outer execution including cleanup, seconds | 619.59 | 1,310.29 |

Uncached input fell by only 32 tokens; output increased by 245. Cached input
fell by 2,176. Reasoning output was 1,396 and 1,453 respectively, already included
in output. Provider cost is unknown. The slower second deployment progressed
through Python, system-package, bootstrap and npm installation; its time is
retained. Even excluding setup, the second agent execution took longer.

The treatment actually selected `common-bindings-instruction@1`, with no
fallback. Its complete rendered coding input shrank from **66,818 to 65,850
bytes**, a 968-byte reduction (1.449%). The baseline's independently planned
input was 66,486 bytes. These byte observations are distinct from cumulative
native session tokens. Both exact coding-input reconstruction audits passed.

The [paired result](paired-result.json), [20-check independent review](independent-pair-review.json),
[execution closure](live/pair-closure.json),
[baseline observation](live/baseline-runtime-observation-v2.json), and
[treatment observation](live/instruction-dedup-runtime-observation-v2.json)
retain the arithmetic and runtime evidence. Both native tasks completed, STOP
succeeded, and owned containers were observed absent twice. The two trials
used **621,589 tokens in four native CLI sessions**. Provider-free preparation
failures remain recorded separately; missing ledgers are not invented as zero
usage. Earlier census and transport experiments remain separate campaigns.

Both arms used the full supervisor through `llm_router`, the qualified Codex
0.160.0 fallback with GPT-6.1 Sol/high, lexical capsules, semantic transport v1,
ordinary completion replies, the same Source384 checkpoint, and 5 CPUs/20 GiB.
Retries were zero. The [27-check preparation review](preparation-02/independent-preparation-review.json)
verified identical settings except metadata representation and output paths.
Baseline ran first. Planning request identities and task CIDs differ; the
initial semantic root matches. This is a comparison between two full-supervisor
variants; native Codex and no-index arms were not run in this pair.

The setup-cache guard now admits the existing formalization resource profile,
and declared controls use the same explicit allowlist. Exact payload pins,
memory thresholds, and native admission checks are preserved. The profile
reserves 3,600 seconds for preparation, 660 for planning, and 1,860 for
implementation, with a 1,800-second coding provider cap. The
[focused qualification](formalization-cache-focused-01/qualification.json)
passed 138 tests with no skips. Host preparation also needed the existing
Harbor virtual environment to precede shared dependency directories; the
failed preparation and corrected import resolution are retained.

The measured runtime is accelerator `c3d4f873acdb62fd5b124328b3e3f28accd8dbdb`
and datasets `c1b1f65f254f5f3ee8f307cdddf923c62450f78e`. Archive05 SHA-256 is
`96aa70a6e10667ee9ca6d54a8baaae2a18378cdab22414c3bb32cc9fd65ed607`.
All 27,468 members were verified, and archive, manifest, and producer bytes were
rechecked unchanged after both runs. Publication also retains later repository
source-inventory work; the measured runtime remains this frozen snapshot.

Actual formalization coverage remains limited. Both runs consumed Source384
checkpoint `2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`,
but all four sampled functions were unsupported: zero source-aligned candidates
and no whole-program semantics verified. IntentIR reported
`fail_open_no_checkpoint` before decomposition and supplied no learned advice
to the planner. The [source diagnosis](diagnostics/public-source-formalization-gaps.json)
shows the selected SecurityIR validator's two-integer scalar grammar and the
separate IntentIR runtime's narrow scalar-contract sentence grammar. Additional
weights alone cannot extend these validators. The current cache layout also
refuses additional learned Intent/numerical assets.

After both timed containers closed, a separate model-free
[numerical projection probe](numerical-public-probe-01/qualification.json)
used only the hash-matched public `eigen.py` and `eval.py` inputs. The existing
`python-numerical-structure@1` path linked all four functions and structurally
translated two completely. It produced ProgramIR and first-order/higher-order
artifacts, 45 pending obligations, two eigenpair contracts and four assertion
requirements. The real metadata/world DuckLakes reopened and replayed in a
fresh process. Native Lake 4.29.1 compiled both generated modules; removing a
required nonempty premise correctly failed. Six negated abstract SMT queries
returned SAT. **Zero source-correctness obligations were discharged.** The
probe took 8.06 seconds and executed no target code, models, or training. Its
[independent review](numerical-public-probe-01/independent-review.json) passed
17 checks, including unchanged original benchmark inputs.

The numerical summary's 8,192-byte bound retains all four source-unit links,
both compound contracts and four assertion requirements, but omits all 45
individual obligation rows and three per-function summaries. Omission counts
are explicit. Further integration should provide bounded retrieval of that
unresolved work, alongside an opt-in structural numerical profile, a broader
validated IntentIR input grammar, and qualified packaging for additional
checkpoints. Those are separate treatments requiring new measurements.

The first offline pair reporter incorrectly compared Harbor's qualified task
name with the short task name. Its [failed report](reporter-initial/failed-pair-report.json)
remains retained. The current reporter reuses the existing exact public
`task.toml` name and task-path matcher. This reporting correction required no
model rerun. The [current reporter qualification](reporter-current/qualification.json)
passed 23 cases, including incorrect qualified-name and task-path refusals;
the initial 21-case qualification remains distinct. The [artifact manifest](artifact-manifest.json)
binds the bounded public evidence; raw provider transcripts, credentials,
owner databases, model weights and runtime archives are excluded.
