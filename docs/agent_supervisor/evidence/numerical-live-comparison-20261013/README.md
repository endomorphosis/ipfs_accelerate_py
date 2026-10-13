# Fresh native Codex, no-index and indexed supervisor pilot

All three fresh `largest-eigenval` attempts passed the original verifier. The indexed supervisor used **24,519 fewer tokens than no-index (6.84%)**, but **220,090 more than native Codex (2.9326 times native)**. This is one sequential attempt per arm, not a general advantage result. The indexed system's extra planning cost and remaining formalization gaps are visible below.

| Arm | Reward | Input tokens | Cached input | Output tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Codex harness | 1.0 | 107,689 | 86,016 | 6,194 | **113,883** |
| No-index supervisor | 1.0 | 346,026 | 283,392 | 12,466 | **358,492** |
| Full indexed supervisor | 1.0 | 327,785 | 265,600 | 6,188 | **333,973** |

Cached input is included in input; it is not added again. Each distinct provider session contributes its final cumulative counters once. All three observations have complete reported native usage. These counters are not an independently verified billing statement. Post-publication refresh accounting was incomplete: auxiliary embedding and text-generation usage was not fully reported. The table therefore covers the complete observed native CLI/router counters, not exhaustive whole-pipeline usage or cost. The [three-arm observation](three-arm-observation-final.json) retains all registered attempts, outcomes, controls and token categories.

## Runtime and phase costs

| Arm | Environment setup s | Agent setup s | Agent execution s | Verifier s | Launch elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Codex harness | 3.07 | 99.44 | 181.69 | 2.05 | 300.13 |
| No-index supervisor | 3.26 | 333.58 | 443.08 | 2.15 | 800.25 |
| Full indexed supervisor | 3.15 | 235.39 | 271.76 | 2.08 | 529.63 |

Launch elapsed time comes from the outer launch receipt; phase durations come from Harbor's interval records. They are distinct measurements and are not forced to add to the same total. Provider installation, package fetching and supervisor runtime installation affect setup time. The trials ran in the order no-index, indexed, then native; host conditions, package caches and provider variability were not randomized.

| Supervisor phase | No-index | Indexed |
| --- | ---: | ---: |
| Planning tokens | 20,965 | 27,526 |
| Coding tokens | 337,527 | 306,447 |
| Planning phase s | 87.48 | 48.83 |
| Planning provider invocation s | 83.17 | 39.35 |
| Coding provider invocation s | 331.77 | 135.82 |
| Initial indexed-context preparation s | Not selected | 14.98 |
| Later indexed-context preparation s | Not selected | 4.83 |
| Doctor phase s | Not selected | 1.56 |
| Post-publication refresh s | Not selected | 15.76 |

Each supervisor recorded two router sessions, one planning and one coding. Native Codex recorded one native session; its internal planning is not separately measured here. Session counts are not counts of internal model turns. The indexed arm spent 6,561 additional planning tokens and 31,080 fewer coding tokens, producing the net 24,519-token reduction relative to no-index. Nested provider/context durations are not added again to agent execution time.

## Controls and scope

All arms used the original task and verifier, model `gpt-6.1-sol`, high reasoning effort, Codex CLI 0.160.0, fresh Docker task containers, five CPUs, 20,480 MiB, a 6,480-second outer agent limit, a 1,800-second agent setup limit, one attempt and zero harness retries. Both supervisors used `llm_router`; native used Harbor's native Codex adapter. The host's global CLI was unchanged: native preparation used an isolated, hash-verified 0.160.0 executable, while the container kept Harbor's original pinned installation behavior.

The supervisors shared the same runtime archive, pinned Source384/GTE assets, legacy metadata and reply modes, disabled source views, and default lexical capsule selection. No new setup-cache policy, target training, Intent384 checkpoint or learned numerical selector was added. They requested the same `python-numerical-control-structure@3` profile; no-index explicitly disabled its indexed consumption.

The independent preparation audit verified the supervisor configurations differ only in arm and job/output naming. Across all three arms, the artifact collector verified equal identity, task, job and agent control digests, while allowing exactly one intentional experiment difference: the native harness does not select the numerical formalization profile. The unmodified generic production comparator therefore still reports a mismatch. The explicit three-arm audit records that difference instead of fabricating a native numerical selection or weakening the comparator.

Equal outer limits do not mean equal internal provider budgets. Supervisors reserved preparation time and capped their planning provider at 600 seconds and coding provider at 1,800 seconds; native coding could use its outer agent allocation. This experiment compares harness behavior under shared outer resources, with different internal scheduling.

The frozen source pins were `ipfs_accelerate_py` **a8f57366ff4110a87f1f13d1e60992f83f689821**, `ipfs_datasets_py` **28b9ef1f0b9da51e87b1e68ee565b43681397109**, and Kit **8ac00bb1974b1378e12091d546d8425af34049e0**. The supervisor runtime archive SHA-256 was `488ae75f90bd9c4c9cc88b520d4db68157984625ea8bbef3874f63da48b5f94a`. The [preparation audit](preparation-independent-audit.json), [native release](next-native-baseline/release.json), and [native preparation review](next-native-baseline/native-preparation-independent-review.json) retain the exact bindings. No producers or frozen pair controls were changed between those trials.

Grok authentication was refused in isolated metadata readiness checks, so the entire comparison used the explicit Codex fallback. Those preflights generated no coding tokens and are separate from the three benchmark attempts. The [route audit](provider-readiness/route-audit.json) retains their scope.

## What the indexed arm actually used

The [locked supervisor observation](pair-observation-final.json) records four indexed symbols, eight full capsules, reuse of the initial indexes, four worker capsules and four replayed native fact rows. Initial indexed summaries were supplied to the planning router. This confirms indexed context participation; it does not isolate which part changed model behavior.

The deterministic numerical frontend structurally covered four functions across two source files. Its 8,064-byte planning summary exposed eight pending hints from **105 unresolved obligations** and recorded a count of two compound contracts, whose bodies were omitted from the compact summary. **Zero source obligations were discharged.** The numerical retrieval session recorded **zero requests**, so the live result does not demonstrate model use of exact-ID paging or the previously measured query-byte reductions.

Source384 used the matched pinned parent checkpoint with zero training steps. It produced four decoded, unverified candidates, all marked `fail_open_source_contract_unsupported`. That learned-decoder limitation is separate from the deterministic frontend's structural coverage. The Doctor abstained. No Intent384 checkpoint was selected; this run does not establish a trained multi-IR pipeline or coordinated LegalIR/UI/UX IR inference.

The retained owner hydration receipt reports three DuckLake catalogs, three links and one world record. The [83-check full-pair review](full-pair-independent-review.json) verified the retained hydration receipts, recorded persistence-file hash bindings and the pinned production snapshot/reopen code path. It did **not** independently reopen the databases. This evidence must not be expanded into a claim that every persisted row was replayed, a complete proof index was built, or symbolic state stayed current through every edit.

Two final-state gaps remain:

- Numerical and Source384 successor contexts were `successor_unavailable`, with `needs_successor=true` and `current=false`, despite generic post-publication refresh reporting `refreshed`. Initial context delivery does not establish a formalization of the final edited program.
- Declared output-hash observation was unavailable in both supervisor arms at `allocation_event`. Native collection does not provide that observer either. Official reward is available, but byte identity of final outputs is not established by these receipts.

## Follow-up work and evidence boundaries

A separate output-observer fix passed **132 focused tests** after integration, recorded in [its latest JUnit result](output-observer-queued-tests-03/junit.xml). It was not part of this frozen runtime and has not been qualified by these live outcomes. It needs a fresh live qualification before being credited with repairing this observation gap.

The next useful work is to repair final-state successor publication, complete native output observation, improve Source384 decoder coverage, and test whether the model actually requests the formal obligation rows. A future controlled experiment can then isolate context selection or retrieval from the rest of the indexed bundle.

The earlier metadata-projection pair recorded 313,979 and 313,980 tokens for two full-supervisor runs. Neither is a baseline for this new comparison. Likewise, prior provider-free retrieval byte reductions are not live token savings. The [historical recipe erratum](historical-recipe-erratum.json) corrects the older pair's reply-mode description without changing frozen controls.

The [23-check three-arm review](three-arm-independent-review.json) independently reproduced the final join, accounting and explicit control audit with no blockers. This pilot establishes the three observed outcomes and their reported costs. It does not establish causal savings, parallel speedup, whole-suite performance, proof of task correctness, or an advantage over native Codex. Raw prompts, CLI transcripts, authentication/session material, databases and hidden verifier contents are excluded from the public report. Database replay, complete auxiliary usage accounting and final-state formalization remain outside the established evidence.

## Evidence seal

This directory contains reviewed aggregate receipts and helper sources, not raw
model transcripts, credentials, databases, hidden verifier contents or binary
assets. Helpers that replay launches require the separately retained local trial
inputs; this seal alone is not a runnable benchmark distribution. No trial is
rerun by verifying its file hashes. Earlier failed development fixtures and the
failed Grok setup probe remain labeled in their qualification records.
