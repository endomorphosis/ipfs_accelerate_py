# Fresh public semantic pack measurement

Runtime `19dbb524c80d6fdfedd5ce15e696cbf911eed8c1` passed **177 runtime checks**,
**47 launcher/exporter checks**, and a fresh public `largest-eigenval` Docker
census. The [closed result](actual-public-census-01.json) and
[operational summary](operational-summary-final01.json) record:

| Complete router coding input | UTF-8 bytes | Reduction from original |
| --- | ---: | ---: |
| Original | 67,201 | — |
| Capsule references | 67,120 | 81 (0.1205%) |
| Inline evidence objects | 66,452 | 749 (1.1146%) |
| Semantic pack factoring | 66,896 | **305 (0.4539%)** |

All three candidates bind the same original complete-input hash and unchanged
6,579-byte wrapper. They are independent representations, not a composition;
their reductions cannot be added. Each selector retains the original unless its
complete candidate is strictly smaller. This is not a native Codex versus
supervisor efficacy comparison. Inline and pack generation dispatch remain
disabled.

## Pack reconstruction and overhead

The pack itself shrank from 6,266 to 5,106 bytes. Six fields shared by all six
references moved to one inline defaults map: current schema and contract
version, empty repository/tree/path strings, and integer-zero byte counts.
Each field must explicitly match on every row. Missing or differing values
stay literal. The 36 factored fields retain their exact reconstruction rule;
every reference remains a separate row with its original role and metadata.

The duplicated outer token estimate is exactly reconstructed from unchanged
inner-pack totals, estimator version and integer sum. Extra estimate fields,
different duplicates and valid incomplete estimates remain literal. No
estimator or provider token counter runs as part of this recipe.

Together these changes remove 1,160 bytes inside the pack. Recipe, instructions
and view structure add 855 bytes, leaving 305 bytes of net complete-input
reduction. Exact restoration and deterministic reprojection passed. Reference
roles, requirements, caveats, identifier representations, the inner contract,
unknown fields and raw source remain intact. Producer identities and admission
artifacts are unchanged. This establishes representation equality, not formal
correctness of program behavior or permission to omit facts.

## Native lifecycle and limits

The profile used full indexed preparation, pinned Source384 inference,
`declared-contract@1` planning, admission, the ordinary worker/source boundary,
owner replay and cleanup. Intent autoencoder inference was disabled. Source384
inference ran in 11.241 seconds and its retained checkpoint/configuration/
inference hashes matched owner currentness checks. Replay did not rerun neural
inference or qualify all IR models.

Initial context took 16.512 seconds, declared planning 4.698, admitted context
3.964 and the census 30.260. The complete owned Docker lifecycle took 220.226
seconds; the outer execution took 221.549 seconds. Timings overlap and come
from one run; they are not comparative throughput evidence.

The capsule preflight made one authenticated request: 4,096 response bytes,
including 2,327 evidence bytes and zero bytes delivered to a provider. Complete
capsule expansion was not verified in this run. The pack and inline-evidence
candidates retain the facts needed for their reconstruction inline.

The owner reproduced the complete inputs and closed observation after the
source session stopped, joined the retained planning receipt and revalidated
admission after return. Source-session accounting was complete, worker cleanup
returned zero, the allocated worktree was removed, and the exact container was
confirmed absent twice. The external exporter verifies closed arithmetic and
provenance joins; it does not independently replay private input semantics.
No hidden evaluator, solution or provider-response reads were reported.

Zero LLM generation calls ran. Provider token usage and cost are unmeasured;
official verification was disabled, task completion is false and reward is
unknown. The measurement excludes CLI system prompts, subsequent tool responses
and repeated request history. No services or resource limits were changed.

## Qualification and retained evidence

[Qualification](qualification-final01.json) joins 55 codec/pack checks (39 new
projection and 16 pack census), 116 router/worker/control checks, and 6 native
owner-replay checks across all three census versions. Raw-required, exact and
conservative capsule contracts, unknown fields, numeric types, fallback, recipe
tampering, legacy contracts and no-I/O behavior are covered. Two authored test
fixtures were corrected during development; failed attempts remain in local
artifacts and are excluded from the final passing total. All changed production
files and each batch's selected tests match the final qualified hashes.

All 27,554 runtime archive members passed verification. The
[final freeze check](final-freeze-check-01.json) verified the runtime and two
dependency commits, 14 qualified source hashes, 11 script/manifest bindings,
21 public-input/asset bindings and recipe after cleanup. The
[manifest](manifest.json) binds the retained closed evidence and design. Raw
prompts, private owner archives and provider transcripts are excluded.

The [next measurement design](history-accounting-design.md) targets bounded tool
event sizes and repeated payload digests in the existing Grok stream parser.
That observer is not implemented yet. Stdout event repetition cannot establish
provider resends: actual request/history bytes must remain unknown until a
supported request-boundary hook supplies evidence after CLI compaction. Native
terminal token accounting remains unchanged.
