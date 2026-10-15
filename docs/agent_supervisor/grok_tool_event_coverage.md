The supervisor's revised Grok telemetry resolves the tool-name and payload-observation gaps seen in the earlier diagnostic. A fresh Docker run retained all 45 tool events, resolved all eight named calls through the documented shell-tool alias, and exceeded no payload bounds. Eight updates explicitly supplied a null status, so event coverage remains conservatively incomplete. This run does not establish token savings or a Terminal Bench score.

The measured runtime is `7c68549bc4e57ff0d1736fefbf2db94b081da013`. The [closed diagnostic](evidence/tool-event-coverage-20261015/live/actual-public-diagnostic-01.json), [arithmetic analysis](evidence/tool-event-coverage-20261015/closed-arithmetic-analysis-01.json), and [evidence manifest](evidence/tool-event-coverage-20261015/manifest.json) contain fixed categories, counts, and hashes without raw prompts, tool arguments, outputs, or private owner artifacts. The earlier [diagnostic](grok_tool_event_census.md) and its receipts remain unchanged.

Ordinary Grok supervisor coding now explicitly selects `grok_tool_event_schema="native-grok-tool-events@2"` through `llm_router`, producing progress `native-grok-stream-progress@5`. The shared parser defaults remain progress @3; its existing census opt-in still produces progress @4 with tool events @1. Planning and structured candidate routes retain their existing behavior. Periodic callbacks remain count-only progress @1, while detailed observations are retained in final invocation receipts. Native final-envelope usage accounting is unchanged.

Tool-name shape and bounded hashes distinguish missing, invalid, and unresolved names. The separate resolved classification recognizes the documented `run_terminal_command` alias as `run_terminal_cmd`; validation recomputes this classification from the name hash. Original exact-name counters remain unchanged. Unnamed updates remain unknown and do not inherit names from other events. Hashes do not authenticate event origin.

The new payload observer allows 131,072 nodes while preserving the 256 KiB payload/frame byte ceilings, depth limit of 32, and 4,096-bit integer limit. This admits dense bounded JSON that exceeded the older 4,096-node observation limit. Bound reasons now distinguish bytes, nodes, depth, and integer bits. The first 64 tool events remain the retention limit. Accepted terminal trailers no longer imply post-final tool events; retained post-final tool rows, omitted rows, and parser failures remain conservative coverage limits. Historical @1 receipts retain their original interpretation: the seven oversized output payloads in the previous diagnostic still have unknown bound reasons.

The fresh public-input `largest-eigenval` run used the full supervisor, Source384 preparation, declared-contract planning, Grok 4.7 through the pinned Grok 1.0.46 CLI, and the original semantic transport and legacy working-set/metadata/reply profiles. IntentIR inference was disabled. There was one coding invocation with an eight-turn cap, zero planning provider calls, and no Doctor continuation, retry, fallback, or official verifier. The 5 CPU/20 GiB resource profile and 32 GiB disk admission floor were unchanged. Pruning configuration was unchanged; this was a telemetry qualification, not a matched optimization ablation.

| Observation | Fresh diagnostic |
|---|---:|
| Observed / retained / omitted tool events | 45 / 45 / 0 |
| Calls / lifecycle updates | 8 / 37 |
| Calls resolved through documented alias | 8 |
| Supplied unresolved tool names | 0 |
| Updates with missing / supplied null status | 0 / 8 |
| Payload fields exceeding any bound | 0 |
| Retained tool-frame bytes | 284,936 |
| Measured rawInput canonical bytes | 13,321 |
| Measured rawOutput canonical bytes | 223,105 |
| Measured content canonical bytes | 41,378 |
| Repeated rawOutput / content bytes | 28 / 17,579 |

The [coverage cross-check](evidence/tool-event-coverage-20261015/closed-coverage-analysis-01.json) shows that `unknown_event_shape` is the only remaining incompleteness flag: eight updates contain explicit null status. All 45 rows have zero unknown top-level fields. The stream completed without timeout or taint, all payload observations fit the bounds, and no retained tool event occurred after the final event. A valid trailer accounts for the parent's one post-final event. Null status is kept distinct from omitted status; it does not establish a completed, failed, or unchanged lifecycle state. The [pinned public protocol audit](evidence/tool-event-coverage-20261015/null-status-protocol-audit.json) found no documented rule equating null update status with unchanged or omitted status, so conservative handling remains.

Content repetition was 42.48% of measured content bytes: 17,173 bytes within the same call and 406 across calls. The repeated rawOutput bytes were seven repeated null values. Payload channels can overlap and must not be summed as unique output. Repeated stdout events do not establish history resends, provider request sizes, or retained model context. Request-history bytes remain unknown.

| Native usage category | Observed incomplete subtotal |
|---|---:|
| Uncached input | 41,336 |
| Cached input | 248,064 |
| Output | 11,256 |
| Total | 300,656 |

Native usage completeness is false; complete token totals and verified cost remain unknown. The CLI reached `max_turns_reached`, exited 1, and returned no usable final response. Supervisor task completion was false. The diagnostic wrapper exited 0 because measurement collection and cleanup succeeded. No official verifier ran, so there is no benchmark reward or correctness score. The subtotal must not be treated as evidence of savings relative to an unmatched earlier run. The pinned CLI's final prompt ledger also excludes compaction and other side-model calls.

The lifecycle took 395.468 seconds, the outer execute step 396.857 seconds, and the Grok stream 163.765 seconds. Source384 preparation retained a checkpoint-bound inference record taking 10.176 seconds; independent worker consumption and formal correctness are not established by that receipt. All 27,555 archive members were verified before launch. Runtime, dependency, source, script, and asset bindings stayed unchanged. Worker cleanup returned 0; exact container absence was observed twice and independently rechecked twice. The [final freeze check](evidence/tool-event-coverage-20261015/final-freeze-check-01.json) rechecked 16 source files, 11 script/qualification bindings, and 21 assets before the source freeze was released for this documentation.

Validation passed **729 cases**: 292 parser/capture/history/usage checks, 312 routing/reporting/archive checks, 80 codec checks, and 45 authored subprocess/launcher/export checks. The [qualification](evidence/tool-event-coverage-20261015/qualification-final01.json) binds their exact source and dependency versions. Two initial fixture expectations were corrected: legacy tool-name counters count calls only, and unnamed updates do not inherit a resolved name. Production semantics were not weakened to satisfy those fixtures. No provider calls occurred during these tests.

The next token-reduction experiment is an explicit paired pruning profile in the fresh benchmark container's Grok home. The [implementation design](evidence/tool-event-coverage-20261015/isolated-pruning-profile-design.md) identifies the actual worker-to-router deployment path and the required configuration binding, comparison, and verification checks. The separate `grok_cli_runner` config writer is outside this route. A deployed configuration hash establishes a declaration; effective settings still need native evidence. Matched task correctness and complete native usage must accompany any savings claim. Initial-prompt byte reductions and repeated event bytes cannot substitute for cumulative token accounting.
