The supervisor now measures bounded Grok tool-event sizes and repetition through the shared router. A fresh public-input `largest-eigenval` diagnostic retained 48 events from eight calls, but its coverage limits prevent attributing the observed token subtotal to repeated model input. This experiment does not establish token savings or a Terminal Bench score.

The measured runtime is commit `24de5a01c43aede768426cca794e36f68c5d3fe4`. The [closed arithmetic analysis](evidence/tool-event-census-20261015/closed-arithmetic-analysis-01.json), [original closed diagnostic](evidence/tool-event-census-20261015/live/actual-public-diagnostic-01.json), and [evidence manifest](evidence/tool-event-census-20261015/manifest.json) retain the observations without prompts, tool arguments, outputs, raw call IDs, or private owner artifacts.

Ordinary Grok coding in `router_implementation_runner` enables `grok_tool_event_census=True` through `llm_router`. The shared provider otherwise remains opt-in. Progress `native-grok-stream-progress@4` adds a validated `native-grok-tool-events@1` object; legacy progress schemas remain accepted. Planning and structured-output routes are unchanged. Periodic callbacks still use count-only progress @1; detailed rows appear only in the final invocation receipt. Native final-envelope usage remains the token-accounting source.

The observer keeps the first 64 events with fixed tool/status categories, hashed call identities, exact frame bytes excluding LF, and separate canonical JSON measurements for `rawInput`, `rawOutput`, and `content`. Repetition is derived from retained digests and call identities. The payload channels can overlap and must not be added as unique output. Validators recompute counts, repetition, coverage, and parent stream joins. Overflow and unsupported structures remain explicit; none of these observations grants proof, execution, or completion authority.

The fresh Docker run used the pinned Grok 1.0.46 CLI and Grok 4.7 through the full supervisor, Source384 preparation, declared-contract planning, original semantic transport, and legacy working-set/metadata/reply profiles. IntentIR inference was disabled, matching the preceding diagnostic controls. There was one coding invocation, an eight-turn cap, no Doctor continuation, retry, or provider fallback, and no official verifier. The 5 CPU/20 GiB resource profile and 32 GiB disk admission floor were unchanged. This is not a matched ablation.

| Observation | Measured value |
|---|---:|
| Tool events retained / omitted | 48 / 0 |
| Distinct call hashes / lifecycle updates | 8 / 40 |
| Exact retained tool-frame bytes | 494,617 |
| Measured rawInput canonical bytes | 16,780 |
| Measured rawOutput canonical bytes | 169,546 |
| Measured content canonical bytes | 78,685 |
| Repeated rawOutput bytes | 28 |
| Repeated content bytes | 28,565 |

Of the repeated content bytes, 28,159 were same-call repetitions and 406 were cross-call repetitions. The repeated rawOutput bytes were seven repeated null values. Content repetition was 36.3% of measured content bytes. These are event repetitions: updates can be replacements, deltas, or duplicate notifications. The CLI's request bodies and history resends were not observed.

| Native usage category | Observed incomplete subtotal |
|---|---:|
| Uncached input | 92,702 |
| Cached input | 232,704 |
| Output | 15,700 |
| Total | 341,106 |

Retained native usage completeness was false; complete-token and verified-cost fields remain unknown. These counters are not a new complete token score. The CLI reached `max_turns_reached`, exited 1, and returned no usable final response. Supervisor task completion was false. The diagnostic wrapper exited 0 because measurement collection and cleanup succeeded; that is distinct from coding success.

The lifecycle took 474.295 seconds; the outer execute step took 475.675 seconds. Source384 preparation retained a checkpoint-bound inference record taking 10.995 seconds. That preparation receipt does not independently establish worker consumption or formal correctness. All 27,555 archive members were verified before launch. Runtime/dependency/source bindings remained unchanged through execution; worker cleanup returned 0 and exact container absence was observed twice, then independently rechecked twice. The source freeze was released after the [final check](evidence/tool-event-census-20261015/final-freeze-check-01.json), before this documentation commit.

The [coverage analysis](evidence/tool-event-census-20261015/closed-coverage-analysis-01.json) identifies the remaining measurement gaps:

- All eight call events contained a `toolName` field classified as unknown. The closed receipt does not establish whether those values were valid strings outside the allowlist. All 48 events had zero unknown top-level fields. Public pinned documentation uses both `run_terminal_cmd` and `run_terminal_command` in different interfaces; this does not establish the actual names in this run. No raw names were inspected or guessed.
- Seven rawOutput objects exceeded a payload bound and are absent from the measured payload total. Their frames were only 29,885–50,211 bytes, below the 256 KiB frame bound. The current receipt does not distinguish node, depth, integer, or payload-byte limits, so increasing a limit without further evidence would be premature.
- Eight update events had unknown status; the receipt cannot distinguish an omitted status from an unrecognized supplied value. A valid final trailer also conservatively marks post-final event coverage incomplete. There were no malformed/oversized frames, row omissions, timeout, or parser taint.
- Provider request bytes and history-resend bytes remain null. Pinned CLI documentation describes old-tool-result pruning, but effective pruning settings and the position of stdout observations relative to pruning were not independently verified. Its final prompt-usage scope excludes compaction and other side-model calls.

The next telemetry revision should add fixed bound-reason and status-presence fields with backward-compatible validation, audit documented tool aliases, and distinguish accepted terminal trailers from post-terminal tool events. An explicit paired pruning profile can then test retained tool-result context, with task correctness and native usage measured separately. A supported request-boundary observation is still needed to establish resend amplification; the [pinned documentation audit](evidence/tool-event-census-20261015/request-boundary-audit.json) found no such hook in the reviewed interface. Initial-prompt byte reductions alone do not explain this run's token subtotal.

Validation passed **595 cases**: 260 codec/parser/usage tests, 300 router/supervisor/comparison/archive tests, and 35 authored subprocess/launcher/export tests. The [qualification](evidence/tool-event-census-20261015/qualification-final01.json) binds the exact source and dependency versions. Authored cases cover separate payload channels, same-call versus cross-call repetition, Unicode, missing/null/empty values, structural limits, tampering, overflow, legacy schemas, timeout cleanup, and bounded callbacks. One initial external fixture incorrectly expected complete tokens from incomplete native categories; its assertions were corrected without weakening production accounting. No paid provider calls occurred during these tests.
