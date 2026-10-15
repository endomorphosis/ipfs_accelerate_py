# First live capsule-budget comparison

The fresh legacy full-supervisor baseline passed `largest-eigenval`. The
`one-capsule@1` attempt stopped during Grok credential readiness, before model
planning or coding. **This pair does not establish a capsule-budget advantage.**
The [readout](readout.json) retains both attempts, including the setup failure.

## Frozen comparison

The runtime was built from `eff8ef3ecc`, after the
[capsule-budget implementation qualification](../capsule-budget-20261014/README.md).
Both preparations used the same archive, task inputs, Source384 configuration,
Grok 4.7 CLI 1.0.46 route through `llm_router`, high reasoning, model planning,
identity source view, `mapped-span@2` edits, legacy metadata and replies,
lexical capsule selection, five CPUs and 20 GiB. The only complete configuration
differences were capsule budget and output directory. Each arm registered one
attempt; no benchmark retry or provider fallback was run.

The [pair lock](pair-lock.json), [control audit](control-audit.json) and
[runtime verification](runtime-verification.json) were completed before launch.
Archive SHA256 is
`13497334d574d8b916df55c3a9b649f90dd8bc14ba56056bdd7c077a3333a17a`.
Both launch records confirm unchanged configuration, preparation and source.
Generic exact-control comparison deliberately differs for the capsule policy;
the treatment was not relabeled as an exact-control match.

## Observed results

| Full-supervisor arm | Official reward | Planning reported tokens | Coding reported tokens | Coding turns | Coding prompt bytes | Launch seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy capsule budget | 1.0 | 24,058 | 1,750,761 | 26 | 78,313 | 2,027.01 |
| `one-capsule@1` | Unknown: setup aborted | Unavailable | Unavailable | Unavailable | Not dispatched | 535.85 |

The token columns are separate native final-envelope observations. Both baseline
envelopes omit an explicit usage-completeness flag. Complete run totals, token
differences and billing totals therefore remain unknown in the readout. No
missing counter is replaced with zero, and no stream counters are summed.

Baseline planning reported 15,817 fresh input, 1,536 cache-read, zero cache-write
and 6,705 output tokens. Coding reported 187,681 fresh input, 1,510,912 cache-read,
zero cache-write and 52,168 output tokens. These categories already comprise
their respective reported totals; cache reads must not be added again.

Planning used one native turn and a 34,233-byte router task input. Its provider
call took 505.96 seconds; the enclosing planning phase took 515.81 seconds.
Coding used 26 native turns and took 1,078.76 seconds in the router invocation.
There were only two observed supervisor router invocations. Setup took 310.29
seconds; agent execution took 1,692.70 seconds. Named phases overlap enclosing
timings and are not additional independent costs. This was a sequential,
single-task experiment on a shared host, not a randomized performance estimate.

## What explains the large reported count

Cached input represents 86.3% of the coding envelope's reported total. The
bounded `native-grok-stream-progress@2` record retained all 26 usage events.
Its cache-read observations rose from 1,536 at the first event to 94,464 at the
last. Event 25 dropped from the preceding 92,032 cached-input observation to
1,536, while fresh input reached 92,997. That pattern is consistent with a cache
miss or renewed processing of existing context; the cause and counter semantics
are unverified. Event ordinals are not authenticated turn identifiers. These
observations are not summed or converted into per-turn spending.

The initial context selected four of eight capsules under the existing 32,768
byte limit. Its 31,939-byte semantic artifact hash joined the actual coding
translation hash, and the final input audit verified the delivered coding input.
The semantic transport transformed a 79,311-byte native object into a
71,868-byte provider object; other prompt material brought the complete router
task input to 78,313 bytes. This input excludes provider harness/system context
and subsequent native conversation history.

Capsule members occupied 12,008 bytes of the translated object; raw-source
members occupied 2,588 bytes. Retained native context occupied 39,711 bytes,
including 18,604 bytes of source-formalization evidence and 13,037 bytes of other
evidence rows. These are nested partitions, not additive independent totals.
Identifier translation and fewer capsules address only part of this payload.

The source broker accepted and delivered six requests: discovery, three source
reads, mapped edit preparation and confirmation. It delivered 16,160 response
JSON bytes containing 3,530 source bytes. No candidate-context retrieval was
observed. Native tool diagnostics retain one `read_file` and 25 `unknown` names;
they cannot identify all ordinary shell reads, edits or tool-result contents.
No native automatic compaction was observed.

Doctor retained a residual and selected model coding. Source384 and numerical
context hashes were verified against the dispatched evidence, but this audit
did not replay neural inference. No source obligations were reported discharged;
neither index presence nor formalization syntax establishes program correctness.
The observed official verifier pass is separate evidence.

## Authentication blocker

The [treatment readiness receipt](one-capsule-readiness.json) records
`authentication_refused`, native exit zero, 144 stdout bytes, no catalog models
and zero text-generation calls. Exit zero did not count as authentication
success. The [fresh metadata-only check](post-failure-readiness.json) reproduced
the same refusal using another disposable copy of the current host credential.
It preserved the host auth file and removed its copied credential and container.
No login, credential writeback, generation or benchmark retry followed.

The classifier recognizes a bounded authentication diagnostic; it does not
prove expiry, refresh-token rotation, revocation or a transport root cause.
Each trial copies the host credential into a writable worker credential store.
Native refreshes there have no writeback path. Stale copied credentials are
therefore a plausible lifecycle issue, not an established diagnosis. A trusted
local sign-in is needed before further live qualification. Worker-modified
credentials must not be blindly imported into the host.

## Follow-up work

1. Restore Grok authentication and register a fresh treatment attempt while
   retaining this failure. Recheck metadata readiness immediately before costly
   setup. Keep the original paired attempt and any continuation distinct.
2. Qualify a trusted, serialized credential-refresh lifecycle and more specific
   closed readiness diagnostics; keep raw credentials and native messages out
   of reports.
3. Test lazy delivery of eligible nonrequired evidence and explicit
   generation-bound context-delta queries. Preserve full expansion and detect
   stale source; measure added retrieval and task outcomes as well as bytes.
4. Add bounded deterministic checking of supported post-edit obligations.
   Current candidate formalization emits facts and pending obligations without
   solver or Lake execution. Unsupported semantics must remain unresolved.
5. Measure planner reasoning separately. The requested 4,096 output-token cap
   is not enforced by the Grok adapter. Local CLI help exposes turn and effort
   controls, but no token-budget flag in the inspected help. Lower effort is an
   experiment, not a hard token cap.

The existing numerical Doctor performance gate remains intentional; a
numerically checked candidate does not establish benchmark performance.

## Retained evidence and validation

The [manifest](manifest.json) binds copied metadata and campaign scripts to
their original retained artifact paths. The [reporter tests](reporting-tests.json)
passed 24 cases and the [launch tests](launch-tests.json) passed 12 before launch.
These qualify reporting and launch logic, separately from the prior 440-test
implementation qualification. No production source was changed for this pair.

The [resource audit](resource-policy-audit.json) distinguishes the unrelated
general scheduler disk threshold from this admitted benchmark path. The
campaign required 32 GiB available before each launch; production admission
checks and limits stayed unchanged. Retained caches were not deleted.

Raw provider transcripts, credentials or their digests, owner databases, signing
keys and the large runtime archive are excluded from this evidence directory.
Some retained bindings use absolute local paths. No native Codex or no-index
baseline was rerun, and older live scores are not substitutes for this control.
