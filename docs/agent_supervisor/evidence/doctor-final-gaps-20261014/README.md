# Doctor lifetime and Grok rejection diagnostics

These changes follow the [fresh repaired live run](../doctor-repaired-live-20261014/README.md).
That run verified the observed Doctor coding input, but still scored zero and
lost final coding usage. Its closed diagnostics narrowed a child failure to the
handoff timing predicate. No live benchmark has run with the changes in this
directory's qualification; the earlier live results remain unchanged.

## Doctor handoff lifetime

The ordinary 90-second handoff window could expire before a later Doctor worker
finished preparation. An authored test reproduced the failure through a real
second child/router/source-broker path by advancing the worker's guard clock
95 seconds while its original coding deadline and descriptor remained valid.
The failing frame matches the timing predicate reached in the live run; the
live record does not identify which timestamp conjunct failed there.

Typed Doctor-v2 admission now selects a maximum 1,800-second handoff interval.
Expiry is still the minimum of that interval, descriptor expiry, the source
broker's original deadline and the admitted original deadline. Ordinary handoffs
retain 90 seconds. Worker validation still requires the full matching Doctor
policy and source inventory, then rechecks expiry. This renews no original
deadline and adds no attempts or execution authority.

The six selected tests cover the late worker and source drift, original and
descriptor expiry, excessive intervals, future issuance, wrong/missing policy,
ordinary compatibility and schema spoofing, and historical Doctor retention and
replay. These use provider/UID doubles and an authored clock; they do not establish
fresh container isolation or successful live model continuation.

## Grok terminal rejection diagnostics

Progress schema `native-grok-stream-progress@3` records the first rejected
post-terminal event using a closed reason vocabulary, known field names, type
shapes, bounded counters and event timing. Arbitrary provider keys, field values,
text, identifiers and usage values are excluded. The validator checks the
rejection occurred after the first terminal and within the recorded stream.

Parser acceptance is unchanged. Historical schemas `@1` and `@2` still validate;
compact callbacks remain `@1`. The full diagnostic reaches CLI metadata, an
authored router subprocess, supervisor final receipts and benchmark comparison.
Per-event usage remains non-summable; missing native final usage remains unknown.

The original live `end` payload was not retained, so this change cannot diagnose
that historical rejection or recover its coding-token count. Before a future
trial, its new pure artifact reader must extract `_validate_terminal_rejection`
alongside the progress validator and retain the validated nested diagnostic.
The historical live reader and its pinned contracts are unchanged.

## Qualification and provenance

[Final qualification](qualification.json) passed **393 distinct cases**: 387
across seven parser/router/reporting modules and six selected cases across three
handoff modules. Both runs finished with source, dependency and installed-package
metadata pins unchanged; all 24 recorded source hashes match the qualified tree.
There were no live provider calls. The earlier 426-case repair qualification
belongs to a different source revision and is not added to this count.

[Independent review](independent-review.json) found no remaining blockers and
verified all 37 copied live-evidence files and their result joins. Its hashes
identify the files at review time; the final qualification paragraph in the
main plan was appended afterward. The package review covers those additions.

The original red lifetime result is retained: one expected failure at the old
90-second gate and one secondary owner-retirement fixture error. The fixture was
corrected before final qualification. `grok-rejection01` is a superseded subset;
`grok-rejection02` passed 387 tests but its overall source-pin check failed when
an unexercised handoff test changed during execution. Only the frozen
`grok-rejection03` and `handoff-green01` runs count as final qualification.
Raw JUnit bytes are preserved, including whitespace in the original failure.

The [copy provenance](copy-provenance.json) and manifest identify exact evidence
bytes. Test runners retain historical absolute paths and refer to a local safe
environment helper; they are experiment records, not portable launch commands.
No provider logs, credentials, private owner state or runtime archives are copied.

Public behavioral/performance validation remains the next functional gap:
structural completion can still accept a correct but slow implementation, and
Doctor continuation currently reacts to process failure. The signed public
acceptance-check design is in the [live-pair report](../doctor-live-pair-20261014/README.md).
Neither this qualification nor the earlier live results establishes token savings
or a completed Terminal Bench task.
