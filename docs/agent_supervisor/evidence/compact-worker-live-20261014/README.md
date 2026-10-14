# Compact worker live observations

These are full-supervisor Terminal-Bench `largest-eigenval` observations using
Grok 4.7 through `llm_router`, the indexed numerical version-3 profile and
`mapped-span@2`. The runtime was built from accelerator commit
`3f08cc43cbd96c9f8a5114091f65566a5962dbbc`. The runtime archive hash is recorded in
the input releases; the archive itself is not published here.

The [original pair](readouts/original-pair.json) changes only source view
`identity@1` versus `python-comments@1` and the required output directory.
Both use legacy semantic metadata and reply contracts. A separately released
third trial changes the comments arm to `common-bindings-instruction@1` metadata
selection. That third trial is outside the original pair, uses a separate
[execution lock](inputs/metadata-execution-lock.json), and was selected after
observing the first trial. These are sequential single observations, not a
randomized experiment or a native Codex baseline.

## Original pair

Both runs passed the official verifier. Each admitted two goals and one task,
with one planning call and one coding call. Native final envelopes reported:

| Observation | Identity | Comments |
| --- | ---: | ---: |
| Planning reported tokens | 24,474 | 24,055 |
| Coding reported tokens | 1,286,290 | 1,597,606 |
| Reported category sum | 1,310,764 | 1,621,661 |
| Fresh input | 87,665 | 222,913 |
| Cached input | 1,185,536 | 1,356,416 |
| Cache writes | 0 | 0 |
| Output | 37,563 | 42,332 |
| Initial coding model prompt bytes | 78,597 | 78,499 |
| Native coding turns | 23 | 26 |
| Successful mapped preparations / confirmations | 1 / 1 | 0 / 0 |
| Broker requests | 7 | 6 |
| Delivered source bytes, including repeated slices | 4,461 | 4,102 |
| Owner response JSON bytes | 19,220 | 15,399 |
| Launcher elapsed seconds | 1,163.80 | 1,026.37 |

**Complete token usage is unknown.** Neither Grok final envelope explicitly
establishes usage completeness. The table sums the reported disjoint fresh,
cached, cache-write and output categories; cached input is counted once. The
frozen collector leaves complete totals and complete-token differences null.
The comments observation reports 310,897 more tokens; it does not demonstrate a
token-saving advantage.

The comments run has five successful broker responses and one retrieval
refusal. Its refusal operation is not retained, so zero successful mappings
does not establish that no mapping was attempted. The native stream records
22 unknown-name tool calls for identity. Comments records 26 tool calls: three
`grep`, three `read_file`, one `search_replace` and nineteen unknown-name calls.
Those names do not establish source paths, original-source expansion or use of
the mapped helper through shell commands. See the
[native shape supplement](readouts/native-shape-original-pair.json).

Owner response byte counts cover canonical JSON response bodies. They exclude
CLI formatting, commands, frame prefixes and conversation history. Native
turn counts, token categories, broker requests and these byte measurements are
different scopes and must not be added together. Supervisor phase timings
also overlap enclosing execution timings.

The [identity diagnosis](readouts/identity-diagnostic.json) and
[control audit](readouts/identity-control-diagnosis.json) distinguish observations
from unverified causes. The large initial coding prompt contains indexed
metadata and formalization evidence; no per-turn numeric usage history was
retained to attribute later token growth to a particular tool or source read.

## Separate metadata treatment

The [third readout](readouts/metadata-treatment.json) records another official
pass with two goals, one task, one planning call and one coding call. The
requested `common-bindings-instruction@1` mode was selected without fallback.
Within that coding call, the recorded complete input changed from 78,553 to
77,585 bytes: 968 bytes smaller (1.23%). This comparison uses that call's own
native and selected inputs, not the different prompt from the preceding run.
The selected input hash and length match the actual router receipt.

| Third-trial observation | Value |
| --- | ---: |
| Planning reported tokens | 24,609 |
| Coding reported tokens | 2,428,025 |
| Reported category sum | 2,452,634 |
| Fresh input | 110,079 |
| Cached input | 2,297,344 |
| Cache writes | 0 |
| Output | 45,211 |
| Native coding turns / tool-call events | 36 / 39 |
| Successful mapped preparations / confirmations | 3 / 3 |
| Broker requests: successful / refused | 14 / 1 |
| Delivered source bytes, including repeated slices | 4,543 |
| Owner response JSON bytes | 33,485 |
| Launcher elapsed seconds | 1,275.25 |

Usage completeness remains unknown. Compared with the comments observation,
its native reported category sum is 830,973 higher. Three mapped edits occurred,
and the prompt projection reduced bytes, but these runs do not demonstrate a
complete-token benefit. Different tool histories and coding turns prevent a
causal interpretation. The refused query kind is again unavailable in the
retained historical audit.

The [release](inputs/metadata-release.json) and
[control audit](inputs/metadata-control-audit.json) bind the change. The reader
validates the closed recorded projection arithmetic and selected-input hash;
it does not reconstruct unavailable private prompt bodies. The final collection
verified all 327 frozen source pins before subsequent telemetry changes.

## Evidence and reporting limits

The initial ordinary-completion configuration was refused before deployment or
model generation. Its [preparation result](inputs/initial-prepare-refusal.json)
retains the zero-model-call observation. The launched Grok runs consistently
use the supported legacy reply contract.

The [reporting freeze](reporting/original-readout-freeze.json) supersedes the
historical freeze because the supervisor's unreceipted-attempt marker is a
mapping. The first reader recognized only literal `True`. The correction keeps
any present non-null, non-false marker from establishing complete totals.
The old reader, checks and review remain in `history/` as superseded evidence;
the [current review](reviews/original-readout.json) and 38 authored checks cover
the correction. This was a reporting-only fix, not a change to the live runtime.
The [third-reader review](reviews/metadata-reader.json) separately covers its
execution lock, control integrity and recorded projection joins.

The `readers/` scripts are exact source copies. Reproducing receipt collection
requires the original private metadata paths and frozen runtime dependencies;
this public bundle intentionally does not include full native receipts,
prompts, provider transcripts, verifier bodies, private databases, credentials
or the runtime archive. Public JSON readouts contain closed metadata projections
and source hashes. They can be audited without reconstructing private model
inputs. The [sealed manifest](manifest.json) binds every included byte after final collection.

[Offline qualification](../repeated-compact-edits-20261013/README.md) covers the
714 distinct passing tests and an authored cross-UID Docker probe. Its three
sequential mapped edits are separate from the model's actual helper use here.
No source-semantic proof, billing completeness or causal advantage is claimed.
