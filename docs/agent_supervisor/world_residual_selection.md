# Measure matched residual requests before dispatch

The strict repository world-packet route can explicitly compare a full-evidence
request with a focused request. Both carry the same sealed residual diagnostics,
the full task and every required reference. A focused input is selected only
when its complete rendered prompt uses strictly fewer bytes and local tokens.
This option leaves existing defaults and provider routing in place.

The earlier [residual request qualification](world_residual_requests.md)
compared original inputs with newly residual-bearing inputs. Those measurements
include the cost of introducing diagnostics. An original input without those
diagnostics cannot serve as this selector's fallback: it would lose the failure
that prompted the repair. The new full alternative retains the baseline's exact
selected evidence population and adds the identical residual context.

## Prepare two current alternatives

Use the current strict baseline and the existing explicit focus helper:

```python
focused = prepare_repository_world_residual_request(
    baseline=current_strict_packet,
    residual_packet=sealed_residual,
    query="arithmetic bound",
    unresolved_question="Which current definition causes this counterexample?",
    limit=4,
)
alternatives = prepare_repository_world_residual_alternatives(
    baseline=current_strict_packet,
    focused=focused,
)
```

Preparation reconstructs the original compilation from its retained receipt and
the same issued live capsule, then verifies it through the original compiler.
It does not perform a new optional-selection compilation or consume another
retrieval-parent slot. Existing native freshness checks still repeat their
captured source queries. Both alternatives retain the original task requirements
and the same detached residual record. If the full alternative cannot fit its
admitted packet bound, preparation refuses.

The returned object supplies `.full_packet`, `.focused_packet` and a detached
`.receipt()`. It retains original live packet, compiler and estimator custody;
it cannot be restored from a historical record. Optional evidence selection
remains advisory. Matching residuals and preserved required facts do not prove
that omitted optional evidence is irrelevant or that model quality is unchanged.

## Select through the existing journal and router

Create the existing private attempt journal before entering the runner, then
pass the exact full packet and its issued alternatives together:

```python
output, invocation, retention = run_repository_world_packet_attempt(
    journal=issued_candidate_journal,
    runner_options=dict(
        prompt=exact_original_task,
        provider="codex_cli", model=pinned_model,
        timeout=180, max_output_tokens=4096,
        world_task_packet=alternatives.full_packet,
        world_task_residual_alternatives=alternatives,
        prompt_token_counter=owner_counter,
    ),
)
```

This is an explicit programmatic route. It requires the existing durable
journal and cannot combine with packet presentation or the older original-only
artifact-retention option. Source-tool guidance and the ordinary coding reply
contract remain supported. There is no additional Harbor or command-line flag.

The router renders both alternatives with the same workspace contract, current
tool guidance and reply instructions. It counts each complete string and the
separate native output schema using the existing `PromptTokenCounter` contract.
Provider/model labels, tokenizer identity, revision, special-token policy and
the counting callback are captured before counting and checked afterward.

A tie, larger focused input, absent counter or ordinary counting failure keeps
the full residual-bearing packet. Failed counts discard all partial counts;
there is no byte-estimate substitute. A requested hard token cap requires
working counts and must accommodate the selected complete input and schema.
Counter identity changes refuse dispatch. A supplied local tokenizer's
descriptor, asset and runtime are checked again after provider initialization,
including when counting failed and selected the full alternative.

All caller callbacks precede the last native source and intent checks. The
existing journal, selected packet and provider boundary remain in use. The
invocation's `world_task_packet` describes the actual selected packet;
`world_task_residual_selection` records both complete-input hashes and counts,
the decision and its dispatch check. A provider failure still retains the
selected input through the existing attempt wrapper.

These local counts exclude provider framing, hidden instructions, tool schemas,
internal conversation history and later turns. An owner-declared tokenizer
binding does not establish remote tokenizer equivalence. Compare complete
native sessions and independently verified outputs through the existing
[session accounting path](world_benchmark_session_accounting.md) before
changing defaults.

## Retain and replay the matched decision

Freeze both live packet artifacts before a worker can edit source. After the
attempt is durably captured, join their independently retained pins with the
complete original router receipt and its pin using
`freeze_repository_world_residual_selection`. Retain the resulting artifact's
SHA-256 independently. `replay_repository_world_residual_selection` reconstructs
both complete inputs and the selected ordinary invocation.

The replay keeps the original receipt intact. Its adapter to ordinary invocation
replay uses a checked internal projection; that derived projection is explicitly
marked as lacking independent custody. Ordinary invocation replay refuses a
selection envelope rather than silently discarding its decision record.

Recorded count arithmetic and decisions can be checked without a tokenizer.
Recomputing counts requires an explicitly supplied trusted counter. No tokenizer
path or callable is loaded from a retained receipt. Without that counter,
count verification remains unknown. Historical replay restores neither live
packet issuance nor source freshness, native signatures, provider ingestion,
billing, formal proof or completion authority.

## Finite qualification on October 15

Qualification passes 493 distinct targeted test cases, with no skipped cases.
This includes existing dispatch, replay, source-tool, prompt-selection and
attempt-retention compatibility, plus the new pair, counter, dispatch and
historical checks. The two malformed foreign-input fixtures were corrected and
rerun individually; their earlier failures remain retained. All 15 qualification
leases were released, and the shared native resource policy stayed unchanged.
Earlier internal budget and string-enum serialization refusals are also retained;
native contract encoding is now reused while incoming literal guards stay strict.

Three authored repair fixtures use genuine native source indexing, the real
repository MCP child, guarded symbol replacement, the current router and a
durable attempt journal. Provider execution and the CLI configuration
observation are scripted. Every selected input keeps the identical sealed
counterexample, full task, exact source and acceptance requirements. Each
repair passes all 91 checks over the declared integer domain `[-3, 3]`, changes
only the requested function, and preserves helper and support bytes.

| Fixture | Full / focused complete bytes | Full / focused local proxy tokens used for selection | Selected |
| --- | --- | --- | --- |
| Broader optional evidence | 26,870 / 16,347 | 8,855 / 4,945 | Focused |
| Focused request grows | 16,343 / 26,911 | 4,917 / 8,789 | Full |
| Counter fails | 26,879 / 16,356 | Unknown / unknown | Full |

The proxy is the cached `o200k_base` encoding. Failed counting records no
partial counts, even though a separate offline inspection can count the
retained strings. The measured cases call the counter twice for their two
complete legacy-reply inputs; the failed case stops at its first failed count.
The replay verifies exact selected input and reconstructs both alternatives.
Explicit trusted-counter replay verifies the measured local counts and leaves
the failed case's verification unknown. All old live packets and the issued
pair refuse after the guarded edit.

These inputs use an explicitly unavailable world projection. No world authority
is manufactured. No live benchmark model calls, Harbor trials, training or
downloads were launched. These finite results establish neither live model
quality, official benchmark reward, remote tokenizer equivalence nor complete
workflow token savings. Defaults remain unchanged.

Private scripts, retained source snapshots, failed qualifications, native lease
records and the finite evidence are stored at
`/home/barberb/lift_coding/artifacts/world-residual-selection-20261015-01`.
The finite report SHA-256 is
`ebe018b8d13221ba2ed0c80c3f0ecc5fc8a5fceac78a3075d4bafb8e7584808f`.
The complete qualification manifest SHA-256 is
`1d38f4ba0aded47a42547dd59ebdd6df39476884bd89fc76baf6d84ed26722eb`.
