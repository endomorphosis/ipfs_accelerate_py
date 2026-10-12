# Observe metadata displacement in a prepared coding input

The [representation displacement observer](../../ipfs_accelerate_py/agent_supervisor/runtime/representation_displacement.py)
records how the existing metadata or instruction view changes one prepared
coding input. It joins the original transport, fixed projection, exact inverse,
literal wrappers, existing selection receipt and selected complete input. Its
closed report retains hashes and bounded counts instead of source text,
instructions, private paths or provider bodies.

This is a prepared-input observation. It does not establish that a provider
received the input, that future reads were avoided or that the complete session
used fewer tokens. Required source facts remain inline. It changes no encoder,
decoder, embedding, checkpoint, database owner, retrieval policy or proof status.

## Existing representation and selection

The upstream owner still admits the task, current semantic source and original
`supervisor-semantic-router-input@1` transport. The existing metadata projector
factors only its allowlisted repeated bindings; shared values remain inline.
The optional instruction projector references an independently admitted
verbatim block already present once in the literal suffix. Both existing
codecs retain exact original-transport restoration.

The observer checks the exact typed view, restores the original transport and
reconstructs the fixed projection. The native and candidate complete inputs
must contain their respective transports exactly once, with byte-identical
prefix and suffix around the replacement. Occurrence and span offsets are
measured in UTF-8 bytes. The selected input must match the existing selection
receipt and selected mode; a caller-nominated digest cannot substitute for the
actual prepared string.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import (
    observe_representation_displacement,
)

# Reuse the view and selection already produced by the existing owner/selector.
observation = observe_representation_displacement(
    view=existing_typed_metadata_view,
    selection=existing_typed_selection,
    native_complete_prompt=exact_native_complete_prompt,
    candidate_complete_prompt=exact_candidate_complete_prompt,
    provider=selected_provider,
    model=selected_model,
)
```

For the instruction treatment, additionally supply its actual
`instruction_block` and `instruction_receipt`. When source-reader guidance is
present, supply that exact existing string as `source_guidance`; it is checked
as additive wrapper content. The observer receives no counter callback and
does not choose a different representation.

The observation records one transport replacement when the candidate is
selected. A fallback retains the original complete input and records no
selected replacement. The receipt does not claim that metadata bytes, an
instruction reference or a latent vector replaces required source bodies or
formal evidence.

## Counter and receipt limits

The existing `supervisor-semantic-metadata-selection@1` uses the named
`utf8-bytes-ceil-div4@1` proxy with complete rendered byte counts. Those units
are not provider tokens.

The existing selection `@2` can record an upstream caller-supplied token counter
identity, qualification and counts. The observer checks their closed shape,
hashes, selected/fallback arithmetic and joins to the retained strings. It does
not construct a trusted tokenizer, call the counter again or authenticate that
the declared counts came from the claimed provider/model. Provider-added
messages, tools, output schema and total session usage are outside that count.
An existing false token-identity or provider-usage verification flag remains
false during historical replay.

The closed `supervisor-representation-displacement@1` receipt is limited to
16,384 canonical JSON bytes. It records `observation_stage=router_prepared_input`,
`treatment=rendered_replacement` or `native_fallback`, UTF-8 transport spans,
unchanged wrapper identities, selected-input digests and net complete-input byte
reduction. Provider/model names and task/scope identities are hashed. A receipt
hash retained by the caller identifies those recorded bytes; it is not a
signature, authenticated delivery receipt, freshness proof or checkpoint
identity. Historical audit reuses the recorded selection rather than replacing
a missing counter with a newly trusted implementation.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import (
    validate_representation_displacement,
)

checked_observation = validate_representation_displacement(
    retained_observation,
    selected_input_sha256=retained_selected_input_sha256,
    selected_input_bytes=retained_selected_input_bytes,
)
```

This validator checks closed structure, recorded arithmetic and the selected
input join. It does not reconstruct absent original prompts or authenticate a
recorded counter. Independent historical audit rebuilds the fixed projection
and observation from retained inputs. After its separate instruction
reconstruction checks, the audit preserves the original signed inclusion
receipt's hash lineage without upgrading historical freshness or replay flags
to current authority. `counter_measurement_replayed_here` and
`counter_identity_authenticated` remain false. The byte-proxy `@1` selection can
be independently repeated without invoking a token counter.

The receipt records `all_raw_sources_unchanged_inline` for ordinary metadata
and fallback. A selected instruction view instead records the duplicated
instruction source reference together with the independently admitted verbatim
block retained in the suffix. Its source preservation profile does not assert
general required-fact coverage. `proposed_avoided_reads` is empty, while actual
avoided-read, IR provenance, required-fact coverage and all authority flags
remain false.

## Runner opt-in and additive guidance

The ordinary coding runner prepares this observation only when its existing
metadata or instruction treatment is explicitly selected. Default legacy
inputs and receipts retain their existing shape. The metadata census keeps its
existing observation and pre-provider stop point; this addition does not
silently change historical census output.

When optional semantic source-view guidance accompanies a metadata treatment,
its literal wrapper bytes remain present in the baseline and candidate
comparison. The `source_guidance` field records that specific overhead as
additive; it does not credit the guidance with displaced source or avoided
future reads. Other unchanged wrapper content remains included in the complete
input identities and byte counts without being individually classified.
Source-only sessions without the metadata treatment do not yet have this
displacement observation.
The source worker's native claim/allocation/actual-child rules and semantic
reader admission remain separate prerequisites.

## Wider context planning and qualification

The [repository context plan](reversible_codebase_views_plan.md#require-evidence-of-displaced-context-before-claiming-compression)
still proposes general decisions across source views, capsules, proof references
and LegalIR, SecurityIR, CodebaseIR, IntentIR and UI/UX IR. This first metadata
observation does not join their schema/version/decoder-task/8D/384D/768D model
identities, validate required-fact coverage or permit source omission. Prior
assets and embeddings remain reused under their existing provenance.

Provider dispatch needs its own actual route observation. Avoided future reads
need matched complete runs, including planning, lookup, expansion, output and
repairs, with official task results. Smaller prepared text and recorded token
counts alone cannot establish that benefit.

The [corrected host qualification](evidence/representation-displacement-20261012/README.md)
passed all **553 distinct cases**: 51 new representation cases and 502 existing
selection, instruction, audit, census, source-wiring and archive controls, with
zero failures, errors or skips. All 40 recorded source bindings stayed fixed.
The retained first run includes the historical instruction-receipt hash failure
and its correction. Provider text and counters are authored fixtures; separate
historical audit subprocesses are exercised. This establishes no live provider
delivery, deployed worker qualification, official task reward, formal proof,
model training or complete-session token saving.
