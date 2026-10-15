# Measure capsule references before model allocation

The router can now measure the complete `capsule-references@1` input without
discovering, allocating or calling a model. This isolates a first-input cost
that was unchanged in the previous candidate-round projection experiment.
The shipped container worker accepts and forwards the same explicit selection.

The measurement uses the normal current semantic encoder, original context,
workspace wrapper, public instruction, Doctor residual, source guidance and
selected reply contract. It compares complete original and candidate inputs;
the existing selector keeps the original unless the candidate is strictly
smaller in UTF-8 bytes. Bytes do not establish provider token savings.

## Qualified measurements

The [closed evidence bundle](evidence/working-set-census-20261015/README.md)
records these authored fixtures, with no live provider calls:

| Fixture | Original bytes | Candidate bytes | Selected bytes |
| --- | ---: | ---: | ---: |
| Indexed public/Doctor flow | 23,202 | 23,249 | **23,202** |
| Router dispatch, legacy reply | 34,580 | 32,977 | **32,977** |
| Router dispatch, structured reply | 35,086 | 33,483 | **33,483** |
| Minimal indexed module | 12,527 | 13,719 | **12,527** |

The native indexed fixture performs real indexed preparation, declared-contract
planning, Doctor residual preparation, original-requirement capture and a
registered router child. Host UID admission and the comparison provider are
authored. The census matches the actual captured complete input. Two capsules
(5,799 bytes) expand through the authenticated owner endpoint and reconstruct
the exact native transport and original context. Public, Doctor and unknown
required fields remain intact. Expansion refuses stale source after an edit;
historical input replay succeeds after cleanup without authorizing dispatch.

On that fixture, references cost 47 extra bytes and are correctly rejected.
The other dispatch fixture saves 1,603 bytes. This variation is why local field
compression cannot establish complete-input savings. Retrieval responses and
repeated CLI history may also cost tokens during a real run.
These compare representations within the same prepared input, not independent
baseline and indexed benchmark runs.

All **235 final qualification tests passed**, with source and dependency pins
unchanged. An earlier 107-test regression batch also passed; it predates the
final exporter hardening and worker wiring. Final batches qualify those edits.
There is no new Terminal Bench score, model token measurement, training result
or checkpoint-inference claim in this experiment.

## Router and worker contract

In an already admitted owner/worker source session, select:

```text
--semantic-working-set capsule-references@1
--semantic-working-set-census-only
```

The owner must also provide the original semantic transport, legacy metadata
view, semantic repository, enabled source profile and exact source descriptor,
container boundary, and any applicable original formalization/public/Doctor
artifacts. These two flags alone do not create an admitted source session.
The API equivalent is `working_set_census_only=True` on `runner.run`.
Grok's explicit eight-turn configuration remains accepted. Structured ordinary
completion retains its existing Codex-only contract.

The explicit census rejects metadata-census mixing, candidate-round dispatch,
mapped source editing and source-lookup routes. A selected projection performs
the same authenticated first-page preflight as normal generation. A fallback
performs no capsule endpoint request. The probe is an owner-side check and adds
zero provider-input bytes; it does not verify complete capsule expansion.

The `router-semantic-working-set-census@1` export contains fixed labels, hashes,
byte counts, selection/fallback status and a closed preflight observation.
Dynamic identities are hashed; source bodies and prompts are not exported.
Token usage is unknown. Proof, execution, completion, settlement, publication
and required-fact omission authority remain false. Export validation rejects
extra component/preflight fields, malformed hashes, historical encodings and
selections, and Boolean/integer substitutions in component measurements.

## Locate the remaining context cost

The opt-in `supervisor-semantic-prompt-component-census@2` separately measures
known semantic reconstruction, refresh lineage, preparation bounds, pack,
worker projection, required raw paths, program paths and task-data projection
fields. It also separates intent/world and code-retrieval evidence from other
evidence. Unknown field names and values stay in closed aggregate groups.
The default `@1` census retains its previous output.

Partitions are nested views of the same bytes, so parent and child totals must
not be added together. The native fixture retained complete-input measurements
and reconstruction checks, not numeric component-group shares. The new schema
is structurally qualified; no live-task component savings are claimed.

## Remaining Terminal Bench integration

The router and installed worker are ready. The Harbor/container owner driver
does not yet expose this new census mode. Its existing metadata-census path
requires a model-generated plan, one planning invocation and Source384 controls.
Declared-contract runs use zero planning invocations; reusing that path by
loosening its checks would misstate provenance.

The next integration needs an explicit declared-plan census profile that
reuses native preparation, allocation, admission and current-source checks;
validates the retained declared graph; launches the registered child with this
flag; checks the closed result and source-session joins; and performs normal
cleanup. It must stop before coding generation and official verification.
Only then can a fresh public-task census guide another full-run comparison.

Capsule references currently defer an allowlist of provenance fields. Required
behavioral constraints, formalization, raw source and unknown fields stay
inline. Whole-codebase reversible minification and selective retrieval of the
larger required evidence groups remain separate work, with explicit freshness,
reconstruction and request-accounting requirements.
