# Measure complete coding inputs before model allocation

The router can now measure the complete `capsule-references@1` input without
discovering, allocating or calling an LLM provider. This isolates a first-input cost
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

## Declared-plan Terminal Bench integration

The Harbor adapter and container owner now expose an explicit
`--working-set-census-only` route. Preparation freezes the full indexed arm,
`declared-contract@1` planning, `capsule-references@1`, enabled source views,
original transport and legacy metadata. Candidate continuation, mapped source
edits and metadata-census mixing are rejected. Harbor's official verifier is
explicitly disabled. The new `terminal-benchmark-declared-controls@9` record
keeps this measurement profile distinct from a coding benchmark.

The owner performs indexed preparation and validates the signed admission.
It rebuilds the declared goal/task graph and matches the retained planning
receipt, including zero planning-provider calls and initial context that was
never supplied to a planning router. The existing model-planned metadata census
retains its separate provenance requirements. Source384 is optional for this
census; a lexical index does not establish learned checkpoint inference.

The registered child uses the normal qualified worker namespace and source
boundary. Current source, native semantic context, public requirements and
applicable Doctor/formalization inputs remain bound to the observation. Grok's
catalog/readiness probe is skipped in this explicit mode; its configured turn
bound is preserved without allocating or calling a provider. The child measures
the complete original and reference inputs through the production renderer and
selector, retaining the original whenever references are larger.

After the source session stops, the owner replays the production encoding,
selection and closed census export, joins the observed requests and independent
deployment qualification, and completes cleanup. Only then can
`working_set_census_succeeded` become true. The collector reports measurement
completion separately; `complete_single_trial_receipt` and task completion stay
false. No coding generation or official verification runs, and there is no new
reward, model token score or efficacy comparison.

The integration passed **267 qualification tests**, including 102 Harbor and
control checks, with unchanged source/dependency pins and zero live provider
calls. Runtime commit: `92b540f004`.

## Fresh public Docker measurement

The [closed public-task evidence](evidence/working-set-owner-census-20261015/README.md)
now closes the owner-integration prerequisite. A fresh `largest-eigenval`
container completed full indexed preparation, pinned Source384 inference,
declared planning, admission, the ordinary worker/source boundary and owner
replay. Cleanup succeeded and the container was confirmed absent twice.

| Complete coding input | Bytes |
| --- | ---: |
| Original | 67,201 |
| Capsule-reference candidate, selected | 67,120 |
| Reduction | **81 (0.12%)** |

The candidate defers 24 provenance fields across two capsules. References and
guidance nearly offset that reduction. Remaining native context occupies
30,343 bytes; translated semantic context occupies 28,751. Together these
contain the large required intent/world, formalization, code-retrieval and
source groups that this projection preserves. This explains its limited effect
on this first input; it does not explain or measure the whole CLI history.

The source preflight returned 4,096 bytes, including 2,327 evidence bytes,
without supplying them to a provider. Complete capsule expansion was not tested
in this public run. The owner independently reproduced the full input counts,
hashes and closed receipt after the source session stopped. Zero planning or
coding generation calls ran, official verification was disabled, and there is
no new reward or token-efficiency claim.

Source384 inference took 9.744 seconds inside the 14.634-second initial-context
phase. The census took 29.572 seconds. The complete owned Docker lifecycle,
including deployment, took 491.373 seconds. These nested, single-run timings are
not a throughput comparison.

## Inline evidence and pack census

Runtime `78c1494bfb` adds an independent reversible candidate. It replaces
escaped JSON summaries with complete typed evidence objects carried inline in
the same input. Required constraints, reference metadata, unknown fields,
public instructions, Doctor evidence and source content remain present.

Inside an already admitted owner/source session, select:

```text
--semantic-working-set capsule-references@1
--semantic-working-set-census-only
--semantic-working-set-census-schema router-semantic-working-set-census@2
```

The declared-plan container owner's existing `--working-set-census-only` route
now selects `@2`. Direct router calls retain the original `@1` default and
output. Selecting `@2` outside census-only mode is rejected. These flags do not
create admission or a source session. Generation dispatch for the new inline
candidate remains disabled.

The codec reconstructs complete required intent/world, source-formalization
and code-retrieval artifacts from their native 6,144-byte UTF-8 chunks before
parsing them. It validates identities, order, digests and byte counts; individual
chunks can end inside a JSON string. Canonical finite floating point retrieval
scores survive the round trip. Valid opaque, noncanonical and nonobject bodies
stay literal; inconsistent known chunk contracts are rejected. The inverse
reconstructs the exact original chunks and original semantic transport.

Both candidate comparisons start from the same original complete input and
retain identical wrappers. They are independent, not composed; their reductions
cannot be added. Each selector retains the original when its candidate is not
strictly smaller. The pure codec performs no file, network or model operations.
Source freshness and admission remain owner checks; representation equality is
not proof of program behavior.

A fresh public `largest-eigenval` Docker run qualified both measurements:

| Complete coding input | UTF-8 bytes | Reduction from original |
| --- | ---: | ---: |
| Original | 67,201 | — |
| Capsule references | 67,120 | 81 (0.12%) |
| Inline evidence objects | **66,452** | **749 (1.11%)** |

Three evidence artifacts spanning four rows lost 1,373 bytes of JSON-string
overhead but gained 624 bytes of replacement structure and guidance. The net
749-byte reduction includes all coding wrappers. No constraints were omitted
by this projection. This measures the first input; CLI system prompts, later
tool responses and repeated history remain outside the measurement.

The new closed pack census reconciles a 6,266-byte pack: 3,298 bytes of reference
values, 1,918 of inner contract values, and other members and punctuation.
Its six references contain 983 bytes of metadata members. Fifteen typed
identifier slots refer to five native identities and occupy 763 serialized
value bytes. These are overlapping measurements, not removable-byte estimates;
different roles can legitimately refer to the same content.

Qualification passed **178 runtime tests** and **33 launcher/exporter tests**.
The fresh run used indexed preparation, pinned Source384 inference, retained
declared planning, admission, the native worker boundary, exact owner replay
and cleanup. Intent autoencoder inference was explicitly disabled in this
profile. Source384 inference took 9.881 seconds; the census took 30.692 seconds
within a 226.284-second owned Docker lifecycle. No planning or coding model
generation ran, the official verifier was disabled, and token usage and reward
remain unmeasured. The exact container was confirmed absent twice.

[Closed results, qualification and provenance](evidence/inline-evidence-census-20261015/README.md)
retain these observations and their limits. This is a bounded prompt
representation experiment, not reversible executable-codebase minification or
a native Codex versus supervisor efficacy comparison.

## Next measured optimization

Use the closed `@2` component census to choose the next target. Its nested
partitions overlap; compare sibling shares rather than adding parent and child
bytes. Keep each experiment source-bound and compare the complete rendered
input, including reference and retrieval guidance overhead.

1. Use the pack census to qualify exact factoring of duplicated token estimates
   and identical reference envelope defaults. Preserve every role, requirement,
   caveat and source obligation. The
   [source-based design](evidence/inline-evidence-census-20261015/pack-factoring-design.md)
   defines eligibility, inverse, fallback and tests; no saving is measured yet.
2. Keep new candidates independent until their composition has its own exact
   inverse and complete-input selection. Include instructions and dictionary
   overhead. A later token comparison must also count retrieval exposure,
   subsequent tool results and history, rather than extrapolating from bytes.
3. For larger reductions in intent/world, retrieval or formalization evidence,
   design an owner-selected disclosure contract before changing model input.
   Preserve required constraints and unknown fields, bind all pages to the
   admitted source generation, reject stale expansions and account for every
   retrieved response.

These are candidates for qualification, not established savings. Reversible
whole-codebase minification remains separate work: this route compresses a
selected prompt representation and does not transform the executable repository.
