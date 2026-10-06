# Terminal Bench context reduction with planning retained

The objective is fewer total supervisor tokens while keeping LLM planning and
coding through `llm_router`. The databases should retain complete source,
metadata and proof records; the model should receive the facts needed for its
current decision. This change implements the first reversible transport
experiment from the [context improvement plan](../supervisor-context-token-economy-20261006/README.md).

## How the components can reduce tokens

| Component | Work outside the LLM | Compact model context |
| --- | --- | --- |
| DuckDB and Quack | Query source symbols, interfaces, dependencies, effects and task scope through the existing owner boundary | Relevant facts and checked local handles |
| DuckLake | Identify changed retained records with cursor and provenance bindings | Changes since the previous qualified context; current source still checked separately |
| Formal systems | Discharge supported obligations, reject incompatible approaches, retain counterexamples and reusable certificates | Claim, status, assumptions, source revision, useful counterexample and certificate handle |
| Semantic capsules | Bind selected facts and unresolved obligations to source and dependency roots | A decision view with visible omissions and available expansion |
| Reversible minification | Share repeated metadata and retain the full identifier dictionary in the controller | Short typed references with literal source, interfaces and task requirements |

For `largest-eigenval`, useful model context includes the general nonsymmetric
matrix domain, possible complex eigenpairs, dominant modulus, nonzero vector,
residual criterion, allowed edit and public validation commands. A symmetric-only
approach needs a justified precondition. Complete floating-point correctness
and timing remain open obligations unless separately established. Compact
formal evidence should reduce repeated reasoning without overstating a proof.

The table describes the intended combined pipeline. Existing catalogs,
selection and decision/delta compilers provide building blocks. Bounded model
lookups, compact proof/status cards and terminal-session deltas still need
integration. The current native coding CLI owns its later conversation and
tool history; an initial-input compressor does not control those turns.

## Implemented experiment

`supervisor-semantic-router-input@2` omits the full alias-to-CID dictionary from
the provider input. The supervisor retains its immutable native translation
table, exact source manifest, task/root/scope bindings and replacement paths.
The model receives short `$semantic_ref` objects with explicit bindings.

The existing encoder verifies the native producer records before dispatch.
Native prompt reconstruction is exact. Structured candidate replies expand
references through the retained dictionary and still pass the existing native
grammar and freshness checks. Source strings, comments, paths, executable names,
requirements, scope and authority retain their native values. The new version
rejects unknown versions, marker collisions, foreign bindings, stale source and
tampered references. Ordinary prose remains literal.

The benchmark exposes an explicit `semantic_transport_schema` selection through
preparation, the Harbor configuration, container driver and coding worker. It is
bound into the complete configuration digest and retained receipts. An archive
missing the qualified implementation is rejected before model dispatch. The
historical/default transport remains `@1`, including its exact prompt, receipt
and table bytes. Historical input audits replay the recorded version and compare
the complete router/model input hashes; the two versions share a native table
CID, so checking that CID alone would be insufficient.

The transport selection applies to coding. For the requested pilot, use direct
LLM planning with no symbolic-planning requirement contract, or the separately
qualified version-1 provider coverage route. Explicitly assert one planning and
one coding session in both arms. Other existing planning modes retain their
own behavior; selecting this coding transport alone does not establish the
number of LLM sessions.

## Evidence and next comparison

The [public-task offline measurement](public-task-transport-measurement.json)
uses cold copies of the public instruction, `eigen.py` and public `eval.py`.
It runs the native semantic producer and context compiler, then compares both
complete coding-input renderings with identical public-instruction and workspace
additions. It verifies exact reconstruction and reports bytes plus a declared
local `o200k_base` input-token estimate. The fixture uses `eigen.py` as its edit
scope; `eval.py` is validation evidence. No Doctor status is invented.

The measured complete input shrinks from **69,785 to 67,546 bytes** and from
**24,019 to 22,800 estimated `o200k_base` tokens**: 2,239 bytes and 1,219 estimated
input tokens (5.08%). Both inputs use the same native envelope, public append
and workspace advisory. These figures apply to this fixture only.

This is a synthetic public-task envelope, not a reconstruction of a historical
provider input or a fresh official trial. Its tokenizer is independently pinned;
the provider's actual encoding and internal turn usage remain unverified.
Prompt shrinkage is therefore separate from total task-token savings.

The causal comparison needs one newly pinned runtime archive used by both full
arms: default `@1` versus explicit `@2`, with planning retained. Match public
inputs, model, CLI, reasoning, resource profile, timeout and retry policy. The
historical 291,432-token successful two-session run remains context, rather than
a matched control for the current runtime. Report official reward, full native
input/output usage, cached input already included in input, later turns, lookup
costs, retries and wall time. A shorter initial prompt can lose its benefit if
the model needs more reads or repair turns.

After this representation experiment, the next treatments are shared IntentIR
source/requirement tables, selected decision capsules and compact formal status,
then checked catalog lookups and context deltas. Compare those separately so
their token and reliability effects can be attributed.

The [qualification](qualification.json), [independent integration review](integration-review.json)
and [artifact manifest](artifact-manifest.json) bind the offline checks and
source. No fresh model/provider calls, official trials, training or model-weight
updates were made for this qualification.
