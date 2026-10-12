# Candidate context refresh qualification — 2026-10-12

The isolated coding worker now rebuilds native datasets capsules and numerical
formal artifacts for a pinned working-source generation. See the
[operator guide](../../worker_candidate_context.md). Candidate artifacts have
zero proof authority, discharge no obligations, and publish no accepted
DuckLake index. This path runs no model, solver or Lake build.

The offline Docker qualification passed on its first attempt in **41.67 s**,
using actual owner UID 1000, worker UID 1001, generated launchers, the
`llm_router` implementation runner, authenticated socket and standalone client.
The model response was an authored fixture. The container had no network,
5 CPUs, 12 GiB and 128 PIDs; it was removed after qualification.

Captured `eigen.py` and `eval.py` produced eight native capsules and 45 pending
obligations. Three instruction/profile/smoke files remained in the freshness
scope and were excluded from program formalization. The numerical frontend
structurally translated `eigen.py`; `eval.py` retained partial coverage with two
of three functions partial/unsupported. Artifacts contain ProgramIR,
first-order/SMT and higher-order Lean projections, with no correctness claim.

| Artifact | Complete bytes | Worker retrieval qualified |
| --- | ---: | --- |
| Summary | 7,618 | Complete paginated artifact |
| Capsules | 48,309 | First page only |
| Formal reports | 476,030 | First page only |

The owner independently rebuilt full artifacts and verified each hash and
returned prefix. An observed edit/restore produced a new generation; two stale
context reads refused. Native unit tests separately verify changed signatures
and argmax/argmin formulas. This is transport qualification, not an LLM repair.

The collector matched **47 requests**, **26 successful pages**, **107,499 response
bytes**, **23,764 source bytes**, and **26,418 candidate-context bytes** across
five candidate pages. All complete wire bodies fit their requested bounds.
Repeated retrievals and refusals are included. Candidate bytes are excluded
from source bytes and included in total wire cost. Private source/formula
bodies and owner databases remain in local artifacts outside this publication.

The full formal artifact greatly exceeds source size and may exceed the shared
64-request/512-KiB session budget after envelope costs. This supports selective
retrieval, not a claim of token savings. There were zero real provider/model
calls, no official Terminal Bench verifier and no full live supervisor START.

The combined regression ran **731 cases: 730 passed, one failed, zero skipped**.
The failure was an outdated archive-inventory expectation missing the three new
modules. Updating that test fixture made all **104 archive capability tests
pass**. All 731 distinct cases therefore have a passing result across two runs;
this is not one green combined run. Production/runner sources did not change.
The exact test-only change is in [closeout-review.json](closeout-review.json).

Tests cover source/producer drift on warm caches, cross-file races, single-use
cursors, nonexecution, metadata-write suppression, unsupported source, large
semantic artifacts and support-only scopes. The blocked-build shutdown test
refuses late delivery and correctly records incomplete accounting while native
work remains active. Hard cancellation remains a documented limitation.

`focused-01` retains an earlier changing-source integration run that exposed the
parser-size mistake and outdated guidance assertion; it is not a passing
qualification. `final-01` and `capability-02` contain the closed regression.

Qualification used datasets `f47161b757190380e9898aff155067c1ad609217` and kit
`8ac00bb1974b1378e12091d546d8425af34049e0`. Concurrent parent publication moved
datasets to `e39fe996bd9e786a84f739fda9dde9a6c70546da`; all 23 direct datasets
candidate producers are byte-identical, as recorded in
[parent-datasets-producer-equivalence.json](parent-datasets-producer-equivalence.json).
That comparison is not a whole-parent regression.

Commands, source bindings, Docker controls, JUnit reports, byte observations and
integration review are retained here. `manifest.json` hashes the published files.
