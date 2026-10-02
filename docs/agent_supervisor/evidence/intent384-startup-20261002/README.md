# Intent384 startup and portable inference qualification — 2026-10-02

The full benchmark harness can now select the published experimental Intent
action checkpoint before goal declarations, replay its saved advice, and supply
a bounded candidate summary to the model planner. Runtime packaging transports
the exact checkpoint and pinned GTE-small assets and relocates their paths.
The shared datasets runtime also fixes a container failure caused by importing
a producer checker that required Git metadata.

The checkpoint remains `Publicus/intent-ir-autoencoder` at immutable revision
`d12dd68fcc54441ded09ed217526e02b7bdfd9e3`, SHA-256
`4f3fd17ea2d908fe36c57a444517f3cc0a983cab26f0f67f2e32371e25def3cd`.
No weights were changed or retrained. See the
[deployment identity](deployment-identity.json) for archive, image, embedding,
and selected code hashes. The datasets change is commit
`8cafb37b2` and leaves the frozen structured numerical reader unchanged.

## Executed checks

| Check | Result | Scope |
| --- | --- | --- |
| Selected supervisor suites | 214 passed, 1 skipped | Startup, archive transport, Harbor forwarding, native planning, effect contracts, task context, and checkpoint replay |
| Datasets runtime and portable source pins | 26 passed | Includes a fresh isolated interpreter without Git metadata and loaded-code/file drift rejection |
| Archive-only Docker preprocessing | 2 expected dispositions | Network disabled; one supported controlled instruction and the original Terminal-Bench prompt |
| Published Intent + fresh Security inference + task context + Lake | 3 expected dispositions | One satisfied, one refuted, one with no enabled inputs; 9 finite cases each |
| Native Doctor daemon regression | 2 successful runs after 1 startup timeout | Separate keyword/signature repair fixture, zero provider calls, no remaining processes |

The [test summary](test-summary.json), [supervisor log](supervisor-tests.log),
and [datasets log](datasets-tests.log) retain the test evidence. A fresh
supervisor seal catalog forced test execution. The skipped test requires an
explicit older paired-text checkpoint descriptor; the new published checkpoint
tests executed. The documentation vocabulary and relative-link gates also
[passed](docs-links.log); packaging pins and closeout navigation checks passed.

The [Docker result](docker-result.json) records 3.461 seconds of preprocessing
for the supported instruction and 0.000452 seconds for out-of-scope screening.
These are single observations, not throughput or token-savings estimates.
The complete qualification took 4.607 seconds. The archive was mounted read-only;
no host home, credentials, network, or original development checkout was needed.
UID 0 inside the rootless container mapped output writes to the invoking host
user. The [driver](qualify_container.py) used a small authored Bottle-shaped
source fixture, not the official benchmark implementation. Only the original
public Terminal-Bench instruction was mounted from that dataset.

The [supported sidecar](docker-supported-advice.json) contains actual numerical
inference and replay. The [original benchmark prompt sidecar](docker-unsupported-advice.json)
reports `fail_open_input_out_of_scope`. Both original instructions remained
unchanged. This test stopped after preprocessing; it did not launch a coding
worker or obtain a Terminal-Bench reward. Container dependencies were installed
during an earlier image build; inference itself ran with `--network none`.

The [context result](context-result.json) and [inputs](context-inputs.json) use
three already-evaluated controlled instructions and one fixed addition function.
This is integration coverage, not a new holdout score. Each case ran startup
selection/replay, independently authored native admission, semantic capsules,
world-state capture, fresh Security checkpoint inference, generated association
v2, and actual Lake. The caller selected `left → capacity`, `right → threshold`,
and integer ranges `[-1, 1]`. Typed formulas came from the decoded fields through
the datasets association builder. Source bytes and task revisions remained
unchanged; tasks remained `ready`. Neither a checked counterexample nor an
empty enabled domain was promoted to satisfaction. The
[context driver](qualify_context.py) records this boundary explicitly.

## Failures retained and remaining work

The initial container run could not write output because of rootless UID
mapping. Subsequent runs exposed the Git-dependent producer import, including
an indirect import in the first attempted fix. The final portable checker has
no dependency on that checkout-only import and still checks exact module paths,
source hashes, and loaded function code. The fixed archive passed offline
preprocessing. Its direct source pins do not cover mutable globals, the entire
Python environment, or numerical model memory.

The [native Doctor run records](native-doctor-runs.json) retain the first START
timeout and subsequent diagnostic and normal successes. No lifecycle code or
deadline was changed, so the initial timeout is not claimed fixed. The normal
successful run completed in 18.387 seconds, published its validated repair, and
left zero processes. This is the existing keyword/signature Doctor operator;
learned arithmetic contracts were not inserted into that unrelated repair.

Learned advice does not supply admission or completion authority. Requirement
contract v2 symbolic planning still uses independently reviewed operations and
reports `not_used_for_symbolic_selection` for neural advice. Generated effect
configuration v2 is available through the task-context API; benchmark context
reuse and refresh do not yet automatically select its code/action mappings.
Counterexample-driven arithmetic proposal generation, fresh post-repair
inference, and native Doctor admission remain a separate integration gap.
No new native-Codex comparison, token score, or Terminal-Bench success is claimed.
