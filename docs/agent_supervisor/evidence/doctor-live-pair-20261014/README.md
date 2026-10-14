# Doctor continuation: first live pair

Two fresh, serial Terminal Bench `largest-eigenval` trials used the full indexed
supervisor through `llm_router` and Grok 4.7. Both selected
`doctor-coding-two-attempts@1`. The only treatment difference was the native
coding turn limit: the default 128 versus an explicit 8. Output paths necessarily
differ. Both used the same frozen runtime archive, source/dependency commits,
model profile, identity source view, disabled mapped edits, resource profile,
and original 1,800-second shared coding deadline. The separate numerical
obligation broker was not selected. This pair does not compare native Codex or
the no-index supervisor.

| Native turn limit | Official reward | Coding children started | Coding receipts | Observed token subtotal | Supervisor time |
| --- | ---: | ---: | ---: | --- | ---: |
| 128 | 0 | 1 | 1 | 1,764,408, planning and coding | 984.97 s |
| 8 | 0 | 2 | 1 | 23,347, **planning only; coding unknown** | 413.55 s |

Both official verifiers passed eigenpair correctness and dominance, and failed
the speed test. No verifier feedback was supplied to either trial. The second
treatment was prepared and frozen before the first result was known, and ran
after the first container had been removed. Execution and collection preserved
all 621 pinned Python sources and the pinned datasets/kit commits. The second
container was also removed after its trial.

The [summary](pair-summary.json), [128-turn readout](turns128/readout-final.json),
[8-turn readout](turns8/readout-final.json), and
[bounded diagnostic observations](diagnostic-observations.json) retain the
measurements. Complete native usage and billing totals remain unknown. These
failed single trials establish no token savings or benchmark advantage.

## What caused the large token count

The default-limit run made one planning router call and one coding router call.
The latter contained 27 native coding turns. Its observed final-envelope counters
were 130,439 uncached input, 1,559,552 cached input, and 52,123 output tokens,
totalling 1,742,114 coding tokens. Planning added 22,294. Reasoning is already
included in output and is not added a second time. Repeated cached input accounts
for most of this observed volume; the retained counters do not establish the
cost or marginal token effect of a particular tool or prompt component.

The eight-turn treatment cannot be used to calculate a reduction. Its first
coding child reached the native turn limit and exited 1. Eight progress usage
events survived, but their counter semantics are unverified and their summation
is forbidden. No usable final coding usage was retained. The second child was
reserved, started, waited and retired, but produced no router receipt. Accounting
therefore correctly reports a potentially unreceipted attempt and incomplete
usage. The 23,347-token observation covers planning alone.

## Gaps exposed by the pair

**Doctor input audit serialization.** The final audit collector dropped the
original `provider`, `requested_output_tokens`, and
`provider_output_token_cap_enforced` fields before sending a Doctor v2 receipt
to its independent subprocess. The parser needs those fields to check the nested
capability claim. Both live audits consequently reported `unknown` with
`no_usable_coding_receipt`. The subsequent fix preserves the original fields
without synthesizing missing values or relaxing validation. The historical
live outcomes remain unknown; private owner artifacts were not exported for
independent host replay.

**Grok terminal-event handling.** The capped run retained one
`max_turns_reached` event followed by one `end` event. The pre-fix parser treated
the latter as invalid post-terminal data, tainted the stream and discarded its
terminal payload. This lost useful failure classification and potentially
final-envelope accounting. The retained metadata establishes the sequence only:
the original `end` payload is unavailable, so whether it contained usable usage
cannot be recovered from this report. Per-event usage must not be promoted to
a complete total.

**Ambiguous child errors.** The bridge's bounded log collector reported
`ambiguous` with no diagnostic. That status can mean multiple error rows, or a
valid row mixed with malformed matching data. The exact population and the
second-child exception are unavailable after container cleanup. Two started children demonstrate
owner lifecycle continuation; they do not demonstrate two model invocations.
Subsequent diagnostic improvements can retain multiple valid errors, but cannot
retroactively identify this exception.

**Public acceptance and continuation.** The default-limit child exited 0 and
the supervisor reported completion after its signed structural smoke check.
That check verifies output bounds and Python syntax, not the public speed
requirement. The public evaluator prints timing medians without asserting their
ordering. Current continuation requires a failed coding child; it does not
evaluate public behavior between successful child exits. The stronger validation
and continuation work is specified below and remains outstanding.

## Public validation work to complete next

1. Add an explicit versioned public-validation profile, preserving the legacy
   structural profile. Derive closed structural, eigenpair-behavior and timing
   checks from the public instruction and evaluator. Generate them in the existing
   signed support file with separate literal argv and acceptance keys. Compare
   unrounded measurements using a documented public criterion; timing instability
   must be inconclusive. Do not import hidden verifier thresholds.
2. Capture a typed validation contract from the already signed public-instruction
   manifest and exact task selection before coding. Bind the manifest, validation
   argv, acceptance keys, immutable inputs, declared outputs, original requirements
   and fixed isolated runner. Doctor's advisory shell strings are not execution
   authority.
3. After actual child retirement, run the signed checks on one captured candidate
   generation under the existing deadline and privilege boundary. Check source
   hashes and modes before and after execution. Retain a private live observation
   tied to the launch, retirement, contract and generation; archived JSON grants
   no continuation, completion or proof authority.
4. Add an explicit validation-aware continuation policy. Thread a typed validation
   failure through the monitor, second-worker preparation, admission revalidation
   and closed-baseline gates. Preserve the child's actual exit code 0 and record
   the owner's validation outcome separately. Reserve at most two attempts with
   no deadline extension or third retry. Give Doctor failed public obligation IDs
   and bounded measurements for the next attempt.
5. Rerun signed canonical validations after accepted publication using the existing
   completion bridge. Candidate checks must not substitute for current-source
   completion evidence. Test correct-but-slow and incorrect-but-fast candidates,
   stale/forged observations, protected-input drift, cleanup failures, and refusal
   to complete with a missing behavioral or performance check.

This work connects machine-checkable public obligations to the existing Doctor
loop. It does not make timing measurements a formal proof of performance or
claim that the current symbolic analysis discharged those obligations.

## Evidence boundaries

The runtime was frozen at `3a7ec1c97be93dab918f005c93a9bbdb42409512` with archive
SHA-256 `2d7bce19663b874610a9e59cee6d5c8b71393145637df2e9918138c3612307fe`.
It predates the fixes accompanying this report. Each copied reader passed its
11 qualification tests before launch. Independent post-run checks confirmed
the reported totals, source pins and exact treatment difference.

[Copy provenance](copy-provenance.json) identifies exact allowlisted original
files and hashes. Absolute paths inside copied records are historical provenance;
the launchers are one-shot experiment records, not portable rerun commands.
Private owner databases, signing keys, credentials, raw prompts/provider text,
runtime archives, and campaign logs are excluded. The first readout also carries
an older descriptive run as `prior`; only the pair declared here shares the
matched controls. A fresh runtime and fresh trials are needed to qualify the
subsequent fixes live.
