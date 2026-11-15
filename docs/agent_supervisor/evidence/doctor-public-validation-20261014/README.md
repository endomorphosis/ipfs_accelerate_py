# Public acceptance checks for Doctor continuation

The previous `largest-eigenval` trials passed correctness and dominance but
failed speed. Structural validation could still report native completion, and
a worker exiting successfully could not request a second Doctor attempt. This
change makes the public behavioral and speed requirements executable before
continuation and during canonical validation after publication.

## Explicit selection

The new policy is `doctor-coding-public-validation-two-attempts@1`. It requires
`--candidate-rounds 2`, `--arm full`, an enabled source view, disabled mapped
edits, and the explicit `terminal-public-task-profile@4` declaration below.
The numerical formalization broker remains a separate treatment. Existing
profiles and the one-attempt default retain their previous behavior.

```json
{
  "schema": "terminal-public-task-profile@4",
  "public_validation_profile": "largest-eigenval-public@1",
  "instruction_sha256": "6fa938389f99648508e5a9772e0d5e95d1102738c68d1ff61bd65469d76c52ab",
  "input_paths": ["eigen.py", "eval.py"],
  "outputs": [
    {"path": "eigen.py", "effect": "modify", "media_type": "text/x-python"}
  ]
}
```

The declaration is frozen into benchmark configuration and bound to the signed
task manifest. Preparation checks the reviewed public instruction and evaluator
hashes. Native admission verifies all three acceptance criteria and exact
validation commands; source partitioning attaches evidence to every criterion.
The implementation is specific to this public task, not general coverage of
Terminal Bench.

## Checks and continuation

The generated checker has separate structural, eigenpair and performance checks.
Behavioral cases include complex and negative dominant eigenvalues, nonsymmetric
matrices, finite nonzero eigenvectors, the residual equation, dominance and input
preservation. They include authored edge cases and seeded samples for sizes
one through ten.

Performance checks use the public sizes two, four, six, eight and ten, with three
batches of 100 paired inputs and alternating candidate/reference execution order.
They compare unrounded medians. Mixed timing direction or a median spread above
20 percent is inconclusive. This stability rule is an authored public-check
policy, not a claim about the hidden verifier's threshold. The owner retains all
15 measurements; Doctor receives five per-size median summaries, check outcomes
and failed obligations. No raw checker output or complete source inventory is
added to that feedback.

After the actual child has exited and its descendants have been cleaned up,
the owner runs the signed commands through the fixed isolated validation
launcher. Output is bounded, and lock wait, runner setup and execution consume
the existing deadline. Full candidate source hashes and file modes are checked
before and after validation. Immutable support inputs, the signed instruction
and original task constraints must remain unchanged. A separate cleanup check
must succeed before the owner retains a validation observation.

A worker exit code of zero remains zero in the record. A typed, live failed
public-check observation can admit a second attempt; an inconclusive check alone
cannot. A nonzero worker exit retains its existing retry path, with the new
profile's observation attached. There are still at most two child attempts and
two outer coding-router invocations, sharing the original deadline and Doctor
proposal allocation. No serialized observation can grant a retry.

The second worker checks the protected contract, observation, source generation,
original obligations and compact feedback binding before invoking the router.
Historical replay retains the first candidate's observation even after the
second worker changes the candidate. Attempt accounting joins the second Doctor
receipt to the first validation digest, status and actual exit code. Completion
still requires independent canonical validation; these empirical checks do not
create a theorem or discharge a symbolic proof obligation.

## Qualification

The accompanying [qualification.json](qualification.json) records **816 distinct
passing tests**: 173 public-profile/contract cases, 353 benchmark wiring cases,
259 legacy/affected continuation cases, three native continuation cases and 28
artifact-reader cases. It records 47 current source hashes, dependency pins and
developmental failures. Native tests use real worker
processes, signed task artifacts, source brokers, Doctor handoffs and historical
input replay. Provider responses, host isolation and native validation results
are explicitly authored boundaries in those tests. Separate checker tests run
the generated numerical code on real NumPy inputs, while deterministic timing
fixtures exercise stable failure and inconclusive outcomes.

Independent review also found deadline checks missing after slow runner setup,
contract/source replay and first-worker Doctor revalidation. The fixes preserve
the original allocation and refuse expired work before launch or reuse. Negative
tests cover forged or detached handles, source and mode drift, changed commands,
receipt substitution, failed cleanup, expired deadlines and third attempts.

The full continuation run passed 261 cases and failed one newly authored test.
That test invalidated its working generation through a source change, then tried
to reuse the stale observation for an independent deadline probe. Production
correctly refused the stale generation first. The test now probes deadline and
mode changes before the source change, with specific refusal assertions. All
three native cases passed on replay. Qualification excludes that entire native
module from the original run and counts the replay instead; the original
failure and single test-file hash replacement remain explicit. All production
hashes remained unchanged. The earlier independent source review is supplemented
by final package review of this test-only correction.

The artifact reader supports the new public-validation observations and Grok
rejection diagnostics through extracted pure validators. It exports only closed
status, reason, timing and digest fields. Re-reading the previous diagnostic run
preserves its historical result and unknown coding usage. Per-event native
progress counters are not summed into a token total.

This qualification does not establish a new live benchmark score or token
savings. A fresh run must use a newly frozen runtime archive, the explicit
profile and policy, and the qualified reader. The earlier live results remain
unchanged. Evidence files omit provider logs, credentials, private owner state
and runtime archives; local test runners retain historical absolute paths and
are provenance records rather than portable launch commands.
