# Package import-alias qualification — 2026-10-07

The Doctor can now repair a unique explicit imported-function alias in a
captured regular Python package. Absolute and relative imports bind to signed
source paths; inert ancestor initializers participate in both the reconstructed
contract and dependency impact graph. See the
[contract guide](../../PACKAGE_ALIAS_CONTRACTS.md) and
[source-bound results](qualification.json).

The first combined run passed 411 checks and exposed one stale negative-test
expectation: the existing owner already rejects duplicate profile keys during
manifest authoring, earlier than the test's expected Doctor refusal. A test-only
correction now asserts the exact early error, duplicate-key cause and absence of
Doctor state. All 17 source-partition tests then passed. The original failed run
is retained; production admission code was not changed.

After preserving concurrent upstream context-audit changes, 22 package, context
audit and native lifecycle checks passed on merge commit `00daa9e1e`. These run
counts overlap and must not be added as independent coverage. All three runs
recorded unchanged source, no skipped tests and matching log/XML hashes. Tested
source bytes equal the committed implementation, except the explicitly
superseded negative-test fixture, whose corrected bytes passed the focused run.

The authored native fixture passes signed indexed planning, real Lean/Z3
obligations, staged candidate execution, native START, validation, publication,
task completion and STOP with zero tracked process members. Its context bundle
is attached and refreshed after publication. DuckLake metadata reports two
stored catalog records and two links. Model entrypoints are forbidden in both
the parent and candidate worker; observed provider calls are zero.

The signed smoke is structural; separate Python behavior checks observe the
original NameError and repaired results. Intent interpretation is authored and
retrieval is lexical. The finite proof assumes ordinary source-based imports;
it does not prove Python's import machinery, donor computation or whole-program
correctness. No new learned autoencoder, embedding, Docker/Terminal-Bench score
or matched token-saving result is claimed.

The datasets dependency remained at `47d492c69`. The optional Kit backend came
from the existing clean editable checkout recorded in the receipt; this does
not qualify the parent workspace's Kit pin. Local raw commands, logs and XML
remain under `artifacts/package-alias-20261007/qualification` in the parent
workspace. Git skipped the installed nonexecutable precommit hook. GitHub's
upstream documentation job could not start because the account is billing
locked; local checks do not establish CI success.
