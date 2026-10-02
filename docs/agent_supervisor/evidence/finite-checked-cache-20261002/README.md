# Finite checked-cache qualification

The selected `python-integer-offset-finite@1` profile satisfies the RPI-007
storage/reconstruction criterion locally: canonical complete-key entries,
artifact bodies, checked table evidence, reverse dependencies and bounded exact
lookup survive a fresh process restart. Positive eligibility still requires a
fresh native Python/Lean run. This is not generic repository-proof coverage,
source-runtime equivalence, worker admission or avoided-checker performance.

The standalone authored fixture is `increment(n) = n + 1` over inputs
`[-2,-1,0,1,2]`. Its requested `n + 1` contract creates one positive entry; its
requested `n + 2` contract creates one explicit refuted entry. The reproducer
reopens both the structural DuckDB/CAS and existing formal cache in a distinct
process, freshly checks the positive entry, reconstructs the negative entry,
and verifies immutable duplicate publication and complete reverse references.

`positive.json` and `refuted.json` retain complete records, full 16-dimensional
owner correspondences and the exact table-only obligation/receipt. Each
`*-artifacts` directory contains its original native observation bundle and all
eleven artifact bodies, including the generated Lean theorem source and compiled
`.olean`. Original temporary paths are historical identifiers. The cache
reconstructs bodies into a fresh private directory and reruns the existing native
integrity validator; it does not require those old paths to exist.

The final new-module suite passed **42 tests in 128.63 seconds**, with zero
failures or skips. It includes every native artifact omission, CAS corruption,
rehashed Lean-statement tampering, independently rehashed SQL-schema tampering,
exact key/producer/source/tool checks, duplicate and concurrent publication,
bounded reverse queries, real failed Python/Lean processes, interrupted
publication recovery and fresh-process reconstruction.

The retained pre-final combined run passed all **126 unchanged neighboring
regressions**: 22 formal-cache tests, 46 finite correspondence tests and 58 full-key
bridge tests. That run also passed 37 then-current new tests and exposed one
incorrect immediate-retry expectation. The existing single-flight owner correctly
retains its failed result for a bounded TTL. The corrected test requires that
refusal and retries after the owner clock advances beyond expiry. Four additional
normative-schema controls and that corrected test passed in a narrow run before
the final 42-test run. Thus **168 distinct current tests passed**; repeated tests
are not added to the total. The diagnostic log/XML are retained explicitly rather
than relabeled as an all-green run.

`qualification.json` records the precise scope and counts. `producer-sources.json`
captures all 44 direct source/compiler/execution/storage owners; `test-sources.json`
pins the qualification files. `manifest.json` hashes this closed evidence set.
No checkpoint, model inference, training corpus, external publication or daemon
permission was introduced by this qualification.

Run the standalone reproducer with the released packages and their installed
dependencies on `PYTHONPATH`:

```sh
python reproduce.py --output /absolute/new/qualification-directory
```

It selects the installed native Lean 4.34.1 binary declared in the script and
the current native Python executable, verifies their byte identities, and keeps
the existing resource checks. See [the API and limits](../../finite_checked_cache.md).
