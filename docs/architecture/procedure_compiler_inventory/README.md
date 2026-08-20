# Procedure compiler inventory

These compact artifacts implement the PCPC prerequisite gate. They describe
the committed starting tree only. Planning documents, board state, generated
receipt-shaped JSON, and unrelated working-tree overlays were excluded.

- `baseline.json` binds the starting repository, tree, package manifest blob,
  sibling gitlinks, audit environment, and every exact test producer command,
  test-source blob, expected result, and typed expected failure.
- `prerequisites.json` gives one closed disposition for each required prior
  authority. Every non-missing disposition binds exact source blobs, Python
  symbols, interface/schema constants, package and applicable submodule
  releases, and its test producers. Missing authorities instead carry typed
  blockers and deterministic negative probes; related implementations are
  explicitly marked non-equivalent.
- `authority_reuse.md` records canonical owners and typed compatibility gaps.

Runtime qualification receipts are written below the configured ignored
`state/agent_supervisor_proof_carrying_procedure_compiler/` directory. Those
receipts must bind the later exact implementation commit/tree. The materializer
runs a deterministic per-authority probe before and after the qualification
suite and executes all 18 declared test producers once on that current tree.
Each producer receipt binds the declared and executed argv, exact pass/fail/error
counts, output digests, current commit/tree, source blobs, and exact sibling
checkout receipt. A non-zero result is admitted only when its counts and every
declared typed-failure fragment match. Per-authority receipts then bind the
exact producer-receipt CIDs used for that disposition; missing authorities get
an explicit `not_applicable_missing_authority` record rather than fabricated
positive test evidence. The materializer rejects missing bindings, altered
baseline/HEAD/working-tree blobs, changed interface or schema constants,
changed or deinitialized gitlinks, dirty sibling checkouts, stale negative
probes, and unbound test producers. The probe, current execution envelope, and
both inventory digests are embedded in the qualification receipt. These
inventory files and their CIDs do not self-certify P0 completion.

Declared test expectations describe an actual fresh archive of commit
`bbf7f68799072c2b81f7d96eac91f2df3c4b3952`, with the recorded gitlinks
materialized. Current-tree qualification must reproduce those outcomes exactly;
it does not trust the archived counts. Matching a typed non-zero outcome is
evidence for the recorded caveat or incompatibility, never passing evidence for
that authority and never permission to represent blocked functionality as
available. The materializer only observes exact gitlink checkouts; it never
initializes, updates, or cleans sibling repositories itself.
