# Procedure compiler inventory

These compact artifacts implement the PCPC prerequisite gate. They preserve
the committed starting tree and the exact source revisions admitted by later
current-tree requalification. Planning documents, board state, generated
receipt-shaped JSON, and unrelated working-tree overlays were excluded.

- `baseline.json` binds the starting repository, tree, package manifest blob,
  sibling gitlinks, audit environment, and every exact test producer command,
  historical test-source blob, any distinct current-tree blob, expected result,
  and typed expected failure.
- `prerequisites.json` gives one closed historical disposition for each
  required prior authority and, where later source was admitted, a separate
  closed `current_disposition`. Exact current qualifications bind source blobs,
  Python symbols, interface/schema constants, package and applicable submodule
  releases, and test producers. The narrow
  `source_presence_with_typed_interface_gap` mode instead records a current
  source and producer qualification without inventing a standalone versioned
  interface or schema. Every current override is explicitly barred from
  becoming the historical comparison baseline. Historical missing authorities
  retain typed blockers and deterministic baseline-negative probes; related
  implementations remain explicitly non-equivalent.
- `authority_reuse.md` records canonical owners and typed compatibility gaps.

Runtime qualification receipts are written below the configured ignored
`state/agent_supervisor_proof_carrying_procedure_compiler/` directory. Those
receipts must bind the later exact implementation commit/tree. The materializer
runs a deterministic per-authority probe before and after the qualification
suite and executes all 22 declared test producers once on that current tree.
Each producer receipt binds the declared and executed argv, exact
pass/fail/error counts, output digests, current commit/tree, current source
blobs, and exact sibling checkout receipt. A non-zero result is admitted only
when its counts and every declared typed-failure fragment match. Per-authority
receipts then bind the exact producer-receipt CIDs used for that disposition and
whether the evidence is admitted as a comparison baseline; current overrides
always seal that field as false. Missing authorities get an explicit
`not_applicable_missing_authority` record rather than fabricated positive test
evidence. The materializer rejects missing bindings, altered
historical baseline, current `HEAD`, and working-tree blobs; changed interface
or schema constants; changed or deinitialized gitlinks; dirty sibling
checkouts; stale negative probes; and unbound test producers. A historical
negative probe is evaluated at the baseline only when a closed current
disposition supersedes it; otherwise it must remain absent at both the baseline
and current tree. The probe, current execution envelope, and both inventory
digests are embedded in the qualification receipt. These inventory files and
their CIDs do not self-certify P0 completion.

The historical blob in each source binding describes commit
`bbf7f68799072c2b81f7d96eac91f2df3c4b3952`. A `current_blob_id` records an
explicitly requalified divergence, while `baseline_absent: true` records a
source first admitted after that baseline. Declared test expectations describe
the current requalification tree with the recorded gitlinks materialized;
qualification must reproduce those outcomes exactly and never substitutes
historical counts. Matching a typed non-zero outcome is
evidence for the recorded caveat or incompatibility, never passing evidence for
that authority and never permission to represent blocked functionality as
available. The materializer only observes exact gitlink checkouts; it never
initializes, updates, or cleans sibling repositories itself.
