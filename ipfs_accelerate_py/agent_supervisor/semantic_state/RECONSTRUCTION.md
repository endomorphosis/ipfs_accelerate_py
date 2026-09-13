# Complete committed reconstruction

`reconstruct_semantic_state` requires the datasets producer API
`semantic_index.committed_snapshot` (introduced by datasets commit
`9c67dd3fab5dab4270720e7a56f1eba926c18a7f`). This consumer builds on the cold
reconstruction implementation in accelerate PR215. Deploying it independently
with an older datasets producer is unsupported; the native launcher must bind
and qualify both loaded source revisions before admission.

Reconstruction explicitly selects every committed path, including names such
as `coverage`, `vendor` and `venv` that ordinary repository scans exclude.
Exact raw-path equality, commit/tree checks, clean-index checks, opaque inputs,
transitive bundle verification and nomination comparison remain required.
Source identity and the complete population are observed again after the cold
scan. These observations do not replace native mutation fences.

The 20,000 entry, 4 MiB per-blob and 128 MiB total defaults remain unchanged.
Git metadata output now also has an explicit 32 MiB bound. An oversized request
raises `ReconstructionBudgetError` before snapshot acquisition; its `.plan`
contains the full metadata inventory, totals, largest size and all budget
deficiencies. It never silently reduces coverage to satisfy the bounds.

Successful observations expose the population CID and `complete-committed`
scope, also bound into the configuration digest. Neither the observation nor
a matching nominated root grants semantic acceptance or task completion.
Multi-gigabyte populations remain refused pending bounded content storage,
streaming/chunked snapshot and scanner/state support, and separately qualified
native execution evidence. Increasing the limits alone does not meet that need.
