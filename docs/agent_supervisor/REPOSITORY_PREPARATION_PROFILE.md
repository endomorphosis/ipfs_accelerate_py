# Explicit repository preparation profiles

`repository_preparation_profile.py` supplies an opt-in finite qualification
profile around the existing datasets source, semantic manifest, model registry,
384-dimensional inference, and training lifecycle owners. It does not implement
another autoencoder, model head, or proof authority.

The independent selections are complete scan policy, semantic index contract
declarations, eager proof contracts/domains, training paths/roles/groups,
inference paths, and model policy. An explicit empty proof selection produces
no eager proof; the existing finite planner still demands its independently
checked evidence. Unsupported and unselected entries remain in the complete
inventory with their native status. Omitting `index_contracts` on the existing
preparation API preserves the earlier proof-contract-derived index behavior.
The closed new profile requires an explicit index selection.

The model policies are `model_off`, `pinned_parent`, `optional_training`, and
`required_training`. Shared-parent inference reads exact registered weights and
captured source without supplying source-derived target labels to the decoder.
Optional training records its numerical work and falls back explicitly to the
pinned parent when the child fails retention. Required training then refuses
before inference. Neither policy promotes a child automatically. The existing
ridge recipe is a complete head refit, with zero optimizer-resume steps;
`numerical_head_refits` and `numerical_inference_calls` are recorded separately.
Development holdout rows used for retention are deployment canaries, not an
untouched final test cohort.

Run the authored finite fixture with an exact absolute JSON profile:

```sh
python -m benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision \
  --output /absolute/fresh/output \
  --python /absolute/python \
  --lean /absolute/lean \
  --preparation-profile /absolute/profile.json
```

The schema is `finite-repository-preparation-profile@1`; serialize the native
`RepositoryPreparationProfile.to_dict()` rather than inventing implicit fields.
It selects managed pipeline resources automatically. The source384 profile
reserves 4096 MiB for numerical phases, 1024 MiB for structural and validation
phases, and a 6144 MiB parent with protected validation/cleanup capacity. The
existing default model-off profile retains its 3072 MiB parent. Host admission,
planning deadlines and proof requirements are unchanged. This remains the
sampled/cooperative envelope; it does not claim hard measurement of external
supervisor/worker RSS or arbitrary transient writes.

Before task admission, `PreparedRepositoryObserver` fully reconstructs the
native semantic manifest and binds the exact selected source head, policy
receipt, complete inventory, model version/artifact, inference result and
preparation report. Repeated observations reuse only that immutable semantic
reconstruction in the current process. Each read rehashes the semantic
manifest, native artifacts and producer bytes, then freshly checks current
source/ignore policy and selected model owners before and after the reads.
Exact descriptor/owner checks allow the driver to avoid repeating the same
semantic validator. Checked proof-cache validation remains independent and is
never memoized by this observer. A new observer, including after restart,
reconstructs again.

`finite-offset-scalar-cohort@1` explicitly adds nine authored scalar sources to
this integration fixture. Their operators have disjoint train/validation/holdout
roles. These sources are not injected into benchmark tasks. The finite offset
obligation, source repair and public check retain their original deterministic
strategy; model outputs are advisory and cannot establish proof, authorize
execution, or report completion. This qualification is not a Terminal-Bench
reward or a token-efficiency comparison.

The retained sixth run completed the full supervised fixture in 195.35 seconds,
including source repair, public validation, publication, independent cold
comparison, and cleanup with no remaining owned processes or leases. Its
planning/admission/context phase took 78.40 seconds with the original 90-second
planning limit and all 15 inventory entries retained. Three real pinned-parent
inference passes ran; there were no head refits or LLM calls.

This verifies checkpoint consumption and supervisor integration. Each inference
pass retained one `fail_open_source_contract_unsupported` row and three
`unqualified_candidate` rows. Those outputs did not establish a verified neural
formalization: the deterministic finite symbolic planner and independent proof
checks governed the repair. No model was promoted.

The five preceding attempts, including two profiled diagnostics, remain in the
evidence. Repeated immutable reconstruction first prompted the bounded observer;
profiling then exposed expensive repeated CID registration lookups. The content
owner now uses a bounded pure CID memo with fresh guarded registry checks and
the original validation fallback. Current source, model, CAS, AST and checked
proof fences remain active. These are finite fixture results, with no matched
Terminal-Bench or Codex efficiency claim.

Raw runs remain under `artifacts/repository-preparation-profile-20261002` in the
parent workspace. Recovery provenance and fresh controls are separately recorded
under `artifacts/repository-preparation-recovery-20261002`.

The qualification package records 83 distinct passing controls across 85
selected test attempts. Two attempts of the independent proof-demand case were
refused before work by the real CPU admission policy. Its final attempt passed
with unchanged runtime owners and thresholds. A test-only helper now bounds
retries of that exact pre-work refusal; the successful run admitted immediately,
so it does not demonstrate runtime pressure recovery or the helper's backoff.
Both refusals and the changed test snapshot remain visible in
`evidence/repository-preparation-profile-20261002/qualification.json`.
