The next pack experiment should start with a reversible projection of duplicate token-accounting fields and constant reference envelopes. The producer contains exact reconstruction rules for these fields. This review inspected implementation code only; it did not inspect private benchmark inputs, execute a projection, or measure token savings.

The source contracts are `ContextPackResult.to_dict` and `ContextPacker.pack` in `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py`, `ContextPack.to_dict` in `semantic_state/contracts.py`, `ContextReference._payload` in `context/context_contracts.py`, and `_slots` in `runtime/semantic_router_translation.py` in the frozen `supervisor-inline-evidence-census-20261015` worktree.

1. Factor the duplicate token estimate first.

   The producer assigns `ContextPack.token_totals = dict(token_estimate.totals)` and `ContextPack.estimator_version = token_estimate.estimator_version`. Its `ContextTokenEstimate` validates that `total` equals the sum of category totals. Consequently, the existing outer `translated_semantic.pack.token_estimate` can be reconstructed from the remaining inner `translated_semantic.pack.pack` as:

   ```python
   {
       "totals": inner_pack["token_totals"],
       "estimator_version": inner_pack["estimator_version"],
       "total": sum(inner_pack["token_totals"].values()),
   }
   ```

   Admit this recipe only when the original object has exactly those three fields, totals are strict nonnegative integers rather than Booleans, and canonical serialized equality matches the complete original object. Preserve all category names and values through copying; the recipe does not run the estimator or reinterpret its policy. If an extension adds another field or changes the duplicate values, retain the literal original object. A short versioned recipe marker must remain visible in the candidate input, with its decoding rule included in the candidate's measured instructions.

2. Factor repeated reference envelopes without merging roles.

   `ContextPackResult.references` serializes each `ContextReference` through `to_dict`, so every normal record repeats the same `schema` and `contract_version`. The current `_reference` constructor also supplies empty `repository_id` and `tree_id`; most references have an empty `path` and zero `byte_count`. These are conditional opportunities, not assumptions to impose on arbitrary inputs.

   An initial fixed allowlist can be limited to `schema`, `contract_version`, `repository_id`, `tree_id`, `path`, and `byte_count`. Lift a field into one inline `reference_defaults` object only if every reference explicitly contains exactly the same canonical value and type. For the last four fields, initially restrict the shared value to the producer's literal empty-string or integer-zero default. Preserve row order, reference count, and every remaining field. Leave fields with any differing row value literal. Unknown outer and metadata keys remain on their original rows.

   Keep `kind`, `tier`, `reference_id`, `referenced_content_id`, `summary`, `token_count`, and the entire `metadata` object literal in the first implementation. This keeps requiredness, caveats, role distinctions, and source requirements visible. Equal source CIDs in target, surrounding, and test roles do not authorize deleting any reference. The fixed defaults are a representation rule, not proof that the references are interchangeable.

3. Consider objective copying only after the two simpler recipes.

   `_worker_payload` passes `base["objective"]` to the packer, whose text normalizer may strip whitespace. The inner pack objective therefore often duplicates `translated_semantic.objective`, but unconditional equality must not be assumed. A fixed copy rule is possible only when their complete canonical string values match. Preserve the literal inner value otherwise. Do not normalize wording while measuring or decoding.

4. Defer derived IDs and admission-dependent reconstruction.

   `pack_cid` is exactly `cid_for_payload(inner_pack.to_dict())` in the current producer, and the current translation `_slots` does not alias the inner pack. It is theoretically reconstructible from the complete retained inner pack through the existing canonical-artifact/CID implementation. However, the opaque ID is useful for model references, and removing it makes the representation dependent on a specific CID recipe. Keep it literal initially; any later experiment must pin the existing CID implementation and verify the exact retained ID, without minting a different semantic identity.

   Some reference `referenced_content_id` values identify the same native CID as inner pack source fields or `metadata.source_cid`. They are not necessarily equal serialized values: the existing transport aliases selected reference fields while leaving the inner pack and metadata strings literal. A naive field-copy rule would break exact restoration. Do not extend alias slots as a side effect of pack factoring. A later independently versioned identifier projection would need explicit dictionary resolution, exact original-path reconstruction, and tests covering aliased and unaliased values.

   Leave `coverage_satisfied`, `budget_exceeded`, risk, route, escalation, assumptions, exclusions, decisions, admission state, and production-slice identity literal. The full packer derives these using budget limits, policy, admissions, and optional inputs that are not all retained in the pack. Re-running that packer is not a pure inverse and could produce different current judgments. No strings containing embedded CIDs should be rewritten by pattern matching.

The implementation should expose a separate opt-in `semantic-pack-view@1`, projection receipt, strict inverse, and complete-input selector. Start from the current typed `EncodedSemanticPrompt`; verify the existing native inverse before projection. Restore exactly the original translated transport first, then use the unchanged semantic translation table to recover native bytes. A final full-byte equality check is required. Do not change native packs, producer CIDs, admission artifacts, or existing transport/census schemas. Unsupported source transport, recipe, malformed record, ambiguous marker, or failed identity check must refuse; valid but ineligible fields stay literal.

Measure the pack experiment against the same original complete model input used by the current independent inline-evidence measurement. Include all copy/default instructions, markers, dictionary bytes, reply schema, workspace/source/public-instruction/Doctor wrappers, and retain the original when the candidate is not strictly smaller. Initially report it independently of capsule references and inline-evidence projection. A later composition needs its own exact inverse and complete-input measurement; independent byte reductions cannot simply be added.

Required qualification covers:

- Real producer outputs for raw-required, exact-substitute, and conservative-with-caveats admissions; empty and nonempty reference populations; explicit program and bounded capsule selections.
- Exact token-estimate copies, zero totals, strict Boolean rejection, unknown category preservation, extra estimate fields, divergent totals/estimator versions, and unsupported schemas.
- Common defaults versus one differing or missing row field, nonempty paths/IDs, unknown metadata, equal CIDs in different required roles, preserved order, UTF-8, escaped strings, and marker collisions.
- Existing alias objects versus native CID strings; unchanged translation table, pack CID, required flags, caveats, source obligations, and full native bytes after reconstruction.
- Tampering with defaults, recipes, rows, native/candidate hashes, retained receipts, complete-input wrappers, and authority flags.
- Provider-free normal-worker capture versus independent owner replay, authenticated source freshness before and after observation, zero provider calls, bounded closed diagnostics, and no raw data or dynamic labels in exported reports.

The current census's `repeated_groups_translated_value_bytes` includes every occurrence in a repeated identity group. It is not the number of removable bytes and must not be presented as a savings estimate. Report measured complete-input bytes for each candidate first; any token-efficiency claim requires a later matched live ablation with final native usage counted once per invocation.
