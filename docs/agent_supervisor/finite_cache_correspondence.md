# Finite source-derived cache-key correspondence

`proof/finite_cache_correspondence.py` adds an owner-derived profile to the
lossless `canonical_cache_key_bridge.py`. It derives both the datasets
`CanonicalProofCacheKey` and the supervisor `ProofCacheKey` from the same current
captured source and native translation. A saved key or claimed checker name is
not an input authority.

The supported profile is `python-integer-offset-finite@1`: one ASCII Python
function with one annotated integer parameter, returning that parameter or the
parameter plus/minus a bounded integer literal, over 1–32 explicitly listed
integers. Native `compile_integer_offset` checks the exact source/ProgramIR/VC/SMT
correspondence. Imports, calls, additional declarations, global value reads,
decorators and other syntax are refused by that existing owner. The requested
offset can differ from the source offset; preparing a key does not establish
that the requested postcondition holds.

```python
contract = IntegerOffsetContract("calc.py", "increment", "n", 2)
inputs = [-2, -1, 0, 1, 2]
report = prepare_finite_cache_correspondence(
    index=index, repository=repository, expected_head=current_head,
    contract=contract, inputs=inputs, tool_policy=sealed_native_tools,
    scheduler=resource_owner,
)
semantic_key, execution_key, bridged_key = verify_finite_cache_correspondence(
    report, index=index, repository=repository, expected_head=current_head,
    contract=contract, inputs=inputs, tool_policy=sealed_native_tools,
    scheduler=resource_owner,
)
```

Verification repeats current-source observation, CAS reads, native lowering,
tool-byte hashing and producer-byte hashing, and compares the entire closed
report. The source owner is observed before and after preparation under one
bounded resource lease. A changed head/source, missing or corrupt CAS object,
changed tool, changed producer, cancellation or expired deadline prevents a
valid result. Reports are bounded to 2 MiB; the existing bridge has its separate
1 MiB envelope bound.

| Native key dimension | Owner-derived material | Supervisor field |
| --- | --- | --- |
| source | Current native head, path, exact source CID/hash/length, closed code dependencies, explicit model mode | `candidate_tree` |
| expression | Complete native ProgramIR | `obligation.program` |
| formalization | Exact compilation and VC, with conditional finite scope | `obligation.formalization` |
| slice | Contract path, function and parameter | `obligation.slice` |
| obligation | Native contract, complete finite domain, exact-int and requested-offset clauses | `obligation.native_request` |
| assumptions | Declared Python runtime assumptions and actual native body-model equation | `premises` |
| bounds | Complete native finite input domain | `obligation.finite_domain` |
| translation | Native compilation/source binding, reviewed producer hashes, explicit model mode | `translator` |
| provider | Native finite-observer producer identity | `solver.provider_id` |
| environment | Python/Lean tool pins, current lowering interpreter ELF pin, declared environment and runtime scope, producer hashes | `toolchain` |
| policy | Native tool policy, future observation operation ceiling, model/trust scope | `policy.native_policy` |
| schema | Exact contract/domain/compiler/correspondence profiles | `theorem_registry.schema_inventory` |
| checker | Exact selected Lean executable identity | `kernel.checker_id` |
| network policy | Preparation performs no target/checker execution; future observer requests no network, without an isolation attestation | `policy.network_policy` |
| evidence kind | `declaration` | `policy.evidence_kind` |
| authority ceiling | `none` | `policy.authority_ceiling` |

The report preserves all material and the explicit 16-field correspondence.
Hashed native dimensions use canonical JSON SHA-256; provider, checker and the
two evidence axes retain their native literal encoding. The existing bridge
then preserves both keys without dropping fields.

Semantic premises and finite input bounds never alias CPU/memory/time limits.
`resource_budget` retains the future observation ceiling and native process
limits separately. Those limits are declarations, not measured usage. Changing
the finite domain changes its semantic identity; changing an operation ceiling
changes policy/execution identity without changing the finite-domain identity.

Model mode is explicitly **off**. This profile does not infer learned model
lineage, invoke a model, train weights, or certify an embedding/checkpoint
dependency. Supporting a learned profile requires its own owner-derived
dependency contract.

The profile preserves native assumptions: an exact built-in integer caller,
unbounded mathematical arithmetic with resource failures excluded, and direct
sequential invocation with module loading/rebinding outside the model. Empty
code-dependency inventory means only that the guarded function has no imports,
calls or global value reads. It does not attest the Python standard library,
Lean libraries, shared libraries, operating system or all transitive imported
packages. The native tool policy retains those runtime trust limits.

All proof, execution, completion, checker-execution and contract-satisfaction
flags remain false. There is no top-level `obligation_id` in either generated
execution key or bridge envelope; existing positive `FormalVerificationCache`
receipt admission therefore still rejects them. A real finite observation can
be joined by exact head/source/domain/contract/compilation/tool identities, but
that join does not convert its receipt into a live proof capability or extend
its finite scope. General receipt admission, positive proof reuse, Quack-native
transaction support, and repository-wide dependency profiles remain separate
work.

Qualification and its retained native example are in
[the evidence directory](evidence/finite-cache-correspondence-20261002/README.md).
