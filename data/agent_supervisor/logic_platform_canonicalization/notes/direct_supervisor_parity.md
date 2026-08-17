# LPC-141 Direct-versus-supervisor parity tests

**Task:** LPC-141 — Direct-versus-supervisor parity tests  
**Goal:** LPC-G140 (`LogicConformanceMatrix@1`)  
**Depends on:** LPC-111 (receipt admission), LPC-140 (hermetic matrix)  
**Interface focus:** Direct datasets contracts ↔ `SupervisorLogicPlatformClient@1`  
**Declared output:** `data/agent_supervisor/logic_platform_canonicalization/notes/direct_supervisor_parity.md`  
**Implied validation suite:** `test/api/test_direct_vs_supervisor_logic_parity.py`  
**Validation:**  
`python -m pytest test/api/test_direct_vs_supervisor_logic_parity.py -q`  
**Acceptance:** Representative operations agree on request, obligation, provider request, verdict, evidence, authority, boundedness, and receipt identities.

## Purpose

The logic platform has two legitimate call paths that must not invent divergent
semantic identities:

1. **Direct** — callers construct and consume typed datasets contracts in-process
   (`LogicObligation@2`, `BackendRequest@2`, `LogicProviderRequest`,
   `LogicProviderResponse@2`, `ProviderExecutionReceipt@2`).
2. **Supervisor-mediated** — `SupervisorLogicPlatformClient@1` handshakes, binds
   task/tree/policy/budget context, converts through
   `SupervisorLogicProviderFacade`, projects untrusted receipts, and only after
   LPC-111 ten-point admission may a result affect completion or merge.

LPC-141 freezes the **parity contract**: for representative operations, both
paths agree on the eight identity dimensions below. The supervisor may add
operational binding (task, tree, policy, cancellation, deadline, correlation)
but must not redefine request digests, obligation digests, provider-request
identity, semantic verdict, evidence kind/lineage, authority ceiling,
boundedness, or receipt content identity.

This lane is **required after LPC-140**. It is hermetic (fixture providers only):
no live prover, no network, no PATH-dependent toolchain, and no mock promoted
to a real-provider gate (that remains LPC-142).

## Eight parity dimensions (normative)

| # | Dimension token | Identity carrier | Agreement rule |
| --- | --- | --- | --- |
| 1 | `request` | `BackendRequest@2.content_digest` | Same admitted slice + obligation inputs mint the same request digest via `from_slice` / `from_obligation` / dict round-trip |
| 2 | `obligation` | `LogicObligation@2.content_digest` | Same admitted slice inputs mint the same obligation digest; wire round-trip preserves it |
| 3 | `provider_request` | Canonical hash of `request_id`, `operation`, `payload`, resource budget (schema key stripped), `network_allowed`, `deadline_unix_ms` | Direct `LogicProviderRequest` equals `to_logic_provider_request(ProviderRequest)` field-for-field on those identity fields |
| 4 | `verdict` | `LogicSemanticVerdict` / client `semantic_verdict` | Direct `ProviderResponseV2.verdict` equals supervisor `ClientResult.semantic_verdict`; operation success never invents `proved` |
| 5 | `evidence` | Evidence kind + source lineage digest | Direct response evidence kind + source digest equals supervisor payload evidence kind + request source digest |
| 6 | `authority` | `LogicEvidenceAuthority` / evidence authority field | Direct response authority equals supervisor-projected authority; provider-claimed upgrades are stripped; simulated kernel claims reduce to candidate |
| 7 | `boundedness` | `LogicBoundedness` | Direct response boundedness equals supervisor payload boundedness |
| 8 | `receipt` | `ProviderExecutionReceipt@2.content_digest` | Same execution lineage yields the same receipt digest; supervisor `receipts(...)` projection preserves it and never auto-admits (`admitted=False` until LPC-111) |

Closed inventory constant in the suite: `PARITY_DIMENSIONS` =
`(request, obligation, provider_request, verdict, evidence, authority, boundedness, receipt)`.

## Representative operations

Compact recipe generator (no bulk golden dumps). Each case binds one domain
slice, obligation, backend request, provider request, typed response axes, and
hermetic execution receipt:

| Operation | Case id | Domain | Focus |
| --- | --- | --- | --- |
| `prove` | `parity-prove` | `security_ir` | Satisfiability-style obligation + unknown verdict |
| `verify` | `parity-verify` | `legal_ir` | Reconstruction-ceiling evidence, resource-bounded |
| `reconstruct` | `parity-reconstruct` | `software_verification` | Reconstruction path, finite-trace boundedness |
| `translate` | `parity-translate` | `crypto_ir` | Non-proof translation axes (`not_applicable` verdict) |
| `capability` | `parity-capability` | `intent_ir` | Non-executable discovery; advisory authority only |

All five are members of `ClientOperation` and `ProofProviderOperation` /
`ProtocolOperationV2` closed vocabularies.

## Canonical call paths compared

```text
Direct path
  DomainLogicSliceV2 (admitted)
    → LogicObligation@2.from_slice
    → BackendRequest@2.from_obligation
    → LogicProviderRequest (datasets wire)
    → LogicProviderResponse@2 axes + ProviderExecutionReceipt@2
    → identity bundle (8 dimensions)

Supervisor-mediated path
  Same slice/obligation/request minting (datasets owns digests)
    → ProviderRequest (supervisor wire)
    → to_logic_provider_request / SupervisorLogicProviderFacade
    → SupervisorLogicPlatformClient@1.invoke / obligation / receipts
    → optional admit_receipt (LPC-111; never auto-admit)
    → identity bundle (8 dimensions)

Parity gate: direct_bundle == supervisor_bundle on all eight fields.
```

### Ownership split (must not invert)

| Package | Owns | Must not |
| --- | --- | --- |
| `ipfs_datasets_py.logic` | Obligation/request digests, provider wire semantics, response axes, execution receipt digests | Invent a second supervisor runtime |
| `ipfs_accelerate_py.agent_supervisor.proof` | Client context binding, facade conversion, untrusted receipt projection, ten-point admission | Redefine semantic digests or promote success to proof |

## Fail-closed parity invariants

1. **Success ≠ proof.** `operation_status=succeeded` leaves
   `semantic_verdict=unknown` and `proof_success=false` unless an explicit
   non-proof-promoting verdict was declared on the payload.
2. **Provider authority stripped.** Fields such as `provider_claimed_authority`
   never survive the client boundary; context ceiling caps authority.
3. **Simulated receipts.** Simulated envelopes that claim kernel authority are
   reduced to `candidate` and remain `admitted=false` / `trusted=false`.
4. **Receipt admission is LPC-111.** Client `receipts(...)` always projects
   untrusted envelopes; only `admit_receipt` may set completion/merge influence.
5. **Lossless provider conversion.** `to_logic_provider_request` preserves
   request id, operation, payload, budget integers, network policy, and deadline.
6. **No second identity.** `BackendRequest@2.from_slice` and
   `from_obligation` on the same inputs share `content_digest` and
   `obligation_digest`.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-110 | Client surface under test (`SupervisorLogicPlatformClient@1`) |
| LPC-111 | Ten-point receipt admission after untrusted projection |
| LPC-130 | Channel (Python/CLI/MCP) name/schema parity; different gate |
| LPC-140 | Hermetic matrix lists this lane as required-after-LPC-140 |
| LPC-052 | Typed response axes (verdict, evidence, authority, boundedness) |
| LPC-032 | Success is not proof; parity tests assert non-inference |
| LPC-142 | Real local provider smoke; mocks here never satisfy that gate |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/direct_supervisor_parity.md` | **This note — declared Outputs path** |
| `test/api/test_direct_vs_supervisor_logic_parity.py` | Implied validation suite (parity gate) |

Task-owned proposal envelope for LPC-141 (fail closed):

* **Declared Outputs:** this note.
* **Implied validation:** the parity suite path above.
* Paths outside that envelope (production modules, protected plan/board
  validators, undeclared companions) are **out of scope** for LPC-141 admission.

## What this task does **not** do

* Does not add a live prover or treat fixture providers as real-provider smoke.
* Does not weaken LPC-111 admission or auto-admit receipts.
* Does not redefine datasets semantic digests inside the supervisor.
* Does not absorb LPC-130 channel name/schema parity or LPC-142 smoke notes.
* Does not edit protected board/plan/validator files.
* Does not skip or `xfail` required parity assertions.

## Acceptance checklist

- Declared note exists and names the eight dimensions, representative
  operations, client interface, and validation command.
- Suite builds compact representative cases for `prove`, `verify`,
  `reconstruct`, `translate`, and `capability`.
- For each case, direct and supervisor identity bundles agree on request,
  obligation, provider request, verdict, evidence, authority, boundedness,
  and receipt.
- Provider request conversion is lossless on identity fields.
- Receipt projection preserves execution receipt digest and never auto-admits.
- Validation:  
  `python -m pytest test/api/test_direct_vs_supervisor_logic_parity.py -q`
