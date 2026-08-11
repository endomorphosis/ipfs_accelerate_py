# Planner/Doctor — Optional ZKP Threat Model (PDR-060)

Status: normative for optional Planner/Doctor attestation  
Interfaces: `ReasoningRunManifest@1`, `PlannerDoctorAttestation@1`  
Contracts: `ipfs_accelerate_py/agent_supervisor/proof/planner_doctor_attestation.py`  
Related doctrine:

- [`agent_supervisor_planner_doctor_threat_model.md`](./agent_supervisor_planner_doctor_threat_model.md) (authority ladder; T12 ZKP overclaim)
- [`agent_supervisor_codebase_proof_zk_threat_model.md`](./agent_supervisor_codebase_proof_zk_threat_model.md) (sim ≠ ATTESTED; private witness)
- [`SWISSKNIFE_CONTRACT_ZK_THREAT_MODEL.md`](./SWISSKNIFE_CONTRACT_ZK_THREAT_MODEL.md) (layered property / integrity / attestation)

This document is the **approved** privacy / fixed-computation claim gate for
Planner/Doctor zero-knowledge. A real proving backend may be selected only for
the exact claim named here. It does **not** authorize ZK as a substitute for
program semantics, inventory completeness, translator soundness, or completion.

## Exact security claim

| Layer | Question answered | Authoritative mechanism | ZK effect |
| --- | --- | --- | --- |
| **Semantic / kernel proof** | Does a Planner or Doctor obligation hold under current roots? | Existing analysis, solvers, and independently reconstructed kernel receipts | **None.** ZK cannot create, replace, or strengthen semantic proof. |
| **Receipt lineage integrity** | Are these the exact typed CIDs (Planner, Doctor, cache, plan, permit, mutation, fixed-point, benchmark, promotion) for this `run_id`, in this order, under this Merkle root and tree/policy pin? | `ReasoningRunManifest@1`, typed preimage recomputation, ordered Merkle root, signature when present | **None required.** Signatures and Merkle proofs already bind public lineage. |
| **Optional ZK attestation** | Does a holder know a **private witness** that satisfies one approved fixed circuit over those public lineage inputs (possession, membership path, or fixed bounded computation) **without revealing the witness**? | Real, capability-qualified cryptographic backend + independent verifier over pinned public inputs | May contribute production `AssuranceLevel.ATTESTED` **only** after the first two layers pass and the backend is production-eligible. |

### Approved predicates (closed)

Use-case id: `pdr-060-receipt-lineage-private-witness`  
Threat-model pin: this document path.

1. **`receipt_lineage`** — public inputs commit to `run_id`, `manifest_id`,
   ordered `lineage_merkle_root`, repository tree, and policy; the circuit
   checks consistency with the fixed codec (no semantic expansion).
2. **`private_witness_possession`** — prover knows openings / secret paths for
   committed leaves or a reviewed private intermediate **without** publishing
   those openings.
3. **`fixed_bounded_computation`** — a pinned circuit/program executed over
   committed public inputs yields a committed result (counter aggregation or
   membership under a fixed program).

Any other predicate is out of scope until this threat model is revised.

## Why ordinary signatures and Merkle proofs are insufficient

Lineage CIDs, ordered Merkle roots, and ordinary signatures are the correct
tools for **public integrity** inside or across cooperating operators:

- they bind typed preimages and reject wrong preimage / order / root / run
  replay through `ReasoningRunManifest@1`;
- a verifier who already trusts the signer’s key can accept the public
  statement.

They are **insufficient** for the optional claim above when **both** hold:

1. **Privacy** — a third-party consumer must accept that a private witness
   (benchmark holdout opening, secret membership path, proprietary intermediate
   receipt body, or proving transcript) satisfies the circuit **without**
   receiving that witness; a signature over a public manifest necessarily
   discloses or omits that material rather than proving possession of it.
2. **Computation without disclosure** — the consumer must check that a **fixed**
   pinned circuit/program was satisfied on the committed public inputs; a
   Merkle root proves leaf inclusion of public CIDs, not that a private
   relation over those leaves held inside a circuit.

Therefore:

- signatures + Merkle prove **what was committed and ordered** for one run;
- ZK (when approved and real) proves **a private relation over those public
  commitments** without witness disclosure;
- ZK never upgrades inventory completeness, translator soundness, or arbitrary
  Python semantics.

If there is no private witness and no cross-trust-boundary verifier, **do not
select a ZK backend**. Public lineage verification alone is enough.

## Roles and trust boundaries

| Role | Responsibility | Must not assert |
| --- | --- | --- |
| Planner / Doctor producers | Emit typed receipts and body-free CIDs | Collapsed evidence tiers or self-certified completion |
| Manifest assembler | Build `ReasoningRunManifest@1` with every lineage slot and Merkle root | Semantic correctness of linked receipts |
| Attestation prover | Hold the private witness; emit public envelope only | `ATTESTED`, production eligibility, or verifier success |
| Independent verifier | Recompute public inputs; check pins; verify cryptographic proof | Theorems not encoded in the circuit |
| Supervisor / gate | Enforce threat model, sim≠ATTESTED, preimage/order/root/run replay | Serialized `authoritative` flags without re-derivation |
| Optional external consumer | Accept only reproduced verifier results | Cross-run envelopes or foreign tree/policy roots |

There is **no** second proof-cache authority root. Cache hits re-derive lineage
and attestation; they never mint `ATTESTED`.

## Protected witness

A **protected witness** is any proving input that is not a declared public
input. Examples under this use case:

- commitment openings for lineage leaves;
- private membership paths;
- secret benchmark holdout bodies used only to prove possession;
- prover randomness and toxic setup waste.

Witnesses must **never** appear in:

- `ReasoningRunManifest` JSON or CIDs;
- public attestation envelopes, verification records, or cache entries;
- taskboards, logs, context capsules, or completion references.

Implementation: `PrivatePlannerDoctorWitness` is non-serializable, redacted in
`repr`, and rejected by public-artifact helpers.

## Public inputs (fixed codec)

The optional circuit binds exactly these public commitments (codec
`planner-doctor-attestation-public-input-codec` v1):

| Slot | Meaning |
| --- | --- |
| `run_id` | Exact reasoning-run identity (blocks cross-run replay) |
| `manifest_id` | Content identity of the typed lineage manifest |
| `lineage_merkle_root` | Ordered Merkle root over typed leaves |
| `repository_tree_id` | Current repository forest / tree pin |
| `policy_id` | Authority / attestation policy pin |
| `circuit_id` / `circuit_version` | Fixed verification program |
| `proving_key_id` / `verifying_key_id` / `ceremony_id` | Setup and key pins |
| `use_case_id` / `threat_model_id` | This approved claim and document |
| codec id / version | Public-input ordering pin |

Evidence types remain **distinct** in the manifest (Planner, Doctor, cache,
plan, permit, mutation, fixed-point, benchmark, promotion). The Merkle leaf for
each slot binds `(run_id, evidence_type, cid)` so type collapse or reordering
changes the root.

## Replay and freshness

| Check | Fail-closed response |
| --- | --- |
| Wrong preimage for a typed CID | `LineagePreimageError` |
| Reordered / collapsed / substituted evidence types | `LineageOrderError` |
| Wrong or forged Merkle root | `LineageRootError` |
| Cross-run replay (`run_id`, tree, policy, or root drift) | `LineageReplayError` |
| Public-input digest / verifying-key / circuit drift | verification rejected |
| Simulated / shadow / unavailable / failed backend | status stays candidate/unavailable/failed; **never** production `ATTESTED` |

## Explicit non-claims

A valid lineage check or optional ZK verification **does not** prove:

- `semantic_correctness`
- `inventory_completeness`
- `translator_soundness`
- `arbitrary_runtime_semantics`
- `goal_completion`
- `theorem_beyond_committed_circuit`

These strings are normative in the contract
(`ATTESTATION_DOES_NOT_PROVE` / `ATTESTATION_SCOPE_STATEMENT`). Promoting an
attestation into any of them is a hard error.

## Backend disposition (sim ≠ ATTESTED)

| Backend outcome | Envelope status | Max assurance | Production / completion gate |
| --- | --- | --- | --- |
| Cryptographic + production-eligible + independent verify + sealed `attested` | `attested` | `ATTESTED` | may satisfy |
| Cryptographic but not sealed / not production-eligible | `generated` / candidate | `CANDIDATE` | fail closed |
| Simulated | `simulated` | `CANDIDATE` | fail closed |
| Shadow rollout | `candidate` | `CANDIDATE` | fail closed |
| Unavailable | `unavailable` | `UNVERIFIED` | fail closed |
| Failed / error / rejected | `failed` / `error` / `rejected` | `UNVERIFIED` | fail closed |

Unavailable, failed, and simulated backends **remain** candidate/unavailable
and **never** emit production `ATTESTED`, semantic correctness, inventory
completeness, or translator-soundness claims.

## Abuse cases

| Abuse | Response |
| --- | --- |
| Claim `ATTESTED` from simulated envelope | Authority stays non-authoritative; production/completion gates fail |
| Collapse Planner CID into Doctor/benchmark/promotion | `LineageOrderError` / distinct type slots |
| Replay another run’s envelope | `LineageReplayError` on `run_id` / root / public inputs |
| Put witness fields in public JSON | `WitnessDisclosureError` |
| Treat ZK as inventory or translator proof | `AttestationClaimPromotionError` |
| Select Groth16/Plonk without this use case | Policy gate rejects; core Planner/Doctor remain operable on public lineage alone |

## Normative references

- Plan §6.3 ZKP boundary — fixed circuit, private witness, lineage; not semantics
- Objectives PDR-G070 / task PDR-060 — invalid preimage and cross-run replay fail;
  simulated ZKP never becomes `ATTESTED`
- Module: `planner_doctor_attestation.py`
  (`ReasoningRunManifest`, `PlannerDoctorAttestation`,
  `verify_planner_doctor_attestation`, `simulated_attestation_cannot_satisfy_attested`)
