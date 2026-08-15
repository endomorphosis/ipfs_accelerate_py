# Incremental Proof Sealer Trust Model

This document describes measured executable behavior of IncrementalProofSealer.
It does not upgrade receipts, manifest aggregation, or integrity commitments.

## Proof classes and nonclaims

An integrity commitment binds exact bytes. An integrity commitment does not establish correct execution.

A signed execution receipt is a trusted signer assertion. Acceptance requires signature verification against the current allowlist. A receipt-aggregation zk proof commits to child receipt identities; it does not prove the underlying tests ran.

A direct execution proof is a declared deterministic computation for one proof unit. An incremental or recursive commit seal is accepted only against an accepted parent.

## Public and private inputs

Public inputs are the cache-key and statement fields that verifiers recompute.
Private inputs and any sensitive witness stay off the public record.
Child signatures are not verified inside the circuit. Test execution is not directly proven.

## Aggregation and recursion

Current production aggregation is manifest aggregation, not recursive proof verification. Recursion is admitted only after a successful backend capability probe.

## Setup, keys, and unknown systems

Record the trusted setup origin. Test-only keys cannot enter a production allowlist. Verifiers consume content-addressed verification keys from an allowlist. No proof key is silently generated. Unknown proof systems are rejected. No arbitrary circuit or executable is accepted as a proof.

## Cache, checkpoints, and durability

Proofs are verified before cache admission. A canonicalization change requires a full checkpoint. A circuit change requires a full checkpoint. A verification-key change requires a full checkpoint. Cache corruption requires a full checkpoint.

Publication uses compare-and-swap. Recovery is WAL-driven. An ambiguous external prover outcome is not success.

## Remaining work before production use

Remaining work before production use includes a real prover backend, recursive verification where claimed, measured (not only estimated) benchmark evidence, and operational key-ceremony documentation.
