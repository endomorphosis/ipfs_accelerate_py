# Managed tool residual (deployment capability closure)

Observed_at: 2026-08-05T00:23:17Z

Deployment receipt still blocks on `supported_capability_closure` and
`required_elevations_complete`. Unavailable / managed-blocked tools:

- apalache, tlc
- coq, isabelle
- vampire, eprover
- tamarin, maude, stack
- proverif, opam
- secpal, souffle
- ergoai, temurin-jdk
- runtime-mtl-external

Missing required elevations: datalog-authorization, secpal-authorization, coq, isabelle.

Prior install/cert tasks (FVT-042..FVT-101 subset) were lease-completed without
closing these host managed gates. Operator requeued those leases
(`formal_verification_managed_tool_residual_requeue_receipt.json`) and opened
residual fan-in tasks FVT-102+ for parallel supervised repair.
