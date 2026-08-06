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

## Parallelization update (2026-08-05T06:05:01Z)

Registered FVT-102..113 as separate bundle lanes in `bundles/index.json` so the supervisor can dispatch tool-family residuals in parallel after FVT-094. Tool install tasks depend on FVT-094 (completed); FVT-113 reseals after FVT-103..112. Host still reports unavailable tools and missing elevations honestly until each residual lane produces sealed installs + live certification receipts.


## Conflict surface narrow (2026-08-05T16:10Z)

Per-tool residual lanes FVT-103..112 no longer claim shared `certify_formal_verification_toolchains.py` / lock / deployment receipt write surfaces (those stay on FVT-102 fan-in and FVT-113 reseal). Predicted paths are tool-family install receipts under `data/.../managed-residuals/<lane>/` so they can run concurrent with FVT-081 (toolchain-release-candidate) and each other.
