# PGIR-208 source-chain acceptance

This directory freezes the successor source chain, not a learned corpus or an
execution authorization. The verifier seals the fourteen PGIR-200 through
PGIR-202 JSON payloads and their historical inputs by exact raw SHA-256 and
CIDv1 identities. It also replays the 21 exact-revision metadata citations
when `--network` is requested.

The result is permanently fail-closed: 7,173 candidates, zero rights-admitted
or materialized rows, a passing empty leakage audit, and the thirteen named
holdouts all remain permanent no-gos. PGIR-205 is not authorized by this
acceptance artifact. A missing published submodule ref is reported as the
typed `unpublished_ref_portability_blocker`, never treated as a successful
portable checkout.
