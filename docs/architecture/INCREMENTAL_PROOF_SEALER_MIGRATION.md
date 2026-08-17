# Incremental Proof Sealer Migration Guide

Existing proof receipts and caches migrate by explicit classification.
There is **no assurance upgrade**.

## Disposition vocabulary

Each legacy object is one of:

- **accept** — current schema, current allowlist, freshly verified
- **adapt** — rewrite envelope only; keep the same evidence class
- **reverify** — re-check cryptography or public inputs before reuse
- **reject** — unknown, stale, simulated-as-real, or unallowlisted
- **simulated** — plumbing-only; never a production seal

Evidence classes stay **integrity-only**, **signed receipt**, or
**direct execution**. A signed receipt is not converted into direct
execution. Simulated artifacts cannot become integrity-only by rename.

## Gates that survive migration

- **verification-key allowlist** membership is re-checked
- **proof verification before cache admission** still applies
- a **schema change** or **canonicalization change** forces a
  **full checkpoint**
- **staged migration** with an explicit **rollback** path
- an **unknown legacy proof system** is rejected, not guessed
- a **sensitive witness** is never copied into a public cache
- a **test-only key** cannot enter production reuse
