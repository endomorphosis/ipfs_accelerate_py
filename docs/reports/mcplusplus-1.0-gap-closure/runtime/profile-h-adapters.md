# MCPP-072 — Profile H negative tests and SwissKnife/accelerate adapters

| Field | Value |
| --- | --- |
| Task | `MCPP-072` |
| Title | Add Profile H negative tests and SwissKnife/accelerate adapters |
| Track | `profile-h-adapters` |
| Goal | `MCPP-G140` |
| Interface | `ProfileHNegativeVector@1` |
| Depends on | `MCPP-071` (hardened quotes, settlement, entitlements, refunds, replay) |
| Status | **implemented** |
| Recorded at (UTC) | `2026-08-16T06:42:58Z` |

## 1. Bound checkouts

Checkout discovery is fail-closed against
`docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json`.

| Runtime | Path | Present | Nested HEAD | Branch |
| --- | --- | --- | --- | --- |
| SwissKnife | `/home/barberb/lift_coding/swissknife` | **yes** | `afdbf885175fde34505ef05a2ea6aac5535ad03e` | `main` |
| Program tree (accelerate) | this worktree | **yes** | (program forest) | gap-closure lane |

Origin discovered for SwissKnife: `https://github.com/endomorphosis/swissknife`.
Forest baseline HEAD match: yes (`operator_checkouts.swissknife.head`).

## 2. Preconditions verified

| Precondition | Evidence |
| --- | --- |
| PaymentAuthorizationBoundary@1 | `ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md` §2 |
| Hardened settlement validators (MCPP-071) | `ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_h.py` |
| Profile H schemas 1.0 | `ipfs_accelerate_py/mcplusplus/schemas/profile-h/1.0/` |
| Valid + transport vectors | `conformance/vectors/profile_h_{artifacts,transport}_valid.json` |
| Accelerate seller adapter | `ipfs_accelerate_py/mcp_server/mcplusplus/profile_h.py` |
| SwissKnife Profile H adapter | bound checkout `scripts/mcpplusplus-profile-h-adapter.cjs` |

## 3. SwissKnife files (adapter surface)

Paths are relative to `/home/barberb/lift_coding/swissknife`. These files are
already present on the bound checkout HEAD; MCPP-072 does not invent a second
adapter and does not touch the program-protected architecture files.

| Path | Role |
| --- | --- |
| `scripts/mcpplusplus-profile-h-adapter.cjs` | Profile H client/control adapter: readiness (HTTP **and** libp2p required), local-test vs facilitator labeling, payment-safe header forward, control-plane method map |
| `test/mcp-plus-plus/profile-h-adapter.test.ts` | Adapter unit/integration tests (readiness, fail-closed unconfigured seller, transport advertisement, header redaction) |
| `docs/mcpplusplus-profile-h-adapter-configuration.md` | Operator configuration for service-specific endpoints |
| `scripts/mcpplusplus-profile-e-http.cjs` | Profile E initialize advertisement of `mcp++/x402-payments` (paired capability flag) |

### SwissKnife handoff command

From `/home/barberb/lift_coding/swissknife`:

```bash
npm run test:run -- test/mcp-plus-plus/profile-h-adapter.test.ts
```

Equivalent:

```bash
npx vitest run --config build-tools/configs/vitest.config.ts test/mcp-plus-plus/profile-h-adapter.test.ts
```

### Observed result (implementation run)

```text
✓ test/mcp-plus-plus/profile-h-adapter.test.ts (8 tests) 11ms
Test Files  1 passed (1)
     Tests  8 passed (8)
```

Covered adapter cases include: service-specific endpoint resolution; unconfigured
seller typed-unavailable; durable HTTP/**libp2p** readiness and honest
`upstreamX402HttpConformance` labeling; capability advertisement only when
requested; payment-safe header forward (no `authorization` leakage); path-prefix
preservation; readiness TTL; ambiguous/non-payment path rejection.

## 4. Accelerate files (adapter surface)

Paths are relative to the program repository root (this worktree).

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/mcp_server/mcplusplus/profile_h.py` | `PaidAcceleratorService` — transport-neutral seller; settlement fence before durable work; HTTP + libp2p entrypoints; cancels/handoffs require independent authorization |
| `tests/mcplusplus_profile_h/test_paid_accelerator.py` | Runtime proof: protected work starts only after payment; retry idempotency; HTTP/libp2p profile surface |
| `tests/mcplusplus_profile_h/conftest.py` | Service fixtures (`PaidAcceleratorService`, facilitator stubs) |
| `ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_h.py` | Shared codec + settlement/authorization boundary validators used by negatives |
| `ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md` | Normative PaymentAuthorizationBoundary@1 and transport non-substitution |

### Accelerate handoff / gate commands

Program gate for this task (authoritative validation):

```bash
cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_negatives.py
```

Accelerate seller runtime suite (supplementary; requires `mcplusplus_profile_h` packaging):

```bash
python -m pytest -q tests/mcplusplus_profile_h/test_paid_accelerator.py
```

Codec/settlement suite from MCPP-070/071:

```bash
cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_codec.py
```

### Observed result (authoritative gate)

```text
47 passed in 0.07s
```

## 5. Negative suite (ProfileHNegativeVector@1)

Declared program output:

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_h_negatives.py` | Adversarial negatives + transport-split proofs |

| Negative | Fail-closed code(s) | Mechanism |
| --- | --- | --- |
| paid-but-unauthorized | `H_PAYMENT_POLICY_DENIED` | `AccessReceipt.decision=allow` with null layer-A decision CIDs rejected |
| authorized-but-unpaid | `H_PAYMENT_REQUIRED` | `AccessReceipt.decision=allow` with null commercial evidence / result rejected |
| replay | `H_PAYMENT_REPLAY` | `validate_replay` on consumed commitment |
| price mismatch | `H_AMOUNT_MISMATCH` / `H_REQUEST_MISMATCH` | `validate_price_version_binding` amount / catalogVersion |
| wrong recipient | `H_AMOUNT_MISMATCH` | quote `payTo` ≠ capability `payee` |
| duplicate settlement | `H_PAYMENT_REPLAY` | `validate_idempotent_settlement` / `validate_idempotent_entitlement` conflicts |
| expired quote | `H_QUOTE_EXPIRED` | `validate_quote_not_expired_for_settlement` / settlement after expiry |
| refund-after-consumed | `H_ENTITLEMENT_EXHAUSTED` | `validate_refund_eligibility` with full consumption |
| forged settlement | `H_AMOUNT_MISMATCH` / `H_VERIFICATION_FAILED` / `H_REQUEST_MISMATCH` | amount/network forge; rejected verification; verificationCid mismatch |

Deny-shaped AccessReceipts (paid-but-unauthorized with commercial evidence and
null result; authorized-but-unpaid with null commercial evidence) remain
structurally valid — payment success never becomes `decision=allow` without both
layers.

## 6. Transport split (x402 HTTP vs libp2p / artifact)

Normative non-substitution: `x402-payments.md` §1 and §7. Libp2p carriage of
Profile H objects is **not** upstream x402 HTTP conformance.

| Carriage | Object family | Codec API | Tests |
| --- | --- | --- | --- |
| **x402 HTTP** | `PaymentRequired`, `PaymentPayload`, `SettlementResponse` | `encode_x402_header` / `decode_x402_header` | transport vector round-trip; v1 rejected; malformed base64 fails closed; Profile H artifacts rejected as x402 kinds |
| **libp2p / Profile E artifact** | `PaymentQuote`, `SettlementReceipt`, `AccessReceipt`, … | `validate_profile_h_artifact` | artifact kinds exclude x402 header objects; x402 objects rejected as Profile H artifacts |

SwissKnife readiness additionally requires **both** `http` and `libp2p` in
`profile.transports` and honest `upstreamX402HttpConformance` labeling
(`local-test` ⇒ false; `facilitator` ⇒ true). Accelerate exposes separate
`handle_http` / `handle_libp2p` / `handle_profile_h_libp2p` paths that share the
same settlement fence and authorization checks.

## 7. Program-tree outputs

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_h_negatives.py` | Negative + transport-split suite |
| `docs/reports/mcplusplus-1.0-gap-closure/runtime/profile-h-adapters.md` | This receipt (files + commands) |

## 8. Acceptance checklist

| Criterion | Result |
| --- | --- |
| All listed negatives fail closed | **yes** — 47/47 gate tests green |
| Adapter receipt lists files | **yes** — §3 SwissKnife, §4 accelerate |
| Adapter receipt lists commands | **yes** — §3 handoff, §4 gate/handoff |
| Transport split tested | **yes** — §6 + `TestTransportSplit` |
| Nested SwissKnife commit recorded | **yes** — `afdbf885175fde34505ef05a2ea6aac5535ad03e` |
| Payment ≠ authorization preserved | **yes** — AccessReceipt allow shapes; accelerate dispatch fence documented |
| Missing SwissKnife checkout would block | **N/A (present)** — path verified against forest |

## 9. Validation (gap-closure gate)

```bash
cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_negatives.py
```

Expected: exit 0, all negatives and transport-split cases pass.

## 10. Notes for downstream tasks

- SwissKnife adapter sources remain on the bound checkout HEAD; this receipt is
  the gap-closure evidence artifact for MCPP-072 (files + commands + observed runs).
- Runtime matrix tasks may mark SwissKnife and accelerate Profile H disposition
  using the commands in §3–§4 and the negative suite in §5.
- A settled quote remains commercial evidence only; it is not a capability token
  (`PaymentAuthorizationBoundary@1`).
