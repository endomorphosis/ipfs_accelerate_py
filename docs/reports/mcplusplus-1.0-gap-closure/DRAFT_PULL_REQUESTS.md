# MCP++ 1.0 Gap-Closure — Draft Pull Request Descriptions

| Field | Value |
| --- | --- |
| Task | `MCPP-082` |
| Goal | `MCPP-G170` |
| Interface | draft PR text per modified repository (plan §12 / PR-19) |
| Companion report | `ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md` |
| Generated (UTC) | `2026-08-16T17:06:48Z` |
| Merge policy | **Draft only** — do **not** merge from automation; do **not** force-push; preserve user overlays |
| Recommendation | **`RC`** (see implementation report §20) |

## How to use

1. Open each PR as a **draft** against the repository’s agreed base (prefer `codex/mcplusplus-1.0-gap-closure` where that branch exists; otherwise the operator-selected integration branch — never force onto dirty operator `main` with a hard reset).
2. Paste the corresponding section below as the PR body.
3. Link CI artifacts and the implementation report.
4. Leave merge to human review after MCPP-083 joined receipt if gates support it.
5. If GitHub credentials or permissions are missing for a remote, treat that as a **documented blocker**, not a silent skip of this text.

### Credential note (this generation environment)

`gh auth status` shows an authenticated `endomorphosis` account with `repo` + `workflow` scopes. Opening live draft PRs is optional and depends on pushed branches existing on each remote. **This file is the required durable output**; live PR URLs may be attached later without rewriting sealed plan artifacts.

### Bound HEADs (program worktree / forest)

| Repo | Remote | HEAD used in report |
| --- | --- | --- |
| `ipfs_accelerate_py` (program) | `https://github.com/endomorphosis/ipfs_accelerate_py` | `0515dbbf198d3a8ca40e3d3302342e41d059e302` (worktree) / forest pin `ea11293b…` |
| Nested Mcp-Plus-Plus (`mcplusplus`) | `https://github.com/endomorphosis/Mcp-Plus-Plus` | worktree gitlink `9e2ce7c8…` / forest `6965f89f…` |
| `ipfs_datasets_py` | `https://github.com/endomorphosis/ipfs_datasets_py` | worktree `d229eef0…` / forest `ac82107e…` |
| `ipfs_kit_py` | `https://github.com/endomorphosis/ipfs_kit_py` | worktree `706d3eb5…` / forest `6196017c…` |
| SwissKnife | **discovered** `https://github.com/endomorphosis/swissknife` | `afdbf885175fde34505ef05a2ea6aac5535ad03e` |

---

## PR-A — `ipfs_accelerate_py` (runtime + monorepo evidence + CI companion)

### Title

```text
draft: MCP++ 1.0 gap-closure — accelerate adapters, evidence, and CiWorkflow@1 companion
```

### Body

```markdown
## Problem

MCP++ existed as a documentation-first protocol with structural validators and
scattered runtime adapters. Coverage/production docs over-claimed. Official MCP
2026-07-28 is not initialize-based, while legacy 2024-11-05 peers remain.
Accelerate owns high-value runtime surfaces (bindings, envelope, durable,
Profile G fencing, Profile H seller, A2A reference adapter) that must be
adapted—not rewritten—under fail-closed honesty rules. Operator dirty overlays
must be preserved.

## Architecture

- Spec repo owns schemas, vectors, validators, matrices (ADR-0001).
- Runtimes own adapters only (accelerate under `mcp_server/mcplusplus/`).
- Dual MCP bindings: `mcp-binding/legacy-2024-11-05` and `mcp-binding/2026-07-28`.
- Conformance ladder: structural → canonical → cryptographic → policy-enforced
  → receipt-signed → proof-verified (ADR-0003). Schema acceptance ≠ implemented.
- Durable journals default DuckDB (Quack/DuckLake local LOAD); SQLite fallback
  via `MCPPLUSPLUS_SQL_ENGINE=sqlite` (ADR-0005 correction).
- Payment ≠ authorization; PeerID/TLS ≠ UCAN authority (KD-14).
- Profile G is neighborhood coordination / fencing, **not** BFT consensus.

## Major changes

- Baseline forest + dirty-overlay inventory under
  `docs/reports/mcplusplus-1.0-gap-closure/baseline/`.
- Runtime adapter matrix
  (`docs/reports/mcplusplus-1.0-gap-closure/runtime/adapter-matrix.md`).
- Accelerate implemented cells: Profile B envelope, G fence, H payment≠auth,
  dual bindings, DurableExecutor bind, A2A handoff (commands in matrix).
- Parent CI: `.github/workflows/mcplusplus-1.0-gap-closure.yml` (`CiWorkflow@1`).
- Implementation report + this draft PR pack (MCPP-082).
- Documentation honesty: no production-ready / 100% claims without artifacts.

## Compatibility

- Baseline MCP peers that never negotiate MCP++ continue to work.
- Dual-binding path selection fail-closed for forgery/downgrade.
- Historical artifact CIDs remain readable under recorded algorithms;
  new mints use `mcpp-jcs-v1` when claiming MCP++ 1.0 identity.
- Does not silently rewrite operator uncommitted files.

## Security

- Fail-closed authorization, policy, signature, lease, fencing, payment paths.
- Profile H negatives: payment success never grants capability.
- Profile G: reject stale fenced completion.
- Threat model / trust boundaries docs authority-class marked.
- Release artifacts may be **unsigned** when signing secrets absent (documented).

## Tests

```bash
# Parent presence gate
test -s .github/workflows/mcplusplus-1.0-gap-closure.yml

# Focused accelerate commands (implemented cells)
python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_envelope.py
python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_profile_g_fence.py
python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py
python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_durable_runtime.py
python -m pytest -q test/api/test_mcplusplus_a2a_handoff.py
cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_negatives.py
```

Broader historical `-k mcplusplus` inventory debt is recorded in baseline
receipts and is **not** silently declared green.

## Deploy / operator notes

1. Prefer isolated worktrees; never `reset --hard` the operator dirty checkout.
2. Point reviewers at
   `ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md`.
3. SQL engine: DuckDB default; optional `MCPPLUSPLUS_SQL_ENGINE=sqlite`.
4. Keep PR in **draft** until MCPP-083 joined receipt and human review.

## Remaining risks

- Remote CI green run not bound for every HEAD (gate 28 residual).
- Forest-wide Verified Execution / real ZK not admitted.
- Several profile×runtime cells remain partial/blocked outside accelerate.
- Baseline suite collection debt under broad selections.

## Explicit non-claims

Not production-ready. Not 100% coverage. Not BFT. Not “zero knowledge” without
real verifier success. Payment/PeerID are not authorization.

## Checklist

- [ ] Draft PR only (no merge from bot)
- [ ] No force-push
- [ ] User overlays intact
- [ ] Implementation report linked
- [ ] CI workflow files present
- [ ] Recommendation RC understood
```

---

## PR-B — `Mcp-Plus-Plus` (nested `ipfs_accelerate_py/mcplusplus` / protocol package)

### Title

```text
draft: MCP++ 1.0 gap-closure — dual bindings, mcpp-jcs-v1, envelope/state, CI, honest docs
```

### Body

```markdown
## Problem

Mcp-Plus-Plus was documentation-first: Profiles A–H as drafts, mostly
structural validators, initialize-shaped capability language, versioned JSON
Schema only for Profile H, and stale “100% / validation complete” docs that
are not evidence. Cross-language identity, dual MCP bindings for 2024-11-05 and
2026-07-28, portable carriers, and installable `mcpp` demo/verifier paths were
required for a release-candidate architecture.

## Architecture

- This repository remains the **canonical** home for normative specs, schemas,
  vectors, validators, matrices, and release bundles (ADR-0001).
- Profiles A–H are transport- and MCP-version-independent (KD-2).
- Dual bindings (ADR-0006): legacy initialize vs current `_meta` only.
- Canonicalization id `mcpp-jcs-v1` = RFC 8785 JCS (ADR-0002).
- Portable family: ExecutionEnvelope@1 / Result / Receipt / PortableError;
  StateRef@1 with exclusive consistency modes.
- Conformance levels prevent promoting schema green to crypto/production.
- Profile bundles (KD-17): Evidence Core, Secure Delegation, Federated Mesh,
  Commerce, Verified Execution.

## Major changes

- Binding docs + compatibility matrix under `docs/spec/bindings/`.
- Architecture overview, threat model, trust boundaries, durable/state guides
  with authority classes (MCPP-078).
- ADRs 0001–0006.
- Four-language validators and integration suites (Python/TS/Go/Rust).
- Cross-language JCS identity evidence (program
  `docs/reports/.../canonical/four-language.json`).
- Nested CI: `.github/workflows/mcplusplus-1.0.yml` (`CiWorkflow@1`) covering
  languages, vectors, crypto negatives, P2P abuse, three-peer/crash, bindings,
  scans, demo smoke, SBOM/license/manifest.
- `mcpp` CLI + `cli/verify_bundle.py` + demo compose presence gates.
- README / testing honesty rewrite (MCPP-079).
- Implementation report under `docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md`.

## Compatibility

- Legacy binding keeps 2024-11-05 initialize path available when offered.
- Current binding must not require initialize.
- Historical CIDs readable under recorded algorithms; no silent byte rewrite.
- Runtimes adapt; they must not fork a second unofficial protocol here.

## Security

- Structural validators alone do not verify UCAN signatures or policy.
- Crypto negatives and transport abuse suites are CI-encoded.
- Threat model marks fail-closed expectations without claiming full runtime
  defeat of every threat.
- Unsigned release artifacts allowed if signing credentials absent and
  documented in the release manifest.

## Tests

```bash
cd ipfs_accelerate_py/mcplusplus   # or Mcp-Plus-Plus checkout root
python -m pytest -q tests-py --maxfail=1
python -m pytest -q tests-py/integration/test_cross_language_jcs.py
python -m pytest -q tests-py/integration/test_transport_abuse.py
python -m pytest -q tests-py/integration/test_mcp_binding_compat.py
python cli/verify_bundle.py --self-test
# TS / Go / Rust
(cd tests-ts && npm test)
(cd tests-go && go test ./...)
(cd tests-rs && cargo test)
```

Baseline receipts (program tree):
`docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-{python,typescript,go,rust}.json`.

## Deploy

1. Install package editable; run `python -m mcpp doctor`.
2. CI produces `release/` artifacts (SBOM, matrix, checksums, manifest).
3. Treat nested package version as conformance pin for runtime adapters.

## Remaining risks

- Profile C/D multi-language validators remain largely structural.
- Real ZK / proof-verified path not default-on.
- Remote CI green for the exact release SHA must still be bound (gate 28).
- Demo one-command operator success should be re-affirmed for the release tag.

## Explicit non-claims

Not production admission of all profiles. Not 100% coverage. Not BFT for
Profile G. Not zero-knowledge without real verifier success.

## Checklist

- [ ] Draft only / no force-push
- [ ] Implementation report linked
- [ ] Nested workflow present
- [ ] Honesty non-claims in README retained
```

---

## PR-C — `ipfs_datasets_py` (policy + P2P + bindings adapter)

### Title

```text
draft: MCP++ 1.0 gap-closure — datasets Profile D/E adapters and dual bindings
```

### Body

```markdown
## Problem

Datasets owns workflow, policy evaluation, and heavy P2P framing paths used by
MCP++ Federated Mesh / Secure Delegation. Gap-closure required fail-closed
policy evaluation (missing evaluator ⇒ deny), versioned P2P framing with abuse
reuse, and dual MCP bindings shared with accelerate—without inventing crypto
verifiers that belong in kit/spec, and without discarding operator dirty
logic/UI-IR overlays.

## Architecture

- Runtime-only adapters under `ipfs_datasets_py` (ADR-0001).
- Profile D temporal deontic evaluation fail-closed.
- Profile E `mcp+p2p` framing + protocol IDs; transport identity ≠ authority.
- Dual bindings implemented jointly with accelerate tests.
- DurableExecutor and A2A adapters are **out of scope / blocked** for datasets
  in the runtime matrix (honest disposition).

## Major changes

- Policy evaluator surface + unit tests
  (`tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py`).
- P2P framing tests
  (`tests/unit/mcp_server/test_mcplusplus_p2p_framing.py`).
- Bindings adapter covered by accelerate joint suite.
- Partial Profile F/G/H modules retained with honest partial disposition.

## Compatibility

- Consumes CID envelopes / vectors from Mcp-Plus-Plus; does not re-own schemas.
- Operator uncommitted P2P/logic files preserved (worktree isolation).

## Security

- Missing policy evaluator denies.
- P2P abuse vectors reused rather than soft-allowing oversize/replay/flood.
- No claim that PeerID grants UCAN capabilities.

## Tests

```bash
cd ipfs_datasets_py
python -m pytest -q tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py
python -m pytest -q tests/unit/mcp_server/test_mcplusplus_p2p_framing.py
# joint bindings (from accelerate root)
python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py
```

Note: baseline receipt recorded exit 5 for broad `-k mcplusplus` selection;
focused commands above are the admission evidence for implemented cells.

## Deploy

Ship as library adapters consumed by MCP servers; no second protocol package.

## Remaining risks

- Profiles A/C Durable/A2A blocked or partial.
- Broader suite selection still needs inventory cleanup.
- Remote CI for datasets-only may lag monorepo companion workflow.

## Explicit non-claims

Not full Profile A–H implementation. Not production-ready mesh admission.

## Checklist

- [ ] Draft only
- [ ] Overlays preserved
- [ ] Matrix dispositions not upgraded without new commands
```

---

## PR-D — `ipfs_kit_py` (UCAN cryptographic verifier + coordination bind)

### Title

```text
draft: MCP++ 1.0 gap-closure — kit UCANVerifier and MCP++ readiness adapters
```

### Body

```markdown
## Problem

Profile C cryptographic delegation must be **real** somewhere in the forest.
Kit already hosted UCAN readiness work; gap-closure required an honest
`UCANVerifier` with adversarial negatives (tamper, unsigned, alg-none, wrong
key, audience, expiry) and partial binds for policy/P2P/G/H without claiming
dual bindings, DurableExecutor, or A2A that were not landed.

## Architecture

- Kit owns crypto-strong C verification for the forest (matrix **implemented**).
- Spec validators may remain structural; kit must not be papered over by them.
- Profile G transport/coordination adapted; authoritative stale-fence gate
  remains recorded on accelerate.
- Payment paths partial; payment≠auth adversarial suite owned primarily by
  accelerate/SwissKnife (MCPP-072).

## Major changes

- `UCANVerifier` + revocation/delegation surfaces under kit MCP++ modules.
- Readiness tests:
  `tests/runtime_readiness/mcplusplus/test_ucan_verifier.py`.
- Partial Profile D/E/G/H surfaces and transport security parity tests.
- Baseline receipt honesty: broad collection may still fail; focused command
  is the implemented-cell evidence.

## Compatibility

- Artifacts remain CID-native; no silent CID rewrite.
- Does not invent SwissKnife origin URLs or rewrite remotes.

## Security

- Cryptographic verify with negatives is mandatory for C admission claims.
- Transport security parity tests present; identity ≠ capability.

## Tests

```bash
cd ipfs_kit_py
python -m pytest -q tests/runtime_readiness/mcplusplus/test_ucan_verifier.py
# supplementary partial surfaces
python -m pytest -q tests/runtime_readiness/mcplusplus/test_transport_security_parity.py
python -m pytest -q tests/test_profile_g_transport.py
```

## Deploy

Consume kit verifier from servers that need cryptographic C; do not claim
four-language validator crypto from this PR alone.

## Remaining risks

- Dual bindings / Durable / A2A **blocked** on kit.
- Baseline broad pytest selection collection debt.
- Profile F ZK not proof-verified.

## Explicit non-claims

Not forest-wide production crypto for every runtime. Not BFT.

## Checklist

- [ ] Draft only
- [ ] Implemented cells keep commands
- [ ] No coverage-threshold lowering
```

---

## PR-E — SwissKnife (discovered origin `endomorphosis/swissknife`)

### Title

```text
draft: MCP++ 1.0 gap-closure — SwissKnife A2A + Profile H adapters (preserve overlays)
```

### Body

```markdown
## Problem

SwissKnife carries rich MCP++ TypeScript adapters (IDL, envelope, UCAN, policy,
P2P, event DAG, Profile G connector) and must interoperate with A2A via a
**URI** extension without inventing a competing task lifecycle. Profile H
seller paths must fail closed when unconfigured and must never treat payment
as authorization. The origin remote must remain the **discovered**
`https://github.com/endomorphosis/swissknife` (upstream anon-kode retained).
Dirty overlay `test-results/.../svd-132.json` must not be discarded.

## Architecture

- Runtime adapters only; normative specs stay in Mcp-Plus-Plus.
- A2A extension URI: `https://mcplusplus.io/extensions/execution/v1`.
- Profile H adapter: HTTP and libp2p readiness; header redaction; fail-closed
  unconfigured seller (MCPP-072).
- Local SwissKnife PASS matrices are **not** forest-wide four-language admission
  (keep A–G as partial unless a gap-closure receipt elevates a cell).

## Major changes

- `SwissKnifeA2AAdapter@1` + tests (`test/mcp-plus-plus/a2a-adapter.test.ts`).
- Profile H adapter script/tests (`profile-h-adapter.test.ts`).
- Receipts in accelerate program tree:
  `docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md`,
  `runtime/profile-h-adapters.md`.
- Preserve dirty operator overlay files.

## Compatibility

- Does not rewrite origin to a guessed URL.
- Works with dual-binding peers when accelerate/datasets advertise both paths;
  SwissKnife dual-binding adapter remains **blocked** in the matrix.

## Security

- Payment ≠ authorization.
- Fail-closed unconfigured seller.
- Header redaction on payment surfaces.
- UCAN library path partial; baseline suite may still fail without crypto mocks.

## Tests

```bash
cd /path/to/swissknife   # discovered origin checkout
npm run test:run -- test/mcp-plus-plus/a2a-adapter.test.ts
npm run test:run -- test/mcp-plus-plus/profile-h-adapter.test.ts
# supplementary partial surfaces (not forest admission alone)
npm run test:run -- test/mcp-plus-plus/mcp-idl.test.ts
```

## Deploy

Ship as SwissKnife service adapters; coordinate release with accelerate
implementation report RC recommendation.

## Remaining risks

- Baseline vitest debt (crypto mock) for full `test/mcp-plus-plus` suite.
- Dual bindings / Durable blocked.
- Self-PASS ≠ four-language admission.

## Explicit non-claims

Not production-ready multi-agent mesh by SwissKnife tests alone.
Not BFT. Not zero-knowledge by default.

## Checklist

- [ ] Origin remains discovered URL
- [ ] Dirty overlay preserved
- [ ] Draft only / no force-push
- [ ] Matrix commands retained
```

---

## PR stack / dependency order (review guidance)

Aligned with sealed plan PR-00…PR-19 (summary for reviewers):

| Order | Focus | Primary repos |
| --- | --- | --- |
| 1 | Baseline + overlays + forest | accelerate evidence |
| 2 | ADRs + architecture honesty | Mcp-Plus-Plus |
| 3 | Bindings + JCS + envelope/state | Mcp-Plus-Plus (+ accelerate/datasets adapters) |
| 4 | Crypto / policy / durable / A2A / P2P / G / H | kit, datasets, accelerate, SwissKnife |
| 5 | CLI/demo + CI + **this report / draft PRs** | all |
| 6 | Joined RC receipt (MCPP-083) | accelerate evidence only |

Suggested Graphite/stack policy if used: open **draft** PRs bottom-up; do not merge until residual gates and human review complete.

---

## Opening drafts with `gh` (optional operator commands)

Only after branches with the intended commits exist on each remote:

```bash
# Example — accelerate monorepo
gh pr create --repo endomorphosis/ipfs_accelerate_py \
  --draft \
  --base codex/mcplusplus-1.0-gap-closure \
  --head <implementation-branch> \
  --title "draft: MCP++ 1.0 gap-closure — accelerate adapters, evidence, and CiWorkflow@1 companion" \
  --body-file /path/to/this-section-extracted.md

# Never:
#   git push --force
#   gh pr merge ...
```

If `gh` lacks permission or the base branch is missing, record the blocker next to the repo name and keep this document as the PR body source of truth.

---

## Acceptance (MCPP-082 PR text)

| Criterion | Result |
| --- | --- |
| Draft PR text per modified repository | **yes** — accelerate, Mcp-Plus-Plus, datasets, kit, SwissKnife |
| Covers problem, architecture, major changes, compatibility, security, tests, deploy, remaining risks | **yes** — each body |
| Merge / force-push forbidden | **yes** — stated |
| User overlays intact | **yes** — stated; SwissKnife origin discovered |
| Recommendation consistent with report | **RC** |

**End of draft pull-request pack for MCPP-082.**
