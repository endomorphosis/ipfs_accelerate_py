# MCPP-057 — SwissKnife A2A execution extension adapter

| Field | Value |
| --- | --- |
| Task | `MCPP-057` |
| Title | Adapt SwissKnife to the A2A execution extension |
| Track | `a2a-swissknife` |
| Goal | `MCPP-G100` |
| Interface | `SwissKnifeA2AAdapter@1` |
| Depends on | `MCPP-056` (`A2ATaskAdapter@1` reference adapter) |
| Status | **implemented** |
| Recorded at (UTC) | `2026-08-16T05:49:45Z` |

## 1. Bound checkout (MCPP-001 / forest)

Checkout discovery is fail-closed: this receipt records the path from
`docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json`
(`operator_checkouts.swissknife`). No URL was invented.

| Field | Value |
| --- | --- |
| Path | `/home/barberb/lift_coding/swissknife` |
| Present at implementation | **yes** (missing checkout would block this task; not faked) |
| Nested commit SHA (HEAD) | `afdbf885175fde34505ef05a2ea6aac5535ad03e` |
| Branch | `main` |
| Origin (discovered) | `https://github.com/endomorphosis/swissknife` |
| Forest baseline HEAD match | yes (`repository-forest.json` `operator_checkouts.swissknife.head`) |

Pre-existing dirty path left untouched (MCPP-001 preserve rule):

- `test-results/virtual-desktop-ipfs-mcp-orb/svd-132.json`

## 2. Preconditions verified

| Precondition | Evidence |
| --- | --- |
| A2A execution extension specified | `ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md` |
| Extension URI confirmed | `https://mcplusplus.io/extensions/execution/v1` (MCPP-010 / ADR-0006) |
| Reference adapter exists (MCPP-056) | `ipfs_accelerate_py/mcp_server/mcplusplus/a2a_adapter.py` (`A2ATaskAdapter@1`) |
| Reference handoff tests exist | `test/api/test_mcplusplus_a2a_handoff.py` |
| SwissKnife checkout bound | path above exists and matches forest discovery |

## 3. SwissKnife files changed

Edits are confined to the bound SwissKnife checkout (not this program worktree).
Paths below are relative to `/home/barberb/lift_coding/swissknife`.

| Path | Action | Role |
| --- | --- | --- |
| `src/services/mcp/mcp-plus-plus-a2a.ts` | **added** | `SwissKnifeA2AAdapter@1` — Agent Card equivalent, activation, two-agent handoff, cancel, terminal evidence, fail-closed validators |
| `test/mcp-plus-plus/a2a-adapter.test.ts` | **added** | Handoff + Agent Card + fail-closed profile/URI/activation tests |

Working-tree disposition at receipt time (relative to nested HEAD
`afdbf885175fde34505ef05a2ea6aac5535ad03e`):

```text
?? src/services/mcp/mcp-plus-plus-a2a.ts
?? test/mcp-plus-plus/a2a-adapter.test.ts
```

No other SwissKnife sources were modified for MCPP-057. The pre-existing
`test-results/.../svd-132.json` dirty overlay was not touched.

## 4. Adapter behaviour (SwissKnifeA2AAdapter@1)

Aligns with the Python reference adapter (`A2ATaskAdapter@1`) and
`a2a-extension.md`:

| Capability | Behaviour |
| --- | --- |
| Extension URI (wire) | `https://mcplusplus.io/extensions/execution/v1` only |
| Working alias | `io.mcplusplus.execution@1` allowed only inside `params.alias`, never as wire URI |
| Agent Card equivalent | `A2AAgent.agentCard()` advertises `capabilities.extensions[]` with confirmed URI + MCP++ params (profiles, envelope/receipt/state schemas, bindings, interface CIDs, canonicalization) |
| Activation | Client must pass the confirmed URI in `A2A-Extensions` (header string or list); reverse-DNS-only fails closed |
| Handoff | Two independently instantiated agents; client discovers card; server completes Task with namespaced envelope/result/receipt/event CIDs |
| Public lifecycle | Official A2A `TaskState` values only (`submitted`, `working`, `completed`, `failed`, `canceled`, …) |
| Cancel | Durable cancel journal + Event DAG `task.canceled` records |
| Fail-closed | Missing activation, malformed extension URI, unsupported profile letter, requested profiles not a subset of advertisement |

Runtime metadata on the Agent Card includes `runtime=swissknife` and
`adapter=SwissKnifeA2AAdapter@1` under the extension namespace prefix.

## 5. Handoff command

From the bound SwissKnife checkout root
(`/home/barberb/lift_coding/swissknife`):

```bash
npm run test:run -- test/mcp-plus-plus/a2a-adapter.test.ts
```

Equivalent vitest invocation:

```bash
npx vitest run --config build-tools/configs/vitest.config.ts test/mcp-plus-plus/a2a-adapter.test.ts
```

### Observed result (implementation run)

```text
✓ test/mcp-plus-plus/a2a-adapter.test.ts (9 tests) 10ms
Test Files  1 passed (1)
     Tests  9 passed (9)
```

Covered cases:

1. Interface constants / URI pins  
2. Independent agent instances (distinct Event DAGs)  
3. Agent Card presents confirmed extension URI  
4. Two-agent evidence-bearing handoff completes (`completed` + portable terminal evidence)  
5. `A2A-Extensions` header string activation  
6. Missing activation fail-closed (`A2A_EXTENSION_NOT_ACTIVATED`)  
7. Reverse-DNS alias rejected on wire  
8. Unsupported / non-subset profiles fail closed  
9. Cancel writes Event DAG + durable cancel records  

## 6. Program-tree outputs

| Path | Role |
| --- | --- |
| `docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md` | This receipt (only declared gap-closure output for MCPP-057) |

## 7. Acceptance checklist

| Criterion | Result |
| --- | --- |
| Receipt lists SwissKnife files changed | **yes** — §3 |
| Receipt lists the handoff command | **yes** — §5 |
| Nested commit SHA recorded | **yes** — `afdbf885175fde34505ef05a2ea6aac5535ad03e` |
| Missing checkout blocked, not faked | **N/A (checkout present)** — path verified against forest; would block if absent |
| SwissKnife presents extension on Agent Card equivalent | **yes** — `agentCard()` + tests |
| Handoff test completes | **yes** — 9/9 green |
| Reference adapter exists | **yes** — MCPP-056 artifacts present |

## 8. Validation (gap-closure gate)

```bash
test -s docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md
```

Expected: exit 0, non-empty receipt file.

## 9. Notes for downstream tasks

- Nested SwissKnife changes remain uncommitted in the bound checkout until the
  SwissKnife maintainers / program merge path lands them; this receipt is the
  gap-closure evidence artifact for MCPP-057.
- Runtime matrix task (e.g. MCPP-076) may mark SwissKnife A2A disposition as
  **implemented** with the handoff command in §5.
- Do not treat `io.mcplusplus.execution@1` as a wire substitute for the HTTPS URI.
