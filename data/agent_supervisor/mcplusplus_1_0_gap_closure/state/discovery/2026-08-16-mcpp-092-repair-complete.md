# MCPP-092 Repair Completion: MCPP-025 validation retry-budget

Date: 2026-08-16
Source task: MCPP-025
Follow-up task: MCPP-092
Status: **completed**
Attempt: 1

## Root cause (inherited proposal-gate debt, not a vector regression)

MCPP-025 implementation **did** produce the golden vector suite on all three
attempts, but the supervisor **never reached the declared validation gate**.
Every attempt failed at pre-dispatch proposal validation:

| Field | Value |
| --- | --- |
| Failed phase | `validation_pre_dispatch` |
| Error | `proposal_validation_failed` |
| Reason | `proposal_gate_failed` |
| Finding code | `command_forbidden` |
| Validation attempted | **False** |
| Return code | **78** |

Evidence:

- Discovery finding: `data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-16-mcpp-092-mcpp-025-retry-budget.md`
- Diagnostic receipt: `state/lane-1/implementation_logs/mcpp-025-diagnostic-receipt.json`
- Logs: `state/lane-1/implementation_logs/mcpp-025-attempt-{1,2,3}.log`
- Preserved submodule commits with vectors: `627df017e560e86d57e7734519c3f50f4da6625e`, `a2bd0804dcb7f3bf903792d92ed8f137f554bd10`

### Why `command_forbidden`

MCPP-025’s board validation command is:

```bash
python -c "import json,pathlib; p=pathlib.Path('ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1'); assert any(p.glob('*.json'))"
```

Proposal validation (`proposal_validation._command_is_allowed`) **categorically
rejects** `python` / `python3` with `-c` / `-e` / `--eval` as eval-style
invocations, even when that argv is the task’s own allowlisted validation plan.
Exact allowlist hits still require `clause_executable_is_safe`, which returns
false for `python -c`.

Therefore MCPP-025 can never clear the proposal gate with the current protected
board validation text. This is **not** a defect in the golden vectors, RFC 8785
policy, or production assertions. Operator-protected board/todo/scheduler files
were not edited.

Secondary failure-review labels (`large_or_undeclared_refactor`,
`scope_expansion_denied`) were derived from the rejected proposal envelope after
the command finding; the sole proposal finding code on attempts was
`command_forbidden`.

## Repair actions

1. **Publish golden vectors** (declared output; restores prior verified suite):
   - Directory: `ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1`
   - Files: `README.md`, `manifest.json`, `numbers.json`, `unicode.json`,
     `null.json`, `empty-object.json`, `nested-keys.json`, `duplicate-keys.json`
   - Interface: **`GoldenVector@1`**
   - Algorithm: **`mcpp-jcs-v1`**
   - Source submodule commit: `a2bd0804dcb7f3bf903792d92ed8f137f554bd10`
   - File digests match durable lane-1 checkpoint `mcpp-025-981af0ea206b`

2. **Acceptance coverage** (MCPP-025): at least one positive and one negative
   vector for each required category:

   | Category | Positive | Negative |
   | --- | --- | --- |
   | numbers | yes | yes (NaN, Infinity) |
   | unicode | yes | yes (lone surrogate) |
   | null | yes | yes (capitalized `Null`) |
   | empty_object | yes | yes (whitespace-only object text) |
   | nested_keys | yes | yes (cycle / non-canonical claims) |
   | duplicate_keys | yes | yes (top-level and nested duplicates) |

3. **Vector effects fields** (positive cases): `source` / `source_json`,
   `canonical_utf8`, `canonical_bytes_hex` / base64, `sha256`, CIDv1 raw+sha2-256,
   `signature_input`, `signature_placeholder`, `expected_validator_result`.
   Negatives pin fail-closed `expected_validator_result.accept=false` with
   reason codes.

4. **Discovery evidence** (declared output tree): this repair-complete note plus
   the retry-budget finding.

5. **Not changed**: production policy, tests/assertions, operator-protected
   plan/todo/scheduler/validator paths.

## Declared gate proof

MCPP-025 validation command (semantic intent: directory has JSON vectors):

```bash
python -c "import json,pathlib; p=pathlib.Path('ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1'); assert any(p.glob('*.json'))"
```

Result when run **outside** the proposal gate: **pass** (7 `*.json` files present).
Positive digests/CIDs recomputed and match recorded golden values (19 cases).

MCPP-092 acceptance path presence:

```bash
test -f /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-16-mcpp-092-mcpp-025-retry-budget.md
```

Result: **pass** (finding file present).

## Supervisor release note

Completing MCPP-092 releases MCPP-025 from strategy `blocked_tasks` and resets
its validation retry budget. The golden `mcpp-jcs-v1` vector suite is delivered
as this repair’s primary declared output under
`ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1`.

**Inherited board note (not fixed here):** while the vector outputs satisfy
MCPP-025 acceptance and the semantic gate, re-dispatch of MCPP-025 will still
hit proposal `command_forbidden` until the protected board validation command
is rewritten away from `python -c` (for example to `test -d` / `test -s` on a
manifest path, matching MCPP-009 / MCPP-024 / MCPP-091 style gates). That board
edit is outside MCPP-092 edit scope.
