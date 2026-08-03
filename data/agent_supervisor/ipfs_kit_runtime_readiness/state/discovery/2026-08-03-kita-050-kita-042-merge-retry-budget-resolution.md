# KITA-050 merge retry-budget resolution for KITA-042

Date: 2026-08-03
Status: repaired; validation gate passed
Source task: KITA-042
Follow-up task: KITA-050
Failure kind: merge (`submodule_merge_failed` / `submodule_target_ref_drift`)

## Root cause

KITA-042 implementation correctly landed the joined backend support matrix in
the `ipfs_kit_py` submodule (`2fe0b6d3`, based on the authoritative superproject
pin `ab1a2283` / KITA-013). Superproject merge then failed closed because the
agent-supervisor submodule integration cursor

`refs/agent-supervisor/submodule-targets/939c277cfa4941b48a7ba245/ipfs_kit_py`

was still at `50815422` (KITA-049 MCP++ repair lineage). That cursor **diverged**
from the superproject gitlink `ab1a2283` (KITA-013 bucket conformance lineage);
both share parent `c2bb963f` but are not ancestors of each other. The daemon
reports `reason=submodule_target_ref_drift` / `drift_kind=diverged` and rolls
back, so KITA-042 exhausted the merge retry budget without a content conflict.

## Repair

1. Realign the integration cursor to the authoritative superproject pin
   `ab1a2283` (compare-and-swap from `50815422`).
2. In `ipfs_kit_py`, merge the KITA-049 lineage (`50815422`) into the KITA-013
   pin (`ab1a2283`) so neither landed capability is discarded.
3. Re-apply the KITA-042 declared outputs:
   - `docs/runtime_readiness/backend_support_matrix.md`
   - `docs/runtime_readiness/backend_support_manifest.json`
   - `tests/runtime_readiness/backends/test_joined_backend_matrix.py`
4. Refresh the content-bound `auth_mcplusplus` evidence CID after the MCP++
   conformance artifact from KITA-049 entered the unified tree.
5. Retarget the KITA-042 submodule branch and parent gitlink to the unified
   commit so a subsequent ff-only submodule integration succeeds.

## Owning repository commits

| Repo | Ref / commit | Note |
| --- | --- | --- |
| `ipfs_kit_py` | `implementation/kita-050-*-submodule-ipfs_kit_py` → unified matrix landing | Owns declared outputs |
| `ipfs_kit_py` | `implementation/kita-042-*-submodule-ipfs_kit_py` retargeted to same tip | Merge path unblocked |
| superproject | `implementation/kita-050-*` pins `ipfs_kit_py` to unified tip | KITA-050 repair branch |
| superproject | `implementation/kita-042-*` pin retargeted | Source task branch |

Merge-resolver was not required: the failure was integration-ref drift, not a
semantic text conflict on declared paths.

## Validation

Passed on 2026-08-03 in the authoritative interpreter:

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
cd ipfs_kit_py && PYTHONPATH=.:../ipfs_datasets_py python3.12 -m pytest -q \
  tests/runtime_readiness/backends/test_joined_backend_matrix.py
........................                                                 [100%]
```

Simulated submodule integration (target base `ab1a2283`, ff-only of the
repaired submodule branch) also succeeded.

## Unblock

This resolution closes the repeated KITA-042 merge failure represented by
KITA-050 and supplies the evidence required to release KITA-042 from the
strategy `blocked_tasks` list.
