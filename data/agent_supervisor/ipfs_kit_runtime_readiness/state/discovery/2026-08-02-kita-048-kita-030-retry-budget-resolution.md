# KITA-048 retry-budget resolution for KITA-030

Date: 2026-08-02
Status: repaired; validation gate passed

## Root cause

`MCPServer` constructed `EventDAGStore` without importing it. Once that error
was exposed, the server also treated the persistent store as a list in the
Profile B parent and frontier paths. The minimal bundled IPFS client exposes
`ipfs_pin_rm`, whereas the legacy wrapper calls `pin_rm`.

## Repair

- Import and construct `EventDAGStore` explicitly, using an isolated temporary
  store unless `MCPPLUSPLUS_EVENT_DAG_DIR` selects durable storage.
- Read Profile B parents and the DAG frontier through `history()` and
  `frontier()` rather than list operations.
- Adapt only the minimal-client unpin compatibility seam; a backend with
  `pin_rm` is unchanged.
- Add a bootstrap regression exercising the Profile B envelope path.

## Validation command

```text
cd ipfs_kit_py && PYTHONPATH=..:../ipfs_datasets_py /usr/bin/python3.12 -m pytest -q tests/runtime_readiness/mcplusplus/test_server_bootstrap.py ipfs_kit_py/mcp_server/tests_e2e_interop.py -k 'profile_c or profile_d or all_five_profiles_smoke or mcppp_envelope'
```
