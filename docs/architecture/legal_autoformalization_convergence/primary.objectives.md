# Legal primary graph objectives

## LACP-G000 Qualify manifest-selected primary graph reconstruction

- Status: active
- Parent:
- Goal: Reconstruct the exact selected primary graph transport with complete population and endpoint accounting while preserving historical evidence and blocking semantic admission.
- Acceptance: All three native qualifications independently validate and replay; full selected-file custody and candidate reports match separately pinned baselines.

## LACP-G011 Qualify primary graph manifest and file custody

- Status: active
- Parent: LACP-G000
- Goal: Verify the raw manifest, complete selected descriptor population, all selected shard bodies and exact wrapped schemas/footer populations. Keep unselected historical shards separate.
- Acceptance: The corresponding result independently validates and replays within its preparatory scope.

## LACP-G040 Qualify lossless primary graph record decoding

- Status: active
- Parent: LACP-G000
- Goal: Preserve every raw wrapped record and exact identity, verify strict JSON/hash/parity contracts, and expose unsupported or missing identities without inferred aliases or authority.
- Acceptance: The corresponding result independently validates and replays within its preparatory scope.

## LACP-G044 Qualify complete candidate reconstruction and endpoint accounting

- Status: active
- Parent: LACP-G000
- Goal: Stream all selected nodes/edges into a separate deterministic candidate with complete physical lineage, duplicate/conflict accounting and global declared-endpoint resolution under finite storage and memory bounds.
- Acceptance: The corresponding result independently validates and replays; complete candidate bytes and census match separately pinned evidence.
