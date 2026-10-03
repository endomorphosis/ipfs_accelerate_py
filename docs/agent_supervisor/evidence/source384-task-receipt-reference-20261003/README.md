# Source384 task receipt references

The full supervisor reached indexed context and planning, then failed because its 78,563-byte Source384 receipt exceeded the task bundle's 32,768-byte inline limit. The canonical receipt producer permits 131,072 bytes. Increasing the inline limit would still duplicate the full receipt for each of up to sixteen tasks.

The writer now emits nominations@3: a closed, at-most-4,096-byte reference to the existing external receipt.json, binding its full SHA-256 and byte count. The bundle and complete receipt limits remain 131,072 bytes. The selected reference is resolved with the existing canonical, regular-file, no-follow, singleton, stable-byte reader. The complete receipt then reaches the unchanged current-source/model validator. Historical observation resolves exact bytes without granting freshness or successor availability. Legacy @1 and @2 readers remain supported.

The final runtime owner passed 56 distinct controls: 44 transport/legacy controls, one actual native owner/publication/STOP/historical-consumer lifecycle, and eleven worker-snapshot/warm-rebind controls. Runs contain 44, 45 and 55 overlapping cases, not 144 distinct cases. Native numerical inference is not executed. The real-checkpoint test has an assertion-only update for the new schema and full historical selection; that test was not rerun in this slice.

The retained native receipt transport replay changes only repository/output for an authored temporary filesystem. Its original 78,563 bytes become 78,734 bytes; every other field, including all 220 source hashes and inventory rows, is unchanged. Sixteen task references occupy 5,925 bytes. A recording validator isolates transport from model replay. This neither renews the old producer pins nor qualifies the relocated receipt as current model/source evidence. No task/model/CAS bodies or auth are included.

This fixes a packaging contract. It does not qualify the remaining full-supervisor deadline, repair correctness, benchmark reward, or automatic successor preparation. The failed full run is retained separately. No deadlines, scheduling policy, model architecture or weights changed.
