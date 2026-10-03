# Qualification probe bulk-data lifetime

After all cold inference checks, the probe keeps only the native worker receipt
and inference key needed for export binding. It releases the raw inference JSON
and parsed candidate/source-map rows before independent warm replay. The full
immutable inference file stays available and is still downloaded and checked
against its digest and key after successful replay. No check or deadline changes.

All 37 tests execute and pass in a fresh store. Five execute the actual embedded
probe against explicit native doubles, checking object unreachability before
replay and preserving digest, model-load, authority and warm-drift rejection.
These controls do not prove native container memory savings. The subsequent
ordinary container fails earlier in initial-context replay under I/O pressure,
before the new probe lifetime boundary is reached.

The original test outputs are retained with a bulk- prefix; the relocation map
explains why their recorded command retains the original output filenames.
