# Container stage timing diagnostic

The unchanged archived runtime executes native source/model/resource checks
under timing wrappers. Actual Source384 completes in 78.368s at the unchanged
90-second limit; initial context is 139.516s and the probe is 162.933s.
Cold index publication takes 43.570s. The numerical worker starts with 40.956s
remaining and returns in 14.525s. Initial inference's four live observations
take 3.852s, 4.231s, 4.130s and 4.208s. Later replay observations remain active.
Nested stage events must not be added twice.

This does not reproduce the preceding ordinary timeout and does not establish
its cause or a general speedup. The probe is instrumented, explicitly marked
diagnostic_only, and cannot substitute for ordinary production qualification
or a full task result. No model/provider calls for planning or coding occur.
The checkpoint produces unverified candidates, not source-qualified proofs.

Actual resources remain five CPUs and 12288 MiB; initial availability is 9320
MiB. Source pins and permitted task inputs remain unchanged, and container
removal is verified. A prior setup attempt was rejected before deployment for
hard-linked archive artifacts. Independent byte-preserving copies corrected
that staging error; the failed controller and correction are hash-referenced.
