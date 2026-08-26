# PGIR-206 successor R1-R6 typed `not_run` evidence

This directory is the complete `RESULT(PGIR-206)` evidence package for the
successor campaign.  PGIR-205 sealed a `no_go` campaign input root and did not
authorize descendant execution.  PGIR-206 therefore did not request a GPU or
other execution lease, start training, create weights or checkpoints, invoke
the proof loop, open hidden labels, evaluate a model, or attempt reducer CAS.
Those absences are represented as typed evidence; they are not missing work
and must not be interpreted as zero-valued measurements.

The closed planned population is still recorded in full:

- R1 uses deterministic seed `0`.
- R2-R6 each use seeds `32`, `33`, and `34`.
- All 16 arm/seed keys have all 12 declared metrics, for 192 metric cells.
- All 15 unordered arm pairs have no winner, effect, or uncertainty.

Every metric value, numerator, confidence interval, and target-attainment
claim is `null`; every denominator and sample count is zero; and
`missing_as_zero` is always false.  Historical results are not copied forward
as current measurements.

Generate absent evidence files, or verify that existing bytes are exact:

```bash
/usr/bin/python3.12 -S \
  data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/build_not_run.py
```

Perform a read-only builder replay:

```bash
/usr/bin/python3.12 -S \
  data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/build_not_run.py \
  --check
```

Independently verify the frozen PGIR-205 inputs, canonical identities, exact
populations, zero-effect receipts, raw artifact bytes, symlink-free inventory,
Git index bindings, and read-only builder replay:

```bash
/usr/bin/python3.12 -S \
  data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1/verify_not_run.py
```

Both programs use only the Python standard library.  They do not probe GPU
state or contact a network.  `build_not_run.py --check` performs no writes.
Any future executable campaign requires a newly admitted superseding freeze
and a new task revision; this package is immutable evidence of the denied run.
