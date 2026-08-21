# Residual intelligence benchmark staging area

This directory contains only small schemas, manifests, and fixtures. Model
weights, tokenized corpora, and large evaluation outputs belong in the existing
managed content-addressed artifact system and must not be committed to Git.

The complete frozen paired benchmark is owned by `VRIF-030`. The tranche-one
manifest is intentionally `staged_not_qualified`: it fixes the 24-family closed
taxonomy and the required partition vocabulary without claiming held-out
quality, promotion eligibility, savings, or production readiness.

`synthetic_split_manifest.json` demonstrates lineage-safe grouping with hidden
held-out/adversarial partitions. `synthetic_training_admission.json` binds all
mandatory admission fields but deliberately records `training_unavailable`;
passing a leakage audit alone does not authorize training.
