# PGIR-207 terminal qualification

This directory is the immutable successor qualification for `RESULT(PGIR-207)`.
It consumes the independently verified, typed `not_run` result from
`RESULT(PGIR-206)` and resolves all sixteen final criteria and all thirty-two
report sections without inventing training, evaluation, proof, promotion, or
publication effects.

The terminal disposition is `no_go`.  That is the required truthful outcome:
the superseding freeze denied descendant execution, PGIR-206 created no
checkpoint and measured no metric cells, and no independent promotion or
publication authority exists.  The qualified-claim text is therefore withheld.

The package contains:

- `acceptance.json` and `report_sections.json`, the closed 16/32 resolution;
- `evaluation_receipt.json` and `proof_receipt.json`, independent projections
  of PGIR-206's explicit `not_run` receipts;
- `decision.json`, `promotion_receipt.json`, and `publication_receipt.json`,
  which record `no_go`, an unchanged promotion pointer, and no upload;
- `result_graph.json`, the complete sixteen-node, twenty-eight-edge dependency
  graph in dependency-first order;
- `recipe.json`, `manifest.json`, `verification_receipt.json`, and
  `result.json`, the deterministic qualification closure; and
- the matching successor report and terminal following board under
  `docs/architecture/proof_grounded_ir_learning/successor-v1/`.

`build_qualification.py` is deterministic and write-once.  Generation is only
valid at the exact completed PGIR-206 recursive repository forest pinned in the
builder.  It refuses a missing or changed input, a non-terminal experiment, an
opened hidden label, an observed training/proof/publication effect, or different
bytes at an output path.

After generation and commit, verify from a fresh recursive checkout with the
standard-library interpreter and no site initialization:

```text
/usr/bin/python3.12 -S data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/build_qualification.py --check
/usr/bin/python3.12 -S data/agent_supervisor/proof_grounded_ir_learning/qualification/successor-v1/verify_qualification.py --fresh-recursive
```

Neither command authorizes execution, promotion, publication, or a new
automated campaign.  The following board contains only `PGIR-212`, blocked on
manual external evidence and explicitly unschedulable.
