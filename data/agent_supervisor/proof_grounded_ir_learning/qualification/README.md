# PGIR-111 qualification, publication decision, and next board

This directory is the immutable `IRQualificationDecision@1` produced for
PGIR-111. It applies the closed 16 final acceptance criteria and 32-section
factual report to already-sealed campaign evidence.

The package verifies successfully but its decision is deliberately `no_go`:
the PGIR-014 freeze reports zero rights-admitted training rows, an
unmaterialized corpus, incomplete required holdouts, and a historical
semantic baseline that is not currently qualified. PGIR-110 therefore never
granted an arm/seed lease or wrote weights. Consequently there is no
candidate checkpoint, promotion is `no_go`, remote publication is `denied`,
and the exact qualified-claim text is withheld. This is an integrity success
and an execution denial, not a fabricated training result.

Run the independent verifier from the repository root:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/qualification/verify_qualification.py --write-receipt
```

Rebuild replay is write-once by default:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/qualification/build_qualification.py
```

The builder refuses to replace different bytes. Any admitted replacement must
use a new task revision and a separately located qualification whose freeze
`previous_root_cid` points at a superseding PGIR-014 root.

Key artifacts:

- `recipe.json`: compact closed qualification recipe.
- `acceptance.json`: all 16 final acceptance criteria with evidence or no-go.
- `report_sections.json`: all 32 report sections with cited bindings.
- `decision.json`: deterministic `no_go` decision.
- `promotion_receipt.json`: independent promotion authority, pointer unchanged.
- `publication_receipt.json`: append-only remote upload denied.
- `result.json`: `RESULT(PGIR-111)`.
- `docs/architecture/proof_grounded_ir_learning/final_report.md`: the 32-section report.
- `docs/architecture/proof_grounded_ir_learning/next.todo.md`: the next board.
