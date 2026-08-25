# PGIR-203 tokenizer admission result

This is a new, write-once successor location. It does not modify the historical
PGIR-014 tokenizer policy.

`RESULT(PGIR-203)` is permanently deterministic-only: the existing LegalIR
canonical vocabulary is exactly identified and its golden token classes and
fail-closed rejection behavior are replayed, but no learned tokenizer or model
checkpoint identity has been submitted for admission. The successor corpus and
split roots additionally certify zero rights-admitted, materialized training
rows. Therefore R2-R6 learned training remains unauthorized.

Verify from the repository root:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/tokenizer/verify_tokenizer_freeze.py
```

`build_tokenizer_freeze.py --initialize` only creates absent JSON receipts and
refuses to replace non-identical bytes. Any future learned-tokenizer admission
must use a new superseding location and satisfy every stated authorization gate.
