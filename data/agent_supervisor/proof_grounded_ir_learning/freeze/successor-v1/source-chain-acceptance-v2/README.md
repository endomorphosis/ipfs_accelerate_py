# PGIR-210 superseding source-chain acceptance

This append-only v2 receipt supersedes the PGIR-208 exact-gitlink rule. It
accepts the sealed PGIR-202 nested commit only after proving it is an ancestor
of the PGIR-204 descendant and that all fourteen sealed payload paths remain
byte-identical. The result remains a permanent no-go and never authorizes
PGIR-205.

Run `python scripts/verify_proof_grounded_ir_learning_successor_source_chain_v2.py --network` for the mandatory live replay of all twenty-one exact-revision metadata citations. Offline or cached responses are not accepted. The portability receipt is deliberately typed as an outer-and-nested unpublished-ref no-go, so a local object database is not mistaken for a fresh recursive-checkout proof.
