# Procedure compiler benchmark

This directory contains the bounded, frozen PCPC benchmark corpus. The manifest
and `case_recipes.fixture` are jointly authoritative: recipes expand deterministically
to one synthetic, content-addressed case for every task-family/partition pair.
The corpus therefore has 138 cases (23 families × 6 partitions), without
checking in private prompts, source snapshots, or large trajectories.

The six partitions are disjoint by case identity. Held-out identities may not
be read by synthesis, rule refinement, resolver training, or CEGIS candidate
generation. Each recipe carries an expected decision and safety metadata;
adversarial recipes additionally state their attack class and mandatory refusal.
The corpus covers recurring repair, recovery, unknown-family refusal, unsafe
near-match refusal, and cross-repository transfer. Large trajectories, source
snapshots, and AST dumps stay outside Git.
