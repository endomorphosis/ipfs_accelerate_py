# Procedure compiler benchmark

This directory contains the bounded benchmark manifest for PCPC. The family
and partition vocabulary is frozen at bootstrap. The case corpus is deliberately
empty during P0 and is marked `scaffold_only`; `PCPC-029` must add independently
reviewed content-addressed cases before any benchmark, transfer, savings, or
promotion claim is possible. No task family currently has a populated
partition; the six partition names are obligations, not present case coverage.

The six partitions are disjoint by case identity. Held-out identities may not
be read by synthesis, rule refinement, resolver training, or CEGIS candidate
generation. Large trajectories, source snapshots, and AST dumps stay outside
Git and are referenced only by validated CIDs.
