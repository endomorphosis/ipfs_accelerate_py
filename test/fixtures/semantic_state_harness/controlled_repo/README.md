# Controlled semantic-state fixture repository

Interface: `ControlledSemanticRepository@1`  
Corpus: `semantic-state-controlled-repo-v1`  
Task: SCH-014 / `sch/fixture@1`

## Purpose

Small, deterministic Python 3.12 / pytest target used by the semantic-compression
harness acceptance matrix. Recipes declare base and mutated trees plus
independent oracles; scanners must read tree bytes and must never import or
execute target modules from this fixture during analysis.

## Layout

| Path | Role |
| --- | --- |
| `controlled_repository.py` | `ControlledSemanticRepository` loader/materializer |
| `mutation_case.py` | `MutationCase`, `FixtureOracle`, and facet oracles |
| `recipes.py` | Compact base-tree file map and mutation catalogue |
| `README.md` | This file |

Target sources live only as path→text recipes inside `recipes.py`. Materializers
write them to a destination directory (optionally as a deterministic Git tree).

## Oracle facets (required per mutation)

1. **changed-symbol** — `FixtureOracle.changed_symbol`
2. **Merkle** — `FixtureOracle.merkle`
3. **invalidation / test / proof** — `FixtureOracle.invalidation`
4. **receipt-freshness** — `FixtureOracle.receipt_freshness`
5. **confidence / raw-source** — `FixtureOracle.confidence`

## Safety constraints

- Post-scan source-race marker bytes never enter declared pack paths.
- Unrelated formatting mutations remain bounded (ops and byte budget).
- Mutation cases are oracle/replay fixtures (`production_eligible=false`).
- No network or native extension is required; opaque/native surfaces are syntactic.

## Usage

```python
from pathlib import Path
import importlib.util

# Prefer the test helper loader in test_fixture_repository.py, or:
repo = ControlledSemanticRepository.load()
base = repo.base_tree()
mutated = repo.mutated_tree("local_function_body")
repo.materialize_base(Path("/tmp/sch-base"))
```
