# Project adapter support matrix

Typed outcomes for repository onboarding (EAAEF-043):

- `preview_only` — empty or skip-only tree
- `unsupported_language` — no recognized language
- `unsupported_build_system` — language without a locked build
- `unsafe_repository` — symlink escape, hooks, bombs
- `insufficient_validation` — no admitted test/static profile
- `human_configuration_required` — policy needs an operator
- `mutation_not_admitted` — inventory only; the generic adapter never admits mutation
- `supported_inventory` — language + build + validation inventory, still not live mutation unless a later gate admits it

The Python adapter compiles structured pytest/ruff argv. Mutation remains independently admitted.
