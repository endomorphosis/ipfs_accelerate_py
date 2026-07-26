# Inference router ownership

`ipfs_accelerate_py` is the single implementation owner for the LLM,
embeddings, multimodal, and voice routers:

- `ipfs_accelerate_py.llm_router`
- `ipfs_accelerate_py.embeddings_router`
- `ipfs_accelerate_py.multimodal_router`
- `ipfs_accelerate_py.voice_router`

`ipfs_accelerate_py.embedding_router` is a singular-name alias for
`embeddings_router`. The corresponding `ipfs_datasets_py` router modules are
also compatibility aliases to these exact module objects.

All provider integrations, registries, caches, traces, progress state, batch
behavior, and compatibility environment handling belong in these canonical
modules. Do not add provider logic to an alias module. This keeps direct
accelerator callers and datasets callers on one runtime implementation and one
set of mutable state.

New configuration should use `IPFS_ACCELERATE_PY_*` names. Relevant
`IPFS_DATASETS_PY_*` names remain accepted where needed for existing datasets
deployments.

Run the accelerator router tests together with the datasets canonical-import
contract whenever changing this boundary.
