# Unified Inference Architecture Compatibility Path

The detailed implementation notes formerly in this file are retained in the
repository history. For the current design, use:

- [Architecture overview](architecture/overview.md) for runtime layers and
  boundaries;
- [Unified inference backend reference](INFERENCE_BACKEND_README.md) for the
  optional `InferenceBackendManager` and `UnifiedInferenceService` modules;
- [HF model server guide](features/hf-model-server/README.md) for the separate
  FastAPI model-serving surface; and
- [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md)
  for source-of-truth rules.

The unified inference service is optional and is not the same thing as the
canonical MCP server or the package-level `get_instance()` API.
