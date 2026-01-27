# Model Manager API Integration - Implementation Status

## Overview

Implementing comprehensive model manager integration with all cached APIs to enable unified model selection based on HuggingFace-compatible pipeline types, as requested in PR comment #3803790547.

## Completed Work

### Phase 1: HuggingFace Pipeline Types & API Model Registry ✅

**Files Created:**
1. `ipfs_accelerate_py/common/pipeline_types.py` (9.7KB)
2. `ipfs_accelerate_py/api_integrations/model_registry.py` (12KB)

**Key Features:**
- 40+ HuggingFace pipeline type definitions
- Pipeline categories: text, vision, audio, multimodal
- Architecture-to-pipeline mapping (BERT, GPT, T5, etc.)
- API provider capability mapping (OpenAI, Claude, Gemini, Groq)
- 20+ pre-registered API models with full metadata
- Query by provider, pipeline type, or capabilities
- Cost tracking and feature flags

**Pipeline Types Implemented:**
- **Text** (10): text-generation/causal_lm, text-classification, summarization, translation, etc.
- **Vision** (7): image-classification, object-detection, image-segmentation, etc.
- **Audio** (5): automatic-speech-recognition, audio-classification, text-to-speech, etc.
- **Multimodal** (4): visual-question-answering, image-to-text, document-qa, etc.

**API Models Registered:**
- OpenAI: GPT-4 Turbo, GPT-4, GPT-3.5 Turbo, Embeddings, Whisper, TTS
- Anthropic: Claude 3 Opus/Sonnet/Haiku (with vision support)
- Google: Gemini Pro/Pro Vision (multimodal)
- Groq: LLaMA 3 70B, Mixtral 8x7B
- Cohere: Command, Embed English v3

**Commit:** ad582f9

## Remaining Work

### Phase 2: Model Manager Integration
**Status:** Not Started

**Tasks:**
- Extend `ModelManager` class with API provider tracking
- Add `get_models_by_pipeline_type()` method (self-hosted + API)
- Create unified query interface
- Add API model caching capability
- Integrate with existing DuckDB/JSON storage

**Files to Modify:**
- `ipfs_accelerate_py/model_manager.py`

**Estimated Work:** 2-3 hours

### Phase 3: CLI/API Dual Mode Support
**Status:** Not Started

**Tasks:**
- Create CLI wrapper scripts for SDK-based tools (Claude, Gemini, Groq)
- Implement fallback logic (try CLI → fallback to API)
- Update all CLI integrations for dual mode
- Add configuration for preferred access method

**Files to Modify:**
- `ipfs_accelerate_py/cli_integrations/claude_code_cli_integration.py`
- `ipfs_accelerate_py/cli_integrations/gemini_cli_integration.py`
- `ipfs_accelerate_py/cli_integrations/groq_cli_integration.py`
- `ipfs_accelerate_py/cli_integrations/base_cli_wrapper.py`

**Estimated Work:** 3-4 hours

### Phase 4: Secrets Manager
**Status:** Not Started

**Tasks:**
- Implement encrypted secrets storage
- Add API key management interface
- Integrate with all API providers
- Support environment variables and config files
- Add key rotation and expiry tracking

**Files to Create:**
- `ipfs_accelerate_py/common/secrets_manager.py`

**Files to Modify:**
- All API integration files
- `ipfs_accelerate_py/cli_integrations/*.py`

**Estimated Work:** 2-3 hours

### Phase 5: Unified Model Selection Interface
**Status:** Not Started

**Tasks:**
- Create unified API for model selection
- Implement filtering by pipeline type
- Return both self-hosted and API models
- Add capability-based filtering
- Integrate with caching infrastructure

**Files to Create:**
- `ipfs_accelerate_py/model_selection.py`

**Estimated Work:** 2 hours

## Total Estimated Remaining Work

**9-12 hours** of development + testing + documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Selection API                       │
│  get_models_for_pipeline(pipeline_type, include_api=True)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────┬──────────────┐
                              │                 │              │
                    ┌─────────▼────────┐ ┌─────▼────┐  ┌─────▼────┐
                    │  Model Manager   │ │   API    │  │ Secrets  │
                    │  (Self-Hosted)   │ │ Registry │  │ Manager  │
                    └──────────────────┘ └──────────┘  └──────────┘
                              │                 │              │
                    ┌─────────▼────────┐ ┌─────▼────┐  ┌─────▼────┐
                    │   DuckDB/JSON    │ │   Cache  │  │ Encrypted│
                    │     Storage      │ │   Layer  │  │  Storage │
                    └──────────────────┘ └──────────┘  └──────────┘
```

## Usage Example (Target)

```python
from ipfs_accelerate_py.model_selection import get_models_for_pipeline

# Get all models for text generation (self-hosted + API)
models = get_models_for_pipeline(
    "text-generation",  # or "causal_lm"
    include_api=True,
    include_self_hosted=True,
    min_context_length=8192,
    max_cost_per_1k=0.01
)

# Results include both types
for model in models:
    print(f"{model.model_name}: {model.source_type}")
    # Output:
    # LLaMA 2 13B: self-hosted
    # GPT-4 Turbo: api
    # Claude 3 Sonnet: api
    # Mistral 7B: self-hosted
```

## Notes

- Phase 1 provides the foundation for all remaining work
- Pipeline types match HuggingFace's official classifications
- API model registry is extensible for custom providers
- Secrets manager will support multiple storage backends
- CLI/API fallback ensures flexibility for users

## Next Steps

1. Continue with Phase 2 (Model Manager Integration)
2. Then proceed sequentially through Phases 3-5
3. Add comprehensive tests for each phase
4. Update documentation as features are completed

## References

- HuggingFace Pipelines: https://huggingface.co/docs/transformers/main_classes/pipelines
- Model Manager: `ipfs_accelerate_py/model_manager.py`
- API Integrations: `ipfs_accelerate_py/api_integrations/`
- CLI Integrations: `ipfs_accelerate_py/cli_integrations/`
