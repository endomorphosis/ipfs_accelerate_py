"""Compatibility projection for the canonical AI service catalog.

Historically this module and :mod:`api_backends.api_models_registry` each
owned a different static model inventory.  The seed rows below are now the
single source of that legacy knowledge.  ``APIModelRegistry`` projects those
rows through the versioned model catalog while preserving the public
``APIModel`` return type and all established lookup helpers.

Deprecation is intentionally reversible: the module and its public imports
remain supported compatibility APIs.  New code should query
``model_catalog.AIServiceCatalog`` through ``ModelManager``.  No removal
version is scheduled, and neither import nor construction performs network
discovery.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..model_catalog.catalog import AIServiceCatalog
from ..model_catalog.schema import LifecycleState, ModelDescriptor, Operation
from ..model_catalog.sources.static import CatalogSourceResult, StaticCatalogSource


LEGACY_REGISTRY_DEPRECATION = {
    "deprecated": True,
    "replacement": "ModelManager model catalog",
    "removal_scheduled": False,
    "reversible": True,
}
STATIC_SOURCE_NAME = "legacy.api-models.static"
RUNTIME_SOURCE_NAME = "legacy.api-models.runtime"
STATIC_SOURCE_PRECEDENCE = 10
RUNTIME_SOURCE_PRECEDENCE = 100


class APIProviderType(Enum):
    """Enumeration retained for legacy API-model callers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    REPLICATE = "replicate"
    META_AI = "meta_ai"

    # Inference engines
    VLLM = "vllm"
    HF_TGI = "hf_tgi"
    HF_TEI = "hf_tei"
    OVMS = "ovms"
    OPEA = "opea"


PROVIDER_ALIASES: Mapping[str, str] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "cohere": "cohere",
    "gemini": "google",
    "google": "google",
    "groq": "groq",
    "hf": "huggingface",
    "hf-tei": "huggingface",
    "hf-tgi": "huggingface",
    "hf_tei": "huggingface",
    "hf_tgi": "huggingface",
    "huggingface": "huggingface",
    "meta-ai": "meta_ai",
    "meta_ai": "meta_ai",
    "meta-llama": "meta_ai",
    "meta_llama": "meta_ai",
    "meta-spark": "meta_ai",
    "meta_spark": "meta_ai",
    "ollama": "ollama",
    "openai": "openai",
    "openai-api": "openai",
    "openai_api": "openai",
    "openvino": "ovms",
    "ovms": "ovms",
    "replicate": "replicate",
    "vllm": "vllm",
    "opea": "opea",
}

_PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "cohere": "Cohere",
    "google": "Google",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "meta_ai": "Meta AI",
    "ollama": "Ollama",
    "openai": "OpenAI",
    "ovms": "OpenVINO Model Server",
}

_PROVIDER_CATALOG_ALIASES = {
    "anthropic": ("claude",),
    "google": ("gemini",),
    "huggingface": ("hf", "hf-tei", "hf-tgi"),
    "meta_ai": ("meta-ai", "meta-llama", "meta-spark"),
    "openai": ("openai-api",),
    "ovms": ("openvino",),
}

_OPERATION_TO_PIPELINE = {
    Operation.TEXT_GENERATE: "text-generation",
    Operation.TEXT_CHAT: "conversational",
    Operation.EMBEDDING_GENERATE: "feature-extraction",
    Operation.VISION_GENERATE: "visual-question-answering",
    Operation.AUDIO_TRANSCRIBE: "automatic-speech-recognition",
    Operation.AUDIO_SYNTHESIZE: "text-to-speech",
}
_PIPELINE_TO_OPERATION = {
    pipeline: operation.value
    for operation, pipeline in _OPERATION_TO_PIPELINE.items()
}
_PIPELINE_TO_OPERATION["image-to-text"] = Operation.VISION_GENERATE.value


@dataclass
class APIModel:
    """The unchanged legacy value object projected from a catalog model."""

    model_id: str
    model_name: str
    provider: APIProviderType
    pipeline_types: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    supports_streaming: bool = False
    cost_per_1k_tokens: Optional[Dict[str, float]] = None
    description: str = ""
    is_multimodal: bool = False
    vision_capable: bool = False
    function_calling: bool = False
    json_mode: bool = False
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-compatible legacy record."""

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider.value,
            "pipeline_types": list(self.pipeline_types),
            "context_length": self.context_length,
            "supports_streaming": self.supports_streaming,
            "cost_per_1k_tokens": (
                None
                if self.cost_per_1k_tokens is None
                else dict(sorted(self.cost_per_1k_tokens.items()))
            ),
            "description": self.description,
            "is_multimodal": self.is_multimodal,
            "vision_capable": self.vision_capable,
            "function_calling": self.function_calling,
            "json_mode": self.json_mode,
            "deprecated": self.deprecated,
        }


# Rich metadata formerly declared directly on APIModelRegistry.
_RICH_MODEL_ROWS: Tuple[Mapping[str, Any], ...] = (
    {
        "model_id": "gpt-4-turbo",
        "model_name": "GPT-4 Turbo",
        "provider": "openai",
        "pipeline_types": (
            "text-generation",
            "conversational",
            "text-classification",
        ),
        "context_length": 128000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
        "vision_capable": True,
        "is_multimodal": True,
        "function_calling": True,
        "json_mode": True,
    },
    {
        "model_id": "gpt-4",
        "model_name": "GPT-4",
        "provider": "openai",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 8192,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
        "function_calling": True,
        "json_mode": True,
    },
    {
        "model_id": "gpt-3.5-turbo",
        "model_name": "GPT-3.5 Turbo",
        "provider": "openai",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 16385,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015},
        "function_calling": True,
        "json_mode": True,
    },
    {
        "model_id": "text-embedding-3-large",
        "model_name": "Text Embedding 3 Large",
        "provider": "openai",
        "pipeline_types": ("feature-extraction",),
        "context_length": 8191,
        "cost_per_1k_tokens": {"input": 0.00013, "output": 0},
    },
    {
        "model_id": "whisper-1",
        "model_name": "Whisper",
        "provider": "openai",
        "pipeline_types": ("automatic-speech-recognition",),
        "cost_per_1k_tokens": {"input": 0.006, "output": 0},
    },
    {
        "model_id": "tts-1",
        "model_name": "Text-to-Speech 1",
        "provider": "openai",
        "pipeline_types": ("text-to-speech",),
        "cost_per_1k_tokens": {"input": 0.015, "output": 0},
    },
    {
        "model_id": "claude-3-opus-20240229",
        "model_name": "Claude 3 Opus",
        "provider": "anthropic",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 200000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.015, "output": 0.075},
        "vision_capable": True,
        "is_multimodal": True,
    },
    {
        "model_id": "claude-3-sonnet-20240229",
        "model_name": "Claude 3 Sonnet",
        "provider": "anthropic",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 200000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
        "vision_capable": True,
        "is_multimodal": True,
    },
    {
        "model_id": "claude-3-haiku-20240307",
        "model_name": "Claude 3 Haiku",
        "provider": "anthropic",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 200000,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125},
    },
    {
        "model_id": "gemini-pro",
        "model_name": "Gemini Pro",
        "provider": "google",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 30720,
        "supports_streaming": True,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.0005},
    },
    {
        "model_id": "gemini-pro-vision",
        "model_name": "Gemini Pro Vision",
        "provider": "google",
        "pipeline_types": (
            "text-generation",
            "visual-question-answering",
            "image-to-text",
        ),
        "context_length": 12288,
        "supports_streaming": True,
        "vision_capable": True,
        "is_multimodal": True,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.0005},
    },
    {
        "model_id": "llama3-70b-8192",
        "model_name": "LLaMA 3 70B",
        "provider": "groq",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 8192,
        "supports_streaming": True,
        "description": "Fast inference on Groq LPU",
    },
    {
        "model_id": "mixtral-8x7b-32768",
        "model_name": "Mixtral 8x7B",
        "provider": "groq",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 32768,
        "supports_streaming": True,
        "description": "Fast inference on Groq LPU",
    },
    {
        "model_id": "command",
        "model_name": "Command",
        "provider": "cohere",
        "pipeline_types": ("text-generation", "conversational"),
        "context_length": 4096,
        "supports_streaming": True,
    },
    {
        "model_id": "embed-english-v3.0",
        "model_name": "Embed English v3",
        "provider": "cohere",
        "pipeline_types": ("feature-extraction",),
        "context_length": 512,
    },
)

# The JSON files consumed by api_backends.api_models_registry before the
# convergence.  They are data here, not files discovered during import.
_LEGACY_BACKEND_INVENTORY: Mapping[str, Tuple[str, ...]] = {
    "claude": (
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-2.1",
        "anthropic/claude-2.0",
        "anthropic/claude-instant-1.2",
        "anthropic/claude-instant-1.1",
        "anthropic/claude-1.0",
        "anthropic/claude-1.2",
        "anthropic/claude-1.3",
        "anthropic/text-embedding-model-1",
    ),
    "gemini": (
        "google/gemini-pro",
        "google/gemini-pro-vision",
        "google/gemini-nano",
        "google/gemini-ultra",
        "google/embedding-001",
        "google/embedding-gecko",
        "google/text-embedding-model-001",
    ),
    "groq": (
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/llama-guard-3-8b",
        "groq/llama3-70b-8192",
        "groq/llama3-8b-8192",
        "groq/mixtral-8x7b-32768",
        "groq/gemma2-9b-it",
        "groq/qwen-2.5-32b",
        "groq/deepseek-r1-distill-qwen-32b",
        "groq/deepseek-r1-distill-llama-70b",
        "groq/deepseek-r1-distill-llama-70b-specdec",
        "groq/whisper-large-v3",
        "groq/whisper-large-v3-turbo",
        "groq/distil-whisper-large-v3-en",
    ),
    "hf_tei": (
        "huggingface/all-mpnet-base-v2",
        "huggingface/all-MiniLM-L6-v2",
        "huggingface/bge-large-en-v1.5",
        "huggingface/bge-base-en-v1.5",
        "huggingface/e5-large-v2",
        "huggingface/e5-base-v2",
        "huggingface/instructor-xl",
        "huggingface/instructor-large",
        "huggingface/gte-large",
        "huggingface/gte-base",
        "huggingface/sentence-t5-xxl",
        "huggingface/sentence-t5-xl",
        "huggingface/multilingual-e5-large",
        "huggingface/multilingual-e5-base",
    ),
    "hf_tgi": (
        "huggingface/falcon-40b",
        "huggingface/falcon-7b",
        "huggingface/mistral-7b",
        "huggingface/mistral-8x7b",
        "huggingface/llama-2-7b",
        "huggingface/llama-2-13b",
        "huggingface/llama-2-70b",
        "huggingface/codellama-7b",
        "huggingface/codellama-13b",
        "huggingface/codellama-34b",
        "huggingface/starcoder-15.5b",
        "huggingface/starcoder2-7b",
        "huggingface/starcoder2-15b",
        "huggingface/phi-2",
        "huggingface/mpt-7b",
        "huggingface/mpt-30b",
        "huggingface/gpt-neox-20b",
    ),
    "ollama": (
        "ollama/llama2",
        "ollama/llama2-uncensored",
        "ollama/mistral",
        "ollama/mixtral",
        "ollama/codellama",
        "ollama/phi",
        "ollama/neural-chat",
        "ollama/starling-lm",
        "ollama/stable-beluga",
        "ollama/wizard-vicuna",
        "ollama/orca-mini",
        "ollama/vicuna",
        "ollama/nous-hermes",
        "ollama/openchat",
        "ollama/deepseek-coder",
        "ollama/sqlcoder",
        "ollama/falcon",
        "ollama/yi",
        "ollama/llava",
        "ollama/bakllava",
        "ollama/gemma",
        "ollama/tinyllama",
        "ollama/stablelm-zephyr",
    ),
    "openai_api": (
        "openai/gpt-4o-mini-audio-preview-2024-12-17",
        "openai/dall-e-3",
        "openai/dall-e-2",
        "openai/gpt-4o-audio-preview-2024-10-01",
        "openai/gpt-4o-audio-preview",
        "openai/o1-mini-2024-09-12",
        "openai/gpt-4o-mini-realtime-preview-2024-12-17",
        "openai/o1-preview-2024-09-12",
        "openai/o1-mini",
        "openai/o1-preview",
        "openai/gpt-4o-mini-realtime-preview",
        "openai/whisper-1",
        "openai/gpt-4-turbo",
        "openai/gpt-4o-mini-audio-preview",
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-4o-realtime-preview-2024-10-01",
        "openai/babbage-002",
        "openai/tts-1-hd-1106",
        "openai/gpt-4o-audio-preview-2024-12-17",
        "openai/tts-1-hd",
        "openai/gpt-4o-2024-08-06",
        "openai/gpt-4o",
        "openai/chatgpt-4o-latest",
        "openai/text-embedding-3-large",
        "openai/gpt-4-turbo-2024-04-09",
        "openai/tts-1",
        "openai/tts-1-1106",
        "openai/davinci-002",
        "openai/gpt-3.5-turbo-1106",
        "openai/omni-moderation-2024-09-26",
        "openai/gpt-3.5-turbo-instruct",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini-2024-07-18",
        "openai/o1",
        "openai/gpt-3.5-turbo-instruct-0914",
        "openai/o1-2024-12-17",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-4o-realtime-preview-2024-12-17",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o-realtime-preview",
        "openai/gpt-3.5-turbo-16k",
        "openai/text-embedding-3-small",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/text-embedding-ada-002",
        "openai/omni-moderation-latest",
        "openai/o3-mini",
        "openai/o3-mini-2025-01-31",
        "openai/gpt-4-0613",
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4-0125-preview",
    ),
    "ovms": (
        "openvino/bert-base-uncased",
        "openvino/roberta-base",
        "openvino/distilbert-base-uncased",
        "openvino/mobilenet-v3-large",
        "openvino/resnet50",
        "openvino/yolov5",
        "openvino/faster-rcnn",
        "openvino/deeplabv3",
        "openvino/wav2vec2-base",
        "openvino/whisper-tiny",
        "openvino/whisper-base",
        "openvino/gpt2",
        "openvino/opt-350m",
        "openvino/bloom-560m",
        "openvino/stable-diffusion-v1-5",
        "openvino/llava-v1.5-7b",
        "openvino/clip-vit-base-patch32",
        "openvino/t5-small",
        "openvino/mt5-small",
    ),
}

_MODEL_ALIASES = {
    "anthropic/claude-3-opus": "claude-3-opus-20240229",
    "anthropic/claude-3-sonnet": "claude-3-sonnet-20240229",
    "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
}


def _json_label(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _infer_pipeline_types(backend: str, qualified_name: str) -> Tuple[str, ...]:
    name = qualified_name.casefold()
    if backend == "hf_tei" or "embedding" in name or "encoder" in name:
        return ("feature-extraction",)
    if "whisper" in name:
        return ("automatic-speech-recognition",)
    if "/tts-" in name:
        return ("text-to-speech",)
    if backend in {"claude", "gemini", "groq", "hf_tgi", "ollama", "openai_api"}:
        return ("text-generation", "conversational")
    return ()


def _canonical_provider(value: Any) -> str:
    if isinstance(value, APIProviderType):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("provider must be a supported provider name")
    key = value.strip().casefold().replace(" ", "_")
    canonical = PROVIDER_ALIASES.get(key)
    if canonical is None:
        raise ValueError("unknown API provider: %s" % value)
    return canonical


def _make_seed_rows() -> Tuple[Mapping[str, Any], ...]:
    """Merge both historic inventories into one immutable catalog seed."""

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order = 0
    for rich in _RICH_MODEL_ROWS:
        row = dict(rich)
        provider = _canonical_provider(row["provider"])
        name = str(row["model_id"]).casefold()
        row["provider"] = provider
        row["model_id"] = name
        row["pipeline_types"] = list(row.get("pipeline_types", ()))
        row["aliases"] = [provider + "/" + name]
        row["legacy_rich"] = True
        row["legacy_order"] = order
        row["legacy_backend_models"] = {}
        merged[(provider, name)] = row
        order += 1

    for backend, names in _LEGACY_BACKEND_INVENTORY.items():
        for qualified_name in names:
            prefix, legacy_name = qualified_name.split("/", 1)
            provider = _canonical_provider(prefix)
            canonical_name = _MODEL_ALIASES.get(
                prefix.casefold() + "/" + legacy_name.casefold(),
                legacy_name.casefold(),
            )
            key = (provider, canonical_name)
            row = merged.get(key)
            if row is None:
                row = {
                    "provider": provider,
                    "model_id": canonical_name,
                    "model_name": legacy_name,
                    "pipeline_types": list(
                        _infer_pipeline_types(backend, qualified_name)
                    ),
                    "legacy_rich": False,
                    "legacy_order": order,
                    "legacy_backend_models": {},
                    "aliases": [],
                }
                merged[key] = row
                order += 1
            aliases = set(row.get("aliases", ()))
            aliases.add(qualified_name.casefold())
            if legacy_name.casefold() != canonical_name:
                aliases.add(legacy_name.casefold())
            row["aliases"] = sorted(aliases)
            backend_models = dict(row.get("legacy_backend_models", {}))
            backend_models[backend] = qualified_name
            row["legacy_backend_models"] = backend_models

    providers = sorted({provider for provider, _ in merged})
    rows: List[Mapping[str, Any]] = []
    for index, provider in enumerate(providers):
        rows.append(
            {
                "provider": provider,
                "__provider_only__": True,
                "display_name": _PROVIDER_DISPLAY_NAMES.get(provider),
                "aliases": list(_PROVIDER_CATALOG_ALIASES.get(provider, ())),
                # Model rows also synthesize a provider.  Give the explicit
                # provider declaration one deterministic point of precedence
                # so aliases and display metadata cannot be displaced.
                "precedence": STATIC_SOURCE_PRECEDENCE + 1,
                "labels": {"legacy.order": str(index)},
            }
        )

    for row in sorted(merged.values(), key=lambda item: int(item["legacy_order"])):
        labels = {
            "legacy.order": str(row["legacy_order"]),
            "legacy.pipeline-types": _json_label(row.get("pipeline_types", ())),
            "legacy.cost": _json_label(row.get("cost_per_1k_tokens")),
            "legacy.supports-streaming": _json_label(
                bool(row.get("supports_streaming", False))
            ),
            "legacy.is-multimodal": _json_label(
                bool(row.get("is_multimodal", False))
            ),
            "legacy.vision-capable": _json_label(
                bool(row.get("vision_capable", False))
            ),
            "legacy.function-calling": _json_label(
                bool(row.get("function_calling", False))
            ),
            "legacy.json-mode": _json_label(bool(row.get("json_mode", False))),
            "legacy.backend-models": _json_label(
                row.get("legacy_backend_models", {})
            ),
            "legacy.rich": _json_label(bool(row.get("legacy_rich", False))),
        }
        output = dict(row)
        for private_name in (
            "legacy_order",
            "legacy_backend_models",
            "legacy_rich",
        ):
            output.pop(private_name, None)
        output["labels"] = labels
        # The catalog has a deliberately smaller, typed operation vocabulary.
        # Keep every legacy pipeline string in its label, while adapting only
        # compatible values to catalog capabilities.
        output["operations"] = sorted(
            {
                _PIPELINE_TO_OPERATION[pipeline]
                for pipeline in output.get("pipeline_types", ())
                if pipeline in _PIPELINE_TO_OPERATION
            }
        )
        if not output["operations"]:
            output["supports_streaming"] = False
            output["function_calling"] = False
        rows.append(output)
    return tuple(rows)


def _freeze_seed_value(value: Any) -> Any:
    """Recursively freeze seed data so callers cannot fork static knowledge."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_seed_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_seed_value(item) for item in value)
    return value


# This deeply immutable tuple is the only canonical legacy seed.
API_MODEL_SEED_ROWS: Tuple[Mapping[str, Any], ...] = tuple(
    _freeze_seed_value(row) for row in _make_seed_rows()
)


def get_api_model_seed_rows() -> Tuple[Mapping[str, Any], ...]:
    """Return defensive copies of the canonical, side-effect-free seed rows."""

    return tuple(
        {
            key: (
                dict(value)
                if isinstance(value, Mapping)
                else list(value)
                if isinstance(value, (list, tuple))
                else value
            )
            for key, value in row.items()
        }
        for row in API_MODEL_SEED_ROWS
    )


def _labels(record: ModelDescriptor) -> Dict[str, str]:
    return dict(record.labels)


def _decode_label(
    labels: Mapping[str, str], name: str, default: Any
) -> Any:
    value = labels.get(name)
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _provider_type(name: str) -> APIProviderType:
    canonical = _canonical_provider(name)
    return APIProviderType(canonical)


def _descriptor_to_api_model(
    descriptor: ModelDescriptor, provider_name: str
) -> APIModel:
    labels = _labels(descriptor)
    pipelines = _decode_label(labels, "legacy.pipeline-types", None)
    if not isinstance(pipelines, list):
        pipelines = []
        for capability in descriptor.capabilities:
            for operation in capability.operations:
                pipeline = _OPERATION_TO_PIPELINE.get(operation)
                if pipeline is not None and pipeline not in pipelines:
                    pipelines.append(pipeline)
    cost = _decode_label(labels, "legacy.cost", None)
    if not isinstance(cost, dict):
        cost = None
    context_length = next(
        (
            capability.max_context_tokens
            for capability in descriptor.capabilities
            if capability.max_context_tokens is not None
        ),
        None,
    )
    operations = {
        operation
        for capability in descriptor.capabilities
        for operation in capability.operations
    }
    vision = bool(
        _decode_label(labels, "legacy.vision-capable", False)
        or Operation.VISION_GENERATE in operations
    )
    return APIModel(
        model_id=descriptor.name,
        model_name=descriptor.display_name or descriptor.name,
        provider=_provider_type(provider_name),
        pipeline_types=list(pipelines),
        context_length=context_length,
        supports_streaming=bool(
            _decode_label(labels, "legacy.supports-streaming", False)
            or Operation.STREAM in operations
        ),
        cost_per_1k_tokens=cost,
        description=descriptor.description,
        is_multimodal=bool(
            _decode_label(labels, "legacy.is-multimodal", False) or vision
        ),
        vision_capable=vision,
        function_calling=bool(
            _decode_label(labels, "legacy.function-calling", False)
            or Operation.TOOL_CALL in operations
        ),
        json_mode=bool(_decode_label(labels, "legacy.json-mode", False)),
        deprecated=descriptor.lifecycle == LifecycleState.DEPRECATED,
    )


def _api_model_to_row(model: APIModel, order: int) -> Mapping[str, Any]:
    provider = _canonical_provider(model.provider)
    qualified = provider + "/" + model.model_id.casefold()
    backend = _default_backend_for_model(model)
    operations = sorted(
        {
            _PIPELINE_TO_OPERATION[pipeline.casefold()]
            for pipeline in model.pipeline_types
            if pipeline.casefold() in _PIPELINE_TO_OPERATION
        }
    )
    return {
        "provider": provider,
        "model_id": model.model_id,
        "model_name": model.model_name,
        "aliases": [qualified],
        "pipeline_types": list(model.pipeline_types),
        "operations": operations,
        "context_length": model.context_length,
        "supports_streaming": model.supports_streaming if operations else False,
        "cost_per_1k_tokens": (
            None
            if model.cost_per_1k_tokens is None
            else dict(model.cost_per_1k_tokens)
        ),
        "description": model.description,
        "is_multimodal": model.is_multimodal,
        "vision_capable": model.vision_capable,
        "function_calling": model.function_calling if operations else False,
        "json_mode": model.json_mode,
        "deprecated": model.deprecated,
        "labels": {
            "legacy.order": str(order),
            "legacy.pipeline-types": _json_label(model.pipeline_types),
            "legacy.cost": _json_label(model.cost_per_1k_tokens),
            "legacy.supports-streaming": _json_label(model.supports_streaming),
            "legacy.is-multimodal": _json_label(model.is_multimodal),
            "legacy.vision-capable": _json_label(model.vision_capable),
            "legacy.function-calling": _json_label(model.function_calling),
            "legacy.json-mode": _json_label(model.json_mode),
            "legacy.backend-models": _json_label({backend: qualified}),
            "legacy.rich": "false",
        },
    }


def _default_backend_for_model(model: APIModel) -> str:
    provider = _canonical_provider(model.provider)
    if provider == "huggingface":
        if "feature-extraction" in model.pipeline_types:
            return "hf_tei"
        return "hf_tgi"
    return {
        "anthropic": "claude",
        "google": "gemini",
        "openai": "openai_api",
        "ovms": "ovms",
    }.get(provider, provider)


class RuntimeAPIModelCatalogSource:
    """Mutable local source used for explicit compatibility additions."""

    source = RUNTIME_SOURCE_NAME
    precedence = RUNTIME_SOURCE_PRECEDENCE
    side_effecting = False

    def __init__(self, rows: Iterable[Mapping[str, Any]] = ()) -> None:
        self._lock = threading.RLock()
        self._rows: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            provider = _canonical_provider(row.get("provider"))
            model_id = str(row.get("model_id", "")).casefold()
            if not model_id:
                raise ValueError("runtime model row requires model_id")
            self._rows[(provider, model_id)] = dict(row)
        self._next_order = len(API_MODEL_SEED_ROWS) + len(self._rows)

    def upsert(self, model: APIModel) -> None:
        if not isinstance(model, APIModel):
            raise TypeError("model must be an APIModel")
        provider = _canonical_provider(model.provider)
        if not isinstance(model.model_id, str):
            raise TypeError("model_id must be a string")
        model_id = model.model_id.strip().casefold()
        if not model_id:
            raise ValueError("model_id must not be empty")
        with self._lock:
            old = self._rows.get((provider, model_id))
            if old is not None:
                order = int(dict(old.get("labels", {})).get("legacy.order", 0))
            else:
                order = self._next_order
            row = _api_model_to_row(model, order)
            validation = StaticCatalogSource(
                (row,),
                source=self.source,
                precedence=self.precedence,
            ).load()
            if validation.error_count or len(validation.models) != 1:
                reason = (
                    validation.diagnostics[0].message
                    if validation.diagnostics
                    else "model did not produce a catalog record"
                )
                raise ValueError("invalid API model: %s" % reason)
            self._rows[(provider, model_id)] = row
            if old is None:
                self._next_order += 1

    def load(self) -> CatalogSourceResult:
        with self._lock:
            rows = tuple(
                self._rows[key]
                for key in sorted(
                    self._rows,
                    key=lambda item: int(
                        dict(self._rows[item].get("labels", {})).get(
                            "legacy.order", 0
                        )
                    ),
                )
            )
        return StaticCatalogSource(
            rows,
            source=self.source,
            precedence=self.precedence,
        ).load()

    refresh = load
    snapshot = load
    read = load


class APIModelRegistry:
    """Legacy APIModel views backed by an ``AIServiceCatalog`` snapshot."""

    # Populated below as projections of API_MODEL_SEED_ROWS.
    OPENAI_MODELS: List[APIModel] = []
    ANTHROPIC_MODELS: List[APIModel] = []
    GOOGLE_MODELS: List[APIModel] = []
    GROQ_MODELS: List[APIModel] = []
    COHERE_MODELS: List[APIModel] = []

    deprecation = LEGACY_REGISTRY_DEPRECATION

    def __init__(
        self,
        catalog: Optional[AIServiceCatalog] = None,
        runtime_source: Optional[RuntimeAPIModelCatalogSource] = None,
    ) -> None:
        self._catalog = catalog or AIServiceCatalog()
        self._runtime_source = runtime_source or RuntimeAPIModelCatalogSource()
        if catalog is None:
            self._catalog.register_source(
                STATIC_SOURCE_NAME,
                StaticCatalogSource(
                    get_api_model_seed_rows(),
                    source=STATIC_SOURCE_NAME,
                    precedence=STATIC_SOURCE_PRECEDENCE,
                ),
                strict=True,
            )
        existing_sources = {state.name for state in self._catalog.source_states()}
        self._runtime_source_registered = (
            self._runtime_source.source not in existing_sources
        )
        if self._runtime_source_registered:
            self._catalog.register_source(
                self._runtime_source.source,
                self._runtime_source,
                strict=True,
            )

    @property
    def catalog(self) -> AIServiceCatalog:
        """Expose the canonical facade for reversible caller migration."""

        return self._catalog

    @property
    def catalog_revision(self) -> str:
        return self._catalog.revision

    def _ordered_descriptors(
        self, include_deprecated: bool = True
    ) -> List[Tuple[ModelDescriptor, str]]:
        snapshot = self._catalog.snapshot()
        provider_names = {
            provider.provider_id: provider.name for provider in snapshot.providers
        }

        def key(item: ModelDescriptor) -> Tuple[int, str, str]:
            labels = _labels(item)
            try:
                order = int(labels.get("legacy.order", "1000000000"))
            except ValueError:
                order = 1000000000
            return (order, provider_names.get(item.provider_id, ""), item.name)

        descriptors = sorted(snapshot.models, key=key)
        return [
            (item, provider_names[item.provider_id])
            for item in descriptors
            if include_deprecated or item.lifecycle != LifecycleState.DEPRECATED
        ]

    def _get_descriptor(self, model_id: Any) -> Optional[ModelDescriptor]:
        """Resolve stable IDs, model aliases, and provider-qualified aliases."""

        if not isinstance(model_id, str) or not model_id.strip():
            return None
        requested = model_id.strip().casefold()
        try:
            descriptor = self._catalog.get(requested, record_type="models")
        except ValueError:
            descriptor = None
        if descriptor is not None:
            return descriptor
        if "/" not in requested:
            return None

        prefix, name = requested.split("/", 1)
        try:
            provider_name = _canonical_provider(prefix)
        except ValueError:
            return None
        canonical_alias = provider_name + "/" + name
        if canonical_alias != requested:
            try:
                descriptor = self._catalog.get(
                    canonical_alias, record_type="models"
                )
            except ValueError:
                descriptor = None
            if descriptor is not None:
                return descriptor

        snapshot = self._catalog.snapshot()
        providers = {
            provider.provider_id: provider.name for provider in snapshot.providers
        }
        matches = [
            model
            for model in snapshot.models
            if model.name == name
            and providers.get(model.provider_id) == provider_name
        ]
        return matches[0] if len(matches) == 1 else None

    def get_model(self, model_id: str) -> Optional[APIModel]:
        """Get a model by stable catalog ID, canonical name, or alias."""

        descriptor = self._get_descriptor(model_id)
        if descriptor is None:
            return None
        providers = {
            provider.provider_id: provider.name
            for provider in self._catalog.snapshot().providers
        }
        return _descriptor_to_api_model(
            descriptor, providers[descriptor.provider_id]
        )

    def resolve_model_id(self, model_id: str) -> Optional[str]:
        """Resolve any model alias to its stable catalog model ID."""

        descriptor = self._get_descriptor(model_id)
        return None if descriptor is None else descriptor.model_id

    def resolve_model_name(self, model_id: str) -> Optional[str]:
        """Resolve any model alias to the legacy canonical model name."""

        model = self.get_model(model_id)
        return None if model is None else model.model_id

    def resolve_provider_id(self, provider: Any) -> Optional[str]:
        """Resolve provider aliases (for example ``gemini``) to a stable ID."""

        try:
            canonical = _canonical_provider(provider)
            descriptor = self._catalog.get(canonical, record_type="providers")
        except (TypeError, ValueError):
            return None
        return None if descriptor is None else descriptor.provider_id

    def get_models_by_provider(self, provider: Any) -> List[APIModel]:
        """Get all models for a provider or provider alias."""

        try:
            canonical = _canonical_provider(provider)
        except ValueError:
            return []
        return [
            _descriptor_to_api_model(descriptor, provider_name)
            for descriptor, provider_name in self._ordered_descriptors()
            if provider_name == canonical
        ]

    def get_models_by_pipeline_type(self, pipeline_type: str) -> List[APIModel]:
        """Get non-deprecated models supporting a legacy pipeline type."""

        if not isinstance(pipeline_type, str):
            return []
        selected = pipeline_type.strip().casefold()
        return [
            model
            for model in self.get_all_models()
            if selected in {item.casefold() for item in model.pipeline_types}
        ]

    def get_all_models(self, include_deprecated: bool = False) -> List[APIModel]:
        """Return deterministic APIModel projections from one snapshot."""

        return [
            _descriptor_to_api_model(descriptor, provider_name)
            for descriptor, provider_name in self._ordered_descriptors(
                include_deprecated=include_deprecated
            )
        ]

    def list_models(
        self,
        provider: Optional[Any] = None,
        pipeline_type: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[APIModel]:
        """Compatibility spelling for deterministic filtered model listing."""

        models = self.get_all_models(include_deprecated=include_deprecated)
        if provider is not None:
            try:
                canonical = _canonical_provider(provider)
            except ValueError:
                return []
            models = [
                model for model in models if model.provider.value == canonical
            ]
        if pipeline_type is not None:
            selected = pipeline_type.casefold()
            models = [
                model
                for model in models
                if selected in {item.casefold() for item in model.pipeline_types}
            ]
        return models

    def search_models(
        self,
        query: str,
        provider: Optional[Any] = None,
        pipeline_type: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[APIModel]:
        """Search legacy fields without performing discovery."""

        if not isinstance(query, str):
            return []
        needle = query.strip().casefold()
        models = self.list_models(
            provider=provider,
            pipeline_type=pipeline_type,
            include_deprecated=include_deprecated,
        )
        if not needle:
            return models
        return [
            model
            for model in models
            if needle
            in " ".join(
                (
                    model.model_id,
                    model.model_name,
                    model.provider.value,
                    model.description,
                    *model.pipeline_types,
                )
            ).casefold()
        ]

    def recommend_models(
        self,
        pipeline_type: str,
        max_cost_per_1k: Optional[float] = None,
        min_context_length: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[APIModel]:
        """Return deterministic, compatibility-oriented recommendations."""

        models = self.get_models_by_pipeline_type(pipeline_type)
        if max_cost_per_1k is not None:
            models = [
                model
                for model in models
                if model.cost_per_1k_tokens is None
                or (
                    sum(model.cost_per_1k_tokens.values())
                    / max(len(model.cost_per_1k_tokens), 1)
                )
                <= max_cost_per_1k
            ]
        if min_context_length is not None:
            models = [
                model
                for model in models
                if model.context_length is not None
                and model.context_length >= min_context_length
            ]

        def rank(model: APIModel) -> Tuple[float, int, str, str]:
            cost = (
                float("inf")
                if model.cost_per_1k_tokens is None
                else sum(model.cost_per_1k_tokens.values())
                / max(len(model.cost_per_1k_tokens), 1)
            )
            return (
                cost,
                -(model.context_length or 0),
                model.provider.value,
                model.model_id,
            )

        models = sorted(models, key=rank)
        if limit is None:
            return models
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return models[:limit]

    def recommend_model(self, pipeline_type: str, **kwargs: Any) -> Optional[APIModel]:
        """Return the first recommendation, or ``None`` when none qualify."""

        kwargs.pop("limit", None)
        models = self.recommend_models(pipeline_type, limit=1, **kwargs)
        return models[0] if models else None

    def validate_model(
        self,
        model_id: str,
        provider: Optional[Any] = None,
        pipeline_type: Optional[str] = None,
    ) -> bool:
        """Validate a model alias and optional provider/pipeline constraints."""

        model = self.get_model(model_id)
        if model is None or model.deprecated:
            return False
        if provider is not None:
            try:
                if model.provider.value != _canonical_provider(provider):
                    return False
            except ValueError:
                return False
        return pipeline_type is None or pipeline_type.casefold() in {
            item.casefold() for item in model.pipeline_types
        }

    def get_all_providers(self) -> List[APIProviderType]:
        """Return provider enums in first-model projection order."""

        providers: List[APIProviderType] = []
        for model in self.get_all_models(include_deprecated=True):
            if model.provider not in providers:
                providers.append(model.provider)
        return providers

    def get_supported_pipeline_types(self) -> List[str]:
        """Return pipeline names in first-model projection order."""

        result: List[str] = []
        for model in self.get_all_models(include_deprecated=True):
            for pipeline_type in model.pipeline_types:
                if pipeline_type not in result:
                    result.append(pipeline_type)
        return result

    def register_custom_model(self, model: APIModel) -> None:
        """Publish a runtime addition through the registered runtime source."""

        if not self._runtime_source_registered:
            raise RuntimeError(
                "injected catalog already owns the runtime API model source"
            )
        self._runtime_source.upsert(model)
        result = self._catalog.refresh(
            (self._runtime_source.source,), raise_on_error=True
        )
        if self._runtime_source.source not in result.refreshed:
            raise RuntimeError("runtime API model source was not refreshed")
        provider = _canonical_provider(model.provider)
        published = self.get_model(provider + "/" + model.model_id)
        if published is None:
            raise RuntimeError("runtime API model was not published")

    def add_model(self, model: APIModel) -> None:
        """Compatibility alias for :meth:`register_custom_model`."""

        self.register_custom_model(model)

    def export_models(self, include_deprecated: bool = True) -> List[Dict[str, Any]]:
        """Export the legacy list shape as JSON-compatible dictionaries."""

        return [
            model.to_dict()
            for model in self.get_all_models(
                include_deprecated=include_deprecated
            )
        ]

    def export_catalog(self) -> Dict[str, Any]:
        """Export the underlying canonical snapshot for migration tooling."""

        return self._catalog.snapshot().to_dict()

    export = export_models

    def get_backend_model_lists(self) -> Dict[str, List[str]]:
        """Project the former backend JSON inventories from catalog labels."""

        result: Dict[str, List[Tuple[int, str]]] = {}
        for descriptor, provider_name in self._ordered_descriptors():
            labels = _labels(descriptor)
            try:
                order = int(labels.get("legacy.order", "1000000000"))
            except ValueError:
                order = 1000000000
            backend_models = _decode_label(
                labels, "legacy.backend-models", {}
            )
            if not isinstance(backend_models, dict) or not backend_models:
                model = _descriptor_to_api_model(descriptor, provider_name)
                backend = _default_backend_for_model(model)
                prefix = "openvino" if provider_name == "ovms" else provider_name
                backend_models = {
                    backend: prefix + "/" + descriptor.name
                }
            for backend, qualified_name in backend_models.items():
                if isinstance(backend, str) and isinstance(qualified_name, str):
                    result.setdefault(backend, []).append(
                        (order, qualified_name)
                    )
        return {
            backend: [
                name for _, name in sorted(values, key=lambda item: item)
            ]
            for backend, values in sorted(result.items())
        }


def _rich_compatibility_models(provider: APIProviderType) -> List[APIModel]:
    result = []
    for row in API_MODEL_SEED_ROWS:
        if row.get("__provider_only__") is True:
            continue
        labels = row.get("labels", {})
        if not isinstance(labels, Mapping) or labels.get("legacy.rich") != "true":
            continue
        if _canonical_provider(row.get("provider")) != provider.value:
            continue
        result.append(
            APIModel(
                model_id=str(row["model_id"]),
                model_name=str(row.get("model_name") or row["model_id"]),
                provider=provider,
                pipeline_types=list(row.get("pipeline_types", ())),
                context_length=row.get("context_length"),
                supports_streaming=bool(row.get("supports_streaming", False)),
                cost_per_1k_tokens=(
                    None
                    if row.get("cost_per_1k_tokens") is None
                    else dict(row["cost_per_1k_tokens"])
                ),
                description=str(row.get("description", "")),
                is_multimodal=bool(row.get("is_multimodal", False)),
                vision_capable=bool(row.get("vision_capable", False)),
                function_calling=bool(row.get("function_calling", False)),
                json_mode=bool(row.get("json_mode", False)),
                deprecated=bool(row.get("deprecated", False)),
            )
        )
    return result


APIModelRegistry.OPENAI_MODELS = _rich_compatibility_models(APIProviderType.OPENAI)
APIModelRegistry.ANTHROPIC_MODELS = _rich_compatibility_models(
    APIProviderType.ANTHROPIC
)
APIModelRegistry.GOOGLE_MODELS = _rich_compatibility_models(APIProviderType.GOOGLE)
APIModelRegistry.GROQ_MODELS = _rich_compatibility_models(APIProviderType.GROQ)
APIModelRegistry.COHERE_MODELS = _rich_compatibility_models(APIProviderType.COHERE)


_global_registry: Optional[APIModelRegistry] = None
_global_registry_lock = threading.Lock()


def get_global_api_model_registry() -> APIModelRegistry:
    """Get the process-wide compatibility projection."""

    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = APIModelRegistry()
    return _global_registry


def get_api_models_for_pipeline(pipeline_type: str) -> List[APIModel]:
    """Return API models supporting a legacy pipeline type."""

    return get_global_api_model_registry().get_models_by_pipeline_type(
        pipeline_type
    )


def get_all_pipeline_types() -> Set[str]:
    """Return all legacy pipeline names as the established set shape."""

    return set(
        get_global_api_model_registry().get_supported_pipeline_types()
    )


__all__ = [
    "API_MODEL_SEED_ROWS",
    "APIModel",
    "APIModelRegistry",
    "APIProviderType",
    "LEGACY_REGISTRY_DEPRECATION",
    "PROVIDER_ALIASES",
    "RuntimeAPIModelCatalogSource",
    "get_all_pipeline_types",
    "get_api_model_seed_rows",
    "get_api_models_for_pipeline",
    "get_global_api_model_registry",
]
