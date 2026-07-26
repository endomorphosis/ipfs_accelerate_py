"""Canonical task-type names shared by task workers and mesh services.

The queue stores task types as strings and older clients use several spellings
for the same operation.  Keep the compatibility aliases here so capability
advertisement and handler dispatch cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable


_TASK_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "text-generation": ("text-generation", "text_generation", "generation"),
    "text2text-generation": ("text2text-generation", "text2text_generation", "text2text"),
    "embedding": ("embedding", "embeddings", "text-embedding", "text_embedding", "text_embeddings"),
    "text-classification": ("text-classification", "text_classification"),
    "hf.pipeline": ("hf.pipeline", "hf_pipeline"),
    "llm.generate": ("llm.generate", "llm_generate"),
    "multimodal-generation": (
        "multimodal-generation",
        "multimodal_generation",
        "vision-generation",
        "vision_generation",
    ),
    "tool.call": ("tool.call", "tool"),
    "voice.tts": (
        "voice.tts",
        "voice_tts",
        "voice-tts",
        "tts",
        "text-to-speech",
        "text_to_speech",
    ),
    "voice.asr": (
        "voice.asr",
        "voice_asr",
        "voice-asr",
        "voice.stt",
        "voice_stt",
        "voice-stt",
        "asr",
        "stt",
        "speech-to-text",
        "speech_to_text",
        "automatic-speech-recognition",
        "automatic_speech_recognition",
        "automatic-speech-recognition",
        "automatic_speech_recognition",
    ),
    "voice.audio-validate": (
        "voice.audio-validate",
        "voice.audio_validate",
        "voice_audio_validate",
        "voice-audio-validate",
        "audio-validate",
        "audio_validate",
        "audio-validation",
        "audio_validation",
        "audio-validation",
        "audio_validation",
    ),
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _TASK_TYPE_ALIASES.items()
    for alias in aliases
}

VOICE_TASK_TYPES: tuple[str, ...] = (
    "voice.tts",
    "voice.asr",
    "voice.audio-validate",
)


def canonical_task_type(task_type: object) -> str:
    """Return the stable task name for a canonical name or known alias."""

    value = str(task_type or "").strip().lower()
    return _ALIAS_TO_CANONICAL.get(value, value)


def task_type_aliases(task_type: object) -> tuple[str, ...]:
    """Return all queue-compatible spellings for ``task_type``."""

    canonical = canonical_task_type(task_type)
    return _TASK_TYPE_ALIASES.get(canonical, (canonical,) if canonical else ())


def normalize_task_types(
    task_types: Iterable[object],
    *,
    expand_aliases: bool = True,
) -> list[str]:
    """Normalize and de-duplicate task names while preserving stable order.

    Alias expansion is required when querying existing queues because tasks
    submitted by older clients retain their original task-type spelling.
    """

    out: list[str] = []
    seen: set[str] = set()
    for raw in task_types:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        values = task_type_aliases(value) if expand_aliases else (canonical_task_type(value),)
        for normalized in values:
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
    return out
