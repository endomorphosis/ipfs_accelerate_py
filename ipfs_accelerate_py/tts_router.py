"""Text-to-speech router – backward-compatibility shim.

The TTS functionality has been merged into :mod:`ipfs_accelerate_py.voice_router`,
which now handles both text-to-speech (TTS) synthesis and speech-to-text (STT)
transcription in a single module.

All names that were previously exported by this module are re-exported here so
that existing code continues to work without any changes:

.. code-block:: python

    # Old import – still works:
    from ipfs_accelerate_py.tts_router import (
        text_to_speech, get_tts_provider, register_tts_provider,
        clear_tts_router_caches, TTSProvider,
    )

    # Preferred new import:
    from ipfs_accelerate_py.voice_router import (
        text_to_speech, get_voice_provider, register_voice_provider,
        clear_voice_router_caches, VoiceProvider,
    )
"""

from .voice_router import (  # noqa: F401
    text_to_speech,
    get_voice_provider as get_tts_provider,
    register_voice_provider as register_tts_provider,
    clear_voice_router_caches as clear_tts_router_caches,
    VoiceProvider as TTSProvider,
    VoiceProvider,
    ProviderInfo,
    ProviderFactory,
    _get_openai_provider,
    _get_elevenlabs_provider,
    _get_huggingface_provider,
    _get_backend_manager_provider,
)

__all__ = [
    "text_to_speech",
    "get_tts_provider",
    "register_tts_provider",
    "clear_tts_router_caches",
    "TTSProvider",
    "VoiceProvider",
    "ProviderInfo",
    "ProviderFactory",
]
