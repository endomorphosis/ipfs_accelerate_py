#!/usr/bin/env python3
"""
Example: Using the TTS Router (now part of voice_router)

This example demonstrates the text-to-speech functionality that integrates
multiple providers (OpenAI TTS, ElevenLabs, HuggingFace Bark) with the
existing endpoint multiplexing capabilities.

The TTS router has been merged into voice_router, which now handles both
text-to-speech (TTS) synthesis and speech-to-text (STT) transcription.
The legacy ``tts_router`` module re-exports everything from ``voice_router``
for backward compatibility.

The router automatically selects the best available provider based on:
- Environment configuration
- Available API keys
- Backend manager endpoints
- Fallback to local HuggingFace models (Bark, SpeechT5, etc.)
"""

import os
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py import (
        text_to_speech,
        get_tts_provider,
        register_tts_provider,
        RouterDeps,
        get_default_router_deps,
        tts_router_available,
    )
    from ipfs_accelerate_py.tts_router import TTSProvider
except ImportError as e:
    logger.error(f"Failed to import tts_router: {e}")
    logger.error("Make sure ipfs_accelerate_py is properly installed")
    exit(1)


def example_openai_tts():
    """Example 1: Text-to-speech via OpenAI TTS API."""
    print("\n=== Example 1: OpenAI TTS ===")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY")):
        print("OPENAI_API_KEY not set, skipping OpenAI TTS example")
        return

    text = "Hello! This is a test of the IPFS Accelerate text-to-speech router."

    try:
        audio_bytes = text_to_speech(
            text,
            provider="openai",
            voice="alloy",
            output_format="mp3",
        )
        print(f"Generated {len(audio_bytes)} bytes of MP3 audio")
        print(f"Text: {text}")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            print(f"Saved to: {tmp.name}")
    except Exception as e:
        print(f"Error: {e}")


def example_with_output_path():
    """Example 2: Save TTS output directly to a file."""
    print("\n=== Example 2: Save to File ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping this example")
        return

    text = "IPFS Accelerate makes distributed machine learning easy."

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        output_path = tmp.name

    try:
        result = text_to_speech(
            text,
            provider="openai",
            voice="nova",
            output_path=output_path,
        )
        print(f"Audio saved to: {result}")
        print(f"File size: {os.path.getsize(result)} bytes")
    except Exception as e:
        print(f"Error: {e}")


def example_elevenlabs_tts():
    """Example 3: Text-to-speech via ElevenLabs."""
    print("\n=== Example 3: ElevenLabs TTS ===")

    if not (os.getenv("ELEVENLABS_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY")):
        print("ELEVENLABS_API_KEY not set, skipping ElevenLabs example")
        return

    text = "ElevenLabs produces highly realistic voices."

    try:
        audio_bytes = text_to_speech(
            text,
            provider="elevenlabs",
        )
        print(f"Generated {len(audio_bytes)} bytes of audio via ElevenLabs")
    except Exception as e:
        print(f"Error: {e}")


def example_local_hf():
    """Example 4: Local HuggingFace TTS (Bark/SpeechT5)."""
    print("\n=== Example 4: Local HuggingFace TTS ===")

    text = "Hello from a local model."

    try:
        audio_bytes = text_to_speech(
            text,
            provider="huggingface",
            model_name="suno/bark-small",
        )
        print(f"Generated {len(audio_bytes)} bytes of WAV audio (local HF model)")
    except Exception as e:
        print(f"Error (local model not available): {e}")


def example_custom_provider():
    """Example 5: Registering a custom TTS provider."""
    print("\n=== Example 5: Custom Provider ===")

    class SilentTTSProvider:
        """A provider that returns a minimal valid WAV file for testing."""
        def synthesize(self, text: str, *, voice=None, model_name=None,
                       device=None, output_format=None, **kwargs) -> bytes:
            import struct
            # Minimal 44-byte WAV header for 0 samples
            num_channels = 1
            sample_rate = 22050
            bits_per_sample = 16
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8
            data_size = 0
            chunk_size = 36 + data_size
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", chunk_size, b"WAVE", b"fmt ", 16,
                1, num_channels, sample_rate, byte_rate,
                block_align, bits_per_sample,
                b"data", data_size,
            )
            return header

    register_tts_provider("silent", lambda: SilentTTSProvider())

    try:
        audio_bytes = text_to_speech(
            "This text is synthesized silently.",
            provider="silent",
        )
        print(f"Custom silent provider generated {len(audio_bytes)} bytes (WAV header only)")
    except Exception as e:
        print(f"Error: {e}")


def example_caching():
    """Example 6: Response caching."""
    print("\n=== Example 6: Response Caching ===")

    os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"

    # Register a counter provider to verify cache hits
    call_count = [0]

    class CountingTTSProvider:
        def synthesize(self, text: str, **kwargs) -> bytes:
            call_count[0] += 1
            return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22V\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    register_tts_provider("counting", lambda: CountingTTSProvider())

    text = "Cache test sentence."
    try:
        # First call (cache miss)
        audio1 = text_to_speech(text, provider="counting")
        # Second call (cache hit)
        audio2 = text_to_speech(text, provider="counting")
        print(f"Provider called {call_count[0]} time(s) for 2 requests (cache working: {call_count[0] == 1})")
        print(f"Audio bytes match: {audio1 == audio2}")
    except Exception as e:
        print(f"Error: {e}")


def example_list_available_providers():
    """Example 7: List all available providers."""
    print("\n=== Example 7: Available Providers ===")

    providers_to_check = [
        "openai",
        "elevenlabs",
        "huggingface",
        "backend_manager",
    ]

    print("Checking available providers:")
    for provider_name in providers_to_check:
        try:
            provider = get_tts_provider(provider_name)
            status = "✓ Available" if provider else "✗ Not available"
        except Exception as e:
            status = f"✗ Error: {str(e)[:60]}"
        print(f"  {provider_name:20} {status}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("TTS Router Examples")
    print("=" * 60)

    if not tts_router_available:
        print("ERROR: tts_router is not available")
        print("Make sure the module is properly installed")
        return

    print("\nTTS Router is available!")

    example_list_available_providers()
    example_openai_tts()
    example_with_output_path()
    example_elevenlabs_tts()
    example_local_hf()
    example_custom_provider()
    example_caching()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
