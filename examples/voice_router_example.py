#!/usr/bin/env python3
"""
Example: Using the Voice Router

This example demonstrates the voice_router functionality that integrates
multiple providers for both text-to-speech (TTS) synthesis and
speech-to-text (STT) transcription with the existing endpoint
multiplexing capabilities.

The router automatically selects the best available provider based on:
- Environment configuration
- Available API keys
- Backend manager endpoints
- Fallback to local HuggingFace models (Bark TTS, Whisper STT)
"""

import os
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py import (
        speech_to_text,
        get_voice_provider,
        register_voice_provider,
        RouterDeps,
        get_default_router_deps,
        voice_router_available,
    )
    from ipfs_accelerate_py.voice_router import VoiceProvider, text_to_speech
except ImportError as e:
    logger.error(f"Failed to import voice_router: {e}")
    logger.error("Make sure ipfs_accelerate_py is properly installed")
    exit(1)


def example_openai_tts():
    """Example 1: Text-to-speech via OpenAI TTS API."""
    print("\n=== Example 1: OpenAI TTS ===")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY")):
        print("OPENAI_API_KEY not set, skipping OpenAI TTS example")
        return

    text = "Hello! This is a test of the IPFS Accelerate voice router text-to-speech."

    try:
        audio_bytes = text_to_speech(
            text,
            provider="openai",
            voice="alloy",
            output_format="mp3",
        )
        print(f"Generated {len(audio_bytes)} bytes of MP3 audio")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            print(f"Saved to: {tmp.name}")
    except Exception as e:
        print(f"Error: {e}")


def example_openai_stt():
    """Example 2: Speech-to-text via OpenAI Whisper API."""
    print("\n=== Example 2: OpenAI Whisper STT ===")

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY")):
        print("OPENAI_API_KEY not set, skipping OpenAI STT example")
        return

    # First generate audio, then transcribe it
    try:
        audio_bytes = text_to_speech(
            "The quick brown fox jumps over the lazy dog.",
            provider="openai",
            voice="alloy",
            output_format="mp3",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        transcription = speech_to_text(audio_path, provider="openai")
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error: {e}")


def example_save_tts_to_file():
    """Example 3: Save TTS output directly to a file."""
    print("\n=== Example 3: Save TTS to File ===")

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
    """Example 4: Text-to-speech via ElevenLabs."""
    print("\n=== Example 4: ElevenLabs TTS ===")

    if not (os.getenv("ELEVENLABS_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY")):
        print("ELEVENLABS_API_KEY not set, skipping ElevenLabs example")
        return

    text = "ElevenLabs produces highly realistic voices."

    try:
        audio_bytes = text_to_speech(text, provider="elevenlabs")
        print(f"Generated {len(audio_bytes)} bytes of audio via ElevenLabs")
    except Exception as e:
        print(f"Error: {e}")


def example_assemblyai_stt():
    """Example 5: Speech-to-text via AssemblyAI."""
    print("\n=== Example 5: AssemblyAI STT ===")

    if not (os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY")):
        print("ASSEMBLYAI_API_KEY not set, skipping AssemblyAI example")
        return

    # Transcribe a public audio URL
    audio_url = "https://storage.googleapis.com/aai-docs-samples/nbc.mp3"

    try:
        transcription = speech_to_text(audio_url, provider="assemblyai")
        print(f"Transcription: {transcription[:200]}...")
    except Exception as e:
        print(f"Error: {e}")


def example_local_hf():
    """Example 6: Local HuggingFace TTS (Bark) + STT (Whisper)."""
    print("\n=== Example 6: Local HuggingFace TTS + STT ===")

    text = "Hello from a local model."

    try:
        audio_bytes = text_to_speech(
            text,
            provider="huggingface",
            model_name="suno/bark-small",
        )
        print(f"Generated {len(audio_bytes)} bytes of WAV audio (local HF TTS)")

        # Try STT round-trip
        transcription = speech_to_text(
            audio_bytes,
            provider="huggingface",
            model_name="openai/whisper-base",
        )
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error (local model not available): {e}")


def example_custom_provider():
    """Example 7: Registering a custom voice provider."""
    print("\n=== Example 7: Custom Provider ===")

    class EchoVoiceProvider:
        """A provider that returns a minimal WAV for TTS and echoes text for STT."""

        def synthesize(self, text: str, *, voice=None, model_name=None,
                       device=None, output_format=None, **kwargs) -> bytes:
            import struct
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

        def transcribe(self, audio, *, model_name=None, language=None,
                       device=None, **kwargs) -> str:
            return "echo transcription"

    register_voice_provider("echo", lambda: EchoVoiceProvider())

    try:
        audio_bytes = text_to_speech("Test synthesis.", provider="echo")
        print(f"Custom echo provider synthesize: {len(audio_bytes)} bytes")

        text = speech_to_text(audio_bytes, provider="echo")
        print(f"Custom echo provider transcribe: {text!r}")
    except Exception as e:
        print(f"Error: {e}")


def example_caching():
    """Example 8: Response caching for TTS."""
    print("\n=== Example 8: Response Caching ===")

    os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"

    call_count = [0]

    class CountingVoiceProvider:
        def synthesize(self, text: str, **kwargs) -> bytes:
            call_count[0] += 1
            return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22V\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        def transcribe(self, audio, **kwargs) -> str:
            return "cached transcription"

    register_voice_provider("counting", lambda: CountingVoiceProvider())

    text = "Cache test sentence."
    try:
        audio1 = text_to_speech(text, provider="counting")
        audio2 = text_to_speech(text, provider="counting")
        print(f"Provider called {call_count[0]} time(s) for 2 requests (cache working: {call_count[0] == 1})")
        print(f"Audio bytes match: {audio1 == audio2}")
    except Exception as e:
        print(f"Error: {e}")


def example_list_available_providers():
    """Example 9: List all available providers."""
    print("\n=== Example 9: Available Providers ===")

    providers_to_check = [
        "openai",
        "elevenlabs",
        "assemblyai",
        "huggingface",
        "backend_manager",
    ]

    print("Checking available providers:")
    for provider_name in providers_to_check:
        try:
            provider = get_voice_provider(provider_name)
            status = "✓ Available" if provider else "✗ Not available"
        except Exception as e:
            status = f"✗ Error: {str(e)[:60]}"
        print(f"  {provider_name:20} {status}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Voice Router Examples")
    print("=" * 60)

    if not voice_router_available:
        print("ERROR: voice_router is not available")
        print("Make sure the module is properly installed")
        return

    print("\nVoice Router is available!")

    example_list_available_providers()
    example_openai_tts()
    example_openai_stt()
    example_save_tts_to_file()
    example_elevenlabs_tts()
    example_assemblyai_stt()
    example_local_hf()
    example_custom_provider()
    example_caching()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
