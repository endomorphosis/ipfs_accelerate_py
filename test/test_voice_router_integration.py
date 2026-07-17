#!/usr/bin/env python3
"""
Integration tests for the Voice Router.

Tests basic functionality without requiring actual API keys or external requests.
"""

import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _minimal_wav() -> bytes:
    """Return a minimal 44-byte WAV file (0 samples, valid header)."""
    num_channels = 1
    sample_rate = 22050
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0
    chunk_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16,
        1, num_channels, sample_rate, byte_rate,
        block_align, bits_per_sample,
        b"data", data_size,
    )


def test_imports():
    """Test that all public symbols import correctly."""
    print("Testing imports...")
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
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_deps():
    """Test RouterDeps functionality."""
    print("\nTesting RouterDeps...")
    try:
        from ipfs_accelerate_py.router_deps import RouterDeps, get_default_router_deps

        deps = RouterDeps()
        deps.set_cached("test_key", "test_value")
        assert deps.get_cached("test_key") == "test_value", "Cache get/set failed"

        result = deps.get_or_create("new_key", lambda: "created_value")
        assert result == "created_value", "get_or_create failed"

        default1 = get_default_router_deps()
        default2 = get_default_router_deps()
        assert default1 is default2, "Default deps should be singleton"

        print("  ✓ RouterDeps tests passed")
        return True
    except Exception as e:
        print(f"  ✗ RouterDeps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_registry():
    """Test provider registration and retrieval."""
    print("\nTesting provider registry...")
    try:
        from ipfs_accelerate_py.voice_router import (
            register_voice_provider,
            get_voice_provider,
            clear_voice_router_caches,
        )

        wav = _minimal_wav()

        class FixedVoiceProvider:
            def synthesize(self, text, *, voice=None, model_name=None,
                           device=None, output_format=None, **kwargs) -> bytes:
                return wav

            def transcribe(self, audio, *, model_name=None, language=None,
                           device=None, **kwargs) -> str:
                return f"transcribed: {len(audio) if isinstance(audio, bytes) else audio}"

        clear_voice_router_caches()
        register_voice_provider("test_voice_registry", lambda: FixedVoiceProvider())

        provider = get_voice_provider("test_voice_registry", use_cache=False)
        assert provider is not None

        result = provider.synthesize("hello")
        assert result == wav, "synthesize() should return the fixture WAV"

        transcript = provider.transcribe(b"\x00\x01\x02")
        assert isinstance(transcript, str)
        assert "transcribed" in transcript

        print("  ✓ Provider registry tests passed")
        return True
    except Exception as e:
        print(f"  ✗ Provider registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_to_speech_custom_provider():
    """Test the text_to_speech() convenience function with a custom provider."""
    print("\nTesting text_to_speech() with custom provider...")
    try:
        from ipfs_accelerate_py.voice_router import (
            register_voice_provider,
            text_to_speech,
            clear_voice_router_caches,
        )

        wav = _minimal_wav()

        class SimpleTTSProvider:
            def synthesize(self, text, *, voice=None, model_name=None,
                           device=None, output_format=None, **kwargs) -> bytes:
                return wav

            def transcribe(self, audio, **kwargs) -> str:
                raise NotImplementedError

        clear_voice_router_caches()
        register_voice_provider("test_tts_fn", lambda: SimpleTTSProvider())

        audio = text_to_speech("Hello world", provider="test_tts_fn")
        assert isinstance(audio, bytes), "Expected bytes"
        assert audio == wav

        print("  ✓ text_to_speech() tests passed")
        return True
    except Exception as e:
        print(f"  ✗ text_to_speech() test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speech_to_text_custom_provider():
    """Test the speech_to_text() convenience function with a custom provider."""
    print("\nTesting speech_to_text() with custom provider...")
    try:
        from ipfs_accelerate_py import speech_to_text
        from ipfs_accelerate_py.voice_router import register_voice_provider, clear_voice_router_caches

        class SimpleSTTProvider:
            def synthesize(self, text, **kwargs) -> bytes:
                raise NotImplementedError

            def transcribe(self, audio, *, model_name=None, language=None,
                           device=None, **kwargs) -> str:
                return "hello world"

        clear_voice_router_caches()
        register_voice_provider("test_stt_fn", lambda: SimpleSTTProvider())

        result = speech_to_text(b"\x00\x01", provider="test_stt_fn")
        assert result == "hello world", f"Expected 'hello world', got {result!r}"

        print("  ✓ speech_to_text() tests passed")
        return True
    except Exception as e:
        print(f"  ✗ speech_to_text() test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_path():
    """Test that text_to_speech() writes to a file when output_path is given."""
    print("\nTesting output_path parameter...")
    import tempfile
    try:
        from ipfs_accelerate_py.voice_router import (
            register_voice_provider,
            text_to_speech,
            clear_voice_router_caches,
        )

        wav = _minimal_wav()

        class FileTTSProvider:
            def synthesize(self, text, **kwargs) -> bytes:
                return wav

            def transcribe(self, audio, **kwargs) -> str:
                raise NotImplementedError

        clear_voice_router_caches()
        register_voice_provider("test_file_tts", lambda: FileTTSProvider())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        result = text_to_speech("Save to file", provider="test_file_tts", output_path=output_path)
        assert result == output_path, "Should return the output path string"

        with open(output_path, "rb") as fh:
            written = fh.read()
        assert written == wav, "Written file content should match"

        os.unlink(output_path)
        print("  ✓ output_path tests passed")
        return True
    except Exception as e:
        print(f"  ✗ output_path test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_caching():
    """Test that response caching works (provider called only once for same input)."""
    print("\nTesting response caching...")
    try:
        from ipfs_accelerate_py.voice_router import (
            register_voice_provider,
            text_to_speech,
            clear_voice_router_caches,
        )
        from ipfs_accelerate_py.router_deps import RouterDeps

        os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"

        call_count = [0]
        wav = _minimal_wav()

        class CountingProvider:
            def synthesize(self, text, **kwargs) -> bytes:
                call_count[0] += 1
                return wav

            def transcribe(self, audio, **kwargs) -> str:
                return "counted"

        clear_voice_router_caches()
        register_voice_provider("test_cache_voice", lambda: CountingProvider())

        # Use a shared RouterDeps so the response cache is shared across calls.
        deps = RouterDeps()
        text = "cache test sentence unique 42"

        audio1 = text_to_speech(text, provider="test_cache_voice", deps=deps)
        # Second call with identical args should hit the response cache.
        audio2 = text_to_speech(text, provider="test_cache_voice", deps=deps)

        assert audio1 == audio2 == wav
        assert call_count[0] == 1, f"Expected 1 provider call (cache hit on 2nd), got {call_count[0]}"

        print(f"  ✓ Response caching tests passed ({call_count[0]} provider call for 2 requests)")
        return True
    except Exception as e:
        print(f"  ✗ Response caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unknown_provider_raises():
    """Test that requesting an unknown provider raises ValueError."""
    print("\nTesting unknown provider error handling...")
    try:
        from ipfs_accelerate_py.voice_router import get_voice_provider, clear_voice_router_caches

        clear_voice_router_caches()
        try:
            get_voice_provider("nonexistent_xyz_provider_12345", use_cache=False)
            print("  ✗ Expected ValueError not raised")
            return False
        except ValueError as e:
            assert "nonexistent_xyz_provider_12345" in str(e)
            print("  ✓ Unknown provider raises ValueError as expected")
            return True
    except Exception as e:
        print(f"  ✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_provider_protocol():
    """Test that VoiceProvider is a runtime-checkable Protocol."""
    print("\nTesting VoiceProvider protocol...")
    try:
        from ipfs_accelerate_py.voice_router import VoiceProvider

        class GoodProvider:
            def synthesize(self, text, *, voice=None, model_name=None,
                           device=None, output_format=None, **kwargs) -> bytes:
                return b""

            def transcribe(self, audio, *, model_name=None, language=None,
                           device=None, **kwargs) -> str:
                return ""

        assert isinstance(GoodProvider(), VoiceProvider), "GoodProvider should satisfy VoiceProvider"
        print("  ✓ VoiceProvider protocol tests passed")
        return True
    except Exception as e:
        print(f"  ✗ VoiceProvider protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Voice Router Integration Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_router_deps,
        test_provider_registry,
        test_text_to_speech_custom_provider,
        test_speech_to_text_custom_provider,
        test_output_path,
        test_response_caching,
        test_unknown_provider_raises,
        test_voice_provider_protocol,
    ]

    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests PASSED ✓")
    else:
        failed = [t.__name__ for t, ok in zip(tests, results) if not ok]
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
