#!/usr/bin/env python3
"""
Backward-compatibility tests for the tts_router shim.

tts_router.py was merged into voice_router.py.  The old module is now a thin
re-export shim.  These tests verify that every symbol that existed in the
original tts_router API still works correctly via the shim, and that providers
registered through the old API share state with the voice_router.
"""

import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _minimal_wav() -> bytes:
    """Return a minimal 44-byte WAV header (0 audio samples)."""
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


def test_shim_imports():
    """All public symbols can be imported from tts_router."""
    print("Testing tts_router shim imports...")
    try:
        from ipfs_accelerate_py.tts_router import (  # noqa: F401
            text_to_speech,
            get_tts_provider,
            register_tts_provider,
            clear_tts_router_caches,
            TTSProvider,
            VoiceProvider,
            ProviderInfo,
            ProviderFactory,
        )
        print("  ✓ All shim imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Shim import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shim_no_private_leak():
    """Private helpers must not be importable from the shim's __all__."""
    print("\nTesting that private helpers are not in tts_router.__all__...")
    try:
        import ipfs_accelerate_py.tts_router as shim
        all_names = shim.__all__
        private_names = [n for n in all_names if n.startswith("_")]
        assert private_names == [], f"Private names in __all__: {private_names}"
        print("  ✓ No private names in __all__")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ttsprovider_is_voiceprovider():
    """TTSProvider must be the same object as VoiceProvider."""
    print("\nTesting TTSProvider is VoiceProvider...")
    try:
        from ipfs_accelerate_py.tts_router import TTSProvider, VoiceProvider
        from ipfs_accelerate_py.voice_router import VoiceProvider as VP

        assert TTSProvider is VP, "TTSProvider should be VoiceProvider from voice_router"
        assert VoiceProvider is VP
        print("  ✓ TTSProvider is VoiceProvider")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aliases_are_same_objects():
    """All backward-compat aliases must point to the identical voice_router callables."""
    print("\nTesting alias identity with voice_router callables...")
    try:
        from ipfs_accelerate_py.tts_router import (
            get_tts_provider,
            register_tts_provider,
            clear_tts_router_caches,
        )
        from ipfs_accelerate_py.voice_router import (
            get_voice_provider,
            register_voice_provider,
            clear_voice_router_caches,
        )

        assert get_tts_provider is get_voice_provider
        assert register_tts_provider is register_voice_provider
        assert clear_tts_router_caches is clear_voice_router_caches
        print("  ✓ All aliases are identical to their voice_router counterparts")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_register_via_shim_shared_registry():
    """Providers registered via register_tts_provider are visible to get_voice_provider."""
    print("\nTesting shared registry across shim and voice_router...")
    try:
        from ipfs_accelerate_py.tts_router import register_tts_provider
        from ipfs_accelerate_py.voice_router import get_voice_provider, clear_voice_router_caches

        wav = _minimal_wav()

        class CompatProvider:
            def synthesize(self, text, *, voice=None, model_name=None,
                           device=None, output_format=None, **kwargs) -> bytes:
                return wav

            def transcribe(self, audio, *, model_name=None, language=None,
                           device=None, **kwargs) -> str:
                raise NotImplementedError

        clear_voice_router_caches()
        register_tts_provider("test_compat_registry", lambda: CompatProvider())

        provider = get_voice_provider("test_compat_registry", use_cache=False)
        assert provider is not None
        result = provider.synthesize("hello")
        assert result == wav
        print("  ✓ Provider registered via shim is visible to voice_router")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_to_speech_via_shim():
    """text_to_speech() imported from tts_router works end-to-end."""
    print("\nTesting text_to_speech() via tts_router shim...")
    try:
        from ipfs_accelerate_py.tts_router import text_to_speech, register_tts_provider
        from ipfs_accelerate_py.voice_router import clear_voice_router_caches

        wav = _minimal_wav()

        class ShimTTSProvider:
            def synthesize(self, text, *, voice=None, model_name=None,
                           device=None, output_format=None, **kwargs) -> bytes:
                return wav

            def transcribe(self, audio, **kwargs) -> str:
                raise NotImplementedError

        clear_voice_router_caches()
        register_tts_provider("test_shim_tts", lambda: ShimTTSProvider())

        audio = text_to_speech("Hello via shim", provider="test_shim_tts")
        assert isinstance(audio, bytes)
        assert audio == wav
        print("  ✓ text_to_speech() works via tts_router shim")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_top_level_package_aliases():
    """TTS aliases are available from the top-level ipfs_accelerate_py package."""
    print("\nTesting top-level package TTS aliases...")
    try:
        from ipfs_accelerate_py import (
            text_to_speech,
            get_tts_provider,
            register_tts_provider,
            clear_tts_router_caches,
            TTSProvider,
            tts_router_available,
            voice_router_available,
        )
        assert tts_router_available is voice_router_available, (
            "tts_router_available and voice_router_available should be in sync"
        )
        print(f"  ✓ Top-level TTS aliases available (tts_available={tts_router_available})")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_router_all_no_stdlib():
    """voice_router.__all__ must not contain stdlib module names."""
    print("\nTesting voice_router.__all__ does not leak stdlib names...")
    try:
        import ipfs_accelerate_py.voice_router as vr

        assert hasattr(vr, "__all__"), "voice_router should define __all__"

        stdlib_names = {
            "hashlib", "json", "os", "logging", "urllib", "dataclass",
            "lru_cache", "annotations", "Callable", "Dict", "Optional",
            "Protocol", "Union", "runtime_checkable",
        }
        leaked = [n for n in vr.__all__ if n in stdlib_names]
        assert leaked == [], f"stdlib names leaked into __all__: {leaked}"
        print(f"  ✓ __all__ is clean ({len(vr.__all__)} public names, no stdlib leaks)")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("TTS Router Backward-Compat Tests")
    print("=" * 60)

    tests = [
        test_shim_imports,
        test_shim_no_private_leak,
        test_ttsprovider_is_voiceprovider,
        test_aliases_are_same_objects,
        test_register_via_shim_shared_registry,
        test_text_to_speech_via_shim,
        test_top_level_package_aliases,
        test_voice_router_all_no_stdlib,
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
