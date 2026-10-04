"""Compatibility target for the voice-router regression suite.

The catalog task's validation contract names this historical path, while the
offline regression tests live in ``test_voice_router_integration.py``.
"""

import pytest

from test.test_voice_router_integration import (
    test_imports as _test_imports,
    test_output_path as _test_output_path,
    test_provider_registry as _test_provider_registry,
    test_response_caching as _test_response_caching,
    test_router_deps as _test_router_deps,
    test_speech_to_text_custom_provider as _test_speech_to_text,
    test_text_to_speech_custom_provider as _test_text_to_speech,
    test_unknown_provider_raises as _test_unknown_provider,
    test_voice_provider_protocol as _test_provider_protocol,
)


@pytest.mark.parametrize(
    "regression",
    (
        _test_imports,
        _test_output_path,
        _test_provider_registry,
        _test_response_caching,
        _test_router_deps,
        _test_speech_to_text,
        _test_text_to_speech,
        _test_unknown_provider,
        _test_provider_protocol,
    ),
    ids=lambda regression: regression.__name__.removeprefix("test_"),
)
def test_voice_router_regression(regression):
    """Require the legacy boolean-style checks to report success."""
    assert regression() is True
