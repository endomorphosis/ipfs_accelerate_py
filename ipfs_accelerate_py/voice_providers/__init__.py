"""Optional voice-provider adapters.

Provider modules in this package are dependency-light and are imported lazily
by :mod:`ipfs_accelerate_py.voice_router`.
"""

from .abby import (
    AbbyCircuitOpenError,
    AbbyIndexTTSProvider,
    AbbyProviderError,
    AbbyProviderReceipt,
    AbbyResiliencePolicy,
    AbbyWhisperProvider,
    HTTPRequest,
    HTTPResponse,
    HuggingFaceWhisperHTTPProvider,
    IndexTTSHTTPProvider,
    PUBLICUS_INDEXTTS_BATCH_API_NAME,
    PUBLICUS_INDEXTTS_BATCH_FN_INDEX,
    PUBLICUS_INDEXTTS_INPUT_COUNT,
    PUBLICUS_INDEXTTS_MODEL,
    PUBLICUS_INDEXTTS_SINGLE_API_NAME,
    PUBLICUS_INDEXTTS_SINGLE_FN_INDEX,
    PUBLICUS_INDEXTTS_SPACE_URL,
    PUBLICUS_INDEXTTS_TIMEOUT_SECONDS,
    PublicusIndexTTSProvider,
)

__all__ = [
    "AbbyCircuitOpenError",
    "AbbyIndexTTSProvider",
    "AbbyProviderError",
    "AbbyProviderReceipt",
    "AbbyResiliencePolicy",
    "AbbyWhisperProvider",
    "HTTPRequest",
    "HTTPResponse",
    "HuggingFaceWhisperHTTPProvider",
    "IndexTTSHTTPProvider",
    "PUBLICUS_INDEXTTS_BATCH_API_NAME",
    "PUBLICUS_INDEXTTS_BATCH_FN_INDEX",
    "PUBLICUS_INDEXTTS_INPUT_COUNT",
    "PUBLICUS_INDEXTTS_MODEL",
    "PUBLICUS_INDEXTTS_SINGLE_API_NAME",
    "PUBLICUS_INDEXTTS_SINGLE_FN_INDEX",
    "PUBLICUS_INDEXTTS_SPACE_URL",
    "PUBLICUS_INDEXTTS_TIMEOUT_SECONDS",
    "PublicusIndexTTSProvider",
]
