"""Published and conservative provider limit/pricing hints.

Values are operator-facing catalog facts, not live account quotas. Unknown
fields stay None so allocation never invents unlimited capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

_STANDARD_LIMITS = {"rpm": 3_000, "tpm": 4_000_000}
_STANDARD_PRICING = {"cached_input": 0.15, "input": 1.25, "output": 4.25}
_CONTRIBUTOR_LIMITS = {"rpm": 100, "tpm": 3_000_000}
_CONTRIBUTOR_PRICING = {"cached_input": 0.002, "input": 0.10, "output": 0.20}


@dataclass(frozen=True)
class ProviderLimitHint:
    provider: str
    protocol: str
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    input_usd_per_1m: Optional[float] = None
    output_usd_per_1m: Optional[float] = None
    cached_input_usd_per_1m: Optional[float] = None
    source: str = "catalog"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROVIDER_LIMIT_HINTS: dict[str, ProviderLimitHint] = {
    "meta_ai": ProviderLimitHint(
        provider="meta_ai",
        protocol="http",
        rpm=int(_STANDARD_LIMITS["rpm"]),
        tpm=int(_STANDARD_LIMITS["tpm"]),
        input_usd_per_1m=float(_STANDARD_PRICING["input"]),
        output_usd_per_1m=float(_STANDARD_PRICING["output"]),
        cached_input_usd_per_1m=float(_STANDARD_PRICING["cached_input"]),
        source="meta-pricing-rate-limits",
    ),
    "muse_code": ProviderLimitHint(
        provider="muse_code",
        protocol="cli",
        rpm=int(_STANDARD_LIMITS["rpm"]),
        tpm=int(_STANDARD_LIMITS["tpm"]),
        input_usd_per_1m=float(_STANDARD_PRICING["input"]),
        output_usd_per_1m=float(_STANDARD_PRICING["output"]),
        cached_input_usd_per_1m=float(_STANDARD_PRICING["cached_input"]),
        source="meta-pricing-rate-limits",
    ),
    "openai": ProviderLimitHint(
        provider="openai",
        protocol="http",
        source="unknown",
    ),
    "openrouter": ProviderLimitHint(
        provider="openrouter",
        protocol="http",
        source="unknown",
    ),
    "xai": ProviderLimitHint(
        provider="xai",
        protocol="http",
        source="unknown",
    ),
    "grok_cli": ProviderLimitHint(
        provider="grok_cli",
        protocol="cli",
        source="unknown",
    ),
    "codex_cli": ProviderLimitHint(
        provider="codex_cli",
        protocol="cli",
        source="unknown",
    ),
    "copilot_cli": ProviderLimitHint(
        provider="copilot_cli",
        protocol="cli",
        source="unknown",
    ),
    "claude_code": ProviderLimitHint(
        provider="claude_code",
        protocol="cli",
        source="unknown",
    ),
    "gemini_cli": ProviderLimitHint(
        provider="gemini_cli",
        protocol="cli",
        source="unknown",
    ),
    "mistral_vibe": ProviderLimitHint(
        provider="mistral_vibe",
        protocol="cli",
        source="unknown",
    ),
    "goose_cli": ProviderLimitHint(
        provider="goose_cli",
        protocol="cli",
        source="backend-dependent",
    ),
}


def limit_hint_for(provider: str, *, model_name: str = "") -> ProviderLimitHint:
    key = str(provider or "").strip().lower().replace("-", "_")
    hint = PROVIDER_LIMIT_HINTS.get(key)
    if key in {"meta_ai", "muse_code"} and str(model_name).endswith("-contributor"):
        return ProviderLimitHint(
            provider=key,
            protocol="http" if key == "meta_ai" else "cli",
            rpm=int(_CONTRIBUTOR_LIMITS["rpm"]),
            tpm=int(_CONTRIBUTOR_LIMITS["tpm"]),
            input_usd_per_1m=float(_CONTRIBUTOR_PRICING["input"]),
            output_usd_per_1m=float(_CONTRIBUTOR_PRICING["output"]),
            cached_input_usd_per_1m=float(_CONTRIBUTOR_PRICING["cached_input"]),
            source="meta-pricing-rate-limits-contributor",
        )
    if hint is not None:
        return hint
    return ProviderLimitHint(provider=key or "unknown", protocol="unknown", source="unknown")
