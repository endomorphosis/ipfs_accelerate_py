from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tools.logic import audit_secpal_upstream as audit


@dataclass
class _FakeResponse:
    url: str
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def geturl(self) -> str:
        return self.url

    def getcode(self) -> int:
        return self.status

    def read(self, _amount: int = -1) -> bytes:
        return self.body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_live_audit_records_retired_official_distribution_without_downloading() -> None:
    calls: list[tuple[str, float]] = []

    def opener(request, *, timeout: float):
        calls.append((request.full_url, timeout))
        if request.full_url == audit.OFFICIAL_PROJECT_URL:
            return _FakeResponse(request.full_url, 404, b"project not found")
        return _FakeResponse(
            "https://www.microsoft.com/en-us/download/404-page",
            404,
            b"We're sorry, this download is no longer available.",
        )

    result = audit.audit_secpal_upstream(
        opener=opener,
        observed_at="2026-08-03T00:00:00+00:00",
    )

    assert calls == [
        (audit.OFFICIAL_PROJECT_URL, audit.DEFAULT_TIMEOUT_SECONDS),
        (audit.OFFICIAL_DOWNLOAD_URL, audit.DEFAULT_TIMEOUT_SECONDS),
    ]
    assert result["historical_release_version"] == "1.1"
    assert result["historical_version_evidence_class"] == (
        "institutional_secondary_reference"
    )
    assert result["official_binary_release_evidence_url"].startswith(
        "https://www.microsoft.com/"
    )
    assert result["official_distribution_retired"] is True
    assert result["authoritative_artifact_available"] is False
    assert result["artifact_download_attempted"] is False
    assert result["artifact_candidate_urls"] == []
    assert result["live_install_ready"] is False
    assert result["python_compatible_engine_is_shadow_only"] is True
    assert len(result["evidence_sha256"]) == 64


def test_live_audit_never_downloads_discovered_artifact_candidate() -> None:
    candidate = "https://download.microsoft.com/download/a/b/c/SecPAL-1.1.msi"
    calls: list[str] = []

    def opener(request, *, timeout: float):
        assert timeout == audit.DEFAULT_TIMEOUT_SECONDS
        calls.append(request.full_url)
        return _FakeResponse(request.full_url, 200, candidate.encode("utf-8"))

    result = audit.audit_secpal_upstream(opener=opener)

    assert calls == [audit.OFFICIAL_PROJECT_URL, audit.OFFICIAL_DOWNLOAD_URL]
    assert result["artifact_candidate_urls"] == [candidate]
    assert result["artifact_download_attempted"] is False
    # Discovery is not installation readiness: checksum, license, runtime,
    # executable semantics, and platform evidence remain separate gates.
    assert result["live_install_ready"] is False


def test_probe_rejects_non_microsoft_input_and_redirect_targets() -> None:
    with pytest.raises(audit.SecPALUpstreamAuditError, match="non-Microsoft"):
        audit.probe_official_page(
            "https://example.invalid/secpal",
            opener=lambda *_args, **_kwargs: None,
        )

    def external_redirect(_request, *, timeout: float):
        assert timeout == audit.DEFAULT_TIMEOUT_SECONDS
        return _FakeResponse("https://example.invalid/payload", 200, b"payload")

    with pytest.raises(audit.SecPALUpstreamAuditError, match="outside Microsoft"):
        audit.probe_official_page(
            audit.OFFICIAL_DOWNLOAD_URL,
            opener=external_redirect,
        )

    handler = audit._MicrosoftOnlyRedirectHandler()
    with pytest.raises(audit.SecPALUpstreamAuditError, match="outside Microsoft"):
        handler.redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://example.invalid/payload",
        )


def test_probe_rejects_oversized_official_response() -> None:
    def oversized(request, *, timeout: float):
        assert timeout == audit.DEFAULT_TIMEOUT_SECONDS
        return _FakeResponse(
            request.full_url,
            200,
            b"x" * (audit.MAX_RESPONSE_BYTES + 1),
        )

    with pytest.raises(audit.SecPALUpstreamAuditError, match="exceeded"):
        audit.probe_official_page(
            audit.OFFICIAL_DOWNLOAD_URL,
            opener=oversized,
        )
