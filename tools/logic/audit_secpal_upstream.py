#!/usr/bin/env python3
"""Explicit live audit of Microsoft's historical SecPAL distribution.

This is a provenance probe, not an installer and not a certification oracle.
It performs bounded GET requests only when explicitly invoked, follows HTTPS
redirects only within Microsoft-owned hostnames, and never downloads an
artifact candidate.  Offline capability discovery and certification must use
the reviewed deployment lock instead.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

SCHEMA_VERSION: Final = "secpal-upstream-live-audit/v1"
INTERFACE: Final = "SecPALUpstreamLiveAudit@1"
OFFICIAL_PROJECT_URL: Final = (
    "https://www.microsoft.com/en-us/research/project/secpal/"
)
OFFICIAL_DOWNLOAD_URL: Final = (
    "https://www.microsoft.com/en-us/download/details.aspx?id=52356"
)
HISTORICAL_RELEASE_VERSION: Final = "1.1"
OFFICIAL_BINARY_RELEASE_EVIDENCE_URL: Final = (
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2009/07/FinalTR.pdf"
)
HISTORICAL_VERSION_REFERENCE_URL: Final = (
    "https://era.ed.ac.uk/handle/1842/31341"
)
MAX_RESPONSE_BYTES: Final = 512 * 1024
DEFAULT_TIMEOUT_SECONDS: Final = 20.0
RETIRED_MARKER: Final = "this download is no longer available"
_ARTIFACT_URL_RE: Final = re.compile(
    r'https://download\.microsoft\.com/[^\s"\'<>]+\.(?:msi|zip|exe)(?:\?[^\s"\'<>]*)?',
    re.IGNORECASE,
)


class _Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def geturl(self) -> str: ...

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(self, *args: object) -> None: ...


class SecPALUpstreamAuditError(ValueError):
    """Raised when a live provenance probe violates its safety contract."""


def _is_microsoft_https_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    return parsed.scheme == "https" and (
        host == "microsoft.com" or host.endswith(".microsoft.com")
    )


class _MicrosoftOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect whose target is not Microsoft-owned HTTPS."""

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        if not _is_microsoft_https_url(newurl):
            raise SecPALUpstreamAuditError(
                "SecPAL upstream probe refused a redirect outside Microsoft: "
                f"{newurl!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_DEFAULT_OPENER: Final = urllib.request.build_opener(
    _MicrosoftOnlyRedirectHandler()
)


def _open_official(request: urllib.request.Request, *, timeout: float) -> _Response:
    return _DEFAULT_OPENER.open(request, timeout=timeout)


def _read_bounded(response: _Response) -> bytes:
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise SecPALUpstreamAuditError(
            f"official response exceeded {MAX_RESPONSE_BYTES} bytes"
        )
    return payload


def probe_official_page(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., _Response] = _open_official,
) -> dict[str, Any]:
    """Probe one Microsoft page and return bounded, secret-safe evidence."""

    if not _is_microsoft_https_url(url):
        raise SecPALUpstreamAuditError(
            f"SecPAL upstream probe refused non-Microsoft HTTPS URL: {url!r}"
        )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "ipfs-accelerate-secpal-provenance-audit/1",
        },
        method="GET",
    )
    try:
        response = opener(request, timeout=timeout_seconds)
        context = response
    except urllib.error.HTTPError as exc:
        # HTTPError carries the authoritative response status/body and is safe
        # to inspect under the same bounded response contract.
        context = exc
    except (OSError, urllib.error.URLError) as exc:
        return {
            "requested_url": url,
            "reachable": False,
            "status": None,
            "final_url": "",
            "microsoft_owned_final_url": False,
            "body_sha256": "",
            "response_bytes": 0,
            "retired_marker_present": False,
            "artifact_candidate_urls": [],
            "error": f"{type(exc).__name__}:{exc}",
        }

    with context as opened:
        final_url = opened.geturl()
        if not _is_microsoft_https_url(final_url):
            raise SecPALUpstreamAuditError(
                "SecPAL upstream probe followed a redirect outside Microsoft: "
                f"{final_url!r}"
            )
        body = _read_bounded(opened)
        status = int(getattr(opened, "status", 0) or opened.getcode())

    text = body.decode("utf-8", errors="replace")
    normalized = html.unescape(text)
    candidates = sorted(set(_ARTIFACT_URL_RE.findall(normalized)))
    return {
        "requested_url": url,
        "reachable": True,
        "status": status,
        "final_url": final_url,
        "microsoft_owned_final_url": True,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "response_bytes": len(body),
        "retired_marker_present": RETIRED_MARKER in normalized.casefold(),
        "artifact_candidate_urls": candidates,
        "error": "",
    }


def audit_secpal_upstream(
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., _Response] = _open_official,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Collect current official-page evidence without downloading binaries."""

    project = probe_official_page(
        OFFICIAL_PROJECT_URL,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    download = probe_official_page(
        OFFICIAL_DOWNLOAD_URL,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    candidates = list(download["artifact_candidate_urls"])
    official_distribution_retired = bool(
        download["status"] == 404 and download["retired_marker_present"]
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "observed_at": observed_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "historical_release_version": HISTORICAL_RELEASE_VERSION,
        "historical_version_reference_url": HISTORICAL_VERSION_REFERENCE_URL,
        "historical_version_evidence_class": "institutional_secondary_reference",
        "official_binary_release_evidence_url": (
            OFFICIAL_BINARY_RELEASE_EVIDENCE_URL
        ),
        "project_page": project,
        "download_page": download,
        "artifact_candidate_urls": candidates,
        "artifact_download_attempted": False,
        "official_distribution_retired": official_distribution_retired,
        "authoritative_artifact_available": bool(
            download["status"] == 200
            and not download["retired_marker_present"]
            and candidates
        ),
        "live_install_ready": False,
        "required_follow_up": [
            "recover an authentic Microsoft-published SecPAL 1.1 artifact",
            "bind its exact SHA-256 and embedded license text",
            "review its runtime, entrypoint, and platform contract",
            "implement and independently certify the genuine vendor adapter",
        ],
        "python_compatible_engine_is_shadow_only": True,
    }
    digest_basis = {
        key: value
        for key, value in payload.items()
        if key not in {"observed_at", "evidence_sha256"}
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            digest_basis,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; stdout is always emitted",
    )
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds <= 60:
        parser.error("--timeout-seconds must be in (0, 60]")
    result = audit_secpal_upstream(timeout_seconds=args.timeout_seconds)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["authoritative_artifact_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
