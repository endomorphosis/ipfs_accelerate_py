"""Deterministic bounded SARIF projection for security and contract findings (VFS-030).

Exports a SARIF 2.1.0 subset that is:

* **Deterministic** — stable rule order, result order, and fingerprints;
* **Bounded** — hard caps on results, message bytes, and artifacts;
* **Reference-only** — artifact locations and content CIDs, never source
  bodies, AST dumps, proof traces, or secret values.

This module is a diagnostic projection.  It does not authorize repairs and
is not completion evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .proof.formal_verification_contracts import content_identity
from .security_contract_analysis import (
    ANALYZER_VERSION as SECURITY_ANALYZER_VERSION,
    FindingClassification,
    SecurityFinding,
    SecurityRuleFamily,
    security_rule_specs,
)


# ---------------------------------------------------------------------------
# Version / bounds / identity
# ---------------------------------------------------------------------------

FINDING_SARIF_VERSION: Final[int] = 1
SARIF_VERSION: Final[str] = "2.1.0"
SARIF_SCHEMA_URI: Final[str] = (
    "https://json.schemastore.org/sarif-2.1.0.json"
)
TOOL_NAME: Final[str] = "ipfs-accelerate-security-contract-analysis"
TOOL_INFORMATION_URI: Final[str] = (
    "https://github.com/endomorphosis/ipfs_accelerate_py"
)
DRIVER_SEMANTIC_VERSION: Final[str] = SECURITY_ANALYZER_VERSION

SARIF_PROJECTION_IS_COMPLETION_EVIDENCE: Final[bool] = False
SARIF_PROJECTION_AUTHORIZES_REPAIR: Final[bool] = False

DEFAULT_MAX_RESULTS: Final[int] = 256
DEFAULT_MAX_MESSAGE_BYTES: Final[int] = 2_048
DEFAULT_MAX_ARTIFACTS: Final[int] = 512
DEFAULT_MAX_RELATED: Final[int] = 32
MAX_RESULTS: Final[int] = 2_048
MAX_MESSAGE_BYTES: Final[int] = 8_192
MAX_ARTIFACTS: Final[int] = 4_096
MAX_SARIF_BYTES: Final[int] = 2_000_000

# Secret / body leakage patterns stripped from any free text that reaches SARIF.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9\-._~+/]+=*"),
    re.compile(r"-----BEGIN[A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?"
               r"-----END[A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)authorization\s*:\s*\S+"),
)

_FORBIDDEN_PROPERTY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_text",
        "source_body",
        "body",
        "code",
        "code_body",
        "secret",
        "secret_value",
        "password",
        "token",
        "api_key",
        "private_key",
        "payload_body",
        "raw_source",
        "ast_body",
        "proof_body",
        "witness_body",
        "snippet",
        "region_snippet",
    }
)

_SEVERITY_TO_LEVEL: Final[Mapping[str, str]] = {
    "info": "note",
    "low": "note",
    "medium": "warning",
    "high": "error",
    "critical": "error",
}

_CLASSIFICATION_TO_KIND: Final[Mapping[FindingClassification, str]] = {
    FindingClassification.VULNERABILITY: "fail",
    FindingClassification.CORRECTNESS_DRIFT: "review",
    FindingClassification.SUSPICION: "review",
    FindingClassification.UNKNOWN_DYNAMIC: "open",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FindingSarifError(ValueError):
    """Malformed SARIF projection input or bound violation."""


class FindingSarifBoundsError(FindingSarifError):
    """A SARIF bound was exceeded."""


class SecretLeakageError(FindingSarifError):
    """Projection attempted to emit secret or source-body material."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise FindingSarifError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise FindingSarifError(f"{field_name} must be a string")
    if "\x00" in value:
        raise FindingSarifError(f"{field_name} must not contain NUL")
    if required and not value.strip():
        raise FindingSarifError(f"{field_name} must be non-empty")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FindingSarifError(f"{field_name} must be an integer")
    if value < minimum:
        raise FindingSarifError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise FindingSarifBoundsError(
            f"{field_name} exceeds maximum {maximum}"
        )
    return value


def redact_text(value: str, *, maximum: int = DEFAULT_MAX_MESSAGE_BYTES) -> str:
    """Strip secret-like substrings and bound message length."""

    if not isinstance(value, str):
        raise FindingSarifError("text must be a string")
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    # Never emit multi-line private key remnants after partial matches.
    if "PRIVATE KEY" in redacted.upper() and "BEGIN" in redacted.upper():
        redacted = "[REDACTED]"
    encoded = redacted.encode("utf-8")
    if len(encoded) > maximum:
        # Truncate on character boundary.
        cut = redacted
        while len(cut.encode("utf-8")) > maximum - 3 and cut:
            cut = cut[:-1]
        redacted = cut + "..."
    return redacted


def _reject_body_properties(properties: Mapping[str, Any]) -> None:
    for key in properties:
        if str(key).lower() in _FORBIDDEN_PROPERTY_KEYS:
            raise SecretLeakageError(
                f"SARIF properties must not include body/secret field {key!r}"
            )


def _level_for_severity(severity: str) -> str:
    return _SEVERITY_TO_LEVEL.get((severity or "medium").lower(), "warning")


def _kind_for_classification(classification: FindingClassification | str) -> str:
    if isinstance(classification, FindingClassification):
        return _CLASSIFICATION_TO_KIND[classification]
    try:
        return _CLASSIFICATION_TO_KIND[FindingClassification(classification)]
    except ValueError:
        return "review"


def _finding_from_value(value: Any) -> SecurityFinding:
    if isinstance(value, SecurityFinding):
        return value
    if isinstance(value, Mapping):
        return SecurityFinding.from_dict(value)
    raise FindingSarifError(
        "finding must be a SecurityFinding or mapping"
    )


# ---------------------------------------------------------------------------
# Rule catalog projection
# ---------------------------------------------------------------------------


def sarif_rules(
    *,
    families: Sequence[SecurityRuleFamily | str] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic SARIF reportingDescriptor entries for rules."""

    specs = list(security_rule_specs())
    if families is not None:
        wanted = {
            f if isinstance(f, SecurityRuleFamily) else SecurityRuleFamily(f)
            for f in families
        }
        specs = [s for s in specs if s.family in wanted]
    rules: list[dict[str, Any]] = []
    for spec in specs:
        rules.append(
            {
                "id": spec.rule_id,
                "name": spec.name.replace(" ", "").replace("/", ""),
                "shortDescription": {
                    "text": redact_text(spec.name, maximum=512)
                },
                "fullDescription": {
                    "text": redact_text(spec.short_description, maximum=1024)
                },
                "defaultConfiguration": {
                    "level": _level_for_severity(spec.default_severity),
                },
                "properties": {
                    "family": spec.family.value,
                    "tags": ["security", "vfs", spec.family.value],
                    "precision": "high",
                    "problem.severity": spec.default_severity,
                },
            }
        )
    rules.sort(key=lambda r: r["id"])
    return rules


# ---------------------------------------------------------------------------
# Artifact + location helpers
# ---------------------------------------------------------------------------


def _artifact_location(
    *,
    uri: str = "",
    uri_base_id: str = "",
    description: str = "",
    index: int | None = None,
) -> dict[str, Any]:
    loc: dict[str, Any] = {}
    if uri:
        loc["uri"] = uri
    if uri_base_id:
        loc["uriBaseId"] = uri_base_id
    if index is not None:
        loc["index"] = index
    if description:
        loc["description"] = {"text": redact_text(description, maximum=256)}
    return loc


def _region(
    *,
    start_line: int = 0,
    start_column: int = 0,
    end_line: int = 0,
    end_column: int = 0,
) -> dict[str, Any] | None:
    if start_line <= 0 and end_line <= 0:
        return None
    region: dict[str, Any] = {}
    if start_line > 0:
        region["startLine"] = start_line
    if start_column > 0:
        region["startColumn"] = start_column
    if end_line > 0:
        region["endLine"] = end_line
    if end_column > 0:
        region["endColumn"] = end_column
    # Intentionally no "snippet" — source bodies must not leak.
    return region


def _physical_location(
    *,
    uri: str = "",
    uri_base_id: str = "",
    start_line: int = 0,
    start_column: int = 0,
    artifact_index: int | None = None,
) -> dict[str, Any] | None:
    if not uri and artifact_index is None:
        return None
    physical: dict[str, Any] = {
        "artifactLocation": _artifact_location(
            uri=uri,
            uri_base_id=uri_base_id,
            index=artifact_index,
        )
    }
    region = _region(start_line=start_line, start_column=start_column)
    if region is not None:
        physical["region"] = region
    return physical


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SarifExportConfig:
    """Bounds and metadata for a SARIF export."""

    max_results: int = DEFAULT_MAX_RESULTS
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    max_related: int = DEFAULT_MAX_RELATED
    tool_name: str = TOOL_NAME
    tool_version: str = DRIVER_SEMANTIC_VERSION
    automation_id: str = ""
    invocation_id: str = ""
    base_uri: str = "%SRCROOT%"
    include_partial_fingerprints: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_results",
            _integer(
                self.max_results,
                field_name="max_results",
                minimum=1,
                maximum=MAX_RESULTS,
            ),
        )
        object.__setattr__(
            self,
            "max_message_bytes",
            _integer(
                self.max_message_bytes,
                field_name="max_message_bytes",
                minimum=64,
                maximum=MAX_MESSAGE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_artifacts",
            _integer(
                self.max_artifacts,
                field_name="max_artifacts",
                minimum=1,
                maximum=MAX_ARTIFACTS,
            ),
        )
        object.__setattr__(
            self,
            "max_related",
            _integer(
                self.max_related,
                field_name="max_related",
                minimum=0,
                maximum=256,
            ),
        )
        object.__setattr__(
            self,
            "tool_name",
            _text(self.tool_name, field_name="tool_name"),
        )
        object.__setattr__(
            self,
            "tool_version",
            _text(self.tool_version, field_name="tool_version"),
        )
        for name in ("automation_id", "invocation_id", "base_uri"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )
        if not isinstance(self.include_partial_fingerprints, bool):
            raise FindingSarifError(
                "include_partial_fingerprints must be a boolean"
            )


# ---------------------------------------------------------------------------
# Result projection
# ---------------------------------------------------------------------------


def _evidence_artifact_uris(finding: SecurityFinding) -> list[str]:
    """Collect artifact reference URIs (cid:...) from evidence."""

    uris: list[str] = []
    ev = finding.evidence
    for cid in ev.artifact_cids:
        uris.append(f"cid:{cid}")
    for cid in ev.counterexample_cids:
        uris.append(f"cid:{cid}")
    for cid in ev.proof_cids:
        uris.append(f"cid:{cid}")
    for cid in ev.runtime_cids:
        uris.append(f"cid:{cid}")
    for cid in ev.graph_slice_cids:
        uris.append(f"cid:{cid}")
    return uris


def _primary_uri(finding: SecurityFinding) -> str:
    if finding.symbols:
        # Prefer a path-like symbol only when it looks like a path.
        for symbol in finding.symbols:
            if "/" in symbol or symbol.endswith(".py"):
                return symbol
    if finding.interfaces:
        return finding.interfaces[0]
    if finding.source_node_id:
        return f"node:{finding.source_node_id}"
    return f"finding:{finding.finding_id[:16]}"


def finding_to_sarif_result(
    finding: SecurityFinding | Mapping[str, Any],
    *,
    config: SarifExportConfig | None = None,
    artifact_index_by_uri: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Project one security finding into a SARIF result object."""

    cfg = config or SarifExportConfig()
    f = _finding_from_value(finding)
    message = redact_text(f.summary, maximum=cfg.max_message_bytes)
    if f.impact:
        impact = redact_text(f.impact, maximum=cfg.max_message_bytes)
        # Keep message bounded when combining.
        combined = f"{message} Impact: {impact}"
        message = redact_text(combined, maximum=cfg.max_message_bytes)

    uri = _primary_uri(f)
    art_index = None
    if artifact_index_by_uri is not None and uri in artifact_index_by_uri:
        art_index = artifact_index_by_uri[uri]

    locations: list[dict[str, Any]] = []
    physical = _physical_location(
        uri=uri,
        uri_base_id=cfg.base_uri if not uri.startswith("cid:") else "",
        artifact_index=art_index,
    )
    if physical is not None:
        locations.append({"physicalLocation": physical})

    # Related locations: threat path node ids as logical references only.
    related: list[dict[str, Any]] = []
    if f.threat_path is not None:
        for node_id in f.threat_path.node_ids[: cfg.max_related]:
            related.append(
                {
                    "id": len(related) + 1,
                    "message": {
                        "text": redact_text(
                            f"path-node:{node_id}", maximum=256
                        )
                    },
                    "physicalLocation": {
                        "artifactLocation": _artifact_location(
                            uri=f"node:{node_id}"
                        )
                    },
                }
            )

    properties: dict[str, Any] = {
        "classification": f.classification.value,
        "family": f.family.value,
        "root_cause_family": f.root_cause_family,
        "severity": f.severity,
        "confidence_millionths": f.confidence_millionths,
        "finding_id": f.finding_id,
        "seed_label": f.seed_label,
        "missing_requirements": list(f.missing_requirements),
        "symbols": list(f.symbols),
        "interfaces": list(f.interfaces),
        "repositories": list(f.repositories),
        "source_node_id": f.source_node_id,
        "sink_node_id": f.sink_node_id,
        "tree_id": f.tree_id,
        "policy_revision": f.policy_revision,
        "analyzer_version": f.analyzer_version,
        "evidence_artifact_cids": list(f.evidence.artifact_cids),
        "evidence_counterexample_cids": list(f.evidence.counterexample_cids),
        "evidence_proof_cids": list(f.evidence.proof_cids),
        "evidence_runtime_cids": list(f.evidence.runtime_cids),
        "evidence_graph_slice_cids": list(f.evidence.graph_slice_cids),
        "security_property_id": (
            f.security_property.property_id
            if f.security_property is not None
            else ""
        ),
        "threat_path_id": (
            f.threat_path.path_id if f.threat_path is not None else ""
        ),
        "threat_path_origin": (
            f.threat_path.origin.value if f.threat_path is not None else ""
        ),
        "has_unknown_dynamic": (
            f.threat_path.has_unknown_dynamic
            if f.threat_path is not None
            else False
        ),
        "is_vulnerability": f.is_vulnerability,
    }
    _reject_body_properties(properties)

    result: dict[str, Any] = {
        "ruleId": f.rule_id,
        "ruleIndex": None,  # filled by exporter
        "level": _level_for_severity(f.severity),
        "kind": _kind_for_classification(f.classification),
        "message": {"text": message},
        "locations": locations,
        "properties": properties,
    }
    if related:
        result["relatedLocations"] = related
    if cfg.include_partial_fingerprints:
        result["partialFingerprints"] = {
            "findingContentId": f.finding_id,
            "primaryLocationLineHash": content_identity(
                {
                    "rule": f.rule_id,
                    "source": f.source_node_id,
                    "sink": f.sink_node_id,
                    "family": f.family.value,
                }
            ),
        }
    # Stable GUID-like identity without leaking content.
    result["guid"] = f.finding_id[:36] if len(f.finding_id) >= 36 else f.finding_id
    result["correlationGuid"] = f.finding_id
    return result


# ---------------------------------------------------------------------------
# Full log export
# ---------------------------------------------------------------------------


def findings_to_sarif(
    findings: Sequence[SecurityFinding | Mapping[str, Any]],
    *,
    config: SarifExportConfig | None = None,
    run_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project findings into a deterministic, bounded SARIF 2.1.0 log.

    Returns a JSON-serializable dict.  Results are ordered by
    ``(ruleId, classification, finding_id)``.  Artifact entries reference
    paths and ``cid:`` URIs only — no embedded source or secret bodies.
    """

    cfg = config or SarifExportConfig()
    decoded = [_finding_from_value(f) for f in findings]
    decoded.sort(
        key=lambda f: (
            f.rule_id,
            f.classification.value,
            f.finding_id,
        )
    )

    truncated = False
    if len(decoded) > cfg.max_results:
        decoded = decoded[: cfg.max_results]
        truncated = True

    rules = sarif_rules()
    rule_index = {rule["id"]: index for index, rule in enumerate(rules)}

    # Collect artifacts: primary locations + evidence CIDs.
    artifact_uris: list[str] = []
    seen_uris: set[str] = set()

    def _add_uri(uri: str) -> None:
        nonlocal truncated
        if not uri or uri in seen_uris:
            return
        if len(artifact_uris) >= cfg.max_artifacts:
            truncated = True
            return
        seen_uris.add(uri)
        artifact_uris.append(uri)

    for finding in decoded:
        _add_uri(_primary_uri(finding))
        for uri in _evidence_artifact_uris(finding):
            _add_uri(uri)

    artifact_uris.sort()
    artifact_index_by_uri = {uri: i for i, uri in enumerate(artifact_uris)}

    artifacts: list[dict[str, Any]] = []
    for uri in artifact_uris:
        entry: dict[str, Any] = {
            "location": _artifact_location(
                uri=uri,
                uri_base_id=(
                    cfg.base_uri
                    if not uri.startswith(("cid:", "node:", "finding:"))
                    else ""
                ),
            ),
            "roles": (
                ["analysisTarget"]
                if not uri.startswith("cid:")
                else ["memoryContents"]
            ),
        }
        # Content reference only — never contents/rendered/text.
        if uri.startswith("cid:"):
            entry["contents"] = {
                # SARIF allows nesting; we store only the identifier.
                "binary": "",  # empty: body intentionally omitted
            }
            # Prefer properties over body for the CID.
            entry["properties"] = {
                "content_cid": uri[4:],
                "body_omitted": True,
            }
            # Remove empty binary to avoid confusion — use properties only.
            del entry["contents"]
        artifacts.append(entry)

    results: list[dict[str, Any]] = []
    for finding in decoded:
        result = finding_to_sarif_result(
            finding,
            config=cfg,
            artifact_index_by_uri=artifact_index_by_uri,
        )
        rid = result["ruleId"]
        result["ruleIndex"] = rule_index.get(rid)
        if result["ruleIndex"] is None:
            # Unknown rule — still emit, leave index unset.
            del result["ruleIndex"]
        results.append(result)

    run_props: dict[str, Any] = {
        "projection": "finding-sarif@1",
        "projection_version": FINDING_SARIF_VERSION,
        "is_completion_evidence": SARIF_PROJECTION_IS_COMPLETION_EVIDENCE,
        "authorizes_repair": SARIF_PROJECTION_AUTHORIZES_REPAIR,
        "truncated": truncated,
        "result_count": len(results),
        "artifact_count": len(artifacts),
        "deterministic": True,
        "bodies_omitted": True,
        "secrets_redacted": True,
    }
    if run_properties:
        _reject_body_properties(run_properties)
        for key, value in run_properties.items():
            if key in _FORBIDDEN_PROPERTY_KEYS:
                raise SecretLeakageError(
                    f"run_properties must not include {key!r}"
                )
            if isinstance(value, str):
                run_props[key] = redact_text(
                    value, maximum=cfg.max_message_bytes
                )
            elif isinstance(value, (int, float, bool)) or value is None:
                run_props[key] = value
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes)
            ):
                run_props[key] = [
                    redact_text(str(v), maximum=256)
                    if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                # Drop non-scalar unstructured blobs.
                run_props[key] = redact_text(str(value), maximum=256)

    tool: dict[str, Any] = {
        "driver": {
            "name": cfg.tool_name,
            "version": cfg.tool_version,
            "semanticVersion": cfg.tool_version,
            "informationUri": TOOL_INFORMATION_URI,
            "rules": rules,
            "properties": {
                "goal_id": "VFS-030",
                "analyzer": SECURITY_ANALYZER_VERSION,
            },
        }
    }

    run: dict[str, Any] = {
        "tool": tool,
        "results": results,
        "artifacts": artifacts,
        "columnKind": "utf16CodeUnits",
        "properties": run_props,
    }
    if cfg.automation_id:
        run["automationDetails"] = {"id": cfg.automation_id}
    if cfg.base_uri:
        run["originalUriBaseIds"] = {
            "SRCROOT": {"uri": cfg.base_uri if cfg.base_uri.endswith("/") else cfg.base_uri + "/"}
            if cfg.base_uri.startswith(("http://", "https://", "file:"))
            else {"description": {"text": cfg.base_uri}},
        }

    log: dict[str, Any] = {
        "$schema": SARIF_SCHEMA_URI,
        "version": SARIF_VERSION,
        "runs": [run],
    }

    # Enforce serialized byte bound.
    encoded = sarif_canonical_bytes(log)
    if len(encoded) > MAX_SARIF_BYTES:
        raise FindingSarifBoundsError(
            f"SARIF log exceeds {MAX_SARIF_BYTES} bytes"
        )
    return log


def sarif_canonical_bytes(log: Mapping[str, Any]) -> bytes:
    """Encode SARIF with sorted keys for deterministic digests."""

    return json.dumps(
        log,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sarif_content_id(log: Mapping[str, Any]) -> str:
    """Content identity of a SARIF log (deterministic)."""

    return content_identity(dict(log))


def assert_no_secret_or_body_leakage(log: Mapping[str, Any]) -> None:
    """Fail closed if a SARIF log appears to embed secrets or source bodies.

    Scans string leaves for private-key blocks and forbids known body
    property keys anywhere in the tree.
    """

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                key_s = str(key)
                if key_s.lower() in _FORBIDDEN_PROPERTY_KEYS:
                    # contents.binary empty is handled; still reject named keys.
                    if key_s.lower() in {"snippet", "source", "source_body", "body", "code"}:
                        raise SecretLeakageError(
                            f"forbidden key {key_s!r} at {path}"
                        )
                walk(value, f"{path}.{key_s}")
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")
        elif isinstance(obj, str):
            upper = obj.upper()
            if "BEGIN" in upper and "PRIVATE KEY" in upper:
                raise SecretLeakageError(
                    f"private key material at {path}"
                )
            # Long base64-looking blobs that look like embedded bodies.
            if len(obj) > 4096 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", obj or ""):
                raise SecretLeakageError(
                    f"oversized opaque blob at {path}"
                )

    walk(log, "$")


def export_security_findings_sarif(
    findings: Sequence[SecurityFinding | Mapping[str, Any]],
    *,
    config: SarifExportConfig | None = None,
    run_properties: Mapping[str, Any] | None = None,
    validate_leakage: bool = True,
) -> dict[str, Any]:
    """High-level export: findings → SARIF with optional leakage assertion."""

    log = findings_to_sarif(
        findings, config=config, run_properties=run_properties
    )
    if validate_leakage:
        assert_no_secret_or_body_leakage(log)
    return log


def contract_finding_to_security_like(
    record: Mapping[str, Any],
    *,
    family: SecurityRuleFamily | str = (
        SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION
    ),
    classification: FindingClassification | str = (
        FindingClassification.CORRECTNESS_DRIFT
    ),
) -> SecurityFinding:
    """Adapt a contract-finding-like mapping into a SecurityFinding shell.

    Used when projecting the contract finding ledger into SARIF without
    re-running dataflow analysis.  Does not upgrade correctness drift into
    a vulnerability.
    """

    from .security_contract_analysis import (
        SecurityEvidence,
        build_security_finding,
        security_rule_spec,
    )

    family_e = (
        family
        if isinstance(family, SecurityRuleFamily)
        else SecurityRuleFamily(family)
    )
    spec = security_rule_spec(family_e)
    evidence_raw = record.get("evidence") or {}
    if isinstance(evidence_raw, Mapping):
        evidence = SecurityEvidence(
            artifact_cids=tuple(evidence_raw.get("artifact_cids") or ()),
            counterexample_cids=tuple(
                evidence_raw.get("counterexample_cids") or ()
            ),
            proof_cids=tuple(evidence_raw.get("proof_cids") or ()),
            runtime_cids=tuple(evidence_raw.get("runtime_cids") or ()),
        )
    else:
        evidence = SecurityEvidence()

    summary = str(
        record.get("summary")
        or record.get("root_cause_family")
        or "contract finding"
    )
    return build_security_finding(
        family=family_e,
        classification=classification,
        summary=redact_text(summary, maximum=DEFAULT_MAX_MESSAGE_BYTES),
        impact="",
        evidence=evidence,
        severity=str(record.get("severity") or "low"),
        confidence_millionths=int(
            record.get("confidence_millionths") or 300_000
        ),
        symbols=tuple(record.get("symbols") or ()),
        interfaces=tuple(record.get("interfaces") or ()),
        repositories=tuple(record.get("repositories") or ()),
        tree_id=str(record.get("tree_id") or ""),
        policy_revision=str(record.get("policy_revision") or ""),
        root_cause_family=str(
            record.get("root_cause_family") or spec.family.value
        ),
        seed_label="contract_projection",
    )


__all__ = [
    "DEFAULT_MAX_ARTIFACTS",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_MAX_RESULTS",
    "DRIVER_SEMANTIC_VERSION",
    "FINDING_SARIF_VERSION",
    "FindingSarifBoundsError",
    "FindingSarifError",
    "MAX_SARIF_BYTES",
    "SARIF_PROJECTION_AUTHORIZES_REPAIR",
    "SARIF_PROJECTION_IS_COMPLETION_EVIDENCE",
    "SARIF_SCHEMA_URI",
    "SARIF_VERSION",
    "SecretLeakageError",
    "SarifExportConfig",
    "TOOL_NAME",
    "assert_no_secret_or_body_leakage",
    "contract_finding_to_security_like",
    "export_security_findings_sarif",
    "finding_to_sarif_result",
    "findings_to_sarif",
    "redact_text",
    "sarif_canonical_bytes",
    "sarif_content_id",
    "sarif_rules",
]
