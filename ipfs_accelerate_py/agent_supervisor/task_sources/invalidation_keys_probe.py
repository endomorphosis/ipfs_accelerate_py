"""Execute invalidation keys without compiling or deleting artifacts.

A changed review pin and a changed applicability pin stale the eligibility
artifact and its user. The conditional compiled artifact stays reusable and
its bytes stay put. Adding a competing definition stales the prior user and
the reverse dependent. A cache pin without those bytes is refused. No
incremental compiler and no Lake build run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dependency_block_resolution import EXPECTED_INVALIDATION_KEYS_PROBE

_COMPILED = b"conditional-compiled-artifact-v1"
_REVIEW_A = "11" * 32
_REVIEW_B = "22" * 32
_APPLICABILITY_A = "33" * 32
_APPLICABILITY_B = "44" * 32
_CACHE_REFUSAL = (
    "previous cache, independent pin and complete artifacts must be supplied together"
)


def measure_invalidation_keys_probe() -> dict[str, object] | None:
    """Return the executed invalidation, or None when a check drifts."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_INVALIDATION_KEYS_PROBE:
        return None
    return measured


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _table(definitions: list[dict[str, str]]) -> bytes:
    return json.dumps(
        {"definitions": definitions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _dependencies():
    from ipfs_datasets_py.logic.autoformal import legal_graph_dependencies

    return legal_graph_dependencies


def _incremental():
    from ipfs_datasets_py.logic.autoformal import legal_corpus_incremental

    return legal_corpus_incremental


def _artifact(subject, key: str, digest: str, **pins: str) -> dict:
    declared = {kind: [] for kind in subject.PIN_KINDS}
    declared.update(
        {
            kind: [{"key": f"{kind}:{key}", "sha256": value}]
            for kind, value in pins.items()
        }
    )
    return {
        "artifact_id": key,
        "artifact_sha256": digest,
        "dependency_pins": declared,
    }


def _manifest(subject, rows: list[dict], edges: list[tuple[str, str]]) -> dict:
    return subject.build_dependency_manifest(
        rows,
        [
            {"dependent_id": dependent, "dependency_id": dependency}
            for dependent, dependency in edges
        ],
        expected_artifact_ids=[row["artifact_id"] for row in rows],
    )


def _observations(subject, manifest: dict) -> dict:
    pins = {kind: {} for kind in subject.PIN_KINDS}
    for row in manifest["artifacts"]:
        for kind in subject.PIN_KINDS:
            pins[kind].update(
                {pin["key"]: pin["sha256"] for pin in row["dependency_pins"][kind]}
            )
    return {
        "expected_manifest_sha256": manifest["manifest_sha256"],
        "observed_artifact_sha256s": {
            row["artifact_id"]: row["artifact_sha256"] for row in manifest["artifacts"]
        },
        "observed_pins": pins,
    }


def _invalidate(subject, manifest: dict, current: dict) -> dict:
    before = json.dumps(manifest, sort_keys=True)
    report = subject.invalidate_dependencies(manifest, **current)
    if json.dumps(manifest, sort_keys=True) != before:
        raise RuntimeError("invalidation rewrote the manifest")
    if report["legal_dependency_inventory_exhaustive"] is not False:
        raise RuntimeError("invalidation claimed an exhaustive legal inventory")
    if report["source_semantics_verified"] is not False or report["proof_authority"] is not False:
        raise RuntimeError("invalidation claimed legal meaning")
    if report["admitted"] is not False:
        raise RuntimeError("invalidation admitted an artifact")
    if report["population_complete"] is not True:
        raise RuntimeError("declared population was dropped")
    if report["coverage_scope"] != "exact_declared_artifacts_pins_and_edges_only":
        raise RuntimeError("invalidation widened its coverage scope")
    return report


def _measure() -> dict[str, object]:
    subject = _dependencies()
    incremental = _incremental()
    compiled_sha = _sha(_COMPILED)
    if compiled_sha != _sha(bytes(_COMPILED)):
        raise RuntimeError("compiled artifact bytes changed before invalidation")
    review = _manifest(
        subject,
        [
            _artifact(subject, "legal-eligibility", compiled_sha,
                      policy=_REVIEW_A, context=_APPLICABILITY_A),
            _artifact(subject, "eligibility-user", compiled_sha),
            _artifact(subject, "conditional-compiled", compiled_sha),
        ],
        [
            ("eligibility-user", "legal-eligibility"),
            ("legal-eligibility", "conditional-compiled"),
        ],
    )
    review_current = _observations(subject, review)
    review_current["observed_pins"]["policy"]["policy:legal-eligibility"] = _REVIEW_B
    changed_review = _invalidate(subject, review, review_current)
    applicability_current = _observations(subject, review)
    applicability_current["observed_pins"]["context"][
        "context:legal-eligibility"
    ] = _APPLICABILITY_B
    changed_applicability = _invalidate(subject, review, applicability_current)
    if _COMPILED != b"conditional-compiled-artifact-v1":
        raise RuntimeError("compiled artifact bytes were destroyed")
    for report, kind in (
        (changed_review, "policy"),
        (changed_applicability, "context"),
    ):
        if report["stale_artifact_ids"] != ["legal-eligibility", "eligibility-user"]:
            raise RuntimeError("review or applicability change did not stale eligibility")
        if report["reusable_under_declared_dependencies"] != ["conditional-compiled"]:
            raise RuntimeError("conditional compiled artifact was not retained")
        if report["seed_artifact_ids"] != ["legal-eligibility"]:
            raise RuntimeError("eligibility was not the invalidation seed")
        reason = report["seed_reasons"]["legal-eligibility"]
        if reason != [
            {
                "code": "dependency_pin_changed",
                "kind": kind,
                "key": f"{kind}:legal-eligibility",
            }
        ]:
            raise RuntimeError("pin change was not the eligibility reason")
        if report["observed_artifact_sha256s"]["conditional-compiled"] != compiled_sha:
            raise RuntimeError("compiled artifact hash changed")
    original = [{"id": "term:first", "body": "covered filer"}]
    competing = [
        {"id": "term:first", "body": "covered filer"},
        {"id": "term:competing", "body": "covered filer"},
    ]
    original_raw = _table(original)
    competing_raw = _table(competing)
    if original[0] not in competing or len(competing) != 2:
        raise RuntimeError("competing table did not keep the prior definition")
    original_sha = _sha(original_raw)
    competing_sha = _sha(competing_raw)
    if original_sha == competing_sha:
        raise RuntimeError("competing definition did not change the table")
    competition = _manifest(
        subject,
        [
            _artifact(subject, "definition-partition", original_sha, definition=original_sha),
            _artifact(subject, "prior-user", compiled_sha),
            _artifact(subject, "reverse-dependent", compiled_sha),
            _artifact(subject, "unrelated-compiled", compiled_sha),
        ],
        [
            ("prior-user", "definition-partition"),
            ("reverse-dependent", "prior-user"),
        ],
    )
    competing_current = _observations(subject, competition)
    competing_current["observed_artifact_sha256s"]["definition-partition"] = competing_sha
    competing_current["observed_pins"]["definition"][
        "definition:definition-partition"
    ] = competing_sha
    added = _invalidate(subject, competition, competing_current)
    if added["stale_artifact_ids"] != [
        "definition-partition",
        "prior-user",
        "reverse-dependent",
    ]:
        raise RuntimeError("competing definition did not stale prior users")
    if added["reusable_under_declared_dependencies"] != ["unrelated-compiled"]:
        raise RuntimeError("unrelated compiled artifact was not reusable")
    if added["propagation_witnesses"] != [
        {"dependency_id": "definition-partition", "dependent_id": "prior-user"},
        {"dependency_id": "prior-user", "dependent_id": "reverse-dependent"},
    ]:
        raise RuntimeError("reverse dependents were not walked")
    if {row["code"] for row in added["seed_reasons"]["definition-partition"]} != {
        "artifact_changed",
        "dependency_pin_changed",
    }:
        raise RuntimeError("competing definition was not an artifact and pin change")
    if original_raw != _table([{"id": "term:first", "body": "covered filer"}]):
        raise RuntimeError("original definition bytes were destroyed")
    if _sha(_COMPILED) != compiled_sha:
        raise RuntimeError("compiled artifact bytes changed after invalidation")
    omitted = _observations(subject, competition)
    omitted["observed_artifact_sha256s"].pop("unrelated-compiled")
    try:
        subject.invalidate_dependencies(competition, **omitted)
    except subject.GraphDependencyError as exc:
        if "every artifact" not in str(exc):
            raise
    else:
        raise RuntimeError("an omitted artifact observation was accepted")
    try:
        incremental._previous(None, "ab" * 32, {"LegalCorpus.Conditional": _COMPILED})
    except Exception as exc:
        if type(exc).__name__ != "CorpusContractError" or _CACHE_REFUSAL not in str(exc):
            raise
    else:
        raise RuntimeError("a cache pin without artifact bytes was accepted")
    for key in ("admitted", "formalized", "proof_authority", "source_semantics_verified"):
        if incremental.FALSE_FLAGS.get(key) is not False:
            raise RuntimeError("incremental cache flags grant authority")
    text = Path(incremental.__file__).read_text(encoding="utf-8")
    if "Only compiler artifacts are reused;" not in text:
        raise RuntimeError("incremental owner no longer retains compiler artifacts")
    if '"fresh_proof_for_reused_modules": False' not in text:
        raise RuntimeError("reused modules claim a fresh proof")
    return {
        "admitted": False,
        "applicability_pin_invalidates_eligibility": True,
        "cache_pin_without_artifact_bytes_refused": True,
        "compiled_artifact_bytes_destroyed": False,
        "compiled_artifact_sha256": compiled_sha,
        "competing_definition_count": 2,
        "competing_definition_invalidates_prior_user": True,
        "competing_definition_invalidates_reverse_dependent": True,
        "competing_definition_sha256": competing_sha,
        "conditional_compiled_artifact_retained": True,
        "coverage_scope": "exact_declared_artifacts_pins_and_edges_only",
        "eligibility_seed": "legal-eligibility",
        "formalized": False,
        "fresh_proof_for_reused_modules": False,
        "incremental_compiler_invoked": False,
        "lake_build_executed": False,
        "legal_dependency_inventory_exhaustive": False,
        "legal_meaning_established": False,
        "omitted_artifact_observation_refused": True,
        "original_definition_count": 1,
        "original_definition_retained_in_competing_table": True,
        "original_definition_sha256": original_sha,
        "population_complete": True,
        "proof_authority": False,
        "review_pin_invalidates_eligibility": True,
        "review_reusable_artifact_ids": ["conditional-compiled"],
        "review_stale_artifact_ids": ["legal-eligibility", "eligibility-user"],
        "reverse_stale_artifact_ids": [
            "definition-partition",
            "prior-user",
            "reverse-dependent",
        ],
        "source_semantics_verified": False,
        "unrelated_compiled_artifact_reused": True,
    }
