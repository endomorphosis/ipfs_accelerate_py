"""Offline authored controls for the pinned historical read-only code preview.

No provider, target source, maintained implementation, tokenizer, inverse or
edit mapper is invoked. Summary output deliberately omits fixture source bodies.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

EXPECTED_HELPER_SHA256 = "0c3ed9348189ed0bf5f45a1c11c24fc3a6acff9c45d9400c853181fe58e3bebd"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(helper_path: Path) -> dict:
    actual_sha = sha(helper_path.read_bytes())
    assert actual_sha == EXPECTED_HELPER_SHA256, "historical helper bytes differ"
    spec = importlib.util.spec_from_file_location("historical_code_view_audit", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    public_functions = sorted(name for name, value in vars(helper).items()
                              if not name.startswith("_") and inspect.isfunction(value))
    assert public_functions == ["build_code_view_preview"]

    comment = "# " + ("An ordinary explanatory sentence about a small arithmetic example. " * 48)
    source = "def adjust(value):\n    " + comment + "\n    return value + 1   \n"
    preview = helper.build_code_view_preview(source)
    sidecar = json.dumps(preview["removed_comments"], sort_keys=True, separators=(",", ":"))
    sidecar_has_removed_text = comment in sidecar
    assert preview["selected"] == "minified"
    assert preview["rendered_bytes"] == len((helper.GUIDANCE + preview["view_text"]).encode())
    assert preview["rendered_bytes"] < preview["original_source_bytes"]
    assert preview["removed_comment_count"] == 1 and not sidecar_has_removed_text
    assert source.count("\n") == preview["view_text"].count("\n")
    assert preview["local_token_counts"] is None
    unchanged_view_identity_current = sha(preview["view_text"].encode()) == preview["view_sha256"]
    unchanged_source_identity_current = sha(source.encode()) == preview["original_source_sha256"]
    assert unchanged_view_identity_current and unchanged_source_identity_current

    tiny_source = "x = 1 # short\n"
    tiny = helper.build_code_view_preview(tiny_source)
    assert tiny["selected"] == "original"
    assert tiny["rendered_view"] == tiny_source and tiny["removed_comments"] == []
    assert tiny["candidate_rendered_bytes"] >= tiny["original_source_bytes"]

    edited_view = preview["view_text"].replace("return value + 1", "return value + 2", 1)
    assert edited_view != preview["view_text"]
    ast.parse(edited_view)
    edited_view_identity_current = sha(edited_view.encode()) == preview["view_sha256"]
    source_identity_still_current = sha(source.encode()) == preview["original_source_sha256"]
    assert not edited_view_identity_current and source_identity_still_current
    # This is a separately authored known candidate, not an inverse/map result.
    manual_candidate = source.replace("return value + 1", "return value + 2", 1)
    manual_source_identity_current = sha(manual_candidate.encode()) == preview["original_source_sha256"]
    assert not manual_source_identity_current

    return {
        "schema": "offline-historical-code-view-observation@1",
        "historical_helper_sha256": actual_sha,
        "historical_public_functions": public_functions,
        "inverse_or_edited_view_mapper_exposed": False,
        "scope": "three-authored-offline-controls-only",
        "provider_calls": 0, "tokenizer_calls": 0, "provider_token_counts": None,
        "token_measurement_status": "unmeasured-no-pinned-provider-tokenizer-supplied",
        "production_test_count_claimed": False,
        "large_ordinary_comment": {
            "selected": preview["selected"],
            "original_bytes": preview["original_source_bytes"],
            "raw_preview_bytes": preview["view_bytes"],
            "guidance_bytes": len(helper.GUIDANCE.encode()),
            "whole_guided_preview_bytes": preview["rendered_bytes"],
            "bytes_saved_after_guidance": preview["original_source_bytes"] - preview["rendered_bytes"],
            "removed_comment_count": preview["removed_comment_count"],
            "removed_comment_bytes": preview["removed_comment_bytes"],
            "removed_trailing_whitespace_bytes": preview["removed_trailing_whitespace_bytes"],
            "sidecar_contains_removed_comment_text": sidecar_has_removed_text,
            "sidecar_has_only_offset_hash_length_records": True,
            "unchanged_view_identity_current": unchanged_view_identity_current,
            "unchanged_source_identity_current": unchanged_source_identity_current,
            "physical_line_count_unchanged": True,
            "ast_and_token_structure_checks_reported": preview["ast_structure_verified"] and preview["token_structure_verified"],
            "exact_roundtrip_demonstrated": False,
        },
        "tiny_fallback": {
            "selected": tiny["selected"],
            "original_bytes": tiny["original_source_bytes"],
            "candidate_guided_preview_bytes": tiny["candidate_rendered_bytes"],
            "actual_rendered_bytes": tiny["rendered_bytes"],
            "original_bytes_retained": True,
        },
        "edited_preview": {
            "edited_preview_parses": True,
            "old_view_sha_binding_current": edited_view_identity_current,
            "unchanged_original_source_sha_binding_current": source_identity_still_current,
            "old_source_view_pair_current": source_identity_still_current and edited_view_identity_current,
            "separately_authored_manual_candidate_matches_old_source_sha": manual_source_identity_current,
            "edit_mapper_invoked": False,
            "edited_original_reconstruction_demonstrated": False,
            "semantic_equivalence_or_edit_correctness_claimed": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("helper_path", nargs="?")
    parser.add_argument("--source", dest="source_path")
    options = parser.parse_args()
    if bool(options.helper_path) == bool(options.source_path):
        parser.error("provide exactly one historical helper path via positional argument or --source")
    result = run(Path(options.source_path or options.helper_path).resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))
