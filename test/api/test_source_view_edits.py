"""Exact compact edits preserve hidden original bytes and refuse ambiguity."""
from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import reversible_source_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime import source_view_edits as module


SOURCE = (b"# SPDX-License-Identifier: MIT\nimport numpy as np  \n"
    b"def select(values):  \n    return np.argmax(values)  # ordinary explanation\n")


def edit(source=SOURCE, *, preimage=b"argmax", replacement=b"argmin", start=None,
        profile="python-comments@1", protected_lines=(), **overrides):
    projection = codec.build_source_view(source, profile=profile, protected_lines=protected_lines)
    start = projection.view.index(preimage) if start is None else start
    arguments = dict(original=source, view=projection.view, manifest=projection.manifest,
        expected_manifest_sha256=projection.manifest_sha256, view_start=start,
        view_end=start + len(preimage), expected_view_preimage=preimage, replacement=replacement)
    arguments.update(overrides)
    return module.map_source_view_edit(**arguments)


def test_changed_compact_call_maps_to_exact_original_and_preserves_hidden_comments():
    result = edit()
    assert result.candidate == SOURCE.replace(b"argmax", b"argmin")
    assert b"# ordinary explanation" in result.candidate and b"# ordinary explanation" not in result.view
    assert result.original_start == SOURCE.index(b"argmax")
    assert result.original_end == result.original_start + len(b"argmax")
    assert result.candidate[:result.original_start] == SOURCE[:result.original_start]
    assert result.candidate[result.original_start + len(b"argmin"):] == SOURCE[result.original_end:]
    assert codec.restore_source_view(original=result.candidate, view=result.view, manifest=result.manifest,
        expected_manifest_sha256=result.manifest_sha256) == result.candidate
    old = codec.build_source_view(SOURCE, profile="python-comments@1")
    assert result.view == old.view.replace(b"argmax", b"argmin")


@pytest.mark.parametrize("source,preimage,replacement", [
    (b"def f(value):  \n    return value + 10  \n", b"10", b"2000"),
    (b"def f(value):\n    return value + 10\n", b" + 10", b""),
    (b"x = 1   \ny = 2\n", b"1", b"3"),
    (b"x = 1   \ny = 2\n", b"\n", b"\n\n"),
])
def test_replacement_deletion_and_nonempty_gap_endpoint_have_exact_round_trip(source, preimage, replacement):
    result = edit(source, preimage=preimage, replacement=replacement)
    assert result.candidate == source.replace(preimage, replacement, 1)
    assert result.receipt["exact_compact_edit_round_trip"] is True


@pytest.mark.parametrize("source,start,replacement,expected", [
    (b"x = 1\n", 0, b"# header\n", b"# header\nx = 1\n"),
    (b"x = 1\n", 6, b"y = 2\n", b"x = 1\ny = 2\n"),
    (b"x = 1\n", 5, b"0", b"x = 10\n"),
    (b"", 0, b"x = 1\n", b"x = 1\n"),
])
def test_unambiguous_insertion_at_exposed_endpoints_or_inside_keep(source, start, replacement, expected):
    result = edit(source, preimage=b"", start=start, replacement=replacement)
    assert result.candidate == expected and result.original_start == result.original_end


@pytest.mark.parametrize("source,start", [
    (b"   \nx = 1\n", 0),
    (b"x = 1   ", 5),
    (b"x = 1   \ny = 2\n", 5),
    (b"x = 1 # ordinary explanation\ny = 2\n", 5),
])
def test_insertion_at_collapsed_hidden_regions_refuses(source, start):
    with pytest.raises(module.SourceViewEditError, match="ambiguous insertion"):
        edit(source, preimage=b"", start=start, replacement=b"0")


def test_nonempty_edit_cannot_cross_a_hidden_region():
    source = b"x = 1   \ny = 2\n"
    with pytest.raises(module.SourceViewEditError, match="crosses a hidden region"):
        edit(source, preimage=b"1\ny", replacement=b"3\nz")


def test_hidden_comment_cannot_swallow_visible_code_after_newline_edit():
    source = b"x = 1 # ordinary explanation\nprint(x)\n"
    with pytest.raises(module.SourceViewEditError, match="exactly match candidate"):
        edit(source, preimage=b"\n", replacement=b"; ")


def test_replacement_removed_by_codec_is_not_misreported_as_applied():
    with pytest.raises(module.SourceViewEditError, match="exactly match candidate"):
        edit(b"x = 1\n", preimage=b"1", replacement=b"2  ")


@pytest.mark.parametrize("replacement", [b"'", b"f'{1}'"])
def test_transformed_python_cannot_silently_fall_back_to_identity(replacement):
    with pytest.raises(module.SourceViewEditError, match="profile, fallback or codec"):
        edit(b"x = 1\n", preimage=b"1", replacement=replacement)


def test_identity_fallback_cannot_silently_become_a_python_transform():
    with pytest.raises(module.SourceViewEditError, match="profile, fallback or codec"):
        edit(b"x = \n", preimage=b"= ", replacement=b"= 1")


def test_stable_identity_fallback_retains_its_explicit_unchecked_syntax_status():
    result = edit(b"x = f'{1}'\n", preimage=b"1", replacement=b"2")
    assert result.candidate == b"x = f'{2}'\n"
    assert result.manifest["effective_profile"] == "identity@1"
    assert result.receipt["fallback_reason"] == "unsupported_fstrings"
    assert result.receipt["python_syntax_checked"] is False


def test_identity_edits_arbitrary_bytes_without_parser_calls(monkeypatch):
    monkeypatch.setattr(codec, "_python_view", lambda *_: pytest.fail("identity parsed bytes"))
    source = b"\xff\x00\r\ninvalid syntax ("
    result = edit(source, preimage=b"\x00\r", replacement=b"\x80", profile="identity@1")
    assert result.candidate == result.view == b"\xff\x80\ninvalid syntax ("
    assert result.receipt["python_syntax_checked"] is False


def test_utf8_character_replacement_preserves_crlf_and_exact_hidden_bytes():
    source = 'message = "caf\u00e9"  # ordinary words\r\n'.encode()
    result = edit(source, preimage="\u00e9".encode(), replacement="\u03bb".encode())
    assert result.candidate == source.replace("\u00e9".encode(), "\u03bb".encode())
    assert result.candidate.endswith(b"# ordinary words\r\n") and result.view.endswith(b"\r\n")
    assert result.receipt["python_syntax_checked"] is True


@pytest.mark.parametrize("preimage,replacement", [(b"\xc3", b"\xc4"), (b"\xa9", b"\xb1")])
def test_valid_utf8_result_still_cannot_edit_half_a_character(preimage, replacement):
    # Both substitutions could produce another valid Unicode character, but
    # compact edit coordinates must name complete UTF-8 character boundaries.
    source = 'message = "caf\u00e9"\n'.encode()
    with pytest.raises(module.SourceViewEditError, match="UTF-8 character boundaries"):
        edit(source, preimage=preimage, replacement=replacement)


def test_invalid_utf8_replacement_refuses_before_profile_fallback():
    with pytest.raises(module.SourceViewEditError, match="UTF-8 character boundaries"):
        edit(b"x = 1\n", preimage=b"1", replacement=b"\xff")


@pytest.mark.parametrize("preimage,replacement", [(b"\r", b""), (b"\n", b"\n\n")])
def test_transformed_python_cannot_split_crlf(preimage, replacement):
    with pytest.raises(module.SourceViewEditError, match="CRLF boundary"):
        edit(b"x = 1\r\n", preimage=preimage, replacement=replacement)


@pytest.mark.parametrize("preimage,replacement", [
    (b"1\n", b"2\n"), (b"1", b"2\n"), (b"1\n", b"2"),
])
def test_protected_line_policy_conservatively_forbids_newline_edits(preimage, replacement):
    source = b"x = 1\n# retained explanation\ny = 2\n"
    with pytest.raises(module.SourceViewEditError, match="caller-protected physical line"):
        edit(source, preimage=preimage, replacement=replacement, protected_lines=(2,))


def test_same_line_edit_preserves_caller_protected_policy():
    source = b"x = 1\n# retained explanation\ny = 2\n"
    result = edit(source, preimage=b"1", replacement=b"3", protected_lines=(2,))
    assert result.manifest["protected_lines"] == [2]
    assert b"# retained explanation" in result.view


@pytest.mark.parametrize("overrides", [
    {"view_start": True}, {"view_end": False}, {"view_start": -1},
    {"view_end": 100000}, {"view_start": 50, "view_end": 49},
    {"original": bytearray(SOURCE)}, {"view": "text"},
    {"expected_view_preimage": "argmax"}, {"replacement": bytearray(b"argmin")},
    {"replacement": b"x" * (module.MAX_EDIT_BYTES + 1)},
    {"expected_view_preimage": b"x" * (module.MAX_EDIT_BYTES + 1)},
    {"original": b"x" * (module.MAX_SOURCE_BYTES + 1)},
    {"view": b"x" * (module.MAX_SOURCE_BYTES + 1)},
    {"manifest": []},
])
def test_invalid_types_offsets_and_bounds_refuse(overrides):
    with pytest.raises(module.SourceViewEditError):
        edit(**overrides)


@pytest.mark.parametrize("overrides", [
    {"original": SOURCE.replace(b"argmax", b"argmin")},
    {"view": b"different view"}, {"expected_manifest_sha256": "0" * 64},
    {"manifest": {}}, {"expected_manifest_sha256": True},
])
def test_stale_source_view_or_manifest_pins_refuse(overrides):
    with pytest.raises(module.SourceViewEditError):
        edit(**overrides)


def test_wrong_preimage_and_no_op_refuse():
    with pytest.raises(module.SourceViewEditError, match="preimage differs"):
        edit(expected_view_preimage=b"argmin")
    with pytest.raises(module.SourceViewEditError, match="no-op"):
        edit(replacement=b"argmax")
    with pytest.raises(module.SourceViewEditError, match="no-op"):
        edit(preimage=b"", start=0, replacement=b"")


def test_candidate_size_bound_is_checked_before_projection(monkeypatch):
    source = b"x" * module.MAX_SOURCE_BYTES
    projected = codec.build_source_view(source, profile="identity@1")
    monkeypatch.setattr(module, "build_source_view", lambda *_args, **_kwargs: pytest.fail("oversized candidate projected"))
    with pytest.raises(module.SourceViewEditError, match="candidate source byte bound"):
        module.map_source_view_edit(original=source, view=projected.view, manifest=projected.manifest,
            expected_manifest_sha256=projected.manifest_sha256, view_start=0, view_end=0,
            expected_view_preimage=b"", replacement=b"x")


def test_rehashed_forged_map_and_authority_still_refuse():
    original = b"x = 1   \n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    for mutation in ("piece", "authority"):
        manifest = json.loads(module._wire(projected.manifest))
        if mutation == "piece":
            manifest["pieces"][0]["start"] = 1
        else:
            manifest["mutation_authority"] = True
        with pytest.raises(module.SourceViewEditError, match="authentication failed"):
            module.map_source_view_edit(original=original, view=projected.view, manifest=manifest,
                expected_manifest_sha256=hashlib.sha256(module._wire(manifest)).hexdigest(),
                view_start=4, view_end=5, expected_view_preimage=b"1", replacement=b"2")


def test_frozen_result_and_exposed_dict_copies_cannot_mutate_candidate_metadata():
    first, second = edit(), edit()
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.candidate = b"forged"
    manifest, receipt = first.manifest, first.receipt
    manifest["pieces"][0]["end"] = 0
    receipt["mutation_authority"] = True
    assert first.manifest == second.manifest and first.receipt == second.receipt
    assert first.manifest_sha256 == hashlib.sha256(first.manifest_json).hexdigest()


def test_receipt_binds_all_inputs_and_outputs_without_granting_authority():
    result = edit()
    receipt = result.receipt
    assert receipt["schema"] == module.SCHEMA
    assert receipt["original_sha256"] == hashlib.sha256(SOURCE).hexdigest()
    assert receipt["candidate_sha256"] == hashlib.sha256(result.candidate).hexdigest()
    assert receipt["candidate_view_sha256"] == hashlib.sha256(result.view).hexdigest()
    assert receipt["candidate_manifest_sha256"] == result.manifest_sha256
    assert receipt["preimage_sha256"] == hashlib.sha256(b"argmax").hexdigest()
    assert receipt["replacement_sha256"] == hashlib.sha256(b"argmin").hexdigest()
    assert receipt["untouched_original_bytes_preserved"] is receipt["exact_compact_edit_round_trip"] is True
    for key in ("source_freshness_verified", "source_semantics_verified", "semantic_equivalence_claimed",
            "proof_authority", "completion_authority", "execution_authority", "mutation_authority",
            "omission_authority", "filesystem_written", "source_executed"):
        assert receipt[key] is False
    assert receipt["source_obligations_discharged"] == receipt["provider_calls"] == receipt["solver_calls"] == 0


def test_target_code_is_never_executed_or_written(tmp_path):
    marker = tmp_path / "target-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('target')\nVALUE = 1\n".encode()
    result = edit(source, preimage=b"VALUE = 1", replacement=b"VALUE = 2")
    assert b"VALUE = 2" in result.candidate and not marker.exists()
    assert list(tmp_path.iterdir()) == []
