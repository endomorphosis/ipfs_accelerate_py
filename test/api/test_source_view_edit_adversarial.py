"""Independent hostile edits against retained original/compact byte boundaries."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import reversible_source_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime import source_view_edits as mapper


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def edit(original, projected, start, end, replacement, **changes):
    return mapper.map_source_view_edit(**(dict(original=original, view=projected.view,
        manifest=projected.manifest, expected_manifest_sha256=projected.manifest_sha256,
        view_start=start, view_end=end, expected_view_preimage=projected.view[start:end],
        replacement=replacement) | changes))


def test_visible_newline_edit_cannot_swallow_code_in_hidden_comment():
    original = b"x = 1 # ordinary explanation\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    assert projected.view == b"x = 1\nprint(x)\n"
    start = projected.view.index(b"\n")
    # The span fits one KEEP. A coordinate-only mapper would swallow print in
    # the original trailing comment while proposing executable print in view.
    with pytest.raises(mapper.SourceViewEditError, match="codec projection"):
        edit(original, projected, start, start + 1, b"; ")


def test_edit_cannot_cross_hidden_comment_even_with_matching_visible_preimage():
    original = b"x = 1 # ordinary explanation\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index(b"1")
    end = projected.view.index(b"print")
    with pytest.raises(mapper.SourceViewEditError, match="hidden region"):
        edit(original, projected, start, end, b"2\n")


def test_insertion_at_collapsed_comment_anchor_is_ambiguous():
    original = b"x = 1 # ordinary explanation\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    gap = projected.view.index(b"\n")
    with pytest.raises(mapper.SourceViewEditError, match="ambiguous"):
        edit(original, projected, gap, gap, b" + 1")


def test_equal_lf_count_does_not_authorize_moving_caller_protected_lines():
    original = b"first = 1\n# preserved contract\nsecond = 2\n"
    projected = codec.build_source_view(original, profile="python-whitespace@1", protected_lines=(2,))
    end = original.index(b"second")
    replacement = b"first = 1 # preserved contract\n\n"
    assert replacement.count(b"\n") == projected.view[:end].count(b"\n") == 2
    with pytest.raises(mapper.SourceViewEditError, match="protected physical line"):
        edit(original, projected, 0, end, replacement)


@pytest.mark.parametrize("offset_delta", [1, 2])
def test_replacement_cannot_start_inside_utf8_code_point(offset_delta):
    original = 'label = "雪"\n'.encode()
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index("雪".encode()) + offset_delta
    with pytest.raises(mapper.SourceViewEditError, match="UTF-8 character boundaries"):
        edit(original, projected, start, start + 1, b"x")


def test_insertion_cannot_split_crlf_into_two_newline_events():
    original = b"x = 1\r\nprint(x)\r\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    split = projected.view.index(b"\r\n") + 1
    with pytest.raises(mapper.SourceViewEditError, match="CRLF boundary"):
        edit(original, projected, split, split, b" ")


def test_unicode_edit_retains_crlf_hidden_trivia_and_absent_final_newline():
    original = 'label = "雪"\r\ncount = 1 # ordinary explanation\r\nprint(label, count)'.encode()
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index("雪".encode())
    result = edit(original, projected, start, start + len("雪".encode()), "山川".encode())
    assert result.candidate == original.replace("雪".encode(), "山川".encode())
    assert b"# ordinary explanation\r\n" in result.candidate
    assert result.candidate.count(b"\r\n") == 2 and not result.candidate.endswith(b"\n")
    assert result.view == projected.view.replace("雪".encode(), "山川".encode())
    assert codec.restore_source_view(original=result.candidate, view=result.view,
        manifest=result.manifest, expected_manifest_sha256=result.manifest_sha256) == result.candidate


def test_injected_comment_cannot_hide_part_of_the_explicit_requested_edit():
    original = b"x = 1\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index(b"1")
    with pytest.raises(mapper.SourceViewEditError, match="codec projection"):
        edit(original, projected, start, start + 1, b"2 # ordinary explanation")


def test_syntax_change_cannot_turn_a_transformed_view_into_identity_fallback():
    original = b"x = 1\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index(b"1")
    with pytest.raises(mapper.SourceViewEditError, match="profile, fallback or codec"):
        edit(original, projected, start, start + 1, b"(")


@pytest.mark.parametrize("tamper", ["erase_delete", "alter_offset", "codec", "protected", "profile"])
def test_forged_mapping_is_rejected_even_when_its_manifest_digest_is_recomputed(tamper):
    original = b"x = 1 # ordinary explanation\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    manifest = deepcopy(projected.manifest)
    if tamper == "erase_delete":
        manifest["pieces"] = [dict(kind="keep", start=0, end=len(original))]
    elif tamper == "alter_offset":
        manifest["pieces"][0]["end"] += 1
    elif tamper == "codec":
        manifest["codec_binding"]["codec_sha256"] = "0" * 64
    elif tamper == "protected":
        manifest["protected_lines"] = [1]
    else:
        manifest["requested_profile"] = "identity@1"
    start = projected.view.index(b"1")
    with pytest.raises(mapper.SourceViewEditError, match="authentication failed"):
        edit(original, projected, start, start + 1, b"2", manifest=manifest,
            expected_manifest_sha256=digest(manifest))


def test_stale_original_cannot_be_repaired_by_supplied_matching_edit_preimage():
    original = b"x = 1 # ordinary explanation\nprint(x)\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    start = projected.view.index(b"1")
    stale = original.replace(b"ordinary", b"modified")
    with pytest.raises(mapper.SourceViewEditError, match="authentication failed"):
        edit(stale, projected, start, start + 1, b"2")


def test_current_codec_drift_is_rejected_even_with_unchanged_original_and_view(monkeypatch):
    original = b"x = 1\n"
    projected = codec.build_source_view(original, profile="python-comments@1")
    changed = dict(codec.current_codec_binding(), codec_sha256="0" * 64)
    monkeypatch.setattr(codec, "current_codec_binding", lambda: changed)
    start = projected.view.index(b"1")
    with pytest.raises(mapper.SourceViewEditError, match="authentication failed"):
        edit(original, projected, start, start + 1, b"2")


def test_binary_identity_edit_is_exact_and_claims_no_python_syntax_or_proof():
    original = b"\xff\x00keep\r\n\x80"
    projected = codec.build_source_view(original, profile="identity@1")
    result = edit(original, projected, 2, 6, b"replacement")
    assert result.candidate == result.view == b"\xff\x00replacement\r\n\x80"
    assert result.receipt["python_syntax_checked"] is False
    assert all(result.receipt[key] is False for key in (
        "source_freshness_verified", "source_semantics_verified", "semantic_equivalence_claimed",
        "proof_authority", "completion_authority", "execution_authority", "mutation_authority",
        "filesystem_written", "source_executed"))


def test_fallback_identity_cannot_silently_switch_to_a_new_compacting_profile():
    original = b'value = f"literal"\n'
    projected = codec.build_source_view(original, profile="python-comments@1")
    assert projected.manifest["fallback_reason"] == "unsupported_fstrings"
    start = projected.view.index(b'f"')
    with pytest.raises(mapper.SourceViewEditError, match="profile, fallback or codec"):
        edit(original, projected, start, start + 1, b"")
