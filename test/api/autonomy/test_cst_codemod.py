from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.cst_codemod import (
    AST_ONLY_NOT_CST,
    COMMENTS_NOT_PRESERVED,
    CST_PRESERVED,
    LIBCST_NOT_USABLE,
    SOURCE_MAP_NOT_PRESERVED,
    UNDECLARED_CST_TRANSFORM,
    cst_codemod_view,
)


def test_missing_cst_payload_does_not_block() -> None:
    view = cst_codemod_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_ast_only_is_not_cst() -> None:
    view = cst_codemod_view({"ast_only": True, "cst_preserving": True})
    assert view["blocks_completion"] is True
    assert view["reason_code"] == AST_ONLY_NOT_CST


def test_libcst_must_not_be_claimed_usable() -> None:
    view = cst_codemod_view({"libcst_usable": True})
    assert view["blocks_completion"] is True
    assert view["reason_code"] == LIBCST_NOT_USABLE
    assert view["completes_task"] is False


def test_cst_preserving_requires_comments_and_source_maps() -> None:
    comments = cst_codemod_view(
        {
            "cst_preserving": True,
            "comments_preserved": False,
            "source_map_preserved": True,
        }
    )
    assert comments["reason_code"] == COMMENTS_NOT_PRESERVED
    maps = cst_codemod_view(
        {
            "cst_preserving": True,
            "comments_preserved": True,
            "source_map_preserved": False,
        }
    )
    assert maps["reason_code"] == SOURCE_MAP_NOT_PRESERVED
    ok = cst_codemod_view(
        {
            "cst_preserving": True,
            "comments_preserved": True,
            "source_map_preserved": True,
        }
    )
    assert ok["preserved"] is True
    assert ok["blocks_completion"] is False
    assert ok["reason_code"] == CST_PRESERVED
    assert ok["completes_task"] is False


def test_undeclared_cst_cannot_complete() -> None:
    view = cst_codemod_view({"undeclared_cst": True})
    assert view["reason_code"] == UNDECLARED_CST_TRANSFORM
    assert view["blocks_completion"] is True
    assert view["accepted_as_authority"] is False
