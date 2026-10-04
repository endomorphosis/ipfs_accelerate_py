"""Chunked evidence must retain exact bytes through native capsule rendering."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompilationError, ContextCompiler, build_text_context_references,
    render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ABSOLUTE_MAX_TEXT_BYTES, ContextBoundsError, ContextBudget, ContextCapsule,
    ContextContractError, ContextIdentityError, ContextReference,
)


@pytest.mark.parametrize("text,limit", [
    ("  leading and trailing  \n", 8),
    ("\t\n \r\n\t ", 2),
    ("def f():\n    return 'two words'\n\n", 11),
    ("αβ γδ\t雪 😀\n\n end ", 7),
    ("😀😀😀", 4),
    ('{"query_text":"word ' + "two words " * 30 + 'last"}', 43),
    ("", 1),
])
def test_exact_chunks_and_roundtrip_reference_identity(text, limit):
    refs = build_text_context_references(text, reference_prefix="exact",
        kind="source", chunk_bytes=limit, required=True)
    assert "".join(ref.summary for ref in refs) == text
    for ref in refs:
        assert ref.byte_count == len(ref.summary.encode()) <= limit
        assert ref.referenced_content_id == "sha256:" + hashlib.sha256(ref.summary.encode()).hexdigest()
        restored = ContextReference.from_dict(json.loads(ref.to_json()))
        assert restored.summary == ref.summary
        assert restored.reference_content_id == ref.reference_content_id


def test_public_query_and_source_remain_exact_through_compile_persist_render():
    # A long JSON string crosses the same 6144-byte chunk boundary that lost
    # U+0020 in the actual Terminal-Bench public instruction.
    query = "Inspect /app/bottle.py. " + "Keep both words and indentation. " * 270
    payload = {"query_text": query, "source": "def f():\n    return 'a b'\n",
        "execution_authority": False, "completion_authority": False}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    refs = build_text_context_references(text, reference_prefix="retrieval",
        kind="code-retrieval-context", chunk_bytes=6144, required=True)
    assert len(refs) > 1
    result = ContextCompiler(ContextBudget(max_input_tokens=20000,
        max_items=64, max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="task:test",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "task:test"},
        authority={"mode": "proposal"}, scope={"paths": ["bottle.py"]},
        acceptance={"criteria": ["pending public check"]}, evidence=refs)
    persisted = json.loads(result.capsule.to_json())
    restored = ContextCapsule.from_dict(persisted)
    wire = json.loads(render_context_capsule(restored))
    rows = sorted(wire["evidence"], key=lambda row: row["reference_id"])
    reconstructed = "".join(row["summary"] for row in rows)
    assert reconstructed == text
    assert json.loads(reconstructed) == payload
    assert hashlib.sha256(json.loads(reconstructed)["query_text"].encode()).hexdigest() == hashlib.sha256(query.encode()).hexdigest()
    assert all(row["metadata"]["required"] is True for row in rows)
    # A change to the now-preserved boundary bytes is not accepted under the
    # original persisted capsule identity.
    persisted["evidence"][0]["summary"] = persisted["evidence"][0]["summary"].rstrip()
    with pytest.raises(ContextIdentityError):
        ContextCapsule.from_dict(persisted)


def test_summary_preservation_keeps_type_nul_and_actual_byte_bounds():
    with pytest.raises(ContextContractError, match="string"):
        ContextReference("ref", "source", summary=7)
    with pytest.raises(ContextContractError, match="NUL"):
        ContextReference("ref", "source", summary=" \x00 ")
    with pytest.raises(ContextBoundsError):
        ContextReference("ref", "source", summary=" " * (ABSOLUTE_MAX_TEXT_BYTES + 1))
    with pytest.raises(ContextCompilationError, match="UTF-8 code point"):
        build_text_context_references("😀", reference_prefix="ref", kind="source", chunk_bytes=3)
    ref = ContextReference(" ref ", " source ", summary=" \nbody\t ")
    assert ref.reference_id == "ref" and ref.kind == "source"
    assert ref.summary == " \nbody\t "
