"""The final audit reconstructs actual translated bytes after source changes."""
import hashlib

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import encode_semantic_router_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt


@pytest.mark.parametrize("transport_schema", ["supervisor-semantic-router-input@1",
                                               "supervisor-semantic-router-input@2"])
def test_historical_translated_projection_remains_verifiable(native, tmp_path, transport_schema):
    repository, _, prompt, source = native
    encoded = encode_semantic_router_prompt(prompt=prompt, repository=repository,
                                            transport_schema=transport_schema)
    workspace = tmp_path / "allocated/removed-worktree"
    model, header = render_model_prompt(prompt=encoded.provider_prompt, purpose="coding",
        workspace=workspace, semantic_transport=True)
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    receipt = {"schema": "router-implementation-invocation@1", "phase": "coding",
        "invocation_id": "fixture", "purpose": "coding", "workspace": str(workspace),
        "prompt_sha256": sha(prompt), "prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": sha(prompt), "native_prompt_bytes": len(prompt.encode()),
        "router_prompt_sha256": sha(encoded.provider_prompt), "router_prompt_bytes": len(encoded.provider_prompt.encode()),
        "model_prompt_sha256": sha(model), "model_prompt_bytes": len(model.encode()),
        "workspace_advisory_sha256": sha(header), "workspace_advisory_bytes": len(header.encode()),
        "semantic_translation": dict(encoded.receipt)}
    parsed = audit._receipt(receipt, workspace.parent)
    changed = {**receipt, "semantic_translation": {**receipt["semantic_translation"],
        "transport_schema": "supervisor-semantic-router-input@999"}}
    with pytest.raises(ValueError, match="transport version"):
        audit._receipt(changed, workspace.parent)
    # A completed publication can change the canonical source. Audit only
    # replays immutable historical artifacts, never claims operational freshness.
    (repository / "mod.py").write_text(source + "# accepted subsequent revision\n")
    actual, advisory, checks = audit._model_projection(rendered=prompt, receipt=parsed, repository=repository)
    assert actual == model and advisory == header and all(checks.values())
    parsed["semantic_translation"] = {**parsed["semantic_translation"], "translation_cid": "cid:foreign"}
    assert not all(audit._model_projection(rendered=prompt, receipt=parsed, repository=repository)[2].values())
    with pytest.raises(ValueError, match="canonical artifact"):
        audit._model_projection(rendered=prompt, receipt=parsed, repository=None)
    receipt["semantic_translation"] = None
    with pytest.raises(ValueError, match="unexplained"):
        audit._receipt(receipt, workspace.parent)
