"""Production-provider context is exact, bounded, and fail closed."""

from __future__ import annotations

import subprocess
import base64
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ImplementationProviderRouter,
    ProductionContractPacket,
    ProviderBounds,
    RouteStatus,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_context_slice import (
    ProductionContextSliceError,
    assert_proposal_covered_by_context,
    build_production_context_slice,
    build_production_evidence_authority,
    verify_production_context_slice,
    verify_production_evidence_authority,
)

TASK = {
    "task_id": "ASE-CONTEXT-001",
    "title": "Make greeting configurable",
    "acceptance": "greet uses the supplied name",
    "outputs": ["src/greeting.py"],
}


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path, source: str | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Context Slice Test")
    _git(repo, "config", "user.email", "context-slice@example.invalid")
    target = repo / "src" / "greeting.py"
    target.parent.mkdir()
    target.write_text(
        source
        if source is not None
        else "def greet(name: str) -> str:\n    return f\"hello {name}\"\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo


def _manifest(repo: Path, **overrides):
    kwargs = {
        "repo_root": repo,
        "task_id": TASK["task_id"],
        "task_payload": TASK,
        "read_paths": ["src/greeting.py"],
        "effect_paths": ["src/greeting.py"],
    }
    kwargs.update(overrides)
    return build_production_context_slice(**kwargs)


def _assert_reason(reason: str, action) -> None:
    with pytest.raises(ProductionContextSliceError) as captured:
        action()
    assert captured.value.reason_code == reason


def _recompute_root(payload: dict) -> dict:
    payload["manifest_cid"] = content_identity(
        {key: value for key, value in payload.items() if key != "manifest_cid"}
    )
    return payload


def _expected_scope(**overrides):
    scope = {
        "expected_read_paths": ["src/greeting.py"],
        "expected_effect_paths": ["src/greeting.py"],
    }
    scope.update(overrides)
    return scope


def test_existing_small_file_edit_receives_exact_context_and_applies(tmp_path) -> None:
    repo = _repo(tmp_path)
    manifest = _manifest(repo)
    source = (repo / "src" / "greeting.py").read_text(encoding="utf-8")
    snapshot = f"git-commit:{_git(repo, 'rev-parse', 'HEAD')}"
    replacement = "def greet(name: str) -> str:\n    return f\"welcome {name}\"\n"
    seen: dict[str, object] = {}

    packet = ProductionContractPacket(
        packet_id="packet:ASE-CONTEXT-001:1",
        snapshot_id=snapshot,
        task_id=TASK["task_id"],
        payload={
            "goal": {"task_id": TASK["task_id"]},
            "scope": {
                "read_paths": ["src/greeting.py"],
                "write_paths": ["src/greeting.py"],
            },
            **manifest.provider_payload(),
        },
    )

    def grok(request):
        seen["tokens"] = request.prompt_tokens
        context = request["provider_input"]["contract_packet"]["context_slice"]
        visible = context["sources"][0]["source_slices"]
        assert len(visible) == 1
        assert visible[0]["kind"] == "whole_file"
        assert visible[0]["utf8_text"] == source
        assert context["sources"][0]["full_visible_coverage"] is True
        return {
            "proposal": {
                "declared_paths": ["src/greeting.py"],
                "files": [{"path": "src/greeting.py", "content": replacement}],
            }
        }

    def codex(request):
        assert request["role"] == "codex-independent-review"
        reviewer_context = request["provider_input"]["evidence_slice"][
            "context_slice"
        ]
        assert reviewer_context == manifest.to_dict()
        assert request.prompt_tokens <= 4096
        return {"decision": "approve", "findings": []}

    def admit(_proposal):
        return {"accepted": True, "reason_code": "admitted"}

    def writer(proposal, lease_id):
        assert lease_id == "lease:context:1"
        assert_proposal_covered_by_context(
            manifest,
            proposal.payload,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        )
        body = proposal.payload.get("proposal", proposal.payload)
        item = body["files"][0]
        (repo / item["path"]).write_text(item["content"], encoding="utf-8")

    result = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=admit,
        writer=writer,
        bounds=ProviderBounds(max_prompt_tokens=4096),
    ).route(
        packet,
        current_snapshot_id=snapshot,
        apply=True,
        writer_lease_id="lease:context:1",
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert result.write_performed is True
    assert 0 < int(seen["tokens"]) <= 4096
    assert (repo / "src" / "greeting.py").read_text(encoding="utf-8") == replacement


def test_manifest_is_deterministic_cid_addressed_and_current(tmp_path) -> None:
    repo = _repo(tmp_path)
    first = _manifest(repo)
    second = _manifest(repo)

    assert first.to_dict() == second.to_dict()
    assert first.manifest_cid.startswith("b")
    record = first.to_dict()["sources"][0]
    assert record["file_cid"].startswith("bafk")
    assert record["partition_root_cid"].startswith("b")
    assert record["residuals"] == []
    verified = verify_production_context_slice(
        first,
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=TASK,
        **_expected_scope(),
    )
    assert verified.manifest_cid == first.manifest_cid


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "./src/greeting.py",
        "src/../src/greeting.py",
        "src\\greeting.py",
        "src/greeting.py\x00ignored",
    ],
)
def test_path_escapes_are_rejected(tmp_path, path: str) -> None:
    repo = _repo(tmp_path)
    _assert_reason(
        "path_escape",
        lambda: _manifest(repo, read_paths=[path], effect_paths=[path]),
    )


def test_symlink_and_nested_repository_boundaries_are_rejected(tmp_path) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    (repo / "linked.py").symlink_to(outside)
    _assert_reason(
        "symlink_escape",
        lambda: _manifest(
            repo,
            read_paths=["linked.py"],
            effect_paths=["linked.py"],
        ),
    )

    nested = repo / "vendor" / "child"
    nested.mkdir(parents=True)
    _git(nested, "init")
    (nested / "module.py").write_text("value = 1\n", encoding="utf-8")
    # The filesystem repository boundary is checked before Git object lookup,
    # so no nested bytes may be read even when the child is not staged.
    _assert_reason(
        "nested_repository_escape",
        lambda: _manifest(
            repo,
            read_paths=["vendor/child/module.py"],
            effect_paths=["vendor/child/module.py"],
        ),
    )


def test_task_blob_tree_and_manifest_staleness_are_rejected(tmp_path) -> None:
    repo = _repo(tmp_path)
    manifest = _manifest(repo)

    changed_task = {**TASK, "acceptance": "different current requirement"}
    _assert_reason(
        "task_binding_stale",
        lambda: verify_production_context_slice(
            manifest,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=changed_task,
            **_expected_scope(),
        ),
    )

    tampered = manifest.to_dict()
    tampered["sources"][0]["source_slices"][0]["utf8_text"] = "not the blob"
    _assert_reason(
        "manifest_cid_mismatch",
        lambda: verify_production_context_slice(
            tampered,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )

    target = repo / "src" / "greeting.py"
    target.write_text("def greet(name):\n    return 'dirty'\n", encoding="utf-8")
    _assert_reason(
        "blob_stale",
        lambda: verify_production_context_slice(
            manifest,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )

    target.write_text(
        "def greet(name: str) -> str:\n    return f\"hello {name}\"\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("next tree\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "new current baseline")
    _assert_reason(
        "tree_stale",
        lambda: verify_production_context_slice(
            manifest,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )


def test_recomputed_root_cannot_widen_corpus_budget_or_authority(tmp_path) -> None:
    repo = _repo(tmp_path)
    manifest = _manifest(repo)

    widened = manifest.to_dict()
    widened["repository_corpus"] = {"src/hidden.py": "hidden = True\n"}
    _recompute_root(widened)
    _assert_reason(
        "corpus_widening",
        lambda: verify_production_context_slice(
            widened,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )

    authority = manifest.to_dict()
    authority["authority"]["repository_write_allowed"] = True
    _recompute_root(authority)
    _assert_reason(
        "authority_claim",
        lambda: verify_production_context_slice(
            authority,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )

    unbounded = manifest.to_dict()
    # Exceed the production context protocol maximum (currently 32_768).
    unbounded["budget"]["max_provider_prompt_tokens"] = 65_536
    unbounded["budget"]["context_token_limit"] = (
        65_536 - unbounded["budget"]["reserved_prompt_tokens"]
    )
    _recompute_root(unbounded)
    _assert_reason(
        "budget_invalid",
        lambda: verify_production_context_slice(
            unbounded,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(),
        ),
    )


def test_new_and_mixed_effects_use_current_baseline_absence_proofs(tmp_path) -> None:
    repo = _repo(tmp_path)
    new_task = {
        **TASK,
        "outputs": ["src/greeting.py", "src/generated/module.py"],
    }
    manifest = build_production_context_slice(
        repo_root=repo,
        task_id=TASK["task_id"],
        task_payload=new_task,
        read_paths=["src/greeting.py"],
        effect_paths=["src/greeting.py", "src/generated/module.py"],
    )
    payload = manifest.to_dict()

    assert [proof["path"] for proof in payload["scope"]["absence_proofs"]] == [
        "src/generated/module.py"
    ]
    assert payload["scope"]["absence_proofs"][0]["absence_cid"].startswith("b")
    verify_production_context_slice(
        manifest,
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=new_task,
        expected_read_paths=["src/greeting.py"],
        expected_effect_paths=["src/greeting.py", "src/generated/module.py"],
    )
    assert_proposal_covered_by_context(
        manifest,
        {
            "files": [
                {
                    "path": "src/generated/module.py",
                    "content": "GENERATED = True\n",
                }
            ]
        },
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=new_task,
        expected_read_paths=["src/greeting.py"],
        expected_effect_paths=["src/greeting.py", "src/generated/module.py"],
    )

    new_patch = (
        "diff --git a/src/generated/module.py b/src/generated/module.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/generated/module.py\n"
        "@@ -0,0 +1 @@\n"
        "+GENERATED = True\n"
    )
    assert_proposal_covered_by_context(
        manifest,
        {"patch": new_patch},
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=new_task,
        expected_read_paths=["src/greeting.py"],
        expected_effect_paths=["src/greeting.py", "src/generated/module.py"],
    )

    generated = repo / "src" / "generated" / "module.py"
    generated.parent.mkdir()
    generated.write_text("occupied = True\n", encoding="utf-8")
    _assert_reason(
        "absence_proof_stale",
        lambda: verify_production_context_slice(
            manifest,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=new_task,
            expected_read_paths=["src/greeting.py"],
            expected_effect_paths=["src/greeting.py", "src/generated/module.py"],
        ),
    )


def test_new_only_effect_does_not_require_fake_source_context(tmp_path) -> None:
    repo = _repo(tmp_path)
    new_task = {**TASK, "outputs": ["generated.py"]}
    manifest = build_production_context_slice(
        repo_root=repo,
        task_id=TASK["task_id"],
        task_payload=new_task,
        read_paths=[],
        effect_paths=["generated.py"],
    )

    assert manifest.to_dict()["sources"] == []
    verify_production_context_slice(
        manifest,
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=new_task,
        expected_read_paths=[],
        expected_effect_paths=["generated.py"],
    )


def test_evidence_directory_expands_more_than_eight_paths_without_widening_write_scope(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    evidence_dir = repo / "evidence"
    evidence_dir.mkdir()
    for index in range(12):
        (evidence_dir / f"source_{index:02d}.py").write_text(
            f"import dependency_{index}\nVALUE_{index} = {index}\n"
            f"def public_{index}():\n    return VALUE_{index}\n",
            encoding="utf-8",
        )
    _git(repo, "add", "evidence")
    _git(repo, "commit", "-m", "evidence corpus")
    task = {
        "task_id": "PCCE-003-LIKE",
        "outputs": ["artifacts/inventory.json", "artifacts/receipt.json"],
    }

    authority = build_production_evidence_authority(
        repo_root=repo,
        task_id=task["task_id"],
        task_payload=task,
        evidence_inputs=["evidence"],
        max_evidence_tokens=16_384,
    )
    payload = authority.to_dict()

    assert len(payload["sources"]) == 8
    assert payload["readiness"]["provider_ready"] is True
    handle = payload["expansion_handles"][0]
    assert handle["authorized_path"] == "evidence"
    assert handle["omitted_source_count"] == 4
    assert len(handle["omitted_path_preview"]) == 4
    assert handle["expansion_cid"].startswith("b")
    assert all("git_blob_oid" in item for item in handle["omitted_path_preview"])
    inventory = payload["declarations"][0]["inventory_entries"]
    assert len(inventory) == 12
    assert {item["path"] for item in inventory} == {
        f"evidence/source_{index:02d}.py" for index in range(12)
    }
    assert all(item["git_object_oid"] for item in inventory)
    assert all(item["summary"]["imports"] for item in inventory)
    assert all(item["summary"]["public_symbols"] for item in inventory)
    assert "write_paths" not in payload
    verify_production_evidence_authority(
        authority,
        repo_root=repo,
        current_task_id=task["task_id"],
        current_task_payload=task,
        expected_evidence_inputs=["evidence"],
    )

    expanded = build_production_evidence_authority(
        repo_root=repo,
        task_id=task["task_id"],
        task_payload=task,
        evidence_inputs=["evidence"],
        max_evidence_tokens=16_384,
        context_round=1,
        parent_evidence_cid=payload["evidence_cid"],
        selected_expansion_cids=[handle["expansion_cid"]],
        expansion_selections={"input:evidence": 1},
    )
    expanded_payload = expanded.to_dict()
    first_paths = {item["path"] for item in payload["sources"]}
    expanded_paths = {item["path"] for item in expanded_payload["sources"]}
    assert len(expanded_paths) == 12
    assert first_paths < expanded_paths
    assert expanded_payload["selection"] == {
        "directory_candidate_count": 12,
        "directory_window_end": 8,
        "directory_window_start": 0,
        "context_round": 1,
        "explicit_file_anchor_count": 0,
    }
    verify_production_evidence_authority(
        expanded,
        repo_root=repo,
        current_task_id=task["task_id"],
        current_task_payload=task,
        expected_evidence_inputs=["evidence"],
        expected_context_round=1,
        expected_parent_evidence_cid=payload["evidence_cid"],
        expected_selected_expansion_cids=[handle["expansion_cid"]],
        expected_expansion_selections={"input:evidence": 1},
    )
    assert expanded_payload["expansion_chain"]["parent_evidence_cid"] == payload[
        "evidence_cid"
    ]


def test_evidence_git_calls_ignore_ambient_routing_and_literalize_paths(
    tmp_path,
    monkeypatch,
) -> None:
    repo = _repo(tmp_path)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    _git(hostile, "init")
    _git(hostile, "config", "user.name", "Hostile Git Test")
    _git(hostile, "config", "user.email", "hostile@example.invalid")
    (hostile / "README.md").write_text("wrong repository\n", encoding="utf-8")
    _git(hostile, "add", ".")
    _git(hostile, "commit", "-m", "hostile")
    task = {"task_id": "EVIDENCE-GIT-ENV", "outputs": ["new.json"]}

    monkeypatch.setenv("GIT_DIR", str(hostile / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(hostile))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(hostile / ".git" / "objects"))
    monkeypatch.setenv(
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        str(hostile / ".git" / "objects"),
    )
    authority = build_production_evidence_authority(
        repo_root=repo,
        task_id=task["task_id"],
        task_payload=task,
        evidence_inputs=["src/greeting.py"],
    )
    assert "def greet" in authority.to_dict()["sources"][0]["utf8_text"]
    _assert_reason(
        "path_escape",
        lambda: build_production_evidence_authority(
            repo_root=repo,
            task_id=task["task_id"],
            task_payload=task,
            evidence_inputs=[":(top,glob)**"],
        ),
    )


def test_evidence_authority_binds_four_governed_gitlink_roots(tmp_path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    _git(outer, "init")
    _git(outer, "config", "user.name", "Evidence Forest Test")
    _git(outer, "config", "user.email", "evidence-forest@example.invalid")
    (outer / "README.md").write_text("control\n", encoding="utf-8")
    _git(outer, "add", "README.md")
    _git(outer, "commit", "-m", "outer baseline")

    roots = (
        "external/ipfs_datasets",
        "external/ipfs_kit",
        "external/ipfs_accelerate",
        "Mcp-Plus-Plus",
    )
    children: list[Path] = []
    for index, namespace in enumerate(roots):
        child = tmp_path / f"child-{index}"
        child.mkdir()
        _git(child, "init")
        _git(child, "config", "user.name", "Evidence Child Test")
        _git(child, "config", "user.email", "evidence-child@example.invalid")
        (child / "src").mkdir()
        (child / "src" / "authority.py").write_text(
            f"ROOT_INDEX = {index}\n",
            encoding="utf-8",
        )
        _git(child, "add", ".")
        _git(child, "commit", "-m", "child baseline")
        subprocess.run(
            [
                "git",
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                str(child),
                namespace,
            ],
            cwd=outer,
            check=True,
            text=True,
            capture_output=True,
        )
        children.append(child)
    _git(outer, "commit", "-am", "bind governed roots")

    task = {"task_id": "PCCE-FOREST", "outputs": ["artifacts/result.json"]}
    declarations = [f"{namespace}/src" for namespace in roots]
    authority = build_production_evidence_authority(
        repo_root=outer,
        task_id=task["task_id"],
        task_payload=task,
        evidence_inputs=declarations,
        governed_repository_roots=roots,
        max_evidence_tokens=16_384,
    )
    payload = authority.to_dict()

    assert [item["root_path"] for item in payload["root_bindings"]] == [
        ".",
        *sorted(roots),
    ]
    assert len(payload["sources"]) == 4
    assert all(item["parent_gitlink_oid"] for item in payload["root_bindings"][1:])
    assert all(
        item["path"].startswith(tuple(f"{root}/" for root in roots))
        for item in payload["sources"]
    )
    verify_production_evidence_authority(
        authority,
        repo_root=outer,
        current_task_id=task["task_id"],
        current_task_payload=task,
        expected_evidence_inputs=declarations,
        governed_repository_roots=roots,
    )

    checkout = outer / roots[2]
    (checkout / "src" / "authority.py").write_text(
        "ROOT_INDEX = 999\n",
        encoding="utf-8",
    )
    _assert_reason(
        "evidence_blob_stale",
        lambda: verify_production_evidence_authority(
            authority,
            repo_root=outer,
            current_task_id=task["task_id"],
            current_task_payload=task,
            expected_evidence_inputs=declarations,
            governed_repository_roots=roots,
        ),
    )


def test_ref_evidence_is_exact_prioritized_and_excludes_control_artifacts(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    baseline_branch = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "-b", "wip/ref-evidence")
    (repo / "src" / "greeting.py").write_text(
        "import selected_dependency\ndef greet(name):\n    return name.upper()\n",
        encoding="utf-8",
    )
    for name in ("other.py", "later.py"):
        (repo / "src" / name).write_text(
            f"def {name[:-3]}():\n    return True\n",
            encoding="utf-8",
        )
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "control.json").write_text("{}\n", encoding="utf-8")
    board = repo / "docs" / "architecture"
    board.mkdir(parents=True)
    (board / "tasks.todo.md").write_text("# mutable board\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "candidate ref")
    candidate_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", baseline_branch)

    task = {"task_id": "REF-EVIDENCE", "outputs": ["src/greeting.py"]}
    preserved_ref = (
        "refs/pcce-candidates/proof-carrying-context-engine-v0.1/r6/"
        "source-task/outer/ref-evidence"
    )
    _git(repo, "update-ref", preserved_ref, candidate_commit)
    candidate_id = "ref-evidence"
    declaration = candidate_id
    baseline_commit = _git(repo, "rev-parse", "HEAD^{commit}")
    baseline_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    candidate_tree = _git(repo, "rev-parse", f"{candidate_commit}^{{tree}}")
    task_cid = content_identity(task)
    record_core = {
        "candidate_id": candidate_id,
        "authority_mode": "operator_signed_cross_task",
        "source_board_namespace": "source-board",
        "source_task_id": "SOURCE-TASK",
        "source_task_cid": content_identity({"task_id": "SOURCE-TASK"}),
        "target_task_id": task["task_id"],
        "target_task_cid": task_cid,
        "repository_namespace": ".",
        "preserved_ref": preserved_ref,
        "origin_base_commit": baseline_commit,
        "origin_base_tree": baseline_tree,
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "merge_base": baseline_commit,
        "ancestry_verified": True,
        "implementation_started_event_id": "event:implementation-started",
        "worktree_preserved_event_id": "event:worktree-preserved",
    }
    record = {**record_core, "record_cid": content_identity(record_core)}
    signer = Ed25519PrivateKey.generate()
    signer_did = ed25519_did_key(signer.public_key())
    appendix_unsigned = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "candidate-ref-authority-appendix@1"
        ),
        "board_namespace": "pcce-board",
        "board_projection_id": "board-projection:test",
        "records": [record],
        "signer_identity_did": signer_did,
    }
    signature = base64.b64encode(
        signer.sign(canonical_json_bytes(appendix_unsigned))
    ).decode("ascii")
    appendix = {
        **appendix_unsigned,
        "signature": signature,
        "appendix_cid": content_identity(
            {**appendix_unsigned, "signature": signature}
        ),
    }
    authority = build_production_evidence_authority(
        repo_root=repo,
        task_id=task["task_id"],
        task_payload=task,
        evidence_inputs=["src/greeting.py"],
        evidence_refs=[declaration],
        candidate_ref_authority_appendix=appendix,
        board_namespace="pcce-board",
        board_projection_id="board-projection:test",
        candidate_authority_signer_did=signer_did,
        priority_paths=["src/greeting.py"],
        max_evidence_tokens=32_768,
        max_ref_diffs=1,
    )
    payload = authority.to_dict()
    ref = payload["ref_bindings"][0]
    assert ref["declared_commit"] == candidate_commit
    assert ref["ref_tree"]
    assert ref["selection"]["selected_paths"] == ["src/greeting.py"]
    assert ref["diffs"][0]["priority_class"] == "owned_or_predicted"
    assert "selected_dependency" in next(
        item["summary"]["imports"]
        for item in ref["changed_paths"]
        if item.get("path") == "src/greeting.py"
    )
    assert ref["changed_paths"] == [
        next(item for item in ref["changed_paths"] if item["path"] == "src/greeting.py")
    ]
    assert ref["excluded_path_classes"] == {
        "generated_artifact": 1,
        "supervisor_control": 1,
        "unrelated": 2,
    }
    assert all(
        not item["path"].startswith(("artifacts/", "docs/architecture/"))
        for item in ref["diffs"]
    )
    assert not any(
        item.get("authorized_candidate_id") == candidate_id
        for item in payload["expansion_handles"]
    )
    verify_production_evidence_authority(
        authority,
        repo_root=repo,
        current_task_id=task["task_id"],
        current_task_payload=task,
        expected_evidence_inputs=["src/greeting.py"],
        expected_evidence_refs=[declaration],
        expected_candidate_ref_authority_appendix=appendix,
        expected_board_namespace="pcce-board",
        expected_board_projection_id="board-projection:test",
        expected_candidate_authority_signer_did=signer_did,
        expected_priority_paths=["src/greeting.py"],
        expected_context_round=0,
    )

    _assert_reason(
        "evidence_ref_invalid",
        lambda: build_production_evidence_authority(
            repo_root=repo,
            task_id=task["task_id"],
            task_payload=task,
            evidence_inputs=["src/greeting.py"],
            evidence_refs=[f"refs/heads/wip/ref-evidence={candidate_commit}"],
            priority_paths=["src/greeting.py"],
        ),
    )


def test_operator_scope_binding_cannot_be_replaced_by_manifest_claims(tmp_path) -> None:
    repo = _repo(tmp_path)
    manifest = _manifest(repo)

    _assert_reason(
        "scope_authority_mismatch",
        lambda: verify_production_context_slice(
            manifest,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            expected_read_paths=["src/greeting.py"],
            expected_effect_paths=["src/greeting.py", "src/provider-chosen.py"],
        ),
    )


def test_scope_widening_secrets_and_prompt_overflow_fail_closed(tmp_path) -> None:
    repo = _repo(tmp_path)
    _assert_reason(
        "scope_widening",
        lambda: _manifest(
            repo,
            symbol_hints={"src/other.py": ["outside_scope"]},
        ),
    )

    target = repo / "src" / "greeting.py"
    target.write_text("api_key = 'super-secret-value'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "credential fixture")
    _assert_reason("secret_detected", lambda: _manifest(repo))

    target.write_text("value = 'safe but budgeted'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "safe budget fixture")
    _assert_reason(
        "context_budget_exceeded",
        lambda: _manifest(
            repo,
            max_provider_prompt_tokens=4096,
            reserved_prompt_tokens=4095,
        ),
    )


def test_ast_slice_has_byte_complete_residual_identity(tmp_path) -> None:
    source = (
        "\"\"\"Calculator utilities.\"\"\"\n"
        "import math\n\n"
        "class Calculator:\n"
        "    def add(self, left: int, right: int) -> int:\n"
        "        return left + right\n\n"
        "    def hidden_multiply(self, left: int, right: int) -> int:\n"
        "        return left * right\n\n"
        "def unrelated() -> float:\n"
        "    return math.pi\n"
    )
    repo = _repo(tmp_path, source)
    manifest = _manifest(
        repo,
        whole_file_bytes=1,
        symbol_hints={"src/greeting.py": ["Calculator.add"]},
    )
    record = manifest.to_dict()["sources"][0]
    visible_text = "".join(item["utf8_text"] for item in record["source_slices"])

    assert "def add" in visible_text
    assert "import math" in visible_text
    assert "hidden_multiply" not in visible_text
    assert record["full_visible_coverage"] is False
    assert record["residuals"]
    partition = sorted(
        [*record["source_slices"], *record["residuals"]],
        key=lambda item: item["byte_start"],
    )
    assert partition[0]["byte_start"] == 0
    assert partition[-1]["byte_end"] == len(source.encode("utf-8"))
    assert all(
        left["byte_end"] == right["byte_start"]
        for left, right in zip(partition, partition[1:], strict=False)
    )


def test_sliced_context_allows_only_visible_patch_preimages(tmp_path) -> None:
    source = (
        "class Calculator:\n"
        "    def add(self, left, right):\n"
        "        left_value = int(left)\n"
        "        right_value = int(right)\n"
        "        values = (left_value, right_value)\n"
        "        total = sum(values)\n"
        "        result = total\n"
        "        return result\n\n"
        "    def multiply(self, left, right):\n"
        "        return left * right\n"
    )
    repo = _repo(tmp_path, source)
    manifest = _manifest(
        repo,
        whole_file_bytes=1,
        symbol_hints={"src/greeting.py": ["Calculator.add"]},
    )
    full_replacement = {
        "files": [{"path": "src/greeting.py", "content": source.upper()}]
    }
    _assert_reason(
        "context_insufficient",
        lambda: assert_proposal_covered_by_context(
            manifest,
            full_replacement,
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(
                expected_symbol_hints={"src/greeting.py": ["Calculator.add"]}
            ),
        ),
    )

    visible_patch = (
        "diff --git a/src/greeting.py b/src/greeting.py\n"
        "--- a/src/greeting.py\n"
        "+++ b/src/greeting.py\n"
        "@@ -2,7 +2,7 @@ class Calculator:\n"
        "     def add(self, left, right):\n"
        "         left_value = int(left)\n"
        "         right_value = int(right)\n"
        "-        values = (left_value, right_value)\n"
        "+        values = [left_value, right_value]\n"
        "         total = sum(values)\n"
        "         result = total\n"
        "         return result\n"
    )
    mismatched_headers = visible_patch.replace(
        "--- a/src/greeting.py",
        "--- a/src/provider-chosen.py",
    )
    _assert_reason(
        "proposal_scope_violation",
        lambda: assert_proposal_covered_by_context(
            manifest,
            {"patch": mismatched_headers},
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(
                expected_symbol_hints={"src/greeting.py": ["Calculator.add"]}
            ),
        ),
    )

    boundary_insertion = (
        "diff --git a/src/greeting.py b/src/greeting.py\n"
        "--- a/src/greeting.py\n"
        "+++ b/src/greeting.py\n"
        "@@ -1,0 +2 @@\n"
        "+    # provider cannot see both sides of this boundary\n"
    )
    _assert_reason(
        "context_insufficient",
        lambda: assert_proposal_covered_by_context(
            manifest,
            {"patch": boundary_insertion},
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(
                expected_symbol_hints={"src/greeting.py": ["Calculator.add"]}
            ),
        ),
    )
    assert_proposal_covered_by_context(
        manifest,
        {"patch": visible_patch},
        repo_root=repo,
        current_task_id=TASK["task_id"],
        current_task_payload=TASK,
        **_expected_scope(
            expected_symbol_hints={"src/greeting.py": ["Calculator.add"]}
        ),
    )

    hidden_patch = (
        "diff --git a/src/greeting.py b/src/greeting.py\n"
        "--- a/src/greeting.py\n"
        "+++ b/src/greeting.py\n"
        "@@ -10,2 +10,2 @@ class Calculator:\n"
        "     def multiply(self, left, right):\n"
        "-        return left * right\n"
        "+        return int(left) * int(right)\n"
    )
    _assert_reason(
        "context_insufficient",
        lambda: assert_proposal_covered_by_context(
            manifest,
            {"patch": hidden_patch},
            repo_root=repo,
            current_task_id=TASK["task_id"],
            current_task_payload=TASK,
            **_expected_scope(
                expected_symbol_hints={"src/greeting.py": ["Calculator.add"]}
            ),
        ),
    )
