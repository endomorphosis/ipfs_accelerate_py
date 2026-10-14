"""Real source-view custody and pure edit mapping; no model or target execution."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit as module
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews, canonical_json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def case(tmp_path, request):
    variant = getattr(request, "param", "lf")
    text = 'def value():\n    note = "雪"\n    return 1 # hidden ordinary explanation\n'
    if variant == "crlf":
        text = text.replace("\n", "\r\n")
    elif variant == "no_final_lf":
        text = text.rstrip("\n")
    elif variant == "fallback":
        text = 'count = 1\nmessage = f"{count}"\n'
    elif variant == "long":
        text += "# ordinary explanation retained outside the compact view\n" * 100
    root, workspace = tmp_path / "canonical", tmp_path / "allocated"
    root.mkdir()
    sources = {"pkg/code.py": text.encode(), "other.py": b"OTHER = 2\n", "instruction.md": b"public instruction\n"}
    for name, raw in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Authored Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    git(root, "worktree", "add", "--detach", "-q", str(workspace))
    policy = SourceViewPolicy.from_mapping(profile="python-comments@1", protected_lines={}) if variant == "shared" else None
    views = WorkerSourceViews(repository=root, workspace=workspace,
        source_hashes={name: sha(raw) for name, raw in sources.items()}, task_id="task:authored",
        scope_cid="scope:authored", semantic_root_cid="semantic:authored", semantic_sha256="a" * 64,
        native_prompt_sha256="b" * 64, source_manifest_sha256="c" * 64,
        **({"view_policy": policy} if policy else {"profile": "python-comments@1"}))
    first = views.working_page(path="pkg/code.py")
    projected = views.working_projection(path="pkg/code.py", expected_workspace_generation=first["workspace_generation"])
    offset = projected.view.index(b"1")
    binding = views.baseline_admission().get("view_policy_binding")
    query = dict(operation="prepare", path="pkg/code.py", expected_workspace_generation=first["workspace_generation"],
        expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"],
        expected_manifest_sha256=first["manifest_sha256"], view_start=offset, view_end=offset + 1,
        expected_view_preimage="1", replacement="23", max_bytes=4096,
        view_policy_binding_sha256=None if binding is None else sha(canonical_json(binding)))
    return dict(root=root, workspace=workspace, sources=sources, views=views, query=query, first=first,
        service=module.WorkerSourceEditService(views=views, allowed_paths=("pkg/code.py", "other.py")))


def invoke(case, query=None):
    query = case["query"] if query is None else query
    envelope = dict(schema="supervisor-worker-source-edit-request@1", session_id="a" * 32,
                    request_id="b" * 32, query=query)
    return case["service"].handle(query, request_sha256=sha(canonical_json(envelope)))


def confirm_query(case, prepared):
    return dict(operation="confirm", proposal_sha256=prepared["proposal_sha256"],
                view_policy_binding_sha256=case["query"]["view_policy_binding_sha256"], max_bytes=4096)


def simulate_existing_worker_write(case, prepared):
    """Fixture-owned write; production service must perform none."""
    before = (case["workspace"] / prepared["path"]).read_bytes()
    assert sha(before) == prepared["before_source_sha256"]
    query = case["query"]
    start, end = prepared["original_start"], prepared["original_end"]
    assert before[start:end] == query["expected_view_preimage"].encode()
    candidate = before[:start] + query["replacement"].encode() + before[end:]
    assert sha(candidate) == prepared["candidate_sha256"]
    (case["workspace"] / prepared["path"]).write_bytes(candidate)
    return candidate


def assert_unwritten(case):
    for location in ("root", "workspace"):
        assert all((case[location] / name).read_bytes() == raw for name, raw in case["sources"].items())
    assert git(case["workspace"], "status", "--porcelain=v1") == b""


@pytest.mark.parametrize("case", ["lf", "crlf", "no_final_lf", "shared", "fallback"], indirect=True)
def test_exact_prepare_worker_write_confirm_has_no_owner_writes_or_authority(case):
    prepared = invoke(case)
    assert_unwritten(case)
    assert prepared["schema"] == module.PREPARED_SCHEMA
    assert prepared["status"] == "prepared_validation_pending"
    assert prepared["source_postimage_verified"] is False
    assert len(canonical_json(prepared)) <= 4096 - module.RESPONSE_RESERVE_BYTES
    assert not any(token in canonical_json(prepared) for token in (b"hidden ordinary", b"note =", b"return 1"))
    candidate = simulate_existing_worker_write(case, prepared)
    confirmed = invoke(case, confirm_query(case, prepared))
    assert confirmed["schema"] == module.CONFIRMED_SCHEMA
    assert confirmed["status"] == "applied_validation_pending"
    assert confirmed["source_postimage_verified"] is True
    assert confirmed["workspace_generation"] != prepared["source_generation"]
    assert confirmed["baseline_evidence_applicable"] is False
    assert len(canonical_json(confirmed)) <= 4096 - module.RESPONSE_RESERVE_BYTES
    expected = case["sources"] | {"pkg/code.py": candidate}
    assert case["views"].working_sources(expected_workspace_generation=confirmed["workspace_generation"]) == expected
    assert confirmed["source_population_sha256"] == sha(canonical_json({path: sha(raw) for path, raw in expected.items()}))
    assert (case["root"] / "pkg/code.py").read_bytes() == case["sources"]["pkg/code.py"]
    for row in (prepared, confirmed):
        assert row["validation_pending"] is True
        for flag in ("owner_writes_performed", "proof_authority", "execution_authority", "publication_authority",
                     "completion_authority", "mutation_authority", "semantic_equivalence_claimed"):
            assert row[flag] is False
        assert row["source_obligations_discharged"] == row["provider_calls"] == row["prover_calls"] == 0
    with pytest.raises(ValueError):
        case["views"].working_sources(expected_workspace_generation=prepared["source_generation"])


def test_unicode_preimage_and_replacement_are_exact_byte_coordinates(case):
    projected = case["views"].working_projection(path="pkg/code.py",
        expected_workspace_generation=case["first"]["workspace_generation"])
    start = projected.view.index("雪".encode())
    case["query"].update(view_start=start, view_end=start + 3, expected_view_preimage="雪", replacement="山川")
    prepared = invoke(case)
    candidate = simulate_existing_worker_write(case, prepared)
    confirmed = invoke(case, confirm_query(case, prepared))
    assert confirmed["preimage_bytes"] == 3 and confirmed["replacement_bytes"] == 6
    assert candidate == case["sources"]["pkg/code.py"].replace("雪".encode(), "山川".encode())
    assert b"# hidden ordinary explanation" in candidate


@pytest.mark.parametrize("field,value", [
    ("view_start", True), ("view_end", 1.2), ("view_start", -1), ("view_end", 1024 * 1024 + 1),
    ("path", "../outside.py"), ("path", ".git/config"), ("path", "/absolute.py"),
    ("replacement", "\ud800"), ("replacement", "\0"), ("replacement", "a" * 4097),
    ("replacement", "雪" * 1366), ("replacement", ["unexpected"]),
    ("expected_view_preimage", b"1"), ("expected_view_preimage", "12"),
    ("expected_workspace_generation", None), ("expected_source_sha256", "A" * 64),
    ("expected_view_sha256", False), ("expected_manifest_sha256", "short"),
    ("view_policy_binding_sha256", {}), ("max_bytes", True), ("max_bytes", 4095), ("max_bytes", 8193),
])
def test_invalid_queries_refuse_closed_and_do_not_consume_or_write(case, field, value):
    query = dict(case["query"], **{field: value})
    with pytest.raises(module.WorkerSourceEditError, match="^invalid_request$"):
        case["service"].handle(query, request_sha256="d" * 64)
    assert_unwritten(case)
    assert invoke(case)["schema"] == module.PREPARED_SCHEMA


@pytest.mark.parametrize("operation", [None, True, [], {}, "apply", ""])
def test_unknown_operations_fail_closed(case, operation):
    with pytest.raises(module.WorkerSourceEditError, match="^invalid_request$"):
        case["service"].handle({"operation": operation}, request_sha256="d" * 64)
    assert_unwritten(case)


@pytest.mark.parametrize("alter", ["extra", "missing", "escaped_size", "bad_request_digest"])
def test_closed_request_and_wire_bounds(case, alter):
    query = dict(case["query"])
    digest = "d" * 64
    if alter == "extra":
        query["lease_id"] = "worker cannot nominate an owner"
    elif alter == "missing":
        del query["expected_manifest_sha256"]
    elif alter == "escaped_size":
        query["replacement"] = "\n" * 4096
    else:
        digest = "bad"
    with pytest.raises(module.WorkerSourceEditError, match="^invalid_request$"):
        case["service"].handle(query, request_sha256=digest)
    assert_unwritten(case)


@pytest.mark.parametrize("field", ["expected_workspace_generation", "expected_source_sha256",
                                  "expected_view_sha256", "expected_manifest_sha256"])
def test_stale_pins_are_not_silently_refreshed(case, field):
    with pytest.raises(module.WorkerSourceEditError):
        invoke(case, dict(case["query"], **{field: "0" * 64}))
    assert_unwritten(case)


@pytest.mark.parametrize("path", ["instruction.md", "missing.py"])
def test_source_population_does_not_grant_program_edit_scope(case, path):
    with pytest.raises(module.WorkerSourceEditError, match="^mapping_refused$"):
        invoke(case, dict(case["query"], path=path))
    assert_unwritten(case)


@pytest.mark.parametrize("paths", [(), ["missing.py"], ["pkg/code.py", "pkg/code.py"], "pkg/code.py", [None]])
def test_constructor_requires_explicit_admitted_program_paths(case, paths):
    with pytest.raises(module.WorkerSourceEditError, match="^edit_unavailable$"):
        module.WorkerSourceEditService(views=case["views"], allowed_paths=paths)
    assert_unwritten(case)


def test_constructor_does_not_accept_a_duck_typed_or_fake_source_owner():
    with pytest.raises(module.WorkerSourceEditError, match="^edit_unavailable$"):
        module.WorkerSourceEditService(views={}, allowed_paths=("code.py",))


def test_mapper_rejection_can_retry_before_a_successful_preparation(case):
    with pytest.raises(module.WorkerSourceEditError, match="^mapping_refused$"):
        invoke(case, dict(case["query"], expected_view_preimage="9"))
    assert_unwritten(case)
    assert invoke(case)["schema"] == module.PREPARED_SCHEMA


@pytest.mark.parametrize("kind", ["cross_hidden", "hidden_insert", "syntax_transition", "no_op"])
def test_existing_mapper_fences_are_not_relaxed(case, kind):
    projection = case["views"].working_projection(path="pkg/code.py",
        expected_workspace_generation=case["first"]["workspace_generation"])
    query = dict(case["query"])
    if kind == "cross_hidden":
        start = query["view_start"]
        query.update(view_end=len(projection.view), expected_view_preimage=projection.view[start:].decode())
    elif kind == "hidden_insert":
        at = projection.view.index(b"\n", query["view_start"])
        query.update(view_start=at, view_end=at, expected_view_preimage="")
    elif kind == "syntax_transition":
        query["replacement"] = "("
    else:
        query["replacement"] = "1"
    with pytest.raises(module.WorkerSourceEditError, match="^mapping_refused$"):
        invoke(case, query)
    assert_unwritten(case)


@pytest.mark.parametrize("location,path", [("root", "pkg/code.py"), ("root", "instruction.md"),
                                           ("workspace", "other.py")])
def test_whole_admitted_preimage_is_checked_before_mapping(case, location, path):
    changed = case[location] / path
    changed.write_bytes(b"external change\n")
    with pytest.raises(module.WorkerSourceEditError):
        invoke(case)
    assert changed.read_bytes() == b"external change\n"
    if path != "pkg/code.py" or location != "workspace":
        assert (case["workspace"] / "pkg/code.py").read_bytes() == case["sources"]["pkg/code.py"]


def test_source_drift_during_pure_mapping_cannot_return_a_proposal(case, monkeypatch):
    native = module.map_source_view_edit
    def drift(**kwargs):
        result = native(**kwargs)
        (case["workspace"] / "other.py").write_bytes(b"OTHER = 3\n")
        return result
    monkeypatch.setattr(module, "map_source_view_edit", drift)
    with pytest.raises(module.WorkerSourceEditError, match="^mapping_refused$"):
        invoke(case)
    assert case["service"]._proposal is None
    assert (case["workspace"] / "pkg/code.py").read_bytes() == case["sources"]["pkg/code.py"]


@pytest.mark.parametrize("phase", ["prepared", "confirmed"])
def test_successful_proposal_and_confirmation_are_one_shot(case, phase):
    prepared = invoke(case)
    if phase == "confirmed":
        simulate_existing_worker_write(case, prepared)
        invoke(case, confirm_query(case, prepared))
        with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
            invoke(case, confirm_query(case, prepared))
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case)


@pytest.mark.parametrize("kind", ["ticket", "no_write", "wrong_write", "sibling", "canonical", "symlink", "hardlink", "policy"])
def test_failed_confirmation_never_claims_postimage_or_retries_write(case, kind, tmp_path):
    prepared = invoke(case)
    query = confirm_query(case, prepared)
    if kind != "no_write":
        simulate_existing_worker_write(case, prepared)
    if kind == "ticket":
        query["proposal_sha256"] = "0" * 64
    elif kind == "wrong_write":
        (case["workspace"] / "pkg/code.py").write_bytes(b"unexpected\n")
    elif kind == "sibling":
        (case["workspace"] / "other.py").write_bytes(b"OTHER = 99\n")
    elif kind == "canonical":
        (case["root"] / "other.py").write_bytes(b"OTHER = 99\n")
    elif kind in {"symlink", "hardlink"}:
        target = case["workspace"] / "pkg/code.py"
        outside = tmp_path / "outside.py"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        if kind == "symlink":
            target.symlink_to(outside)
        else:
            os.link(outside, target)
    elif kind == "policy":
        query["view_policy_binding_sha256"] = "0" * 64
    before = (case["workspace"] / "pkg/code.py").read_bytes()
    with pytest.raises(module.WorkerSourceEditError, match="^confirmation_refused$"):
        invoke(case, query)
    assert (case["workspace"] / "pkg/code.py").read_bytes() == before
    assert case["views"]._working is None
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case, confirm_query(case, prepared))


def test_mutating_returned_proposal_cannot_change_owner_expected_postimage(case):
    prepared = invoke(case)
    actual = deepcopy(prepared)
    prepared.update(candidate_sha256="0" * 64, original_start=0, proposal_sha256="0" * 64)
    simulate_existing_worker_write(case, actual)
    assert invoke(case, confirm_query(case, actual))["candidate_sha256"] == actual["candidate_sha256"]


def test_response_budget_failure_is_before_proposal_consumption(case, monkeypatch):
    monkeypatch.setattr(module, "RESPONSE_RESERVE_BYTES", 4000)
    with pytest.raises(module.WorkerSourceEditError, match="^invalid_request$"):
        invoke(case)
    assert case["service"]._proposal is None
    assert_unwritten(case)


def test_confirm_before_prepare_does_not_create_or_write_anything(case):
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case, dict(operation="confirm", proposal_sha256="0" * 64,
                         view_policy_binding_sha256=None, max_bytes=4096))
    assert_unwritten(case)


def repeated_service(case):
    case["service"] = module.WorkerSourceEditService(views=case["views"],
        allowed_paths=("pkg/code.py", "other.py"), mode=module.REPEATED_MODE)


def fresh_edit(case, *, path="pkg/code.py", preimage="1", replacement="23"):
    page = case["views"].working_page(path=path)
    projection = case["views"].working_projection(path=path,
        expected_workspace_generation=page["workspace_generation"])
    offset = projection.view.index(preimage.encode())
    case["query"] = dict(case["query"], path=path,
        expected_workspace_generation=page["workspace_generation"],
        expected_source_sha256=page["source_sha256"], expected_view_sha256=page["view_sha256"],
        expected_manifest_sha256=page["manifest_sha256"], view_start=offset,
        view_end=offset + len(preimage.encode()), expected_view_preimage=preimage,
        replacement=replacement)
    return case["query"]


def apply_and_confirm(case):
    prepared = invoke(case)
    candidate = simulate_existing_worker_write(case, prepared)
    confirmed = invoke(case, confirm_query(case, prepared))
    return prepared, candidate, confirmed


@pytest.mark.parametrize("case", ["lf", "crlf", "no_final_lf", "shared", "fallback"], indirect=True)
def test_repeated_edits_same_then_other_program_preserve_exact_population(case):
    repeated_service(case)
    before = dict(case["sources"])
    generations = {case["first"]["workspace_generation"]}
    first, candidate, confirmed = apply_and_confirm(case)
    before["pkg/code.py"] = candidate
    generations.add(confirmed["workspace_generation"])
    first_fields, confirmed_fields = set(first), set(confirmed)
    for path, preimage, replacement in (("pkg/code.py", "23", "345"), ("other.py", "2", "67")):
        fresh_edit(case, path=path, preimage=preimage, replacement=replacement)
        prepared, candidate, confirmed = apply_and_confirm(case)
        assert set(prepared) == first_fields and set(confirmed) == confirmed_fields
        assert prepared["source_generation"] in generations
        assert confirmed["workspace_generation"] not in generations
        generations.add(confirmed["workspace_generation"])
        before[path] = candidate
        assert case["views"].working_sources(expected_workspace_generation=confirmed["workspace_generation"]) == before
        assert confirmed["source_population_sha256"] == sha(canonical_json({name: sha(raw) for name, raw in before.items()}))
        assert confirmed["validation_pending"] and not confirmed["baseline_evidence_applicable"]
        assert confirmed["source_obligations_discharged"] == 0
    assert len(generations) == 4
    assert before["pkg/code.py"] == case["sources"]["pkg/code.py"].replace(b"1", b"345", 1)
    assert before["other.py"] == b"OTHER = 67\n"
    assert all((case["root"] / path).read_bytes() == raw for path, raw in case["sources"].items())
    assert (case["workspace"] / "instruction.md").read_bytes() == case["sources"]["instruction.md"]


@pytest.mark.parametrize("mode", [None, True, [], {}, "mapped-span@3", ""])
def test_invalid_modes_do_not_create_an_edit_service(case, mode):
    with pytest.raises(module.WorkerSourceEditError, match="^edit_unavailable$"):
        module.WorkerSourceEditService(views=case["views"], allowed_paths=("pkg/code.py",), mode=mode)
    assert_unwritten(case)


def test_repeated_mode_has_only_one_outstanding_proposal(case):
    repeated_service(case)
    prepared = invoke(case)
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case, dict(case["query"], replacement="34"))
    assert_unwritten(case)
    simulate_existing_worker_write(case, prepared)
    invoke(case, confirm_query(case, prepared))
    fresh_edit(case, preimage="23", replacement="45")
    assert invoke(case)["status"] == "prepared_validation_pending"


@pytest.mark.parametrize("mode", [module.MODE, module.REPEATED_MODE])
@pytest.mark.parametrize("outcome", ["confirmed", "failed"])
def test_pending_confirmation_matcher_is_read_only_and_never_revives_consumed_ticket(case, mode, outcome):
    service = module.WorkerSourceEditService(views=case["views"],
        allowed_paths=("pkg/code.py", "other.py"), mode=mode)
    case["service"] = service
    assert service.pending_confirmation_matches("0" * 64) is False
    prepared = invoke(case)
    snapshot = (service._proposal, service._confirmation_attempted,
        service._successful_preparations, case["views"]._working, case["views"]._working_counter)
    for ticket in (None, [], {}, True, 1, "", "A" * 64, "0" * 64):
        assert service.pending_confirmation_matches(ticket) is False
    assert service.pending_confirmation_matches(prepared["proposal_sha256"]) is (mode == module.REPEATED_MODE)
    assert (service._proposal, service._confirmation_attempted,
        service._successful_preparations, case["views"]._working, case["views"]._working_counter) == snapshot
    assert_unwritten(case)
    if outcome == "confirmed":
        simulate_existing_worker_write(case, prepared)
        invoke(case, confirm_query(case, prepared))
    else:
        with pytest.raises(module.WorkerSourceEditError, match="^confirmation_refused$"):
            invoke(case, confirm_query(case, prepared))
    assert service.pending_confirmation_matches(prepared["proposal_sha256"]) is False
    if mode == module.REPEATED_MODE and outcome == "confirmed":
        fresh_edit(case, preimage="23", replacement="45")
        next_prepared = invoke(case)
        assert service.pending_confirmation_matches(prepared["proposal_sha256"]) is False
        assert service.pending_confirmation_matches(next_prepared["proposal_sha256"]) is True


@pytest.mark.parametrize("stale", ["old", "foreign"])
@pytest.mark.parametrize("written", [False, True])
def test_repeated_mode_stale_confirmation_cannot_consume_new_active_proposal(case, stale, written):
    repeated_service(case)
    old, _, _ = apply_and_confirm(case)
    fresh_edit(case, preimage="23", replacement="45")
    active = invoke(case)
    if written:
        simulate_existing_worker_write(case, active)
    snapshot = case["views"]._working
    query = confirm_query(case, old)
    if stale == "foreign":
        query["proposal_sha256"] = "0" * 64
    with pytest.raises(module.WorkerSourceEditError, match="^confirmation_refused$"):
        invoke(case, query)
    assert case["views"]._working is snapshot
    if not written:
        simulate_existing_worker_write(case, active)
    assert invoke(case, confirm_query(case, active))["source_postimage_verified"]
    fresh_edit(case, preimage="45", replacement="67")
    assert invoke(case)["status"] == "prepared_validation_pending"


@pytest.mark.parametrize("failure", ["no_write", "wrong_write", "sibling", "canonical", "policy", "response_budget"])
def test_repeated_mode_matching_failed_confirmation_blocks_even_after_repair(case, failure, monkeypatch):
    repeated_service(case)
    prepared = invoke(case)
    if failure != "no_write":
        expected = simulate_existing_worker_write(case, prepared)
    else:
        expected = case["sources"]["pkg/code.py"].replace(b"1", b"23", 1)
    query = confirm_query(case, prepared)
    if failure == "wrong_write":
        (case["workspace"] / "pkg/code.py").write_bytes(b"wrong source\n")
    elif failure == "sibling":
        (case["workspace"] / "other.py").write_bytes(b"OTHER = 999\n")
    elif failure == "canonical":
        (case["root"] / "other.py").write_bytes(b"OTHER = 999\n")
    elif failure == "policy":
        query["view_policy_binding_sha256"] = "0" * 64
    elif failure == "response_budget":
        monkeypatch.setattr(module, "RESPONSE_RESERVE_BYTES", 4000)
    with pytest.raises(module.WorkerSourceEditError, match="^confirmation_refused$"):
        invoke(case, query)
    assert case["views"]._working is None
    monkeypatch.setattr(module, "RESPONSE_RESERVE_BYTES", 1024)
    (case["workspace"] / "pkg/code.py").write_bytes(expected)
    for location in ("root", "workspace"):
        (case[location] / "other.py").write_bytes(case["sources"]["other.py"])
    # Repairing bytes and reading a fresh generation cannot retry an uncertain
    # matching confirmation or reopen its consumed edit slot.
    fresh_edit(case, preimage="23", replacement="45")
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case, confirm_query(case, prepared))
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case)


def test_repeated_successful_preparations_are_bounded_at_sixteen(case):
    repeated_service(case)
    generations, proposals = {case["first"]["workspace_generation"]}, set()
    value = "1"
    assert module.MAX_PREPARATIONS == 16
    for number in range(16):
        next_value = str(number + 2)
        fresh_edit(case, preimage=value, replacement=next_value)
        # A rejected mapping neither writes nor consumes a successful slot.
        with pytest.raises(module.WorkerSourceEditError, match="^mapping_refused$"):
            invoke(case, dict(case["query"], expected_view_preimage="x" * len(value)))
        prepared, _, confirmed = apply_and_confirm(case)
        assert prepared["proposal_sha256"] not in proposals
        assert confirmed["workspace_generation"] not in generations
        proposals.add(prepared["proposal_sha256"]); generations.add(confirmed["workspace_generation"])
        value = next_value
    fresh_edit(case, preimage=value, replacement="18")
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case)
    with pytest.raises(module.WorkerSourceEditError, match="^proposal_consumed$"):
        invoke(case, confirm_query(case, prepared))
    assert (case["workspace"] / "pkg/code.py").read_bytes() == case["sources"]["pkg/code.py"].replace(b"1", b"17", 1)
    assert len(proposals) == 16 and len(generations) == 17


@pytest.mark.parametrize("case", ["long"], indirect=True)
def test_repeated_edit_then_revert_cannot_revive_old_generation_cursor_or_proposal(case):
    repeated_service(case)
    first_query = deepcopy(case["query"])
    old_page = case["views"].working_page(path="pkg/code.py", view="original", max_bytes=2048)
    assert old_page["next_cursor"] is not None
    old, _, forward = apply_and_confirm(case)
    fresh_edit(case, preimage="23", replacement="1")
    _, reverted, backward = apply_and_confirm(case)
    assert reverted == case["sources"]["pkg/code.py"]
    assert len({old_page["workspace_generation"], forward["workspace_generation"], backward["workspace_generation"]}) == 3
    with pytest.raises(ValueError):
        case["views"].working_page(path="pkg/code.py", view="original", max_bytes=2048,
            cursor=old_page["next_cursor"], expected_workspace_generation=old_page["workspace_generation"])
    with pytest.raises(module.WorkerSourceEditError):
        invoke(case, first_query)
    fresh_edit(case, replacement="9")
    active = invoke(case)
    assert active["proposal_sha256"] != old["proposal_sha256"]
    with pytest.raises(module.WorkerSourceEditError, match="^confirmation_refused$"):
        invoke(case, confirm_query(case, old))
    simulate_existing_worker_write(case, active)
    assert invoke(case, confirm_query(case, active))["source_postimage_verified"]
