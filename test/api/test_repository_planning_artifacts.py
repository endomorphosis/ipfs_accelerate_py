"""Planning artifacts cannot overwrite source through links or path aliases."""
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import repository_planning_artifacts as artifacts


@pytest.fixture
def destinations(tmp_path):
    source, state, output = (tmp_path / name for name in ("source", "state", "artifacts"))
    source.mkdir()
    state.mkdir()
    original = source / "answer.py"
    original.write_bytes(b"def answer(): return 1\n")
    return source, state, output, original


def write(destinations, *, directory=None, name="proposal.json", text="{}\n"):
    source, state, output, _original = destinations
    return artifacts.write_repository_planning_artifact(
        directory=output if directory is None else directory, name=name, text=text,
        excluded_roots=(source, state))


def test_regular_artifact_replacement_is_complete_and_private(destinations):
    source, state, output, original = destinations
    output.mkdir()
    previous = output / "proposal.json"
    previous.write_text("previous receipt")
    before_inode = previous.stat().st_ino
    text = '{"review":"café ✓"}\n'

    assert write(destinations, text=text) == len(text.encode("utf-8"))
    assert previous.read_bytes() == text.encode("utf-8")
    assert previous.stat().st_ino != before_inode
    assert previous.stat().st_mode & 0o777 == 0o600
    assert list(output.iterdir()) == [previous]
    assert original.read_bytes() == b"def answer(): return 1\n"


def test_unsafe_names_refuse_before_creating_artifact_directories(destinations):
    output = destinations[2]
    for name in ("", ".", "..", "../escape", "a/b", "a\\b", "a\x00b", "a\nb", "x" * 256):
        with pytest.raises(artifacts.RepositoryPlanningArtifactError, match="artifact_name"):
            write(destinations, name=name)
    assert not output.exists()


def test_hardlinks_and_nonregular_destinations_preserve_source(destinations):
    _source, _state, output, original = destinations
    output.mkdir()
    destination = output / "proposal.json"
    destination.hardlink_to(original)
    with pytest.raises(artifacts.RepositoryPlanningArtifactError, match="destination"):
        write(destinations, text="must not replace the linked source")
    assert destination.samefile(original)
    assert original.read_bytes() == b"def answer(): return 1\n"
    destination.unlink()
    destination.mkdir()
    with pytest.raises(artifacts.RepositoryPlanningArtifactError, match="destination"):
        write(destinations)
    assert destination.is_dir()
    assert list(output.iterdir()) == [destination]


def test_parent_symlinks_are_never_followed(destinations, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(OSError):
        write(destinations, directory=alias / "new-child")
    assert not (elsewhere / "new-child").exists()


def test_body_limit_counts_utf8_bytes_without_partial_replacement(destinations):
    output = destinations[2]
    exact = "é" * (artifacts.MAX_ARTIFACT_BYTES // 2)
    assert write(destinations, text=exact) == artifacts.MAX_ARTIFACT_BYTES
    with pytest.raises(artifacts.RepositoryPlanningArtifactError, match="byte_limit"):
        write(destinations, text=exact + "x")
    assert (output / "proposal.json").read_text() == exact
    assert list(output.iterdir()) == [output / "proposal.json"]


def test_failed_atomic_publication_removes_temporary_file(destinations, monkeypatch):
    output = destinations[2]
    write(destinations, text="original receipt")

    def refuse(*_arguments, **_keywords):
        raise OSError("authored publication failure")

    monkeypatch.setattr(artifacts.os, "replace", refuse)
    with pytest.raises(OSError, match="authored publication failure"):
        write(destinations, text="replacement receipt")
    assert (output / "proposal.json").read_text() == "original receipt"
    assert list(output.iterdir()) == [output / "proposal.json"]


def test_excluded_roots_and_their_ancestors_refuse_publication(destinations):
    source, state, _output, original = destinations
    for directory in (source, source / "nested", state, state / "nested", source.parent):
        with pytest.raises(artifacts.RepositoryPlanningArtifactError, match="overlap"):
            write(destinations, directory=directory)
    assert original.read_bytes() == b"def answer(): return 1\n"
    assert list(source.iterdir()) == [original]
    assert list(state.iterdir()) == []


def test_destination_link_inserted_at_replace_cannot_overwrite_source(destinations, monkeypatch):
    _source, _state, output, original = destinations
    output.mkdir()
    destination = output / "proposal.json"
    replace = artifacts.os.replace

    def insert_link_then_replace(*arguments, **keywords):
        destination.symlink_to(original)
        return replace(*arguments, **keywords)

    monkeypatch.setattr(artifacts.os, "replace", insert_link_then_replace)
    assert write(destinations, text="new receipt") == len(b"new receipt")
    assert not destination.is_symlink()
    assert destination.read_text() == "new receipt"
    assert original.read_bytes() == b"def answer(): return 1\n"
    assert list(output.iterdir()) == [destination]
