from pathlib import Path
def test_three_supervisor_ten_daemon_fixture_exists():
    root = Path(__file__).resolve().parents[2] / "test/fixtures/logic_governed_semantic_work_fabric"
    assert (root / "manifest.json").is_file()
    data = (root / "manifest.json").read_text(encoding="utf-8")
    assert "supervisors" in data and "daemons" in data
