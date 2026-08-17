from pathlib import Path
def test_benchmark_corpus_exists():
    root = Path(__file__).resolve().parents[2] / "benchmarks/logic_governed_semantic_work_fabric"
    assert (root / "manifest.json").is_file()
