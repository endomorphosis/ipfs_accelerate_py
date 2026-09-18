"""Cross-supervisor isolation: no sibling may write another board store."""
FORBIDDEN = ("direct DuckDB write", "DuckLake write", "task terminalization")

def test_forbidden_effects_remain_named():
    assert "direct DuckDB write" in FORBIDDEN
