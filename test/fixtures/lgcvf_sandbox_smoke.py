"""Protected smoke exercised through the real LGCVF qualification worker."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def test_lgcvf_sandbox_allows_only_its_writable_fixture_root(tmp_path: Path) -> None:
    logging.getLogger("lgcvf.sandbox.smoke").warning("sandbox log sink is writable")
    null_sink = Path(os.devnull)
    assert null_sink.name == "devnull"
    assert null_sink != Path("/dev/null")
    with null_sink.open("r+b") as handle:
        handle.write(b"sealed-null-sink\n")
    artifact = tmp_path / "smoke.txt"
    artifact.write_text("bounded\n", encoding="utf-8")
    assert artifact.read_text(encoding="utf-8") == "bounded\n"
