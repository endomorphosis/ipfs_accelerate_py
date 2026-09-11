#!/usr/bin/env python3
"""Exact isolated-interpreter entry point for the retained SPAR capture driver."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ops.agent_supervisor.spar_retained_capture_driver import main

if __name__ == "__main__":
    raise SystemExit(main())
