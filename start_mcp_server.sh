#!/bin/bash
cd /home/barberb/ipfs_accelerate_py
source /home/barberb/ipfs_accelerate_py/.venv/bin/activate
export PYTHONPATH=/home/barberb/ipfs_accelerate_py
python cli.py mcp start --host 0.0.0.0 --port 9000