#!/bin/bash
cd /home/devel/ipfs_accelerate_py
source ipfs_env/bin/activate
export PYTHONPATH=/home/devel/ipfs_accelerate_py
python mcp_jsonrpc_server.py --host 0.0.0.0 --port 9000