#!/bin/sh
set -eu
cd /home/barberb/lift_coding/.worktrees/ipfs_accelerate-lpc/data/agent_supervisor/logic_platform_canonicalization/worktrees/workspace-596bf9b08990-8705dbac9573
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
/usr/bin/python3.12 -m pytest test/api/test_canonical_logic_adapter.py -q
