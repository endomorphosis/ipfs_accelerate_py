# Supervisor execution import lifetime

Execution-only Doctor, native owner, admitted runtime and route-policy imports
now occur after indexing and planning, in the phase that needs them. Imports
remain inside the existing work deadline; Doctor import time is charged to its
phase. No source-currentness check, lease requirement or deadline is removed.

All 56 actual focused controls pass in a fresh test store. Five fresh-interpreter
controls refuse execution imports to verify early-phase isolation, the later
phase boundary and cleanup. The first cached run had five passes and 51 AST-seal
skips; that result is retained separately and not counted as 56 actual passes.

Fresh-process host measurements reduced driver import RSS from 189280 KiB to
44916 KiB. After the same authored tiny vector-index build and shared producer
imports, RSS fell from 297432 KiB to 246880 KiB, near preparation's 246928 KiB.
Each measurement is a single process sample, not a statistical performance
study or container admission result. The change reduces avoidable live memory;
it does not establish why the preceding native resource request was refused.
Docker qualification and the full Harbor task have separate outcome receipts.
