# Worker argument identity

Worker diagnostics prefer exact procfs arguments over empty or truncated process-list display text. A display string cannot override an available exact executable, module or script position. The decoder preserves empty downstream values, and recognition supports ordinary bundled Python flags while rejecting inline Python commands and malformed argument records.

This changes existing diagnostic recognition only. Generic main retains its existing sealed-runner receipt checks and ordinary worker diagnostic contract; it does not acquire the separate native PCTDD ordinary-provider receipt subsystem. No diagnostic match grants task completion, callback closure, retry or restart authority. The native PCTDD counterpart separately retains its stronger active-attempt ordinary receipt checks.

Regression tests cover missing and conflicting display text, empty arguments, interpreter/module/script positions and procfs decoding.
