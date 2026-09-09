# Retained attempt dispatch barrier

A running attempt belonging to the logical owner remains a dispatch barrier when a changed prefix or shard selection excludes it. Missing process metadata, denied process inspection, a dead PID, or a closed daemon record cannot settle the attempt or prove that provider effects are absent.

The daemon first reconciles prepared completions and expired coordination claims across the owner's full attempt population. Only a native terminal transition removes a running attempt from that population. The expiry pass does not dispatch new provider work; a later pass may dispatch once no retained running attempt blocks it. Durable provider or effect evidence continues through the existing blocked or completion reconciliation paths.

Native DuckDB regression coverage verifies the unchanged shared task, claim, and attempt across repeated passes for missing metadata, malformed birth records, denied inspection, closed or dead processes, PID reuse, and a live process. A positive case verifies exact lease expiry, terminalization, and later dispatch in separate passes.
