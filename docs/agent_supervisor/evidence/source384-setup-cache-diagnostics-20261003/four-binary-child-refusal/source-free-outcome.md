# Provider-binary cache diagnostic: child admission still refused

Both advice stages completed: **26905 archive files /3914232094 bytes**, followed by **four pinned public Codex executable copies /628736528 bytes**. All four bodies matched independent pins and the post-worker-boundary exposure receipt. The helpers reported no advice errors and unchanged executable metadata. These byte totals measure advised files, not physical bytes freed.

Observed availability was 5256 → 8194 MiB across archive advice, then 8194 → 8812 MiB across binary advice. Actual cgroups remained **5 CPU /12288 MiB**.

The native root lease admitted. The exact traceback enters `index.prepare_current` and refuses its first child resource lease, before repository snapshot/publication and model inference. Owner-store initialization may already have occurred. The failure-handling sample was8592 MiB;8602 MiB was the earlier root estimate, so comparing those numbers does not identify the child decision or establish its cause.

The result remains **qualified=false**. Initial context lasted37.209 seconds, the native probe48.697 seconds, and the whole diagnostic298.584 seconds. Model-load and coverage counters remain **null**; no inference export exists. Provider calls and official verifier execution were both reported zero/false. All218 original inputs matched across deployment, before/after source pins matched, and the retained Docker filter found no remaining container.

`source-free-outcome.json` contains selected scalar/count metadata and receipt hashes; `independent-outcome-audit.json` binds the exact helper generation, trace, frozen owner control flow, exposure hashes and preservation checks. The subsequently written owner verdict is `verdict.json` (SHA256 `b05da263ce4af6b60e8697409d4b714a7a1b99adda502249326f73d0387021c9`), which independently reaches the same child-admission conclusion. No raw source, embeddings, native model bodies, auth or executable payloads were copied.

Earlier outcomes stay separate: `../preceding-archive-only-outcome.json` records the archive-only root refusal at8195 MiB, and `../first-safe-refusal-outcome.json` records the uploaded-public-receipt protection refusal before binary reads/advice. This instrumented diagnostic does not establish production qualification, a benchmark score, or an efficiency advantage. No production cache policy was enabled.
