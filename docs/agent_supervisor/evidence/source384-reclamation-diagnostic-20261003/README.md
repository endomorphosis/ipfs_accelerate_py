# Process reclamation diagnostic

This instrumented container run passes native preparation and replay with the
unchanged archive and limits. Initial context takes 145.804s and the complete
probe 168.970s. Immediately before warm replay, garbage collection collects
43,539 objects without reducing observed RSS. glibc malloc_trim(0) then reduces
RSS from 459616 to 450860 KiB (8756 KiB); the interval takes 0.189s.
The live context hashes are identical before and after the intervention.

Before reclamation the cgroup already reports 10998 MiB available, unlike the
failed ordinary warm run's after-unwind 8615 MiB. Thus this does not reproduce
that refusal or establish allocator retention as its cause. Source pins and
permitted task files are unchanged; all normal checks and deadlines execute.
Container cleanup is verified. No allocator trimming is added to production.
This is diagnostic evidence, not production qualification, a task score,
source-qualified proof or a benchmark-efficiency comparison.
