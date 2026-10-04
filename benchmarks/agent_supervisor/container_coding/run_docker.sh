#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$bench_dir/../../.." && pwd)"
output_dir="${1:-$repo_dir/artifacts/container_coding_qualification}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
image_tag=ipfs-supervisor-coding-qualification:local
docker build -t "$image_tag" "$bench_dir"
image_id="$(docker image inspect --format '{{.Id}}' "$image_tag")"
printf '%s\n' "$image_id" > "$output_dir/image-id.txt"
git -C "$repo_dir" rev-parse HEAD > "$output_dir/source-head.txt"
git -C "$repo_dir" status --short > "$output_dir/source-status.txt"
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 3 \
  --tmpfs /tmp:rw,nosuid,size=128m \
  --mount "type=bind,src=$repo_dir,dst=/source,readonly" \
  --entrypoint python "$image_id" -m unittest discover -s /opt/benchmark -p test_run.py
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 3 \
  --tmpfs /tmp:rw,nosuid,size=128m \
  --mount "type=bind,src=$repo_dir,dst=/source,readonly" \
  --mount "type=bind,src=$output_dir,dst=/results" \
  "$image_id" --output /results/qualification.json --repeats 3
