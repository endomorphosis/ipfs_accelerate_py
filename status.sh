#!/bin/bash
curl 127.0.0.1:9999/status \
    -X POST \
    -d '{"models":["thenlper/gte-small"]}' \
    -H 'Content-Type: application/json'