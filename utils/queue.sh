#!/bin/bash
curl 127.0.0.1:9999/queue \
    -X POST \
    -d '{"models":["thenlper/gte-small"], "batch_data": ["This is a test sentence."]}' \
    -H 'Content-Type: application/json'