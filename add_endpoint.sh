#!/bin/bash
curl 127.0.0.1:9999/add_endpoint \
    -X POST \
    -d '{"models":["thenlper/gte-small"], "endpoint_type": "local_endpoints" , ["thenlper/gte-small", "openvino", "512"] }' \
    -H 'Content-Type: application/json'
