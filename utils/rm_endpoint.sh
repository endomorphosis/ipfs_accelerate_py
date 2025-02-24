#!/bin/bash
curl 127.0.0.1:9999/rm_endpoint \
    -X POST \
    -d '{"models":["thenlper/gte-small"], "endpoint_type": "local_endpoints" , "backend": "openvino" }' \
    -H 'Content-Type: application/json'

