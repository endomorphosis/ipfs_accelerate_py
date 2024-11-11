#!/bin/bash
curl 127.0.0.1:9999/init \
    -X POST \
    -d '{"models":["thenlper/gte-small"], "resources": {"tei_endpoints": [ [ "thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", "512" ] , [ "thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", "512" ] ] , "local_endpoints": [ [ "thenlper/gte-small", "openvino", "512" ] ] } }' \
    -H 'Content-Type: application/json'

