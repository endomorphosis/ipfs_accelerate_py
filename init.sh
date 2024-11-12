#!/bin/bash
curl 127.0.0.1:9999/init \
    -X POST \
    -d '{"models":["thenlper/gte-small"], "resources": {"tei_endpoints": [ [ "thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", "512" ] , [ "thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", "512" ] ] , "local_endpoints": [ [ "thenlper/gte-small", "openvino:0", "512" ], [ "thenlper/gte-small", "cuda:0", "512" ], [ "thenlper/gte-small", "cuda:1", "512" ],  ] , "openvino_endpoints" : [["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", "4095"]] } }' \
    -H 'Content-Type: application/json'

