#!/bin/bash
set -exo pipefail

uv pip compile third_party/vbench/requirements.txt \
    -o third_party/vbench/requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
