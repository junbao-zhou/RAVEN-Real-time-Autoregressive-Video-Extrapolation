#!/bin/bash
set -exo pipefail

uv pip compile tools/requirements.txt \
    -o tools/requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
