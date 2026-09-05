#!/usr/bin/env bash
set -euo pipefail

git add .gitignore analyze_dmac.py analyze_dmac_multi.py tests/test_analyze_dmac.py \
  tests/test_analyze_dmac_multi.py README.md PROJECT_STATE.md gh.sh
git commit -m "feat: add single and multi-strategy DMAC analyzers"
git push -u origin main
