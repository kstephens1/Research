#!/usr/bin/env bash
set -euo pipefail

git add .gitignore analyze_holding.py requirements.txt tests/test_analyze_holding.py \
  README.md PROJECT_STATE.md gh.sh
git commit -m "feat: add VectorBT buy-and-hold analyzer"
git push -u origin main
