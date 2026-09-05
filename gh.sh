#!/usr/bin/env bash
set -euo pipefail

git add analyze_holding.py compare_strategies_by_symbol.py \
  tests/test_analyze_holding.py tests/test_compare_strategies_by_symbol.py \
  README.md PROJECT_STATE.md gh.sh
git commit -m "feat: add out-of-sample ticker strategy comparison"
git push -u origin main
