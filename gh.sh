#!/usr/bin/env bash
set -euo pipefail

git add AGENTS.md .gitignore PROJECT_STATE.md gh.sh
git commit -m "chore: initialize research repository"
git push
