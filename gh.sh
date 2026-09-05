#!/usr/bin/env bash
set -euo pipefail

git add AGENTS.md .gitignore PROJECT_STATE.md gh.sh
git commit -m "chore: record GitHub repository setup"
git push -u origin main
