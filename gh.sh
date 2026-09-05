#!/usr/bin/env bash
set -euo pipefail

git add .gitignore README.md PROJECT_STATE.md gh.sh
git commit -m "docs: add private Jupyter research workflow"
git push -u origin main
