#!/usr/bin/env bash
# One-shot: initialise the repo, commit, create it on GitHub, push.
set -euo pipefail

REPO_NAME="ar-school-tour"
VISIBILITY="--public"

cd "$(dirname "$0")"
echo "==> working in $(pwd)"

[ -f README.md ] && [ -d scripts ] || {
    echo "!! README.md or scripts/ missing — wrong folder?"; exit 1; }

git config --global user.name  >/dev/null 2>&1 || git config --global user.name  "viethaa"
git config --global user.email >/dev/null 2>&1 || git config --global user.email "vietha.icloud@gmail.com"

[ -d .git ] || git init -b main
git add -A

echo
echo "==> staged for commit:"
git diff --cached --name-only | sed 's/^/    /'
echo

if git diff --cached --name-only | grep -qiE '\.(jpg|jpeg|png|mp4|mov|keras|h5)$|^data/|^videos|^\.venv/'; then
    echo "!! Data or model files are staged. They belong on Hugging Face, not GitHub."
    echo "   Check .gitignore, then run:  git reset"
    exit 1
fi

git commit -m "Room recognition: dataset pipeline, baseline and training notebooks" \
    || echo "(nothing new to commit)"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
        git remote get-url origin >/dev/null 2>&1 || \
            git remote add origin "https://github.com/viethaa/$REPO_NAME.git"
        git push -u origin main
    else
        gh repo create "$REPO_NAME" $VISIBILITY --source=. --remote=origin --push
    fi
    echo
    echo "==> done: $(gh repo view "$REPO_NAME" --json url -q .url)"
else
    echo "gh CLI not installed or not logged in."
    echo
    echo "Install it:   brew install gh && gh auth login"
    echo "then re-run:  bash push_to_github.sh"
    echo
    echo "Or create the repo at https://github.com/new (name it '$REPO_NAME',"
    echo "do NOT tick 'Add a README'), then run:"
    echo
    echo "    git remote add origin https://github.com/viethaa/$REPO_NAME.git"
    echo "    git push -u origin main"
fi
