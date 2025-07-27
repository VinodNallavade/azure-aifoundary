#!/bin/bash
# Usage: ./push_to_github.sh "commit message"

FILE="index.html"
COMMIT_MSG=${1:-"Auto-commit: updated index.html"}

git add "$FILE"
git commit -m "$COMMIT_MSG"
git push origin main  # or your branch name
if [ $? -eq 0 ]; then
    echo "Changes pushed to GitHub successfully."
else
    echo "Failed to push changes to GitHub."
fi