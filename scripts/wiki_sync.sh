#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/docs/wiki"
WIKI_URL="${WIKI_URL:-https://github.com/mohamedsaid2710/Distractor_software-.wiki.git}"
WIKI_DIR="${WIKI_DIR:-/tmp/Distractor_software-wiki}"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: Source docs directory not found: $SRC_DIR" >&2
  exit 1
fi

if [ ! -d "$WIKI_DIR/.git" ]; then
  echo "Cloning wiki repo into: $WIKI_DIR"
  git clone "$WIKI_URL" "$WIKI_DIR"
else
  echo "Updating existing wiki clone in: $WIKI_DIR"
  git -C "$WIKI_DIR" pull --ff-only
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete --exclude '.git' "$SRC_DIR"/ "$WIKI_DIR"/
else
  echo "rsync not found; using cp fallback"
  find "$WIKI_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
  cp -a "$SRC_DIR"/. "$WIKI_DIR"/
fi

echo ""
echo "Synced docs/wiki -> $WIKI_DIR"
echo ""
git -C "$WIKI_DIR" status --short

echo ""
echo "Next steps:"
echo "  cd $WIKI_DIR"
echo "  git add ."
echo "  git commit -m \"Update wiki docs\""
echo "  git push"
