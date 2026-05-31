#!/usr/bin/env bash
# Symlink the weaviate_engram plugin directory into $HERMES_HOME/plugins/
# so Hermes Agent discovers it on next start.
#
# Usage: ./install.sh

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGINS_DIR="$HERMES_HOME/plugins"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$REPO_ROOT/weaviate_engram"
TARGET="$PLUGINS_DIR/weaviate_engram"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "✗ Plugin source directory not found at $SOURCE_DIR" >&2
    exit 1
fi

mkdir -p "$PLUGINS_DIR"

if [ -L "$TARGET" ]; then
    existing="$(readlink "$TARGET")"
    if [ "$existing" = "$SOURCE_DIR" ]; then
        echo "✓ Already installed: $TARGET → $SOURCE_DIR"
        exit 0
    fi
    echo "→ Replacing existing symlink: $TARGET (was → $existing)"
    rm "$TARGET"
elif [ -e "$TARGET" ]; then
    echo "✗ $TARGET exists and is not a symlink. Move or remove it first." >&2
    exit 1
fi

ln -s "$SOURCE_DIR" "$TARGET"
echo "✓ Installed: $TARGET → $SOURCE_DIR"

echo ""
echo "→ Next: activate the provider"
echo "    hermes memory setup    # pick weaviate_engram"
echo ""
echo "  The wizard installs the Engram SDK and prompts for your API key."
