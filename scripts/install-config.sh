#!/usr/bin/env bash
# Copia examples/config.yaml para ~/.config/gptcli/config.yaml
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$SRC_DIR/examples/config.yaml"
DEST_DIR="$HOME/.config/gptcli"
DEST="$DEST_DIR/config.yaml"

if [ ! -f "$SRC" ]; then
  echo "arquivo de exemplo não encontrado: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp -v "$SRC" "$DEST"
chmod 600 "$DEST"
echo "exemplo instalado em: $DEST"
