#!/usr/bin/env bash
# Clone TrackNetV3 and fetch its pretrained weights (Stage 3 shuttle tracker).
# Run from the repo root. On Kaggle, run this in a notebook cell with leading '!'.
set -e

DEST="${1:-third_party/TrackNetV3}"
mkdir -p "$(dirname "$DEST")"

if [ ! -d "$DEST" ]; then
  echo "[setup] cloning TrackNetV3 -> $DEST"
  git clone https://github.com/qaz812345/TrackNetV3 "$DEST"
else
  echo "[setup] $DEST already exists, skipping clone"
fi

pip -q install gdown || true

# Pretrained weights: a single zip on Google Drive -> unzips to ckpts/.
# (id from the TrackNetV3 README; update if upstream changes the link.)
GD_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
CKPT_DIR="$DEST/ckpts"
ZIP="$DEST/TrackNetV3_ckpts.zip"

# gdrive downloads flake (truncated zips -> 0-byte .pt files). Treat a weight as
# valid only if it's a real, multi-KB file, and retry the whole fetch a few times.
weights_ok() {
  for w in TrackNet_best.pt InpaintNet_best.pt; do
    [ -s "$CKPT_DIR/$w" ] || return 1
    sz=$(stat -c%s "$CKPT_DIR/$w" 2>/dev/null || echo 0)
    [ "$sz" -ge 100000 ] || return 1   # >=100KB; an empty/partial file fails
  done
  return 0
}

if ! weights_ok; then
  for attempt in 1 2 3 4; do
    echo "[setup] downloading TrackNetV3 weights zip (attempt $attempt)"
    rm -f "$ZIP"; rm -rf "$CKPT_DIR"
    gdown "$GD_ID" -O "$ZIP" || { echo "[setup] gdown failed"; sleep 5; continue; }
    if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
      echo "[setup] zip integrity check FAILED, retrying"; sleep 5; continue
    fi
    ( cd "$DEST" && unzip -o TrackNetV3_ckpts.zip )
    if weights_ok; then echo "[setup] weights verified"; break; fi
    echo "[setup] weights still invalid after unzip, retrying"; sleep 5
  done
fi

if weights_ok; then
  echo "[setup] ready: $CKPT_DIR/{TrackNet_best.pt,InpaintNet_best.pt}"
  echo "        run: python src/shuttle_tracker.py --source match.mp4 \\"
  echo "             --repo $DEST --tracknet-ckpt ckpts/TrackNet_best.pt \\"
  echo "             --inpaint-ckpt ckpts/InpaintNet_best.pt --out shuttle.json"
else
  echo "[setup] WARN: weights not found. Download manually from the link in"
  echo "        $DEST/README.md, unzip into $CKPT_DIR/, then run shuttle_tracker.py"
fi
