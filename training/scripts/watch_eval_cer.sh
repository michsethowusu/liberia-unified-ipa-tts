set -euo pipefail

REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
RUN_DIR=${RUN_DIR:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs/guided_teacher}
OUT_ROOT=${OUT_ROOT:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer}
POLL_SECS=${POLL_SECS:-900}
SEEN="$OUT_ROOT/seen_checkpoints.txt"

mkdir -p "$OUT_ROOT"
touch "$SEEN"

while true; do
  for CKPT in "$RUN_DIR"/checkpoint_*.pt "$RUN_DIR"/best_model.safetensors; do
    [[ -f "$CKPT" ]] || continue
    MTIME=$(stat -c %Y "$CKPT")
    BASE=$(basename "$CKPT")
    KEY="$BASE $MTIME"
    if grep -Fxq "$KEY" "$SEEN"; then
      continue
    fi
    NAME="${BASE%.pt}"
    NAME="${NAME%.safetensors}_$MTIME"
    echo "$(date -Is) evaluating $CKPT" >> "$OUT_ROOT/watcher.log"
    if "$REPO/training/scripts/eval_cer.sh" "$NAME" "$CKPT" >> "$OUT_ROOT/watcher.log" 2>&1; then
      echo "$KEY" >> "$SEEN"
    else
      echo "$KEY FAILED" >> "$SEEN"
    fi
  done
  sleep "$POLL_SECS"
done
