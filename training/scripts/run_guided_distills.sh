set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
RUN_ROOT=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs

cd "$AFRICA_TTS"

for SIZE in 100m 30m; do
  CONFIG="$REPO/training/configs/distill_guided_${SIZE}.yaml"
  LOG="$RUN_ROOT/distill_${SIZE}.log"
  echo "$(date -Is) starting $SIZE" > "$LOG"
  "$PY" -m training.train "$CONFIG" >> "$LOG" 2>&1
  echo "$(date -Is) finished $SIZE" >> "$LOG"
done
