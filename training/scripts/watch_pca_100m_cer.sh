set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
RUN=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs/guided_distill_100m_pca
CKPT="$RUN/checkpoint_00010000.pt"
EXPORT=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts/model_exports/100m_pca_10k
NAME=ood_100m_pca_10k
LOG="$RUN/cer_watch_10k.log"

{
  while [[ ! -f "$CKPT" ]]; do
    sleep 120
  done
  echo "$(date -Is) exporting $RUN/best_model.safetensors"
  "$PY" "$REPO/training/scripts/export_model_hf.py" \
    --run-dir "$RUN" \
    --checkpoint "$RUN/best_model.safetensors" \
    --repo-id AfriSpeech/liberian-unified-ipa-tts \
    --name "Liberian Unified IPA TTS 100M PCA" \
    --export-dir "$EXPORT"
  echo "$(date -Is) running CER"
  CFG=1.0 EOS_THRESHOLD=-1.0 BASE_CONFIG="$EXPORT/config.yaml" \
    bash "$REPO/training/scripts/eval_cer.sh" "$NAME" "$EXPORT/model.safetensors"
  echo "$(date -Is) finished CER"
} > "$LOG" 2>&1
