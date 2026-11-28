set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
RUN_ROOT=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs
EXPORT_ROOT=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts/model_exports

wait_for_finished() {
  local log="$1"
  while [[ ! -f "$log" ]] || ! grep -q "finished" "$log"; do
    sleep 60
  done
}

mkdir -p "$EXPORT_ROOT"

wait_for_finished "$RUN_ROOT/distill_100m.log"
"$PY" "$REPO/training/scripts/export_model_hf.py" \
  --run-dir "$RUN_ROOT/guided_distill_100m" \
  --repo-id AfriSpeech/liberian-unified-ipa-tts \
  --name "UnifiedIPA Pocket TTS 100M" \
  --export-dir "$EXPORT_ROOT/100m"
CFG=1.0 BASE_CONFIG="$EXPORT_ROOT/100m/config.yaml" \
  bash "$REPO/training/scripts/eval_cer.sh" ood_100m "$RUN_ROOT/guided_distill_100m/best_model.safetensors"

wait_for_finished "$RUN_ROOT/distill_30m.log"
"$PY" "$REPO/training/scripts/export_model_hf.py" \
  --run-dir "$RUN_ROOT/guided_distill_30m" \
  --repo-id AfriSpeech/liberian-unified-ipa-tts-30m \
  --name "UnifiedIPA Pocket TTS 30M" \
  --export-dir "$EXPORT_ROOT/30m"
CFG=1.0 BASE_CONFIG="$EXPORT_ROOT/30m/config.yaml" \
  bash "$REPO/training/scripts/eval_cer.sh" ood_30m "$RUN_ROOT/guided_distill_30m/best_model.safetensors"
