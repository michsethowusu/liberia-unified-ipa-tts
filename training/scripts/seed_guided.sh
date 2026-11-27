set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
BEST=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts/runs/unified_ipa_teacher/best_checkpoint.pt
RUN=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs/guided_teacher

cd "$AFRICA_TTS"
"$PY" "$REPO/training/scripts/seed_guided_checkpoint.py" \
  --config "$REPO/training/configs/finetune_guided_tts.yaml" \
  --best-checkpoint "$BEST" \
  --out "$RUN/checkpoint_00000000.pt"
