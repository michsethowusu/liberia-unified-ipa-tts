set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
CONFIG=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts/training/configs/finetune_guided_tts.yaml

cd "$AFRICA_TTS"
"$PY" -m training.train "$CONFIG"
