set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
CONFIG=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts/configs/finetune_unified_ipa.yaml

cd "$AFRICA_TTS"
"$PY" -m training.train "$CONFIG"
