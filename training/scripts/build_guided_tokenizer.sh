set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
RECIPE=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts
DATA=/mnt/volume_d2wey28/data/unified_guided_tts

mkdir -p "$RECIPE/tokenizer"

cd "$AFRICA_TTS"
"$PY" recipes/africa/data/build_ipa_tokenizer.py "$RECIPE/tokenizer" \
  --train "$DATA/manifests/train.jsonl" \
  --valid "$DATA/manifests/valid.jsonl" \
  --vocab 4000 \
  --lines 2000000
"$PY" -m training.scripts.convert_tokenizer "$RECIPE/tokenizer/ipa_sp4000.model" \
  --out "$RECIPE/tokenizer/ipa_sp4000.json" \
  --texts "$DATA/manifests/valid.jsonl"
