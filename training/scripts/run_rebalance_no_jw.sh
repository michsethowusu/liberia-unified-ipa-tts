set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
RECIPE=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts
DATA=/mnt/volume_d2wey28/data/unified_ipa_tts

"$PY" "$RECIPE/scripts/rebalance_no_jw.py"

bash "$RECIPE/scripts/build_tokenizer.sh"

"$PY" "$RECIPE/scripts/check_manifests.py" \
  "$DATA/manifests/train.jsonl" \
  "$DATA/manifests/valid.jsonl" \
  --tokenizer "$RECIPE/tokenizer/ipa_sp4000.json"
