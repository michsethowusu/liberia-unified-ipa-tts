set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
RECIPE=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts
DATA=/mnt/volume_d2wey28/data/unified_ipa_tts
WORKERS=${WORKERS:-8}

EXTRA=()
if [[ -n "${MAX_SHARDS_PER_SOURCE:-}" ]]; then
  EXTRA+=(--max-shards-per-source "$MAX_SHARDS_PER_SOURCE")
fi
if [[ -n "${MAX_ROWS_PER_SHARD:-}" ]]; then
  EXTRA+=(--max-rows-per-shard "$MAX_ROWS_PER_SHARD")
fi
if [[ -n "${TARGET_ENGLISH_HOURS:-}" ]]; then
  EXTRA+=(--target-english-hours "$TARGET_ENGLISH_HOURS")
fi

"$PY" "$RECIPE/scripts/prepare_unified.py" --out "$DATA" --workers "$WORKERS" "${EXTRA[@]}"

cd "$AFRICA_TTS"
"$PY" recipes/africa/data/align_ctc.py "$DATA/manifests/raw_train.jsonl" "$DATA/manifests/aligned_train.jsonl" --batch-size 64 --threads 16
"$PY" recipes/africa/data/align_ctc.py "$DATA/manifests/raw_valid.jsonl" "$DATA/manifests/aligned_valid.jsonl" --batch-size 64 --threads 16

"$PY" "$RECIPE/scripts/finalize_manifest.py" "$DATA/manifests/aligned_train.jsonl" "$DATA/manifests/train.jsonl"
"$PY" "$RECIPE/scripts/finalize_manifest.py" "$DATA/manifests/aligned_valid.jsonl" "$DATA/manifests/valid.jsonl"

bash "$RECIPE/scripts/build_tokenizer.sh"

"$PY" "$RECIPE/scripts/check_manifests.py" \
  "$DATA/manifests/train.jsonl" \
  "$DATA/manifests/valid.jsonl" \
  --tokenizer "$RECIPE/tokenizer/ipa_sp4000.json"
