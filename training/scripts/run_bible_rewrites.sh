set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
DATA=/mnt/volume_d2wey28/data/unified_ipa_tts
OUT=/mnt/volume_d2wey28/data/unified_guided_tts/rewrites
KEYS=/mnt/volume_d2wey28/.secrets/gemini_keys.txt

mkdir -p "$OUT"
N=$(grep -cve '^[[:space:]]*$' "$KEYS")
CACHE=("$OUT"/*.jsonl)

for i in $(seq 0 $((N - 1))); do
  "$PY" "$REPO/training/scripts/rewrite_standard_english.py" \
    --manifests "$DATA/manifests/train.jsonl" "$DATA/manifests/valid.jsonl" \
    --out "$OUT/bible_shard_${i}.jsonl" \
    --cache "${CACHE[@]}" \
    --key-file "$KEYS" --key-index "$i" --key-count "$N" \
    --backend gemini-live --sources kolokwa_bible --workers 3 \
    > "$OUT/bible_shard_${i}.log" 2>&1 &
done
wait
echo "all $N bible rewrite shards finished"
