set -euo pipefail

REMOTE=${REMOTE:-h200:/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer/}
LOCAL=${LOCAL:-/home/owusus/Documents/GitHub/liberia-unified-ipa-tts/samples/cer}

mkdir -p "$LOCAL"
rsync -av \
  --include='*/' \
  --include='*.wav' \
  --include='index.html' \
  --include='meta.json' \
  --include='summary.json' \
  --include='*_summary.json' \
  --include='*_details.jsonl' \
  --exclude='*' \
  "$REMOTE" "$LOCAL/"
