set -euo pipefail

SUMMARY=${SUMMARY:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer/ood_all_short_best/ood_all_short_best_summary.json}
SYNC=${SYNC:-/home/owusus/Documents/GitHub/liberia-unified-ipa-tts/training/scripts/sync_cer_samples.sh}

until ssh h200 "test -f '$SUMMARY'"; do
  sleep 30
done

"$SYNC"
