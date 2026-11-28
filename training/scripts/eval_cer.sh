set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
OUT_ROOT=${OUT_ROOT:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer}
NAME=${1:-ckpt}
CKPT=${2:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs/guided_teacher/best_model.safetensors}
MANIFEST=${MANIFEST:-/mnt/volume_d2wey28/data/unified_guided_tts/manifests/valid.jsonl}
NATURAL_MANIFEST=${NATURAL_MANIFEST:-$MANIFEST}
PROMPTS=${PROMPTS:-/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts/training/data/cer_ood_all.jsonl}
LANGS=${LANGS:-kolokwa,liberian_english,loma,kpelle}
TOKENIZER=${TOKENIZER:-/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/tokenizer/ipa_sp4000.json}
BASE_CONFIG=${BASE_CONFIG:-/mnt/volume_d2wey28/models/africa-tts-base/config.yaml}
SHERPA_DIR=${SHERPA_DIR:-/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05}
PER_LANG=${PER_LANG:-16}
NATURAL_PER_LANG=${NATURAL_PER_LANG:-50}
SEED=${SEED:-2026}
NUM_THREADS=${NUM_THREADS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}
CFG=${CFG:-1.0}
EOS_THRESHOLD=${EOS_THRESHOLD:-}

EOS_ARGS=()
if [[ -n "$EOS_THRESHOLD" ]]; then
  EOS_ARGS=(--eos-threshold "$EOS_THRESHOLD")
fi

mkdir -p "$OUT_ROOT" "$OUT_ROOT/$NAME"

if [[ ! -f "$OUT_ROOT/natural_summary.json" || ${REFRESH_NATURAL:-0} == 1 ]]; then
  "$PY" "$REPO/training/scripts/eval_cer.py" \
    --manifest "$NATURAL_MANIFEST" \
    --per-lang "$NATURAL_PER_LANG" \
    --seed "$SEED" \
    --model-dir "$SHERPA_DIR" \
    --num-threads "$NUM_THREADS" \
    --batch-size "$BATCH_SIZE" \
    --name natural \
    --out-dir "$OUT_ROOT"
fi

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m unified_tts.generate_samples \
  --manifest "$MANIFEST" \
  --out "$OUT_ROOT/$NAME/samples" \
  --checkpoint "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --base-config "$BASE_CONFIG" \
  --per-lang "$PER_LANG" \
  --langs "$LANGS" \
  --prompts "$PROMPTS" \
  --cfg "$CFG" \
  "${EOS_ARGS[@]}" \
  --seed "$SEED"

"$PY" "$REPO/training/scripts/eval_cer.py" \
  --generated-meta "$OUT_ROOT/$NAME/samples/meta.json" \
  --per-lang "$PER_LANG" \
  --seed "$SEED" \
  --model-dir "$SHERPA_DIR" \
  --num-threads "$NUM_THREADS" \
  --batch-size "$BATCH_SIZE" \
  --name "$NAME" \
  --out-dir "$OUT_ROOT/$NAME"
