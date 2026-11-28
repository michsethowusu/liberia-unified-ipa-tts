set -euo pipefail

PY=/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python
AFRICA_TTS=/mnt/volume_d2wey28/projects/africa-tts
REPO=/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts
RUN_ROOT=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/runs
TEACHER="$RUN_ROOT/guided_teacher"
BASE_CONFIG=/mnt/volume_d2wey28/models/africa-tts-base/config.yaml
TOKENIZER=/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/tokenizer/ipa_sp4000.json
COV="$RUN_ROOT/guided_teacher_cov.pt"
INIT_100M="$RUN_ROOT/guided_100m_init_pca.pt"
INIT_30M="$RUN_ROOT/guided_30m_init_pca.pt"
LOG="$RUN_ROOT/pca_distill_100m.log"
CONFIG="$REPO/training/configs/distill_guided_100m_pca.yaml"

cd "$AFRICA_TTS"

{
  echo "$(date -Is) starting residual covariance"
  "$PY" -m training.scripts.residual_cov "$TEACHER/args.yaml" "$TEACHER/best_checkpoint.pt" "$COV" --batches 32 --ema

  echo "$(date -Is) building 100m PCA seed"
  "$PY" -m training.scripts.shrink_checkpoint "$TEACHER/best_checkpoint.pt" --ema \
    --config "$BASE_CONFIG" \
    --override flow_lm.transformer.num_layers=10 \
    --override flow_lm.transformer.d_model=896 \
    --override flow_lm.transformer.num_heads=14 \
    --override flow_lm.transformer.hidden_scale=4 \
    --override flow_lm.lookup_table.dim=896 \
    --override flow_lm.z_dim=1024 \
    --override "flow_lm.lookup_table.tokenizer=tokenizers" \
    --override "flow_lm.lookup_table.tokenizer_path=$TOKENIZER" \
    --override flow_lm.lookup_table.n_bins=4000 \
    --cov "$COV" --out "$INIT_100M"

  echo "$(date -Is) building 30m PCA seed"
  "$PY" -m training.scripts.shrink_checkpoint "$TEACHER/best_checkpoint.pt" --ema \
    --config "$BASE_CONFIG" \
    --override flow_lm.transformer.num_layers=12 \
    --override flow_lm.transformer.d_model=640 \
    --override flow_lm.transformer.num_heads=10 \
    --override flow_lm.transformer.hidden_scale=1 \
    --override flow_lm.lookup_table.dim=640 \
    --override flow_lm.z_dim=1024 \
    --override "flow_lm.lookup_table.tokenizer=tokenizers" \
    --override "flow_lm.lookup_table.tokenizer_path=$TOKENIZER" \
    --override flow_lm.lookup_table.n_bins=4000 \
    --cov "$COV" --out "$INIT_30M"

  echo "$(date -Is) starting 100m PCA-seeded distillation"
  "$PY" -m training.train "$CONFIG"
  echo "$(date -Is) finished 100m PCA-seeded distillation"
} > "$LOG" 2>&1
