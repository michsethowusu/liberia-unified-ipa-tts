# Liberia Unified TTS

Training and inference code for a unified Pocket TTS model covering Kolokwa/Liberian English text, Loma IPA, and Kpelle IPA.

## Inference pipeline

- No T5 model is used at training or inference time.
- `kolokwa`: text -> `<lang:kolokwa>` prefix -> TTS model.
- `liberian_english`: text -> `<lang:liberian_english>` prefix -> TTS model.
- `loma`: text -> `africa-g2p` code `lom` -> `<lang:loma>` prefix -> TTS model.
- `kpelle`: text -> `africa-g2p` code `xpe` -> `<lang:kpelle>` prefix -> TTS model.
- Language labels are conditioning tokens stored as `condition_prefix` in the manifest and prepended during inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Guided finetune data

Sources `liberian_english` and `kolokwa_movies` from `michsethowusu/liberian-english-speech` are dropped. `kolokwa_jw_formal` rows are rewritten, then split by duration so roughly half of the rows/hours are trained with `<lang:liberian_english>` and half with `<lang:kolokwa>`. For both halves, the original JW Kolokwa-style wording and the Gemini-rewritten standard-English wording are paired on the same audio. `kolokwa_bible` is also rewritten, and each Bible clip is paired under `<lang:kolokwa>` with its original wording and the rewritten standard-English wording. Loma and Kpelle keep `africa-g2p` IPA.
 Four reserved tokens (`<lang:kolokwa>`, `<lang:liberian_english>`, `<lang:loma>`, `<lang:kpelle>`) remain active and are stored as `condition_prefix` in each manifest row. There is no standalone standard-English tag: rewritten JW text and HiFiTTS-2 replay both use `<lang:liberian_english>` so clean English pronunciation reinforces Liberian English. The H200 loader prepends the token after word-cuts, and inference passes the same token. A fresh tokenizer is trained with those tokens reserved, the text embedding is reset, and the rest is finetuned from `AfriSpeech/africa-tts-base` rather than from scratch.

```bash
pip install "afrispeech-synth[gemini] @ git+https://github.com/michsethowusu/gemini-datagen.git"

KEYS=/mnt/volume_d2wey28/.secrets/gemini_keys.txt
N=$(grep -cve '^[[:space:]]*$' "$KEYS")
for i in $(seq 0 $((N - 1))); do
  python training/scripts/rewrite_standard_english.py \
    --manifests /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl \
    --out /mnt/volume_d2wey28/data/unified_guided_tts/rewrites/shard_${i}.jsonl \
    --key-file "$KEYS" --key-index "$i" --key-count "$N" \
    --backend gemini-live --sources kolokwa_jw_formal --workers 3 &
done
wait

python training/scripts/build_guided_manifests.py \
  --train-in /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl \
  --valid-in /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl \
  --rewrite-cache /mnt/volume_d2wey28/data/unified_guided_tts/rewrites/shard_*.jsonl \
  --hifi-train /mnt/volume_d2wey28/data/africa-tts/aligned_ctc/hifitts2_train.jsonl \
  --hifi-valid /mnt/volume_d2wey28/data/africa-tts/aligned_ctc/hifitts2_valid.jsonl \
  --hifi-train-hours 20 --hifi-valid-hours 0.5 \
  --out /mnt/volume_d2wey28/data/unified_guided_tts/manifests

bash training/scripts/build_guided_tokenizer.sh
bash training/scripts/train_guided.sh
```

## Single sample

```bash
python -m unified_tts.cli \
  --lang kolokwa \
  --text "Wetin we coming learn about our brother and sister them" \
  --out out/kolokwa.wav
```

The CLI defaults to `AfriSpeech/liberian-unified-ipa-tts` with `cfg=1.0` and a bundled per-language voice prompt. Pass `--voice /path/to/voice.wav` to override the voice, or `--cfg 2.0` with `--checkpoint hf://AfriSpeech/liberian-unified-ipa-tts-teacher/model.safetensors` for the teacher.

## Regenerate sample page

```bash
python -m unified_tts.generate_samples \
  --manifest /mnt/volume_d2wey28/data/unified_guided_tts/manifests/valid.jsonl \
  --out /mnt/volume_d2wey28/data/unified_ipa_tts/samples/teacher_best_inference
```

This writes `audio/*.wav`, `meta.json`, and `index.html`.

## Out-of-domain samples

Pass `--prompts` with a JSONL file of `{ "language": "kpelle", "text": "..." }` rows. Voice prompts come from `unified_tts/voice_prompts/`, but the spoken text comes from the prompt file.

```bash
python -m unified_tts.generate_samples \
  --prompts prompts.jsonl \
  --out /mnt/volume_d2wey28/data/unified_ipa_tts/samples/out_of_domain
```

Useful text sources:

- Kpelle: `https://huggingface.co/datasets/IARG-UF/English-Kpelle-Corpus`
- Kpelle UDHR: `http://efele.net/udhr/d/udhr_gkp.html` (Guinean Kpelle; verify orthography against Liberian Kpelle)
- Loma and Kpelle: `https://github.com/AfriSpeech/africa-corpus-builder` with language codes `lom` and `xpe`

## CER during training

CER is computed with the CPU-only `sherpa-onnx` Omnilingual ASR CTC 300M v2 model, so it can run beside GPU training. It reports macro and micro CER overall, per language, and per conditioning prefix for both natural evaluation audio and generated TTS audio. The ASR model is not language-conditioned in the sherpa CTC API and is imperfect for these varieties, so use the natural-audio baseline as the floor and compare TTS checkpoints relative to it.

```bash
bash training/scripts/eval_cer.sh natural_baseline /path/to/checkpoint.safetensors
tmux new -s cer-watch -d 'bash training/scripts/watch_eval_cer.sh'
```

Results go to `/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer/`. Override `SHERPA_DIR` to use the int8 model; override `PER_LANG`, `NATURAL_PER_LANG`, `POLL_SECS`, `RUN_DIR`, or `OUT_ROOT` as needed.

## Training

Training configs and scripts are in `training/`.

```bash
bash training/scripts/train_teacher.sh
bash training/scripts/train_distill.sh
```

The guided teacher config is `training/configs/finetune_guided_tts.yaml`. The canonical CFG-baked student config is `training/configs/distill_guided_samewidth_110m.yaml`: a same-width 6-layer student that copies the teacher text table, flow head, EOS head, and latent stats directly. Width-reduced PCA configs remain in `training/configs/` as experiments and are not the recommended release path.

The training scripts expect the H200 paths used in this project:

- recipe: `/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts`
- data: `/mnt/volume_d2wey28/data/unified_ipa_tts`
- Africa TTS: `/mnt/volume_d2wey28/projects/africa-tts`
- Pocket TTS env: `/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python`

## Models

- Default distilled TTS: `hf://AfriSpeech/liberian-unified-ipa-tts/model.safetensors` (`cfg=1.0`)
- Teacher TTS: `hf://AfriSpeech/liberian-unified-ipa-tts-teacher/model.safetensors` (`cfg=2.0`)
- Smaller distilled TTS: `hf://AfriSpeech/liberian-unified-ipa-tts-30m/model.safetensors` (`cfg=1.0`)
- T5: `https://huggingface.co/AfriSpeech/t5-small-english-kolokwa`
- Kolokwa G2P: `https://huggingface.co/michsethowusu/kolokwa-g2p-bible`
- Liberian English G2P: `https://huggingface.co/michsethowusu/kolokwa-g2p-jw`
- Local-language G2P: `https://github.com/AfriSpeech/africa-g2p`
