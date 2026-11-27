# Liberia Unified TTS

Training and inference code for a unified Pocket TTS model covering Kolokwa/Liberian English text, Loma IPA, and Kpelle IPA.

## Inference pipeline

- No T5 model is used at training or inference time.
- `kolokwa` accepts Liberian English transcript text or standard English text directly -> TTS model.
- `liberian_english` is folded into the `kolokwa` training bucket.
- `loma`: text -> `africa-g2p` code `lom` -> TTS model.
- `kpelle`: text -> `africa-g2p` code `xpe` -> TTS model.
- Language labels are metadata, not conditioning tokens.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Guided finetune data

Liberian English rows are relabeled `kolokwa`, subsampled to match the Kolokwa hours, and duplicated: one row keeps the Liberian English transcript, and one row uses a Gemini-rewritten standard English transcript. Loma and Kpelle keep `africa-g2p` IPA. A fresh tokenizer is trained over this mixed target field, the text embedding is reset, and all other weights are seeded from the previous best EMA checkpoint.

```bash
pip install "afrispeech-synth[gemini] @ git+https://github.com/michsethowusu/gemini-datagen.git"

KEYS=/mnt/volume_d2wey28/.secrets/gemini_keys.txt
N=$(grep -cve '^[[:space:]]*$' "$KEYS")
for i in $(seq 0 $((N - 1))); do
  python training/scripts/rewrite_standard_english.py \
    --manifests /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl \
    --out /mnt/volume_d2wey28/data/unified_guided_tts/rewrites/shard_${i}.jsonl \
    --key-file "$KEYS" --key-index "$i" --key-count "$N" \
    --backend gemini-live --balance-le-to kolokwa --workers 3 &
done
wait

python training/scripts/build_guided_manifests.py \
  --train-in /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl \
  --valid-in /mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl \
  --rewrite-cache /mnt/volume_d2wey28/data/unified_guided_tts/rewrites/shard_*.jsonl \
  --balance-le-to kolokwa \
  --out /mnt/volume_d2wey28/data/unified_guided_tts/manifests

bash training/scripts/build_guided_tokenizer.sh
bash training/scripts/seed_guided.sh
bash training/scripts/train_guided.sh
```

## Single sample

```bash
python -m unified_tts.cli \
  --lang kolokwa \
  --text "Wetin we coming learn about our brother and sister them" \
  --voice /path/to/voice.wav \
  --out out/kolokwa.wav
```

Use `--cfg 1.0` for a distilled model.

## Regenerate sample page

```bash
python -m unified_tts.generate_samples \
  --manifest /mnt/volume_d2wey28/data/unified_guided_tts/manifests/valid.jsonl \
  --out /mnt/volume_d2wey28/data/unified_ipa_tts/samples/teacher_best_inference
```

This writes `audio/*.wav`, `meta.json`, and `index.html`.

## Out-of-domain samples

Pass `--prompts` with a JSONL file of `{ "language": "kpelle", "text": "..." }` rows. Voice prompts still come from the manifest, but the spoken text comes from the prompt file.

```bash
python -m unified_tts.generate_samples \
  --prompts prompts.jsonl \
  --out /mnt/volume_d2wey28/data/unified_ipa_tts/samples/out_of_domain
```

Useful text sources:

- Kpelle: `https://huggingface.co/datasets/IARG-UF/English-Kpelle-Corpus`
- Kpelle UDHR: `http://efele.net/udhr/d/udhr_gkp.html` (Guinean Kpelle; verify orthography against Liberian Kpelle)
- Loma and Kpelle: `https://github.com/AfriSpeech/africa-corpus-builder` with language codes `lom` and `xpe`

## Training

Training configs and scripts are in `training/`.

```bash
bash training/scripts/train_teacher.sh
bash training/scripts/train_distill.sh
```

The teacher config is `training/configs/finetune_unified_ipa.yaml`; the six-layer distillation config is `training/configs/depth_distill_unified_ipa.yaml`.

The training scripts expect the H200 paths used in this project:

- recipe: `/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts`
- data: `/mnt/volume_d2wey28/data/unified_ipa_tts`
- Africa TTS: `/mnt/volume_d2wey28/projects/africa-tts`
- Pocket TTS env: `/mnt/volume_d2wey28/projects/pocket-tts-twi/.venv/bin/python`

## Models

- TTS checkpoint used for sample generation: `runs/unified_ipa_teacher/best_model.safetensors`
- T5: `https://huggingface.co/AfriSpeech/t5-small-english-kolokwa`
- Kolokwa G2P: `https://huggingface.co/michsethowusu/kolokwa-g2p-bible`
- Liberian English G2P: `https://huggingface.co/michsethowusu/kolokwa-g2p-jw`
- Local-language G2P: `https://github.com/AfriSpeech/africa-g2p`
