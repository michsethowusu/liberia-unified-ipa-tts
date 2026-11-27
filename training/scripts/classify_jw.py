import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = "AfriSpeech/kolokwa-classifier"
TARGET_SR = 16000


def load_audio(row):
    info = sf.info(row["path"])
    sr = info.samplerate
    start = int(round(float(row.get("start", 0.0)) * sr))
    frames = int(round(float(row.get("duration", 0.0)) * sr))
    wav, sr = sf.read(row["path"], start=start, frames=frames, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        gcd = math.gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // gcd, sr // gcd).astype(np.float32)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForAudioClassification.from_pretrained(MODEL, dtype=dtype).to(args.device).eval()

    rows = []
    for path in args.manifests:
        for line in Path(path).open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("source") == "kolokwa_jw":
                rows.append(row)
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            done.add(json.loads(line).get("id"))
    rows = [r for r in rows if r.get("id") not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows):,} JW rows to score, {len(done):,} already scored", flush=True)

    ok = failed = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            wavs = [None] * len(batch)
            for i, row in enumerate(batch):
                try:
                    wavs[i] = load_audio(row)
                except Exception:
                    wavs[i] = None
            valid = [(i, w) for i, w in enumerate(wavs) if w is not None]
            if valid:
                inputs = feature_extractor(
                    [w for _, w in valid],
                    sampling_rate=TARGET_SR,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = {
                    key: value.to(args.device, dtype=dtype) if value.is_floating_point() else value.to(args.device)
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits.float(), dim=-1)
                scores = probs[:, 1].cpu().tolist()
            else:
                scores = []
            score_iter = iter(scores)
            for i, row in enumerate(batch):
                out = dict(row)
                if wavs[i] is None:
                    out.update({"p_kolokwa": None, "p_formal": None, "classifier_label": "error"})
                    failed += 1
                else:
                    p = next(score_iter)
                    out.update({
                        "p_kolokwa": round(float(p), 6),
                        "p_formal": round(float(1.0 - p), 6),
                        "classifier_label": "pure_kolokwa" if p >= 0.5 else "formal_liberian_english",
                    })
                    ok += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            if (start // args.batch_size) % 50 == 0:
                print(f"{start + len(batch):,}/{len(rows):,} | ok {ok:,} failed {failed:,}", flush=True)
    print(f"done: {ok:,} scored, {failed:,} failed", flush=True)


if __name__ == "__main__":
    main()
