import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

LANGS = ["kolokwa", "liberian_english", "loma", "kpelle"]


def rms(audio):
    if audio.size == 0:
        return 1e-6
    return float(np.sqrt(np.mean(audio**2)))


def load_clip(row):
    path = Path(str(row.get("path", "")))
    if not path.exists():
        return None
    start = float(row.get("start", 0) or 0)
    duration = float(row.get("duration", 0) or 0)
    info = sf.info(str(path))
    if start < 0.1 or start + duration > info.duration - 0.1:
        return None
    first = int(start * info.samplerate)
    last = int((start + duration) * info.samplerate)
    audio, sample_rate = sf.read(
        str(path), start=first, frames=last - first, dtype="float32", always_2d=True
    )
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sample_rate


def score_clip(audio, sample_rate):
    edge = int(0.12 * sample_rate)
    left = rms(audio[:edge])
    right = rms(audio[-edge:])
    middle = rms(audio[edge:-edge])
    return max(left, right) / max(middle, 1e-6), left, right, middle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_guided_tts/manifests/valid.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    candidates = {lang: [] for lang in LANGS}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        lang = str(row.get("language", ""))
        if lang not in LANGS:
            continue
        text = str(row.get("transcript_orig") or row.get("transcript") or "").strip()
        duration = float(row.get("duration", 0) or 0)
        if not text or not 2.0 <= duration <= 4.5:
            continue
        clip = load_clip(row)
        if clip is None:
            continue
        audio, sample_rate = clip
        edge_ratio, left, right, middle = score_clip(audio, sample_rate)
        if edge_ratio > 0.35:
            continue
        candidates[lang].append(
            {
                "language": lang,
                "id": row.get("id"),
                "source": row.get("source"),
                "text": text,
                "duration": duration,
                "start": float(row.get("start", 0) or 0),
                "edge_ratio": edge_ratio,
                "left": left,
                "right": right,
                "middle": middle,
                "path": str(row.get("path")),
                "file": f"{lang}.wav",
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang in LANGS:
        ranked = sorted(candidates[lang], key=lambda item: item["edge_ratio"])[: args.top]
        print(lang, len(candidates[lang]), "candidates")
        for rank, item in enumerate(ranked, 1):
            print(" ", rank, round(item["edge_ratio"], 4), item["id"], item["text"][:100])
            if rank == 1:
                audio, sample_rate = sf.read(item["path"], dtype="float32", always_2d=True)
                first = int(item["start"] * sample_rate)
                last = int((item["start"] + item["duration"]) * sample_rate)
                clip = audio[first:last]
                out_path = args.out / f"{lang}.wav"
                sf.write(out_path, clip, sample_rate)
                item["wav"] = str(out_path)
                manifest.append(item)
    (args.out / "voice_prompts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
