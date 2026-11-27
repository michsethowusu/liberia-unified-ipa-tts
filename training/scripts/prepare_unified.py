import argparse
import io
import json
import math
import random
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

TARGET_SR = 24000
GAP_SEC = 0.08
MIN_DUR = 1.0
MAX_DUR = 30.0
KNOWN_KOLOKWA_HOURS = 162.19
KNOWN_ENGLISH_HOURS = 829.15


@dataclass
class SourceSpec:
    name: str
    repo: str
    prefix: str
    language: str
    iso: str
    g2p: str
    ipa_mode: str
    balance: bool = False


SOURCES = [
    SourceSpec("kolokwa_bible", "michsethowusu/liberian-english-speech-bible", "data/", "kolokwa", "eng", "pxeus", "field"),
    SourceSpec("kolokwa_movies", "michsethowusu/liberian-english-speech", "data/kolokwa/", "kolokwa", "eng", "pxeus", "field"),
    SourceSpec("liberian_english", "michsethowusu/liberian-english-speech", "data/liberian_english/", "liberian_english", "eng", "eng", "field", True),
    SourceSpec("loma", "michsethowusu/loma-speech-text", "data/", "loma", "lom", "lom", "g2p"),
    SourceSpec("kpelle", "michsethowusu/kpelle-speech-text", "data/", "kpelle", "eng", "xpe", "g2p"),
]


def clean_text(value):
    if not value:
        return ""
    return " ".join(str(value).strip().split())


def has_number(text):
    return any(unicodedata.category(ch) in ("Nd", "No", "Nl") for ch in text)


def clean_ipa(value):
    if not value:
        return ""
    return " ".join(str(value).strip().split())


def list_files(spec):
    files = HfApi().list_repo_files(spec.repo, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet") and f.startswith(spec.prefix))


def process_shard(job):
    spec, file, selection, split, out_root, work_dir, max_rows = job
    path = hf_hub_download(spec.repo, file, repo_type="dataset")
    balance_fraction = 1.0
    selection_rng = None
    if selection is not None:
        balance_fraction, seed = selection
        selection_rng = random.Random(f"{seed}:{file}")
    stem = Path(file).stem
    flac_path = out_root / "audio" / spec.name / split / f"{stem}.flac"
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = work_dir / f"{spec.name}__{stem}.jsonl"

    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in ["audio", "text", "duration", "phonemes", "id", "source"] if c in names]
    table = pf.read(columns=cols)
    data = {c: table.column(c).to_pylist() for c in cols}

    need_g2p = spec.ipa_mode == "g2p"
    g2p_fn = None
    if need_g2p:
        from africa_g2p import g2p as g2p_fn

    stats = {"clips": 0, "empty": 0, "number": 0, "bad": 0, "length": 0, "missing_ipa": 0, "hours": 0.0}
    writer = None
    pos = 0.0
    count = 0
    rows = []

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for idx in range(pf.metadata.num_rows):
            if max_rows and count >= max_rows:
                break
            if selection_rng is not None and selection_rng.random() > balance_fraction:
                continue
            text = clean_text(data.get("text", [None] * pf.metadata.num_rows)[idx])
            if not text:
                stats["empty"] += 1
                continue
            if has_number(text):
                stats["number"] += 1
                continue

            ipa = ""
            if spec.ipa_mode == "field":
                ipa = clean_ipa(data.get("phonemes", [None] * pf.metadata.num_rows)[idx])
            elif g2p_fn is not None:
                ipa = clean_ipa(g2p_fn(text, lang=spec.g2p, output="ipa"))
            if not ipa:
                stats["missing_ipa"] += 1
                continue

            audio_entry = data.get("audio", [None] * pf.metadata.num_rows)[idx]
            audio_bytes = audio_entry.get("bytes") if isinstance(audio_entry, dict) else None
            if not audio_bytes:
                stats["bad"] += 1
                continue

            try:
                wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
                if sr != TARGET_SR:
                    gcd = math.gcd(sr, TARGET_SR)
                    wav = resample_poly(wav, TARGET_SR // gcd, sr // gcd, axis=0).astype(np.float32)
            except Exception:
                stats["bad"] += 1
                continue

            dur = len(wav) / TARGET_SR
            if dur < MIN_DUR or dur > MAX_DUR:
                stats["length"] += 1
                continue

            if writer is None:
                writer = sf.SoundFile(flac_path, mode="w", samplerate=TARGET_SR, channels=1, format="FLAC", subtype="PCM_16")

            writer.write(np.clip(wav, -1.0, 1.0))
            writer.write(np.zeros(int(GAP_SEC * TARGET_SR), dtype=np.float32))

            row_id = str(data.get("id", [None] * pf.metadata.num_rows)[idx] or f"{stem}_{idx:05d}")
            row = {
                "id": f"{spec.name}:{row_id}",
                "path": str(flac_path),
                "start": round(pos, 4),
                "duration": round(dur, 4),
                "transcript": text,
                "ipa": ipa,
                "language": spec.language,
                "iso": spec.iso,
                "g2p": spec.g2p,
                "source": spec.name,
                "hf_source": str(data.get("source", [""] * pf.metadata.num_rows)[idx] or ""),
                "split": split,
                "align": "pending",
                "words": None,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            pos += dur + GAP_SEC
            stats["clips"] += 1
            stats["hours"] += dur / 3600.0
            count += 1

    if writer is not None:
        writer.close()
    if not rows:
        jsonl_path.unlink(missing_ok=True)
    return {"stats": stats, "jsonl": str(jsonl_path) if rows else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--target-english-hours", type=float, default=0.0)
    ap.add_argument("--max-shards-per-source", type=int, default=0)
    ap.add_argument("--max-rows-per-shard", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work_dir = args.out / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files_by_source = {spec.name: list_files(spec) for spec in SOURCES}
    if args.max_shards_per_source:
        files_by_source = {k: v[:args.max_shards_per_source] for k, v in files_by_source.items()}

    kolokwa_seconds = KNOWN_KOLOKWA_HOURS * 3600.0
    target_seconds = args.target_english_hours * 3600.0 if args.target_english_hours else kolokwa_seconds
    balance_fraction = min(1.0, target_seconds / (KNOWN_ENGLISH_HOURS * 3600.0))
    print(f"kolokwa hours: {kolokwa_seconds / 3600:.2f} | liberian english target: {target_seconds / 3600:.2f} | fraction: {balance_fraction:.4f}", flush=True)

    jobs = []
    for spec in SOURCES:
        files = files_by_source[spec.name]
        for file_idx, file in enumerate(files):
            split = "valid" if args.val_every and file_idx % args.val_every == args.val_every - 1 else "train"
            selection = (balance_fraction, args.seed) if spec.balance else None
            jobs.append((spec, file, selection, split, args.out, work_dir, args.max_rows_per_shard))

    print(f"processing {len(jobs):,} shards", flush=True)
    summary = {}
    jsonls = {"train": [], "valid": []}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job, result in zip(jobs, executor.map(process_shard, jobs)):
            spec = job[0]
            st = result["stats"]
            key = spec.name
            summary.setdefault(key, {"clips": 0, "hours": 0.0, "empty": 0, "number": 0, "bad": 0, "length": 0, "missing_ipa": 0})
            for k in summary[key]:
                summary[key][k] += st[k]
            if result["jsonl"]:
                jsonls[job[3]].append(result["jsonl"])

    for split, paths in jsonls.items():
        out = manifest_dir / f"raw_{split}.jsonl"
        with out.open("w", encoding="utf-8") as fout:
            for path in paths:
                with Path(path).open(encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"wrote {out} from {len(paths):,} shards", flush=True)

    summary["_config"] = {
        "kolokwa_hours": round(kolokwa_seconds / 3600.0, 2),
        "liberian_english_target_hours": round(target_seconds / 3600.0, 2),
        "seed": args.seed,
        "val_every": args.val_every,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, st in summary.items():
        if name.startswith("_"):
            continue
        print(f"{name}: {st['clips']:,} clips, {st['hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
