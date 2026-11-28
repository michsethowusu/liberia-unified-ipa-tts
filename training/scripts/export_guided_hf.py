import argparse
import functools
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi

SUBSETS = {
    "<lang:kolokwa>": "kolokwa",
    "<lang:liberian_english>": "liberian_english",
    "<lang:loma>": "loma",
    "<lang:kpelle>": "kpelle",
}

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("text", pa.string()),
        pa.field("language", pa.string()),
        pa.field("condition_prefix", pa.string()),
        pa.field("text_style", pa.string()),
        pa.field("source", pa.string()),
        pa.field("hf_source", pa.string()),
        pa.field("source_language", pa.string()),
        pa.field("source_transcript", pa.string()),
        pa.field("duration", pa.float32()),
        pa.field("start", pa.float32()),
        pa.field("sample_rate", pa.int32()),
        pa.field("source_path", pa.string()),
        pa.field("words", pa.string()),
    ]
)

README_HEAD = """---
license: cc-by-nc-4.0
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: Liberian Unified Guided TTS
dataset_info:
  configs:
"""

README_FEATURE_BLOCK = """    features:
    - name: audio
      dtype: audio
      audio:
        sampling_rate: 24000
    - name: text
      dtype: string
    - name: language
      dtype: string
    - name: condition_prefix
      dtype: string
    - name: text_style
      dtype: string
    - name: source
      dtype: string
    - name: hf_source
      dtype: string
    - name: source_language
      dtype: string
    - name: source_transcript
      dtype: string
    - name: duration
      dtype: float32
    - name: start
      dtype: float32
    - name: sample_rate
      dtype: int32
    - name: source_path
      dtype: string
    - name: words
      dtype: string
"""

README_TAIL = """
---

# Liberian Unified Guided TTS

Guided TTS training data for Kolokwa, Liberian English, Loma, and Kpelle. Audio comes from Liberian religious speech sources, JW Kolokwa materials, and a small HiFiTTS-2 standard-English replay set. Each subset corresponds to a language-conditioning prefix.

Subset `kolokwa` contains Kolokwa bible audio plus the Kolokwa half of JW Kolokwa data, with both original wording and rewritten standard-English wording. Subset `liberian_english` contains the Liberian-English half of JW Kolokwa data with both wording variants plus HiFiTTS-2 replay. Loma and Kpelle use `africa-g2p` IPA transcripts.
"""


@functools.lru_cache(maxsize=4)
def read_source(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sample_rate


def make_audio(row):
    audio, sample_rate = read_source(str(row.get("path")))
    start = int(float(row.get("start", 0) or 0) * sample_rate)
    duration = float(row.get("duration", 0) or 0)
    end = start + int(duration * sample_rate) if duration > 0 else len(audio)
    clip = audio[start:end]
    buffer = io.BytesIO()
    sf.write(buffer, clip, sample_rate, format="FLAC", subtype="PCM_16")
    return {"bytes": buffer.getvalue(), "path": "audio.flac"}, sample_rate


def make_record(row):
    audio, sample_rate = make_audio(row)
    words = row.get("words")
    return {
        "id": str(row.get("id") or ""),
        "audio": audio,
        "text": str(row.get("transcript_orig") or row.get("transcript") or ""),
        "language": str(row.get("language") or ""),
        "condition_prefix": str(row.get("condition_prefix") or ""),
        "text_style": str(row.get("text_style") or ""),
        "source": str(row.get("source") or ""),
        "hf_source": str(row.get("hf_source") or ""),
        "source_language": str(row.get("source_language") or ""),
        "source_transcript": str(row.get("source_transcript") or ""),
        "duration": float(row.get("duration", 0) or 0),
        "start": float(row.get("start", 0) or 0),
        "sample_rate": int(sample_rate),
        "source_path": str(row.get("path") or ""),
        "words": json.dumps(words or [], ensure_ascii=False),
    }


def read_manifest(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_rows(manifests, limit):
    grouped = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(int))
    for split, path in manifests.items():
        for row in read_manifest(path):
            subset = SUBSETS.get(str(row.get("condition_prefix")))
            if not subset:
                continue
            if limit and counts[subset][split] >= limit:
                continue
            grouped[subset][split].append(row)
            counts[subset][split] += 1
    return grouped, counts


def write_shards(rows, out_dir, shard_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(rows), shard_size):
        chunk = rows[start : start + shard_size]
        records = [make_record(row) for row in chunk]
        path = out_dir / f"part-{start // shard_size:05d}.parquet"
        with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
            writer.write_batch(pa.RecordBatch.from_pylist(records, schema=SCHEMA))
        written += len(records)
        print(f"  {path.name}: {len(records)} rows", flush=True)
    return written


def write_readme(staging, counts):
    parts = [README_HEAD]
    for subset in sorted(counts):
        parts.append(f"  - config_name: {subset}\n")
        parts.append(README_FEATURE_BLOCK)
        parts.append("    splits:\n")
        for split in ("train", "valid"):
            parts.append(f"    - name: {split}\n")
            parts.append(f"      num_examples: {counts[subset][split]}\n")
    parts.append(README_TAIL)
    (staging / "README.md").write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_guided_tts/manifests"))
    parser.add_argument("--staging", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_guided_tts/hf_staging"))
    parser.add_argument("--repo-id", default="AfriSpeech/liberian-unified-ipa-tts-guided")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--limit-per-subset", type=int, default=0)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    manifests = {
        "train": args.manifest_dir / "train.jsonl",
        "valid": args.manifest_dir / "valid.jsonl",
    }
    grouped, counts = group_rows(manifests, args.limit_per_subset)

    if args.staging.exists():
        shutil.rmtree(args.staging)
    args.staging.mkdir(parents=True, exist_ok=True)

    for subset in sorted(grouped):
        for split in ("train", "valid"):
            rows = grouped[subset][split]
            if not rows:
                continue
            print(f"{subset}/{split}: {len(rows)} rows", flush=True)
            write_shards(rows, args.staging / subset / split, args.shard_size)

    write_readme(args.staging, counts)
    print(f"staged {args.staging}", flush=True)

    if args.no_upload:
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.staging),
        commit_message="Upload guided Liberian unified TTS data with per-language subsets",
    )
    print(f"uploaded {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
