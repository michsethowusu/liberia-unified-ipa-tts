import argparse
import json
import math
import random
import re
import string
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import sherpa_onnx

BASE = "/mnt/volume_d2wey28/models/sherpa"
FP32_DIR = f"{BASE}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05"
INT8_DIR = f"{BASE}/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"
PUNCT = string.punctuation + "“”‘’–—…¡¿"


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.translate(str.maketrans({char: " " for char in PUNCT}))
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def edit_distance(ref, hyp):
    if not ref:
        return len(hyp), 0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_char in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            )
        previous = current
    return previous[-1], len(ref)


def reference_from(row):
    for key in ("transcript_orig", "input_text", "text", "transcript"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def path_from(row):
    for key in ("wav", "path"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def language_from(row):
    for key in ("language", "lang"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def limit_rows(rows, per_lang, seed):
    if not per_lang:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[language_from(row)].append(row)
    rng = random.Random(seed)
    kept = []
    for lang in sorted(grouped):
        items = grouped[lang]
        rng.shuffle(items)
        kept.extend(items[:per_lang])
    return kept


def load_audio(row):
    audio, sample_rate = sf.read(path_from(row), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    start = float(row.get("start", 0) or 0)
    duration = float(row.get("duration", 0) or 0)
    if start > 0:
        audio = audio[int(start * sample_rate) :]
    if duration > 0:
        audio = audio[: int(duration * sample_rate)]
    return sample_rate, audio


def summarize(items, skipped):
    groups = defaultdict(list)
    for item in items:
        groups["overall"].append(item)
        groups[item["language"]].append(item)
        if item.get("condition_prefix"):
            groups["style:" + item["condition_prefix"]].append(item)
    result = {"skipped": skipped, "groups": {}}
    for name, rows in groups.items():
        edits = sum(row["edits"] for row in rows)
        chars = sum(row["ref_chars"] for row in rows)
        result["groups"][name] = {
            "count": len(rows),
            "macro_cer": sum(row["cer"] for row in rows) / len(rows),
            "micro_cer": edits / chars if chars else math.nan,
            "edits": edits,
            "ref_chars": chars,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="CER eval with sherpa-onnx Omnilingual ASR CTC")
    parser.add_argument("--manifest", type=Path, help="JSONL with path and transcript fields")
    parser.add_argument("--generated-meta", type=Path, help="meta.json from unified_tts.generate_samples")
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/cer"))
    parser.add_argument("--name", default="eval")
    parser.add_argument("--model-dir", type=Path, default=Path(FP32_DIR))
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--per-lang", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.manifest and not args.generated_meta:
        raise SystemExit("provide --manifest or --generated-meta")
    if args.int8 and args.model_dir == Path(FP32_DIR):
        args.model_dir = Path(INT8_DIR)

    model_file = args.model_file or args.model_dir / ("model.int8.onnx" if args.int8 else "model.onnx")
    tokens_file = args.tokens_file or args.model_dir / "tokens.txt"

    recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(model_file),
        tokens=str(tokens_file),
        num_threads=args.num_threads,
        provider="cpu",
    )

    all_rows = []
    if args.manifest:
        rows = limit_rows(read_jsonl(args.manifest), args.per_lang, args.seed)
        for row in rows:
            row["eval_kind"] = "natural"
        all_rows.extend(rows)
    if args.generated_meta:
        meta = json.loads(args.generated_meta.read_text(encoding="utf-8"))
        rows = meta.get("samples", [])
        for row in rows:
            row["eval_kind"] = "generated"
        all_rows.extend(limit_rows(rows, args.per_lang, args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.out_dir / f"{args.name}_details.jsonl"
    summary_path = args.out_dir / f"{args.name}_summary.json"

    items = []
    skipped = []
    with details_path.open("w", encoding="utf-8") as details:
        for start in range(0, len(all_rows), args.batch_size):
            batch_rows = all_rows[start : start + args.batch_size]
            streams = []
            valid = []
            for row in batch_rows:
                path = path_from(row)
                ref = normalize(reference_from(row))
                try:
                    sample_rate, audio = load_audio(row)
                    duration = len(audio) / sample_rate
                    if duration > 40.0:
                        skipped.append({"id": row.get("id"), "reason": "audio_over_40s", "duration": duration})
                        continue
                except Exception as exc:
                    skipped.append({"id": row.get("id"), "reason": "audio_read_error", "error": str(exc)})
                    continue
                stream = recognizer.create_stream()
                stream.accept_waveform(int(sample_rate), audio)
                streams.append(stream)
                valid.append((row, ref, duration))
            if not streams:
                continue
            recognizer.decode_streams(streams)
            for (row, ref, duration), stream in zip(valid, streams):
                hyp = normalize(stream.result.text)
                edits, chars = edit_distance(ref, hyp)
                item = {
                    "id": row.get("id", row.get("source_id")),
                    "language": language_from(row),
                    "condition_prefix": row.get("condition_prefix", row.get("text_style")),
                    "source": row.get("source"),
                    "eval_kind": row.get("eval_kind"),
                    "wav": path_from(row),
                    "duration": duration,
                    "ref": ref,
                    "hyp": hyp,
                    "edits": edits,
                    "ref_chars": chars,
                    "cer": edits / chars if chars else math.nan,
                }
                items.append(item)
                details.write(json.dumps(item, ensure_ascii=False) + "\n")
                details.flush()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "model": str(model_file),
        "tokens": str(tokens_file),
        "manifest": str(args.manifest) if args.manifest else None,
        "generated_meta": str(args.generated_meta) if args.generated_meta else None,
        "per_lang": args.per_lang,
        "summary": summarize(items, skipped),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary["summary"]["groups"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {details_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
