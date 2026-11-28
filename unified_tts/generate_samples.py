import argparse
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from .cli import DEFAULT_BASE_CONFIG, DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER
from .tts import UnifiedTTS
from .voices import builtin_voice_path

LANGS = ["kolokwa", "liberian_english", "loma", "kpelle"]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def load_prompts(path, langs):
    prompts = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        lang = str(row.get("language", row.get("lang", ""))).strip().lower()
        text = str(row.get("text", row.get("transcript_orig", ""))).strip()
        if lang not in langs or not text:
            continue
        prompts.append(
            {
                "language": lang,
                "text": text,
                "style": row.get("text_style", row.get("style")),
                "source": row.get("source", "prompt_file"),
                "source_id": row.get("id", f"{lang}:{len(prompts) + 1:02d}"),
            }
        )
    return prompts


def pick_prompt(rows, rng):
    pool = [
        row
        for row in rows
        if 2.5 <= float(row.get("duration", 0)) <= 4.0
        and row.get("transcript_orig")
        and Path(row.get("path", "")).exists()
    ]
    if not pool:
        pool = [
            row
            for row in rows
            if 1.5 <= float(row.get("duration", 0)) <= 5.0
            and row.get("transcript_orig")
            and Path(row.get("path", "")).exists()
        ]
    if not pool:
        raise RuntimeError("no usable voice prompt")
    return rng.choice(pool)


def pick_texts(rows, prompt_id, count, rng):
    pool = []
    for row in rows:
        text = str(row.get("transcript_orig", "")).strip()
        duration = float(row.get("duration", 0))
        if not text or row.get("id") == prompt_id:
            continue
        if duration < 1.0 or duration > 12.0 or len(text) > 220:
            continue
        pool.append(row)
    rng.shuffle(pool)
    return pool[:count]


def write_html(out_dir, metadata, langs):
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Unified IPA TTS Samples</title><style>",
        "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5}",
        "section{border:1px solid #ddd;border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.25rem}",
        "h2{margin-top:0;text-transform:capitalize}.item{padding:.75rem 0;border-top:1px solid #eee}",
        ".item:first-of-type{border-top:none}audio{width:100%;margin:.35rem 0}",
        "code{font-size:.92rem;white-space:pre-wrap;word-break:break-word}",
        ".meta{color:#666;font-size:.85rem}.voice{background:#f7f7f7;padding:.75rem;border-radius:8px;margin-bottom:.75rem}",
        ".sub{color:#555;margin-bottom:2rem}</style></head><body>",
        "<h1>Unified IPA TTS Samples</h1>",
        f"<div class=\"sub\">Frontend: transcript for Kolokwa/Liberian English, africa-g2p for Loma/Kpelle · Checkpoint: {html.escape(metadata['checkpoint'])}</div>",
    ]
    by_lang = defaultdict(list)
    for sample in metadata["samples"]:
        by_lang[sample["language"]].append(sample)
    for lang in langs:
        samples = by_lang[lang]
        if not samples:
            continue
        parts.append(f"<section><h2>{html.escape(lang)}</h2>")
        parts.append(f"<div class=\"voice\"><strong>Voice prompt</strong><audio controls preload=\"none\" src=\"{html.escape(samples[0]['voice_rel'])}\"></audio></div>")
        for sample in samples:
            input_text = str(sample.get("input_text", sample["text"]))
            parts.append("<div class=\"item\">")
            parts.append(f"<audio controls preload=\"none\" src=\"{html.escape(sample['wav_rel'])}\"></audio>")
            parts.append(f"<div><strong>Input grapheme:</strong> {html.escape(input_text)}</div>")
            label = "Model text" if sample.get("frontend") == "transcript" else "IPA"
            model_text = str(sample.get("text", sample.get("ipa", "")))
            parts.append(f"<div><strong>{label}:</strong> <code>{html.escape(model_text)}</code></div>")
            parts.append(f"<div class=\"meta\">source: {html.escape(str(sample.get('source', '')))} · {html.escape(str(sample.get('source_id', '')))}</div>")
            parts.append("</div>")
        parts.append("</section>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_guided_tts/manifests/valid.jsonl"))
    parser.add_argument("--prompts", type=Path, help="JSONL with language and text fields for out-of-domain samples")
    parser.add_argument("--out", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/samples/teacher_best_inference"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--per-lang", type=int, default=3)
    parser.add_argument("--langs", default=",".join(LANGS))
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--eos-threshold", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts = UnifiedTTS.load(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        base_config=args.base_config,
        cfg=args.cfg,
        temperature=args.temperature,
        eos_threshold=args.eos_threshold,
    )

    manifest_rows = [row for row in load_jsonl(args.manifest) if row.get("language") in langs]
    voice_by_lang = defaultdict(list)
    for row in manifest_rows:
        voice_by_lang[row["language"]].append(row)

    if args.prompts:
        sample_rows = load_prompts(args.prompts, langs)
    else:
        sample_rows = [
            {
                "language": row["language"],
                "text": row["transcript_orig"],
                "style": row.get("text_style", row.get("style")),
                "source": row.get("source"),
                "source_id": row.get("id"),
            }
            for row in manifest_rows
        ]
    sample_by_lang = defaultdict(list)
    for row in sample_rows:
        sample_by_lang[row["language"]].append(row)

    rng = random.Random(args.seed)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "cfg": args.cfg,
        "temperature": args.temperature,
        "frontend": "transcript/africa-g2p",
        "prompts": str(args.prompts) if args.prompts else None,
        "samples": [],
    }

    for lang in langs:
        builtin = builtin_voice_path(lang)
        voice_path = audio_dir / f"{lang}_voice.wav"
        if builtin:
            shutil.copyfile(builtin, voice_path)
            prompt_id = f"builtin:{lang}"
            prompt = {}
        else:
            prompt = pick_prompt(voice_by_lang[lang], rng)
            voice_audio, voice_sr = sf.read(prompt["path"], dtype="float32", always_2d=True)
            voice_audio = voice_audio[: int(4.0 * voice_sr)]
            sf.write(voice_path, voice_audio, voice_sr)
            prompt_id = prompt["id"]
        if args.prompts:
            texts = sample_by_lang[lang][: args.per_lang]
        else:
            texts = pick_texts(voice_by_lang[lang], prompt.get("id"), args.per_lang, rng)
        print(f"{lang}: voice {prompt_id}, {len(texts)} texts", flush=True)
        for index, row in enumerate(texts):
            input_text = str(row.get("text", row.get("transcript_orig", ""))).strip()
            wav_path = audio_dir / f"{lang}_{index + 1:02d}.wav"
            result = tts.generate(
                input_text,
                lang,
                voice_path,
                out=wav_path,
                seed=args.seed + index + langs.index(lang) * 1000,
                style=row.get("style"),
            )
            metadata["samples"].append(
                {
                    "language": lang,
                    "input_text": input_text,
                    "text": result["text"],
                    "condition_prefix": result["condition_prefix"],
                    "translated_text": result["translated_text"],
                    "frontend": result["frontend"],
                    "wav": str(wav_path),
                    "wav_rel": f"audio/{wav_path.name}",
                    "voice": str(voice_path),
                    "voice_rel": f"audio/{voice_path.name}",
                    "voice_id": prompt_id,
                    "source": row.get("source"),
                    "source_id": row.get("source_id", row.get("id")),
                }
            )
            print(f"  {wav_path.name}: {result['text']}", flush=True)

    (args.out / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(args.out, metadata, langs)
    print(f"done: {len(metadata['samples'])} samples in {args.out}", flush=True)


if __name__ == "__main__":
    main()
