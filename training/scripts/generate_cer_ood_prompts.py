import argparse
import json
import os
import random
import re
import time
from pathlib import Path

LANGS = {
    "loma": ("Loma", "lom", "loma"),
    "kpelle": ("Kpelle", "xpe", "kpelle"),
}

PROMPT = """Write exactly {count} short, natural, everyday sentences in {name}.
They must not be religious, biblical, or Bible-study sentences.
They should sound like normal speech about home, school, work, travel, weather, food, or town life.
Keep each sentence between 6 and 14 words.
Return only JSONL, one object per line, with this exact shape:
{{"text": "sentence here"}}
Do not add labels, numbering, code fences, or explanations."""


def read_key(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.split("=", 1)[1].strip().strip('"').strip("'") if "=" in line else line
    raise SystemExit("no key found")


def parse_output(raw, count):
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("{"):
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
        else:
            text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line)
        text = text.strip().strip('"').strip("'")
        if text:
            rows.append(text)
    unique = list(dict.fromkeys(rows))
    if len(unique) < count:
        raise RuntimeError(f"only parsed {len(unique)}/{count} sentences")
    return unique[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=sorted(LANGS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--key-file", type=Path, default=Path("/mnt/volume_d2wey28/.secrets/gemini_keys.txt"))
    ap.add_argument("--backend", default="gemini-live")
    ap.add_argument("--model", default="models/gemini-3.1-flash-live-preview")
    args = ap.parse_args()

    os.environ["GEMINI_API_KEY"] = read_key(args.key_file)
    name, _, style = LANGS[args.language]
    prompt = PROMPT.format(count=args.count, name=name)
    texts = None
    for attempt in range(6):
        try:
            from afrispeech_synth import generate_text

            raw = generate_text(prompt, language="eng", backend=args.backend, model=args.model)
            texts = parse_output(raw, args.count)
            break
        except Exception as exc:
            print(f"attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            time.sleep(min(30, 2**attempt) + random.Random().random())
    if not texts:
        raise SystemExit("generation failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for index, text in enumerate(texts, 1):
            row = {
                "language": args.language,
                "text": text,
                "style": style,
                "source": "ood_gemini",
                "source_id": f"{args.language}_ood_{index:02d}",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(texts)} {args.language} prompts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
