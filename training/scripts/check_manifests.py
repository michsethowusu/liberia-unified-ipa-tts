import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def has_number(text):
    return any(unicodedata.category(ch) in ("Nd", "No", "Nl") for ch in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--tokenizer")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer) if args.tokenizer else None
    for manifest in args.manifests:
        counts = defaultdict(int)
        hours = defaultdict(float)
        tokens = 0
        words = 0
        errors = 0
        for line in Path(manifest).open(encoding="utf-8"):
            row = json.loads(line)
            lang = row.get("language", "unknown")
            if not row.get("transcript") or has_number(row["transcript"]) or not row.get("words"):
                errors += 1
                continue
            if not Path(row["path"]).exists():
                errors += 1
                continue
            w = row["words"]
            if any(float(x.get("end", 0)) <= float(x.get("start", 0)) for x in w):
                errors += 1
                continue
            counts[lang] += 1
            hours[lang] += float(row.get("duration", 0)) / 3600.0
            words += len(w)
            if tok is not None:
                tokens += len(tok.encode(row["transcript"]).ids)
        print(f"{manifest}: {sum(counts.values()):,} rows, {sum(hours.values()):.2f} h, {errors} errors")
        for lang in sorted(counts):
            print(f"  {lang}: {counts[lang]:,} rows, {hours[lang]:.2f} h")
        if tok is not None and words:
            print(f"  tokens/word: {tokens / words:.2f}")


if __name__ == "__main__":
    main()
