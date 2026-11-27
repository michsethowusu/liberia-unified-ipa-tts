import argparse
import json
from collections import defaultdict
from pathlib import Path

TEXT_LANGS = {"kolokwa", "liberian_english"}


def load_rewrite_cache(paths):
    cache = {}
    for path in paths or []:
        with Path(path).open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                if row.get("id") and row.get("rewrite"):
                    cache[row["id"]] = row["rewrite"]
    return cache


def allocate_tokens(total, weights, k):
    if k <= 0:
        return []
    if total <= k:
        return [1] * total + [0] * (k - total)
    positive = [max(1e-6, float(w)) for w in weights]
    raw = [total * w / sum(positive) for w in positive]
    counts = [max(1, int(x)) for x in raw]
    diff = total - sum(counts)
    if diff > 0:
        order = sorted(range(k), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:diff]:
            counts[i] += 1
    elif diff < 0:
        order = sorted(range(k), key=lambda i: raw[i] - counts[i])
        for i in order[:-diff]:
            if counts[i] > 1:
                counts[i] -= 1
    while sum(counts) < total:
        for i in range(k):
            if sum(counts) >= total:
                break
            counts[i] += 1
    while sum(counts) > total:
        for i in reversed(range(k)):
            if sum(counts) <= total:
                break
            if counts[i] > 1:
                counts[i] -= 1
    return counts


def make_word(group, text):
    scores = [float(w.get("score")) for w in group if w.get("score") is not None]
    return {
        "word": text,
        "start": round(min(float(w.get("start", 0)) for w in group), 3),
        "end": round(max(float(w.get("end", 0)) for w in group), 3),
        "score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "orig": text,
    }


def align_rewrite(text, words):
    tokens = str(text).split()
    if not tokens or not words:
        return []
    if len(tokens) <= len(words):
        n = len(tokens)
        out = []
        for i, token in enumerate(tokens):
            start = i * len(words) // n
            end = (i + 1) * len(words) // n
            out.append(make_word(words[start:end], token))
        return out
    weights = [max(0.01, float(w.get("end", 0)) - float(w.get("start", 0))) for w in words]
    counts = allocate_tokens(len(tokens), weights, len(words))
    out = []
    pos = 0
    for word, count in zip(words, counts):
        chunk = tokens[pos : pos + max(1, count)]
        pos += len(chunk)
        out.append(make_word([word], " ".join(chunk)))
    if pos < len(tokens):
        out[-1]["word"] += " " + " ".join(tokens[pos:])
        out[-1]["orig"] = out[-1]["word"]
    return out


def text_row(row):
    out = dict(row)
    words = []
    for word in out.get("words") or []:
        item = dict(word)
        original = str(item.get("orig") or "").strip()
        if original:
            item["word"] = original
        words.append(item)
    out["words"] = words
    out["transcript"] = " ".join(
        str(word.get("word") or "").strip() for word in words if str(word.get("word") or "").strip()
    )
    out["transcript_orig"] = out["transcript"]
    return out


def transform_row(row, rewrites):
    source_language = row.get("language", "")
    if source_language not in TEXT_LANGS:
        return [row]

    base = text_row(row)
    base["language"] = "kolokwa"
    base["source_language"] = source_language
    base["source_transcript"] = str(row.get("transcript_orig") or row.get("transcript") or "")
    base["text_style"] = "liberian_english" if source_language == "liberian_english" else "kolokwa"

    if source_language != "liberian_english":
        return [base]

    rewrite = rewrites.get(str(row.get("id") or ""))
    if not rewrite:
        return [base]

    standard = dict(base)
    standard["id"] = f"{row.get('id', '')}:standard_english"
    standard["transcript"] = " ".join(rewrite.split())
    standard["transcript_orig"] = standard["transcript"]
    standard["words"] = align_rewrite(standard["transcript"], row.get("words") or [])
    standard["text_style"] = "standard_english"
    return [base, standard]


def convert(inp, out, rewrites):
    counts = defaultdict(int)
    hours = defaultdict(float)
    skipped = 0
    with Path(out).open("w", encoding="utf-8") as fout:
        with Path(inp).open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                results = transform_row(row, rewrites)
                for result in results:
                    if not result.get("transcript") or not result.get("words"):
                        skipped += 1
                        continue
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    key = f"{result.get('language', 'unknown')}/{result.get('text_style', 'ipa')}"
                    counts[key] += 1
                    hours[key] += float(result.get("duration", 0)) / 3600.0
    print(f"wrote {out}: {sum(counts.values()):,} rows, {sum(hours.values()):.2f} h, {skipped:,} skipped", flush=True)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]:,} rows, {hours[key]:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in")
    ap.add_argument("--valid-in")
    ap.add_argument("--rewrite-cache", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rewrites = load_rewrite_cache(args.rewrite_cache)
    print(f"loaded {len(rewrites):,} standard-English rewrites", flush=True)
    if args.train_in:
        convert(args.train_in, args.out / "train.jsonl", rewrites)
    if args.valid_in:
        convert(args.valid_in, args.out / "valid.jsonl", rewrites)


if __name__ == "__main__":
    main()
