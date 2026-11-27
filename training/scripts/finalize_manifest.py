import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def has_number(text):
    return any(unicodedata.category(ch) in ("Nd", "No", "Nl") for ch in text)


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


def finalize_row(row):
    words = row.get("words")
    tokens = str(row.get("ipa") or "").split()
    if not words or not tokens or has_number(" ".join(tokens)):
        return None

    groups = []
    if len(tokens) < len(words):
        n_groups = len(tokens)
        for i in range(n_groups):
            a = i * len(words) // n_groups
            b = (i + 1) * len(words) // n_groups
            groups.append(words[a:b])
    else:
        weights = [max(0.01, float(w.get("end", 0)) - float(w.get("start", 0))) for w in words]
        counts = allocate_tokens(len(tokens), weights, len(words))
        pos = 0
        for i, count in enumerate(counts):
            groups.append(words[i:i + 1])
            groups[-1][0]["__ipa_tokens"] = tokens[pos:pos + count]
            pos += count

    out_words = []
    pos = 0
    for group in groups:
        if len(tokens) < len(row["words"]):
            chunk = [tokens[pos]]
            pos += 1
        else:
            chunk = group[0].get("__ipa_tokens") or []
        if not chunk:
            return None
        start = min(float(w.get("start", 0)) for w in group)
        end = max(float(w.get("end", 0)) for w in group)
        scores = [w.get("score") for w in group if w.get("score") is not None]
        score = round(sum(float(s) for s in scores) / len(scores), 3) if scores else 0.0
        out_words.append({
            "word": " ".join(chunk),
            "start": round(start, 3),
            "end": round(end, 3),
            "score": score,
            "orig": " ".join(str(w.get("word", "")) for w in group),
        })

    out = {
        "id": row.get("id", ""),
        "path": row["path"],
        "start": row["start"],
        "duration": row["duration"],
        "transcript": " ".join(tokens),
        "language": row.get("language", ""),
        "iso": row.get("iso", "eng"),
        "g2p": row.get("g2p", ""),
        "source": row.get("source", ""),
        "hf_source": row.get("hf_source", ""),
        "split": row.get("split", "train"),
        "align": "ctc",
        "words": out_words,
        "transcript_orig": row.get("transcript", ""),
        "ipa": row.get("ipa", ""),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()

    counts = defaultdict(int)
    hours = defaultdict(float)
    skipped = 0
    with Path(args.out).open("w", encoding="utf-8") as fout:
        with Path(args.inp).open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                out = finalize_row(row)
                if out is None:
                    skipped += 1
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                key = out["language"]
                counts[key] += 1
                hours[key] += float(out["duration"]) / 3600.0
    print(f"wrote {args.out}: {sum(counts.values()):,} rows, {sum(hours.values()):.2f} h, {skipped:,} skipped", flush=True)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]:,} rows, {hours[key]:.2f} h", flush=True)


if __name__ == "__main__":
    main()
