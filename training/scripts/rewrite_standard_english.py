import argparse
import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT = """Rewrite the source text in natural standard English.
Keep the meaning, names, and as much of the original wording as possible.
Do not explain, quote, or label the answer. Return only the rewritten sentence.

Source: Your People Who na Die Will Come Back to Life!
Standard English: Your people who have died will come back to life!

Source: Wetin we coming learn about our brother and sister them
Standard English: What we are going to learn about our brothers and sisters

Source: {text}
Standard English:"""


def load_cache(path):
    cache = {}
    if not Path(path).exists():
        return cache
    with Path(path).open(encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            if row.get("id") and row.get("rewrite"):
                cache[row["id"]] = row["rewrite"]
    return cache


def collect(manifests, cache, limit):
    rows = []
    seen = set()
    for manifest in manifests:
        with Path(manifest).open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                if row.get("language") != "liberian_english":
                    continue
                row_id = str(row.get("id") or "")
                text = str(row.get("transcript_orig") or row.get("transcript") or "").strip()
                if not row_id or not text or row_id in seen or row_id in cache:
                    continue
                seen.add(row_id)
                rows.append({"id": row_id, "text": text})
                if limit and len(rows) >= limit:
                    return rows
    return rows


def rewrite_one(row, key, model, rng):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": PROMPT.format(text=row["text"])}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 128,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": key,
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            text = text.strip('"').strip("'").strip()
            if text:
                return text
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        time.sleep(min(30, 2**attempt) + rng.random())
    raise RuntimeError(f"Gemini failed for {row['id']}: {type(last_error).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY or GOOGLE_API_KEY before running this script")

    cache = load_cache(args.out)
    rows = collect(args.manifests, cache, args.limit)
    print(f"rewriting {len(rows):,} rows with {args.model}; {len(cache):,} cached", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    rng = random.Random(2026)
    done = 0
    with args.out.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rewrite_one, row, key, args.model, rng): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            rewrite = future.result()
            with lock:
                fout.write(json.dumps({"id": row["id"], "source": row["text"], "rewrite": rewrite}, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 100 == 0:
                    print(f"rewritten {done:,}/{len(rows):,}", flush=True)
    print(f"done: {done:,} rewrites in {args.out}", flush=True)


if __name__ == "__main__":
    main()
