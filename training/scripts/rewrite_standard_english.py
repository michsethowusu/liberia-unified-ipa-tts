import argparse
import json
import os
import random
import threading
import time
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


def parse_key(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" in line:
        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return line


def read_keys(key_file):
    keys = []
    env_keys = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    for line in env_keys.replace(",", "\n").splitlines():
        key = parse_key(line)
        if key:
            keys.append(key)
    if key_file:
        for line in Path(key_file).read_text(encoding="utf-8").splitlines():
            key = parse_key(line)
            if key:
                keys.append(key)
    keys = list(dict.fromkeys(keys))
    if not keys:
        raise SystemExit("set GEMINI_API_KEY, GOOGLE_API_KEY, or --key-file")
    return keys


def load_cache(paths):
    cache = {}
    for path in paths:
        if not Path(path).exists():
            continue
        with Path(path).open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                if row.get("id") and row.get("rewrite"):
                    cache[row["id"]] = row["rewrite"]
    return cache


def sample_balanced_le(rows, balance_to, seed=2026):
    if not balance_to:
        return rows
    target = sum(float(r.get("duration", 0)) for r in rows if r.get("language") == balance_to) / 3600.0
    le_rows = [r for r in rows if r.get("language") == "liberian_english"]
    if not target or not le_rows:
        return rows
    pool = list(le_rows)
    random.Random(seed).shuffle(pool)
    chosen = set()
    hours = 0.0
    for row in pool:
        if hours >= target:
            break
        chosen.add(row["_index"])
        hours += float(row.get("duration", 0)) / 3600.0
    sampled = [r for r in rows if r.get("language") != "liberian_english" or r["_index"] in chosen]
    print(f"balanced liberian_english to {hours:.2f}h (target {target:.2f}h)", flush=True)
    return sampled


def load_rows(path, balance_to):
    rows = []
    with Path(path).open(encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            row["_index"] = len(rows)
            rows.append(row)
    return sample_balanced_le(rows, balance_to)


def collect(manifests, cache, limit, key_index, key_count, balance_to, languages):
    rows = []
    seen = set()
    total = 0
    for manifest in manifests:
        for row in load_rows(manifest, balance_to):
            if row.get("language") not in languages:
                continue
            row_id = str(row.get("id") or "")
            text = str(row.get("transcript_orig") or row.get("transcript") or "").strip()
            if not row_id or not text or row_id in seen or row_id in cache:
                continue
            seen.add(row_id)
            index = total
            total += 1
            if index % key_count == key_index:
                rows.append({"id": row_id, "text": text, "index": index})
            if limit and len(rows) >= limit:
                return rows
    return rows


def rewrite_one(row, backend, model):
    from afrispeech_synth import generate_text

    last_error = None
    for attempt in range(6):
        try:
            text = generate_text(
                PROMPT.format(text=row["text"]),
                language="eng",
                backend=backend,
                model=model,
            ).strip()
            text = text.strip('"').strip("'").strip()
            if text:
                return text
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        jitter = random.Random(f"{row['id']}:{attempt}").random()
        time.sleep(min(30, 2**attempt) + jitter)
    raise RuntimeError(f"Gemini failed for {row['id']}: {type(last_error).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["gemini", "gemini-live"], default="gemini-live")
    ap.add_argument("--balance-le-to", default="kolokwa")
    ap.add_argument("--languages", nargs="+", default=["liberian_english", "kolokwa"])
    ap.add_argument("--model")
    ap.add_argument("--key-file", type=Path, default=os.environ.get("GEMINI_KEY_FILE"))
    ap.add_argument("--key-index", type=int, default=0)
    ap.add_argument("--key-count", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    keys = read_keys(args.key_file)
    if args.key_index >= len(keys):
        raise SystemExit(f"key-index {args.key_index} out of range for {len(keys)} keys")
    os.environ["GEMINI_API_KEY"] = keys[args.key_index]
    model = args.model or (
        "models/gemini-3.1-flash-live-preview"
        if args.backend == "gemini-live"
        else "gemini-3.1-flash-tts-preview"
    )

    cache = load_cache([args.out])
    rows = collect(
        args.manifests,
        cache,
        args.limit,
        args.key_index,
        args.key_count,
        args.balance_le_to,
        set(args.languages),
    )
    print(
        f"rewriting {len(rows):,} rows with {args.backend}/{model} key {args.key_index + 1}/{len(keys)}; "
        f"{len(cache):,} cached",
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = 0
    with args.out.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rewrite_one, row, args.backend, model): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            rewrite = future.result()
            with lock:
                fout.write(json.dumps({"id": row["id"], "source": row["text"], "rewrite": rewrite}, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % 100 == 0:
                    print(f"key {args.key_index}: rewritten {done:,}/{len(rows):,}", flush=True)
    print(f"done: key {args.key_index} wrote {done:,} rewrites to {args.out}", flush=True)


if __name__ == "__main__":
    main()
