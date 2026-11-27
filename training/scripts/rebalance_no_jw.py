import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def duration(row):
    return float(row.get("duration", 0.0))


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def rebalance(rows, seed):
    kept = [r for r in rows if r.get("source") != "kolokwa_jw"]
    target = sum(duration(r) for r in kept if r.get("source") in {"kolokwa_bible", "kolokwa_movies"})
    english = [r for r in kept if r.get("source") == "liberian_english"]
    others = [r for r in kept if r.get("source") != "liberian_english"]
    order = list(range(len(english)))
    random.Random(seed).shuffle(order)
    chosen = set()
    total = 0.0
    for idx in order:
        if total >= target:
            break
        chosen.add(idx)
        total += duration(english[idx])
    selected = [r for idx, r in enumerate(english) if idx in chosen]
    return others + selected, target, total


def report(name, rows):
    by_lang = defaultdict(lambda: [0, 0.0])
    by_src = defaultdict(lambda: [0, 0.0])
    for r in rows:
        d = duration(r) / 3600.0
        by_lang[r.get("language", "")][0] += 1
        by_lang[r.get("language", "")][1] += d
        by_src[r.get("source", "")][0] += 1
        by_src[r.get("source", "")][1] += d
    print(f"{name}: {len(rows):,} rows, {sum(v[1] for v in by_lang.values()):.2f}h", flush=True)
    for key in sorted(by_lang):
        count, hours = by_lang[key]
        print(f"  lang {key}: {count:,} rows, {hours:.2f}h", flush=True)
    for key in sorted(by_src):
        count, hours = by_src[key]
        print(f"  src  {key}: {count:,} rows, {hours:.2f}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl"))
    ap.add_argument("--valid", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_rows = load(args.train)
    valid_rows = load(args.valid)
    new_train, target_train, chosen_train = rebalance(train_rows, args.seed)
    new_valid, target_valid, chosen_valid = rebalance(valid_rows, args.seed)

    for path, backup in [(args.train, args.train.with_name("train_with_jw.jsonl")), (args.valid, args.valid.with_name("valid_with_jw.jsonl"))]:
        if not backup.exists():
            shutil.copy2(path, backup)

    for rows, path in [(new_train, args.train), (new_valid, args.valid)]:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    print(f"train english target {target_train / 3600:.2f}h, selected {chosen_train / 3600:.2f}h", flush=True)
    print(f"valid english target {target_valid / 3600:.2f}h, selected {chosen_valid / 3600:.2f}h", flush=True)
    report("train", new_train)
    report("valid", new_valid)


if __name__ == "__main__":
    main()
