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


def take_until(rows, seconds, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    chosen = set()
    total = 0.0
    for idx in order:
        if total >= seconds:
            break
        chosen.add(idx)
        total += duration(rows[idx])
    return [row for idx, row in enumerate(rows) if idx in chosen], total


def merge_split(rows, scores, threshold, seed):
    scored = {r["id"]: r for r in scores if r.get("id") and r.get("p_kolokwa") is not None}
    out = []
    skipped = 0
    for row in rows:
        if row.get("source") != "kolokwa_jw":
            out.append(row)
            continue
        score = scored.get(row.get("id"))
        if score is None:
            skipped += 1
            continue
        new = dict(row)
        new["p_kolokwa"] = score["p_kolokwa"]
        new["p_formal"] = score["p_formal"]
        if score["p_kolokwa"] >= threshold:
            new["language"] = "kolokwa"
            new["source"] = "kolokwa_jw_kolokwa"
            new["classifier_label"] = "pure_kolokwa"
        else:
            new["language"] = "liberian_english"
            new["source"] = "kolokwa_jw_formal"
            new["classifier_label"] = "formal_liberian_english"
        out.append(new)

    jw_formal = [r for r in out if r.get("source") == "kolokwa_jw_formal"]
    english_pool = [r for r in out if r.get("source") == "liberian_english"]
    english_target = sum(duration(r) for r in jw_formal)
    selected_english, selected_english_seconds = take_until(english_pool, english_target, seed)
    english = jw_formal + selected_english

    bible = [r for r in out if r.get("source") == "kolokwa_bible"]
    movies = [r for r in out if r.get("source") == "kolokwa_movies"]
    jw_kolokwa = [r for r in out if r.get("source") == "kolokwa_jw_kolokwa"]
    kolokwa_target = sum(duration(r) for r in bible)
    selected_movies_seconds = sum(duration(r) for r in movies)
    kolokwa = bible + jw_kolokwa + movies

    balanced_sources = {
        "kolokwa_jw_formal",
        "liberian_english",
        "kolokwa_bible",
        "kolokwa_movies",
        "kolokwa_jw_kolokwa",
    }
    untouched = [r for r in out if r.get("source") not in balanced_sources]
    return (
        untouched + kolokwa + english,
        english_target,
        english_target + selected_english_seconds,
        kolokwa_target,
        selected_movies_seconds,
        skipped,
    )


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
        print(f"  lang {key}: {by_lang[key][0]:,} rows, {by_lang[key][1]:.2f}h", flush=True)
    for key in sorted(by_src):
        print(f"  src  {key}: {by_src[key][0]:,} rows, {by_src[key][1]:.2f}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/jw_classifier_scores.jsonl"))
    ap.add_argument("--backup-train", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train_with_jw.jsonl"))
    ap.add_argument("--backup-valid", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid_with_jw.jsonl"))
    ap.add_argument("--train", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/train.jsonl"))
    ap.add_argument("--valid", type=Path, default=Path("/mnt/volume_d2wey28/data/unified_ipa_tts/manifests/valid.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    scores = load(args.scores)
    backup_train = load(args.backup_train)
    backup_valid = load(args.backup_valid)
    new_train, eng_target_train, eng_chosen_train, kol_target_train, kol_chosen_train, skipped_train = merge_split(backup_train, scores, args.threshold, args.seed)
    new_valid, eng_target_valid, eng_chosen_valid, kol_target_valid, kol_chosen_valid, skipped_valid = merge_split(backup_valid, scores, args.threshold, args.seed + 100)

    for path, backup in [(args.train, args.train.with_name("train_no_classifier.jsonl")), (args.valid, args.valid.with_name("valid_no_classifier.jsonl"))]:
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)

    for rows, path in [(new_train, args.train), (new_valid, args.valid)]:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    print(
        f"train english target {eng_target_train / 3600:.2f}h, chosen {eng_chosen_train / 3600:.2f}h; "
        f"kolokwa bible target {kol_target_train / 3600:.2f}h, movies chosen {kol_chosen_train / 3600:.2f}h; "
        f"skipped {skipped_train:,}",
        flush=True,
    )
    print(
        f"valid english target {eng_target_valid / 3600:.2f}h, chosen {eng_chosen_valid / 3600:.2f}h; "
        f"kolokwa bible target {kol_target_valid / 3600:.2f}h, movies chosen {kol_chosen_valid / 3600:.2f}h; "
        f"skipped {skipped_valid:,}",
        flush=True,
    )
    report("train", new_train)
    report("valid", new_valid)


if __name__ == "__main__":
    main()
