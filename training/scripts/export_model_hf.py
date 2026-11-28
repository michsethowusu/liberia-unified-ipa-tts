import argparse
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import HfApi


def dotted_set(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def write_model_card(path, name, repo_id, cfg, checkpoint):
    text = f"""---
license: cc-by-nc-4.0
library_name: pocket-tts
tags:
- text-to-speech
- pocket-tts
- liberian-english
- kolokwa
- loma
- kpelle
---

# {name}

UnifiedIPA Pocket TTS model trained by AfriSpeech for Kolokwa, Liberian English, Loma IPA, and Kpelle IPA.

Repository: `{repo_id}`

Source checkpoint: `{checkpoint}`

Distilled CFG baking: inference should use `cfg={cfg}`.

Built-in voice prompts are bundled under `voice_prompts/`.
"""
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("/mnt/volume_d2wey28/projects/liberian-english-speech/unified_guided_tts/tokenizer/ipa_sp4000.json"))
    ap.add_argument("--voice-dir", type=Path, default=Path("/mnt/volume_d2wey28/projects/liberia-unified-ipa-tts/unified_tts/voice_prompts"))
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    checkpoint = args.checkpoint or args.run_dir / "best_model.safetensors"
    args.export_dir.mkdir(parents=True, exist_ok=True)

    resolved = yaml.safe_load((args.run_dir / "args.yaml").read_text(encoding="utf-8"))
    base_config = Path(resolved["model_config"])
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    for key, value in resolved.get("model_overrides", {}).items():
        dotted_set(config, key, value)

    config["weights_path"] = f"hf://{args.repo_id}/model.safetensors"
    config["replace_characters"] = {}
    config["capitalize_first_letter"] = False
    config["flow_lm"]["lookup_table"]["tokenizer"] = "tokenizers"
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = f"hf://{args.repo_id}/tokenizer.json"
    config["flow_lm"]["lookup_table"]["n_bins"] = 4000

    shutil.copyfile(checkpoint, args.export_dir / "model.safetensors")
    shutil.copyfile(args.tokenizer, args.export_dir / "tokenizer.json")
    (args.export_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_model_card(args.export_dir / "README.md", args.name, args.repo_id, args.cfg, checkpoint)

    voice_out = args.export_dir / "voice_prompts"
    if voice_out.exists():
        shutil.rmtree(voice_out)
    shutil.copytree(args.voice_dir, voice_out)
    voice_manifest = voice_out / "voice_prompts.json"
    if voice_manifest.exists():
        rows = json.loads(voice_manifest.read_text(encoding="utf-8"))
        for row in rows:
            row["wav"] = str(Path("voice_prompts") / f"{row['language']}.wav")
            row["file"] = f"{row['language']}.wav"
        voice_manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.push:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=str(args.export_dir), repo_type="model")
        print(f"pushed {args.repo_id}")
    else:
        print(f"prepared {args.export_dir}")


if __name__ == "__main__":
    main()
