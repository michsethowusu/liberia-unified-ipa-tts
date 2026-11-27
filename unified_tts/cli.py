import argparse
import json
from pathlib import Path

from .g2p import SUPPORTED_LANGS
from .tts import UnifiedTTS

DEFAULT_CHECKPOINT = Path("/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts/runs/unified_ipa_teacher/best_model.safetensors")
DEFAULT_TOKENIZER = Path("/mnt/volume_d2wey28/projects/liberian-english-speech/unified_ipa_tts/tokenizer/ipa_sp4000.json")
DEFAULT_BASE_CONFIG = Path("/mnt/volume_d2wey28/projects/pocket-tts-twi/pocket_tts/config/english_2026-04_24l.yaml")


def build_parser():
    parser = argparse.ArgumentParser(description="Unified IPA TTS inference")
    parser.add_argument("--text")
    parser.add_argument("--lang")
    parser.add_argument("--voice")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=Path, help="JSON list of {lang,text,voice,out} objects")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-t5", action="store_true")
    parser.add_argument("--list-langs", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.list_langs:
        print("\n".join(sorted(SUPPORTED_LANGS)))
        return
    if not args.batch and (not args.text or not args.lang or not args.voice):
        raise SystemExit("provide --text --lang --voice, or --batch")

    tts = UnifiedTTS.load(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        base_config=args.base_config,
        device=args.device,
        cfg=args.cfg,
        temperature=args.temperature,
        use_t5=not args.no_t5,
    )

    if args.batch:
        items = json.loads(args.batch.read_text(encoding="utf-8"))
        for index, item in enumerate(items):
            out = Path(item["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            result = tts.generate(
                item["text"],
                item["lang"],
                item["voice"],
                out=out,
                seed=args.seed + index,
            )
            print(json.dumps({k: v for k, v in result.items() if k != "audio"}, ensure_ascii=False), flush=True)
        return

    if args.out is None:
        raise SystemExit("provide --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = tts.generate(
        args.text,
        args.lang,
        args.voice,
        out=args.out,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "audio"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
