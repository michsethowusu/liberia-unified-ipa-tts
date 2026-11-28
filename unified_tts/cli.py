import argparse
import json
from pathlib import Path

from .g2p import SUPPORTED_LANGS
from .tts import UnifiedTTS

DEFAULT_CHECKPOINT = "hf://AfriSpeech/liberian-unified-ipa-tts/model.safetensors"
DEFAULT_TOKENIZER = "hf://AfriSpeech/liberian-unified-ipa-tts/tokenizer.json"
DEFAULT_BASE_CONFIG = "hf://AfriSpeech/liberian-unified-ipa-tts/config.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description="Unified IPA TTS inference")
    parser.add_argument("--text")
    parser.add_argument("--lang")
    parser.add_argument("--style", help="kolokwa, liberian_english, loma, or kpelle")
    parser.add_argument("--voice")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=Path, help="JSON list of {lang,text,voice,out} objects")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--eos-threshold", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--list-langs", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.list_langs:
        print("\n".join(sorted(SUPPORTED_LANGS)))
        return
    if not args.batch and (not args.text or not args.lang):
        raise SystemExit("provide --text --lang, or --batch")

    tts = UnifiedTTS.load(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        base_config=args.base_config,
        device=args.device,
        cfg=args.cfg,
        temperature=args.temperature,
        eos_threshold=args.eos_threshold,
    )

    if args.batch:
        items = json.loads(args.batch.read_text(encoding="utf-8"))
        for index, item in enumerate(items):
            out = Path(item["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            result = tts.generate(
                item["text"],
                item["lang"],
                item.get("voice"),
                out=out,
                seed=args.seed + index,
                style=item.get("style", item.get("text_style")),
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
        style=args.style,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "audio"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
