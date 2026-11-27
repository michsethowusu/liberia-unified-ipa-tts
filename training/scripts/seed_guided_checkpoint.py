import argparse
from pathlib import Path

import torch

from training.args import load_args
from training.modules.builders import build_models

EMBEDDING_KEY = "flow_lm.conditioner.embed.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--best-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_args = load_args(args.config)
    torch.manual_seed(train_args.seed)
    model, _, _ = build_models(train_args)
    expected = model.state_dict()
    fresh_embedding = expected[EMBEDDING_KEY].detach().cpu().clone()

    best = torch.load(args.best_checkpoint, map_location="cpu", weights_only=True)
    source = best.get("ema") or best["model"]
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise SystemExit(f"checkpoint shape mismatch: missing={missing[:10]} extra={extra[:10]}")
    bad = [k for k in expected if tuple(source[k].shape) != tuple(expected[k].shape)]
    if bad:
        raise SystemExit(f"checkpoint tensor mismatch: {bad[:10]}")

    state = {k: v.detach().cpu().clone() for k, v in source.items()}
    state[EMBEDDING_KEY] = fresh_embedding
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "model": state, "ema": state}, out)
    print(f"wrote {out}: seeded from EMA of {args.best_checkpoint}, text embedding reset", flush=True)


if __name__ == "__main__":
    main()
