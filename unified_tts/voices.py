import json
from functools import lru_cache
from pathlib import Path

from .g2p import normalize_lang

VOICE_DIR = Path(__file__).parent / "voice_prompts"


@lru_cache(maxsize=1)
def builtin_voices():
    path = VOICE_DIR / "voice_prompts.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_lang(row["language"]): row for row in rows}


def builtin_voice_path(lang):
    lang = normalize_lang(lang)
    row = builtin_voices().get(lang)
    if not row:
        return None
    wav = VOICE_DIR / str(row.get("file") or Path(str(row.get("wav", ""))).name)
    return wav if wav.exists() else None
