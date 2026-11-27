from functools import lru_cache

from huggingface_hub import hf_hub_download

AFRICA_G2P_CODES = {"loma": "lom", "kpelle": "xpe"}
DEEP_PHONEMIZER_MODELS = {
    "kolokwa": "michsethowusu/kolokwa-g2p-bible",
    "liberian_english": "michsethowusu/kolokwa-g2p-jw",
}
T5_LANGS = {"kolokwa", "liberian_english"}
SUPPORTED_LANGS = set(DEEP_PHONEMIZER_MODELS) | set(AFRICA_G2P_CODES)


def normalize_lang(lang):
    key = lang.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "english": "liberian_english",
        "liberian": "liberian_english",
        "lir": "liberian_english",
        "kpe": "kpelle",
        "xpe": "kpelle",
        "loi": "loma",
        "lom": "loma",
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported language: {lang}")
    return key


@lru_cache(maxsize=None)
def load_phonemizer(repo_id):
    from phonemize import Phonemizer

    checkpoint = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
    return Phonemizer.from_checkpoint(checkpoint)


def deep_phonemizer(text, lang):
    repo_id = DEEP_PHONEMIZER_MODELS[lang]
    phonemizer = load_phonemizer(repo_id)
    return phonemizer(text, lang="en_lr").strip()


def africa_g2p_phonemizer(text, lang):
    from africa_g2p import g2p as africa_g2p

    return africa_g2p(text, lang=AFRICA_G2P_CODES[lang], output="ipa").strip()


def to_ipa(text, lang):
    lang = normalize_lang(lang)
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    if lang in DEEP_PHONEMIZER_MODELS:
        return deep_phonemizer(text, lang)
    return africa_g2p_phonemizer(text, lang)
