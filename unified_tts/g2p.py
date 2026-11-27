TEXT_LANGS = {"kolokwa", "liberian_english"}
AFRICA_G2P_CODES = {
    "loma": "lom",
    "kpelle": "xpe",
}
T5_LANGS = set()
SUPPORTED_LANGS = TEXT_LANGS | set(AFRICA_G2P_CODES)


def normalize_lang(lang):
    key = str(lang).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "english": "liberian_english",
        "liberian": "liberian_english",
        "liberian_english": "liberian_english",
        "lir": "liberian_english",
        "kpe": "kpelle",
        "xpe": "kpelle",
        "kpelle": "kpelle",
        "loi": "loma",
        "lom": "loma",
        "loma": "loma",
        "kolokwa": "kolokwa",
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported language: {lang}")
    return key


def normalize_text(text):
    return " ".join(str(text).strip().split())


def to_ipa(text, lang):
    from africa_g2p import g2p as africa_g2p

    lang = normalize_lang(lang)
    if lang in TEXT_LANGS:
        raise ValueError(f"{lang} uses transcript text, not africa-g2p")
    text = normalize_text(text)
    if not text:
        return ""
    return africa_g2p(text, lang=AFRICA_G2P_CODES[lang], output="ipa").strip()
