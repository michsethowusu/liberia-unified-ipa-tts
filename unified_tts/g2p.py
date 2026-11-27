AFRICA_G2P_CODES = {
    "kolokwa": "eng",
    "liberian_english": "eng",
    "loma": "lom",
    "kpelle": "xpe",
}
T5_LANGS = {"kolokwa"}
SUPPORTED_LANGS = set(AFRICA_G2P_CODES)


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


def to_ipa(text, lang):
    from africa_g2p import g2p as africa_g2p

    lang = normalize_lang(lang)
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    return africa_g2p(text, lang=AFRICA_G2P_CODES[lang], output="ipa").strip()
