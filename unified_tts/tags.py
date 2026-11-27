CONDITION_PREFIXES = {
    "kolokwa": "<lang:kolokwa>",
    "liberian_english": "<lang:liberian_english>",
    "standard_english": "<lang:standard_english>",
    "loma": "<lang:loma>",
    "kpelle": "<lang:kpelle>",
}

STYLE_PREFIXES = {
    "kolokwa": "<lang:kolokwa>",
    "liberian_english": "<lang:liberian_english>",
    "standard_english": "<lang:kolokwa>",
    "english_tts": "<lang:standard_english>",
}


def normalize_style(language, text_style=None):
    style = str(text_style or "").strip().lower()
    if style in STYLE_PREFIXES:
        return style
    lang = str(language or "").strip().lower()
    return lang if lang in CONDITION_PREFIXES else "kolokwa"


def condition_prefix(language, text_style=None):
    return CONDITION_PREFIXES[normalize_style(language, text_style)]
