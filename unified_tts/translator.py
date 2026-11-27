from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .g2p import T5_LANGS, normalize_lang

MODEL_ID = "AfriSpeech/t5-small-english-kolokwa"
PREFIX = "translate English to Liberian English: "


@lru_cache(maxsize=None)
def _load(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    return tokenizer, model


def should_translate(lang):
    return normalize_lang(lang) in T5_LANGS


class Translator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = _load(self.device)

    def translate(self, text):
        text = " ".join(str(text).strip().split())
        if not text:
            return ""
        prompt = f"{PREFIX}{text}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=2,
                early_stopping=True,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
