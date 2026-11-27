from pathlib import Path

import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel

from .g2p import to_ipa
from .guidance import GuidedTTSModel
from .translator import Translator, should_translate


class UnifiedTTS:
    def __init__(self, model, device, cfg, temperature, use_t5):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.temperature = temperature
        self.use_t5 = use_t5
        self.sample_rate = model.sample_rate
        self.translator = None
        self._states = {}

    @classmethod
    def load(
        cls,
        checkpoint,
        tokenizer,
        base_config,
        device=None,
        cfg=2.0,
        temperature=0.3,
        use_t5=True,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
        config["weights_path"] = str(checkpoint)
        config["replace_characters"] = {}
        config["capitalize_first_letter"] = False
        config["flow_lm"]["lookup_table"]["tokenizer"] = "tokenizers"
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
        config["flow_lm"]["lookup_table"]["n_bins"] = 4000
        config_path = Path(checkpoint).with_name("unified_inference_config.yaml")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        model = TTSModel.load_model(config=str(config_path), temp=temperature).to(device)
        if cfg != 1.0:
            model = GuidedTTSModel.wrap(model, cfg)
        return cls(model, device, cfg, temperature, use_t5)

    def _translator(self):
        if self.use_t5 and self.translator is None:
            self.translator = Translator(device=self.device)
        return self.translator

    def phonemize(self, text, lang):
        translated = None
        if self.use_t5 and should_translate(lang):
            translated = self._translator().translate(text)
            text = translated
        return {
            "text": text,
            "translated_text": translated,
            "ipa": to_ipa(text, lang),
        }

    def _state(self, voice):
        key = str(voice)
        if key not in self._states:
            self._states[key] = self.model.get_state_for_audio_prompt(voice)
        return self._states[key]

    def generate(self, text, lang, voice, out=None, seed=None):
        phonemes = self.phonemize(text, lang)
        if seed is not None:
            torch.manual_seed(seed)
        audio = self.model.generate_audio(self._state(voice), phonemes["ipa"], max_tokens=250)
        audio = audio.detach().cpu().numpy().astype("float32")
        result = {**phonemes, "audio": audio, "voice": str(voice)}
        if out is not None:
            sf.write(out, audio, self.sample_rate)
            result["wav"] = str(out)
        return result
