from pathlib import Path

import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel

from .g2p import TEXT_LANGS, normalize_lang, normalize_text, to_ipa
from .guidance import GuidedTTSModel


class UnifiedTTS:
    def __init__(self, model, device, cfg, temperature):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.temperature = temperature
        self.sample_rate = model.sample_rate
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
        return cls(model, device, cfg, temperature)

    def phonemize(self, text, lang):
        lang = normalize_lang(lang)
        if lang in TEXT_LANGS:
            model_text = normalize_text(text)
            frontend = "transcript"
        else:
            model_text = to_ipa(text, lang)
            frontend = "africa-g2p"
        return {
            "text": model_text,
            "translated_text": None,
            "frontend": frontend,
        }

    def _state(self, voice):
        key = str(voice)
        if key not in self._states:
            self._states[key] = self.model.get_state_for_audio_prompt(voice)
        return self._states[key]

    def generate(self, text, lang, voice, out=None, seed=None):
        model_input = self.phonemize(text, lang)
        if seed is not None:
            torch.manual_seed(seed)
        audio = self.model.generate_audio(self._state(voice), model_input["text"], max_tokens=250)
        audio = audio.detach().cpu().numpy().astype("float32")
        result = {**model_input, "audio": audio, "voice": str(voice)}
        if out is not None:
            sf.write(out, audio, self.sample_rate)
            result["wav"] = str(out)
        return result
