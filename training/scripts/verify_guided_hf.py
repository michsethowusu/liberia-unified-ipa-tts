from datasets import load_dataset
from huggingface_hub import HfApi

REPO_ID = "AfriSpeech/liberian-unified-ipa-tts-guided"

api = HfApi()
info = api.dataset_info(REPO_ID, files_metadata=False)
print("repo", info.id)
print("private", info.private)
print("license", (info.cardData or {}).get("license"))
print("configs", [c.get("config_name") for c in (info.cardData or {}).get("dataset_info", {}).get("configs", [])])
for subset in ["kolokwa", "liberian_english", "loma", "kpelle"]:
    dataset = load_dataset(REPO_ID, subset)
    row = dataset["train"][0]
    print(subset, {split: len(split_data) for split, split_data in dataset.items()}, row["condition_prefix"], len(row["audio"]["bytes"]))
