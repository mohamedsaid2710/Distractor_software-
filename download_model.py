#!/usr/bin/env python3

import argparse
import os
from huggingface_hub import snapshot_download

MODEL_SPECS = {
    "german": {
        "repo_id": "dbmdz/german-gpt2",
        "local_dir": "models/dbmdz-german-gpt2",
    },
    "english": {
        "repo_id": "openai-community/gpt2-medium",
        "local_dir": "models/openai-community-gpt2-medium",
    },
    "arabic": {
        "repo_id": "aubmindlab/aragpt2-medium",
        "local_dir": "models/aubmindlab-aragpt2-medium",
    },
}

# Keep only the files needed by the current pipeline.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.onnx", "onnx/*", "*/onnx/*"]


def has_weights(local_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))
        or os.path.exists(os.path.join(local_dir, "model.safetensors"))
    )


def download_model(name: str) -> None:
    spec = MODEL_SPECS[name]
    repo_id = spec["repo_id"]
    local_dir = spec["local_dir"]

    print(f"Downloading {repo_id} to {local_dir}...")

    if has_weights(local_dir):
        print(f"Model weights already present in {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=IGNORE_PATTERNS,
        )
        if has_weights(local_dir):
            print(f"Success: model downloaded to {local_dir}")
        else:
            print(
                "Download completed, but no weight file was found "
                f"in {local_dir}. Check Hugging Face files for {repo_id}."
            )
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        print("Check your internet connection and model access.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local model files for this repo")
    parser.add_argument("--english", action="store_true", help="download openai-community/gpt2-medium")
    parser.add_argument("--german", action="store_true", help="download dbmdz/german-gpt2")
    parser.add_argument("--arabic", action="store_true", help="download aubmindlab/aragpt2-medium")
    parser.add_argument("--all", action="store_true", help="download all models")
    args = parser.parse_args()

    if args.all or (not args.english and not args.german and not args.arabic):
        for name in MODEL_SPECS:
            download_model(name)
        return

    if args.english:
        download_model("english")
    if args.german:
        download_model("german")
    if args.arabic:
        download_model("arabic")


if __name__ == "__main__":
    main()
