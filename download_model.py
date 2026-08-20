#!/usr/bin/env python3

import argparse
import os
from huggingface_hub import snapshot_download

MODEL_SPECS = {
    "german": {
        "repo_id": "benjamin/gerpt2",
        "local_dir": "models/benjamin-gerpt2",
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
    """True only for a COMPLETE local model.

    Weights alone used to count as present, so an interrupted download -- the
    weights are the first and largest file -- looked finished and every later
    retry skipped it. The failure then surfaced much later as "Failed to load
    tokenizer", which points at the wrong thing. A model needs its config and
    its tokenizer files to load at all.
    """
    weights = (
        os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))
        or os.path.exists(os.path.join(local_dir, "model.safetensors"))
    )
    config = os.path.exists(os.path.join(local_dir, "config.json"))
    tokenizer = any(
        os.path.exists(os.path.join(local_dir, name))
        for name in ("tokenizer.json", "vocab.json", "vocab.txt",
                     "spiece.model", "tokenizer_config.json")
    )
    return weights and config and tokenizer


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


# Approximate on-disk size of each download, for the footprint summary.
SIZES_GB = {
    "english": 1.5,     # openai-community/gpt2-medium
    "german": 0.5,      # benjamin/gerpt2
    "arabic": 1.5,      # aubmindlab/aragpt2-medium
    "fasttext": 7.2,    # cc.<lang>.300.bin, PER LANGUAGE (plus ~4.5 GB .gz)
    "spacy_en": 0.6,    # en_core_web_lg, installed by pip
    "stanza": 0.6,      # per language, downloaded on first use
    "farasa": 0.24,     # Arabic only, downloaded on first use; also needs a JVM
}


def print_footprint() -> None:
    """What a full install actually costs on disk.

    No single figure for this existed anywhere, so the 7 GB per language of
    fastText vectors -- downloaded automatically on a new user's first run,
    because `semantic_filter: True` is the default in all three params files --
    was a surprise rather than a decision.
    """
    print("Download footprint (approximate, on disk):")
    print(f"  English : GPT-2 medium {SIZES_GB['english']:.1f} GB"
          f" + spaCy en_core_web_lg {SIZES_GB['spacy_en']:.1f} GB")
    print(f"  German  : GerPT2 {SIZES_GB['german']:.1f} GB"
          f" + Stanza de {SIZES_GB['stanza']:.1f} GB  (HanTa ships with the package)")
    print(f"  Arabic  : AraGPT2 medium {SIZES_GB['arabic']:.1f} GB"
          f" + Stanza ar {SIZES_GB['stanza']:.1f} GB"
          f" + Farasa {SIZES_GB['farasa']:.2f} GB (needs a Java runtime)")
    print(f"  fastText: {SIZES_GB['fasttext']:.1f} GB PER LANGUAGE, only if"
          f" semantic_filter is True (it is, by default)")
    total = (SIZES_GB['english'] + SIZES_GB['german'] + SIZES_GB['arabic']
             + SIZES_GB['spacy_en'] + 2 * SIZES_GB['stanza'] + SIZES_GB['farasa']
             + 3 * SIZES_GB['fasttext'])
    print(f"  ALL THREE LANGUAGES with semantic filtering: ~{total:.0f} GB")
    print("  Set `semantic_filter: False` in the params file to skip the"
          " fastText vectors entirely.")


def download_fasttext(lang: str) -> None:
    """Fetch the fastText vectors for one language, deliberately.

    `semantic_filter: True` is the default, and semantic_filter.py downloads
    ~7 GB on first use with no warning if the file is absent. Doing it here
    makes it an explicit, resumable step like the Hugging Face models.
    """
    target_dir = os.path.expanduser("~/.fasttext")
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, f"cc.{lang}.300.bin")
    if os.path.exists(dest):
        print(f"fastText vectors already present at {dest}")
        return
    try:
        import fasttext.util
    except ImportError:
        print("fasttext is not installed. Install it with: pip install '.[semantic]'")
        return
    cwd = os.getcwd()
    print(f"Downloading fastText '{lang}' vectors (~{SIZES_GB['fasttext']:.1f} GB) "
          f"to {target_dir} ...")
    try:
        os.chdir(target_dir)          # fasttext.util writes to the CWD
        fasttext.util.download_model(lang, if_exists="ignore")
        print(f"Success: {dest}")
    except Exception as e:
        print(f"Error downloading fastText '{lang}': {e}")
    finally:
        os.chdir(cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local model files for this repo")
    parser.add_argument("--english", action="store_true", help="download openai-community/gpt2-medium")
    parser.add_argument("--german", action="store_true", help="download benjamin/gerpt2")
    parser.add_argument("--arabic", action="store_true", help="download aubmindlab/aragpt2-medium")
    parser.add_argument("--all", action="store_true", help="download all language models")
    parser.add_argument("--fasttext", choices=["en", "de", "ar"], action="append",
                        metavar="LANG",
                        help="download fastText vectors for LANG (~7 GB each), "
                             "needed only when semantic_filter is True. Repeatable.")
    parser.add_argument("--footprint", action="store_true",
                        help="print the total download size and exit")
    args = parser.parse_args()

    if args.footprint:
        print_footprint()
        return

    for lang in (args.fasttext or []):
        download_fasttext(lang)

    wants_hf = args.all or args.english or args.german or args.arabic
    if not wants_hf and args.fasttext:
        return

    if args.all or not wants_hf:
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
