#!/usr/bin/env python3
"""Download what ONE params file needs, and nothing else.

    uv run python tools/download_models.py -p config/params_de.txt            # download
    uv run python tools/download_models.py -p config/params_de.txt --dry-run  # list + sizes only

For the params file given, this fetches:
  - the language model named in `hf_model_name` (into the Hugging Face cache,
    where the run finds it without network access);
  - the Stanza model for German or Arabic (tags the input sentences);
  - the fastText vectors for the language, only if `semantic_filter: True`.

The English spaCy model is a Python package and comes with `uv sync`.
Everything here would otherwise download on the first run; doing it first
is what you want before a cluster or Colab job.
"""

import argparse
import os
import sys

import os as _os
import sys as _sys

# Make src/maze_automate importable when run as `python tools/<script>.py`.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))


FASTTEXT_GB = 7.2       # cc.<lang>.300.bin, per language
STANZA_GB = 0.6         # per language
STANZA_PROCESSORS = "tokenize,mwt,pos,lemma"   # what selection.py loads

# The weights are published in several formats; the run needs one of them.
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _hf_token():
    return os.environ.get("HF_TOKEN") or None


def _hf_files(repo_id):
    """(files to download, total bytes) for a Hugging Face model."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id, files_metadata=True, token=_hf_token())
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    weights = next((w for w in _WEIGHT_FILES if w in sizes), None)
    if weights is None:
        sharded = [f for f in sizes if f.endswith(".safetensors")] or \
                  [f for f in sizes if f.endswith(".bin")]
        keep_weights = sharded + [f for f in sizes if f.endswith(".index.json")]
    else:
        keep_weights = [weights]
    other = [f for f in sizes
             if "/" not in f and f.endswith((".json", ".txt", ".model"))
             and not f.endswith(".index.json")]
    files = keep_weights + other
    return files, sum(sizes[f] for f in files)


def plan(params):
    """What this params file needs: a list of (label, size in GB, action)."""
    lang = params["language"]
    items = []

    model = params.get("hf_model_name")
    if model and os.path.isdir(model):
        items.append((f"language model {model} (local folder, nothing to download)", 0.0, None))
    elif model:
        files, size = _hf_files(model)

        def fetch(model=model, files=files):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, allow_patterns=files, token=_hf_token())
        items.append((f"language model {model}", size / 1e9, fetch))

    if lang in ("de", "ar"):
        def fetch_stanza(lang=lang):
            import stanza
            stanza.download(lang, processors=STANZA_PROCESSORS, verbose=False)
        items.append((f"Stanza {lang} (sentence tagging)", STANZA_GB, fetch_stanza))

    if params.get("semantic_filter", False):
        def fetch_fasttext(lang=lang):
            import fasttext.util
            target = os.path.expanduser("~/.fasttext")
            os.makedirs(target, exist_ok=True)
            if os.path.exists(os.path.join(target, f"cc.{lang}.300.bin")):
                return
            cwd = os.getcwd()
            try:
                os.chdir(target)            # fasttext.util writes to the CWD
                fasttext.util.download_model(lang, if_exists="ignore")
            finally:
                os.chdir(cwd)
        items.append((f"fastText {lang} vectors (semantic_filter: True)",
                      FASTTEXT_GB, fetch_fasttext))
    return items


def main():
    ap = argparse.ArgumentParser(
        description="Download the models one params file needs.")
    ap.add_argument("-p", "--params", required=True, help="params file, e.g. config/params_de.txt")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be downloaded, with sizes, and exit")
    args = ap.parse_args()

    from maze_automate.params import set_params, ParamsError
    try:
        params = set_params(args.params)
    except (ParamsError, FileNotFoundError) as e:
        sys.exit(f"ERROR [params file]: {e}")

    try:
        items = plan(params)
    except Exception as e:
        sys.exit(f"ERROR [hf_model_name]: cannot look up '{params.get('hf_model_name')}' "
                 f"on Hugging Face: {e}")

    total = sum(size for _, size, _ in items)
    print(f"{args.params} needs:")
    for label, size, _ in items:
        print(f"  {size:5.1f} GB  {label}")
    print(f"  {total:5.1f} GB  total (approximate; anything already present is skipped)")
    if args.dry_run:
        return

    failed = []
    for label, _, action in items:
        if action is None:
            continue
        print(f"\n>>> {label}", flush=True)
        try:
            action()
            print("    done")
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(label)
    if failed:
        sys.exit("\nNot downloaded: " + "; ".join(failed))
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
