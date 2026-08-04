import os
import re
import compileall
from set_params import set_params
from input import read_input

# Path relative to the script's directory.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)

# Directories to skip during compilation (large or irrelevant for syntax checks).
# `models/` is deliberately NOT skipped: hf_scorer.py and the per-language
# adapters live there, and skipping the whole tree meant the shared scoring core
# was never syntax-checked by CI. Downloaded Hugging Face weight directories are
# excluded separately below, since those may ship third-party .py files.
_SKIP_DIRS = {'.venv', 'venv', 'env', '__pycache__', 'node_modules', '.git'}

PARAM_FILES = ("params_en.txt", "params_de.txt", "params_ar.txt")
SAMPLE_FILES = ("English_sample.txt", "german_sample.txt", "arabic_sample.txt")
REQUIRED_PARAM_KEYS = ("min_delta", "min_abs", "num_to_test")


def _downloaded_model_dirs():
    """Names of HF snapshot dirs under models/ (anything that isn't a *_code package)."""
    models_dir = os.path.join(ROOT, "models")
    if not os.path.isdir(models_dir):
        return []
    return [
        name for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
        and not name.endswith("_code")
        and not name.startswith(("_", "."))
    ]


def _skip_regex():
    skip = set(_SKIP_DIRS) | set(_downloaded_model_dirs())
    return re.compile(r'[\\/](' + '|'.join(re.escape(d) for d in sorted(skip)) + r')[\\/]')


def main() -> None:
    if not compileall.compile_dir(ROOT, quiet=1, rx=_skip_regex()):
        raise RuntimeError("Smoke test failed: Python compilation errors were found.")

    # These files are part of the repository contract, so a missing one is a
    # failure rather than something to silently skip over.
    for path in PARAM_FILES:
        full_path = os.path.join(ROOT, path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"Smoke test failed: required params file is missing: {path}")
        params = set_params(full_path)
        for key in REQUIRED_PARAM_KEYS:
            if key not in params:
                raise RuntimeError(f"Missing required key '{key}' in {full_path}")

    for sample in SAMPLE_FILES:
        sp = os.path.join(ROOT, sample)
        if not os.path.exists(sp):
            raise RuntimeError(f"Smoke test failed: required sample file is missing: {sample}")
        read_input(sp)

    print("Smoke test passed")


if __name__ == "__main__":
    main()
