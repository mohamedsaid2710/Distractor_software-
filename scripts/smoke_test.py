import os
import compileall
from set_params import set_params
from input import read_input

# Path relative to the script's directory.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)

# Directories to skip during compilation (large or irrelevant for syntax checks).
_SKIP_DIRS = {'.venv', 'venv', 'env', '__pycache__', 'models', 'node_modules', '.git'}

def main() -> None:
    import re
    skip_rx = re.compile(r'[\\/](' + '|'.join(re.escape(d) for d in _SKIP_DIRS) + r')[\\/]')
    if not compileall.compile_dir(ROOT, quiet=1, rx=skip_rx):
        raise RuntimeError("Smoke test failed: Python compilation errors were found.")

    for path in ("params_en.txt", "params_de.txt", "params_ar.txt"):
        full_path = os.path.join(ROOT, path)
        if os.path.exists(full_path):
            params = set_params(full_path)
            for key in ("min_delta", "min_abs", "num_to_test"):
                if key not in params:
                    raise RuntimeError(f"Missing required key '{key}' in {full_path}")

    # Check for presence of sample files
    for sample in ("English_sample.txt", "german_sample.txt", "arabic_sample.txt"):
        sp = os.path.join(ROOT, sample)
        if os.path.exists(sp):
            read_input(sp)

    print("Smoke test passed")

if __name__ == "__main__":
    main()
