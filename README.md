# $\color{RoyalBlue}\text{Distractor Software}$

**Maze-style distractor stimulus generator for (psycho)linguistic experiments in English, German, and Arabic.**

Built on Transformer-based language models (GPT-2), the pipeline selects real-word distractors that are contextually implausible while matching the target word in length and frequency range.

## $\color{DarkSlateGray}\text{Attribution}$

>Based on the original [Maze repository](https://github.com/vboyce/Maze) by Victoria Boyce. This implementation has been extensively adapted and now features distinct Transformer models, automated language-specific NLP tools, GPU batch processing, semantic embeddings, and an interactive config-tuning workflow.

>Special thanks to [Titus von der Malsburg](https://tmalsburg.github.io/) for all the help. His suggestions and feedback have been a huge part of why this software keeps improving.

## $\color{SteelBlue}\text{Overview and Capabilities}$

- **Supported Languages:** 
  - English (`gpt2-medium` via spaCy `en_core_web_lg`)
  - German (`benjamin/gerpt2` via HanTa morphological tagging & `Stanza` context tagging)
  - Arabic (`aubmindlab/aragpt2-medium` via Farasa `farasapy` & `Stanza`)
- **Generation Modes:** Choose between threshold-first (Mode A) or maximum-implausibility scoring (Mode B).
- **Linguistic Precision:**
  - Length and Zipf frequency matching.
  - Optional **fastText Semantic Filtering** to reject words from similar domains (e.g., avoiding "Apple" -> "Orange").
  - **Part-Of-Speech Matching** to ensure natural grammar structure (Verbs match Verbs, Nouns match Nouns).
- **Fast GPU Processing:** Batch-optimized surprisal scoring scales automatically to available hardware.
- **First-Word Placeholder:** Because the first word of a sentence has no prior context, generating a meaningful linguistic distractor is impossible. The software automatically uses a length-matched placeholder (e.g., `x-x-x`) for the first token to ensure a neutral start for participants.
- **Output Formats:** Standard delimited tables or ready-to-deploy PCIbex lines (`ibexify`).

## $\color{SteelBlue}\text{Quick Start}$

It is **highly recommended** to run this software on a GPU-enabled environment (like Google Colab or an academic computing cluster).

```bash
# Clone the repository
git clone https://github.com/mohamedsaid2710/Distractor_software-.git
cd Distractor_software-

# Install dependencies using uv (Fast & Reliable)
uv sync
source .venv/bin/activate  # (Optional) Activate to drop the `uv run` prefix below
```

> **Note:** `uv sync` installs PyTorch from the CUDA 11.8 index pinned in `pyproject.toml`. Those wheels are built for **x86-64 Linux and Windows only** — on Apple Silicon or other ARM hosts there is no matching wheel and `uv sync` fails; install PyTorch for your platform from PyPI first. On x86-64 the CUDA build also runs fine without an NVIDIA GPU, it is just a large download. To use the smaller CPU build instead:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cpu
> source .venv/bin/activate    # then run the commands below as `python distract.py ...`
> ```
> Do **not** keep using the `uv run` prefix afterwards: it re-syncs the environment against `uv.lock` before each command and would silently reinstall the CUDA build. Use the activated venv, or `uv run --no-sync`.

> **Note:** NLP/fastText models are huge. They will automatically download on the very first run. If you are preparing a remote execution, see the [Offline Model Loading guide](https://github.com/mohamedsaid2710/Distractor_software-/wiki) on the Wiki.

### Basic Invocations

Run the pipeline using the `-i` (input), `-o` (output), and `-p` (parameter configuration) flags.

**English (en)**:
```bash
uv run python distract.py -i English_sample.txt -o output_en.txt -p params_en.txt -f delim
```

**German (de)**:
```bash
uv run python distract.py -i german_sample.txt -o output_de.txt -p params_de.txt -f delim
```

**Arabic (ar)**:
```bash
uv run python distract.py -i arabic_sample.txt -o output_ar.txt -p params_ar.txt -f delim
```

For quality validation of a generated file, run:
```bash
uv run python assess_output.py -i English_sample.txt -o output_en.txt -p params_en.txt --min-delta 0 --strict
```

> **Note:** `assess_output.py` reads both files and writes neither — `-i` is the original
> input sentences and `-o` is the already-generated output file to check. The report is
> printed to the terminal.

## $\color{SteelBlue}\text{Wiki Documentation}$

> 💡 **Why is this README not ENOUGH?**
> Because this software offers granular control over surprisal thresholds, BPE tokenization scaling, and semantic filtering logic, **all detailed documentation has been moved to the Wiki.**
> 
> Please consult the Wiki to understand how to format your parameters, tune the GPT-2 implausibility scores, or prepare files for Ibex Farm.

- 📖 **[Home & Architecture Overview](https://github.com/mohamedsaid2710/Distractor_software-/wiki)**
- 🧠 **[Models Map](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Models-Map)** (How GPT, fastText, Stanza, spaCy, and Farasa power the distractor brain)
- 🚀 **[Detailed Usage Guide](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Usage)** (Installation, generation, and assessment)
- ⚙️ **[Full Configuration Reference](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Config-Reference)** (Understand `min_delta`, `min_abs`, and `semantic_filter`)
- ➕ **[Adding a New Language](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Adding-a-Language)** (Step-by-step guide for new NLP models)
- 📦 **[Ibex Integration](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Ibex-Integration)** (Generating and using PCIbex outputs)
- 🛠️ **[Troubleshooting](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Troubleshooting)** (Common errors and solutions)
- 🗺️ **[Code Map](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Code-Map)** (File-by-file overview of the codebase)


>## $\color{Crimson}\text{CAUTION}$ 
>This software utilizes large language models trained on massive datasets. Consequently, generated distractors may occasionally contain offensive or harmful terms. Therefore, make sure to always add these to exclude_*.txt files. 

>## $\color{SeaGreen}\text{Important Note:}$ 
> The output is not guaranteed to be **error-free** or **linguistically** perfect. Manual review and verification are recommended before using generated stimuli in formal research.

## 📜 Citation

If you use this software in your research, please cite it as follows:

**APA (7th ed.):**
> Said, M. (2026). *Distractor_software: Automated maze-task distractor generation for English, German, and Arabic* (Version 1.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21570030

**BibTeX:**
```bibtex
@software{said_2026_distractor,
  author       = {Said, Mohammed},
  title        = {Distractor\_software: Automated maze-task distractor generation
                  for English, German, and Arabic},
  year         = {2026},
  month        = aug,
  publisher    = {Zenodo},
  version      = {1.0.2},
  doi          = {10.5281/zenodo.21570030},
  url          = {https://doi.org/10.5281/zenodo.21570030}
}
```

> The DOI above is the *concept* DOI, which always resolves to the most recent
> release and never changes. To cite one specific version instead, use that
> version's own DOI — `10.5281/zenodo.21570031` for v1.0.0, or the version DOI
> shown on the Zenodo record for any later release.


## $\color{DarkSlateGray}\text{License}$

MIT License. See [LICENSE](LICENSE) for details.
