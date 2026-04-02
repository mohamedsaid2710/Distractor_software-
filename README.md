# Distractor Software

**Maze-style distractor stimulus generator for psycholinguistic experiments in English, German, and Arabic.**

Built on Transformer-based language models (GPT-2), the pipeline selects real-word distractors that are contextually implausible while matching the target word in length and frequency range.

## Attribution

Based on the original [Maze repository](https://github.com/vboyce/Maze) by Victoria Boyce. This implementation has been extensively adapted and now features distinct Transformer models, automated language-specific NLP tools, GPU batch processing, semantic embeddings, and an interactive config-tuning workflow.

## Overview & Capabilities

- **Supported Languages:** 
  - English (`gpt2-medium` via spaCy `en_core_web_lg`)
  - German (`benjamin/gerpt2` via Stanza neural tagging)
  - Arabic (`aubmindlab/aragpt2-medium` via Farasa `farasapy`)
- **Generation Modes:** Choose between threshold-first (Mode A) or maximum-implausibility scoring (Mode B).
- **Linguistic Precision:**
  - Length and ZIPF frequency matching.
  - Optional **fastText Semantic Filtering** to reject words from similar domains (e.g., avoiding "Apple" -> "Orange").
  - **Part-Of-Speech Matching** to ensure natural grammar structure (Verbs match Verbs, Nouns match Nouns).
- **Fast GPU Processing:** Batch-optimized surprisal scoring scales automatically to available hardware.
- **Output Formats:** Standard delimited tables or ready-to-deploy PCIbex lines (`ibexify`).

## Quick Start

It is **highly recommended** to run this software on a GPU-enabled environment (like Google Colab or an academic computing cluster).

```bash
# Clone the repository
git clone https://github.com/mohamedsaid2710/Distractor_software-.git
cd Distractor_software-

# Install the strict dependencies (Requires Python 3.12+)
pip install -r requirements.txt
```

> **Note:** NLP/fastText models are huge. They will automatically download on the very first run. If you are preparing a remote execution, see the [Offline Model Loading guide](https://github.com/mohamedsaid2710/Distractor_software-/wiki) on the Wiki.

### Basic Invocations

Run the pipeline using the `-i` (input), `-o` (output), and `-p` (parameter configuration) flags.

**English (EN)**:
```bash
python distract.py -i English_sample.txt -o output_en.txt -p params_en.txt -f delim
```

**German (DE)**:
```bash
python distract.py -i german_sample.txt -o output_de.txt -p params_de.txt -f delim
```

**Arabic (AR)**:
```bash
python distract.py -i arabic_sample.txt -o output_ar.txt -p params_ar.txt -f delim
```

For quality validation of a generated file, run:
```bash
python assess_output.py -i output_en.txt -o output_en_assessed.txt -p params_en.txt --min-delta 0 --strict
```

## Documentation & Wiki

> 💡 **Why is this README not ENOUGH?**
> Because this software offers granular control over surprisal thresholds, BPE tokenization scaling, and semantic filtering logic, **all detailed documentation has been moved to the Wiki.**
> 
> Please consult the Wiki to understand how to format your parameters, tune the GPT-2 implausibility scores, or prepare files for Ibex Farm.

- 📖 **[Home & Architecture Overview](https://github.com/mohamedsaid2710/Distractor_software-/wiki)**
- 🚀 **[Detailed Usage Guide](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Usage)** (Installation, generation, and assessment)
- ⚙️ **[Full Configuration Reference](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Config-Reference)** (Understand `min_delta`, `min_abs`, and `semantic_filter`)
- ➕ **[Adding a New Language](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Adding-a-Language)** (Step-by-step guide for new NLP models)
- 📦 **[Ibex Integration](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Ibex-Integration)** (Generating and using PCIbex outputs)
- 🛠️ **[Troubleshooting](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Troubleshooting)** (Common errors and solutions)
- 🗺️ **[Code Map](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Code-Map)** (File-by-file overview of the codebase)

## License

MIT License. See [LICENSE](LICENSE) for details.
