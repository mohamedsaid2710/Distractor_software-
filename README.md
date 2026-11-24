# Distractor Software

Generates distractors for the **Maze task** in **English, German and Arabic**.
For every word of a sentence it picks a real word of similar length and frequency
that a language model (GPT-2) finds very unlikely in that context.

Based on the [Maze repository](https://github.com/vboyce/Maze) by Victoria Boyce.
Special thanks to [Titus von der Malsburg](https://tmalsburg.github.io/) for his
help and feedback.

## Install

A GPU (e.g. Google Colab) is strongly recommended.

```bash
git clone https://github.com/mohamedsaid2710/Distractor_software-.git
cd Distractor_software-
uv sync
```

## Run

```bash
uv run python run_pipeline.py -i examples/english_sample.txt -o output_en.txt -p config/params_en.txt
```

For German or Arabic use `config/params_de.txt` or `config/params_ar.txt`.
Add `-f ibex` for PCIbex-ready lines.

**Input:** one sentence per line, `tag;item_id;sentence`. Sentences with the same
item id get the same distractors.

```
exp_a;1;The cat sat on the mat.
exp_b;1;The cats sat on the mat.
```

**Output:** `output_en.txt` with the distractors, and `output_en.report.csv`,
which shows for every word whether length, frequency and surprisal were met.

**Settings** are in `config/`: one params file and one exclude list per language.
Add any distractor you do not want to the exclude list.

## Repository layout

| Folder | Contents |
|---|---|
| `config/` | params files and exclude lists: the only files you normally edit |
| `examples/` | sample input files |
| `tools/` | `download_models.py` (fetch models before a Colab or cluster run), `assess_output.py`, `ibexify.py`, `build_lexicon.py` |
| `src/maze_automate/` | the code |

## Documentation

The [Wiki](https://github.com/mohamedsaid2710/Distractor_software-/wiki) explains
every setting, the selection procedure, the models and lexicons, and troubleshooting.

> **Tune the settings for your stimuli.** The params files in `config/` are a
> starting point, not a fit for every experiment: check the quality report and
> adjust the thresholds, frequency band, POS matching and proper-noun exclusion
> (`exclude_propn_candidates`) to your materials. See the
> [Config Reference](https://github.com/mohamedsaid2710/Distractor_software-/wiki/Config-Reference).

> **Actively maintained.** The software is updated regularly. Before generating
> stimuli, make sure you are on the latest version (`git pull`, then `uv sync`).

> **Caution:** the language models are trained on web text, so a distractor can
> occasionally be offensive. Always review the output before using it.

## Citation

> Said, M. (2026). *Distractor_software: Automated maze-task distractor generation for English, German, and Arabic* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21570030

This DOI always points to the latest release; each release also has its own DOI
on the Zenodo page.

The part-of-speech lexicons are built from:

- Brysbaert, M., New, B., & Keuleers, E. (2012). Adding part-of-speech information to the SUBTLEX-US word frequencies. *Behavior Research Methods, 44*(4), 991–997. https://doi.org/10.3758/s13428-012-0190-4
- Heister, J., Würzner, K.-M., Bubenzer, J., Pohl, E., Hanneforth, T., Geyken, A., & Kliegl, R. (2011). dlexDB – eine lexikalische Datenbank für die psychologische und linguistische Forschung. *Psychologische Rundschau, 62*(1), 10–20. https://doi.org/10.1026/0033-3042/a000029
- Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., Alhafni, B., Inoue, G., Eryani, F., Erdmann, A., & Habash, N. (2020). CAMeL Tools: An open source Python toolkit for Arabic natural language processing. In *Proceedings of the 12th Language Resources and Evaluation Conference* (pp. 7022–7032). European Language Resources Association. https://aclanthology.org/2020.lrec-1.868

## License

The code is MIT ([LICENSE](LICENSE)). **The POS lexicons are not**: each follows the
terms of its source (English: SUBTLEX-US, non-commercial research use; German: CC BY-SA
4.0; Arabic: GPL-2.0). See
[src/maze_automate/lexicons/LICENSE.md](src/maze_automate/lexicons/LICENSE.md).
