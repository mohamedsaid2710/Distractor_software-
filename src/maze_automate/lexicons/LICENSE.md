# Licences of the POS lexicons

The code of this project is under the MIT licence (`LICENSE` at the repository root).
**That licence does not cover the three lexicon files in this folder.** Each one is
derived from a published lexical resource and is distributed under that resource's
terms. Anyone who uses, copies or redistributes a lexicon file must follow those terms
and cite the source.

| File | Derived from | Terms |
|---|---|---|
| `en.json` | SUBTLEX-US with part-of-speech information (Brysbaert, New, & Keuleers, 2012), https://osf.io/djpqz/ | The authors state no licence. The file is provided for **non-commercial research use only**, under whatever terms the SUBTLEX-US authors set. For any other use, ask the authors. |
| `de.json` | dlexDB annotated lexical data (Heister et al., 2011), https://doi.org/10.5281/zenodo.15264903 | **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/). Adapted: only each word's dominant part of speech and its share are kept; spellings with ß restored. |
| `ar.json` | CAMeL Tools morphological database calima-msa-r13 (Obeid et al., 2020), derived from Aramorph 1.2.1 | **GNU GPL version 2** (`GPL-2.0.txt` in this folder). Copyright (c) 2002 QAMUS LLC, (c) 2002 Trustees of the University of Pennsylvania, portions (c) 2006 Jon Dehdari. |

The lexicons contain no text from the sources beyond word forms, part-of-speech labels
and proportions computed from them. They are provided without any warranty. Full
references are in the README and in each file's `_meta` block.
