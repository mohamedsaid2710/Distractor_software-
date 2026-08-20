import csv
import json


def ibex_line(tag, item_id, sentence, distractors):
    """Format one PCIbex Maze item line.

    Every text field goes through json.dumps, which is a superset of what a JS
    string literal needs.  The previous `.replace('"', chr(92)+chr(34))` escaped
    quotes but not backslashes, so a target containing one closed the literal
    early and PCIbex parsed the remainder of the line as code -- corrupting the
    whole item list, not just that item.  An embedded newline (which a quoted
    CSV field can carry) produced an unterminated literal with the same effect.

    Output is byte-identical to the old format for text without backslashes,
    newlines or control characters, i.e. for all ordinary stimuli.
    """
    s = json.dumps(sentence, ensure_ascii=False)
    d = json.dumps(distractors, ensure_ascii=False)
    t = json.dumps(tag, ensure_ascii=False)
    return f'[[{t}, {repr(item_id)}], "Maze", {{s:{s}, a:{d}}}], \n'


def save_delim(outfile, all_sentences):
    '''Saves results to a file in semicolon delimited format
    basically same as the original input with another column for distractor sentence
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none
    will write a semicolon delimited file with
    column 1 = "tag"/condition copied over from item_to_info (from input file)
    column 2 = item number
    column 3 = good sentence
    column 4 = string of distractor words in order.
    column 5 = string of labels in order. '''
    with open(outfile, 'w+', encoding='utf-8', newline="") as f:
        writer=csv.writer(f,delimiter=";")
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag,sentence.id,sentence.word_sentence,sentence.distractor_sentence,sentence.label_sentence])


def save_ibex(outfile, all_sentences):
    '''Saves results to a file in ibex format
    File contents can be copied into the items list of a maze_ibex file
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none'''
    with open(outfile, 'w+', encoding='utf-8', newline='') as f:
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                f.write(ibex_line(sentence.tag, sentence.id,
                                  sentence.word_sentence, sentence.distractor_sentence))



def append_results(outfile, sentence_set, outformat):
    '''Appends a single sentence_set's results to the specified outfile.
    Ensures that output is flushed so user can see it immediately.'''
    if outformat == "delim":
        with open(outfile, 'a', encoding='utf-8', newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag, sentence.id, sentence.word_sentence, 
                                 sentence.distractor_sentence, sentence.label_sentence])
            f.flush()
    elif outformat == "ibex":
        with open(outfile, 'a', encoding='utf-8', newline='') as f:
            for sentence in sentence_set.sentences:
                f.write(ibex_line(sentence.tag, sentence.id,
                                  sentence.word_sentence, sentence.distractor_sentence))
            f.flush()

REPORT_COLUMNS = [
    "item_id", "label", "position", "target", "distractor",
    "target_len", "distractor_len", "len_delta", "len_ok",
    "target_zipf", "distractor_zipf", "zipf_delta",
    "band_min_zipf", "band_max_zipf", "in_band",
    "target_surprisal", "achieved_surprisal", "threshold", "meets_threshold",
    "pool_size", "freq_widen_steps", "pos_filter", "used_fallback",
    "relaxation", "is_placeholder",
]


def report_path(outfile):
    """Sidecar path for *outfile*: 'out.txt' -> 'out.report.csv'."""
    base, _sep, _ext = outfile.rpartition(".")
    return (base or outfile) + ".report.csv"


def init_report(outfile):
    """Create the sidecar with its header row, replacing any previous one."""
    with open(report_path(outfile), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(REPORT_COLUMNS)


def append_report(outfile, sentence_set, lang, len_tolerance=0):
    """Append one row per distractor position for this item.

    The delim and ibex files carry only the chosen word, so a distractor that
    came out of the desperation fallback looks exactly like one that met every
    criterion.  This sidecar is what lets a paper state how many items actually
    met them, and lets a borderline item be found and replaced.
    """
    try:
        import wordfreq as _wf
    except ImportError:
        _wf = None

    def zipf(word):
        if not _wf or not word:
            return None
        form = word.lower()
        if lang == 'ar':
            from wordfreq_distractor import strip_arabic_diacritics
            form = strip_arabic_diacritics(form)
        return round(_wf.zipf_frequency(form, lang), 3)

    from utils import strip_punct

    rows = []
    seen = set()
    for sentence in sentence_set.sentences:
        for i, lab in enumerate(sentence.labels):
            if i == 0 or lab not in sentence_set.labels:
                continue
            if lab in seen:      # one distractor per label; report it once
                continue
            seen.add(lab)
            label = sentence_set.labels[lab]
            rep = getattr(label, "report", {}) or {}
            target = strip_punct(sentence.words[i])
            dist = strip_punct(label.distractor or "")
            tz, dz = zipf(target), zipf(dist)
            lo, hi = rep.get("band_min_zipf"), rep.get("band_max_zipf")
            achieved, thresh = rep.get("achieved_surprisal"), rep.get("threshold")
            rows.append([
                sentence_set.id, lab, i, target, dist,
                len(target), len(dist), len(dist) - len(target),
                abs(len(dist) - len(target)) <= len_tolerance,
                tz, dz, (None if tz is None or dz is None else round(dz - tz, 3)),
                lo, hi,
                (None if dz is None or lo is None else lo <= dz <= hi),
                rep.get("target_surprisal"), achieved, thresh,
                (None if achieved is None or thresh is None else achieved >= thresh),
                rep.get("pool_size"), rep.get("freq_widen_steps"),
                rep.get("pos_filter"), rep.get("used_fallback"),
                rep.get("relaxation"), rep.get("is_placeholder"),
            ])

    if rows:
        with open(report_path(outfile), "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
            f.flush()


def summarize_report(outfile):
    """Print a compliance summary of the sidecar, and return it as a dict.

    These are the numbers a methods section needs: how many positions actually
    met the length, frequency and surprisal criteria, and how many were
    resolved by a relaxation.  Split by target frequency because the two groups
    fail in opposite directions -- content words are frequency-matched but can
    miss the surprisal floor, while high-frequency function words clear the
    floor easily and cannot be frequency-matched, there being too few short
    words at that frequency.
    """
    path = report_path(outfile)
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    if not rows:
        return {}

    def truthy(rows_, field):
        return sum(1 for r in rows_ if r.get(field) == "True")

    def block(label, rows_):
        if not rows_:
            return
        n = len(rows_)
        print(f"    {label:<22} n={n:<4} "
              f"length {truthy(rows_, 'len_ok')}/{n}  "
              f"frequency {truthy(rows_, 'in_band')}/{n}  "
              f"surprisal {truthy(rows_, 'meets_threshold')}/{n}  "
              f"relaxed {truthy(rows_, 'used_fallback')}/{n}")

    def target_zipf(r):
        try:
            return float(r["target_zipf"])
        except (TypeError, ValueError):
            return 0.0

    print("\n>>> QUALITY REPORT  (" + path + ")")
    block("all positions", rows)
    block("content (Zipf<=6)", [r for r in rows if target_zipf(r) <= 6])
    block("function (Zipf>6)", [r for r in rows if target_zipf(r) > 6])

    placeholders = truthy(rows, "is_placeholder")
    if placeholders:
        print(f"    WARNING: {placeholders} position(s) emitted a placeholder "
              f"rather than a real word.")
    widened = sum(1 for r in rows if r.get("freq_widen_steps") not in ("0", "", None))
    if widened:
        print(f"    {widened}/{len(rows)} position(s) needed the frequency band "
              f"widened to fill the pool.")
    return {
        "n": len(rows),
        "exact_length": sum(1 for r in rows if r["len_delta"] == "0"),
        "len_ok": truthy(rows, "len_ok"),
        "in_band": truthy(rows, "in_band"),
        "met_threshold": truthy(rows, "meets_threshold"),
        "used_fallback": truthy(rows, "used_fallback"),
        "placeholders": placeholders,
    }
