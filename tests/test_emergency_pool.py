"""The last-resort emergency pools draw only from the vocabulary, never from
lexicon words the vocabulary leaves out (Arabic clitics, excluded words)."""
from maze_automate.candidates import wordfreq_dict
from maze_automate.candidates import distractor


def test_emergency_pool_is_limited_to_the_vocabulary():
    d = wordfreq_dict({})
    d.words = [distractor("كتاب", 10.0)]
    d.pos_cache = {}
    d._build_length_index()
    d.nouns_by_len[4].update({"كتاب", "بكتب"})       # بكتب: a clitic form
    assert d.get_emergency_pool(4, is_noun=True) == ["كتاب"]
