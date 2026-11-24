"""A stand-in language model for end-to-end tests: no weights, no GPU.

Surprisal is a deterministic pseudo-random number per (context, word), the
same in every process, so a run is reproducible and every code path that
depends on the scores (passing, below threshold, fallback) gets exercised.
"""
import hashlib


class FakeScorer:
    LANG_NAME = None          # not checked against `language`
    model_batch_size = 64

    def __init__(self, params=None):
        self.params = params or {}

    def empty_sentence(self):
        return []

    def update(self, hidden, word):
        return list(hidden) + [word], None

    def get_surprisal_from_hidden(self, hidden, word):
        key = ("|".join(hidden[-3:]) + "#" + word).encode("utf-8")
        return int(hashlib.md5(key).hexdigest()[:6], 16) % 4000 / 100.0   # 0-40 bits

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=None):
        return [self.get_surprisal_from_hidden(hidden, w) for w in words]
