"""Tokenizer for the NLU TFLite models — port of the Dart
`nlu_tokenizer.dart`, which itself mirrors the original training pipeline's
tokenizer. Keep this in sync with that file if either changes."""

import json
import re

_TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


def build_length_buckets(vocab: dict) -> dict[int, list[str]]:
    """Vocab words bucketed by length, computed once per vocab at load time
    so a fuzzy-correction lookup only compares a token against same-length
    (+/-1) vocab words instead of scanning the whole vocab on every call."""
    buckets: dict[int, list[str]] = {}
    for word in vocab:
        buckets.setdefault(len(word), []).append(word)
    return buckets


def _is_edit_distance_one(a: str, b: str) -> bool:
    """True if a and b differ by exactly one insertion, deletion or
    substitution (classic edit distance == 1, no transposition)."""
    if a == b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False

    if len(a) == len(b):
        diffs = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diffs += 1
                if diffs > 1:
                    return False
        return diffs == 1

    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            i += 1
    return True


def _nearest_vocab_word(token: str, buckets: dict[int, list[str]]) -> str | None:
    """Fuzzy-corrects an out-of-vocabulary token to the nearest vocab word
    exactly one edit away, only for tokens of 4+ characters (shorter words
    produce too many false-positive corrections)."""
    if len(token) < 4:
        return None
    for length in (len(token), len(token) - 1, len(token) + 1):
        for candidate in buckets.get(length, ()):
            if _is_edit_distance_one(token, candidate):
                return candidate
    return None


def _resolve_token(token: str, vocab: dict, buckets: dict[int, list[str]]) -> int:
    exact = vocab.get(token)
    if exact is not None:
        return exact
    corrected = _nearest_vocab_word(token, buckets)
    return vocab[corrected] if corrected is not None else 1  # 1 = <unk>


def encode_tokens(text: str, vocab: dict, buckets: dict[int, list[str]], max_len: int) -> list[int]:
    """Encodes text into a fixed-length max_len array of vocab ids:
    lowercase, split on runs of letters/apostrophes, unknown words get an
    edit-distance-1 fuzzy-corrected match if one exists (e.g. "wrkout" ->
    "workout") else 1 (<unk>), right-padded with 0 (<pad>)."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    ids = [_resolve_token(t, vocab, buckets) for t in tokens[:max_len]]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def load_vocab(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
