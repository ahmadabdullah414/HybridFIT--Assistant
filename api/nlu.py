"""Language detection + intent classification — port of the Dart
`language_detector_service.dart` / `intent_classifier_service.dart`.

Uses TensorFlow's bundled `tf.lite.Interpreter` (rather than the standalone
`tflite-runtime` package, which has spotty prebuilt-wheel coverage) so this
installs reliably on any platform via `pip install tensorflow-cpu`.
"""

import json

import numpy as np
import tensorflow as tf

from tokenizer import build_length_buckets, encode_tokens, load_vocab


def _run(interpreter: tf.lite.Interpreter, ids: list[int]) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    dtype = input_details["dtype"]
    arr = np.array([ids], dtype=np.float32 if dtype == np.float32 else np.int32).astype(dtype)

    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


class LanguageDetector:
    """English-vs-Roman-Urdu detector. Single sigmoid output; > 0.5 means
    Roman Urdu, matching the training pipeline."""

    MAX_LEN = 32

    def __init__(self, model_path: str, vocab_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.vocab = load_vocab(vocab_path)
        self.buckets = build_length_buckets(self.vocab)

    def is_roman_urdu(self, text: str) -> bool:
        ids = encode_tokens(text, self.vocab, self.buckets, self.MAX_LEN)
        output = _run(self.interpreter, ids)
        return float(output[0]) > 0.5


class IntentClassifier:
    """44-way intent classifier. Argmax over the softmax output indexes
    into intent_labels.json. Used for topic routing (specifically the
    off_topic_redirect class) — never trusted alone for medical/abusive
    safety, which is deterministic keyword/regex instead."""

    MAX_LEN = 40

    def __init__(self, model_path: str, vocab_path: str, labels_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.vocab = load_vocab(vocab_path)
        self.buckets = build_length_buckets(self.vocab)
        with open(labels_path, encoding="utf-8") as f:
            self.labels = json.load(f)

    def classify(self, text: str) -> str:
        ids = encode_tokens(text, self.vocab, self.buckets, self.MAX_LEN)
        output = _run(self.interpreter, ids)
        return self.labels[int(np.argmax(output))]
