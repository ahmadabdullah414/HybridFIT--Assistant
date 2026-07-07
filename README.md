# HybridFIT Assistant — Fit Bot model export

Offline fitness-coach chatbot for the Hybrid Fit app: a fine-tuned Qwen2.5-0.5B
GGUF language model plus small NLU helper models (language detection, intent
classification, entity gazetteers).

See [`flutter/README.md`](flutter/README.md) for the full Flutter integration
guide (pipeline order, safety-gate logic, system prompt, tokenizer details).

```
flutter/
  llm/
    fitbot-q4_k_m.gguf        # fine-tuned Qwen2.5-0.5B, int4 quantized (~374 MB, Git LFS)
  nlu/
    intent_classifier.tflite  # 44-way intent classifier
    intent_vocab.json
    intent_labels.json
    language_detector.tflite  # English vs Roman Urdu
    language_vocab.json
    gazetteers.json           # exercise/muscle/equipment entity lists
  README.md                   # full integration guide
```

`fitbot-q4_k_m.gguf` is tracked with [Git LFS](https://git-lfs.github.com) —
run `git lfs pull` after cloning to fetch the actual model weights.
