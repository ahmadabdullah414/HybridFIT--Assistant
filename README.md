---
title: HybridFit Assistant
emoji: 💪
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# HybridFIT Assistant — Fit Bot model + hosted API

Fitness-coach chatbot for the Hybrid Fit app: a fine-tuned Qwen2.5-0.5B GGUF
language model plus small NLU helper models (language detection, intent
classification, entity gazetteers). The Flutter app talks to this repo two
ways:

1. **On-device (offline)** — the mobile app downloads `flutter/llm/*.gguf`
   directly and runs everything locally. See
   [`flutter/README.md`](flutter/README.md) for that integration guide.
2. **Hosted API** (`api/`) — a FastAPI service exposing the exact same
   pipeline (language detection -> intent classification -> safety gate ->
   generation) over HTTP, for when the app calls a remote chatbot instead
   of running the model on-device.

```
flutter/
  llm/fitbot-q4_k_m.gguf      # fine-tuned Qwen2.5-0.5B, int4 quantized (~374 MB, Git LFS)
  nlu/                        # language/intent TFLite models + vocabs + gazetteers
  README.md                   # on-device Flutter integration guide
api/
  main.py                     # FastAPI app — POST /chat, GET /health
  llm.py                      # llama-cpp-python generation (port of llama_chat_service.dart)
  nlu.py                      # TFLite language/intent inference (port of the *_service.dart files)
  safety.py                   # deterministic safety gate (port of safety_gate.dart)
  tokenizer.py                # typo-tolerant NLU tokenizer (port of nlu_tokenizer.dart)
  requirements.txt
Dockerfile                    # Hugging Face Spaces (Docker SDK) entry point
```

`fitbot-q4_k_m.gguf` is tracked with [Git LFS](https://git-lfs.github.com) —
run `git lfs pull` after cloning to fetch the actual model weights.

## Hosted API

### Deploying (Hugging Face Spaces, free)

Streamlit Cloud can only serve an interactive web page, not a callable JSON
API, so this is built as a plain Docker app instead and deployed on
**Hugging Face Spaces**, which has a free CPU tier and native Docker
support:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   — pick **Docker** as the SDK, any name, public or private, free CPU tier.
2. In the Space's **Settings -> Repository secrets**, add a secret named
   `API_KEY` with a long random value — this is what the Flutter app must
   send back on every request.
3. Push this repo's contents to the Space's own git remote (shown on the
   Space's page, looks like `https://huggingface.co/spaces/<user>/<space>`).
   The Space rebuilds automatically on every push. First build will take a
   while (downloading/installing `tensorflow-cpu` + `llama-cpp-python` and
   copying the ~374 MB model).
4. Once built, the API is live at `https://<user>-<space>.hf.space`.

### Calling it

```
POST /chat
X-API-Key: <the API_KEY secret>
Content-Type: application/json

{
  "message": "How much protein do I need?",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey! How can I help with your training today?"}
  ]
}
```

Response:
```json
{ "reply": "..." }
```

Stateless — the caller sends the recent turns with every request; the
server keeps nothing between calls (`history` mirrors what the Flutter app
already keeps locally via `ChatLocalStore`).

`GET /health` returns `{"status": "ok"}` once the models have finished
loading — use it as the Space's readiness check.
