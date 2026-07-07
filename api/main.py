"""HybridFit Assistant — hosted Fit Bot API.

Mirrors the Flutter app's offline pipeline exactly (see
lib/features/chatbot/services/fitbot_chatbot_service.dart): language
detection -> intent classification -> deterministic safety gate -> (skip
straight to a fixed redirect if triggered) -> LLM generation.

Stateless: the caller sends the recent conversation history with every
request; nothing is retained server-side between calls.
"""

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from llm import ChatTurn, LlamaChatService
from nlu import IntentClassifier, LanguageDetector
from safety import BOT_IDENTITY_REDIRECT, is_false_identity_claim, safety_check

_BASE = os.path.dirname(os.path.abspath(__file__))
_NLU_DIR = os.path.join(_BASE, "..", "flutter", "nlu")
_LLM_PATH = os.path.join(_BASE, "..", "flutter", "llm", "fitbot-q4_k_m.gguf")

API_KEY = os.environ.get("API_KEY")  # set as a Hugging Face Space "Repository secret"

app = FastAPI(title="HybridFit Assistant API")

_language_detector: LanguageDetector | None = None
_intent_classifier: IntentClassifier | None = None
_chat_service: LlamaChatService | None = None


@app.on_event("startup")
def _load_models() -> None:
    global _language_detector, _intent_classifier, _chat_service
    _language_detector = LanguageDetector(
        model_path=os.path.join(_NLU_DIR, "language_detector.tflite"),
        vocab_path=os.path.join(_NLU_DIR, "language_vocab.json"),
    )
    _intent_classifier = IntentClassifier(
        model_path=os.path.join(_NLU_DIR, "intent_classifier.tflite"),
        vocab_path=os.path.join(_NLU_DIR, "intent_vocab.json"),
        labels_path=os.path.join(_NLU_DIR, "intent_labels.json"),
    )
    _chat_service = LlamaChatService(model_path=_LLM_PATH)


class HistoryTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryTurn] = []


class ChatResponse(BaseModel):
    reply: str


def _require_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        # Misconfigured deployment — fail loudly rather than silently
        # running open to the public.
        raise HTTPException(status_code=500, detail="Server is missing its API_KEY configuration.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    _require_api_key(x_api_key)

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty.")

    is_roman_urdu = _language_detector.is_roman_urdu(message)
    intent = _intent_classifier.classify(message)

    redirect = safety_check(message, is_roman_urdu=is_roman_urdu, intent=intent)
    if redirect is not None:
        return ChatResponse(reply=redirect)

    history = [ChatTurn(t.role, t.content) for t in request.history]
    reply = _chat_service.generate_reply(history, message)
    # Defense in depth: bot_identity is already hard-gated above, but this
    # small fine-tuned model has volunteered "I'm Junaid" unprompted even on
    # a plain "hello" (intent misclassified as something else) — catch it on
    # the actual output too rather than trusting the intent gate alone.
    if is_false_identity_claim(reply):
        reply = BOT_IDENTITY_REDIRECT
    return ChatResponse(reply=reply)
