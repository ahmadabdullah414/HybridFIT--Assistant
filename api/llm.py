"""Offline LLM generation — port of the Dart `llama_chat_service.dart`.

Unlike the on-device version, this server is stateless: the caller (the
Flutter app) sends the recent conversation turns with every request rather
than the server keeping its own rolling history — simpler and safer for a
shared server handling multiple members at once.
"""

import re

from llama_cpp import Llama

# System prompt the model was fine-tuned against — must be reproduced
# exactly, not paraphrased, since the LLM's safety/scope behavior for
# borderline replies depends on matching its training conditioning. Says
# "always reply in English regardless of input language" because Roman
# Urdu input never reaches here at all — the safety gate hard-redirects it
# before generation (see safety.py) — so this is just reinforcement, not
# translation logic the model needs to perform.
SYSTEM_PROMPT = """You are HybridFit Assistant, a professional, friendly, and concise AI
fitness coach built by Muhammad Junaid for the HybridFit app. You are
an AI, not Muhammad Junaid himself. You help with workouts, exercise
form, strength training, cardio, weight loss, muscle gain, nutrition,
recovery, and healthy lifestyle questions. Always reply in English,
regardless of what language the user writes in. Keep answers short and
to the point unless the question needs step-by-step detail. Never
discuss steroids, prescription medication, dosages, or diagnose
medical conditions -- redirect those to a licensed professional. Stay
strictly within the fitness/nutrition/wellness domain."""

# Sampling occasionally (rare, but confirmed in testing) produces a stray
# run of non-Latin script -- e.g. Chinese characters bleeding into an
# otherwise-English answer. English-only output is a hard product
# requirement, so this is treated as a bad sample and regenerated rather
# than shown to the user.
_NON_LATIN_LEAK = re.compile(r"[一-鿿぀-ヿ가-퟿؀-ۿऀ-ॿ]")

MAX_GENERATION_RETRIES = 2  # two retries (three attempts total)
MAX_HISTORY_MESSAGES = 6
# Lower than the on-device Dart version's 512 — that runs natively with no
# outer timeout, but this server sits behind Hugging Face's own gateway,
# which has an upstream timeout. An open-ended prompt like a plain "hello"
# was observed generating for 80+ seconds on the free CPU tier and getting
# cut off by the gateway (HTTP 500) before finishing. Capping the budget
# here bounds worst-case latency instead of relying on the model to stop
# itself early, which it doesn't reliably do.
MAX_TOKENS = 200


class ChatTurn:
    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class LlamaChatService:
    """Wraps `llama_cpp.Llama` with the exact ChatML framing Fit Bot was
    fine-tuned on. One instance loaded once at process startup and shared
    across requests."""

    def __init__(self, model_path: str):
        # n_threads pinned to match Hugging Face's "cpu-basic" free tier (2
        # vCPUs) explicitly, rather than trusting auto-detection — a
        # container can report the host's full core count even when its
        # actual CPU quota is much smaller, causing thread oversubscription
        # that slows generation down instead of speeding it up.
        self._llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=0, n_threads=2, verbose=False)

    def _build_prompt(self, history: list[ChatTurn], user_message: str) -> str:
        parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
        for turn in history[-MAX_HISTORY_MESSAGES:]:
            parts.append(f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    def _generate_once(self, prompt: str) -> str:
        output = self._llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.45,
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["<|im_end|>"],
        )
        return output["choices"][0]["text"].strip()

    def generate_reply(self, history: list[ChatTurn], user_message: str) -> str:
        prompt = self._build_prompt(history, user_message)
        reply = ""
        for _ in range(MAX_GENERATION_RETRIES + 1):
            reply = self._generate_once(prompt)
            if not _NON_LATIN_LEAK.search(reply):
                break
        return reply
