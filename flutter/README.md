# Fit Bot — Flutter Integration Guide

This folder contains everything needed to run Fit Bot fully offline in a
Flutter app:

```
export/flutter/
  llm/
    fitbot-q4_k_m.gguf       # fine-tuned Qwen2.5-0.5B, int4 quantized (~374 MB)
  nlu/
    intent_classifier.tflite  # 44-way intent classifier (~660 KB), includes founder_info/brand_info/bot_identity
    intent_vocab.json         # word -> id vocab for the intent model
    intent_labels.json        # id -> intent name list
    language_detector.tflite  # English vs Roman Urdu (~130 KB)
    language_vocab.json       # word -> id vocab for the language model
    gazetteers.json           # exercise/muscle/equipment name lists (entity extraction)
```

Total package size: **~380 MB**, all offline, no network calls needed at
runtime.

## 1. Package choice: `llama_cpp_dart`

Two Flutter/Dart bindings for llama.cpp exist. We recommend **`llama_cpp_dart`**:

| | `llama_cpp_dart` | `fllama` |
|---|---|---|
| Platforms | Android, iOS, macOS, Linux, Windows | Android, iOS, HarmonyOS |
| Maintenance | v0.2.2 stable (6 months ago), active dev prerelease | v0.0.1, no publish in 19 months |
| Recommendation | **Use this** | Fallback only if you hit a `llama_cpp_dart` platform gap |

Add to `pubspec.yaml`:
```yaml
dependencies:
  llama_cpp_dart: ^0.2.2
  tflite_flutter: ^0.11.0   # for the intent/language classifiers
```

`llama_cpp_dart` needs the native `libllama` shared library for your target
platform(s) alongside the Dart package — follow that package's own
platform build/bundling instructions (it documents Android/iOS/desktop
build steps) since this changes as the package evolves.

## 2. Bundling the model files

**Don't embed the 380MB directly in the app bundle** — app stores penalize
large initial downloads and some have hard size caps. Instead:

1. Ship the small NLU files (`nlu/*.tflite`, `*.json`, ~1MB total) as normal
   Flutter assets (`pubspec.yaml` -> `flutter: assets:`).
2. On first app launch, download `fitbot-q4_k_m.gguf` from your own
   CDN/object storage into the app's documents directory
   (`path_provider`'s `getApplicationDocumentsDirectory()`), show a
   progress bar, and cache it there for all future offline use.
3. Verify the downloaded file's checksum before use (compute a SHA-256 once
   and hardcode it, compare after download).

If you'd rather bundle it directly (e.g. an internal/enterprise app where
download size doesn't matter), you can add it as a Flutter asset instead —
just expect a slow first `flutter build` and a large app binary.

## 3. Runtime pipeline

Re-implement this same order in Dart (mirrors `fitbot/pipeline.py`):

```
user text
  -> language_detector.tflite   (English vs Roman Urdu)
  -> intent_classifier.tflite   (44 intents; also used for entity/topic routing)
  -> regex/keyword safety gate  (MUST run before the LLM, see below)
       -> if triggered: return the fixed redirect string, skip the LLM entirely
  -> gazetteer entity extraction (optional, for personalization -- exercise/
     muscle/equipment/goal/duration mentions; see gazetteers.json)
  -> llama_cpp_dart generate()   (only reached if the safety gate passed)
  -> show reply, append to conversation history (last ~6 turns)
```

**Why the safety gate is a separate hard-coded step and not left to the
LLM**: during evaluation, the fine-tuned model's own judgment about what to
refuse was inconsistent -- it sometimes over-refused ordinary fitness
questions and sometimes answered a steroids question directly instead of
redirecting. A deterministic keyword/regex check that runs *before*
generation and can short-circuit it entirely is what actually guarantees
the safety behavior. Port the logic in `fitbot/nlu/safety.py` to Dart
directly -- it's plain keyword/regex matching, no model needed. Note the
Roman Urdu check runs FIRST, before anything else, and skips straight to a
fixed redirect -- the bot no longer attempts to understand or answer Roman
Urdu at all (see §7):

**Normalize before matching against ANY keyword list below.** Adversarial
testing found that spacing out letters ("s t e r o i d s") and leetspeak
substitution ("stup1d", "garb@ge") both slipped straight past plain keyword
matching and reached generation. Port `normalize_for_matching()` from
`fitbot/nlu/safety.py`:

```dart
String normalizeForMatching(String text) {
  var low = text.toLowerCase();
  // Collapse 3+ single letters separated by single spaces: "s t e r o i d s" -> "steroids"
  low = low.replaceAllMapped(
    RegExp(r'\b(?:[a-zA-Z]\s+){2,}[a-zA-Z]\b'),
    (m) => m.group(0)!.replaceAll(' ', ''),
  );
  // Common leetspeak substitutions
  const map = {'1': 'i', '3': 'e', '4': 'a', '0': 'o', '@': 'a', r'$': 's'};
  return low.split('').map((c) => map[c] ?? c).join();
}
```
Every check below (`isAbusive`, `isMedical`, off-topic, personal-info,
contact) must run against `normalizeForMatching(text)`, not the raw string.

```dart
final medicalKeywords = ['steroid', 'anabolic', 'sarms', ... ];
final offTopicKeywords = [
  'stock market', 'election', 'my taxes', 'python homework', 'my code',
  "what's the weather", 'weather like', 'my homework', 'unrelated question',
  'not fitness related', 'off topic', 'ignore previous instructions',
  'ignore all previous', 'ignore your instructions', ...
];
// abusive: regex patterns for obfuscated profanity, see safety.py

const ownerContactEmail = "hybridstrengthnfitness@gmail.com";

// Anything asking for a private detail we were never given. Checked before
// the contact-email check, and kept deliberately broad -- the model must
// NEVER be allowed to invent a phone number, address, or bank/financial
// detail for the owner just because it sounds plausible. Found via testing:
// asked about Junaid's "neighborhood", the model confidently invented a
// wife, kids, and a second fake company -- family/personal-life questions
// need to be caught here too, not just literal "address"/"phone number".
final personalInfoKeywords = [
  'phone number', 'cell number', 'personal cell', 'mobile number', 'whatsapp number',
  'contact number', 'personal number', 'home address', 'personal address',
  'residential address', 'where does junaid live', 'where do you live',
  'where he lives', 'what neighborhood', 'what city does', 'which city does',
  'bank account', 'bank details', 'iban', 'credit card', 'debit card',
  'cnic', 'national id', 'id card number', 'social security', 'passport number',
  'his wife', 'his kids', 'his children', 'his family', 'is he married',
  'is junaid married', "junaid's wife", "junaid's family", "junaid's kids",
];

// Contact/email requests get a fixed, exact answer -- never generated by
// the LLM, since a single wrong character breaks an email address.
final contactKeywords = [
  "owner's email", 'owner email', "junaid's email", 'junaid email',
  'contact junaid', 'contact the owner', 'contact hybridfit', 'hybridfit email',
  "hybridfit's email", 'how can i contact', 'how do i contact', 'get in touch with',
  'reach out to junaid', 'reach junaid', 'contact information', 'contact info',
  'email address', 'official email', 'your email', 'what is your email',
];

const englishOnlyRedirect =
    "I can only understand English right now. Please type your message in "
    "English and I'll be happy to help.";
const personalInfoRedirect =
    "I don't have that information to share. For anything else, you can "
    "reach out at $ownerContactEmail.";
const contactRedirect =
    "You can reach Muhammad Junaid and the HybridFit team at $ownerContactEmail.";

// IMPORTANT: off-topic detection needs the intent classifier's prediction,
// not just a keyword check. An earlier version required an explicit
// fitness keyword on *every* message with no other signal, and broke
// ordinary multi-turn conversation -- short follow-ups like "I'm a
// complete beginner" or "what about creatine" don't restate the topic and
// got wrongly blocked. The fix: only treat a message as off-topic if it
// matches the explicit blocklist above, OR the intent classifier itself
// predicts the dedicated `off_topic_redirect` class. Don't add a
// requirement for a fitness keyword to be present -- that's the bug.
String? safetyCheck(String text, String language, String intent) {
  if (language == 'roman_urdu') return englishOnlyRedirect;
  final low = normalizeForMatching(text); // NOT text.toLowerCase() -- see above
  if (personalInfoKeywords.any((kw) => low.contains(kw))) return personalInfoRedirect;
  if (contactKeywords.any((kw) => low.contains(kw))) return contactRedirect;
  if (isAbusive(low)) return abusiveRedirectEn;   // profanity regex also runs on normalized text
  if (isMedical(low)) return medicalRedirectEn;
  final offTopic = offTopicKeywords.any((kw) => low.contains(kw))
      || intent == 'off_topic_redirect';
  if (offTopic) return offTopicRedirectEn;
  return null; // safe to proceed to the LLM
}
```

**On the owner's contact email**: `hybridstrengthnfitness@gmail.com` is answered as a fixed string via this gate, not generated by the LLM -- do not add it to any training/prompt data expecting the model to recall it, since a small model can occasionally garble an exact string like an email address. The gate above is the single source of truth for it.

**On generation output quality**: sampling occasionally (rare, but confirmed in testing) produces a stray run of non-Latin script -- e.g. Chinese characters bleeding into an otherwise-English answer. Since English-only output is a hard product requirement, treat this as a bad sample and regenerate rather than showing it to the user:
```dart
final nonLatinLeak = RegExp(
  r'[一-鿿぀-ヿ가-퟿؀-ۿऀ-ॿ]',
);
String generateWithRetry(String prompt, int maxRetries) {
  for (var i = 0; i <= maxRetries; i++) {
    final result = llamaGenerate(prompt); // your llama_cpp_dart call
    if (!nonLatinLeak.hasMatch(result)) return result;
  }
  return llamaGenerate(prompt); // give up and return the last attempt after maxRetries
}
```
Two retries (three attempts total) is what the Python reference implementation (`fitbot/generation/generator.py`) uses.

## 4. Tokenizing for the NLU TFLite models

The `.tflite` files expect **integer token-id arrays as input, not raw
text** -- tokenization happens in Dart using the shipped vocab JSON, not
inside the model graph (this was a deliberate choice for portability). Port
`fitbot/nlu/tokenizer.py`'s logic, **including the typo-tolerance fallback**
-- without it, a single misspelled word (e.g. "wrkout" instead of
"workout") falls back to `<unk>` and can throw off classification:

```dart
List<int> encode(String text, Map<String, int> vocab, int maxLen) {
  final tokens = RegExp(r"[a-zA-Z']+")
      .allMatches(text.toLowerCase())
      .map((m) => m.group(0)!)
      .toList();
  final ids = tokens.take(maxLen).map((t) => resolveToken(t, vocab)).toList();
  while (ids.length < maxLen) { ids.add(0); } // 0 = <pad>
  return ids;
}

int resolveToken(String token, Map<String, int> vocab) {
  if (vocab.containsKey(token)) return vocab[token]!;
  final corrected = nearestVocabWord(token, vocab); // edit-distance-1 fuzzy match, see below
  return corrected != null ? vocab[corrected]! : 1; // 1 = <unk>
}
```
Port `_nearest_vocab_word` / `_edit_distance_1` from `fitbot/nlu/tokenizer.py`
verbatim -- the core idea: bucket vocab words by length, only check the
token's own length and +/-1 (fast even against a 10k-word vocab), and only
attempt fuzzy correction for tokens of 4+ characters (shorter words produce
too many false-positive corrections). This benefits both the intent
classifier and the language detector, and needs no model retraining --
it's a pure preprocessing fix. The LLM itself doesn't need this: Qwen's
subword tokenizer and broad pretraining already handle minor typos
reasonably well on its side.

- `language_vocab.json` / `intent_vocab.json`: `{"word": id, ...}`, max_len 32 / 40 respectively.
- Language model output: single sigmoid value; `> 0.5` = Roman Urdu.
- Intent model output: 44-way softmax; argmax index into `intent_labels.json`.

## 5. System prompt for the LLM

Use exactly this system prompt (it's what the model was fine-tuned against):

```
You are HybridFit Assistant, a professional, friendly, and concise AI
fitness coach built by Muhammad Junaid for the HybridFit app. You are
an AI, not Muhammad Junaid himself. You help with workouts, exercise
form, strength training, cardio, weight loss, muscle gain, nutrition,
recovery, and healthy lifestyle questions. Always reply in English,
regardless of what language the user writes in. Keep answers short and
to the point unless the question needs step-by-step detail. Never
discuss steroids, prescription medication, dosages, or diagnose
medical conditions -- redirect those to a licensed professional. Stay
strictly within the fitness/nutrition/wellness domain.
```

Format each turn in ChatML (Qwen's native format):
```
<|im_start|>system
{system prompt}<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
```
Stop generation at the `<|im_end|>` token.

## 6. No on-device RAG in this build

The Python training/reference pipeline (`fitbot/pipeline.py`) uses semantic
search (RAG) over a ~29K-document knowledge base to ground answers during
generation. That retrieval index is **not included in this export** --
shipping the sentence-embedding model would roughly double the app's
offline footprint, and the fine-tuned LLM already absorbed the corpus's
knowledge during training. The on-device app therefore relies on the
fine-tuned model directly (same as steps above), without live retrieval.
If you want on-device RAG later, `scripts/08_build_rag_index.py` and
`fitbot/rag/retriever.py` show the approach (NumPy brute-force cosine
search over sentence-transformer embeddings) and could be re-implemented
with a much smaller/quantized embedding model.

## 7. Roman Urdu is hard-gated, not answered

Earlier builds tried to have the LLM understand and reply to Roman Urdu
directly, but the small 0.5B model's Roman Urdu output was unreliable
(garbled, sometimes mixing in stray characters from other languages) even
after being told to translate and reply in English. The product decision
is now: **the bot doesn't attempt to understand Roman Urdu input at all.**
If `language_detector.tflite` predicts Roman Urdu, the safety gate returns
`englishOnlyRedirect` ("I can only understand English right now...")
immediately and the message never reaches the LLM -- this is a hard gate,
not a training bias, so it's 100% reliable regardless of what the LLM
itself would have done. See §3's `safetyCheck` for the exact logic. Do not
add any translation or "detect language and switch reply language" logic
in Dart -- Roman Urdu in always means the fixed redirect out, nothing else.
