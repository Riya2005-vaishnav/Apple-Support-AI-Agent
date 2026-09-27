"""
intents.py
----------
Defines our intent taxonomy (discovered via clustering in Phase 2) and
provides a classify() function that uses Gemini to assign one of these
intents to a new customer message.

HOW CLASSIFICATION WORKS HERE:
We use "few-shot prompting" — instead of training a custom model, we
give Gemini the taxonomy, a short definition of each intent, and 1-2
real examples per intent, then ask it to classify a NEW message the
same way. This works well for a small, well-defined label set and
needs no training data or GPU.

USAGE (as a library):
    from intents import classify
    result = classify("my battery is draining so fast since the update")
    # -> {"intent": "post_update_battery_drain", "confidence": "high"}
"""

import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

  # fast + cheap, good for classification
GENERATION_MODEL = "gemini-3.5-flash-lite"

# Taxonomy discovered from clustering real AppleSupport customer messages
# (see notebooks/exploration and src/discover_intents.py for how these
# were derived). Each intent has a short definition and 1-2 real
# examples, used as few-shot context for the classifier.
INTENTS = {
    "account_access_data_loss": {
        "description": "Customer can't sign in, access iCloud/App Store, or their "
                        "purchased/synced content (photos, music, playlists) has disappeared.",
        "examples": [
            "Can't use the App Store; refuses to sign in my Apple ID. Can't download anything.",
            "all my playlist and songs I picked for Apple Music are now gone",
        ],
    },
    "post_update_performance_issue": {
        "description": "Phone is freezing, restarting, slow, or apps are crashing after "
                        "a software update. Does NOT primarily mention battery.",
        "examples": [
            "why is my phone freezing every second after your latest update?",
            "my phone acts like it has no cellular service, doesn't wanna load anything, is it cuz of the update?",
        ],
    },
    "post_update_battery_drain": {
        "description": "Battery draining unusually fast or not holding a charge, "
                        "specifically after installing a software update.",
        "examples": [
            "heavy Battery drain on iOS 11.1 on a iPhone 7. Worst than 11.0.3",
            "This new update on my iPhone has really slowed down my battery life",
        ],
    },
    "known_widespread_bug": {
        "description": "A specific, named, reproducible software bug affecting many users "
                        "(e.g. a particular glitch), as opposed to a general performance complaint.",
        "examples": [
            "fix the I problem, it's annoying - autocorrect keeps replacing the letter I",
            "you guys better fix this glitch with the LETTER I",
        ],
    },
    "billing_refund": {
        "description": "Customer was charged incorrectly, charged twice, or wants a refund.",
        "examples": [
            "billed twice for icloud storage",
            "I was charged for a subscription I already cancelled, please refund",
        ],
    },
    "how_to_question": {
        "description": "Customer wants guidance on how to do something. Nothing is broken; "
                        "they just don't know how to use a feature.",
        "examples": [
            "New phone and it wont let me screenshot... thanks apple",
            "how do I turn off read receipts in messages",
        ],
    },
    "ambiguous_insufficient_context": {
        "description": "Message is too short, vague, or context-dependent to classify "
                        "confidently on its own (e.g. a reactive follow-up in a thread).",
        "examples": [
            "This keeps popping up too.",
            "MINE DOES THE SAME I GET SO MAD",
        ],
    },
}


def _build_prompt(message: str) -> str:
    """
    Builds the few-shot classification prompt. We ask for strict JSON
    output so the response is easy to parse programmatically, and we
    ask for a confidence level so the escalation logic (Phase 6) can
    use it later.
    """
    taxonomy_block = ""
    for intent_id, info in INTENTS.items():
        examples = "\n".join(f'    - "{ex}"' for ex in info["examples"])
        taxonomy_block += (
            f"- {intent_id}: {info['description']}\n"
            f"  Examples:\n{examples}\n"
        )

    return f"""You are classifying customer support messages sent to Apple's
support Twitter account into exactly one of the following intents.

{taxonomy_block}

Classify this new message:
"{message}"

Respond with ONLY valid JSON, no other text, in this exact format:
{{"intent": "<one of the intent ids above>", "confidence": "<high|medium|low>"}}
"""


def classify(message: str) -> dict:
    """
    Classifies a single customer message. Returns a dict like
    {"intent": "billing_refund", "confidence": "high"}.

    If Gemini's response isn't valid JSON (rare, but LLMs occasionally
    add stray text), we fall back to a safe default rather than
    crashing the whole pipeline on one bad message.
    """
    prompt = _build_prompt(message)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    raw_text = response.text.strip()
    # Strip markdown code fences if the model adds them despite instructions
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw_text)
        if result.get("intent") not in INTENTS:
            raise ValueError(f"Unknown intent returned: {result.get('intent')}")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"WARNING: could not parse classifier output ({e}). Raw: {raw_text!r}")
        return {"intent": "ambiguous_insufficient_context", "confidence": "low"}


def classify_batch(messages: list) -> list:
    """
    Classifies MANY messages in a single API call, instead of one call
    per message. This matters because Gemini's free tier limits the
    NUMBER OF REQUESTS per minute, not how much content is inside each
    request - so batching 15 messages into 1 call uses the same quota
    as classifying just 1 message alone.

    Returns a list of dicts, same order as the input messages.
    """
    taxonomy_block = ""
    for intent_id, info in INTENTS.items():
        examples = "\n".join(f'    - "{ex}"' for ex in info["examples"])
        taxonomy_block += (
            f"- {intent_id}: {info['description']}\n"
            f"  Examples:\n{examples}\n"
        )

    numbered_messages = "\n".join(f'{i}. "{msg}"' for i, msg in enumerate(messages))

    prompt = f"""You are classifying customer support messages sent to Apple's
support Twitter account into exactly one of the following intents.

{taxonomy_block}

Classify EACH of these {len(messages)} messages:
{numbered_messages}

Respond with ONLY a valid JSON array, no other text, with one object per
message in the SAME ORDER, in this exact format:
[{{"index": 0, "intent": "<intent_id>", "confidence": "<high|medium|low>"}}, ...]
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    raw_text = response.text.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        results = json.loads(raw_text)
        if len(results) != len(messages):
            raise ValueError(f"Expected {len(messages)} results, got {len(results)}")
        # Sort by index just in case the model reorders them
        results = sorted(results, key=lambda r: r["index"])
        output = []
        for r in results:
            if r.get("intent") not in INTENTS:
                r["intent"] = "ambiguous_insufficient_context"
                r["confidence"] = "low"
            output.append(r)
        return output
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"WARNING: batch classification parse failed ({e}). Falling back "
              f"to 'ambiguous_insufficient_context' for this whole batch.")
        return [{"index": i, "intent": "ambiguous_insufficient_context", "confidence": "low"}
                for i in range(len(messages))]


def classify_batch_with_retry(messages: list, max_retries: int = 6) -> list:
    """
    Same as classify_batch, but automatically waits and retries on:
      - HTTP 429 (rate limit / quota exceeded)
      - HTTP 503 (Google's servers temporarily overloaded - transient,
        unrelated to our own usage, just needs a short wait and retry)
    """
    for attempt in range(max_retries):
        try:
            return classify_batch(messages)
        except genai_errors.ServerError as e:
            wait_seconds = 20 * (attempt + 1)
            print(f"Server temporarily overloaded (503). Waiting {wait_seconds}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(e))
            wait_seconds = int(match.group(1)) + 5 if match else 20 * (attempt + 1)
            print(f"Rate limited. Waiting {wait_seconds}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Exceeded max retries.")


if __name__ == "__main__":
    # Quick manual test
    test_messages = [
        "my battery is draining so fast since the last update",
        "I was charged twice for my icloud storage this month",
        "how do i turn off notifications for messages",
    ]
    for msg in test_messages:
        result = classify(msg)
        print(f"{msg!r} -> {result}")