"""
reply_agent.py
---------------
The full AI support agent pipeline: classify -> retrieve -> draft reply
-> decide escalate/auto (with a stated reason).

This is the actual "agent" the assignment asks for - everything else
(data_prep, intents, retrieval) was building the pieces this file
wires together.

USAGE (as a library):
    from reply_agent import handle_message
    result = handle_message("my icloud storage was double charged")

USAGE (from command line, for quick manual testing):
    python src/reply_agent.py "my battery drains so fast since the update"
"""

import sys
import json
from intents import classify, INTENTS, client, GENERATION_MODEL
from retrieval import retrieve_similar

# Intents that, based on patterns observed while manually labeling the
# golden set, almost always require escalation - either because they
# involve money/personal data (needs verification), or because the
# intent itself means "not enough info to act on this automatically."
ALWAYS_ESCALATE_INTENTS = {"billing_refund", "account_access_data_loss", "ambiguous_insufficient_context"}

# If the classifier isn't at least this confident, we don't trust its
# call enough to auto-handle the message.
MIN_CONFIDENCE_FOR_AUTO = "high"

# If nothing in our retrieval index is at least this similar, we treat
# the situation as novel - no good precedent to safely imitate.
MIN_SIMILARITY_FOR_AUTO = 0.45


def draft_reply(message: str, intent: str, retrieved: list) -> str:
    """
    Drafts a reply grounded in real past resolutions for this brand.
    We show the model 3 similar past (customer message -> brand reply)
    pairs as few-shot examples, so the draft mimics the brand's actual
    tone and resolution style instead of generic LLM politeness.
    """
    examples_block = ""
    for i, r in enumerate(retrieved, 1):
        examples_block += (
            f"Example {i} (similarity: {r['similarity']:.2f}):\n"
            f"  Customer: {r['past_customer_msg']}\n"
            f"  Brand reply: {r['past_brand_reply']}\n\n"
        )

    prompt = f"""You are AppleSupport, replying to a customer on Twitter.
Here is how this brand has replied to similar past issues:

{examples_block}
Now draft a reply to this NEW customer message, in the same tone and
style as the examples above. The message's intent has been classified
as: {intent}

New customer message: "{message}"

Respond with ONLY the reply text, nothing else - no preamble, no labels.
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def decide_escalation(intent: str, confidence: str, max_similarity: float) -> dict:
    """
    Decides whether to auto-handle or escalate this message, and WHY.
    Kept as simple, explainable if/else logic (not another LLM call) -
    this makes the decision cheap, fast, and easy to defend/explain
    live in an interview, since you can point to the exact rule that
    fired for any given message.
    """
    if intent in ALWAYS_ESCALATE_INTENTS:
        if intent == "ambiguous_insufficient_context":
            reason = "message is too vague or lacks context to act on confidently"
        else:
            reason = (f"intent '{intent}' typically requires account-specific "
                      f"verification (money or personal data involved)")
        return {"decision": "escalate", "reason": reason}

    if confidence != MIN_CONFIDENCE_FOR_AUTO:
        return {
            "decision": "escalate",
            "reason": f"classifier confidence was '{confidence}', not high enough "
                      f"to trust an automated response",
        }

    if max_similarity < MIN_SIMILARITY_FOR_AUTO:
        return {
            "decision": "escalate",
            "reason": f"no sufficiently similar past case found (best match "
                      f"similarity: {max_similarity:.2f}), situation may be novel",
        }

    return {
        "decision": "auto",
        "reason": "generic, high-confidence issue with a known precedent, "
                  "no account access needed",
    }


def handle_message(message: str) -> dict:
    """
    Runs the full pipeline on one customer message and returns a
    complete result: intent, drafted reply, and the escalate/auto
    decision with its reason.
    """
    classification = classify(message)
    intent = classification["intent"]
    confidence = classification["confidence"]

    retrieved = retrieve_similar(message, k=3)
    max_similarity = max((r["similarity"] for r in retrieved), default=0.0)

    reply = draft_reply(message, intent, retrieved)
    escalation = decide_escalation(intent, confidence, max_similarity)

    return {
        "message": message,
        "intent": intent,
        "confidence": confidence,
        "drafted_reply": reply,
        "decision": escalation["decision"],
        "decision_reason": escalation["reason"],
        "top_similarity": max_similarity,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/reply_agent.py "customer message here"')
        sys.exit(1)

    message = sys.argv[1]
    result = handle_message(message)

    print(f"\nMESSAGE: {result['message']}")
    print(f"INTENT: {result['intent']} (confidence: {result['confidence']})")
    print(f"TOP SIMILARITY: {result['top_similarity']:.2f}")
    print(f"DECISION: {result['decision'].upper()} - {result['decision_reason']}")
    print(f"\nDRAFTED REPLY:\n{result['drafted_reply']}")