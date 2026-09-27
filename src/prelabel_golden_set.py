"""
prelabel_golden_set.py
------------------------
Runs our OWN classifier over the golden set candidates to fill in a
DRAFT intent guess and a draft escalate/auto suggestion, so the human
labeling step becomes "read and confirm/correct" instead of "recall
and type from scratch." This is much faster, but still requires
actually reading each message - you're checking the machine's work,
not rubber-stamping it.

IMPORTANT HONESTY NOTE FOR YOUR REPORT/DECISION LOG:
Using the same classifier to pre-label its own evaluation set can bias
results if you just accept every suggestion blindly - the golden set
would stop being an independent check on the classifier. To guard
against this:
  1. You must actually read every message and genuinely judge it,
     not just click "accept" without thinking.
  2. We recommend labeling ~20-30 of these completely from scratch
     first (ignoring the suggestion column), as a spot-check, and
     comparing how often you agreed with the draft. Report that
     agreement rate - it's a genuinely useful, honest number for
     your "what's misleading about my headline number" section.

USAGE:
    python src/prelabel_golden_set.py
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, "src")
from intents import classify_batch_with_retry

BATCH_SIZE = 15
SLEEP_SECONDS = 14  # free tier allows 5 requests/minute -> ~12s minimum between calls
CHECKPOINT_PATH = "eval/_prelabel_checkpoint.csv"

# Rough starting heuristic for escalate/auto - also just a DRAFT for you to judge,
# not a rule to trust blindly.
LIKELY_ESCALATE_INTENTS = {"billing_refund", "account_access_data_loss", "ambiguous_insufficient_context"}


def suggest_escalation(intent: str) -> str:
    return "escalate" if intent in LIKELY_ESCALATE_INTENTS else "auto"


if __name__ == "__main__":
    df = pd.read_csv("eval/golden_set_candidates.csv")
    messages = df["customer_msg_clean"].tolist()

    suggested_intents = []
    suggested_escalations = []
    start_index = 0

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        suggested_intents = checkpoint["suggested_intent"].tolist()
        suggested_escalations = checkpoint["suggested_escalate_or_auto"].tolist()
        start_index = len(suggested_intents)
        print(f"Resuming from checkpoint: {start_index} messages already labeled.")

    for i in range(start_index, len(messages), BATCH_SIZE):
        batch = messages[i : i + BATCH_SIZE]
        results = classify_batch_with_retry(batch)
        for r in results:
            suggested_intents.append(r["intent"])
            suggested_escalations.append(suggest_escalation(r["intent"]))

        # Save checkpoint after every batch so a crash doesn't lose progress
        pd.DataFrame({
            "suggested_intent": suggested_intents,
            "suggested_escalate_or_auto": suggested_escalations,
        }).to_csv(CHECKPOINT_PATH, index=False)

        print(f"Pre-labeled {min(i + BATCH_SIZE, len(messages))}/{len(messages)}")
        time.sleep(SLEEP_SECONDS)

    df["suggested_intent"] = suggested_intents
    df["suggested_escalate_or_auto"] = suggested_escalations

    # Reorder so the suggestions sit right next to the columns you'll edit
    cols = [
        "customer_msg_clean", "brand_reply_clean", "candidate_intent_heuristic",
        "suggested_intent", "true_intent",
        "suggested_escalate_or_auto", "escalate_or_auto",
        "good_reply_should_include", "escalate_reason",
    ]
    df = df[cols]

    out_path = "eval/golden_set_prelabeled.csv"
    df.to_csv(out_path, index=False)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\nSaved {len(df)} pre-labeled candidates to {out_path}")
    print("Next: open this file. For each row, read the message, check the")
    print("'suggested_intent' and 'suggested_escalate_or_auto' columns, and")
    print("copy them into 'true_intent'/'escalate_or_auto' if correct, or")
    print("type the correct value if the suggestion is wrong.")