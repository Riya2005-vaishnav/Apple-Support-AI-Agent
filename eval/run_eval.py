"""
run_eval.py
------------
Runs the full agent pipeline (reply_agent.handle_message) against a
sample of the golden set, alongside two baselines, and reports
accuracy comparisons. This produces the "results vs at least two
baselines" section of the report.

BASELINES:
  - Trivial: always predict the most common intent in the golden set.
    This tells us the "floor" - any real system must beat this to be
    worth anything.
  - Simple: the keyword-matching heuristic from build_golden_set.py,
    reused here as an actual classifier (not just a sampling tool).
    No LLM involved, no API calls, near-instant.

USAGE:
    python eval/run_eval.py --n 50
"""

import os
import re
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, "src")
from reply_agent import handle_message, ALWAYS_ESCALATE_INTENTS
from google.genai import errors as genai_errors

GOLDEN_SET_PATH = "eval/golden_set_final.csv"
CHECKPOINT_PATH = "eval/_run_eval_checkpoint.csv"
RESULTS_PATH = "eval/eval_results.csv"

# Same keyword rules used in build_golden_set.py, reused here as the
# "simple baseline" classifier.
KEYWORD_RULES = {
    "account_access_data_loss": ["sign in", "icloud", "apple id", "can't access",
                                   "cant access", "disappeared", "missing", "gone",
                                   "lost my", "purchase history"],
    "post_update_performance_issue": ["freez", "restart", "crash", "slow", "lag",
                                        "won't load", "wont load"],
    "post_update_battery_drain": ["battery", "drain", "won't hold a charge",
                                    "wont hold a charge"],
    "known_widespread_bug": ["glitch", "bug", "letter i", "autocorrect", "emoji"],
    "billing_refund": ["charge", "billed", "refund", "money", "subscription", "payment"],
    "how_to_question": ["how do i", "how to", "how can i", "is there a way"],
}


def simple_baseline_classify(message: str) -> str:
    """
    The keyword-based 'simple' baseline. Checks each intent's keyword
    list in order; returns the first match, or 'ambiguous_insufficient_context'
    if nothing matches or the message is very short.
    """
    lower_msg = message.lower()
    if len(message.split()) <= 7:
        return "ambiguous_insufficient_context"
    for intent, keywords in KEYWORD_RULES.items():
        if any(kw in lower_msg for kw in keywords):
            return intent
    return "ambiguous_insufficient_context"


def simple_baseline_escalate(predicted_intent: str) -> str:
    """
    Simple baseline's escalation call: just checks if the predicted
    intent is one of the always-escalate intents. No confidence or
    similarity signal available to a baseline this simple.
    """
    return "escalate" if predicted_intent in ALWAYS_ESCALATE_INTENTS else "auto"


def handle_message_with_retry(message: str, max_retries: int = 6) -> dict:
    """
    Wraps handle_message with retry-on-rate-limit, since a 50+ message
    eval run will likely hit Gemini's free tier generation limits.
    """
    for attempt in range(max_retries):
        try:
            return handle_message(message)
        except genai_errors.ServerError:
            wait_seconds = 20 * (attempt + 1)
            print(f"  Server overloaded (503). Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(e))
            wait_seconds = int(match.group(1)) + 5 if match else 20 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
    raise RuntimeError("Exceeded max retries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of golden set rows to evaluate")
    args = parser.parse_args()

    golden = pd.read_csv(GOLDEN_SET_PATH)
    sample = golden.sample(n=min(args.n, len(golden)), random_state=42).reset_index(drop=True)

    # Trivial baseline: majority class, computed from the FULL golden set
    # (not just the sample), since that's the more honest ground truth.
    majority_intent = golden["true_intent"].mode()[0]
    majority_escalate = golden["escalate_or_auto"].mode()[0]
    print(f"Trivial baseline: always predicts intent='{majority_intent}', "
          f"escalate_or_auto='{majority_escalate}'")

    results = []
    start_index = 0
    if os.path.exists(CHECKPOINT_PATH):
        results = pd.read_csv(CHECKPOINT_PATH).to_dict("records")
        start_index = len(results)
        print(f"Resuming from checkpoint: {start_index} rows already evaluated.")

    for i in range(start_index, len(sample)):
        row = sample.iloc[i]
        message = row["customer_msg_clean"]
        print(f"Evaluating {i + 1}/{len(sample)}: {message[:60]}...")

        pipeline_result = handle_message_with_retry(message)

        simple_intent = simple_baseline_classify(message)
        simple_escalate = simple_baseline_escalate(simple_intent)

        results.append({
            "message": message,
            "true_intent": row["true_intent"],
            "true_escalate": row["escalate_or_auto"],

            "pipeline_intent": pipeline_result["intent"],
            "pipeline_escalate": pipeline_result["decision"],
            "pipeline_reply": pipeline_result["drafted_reply"],
            "pipeline_top_similarity": pipeline_result["top_similarity"],

            "trivial_intent": majority_intent,
            "trivial_escalate": majority_escalate,

            "simple_intent": simple_intent,
            "simple_escalate": simple_escalate,
        })

        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        time.sleep(3)  # brief pause between messages to ease rate limits

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # Compute and print accuracy comparisons
    def accuracy(pred_col, true_col):
        return (df[pred_col] == df[true_col]).mean()

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(df)} messages evaluated)")
    print(f"{'=' * 60}")
    print(f"{'Model':<12} {'Intent Acc':<12} {'Escalate Acc':<12}")
    print(f"{'Trivial':<12} {accuracy('trivial_intent', 'true_intent'):<12.2%} "
          f"{accuracy('trivial_escalate', 'true_escalate'):<12.2%}")
    print(f"{'Simple':<12} {accuracy('simple_intent', 'true_intent'):<12.2%} "
          f"{accuracy('simple_escalate', 'true_escalate'):<12.2%}")
    print(f"{'Pipeline':<12} {accuracy('pipeline_intent', 'true_intent'):<12.2%} "
          f"{accuracy('pipeline_escalate', 'true_escalate'):<12.2%}")
    print(f"\nFull per-row results saved to {RESULTS_PATH}")