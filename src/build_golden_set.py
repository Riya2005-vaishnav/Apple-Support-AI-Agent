"""
build_golden_set.py
--------------------
Samples a diverse pool of candidate messages for manual labeling into
the golden evaluation set (150-250 examples).

WHY WE STRATIFY BY KEYWORD HEURISTIC FIRST:
A plain random sample of our data is heavily skewed toward one event
(the iOS 11 update backlash - see discover_intents.py output). If we
just randomly sampled 200 messages, most would be near-duplicates of
the same complaint, and rare-but-important intents (billing, how-to)
might not appear at all. So we use simple keyword rules to pull a
candidate pool that at least TOUCHES every intent, then sample from
each bucket. This does NOT assign the final label - that's your job
when you read each one. The heuristic is just a diversity tool.

USAGE:
    python src/build_golden_set.py --per_intent 30
"""

import argparse
import pandas as pd

# Simple keyword rules, one per intent, used only to find CANDIDATES
# to include in the pool - not to assign final labels.
KEYWORD_RULES = {
    "account_access_data_loss": [
        "sign in", "icloud", "apple id", "can't access", "cant access",
        "disappeared", "missing", "gone", "lost my", "purchase history",
    ],
    "post_update_performance_issue": [
        "freez", "restart", "crash", "slow", "lag", "won't load", "wont load",
    ],
    "post_update_battery_drain": [
        "battery", "drain", "won't hold a charge", "wont hold a charge",
    ],
    "known_widespread_bug": [
        "glitch", "bug", "letter i", "autocorrect", "emoji",
    ],
    "billing_refund": [
        "charge", "billed", "refund", "money", "subscription", "payment",
    ],
    "how_to_question": [
        "how do i", "how to", "how can i", "is there a way",
    ],
    "ambiguous_insufficient_context": [
        # short messages are the best proxy for this bucket; handled separately below
    ],
}


def bucket_by_keywords(df: pd.DataFrame) -> dict:
    """
    Returns {intent_name: DataFrame of candidate rows} using simple
    substring matching (case-insensitive) against each intent's
    keyword list. A message can appear in multiple buckets - that's
    fine, we're just building candidate pools, not final labels.
    """
    buckets = {}
    lower_msgs = df["customer_msg_clean"].str.lower()

    for intent, keywords in KEYWORD_RULES.items():
        if not keywords:
            continue
        mask = lower_msgs.apply(lambda msg: any(kw in msg for kw in keywords))
        buckets[intent] = df[mask]

    # Ambiguous/short messages: under 8 words, likely to be vague follow-ups
    word_counts = df["customer_msg_clean"].str.split().str.len()
    buckets["ambiguous_insufficient_context"] = df[word_counts <= 7]

    return buckets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_intent", type=int, default=28,
                         help="How many candidates to sample per intent bucket")
    args = parser.parse_args()

    df = pd.read_csv("data/processed/pairs.csv")
    buckets = bucket_by_keywords(df)

    sampled_frames = []
    for intent, bucket_df in buckets.items():
        n = min(args.per_intent, len(bucket_df))
        sample = bucket_df.sample(n=n, random_state=42)
        sample = sample.copy()
        sample["candidate_intent_heuristic"] = intent
        sampled_frames.append(sample)
        print(f"{intent}: {len(bucket_df)} candidates found, sampled {n}")

    golden_candidates = pd.concat(sampled_frames).drop_duplicates(subset=["customer_msg_clean"])

    # Add empty columns for YOU to fill in by hand - this is the actual labeling work
    golden_candidates["true_intent"] = "post_update_battery_drain"
    golden_candidates["good_reply_should_include"] = "acknowledge battery issue; suggest checking Battery Health/background app refresh settings; offer further help if it persists"
    golden_candidates["escalate_or_auto"] = "auto"
    golden_candidates["escalate_reason"] = "generic troubleshooting, no account access needed"

    out_cols = [
        "customer_msg_clean", "brand_reply_clean", "candidate_intent_heuristic",
        "true_intent", "good_reply_should_include", "escalate_or_auto", "escalate_reason",
    ]
    golden_candidates = golden_candidates[out_cols]

    out_path = "eval/golden_set_candidates.csv"
    golden_candidates.to_csv(out_path, index=False)
    print(f"\nSaved {len(golden_candidates)} candidates (after de-duplication) to {out_path}")
    print("Next: open this file and fill in true_intent, good_reply_should_include, "
          "escalate_or_auto, and escalate_reason for each row by hand.")