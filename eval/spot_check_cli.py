"""
spot_check_cli.py
-------------------
Independent labeling spot-check: shows you a sample of golden set
messages WITHOUT revealing the existing label, asks you to judge them
fresh, then reveals the existing label and computes agreement. This
gives an honest measure of how much the AI-assisted pre-labeling
shortcut affected the golden set (see report.md Section 4).

USAGE:
    python eval/spot_check_cli.py --n 20
"""

import os
import argparse
import pandas as pd

GOLDEN_SET_PATH = "eval/golden_set_final.csv"
OUTPUT_PATH = "eval/spot_check_results.csv"

VALID_INTENTS = [
    "account_access_data_loss", "post_update_performance_issue",
    "post_update_battery_drain", "known_widespread_bug",
    "billing_refund", "how_to_question", "ambiguous_insufficient_context",
]


def ask_intent() -> str:
    while True:
        answer = input("Your independent guess for true_intent: ").strip()
        if answer in VALID_INTENTS:
            return answer
        print(f"  Not valid. Choose one of: {', '.join(VALID_INTENTS)}")


def ask_escalate() -> str:
    while True:
        answer = input("Your independent guess for escalate_or_auto: ").strip()
        if answer in ("auto", "escalate"):
            return answer
        print("  Not valid. Type 'auto' or 'escalate'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    golden = pd.read_csv(GOLDEN_SET_PATH)
    sample = golden.sample(n=min(args.n, len(golden)), random_state=99).reset_index(drop=True)

    results = []
    start_index = 0
    if os.path.exists(OUTPUT_PATH):
        results = pd.read_csv(OUTPUT_PATH).to_dict("records")
        start_index = len(results)
        print(f"Resuming: {start_index} rows already done.\n")

    for i in range(start_index, len(sample)):
        row = sample.iloc[i]
        print(f"\n{'=' * 70}")
        print(f"Row {i + 1}/{len(sample)}")
        print(f"MESSAGE: {row['customer_msg_clean']}")
        print("-" * 70)

        my_intent = ask_intent()
        my_escalate = ask_escalate()

        results.append({
            "message": row["customer_msg_clean"],
            "existing_intent": row["true_intent"],
            "my_intent": my_intent,
            "intent_agree": my_intent == row["true_intent"],
            "existing_escalate": row["escalate_or_auto"],
            "my_escalate": my_escalate,
            "escalate_agree": my_escalate == row["escalate_or_auto"],
        })

        # Save after every row - a Ctrl+C or crash won't lose progress
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    intent_agreement = df["intent_agree"].mean()
    escalate_agreement = df["escalate_agree"].mean()

    print(f"\n{'=' * 70}")
    print(f"SPOT-CHECK RESULTS ({len(df)} rows)")
    print(f"{'=' * 70}")
    print(f"Intent agreement:    {intent_agreement:.1%}")
    print(f"Escalation agreement: {escalate_agreement:.1%}")
    print(f"\nDisagreements:")
    disagreements = df[~df["intent_agree"] | ~df["escalate_agree"]]
    for _, row in disagreements.iterrows():
        print(f"  '{row['message'][:60]}...'")
        print(f"    existing: {row['existing_intent']} / {row['existing_escalate']}")
        print(f"    yours:    {row['my_intent']} / {row['my_escalate']}")
    print(f"\nSaved to {OUTPUT_PATH}")