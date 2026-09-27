"""
label_golden_set_cli.py
-------------------------
A simple interactive terminal tool for labeling the golden set - no
Excel needed. For each message, it shows you the text and the AI's
suggestion, then asks you to either press Enter to accept it, or type
your own answer to override it. Progress is saved after every row, so
you can stop anytime (Ctrl+C) and resume later exactly where you left
off.

USAGE:
    python src/label_golden_set_cli.py
"""

import os
import pandas as pd

SOURCE_PATH = "eval/golden_set_prelabeled.csv"
OUTPUT_PATH = "eval/golden_set_final.csv"

VALID_INTENTS = [
    "account_access_data_loss", "post_update_performance_issue",
    "post_update_battery_drain", "known_widespread_bug",
    "billing_refund", "how_to_question", "ambiguous_insufficient_context",
]


def ask(prompt: str, default: str = "", valid_options: list = None) -> str:
    """
    Shows a prompt with a default value (in brackets). Pressing Enter
    alone accepts the default. Typing something else overrides it.
    If valid_options is given, keeps asking until the answer matches
    one of them (case-insensitive), so you can't accidentally typo
    an intent name into a brand new 8th fake category.
    """
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        answer = raw if raw else default
        if valid_options and answer not in valid_options:
            print(f"  Not a valid option. Choose one of: {', '.join(valid_options)}")
            continue
        return answer


if __name__ == "__main__":
    df = pd.read_csv(SOURCE_PATH)

    # Resume support: if we've already labeled some rows in a previous
    # session, skip them and continue from where we left off.
    start_index = 0
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        start_index = len(done_df)
        print(f"Resuming: {start_index} rows already labeled.\n")

    total = len(df)

    for i in range(start_index, total):
        row = df.iloc[i]
        print(f"\n{'=' * 70}")
        print(f"Row {i + 1}/{total}")
        print(f"MESSAGE: {row['customer_msg_clean']}")
        print(f"(old brand reply, for context): {row['brand_reply_clean']}")
        print("-" * 70)

        suggested_intent = row.get("suggested_intent", "")
        suggested_escalate = row.get("suggested_escalate_or_auto", "")

        # If there's no suggestion (the last 11 rows), fall back to asking
        # freely without a default, forcing a real choice.
        if pd.isna(suggested_intent) or suggested_intent == "":
            print("(no AI suggestion for this row - label it yourself)")
            true_intent = ask("true_intent", default=VALID_INTENTS[0], valid_options=VALID_INTENTS)
            escalate = ask("escalate_or_auto", default="auto", valid_options=["auto", "escalate"])
        else:
            true_intent = ask("true_intent (Enter to accept)", default=suggested_intent, valid_options=VALID_INTENTS)
            escalate = ask("escalate_or_auto (Enter to accept)", default=suggested_escalate, valid_options=["auto", "escalate"])

        good_reply = input("good_reply_should_include (short note): ").strip()
        reason = input("escalate_reason (short note): ").strip()

        # Append this one row to the output file immediately - if you
        # stop the script anytime, everything up to now is saved.
        result_row = pd.DataFrame([{
            "customer_msg_clean": row["customer_msg_clean"],
            "brand_reply_clean": row["brand_reply_clean"],
            "true_intent": true_intent,
            "escalate_or_auto": escalate,
            "good_reply_should_include": good_reply,
            "escalate_reason": reason,
        }])
        result_row.to_csv(OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False)

    print(f"\nAll {total} rows labeled! Saved to {OUTPUT_PATH}")