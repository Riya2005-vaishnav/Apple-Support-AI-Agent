"""
bulk_finalize_golden_set.py
------------------------------
Fills in ALL remaining unlabeled rows at once, using:
  - the AI's suggested_intent / suggested_escalate_or_auto as the label
  - a short templated note per intent (reasonable default, not
    perfectly tailored to each message)

This trades per-row precision for speed. It does NOT replace reading
through the result - you still need to skim the final file and fix
any row that looks wrong. Keeps whatever you already labeled manually
in golden_set_final.csv untouched.

USAGE:
    python src/bulk_finalize_golden_set.py
"""

import os
import pandas as pd

PRELABELED_PATH = "eval/golden_set_prelabeled.csv"
FINAL_PATH = "eval/golden_set_final.csv"

# Reasonable default notes per intent - EDIT any of these during your
# skim pass if a specific message needs something more specific.
TEMPLATE_NOTES = {
    "account_access_data_loss": (
        "check account/iCloud status, verify identity",
        "account-specific issue, needs verification",
    ),
    "post_update_performance_issue": (
        "ask device/iOS version, suggest restart",
        "generic troubleshooting, no account access needed",
    ),
    "post_update_battery_drain": (
        "ask device/iOS version, suggest battery troubleshooting steps",
        "generic troubleshooting, no account access needed",
    ),
    "known_widespread_bug": (
        "acknowledge known bug, give workaround or fixed version",
        "known issue, workaround available",
    ),
    "billing_refund": (
        "apologize, ask for order/receipt ID, route to billing team",
        "involves money, needs account-specific verification",
    ),
    "how_to_question": (
        "give clear how-to steps or link to relevant setting",
        "generic how-to question, no account access needed",
    ),
    "ambiguous_insufficient_context": (
        "ask for more detail or clarification",
        "not enough information to determine intent",
    ),
}

if __name__ == "__main__":
    prelabeled = pd.read_csv(PRELABELED_PATH)

    already_done = pd.DataFrame()
    n_done = 0
    if os.path.exists(FINAL_PATH):
        already_done = pd.read_csv(FINAL_PATH)
        n_done = len(already_done)
        print(f"Keeping {n_done} rows you already labeled by hand.")

    remaining = prelabeled.iloc[n_done:].copy()

    new_rows = []
    for _, row in remaining.iterrows():
        intent = row.get("suggested_intent", "")
        escalate = row.get("suggested_escalate_or_auto", "")

        # Rows with no suggestion (the last 11) - fall back to a safe default
        # and flag them clearly so you know to double check these specifically.
        if pd.isna(intent) or intent == "":
            intent = "ambiguous_insufficient_context"
            escalate = "escalate"
            good_reply, reason = "NEEDS MANUAL REVIEW - no AI suggestion available", "NEEDS MANUAL REVIEW"
        else:
            good_reply, reason = TEMPLATE_NOTES.get(intent, ("", ""))

        new_rows.append({
            "customer_msg_clean": row["customer_msg_clean"],
            "brand_reply_clean": row["brand_reply_clean"],
            "true_intent": intent,
            "escalate_or_auto": escalate,
            "good_reply_should_include": good_reply,
            "escalate_reason": reason,
        })

    final_df = pd.concat([already_done, pd.DataFrame(new_rows)], ignore_index=True)
    final_df.to_csv(FINAL_PATH, index=False)

    n_flagged = sum(1 for r in new_rows if "NEEDS MANUAL REVIEW" in r["escalate_reason"])
    print(f"\nDone. {len(final_df)} total rows saved to {FINAL_PATH}.")
    print(f"{n_flagged} rows are flagged 'NEEDS MANUAL REVIEW' (no AI suggestion existed) - "
          f"find and fix these first.")
    print("Next: open this file and skim through it, fixing any row where the "
          "intent, escalate call, or note looks wrong for that specific message.")