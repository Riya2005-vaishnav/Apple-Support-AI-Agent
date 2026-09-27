"""
finalize_partial_prelabel.py
------------------------------
If prelabel_golden_set.py got interrupted (e.g. by a daily quota
limit) partway through, this takes whatever the checkpoint saved and
produces the final working file anyway - any remaining rows just get
blank suggestion columns, meaning you label those few by hand with
no AI assist. Given it's usually only a handful of rows, that's a
trivial amount of extra manual work rather than waiting a full day
for the quota to reset.

USAGE:
    python src/finalize_partial_prelabel.py
"""

import os
import pandas as pd

CHECKPOINT_PATH = "eval/_prelabel_checkpoint.csv"

if __name__ == "__main__":
    df = pd.read_csv("eval/golden_set_candidates.csv")

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        n_done = len(checkpoint)
    else:
        checkpoint = pd.DataFrame({"suggested_intent": [], "suggested_escalate_or_auto": []})
        n_done = 0

    n_total = len(df)
    n_remaining = n_total - n_done

    suggested_intents = checkpoint["suggested_intent"].tolist() + [""] * n_remaining
    suggested_escalations = checkpoint["suggested_escalate_or_auto"].tolist() + [""] * n_remaining

    df["suggested_intent"] = suggested_intents
    df["suggested_escalate_or_auto"] = suggested_escalations

    cols = [
        "customer_msg_clean", "brand_reply_clean", "candidate_intent_heuristic",
        "suggested_intent", "true_intent",
        "suggested_escalate_or_auto", "escalate_or_auto",
        "good_reply_should_include", "escalate_reason",
    ]
    df = df[cols]

    out_path = "eval/golden_set_prelabeled.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}: {n_done} rows have AI suggestions, "
          f"{n_remaining} rows need fully manual labeling (rows {n_done + 1}-{n_total}).")