"""
fix_row6.py
------------
One-time fix for a row that got garbled during interactive labeling
(a display glitch caused repeated text to get typed into one field,
and the row ended up missing its last value). Finds that specific row
by matching a unique piece of its message text, and overwrites just
the two broken fields with clean values.

USAGE:
    python src/fix_row6.py
"""

import pandas as pd

PATH = "eval/golden_set_final.csv"

if __name__ == "__main__":
    df = pd.read_csv(PATH)

    # Find the broken row by a unique snippet of its message text
    mask = df["customer_msg_clean"].str.contains("Have tried that option 5 times", na=False)

    if mask.sum() != 1:
        print(f"Expected to find exactly 1 matching row, found {mask.sum()}. "
              f"Please check the file manually.")
    else:
        df.loc[mask, "good_reply_should_include"] = "acknowledge delay, escalate to specialist team"
        df.loc[mask, "escalate_reason"] = "account recovery stuck without response, needs specialist follow-up"
        df.to_csv(PATH, index=False)
        print("Fixed row 6 successfully.")