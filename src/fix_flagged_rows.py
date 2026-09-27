"""
fix_flagged_rows.py
---------------------
Fixes the 11 rows flagged "NEEDS MANUAL REVIEW" with real values,
matched by a unique snippet of each message's text.

USAGE:
    python src/fix_flagged_rows.py
"""

import pandas as pd

PATH = "eval/golden_set_final.csv"

# Each entry: (unique text snippet to find the row, true_intent, escalate_or_auto, good_reply_note, escalate_reason)
FIXES = [
    ("Yeah that didn", "ambiguous_insufficient_context", "escalate",
     "ask what was tried, get more detail", "too vague to act on"),
    ("Fix. This. Glitch. NOW", "ambiguous_insufficient_context", "escalate",
     "ask which device/issue specifically", "no detail on what bug or device"),
    (". fix this like wtf", "ambiguous_insufficient_context", "escalate",
     "ask which device/issue specifically", "no detail on what's broken"),
    ("Only happens on my phone?", "ambiguous_insufficient_context", "escalate",
     "ask what issue and device", "fragment, missing context of original issue"),
    ("It's already been resolved", "ambiguous_insufficient_context", "auto",
     "acknowledge resolution, no action needed", "issue already resolved, nothing to escalate"),
    ("Okay got it", "ambiguous_insufficient_context", "auto",
     "acknowledge, confirm resolution", "closing remark, no new issue raised"),
    ("United Kingdom", "how_to_question", "auto",
     "use as context to continue prior troubleshooting thread", "fragment of larger conversation, not standalone"),
    ("Thank you", "ambiguous_insufficient_context", "auto",
     "acknowledge thanks, no action needed", "closing remark, no issue to resolve"),
    ("fix iOS 11", "ambiguous_insufficient_context", "escalate",
     "ask specific device and issue details", "too vague, no specific problem stated"),
    ("somebody get these damn question marks", "known_widespread_bug", "auto",
     "point to iOS 11.1.1 autocorrect fix", "known bug with published fix"),
    ("please help us lmao", "ambiguous_insufficient_context", "escalate",
     "ask specific device and issue details", "too vague to act on"),
]

if __name__ == "__main__":
    df = pd.read_csv(PATH)
    fixed_count = 0

    for snippet, intent, escalate, good_reply, reason in FIXES:
        mask = df["customer_msg_clean"].str.contains(snippet, na=False, regex=False)
        n_matches = mask.sum()
        if n_matches != 1:
            print(f"WARNING: expected 1 match for '{snippet}', found {n_matches}. Skipping.")
            continue
        df.loc[mask, "true_intent"] = intent
        df.loc[mask, "escalate_or_auto"] = escalate
        df.loc[mask, "good_reply_should_include"] = good_reply
        df.loc[mask, "escalate_reason"] = reason
        fixed_count += 1

    df.to_csv(PATH, index=False)
    print(f"Fixed {fixed_count} of {len(FIXES)} flagged rows.")

    still_flagged = df["escalate_reason"].str.contains("NEEDS MANUAL REVIEW", na=False)
    remaining = still_flagged.sum()
    print(f"{remaining} rows still flagged as needing manual review.")

    if remaining > 0:
        print("\nRemaining flagged row(s):")
        for idx in df[still_flagged].index:
            print(f"  Row {idx}: {df.loc[idx, 'customer_msg_clean']!r}")

        # We know exactly one row should still be flagged at this point: the
        # short "Thank you" message, which couldn't be matched uniquely by
        # text since other rows also contain the phrase "Thank you" within
        # a longer message. Fix it directly using its flagged status instead.
        if remaining == 1:
            last_row_text = df.loc[still_flagged, "customer_msg_clean"].iloc[0]
            if "Thank you" in last_row_text:
                df.loc[still_flagged, "true_intent"] = "ambiguous_insufficient_context"
                df.loc[still_flagged, "escalate_or_auto"] = "auto"
                df.loc[still_flagged, "good_reply_should_include"] = "acknowledge thanks, no action needed"
                df.loc[still_flagged, "escalate_reason"] = "closing remark, no issue to resolve"
                df.to_csv(PATH, index=False)
                print("\nFixed the last remaining row directly.")
            else:
                print(f"\nThe remaining row doesn't look like the expected 'Thank you' "
                      f"message - please check it manually: {last_row_text!r}")