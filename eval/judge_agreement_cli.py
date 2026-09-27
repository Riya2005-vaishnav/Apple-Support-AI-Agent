"""
judge_agreement_cli.py
------------------------
Human-agreement-with-judge check: shows you a sample of (customer
message, drafted reply) pairs WITHOUT revealing the LLM-judge's scores,
asks you to score them yourself on the same rubric, then reveals the
judge's scores and computes agreement. This is the required evidence
for "how well your judge agrees with a human."

USAGE:
    python eval/judge_agreement_cli.py --n 15
"""

import os
import argparse
import pandas as pd

JUDGE_SCORES_PATH = "eval/judge_scores.csv"
OUTPUT_PATH = "eval/judge_agreement_results.csv"


def ask_score(dimension: str) -> int:
    while True:
        raw = input(f"Your score for {dimension} (1-5): ").strip()
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print("  Please enter a number from 1 to 5.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()

    judged = pd.read_csv(JUDGE_SCORES_PATH)
    sample = judged.sample(n=min(args.n, len(judged)), random_state=123).reset_index(drop=True)

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
        print(f"CUSTOMER MESSAGE: {row['message']}")
        print(f"\nDRAFTED REPLY: {row['pipeline_reply']}")
        print("-" * 70)

        my_helpfulness = ask_score("helpfulness")
        my_tone = ask_score("tone_match")
        my_grounding = ask_score("grounding")

        results.append({
            "message": row["message"],
            "reply": row["pipeline_reply"],
            "judge_helpfulness": row["helpfulness"],
            "my_helpfulness": my_helpfulness,
            "judge_tone_match": row["tone_match"],
            "my_tone_match": my_tone,
            "judge_grounding": row["grounding"],
            "my_grounding": my_grounding,
        })

        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    df = pd.DataFrame(results)

    # Agreement = "within 1 point" on a 1-5 scale - stricter exact-match
    # agreement is often noisy on subjective 1-5 scoring, so both are shown.
    def within_one(judge_col, my_col):
        return (abs(df[judge_col] - df[my_col]) <= 1).mean()

    def exact_match(judge_col, my_col):
        return (df[judge_col] == df[my_col]).mean()

    print(f"\n{'=' * 70}")
    print(f"JUDGE AGREEMENT RESULTS ({len(df)} rows)")
    print(f"{'=' * 70}")
    for dim in ["helpfulness", "tone_match", "grounding"]:
        exact = exact_match(f"judge_{dim}", f"my_{dim}")
        close = within_one(f"judge_{dim}", f"my_{dim}")
        print(f"{dim:<15} exact match: {exact:.1%}   within 1 point: {close:.1%}")

    print(f"\nSaved to {OUTPUT_PATH}")