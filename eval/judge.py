"""
judge.py
---------
LLM-as-judge: scores a drafted reply's quality against a simple rubric.
Used to evaluate reply_agent.py's drafted replies beyond just "did the
intent match" - actually judging whether the reply itself is good.

RUBRIC (each scored 1-5):
  - helpfulness: does the reply actually address the customer's problem?
  - tone_match: does it sound like AppleSupport's real voice (not generic)?
  - grounding: is it consistent with the retrieved past precedents, rather
    than inventing unsupported claims?

USAGE (as a library):
    from judge import judge_reply
    scores = judge_reply(customer_msg, drafted_reply, retrieved_precedents)

USAGE (batch, over eval_results.csv from run_eval.py):
    python eval/judge.py --batch
"""

import os
import re
import sys
import time
import json
import argparse
import pandas as pd

sys.path.insert(0, "src")
from intents import client, GENERATION_MODEL
from google.genai import errors as genai_errors

RESULTS_PATH = "eval/eval_results.csv"
JUDGE_SCORES_PATH = "eval/judge_scores.csv"


def _build_judge_prompt(customer_msg: str, drafted_reply: str) -> str:
    return f"""You are an expert quality reviewer for customer support replies.
Score the DRAFTED REPLY below on three dimensions, each from 1 (poor) to
5 (excellent):

- helpfulness: does it actually address the customer's specific problem?
- tone_match: does it sound like a real, professional brand support agent
  (not generic, not robotic, not overly casual)?
- grounding: does it stay consistent with what a real support reply would
  say, without inventing specific facts, promises, or details it can't
  actually know?

Customer message: "{customer_msg}"

Drafted reply: "{drafted_reply}"

Respond with ONLY valid JSON, no other text, in this exact format:
{{"helpfulness": <1-5>, "tone_match": <1-5>, "grounding": <1-5>, "justification": "<one short sentence>"}}
"""


def judge_reply(customer_msg: str, drafted_reply: str) -> dict:
    """
    Scores one drafted reply. Returns a dict with helpfulness, tone_match,
    grounding (each 1-5), and a short justification. Falls back to a
    neutral score if the model's response isn't parseable JSON, rather
    than crashing a whole batch run over one bad response.
    """
    prompt = _build_judge_prompt(customer_msg, drafted_reply)
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)

    raw_text = response.text.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw_text)
        for key in ("helpfulness", "tone_match", "grounding"):
            result[key] = int(result[key])
        return result
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"WARNING: could not parse judge output ({e}). Raw: {raw_text!r}")
        return {"helpfulness": 3, "tone_match": 3, "grounding": 3,
                "justification": "PARSE_FAILED - neutral default score"}


def judge_reply_with_retry(customer_msg: str, drafted_reply: str, max_retries: int = 6) -> dict:
    """Same as judge_reply, but retries automatically on rate-limit/server errors."""
    for attempt in range(max_retries):
        try:
            return judge_reply(customer_msg, drafted_reply)
        except genai_errors.ServerError:
            wait_seconds = 20 * (attempt + 1)
            print(f"  Server overloaded. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(e))
            wait_seconds = int(match.group(1)) + 5 if match else 20 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
    raise RuntimeError("Exceeded max retries.")


def run_batch_judging():
    """
    Runs the judge over every row in eval_results.csv (produced by
    run_eval.py), scoring the pipeline's drafted replies. Checkpointed
    the same way as other long-running scripts in this project.
    """
    df = pd.read_csv(RESULTS_PATH)

    scores = []
    start_index = 0
    checkpoint_path = "eval/_judge_checkpoint.csv"
    if os.path.exists(checkpoint_path):
        scores = pd.read_csv(checkpoint_path).to_dict("records")
        start_index = len(scores)
        print(f"Resuming from checkpoint: {start_index} rows already judged.")

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        print(f"Judging {i + 1}/{len(df)}...")
        result = judge_reply_with_retry(row["message"], row["pipeline_reply"])
        scores.append(result)
        pd.DataFrame(scores).to_csv(checkpoint_path, index=False)
        time.sleep(3)

    scores_df = pd.DataFrame(scores)
    combined = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    combined.to_csv(JUDGE_SCORES_PATH, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nAverage scores across {len(combined)} replies:")
    print(f"  Helpfulness: {combined['helpfulness'].mean():.2f}/5")
    print(f"  Tone match:  {combined['tone_match'].mean():.2f}/5")
    print(f"  Grounding:   {combined['grounding'].mean():.2f}/5")
    print(f"\nSaved to {JUDGE_SCORES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Judge all rows in eval_results.csv")
    args = parser.parse_args()

    if args.batch:
        run_batch_judging()
    else:
        print("Use --batch to judge all rows in eval/eval_results.csv")