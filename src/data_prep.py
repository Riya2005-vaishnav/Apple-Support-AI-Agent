"""
data_prep.py
------------
Turns the raw Kaggle "Customer Support on Twitter" CSV into clean
(customer_message -> brand_reply) pairs for a single brand.

WHY THIS FILE EXISTS:
The raw data is a flat table of tweets. A "conversation" is scattered
across rows, connected only by `in_response_to_tweet_id` pointers.
This script walks those pointers to reconstruct real exchanges.

USAGE:
    python src/data_prep.py --brand AppleSupport --sample_size 5000

You must first download `twcs.csv` yourself from:
    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
and place it at data/raw/twcs.csv (it's ~350MB, too big to check into git —
see README for how we handle this).
"""

import argparse
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw twcs.csv file.

    We only need a handful of columns to keep memory usage sane —
    the full file is ~3M rows and loading everything at once on a
    laptop can be slow/crash-prone.
    """
    cols = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    df = pd.read_csv(path, usecols=cols)

    # inbound comes in as the string "True"/"False" in some exports — normalize it
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    return df


def filter_brand_replies(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Return only the rows where the BRAND is the one replying.
    `author_id` for brand accounts is literally the brand's Twitter
    handle (e.g. "AppleSupport"), so this filter is a simple equality
    check — no fuzzy matching needed.
    """
    brand_replies = df[(df["author_id"] == brand) & (~df["inbound"])]
    print(f"Found {len(brand_replies)} replies from {brand}")
    return brand_replies


def reconstruct_pairs(df: pd.DataFrame, brand_replies: pd.DataFrame) -> pd.DataFrame:
    """
    For each brand reply, find the customer tweet it was replying to.

    HOW THIS WORKS:
    - We index the full dataset by tweet_id so lookups are O(1) instead
      of scanning the whole table for every single reply (that would be
      painfully slow at 3M rows).
    - For each brand reply, we look up `in_response_to_tweet_id` in that
      index. If we find a row AND that row is inbound (from a customer),
      we've reconstructed a real (customer_msg -> brand_reply) pair.
    - Some brand replies are actually replying to ANOTHER brand-agent
      tweet (e.g. a follow-up in the same thread) or to a tweet that
      isn't in the dataset at all (deleted, or brand replying to itself).
      Those get dropped — we only want genuine customer-originated pairs.
    """
    # Index by tweet_id for fast lookup. Using .set_index avoids repeated
    # linear scans, which matters a lot at this scale.
    by_id = df.set_index("tweet_id")

    pairs = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue  # this reply wasn't a reply to anything we can trace

        parent_id = int(parent_id)
        if parent_id not in by_id.index:
            continue  # the original tweet isn't in our loaded data

        parent = by_id.loc[parent_id]
        # a tweet_id can rarely map to multiple rows if duplicated; guard for that
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]

        if not parent["inbound"]:
            continue  # brand was replying to another brand tweet, not a customer

        pairs.append({
            "thread_id": parent_id,
            "customer_msg": parent["text"],
            "brand_reply": reply_row["text"],
            "customer_tweet_id": parent_id,
            "brand_tweet_id": reply_row["tweet_id"],
            "timestamp": reply_row["created_at"],
        })

    pairs_df = pd.DataFrame(pairs)
    print(f"Reconstructed {len(pairs_df)} customer->brand pairs")
    return pairs_df


def clean_text(text: str) -> str:
    """
    Light cleaning: strip @mentions (they're just routing, not content)
    and collapse whitespace. We deliberately do NOT strip things like
    order numbers or punctuation — those can matter for intent/context.
    """
    import re
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--raw_path", default="data/raw/twcs.csv")
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    df = load_raw_data(args.raw_path)
    brand_replies = filter_brand_replies(df, args.brand)
    pairs_df = reconstruct_pairs(df, brand_replies)

    # Clean both sides of the pair
    pairs_df["customer_msg_clean"] = pairs_df["customer_msg"].apply(clean_text)
    pairs_df["brand_reply_clean"] = pairs_df["brand_reply"].apply(clean_text)

    # Drop obviously broken rows (empty after cleaning, or near-duplicate spam)
    pairs_df = pairs_df[pairs_df["customer_msg_clean"].str.len() > 5]
    pairs_df = pairs_df.drop_duplicates(subset=["customer_msg_clean"])

    # Subsample with a FIXED seed. This matters a lot: whoever reproduces
    # your results (a Hiver reviewer, running your README) needs to get
    # the exact same subsample you did, or your reported numbers won't match.
    if len(pairs_df) > args.sample_size:
        pairs_df = pairs_df.sample(n=args.sample_size, random_state=42)

    out_path = "data/processed/pairs.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"Saved {len(pairs_df)} pairs to {out_path}")