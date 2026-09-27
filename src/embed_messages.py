"""
embed_messages.py
------------------
Embeds customer messages using Gemini's embedding model, so we can
cluster them and discover the real intent categories in the data.

WHY WE EMBED FIRST:
We don't want to invent intent categories out of thin air. Instead,
we turn each message into a vector (a list of numbers capturing its
meaning), then group similar vectors together (clustering). Reading
those clusters tells us what customers are ACTUALLY writing about.

USAGE:
    python src/embed_messages.py
"""

import os
import re
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()  # reads GEMINI_API_KEY from your .env file

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
CHECKPOINT_PATH = "data/processed/_embedding_checkpoint.npy"


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts in one API call (cheaper and faster than
    one call per message).
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def embed_batch_with_retry(texts: list[str], max_retries: int = 6) -> list[list[float]]:
    """
    Same as embed_batch, but automatically waits and retries when we
    hit the free-tier rate limit (HTTP 429), instead of crashing.

    WHY THIS IS NEEDED:
    Gemini's free tier for embeddings allows only a small number of
    requests per minute. When we exceed it, the API returns a 429
    error and tells us how long to wait before retrying (in
    `retryDelay`). We read that value and sleep for it, then try again.
    If no delay is given, we fall back to a growing wait (30s, 60s, 90s...).
    """
    for attempt in range(max_retries):
        try:
            return embed_batch(texts)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise  # not a rate-limit error, don't swallow it

            # Try to read Google's suggested wait time from the error text
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(e))
            wait_seconds = int(match.group(1)) + 5 if match else 30 * (attempt + 1)

            print(f"Rate limited. Waiting {wait_seconds}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)

    raise RuntimeError("Exceeded max retries on rate limit — try again later "
                        "or reduce batch_size / increase sleep_seconds.")


def embed_all(messages: list[str], batch_size: int = 10, sleep_seconds: int = 6) -> np.ndarray:
    """
    Loop through all messages in small batches, saving a checkpoint
    after every batch. If the script crashes or is stopped, rerunning
    it will pick up where it left off instead of re-embedding
    (and re-paying for) messages we already processed.
    """
    all_embeddings = []
    start_index = 0

    if os.path.exists(CHECKPOINT_PATH):
        all_embeddings = list(np.load(CHECKPOINT_PATH))
        start_index = len(all_embeddings)
        print(f"Resuming from checkpoint: {start_index} embeddings already done.")

    for i in range(start_index, len(messages), batch_size):
        batch = messages[i : i + batch_size]
        embeddings = embed_batch_with_retry(batch)
        all_embeddings.extend(embeddings)

        np.save(CHECKPOINT_PATH, np.array(all_embeddings))
        print(f"Embedded {min(i + batch_size, len(messages))}/{len(messages)}")
        time.sleep(sleep_seconds)  # stay safely under the free-tier rate limit

    return np.array(all_embeddings)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/pairs.csv")

    # For intent DISCOVERY, we don't need all 5000 — a representative
    # sample of ~500-800 is enough to find the clusters, and much faster
    # / cheaper to embed. We'll embed the full set later only if needed
    # for retrieval.
    sample = df.sample(n=min(600, len(df)), random_state=42).reset_index(drop=True)

    messages = sample["customer_msg_clean"].tolist()
    embeddings = embed_all(messages)

    np.save("data/processed/intent_discovery_embeddings.npy", embeddings)
    sample.to_csv("data/processed/intent_discovery_sample.csv", index=False)
    print(f"Saved {len(messages)} embeddings, shape {embeddings.shape}")

    # Clean up the checkpoint file now that we've saved the final output
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)