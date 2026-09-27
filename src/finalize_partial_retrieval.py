"""
finalize_partial_retrieval.py
--------------------------------
Finalizes the retrieval index using whatever embeddings made it into
the checkpoint, instead of waiting for the full target count. Given
repeated daily quota interruptions, 1300+ vectors is still a solid
retrieval index - more precedents than needed per intent - so we cut
losses here rather than lose more days to quota resets.

USAGE:
    python src/finalize_partial_retrieval.py
"""

import os
import numpy as np
import pandas as pd

CHECKPOINT_PATH = "data/processed/_retrieval_embedding_checkpoint.npy"
EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"
METADATA_PATH = "data/processed/retrieval_metadata.csv"
SAMPLE_SIZE = 1500  # the original target sample size used in retrieval.py's build_index

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found - nothing to finalize.")
        exit()

    embeddings = np.load(CHECKPOINT_PATH)
    n_done = len(embeddings)

    # Recreate the same sample that build_index() would have used, so the
    # metadata rows line up exactly with the embeddings we already have.
    df = pd.read_csv("data/processed/pairs.csv")
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=7).reset_index(drop=True)
    metadata = sample.iloc[:n_done]

    np.save(EMBEDDINGS_PATH, embeddings)
    metadata.to_csv(METADATA_PATH, index=False)

    print(f"Finalized retrieval index with {n_done} vectors (out of the {SAMPLE_SIZE} originally planned).")
    print(f"Saved to {EMBEDDINGS_PATH} and {METADATA_PATH}")