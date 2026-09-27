"""
retrieval.py
------------
Builds an embedding index over past (customer_msg -> brand_reply) pairs,
so we can find real precedents for how AppleSupport has handled similar
issues before. This is what makes reply drafting "grounded" instead of
generic.

WHY LOCAL EMBEDDINGS (NOT GEMINI):
We originally used Gemini's embedding API, but its free-tier quota
(a small number of requests per day) got exhausted repeatedly and
blocked progress for multiple sessions. Retrieval needs an embedding
call for EVERY new customer message at runtime (not just once when
building the index), so this bottleneck would keep recurring during
evaluation. Switching to a local model removes rate limits entirely -
it's slower on CPU than an API call, but has no quota, no cost, and
no waiting. We still use Gemini for classification and reply drafting
(src/intents.py), since that quota has been far more workable.

WHY COSINE SIMILARITY + NUMPY (NOT A VECTOR DATABASE):
At ~1,500 vectors, a brute-force comparison against every vector takes
milliseconds. A dedicated vector DB (FAISS, Pinecone, etc.) starts to
matter at millions of vectors, not thousands. Keeping this simple means
you can explain exactly how it works, line by line, in your interview.

USAGE:
    python src/retrieval.py --build          # builds the index (run once)
    python src/retrieval.py --query "my icloud storage was double charged"
"""

import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# A small, fast, well-regarded general-purpose embedding model.
# Downloads once (~90MB) on first use, then runs entirely locally.
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_SAMPLE_SIZE = 1500
EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"
METADATA_PATH = "data/processed/retrieval_metadata.csv"

_model = None


def get_model() -> SentenceTransformer:
    """
    Loads the embedding model once and reuses it, rather than reloading
    it on every call (loading is the slow part - actual embedding of
    text is fast once the model is in memory).
    """
    global _model
    if _model is None:
        print(f"Loading local embedding model ({MODEL_NAME})... this may take "
              f"a moment the first time as it downloads.")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index():
    """
    Embeds a sample of customer messages from our processed pairs and
    saves both the embeddings and their matching metadata (the original
    customer message + the brand's actual reply) to disk. This only
    needs to run once - after that, retrieve_similar() just loads the
    saved files. No rate limits, so no checkpointing needed - this
    typically finishes in well under a minute even for 1500 messages.
    """
    df = pd.read_csv("data/processed/pairs.csv")
    sample = df.sample(n=min(INDEX_SAMPLE_SIZE, len(df)), random_state=7).reset_index(drop=True)
    messages = sample["customer_msg_clean"].tolist()

    model = get_model()
    print(f"Embedding {len(messages)} messages locally...")
    embeddings = model.encode(messages, show_progress_bar=True, convert_to_numpy=True)

    np.save(EMBEDDINGS_PATH, embeddings)
    sample.to_csv(METADATA_PATH, index=False)
    print(f"Index built: {embeddings.shape[0]} vectors, saved to {EMBEDDINGS_PATH}")


def _cosine_similarity(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between one query vector and every
    vector in the index, all at once (vectorized, no loop). Returns
    an array of similarity scores, one per indexed message.
    """
    query_norm = query_vec / np.linalg.norm(query_vec)
    index_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return index_norms @ query_norm


def retrieve_similar(new_message: str, k: int = 3) -> list:
    """
    Given a new customer message, returns the k most similar past
    (customer_msg, brand_reply) pairs, most similar first. Each result
    includes the similarity score, which the reply-drafting step and
    the escalation logic will both use (e.g. low max similarity ->
    no good precedent exists -> lean toward escalating).
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    model = get_model()
    query_embedding = model.encode([new_message], convert_to_numpy=True)[0]
    similarities = _cosine_similarity(query_embedding, embeddings)

    top_k_idx = np.argsort(similarities)[::-1][:k]
    results = []
    for idx in top_k_idx:
        results.append({
            "past_customer_msg": metadata.iloc[idx]["customer_msg_clean"],
            "past_brand_reply": metadata.iloc[idx]["brand_reply_clean"],
            "similarity": float(similarities[idx]),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the retrieval index")
    parser.add_argument("--query", type=str, help="Test retrieval with a sample message")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        results = retrieve_similar(args.query)
        for r in results:
            print(f"\n[similarity: {r['similarity']:.3f}]")
            print(f"  Past customer msg: {r['past_customer_msg']}")
            print(f"  Past brand reply:  {r['past_brand_reply']}")
    else:
        print("Use --build to build the index, or --query \"your message\" to test retrieval.")