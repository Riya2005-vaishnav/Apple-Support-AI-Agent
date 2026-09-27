"""
discover_intents.py
--------------------
Clusters the embedded customer messages into groups, and shows you
representative examples from each group so you can read them and
assign a real intent name.

WHY THIS IS SEMI-MANUAL:
K-means will happily produce N mathematically coherent clusters, but
it has no idea what "billing complaint" or "device broken" means.
Only a human reading real examples can assign that meaning. This
script does the grouping; you do the labeling.

USAGE:
    python src/discover_intents.py --n_clusters 8
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_clustering(embeddings: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    Groups the embeddings into n_clusters groups. Returns an array
    where element i is the cluster ID assigned to message i.

    random_state is fixed so this is reproducible — running this
    again gives you the exact same clusters, which matters for your
    README's "reproduce in 15 minutes" requirement.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids, kmeans


def show_representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    kmeans: KMeans,
    n_examples: int = 8,
):
    """
    For each cluster, find and print the messages CLOSEST to that
    cluster's center. These are the most "typical" examples of
    whatever pattern that cluster represents — much more useful to
    read than random examples, which might include edge cases.
    """
    df = df.copy()
    df["cluster"] = cluster_ids

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_mask = cluster_ids == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_messages = df.loc[cluster_mask, "customer_msg_clean"].tolist()

        center = kmeans.cluster_centers_[cluster_id]
        # distance of each message's embedding from its cluster's center
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_indices = np.argsort(distances)[:n_examples]

        print(f"\n{'=' * 60}")
        print(f"CLUSTER {cluster_id}  ({cluster_mask.sum()} messages)")
        print(f"{'=' * 60}")
        for idx in closest_indices:
            print(f"  - {cluster_messages[idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=8)
    args = parser.parse_args()

    embeddings = np.load("data/processed/intent_discovery_embeddings.npy")
    df = pd.read_csv("data/processed/intent_discovery_sample.csv")

    cluster_ids, kmeans = run_clustering(embeddings, args.n_clusters)
    show_representative_examples(df, embeddings, cluster_ids, kmeans)

    # Save cluster assignments so we can revisit this later without
    # re-running k-means (e.g. once you've named each cluster)
    df["cluster"] = cluster_ids
    df.to_csv("data/processed/clustered_sample.csv", index=False)
    print(f"\nSaved cluster assignments to data/processed/clustered_sample.csv")