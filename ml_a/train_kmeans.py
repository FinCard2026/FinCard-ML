# ml_a/train_kmeans.py

import json
import os

import joblib
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_a.synthetic_data import generate_synthetic_profiles
from ml_a.encoder import encode_many
from ml_a.cluster_meta import CLUSTER_META


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "kmeans_pipeline.joblib")
SUMMARY_PATH = os.path.join(ARTIFACT_DIR, "cluster_summary.json")
DATA_PATH = os.path.join("data", "synthetic_user_profiles.csv")


def train_kmeans(n_clusters: int = 5, random_state: int = 42):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    profiles_df = generate_synthetic_profiles(n_per_cluster=100, random_state=random_state)
    profiles_df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    profiles = profiles_df.to_dict(orient="records")
    X = encode_many(profiles)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])

    pipeline.fit(X)

    labels = pipeline.named_steps["kmeans"].labels_
    profiles_df["cluster_id_raw"] = labels

    summary = {
        "model_version": "kmeans_v1.0",
        "n_clusters": n_clusters,
        "feature_count": X.shape[1],
        "train_rows": len(X),
        "cluster_meta": CLUSTER_META,
        "raw_cluster_counts": profiles_df["cluster_id_raw"].value_counts().to_dict(),
    }

    joblib.dump(pipeline, MODEL_PATH)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] model saved to {MODEL_PATH}")
    print(f"[OK] summary saved to {SUMMARY_PATH}")
    print(f"[OK] synthetic data saved to {DATA_PATH}")


if __name__ == "__main__":
    train_kmeans()