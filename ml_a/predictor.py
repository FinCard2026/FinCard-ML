# ml_a/predictor.py

import os
import math

import joblib
import numpy as np

from ml_a.encoder import encode_user_profile
from ml_a.cluster_meta import CLUSTER_META


MODEL_PATH = os.path.join("artifacts", "kmeans_pipeline.joblib")


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"K-Means model artifact not found: {MODEL_PATH}. "
            "Run `python -m ml_a.train_kmeans` first."
        )
    return joblib.load(MODEL_PATH)


def _calc_similarity(distance: float) -> float:
    """
    거리값을 0~1 유사도로 변환.
    거리가 가까울수록 1에 가까움.
    """
    similarity = math.exp(-distance / 5)
    return round(float(similarity), 2)


def _apply_demo_cluster_override(user_profile: dict, cluster_id: int) -> int:
    """
    데모 안정성을 위한 최소 보정 규칙.
    K-Means 결과가 사용자 직관과 너무 다르게 나오는 것을 방지한다.

    기본 흐름:
    1. K-Means가 먼저 cluster_id를 예측한다.
    2. 직업/주거/목적이 명확한 경우에만 cluster_id를 보정한다.
    3. 명확한 조건이 없으면 K-Means 결과를 그대로 사용한다.
    """
    job_type = user_profile.get("job_type")
    housing_type = user_profile.get("housing_type")
    goal = user_profile.get("goal")
    monthly_income = user_profile.get("monthly_income", 0) or 0
    income_stability = user_profile.get("income_stability")

    # 1. 취업준비생/무직자는 취업준비·무소득형으로 보정
    if job_type in ["job_seeker", "unemployed"]:
        return 4

    # 2. 월세 + 주거 목적이 명확하면 월세 부담형으로 보정
    if housing_type == "monthly_rent" and goal == "housing":
        return 3

    # 3. 저소득 대학생은 저소득 대학생형으로 보정
    if job_type == "student" and monthly_income <= 1000000:
        return 0

    # 4. 프리랜서 + 소득 불안정이면 프리랜서·소득불안정형으로 보정
    if job_type == "freelancer" and income_stability == "unstable":
        return 2

    # 5. 정규직/계약직 + 자산형성/절세 목적이면 사회초년생 저축형으로 보정
    if job_type in ["full_time", "contract"] and goal in ["asset_building", "tax"]:
        return 1

    return cluster_id


def predict_cluster(user_profile: dict) -> dict:
    """
    사용자 profile을 입력받아 cluster_id, cluster_name, similarity_score 반환.
    ML-B response_builder의 cluster_result로 그대로 넘길 수 있다.
    """
    pipeline = _load_model()

    X = encode_user_profile(user_profile)
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    kmeans = pipeline.named_steps["kmeans"]

    # 1. K-Means로 1차 군집 예측
    cluster_id = int(kmeans.predict(X_scaled)[0])

    # 2. 데모 안정성을 위한 rule-based 후처리 보정
    cluster_id = _apply_demo_cluster_override(user_profile, cluster_id)

    # 3. 보정된 cluster_id 기준으로 중심점과의 거리 계산
    center = kmeans.cluster_centers_[cluster_id]
    distance = float(np.linalg.norm(X_scaled[0] - center))
    similarity_score = _calc_similarity(distance)

    meta = CLUSTER_META.get(cluster_id, {
        "cluster_name": "기타 유형",
        "cluster_description": "명확한 유형을 판단하기 어려운 사용자입니다.",
    })

    return {
        "cluster_id": cluster_id,
        "cluster_name": meta["cluster_name"],
        "cluster_description": meta["cluster_description"],
        "similarity_score": similarity_score,
    }