# category_scores.py
# Section 10-1 기준: 카테고리 적합도 계산
# 공식: 0.40 * Cluster Category Preference + 0.35 * Goal Match + 0.25 * Profile Completeness

# 군집별 카테고리 선호도 테이블 (Section 9 군집 정의 기반)
# 이수빈(ML-A) K-Means 학습 완료 후 실측값으로 교체 필요
CLUSTER_CATEGORY_PREFERENCE = {
    0: {  # 저소득 대학생형
        "asset_building": 0.6,
        "education": 0.8,
        "living_support": 0.7,
        "housing": 0.4,
        "employment": 0.5,
    },
    1: {  # 사회초년생 저축형
        "asset_building": 0.9,
        "tax": 0.7,
        "housing": 0.5,
        "employment": 0.4,
        "living_support": 0.3,
    },
    2: {  # 프리랜서·소득불안정형
        "asset_building": 0.8,
        "housing": 0.7,
        "living_support": 0.6,
        "tax": 0.5,
        "employment": 0.3,
    },
    3: {  # 월세 부담형
        "housing": 0.9,
        "living_support": 0.7,
        "asset_building": 0.5,
        "employment": 0.3,
        "education": 0.2,
    },
    4: {  # 취업준비·무소득형
        "employment": 0.9,
        "living_support": 0.8,
        "asset_building": 0.4,
        "education": 0.5,
        "housing": 0.3,
    },
}


def calc_category_scores(
    cluster_id: int,
    goal: str,
    completeness_score: float,
    selected_categories: list,
) -> list:
    """
    카테고리별 적합도 점수 계산 후 rank 부여, 내림차순 정렬 반환

    Returns:
        [{"category": str, "score": float, "rank": int}, ...]
    """
    preferences = CLUSTER_CATEGORY_PREFERENCE.get(cluster_id, {})
    scores = []

    for category, pref in preferences.items():
        if goal == category:
            goal_match = 1.0
        elif category in selected_categories:
            goal_match = 0.5
        else:
            goal_match = 0.0

        score = (
            0.40 * pref
            + 0.35 * goal_match
            + 0.25 * completeness_score
        )
        scores.append({"category": category, "score": round(score, 2)})

    scores.sort(key=lambda x: x["score"], reverse=True)

    for i, s in enumerate(scores):
        s["rank"] = i + 1

    return scores
