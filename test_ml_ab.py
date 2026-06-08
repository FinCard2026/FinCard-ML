# test_ml_ab.py

import json
import pandas as pd

from ml_a.predictor import predict_cluster
from ml_b.response_builder import build_response


REQUEST_ID = "req_demo_001"
BENEFIT_MASTER_PATH = "data/benefit_master.csv"


USER_PROFILE = {
    "age": 29,
    "job_type": "freelancer",
    "monthly_income": 900000,
    "region": "seoul",
    "housing_type": "monthly_rent",
    "is_student": False,
    "is_job_seeker": False,
    "goal": "asset_building",
    "selected_categories": ["asset_building", "housing", "policy_finance"],
    "has_income_proof": False,
    "tax_reported": False,
    "is_independent_household": True,
    "is_homeless_or_no_house": "unknown",
    "income_stability": "unstable",
}


def load_candidate_policies(user_profile: dict) -> list:
    df = pd.read_csv(BENEFIT_MASTER_PATH)

    # benefit_id 없는 첫 안내 행 제거
    df = df[df["benefit_id"].notna()].copy()

    age = user_profile.get("age")
    region = user_profile.get("region")
    selected_categories = user_profile.get("selected_categories", [])

    # 나이 조건 필터
    df = df[(df["age_min"] <= age) & (df["age_max"] >= age)]

    # 지역 조건 필터: all 또는 사용자 지역
    df = df[(df["region"] == "all") | (df["region"] == region)]

    # 관심 카테고리 우선 필터
    if selected_categories:
        filtered = df[df["category"].isin(selected_categories)]
        if len(filtered) > 0:
            df = filtered

    candidate_policies = []
    for _, row in df.iterrows():
        candidate_policies.append({
            "policy_id": row["benefit_id"],
            "policy_name": row["benefit_name"],
            "category": row["category"],
            "application_status": row["application_status"],
            "requires_income_proof": bool(row["requires_income_proof"]),
            "is_loan_type": bool(row["is_loan_type"]),
        })

    return candidate_policies


if __name__ == "__main__":
    cluster_result = predict_cluster(USER_PROFILE)
    candidate_policies = load_candidate_policies(USER_PROFILE)

    response = build_response(
        request_id=REQUEST_ID,
        user_profile=USER_PROFILE,
        candidate_policies=candidate_policies,
        cluster_result=cluster_result,
    )

    print(json.dumps(response, ensure_ascii=False, indent=2))