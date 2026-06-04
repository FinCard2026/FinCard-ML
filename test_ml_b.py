# test_ml_b.py
# Section 6-2 Request 예시 기준으로 전체 흐름 검증
# 명세서 Section 11-1 Response 구조와 비교

import json
from ml_b.response_builder import build_response

# Section 6-2 예시 입력
REQUEST_ID = "req_20260530_002"

USER_PROFILE = {
    "age": 26,
    "job_type": "freelancer",
    "monthly_income": 1500000,
    "region": "seoul",
    "housing_type": "monthly_rent",
    "is_student": False,
    "is_job_seeker": False,
    "goal": "asset_building",
    "selected_categories": ["asset_building", "housing"],
    "financial_literacy_level": "beginner",
    "point_balance": 120,
    "has_income_proof": True,
    "tax_reported": True,
    "is_independent_household": True,
    "is_homeless_or_no_house": "unknown",
    "income_stability": "unstable",
}

CANDIDATE_POLICIES = [
    {
        "policy_id": "p001",
        "policy_name": "청년내일저축계좌",
        "category": "asset_building",
        "application_status": "open",
        "deadline": "2026-06-30",
        "requires_income_proof": True,
        "is_loan_type": False,
    },
    {
        "policy_id": "p002",
        "policy_name": "청년월세지원",
        "category": "housing",
        "application_status": "open",
        "deadline": "2026-07-15",
        "requires_income_proof": True,
        "is_loan_type": False,
    },
]

# ML-A mock 결과 (이수빈 연동 전 고정값)
CLUSTER_RESULT = {
    "cluster_id": 2,  # 프리랜서·소득불안정형
    "similarity_score": 0.86,
}


if __name__ == "__main__":
    response = build_response(
        request_id=REQUEST_ID,
        user_profile=USER_PROFILE,
        candidate_policies=CANDIDATE_POLICIES,
        cluster_result=CLUSTER_RESULT,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
