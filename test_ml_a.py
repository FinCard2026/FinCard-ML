# test_ml_a.py

import json

from ml_a.predictor import predict_cluster


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
    "has_income_proof": True,
    "tax_reported": True,
    "is_independent_household": True,
    "is_homeless_or_no_house": "unknown",
    "income_stability": "unstable",
}


if __name__ == "__main__":
    result = predict_cluster(USER_PROFILE)
    print(json.dumps(result, ensure_ascii=False, indent=2))