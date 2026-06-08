# ml_a/encoder.py

import pandas as pd

JOB_TYPES = [
    "student",
    "freelancer",
    "job_seeker",
    "part_time",
    "contract",
    "full_time",
    "unemployed",
    "other",
]

HOUSING_TYPES = [
    "with_family",
    "monthly_rent",
    "jeonse",
    "own",
    "dormitory",
    "other",
    "unknown",
]

REGIONS = [
    "all",
    "seoul",
    "gyeonggi",
    "incheon",
    "busan",
    "other",
]

GOALS = [
    "housing",
    "employment",
    "asset_building",
    "living_support",
    "education",
    "tax",
    "policy_finance",
    "other",
]


FEATURE_COLUMNS = (
    ["age", "monthly_income"]
    + [f"job_type_{j}" for j in JOB_TYPES]
    + [f"housing_type_{h}" for h in HOUSING_TYPES]
    + [f"region_{r}" for r in REGIONS]
    + [f"goal_{g}" for g in GOALS]
    + [
        "is_student",
        "is_job_seeker",
        "has_income_proof",
        "tax_reported",
        "is_independent_household",
        "is_homeless_or_no_house",
        "income_stability_stable",
        "income_stability_unstable",
        "active_job_search",
        "has_part_time_income",
    ]
)


def _bool_to_int(value):
    if value is True:
        return 1
    if value is False:
        return 0
    if value == "unknown" or value is None or value == "":
        return 0
    return int(bool(value))


def encode_user_profile(user_profile: dict) -> pd.DataFrame:
    """
    Backend에서 전달받은 user_profile dict를 K-Means 입력용 feature vector로 변환한다.
    반환값은 sklearn pipeline에 넣을 수 있도록 DataFrame 형태로 반환한다.
    """
    row = {col: 0 for col in FEATURE_COLUMNS}

    age = user_profile.get("age", 0)
    monthly_income = user_profile.get("monthly_income", 0)

    row["age"] = float(age or 0)
    row["monthly_income"] = float(monthly_income or 0) / 10000  # 원 단위 → 만원 단위

    job_type = user_profile.get("job_type", "other")
    if job_type not in JOB_TYPES:
        job_type = "other"
    row[f"job_type_{job_type}"] = 1

    housing_type = user_profile.get("housing_type", "unknown")
    if housing_type not in HOUSING_TYPES:
        housing_type = "unknown"
    row[f"housing_type_{housing_type}"] = 1

    region = user_profile.get("region", "other")
    if region not in REGIONS:
        region = "other"
    row[f"region_{region}"] = 1

    goal = user_profile.get("goal", "other")
    if goal not in GOALS:
        goal = "other"
    row[f"goal_{goal}"] = 1

    row["is_student"] = _bool_to_int(user_profile.get("is_student", job_type == "student"))
    row["is_job_seeker"] = _bool_to_int(user_profile.get("is_job_seeker", job_type == "job_seeker"))

    row["has_income_proof"] = _bool_to_int(user_profile.get("has_income_proof", "unknown"))
    row["tax_reported"] = _bool_to_int(user_profile.get("tax_reported", "unknown"))
    row["is_independent_household"] = _bool_to_int(user_profile.get("is_independent_household", "unknown"))
    row["is_homeless_or_no_house"] = _bool_to_int(user_profile.get("is_homeless_or_no_house", "unknown"))
    row["active_job_search"] = _bool_to_int(user_profile.get("active_job_search", "unknown"))
    row["has_part_time_income"] = _bool_to_int(user_profile.get("has_part_time_income", "unknown"))

    income_stability = user_profile.get("income_stability", "unknown")
    if income_stability == "stable":
        row["income_stability_stable"] = 1
    elif income_stability == "unstable":
        row["income_stability_unstable"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def encode_many(user_profiles: list[dict]) -> pd.DataFrame:
    frames = [encode_user_profile(profile) for profile in user_profiles]
    return pd.concat(frames, ignore_index=True)