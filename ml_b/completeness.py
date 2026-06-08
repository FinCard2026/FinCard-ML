# completeness.py
# Section 10-2 기준: 사용자 유형별 필수 feature 응답 완성도 계산

REQUIRED_FEATURES = {
    "freelancer": ["has_income_proof", "tax_reported", "income_stability"],
    "monthly_rent": ["is_independent_household", "is_homeless_or_no_house", "rent_amount_known"],
    "job_seeker": ["active_job_search", "previous_employment_program"],
    "unemployed": ["active_job_search", "previous_employment_program"],
    "student": ["has_part_time_income", "education_support_interest", "housing_support_interest"],
    "part_time": ["has_part_time_income", "income_stability"],
}

UNKNOWN_VALUES = {None, "unknown", -1, ""}


def calc_completeness(user_profile: dict) -> float:
    """
    사용자 유형(job_type, housing_type) 기반으로
    필수 feature 중 실제 응답된 비율 반환 (0.0 ~ 1.0)

    unknown / None / -1 은 미응답으로 간주
    """
    job_type = user_profile.get("job_type", "")
    housing_type = user_profile.get("housing_type", "")

    required = set()

    if job_type in REQUIRED_FEATURES:
        required |= set(REQUIRED_FEATURES[job_type])

    if housing_type == "monthly_rent":
        required |= set(REQUIRED_FEATURES["monthly_rent"])

    if not required:
        return 1.0  # 필수 feature 없으면 완성도 최대

    answered = sum(
        1 for f in required
        if user_profile.get(f) not in UNKNOWN_VALUES
    )

    return round(answered / len(required), 2)
